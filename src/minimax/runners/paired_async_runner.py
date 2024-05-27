"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from enum import Enum
from functools import partial
from typing import Tuple, Optional
import inspect

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import optax
import flax
import flax.linen as nn
from flax.core.frozen_dict import FrozenDict

import minimax.envs as envs
from minimax.util import pytree as _tree_util
from minimax.util.rl import (
	AgentPop,
	VmapTrainState,
	RolloutStorage,
	RollingStats,
	RolloutBatch,
	UEDScore,
	compute_ued_scores
)
from collections import OrderedDict

def merge_dict(x, y):
	for a in x.keys() :
		x[a] = jnp.concatenate([x[a], y[a]])

def merge_batches(x, y):
	batch_kwargs = dict(
			obs=merge_dict(x[0], y[0]),
			actions=jnp.concatenate([x[1], y[1]]),
			rewards=jnp.concatenate([x[2], y[2]]),
			dones=jnp.concatenate([x[3], y[3]]),
			log_pis=jnp.concatenate([x[4], y[4]]),
			values=jnp.concatenate([x[5], y[5]]),
			targets=jnp.concatenate([x[6], y[6]]),
			advantages=jnp.concatenate([x[7], y[7]]),
			carry=(jnp.concatenate([x[8][0], y[8][0]]),jnp.concatenate([x[8][1], y[8][1]])),
		)
	return RolloutBatch(**batch_kwargs)
		
class PAIREDASYNCRunner:
	"""
	Orchestrates rollouts across one or more students and teachers. 
	The main components at play:
	- AgentPop: Manages train state and batched inference logic 
		for a population of agents.
	- BatchUEDEnv: Manages environment step and reset logic for a 
		population of agents batched over a pair of student and 
		teacher MDPs.
	- RolloutStorage: Manages the storing and sampling of collected txns.
	- PPO: Handles PPO updates, which take a train state + batch of txns.
	"""
	def __init__(
		self, 
		env_name,
		env_kwargs,
		ued_env_kwargs,
		student_agents,
		n_students=2,
		n_parallel=1,
		n_eval=1,
		n_rollout_steps=250,
		lr=1e-4,
		lr_final=None,
		lr_anneal_steps=0,
		max_grad_norm=0.5,
		discount=0.99,
		gae_lambda=0.95,
		adam_eps=1e-5,
		teacher_lr=None,
		teacher_lr_final=None,
		teacher_lr_anneal_steps=None,
		teacher_discount=0.99,
		teacher_gae_lambda=0.95,
		teacher_agents=None,
		ued_score='relative_regret',
		track_env_metrics=False,
		n_unroll_rollout=1,
		render=False,
		n_devices=1):
		assert n_parallel % n_devices == 0, 'Num envs must be divisible by num devices.'

		ued_score = UEDScore[ued_score.upper()]

		assert len(student_agents) == 1, \
			'Only one type of student supported.'
		assert not (n_students > 2 and ued_score in [UEDScore.RELATIVE_REGRET, UEDScore.MEAN_RELATIVE_REGRET]), \
			'Standard PAIRED uses only 2 students.'
		assert teacher_agents is None or len(teacher_agents) == 1, \
			'Only one type of teacher supported.'

		self.n_students = n_students
		self.n_parallel = n_parallel // n_devices
		self.n_eval = n_eval
		self.n_devices = n_devices
		self.step_batch_size = n_students*n_eval*n_parallel
		self.n_rollout_steps = n_rollout_steps
		self.n_updates = 0
		self.lr = lr
		self.lr_final = lr if lr_final is None else lr_final
		self.lr_anneal_steps = lr_anneal_steps
		self.teacher_lr = \
			lr if teacher_lr is None else lr
		self.teacher_lr_final = \
			self.lr_final if teacher_lr_final is None else teacher_lr_final
		self.teacher_lr_anneal_steps = \
			lr_anneal_steps if teacher_lr_anneal_steps is None else teacher_lr_anneal_steps
		self.max_grad_norm = max_grad_norm
		self.adam_eps = adam_eps
		self.ued_score = ued_score
		self.track_env_metrics = track_env_metrics

		self.n_unroll_rollout = n_unroll_rollout
		self.render = render


		self.bob_pop = AgentPop(student_agents[0], n_agents=1) # Maybe to change
		self.alice_pop = AgentPop(student_agents[0], n_agents=1) # Maybe to change

		if teacher_agents is not None:
			self.teacher_pop = AgentPop(teacher_agents[0], n_agents=1)

		# This ensures correct partial-episodic bootstrapping by avoiding
		# any termination purely due to timeouts.
		# env_kwargs.max_episode_steps = self.n_rollout_steps + 1
		self.benv = envs.BatchUEDEnv(
			env_name=env_name,
			n_parallel=self.n_parallel,
			n_eval=n_eval,
			env_kwargs=env_kwargs,
			ued_env_kwargs=ued_env_kwargs,
			wrappers=['monitor_return', 'monitor_ep_metrics'],
			ued_wrappers=[]
		)
		self.teacher_n_rollout_steps = \
			self.benv.env.ued_max_episode_steps()

		self.alice_rollout = RolloutStorage(
			discount=discount,
			gae_lambda=gae_lambda,
			n_steps=n_rollout_steps,
			n_agents=1, # may change
			n_envs=self.n_parallel,
			n_eval=self.n_eval,
			action_space=self.benv.env.action_space(),
			obs_space=self.benv.env.observation_space(),
			agent=self.alice_pop.agent
		)

		self.bob_rollout = RolloutStorage(
			discount=discount,
			gae_lambda=gae_lambda,
			n_steps=n_rollout_steps,
			n_agents=1, # may change
			n_envs=self.n_parallel,
			n_eval=self.n_eval,
			action_space=self.benv.env.action_space(),
			obs_space=self.benv.env.observation_space(),
			agent=self.bob_pop.agent
		)

		self.teacher_rollout = RolloutStorage(
			discount=teacher_discount,
			gae_lambda=teacher_gae_lambda,
			n_steps=self.teacher_n_rollout_steps,
			n_agents=1,
			n_envs=self.n_parallel,
			n_eval=1,
			action_space=self.benv.env.ued_action_space(),
			obs_space=self.benv.env.ued_observation_space(),
			agent=self.teacher_pop.agent,
		)

		ued_monitored_metrics = ('return',)
		self.ued_rolling_stats = RollingStats(	
			names=ued_monitored_metrics,
			window=10,
		)
		
		monitored_metrics = self.benv.env.get_monitored_metrics()
		self.rolling_stats = RollingStats(
			names=monitored_metrics,
			window=10,
		)

		self._update_ep_stats = jax.vmap(jax.vmap(self.rolling_stats.update_stats))
		self._update_ued_ep_stats = jax.vmap(jax.vmap(self.ued_rolling_stats.update_stats))

		if self.render:
			from envs.viz.grid_viz import GridVisualizer
			self.viz = GridVisualizer()
			self.viz.show()

	def reset(self, rng):
		self.n_updates = 0

		n_parallel = self.n_parallel*self.n_devices

		rng, alice_rng, bob_rng, teacher_rng = jax.random.split(rng,4)
		alice_info = self._reset_pop(
				alice_rng, 
				self.alice_pop, 
				partial(self.benv.reset, sub_batch_size=n_parallel*self.n_eval),
				n_parallel_ep=n_parallel*self.n_eval,
				lr_init=self.lr,
				lr_final=self.lr_final,
				lr_anneal_steps=self.lr_anneal_steps)

		bob_info = self._reset_pop(
				bob_rng, 
				self.bob_pop, 
				partial(self.benv.reset, sub_batch_size=n_parallel*self.n_eval),
				n_parallel_ep=n_parallel*self.n_eval,
				lr_init=self.lr,
				lr_final=self.lr_final,
				lr_anneal_steps=self.lr_anneal_steps)

		teacher_info = self._reset_pop(
			teacher_rng, 
			self.teacher_pop, 
			partial(self.benv.reset_teacher, n_parallel=n_parallel),
			n_parallel_ep=n_parallel,
			lr_init=self.teacher_lr,
			lr_final=self.teacher_lr_final,
			lr_anneal_steps=self.teacher_lr_anneal_steps)

		return (
			rng,
			*alice_info,
			*bob_info,
			*teacher_info
		)

	def _reset_pop(
		self, 
		rng, 
		pop, 
		env_reset_fn, 
		n_parallel_ep=1,
		lr_init=3e-4,
		lr_final=3e-4,
		lr_anneal_steps=0):
		rng, *vrngs = jax.random.split(rng, pop.n_agents+1)
		reset_out = env_reset_fn(jnp.array(vrngs))
		if len(reset_out) == 2:
			obs, state = reset_out
		else:
			obs, state, extra = reset_out
		dummy_obs = jax.tree_util.tree_map(lambda x: x[0], obs) # for one agent only

		rng, subrng = jax.random.split(rng)
		if pop.agent.is_recurrent:
			carry = pop.init_carry(subrng, obs)
		else:
			carry = None

		rng, subrng = jax.random.split(rng)
		params = pop.init_params(subrng, dummy_obs)

		schedule_fn = optax.linear_schedule(
			init_value=-float(lr_init),
			end_value=-float(lr_final),
			transition_steps=lr_anneal_steps,
		)

		tx = optax.chain(
			optax.clip_by_global_norm(self.max_grad_norm),
			optax.scale_by_adam(eps=self.adam_eps),
			optax.scale_by_schedule(schedule_fn),
		)

		train_state = VmapTrainState.create(
			apply_fn=pop.agent.evaluate,
			params=params,
			tx=tx
		)
		
		ep_stats = self.rolling_stats.reset_stats(
			batch_shape=(pop.n_agents,n_parallel_ep))

		return train_state, state, obs, carry, ep_stats

	def get_checkpoint_state(self, state):
		_state = list(state)
		_state[1] = state[1].state_dict
		_state[6] = state[6].state_dict
		_state[11] = state[11].state_dict
		return _state

	def load_checkpoint_state(self, runner_state, state):
		runner_state = list(runner_state)
		runner_state[1] = runner_state[1].load_state_dict(state[1])
		runner_state[6] = runner_state[6].load_state_dict(state[6])
		runner_state[11] = runnner_state[11].load_state_dice(state[11])
		return tuple(runner_state)

	@partial(jax.jit, static_argnums=(0,2,3))
	def _get_transition(
		self,
		rng, 
		pop, 
		rollout_mgr, 
		rollout, 
		params, 
		state, 
		obs, 
		carry, 
		done,
		reset_state=None,
		extra=None):
		# Sample action
		value, pi_params, next_carry = pop.act(params, obs, carry, done)
		pi = pop.get_action_dist(pi_params)
		rng, subrng = jax.random.split(rng)
		action = pi.sample(seed=subrng)
		log_pi = pi.log_prob(action)

		rng, *vrngs = jax.random.split(rng, pop.n_agents+1)

		if pop is self.alice_pop or pop is self.bob_pop:
			step_fn = self.benv.step_student
		else:
			step_fn = self.benv.step_alice_teacher
		step_args = (jnp.array(vrngs), state, action)

		if reset_state is not None: # Needed for student to reset to same instance
			step_args += (reset_state,)

		if extra is not None:
			step_args += (extra,)
			next_obs, next_state, reward, done, info, extra = step_fn(*step_args)
		else:
			next_obs, next_state, reward, done, info = step_fn(*step_args)

		# Add transition to storage
		step = (obs, action, reward, done, log_pi, value)
		if carry is not None:
			step += (carry,)

		rollout = rollout_mgr.append(rollout, *step)

		if self.render and pop is self.bob_pop:
			self.viz.render(
				self.benv.env.env.params, 
				jax.tree_util.tree_map(lambda x: x[0][0], state))

		return rollout, next_state, next_obs, next_carry, done, info, extra

	@partial(jax.jit, static_argnums=(0,2,3,4))
	def _rollout(
		self, 
		rng, 
		pop, 
		rollout_mgr,
		n_steps,
		params, 
		state, 
		obs, 
		carry, 
		done,
		reset_state=None, 
		extra=None, 
		ep_stats=None):
		rngs = jax.random.split(rng, n_steps)

		rollout = rollout_mgr.reset()

		def _scan_rollout(scan_carry, rng):
			(rollout, 
			 state, 
			 obs, 
			 carry,
			 done, 
			 extra, 
			 ep_stats) = scan_carry
			
			next_scan_carry = \
				self._get_transition(
					rng,
					pop, 
					rollout_mgr,
					rollout,
					params, 
					state, 
					obs, 
					carry, 
					done,
					reset_state, 
					extra)

			(rollout, 
			 next_state, 
			 next_obs, 
			 next_carry, 
			 done, 
			 info, 
			 extra) = next_scan_carry

			if ep_stats is not None:
				_ep_stats_update_fn = self._update_ep_stats \
					if pop is self.bob_pop or pop is self.alice_pop else self._update_ued_ep_stats

				ep_stats = _ep_stats_update_fn(ep_stats, done, info)

			return (rollout, next_state, next_obs, next_carry, done, extra, ep_stats), None

		(rollout, state, obs, carry, done, extra, ep_stats), _ = jax.lax.scan(
			_scan_rollout,
			(rollout, state, obs, carry, done, extra, ep_stats),
			rngs,
			length=n_steps,
			unroll=self.n_unroll_rollout
		)

		return rollout, state, obs, carry, extra, ep_stats

	@partial(jax.jit, static_argnums=(0,))
	def _compile_stats(self, 
		bob_update_stats, bob_ep_stats,
		alice_update_stats, alice_ep_stats, 
		ued_update_stats, ued_ep_stats,
		env_metrics=None,
		grad_stats=None, ued_grad_stats=None):
		mean_returns_by_bob = jax.vmap(lambda x: x.mean())(bob_ep_stats['return'])
		mean_returns_by_alice = jax.vmap(lambda x: x.mean())(alice_ep_stats['return'])
		mean_returns_by_teacher = jax.vmap(lambda x: x.mean())(ued_ep_stats['return'])

		mean_bob_ep_stats = jax.vmap(lambda info: jax.tree_map(lambda x: x.mean(), info))(
			{k:bob_ep_stats[k] for k in self.rolling_stats.names}
		)
		mean_alice_ep_stats = jax.vmap(lambda info: jax.tree_map(lambda x: x.mean(), info))(
			{k:alice_ep_stats[k] for k in self.rolling_stats.names}
		)
		ued_mean_ep_stats = jax.vmap(lambda info: jax.tree_map(lambda x: x.mean(), info))(
			{k:ued_ep_stats[k] for k in self.ued_rolling_stats.names}
		)

		bob_stats = {
			f'mean_{k}':v for k,v in mean_bob_ep_stats.items()
		}
		bob_stats.update(bob_update_stats)

		
		alice_stats = {
			f'mean_{k}':v for k,v in mean_bob_ep_stats.items()
		}
		alice_stats.update(alice_update_stats)

		stats = {}
		for i in range(1): #to change
			_bob_stats = jax.tree_util.tree_map(lambda x: x[i], bob_stats) # for agent0
			stats.update({f'{k}_bob{i}':v for k,v in _bob_stats.items()})

		for i in range(1): # to change
			_alice_stats = jax.tree_util.tree_map(lambda x: x[i], alice_stats) # for agent0
			stats.update({f'{k}_alice{i}':v for k,v in _alice_stats.items()})
		
		teacher_stats = {
			f'mean_{k}_tch':v for k,v in ued_mean_ep_stats.items()
		}
		teacher_stats.update({
			f'{k}_tch':v[0] for k,v in ued_update_stats.items()
		})
		stats.update(teacher_stats)

		if self.track_env_metrics:
			passable_mask = env_metrics.pop('passable')
			mean_env_metrics = jax.tree_util.tree_map(
				lambda x: jnp.where(passable_mask.sum() > 0, (x*passable_mask).sum()/passable_mask.sum(), 0), 
				env_metrics
			)
			mean_env_metrics.update({'passable_ratio': passable_mask.mean()})
			stats.update({
				f'env/{k}':v for k,v in mean_env_metrics.items()
			})

		if self.n_devices > 1:
			stats = jax.tree_map(lambda x: jax.lax.pmean(x, 'device'), stats)

		return stats

	def get_shmap_spec(self):
		runner_state_size = len(inspect.signature(self.run).parameters)
		in_spec = [P(None,'device'),]*(runner_state_size)
		out_spec = [P(None,'device'),]*(runner_state_size)

		in_spec[:2] = [P(None),]*2
		in_spec[6] = P(None)
		in_spec = tuple(in_spec)
		out_spec = (P(None),) + in_spec

		return in_spec, out_spec

	@partial(jax.jit, static_argnums=(0,))
	def run(
		self, 
		rng, 
		bob_train_state, 
		bob_state,
		bob_obs,
		bob_carry,
		bob_ep_stats,
		alice_train_state,
		alice_state,
		alice_obs,
		alice_carry,
		alice_ep_stats,
		ued_train_state,
		ued_state,
		ued_obs,
		ued_carry,
		ued_ep_stats,
		):
		"""
		Perform one update step: rollout teacher + students
		"""
		if self.n_devices > 1:
			rng = jax.random.fold_in(rng, jax.lax.axis_index('device'))

		# === Reset teacher env + rollout teacher
		rng, *vrngs = jax.random.split(rng, self.teacher_pop.n_agents+1)
		ued_reset_out = self.benv.reset_teacher(jnp.array(vrngs))
		if len(ued_reset_out) > 2:
			ued_obs, ued_state, ued_extra = ued_reset_out
		else:
			ued_obs, ued_state = ued_reset_out
			ued_extra = None

		# Reset UED ep_stats
		if self.ued_rolling_stats is not None:
			ued_ep_stats = self.ued_rolling_stats.reset_stats(
				batch_shape=(1,self.n_parallel))
		else:
			ued_ep_stats = None

		tch_rollout_batch_shape = (1,self.n_parallel*self.n_eval)
		done = jnp.zeros(tch_rollout_batch_shape, dtype=jnp.bool_)
		rng, subrng = jax.random.split(rng)
		ued_rollout, ued_state, ued_obs, ued_carry, _, ued_ep_stats = \
			self._rollout(
				subrng,
				self.teacher_pop,
				self.teacher_rollout,
				self.teacher_n_rollout_steps,
				jax.lax.stop_gradient(ued_train_state.params), 
				ued_state, 
				ued_obs, 
				ued_carry,
				done, 
				extra=ued_extra, 
				ep_stats=ued_ep_stats
			)

		# === Reset student to new envs + rollout students
		rng, *vrngs = jax.random.split(rng, self.teacher_pop.n_agents+1)
		alice_obs, alice_start_state, alice_extra = jax.tree_util.tree_map(
			lambda x:x.squeeze(0), self.benv.reset_student(
				jnp.array(vrngs),
				ued_state, 
				self.alice_pop.n_agents)) 
				# jeśli dobrze rozumiem z jakiegoś powodu to jest nie vmapowane
		alice_reset_state = alice_state
		# Reset student ep_stats
		st_rollout_batch_shape = (1,self.n_parallel*self.n_eval)
		ep_stats = self.rolling_stats.reset_stats(
			batch_shape=st_rollout_batch_shape)

		# Rollout Alice
		done = jnp.zeros(st_rollout_batch_shape, dtype=jnp.bool_)
		rng, subrng = jax.random.split(rng)
		alice_rollout, alice_state, alice_obs, alice_carry, _, alice_ep_stats = \
			self._rollout(
				subrng, 
				self.alice_pop,
				self.alice_rollout,
				self.n_rollout_steps,
				jax.lax.stop_gradient(alice_train_state.params),
				alice_start_state, 
				alice_obs, 
				alice_carry, 
				done,
				reset_state=alice_reset_state, 
				extra=alice_extra, 
				ep_stats=alice_ep_stats)

		# Add rewards
		rng, *vrngs = jax.random.split(rng, self.teacher_pop.n_agents+1)
		bob_obs, bob_start_state, bob_extra = jax.tree_util.tree_map(
			lambda x:x.squeeze(0), self.benv.reset_student(
				jnp.array(vrngs),
				ued_state, 
				self.bob_pop.n_agents)) 

		rng, *vrngs = jax.random.split(rng, self.teacher_pop.n_agents+1)
		_, bob_reset_state = self.benv.add_reward_structure_for_bob(
				jnp.array(vrngs),
				alice_reset_state, 
				alice_state) #???? nie wiem
				
		# Bob rollout
		rng, subrng = jax.random.split(rng)
		bob_rollout, bob_state, bob_obs, bob_carry, bob_extra, bob_ep_stats = \
			self._rollout(
				subrng, 
				self.bob_pop,
				self.bob_rollout,
				self.n_rollout_steps,
				jax.lax.stop_gradient(bob_train_state.params),
				bob_start_state, 
				bob_obs, 
				bob_carry, 
				done,
				reset_state=bob_reset_state, 
				extra=bob_extra, 
				ep_stats=bob_ep_stats)

		# === Update Bob with PPO
		# PPOAgent vmaps over the train state and batch. Batch must be N x EM
		bob_rollout_last_value = self.bob_pop.get_value(
			jax.lax.stop_gradient(bob_train_state.params), bob_obs, bob_carry
		)
		bob_train_batch = self.bob_rollout.get_batch(
			bob_rollout, 
			bob_rollout_last_value
		)
		rng, subrng = jax.random.split(rng)
		bob_train_state, bob_update_stats = self.bob_pop.update(subrng, bob_train_state, bob_train_batch)
		

		# === Update Alice with PPO


		alice_score, _ = compute_ued_scores(UEDScore.REVERSED_POSITIVE_RETURN, bob_train_batch, self.n_eval)
		alice_rollout = self.alice_rollout.set_final_reward(alice_rollout, alice_score)
		alice_train_batch = self.alice_rollout.get_batch(
			alice_rollout, 
			jnp.zeros((1, self.n_parallel)) # Last step terminates episode
		)

		# ued_ep_stats = self._update_ued_ep_stats(
		# 	ued_ep_stats, 
		# 	jnp.ones((1,len(ued_score),1), dtype=jnp.bool_),
		# 	{'return': jnp.expand_dims(ued_score, (0,-1))}
		# )

		rng, subrng = jax.random.split(rng)
		alice_train_state, alice_update_stats = self.alice_pop.update(subrng, alice_train_state, alice_train_batch)

		# === Update teacher with PPO
		# - Compute returns per env per agent
		# - Compute batched returns based on returns per env per agent
		alice_bob_train_batch = merge_batches(bob_train_batch, alice_train_batch)
		ued_score, _ = compute_ued_scores(self.ued_score, alice_bob_train_batch, self.n_eval)
		ued_rollout = self.teacher_rollout.set_final_reward(ued_rollout, ued_score)
		ued_train_batch = self.teacher_rollout.get_batch(
			ued_rollout, 
			jnp.zeros((1, self.n_parallel)) # Last step terminates episode
		)

		ued_ep_stats = self._update_ued_ep_stats(
			ued_ep_stats, 
			jnp.ones((1,len(ued_score),1), dtype=jnp.bool_),
			{'return': jnp.expand_dims(ued_score, (0,-1))}
		)
		# Update teacher, batch must be 1 x Ex1
		rng, subrng = jax.random.split(rng)
		ued_train_state, ued_update_stats = self.teacher_pop.update(subrng, ued_train_state, ued_train_batch)

		# --------------------------------------------------
		# Collect metrics
		if self.track_env_metrics:
			env_metrics = self.benv.get_env_metrics(bob_reset_state)
		else:
			env_metrics = None

		grad_stats, ued_grad_stats = None, None

		stats = self._compile_stats(
			bob_update_stats, bob_ep_stats, 
			alice_update_stats, alice_ep_stats,
			ued_update_stats, ued_ep_stats,
			env_metrics,
			grad_stats, ued_grad_stats)
		

		stats.update(dict(n_updates=bob_train_state.n_updates[0]))
		
		bob_train_state = bob_train_state.increment()
		ued_train_state = ued_train_state.increment()
		alice_train_state = alice_train_state.increment()
		
		
		self.n_updates += 1
		
		return (
			stats, 
			rng,
			bob_train_state, bob_state, bob_obs, bob_carry, bob_ep_stats,
			alice_train_state, alice_state, alice_obs, alice_carry, alice_ep_stats,
			ued_train_state, ued_state, ued_obs, ued_carry, ued_ep_stats
		)
