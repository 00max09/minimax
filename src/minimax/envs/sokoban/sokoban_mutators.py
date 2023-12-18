"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from functools import partial
from enum import IntEnum

import numpy as np
import jax
import jax.numpy as jnp

from .sokoban import FieldStates
from minimax.envs.registration import register_mutator


class Mutations(IntEnum):
    # Turn left, turn right, move forward
    NO_OP = 0
    FLIP_WALL = 1
    MOVE_GOAL = 2

def mutate_level(self, num_edits):
        num_tiles = (self.room_dim[0]-2)*(self.room_dim[1]-2)
        edit_locs = list(set(np.random.randint(0, num_tiles, num_edits)))
        actions = np.random.randint(0, 1, len(edit_locs))

        free_mask = self.game_start_room[:][:]
        #free_mask[self.agent_start_pos[1]-1, self.agent_start_pos[0]-1] = False
        #free_mask[self.goal_pos[1]-1, self.goal_pos[0]-1] = False

        for loc, a in zip(edit_locs, actions):
          x = loc % (self.width - 2) + 1
          y = loc // (self.width - 2) + 1

          if(self.game_start_room[x][y][FieldStates.wall] == 1) :
            self.game_start_room[x][y][FieldStates.wall] = 0
            self.game_start_room[x][y][FieldStates.empty] = 1
          elif (self.game_start_room[x][y][FieldStates.empty] == 1):
            self.game_start_room[x][y][FieldStates.wall] = 1
            self.game_start_room[x][y][FieldStates.empty] = 0

        # Reset meta info

        self.step_count = 0
        self.adversary_step_count = 0
        self.reset_metrics()
        self.compute_metrics()
        self.reset_agent()
        image = self.render()
        obs = {
            'image': image,
            'time_step': [self.adversary_step_count],
            'random_z': self.generate_random_z()
        }


        return obs

def flip_wall(rng, state):
	wall_map = state.wall_map
	h,w = wall_map.shape
	wall_mask = jnp.ones((h*w,), dtype=jnp.bool_)

	goal_idx = w*state.goal_pos[1] + state.goal_pos[0]
	agent_idx = w*state.agent_pos[1] + state.agent_pos[0]
	wall_mask = wall_mask.at[goal_idx].set(False)
	wall_mask = wall_mask.at[agent_idx].set(False)

	flip_idx = jax.random.choice(rng, np.arange(h*w), p=wall_mask)
	flip_y = flip_idx//w
	flip_x = flip_idx%w

	flip_val = ~wall_map.at[flip_y,flip_x].get()
	next_wall_map = wall_map.at[flip_y,flip_x].set(flip_val)

	return state.replace(wall_map=next_wall_map)


def move_goal(rng, state):
	wall_map = state.wall_map
	h,w = wall_map.shape
	wall_mask = wall_map.flatten()

	goal_idx = w*state.goal_pos[1] + state.goal_pos[0]
	agent_idx = w*state.agent_pos[1] + state.agent_pos[0]
	wall_mask = wall_mask.at[goal_idx].set(True)
	wall_mask = wall_mask.at[agent_idx].set(True)

	next_goal_idx = jax.random.choice(rng, np.arange(h*w), p=~wall_mask)
	next_goal_y = next_goal_idx//w
	next_goal_x = next_goal_idx%w

	next_wall_map = wall_map.at[next_goal_y,next_goal_x].set(False)
	next_goal_pos = jnp.array([next_goal_x,next_goal_y], dtype=jnp.uint32)

	return state.replace(wall_map=next_wall_map, goal_pos=next_goal_pos)


@partial(jax.jit, static_argnums=(1,3))
def move_goal_flip_walls(rng, params, state, n=1):
	if n == 0:
		return state

	def _mutate(carry, step):
		state = carry
		rng, mutation = step

		rng, arng, brng = jax.random.split(rng,3)

		is_flip_wall = jnp.equal(mutation, Mutations.FLIP_WALL.value)
		mutated_state = flip_wall(arng, state)
		next_state = jax.tree_map(lambda x,y: jax.lax.select(is_flip_wall, x, y), mutated_state, state)

		is_move_goal = jnp.equal(mutation, Mutations.MOVE_GOAL.value)
		mutated_state = move_goal(brng, state)
		next_state = jax.tree_map(lambda x,y: jax.lax.select(is_move_goal, x, y), mutated_state, next_state)

		return next_state, None

	rng, nrng, *mrngs = jax.random.split(rng, n+2)
	mutations = jax.random.choice(nrng, np.arange(len(Mutations)), (n,))

	state, _ = jax.lax.scan(_mutate, state, (jnp.array(mrngs), mutations))

	# Update state maze_map
	next_maze_map = make_maze_map(
		params,
		state.wall_map, 
		state.goal_pos, 
		state.agent_pos, 
		state.agent_dir_idx, 
		pad_obs=True)

	return state.replace(maze_map=next_maze_map)

# Register the mutators
if hasattr(__loader__, 'name'):
  module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
  module_path = __loader__.fullname

register_mutator(env_id='Sokoban', mutator_id=None, entry_point=module_path + ':basic_soko_mut')