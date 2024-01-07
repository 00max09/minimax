from dataclasses import dataclass
from collections import namedtuple, OrderedDict
from functools import partial
from enum import IntEnum
import sys
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from typing import Tuple, Optional
import chex
from flax import struct
from flax.core.frozen_dict import FrozenDict

from minimax.envs import environment, spaces
from minimax.envs.registration import register_ued
from .common import EnvInstance

class SequentialActions(IntEnum):
    skip = 0
    wall = 1
    goal = 2
    agent = 3


@struct.dataclass
class EnvState:
    maze_map: chex.Array
    time: int
    terminal: bool
    agent_pos: chex.Array
    unmatched_boxes: int

class FieldStates(IntEnum):
    wall = 0
    empty = 1
    target = 2
    box_target = 3
    box = 4
    player = 5
    player_target = 6

@struct.dataclass
class EnvParams:
    height: int = 15
    width: int = 15
    n_walls: int = 25
    noise_dim: int = 50
    replace_wall_pos: bool = False
    fixed_n_wall_steps: bool = False
    first_wall_pos_sets_budget: bool = False
    use_seq_actions: bool = (False,)
    set_agent_dir: bool = False
    normalize_obs: bool = False
    singleton_seed: int = -1


class UEDSokoban(environment.Environment):
    def __init__(
        self,
        height=13,
        width=13,
        n_walls=25,
        noise_dim=16,
        replace_wall_pos=False,
        fixed_n_wall_steps=False,
        first_wall_pos_sets_budget=False,
        use_seq_actions=False,
        set_agent_dir=False,
        normalize_obs=False,
    ):
        """
        Using the original action space requires ensuring proper handling
        of a sequence with trailing dones, e.g. dones: 0 0 0 0 1 1 1 1 1 ... 1.
        Advantages and value losses should only be computed where ~dones[0].
        """
        assert not (
            first_wall_pos_sets_budget and fixed_n_wall_steps
        ), "Setting first_wall_pos_sets_budget=True requires fixed_n_wall_steps=False."

        super().__init__()

        self.n_tiles = height * width
        self.action_set = jnp.array(
            jnp.arange(self.n_tiles)
        )  # go straight, turn left, turn right, take action

        self.params = EnvParams(
            height=height,
            width=width,
            n_walls=n_walls,
            noise_dim=noise_dim,
            replace_wall_pos=replace_wall_pos,
            fixed_n_wall_steps=fixed_n_wall_steps,
            first_wall_pos_sets_budget=first_wall_pos_sets_budget,
            use_seq_actions=False,
            set_agent_dir=set_agent_dir,
            normalize_obs=normalize_obs,
        )
    
    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: int,
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        params = self.params
        if action >= self.n_tiles:
            raise ValueError("Position passed to step_adversary is outside the grid.")

        # Resample block count if necessary, based on first loc
        #if self.resample_n_clutter and not self.n_clutter_sampled:
        #    n_clutter = int((action / self.adversary_action_dim) * self.n_clutter)
        #    self.adversary_max_steps = n_clutter + 2
        #    self.n_clutter_sampled = True

        if state.time < params.n_walls:
            # Add offset of 1 for outside walls
            x = int(action % (params.width - 2)) + 1
            y = int(action / (params.width - 2)) + 1
            done = False

            # if self.choose_goal_last:
            #  should_choose_goal = self.adversary_step_count == self.adversary_max_steps - 2
            #  should_choose_agent = self.adversary_step_count == self.adversary_max_steps - 1
            # else:
            should_choose_goal = state.time == params.n_walls-3
            should_choose_agent = state.time == params.n_walls-2
            should_choose_box = state.time == params.n_walls-1
            # print(f"{self.adversary_step_count}/{self.adversary_max_steps}", flush=True)
            # print(f"goal/agent = {should_choose_goal}/{should_choose_agent}", flush=True)

            # Place goal
            if should_choose_goal:
                # If there is goal noise, sometimes randomly place the goal
                state.maze_map[x][y] = jnp.zeros(7)  # Remove any walls that might be in this loc
                state.maze_map[x][y][FieldStates.target] = 1

            # Place the agent
            elif should_choose_agent:
                state.maze_map[x][y] = jnp.zeros(7)
                state.maze_map[x][y][FieldStates.player] = 1

            elif should_choose_box:
                state.maze_map[x][y] = jnp.zeros(7)
                state.maze_map[x][y][FieldStates.box] = 1
                state.unmatched_boxes+=1

            # Place wall
            elif state.time < params.n_walls:
                # If there is already an object there, action does nothing
                if state.maze_map[x][y][FieldStates.empty] == 1:
                    state.maze_map[x][y][FieldStates.empty] = 0
                    state.maze_map[x][y][FieldStates.wall] = 0
                    #self.n_clutter_placed += 1

        state.time += 1 #self.adversary_step_count += 1
        print(state.maze_map, file=sys.stderr)
        # End of episode
        if state.time >= params.n_walls + 3:
            done = True
            self.reset_metrics()
            self.compute_metrics()
        else:
            done = False
        obs = self.get_obs(state)
        return (
            lax.stop_gradient(obs),
            lax.stop_gradient(state),
            0,
            done,
            {}
        )

    @staticmethod
    def align_kwargs(kwargs, other_kwargs):
        kwargs.update(dict(
            height=other_kwargs['height'],
            width=other_kwargs['width'],
        ))

        return kwargs
    
    def get_env_instance(
            self, 
            key: chex.PRNGKey,
            state: EnvState
        ) -> chex.Array:
        """
        Converts internal encoding to an instance encoding that 
        can be interpreted by the `set_to_instance` method 
        the paired Environment class.
        """

        # Make wall map
        maze_map = state.maze_map
        agent_pos = state.agent_pos
        unmatched_boxes = state.unmatched_boxes

        return EnvInstance(
            agent_pos=agent_pos,
            maze_map=maze_map,
            unmatched_boxes=unmatched_boxes
        )
    
    def is_terminal(self, state: EnvState) -> bool:	
        done_steps = state.time >= self.max_episode_steps()
        return jnp.logical_or(done_steps, state.terminal)
    
    def _get_post_terminal_obs(self, state: EnvState):
        dtype = jnp.float32 if self.params.normalize_obs else jnp.uint8
        image = jnp.zeros((
            self.params.height+2, self.params.width+2, 3), dtype=dtype
        )

        return OrderedDict(dict(
            image=image,
            time=state.time,
            noise=jnp.zeros(self.params.noise_dim, dtype=jnp.float32),
        ))

    def get_obs(self, state: EnvState):
        
        image = state.maze_map
        return OrderedDict(dict(
            image = image,
            time=state.time,
        ))

    @property
    def default_params(self):
        return EnvParams()
    
    @property
    def name(self) -> str:
        """Environment name."""
        return "UEDSokoban"
    
    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return len(self.action_set)
    
    def action_space(self) -> spaces.Discrete:
        """Action space of the environment."""
        params = self.params
        return spaces.Discrete(
            params.height*params.width,
            dtype=jnp.uint32
        )

    def observation_space(self) -> spaces.Dict:
        """Observation space of the environment."""
        params = self.params
        max_episode_steps = self.max_episode_steps()
        spaces_dict = {
            'image':spaces.Box(0, 255, (params.height+2, params.width+2, 3)),
            'time': spaces.Discrete(max_episode_steps),
        }
        
        return spaces.Dict(spaces_dict)
    
    def max_episode_steps(self) -> int:    
        max_episode_steps = self.params.n_walls + 2 + self.params.num_boxes * 2 
        return max_episode_steps
    
if hasattr(__loader__, 'name'):
  module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
  module_path = __loader__.fullname

register_ued(env_id='Sokoban', entry_point=module_path + ':UEDSokoban')    