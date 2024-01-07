from typing import Tuple, Optional

import jax
import jax.numpy as jnp
from flax import struct
import chex
from jax import lax

from minimax.envs.registration import register
from .sokoban import (
    Sokoban,
    EnvParams,
    EnvState,
    Actions,
    FieldStates
)

class SokobanSingleton(Sokoban):
    def __init__(
        self,
        maze_map,
        agent_start_pos,
        unmatched_boxes,
        dim_room=(10, 10),
        max_steps=jnp.inf,
        num_boxes=4,
        num_gen_steps=None,
        fast_state_eq=False,
        penalty_for_step=-0.1,
        # currently penalty_box_off_target is implicitly = - reward_box_on_target
        # penalty_box_off_target=-1,
        reward_box_on_target=1,
        reward_finished=10,
        seed=None,
        load_boards_from_file=None,
        load_boards_lazy=True,
        
    ):	
        super().__init__(
            height=dim_room[0],
            width=dim_room[1],
            num_boxes=num_boxes,
            num_gen_steps=num_gen_steps,
            fast_state_eq=fast_state_eq,
            penalty_for_step=penalty_for_step,
            reward_box_on_target=reward_box_on_target,
            reward_finished=reward_finished,
            seed = seed,
        )

        self.maze_map = maze_map
        self.agent_start_pos = agent_start_pos
        self.unmatched_boxes = unmatched_boxes


    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters
        return EnvParams()

    def reset_env(
        self, 
        key: chex.PRNGKey, 
    ) -> Tuple[chex.Array, EnvState]:

        agent_pos = self.agent_start_pos
        maze_map = self.maze_map
        unmatched_boxes = self.unmatched_boxes
        state = EnvState(
            agent_pos=agent_pos,
            maze_map=maze_map,
            start_map=maze_map,
            time=0,
            terminal=False,
            unmatched_boxes=unmatched_boxes
        )

        return self.get_obs(state), state
        

class MicroRoom(SokobanSingleton):
    def __init__(
        self, 
        normalize_obs=False):
        maze_map = self._gen_grid()
        unmatched_boxes = 1 
        agent_pos = (0,0)

        super().__init__(
            agent_start_pos = agent_pos,
            maze_map=maze_map,
            unmatched_boxes=unmatched_boxes,
            dim_room=(3,3),
            num_boxes=1
        )

    def _gen_grid(self):
        # Create the grid
        room = jnp.zeros(shape = (3, 3, 7))
        room = room.at[:,:,FieldStates.empty].set(1)
        room = room.at[0,0,FieldStates.empty].set(0)
        room = room.at[1,1,FieldStates.empty].set(0)
        room = room.at[2,2,FieldStates.empty].set(0)
        room = room.at[0,0,FieldStates.player].set(1)
        room = room.at[1,1,FieldStates.box].set(1)
        room = room.at[2,2,FieldStates.target].set(1)
        return room


class Mini2Room(SokobanSingleton):
    def __init__(
        self, 
        normalize_obs=False):
        maze_map = self._gen_grid()
        unmatched_boxes = 1 
        agent_pos = (0,0)

        super().__init__(
            agent_start_pos = agent_pos,
            maze_map=maze_map,
            unmatched_boxes=unmatched_boxes,
            dim_room=(5,5),
            num_boxes=1
        )

    def _gen_grid(self):
        # Create the grid
        room = jnp.array([
            [5, 1, 0, 1, 1],
            [1, 1, 0, 1, 1],
            [1, 1, 4, 1, 1],
            [1, 1, 0, 1, 1],
            [1, 1, 0, 1, 2],
            ])
        room = jnp.squeeze(jnp.eye(7, dtype=jnp.uint8)[room.reshape(-1)]).reshape(
            room.shape + (7,)
        )
        return room

class MiniRoom(SokobanSingleton):
    def __init__(
        self, 
        normalize_obs=False):
        maze_map = self._gen_grid()
        unmatched_boxes = 1 
        agent_pos = (0,0)

        super().__init__(
            agent_start_pos = agent_pos,
            maze_map=maze_map,
            unmatched_boxes=unmatched_boxes,
            dim_room=(5,5),
            num_boxes=1
        )

    def _gen_grid(self):
        # Create the grid
        room = jnp.array([
            [5, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 4, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 2],
            ])
        room = jnp.squeeze(jnp.eye(7, dtype=jnp.uint8)[room.reshape(-1)]).reshape(
            room.shape + (7,)
        )
        return room

class MiniMaze(SokobanSingleton):
    def __init__(
        self, 
        normalize_obs=False):
        maze_map = self._gen_grid()
        unmatched_boxes = 1 
        agent_pos = (0,0)

        super().__init__(
            agent_start_pos = agent_pos,
            maze_map=maze_map,
            unmatched_boxes=unmatched_boxes,
            dim_room=(5,5),
            num_boxes=1
        )

    def _gen_grid(self):
        # Create the grid
        room = jnp.array([
            [5, 0, 1, 1, 1],
            [1, 0, 4, 0, 1],
            [1, 0, 2, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
            ])
        room = jnp.squeeze(jnp.eye(7, dtype=jnp.uint8)[room.reshape(-1)]).reshape(
            room.shape + (7,)
        )
        return room
    # def step_env(self,
    #              key: chex.PRNGKey,
    #              state: EnvState,
    #              action: int
    #     ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        
    #     a = action
    #     jax.debug.print("pre_maze_map : {}",jnp.argmax(state.maze_map,axis=2))
    #     jax.debug.print("action : {}", a )
        
    #     new_state, reward = self.step_agent(key, state, a)
    #     new_state = new_state.replace(time=new_state.time+1)
    #     done = self.is_terminal(new_state)
    #     new_state = new_state.replace(terminal=done)
    #     jax.debug.print("maze_map : {}",jnp.argmax(new_state.maze_map,axis=2))
    #     jax.debug.print("action : {}", a )
    #     return (
    #         lax.stop_gradient(self.get_obs(new_state)),
    #         lax.stop_gradient(new_state),
    #         reward.astype(jnp.float32),
    #         done,
    #         {},
    #     )
class TwoRooms(SokobanSingleton):
    def __init__(
        self, 
        normalize_obs=False):
        maze_map = self._gen_grid()
        unmatched_boxes = 1 
        agent_pos = (1,1)

        super().__init__(
            agent_start_pos = agent_pos,
            maze_map=maze_map,
            unmatched_boxes=unmatched_boxes,
            dim_room=(10,10),
            num_boxes=1
        )

    def _gen_grid(self):
        # Create the grid
        room = jnp.zeros(shape = (10, 10, 7))
        room = room.at[:,:,FieldStates.empty].set(1)
        for i in range(10):
            room = room.at[i, 0, FieldStates.wall].set(1)
            room = room.at[i, 9, FieldStates.wall].set(1)
            room = room.at[i, 0, FieldStates.empty].set(0)
            room = room.at[i, 9, FieldStates.empty].set(0)
            
        for z in range(10):
            room = room.at[0, z, FieldStates.wall].set(1)
            room = room.at[9, z, FieldStates.wall].set(1)
            
            room = room.at[0, z, FieldStates.empty].set(0)
            room = room.at[9, z, FieldStates.empty].set(0)
        room = room.at[2,2,FieldStates.player].set(1)
        room = room.at[3,3,FieldStates.box].set(1)
        room = room.at[9,9,FieldStates.target].set(1)
        room = room.at[2,2,FieldStates.empty].set(0)
        room = room.at[3,3,FieldStates.empty].set(0)
        room = room.at[9,9,FieldStates.empty].set(0)

        for z in range(1,5) :
            room = room.at[5,z,FieldStates.wall].set(1)
            room = room.at[5,z,FieldStates.empty].set(0)
        for z in range(6,10) :
            room = room.at[5,z,FieldStates.wall].set(1)
            room = room.at[5,z,FieldStates.empty].set(0)
        return room


if hasattr(__loader__, 'name'):
  module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
  module_path = __loader__.fullname

register(env_id='Sokoban-TwoRooms', entry_point=module_path + ':TwoRooms')

register(env_id='Sokoban-MicroRoom', entry_point=module_path + ':MicroRoom')


register(env_id='Sokoban-MicroRoom', entry_point=module_path + ':MiniRoom')
register(env_id='Sokoban-MicroRoom', entry_point=module_path + ':Mini2Room')
register(env_id='Sokoban-MicroRoom', entry_point=module_path + ':MiniMaze')