from typing import Tuple, Optional

import jax
import jax.numpy as jnp
from flax import struct
import chex

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
        mode="one_hot",
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
            mode=mode,
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
            unmatched_boxes=unmatched_boxes
        )

    def _gen_grid(self):
        # Create the grid
        room = jnp.zeros(shape = (10+2, 10+2, 7))
        room = room.at[:,:,FieldStates.empty].set(1)
        for i in range(10+2):
            room = room.at[i, 0, FieldStates.wall].set(1)
            room = room.at[i, 10+1, FieldStates.wall].set(1)
        for z in range(10+2):
            room = room.at[0, z, FieldStates.wall].set(1)
            room = room.at[10+1, z, FieldStates.wall].set(1)
        room = room.at[2,2,FieldStates.player].set(1)
        room = room.at[3,3,FieldStates.box].set(1)
        room = room.at[9,9,FieldStates.box_target].set(1)
        for z in range(1,5) :
            room = room.at[5,z,FieldStates.wall].set(1)
        for z in range(6,10) :
            room.at[5,z,FieldStates.wall].set(1)
        return room


if hasattr(__loader__, 'name'):
  module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
  module_path = __loader__.fullname

register(env_id='Sokoban-TwoRooms', entry_point=module_path + ':TwoRooms')