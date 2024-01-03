from minimax.envs import environment, spaces
import jax
from jax import lax
import jax.numpy as jnp
from enum import IntEnum
import chex
from flax import struct
from flax.core.frozen_dict import FrozenDict
import pkg_resources
import sys

from minimax.envs.registration import register
from collections import namedtuple, OrderedDict

from typing import Tuple, Optional

from .common import EnvInstance, render


class Actions(IntEnum):
    left = 0
    right = 1
    up = 2
    down = 3


class FieldStates(IntEnum):
    wall = 0
    empty = 1
    target = 2
    box_target = 3
    box = 4
    player = 5
    player_target = 6


RENDERING_MODES = ["one_hot", "rgb_array", "tiny_rgb_array"]


@struct.dataclass
class EnvState:
    agent_pos: chex.Array
    maze_map: chex.Array
    start_map: chex.Array
    time: int
    terminal: bool
    unmatched_boxes: int


@struct.dataclass
class EnvParams:
    height: int = 15
    width: int = 15
    replace_wall_pos: bool = False
    max_episode_steps: int = 250
    singleton_seed: int = -1
    num_boxes: int = 4


class Sokoban(environment.Environment):
    def __init__(
        self,
        height=10,
        width=10,
        max_episode_steps=250,
        num_boxes=4,
        num_gen_steps=None,
        fast_state_eq=False,
        penalty_for_step=-0.1,
        # currently penalty_box_off_target is implicitly = - reward_box_on_target
        # penalty_box_off_target=-1,
        reward_box_on_target=1,
        reward_finished=10,
        seed=None,
    ):
        super().__init__()
        self.obs_shape = (height, width, 3)
        self.action_set = jnp.array(
            [Actions.left, Actions.right, Actions.up, Actions.down]
        )

        self.params = EnvParams(
            height=height,
            width=width,
            # replace_wall_pos=replace_wall_pos and not sample_n_walls,
            max_episode_steps=max_episode_steps,
            num_boxes=num_boxes,
            # max_episode_steps=max_episode_steps,
            # normalize_obs=normalize_obs,
            # sample_n_walls=sample_n_walls,
            # obs_agent_pos=obs_agent_pos,
            singleton_seed=-1,
            # num_gen_steps = num_gen_steps,
            # penalty_for_step = penalty_for_step,
            # reward_box_on_target = reward_box_on_target,
            # reward_finished = reward_finished,
        )

        # Penalties and Rewards
        # self.penalty_box_off_target = penalty_box_off_target

        # self.state_space = self.observation_space  # state == observation

        # self._internal_state = None
        # self.fast_state_eq = fast_state_eq

        # self._surfaces = load_surfaces()
        # self.initial_internal_state_hash = None
        # self.load_boards_from_file = load_boards_from_file
        # self.boards_from_file = None

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters
        return EnvParams()

    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: int
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        # jax.debug.print("pre_maze_map : {}",jnp.argmax(state.maze_map,axis=2))
        # jax.debug.print("action : {}", a )
        a = action
        new_state, reward = self.step_agent(key, state, a)
        new_state = new_state.replace(time=new_state.time + 1)
        done = self.is_terminal(state)
        new_state = new_state.replace(terminal=done)
        # jax.debug.print("maze_map : {}",jnp.argmax(new_state.maze_map,axis=2))
        # jax.debug.print("action : {}", a )
        return (
            lax.stop_gradient(self.get_obs(new_state)),
            lax.stop_gradient(new_state),
            reward.astype(jnp.float32),
            done,
            {},
        )
        # self._internal_state = HashableState(*raw_state, fast_eq=self.fast_state_eq)
        # return (self._internal_state.one_hot, rew, done, {"solved": done}

    def reset_env(self, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        params = self.params
        map_size = params.width * params.height
        agent_box_end_poses = jax.random.permutation(key, map_size)
        room = jax.random.randint(key, (params.height, params.width), 0, 2)
        agent_pos = (
            agent_box_end_poses[0] // params.width,
            agent_box_end_poses[0] % params.width,
        )
        box_pos = (
            agent_box_end_poses[1] // params.width,
            agent_box_end_poses[1] % params.width,
        )
        end_pos = (
            agent_box_end_poses[2] // params.width,
            agent_box_end_poses[2] % params.width,
        )

        room = room.at[agent_pos].set(FieldStates.player)
        room = room.at[box_pos].set(FieldStates.box)
        room = room.at[end_pos].set(FieldStates.target)
        room = jnp.squeeze(jnp.eye(7, dtype=jnp.uint8)[room.reshape(-1)]).reshape(
            room.shape + (7,)
        )
        unmatched_boxes_ = 1
        state = EnvState(
            agent_pos=agent_pos,
            maze_map=room,
            start_map=room,
            time=0,
            terminal=False,
            unmatched_boxes=unmatched_boxes_,
        )
        return self.get_obs(state), state

    def set_env_instance(self, encoding: EnvInstance):
        """
        Instance is encoded as a PyTree containing the following fields:
        agent_pos, agent_dir, goal_pos, wall_map
        """
        # params = self.params
        agent_pos = encoding.agent_pos
        maze_map = encoding.maze_map
        unmatched_boxes = encoding.unmatched_boxes

        state = EnvState(
            agent_pos=agent_pos,
            maze_map=maze_map,
            start_map=maze_map,
            time=0,
            terminal=False,
            unmatched_boxes=unmatched_boxes,
        )

        return self.get_obs(state), state

    def get_obs(self, state: EnvState) -> chex.Array:
        """Return grid view."""

        image = render(self.params, state.maze_map).astype(jnp.uint8)
        # if self.params.normalize_obs:
        #    image = image/10.0

        obs_dict = dict(image=image)

        return OrderedDict(obs_dict)

    def step_agent(
        self, key: chex.PRNGKey, state: EnvState, action: int
    ) -> Tuple[EnvState, float]:
        # Copy to be supported by numba. Possibly can be done better
        params = self.params
        # wall = 0
        empty = 1
        target = 2
        box_target = 3
        box = 4
        player = 5
        player_target = 6

        delta_x, delta_y = None, None

        delta_x = (action < 2) * (2 * action - 1)

        delta_y = (action >= 2) * (2 * action - 5)
        # if action == 0:
        #     delta_x, delta_y = -1, 0
        # elif action == 1:
        #     delta_x, delta_y = 1, 0
        # elif action == 2:
        #     delta_x, delta_y = 0, -1
        # elif action == 3:
        #     delta_x, delta_y = 0, 1

        one_hot = state.maze_map
        agent_pos = state.agent_pos
        unmatched_boxes = state.unmatched_boxes

        arena = jnp.zeros(shape=(3,), dtype=jnp.uint8)

        for i in range(3):
            index_x = agent_pos[0] + i * delta_x
            index_y = agent_pos[1] + i * delta_y
            is_in_place = jnp.logical_and(
                jnp.logical_and(index_x < one_hot.shape[0], index_y < one_hot.shape[0]),
                jnp.logical_and(index_x >= 0, index_y >= 0),
            )
            def _find_pos(_):
                return jnp.where(one_hot[index_x, index_y, :] == 1, size=1, fill_value=0)[0][0]
            
            value_to_set = jax.lax.cond(
                is_in_place,
                _find_pos,
                lambda _: 0,
                None
            ) 
            #jnp.where(
            #     one_hot[index_x, index_y, :] == 1, size=1, fill_value=0
            # )[0][0]
            # jax.debug.print("value_to_set : {}", value_to_set)

            arena = arena.at[i].set(
                value_to_set
            )  # _find_pos(one_hot, index_x, index_y))
            # jax.debug.print("arenax : {}", arena[i])

        # jax.debug.print("arena : {}", arena)
        new_unmatched_boxes_ = unmatched_boxes
        new_agent_pos = agent_pos
        new_arena = jnp.copy(arena)
        # jax.debug.print("new_arena : {}",new_arena)

        box_moves = jnp.logical_and(
            jnp.logical_or(arena[1] == box, arena[1] == box_target),
            jnp.logical_or(arena[2] == empty, arena[2] == 2),
        )

        agent_moves = jnp.logical_or(
            arena[1] == empty, jnp.logical_or(arena[1] == target, box_moves)
        )

        def _move_agent(_):
            targets = (
                (arena == target).astype(jnp.uint8)
                + (arena == box_target).astype(jnp.uint8)
                + (arena == player_target).astype(jnp.uint8)
            )
            last_field = jax.lax.cond(
                box_moves,
                lambda _: box - 2 * targets[2],
                lambda _: (arena[2] - targets[2]).astype(jnp.uint8),
                None,
            )
            new_arena = jnp.array([empty, player, last_field]).astype(
                jnp.uint8
            ) + targets.astype(jnp.uint8)
            new_agent_pos = (
                (agent_pos[0] + delta_x).astype(jnp.int32),
                (agent_pos[1] + delta_y).astype(jnp.int32),
            )
            new_unmatched_boxes2 = jax.lax.cond(
                box_moves,
                lambda _: (unmatched_boxes - (targets[2] - targets[1])).astype(
                    jnp.int32
                ),
                lambda _: new_unmatched_boxes_,
                None,
            )
            return new_arena, new_agent_pos, new_unmatched_boxes2

        # return new_one_hot, new_agent_pos, new_unmatched_boxes_, done, reward
        new_arena, new_agent_pos, new_unmatched_boxes_ = jax.lax.cond(
            agent_moves,
            _move_agent,
            lambda *x: (new_arena, new_agent_pos, new_unmatched_boxes_),
            None,
        )

        # targets = (arena == target).astype(jnp.int8) + \
        #           (arena == box_target).astype(jnp.int8) + \
        #           (arena == player_target).astype(jnp.int8)
        # if box_moves:
        #     last_field = box - 2 * targets[2]  # Weirdness due to inconsistent target non-target
        # else:
        #     last_field = arena[2] - targets[2]

        # new_arena = jnp.array([empty, player, last_field]).astype(jnp.uint8) + targets.astype(jnp.uint8)
        # new_agent_pos = (agent_pos[0] + delta_x, agent_pos[1] + delta_y)

        # if box_moves:
        #     new_unmatched_boxes_ = int(unmatched_boxes - (targets[2] - targets[1]))

        new_one_hot = jnp.copy(one_hot)
        for i in range(3):
            index_x = agent_pos[0] + i * delta_x
            index_y = agent_pos[1] + i * delta_y

            # if index_x < one_hot.shape[0] and index_y < one_hot.shape[0]:
            one_hot_field = jnp.zeros(shape=7)
            one_hot_field = one_hot_field.at[new_arena[i]].set(1)
            new_one_hot = new_one_hot.at[index_x, index_y, :].set(
                one_hot_field, mode="drop"
            )

        done = new_unmatched_boxes_ == 0
        # reward = 0.1 - 1*((new_unmatched_boxes_.astype(float) - unmatched_boxes.astype(float))) #0.1 - params.reward_box_on_target * (float(new_unmatched_boxes_) - float(unmatched_boxes))
        reward = 10 * done
        return (
            state.replace(
                maze_map=new_one_hot,
                agent_pos=new_agent_pos,
                unmatched_boxes=new_unmatched_boxes_,
                terminal=done,
            ),
            reward,
        )

    # def reset(self):
    #     #if self.load_boards_from_file:
    #     #    if self.boards_from_file is None: # the case of lazy loading
    #     #        self.boards_from_file = np.load(self.load_boards_from_file)
    #     #    index = random.randint(0, len(self.boards_from_file)-1)
    #     #    one_hot = self.boards_from_file[index]
    #     #else:
    #     #    self._slave_env.reset()
    #     #    one_hot = self._slave_env.render(mode="one_hot")
    #     self.game_start_room = self.generate_default_room()
    #     self.restore_full_state_from_np_array_version(self.game_start_room)
    #     self.initial_internal_state_hash = hash(self._internal_state)
    #     return self._internal_state.one_hot
    def is_terminal(self, state: EnvState) -> bool:
        """Check whether state is terminal."""
        done_steps = state.time >= self.params.max_episode_steps
        return jnp.logical_or(done_steps, state.terminal)

    @property
    def name(self) -> str:
        """Environment name."""
        return "Sokoban"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return len(self.action_set)

    def action_space(self) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(len(self.action_set), dtype=jnp.uint32)

    def observation_space(self) -> spaces.Dict:
        """Observation space of the environment."""
        spaces_dict = {
            "image": spaces.Box(0, 255, self.obs_shape),
        }
        return spaces.Dict(spaces_dict)

    # box of maze map is probably wrong
    def state_space(self) -> spaces.Dict:
        """State space of the environment."""
        params = self.params
        h = params.height
        w = params.width
        return spaces.Dict(
            {
                "agent_pos": spaces.Box(0, max(w, h), (2,), dtype=jnp.uint32),
                "maze_map": spaces.Box(0, 255, (w, h, 3), dtype=jnp.uint32),
                "time": spaces.Discrete(params.max_episode_steps),
                "terminal": spaces.Discrete(2),
            }
        )

    def max_episode_steps(self) -> int:
        return self.params.max_episode_steps

    # @TODO
    def get_env_metrics(self, state: EnvState) -> dict:
        # n_walls = state.wall_map.sum()
        # shortest_path_length = _graph_util.shortest_path_len(
        #     state.wall_map,
        #     state.agent_pos,
        #     state.goal_pos
        #
        n_walls = jnp.sum(state.maze_map[:][:][FieldStates.wall])
        return dict(
            n_wall=n_walls
            # n_walls=n_walls,
            #    shortest_path_length=shortest_path_length,
            #   passable=shortest_path_length > 0,
        )

    def get_eval_solved_rate_fn(self):
        def _fn(ep_stats):
            return ep_stats["return"] > 0

        return _fn

    # def reset_student(self):
    #     self.restore_full_state_from_np_array_version(self.game_start_room)
    #     starting_observation = self.render()
    #     return starting_observation

    # def render(self, mode='one_hot'):
    #     assert mode in RENDERING_MODES, f"Only {RENDERING_MODES} are supported, not {mode}"
    #     if mode == 'one_hot':
    #         return self._internal_state.one_hot
    #     render_surfaces = None
    #     if mode == 'rgb_array':
    #         render_surfaces = self._surfaces['16x16pixels']
    #     if mode == 'tiny_rgb_array':
    #         render_surfaces = self._surfaces['8x8pixels']

    #     size_x = self._internal_state.one_hot.shape[0]*render_surfaces.shape[1]
    #     size_y = self._internal_state.one_hot.shape[1]*render_surfaces.shape[2]

    #     res = jnp.tensordot(self._internal_state.one_hot, render_surfaces, (-1, 0))
    #     res = jnp.transpose(res, (0, 2, 1, 3, 4))
    #     res = jnp.reshape(res, (size_x, size_y, 3))
    #     return res


# def load_surfaces():

#     # Necessarily keep the same order as in FieldStates
#     assets_file_name = ['wall.png', 'floor.png', 'box_target.png', 'box_on_target.png',
#                         'box.png', 'player.png', 'player_on_target.png']
#     sizes = ['8x8pixels', '16x16pixels']

#     resource_package = __name__
#     surfaces = {}
#     for size in sizes:
#         surfaces[size] = []
#         for asset_file_name in assets_file_name:
#             asset_path = pkg_resources.resource_filename(resource_package, '/'.join(('surface', size, asset_file_name)))
#             asset_np_array = jnp.array(Image.open(asset_path))
#             surfaces[size].append(asset_np_array)

#         surfaces[size] = jnp.stack(surfaces[size])

#     return surfaces


# class HashableState:
#     state = jnp.random.get_state()
#     jnp.random.seed(0)
#     hash_key = jnp.random.normal(size=10000)
#     jnp.random.set_state(state)

#     def __init__(self, one_hot, agent_pos, unmached_boxes, fast_eq=False):
#         self.one_hot = one_hot
#         self.agent_pos = agent_pos
#         self.unmached_boxes = unmached_boxes
#         self._hash = None
#         self.fast_eq = fast_eq
#         self._initial_state_hash = None

#     def __iter__(self):
#         yield from [self.one_hot, self.agent_pos, self.unmached_boxes]

#     def __hash__(self):
#         if self._hash is None:
#             flat_np = self.one_hot.flatten()
#             self._hash = int(jnp.dot(flat_np, HashableState.hash_key[:len(flat_np)]) * 10e8)
#         return self._hash

#     def __eq__(self, other):
#         if self.fast_eq:
#             return hash(self) == hash(other)  # This is a conscious decision to speed up.
#         else:
#             return jnp.array_equal(self.one_hot, other.one_hot)

#     def __ne__(self, other):
#         return not self.__eq__(other)

#     def get_raw(self):
#         return self.one_hot, self.agent_pos, self.unmached_boxes

#     def get_np_array_version(self):
#         return self.one_hot

# Register the env
if hasattr(__loader__, "name"):
    module_path = __loader__.name
elif hasattr(__loader__, "fullname"):
    module_path = __loader__.fullname

register(env_id="Sokoban", entry_point=module_path + ":Sokoban")


# def _find_pos(one_hot: chex.Array, x: jnp.uint32, y: jnp.uint32) -> jnp.uint8:
#     jax.debug.print("pos : {}", jnp.where(one_hot[x, y, :] == 1, size=1, fill_value=0))

#     jax.debug.print(
#         "pos : {}", jnp.where(one_hot[x, y, :] == 1, size=1, fill_value=0)[0]
#     )

#     jax.debug.print(
#         "pos : {}", jnp.where(one_hot[x, y, :] == 1, size=1, fill_value=0)[0][0]
#     )
#     return jnp.where(one_hot[x, y, :] == 1, size=1, fill_value=0)[0][0]
