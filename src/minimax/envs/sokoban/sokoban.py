from minimax.envs import environment, spaces
import jax
import jax.numpy as jnp
from enum import IntEnum

import pkg_resources
from PIL import Image

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

RENDERING_MODES = ['one_hot', 'rgb_array', 'tiny_rgb_array']

class Sokoban(environment.Environment):
    def __init__(
        self,
        dim_room=(10, 10),
        max_steps=np.inf,
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
        self._seed = seed
        self.mode = mode
        self.num_gen_steps = num_gen_steps
        self.dim_room = dim_room
        self.max_steps = max_steps
        self.num_boxes = num_boxes

        # Penalties and Rewards
        self.penalty_for_step = penalty_for_step
        # self.penalty_box_off_target = penalty_box_off_target
        self.reward_box_on_target = reward_box_on_target
        self.reward_finished = reward_finished
        self.action_space = jnp.array([Actions.left, Actions.right, Actions.up, Actions.down])
        self.observation_space = (self.dim_room[0], self.dim_room[1], 7)
        self.state_space = self.observation_space  # state == observation

        self._internal_state = None
        self.fast_state_eq = fast_state_eq

        self._surfaces = load_surfaces()
        self.initial_internal_state_hash = None
        #self.load_boards_from_file = load_boards_from_file
        #self.boards_from_file = None
        
    def step_student(self, action):
        raw_state, rew, done = step(self._internal_state.get_raw(), action,
                                    self.penalty_for_step,
                                    self.reward_box_on_target,
                                    self.reward_finished)
        self._internal_state = HashableState(*raw_state, fast_eq=self.fast_state_eq)
        return self._internal_state.one_hot, rew, done, {"solved": done}

    def generate_default_room(self):
        """
        Generates basic empty Sokoban room with one box, represented by an integer matrix.
        The elements are encoded in one hot fashion
        :return: Numpy 3d Array
        """
        room = jnp.zeros(shape = (self.dim_room[0]+2, self.dim_room[1]+2, 7))
        room[:][:][FieldStates.empty] = 1
        for i in range(self.dim_room[0]+2):
            room[i][0][FieldStates.wall] = 1
            room[i][self.dim_room[1]+1][FieldStates.wall] = 1
        for z in range(self.dim_room[1]+2):
            room[0][z][FieldStates.wall] = 1
            room[self.dim_room[0]+1][FieldStates.wall] = 1
        room[2][2][FieldStates.player] = 1
        room[3][3][FieldStates.box] = 1
        room[4][4][FieldStates.box_target] = 1
        
        return room
    

    def reset(self):
        #if self.load_boards_from_file:
        #    if self.boards_from_file is None: # the case of lazy loading
        #        self.boards_from_file = np.load(self.load_boards_from_file)
        #    index = random.randint(0, len(self.boards_from_file)-1)
        #    one_hot = self.boards_from_file[index]
        #else:
        #    self._slave_env.reset()
        #    one_hot = self._slave_env.render(mode="one_hot")
        self.game_start_room = self.generate_default_room()
        self.restore_full_state_from_np_array_version(self.game_start_room)
        self.initial_internal_state_hash = hash(self._internal_state)
        return self._internal_state.one_hot
   
    def reset_student(self):
        self.restore_full_state_from_np_array_version(self.game_start_room)
        starting_observation = self.render()
        return starting_observation
        
    def render(self, mode='one_hot'):
        assert mode in RENDERING_MODES, f"Only {RENDERING_MODES} are supported, not {mode}"
        if mode == 'one_hot':
            return self._internal_state.one_hot
        render_surfaces = None
        if mode == 'rgb_array':
            render_surfaces = self._surfaces['16x16pixels']
        if mode == 'tiny_rgb_array':
            render_surfaces = self._surfaces['8x8pixels']

        size_x = self._internal_state.one_hot.shape[0]*render_surfaces.shape[1]
        size_y = self._internal_state.one_hot.shape[1]*render_surfaces.shape[2]

        res = jnp.tensordot(self._internal_state.one_hot, render_surfaces, (-1, 0))
        res = jnp.transpose(res, (0, 2, 1, 3, 4))
        res = jnp.reshape(res, (size_x, size_y, 3))
        return res


def load_surfaces():

    # Necessarily keep the same order as in FieldStates
    assets_file_name = ['wall.png', 'floor.png', 'box_target.png', 'box_on_target.png',
                        'box.png', 'player.png', 'player_on_target.png']
    sizes = ['8x8pixels', '16x16pixels']

    resource_package = __name__
    surfaces = {}
    for size in sizes:
        surfaces[size] = []
        for asset_file_name in assets_file_name:
            asset_path = pkg_resources.resource_filename(resource_package, '/'.join(('surface', size, asset_file_name)))
            asset_np_array = jnp.array(Image.open(asset_path))
            surfaces[size].append(asset_np_array)

        surfaces[size] = jnp.stack(surfaces[size])

    return surfaces


def step(state, action, penalty_for_step, reward_box_on_target, reward_finished):
    # Copy to be supported by numba. Possibly can be done better
    # wall = 0
    empty = 1
    target = 2
    box_target = 3
    box = 4
    player = 5
    player_target = 6

    delta_x, delta_y = None, None
    if action == 0:
        delta_x, delta_y = -1, 0
    elif action == 1:
        delta_x, delta_y = 1, 0
    elif action == 2:
        delta_x, delta_y = 0, -1
    elif action == 3:
        delta_x, delta_y = 0, 1

    one_hot, agent_pos, unmatched_boxes = state

    arena = jnp.zeros(shape=(3,), dtype=jnp.uint8)
    for i in range(3):
        index_x = agent_pos[0] + i * delta_x
        index_y = agent_pos[1] + i * delta_y
        if index_x < one_hot.shape[0] and index_y < one_hot.shape[0]:
            arena[i] = jnp.where(one_hot[index_x, index_y, :] == 1)[0][0]

    new_unmatched_boxes_ = unmatched_boxes
    new_agent_pos = agent_pos
    new_arena = jnp.copy(arena)

    box_moves = (arena[1] == box or arena[1] == box_target) and \
                (arena[2] == empty or arena[2] == 2)

    agent_moves = arena[1] == empty or arena[1] == target or box_moves

    if agent_moves:
        targets = (arena == target).astype(jnp.int8) + \
                  (arena == box_target).astype(jnp.int8) + \
                  (arena == player_target).astype(jnp.int8)
        if box_moves:
            last_field = box - 2 * targets[2]  # Weirdness due to inconsistent target non-target
        else:
            last_field = arena[2] - targets[2]

        new_arena = jnp.array([empty, player, last_field]).astype(jnp.uint8) + targets.astype(jnp.uint8)
        new_agent_pos = (agent_pos[0] + delta_x, agent_pos[1] + delta_y)

        if box_moves:
            new_unmatched_boxes_ = int(unmatched_boxes - (targets[2] - targets[1]))

    new_one_hot = jnp.copy(one_hot)
    for i in range(3):
        index_x = agent_pos[0] + i * delta_x
        index_y = agent_pos[1] + i * delta_y
        if index_x < one_hot.shape[0] and index_y < one_hot.shape[0]:
            one_hot_field = jnp.zeros(shape=7)
            one_hot_field[new_arena[i]] = 1
            new_one_hot[index_x, index_y, :] = one_hot_field

    done = (new_unmatched_boxes_ == 0)
    reward = penalty_for_step - reward_box_on_target * (float(new_unmatched_boxes_) - float(unmatched_boxes))
    if done:
        reward += reward_finished

    new_state = (new_one_hot, new_agent_pos, new_unmatched_boxes_)

    return new_state, reward, done

class HashableState:
    state = jnp.random.get_state()
    jnp.random.seed(0)
    hash_key = jnp.random.normal(size=10000)
    jnp.random.set_state(state)

    def __init__(self, one_hot, agent_pos, unmached_boxes, fast_eq=False):
        self.one_hot = one_hot
        self.agent_pos = agent_pos
        self.unmached_boxes = unmached_boxes
        self._hash = None
        self.fast_eq = fast_eq
        self._initial_state_hash = None

    def __iter__(self):
        yield from [self.one_hot, self.agent_pos, self.unmached_boxes]

    def __hash__(self):
        if self._hash is None:
            flat_np = self.one_hot.flatten()
            self._hash = int(jnp.dot(flat_np, HashableState.hash_key[:len(flat_np)]) * 10e8)
        return self._hash

    def __eq__(self, other):
        if self.fast_eq:
            return hash(self) == hash(other)  # This is a conscious decision to speed up.
        else:
            return jnp.array_equal(self.one_hot, other.one_hot)

    def __ne__(self, other):
        return not self.__eq__(other)

    def get_raw(self):
        return self.one_hot, self.agent_pos, self.unmached_boxes

    def get_np_array_version(self):
        return self.one_hot
