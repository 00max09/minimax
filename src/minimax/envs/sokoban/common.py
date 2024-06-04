
import copy

import numpy as np
import jax.numpy as jnp
from flax import struct
import chex


COLORS = {
    'grey_wall' : jnp.array([192,192,192]),
    'red'   : jnp.array([255, 0, 0]),
    'green' : jnp.array([0, 255, 0]),
    'blue'  : jnp.array([0, 0, 255]),
    'purple': jnp.array([112, 39, 195]),
    'yellow': jnp.array([255, 255, 0]),
    'grey'  : jnp.array([100, 100, 100])
}

OBJECT_TO_INDEX = {
	"wall": 0,
	"empty": 1,
	"target": 2,
	"box_target": 3,
	"box": 4,
	"player": 5,
	"player_target": 6,
}


COLOR_TO_INDEX = {
    'grey_wall' : 0,
    'red'   : 1,
    'green' : 2,
    'blue'  : 3,
    'purple': 4,
    'yellow': 5,
    'grey'  : 6,
}


# Map of agent direction indices to vectors
DIR_TO_VEC = jnp.array([
	# Pointing right (positive X)
	(1, 0), # right
	(0, 1), # down
	(-1, 0), # left
	(0, -1), # up
], dtype=jnp.int8)

@struct.dataclass
class EnvInstance:
	agent_pos: chex.Array
	maze_map: chex.Array
	unmatched_boxes : int

def render(params, one_hot):
    render_surfaces = jnp.stack([jnp.tile(y, (1,1,1)) for y in COLORS.values()])

    size_x = one_hot.shape[0]*render_surfaces.shape[1]
    size_y = one_hot.shape[1]*render_surfaces.shape[2]

    res = jnp.tensordot(one_hot, render_surfaces, (-1, 0))
    res = jnp.transpose(res, (0, 2, 1, 3, 4))
    res = jnp.reshape(res, (size_x, size_y, 3))
    return res