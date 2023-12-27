
import copy

import numpy as np
import jax.numpy as jnp
from flax import struct
import chex


COLORS = {
    'black' : jnp.array([0,0,0]),
    'red'   : jnp.array([255, 0, 0]),
    'green' : jnp.array([0, 255, 0]),
    'blue'  : jnp.array([0, 0, 255]),
    'purple': jnp.array([112, 39, 195]),
    'yellow': jnp.array([255, 255, 0]),
    'grey'  : jnp.array([100, 100, 100])
}

OBJECT_TO_INDEX = {
	"empty": 0,
	"player": 1,
	"target": 2,
	"box_target": 3,
	"box": 4,
	"player": 5,
	"player_target": 6,
}



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