
import copy

import numpy as np
import jax.numpy as jnp
from flax import struct
import chex


COLORS = {
    'black' : np.array([0,0,0]),
    'red'   : np.array([255, 0, 0]),
    'green' : np.array([0, 255, 0]),
    'blue'  : np.array([0, 0, 255]),
    'purple': np.array([112, 39, 195]),
    'yellow': np.array([255, 255, 0]),
    'grey'  : np.array([100, 100, 100])
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
    
    render_surfaces = np.stack([np.tile(i, (8,8,1)) for i in COLORS])

    size_x = one_hot.shape[0]*render_surfaces.shape[1]
    size_y = one_hot.shape[1]*render_surfaces.shape[2]

    res = np.tensordot(one_hot, render_surfaces, (-1, 0))
    res = np.transpose(res, (0, 2, 1, 3, 4))
    res = np.reshape(res, (size_x, size_y, 3))
    return res