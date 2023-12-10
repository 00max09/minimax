
import copy

import numpy as np
import jax.numpy as jnp
from flax import struct
import chex


@struct.dataclass
class EnvInstance:
	agent_pos: chex.Array
	maze_map: chex.Array
	unmatched_boxes : int