
import copy

import numpy as np
import jax.numpy as jnp
from flax import struct
import chex


@struct.dataclass
class EnvInstance:
	agent_pos: chex.Array
	agent_dir_idx: int
	goal_pos: chex.Array
	wall_map: chex.Array
