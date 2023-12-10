"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import jax
import jax.numpy as jnp

from minimax.envs.registration import register_comparator


@jax.jit
def is_equal_map(a, b):
    return jnp.equal(a.maze_map, b.maze_map).all()


# Register the mutators
if hasattr(__loader__, 'name'):
  module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
  module_path = __loader__.fullname

register_comparator(env_id='Sokoban', comparator_id=None, entry_point=module_path + ':is_equal_map')