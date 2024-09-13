from functools import partial
import jax
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as np
import jax.example_libraries.optimizers

import jax.experimental.host_callback
debug = True
print_level = 1
def jax_print(x, *, level=0, **kwargs):
    # filter prints by debug flag and verbosity level
    if debug and level <= print_level:
        jax.experimental.host_callback.id_print(x, **kwargs)

# NON-HOLOMORPHIC
@partial(jax.jit)
def norm_3vec(Rij):
    return np.sqrt(np.abs(np.sum(Rij**2, axis=-1)))

def to_relative(R):
    assert R.shape[1] == 2, "must be a 2-nucleon system"
    return R[:,0] - R[:,1]
