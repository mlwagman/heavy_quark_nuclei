from functools import partial
import jax
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as np
#import jax.experimental.optimizers
import jax.example_libraries.optimizers

def parse_table(fname):
    rs = []
    vnn = []
    with open(fname, 'r') as f:
        for line in f:
            if '#' in line:
                line = line.split('#', 1)[0]
            line = line.strip()
            if len(line) == 0: continue
            tokens = list(filter(lambda x: len(x) > 0, line.split(' ')))
            assert len(tokens) == 8
            rs.append(float(tokens[0]))
            vnn.append(tuple(float(t) for t in tokens[1:7])) # skip vem
    return np.array(rs), np.array(vnn)

class hashabledict(dict):
  def __key(self):
    return tuple((k,self[k]) for k in sorted(self))
  def __hash__(self):
    return hash(self.__key())
  def __eq__(self, other):
    return self.__key() == other.__key()

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

# HOLOMORPHIC
@partial(jax.jit)
def norm_3vec_sq(Rij):
    return np.sum(Rij**2, axis=-1)

def to_relative(R):
    assert R.shape[1] == 2, "must be a 2-nucleon system"
    return R[:,0] - R[:,1]
