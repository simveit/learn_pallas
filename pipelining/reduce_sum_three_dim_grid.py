import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp
import numpy as np
from timing_util import benchmark
import functools
import matplotlib.pyplot as plt
import seaborn as sns

"""
Perform simple reduction along first axis.
"""
def sum_kernel(x_ref, o_ref):
  @pl.when(pl.program_id(axis=2) == 0)
  def _():
    o_ref[...] = jnp.zeros_like(o_ref)

  o_ref[...] += x_ref[...]

@functools.partial(jax.jit, static_argnames=["b1", "b2"])
def sum(x: jax.Array, b1:int, b2:int) -> jax.Array:
  g0, *out_shape = x.shape   # grid is the first dimension of x
  g1 = out_shape[0] // b1
  g2 = out_shape[1] // b2
  return pl.pallas_call(
      sum_kernel,
      grid=(g1, g2, g0),
      # None in `block_shape` means we pick a size of 1 and squeeze it away
      # NOTE: We need to reduce over the rightmost dimension! 
      # SEE: https://github.com/google/jax/issues/23099
      in_specs=[pl.BlockSpec((None, b1, b2), lambda i, j, k: (k, i, j))],
      out_specs=pl.BlockSpec((b1, b2), lambda i, j, k: (i, j)),
      out_shape=jax.ShapeDtypeStruct(out_shape, x.dtype)
  )(x)

arr = jax.numpy.ones((8, 4096, 4096))
# Currently only supports 2D reduction, i.e. b1 or b2 must be 512.
time_diff = []
for b1 in [256, 512, 1024]:
  for b2 in [256, 512, 1024]:
    assert np.allclose(sum(arr, b1, b2), arr.sum(axis=0))
    t = benchmark(sum)(arr, b1, b2)
    t1 = benchmark(arr.sum)(axis=0)
    time_diff.append(t1 - t)


time_diff_matrix = np.array(time_diff).reshape(3, 3)
sns.heatmap(time_diff_matrix, annot=True, cmap='YlGnBu', xticklabels=[256, 512, 1024], yticklabels=[256, 512, 1024])
plt.xlabel('b2')
plt.ylabel('b1')
plt.title(f'Time Difference over Block sizes\nOperation: Sum reduce first dim of {arr.shape}', fontsize=8)
plt.savefig('time_difference_three_dim_reduce.png')