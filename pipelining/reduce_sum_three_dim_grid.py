import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp
import numpy as np
from timing_util import benchmark
import functools

"""
Perform simple reduction along first axis.
"""
def sum_kernel(x_ref, o_ref):
  @pl.when(pl.program_id(axis=0) == 0)
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
      grid=(g0, g1, g2),
      # None in `block_shape` means we pick a size of 1 and squeeze it away
      in_specs=[pl.BlockSpec((None, b1, b2), lambda i, j, k: (i, j, k))],
      out_specs=pl.BlockSpec((b1, b2), lambda i, j, k: (j, k)),
      out_shape=jax.ShapeDtypeStruct(out_shape, x.dtype)
  )(x)

arr = jax.numpy.ones((8, 512, 512))
# Currently only supports 2D reduction, i.e. b1 or b2 must be 512.
# TODO: Support 3D reduction
b1 = 256
b2 = 512
assert np.allclose(sum(arr, b1, b2), arr.sum(axis=0))
t = benchmark(sum)(arr, b1, b2)
print(f"Time: {t} s")