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

@jax.jit
def sum(x: jax.Array) -> jax.Array:
  grid, *out_shape = x.shape   # grid is the first dimension of x
  return pl.pallas_call(
      sum_kernel,
      grid=grid,
      # None in `block_shape` means we pick a size of 1 and squeeze it away
      in_specs=[pl.BlockSpec((None, *out_shape), lambda i: (i, 0, 0))],
      out_specs=pl.BlockSpec(out_shape, lambda i: (0, 0)),
      out_shape=jax.ShapeDtypeStruct(out_shape, x.dtype)
  )(x)

arr = jax.numpy.ones((8, 512, 512))
assert np.allclose(sum(arr), arr.sum(axis=0))
t = benchmark(sum)(arr)
print(f"Time: {t} s")