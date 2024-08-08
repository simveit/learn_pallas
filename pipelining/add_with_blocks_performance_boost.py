import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp
import numpy as np
from timing_util import benchmark
import functools
from plotting_util import plot_and_save_heatmap

"""
In this example we will show how to tune the block sizes for pipelining to get the best performance.
Please note that this is very simple example and the performance might not be optimal.
"""

# Kernel stays same as above
def add_matrices_kernel(x_vmem_ref, y_vmem_ref, z_vmem_ref):
  # Load x and y from VMEM into VREGs
  x_vregs = x_vmem_ref[:, :]
  y_vregs = y_vmem_ref[:, :]
  # Execute a vectorized add
  z_vregs = x_vregs + y_vregs
  # Store the output values in VREGs back into VMEM
  z_vmem_ref[:, :] = z_vregs

"""
In the below kernel we parametrize the block sizes.
We calculate the grid size accordingly.
If we have for example 256x2048 blocks g0 = 2048 // 256 = 8 and g1 = 2048 // 2048 = 1.
"""
@functools.partial(jax.jit, static_argnames=['b0', 'b1'])
def add_matrices_pipelined(x: jax.Array, y: jax.Array, b0: int, b1: int) -> jax.Array:
  block_spec = pl.BlockSpec((b0, b1), lambda i, j: (i, j))
  g0 = x.shape[0] // b0
  g1 = x.shape[1] // b1
  return pl.pallas_call(
      add_matrices_kernel,
      out_shape=x,
      in_specs=[block_spec, block_spec],
      out_specs=block_spec,
      grid=(g0, g1)
  )(x, y)

BLOCK_SIZES = [(128, 128), (128, 256), (128, 512), (128, 1024), (128, 2048),
               (256, 256), (256, 512), (256, 1024), (256, 2048),
               (512, 512,), (512, 1024), (512, 2048),
               (1024, 1024)]

times = {}
for b0, b1 in BLOCK_SIZES:
  A = jax.numpy.ones((32_768, 32_768), dtype=jnp.float32)
  B = jax.numpy.ones((32_768, 32_768), dtype=jnp.float32)
  time = benchmark(add_matrices_pipelined, ntrials=10)(A, B, b0, b1)
  print(f"Time for block size {b0}x{b1}: {time}")
  assert jnp.allclose(add_matrices_pipelined(A, B, b0, b1), A + B)
  times[(b0, b1)] = time

plot_and_save_heatmap(times, BLOCK_SIZES, filename='heatmap_pipelined.png')