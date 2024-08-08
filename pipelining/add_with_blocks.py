import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp
import numpy as np
from timing_util import benchmark

"""
We saw in naive_add_kernel.py that the kernel was not scalable.
The solution to that is a concept called pipelining.
The idea is to split the computation into smaller blocks that fit into VMEM.
While we compute the result for chunk i, we can already start copying chunk i+1 from HBM to VMEM. 
This will allow to both deal with memory constrains as well as overcoming bandwith constraints.
For more details, see: https://jax.readthedocs.io/en/latest/pallas/tpu/pipelining.html#tpu-and-its-memory-spaces
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
Below we define the pipelined version of matrix addition.
We chunk our matrices of size (2048, 2048) into smaller blocks of size (256, 2048).
Grid size is 8, so we run the kernel 8 times, each time with a different block.
The individual blocks always have the size (256, 2048).
Note: The block size here is chosen arbitrarily and not tuned for performance.
      This example is intended to show the concept of pipelining to overcome memory constraints.
"""
@jax.jit
def add_matrices_pipelined(x: jax.Array, y: jax.Array) -> jax.Array:
  block_spec = pl.BlockSpec((256, 2048), lambda i: (i, 0))
  return pl.pallas_call(
      add_matrices_kernel,
      out_shape=x,
      in_specs=[block_spec, block_spec],
      out_specs=block_spec,
      grid=(8,)
  )(x, y)

A = jax.numpy.ones((2048, 2048), dtype=jnp.float32)
B = jax.numpy.ones((2048, 2048), dtype=jnp.float32)

# This will work
C = add_matrices_pipelined(A, B)