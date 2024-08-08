import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp
import numpy as np
from timing_util import benchmark

"""
Note:
The below kernel is not scalable.
In VMEM we only have 16MB of memory.
f32[2048, 2048] occupies already 4 * 2048 * 2048 ~ 16.7 MB, so the kernel is already over the limit.
"""

def add_matrices_kernel(x_vmem_ref, y_vmem_ref, z_vmem_ref):
  # Load x and y from VMEM into VREGs
  x_vregs = x_vmem_ref[:, :]
  y_vregs = y_vmem_ref[:, :]
  # Execute a vectorized add
  z_vregs = x_vregs + y_vregs
  # Store the output values in VREGs back into VMEM
  z_vmem_ref[:, :] = z_vregs

@jax.jit
def add_matrices(x: jax.Array, y: jax.Array) -> jax.Array:
  # pallas_call will first allocate scratch buffers for `x` and `y` in VMEM.
  # It will then copy `x` and `y` from HBM into VMEM.
  z = pl.pallas_call(
      add_matrices_kernel, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype)
  )(x, y)
  # pallas_call will also copy the output from VMEM back into HBM.
  return z

A = jax.numpy.ones((2048, 2048), dtype=jnp.float32)
B = jax.numpy.ones((2048, 2048), dtype=jnp.float32)

# This will fail
C = add_matrices(A, B)