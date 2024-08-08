import timeit
import jax 

def benchmark(f, ntrials: int = 10):
  def run(*args, **kwargs):
    # Compile function first
    jax.block_until_ready(f(*args, **kwargs))
    # Time function
    result = timeit.timeit(lambda: jax.block_until_ready(f(*args, **kwargs)),
                           number=ntrials)
    time = result / ntrials
    # print(f"Time: {time}")
    return time
  return run