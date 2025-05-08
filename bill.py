from jax import numpy as jnp
import numpy as np
import marray

ja = jnp.arange(5) + 1
print(f"{ja.sum()=}")

mxp = marray.__getattr__("jax.numpy")
a = mxp.asarray(ja)
print(f"{mxp.sum(a)=}")

print(f"{np.prod(a)=}")

print(f"{a.prod()=}")