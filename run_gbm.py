import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax.random as jr

from _gbm import multi_simulate_gbm

if __name__ == "__main__":
    keys = jr.split(jr.key(123), 100)
    sols = multi_simulate_gbm(
        1.0,
        0.1,
        1.0,
        0.01,
        jnp.array([1.0]),
        keys
    )
    plt.plot(sols.ts[0], sols.ys[..., 0].T, color="tab:red", alpha=0.25)
    plt.plot(sols.ts[0], jnp.exp(sols.ts[0]), color="black")
    plt.show()