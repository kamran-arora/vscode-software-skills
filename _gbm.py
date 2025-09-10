import jax
import jax.numpy as jnp
import diffrax as dfx
import lineax as lx

"""
Simulate geometric Brownian motion in Diffrax using Euler-Maruyama timestepping
"""

def _vf(t, y, args):
    mu, sigma = args
    return mu*y

def _diffusion(t, y, args):
    mu, sigma = args
    return lx.DiagonalLinearOperator(sigma*y)

def _get_terms(bm):
    return dfx.MultiTerm(dfx.ODETerm(_vf), dfx.ControlTerm(_diffusion, bm))

def simulate_gbm(mu: float, sigma: float, T: float, dt0: float, y0: jax.Array, key):
    """
    Simulate a geometric Brownian motion

    **args:**
    - mu: drift
    - sigma: volatility
    - T: final time
    - dt0: timestep
    - y0: initial condition
    - key: jax.random.key
    """
    args = (mu, sigma)
    ts = jnp.linspace(0.0, T, int(T/dt0)+1)
    saveat=dfx.SaveAt(ts=ts)
    bm = dfx.UnsafeBrownianPath(shape=(1,), key=key)
    terms = _get_terms(bm)
    sol = dfx.diffeqsolve(
        terms,
        dfx.Euler(),
        0.0,
        T,
        0.01,
        y0,
        args,
        saveat=saveat,
        adjoint=dfx.ForwardMode(),
    )
    return sol

multi_simulate_gbm = jax.vmap(
    simulate_gbm,
    in_axes=(
        None,
        None,
        None,
        None,
        None,
        0,
    )
)