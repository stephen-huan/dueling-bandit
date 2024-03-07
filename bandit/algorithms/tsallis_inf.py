from functools import partial
from typing import TypeAlias

import jax.numpy as jnp
from jax import Array, lax, random

from ..utils import Draw, KeyArray, Loss, S, jit

# Tsallis-INF ([11])


@jit
def learning_rate(t: int, rv: bool) -> Array:
    """Return the learning rate (theorem 1 of [11])."""
    return jnp.where(
        rv,
        4 * jnp.sqrt(jnp.reciprocal(t)),
        2 * jnp.sqrt(jnp.reciprocal(t)),
    )


@jit
def omd_w(x: float | Array, losses: Array, lr: Array) -> Array:
    """Return the weights for a given normalizing constant."""
    return 4 / jnp.square(lr * (losses - x))


@jit
def omd_monotone(
    losses: Array,
    lr: Array,
    eps: float | Array = 1e-12,
) -> tuple[Array, Array]:
    """Binary search for the normalizing factor."""
    right = jnp.min(losses)
    left = lax.while_loop(
        lambda left: jnp.sum(omd_w(left, losses, lr)) > 1,
        lambda left: left * 2,
        -jnp.ones((), dtype=losses.dtype),
    )
    Data: TypeAlias = tuple[Array, Array, Array]  # type: ignore

    def midpoint(left: Array, right: Array) -> Array:
        """Midpoint of left and right."""
        return (left + right) / 2

    def body_fun(data: Data) -> Data:
        """Binary search."""
        left, right, _ = data
        middle = midpoint(left, right)
        w_sum = jnp.sum(omd_w(middle, losses, lr))
        right = jnp.where(w_sum > 1, middle, right)
        left = jnp.where(w_sum > 1, left, middle)
        return left, right, w_sum

    left, right, _ = lax.while_loop(
        lambda data: ~jnp.isclose(data[-1], 1, rtol=eps),
        body_fun,
        (left, right, jnp.sum(omd_w(midpoint(left, right), losses, lr))),
    )
    middle = midpoint(left, right)
    w = omd_w(middle, losses, lr)
    return middle, w


@jit
def omd_newton(
    x: Array,
    losses: Array,
    lr: Array,
    eps: float | Array = 1e-12,
) -> tuple[Array, Array]:
    """Newton's method for the weights, algorithm 2 of [11]."""
    Data: TypeAlias = tuple[Array, Array]  # type: ignore

    def body_fun(data: Data) -> Data:
        """Update step."""
        x, w = data
        w_sum = jnp.sum(w)
        x -= (w_sum - 1) / (lr * jnp.sum(jnp.sqrt(w**3)))
        w = omd_w(x, losses, lr)
        p_sum = jnp.sum(w)
        return lax.cond(
            # not making progress, switch to safe binary search
            ~jnp.isfinite(p_sum) | (jnp.abs(p_sum - 1) > jnp.abs(w_sum - 1)),
            lambda args: omd_monotone(*args),
            lambda _: (x, w),
            (losses, lr, eps),
        )

    return lax.while_loop(
        lambda data: ~jnp.isclose(jnp.sum(data[-1]), 1, rtol=eps),
        body_fun,
        (x, omd_w(x, losses, lr)),
    )


@jit
def loss_estimator(loss: Loss, w_i: Array, lr: Array, rv: bool) -> Array:
    """
    Compute importance-weighted (IW) or reduced-variance (RV) loss estimators.

    These estimators are an unbiased estimate for the loss.
    """
    b = 1 / 2 * (w_i >= jnp.square(lr))
    return jnp.where(
        rv,
        (loss - b) / w_i + b,
        loss / w_i,
    )


@partial(jit, static_argnums=(1, 2, 4))
def tsallis_inf(
    rng: KeyArray, K: int, draw: Draw, state: S, T: int, rv: bool = True
) -> tuple[Array, S]:
    """The Tsallis-INF multi-armed bandit algorithm, algorithm 1 of [11]."""
    losses = jnp.zeros(K)
    # warm start
    x = -jnp.sqrt(K)
    Data: TypeAlias = tuple[Array, S, KeyArray, Array]  # type: ignore
    data = (losses, state, rng, x)

    def body_fun(t: int, data: Data) -> Data:
        """Inner Online Mirror Descent (OMD) loop."""
        losses, state, rng, x = data
        rng, subkey = random.split(rng, num=2)
        # sample arm from distribution
        lr = learning_rate(t, rv)
        x, w = omd_newton(x, losses, lr)
        arm = random.choice(subkey, K, p=w)
        # observe outcome
        loss, state = draw(state, arm)
        # update cumulative losses with unbiased loss
        unbiased_loss = loss_estimator(loss, w[arm], lr, rv)
        losses = losses.at[arm].add(unbiased_loss)
        return losses, state, rng, x

    losses, state, *_ = lax.fori_loop(1, T + 1, body_fun, data)
    return losses, state
