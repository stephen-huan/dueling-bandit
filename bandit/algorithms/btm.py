from functools import partial
from typing import Callable, TypeAlias

import jax.numpy as jnp
from jax import Array, lax, random

from ..utils import (
    Arm,
    Duel,
    KeyArray,
    S,
    argmax_masked,
    argmin_masked,
    jit,
    max_masked,
    min_masked,
)

# Beat the Mean Bandit ([10])


@jit
def statistics(wins: Array) -> tuple[Array, Array]:
    """Return statistics from the wins."""
    winsb = jnp.sum(wins, axis=1)
    nums = wins + wins.T
    numsb = jnp.sum(nums, axis=1)
    probb = jnp.where(numsb != 0, winsb / numsb, 0.5)
    return numsb, probb


@partial(jit, static_argnums=(1, 2, 4, 5))
def btm(
    rng: KeyArray,
    K: int,
    duel: Duel,
    state: S,
    T: int,
    c: Callable[[Array], Array],
    N: float = jnp.inf,
) -> tuple[Arm, S, int]:
    """The Beat the Mean Bandit (BTM) algorithm 1 of [10]."""
    W = jnp.ones(K, dtype=jnp.bool_)
    wins = jnp.zeros((K, K))
    t = 0

    Data: TypeAlias = tuple[Array, Array, int, S, KeyArray]  # type: ignore
    data = (W, wins, t, state, rng)

    def cond_fun(data: Data) -> Array:
        """Whether to stop the loop."""
        W, wins, t, *_ = data
        numsb, _ = statistics(wins)
        n_min = min_masked(numsb, W)
        return (jnp.sum(W) > 1) & (t < T) & (n_min < N)

    def body_fun(data: Data) -> Data:
        """Inner loop."""
        W, wins, t, state, rng = data
        rng, subkey1, subkey2, subkey3 = random.split(rng, num=4)
        # break ties randomly
        numsb, _ = statistics(wins)
        b = argmin_masked(subkey1, numsb, W)
        # select b' randomly, compare b and b'
        bp = random.choice(subkey2, K, p=W / jnp.sum(W))
        outcome, state = duel(state, b, bp)
        # update statistics
        wins = lax.cond(
            outcome,
            lambda wins, winner, loser: wins.at[winner, loser].add(1),
            lambda wins, loser, winner: wins.at[winner, loser].add(1),
            *(wins, b, bp),
        )
        t += 1
        numsb, probb = statistics(wins)
        n_min = min_masked(numsb, W)
        c_max = jnp.where(n_min != 0, c(n_min), 1)
        W, wins = lax.cond(
            min_masked(probb, W) + c_max <= max_masked(probb, W) - c_max,
            lambda W, wins: (
                # update working set
                W.at[bp := argmin_masked(subkey3, probb, W)].set(False),
                # remove comparisons with b'
                wins.at[:, bp].set(0),
            ),
            lambda W, wins: (W, wins),
            *(W, wins),
        )
        return W, wins, t, state, rng

    W, wins, t, state, *_ = lax.while_loop(cond_fun, body_fun, data)
    _, probb = statistics(wins)
    return argmax_masked(rng, probb, W), state, t


@partial(jit, static_argnums=(1, 2, 4))
def btm_online(
    rng: KeyArray,
    K: int,
    duel: Duel,
    state: S,
    T: int,
    gamma: float = 1,
) -> tuple[Arm, S]:
    """The Beat the Mean (online) algorithm 2 of [10]."""
    delta = jnp.reciprocal(2 * T * K)

    def c(n: Array) -> Array:
        """Confidence bound for the online setting in equation (4)."""
        # tighter confidence bound when gamma = 1 in equation (9)
        coef = jnp.where(gamma > 1, 3 * gamma**2, 1)
        return coef * jnp.sqrt(-jnp.log(delta) / n)

    # explore
    b, state, t = btm(rng, K, duel, state, T, c)
    # exploit
    state = lax.fori_loop(
        0, T - t, lambda _, state: duel(state, b, b)[1], state
    )
    return b, state
