from functools import partial
from typing import TypeAlias

import jax.numpy as jnp
from jax import Array, lax, random

from ..utils import Arm, Duel, KeyArray, S, jit
from .tsallis_inf import learning_rate, loss_estimator, omd_newton

# Versatile Dueling Bandits (VDB) ([8])


@partial(jit, static_argnums=(1, 2, 4))
def vdb_ind(
    rng: KeyArray,
    K: int,
    duel: Duel,
    state: S,
    T: int,
    rv: bool = True,
) -> tuple[Arm, S]:
    """The Versatile-DB (VDB) algorithm 3 of [8]."""
    losses = jnp.zeros((2, K))
    x = [-jnp.sqrt(K)] * 2

    Data: TypeAlias = tuple[Array, S, KeyArray, Array]  # type: ignore
    data = (losses, state, rng, x)

    def body_fun(t: int, data: Data) -> Data:
        """Inner loop."""
        losses, state, rng, x = data
        lr = learning_rate(t, rv)
        w = [jnp.empty(0)] * 2
        arms = [jnp.empty(0)] * 2
        rng, *subkeys = random.split(rng, num=3)
        for i in range(2):
            # choose from distribution (update x for warm start)
            x[i], w[i] = omd_newton(x[i], losses[i], lr)
            # sample arms
            arms[i] = random.choice(subkeys[i], K, p=w[i])
        # observe outcome
        outcome, state = duel(state, arms[0], arms[1])
        outcomes = jnp.array([1 - outcome, outcome])
        for i in range(2):
            # construct unbiased estimator by IW or RV sampling
            loss = loss_estimator(outcomes[i], w[i][arms[i]], lr, rv)
            # update losses
            losses = losses.at[i, arms[i]].add(loss)
        return losses, state, rng, x

    losses, state, *_ = lax.fori_loop(1, T + 1, body_fun, data)
    return jnp.argmin(jnp.sum(losses, axis=0)), state


@partial(jit, static_argnums=(1, 2, 4))
def vdb(
    rng: KeyArray,
    K: int,
    duel: Duel,
    state: S,
    T: int,
    rv: bool = True,
) -> tuple[Arm, S]:
    """The Versatile-DB (VDB) algorithm 3 of [8], remark 2."""
    losses = jnp.zeros(K)
    x = -jnp.sqrt(K)

    Data: TypeAlias = tuple[Array, S, KeyArray, Array]  # type: ignore
    data = (losses, state, rng, x)

    def body_fun(t: int, data: Data) -> Data:
        """Inner loop."""
        losses, state, rng, x = data
        lr = learning_rate(t, rv)
        # choose from distribution (update x for warm start)
        x, w = omd_newton(x, losses, lr)
        # sample arms
        rng, *subkeys = random.split(rng, num=3)
        arms = [random.choice(subkeys[i], K, p=w) for i in range(2)]
        # observe outcome
        outcome, state = duel(state, arms[0], arms[1])
        outcomes = jnp.array([1 - outcome, outcome])
        for i in range(2):
            # construct unbiased estimator by IW or RV sampling
            loss = loss_estimator(outcomes[i], w[arms[i]], lr, rv)
            # update losses
            losses = losses.at[arms[i]].add(loss / 2)
        return losses, state, rng, x

    losses, state, *_ = lax.fori_loop(1, T + 1, body_fun, data)
    return jnp.argmin(losses), state
