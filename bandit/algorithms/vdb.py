from functools import partial
from typing import TypeAlias

import jax.numpy as jnp
from equinox.nn import State
from jax import Array, lax, random

from ..utils import Arm, Duel, History, KeyArray, index_dtype, jit
from .tsallis_inf import learning_rate, loss_estimator, omd_newton

# Versatile Dueling Bandits (VDB) methods ([5])


@partial(jit, static_argnums=(1, 2, 4))
def vdb_ind(
    rng: KeyArray,
    K: int,
    duel: Duel,
    state: State,
    T: int,
    rv: bool = True,
) -> tuple[Arm, History]:
    """The Versatile-DB (VDB) algorithm 3 of [5]."""
    losses = jnp.zeros((2, K))
    int_dtype = index_dtype(jnp.arange(K))
    history = jnp.zeros((T, 2), dtype=int_dtype)
    x = [-jnp.sqrt(K)] * 2

    Data: TypeAlias = tuple[  # type: ignore
        Array, History, KeyArray, State, Array
    ]
    data = (losses, history, rng, state, x)

    def body_fun(t: int, data: Data) -> Data:
        """Inner loop."""
        losses, history, rng, state, x = data
        lr = learning_rate(t, rv)
        w = [jnp.zeros(0)] * 2
        arms = jnp.zeros(2, dtype=int_dtype)
        rng, *subkeys = random.split(rng, num=3)
        for i in range(2):
            # choose from distribution (update x for warm start)
            x[i], w[i] = omd_newton(x[i], losses[i], lr)
            # sample arms
            arms = arms.at[i].set(
                int_dtype(random.choice(subkeys[i], K, p=w[i]))
            )
            history = history.at[t - 1, i].set(arms[i])
        # observe outcome
        outcome, state = duel(state, arms[0], arms[1])
        outcomes = jnp.array([1 - outcome, outcome])
        for i in range(2):
            # construct unbiased estimator by IW or RV sampling
            loss = loss_estimator(outcomes[i], w[i][arms[i]], lr, rv)
            # update losses
            losses = losses.at[i, arms[i]].add(loss)
        return losses, history, rng, state, x

    losses, history, *_ = lax.fori_loop(1, T + 1, body_fun, data)
    return jnp.argmin(jnp.sum(losses, axis=0)), history


@partial(jit, static_argnums=(1, 2, 4))
def vdb(
    rng: KeyArray,
    K: int,
    duel: Duel,
    state: State,
    T: int,
    rv: bool = True,
) -> tuple[Arm, History]:
    """The Versatile-DB (VDB) algorithm 3 of [5], remark 2."""
    losses = jnp.zeros(K)
    int_dtype = index_dtype(jnp.arange(K))
    history = jnp.zeros((T, 2), dtype=int_dtype)
    x = -jnp.sqrt(K)

    Data: TypeAlias = tuple[  # type: ignore
        Array, History, KeyArray, State, Array
    ]
    data = (losses, history, rng, state, x)

    def body_fun(t: int, data: Data) -> Data:
        """Inner loop."""
        losses, history, rng, state, x = data
        lr = learning_rate(t, rv)
        # choose from distribution (update x for warm start)
        x, w = omd_newton(x, losses, lr)
        # sample arms
        rng, *subkeys = random.split(rng, num=3)
        arms = jnp.zeros(2, dtype=int_dtype)
        for i in range(2):
            arms = arms.at[i].set(int_dtype(random.choice(subkeys[i], K, p=w)))
            history = history.at[t - 1, i].set(arms[i])
        # observe outcome
        outcome, state = duel(state, arms[0], arms[1])
        outcomes = jnp.array([1 - outcome, outcome])
        for i in range(2):
            # construct unbiased estimator by IW or RV sampling
            loss = loss_estimator(outcomes[i], w[arms[i]], lr, rv)
            # update losses
            losses = losses.at[arms[i]].add(loss / 2)
        return losses, history, rng, state, x

    losses, history, *_ = lax.fori_loop(1, T + 1, body_fun, data)
    return jnp.argmin(losses), history
