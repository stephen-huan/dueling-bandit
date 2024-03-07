from functools import partial
from typing import TypeAlias

import jax.numpy as jnp
from equinox.nn import State
from jax import Array, lax

from .. import utils
from ..utils import Arm, Duel, History, KeyArray, index_dtype, jit


@partial(jit, static_argnums=(1, 2, 4))
def naive(
    _rng: KeyArray,
    K: int,
    duel: Duel,
    state: State,
    T: int,
) -> tuple[Arm, History]:
    """Plays every pair of arms against each other and picks the winner."""
    _rng  # pyright: ignore [reportUnusedExpression]
    wins = jnp.zeros((K, K))
    int_dtype = index_dtype(jnp.arange(K))
    history = jnp.zeros((T, 2), dtype=int_dtype)

    Data: TypeAlias = tuple[Array, History, State]  # type: ignore
    data = (wins, history, state)

    def body_fun(t: int, data: Data) -> Data:
        """Inner loop."""
        # pick the pair that have been played the least
        wins, history, state = data
        nums = wins + wins.T
        nums = jnp.fill_diagonal(nums, T + 1, inplace=False)
        index = jnp.argmin(nums)
        arm1, arm2 = jnp.unravel_index(index, nums.shape)
        history = history.at[t, 0].set(int_dtype(arm1))
        history = history.at[t, 1].set(int_dtype(arm2))
        # play them against each other and update statistics
        outcome, state = duel(state, arm1, arm2)
        wins = lax.cond(
            outcome,
            lambda wins, winner, loser: wins.at[winner, loser].add(1),
            lambda wins, loser, winner: wins.at[winner, loser].add(1),
            *(wins, arm1, arm2),
        )
        return wins, history, state

    wins, history, *_ = lax.fori_loop(0, T, body_fun, data)
    return utils.copeland_winner_wins(wins), history
