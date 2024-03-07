from functools import partial
from typing import TypeAlias

import jax.numpy as jnp
from jax import Array, lax

from .. import utils
from ..utils import Arm, Duel, KeyArray, S, jit


@partial(jit, static_argnums=(1, 2, 4))
def naive(
    _: KeyArray,
    K: int,
    duel: Duel,
    state: S,
    T: int,
) -> tuple[Arm, S]:
    """Plays every pair of arms against each other and picks the winner."""
    wins = jnp.zeros((K, K))

    Data: TypeAlias = tuple[Array, S]  # type: ignore
    data = (wins, state)

    def body_fun(_: int, data: Data) -> Data:
        """Inner loop."""
        # pick the pair that have been played the least
        wins, state = data
        nums = wins + wins.T
        nums = jnp.fill_diagonal(nums, T + 1, inplace=False)
        index = jnp.argmin(nums)
        arm1, arm2 = jnp.unravel_index(index, nums.shape)
        # play them against each other and update statistics
        outcome, state = duel(state, arm1, arm2)
        wins = lax.cond(
            outcome,
            lambda wins, winner, loser: wins.at[winner, loser].add(1),
            lambda wins, loser, winner: wins.at[winner, loser].add(1),
            *(wins, arm1, arm2),
        )
        return wins, state

    wins, state = lax.fori_loop(0, T, body_fun, data)
    return utils.copeland_winner_wins(wins), state
