import jax.numpy as jnp
from equinox.nn import StateIndex
from jax import random

from ..utils import KeyArray, condorcet_regret
from .common import (
    Problem,
    duel_matrix,
    is_condorcet_winner,
    preference_matrix_get,
    shuffle_matrix,
)


class RankingProblem(Problem):
    """
    The ranking setup in https://github.com/jxiong21029/LuxS2/issues/1.

    Suppose we have agents with hidden "ratings" r_1, ..., r_n and player A
    wins a match against player B with probability 1/(1 + exp(r_B - r_A)).

    Note that this problem satisfies a total ordering on the arms,
    stochastic transitivity, and the stochastic triangle inequality.
    """

    def __init__(
        self, rng: KeyArray, K: int, rating_max: float, rating_min: float = 0
    ) -> None:
        """Intialize the state of the problem."""
        self.K = K

        rng, subkey = random.split(rng)
        ratings = random.uniform(
            subkey, (K,), minval=rating_min, maxval=rating_max
        )
        p = jnp.reciprocal(1 + jnp.exp(ratings - ratings[:, jnp.newaxis]))

        self.index = StateIndex({"rng": rng, "p": p})

    duel_function = staticmethod(duel_matrix)

    preference_matrix = preference_matrix_get

    regret_function = staticmethod(condorcet_regret)

    is_winner = is_condorcet_winner

    shuffle = shuffle_matrix
