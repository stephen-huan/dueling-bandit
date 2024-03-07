import jax.numpy as jnp
from equinox.nn import StateIndex
from jax import random

from ..utils import KeyArray, copeland_regret
from .common import (
    Problem,
    duel_matrix,
    is_copeland_winner,
    preference_matrix_get,
    shuffle_matrix,
)


class RandomProblem(Problem):
    """A completely random problem."""

    def __init__(self, rng: KeyArray, K: int) -> None:
        """Intialize the state of the problem."""
        self.K = K

        p = jnp.zeros((K, K))
        lower = jnp.tril_indices(n=K, k=-1)
        rng, subkey = random.split(rng)
        p = p.at[lower].set(random.uniform(subkey, (lower[0].shape[0],)))
        p = p + (1 - p.T)
        p = p.at[lower].add(-1)
        p = jnp.fill_diagonal(p, 0.5, inplace=False)

        self.index = StateIndex({"rng": rng, "p": p})

    duel = duel_matrix

    preference_matrix = preference_matrix_get

    regret_function = staticmethod(copeland_regret)

    is_winner = is_copeland_winner

    shuffle = shuffle_matrix
