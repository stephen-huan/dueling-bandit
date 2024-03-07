import jax.numpy as jnp
from equinox.nn import State, StateIndex
from jax import random

from ..utils import KeyArray, condorcet_regret
from .common import (
    Problem,
    duel_matrix,
    is_condorcet_winner,
    preference_matrix_get,
    shuffle_matrix,
    permute_matrix
)


class CondorcetProblem(Problem):
    """A problem extended with a guaranteed Condorcet winner."""

    def __init__(self, rng: KeyArray, base: tuple[Problem, State]) -> None:
        """Intialize the state of the problem."""
        problem, state = base
        self.K = problem.K + 1
        rng, subkey = random.split(rng)

        p = problem.preference_matrix(state)
        rest = random.uniform(subkey, (problem.K, 1), minval=0.5, maxval=1)
        p = jnp.block(
            [
                [0.5, rest.T],
                [1 - rest, p],
            ]
        )
        # avoid Condorcet winner always placed first
        p = permute_matrix(subkey, p)
        self.index = StateIndex({"rng": rng, "p": p})

    duel = duel_matrix

    preference_matrix = preference_matrix_get

    regret_function = staticmethod(condorcet_regret)

    is_winner = is_condorcet_winner

    shuffle = shuffle_matrix
