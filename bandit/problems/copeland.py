from functools import partial
from typing import TypeAlias

import jax.numpy as jnp
from equinox import filter_jit
from equinox.nn import StateIndex, make_with_state
from jax import Array, lax, random

from ..utils import KeyArray, copeland_regret, copeland_winners
from .common import (
    Problem,
    duel_matrix,
    is_copeland_winner,
    jit,
    preference_matrix_get,
    shuffle_matrix,
)
from .random import RandomProblem


@filter_jit
def sample_preference(problemclass: type[Problem], *args, **kwargs) -> Array:
    """Sample a preference matrix."""
    problem, state = make_with_state(problemclass)(*args, **kwargs)
    return problem.preference_matrix(state)


@partial(jit, static_argnums=(0, 2))
def rejection_sample(
    problemclass: type[Problem], rng: KeyArray, K: int
) -> Array:
    """Sample a preference matrix with a unique Copeland winner."""
    get_matrix = partial(sample_preference, problemclass, K=K)
    p = get_matrix(rng)

    Data: TypeAlias = tuple[Array, KeyArray]  # type: ignore
    data = (p, rng)

    def body_fun(data: Data) -> Data:
        """Brute-force sample until satisfied"""
        _, rng = data
        rng, subkey = random.split(rng)
        p = get_matrix(subkey)
        return p, rng

    p, *_ = lax.while_loop(
        lambda data: jnp.sum(copeland_winners(data[0])) > 1, body_fun, data
    )
    return p


class CopelandProblem(Problem):
    """A problem with a unique Copeland winner."""

    def __init__(self, rng: KeyArray, K: int) -> None:
        """Intialize the state of the problem."""
        self.K = K
        rng, subkey = random.split(rng)
        p = rejection_sample(RandomProblem, subkey, K)
        self.index = StateIndex({"rng": rng, "p": p})

    duel_function = staticmethod(duel_matrix)

    preference_matrix = preference_matrix_get

    regret_function = staticmethod(copeland_regret)

    is_winner = is_copeland_winner

    shuffle = shuffle_matrix
