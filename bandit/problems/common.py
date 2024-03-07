"""
Common utilities between all problem environments.
"""
from abc import abstractmethod
from typing import TypeAlias

import jax.numpy as jnp
from equinox import Module, filter_jit
from equinox.nn import State, StateIndex
from jax import Array, random

from ..utils import (
    Arm,
    BanditAlgorithm,
    History,
    KeyArray,
    Loss,
    Preferences,
    S,
    Win,
    condorcet_winner,
    copeland_winners,
    index_dtype,
    jit,
    validate_preferences,
)


class Problem(Module):
    """An instance of a general dueling bandit problem."""

    K: int
    index: StateIndex

    @abstractmethod
    def duel(self, state: S, arm1: Arm, arm2: Arm) -> tuple[Array, S]:
        """Whether arm1 beats arm2."""

    @abstractmethod
    def preference_matrix(self, state: State) -> Preferences:
        """Return the preference matrix for the problem."""

    @staticmethod
    @abstractmethod
    def regret_function(p: Preferences, history: History) -> Loss:
        """Cumulative regret for the given prefrence matrix and history."""

    @filter_jit
    def regret(self, state: State, history: History) -> Loss:
        """Return the regret for a particular history."""
        return self.regret_function(self.preference_matrix(state), history)

    @abstractmethod
    def is_winner(self, state: State, arm: Arm) -> Win:
        """Whether the arm is a considered a winner."""

    @abstractmethod
    def shuffle(self, state: State) -> State:
        """Shuffle the internal state to prevent spurious correlations."""


# common implementation patterns


@filter_jit
def duel_matrix(
    self, state: State, arm1: Arm, arm2: Arm
) -> tuple[Array, State]:
    """A duel for problems defined by explicit preference matrices."""
    params = state.get(self.index)
    p = params["p"]
    rng, subkey = random.split(params["rng"])
    state = state.set(self.index, {"rng": rng, "p": p})
    return random.bernoulli(subkey, p[arm1, arm2]), state


@filter_jit
def preference_matrix_get(self, state: State) -> Preferences:
    """Return the preference matrix for the problem."""
    return state.get(self.index)["p"]


@jit
def permute_matrix(rng: KeyArray, x: Array) -> Array:
    """Permute a matrix."""
    permute = random.permutation(rng, x.shape[0])
    return x[jnp.ix_(permute, permute)]


@filter_jit
def shuffle_matrix(self, state: State) -> State:
    """Shuffle the internal state to prevent spurious correlations."""
    params = state.get(self.index)
    p = params["p"]
    rng, subkey = random.split(params["rng"])
    p = permute_matrix(subkey, p)
    state = state.set(self.index, {"rng": rng, "p": p})
    return state


@filter_jit
def is_copeland_winner(self, state: State, arm: Arm) -> Array:
    """Whether the arm is a considered a Copeland winner."""
    return copeland_winners(self.preference_matrix(state))[arm]


@filter_jit
def is_condorcet_winner(self, state: State, arm: Arm) -> Win:
    """Whether the arm is a considered a Condorcet winner."""
    return arm == condorcet_winner(self.preference_matrix(state))


# useful methods


def run_problem(
    rng: KeyArray,
    problem: Problem,
    state: State,
    bandit_algorithm: BanditAlgorithm,
    T: int,
) -> tuple[Arm, History]:
    """Run the bandit algorithm on the problem."""
    assert validate_preferences(
        problem.preference_matrix(state)
    ), "invalid problem setup"

    int_dtype = index_dtype(jnp.arange(problem.K))
    history = jnp.zeros((T, 2), dtype=int_dtype)
    Data: TypeAlias = tuple[History, int, State]  # type: ignore
    data = (history, 0, state)

    @jit
    def duel(data: Data, arm1: Arm, arm2: Arm) -> tuple[Array, Data]:
        """Wrap the duel of the problem to track queries."""
        history, t, state = data
        history = history.at[t, 0].set(int_dtype(arm1))
        history = history.at[t, 1].set(int_dtype(arm2))
        outcome, state = problem.duel(state, arm1, arm2)
        return outcome, (history, t + 1, state)

    result, (history, t, _) = bandit_algorithm(rng, problem.K, duel, data, T)
    assert t == T, f"algorithm used {t} comparisions for time horizon {T}"
    return result, history
