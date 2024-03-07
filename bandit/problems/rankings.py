import jax.numpy as jnp
from equinox import filter_jit
from equinox.nn import State, StateIndex
from jax import Array, random

from ..utils import Arm, KeyArray, Preferences, condorcet_regret
from .common import Problem, is_condorcet_winner


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
        self.index = StateIndex((rng, ratings))

    @staticmethod
    @filter_jit
    def duel_function(
        index: StateIndex, state: State, arm1: Arm, arm2: Arm
    ) -> tuple[Array, State]:
        """Whether arm1 beats arm2."""
        rng, ratings = state.get(index)
        rng, subkey = random.split(rng)
        p = jnp.reciprocal(1 + jnp.exp(ratings[arm2] - ratings[arm1]))
        state = state.set(index, (rng, ratings))
        return random.bernoulli(subkey, p), state

    @filter_jit
    def preference_matrix(self, state: State) -> Preferences:
        """Return the preference matrix for the problem."""
        _, ratings = state.get(self.index)
        return jnp.reciprocal(1 + jnp.exp(ratings - ratings[:, jnp.newaxis]))

    regret_function = staticmethod(condorcet_regret)

    is_winner = is_condorcet_winner

    @filter_jit
    def shuffle(self, state: State) -> State:
        """Shuffle the internal state to prevent spurious correlations."""
        rng, ratings = state.get(self.index)
        rng, subkey = random.split(rng)
        ratings = random.permutation(subkey, ratings)
        return state.set(self.index, (rng, ratings))
