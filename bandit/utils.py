"""
Helper library for shared types and utilites.
"""
from bisect import bisect_left
from typing import TYPE_CHECKING, Callable, Concatenate, ParamSpec, TypeVar

import jax
import jax.numpy as jnp
from equinox.nn import State
from jax import Array

KeyArray = Array
# multi-armed bandit arm
Arm = int | jnp.integer | Array
# loss from pulling an arm
Loss = float | jnp.floating | Array
# general state (like equinox's State)
S = TypeVar("S")
# function that takes an arm and returns the reward
Draw = Callable[[S, Arm], tuple[Loss, S]]
# result of a duel between a pair of arms
Win = bool | jnp.bool_ | Array
# function that takes two arms and returns whether the first one won
Duel = Callable[[S, Arm, Arm], tuple[Win, S]]
# history of queries by a bandit algorithm
History = Array
# bandit algorithm
BanditAlgorithm = Callable[[KeyArray, int, Duel, S, int], tuple[Arm, S]]
# preference matrix
Preferences = Array


jit = jax.jit  # type: ignore

if TYPE_CHECKING:
    P1 = ParamSpec("P1")
    P2 = ParamSpec("P2")
    T2 = TypeVar("T2")
    Fun = Callable[P2, T2]

    def wrap(
        _: Callable[Concatenate[Callable, P1], Callable]
    ) -> Callable[[Callable], Callable[Concatenate[Fun, P1], Fun]]:
        """Workaround LSP showing JitWrapped on hover."""
        ...

    @wrap(jax.jit)
    def jit():
        ...


def clone_state(state: State) -> State:
    """Clone the state."""
    leaves, treedef = jax.tree_util.tree_flatten(state)
    return jax.tree_util.tree_unflatten(treedef, leaves)


def index_dtype(x: Array, unsigned: bool = True):
    """Return the smallest integer datatype that can represent indices in x."""
    max_value = lambda dtype: jnp.iinfo(dtype).max  # noqa: E731
    dtypes = sorted(
        [jnp.uint8, jnp.uint16, jnp.uint32, jnp.uint64]
        if unsigned
        else [jnp.int8, jnp.int16, jnp.int32, jnp.int64],
        key=max_value,
    )
    sizes = list(map(max_value, dtypes))
    return dtypes[bisect_left(sizes, x.shape[0] - 1)]


def validate_preferences(p: Preferences) -> bool:
    """Validate the given preference matrix."""
    assert ((0 <= p) & (p <= 1)).all(), "entries must be valid probabilities"
    # this is technically redundant
    assert jnp.allclose(
        jnp.diagonal(p), 0.5
    ), "playing an arm against itself must tie"
    assert jnp.allclose(p + p.T, 1), "probabilities should be skew-symmetric"
    return True


@jit
def d_kl(p: Array | float, q: Array | float) -> Array:
    """Kullback-Leibler (KL) divergence between Bernoulli distributions."""
    # 0 log 0 := 0
    return jnp.where(
        p == 0,
        -jnp.log1p(-q),
        jnp.where(
            p == 1,
            -jnp.log(q),
            (
                p * (jnp.log(p) - jnp.log(q))
                + (1 - p) * (jnp.log1p(-p) - jnp.log1p(-q))
            ),
        ),
    )


# Copeland functions (see [9, 4, 6])


@jit
def copeland_scores(p: Preferences, normalize: bool = False) -> Array:
    """
    Compute the (normalized) Copeland scores for each arm.

    The Copeland score for a given arm is the number of
    arms beaten by it (wins with probability > 1/2).
    """
    scores = jnp.sum(p > 0.5, axis=1)
    return jnp.where(normalize, scores / (p.shape[1] - 1), scores)


@jit
def copeland_winners(p: Preferences) -> Array:
    """
    Return whether each arm is a Copeland winner.

    A Copeland winner is defined as an arm with the highest Copeland score.
    """
    scores = copeland_scores(p)
    return scores == jnp.max(scores)


@jit
def copeland_winner(p: Preferences) -> Arm:
    """Return an arbitrary Copeland winner if there are multiple."""
    return jnp.argmax(copeland_scores(p))


@jit
def copeland_regret(p: Preferences, history: History) -> Array:
    """Cumulative regret w.r.t. to Copeland winners."""
    scores = copeland_scores(p, normalize=True)
    return history.shape[0] * jnp.max(scores) - jnp.sum(scores[history]) / 2


@jit
def copeland_winner_wins(wins: Array) -> Arm:
    """Return an arbitrary Copeland winner from a number of wins matrix."""
    nums = wins + wins.T
    # define 0/0 := 1/2
    p = jnp.where(nums != 0, wins / nums, 0.5)
    p = jnp.fill_diagonal(p, 0.5, inplace=False)
    return copeland_winner(p)


# Condorcet functions (see [7, 3, 5])


@jit
def condorcet_winner(p: Preferences) -> Arm:
    """
    Return the (necessarily unique) Condorcet winner.

    A Condorcet winner is defined as an arm which beats all other arms.
    """
    scores = copeland_scores(p)
    arm = jnp.argmax(scores)
    return jnp.where(scores[arm] == p.shape[0] - 1, arm, -1)


@jit
def condorcet_regret(p: Preferences, history: History) -> Array:
    """Cumulative regret w.r.t. a Condorcet winner."""
    best = condorcet_winner(p)
    return jnp.sum(p[best, history[:, 0]] + p[best, history[:, 1]] - 1) / 2
