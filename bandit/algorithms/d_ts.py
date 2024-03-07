from functools import partial
from typing import TypeAlias

import jax.numpy as jnp
from jax import Array, lax, random, vmap

from .. import utils
from ..utils import Arm, Duel, KeyArray, S, jit

# Double Thompson Sampling ([6])


@jit
def d_ts_plus(theta: Array) -> Arm:
    """Return the arm that minimizes regret (corollary 1 of [6])."""
    scores = utils.copeland_scores(theta)
    score = jnp.max(scores)
    winners = utils.copeland_winners(theta)

    def regret(i: Arm) -> Array:
        """Expected regret of using i as the first candidate."""
        loss = score - (scores[i] + scores[:]) / 2
        return jnp.where(
            theta[i, :] != 0.5, loss / utils.d_kl(theta[i, :], 0.5), 0
        ).sum()

    regrets = vmap(regret)(jnp.arange(scores.shape[0]))
    return jnp.argmin(jnp.where(winners, regrets, jnp.inf))


@partial(jit, static_argnums=(1, 2, 4, 6))
def d_ts(
    rng: KeyArray,
    K: int,
    duel: Duel,
    state: S,
    T: int,
    # [6] proves with alpha = 0.5 (theorem 1) and uses alpha = 0.51 (section 5)
    # [5] recommends alpha = 0.6 in section 7
    alpha: float = 0.51,
    plus: bool = True,
) -> tuple[Arm, S]:
    """The Double Thompson Sampling (D-TS) algorithm 1 of [6]."""
    B = jnp.zeros((K, K))

    Data: TypeAlias = tuple[Array, S, KeyArray]  # type: ignore
    data = (B, state, rng)

    def body_fun(t: int, data: Data) -> Data:
        """Inner loop."""
        B, state, rng = data
        rng, subkey1, subkey2, subkey3 = random.split(rng, num=4)
        # phase 1: choose the first candidate
        total = B + B.T
        std = jnp.sqrt(alpha * jnp.log(t) / total)
        # x/0 := 1 for all x
        U = jnp.where(total != 0, B / total + std, 2)
        L = jnp.where(total != 0, B / total - std, 0)
        # u_ii = l_ii = 1/2
        U = jnp.fill_diagonal(U, 0.5, inplace=False)
        L = jnp.fill_diagonal(L, 0.5, inplace=False)
        # upper bound of the normalized Copeland score
        C = utils.copeland_winners(U)
        # sample from beta distribution
        lower = jnp.tril_indices(n=K, k=-1)
        theta = jnp.zeros((K, K))
        theta = theta.at[lower].set(
            random.beta(subkey1, B[lower] + 1, B.T[lower] + 1)
        )
        theta = theta + (1 - theta.T)
        theta = theta.at[lower].add(-1)
        theta = jnp.fill_diagonal(theta, 0.5, inplace=False)
        # choose only from C to eliminate non-winner arms; break ties randomly
        if not plus:
            scores = jnp.where(C, utils.copeland_scores(theta), -1)
            perm = random.permutation(subkey2, K)
            arm1 = perm[jnp.argmax(scores[perm])]
        else:
            arm1 = d_ts_plus(theta)
        # phase 2: choose the second candidate
        theta2 = random.beta(subkey3, B[:, arm1] + 1, B[arm1, :] + 1)
        theta2 = theta2.at[arm1].set(0.5)
        # choosing only from uncertain pairs
        arm2 = jnp.argmax(jnp.where(L[:, arm1] <= 0.5, theta2, -1))
        # compare the pair, observe the outcome, and update B
        outcome, state = duel(state, arm1, arm2)
        B = lax.cond(
            outcome,
            lambda B, winner, loser: B.at[winner, loser].add(1),
            lambda B, loser, winner: B.at[winner, loser].add(1),
            *(B, arm1, arm2),
        )
        return B, state, rng

    B, state, *_ = lax.fori_loop(1, T + 1, body_fun, data)
    return utils.copeland_winner_wins(B), state
