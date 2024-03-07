"""
Algorithms for the multi-armed bandit and dueling multi-armed bandit problems.

See the following references (not all of these are currently implemented):
[1] A. Agarwal, R. Ghuge, and V. Nagarajan, "An Asymptotically
    Optimal Batched Algorithm for the Dueling Bandit Problem."
    arXiv, Sep. 2022. Available: <https://arxiv.org/abs/2209.12108>
[2] A. Agarwal, R. Ghuge, and V. Nagarajan, "Batched Dueling Bandits."
    arXiv, Feb. 2022. Available: <https://arxiv.org/abs/2202.10660>
[3] J. Komiyama, J. Honda, H. Kashima, and H. Nakagawa, "Regret
    Lower Bound and Optimal Algorithm in Dueling Bandit Problem."
    arXiv, Jun. 2015. Available: <https://arxiv.org/abs/1506.02550>
[4] J. Komiyama, J. Honda, and H. Nakagawa, "Copeland Dueling Bandit Problem:
    Regret Lower Bound, Optimal Algorithm, and Computationally Efficient
    Algorithm." arXiv, May 2016. Available: <https://arxiv.org/abs/1605.01677>
[5] A. Saha and P. Gaillard, "Versatile Dueling Bandits:
    Best-of-both-World Analyses for Online Learning from Preferences."
    arXiv, Feb. 2022. Available: <https://arxiv.org/abs/2202.06694>
[6] H. Wu and X. Liu, "Double Thompson Sampling for Dueling Bandits."
    arXiv, Oct. 2016. Available: <https://arxiv.org/abs/1604.07101>
[7] Y. Yue and T. Joachims, "Beat the mean bandit," in
    *Proceedings of the 28th International Conference on International
    Conference on Machine Learning*, Jun. 2011, pp. 241–248.
[8] J. Zimmert and Y. Seldin, "Tsallis-INF: An Optimal
    Algorithm for Stochastic and Adversarial Bandits." arXiv,
    Mar. 2022. Available: <https://arxiv.org/abs/1807.07623>
[9] M. Zoghi, Z. Karnin, S. Whiteson, and M. de Rijke,
    "Copeland Dueling Bandits." arXiv, May 2015. doi:
    [10.48550/arXiv.1506.00312](https://doi.org/10.48550/arXiv.1506.00312).
"""
from typing import Callable

import numpy as np
from scipy.stats import beta

from .. import utils
from ..utils import Arm, Duel

# Copeland Dueling Bandits ([9])


def find_kl(
    p: float, dist: float, less: bool = True, eps: float = 1e-6
) -> float:
    """Find q farthest from p such that D_kl(p || q) <= bound."""
    left, right = (0, p) if less else (p, 1)
    while right - left > eps:
        q = (left + right) / 2
        within = utils.d_kl(p, q) <= dist
        if within and less or (not within and not less):
            right = q
        else:
            left = q
    return right if less else left


def kl_arm(
    K: int,
    reward: Callable[[Arm, int], tuple[bool, int]],
    T: int,
    delta: float,
    eps: float,
) -> tuple[Arm, int]:
    """Solve the multi-armed bandit problem by algorithm 4 of [9]."""
    rewards = {i: 0.0 for i in range(K)}
    intervals = {i: (0.0, 1.0) for i in range(K)}
    B = set(range(K))
    t = 2
    queries = 0
    left, right = map(tuple, zip(*intervals.values()))
    while max(right) == 1 or (1 - max(left)) / (1 - max(right)) > 1 + eps:
        for i in B:
            outcome, num = reward(i, T - queries)
            rewards[i] += outcome
            queries += num
            p = rewards[i] / t
            dist = (np.log(4 * t * K / delta) + 2 * np.log(np.log(t))) / t
            intervals[i] = (find_kl(p, dist, True), find_kl(p, dist, False))
            # ran out of queries, force-terminate
            if queries == T:
                return max(B, key=lambda i: intervals[i][0]), queries
        remove = {
            i for i in B if any(intervals[i][1] < intervals[j][0] for j in B)
        }
        B -= remove
        for i in remove:
            del rewards[i]
            del intervals[i]
        t += 1
        left, right = map(tuple, zip(*intervals.values()))
    return max(B, key=lambda i: intervals[i][0]), queries


def copeland_bandit(
    K: int,
    duel: Duel,
    T: int,
    delta: float,
    eps: float,
    rng: np.random.Generator = np.random.default_rng(),
) -> tuple[Arm, int]:
    """Solve the dueling bandit problem by algorithm 2 of [9]."""

    def reward(i: Arm, limit: int) -> tuple[bool, int]:
        """The random variable based on uniform sampling."""
        candidates = np.delete(np.arange(K), i)
        # pick j != i uniformly from the arms
        j = rng.choice(candidates)
        win = num = 0
        p = 1 / 2
        # query the pair until w.p. 1 - delta/K^2 that p_{ij} > 1/2
        while p < 1 - delta / (K * K) and num < limit:
            win += duel(i, j)
            num += 1
            # https://www.stat.cmu.edu/~larry/=sml/Bayes.pdf#page=5
            # assume uniform prior on p_{ij}, conjugate prior is Beta
            p = beta.cdf(0.5, win + 1, num - win + 1)
            p = max(p, 1 - p)  # type: ignore
        return win / num > 0.5, num

    return kl_arm(K, reward, T, delta, eps)


def scb(
    K: int,
    duel: Duel,
    T: int,
    rng: np.random.Generator = np.random.default_rng(),
) -> Arm:
    """The Scalable Copeland Bandits (SCB) algorithm 3 of [9]."""
    r = 1
    final = 0
    while T > 0:
        # t = 2 ** (2**r)
        t = 2**r
        arm, queries = copeland_bandit(
            K, duel, min(t, T), np.log(t) / t, 0, rng
        )
        for _ in range(min(t, T) - queries):
            final = arm
            duel(arm, arm)
        T -= t
        r += 1
    return final
