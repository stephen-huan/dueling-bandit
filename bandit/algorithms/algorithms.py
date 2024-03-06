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

# Beat the Mean Bandit ([7])


def argmin_tiebreak(
    scores: list,
    rng: np.random.Generator = np.random.default_rng(),
) -> Arm:
    """Return the argmin of the score array, breaking ties randomly."""
    candidates = np.arange(len(scores))[scores == np.min(scores)]
    return rng.choice(candidates)


def btm(
    K: int,
    duel: Duel,
    T: int,
    c: Callable[[int], float],
    N: float = float("inf"),
    rng: np.random.Generator = np.random.default_rng(),
) -> tuple[Arm, int]:
    """The Beat the Mean Bandit (BTM) algorithm 1 of [7]."""
    W = list(range(K))

    def get_pb(
        W: list[int],
        nums: dict[int, list[int]],
        wins: dict[int, list[int]],
    ) -> list[float]:
        """Get the empirical probabilities."""
        return [
            len(wins[b]) / len(nums[b]) if len(nums[b]) != 0 else 0.5
            for b in W
        ]

    def remove(x: list[int], b: int) -> list[int]:
        """Remove b from the list x."""
        return [v for v in x if v != b]

    nums = {b: [] for b in W}
    wins = {b: [] for b in W}
    n_best = 0
    c_best = 1
    t = 0
    while len(W) > 1 and t < T and n_best < N:
        # break ties randomly
        b = W[argmin_tiebreak([len(nums[b]) for b in W], rng)]
        # select b' randomly, compare b and b'
        bp = rng.choice(W)
        if duel(b, bp):
            wins[b].append(bp)
        nums[b].append(bp)
        t += 1
        n_best = min(map(len, nums.values()))
        c_best = c(n_best) if n_best != 0 else 1
        pb = get_pb(W, nums, wins)
        if np.min(pb) + c_best <= np.max(pb) - c_best:
            bp = W[argmin_tiebreak(pb, rng)]
            # remove comparisons with b'
            for b in W:
                wins[b] = remove(wins[b], bp)
                nums[b] = remove(nums[b], bp)
            del wins[bp]
            del nums[bp]
            # update working set
            W = remove(W, bp)
    pb = get_pb(W, nums, wins)
    return W[np.argmax(pb)], t


def btm_online(
    K: int,
    duel: Duel,
    T: int,
    gamma: float = 1,
    rng: np.random.Generator = np.random.default_rng(),
) -> Arm:
    """The Beat the Mean (online) algorithm 2 of [7]."""
    delta = 1 / (2 * T * K)
    if gamma != 1:

        def c(n: int) -> float:
            """Confidence bound for the online setting in equation (4)."""
            return 3 * gamma**2 * np.sqrt(-np.log(delta) / n)

    else:

        def c(n: int) -> float:
            """Tighter confidence bound when gamma = 1 in equation (9)."""
            return np.sqrt(-np.log(delta) / n)

    # explore
    b, t = btm(K, duel, T, c, rng=rng)
    # exploit
    for _ in range(T - t):
        duel(b, b)
    return b


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


# Double Thompson Sampling ([6])


def d_ts_plus(theta: np.ndarray) -> Arm:
    """Return the arm that minimizes regret (corollary 1 of [6])."""
    scores = utils.copeland_scores(theta)
    score = np.max(scores)
    winners = utils.copeland_winners(theta)

    def loss(i: Arm, j: Arm) -> float:
        """Loss of comparing arm i to arm j."""
        return score - (scores[i] + scores[j]) / 2

    def regret(i: Arm) -> float:
        """Expected regret of using i as the first candidate."""
        return sum(
            loss(i, j) / utils.d_kl(theta[i, j], 0.5)
            for j in range(theta.shape[0])
            if theta[i, j] != 0.5
        )

    return min(winners, key=regret)


def d_ts(
    K: int,
    duel: Duel,
    T: int,
    # [6] proves with alpha = 0.5 (theorem 1) and uses alpha = 0.51 (section 5)
    # [5] recommends alpha = 0.6 in section 7
    alpha: float = 0.51,
    plus: bool = True,
    rng: np.random.Generator = np.random.default_rng(),
) -> Arm:
    """The Double Thompson Sampling (D-TS) algorithm 1 of [6]."""
    B = np.zeros((K, K))
    for t in range(1, T + 1):
        # phase 1: choose the first candidate
        total = B + B.T
        mask = total != 0
        U = np.zeros((K, K))
        U[mask] = B[mask] / total[mask] + np.sqrt(
            alpha * np.log(t) / total[mask]
        )
        # x/0 := 1 for all x
        U[~mask] = 2
        L = np.zeros((K, K))
        L[mask] = B[mask] / total[mask] - np.sqrt(
            alpha * np.log(t) / total[mask]
        )
        L[~mask] = 0
        # u_ii = l_ii = 1/2
        np.fill_diagonal(U, 0.5)
        np.fill_diagonal(L, 0.5)
        # upper bound of the normalized Copeland score
        C = utils.copeland_winners(U)
        # sample from beta distribution
        lower = np.tril_indices(n=K, k=-1)
        theta = np.zeros((K, K))
        theta[lower] = rng.beta(B[lower] + 1, B.T[lower] + 1)
        theta = theta + (1 - theta.T)
        theta[lower] -= 1
        np.fill_diagonal(theta, 0.5)
        # choose only from C to eliminate non-winner arms; break ties randomly
        if not plus:
            rng.shuffle(C)
            arm1 = C[np.argmax(utils.copeland_scores(theta)[C])]
        else:
            arm1 = d_ts_plus(theta)
        # phase 2: choose the second candidate
        theta2 = rng.beta(B[:, arm1] + 1, B[arm1, :] + 1)
        theta2[arm1] = 0.5
        # choosing only from uncertain pairs
        C2 = np.arange(K)[L[:, arm1] <= 0.5]
        arm2 = C2[np.argmax(theta2[C2])]
        # compare the pair, observe the outcome, and update B
        winner, loser = (arm1, arm2) if duel(arm1, arm2) else (arm2, arm1)
        B[winner, loser] += 1
    return utils.copeland_winner_wins(B)
