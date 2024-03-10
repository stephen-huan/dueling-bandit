from functools import partial

import jax
import numpy as np
from equinox.nn import make_with_state
from jax import random

from bandit import algorithms, problems

# enable int64/float64
jax.config.update("jax_enable_x64", True)
# set random seed
rng = random.key(1)

setup_name = "4"
K = 10
Ts = [25, 50, 75, *np.arange(100, 450, 50)]
Ts_alt = [*np.arange(400, 1200, 200)]
trials = int(1e4)

rng, *subkeys = random.split(rng, num=6)
prob_list = {
    "trivial": make_with_state(problems.RankingProblem)(subkeys[0], K, 100),
    "easy": (
        make_with_state(problems.CondorcetProblem)(
            subkeys[1],
            make_with_state(problems.RandomProblem)(subkeys[2], K - 1),
        )
    ),
}
alg_list = {
    "naive": algorithms.naive,
    "dts+": (partial(algorithms.d_ts, alpha=0.51, plus=True), Ts, int(1e2)),
    "knockout": (
        partial(algorithms.knockout_online, gamma=1, exact_T=False),
        Ts_alt,
        int(1e4),
    )
}
