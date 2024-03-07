from functools import partial

import jax
import numpy as np
from equinox.nn import make_with_state
from jax import random

from bandit import algorithms, problems

# enable int64/float64
jax.config.update("jax_enable_x64", True)
# set random seed
rng = random.key(0)

setup_name = "1"
K = 10
Ts = [*np.arange(1, 1000, 100)]
# Ts = [*np.arange(1, 1000, 100), 10_000, 20_000, 40_000, 80_000]
trials = int(5e1)

rng, *subkeys = random.split(rng, num=6)
prob_list = {
    "trivial": make_with_state(problems.RankingProblem)(subkeys[0], K, 100),
    "easy": (
        make_with_state(problems.CondorcetProblem)(
            subkeys[1],
            make_with_state(problems.RandomProblem)(subkeys[2], K - 1),
        )
    ),
    "medium": make_with_state(problems.CopelandProblem)(subkeys[3], K),
    "hard": make_with_state(problems.RandomProblem)(subkeys[4], K),
}
alg_list = {
    "naive": algorithms.naive,
    "beat-the-mean": partial(algorithms.btm_online, gamma=1),
    # "scb": partial(algorithms.scb),
    "dts": partial(algorithms.d_ts, alpha=0.51, plus=False),
    "dts+": partial(algorithms.d_ts, alpha=0.51, plus=True),
    "vdb-iw-independent": partial(algorithms.vdb_ind, rv=False),
    "vdb-iw-shared": partial(algorithms.vdb, rv=False),
    "vdb-rv-independent": partial(algorithms.vdb_ind, rv=True),
    "vdb-rv-shared": partial(algorithms.vdb, rv=True),
}
