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

setup_name = "3"
K = 10
Ts = [*np.arange(1, 1000, 50)]
trials = int(2e2)

rng, *subkeys = random.split(rng, num=4)
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
    "dts+": partial(algorithms.d_ts, alpha=0.51, plus=True),
    "vdb-rv-shared": partial(algorithms.vdb, rv=True),
}
