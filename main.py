from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from jax import random

from bandit import problems
from bandit.utils import clone_state
from setups.setup1 import Ts, alg_list, prob_list, rng, setup_name, trials

jnp.set_printoptions(precision=3, suppress=True)

sns.set_theme(context="paper", style="darkgrid")

Path("figures").mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    data = {
        "problem": [],
        "algorithm": [],
        "time": [],
        "regret": [],
        "winner": [],
    }

    for problem_name, (problem, state) in prob_list.items():
        for algorithm_name, algorithm in alg_list.items():
            if isinstance(algorithm, tuple):
                algorithm, times, tries = algorithm
            else:
                times, tries = Ts, trials
            for T in times:
                for _ in range(tries):
                    state = problem.shuffle(state)
                    rng, subkey = random.split(rng)
                    k, history = problems.run_problem(
                        subkey, problem, clone_state(state), algorithm, T
                    )
                    regret = problem.regret(state, history)
                    data["problem"].append(problem_name)
                    data["algorithm"].append(algorithm_name)
                    t = history.shape[0]
                    t = 25 * jnp.ceil(t / 25)
                    data["time"].append(int(t))
                    data["regret"].append(float(regret))
                    data["winner"].append(bool(problem.is_winner(state, k)))
    data = pd.DataFrame(data)
    # print(data)

    sns.relplot(
        data=data,
        x="time",
        y="regret",
        hue="algorithm",
        col="problem",
        col_wrap=2,
        kind="line",
    )
    plt.savefig(f"figures/regret_{setup_name}.png")
    plt.clf()

    sns.relplot(
        data=data,
        x="time",
        y="winner",
        hue="algorithm",
        col="problem",
        col_wrap=2,
        kind="line",
    )
    plt.savefig(f"figures/winner_{setup_name}.png")
