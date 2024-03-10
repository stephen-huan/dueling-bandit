from functools import partial
from typing import TypeAlias

import jax.numpy as jnp
from jax import Array, lax, random

from ..utils import Arm, Duel, KeyArray, S, jit

# Knockout ([4])


@jit
def permutation_prefix(rng: KeyArray, x: Array, n: int) -> Array:
    """Permute x[:n] in-place."""
    Data: TypeAlias = tuple[Array, KeyArray]  # type: ignore
    data = (x, rng)

    def body_fun(i: int, data: Data) -> Data:
        """Inner loop."""
        x, rng = data
        rng, subkey = random.split(rng)
        j = random.randint(subkey, (), i, n)
        x = x.at[jnp.array([i, j])].set((x[j], x[i]))
        return x, rng

    x, _ = lax.fori_loop(0, n - 1, body_fun, data)
    return x


@partial(jit, static_argnums=2)
def compare(
    i: Arm, j: Arm, duel: Duel, state: S, eps: Array, delta: Array
) -> tuple[Arm, S, int]:
    """The Compare algorithm 1 of [4]."""
    m = jnp.log(2 / delta) / (2 * jnp.square(eps))
    r = 0
    w = jnp.zeros(())

    Data: TypeAlias = tuple[int, Array, S]  # type: ignore
    data = (r, w, state)

    def cond_fun(data: Data) -> Array:
        """Whether to stop the loop."""
        r, w, _ = data
        p = jnp.where(r != 0, w / r, 1 / 2)
        c = jnp.where(
            r != 0,
            jnp.sqrt(jnp.log(4 * jnp.square(r) / delta) / (2 * r)),
            1 / 2,
        )
        return (jnp.abs(p - 1 / 2) <= c - eps) & (r <= m)

    def body_fun(data: Data) -> Data:
        """Inner loop."""
        r, w, state = data
        outcome, state = duel(state, i, j)
        w += outcome
        r += 1
        return r, w, state

    r, w, state = lax.while_loop(cond_fun, body_fun, data)
    return jnp.where(w / r <= 1 / 2, j, i), state, r


@partial(jit, static_argnums=3)
def knockout_round(
    rng: KeyArray,
    s: Array,
    n: int,
    duel: Duel,
    state: S,
    eps: Array,
    delta: Array,
) -> tuple[Array, int, S, int]:
    """The Knockout-Round algorithm 2 of [4]."""
    m = (n + 1) // 2
    # adjacent indices in s are pairs
    s = permutation_prefix(rng, s, n)
    t = 0

    Data: TypeAlias = tuple[Array, S, int]  # type: ignore
    data = (s, state, t)

    def body_fun(i: int, data: Data) -> Data:
        """Inner loop."""
        s, state, t = data
        arm, state, r = compare(
            s[2 * i], s[2 * i + 1], duel, state, eps, delta
        )
        s = s.at[i].set(arm)
        return s, state, t + r

    s, state, t = lax.fori_loop(0, m - (n % 2), body_fun, data)
    # copy last element
    s = jnp.where(n % 2 == 1, s.at[m - 1].set(s[n - 1]), s)
    return s, m, state, t


@partial(jit, static_argnums=(1, 2))
def knockout(
    rng: KeyArray,
    K: int,
    duel: Duel,
    state: S,
    eps: Array,
    delta: Array,
    gamma: float,
) -> tuple[Arm, S, int]:
    """The Knockout algorithm 3 of [4]."""
    i = 1
    s = jnp.arange(K)
    c = jnp.pow(2, 1 / 3) - 1
    n = s.shape[0]
    t = 0
    Data: TypeAlias = tuple[Array, S, int, KeyArray, int, int]  # type: ignore
    data = (s, state, t, rng, n, i)

    def body_fun(data: Data) -> Data:
        """Inner loop."""
        s, state, t, rng, n, i = data
        eps_i = c * eps / (gamma * jnp.pow(2, i / 3))
        delta_i = delta / jnp.pow(2, i)
        rng, subkey = random.split(rng)
        s, n, state, r = knockout_round(
            subkey, s, n, duel, state, eps_i, delta_i
        )
        return s, state, t + r, rng, n, i + 1

    s, state, t, *_ = lax.while_loop(lambda data: data[-2] > 1, body_fun, data)
    return s[0], state, t


@partial(jit, static_argnums=(1, 2))
def knockout_online(
    rng: KeyArray,
    K: int,
    duel: Duel,
    state: S,
    T: int,
    gamma: float = 1,
    exact_T: bool = False,
) -> tuple[Arm, S]:
    """Knockout adapted to an online setting."""
    # from beat the mean
    delta = jnp.reciprocal(2 * T * K)
    # epsilon picked such that comparisons <= T
    # see the proof of theorem 3
    c = jnp.pow(2, 1 / 3) - 1
    eps = jnp.sqrt(
        K
        * jnp.square(gamma)
        / (2 * jnp.square(c))
        * (jnp.pow(2, 1 / 3) / jnp.square(c) + jnp.log(2 / delta) / c)
        / T
    )
    arm, state, t = knockout(rng, K, duel, state, eps, delta, gamma)
    # waste the remaining comparisons
    state = lax.cond(
        exact_T,
        lambda state: lax.fori_loop(
            0, T - t, lambda _, state: duel(state, arm, arm)[1], state
        ),
        lambda state: state,
        state,
    )
    return arm, state
