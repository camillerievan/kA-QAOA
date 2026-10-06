"""Minimal QAOA for k-uniform hypergraph PUBO problems (NumPy + SciPy only).

This is the shared core of the simplified reproduction package. It deliberately
depends on nothing but NumPy and SciPy: the cost Hamiltonian of a PUBO is
diagonal in the computational basis, so a statevector simulation needs no
quantum-computing library at all.

    cost layer   exp(-i gamma C) is an elementwise phase on the statevector
    mixer layer  exp(-i beta X_j) is a 2x2 rotation applied qubit by qubit

Conventions
-----------
    |psi> = prod_{l=1..p} [ exp(-i beta_l B) exp(-i gamma_l C) ] |+>^n,
    B = sum_j X_j,  and we MAXIMISE C.

Angle grouping (the variants compared in the paper)
    sa   one cost angle and one mixer angle per layer
    ka   one cost angle per interaction order k, plus one mixer angle per qubit
    ma   one cost angle per cost term,          plus one mixer angle per qubit

Self-test:  python qaoa.py
"""
from __future__ import annotations

import itertools
import re

import numpy as np

VARIANTS = ("sa", "ka", "ma")


# ---------------------------------------------------------------------------
# problem construction
# ---------------------------------------------------------------------------

def cyclic_terms(n: int, k: int = 3):
    """3-uniform cyclic sign-alternating hypergraph, C(x) = sum_i (-1)^i prod x.

    This is the benchmark family of Section 4.1.
    """
    return [((-1.0) ** i, tuple(sorted((i + j) % n for j in range(k))))
            for i in range(n)]


def load_hyperw(path, limit: int | None = None):
    """Read hypergraphs from a .hyperw file, one per line.

    Each line is a set of terms written as {(coefficient, v1, v2, ...), ...},
    so (-5, 2, 5, 11) is the term -5 * x2 * x5 * x11.  This is the format the
    main project writes, and it is how the published instances travel with
    this package instead of being generated afresh.

    Returns a list of term lists, in the same shape as random_terms().
    """
    graphs = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            terms = []
            for group in re.findall(r"\(([^)]*)\)", line):
                nums = [int(v) for v in re.findall(r"-?\d+", group)]
                if len(nums) < 2:
                    raise ValueError(f"{path}:{lineno}: malformed term ({group})")
                terms.append((float(nums[0]), tuple(sorted(nums[1:]))))
            if not terms:
                raise ValueError(f"{path}:{lineno}: no terms found")
            graphs.append(terms)
            if limit is not None and len(graphs) >= limit:
                break
    if not graphs:
        raise ValueError(f"{path}: no hypergraphs found")
    return graphs


def random_pm1_terms(n: int, rng: np.random.Generator, k: int = 3):
    """3-uniform hypergraph on a cycle with coefficients in {-1, 0, +1}.

    The family of Section 4.2.  A zero coefficient removes that hyperedge, so
    the number of active terms varies between instances.
    """
    return [(float(rng.integers(-1, 2)),
             tuple(sorted((i + j) % n for j in range(k))))
            for i in range(n)]


def random_terms(n: int, rng: np.random.Generator, lo: int = -10, hi: int = 10,
                 k: int = 3):
    """3-uniform hypergraph on a cycle with random integer coefficients.

    Coefficients are drawn uniformly from [lo, hi], excluding zero, which is the
    family used in Section 4.3.
    """
    terms = []
    for i in range(n):
        c = 0
        while c == 0:
            c = int(rng.integers(lo, hi + 1))
        terms.append((float(c), tuple(sorted((i + j) % n for j in range(k)))))
    return terms


# ---------------------------------------------------------------------------
# the cost Hamiltonian, diagonalised
# ---------------------------------------------------------------------------

def cost_vector(terms, n: int) -> np.ndarray:
    """Diagonal of the cost Hamiltonian: C(x) for every one of the 2**n states.

    Because the Hamiltonian is diagonal in the computational basis, this vector
    IS its spectrum -- no matrix needs to be formed or numerically diagonalised.
    Bit j of the state index is variable x_j (bit 0 is the least significant).
    """
    idx = np.arange(1 << n, dtype=np.int64)
    bits = [((idx >> j) & 1).astype(np.float64) for j in range(n)]
    out = np.zeros(1 << n, dtype=np.float64)
    for coeff, support in terms:
        prod = np.ones(1 << n, dtype=np.float64)
        for j in support:
            prod *= bits[j]
        out += coeff * prod
    return out


def ising_terms(pubo_terms, n: int):
    """PUBO terms in x -> Ising terms in z, via x_j = (1 - z_j)/2.

    A degree-k monomial expands into 2**k terms of order 0..k:

        prod_{j in S} x_j = 2**-k * sum_{T subset S} (-1)**|T| prod_{j in T} z_j

    The order-0 (identity) piece is a constant: it shifts the cost by a fixed
    amount and contributes only a global phase in the circuit, so it is dropped.
    What remains are the one-, two- and three-body Z terms that the cost layer
    actually implements as Rz, Rzz and Rzzz rotations -- and whose interaction
    order is what kA-QAOA groups its angles by.
    """
    acc = {}
    for coeff, support in pubo_terms:
        k = len(support)
        for r in range(k + 1):
            for sub in itertools.combinations(sorted(support), r):
                acc[sub] = acc.get(sub, 0.0) + coeff * ((-1.0) ** r) / (2.0 ** k)
    return [(v, s) for s, v in sorted(acc.items()) if s and abs(v) > 1e-12]


def ising_phase_vectors(z_terms, n: int):
    """For each Ising term, the vector of prod_j z_j over all 2**n states."""
    idx = np.arange(1 << n, dtype=np.int64)
    z = [1.0 - 2.0 * ((idx >> j) & 1) for j in range(n)]     # z_j = +1 / -1
    out = []
    for coeff, support in z_terms:
        v = np.full(1 << n, coeff, dtype=np.float64)
        for j in support:
            v *= z[j]
        out.append(v)
    return out


def spectrum_extremes(costs: np.ndarray):
    """(optimum, worst, degeneracy of the optimum) for a MAXIMISATION problem."""
    c_opt = float(costs.max())
    c_wst = float(costs.min())
    degeneracy = int(np.count_nonzero(np.isclose(costs, c_opt)))
    return c_opt, c_wst, degeneracy


DENSE_MAX = 12          # 2**12 squared is 134 MB in float64; 2**13 is 537 MB


def spectrum(costs: np.ndarray, n: int, method: str = "auto"):
    """(optimum, worst, degeneracy) from the spectrum of the cost Hamiltonian.

    The cost Hamiltonian is diagonal in the computational basis, so its
    eigenvalues are exactly the attainable values of the cost function.

      'dense'     build it as a matrix and call scipy.linalg.eigh.  This is the
                  procedure described in the manuscript, and is the slower,
                  more literal route: about three seconds at n = 12.
      'diagonal'  read the diagonal directly.  Identical result, instant, and
                  the only option once the matrix no longer fits in memory.
      'auto'      dense up to n = DENSE_MAX, diagonal above it.

    The self-test checks the two agree.
    """
    if method == "auto":
        method = "dense" if n <= DENSE_MAX else "diagonal"

    if method == "dense":
        from scipy.linalg import eigh
        w = eigh(np.diag(costs), eigvals_only=True)
        c_opt, c_wst = float(w[-1]), float(w[0])
        return c_opt, c_wst, int(np.count_nonzero(np.isclose(w, c_opt)))

    if method == "diagonal":
        return spectrum_extremes(costs)

    raise ValueError(f"unknown spectrum method {method!r}")


def approximation_ratio(expected: float, c_opt: float) -> float:
    """Approximation ratio as defined in the manuscript.

        AR = <C> / C_opt

    with <C> the expected cost of the output state and C_opt the largest
    eigenvalue of the cost Hamiltonian, these objectives being maximised.
    It equals one when the optimiser reaches the optimum.
    """
    if c_opt == 0:
        return float('nan')              # undefined for a zero optimum
    return expected / c_opt


# ---------------------------------------------------------------------------
# angle bookkeeping
# ---------------------------------------------------------------------------

def angle_counts(z_terms, n: int, variant: str):
    """(cost angles, mixer angles) per layer, for one variant.

    `z_terms` are the ISING terms: SA uses a single cost angle, kA one per
    interaction order k, and MA one per term.  The mixer is never grouped.
    """
    if variant == "sa":
        return 1, 1
    if variant == "ka":
        orders = sorted({len(s) for _, s in z_terms})
        return len(orders), n
    if variant == "ma":
        return len(z_terms), n
    raise ValueError(variant)


def _cost_angle_index(z_terms, variant: str):
    """Which cost angle each Ising term uses."""
    if variant == "sa":
        return [0] * len(z_terms)
    if variant == "ka":
        orders = sorted({len(s) for _, s in z_terms})
        return [orders.index(len(s)) for _, s in z_terms]
    if variant == "ma":
        return list(range(len(z_terms)))
    raise ValueError(variant)


def n_params(z_terms, n: int, variant: str, p: int) -> int:
    c, m = angle_counts(z_terms, n, variant)
    return p * (c + m)


# ---------------------------------------------------------------------------
# the circuit
# ---------------------------------------------------------------------------

def _apply_mixer(psi: np.ndarray, n: int, betas: np.ndarray) -> np.ndarray:
    """exp(-i beta_j X_j) on each qubit in turn."""
    psi = psi.reshape(-1)
    for j, beta in enumerate(betas):
        c, s = np.cos(beta), np.sin(beta)
        v = psi.reshape(-1, 2, 1 << j)          # split on bit j
        a, b = v[:, 0, :].copy(), v[:, 1, :].copy()
        v[:, 0, :] = c * a - 1j * s * b
        v[:, 1, :] = c * b - 1j * s * a
        psi = v.reshape(-1)
    return psi


def qaoa_state(z_terms, n: int, p: int, angles: np.ndarray, variant: str,
               phase_vectors=None, per_layer=False):
    """Run the circuit. Returns the final statevector (or a list, one per layer).

    `z_terms` are the Ising terms; `angles` is laid out as
    [all mixer angles | all cost angles], layer-major, matching the ordering
    used to generate the initial angles.
    """
    n_cost, n_mix = angle_counts(z_terms, n, variant)
    betas = angles[:p * n_mix].reshape(p, n_mix)
    gammas = angles[p * n_mix:].reshape(p, n_cost)
    which = _cost_angle_index(z_terms, variant)

    if phase_vectors is None:
        phase_vectors = ising_phase_vectors(z_terms, n)

    dim = 1 << n
    psi = np.full(dim, 1.0 / np.sqrt(dim), dtype=np.complex128)
    states = []

    for layer in range(p):
        phase = np.zeros(dim, dtype=np.float64)
        for t, pv in enumerate(phase_vectors):
            phase += gammas[layer, which[t]] * pv
        psi = psi * np.exp(-1j * phase)
        mix = betas[layer] if n_mix > 1 else np.repeat(betas[layer], n)
        psi = _apply_mixer(psi, n, mix)
        if per_layer:
            states.append(psi.copy())

    return states if per_layer else psi


def expected_cost(psi: np.ndarray, costs: np.ndarray) -> float:
    return float(np.sum(np.abs(psi) ** 2 * costs))


def optimum_probability(psi: np.ndarray, costs: np.ndarray, c_opt: float) -> float:
    """Total probability of measuring an optimal assignment (percent)."""
    mask = np.isclose(costs, c_opt)
    return float(np.sum(np.abs(psi[mask]) ** 2) * 100.0)


# ---------------------------------------------------------------------------
# initial angles
# ---------------------------------------------------------------------------

def initial_angles(z_terms, n: int, variant: str, p: int, strategy: str,
                   rng: np.random.Generator) -> np.ndarray:
    """Starting angles. Strategies match those compared in the paper."""
    n_cost, n_mix = angle_counts(z_terms, n, variant)
    n_b, n_g = p * n_mix, p * n_cost

    if strategy == "tqa75":                      # trotterised quantum annealing
        dt = 0.75
        b = np.concatenate([np.full(n_mix, (1.0 - (l + 1) / p) * dt) for l in range(p)])
        g = np.concatenate([np.full(n_cost, ((l + 1) / p) * dt) for l in range(p)])
        return np.concatenate([b, g])
    if strategy == "one_p":
        return np.full(n_b + n_g, 1.0 / p)
    if strategy == "rand_1_p":
        return rng.uniform(0.0, 1.0 / p, n_b + n_g)
    if strategy == "rand_2pi_p":
        return rng.uniform(0.0, 2.0 * np.pi / p, n_b + n_g)
    raise ValueError(strategy)


# ---------------------------------------------------------------------------
# stabilizer Renyi entropy M_2
# ---------------------------------------------------------------------------

def m2_renyi(psi: np.ndarray, n: int, chunk: int = 256) -> float:
    """Stabilizer Renyi entropy M_2 (Leone, Oliviero & Hamma).

        M_2 = -log2( sum_P <psi|P|psi>^4 / 2**n )

    For a Pauli P = X^a Z^b we have

        <psi| X^a Z^b |psi> = sum_x conj(psi_x) (-1)^(b.x) psi_(x xor a),

    which for fixed a is a Walsh-Hadamard transform over b.  Sweeping a gives
    the whole 4**n Pauli spectrum in O(n * 4**n) instead of O(8**n).  We process
    a in chunks and run the butterfly in place to keep memory bounded.

    Zero for stabilizer states; log2(4/3) per qubit for T states.
    """
    dim = 1 << n
    psi = np.asarray(psi).reshape(dim)
    cpsi = np.conj(psi)
    idx = np.arange(dim)

    total = 0.0
    for a0 in range(0, dim, chunk):
        a = idx[a0:a0 + chunk]
        f = cpsi[None, :] * psi[a[:, None] ^ idx[None, :]]   # f[a, x]
        h = 1
        while h < dim:                                        # in-place FWHT
            f = f.reshape(f.shape[0], -1, 2, h)
            x, y = f[:, :, 0, :], f[:, :, 1, :]
            u = x + y
            y -= x
            x[...] = u
            f = f.reshape(f.shape[0], -1)
            h *= 2
        total += float(np.sum(np.abs(f) ** 4))

    return float(-np.log2(total / dim))


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # cost vector matches a direct evaluation
    terms = [(3.0, (0, 1)), (-2.0, (1, 2))]
    cv = cost_vector(terms, 3)
    for i in range(8):
        x = [(i >> j) & 1 for j in range(3)]
        assert abs(cv[i] - (3 * x[0] * x[1] - 2 * x[1] * x[2])) < 1e-12
    print("cost_vector ....... ok")

    # diagonalising the cost Hamiltonian agrees with reading its diagonal
    for n_chk in (4, 6, 8):
        cv_chk = cost_vector(cyclic_terms(n_chk), n_chk)
        assert spectrum(cv_chk, n_chk, "dense") == spectrum(cv_chk, n_chk, "diagonal")
    rnd = cost_vector(random_terms(9, np.random.default_rng(0)), 9)
    assert spectrum(rnd, 9, "dense") == spectrum(rnd, 9, "diagonal")
    print("spectrum .......... ok (eigh agrees with the diagonal)")

    # the AR is one at the optimum and scales with the attained value
    c_opt, c_wst, deg = spectrum_extremes(cv)
    assert approximation_ratio(c_opt, c_opt) == 1.0
    assert abs(approximation_ratio(c_opt / 2, c_opt) - 0.5) < 1e-12
    print("approximation_ratio ok (one at the optimum)")

    # the Ising expansion reproduces the PUBO cost, up to the dropped constant
    for n in (5, 6, 7):
        pub = cyclic_terms(n)
        zt = ising_terms(pub, n)
        rebuilt = sum(ising_phase_vectors(zt, n))
        diff = cost_vector(pub, n) - rebuilt
        assert np.allclose(diff, diff[0]), n          # differ by a constant only
    print("ising_terms ....... ok (matches PUBO up to a constant)")

    # term counts and angle counts match the formulas quoted in the paper
    for n in range(5, 13):
        zt = ising_terms(cyclic_terms(n), n)
        eps = n % 2
        got = {k: sum(1 for _, s in zt if len(s) == k) for k in (1, 2, 3)}
        assert got == {1: n, 2: n + eps, 3: n}, (n, got)
        assert angle_counts(zt, n, "sa") == (1, 1)
        assert angle_counts(zt, n, "ka") == (3, n)          # -> n + 3 per layer
        assert angle_counts(zt, n, "ma") == (3 * n + eps, n)
    print("angle_counts ...... ok (sa 2, ka n+3, ma 4n+eps per layer)")

    # the state stays normalised
    zt = ising_terms(cyclic_terms(6), 6)
    rng = np.random.default_rng(0)
    for variant in VARIANTS:
        ang = initial_angles(zt, 6, variant, 2, "tqa75", rng)
        psi = qaoa_state(zt, 6, 2, ang, variant)
        assert abs(np.sum(np.abs(psi) ** 2) - 1.0) < 1e-10, variant
    print("qaoa_state ........ ok (normalised for sa/ka/ma)")

    # M_2: zero for |0...0>, log2(4/3) per qubit for T states
    for n in (2, 3, 4):
        z = np.zeros(1 << n, dtype=np.complex128); z[0] = 1.0
        assert abs(m2_renyi(z, n)) < 1e-9, n
        t1 = np.array([1, np.exp(1j * np.pi / 4)], dtype=np.complex128) / np.sqrt(2)
        st = t1
        for _ in range(n - 1):
            st = np.kron(st, t1)
        assert abs(m2_renyi(st, n) - n * np.log2(4 / 3)) < 1e-9, n
    print("m2_renyi .......... ok (0 for |0>, n*log2(4/3) for T states)")

    print("\nqaoa.py self-test passed")
