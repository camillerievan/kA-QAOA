# Structured Parameterization and Non-Stabilizerness in Hypergraph QAOA

A self-contained, runnable implementation of the procedure reported in
*Structured Parameterization and Non-Stabilizerness in Hypergraph QAOA*. It
recomputes some of the published results exactly and the rest in distribution;
see **What reproduces, and what does not** below.
Everything runs from a clean
checkout with no edits and no database.

## Requirements

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Four packages, all pinned: NumPy and SciPy for the optimisers, the spectrum and
the magic measurement; **Qulacs** for the quantum circuits, which every
experiment runs through; and matplotlib for `plot.py` only.

These are the versions this package was verified with, and the only ones on
record (Appendix F of the manuscript lists the same ones). The published runs
were made on a university cluster under Python 3.10, and their package versions
were not logged.

## Running

```
python qaoa.py                 # self-test of the core (a few seconds)
python circuit.py              # check the circuit against the statevector route
python run_cyclic.py           # Section 4.1
python run_random_pm1.py       # Section 4.2
python run_magic12.py          # Section 4.3
python plot.py                 # draw the figures from any result set
```

Each experiment asks whether you want a **sample** run (minutes, for checking
that it works) or the **full** sweep (hours, the scale reported in the paper).
Pass `--sample` or `--full` to skip the question. All accept `--help`.

| script | family | sample | full |
|---|---|---|---|
| `run_cyclic.py` | 3-uniform cyclic sign-alternating | `n` = 4…8, `p` = 1…2, one setting | `n` = 4…15, `p` = 1…3, 2 optimisers × 4 initialisations |
| `run_random_pm1.py` | random `J ∈ {-1,0,+1}`, `n` = 12 | 3 instances, `p` = 1…2 | 44 instances, `p` = 1…5, 2 optimisers × 4 initialisations |
| `run_magic12.py` | the 70 published `n` = 12 hypergraphs, with magic | 3 instances, `p` = 1…2 | 70 instances, `p` = 1…5, Powell + TQA |

`run_magic12.py --no-magic` skips the stabilizer Rényi entropy, which is the
slowest step.

## How the circuits are run

The QAOA is an explicit Qulacs circuit (`circuit.py`), built once per
(instance, depth, variant) with its angles rebound on each objective call. A
k-body cost term is applied as one native multi-qubit Pauli rotation, which is
exact and fast.

Hardware cost is reported separately, by compiling the same layer the textbook
way — a CNOT ladder per term, one `Rz`, the ladder undone — and counting the
result. Those are the `gates`, `cx`, `depth` and `cx_depth` columns. The two
circuits implement the same unitary; only the second is a claim about hardware.
`python circuit.py` checks the circuit against a direct statevector
implementation and reports the largest amplitude difference.

## Optima

`qaoa.spectrum()` returns the optimum, the worst attainable value and the
degeneracy from the spectrum of the cost Hamiltonian. Up to `n` = 12 it builds
the matrix and calls `scipy.linalg.eigh`, which is the procedure described in
the manuscript; beyond that the matrix no longer fits in memory and it reads the
diagonal instead. The two agree exactly, and the self-test asserts it. Pass
`--spectrum dense|diagonal` to force either.

## Output

Written to `results/` as CSV, one row per run — the raw per-instance data, not
just aggregates.

| file | contents |
|---|---|
| `cyclic.csv` | one row per (n, p, variant, optimiser, initialisation) |
| `random12.csv` | one row per (instance, p, variant) |
| `random12_magic.csv` | one row per (instance, p, variant, layer): M₂ |
| `random_pm1.csv` | one row per (instance, p, variant, optimiser, initialisation) |

These are the files a run writes. Of them, only `cyclic.csv`,
`random12.csv` and `random12_magic.csv` have published counterparts under
`results/`; `random_pm1.csv` is produced by `run_random_pm1.py` but no
published copy is shipped, because the Section 4.2 instances were drawn
without a recorded seed and cannot be recovered.

Every row carries `c_opt`, `c_wst`, `degeneracy`, `angles`, `terms`, `AR`,
`prob_opt_pct`, `nfev`, `seed`, and
the circuit cost (`gates`, `cx`, `depth`, `cx_depth`), so each figure and the
circuit-cost table can be rebuilt from the CSV alone. `plot.py` draws any of
them: `python plot.py` and pick a run.

## Figure-to-script mapping

| paper figure | script | filter on the CSV |
|---|---|---|
| Fig. A1–A3 (cyclic: AR, AP, nfev) | `run_cyclic.py` | `family = cyclic`, group by `n`, `p`, `variant` |
| Table C2 (nfev by initialisation) | `run_cyclic.py` | group by `initial_angles` |
| Table C1 (nfev by optimiser) | `run_cyclic.py` | group by `optimiser` |
| Appendix B (random coefficients, J ∈ {-1,0,+1}) | `run_random_pm1.py` | `random_pm1.csv`, group by `p`, `variant` |
| Fig. 3a (magic barrier) | `run_magic12.py` | `random12_magic.csv`, `p = 5`, mean M₂ by `layer`, `variant` |
| Fig. 3b (AR vs depth) | `run_magic12.py` | `random12.csv`, mean `AR` by `p`, `variant` |
| Fig. 3c (nfev vs depth) | `run_magic12.py` | `random12.csv`, mean `nfev` by `p`, `variant` |

## Method notes

**Optimum and worst case.** Both come from the spectrum of the cost
Hamiltonian. Up to *n* = 12 the matrix is formed and diagonalised with
`scipy.linalg.eigh`, which is the procedure the manuscript describes; above that
it no longer fits in memory, and since the Hamiltonian is diagonal the spectrum
is read off the diagonal instead. The two agree exactly and the self-test
asserts it. The same computation gives the degeneracy of the optimum, as the
multiplicity of the extreme eigenvalue; `run_magic12.py` uses it to accept only
instances with a unique ground state. Pass `--spectrum dense|diagonal` to force
either route.

**Approximation ratio.** Reported as

```
AR = <C> / C_opt
```

with `<C>` the expected cost of the output distribution and `C_opt` the largest
eigenvalue of the cost Hamiltonian, these objectives being maximised. This is
the definition used in the manuscript, and it equals one when the optimiser
reaches the optimum. `c_opt` and `c_wst` are on every row, so other
normalisations can be computed from the raw data if wanted.

**Angle grouping.** Applied to the *Ising* terms, after the substitution
x = (1 − z)/2, since those are what the cost layer implements as Rz, Rzz and
Rzzz rotations. Per layer this gives 2 angles for SA-QAOA, *n* + 3 for kA-QAOA,
and 4*n* + ε_n for MA-QAOA on the cyclic family, matching the counts quoted in
the paper. The self-test in `qaoa.py` asserts this.

**Optimiser settings.** BFGS with SciPy's default `gtol = 1e-5` and numerically
differentiated gradients; Powell with `xtol = ftol = 1e-4`. Both capped at
`maxiter = 500`, which is our choice rather than a SciPy default.

**Seeds.** Every random draw in this package goes through an explicitly seeded
`numpy.random.default_rng`, so any run made here is reproducible and `--seed`
changes the whole run. The seeding is a property of this package rather than a
reconstruction of the earlier runs: the two random initialisation strategies
(`rand_1_p` and `rand_2pi_p`) drew from an unseeded generator when the published
results were produced, so re-running them here gives a different, statistically
equivalent draw.

Seeding was added when this package was prepared, after the published runs had
been made, and the published runs were deliberately left as they are. From this
release onwards the seed is recorded with every run.

**What reproduces, and what does not.** The CSV files under `results/` are
exports of the original runs; they are data, not output of this code. The
Section 4.3 tables and figures, and the evaluation counts at matched quality,
are computed from them and can be recomputed exactly.

Re-running the code is a different matter. It solves the same instances and
preserves the ordering between the variants, but not the individual values:
evaluation counts differ, and in some cases so do approximation ratios. This
is an independent implementation, and arbitrary choices made differently --
notably the angle parametrisation convention -- send the optimiser along a
different path. That applies to the deterministic initialisation strategies as
much as to the random ones.

**Not in this release.** Section 4.1 of the paper -- its text, Figures A1-A3,
Tables C1 and C2, and the speed-up ratios quoted in the Conclusion -- reports
an earlier set of the original runs, including those using the Powell
optimiser. Those runs are not among the files shipped here, so those results
cannot be recomputed from `results/`. The results of Section 4.2 are likewise
not shipped: its instances were drawn without a recorded seed and cannot be
recovered. What can be recomputed exactly from `results/` is Section 4.3
(Tables 2 and 3, Figure 3) and Table C3.

**Magic.** M₂ is the stabilizer Rényi entropy of Leone, Oliviero and Hamma,

```
M2 = -log2( sum_P <psi|P|psi>^4 / 2**n )
```

computed over the full 4ⁿ Pauli group by a Walsh–Hadamard transform, in
O(n · 4ⁿ) rather than O(8ⁿ). The self-test checks it against two analytically
known cases: zero for a computational basis state, and n·log2(4/3) for a product
of T states.

## Relation to the published results

Regenerated figures should agree with the published ones in ordering and
magnitude, but exact values can differ wherever an arbitrary choice is made
differently, notably the angle parametrisation convention.

The problems themselves are not a difference. `run_magic12.py` solves the same
70 hypergraphs the manuscript used, shipped as
`instances/random12_published.hyperw`; each is checked on load for a unique
ground state. `run_cyclic.py` builds its family from the definition in the
paper.
