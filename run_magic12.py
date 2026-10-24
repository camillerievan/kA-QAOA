"""Section 4.3: the 70 published 12-variable hypergraphs, with their magic.

Solves the hypergraphs the manuscript used, read from
instances/random12_published.hyperw, as explicit Qulacs circuits. Records the
approximation ratio, the fidelity, the number of objective evaluations, the
per-circuit cost, and the stabilizer Renyi entropy after each layer.

The optimum and the worst attainable value come from the spectrum of the cost
Hamiltonian, obtained by diagonalisation (scipy.linalg.eigh); the degeneracy
falls out of the same step as the multiplicity of the extreme eigenvalue.

    python run_magic12.py                 # asks sample or full
    python run_magic12.py --full          # all 70 instances, p = 1..5
    python run_magic12.py --sample        # 3 instances, p = 1..2
    python run_magic12.py --no-magic      # skip M_2, much faster

Writes random12.csv and random12_magic.csv into results/<name>/.
"""
from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np

import datasource as ds
from circuit import optimise, verify
from qaoa import (VARIANTS, approximation_ratio, cost_vector, initial_angles,
                  ising_terms, load_hyperw, optimum_probability, spectrum)

N_QUBITS = 12
OPTIMISER = 'Powell'           # Section 4.3 uses Powell with TQA_75 only
INITIAL = 'tqa75'
MAXITER = 500

HERE = os.path.dirname(os.path.abspath(__file__))
PUBLISHED = os.path.join(HERE, 'instances', 'random12_published.hyperw')

RUN_FIELDS = ('instance', 'n', 'p', 'variant', 'optimiser', 'initial_angles',
              'angles', 'terms', 'gates', 'cx', 'depth', 'cx_depth',
              'c_opt', 'c_wst', 'degeneracy', 'expectation', 'AR',
              'prob_opt_pct', 'nfev', 'nit', 'success', 'seconds')
MAGIC_FIELDS = ('instance', 'n', 'p', 'variant', 'layer', 'M2')


def load_instances(limit=None, method='auto'):
    """The published hypergraphs, with the spectrum of each."""
    out = []
    for i, pubo in enumerate(load_hyperw(PUBLISHED, limit=limit)):
        costs = cost_vector(pubo, N_QUBITS)
        c_opt, c_wst, deg = spectrum(costs, N_QUBITS, method)
        if deg != 1:
            raise RuntimeError(f"instance {i} has a degenerate ground state ({deg})")
        out.append((pubo, costs, c_opt, c_wst, deg))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--full', action='store_true', help='all 70 instances, p = 1..5')
    ap.add_argument('--sample', action='store_true', help='3 instances, p = 1..2')
    ap.add_argument('--no-magic', action='store_true', help='skip the M_2 measurement')
    ap.add_argument('--spectrum', default='auto', choices=('auto', 'dense', 'diagonal'),
                    help="'dense' diagonalises the matrix, 'diagonal' reads it")
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--name', help='name for this run (prompted for if omitted)')
    args = ap.parse_args()

    if args.full:
        full = True
    elif args.sample:
        full = False
    else:
        full = ds.ask_run_size('3 instances, p = 1..2, about a minute',
                               '70 instances, p = 1..5, several hours')
    instances_n, layers = (70, 5) if full else (3, 2)

    print("\nchecking the circuit against the direct statevector route")
    if not verify(nmax=6, pmax=2):
        raise SystemExit('circuit does not reproduce the statevector result')

    name = args.name or ds.ask_run_name('random12' if full else 'random12_sample')
    folder = ds.run_dir(name, create=True)
    print(f"\nsaving to run '{ds.sanitise(name)}'")

    print(f"reading instances and diagonalising ({args.spectrum})")
    instances = load_instances(instances_n, args.spectrum)
    print(f"{len(instances)} instances, all with a unique ground state\n")

    runs, magics, t0 = [], [], time.monotonic()
    for i, (pubo, costs, c_opt, c_wst, deg) in enumerate(instances):
        z_terms = ising_terms(pubo, N_QUBITS)
        print(f"instance {i}: {len(pubo)} terms, optimum {c_opt:.0f}, "
              f"worst {c_wst:.0f}")
        for p in range(1, layers + 1):
            for variant in VARIANTS:
                # one independent stream per run, derived from --seed
                rng = np.random.default_rng(np.random.SeedSequence(
                    [args.seed, i, p, VARIANTS.index(variant)]))
                x0 = initial_angles(z_terms, N_QUBITS, variant, p, INITIAL, rng)
                want_magic = (not args.no_magic) and p == layers
                r, magic = optimise(z_terms, costs, N_QUBITS, p, variant, x0,
                                    OPTIMISER, MAXITER, magic_layers=want_magic)
                psi = r.pop('psi')
                r |= {'instance': i, 'n': N_QUBITS, 'p': p, 'variant': variant,
                      'optimiser': OPTIMISER, 'initial_angles': INITIAL,
                      'c_opt': c_opt, 'c_wst': c_wst, 'degeneracy': deg,
                      'AR': approximation_ratio(r['expectation'], c_opt),
                      'prob_opt_pct': optimum_probability(psi, costs, c_opt)}
                runs.append(r)
                for layer, m in enumerate(magic, start=1):
                    magics.append({'instance': i, 'n': N_QUBITS, 'p': p,
                                   'variant': variant, 'layer': layer, 'M2': m})
                extra = f"  M2 peak {max(magic):.3f}" if magic else ''
                print(f"   p={p} {variant:>2}  AR={r['AR']:.4f}  "
                      f"nfev={r['nfev']:>6}  F={r['prob_opt_pct']:6.2f}%  "
                      f"angles={r['angles']:>3}  cx={r['cx']:>4}{extra}")

    with open(os.path.join(folder, 'random12.csv'), 'w', newline='',
              encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=RUN_FIELDS, extrasaction='ignore')
        w.writeheader()
        w.writerows(runs)
    if magics:
        with open(os.path.join(folder, 'random12_magic.csv'), 'w', newline='',
                  encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=MAGIC_FIELDS)
            w.writeheader()
            w.writerows(magics)

    print(f"\n{len(runs)} runs in {time.monotonic() - t0:.1f}s -> {folder}")
    if magics:
        print(f"{len(magics)} magic rows")


if __name__ == '__main__':
    main()
