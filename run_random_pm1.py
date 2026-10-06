"""Section 4.2: random-coefficient hypergraphs with J_i in {-1, 0, +1}.

Twelve-variable 3-uniform hypergraphs whose coefficients are drawn from
{-1, 0, +1}; a zero removes that hyperedge, so the number of active terms
varies between instances. Each instance is solved as an explicit Qulacs
circuit, under both optimisers and all four initialisation strategies, which is
the aggregation the figures of Section 4.2 average over.

The optimum, the worst attainable value and the degeneracy all come from the
spectrum of the cost Hamiltonian, obtained by diagonalisation. Unlike the
instances of Section 4.3, these are not filtered for a unique ground state:
small integer coefficients make ties common, which is why the fidelity is
defined as the total probability of measuring any optimal assignment.

    python run_random_pm1.py            # asks sample or full
    python run_random_pm1.py --sample   # 3 instances, p = 1..2, one setting
    python run_random_pm1.py --full     # 44 instances, p = 1..5, all settings

Writes random_pm1.csv into results/<name>/.
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
                  ising_terms, optimum_probability, random_pm1_terms, spectrum)

N_QUBITS = 12
MAXITER = 500
OPTIMISERS = ('BFGS', 'Powell')
INITIALS = ('one_p', 'rand_1_p', 'rand_2pi_p', 'tqa75')

FIELDS = ('instance', 'n', 'p', 'variant', 'optimiser', 'initial_angles', 'seed',
          'angles', 'terms', 'gates', 'cx', 'depth', 'cx_depth',
          'c_opt', 'c_wst', 'degeneracy', 'expectation', 'AR', 'prob_opt_pct',
          'nfev', 'nit', 'success', 'seconds')


def make_instances(count, seed, method='auto'):
    """Random {-1,0,+1} instances, with the spectrum of each."""
    out = []
    for i in range(count):
        rng = np.random.default_rng(seed * 1000 + i)
        pubo = random_pm1_terms(N_QUBITS, rng)
        if not any(c for c, _ in pubo):          # all coefficients zero
            continue
        costs = cost_vector(pubo, N_QUBITS)
        out.append((pubo, costs) + spectrum(costs, N_QUBITS, method))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--full', action='store_true')
    ap.add_argument('--sample', action='store_true')
    ap.add_argument('--spectrum', default='auto',
                    choices=('auto', 'dense', 'diagonal'))
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--name', help='name for this run (prompted for if omitted)')
    args = ap.parse_args()

    if args.full:
        full = True
    elif args.sample:
        full = False
    else:
        full = ds.ask_run_size(
            '3 instances, p = 1..2, Powell with TQA only, a few minutes',
            '44 instances, p = 1..5, both optimisers and all four '
            'initialisations, many hours')

    if full:
        count, layers = 44, 5
        settings = [(o, i) for o in OPTIMISERS for i in INITIALS]
    else:
        count, layers = 3, 2
        settings = [('Powell', 'tqa75')]

    print("\nchecking the circuit against the direct statevector route")
    if not verify(nmax=6, pmax=2):
        raise SystemExit('circuit does not reproduce the statevector result')

    name = args.name or ds.ask_run_name('random_pm1' if full else 'random_pm1_sample')
    folder = ds.run_dir(name, create=True)
    print(f"\nsaving to run '{ds.sanitise(name)}'")

    print(f"building instances and diagonalising ({args.spectrum})")
    instances = make_instances(count, args.seed, args.spectrum)
    degs = [d for *_, d in instances]
    print(f"{len(instances)} instances, ground-state degeneracy "
          f"{min(degs)} to {max(degs)}\n")

    rows, t0 = [], time.monotonic()
    for i, (pubo, costs, c_opt, c_wst, deg) in enumerate(instances):
        z_terms = ising_terms(pubo, N_QUBITS)
        active = sum(1 for c, _ in pubo if c)
        print(f"instance {i}: {active} active terms, optimum {c_opt:.0f}, "
              f"worst {c_wst:.0f}, degeneracy {deg}")
        for p in range(1, layers + 1):
            for variant in VARIANTS:
                for optimiser, init in settings:
                    rng = np.random.default_rng(args.seed * 100 + i)
                    x0 = initial_angles(z_terms, N_QUBITS, variant, p, init, rng)
                    r, _ = optimise(z_terms, costs, N_QUBITS, p, variant, x0,
                                    optimiser, MAXITER)
                    psi = r.pop('psi')
                    r |= {'instance': i, 'n': N_QUBITS, 'p': p,
                          'variant': variant, 'optimiser': optimiser,
                          'initial_angles': init, 'seed': args.seed,
                          'c_opt': c_opt, 'c_wst': c_wst, 'degeneracy': deg,
                          'AR': approximation_ratio(r['expectation'], c_opt),
                          'prob_opt_pct': optimum_probability(psi, costs, c_opt)}
                    rows.append(r)
                print(f"   p={p} {variant:>2}  "
                      f"AR={np.mean([x['AR'] for x in rows[-len(settings):]]):.4f}  "
                      f"nfev={int(np.mean([x['nfev'] for x in rows[-len(settings):]])):>6}  "
                      f"angles={rows[-1]['angles']:>3}  cx={rows[-1]['cx']:>4}")

    with open(os.path.join(folder, 'random_pm1.csv'), 'w', newline='',
              encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)

    print(f"\n{len(rows)} runs in {time.monotonic() - t0:.1f}s -> {folder}")


if __name__ == '__main__':
    main()
