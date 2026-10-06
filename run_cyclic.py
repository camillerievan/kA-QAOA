"""Section 4.1: 3-uniform cyclic sign-alternating hypergraphs.

C(x) = sum_i (-1)^i x_i x_(i+1 mod n) x_(i+2 mod n), for n = 4 upwards. Each
size is solved as an explicit Qulacs circuit, under both optimisers and all
four initialisation strategies, which is the aggregation the figures of
Section 4.1 average over.

The optimum and the worst attainable value come from the spectrum of the cost
Hamiltonian, obtained by diagonalisation up to n = 12 and by reading the
diagonal above that, where the matrix no longer fits in memory. The two agree
exactly; the self-test in qaoa.py checks it.

    python run_cyclic.py            # asks sample or full
    python run_cyclic.py --sample   # n = 4..8, p = 1..2, one setting
    python run_cyclic.py --full     # n = 4..15, p = 1..3, all settings

Writes cyclic.csv into results/<name>/.
"""
from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np

import datasource as ds
from circuit import optimise, verify
from qaoa import (VARIANTS, approximation_ratio, cost_vector, cyclic_terms,
                  initial_angles, ising_terms, optimum_probability, spectrum)

MAXITER = 500
OPTIMISERS = ('BFGS', 'Powell')
INITIALS = ('one_p', 'rand_1_p', 'rand_2pi_p', 'tqa75')

FIELDS = ('family', 'n', 'p', 'variant', 'optimiser', 'initial_angles', 'seed',
          'angles', 'terms', 'gates', 'cx', 'depth', 'cx_depth',
          'c_opt', 'c_wst', 'degeneracy', 'expectation', 'AR', 'prob_opt_pct',
          'nfev', 'nit', 'success', 'seconds')


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
            'n = 4..8, p = 1..2, Powell with TQA only, a few minutes',
            'n = 4..15, p = 1..3, both optimisers and all four '
            'initialisations, many hours')

    if full:
        sizes, layers = range(4, 16), 3
        settings = [(o, i) for o in OPTIMISERS for i in INITIALS]
    else:
        sizes, layers = range(4, 9), 2
        settings = [('Powell', 'tqa75')]

    print("\nchecking the circuit against the direct statevector route")
    if not verify(nmax=6, pmax=2):
        raise SystemExit('circuit does not reproduce the statevector result')

    name = args.name or ds.ask_run_name('cyclic' if full else 'cyclic_sample')
    folder = ds.run_dir(name, create=True)
    print(f"\nsaving to run '{ds.sanitise(name)}'\n")

    rows, t0 = [], time.monotonic()
    for n in sizes:
        pubo = cyclic_terms(n)
        costs = cost_vector(pubo, n)
        c_opt, c_wst, deg = spectrum(costs, n, args.spectrum)
        z_terms = ising_terms(pubo, n)
        print(f"n={n:>2}: {len(z_terms)} Ising terms, optimum {c_opt:.0f}, "
              f"worst {c_wst:.0f}, degeneracy {deg}")
        for p in range(1, layers + 1):
            for variant in VARIANTS:
                first = len(rows)
                for optimiser, init in settings:
                    rng = np.random.default_rng(args.seed * 100 + n)
                    x0 = initial_angles(z_terms, n, variant, p, init, rng)
                    r, _ = optimise(z_terms, costs, n, p, variant, x0,
                                    optimiser, MAXITER)
                    psi = r.pop('psi')
                    r |= {'family': 'cyclic', 'n': n, 'p': p, 'variant': variant,
                          'optimiser': optimiser, 'initial_angles': init,
                          'seed': args.seed, 'c_opt': c_opt, 'c_wst': c_wst,
                          'degeneracy': deg,
                          'AR': approximation_ratio(r['expectation'], c_opt),
                          'prob_opt_pct': optimum_probability(psi, costs, c_opt)}
                    rows.append(r)
                made = rows[first:]
                print(f"   p={p} {variant:>2}  "
                      f"AR={np.mean([x['AR'] for x in made]):.4f}  "
                      f"nfev={int(np.mean([x['nfev'] for x in made])):>6}  "
                      f"angles={made[-1]['angles']:>3}  cx={made[-1]['cx']:>4}")

    with open(os.path.join(folder, 'cyclic.csv'), 'w', newline='',
              encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)

    print(f"\n{len(rows)} runs in {time.monotonic() - t0:.1f}s -> {folder}")


if __name__ == '__main__':
    main()
