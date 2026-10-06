"""Turn the result CSVs into the tables behind the paper's figures.

Reads whatever exists in results/ and prints the aggregates, so a reader can
check a figure without writing any analysis code of their own.

    python summarise.py
"""
from __future__ import annotations

import csv
import os
import statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
ORDER = ("sa", "ka", "ma")


def load(name):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def table(title, rows, key, value, fmt="{:.4f}", agg=st.mean):
    """One row per `key`, one column per variant."""
    print(f"\n{title}")
    grid = defaultdict(dict)
    for r in rows:
        grid[r[key]].setdefault(r["variant"], []).append(float(r[value]))
    print(f"{key:>6} " + " ".join(f"{v.upper() + '-QAOA':>12}" for v in ORDER))
    print("-" * (7 + 13 * len(ORDER)))
    for k in sorted(grid, key=lambda x: (len(x), x)):
        cells = []
        for v in ORDER:
            vals = grid[k].get(v)
            cells.append(fmt.format(agg(vals)) if vals else "-")
        print(f"{k:>6} " + " ".join(f"{c:>12}" for c in cells))


def main():
    cyc = load("cyclic.csv")
    if cyc:
        print("=" * 60)
        print("EXPERIMENT 1  3-uniform cyclic sign-alternating hypergraphs")
        print("=" * 60)
        table("Approximation ratio, mean over n and configurations, by depth",
              cyc, "p", "AR")
        table("Objective evaluations, mean, by depth", cyc, "p", "nfev", "{:.0f}")
        table("Free variational angles, by depth", cyc, "p", "angles", "{:.0f}")

        for field, label in (("optimiser", "optimiser"),
                             ("initial_angles", "initialisation")):
            print(f"\nObjective evaluations by {label} (mean over all runs)")
            grid = defaultdict(dict)
            for r in cyc:
                grid[r[field]].setdefault(r["variant"], []).append(float(r["nfev"]))
            print(f"{label:>14} " + " ".join(f"{v.upper():>10}" for v in ORDER))
            print("-" * (15 + 11 * len(ORDER)))
            for k in sorted(grid):
                print(f"{k:>14} " + " ".join(
                    f"{st.mean(grid[k][v]):>10.0f}" if v in grid[k] else f"{'-':>10}"
                    for v in ORDER))

    runs = load("random12.csv")
    if runs:
        print("\n" + "=" * 60)
        print("EXPERIMENT 2  12-qubit random-coefficient hypergraphs")
        print("=" * 60)
        n_inst = len({r["instance"] for r in runs})
        print(f"{n_inst} instances, Powell optimiser, TQA initialisation")
        table("Approximation ratio by depth", runs, "p", "AR")
        table("Probability of the optimum (%) by depth", runs, "p",
              "prob_opt_pct", "{:.2f}")
        table("Objective evaluations by depth", runs, "p", "nfev", "{:.0f}")

    magic = load("random12_magic.csv")
    if magic:
        p = max(int(r["p"]) for r in magic)
        rows = [r for r in magic if int(r["p"]) == p]
        table(f"Absolute magic M_2 by layer within a p = {p} circuit",
              rows, "layer", "M2")

        print("\nMagic barrier, that is the peak of each curve")
        peak = defaultdict(list)
        for r in rows:
            peak[(r["instance"], r["variant"])].append(float(r["M2"]))
        by_variant = defaultdict(list)
        for (inst, v), vals in peak.items():
            by_variant[v].append(max(vals))
        for v in ORDER:
            if by_variant[v]:
                print(f"   {v.upper() + '-QAOA':>10}: {st.mean(by_variant[v]):.3f}")
        lowest = defaultdict(int)
        insts = {i for i, _ in peak}
        for inst in insts:
            got = {v: max(peak[(inst, v)]) for v in ORDER if (inst, v) in peak}
            if got:
                lowest[min(got, key=got.get)] += 1
        print("   lowest barrier per instance: " +
              ", ".join(f"{v.upper()} {lowest[v]}/{len(insts)}" for v in ORDER))

    if not any((cyc, runs, magic)):
        print("No results yet. Run run_cyclic.py or run_magic12.py first.")


if __name__ == "__main__":
    main()
