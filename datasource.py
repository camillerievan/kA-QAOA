"""Where results come from, and where they go.

A *run* is a named folder under results/ holding the CSVs from one execution.
The results reported in the manuscript carry the suffix _published,
and are split one per family of figures:

    results/
        cyclic_published/       3-uniform cyclic, n = 4 to 15
            cyclic.csv
            SOURCE.txt
        path_published/         3-uniform path, n = 4 to 14
            cyclic.csv
            SOURCE.txt
        magic_published/        those same 70 instances, with M_2
            random12.csv
            random12_magic.csv
            SOURCE.txt
        my_sweep/               anything you generate yourself
            random12.csv
            random12_magic.csv
            cyclic.csv

Everything is plain CSV, so no database is needed to reproduce any figure
in the paper: run the scripts, pick a run, plot it.
"""
from __future__ import annotations

import csv
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")



# ---------------------------------------------------------------------------
# run names
# ---------------------------------------------------------------------------

def sanitise(name: str) -> str:
    """A run name safe to use as a folder name."""
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip()).strip("._-")
    return clean or "unnamed"


def ask_run_name(default: str | None = None) -> str:
    """Prompt for a name to file this run under."""
    hint = f" [{default}]" if default else ""
    raw = input(f"Name for this test run{hint}: ").strip()
    if not raw and default:
        raw = default
    return sanitise(raw)


def ask_run_size(sample: str, full: str) -> bool:
    """Ask whether to do a quick sample or the full sweep. True means full.

    `sample` and `full` describe each option, e.g. "3 instances, p = 1..2,
    about a minute".
    """
    print("\nHow much do you want to run?\n")
    print(f"  1  sample   {sample}")
    print(f"  2  full     {full}")
    while True:
        raw = input("\n> ").strip().lower()
        if raw in ("1", "s", "sample", ""):
            return False
        if raw in ("2", "f", "full"):
            return True
        print("Please answer 1 for a sample or 2 for the full run.")


def run_dir(name: str, create: bool = False) -> str:
    path = os.path.join(RESULTS, sanitise(name))
    if create:
        os.makedirs(path, exist_ok=True)
    return path


# the runs that back the figures in the manuscript, in the order the paper
# presents them; anything else is listed after, newest first
MANUSCRIPT = ("cyclic_published", "path_published", "magic_published")

DESCRIPTION = {
    "cyclic_published": "3-uniform cyclic hypergraphs, n = 4 to 15",
    "path_published":   "3-uniform path hypergraphs, n = 4 to 14",
    "magic_published":  "those same instances with their magic, per layer",
}


def list_runs() -> list[str]:
    """Named runs on disk, the manuscript ones first, then newest first."""
    runs = []
    if os.path.isdir(RESULTS):
        for entry in os.scandir(RESULTS):
            if entry.is_dir() and any(f.endswith(".csv") for f in os.listdir(entry.path)):
                runs.append((entry.stat().st_mtime, entry.name))
    known = [n for n in MANUSCRIPT if n in {r[1] for r in runs}]
    rest = [n for _, n in sorted(runs, reverse=True) if n not in known]
    return known + rest


# ---------------------------------------------------------------------------
# reading a run
# ---------------------------------------------------------------------------

def read_csv(run: str, filename: str) -> list[dict]:
    path = os.path.join(run_dir(run), filename)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# one entry point for both sources
# ---------------------------------------------------------------------------

def available_sources() -> list[str]:
    return list_runs()


FILES = {"runs": "random12.csv",
         "magic": "random12_magic.csv",
         "cyclic": "cyclic.csv",
         "random_pm1": "random_pm1.csv"}


def load(source: str, what: str) -> list[dict]:
    """Rows from one run. `what` is 'runs', 'magic' or 'cyclic'."""
    return read_csv(source, FILES[what])


def choose_source(prompt: str = "Which test run?") -> str | None:
    """Menu of available runs."""
    sources = available_sources()
    if not sources:
        print("No results yet. Run run_cyclic.py or run_magic12.py first.")
        return None
    if len(sources) == 1:
        print(f"Using the only run available: {sources[0]}")
        return sources[0]

    print(f"\n{prompt}\n")
    width = max(len(s) for s in sources)
    for i, s in enumerate(sources, start=1):
        tag = DESCRIPTION.get(s, "")
        print(f"  {i}  {s:<{width}}" + (f"   {tag}" if tag else ""))
    print("  0  cancel")
    raw = input("\n> ").strip()
    if raw in ("0", "q", ""):
        return None
    if raw.isdigit() and 1 <= int(raw) <= len(sources):
        return sources[int(raw) - 1]
    if raw in sources:
        return raw
    print(f"'{raw}' is not one of the options.")
    return choose_source(prompt)


if __name__ == "__main__":
    print("results folder:", RESULTS)
    for src in available_sources():
        bits = []
        for what in ("runs", "magic", "cyclic", "random_pm1"):
            rows = load(src, what)
            if rows:
                bits.append(f"{FILES[what]} {len(rows)} rows")
        print(f"  {src:<20} " + (", ".join(bits) or "(empty)"))
