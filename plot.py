"""Plot the results in the style of the figures in the paper.

The menu numbers are fixed, so a number always means the same figure:

    1  Approximation ratio      AR = <C> / C_opt
    2  Fidelity                 probability of measuring the optimal state
    3  Objective evaluations    nfev
    4  Magic, normalised        M~_2, each instance divided by its own maximum
    5  Magic, absolute          M_2 in its own units
    6  all of them

Entries a run has no data for are simply absent from its menu.

Which axis a figure uses follows the shape of the data, as in the paper:

    cyclic, path  many problem sizes  ->  metric against n, one figure per p
    random        a single size       ->  metric against p
    magic         per-layer M_2       ->  magic against layer

Run with no arguments for a menu, or name a metric directly:

    python plot.py                            # interactive
    python plot.py ar --run random_published  # one figure
    python plot.py all --run cyclic_published --p 1 --save
    python plot.py stacked --run path_published   # three-panel layout

Each series shows the mean over instances, a shaded bar of one standard
deviation, and a thin whisker spanning min to max, with the variants offset
horizontally so they do not overlap -- the convention used in the paper.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import statistics as st
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import datasource as ds

# variant -> (label, colour, marker, horizontal offset), matching the paper
STYLE = {
    "ma": ("MA-QAOA",   "tab:red",    "s", 0.00),
    "ka": ("$k$A-QAOA", "tab:orange", "o", 0.12),
    "am": ("AA-QAOA",   "tab:green",  "^", 0.24),
    "sa": ("SA-QAOA",   "tab:blue",   "D", 0.36),
}
# the continuous-depth study compares a setting, not an angle strategy, so it
# brings its own two series
STYLE.update({
    "cd_on":  ("QAOA-CD",     "tab:purple", "o", 0.00),
    "cd_off": ("QAOA, fixed", "tab:brown",  "s", 0.18),
})
ORDER = ("ma", "ka", "am", "sa", "cd_on", "cd_off")

METRICS = {
    "stacked":    ("AR, fidelity and nfev, stacked in one figure", ""),
    "magic_norm": ("Magic (normalised)", r"$\tilde{M}_2$"),
    "magic_abs":  ("Magic (absolute)",   r"$M_2$"),
    "ar":         ("Approximation ratio", "AR"),
    "fidelity":   ("Fidelity",           "$F$"),
    "nfev":       ("Objective evaluations (nfev)", "nfev"),
}
# names accepted on the command line (the menu uses the fixed slots below)
ALIASES = {"magic": "magic_norm", "m": "magic_norm", "f": "fidelity",
           "n": "nfev", "paper": "stacked", "panels": "stacked"}





def draw(series, xlabel, ylabel, title, origin=False, logy=False, bounds=None):
    """series: {variant: {x: [values over instances]}}

    `bounds` is the range the quantity can take, if it is bounded. The spread
    bars are clipped to it, so a fidelity never appears to exceed certainty.
    """
    fig, ax = plt.subplots(figsize=(6.2, 3.6))

    def clip(v):
        """Hold a spread bar inside the range, ignoring either end left open."""
        if bounds is None:
            return v
        lo, hi = bounds
        if lo is not None:
            v = max(v, lo)
        if hi is not None:
            v = min(v, hi)
        return v

    for variant in ORDER:
        if variant not in series or not series[variant]:
            continue
        label, colour, marker, dx = STYLE[variant]
        xs = sorted(series[variant])
        if origin:
            xs = [0] + xs

        mean, lo, hi, sd = [], [], [], []
        for x in xs:
            vals = series[variant].get(x, [0.0]) if x != 0 or not origin else [0.0]
            if x == 0 and origin:
                vals = [0.0]
            mean.append(st.mean(vals))
            lo.append(min(vals))
            hi.append(max(vals))
            sd.append(st.pstdev(vals) if len(vals) > 1 else 0.0)

        shifted = [x + dx for x in xs]

        # min-max whisker, thin and black, as in the paper
        for x, a, b in zip(shifted, lo, hi):
            if b > a:
                ax.plot([x, x], [a, b], color="black", lw=0.8, zorder=2)
        # one standard deviation, thick and translucent
        for x, m, s in zip(shifted, mean, sd):
            if s > 0:
                ax.plot([x, x], [clip(m - s), clip(m + s)], color=colour, lw=7,
                        alpha=0.30, solid_capstyle="butt", zorder=1)

        ax.plot(shifted, mean, marker=marker, color=colour, label=label,
                lw=1.4, ms=5, zorder=3)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if title:
        ax.set_title(title, fontsize=10)
    if logy:
        ax.set_yscale("log")
    ax.grid(alpha=0.35, ls=":")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# one builder per metric
# ---------------------------------------------------------------------------

def build_magic(source, normalised: bool):
    rows = ds.load(source, "magic")
    if not rows:
        print(f"No per-layer magic data in '{source}'.")
        return None
    depth = max(int(r["p"]) for r in rows)
    rows = [r for r in rows if int(r["p"]) == depth]

    # per instance, the maximum over every variant and layer: the normaliser
    inst_max = defaultdict(float)
    for r in rows:
        inst_max[r["instance"]] = max(inst_max[r["instance"]], float(r["M2"]))

    series = defaultdict(lambda: defaultdict(list))
    for r in rows:
        v = float(r["M2"])
        if normalised:
            m = inst_max[r["instance"]]
            v = v / m if m > 0 else 0.0
        series[r["variant"]][int(r["layer"])].append(v)

    n_inst = len({r["instance"] for r in rows})
    key = "magic_norm" if normalised else "magic_abs"
    noun = "instance" if n_inst == 1 else "instances"
    # x is the layer reached inside the deepest circuit, which is the depth of
    # the state being measured, so it carries the same label p as everywhere else
    return draw(series, "$p$", METRICS[key][1],
                f"{METRICS[key][0]}, {n_inst} {noun}, depth $p={depth}$  [{source}]",
                origin=True, bounds=(0.0, 1.0) if normalised else (0.0, None))


def run_rows(source):
    """The per-run rows of a source, whichever file they live in."""
    return (ds.load(source, "runs") or ds.load(source, "cyclic")
            or ds.load(source, "random_pm1") or ds.load(source, "qaoacd"))


def sizes(source):
    """The distinct problem sizes n present in a source."""
    return sorted({int(r["n"]) for r in run_rows(source) if r.get("n")})


def depths(source):
    """The distinct circuit depths p present in a source."""
    return sorted({int(r["p"]) for r in run_rows(source) if r.get("p")})


# how each metric is drawn, depending on which axis it is plotted against.
# Against n the paper uses a percentage fidelity and a linear nfev axis in
# thousands; against p it uses a fraction and a logarithmic axis.
AXIS_STYLE = {
    ("ar", "n"):       dict(bounds=(0.0, 1.0)),
    ("ar", "p"):       dict(bounds=(0.0, 1.0)),
    ("fidelity", "n"): dict(ylabel="$F$", bounds=(0.0, 100.0)),
    ("fidelity", "p"): dict(ylabel="$F$", bounds=(0.0, 100.0)),
    # nfev is drawn as the count itself, on a linear axis in thousands, which
    # is how the dissertation presents it; no logarithm is taken anywhere
    ("nfev", "n"):     dict(scale=1e-3, ylabel=r"nfev ($\times 10^3$)",
                            bounds=(0.0, None)),
    ("nfev", "p"):     dict(scale=1e-3, ylabel=r"nfev ($\times 10^3$)",
                            bounds=(0.0, None)),
}


def build_from_runs(source, column, key, depth=None):
    """One figure per depth against n, or a single figure against p.

    A source that sweeps several problem sizes (the cyclic and path families)
    is plotted as the metric against n, one figure per depth p, which is the
    layout used in the paper. A source at a single problem size (the random
    coefficient hypergraphs) is plotted against p instead.
    """
    rows = run_rows(source)
    if not rows:
        print(f"No run results in '{source}'.")
        return []
    rows = [r for r in rows if r.get(column) not in (None, "")]
    if not rows:
        print(f"'{source}' has no {column} values.")
        return []

    all_n = sorted({int(r["n"]) for r in rows if r.get("n")})
    axis = "n" if len(all_n) > 1 else "p"
    style = AXIS_STYLE.get((key, axis), {})
    scale = style.get("scale", 1.0)
    logy = style.get("logy", False)
    bounds = style.get("bounds")
    ylabel = style.get("ylabel", METRICS[key][1])

    if axis == "p":
        series = defaultdict(lambda: defaultdict(list))
        for r in rows:
            series[r["variant"]][int(r["p"])].append(float(r[column]) * scale)
        n_inst = len({r.get("instance") or r.get("n") for r in rows})
        what = "instance" if n_inst == 1 else "instances"
        title = (f"{METRICS[key][0]}, $n = {all_n[0]}$, "
                 f"{n_inst} {what}  [{source}]")
        return [("", draw(series, "$p$", ylabel, title, logy=logy,
                          bounds=bounds))]

    # one figure per depth, the metric against n
    wanted = depth if depth else sorted({int(r["p"]) for r in rows})
    figures = []
    for p in wanted:
        at_p = [r for r in rows if int(r["p"]) == p]
        if not at_p:
            print(f"'{source}' has no rows at depth p = {p}.")
            continue
        series = defaultdict(lambda: defaultdict(list))
        for r in at_p:
            series[r["variant"]][int(r["n"])].append(float(r[column]) * scale)
        title = f"{METRICS[key][0]}, $p = {p}$  [{source}]"
        figures.append((f"_p{p}", draw(series, "$n$", ylabel, title,
                                      logy=logy, bounds=bounds)))
    return figures


def build_stacked(source, only_n=None):
    """Three stacked panels: approximation ratio, fidelity, nfev, against p.

    This is the layout used for the hypergraph families in the paper. Magic is
    not included: those datasets do not carry it.
    """
    rows = ds.load(source, "runs") or ds.load(source, "cyclic")
    if not rows:
        print(f"No run results in '{source}'.")
        return None

    sizes = sorted({int(r["n"]) for r in rows if r.get("n")})
    if only_n is not None:
        rows = [r for r in rows if int(r["n"]) == only_n]
        if not rows:
            print(f"'{source}' has no rows with n = {only_n}. Available: {sizes}")
            return None

    panels = [("AR", "AR", "(a) Approximation ratio", 1.0, False),
              ("prob_opt_pct", "$F$", "(b) Fidelity", 1.0, False),
              ("nfev", r"nfev ($\times 10^3$)",
               "(c) Number of function evaluations", 1e-3, False)]

    fig, axes = plt.subplots(3, 1, figsize=(5.4, 9.2))
    for ax, (column, ylabel, caption, scale, logy) in zip(axes, panels):
        usable = [r for r in rows if r.get(column) not in (None, "")]
        series = defaultdict(lambda: defaultdict(list))
        for r in usable:
            series[r["variant"]][int(r["p"])].append(float(r[column]) * scale)

        for variant in ORDER:
            if variant not in series:
                continue
            label, colour, marker, dx = STYLE[variant]
            xs = sorted(series[variant])
            mean = [st.mean(series[variant][x]) for x in xs]
            lo = [min(series[variant][x]) for x in xs]
            hi = [max(series[variant][x]) for x in xs]
            sd = [st.pstdev(series[variant][x]) if len(series[variant][x]) > 1 else 0.0
                  for x in xs]
            shifted = [x + dx for x in xs]
            for x, a, b in zip(shifted, lo, hi):
                if b > a:
                    ax.plot([x, x], [a, b], color="black", lw=0.8, zorder=2)
            for x, m, sv in zip(shifted, mean, sd):
                if sv > 0:
                    ax.plot([x, x], [m - sv, m + sv], color=colour, lw=7,
                            alpha=0.30, solid_capstyle="butt", zorder=1)
            ax.plot(shifted, mean, marker=marker, color=colour, label=label,
                    lw=1.3, ms=4.5, zorder=3)

        ax.set_xlabel("$p$")
        ax.set_ylabel(ylabel)
        ax.set_title(caption, fontsize=9, color="darkgreen", y=-0.34)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if logy:
            ax.set_yscale("log")
        ax.grid(alpha=0.35, ls=":")

    axes[0].legend(loc="lower right", fontsize=8)
    scope = (f"$n = {only_n}$" if only_n is not None
             else (f"$n = {sizes[0]}$" if len(sizes) == 1
                   else f"$n = {sizes[0]}$ to ${sizes[-1]}$"))
    fig.suptitle(f"{scope}, {len({r.get('instance', '') for r in rows})} graphs"
                 f"  [{source}]", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.subplots_adjust(hspace=0.55)
    return fig


BUILDERS = {
    "stacked":    lambda src, p: [("", build_stacked(src))],
    "magic_norm": lambda src, p: [("", build_magic(src, True))],
    "magic_abs":  lambda src, p: [("", build_magic(src, False))],
    "ar":         lambda src, p: build_from_runs(src, "AR", "ar", p),
    "fidelity":   lambda src, p: build_from_runs(src, "prob_opt_pct", "fidelity", p),
    "nfev":       lambda src, p: build_from_runs(src, "nfev", "nfev", p),
}


# fixed menu slots, so a number always means the same figure
SLOTS = [("1", "ar"), ("2", "fidelity"), ("3", "nfev"),
         ("4", "magic_norm"), ("5", "magic_abs")]
ALL_SLOT = "6"


# the column each run-based metric needs; a run that recorded nothing in it
# does not offer that figure
NEEDS = {"ar": "AR", "fidelity": "prob_opt_pct", "nfev": "nfev"}


def available_metrics(source):
    """Only the figures this run actually has the data for, in menu order."""
    rows = run_rows(source)
    has_magic = bool(ds.load(source, "magic"))
    keys = []
    for _, key in SLOTS:
        if key in ("magic_norm", "magic_abs"):
            if has_magic:
                keys.append(key)
        elif any(r.get(NEEDS[key]) not in (None, "") for r in rows):
            keys.append(key)
    return keys


def ask_depth(source):
    """Which depth to plot against n. Only asked of multi-size sources."""
    available = depths(source)
    if len(sizes(source)) <= 1 or not available:
        return None                       # plotted against p, no choice needed

    span = ", ".join(str(p) for p in available)
    raw = input(f"\nWhich depth p? [{span}, or a for all] ").strip().lower()
    if raw in ("a", "all", ""):
        return available
    chosen = [int(t) for t in re.findall(r"\d+", raw) if int(t) in available]
    if chosen:
        return chosen
    print(f"'{raw}' is not one of the depths available.")
    return ask_depth(source)


def menu(source):
    usable = available_metrics(source)
    if not usable:
        print(f"'{source}' holds no plottable data.")
        return []

    print("\nWhich figure?\n")
    for slot, key in SLOTS:
        if key in usable:
            print(f"  {slot}  {METRICS[key][0]}")
    print(f"  {ALL_SLOT}  all of them, each as its own figure")
    print("  0  quit")

    choice = input("\n> ").strip().lower()
    if choice in ("0", "q", ""):
        return []
    if choice in (ALL_SLOT, "all"):
        return usable

    key = dict(SLOTS).get(choice) or ALIASES.get(choice, choice)
    if key in usable:
        return [key]
    if key in METRICS:
        print(f"'{choice}' needs data that '{source}' does not have.")
    else:
        print(f"'{choice}' is not one of the options.")
    return menu(source)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("metric", nargs="?", help="magic_norm, magic_abs, ar, fidelity, nfev, all")
    ap.add_argument("--save", action="store_true", help="write PNGs instead of showing them")
    ap.add_argument("--run", help="which test run to plot (prompted for if omitted)")
    ap.add_argument("--p", type=int, nargs="+", metavar="P",
                    help="depth(s) to plot against n; only for runs that sweep n")
    args = ap.parse_args()

    source = args.run or ds.choose_source("Which test run do you want to plot?")
    if source is None:
        return

    if args.metric:
        key = ALIASES.get(args.metric.lower(), args.metric.lower())
        usable = available_metrics(source)
        if usable and "stacked" not in usable:
            usable = usable + ["stacked"]   # command line only, not in the menu
        keys = [k for k in usable if k != "stacked"] if key == "all" else [key]
        if key != "all" and key not in METRICS:
            sys.exit(f"unknown metric '{args.metric}'. "
                     f"Choose from: {', '.join(METRICS)}, all")
        if key != "all" and key not in usable:
            sys.exit(f"'{key}' needs data that '{source}' does not have. "
                     f"Available here: {', '.join(usable)}")
        depth = args.p or (depths(source) if len(sizes(source)) > 1 else None)
    else:
        keys = menu(source)
        if not keys:
            return
        depth = args.p or ask_depth(source)

    for key in keys:
        for suffix, fig in BUILDERS[key](source, depth):
            if fig is None:
                continue
            if args.save:
                out = os.path.join(ds.run_dir(source, create=True),
                                   f"fig_{key}{suffix}.png")
                fig.savefig(out, dpi=200)
                plt.close(fig)
                print(f"wrote {out}")
        if not args.save:
            plt.show()


if __name__ == "__main__":
    main()
