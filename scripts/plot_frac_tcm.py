#!/usr/bin/env python
#
# Plots fraction of tcmalloc time compared to full simulation.
#
# Usage:
#   Plot the data for a particular optimization.
#       python plot_frac_tcm.py --db prof.db [--opt opt]

import argparse
import sqlite3
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

import db_common

FONT_SIZE = 16
FORMAT = "pdf"

def matplotlib_init():
    mpl.rcParams.update({"font.size": FONT_SIZE,
                         "mathtext.default": "regular"})
    mpl.rc("xtick", labelsize=FONT_SIZE)
    mpl.rc("ytick", labelsize=FONT_SIZE)
    mpl.rc("legend", **{"fontsize": FONT_SIZE-2})


def plot_tcm_fraction(db, opt, bmarks, suff=""):
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    runs = db_common.get_all_runs(db)

    bmarks = ["WSC (Kanev et al.)"] + bmarks
    bmarks.reverse() # so they show up correctly top-down

    all_tcm_cycles = np.zeros((len(bmarks), len(runs)))
    all_cycles = np.zeros((len(bmarks), len(runs)))

    # add motivation data
    all_tcm_cycles[-1,] = np.zeros(len(runs)) + 0.069
    all_cycles[-1,] = np.zeros(len(runs)) + 1.0

    for i, bmk in enumerate(bmarks[:-1]):
          tcm_cycles, _ = db_common.get_total_cycles_data(
                  cursor, bmk, db_common.profiled_funcs, opt)
          all_tcm_cycles[i,:] = tcm_cycles
          total_sim_cycles = db_common.get_total_sim_cycles(cursor, bmk, opt)
          all_cycles[i,:] = total_sim_cycles

    mean_tcm_cycles = np.mean(all_tcm_cycles, axis=1)
    std_tcm_cycles = np.std(all_tcm_cycles, axis=1)
    print mean_tcm_cycles

    mean_all_cycles = np.mean(all_cycles, axis=1)
    print mean_all_cycles
    std_all_cycles = np.std(all_cycles, axis=1)

    tcm_frac = 100 * mean_tcm_cycles / mean_all_cycles

    fig = plt.figure()
    ax = fig.add_subplot(111)

    bar_centers = np.arange(0, len(bmarks))
    bar_height = 0.6
    ax.barh(bar_centers, tcm_frac, label=bmarks, height=bar_height,
            color="#93160d", alpha=0.85,
            edgecolor="#333333")

    ax.set_yticks(bar_centers+bar_height/2)
    labels = [db_common.benchmark_labels[b] for b in bmarks]
    ax.set_yticklabels(labels)

    delta_y = (1.0 - bar_height) / 2
    ax.set_ylim([-delta_y, len(bmarks) - delta_y])

    if (suff != "all"):
        ax.set_xlim([0, 8])

    # Annotate masstree.same.
    if suff == "large":
        idx = bmarks.index("masstree.same")
        masstree_frac = tcm_frac[idx]
        ax.text(ax.get_xlim()[1] + 0.25, idx, "%2.1f%%" % masstree_frac)

    ax.set_xlabel("Time spent in tcmalloc (%)")
    ax.grid(axis="x", ls="-", color="#cccccc")
    ax.set_axisbelow(True)

    lim = (ax.get_xlim()[1] - ax.get_xlim()[0]) /\
          (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.set_aspect(0.6 * lim)

    fig.tight_layout()
    fname = "../graphs_raw/tcm_frac"
    if suff:
        fname += "_" + suff
    fname += "." + FORMAT
    plt.savefig(fname, bbox_inches="tight", dpi=200)
    plt.close()

    conn.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", help="SQLite3 DB.")
    parser.add_argument("--opt", default="baseline", help="Optimization to plot.")
    args = parser.parse_args()

    matplotlib_init()
    all_bmarks, _ = db_common.get_all_benchmarks(args.db)
    plot_tcm_fraction(args.db, args.opt, all_bmarks, "all")
    large_bmarks, _ = db_common.get_large_benchmarks(args.db)
    plot_tcm_fraction(args.db, args.opt, large_bmarks, "large")

if __name__ == "__main__":
    main()
