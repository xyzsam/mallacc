#!/usr/bin/env python
#
# Plots speedup from the instruction ignoring limit study.
#
# Example usage to plot the results (after creating the DB):
#    ./plot_ideal_speedup.py plot --db results.db --slice [optional sim slice number]

import argparse
import matplotlib
import numpy as np
import re
import sqlite3
import os
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from db_common import *

def plot_speedup(db, opt, slice_no=None):
    """ Plots speedup for the limit study. """
    # XXX: this needs updating to the new sim_stats table schema
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    fdo = opt.endswith("fdo")
    baseline = "baseline-fdo" if fdo else "baseline"

    if slice_no == None:
        pop_query = ("select benchmark, sum(sim_cycle), sum(all_insn) from %s where "
                     "optimization=\"%s\" and benchmark like \"ubench.%%\" "
                     "group by benchmark order by benchmark") % (SIM_STATS_TABLE_NAME, opt)
    else:
        pop_query = ("select benchmark, sim_cycle, all_insn from %s where "
                     "optimization=\"%s\" and slice=%d and benchmark like \"ubench.%%\" "
                     "order by benchmark") % (SIM_STATS_TABLE_NAME, opt, slice_no)
    print pop_query

    cursor.execute(pop_query)
    results = cursor.fetchall()

    benchmarks = []
    pop_cycles = []
    pop_insns = []
    for row in results:
        benchmarks.append(row[0])
        pop_cycles.append(row[1])
        pop_insns.append(row[2])
    pop_cycles = np.array(pop_cycles).astype(float)
    pop_insns = np.array(pop_insns).astype(float)

    if slice_no == None:
        baseline_query = ("select benchmark, sum(sim_cycle), sum(all_insn) from %s where "
                          "optimization=\"%s\" and benchmark like \"ubench.%%\" "
                          "group by benchmark order by benchmark") % (SIM_STATS_TABLE_NAME, baseline)
    else:
        baseline_query = ("select benchmark, sim_cycle, all_insn from %s where "
                          "optimization=\"%s\" and slice=%d and "
                          "benchmark like \"ubench.%%\" order by benchmark") % (SIM_STATS_TABLE_NAME, baseline, slice_no)
    cursor.execute(baseline_query)
    results = cursor.fetchall()

    baseline_cycles = []
    baseline_insns = []
    for row in results:
        baseline_cycles.append(row[1])
        baseline_insns.append(row[2])
    baseline_cycles = np.array(baseline_cycles).astype(float)
    baseline_insns = np.array(baseline_insns).astype(float)

    # Compute the metric to plot.
    # speedup = np.divide(baseline_cycles - pop_cycles, baseline_cycles) * 100
    speedup = np.divide(baseline_cycles, pop_cycles)
    # insn_diff = (baseline_insns - pop_insns)/12.0
    value = speedup
    print benchmarks
    print value

    max_y = max(value)
    min_y = min(value)

    benchmarks = [b.encode("utf-8") for b in benchmarks]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    bar_centers = np.arange(0, len(benchmarks))
    bar_width = 0.3

    ax.bar(bar_centers, value, width=bar_width)
    ax.set_xticks(bar_centers+bar_width/2)
    ax.set_xticklabels(benchmarks)
    ax.set_xlim(left=-bar_width, right=len(benchmarks)-bar_width)
    ax.set_ylim(bottom=min_y-0.1, top=max_y+0.1)
    ax.set_ylabel("Speedup")
    ax.grid(axis="y", ls="-", color="#cccccc")
    ax.set_axisbelow(True)

    fig.tight_layout()
    plt.savefig("../graphs_raw/speedup_%s.pdf" % opt, bbox_inches="tight")
    plt.close()

    conn.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", help="DB to read from or write to.")
    parser.add_argument("--slice", type=int, help="Which slice to plot.")
    parser.add_argument("--opt", default="pop", help="Which optimization to plot.")
    args = parser.parse_args()

    if args.db is None:
        print "Need to specify DB."
        exit(1)
    plot_speedup(args.db, args.opt, args.slice)


if __name__ == "__main__":
    main()
