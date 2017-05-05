#!/usr/bin/env python
#
# Plots simulation profiling results on tcmalloc.
#
# Usage:
#   Plot the data for a particular optimization.
#       python plot_profiling_data.py plot-[type_of_plot] --db prof.db --opt [opt]
#
#   Scripts should be run from the scripts/ directory.  Depending on your
#   directory structure, you may need to modify parse_profiling_filepath() to return the
#   correct metadata fields.

import argparse
import cPickle as pickle
import math
import re
import sqlite3
import os
import warnings
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from colormap_helper import get_colormap
from fake_semilogx import *
from db_common import *
import scipy.stats as st

opt_labels = {
    "baseline": "Baseline",
    "ideal-sampling": "Ideal sampling",
    "real-sampling": "Real sampling",
    "ideal-size-class": "Ideal size class",
    "real-size-class": "Real size class",
    "ideal-list": "Ideal list push/pop",
    "real-list": "Real list push/pop with size class",
    "real-list-only": "Real list push/pop",
    "realistic": "All optimizations",
    "limit": "Limit study",
}

GnBu = get_colormap(plt.cm.GnBu, 5)

FONT_SIZE = 16
FORMAT = "pdf"

def matplotlib_init():
    mpl.rcParams.update({"font.size": FONT_SIZE,
                         "mathtext.default": "regular",
                         "pdf.fonttype": 42,
                         "ps.fonttype": 42})
    mpl.rc("xtick", labelsize=FONT_SIZE)
    mpl.rc("ytick", labelsize=FONT_SIZE)
    mpl.rc("legend", **{"fontsize": FONT_SIZE-2})


def get_pretty_opt_label(opt_name):
    """ Converts a raw optimization label to a pretty format for plotting. """
    result = []
    for opt, pretty in opt_labels.iteritems():
        if opt in opt_name:
            result.append(pretty)
    if len(pretty) == 0:
        raise ValueError("Could not reformat %s to a pretty format." % opt_name)
    return ", ".join(result)

def plot_baseline(db, bmarks, baseline="baseline", suff="", cdf=False):
    """ Plots the baseline malloc and free profiling results. """
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    plots = {"malloc": malloc_funcs,
             "free": free_funcs}
    min_logx = 1
    max_logx = 5
    cm = get_colormap(plt.cm.Paired, 10)

    picklefile_name = "cdf_hists.pkl"
    picklefile = None
    read_picklefile = os.path.exists(picklefile_name)
    if not read_picklefile:
        picklefile = open("cdf_hists.pkl", "w")
        pickled_data = {}
    else:
        picklefile = open("cdf_hists.pkl", "r")
        pickled_data = pickle.load(picklefile)
    for profile_label, func_list in plots.iteritems():
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if not read_picklefile:
            pickled_data[profile_label] = {}
        for i, bmark in enumerate(bmarks):
            if not read_picklefile:
                baseline_data = get_cycles_data(cursor, bmark, func_list, baseline)
                weights = 100.0 * baseline_data / (sum(baseline_data))
                baseline_hist, baseline_edges = np.histogram(baseline_data,
                        bins=np.logspace(min_logx, max_logx, (max_logx-min_logx+1)*10),
                        weights=weights)
                pickled_data[profile_label][bmark] = (baseline_hist, baseline_edges)
            else:
                baseline_hist, baseline_edges = pickled_data[profile_label][bmark]
            # we do a semi-log plot manually, so we can set the aspect ratio
            # and keep consistent font sizes
            if bmark.startswith("4"):
                linestyle = "solid"
            elif bmark.startswith("xapian"):
                linestyle = "dashed"
            elif bmark.startswith("masstree"):
                linestyle = "solid"
            else:
                linestyle = "dotted"
            if cdf:
                baseline_cdf = np.cumsum(baseline_hist).astype(float)
                if bmark.startswith("masstree"):
                    plot_semilogx(ax, baseline_edges[::3], baseline_cdf[::3],
                                  linewidth=0, c=cm[i], marker="o",
                                  label=benchmark_labels[bmark])
                    plot_semilogx(ax, baseline_edges[:-1], baseline_cdf,
                                  linewidth=2, linestyle=linestyle, c=cm[i])
                else:
                    plot_semilogx(ax, baseline_edges[:-1], baseline_cdf,
                                  linewidth=2, linestyle=linestyle, c=cm[i],
                                  label=benchmark_labels[bmark])

            else:
                plot_semilogx(ax, baseline_edges[:-1], baseline_hist,
                              linewidth=2, linestyle=linestyle, c=cm[i],
                              label=benchmark_labels[bmark])

        if cdf:
            plt.vlines(np.log10(100), 0, 100, color="#222222", linestyles="dashed", linewidth=2)
            plt.text(np.log10(30), 9, "fast path",
                     fontsize=FONT_SIZE-4, color="#222222", backgroundcolor="white")
            plt.annotate("",
                         xy=(np.log10(30), 5), xycoords="data",
                         xytext=(np.log10(80), 5), textcoords="data",
                         arrowprops=dict(arrowstyle="-|>",
                                         facecolor="w",
                                         edgecolor="#222222"))

        ax.set_ylim(bottom=0)
        if cdf:
            ax.set_ylim(top=100)
        plt.minorticks_on()
        set_axis_limits(ax, max_logx, min_logx)

        lim = (ax.get_xlim()[1] - ax.get_xlim()[0]) /\
              (ax.get_ylim()[1] - ax.get_ylim()[0])
        ax.set_aspect(0.5 * lim)

        ax.grid(axis="y", ls="-", color="#cccccc")
        ax.set_axisbelow(True)
        ax.set_xlabel("%s duration (cycles)" % profile_label)

        if cdf:
            ax.set_ylabel("Time in calls (CDF %)")
            legend = ax.legend(loc="lower right", ncol=2, handlelength=2, fontsize=12)
            fprefix = "cdf"
        else:
            ax.set_ylabel("Time in calls (PDF %)")
            legend = ax.legend(loc="upper right", ncol=2)
            fprefix = "pdf"
        legend.get_frame().set_edgecolor('w')
        for legobj in legend.legendHandles:
            legobj.set_linewidth(10)
        fname = "../graphs_raw/%s_%s%s.%s" % (fprefix, profile_label, suff, FORMAT)
        plt.savefig(fname, bbox_inches="tight", dpi=200)
        plt.close()
    conn.close()

    if picklefile:
        if not read_picklefile:
            pickle.dump(pickled_data, picklefile)
        picklefile.close()

def plot_baseline_vs_opt_profiling(db, base_opt, baseline="baseline"):
    """ Plot the CDFs for an optimization and the baseline of a benchmark. """
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    bmarks, _ = get_all_benchmarks(db)
    plots = {"malloc": malloc_funcs,
             "free": free_funcs}

    cm = get_colormap(plt.cm.inferno, 5)

    min_logx = 0
    max_logx = 5

    for bmark in bmarks:
        print "Plotting %s" % bmark
        for profile_label, func_list in plots.iteritems():
            print "  Plotting %s" % profile_label
            fig = plt.figure()
            ax = fig.add_subplot(111)

            # If opt is in opt_labels, then just plot that opt. Otherwise, try
            # ideal-opt and real-opt. If either of them are not in opt_labels,
            # throw an error.
            all_opts = [baseline]
            if base_opt in opt_labels:
                all_opts.append(base_opt)
            elif base_opt == "all":
                all_opts += ["limit", "realistic"]
            else:
                ideal_opt = "ideal-" + base_opt
                real_opt = "real-" + base_opt
                if ideal_opt in opt_labels:
                    all_opts.append(ideal_opt)
                if real_opt in opt_labels:
                    all_opts.append(real_opt)
                if len(all_opts) == 1:
                    raise ValueError("% is not a known optimization." % base_opt)

            maxy = 0
            for i, opt in enumerate(all_opts):
                data = get_cycles_data(cursor, bmark, func_list, opt)
                if len(data) == 0:
                    continue

                weights = 100.0 * data / (sum(data))
                hist, edges = np.histogram(
                        data, range=(1, max(data)),
                        bins=np.logspace(min_logx, max_logx, (max_logx-min_logx+1)*10),
                        weights=weights)
                cdf = np.cumsum(hist).astype(float)
                opt_label = get_pretty_opt_label(opt)
                plot_semilogx(ax, edges[:-1], hist,
                              linewidth=2, color=cm[i+1],
                              label="%s %s" % (opt_label, profile_label))
                maxy = max(maxy, max(hist))

            ylim = min(100, np.floor((maxy + 10) / 10) * 10)
            plt.minorticks_on()
            ax.set_ylim([0, ylim])
            ax.set_xlabel("call duration (cycles)")
            ax.set_ylabel("Time in calls (PDF %)")
            ax.set_title(bmark)

            set_axis_limits(ax, max_logx, min_logx)

            lim = (ax.get_xlim()[1] - ax.get_xlim()[0]) /\
                  (ax.get_ylim()[1] - ax.get_ylim()[0])
            ax.set_aspect(0.5 * lim)

            legend = ax.legend(loc=0)
            if legend:
                legend.get_frame().set_edgecolor('w')
            fname = "../graphs_raw/%s_%s_diff_pdf_%s.%s" % (bmark, profile_label, base_opt, FORMAT)
            plt.savefig(fname, bbox_inches="tight", dpi=200)
            plt.close()

    conn.close()

def plot_ideal_speedup(db, bmarks_, opts, baseline="baseline", suff="_large", malloc_only=False, free_only=False):
    """ Plot ideal speedup using profiling data. """
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    bmarks = list(bmarks_)
    bmarks.reverse() # so they show up correctly top-down
    runs = get_all_runs(db)

    if malloc_only:
        funcs = malloc_funcs
        suff += "_malloc"
    elif free_only:
        funcs = free_funcs
        suff += "_free"
    else:
        funcs = profiled_funcs

    COLORS = ["#93160d", "#86a6d5"]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    all_data = []
    all_labels = []

    for bmark in bmarks:
        bmark_data = []
        for opt in opts:
            baseline_run_data = []
            opt_run_data = []
            for run in runs:
                print "Getting data for", bmark, opt, run
                baseline_cycles, _ = get_total_cycles_data(cursor, bmark, funcs, baseline, run)
                opt_cycles, _ = get_total_cycles_data(cursor, bmark, funcs, opt, run)
                if baseline_cycles == 0 or opt_cycles == 0:
                    continue
                baseline_run_data.append(baseline_cycles)
                opt_run_data.append(opt_cycles)

            baseline_sum = np.nansum(baseline_run_data)
            baseline_mean = np.nanmean(baseline_run_data)
            opt_sum = np.nansum(opt_run_data)
            opt_mean = np.nanmean(opt_run_data)

            speedup = baseline_sum / opt_sum
            std = (np.nanstd(baseline_run_data)/baseline_mean + np.nanstd(opt_run_data)/opt_mean) * speedup
            speedup *= 100  # pct
            std *= 100  # pct

            # if bmark == "xapian.query_wiki_pages":
            #     std = 0.0 # XXX: FIX HOW I MERGED XAPIAN IN THE DB

            bmark_data.append([speedup, std])

        all_data.append(bmark_data)
        all_labels.append(benchmark_labels[bmark])

    all_data = np.array(all_data)

    all_speedup = all_data[:,:,0]
    all_std = all_data[:,:,1]
    max_x = np.max(all_speedup)
    min_x = np.min(all_speedup)

    # Compute geomean speedup.
    geomean_speedup = st.mstats.gmean(all_speedup, axis=0)
    avg_std = np.mean(all_std, axis=0)
    all_speedup = np.append([geomean_speedup], all_speedup, axis=0)
    all_std = np.append([avg_std], all_std, axis=0)
    all_labels.insert(0, "Geomean")

    all_speedup = all_speedup - 100  # in units of percent spd

    bar_centers = np.arange(0, len(all_labels))
    TOTAL_HEIGHT = 0.6
    bar_height = TOTAL_HEIGHT / (len(opts))

    for i, opt in enumerate(opts):
        ax.barh(bar_centers + i * bar_height, all_speedup[:,i],
                height = bar_height,
                xerr=all_std[:,i],
                color=COLORS[i], alpha=0.85,
                ecolor="#555555",
                label=opt_labels[opt],
                edgecolor="#333333")

    ax.set_yticks(bar_centers + TOTAL_HEIGHT/2)
    ax.set_yticklabels(all_labels)

    delta_y = (1.0 - TOTAL_HEIGHT) / 2
    ax.set_ylim([-delta_y, len(all_labels) - delta_y])

    if malloc_only:
        label = "malloc()"
    else:
        label = "tcmalloc"
    ax.set_xlabel(label + " time improvement (%)")
    ax.grid(axis="x", ls="-", color="#cccccc")
    ax.set_axisbelow(True)

    lim = (ax.get_xlim()[1] - ax.get_xlim()[0]) /\
          (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.set_aspect(0.8 * lim)

    if (len(opts) > 1):
        ax.legend(frameon=False,
                  loc="upper left",
                  ncol=3, bbox_to_anchor=(0., 1.05, 1., .102))

    fig.tight_layout()

    fname = "../graphs_raw/speedup_%s" % "_".join(opts)
    if suff:
        fname += suff
    fname += "." + FORMAT
    plt.savefig(fname, bbox_inches="tight", dpi=200)
    plt.close()

    conn.close()

def plot_stacked_cycles(db):
    """ Plot the cumulative cycles in the fast path. """
    opts = ["baseline", "ideal-sampling", "ideal-size-class", "ideal-list", "limit"]
    labels = ["Baseline", "Sampling", "Size class", "Push/pop", "Combined"]
    colors = ["#cccccc", "#7fd086", "#caa878", "#86a6d5"]

    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    bmarks, _ = get_ubenchmarks(db)
    runs = get_all_runs(db)

    # All total cycles across runs in a 3D array.
    all_runs_avg_call_length = np.zeros((len(runs), len(opts), len(bmarks)))

    for run in runs:
        avg_call_length = [[] for opt in opts]
        bmk_labels = []
        for bmark in bmarks:
            print "Getting data for", bmark
            bmk_labels.append(bmark)
            for i, opt in enumerate(opts):
                data, num_calls = get_total_cycles_data(
                        cursor, bmark, malloc_funcs, opt, run,
                        fastpath_only=True)
                print num_calls
                if num_calls > 0:
                    avg_call_length[i].append(data / num_calls)
                else:
                    avg_call_length[i].append(float("NaN"))

        avg_call_length = np.array(avg_call_length)
        all_runs_avg_call_length[run,:,:] = avg_call_length

    # Compute mean and std call length
    mean_cycles = np.nanmean(all_runs_avg_call_length, axis=0)
    mean_std = np.nanstd(all_runs_avg_call_length, axis=0)

    deltas = np.zeros((len(opts), len(bmarks)))
    for i in range(1, len(opts)):
        deltas[i,:] = mean_cycles[0,:] - mean_cycles[i,:]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    bar_centers = np.arange(0, len(bmk_labels))
    bar_height = 0.6

    # Plot baseline cycles in the background
    ax.barh(bar_centers, mean_cycles[0,:], height = bar_height,
            label=labels[0],
            color=colors[0],
            lw=0)

    # Plot all cycles just a touch offset in the y direction
    ax.barh(bar_centers + 0.15, deltas[-1,:], height = bar_height,
            label=labels[-1],
            color="#93160d", alpha=0.85,
            zorder = 1,
            hatch="//",
            lw=0)

    # ...and then each of the deltas stacked
    cume_cycles = np.zeros(len(bmarks))
    for i in range(1, len(opts) - 1):
        ax.barh(bar_centers, deltas[i,:], height = bar_height,
                left = cume_cycles,
                label=labels[i],
                color=colors[i],
                xerr=mean_std[i,:],
                ecolor="#666666",
                edgecolor="#333333")
        cume_cycles += deltas[i,:]

    ax.set_yticks(bar_centers+bar_height/2)
    ax.set_yticklabels(bmk_labels)
    ax.yaxis.tick_right()

    delta_y = (1.0 - bar_height) / 2
    ax.set_ylim([-delta_y, len(bmarks) - delta_y])

    ax.set_xlabel("Fast path cycles")
    ax.set_xlim((0, 1.05 * np.max(mean_cycles[0:,])))

    lim = (ax.get_xlim()[1] - ax.get_xlim()[0]) /\
          (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.set_aspect(0.6 * lim)

    ax.grid(axis="x", ls="-", color="#cccccc")
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    # hardcode order because I'm feeling lazy. Doesn't seem to be an easy way to order
    # elements horizontally either...
    # And add a gap, so all real opts are on the same line, same as stack order.
    empty_h = mpl.patches.Rectangle((0,0), 1, 1, fill=False, edgecolor='none', visible=False)
    handles = [handles[0], handles[2], handles[1], handles[3], empty_h, handles[4]]
    labels = [labels[0], labels[2], labels[1], labels[3], "", labels[4]]

    ax.legend(handles, labels,
              frameon=False,
              loc="upper left",
              ncol=3, bbox_to_anchor=(0., 1.15, 1., .102))
    fig.tight_layout()
    fname = "../graphs_raw/stacked_cycles.%s" % FORMAT
    plt.savefig(fname, bbox_inches="tight", dpi=200)
    plt.close()

    conn.close()

def plot_stacked_speedup(db):
    """ Plot the cumulative speedup over multiple optimizations. """
    opts = ["sampling", "list", "list-size-class"]
    labels = ["Sampling", "Push/Pop", "Size class"]
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    bmarks, _ = get_all_benchmarks(db)
    runs = get_all_runs(db)
    baseline = "baseline"

    # All total speedups across runs in a 3D array.
    all_runs_total_speedup = np.zeros((len(runs), len(opts), len(bmarks)))

    runs = [0]
    for run in runs:
      total_speedup = [[] for opt in opts]
      bmk_labels = []
      for bmark in bmarks:
          if not bmark.startswith("ubench"):
              continue
          print "Getting data for", bmark
          bmk_labels.append(bmark)
          baseline_data, baseline_calls = get_total_cycles_data(
                  cursor, bmark, profiled_funcs, baseline, run, True)
          for i, opt in enumerate(opts):
              data, num_calls = get_total_cycles_data(
                      cursor, bmark, profiled_funcs, opt, run, True)
              total_speedup[i].append(baseline_data/data)

      total_speedup = np.array(total_speedup)
      print total_speedup
      all_runs_total_speedup[run,:,:] = total_speedup

    # Compute mean speedup and std deviation.
    mean_speedup = np.mean(all_runs_total_speedup, axis=0)
    mean_std = np.std(all_runs_total_speedup, axis=0)
    print "Std dev"
    print mean_std

    # Convert to incremental speedups.
    max_y = np.max(total_speedup)
    mean_speedup[1:,:] = np.diff(mean_speedup, axis=0)
    print "Incremental mean speedup."
    print mean_speedup

    fig = plt.figure()
    ax = fig.add_subplot(111)
    bar_centers = np.arange(0, len(bmk_labels))
    bar_width = 0.3
    bottom = np.zeros(len(bmk_labels))

    for i in range(len(opts)):
        ax.bar(bar_centers, mean_speedup[i,:],
               color=GnBu[i], width=bar_width, bottom=bottom,
               yerr=mean_std[i,:], label=labels[i])
        bottom += mean_speedup[i, :]

    # Strip prefix from ubenchmarks.
    bmk_labels = [b[7:] for b in bmk_labels]

    ax.set_xticks(bar_centers+bar_width/2)
    ax.set_xticklabels(bmk_labels)
    ax.set_xlim(left=-bar_width, right=len(bmk_labels)-bar_width)
    ax.set_ylim(bottom=1.0, top=max_y+0.1)
    ax.set_ylabel("Speedup")
    ax.grid(axis="y", ls="-", color="#cccccc")
    ax.set_axisbelow(True)
    ax.legend(loc="best")
    fig.tight_layout()
    fname = "../graphs_raw/stacked_speedups.%s" % FORMAT
    plt.savefig(fname, bbox_inches="tight")
    plt.close()

    conn.close()

def plot_fastpath_deltas(db):
    """ Plot the change in avg cycles per fastpath call. """
    opts = ["sampling", "list", "list-size-class"]
    labels = ["Baseline", "Sampling", "Push/Pop", "Size class"]
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    bmarks, _ = get_all_benchmarks(db)
    runs = get_all_runs(db)
    baseline = "baseline"

    # All avg fastpath cycles across runs in a 3D array.
    all_runs_avg_cycles = np.zeros((len(runs), len(opts) + 1, len(bmarks)))

    plots = {"malloc": malloc_funcs,
             "free": free_funcs}

    for plot, funcs in plots.iteritems():
      for run in runs:
        avg_cycles = np.zeros((len(opts) + 1, len(bmarks)))
        bmk_labels = []
        for i, bmark in enumerate(bmarks):
            if not bmark.startswith("ubench"):
                continue
            print "Getting data for", bmark
            bmk_labels.append(bmark)
            baseline_data, baseline_calls = get_total_cycles_data(
                    cursor, bmark, funcs, baseline, run, True)
            avg_cycles[0, i] = baseline_data / baseline_calls
            for j, opt in enumerate(opts):
                data, num_calls = get_total_cycles_data(
                        cursor, bmark, funcs, opt, run, True)
                avg_cycles[j+1, i] = data/num_calls

        print avg_cycles
        all_runs_avg_cycles[run,:,:] = avg_cycles

      # Compute mean and stddev across runs
      mean_avg_cycles = np.mean(all_runs_avg_cycles, axis=0)
      mean_std = np.std(all_runs_avg_cycles, axis=0)

      fig = plt.figure()
      ax = fig.add_subplot(111)
      bar_centers = np.arange(0, len(bmk_labels))
      bar_width = 0.15
      bottom = np.zeros(len(bmk_labels))

      for i in range(len(opts)+1):
          ax.bar(bar_centers + i*bar_width, mean_avg_cycles[i,:],
                 color=GnBu[i], width=bar_width, label=labels[i],
                 yerr=mean_std[i,:])

      # Strip prefix from ubenchmarks.
      bmk_labels = [b[7:] for b in bmk_labels]

      ax.set_xticks(bar_centers+(bar_width * (len(opts)+1))/2)
      ax.set_xticklabels(bmk_labels)
      ax.set_xlim(left=-bar_width, right=len(bmk_labels)-bar_width)
      ax.set_ylabel("Average cycles per call")
      ax.grid(axis="y", ls="-", color="#cccccc")
      ax.set_axisbelow(True)
      ax.legend(loc="best")
      lim = (ax.get_xlim()[1] - ax.get_xlim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
      ax.set_aspect(0.5*lim)
      fig.tight_layout()
      fname = "../graphs_raw/avg_fastpath_cycles_per_%s.%s" % (plot, FORMAT)
      plt.savefig(fname, bbox_inches="tight")
      plt.close()

    conn.close()

def plot_malloc_vs_free_time(db, opt):
    """ Plot total time spent in malloc vs free. """
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    bmarks, _ = get_all_benchmarks(db)
    baseline = "baseline-fdo"

    plots = {"malloc": malloc_funcs,
             "free": free_funcs}
    malloc_time = []
    free_time = []
    bmk_labels = []
    for bmark in bmarks:
        if not bmark.startswith("ubench"):
            continue
        print "Getting data for", bmark
        bmk_labels.append(bmark)
        malloc_time.append(get_total_cycles_data(
                cursor, bmark, plots["malloc"], opt))
        free_time.append(get_total_cycles_data(
                cursor, bmark, plots["free"], opt))

    # Scale data.
    malloc_time = np.array(malloc_time)/1e6
    free_time = np.array(free_time)/1e6

    bar_centers = np.arange(0, len(bmk_labels))
    bar_width = 0.25
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(bar_centers, malloc_time,
           width=bar_width, color=GnBu[0], label="malloc")
    ax.bar(bar_centers+bar_width, free_time,
           width=bar_width, color=GnBu[1], label="free")
    ax.set_xticks(bar_centers + bar_width)
    ax.set_xticklabels(bmk_labels)
    ax.set_ylabel("Cycles (M)")
    ax.set_xlim(left=-bar_width, right=len(bmk_labels)-bar_width)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", ls="-", color="#cccccc")
    ax.set_axisbelow(True)
    ax.set_title(get_pretty_opt_label(opt))
    ax.legend(loc="best")

    fig.tight_layout()
    fname = "../graphs_raw/malloc_vs_free_%s.%s" % (opt, FORMAT)
    plt.savefig(fname, bbox_inches="tight", dpi=200)
    plt.close()
    conn.close()

def plot_cache_size_sweep(db, horiz=False):
    """ Plots speedup for a sweep of cache sizes.

    This can also be used to plot speedup for a single cache size if there is just one
    cache size in the dataset, but you have to run parse_results.fix_cache_metadata
    first.
    """
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    bmarks, _ = get_all_benchmarks(db)
    print bmarks
    cache_sizes = get_cache_sizes(cursor)
    runs = get_all_runs(db)
    baseline = "baseline"

    plots = {"malloc": malloc_funcs,
             "free": free_funcs}

    bmark_labels = []

    # Collect all data across all runs.
    print "Collecting data for all realistic speedups."
    all_data = []
    for bmark in bmarks:
        if "ubench" in bmark:
            bmark_labels.append(bmark[7:])
        else:
            bmark_labels.append(bmark)
        bmark_data = []
        for size in cache_sizes:
            baseline_run_data = []
            opt_run_data = []
            for run in runs:
                print "Getting data for", bmark
                baseline_cycles, baseline_calls = get_total_cycles_data(
                        cursor, bmark, plots["malloc"], "baseline", run, cache_size=32)
                opt_cycles, opt_calls = get_total_cycles_data(
                        cursor, bmark, plots["malloc"], "realistic", run, cache_size=size)
                if baseline_calls == 0 or opt_calls == 0:
                    continue
                baseline_run_data.append(baseline_cycles/baseline_calls)
                opt_run_data.append(opt_cycles/opt_calls)

            baseline_sum = np.sum(baseline_run_data)
            baseline_mean = np.mean(baseline_run_data)
            opt_sum = np.sum(opt_run_data)
            opt_mean = np.mean(opt_run_data)

            speedup = baseline_sum / opt_sum
            std = (np.std(baseline_run_data)/baseline_mean + np.std(opt_run_data)/opt_mean) * speedup

            bmark_data.append([speedup, std])
        all_data.append(bmark_data)

    # Dimensions = (bmarks, sizes, 2)
    all_data = np.array(all_data)

    print all_data
    mean_data = all_data

    # Get ideal speedup.
    print "Collecting data for ideal speedups."
    ideal_data = []
    for bmark in bmarks:
        baseline_run_data = []
        opt_run_data = []
        for run in runs:
            # print "Getting data for", bmark
            baseline_cycles, baseline_calls = get_total_cycles_data(
                    cursor, bmark, plots["malloc"], "baseline", run, cache_size=32)
            opt_cycles, opt_calls = get_total_cycles_data(
                    cursor, bmark, plots["malloc"], "limit", run, cache_size=32)
            if baseline_calls == 0 or opt_calls == 0:
                continue
            baseline_run_data.append(baseline_cycles/baseline_calls)
            opt_run_data.append(opt_cycles/opt_calls)

        baseline_sum = np.sum(baseline_run_data)
        baseline_mean = np.mean(baseline_run_data)
        opt_sum = np.sum(opt_run_data)
        opt_mean = np.mean(opt_run_data)

        speedup = baseline_sum/opt_sum
        std = (np.std(baseline_run_data)/baseline_mean + np.std(opt_run_data)/opt_mean) * speedup

        ideal_data.append([speedup, std])
    ideal_data = np.array(ideal_data)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    bar_centers = np.arange(0, len(bmarks))
    bar_width = 1.0/(len(cache_sizes)+2)
    cm = get_colormap(plt.cm.GnBu, len(cache_sizes))

    # Realistic speedups
    for i, size in enumerate(cache_sizes):
        if horiz:
            ax.barh(bar_centers + i*bar_width, all_data[:, i, 0],
                    xerr=all_data[:, i, 1], height=bar_width, color=cm[i], label="%d" % size)
        else:
            ax.bar(bar_centers + i*bar_width, all_data[:, i, 0],
                   yerr=all_data[:, i, 1], width=bar_width, color=cm[i], label="%d" % size)
    # Ideal speedups
    if horiz:
        ax.barh(bar_centers + len(cache_sizes)*bar_width, ideal_data[:, 0],
                xerr=ideal_data[:, 1], height=bar_width, color="red", label="Limit")
    else:
        ax.bar(bar_centers + len(cache_sizes)*bar_width, ideal_data[:, 0],
               yerr=ideal_data[:, 1], width=bar_width, color="red", label="Limit")

    if horiz:
        ax.set_yticks(bar_centers + (len(cache_sizes)+1)*bar_width/2)
        ax.set_yticklabels(bmark_labels, ha="right")
        ax.set_xlabel("malloc speedup")
        ax.set_ylim((-bar_width, len(bmarks)+2))
        ax.set_xlim((0, 1.5))
        ax.grid(axis="x", ls="-", color="#cccccc")
    else:
        ax.set_xticks(bar_centers + (len(cache_sizes)+1)*bar_width/2)
        ax.set_xticklabels(bmark_labels, rotation=10, ha="center")
        ax.set_ylabel("malloc speedup")
        ax.set_xlim((-bar_width, len(bmarks)-bar_width))
        ax.set_ylim((0, 1.8))
        ax.grid(axis="y", ls="-", color="#cccccc")
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", ncol=6)
    fig.tight_layout()
    fname = "../graphs_raw/cache_size_sweep_malloc.%s" % FORMAT
    plt.savefig(fname, bbox_inches="tight", dpi=200)
    plt.close()
    conn.close()

def plot_hit_rates(db):
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    bmarks, _ = get_ubenchmarks(db)
    cache_sizes = get_cache_sizes(cursor)
    runs = [0]  # get_all_runs(db)
    baseline = "baseline"
    realistic = "realistic"
    stats = ["size_hits", "size_misses", "head_hits", "head_misses"]

    plots = {"malloc": malloc_funcs,
             "free": free_funcs}

    bmark_labels = []
    all_data = []
    all_run_size = np.zeros((len(runs), len(cache_sizes), len(bmarks)))
    all_run_head = np.zeros((len(runs), len(cache_sizes), len(bmarks)))

    # Collect all data across all runs.
    print "Collecting data for all realistic speedups."
    for run in runs:
        all_size_data = []
        all_head_data = []
        for size in cache_sizes:
            bmark_size_data = []
            bmark_head_data = []
            for bmark in bmarks:
                # print "Getting data for", bmark
                bmark_labels.append(bmark[7:])
                data = get_sim_stats(cursor, stats, bmark, realistic, run, cache_size=size)
                size_hit_rate = 100*float(data[0])/(data[0] + data[1])
                head_hit_rate = 100*float(data[2])/(data[2] + data[3])
                bmark_size_data.append(size_hit_rate)
                bmark_head_data.append(head_hit_rate)
            all_size_data.append(bmark_size_data)
            all_head_data.append(bmark_head_data)
        all_run_size[run, :] = np.array(all_size_data)
        all_run_head[run, :] = np.array(all_head_data)

    mean_size_data = np.mean(all_run_size, axis=0).astype(float)
    mean_head_data = np.mean(all_run_head, axis=0).astype(float)
    print mean_size_data
    print mean_head_data
    std_size = np.std(all_run_size, axis=0).astype(float)
    std_head = np.std(all_run_head, axis=0).astype(float)

    fig = plt.figure()
    axes = []
    axes.append(fig.add_subplot(211))
    ax = axes[0]
    bar_centers = np.arange(0, len(bmarks))
    bar_width = 1.0/(len(cache_sizes)+4)
    cm = get_colormap(plt.cm.GnBu, len(cache_sizes))

    # Size hit rates
    for i, size in enumerate(cache_sizes):
        ax.bar(bar_centers + i*bar_width, mean_size_data[i, :],
               width=bar_width, color=cm[i]) #, label="%d" % size)

    # Head pointer rates
    axes.append(fig.add_subplot(212))
    ax = axes[1]
    for i, size in enumerate(cache_sizes):
        ax.bar(bar_centers + i*bar_width, mean_head_data[i, :],
               width=bar_width, color=cm[i], label="%d" % size)

    for i in range(len(axes)):
        axes[i].set_xticks(bar_centers + len(cache_sizes)*bar_width/2)
        axes[i].set_xticklabels([])
        axes[i].set_xlim((-bar_width, len(bmarks)-bar_width))
        axes[i].set_ylim(bottom=0, top=110)
        axes[i].grid(axis="y", ls="-", color="#cccccc")
        axes[i].set_axisbelow(True)

    axes[0].set_ylabel("Size class")
    axes[1].set_ylabel("Head pointer")
    axes[1].set_xticklabels(bmark_labels)
    # axes[1].legend(loc="best")

    fig.tight_layout()
    fname = "../graphs_raw/cache_size_sweep_malloc_hit_rates.%s" % FORMAT
    plt.savefig(fname, bbox_inches="tight", dpi=200)
    plt.close()
    conn.close()

def plot_overall_speedup(db, opt, baseline="baseline"):
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    runs = get_all_runs(db)

    bmarks, _ = get_large_benchmarks(db)
    with open("../data/speedup.tab", "w") as f:
        all_baseline = []
        all_opt = []
        for run in runs:
            baseline_cycles = 0
            opt_cycles = 0
            for bmk in bmarks:
                try:
                    baseline_sim_cycles = float(get_total_sim_cycles(cursor, bmk, baseline, run, None))
                    opt_sim_cycles = float(get_total_sim_cycles(cursor, bmk, opt, run, None))
                    speedup = 100 * (1.0 - opt_sim_cycles / baseline_sim_cycles)
                    #print "%s %.2f %%" % (bmk, speedup)
                    #all_baseline.append(baseline_sim_cycles)
                    #all_opt.append(opt_sim_cycles)
                    baseline_cycles += baseline_sim_cycles
                    opt_cycles += opt_sim_cycles
                except TypeError:
                    continue

            all_baseline.append(baseline_cycles)
            all_opt.append(opt_cycles)

        print all_baseline
        print all_opt
        bmk = "xapian.query_wiki_pages"
        all_baseline = np.array(all_baseline)
        all_opt = np.array(all_opt)
        tt = st.ttest_ind(all_baseline, all_opt, equal_var=True)

        total_baseline = np.sum(all_baseline)
        total_opt = np.sum(all_opt)
        total_speedup = total_opt / total_baseline
        total_std = (np.nanstd(all_baseline)/total_baseline + np.nanstd(all_opt)/total_opt) * total_speedup

        total_speedup = 100 * (1.0 - total_speedup)
        total_std *= 100
        print "TOTAL: %s %.2f %% %.2f %%" % (bmk, total_speedup, total_std)
        print tt
        p_val = tt[1] / 2 # /2 for single-sided t-test

        if p_val < 0.05:
            f.write('\\bmk{%s} & %.2f\\%% & %.2f\\%% & %.3f\\\\\n' % (bmk, total_speedup, total_std, p_val))


def plot_ideal_vs_realistic_opts(db):
    """ Plots the fraction of ideal speedup achieved by the realistic optimizations. """
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    base_opts = ["size-class", "list-only", "sampling"]
    opt_labels = ["Size class", "Free list", "Sampling"]
    bmarks, _ = get_all_benchmarks(db)

    all_data = []
    all_realistic_data = []
    all_ideal_data = []
    bmark_labels = []
    for base_opt in base_opts:
        all_opt_data = []
        realistic_data = []
        ideal_data = []
        print base_opt
        for bmark in bmarks:
            bmark_labels.append(bmark[7:])
            print bmark
            ideal_cycles, ideal_calls = get_total_cycles_data(
                    cursor, bmark, malloc_funcs, "ideal-%s" % base_opt, 0, fastpath_only=True)
            real_cycles, real_calls = get_total_cycles_data(
                    cursor, bmark, malloc_funcs, "real-%s" % base_opt, 0, fastpath_only=True)
            baseline_cycles, baseline_calls = get_total_cycles_data(
                    cursor, bmark, malloc_funcs, "baseline", 0, fastpath_only=True)
            print ideal_cycles, ideal_calls
            print real_cycles, real_calls
            print baseline_cycles, baseline_calls

            ideal_per_call = float(ideal_cycles)/ideal_calls
            real_per_call = float(real_cycles)/real_calls
            baseline_per_call = float(baseline_cycles)/baseline_calls

            ideal_speedup = baseline_per_call/ideal_per_call
            real_speedup = baseline_per_call/real_per_call
            frac_ideal_speedup = real_speedup/ideal_speedup

            all_opt_data.append(frac_ideal_speedup)
            realistic_data.append(baseline_per_call - real_per_call)
            ideal_data.append(baseline_per_call - ideal_per_call)
        all_data.append(all_opt_data)
        all_realistic_data.append(realistic_data)
        all_ideal_data.append(ideal_data)

    all_data = np.array(all_data)
    all_realistic_data = np.array(all_realistic_data)
    all_ideal_data = np.array(all_ideal_data)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    bar_width = 1.0/(3*len(base_opts)+3)
    bar_centers = np.arange(len(bmarks))
    cm = get_colormap(plt.cm.GnBu, len(base_opts))
    for i, opt in enumerate(base_opts):
        ax.bar(bar_centers + (3*i)*bar_width, all_realistic_data[i, :],
               width=bar_width, label=opt_labels[i], color=cm[i])
    for i, opt in enumerate(base_opts):
        ax.bar(bar_centers + (3*i+1)*bar_width, all_ideal_data[i, :],
               width=bar_width, color=cm[i], hatch="/")

    ax.set_xticks(bar_centers + (len(base_opts)*bar_width)/2)
    ax.set_xticklabels(bmark_labels, rotation=10, ha="center")
    ax.set_ylabel("Per call cycles reduction")
    # ax.set_ylim(bottom=0, top=2)
    ax.set_xlim(left=-bar_width, right=len(bmarks)-bar_width)
    ax.grid(axis="y", ls="-", color="#cccccc")
    ax.set_axisbelow(True)
    ax.legend(loc="best", ncol=3)

    lim = (ax.get_xlim()[1] - ax.get_xlim()[0]) /\
          (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.set_aspect(0.5 * lim)

    plt.savefig("../graphs_raw/ideal_vs_realistic_speedup.%s" % FORMAT,
                bbox_inches="tight", dpi=200)

    conn.close()

def plot_annotated_perl_baseline(db):
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    min_logx = 1
    max_logx = 5
    cm = get_colormap(plt.cm.inferno, 4)
    bmark = "400.perlbench.diffmail"
    baseline = "baseline"
    profile_label = "malloc"

    fig = plt.figure()
    ax = fig.add_subplot(111)
    baseline_data = get_cycles_data(cursor, bmark, malloc_funcs, baseline)
    weights = 100.0 * baseline_data / (sum(baseline_data))
    baseline_hist, baseline_edges = np.histogram(baseline_data,
            bins=np.logspace(min_logx, max_logx, (max_logx-min_logx+1)*10),
            weights=weights)
    # we do a semi-log plot manually, so we can set the aspect ratio
    # and keep consistent font sizes
    plot_semilogx(ax, baseline_edges[:-1], baseline_hist,
                  linewidth=2, c=cm[2], label=bmark)

    # Annotate the peaks.
    plt.text(np.log10(13), 40, "Fast path",
               fontsize=FONT_SIZE-4, color="#222222", backgroundcolor="white")
    plt.annotate("",
                 xy=(np.log10(15), 32), xycoords="data",
                 xytext=(np.log10(15), 39), textcoords="data",
                 arrowprops=dict(arrowstyle="-|>",
                                 facecolor="w",
                                 edgecolor="#222222"))

    plt.text(np.log10(4200), 30, "Get from\ncentral cache",
               fontsize=FONT_SIZE-4, color="#222222", backgroundcolor="white", ha="center")
    plt.annotate("",
                 xy=(np.log10(4500), 10), xycoords="data",
                 xytext=(np.log10(4200), 28), textcoords="data",
                 arrowprops=dict(arrowstyle="-|>",
                                 facecolor="w",
                                 edgecolor="#222222"))

    plt.text(np.log10(36000), 30, "Get from\npage allocator",
               fontsize=FONT_SIZE-4, color="#222222", backgroundcolor="white", ha="center")
    plt.annotate("",
                 xy=(np.log10(38000), 15), xycoords="data",
                 xytext=(np.log10(36000), 28), textcoords="data",
                 arrowprops=dict(arrowstyle="-|>",
                                 facecolor="w",
                                 edgecolor="#222222"))

    ax.set_ylim(bottom=0, top=50)
    plt.minorticks_on()
    set_axis_limits(ax, max_logx, min_logx)

    lim = (ax.get_xlim()[1] - ax.get_xlim()[0]) /\
          (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.set_aspect(0.5 * lim)

    ax.grid(axis="y", ls="-", color="#cccccc")
    ax.set_axisbelow(True)
    ax.set_xlabel("%s duration (cycles)" % profile_label)

    ax.set_ylabel("Time in calls (PDF %)")
    legend = ax.legend(loc="upper right")
    fprefix = "pdf"
    legend.get_frame().set_edgecolor('w')
    fname = "../graphs_raw/%s_%s_%s_annotated.%s" % (bmark, fprefix, profile_label, FORMAT)
    plt.savefig(fname, bbox_inches="tight", dpi=200)
    plt.close()
    conn.close()

def paper_plots(db):
    #plot_annotated_perl_baseline(db)

    large_bmks, _ = get_large_benchmarks(db)
    all_bmks, _ = get_all_benchmarks(db)
    #plot_baseline(db, large_bmks, "baseline", "_large", cdf=True)

    # Run with cache_sweep_8.db
    #plot_cache_size_sweep(db, horiz=False)

    #plot_ideal_speedup(db, large_bmks, ["limit"], "baseline", "")
    #plot_ideal_speedup(db, large_bmks, ["realistic"], "baseline", "")
    #plot_ideal_speedup(db, all_bmks, ["realistic", "limit"], "baseline", "_all")
    #plot_ideal_speedup(db, large_bmks, ["realistic", "limit"], "baseline", "_large")
    plot_ideal_speedup(db, large_bmks, ["realistic"], "baseline", "_large", True)

    #plot_baseline_vs_opt_profiling(db, "all", "baseline")

    #plot_stacked_cycles(db)
    print "ideal"
    #plot_overall_speedup(db, "limit")
    print "real"
    #plot_overall_speedup(db, "realistic")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode",
            choices=["plot-baseline", "plot-diff",
                     "plot-speedup", "plot-stacked-speedup",
                     "plot-stacked-cycles", "plot-fastpath",
                     "plot-malloc-vs-free", "plot-cache-sweep",
                     "plot-hit-rates", "plot-sim-speedup",
                     "plot-ideal-vs-real", "paper-plots"])
    parser.add_argument("--db", help="SQLite3 DB.")
    parser.add_argument("--opt", default="pop-fdo", help="Optimization to plot.")
    parser.add_argument("--benchmark", help="Plot a specific benchmark.")
    parser.add_argument("--cdf", action="store_true", default=False,
            help="For certain plots, plot CDF instead of PDF.")
    args = parser.parse_args()

    matplotlib_init()

    if args.mode == "plot-baseline":
        plot_baseline(args.db, get_all_benchmarks(args.db)[0], cdf=args.cdf)
    elif args.mode == "plot-diff":
        plot_baseline_vs_opt_profiling(args.db, args.opt)
    elif args.mode == "plot-diff-combined":
        plot_baseline_vs_opt_combined(args.db, args.opt, args.benchmark)
    elif args.mode == "plot-time-series":
        plot_time_series(args.db, args.opt, args.benchmark)
    elif args.mode == "plot-speedup":
        plot_ideal_speedup(args.db, args.opt)
    elif args.mode == "plot-stacked-speedup":
        plot_stacked_speedup(args.db)
    elif args.mode == "plot-stacked-cycles":
        plot_stacked_cycles(args.db)
    elif args.mode == "plot-malloc-vs-free":
        plot_malloc_vs_free_time(args.db, args.opt)
    elif args.mode == "plot-fastpath":
        plot_fastpath_deltas(args.db)
    elif args.mode == "plot-cache-sweep":
        plot_cache_size_sweep(args.db, horiz=True)
    elif args.mode == "plot-hit-rates":
        plot_hit_rates(args.db)
    elif args.mode == "plot-ideal-vs-real":
        plot_ideal_vs_realistic_opts(args.db)
    elif args.mode == "paper-plots":
        paper_plots(args.db)
    elif args.mode == "plot-sim-speedup":
        plot_overall_speedup(args.db, args.opt)

if __name__ == "__main__":
    main()
