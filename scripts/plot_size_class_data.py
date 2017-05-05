#!/usr/bin/env python
#
# Plots distributions of size classes observed in SPEC.
#
# Usage:
#   ./plot_size_class_data.py read --results-dir /path/to/spec/top/dir --db spec.db
#   ./plot_size_class_data.py [plot-pdf|plot-cdf] --db spec.db

import argparse
import fnmatch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cPickle as pickle
import numpy as np
import os
import re
import sqlite3
import time
from colormap_helper import get_colormap
from fake_semilogx import *

spec_benchmarks = [
    "400.perlbench",
    "465.tonto",
    "471.omnetpp",
    "483.xalancbmk",
]

xapian_benchmarks = [
    "xapian.abstracts",
    "xapian.pages",
]


extension = "O3gcc"
FONT_SIZE = 16
FORMAT = "pdf"
NUM_SZ_CLASSES = 88
grays = get_colormap(plt.cm.gray, 5)

def matplotlib_init():
    mpl.rcParams.update({"font.size": FONT_SIZE,
                         "mathtext.default": "regular"})
    mpl.rc("xtick", labelsize=FONT_SIZE)
    mpl.rc("ytick", labelsize=FONT_SIZE)
    mpl.rc("legend", **{"fontsize": FONT_SIZE-2})

class BenchmarkDataParser(object):
    def __init__(self, prefix, get_profiling_file_func, parse_filepath_func):
        self.get_profiling_files = get_profiling_file_func
        self.parse_filepath = parse_filepath_func
        self.prefix = prefix

# This is set by the command line bmk argument.
bmk_suite = ""

##############################################################################
#
# These are file parsing functions analyzing size classes specific to each
# benchmark suite, since each suite has a different directory structure. Since
# size classes are analyzed from native runs and not simulation, these
# functions will differ from those in plot_profiling_files, which is meant to
# plot simulation results.
#
##############################################################################

def get_spec_profiling_files(results_dir):
    """ Gets all stderr files from the result dir for our benchmarks. """
    rundirname = "benchspec/CPU2006/%s/run"
    runrefname = "run_base_ref_%s" % extension
    matches = []
    for benchmark in spec_benchmarks:
      full_path = os.path.join(results_dir, rundirname % benchmark)
      for root, dirnames, filenames in os.walk(full_path):
          for filename in fnmatch.filter(filenames, '*.err'):
            if runrefname in root:
                matches.append(os.path.join(root, filename))
    return matches

def parse_spec_filepath(filepath):
    path = filepath.split("/")
    run = path[-2][-4:]  # Last four digits.
    bmk = path[-4]
    return bmk, run

def get_ubench_profiling_files(results_dir):
    """ Gets all stderr files from result dir of ubenchmarks. """
    matches = []
    for root, dirnames, filenames in os.walk(results_dir):
        for filename in fnmatch.filter(filenames, '*.err'):
            matches.append(os.path.join(root, filename))
    return matches

def parse_ubench_filepath(filepath):
    path = filepath.split("/")
    bmk = path[-1][:-4]
    run = 0
    return bmk, run

def get_general_profiling_files(results_dir):
    return get_ubench_profiling_files(results_dir)

def parse_general_filepath(filepath):
    return parse_ubench_filepath(filepath)

data_parsers = {
    "spec": BenchmarkDataParser(
        "spec_", get_spec_profiling_files, parse_spec_filepath),
    "ubench": BenchmarkDataParser(
        "ubench_", get_ubench_profiling_files, parse_ubench_filepath),
    "general": BenchmarkDataParser(
        "", get_general_profiling_files, parse_general_filepath),
}

# Database and data parsing routines.

def create_database(db, overwrite_db):
    """ Creates the database. """
    if os.path.exists(db):
      if overwrite_db:
        os.remove(db)
      else:
        return

    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    queries = []
    queries.append("create table data (size_class integer, "
                                      "benchmark integer, "
                                      "run integer)")
    queries.append("create table size_classes (size_class integer, "
                                              "size integer)")
    queries.append("create table benchmarks (id integer primary key, benchmark text) ")
    queries.append("create table runs (id integer primary key)")

    for query in queries:
        cursor.execute(query)

    conn.commit()
    conn.close()

def get_or_store_metadata(cursor, benchmark):
    """ Return ids for benchmark, or stores it if it does not exist in the db. """
    query = "select id from benchmarks where benchmark = ?"
    cursor.execute(query, (benchmark,))
    result = cursor.fetchone()
    if not result:
        cursor.execute("insert into benchmarks (benchmark) values (?)", (benchmark,))
        cursor.execute(query, (benchmark,))
        result = cursor.fetchone()
    benchmark_id = result[0]
    return benchmark_id

def store_to_db(files, db):
    conn = sqlite3.connect(db)
    conn.isolation_level = None
    cursor = conn.cursor()
    cursor.execute("PRAGMA synchronous = OFF")
    cursor.execute("PRAGMA journal_mode = MEMORY")
    cursor.execute("begin transaction")

    data_query = "insert into data (size_class, benchmark, run) values (?, ?, ?)"
    runs_query = "insert into runs (id) values (?)"
    size_class_query = "insert into size_classes (size_class, size) values (?, ?)"
    runs = set()
    size_classes = {}
    regex = re.compile("(?<=:0] SC:  )(\d+)\s+(\d+)")
    for i, fpath in enumerate(files):
        benchmark, run = data_parsers[bmk_suite].parse_filepath(fpath)
        bid = get_or_store_metadata(cursor, benchmark)
        runs.add(run)
        sizes = []

        print "[%3.0f %%] Parsing %s" % (100.0 * i/len(files), fpath)
        with open(fpath, "r") as f:
            for line in f:
                m = re.search(regex, line)
                if not m:
                    continue
                size_class = m.group(1)
                size = m.group(2)
                sizes.append((size_class, bid, run))
                size_classes[size_class] = size
            cursor.executemany(data_query, sizes)

    for run in runs:
        cursor.execute(runs_query, (run,))
    for size_class, size in size_classes.iteritems():
        cursor.execute(size_class_query, (size_class, size))
    cursor.execute("end transaction")

    conn.commit()
    conn.close()

def parse_profiling_data(results_dir, db, overwrite_db=True):
    ''' Parse all profiling data in results_dir and store to SQLite3 DB.  '''
    start = time.time()
    profiling_files = data_parsers[bmk_suite].get_profiling_files(results_dir)
    create_database(db, overwrite_db)
    store_to_db(profiling_files, db)
    end = time.time()
    print ""
    print "Total time spent reading prof files: ", (end - start)/60.0, " minutes."

def get_size_class_data(cursor, bmark, run=0):
    size_query = ("select size from size_classes, data where "
                  "data.size_class = size_classes.size_class and "
                  "benchmark=? and run=?")
    size_class_query = "select size_class from data where benchmark=? and run=?"
    bid = get_or_store_metadata(cursor, bmark)
    cursor.execute(size_class_query, (bid, run))
    results = cursor.fetchall()
    size_classes = []
    for row in results:
        size_classes.append(row[0])
    return size_classes

def get_all_benchmarks(cursor):
    """ Return all benchmarks from the DB. """
    query = "select benchmark, id from benchmarks order by benchmark asc"

    cursor.execute(query)
    results = cursor.fetchall()
    benchmarks = [row[0] for row in results]
    ids = [row[1] for row in results]
    return benchmarks, ids

# Plotting routines

def plot_top_size_classes(db):
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    bmarks, _ = get_all_benchmarks(cursor)
    cm = get_colormap(plt.cm.Paired, len(bmarks))
    linestyles = [".", "o", "+"]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i, bmark in enumerate(bmarks):
        print "Plotting %s" % bmark

        baseline_data = get_size_class_data(cursor, bmark)
        if len(baseline_data) == 0:
            continue
        print min(baseline_data), max(baseline_data)

        # Compute histograms
        bins = NUM_SZ_CLASSES
        weight = 100.0 / (len(baseline_data))
        baseline_hist, baseline_edges = np.histogram(
                baseline_data, range=(0, NUM_SZ_CLASSES), bins=bins)
        idx = np.argsort(baseline_hist)
        baseline_hist[::-1].sort()
        cdf = np.cumsum(baseline_hist).astype(float)
        cdf = 100.0 * cdf/cdf[-1]
        if bmark in spec_benchmarks:
            marker = "."
        elif bmark in xapian_benchmarks:
            marker = "^"
        else:
            marker = "o"
        ax.plot(range(1, bins+1), cdf, linewidth=1.5, color=cm[i], label="%s" % bmark, marker=marker)

    ax.set_xlabel("Size classes")
    ax.set_ylabel("malloc() calls (% CDF)")
    ax.set_xlim(left=0.5, right=30)
    ax.set_ylim(bottom=0, top=105)

    lim = (ax.get_xlim()[1] - ax.get_xlim()[0]) /\
          (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.set_aspect(0.5 * lim)

    legend = ax.legend(loc=0)
    legend.get_frame().set_edgecolor('w')
    fname = "../graphs_raw/size_class_cdf.%s" % FORMAT
    plt.savefig(fname, bbox_inches="tight", dpi=200)
    plt.close()

def plot_size_class_pdf(db):
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    bmarks, _ = get_all_benchmarks(cursor)
    cm = get_colormap(plt.cm.inferno, len(bmarks))

    fig = plt.figure()
    ax = fig.add_subplot(111)

    min_logx = 1
    max_logx = 5

    for i, bmark in enumerate(bmarks):
        print "Plotting %s" % bmark

        baseline_data = get_size_class_data(cursor, bmark)
        if len(baseline_data) == 0:
            continue

        # Compute histograms
        bins = np.logspace(min_logx, max_logx, NUM_SZ_CLASSES)
        weight = 100.0 / (len(baseline_data))
        baseline_hist, baseline_edges = np.histogram(baseline_data, bins=bins)

        baseline_hist = baseline_hist * weight
        plot_semilogx(ax, baseline_edges[:-1], baseline_hist,
                      linewidth=1.5, color=cm[i], label="%s" % bmark)

    set_axis_limits(ax, max_logx, min_logx)
    ax.set_xlabel("Size class")
    ax.set_ylabel("Frequency")

    lim = (ax.get_xlim()[1] - ax.get_xlim()[0]) /\
          (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.set_aspect(0.5 * lim)

    legend = ax.legend(loc=0)
    legend.get_frame().set_edgecolor('w')
    fname = "../graphs_raw/size_class_pdf.%s" % FORMAT
    plt.savefig(fname, bbox_inches="tight", dpi=200)
    plt.close()

def plot_size_class_map(size_class_map_file):
    pickle_file = "size_class_data.pkl"
    if not os.path.exists(pickle_file):
      # Parse the given file for size class data.
      index_map = []
      size_map = []
      s2s_map = []  # requested to allocated size map.
      index_flag = "Class index to size class"
      size_flag = "Size class to size"
      unrounded_flag = "Unrounded size"
      done_flag = "Done"
      index_started = False
      size_started = False

      with open(size_class_map_file, "r") as f:
        for line in f:
          if index_flag in line:
            index_started = True
            size_started = False
            continue
          elif size_flag in line:
            index_started = False
            size_started = True
            continue
          elif unrounded_flag in line:
            index_started = False
            size_started = False
            continue
          elif done_flag in line:
            break

          split_line = line.split()
          if index_started:
            # index = int(split_line[1])
            size_class = int(split_line[2])
            index_map.append(size_class)
            continue
          elif size_started:
            # size_class = int(split_line[1])
            size = int(split_line[2])
            size_map.append(size)
            continue
          else:
            requested_size = int(split_line[1])
            size_class = int(split_line[2])
            allocated_size = int(split_line[3])
            s2s_map.append([size_class, allocated_size])
            continue
      s2s_map = np.array(s2s_map)

      parsed_data = {
        "index_map": index_map,
        "size_map": size_map,
        "s2s_map": s2s_map,
      }

      with open(pickle_file, "w") as f:
        pickle.dump(parsed_data, f)
    else:
      # Unpickle!
      with open(pickle_file, "r") as f:
        parsed_data = pickle.load(f)
        index_map = parsed_data["index_map"]
        size_map = parsed_data["size_map"]
        s2s_map = parsed_data["s2s_map"]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(index_map)
    ax.grid(axis="both", ls="-", color="#cccccc")
    # This is the alignment change.
    ax.axvline(128, color="#555555", linewidth=1.5, linestyle="--")
    ax.set_axisbelow(True)
    ax.set_xlabel("Size class index")
    ax.set_ylabel("Size class")
    plt.savefig("../graphs_raw/index_map.pdf", bbox_inches="tight", dpi=200)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.diff(index_map))
    ax.grid(axis="both", ls="-", color="#cccccc")
    ax.set_axisbelow(True)
    ax.axvline(128, color="#555555", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Size class index")
    ax.set_ylabel("Size class")
    plt.savefig("../graphs_raw/index_map_diffs.pdf", bbox_inches="tight", dpi=200)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(size_map)
    ax.grid(axis="both", ls="-", color="#cccccc")
    ax.set_axisbelow(True)
    ax.set_xlabel("Size class")
    ax.set_ylabel("Allocated size")
    plt.savefig("../graphs_raw/size_map.pdf", bbox_inches="tight", dpi=200)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.semilogy(np.diff(size_map, n=1))
    ax.grid(axis="both", ls="-", color="#cccccc")
    ax.set_axisbelow(True)
    ax.set_xlabel("Size class")
    ax.set_ylabel("Increase in size")
    plt.savefig("../graphs_raw/size_map_diffs.pdf", bbox_inches="tight", dpi=200)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.loglog(s2s_map[:, 1])
    ax.grid(axis="both", ls="-", color="#cccccc")
    ax.set_axisbelow(True)
    ax.set_xlabel("Requested size")
    ax.set_ylabel("Allocated size")
    ax.axvline(3328, color="#555555", linewidth=1.5, linestyle="--")
    ax.axvline(32768, color="#555555", linewidth=1.5, linestyle="--")
    plt.savefig("../graphs_raw/s2s_map.pdf", bbox_inches="tight", dpi=200)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.diff(s2s_map[:, 1]))
    ax.grid(axis="both", ls="-", color="#cccccc")
    ax.set_axisbelow(True)
    ax.set_xlabel("Requested size")
    ax.set_ylabel("Allocated size increase")
    ax.axvline(3328, color="#555555", linewidth=1.5, linestyle="--")
    ax.axvline(32768, color="#555555", linewidth=1.5, linestyle="--")
    plt.savefig("../graphs_raw/s2s_map_diff.pdf", bbox_inches="tight", dpi=200)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(s2s_map[:, 0])
    ax.grid(axis="both", ls="-", color="#cccccc")
    ax.set_axisbelow(True)
    ax.set_xlabel("Requested size")
    ax.set_ylabel("Size class")
    ax.axvline(3328, color="#555555", linewidth=1.5, linestyle="--")
    ax.axvline(32768, color="#555555", linewidth=1.5, linestyle="--")
    plt.savefig("../graphs_raw/size_to_size_class.pdf", bbox_inches="tight", dpi=200)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["read", "plot-pdf", "plot-cdf", "plot-size-map"])
    parser.add_argument("--bmk", choices=["spec", "ubench", "general"], help=
        "Benchmark suite being analyzed. Only needed for read mode. "
        "Different benchmark suites have different output directory structures. "
        "If not sure, pick general, which will analyze all *.err files.")
    parser.add_argument("--results-dir", help="Results top level directory.")
    parser.add_argument("--db", help="SQLite3 database file.")
    parser.add_argument("--size-class-map", help="A dump of size class index "
        "to size class and size class to allocated size.")
    args = parser.parse_args()

    matplotlib_init()

    if args.mode == "read":
        if not args.bmk:
            print "Need to specify benchmark type (--bmk)!"
            return -1
        global bmk_suite
        bmk_suite = args.bmk
        parse_profiling_data(args.results_dir, args.db)
    elif args.mode == "plot-pdf":
        plot_size_class_pdf(args.db)
    elif args.mode == "plot-cdf":
        plot_top_size_classes(args.db)
    elif args.mode == "plot-size-map":
        plot_size_class_map(args.size_class_map)

if __name__ == "__main__":
    main()
