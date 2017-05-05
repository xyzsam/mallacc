#!/usr/bin/env python
#
# Creates a SQLite DB from a results directory.
#
# Usage:
#   First, parse profiling data and store to a SQLite database.
#       python parse_results.py read --results-dir /path/to/results --db prof.db
#   Then, generate aux tables in the database.
#       python parse_results.py analyze --db prof.db
#
#   After that, all plotting scripts point to the single database.

import argparse
import math
import fnmatch
import os
import re
import sqlite3
import time
import sys
sys.path.append(os.path.join(os.environ["XIOSIM_TREE"], "scripts"))
import xiosim_stat as xs

import db_common

def get_profiling_files(results_dir):
    """ Returns all profiling output files under results_dir, recursively. """
    matches = []
    for root, dirnames, filenames in os.walk(results_dir, followlinks=True):
        for filename in fnmatch.filter(filenames, '*.prof.*'):
            matches.append(os.path.join(root, filename))
    return matches

def parse_profiling_filepath(filepath):
    """ Get benchmark, profiled function number, optimization, and run from filepath. """
    path = filepath.split("/")
    fname = path[-1]
    run = path[-2]
    opt = path[-3]
    benchmark = path[-4]
    cache_size = path[-5]
    func_no = int(fname.split(".")[-1])
    return benchmark, func_no, opt, run, cache_size

def get_simout_files(results_dir):
    """ Returns all sim.out.slice.* files under results_dir, recursively. """
    matches = []
    for root, dirnames, filenames in os.walk(results_dir, followlinks=True):
        for filename in fnmatch.filter(filenames, '*.sim.out.slice.*'):
            matches.append(os.path.join(root, filename))
    return matches

def parse_simout_filepath(filepath):
    """ Gets the benchmark optimizations and run from the filepath. """
    split = filepath.split("/")
    run = split[-2]
    opt = split[-3]
    benchmark = split[-4]
    cache_size = split[-5]
    fname = split[-1]
    slice_no = re.search("(?<=slice.)[0-9]+", fname).group(0)
    return (benchmark, opt, run, slice_no, cache_size)

def get_or_store_metadata(cursor, benchmark, function_no, optimization, cache_sz=None):
    """ Returns ids for benchmark, function, and optimization.

    If they are not present in their tables, they are added to the tables.

    Args:
        benchmark: Name of benchmark as string.
        function_no: The suffix index of the profiling output file.
        optimization: Name of the optimization, or "baseline" if none.
    """
    if benchmark != None:
        query = "select * from benchmarks where benchmark=?"
        cursor.execute(query, (benchmark,))
        result = cursor.fetchone()
        if not result:
            cursor.execute("insert into benchmarks (benchmark) values (?)", (benchmark,))
            cursor.execute(query, (benchmark,))
            result = cursor.fetchone()
        benchmark_id = result[0]
    else:
        benchmark_id = None

    if function_no != None:
        query = "select * from functions where id=? and function=?"
        values = (function_no, db_common.profiled_funcs[function_no])
        cursor.execute(query, values)
        result = cursor.fetchone()
        if not result:
            cursor.execute("insert into functions (id, function) values (?,?)", values)
            cursor.execute(query, values)
            result = cursor.fetchone()
        function_id = result[0]
    else:
        function_id = None

    if optimization != None:
        query = "select * from optimizations where optimization=?"
        cursor.execute(query, (optimization,))
        result = cursor.fetchone()
        if not result:
            cursor.execute("insert into optimizations (optimization) values (?)", (optimization,))
            cursor.execute(query, (optimization,))
            result = cursor.fetchone()
        optimization_id = result[0]
    else:
        optimization_id = None

    if cache_sz != None:
        query = "select * from cache_sizes where size=?"
        cursor.execute(query, (cache_sz,))
        result = cursor.fetchone()
        if not result:
            cursor.execute("insert into cache_sizes (size) values (?)", (cache_sz,))
            cursor.execute(query, (cache_sz,))
            result = cursor.fetchone()
        cache_sz = result[0]
    else:
        cache_sz = None

    return benchmark_id, function_id, optimization_id, cache_sz

def generate_analysis_tables(db, overwrite=False):
    """ Create auxillary tables for faster analysis and plotting. """
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    table_name = db_common.PROF_FASTPATH_CYCLES_TABLE_NAME
    # Check if table exists and abort if so, unless overwrite is True.
    cursor.execute(
          "select count(*) from SQLITE_MASTER where type='table' and name='%s'" % table_name)
    result = cursor.fetchone()[0]
    if result > 0:
      if overwrite:
        print "WARNING! ABOUT TO DROP TABLE FASTPATH_CYCLES!!! Pausing for 5 seconds..."
        time.sleep(5)
        cursor.execute("drop table %s" % table_name)
      else:
        return

    query = ("create table %s (id integer primary key, "
                              "cycles integer,"
                              "calls integer,"
                              "benchmark integer,"
                              "function integer,"
                              "optimization integer,"
                              "run integer,"
                              "cache_size integer)" % table_name)
    cursor.execute(query)
    insert_query = ("insert into %s "
                    "(cycles, calls, benchmark, function, optimization, run, cache_size) "
                    "select sum(cycles), count(cycles), benchmark, function, optimization, run, cache_size "
                    "from %s where cycles < 100 "
                    "group by benchmark, function, optimization, run, cache_size" % (
                      table_name, db_common.PROF_TABLE_NAME))
    cursor.execute(insert_query)
    conn.commit()
    conn.close()

    print "Analysis tables generated."

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
    queries.append("create table %s (cycles integer, "
                                    "benchmark integer, "
                                    "function integer, "
                                    "optimization integer,"
                                    "run integer,"
                                    "cache_size integer)" % db_common.PROF_TABLE_NAME)
    queries.append("create table %s (id integer primary key, "
                                    "cycles integer,"
                                    "calls integer,"
                                    "benchmark integer,"
                                    "function integer,"
                                    "optimization integer,"
                                    "run integer,"
                                    "cache_size integer)" % db_common.PROF_TOTAL_CYCLES_TABLE_NAME)

    stat_fields = "".join(["%s integer," % f for f in db_common.STATS.iterkeys()])
    create_sim_query = ("create table %s (id integer primary key,"
                                         "%s"
                                         "benchmark integer,"
                                         "optimization integer,"
                                         "run integer,"
                                         "cache_size integer,"
                                         "slice integer)") % (db_common.SIM_STATS_TABLE_NAME, stat_fields)
    queries.append(create_sim_query)
    queries.append("create table benchmarks (id integer primary key, benchmark text) ")
    queries.append("create table functions (id integer primary key, function text) ")
    queries.append("create table optimizations (id integer primary key, optimization text) ")
    queries.append("create table runs (id integer)")
    queries.append("create table cache_sizes (size integer)")

    for query in queries:
        cursor.execute(query)

    conn.commit()
    conn.close()

def store_profiling_to_db(files, db, is_cache_sweep):
    conn = sqlite3.connect(db)
    conn.isolation_level = None
    cursor = conn.cursor()
    cursor.execute("PRAGMA synchronous = OFF")
    cursor.execute("PRAGMA journal_mode = MEMORY")
    cursor.execute("begin transaction")

    data_query = ("insert into %s (cycles, benchmark, function, optimization, run, cache_size) "
                  "values (?, ?, ?, ?, ?, ?)") % db_common.PROF_TABLE_NAME
    # Compute total cycles as we go.
    total_cycles_query = ("insert into %s (cycles, calls, benchmark, "
                          "function, optimization, run, cache_size) values (?, ?, ?, ?, ?, ?, ?)") % db_common.PROF_TOTAL_CYCLES_TABLE_NAME
    runs_query = "insert into runs (id) values (?)"
    cache_query = "insert into cache_sizes (size) values (?)"
    # Track runs
    runs = set()
    cache_sizes = set()
    for i, fpath in enumerate(files):
        total_cycles = 0
        total_calls = 0
        benchmark, func_no, opt, run, cache_sz = parse_profiling_filepath(fpath)
        if not is_cache_sweep:
            cache_sz = None
        print "[%2.2f %%] Parsing %s" % (100.0 * i/len(files), fpath)
        bid, fid, oid, cs = get_or_store_metadata(cursor, benchmark, func_no, opt, cache_sz)

        cycles_array = []
        with open(fpath, "r") as f:
            for line in f:
                cycles = int(line)
                cycles_array.append((cycles, bid, fid, oid, run, cs))
                #cursor.execute(data_query, (cycles, bid, fid, oid, run, cs))
                total_cycles += cycles
                total_calls += 1

        cursor.executemany(data_query, cycles_array)

        # Update the total cycles.
        cursor.execute(total_cycles_query, (
                total_cycles, total_calls, bid, fid, oid, run, cs))
        runs.add(run)

    # Write all runs to the runs table.
    for run in runs:
        cursor.execute(runs_query, (run,))

    # Do the same for cache sizes.
    if is_cache_sweep:
        for size in cache_sizes:
            cursor.execute(cache_query, (size,))

    cursor.execute("end transaction")

    conn.commit()
    conn.close()

def parse_profiling_data(results_dir, db, is_cache_sweep):
    ''' Parse all profiling data in results_dir and store to SQLite3 DB.  '''
    start = time.time()
    profiling_files = get_profiling_files(results_dir)
    store_profiling_to_db(profiling_files, db, is_cache_sweep)
    end = time.time()
    print ""
    print "Total time spent reading prof files: ", (end - start)/60.0, " minutes."

def insert_sim_stats(cursor, stats, bid, oid, run, slice_no, cache_size):
    """ Inserts a row into the database. """
    if cache_size == None:
        cache_size = -1

    for i in range(len(stats)):
      if math.isnan(stats[i]):
            stats[i] = -1

    query = "insert into %s (%s) values (%s)"

    columns = [str(k) for k in db_common.STATS.iterkeys()]
    columns += ["benchmark", "optimization", "run", "slice", "cache_size"]
    columns_str = ",".join(columns)

    values = stats + [bid, oid, run, slice_no, cache_size]
    values_str = ",".join([str(v) for v in values])

    query = query % (db_common.SIM_STATS_TABLE_NAME, columns_str, values_str)
    print query
    cursor.execute(query)

def store_sim_to_db(sim_files, db, is_cache_sweep):
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    for simout_file in sim_files:
        print "Parsing %s" % simout_file
        bmk, opt, run, sno, cache_size = parse_simout_filepath(simout_file)
        if not is_cache_sweep:
            cache_size = None
        bid, _, oid, cs = get_or_store_metadata(cursor, bmk, None, opt, cache_size)
        stats = [xs.GetStat(simout_file, re) for re in db_common.STATS.itervalues()]
        insert_sim_stats(cursor, stats, bid, oid, run, sno, cs)

    conn.commit()
    conn.close()

def parse_sim_data(results_dir, db, is_cache_sweep):
    ''' Parse all simulation data in results_dir and store to SQLite3 DB.  '''
    start = time.time()
    sim_files = get_simout_files(results_dir)
    store_sim_to_db(sim_files, db, is_cache_sweep)
    end = time.time()
    print ""
    print "Total time spent reading simout files: ", (end - start)/60.0, " minutes."

def parse_all(results_dir, db, overwrite_db=True, is_cache_sweep=False):
    create_database(db, overwrite_db)
    parse_profiling_data(results_dir, db, is_cache_sweep)
    parse_sim_data(results_dir, db, is_cache_sweep)

def fix_cache_metadata(db):
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    queries = []
    queries.append("update tcm_prof_length set cache_size=32")
    queries.append("update tcm_total_cycles set cache_size=32")
    queries.append("insert into cache_sizes (size) values (32)")

    for query in queries:
        cursor.execute(query)
    conn.commit()
    conn.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["read", "analyze", "fix-cache"])
    parser.add_argument("--results-dir", help="Directory of results.")
    parser.add_argument("--db", help="SQLite3 DB.")
    parser.add_argument("--is-cache-sweep", action="store_true", help=
        "True if this directory is a sweep of cache sizes.")
    parser.add_argument("--no-overwrite-db", action="store_true", help=
        "Don't overwrite the existing db.")

    args = parser.parse_args()

    if args.mode == "read":
        parse_all(args.results_dir, args.db, is_cache_sweep=args.is_cache_sweep,
                  overwrite_db=not args.no_overwrite_db)
    elif args.mode == "analyze":
        generate_analysis_tables(args.db, (not args.no_overwrite_db))
    elif args.mode == "fix-cache":
        fix_cache_metadata(args.db)

if __name__ == "__main__":
    main()
