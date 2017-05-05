#!/usr/bin/env python

import argparse
import fnmatch
import math
import re
import os
import warnings
import time
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from colormap_helper import get_colormap

import mysql
import mysql.connector as mconn

conn = None
cursor = None

def connect():
    global conn
    global cursor

    conn = mconn.connect(
        user="user", password="password", database="db", host="localhost")
    cursor = conn.cursor()

def close():
    conn.close()

def parse_filepath(filepath):
    """ Get benchmark, profiled function number, optimization, and run from filepath. """
    path = filepath.split("/")
    fname = path[-1]
    run = path[-2]
    opt = path[-3]
    benchmark = path[-4]
    func_no = int(fname.split(".")[-1])
    return benchmark, func_no, opt, run

def get_or_store_metadata(cursor, conn, benchmark, function_no, optimization):
    """ Returns ids for benchmark, function, and optimization.

    If they are not present in their tables, they are added to the tables.

    Args:
        benchmark: Name of benchmark as string.
        function_no: The suffix index of the profiling output file.
        optimization: Name of the optimization, or "baseline" if none.
    """
    query = "select * from benchmarks where benchmark=?"
    cursor.execute(query, (benchmark,))
    result = cursor.fetchone()
    if not result:
        cursor.execute("insert into benchmarks (benchmark) values (?)", (benchmark,))
        cursor.execute(query, (benchmark,))
        result = cursor.fetchone()
    benchmark_id = result[0]

    query = "select * from functions where id=? and function=?"
    values = (function_no, profiled_funcs[function_no])
    cursor.execute(query, values)
    result = cursor.fetchone()
    if not result:
        cursor.execute("insert into functions (id, function) values (?,?)", values)
        cursor.execute(query, values)
        result = cursor.fetchone()
    function_id = result[0]

    query = "select * from optimizations where optimization=?"
    cursor.execute(query, (optimization,))
    result = cursor.fetchone()
    if not result:
        cursor.execute("insert into optimizations (optimization) values (?)", (optimization,))
        cursor.execute(query, (optimization,))
        result = cursor.fetchone()
    optimization_id = result[0]

    return benchmark_id, function_id, optimization_id


def create_database(overwrite_db):
    """ Creates the database. """
    # Drop all tables if they already exist.
    cursor.execute("drop table if exists data")
    cursor.execute("drop table if exists total_cycles")
    cursor.execute("drop table if exists benchmarks")
    cursor.execute("drop table if exists functions")
    cursor.execute("drop table if exists optimizations")
    cursor.execute("drop table if exists runs")
    conn.commit()

    query = ("create table data (id integer auto_increment, "
                                "cycles integer, "
                                "benchmark integer, "
                                "function integer, "
                                "optimization integer,"
                                "run integer,"
                                "primary key (id))")
    cursor.execute(query)
    query = ("create table total_cycles (id integer auto_increment, "
                                        "cycles integer,"
                                        "calls integer,"
                                        "benchmark integer,"
                                        "function integer,"
                                        "optimization integer,"
                                        "run integer,"
                                        "primary key (id))")
    cursor.execute(query)
    query = ("create table benchmarks (id integer primary key auto_increment, benchmark varchar(50)) ")
    cursor.execute(query)
    query = ("create table functions (id integer primary key auto_increment, function varchar(20)) ")
    cursor.execute(query)
    query = ("create table optimizations (id integer primary key auto_increment, optimization varchar(20)) ")
    cursor.execute(query)
    query = ("create table runs (id integer)")
    cursor.execute(query)

    conn.commit()

def store_to_db(files):
    data_query = ("insert into data (cycles, benchmark, function, optimization, run) "
                  "values (%s, %s, %s, %s, %s)")
    # Compute total cycles as we go.
    total_cycles_query = ("insert into total_cycles (cycles, calls, benchmark, "
                          "function, optimization, run) values (%s, %s, %s, %s, %s, %s)")
    runs_query = "insert into runs (id) values (%s)"
    # Track runs
    runs = set()
    for fpath in files:
        total_cycles = 0
        total_calls = 0
        print "Parsing ", fpath
        benchmark, func_no, opt, run = parse_filepath(fpath)
        bid, fid, oid = get_or_store_metadata(cursor, conn, benchmark, func_no, opt)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cycles = np.genfromtxt(fpath, delimiter=",")
        total_cycles = sum(cycles)
        total_calls = len(cycles)

        cycles_array = [(c, bid, fid, oid, run) for c in cycles]
        cursor.executemany(data_query, cycles_array)

        # Update the total cycles.
        cursor.execute(total_cycles_query, (
                total_cycles, total_calls, bid, fid, oid, run))
        runs.add(run)

    # Write all runs to the runs table.
    for run in runs:
        cursor.execute(runs_query, (run,))

    conn.commit()

def get_profiling_files(results_dir):
    """ Returns all profiling output files under results_dir, recursively. """
    matches = []
    for root, dirnames, filenames in os.walk(results_dir):
        for filename in fnmatch.filter(filenames, 'ubench.*.prof.*'):
            matches.append(os.path.join(root, filename))
    return matches

def parse_profiling_data(results_dir, overwrite_db=True):
    ''' Parse all profiling data in results_dir and store to SQLite3 DB.  '''
    profiling_files = get_profiling_files(results_dir)
    create_database(overwrite_db)
    store_to_db(profiling_files)

def main():
    connect()
    parse_profiling_data("/home/samxi/malloc_out")
    close()

if __name__ == "__main__":
    main()
