#!/usr/bin/env python

import os
import numpy as np
import sqlite3
import sys
sys.path.append(os.path.join(os.environ["XIOSIM_TREE"], "scripts"))
import xiosim_stat as xs

PROF_TABLE_NAME = "tcm_prof_length"
PROF_TOTAL_CYCLES_TABLE_NAME = "tcm_total_cycles"
PROF_FASTPATH_CYCLES_TABLE_NAME = "tcm_fastpath_cycles"

SIM_STATS_TABLE_NAME = "sim_stats"
STATS = {
    "sim_cycle" : xs.PerfStatRE("c0.sim_cycle"),
    "all_insn" : xs.PerfStatRE("all_insn"),
    "size_hits": xs.PerfStatRE("c0.size_class_cache.size_hits"),
    "size_misses": xs.PerfStatRE("c0.size_class_cache.size_misses"),
    "size_insertions": xs.PerfStatRE("c0.size_class_cache.size_insertions"),
    "head_hits": xs.PerfStatRE("c0.size_class_cache.head_hits"),
    "head_misses": xs.PerfStatRE("c0.size_class_cache.head_misses"),
    "head_updates": xs.PerfStatRE("c0.size_class_cache.head_updates"),
}

profiled_funcs = ["tc_malloc",
                  "tc_free",
                  "tc_new",
                  "tc_delete",
                  "tc_new_nothrow",
                  "tc_delete_nothrow",
                  "tc_newarray",
                  "tc_deletearray",
                  "tc_newarray_nothrow",
                  "tc_deletearray_nothrow"
]
malloc_funcs = ["tc_malloc", "tc_new", "tc_new_nothrow",
                "tc_newarray", "tc_newarray_nothrow"]
free_funcs = ["tc_free", "tc_delete", "tc_delete_nothrow",
              "tc_deletearray", "tc_deletearray_nothrow"]
benchmark_labels = {
    "WSC (Kanev et al.)": "WSC (Kanev et al.)",
    "400.perlbench.diffmail": "400.perlbench",
    "465.tonto.tonto": "465.tonto",
    "471.omnetpp.omnetpp": "471.omnetpp",
    "483.xalancbmk.ref": "483.xalancbmk",
    "masstree.same": "masstree.same",
    "masstree.wcol1": "masstree.wcol1",
    "xapian.query_wiki_abstracts": "xapian.abstracts",
    "xapian.query_wiki_pages": "xapian.pages",
    "ubench.tp": "ubench.tp",
    "ubench.tp_small": "ubench.tp_small",
    "ubench.sized_deletes": "ubench.sized_deletes",
    "ubench.gauss": "ubench.gauss",
    "ubench.gauss_free": "ubench.gauss_free",
    "ubench.antagonist": "ubench.antagonist",
}


def get_metadata_ids(cursor, benchmark, functions, optimization):
    """ Get the ids of the metadata fields. """
    def list_to_str_tuple(l):
      # Convert a list to stringified tuple.
      if len(l) == 0:
        return "()"
      elif len(l) == 1:
        return str(tuple(l)).replace(",", "")
      else:
        return str(tuple(l))

    bid_query = "select id from benchmarks where benchmark = ?"
    fid_query = "select id from functions where function in %s"
    oid_query = "select id from optimizations where optimization = ?"

    if benchmark != None:
        cursor.execute(bid_query, (benchmark,))
        result = cursor.fetchone()
        bid = result[0] if result else None
    else:
        bid = None

    if functions != None:
        cursor.execute(fid_query % list_to_str_tuple(functions))
        fid = cursor.fetchall()
        fids = [e[0] for e in fid]
        fid = list_to_str_tuple(fids)
    else:
        fid = None

    if optimization != None:
        cursor.execute(oid_query, (optimization,))
        result = cursor.fetchone()
        oid = result[0] if result else None
    else:
        oid = None

    return fid, bid, oid

def get_cycles_data(cursor, benchmark, functions, optimization,
                    run=None, fastpath_only=False, limit=None):
    """ Get cycles data for some selection. """
    fid, bid, oid = get_metadata_ids(cursor, benchmark, functions, optimization)
    query = ("select cycles from %s where benchmark=? and function in %s and "
             "optimization=? ") % (PROF_TABLE_NAME, fid)
    query_values = [bid, oid]
    if run:
        query += "and run=? "
        query_values.append(run)
    if fastpath_only:
        query += " and cycles < 100"

    query += " order by rowid asc"

    if limit:
        query += " limit ?"
        query_values.append(limit)

    cursor.execute(query, tuple(query_values))
    results = cursor.fetchall()
    data = []
    for row in results:
        data.append(row[0])

    return np.array(data).astype(float)

def get_total_cycles_data(cursor, benchmark, functions, optimization, run=None, fastpath_only=False, skip_analysis_tables=False, cache_size=None):
    """ Get total cycles for a selection. """
    fid, bid, oid = get_metadata_ids(cursor, benchmark, functions, optimization)

    use_analysis_tables = False

    if not skip_analysis_tables:
        if fastpath_only:
            table = PROF_FASTPATH_CYCLES_TABLE_NAME
        else:
            table = PROF_TOTAL_CYCLES_TABLE_NAME
        # Check if we have a total_cycles or fastpath_cycles table in the DB. If
        # so, use that.  Otherwise, compute the total cycles dynamically.
        query = "select count(*) from SQLITE_MASTER where type='table' and name='%s'" % table
        cursor.execute(query)
        results = cursor.fetchone()[0]
        use_analysis_tables = (results == 1)

    if use_analysis_tables:
        # We have a total_cycles or fastpath_cycles table. Use it.
        query = ("select sum(cycles), sum(calls) from %s "
                 "where benchmark=? and "
                 "function in %s and optimization=?")
    else:
        # Fallback to computing the total cycles.
        table = PROF_TABLE_NAME
        query = ("select sum(cycles), count(*) from %s where benchmark=? and "
                 "function in %s and optimization=?")

    if run != None:
        query += " and run=?"

    if fastpath_only and not use_analysis_tables:
        # We want fastpath only but we don't have the table.  NOTE: This is
        # only valid if we differentiate speedup between malloc and free!!
        query += " and cycles < 100"

    if cache_size != None:
        query += " and cache_size=?"

    query_values = (bid, oid)
    if run != None:
        query_values += (run,)
    if cache_size != None:
        query_values += (cache_size,)

    print query % (table, fid), query_values
    cursor.execute(query % (table, fid), query_values)
    results = cursor.fetchone()
    # print results
    if results[0] == None or results[1] == None:
        return (0, 0)
    total_cycles = float(results[0])
    total_rows = float(results[1])
    return total_cycles, total_rows

def get_all_benchmarks(db):
    """ Return all benchmarks from the DB. """
    query = "select benchmark, id from benchmarks order by benchmark asc"

    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    cursor.execute(query)
    results = cursor.fetchall()
    benchmarks = [row[0] for row in results]
    ids = [row[1] for row in results]
    conn.close()
    return benchmarks, ids

def get_all_runs(db):
    """ Return all runs from the DB. """
    query = "select distinct(id) from runs order by id asc"

    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    cursor.execute(query)
    results = cursor.fetchall()
    runs = [row[0] for row in results]
    conn.close()
    return runs

def get_all_opts(db):
    """ Return all optimizations from the DB. """
    query = "select optimization, id from optimizations order by id asc"

    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    cursor.execute(query)
    results = cursor.fetchall()
    opts = [row[0] for row in results]
    ids = [row[1] for row in results]
    conn.close()
    return opts, ids

def get_spec_benchmarks(db):
    """ Get the list of spec benchmarks from the DB. """
    query = ("select benchmark, id from benchmarks "
             "where benchmark like \"4%\" "
             "order by benchmark asc")
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    cursor.execute(query)
    results = cursor.fetchall()
    benchmarks = [row[0] for row in results]
    ids = [row[1] for row in results]
    conn.close()
    return benchmarks, ids

def get_ubenchmarks(db):
    """ Get the list of ubenchmarks from the DB. """
    query = ("select benchmark, id from benchmarks "
             "where benchmark like \"ubench.%\" "
             "order by benchmark asc")
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    cursor.execute(query)
    results = cursor.fetchall()
    benchmarks = [row[0] for row in results]
    ids = [row[1] for row in results]
    conn.close()
    return benchmarks, ids

def get_large_benchmarks(db):
    """ Get the list of ubenchmarks from the DB. """
    query = ("select benchmark, id from benchmarks "
             "where benchmark not like \"ubench.%\" "
             "order by benchmark asc")
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    cursor.execute(query)
    results = cursor.fetchall()
    benchmarks = [row[0] for row in results]
    ids = [row[1] for row in results]
    conn.close()
    return benchmarks, ids

def get_total_sim_cycles(cursor, benchmark, optimization, run=None, sim_slice=None):
    """ Get simulated cycles data for some selection. """
    _, bid, oid = get_metadata_ids(cursor, benchmark, None, optimization)
    query = ("select sum(sim_cycle) from %s where "
             "benchmark=? and "
             "optimization=? ") % SIM_STATS_TABLE_NAME

    query_values = [bid, oid]
    if run != None:
        query += " and run=?"
        query_values.append(run)

    if sim_slice:
        query += "and slice=? "
        query_values.append(sim_slice)

    query_values = tuple(query_values)
    #print query, query_values
    cursor.execute(query, query_values)
    results = cursor.fetchone()
    return results[0]

def get_sim_stats(cursor, stats, benchmark, optimization, run, sim_slice=1, cache_size=None):
    _, bid, oid = get_metadata_ids(cursor, benchmark, None, optimization)
    stats_str = ",".join(stats)

    query = ("select %s from %s where "
             "benchmark=? and "
             "optimization=? and "
             "run=? ") % (stats_str, SIM_STATS_TABLE_NAME)
    query_values = (bid, oid, run)

    if sim_slice:
        query += "and slice=? "
        query_values = (query_values[0], query_values[1], query_values[2], sim_slice)

    if cache_size:
        query += "and cache_size=? "
        query_values = (query_values[0], query_values[1], query_values[2], sim_slice, cache_size)

    cursor.execute(query, query_values)
    results = cursor.fetchone()
    return results

def get_cache_sizes(cursor):
    query = "select distinct(size) from cache_sizes order by size"
    cursor.execute(query)
    results = cursor.fetchall()
    sizes = []
    for row in results:
        sizes.append(int(row[0]))
    return sizes
