#!/usr/bin/env python
#
# Merge the ten chunks of Xapian full pages together.
#
# Make sure we keep different functions/optimizations/runs separate!

import argparse
import sqlite3

import parse_results

def get_all_bmks(cursor):
  all_bmks = []
  query = "select id, benchmark from benchmarks where benchmark like 'xapian.query_wiki_pages.%'"
  cursor.execute(query)
  results = cursor.fetchall()
  for row in results:
    all_bmks.append(int(row[0]))

  return all_bmks

def get_all_opts(cursor):
  all_opts = []
  query = "select id from optimizations"
  cursor.execute(query)
  results = cursor.fetchall()
  for row in results:
    all_opts.append(int(row[0]))
  return all_opts

def get_all_runs(cursor):
  all_runs = []
  query = "select id from runs"
  cursor.execute(query)
  results = cursor.fetchall()
  for row in results:
    all_runs.append(int(row[0]))
  return all_runs

def move_tables(db):
  """ Move data from "fixed" tables to the regular tables.

  This assumes the relevant data from the regular tables has already been deleted.
  """
  conn = sqlite3.connect(db)
  cursor = conn.cursor()

  tcm_insert_query = ("insert into tcm_total_cycles (benchmark, function, optimization, run, "
                      "cache_size, cycles, calls) "
                      "select benchmark, function, optimization, run, cache_size, cycles, calls "
                      "from tcm_total_cycles_fixed")

  sim_insert_query = ("insert into sim_stats (size_hits, head_hits, "
                      "size_insertions, head_updates, head_misses, sim_cycle, "
                      "size_misses, all_insn, benchmark, optimization, run, "
                      "cache_size, slice) select size_hits, head_hits, "
                      "size_insertions, head_updates, head_misses, sim_cycle, "
                      "size_misses, all_insn, benchmark, optimization, run, "
                      "cache_size, slice from sim_stats_fixed")

  cursor.execute(tcm_insert_query)
  # cursor.execute(sim_insert_query)
  conn.commit()
  conn.close()


def fix_tcm_total_cycles(db):
  conn = sqlite3.connect(db)
  cursor = conn.cursor()

  all_bmks = get_all_bmks(cursor)
  all_runs = get_all_runs(cursor)
  all_opts = get_all_opts(cursor)
  bmk_str = str(tuple(all_bmks))

  # CREATE TABLE tcm_total_cycles (id integer primary key, cycles integer,calls
  # integer,benchmark integer,function integer,optimization integer,run
  # integer,cache_size integer);

  cursor.execute("delete from tcm_total_cycles_fixed")

  insert_query = ("insert into tcm_total_cycles_fixed (function, optimization, run, "
                  "cache_size, cycles, calls) "
                  "select function, optimization, run, cache_size, sum(cycles), sum(calls) "
                  "from tcm_total_cycles where benchmark in %s group by "
                  "optimization, run, function, cache_size") % bmk_str

  cursor.execute(insert_query)
  conn.commit()
  conn.close()

def fix_sim_stats(db):
  conn = sqlite3.connect(db)
  cursor = conn.cursor()

  all_bmks = get_all_bmks(cursor)

  new_bmk_id = parse_results.get_or_store_metadata(
      cursor, "xapian.query_wiki_pages", None, None)[0]

  all_runs = get_all_runs(cursor)
  all_opts = get_all_opts(cursor)

  # create_table = ("CREATE TABLE sim_stats (id integer primary key, size_hits integer, "
  #                 "head_hits integer,size_insertions integer,head_updates integer, "
  #                 "head_misses integer,sim_cycle integer,size_misses integer, "
  #                 "all_insn integer,benchmark integer,optimization integer, "
  #                 "run integer,cache_size integer,slice integer)")

  cache_size = 32
  sim_slice = 0

  sum_query = ("select sum(size_hits), sum(head_hits), sum(size_insertions), "
               "sum(head_updates), sum(head_misses), sum(sim_cycle), "
               "sum(size_misses), sum(all_insn) from sim_stats where "
               "run=%d and benchmark in %s and optimization=%d")

  insert_query = ("insert into sim_stats_fixed (size_hits, head_hits, "
                  "size_insertions, head_updates, head_misses, sim_cycle, "
                  "size_misses, all_insn, benchmark, optimization, run, "
                  "cache_size, slice) values (%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d)")

  cursor.execute("delete from sim_stats_fixed")

  bmk_str = str(tuple(all_bmks))
  for opt in all_opts:
    print "Optimziation: ", opt
    for run in all_runs:
      temp_query = sum_query % (run, bmk_str, opt)
      cursor.execute(temp_query)
      results = cursor.fetchone()

      if None in results:
        continue

      results_list = list(results)
      results_list.append(new_bmk_id)
      results_list.append(opt)
      results_list.append(run)
      results_list.append(cache_size)
      results_list.append(sim_slice)

      cursor.execute(insert_query % tuple(results_list))

  conn.commit()
  conn.close()

def fix_db_wrong(db):
  conn = sqlite3.connect(db)
  cursor = conn.cursor()

  all_bmks = get_all_bmks(cursor)
  print all_bmks

  new_bmk = min(all_bmks)
  queries = []
  run = 0
  for bmk in all_bmks:
    queries.append("update tcm_prof_length set run = %d, benchmark = %d where benchmark = %d and run = 0" % (run, new_bmk, bmk))
    queries.append("update sim_stats set run = %d, benchmark = %d where benchmark = %d and run = 0" % (run, new_bmk, bmk))
    queries.append("update tcm_total_cycles set run = %d, benchmark = %d where benchmark = %d and run = 0" % (run, new_bmk, bmk))
    run += 1
    queries.append("update tcm_prof_length set run = %d, benchmark = %d where benchmark = %d and run = 1" % (run, new_bmk, bmk))
    queries.append("update sim_stats set run = %d, benchmark = %d where benchmark = %d and run = 1" % (run, new_bmk, bmk))
    queries.append("update tcm_total_cycles set run = %d, benchmark = %d where benchmark = %d and run = 1" % (run, new_bmk, bmk))
    run += 1

  for query in queries:
    print query
    cursor.execute(query)

  all_bmks = [str(b) for b in all_bmks]
  bmk_str = ",".join(all_bmks)
  queries = []
  queries.append("delete from benchmarks where id in (%s)" % bmk_str)
  queries.append("insert into benchmarks (id, benchmark) values (%d, '%s')" % (new_bmk, "xapian.query_wiki_pages"))

  for query in queries:
    print query
    cursor.execute(query)

  conn.commit()

  conn.close()

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("db")
  args = parser.parse_args()

  # fix_sim_stats(args.db)
  # fix_tcm_total_cycles(args.db)
  move_tables(args.db)

if __name__ == "__main__":
  main()
