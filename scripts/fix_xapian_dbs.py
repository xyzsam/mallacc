#!/usr/bin/env python

import argparse
import sqlite3

def fix_db(db):
  conn = sqlite3.connect(db)
  cursor = conn.cursor()

  all_bmks = []
  query = "select id, benchmark from benchmarks where benchmark like 'xapian.query_wiki_pages.%'"
  cursor.execute(query)
  results = cursor.fetchall()
  for row in results:
    all_bmks.append(int(row[0]))

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

  fix_db(args.db)

if __name__ == "__main__":
  main()
