#!/usr/bin/env python

import getpass
import os.path

import bmk

USER = getpass.getuser()

benchDir = '/home/%s/xapian/benchmarks' % USER

class XapianBenchRun(bmk.BenchmarkRun):

    def __init__(self, bmk, executable, args, input, output, error, name, expected, env=""):
        super(XapianBenchRun, self).__init__(bmk, executable, args, input, output, error, name, expected, env=env)
        self.needs_pin_points = False  # For now.
        self.needs_roi = True

    def GetDir(self):
        return os.path.join(benchDir, self.executable)

runs = [
    XapianBenchRun('xapian', 'query_tiny_index', 'tiny_index queries.txt', '', '', '', 'query_tiny_index', ['tiny_index', 'queries.txt'], ''),
    XapianBenchRun('xapian', 'query_wiki_abstracts', 'stub_database.db queries_50.txt', '', '', '', 'query_wiki_abstracts', ['stub_database.db', 'queries_50.txt'], ''),
    XapianBenchRun('xapian.query_wiki_pages', 'query_wiki_pages', 'stub_database.db queries_25.txt', '', '', '', '25q', ['stub_database.db', 'queries_25.txt'], ''),
    XapianBenchRun('xapian.query_wiki_pages', 'query_wiki_pages', 'stub_database.db query_1.txt', '', '', '', '1', ['stub_database.db', 'query_1.txt'], ''),
    XapianBenchRun('xapian.query_wiki_pages', 'query_wiki_pages', 'stub_database.db query_2.txt', '', '', '', '2', ['stub_database.db', 'query_2.txt'], ''),
    XapianBenchRun('xapian.query_wiki_pages', 'query_wiki_pages', 'stub_database.db query_3.txt', '', '', '', '3', ['stub_database.db', 'query_3.txt'], ''),
    XapianBenchRun('xapian.query_wiki_pages', 'query_wiki_pages', 'stub_database.db query_4.txt', '', '', '', '4', ['stub_database.db', 'query_4.txt'], ''),
    XapianBenchRun('xapian.query_wiki_pages', 'query_wiki_pages', 'stub_database.db query_5.txt', '', '', '', '5', ['stub_database.db', 'query_5.txt'], ''),
    XapianBenchRun('xapian.query_wiki_pages', 'query_wiki_pages', 'stub_database.db query_6.txt', '', '', '', '6', ['stub_database.db', 'query_6.txt'], ''),
    XapianBenchRun('xapian.query_wiki_pages', 'query_wiki_pages', 'stub_database.db query_7.txt', '', '', '', '7', ['stub_database.db', 'query_7.txt'], ''),
    XapianBenchRun('xapian.query_wiki_pages', 'query_wiki_pages', 'stub_database.db query_8.txt', '', '', '', '8', ['stub_database.db', 'query_8.txt'], ''),
    XapianBenchRun('xapian.query_wiki_pages', 'query_wiki_pages', 'stub_database.db query_9.txt', '', '', '', '9', ['stub_database.db', 'query_9.txt'], ''),
    XapianBenchRun('xapian.query_wiki_pages', 'query_wiki_pages', 'stub_database.db query_10.txt', '', '', '', '10', ['stub_database.db', 'query_10.txt'], ''),
]

def GetRun(name):
    res = None
    for curr_run in runs:
        if curr_run.name == name:
            res = curr_run
            break
    if res == None:
        raise ValueError('No benchmark %s in %s' % (name, __name__))
    return res
