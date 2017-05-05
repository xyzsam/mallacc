#!/usr/bin/python
import getpass
import os.path

import bmk

USER = getpass.getuser()

# Configuration params
# benchDir = '/home/%s/masstree-beta/install_all_magic' % USER
benchDir = '/home/%s/masstree-beta/install_tb' % USER

class MTBenchRun(bmk.BenchmarkRun):

    def __init__(self, bmk, executable, args, input, output, error, name, expected, env=""):
        super(MTBenchRun, self).__init__(bmk, executable, args, input, output, error, name, expected, env=env)
        self.needs_pin_points = False
        self.needs_roi = False

    def GetDir(self):
        return benchDir

runs = [
    MTBenchRun('masstree', 'mttest_integrated', '-j1 mycsba masstree', '', '', '', 'ycsb', [".git"],
        "TBENCH_QPS=100 TBENCH_MAXREQS=3000 TBENCH_WARMUPREQS=14000 TBENCH_MINSLEEPNS=10000"),
    # MTBenchRun('masstree', 'mttest', '-j1 --no-notebook -l 10000000 same', '', '', '', 'same', [".git"], ""),
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
