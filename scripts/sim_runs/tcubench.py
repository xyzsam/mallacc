#!/usr/bin/python
import getpass
import os.path

import bmk

USER = getpass.getuser()

# Configuration params
benchDir = '/home/%s/gperftools/output_all_magic' % USER

class TCUBenchRun(bmk.BenchmarkRun):

    def __init__(self, bmk, executable, args, input, output, error, name, expected, env=""):
        super(TCUBenchRun, self).__init__(bmk, executable, args, input, output, error, name, expected, env=env)
        self.needs_pin_points = False
        self.needs_roi = True

    def GetDir(self):
        return benchDir

runs = [
    TCUBenchRun('ubench', 'tp', '', '', '', '', 'tp', [], ""),
    TCUBenchRun('ubench', 'tp_dep', '', '', '', '', 'tp_dep', [], ""),
    TCUBenchRun('ubench', 'gauss', '', '', '', '', 'gauss', [], ""),
    TCUBenchRun('ubench', 'gauss_free', '', '', '', '', 'gauss_free', [], ""),
    TCUBenchRun('ubench', 'antagonist', '', '', '', '', 'antagonist', [], ""),
    TCUBenchRun('ubench', 'tp_small', '', '', '', '', 'tp_small', [], ""),
    TCUBenchRun('ubench', 'sized_deletes', '', '', '', '', 'sized_deletes', [], ""),
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
