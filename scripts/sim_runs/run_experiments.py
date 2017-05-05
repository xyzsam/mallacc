#!/usr/bin/env python
#
# Runs all the ubench experiments.

import argparse
import getpass
import os
import run_profiles

# Sequence of optimizations to run.
opts = {
#     "ubench" : ["baseline", "real-sampling", "real-size-class", "real-list", "realistic", "limit"]
#     "spec" : ["baseline", "all"],
#     "xapian": ["baseline", "all"],
    "masstree": ["baseline"]
}

user = getpass.getuser()

def RunOptimized(bmk_set, opt, clean_start):
    run_profiles.ConfigureAndRun(
        bmk_set, opt, "condor",
        delete_existing_runs=clean_start)
    condor_file = "sim_%s_%s.con" % (bmk_set, opt)
    os.rename("sim.con", condor_file)
    os.system("condor_submit %s" % condor_file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-runs", type=int, default=1, help="Number of runs.")
    args = parser.parse_args()

    # Start with a clean slate of runs each time.
    for bmk_set, bmk_opts in opts.items():
        print bmk_set, bmk_opts
        for opt in bmk_opts:
            for i in range(args.num_runs):
                RunOptimized(bmk_set, opt, i==0)

if __name__ == "__main__":
    main()
