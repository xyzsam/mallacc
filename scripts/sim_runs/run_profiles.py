#!/usr/bin/env python

import shutil
import sys
import os.path
sys.path.append(os.path.join(os.environ["XIOSIM_TREE"], "scripts"))

import argparse
import functools
import getpass
from multiprocessing import Pool

import xiosim_driver as xd
import masstree
import spec
import tcubench
import xapianbench
import condor

USER = getpass.getuser()
# Configuration params
RUN_DIR_ROOT = "/home/%s/malloc_out" % USER  # Benchmarks will execute in subdirectories of this
CONFIG_FILE = "xiosim/config/H.cfg"      # Starting config file (relative to XIOSIM_TREE)

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

def CreateConfuseStrList(strlist):
    funcs = ["\"%s\"" % s for s in strlist]
    concat = "{%s}" % (",".join(funcs))
    return concat

def GetAllIgnoredInstructions(optimization):
    ignored = []
    for sym_name, opt_offsets in _IGNORED_INSNS.iteritems():
        offsets = opt_offsets[optimization]
        for offset in offsets:
            ignored.append("%s+%s" % (sym_name, hex(offset)))
    return ignored

def ApplyOptimizations(xio, optimizations):
    for optimization in optimizations:
        if optimization.startswith("baseline"):
            xio.AddTCMOption("-freelist_mode baseline")
            xio.AddTCMOption("-size_class_mode baseline")
            xio.AddTCMOption("-sampling_mode baseline")

        elif optimization == "ideal-cache":
            xio.AddTCMOption("-size_class_mode ideal")
            xio.AddTCMOption("-freelist_mode ideal")
        elif optimization == "ideal-size-class":
            xio.AddTCMOption("-size_class_mode ideal")
        elif optimization == "ideal-list":
            xio.AddTCMOption("-freelist_mode ideal")
        elif optimization == "ideal-sampling":
            xio.AddTCMOption("-sampling_mode ideal")

        elif optimization == "real-size-class":
            xio.AddTCMOption("-size_class_mode realistic")
        elif optimization == "real-list-only":
            # Runs using the size class cache only for list heads.
            # Generally, we should run with the size class optimization so that
            # it creates valid cache entries that the list heads later
            # populate.  Without the size class optimization, each cache entry
            # is missing allocated size information.
            xio.AddTCMOption("-freelist_mode realistic")
        elif optimization == "real-list":
            xio.AddTCMOption("-freelist_mode realistic")
            xio.AddTCMOption("-size_class_mode realistic")
        elif optimization == "real-sampling":
            xio.AddTCMOption("-sampling_mode realistic")

        elif optimization.startswith("realistic"):
            xio.AddTCMOption("-freelist_mode realistic")
            xio.AddTCMOption("-size_class_mode realistic")
            xio.AddTCMOption("-sampling_mode realistic")
        elif optimization.startswith("limit"):
            xio.AddTCMOption("-freelist_mode ideal")
            xio.AddTCMOption("-size_class_mode ideal")
            xio.AddTCMOption("-sampling_mode ideal")

def EnvAppend(env, extra_env):
    if not env:
        return extra_env
    else:
        return env + " " + extra_env

def CreateDriver(env=""):
    XIOSIM_INSTALL = os.environ["XIOSIM_INSTALL"]
    XIOSIM_TREE = os.environ["XIOSIM_TREE"]
    ARCH = os.environ["TARGET_ARCH"]
    use_own_lib = ("XIOSIM_LOADER_LIB" in os.environ)
    if use_own_lib:
        ld_env = "LD_LIBRARY_PATH=%s/lib" % os.environ["XIOSIM_LOADER_LIB"]
        if ARCH == "k8":
            ld_env += "64"
        env = EnvAppend(env, ld_env)
    xio = xd.XIOSimDriver(XIOSIM_INSTALL, XIOSIM_TREE, ARCH,
                          clean_arch=True, env=env)
    return xio


def WriteBmkConfig(xio, run, run_dir):
    ''' Create a temp benchmark config file in the run directory.'''
    cfg_file = run.GenerateConfigFile(run_dir)
    bmk_cfg = os.path.join(run_dir, "bmk_cfg")
    with open(bmk_cfg, "w") as f:
        for l in cfg_file:
            f.write(l)
    return bmk_cfg


def WriteArchConfig(xio, run_dir, start_config, changes):
    ''' Create a temp config file in the run directory,
    starting with @start_config and applying a dictionary or
    paramter changes. '''
    cfg_file = os.path.join(run_dir, os.path.basename(start_config))
    cfg_contents = xio.GenerateConfigFile(start_config, changes)
    with open(cfg_file, "w") as f:
        for l in cfg_contents:
            f.write(l)
    return cfg_file

def GetRuns(category):
    runs = []
    if category == "ubench":
        runs.append(tcubench.GetRun("ubench.tp"))
        runs.append(tcubench.GetRun("ubench.gauss"))
        runs.append(tcubench.GetRun("ubench.gauss_free"))
        runs.append(tcubench.GetRun("ubench.antagonist"))
        runs.append(tcubench.GetRun("ubench.tp_small"))
        runs.append(tcubench.GetRun("ubench.sized_deletes"))
    elif category == "spec":
        runs.append(spec.GetRun("400.perlbench.diffmail"))
        runs.append(spec.GetRun("465.tonto.tonto"))
        runs.append(spec.GetRun("471.omnetpp.omnetpp"))
        runs.append(spec.GetRun("483.xalancbmk.ref"))
    elif category == "xapian":
        runs.append(xapianbench.GetRun("xapian.query_tiny_index"))
        runs.append(xapianbench.GetRun("xapian.query_wiki_abstracts"))
        runs.append(xapianbench.GetRun("xapian.query_wiki_pages.1"))
        runs.append(xapianbench.GetRun("xapian.query_wiki_pages.2"))
        runs.append(xapianbench.GetRun("xapian.query_wiki_pages.3"))
        runs.append(xapianbench.GetRun("xapian.query_wiki_pages.4"))
        runs.append(xapianbench.GetRun("xapian.query_wiki_pages.5"))
        runs.append(xapianbench.GetRun("xapian.query_wiki_pages.6"))
        runs.append(xapianbench.GetRun("xapian.query_wiki_pages.7"))
        runs.append(xapianbench.GetRun("xapian.query_wiki_pages.8"))
        runs.append(xapianbench.GetRun("xapian.query_wiki_pages.9"))
        runs.append(xapianbench.GetRun("xapian.query_wiki_pages.10"))
    elif category == "masstree":
        runs.append(masstree.GetRun("masstree.ycsb"))
        # runs.append(masstree.GetRun("masstree.same"))
    return runs

def GetNextRunNumber(directory):
    ''' Return the next run number in this top level run directory. '''
    if not os.path.exists(directory):
        return 0
    dirs = os.listdir(directory)
    run_num = 0
    while str(run_num) in dirs:
        run_num += 1
    return run_num

def ConfigureSimRun(bmk_run, optimizations, delete_existing_runs, trace):
    # Create a brand new directory to execute in
    top_dir = os.path.join(RUN_DIR_ROOT, bmk_run.name, "-".join(optimizations))
    if delete_existing_runs and os.path.exists(top_dir):
        shutil.rmtree(top_dir)
    run_num = GetNextRunNumber(top_dir)
    run_dir = os.path.join(top_dir, str(run_num))
    bmk_run.CreateRunDir(run_dir)
    xio = CreateDriver(env=bmk_run.env)

    # Grab a benchmark description file
    bmk_cfg = WriteBmkConfig(xio, bmk_run, run_dir)
    xio.AddBmks(bmk_cfg)

    # We want to get results in a separate results directory, a tad cleaner
    outfile = os.path.join(run_dir, "%s.sim.out" % bmk_run.name)
    prof_out = os.path.join(run_dir, "%s.prof" % bmk_run.name)
    ztrace_prefix = os.path.join(run_dir, "ztrace")
    repl = {
        "system_cfg.output_redir" : outfile,
        "system_cfg.profiling_cfg.file_prefix" : prof_out,
        "system_cfg.profiling_cfg.start" : CreateConfuseStrList(profiled_funcs),
        "core_cfg.exec_cfg.exeu sampling.latency" : "9983",
        "core_cfg.exec_cfg.exeu sampling.rate" : "9983",
    }

    orig_cfg = os.path.join(xio.GetTreeDir(), CONFIG_FILE)
    arch_cfg = WriteArchConfig(xio, run_dir, orig_cfg, repl)
    xio.AddConfigFile(arch_cfg)

    xio.AddPinOptions()
    xio.AddIgnoreOptions()
    if trace:
      xio.AddTraceFile("static_trace")
    ApplyOptimizations(xio, optimizations)
    if bmk_run.NeedsPinPoints():
        # Grab a pinpoints file from the original SPEC repo
        ppfile = os.path.join(bmk_run.directory, "%s.pintool.1.pp" % bmk_run.name)
        xio.AddPinPointFile(ppfile)
    elif bmk_run.NeedsROI():
        xio.AddROIOptions()

    xio.cmd += "-pthreads "
    # for now, hits some weird deadlock
    if bmk_run.bmk.startswith("masstree"):
        xio.cmd += "-timing_virtualization false "


    xio.DisableSpeculation()

    # Do a dry run so xio knows about the run directory
    out_file = os.path.join(run_dir, "harness.out")
    err_file = os.path.join(run_dir, "harness.err")
    xio.Exec(stdout_file=out_file, stderr_file=err_file, cwd=run_dir, dry_run=True)
    return xio


def RunRun(xio):
    run_dir = xio.GetRunDir()
    out_file = os.path.join(run_dir, "harness.out")
    err_file = os.path.join(run_dir, "harness.err")
    ret = xio.Exec(stdout_file=out_file, stderr_file=err_file, cwd=run_dir)
    if ret != 0:
        raise Exception("XIOSim run failed (errcode %d)" % ret)

def ConfigureAndRun(bmks, opts, how_to_run, directory="",
                    delete_existing_runs=False, trace=False):
    opts = opts.split(",")
    bmk_runs = GetRuns(bmks)

    # sampling_env = "TCMALLOC_SAMPLE_PARAMETER=0"
    # for run in bmk_runs:
    #     if not "TCMALLOC_SAMPLE_PARAMETER" in run.env:
    #         run.env = EnvAppend(run.env, sampling_env)

    # Override the default directory with an alternative one.
    if directory:
        for bmk_run in bmk_runs:
            bmk_run.directory = directory

    sim_runs = [ConfigureSimRun(bmk_run, opts, delete_existing_runs, trace) for bmk_run in bmk_runs]

    if how_to_run == "native":
        pool = Pool(processes=4)
        pool.map(RunRun, sim_runs)
#        for r in sim_runs:
#            RunRun(r)
    elif how_to_run == "condor":
        print "Preparing to write condor out."
        condor_out = "sim.con"
        with open("sim.con", "w") as f:
            condor.Boilerplate(f)
            for r in sim_runs:
                condor.AddSimRun(r, f)
        print "Output written to %s." % condor_out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bmks", choices=["ubench", "spec", "xapian", "masstree"], help=
        "Which set of benchmarks to run.")
    parser.add_argument("opt", help="Select the optimization(s) to simulate "
        "(baseline = none, pop = Ignore the pop load/stores. Separate with "
        "commas.")
    parser.add_argument("run", choices=["native", "condor"], help=
        "How to execute runs -- natively on the current host, or through condor",
        default="native")
    parser.add_argument("--delete-existing-runs", type=bool, default=False,
        help="Delete all existing run directories and start from run 0.")
    parser.add_argument("--trace", type=bool, default=False,
        help="Enable static and dynamic tracing, if the binary supports it.")
    args = parser.parse_args()
    ConfigureAndRun(args.bmks, args.opt, args.run,
                    delete_existing_runs=args.delete_existing_runs,
                    trace=args.trace)

if __name__ == "__main__":
  main()
