#!/usr/bin/env python
from __future__ import print_function

_MACH_LIST = ["rb%d.int.seas.harvard.edu" % num for num in range(22, 27)]
_MACH_REQ = ["(Machine == \"%s\")" % mach for mach in _MACH_LIST]
_MACH_REQ_ALL = " || ".join(_MACH_REQ)

def Boilerplate(f):
    header = [
"Universe        = vanilla",
"Requirements    = (OpSys == \"LINUX\") && (Arch == \"X86_64\") && (%s)" % _MACH_REQ_ALL,
"Notification    = Never",
"request_cpus    = 3",
"GetEnv          = False",
"",
"###################################################################################################",
""
]

    for line in header:
        print(line, file=f)

def AddSimRun(run, f):
    spl = run.cmd.split()
    # Condor tries to be smart and replaces /usr/bin/arch
    # Just ignore it, it appears that condor disables ASLR anyway.
    executable = spl[3]
    args = " ".join(spl[4:])

    runtext = [
"InitialDir      = %s" % run.GetRunDir(),
"+WorkingDir      = \"$(InitialDir)\"",
"Executable      = %s" % executable,
"Arguments       = %s" % args,
"Output          = harness.out",
"Error           = harness.err",
"Queue",
""
]
    for line in runtext:
        print(line, file=f)
