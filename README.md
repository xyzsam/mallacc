Mallacc: Accelerating Memory Allocation
=======================================

This was a project to build an in-core hardware accelerator for malloc(). For
more information, please refer to this paper:

Svilen Kanev, Sam (Likun) Xi, Gu-Yeon Wei, and David Brooks.  
Mallacc: Accelerator Memory Allocation.  
Architectural Support for Operating Systems and Programming Languages, April 2017.
[PDF](http://www.samxi.org/papers/kanev_asplos2017.pdf)

This repository contains all the information necessary to reproduce our
experiments and results.

# Dependencies #

Mallacc's dependencies are listed below. If you only want to reproduce the basic
experiments that can be run natively on your machine, you only need to clone
`gperftools`. To run the larger cloud workloads natively, you need to clone
`tcmalloc-cloud-workloads` and the pre-built Xapian databases. Finally, if you
want to reproduce simulation results of Mallacc speedup, you will need to
install XIOSim.

1. [gperftools](https://github.com/s-kanev/gperftools/) - our customized
   version of tcmalloc that uses our newly proposed instructions.  This
   includes a collection of microbenchmarks designed to stress various parts of
   malloc.
2. [tcmalloc-cloud-workloads](https://github.com/xyzsam/tcmalloc-cloud-workloads) -
   A collection of larger ''cloud'' workloads that we used to evaluate malloc
   speedup in more realistic scenarios. Includes Xapian and Masstree.
3. Pre-built Xapian databases of Wikipedia articles.
   * [abstracts](https://storage.googleapis.com/mallacc/wiki_abstracts.tar.xz):
     A search database over wikipedia abstracts only.
   * [full-pages](https://storage.googleapis.com/mallacc/wiki_pages.tar.xz): A
     search database over full Wikipedia pages (text only).
4. [XIOSim](https://github.com/s-kanev/XIOSim) - an x86 simulator. The
   malloc_hw branch includes support for malloc instructions and specialized
   hardware.

# Installation #

gperftools should be installed first, since the entire project revolves around
malloc.  Furthermore, all the microbenchmarks are provided in that repository.

Our system has toolchains installed in non-standard locations, so our build flow
relies on writing `do_configure.sh` scripts that execute `configure` with hard-coded
toolchain locations and other default options. All installable packages come
with such a script. You will need to edit each script to change the toolchain
locations to match your target system, or simply remove all those options if your
toolchains are located in the default directories (e.g. `/usr/bin`). Execute the
`do_configure.sh` script, then run `make` and `make install` from within the build
directory to build and install the package.

Other than the toolchain installation directories, do not remove any of the
other CFLAGS/CXXFLAGS specified, like flags about the ABI, sized deletes, etc.
These are important.

## gperftools ##

Clone the gperftools repository linked above. Installation instructions are in
[README.md](https://github.com/s-kanev/gperftools/README.md).

## Xapian ##

The version of Xapian we used is provided in tcmalloc-cloud-workloads/xapian.
We provide both the Xapian core library, which is used to build search
applications on top of Xapian, and Omega, a search engine that was built using
Xapian's core library. Omega was used to index a dump of Wikipedia's articles
full pages and abstracts; these are the prebuilt Xapian databases that we are
providing (since they take a long time to build and require a large amount of
RAM). We have written some basic client testbenches that run queries over these
databases using the core library.

For build and execution instructions, please see the
[tcmalloc-cloud-workloads](https://github.com/xyzsam/tcmalloc-cloud-workloads)
repo.

## Masstree ##

Masstree is located in tcmalloc-cloud-workloads/masstree. Installation
and execution instructions are in the
[tcmalloc-cloud-workloads](https://github.com/xyzsam/tcmalloc-cloud-workloads)
repo.

## XIOSim ##

XIOSim is the x86 cycle-level simulator that we used to simulate Mallacc
speedups.  You will need to checkout the `malloc_hw` branch, which implements
all the malloc cache functionality.

Please see the XIOSim
[README](https://github.com/s-kanev/XIOSim/tree/malloc_hw/README.md) for
installation instructions.

# Running simulations #

You will need to build gperftools with all Mallacc instructions and optimizations
enabled and link all the benchmarks with this version of tcmalloc. XIOSim is
able to replace any of the placeholder instructions with either the original
code (the baseline case), an implementation of what the corresponding Mallacc
instruction would actually do (e.g. pop the head off a linked list), or an
ideal implementation where the Mallacc functionality is emulated but the
instruction itself takes zero time (the limit study), based on which command line
arguments are passed to the simulator.

To run a simulation, please look at
[run_profiles.py](scripts/sim_runs/run_profiles.py).  You will need to replace
some hard-coded path names to suit your system and environment.
