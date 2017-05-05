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

Dependencies
------------

1. [gperftools](https://github.com/s-kanev/gperftools/tree/malloc_hw) - our
   customized version of tcmalloc that uses our newly proposed instructions.
   This includes a collection of microbenchmarks designed to stress various
   parts of malloc.
2. [tcmalloc-cloud-workloads](https://github.com/xyzsam/tcmalloc-cloud-workloads) -
   A collection of larger ``cloud'' workloads that we used to evaluate malloc
   speedup in more realistic scenarios. Includes Xapian and Masstree.
3. Pre-built Xapian databases of Wikipedia articles.
   * [abstracts](https://storage.googleapis.com/mallacc/wiki_abstracts.tar.xz):
     A search database over wikipedia abstracts only.
   * [full-pages](https://storage.googleapis.com/mallacc/wiki_pages.tar.xz): A
     search database over full Wikipedia pages (text only).
4. [XIOSim](https://github.com/s-kanev/XIOSim/tree/malloc_hw) - an x86
   simulator. This branch includes support for malloc instructions and
   specialized hardware.

Installation
------------


# General #

# tcmalloc #

# Xapian #

# Masstree #

# XIOSim #


Running native experiments
--------------------------

Running simulations
-------------------

