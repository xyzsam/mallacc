# Building SPEC CPU2006 for Mallacc #

## For native runs
Build SPEC CPU2006 through the typical runspec flow. We used the following flags
in our config file (adjust for your toolchain and gperftools install location):
```bash
CC           = /group/vlsiarch/skanev/toolchain_6_1/bin/gcc
CXX          = /group/vlsiarch/skanev/toolchain_6_1/bin/g++
FC           = /group/vlsiarch/skanev/toolchain_6_1/bin/gfortran

COPTIMIZE    = -O3 -g3 -static -mfpmath=sse -fgnu89-inline
CXXOPTIMIZE  = -O3 -g3 -static -mfpmath=sse -fabi-version=2 -D_GLIBCXX_USE_CXX11_ABI=0 -fsized-deallocation
FOPTIMIZE    = -O3 -g3 -static -mfpmath=sse

LIBS = -T /home/skanev/gperftools/install/spec_linker.ld /home/skanev/gperftools/install/lib/libtcmalloc.a /home/skanev/libunwind/install/lib/libunwind.a /group/vlsiarch/skanev/toolchain_6_1/lib64/libstdc++.a -lm -pthread
```

Note the extra linker script and linking with the version of `libtcmalloc.a` without any extra Mallacc instructions.


## For simulation
Simulating the full duration of SPEC ref is very impractical. We use SimPoints to subset runs, and in particular,
the [PinPoints implementation](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.75.2435&rep=rep1&type=pdf).

Follow the instructions [here](https://software.intel.com/en-us/articles/pintool-pinpoints) to generate PinPoints for
your newly built SPEC binaries.
We used the following parameters for our runs:
```bash
export SLICE_SIZE=1000000000
export MAXK=5
export CUTOFF=".99"
```
The single point length of 1B instructions is especially important -- it makes sure that each slice contains a large enough
number of memory allocations/deallocations. After this step, you will be left with `.pp` files for each benchmark/input.

Now, build another copy of SPEC in a new output location, this time linking with `libtcmalloc.a` with
the Mallacc instructions. This should be the only difference in your SPEC config files:
```bash
# ... same as above
LIBS = -T /home/skanev/gperftools/install_all_magic/spec_linker.ld  /home/skanev/gperftools/install_all_magic/lib/libtcmalloc.a /home/skanev/libunwind/install/lib/libunwind.a /group/vlsiarch/skanev/toolchain_6_1/lib64/libstdc++.a -lm -pthread
```

Because the binaries for simulation include custom instructions that don't run natively, we can't just repeat the
same flow and get representative pinpoints for them. We will reuse the ones from the native run, with small manual
adjustments.

For each `.pp` file, do the following:

- Disassemble the corresponding SPEC benchmark binaries for both the native and simulated case.
- Find the `markedInstructions` section in the `.pp` file.
- For each instruction mark (start or end of a PinPoint):
    + look at the address for each instruction
    + make sure the instructions in the native and simulated binaries at that address match (and are also
    in the same function, surrounded by similar code).
    + for the few cases which don't match, find out the new address of that instruction in the simulated binary.
    This is usually easier than it sounds -- the only difference between the two binaries should come from allocator
    code and different linker section offsets -- so the instruction just lands at the same offset on a different page.
    Just search for the last three hex digits of the native instruction address in the simulated binary's disassembly,
    until you see the familiar pattern of surrounding code. Once you have its address in the simulated binary, update
    the appropriate line in the `.pp` file and use it for simulation.