#!/bin/bash

BMK_LINE="./Xalan_base.O3gcc -v t5.xml xalanc.xsl"
BMK_LINE="/home/skanev/gperftools/output_no_magic/gauss_native"


SAMPLING_EVENTS="INSTRUCTIONS_RETIRED:period=2000003"
COUNTING_EVENTS="INSTRUCTIONS_RETIRED"

SAMPLING_LINE="perf record --pfm-events ${SAMPLING_EVENTS} -o GLOB -- ${BMK_LINE}"
COUNTING_LINE="perf stat --pfm-events ${COUNTING_EVENTS} -- ${BMK_LINE}"

export TIMEFORMAT="%3R"

for i in {1..10};
do
    time ${SAMPLING_LINE/GLOB/perf.data.$i} > /dev/null 2> /dev/null
    time ${COUNTING_LINE} > /dev/null 2> /dev/null
done
