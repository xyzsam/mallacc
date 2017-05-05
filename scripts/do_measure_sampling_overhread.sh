#!/bin/bash

taskset -c 0 ./measure_sampling_overhead.sh &> time.out

SAMPLING="awk 'NR % 2 {print;}' time.out"
COUNTING="awk 'NR % 2 == 0 {print;}' time.out"

echo "Sampling time:"
SAMPLING_TIME=`eval ${SAMPLING} | ./mean.py`
echo ${SAMPLING_TIME}
eval ${SAMPLING} | ./std.py

echo "Counting time:"
COUNTING_TIME=`eval ${COUNTING} | ./mean.py`
echo ${COUNTING_TIME}
eval ${COUNTING} | ./std.py

echo "Delta:"
DELTA=`echo ${SAMPLING_TIME}-${COUNTING_TIME} | bc -l`
echo ${DELTA}

echo "Num interrupts:"
NUM_INTERRUPTS=`eval 'for i in {1..10};
do
    perf script -i "perf.data.$i" | wc -l
done' | ./mean.py`
echo ${NUM_INTERRUPTS}

echo "Delta per interrupt (us):"
echo "1000.0 * 1000.0 * ${DELTA}/${NUM_INTERRUPTS}" | bc -l
