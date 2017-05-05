#!/usr/bin/python
import sys
import numpy as np

nums = []
for line in sys.stdin:
    nums.append(float(line))

print np.average(nums)
