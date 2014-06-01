#!/usr/bin/env python
# makesaxpycl.py. Used for running different versions of saxpy.c (a program
# that measures the time it takes to perform saxpy using OpenCL
# with different sizes of input) with various compilation options. 
# Written by Peter Murphy. (c) 2013, 2014. 

import sys;
import subprocess;
from commoncompile import *

# These set the ranges to try out. 

if len(sys.argv) >= 3:
    MINMATSIZE = int(sys.argv[1]);
    MAXMATSIZE = int(sys.argv[2]);
    NOITERS = str(int(sys.argv[3]));
else:
    MINMATSIZE = 8192 #16384;
    MAXMATSIZE = 8192 #2097152 # Because 4194304 gives problems, let alone 33554432;
    NOITERS = "20";

# Now we try out the executables.

for k in ["3"]: #EFF_OPTIONS:
    for j in ["mpur"]: #OPENMP_OPTIONS:
        ourFile = "./" + TIMESAXPYCLCREATE + j + "saxpycl" + k; 
        i = MINMATSIZE; # The minimum iteration amount
        print ourFile;
        while i <= MAXMATSIZE:
           # subprocess.call([ourFile, str(i - 1), NOITERS]);
            subprocess.call([ourFile, str(i), NOITERS, "22", "True", "o"]);
           # subprocess.call([ourFile, str(i + 1), NOITERS]);
            i *= 2;

