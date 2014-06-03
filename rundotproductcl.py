#!/usr/bin/env python
# rundotproductcl.py. Used for making different versions of dotproductcl.c (a 
# program that measures the time it takes to perform dot product and other
# reduction operations using OpenCL with different sizes of input) 
# with various compilation options. 
# Easier than using the 'make' executable.
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
    MINMATSIZE =  256;
    MAXMATSIZE = 2097152 #33554432;
    NOITERS = "20";

# Now we try out the executables.

for k in ["3"]: #EFF_OPTIONS:
    for j in ["mpur"]: #OPENMP_OPTIONS:
        ourFile = "./" + TIMEDOTPRODUCTCLCREATE + j + "dotproductcl" + k; 
        i = MINMATSIZE; # The minimum iteration amount
        print ourFile;
        while i <= MAXMATSIZE:
           # subprocess.call([ourFile, str(i - 1), NOITERS]);
            subprocess.call([ourFile, str(i), NOITERS]);
           # subprocess.call([ourFile, str(i + 1), NOITERS]);
            i *= 2;

