#!/usr/bin/env python
# rundiagmatrixcl.py. Used for running different versions of diagmatrixcl.c 
# (a program that measures the time it takes to perform diagonal matrix
# multiplication using OpenCL with different sizes of input) with various
# compilation options. 
# Written by Peter Murphy. (c) 2013-4.

import sys;
import subprocess;
from commoncompile import *

# These set the ranges to try out. 

if len(sys.argv) >= 3:
    MINMATSIZE = int(sys.argv[1]);
    MAXMATSIZE = int(sys.argv[2]);
    NOITERS = str(int(sys.argv[3]));
else:
    MINMATSIZE =  10 #16384;
    MAXMATSIZE = 10 #33554432;
    NOITERS = "10";

# Now we try out the executables.

for k in EFF_OPTIONS:
    for j in OPENMP_OPTIONS:
        ourFile = "./" + TDIAGMATRIXCLCREATE + j + "diagmatrixcl" + k; 
        i = MINMATSIZE; # The minimum iteration amount
        print ourFile;
        while i <= MAXMATSIZE:
            subprocess.call([ourFile, str(i), "1", NOITERS, "1"]);
            subprocess.call([ourFile, str(i), "3", NOITERS, "1"]);
            subprocess.call([ourFile, str(i), "5", NOITERS, "1"]);            
            subprocess.call([ourFile, str(i), "7", NOITERS, "1"]);            
            subprocess.call([ourFile, str(i), "9", NOITERS, "1"]);
            subprocess.call([ourFile, str(i), "11", NOITERS, "1"]);
            subprocess.call([ourFile, str(i), "13", NOITERS, "1"]);
            i *= 2;

