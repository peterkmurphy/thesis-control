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
    MINMATSIZE =  256;
    MAXMATSIZE = 4194304 #33554432;
    NOITERS = "100";

# Now we try out the executables.

for k in ["3"]: #EFF_OPTIONS:
    for j in ["mpur"]: #OPENMP_OPTIONS:
        ourFile = "./" + TDIAGMATRIXCLCREATE + j + "diagmatrixcl" + k; 
        i = MINMATSIZE; # The minimum iteration amount
        print ourFile;
        while i <= MAXMATSIZE:
           # subprocess.call([ourFile, str(i), "1", NOITERS]);
           # subprocess.call([ourFile, str(i), "3", NOITERS, "true"]);
            subprocess.call([ourFile, str(i), "5", NOITERS, "true"]);            
           # subprocess.call([ourFile, str(i), "9", NOITERS, "true"]);            
           # subprocess.call([ourFile, str(i), "27", NOITERS, "true"]);
          #//  subprocess.call([ourFile, str(i), "11", NOITERS]);
          #  subprocess.call([ourFile, str(i), "13", NOITERS]);
            i *= 2;

