#!/usr/bin/env python
# runtestit.py. Used for running different versions of testucds.c (a program
# that tests Ultra Compressed Diagonal Storage and Conjugate Gradient and other
# routines for correctness) with various compilation options.
# Written by Peter Murphy. (c) 2013. 

import sys
import subprocess;
from commoncompile import *

# These set the ranges to try out. 

if len(sys.argv) >= 4:
    MINMATSIZE = int(sys.argv[1]);
    MAXMATSIZE = int(sys.argv[2]);
    NOITERS = str(int(sys.argv[3]));
else:
    MINMATSIZE = 64;
    MAXMATSIZE = 65536;
    NOITERS = "1";

# Now we try out the executables.

for k in EFF_OPTIONS:
    for j in ["d", "dmp", "dur", "dmpur"]: #OPENMP_OPTIONS:
        ourFile = "./" + TESTDIRCREATE + j + "tucds" + k; 
        i = MINMATSIZE; # The minimum iteration amount
        print ourFile;
        while i <= MAXMATSIZE:
            subprocess.call([ourFile, str(i), NOITERS]);
            i = i * 2;

