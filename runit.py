#!/usr/bin/env python
# makeit.py. Used for running different versions of runucds.c (a program
# that measures the time it takes to multiply two matrices using Ultra
# Compressed Diagonal Storage, vector norms, scalar products and other
# routine per per size of the input) with various compilation options. 
# Written by Peter Murphy. (c) 2013. 

import sys;
import subprocess;
from commoncompile import *

# These set the ranges to try out. 

if len(sys.argv) >= 3:
    MINMATSIZE = int(sys.argv[1]);
    MAXMATSIZE = int(sys.argv[2]);
    NOITERS = str(int(sys.argv[3]));
else:
    MINMATSIZE = 16384;
    MAXMATSIZE = 4194304 #16384 2097152;
    NOITERS = "3";

# Now we try out the executables.

for k in ["0", "3"]: #EFF_OPTIONS:
    for j in ["", "mp", "ur", "mpur"]: #OPENMP_OPTIONS:
        ourFile = "./" + TIMEDIRCREATE + j + "ucds" + k; 
        i = MINMATSIZE; # The minimum iteration amount
        print ourFile;
        while i <= MAXMATSIZE:
            subprocess.call([ourFile, str(i), NOITERS]);
            i *= 2;

