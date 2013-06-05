#!/usr/bin/env python
# runtestit.py. Used for running different versions of testucds.c (a program
# that tests Ultra Compressed Diagonal Storage and Conjugate Gradient and other
# routines for correctness) with various compilation options.
# Written by Peter Murphy. (c) 2013. 

import sys
import subprocess;

# This matches gcc optimization arguments with the resulting file name.
# First, we list the efficiency suffixes

EFF_OPTIONS = ["0", "1", "2", "3", "fast"];

# Then we state OpenMP prefixes.

OPENMP_OPTIONS = ["", "mp", "ur", "mpur"];

# These set the ranges to try out. 

if len(sys.argv) >= 4:
    MINMATSIZE = int(sys.argv[1]);
    MAXMATSIZE = int(sys.argv[2]);
    NOITERS = str(int(sys.argv[3]));
else:
    MINMATSIZE = 2;
    MAXMATSIZE = 128;
    NOITERS = "1";

# Now we try out the executables.

for k in EFF_OPTIONS:
    for j in OPENMP_OPTIONS:
        ourFile = "./" + j + "tucds" + k; 
        i = MINMATSIZE; # The minimum iteration amount
        print ourFile;
        while i <= MAXMATSIZE:
            subprocess.call([ourFile, str(i), NOITERS]);
            i = i * 2;

