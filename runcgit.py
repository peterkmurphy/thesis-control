#!/usr/bin/env python
# runcgit.py. Used for running different versions of runconjgrad.c (a program
# that measures the time it takes to execute conjugate gradient) with various
# compilation options. 
# Written by Peter Murphy. (c) 2013.

import sys;
import subprocess;

# This matches gcc optimization arguments with the resulting file name.
# First, we list the efficiency suffixes

EFF_OPTIONS = ["0", "1", "2", "3", "fast"];

# Then we state OpenMP prefixes.

OPENMP_OPTIONS = ["", "mp", "ur", "mpur"];

# These set the ranges to try out. 

if len(sys.argv) >= 3:
    MINMATSIZE = int(sys.argv[1]);
    MAXMATSIZE = int(sys.argv[2]);
    NOITERS = str(int(sys.argv[3]));
else:
    MINMATSIZE = 16384;
    MAXMATSIZE = 2097152;
    NOITERS = "100";

# Now we try out the executables.

for k in EFF_OPTIONS:
    for j in OPENMP_OPTIONS:
        ourFile = "./" + j + "ucdscg" + k; 
        i = MINMATSIZE; # The minimum iteration amount
        print ourFile;
        while i <= MAXMATSIZE:
            subprocess.call([ourFile, str(i), NOITERS]);
            i *= 2;

