#!/usr/bin/env python
# makeit.py. Used for making different versions of runucds.c (a program
# that measures the time it takes to multiply two matrices using Ultra
# Compressed Diagonal Storage, vector norms, scalar products and other
# routine per per size of the input) with various compilation options. 
# Easier than using the 'make' executable.
# Written by Peter Murphy. (c) 2013.

import subprocess;

# This matches gcc optimization arguments with the resulting file name.
# First, we list the efficiency options.

EFF_OPTIONS = ["0", "1", "2", "3", "fast"];

# Then we state the main OpenMP option.

OPENMPOP = "-fopenmp";

# Then we add a condition for loop unrolling.

LOOPUNROLL = "-funroll-loops"

# Now we build the compile options.

for eff in EFF_OPTIONS:
    EFF_OP = "-O" + eff;
    for ismp in [True, False]:
        for isunroll in [True, False]:
            if ismp and not isunroll:
                x = subprocess.call(["gcc", "-Wall", OPENMPOP, EFF_OP, "ucds.c", 
                    "runucds.c", "-o", "mpucds" + eff, "-lrt", "-lm"]);
            elif (not ismp) and (not isunroll):
                x = subprocess.call(["gcc", "-Wall", EFF_OP, "ucds.c", 
                    "runucds.c", "-o", "ucds" + eff, "-lrt", "-lm"]);
            elif ismp and isunroll:
                x = subprocess.call(["gcc", "-Wall", OPENMPOP, EFF_OP, LOOPUNROLL, "ucds.c", 
                    "runucds.c", "-o", "mpurucds" + eff, "-lrt", "-lm"]);                
            else:    
                x = subprocess.call(["gcc", "-Wall", EFF_OP, LOOPUNROLL, "ucds.c", 
                    "runucds.c", "-o", "urucds" + eff, "-lrt", "-lm"]);           



