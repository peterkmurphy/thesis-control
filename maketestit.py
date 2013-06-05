#!/usr/bin/env python
# maketestit.py. Used for making different versions of testucds.c (a program
# that tests Ultra Compressed Diagonal Storage, Conjugate Gradient and other
# routines for correctness) with various compilation options. Easier than 
# using the 'make' executable.
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
                    "testucds.c", "-o", "mptucds" + eff, "-lrt", "-lm"]);
            elif (not ismp) and (not isunroll):
                x = subprocess.call(["gcc", "-Wall", EFF_OP, "ucds.c", 
                    "testucds.c", "-o", "tucds" + eff, "-lrt", "-lm"]);
            elif ismp and isunroll:
                x = subprocess.call(["gcc", "-Wall", OPENMPOP, EFF_OP, LOOPUNROLL, "ucds.c", 
                    "testucds.c", "-o", "mpurtucds" + eff, "-lrt", "-lm"]);                
            else:    
                x = subprocess.call(["gcc", "-Wall", EFF_OP, LOOPUNROLL, "ucds.c", 
                    "testucds.c", "-o", "urtucds" + eff, "-lrt", "-lm"]);                

                
                

