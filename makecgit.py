#!/usr/bin/env python
# makecgit.py. Used for making different versions of runconjgrad.c (a program
# that measures the time it takes to execute conjugate gradient) with various
# compilation options. Easier than using the 'make' executable.
# Written by Peter Murphy. (c) 2013.

import subprocess;
import os;
import errno;

# This matches gcc optimization arguments with the resulting file name.
# First, we list the efficiency options.

EFF_OPTIONS = ["0", "1", "2", "3", "fast"];

# Then we state the main OpenMP option.

OPENMPOP = "-fopenmp";

# Then we add a condition for loop unrolling.

LOOPUNROLL = "-funroll-loops"

# Now we add a subdirectory for executables to be created in.

DIRCREATE = "timecg/"

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

make_sure_path_exists(DIRCREATE);

# Now we build the compile options.

for eff in EFF_OPTIONS:
    EFF_OP = "-O" + eff;
    for ismp in [True, False]:
        for isunroll in [True, False]:
            if ismp and not isunroll:
                x = subprocess.call(["gcc", "-Wall", OPENMPOP, EFF_OP, "ucds.c", 
                    "runconjgrad.c", "-o", DIRCREATE + "mpucdscg" + eff, "-lrt", "-lm"]);
            elif (not ismp) and (not isunroll):
                x = subprocess.call(["gcc", "-Wall", EFF_OP, "ucds.c", 
                    "runconjgrad.c", "-o", DIRCREATE + "ucdscg" + eff, "-lrt", "-lm"]);
            elif ismp and isunroll:
                x = subprocess.call(["gcc", "-Wall", OPENMPOP, EFF_OP, LOOPUNROLL, "ucds.c", 
                    "runconjgrad.c", "-o", DIRCREATE + "mpurucdscg" + eff, "-lrt", "-lm"]);                
            else:    
                x = subprocess.call(["gcc", "-Wall", EFF_OP, LOOPUNROLL, "ucds.c", 
                    "runconjgrad.c", "-o", DIRCREATE + "urucdscg" + eff, "-lrt", "-lm"]);  