#!/usr/bin/env python
# maketestit.py. Used for making different versions of testucds.c (a program
# that tests Ultra Compressed Diagonal Storage, Conjugate Gradient and other
# routines for correctness) with various compilation options. Easier than 
# using the 'make' executable.
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

DIRCREATE = "icctest/"

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

make_sure_path_exists(DIRCREATE);

# Now we build the compile options.
q = subprocess.call(["/bin/sh", "/opt/intel/composer_xe_2013.4.183/bin/compilervars.sh", "intel64"]);
for eff in EFF_OPTIONS:
    EFF_OP = "-O" + eff;
    for ismp in [False, True]:
        for isunroll in [False, True]:
            if ismp and not isunroll:
                x = subprocess.call(["/opt/intel/bin/icc", "-Wall", OPENMPOP, EFF_OP, "ucds.c", 
                    "testucds.c", "-o", DIRCREATE + "mpitucds" + eff, "-lrt", "-lm"]);
            elif (not ismp) and (not isunroll):
                x = subprocess.call(["/opt/intel/bin/icc", "-Wall", EFF_OP, "ucds.c", 
                    "testucds.c", "-o", DIRCREATE + "itucds" + eff, "-lrt", "-lm"]);
            elif ismp and isunroll:
                x = subprocess.call(["/opt/intel/bin/icc", "-Wall", OPENMPOP, EFF_OP, LOOPUNROLL, "ucds.c", 
                    "testucds.c", "-o", DIRCREATE + "mpuritucds" + eff, "-lrt", "-lm"]);                
            else:    
                x = subprocess.call(["/opt/intel/bin/icc", "-Wall", EFF_OP, LOOPUNROLL, "ucds.c", 
                    "testucds.c", "-o", DIRCREATE + "uritucds" + eff, "-lrt", "-lm"]);                

                
                

