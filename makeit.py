# makeit.py. Used for making different versions of ucds with
# various compilation options. Easier than make.
# // Written by Peter Murphy. (c) 2013. 

import subprocess;

# This matches gcc optimization arguments with the resulting file name.
# First, we list the efficiency options.

EFF_OPTIONS = ["0", "1", "2", "3", "fast"];

# Then we state the main OpenMP option.

OPENMPOP = "-fopenmp";

# Now we build the compile options.

for eff in EFF_OPTIONS:
    EFF_OP = "-O" + eff;
    for ismp in [True, False]:
        if ismp:
            x = subprocess.call(["gcc", "-Wall", OPENMPOP, EFF_OP, "ucds.c", 
                "testucds.c", "-o", "mpucds" + eff, "-lrt", "-lm"]);
        else:
            x = subprocess.call(["gcc", "-Wall", EFF_OP, "ucds.c", 
                "testucds.c", "-o", "ucds" + eff, "-lrt", "-lm"]);            

