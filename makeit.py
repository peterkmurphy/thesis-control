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
                "-o", "mpucds" + eff, "-lrt"]);
        else:
            x = subprocess.call(["gcc", "-Wall", EFF_OP, "ucds.c", 
                "-o", "ucds" + eff, "-lrt"]);            
        



#OPTTOFILE = {"": "ucds", "-O1": "ucds1", "-O2": "ucds2", 
#    "-O3": "ucds3", "-Ofast": "ucdsf"};
#    
#for k, v in OPTTOFILE.iteritems():
#    if k == "":
#        x = subprocess.call(["gcc", "-Wall", "ucds.c", "-o", v, "-lrt"]);
#    else:
#        x = subprocess.call(["gcc", "-Wall", "ucds.c", k, "-o", v, "-lrt"]);
        

