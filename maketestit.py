#!/usr/bin/env python
# maketestit.py. Used for making different versions of testucds.c (a program
# that tests Ultra Compressed Diagonal Storage, Conjugate Gradient and other
# routines for correctness) with various compilation options. Easier than 
# using the 'make' executable.
# Written by Peter Murphy. (c) 2013. 2014. 

import subprocess;
from commoncompile import *

# Now we add a subdirectory for executables to be created in.

TESTDIRCREATE = "test/"

make_sure_path_exists(TESTDIRCREATE);

# Now we build the compile options.

for eff in EFF_OPTIONS:
    EFF_OP = "-O" + eff;
    for ismp in [True, False]:
        for isunroll in [True, False]:
            for bigfloatem in [True, False]:
                ourseq = ["gcc", "-Wall", "-Wno-unknown-pragmas"];
                ourexecute = "";
                if bigfloatem:
                    ourseq.append(BIGDOUBLEOPTION);
                    ourexecute += "d";
                if ismp:
                    ourseq.append(OPENMPOP);
                    ourexecute += "mp";
                ourseq.append(EFF_OP);
                if isunroll:
                    ourseq.append(LOOPUNROLL);
                    ourexecute += "ur";
                ourexecute += "tucds";
                ourseq.extend(["ucds.c", "projcommon.c", "testucds.c", "-o"]);
                ourseq.extend([TESTDIRCREATE + ourexecute + eff, "-lrt", "-lm"]);
                x = subprocess.call(ourseq);




#for eff in EFF_OPTIONS:
#    EFF_OP = "-O" + eff;
#    for ismp in [True, False]:
#        for isunroll in [True, False]:
#            if ismp and not isunroll:
#                x = subprocess.call(["gcc", "-Wall", "-Wno-unknown-pragmas", OPENMPOP, EFF_OP, "ucds.c", "projcommon.c",
#                    "testucds.c", "-o", DIRCREATE + "mptucds" + eff, "-lrt", "-lm"]);
#            elif (not ismp) and (not isunroll):
#                x = subprocess.call(["gcc", "-Wall", "-Wno-unknown-pragmas", EFF_OP, "ucds.c", "projcommon.c",
#                    "testucds.c", "-o", DIRCREATE + "tucds" + eff, "-lrt", "-lm"]);
#            elif ismp and isunroll:
#                x = subprocess.call(["gcc", "-Wall", "-Wno-unknown-pragmas", OPENMPOP, EFF_OP, LOOPUNROLL, "ucds.c", "projcommon.c",
#                    "testucds.c", "-o", DIRCREATE + "mpurtucds" + eff, "-lrt", "-lm"]);                
#            else:    
#                x = subprocess.call(["gcc", "-Wall", "-Wno-unknown-pragmas", EFF_OP, LOOPUNROLL, "ucds.c", "projcommon.c",
#                    "testucds.c", "-o", DIRCREATE + "urtucds" + eff, "-lrt", "-lm"]);                

                
                

