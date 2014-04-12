#!/usr/bin/env python
# maketestit.py. Used for making different versions of testucds.c (a program
# that tests Ultra Compressed Diagonal Storage, Conjugate Gradient and other
# routines for correctness) with various compilation options. Easier than 
# using the 'make' executable.
# Written by Peter Murphy. (c) 2013. 2014. 

import subprocess;
from commoncompile import *

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

