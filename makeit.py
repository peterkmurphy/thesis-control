#!/usr/bin/env python
# makeit.py. Used for making different versions of runucds.c (a program
# that measures the time it takes to multiply two matrices using Ultra
# Compressed Diagonal Storage, vector norms, scalar products and other
# routine per per size of the input) with various compilation options. 
# Easier than using the 'make' executable.
# Written by Peter Murphy. (c) 2013.

import subprocess;
from commoncompile import *

make_sure_path_exists(TIMEDIRCREATE);

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
                ourexecute += "ucds";
                ourseq.extend(["ucds.c", "projcommon.c", "runucds.c", "-o"]);
                ourseq.extend([TIMEDIRCREATE + ourexecute + eff, "-lrt", "-lm"]);
                x = subprocess.call(ourseq);
                
       



