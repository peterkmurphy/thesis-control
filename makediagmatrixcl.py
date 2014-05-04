#!/usr/bin/env python
# makediagmatrixcl.py. Used for making different versions of diagmatrixcl.c 
# (a program that measures the time it takes to perform diagonal matrix
# multiplication using OpenCL with different sizes of input) with various
# compilation options. Easier than using the 'make' executable.
# Written by Peter Murphy. (c) 2013-4.

import subprocess;
from commoncompile import *

make_sure_path_exists(TDIAGMATRIXCLCREATE);

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
                ourexecute += "diagmatrixcl";
                ourseq.extend(["diagmatrixcl.c", "openclstuff.c", "projcommon.c", "-o"]);
                ourseq.append(TDIAGMATRIXCLCREATE + ourexecute + eff);
                ourseq.extend(OPENCLLIB);
                x = subprocess.call(ourseq);
                
       



