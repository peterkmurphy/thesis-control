#!/usr/bin/env python
# makesaxpy.py. Used for making different versions of saxpy.c (a program
# that measures the time it takes to perform saxpy using OpenCL
# with different sizes of input) with various compilation options. 
# Easier than using the 'make' executable.
# Written by Peter Murphy. (c) 2013-4.

import subprocess;
from commoncompile import *

make_sure_path_exists(TIMESAXPYCLCREATE);

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
                ourexecute += "saxpycl";
                ourseq.extend(["saxpy.c", "openclstuff.c", "projcommon.c", "-o"]);
                ourseq.append(TIMESAXPYCLCREATE + ourexecute + eff);
                ourseq.extend(OPENCLLIB);
                x = subprocess.call(ourseq);
                
       



