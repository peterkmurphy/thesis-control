#!/usr/bin/env python
# makecgit.py. Used for making different versions of runconjgrad.c (a program
# that measures the time it takes to execute conjugate gradient) with various
# compilation options. Easier than using the 'make' executable.
# Written by Peter Murphy. (c) 2013.

import subprocess;
from commoncompile import *

# Now we add a subdirectory for executables to be created in.

DIRCGCREATE = "timecg/"

make_sure_path_exists(DIRCGCREATE);

# Now we build the compile options.

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
                ourexecute += "ucdscg";
                ourseq.extend(["ucds.c", "projcommon.c", "runconjgrad.c", "-o"]);
                ourseq.extend([DIRCGCREATE + ourexecute + eff, "-lrt", "-lm"]);
                x = subprocess.call(ourseq);






#for eff in EFF_OPTIONS:
#    EFF_OP = "-O" + eff;
##    for ismp in [True, False]:
  #      for isunroll in [True, False]:
   #         if ismp and not isunroll:
    #            x = subprocess.call(["gcc", "-Wall", "-Wno-unknown-pragmas", OPENMPOP, EFF_OP, "ucds.c", "projcommon.c",
     #               "runconjgrad.c", "-o", DIRCREATE + "mpucdscg" + eff, "-lrt", "-lm"]);
      #      elif (not ismp) and (not isunroll):
       #         x = subprocess.call(["gcc", "-Wall", "-Wno-unknown-pragmas", EFF_OP, "ucds.c", "projcommon.c",
        #            "runconjgrad.c", "-o", DIRCREATE + "ucdscg" + eff, "-lrt", "-lm"]);
        #    elif ismp and isunroll:
         #       x = subprocess.call(["gcc", "-Wall", "-Wno-unknown-pragmas", OPENMPOP, EFF_OP, LOOPUNROLL, "ucds.c", "projcommon.c",
          #          "runconjgrad.c", "-o", DIRCREATE + "mpurucdscg" + eff, "-lrt", "-lm"]);                
           # else:    
            #    x = subprocess.call(["gcc", "-Wall", "-Wno-unknown-pragmas", EFF_OP, LOOPUNROLL, "ucds.c", "projcommon.c",
             #       "runconjgrad.c", "-o", DIRCREATE + "urucdscg" + eff, "-lrt", "-lm"]);  
             