# runit.py. Used for running different versions of ucds with
# various compilation options. Easier than make.
# // Written by Peter Murphy. (c) 2013. 

import subprocess;

# This matches gcc optimization arguments with the resulting file name.
# First, we list the efficiency suffixes

EFF_OPTIONS = ["0", "1", "2", "3", "fast"];

# Then we state OpenMP prefixes.

OPENMP_OPTIONS = ["", "mp"];

# These give ranges to try out.

MINMATSIZE = 1024;
MAXMATSIZE = 8193 #1000000;

# Now we try out the executables.

for k in EFF_OPTIONS:
    for j in OPENMP_OPTIONS:
        ourFile = "./" + j + "ucdscg" + k; 
        i = MINMATSIZE; # The minimum iteration amount
        print ourFile;
        while i < MAXMATSIZE:
            subprocess.call([ourFile, str(i), "1"]);
            i *= 2;

