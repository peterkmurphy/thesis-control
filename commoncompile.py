#!/usr/bin/env python
# commoncompile.py. Includes common compiler options for use by other
# Python files.
# Written by Peter Murphy. (c) 2014.

import os;
import errno;

# All these options matches gcc optimization arguments with the resulting file name.

# First, we add an option for making "FLPT" as doubles. (The default is
# float).

BIGDOUBLEOPTION = "-DBIGFLOAT"

# Then, we list the efficiency options.

EFF_OPTIONS = ["0", "1", "2", "3", "fast"];

# Then we state the main OpenMP option.

OPENMPOP = "-fopenmp";

# Then we add a condition for loop unrolling.

LOOPUNROLL = "-funroll-loops"

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise