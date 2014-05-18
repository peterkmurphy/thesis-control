/* 
// projcommon.h. Contains common functionality for working with vectors.
// dealing with both OpenCL and OpenMP.
// Written by Peter Murphy. (c) 2013, 2014
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#ifndef PROJCOMMON_H
#define PROJCOMMON_H

/* Quick and dirty min and max. */

#define max(x, y) (((x) > (y)) ? (x) : (y))
#define min(x, y) (((x) < (y)) ? (x) : (y))

/* 
// We use typedefs for integers and floating point numbers so their size
// and base types can be changed in one place.
*/

typedef int INTG;

// The compiler has to have the BIGFLOAT symbol to have FLPT as double.
// Otherwise, it will be interpreted as float.

#ifdef BIGFLOAT
    typedef double FLPT;
    #define FLPTSTR "double"
#else
    typedef float FLPT;
    #define FLPTSTR "float"
#endif

/* We use a typedef for storing time lengths. */

typedef uint64_t TLEN;

/* And timelengths per second. On Linux, it is 10^9 for nanoseconds. */

#define TLPERS 1000000000

/* But values will be displayed in Megahertz, so we need to divide. */

#define MEGAHERTZ 1000000.0

/*
// The timespecDiff routine gives the differences in nanoseconds between
// two events ptime1 (the end) and ptime2 (the start).
*/

TLEN timespecDiff(struct timespec *ptime1, struct timespec *ptime2);

/* 
// The dassign function assigns memory for a FLPT vector. The argument:
// - isize: the number of elements in the vector.
// The return value is the vector.
// Note: the space for the vector should be deallocated after use using
// the free function or similar. 
*/

FLPT * dassign(const INTG isize);

/* 
// The iassign function assigns memory for a INTG vector. The argument:
// - isize: the number of elements in the vector.
// The return value is the vector.
// Note: the space for the vector should be deallocated after use using
// the free function or similar. 
*/

INTG * iassign(const INTG isize);

// Sets a vector fItem of floats of size iSize to 0.0. Returns it as well.

FLPT * SetFNull(INTG iSize, FLPT * fItem);

// Sets a vector iItem of integers of size iSize to 0. Returns it as well.

INTG * SetNull(INTG iSize, INTG * iItem);

// Sets a vector fItem of floats of size iSize to fVal. Returns it as well.

FLPT * SetFValue(INTG iSize, FLPT fVal, FLPT * fItem);

// Sets a vector fItem of floats of size iSize to fVal. Returns it as well.

INTG * SetValue(INTG iSize, INTG iVal, INTG * iItem);

// Generates a vector from 1, 2, 3,... to iSize. Sets it to iItem and returns it.

INTG * SetIncrease(INTG iSize, INTG * iItem);

// Generates a vector from 1.0, 2.0, 3.0, ... to iSize. Sets it to fItem and returns it.

FLPT * SetFIncrease(INTG iSize, FLPT * fItem);

// Functions for printing vectors.
// This prints a vector of floats.

void printvector(const char* name, INTG isize, const FLPT* dvector);

// This prints a vector of integers.

void printintvector(const char* name, INTG isize, const INTG* ivector);

// This gets the sum of numbers from 1 to n. 

FLPT sumofnumbers(INTG nvalue);

// This gets the sum of squares of numbers from 1 to n.

FLPT sumofnumberssq(INTG nvalue);

// This gets the sum of numbers from m to n inclusive (as a float).

FLPT sumofnumbersmton(INTG mvalue, INTG nvalue);

// This gets the sum of squares of numbers from m to n inclusive.

FLPT sumofnumberssqmton(INTG mvalue, INTG nvalue);

#endif // PROJCOMMON_H 
