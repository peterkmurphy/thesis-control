/* 
// projcommon.c. Implements common functionality for working with 
// vectors dealing with both OpenCL and OpenMP.
// Written by Peter Murphy. (c) 2013, 2014
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <omp.h>
#include "projcommon.h"

/* Function implementations. */

FLPT * dassign(const INTG isize)
{
    return (FLPT *)malloc(isize * sizeof(FLPT));
}

INTG * iassign(const INTG isize)
{
    return (INTG *)malloc(isize * sizeof(INTG));
}

/*
// The following routine is adapted from the following Stack Overflow
// article: http://stackoverflow.com/questions/361363/ \
// how-to-measure-time-in-milliseconds-using-ansi-c 
//
// It gives the differences in nanoseconds between two events
*/

TLEN timespecDiff(struct timespec *ptime1, struct timespec *ptime2)
{
    const int NSPERS = TLPERS; /* Nanoseconds per seconds. */
    return ((ptime1->tv_sec * NSPERS) + ptime1->tv_nsec) -
           ((ptime2->tv_sec * NSPERS) + ptime2->tv_nsec);
}

// Sets a vector fItem of floats of size iSize to 0.0. Returns it as well.

FLPT * SetFNull(INTG iSize, FLPT * fItem)
{
    return SetFValue(iSize, 0.0, fItem);
}

// Sets a vector iItem of integers of size iSize to 0. Returns it as well.

INTG * SetNull(INTG iSize, INTG * iItem)
{
    return SetValue(iSize, 0, iItem);
}

// Sets a vector fItem of floats of size iSize to fVal. Returns it as well.

FLPT * SetFValue(int iSize, FLPT fVal, FLPT * fItem)
{
    INTG i;
    for (i = 0; i < iSize; i++)
    {
        fItem[i] = fVal;
    }
    return fItem;
}

// Sets a vector fItem of floats of size iSize to fVal. Returns it as well.

INTG * SetValue(INTG iSize, INTG iVal, INTG * iItem)
{
    INTG i;
    for (i = 0; i < iSize; i++)
    {
        iItem[i] = iVal;
    }
    return iItem;
}

// Generates a vector from 1, 2, 3 to iSize. Sets it to iItem and returns it.

INTG * SetIncrease(INTG iSize, INTG * iItem)
{
    INTG i;
    for (i = 0; i < iSize; i++)
    {
        iItem[i] = (i + 1);
    }
    return iItem;
}



// Generates a vector from 1.0, 2.0, 3.0 to iSize. Sets it to fItem and returns it.

FLPT * SetFIncrease(INTG iSize, FLPT * fItem)
{
    INTG i;
    for (i = 0; i < iSize; i++)
    {
        fItem[i] = (FLPT)(i + 1);
    }
    return fItem;
}


void printvector(const char* name, INTG isize, const FLPT* dvector)
{
    printf("%s: [", name);
    INTG i;
    for (i = 0; i < (isize - 1); i++)
    {
        printf("%f, ", dvector[i]);
    }
    printf("%f]\n", dvector[isize - 1]);
}

void printintvector(const char* name, INTG isize, const INTG* ivector)
{
    printf("%s: [", name);
    INTG i;
    for (i = 0; i < (isize - 1); i++)
    {
        printf("%d, ", ivector[i]);
    }
    printf("%d]\n", ivector[isize - 1]);
}


