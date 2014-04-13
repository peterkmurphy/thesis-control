/* 
// openclstuff.h. Common functionality for OpenCL.
// Written by Peter Murphy. (c) 2014
*/

#include <time.h>
#include <CL/cl.h>
#include "projcommon.h"

#ifndef OPENCLSTUFF_H
#define OPENCLSTUFF_H

// This defines a GCAQ - GPU Content And Queue - item.

typedef struct gpu_contextandqueue {
    cl_context TheContext;
    cl_command_queue TheQueue;
    cl_device_id TheDeviceId;
} GCAQ;

// This creates a pointer to a GCAQ - GPU Content And Queue - item.

GCAQ * GCAQSetup();

// This releases the resources for a GCAQ item.

void GCAQShutdown (GCAQ * TheGCAQ);

// This defines a GPAK - GPU Program And Kernel - item.

//typedef struct gpu_programandkernel {
//    cl_program TheProgram;
//    cl_kernel TheKernel;
//    size_t TheMaxWorkGroupSize;
//} GPAK;

typedef struct gpu_programandkernel {
   cl_program TheProgram;
   INTG iNoKernels;
   cl_kernel * TheKernels;
   char ** TheKernelSources;
   size_t * TheMaxWorkGroupSizes;
} GPAK; 

// This sets up a GPAK - GPU Program And Kernel - item. Arguments:
// TheGCAQ - a reference to a GPU Content and Queue Item.
// szName - the name of the method.
// szSource - the code of the method for compilation.
// szOptions - the options to pass to the compiler.

//GPAK * GPAKSetup(GCAQ * TheGCAQ, const char *szName, const char *szSource, const char *szOptions);

GPAK * GPAKSetup(GCAQ * TheGCAQ, INTG iNoKernels, char **szSources, const char *szOptions);

// This releases the resources for a GCAQ item.

void GPAKShutdown (GPAK * TheGPAK);

// Sets a vector fItem of floats of size iSize to 0.0. Returns it as well.

FLPT * SetFNull(INTG iSize, FLPT * fItem);

// Sets a vector iItem of integers of size iSize to 0. Returns it as well.

INTG * SetNull(INTG iSize, INTG * iItem);

// Sets a vector fItem of floats of size iSize to fVal. Returns it as well.

FLPT * SetFValue(INTG iSize, FLPT fVal, FLPT * fItem);

// Sets a vector fItem of floats of size iSize to fVal. Returns it as well.

INTG * SetValue(INTG iSize, INTG iVal, INTG * iItem);

// Generates a vector from 1.0, 2.0, 3.0 to iSize. Sets it to fItem and returns it.

FLPT * SetFIncrease(INTG iSize, FLPT * fItem);

/*
// The printvector function prints a vector of floats to standard output. The
// arguments:
// name: the name of the variable represented.
// isize: the size of the vector.
// dvector: the vector.
// The function returns no arguments.
*/

void printvector(const char* name, INTG isize, const FLPT* dvector);

/*
// The printintvector function prints a vector of ints to standard output. The
// arguments:
// name: the name of the variable represented.
// isize: the size of the vector.
// ivector: the vector.
// The function returns no arguments.
*/

void printintvector(const char* name, INTG isize, const INTG* ivector);


#endif // OPENCLSTUFF_H