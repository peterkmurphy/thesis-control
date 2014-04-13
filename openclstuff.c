/* 
// openclstuff.c. Implementation of Common functionality for OpenCL.
// Written by Peter Murphy. (c) 2014
*/

#include <stdio.h>
#include <time.h>
#include <CL/cl.h>
#include "projcommon.h"
#include "openclstuff.h"

// This creates a pointer to a GCAQ - GPU Content And Queue - item.

GCAQ * GCAQSetup()
{
    cl_uint uiNumPlatforms = 0;
    cl_uint uiNumDevices = 0;
    cl_platform_id ThePlatformId;
    cl_device_id TheDeviceId;
    cl_context_properties TheContextProperties[3];
    cl_int err;
    
    // Get all platforms available.
    
    if (clGetPlatformIDs(1, &ThePlatformId, &uiNumPlatforms) != CL_SUCCESS)
    {
        printf("We can't get the platform id.\n");
        return NULL;
    }
    
    // Is there a supported GPU device?
    
    if (clGetDeviceIDs (ThePlatformId, CL_DEVICE_TYPE_GPU, 1, &TheDeviceId,
        &uiNumDevices) != CL_SUCCESS)
    {
        printf("We can't get the device id.\n");
        return NULL;
    }
    
    // Then we can return a GCAQ pointer.
    
    GCAQ * TheGCAQ = (GCAQ *) malloc(1 * sizeof(GCAQ));

    // The context properties list must finish with a zero.
    
    TheContextProperties[0]= CL_CONTEXT_PLATFORM;
    TheContextProperties[1]= (cl_context_properties) ThePlatformId;
    TheContextProperties[2]= 0;
 
    // We create a context with the GPU device.
    
    TheGCAQ->TheContext = clCreateContext(TheContextProperties, 1, &TheDeviceId, NULL, NULL, &err);
 
    // We create a command queue using the context and the device.
    
    TheGCAQ->TheQueue = clCreateCommandQueue(TheGCAQ->TheContext, TheDeviceId, 0, &err);
    TheGCAQ->TheDeviceId = TheDeviceId;
    return TheGCAQ;
    
}

// This releases the resources for a GCAQ item.

void GCAQShutdown (GCAQ * TheGCAQ)
{
    clReleaseCommandQueue(TheGCAQ->TheQueue);
    clReleaseContext(TheGCAQ->TheContext);
    free(TheGCAQ);
}

// This sets up a GPAK - GPU Program And Kernel - item. Arguments:
// TheGCAQ - a reference to a GPU Content and Queue Item.
// szName - the name of the method.
// szSource - the code of the method for compilation.

//typedef struct gpu_programandkernel {
//    cl_program TheProgram;
//    cl_kernel TheKernel;
//    size_t TheMaxWorkGroupSize;
//} GPAK;





GPAK * GPAKSetup(GCAQ * TheGCAQ, INTG iNoKernels, char **szSources, const char *szOptions)
{
    cl_int err;
    cl_program TheProgram = clCreateProgramWithSource(TheGCAQ->TheContext,iNoKernels,
        (const char **)szSources, NULL, &err);
    if (clBuildProgram(TheProgram, 0, NULL, szOptions, NULL, NULL) != CL_SUCCESS)
    {
        printf("The program has failed to build.\n");
        return NULL;
    }
    GPAK * TheGPAK = (GPAK *) malloc(iNoKernels * sizeof(GPAK));
    TheGPAK->TheProgram = TheProgram;
    TheGPAK->TheKernelSources = szSources;
    TheGPAK->iNoKernels = iNoKernels;
    cl_kernel * TheKernelBuffer = (cl_kernel *) malloc (iNoKernels * sizeof(cl_kernel));
    clCreateKernelsInProgram (TheProgram, iNoKernels, TheKernelBuffer, NULL);
    TheGPAK->TheKernels = TheKernelBuffer; 
    TheGPAK->TheMaxWorkGroupSizes = (size_t *) malloc (iNoKernels * sizeof(size_t));
    size_t iMaxWorkGroupSize;
    size_t iErr;
    int iIter;
    for (iIter = 0; iIter < TheGPAK->iNoKernels; iIter++)
    {
        clGetKernelWorkGroupInfo (TheGPAK->TheKernels[iIter], TheGCAQ->TheDeviceId, CL_KERNEL_WORK_GROUP_SIZE, 
            sizeof(size_t), &iMaxWorkGroupSize, &iErr);
        TheGPAK->TheMaxWorkGroupSizes[iIter] = iMaxWorkGroupSize;
    }
    return TheGPAK;
}
 



// This releases the resources for a GCAQ item.

//void GPAKShutdown (GPAK * TheGPAK)
//{
//    clReleaseProgram(TheGPAK->TheProgram);
//    clReleaseKernel(TheGPAK->TheKernel);
//    free(TheGPAK);
//}



void GPAKShutdown (GPAK * TheGPAK)
{
    clReleaseProgram(TheGPAK->TheProgram);
    int iIter;
    for (iIter = 0; iIter < TheGPAK->iNoKernels; iIter++)
    {
        clReleaseKernel(TheGPAK->TheKernels[iIter]);
    }
    free(TheGPAK->TheKernels);
    free(TheGPAK->TheMaxWorkGroupSizes);
    free(TheGPAK);
}

    
    
    
    
    
    
    
    
    
    

