// diagonalmatrix.c. Written by Peter Murphy.
// Copyright (c) 2014.
// http://developer.amd.com/resources/documentation-articles/articles-whitepapers/opencl-optimization-case-study-diagonal-sparse-matrix-vector-multiplication-test/


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <CL/cl.h>
#include "openclstuff.h"
 
// This is changed depending on whether floats or doubles are possible.

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif



char *szDiagMult =
	"#if BIGFLOAT \n"\
	"#define FLPT double\n"\
	"#else\n"\
	"#define FLPT float\n"\
	"#endif\n"\
"    __kernel void dia_basic(__global FLPT *A, __const int rows, \n"\
"    __const int diags, __global int *offsets, \n"\
"    __global FLPT *x, __global FLPT *y) \n"\
"{ \n"\
"    int row = get_global_id(0); \n"\
"    FLPT accumulator = 0; \n"\
"    for(int diag = 0; diag < diags; diag++) \n"\
"    { \n"\
"        int col = row + offsets[diag]; \n"\
"        if ((col >= 0) && (col < rows)) \n"\
"        { \n"\
"            float m = A[diag*rows + row]; \n"\
"            float v = x[col]; \n"\
"            accumulator += m * v; \n"\
"        } \n"\
"    } \n"\
"    y[row] = accumulator; \n"\
"} \n";

// __const int pitch_A,
char *szDiagAligned = "#if BIGFLOAT \n"\
	"#define FLPT double\n"\
	"#else\n"\
	"#define FLPT float\n"\
	"#endif\n"\
" __kernel void dia_spmv(__global FLPT *A,\n"\
"   __const int rows, __const int diags, __global int *offsets, \n"\
"              __global FLPT *x, __global FLPT *y) { \n"\
"  int row = get_global_id(0); \n"\
"  FLPT accumulator = 0; \n"\
"  __global FLPT* matrix_offset = A + row; \n"\
"  for(int diag = 0; diag < diags; diag++) { \n"\
"    int col = row + offsets[diag]; \n"\
"    if ((col >= 0) && (col < rows)) { \n"\
"      FLPT m = *matrix_offset; \n"\
"      FLPT v = x[col]; \n"\
"      accumulator += m * v; \n"\
"    } \n"\
"    matrix_offset += rows; \n"\
"  } \n"\
"  y[row] = accumulator; \n"  \
"} \n"; 


/*
FLPT * generateData(int iSize, FLPT * fItem)
{
    int i;
    for (i = 0; i < iSize; i++)
    {
        fItem[i] = i + 1;
    }
    return fItem;
}

FLPT * SetOne(int iSize, FLPT * fItem)
{
    int i;
    for (i = 0; i < iSize; i++)
    {
        fItem[i] = 1.0;
    }
    return fItem;
}
*/
int main(int argc, char *argv[])
{
    printf("Hooya!\n");

    int iNumRows = 1;
    int iNumDiags = 1;
    int iNumReps = 1;
    int iCheck;
    if (argc > 1)
    {
        iCheck = atoi(argv[1]);
        if (iCheck != 0)
        {
            iNumRows = iCheck;
        }
    }
    if (argc > 2)
    {
        iCheck = atoi(argv[2]);
        if (iCheck != 0)
        {
            iNumDiags = iCheck;
        }
    }
    if (argc > 3)
    {
        iCheck = atoi(argv[3]);
        if (iCheck != 0)
        {
            iNumReps = iCheck;
        }
    }    
    int bPrint = 0;
    if (argc > 4)
    {
        bPrint = 1;
    }
    
    GCAQ * TheGCAQ = GCAQSetup();
    if (TheGCAQ == NULL)
    {
        return 1;
    }
    const int iNumberOfKernels = 2;
#if BIGFLOAT
	const char *szFloatOpt = "-DBIGFLOAT";
#else
	const char *szFloatOpt = NULL;
#endif
	char *ourKernelStrings[2] =
		{ szDiagAligned, szDiagMult}; //, szDiagAligned};


  	GPAK *TheGPAK = GPAKSetup(TheGCAQ, iNumberOfKernels, ourKernelStrings, szFloatOpt);
    if (TheGPAK == NULL)
    {
        GCAQShutdown(TheGCAQ);
        return 2;
    }    

    
    
    
    cl_mem inputA, inputOff, inputX, outputY;
 
    size_t global = iNumRows * iNumDiags;
 
    FLPT* inputDataA = (FLPT *) malloc(iNumRows * iNumDiags * sizeof(FLPT));
    SetFValue(iNumRows * iNumDiags, 1.0, inputDataA);
 //   SetOne(iNumRows * iNumDiags, inputDataA);
    int* inputDataOff = (int *) malloc(iNumDiags * sizeof(int));
//    SetNull(iNumDiags, inputDataOff);
    int iCount;
    for (iCount = 0; iCount < iNumDiags; iCount++)
    {
        if (iCount % 2 == 0)
        {
            inputDataOff[iCount] = iCount / 2;
  //          printf("%d - ", iCount / 2);
        }
        else
        {
            inputDataOff[iCount] = -(iCount / 2) - 1;
     /////       printf("%d - ", -(iCount / 2) - 1);
        }
  //      printf("%d\n", iCount);
    }
      
    FLPT* inputDataX =  (FLPT *) malloc(iNumRows * sizeof(FLPT));
    SetFValue(iNumRows, 1.0, inputDataX);
//    SetOne(iNumRows, inputDataX); 
    FLPT* outputDataY = (FLPT *) malloc(iNumRows * sizeof(FLPT));
    SetFNull(iNumRows, outputDataY);
    int i;

	struct timespec start[iNumberOfKernels];
	struct timespec end[iNumberOfKernels];
    
// create buffers for the input and ouput

    int err; 
    inputA = clCreateBuffer(TheGCAQ->TheContext, CL_MEM_READ_ONLY, sizeof(FLPT) * iNumRows * iNumDiags, NULL, &err);
    if (err != CL_SUCCESS)
    {
        printf("Error allocating for A");
        return 3;
    }
    
    inputOff = clCreateBuffer(TheGCAQ->TheContext, CL_MEM_READ_ONLY, sizeof(int) * iNumDiags, NULL, &err);
    if (err != CL_SUCCESS)
    {
        printf("Error allocating for Offsets");
        return 4;
    }

    inputX = clCreateBuffer(TheGCAQ->TheContext, CL_MEM_READ_ONLY, sizeof(FLPT) * iNumRows, NULL, &err);
    if (err != CL_SUCCESS)
    {
        printf("Error allocating for X");
        return 5;
    }
    
    outputY = clCreateBuffer(TheGCAQ->TheContext, CL_MEM_WRITE_ONLY, sizeof(FLPT) * iNumRows, NULL, &err);
    if (err != CL_SUCCESS)
    {
        printf("Error allocating for output");
        return 6;
    }

    int rep;
	int iKernel;

	for (iKernel = 0; iKernel < iNumberOfKernels; iKernel++)
	{
		// printf("%ld\n", TheGPAK->TheMaxWorkGroupSizes[iKernel]);

		for (i = 0; i < iNumRows; i++)
		{
			outputDataY[i] = 0.0;
		}
        clock_gettime(CLOCK_MONOTONIC, &(start[iKernel]));
		for (rep = 0; rep < iNumReps; rep++)
		{
// load data into the input buffer
    clEnqueueWriteBuffer(TheGCAQ->TheQueue, inputA, CL_TRUE, 0, sizeof(FLPT) * iNumRows * iNumDiags, inputDataA, 0, NULL, NULL);
    clEnqueueWriteBuffer(TheGCAQ->TheQueue, inputOff, CL_TRUE, 0, sizeof(int) * iNumDiags, inputDataOff, 0, NULL, NULL);
    clEnqueueWriteBuffer(TheGCAQ->TheQueue, inputX, CL_TRUE, 0, sizeof(FLPT) * iNumRows, inputDataX, 0, NULL, NULL);
// set the argument list for the kernel command

    clSetKernelArg(TheGPAK->TheKernels[iKernel], 0, sizeof(cl_mem), &inputA);
    clSetKernelArg(TheGPAK->TheKernels[iKernel], 1, sizeof(int), &iNumRows);
    clSetKernelArg(TheGPAK->TheKernels[iKernel], 2, sizeof(int), &iNumDiags);
    clSetKernelArg(TheGPAK->TheKernels[iKernel], 3, sizeof(cl_mem), &inputOff);
    clSetKernelArg(TheGPAK->TheKernels[iKernel], 4, sizeof(cl_mem), &inputX);
    clSetKernelArg(TheGPAK->TheKernels[iKernel], 5, sizeof(cl_mem), &outputY);
  
// enqueue the kernel command for execution

    clEnqueueNDRangeKernel(TheGCAQ->TheQueue, TheGPAK->TheKernels[iKernel], 1, NULL, &global, NULL, 0, NULL, NULL);
    clFinish(TheGCAQ->TheQueue);

// copy the results from out of the output buffer

    clEnqueueReadBuffer(TheGCAQ->TheQueue, outputY, CL_TRUE, 0, sizeof(FLPT) * iNumRows, outputDataY, 0, NULL, NULL);		
       
 ////   if (iKernel == 1) {
  //      printf("ADSADS\n");
  //          return(1);     
  //  }
}
		clock_gettime(CLOCK_MONOTONIC, &(end[iKernel]));

		// printing the results.

    		if (bPrint)
		{

            printf("output: ");
            for(i=0;i<iNumRows; i++)
            {
                printf("%f ",outputDataY[i]);
            }
            printf("\n");
		}
    
    
	}

   
// print the results

    printf("%d - %d - %d - ", iNumRows, iNumDiags, iNumReps);
	for (iKernel = 0; iKernel < iNumberOfKernels; iKernel++)
	{
		printf("%f - ", (1.0 * TLPERS * iNumRows * iNumDiags * iNumReps) /
			   (MEGAHERTZ * timespecDiff(&(end[iKernel]), &(start[iKernel]))));
	}
	printf("\n");


    
// cleanup - release OpenCL resources

    free(inputDataX);
    free(inputDataA);
    free(inputDataOff);
    free(outputDataY);
    clReleaseMemObject(outputY);
    clReleaseMemObject(inputOff);
    clReleaseMemObject(inputX);
    clReleaseMemObject(inputA);
    GPAKShutdown(TheGPAK);
    GCAQShutdown (TheGCAQ);
    return 0;
}