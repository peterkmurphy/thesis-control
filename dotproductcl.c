// dotproduct.c. Written by Peter Murphy.
// Copyright (c) 2014.

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <CL/cl.h>
#include "openclstuff.h"
 
// This is changed depending on whether floats or doubles are possible.
// Based on ideas from here.
// http://developer.amd.com/resources/documentation-articles/articles-whitepapers/opencl-optimization-case-study-diagonal-sparse-matrix-vector-multiplication-test/
// Based on http://www.cmsoft.com.br/index.php?option=com_content&view=category&layout=blog&id=144&Itemid=203


#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif


char *szExamine =
	"#if BIGFLOAT \n"
	"#define FLPT double\n"
	"#else\n"
	"#define FLPT float\n"
	"#endif\n"
"__kernel \n"\
"void examine( \n"\
"            __const int length, \n"\
"            __global FLPT* fin, \n"\
"            __global FLPT* fout, \n"\
"            __local FLPT* ftemp, \n"\
"            __global int* globalids, \n"\
"            __global int* globalsizes, \n"\
"            __global int* localids, \n"\
"            __global int* localsizes, \n"\
"            __global int* groupids, \n"\
"            __global int* numgroups, \n"\
"            __global FLPT* freduce){ \n"\
" \n"\
"  int global_index = get_global_id(0); \n"\
"  int local_index = get_local_id(0); \n"\
"  int global_size = get_global_size(0); \n"\
"  int local_size = get_local_size(0); \n"\
"  int group_index = get_group_id(0); \n"\
"  int num_groups = get_num_groups(0); \n"\
"  if (global_index < length) { \n"\
"  freduce[global_index] = 0.0; \n"\
"  fout[global_index]=fin[global_index] * fin[global_index];  \n"\
"  ftemp[local_index] = fout[global_index];\n"\
"  barrier(CLK_LOCAL_MEM_FENCE); \n"\
"  globalids[global_index]=global_index;  \n"\
"  globalsizes[global_index]=global_size;  \n"\
"  localids[global_index]=local_index;  \n"\
"  localsizes[global_index]=local_size;  \n"\
"  groupids[global_index]=group_index;  \n"\
"  numgroups[global_index]=num_groups; \n"\
"  for(int offset = 1; \n"\
"      offset < local_size; \n"\
"      offset <<= 1) { \n"\
"    int mask = (offset << 1) - 1; \n"\
"    if ((local_index & mask) == 0) { \n"\
"      float other = ftemp[local_index + offset]; \n"\
"      float mine = ftemp[local_index]; \n"\
"      ftemp[local_index] = mine + other; \n"\
"    } \n"\
"    barrier(CLK_LOCAL_MEM_FENCE); \n"\
"  } \n"\
"  if (local_index == 0) { \n"\
"    freduce[group_index] = ftemp[0]; \n"\
"  }\n"\
"} \n"\
"} \n";

char *szDotProduct =
	"#if BIGFLOAT \n"
	"#define FLPT double\n"
	"#else\n"
	"#define FLPT float\n"
	"#endif\n"
"__kernel \n"\
"void dotproduct( \n"\
"            __const int length, \n"\
"            __global FLPT* fin, \n"\
"            __global FLPT* fout, \n"\
"            __local FLPT* ftemp, \n"\
"            __global int* globalids, \n"\
"            __global int* globalsizes, \n"\
"            __global int* localids, \n"\
"            __global int* localsizes, \n"\
"            __global int* groupids, \n"\
"            __global int* numgroups, \n"\
"            __global FLPT* freduce){ \n"\
" \n"\
"  int global_index = get_global_id(0); \n"\
"  int local_index = get_local_id(0); \n"\
"  int global_size = get_global_size(0); \n"\
"  int local_size = get_local_size(0); \n"\
"  int group_index = get_group_id(0); \n"\
"  int num_groups = get_num_groups(0); \n"\
"  if (global_index < length) { \n"\
"  freduce[global_index] = 0.0; \n"\
"  fout[global_index]= 0.0;  \n"\
"  ftemp[local_index] = fin[global_index] * fin[global_index];  \n"\
"  barrier(CLK_LOCAL_MEM_FENCE); \n"\
"  globalids[global_index]=global_index;  \n"\
"  globalsizes[global_index]=global_size;  \n"\
"  localids[global_index]=local_index;  \n"\
"  localsizes[global_index]=local_size;  \n"\
"  groupids[global_index]=group_index;  \n"\
"  numgroups[global_index]=num_groups; \n"\
"  for(int offset = 1; \n"\
"      offset < local_size; \n"\
"      offset <<= 1) { \n"\
"    int mask = (offset << 1) - 1; \n"\
"    if ((local_index & mask) == 0) { \n"\
"      float other = ftemp[local_index + offset]; \n"\
"      float mine = ftemp[local_index]; \n"\
"      ftemp[local_index] = mine + other; \n"\
"    } \n"\
"    barrier(CLK_LOCAL_MEM_FENCE); \n"\
"  } \n"\
"  if (local_index == 0) { \n"\
"    freduce[group_index] = ftemp[0]; \n"\
"  }\n"\
"} \n"\
"} \n";


char *szReduce =
	"#if BIGFLOAT \n"
	"#define FLPT double\n"
	"#else\n"
	"#define FLPT float\n"
	"#endif\n"
"__kernel \n"\
"void reduce( \n"\
"            __const int length, \n"\
"            __global FLPT* fin, \n"\
"            __global FLPT* fout, \n"\
"            __local FLPT* ftemp, \n"\
"            __global int* globalids, \n"\
"            __global int* globalsizes, \n"\
"            __global int* localids, \n"\
"            __global int* localsizes, \n"\
"            __global int* groupids, \n"\
"            __global int* numgroups, \n"\
"            __global FLPT* freduce){ \n"\
" \n"\
"  int global_index = get_global_id(0); \n"\
"  int local_index = get_local_id(0); \n"\
"  int global_size = get_global_size(0); \n"\
"  int local_size = get_local_size(0); \n"\
"  int group_index = get_group_id(0); \n"\
"  int num_groups = get_num_groups(0); \n"\
"  if (global_index < length) { \n"\
"  freduce[global_index] = 0.0; \n"\
"  fout[global_index]= 0.0;  \n"\
"  ftemp[local_index] = fin[global_index]; \n"\
"  barrier(CLK_LOCAL_MEM_FENCE); \n"\
"  globalids[global_index]=global_index;  \n"\
"  globalsizes[global_index]=global_size;  \n"\
"  localids[global_index]=local_index;  \n"\
"  localsizes[global_index]=local_size;  \n"\
"  groupids[global_index]=group_index;  \n"\
"  numgroups[global_index]=num_groups; \n"\
"  for(int offset = 1; \n"\
"      offset < local_size; \n"\
"      offset <<= 1) { \n"\
"    int mask = (offset << 1) - 1; \n"\
"    if ((local_index & mask) == 0) { \n"\
"      float other = ftemp[local_index + offset]; \n"\
"      float mine = ftemp[local_index]; \n"\
"      ftemp[local_index] = mine + other; \n"\
"    } \n"\
"    barrier(CLK_LOCAL_MEM_FENCE); \n"\
"  } \n"\
"  if (local_index == 0) { \n"\
"    freduce[group_index] = ftemp[0]; \n"\
"  }\n"\
"} \n"\
"} \n";





int main(int argc, char *argv[])
{
    int iGlobalSize = 1;
    int iCheck1, iCheck2, iCheck3, iCheck4;
    size_t iGlobalWorkSize = -1;
    size_t iLocalWorkSize = -1;
    if (argc > 1) // Size of input vector
    {
        iCheck1 = atoi(argv[1]);
        if (iCheck1 != 0)
        {
            iGlobalSize = iCheck1;
        }
    }
    int iNoReps = 100; // Number of repetitions.
	if (argc > 2)
	{
		iCheck2 = atoi(argv[2]);
        if (iCheck2 != 0)
        {
            iNoReps = iCheck2;
        }        
	}
    
    
    if (argc > 3) // Global work size
    {
        iCheck3 = atoi(argv[3]);
        if (iCheck3 != 0)
        {
            iGlobalWorkSize = iCheck3;
        }
    }
    if (argc > 4) // Local work size
    {
        iCheck4 = atoi(argv[4]);
        if (iCheck4 != 0)
        {
            iLocalWorkSize = iCheck4;
        }
    }
    int bPrint = 0;
	if (argc > 5)
	{
		bPrint = 1;
	}
    
    
    printf("The global size is %d, the global work size is %ld, and the local work size is %ld. \n", iGlobalSize, iGlobalWorkSize, iLocalWorkSize);
    size_t * ipGlobalWorkParam = NULL;
    if (iGlobalWorkSize != -1)
    {
        ipGlobalWorkParam = &iGlobalWorkSize;
    }
    
    size_t * ipLocalWorkParam = NULL;
    if (iLocalWorkSize != -1)
    {
        ipLocalWorkParam = &iLocalWorkSize;
    }
  
    
    GCAQ * TheGCAQ = GCAQSetup();
    if (TheGCAQ == NULL)
    {
        return 1;
    }

#if BIGFLOAT
	const char *szFloatOpt = "-DBIGFLOAT";
#else
	const char *szFloatOpt = NULL;
#endif
	char *ourKernelStrings[2] =
		{ szDotProduct, szReduce};


  	GPAK *TheGPAK = GPAKSetup(TheGCAQ, 2, ourKernelStrings, szFloatOpt);
    if (TheGPAK == NULL)
    {
        GCAQShutdown(TheGCAQ);
        return 2;
    }

 
    

    FLPT* inputDataF = (FLPT *) malloc(iGlobalSize * sizeof(FLPT));
    SetFIncrease(iGlobalSize, inputDataF);
    
    // For the dot product.
    
    FLPT* outputDataD = (FLPT *) malloc(iGlobalSize * sizeof(FLPT));
    SetFNull(iGlobalSize, outputDataD);

    // For the reduction.
    
    FLPT* outputDataR = (FLPT *) malloc(iGlobalSize * sizeof(FLPT));
    SetFNull(iGlobalSize, outputDataR);
    
    
    
    FLPT* outputDataF = (FLPT *) malloc(iGlobalSize * sizeof(FLPT));
    SetFNull(iGlobalSize, outputDataF);
    int* outputDataGlID = (int *) malloc(iGlobalSize * sizeof(int));
    SetNull(iGlobalSize, outputDataGlID);
    int* outputDataGSize = (int *) malloc(iGlobalSize * sizeof(int));
    SetNull(iGlobalSize, outputDataGSize);
    int* outputDataLcID = (int *) malloc(iGlobalSize * sizeof(int));
    SetNull(iGlobalSize, outputDataLcID);
    int* outputDataLSize = (int *) malloc(iGlobalSize * sizeof(int));
    SetNull(iGlobalSize, outputDataLSize);
    int* outputDataGrID = (int *) malloc(iGlobalSize * sizeof(int));
    SetNull(iGlobalSize, outputDataGrID);
    int* outputDataNGSize = (int *) malloc(iGlobalSize * sizeof(int));
    SetNull(iGlobalSize, outputDataNGSize);
    
	struct timespec start[2];
	struct timespec end[2];
    
// create buffers for the input and ouput

    int err; 
    cl_mem inputF, outputF, outputAll, outputGlID, outputGSize, outputLcID, outputLSize, outputGrID, outputNGSize;
    inputF = clCreateBuffer(TheGCAQ->TheContext, CL_MEM_READ_ONLY, iGlobalSize * sizeof(FLPT), NULL, &err);
    if (err != CL_SUCCESS)
    {
        printf("Error allocating for F");
        return 3;
    }
    
    
    outputGlID = clCreateBuffer(TheGCAQ->TheContext, CL_MEM_WRITE_ONLY, iGlobalSize * sizeof(int), NULL, &err);
    if (err != CL_SUCCESS)
    {
        printf("Error allocating for output 1");
        return 4;
    }


    
    outputGSize = clCreateBuffer(TheGCAQ->TheContext, CL_MEM_WRITE_ONLY, iGlobalSize * sizeof(int), NULL, &err);
    if (err != CL_SUCCESS)
    {
        printf("Error allocating for output 2");
        return 5;
    }


    
    outputLcID = clCreateBuffer(TheGCAQ->TheContext, CL_MEM_WRITE_ONLY, iGlobalSize * sizeof(int), NULL, &err);
    if (err != CL_SUCCESS)
    {
        printf("Error allocating for output 3");
        return 6;
    }

    
    outputLSize = clCreateBuffer(TheGCAQ->TheContext, CL_MEM_WRITE_ONLY, iGlobalSize * sizeof(int), NULL, &err);
    if (err != CL_SUCCESS)
    {
        printf("Error allocating for output 4");
        return 7;
    }

    
    outputGrID = clCreateBuffer(TheGCAQ->TheContext, CL_MEM_WRITE_ONLY, iGlobalSize * sizeof(int), NULL, &err);
    if (err != CL_SUCCESS)
    {
        printf("Error allocating for output 5");
        return 8;
    }

    outputNGSize = clCreateBuffer(TheGCAQ->TheContext, CL_MEM_WRITE_ONLY, iGlobalSize * sizeof(int), NULL, &err);
    if (err != CL_SUCCESS)
    {
        printf("Error allocating for output 6");
        return 9;
    }
    outputF = clCreateBuffer(TheGCAQ->TheContext, CL_MEM_WRITE_ONLY, iGlobalSize * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS)
    {
        printf("Error allocating for output 7");
        return 9;
    }
    outputAll = clCreateBuffer(TheGCAQ->TheContext, CL_MEM_WRITE_ONLY, iGlobalSize * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS)
    {
        printf("Error allocating for output 8");
        return 9;
    }

    clEnqueueWriteBuffer(TheGCAQ->TheQueue, inputF, CL_TRUE, 0, iGlobalSize * sizeof(FLPT), inputDataF, 0, NULL, NULL);

    int iRep;
    int iKernel;
    for (iKernel = 0; iKernel < 2; iKernel++)
    {    
		clock_gettime(CLOCK_MONOTONIC, &(start[iKernel]));
		for (iRep = 0; iRep < iNoReps; iRep++)
		{
        
        
        
    clSetKernelArg(TheGPAK->TheKernels[iKernel], 0, sizeof(int), &iGlobalSize);
    clSetKernelArg(TheGPAK->TheKernels[iKernel], 1, sizeof(cl_mem), &inputF);
    clSetKernelArg(TheGPAK->TheKernels[iKernel], 2, sizeof(cl_mem), &outputF);
    clSetKernelArg(TheGPAK->TheKernels[iKernel], 3, iLocalWorkSize * sizeof(float), NULL);
    clSetKernelArg(TheGPAK->TheKernels[iKernel], 4, sizeof(cl_mem), &outputGlID);
    clSetKernelArg(TheGPAK->TheKernels[iKernel], 5, sizeof(cl_mem), &outputGSize);
    clSetKernelArg(TheGPAK->TheKernels[iKernel], 6, sizeof(cl_mem), &outputLcID);
    clSetKernelArg(TheGPAK->TheKernels[iKernel], 7, sizeof(cl_mem), &outputLSize);
    clSetKernelArg(TheGPAK->TheKernels[iKernel], 8, sizeof(cl_mem), &outputGrID);
    clSetKernelArg(TheGPAK->TheKernels[iKernel], 9, sizeof(cl_mem), &outputNGSize);
    clSetKernelArg(TheGPAK->TheKernels[iKernel], 10, sizeof(cl_mem), &outputAll);    
    clEnqueueNDRangeKernel(TheGCAQ->TheQueue, TheGPAK->TheKernels[iKernel], 1, NULL, ipGlobalWorkParam, ipLocalWorkParam, 0, NULL, NULL);
    clFinish(TheGCAQ->TheQueue);
 
// copy the results from out of the output buffer

    clEnqueueReadBuffer(TheGCAQ->TheQueue, outputGlID, CL_TRUE, 0, iGlobalSize * sizeof(int), outputDataGlID, 0, NULL, NULL);
    clEnqueueReadBuffer(TheGCAQ->TheQueue, outputGSize, CL_TRUE, 0, iGlobalSize * sizeof(int), outputDataGSize, 0, NULL, NULL);    
    clEnqueueReadBuffer(TheGCAQ->TheQueue, outputLcID, CL_TRUE, 0, iGlobalSize * sizeof(int), outputDataLcID, 0, NULL, NULL);
    clEnqueueReadBuffer(TheGCAQ->TheQueue, outputLSize, CL_TRUE, 0, iGlobalSize * sizeof(int), outputDataLSize, 0, NULL, NULL);
    clEnqueueReadBuffer(TheGCAQ->TheQueue, outputGrID, CL_TRUE, 0, iGlobalSize * sizeof(int), outputDataGrID, 0, NULL, NULL);
    clEnqueueReadBuffer(TheGCAQ->TheQueue, outputNGSize, CL_TRUE, 0, iGlobalSize * sizeof(int), outputDataNGSize, 0, NULL, NULL);
    clEnqueueReadBuffer(TheGCAQ->TheQueue, outputF, CL_TRUE, 0, iGlobalSize * sizeof(float), outputDataF, 0, NULL, NULL); 
    if (iKernel == 0)
    {
        clEnqueueReadBuffer(TheGCAQ->TheQueue, outputAll, CL_TRUE, 0, iGlobalSize * sizeof(float), outputDataD, 0, NULL, NULL);
    }
    else
    {
        clEnqueueReadBuffer(TheGCAQ->TheQueue, outputAll, CL_TRUE, 0, iGlobalSize * sizeof(float), outputDataR, 0, NULL, NULL);
    }
        
		clock_gettime(CLOCK_MONOTONIC, &(end[iKernel]));    
    }
    
    }


    clReleaseMemObject(inputF);
    clReleaseMemObject(outputF);
    clReleaseMemObject(outputAll);
    clReleaseMemObject(outputGlID);
    clReleaseMemObject(outputGSize);
    clReleaseMemObject(outputLcID);
    clReleaseMemObject(outputLSize);
    clReleaseMemObject(outputGrID);
    clReleaseMemObject(outputNGSize);     
    
// print the results
    
    if (bPrint)
    {
    int i;
    printf("output %d: \n", iGlobalSize);
    
    
    for(i=0;i<iGlobalSize; i++)
    {
        printf("%d - %f - %f - %f - %d- %d- %d- %d- %d- %d\n", i, inputDataF[i], outputDataD[i], outputDataR[i], outputDataGlID[i],
        outputDataGSize[i], outputDataLcID[i], outputDataLSize[i],
            outputDataGrID[i], outputDataNGSize[i]);
    }
    }
// cleanup - release OpenCL resources
    
    
    free(inputDataF);
    free(outputDataF);
    free(outputDataD);
    free(outputDataR);
    free(outputDataGlID);
    free(outputDataGSize);
    free(outputDataLcID);
    free(outputDataLSize);
    free(outputDataGrID);
    free(outputDataNGSize);
   
    GPAKShutdown(TheGPAK);
    GCAQShutdown (TheGCAQ);
    printf("%d - ", iGlobalSize);
	for (iKernel = 0; iKernel < 2; iKernel++)
	{
		printf("%f - ", (1.0 * TLPERS * iGlobalSize * iNoReps) /
			   (MEGAHERTZ * timespecDiff(&(end[iKernel]), &(start[iKernel]))));
	}
    printf("\n");
    return 0;
}
