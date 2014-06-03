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


char *szDotProduct =
	"#if BIGFLOAT \n"
"#if defined(cl_khr_fp64)\n"
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
"#elif defined(cl_amd_fp64)\n"
"#pragma OPENCL EXTENSION cl_amd_fp64 : enable\n"
"#endif\n"
	"#define FLPT double\n"
	"#else\n"
	"#define FLPT float\n"
	"#endif\n"
"__kernel \n"\
"void dotproduct( \n"\
"            __const int length, \n"\
"            __global FLPT* fin, \n"\
"            __local FLPT* ftemp, \n"\
"            __global FLPT* freduce){ \n"\
" \n"\
"  int global_index = get_global_id(0); \n"\
"  int local_index = get_local_id(0); \n"\
"  int local_size = get_local_size(0); \n"\
"  int group_index = get_group_id(0); \n"\
"  if (global_index < length) { \n"\
"  freduce[global_index] = 0.0; \n"\
"  ftemp[local_index] = fin[global_index] * fin[global_index];  \n"\
"  barrier(CLK_LOCAL_MEM_FENCE); \n"\
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


char *szDotProduct2 =
	"#if BIGFLOAT \n"
"#if defined(cl_khr_fp64)\n"
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
"#elif defined(cl_amd_fp64)\n"
"#pragma OPENCL EXTENSION cl_amd_fp64 : enable\n"
"#endif\n"
	"#define FLPT double\n"
	"#define FLPTV double2\n"    
	"#else\n"
	"#define FLPT float\n"
	"#define FLPTV float2\n"
	"#endif\n"
"__kernel \n"\
"void dotproduct2( \n"\
"            __const int length, \n"\
"            __global FLPTV* fin, \n"\
"            __local FLPT* ftemp, \n"\
"            __global FLPT* freduce){ \n"\
" \n"\
"  int global_index = get_global_id(0); \n"\
"  int local_index = get_local_id(0); \n"\
"  int local_size = get_local_size(0); \n"\
"  int group_index = get_group_id(0); \n"\
"  if (global_index < length) { \n"\
"  freduce[global_index] = 0.0; \n"\
"  ftemp[local_index/2] = dot(fin[global_index/2],fin[global_index/2]);  \n"\
"  barrier(CLK_LOCAL_MEM_FENCE); \n"\
"  for(int offset = 1; \n"\
"      offset < local_size/2; \n"\
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

char *szDotProduct4 =
	"#if BIGFLOAT \n"
"#if defined(cl_khr_fp64)\n"
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
"#elif defined(cl_amd_fp64)\n"
"#pragma OPENCL EXTENSION cl_amd_fp64 : enable\n"
"#endif\n"
	"#define FLPT double\n"
	"#define FLPTV double4\n"    
	"#else\n"
	"#define FLPT float\n"
	"#define FLPTV float4\n"
	"#endif\n"
"__kernel \n"\
"void dotproduct4( \n"\
"            __const int length, \n"\
"            __global FLPTV* fin, \n"\
"            __local FLPT* ftemp, \n"\
"            __global FLPT* freduce){ \n"\
" \n"\
"  int global_index = get_global_id(0); \n"\
"  int local_index = get_local_id(0); \n"\
"  int local_size = get_local_size(0); \n"\
"  int group_index = get_group_id(0); \n"\
"  if (global_index < length) { \n"\
"  freduce[global_index] = 0.0; \n"\
"  ftemp[local_index/4] = dot(fin[global_index/4],fin[global_index/4]);  \n"\
"  barrier(CLK_LOCAL_MEM_FENCE); \n"\
"  for(int offset = 1; \n"\
"      offset < local_size/4; \n"\
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
"#if defined(cl_khr_fp64)\n"
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
"#elif defined(cl_amd_fp64)\n"
"#pragma OPENCL EXTENSION cl_amd_fp64 : enable\n"
"#endif\n"
	"#define FLPT double\n"
	"#else\n"
	"#define FLPT float\n"
	"#endif\n"
"__kernel \n"\
"void reduce( \n"\
"            __const int length, \n"\
"            __global FLPT* fin, \n"\
"            __local FLPT* ftemp, \n"\
"            __global FLPT* freduce){ \n"\
" \n"\
"  int global_index = get_global_id(0); \n"\
"  int local_index = get_local_id(0); \n"\
"  int local_size = get_local_size(0); \n"\
"  int group_index = get_group_id(0); \n"\
"  if (global_index < length) { \n"\
"  freduce[global_index] = 0.0; \n"\
"  ftemp[local_index] = fin[global_index]; \n"\
"  barrier(CLK_LOCAL_MEM_FENCE); \n"\
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


char *szReduce2 =
	"#if BIGFLOAT \n"
"#if defined(cl_khr_fp64)\n"
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
"#elif defined(cl_amd_fp64)\n"
"#pragma OPENCL EXTENSION cl_amd_fp64 : enable\n"
"#endif\n"
	"#define FLPT double\n"
	"#define FLPTV double2\n"    
	"#else\n"
	"#define FLPT float\n"
	"#define FLPTV float2\n"
	"#endif\n"
"__kernel \n"\
"void reduce2( \n"\
"            __const int length, \n"\
"            __global FLPTV* fin, \n"\
"            __local FLPT* ftemp, \n"\
"            __global FLPT* freduce){ \n"\
" \n"\
"  int global_index = get_global_id(0); \n"\
"  int local_index = get_local_id(0); \n"\
"  int local_size = get_local_size(0); \n"\
"  int group_index = get_group_id(0); \n"\
"  if (global_index < length) { \n"\
"  freduce[global_index] = 0.0; \n"\
"  ftemp[local_index/2] = fin[global_index/2].s0 + fin[global_index/2].s1; \n"\
"  barrier(CLK_LOCAL_MEM_FENCE); \n"\
"  for(int offset = 1; \n"\
"      offset < local_size/2; \n"\
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






char *szReduce4 =
	"#if BIGFLOAT \n"
"#if defined(cl_khr_fp64)\n"
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
"#elif defined(cl_amd_fp64)\n"
"#pragma OPENCL EXTENSION cl_amd_fp64 : enable\n"
"#endif\n"
	"#define FLPT double\n"
	"#define FLPTV double4\n"    
	"#else\n"
	"#define FLPT float\n"
	"#define FLPTV float4\n"
	"#endif\n"
"__kernel \n"\
"void reduce4( \n"\
"            __const int length, \n"\
"            __global FLPTV* fin, \n"\
"            __local FLPT* ftemp, \n"\
"            __global FLPT* freduce){ \n"\
" \n"\
"  int global_index = get_global_id(0); \n"\
"  int local_index = get_local_id(0); \n"\
"  int local_size = get_local_size(0); \n"\
"  int group_index = get_group_id(0); \n"\
"  if (global_index < length) { \n"\
"  freduce[global_index] = 0.0; \n"\
"  ftemp[local_index/4] = fin[global_index/4].s0 + fin[global_index/4].s1 + fin[global_index/4].s2 + fin[global_index/4].s3; \n"\
"  barrier(CLK_LOCAL_MEM_FENCE); \n"\
"  for(int offset = 1; \n"\
"      offset < local_size/4; \n"\
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
    
/*    
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
    } */
    int bPrint = 0;
	if (argc > 3) // Originally 5.
	{
		bPrint = 1;
	}

 //   printf("The global size is %d, the global work size is %ld, and the local work size is %ld. \n", iGlobalSize, iGlobalWorkSize, iLocalWorkSize);
/*    size_t * ipGlobalWorkParam = NULL;
    if (iGlobalWorkSize != -1)
    {
        ipGlobalWorkParam = &iGlobalWorkSize;
    }
    
    size_t * ipLocalWorkParam = NULL;
    if (iLocalWorkSize != -1)
    {
        ipLocalWorkParam = &iLocalWorkSize;
    } */
    
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
    const int iNoKernels = 1;
	char *ourKernelStrings[6] =
		{ szDotProduct, szReduce, szDotProduct2, szReduce2, szDotProduct4, szReduce4};


  	GPAK *TheGPAK = GPAKSetup(TheGCAQ, iNoKernels, ourKernelStrings, szFloatOpt);
    if (TheGPAK == NULL)
    {
        GCAQShutdown(TheGCAQ);
        return 2;
    }

    INTG iTypicalWorkgroupNo = TheGPAK->TheMaxWorkGroupSizes[0];
    
    INTG iExpOutputSize = ioutsize(iGlobalSize, iTypicalWorkgroupNo);
    FLPT * fExpDotProdResult = (FLPT *) malloc(iExpOutputSize * sizeof(FLPT));
    FLPT * fExpReduceResult = (FLPT *) malloc(iExpOutputSize * sizeof(FLPT));
    fdotprodexpresult(iGlobalSize, iTypicalWorkgroupNo, fExpDotProdResult);
    freduceexpresult(iGlobalSize, iTypicalWorkgroupNo, fExpReduceResult);
 //   printvector("dot prod", iExpOutputSize, fExpDotProdResult);
 //   printvector("reduce", iExpOutputSize, fExpReduceResult);

    FLPT* inputDataF = (FLPT *) malloc(iGlobalSize * sizeof(FLPT));
    SetFIncrease(iGlobalSize, inputDataF);
    
    // For the dot product.
    
    FLPT* outputDataD = (FLPT *) malloc(iGlobalSize * sizeof(FLPT));
    SetFNull(iGlobalSize, outputDataD);

    // For the reduction.
    
    FLPT* outputDataR = (FLPT *) malloc(iGlobalSize * sizeof(FLPT));
    SetFNull(iGlobalSize, outputDataR);
    
	struct timespec start[iNoKernels];
	struct timespec end[iNoKernels];
    
// create buffers for the input and ouput

    int err; 
    cl_mem inputF, outputF, outputAll;
    inputF = clCreateBuffer(TheGCAQ->TheContext, CL_MEM_READ_ONLY, iGlobalSize * sizeof(FLPT), NULL, &err);
    if (err != CL_SUCCESS)
    {
        printf("Error allocating for F");
        return 3;
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
    int i;
    int iLengthTotal = iGlobalSize;
    size_t iGlobalWorkThing = iGlobalSize;
    int iSomething = 1;
    for (iKernel = 0; iKernel < iNoKernels; iKernel++)
    {    
		for (i = 0; i < iLengthTotal; i++)
		{
			outputDataD[i] = 0.0;
            outputDataR[i] = 0.0;
		}
        
        clock_gettime(CLOCK_MONOTONIC, &(start[iKernel]));
		for (iRep = 0; iRep < iNoReps; iRep++)
		{
            clSetKernelArg(TheGPAK->TheKernels[iKernel], 0, sizeof(int), &iLengthTotal);
            clSetKernelArg(TheGPAK->TheKernels[iKernel], 1, sizeof(cl_mem), &inputF);
            clSetKernelArg(TheGPAK->TheKernels[iKernel], 2, iSomething * iLocalWorkSize * sizeof(float), NULL); // Was 3
            clSetKernelArg(TheGPAK->TheKernels[iKernel], 3, sizeof(cl_mem), &outputAll); // Was 4         
            clEnqueueNDRangeKernel(TheGCAQ->TheQueue, TheGPAK->TheKernels[iKernel], 1,
                NULL, &iGlobalWorkThing, &(TheGPAK->TheMaxWorkGroupSizes[iKernel]), 0, NULL, NULL);
            clFinish(TheGCAQ->TheQueue);
 
// copy the results from out of the output buffer

            if (iKernel % 2 ==  0)
            {
                clEnqueueReadBuffer(TheGCAQ->TheQueue, outputAll, CL_TRUE, 0, iExpOutputSize * sizeof(float), outputDataD, 0, NULL, NULL);
            }
            else
            {
                clEnqueueReadBuffer(TheGCAQ->TheQueue, outputAll, CL_TRUE, 0, iExpOutputSize * sizeof(float), outputDataR, 0, NULL, NULL);
            }
        
        }
        clock_gettime(CLOCK_MONOTONIC, &(end[iKernel]));    
        if (bPrint)
        {
            for (i = 0; i < iExpOutputSize; i++)
			{
				if (iKernel % 2 == 0)
                {
                    if (outputDataD[i] != fExpDotProdResult[i])
                    {
                        printf
                            ("A problem at kernel %d and iteration %d for actual value %f but expected value %f!\n",
                            iKernel, i, outputDataD[i], fExpDotProdResult[i]);
                        break;
                    }
                }
                else
                {
                    if (outputDataR[i] != fExpReduceResult[i])
                    {
                        printf
                            ("A problem at kernel %d and iteration %d for actual value %f but expected value %f!\n",
                            iKernel, i, outputDataR[i], fExpReduceResult[i]);
                        break;
                    }
                }
			}    
        }
   //     if ((iKernel % 2) == 1)
   //     {
   //         iLengthTotal = iLengthTotal / 2;
    //        iSomething = iSomething * 2;
    //        iGlobalWorkThing = iGlobalWorkThing / 2;
  //      }
    }


    clReleaseMemObject(inputF);
    clReleaseMemObject(outputF);
    clReleaseMemObject(outputAll);
    
// print the results
//    if (bPrint)
//    {
//        printf("output %d: \n", iGlobalSize);
//        for(i=0;i<iExpOutputSize; i++)
//        {
//            printf("%d - %f - %f\n", i, outputDataD[i], outputDataR[i]);
//        }   
//    }
    
// cleanup - release OpenCL resources
    
    free(inputDataF);
    free(outputDataD);
    free(outputDataR);
   
    GPAKShutdown(TheGPAK);
    GCAQShutdown (TheGCAQ);
    printf("%d - ", iGlobalSize);
	for (iKernel = 0; iKernel < iNoKernels; iKernel++)
	{
		printf("%f - ", (1.0 * TLPERS * iGlobalSize * iNoReps) /
			   (MEGAHERTZ * timespecDiff(&(end[iKernel]), &(start[iKernel]))));
	}
    printf("\n");
    return 0;
}
