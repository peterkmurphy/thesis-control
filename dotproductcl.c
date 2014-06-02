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
"            __global FLPT* freduce){ \n"\
" \n"\
"  int global_index = get_global_id(0); \n"\
"  int local_index = get_local_id(0); \n"\
"  int global_size = get_global_size(0); \n"\
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
"            __global FLPT* freduce){ \n"\
" \n"\
"  int global_index = get_global_id(0); \n"\
"  int local_index = get_local_id(0); \n"\
"  int global_size = get_global_size(0); \n"\
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


int main(int argc, char *argv[])
{
    int iGlobalSize = 1;
    int iCheck1, iCheck2;

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
    int bPrint = 0;
	if (argc > 3) // Originally 5
	{
		bPrint = 1;
	}    
    
/*    
    int iCheck3, iCheck4;    
    size_t iGlobalWorkSize = -1;
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
*/  
    
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
    
    const int iNoKernels = 2;
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

    int iRep; // Used for repetitions.
    int iKernel; // Used for counting the kernels used.
    
    size_t iNoWorkGroups = iGlobalSize;
    
    for (iKernel = 0; iKernel < iNoKernels; iKernel++)
    {    
		size_t iLocalWorkSize = TheGPAK->TheMaxWorkGroupSizes[iKernel]; // Different but same.
        size_t iWorkGroupSize = TheGPAK->TheMaxWorkGroupSizes[iKernel]; // Same but different
        clock_gettime(CLOCK_MONOTONIC, &(start[iKernel]));
		for (iRep = 0; iRep < iNoReps; iRep++)
		{
            clSetKernelArg(TheGPAK->TheKernels[iKernel], 0, sizeof(int), &iGlobalSize);
            clSetKernelArg(TheGPAK->TheKernels[iKernel], 1, sizeof(cl_mem), &inputF);
            clSetKernelArg(TheGPAK->TheKernels[iKernel], 2, sizeof(cl_mem), &outputF);
            clSetKernelArg(TheGPAK->TheKernels[iKernel], 3, iLocalWorkSize * sizeof(float), NULL);
            clSetKernelArg(TheGPAK->TheKernels[iKernel], 4, sizeof(cl_mem), &outputAll);         
            clEnqueueNDRangeKernel(TheGCAQ->TheQueue, TheGPAK->TheKernels[iKernel], 1, NULL, 
                &iNoWorkGroups, &iWorkGroupSize, 0, NULL, NULL);
            clFinish(TheGCAQ->TheQueue);
 
// copy the results from out of the output buffer

            clEnqueueReadBuffer(TheGCAQ->TheQueue, outputF, CL_TRUE, 0, iGlobalSize * sizeof(float), outputDataF, 0, NULL, NULL); 
            if (iKernel == 0)
            {
                clEnqueueReadBuffer(TheGCAQ->TheQueue, outputAll, CL_TRUE, 0, iGlobalSize * sizeof(float), outputDataD, 0, NULL, NULL);
            }
            else
            {
                clEnqueueReadBuffer(TheGCAQ->TheQueue, outputAll, CL_TRUE, 0, iGlobalSize * sizeof(float), outputDataR, 0, NULL, NULL);
            }
        }
		clock_gettime(CLOCK_MONOTONIC, &(end[iKernel]));    
    }


    clReleaseMemObject(inputF);
    clReleaseMemObject(outputF);
    clReleaseMemObject(outputAll);
    
// print the results
    
    if (bPrint)
    {
        int i;
        printf("output %d: \n", iGlobalSize);
        for(i=0;i<iGlobalSize; i++)
        {
            printf("%d - %f - %f - %f\n", i, inputDataF[i], outputDataD[i], outputDataR[i]);
        }
    }
// cleanup - release OpenCL resources
    
    
    free(inputDataF);
    free(outputDataF);
    free(outputDataD);
    free(outputDataR);
   
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