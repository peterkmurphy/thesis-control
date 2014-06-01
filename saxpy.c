// saxpy.c. Written by Peter Murphy.
// Copyright (c) 2014.

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <CL/cl.h>
#include "projcommon.h"
#include "openclstuff.h"

#define KERNEL_SIZE 512


char *szSaxpyNormal =
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
	"__kernel void saxpy(__global FLPT *inputY, \n"
	"        __global FLPT *inputX,  FLPT inputA, __global FLPT *output, int iLength)\n"
	"{\n"
	"    size_t id = get_global_id(0);\n"
    "    if (id < iLength) { \n"
	"    output[id] = inputY[id] +  (inputA * inputX[id]);\n" "}}\n";

char *szSaxpyVect2 =
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
	"__kernel void saxpy2(__global FLPTV *inputY, \n"
	"        __global FLPTV *inputX,  FLPT inputA, __global FLPTV *output, int iLength)\n"
	"{\n"
	"    size_t id = get_global_id(0);\n"
    "    if (id < iLength) { \n"    
	"    output[id] = inputY[id] +  (inputA * inputX[id]);\n" "}}\n";


char *szSaxpyVect4 =
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
	"__kernel void saxpy4(__global FLPTV *inputY, \n"
	"        __global FLPTV *inputX,  FLPT inputA, __global FLPTV *output, int iLength)\n"
	"{\n"
	"    size_t id = get_global_id(0);\n"
    "    if (id < iLength) { \n"     
	"    output[id] = inputY[id] +  (inputA * inputX[id]);\n" "}}\n";




char *szSaxpyVect8 =
	"#if BIGFLOAT \n"
"#if defined(cl_khr_fp64)\n"
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
"#elif defined(cl_amd_fp64)\n"
"#pragma OPENCL EXTENSION cl_amd_fp64 : enable\n"
"#endif\n"
	"#define FLPT double\n"
	"#define FLPTV double8\n"
	"#else\n"
	"#define FLPT float\n"
	"#define FLPTV float8\n"
	"#endif\n"
	"__kernel void saxpy8(__global FLPTV *inputY, \n"
	"        __global FLPTV *inputX,  FLPT inputA, __global FLPTV *output, int iLength)\n"
	"{\n"
	"    size_t id = get_global_id(0);\n"
    "    if (id < iLength) { \n"     
	"    output[id] = inputY[id] +  (inputA * inputX[id]);\n" "}}\n";

char *szSaxpyVect16 =
	"#if BIGFLOAT \n"
"#if defined(cl_khr_fp64)\n"
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
"#elif defined(cl_amd_fp64)\n"
"#pragma OPENCL EXTENSION cl_amd_fp64 : enable\n"
"#endif\n"
	"#define FLPT double\n"
	"#define FLPTV double16\n"
	"#else\n"
	"#define FLPT float\n"
	"#define FLPTV float16\n"
	"#endif\n"
	"__kernel void saxpy16(__global FLPTV *inputY, \n"
	"        __global FLPTV *inputX,  FLPT inputA, __global FLPTV *output, int iLength)\n"
	"{\n"
	"    size_t id = get_global_id(0);\n"
    "    if (id < iLength) { \n"      
	"    output[id] = inputY[id] +  (inputA * inputX[id]);\n" "}}\n";

char *szSaxpyImage =
"__constant sampler_t sampler = \n"
"CLK_NORMALIZED_COORDS_FALSE\n"
"| CLK_ADDRESS_CLAMP_TO_EDGE \n"
"| CLK_FILTER_NEAREST; \n"
" \n"
"__kernel void horizontal_reflect(read_only image2d_t src, \n"
"write_only image2d_t dst) \n"
"{ \n"
"    int x = get_global_id(0); \n"
"    // x-coord \n"
"    int y = get_global_id(1); \n"
"    // y-coord \n"
"    int width = get_image_width(src); \n"
"    float4 src_val = read_imagef(src, sampler, (int2)(x, y)); \n" 
"    write_imagef(dst, (int2)(x, y), src_val); \n"
"} \n";



int main(int argc, char *argv[])
{
	int iVal = 8192;
	if (argc > 1)
	{
		iVal = atoi(argv[1]);
	}

	int iNoReps = 100;
	if (argc > 2)
	{
		iNoReps = atoi(argv[2]);
	}
	FLPT fAValue = 10.0;
	if (argc > 3)
	{
		FLPT fBValue = atof(argv[3]);
		if (fBValue != 0.0)
		{
			fAValue = fBValue;
		}
	}
	int bPrint = 0;
	if (argc > 4)
	{
		bPrint = 1;
	}

	const int DATA_SIZE = iVal;
	GCAQ *TheGCAQ = GCAQSetup();
	if (TheGCAQ == NULL)
	{
		return 1;
	}
    
#if BIGFLOAT
	const char *szFloatOpt = "-DBIGFLOAT";
#else
	const char *szFloatOpt = NULL;
#endif
    const int iNoKernels = 5;
	char *ourKernelStrings[6] =
		{ szSaxpyNormal, szSaxpyVect2, szSaxpyVect4, szSaxpyVect8,
szSaxpyVect16, szSaxpyImage };
	GPAK *TheGPAK = GPAKSetup(TheGCAQ, iNoKernels, ourKernelStrings, szFloatOpt);
	if (TheGPAK == NULL)
	{
		GCAQShutdown(TheGCAQ);
		return 2;
	}

	cl_mem inputY, inputX, output;
	size_t global = DATA_SIZE;

	FLPT *inputDataY = (FLPT *) malloc(DATA_SIZE * sizeof(FLPT));
	SetFIncrease(DATA_SIZE, inputDataY);

	FLPT *inputDataX = (FLPT *) malloc(DATA_SIZE * sizeof(FLPT));
	SetFIncrease(DATA_SIZE, inputDataX);
	FLPT inputDataA = fAValue;
	FLPT *results = (FLPT *) malloc(DATA_SIZE * sizeof(FLPT));
	SetFNull(DATA_SIZE, results);
	int i;

	struct timespec start[iNoKernels];
	struct timespec end[iNoKernels];


	int err;
	inputY =
		clCreateBuffer(TheGCAQ->TheContext, CL_MEM_READ_ONLY,
					   sizeof(FLPT) * DATA_SIZE, NULL, &err);
	if (err != CL_SUCCESS)
	{
		printf("Error allocating for Y");
		return 3;
	}

	inputX =
		clCreateBuffer(TheGCAQ->TheContext, CL_MEM_READ_ONLY,
					   sizeof(FLPT) * DATA_SIZE, NULL, &err);
	if (err != CL_SUCCESS)
	{
		printf("Error allocating for X");
		return 4;
	}

	output =
		clCreateBuffer(TheGCAQ->TheContext, CL_MEM_WRITE_ONLY,
					   sizeof(FLPT) * DATA_SIZE, NULL, &err);
	if (err != CL_SUCCESS)
	{
		printf("Error allocating for output");
		return 5;
	}
	int rep;
    int iWorkSize = DATA_SIZE;
    
// Code for images go here. But we'll remove it for now!
    
	int iKernel;
	for (iKernel = 0; iKernel < iNoKernels-1; iKernel++)
	{

		for (i = 0; i < DATA_SIZE; i++)
		{
			results[i] = 0.0;
		}
        // printf("%ld\n", TheGPAK->TheMaxWorkGroupSizes[iKernel]);
		clock_gettime(CLOCK_MONOTONIC, &(start[iKernel]));
		for (rep = 0; rep < iNoReps; rep++)
		{
			// load data into the input buffer
			clEnqueueWriteBuffer(TheGCAQ->TheQueue, inputY, CL_TRUE, 0,
								 sizeof(FLPT) * DATA_SIZE, inputDataY, 0, NULL,
								 NULL);

			clEnqueueWriteBuffer(TheGCAQ->TheQueue, inputX, CL_TRUE, 0,
								 sizeof(FLPT) * DATA_SIZE, inputDataX, 0, NULL,
								 NULL);
			// set the argument list for the kernel command

			clSetKernelArg(TheGPAK->TheKernels[iKernel], 0, sizeof(cl_mem),
						   &inputY);
			clSetKernelArg(TheGPAK->TheKernels[iKernel], 1, sizeof(cl_mem),
						   &inputX);
			clSetKernelArg(TheGPAK->TheKernels[iKernel], 2, sizeof(FLPT),
						   &inputDataA);
			clSetKernelArg(TheGPAK->TheKernels[iKernel], 3, sizeof(cl_mem),
						   &output);
			clSetKernelArg(TheGPAK->TheKernels[iKernel], 4, sizeof(int),
						   &iWorkSize);
			// enqueue the kernel command for execution

			clEnqueueNDRangeKernel(TheGCAQ->TheQueue,
								   TheGPAK->TheKernels[iKernel], 1, NULL,
								   &global,
								NULL, //   &(TheGPAK->TheMaxWorkGroupSizes[iKernel]),
								   0, NULL, NULL);
			clFinish(TheGCAQ->TheQueue);

			// copy the results from out of the output buffer

			clEnqueueReadBuffer(TheGCAQ->TheQueue, output, CL_TRUE, 0,
								sizeof(FLPT) * DATA_SIZE, results, 0, NULL,
								NULL);
		}
		clock_gettime(CLOCK_MONOTONIC, &(end[iKernel]));

		// printing the results.

		if (bPrint)
		{

			for (i = 0; i < DATA_SIZE; i++)
			{
				if (results[i] != (FLPT) ((i + 1) * (fAValue + 1)))
				{
					printf
						("A problem at kernel %d and iteration %d for actual value %f but expected value %f!\n",
						 iKernel, i, results[i],
						 ((FLPT) ((i + 1) * (fAValue + 1))));
					break;
				}
			}
		}
        iWorkSize = iWorkSize / 2;
		global = global / 2;
	}
	// print the results


	// cleanup - release OpenCL resources

	clReleaseMemObject(inputY);
	clReleaseMemObject(inputX);
	clReleaseMemObject(output);
	free(inputDataX);
	free(inputDataY);
	free(results);
	GPAKShutdown(TheGPAK);
	GCAQShutdown(TheGCAQ);
	// printf("%d - %f\n", iVal, timespecDiff(&end, &start) / 1000000000.0);
	printf("%d - ", iVal);
	for (iKernel = 0; iKernel < iNoKernels; iKernel++)
	{
		printf("%f - ", (1.0 * TLPERS * iVal * iNoReps) /
			   (MEGAHERTZ * timespecDiff(&(end[iKernel]), &(start[iKernel]))));
	}
	printf("\n");
	return 0;
}
