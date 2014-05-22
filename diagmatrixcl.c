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

char *szDiagAligned = 
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
" __kernel void dia_align(__global FLPT *A,\n"\
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


char *szDiagLocal = 
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
" __kernel void dia_local(__global FLPT *A,\n"\
"   __const int rows, __const int diags, __global int *offsets, \n"\
"              __global FLPT *x, __global FLPT *y, __local FLPT* ftemp) { \n"\
"  int local_id = get_local_id(0); \n"\
"  int offset_id = local_id; \n"\
"  while ((offset_id < 256) && (offset_id < diags)) { \n"\
"    ftemp[offset_id] = offsets[offset_id]; \n"\
"    offset_id = offset_id + get_local_size(0); \n"\
"  } \n"\
"  barrier(CLK_LOCAL_MEM_FENCE); \n"\
"  int row = get_global_id(0); \n"\
"  FLPT accumulator = 0; \n"\
"  __global FLPT* matrix_offset = A + row; \n"\
"  for(int diag = 0; diag < diags; diag++) { \n"\
"    int col = row + ftemp[diag]; \n"\
"    if ((col >= 0) && (col < rows)) { \n"\
"      FLPT m = *matrix_offset; \n"\
"      FLPT v = x[col]; \n"\
"      accumulator += m * v; \n"\
"    } \n"\
"    matrix_offset += rows; \n"\
"  } \n"\
"  y[row] = accumulator; \n"  \
"} \n"; 


char *szDiagAligned2 =
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
" __kernel void dia_alignv2(__global FLPT *A,\n"\
"   __const int rows, __const int diags, __global int *offsets, \n"\
"              __global FLPT *x, __global FLPT *y) { \n"\
"  int row = get_global_id(0) * 2; \n"\
"  FLPTV accumulator = 0; \n"\
"  __global FLPT* matrix_offset = A + row; \n"\
"  for(int diag = 0; diag < diags; diag++) { \n"\
"    int col = row + offsets[diag]; \n"\
"    FLPTV m = vload2(0, matrix_offset); \n"\
"    FLPTV v; \n"\
"    if ((col >= 0) && (col < rows - 2)) { \n"\
"      v = vload2(0, x + col); \n"\
"    } else { \n"\
"      int2 id = col + (int2)(0, 1); \n"\
"      int2 in_bounds = (id >= 0) && (id < rows); \n"\
"      v.x = in_bounds.x ? x[id.x] : 0; \n"\
"      v.y = in_bounds.y ? x[id.y] : 0; \n"\
"    } \n"\
"    accumulator += m * v; \n"\
"    matrix_offset += rows; \n"\
"  } \n"\
"  vstore2(accumulator, 0, row + y); \n"\
"} \n"; 


char *szDiagLocal2 =
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
" __kernel void dia_localv2(__global FLPT *A,\n"\
"   __const int rows, __const int diags, __global int *offsets, \n"\
"              __global FLPT *x, __global FLPT *y, __local FLPT* ftemp) { \n"\
"  int local_id = get_local_id(0); \n"\
"  int offset_id = local_id; \n"\
"  while ((offset_id < 256) && (offset_id < diags)) { \n"\
"    ftemp[offset_id] = offsets[offset_id]; \n"\
"    offset_id = offset_id + get_local_size(0); \n"\
"  } \n"\
"  barrier(CLK_LOCAL_MEM_FENCE); \n"\
"  int row = get_global_id(0) * 2; \n"\
"  FLPTV accumulator = 0; \n"\
"  __global float* matrix_offset = A + row; \n"\
"  for(int diag = 0; diag < diags; diag++) { \n"\
"    int col = row + ftemp[diag]; \n"\
"    FLPTV m = vload2(0, matrix_offset); \n"\
"    FLPTV v; \n"\
"    if ((col >= 0) && (col < rows - 2)) { \n"\
"      v = vload2(0, x + col); \n"\
"    } else { \n"\
"      int2 id = col + (int2)(0, 1); \n"\
"      int2 in_bounds = (id >= 0) && (id < rows); \n"\
"      v.x = in_bounds.x ? x[id.x] : 0; \n"\
"      v.y = in_bounds.y ? x[id.y] : 0; \n"\
"    } \n"\
"    accumulator += m * v; \n"\
"    matrix_offset += rows; \n"\
"  } \n"\
"  vstore2(accumulator, 0, row + y); \n"\
"} \n";


char *szDiagAligned4 =
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
" __kernel void dia_alignv4(__global FLPT *A,\n"\
"   __const int rows, __const int diags, __global int *offsets, \n"\
"              __global FLPT *x, __global FLPT *y) { \n"\
"  int row = get_global_id(0) * 4; \n"\
"  FLPTV accumulator = 0; \n"\
"  __global FLPT* matrix_offset = A + row; \n"\
"  for(int diag = 0; diag < diags; diag++) { \n"\
"    int col = row + offsets[diag]; \n"\
"    FLPTV m = vload4(0, matrix_offset); \n"\
"    FLPTV v; \n"\
"    if ((col >= 0) && (col < rows - 4)) { \n"\
"      v = vload4(0, x + col); \n"\
"    } else { \n"\
"      int4 id = col + (int4)(0, 1, 2, 3); \n"\
"      int4 in_bounds = (id >= 0) && (id < rows); \n"\
"      v.x = in_bounds.x ? x[id.x] : 0; \n"\
"      v.y = in_bounds.y ? x[id.y] : 0; \n"\
"      v.z = in_bounds.z ? x[id.z] : 0; \n"\
"      v.w = in_bounds.w ? x[id.w] : 0; \n"\
"    } \n"\
"    accumulator += m * v; \n"\
"    matrix_offset += rows; \n"\
"  } \n"\
"  vstore4(accumulator, 0, row + y); \n"\
"} \n"; 


char *szDiagLocal4 =
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
" __kernel void dia_localv4(__global FLPT *A,\n"\
"   __const int rows, __const int diags, __global int *offsets, \n"\
"              __global FLPT *x, __global FLPT *y, __local FLPT* ftemp) { \n"\
"  int local_id = get_local_id(0); \n"\
"  int offset_id = local_id; \n"\
"  while ((offset_id < 256) && (offset_id < diags)) { \n"\
"    ftemp[offset_id] = offsets[offset_id]; \n"\
"    offset_id = offset_id + get_local_size(0); \n"\
"  } \n"\
"  barrier(CLK_LOCAL_MEM_FENCE); \n"\
"  int row = get_global_id(0) * 4; \n"\
"  FLPTV accumulator = 0; \n"\
"  __global float* matrix_offset = A + row; \n"\
"  for(int diag = 0; diag < diags; diag++) { \n"\
"    int col = row + ftemp[diag]; \n"\
"    FLPTV m = vload4(0, matrix_offset); \n"\
"    FLPTV v; \n"\
"    if ((col >= 0) && (col < rows - 4)) { \n"\
"      v = vload4(0, x + col); \n"\
"    } else { \n"\
"      int4 id = col + (int4)(0, 1, 2, 3); \n"\
"      int4 in_bounds = (id >= 0) && (id < rows); \n"\
"      v.x = in_bounds.x ? x[id.x] : 0; \n"\
"      v.y = in_bounds.y ? x[id.y] : 0; \n"\
"      v.z = in_bounds.z ? x[id.z] : 0; \n"\
"      v.w = in_bounds.w ? x[id.w] : 0; \n"\
"    } \n"\
"    accumulator += m * v; \n"\
"    matrix_offset += rows; \n"\
"  } \n"\
"  vstore4(accumulator, 0, row + y); \n"\
"} \n";

char *szDiagAligned8 =
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
" __kernel void dia_alignv8(__global FLPT *A,\n"\
"   __const int rows, __const int diags, __global int *offsets, \n"\
"              __global FLPT *x, __global FLPT *y) { \n"\
"  int row = get_global_id(0) * 8; \n"\
"  FLPTV accumulator = 0; \n"\
"  __global FLPT* matrix_offset = A + row; \n"\
"  for(int diag = 0; diag < diags; diag++) { \n"\
"    int col = row + offsets[diag]; \n"\
"    FLPTV m = vload8(0, matrix_offset); \n"\
"    FLPTV v; \n"\
"    if ((col >= 0) && (col < rows - 8)) { \n"\
"      v = vload8(0, x + col); \n"\
"    } else { \n"\
"      int8 id = col + (int8)(0, 1, 2, 3, 4, 5, 6, 7); \n"\
"      int8 in_bounds = (id >= 0) && (id < rows); \n"\
"      v.s0 = in_bounds.s0 ? x[id.s0] : 0; \n"\
"      v.s1 = in_bounds.s1 ? x[id.s1] : 0; \n"\
"      v.s2 = in_bounds.s2 ? x[id.s2] : 0; \n"\
"      v.s3 = in_bounds.s3 ? x[id.s3] : 0; \n"\
"      v.s4 = in_bounds.s4 ? x[id.s4] : 0; \n"\
"      v.s5 = in_bounds.s5 ? x[id.s5] : 0; \n"\
"      v.s6 = in_bounds.s6 ? x[id.s6] : 0; \n"\
"      v.s7 = in_bounds.s7 ? x[id.s7] : 0; \n"\
"    } \n"\
"    accumulator += m * v; \n"\
"    matrix_offset += rows; \n"\
"  } \n"\
"  vstore8(accumulator, 0, row + y); \n"\
"} \n"; 


char *szDiagLocal8 =
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
" __kernel void dia_localv8(__global FLPT *A,\n"\
"   __const int rows, __const int diags, __global int *offsets, \n"\
"              __global FLPT *x, __global FLPT *y, __local FLPT* ftemp) { \n"\
"  int local_id = get_local_id(0); \n"\
"  int offset_id = local_id; \n"\
"  while ((offset_id < 256) && (offset_id < diags)) { \n"\
"    ftemp[offset_id] = offsets[offset_id]; \n"\
"    offset_id = offset_id + get_local_size(0); \n"\
"  } \n"\
"  barrier(CLK_LOCAL_MEM_FENCE); \n"\
"  int row = get_global_id(0) * 8; \n"\
"  FLPTV accumulator = 0; \n"\
"  __global float* matrix_offset = A + row; \n"\
"  for(int diag = 0; diag < diags; diag++) { \n"\
"    int col = row + ftemp[diag]; \n"\
"    FLPTV m = vload8(0, matrix_offset); \n"\
"    FLPTV v; \n"\
"    if ((col >= 0) && (col < rows - 8)) { \n"\
"      v = vload8(0, x + col); \n"\
"    } else { \n"\
"      int8 id = col + (int8)(0, 1, 2, 3, 4, 5, 6, 7); \n"\
"      int8 in_bounds = (id >= 0) && (id < rows); \n"\
"      v.s0 = in_bounds.s0 ? x[id.s0] : 0; \n"\
"      v.s1 = in_bounds.s1 ? x[id.s1] : 0; \n"\
"      v.s2 = in_bounds.s2 ? x[id.s2] : 0; \n"\
"      v.s3 = in_bounds.s3 ? x[id.s3] : 0; \n"\
"      v.s4 = in_bounds.s4 ? x[id.s4] : 0; \n"\
"      v.s5 = in_bounds.s5 ? x[id.s5] : 0; \n"\
"      v.s6 = in_bounds.s6 ? x[id.s6] : 0; \n"\
"      v.s7 = in_bounds.s7 ? x[id.s7] : 0; \n"\
"    } \n"\
"    accumulator += m * v; \n"\
"    matrix_offset += rows; \n"\
"  } \n"\
"  vstore8(accumulator, 0, row + y); \n"\
"} \n";

char *szDiagAligned16 =
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
" __kernel void dia_alignv16(__global FLPT *A,\n"\
"   __const int rows, __const int diags, __global int *offsets, \n"\
"              __global FLPT *x, __global FLPT *y) { \n"\
"  int row = get_global_id(0) * 16; \n"\
"  FLPTV accumulator = 0; \n"\
"  __global FLPT* matrix_offset = A + row; \n"\
"  for(int diag = 0; diag < diags; diag++) { \n"\
"    int col = row + offsets[diag]; \n"\
"    FLPTV m = vload16(0, matrix_offset); \n"\
"    FLPTV v; \n"\
"    if ((col >= 0) && (col < rows - 16)) { \n"\
"      v = vload16(0, x + col); \n"\
"    } else { \n"\
"      int16 id = col + (int16)(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15); \n"\
"      int16 in_bounds = (id >= 0) && (id < rows); \n"\
"      v.s0 = in_bounds.s0 ? x[id.s0] : 0; \n"\
"      v.s1 = in_bounds.s1 ? x[id.s1] : 0; \n"\
"      v.s2 = in_bounds.s2 ? x[id.s2] : 0; \n"\
"      v.s3 = in_bounds.s3 ? x[id.s3] : 0; \n"\
"      v.s4 = in_bounds.s4 ? x[id.s4] : 0; \n"\
"      v.s5 = in_bounds.s5 ? x[id.s5] : 0; \n"\
"      v.s6 = in_bounds.s6 ? x[id.s6] : 0; \n"\
"      v.s7 = in_bounds.s7 ? x[id.s7] : 0; \n"\
"      v.s8 = in_bounds.s8 ? x[id.s8] : 0; \n"\
"      v.s9 = in_bounds.s9 ? x[id.s9] : 0; \n"\
"      v.sa = in_bounds.sa ? x[id.sa] : 0; \n"\
"      v.sb = in_bounds.sb ? x[id.sb] : 0; \n"\
"      v.sc = in_bounds.sc ? x[id.sc] : 0; \n"\
"      v.sd = in_bounds.sd ? x[id.sd] : 0; \n"\
"      v.se = in_bounds.se ? x[id.se] : 0; \n"\
"      v.sf = in_bounds.sf ? x[id.sf] : 0; \n"\
"    } \n"\
"    accumulator += m * v; \n"\
"    matrix_offset += rows; \n"\
"  } \n"\
"  vstore16(accumulator, 0, row + y); \n"\
"} \n"; 


char *szDiagLocal16 =
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
" __kernel void dia_localv16(__global FLPT *A,\n"\
"   __const int rows, __const int diags, __global int *offsets, \n"\
"              __global FLPT *x, __global FLPT *y, __local FLPT* ftemp) { \n"\
"  int local_id = get_local_id(0); \n"\
"  int offset_id = local_id; \n"\
"  while ((offset_id < 256) && (offset_id < diags)) { \n"\
"    ftemp[offset_id] = offsets[offset_id]; \n"\
"    offset_id = offset_id + get_local_size(0); \n"\
"  } \n"\
"  barrier(CLK_LOCAL_MEM_FENCE); \n"\
"  int row = get_global_id(0) * 16; \n"\
"  FLPTV accumulator = 0; \n"\
"  __global float* matrix_offset = A + row; \n"\
"  for(int diag = 0; diag < diags; diag++) { \n"\
"    int col = row + ftemp[diag]; \n"\
"    FLPTV m = vload16(0, matrix_offset); \n"\
"    FLPTV v; \n"\
"    if ((col >= 0) && (col < rows - 8)) { \n"\
"      v = vload16(0, x + col); \n"\
"    } else { \n"\
"      int16 id = col + (int16)(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15); \n"\
"      int16 in_bounds = (id >= 0) && (id < rows); \n"\
"      v.s0 = in_bounds.s0 ? x[id.s0] : 0; \n"\
"      v.s1 = in_bounds.s1 ? x[id.s1] : 0; \n"\
"      v.s2 = in_bounds.s2 ? x[id.s2] : 0; \n"\
"      v.s3 = in_bounds.s3 ? x[id.s3] : 0; \n"\
"      v.s4 = in_bounds.s4 ? x[id.s4] : 0; \n"\
"      v.s5 = in_bounds.s5 ? x[id.s5] : 0; \n"\
"      v.s6 = in_bounds.s6 ? x[id.s6] : 0; \n"\
"      v.s7 = in_bounds.s7 ? x[id.s7] : 0; \n"\
"      v.s8 = in_bounds.s8 ? x[id.s8] : 0; \n"\
"      v.s9 = in_bounds.s9 ? x[id.s9] : 0; \n"\
"      v.sa = in_bounds.sa ? x[id.sa] : 0; \n"\
"      v.sb = in_bounds.sb ? x[id.sb] : 0; \n"\
"      v.sc = in_bounds.sc ? x[id.sc] : 0; \n"\
"      v.sd = in_bounds.sd ? x[id.sd] : 0; \n"\
"      v.se = in_bounds.se ? x[id.se] : 0; \n"\
"      v.sf = in_bounds.sf ? x[id.sf] : 0; \n"\
"    } \n"\
"    accumulator += m * v; \n"\
"    matrix_offset += rows; \n"\
"  } \n"\
"  vstore16(accumulator, 0, row + y); \n"\
"} \n";

// This is a quick and dirty routine that assumes that we have 
// (a) All non-zero values in the diagonal matrix are 1,
// (b) iNoDiagonals is odd.
// (c) All diagonals go from (-iNoDiagonal/2, iNoDiagonal/2)
// If you do anything different, write a new function.


FLPT * flptcalculateeasydiagonal(INTG iMatrixSize, INTG iNoDiagonals, 
    FLPT * fToWrite)
{
    INTG iIter;
    INTG HalfVal = iNoDiagonals / 2;
    for (iIter = 0; iIter < iMatrixSize; iIter++)
    {
        if (iIter < HalfVal)
        {
            fToWrite[iIter] =  1.0 * (iNoDiagonals - HalfVal + iIter);
        }
        else if (iIter >= (iMatrixSize - HalfVal))
        {
            
            fToWrite[iIter] =  1.0 * (iNoDiagonals + (iMatrixSize - HalfVal) - iIter - 1);   
        }
        else
        {
            fToWrite[iIter] = 1.0 * iNoDiagonals;
        }
    }
    return fToWrite;
}
    
    



int main(int argc, char *argv[])
{
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
    const int iNumberOfKernels = 11;
#if BIGFLOAT
	const char *szFloatOpt = "-DBIGFLOAT";
#else
	const char *szFloatOpt = NULL;
#endif
	char *ourKernelStrings[11] =
		{  szDiagMult, szDiagLocal, szDiagAligned, szDiagLocal2, szDiagAligned2,
            szDiagLocal4, szDiagAligned4, szDiagLocal8, szDiagAligned8,
            szDiagLocal16, szDiagAligned16}; 


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

    FLPT * fDiagResultTest = (FLPT *) malloc(iNumRows * sizeof(FLPT));
    flptcalculateeasydiagonal(iNumRows, iNumDiags, fDiagResultTest);
    printvector("Result to check", iNumRows, fDiagResultTest);
    
    
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
            
// The local versions of matrix version code uses a local argument.
    
    if ((iKernel % 2) == 1)
    {
        clSetKernelArg(TheGPAK->TheKernels[iKernel], 6, 256 * sizeof(float), NULL);
    }
  
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

			for (i = 0; i < iNumRows; i++)
			{
				if (outputDataY[i] != fDiagResultTest[i])
				{
					printf
						("A problem at kernel %d and iteration %d for actual value %f but expected value %f!\n",
						 iKernel, i, outputDataY[i], fDiagResultTest[i]);
					break;
				}
			}
		}



    //		if (bPrint)
	//	{

     //       printf("output: ");
    //        for(i=0;i<iNumRows; i++)
    //        {
    //            printf("%f ",outputDataY[i]);
    //        }
    //        printf("\n");
	//	}
    
    
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
    free(fDiagResultTest);
    clReleaseMemObject(outputY);
    clReleaseMemObject(inputOff);
    clReleaseMemObject(inputX);
    clReleaseMemObject(inputA);
    GPAKShutdown(TheGPAK);
    GCAQShutdown (TheGCAQ);
    return 0;
}