/* 
// ucds.h. Header for Ultra Compressed Diagonal Storage.
// Written by Peter Murphy. (c) 2013
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <omp.h>

#ifndef UCDS_H
#define UCDS_H

/* Quick and dirty min and max. */

#define max(x, y) (((x) > (y)) ? (x) : (y))
#define min(x, y) (((x) < (y)) ? (x) : (y))

/* 
// We use typedefs for integers and floating point numbers so their size
// and base types can be changed in one place.
*/

typedef int INTG;
typedef double FLPT;

/* We use a typedef for storing time lengths. */

typedef uint64_t TLEN;

/* And timelengths per second. On Linux, it is 10^9 for nanoseconds. */

#define TLPERS 1000000000

/* But values will be displayed in Megahertz, so we need to divide. */

#define MEGAHERTZ 1000000.0

/* Useful for defining number of diagonals in matrices. */

#define SMALLDIAG 3
#define MIDDIAG 5
#define LARGEDIAG 27

/* Useful for counting number of test cases. */

#define NUMTESTS 10

/* Defines minimum matrix size for test 3 with 5 diagonals. */

#define MINDIAGT3 4

/* Defines minimum matrix size for test with 27 diagonals. */

#define MINDIAGT27 14

/*
// The dot product performs the dot product of two vectors and returns
// the result. The arguments:
// - lvecsize: the size of the vector arguments;
// - dleftvec: the vector on the left hand side of the argument.
// - drightvec: the vector on the right hand side of the argument.
*/

FLPT ddotprod (const INTG lvectsize, const FLPT * dleftvec, 
    const FLPT * drightvec);

/*
// The scalar product performs the product between a scalar and a vector
// and returns the result:
// - lvecsize: the size of the vector argument;
// - dscalar: the scalar.
// - dvectin: the input vector.
// - dvectout: the output vector.
*/

FLPT * dscalarprod (const INTG lvectsize, const FLPT dscalar, 
    const FLPT * dvectin, FLPT * dvectout);

/*
// The vector sum adds two vectors and returns the result. The arguments:
// - lvecsize: the size of the vector arguments;
// - dleftvec: the vector on the left hand side of the argument.
// - drightvec: the vector on the right hand side of the argument.
// - dvectout: the output vector.
*/

FLPT * dvectadd (const INTG lvectsize, const FLPT * dleftvec, 
    const FLPT * drightvec, FLPT * dvectout);

/*
// The vector difference subtracts two vectors and returns the result.
// The arguments:
// - lvecsize: the size of the vector arguments;
// - dleftvec: the vector on the left hand side of the argument.
// - drightvec: the vector on the right hand side of the argument.
// - dvectout: the output vector.
*/

FLPT * dvectsub (const INTG lvectsize, const FLPT * dleftvec, 
    const FLPT * drightvec, FLPT * dvectout);

/* 
// The vector norm calculates the norm of a vector. Arguments:
// lvectsize - the size of the vector.
// mode - the type of norm. Choose 1 for a 1-norm, 2 for a 2-norm, and
// any other value for an infinity norm.
// dvectin - the vector to be analysed.
// The return value is the norm.
*/

FLPT dvectnorm (const INTG lvectsize, const INTG mode, 
    const FLPT * dvectin);


/* The daltnorm function is an alternative; it uses recursion. */


FLPT daltnorm (const INTG lvectsize, const INTG mode, 
    const FLPT * dvectin);
    

/* 
// The ucds ("Ultra Compressed Diagonal Storage") scheme is a modification
// of the cds scheme defined above. Rather than store diagonals as a 
// contiguous band, store individual diagonals with their diagonal index.
// The type is more suitable for the matrices created by 3D EPDEs (or their
// discretisation thereof). To make storage and lookup more efficient, the
// values in all diagonals are stored in one big double[], rather than a
// double[][] or a **double.
// 
// For each element A[i][j] in A, we assign it to the diagonal with an index
// (j - i). We then define the ucds structure as follows:
// - lmatsize: the size of the square matrix represented.
// - lnumdiag: the number of diagonals represented.
// - ldiagindices: an array of the indices of the diagonals stored here.
// - ddiagelems: an array of the elements in the diagonal. The length of the
// array is ldiagindex*lmatsize, and the first element of the i-th diagonal
// is at ddiagelems[i*lmatsize].
//
// Example: the following matrix: 
//
//                                [ 1 2 ]
//                                [ 3 4 ] 
//
// Would have lmatsize = 2, lnumdiag = 3, ldiagindices = [-1, 0, 1] and
// ddiagelems = [3.0, N, 1.0, 4.0, N, 2.0]. Here, N means not important.
*/    

typedef struct {
    INTG lmatsize;
    INTG lnumdiag;
    INTG * ldiagindices; 
    FLPT *ddiagelems; 
} ucds;

/* 
// The create_ucds function creates and allocates an instance of the ucds 
// structure using the following arguments:
//
// - lmatsize: the size of the square matrix represented.
// - ldiagindices: an array of diagonal indices (in ascending order with
//   no repeated values).
// - lnumdiag: the size of ldiagindices.
//
// If successful, a ucds* is returned; otherwise, the function returns NULL.
//
// Note 1: the user has to initialise the values themselves.
// Note 2: unpredicatable behaviour may result if ldiagindices is unsorted
// or contains repeated values.
*/

ucds* create_ucds(const INTG lmatsize, INTG * ldiagindices, 
    const INTG lnumdiag);


/* The destroy_ucds function deallocates and destroys a ucds instance. */

void destroy_ucds(ucds * ourucds);

/* 
// The multiply_ucds performs matrix vector multiplication using the ucds type.
// The arguments are:
//
// - ourucds: a pointer to a ucds instance.
// - dvector: a vector that can be passed in as a double[] type. Note that 
// the programmer should make sure that there is at least ourucds->lmatsize
// instances.
// - dret: a vector to be set to the result of the multiplication. This 
// must have ourucds->lmatsize values allocated to it already.
//
// If successful, the function sets dret to the result of the multiplication.
// Otherwise, it returns NULL.
//
// Note: this version parallelises the outer loop.
*/

FLPT * multiply_ucds(const ucds *ourucds, const FLPT *dvector, FLPT * dret);

/* This is the same, except it parallelises the inner loop. */


FLPT * multiply_ucdsalt(const ucds *ourucds, const FLPT *dvector, FLPT * dret);


/* 
// The multiply_ucds27 routine is like the multiply_ucds routine; the only
// difference is that the number of diagonals is hardwired at 27. This is
// only used to check how performance is optimised under these conditions.
// 
// The multiply_ucds5 is similar, except that the number of diagonals is
// hardwired at 5.
//
// The arguments and return values are otherwise the same.
//
// Note: outer loop parallel:
//
*/

FLPT * multiply_ucds27(const ucds *ourucds, const FLPT *dvector, FLPT * dret);

/* Inner loop parallel */

FLPT * multiply_ucdsalt27(const ucds *ourucds, const FLPT *dvector, FLPT * dret);

/* Outer loop parallel */

FLPT * multiply_ucds5(const ucds *ourucds, const FLPT *dvector, FLPT * dret);

/* Inner loop parallel */

FLPT * multiply_ucdsalt5(const ucds *ourucds, const FLPT *dvector, FLPT * dret);

/* 
// mmatrix_ucds generates a "sample" M-matrix in ucds form. A M-matrix is 
// a matrix which is strictly diagonally dominant, but all off-diagonal 
// entries are negative or zero. The arguments are:
//
// - lmatsize: the size of the square matrix represented.
// - ldiagindices: an array of diagonal indices (in ascending order with
//   no repeated values).
// - ddiagvals: a list of values to set the diagonals for the indices.
//   listed in ldiagindices.
// - lindicessize: the size of ldiagindices (and ddiagvals).
//
// If successful, a ucds* is returned; otherwise, the function returns NULL.
//
// Note: the conditions for M-matrixhood are checked when initialising 
// the structure. If they aren't satisfied, the function returns NULL.
*/

ucds * mmatrix_ucds(const INTG lmatsize, INTG * ldiagindices, FLPT *
    ddiagvals, const INTG lnumdiag);

/*
// The mmtestbed structure allows all the test data for a test to
// be embedded in one structure. This saves an excess of variables, and
// an excess of confusion. The components are:
// - lnumdiag: the number of diagonals for the test.
// - ldiagindices: the indices of the diagonals in the test. 
// - ddiagelems: the numerical value used in each diagonals. 
// - ourucds: the resulting UCDS structure. 
// - dret: the vector used to contain the result of caclulations. 
// - thefp: a pointer (of type fpmult) to the multiplication function.
// - testlen: the time taken to execute the test (in ns).     
//
// The fpmult typedef represents pointers to multiplication functions used
// for these tests.
*/

typedef FLPT * (* fpmult) (const ucds *, const FLPT *, FLPT *);

typedef struct {
    INTG lnumdiag;
    INTG * ldiagindices;
    FLPT * ddiagelems;
    ucds * ourucds;
    FLPT * dret;
    fpmult thefp;
    TLEN testlen;
} mmtestbed;

/*
// The dsetvector routine creates and initialises a vector, so that
// all values are set to one constant. The code is expressed as a
// routine so that there's no premature optimisation happening from
// having the functionality inline. Arguments:
// - isize: the size of the vector.
// - dvalue: the value to set every element in the vector. 
//
// The function returns the vector if successful, and NULL otherwise.
*/

FLPT * dsetvector(const INTG isize, const FLPT dvalue);

/*
// The createspdd function creates initialisation data for the 
// mmatrix_ucds function above. Arguments:
// inodiags: the number of diagonals
// ldiagelems: the indices for the diagonals.
// ddiagvals: the values on the diagonals for those indices.
//
// The function returns 0 for failure, and 1 for success.
*/

INTG createspdd(INTG inodiags, INTG * ldiagelems, FLPT * ddiagvals);

/* The dassign function assigns a double vector of size isize. */

FLPT * dassign(const INTG isize);

/*
// This is an imp[lementation of the conjugate gradient algorithm,
// which tries to return the solution x to the equation Ax = b.
// Arguments
// ucdsa: the matrix a
// dvectb: the vector b
// dvectx: the vector x (which is returned).
// imode: the mode of the norm used to measure the "error".
// derror: the minimum error that the function can accept.
// icount: the number of iterations (which is set and passed).
//
// The function returns x.
*/

FLPT * conjgrad(const ucds * ucdsa, const FLPT * dvectb, const FLPT *dvectx0,
    FLPT * dvectx, const INTG imode, const INTG itype, const FLPT derror, INTG * iiter);


FLPT * dvecoverwrite (const INTG lvectsize, FLPT * doverwrite, 
    const FLPT dconst, const FLPT * drightvec);


    







#endif /* UCDS_H */    
