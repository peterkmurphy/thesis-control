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

#define NUMTESTS 12

/* Defines minimum matrix size for test 3 with 5 diagonals. */

#define MINDIAGT3 4

/* Defines minimum matrix size for test with 27 diagonals. */

#define MINDIAGT27 14

/* The following are definitions for vector related routines. */

/* 
// The dassign function assigns memory for a FLPT vector. The argument:
// - isize: the number of elements in the vector.
// The return value is the vector.
// Note: the space for the vector should be deallocated after use using
// the free function or similar. 
*/

FLPT * dassign(const INTG isize);

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
// The drandomvector does the same, except that the values are random
// (between 0 and 1).
*/

FLPT * drandomvector(const INTG isize);

/*
// The doverwrite function acts like the dsetvector function: it sets
// the values of a vector (dret) to a value (dvalue). However, dret
// must be already initialised.
*/

FLPT * doverwritevector(const INTG isize, const FLPT dvalue, FLPT* dret);

/* The doverwriterandom tries it with random vectors. */

FLPT * doverwriterandom(const INTG isize, FLPT* dret);

/* 
// The dveccopy copies a vector from one value to another. Arguments:
// lvectsize: the size of the vector.
// doverwrite: the destination vector.
// dsource: the dsource vector.
// The function returns doverwrite.
*/

FLPT * dveccopy (const INTG lvectsize, FLPT * doverwrite, 
    const FLPT *dsource);

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
// The dselfdprod performs the dot product between a vector and itself.
// The arguments:
// - lvecsize: the size of the vector arguments;
// - dvector: the vector itself.
// The function returns the result.
*/

FLPT dselfdprod(const INTG lvectsize, const FLPT * dvector);

/*
// The scalar product performs the product between a scalar and a vector
// and returns the result:
// - lvecsize: the size of the vector argument;
// - dscalar: the scalar.
// - dvectin: the input vector.
// - dvectout: the output vector.
// The function returns dvectout.
*/

FLPT * dscalarprod (const INTG lvectsize, const FLPT dscalar, 
    const FLPT * dvectin, FLPT * dvectout);

/*
// The vector sum adds two vectors and returns the result. The arguments:
// - lvecsize: the size of the vector arguments;
// - dleftvec: the vector on the left hand side of the argument.
// - drightvec: the vector on the right hand side of the argument.
// - dvectout: the output vector.
// The function returns dvectout.
*/

FLPT * dvectadd (const INTG lvectsize, const FLPT * dleftvec, 
    const FLPT * drightvec, FLPT * dvectout);

/*
// The dvectsub function subtracts two vectors and returns the result.
// The arguments:
// - lvecsize: the size of the vector arguments;
// - dleftvec: the vector on the left hand side of the argument.
// - drightvec: the vector on the right hand side of the argument.
// - dvectout: the output vector.
*/

FLPT * dvectsub (const INTG lvectsize, const FLPT * dleftvec, 
    const FLPT * drightvec, FLPT * dvectout);

/*
// The daddinsitu function effectively calculates doverwrite += 
// dconst * drightvec. It is used as it is quicker to modify an
// existing vector rather than create one anew with the right
// value. Arguments:
// lvecsize: the size of the vector.
// doverwrite: the vector to be added to and adjusted.
// dconst: the amount to multiply drightvect by in the sum.
// drightvec: the vector to be added on the right hand side (when
// multiplied by dconst).
// The function returns doverwrite.
*/

FLPT * daddinsitu (const INTG lvectsize, FLPT * doverwrite, 
    const FLPT dconst, const FLPT * drightvec);

/*
// The daddtwosums functions overwrites dadjust as follows:
// dadjust = dleftconst * dleftvect + dreftconst * dreftvect.
// lvectsize is the size of the vector. The function returns
// dadjust.
*/

FLPT * daddtwosums (const INTG lvectsize, FLPT * dadjust, 
    const FLPT *dleftvec, const FLPT dleftconst, 
    const FLPT * drightvec, const FLPT drightconst);
/* 
// The dvectnorm function calculates the norm of a vector. Arguments:
// lvectsize: the size of the vector.
// mode: the type of norm Choose 1 for a 1-norm, 2 for a 2-norm, and
// any other value for an infinity norm.
// dvectin: the vector to be analysed.
// The function returns the value of the norm.
// Note: the dvectnorm function generates the norm by iterating through
// the values in dvectin. 
*/

FLPT dvectnorm (const INTG lvectsize, const INTG mode, 
    const FLPT * dvectin);

/* 
// The daltnorm function is an alternative implementation of the vector
// norm. It uses recursion instead of iteration. The arguments are
// otherwise the same.
*/

FLPT daltnorm (const INTG lvectsize, const INTG mode, 
    const FLPT * dvectin);

/* Some thing: A */


FLPT * dvecoverwrite (const INTG lvectsize, FLPT * doverwrite, 
    const FLPT dconst, const FLPT * drightvec);

/* The following are definitions for UCDS related structures and routines. */

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
// Note 1: the user has to set the values of ldiagindices and ddiagelems
// themselves after allocation. An alternate function for creating an UCDS
// is mmatrix_ucds (q.v.), where all values in any diagonal are equal to
// each other (but not necessarily to the values in other diagonals).
// Note 2: unpredicatable behaviour may result if ldiagindices is unsorted
// or contains repeated values.
// Note 3: The destroy_ucds function can be used to deallocate the ucds
// created here.
*/

ucds* create_ucds(const INTG lmatsize, INTG * ldiagindices, 
    const INTG lnumdiag);

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

// ? Does it make sense?

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
// Note: this function parallelises the outer loop under OpenMP
*/

FLPT * multiply_ucds(const ucds *ourucds, const FLPT *dvector, FLPT * dret);

/* 
// The multiply_ucdsalt function is an alternative implementation for 
// UCDS vector multiplication. The difference is that the OpenMP code
// parallelises the inner loop. The code is otherwise the same.
*/

FLPT * multiply_ucdsalt(const ucds *ourucds, const FLPT *dvector, FLPT * dret);

/* 
// The multiply_ucds27 routine is like the multiply_ucds routine; the only
// difference is that the number of diagonals is hardwired at 27 by const
// statements. This is only used to check how performance is optimised 
// under these conditions.
// 
// The multiply_ucds5 is similar, except that the number of diagonals is
// hardwired at 5.
//
// These routines use OpenMP outer loop parallelisation.
*/

FLPT * multiply_ucds27(const ucds *ourucds, const FLPT *dvector, FLPT * dret);
FLPT * multiply_ucds5(const ucds *ourucds, const FLPT *dvector, FLPT * dret);

/* 
// The multiply_ucdsaltX (X an integer) are like the multiply_ucdsX functions:
// hardwire the number of diagonals at X to check the performance as consts. The only
// difference is that these use inner loop parallelisation. The arguments
// are otherwise the same.
*/

FLPT * multiply_ucdsalt27(const ucds *ourucds, const FLPT *dvector, FLPT * dret);
FLPT * multiply_ucdsalt5(const ucds *ourucds, const FLPT *dvector, FLPT * dret);

/* 
// These four functions use #defines rather than consts, but are otherwise the
// same.
*/

FLPT * multiply_ucdsd27(const ucds *ourucds, const FLPT *dvector, FLPT * dret);
FLPT * multiply_ucdsd5(const ucds *ourucds, const FLPT *dvector, FLPT * dret);
FLPT * multiply_ucdsaltd27(const ucds *ourucds, const FLPT *dvector, FLPT * dret);
FLPT * multiply_ucdsaltd5(const ucds *ourucds, const FLPT *dvector, FLPT * dret);

/* This code is for testing. */

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
//
// The fpnorm typedef represents pointers to vector norm functions used 
// for these tests.
*/

typedef FLPT * (* fpmult) (const ucds *, const FLPT *, FLPT *);

typedef FLPT (* fpnorm) (const INTG, const INTG, const FLPT *);

typedef struct {
    INTG lnumdiag;
    INTG * ldiagindices;
    FLPT * ddiagelems;
    ucds * ourucds;
    FLPT * dret;
    fpmult thefp;
    TLEN testlen;
    INTG inoreps;
} mmtestbed;

/*
// The mmtestsetup does the hard work of setting up a mmtestbed item.
// The only parameters are the lnumdiag parameter (for the number of
// diagonals), the mmref item (which references the mmtestbed
// instance itself), and ivectsize (the size of vectors for the test).
*/

mmtestbed * mmsetup(INTG lnumdiag, INTG ivectsize, mmtestbed * mmref);

/* The mmsetdown element destroys the information in a mmsetup instance. */

void mmdestroy(mmtestbed * mmref);
    
/*
// The timespecDiff routine gives the differences in nanoseconds between
// two events ptime1 (the end) and ptime2 (the start).
*/

TLEN timespecDiff(struct timespec *ptime1, struct timespec *ptime2);

/*
// The printvector function prints a vector to standard output. The
// arguments:
// name: the name of the variable represented.
// isize: the size of the vector.
// dvector: the vector.
// The function returns no arguments.
*/

void printvector(const char* name, INTG isize, const FLPT* dvector);


/*
// The dconjgrad function is an implementation of the conjugate gradient
// algorithm - a Kyrlov subspace iterative method for returning the 
// vector solution x to the linear equation ax = b. The arguments are:
// ucdsa: the matrix a (in UCDS form)
// dvectb: the vector b
// dvectx0: a starting guess for the solution x.
// dvectx: the vector x (which is returned).
// fpucdsmult: a reference to a ucds multiplication function (of type fpmult). 
// fpdnorm: a reference to a norm returning function (of type fpnorm).
// imode: the mode of the norm used to measure the "error".
// derror: the maximum error in a possible solution.
// inoiter: if this parameter is not NULL, the function sets it to the number of
// iterations necessary to arrive at a solution. (This parameter is ignored
// if it is NULL
//
// If successful, the function returns dvectx (which represents the vector x).
// Otherwise, it returns NULL.
*/

FLPT * dconjgrad(const ucds * ucdsa, const FLPT * dvectb, const FLPT *dvectx0,
    FLPT * dvectx, fpmult fpucdsmult, fpnorm fpdnorm, 
    INTG imode, const FLPT derror, INTG * inoiter);

#endif /* UCDS_H */    
