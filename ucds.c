/* 
// cds.c. Implementation of Ultra Compressed Diagonal Storage.
// Written by Peter Murphy. (c) 2013
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <omp.h>

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

/* Useful for defining number of diagonals in matrices. */

#define SMALLDIAG 3
#define MIDDIAG 5
#define LARGEDIAG 27

/* Useful for counting number of test cases. */

#define NUMTESTS 4

/* Defines minimum matrix size for test 3 with 5 diagonals. */

#define MINDIAGT3 4

/* Defines minimum matrix size for test with 27 diagonals. */

#define MINDIAGT27 14

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

ucds* create_ucds(const INTG lmatsize, INTG * ldiagindices, const INTG lnumdiag)
{
    if ((lmatsize < 1) || (ldiagindices[0] < (1 - lmatsize)) || 
        (ldiagindices[0] > ldiagindices[lnumdiag - 1]) 
        || ((lmatsize - 1) < ldiagindices[lnumdiag - 1])) 
    {
        return NULL;
    }
    ucds* ourucds = (ucds *)malloc(1 * sizeof(ucds));
    ourucds->lmatsize = lmatsize;
    ourucds->lnumdiag = lnumdiag;
    ourucds->ldiagindices = ldiagindices;
    ourucds->ddiagelems = (FLPT *)malloc(lnumdiag * lmatsize * 
        sizeof(FLPT));
    return ourucds;
}

/* The destroy_ucds function deallocates and destroys a ucds instance. */

void destroy_ucds(ucds * ourucds)
{
    free(ourucds->ddiagelems); 
    free(ourucds);
}

/* 
// The multiply_ucds performs matrix vector multiplication using the ucds type.
// The arguments are:
//
// - ourucds: a pointer to a ucds instance.
// - dvector: a vector that can be passed in as a double[] type. Note that 
// the programmer should make sure that there is at least ourucds->lmatsize
// instances.
//
// If successful, the function creates and allocates a vector that is the
// result of the desired matrix multiplication. Otherwise, it returns 
// NULL.
*/

FLPT * multiply_ucds(const ucds *ourucds, const FLPT *dvector)
{
    FLPT * dret = (FLPT *) calloc (ourucds->lmatsize,sizeof(FLPT));
    INTG i, j; /* Iteration variables */
    INTG lrevindex; /* Current diagonal index to evaluate. */
    INTG miniter, maxiter; /* Sets range to iterate over in value lookup. */
    #pragma omp parallel for private(lrevindex, miniter, maxiter)
    for (i = 0; i < ourucds->lnumdiag; i++)
    {
        lrevindex = ourucds->ldiagindices[i];
        miniter = max(0, lrevindex);
        maxiter = min(ourucds->lmatsize - 1, ourucds->lmatsize - 1 
            + lrevindex);
        #pragma omp parallel for
        for (j = miniter; j <= maxiter; j++)
        {
            dret[j - lrevindex] += ourucds->ddiagelems[i*ourucds->lmatsize 
                + j] * dvector[j];
        }
    }
    return dret; 
}

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
    ddiagvals, const INTG lnumdiag)
{
    
/* 
// Before allocating space for the cds, we have to check that parameters 
// make a M-matrix creation possible.
*/

    if ((ldiagindices[0] > 0) || (ldiagindices[lnumdiag - 1] < 0))
    {
        return NULL;
    }
    
    register FLPT dsum = 0.0; /* This checks diagonal dominance. */ 
    INTG i, j; /* Iteration variables. */
    for (i = 0; i < lnumdiag; i++)
    {
        dsum += ddiagvals[i];
    }
    if (dsum < 0.0)
    {
        return NULL;
    }
    
/* Now we can allocate the cds. */    
    
    ucds * ourucds = create_ucds(lmatsize, ldiagindices, lnumdiag);
    #pragma omp parallel for
    for (i = 0; i < lnumdiag; i++)
    {
        #pragma omp parallel for
        for (j = 0; j < lmatsize; j++)
        {
            ourucds->ddiagelems[i*lmatsize + j] = ddiagvals[i];
        }
    }
    return ourucds;
}

/*
// The following routine is adapted from the following Stack Overflow
// article: http://stackoverflow.com/questions/361363/ \
// how-to-measure-time-in-milliseconds-using-ansi-c 
//
// It gives the differences in nanoseconds between two events
*/

TLEN timespecDiff(struct timespec *ptime1, struct timespec *ptime2)
{
    const int NSPERS = TLPERS; /* Nanoseconds per seconds. */
    return ((ptime1->tv_sec * NSPERS) + ptime1->tv_nsec) -
           ((ptime2->tv_sec * NSPERS) + ptime2->tv_nsec);
}


int main(int argc, char *argv[])
{

    if (argc < 2)
    {
        printf ("To execute this, type:\n\nudcs n\n\nWhere n > 1 is the ");
        printf ("size of the elements to be multiplied and tested.\n");
        return(0);
    }
    const INTG imatsize = atoi(argv[1]);
    if (imatsize <= 1)
    {
        printf ("Please pass a numerical argument greater or equal to 2.\n");
        return(0);
    }

    INTG i; /* Iteration variable. */
    struct timespec start, end; /* Contains test start and end times. */
    TLEN testLengths[NUMTESTS]; /*Stores the lengths of various tests. */
    
/* What we do now is set up a big whopping random test vector. */

    clock_gettime(CLOCK_REALTIME, &start);
    srand(start.tv_sec * TLPERS + start.tv_nsec);
    FLPT * dvector = (FLPT *)malloc(imatsize * sizeof (FLPT));
    if (dvector == NULL)
    {
        return (0);
    }
    #pragma omp parallel for
    for (i = 0; i < imatsize; i++)
    {
        dvector[i] = (FLPT)rand()/RAND_MAX;
    }

/* 
// Now the first test is to see how it multiplies with an identity matrix
// stored in diagonal form.
*/

    const INTG inumsimpdiag = SMALLDIAG;
    INTG lsimpelems[SMALLDIAG] = {-1, 0, 1};
    FLPT delems_id[SMALLDIAG] = {0.0, 1.0, 0.0};  
    ucds * ouriducds = mmatrix_ucds(imatsize, lsimpelems, delems_id, 
        inumsimpdiag);  

/*
// The next test is to test how the test vector multiples with a UCDS matrix
// when the main diagonals are 1, the immediate off diagonals above and below
// are -1, and all other diagonals are 0. An example of a matrix that looks
// like it is the following:
//    
//    [  4 -1  ]
//    [ -1  4  ] 
*/  

    FLPT delems_pde[SMALLDIAG] = {-1.0, 4.0, -1.0}; 
    ucds * ourpdeucds = mmatrix_ucds(imatsize, lsimpelems, delems_pde, 
        inumsimpdiag);  

/* The next test involves a matrix for a 2D EPDE over a larger grid. */
    
    const INTG inummiddiag = MIDDIAG;
    INTG lmidelems[MIDDIAG] = {-3, -1, 0, 1, 3};
    FLPT dmidel_vals[MIDDIAG] = {-1.0, -1.0, 4.0, -1.0, -1.0};
    ucds * ourmiducds = NULL;
    const INTG middiagbarrier = MINDIAGT3; /* Min matrix size for this test. */
    if (imatsize >= middiagbarrier)
    {
        ourmiducds = mmatrix_ucds(imatsize, lmidelems, dmidel_vals, 
            inummiddiag);  
    }
    
/* Another test involves 27 diagonals - like a 3D EPDE stencil. */ 

    const INTG inumlargdiag = LARGEDIAG;
    INTG llargelems[LARGEDIAG];
    FLPT dlargel_vals[LARGEDIAG];
    for (i = 0; i < LARGEDIAG; i++)
    {
        llargelems[i] = i - 14;
        if (i == 14)
        {
            dlargel_vals[i] = 26.0;
        }
        else
        {
            dlargel_vals[i] = -1.0;          
        }
    }
    ucds * ourlargeucds = NULL;
    const INTG largediagbarrier = MINDIAGT27; /* Min matrix size . */
    if (imatsize >= largediagbarrier)
    {
        ourlargeucds = mmatrix_ucds(imatsize, llargelems, dlargel_vals, 
            inumlargdiag);  
    }    
    
    clock_gettime(CLOCK_MONOTONIC, &start);

/* Now we do multiplying and counting the time elapsed. */
    
    FLPT * dsame = multiply_ucds(ouriducds, dvector);
    clock_gettime(CLOCK_MONOTONIC, &end);
    testLengths[0] = timespecDiff(&end, &start);
    clock_gettime(CLOCK_MONOTONIC, &start);    
    FLPT * dpde = multiply_ucds(ourpdeucds, dvector);
    clock_gettime(CLOCK_MONOTONIC, &end);
    testLengths[1] = timespecDiff(&end, &start);
    FLPT * dmidpde = NULL;
    if (imatsize >= middiagbarrier)
    {
        clock_gettime(CLOCK_MONOTONIC, &start); 
        dmidpde = multiply_ucds(ourmiducds, dvector);
        clock_gettime(CLOCK_MONOTONIC, &end);
        testLengths[2] = timespecDiff(&end, &start);
    }
    else
    {
        testLengths[2] = 0;
    }
    FLPT * dlargpde = NULL;
    if (imatsize >= largediagbarrier)
    {
        clock_gettime(CLOCK_MONOTONIC, &start); 
        dlargpde = multiply_ucds(ourlargeucds, dvector);
        clock_gettime(CLOCK_MONOTONIC, &end);
        testLengths[3] = timespecDiff(&end, &start);
    }
    else
    {
        testLengths[3] = 0;
    }
    for (i = 0; i < NUMTESTS; i++)
    {
        printf("%f, ", (FLPT)testLengths[i]/TLPERS);

    }
    printf("%d\n", imatsize);
    
/* The last stage is to free up all the memory used. */  

    if (imatsize >= largediagbarrier)
    {
        destroy_ucds(ourlargeucds);
        free(dlargpde);
    }
    if (imatsize >= middiagbarrier)
    {
        destroy_ucds(ourmiducds);
        free(dmidpde);
    }
    destroy_ucds(ouriducds);  
    destroy_ucds(ourpdeucds);  
    free(dpde);
    free(dsame);
    free(dvector);  

    return 0;
}
