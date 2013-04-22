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

#define NUMTESTS 10

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
// - dret: a vector to be set to the result of the multiplication. This 
// must have ourucds->lmatsize values allocated to it already.
//
// If successful, the function sets dret to the result of the multiplication.
// Otherwise, it returns NULL.
*/

FLPT * multiply_ucds(const ucds *ourucds, const FLPT *dvector, FLPT * dret)
{
    if ((ourucds == NULL) || (dvector == NULL) || (dret == NULL))
    {
        return NULL;
    }
    
    INTG i, j; /* Iteration variables */
    INTG lrevindex; /* Current diagonal index to evaluate. */
    INTG miniter, maxiter; /* Sets range to iterate over in value lookup. */
    
    for (i = 0; i < ourucds->lmatsize; i++)
    {
        dret[i] = 0.0;
    }
    
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
// The multiply_ucds27 routine is like the multiply_ucds routine; the only
// difference is that the number of diagonals is hardwired at 27. This is
// only used to check how performance is optimised under these conditions.
// 
// The multiply_ucds5 is similar, except that the number of diagonals is
// hardwired at 5.
//
// The arguments and return values are otherwise the same.
*/

FLPT * multiply_ucds27(const ucds *ourucds, const FLPT *dvector, FLPT * dret)
{
    if ((ourucds == NULL) || (dvector == NULL) || (dret == NULL))
    {
        return NULL;
    }

    const INTG idiagnum = 27;
    INTG i, j; /* Iteration variables */
    INTG lrevindex; /* Current diagonal index to evaluate. */
    INTG miniter, maxiter; /* Sets range to iterate over in value lookup. */
    
    for (i = 0; i < ourucds->lmatsize; i++)
    {
        dret[i] = 0.0;
    }
    
    #pragma omp parallel for private(lrevindex, miniter, maxiter)
    for (i = 0; i < idiagnum; i++)
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

FLPT * multiply_ucds5(const ucds *ourucds, const FLPT *dvector, FLPT * dret)
{
    if ((ourucds == NULL) || (dvector == NULL) || (dret == NULL))
    {
        return NULL;
    }

    const INTG idiagnum = 5;
    INTG i, j; /* Iteration variables */
    INTG lrevindex; /* Current diagonal index to evaluate. */
    INTG miniter, maxiter; /* Sets range to iterate over in value lookup. */
    
    for (i = 0; i < ourucds->lmatsize; i++)
    {
        dret[i] = 0.0;
    }
    
    #pragma omp parallel for private(lrevindex, miniter, maxiter)
    for (i = 0; i < idiagnum; i++)
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


FLPT * dsetvector(const INTG isize, const FLPT dvalue)
{
    FLPT * dret = (FLPT *) malloc(isize * sizeof (FLPT)); /* Return value. */
    INTG i; /* Iteration variable. */
    if (dret == NULL)
    {
        return NULL;
    }
    for (i = 0; i < isize; i++)
    {
        dret[i] = dvalue;
    }
    return dret;
}


INTG createspdd(INTG inodiags, INTG * ldiagelems, FLPT * ddiagvals)
{
    if ((inodiags < 1) || (inodiags % 2 == 0))
    {
        return 0; /* inodiags has to be odd to create spd matrices. */
    }
    
    INTG i;
    INTG imidpoint = inodiags / 2;
    for (i = 0; i < inodiags; i++)
    {
        ldiagelems[i] = i - imidpoint;
        if (i == imidpoint)
        {
            ddiagvals[i] = 2.0 * imidpoint;
        }
        else
        {
            ddiagvals[i] = -1.0;          
        }
    }  
    return 1; /* Success! */
}
    

int main(int argc, char *argv[])
{
    
/* 
// There are two arguments for the program. The first is the size of
// matrices used for testing. The second is the number of repetitions
// of matrix multiplication. Both these arguments are necessary, and 
// there are also lower bounds on acceptable values. The following 
// code does validation on this.
*/    

    const INTG iminmatsize = MINDIAGT27;
    
    if (argc < 3)
    {
        printf("To execute this, type:\n\nudcs n m\n\nWhere:\nn (>= ");
        printf("%d) ", iminmatsize);
        printf("is the size of the matrices to be multiplied and tested;");
        printf("\nm (>= 1) is the number of repetitions.\n\n");
        return(0);
    }
    const INTG imatsize = atoi(argv[1]);
    if (imatsize < iminmatsize)
    {
        printf("Please pass a matrix size greater or equal to %d.\n",
            iminmatsize);
        return(0);
    }
    const INTG inoreps = atoi(argv[2]);
    if (inoreps < 1)
    {
        printf("Please pass a number of repetitions greater or equal to 1.\n");
        return(0);
    }    

/* 
// Some useful variables:
// - i, j: general purpose iteration variables.
// - start, end: contains start and end times.
// - dvector: a test vector; consists solely of 1.0s.
*/
    
    INTG i, j;
    struct timespec start, end;
    FLPT * dvector = dsetvector(imatsize, 1.0);
    if (dvector == NULL)
    {
        printf("The function is unable to allocate a simple vector.\n"); 
        return (0);
    }

    
/* Now we initialise the test data.*/
    
    const INTG inotests = NUMTESTS;

/* Code for 3 diagonal UCDS. */
    
    const INTG inumsimpdiag = SMALLDIAG;
    INTG lsimpelems[SMALLDIAG] = {-1, 0, 1};
    
/* For simple identity matrices. */    
    
    FLPT delems_id[SMALLDIAG] = {0.0, 1.0, 0.0};  
    
/* For simple PDEs. */    
    
    FLPT delems_pde[SMALLDIAG] = {-1.0, 4.0, -1.0};     
    
/* Code for 5 diagonal UCDS. */

    const INTG inummiddiag = MIDDIAG;
    INTG lmidelems[MIDDIAG] = {-3, -1, 0, 1, 3};
    FLPT dmidel_vals[MIDDIAG] = {-1.0, -1.0, 4.0, -1.0, -1.0};    

/* Code for 27 diagonal UCDS. */
    
    const INTG inumlargdiag = LARGEDIAG;
    INTG llargelems[LARGEDIAG];
    FLPT dlargel_vals[LARGEDIAG];
    i = createspdd(inumlargdiag, llargelems, dlargel_vals);

    
/* 
// Code for other UCDSs with different numbers of diagonals. 
// The numbers are chosen (9, 15, 45, 81) because they are
// useful to see how UCDS performs for larger problems, rather
// than how they do with EPDEs.
*/

    const INTG inum9 = 9;
    INTG l9elems[inum9];
    FLPT d9vals[inum9];
    i = createspdd(inum9, l9elems, d9vals);    
    const INTG inum15 = 15;
    INTG l15elems[inum15];
    FLPT d15vals[inum15];
    i = createspdd(inum15, l15elems, d15vals);    
    const INTG inum45 = 45;
    INTG l45elems[inum45];
    FLPT d45vals[inum45];
    i = createspdd(inum45, l45elems, d45vals);    
    const INTG inum81 = 81;
    INTG l81elems[inum81];
    FLPT d81vals[inum81];
    i = createspdd(inum81, l81elems, d81vals);    
  
    
    
    
    
    
/* The test bed itself. */    
    
    mmtestbed ourtestbed[inotests];
    for (i = 0; i < inotests; i++)
    {
        ourtestbed[i].dret = (FLPT *) malloc (imatsize * sizeof(FLPT));
        if (i == 7)
        {
            ourtestbed[i].thefp = &multiply_ucds27;
        }
        else if (i == 3)
        {
            ourtestbed[i].thefp = &multiply_ucds5;
        }
        else
        {
            ourtestbed[i].thefp = &multiply_ucds;
        }

/* 
// Then we set the related objects - the number of diagonals, their indices
// and their values.
*/        

        switch(i)
        {
            case 0: /* Identity matrix */
            case 1: /* -1, 4, 1 stencil */
                ourtestbed[i].lnumdiag = inumsimpdiag;
                ourtestbed[i].ldiagindices = lsimpelems;
                if (i == 0)
                {
                    ourtestbed[i].ddiagelems = delems_id;
                }
                else
                {
                    ourtestbed[i].ddiagelems = delems_pde;
                }
                break;
            case 2: /* -1, -1, 4, 1, 1, stencil */
            case 3: /* Ditto with multiply_ucds5 function. */
                ourtestbed[i].lnumdiag = inummiddiag;
                ourtestbed[i].ldiagindices = lmidelems;
                ourtestbed[i].ddiagelems = dmidel_vals;
                break;
            case 4: /* 9 diagonal matrix. */
                ourtestbed[i].lnumdiag = inum9;
                ourtestbed[i].ldiagindices = l9elems;
                ourtestbed[i].ddiagelems = d9vals;
                break;
            case 5: /* 15 diagonal matrix. */
                ourtestbed[i].lnumdiag = inum15;
                ourtestbed[i].ldiagindices = l15elems;
                ourtestbed[i].ddiagelems = d15vals;
                break;                
            case 6: /* 27 diagonal matrix. */
            case 7: /* Ditto with multiply_ucds27 */
                ourtestbed[i].lnumdiag = inumlargdiag;
                ourtestbed[i].ldiagindices = llargelems;
                ourtestbed[i].ddiagelems = dlargel_vals;
                break;
            case 8: /* 45 diagonal matrix. */
                ourtestbed[i].lnumdiag = inum45;
                ourtestbed[i].ldiagindices = l45elems;
                ourtestbed[i].ddiagelems = d45vals;
                break;                 
            default: /* 81 diagonal matrix. */    
                ourtestbed[i].lnumdiag = inum81;
                ourtestbed[i].ldiagindices = l81elems;
                ourtestbed[i].ddiagelems = d81vals;
                break;             
        }

/* Then we initialise the UCDS object. */
        
        ourtestbed[i].ourucds = mmatrix_ucds(imatsize, 
            ourtestbed[i].ldiagindices, ourtestbed[i].ddiagelems, 
            ourtestbed[i].lnumdiag);
    }

/* Now we run the tests. */
    
    for (i = 0; i < inotests; i++)
    {
        clock_gettime(CLOCK_MONOTONIC, &start); 
        for (j = 0; j < inoreps; j++)
        {
            ourtestbed[i].dret = (* ourtestbed[i].thefp)(ourtestbed[i].ourucds,
            dvector, ourtestbed[i].dret);
        } 
        clock_gettime(CLOCK_MONOTONIC, &end);
        ourtestbed[i].testlen = timespecDiff(&end, &start);
    }
    

/* Then we print the tests. */    

    for (i = 0; i < inotests; i++)
    {
        printf("%f, ", (FLPT)((1.0 * TLPERS * imatsize * inoreps * 
            ourtestbed[i].lnumdiag)/(1000000.0 * ourtestbed[i].testlen)));
    }
    printf("%d\n", imatsize);
    
/* The last state is to free up all the memory used. */
    
    free(dvector); 
    for (i = 0; i < inotests; i++)
    {
        free(ourtestbed[i].dret);
        destroy_ucds(ourtestbed[i].ourucds);
    }
    return 0;
}
