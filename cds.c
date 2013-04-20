/* 
// cds.c. Implementation of Compressed Diagonal Storage.
// Written by Peter Murphy. (c) 2013
*/

#include <stdio.h>
#include <stdlib.h>

/* Quick and dirty min and max. */

#define max(x, y) (((x) > (y)) ? (x) : (y))
#define min(x, y) (((x) < (y)) ? (x) : (y))

/* 
// The cds structure is straight out of "Templates for the Solutions of 
// Iterative Systems" [Barrett, et. al., 1994, p. 59 to 61]. It is used
// for banded square matrices, where non-zero elements are limited to
// certain diagonals in the matrix, and the diagonals are limited to a
// contiguous band. 
//
// For each element A[i][j] in A, we assign the diagonal index to j - i.
// Then the cds structure has the following members:
// - lmatsize: the size of the square matrix represented.
// - lstart: the minimum diagonal index represented.
// - lend: the maximum diagonal index represented.
// - delements: a two dimensional array representing the elements.
// 
//
// Example: the following matrix: 
//
//                                [ 1 2 ]
//                                [ 3 4 ] 
//
// Could be represented as three diagonals (where N means not important)
//
//                                [ 3 N ]
//                                [ 1 4 ]
//                                [ N 2 ] 
*/

typedef struct  {
    long lmatsize; 
    long lstart; 
    long lend; 
    double **delements; 
} cds;


/* 
// The create_cds function creates and allocates an instance of the cds 
// structure using the following arguments:
//
// - lmatsize: the size of the square matrix represented.
// - lstart: the minimum diagonal index represented.
// - lend: the maximum diagonal index represented.
//
// If successful, a cds* is returned; otherwise, the function returns NULL.
//
// Note: the user has to initialise the values themselves.
*/

cds* create_cds(long lmatsize, long lstart, long lend)
{
    if ((lmatsize < 1) || (lstart < (1 - lmatsize)) || (lstart > lend) 
        || ((lmatsize - 1) < lend)) 
    {
        return NULL;
    }
    cds* ourcds = (cds *)malloc(1 * sizeof(cds));
    ourcds->lmatsize = lmatsize;
    ourcds->lstart = lstart;
    ourcds->lend = lend;
    unsigned int ulnumdiag = lend - lstart + 1;
    ourcds->delements = (double **)malloc(ulnumdiag * sizeof(double*));
    int i;
    for (i = 0; i < ulnumdiag; i++)
    {
        ourcds->delements[i] = (double *)malloc(lmatsize * sizeof(double));
    }
    return ourcds;
}

/* The destroy_cds function deallocates and destroys a cds instance. */

void destroy_cds(cds * ourcds)
{
    int i;
    unsigned int ulnumdiag = ourcds->lend - ourcds->lstart + 1;
    for (i = 0; i < ulnumdiag; i++)
    {
        free(ourcds->delements[i]);
    }
    free(ourcds->delements); 
    free(ourcds);
}

/* 
// The multiply_cds performs matrix vector multiplication using the cds type.
// The arguments are:
//
// - ourcds: a pointer to a cds instance.
// - dvector: a vector that can be passed in as a double[] type. Note that 
// the programmer should make sure that there is at least ourcds->lmatsize
// instances.
//
// If successful, the function creates and allocates a vector that is the
// result of the desired matrix multiplication. Otherwise, it returns 
// NULL.
*/


double * multiply_cds(cds *ourcds, double *dvector)
{
    double * dret = (double *) calloc (ourcds->lmatsize,sizeof(double));
    int i, j;
    int miniter, maxiter;
    for (i = ourcds->lstart; i <= ourcds->lend; i++)
    {
        miniter = max(0, i);
        maxiter = min(ourcds->lmatsize - 1, ourcds->lmatsize - 1 + i);
        for (j = miniter; j <= maxiter; j++)
        {
            dret[j - i] += ourcds->delements[i - ourcds->lstart][j] * 
                dvector[j];
        }
    }
    return dret;
}

/* 
// mmatrix_cds generates a "sample" M-matrix in cds form. A M-matrix is 
// a matrix which is strictly diagonally dominant, but all off-diagonal 
// entries are negative or zero. The arguments are:
//
// - lmatsize: the size of the square matrix represented.
// - lstart: the minimum diagonal index represented.
// - lend: the maximum diagonal index represented.
// - maindiagelem: the value to set all elements on the center diagonal. 
// This should be positive.
// - offdiagelement: the value to set all elements off the main diagonal.
// This should be negative.
//
// If successful, a cds* is returned; otherwise, the function returns NULL.
*/


cds * mmatrix_cds(long lmatsize, long lstart, long lend, double maindiagelem,
    double offdiagelem)
{
    
/* 
// Before allocating space for the cds, we have to check that parameters 
// make a M-matrix creation possible.
*/
    
    if ((lstart > 0) || (lend < 0))
    {
        return NULL;
    }
    if ((maindiagelem < 0.0) || (offdiagelem > 0.0))
    {
        return NULL;
    }
    
/* This checks diagonal dominance. */

    if (maindiagelem < ((lstart - lend) * offdiagelem))
    {
        return NULL;
    }
    
/* Now we can allocate the cds. */    
    
    cds * ourcds = create_cds(lmatsize, lstart, lend);
    int i, j;
    for (i = ourcds->lstart; i <= ourcds->lend; i++)
    {
        int irealindex = i - ourcds->lstart;
        for (j = 0; j < ourcds->lmatsize; j++)
        {
            if (i == 0)
            {
               ourcds->delements[irealindex][j] =  maindiagelem;
            }
            else
            {
               ourcds->delements[irealindex][j] =  offdiagelem; 
            }
        }
    }
    return ourcds;
}
    
/* 
// The ucds ("Ultra Compressed Diagonal Storage") scheme is a modification
// of the cds scheme defined above. Rather than store diagonals as a 
// contiguous band, store individual diagonals with their diagonal index.
// The type is more suitable for the matrices created by 3D EPDEs (or their
// discretisation thereof).
// 
// For each element A[i][j] in A, we assign it to the diagonal with an index
// (j - i). We then use the ucdsdiag structure to represent diagonals, where
// - ldiagindex: the index of the diagonal.
// - ddiagelems: the elements in the diagonal.
//
// Then the cds structure has the following members:
// - lmatsize: the size of the square matrix represented.
// - lnumdiag: the number of diagonals represented.
// - ucdsdelem: the diagonal entries.
//
// Example: the following matrix: 
//
//                                [ 1 2 ]
//                                [ 3 4 ] 
//
// Could be represented as three diagonals (where N means not important)
//
//                               {ldiagindex = -1, ddiagelems = [ 3 N ],
//                                ldiagindex = 0, ddiagelems = [ 1 4 ],
//                                ldiagindex = 1, ddiagelems = [ N 2 ]}. 
*/    


typedef struct {
    long ldiagindex;
    double *ddiagelems;
} ucdsdiag;

typedef struct {
    long lmatsize; 
    long lnumdiag;
    ucdsdiag *ucdsdelem;
} ucds;

/* 
// The create_ucds function creates and allocates an instance of the ucds 
// structure using the following arguments:
//
// - lmatsize: the size of the square matrix represented.
// - ldiagindices: an array of diagonal indices (in ascending order with
//   no repeated values).
// - lindicessize: the size of ldiagindices.
//
// If successful, a ucds* is returned; otherwise, the function returns NULL.
//
// Note 1: the user has to initialise the values themselves.
// Note 2: unpredicatable behaviour may result if ldiagindices is unsorted
// or contains repeated values.
*/

ucds* create_ucds(long lmatsize, long * ldiagindices, long lnumdiag)
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
    ourucds->ucdsdelem = (ucdsdiag *)malloc(lnumdiag * sizeof(ucdsdiag));
    int i;
    for (i = 0; i < lnumdiag; i++)
    {
        ourucds->ucdsdelem[i].ldiagindex = ldiagindices[i];
        ourucds->ucdsdelem[i].ddiagelems = (double *)malloc(lmatsize 
            * sizeof(double));
    }
    return ourucds;
}

/* The destroy_ucds function deallocates and destroys a ucds instance. */

void destroy_ucds(ucds * ourucds)
{
    int i;
    for (i = 0; i < ourucds->lnumdiag; i++)
    {
        free(ourucds->ucdsdelem[i].ddiagelems);
    }
    free(ourucds->ucdsdelem); 
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

double * multiply_ucds(ucds *ourucds, double *dvector)
{
    double * dret = (double *) calloc (ourucds->lmatsize,sizeof(double));
    int i, j;
    int miniter, maxiter;
    long lrevindex;
    for (i = 0; i < ourucds->lnumdiag; i++)
    {
        lrevindex = ourucds->ucdsdelem[i].ldiagindex;
        miniter = max(0, lrevindex);
        maxiter = min(ourucds->lmatsize - 1, ourucds->lmatsize - 1 + lrevindex);
        for (j = miniter; j <= maxiter; j++)
        {
            dret[j - lrevindex] += ourucds->ucdsdelem[i].ddiagelems[j] * 
                dvector[j];
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


ucds * mmatrix_ucds(long lmatsize, long * ldiagindices, double *
    ddiagvals, long lnumdiag)
{
    
/* 
// Before allocating space for the cds, we have to check that parameters 
// make a M-matrix creation possible.
*/
    
    if ((ldiagindices[0] > 0) || (ldiagindices[lmatsize - 1] < 0))
    {
        return NULL;
    }
    
/* This checks diagonal dominance. */    
    
    double dsum = 0.0;
    int i, j;
    for (i = 0; i < lmatsize; i++)
    {
        dsum += ddiagvals[i];
    }
    if (dsum < 0.0)
    {
        return NULL;
    }
    
/* Now we can allocate the cds. */    
    
    ucds * ourucds = create_ucds(lmatsize, ldiagindices, lnumdiag);
    for (i = 0; i < lnumdiag; i++)
    {
        for (j = 0; j < lmatsize; j++)
        {
            ourucds->ucdsdelem[i].ddiagelems[j] = ddiagvals[i];
        }
    }
    return ourucds;
}


int main()
{

/*
// This code will test the structures and the functions written
// for CDS (and UCDS). The first test is to represent the following
// matrix in CDS form:
//
//    [ 1 2 ]
//    [ 3 4 ] 
// 
// (Note: "T" as used below means traspose.
*/    
    
  cds * ourcds = create_cds(2, -1, 1);
  ourcds->delements[0][0] = 3.0;
  ourcds->delements[1][0] = 1.0;
  ourcds->delements[1][1] = 4.0;
  ourcds->delements[2][1] = 2.0;
    
/*
// And multiply it with the vector [5 6]T. (The result should be
// [17 39]T.
*/    
    
    
  double * dvector = (double *)malloc(2 * sizeof (double));
  dvector[0] = 5.0;
  dvector[1] = 6.0;
  double * dret = multiply_cds(ourcds, dvector);
  printf("Our first vector is [%f, %f].\n", dret[0], dret[1]);
  
/* 
// The next step is to use the mmatrix_cds function to set up the
// 2 by 2 identity matrix, and multiply it with [5 6]T.
// (The result should of course be [5 6]T.
*/
  
  
  cds * ouridentitycds = mmatrix_cds(2, 0, 0, 1.0, 0.0);
  double * dsame = multiply_cds(ouridentitycds, dvector);
  printf("Our second vector is [%f, %f].\n", dsame[0], dsame[1]); 

/*
// For the third stage, we use mmatrix_cds to set up this matrix:
//
//    [  4 -1  ]
//    [ -1  4  ] 
// 
// We multiply it with [5 6]T. The result should be [14 19]T.
*/  
  
  cds * ourpdecds = mmatrix_cds(2, -1, 1, 4.0, -1.0);
  double * dpde = multiply_cds(ourpdecds, dvector);
  printf("Our third vector is [%f, %f].\n", dpde[0], dpde[1]); 
  
/* The fourth stage is the same as the first, except that UCDS is used. */  
  
  long lelems[3] = {-1, 0, 1};
  ucds * ourucds = create_ucds(2, lelems, 3);
  ourucds->ucdsdelem[0].ddiagelems[0] = 3.0;
  ourucds->ucdsdelem[1].ddiagelems[0] = 1.0;
  ourucds->ucdsdelem[1].ddiagelems[1] = 4.0;
  ourucds->ucdsdelem[2].ddiagelems[1] = 2.0;  
  double * dret2 = multiply_ucds(ourucds, dvector);
  printf("Our fourth vector is [%f, %f].\n", dret2[0], dret2[1]); 
  
/* The fifth stage is to set up a 2 by 2 matrix using UCDS. */

  double delems_id[3] = {0.0, 1.0, 0.0};  
  ucds * ouriducds = mmatrix_ucds(2, lelems, delems_id, 3);  
  double * dret3 = multiply_ucds(ouriducds, dvector);
  printf("Our fifth vector is [%f, %f].\n", dret3[0], dret3[1]);  

/* And the sixth stage is like the fourth, except for using UCDS. */

  double delems_pde[3] = {-1.0, 4.0, -1.0};  
  ucds * ourpdeucds = mmatrix_ucds(2, lelems, delems_pde, 3);  
  double * dret4 = multiply_ucds(ourpdeucds, dvector);
  printf("Our sixth vector is [%f, %f].\n", dret4[0], dret4[1]);  
  
/* The sevent stage is to free up all the memory used. */  
  
  free(dret2);
  free(dret3);
  free(dret4);  
  destroy_ucds(ourucds);  
  destroy_ucds(ouriducds);  
  destroy_ucds(ourpdeucds);  
  free(dpde);
  free(dsame);
  free(dret);
  free(dvector);  
  destroy_cds(ourpdecds); 
  destroy_cds(ouridentitycds);   
  destroy_cds(ourcds); 
  return 0;
}
