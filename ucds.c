/* 
// ucds.c. Implementation of Ultra Compressed Diagonal Storage.
// Written by Peter Murphy. (c) 2013
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <omp.h>
#include "ucds.h"

/* Function implementations. */

FLPT * dassign(const INTG isize)
{
    return (FLPT *)malloc(isize * sizeof(FLPT));
}


FLPT ddotprod (const INTG lvectsize, const FLPT * dleftvec, 
    const FLPT * drightvec)
{
    INTG i; /* An iteration variable. */
    FLPT dresult = 0.0; /* The result. */
    #pragma omp parallel for reduction(+:dresult)
    for (i = 0; i < lvectsize; i++)
    {
        dresult += (dleftvec[i] * drightvec[i]);
    }
    return dresult;
}


FLPT * dscalarprod (const INTG lvectsize, const FLPT dscalar, 
    const FLPT * dvectin, FLPT * dvectout)
{
    INTG i; /* An iteration variable. */
    #pragma omp parallel for
    for (i = 0; i < lvectsize; i++)
    {
        dvectout[i] = dscalar * dvectin[i];
    }
    return dvectout;
}


FLPT * dvectadd (const INTG lvectsize, const FLPT * dleftvec, 
    const FLPT * drightvec, FLPT * dvectout)
{
    INTG i; /* An iteration variable. */
    #pragma omp parallel for
    for (i = 0; i < lvectsize; i++)
    {
        dvectout[i] = dleftvec[i] + drightvec[i];
    }
    return dvectout;
}


FLPT * dvectsub (const INTG lvectsize, const FLPT * dleftvec, 
    const FLPT * drightvec, FLPT * dvectout)
{
    INTG i; /* An iteration variable. */
    #pragma omp parallel for
    for (i = 0; i < lvectsize; i++)
    {
        dvectout[i] = dleftvec[i] - drightvec[i];
    }
    return dvectout;
}


FLPT dvectnorm (const INTG lvectsize, const INTG mode, 
    const FLPT * dvectin)
{
    INTG i; /* An iteration variable. */
    FLPT dresult = 0.0; /* The result. */
    if (mode == 1)
    {
        #pragma omp parallel for reduction(+:dresult)
        for (i = 0; i < lvectsize; i++)
        {
            dresult += fabs(dvectin[i]);
        }
    }
    else if (mode == 2)
    {
        #pragma omp parallel for reduction(+:dresult)
        for (i = 0; i < lvectsize; i++)
        {
            dresult += pow(dvectin[i], 2);
        }
        dresult = sqrt(dresult);
    }
    else /* Infinity mode */
    {
        #pragma omp parallel for 
        for (i = 0; i < lvectsize; i++)
        {
            #pragma omp critical
            if (dresult < dvectin[i])
            {
                dresult = dvectin[i];
            }
        }
    }        
    return dresult;
}

FLPT daltnorm (const INTG lvectsize, const INTG mode, 
    const FLPT * dvectin)
{
    FLPT dresult = 0.0; /* The result. */
    if (lvectsize == 1)
    {
        return fabs(dvectin[0]);
    }
    INTG lhalf = lvectsize / 2;
    INTG lremd = lvectsize - lhalf;
    FLPT fbr1;
    FLPT fbr2;
    FLPT * dsndptr = &(dvectin[lhalf]);
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            fbr1 = daltnorm(lhalf, mode, dvectin);
        } 
        #pragma omp section
        {
            fbr2 = daltnorm(lremd, mode, dsndptr);
        } 
    }    
    if (mode == 1)
    {
        dresult = fbr1 + fbr2;
    }
    else if (mode == 2)
    {
        dresult = sqrt(pow(fbr1,2) + pow(fbr2, 2));
    }
    else /* Infinity mode */
    {
        dresult = max(fbr1, fbr2);
    }        
    return dresult;
}

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
    ourucds->ddiagelems = dassign(lnumdiag * lmatsize);
    return ourucds;
}

void destroy_ucds(ucds * ourucds)
{
    free(ourucds->ddiagelems); 
    free(ourucds);
}


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
    
    #pragma omp parallel for private(lrevindex, miniter, maxiter, j)
    for (i = 0; i < ourucds->lnumdiag; i++)
    {
        lrevindex = ourucds->ldiagindices[i];
        miniter = max(0, lrevindex);
        maxiter = min(ourucds->lmatsize - 1, ourucds->lmatsize - 1 
            + lrevindex);
        //#pragma omp parallel for
        for (j = miniter; j <= maxiter; j++)
        {
            dret[j - lrevindex] += ourucds->ddiagelems[i*ourucds->lmatsize 
                + j] * dvector[j];
        }
    }
    return dret; 
}

FLPT * multiply_ucdsalt(const ucds *ourucds, const FLPT *dvector, FLPT * dret)
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
    
    //##pragma omp parallel for private(lrevindex, miniter, maxiter, j)
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
    
    #pragma omp parallel for private(lrevindex, miniter, maxiter, j)
    for (i = 0; i < idiagnum; i++)
    {
        lrevindex = ourucds->ldiagindices[i];
        miniter = max(0, lrevindex);
        maxiter = min(ourucds->lmatsize - 1, ourucds->lmatsize - 1 
            + lrevindex);
        //#pragma omp parallel for
        for (j = miniter; j <= maxiter; j++)
        {
            dret[j - lrevindex] += ourucds->ddiagelems[i*ourucds->lmatsize 
                + j] * dvector[j];
        }
    }
    return dret; 
}


FLPT * multiply_ucdsalt27(const ucds *ourucds, const FLPT *dvector, FLPT * dret)
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
    
    //#pragma omp parallel for private(lrevindex, miniter, maxiter, j)
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
    
    #pragma omp parallel for private(lrevindex, miniter, maxiter, j)
    for (i = 0; i < idiagnum; i++)
    {
        lrevindex = ourucds->ldiagindices[i];
        miniter = max(0, lrevindex);
        maxiter = min(ourucds->lmatsize - 1, ourucds->lmatsize - 1 
            + lrevindex);
        //#pragma omp parallel for
        for (j = miniter; j <= maxiter; j++)
        {
            dret[j - lrevindex] += ourucds->ddiagelems[i*ourucds->lmatsize 
                + j] * dvector[j];
        }
    }
    return dret; 
}

FLPT * multiply_ucdsalt5(const ucds *ourucds, const FLPT *dvector, FLPT * dret)
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
    
    //#pragma omp parallel for private(lrevindex, miniter, maxiter, j)
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

FLPT * dsetvector(const INTG isize, const FLPT dvalue)
{
    FLPT * dret = dassign(isize); /* Return value. */
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

/*
// Constructs doverwrite = doverright + dconst * drightvec.
*/

FLPT * dvecoverwrite (const INTG lvectsize, FLPT * doverwrite, 
    const FLPT dconst, const FLPT * drightvec)
{
    INTG i; /* An iteration variable. */
    #pragma omp parallel for
    for (i = 0; i < lvectsize; i++)
    {
        doverwrite[i] += dconst * drightvec[i];
    }
    return doverwrite;
}

FLPT * dveccopy (const INTG lvectsize, FLPT * doverwrite, 
    const FLPT *dsource)
{
    INTG i; /* An iteration variable. */
    #pragma omp parallel for
    for (i = 0; i < lvectsize; i++)
    {
        doverwrite[i] = dsource[i];
    }
    return doverwrite;
}


FLPT * dvecadjust (const INTG lvectsize, FLPT * dadjust, 
    const FLPT *dleftvec, const FLPT dleftconst, const FLPT * drightvec, const FLPT drightconst)
{
    INTG i; /* An iteration variable. */
    #pragma omp parallel for
    for (i = 0; i < lvectsize; i++)
    {
        dadjust[i] = (dleftvec[i] * dleftconst) + (drightvec[i] * drightconst);
    }
    return dadjust;
}

void printvector(char* name, INTG isize, const FLPT* fvector)
{
    printf("%s: [", name);
    INTG i;
    for (i = 0; i < (isize - 1); i++)
    {
        printf("%f, ", fvector[i]);
    }
    printf("%f]\n", fvector[isize - 1]);
}


FLPT * conjgrad(const ucds * ucdsa, const FLPT * dvectb, const FLPT *dvectx0,
    FLPT * dvectx, const INTG imode, const INTG itype, const FLPT derror, INTG * iiter)
{
    INTG icount = 0; /* The iteration count. */
    INTG ivectorsize = ucdsa->lmatsize; /* The size of all matrices and vectors. */
    FLPT * drvect[2]; /* Stores r_even and r_odd vectors. */
    FLPT drdotpd[2]; /* Stores dot products of same. */
    FLPT * dpvect = dassign(ivectorsize); /* Stores p vector. */
    FLPT * dapproduct = dassign(ivectorsize); /* Stores products of A and p_k. */
    FLPT dalpha, dbeta; /* Alpha and beta. */
    
/* Initialise the algorithm. */

    drvect[0] = dassign(ivectorsize);
    drvect[1] = dassign(ivectorsize);
    printvector("b", ivectorsize, dvectb);
    printvector("x0", ivectorsize, dvectx0);
    dapproduct = multiply_ucds(ucdsa, dvectx0, dapproduct);
    printvector("a.p", ivectorsize, dapproduct);
    drvect[0] = dvectsub (ivectorsize, dvectb, dapproduct, drvect[0]);
    printvector("r", ivectorsize, drvect[0]);
    dveccopy(ivectorsize, dpvect, drvect[0]);
    dveccopy(ivectorsize, dvectx, dvectx0);
    printvector("p", ivectorsize, dpvect);
    drdotpd[icount % 2] = ddotprod(ivectorsize, drvect[icount % 2], drvect[icount % 2]);
    while (1)
    {
        dapproduct = multiply_ucds(ucdsa, dpvect, dapproduct);
        printvector("a.p", ivectorsize, dapproduct);
        dalpha = drdotpd[icount % 2] / ddotprod(ivectorsize, dpvect, dapproduct);
        printf("alpha: %f\n", dalpha);
        printvector("x", ivectorsize, dvectx);
        dvecoverwrite (ivectorsize, dvectx, dalpha, dpvect);
        printvector("x1", ivectorsize, dvectx);
        drvect[(icount + 1) % 2] = dvecadjust (ivectorsize, drvect[(icount + 1) % 2], 
            drvect[icount % 2], 1.0, dapproduct, dalpha * (-1.0));
        printvector("r", ivectorsize, drvect[(icount + 1) % 2]);
        if (itype == 0)
        {
            printf("%d\n", dvectnorm(ivectorsize, imode, drvect[(icount + 1) % 2]));
            if (dvectnorm(ivectorsize, imode, drvect[(icount + 1) % 2]) < derror)
            {
                break;
            }
        }
        else
        {
            if (daltnorm(ivectorsize, imode, drvect[(icount + 1) % 2]) < derror)
            {
                break;
            }
        }
        drdotpd[(icount % 2) + 1] = ddotprod(ivectorsize, drvect[(icount % 2) + 1], drvect[(icount % 2) + 1]);
        dbeta = ddotprod(ivectorsize, drvect[(icount + 1) % 2], drvect[(icount + 1) % 2])
             / ddotprod(ivectorsize, drvect[icount % 2], drvect[icount % 2]);
        printf("beta: %f\n", dbeta);
        dpvect = dvecadjust (ivectorsize, dpvect, drvect[(icount + 1) % 2], 1.0, 
            dpvect, dbeta); 
        printvector("p", ivectorsize, dpvect);        
        icount = icount + 1;
    }
    printf("D");
    if (iiter != NULL)
    {
        *iiter = icount;
    }
    free(dpvect);
    free(dapproduct);
    free (drvect[0]);
    free (drvect[1]);
    printvector("x", ivectorsize, dvectx);
    return dvectx;
}

void testconjgrad()
{
    printf("Before");    
    INTG ldiagindices[3] = {-1, 0, 1};
    FLPT ddiagvals[3] = {1.0, 4.0, 1.0};    
    ucds * ucdsa = mmatrix_ucds(2, ldiagindices, ddiagvals, 3);
    ucdsa->ddiagelems[3] = 3.0;
 //   printf("%f", ucdsa->ddiagelems[0]);
    FLPT * vectb = dassign(2);
    vectb[0] = 1.0;
    vectb[1] = 2.0;
    FLPT * vectx0 = dassign(2);
    vectx0[1] = 1.0;
    vectx0[0] = 2.0;
    FLPT * vectx = dassign(2);
    printf("Before");
    conjgrad(ucdsa, vectb, vectx0, vectx, 1, 0, 0.1, NULL);
    free(vectb);
    free(vectx);
    free(vectx0);
    destroy_ucds(ucdsa);
}
    
 


int main(int argc, char *argv[])
{
    INTG tldiagindices[3] = {-1, 0, 1};
    FLPT tddiagvals[3] = {1.0, 4.0, 1.0};    
    ucds * ucdsa = mmatrix_ucds(2, tldiagindices, tddiagvals, 3);
    ucdsa->ddiagelems[3] = 3.0;
    printvector("ucds", 6, ucdsa->ddiagelems);
    FLPT * vectb = dassign(2);
    vectb[0] = 1.0;
    vectb[1] = 2.0;
    FLPT * vectx0 = dassign(2);
    vectx0[1] = 1.0;
    vectx0[0] = 2.0;
    FLPT * vectx = dassign(2);
    conjgrad(ucdsa, vectb, vectx0, vectx, 2, 0, 0.1, NULL);
    free(vectb);
    free(vectx);
    free(vectx0);
    destroy_ucds(ucdsa);    
    
    
  //  testconjgrad();
    
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
        ourtestbed[i].dret = dassign(imatsize);
        if (i == 7)
        {
            ourtestbed[i].thefp = &multiply_ucdsalt27;
        }
        else if (i == 3)
        {
            ourtestbed[i].thefp = &multiply_ucdsalt5;
        }
        else
        {
            ourtestbed[i].thefp = &multiply_ucdsalt;
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
 
    
/* 
// What we now do is time the subsidiary operations created for
// the conjugate gradient operation.
*/
    
    FLPT tdotprod;
    FLPT tscalprod;
    FLPT tvectadd;
    FLPT tvectsub;
    FLPT tnorm1;
    FLPT tnorm2;
    FLPT tnorminf;
    FLPT taltnorm1;
    FLPT taltnorm2;
    FLPT taltnorminf;

/* These are dummy variables for taking the outputs of functions. */

    FLPT ddummy;
    FLPT * dvectout = dsetvector(imatsize, 0.0);
    
/* Here are the tests. */    
    
    clock_gettime(CLOCK_MONOTONIC, &start); 
    for (j = 0; j < inoreps; j++)
    {
        ddummy = ddotprod (imatsize, dvector, dvector);
    } 
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf ("%f: ", ddummy);
    tdotprod = (1.0 * TLPERS * inoreps * imatsize) /
        (MEGAHERTZ * timespecDiff(&end, &start)); 

    
    clock_gettime(CLOCK_MONOTONIC, &start); 
    for (j = 0; j < inoreps; j++)
    {
        dvectout = dscalarprod (imatsize, 2.0, dvector, dvectout);
    } 
    clock_gettime(CLOCK_MONOTONIC, &end);
    tscalprod = (1.0 * TLPERS * inoreps * imatsize) /
        (MEGAHERTZ * timespecDiff(&end, &start));     
    

    clock_gettime(CLOCK_MONOTONIC, &start); 
    for (j = 0; j < inoreps; j++)
    {
        dvectout = dvectadd(imatsize, dvector, dvector, dvectout);
    } 
    clock_gettime(CLOCK_MONOTONIC, &end);
    tvectadd = (1.0 * TLPERS * inoreps * imatsize) /
        (MEGAHERTZ * timespecDiff(&end, &start));    
    

    clock_gettime(CLOCK_MONOTONIC, &start); 
    for (j = 0; j < inoreps; j++)
    {
        dvectout = dvectsub(imatsize, dvector, dvector, dvectout);
    } 
    clock_gettime(CLOCK_MONOTONIC, &end);
    tvectsub = (1.0 * TLPERS * inoreps * imatsize) /
        (MEGAHERTZ * timespecDiff(&end, &start));  

    clock_gettime(CLOCK_MONOTONIC, &start); 
    for (j = 0; j < inoreps; j++)
    {
        ddummy = dvectnorm(imatsize, 1, dvector);
    } 
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf ("%f: ", ddummy);
    tnorm1 = (1.0 * TLPERS * inoreps * imatsize) /
        (MEGAHERTZ * timespecDiff(&end, &start));     
    
    clock_gettime(CLOCK_MONOTONIC, &start); 
    for (j = 0; j < inoreps; j++)
    {
        ddummy = dvectnorm(imatsize, 2, dvector);
    } 
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf ("%f: ", ddummy);
    tnorm2 = (1.0 * TLPERS * inoreps * imatsize) /
        (MEGAHERTZ * timespecDiff(&end, &start));  

    clock_gettime(CLOCK_MONOTONIC, &start); 
    for (j = 0; j < inoreps; j++)
    {
        ddummy = dvectnorm(imatsize, 3, dvector);
    } 
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf ("%f: ", ddummy);
    tnorminf = (1.0 * TLPERS * inoreps * imatsize) /
        (MEGAHERTZ * timespecDiff(&end, &start));


    clock_gettime(CLOCK_MONOTONIC, &start); 
    for (j = 0; j < inoreps; j++)
    {
        ddummy = daltnorm(imatsize, 1, dvector);
    } 
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf ("%f: ", ddummy);
    taltnorm1 = (1.0 * TLPERS * inoreps * imatsize) /
        (MEGAHERTZ * timespecDiff(&end, &start));     
    
    clock_gettime(CLOCK_MONOTONIC, &start); 
    for (j = 0; j < inoreps; j++)
    {
        ddummy = daltnorm(imatsize, 2, dvector);
    } 
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf ("%f: ", ddummy);
    taltnorm2 = (1.0 * TLPERS * inoreps * imatsize) /
        (MEGAHERTZ * timespecDiff(&end, &start));  

    clock_gettime(CLOCK_MONOTONIC, &start); 
    for (j = 0; j < inoreps; j++)
    {
        ddummy = daltnorm(imatsize, 3, dvector);
    } 
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf ("%f: ", ddummy);
    taltnorminf = (1.0 * TLPERS * inoreps * imatsize) /
        (MEGAHERTZ * timespecDiff(&end, &start));



    

/* Then we print the tests. */    

    printf("%f, %f, %f, %f, %f, %f, %f, %f, %f, %f; ", tdotprod, tscalprod, tvectadd,
        tvectsub, tnorm1, tnorm2, tnorminf, taltnorm1, taltnorm2, taltnorminf);
    
    
    for (i = 0; i < inotests; i++)
    {
        printf("%f - ", (FLPT)((1.0 * TLPERS * imatsize * inoreps * 
            ourtestbed[i].lnumdiag)/(1000000.0 * ourtestbed[i].testlen)));
    }
    printf("%d\n", imatsize);
    
/* The last state is to free up all the memory used. */
    
    free(dvector);
    free(dvectout);    
    for (i = 0; i < inotests; i++)
    {
        free(ourtestbed[i].dret);
        destroy_ucds(ourtestbed[i].ourucds);
    }
    return 0;
}
