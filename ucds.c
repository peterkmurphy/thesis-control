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

FLPT dselfdprod(const INTG lvectsize, const FLPT * dvector)
{
    return ddotprod(lvectsize, dvector, dvector);
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

FLPT * daddinsitu (const INTG lvectsize, FLPT * doverwrite, 
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

FLPT * daddtwosums (const INTG lvectsize, FLPT * dadjust, 
    const FLPT *dleftvec, const FLPT dleftconst, 
    const FLPT * drightvec, const FLPT drightconst)
{
    INTG i; /* An iteration variable. */
    #pragma omp parallel for
    for (i = 0; i < lvectsize; i++)
    {
        dadjust[i] = (dleftvec[i] * dleftconst) + 
            (drightvec[i] * drightconst);
    }
    return dadjust;
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

void printvector(const char* name, INTG isize, const FLPT* dvector)
{
    printf("%s: [", name);
    INTG i;
    for (i = 0; i < (isize - 1); i++)
    {
        printf("%f, ", dvector[i]);
    }
    printf("%f]\n", dvector[isize - 1]);
}

FLPT * dconjgrad(const ucds * ucdsa, const FLPT * dvectb, const FLPT *dvectx0,
    FLPT * dvectx, fpmult fpucdsmult, fpnorm fpdnorm, const INTG imode, 
    const FLPT derror, INTG * iiter)
{
    INTG icount = 0; /* The iteration count. */
    INTG ivectorsize = ucdsa->lmatsize; /* The size of matrices and vectors. */
    FLPT * drvect[2]; /* Stores r_even and r_odd vectors. */
    FLPT drdotpd[2]; /* Stores dot products of same. */
    FLPT * dpvect = dassign(ivectorsize); /* Stores p vector. */
    FLPT * dapproduct = dassign(ivectorsize); /* For product of A and p_k. */
    FLPT dalpha, dbeta; /* Alpha and beta. */
    
/* Initialise the algorithm. */

    drvect[0] = dassign(ivectorsize);
    drvect[1] = dassign(ivectorsize);
    dapproduct = fpucdsmult(ucdsa, dvectx0, dapproduct);
    drvect[0] = dvectsub (ivectorsize, dvectb, dapproduct, drvect[0]);
    dveccopy(ivectorsize, dpvect, drvect[0]);
    dveccopy(ivectorsize, dvectx, dvectx0);
    drdotpd[icount % 2] = dselfdprod(ivectorsize, drvect[icount % 2]);
    while (1)
    {
        dapproduct = fpucdsmult(ucdsa, dpvect, dapproduct);
        dalpha = drdotpd[icount % 2] / ddotprod(ivectorsize, dpvect, 
            dapproduct);
        daddinsitu(ivectorsize, dvectx, dalpha, dpvect);
        drvect[(icount + 1) % 2] = daddtwosums(ivectorsize, 
            drvect[(icount + 1) % 2], drvect[icount % 2], 1.0, 
            dapproduct, dalpha * (-1.0));
        if (fpdnorm(ivectorsize, imode, drvect[(icount + 1) % 2]) < derror)
        {
            break;
        }        
        drdotpd[(icount % 2) + 1] = dselfdprod(ivectorsize, 
            drvect[(icount % 2) + 1]);
        dbeta = dselfdprod(ivectorsize, drvect[(icount + 1) % 2])
             / dselfdprod(ivectorsize, drvect[icount % 2]);
        dpvect = daddtwosums(ivectorsize, dpvect, drvect[(icount + 1) % 2], 
            1.0, dpvect, dbeta); 
        icount = icount + 1;
    }
    if (iiter != NULL)
    {
        *iiter = icount;
    }
    free(dpvect);
    free(dapproduct);
    free(drvect[0]);
    free(drvect[1]);
    return dvectx;
}



