/* 
// runucds.c. Measures the time it takes to multiply two matrices using
// Ultra Compressed Diagonal Storage; measures the time to execute other
// routines such as vector norms and scalar products. Attempts to measure
// the time taken per size of the input - preferably in MFP/s.
// Written by Peter Murphy. (c) 2013
*/

#include "projcommon.h"
#include "ucds.h"

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
        printf("To execute this, type:\n\n[exec] n m\n\nWhere:\nn (>= ");
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
    FLPT * didentvector = dsetvector(imatsize, 1.0); /* Has equal values. */
    FLPT * dvectout = dsetvector(imatsize, 0.0);
    if (didentvector == NULL)
    {
        printf("The function is unable to allocate a simple vector.\n"); 
        return (0);
    }

/* Now this is an attempt to set up a test environment. */

//    FLPT *dzerovector = dsetvector(imatsize, 0.0); 
    const INTG inotests = 20;
//    INTG icount;
    
/* The test bed itself. */    
    
    mmtestbed ourtestbed[inotests];
//    INTG iminisize = (imatsize * 2) - 1; /* Number of diagonals. */
    INTG immindices[20] = {5, 5, 5, 5, 5, 5, 9, 9, 15, 15, 27, 27, 27, 27, 27, 27, 
        45, 45, 81, 81};    
    
    for (i = 0; i < inotests; i++)
    {
    //    printf("Diag: %d\n", immindices[i]);
 //           printf("Diag: %d\n", immindices[i]);
            mmsetup(immindices[i], imatsize, &(ourtestbed[i]));
            if (i == 13)
            {
                ourtestbed[i].thefp = &multiply_ucdsalt27;
            }
            else if (i == 15)
            {
                ourtestbed[i].thefp = &multiply_ucdsaltd27;
            } 

            if (i == 12)
            {
                ourtestbed[i].thefp = &multiply_ucds27;
            }
            else if (i == 14)
            {
                ourtestbed[i].thefp = &multiply_ucdsd27;
            }  
            else if (i == 3)
            {
                ourtestbed[i].thefp = &multiply_ucdsalt5;
            }
            else if (i == 5)
            {
                ourtestbed[i].thefp = &multiply_ucdsaltd5;
            }
            else if (i == 2)
            {
                ourtestbed[i].thefp = &multiply_ucds5;
            }
            else if (i == 4)
            {
                ourtestbed[i].thefp = &multiply_ucdsd5;
            }            
            else if ((i % 2) == 1)
            { 
                ourtestbed[i].thefp = &multiply_ucdsalt;
            }
            else
            {
                ourtestbed[i].thefp = &multiply_ucds;
            }
    }        


/* Now we run the tests. */
    
    for (i = 0; i < inotests; i++)
    {
        clock_gettime(CLOCK_MONOTONIC, &start); 
        for (j = 0; j < inoreps; j++)
        {
            ourtestbed[i].dret = (* ourtestbed[i].thefp)(ourtestbed[i].ourucds,
            didentvector, ourtestbed[i].dret);
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
    
/* Here are the tests. */    
    
    clock_gettime(CLOCK_MONOTONIC, &start); 
    for (j = 0; j < inoreps; j++)
    {
        ddummy = ddotprod (imatsize, didentvector, didentvector);
    } 
    clock_gettime(CLOCK_MONOTONIC, &end);
    tdotprod = (1.0 * TLPERS * inoreps * imatsize) /
        (MEGAHERTZ * timespecDiff(&end, &start)); 

    
    clock_gettime(CLOCK_MONOTONIC, &start); 
    for (j = 0; j < inoreps; j++)
    {
        dvectout = dscalarprod (imatsize, 2.0, didentvector, dvectout);
    } 
    clock_gettime(CLOCK_MONOTONIC, &end);
    tscalprod = (1.0 * TLPERS * inoreps * imatsize) /
        (MEGAHERTZ * timespecDiff(&end, &start));     
    

    clock_gettime(CLOCK_MONOTONIC, &start); 
    for (j = 0; j < inoreps; j++)
    {
        dvectout = dvectadd(imatsize, didentvector, didentvector, dvectout);
    } 
    clock_gettime(CLOCK_MONOTONIC, &end);
    tvectadd = (1.0 * TLPERS * inoreps * imatsize) /
        (MEGAHERTZ * timespecDiff(&end, &start));    
    

    clock_gettime(CLOCK_MONOTONIC, &start); 
    for (j = 0; j < inoreps; j++)
    {
        dvectout = dvectsub(imatsize, didentvector, didentvector, dvectout);
    } 
    clock_gettime(CLOCK_MONOTONIC, &end);
    tvectsub = (1.0 * TLPERS * inoreps * imatsize) /
        (MEGAHERTZ * timespecDiff(&end, &start));  

    clock_gettime(CLOCK_MONOTONIC, &start); 
    for (j = 0; j < inoreps; j++)
    {
        ddummy = dvectnorm(imatsize, 1, didentvector);
    } 
    clock_gettime(CLOCK_MONOTONIC, &end);
    tnorm1 = (1.0 * TLPERS * inoreps * imatsize) /
        (MEGAHERTZ * timespecDiff(&end, &start));     
    
    clock_gettime(CLOCK_MONOTONIC, &start); 
    for (j = 0; j < inoreps; j++)
    {
        ddummy = dvectnorm(imatsize, 2, didentvector);
    } 
    clock_gettime(CLOCK_MONOTONIC, &end);
    tnorm2 = (1.0 * TLPERS * inoreps * imatsize) /
        (MEGAHERTZ * timespecDiff(&end, &start));  

    clock_gettime(CLOCK_MONOTONIC, &start); 
    for (j = 0; j < inoreps; j++)
    {
        ddummy = dvectnorm(imatsize, 3, didentvector);
    } 
    clock_gettime(CLOCK_MONOTONIC, &end);
    tnorminf = (1.0 * TLPERS * inoreps * imatsize) /
        (MEGAHERTZ * timespecDiff(&end, &start));


    clock_gettime(CLOCK_MONOTONIC, &start); 
    for (j = 0; j < inoreps; j++)
    {
        ddummy = daltnorm(imatsize, 1, didentvector);
    } 
    clock_gettime(CLOCK_MONOTONIC, &end);
    taltnorm1 = (1.0 * TLPERS * inoreps * imatsize) /
        (MEGAHERTZ * timespecDiff(&end, &start));     
    
    clock_gettime(CLOCK_MONOTONIC, &start); 
    for (j = 0; j < inoreps; j++)
    {
        ddummy = daltnorm(imatsize, 2, didentvector);
    } 
    clock_gettime(CLOCK_MONOTONIC, &end);
    taltnorm2 = (1.0 * TLPERS * inoreps * imatsize) /
        (MEGAHERTZ * timespecDiff(&end, &start));  

    clock_gettime(CLOCK_MONOTONIC, &start); 
    for (j = 0; j < inoreps; j++)
    {
        ddummy = daltnorm(imatsize, 3, didentvector);
    } 
    clock_gettime(CLOCK_MONOTONIC, &end);
    taltnorminf = (1.0 * TLPERS * inoreps * imatsize) /
        (MEGAHERTZ * timespecDiff(&end, &start));

/* Then we print the tests. */    

    printf("%f, %f, %f, %f, %f, %f, %f, %f, %f, %f; ", tdotprod, tscalprod, tvectadd,
        tvectsub, tnorm1, tnorm2, tnorminf, taltnorm1, taltnorm2, taltnorminf);
    
    
    for (i = 0; i < inotests; i++)
    {
      //  printf(" num %d, %d, %d, %ld\n", imatsize, inoreps, ourtestbed[i].lnumdiag, (TLPERS * imatsize * inoreps * 
        //    ourtestbed[i].lnumdiag));
      //  printf(" denum %f\n", (MEGAHERTZ * ourtestbed[i].testlen));
        printf("%f - ", (FLPT)((1.0 * TLPERS * imatsize * inoreps * 
            ourtestbed[i].lnumdiag)/(MEGAHERTZ * ourtestbed[i].testlen)));
    }
    printf("%d\n", imatsize);
    
/* The last state is to free up all the memory used. */
    
    free(didentvector);
    free(dvectout);    
    for (i = 0; i < inotests; i++)
    {
        mmdestroy(&(ourtestbed[i]));
    }
    (void)ddummy; // To disable warning: variable 'ddummy' set but not used/
    return 0;
}
