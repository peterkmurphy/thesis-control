/* 
// runconjgrad.c. Runs tests that measures the time taken to
// execute conjugate gradient.
// Written by Peter Murphy. (c) 2013
*/

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
    FLPT *didentvector = dsetvector(imatsize, 1.0);
    if (didentvector == NULL)
    {
        printf("The function is unable to allocate a simple vector.\n"); 
        return (0);
    }

/* Now this is an attempt to set up a test environment. */

    FLPT *dzerovector = dsetvector(imatsize, 0.0); 
    const INTG inotests = 20;
    INTG icount;    
    
   
/* 
// Code for other UCDSs with different numbers of diagonals. 
// The numbers are chosen (9, 15, 45, 81) because they are
// useful to see how UCDS performs for larger problems, rather
// than how they do with EPDEs.
*/
    
/* The test bed itself. */    
     
    mmtestbed ourtestbed[inotests];
    INTG iminisize = (imatsize * 2) - 1; /* Number of diagonals. */
    INTG imaxiter;
    INTG immindices[20] = {5, 5, 5, 5, 5, 5, 9, 9, 15, 15, 27, 27, 27, 27, 27, 27, 
        45, 45, 81, 81};
    for (i = 0; i < inotests; i++)
    {
//        printf("Diag: %d\n", immindices[i]);
        if (immindices[i] < iminisize)
        {
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
    }        
    
    
/* Now we run the tests. */
    
    for (i = 0; i < inotests; i++)
    { 
    //    printf ("%d\n", i);
        ourtestbed[i].inoreps = 0;
        if (immindices[i] < iminisize)
        {
    //        printf ("%d\n", i);
            clock_gettime(CLOCK_MONOTONIC, &start);
            for (j = 0; j < inoreps; j++)
            {
                dconjgrad(ourtestbed[i].ourucds,
                    didentvector, dzerovector, ourtestbed[i].dret,
                    ourtestbed[i].thefp, dvectnorm, 2, 0.1, &icount); /* &istore */
       //         printf("%d, %d \n", icount, ourtestbed[i].inoreps);
                ourtestbed[i].inoreps += icount;
              //  printf("icount: %d\n", icount);
            } 
            clock_gettime(CLOCK_MONOTONIC, &end);
            ourtestbed[i].testlen = timespecDiff(&end, &start);
        } 
    }

    for (i = 0; i < inotests; i++)
    {
        printf("%f - ", (FLPT) ourtestbed[i].testlen / (/*(FLPT)TLPERS1.0 * */(FLPT)ourtestbed[i].inoreps)); // (FLPT)((1.0 * TLPERS * imatsize * inoreps * ourtestbed[i].inoreps * 
     //       ourtestbed[i].lnumdiag)/(1000000.0 * ourtestbed[i].testlen)));
    }
    printf("%d\n", imatsize);
    
/* The last state is to free up all the memory used. */
    
    free(didentvector);
    for (i = 0; i < inotests; i++)
    {
        if (immindices[i] < iminisize)
        {
            mmdestroy(&(ourtestbed[i]));
        }
    }        

    free(dzerovector); 
//    printf("Made it!\n");
    return 0;
}
