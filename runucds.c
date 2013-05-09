/* 
// testucds.c. Runs tests on Ultra Compressed Diagonal Storage.
// Written by Peter Murphy. (c) 2013
*/

#include "ucds.h"


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
    printvector("b", 2, vectb);
    FLPT * vectx = dassign(2);
 //   conjgrad(ucdsa, vectb, vectx0, vectx, 2, 0, 0.1, NULL);
    printf("Heeya\n");
    dconjgrad(ucdsa, vectb, vectx0, vectx, &multiply_ucds, &dvectnorm, 0, 0.1, NULL);
    printvector("x", 2, vectx);
    FLPT * vectbnew = dassign(2);
    vectbnew = multiply_ucds(ucdsa, vectx, vectbnew);
  //  conjgrad(ucdsa, vectx, vectx0, vectbnew, 2, 0, 0.1, NULL);
    
    printvector("bnew", 2, vectbnew);
    free(vectbnew);
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
