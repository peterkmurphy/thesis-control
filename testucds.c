/* 
// testucds.c. Runs tests on Ultra Compressed Diagonal Storage
// for correctness, not time.
// Written by Peter Murphy. (c) 2013
*/

#include "ucds.h"

/*
// We do some "expected value" functions for vectors consisting
// only of a particular value. 
*/

FLPT expectedvaluenorm(const INTG ivectsize, const INTG imode, 
    const FLPT dvalue)
{
    if (imode == 1)
    {
        return dvalue * (FLPT) ivectsize;
    }
    else if (imode == 2)
    {
        return sqrt(pow(dvalue, 2) * (FLPT) ivectsize);
    }
    else /* Infinity Norm */
    {
        return dvalue;
    }
}

/* 
// We have an expected vector vector, where one expects all values to equal
// a certain value.
*/

INTG bisallvalues(const INTG ivectsize, const FLPT dvalue, 
    const FLPT * dvector)
{
    INTG i; /* An iterator. */
    for (i = 0; i < ivectsize; i++)
    {
        if (dvector[i] != dvalue)
        {
            return 0; /* Fail. */
        }
    }
    return 1; /* True */
}

/*
// This tests the multiplication functions to see if they are equal to 
// each other.
*/

INTG btestmult(const INTG ivectsize, const ucds * ucdsa, const INTG inoreps)
{
    FLPT * dmultresult = dassign(ivectsize); /* Multiply above with ucdsa. */
    FLPT * daltmultresult = dassign(ivectsize); /* The alt result. */ 
    FLPT * ddifference = dassign(ivectsize); /* The difference between them. */
    FLPT * dvectorb = dassign(ivectsize);
    
/* Possible multiplication and pointer functions. */    
    
    INTG i, j; /* Iteration variables. */
    INTG ifailurecount = 0; /* This stores how many failures. */
    FLPT dnorm; /* To store the norm. */
    
    for (i = 0; i < inoreps; i++) /* Over norm modes. */
    {
        doverwriterandom(ivectsize, dvectorb);
        multiply_ucds(ucdsa, dvectorb, dmultresult);
        multiply_ucdsalt(ucdsa, dvectorb, daltmultresult);
        dvectsub (ivectsize, dmultresult, daltmultresult, ddifference);
 //       printvector("dvectorb", ivectsize, dvectorb);
 //       printvector("dmultresult", ivectsize, dmultresult);
 //       printvector("daltmultresult", ivectsize, daltmultresult);
 //       printvector("ddifference", ivectsize, ddifference);
        for (j = 0; j < 3; j++)
        {
            dnorm = dvectnorm(ivectsize, j, ddifference);
            if (dnorm > 0.1)
            {
                ifailurecount++;
            }
 //           printf("The norm is %f.\n", dnorm); 
            dnorm = daltnorm(ivectsize, j, ddifference);
            if (dnorm > 0.1)
            {
                ifailurecount++;
            }
 //           printf("The norm is %f.\n", dnorm);
        }
    }
    free(dvectorb);
    free(daltmultresult);
    free(dmultresult);
    free(ddifference);
    return ifailurecount;
}

/* 
// This tests the conjugate gradient vector function by passing in an ucds, A,
// and a vector b, and seeing that the conjugate gradient function returns x,
// the solution of Ax = b.
*/

INTG btestconggrad(const INTG ivectsize, const ucds * ucdsa, 
    const FLPT dminerror, const INTG inoreps)
{
    FLPT * dvect0 = dsetvector(ivectsize, 0.0); /* Starting vector. */
    FLPT * dconjresult = dassign(ivectsize); /* The result of conj. grad. */
    FLPT * dmultresult = dassign(ivectsize); /* Multiply above with ucdsa. */
    FLPT * ddifference = dassign(ivectsize); /* The difference between them. */
    FLPT * dvectorb = dassign(ivectsize);
    
/* Possible multiplication and pointer functions. */    
    
    fpmult fpmultfuns[2] = {&multiply_ucds, &multiply_ucdsalt}; 
    fpnorm fpnormfuns[2] = {&dvectnorm, &daltnorm};
    INTG i, j, k, l; /* Iteration variables. */
    INTG icount; /* A count of how many iterations. */
    INTG ifailurecount; /* This stores how many failures. */
    INTG istore;
    INTG itotalfailures = 0;
    FLPT dnorm;
    
    for (i = 0; i < 3; i++) /* Over norm modes. */
    {
        for (j = 0; j < 2; j++) /* Over norm functions. */
        {
            doverwriterandom(ivectsize, dvectorb);
            for (k = 0; k < 2; k++) /* Over multiplication functions. */
            {
    //            printvector("dvectorb", ivectsize, dvectorb);
                icount = 0;
                ifailurecount = 0;
    //            printf("Mode %d-%d-%d: ", i, j, k);
                for (l = 0; l < inoreps; l++)
                {
                    dconjgrad(ucdsa, dvectorb, dvect0, dconjresult,
                        fpmultfuns[k], fpnormfuns[j], i, 
                        dminerror, &istore);
            //        printvector("dconjresult", ivectsize, dconjresult);
                    if (istore > icount)
                    {   
                        icount = istore;
                    }
                    fpmultfuns[k](ucdsa, dconjresult, dmultresult);
                    dvectsub (ivectsize, dvectorb, dmultresult, ddifference);
                    dnorm = fpnormfuns[j](ivectsize, i, ddifference);
            //        printvector("ddifference", ivectsize, ddifference);
                    if (dnorm > dminerror)
                    {
                        printf("This is dnorm: %f.\n", dnorm);
                        printf("Mode %d-%d-%d: ", i, j, k);
                        printvector("dconjresult", ivectsize, dconjresult);
                        printvector("ddifference", ivectsize, ddifference);
                        ifailurecount++;                    
                    }
                }
    //            printf("reps: %d, maxcount: %d; failures: %d\n", 
    //                inoreps, icount, ifailurecount);
                itotalfailures += ifailurecount;
            }
        }
    }
    free(dvect0);
    free(dconjresult);
    free(dmultresult);
    free(ddifference);
    free(dvectorb);
    return itotalfailures;
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

    const INTG iminmatsize = 2;
    
    if (argc < 3)
    {
        printf("To execute this, type:\n\nudcs n m\n\nWhere:\nn (>= ");
        printf("%d) ", iminmatsize);
        printf("is the size of the matrices to be multiplied and tested;");
        printf("\nm (>= 1) is the number of repetitions.\n\n");
        return(0);
    }
    const INTG imatsize = atoi(argv[1]);
//    printf("We start it with a matrix size of %d!\n", imatsize);
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

/* These are variables for taking the outputs of functions. */

    FLPT ddummy; /* For output values. */
    FLPT dexpect; /* For expected values. */
    FLPT * didentvector = dsetvector(imatsize, 1.0); /* Has equal values. */
    FLPT * dvectout = dsetvector(imatsize, 0.0);

    
    INTG i,j; /* An iteration variable. */
    
/* Here are the tests. */    
  
    ddummy = ddotprod (imatsize, didentvector, didentvector);
    if (ddummy != (imatsize * 1.0))
    {
        printf("Dot product's broken!\n");
    }
    dscalarprod (imatsize, 2.0, didentvector, dvectout);
    dvectadd(imatsize, didentvector, didentvector, dvectout);
    dvectsub(imatsize, didentvector, didentvector, dvectout);
    for (i = 0; i < 3; i++)
    {
        ddummy = dvectnorm(imatsize, i, didentvector);
        dexpect = expectedvaluenorm(imatsize, i, 1.0);
        if ((ddummy - dexpect) > 0.01)
        {
            printf("Norm: %d, ddummy: %f; expected: %f\n", i, ddummy, dexpect);
        }
        ddummy = daltnorm(imatsize, i, didentvector);
        dexpect = expectedvaluenorm(imatsize, i, 1.0);
        if ((ddummy - dexpect) > 0.01)
        {
            printf("Alt Norm: %d, ddummy: %f; expected: %f\n", i, ddummy, dexpect);
        }
    }


/* Now this is an attempt to set up a test environment. */

    FLPT *dzerovector = dsetvector(imatsize, 0.0); 
    FLPT * dmultvector = dsetvector(imatsize, 1.0);
    FLPT * ddifvector = dassign(imatsize);
    FLPT dnorm;
    const FLPT dmaxerror = 0.001;
    const INTG inotests = 10;
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
//    printf("The maximum number of diagonals is %d\n", iminisize); 
    INTG immindices[10] = {5, 5, 5, 9, 15, 27, 27, 27, 45, 81};
    for (i = 0; i < inotests; i++)
    {
    //    printf("Diag: %d\n", immindices[i]);
        if (immindices[i] <= iminisize)
        {
 //           printf("Diag: %d\n", immindices[i]);
            mmsetup(immindices[i], imatsize, &(ourtestbed[i]));
            if (i == 6)
            {
                ourtestbed[i].thefp = &multiply_ucdsalt27;
            }
            else if (i == 7)
            {
                ourtestbed[i].thefp = &multiply_ucdsaltd27;
            }            
            else if (i == 1)
            {
                ourtestbed[i].thefp = &multiply_ucdsalt5;
            }
            else if (i == 2)
            {
                ourtestbed[i].thefp = &multiply_ucdsaltd5;
            }
            else
            { 
                ourtestbed[i].thefp = &multiply_ucdsalt;
            }
        }
    }        
    
    
/* Now we run the tests. */
    
    for (i = 0; i < inotests; i++)
    { 
    //    printf ("%d\n", i);
        if (immindices[i] <= iminisize)
        {
    //        printf ("%d\n", i);
            for (j = 0; j < inoreps; j++)
            {
   //             printf("Diag Run: %d\n", immindices[i]);
   //             printucds("Ourucds", ourtestbed[i].ourucds);
                dconjgrad(ourtestbed[i].ourucds,
                    didentvector, dzerovector, ourtestbed[i].dret,
                    ourtestbed[i].thefp, dvectnorm, 2, dmaxerror, &icount); /* &istore */
                ourtestbed[i].inoreps += icount;
   //             printvector("dret", imatsize, ourtestbed[i].dret);
                ourtestbed[i].thefp(ourtestbed[i].ourucds, ourtestbed[i].dret, dmultvector);
                dvectsub (imatsize, didentvector, dmultvector, ddifvector);
                dnorm = dvectnorm(imatsize, 2, ddifvector);
             //   printf("Norm %f\n", dnorm);
                if (dnorm > dmaxerror)
                {
                    printf("We have a problem!\n");                    
                }
            }    
        } 
    }
 
    if (imatsize >= 7)
    {
        srand (time(NULL));
        INTG tldiagindices[5] = {-3, -1, 0, 1, 3};
        FLPT tddiagvals[5] = {-1.0, -1.0, 4.0, -1.0, -1.0};
        INTG inoerrors;        
        ucds * ucdsa = mmatrix_ucds(imatsize, tldiagindices, tddiagvals, 3);
        inoerrors = btestmult(imatsize, ucdsa, inoreps);
        if (inoerrors != 0)
        {
            printf("Multiplication errors: %d\n", btestmult(imatsize, ucdsa, inoreps));
        }
        inoerrors = btestconggrad(imatsize, ucdsa, 0.01, inoreps);
        if (inoerrors != 0)
        {
            printf("We had errors with a matrix size of %d!\n", imatsize);
        }
        destroy_ucds(ucdsa);
    }

/* The last state is to free up all the memory used. */

 
    free(didentvector);
    free(dvectout);
    for (i = 0; i < inotests; i++)
    {
        if (immindices[i] <= iminisize)
        {
 //           printf("Diag Free: %d\n", immindices[i]);
            mmdestroy(&(ourtestbed[i]));
        }
    }        

    free(dzerovector); 
    free(dmultvector);
    free(ddifvector);
//    printf("We made it with a matrix size of %d!\n", imatsize);
    return 0;
}
