/* 
// testucds.c. Runs tests on Ultra Compressed Diagonal Storage
// and Conjugate Gradient for correctness, not time. Also 
// checks routines like norms, vector products and scalar
// products for correctness.
// Written by Peter Murphy. (c) 2013
*/

#include "projcommon.h"
#include "ucds.h"

/*
// Norms on vectors consisting only of a particular value would
// return an expected number; this function checks if the norms
// actually return the expected value. 
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
// This checks if all the value in the vector dvector (with size
// ivectsize) are equal to the given value dvalue.
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
        for (j = 0; j < 3; j++)
        {
            dnorm = dvectnorm(ivectsize, j, ddifference);
            if (dnorm > 0.1)
            {
                ifailurecount++;
            }
            dnorm = daltnorm(ivectsize, j, ddifference);
            if (dnorm > 0.1)
            {
                ifailurecount++;
            }
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
    FLPT * dvect0 = dsetvector(ivectsize, 1.0); /* Starting vector. */
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
                icount = 0;
                ifailurecount = 0;
                for (l = 0; l < inoreps; l++)
                {
                    dconjgrad(ucdsa, dvectorb, dvect0, dconjresult,
                        fpmultfuns[k], fpnormfuns[j], i, 
                        dminerror, &istore);
                    if (istore > icount)
                    {   
                        icount = istore;
                    }
                    fpmultfuns[k](ucdsa, dconjresult, dmultresult);
                    dvectsub (ivectsize, dvectorb, dmultresult, ddifference);
                    dnorm = fpnormfuns[j](ivectsize, i, ddifference);
                    if (dnorm > dminerror)
                    {
                        printf("This is dnorm: %f.\n", dnorm);
                        printf("Mode %d-%d-%d: ", i, j, k);
                        printvector("dvectorb", ivectsize, dvectorb);
                        printvector("dvect0", ivectsize, dvect0);
                        printvector("dmultresult", ivectsize, dmultresult);                        
                        printvector("dconjresult", ivectsize, dconjresult);
                        printvector("ddifference", ivectsize, ddifference);
                        ifailurecount++;                    
                    }
                    else
                    {
                         printf("Mode %d-%d-%d: succeeded.\n", i, j, k);
                    }
                }
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
    FLPT * dresultvector = dassign(imatsize);
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
    INTG immindices[10] = {5, 5, 5, 9, 15, 27, 27, 27, 45, 81};
    for (i = 0; i < inotests; i++)
    {
        if (immindices[i] <= iminisize)
        {
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
    
//    printf("HHH\n");
//    return(0);    
/* Now we run the tests. */
    INTG inoerrors; 
    for (i = 0; i < inotests; i++)
    { 
        if (immindices[i] <= iminisize)
        {
            for (j = 0; j < inoreps; j++)
            {
                inoerrors = btestmult(imatsize, ourtestbed[i].ourucds, inoreps);
                if (inoerrors != 0)
                {
                    printf("Multiplication errors: %d\n", inoerrors);
                }
                
                
                
        /*        dconjgrad(ourtestbed[i].ourucds,
                    didentvector, dzerovector, ourtestbed[i].dret,
                    ourtestbed[i].thefp, dvectnorm, 2, dmaxerror, &icount); // &istore 
                ourtestbed[i].inoreps += icount;
                ourtestbed[i].thefp(ourtestbed[i].ourucds, ourtestbed[i].dret, dmultvector);
                dvectsub (imatsize, didentvector, dmultvector, ddifvector);
                dnorm = dvectnorm(imatsize, 2, ddifvector);
                if (dnorm > dmaxerror)
                {
                    printf("CG: we have a problem with test %d with norm as %f and maxerror as %f after %d iterations!\n",
                     i, dnorm, dmaxerror, icount);                    
                }
                else
                {
           //         printf("CG succeeded with test %d with norm as %f and maxerror as %f after %d iterations!\n",
           //          i, dnorm, dmaxerror, icount);                 
                } */
            }    
        } 
    }
 
    if (imatsize >= 7)
    {
        srand (time(NULL));
        INTG tldiagindices[5] = {-3, -1, 0, 1, 3};
        FLPT tddiagvals[5] = {-1.0, -1.0, 4.0, -1.0, -1.0};
       
        ucds * ucdsa = mmatrix_ucds(imatsize, tldiagindices, tddiagvals, 3);
        inoerrors = btestmult(imatsize, ucdsa, inoreps);
        if (inoerrors != 0)
        {
            printf("Multiplication errors: %d\n", btestmult(imatsize, ucdsa, inoreps));
        }
        
        dconjgrad(ucdsa, didentvector, dzerovector, dresultvector,
                    &multiply_ucds, dvectnorm, 2, dmaxerror, &icount); /* &istore */
        //        ourtestbed[i].inoreps += icount;
        multiply_ucds(ucdsa, dresultvector, dmultvector);
        dvectsub (imatsize, didentvector, dmultvector, ddifvector);
        dnorm = dvectnorm(imatsize, 2, ddifvector);
        if (dnorm > dmaxerror)
        {
            printf("CG: we have a problem with test %d with norm as %f and maxerror as %f after %d iterations!\n",
                 -1, dnorm, dmaxerror, icount);                    
        }
        dconjgrad(ucdsa, didentvector, dzerovector, dresultvector,
                    &multiply_ucdsalt, dvectnorm, 2, dmaxerror, &icount); /* &istore */
        //        ourtestbed[i].inoreps += icount;
        multiply_ucdsalt(ucdsa, dresultvector, dmultvector);
        dvectsub (imatsize, didentvector, dmultvector, ddifvector);
        dnorm = dvectnorm(imatsize, 2, ddifvector);
        if (dnorm > dmaxerror)
        {
            printf("CG: we have a problem with test %d with norm as %f and maxerror as %f after %d iterations!\n",
                 -2, dnorm, dmaxerror, icount);                    
        }
        
        
        
      //  inoerrors = btestconggrad(imatsize, ucdsa, 0.001, inoreps);
      //  if (inoerrors != 0)
      //  {
      //      printf("We had errors with a matrix size of %d!\n", imatsize);
      //  }
        destroy_ucds(ucdsa);
    } 

/* The last state is to free up all the memory used. */

 
    free(didentvector);
    free(dvectout);
    for (i = 0; i < inotests; i++)
    {
        if (immindices[i] <= iminisize)
        {
            mmdestroy(&(ourtestbed[i]));
        }
    }        

    free(dzerovector); 
    free(dmultvector);
    free(ddifvector);
    free(dresultvector);
    printf("We made it with a matrix size of %d!\n", imatsize);
    return 0;
}
