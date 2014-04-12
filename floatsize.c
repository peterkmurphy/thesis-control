#include <stdio.h>
#include "projcommon.h"


int main(int argc, char *argv[])
{
    printf("The size of a float is %ld.\n", sizeof(FLPT));
    FLPT fPI = 3.1415926535897;
    printf("Our float is %.10f.\n", fPI);    
    return 0;
}