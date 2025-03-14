//test_gprof.c, inspired by http://www.thegeekstuff.com/2012/08/gprof-tutorial
//
#include<stdio.h>
#include <stdlib.h>

static void square(long n)
{ printf("Inside square \n");
  for(long i=0;i<n*n;i++);
}

void cube(long n)
{ printf("Inside cube \n");
  for(long i=0;i<n*n/10000*n;i++);
}

int main(int argc, char *argv[])
{
    long n;
    if(argc == 2) n=atoi(argv[1]); else n=4000;
    printf("Inside main\n");
    square(n);
    cube(n);
    return 0;
}

