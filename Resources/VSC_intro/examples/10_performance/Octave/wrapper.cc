#include <oct.h>
#include <iostream>
#include <stdio.h>
extern "C" {
void setup_(int*, int*, double*);
}
using namespace std;

//      a=setup_(n, d)
DEFUN_DLD(setup_c, args, nargout, "Setup a nice matrix")
{
  int nargin=args.length();
  if (nargin !=2) { 
    cout << "Error: Parameter: (matrix size) n, (some value) d" << nargin << endl;
    return octave_value(-1);
  }
  int n(args(0).int_value());
  int d(args(1).int_value());
  Matrix a(n, n);
    /*  create pointers to the matrices */
  double * ap = a.fortran_vec();
    /*  call the C subroutine */
  setup_(&n, &d, ap);
  return(octave_value_list(octave_value(a)));
}

