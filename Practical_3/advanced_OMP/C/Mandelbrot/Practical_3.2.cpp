#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

# define NPOINTS 2000
# define MAXITER 2000

struct complex{
  double real;
  double imag;
};

int main(){
  int i, j, iter, numoutside = 0;
  double area, error, ztemp;
  struct complex z, c;

  // Parallel region with OpenMP
  #pragma omp parallel for private(j, z, c, iter, ztemp) reduction(+:numoutside) schedule(dynamic, 10)
  for (i = 0; i < NPOINTS; i++) {
    for (j = 0; j < NPOINTS; j++) {
      c.real = -2.0 + 2.5 * (double)(i) / (double)(NPOINTS) + 1.0e-7;
      c.imag = 1.125 * (double)(j) / (double)(NPOINTS) + 1.0e-7;
      z = c;
      for (iter = 0; iter < MAXITER; iter++) {
        ztemp = (z.real * z.real) - (z.imag * z.imag) + c.real;
        z.imag = z.real * z.imag * 2 + c.imag;
        z.real = ztemp;
        if ((z.real * z.real + z.imag * z.imag) > 4.0) {
          numoutside++;
          break;
        }
      }
    }
  }

  // Calculate area and error
  area = 2.0 * 2.5 * 1.125 * (double)(NPOINTS * NPOINTS - numoutside) / (double)(NPOINTS * NPOINTS);
  error = area / (double)NPOINTS;

  // Output results
  printf("Area of Mandelbrot set = %12.8f +/- %12.8f\n", area, error);

  return 0;
}
