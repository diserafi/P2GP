========================================================================================
                                     P2GP v. 1.0
========================================================================================

P2GP is a Matlab package implementing the Proportionality-based Two-phase Gradient
Projection (P2GP) method, for the solution of Quadratic Programming problems with
a Single Linear constraint and Bounds on the variables (SLBQPs).

The package also includes SLBQPgen, a code for the generation of (convex and non-convex)
SLBQP test problems.

For details see

  D. di Serafino, G. Toraldo, M. Viola and J. Barlow,
  "A two-phase gradient method for quadratic programming problems with
  a single linear constraint and bounds on the variables", 2017,
  http://arxiv.org/abs/1705.01797
  or
  http://www.optimization-online.org/DB_HTML/2017/05/5992.html

Authors:
  Daniela di Serafino (daniela.diserafino@unicampania.it)
  Gerardo Toraldo (toraldo@unina.it)
  Marco Viola (marco.viola@uniroma1.it)

------------------------------------------------------------------------------------------

LIST OF P2GP FILES:

- p2gp.m                : main function
- simproj.m             : projection into the feasible set
- projgrad.m            : computation of the projected gradient
- checkfeas.m           : check on the feasible set
- gradcon.m             : conjugate gradient method
- cg4dklin.m, cg4dk.m   : functions for solving the subproblem in the minimization phase
                          via the CG method, in case of SLBQPs and BQPs (bound constraints
                          only), respectively
- sdc4dklin.m, sdc4dk.m : functions for solving the subproblem in the minimization phase
                          via the SDC or the SDA method, in case of SLBQPs and BQPs,
                          respectively
- linesearch1.m         : monotone line search for the identification phase
- linesearch2.m         : monotone line-search for the minimization phase

Other files:

test_basic.m            : basic example of use of P2GP
test.m                  : example of use of P2GP on some problems built with the SLBQPgen
                          problem generator (stored in the folder './ExampleProblems')

------------------------------------------------------------------------------------------

LIST OF SLBQPgen FILES (in ./SLBQPgen)

- SLBQPgen.m            : main function
- MatVetProduct.m       : computation of Hessian-vector product
- launch_generator.m    : example of use of SLBQPgen

------------------------------------------------------------------------------------------
