% -------------------------------------------------------------------------
% Copyright (C) 2017 by D. di Serafino, G. Toraldo, M. Viola.
%
%                           COPYRIGHT NOTIFICATION
% 
% This file is part of P2GP.
% 
% P2GP is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% P2GP is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with P2GP.  If not, see <http://www.gnu.org/licenses/>.
% -------------------------------------------------------------------------

function [d,P,c,l,u,q,b,x0vect,x_bar] = SLBQPgen(n,ncond,zeroeig,negeig, ...
                                        naxsol,degvar,ndeg,linear,nax0)
% 
%==========================================================================
% 
% SLBQPgen generates quadratic programming problems subject to bound
% constraints and one linear equality contraint
%
%               min (1/2) x' H x - c'x
%               s.t. l <= x <= u
%                    q'x = b
%
% as described in [1].
% The algorithm is a generalization of the problem generator for BQPs by
% More' and Toraldo [2]. The Hessian matrix is built as in [3].
% 
% The Hessian matrix of the problem has the form
%
%                               H = G D G'
%
% where D is a diagonal matrix and G is of the form
%        G = (I - 2 p_3 (p_3)') (I - 2 p_2 (p_2)')(I - 2 p_1 (p_1)'),
% with p_j unit vectors.
% Only the diagonal vector d = diag(D) and the unitary vectors p1, p2 and p3
% are stored (the latter are in a matrix P).
% 
% =========================================================================
% 
% Authors:
%   Daniela di Serafino (daniela.diserafino@unicampania.it),
%   Gerardo Toraldo (toraldo@unina.it),
%   Marco Viola (marco.viola@uniroma1.it)
% 
% Version: 1.0
% Last Update: July 24, 2017
%
% REFERENCES
% [1] D. di Serafino, G. Toraldo, M. Viola and J. Barlow,
%     "A two-phase gradient method for quadratic programming problems with
%      a single linear constraint and bounds on the variables", 2017,
%      http://arxiv.org/abs/1705.01797
%      or
%      http://www.optimization-online.org/DB_HTML/2017/05/5992.html
%
% [2] J. More' and G. Toraldo,
%     "Algorithms for bound constrained quadratic programming problems",
%     Numerische Mathematik, 55 (1989), pp. 377-400
% 
% [3] Y.-H. Dai and R. Fletcher,
%     "Projected Barzilai-Borwein methods for large-scale box-
%     constrained quadratic programming",
%     Numerische Mathematik, 100 (2005), pp. 21-47
% 
%==========================================================================
% 
% INPUT ARGUMENTS
% 
% n        = integer, number of variables;
% ncond    = integer, log_{10} cond(H);
% zeroeig  = double in [0, 1), fraction of zero eigenvalues of H;
% negeig   = double in [0, 1), fraction of negative eigenvalues of H;
% naxsol   = double in [0, 1), fraction of active variables at x_bar;
% degvar   = double in [0, 1), fraction of active variables at x_bar that are degenerate;
% ndeg     = integer, amount of near-degeneracy;
% linear   = logical, true for SLBQPs, and false for BQPs;
% nax0     = double in [0,1) or vector of doubles in [0,1], fraction of active
%            variables at the starting point or points.
% 
% OUTPUT ARGUMENTS
% 
% d        = vector of doubles, eigenvalues of the matrix H (i.e. entries of the diagonal of D);
% P        = matrix of doubles of size n-by-3 whose columns contain the
%            three unitary vectors for the construction of the matrix G;
% c        = vector of doubles, coefficients of the linear term of the objective function;
% l,u      = vectors of doubles, upper and lower bounds on the variables;
% q        = vector of doubles, coefficients of the linear constraint;
% b        = double, right-hand side of the linear constraint;
% x0vect   = matrix of doubles of size n-by-k, where k is the length of vector nax0;
%            the i-th column is a point with fraction of active box variables
%            indicated by the i-th element of nax0;
% x_bar    = vector of doubles, stationary point for the problem; if the
%            problem is convex it represents the solution of the
%            problem.
% 
%==========================================================================

%% Build the diagonal matrix D
ee = zeros(n,1); % preallocation, only for speed
n1 = n-1;
for i = 1:n
    ee(i) = (i-1)/n1 * ncond;
end
d = 10.^ee;

%% Change the sign of negeig eigenvalues
if negeig > 1
    error('The percentage of negative eigenvalues must be a scalar between 0 and 1');
elseif negeig>0
   tempind = rand(size(d));
   d(tempind <= negeig) = -d(tempind <= negeig);
end

%% Set to 0 zeroeig eigenvalues
if zeroeig > 1
    error('The percentage of zero eigenvalues must be a scalar between 0 and 1');
elseif zeroeig>0
   tempind = rand(size(d));
   tempind(1) = 1; tempind(end) = 1;
   d(tempind <= zeroeig) = 0;
end

%% Generate the unitary vectors needed for the construction of matrix G
P = -1+2*rand(n,3);
P(:,1) = P(:,1)/norm(P(:,1));
P(:,2) = P(:,2)/norm(P(:,2));
P(:,3) = P(:,3)/norm(P(:,3));

%% Generate the stationary point as a vector in [-1,1]^n
x_bar = -1 + 2*rand(n,1);

%% If linear, generate the linear equality constraint
if linear
    q = -1+2*rand(size(x_bar));
    b = q'*x_bar;
else
    q = [];
    b = [];
end



%% Find the indices of the constraints that are active at the solution
mu = rand(n,1);
mmu = mu <= naxsol;
iactive = find(mmu);

nactive = length(iactive);

%% Set the corresponding residuals (r) according to degvar and ndeg; the remaining residuals are set to zero.
r = zeros(n,1);
if ndeg<0 || degvar < 0 
    error('ndeg and degvar must be respectively a non-negative integer and a scalar in [0,1]');
elseif degvar > 0
    temprand = rand(nactive,1);
    DegInd = temprand < degvar;
    r(iactive) = 10.^(-ndeg*mu(iactive));
    r(iactive(DegInd)) = 0;          % null residual for degenerate variables
else
    r(iactive) = 10.^(-ndeg*mu(iactive));
end
irand = find(rand(n,1) <= .5);
r(irand) = -r(irand);            % change sign of (approximately) half the residuals

%% Compute the linear term of the objective function
if ~linear
    c = MatVetProduct(d,P,x_bar)-r;
else
    gamma = 0;
    while gamma == 0
        gamma = -1 + 2*rand;
    end
    c = MatVetProduct(d,P,x_bar)-r-gamma*q;
end

%% Set lower and upper bounds
l = -ones(n,1);
templogvectlow = false(n,1);
templogvectupp = false(n,1);
if degvar > 0
    ndegvars = length(iactive(DegInd));
    randomlowuppdeg = rand(ndegvars,1);
    templogvectlow(iactive(DegInd)) = randomlowuppdeg<0.5;
    templogvectupp(iactive(DegInd)) = randomlowuppdeg>=0.5;
end
low = r>0;
iactlow = find(mmu & (low | templogvectlow));
l(iactlow) = x_bar(iactlow);
u = ones(n,1);
upp = r<0;
iactupp = find(mmu & (upp | templogvectupp));
u(iactupp) = x_bar(iactupp);

%% Choose the indices of the constraints that are active at the starting points and generate them
numstart = size(nax0,2);
x0vect = zeros(n,numstart);
for ix0 = 1:numstart
    mu0 = rand(n,1);
    mmu0 = mu0 <= nax0(ix0);
    choice = rand(n,1) <= .5;
    iact0low = find (mmu0 & choice);
    iact0upp = find (mmu0 & (~choice));
    x0 = (l+u)/2;
    x0(iact0low) = l(iact0low);
    x0(iact0upp) = u(iact0upp);
    x0vect(:,ix0) = x0;
end

end
