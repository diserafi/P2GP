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

function [x,dir1,it,nprod,flag_cg,info] = ...
    cg4dklin(WorkingSet,A,c,qk,x,dir1,xi,itmax,nprod,info)

%==========================================================================
% The function uses the CG algorithm to find an approximate solution for
% 
%                     min  0.5 x'*A*x - c'*x
%                     s.t. qk'*x = 0
% 
% It is customized for the minimization phase of the P2GP for SLBQP problems.
%==========================================================================
% 
% Authors:
%   Daniela di Serafino (daniela.diserafino@unicampania.it),
%   Gerardo Toraldo (toraldo@unina.it),
%   Marco Viola (marco.viola@uniroma1.it)
% 
% Version: 1.0
% Last Update: July 24, 2017
%
% REFERENCES:
% [1] D. di Serafino, G. Toraldo, M. Viola and J. Barlow,
%     "A two-phase gradient method for quadratic programming problems with
%      a single linear constraint and bounds on the variables", 2017
%
% Available from ArXiv
%     http://arxiv.org/abs/1705.01797
% and Optimization Online
%     http://www.optimization-online.org/DB_HTML/2017/05/5992.html
%==========================================================================
% 
% INPUT ARGUMENTS
%
% WorkingSet  = vector of logical, the true entries indicate the variables to
%               which the problem is restricted;
% A       = sparse or dense square matrix, double, Hessian of the objective function;
%           it may also be a handle to a function which computes H*x, where x is a vector,
%           e.g., A = @(x) prodfunc(x,params);
% c       = vector of doubles, coefficients of the linear term of the objective function;
% qk      = vector of doubles, coefficients of the linear constraint;
% x       = vector of doubles, starting point;
% dir1    = [optional] vector of doubles, first direction taken by the CG algorithm;
% xi      = double, parameter for the stopping criterion see ref. [1], eq. (4.18);
% itmax   = integer, maximum number of steps;
% nprod   = integer, number of matrix-vector products already performed by P2GP;
% info    = [optional] struct variable, containing information on the projection 
%           of problem onto the linear constraint made in a previous call.
%
% Remark: dir1 and info are used to recover the minimization started in a
% previous call.
%==========================================================================
%
% OUTPUT ARGUMENTS
%
% x       = vector of doubles, computed solution;
% dir1    = vector of doubles, last direction taken by the CG algorithm;
% it      = integer, number of CG iteration;
% nprod   = integer, number of matrix-vector products performed;
% flag_cg = integer, information on the execution
%         -10 - P2GP found a descent direction with nonpositive curvature,
%               this direction is provided as the output x;
%           0 - the algorithm found a point satisfying the stopping criterion,
%           1 - the algorithm found a stationary point for the problem,
%           3 - the did not find a stationary point and itmax steps have
%               been executed.
% info    = struct variable, containing information on the projection of 
%           problem onto the linear constraint.
% 
%==========================================================================

c(~WorkingSet) = 0;
x(~WorkingSet) = 0;
qk(~WorkingSet) = 0;

if ~isempty(dir1)
    dir1(~WorkingSet) = 0;
end
etagrad = 1e-16;

% construction of rho and w for the unconstrained quadratic problem
if isempty(info)
    ind = find(WorkingSet,1,'first');

    rho = -sign(qk(ind))*norm(qk);
    w = qk;
    w(ind) = w(ind) - rho;
    w = w/norm(w);
else
    ind = info.ind;
    rho = info.rho;
    w = info.w;
end

NewWorkSet = WorkingSet;
NewWorkSet(ind) = 0;

% construction of the function handlers for the product and the new linear term
P = @(x) (x - 2*w*(w'*x));
PHP = @(v) hessprod(v,A,NewWorkSet,w);

z = P(x);
z(ind) = 0;

Pc = P(c);
Pc(ind) = 0;

PHPz = PHP(z);
nprod = nprod+1;
g1 = PHPz - Pc;
fval_old = 0.5*z'*PHPz - z'*Pc;

norma_grad_cg = norm(g1);

it = 0;
flag = 1;
flag_cg = 0;
exit = 0;
maxdiff = 0;

if isempty(dir1)
    dir1 = -g1;
end

while ~exit && ( (it<itmax) && (flag) )
    
    PHPd1 = PHP(dir1);
    nprod = nprod+1;
    d1PHPd1 = dir1'*PHPd1;
    
    if d1PHPd1>eps
        alfa = (dir1'*g1)/(d1PHPd1); %<0
        z = z - alfa*dir1;
        
        if mod(it,20)==0
            PHPz = PHP(z);
            nprod = nprod+1;
            g1 = PHPz - Pc;
            fval = 0.5*z'*PHPz - z'*Pc;
        else
            PHPz = PHPz - alfa*PHPd1;
            g1 = g1 - alfa*PHPd1;
            fval = 0.5*z'*PHPz - z'*Pc;
        end
        
        beta = (g1'*PHPd1)/(d1PHPd1);
        dir1 = -g1+beta*dir1;
        
        diff = fval_old-fval;
        norma_grad_cg = norm(g1);
        
        flag = (diff > xi*maxdiff) && (norma_grad_cg >= etagrad);
        maxdiff = max(diff,maxdiff);
        fval_old = fval;
    else
        exit = 1;
        if norm(dir1)>eps
            z = dir1;
            flag_cg = -10;
        end
    end
    it = it+1;
    
end

if flag_cg >= 0
    flag_cg = (norma_grad_cg < etagrad);
end
if ~flag_cg && (it >= itmax)
    flag_cg = 3;
end

x = P(z);

info = struct('rho',rho,'w',w,'ind',ind);

end

function y = hessprod(x,Hess,Ind,w)
if ~isempty(Ind)
    x(~Ind) = 0;
end

y = x - 2*w*(w'*x);

if isa(Hess,'function_handle')
    y = Hess(y);
else
    y = Hess*y;
end

y = y - 2*w*(w'*y);

if ~isempty(Ind)
    y(~Ind) = 0;
end
end