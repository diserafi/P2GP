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

function [x,dir1,it,nprod,flag_cg] = cg4dk(WorkingSet,A,c,x,dir1,xi,itmax,nprod)

%==========================================================================
% The function uses the CG algorithm to find an approximate solution for
% 
%                       min 0.5 x' A x - c'x
% 
% restricted to the variables indicated by the vector SubInd.
% It is customized for the minimization phase of the P2GP for BQP problems.
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
% x       = vector of doubles, starting point;
% dir1    = [optional] vector of doubles, first direction taken by the CG algorithm;
% xi      = double, parameter for the stopping criterion see ref. [1], eq. (4.18);
% itmax   = integer, maximum number of steps;
% nprod   = integer, number of matrix-vector products already performed by P2GP;
%
% Remark: dir1 is used to recover the minimization started in a previous call.
% 
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
% 
%==========================================================================

c(~WorkingSet) = 0;
x(~WorkingSet) = 0;
if ~isempty(dir1)
    dir1(~WorkingSet) = 0;
end

etagrad = 1e-16;
prod = @(v) res_hessprod(v,A,WorkingSet);

if isempty(dir1)
    Ax = x;
    dir1 = c - Ax;
    r1 = c - Ax;
else
    Ax = prod(x);
    nprod = nprod+1;
    r1 = c - Ax;
end


fval_old = 0.5*(x'*Ax)-c'*x;
it = 0;
decr = 1;
maxdiff = 0;
rtr_new = r1'*r1;    
    
exit = 0;
flag_cg = 0;

while ~exit && (it<itmax) && (decr)
    
    it = it+1;
    Ad1 = prod(dir1);
    dAd1 = (dir1'*Ad1);
    nprod = nprod+1;
    
    if dAd1>eps
        alfa = (rtr_new)/dAd1;
        x = x+alfa*dir1;
        if (mod(it,20) == 0)
            Ax = prod(x);
            r1 = c - Ax;
            nprod = nprod+1;
        else
            Ax = Ax+alfa*Ad1;
            r1 = r1-alfa*Ad1;
        end
        
        rtr_old = rtr_new;
        rtr_new = r1'*r1;
        beta = (rtr_new)/(rtr_old);
        dir1 = r1+beta*dir1;
        fval = 0.5*(x'*Ax)-c'*x;
        diff = fval_old-fval;
        
        decr = diff > xi*maxdiff;
        
        maxdiff = max(diff,maxdiff);
        fval_old = fval;
        norma_grad_cg = norm(r1);
    else
        exit = true;
        if norm(dir1) > eps
            x = dir1;
            flag_cg = -10;
        end
    end

end

if flag_cg >= 0
    flag_cg = (norma_grad_cg < etagrad);
end
if ~flag_cg && (it >= itmax)
    flag_cg = 3;
end


end


function y = res_hessprod(x,Hess,Ind)
    if ~isempty(Ind)
        x(~Ind) = 0;
    end
    if isa(Hess,'function_handle')
        y = Hess(x);
    else
        y = Hess*x;
    end
    if ~isempty(Ind)
        y(~Ind) = 0;
    end
end

