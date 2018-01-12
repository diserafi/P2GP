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

function [xk,fk,gk,Gnorm,it,nprod,flag,funvalues,indfunval,gradnorm,indgrad,tolg] = ...
    gradcon(H,c,xk,nprod,maxprod,AbsTol,Reltol,funvalues,indfunval,gradnorm,indgrad,InfNorm)

%==========================================================================
% This algorithm computes an approximate solution (or stationary point) of
% the unconstrained quadratic programming problem
%
%               min (1/2) x' H x - c'x
%
% by means of the Conjugate Gradient method; it is part of the P2GP algorithm.
%
% The algorithm generally stops when
%
%              gradnorm <= max(AbsTol, RelTol*gradnorm0),              (*)
%
% where gradnorm and gradnorm0 are the gradient norm at the current point and
% at the starting point, respectively, or when the problem is discovered to be unbounded.
% 
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
% H       = sparse or dense square matrix, double, Hessian of the objective function;
%           it may also be a handle to a function which computes H*x, where x is a vector,
%           e.g., H = @(x) prodfunc(x,params);
% c       = vector of doubles, coefficients of the linear term of the objective function;
% xk      = vector of doubles, starting point;
% nprod   = integer, number of matrix-vector products already performed by P2GP;
% maxprod = integer, maximum number of matrix-vector products;
% AbsTol  = double, absolute tolerance in the stopping criterion;
% RelTol  = double, relative tolerance in the stopping criterion;
% funvalues = vector of doubles, history of the obj fun;
% indfunval = integer, position of the last nonzero element of funvalues;
% gradnorm = vector of doubles, history of the projected gradient norm
%            (2-norm or inf-norm, according to the value of InfNorm parameter);
% indgrad  = integer, position of the last nonzero element of gradnorm;
% InfNorm  = logical, if true the inf-norm is used in the stopping criterion,
%            if false the 2-norm is used;
%
%==========================================================================
%
% OUTPUT ARGUMENTS
%
% xk       = vector of doubles, computed solution;
% fk       = double, objective function value at xk;
% gk       = vector of doubles, gradient at xk;
% Gnorm    = double, gradient norm at xk (inf-norm or 2-norm according
%            to the value of the 'InfNorm' parameter);
% it       = integer, number of CG iterations,
% nprod    = integer, number of matrix-vector products performed;
% flag     = integer, information on the execution
%          -10 - the algorithm found that the problem is unbounded below,
%            0 - the algorithm found a point satisfying the stopping criterion (*) (see line 14),
%            2 - the stopping criterion (*) was not satisfied,
% funvalues = vector of doubles, history of the obj fun;
% indfunval = integer, position of the last nonzero element of funvalues;
% gradnorm = vector of doubles, history of the projected gradient norm
%            (2-norm or inf-norm, according to the value of InfNorm parameter);
% indgrad  = integer, position of the last nonzero element of gradnorm;
% tolg     = double, tolerance on the gradient norm which had to be satisied;
%==========================================================================

if isa(H,'function_handle')
    Hx = H(xk);
else
    Hx=H*xk;
end
nprod=nprod+1;
r1=c-Hx;
d1=r1;
if InfNorm
    Gnorm=norm(r1,Inf);
else
    Gnorm=norm(r1);
end
tolg = max(AbsTol,Reltol*Gnorm);
fk=0.5*(xk'*Hx)-c'*xk;
it = 0;
flag = 0;

rtr_new=r1'*r1;
exit = false;

while (~exit && (nprod<maxprod) && (Gnorm > tolg))
    
    it=it+1;
    if isa(H,'function_handle')
        Hd1=H(d1);
    else
        Hd1=H*d1;
    end
    nprod=nprod+1;
    d1Hd1 = d1'*Hd1;
    
    if d1Hd1>=eps
        alfa = (rtr_new)/(d1Hd1);
        xk = xk+alfa*d1;

        if (mod(it,25)==0)
            if isa(H,'function_handle')
                Hx=H(xk);
            else
                Hx=H*xk;
            end
            nprod=nprod+1;
        else
            Hx = Hx+alfa*Hd1;
        end
        r1=c-Hx;

        rtr_old = rtr_new;
        rtr_new = r1'*r1;
        beta=(rtr_new)/(rtr_old);
        d1=r1+beta*d1;

        fk = 0.5*(xk'*Hx)-c'*xk;
        if InfNorm
            Gnorm=norm(r1,Inf);
        else
            Gnorm=norm(r1);
        end

        gradnorm(indgrad) = Gnorm;
        indgrad = indgrad +1;
        funvalues(indfunval) = fk;
        indfunval = indfunval + 1;
    else
        exit = true;
        if norm(d1) > eps
            xk = d1;
            flag = -10;
        end
    end
end

gk = -r1;
if (nprod == maxprod) && (Gnorm > tolg)
   flag = 2; 
end


end

