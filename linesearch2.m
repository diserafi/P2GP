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

function [xmin,fmin,gmin,it,nprod,nproj] = ...
    linesearch2(H,c,xk,fk,gk,dk,l,u,q,b,checkflag,LSType,nprod,nproj)

%==========================================================================
% This function performs the projected line search used by P2GP in the
% minimization phase.
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
%
% [1] D. di Serafino, G. Toraldo, M. Viola and J. Barlow,
%     "A two-phase gradient method for quadratic programming problems with
%      a single linear constraint and bounds on the variables", 2017
%
% Available from ArXiv
%     http://arxiv.org/abs/1705.01797
% and Optimization Online
%     http://www.optimization-online.org/DB_HTML/2017/05/5992.html
%
% [2] J.J. More' and G. Toraldo,
%     "Algorithms for bound constrained quadratic programming problems",
%     Numerische Mathematik, 55 (1989), pp. 377-400
%
% [3] J.J. More' and G. Toraldo,
%     "On the solution of large quadratic programming problems with bound
%     constraints", SIAM Journal on Optimization 1 (1991), pp. 93-113
%==========================================================================
%
% INPUT ARGUMENTS
%
% H       = sparse or dense square matrix, double, Hessian of the objective function;
%           it may also be a handle to a function which computes H*x, where x is a vector,
%           e.g., H = @(x) prodfunc(x,params);
% c       = vector of doubles, coefficients of the linear term of the objective function;
% xk      = vector of doubles, current point;
% fk      = double, obj fun value at current point;
% gk      = vector of doubles, gradient at current point;
% dk      = vector of doubles, line-search direction;
% l,u     = vectors of doubles, upper and lower bounds on the variables;
% q       = vector of doubles, coefficients of the linear constraint;
% b       = double, right-hand side of the linear constraint;
% checkflag = integer, used in the call to the projection function;
%             its value is 1 if the problem is BQP, 0 if the problem is SLBQP;
% LSType  = integer, type of linesearch used [default 1]
%               0 - linesearch over the feasible direction
%                   (see. Bertsekas `Nonlinear Programming', 1999, section 2.3.1),
%              >0 - linesearch along the projection arc;
% nprod   = integer, number of matrix-vector products already performed by P2GP;
% nproj   = integer, number of calls to the projection function already performed by P2GP;
%
%==========================================================================
%
% OUTPUT ARGUMENTS
%
% xmin      = vector of doubles, computed solution;
% fmin    = double, obj fun value at xmin;
% gmin    = vector of doubles, gradient at xmin;
% it      = integer, number of line search iterations;
% nprod   = integer, number of matrix-vector products performed;
% nproj   = integer, number of calls to the projection routine;
%
%==========================================================================

mu = 1e-4;
it = 0;
maxit = min(10,max(5,ceil(length(xk))));

%--------------------------------------------------------------------------
% Compute the steplength
% first attempt: alpha = 1
% (minimizes the reduced o.f. over the given direction - see (4.3) in
% Ref. [3])
%--------------------------------------------------------------------------
alfa = 1;

sigma1 = 1e-2;
sigma2 = 1/2;
xmin = xk+alfa*dk;

lbcon = (xmin>=l);
ubcon = (xmin<=u);


IsIn = all(lbcon) && all(ubcon);


if (IsIn)
    if isa(H,'function_handle')
        Ax = H(xmin);
    else
        Ax = H*xmin;
    end
    nprod = nprod+1;
    fmin = 0.5*(xmin'*Ax)-c'*xmin;
    gmin = Ax-c;
else
    xmin = simproj(xmin,l,u,q,b,checkflag);
    nproj = nproj + 1;
    
    if LSType>0
        if isa(H,'function_handle')
            Ax = H(xmin);
        else
            Ax = H*xmin;
        end
        nprod = nprod+1;
        fmin = 0.5*(xmin'*Ax)-c'*xmin;
        gtd = gk'*(xmin-xk);
    else
        dk = xmin-xk;
        gtd = gk'*(dk);
        alfa = 1;
        
        if isa(H,'function_handle')
            Ad = H(dk);
        else
            Ad = H*dk;
        end
        nprod = nprod+1;
        Ax0 = gk + c;
        fmin = fk + dk'*(0.5*Ad + gk);
    end
    e = gtd*alfa;
    
    while (fmin > fk + mu*e) && (it < maxit)
        if LSType>0
            alfa1 = (-gtd*(alfa^2))/(2*(fmin-fk-e));
            alfa1 = max(sigma1*alfa,min(sigma2*alfa,alfa1));
            xmin = xk+alfa1*dk;
            xmin = simproj(xmin,l,u,q,b,checkflag);
            nproj = nproj + 1;
        else
            alfa1 = alfa*sigma2;
            xmin = xk+alfa1*dk;
        end
        
        if LSType>0
            if isa(H,'function_handle')
                Ax = H(xmin);
            else
                Ax = H*xmin;
            end
            nprod = nprod+1;
            fmin = 0.5*(xmin'*Ax)-c'*xmin;
        else
            fmin = fk + alfa1*dk'*(0.5*alfa1*Ad + gk);
        end
        e = alfa1*gk'*(xmin-xk);
        alfa = alfa1;
        it = it+1;
    end
    
    
    if it >= maxit
        projdk = dk;
        
        if isa(H,'function_handle')
            Apdk = H(projdk);
        else
            Apdk = H*projdk;
        end
        nprod = nprod+1;
        
        unbounded = any((projdk>0 & u>1e20) | (projdk<0 & l<-1e20));
        if unbounded
            error('The problem is unbounded from below');
        end
        
        ind1 = find(projdk<0 & l>=-1e20);
        ind2 = find(projdk>0 & u<=1e20);
        
        if isempty(ind1)
            alfavec1 = [];
        else
            alfavec1 = (l(ind1)-xk(ind1))./projdk(ind1);
        end
        
        if isempty(ind2)
            alfavec2 = [];
        else
            alfavec2 = (u(ind2)-xk(ind2))./projdk(ind2);
        end
        
        alfavect = [alfavec1; alfavec2];
        minbreak = min(alfavect(alfavect>0));
        
        xtemp = xk + minbreak*projdk;
        if isa(H,'function_handle')
            Ax = H(xtemp);
        else
            Ax = H*xtemp;
        end
        nprod = nprod+1;
        ftemp = 0.5*(xtemp'*Ax)-c'*xtemp;
        
        if ftemp < fk
            xmin = xtemp;
            fmin = ftemp;
        else
            tempdAd = projdk'*Apdk;
            if tempdAd > 0
                alfa = -(gk'*projdk)/tempdAd;
            else
                alfa = 0;
            end
            xmin = xk + alfa*projdk;
            if isa(H,'function_handle')
                Ax = H(xmin);
            else
                Ax = H*xmin;
            end
            nprod = nprod+1;
            fmin = 0.5*(xmin'*Ax)-c'*xmin;
        end
    end
    
    if ~exist('Ax','var')
        Ax = Ax0 + alfa*Ad;
    end
    gmin = Ax-c;
end

end