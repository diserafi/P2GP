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

function [x_proj,Ivar,flag] = simproj(c,l,u,q,b,checkflag)

%==========================================================================
% This function computes the projection of a point c onto the sets
%             (i)           l <= x <= u
% or
%             (ii)     l <= x <= u && q' x = b
% needed in the P2GP algorithm.
% The algorithm used in the case (ii) is a generalization of the Dai-Fletcher
% algorithm for the solution of
%                           min 0.5 * ||x - c||^2
%                       subj to sum(x) = b    
%                               x >= 0
%
% and has been adapted from "projectDF" included in the package IRMA by
% R. Cavicchioli, R. Zanella, G. Zanghirati and L. Zanni (see below for the
% documentation of the original version).
%
%==========================================================================
%
% Authors of this version:
%   Daniela di Serafino (daniela.diserafino@unicampania.it),
%   Gerardo Toraldo (toraldo@unina.it),
%   Marco Viola (marco.viola@uniroma1.it)
% 
% Version: 1.0
% Last Update: July 24, 2017
%
% REFERENCES:
%     D di Serafino, G. Toraldo, M. Viola, J. Barlow,
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
% c       = vector of doubles, point to be projected;
% l,u     = vectors of doubles, upper and lower bounds on the variables;
% q       = vector of doubles, coefficients of the linear constraint;
% b       = double, right-hand side of the linear constraint;
% checkflag = integer, its value is 1 if the set is made only by a box,
%             0 if there is a valid linear equality constraint;
%
%==========================================================================
%
% OUTPUT ARGUMENTS
%
% x_proj    = vector of doubles, projection of c over the specified set;
% Ivar      = integer vector with the same length as x_proj,
%            -1 - the corresponding entry of x_proj is on its lower bound,
%             0 - the corresponding entry of x_proj is free (l(i) < x_proj(i) < u(i)),
%             1 - the corresponding entry of x_proj is on its upper bound,
%             2 - the corresponding entry of x_proj is fixed because l(i)=u(i);
% flag      = integer, information on the execution
%             1 - the projection has been successfully computed,
%            -1 - the specified set is empty.
% 
%==========================================================================
%==========================================================================
%                  DOCUMENTATION OF THE ORIGINAL VERSION
%
% projectDF - Dai-Fletcher algorithm for separable simply constrained QPs
%   This function applies the secant-based Dai-Fletcher algorithm [1] to solve
%   the separable, singly linearly and nonnegatively constrained quadratic  
%   programming problem 
%                           min 0.5 * x'*diag(dia)*x - c'*x
%                       subj to sum(x) = b    
%                               x >= 0
%
%   [1] Y. H. Dai, R. Fletcher, "New algorithms for singly linearly constrained
%       quadratic programs subject to lower and upper bounds",
%       Math. Program., Ser. A 106, 403 - 421 (2006).
%
% SYNOPSIS
%   [x, biter, siter,r] = projectDF(b, c, dia)
%
% INPUT
%   b     (double)       - rhs of the linear constraint
%   c     (double array) - coefficient vector of the objctive's linear term
%   dia   (double array) - diagonal elements of the Hessian
%
% OUTPUT
%   x     (double array) - solution vector
%   biter (pos. integer) - total number of backtracking iterations
%   iter  (pos. integer) - total number of secant iterations
%   r     (double)       - linear constraint residual value (sum(x)-b) 
%                          at the solution
%
% -------------------------------------------------------------------------
%
% This software is developed within the research project
%
%        PRISMA - Optimization methods and software for inverse problems
%                           http://www.unife.it/prisma
%
% funded by the Italian Ministry for University and Research (MIUR), under
% the PRIN2008 initiative, grant n. 2008T5KA4L, 2010-2012. This software is
% part of the package "IRMA - Image Reconstruction in Microscopy and Astronomy"
% currently under development within the PRISMA project.
%
% Version: 1.0
% Date:    July 2011
%
% Authors:
%   Riccardo Zanella, Gaetano Zanghirati
%    Dept. of Mathematics, University of Ferrara, Italy
%    riccardo.zanella@unife.it, g.zanghirati@unife.it
%   Roberto Cavicchioli, Luca Zanni
%    Dept. of Pure Appl. Math., Univ. of Modena and Reggio Emilia, Italy
%    roberto.cavicchioli@unimore.it, luca.zanni@unimore.it
%
% Software homepage: http://www.unife.it/irma
%                    http://www.unife.it/prisma/software
%
% Copyright (C) 2011 by R. Cavicchioli, R. Zanella, G. Zanghirati, L. Zanni.
% -------------------------------------------------------------------------
%
% COPYRIGHT NOTIFICATION
%
% Permission to copy and modify this software and its documentation for
% internal research use is granted, provided that this notice is retained
% thereon and on all copies or modifications. The authors and their
% respective Universities makes no representations as to the suitability
% and operability of this software for any purpose. It is provided "as is"
% without express or implied warranty. Use of this software for commercial
% purposes is expressly prohibited without contacting the authors.
%
% This program is free software; you can redistribute it and/or modify it
% under the terms of the GNU General Public License as published by the
% Free Software Foundation; either version 3 of the License, or (at your
% option) any later version.
%
% This program is distributed in the hope that it will be useful, but
% WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
% See the GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License along
% with this program; if not, either visite http://www.gnu.org/licenses/
% or write to
% Free Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
% =========================================================================

flag = 0;

if checkflag == 0  
    
    lambda = 0;                 % initial lambda
    dlambda = 1;                % initial step
    tol_r = 1e-11 * b;          % tolerance on the function
    tol_lam = 1e-11;            % tolerance on the step
    biter = 0;                  % bracketing phase iterations
    siter = 0;                  % secant phase iterations
    maxprojections = 1000;      % maximum number of iterations
    
    % Bracketing Phase
    x_proj = min(max(l,c+q*lambda),u); r = q'*x_proj - b;
    
    % check abs(r) < tol
    if ( abs(r) < tol_r )
        Ivar = zeros(size(x_proj));
        Ivar (x_proj <= l) = -1;
        Ivar (x_proj >= u) = 1;
        Ivar (l == u) = 2;
        flag = 1;
        return;
    end
    
    if r < 0
        lambdal = lambda;
        rl = r;
        lambda = lambda+dlambda;
        x_proj = min(max(l,c+q*lambda),u); r = q'*x_proj - b;
        while r < 0
            biter = biter+1;
            lambdal = lambda;
            s = max(rl/r-1, 0.1);
            dlambda = dlambda+dlambda/s;
            lambda = lambda+dlambda;
            rl = r;
            x_proj = min(max(l,c+q*lambda),u); r = q'*x_proj - b;
        end
        lambdau = lambda;
        ru = r;
    else
        lambdau = lambda;
        ru = r;
        lambda = lambda-dlambda;
        x_proj = min(max(l,c+q*lambda),u); r = q'*x_proj - b;
        while r > 0
            biter = biter+1;
            lambdau = lambda;
            s = max(ru/r-1, 0.1);
            dlambda = dlambda+dlambda/s;
            lambda = lambda-dlambda;
            ru = r;
            x_proj = min(max(l,c+q*lambda),u);  r = q'*x_proj - b;
        end
        lambdal = lambda;
        rl = r;
    end
    
    % check ru and rl
    if (abs(ru) < tol_r)
        x_proj = min(max(l,c+q*lambdau),u);
        if nargout>1
            Ivar = zeros(size(x_proj));
            Ivar (x_proj <= l) = -1;
            Ivar (x_proj >= u) = 1;
            Ivar (l == u) = 2;
            flag = 1;
        end
        return;
    end
    if (abs(rl) < tol_r)
        x_proj = min(max(l,c+q*lambdal),u);

        if nargout>1 
            Ivar = zeros(size(x_proj));
            Ivar (x_proj <= l) = -1;
            Ivar (x_proj >= u) = 1;
            Ivar (l == u) = 2;
            flag = 1;
        end
        return;
    end
    
    % Secant Phase
    s = 1-rl/ru;
    dlambda = dlambda/s;
    lambda = lambdau-dlambda;
    x_proj = min(max(l,c+q*lambda),u); r = q'*x_proj - b;
    maxit_s = maxprojections - biter;
    
    % Main loop
    while ( abs(r) > tol_r && ...
            dlambda > tol_lam * (1 + abs(lambda)) && ...
            siter < maxit_s )
        siter = siter + 1;
        if r > 0
            if s <= 2
                lambdau = lambda;
                ru = r;
                s = 1-rl/ru;
                dlambda = (lambdau-lambdal)/s;
                lambda = lambdau - dlambda;
            else
                s = max(ru/r-1, 0.1);
                dlambda = (lambdau-lambda) / s;
                lambda_new = max(lambda - dlambda, 0.75*lambdal+0.25*lambda);
                lambdau = lambda;
                ru = r;
                lambda = lambda_new;
                s = (lambdau - lambdal) / (lambdau-lambda);
            end
        else
            if s >= 2
                lambdal = lambda;
                rl = r;
                s = 1-rl/ru;
                dlambda = (lambdau-lambdal)/s;
                lambda = lambdau - dlambda;
            else
                s = max(rl/r-1, 0.1);
                dlambda = (lambda-lambdal) / s;
                lambda_new = min(lambda + dlambda, 0.75*lambdau+0.25*lambda);
                lambdal = lambda;
                rl = r;
                lambda = lambda_new;
                s = (lambdau - lambdal) / (lambdau-lambda);
            end
        end
        x_proj = min(max(l,c+q*lambda),u); r = q'*x_proj - b;
    end
    
    if siter == maxit_s
        x_proj = [];
        Ivar = [];
        flag = -1;
        return;
    else
        if nargout>1
            Ivar = zeros(size(x_proj));
            Ivar (x_proj <= l) = -1;
            Ivar (x_proj >= u) = 1;
            Ivar (l == u) = 2;
            flag = 1;
        end
        return;
    end
    
else
    if all(l<=u)
        x_proj = min(max(l,c),u);
        if nargout>1
            Ivar = zeros(size(x_proj));
            Ivar (x_proj <= l) = -1;
            Ivar (x_proj >= u) = 1;
            Ivar (l == u) = 2;
            flag = 1;
        end
        return;
    end
end

x_proj = [];
Ivar = [];
flag = -1;

end
