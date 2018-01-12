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

function [xsol,info,it,nprod,flag_sdc] = ...
    sdc4dk(WorkingSet,A,c,x,info,h,m,xi,itmax,nprod,sda,monotone)

%==========================================================================
% This procedure computes a solution to the following quadratic programming
% problem
% 
%                       min 0.5 x' A x - c'x
% 
% by using the SDC or the SDA gradient method.
% It is customized for the minimization phase of P2GP for BQP problems.
%
% For details see
%
% [A]   R. De Asmundis, D. di Serafino, W.W. Hager, G. Toraldo and H. Zhang,
%       "An efficient gradient method using the Yuan steplength",
%       Computational Optimization and Application, 59 (3), 2014, pp. 541-563, ISSN: 0926-6003
%       DOI: 10.1007/s10589-014-9669-5
% [B]   R. De Asmundis, D. di Serafino, F. Riccio and G. Toraldo,
%       "On spectral properties of steepest descent methods",
%       IMA Journal of Numerical Analysis, 33, 2013, pp. 1416-1435, ISSN: 0272-4979,
%       DOI: 10.1093/imanum/drs056
% 
%  See also
% 
% [C]   R. De Asmundis, D. di Serafino and G. Landi,
%       "On the regularizing behavior of the SDA and SDC gradient methods
%       in the solution of linear ill-posed problems",
%       Journal of Computational and Applied Mathematics, 302, 2016, pp. 81-93, ISSN: 0377-0427,
%       DOI: 10.1016/j.cam.2016.01.007
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
% WorkingSet  = vector of logical, the true entries indicate the variables to
%               which the problem is restricted;
% A       = sparse or dense square matrix, double, Hessian of the objective function;
%           it may also be a handle to a function which computes H*x, where x is a vector,
%           e.g., A = @(x) prodfunc(x,params);
% c       = vector of doubles, coefficients of the linear term of the objective function;
% x       = vector of doubles, starting point;
% info    = [optional] struct variable, containing information on the last
%           step made in a previous call;
% h,m     = integers, parameter for SDC and SDA algorithms, see [1] eq (4.16);
% xi      = double, parameter for the stopping criterion see ref. [1], eq. (4.18);
% itmax   = integer, maximum number of steps;
% nprod   = integer, number of matrix-vector products already performed by P2GP;
% sda     = logical, if true SDA is used instead of SDC;
% monotone = logical, if true the algorithm is forced to be monotone.
%
% Remark: info is used to recover the minimization started in a previous call.
% 
%==========================================================================
%
% OUTPUT ARGUMENTS
%
% xsol    = vector of doubles, computed solution;
% info    = struct variable, containing information on the last step;
% it      = integer, number of SDC/SDA iteration;
% nprod   = integer, number of matrix-vector products performed;
% flag_sdc = integer, information on the execution
%         -10 - P2GP found a descent direction with nonpositive curvature,
%               this direction is provided as the output x;
%           0 - the algorithm found a point satisfying the stopping criterion,
%           1 - the algorithm found a stationary point for the problem,
%           2 - the algorithm found a point which increases the obj fun value,
%           3 - the did not find a stationary point and itmax steps have
%               been executed.
% 
%==========================================================================

% Initialise procedure
complete = 1; % k: perform at least k SDC/SDA sequence; 0: otherwise
flag_sdc = 0;

nmon_steps = 0;

maxdiff_sd = 0;
maxdiff_c = 0;

etagrad = 1e-16;

c(~WorkingSet) = 0;
x(~WorkingSet) = 0;

prod = @(v) res_hessprod(v,A,WorkingSet);


if ~isempty(info)
    
    yalpha = info.yalpha;
    sdalpha_old = info.sdalpha_old;
    ngrad = info.ngrad;
    ngrad_old = info.ngrad_old;
    grad = info.grad;
    gg = info.gg;
    Ag = info.Ag;
    Hx = info.Ax;
    fun = info.fun;
    it = info.last_it;
    itmax = info.last_it+itmax;
    
else
    
    it = 0;
    Hx = prod(x);
    nprod = nprod+1;
    grad = Hx-c;
    fun = 0.5*x'*Hx-c'*x;
    
    gg = grad'*grad;
    Ag = prod(grad);
    nprod = nprod+1;
    ngrad = sqrt(gg);
    
end




%--------------------------------------------------------------------------
% Check value of h
%--------------------------------------------------------------------------
if (h < 2)
    error('\nh = %d, but h must be >= 2\n',h);
end

%--------------------------------------------------------------------------
% Set some values used in the main loop
%--------------------------------------------------------------------------
modulo = m + h;

cont_cond = 1;

%--------------------------------------------------------------------------
% SDC/SDA main loops
% outer loop: ensure that a descent direction is found
% inner loop: perform SDC iterations until
%             diff <= eta*maxdiff and fun_old-fun > 0 and it > maxit
%--------------------------------------------------------------------------

exit = 0;

while ~exit && (it <= itmax) && (cont_cond)
    
    %----------------------------------------------------------------------
    % Compute steplength
    %----------------------------------------------------------------------
    
    gAg = grad'*Ag;
    
    if gAg > eps
        
        sdalpha = gg/(gAg);
        
        if(mod(it,modulo) < h)
            
            alfa = sdalpha;
            
        elseif(mod(it,modulo) == h)
            
            if sda
                yalpha = 1./ (1./sdalpha_old + 1./sdalpha);
                alfa = yalpha;
            else
                % Yuan steplength
                yalpha = 2./ ( sqrt((1./sdalpha_old - 1./sdalpha)^2 + 4*gg/(sdalpha_old*ngrad_old)^2) + (1./sdalpha_old + 1./sdalpha) );
                alfa = yalpha;
            end
            
        else
            
            if (monotone)
                yalpham = min(yalpha,2*sdalpha);
                alfa = yalpham;
            else
                alfa = yalpha;
            end
            
        end
        
        % Update solution, gradient, objective function and iteration counter
        x = x - alfa*grad;
        ngrad_old = ngrad;
        fun_old = fun;
        sdalpha_old = sdalpha;
        
        
        if (mod(it,20) == 0)         % direct computation of Ax and grad
            Hx = prod(x);                % after each 25 steps
            grad = Hx-c;
            fun = 0.5*x'*Hx-c'*x;
            
            nprod = nprod+1;
        else                          % recursive computation of Ax and grad
            Hx = Hx - alfa*Ag;
            grad = grad - alfa*Ag;
            fun = 0.5*x'*Hx-c'*x;
        end
        
        
        gg = grad'*grad;
        ngrad = sqrt(gg);
        Ag = prod(grad);
        nprod = nprod+1;
        
        % Check monotonicity
        if (fun >= fun_old)
            nmon_steps = nmon_steps+1;
        end
        
        diff = fun_old-fun;
       
        if diff <0
            cont_cond = 1;
        else
            if(mod(it,modulo) < h)
                cont_cond = (diff > xi*maxdiff_sd) || (it < (h+m)*complete);
                maxdiff_sd = max(diff,maxdiff_sd);
                
            elseif(mod(it,modulo) == h)
                cont_cond = 1;
            else
                cont_cond = (diff > xi*maxdiff_c) || (it < (h+m)*complete);
                maxdiff_c = max(diff,maxdiff_c);
            end
        end
        
    else
        exit = 1;
        if gg > eps
            x = -grad;
            flag_sdc = -10;
        end
    end
    
    it = it+1;
    
end

decr = (-c'*x) < 0;
if flag_sdc >= 0
    if (ngrad < etagrad)
        flag_sdc = 1;
    elseif ~decr
        flag_sdc = 2;
    end
    if (it >= itmax && ngrad >= etagrad)
        flag_sdc = 3;
    end
end


if gAg>0
    info = struct('yalpha',yalpha,'sdalpha_old',sdalpha_old,'ngrad',ngrad,'ngrad_old',ngrad_old,'grad',grad,'Ag',Ag,'Ax',Hx,'gg',gg,'last_it',it,'fun',fun);
else
    info = struct([]);
end

xsol = x;

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
