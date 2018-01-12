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

function [xsol,out_info,it,nprod,flag_sdc,info_proj] = ...
    sdc4dklin(WorkingSet,A,c,qk,x,info,h,m,xi,itmax,nprod,sda,monotone,info_proj)

%==========================================================================
% This procedure computes a solution to the following quadratic programming
% problem
% 
%                       min 0.5 x' A x - c'x
%                       s.t. qk'*x=0 
% 
% by using the SDC gradient method.
% It is customized for the minimization phase of the P2GP for SLBQP problems.
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
% qk      = vector of doubles, coefficients of the linear constraint;
% x       = vector of doubles, starting point;
% info    = [optional] struct variable, containing information on the last
%           step made in a previous call;
% h,m     = integers, parameter for SDC and SDA algorithms, see [1] eq (4.16);
% xi      = double, parameter for the stopping criterion see ref. [1], eq. (4.18);
% itmax   = integer, maximum number of steps;
% nprod   = integer, number of matrix-vector products already performed by P2GP;
% sda     = logical, if true SDA is used instead of SDC;
% monotone = logical, if true the algorithm is forced to be monotone.
% info_proj = [optional] struct variable, containing information on the projection 
%             of problem onto the linear constraint made in a previous call.
%
% Remark: info is used to recover the minimization started in a previous call.
% 
%==========================================================================
%
% OUTPUT ARGUMENTS
%
% xsol    = vector of doubles, computed solution;
% out_info = struct variable, containing information on the last step;
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
% info_proj = struct variable, containing information on the projection of 
%             problem onto the linear constraint.
% 
%==========================================================================

c(~WorkingSet) = 0;
x(~WorkingSet) = 0;
qk(~WorkingSet) = 0;

etagrad = 1e-16;


% construction of rho and w for the unconstrained quadratic problem
if isempty(info_proj)
    ind = find(WorkingSet,1,'first');
    
    rho = -sign(qk(ind))*norm(qk);
    w = qk;
    w(ind) = w(ind) - rho;
    w = w/norm(w);
else
    ind = info_proj.ind;
    rho = info_proj.rho;
    w = info_proj.w;
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


%--------------------------------------------------------------------------
% Initialise procedure
%--------------------------------------------------------------------------
complete = 1; % 1: perform at least one SDC sequence; 0: otherwise
flag_sdc = 0;

nmon_steps = 0;

maxdiff_sd = 0;
maxdiff_c = 0;

if ~isempty(info)
    
    yalpha = info.yalpha;
    sdalpha_old = info.sdalpha_old;
    ngrad = info.ngrad;
    ngrad_old = info.ngrad_old;
    grad = info.grad;
    gg = info.gg;
    PHPg = info.PHPg;
    PHPz = info.PHPz;
    fun = info.fun;
    it = info.last_it;
    itmax = info.last_it+itmax;
    
else
    
    it = 0;
    PHPz = PHP(z);
    nprod = nprod+1;
    grad = PHPz - Pc;
    fun = 0.5*z'*PHPz - z'*Pc;
    
    gg = grad'*grad;
    PHPg = PHP(grad);
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
    
    gAg = grad'*PHPg;
    
    
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
        
        %----------------------------------------------------------------------
        % Update solution, gradient, objective function and iteration counter
        %----------------------------------------------------------------------
        z = z - alfa*grad;
        ngrad_old = ngrad;
        fun_old = fun;
        sdalpha_old = sdalpha;
        
        
        if (mod(it,20) == 0)         % direct computation of Ax and grad
            PHPz = PHP(z);
            nprod = nprod+1;
            grad = PHPz - Pc;
            fun = 0.5*z'*PHPz - z'*Pc;
        else                          % recursive computation of Ax and grad
            PHPz = PHPz - alfa*PHPg;
            grad = grad - alfa*PHPg;
            fun = 0.5*z'*PHPz - z'*Pc;
        end
        
        
        gg = grad'*grad;
        ngrad = sqrt(gg);
        PHPg = PHP(grad);
        nprod = nprod+1;
        
        %------------------------------------------------------------------
        % Check monotonicity
        %------------------------------------------------------------------
        if (fun >= fun_old)
            nmon_steps = nmon_steps+1;
        end
        
        %------------------------------------------------------------------
        % Set stopping condition of inner while loop
        % (fun stagnates, fun-fun_old
        %------------------------------------------------------------------
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
            z = -grad;
            flag_sdc = -10;
        end
    end
    it = it+1;
    
    
end

x = P(z);
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

out_info = struct('yalpha',yalpha,'sdalpha_old',sdalpha_old,'ngrad',ngrad,'ngrad_old',ngrad_old,'grad',grad,'PHPg',PHPg,'PHPz',PHPz,'gg',gg,'last_it',it,'fun',fun);
info_proj = struct('rho',rho,'w',w,'ind',ind);

xsol = x;

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