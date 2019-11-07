% ------------------------------------------------------------------------------
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
% ------------------------------------------------------------------------------

function [xmin,Ivar,fmin,gmin,it,nprod,nproj,alfa,info,stepkind] = ...
    linesearch1(H,c,xk,fk,gk,dk,l,u,q,b,checkflag,mu,it_pg,maxgp,RelaxedStep,GPType,LSType,info,nprod,nproj)

%==========================================================================
% This function performs the projected line search used by P2GP in
% the identification phase.
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
% xk      = vector of doubles, current point;
% fk      = double, obj fun value at current point;
% gk      = vector of doubles, gradient at current point;
% dk      = vector of doubles, line-search direction;
% l,u     = vectors of doubles, upper and lower bounds on the variables;
% q       = vector of doubles, coefficients of the linear constraint;
% b       = double, right-hand side of the linear constraint;
% checkflag = integer, used in the call to the projection function;
%             its value is 1 if the problem is BQP, 0 if the problem is SLBQP;
% mu      = double, coefficient for the Armijo sufficient decrease condition,
%           see mu_1 in ref. [1], eq. (4.1);
% it_pg   = integer, indicates the index of the current PG iteration,
%           used for the choice of the steplength;
% maxgp   = integer, maximum number of consecutive GP steps, used in case
%           of step relaxation;
% RelaxedStep = logical, if true use steplength relaxaion for GP;
% GPType  = integer, type of Gradient Projection (GP) algorithm used in
%              the identification phase, values 0, 1, 2, 3 or >3  [default 4]
%               0 - GP with Cauchy steplength,
%               1 - Projected Barzilai-Borwein (PBB) with steplength BB1,
%               2 - PBB with steplength BB2,
%               3 - Projected Adaptive BB (PABBmin) with
%                   variable threshold tau and memory = 3,
%              >3 - PABBmin with fixed threshold tau = 0.2 and memory = 3;
% LSType  = integer, type of linesearch used [default 1]
%               0 - linesearch over the feasible direction
%                   (see. Bertsekas `Nonlinear Programming', 1999, section 2.3.1),
%              >0 - linesearch along the projection arc;
% info    = struct variable, containing information for the computation of the steplength;
% nprod   = integer, number of matrix-vector products already performed by P2GP;
% nproj   = integer, number of calls to the projection routine already performed by P2GP;
%
%==========================================================================
%
% OUTPUT ARGUMENTS
%
% xmin      = vector of doubles, computed solution;
% Ivar      = integer vector with the same length as xmin,
%            -1 - the corresponding entry of xmin is on its lower bound,
%             0 - the corresponding entry of xmin is free (l(i) < xmin(i) < u(i)),
%             1 - the corresponding entry of xmin is on its upper bound,
%             2 - the corresponding entry of xmin is fixed because l(i)=u(i);
% fmin    = double, obj fun value at xmin;
% gmin    = vector of doubles, gradient at xmin;
% it      = integer, number of line search iterations;
% nprod   = integer, number of matrix-vector products performed;
% nproj   = integer, number of calls to the projection routine;
% alfa    = double, step taken along dk to obtain xmin;
% info    = struct variable, containing information for the computation of the steplength;
% stepkind = integer, information about the execution
%            0 - linesearch performed over dk,
%            1 - maximum feasible step taken along the projection of dk onto the tangent cone,
%            2 - linesearch performed over the projection of dk onto the tangent cone,
%            6 - the problem is unbounded from below.
% 
%==========================================================================


NegCurveGrad = false;
MaxFeasStep = false;
linesearch = 1;
it = 0;
maxit = min(10,max(5,ceil(length(xk))));

if GPType==4
    tau = 0.2;
    mem = 3;
elseif it_pg>=2
    tau = info.tau;
    mem = info.mem;
end

% Compute the first steplenght according to the value of BBStepsize
dphi0 = gk'*gk;

if ~GPType % Cauchy Steepest Descent
    if isa(H,'function_handle')
        Agk = H(gk);
    else
        Agk = H*gk;
    end
    nprod = nprod+1;
    gAg = gk'*Agk;
    alfa_sd = dphi0/(gAg);  %Cauchy step
    if (RelaxedStep == 1)                     % relaxed step
        if (mod(it_pg,2)==0)
            alfa = 2*((it_pg/maxgp)*0.8+0.2)*alfa_sd;
        else
            alfa = 2*(((maxgp-it_pg)/maxgp)*0.8+0.2)*alfa_sd;
        end
    else
        alfa = alfa_sd;
    end
    if gAg <= 0
        NegCurveGrad = true;
    end
    
else
    if it_pg >=2
        sk = info.s_k;
        yk = info.y_k;
    else
        sk = [];
        yk = [];
    end
    if it_pg<2 || all(yk==0) || all(sk==0)
        if isa(H,'function_handle')
            Agk = H(gk);
        else
            Agk = H*gk;
        end
        nprod = nprod+1;
        gAg = gk'*Agk;
        alfa = dphi0/(gAg);  %Cauchy step
        if gAg <= 0
            NegCurveGrad = true;
        end
    else
        sk = info.s_k;
        yk = info.y_k;
        sty = sk'*yk;
        if GPType==1 
            alfa = (sk'*sk)/sty; %BB1
        elseif GPType==2
            alfa = sty/(yk'*yk); %BB2
        else %PABBmin method
            BB1 = (sk'*sk)/sty; %BB1
            BB2 = sty/(yk'*yk); %BB2
            
            if mem>1
                info.bb2mem(rem(it_pg,mem)+1) = BB2;
                BB2min = min(info.bb2mem);
            else
                BB2min = BB2;
            end
            
            if BB2 < tau*BB1
                alfa = BB2min; %BB1
                if GPType==3
                    tau = tau*0.9;
                end
            else
                alfa = BB1; %BB2
                if GPType==3
                    tau = tau*1.1;
                end
            end
            
            if alfa < eps
                    NegCurveGrad = true;
            end
            
        end
    end
end

stepkind = 0;

if NegCurveGrad
    Ivar = zeros(size(xk));
    Ivar(xk <= l) = -1;
    Ivar(xk >= u) = 1;
    Ivar(l == u) = 2;
    
    [projdk] = projgrad(gk,Ivar,q,checkflag);
    nproj = nproj + 1;
    
    if isa(H,'function_handle')
        Apdk = H(projdk);
    else
        Apdk = H*projdk;
    end
    tempdAd = projdk'*Apdk;
    nprod = nprod +1;
    
    MaxFeasStep = tempdAd<eps;
end

if NegCurveGrad
    if MaxFeasStep
        ind1 = find(projdk<0);
        ind2 = find(projdk>0);
        
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
        alfavect = sort(alfavect(alfavect>0));
        if (max(alfavect) == Inf)
            fprintf('\nThe problem is unbounded from below.\n');
            stepkind = 6;
            xmin = xk;
            Ivar = [];
            fmin = [];
            gmin = gk;
            alfa = Inf;
            info = [];
            return;
        end

        if ~isempty(alfavect)
            alfafeas = alfavect(end);
            alfaind  = length(alfavect);
            linesearch = 3;
        else
            alfafeas = [];
        end
        
        if ~isempty(alfafeas)
            alfa = alfafeas;
        else
            alfa = 0;
        end
        dk = projdk;
        dphi0 = -gk'*dk;
        stepkind = 1;

    else
        alfa = -(gk'*projdk)/tempdAd;
        linesearch = 2;
        stepkind = 2;
    end
    if it_pg>=2
        info.tau = -1;
    end
else
    if it_pg>=2
        info.tau = tau;
    end
end

% Steplenght reduction iterations
e = dphi0*alfa;
if ~isnan(alfa)
    xmin = xk+alfa*dk;
else
    xmin = xk;
end
[xmin,Ivar] = simproj(xmin,l,u,q,b,checkflag);
nproj = nproj + 1;

if LSType>0
    if isa(H,'function_handle')
        Ax = H(xmin);
    else
        Ax = H*xmin;
    end
    nprod = nprod+1;
    fmin = 0.5*(xmin'*Ax)-c'*xmin;
    gdx = gk'*(xmin-xk);
else
    dk = xmin-xk;
    gdx = gk'*(dk);
    dphi0 = -gdx;
    alfa = 1;
    linesearch = 1;

    if isa(H,'function_handle')
        Ad = H(dk);
    else
        Ad = H*dk;
    end
    nprod = nprod+1;
    Ax0 = gk + c;
    fmin = fk + dk'*(0.5*Ad + gk);
end
sigma1 = 1e-2;
sigma2 = 1/2;

if ~isnan(alfa) && linesearch
    while (fmin > fk+mu*gdx) && (it < maxit)
        if LSType>0
            if linesearch == 1
                alfa1 = (dphi0*(alfa^2))/(2*(fmin-fk+e));
                alfa1 = max(sigma1*alfa,min(sigma2*alfa,alfa1));
            else
                if linesearch == 2
                    alfa1 = alfa*sigma2;
                else
                    alfaind = ceil(alfaind/2);
                    alfa1 = alfavect(alfaind);
                end
            end
            xmin = xk+alfa1*dk;
            [xmin,Ivar] = simproj(xmin,l,u,q,b,checkflag);
            nproj = nproj + 1;
        else
            alfa1 = alfa*sigma2;
            xmin = xk+alfa1*dk;
            Ivar = zeros(size(xmin));
            Ivar(xmin<=l) = -1;
            Ivar(xmin>=u) = 1;
            Ivar(l==u) = 2;
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
        gdx = gk'*(xmin-xk);
        e = alfa1*gdx;
        alfa = alfa1;
        it = it+1;
    end
    
    if it >= maxit
        if linesearch==3
            alfa = alfavect(1);
            xmin = xk+alfa*dk;
        else
            if ~NegCurveGrad
                projdk = projgrad(gk,Ivar,q,checkflag);
                nproj = nproj + 1;
                if isa(H,'function_handle')
                    Apdk = H(projdk);
                else
                    Apdk = H*projdk;
                end
                tempdAd = projdk'*Apdk;
                nprod = nprod+1;
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
            
            if tempdAd > 0
                alfa_cauchy = -(gk'*projdk)/tempdAd;
            else
                alfa_cauchy = Inf;
            end
            
            if isempty(minbreak)
               if tempdAd > 0
                   alfa = alfa_cauchy;
               else
                   alfa = 0;
               end
            else
                alfa = min(minbreak,alfa_cauchy);
            end
            xmin = xk + alfa*projdk;
        end
        Ivar = zeros(size(xmin));
        Ivar(xmin<=l) = -1;
        Ivar(xmin>=u) = 1;
        Ivar(l==u) = 2;
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