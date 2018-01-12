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

function [xk,fk,gk,pgnorm,nprod,nproj,flag,otherinfo] ...
    = p2gp(H,c,l,u,q,b,xk,options)

%==========================================================================
% This code computes an approximate solution (or stationary point)
% of the constrained quadratic programming problem
%
%               min (1/2) x' H x - c'x
%               s.t. l <= x <= u
%                    q'x = b
%
% using the P2GP algorithm. H is not required to be positive definite.
%
% The algorithm usually stops when
%
%              PGnorm <= max(AbsTol, RelTol*PGnorm0),              (*)
%
% where PGnorm and PGnorm0 are the projected gradient norm at the current
% point and at the starting point, respectively, or when the problem is
% discovered to be unbounded.
% See the output parameter info for other situations where P2GP may stop.
%==========================================================================
%
% Authors:
%   Daniela di Serafino (daniela.diserafino@unicampania.it),
%   Gerardo Toraldo (toraldo@unina.it),
%   Marco Viola (marco.viola@uniroma1.it)
% 
% Version: 1.0
% Last Update: January 12, 2018
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
% l,u     = vectors of doubles, upper and lower bounds on the variables (-inf and +inf
%           must be specified for variables that are unbounded below or above);
% q       = vector of doubles, coefficients of the linear constraint;
% b       = double, right-hand side of the linear constraint;
% xk      = [optional] vector of doubles, starting point (not required to be feasible);
%
% options = [optional] struct variable with the following (possible) entries,
%           to be specified as pairs ('propertyname', propertyvalue):
%           AbsTol      : double, absolute tolerance in the stopping criterion [default 1e-6];
%           RelTol      : double, relative tolerance in the stopping criterion [default 1e-6];
%           InfNorm     : logical, if true the inf-norm is used in the
%                         stopping criterion, if false the 2-norm is used [default false];
%           MaxProd     : integer, maximum number of matrix-vector products [default 30000];
%           MaxProj     : integer, maximum number of projections
%                         on the feasible set[default 30000];
%           MaxOutIt    : integer, maximum number of outer iterations [default 2000];
%           MaxTotGP    : integer, maximum number of GP steps [default 30000];
%           MaxTotMin   : integer, maximum number of steps of
%                         the minimization phase algorithm [default 30000];
%           GPType      : integer, type of Gradient Projection (GP) algorithm used in
%                         the identification phase, values 0, 1, 2, 3 or >3  [default 4]
%                          0 - GP with Cauchy steplength,
%                          1 - Projected Barzilai-Borwein (PBB) with steplength BB1,
%                          2 - PBB with steplength BB2,
%                          3 - Projected Adaptive BB (PABBmin) with
%                              variable threshold tau and memory = 3,
%                         >3 - PABBmin with fixed threshold tau = 0.2 and memory = 3;
%           RelaxedStep : logical, if true use steplength relaxaion for GP [default false];
%           MaxGP       : integer, maximum number of consecutive GP steps [default 50];
%           Eta         : double, parameter for GP stopping criterion,
%                         see ref. [1], eq. (4.6) [default 0.1];
%           Mu          : double, coefficient for the Armijo sufficient decrease
%                         condition, see mu_1 in ref. [1], eq. (4.1) [default 1e-4];
%           UncMinType  : integer, values 0, 1, 2 or >2 [default 3]
%                          0 - skip the minimization phase (P2GP becomes a GP method)
%                          1 - use SDC in the minimization phase,
%                          2 - use SDA in the minimization phase,
%                         >2 - use CG in the minimization phase;
%           MaxStepMin  : integer, maximum number of consecutive steps of
%                         the minimization phase algorithm [default 50];
%           Xi          : double, parameter for the stopping criterion in
%                         the minimization phase, see ref. [1], eq. (4.18) [default 0.5];
%           MaxRestart  : integer, maximum number of consecutive calls to the
%                         SDC/SDA/CG algorithm in the minimization phase [default 100];
%           Proportioning : logical, if true the minimization phase is stopped
%                         considering the proportionality criterion, if false
%                         the minimization phase is stopped considering the bindingness
%                         of the current face of the feasible set [default true];
%           Gamma       : double, starting value of Gamma for the proportionality
%                         criterion [defaut 1];
%           AdaptGamma  : logical, if true Gamma is adaptively modified during
%                         the execution of P2GP [default true];
%           H_sdc       : integer, parameter for SDC and SDA, see [1] eq (4.16) [default 6];
%           M_sdc       : integer, parameter for SDC and SDA, see [1] eq (4.16) [default 4];
%           Monotone    : logical, if true forces monotonicity of SDC and SDA [default false];
%           Verbosity   : logical, if true details on the computation are shown [default false];
%           PartialResults : logical, if true partial results about convergence
%                            are printed [default false];
%           GeneratePlots : logical, if true plots of obj fun value, obj fun variation
%                         and projected gradient norm are generated [default false];
%
%==========================================================================
%
% OUTPUT ARGUMENTS
%
% xk      = vector of doubles, computed solution;
% fk      = double, objective function value at xk;
% gk      = vector of doubles, gradient at xk;
% pgnorm  = double, projected gradient norm at xk
%           (inf-norm or 2-norm according to the value of the 'InfNorm' parameter);
% nprod   = integer, number of matrix-vector products performed;
% nproj   = integer, number of calls to the projection routine;
% flag    = integer, information on the execution
%           0 - P2GP found a point satisfying the stopping criterion (*) (see line 16),
%           1 - the problem is not strictly convex, P2GP found a point satisfying
%               the stopping criterion (*),
%           2 - the stopping criterion (*) was not satisfied,
%               P2GP stopped because nprod > MaxProd
%           3 - the stopping criterion (*) was not satisfied,
%               P2GP stopped because nproj > MaxProj
%           4 - the stopping criterion (*) was not satisfied,
%               P2GP stopped because itergp > MaxTotGP
%           5 - the stopping criterion (*) was not satisfied,
%               P2GP stopped because itermin > MaxTotMin
%           6 - P2GP found that the problem is unbounded below,
%
% otherinfo = struct variable with the following entries
%             ivar        : integer vector with the same length as xk,
%                           -1 - the corresponding entry of xk is on its lower bound,
%                            0 - the corresponding entry of xk is free (l(i) < xk(i) < u(i)),
%                            1 - the corresponding entry of xk is on its upper bound,
%                            2 - the corresponding entry of xk is fixed because l(i)=u(i);
%             nact        : integer, number of active variables at xk;
%             nact0       : integer, number of active variables at the starting point;
%             funvalues   : vector of doubles, history of the obj fun;
%             pgradnorm   : vector of doubles, history of the projected gradient norm
%                           (2-norm or inf-norm, according to the value of InfNorm parameter);
%             iter        : integer, total number of outer iterations,
%             itergp      : integer, total number of GP steps;
%             itridgp     : integer, total number of obj fun reductions in
%                           the line searches after GP steps;
%             nmaxgp      : integer, total number of times GP has reached
%                           the maximum number of consecutive GP steps;
%             itermin     : integer, total number of steps of
%                           the minimization phase algorithm (CG/SDC/SDA);
%             itridmin    : integer, total number of obj fun reductions
%                           in the line searches of the minimization phase;
%             nmaxstepmin : integer, total number of times the minimization
%                           phase algorithm (CG/SDC/SDA) has reached
%                           the maximum number of consecutive steps;
%             callmin     : integer, total number of calls to the CG/SDC/SDA algorithm
%                           in the minimization phase;
%             nrest       : integer, total number of restarts of the minimization phase
%                           (a restart is when CG/SDC/SDA is called again
%                           without performing GP steps);
%             nprop       : integer, number of times the minimization phase
%                           continued on a smaller face because the
%                           proportionality condition was satisfied;
%             gamma       : double, final value of the parameter Gamma
%                           for the proportionality condition;
%             nondecr     : integer, total number of times SDC/SDA produces
%                           a non-descent direction (if Monotone == true this cannot happen);
%             exitphase   : string, indicates whether the algorithm exited after
%                           the identification or the minimization phase;
%             ngradpro    : integer, number of direct computations of
%                           the projected gradient (info for code developers);
%             time        : double, total execution time in seconds (computed using tic and toc);
%==========================================================================

if nargin < 6
    error('The first five input arguments must be provided\n');
end
if nargin < 8
    options = struct([]);
end
if isempty(H)
    error('Enter Hessian matrix\n');
end
if isempty(c)
    error('Enter vector c\n');
end
n = length(c); % Dimension of the problem

%% Check the feasibility of the problem
[checkflag,l,u] = checkfeas(n,l,u,q,b);

%% Set the starting point if not defined
if nargin < 7 || isempty(xk)
    fprintf('Starting point is set to automatically\n');

    infl = isinf(l);
    infu = isinf(u);
    
    xk = zeros(size(c));

    lumean = ~infl & ~infu;
    xk(lumean) = (l(lumean)+u(lumean))/2;

    indinflow = infl & ~infu;
    xk(indinflow) = u(indinflow)-1;
    
    indinfup = infu & ~infl;
    xk(indinfup) = l(indinfup)+1;
end


%% Default parameters

AbsTol = 1e-6;
RelTol = 1e-6;
InfNorm = false;
MaxProd = 30000;
MaxProj = 30000;
MaxOutIt = 2000;
MaxTotGP = 30000;
MaxTotMin = 30000;
GPType = 4;
RelaxedStep = false;
MaxGP = 50;
Eta = 0.1;
Mu = 1e-4;
UncMinType = 3;
MaxStepMin = 50;
Xi = 0.5;
MaxRestart = 100;
Proportioning = true;
Gamma = 1;
AdaptGamma = true;
H_sdc = 6;
M_sdc = 4;
Monotone = false;
Verbosity= false;
PartialResults = false;
GeneratePlots = false;


%% Set user options
optionnames = fieldnames(options);
for i=1:numel(optionnames)
    switch upper(optionnames{i})
        case 'ABSTOL'
            AbsTol = options.(optionnames{i});
        case 'RELTOL'
            RelTol = options.(optionnames{i});
        case 'INFNORM'
            InfNorm = options.(optionnames{i});
        case 'MAXPROD'
            MaxProd = options.(optionnames{i});
        case 'MAXPROJ'
            MaxProj = options.(optionnames{i});
        case 'MAXOUTIT'
            MaxOutIt = options.(optionnames{i});
        case 'MAXTOTGP'
            MaxTotGP = options.(optionnames{i});
        case 'MAXTOTMIN'
            MaxTotMin = options.(optionnames{i});
        case 'GPTYPE'
            GPType = options.(optionnames{i});
        case 'RELAXEDSTEP'
            RelaxedStep = options.(optionnames{i});
        case 'MAXGP'
            MaxGP = options.(optionnames{i});
        case 'ETA'
            Eta = options.(optionnames{i});
        case 'MU'
            Mu = options.(optionnames{i});
        case 'UNCMINTYPE'
            UncMinType = options.(optionnames{i});
        case 'MAXSTEPMIN'
            MaxStepMin = options.(optionnames{i});
        case 'XI'
            Xi = options.(optionnames{i});
        case 'MAXRESTART'
            MaxRestart = options.(optionnames{i});
        case 'PROPORTIONING'
            Proportioning = options.(optionnames{i});
        case 'GAMMA'
            Gamma = options.(optionnames{i});
        case 'ADAPTGAMMA'
            AdaptGamma = options.(optionnames{i});
        case 'H_SDC'
            H_sdc = options.(optionnames{i});
        case 'M_SDC'
            M_sdc = options.(optionnames{i});
        case 'MONOTONE'
            Monotone = options.(optionnames{i});
        case 'VERBOSITY'
            Verbosity = options.(optionnames{i});
        case 'PARTIALRESULTS'
            PartialResults = options.(optionnames{i});
        case 'GENERATEPLOTS'
            GeneratePlots = options.(optionnames{i});
        otherwise
            error(['Unrecognized option: ''' optionnames{i} '''']);
    end
end


%% Initialization
iter = 0;
nprod = 0;
nproj = 0;
ngradpro = 0;
itergp = 0;
itridgp = 0;
nmaxgp = 0;
callmin = 0;
itermin = 0;
itridmin = 0;
nrest = 0;
nmaxstepmin = 0;
nprop = 0;
exitphase = 'none';
nondecr = 0;
funvalues = zeros(MaxProd,1);
indfunval = 1;
pgradnorm = zeros(MaxProd,1);
indpgrad = 1;

% Time measure starts after initialization
startingtime = tic;

if checkflag == -1
    
    error('\nThe feasible set is empty\n');
    
elseif checkflag == 2
    
%% If the problem is unconstrained, then CG is used
    fprintf('The problem is unconstrained, using CG to find the solution\n')
    [xk,fk,gk,pgnorm,iter,nprod,flag,funvalues,indfunval,pgradnorm,indpgrad,tol] = ...
       gradcon(H,c,xk,nprod,MaxProd,AbsTol,RelTol,funvalues,indfunval,pgradnorm,indpgrad,InfNorm);
    time = toc(startingtime);
    nact0 = 0;
    nact = 0;
    ivar = zeros(size(xk));

    if flag < 0
        fprintf('\nThe problem is unbounded from below.\n');
        flag = 6;
    end
    
else

    %% P2GP starts
    
    % The starting point is projected on the feasible set
    [xk,ivar,flag] = simproj(xk,l,u,q,b,checkflag);
    nproj = nproj + 1;

    if (flag==-1)
        % If flag=-1 projection has failed because the feasible set is empty
        % the algorithm is stopped
        error('\nThe feasible set is empty\n');
    else
        flag = 0;
        
        if isa(H,'function_handle')
            Hxk = H(xk);
        else
            Hxk = H*xk;
        end
        nprod = nprod + 1;
        fk = 0.5*(xk'*Hxk)-c'*xk;
        funvalues(indfunval) = fk;
        indfunval = indfunval + 1;
        
        gk = Hxk-c;
        
        nact0 = length(find(ivar~=0)); %initial active set
        
        % Compute the projected gradient and its norm at the (projected)
        % starting point
        if checkflag==0
            aF = q(ivar==0);
            aFaF = aF'*aF;
            if aFaF > 0
                rho_f = (gk(ivar==0)'*aF)/(aFaF);
                hk = gk - q*rho_f;
            else
                hk = gk;
            end
        else
            hk = gk;
        end
        free_vars = (ivar==0);
        phi = hk.*free_vars;
        
        chop_vars = (((ivar==-1) & (hk<0)) | ((ivar==1) & (hk>0)));
        
        if sum(chop_vars)~=0
            pgrad = projgrad(gk,ivar,q,checkflag);
            ngradpro = ngradpro + 1;
            nproj = nproj+1;
        else
            % If the active set is binding, projected gradient = -phi
            pgrad = -phi;
        end

        if InfNorm
            pgnorm = norm(pgrad,Inf);
        else
            pgnorm = norm(pgrad);
        end

        pgnorm0 = pgnorm;

        dk = -gk;  % Set the negative gradient as search direction
        
        % Compute the tolerance for the stopping criterion        
        [tol,whichtol] = max([AbsTol,RelTol*pgnorm0]);

        % Vector of partial tolerances for PartialResults
        tolvector = [1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9];
        if whichtol ==1
            parttolvector = tolvector; 
        else
            parttolvector = tolvector*pgnorm0;
        end
        tolindex = 1;
        
        pgradnorm(indpgrad) = pgnorm;
        indpgrad = indpgrad + 1;
        
        if(Verbosity)
            fprintf('\n***** Initial obj fun value = %d    initial PGnorm = %d',fk,pgnorm0);
        end
        
        % Modify maxgp and eta if "pure" GP has been chosen        
        if ( ~UncMinType )
            MaxGP = min(MaxProd,MaxProj);
            Eta = 0.0;
        end
        
        % skipmin = 1 is used to skip the minimization phase in certain
        % cases
        skipmin = 0;
        % realmaxgp is needed to let P2GP continue the identification phase 
        % when the problem is non-strictly convex and a maximum feasible
        % step with negative curvature is taken
        realmaxgp = MaxGP;

        while ((pgnorm>tol) && (iter<MaxOutIt) && (nprod<MaxProd) && ...
               (nproj<MaxProj) && (itergp<MaxTotGP) && (itermin<MaxTotMin))
            
            if skipmin == 0
                iter = iter+1;
                MaxGP = realmaxgp;
            else
                MaxGP = MaxGP - it_gp;
            end
            %--------------------------------------------------------------------
            % Start of the GP phase
            %
            % The GP phase stops if one of the following conditions holds:
            % - the projected gradient satisfies stopping criterion (*)
            %   (see line 16)
            % - the active-set has not changed (chng_act = 0),
            % - there is no progress in reducing the obj fun (see [1] eq (4.6)),
            % - the maximum number of consecutive GP steps has been reached.
            %--------------------------------------------------------------------
            
            chng_act = 1;
            objfun_red = 1;
            
            it_gp = 0;
            maxdiff = 0;
            
            x_old = xk;
            g_old = gk;
            fval_old = fk;
            
            min_pgiter = 0;
            if GPType>=3
                % At least 5 consecutive GP steps must be executed since PABB memory is 3
                min_pgiter = 5;
            end
            
            stop = 0;
            skipmin = 0;

            while ~stop && ((pgnorm>tol) && (nprod<MaxProd) && (nproj<MaxProj) ...
                    && (itergp<MaxTotGP) && (itermin<MaxTotMin)) && ...
                    (((chng_act) && (objfun_red) && (it_gp<MaxGP)) || ...
                    (it_gp < min_pgiter))
              
                exitphase = 'identification';
                
                it_gp = it_gp+1;
                
                if it_gp<2
                    info = [];
                elseif it_gp==2
                    info = struct('s_k',xk-x_old,'y_k',gk-g_old,'bb2mem',[],'tau',0.5,'mem',3);
                    info.bb2mem = Inf(info.mem,1);
                else
                    info.s_k = xk-x_old;
                    info.y_k = gk-g_old;
                end
                
                x_old = xk;
                g_old = gk;
                
                % Projected line search
                [xk,ivar_new,fmin,gk,it_int,nprod,nproj,alfa,info,stepkind] = ...
                    linesearch1(H,c,xk,fk,gk,dk,l,u,q,b,checkflag,Mu,it_gp,MaxGP,RelaxedStep,GPType,info,nprod,nproj);
                itridgp = itridgp+it_int;
                
                if stepkind>5
                    flag = stepkind;
                    nact = size(find(ivar~=0),1);
                    time = toc(startingtime);
                    funvalues = funvalues(1:indfunval);
                    pgradnorm = pgradnorm(1:indpgrad);
                    if nargout>7
                        otherinfo = struct('ivar',ivar,'nact',nact,'nact0',nact0,'funvalues',funvalues,'pgradnorm',pgradnorm,...
                            'iter',iter,'itergp',itergp,'itridgp',itridgp,'nmaxgp',nmaxgp,...
                            'itermin',itermin,'itridmin',itridmin,'nmaxstepmin',nmaxstepmin,'callmin',callmin,'nrest',nrest,...
                            'nprop',nprop,'gamma',Gamma,'nondecr',nondecr,'exitphase',exitphase,'ngradpro',ngradpro,'time',time);
                    end
                    return;
                end
                
                diff = fk-fmin;
                objfun_red = (diff > Eta*maxdiff);
                if ~UncMinType
                    objfun_red = objfun_red || diff<0;
                end
                maxdiff = max(diff,maxdiff);
                fk = fmin;
                
                chng_act =  any(ivar ~= ivar_new);  % Active-set variation
                ivar = ivar_new;
                
                % Ignore the stagnation of the active set in case of pure GP 
                if ( ~UncMinType )
                    chng_act = 1;
                end
                
                % Compute the projected gradient and its norm
                if checkflag==0
                    aF = q(ivar==0);
                    aFaF = aF'*aF;
                    if aFaF > 0
                        rho_f = (gk(ivar==0)'*aF)/(aFaF);
                        hk = gk - q*rho_f;
                    else
                        hk = gk;
                    end
                else
                    hk = gk;
                end
                free_vars = (ivar==0);
                phi = hk.*free_vars;

                chop_vars = (((ivar==-1) & (hk<0)) | ((ivar==1) & (hk>0)));

                if sum(chop_vars)~=0
                    pgrad = projgrad(gk,ivar,q,checkflag);
                    ngradpro = ngradpro + 1;
                    nproj = nproj + 1;
                else
                    pgrad = -phi;
                end

                if InfNorm
                    pgnorm = norm(pgrad,Inf);
                else
                    pgnorm = norm(pgrad);
                end

                dk = -gk;
                
                funvalues(indfunval) = fk;
                indfunval = indfunval + 1;
                pgradnorm(indpgrad) = pgnorm;
                indpgrad = indpgrad + 1;
                
                if(Verbosity)
                    fprintf('\n *****\nit_gp = %d \n #reduction = %d - alpha = %d\n  fval = %d \n fval_old-fval = %d \n |x_old-xk|_2 = %d\n pgnorm = %d\n #active = %d', ...
                            it_gp,it_int,alfa,fk,(fval_old-fk),norm(x_old-xk),pgnorm,sum(ivar~=0));
                end
                
                fval_old = fk;
                
                if isempty(info)
                    maxfeas = false;
                else
                    maxfeas = (info.tau < 0);
                end
                
                if (isnan(alfa) || maxfeas)
                    stop = 1;
                    skipmin = 1;
                end
            end
            
            if (it_gp >= MaxGP)
                nmaxgp = nmaxgp+1;
            end
            
            itergp = itergp+it_gp;

            if(Verbosity)
                if(~chng_act)
                    fprintf('\n Identification Phase terminates: active-set has not changed - GPsteps = %d',it_gp)
                end
                if (~objfun_red)
                    fprintf('\n Identification Phase terminates: sufficient decrease not achieved - GPsteps = %d',it_gp)
                end
                if (it_gp >= MaxGP)
                    fprintf('\n Identification Phase terminates: maximum number of GP steps reached - GPsteps = %d',it_gp)
                end
                if  (nprod>=MaxProd)
                    fprintf('\n Identification Phase terminates: maximum number of matrix-vector products reached - GPsteps = %d',it_gp)
                end
                if  (nproj>=MaxProj)
                    fprintf('\n Identification Phase terminates: maximum number of projections reached - GPsteps = %d',it_gp)
                end
                if  (itergp>=MaxTotGP)
                    fprintf('\n Identification Phase terminates: maximum number of GP steps reached - GPsteps = %d',it_gp)
                end
            end
            
            %--------------------------------------------------------
            % End of GP phase
            %--------------------------------------------------------
            
            active = find(ivar ~= 0);     % Active-set
            nact = size(active,1);
            contmin = (n-nact)>(1-checkflag); % Check if there are variable left free
            
            numbind = sum([hk(ivar == -1) >= 0; hk(ivar == 1) <= 0]);            
            
            if(Verbosity)
                fprintf('\n\n Number of active box constraints = %d of %d',nact,n);
                fprintf('\n\n Number of binding box constraints = %d of %d',numbind,n);
                fprintf('\n Number of variables near the bound (1e-6) = %d\n',nnz((abs(xk-l)<1e-6)|(abs(u-xk)<1e-6)));
            end
            
            %--------------------------------------------------------
            % Start of Minimization Phase
            %--------------------------------------------------------

            if UncMinType && ~skipmin
                i = 0;
                x_old = xk;
                
                dk = zeros(size(xk));
                recover = 0;
                                               
                %---------------------------------------------------------------
                % The Minimization Phase stops if one of the following conditions
                % holds:
                % - the projected gradient satisfies stopping criterion (*)
                %   (see line 16)
                % - the current iterate is non-proportional,
                % - the active-set has changed (chng_act = 0) and the current
                %   iterate is non proportional,
                % - the maximum number of consecutive calls to the minimization
                %   algorithm has been reached,
                % - the minimum on the face has been found,
                % - the current iterate activates n constraints.
                %---------------------------------------------------------------
                
                while ( (pgnorm>tol) && (contmin) && (nprod<MaxProd) && ...
                        (nproj<MaxProj) && (itergp<MaxTotGP) && (itermin<MaxTotMin))
                    
                    exitphase = 'minimization';
                    
                    i = i+1;       % used for updating callmin
                                       
                    if recover==0
                        WorkingSet = (ivar==0);
                        info_last = struct([]);
                        if checkflag==0
                            info_proj = struct([]);
                        end
                    end
                    
                    % Call the minimization phase routine which solves the subproblem obtained
                    % considering only the free variables with the reduced linear constraint
                    if checkflag==0     % Functions for SLBQPs
                        if UncMinType>2 % CG
                            [dk,info_last,it_min_part,nprod,flag_min,info_proj] = ...
                                cg4dklin(WorkingSet,...
                                H,...                    reduced Hessian matrix
                                -gk,...                  coefs of reduced linear term
                                q,...                    coefs of reduced linear constraint
                                dk,...                   starting point of the subproblem (direction)
                                info_last,...            starting direction of the subproblem
                                Xi,MaxStepMin,nprod,...
                                info_proj);%             information on the Householder transformation for the subproblem

                            itermin = itermin+it_min_part;
                        else % SDC/SDA
                            [dk,info_last,it_min_part,nprod,flag_min,info_proj] = ...
                                sdc4dklin(WorkingSet,...
                                H,...                    reduced Hessian matrix
                                -gk,...                  coefs of reduced linear term
                                q,...                    coefs of reduced linear constraint
                                dk,...                   starting point of the subproblem (direction)
                                info_last,...            information on the termination of previous call
                                H_sdc,M_sdc,...          parameters for the steplength choice in SDC and SDA
                                Xi,MaxStepMin,nprod,...
                                UncMinType==2,...        0: use SDC, 1: use SDA
                                Monotone,...             0: non-monotone, 1: monotone
                                info_proj);%             information on the Householder transformation for the subproblem
                        end
                    else % Functions for BQPs
                        if UncMinType>2 %CG
                            [dk,info_last,it_min_part,nprod,flag_min] = ...
                                cg4dk(WorkingSet,...
                                H,...                    reduced Hessian matrix
                                -gk,...                  coefs of reduced linear term
                                dk,...                   starting point of the subproblem (direction)
                                info_last,...            starting direction of the subproblem
                                Xi,MaxStepMin,nprod);

                            itermin = itermin+it_min_part;
                        else % SDC/SDA
                            [dk,info_last,it_min_part,nprod,flag_min] = ...
                                sdc4dk(WorkingSet,...
                                H,...                    reduced Hessian matrix
                                -gk,...                  coefs of reduced linear term
                                dk,...                   starting point of the subproblem (direction)
                                info_last,...            information on the termination of previous call
                                H_sdc,M_sdc,...          parameters for the steplength choice in SDC and SDA
                                Xi,MaxStepMin,nprod,...
                                UncMinType==2,...        0: use the SDC, 1: use SDA
                                Monotone);%              0: non-monotone, 1: monotone
                            
                        end
                    end
                    
                    recover = 1;
                    skiplinesearch = false;
                    negcurvestep = false;
                    
                    if flag_min==3
                        nmaxstepmin = nmaxstepmin +1;
                    end
                    
                    if flag_min == 2
                        nondecr = nondecr +1;
                        itermin = itermin + it_min_part;
                        
                        if options.Verbosity
                            fprintf('\nDirection computed by SDC/SDA does not guarantee obj decrease ... Back to GP.\n')
                        end
                        
                        dk = -gk;
                        contmin = 0;
                    else
                        if (flag_min < 0) % Non-positive curvature direction found
                            flag = 1;
                            
                            projdk = dk;
                            
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
                            
                            alfafeas = min(alfavect(alfavect > 0));
                            
                            if (alfafeas== Inf)
                                if flag_min < 0
                                    fprintf('\nThe problem is unbounded from below.\n');
                                    flag = 6;
                                end
                                nact = size(find(ivar~=0),1);
                                time = toc(startingtime);
                                funvalues = funvalues(1:indfunval);
                                pgradnorm = pgradnorm(1:indpgrad);
                                if nargout>7
                                    otherinfo = struct('ivar',ivar,'nact',nact,'nact0',nact0,'funvalues',funvalues,'pgradnorm',pgradnorm,...
                                        'iter',iter,'itergp',itergp,'itridgp',itridgp,'nmaxgp',nmaxgp,...
                                        'itermin',itermin,'itridmin',itridmin,'nmaxstepmin',nmaxstepmin,'callmin',callmin,'nrest',nrest,...
                                        'nprop',nprop,'gamma',Gamma,'nondecr',nondecr,'exitphase',exitphase,'ngradpro',ngradpro,'time',time);
                                end
                                return;
                            end

                            if ~isempty(alfafeas)
                                dk = projdk*alfafeas;
                                negcurvestep = true;
                            else
                                dk = zeros(size(dk));
                                skiplinesearch = true;
                            end
                            
                            recover = 0;
                        end
                        
                        if skiplinesearch
                            if (Verbosity)
                                fprintf('\nDirection is zero ... Skipping line search ...\n');
                            end
                            
                            xmin = xk;
                            ivar_new = ivar;
                            fmin = fk;
                            gmin = gk;
                            
                        else
                            % Creating the temporary bound vectors which will help to project on the original face 
                            l_temp = l;
                            u_temp = u;
                            
                            if checkflag==0
                                l_temp(~WorkingSet) = xk(~WorkingSet);
                                u_temp(~WorkingSet) = xk(~WorkingSet);
                            end
                            
                            if negcurvestep
                                % Maximum feasible step along the direction generated by the minimization phase algorithm
                                xmin = xk + dk;
                                [xmin_new,ivar_new] = simproj(xmin,l_temp,u_temp,q,b,checkflag);
                                nproj = nproj + 1;
                                
                                xmin = xmin_new;
                                if isa(H,'function_handle')
                                    Hxmin = H(xmin);
                                else
                                    Hxmin = H*xmin;
                                end
                                nprod = nprod + 1;
                                fmin = 0.5*(xmin'*Hxmin)-c'*xmin;
                                gmin = Hxmin - c;
                            else
                                % Projected line search on the direction generated by the minimization phase
                                [xmin,fmin,gmin,it_int,nprod,nproj] = ...
                                    linesearch2(H,c,xk,fk,gk,dk,l_temp,u_temp,q,b,checkflag,nprod,nproj);
                                
                                ivar_new = zeros(size(xmin));
                                ivar_new (xmin <= l) = -1;
                                ivar_new (xmin >= u) = 1;
                                ivar_new (l == u) = 2;
                                
                                itridmin = itridmin+it_int;
                            end
                        end
                        
                        fvar = fk-fmin;
                        
                        if (Verbosity)
                            fprintf('\n****\nMinimization steps = %d \n fval = %d \n fval_old-fval = %d\n |x_old-xk|_2 = %d\n pgnorm = %d', ...
                                    it_min_part,fmin,fvar,norm(x_old-xmin,2),pgnorm);
                        end
                        
                        chng_act =  any(ivar ~= ivar_new);  % active-set variation
                        ivar = ivar_new;
                        active = find(ivar ~= 0);
                        nact = size(active,1);
                        
                        % Computing the Projected Gradient norm
                        if checkflag==0
                            aF = q(ivar==0);
                            aFaF = aF'*aF;
                            if aFaF > 0
                                rho_f = (gmin(ivar==0)'*aF)/(aFaF);
                                hmin = gmin - q*rho_f;
                            else
                                hmin = gmin;
                            end
                        else
                            hmin = gmin;
                        end
                        free_vars = (ivar==0);
                        phi = hmin.*free_vars;
                        
                        chop_vars = (((ivar==-1) & (hmin<0)) | ((ivar==1) & (hmin>0)));
                        
                        if sum(chop_vars)~=0
                            pgrad = projgrad(gmin,ivar,q,checkflag);
                            ngradpro = ngradpro + 1;
                            nproj = nproj + 1;
                            if checkflag==0
                                beta = -pgrad-phi;
                            else
                                beta = -pgrad.*chop_vars;
                            end
                        else
                            pgrad = -phi;
                            beta = zeros(size(gmin));
                        end
                        
                        if InfNorm
                            pgnorm = norm(pgrad,Inf);
                        else
                            pgnorm = norm(pgrad);
                        end
                        
                        % Check the stopping conditions for the minimization 
                        % on the current face
                        contmin = (~(chng_act) && (i<=MaxRestart) && ...
                                  (n-nact)>(1-checkflag) && (flag_min==0));
                        
                        % Compute norms for the proportionality check
                        chopped_grad = norm(beta,Inf);
                        free_grad = norm(phi);
                        if sum(chop_vars)==0
                            bind = 1;
                        else
                            bind = 0;
                        end
                        
                        if Verbosity
                            fprintf('\nMinimization stopped: norm phi = %f, norm_inf beta = %f\n            Gamma = %f',free_grad,chopped_grad,Gamma);
                        end
                        
                        if Proportioning && flag_min>=0 && flag_min~=1
                            proportional = (chopped_grad <= free_grad*Gamma);
                            if (~proportional)
                                if AdaptGamma
                                    Gamma = Gamma*1.1;
                                    Gamma = max(Gamma,1);
                                end
                                contmin = 0;
                            elseif (proportional && chng_act && (pgnorm > tol) ...
                                    && (iter<MaxOutIt) && (nprod<MaxProd) && ...
                                    (nproj<MaxProj) && (itergp<MaxTotGP) && ...
                                    (itermin<MaxTotMin))
                                if AdaptGamma
                                    Gamma = Gamma*0.9;
                                    Gamma = max(Gamma,1);
                                end
                                xk = xmin;
                                fk = fmin;
                                gk = gmin;
                                
                                contmin = (i<=MaxRestart) && (n-nact)>(1-checkflag);
                                if Verbosity && contmin
                                    fprintf('\n Proportional iteration ... Continuing with minimization.\n');
                                end
                                x_old = xk;
                                dk = zeros(size(xk));
                                recover = 0;
                                nprop = nprop+1;
                            end
                        else
                            contmin = contmin && bind;
                        end
                        
                        if (~contmin) || (pgnorm <= tol) || (nprod>=MaxProd) ...
                                || (nproj>=MaxProj) || (itergp>=MaxTotGP) ...
                                || (itermin>=MaxTotMin)
                            
                            if UncMinType < 3
                                itermin = itermin + it_min_part;
                            end

                            xk = xmin;
                            fk = fmin;
                            gk = gmin;
                            hk = hmin;
                            
                            dk = -gk;
                            
                            funvalues(indfunval) = fk;
                            indfunval = indfunval + 1;
                            pgradnorm(indpgrad) = pgnorm;
                            indpgrad = indpgrad + 1;
                            
                            numbind = sum([hk(ivar == -1) >= 0; hk(ivar == 1) <= 0]);
                            
                            if(Verbosity)
                                fprintf('\n\n Number of active bound constraints = %d of %d',nact,n);
                                fprintf('\n\n Number of binding bound constraints = %d of %d',numbind,n);
                                fprintf('\n Number of variables near their bounds (distance < 1e-6) = %d\n',nnz((abs(xk-l)<1e-6)|(abs(u-xk)<1e-6)));
                            end
                        end
                        
                        if (Verbosity)
                            if contmin
                                fprintf('\nMinimization Phase continues: active set has not changed - itermin = %d\n',itermin);
                            end
                            if ((~bind) && ~Proportioning)
                                fprintf('\nMinimization Phase terminates: active set is not binding - itermin = %d\n',itermin);
                            end
                            if chng_act
                                fprintf('\nMinimization Phase terminates: active set changed - itermin = %d\n',itermin);
                            end
                            if i>MaxRestart
                                fprintf('\nMinimization Phase terminates: more than %d restarts - itermin = %d\n',MaxRestart,itermin);
                            end
                            if (n-nact)==0
                                fprintf('\nMinimization Phase terminates: no more free variables - itermin = %d\n',itermin);
                            end
                            if flag_min == 1
                                fprintf('\nMinimization Phase terminates: minimizer of the restricted problem has been reached - itermin = %d\n',itermin);
                            end
                        end
                        
                        callmin = callmin+i;
                        if ( contmin)
                            nrest = nrest+1;
                        end
                    end
                end
            end
            
            if (pgnorm < parttolvector(tolindex)) && (PartialResults)
                if tolindex==1 || Verbosity
                    fprintf('\n TOL         time  nprod     it  it1st  itmin   lsgp  lsmin   nact   fval             pgnorm');
                end
                fprintf('\n%.e  %8.5f  %5d  %5d  %5d  %5d  %5d  %5d  %5d   %.7e  %.2e    %5d    %5d', ...
                    tolvector(tolindex),toc(startingtime),nprod,iter,itergp,itermin,itridgp,itridmin,nact,fk,pgnorm');
                
                tolindex = tolindex+1;
            end
            
            if (Verbosity)
                command = input('\n Press ENTER to proceed or Ctrl+C to abort\n','s');
            end
            
            funvalues(indfunval) = fk;
            indfunval = indfunval + 1;
            pgradnorm(indpgrad) = pgnorm;
            indpgrad = indpgrad + 1;
        end
    end
    
    nact = size(find(ivar~=0),1);
    time = toc(startingtime);
    
end

if (~PartialResults) && Verbosity
    fprintf('\n             time  nprod     it  it1st  itmin   lsgp  lsmin   nact   fval             pgnorm');
end
if Verbosity || PartialResults
    fprintf('\n EXIT  %8.5f  %5d  %5d  %5d  %5d  %5d  %5d  %5d   %.7e  %.2e\n',...
        time,nprod,iter,itergp,itermin,itridgp,itridmin,nact,fk,pgnorm');
end

funvalues = funvalues(1:indfunval);
pgradnorm = pgradnorm(1:indpgrad);

if(GeneratePlots)
    figure(1)
    semilogy(funvalues-min(funvalues))
    title('Objective Function Values (f(x) - f(min))');
    figure(2)
    diff = (funvalues(1:end-1)-funvalues(2:end))./abs(funvalues(1:end-1));
    semilogy(abs(diff),'r')
    title('Objective Function Variation');
    figure(3)
    semilogy(pgradnorm,'g')
    title('Projected Gradient Norm');
end

if flag==0
    fprintf('\nP2GP found a point which satisfies the stopping criterion.\n')
elseif flag ==1
    fprintf('\nThe problem is not strictly convex, but P2GP found a point which satisfies the stopping criterion.\n')
end

if pgnorm > tol
    if(nprod>=MaxProd)
        flag = 2;
        fprintf('\nThe required tolerance was not satisfied, P2GP stopped because nprod > maxprod.\n')
    elseif (nproj>=MaxProj)
        flag = 3;
        fprintf('\nThe required tolerance was not satisfied, P2GP stopped because nproj > maxproj.\n')
    elseif (itergp>=MaxTotGP)
        flag = 4;
        fprintf('\nThe required tolerance was not satisfied, P2GP stopped because itergp > maxtotgp.\n')
    elseif (itermin>=MaxTotMin)
        flag = 5;
        fprintf('\nThe required tolerance was not satisfied, P2GP stopped because itermin > maxtotmin.\n')
    end
end

if nargout>7
    otherinfo = struct('ivar',ivar,'nact',nact,'nact0',nact0,'funvalues',funvalues,'pgradnorm',pgradnorm,...
        'iter',iter,'itergp',itergp,'itridgp',itridgp,'nmaxgp',nmaxgp,...
        'itermin',itermin,'itridmin',itridmin,'nmaxstepmin',nmaxstepmin,'callmin',callmin,'nrest',nrest,...
        'nprop',nprop,'gamma',Gamma,'nondecr',nondecr,'exitphase',exitphase,'ngradpro',ngradpro,'time',time);
end
end
