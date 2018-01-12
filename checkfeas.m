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

function [flag,l,u] = checkfeas(n,l,u,q,b)

%==========================================================================
% This function checks the properties of the feasible set
% 
% INPUT ARGUMENTS
% 
% l,u     = vectors of doubles, upper and lower bounds on the variables;
% q       = vector of doubles, coefficients of the linear constraint;
% b       = double, right-hand side of the linear constraint;
%
% OUTPUT ARGUMENTS
% flag    = integer, information on the feasible set
%          -1 - the feasible set is empty;
%           0 - the feasible set consists of box constraint and a valid
%               linear equality constraint;
%           1 - the feasible set consists of box constraints only;
%           2 - the problem is unconstrained;
% l,u     = vectors of doubles, upper and lower bounds on the variables;
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

if ((isempty(u) || all(isinf(u))) && (isempty(l) || all(isinf(l))) && (isempty(q) || isempty(b)))
    flag = 2;
    l = -Inf;
    u = Inf;
    return;
end


if isempty(l)
    l = -Inf; % The problem is assumed to be unbounded from below
end
if length(l) == 1
    l = l.*ones(n,1);
elseif length(l) < n
    error('The lower bound has to be either a vector of the same size as xk or a scalar\n');
end
if isempty(u)
    u = Inf; % The problem is assumed to be unbounded from above
end
if length(u) == 1
    u = u.*ones(n,1);
elseif length(u) < n
    error('The upper bound has to be either a vector of the same size as xk or a scalar\n');
end

flag = 0;

infbound = 1e20;
ninfbound = -1e20;

l_bound = find( l > ninfbound ); % Variables bounded from below
u_bound = find( u < infbound );  % Variables bounded from above
l(~l_bound) = -Inf;
u(~u_bound) = Inf;

checkbound = all(u >= l);

if checkbound == 0
    flag = -1;
    return;
end
if (isempty(q) || isempty(b))
    flag = 1;
    return;
end

end