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

function [PG] = projgrad(g,Ivar,q,checkflag)

%==========================================================================
% This function computes the projection of the direction -g onto the cone
% tangent to the set defined by
%             (i)           l <= x <= u
% or
%             (ii)     l <= x <= u && q' x = b
% at a point x. It can be used for the computation of the projected
% gradient at a given point.
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
% g       = vector of doubles, direction (e.g. the gradient);
% Ivar    = integer vector with the same length as xmin,
%          -1 - the corresponding entry of x is on its lower bound,
%           0 - the corresponding entry of x is free (l(i) < x(i) < u(i)),
%           1 - the corresponding entry of x is on its upper bound,
%           2 - the corresponding entry of x is fixed because l(i)=u(i);
% q       = vector of doubles, coefficients of the linear constraint;
% checkflag = integer, used in the call of the projection function;
%             its value is 1 if the problem is BQP, 0 if the problem is SLBQP;
%
%==========================================================================
%
% OUTPUT ARGUMENTS
%
% PG    = vector of doubles, projection of -g over the tangent cone at x;
% 
%==========================================================================

    l = -Inf*ones(size(g));
    u = Inf*ones(size(g));
    
    l((Ivar == -1) | (Ivar == 2)) = 0;
    u((Ivar == 1) | (Ivar == 2)) = 0;
    
    a0 = 0;
    
    PG = simproj(-g,l,u,q,a0,checkflag);

end