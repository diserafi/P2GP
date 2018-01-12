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

function [y] = MatVetProduct(d,P,x)

%==========================================================================
%
% This function allows to compute the matrix-vector product H*x in the case
% 
%                               H = G D G'
%
% where D is a diagonal matrix and G is of the form
%        G = (I - 2 p_3 (p_3)') (I - 2 p_2 (p_2)')(I - 2 p_1 (p_1)'),
% with p_j unit vectors.
% Only the diagonal vector d = diag(D) and the unit vectors p1, p2 and p3
% are stored (the latter are in a matrix P).
%
%==========================================================================
%
% Authors:
%   Daniela di Serafino (daniela.diserafino@unicampania.it),
%   Gerardo Toraldo (toraldo@unina.it),
%   Marco Viola (marco.viola@uniroma1.it)
%
%==========================================================================
% 
% INPUT ARGUMENTS
% 
% d  = vector of doubles, entries of the diagonal of D;
% P  = matrix of doubles of size n-by-3 whose columns contain the three 
%      unitary vectors for the construction of the matrix G;
% x  = vector of doubles, point at which compute the matrix-vector product.
% 
% OUTPUT ARGUMENTS
% 
% y  = vector of doubles, result of the matrix-vector product.
% 
%==========================================================================

    k = size(P,2);
    
    y = x;
    for i=k:-1:1
        y = y - 2*P(:,i)*(P(:,i)'*y);
    end
    y = d.*y;
    for i=1:k
        y = y - 2*P(:,i)*(P(:,i)'*y);
    end
end