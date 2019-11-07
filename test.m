% This script provides an example of use of P2GP

clear all
addpath('ExampleProblems');

% List of the available example problems.
% All the problems have been built with the test problem generator
% included in this package (see SLBQPgen), which extends the problem
% generator by More' and Toraldo [2].
%
% See:
% [1] D di Serafino, G. Toraldo, M. Viola and J. Barlow,
%     "A two-phase gradient method for quadratic programming problems with
%      a single linear constraint and bounds on the variables", 2017
%
% [2] J. More' and G. Toraldo,
%     "Algorithms for bound constrained quadratic programming problems",
%     Numerische Mathematik, 55 (1989), pp. 377-400.  
%
% All problems share the following parameters
%   n      = 10000;
%   ncond  = 6;
%   naxsol = 0.5;
%   degvar = 0;
%   ndeg   = 1;
%   nax0   = 0.5.
%
problems = {'SLBQP1_strcvx',... linear = 1, negeig = 0,   zeroeig = 0
            'SLBQP2_cvx',...    linear = 1, negeig = 0,   zeroeig = 0.2
            'SLBQP3_noncvx',... linear = 1, negeig = 0.2, zeroeig = 0
            'BQP1_strcvx',...   linear = 0, negeig = 0,   zeroeig = 0
            'BQP2_cvx',...      linear = 0, negeig = 0,   zeroeig = 0.2
            'BQP3_noncvx'};%    linear = 0, negeig = 0.2, zeroeig = 0
        
% For the problems with negeig = 0, the solution of the problem is provided
% by means of the vector xsol
        
%% Choose the problem to solve
pbind = 1; % integer in the range 1--6

%% Load the problem
load(problems{pbind});

% The Hessian matrix is saved in factorized form:
%
%                               H = G D G'
%
% where D is a diagonal matrix and G has the form
%       G = (I - 2 p_3 (p_3)') (I - 2 p_2 (p_2)')(I - 2 p_1 (p_1)'),
% with p_j unit vectors.
% Only the diagonal vector d=diag(D) and the unit vectors p1, p2 and p3
% are stored (the latter are in the matrix P)
%
% The function MatVetProd allows to create a function handle
% for the Hessian-vector product

H = @(x) MatVetProduct(D,P,x);

%% Run P2GP with standard options and the provided starting point

options = struct([]);
[x,fx,gx,pgnorm,nprod,nproj,flag,otherinfo] = p2gp(H,c,l,u,q,b,x0,options);

