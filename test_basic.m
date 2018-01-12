% This script provides an example of use of P2GP

%% Create a 100-by-100 positive definite matrix with condition number 1e6

rng('default');

temp = linspace(-6,0,100);
rc = 10.^(temp);
H = sprandsym(100,0.9,rc);

c = H*ones(100,1);

l = -1;
u = 1;

q = ones(100,1);
b = 1;

%% Run P2GP with standard options

[x,fx,gx,pgnorm,nprod,nproj,flag,otherinfo] = p2gp(H,c,l,u,q,b);

