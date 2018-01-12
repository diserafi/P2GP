% 
% This routine provides an example of use of SLBQPgen.
% 
% The example generates a set of strictly convex SLBQPs.
% Note that non-strictly convex and non-convex problems may be also generated.
% For details see
% 
% [1] D. di Serafino, G. Toraldo, M. Viola and J. Barlow,
%     "A two-phase gradient method for quadratic programming problems with
%      a single linear constraint and bounds on the variables", 2017,
%      http://arxiv.org/abs/1705.01797
%      or
%      http://www.optimization-online.org/DB_HTML/2017/05/5992.html
%
% 

dirname = 'SLBQP';
mkdir(dirname)
filename = [dirname,'_info.txt'];
diaryname =  fullfile(dirname, filename);
diary(diaryname)

seed = 0;
s = RandStream.create('mt19937ar','Seed',seed);

%% Parameter for the generator
n_vect        = [10000,20000];          % size of the prolems
cond_vect     = [4, 5, 6];              % ncond of the problems
ndeg_vect     = [0, 1, 3];              % ndeg of the problems
degvar_vect   = 0;                      % degvar of the problems
negeig_vect   = 0;                      % number of negative eigenvalues 
zeroeig_vect  = 0;                      % number of zero eigenvalues
act_vect      = [0.0, 0.1, 0.5, 0.9];   % vector used for naxsol and nax0
linear        = true;                   % the problems are SLBQPs

%% Initialization
index = 1;
ipne = 1;
ip0e = 1;
ipdvar = 1;
fprintf('Prob\tsize\tcond(H)\tndeg\t#act x_bar\n');

for ideg = 1:length(ndeg_vect)
    for in = 1:length(n_vect)
        for icond = 1:length(cond_vect)
            for ipsol = 2:length(act_vect)
                    
                    pbname  = [dirname, num2str(index)];
                    n       = n_vect(in);
                    ncond   = cond_vect(icond);
                    zeroeig = zeroeig_vect(ip0e);
                    negeig  = negeig_vect(ipne);
                    naxsol  = act_vect(ipsol);
                    degvar  = degvar_vect(ipdvar);
                    ndeg    = ndeg_vect(ideg); % for each problem 4 different starting points will be generated
                    nax0    = act_vect;
                    
                    [d,P,c,l,u,q,b,x0vect,x_bar] = SLBQPgen(n,ncond,zeroeig,negeig,naxsol,degvar,ndeg,linear,nax0);
                                        
                    fprintf('\n%5s\t%d\t%.2e\t%d\t%d\t%.7e\n',pbname,n,10^ncond,ndeg,sum(x_bar == l | x_bar == u));
                    
                    save(fullfile(dirname,pbname),'d','P','c','l','u','q','b','x0vect','x_bar')
                    
                    index = index + 1;
            end
        end
    end
end

diary off