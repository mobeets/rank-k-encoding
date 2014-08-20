
addpath ~/code/huklabBasics/rcg/
% make coordinates
dv.pa.mtrf = 1;
dv.pa.v1rf = .1;
dv.pa.center = [0 5];
dv.pa.theta  = 45;
dv.st.Gpars = nan;
dv = makeGaborPositionsFixed(dv);

%% make prior covariance

prs = [1 10 10];
xx = dv.pa.pos(1,:)';
yy = dv.pa.pos(2,:)';
nx = numel(xx);
wtrue = exp(-.5 * ((xx).^2 + (yy-5).^2)/2);



nTrials = 200;
s2 = 2;
strue = 2;
X = randn(nTrials, nx);
Y = X*wtrue + randn(nTrials,1)*s2;

p0 = prs(1);
dels = prs(2:end);

sqdists = bsxfun(@minus,xx,xx').^2 + bsxfun(@minus, yy, yy').^2;

Cprior = exp(-p0 - .5*sqdists./dels(1));


% full posterior 
S = Cprior - Cprior*X' * ((s2*eye(nTrials) + X*Cprior*X')\(X*Cprior));
u = S*(Y'*X)';

[s2g, p0g, dg] = ndgrid([.1 .5 1 2 5 10], [.1 .5 1 10 50 100], [0 1 2 5 10 20 100]);

hyprs = [s2g(:) p0g(:) dg(:)];
nIter = size(hyprs,1);
logEvids = nan(nIter,1);
uIter = nan(nIter, nx);
for iter = 1:nIter
    
    s2   = hyprs(iter,1);
    p0   = hyprs(iter,2);
    dels = hyprs(iter,3);
    
    Cprior = exp(-p0 - .5*sqdists./dels);
    try
        A = (s2*eye(nTrials) + X*Cprior*X');
        S = Cprior - Cprior*X' * (A\(X*Cprior));
        u = S*(Y'*X)';

        logE = @(s2,Cprior) logdet(2*pi*S) - logdet(2*pi*s2*eye(nx)) - logdet(2*pi*Cprior) ...
        -.5*Y'*(eye(nTrials)/s2 - (X*S*X')/s2^2)*Y; 


    logEvids(iter) = logE(s2, Cprior);
    uIter(iter,:)  = u;
    catch me
    end
end

[m, ii] = max(logEvids);
hyprsMapp = hyprs(ii,:);
umap = uIter(ii,:)';

%%

%
figure(1); clf, 
subplot(2,2,1:2)
plot(wtrue/norm(wtrue), 'k'); hold on
plot(umap/norm(umap), 'b');
wls = (X'*X + 100*eye(nx))\(Y'*X)';
plot(wls/norm(wls), 'r')

subplot(223)
plotWeights(wtrue/norm(wtrue), xx, yy, dv.pa.v1rf);
subplot(224)
plotWeights(umap/norm(umap), xx, yy, dv.pa.v1rf);