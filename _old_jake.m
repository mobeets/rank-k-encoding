
%% stimulus

% build stimulus
nTimesteps = 100;
stim = 5*randn(nTimesteps, 1);

%% response

% history filter
mLags = 10;
x = 1:mLags;
k = 5; th = 1;
w_true = x(:).^(k-1).*exp(-x(:)/th);
b_true = 5 - x(:).^(k-1).*exp(-x(:)/th);

c = 30;
noisesigma = 5;
f = @(x) x.^2;

figure(1); clf
subplot(131)
stim_bounds = linspace(min(stim), max(stim), 100);

% make some basis functions
k = 10;
mu = linspace(min(stim), max(stim), k);
s = mean(diff(mu))/2;
fi = @(x, m) exp(-.5* (x-m).^2/s.^2);

plot(stim_bounds, f(stim_bounds), 'k')
title('input nonlinearity')
subplot(133)
plot(w_true)
title('true w weights')
subplot(132)
plot(b_true)
title('true b weights')

%% reshape stimulus to have lags

X = nan(mLags, nTimesteps-mLags);
for ii = mLags:nTimesteps
    X(:, ii-mLags+1) = (stim((ii-mLags+1):ii));
end
R = c + w_true'*f(X) + randn(1, nTimesteps-mLags+1)*noisesigma;

% Plot response
figure(2); clf
subplot(1,2,1)
plot(w_true)
subplot(1,2,2)
plot(stim, 'k'); hold on
plot(R, 'r')

%% Recover weights using linear least squares regression

Xdesign = [ones(1,size(X,2)); X];
wls = (Xdesign*Xdesign')\(Xdesign*R');

figure(2); 
subplot(121)
hold on; 
plot(wls(2:end), 'r')
subplot(122)
plot(wls'*Xdesign, 'b')

%% Recover weights using alternating least squares

fb = bsxfun(fi, X(:), mu);
fb = reshape(fb(:), size(X, 1), size(X, 2), size(mu, 2));
R = c + w_true'*fb + randn(1, nTimesteps-mLags+1)*noisesigma;
