%% stimulus
nTimesteps = 2000;
S = 5*randn(nTimesteps, 1);

%% response
nLags = 8;
nRank = 2;
R = resp(S, nLags, nRank);

%% init
rmse = @(a, b) sqrt((a-b)*(a-b)'); % for assessing fits
Rh1 = nan(size(R));
Rh2 = nan(size(R));
Rh3 = nan(size(R));
Rh3b = nan(size(R));
Rh4 = nan(size(R));
Rh4b = nan(size(R));
Rh4c = nan(size(R));

%% fit (linear)
Rh1 = linreg(S, R, nLags);

%% fit (bilinear)
Rh2 = rankreg(S, R, nLags, 1);

%% fit (rank-2)
Rh3 = rankreg(S, R, nLags, 2);

%% fit (rank-2), manually minimizing hyperparameter for regularizer
lambda = optimizeHyperparameter(S, R, nLags, 2, [1e-3 1e-3]);
Rh3b = rankreg(S, R, nLags, 2, lambda);

%% fit (full rank), using fixed-point ridge regression

Rh4 = rankreg(S, R, nLags, Inf, 'ridge');

%% fit (full rank), using fixed-point ARD

Rh4b = rankreg(S, R, nLags, Inf, 'ARD');

%% fit (full rank), manually minimizing hyperparameter for regularizer

lambda = optimizeHyperparameter(S, R, nLags, Inf, 1e-3);
Rh4c = rankreg(S, R, nLags, Inf, lambda);

%% write results

disp(['rmse (linear) = ' num2str(rmse(R, Rh1))]);
disp(['rmse (bilinear, no reg) = ' num2str(rmse(R, Rh2))]);
disp(['rmse (rank-2, no reg) = ' num2str(rmse(R, Rh3))]);
disp(['rmse (rank-2, manual ridge) = ' num2str(rmse(R, Rh3b))]);
disp(['rmse (full rank, ridge fixed point) = ' num2str(rmse(R, Rh4))]);
disp(['rmse (full rank, ARD fixed point) = ' num2str(rmse(R, Rh4b))]);

%% plot

figure(6); clf; hold on;
plot(R, 'k');
% plot(Rh1, 'c');
% plot(Rh2, 'b');
% plot(Rh3, 'r');
plot(Rh4, 'b');
plot(Rh4b, 'r');
title('response');

