
%% stimulus
nTimesteps = 2000;
S = 5*randn(nTimesteps, 1);

%% response
nLags = 8;
nRank = 1;
R = resp(S, nLags, nRank);

%% fit (linear)
Rh1 = linreg(S, R, nLags);

%% fit (bilinear)
Rh2 = rankreg(S, R, nLags, 1);

%% fit (rank-2)
Rh3 = rankreg(S, R, nLags, 2);

%% fit (full rank), minimizing hyperparameter for regularizer
hyperparam_optimize = false;
if hyperparam_optimize
    rmse = @(a, b) sqrt(sum((a-b).^2));
    error = @(lambda) rmse(R, rankreg(S, R, nLags, Inf, lambda));
    lmb = fminunc(error, 1);
    Rh4 = rankreg(S, R, nLags, Inf, lmb);
else
    Rh4 = rankreg(S, R, nLags, Inf, lmb);
end

%% fit (full rank), choosing hyperparameters via ridge regression or ARD

Rh4 = rankreg(S, R, nLags, Inf, 'ARD');
disp(['rmse (rank-2) = ' num2str(rmse(R, Rh3))]);
disp(['rmse (full rank) = ' num2str(rmse(R, Rh4))]);

%% write results
rmse = @(a, b) sqrt(sum((a-b).^2));
disp(['rmse (linear) = ' num2str(rmse(R, Rh1))]);
disp(['rmse (bilinear) = ' num2str(rmse(R, Rh2))]);
disp(['rmse (rank-2) = ' num2str(rmse(R, Rh3))]);
disp(['rmse (full rank) = ' num2str(rmse(R, Rh4))]);

%% plot
figure(6); clf; hold on;
plot(R, 'k');
% plot(Rh1, 'c');
% plot(Rh2, 'b');
% plot(Rh3, 'r');
plot(Rh4, 'g');
title('response');

