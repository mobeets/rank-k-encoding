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
% lambda = optimizeHyperparameter(S, R, nLags, 2, [1e-3 1e-3]);
% Rh3b = rankreg(S, R, nLags, 2, lambda);

%% fit (full rank), using fixed-point ridge regression

Rh4 = rankreg(S, R, nLags, Inf, 'ridge');

%% fit (full rank), using fixed-point ARD

Rh4b = rankreg(S, R, nLags, Inf, 'ARD');

%% fit (full rank), manually minimizing hyperparameter for regularizer

% lambda = optimizeHyperparameter(S, R, nLags, Inf, 1e-3);
% Rh4c = rankreg(S, R, nLags, Inf, lambda);

%% write results

disp(['rmse (linear) = ' num2str(rmse(R, Rh1))]);
disp(['rmse (bilinear, no reg) = ' num2str(rmse(R, Rh2))]);
disp(['rmse (rank-2, no reg) = ' num2str(rmse(R, Rh3))]);
disp(['rmse (rank-2, manual ridge) = ' num2str(rmse(R, Rh3b))]);
disp(['rmse (full rank, ridge fixed point) = ' num2str(rmse(R, Rh4))]);
disp(['rmse (full rank, ARD fixed point) = ' num2str(rmse(R, Rh4b))]);

%% plot stimulus

figure(8); clf; hold on;
set(gca,'FontSize', 14);
plot(S, 'k-', 'MarkerSize', 5);
xlabel('time');
ylabel('stimulus');

%% plot response

figure(9); clf; hold on;
set(gca,'FontSize', 14);
plot(R, 'k-', 'MarkerSize', 5, 'MarkerFaceColor', 'k', 'LineWidth', 1);
% plot([1 numel(R)], [mean(R) mean(R)], 'r--', 'LineWidth', 1.2);
xlabel('time');
ylabel('response');

%% plot stimulus vs. response

figure(10); clf; hold on;
set(gca,'FontSize', 14);
scatter(S(nLags:end), R, 5, 'k', 'filled');
xlabel('stimulus');
ylabel('response');

%% plot

figure(11); clf; hold on;

sz = 30;
% scatter(R, (R - Rh1), 4, 'yo');
% scatter(R, (R - Rh2), sz, 'go', 'filled');
scatter(R, (R - Rh3), sz, 'bo', 'filled');
scatter(R, (R - Rh4), sz, 'ro', 'filled');

% scatter(mean(R), mean(R-Rh1), 20, 'ko');
% scatter(mean(R), mean(R-Rh2), 20, 'ko', 'filled');
scatter(mean(R), mean(R-Rh3), 20, 'ko', 'filled');
scatter(mean(R), mean(R-Rh4), 20, 'ko', 'filled');

set(gca,'FontSize', 14);
xlabel('response');
ylabel('residual');
% legend('linear', 'bilinear', 'rank-2', 'full rank', 'Location', 'SouthEast');
legend('rank-2', 'full rank', 'Location', 'SouthEast');
% figure(7); hist(R, 30);

%% plot
figure(12); hold on;
set(gca,'FontSize', 14);
plot(R, 'k');
plot(Rh3, 'b');
xlabel('time');
ylabel('spike rate');
legend('actual', 'rank-2');
% plot(Rh2, 'g');
% plot(Rh3, 'r');
% plot(Rh4, 'b');
% plot(Rh4b, 'r');
% title('response');

