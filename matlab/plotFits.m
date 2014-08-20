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
