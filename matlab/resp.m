function R = resp(S, nLags, nRank)

    nTimesteps = size(S, 1);
    X = design(S, nLags);

    % history filter
    x = 1:nLags;
    w_true = @(k, th) (x(:).^(k-1)).*exp(-x(:)/th);

    % input nonlinearity, noise, constant
    f1 = @(x) 1e-2*x.^3;
    f2 = @(x) 1*x.^2;
    noisesigma = 1;
    c = 250;

    if nRank == 1
        R = w_true(5,1)'*f1(X) + c + randn(1, size(X, 2))*noisesigma;
        ploti(1, S, w_true(5,1), f1);
    elseif nRank == 2
        R = w_true(5,1)'*f1(X) + w_true(2,1)'*f2(X) + c + randn(1, size(X, 2))*noisesigma;
        ploti(1, S, w_true(5,1), f1);
        ploti(2, S, w_true(2,1), f2);
        set(gca,'FontSize', 14);
        figure(3); plot(R, 'k-', 'MarkerSize', 2); xlabel('time'); ylabel('spike rate');
    end
    
end
%% plot
function ploti(i, S, w_true, f)

    figure(i); clf; colormap(gray);
% 
%     subplot(141)
%     plot(S, 'k')
%     title('stimulus')

    subplot(131);
    set(gca,'FontSize', 12);
    stim_bounds = linspace(min(S), max(S), 10);
    plot(stim_bounds, f(stim_bounds), 'ko', 'MarkerFaceColor', 'k');
    title('b, input nonlinearity');
    xlabel('stimulus value');
    ylabel('weight');

    subplot(132)
    set(gca,'FontSize', 12);
    plot(0:numel(w_true)-1, w_true, 'ko', 'MarkerFaceColor', 'k');
    xlim([0-0.5, numel(w_true)-0.5]);
    title('w, stimulus history weights');
    xlabel('stimulus history');
    ylabel('weight');

    subplot(133)
    set(gca,'FontSize', 12);
    image(w_true*f(stim_bounds));
    xlabel('nonlinearity');
    ylabel('history');
    title('C');
end
