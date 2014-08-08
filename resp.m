function R = resp(S, nLags, nRank)

    nTimesteps = size(S, 1);
    X = design(S, nLags);

    % history filter
    x = 1:nLags;
    w_true = @(k, th) x(:).^(k-1).*exp(-x(:)/th);

    % input nonlinearity, noise, constant
    f1 = @(x) x.^3;
    f2 = @(x) 40*x.^2;
    noisesigma = 5;
    c = 30;

    if nRank == 1
        R = w_true(5,1)'*f1(X) + c + randn(1, size(X, 2))*noisesigma;
        ploti(1, S, w_true(5,1), f1);
    elseif nRank == 2
        R = w_true(5,1)'*f1(X) + 10*w_true(2,1)'*f2(X) + c + randn(1, size(X, 2))*noisesigma;
        ploti(2, S, 10*w_true(2,1), f2);
    end
    
end
%% plot
function ploti(i, S, w_true, f)

    figure(i); clf;

    subplot(141)
    plot(S, 'k')
    title('stimulus')

    subplot(142)
    stim_bounds = linspace(min(S), max(S), 10);
    plot(stim_bounds, f(stim_bounds), 'k')
    title('input nonlinearity')

    subplot(143)
    plot(w_true, '-o')
    title('true w weights')

    subplot(144)
    image(w_true*f(stim_bounds))
    title('C')
end
