function R = resp(S, nLags)

    nTimesteps = size(S, 1);
    X = design(S, nLags);

    % history filter
    x = 1:nLags;
    k = 5;
    th = 1;
    w_true = x(:).^(k-1).*exp(-x(:)/th);

    % input nonlinearity, noise, constant
    f = @(x) x.^2;
    noisesigma = 5;
    c = 30;

    R = w_true'*f(X) + c + randn(1, size(X, 2))*noisesigma;

    ploti(1, S, w_true, f);

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
    plot(w_true)
    title('true w weights')

    subplot(144)
    image(w_true*f(stim_bounds))
    title('C')
end
