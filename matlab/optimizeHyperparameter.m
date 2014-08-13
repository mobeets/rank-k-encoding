function lambda = optimizeHyperparameter(S, R, nLags, nRank, lambda0)
    rmse = @(a, b) sqrt((a-b)*(a-b)');
    error = @(lambda) rmse(R, rankreg(S, R, nLags, nRank, lambda));
    options = optimoptions(@fminunc, 'Algorithm', 'quasi-newton', 'Display', 'off');
    lambda = fminunc(error, lambda0, options);
end
