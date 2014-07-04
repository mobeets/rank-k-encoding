using Distributions

function design(S, nLags)
    # returns [nLags x length(S)-nLags+1]
    return hcat([S[i-nLags+1:i] for i in nLags:length(S)]...);
end

function resp(S, nLags)
    nTimesteps = length(S);
    X = design(S, nLags);

    # history filter
    x = 1:nLags;
    k = 5;
    th = 1;
    w_true = x.^(k-1).*exp(-x/th);

    # input nonlinearity, noise, constant
    f = x -> x.^2;
    noisesigma = 5;
    c = 30;

    R = w_true'*f(X) + c + randn(1, size(X, 2))*noisesigma;
    return R;
end

function linreg(S, R, nLags)
    X = design(S, nLags);
    Xa = [ones(1, size(X, 2)); X];
    w_h = (Xa*Xa') \ (Xa*R');
    Rh = w_h'*Xa;
    return Rh;
end

function rankreg(S, R, nLags, nRank)
    X = design(S, nLags);

    # make some basis functions
    nBases = 10;
    mu = linspace(minimum(S), maximum(S), nBases);
    sig = mean(diff(mu))/2;
    bi = (x, m) -> pdf(Normal(m, sig), x);

    # apply basis functions to design matrix

    # augment design matrix to fit constant

    # fit

end

function main()
    # stimulus
    nTimesteps = 100;
    S = 5*randn(nTimesteps,1);

    # response
    nLags = 8;
    R = resp(S, nLags);

    # fit: linear, bilinear, full rank
    Rh1 = linreg(S, R, nLags);
    Rh2 = rankreg(S, R, nLags, 2);
    Rh3 = rankreg(S, R, nLags, Inf);

    # write results
    rmse = (a,b) -> sqrt(sum((a-b).^2));
    println(["rmse (linear) = " rmse(R, Rh1)])
    println(["rmse (bilinear) = " rmse(R, Rh2)])
    println(["rmse (full rank) = " rmse(R, Rh3)])

    # plot

end
