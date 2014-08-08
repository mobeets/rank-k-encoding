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

ols = (X,Y) -> (X*X')\(X*Y');

function linreg(S, R, nLags)
    X = design(S, nLags);
    Xa = [ones(1, size(X, 2)); X];
    w_h = ols(Xa, R);
    Rh = w_h'*Xa;
    return Rh;
end

# RANK-K REGRESSION

function augment(M)
    # append one column and row to top-left of each M(t), all zeros except for 1 in top-left corner
    function augment_inner(A)
        Ap = eye([size(A)...]+1...);
        Ap[2:end, 2:end] = A;
        return Ap;
    end
    return reshape(hcat([augment_inner(M[:,:,i]) for i=1:size(M,3)]...), size(M, 1)+1, size(M, 2)+1, size(M, 3));
end


function alternating_lsq(M, R)
    # recover weights using alternating least squares
    b_h = ones(size(M, 1), 1); # initial guess

    mult_w = (M, w) -> vcat([w'*M[:,:,t]' for t=1:size(M,3)]...)';
    mult_b = (M, b) -> vcat([b'*M[:,:,t] for t=1:size(M,3)]...)';
    fit_w = (M, b, R) -> ols(mult_b(M, b), R);
    fit_b = (M, w, R) -> ols(mult_w(M, w), R);
    niters = 1000;
    for i = 1:niters
       w_h = fit_w(M, b_h, R);
       b_h = fit_b(M, w_h, R);
    end
    return w_h*b_h', w_h, b_h

end

function rankreg(S, R, nLags, nRank)
    X = design(S, nLags);

    # apply some basis functions to design matrix
    nBases = 10;
    mu = linspace(minimum(S), maximum(S), nBases);
    sig = mean(diff(mu))/2;
    bi = (x, m) -> pdf(Normal(m, sig), x);
    M = vcat([bi(X, m) for m in mu]...);
    M = reshape(M, size(X, 1), length(mu), size(X, 2));

    # augment design matrix to fit constant
    M = augment(M);

    # fit
    if nRank == 2
        C, w_h, b_h = alternating_lsq(M, R);
        label = "bilinear";
    elseif isinf(nRank)
        # C = fullrank_lsq(M, R);
        label = "full rank";
    else
        error("Not yet implemented.")
    end

    # response
    return [sum(sum(C'.*M[:,:,t])) for t in 1:size(M,3)];
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
    # Rh3 = rankreg(S, R, nLags, Inf);

    # write results
    rmse = (a,b) -> sqrt(sum((a-b).^2));
    println(["rmse (linear) = " rmse(R, Rh1)])
    println(["rmse (bilinear) = " rmse(R, Rh2)])
    # println(["rmse (full rank) = " rmse(R, Rh3)])

    # plot

end

main()
