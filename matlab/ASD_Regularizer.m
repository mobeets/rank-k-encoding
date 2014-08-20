function [C, dC] = ASD_Regularizer(theta, Sigma, mu, Ds)
    if nargin < 1
        [Sigma, mu, p, ds, Ds] = defaultValues();
    else
        p = theta(1);
        ds = theta(2:end);
    end
    G = zeros(size(Ds,1));
    for ii = 1:numel(Ds)
        D = Ds(:,:,ii);
        assert(isequal(D, D'), 'D must be symmetric.');
        assert(isequal(diag(D), zeros(size(D, 1), 1)), 'D must have a zero diagonal.');
        G = G + D/ds(ii)^2;
    end
    C = exp(-p - 0.5*G);
%     assert(all(eig(C) > 0), 'C is not positive definite--check your distance matrices.');

    if nargout > 1
        dC = gradient(C, Sigma, mu, Ds, ds);
    end
end

function dC = gradient(C, Sigma, mu, Ds, ds)
    A = (C - Sigma - mu*mu')/C;
    dC = zeros(numel(ds)+1, 1);
    dC(1) = 0.5*trace(A);
    for ii = 1:numel(ds)
        dC(ii+1) = -0.5*trace(A*(C .* Ds/ds^3)/C);
    end
end

function [Sigma, mu, p, ds, Ds] = defaultValues(nD)
    ds = ones(nd, 1);
    p = -1;
    n = 99;
    Sigma = eye(n, n);
    mu = randn(n, 1);
    successFcn = @(D) all(eig(exp(1 - 0.5*D)) > 0);
    Ds = randomDistances(n, nD, successFcn);
end
