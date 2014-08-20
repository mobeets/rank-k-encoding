function [v, dv] = ASD_logEvidence(theta, X, Y, Sigma, mu, Ds)
    [C, dC] = ASD_Regularizer(theta(2:end), Sigma, mu, Ds);
    v = -logE(C, theta(1), Sigma, X, Y);
    if nargout > 1
        dssq = dlogE_dssq(C, theta(1), X, Y, Sigma, mu);
        dv = -[dssq dC];
    end
end

function v = logE(C, sig, Sigma, X, Y)
    n = size(Sigma, 1);
    logDet = @(A) 2*sum(diag(chol(A)));
    z1 = 2*pi*Sigma;
    z2 = 2*pi*sig^2*eye(n, n);
    z3 = 2*pi*C;
    logZ = 0.5*(logDet(z1) - (logDet(z2) + logDet(z3)));
    B = (1/sig^2) - (X'*Sigma*X)/sig^4;
    v = logZ - 0.5*Y*B*Y';
end

function v = dlogE_dssq(C, sig, X, Y, Sigma, mu)
    T = numel(Y);
    n = size(Sigma, 1);
    V1 = eye(n, n) - Sigma\C;
    V2 = (Y - mu'*X)*(Y - mu'*X)';
    V = -T + trace(V1) + V2/sig^2;
    v = V/sig^2;
end
