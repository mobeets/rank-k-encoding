function [theta, Ds, Dt] = evidence(X, Y, Sigma, mu, theta_ridge, sigmasq_ridge)

    successFcn = @(Ds, Dt) all(eig(exp(1 - 0.5*(Ds + Dt))) > 0);
    [Ds, Dt] = randomDistances(size(Sigma, 1), successFcn);
    objfcn = @(theta) ASD_logf(theta, X, Y, Sigma, mu, Ds, Dt);
    t0 = [sqrt(sigmasq_ridge), -log(theta_ridge) 1.0 1.0];

    options = optimoptions(@fminunc, 'GradObj', 'on');
    theta = fminunc(objfcn, t0, options);
end

function [logE, dlogE] = ASD_logf(theta, X, Y, Sigma, mu, Ds, Dt)
    [C, dC] = ASD(theta(2:end), Sigma, mu, Ds, Dt);
    logE = -logf(C, theta(1), Sigma, X, Y);
    if nargout > 1
        dssq = dlogE_dssq(C, theta(1), X, Y, Sigma, mu);
        dlogE = -[dssq dC];
    end
end

function logE = logf(C, sig, Sigma, X, Y)
    n = size(Sigma, 1);
    logDet = @(A) 2*sum(diag(chol(A)));
    z1 = 2*pi*Sigma;
    z2 = 2*pi*sig^2*eye(n, n);
    z3 = 2*pi*C;
%     Z = sqrt(abs(z1)/(abs(z2)*abs(z3)));
    if ~all(eig(z3) > 0)
        1;
    else
        2; %disp('.');
    end
    logZ = 0.5*(logDet(z1) - (logDet(z2) + logDet(z3)));

    B = (1/sig^2) - (X'*Sigma*X)/sig^4;
%     E = sqrt(Z)*exp(-0.5*Y*B*Y');
    logE = logZ - 0.5*Y*B*Y';

end

function v = dlogE_dssq(C, sig, X, Y, Sigma, mu)
    T = numel(Y);
    n = size(Sigma, 1);
    V1 = eye(n, n) - Sigma\C;
    V2 = (Y - mu'*X)*(Y - mu'*X)';
    V = -T + trace(V1) + V2/sig^2;
    v = V/sig^2;
end
