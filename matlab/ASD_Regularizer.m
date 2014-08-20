function [C, dC] = ASD_Regularizer(theta, Sigma, mu, Ds, Dt)
    if nargin < 1
        [Sigma, mu, ds, dt, Ds, Dt] = defaultValues();
    else
        ds = theta(1);
        dt = theta(2);
        p = theta(3);
    end
    assert(isequal(Ds, Ds'), 'Ds must be symmetric.');
    assert(isequal(Dt, Dt'), 'Dt must be symmetric.');
    assert(isequal(diag(Ds), zeros(size(Ds, 1), 1)), 'Ds must have a zero diagonal.');
    assert(isequal(diag(Dt), zeros(size(Dt, 1), 1)), 'Dt must have a zero diagonal.');
    
    C = exp(-p - 0.5*((Ds/ds^2) + (Dt/dt^2)));
%     assert(all(eig(C) > 0), 'C is not positive definite--I think your distance matrices suck.');

    if nargout > 1
        A = (C - Sigma - mu*mu')/C;
        dlogE_dp = 0.5*trace(A);
        dlogE_dds = -0.5*trace(A*(C .* Ds/ds^3)/C);
        dlogE_ddt = -0.5*trace(A*(C .* Dt/dt^3)/C);
        dC = [dlogE_dp dlogE_dds dlogE_ddt];
    end
end
function [Sigma, mu, ds, dt, Ds, Dt] = defaultValues()
    ds = 1.0;
    dt = 1.0;
    p = -1;
    n = 99;
    Sigma = eye(n, n);
    mu = randn(n, 1);
    successFcn = @(Ds, Dt) all(eig(exp(1 - 0.5*(Ds + Dt))) > 0);
    [Ds, Dt] = randomDistances(n, successFcn);
end
