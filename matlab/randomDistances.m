function [Ds, Dt] = randomDistances(n, successFcn)
    % creates two squared distance matrices
    Ds = randomDistance(n);
    Dt = randomDistance(n);
    while ~successFcn(Ds, Dt)
        Ds = randomDistance(n);
        Dt = randomDistance(n);
    end
end
function D = randomDistance(n)
    R = 5*randn(n, n);
    D = R'*R;
    D = D - diag(diag(D));
    D = D.^2;
end
