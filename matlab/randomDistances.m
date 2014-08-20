function Ds = randomDistances(n, nD, successFcn)
    % creates nD squared (n-by-n) distance matrices,
    % each satisfying successFcn
    Ds = zeros(n, n, nD);
    for ii = 1:nD
        D = randomDistance(n);
        while ~successFcn(D)
            D = randomDistance(n);
        end
        Ds(:,:,ii) = D;
    end
end
function D = randomDistance(n)
    R = 5*randn(n, n);
    D = R'*R;
    D = D - diag(diag(D));
    D = D.^2;
end
