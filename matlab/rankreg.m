
function Rh = rankreg(S, R, nLags, nRank)
    % size(w) == [nLags 1]
    % size(b) == [k 1]
    % size(M) == [k nLags nTimesteps]

    X = design(S, nLags);

    % make some basis functions
    nBases = 10;
    mu = linspace(min(S), max(S), nBases);
    sig = mean(diff(mu))/2;
    bi = @(x, m) normpdf(x, m, sig);

    % apply basis functions to design matrix
    M = bsxfun(bi, X(:), mu);
    M = reshape(M(:), size(X, 1), size(X, 2), size(mu, 2));
    M = permute(M, [3 1 2]);

    % augment to fit constant rate
    M = augment(M);

    if nRank == 2
        [C, w_h, b_h] = alternating_lsq(M, R);
        label = 'bilinear';
        fi = 3;
    elseif isinf(nRank)
        C = fullrank_lsq(M, R);
        label = 'full rank';
        fi = 4;
    else
        error('Not yet implemented.')
    end
    
    % calculate response
    nTimesteps = size(M, 3);
    Rh = zeros(1, nTimesteps);
    for t = 1:nTimesteps
        Rh(t) = sum(sum(C'.*M(:,:,t)));
    end

    ploti(fi, label, b_h, w_h, C);

end

%% plot
function ploti(i, label, b_h, w_h, C)

    figure(i); clf;

    subplot(131)
    plot(b_h(2:end), 'r')
    title(['b weights (' label ')'])

    subplot(132)
    plot(w_h(2:end), 'b')
    title(['w weights (' label ')'])

    subplot(133)
    image(C(2:end, 2:end));
    title('C')

end
%%
function [C, w_h, b_h] = alternating_lsq(M, R)
    % recover weights using alternating least squares

    nBases = size(M, 1);
    b_h = ones(nBases, 1); % initial guess
    niters = 1000;
    for i = 1:niters
       w_h = fit_w(M, b_h, R);
       b_h = fit_b(M, w_h, R);
    end
    C = w_h*b_h';

end
%%
function C = fullrank_lsq(M, R)
    % want C = (M*M')\(M*R') but M is tensor so we gotta do it ourselves

    nBases = size(M, 1);
    nLags = size(M, 2);
    
    function A = part1(M)
        % should be square, size [nLags x nLags] x nBases
        A = zeros(nLags, nLags, nBases);
        M2 = permute(M, [2, 3, 1]);
        Mt = permute(M, [3, 2, 1]);
        for i = 1:nBases
            A(:,:,i) = M2(:,:,i) * Mt(:,:,i);
        end
    end

    function B = part2(M, R)
        % should be size [nLags x 1] x nBases
        B = zeros(nLags, 1, nBases);
        M2 = permute(M, [2, 3, 1]);
        for i = 1:nBases
            B(:,:,i) = M2(:,:,i)*R';
        end
    end

    A = part1(M); % M*M'
    B = part2(M, R); % M*R'
    C = zeros(nLags, nBases);
    for j = 1:nBases
        C(:,j) = A(:,:,j) \ B(:,:,j);
    end
    
end
%%
function Ma = augment(M)

    function Ap = augment_inner(A)
        Ap = eye(size(A)+1);
        Ap(2:end, 2:end) = A;
    end

    Ma = zeros(size(M,1)+1, size(M,2)+1, size(M,3));
    for i = 1:size(M,3)
        Ma(:,:,i) = augment_inner(M(:,:,i));
    end

end
%%
function B = mult_b(M, b)

    nLags = size(M, 2);
    nTimesteps = size(M, 3);
    B = zeros(nLags, nTimesteps);
    for t = 1:nTimesteps
        B(:, t) = b'*M(:,:,t);
    end

end
%%
function W = mult_w(M, w)

    nBases = size(M, 1);
    nTimesteps = size(M, 3);
    W = zeros(nBases, nTimesteps);
    for t = 1:nTimesteps
        W(:,t) = w'*M(:,:,t)';
    end

end
%%
function w_h = fit_w(M, b, R)

    B = mult_b(M, b);
    w_h = (B*B')\(B*R');

end
%%
function b_h = fit_b(M, w, R)

    W = mult_w(M, w);
    b_h = (W*W')\(W*R');

end
