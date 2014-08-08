function Rh = rankreg(S, R, nLags, nRank, lambda)
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

    if nRank == 1
        M = augment(M, 1);
        if nargin < 5
            lambda = [0.5 0.5];
        end
        [C, w_h, b_h] = alternating_lsq(M, R, lambda);
        label = 'bilinear';
        fi = 3;
    elseif isinf(nRank)
        M = augment(M, 1);
        [theta, sigmasq] = regularizer_hyperparams(M, R, 'ARD');
        lambda = theta; % no sigmasq, right?
        disp(['lambda=' num2str(lambda)]);
        C = fullrank_lsq(M, R, lambda);
        label = 'full rank';
        fi = 5;
    else
        M = augment(M, nRank);
        if nargin < 5
            lambda = 1;
        end
        [C, w_h, b_h] = alternating_lsq(M, R, lambda);
        label = ['rank-' num2str(nRank)];
        fi = 4;
    end
    
    % calculate response
    Rh = response(M, C);

    if ~isinf(nRank)
        ploti1(fi, label, C, b_h, w_h);
    else
        figure(fi); clf;
        ploti(C);
    end

end

%% plot
function ploti(C)

    image(C(2:end, 2:end));
    title('C')

end
function ploti1(i, label, C, b_h, w_h)

    figure(i); clf;

    subplot(131)
    plot(b_h(2:end), 'ro-')
    title(['b weights (' label ')'])

    subplot(132)
    plot(w_h(2:end), 'bo-')
    title(['w weights (' label ')'])

    subplot(133)
    ploti(C);

end
%%
function Rh = response(M, C)
    nTimesteps = size(M, 3);
    Rh = zeros(1, nTimesteps);
    for t = 1:nTimesteps
        Rh(t) = sum(sum(C'.*M(:,:,t)));
    end
end

function M2 = reshape_M(M)
    nBases = size(M, 1);
    nLags = size(M, 2);
    nTimesteps = size(M, 3);
    M2 = nan(nLags*nBases, nTimesteps);
    for t = 1:nTimesteps
       M2(:,t) = reshape(M(:,:,t), [1 nLags*nBases]);
       assert(isequal(reshape(M2(:,t), [nBases nLags]), M(:,:,t)));
    end
end

function C = fullrank_lsq(M, R, lambda)
    % C = (M*M')\(M*R') but M is tensor so we gotta reshape before/after
    nBases = size(M, 1);
    nLags = size(M, 2);
    M2 = reshape_M(M);
    Reg = diag_regularizer(lambda, size(M2,1));
    C2 = (M2*M2' + Reg)\(M2*R');
    C = reshape(C2, [nBases nLags])';
end

function Reg = diag_regularizer(lambda, d)
    if numel(lambda) == 1
        lambda = lambda*ones(d, 1);
    end
    Reg = diag(lambda);
end
%%
function [theta, sigmasq] = regularizer_hyperparams(M, R, kind)
    % finds regularizer hyperparameter using fixed-point algorithm
    % kind: 'ridge', 'ARD'
    % source: Park, Pillow (2011) Methods
    
    % controls # of iterations
    tol = 1e-7;
    maxiters = 1000;
    
    % variables reused
    M2 = reshape_M(M)';
    n = numel(R);
    d = size(M2, 2); % parameter dimensionality
    stim_cov = M2'*M2;
    sta = M2'*R';
    C = fullrank_lsq(M, R, 1e-5); % ML estimate
    
    % initial guesses
    sigmasqs = nan(maxiters+1, 1);
    switch kind
        case 'ridge'
            thetas = nan(maxiters+1, 1);
            errs = R - response(M, C);
            thetas(1) = 1e-6;
            sigmasqs(1) = errs*errs' / n;
        case 'ARD'
            figure; colormap(gray); imagesc(reshape(sta, size(M,1), size(M,2))); title('sta');
            thetas = nan(maxiters+1, d);           
            [th, sqs] = regularizer_hyperparams(M, R, 'ridge');
            thetas(1, :) = th;
            sigmasqs(1) = sqs;
    end
    
    % iterate
    for ii=1:maxiters
        switch kind
            case 'ridge'
                [thetas, sigmasqs] =  ridge_update(thetas, sigmasqs, ii, n, d, stim_cov, sta, M2, R);
            case 'ARD'
                [thetas, sigmasqs] =  ARD_update(thetas, sigmasqs, ii, n, d, stim_cov, sta, M2, R);
        end
        % check if changes in updates are within tolerance
        if abs(sigmasqs(ii+1) - sigmasqs(ii)) < tol
            break;
        end
    end
    theta = thetas(ii+1,:);
    sigmasq = sigmasqs(ii+1);
end

function [lambda, mu] = posterior_mean_and_cov(stim_cov, sta, d, theta, sigmasq)
    % update posterior mean and covariance
    lambda = inv(stim_cov/sigmasq + diag_regularizer(theta, d));
    mu = lambda*sta/sigmasq;
end

function [thetas, sigmasqs] = ridge_update(thetas, sigmasqs, ii, n, d, stim_cov, sta, M2, R)
    [lambda, mu] = posterior_mean_and_cov(stim_cov, sta, d, thetas(ii), sigmasqs(ii));
    thetas(ii+1) = (d - thetas(ii)*trace(lambda))/(mu'*mu);
	errs = (R' - M2*mu)';
	sigmasqs(ii+1) = (errs*errs')/(n - d - thetas(ii)*trace(lambda));
end

function [thetas, sigmasqs] = ARD_update(thetas, sigmasqs, ii, n, d, stim_cov, sta, M2, R)
    if ii == 10000
        keyboard
    end
    [lambda, mu] = posterior_mean_and_cov(stim_cov, sta, d, thetas(ii,:), sigmasqs(ii));
    lambda_diag = diag(lambda);
    thetas(ii+1, :) = (1 - thetas(ii,:)'.*lambda_diag)./mu;
    thetas(ii+1, isnan(thetas(ii+1,:))) = 1e-8;
%     colormap(gray); imagesc(lambda); drawnow; pause(0.4);
%     plot(mu, 'o'); drawnow; pause(0.4);
%     plot(thetas(ii+1,:), 'o'); drawnow; pause(0.5);
    plot(lambda_diag, 'o'); drawnow; pause(0.6);
    errs = (R' - M2*mu)';
    sigmasqs(ii+1, :) = (errs*errs')/(n - d - thetas(ii,:)*lambda_diag);
end
%%
function Reg = ahrens_regularizer(lambda, bi, mu, nLags)
    % this is ignoring the penalty for high second derivatives,
    % since that involves calculating something specific to our basis
    S = linspace(min(mu), max(mu), 100);
    B = bsxfun(bi, S', mu);
    FirstDeriv = zeros(numel(mu), numel(mu));
    for i = 1:numel(mu)
        FirstDeriv(i, :) = (B'*B(:, i))';
    end
    % is this right? seems to not depend on lags.
    % also, now we're one dim short, since we're not including const
    % i.e. Reg is square with dim nLags*(nBases-1)
    Reg = lambda*repmat(FirstDeriv, nLags, nLags);
end

function C = svd_fullrank(M, k)
    % rank-k approximation of C, plus const
    [u,s,v] = svds(C, k);
    const = C(1, 1);
    C = u*s*v';
    C(1,1) = const; % still need to keep const for low rmse
end

%%
function Ma = augment(M, k)
    % k is number of blocks of M
    % output for k=1, in block form, is: [1 0; 0 M]
    % output for k=2, in block form, is: [1 0 0; 0 M 0; 0 0 M]
    
    function Ap = augment_inner(A)
        Ap = zeros(k*size(A) + 1);
        Ap(1,1) = 1;
        for ii=1:k
            first = (ii-1)*size(A) + 2;
            last = ii*size(A) + 1;
            Ap(first(1):last(1), first(2):last(2)) = A;
        end
    end
    
    newdim = @(M, i) k*size(M, i) + 1;
    Ma = zeros(newdim(M,1), newdim(M,2), size(M,3));
    for i = 1:size(M,3)
        Ma(:,:,i) = augment_inner(M(:,:,i));
    end

end
%%
function [C, w_h, b_h] = alternating_lsq(M, R, lambda)
    % recover weights using alternating least squares
    nBases = size(M, 1);
    b_h = ones(nBases, 1); % initial guess
    niters = 1000;
    for i = 1:niters
       w_h = fit_w(M, b_h, R);%, lambda(1));
       b_h = fit_b(M, w_h, R);%, lambda(2));
    end
    C = w_h*b_h';

end
%%
function B = mult_b(M, b)

    nLags = size(M, 2);
    nTimesteps = size(M, 3);
    B = zeros(nLags, nTimesteps);
    for t = 1:nTimesteps
        B(:,t) = b'*M(:,:,t);
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
function w_h = fit_w(M, b, R, lambda)
    if nargin < 4
        lambda = 0.0;
    end
    B = mult_b(M, b);
    Reg = diag_regularizer(lambda, size(B,1));
    w_h = (B*B' + Reg)\(B*R');

end
%%
function b_h = fit_b(M, w, R, lambda)

    W = mult_w(M, w);
    b_h = (W*W')\(W*R');

end
