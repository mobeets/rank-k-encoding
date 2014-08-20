function [D, w] = randomDistancesGaussianWeights(b, n, nw, doPlot)
    % returns a square (nw-by-nw) matrix of distances between randomly
    % selected 2d positions, and a vector of weights, where the weight of a
    % point is given by the pdf of Normal((0,0), (b,b)) at that point.
    %
    % b is bounds, and variance of gaussian
    % n is step size within those bounds
    % nw is number of positions to be drawn
    % doPlot, if true, shows points and their respective weights

    if nargin < 1
        b = 10;
        n = 100;
        nw = 100;
    elif nargin < 4
        doPlot = true;
    end

    xi = linspace(-b, b, n);
    yi = xi;
    idx = randi(n, 2, nw);
    x = xi(idx(1,:));
    y = yi(idx(2,:));
    pts = [x; y]';

    w = mvnpdf(pts, [0 0], [b b]);
    if doPlot
        figure; scatter(x, y, 1e4*w + 2, 'filled', 'k');
    end
    D = squareform(pdist(pts, 'euclidean'));
end
