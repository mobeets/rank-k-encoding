b = 10;
n = 100;
nw = 100;

xi = linspace(-b, b, n);
yi = xi;
idx = randi(n, 2, nw);
x = xi(idx(1,:));
y = yi(idx(2,:));
pts = [x; y]';

ps = mvnpdf(pts, [0 0], [b b]);
figure; scatter(x, y, 1e4*ps + 2, 'filled', 'k');
D = squareform(pdist(pts, 'euclidean'));
