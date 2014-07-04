
%% messing with tensors

nx = 10;
ny = 8;
nT = 100;

x = 1:nx;
y = 1:ny;

[xx, yy, tt] = ndgrid(x, y, 1:nT);

%%
figure(11); clf

M = nan(nT, nx*ny);
for ii = 1:nT
   M(ii,:) = reshape(xx(:,:,ii), [1 nx*ny]);
%    M(ii,:) = reshape(xx(:,:,ii), [nx*ny 1])';
end

imagesc(M)
%%
    

Xreshape = reshape(xx, [nx*nx 1]);
Xfull = reshape(Xreshape, [nx nx]);