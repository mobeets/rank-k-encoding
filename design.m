function X = design(S, nLags)

nTimesteps = size(S, 1);
X = nan(nLags, nTimesteps-nLags);
for ii = nLags:nTimesteps
    X(:, ii-nLags+1) = S((ii-nLags+1):ii);
end
