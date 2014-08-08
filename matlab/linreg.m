function Rh = linreg(S, R, nLags)

    X = design(S, nLags);

    % Recover weights using linear least squares regression
    Xa = [ones(1,size(X,2)); X];
    w_h = (Xa*Xa')\(Xa*R');
    Rh = w_h'*Xa;

    ploti(2, w_h, Rh);

end

%% plot
function ploti(i, w_h, Rh)

    figure(i); clf;

    subplot(121)
    plot(w_h(2:end), 'r')
    title('weights (linear least squares)')

    subplot(122)
    plot(Rh, 'b')
    title('response (linear least squares)')

end
