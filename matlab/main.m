function main()

    %% stimulus
    nTimesteps = 100;
    S = 5*randn(nTimesteps, 1);

    %% response
    nLags = 8;
    R = resp(S, nLags);
    
    %% fit (linear)
    Rh1 = linreg(S, R, nLags);

    %% fit (bilinear)
    Rh2 = rankreg(S, R, nLags, 2);
    
    %% fit (full rank)
%     Rh3 = rankreg(S, R, nLags, Inf);
    
    %% write results
    rmse = @(a, b) sqrt(sum((a-b).^2));
    disp(['rmse (linear) = ' num2str(rmse(R, Rh1))]);
    disp(['rmse (bilinear) = ' num2str(rmse(R, Rh2))]);
%     disp(['rmse (full rank) = ' num2str(rmse(R, Rh3))]);
    
    %% plot
    figure(5); clf; hold on;
    plot(R, 'r')
    plot(Rh1, 'c');
    plot(Rh2, 'b');
    title('response');

end
