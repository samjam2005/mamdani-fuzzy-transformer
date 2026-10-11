function y = gauss2mf(x, params)
%GAUSS2MF Compatibility implementation of MATLAB Fuzzy Logic Toolbox gauss2mf.
% params is [sigma1 center1 sigma2 center2].

    if numel(params) ~= 4
        error("gauss2mf expects [sigma1 center1 sigma2 center2]");
    end
    sigma1 = params(1);
    center1 = params(2);
    sigma2 = params(3);
    center2 = params(4);
    y = zeros(size(x));
    left = x <= center1;
    y(left) = exp(-((x(left) - center1).^2) ./ (2 * sigma1^2));
    y(~left) = exp(-((x(~left) - center2).^2) ./ (2 * sigma2^2));
end
