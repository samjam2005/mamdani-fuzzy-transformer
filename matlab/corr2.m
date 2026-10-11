function r = corr2(a, b)
%CORR2 Compatibility implementation of Image Processing Toolbox corr2.

    a = double(a(:));
    b = double(b(:));
    if numel(a) ~= numel(b)
        error("corr2 inputs must contain the same number of elements");
    end
    a = a - mean(a);
    b = b - mean(b);
    r = sum(a .* b) / sqrt(sum(a.^2) * sum(b.^2));
end
