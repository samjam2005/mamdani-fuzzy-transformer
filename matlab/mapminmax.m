function y = mapminmax(x, ymin, ymax)
%MAPMINMAX Compatibility implementation for the forward scaling call used here.

    if nargin < 2
        ymin = -1;
    end
    if nargin < 3
        ymax = 1;
    end
    row_min = min(x, [], 2);
    row_max = max(x, [], 2);
    row_range = row_max - row_min;
    y = zeros(size(x));
    variable_rows = row_range > 0;
    y(variable_rows, :) = ymin + (ymax - ymin) .* ...
        (x(variable_rows, :) - row_min(variable_rows)) ./ row_range(variable_rows);
    y(~variable_rows, :) = ymin;
end
