function result = run_bcm_volatility(source_dir, data_path, output_dir)
%RUN_BCM_VOLATILITY Run the archived ieRSPOP++ BCM volatility path.
% This invokes the original online trainer and error calculator while avoiding
% the plotting-only defects in ron_test_vol.m.

    if ~isfolder(source_dir)
        error("Source directory not found: %s", source_dir);
    end
    if ~isfile(data_path)
        error("Data file not found: %s", data_path);
    end
    if ~isfolder(output_dir)
        mkdir(output_dir);
    end

    addpath(fileparts(mfilename("fullpath")), "-begin");
    addpath(source_dir, "-end");
    loaded = load(data_path, "voldata");
    series = loaded.voldata(:);
    sequence_length = 5;
    examples = numel(series) - sequence_length;
    inputs = zeros(examples, sequence_length);
    targets = zeros(examples, 1);
    for index = 1:examples
        inputs(index, :) = series(index:index + sequence_length - 1);
        targets(index) = series(index + sequence_length);
    end

    first_test_row = 487;
    evaluation_targets = targets(first_test_row:end);
    evalc("ensemble = update_ron_trainOnline(inputs, targets, 'ierspop', 120);");
    ensemble = ron_calcErrors(ensemble, evaluation_targets);
    actual = evaluation_targets;
    predicted = ensemble.predicted(first_test_row:end);
    error_values = actual - predicted;
    denominator = sum((actual - mean(actual)).^2);
    result = struct( ...
        "model", "ieRSPOP++ BCM", ...
        "evaluation", "chronological prequential (predict then learn)", ...
        "total_examples", examples, ...
        "test_examples", numel(actual), ...
        "first_test_window_matlab", first_test_row, ...
        "rmse", sqrt(mean(error_values.^2)), ...
        "mae", mean(abs(error_values)), ...
        "pearson_r", corr2(actual, predicted), ...
        "predictive_r_squared", 1 - sum(error_values.^2) / denominator, ...
        "reported_rule_count", ensemble.num_rules);

    writematrix([(first_test_row:examples)', actual, predicted, error_values], ...
        fullfile(output_dir, "predictions.csv"));
    fid = fopen(fullfile(output_dir, "metrics.json"), "w");
    fprintf(fid, "%s\n", jsonencode(result, PrettyPrint=true));
    fclose(fid);
    save(fullfile(output_dir, "ensemble.mat"), "ensemble", "result");
end
