%% Plot relative error of estimated fill ratios averaged across all fill ratios
close all
clearvars

%% Parameters
param_plot_mean_rel_err % call the parameter file

%% Process and create figures

% Remove perfect sensor data to prevent downwards skew on figure
if xrange(end) == 1.0
    xrange = xrange(1:end-1);
end

% Import data and compute relative error
[c200_o100_err_mean, c200_o100_err_std] = compute_rel_err(c200_o100_filepaths, xrange, yrange);
[c200_o250_err_mean, c200_o250_err_std] = compute_rel_err(c200_o250_filepaths, xrange, yrange);
[c200_o500_err_mean, c200_o500_err_std] = compute_rel_err(c200_o500_filepaths, xrange, yrange);
[c200_o1000_err_mean, c200_o1000_err_std] = compute_rel_err(c200_o1000_filepaths, xrange, yrange);

% Compute trendlines
c200_o100_err_avg_trend = mean(c200_o100_err_mean, 1);
c200_o250_err_avg_trend = mean(c200_o250_err_mean, 1);
c200_o500_err_avg_trend = mean(c200_o500_err_mean, 1);
c200_o1000_err_avg_trend = mean(c200_o1000_err_mean, 1);

% Plot data
sz_factor = 5e3;

figure
ax_sct = gca;
ax_sct.YScale = 'log';
ax_sct.XLim = [xrange(1)-0.025, xrange(end)+0.025];

grid on
hold on

dummy_for_marker_sz_lgd = plot(nan(1), nan(1), 'k');
dummy_for_solid_line_lgd = plot(nan(1), nan(1), 'k');

plot(xrange, c200_o100_err_avg_trend, 'Color', [0 0.4470 0.7410])
plot(xrange, c200_o250_err_avg_trend, 'Color', [0.8500 0.3250 0.0980]);
plot(xrange, c200_o500_err_avg_trend, 'Color', [0.9290 0.6940 0.1250]);
plot(xrange, c200_o1000_err_avg_trend, 'Color', [0.4940 0.1840 0.5560]);

scatter(xrange, c200_o100_err_mean, sz_factor .* c200_o100_err_std, [0 0.4470 0.7410], 's');
scatter(xrange, c200_o250_err_mean, sz_factor .* c200_o100_err_std, [0.8500 0.3250 0.0980], 'x');
scatter(xrange, c200_o500_err_mean, sz_factor .* c200_o100_err_std, [0.9290 0.6940 0.1250], 'o');
scatter(xrange, c200_o1000_err_mean, sz_factor .* c200_o100_err_std, [0.4940 0.1840 0.5560], 'd');

ax_sct.XLabel.String = 'Sensor probability, P(b|b) = P(w|w)';
ax_sct.YLabel.String = 'Average relative error';
ax_sct.Title.String = "Relative estimation error by " + num2str(sim_cycles) + " agents averaged across fill ratios";

legend_subset = [ax_sct.Children(10), ax_sct.Children(7), ...
    ax_sct.Children(4) ax_sct.Children(1), dummy_for_marker_sz_lgd, ...
    dummy_for_solid_line_lgd];
legend_subset_text = ["100 observations", "250 observations",...
    "500 observations", "1000 observations", ...
    "Marker size: (scaled) std. dev. across fill ratios", ...
    "Solid lines: average across 3 experiments"];

[~, lgd_obj_hdl] = legend(legend_subset, legend_subset_text, "Location", "Best");

% Modify dummy legend entries to turn line off and adjust position
lgd_obj_hdl(11).Visible = 'off';
lgd_obj_hdl(13).Visible = 'off';
lgd_obj_hdl(5).Position = [0.06 lgd_obj_hdl(5).Position(2:3)];
lgd_obj_hdl(6).Position = [0.06 lgd_obj_hdl(6).Position(2:3)];

function [error_data_mean, error_data_std] = compute_rel_err(filepaths_cell_arr, prob_range, fr_range)

    error_data_mean = zeros( size(filepaths_cell_arr, 1), length(prob_range) );
    error_data_std = zeros( size(filepaths_cell_arr, 1), length(prob_range) );

    for i = 1:size(filepaths_cell_arr, 1)

        % Import f_hat and fill ratio data pair for current experiment
        f_hat_mat = import_heatmap_data(filepaths_cell_arr{i}{1}, length(prob_range));
        fr_mat = import_sim_fill_ratios(filepaths_cell_arr{i}{2}, fr_range);
        
        % Compute relative error
        error_data = abs(fr_mat(:, 2) - f_hat_mat);
        
        % Find mean and std dev across all fill ratios for all the sensor
        % probabilities in current experiment
        error_data_mean(i, :) = mean(error_data, 1);
        error_data_std(i, :) = std(error_data, 0, 1);        
    end
end