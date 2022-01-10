%% Plot heatmap and surface figures
close all
clearvars

%% Parameters
param_plot_heat_surf % call the parameter file

%% Process and create figures

x_axis_lbl = string(xrange);
y_axis_lbl = string(yrange);

% Create surface plot x and y meshgrids
[sensor_probs, fill_ratios] = meshgrid(xrange, yrange);

% Plot estimation error figures (w.r.t. to desired fill ratio)
f_hat_heatmap_data = import_heatmap_data(f_hat_filepath, length(xrange));

error_data = abs(yrange' - f_hat_heatmap_data);

figure
h_f_hat = heatmap(x_axis_lbl, y_axis_lbl, log10(error_data), 'Colormap', summer);
h_f_hat.XLabel = 'Sensor probability, P(b|b) = P(w|w)';
h_f_hat.YLabel = 'Black tile fill ratio';
h_f_hat.Title = "Relative estimation error (w.r.t. des. fill ratio) of " + num2str(num_obs) + " observations by " + num2str(sim_cycles) + " agents (10^x scale)";

figure
s_f_hat = surf(sensor_probs, fill_ratios, error_data);
s_f_hat.Parent.XLabel.String = 'Sensor probability, P(b|b) = P(w|w)';
s_f_hat.Parent.YLabel.String = 'Black tile fill ratio';
s_f_hat.Parent.ZLabel.String = 'Relative error';
s_f_hat.Parent.ZScale = 'log';
s_f_hat.Parent.Title.String = "Relative estimation error (w.r.t. des. fill ratio) of " + num2str(num_obs) + " observations by " + num2str(sim_cycles) + " agents";

% Plot estimation error figures (w.r.t. to average fill ratio)
fill_ratio_data = import_sim_fill_ratios(sim_fill_ratios_filepath, yrange);

error_data_2 = abs(fill_ratio_data(:, 2) - f_hat_heatmap_data);

figure
h_f_hat_2 = heatmap(x_axis_lbl, y_axis_lbl, log10(error_data_2), 'Colormap', summer);
h_f_hat_2.XLabel = 'Sensor probability, P(b|b) = P(w|w)';
h_f_hat_2.YLabel = 'Black tile fill ratio';
h_f_hat_2.Title = "Relative estimation error (w.r.t. avg. fill ratio) of " + num2str(num_obs) + " observations by " + num2str(sim_cycles) + " agents (10^x scale)";

figure
s_f_hat_2 = surf(sensor_probs, fill_ratios, error_data_2);
s_f_hat_2.Parent.XLabel.String = 'Sensor probability, P(b|b) = P(w|w)';
s_f_hat_2.Parent.YLabel.String = 'Black tile fill ratio';
s_f_hat_2.Parent.ZLabel.String = 'Relative error';
s_f_hat_2.Parent.ZScale = 'log';
s_f_hat_2.Parent.Title.String = "Relative estimation error (w.r.t. avg. fill ratio) of " + num2str(num_obs) + " observations by " + num2str(sim_cycles) + " agents";

% Plot inverse Fisher information heatmap
fisher_inv_heatmap_data = import_heatmap_data(fisher_inv_filepath, length(xrange));

norm_factor_fisher_inv = max( max( fisher_inv_heatmap_data ) );
norm_fisher_inv_heatmap_data = fisher_inv_heatmap_data ./ norm_factor_fisher_inv;

figure
h_fisher_inv = heatmap(x_axis_lbl, y_axis_lbl, log10(fisher_inv_heatmap_data), 'Colormap', summer);
h_fisher_inv.XLabel = 'Sensor probability, P(z=b|x=b) = P(z=w|x=w)';
h_fisher_inv.YLabel = 'Black tile fill ratio';
h_fisher_inv.Title = "Standard error of " + num2str(num_obs) + " observations by " + num2str(sim_cycles) + " agents (10^x scale)";

figure
s_fisher_inv = surf(sensor_probs, fill_ratios, fisher_inv_heatmap_data);
s_fisher_inv.Parent.XLabel.String = 'Sensor probability, P(b|b) = P(w|w)';
s_fisher_inv.Parent.YLabel.String = 'Black tile fill ratio';
s_fisher_inv.Parent.ZLabel.String = 'Standard error';
s_fisher_inv.Parent.Title.String = "Standard error of " + num2str(num_obs) + " observations by " + num2str(sim_cycles) + " agents";
s_fisher_inv.Parent.ZScale = 'log';