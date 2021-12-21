%% Plot heatmap and surface figures
close all
clear vars

%% Parameters
param_plot_heat_surf % call the parameter file

%% Process and create figures

x_axis_lbl = string(xrange);
y_axis_lbl = string(yrange);

% Create surface plot x and y meshgrids
[sensor_probs, fill_ratios] = meshgrid(xrange, yrange);

% Plot estimation error figures
f_hat_heatmap_data = import_heatmap_data(f_hat_filepath, length(xrange));

error_data = abs(yrange' - f_hat_heatmap_data);
% norm_factor_error = max( max( error_data ) );
norm_error_data = error_data ./ yrange';

figure
h_f_hat = heatmap(x_axis_lbl, y_axis_lbl, log10(norm_error_data), 'Colormap', summer);
h_f_hat.XLabel = 'Sensor probability, P(b|b) = P(w|w)';
h_f_hat.YLabel = 'Black tile fill ratio';
h_f_hat.Title = 'Absolute estimation error of 1000 observations by 200 agents';

figure
s_f_hat = surf(sensor_probs, fill_ratios, norm_error_data);
s_f_hat.Parent.XLabel.String = 'Sensor probability, P(b|b) = P(w|w)';
s_f_hat.Parent.YLabel.String = 'Black tile fill ratio';
s_f_hat.Parent.ZLabel.String = 'Absolute error';
s_f_hat.Parent.ZScale = 'log';
s_f_hat.Parent.Title.String = 'Absolute estimation error of 1000 observations by 200 agents';

% Plot inverse Fisher information heatmap
fisher_inv_heatmap_data = import_heatmap_data(fisher_inv_filepath, length(xrange));

norm_factor_fisher_inv = max( max( fisher_inv_heatmap_data ) );
norm_fisher_inv_heatmap_data = fisher_inv_heatmap_data ./ norm_factor_fisher_inv;

figure
h_fisher_inv = heatmap(x_axis_lbl, y_axis_lbl, log10(fisher_inv_heatmap_data), 'Colormap', summer);
h_fisher_inv.XLabel = 'Sensor probability, P(z=b|x=b) = P(z=w|x=w)';
h_fisher_inv.YLabel = 'Black tile fill ratio';
h_fisher_inv.Title = 'Standard error of 1000 observations by 200 agents';

figure
s_fisher_inv = surf(sensor_probs, fill_ratios, fisher_inv_heatmap_data);
s_fisher_inv.Parent.XLabel.String = 'Sensor probability, P(b|b) = P(w|w)';
s_fisher_inv.Parent.YLabel.String = 'Black tile fill ratio';
s_fisher_inv.Parent.ZLabel.String = 'Standard error';
s_fisher_inv.Parent.Title.String = 'Standard error of 1000 observations by 200 agents';
s_fisher_inv.Parent.ZScale = 'log';