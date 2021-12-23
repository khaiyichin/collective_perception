clearvars
close all
clc

%% Parameters
param_simple_single_agent_no_motion % call the parameter file

%% Simulation

% Define agent's conditional probabilities
p_b_b = b;      % black given black
p_w_b = 1 - b;  % black given white
p_w_w = w;      % white given white
p_b_w = 1 - w;  % white given black

p_z_1_sim = zeros(sim_cycles, N); % probability that the agent sees a black tile (h/t)
agent_observations = zeros(sim_cycles, N); % agent observations
f = zeros(sim_cycles, N); % actual fill ratios (each column in the same row should have the same value)

% Feed the agent with tiles occurences
for i = 1:sim_cycles

    prev_obs = 0;
    curr_obs = 0;
    
    tile_occurences = generate_tiles(desired_fill_ratio, N);

    f(i, :) = ones(1, N) .* nnz(tile_occurences)/N; % actual fill ratio
    
    for j = 1:N
        
        if tile_occurences(j) % if black tile encountered
            curr_obs = observe_color(tile_occurences(j), p_b_b);
        else % white tile is encountered
            curr_obs = observe_color(tile_occurences(j), p_w_w);
        end
        
        agent_observations(i, j) = curr_obs;
        p_z_1_sim(i, j) = ( prev_obs + curr_obs ) / j; % record estimation (h/t)
        prev_obs = prev_obs + curr_obs;
        
    end
end

%% Analysis

% Compute the actual probability P(z = 1)
p_z_1_act = b * mean(f, 1) + (1 - w) * (1 - mean(f, 1));

% Compute the mean and std dev of simulated probability P(z = 1)
p_z_1_sim_mean = mean(p_z_1_sim, 1);
p_z_1_sim_std = std(p_z_1_sim, 0, 1);

% Compute estimated fill ratio f_hat based on simulated observations
f_hat = zeros(sim_cycles, N);

for i = 1:sim_cycles
    for j = 1:N

        if p_z_1_sim(i, j) <= (1 - w)
            f_hat(i, j) = 0;
        elseif p_z_1_sim(i, j) >= b
            f_hat(i, j) = 1;
        else
            f_hat(i, j) = (p_z_1_sim(i, j) + w - 1) / (b + w - 1);
        end
        
    end
end

% Compute the mean and std dev of f_hat across the simulation cycles
f_hat_mean = mean(f_hat, 1);
f_hat_std = std(f_hat, 0, 1);

% Compute the mean and std dev of f_hat_err across the simulation cycles
f_hat_err_mean = mean(f - f_hat, 1);
f_hat_err_std = sqrt( var(f - f_hat, 0, 1) );

% Compute the variance of f_hat based on simulated observations
fisher_inv_sqrt = zeros(sim_cycles, N);

for i = 1:sim_cycles
    for j = 1:N
        h = sum(agent_observations(i, 1:j));
        fisher_inv_sqrt(i, j) = sqrt(compute_fisher_inv(h, j, b, w));
    end
end

% Compute the mean and std dev of fisher_inv_sqrt across the simulation
% cycles
fisher_inv_sqrt_mean = mean(fisher_inv_sqrt, 1);
fisher_inv_sqrt_std = std(fisher_inv_sqrt, 0, 1);

%% Plot generation

% Plot the simulated probability w.r.t. observations
conf_bounds_p_z_1 = [ p_z_1_sim_mean + p_z_1_sim_std, p_z_1_sim_mean(end:-1:1) - p_z_1_sim_std(end:-1:1) ];

figure
p = fill( [1:N, N:-1:1], conf_bounds_p_z_1, [0.8, 0.8, 1] );
p.FaceColor = [0.8 0.8 1];
p.EdgeColor = 'none'; 

hold on
plot(1:N, p_z_1_sim_mean, 'Color', [0.0, 0.447, 0.741])
ylim( [max( min(conf_bounds_p_z_1), 0 ), min( max(conf_bounds_p_z_1), 1 )] )

plot([1,N], ones(1, 2)*p_z_1_act(1), '-.k')

title("Probability of black tiles from simulated observations",...
    'Interpreter', 'latex', 'Fontsize', 14)
xlabel('Number of observations', 'Interpreter', 'latex', 'Fontsize', 14)
ylabel('$P(z=1)$', 'Interpreter', 'latex', 'Fontsize', 14)
legend("Sample 1\sigma of " + sim_cycles + " simulated probabilities",...
    "Sample mean of " + sim_cycles + " simulated probabilities" ,...
    "Computed probability using L.o.T.P. = " + p_z_1_act(1), "Location", "Best")
grid on
hold off

% Plot the estimated fill ratio w.r.t. observations
conf_bounds_f_hat = [ f_hat_mean + f_hat_std, f_hat_mean(end:-1:1) - f_hat_std(end:-1:1) ];

figure
p = fill( [1:N, N:-1:1], conf_bounds_f_hat, 'b' );
p.FaceColor = [0.8 0.8 1];
p.EdgeColor = 'none'; 

hold on
plot(1:N, f_hat_mean)
ylim( [max( min(conf_bounds_f_hat), 0 ), min( max(conf_bounds_f_hat), 1 )] )

plot([1,N], ones(1, 2)*mean(f(:, 1), 1), '-.k')

title('Estimated $\hat{f}$ from simulated observations', 'Interpreter', 'latex', 'Fontsize', 14)
xlabel('Number of observations', 'Interpreter', 'latex', 'Fontsize', 14)
ylabel('Estimated fill ratio, $\hat{f}$', 'Interpreter', 'latex', 'Fontsize', 14)
legend("Sample 1\sigma of " + sim_cycles + " estimated fill ratios",...
    "Sample mean of " + sim_cycles + " estimated fill ratios",...
    "Sample mean of " + sim_cycles + " actual fill ratios = " + mean(f(:, 1), 1), "Location", "Best")
grid on
hold off

% Plot the estimated fill ratio error w.r.t. observations
conf_bounds_f_hat_err = [ f_hat_err_mean + f_hat_err_std, f_hat_err_mean(end:-1:1) - f_hat_err_std(end:-1:1) ];

figure
p = fill( [1:N, N:-1:1], conf_bounds_f_hat_err, 'b' );
p.FaceColor = [0.8 0.8 1];
p.EdgeColor = 'none'; 

hold on
plot(1:N, f_hat_err_mean)
ylim( [max( min(conf_bounds_f_hat_err), -0.4 ), min( max(conf_bounds_f_hat_err), 1.4 )] )

plot([1,N], zeros(1, 2), '-.k')

title('Estimated $\hat{f}$ error from simulated observations', 'Interpreter', 'latex', 'Fontsize', 14)
xlabel('Number of observations', 'Interpreter', 'latex', 'Fontsize', 14)
ylabel('Estimated fill ratio error, $f - \hat{f}$', 'Interpreter', 'latex', 'Fontsize', 14)
legend("Sample 1\sigma of " + sim_cycles + " estimated fill ratio errors",...
    "Sample mean of " + sim_cycles + " estimated fill ratio errors", "Location", "Best")
grid on
hold off

% Plot the inverse of Fisher information (variance)
figure

semilogy(1:N, fisher_inv_sqrt_mean, 2:N, fisher_inv_sqrt_std(2:end)) % due to the way the variance is set up, the first observation will always yield te same result
hold on
title('Root inverse Fisher information of $\hat{f}$', 'Interpreter', 'latex', 'Fontsize', 14)
xlabel('Number of observations', 'Interpreter', 'latex', 'Fontsize', 14)
ylabel('$\sqrt{\mathrm{Var}[\hat{f}_{general}]}$', 'Interpreter', 'latex', 'Fontsize', 14)
legend("Sample mean of $\sqrt{\mathrm{Var}[\hat{f}_{general}]}$ over " + sim_cycles + " experiments",...
    "Sample 1$\sigma$ of $\sqrt{\mathrm{Var}[\hat{f}_{general}]}$ over " + sim_cycles + " experiments",...
    "Interpreter", "latex", "Location", "Best")
grid on
hold off

%% Functions
function tiles = generate_tiles(fill_ratio, total_tiles)

    % Generate actual number of colored tiles
    tiles = binornd(1, ones(1, total_tiles) * fill_ratio);
end

function observed_color = observe_color(tile_color, color_prob)
% tile_color must be in the form of 0 or 1

    % Decide whether to go with tile_color based on color_prob
    if (rand(1) < color_prob)
        observed_color = tile_color;
    else
        observed_color = 1 - tile_color;
    end
end

function fisher_inv = compute_fisher_inv(h, t, b, w)
    if h <= (1-w)*t
        fisher_inv = w^2 * (w-1)^2 / ...
            ( (b+w-1)^2 * (t*w^2 - 2*(t-h)*w + (t-h)) );
    elseif h >= b*t
        fisher_inv = b^2 * (b-1)^2 / ...
            ( (b+w-1)^2 * (t*b^2 - 2*h*b + h) );
    else
        fisher_inv = h * (t-h) / ...
            ( t^3 * (b+w-1)^2 );
    end
end