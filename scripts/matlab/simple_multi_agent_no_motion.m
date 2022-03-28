clearvars
close all
clc

%{
Assumptions:
- full communication with all agents
- same sensor quality among all agents
- communicate after each observation
%}

%% Parameters

n_agents = 50;                  % number of agents
n_obs = 1000;                       % total number of observations
obs_comms_ratio = 1;            % ratio of observations per communication round
b = 0.65;                        % sensor probability to black tile
w = b;                          % sensor probability to white tile
n_experiments = 5;                % number of experiments
desired_fill_ratio = 0.8;   % fill ratio, f (can be set to rand(1))

% Call parameter file otherwise

%% Simulation

% Define agent's conditional probabilities
p_b_b = b;      % black given black
p_w_b = 1 - b;  % black given white
p_w_w = w;      % white given white
p_b_w = 1 - w;  % white given black

% Initialize data containers
p_z_1_sim = zeros(n_agents, n_obs, n_experiments); % probability that the agent sees a black tile (h/t)
agent_observations = zeros(n_agents, n_obs, n_experiments); % agent observations
f = zeros(n_agents, 1, n_experiments); % actual fill ratios (each column in the same row should have the same value)
x_hat = zeros(n_agents, n_obs, n_experiments); % local estimates
x_bar = zeros(n_agents, n_obs, n_experiments); % social estimates
x = zeros(n_agents, n_obs, n_experiments); % final local guess
fisher_inv = zeros(n_agents, n_obs, n_experiments); % fisher inverse (variances)
fisher_inv_bar = zeros(n_agents, n_obs, n_experiments); % social fisher inverse (variances)

% Run through all experiments
for exp_ind = 1:n_experiments

    % Generate observed tiles for agents
    tile_occurences = generate_tiles(n_agents, desired_fill_ratio, n_obs);
    black_tiles_per_row = sum(tile_occurences ~= 0, 2);
    f(:, :, exp_ind) = ones(n_agents, 1) .* black_tiles_per_row ./ n_obs; % actual fill ratio

    prev_obs = zeros(n_agents, 1); % initialize observation collection

    % Go through each observation cycle
    for obs_ind = 1:n_obs

        % Perform local actions
        curr_obs = zeros(n_agents, 1);

        % Make local observation for each agent
        for agt_ind = 1:n_agents

            if tile_occurences(agt_ind, obs_ind) % if black tile encountered
                curr_obs(agt_ind,1) = observe_color(tile_occurences(agt_ind, obs_ind), p_b_b);
            else % white tile is encountered
                curr_obs(agt_ind,1) = observe_color(tile_occurences(agt_ind, obs_ind), p_w_w);
            end    
        end

        % Perform local estimate
        agent_observations(:, obs_ind, exp_ind) = curr_obs;
        prev_obs = prev_obs + curr_obs;
        p_z_1_sim(:, obs_ind, exp_ind) = prev_obs / obs_ind; % record estimation (h/t)

        % Local computation
        for agt_ind = 1:n_agents

            % Make local estimate
            x_hat(agt_ind, obs_ind, exp_ind) = ...
                estimate_x( p_z_1_sim(agt_ind, obs_ind, exp_ind), b, w );

            % Sum all observed black tile occurrences
            h = sum( agent_observations(agt_ind, 1:obs_ind, exp_ind) );

            % Compute Fisher inverse based on h
            fisher_inv(agt_ind, obs_ind, exp_ind) = ...
                compute_fisher_inv(h, obs_ind, b, w);
        end

        % Compute social values once all agents have processed local
        % estimate
        for agt_ind = 1:n_agents
            [x_bar(agt_ind, obs_ind, exp_ind),...
                fisher_inv_bar(agt_ind, obs_ind, exp_ind)] = ...
                    solve_g_x( x(1:end ~= agt_ind, obs_ind, exp_ind),...
                               fisher_inv(1:end ~= agt_ind, obs_ind, exp_ind) );
        end

        % Solve primal function (analytical form exists)
        if obs_ind ~= n_obs
            x(:, obs_ind+1, exp_ind) = ...
                solve_f_x(x_hat(:, obs_ind, exp_ind),... % local estimate
                          x_bar(:, obs_ind, exp_ind),... % social estimate
                          1./fisher_inv(:, obs_ind, exp_ind),... % local fisher info (alpha)
                          1./fisher_inv_bar(:, obs_ind, exp_ind)); % social fisher info (rho)
        end
    end
end

%% Analysis

% Compute the mean and std
x_hat_trajectories_avg = zeros(n_experiments, n_obs);
x_hat_trajectories_std = zeros(n_experiments, n_obs);
x_trajectories_avg = zeros(n_experiments, n_obs);
x_trajectories_std = zeros(n_experiments, n_obs);
alpha_trajectories_avg = zeros(n_experiments, n_obs);
alpha_trajectories_std = zeros(n_experiments, n_obs);

for i = 1:n_experiments
    x_hat_trajectories_avg(i, :) = mean(x_hat(:, :, i), 1);
    x_hat_trajectories_std(i, :) = std(x_hat(:, :, i), 0, 1);
    x_trajectories_avg(i, :) = mean(x(:, :, i), 1);
    x_trajectories_std(i, :) = std(x(:, :, i), 0, 1);
    alpha_trajectories_avg(i, :) = mean(1./fisher_inv(:, :, i), 1);
    alpha_trajectories_std(i, :) = std(1./fisher_inv(:, :, i), 0, 1);
end

%% Plot data

% Plot agent averaged x trajectory for 1st experiment
% conf_bounds_x = [ x_trajectories_avg(1, 1:end) + x_trajectories_std(1, 1:end),...
%     x_trajectories_avg(1, end:-1:1) - x_trajectories_std(1, end:-1:1) ];
% 
% figure
% p = fill( [1:n_obs, n_obs:-1:1], conf_bounds_x, 'b' );
% p.FaceColor = [0.8 0.8 1];
% p.EdgeColor = 'none';
% hold on
% 
% plot(1:n_obs, x_trajectories_avg(1, 1:end));
% title("Average $x$ across " + n_agents + " agents",...
%     'Interpreter', 'latex', 'Fontsize', 14)
% xlabel('Number of observations', 'Interpreter', 'latex', 'Fontsize', 14)
% ylabel('$x$', 'Interpreter', 'latex', 'Fontsize', 14)
% grid on
% hold off

% Plot x trajectories across different experiments
figure

for i = 1:n_experiments
    subplot(n_experiments, 2, 2*i)
    conf_bounds_x = [ (desired_fill_ratio - x_trajectories_avg(i, :)) + x_trajectories_std(i, 1:end),...
        (desired_fill_ratio - x_trajectories_avg(i, end:-1:1)) - x_trajectories_std(i, end:-1:1) ];
    p = fill( [1:n_obs, n_obs:-1:1], conf_bounds_x, 'b' );
    p.FaceColor = [0.8 0.8 1];
    p.EdgeColor = 'none';
    hold on

    title("Agent-averaged $x-x^*$ across " + n_agents + " agents, experiment " + i,...
    'Interpreter', 'latex', 'Fontsize', 14)

    plot(1:n_obs, desired_fill_ratio - x_trajectories_avg(i, 1:end));

    grid on
    hold off
end
xlabel('Number of observations', 'Interpreter', 'latex', 'Fontsize', 14)

subplot(n_experiments, 2, [1:2:2*n_experiments])
total_mean = mean(x_trajectories_avg, 1);
total_std = std(x_trajectories_avg, 0, 1);
conf_bounds_total = [ (desired_fill_ratio - total_mean) + total_std(1:end),...
    (desired_fill_ratio - total_mean(end:-1:1)) - total_std(end:-1:1) ];
p = fill( [1:n_obs, n_obs:-1:1], conf_bounds_total, 'b' );
p.FaceColor = [0.8 0.8 1];
p.EdgeColor = 'none';
hold on
plot(1:n_obs, desired_fill_ratio - total_mean)
xlim([50 n_obs])
ylim([-0.1 0.1])
title("Absolute final estimate error across " + n_agents + " agents and " + n_experiments + " experiments", "Interpreter", "Latex", "Fontsize", 14);
xlabel('Number of observations', 'Interpreter', 'latex', 'Fontsize', 14)
ylabel('$x-x^*$', 'Interpreter', 'latex', 'Fontsize', 14)

legend("Sample 1$\sigma$ of $x^*-x$",...
    "Sample mean of $x^*-x$", "Interpreter", "Latex", "Location", "Best")
grid on
hold off

% Plot inverse of Fisher info across different agents
figure

for i = 1:n_experiments
    semilogy( 1:n_obs, mean(fisher_inv(:, 1:end, i), 1) );
    hold on
end

title("var[$\hat{x}$] agents for " + n_experiments + " experiments",...
    'Interpreter', 'latex', 'Fontsize', 14)
xlabel('Number of observations', 'Interpreter', 'latex', 'Fontsize', 14)
ylabel('var[$x$]', 'Interpreter', 'latex', 'Fontsize', 14)
grid on
hold off

% Investigate low-pass filtering properties of random agents
% input = x_hat, output = x
figure
% subplot(2,1,1)
% plot(1:n_obs, x_hat(1, :, 1));
% subplot(2,1,2)
% plot(1:n_obs, x(1,:,1))

exps = randi(n_experiments, [1,3]);
agts = randi(n_agents, [1,3]);

for i = 1:length(exps)
    subplot(length(exps), 1, i);
    plot( 1:n_obs, x_hat(agts(i), :, exps(i)), 1:n_obs, x(agts(i), :, exps(i)) );
    title("Random agent i=" + agts(i) + " from random experiment " + exps(i), 'Interpreter', 'latex', 'Fontsize', 14)
    ylabel('$x$ and $\hat{x}$', 'Interpreter', 'latex', 'Fontsize', 14)
    legend("$\hat{x}$", "$x$", 'Interpreter', 'latex', 'Fontsize', 14, "Location", "Best")
    grid on
end
xlabel('Number of observations', 'Interpreter', 'latex', 'Fontsize', 14)

% Analyze frequency response at the final observation step
denom = fisher_inv(:,end,1) + fisher_inv_bar(:,end,1); % evaluate at the end

D_inv = diag(ones(n_agents, 1)) ./ (n_agents -1) ; % inverse of degree matrix (fully connected)
A = double(~eye(n_agents)); % adjacency matrix
F_tilde = eye(n_agents) .* fisher_inv(:, end, 1) ./ denom;
F = F_tilde * D_inv * A;

G = eye(n_agents) .* fisher_inv_bar(:,end,1) ./ denom;

sys = ss(F, G, eye(n_agents), 0);

if n_agents < 4
    figure;
    bode_opt = bodeoptions;
    bode_opt.FreqUnits = 'Hz';
    bode(sys, bode_opt);
    grid on
end

figure;
sigma_opt = sigmaoptions;
sigma_opt.FreqUnits = 'Hz';
sigma(sys, sigma_opt);
grid on

%% Functions
function tiles = generate_tiles(n_agents, fill_ratio, total_tiles)

    % Generate actual number of colored tiles
    tiles = binornd(1, ones(n_agents, total_tiles) * fill_ratio);
end

function observed_color = observe_color(tile_color, color_prob)
% tile_color must be in the form of 0 or 1

    %{ 
    THE FOLLOWING SNIPPET ASSUMES ALL INCOMING TILE COLORS ARE BLACK

    % Generate column vector of random numbers
    obs_criteria = rand(n_agents, 1);

    % Decide whether to go with tile_color based on color_prob
    observed_color = double(obs_criteria < color_prob);

    %}

    if (rand(1) < color_prob)
        observed_color = tile_color;
    else
        observed_color = 1 - tile_color;
    end
end

% Provide local estimate
function x_hat = estimate_x(p_z_1_sim, b, w)
        if p_z_1_sim <= (1 - w)
            x_hat = 0;
        elseif p_z_1_sim >= b
            x_hat = 1;
        else
            x_hat = (p_z_1_sim + w - 1) / (b + w - 1);
        end
end

% Compute variance of local estimate
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

% Solve primal function
% for now use the simplified kalman filter combination equation
function x_next = solve_f_x(x_hat_arr, x_bar_arr, alpha_arr, rho_arr)
    x_next = (alpha_arr .* x_hat_arr + rho_arr .* x_bar_arr) ./...
        (alpha_arr + rho_arr);
end

% Compute social estimate and variance
function [x_bar, fisher_inv_bar] = solve_g_x(x_arr, fish_inv_arr)
    x_bar = mean(x_arr, 1);
    fisher_inv_bar = mean(fish_inv_arr, 1);
end