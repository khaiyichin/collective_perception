clearvars
close all
clc

%{
Assumptions:
- number of experiments = 1
- number of agents = 1
%}

%% Parameters

n_calib = 0;                  % number of calibration steps
n_obs = 5e2;                       % number of observation steps
n_sensor_hyp = 9;  % number of sensor accuracy hypotheses starting from b > 0.5
obs_comms_ratio = 1;            % ratio of observations per communication round
b = zeros(n_sensor_hyp, 1);                        % sensor probability to black tile
acc = linspace(0.55, 0.95, n_sensor_hyp);
for i = 1:n_sensor_hyp; b(i) = acc(i); end
b_act = 0.95;
w = b;                          % sensor probability to white tile
w_act = b_act;
desired_fill_ratio = 0.85;   % fill ratio, f (can be set to rand(1))

% Call parameter file otherwise

%% Simulation

% Define agent's conditional probabilities
p_b_b = b_act;      % black given black
p_w_b = 1 - b_act;  % black given white
p_w_w = w_act;      % white given white
p_b_w = 1 - w_act;  % white given black

% Initialize data containers
avg_b_obsvd = zeros(n_obs, 1); % average number of black tiles observed (h/t)
agent_observations = zeros(n_obs, 1); % agent observations
x_hat = zeros(n_obs, n_sensor_hyp); % local estimates
alpha = zeros(n_obs, n_sensor_hyp); % local confidence
estimate_qualities = zeros(n_obs, n_sensor_hyp); % quality of x_hat estimates
chosen_sensor_acc = zeros(n_obs, 1); % indices of chosen sensor accuracy


% Generate observed tiles for agents
tile_occurences = generate_tiles(1, desired_fill_ratio, n_obs);
black_tiles_per_row = sum(tile_occurences ~= 0, 2);
f = black_tiles_per_row ./ n_obs; % actual fill ratio

prev_obs = 0; % initialize observation collection

% Go through each observation cycle
for obs_ind = 1:n_obs

    % Perform local actions
    curr_obs = 0;

    % Make local observation
    if tile_occurences(obs_ind) % if black tile encountered
        curr_obs = observe_color(tile_occurences(obs_ind), p_b_b);
    else % white tile is encountered
        curr_obs = observe_color(tile_occurences(obs_ind), p_w_w);
    end

    % Perform local estimate
    agent_observations(obs_ind) = curr_obs;
    prev_obs = prev_obs + curr_obs;
    avg_b_obsvd(obs_ind) = prev_obs / obs_ind; % record estimation (h/t)

    % Sum all observed black tile occurrences
    h = sum( agent_observations(1:obs_ind) );

    % Compute local values
    for sensor_ind = 1:n_sensor_hyp

        % Make local estimate
        x_hat(obs_ind, sensor_ind) = ...
            estimate_x( avg_b_obsvd(obs_ind), b(sensor_ind), w(sensor_ind) );

        % Compute confidence based on h
        alpha(obs_ind, sensor_ind) = ...
            compute_alpha(h, obs_ind, b(sensor_ind), w(sensor_ind));
    
        % Evaluate the quality of x_hat to determine which x_hat and alpha to communicate
        estimate_qualities(obs_ind, :) = evaluate_x_hat_quality(h, ...
            obs_ind, ...
            reshape(x_hat(obs_ind, :), [1, n_sensor_hyp]), ...
            reshape(b, [1, n_sensor_hyp]), ...
            reshape(w, [1, n_sensor_hyp]));
        
        % qualities is a 1xn_sensor_hyp vector
        [~, chosen_sensor_acc(obs_ind)] = max(estimate_qualities(obs_ind, :));
    end
end

%% Analysis

%% Plot data

% Plot estimate qualities
figure
for i = 1:n_sensor_hyp
    plot(1:n_obs, estimate_qualities(1:end, i), "DisplayName", num2str(b(i)));
    text(n_obs, estimate_qualities(end, i), num2str(b(i)));
    hold on
end
title("Estimate qualities",...
    'Interpreter', 'latex', 'Fontsize', 14)
xlabel('Number of observations', 'Interpreter', 'latex', 'Fontsize', 14)
ylabel('P($\sum{z}^l = n$)', 'Interpreter', 'latex', 'Fontsize', 14)
grid on
hold off
legend show

% Plot x_hat trajectory
figure
for i = 1:n_sensor_hyp
    plot(1:n_obs, x_hat(1:end, i), "DisplayName", num2str(b(i)));
    text(n_obs, x_hat(end, i), num2str(b(i)));
    hold on
end
yline(desired_fill_ratio, "DisplayName", "Fill Ratio")
title("$\hat{x}$",'Interpreter', 'latex', 'Fontsize', 14)
xlabel('Number of observations', 'Interpreter', 'latex', 'Fontsize', 14)
ylabel('$\hat{x}$', 'Interpreter', 'latex', 'Fontsize', 14)
grid on
hold off
legend show

% Plot chosen sensor index
figure
plot(1:n_obs, chosen_sensor_acc)
text(n_obs, chosen_sensor_acc(end), num2str(b(chosen_sensor_acc(end))));
title("Chosen sensor accuracy",...
    'Interpreter', 'latex', 'Fontsize', 14)
xlabel('Number of observations', 'Interpreter', 'latex', 'Fontsize', 14)
ylabel('Sensor index', 'Interpreter', 'latex', 'Fontsize', 14)
ylim([0,10])
grid on
hold off

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
function x_hat = estimate_x(avg_b_obsvd, b, w)
        if avg_b_obsvd <= (1 - w)
            x_hat = 0;
        elseif avg_b_obsvd >= b
            x_hat = 1;
        else
            x_hat = (avg_b_obsvd + w - 1) / (b + w - 1);
        end
end

% Compute confidence of local estimate
function alpha = compute_alpha(h, t, b, w)
    if h <= (1-w)*t
        alpha = (b+w-1)^2 * (t*w^2 - 2*(t-h)*w + (t-h)) / ...
            (w^2 * (w-1)^2);
    elseif h >= b*t
        alpha = (b+w-1)^2 * (t*b^2 - 2*h*b + h) / ...
            (b^2 * (b-1)^2);
    else
        alpha = t^3 * (b+w-1)^2 / ...
            (h * (t-h));
    end
end

% Compute social estimate and confidence
function [x_bar, beta] = compute_social_values(x_arr, alpha)
    beta = sum(alpha);
    x_bar = sum(alpha .* x_arr) / beta;
end

% Compute informed estimate
function x_next = solve_f_x(x_hat_arr, x_bar_arr, alpha_arr, beta_arr)
    x_next = (alpha_arr .* x_hat_arr + beta_arr .* x_bar_arr) ./ ...
        (alpha_arr + beta_arr);
end

% Update sensor quality(-ies if h, t, x_hat, b, and w are vectors)
function qual = evaluate_x_hat_quality(h, t, x_hat, b, w)
    if size(x_hat) ~= size(b) | size(b) ~= size(w)
        error("Sizes of the input to evaluate x_hat quality is inconsistent.");
    end
    p_observed_black = b.*x_hat + (1 - w).*(1 - x_hat);
    qual = binopdf(h, t, p_observed_black);
end