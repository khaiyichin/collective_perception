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

n_agents = 10;                  % number of agents
n_obs = 100;                       % total number of observations
obs_comms_ratio = 1;            % ratio of observations per communication round
b = 0.65;                        % sensor probability to black tile
w = b;                          % sensor probability to white tile
n_experiments = 1;                % number of experiments
desired_fill_ratio = 0.7;   % fill ratio, f (can be set to rand(1))

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
x_bar = zeros(2, n_obs, n_experiments); % social estimates
x = zeros(n_agents, n_obs, n_experiments); % final local guess
fisher_inv= zeros(n_agents, n_obs, n_experiments); % fisher inverse (variances)

% Run through all experiments
for exp_ind = 1:n_experiments

    % Generate observed tiles for agents
    tile_occurences = generate_tiles(n_agents, desired_fill_ratio, n_obs);
    black_tiles_per_row = sum(tile_occurences ~= 0, 2);
    f(:, :, exp_ind) = ones(n_agents, 1) .* black_tiles_per_row ./ n_obs; % actual fill ratio

    prev_obs = zeros(n_agents, 1); % initialize observation collection

    % Go through each observation cycle
    for obs_ind = 1:n_obs-1

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

            % Define initial guess
            if obs_ind == 1
                x0 = x(agt_ind, obs_ind, exp_ind);
            else
                x0 = x(agt_ind, obs_ind-1, exp_ind);
            end
        end

        % Solve primal function (analytical form exists)
        x(:, obs_ind, exp_ind) = ...
            solve_f_x(x_hat(:, obs_ind, exp_ind),...
                      x_bar(1, obs_ind, exp_ind),... % social estimate
                      1./fisher_inv(:, obs_ind, exp_ind),... % local fisher info
                      1./x_bar(2, obs_ind, exp_ind)); % social fisher info (rho)

        % Compute social estimate and confidences
        x_bar(:, obs_ind+1, exp_ind) = ...
            solve_g_x( x_hat(:, obs_ind, exp_ind),...
                       fisher_inv(:, obs_ind, exp_ind) );
    end
end

%% Analysis

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
function x_bar = solve_g_x(x_hat_arr, fish_inv_arr)
    x_bar(1,1) = mean(x_hat_arr);
    x_bar(2,1) = mean(fish_inv_arr);
end