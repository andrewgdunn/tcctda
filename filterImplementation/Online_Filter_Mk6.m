function [] = Online_Filter_Mk6()
clc
format short g
tic
%% FILTERING PROBLEM PARAMETERS
% Universal Parameters
std_P = 0.01;
std_Y = .05;

% Statistical Analysis Parameters
alpha_stats = 0.05;
alpha_intermittent = 0.1;
Z_alpha_stats = norminv(1-alpha_stats/2);
Z_alpha_intermittent = norminv(1-alpha_intermittent/2);

% Graphical Parameters
plot_ideal_data = 0;
stat_plot_buffer = 5; % 0 == graph everything --> can cause scaling problems

plots = 1;
plot_sensor_stats = 1;
plot_filter_stats = 1;

num_stds_accumulating = 3;

%% FILTERING PROBLEM RANDOM VARIABLE GENERATION

% Experiment Parameters
fault_type = 1;
plot_ideal_data = 1;

% Fault type == 1 (Step) Parameters
max_P = 3;
max_delP = 1;
max_delP_precision = 0.1;

% Fault type == 2 (Ramp) Parameters
max_m = 0.1;
slope_jump = 0.005; % REMOVE LATER

% Fault type == 3 (Intermittant) Parameters
max_delP_i = 1;
max_std_delP_i = 0.1;

sample_min_f = 50;
sample_max_f = 80;
sample_min_n = 60;
sample_max_n = 100;

% Universal Parameters
% P = 2*max_P*(rand - 0.5);
P = round(2*max_P*(rand - 0.5));
sensor_hz = 10;
dt = 1/sensor_hz;
t_lower = 0;
t_upper = 200;
t_inj_loc = floor((t_upper - t_lower)*sensor_hz*rand);


if fault_type == 1
    % Fault type == 1 (Step) Parameters
    % del_P = 2*max_delP*(rand-0.5);
    del_P = sign(rand-.5).*ceil(2*max_delP*abs(rand-0.5)/max_delP_precision)*max_delP_precision;
    t_inj_loc = 500
    del_P = .1
    
elseif fault_type == 2 
    % Fault type == 2 (Ramp) Parameters
    % slope_P = 2*max_m*(rand-0.5)
    slope_P = ceil(2*max_m*abs(rand-0.5)/slope_jump)*slope_jump*sign(rand-.5);
    t_inj_loc = 600
    slope_P = sign(rand-0.5)*0.002
    
elseif fault_type == 3
    % Fault type == 3 (Intermittant) Parameters
    % mu_del_P = 2*max_delP_i*(rand-0.5);
%     mu_del_P = sign(rand-.5).*ceil(2*max_delP_i*abs(rand-0.5));
    t_inj_loc = 200;
    mu_del_P = 0.3;
    std_del_P = max_std_delP_i*rand;

%     mu_f_stochastic = ceil((sample_max_f - sample_min_f)*rand)/sensor_hz;
%     mu_n_stochastic = ceil((sample_max_n - sample_min_n)*rand)/sensor_hz;
    mu_f_stochastic = 8;
    mu_n_stochastic = 16;
end


% DEPENDENT PARAMETER CALCULATION AND TIME SERIES DATA GENERATION
% Universal
t_inj = t_inj_loc/sensor_hz;
N_samples = (t_upper - t_lower)*sensor_hz;
t_hist = linspace(t_lower,t_upper,N_samples + 1);
t_hist = t_hist(1:end-1);
P_blank = ones(1,N_samples);

% Actual load history (P_hist)
% Sensor history (Y_hist)
if fault_type == 1
    % Fault type == 1 (Step)
    P_hist = [P*P_blank(1:t_inj_loc), (P+del_P)*P_blank(t_inj_loc+1:end)] + randn(1,N_samples)*std_P;
    Y_hist = P_hist + randn(1,N_samples)*std_Y;
    
elseif fault_type == 2
    % Fault type == 2 (Ramp)
    P_hist = [P*P_blank(1:t_inj_loc), (P+slope_P.*(t_hist(t_inj_loc+1:end)-t_inj)).*P_blank(t_inj_loc+1:end)] + randn(1,N_samples)*std_P;
    Y_hist = P_hist + randn(1,N_samples)*std_Y;

elseif fault_type == 3
    % Fault type == 3 (Intermittant)
    P_hist = P*P_blank;
    t_min_f = sample_min_f/sensor_hz;
    t_min_n = sample_min_n/sensor_hz;

    mu_f = mu_f_stochastic + t_min_f;
    mu_n = mu_n_stochastic + t_min_n;

    location = t_inj_loc;
    end_location = N_samples;
    while location < end_location
        del_P = mu_del_P + std_del_P*randn;
        T_f = t_min_f + round(exprnd(mu_f_stochastic,1)*sensor_hz)/sensor_hz;
        T_n = t_min_n + round(exprnd(mu_n_stochastic,1)*sensor_hz)/sensor_hz;

        loc_f = location;
        loc_n = loc_f + T_f*sensor_hz;
        if loc_n > end_location
            P_hist(loc_f+1:end) = del_P + P_hist(loc_f+1:end);
        else
            P_hist(loc_f+1:loc_n) = del_P + P_hist(loc_f+1:loc_n);
        end

        location = loc_n + T_n*sensor_hz;
        if location > end_location
            P_hist(loc_n+1:end) = P_hist(loc_n+1:end);
        else
            P_hist(loc_n+1:location) = P_hist(loc_n+1:location);
        end
    end
    P_hist = P_hist + randn(1,N_samples)*std_P;
    Y_hist = P_hist + randn(1,N_samples)*std_Y;

end

P_hist = [P_hist; zeros(1,N_samples)];
Z_hist = [Y_hist; zeros(1,N_samples)];

P_hist(2,2:end) = (P_hist(1,2:end) - P_hist(1,1:end-1))./(t_hist(1,2:end) - t_hist(1,1:end-1));
Z_hist(2,2:end) = (Z_hist(1,2:end) - Z_hist(1,1:end-1))./(t_hist(1,2:end) - t_hist(1,1:end-1));

Y_hist = Z_hist(1,:);
m_hist = Z_hist(2,:);

%% ADAPT-LITE SAMPLE DATA USAGE

% % % % % % smallest jump
% load Exp_1127_032f_pb_ADAPT-Lite
% sensor = IT281;
% t_inj_ACTUAL = 120;

% % % % % % smallest slope - SENSOR EVENT
% load Exp_1127_020f_pb_ADAPT-Lite
% sensor = IT240;
% t_inj_ACTUAL = 90;

% % % % smallest slope - COMPONENT EVENT
% load Exp_1151_pb_ADAPT-Lite
% sensor = IT267;
% t_inj_ACTUAL = 30;

% % % % % largest jump
% load Exp_1127_023f_pb_ADAPT-Lite
% sensor = IT267;
% t_inj_ACTUAL = 40;

% % % % % % largest slope
% load Exp_1127_026f_pb_ADAPT-Lite
% sensor = IT267;
% t_inj_ACTUAL = 50;

% % % % % % stuck fan jump
% load Exp_1127_041f_pb_ADAPT-Lite
% sensor = ST516;
% t_inj_ACTUAL = 170;

% % % % % % intermittent - LARGE
% load Exp_1127_017f_pb_ADAPT-Lite
% sensor = IT240;
% t_inj_ACTUAL = 35;

% % % % % % % intermittent - SMALL
% load Exp_1127_014f_pb_ADAPT-Lite
% sensor = E281;
% t_inj_ACTUAL = 35;

% indices = isfinite(sensor);
% t_hist = Time(indices)';
% Y_hist = sensor(indices)';
% N_samples = length(t_hist);
% 
% sensor_hz = round(mean(t_hist(2:end) - t_hist(1:end-1))^-1)
% t_inj_loc_ACTUAL = t_inj_ACTUAL*sensor_hz
% 
% P_hist = mean(Y_hist(1:50))*ones(1,N_samples);
% 
% m_hist = [0,(Y_hist(1,2:end) - Y_hist(1,1:end-1))./(t_hist(1,2:end) - t_hist(1,1:end-1))];

%% FILTER DATA SET INITIALIZATIONS
% KALMAN FILTER PARAMETERS
% x_k = F_k*x_k-1 + B_k*u_k + G_k*w_k
% y_k = H_k*x_k + I*v_k
% w_k ~ N(0,Q_k)
% v_k ~ N(0,R_k)
% x(k|k-1) = F_k*x(k-1|k-1) + B_k*u_k
% P(k|k-1) = F_k*P(k-1|k-1)*F_k' + Q_k 
% B = 0;
% F = [1, 0; 0, 1] || [1, dt; 0, 1]
% SS=       1      ||       2
SS_model = 1; 
H = [1, 0 ; 0 1];
Q = [1,0;0,1]*std_P^2;
R = [1,0;0,1]*std_Y^2;

% Kalman Filter :: x(t) = [X(t),M(t)]
% uo = u = 0;
X = zeros(1,N_samples);
M = zeros(1,N_samples);
M_temp = zeros(1,N_samples);
% State Space model switching for different slope calculation methods
if SS_model == 1
    M_cluster_size = 2;
elseif SS_model == 2
    M_cluster_size = 1;
end
slope_ignore_buffer = M_cluster_size;

% KF Zero'th Point/Initial Condition Values
L = 200;
x0_0 = zeros(2,1);
P0_0 = eye(2)*L;

%% STATISTICAL DATA STORAGE ARRAY INITIALIZATION
% % % Kalman Filtered State Fault Interval Statistical Data
% Filtered Sensor State Data
start_X = zeros(1,N_samples);
count_X = zeros(1,N_samples);
cumsum_X = zeros(1,N_samples);

mu_UCI_X = zeros(1,N_samples);
mu_X = zeros(1,N_samples);
mu_LCI_X = zeros(1,N_samples);
std_X = zeros(1,N_samples);

% Sensor State Periodic Correlation Coefficient
R_X = zeros(1,N_samples);

% Filtered Sensor Slope/Rate of Change Data
start_M = zeros(1,N_samples);
count_M = zeros(1,N_samples);
cumsum_M = zeros(1,N_samples);

mu_UCI_M = zeros(1,N_samples);
mu_M = zeros(1,N_samples);
mu_LCI_M = zeros(1,N_samples);
std_M = zeros(1,N_samples);

% % % Raw sensor feed statistical data (for comparison/plotting purposes)
% Raw Sensor State Data
start_Y = zeros(1,N_samples);
count_Y = zeros(1,N_samples);
cumsum_Y = zeros(1,N_samples);

mu_UCI_Y = zeros(1,N_samples);
mu_Y = zeros(1,N_samples);
mu_LCI_Y = zeros(1,N_samples);
std_Y = zeros(1,N_samples);

% Raw Sensor Slope/Rate of Change Data
start_m = zeros(1,N_samples);
count_m = zeros(1,N_samples);
cumsum_m = zeros(1,N_samples);

mu_UCI_m = zeros(1,N_samples);
mu_m = zeros(1,N_samples);
mu_LCI_m = zeros(1,N_samples);
std_m = zeros(1,N_samples);



%% ANALYSIS LOOP SETTINGS AND DATA STORAGE INITIALIZATION
running = 1;
simulation_end = N_samples;
data_start = 1;
interval_restart = 0;

% Filter values
event_count = 0;

% % % Signal_parameters initialization
% Store the accumulated statistical data for each interval in the sensor
signal_parameters = zeros(8,1);
%  1 = start index
%  2 = end index
%  3 = mu_X lower bound CI
%  4 = mu_X
%  5 = mu_X upper bound CI
%  6 = mu_M lower bound CI
%  7 = mu_M
%  8 = mu_M upper bound CI

% % % Fault_parameters initialization
% Store the calculated data characterizing any of the three fault types that occur (0 == nominal)
fault_parameters = zeros(17,1); 
%  1 = fault type
%  2 = fault injection index
%  3 = del_P lower bound CI
%  4 = del_P
%  5 = del_P upper bound CI
%  6 = m lower bound CI
%  7 = m
%  8 = m upper bound CI
%  9 = mu_n lower bound CI
% 10 = mu_n
% 11 = mu_n upper bound CI
% 12 = mu_f lower bound CI
% 13 = mu_f
% 14 = mu_f upper bound CI
% 15 = mu_ratio lower bound CI
% 16 = mu_ratio
% 17 = mu_ratio upper bound CI

%% WAIT FLAGS AND PARAMETERS
% Wait parameters for choosing step/slope error
wait_verify_count = 0;
wait_flag = 0;
wait_samples = 50;
if sensor_hz == 2
    wait_samples = 50;
elseif sensor_hz == 5
    wait_samples = 80;
elseif sensor_hz == 10
    wait_samples = 100;
end
decision_flag = 0;

% Wait paramters for considering intermittent after step fault decision
wait_intermittent_flag = 0;
wait_intermittent_count = 0;
wait_intermittent = 80;


%% RULE FLAGS AND CLAIM SITE INITIALIZATION
R12_claim_count = 0;
R12_claim_flag = 0;
R12_t_inj_loc = 0;
R12_data_restart = data_start;
R12_claim_sites = [];

R34_claim_count = 0;
R34_claim_flag = 0;
R34_t_inj_loc = 0;
R34_data_restart = data_start;
R34_claim_sites = [];

R56_claim_count = 0;
R56_claim_flag = 0;
R56_claim_sites = [];

%% RULE PARAMETERS
% Rule 1 - N points beyond 3-sigma historical value
if sensor_hz == 2
    R1_history = 3;
    R1_points = 3;
    R1_num_stds = 4;
elseif sensor_hz == 5
    R1_history = 10;
    R1_points = 10;
    R1_num_stds = 3;
elseif sensor_hz == 10
    R1_history = 10;
    R1_points = 10;
    R1_num_stds = 3;
end

% Rule 2 - Sliding Average Jump Location Localization
R2_interval_length = 20;
R2_mean_buffer = 5;
if sensor_hz == 2
    R2_interval_length = 5;
    R2_mean_buffer = 1;
elseif sensor_hz == 5
    R2_interval_length = 20;
    R2_mean_buffer = 5;
elseif sensor_hz == 10
    R2_interval_length = 20;
    R2_mean_buffer = 5;
end

% Rule 3 - Jump/Slope Detection Verification
R3_history = 10;
R3_points = 10;
R3_num_stds = 1;

% Rule 4 - Sliding Average Jump Location Localization
R4_interval_length = 20;
R4_mean_buffer = 5;

% Rule 5 - Periodic R^2 Calculation
R5_slope_inference_mode = 0;
R5_R_threshold = 0.6;
if sensor_hz == 2
    R5_samples = 60;
elseif sensor_hz == 5
    R5_samples = 120;
elseif sensor_hz == 10
    R5_samples = 180;
end

% Rule 6 - Accumulating R^2 Calculation and Injection Traceback
R6_R_restart = [R5_R_threshold:0.01:.99];
R6_restart_count = 0;
R6_num_restarts = length(R6_R_restart);

% Decision Tree Parameters
if sensor_hz == 2
    X_vs_R_claim_gap_min = 40;
elseif sensor_hz == 5
    X_vs_R_claim_gap_min = 60;
elseif sensor_hz == 10
    X_vs_R_claim_gap_min = 80;
end



%% ANALYSIS LOOP EXECUTION
while running == 1;
    %% KALMAN FILTER INITIAL STEP
    % Calculate the F matrix of the KF for each step depending on state space model selection
    if SS_model == 1
        F_tt = [1, 0; 0, 1];
    elseif SS_model == 2
        if data_start == 1
            del_t = t_hist(data_start);
        else
            del_t = t_hist(data_start) - t_hist(data_start-1);
        end
        F_tt = [1, del_t; 0, 1];
    end
    
    % Collect sensor data for KF step
    Z_in = [Y_hist(data_start);m_hist(data_start)];
    % KF update step from initial/restart 'unknown' point
    [xk_k,pk_k] = kalman_filter_step(x0_0,P0_0,F_tt,H,Q,R,Z_in);
    % Store KF updated state values
    X(data_start) = xk_k(1);
	M(data_start) = xk_k(2);
    
    % Store the final time of each interval upon restarting the next
    if data_start > 1
        signal_parameters(2,event_count) = data_start-1;
    end

    %% INITIALIZE FAULT PARAMETERS AFTER DECISION IS MADE
    if fault_parameters(1) == 1
        % NONE, ALL ONLINE
    elseif fault_parameters(1) == 2
        % NONE, ALL ONLINE
    elseif fault_parameters(1) == 3
        % Calculate the characteristic parameters corresponding to an intermittent fault
        [fault_parameters] = intermittent_statistics_calculation(event_count,signal_parameters,fault_parameters,Z_alpha_intermittent);
    end

    
    for tt = data_start + 1:simulation_end
        %% KALMAN FILTER UPDATE STEP
        % Store the previous "new" state values as "prior" state values in KF
        xk1_k1 = xk_k;
        pk1_k1 = pk_k;

        % Calculate the F matrix of the KF for each step depending on state space model selection
        if SS_model == 1
            F_tt = [1, 0; 0, 1];
        elseif SS_model == 2
            del_t = t_hist(tt) - t_hist(tt-1);
            F_tt = [1, del_t; 0, 1];
        end
        
        % Collect sensor data for KF step
        Z_in = [Y_hist(tt);m_hist(tt)];
        % KF update step from initial/restart 'unknown' point
        [xk_k,pk_k] = kalman_filter_step(xk1_k1,pk1_k1,F_tt,H,Q,R,Z_in);
        % Store KF updated state values
        X(tt) = xk_k(1);
        
        % Calculate the slope value M as apporpriate to the state space model used
        if SS_model == 1
            % Single state --> must use linear regression to calculate the rate of change
            % can take in clusters instead of two points to help reduce noise, at the expense of history/speed
            if tt > M_cluster_size
                [M(tt),EMPTY] = linear_regression(t_hist(tt-M_cluster_size+1:tt),X(tt-M_cluster_size+1:tt));
            elseif tt > M_cluster_size/2
                [M(tt),EMPTY] = linear_regression(t_hist(1:tt),X(1:tt));
            end
        elseif SS_model == 2
            % Dual state --> can use M directly
            % take cluster averages to help filter out additional noise
            M_temp(tt) = xk_k(2);
            if tt > M_cluster_size
                M(tt) = mean(M_temp(tt-M_cluster_size+1:tt));
            end
        end
        
        %% INTERVAL STATISTICAL DATA CALCULATION
        if interval_restart == 1 || tt == 2 || tt == 1
            % Calculate the data for the first point of any interval, as it gets skipped in the loops otherwise
            kk = data_start;
            interval_restart = 1;
            % Filtered State X Accumulated Statistics for Interval:
            [start_X(kk),count_X(kk),cumsum_X(kk),mu_X(kk),mu_LCI_X(kk),mu_UCI_X(kk),std_X(kk)] = ...
                Online_statistics(interval_restart,0,kk,0,0,0,X,Z_alpha_stats);
            % Filtered State M Accumulated Statistics for Interval:
            [start_M(kk),count_M(kk),cumsum_M(kk),mu_M(kk),mu_LCI_M(kk),mu_UCI_M(kk),std_M(kk)] = ...
                Online_statistics(interval_restart,slope_ignore_buffer,kk,0,0,0,M,Z_alpha_stats);
            % Raw State Y Accumulated Statistics for Interval:
            [start_Y(kk),count_Y(kk),cumsum_Y(kk),mu_Y(kk),mu_LCI_Y(kk),mu_UCI_Y(kk),std_Y(kk)] = ...
                Online_statistics(interval_restart,0,kk,0,0,0,Y_hist,Z_alpha_stats);
            % Raw State m Accumulated Statistics for Interval:
            [start_m(kk),count_m(kk),cumsum_m(kk),mu_m(kk),mu_LCI_m(kk),mu_UCI_m(kk),std_m(kk)] = ...
                Online_statistics(interval_restart,slope_ignore_buffer,kk,0,0,0,m_hist,Z_alpha_stats);
            
            % Exit initial point calculation and bypass until a new interval restart occurs
            interval_restart = 0;
        end
        
        % Filtered State X Accumulated Statistics for Interval:
        [start_X(tt),count_X(tt),cumsum_X(tt),mu_X(tt),mu_LCI_X(tt),mu_UCI_X(tt),std_X(tt)] = ...
            Online_statistics(interval_restart,0,tt,start_X(tt-1),count_X(tt-1),cumsum_X(tt-1),X,Z_alpha_stats);
        % Filtered State M Accumulated Statistics for Interval:    
        [start_M(tt),count_M(tt),cumsum_M(tt),mu_M(tt),mu_LCI_M(tt),mu_UCI_M(tt),std_M(tt)] = ...
            Online_statistics(interval_restart,slope_ignore_buffer,tt,start_M(tt-1),count_M(tt-1),cumsum_M(tt-1),M,Z_alpha_stats);
        % Raw State Y Accumulated Statistics for Interval:
        [start_Y(tt),count_Y(tt),cumsum_Y(tt),mu_Y(tt),mu_LCI_Y(tt),mu_UCI_Y(tt),std_Y(tt)] = ...
            Online_statistics(interval_restart,0,tt,start_Y(tt-1),count_Y(tt-1),cumsum_Y(tt-1),Y_hist,Z_alpha_stats);
        % Raw State m Accumulated Statistics for Interval:
        [start_m(tt),count_m(tt),cumsum_m(tt),mu_m(tt),mu_LCI_m(tt),mu_UCI_m(tt),std_m(tt)] = ...
            Online_statistics(interval_restart,slope_ignore_buffer,tt,start_m(tt-1),count_m(tt-1),cumsum_m(tt-1),m_hist,Z_alpha_stats);
        
        %% ONLINE SIGNAL PARAMETERS UPDATE STEP
        % Resize signal_parameters array with each new interval
        if size(signal_parameters,2) == event_count 
            % Resize for next interval
            signal_parameters = [signal_parameters,zeros(8,1)];
            % Recalculation of previous signal interval final X & M statistical values. 
            signal_parameters(2,event_count) = data_start-1;
            signal_parameters(3,event_count) = mu_LCI_X(data_start-1);
            signal_parameters(4,event_count) = mu_X(data_start-1);
            signal_parameters(5,event_count) = mu_UCI_X(data_start-1);
            signal_parameters(6,event_count) = mu_LCI_M(data_start-1);
            signal_parameters(7,event_count) = mu_M(data_start-1);
            signal_parameters(8,event_count) = mu_UCI_M(data_start-1);
        end
        % Current online estimates of the statistics of the current interval
        signal_parameters(1,event_count + 1) = data_start;
        signal_parameters(2,event_count + 1) = tt;
        signal_parameters(3,event_count + 1) = mu_LCI_X(tt);
        signal_parameters(4,event_count + 1) = mu_X(tt);
        signal_parameters(5,event_count + 1) = mu_UCI_X(tt);
        signal_parameters(6,event_count + 1) = mu_LCI_M(tt);
        signal_parameters(7,event_count + 1) = mu_M(tt);
        signal_parameters(8,event_count + 1) = mu_UCI_M(tt);
        
        %% ONLINE FAULT PARAMETERS UPDATE STEP
        if fault_parameters(1) == 1
            % Calculate the online fault statistic estimations for a step fault
            if signal_parameters(3,1) > signal_parameters(5,2)
                % Ordered for proper sign for a jump with negative magnitude
                del_P_min = signal_parameters(5,2) - signal_parameters(3,1);
                del_P_avg = signal_parameters(4,2) - signal_parameters(4,1);
                del_P_max =  signal_parameters(3,2) - signal_parameters(5,1);
            elseif signal_parameters(5,1) < signal_parameters(3,2)
                % Ordered for proper sign for a jump with positive magnitude
                del_P_min = signal_parameters(3,2) - signal_parameters(5,1);
                del_P_avg = signal_parameters(4,2) - signal_parameters(4,1);
                del_P_max =  signal_parameters(5,2) - signal_parameters(3,1);
            end
            % Store parameters
            fault_parameters(3) = del_P_min;
            fault_parameters(4) = del_P_avg;
            fault_parameters(5) = del_P_max;
        elseif fault_parameters(1) == 2
            % Calculate the online fault statistic estimations for a ramp fault
            t_slope = t_hist(signal_parameters(1,2):tt);
            x_slope = X(signal_parameters(1,2):tt);
            [M_update,EMPTY] = linear_regression(t_slope,x_slope);
            % Store parameters
            fault_parameters(6) = mu_LCI_M(tt);
            fault_parameters(7) = M_update;
            fault_parameters(8) = mu_UCI_M(tt);
        elseif fault_parameters(1) == 3
            % NONE, RECALCULATED AFTER EACH INTERVAL
        end
        
        %% FILTER FAULT DETECTION RULES
        % % % RULE 1 - N-POINT BACKWARD SIGMA VALUE THRESHOLD CROSSING
        % Lower limit to avoid dimension errors and repeated detection of the same event
        R1_lower_limit = R2_interval_length + R12_data_restart;
        if tt > R1_lower_limit
            
            % Pull out past statistical data for comparison against current
            MU = mu_X(tt-R1_history);
            STD = std_X(tt-R1_history);
            % Rule analysis of sample interval
            [jump_flag] = KF_RULE_number_points_above_std(tt,R1_history,MU,STD,R1_points,R1_num_stds,X);
            
            % If rule is triggered, lay a claim to the location and store the defining parameters
            if jump_flag == 1
                % Begin wait period before deciding between step/ramp errors
                if fault_parameters(1) == 0
                    wait_flag = 1;
                end
                % Rule lays claim to having found a fault
                R12_claim_flag = 1;
                R12_claim_count = R12_claim_count + 1;
                
                % % % RULE 2 - SLIDING AVERAGE JUMP LOCATION VERIFICATION
                % Generate the interval to do event localization
                t_interval = [tt-R2_interval_length:1:tt];
                % Determine precise location estimate of fault/event
                [R12_t_inj_loc] = T_INJ_REFINE_sliding_average(R2_mean_buffer,t_interval,Y_hist);
                
                % Store claimed t_inj_loc values as found and reset lower limit value for rules
                R12_claim_sites = [R12_claim_sites,R12_t_inj_loc];
                R12_data_restart = R12_t_inj_loc + 1;

                % Intermittant fault -- keep detecting every event that occurs once determined and reset appropriately
                if fault_parameters(1) == 3
                    data_start = R12_t_inj_loc + 1;
                    event_count = event_count + 1;
                    interval_restart = 1;
                    break
                end
            end
        end

        
        % % % RULE 3 - N-POINT BACKWARD SIGMA VALUE THRESHOLD CROSSING
        % Lower limit to avoid dimension errors and repeated detection of the same event
        R3_lower_limit = R4_interval_length + M_cluster_size + R34_data_restart;
        if tt > R3_lower_limit
            
            % Pull out past statistical data for comparison against current
            MU = mu_M(tt-R3_history);
            STD = std_M(tt-R3_history);
            % Rule analysis of sample interval
            [jump_flag] = KF_RULE_number_points_above_std(tt,R3_history,MU,STD,R3_points,R3_num_stds,M);
            
            % If rule is triggered, lay a claim to the location and store the defining parameters
            if jump_flag == 1
                % Begin wait period before deciding between step/ramp errors
                wait_flag = 1;
                % Rule lays claim to having found a fault
                R34_claim_flag = 1;
                R34_claim_count = R34_claim_count + 1;
                
                % % % RULE 4 - SLIDING AVERAGE JUMP LOCATION VERIFICATION
                % Generate the interval to do event localization
                t_interval = [tt-R4_interval_length:1:tt];
                % Determine precise location estimate of fault/event
                [R34_t_inj_loc] = T_INJ_REFINE_sliding_average(R4_mean_buffer,t_interval,M);

                % Store claimed t_inj_loc values as found and reset lower limit value for rules
                R34_claim_sites = [R34_claim_sites,R34_t_inj_loc];
                R34_data_restart = R34_t_inj_loc + 1;
            end
        end

        
        % % %  RULE 5 - PERIODIC R THRESHOLD CROSSING(S)
        % Forward scanning interval, runs until sufficient correlation value occurs (if ever)
        if R5_slope_inference_mode == 0
            % Lower limit to avoid dimension errors and repeated detection of the same event
            R5_lower_limit = 1 + R5_samples;
            
            % Calculate R^2 value of data interval if index values permit
            if tt > R5_lower_limit 
                R_X(tt) = R_correlation(t_hist(tt-R5_samples:tt),X(tt-R5_samples:tt)).^2;
                
                % Determine if current R^2 value high enough to begin accumulating instead of periodic analysis
                if R_X(tt) > R5_R_threshold
                    R5_slope_inference_mode = 1;
                    % Fix anchor point from which to start all future R^2 calculations
                    if tt - R5_samples >= 1
                        R56_anchor = tt - R5_samples;
                    else
                        R56_anchor = 1;
                    end
                end
            end
        % Anchored interval for thorough R^2 calculation and ramp injection calculation
        elseif R5_slope_inference_mode == 1
            % Calculate the R^2 value of data from the high correlation anchor point to current location
            R_X(tt) = R_correlation(t_hist(R56_anchor:tt),X(R56_anchor:tt)).^2;
            
            % % % RULE 6 - R TRIGGERED LINEAR REGRESSION TRACEBACK
            if R6_restart_count < R6_num_restarts
                if R_X(tt) >= R6_R_restart(R6_restart_count + 1)
                    % Begin wait period before deciding between step/ramp errors
                    wait_flag = 1;
                    % Rule lays claim to having found a fault
                    R56_claim_flag = 1;
                    R56_claim_count = R56_claim_count + 1;
                    
                    % Calculate corresponding slope/intercept of interval to guess a more accurate injection point
                    [m_R,yo_R] = linear_regression(t_hist(R56_anchor:tt),X(R56_anchor:tt));
                    x_nominal = mu_X(R56_anchor);
                    t_inj_guess_R = round((x_nominal - yo_R)/m_R);
                    R56_t_inj_loc = t_inj_guess_R*sensor_hz;

                    % Store claimed t_inj_loc values as found and reset lower limit value for rules
                    R56_claim_sites = [R56_claim_sites,R56_t_inj_loc];
                    R56_anchor = R56_t_inj_loc;
                    R6_restart_count = R6_restart_count + 1;
                end
            end
        end

        
        % Wait interval update until time to determine fault
        if wait_flag == 1
            wait_verify_count = wait_verify_count + 1;
        end
        
        % Wait interval update until time to determine fault
        if wait_intermittent_flag == 1
            wait_intermittent_count = wait_intermittent_count + 1;
        end
        
        %% FILTER FAULT IDENTIFICATION DECISION MAKER
        % Once waiting period has ended, make a decision of what fault has occurred, if any, based on the appropriate
        % statistics that can be used to discriminate between different combinations of rule fault claims
        if wait_samples == wait_verify_count && decision_flag == 0
            
            R12_claim_sites
            R34_claim_sites
            R56_claim_sites
            
            if R12_claim_flag == 1 && R34_claim_flag == 0 && R56_claim_flag == 0
                %% X-JUMP ONLY CLAIM
                % Store first inferred value (later values polluted) and reset appropriately 
                data_start = R12_claim_sites(1) + 1;
                R12_data_restart = data_start;
                event_count = event_count + 1;
                interval_restart = 1;
                fault_parameters(1) = 1;
                fault_parameters(2) = R12_claim_sites(1);
                wait_intermittent_flag = 1;
            elseif R12_claim_flag == 0 && R34_claim_flag == 1 && R56_claim_flag == 0
                %% M-JUMP ONLY CLAIM
                % Store first inferred value (later values polluted) and reset appropriately 
                data_start = R34_claim_sites(1) + 1;
                R34_data_restart = data_start;
                event_count = event_count + 1;
                interval_restart = 1;
                fault_parameters(1) = 2;
                fault_parameters(2) = R34_claim_sites(1);
            elseif R12_claim_flag == 0 && R34_claim_flag == 0 && R56_claim_flag == 1
                %% R^2 CORRELATION ONLY CLAIM
                % Scan for the first duplicated t_inj value inferred, will likely be the closest value to the true
                % injection site. If present, use as the restart point. 
                inj_loc = 0;
                for ii = 1:length(R56_claim_sites)-1
                    if R56_claim_sites(ii) == R56_claim_sites(ii+1)
                        inj_loc = ii;
                        break
                    end
                end
                
                % Store values
                if inj_loc == 0
                    % If no t_inj claim duplication occurs, simply use the final value calculated
                    data_start = R56_claim_sites(end) + 1;
                    fault_parameter(2) = R56_claim_sites(end);
                else
                    % Use the first duplicate claim value
                    data_start = R56_claim_sites(inj_loc) + 1;
                    fault_parameters(2) = R56_claim_sites(inj_loc);
                end
                % Restart values
                event_count = event_count + 1;
                interval_restart = 1;
                fault_parameters(1) = 2;
            elseif R12_claim_flag == 1 && R34_claim_flag == 1 && R56_claim_flag == 0
                %% X-JUMP AND M-JUMP COMPETING CLAIMS
                % R^2 values for various intervals for comparison
                R_post_X_step = R_correlation(t_hist(R12_claim_sites(1):tt),X(R12_claim_sites(1):tt))^2;
                R_post_M_step = R_correlation(t_hist(R34_claim_sites(1):tt),M(R34_claim_sites(1):tt))^2;
                interval = (tt - R12_claim_sites(1));
                R_step_gap = R_correlation(t_hist(tt-2*interval:tt),X(tt-2*interval:tt))^2;
                
                
                if R12_claim_sites(1) == R34_claim_sites(1)
                    % X & M jump locations agree, almost certain to be the injection location of a step fault
                    data_start = R12_claim_sites(1) + 1;
                    R12_data_restart = data_start;
                    event_count = event_count + 1;
                    interval_restart = 1;
                    fault_parameters(1) = 1;
                    fault_parameters(2) = R12_claim_sites(1);
                    wait_intermittent_flag = 1;
                elseif R34_claim_sites(1) < R12_claim_sites(1)
                    % For a large M-jump triggering an X-jump, M will lead X due to gradual increase in X value but
                    % sudden increase in M value
                    data_start = R34_claim_sites(1) + 1;
                    R34_data_restart = data_start;
                    event_count = event_count + 1;
                    interval_restart = 1;
                    fault_parameters(1) = 2;
                    fault_parameters(2) = R34_claim_sites(1);
                end
            elseif R12_claim_flag == 1 && R34_claim_flag == 0 && R56_claim_flag == 1
                %% X-JUMP AND R^2 CORRELATION COMPETING CLAIMS
                % R^2 values for various intervals for comparison
                R_post_X_step = R_correlation(t_hist(R12_claim_sites(1):tt),X(R12_claim_sites(1):tt))^2;
                R_post_56anchor = R_correlation(t_hist(R56_claim_sites(end):tt),X(R56_claim_sites(end):tt))^2;
                interval = (tt - R12_claim_sites(1));
                R_step_gap = R_correlation(t_hist(tt-2*interval:tt),X(tt-2*interval:tt))^2;
                % Staggar between X-jump claims and R^2 claims
                X_vs_R_gap = R12_claim_sites(1) - R56_claim_sites(end);
                
                % If X-jump claim far enough behind, indicates a slope is present (small slope causes)
                if X_vs_R_gap >= X_vs_R_claim_gap_min
                    % Scan for the first duplicated t_inj value inferred, will likely be the closest value to the true
                    % injection site. If present, use as the restart point.
                    inj_loc = 0;
                    for ii = 1:length(R56_claim_sites)-1
                        if R56_claim_sites(ii) == R56_claim_sites(ii+1)
                            inj_loc = ii;
                            break
                        end
                    end
                    if inj_loc == 0
                        % If no t_inj claim duplication occurs, simply use the final value calculated
                        data_start = R56_claim_sites(end) + 1;
                        fault_parameter(2) = R56_claim_sites(end);
                    else
                        % Use the first duplicate claim value
                        data_start = R56_claim_sites(inj_loc) + 1;
                        fault_parameters(2) = R56_claim_sites(inj_loc);
                    end
                    % Restart values
                    event_count = event_count + 1;
                    interval_restart = 1;
                    fault_parameters(1) = 2;
                elseif (R_post_X_step < R_post_56anchor || R_post_X_step < R_step_gap) &&  R_post_56anchor < R_step_gap
                    % If an X-jump occurs, will lead to false positive high R^2 value over the interval higher than all
                    % others (when the two coincide roughly, as for a step but NOT a small slope)
                    data_start = R12_claim_sites(1) + 1;
                    R12_data_restart = data_start;
                    event_count = event_count + 1;
                    interval_restart = 1;
                    fault_parameters(1) = 1;
                    fault_parameters(2) = R12_claim_sites(1);
                    R12_claim_sites = [];
                    R12_claim_count = 0;
                    wait_intermittent_flag = 1;
                else
                    % R_56 R^2 value prevailing by default, indicating a true slope exists after the injection location 
                    % and thus the fault is a ramp
                    
                    % Scan for the first duplicated t_inj value inferred, will likely be the closest value to the true
                    % injection site. If present, use as the restart point.
                    inj_loc = 0;
                    for ii = 1:length(R56_claim_sites)-1
                        if R56_claim_sites(ii) == R56_claim_sites(ii+1)
                            inj_loc = ii;
                            break
                        end
                    end
                    if inj_loc == 0
                        % If no t_inj claim duplication occurs, simply use the final value calculated
                        data_start = R56_claim_sites(end) + 1;
                        fault_parameter(2) = R56_claim_sites(end);
                    else
                        % Use the first duplicate claim value
                        data_start = R56_claim_sites(inj_loc) + 1;
                        fault_parameters(2) = R56_claim_sites(inj_loc);
                    end
                    % Restart values
                    event_count = event_count + 1;
                    interval_restart = 1;
                    fault_parameters(1) = 2;
                end
            elseif R12_claim_flag == 0 && R34_claim_flag == 1 && R56_claim_flag == 1
                %% M-JUMP AND R^2 CORRELATION COMPETING CLAIMS
                % Indicates a ramp fault exclusively, but need to determine best guess
                % If M-jump and R^2 claims agree, that is the injection point
                t_inj_consensus = 0;
                for ii = 1:R56_claim_count
                    for jj = 1:R34_claim_count
                        if R34_claim_sites(jj) == R56_claim_sites(ii)
                            t_inj_consensus = R34_claim_sites(jj);
                        end
                    end
                end
                
                % Use consensus point as the injection location and restart
                if t_inj_consensus ~= 0
                    data_start = t_inj_consensus + 1;
                    R34_data_restart = data_start;
                    event_count = event_count + 1;
                    interval_restart = 1;
                    fault_parameters(1) = 2;
                    fault_parameters(2) = R34_claim_sites(1);
                else
                    % If no M-jump and R^2 consensus exists, us R^2 guesses in same fashion as before
                    % Scan for the first duplicated t_inj value inferred, will likely be the closest value to the true
                    % injection site. If present, use as the restart point.
                    inj_loc = 0;
                    for ii = 1:length(R56_claim_sites)-1
                        if R56_claim_sites(ii) == R56_claim_sites(ii+1)
                            inj_loc = ii;
                            break
                        end
                    end
                    if inj_loc == 0
                        % If no t_inj claim duplication occurs, simply use the final value calculated
                        data_start = R56_claim_sites(end) + 1;
                        fault_parameter(2) = R56_claim_sites(end);
                    else
                        % Use the first duplicate claim value
                        data_start = R56_claim_sites(inj_loc) + 1;
                        fault_parameters(2) = R56_claim_sites(inj_loc);
                    end
                    % Restart values
                    event_count = event_count + 1;
                    interval_restart = 1;
                    fault_parameters(1) = 2;
                end
            elseif R12_claim_flag == 1 && R34_claim_flag == 1 && R56_claim_flag == 1
                %% X-JUMP, M-JUMP AND R^2 CORRELATION COMPETING CLAIMS
                % R^2 values for various intervals for comparison
                R_post_X_step = R_correlation(t_hist(R12_claim_sites(1):tt),X(R12_claim_sites(1):tt))^2;
                R_post_56anchor = R_correlation(t_hist(R56_claim_sites(end):tt),X(R56_claim_sites(end):tt))^2;
                interval = (tt - R12_claim_sites(1));
                R_step_gap = R_correlation(t_hist(tt-2*interval:tt),X(tt-2*interval:tt))^2;
                % Staggar between X-jump claims and R^2 claims
                X_vs_R_gap = R12_claim_sites(1) - R56_claim_sites(end);

                
                if R12_claim_sites(1) == R34_claim_sites(1)
                    % X & M jump locations agree, almost certain to be the injection location of a step fault
                    data_start = R12_claim_sites(1) + 1;
                    R12_data_restart = data_start;
                    event_count = event_count + 1;
                    interval_restart = 1;
                    fault_parameters(1) = 1;
                    fault_parameters(2) = R12_claim_sites(1);
                    wait_intermittent_flag = 1;
                elseif X_vs_R_gap >= X_vs_R_claim_gap_min
                    % If X-jump claim far enough behind, indicates a slope is present (small slope causes)
                    % Scan for the first duplicated t_inj value inferred, will likely be the closest value to the true
                    % injection site. If present, use as the restart point.
                    inj_loc = 0;
                    for ii = 1:length(R56_claim_sites)-1
                        if R56_claim_sites(ii) == R56_claim_sites(ii+1)
                            inj_loc = ii;
                            break
                        end
                    end
                    if inj_loc == 0
                        % If no t_inj claim duplication occurs, simply use the final value calculated
                        data_start = R56_claim_sites(end) + 1;
                        fault_parameter(2) = R56_claim_sites(end);
                    else
                        % Use the first duplicate claim value
                        data_start = R56_claim_sites(inj_loc) + 1;
                        fault_parameters(2) = R56_claim_sites(inj_loc);
                    end
                    % Restart values
                    event_count = event_count + 1;
                    interval_restart = 1;
                    fault_parameters(1) = 2;
                % If R^2 post injection is less than across the event or the accumulating R^2, indicates an X-jump fault
                elseif (R_post_X_step < R_post_56anchor || R_post_X_step < R_step_gap) &&  R_post_56anchor < R_step_gap
                    data_start = R12_claim_sites(1) + 1;
                    R12_data_restart = data_start;
                    event_count = event_count + 1;
                    interval_restart = 1;
                    fault_parameters(1) = 1;
                    fault_parameters(2) = R12_claim_sites(1);
                    R12_claim_sites = [];
                    R12_claim_count = 0;
                    wait_intermittent_flag = 1;
                else
                    % If no M-jump and R^2 consensus exists, us R^2 guesses in same fashion as before
                    % Scan for the first duplicated t_inj value inferred, will likely be the closest value to the true
                    % injection site. If present, use as the restart point.
                    t_inj_consensus = 0;
                    for ii = 1:R56_claim_count
                        for jj = 1:R34_claim_count
                            if R34_claim_sites(jj) == R56_claim_sites(ii)
                                t_inj_consensus = R34_claim_sites(jj);
                            end
                        end
                    end
                    % Use consensus point as the injection location and restart
                    if t_inj_consensus ~= 0
                        data_start = t_inj_consensus + 1;
                        R34_data_restart = data_start;
                        event_count = event_count + 1;
                        interval_restart = 1;
                        fault_parameters(1) = 2;
                        fault_parameters(2) = R34_claim_sites(1);
                    else
                        % If no M-jump and R^2 consensus exists, us R^2 guesses in same fashion as before
                        % Scan for the first duplicated t_inj value inferred, will likely be the closest value to the true
                        % injection site. If present, use as the restart point.
                        inj_loc = 0;
                        for ii = 1:length(R56_claim_sites)-1
                            if R56_claim_sites(ii) == R56_claim_sites(ii+1)
                                inj_loc = ii;
                                break
                            end
                        end
                        if inj_loc == 0
                            % If no t_inj claim duplication occurs, simply use the final value calculated
                            data_start = R56_claim_sites(end) + 1;
                            fault_parameter(2) = R56_claim_sites(end);
                        else
                            % Use the first duplicate claim value
                            data_start = R56_claim_sites(inj_loc) + 1;
                            fault_parameters(2) = R56_claim_sites(inj_loc);
                        end
                        % Restart values
                        event_count = event_count + 1;
                        interval_restart = 1;
                        fault_parameters(1) = 2;
                    end
                end
            end
            % Terminate the decision loop and restart/resume analysis process having identified the fault type
            break
        end
        
        %% DECIDE INTERMITTANT VS STEP FAULT
        % If a step fault was inferred, analyze after a similar wait period to see if fault is intermittent
        if wait_intermittent == wait_intermittent_count && fault_parameters(1) ~= 3
            % If new X-jump claims have occurred, fault is intermittent and analysis switches accordingly
            if R12_claim_count > 1
                data_start = 1;
                R12_data_restart = data_start;
                interval_restart = 0;
                event_count = 0;
                fault_parameters(1) = 3;
                break
            else
                % return step function error and values
            end
        end
        
    end
    %% ANALYSIS LOOP TERMINATION
    % Once the final time/raw state points are analyzed, do final calculations of sensor data
    if tt == simulation_end
        signal_parameters(2,end) = tt;
        if fault_parameters(1) == 3
            % Include the last two intervals of an intermittent fault in the fault parameter data
            [fault_parameters] = intermittent_statistics_calculation(event_count+2,signal_parameters,fault_parameters,Z_alpha_intermittent);
        end
        % Terminate loop execution
        running = 0;
    end
end


%% GRAPHICAL ANALYSIS


run_time_total = toc;
run_time_total


event_count

signal_parameters

fault_parameters

% KALMAN FILTER
if plots == 1
    figure
    if plot_ideal_data == 0
        plot(t_hist,Y_hist,'r--')
    elseif plot_ideal_data == 1
        plot(t_hist,P_hist(1,:),'k-',t_hist,Y_hist,'r--')
    end
    grid on
    title('Sensor State X(t) vs. Time -- ACCUMULATING STATS')
    hold on
    plot(t_hist,X,'g-',t_hist,X,'gx')
    % ACCUMULATED STATISTICAL DATA
    if plot_sensor_stats == 1
        for ii = 1:event_count + 1
            bounds = signal_parameters(1:2,ii);
            interval = bounds(1):1:bounds(2);
            plot_fault_interval_stat_data(stat_plot_buffer,'b', num_stds_accumulating,  t_hist(interval), mu_Y(interval), mu_LCI_Y(interval), mu_UCI_Y(interval), std_Y(interval))
        end
    end
    if plot_filter_stats == 1
        for ii = 1:event_count + 1
            bounds = signal_parameters(1:2,ii);
            interval = bounds(1):1:bounds(2);
            plot_fault_interval_stat_data(stat_plot_buffer,'m', num_stds_accumulating,  t_hist(interval), mu_X(interval), mu_LCI_X(interval), mu_UCI_X(interval), std_X(interval))
        end
    end

    
    figure
    if plot_ideal_data == 0
        plot(t_hist,m_hist,'r--')
    elseif plot_ideal_data == 1
        plot(t_hist,P_hist(2,:),'k-',t_hist,m_hist,'r--')
    end
    grid on
    hold on
    plot(t_hist,M,'g-',t_hist,M,'gx')
    title('Sensor State M(t) vs. Time -- ACCUMULATING STATS')
    hold on
    % ACCUMULATED STATISTICAL DATA
    if plot_sensor_stats == 1
        for ii = 1:event_count + 1
            bounds = signal_parameters(1:2,ii);
            interval = bounds(1):1:bounds(2);
            plot_fault_interval_stat_data(stat_plot_buffer,'b', num_stds_accumulating,  t_hist(interval), mu_m(interval), mu_LCI_m(interval), mu_UCI_m(interval), std_m(interval))
        end
    end
    if plot_filter_stats == 1
        for ii = 1:event_count + 1
            bounds = signal_parameters(1:2,ii);
            interval = bounds(1):1:bounds(2);
            plot_fault_interval_stat_data(stat_plot_buffer,'m', num_stds_accumulating,  t_hist(interval), mu_M(interval), mu_LCI_M(interval), mu_UCI_M(interval), std_M(interval))
        end
    end 
    
    figure
    plot(t_hist,R_X,'k-')
    grid on
end


