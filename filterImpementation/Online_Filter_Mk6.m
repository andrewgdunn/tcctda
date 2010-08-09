function [] = Online_Filter_Mk6()
clc
format short g
tic
%% FILTERING PROBLEM PARAMETERS

% Experiment Parameters
fault_type = 2;

% Universal Parameters
std_P = 0.01;
std_Y = .05;

% Statistical Analysis Parameters
alpha = 0.05;
Z_alpha = norminv(1-alpha/2);

% Event Inference Parameters
% identified_fault = 0; % 0 = nominal, 1 = jump/intermittant, 2 = ramp
data_ignore = 0;

% Graphical Parameters
stat_plot_buffer = 5; % 0 == graph everything

plots = 1;

plot_sensor_stats = 1;
plot_filter_stats = 1;

num_stds_accumulating = 3;

%%
% % % % % %% FILTERING PROBLEM RANDOM VARIABLE GENERATION
% % % % % % CURRENTLY DISCRETE JUMP INTERVALS FOR TESTING -- SWITCH TO CONTINUOUS LATER
% % % % % 
% % % % % % Universal Parameters
% % % % % t_inj_loc = floor((t_upper - t_lower)*sensor_hz*rand);
% % % % % % P = 2*max_P*(rand - 0.5);
% % % % % P = round(2*max_P*(rand - 0.5));
% % % % % sensor_hz = 10;
% % % % % dt = 1/sensor_hz;
% % % % % t_lower = 0;
% % % % % t_upper = 200;
% % % % % % Fault type == 1 (Step) Parameters
% % % % % max_P = 5;
% % % % % max_delP = 1;
% % % % % max_delP_precision = 0.1;
% % % % % 
% % % % % % Fault type == 2 (Ramp) Parameters
% % % % % max_m = 0.1;
% % % % % slope_jump = 0.001; % REMOVE LATER
% % % % % 
% % % % % % Fault type == 3 (Intermittant) Parameters
% % % % % max_delP_i = 1;
% % % % % max_std_delP_i = 0.05;
% % % % % 
% % % % % sample_min_f = 15;
% % % % % sample_max_f = 30;
% % % % % sample_min_n = 20;
% % % % % sample_max_n = 50;
% % % % % 
% % % % % if fault_type == 1
% % % % %     % Fault type == 1 (Step) Parameters
% % % % %     % del_P = 2*max_delP*(rand-0.5);
% % % % %     del_P = sign(rand-.5).*ceil(2*max_delP*abs(rand-0.5)/max_delP_precision)*max_delP_precision;
% % % % %     t_inj_loc = 600
% % % % %     del_P = .1
% % % % %     
% % % % % elseif fault_type == 2 
% % % % %     % Fault type == 2 (Ramp) Parameters
% % % % %     % slope_P = 2*max_m*(rand-0.5)
% % % % %     slope_P = ceil(2*max_m*abs(rand-0.5)/slope_jump)*slope_jump*sign(rand-.5);
% % % % %     t_inj_loc = 800
% % % % %     slope_P = sign(rand-0.5)*0.005
% % % % %     
% % % % % elseif fault_type == 3
% % % % %     % Fault type == 3 (Intermittant) Parameters
% % % % %     % mu_del_P = 2*max_delP_i*(rand-0.5);
% % % % %     mu_del_P = sign(rand-.5).*ceil(2*max_delP_i*abs(rand-0.5));
% % % % %     std_del_P = max_std_delP_i*rand;
% % % % % 
% % % % %     mu_f_stochastic = ceil((sample_max_f - sample_min_f)*rand)/sensor_hz;
% % % % %     mu_n_stochastic = ceil((sample_max_n - sample_min_n)*rand)/sensor_hz;
% % % % % end
% % % % % 
% % % % % 
% % % % % %% DEPENDENT PARAMETER CALCULATION AND TIME SERIES DATA GENERATION
% % % % % % Universal
% % % % % t_inj = t_inj_loc/sensor_hz;
% % % % % N_samples = (t_upper - t_lower)*sensor_hz;
% % % % % t_hist = linspace(t_lower,t_upper,N_samples + 1);
% % % % % t_hist = t_hist(1:end-1);
% % % % % P_blank = ones(1,N_samples);
% % % % % 
% % % % % % Actual load history (P_hist)
% % % % % % Sensor history (Y_hist)
% % % % % if fault_type == 1
% % % % %     % Fault type == 1 (Step)
% % % % %     P_hist = [P*P_blank(1:t_inj_loc), (P+del_P)*P_blank(t_inj_loc+1:end)] + randn(1,N_samples)*std_P;
% % % % %     Y_hist = P_hist + randn(1,N_samples)*std_Y;
% % % % %     
% % % % % elseif fault_type == 2
% % % % %     % Fault type == 2 (Ramp)
% % % % %     P_hist = [P*P_blank(1:t_inj_loc), (P+slope_P.*(t_hist(t_inj_loc+1:end)-t_inj)).*P_blank(t_inj_loc+1:end)] + randn(1,N_samples)*std_P;
% % % % %     Y_hist = P_hist + randn(1,N_samples)*std_Y;
% % % % % 
% % % % % elseif fault_type == 3
% % % % %     % Fault type == 3 (Intermittant)
% % % % %     P_hist = P*P_blank;
% % % % %     t_min_f = sample_min_f/sensor_hz;
% % % % %     t_min_n = sample_min_n/sensor_hz;
% % % % % 
% % % % %     mu_f = mu_f_stochastic + t_min_f;
% % % % %     mu_n = mu_n_stochastic + t_min_n;
% % % % % 
% % % % %     location = t_inj_loc;
% % % % %     end_location = N_samples;
% % % % %     while location < end_location
% % % % %         del_P = mu_del_P + std_del_P*randn;
% % % % %         T_f = t_min_f + round(exprnd(mu_f_stochastic,1)*sensor_hz)/sensor_hz;
% % % % %         T_n = t_min_n + round(exprnd(mu_n_stochastic,1)*sensor_hz)/sensor_hz;
% % % % % 
% % % % %         loc_f = location;
% % % % %         loc_n = loc_f + T_f*sensor_hz;
% % % % %         if loc_n > end_location
% % % % %             P_hist(loc_f+1:end) = del_P + P_hist(loc_f+1:end);
% % % % %         else
% % % % %             P_hist(loc_f+1:loc_n) = del_P + P_hist(loc_f+1:loc_n);
% % % % %         end
% % % % % 
% % % % %         location = loc_n + T_n*sensor_hz;
% % % % %         if location > end_location
% % % % %             P_hist(loc_n+1:end) = P_hist(loc_n+1:end);
% % % % %         else
% % % % %             P_hist(loc_n+1:location) = P_hist(loc_n+1:location);
% % % % %         end
% % % % %     end
% % % % %     P_hist = P_hist + randn(1,N_samples)*std_P;
% % % % %     Y_hist = P_hist + randn(1,N_samples)*std_Y;
% % % % % 
% % % % % end
% % % % % 
% % % % % P_hist = [P_hist; zeros(1,N_samples)];
% % % % % Z_hist = [Y_hist; zeros(1,N_samples)];
% % % % % 
% % % % % P_hist(2,2:end) = (P_hist(1,2:end) - P_hist(1,1:end-1))./(t_hist(1,2:end) - t_hist(1,1:end-1));
% % % % % Z_hist(2,2:end) = (Z_hist(1,2:end) - Z_hist(1,1:end-1))./(t_hist(1,2:end) - t_hist(1,1:end-1));
% % % % % 
% % % % % Y_hist = Z_hist(1,:);
% % % % % m_hist = Z_hist(2,:);

%% ADAPT-LITE SAMPLE DATA USAGE

% % % % % % smallest jump
% load Exp_1127_032f_pb_ADAPT-Lite
% sensor = IT281;
% t_inj_ACTUAL = 120;

% % % % % % smallest slope - SENSOR EVENT
% load Exp_1127_020f_pb_ADAPT-Lite
% sensor = IT240;
% t_inj_ACTUAL = 90;

% % % % % % % smallest slope - COMPONENT EVENT
% load Exp_1151_pb_ADAPT-Lite
% sensor = IT267;
% t_inj_ACTUAL = 30;

% % % % % % % largest jump
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

% % % % % intermittant
load Exp_1127_017f_pb_ADAPT-Lite
sensor = IT240;
t_inj_ACTUAL = 35;

indices = isfinite(sensor);
t_hist = Time(indices)';
Y_hist = sensor(indices)';
N_samples = length(t_hist);

sensor_hz = round(mean(t_hist(2:end) - t_hist(1:end-1))^-1)
t_inj_loc_ACTUAL = t_inj_ACTUAL*sensor_hz

P_hist = mean(Y_hist(1:50))*ones(1,N_samples);

m_hist = [0,(Y_hist(1,2:end) - Y_hist(1,1:end-1))./(t_hist(1,2:end) - t_hist(1,1:end-1))];



%% FILTER DATA SET INITIALIZATIONS

% Sensor Parameterization Storage
inferred_parameters = [];
event_count = 0;

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

if SS_model == 1
    M_cluster_size = 2;
elseif SS_model == 2
    M_cluster_size = 1;
end
slope_ignore_buffer = M_cluster_size;

% KF Zero'th Point Values
L = 200;
x0_0 = zeros(2,1);
P0_0 = eye(2)*L;

% Kalman Filtered State Fault Interval Statistical Data
start_X = zeros(1,N_samples);
count_X = zeros(1,N_samples);
cumsum_X = zeros(1,N_samples);

mu_UCI_X = zeros(1,N_samples);
mu_X = zeros(1,N_samples);
mu_LCI_X = zeros(1,N_samples);
std_X = zeros(1,N_samples);

R_X = zeros(1,N_samples);

start_M = zeros(1,N_samples);
count_M = zeros(1,N_samples);
cumsum_M = zeros(1,N_samples);

mu_UCI_M = zeros(1,N_samples);
mu_M = zeros(1,N_samples);
mu_LCI_M = zeros(1,N_samples);
std_M = zeros(1,N_samples);

% Raw sensor feed statistical data (for comparison/plotting purposes)
start_Y = zeros(1,N_samples);
count_Y = zeros(1,N_samples);
cumsum_Y = zeros(1,N_samples);

mu_UCI_Y = zeros(1,N_samples);
mu_Y = zeros(1,N_samples);
mu_LCI_Y = zeros(1,N_samples);
std_Y = zeros(1,N_samples);

start_m = zeros(1,N_samples);
count_m = zeros(1,N_samples);
cumsum_m = zeros(1,N_samples);

mu_UCI_m = zeros(1,N_samples);
mu_m = zeros(1,N_samples);
mu_LCI_m = zeros(1,N_samples);
std_m = zeros(1,N_samples);



%% ANALYSIS LOOP SETTINGS
running = 1;
simulation_end = N_samples;
data_start = 1;
interval_restart = 0;

% Filter values
event_count = 0;

signal_parameters = zeros(8,1);
%  1 = start index
%  2 = end index
%  3 = mu_X lower bound CI
%  4 = mu_X
%  5 = mu_X upper bound CI
%  6 = mu_M lower bound CI
%  7 = mu_M
%  8 = mu_M upper bound CI

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


wait_verify_count = 0;
wait_flag = 0;
wait_samples = 50;

% Rule flags
R12_claim_count = 0;
R12_claim_flag = 0;
R12_t_inj_loc = 0;

R34_claim_count = 0;
R34_claim_flag = 0;
R34_t_inj_loc = 0;

R56_claim_count = 0;
R56_claim_flag = 0;
R56_t_inj_loc = 0;


% Rule 1 - N points beyond 3-sigma historical value
R1_history = 10;
R1_points = 10;
R1_num_stds = 3;

% Rule 2 - Sliding Average Jump Location Localization
R2_interval_length = 20;
R2_mean_buffer = 5;

% Rule 3 - Jump/Slope Detection Verification
R3_history = 10;
R3_points = 10;
R3_num_stds = 1;

% Rule 4 - Sliding Average Jump Location Localization
R4_interval_length = 20;
R4_mean_buffer = 5;

% Rule 5 - 
R5_slope_inference_mode = 0;
R5_samples = 100;
R5_R_threshold = 0.6;

% Rule 6
R6_R_restart = [0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99, 0.995, 0.999];
R6_restart_count = 0;
R6_num_restarts = length(R6_R_restart);




%% ANALYSIS LOOP EXECUTION
while running == 1;
    %% FILTER INITIAL STEP
    % Kalman Filter
    
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
    
    
    Z_in = [Y_hist(data_start);m_hist(data_start)];
    [xk_k,pk_k] = kalman_filter_step(x0_0,P0_0,F_tt,H,Q,R,Z_in);
    X(data_start) = xk_k(1);
	M(data_start) = xk_k(2);
    
    if data_start > 1
        signal_parameters(2,event_count) = data_start-1;
    end

    %% INITIALIZE FAULT PARAMETERS AFTER DECISION IS MADE
    if fault_parameters(1) == 1
        % NONE, ALL ONLINE
    elseif fault_parameters(1) == 2
        % NONE, ALL ONLINE
    elseif fault_parameters(1) == 3
        [fault_parameters] = intermittant_statistics_calculation(event_count,signal_parameters,fault_parameters,Z_alpha);
        fault_parameters
    end

    
    
    
    for tt = data_start + 1:simulation_end
        %% KALMAN FILTER STEP

        xk1_k1 = xk_k;
        pk1_k1 = pk_k;

        if SS_model == 1
            F_tt = [1, 0; 0, 1];
        elseif SS_model == 2
            del_t = t_hist(tt) - t_hist(tt-1);
            F_tt = [1, del_t; 0, 1];
        end
        
        Z_in = [Y_hist(tt);m_hist(tt)];
        [xk_k,pk_k] = kalman_filter_step(xk1_k1,pk1_k1,F_tt,H,Q,R,Z_in);
        X(tt) = xk_k(1);
        
        if SS_model == 1
            if tt > M_cluster_size
                [M(tt),EMPTY] = linear_regression(t_hist(tt-M_cluster_size+1:tt),X(tt-M_cluster_size+1:tt));
            elseif tt > M_cluster_size/2
                [M(tt),EMPTY] = linear_regression(t_hist(1:tt),X(1:tt));
            end
        elseif SS_model == 2
            M_temp(tt) = xk_k(2);
            if tt > M_cluster_size
                M(tt) = mean(M_temp(tt-M_cluster_size+1:tt));
            end
        end
        
        %% INTERVAL STATISTICAL DATA CALCULATION
        if interval_restart == 1 || tt == 2 || tt == 1
            kk = data_start;
            interval_restart = 1;
            
            [start_X(kk),count_X(kk),cumsum_X(kk),mu_X(kk),mu_LCI_X(kk),mu_UCI_X(kk),std_X(kk)] = ...
                Online_statistics(interval_restart,0,kk,0,0,0,X,Z_alpha);
            
            [start_M(kk),count_M(kk),cumsum_M(kk),mu_M(kk),mu_LCI_M(kk),mu_UCI_M(kk),std_M(kk)] = ...
                Online_statistics(interval_restart,slope_ignore_buffer,kk,0,0,0,M,Z_alpha);
            
            [start_Y(kk),count_Y(kk),cumsum_Y(kk),mu_Y(kk),mu_LCI_Y(kk),mu_UCI_Y(kk),std_Y(kk)] = ...
                Online_statistics(interval_restart,0,kk,0,0,0,Y_hist,Z_alpha);
            
            [start_m(kk),count_m(kk),cumsum_m(kk),mu_m(kk),mu_LCI_m(kk),mu_UCI_m(kk),std_m(kk)] = ...
                Online_statistics(interval_restart,slope_ignore_buffer,kk,0,0,0,m_hist,Z_alpha);
            
            interval_restart = 0;
        end
        
        [start_X(tt),count_X(tt),cumsum_X(tt),mu_X(tt),mu_LCI_X(tt),mu_UCI_X(tt),std_X(tt)] = ...
            Online_statistics(interval_restart,0,tt,start_X(tt-1),count_X(tt-1),cumsum_X(tt-1),X,Z_alpha);
            
        [start_M(tt),count_M(tt),cumsum_M(tt),mu_M(tt),mu_LCI_M(tt),mu_UCI_M(tt),std_M(tt)] = ...
            Online_statistics(interval_restart,slope_ignore_buffer,tt,start_M(tt-1),count_M(tt-1),cumsum_M(tt-1),M,Z_alpha);

        %% ONLINE SIGNAL PARAMETERS UPDATE STEP
        if size(signal_parameters,2) == event_count 
            % Resize for next interval
            signal_parameters = [signal_parameters,zeros(8,1)];
            % Recalculation of previous signal interval final values. 
            signal_parameters(3,event_count) = mu_LCI_X(data_start-1);
            signal_parameters(4,event_count) = mu_X(data_start-1);
            signal_parameters(5,event_count) = mu_UCI_X(data_start-1);
            signal_parameters(6,event_count) = mu_LCI_M(data_start-1);
            signal_parameters(7,event_count) = mu_M(data_start-1);
            signal_parameters(8,event_count) = mu_UCI_M(data_start-1);
        end
        % Current online estimates
        signal_parameters(1,event_count + 1) = data_start;
        signal_parameters(3,event_count + 1) = mu_LCI_X(tt);
        signal_parameters(4,event_count + 1) = mu_X(tt);
        signal_parameters(5,event_count + 1) = mu_UCI_X(tt);
        signal_parameters(6,event_count + 1) = mu_LCI_M(tt);
        signal_parameters(7,event_count + 1) = mu_M(tt);
        signal_parameters(8,event_count + 1) = mu_UCI_M(tt);
        
        [start_Y(tt),count_Y(tt),cumsum_Y(tt),mu_Y(tt),mu_LCI_Y(tt),mu_UCI_Y(tt),std_Y(tt)] = ...
            Online_statistics(interval_restart,0,tt,start_Y(tt-1),count_Y(tt-1),cumsum_Y(tt-1),Y_hist,Z_alpha);

        [start_m(tt),count_m(tt),cumsum_m(tt),mu_m(tt),mu_LCI_m(tt),mu_UCI_m(tt),std_m(tt)] = ...
            Online_statistics(interval_restart,slope_ignore_buffer,tt,start_m(tt-1),count_m(tt-1),cumsum_m(tt-1),m_hist,Z_alpha);
        
        %% ONLINE FAULT PARAMETERS UPDATE STEP
        %  1 = fault type
        %  2 = fault injection index
        %  3 = del_P lower bound CI
        %  4 = del_P
        %  5 = del_P upper bound CI
        %  6 = m lower bound CI
        %  7 = m (calculated from X, NOT from M)
        %  8 = m upper bound CI
        %  9 = mu_f lower bound CI
        % 10 = mu_f
        % 11 = mu_f upper bound CI
        % 12 = mu_n lower bound CI
        % 13 = mu_n
        % 14 = mu_n upper bound CI
        % 15 = mu_ratio lower bound CI
        % 16 = mu_ratio
        % 17 = mu_ratio upper bound CI
        if fault_parameters(1) == 1
            if signal_parameters(3,1) > signal_parameters(5,2)
                del_P_min = signal_parameters(5,2) - signal_parameters(3,1);
                del_P_avg = signal_parameters(4,2) - signal_parameters(4,1);
                del_P_max =  signal_parameters(3,2) - signal_parameters(5,1);
            elseif signal_parameters(5,1) < signal_parameters(3,2)
                del_P_min = signal_parameters(3,2) - signal_parameters(5,1);
                del_P_avg = signal_parameters(4,2) - signal_parameters(4,1);
                del_P_max =  signal_parameters(5,2) - signal_parameters(3,1);
            end
            fault_parameters(3) = del_P_min;
            fault_parameters(4) = del_P_avg;
            fault_parameters(5) = del_P_max;
            fault_parameters(3:5)
        elseif fault_parameters(1) == 2
            t_slope = t_hist(signal_parameters(1,2):tt);
            x_slope = X(signal_parameters(1,2):tt);
            [M_update,EMPTY] = linear_regression(t_slope,x_slope);
            fault_parameters(6) = mu_LCI_M(tt);
            fault_parameters(7) = M_update;
            fault_parameters(8) = mu_UCI_M(tt);
        elseif fault_parameters(1) == 3
            % NONE, RECALCULATED AFTER EACH INTERVAL
        end
        
        
        
        
        
        
        %% FILTER FAULT DETECTION RULES
        

        % RULE 1 - N-POINT BACKWARD SIGMA VALUE THRESHOLD CROSSING
        R1_lower_limit = R2_interval_length + data_start;
        if tt > R1_lower_limit
            MU = mu_X(tt-R1_history);
            STD = std_X(tt-R1_history);

            [jump_flag] = KF_RULE_number_points_above_std(tt,R1_history,MU,STD,R1_points,R1_num_stds,X);

            if jump_flag == 1

                % RULE 2 - SLIDING AVERAGE JUMP LOCATION VERIFICATION
                t_interval = [tt-R2_interval_length:1:tt];
                [R12_t_inj_loc] = T_INJ_REFINE_sliding_average(R2_mean_buffer,t_interval,Y_hist);

                disp('JUMP TRIGGER')
                tt
                R12_t_inj_loc


                [inferred_parameters,data_start] = inferred_parameter_storage_SS1(data_start,N_samples,inferred_parameters,R12_t_inj_loc,sensor_hz,data_ignore,t_hist,X);
%                 m_hist(data_start) = 0;
%                 tt
                event_count = event_count + 1;
%                 data_start = R12_t_inj_loc + 1;
                interval_restart = 1;
                fault_parameters(1) = 3;
                if event_count == 1
                    fault_parameters(2) = R12_t_inj_loc;
                end

%                     tt = simulation_end;
                break
            end
        end
       
        

        % RULE 3 - N-POINT BACKWARD SIGMA VALUE THRESHOLD CROSSING
        R3_lower_limit = R4_interval_length + M_cluster_size + data_start;
        if tt > R3_lower_limit
            MU = mu_M(tt-R3_history);
            STD = std_M(tt-R3_history);

            [jump_flag] = KF_RULE_number_points_above_std(tt,R3_history,MU,STD,R3_points,R3_num_stds,M);

            if jump_flag == 1
                % RULE 4 - SLIDING AVERAGE JUMP LOCATION VERIFICATION
                t_interval = [tt-R4_interval_length:1:tt];
                [R34_t_inj_loc] = T_INJ_REFINE_sliding_average(R4_mean_buffer,t_interval,M);

                disp('SLOPE TRIGGER')
                tt
                R34_t_inj_loc
                [inferred_parameters,data_start] = inferred_parameter_storage_SS1(data_start,N_samples,inferred_parameters,R34_t_inj_loc,sensor_hz,data_ignore,t_hist,X);
%                 m_hist(data_start) = 0;
                event_count = event_count + 1;
%                 data_start = R34_t_inj_loc + 1;
                interval_restart = 1;
                fault_parameters(1) = 2;
                fault_parameters(2) = R34_t_inj_loc;
% 
% %                     tt = simulation_end;
                break
            end
        end

        
        % RULE 5 - PERIODIC R THRESHOLD CROSSING(S)
        if R5_slope_inference_mode == 0
            R5_lower_limit = 1 + R5_samples;
            if tt > R5_lower_limit 
                R_X(tt) = R_correlation(t_hist(tt-R5_samples:tt),X(tt-R5_samples:tt)).^2;
                if R_X(tt) > R5_R_threshold
                    R5_slope_inference_mode = 1;
                    if tt - R5_samples >= 1
                        R56_anchor = tt - R5_samples;
                    else
                        R56_anchor = 1;
                    end
                end
            end
        elseif R5_slope_inference_mode == 1
            R_X(tt) = R_correlation(t_hist(R56_anchor:tt),X(R56_anchor:tt)).^2;
            % RULE 6 - R TRIGGERED LINEAR REGRESSION TRACEBACK
            if R6_restart_count < R6_num_restarts
                if R_X(tt) >= R6_R_restart(R6_restart_count + 1)
                    [m_R,yo_R] = linear_regression(t_hist(R56_anchor:tt),X(R56_anchor:tt));

                     x_nominal = mu_X(R56_anchor);

                     t_inj_guess_R = round((x_nominal - yo_R)/m_R);
                     R56_t_inj_loc = t_inj_guess_R*sensor_hz;

                     R56_anchor = R56_t_inj_loc;
%                      tt
%                      R6_R_restart(R6_restart_count + 1)
%                      R56_anchor = t_inj_loc_guess_R
%                      m_R
                     
%                      data_start = R56_t_inj_loc + 1;
                     R6_restart_count = R6_restart_count + 1;
                end
            end
        end
        
        % Wait interval update until time to determine fault
        if wait_flag == 1
            wait_verify_count = wait_verify_count + 1;
        end
        
        %% FILTER FAULT IDENTIFICATION DECISION MAKER
        if wait_samples == wait_verify_count
            
            
            
            
            
            fault_parameters(1) = 4;
            fault_parameters(2) = t_inj_loc_final;
            
        end
        
        
    end
    %% MULTI-FILTER ANALYSIS LOOP TERMINATION
    if tt == simulation_end
        signal_parameters(2,end) = tt;
        [fault_parameters] = intermittant_statistics_calculation(event_count+2,signal_parameters,fault_parameters,Z_alpha);
        
        
        t_inj_loc_guess = tt;
        [inferred_parameters,data_restart] = inferred_parameter_storage_SS1(data_start,N_samples,inferred_parameters,t_inj_loc_guess,sensor_hz,data_ignore,t_hist,X);

        
        running = 0;
    end
end


%% GRAPHICAL ANALYSIS


run_time_total = toc;
run_time_total


event_count
inferred_parameters

signal_parameters

fault_parameters

% KALMAN FILTER
if plots == 1
    figure
    plot(t_hist,P_hist(1,:),'k-',t_hist,Y_hist,'r--')
    title('Sensor State X(t) vs. Time -- ACCUMULATING STATS')
    hold on
    plot(t_hist,X,'g-',t_hist,X,'gx')
    % ACCUMULATED STATISTICAL DATA
    if plot_sensor_stats == 1
        for ii = 1:event_count + 1
            bounds = inferred_parameters(1:2,ii);
            interval = bounds(1):1:bounds(2);
            plot_fault_interval_stat_data(stat_plot_buffer,'b', num_stds_accumulating,  t_hist(interval), mu_Y(interval), mu_LCI_Y(interval), mu_UCI_Y(interval), std_Y(interval))
        end
    end
    if plot_filter_stats == 1
        for ii = 1:event_count + 1
            bounds = inferred_parameters(1:2,ii);
            interval = bounds(1):1:bounds(2);
            plot_fault_interval_stat_data(stat_plot_buffer,'m', num_stds_accumulating,  t_hist(interval), mu_X(interval), mu_LCI_X(interval), mu_UCI_X(interval), std_X(interval))
        end
    end

    
   figure
%     plot(t_hist,P_hist(2,:),'k-')
%     plot(t_hist,P_hist(2,:),'k-',t_hist,m_hist,'r--')
    hold on
    plot(t_hist,M,'g-',t_hist,M,'gx')
    title('Sensor State M(t) vs. Time -- ACCUMULATING STATS')
    hold on
    % ACCUMULATED STATISTICAL DATA
    if plot_sensor_stats == 1
        for ii = 1:event_count + 1
            bounds = inferred_parameters(1:2,ii);
            interval = bounds(1):1:bounds(2);
            plot_fault_interval_stat_data(stat_plot_buffer,'b', num_stds_accumulating,  t_hist(interval), mu_m(interval), mu_LCI_m(interval), mu_UCI_m(interval), std_m(interval))
        end
    end
    if plot_filter_stats == 1
        for ii = 1:event_count + 1
            bounds = inferred_parameters(1:2,ii);
            interval = bounds(1):1:bounds(2);
            plot_fault_interval_stat_data(stat_plot_buffer,'m', num_stds_accumulating,  t_hist(interval), mu_M(interval), mu_LCI_M(interval), mu_UCI_M(interval), std_M(interval))
        end
    end 
    
    figure
    plot(t_hist,R_X,'k-')
    grid on
end


