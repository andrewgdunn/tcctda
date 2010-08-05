function [] = Online_Filter_Mk5()
clc
format short g
tic
%% FILTERING PROBLEM PARAMETERS

% Experiment Parameters
fault_type = 2;

% Universal Parameters
std_P = 0.01;
std_Y = .05;
sensor_hz = 10;
dt = 1/sensor_hz;
t_lower = 0;
t_upper = 200;

% Statistical Analysis Parameters
alpha = 0.01;
Z_alpha = norminv(1-alpha/2);

% Event Inference Parameters
% identified_fault = 0; % 0 = nominal, 1 = jump/intermittant, 2 = ramp
data_ignore = 0;

% Graphical Parameters
stat_plot_buffer = 5; % 0 == graph everything

plot_state = 1;
plot_slope = 0;

plot_sensor_stats = 1;
plot_filter_stats = 1;

num_stds_accumulating = 3;

% Fault type == 1 (Step) Parameters
max_P = 5;
max_delP = 1;
max_delP_precision = 0.1;

% Fault type == 2 (Ramp) Parameters
max_m = 0.1;
slope_jump = 0.001; % REMOVE LATER

% Fault type == 3 (Intermittant) Parameters
max_delP_i = 1;
max_std_delP_i = 0.05;

sample_min_f = 15;
sample_max_f = 30;
sample_min_n = 20;
sample_max_n = 50;

%% KALMAN FILTER PARAMETERS
% x_k = F_k*x_k-1 + B_k*u_k + G_k*w_k
% y_k = H_k*x_k + I*v_k
% w_k ~ N(0,Q_k)
% v_k ~ N(0,R_k)
% x(k|k-1) = F_k*x(k-1|k-1) + B_k*u_k
% P(k|k-1) = F_k*P(k-1|k-1)*F_k' + Q_k 
% B = 0;

% dt == 0 causes slope offset for 0.05 or greater, but has much narrower CI's even for slope
F = [1, dt; 0, 1];
H = [1, 0 ; 0 1];
Q = [1,0;0,1]*std_P^2;
R = [1,0;0,1]*std_Y^2;


%% FILTERING PROBLEM RANDOM VARIABLE GENERATION
% CURRENTLY DISCRETE JUMP INTERVALS FOR TESTING -- SWITCH TO CONTINUOUS LATER

% Universal Parameters
t_inj_loc = floor((t_upper - t_lower)*sensor_hz*rand);
% P = 2*max_P*(rand - 0.5);
P = round(2*max_P*(rand - 0.5));

if fault_type == 1
    % Fault type == 1 (Step) Parameters
    % del_P = 2*max_delP*(rand-0.5);
    del_P = sign(rand-.5).*ceil(2*max_delP*abs(rand-0.5)/max_delP_precision)*max_delP_precision;
    t_inj_loc = 600
    del_P = .1
    
elseif fault_type == 2 
    % Fault type == 2 (Ramp) Parameters
    % slope_P = 2*max_m*(rand-0.5)
    slope_P = ceil(2*max_m*abs(rand-0.5)/slope_jump)*slope_jump*sign(rand-.5);
    t_inj_loc = 800
    slope_P = sign(rand-0.5)*0.005
    
elseif fault_type == 3
    % Fault type == 3 (Intermittant) Parameters
    % mu_del_P = 2*max_delP_i*(rand-0.5);
    mu_del_P = sign(rand-.5).*ceil(2*max_delP_i*abs(rand-0.5));
    std_del_P = max_std_delP_i*rand;

    mu_f_stochastic = ceil((sample_max_f - sample_min_f)*rand)/sensor_hz;
    mu_n_stochastic = ceil((sample_max_n - sample_min_n)*rand)/sensor_hz;
end


%% DEPENDENT PARAMETER CALCULATION AND TIME SERIES DATA GENERATION
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


% % % % % % smallest slope
% load Exp_1127_020f_pb_ADAPT-Lite
% sensor = IT240;

% % % % % % smallest jump
% load Exp_1127_032f_pb_ADAPT-Lite
% sensor = IT281;

% % % % % % % largest slope
% % % % NOT REALLY --> largest in component, actual sensor only -.001, BUT R WILL TRIGGER
% load Exp_1151_pb_ADAPT-Lite
% sensor = IT267;
% % sensor = E265;

% % % % % largest slope
load Exp_1127_026f_pb_ADAPT-Lite
sensor = IT267;

indices = isfinite(sensor);
t_hist = Time(indices)';
Y_hist = sensor(indices)';
N_samples = length(t_hist);

mean(t_hist(2:end) - t_hist(1:end-1))

P_hist = zeros(1,N_samples);

m_hist = [0,(Y_hist(1,2:end) - Y_hist(1,1:end-1))./(t_hist(1,2:end) - t_hist(1,1:end-1))];



%% FILTER DATA SET INITIALIZATIONS

% Sensor Parameterization Storage
inferred_parameters = [];
event_count = 0;


% Kalman Filter :: x(t) = [X(t),M(t)]
% uo = u = 0;
X = zeros(1,N_samples);
M = zeros(1,N_samples);

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

start_M = zeros(1,N_samples);
count_M = zeros(1,N_samples);
cumsum_M = zeros(1,N_samples);

mu_UCI_M = zeros(1,N_samples);
mu_M = zeros(1,N_samples);
mu_LCI_M = zeros(1,N_samples);
std_M = zeros(1,N_samples);

R_X = zeros(1,N_samples);
R_X_PER = zeros(1,N_samples);
R_X_PER2 = zeros(1,N_samples);
R_X_PER3 = zeros(1,N_samples);
R_X_PER4 = zeros(1,N_samples);
R_X_PER5 = zeros(1,N_samples);

R_X_SMT = zeros(1,N_samples);


mu_UCI_X_PER = zeros(1,N_samples);
mu_X_PER = zeros(1,N_samples);
mu_LCI_X_PER = zeros(1,N_samples);
std_X_PER = zeros(1,N_samples);


mu_UCI_M_PER = zeros(1,N_samples);
mu_M_PER = zeros(1,N_samples);
mu_LCI_M_PER = zeros(1,N_samples);
std_M_PER = zeros(1,N_samples);

R_M = zeros(1,N_samples);
R_M_PER = zeros(1,N_samples);
R_M_PER2 = zeros(1,N_samples);
R_M_SMT = zeros(1,N_samples);



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


% FILTER 1 RULES
R3_Rsquared_X = zeros(1,N_samples);




%% ANALYSIS LOOP SETTINGS
running = 1;
simulation_end = N_samples;
data_start = 1;
interval_restart = 0;

% FILTER ONE
identified_fault = 0;
jump_check_count = 0;
% Rule 1 - N points beyond 3-sigma historical value
R1_history = 10;
R1_points = 10;
R1_num_stds = 3;
% Rule 2 - Sliding Average Jump Location Localization
R1_interval_length = 20;
R1_mean_buffer = 5;

% Rule 3 - Jump/Slope Detection Verification
R3_history = 20;
R3_points = 20;
R3_num_stds = 1;

R3_interval_length = 30;
R3_mean_buffer = 5;

% R3_samples = 40;
% R3_jump_check_R = 0.6;

% Set m_per equal to (or less than, but NO MORE THAN) the # of steps one waits to decide between a
% step fault and an intermittant fault

sample_period = 40;
m_per = 0;
ignore_buffer = m_per;
M_temp = zeros(1,N_samples);

R_period = 100;
R_period2 = 200;
R_period3 = 300;
R_period4 = 5000;
R_period5 = 5000;

R_period_SMT = 200;
R_thresh_SMT = 0.5;
R_pass_SMT = 0;

Z_scale = 1;
break_flag = 0;

%% ANALYSIS LOOP EXECUTION
while running == 1;
    %% FILTER INITIAL STEP
    % Kalman Filter
    
    if data_start == 1
        del_t = t_hist(data_start);
        F_i = [1, del_t; 0, 1];
    else
        del_t = t_hist(data_start) - t_hist(data_start-1);
        F_i = [1, del_t; 0, 1];
    end
    
%     F_i = [1, 0; 0, 1];
    
    Z_in = [Y_hist(data_start);Z_scale*m_hist(data_start)];
    [xk_k,pk_k] = kalman_filter_step(x0_0,P0_0,F_i,H,Q,R,Z_in);
    X(data_start) = xk_k(1);
%     M(data_start) = 0;

    if break_flag == 1
        break
    end
    
    for tt = data_start + 1:simulation_end
        %% KALMAN FILTER STEP

        xk1_k1 = xk_k;
        pk1_k1 = pk_k;

        del_t = t_hist(tt) - t_hist(tt-1);
        F_i = [1, del_t; 0, 1];
        
%         F_i = [1, 0; 0, 1];
        
        Z_in = [Y_hist(tt);Z_scale*m_hist(tt)];
        [xk_k,pk_k] = kalman_filter_step(xk1_k1,pk1_k1,F_i,H,Q,R,Z_in);
        X(tt) = xk_k(1);
        M_temp(tt) = xk_k(2);
        if tt > m_per
            M(tt) = mean(M_temp(tt-m_per:tt));
        end

%         if tt > m_per
%             [M(tt),~] = linear_regression(t_hist(tt-m_per+1:tt),X(tt-m_per+1:tt));
%         elseif tt > m_per/2
%             [M(tt),~] = linear_regression(t_hist(1:tt),X(1:tt));
%         end
        
        %% INTERVAL STATISTICAL DATA CALCULATION
        if interval_restart == 1 || tt == 2 || tt == 1
            kk = data_start;
            interval_restart = 1;
            
            [start_X(kk),count_X(kk),cumsum_X(kk),mu_X(kk),mu_LCI_X(kk),mu_UCI_X(kk),std_X(kk)] = ...
                Online_statistics(interval_restart,ignore_buffer,kk,0,0,0,X,Z_alpha);
            
            [start_M(kk),count_M(kk),cumsum_M(kk),mu_M(kk),mu_LCI_M(kk),mu_UCI_M(kk),std_M(kk)] = ...
                Online_statistics(interval_restart,ignore_buffer,kk,0,0,0,M,Z_alpha);
            
            [start_Y(kk),count_Y(kk),cumsum_Y(kk),mu_Y(kk),mu_LCI_Y(kk),mu_UCI_Y(kk),std_Y(kk)] = ...
                Online_statistics(interval_restart,ignore_buffer,kk,0,0,0,Y_hist,Z_alpha);
            
            [start_m(kk),count_m(kk),cumsum_m(kk),mu_m(kk),mu_LCI_m(kk),mu_UCI_m(kk),std_m(kk)] = ...
                Online_statistics(interval_restart,ignore_buffer,kk,0,0,0,m_hist,Z_alpha);
            
            interval_restart = 0;

        end
        
        
        [start_X(tt),count_X(tt),cumsum_X(tt),mu_X(tt),mu_LCI_X(tt),mu_UCI_X(tt),std_X(tt)] = ...
            Online_statistics(interval_restart,ignore_buffer,tt,start_X(tt-1),count_X(tt-1),cumsum_X(tt-1),X,Z_alpha);
            
        [start_M(tt),count_M(tt),cumsum_M(tt),mu_M(tt),mu_LCI_M(tt),mu_UCI_M(tt),std_M(tt)] = ...
            Online_statistics(interval_restart,ignore_buffer,tt,start_M(tt-1),count_M(tt-1),cumsum_M(tt-1),M,Z_alpha);

        [start_Y(tt),count_Y(tt),cumsum_Y(tt),mu_Y(tt),mu_LCI_Y(tt),mu_UCI_Y(tt),std_Y(tt)] = ...
            Online_statistics(interval_restart,ignore_buffer,tt,start_Y(tt-1),count_Y(tt-1),cumsum_Y(tt-1),Y_hist,Z_alpha);

        [start_m(tt),count_m(tt),cumsum_m(tt),mu_m(tt),mu_LCI_m(tt),mu_UCI_m(tt),std_m(tt)] = ...
            Online_statistics(interval_restart,ignore_buffer,tt,start_m(tt-1),count_m(tt-1),cumsum_m(tt-1),m_hist,Z_alpha);
        
        R_X(tt) = R_correlation(t_hist(1:tt),X(1:tt))^2;
        if tt > R_period
            R_X_PER(tt) = R_correlation(t_hist(tt-R_period:tt),X(tt-R_period:tt))^2;
        end
        if tt > R_period2
            R_X_PER2(tt) = R_correlation(t_hist(tt-R_period2:tt),X(tt-R_period2:tt))^2;
        end
        if tt > R_period3
            R_X_PER3(tt) = R_correlation(t_hist(tt-R_period3:tt),X(tt-R_period3:tt))^2;
        end
        if tt > R_period4
            R_X_PER4(tt) = R_correlation(t_hist(tt-R_period4:tt),X(tt-R_period4:tt))^2;
        end
        if tt > R_period5
            R_X_PER5(tt) = R_correlation(t_hist(tt-R_period5:tt),X(tt-R_period5:tt))^2;
        end

%         if tt > R_period_SMT 
%             if R_X_SMT(tt-1) < R_thresh_SMT 
%                 R_X_SMT(tt) = R_correlation(t_hist(tt-R_period_SMT:tt),X(tt-R_period_SMT:tt))^2;
%             elseif R_X_SMT(tt-1) >= R_thresh_SMT && R_pass_SMT == 0
%                 anchor_SMT = tt-1;
%                 R_X_SMT(tt) = R_correlation(t_hist(tt-anchor_SMT:tt),X(tt-anchor_SMT:tt))^2;
%                 R_pass_SMT = 1;
%             elseif R_X_SMT(tt-1) >= R_thresh_SMT && R_pass_SMT == 1
%                 R_X_SMT(tt) = R_correlation(t_hist(tt-anchor_SMT:tt),X(tt-anchor_SMT:tt))^2;
%             end
%         end
        
        R_M(tt) = R_correlation(t_hist(1:tt),M(1:tt))^2;
        if tt > R_period
            R_M_PER(tt) = R_correlation(t_hist(tt-R_period:tt),M(tt-R_period:tt))^2;
        end
        
%         if tt > sample_period
%             [mu_X_PER(tt),mu_LCI_X_PER(tt),mu_UCI_X_PER(tt),std_X_PER(tt)] = ...
%                 Online_statistics_periodic(sample_period,tt,X,Z_alpha);
%             
%             [mu_M_PER(tt),mu_LCI_M_PER(tt),mu_UCI_M_PER(tt),std_M_PER(tt)] = ...
%                 Online_statistics_periodic(sample_period,tt,M,Z_alpha);
%         end

            if mod(tt,sample_period) == 0
                [mu,mu_L,mu_U,std] = ...
                    Online_statistics_periodic(sample_period-1,tt,X,Z_alpha);
                mu_X_PER(tt-sample_period+1:tt) = mu;
                mu_LCI_X_PER(tt-sample_period+1:tt) = mu_L;
                mu_UCI_X_PER(tt-sample_period+1:tt) = mu_U;
                std_X_PER(tt-sample_period+1:tt) = std;

                [mu,mu_L,mu_U,std] = ...
                    Online_statistics_periodic(sample_period-1,tt,M,Z_alpha);
                mu_M_PER(tt-sample_period+1:tt) = mu;
                mu_LCI_M_PER(tt-sample_period+1:tt) = mu_L;
                mu_UCI_M_PER(tt-sample_period+1:tt) = mu_U;
                std_M_PER(tt-sample_period+1:tt) = std;
            elseif tt == data_start + 1
                [mu,mu_L,mu_U,std] = ...
                    Online_statistics_periodic(data_start,tt,X,Z_alpha);
                mu_X_PER(1:tt) = mu;
                mu_LCI_X_PER(1:tt) = mu_L;
                mu_UCI_X_PER(1:tt) = mu_U;
                std_X_PER(1:tt) = std;

                [mu,mu_L,mu_U,std] = ...
                    Online_statistics_periodic(data_start,tt,M,Z_alpha);
                mu_M_PER(1:tt) = mu;
                mu_LCI_M_PER(1:tt) = mu_L;
                mu_UCI_M_PER(1:tt) = mu_U;
                std_M_PER(1:tt) = std;
            end

        
        
        %% FILTER RESTART CONDITIONALS
        
%         if identified_fault == 0 || identified_fault == 1
%             % RULE 1 - N-POINT BACKWARD SIGMA VALUE THRESHOLD CROSSING
%             R1_lower_limit = R1_interval_length + data_start;
%             if tt > R1_lower_limit
%                 MU = mu_X(tt-R1_history);
%                 STD = std_X(tt-R1_history);
% 
%                 [jump_flag] = KF_RULE_number_points_above_std(tt,R1_history,MU,STD,R1_points,R1_num_stds,X);
% 
%                 if jump_flag == 1
%                     % RULE 2 - SLIDING AVERAGE JUMP LOCATION VERIFICATION
%                     t_interval = [tt-R1_interval_length:1:tt];
%                     [t_inj_loc_guess] = T_INJ_REFINE_sliding_average(R1_mean_buffer,t_interval,Y_hist);
%                     
%                     [inferred_parameters,data_start] = inferred_parameter_storage_SS1(data_start,N_samples,inferred_parameters,t_inj_loc_guess,sensor_hz,data_ignore,t_hist,X);
%                     m_hist(data_start) = 0;
%                     event_count = event_count + 1;
%                     interval_restart = 1;
%                     
% %                     tt = simulation_end;
%                     break
%                 end
%             end
%         end
       
        
%         if identified_fault == 0
            % RULE 3 - N-POINT BACKWARD SIGMA VALUE THRESHOLD CROSSING
            R3_lower_limit = R3_interval_length + m_per + data_start;
            if tt > R3_lower_limit
                MU = mu_M(tt-R3_history);
                STD = std_M(tt-R3_history);

                [jump_flag] = KF_RULE_number_points_above_std(tt,R3_history,MU,STD,R3_points,R3_num_stds,M);

                if jump_flag == 1
                    % RULE 2 - SLIDING AVERAGE JUMP LOCATION VERIFICATION
                    t_interval = [tt-R3_interval_length:1:tt];
                    [t_inj_loc_guess] = T_INJ_REFINE_sliding_average(R3_mean_buffer,t_interval,M);
                    
                    [inferred_parameters,data_start] = inferred_parameter_storage_SS1(data_start,N_samples,inferred_parameters,t_inj_loc_guess,sensor_hz,data_ignore,t_hist,X);
                    m_hist(data_start) = 0;
                    event_count = event_count + 1;
                    interval_restart = 1;
                    
%                     tt = simulation_end;
                    break
                end
            end
%         end
        
        
        
        
        
%         % RULE 3 - JUMP/SLOPE EVENT VERIFICATION
% %         if identified_fault == 0;
%             R3_lower_limit = 1 + R3_samples;
%             if tt > R3_lower_limit
%                 R3_Rsquared_X(tt) = R_correlation(t_hist(tt-R3_samples:tt),X(1,tt-R3_samples:tt)).^2;
%             end
% 
%             if event_count ~= jump_check_count
%                 
%                 jump_loc = inferred_parameters(2,jump_check_count + 1);
%                 if tt > jump_loc + R3_samples && jump_loc + R3_samples < N_samples
% 
%                     T_L = t_hist(jump_loc:jump_loc + R3_samples/2);
%                     X_L = R3_Rsquared_X(jump_loc:jump_loc + R3_samples/2);
%                     R_left = R_correlation(T_L,X_L);
%                     R_left = sign(R_left)*R_left^2;
% 
%                     T_R = t_hist(jump_loc + R3_samples/2:jump_loc + R3_samples);
%                     X_R = R3_Rsquared_X(jump_loc + R3_samples/2:jump_loc + R3_samples);
%                     R_right = R_correlation(T_R,X_R);
%                     R_right = sign(R_right)*R_right^2;
% 
% %                     if R_left > R3_jump_check_R && R_right < -R3_jump_check_R
% %                         identified_fault = 1
% %                     else
% %                         identified_fault = 2
% %                         inferred_parameters = [];
% %                     end
%                   
%                     if jump_check_count == 1
%                     R_left
%                     R_right
%                     
%                     t_inj_loc
%                     t_inj_loc_guess
%                     
%                     jump_check_count
%                     end
%                     
%                     jump_check_count = jump_check_count + 1;
%                 end
%             end
%         end
        
    end
    %% MULTI-FILTER ANALYSIS LOOP TERMINATION
    if tt == simulation_end
        t_inj_loc_guess = tt;
        if identified_fault == 2
            [inferred_parameters,data_restart] = inferred_parameter_storage_SS1(1,N_samples,inferred_parameters,t_inj_loc_guess,sensor_hz,data_ignore,t_hist,X);
        else
            [inferred_parameters,data_restart] = inferred_parameter_storage_SS1(data_start,N_samples,inferred_parameters,t_inj_loc_guess,sensor_hz,data_ignore,t_hist,X);
        end
        
        running = 0;
    end
end


%% GRAPHICAL ANALYSIS


run_time_total = toc;
run_time_total

t_inj_loc


event_count
inferred_parameters



% INFP = inferred_parameters;
% 
% value = 100;
% 
% prior_t = t_hist(INFP(2,1)-value:INFP(2,1));
% post_t = t_hist(INFP(2,1):INFP(2,1)+value);
% 
% prior_X = X(INFP(2,1)-value:INFP(2,1));
% post_X = X(INFP(2,1):INFP(2,1)+value);
% 
% prior_Y = Y_hist(1,INFP(2,1)-value:INFP(2,1));
% post_Y = Y_hist(1,INFP(2,1):INFP(2,1)+value);
% 
% R_correlation(prior_t,prior_X)
% R_correlation(post_t,post_X)
% 
% 
% R_correlation(prior_t,prior_Y)
% R_correlation(post_t,post_Y)
% 
% figure
% plot(t_hist,slope_accum,'k-')
% % axis([0,t_hist(end),-.01,.01])
% title('detected fault')
% grid on
% hold on
% plot(t_hist,slope_fault,'ko')
% plot(t_hist,slope_accum_Y,'r-')
% plot(t_hist,slope_fault_Y,'ro')
% 
% 
% figure
% plot(t_hist,slope_accum_T,'k-')
% % axis([0,t_hist(end),-.01,.01])
% title('TRUE fault')
% grid on
% hold on
% plot(t_hist,slope_fault_T,'ko')
% plot(t_hist,slope_accum_T_Y,'r-')
% plot(t_hist,slope_fault_T_Y,'ro')


% figure
% plot(t_hist,slope_period,'k-')
% title('PERIODIC slope')
% grid on
% hold on
% % plot(t_hist,slope_period_Y,'ro')
% plot(t_hist,M,'ro')


% buffer = 20;
% [m_before,~] = linear_regression(t_hist(INFP(2,1)-buffer:INFP(2,1)),X(INFP(2,1)-buffer:INFP(2,1)));
% 
% [m_after,~] = linear_regression(t_hist(INFP(1,2):INFP(1,2)+buffer),X(INFP(1,2):INFP(1,2)+buffer));
% 
% m_before
% m_after
% 
% X_slopes = (X(2:end) - X(1:end-1)) ./ (t_hist(2:end) - t_hist(1:end-1));
% 
% X_slopes(INFP(2,1)-10:INFP(2,1)+10)
% 
% figure
% plot(t_hist(2:end),X_slopes)



% M_R = zeros(2,event_count + 1);
% KF2_M_R = zeros(2,event_count + 1);
% 
% INFP = inferred_parameters;
% for i = 1:event_count+1
%     [M_R(1,i), ~] = linear_regression(t_hist(INFP(1,i):INFP(2,i)),X(INFP(1,i):INFP(2,i)));
%     
%     M_R(2,i) = R_correlation(t_hist(INFP(1,i):INFP(2,i)),X(INFP(1,i):INFP(2,i)));
% end
% 
% M_R'


% t_hist(inferred_parameters(2,1))
% if fault_type == 1
%     del_P
% elseif fault_type == 2
%     slope_P
% end
% t_inj


% KALMAN FILTER
if plot_state == 1
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
%     if plot_sensor_stats == 1
%         for ii = 1:event_count + 1
%             bounds = inferred_parameters(1:2,ii);
%             interval = bounds(1):1:bounds(2);
%             plot_fault_interval_stat_data(stat_plot_buffer,'b', num_stds_accumulating,  t_hist(interval), mu_m(interval), mu_LCI_m(interval), mu_UCI_m(interval), std_m(interval))
%         end
%     end
    if plot_filter_stats == 1
        for ii = 1:event_count + 1
            bounds = inferred_parameters(1:2,ii);
            interval = bounds(1):1:bounds(2);
            plot_fault_interval_stat_data(stat_plot_buffer,'m', num_stds_accumulating,  t_hist(interval), mu_M(interval), mu_LCI_M(interval), mu_UCI_M(interval), std_M(interval))
        end
    end 

   figure
    plot(t_hist,P_hist(1,:),'k-',t_hist,Y_hist,'r--')
    title('Sensor State X(t) vs. Time -- PERIODIC STATS')
    hold on
    plot(t_hist,X,'g-',t_hist,X,'gx')
    % ACCUMULATED STATISTICAL DATA
    if plot_sensor_stats == 1
%         for ii = 1:event_count + 1
%             bounds = inferred_parameters(1:2,ii);
%             interval = bounds(1):1:bounds(2);
%             plot_fault_interval_stat_data(stat_plot_buffer,'b', num_stds_accumulating,  t_hist(interval), mu_Y_PER(interval), mu_LCI_Y_PER(interval), mu_UCI_Y_PER(interval), std_Y_PER(interval))
%         end
    end
    if plot_filter_stats == 1
        for ii = 1:event_count + 1
            bounds = inferred_parameters(1:2,ii);
            interval = bounds(1):1:bounds(2);
            plot_fault_interval_stat_data(stat_plot_buffer,'b', num_stds_accumulating,  t_hist(interval), mu_X_PER(interval), mu_LCI_X_PER(interval), mu_UCI_X_PER(interval), std_X_PER(interval))
        end
    end 
    
   figure
%     plot(t_hist,P_hist(2,:),'k-')
%     plot(t_hist,P_hist(2,:),'k-',t_hist,m_hist,'r--')
    hold on
    plot(t_hist,M,'g-',t_hist,M,'gx')
    title('Sensor State M(t) vs. Time -- PERIODIC STATS')
    hold on
    % ACCUMULATED STATISTICAL DATA
%     if plot_sensor_stats == 1
%         for ii = 1:event_count + 1
%             bounds = inferred_parameters(1:2,ii);
%             interval = bounds(1):1:bounds(2);
%             plot_fault_interval_stat_data(stat_plot_buffer,'b', num_stds_accumulating,  t_hist(interval), mu_m_PER(interval), mu_LCI_m_PER(interval), mu_UCI_m_PER(interval), std_m_PER(interval))
%         end
%     end
    if plot_filter_stats == 1
        for ii = 1:event_count + 1
            bounds = inferred_parameters(1:2,ii);
            interval = bounds(1):1:bounds(2);
            plot_fault_interval_stat_data(stat_plot_buffer,'b', num_stds_accumulating,  t_hist(interval), mu_M_PER(interval), mu_LCI_M_PER(interval), mu_UCI_M_PER(interval), std_M_PER(interval))
        end
    end
    
    figure
    plot(t_hist,R_X,'k-',t_hist,R_X_PER,'r-',t_hist,R_X_PER2,'b-',t_hist,R_X_PER3,'g-',t_hist,R_X_PER4,'c-',t_hist,R_X_PER5,'m-')
    title('X R VALUES')
    axis([0,t_hist(end),0,1])
    legend('accum','period 1','period 2','period 3','period 4','period 5','location','southoutside')
%     axis([0,t_hist(end),-1,1])
    
    figure
    plot(t_hist,R_M,'k-',t_hist,R_M_PER,'r-')
    title('M R VALUES')
    axis([0,t_hist(end),0,1])
%     axis([0,t_hist(end),-1,1])
    
    
    t_i = t_inj_loc;
t_i = inferred_parameters(2,1);

interval = 20;

val_before = mean(X(1:t_i));
val_gap = mean(X(t_i:t_i + interval));
val_after = mean(X(t_i + interval:t_i + 2*interval));
val_both = mean(X(t_i:t_i + 2*interval));

slope_before_m = mean(M(1:t_i));
slope_gap_m = mean(M(t_i:t_i + interval));
slope_after_m = mean(M(t_i + interval:t_i + 2*interval));
slope_both_m = mean(M(t_i:t_i + 2*interval));

slope_before_x = linear_regression(t_hist(1:t_i),X(1:t_i));
slope_gap_x = linear_regression(t_hist(t_i:t_i + interval),X(t_i:t_i + interval));
slope_after_x = linear_regression(t_hist(t_i + interval:t_i + 2*interval),X(t_i + interval:t_i + 2*interval));
slope_both_x = linear_regression(t_hist(t_i:t_i + 2*interval),X(t_i:t_i + 2*interval));


RX1 = R_correlation(t_hist(1:t_i),X(1:t_i));
RX2 = R_correlation(t_hist(t_i:t_i + interval),X(t_i:t_i + interval));
RX3 = R_correlation(t_hist(t_i + interval:t_i + 2*interval),X(t_i + interval:t_i + 2*interval));
RX4 = R_correlation(t_hist(t_i:t_i + 2*interval),X(t_i:t_i + 2*interval));

RM1 = R_correlation(t_hist(1:t_i),M(1:t_i));
RM2 = R_correlation(t_hist(t_i:t_i + interval),M(t_i:t_i + interval));
RM3 = R_correlation(t_hist(t_i + interval:t_i + 2*interval),M(t_i + interval:t_i + 2*interval));
RM4 = R_correlation(t_hist(t_i:t_i + 2*interval),M(t_i:t_i + 2*interval));

before = [val_before,0, RX1^2; slope_before_m,slope_before_x, RM1^2];
gap = [val_gap,0, RX2^2; slope_gap_m,slope_gap_x, RM2^2];
after = [val_after,0, RX3^2; slope_after_m,slope_after_x, RM3^2];
both = [val_both,0, RX4^2; slope_both_m,slope_both_x, RM4^2];

before
gap
after
both

%     figure
%     plot(t_hist,R3_Rsquared_X,'ro')   
%     axis([0,N_samples/sensor_hz,0,1])
%     hold on
%     jump_loc = inferred_parameters(2,1);
%     if jump_loc + R3_samples < N_samples
%         plot(t_hist(jump_loc:jump_loc + R3_samples),R3_Rsquared_X(jump_loc:jump_loc + R3_samples),'ko')
%     end
    
%     figure
%     plot(t_hist,R3_Rsquared_X,'rx')   
%     axis([0,N_samples/sensor_hz,0,1])
%     hold on
%     jump_loc = inferred_parameters(2,1);
%     if jump_loc + R3_samples < N_samples
%         plot(t_hist(jump_loc:jump_loc + R3_samples),R3_Rsquared_X(jump_loc:jump_loc + R3_samples),'ko')
%     
%         T_L = t_hist(jump_loc:jump_loc + R3_samples/2);
%         X_L = R3_Rsquared_X(jump_loc:jump_loc + R3_samples/2);
%         R_left = R_correlation(T_L,X_L);
%         R_left = sign(R_left)*R_left^2;
% 
%         T_R = t_hist(jump_loc + R3_samples/2:jump_loc + R3_samples);
%         X_R = R3_Rsquared_X(jump_loc + R3_samples/2:jump_loc + R3_samples);
%         R_right = R_correlation(T_R,X_R);
%         R_right = sign(R_right)*R_right^2;
% 
%     end
    
    

end
% 
% if plot_slope == 1
%     figure
%     plot(t_hist,P_hist(2,:),'k-',t_hist,Z_hist(2,:),'r--')
%     title('Sensor Data Slope m(t) vs. Time -- ACCUMULATING STATS')
%     hold on
%     plot(t_hist,X_KF(2,:),'g-',t_hist,X_KF(2,:),'gx')
%     % ACCUMULATED STATISTICAL DATA
%     if plot_sensor_stats == 1
%         [MU_M_A,STD_M_A] = plot_accumulating_stat_data(stat_plot_buffer, 'b', alpha, num_stds_accumulating,  t_hist, Z_hist(2,:));
%     end
%     if plot_filter_stats == 1
%         [MU_KF_M_A,STD_KF_M_A] = plot_accumulating_stat_data(stat_plot_buffer, 'm', alpha, num_stds_accumulating,  t_hist, X_KF(2,:));
%     end
% end





