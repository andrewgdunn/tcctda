function [] = Online_Filter_Mk4()
clc
format short g
tic
%% FILTERING PROBLEM PARAMETERS

% Experiment Parameters
fault_type = 2;

% Universal Parameters
std_P = 0.01;
std_Y = .1;
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
F1 = 1;
H1 = 1;
Q1 = std_P^2;
R1 = std_Y^2;


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
    del_P = .6
    
elseif fault_type == 2 
    % Fault type == 2 (Ramp) Parameters
    % slope_P = 2*max_m*(rand-0.5)
    slope_P = ceil(2*max_m*abs(rand-0.5)/slope_jump)*slope_jump*sign(rand-.5);
    t_inj_loc = 800
    slope_P = sign(rand-0.5)*0.5
    
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
% Z_hist(2,2:end) = (Z_hist(1,2:end) - Z_hist(1,1:end-1))./(t_hist(1,2:end) - t_hist(1,1:end-1));


% % % load Exp_1127_032f_pb_ADAPT-Lite
% % % 
% % % 
% % % indices = isfinite(IT281);
% % % t_hist = Time(indices)';
% % % Y_hist = IT281(indices)';
% % % N_samples = length(t_hist);
% % % 
% % % P_hist = zeros(1,N_samples);
% % % 
% % % Z_hist = [Y_hist;zeros(1,N_samples)];



%% FILTER DATA SET INITIALIZATIONS

% Sensor Parameterization Storage
inferred_parameters_KF1 = [];
event_count_KF1 = 0;


% Kalman Filter :: x(t) = [X(t)]
% uo = u = 0;
X_KF1 = zeros(1,N_samples);

% KF Zero'th Point Values
L = 200;

x10_0 = 0;
P10_0 = L;


% Kalman Filtered State Fault Interval Statistical Data
pre_stat_KF1 = zeros(3,N_samples);
mu_X_KF1 = zeros(3,N_samples);
std_X_KF1 = zeros(1,N_samples);

pre_stat_Z1 = zeros(3,N_samples,2);
mu_Z1 = zeros(3,N_samples,2);
std_Z1 = zeros(1,N_samples,2);

% FILTER 1 RULES
KF1_R3_Rsquared_X = zeros(1,N_samples);




%% ANALYSIS LOOP SETTINGS
running = 1;
simulation_end = N_samples;
data_start_KF1 = 1;
interval_restart = 0;

% FILTER ONE
KF1_identified_fault = 0;
KF1_jump_check_count = 0;
% Rule 1 - N points beyond 3-sigma historical value
KF1_R1_type_mu = 2; % REMOVE IN FINAL VERSION
KF1_R1_history = 10;
KF1_R1_points = 10;
KF1_R1_num_stds = 3;
% Rule 2 - Sliding Average Jump Location Localization
KF1_R1_interval_length = 20;
KF1_R1_mean_buffer = 5;
% Rule 3 - Jump/Slope Detection Verification
KF1_R3_samples = 60;
KF1_R3_jump_check_R = 0.6;


break_flag = 0;

m_per = 40;
slope_accum = zeros(1,N_samples);
slope_fault = zeros(1,N_samples);
slope_accum_Y = zeros(1,N_samples);
slope_fault_Y = zeros(1,N_samples);

slope_period = zeros(1,N_samples);
slope_period_Y = zeros(1,N_samples);

slope_accum_T = zeros(1,N_samples);
slope_fault_T = zeros(1,N_samples);
slope_accum_T_Y = zeros(1,N_samples);
slope_fault_T_Y = zeros(1,N_samples);


%% ANALYSIS LOOP EXECUTION
while running == 1;
    %% FILTER INITIAL STEP
    % Kalman Filter
    [x1k_k,p1k_k,X_KF1] = kalman_filter_step(data_start_KF1,x10_0,P10_0,F1,H1,Q1,R1,X_KF1,Z_hist(1,:));
    
%     if break_flag == 1
%         break
%     end
    
    
    for tt = data_start_KF1 + 1:simulation_end
        %% NOISE STANDARD DEVIATION UPDATE STEP
        % For online std(y_hist) calculation up until this point if the value falls below the
        % currently used value for better estimation
        
        % Stop after a fixed number of points or ANY event occurs, whichever comes first
        
        %% KALMAN FILTER STEP

        x1k1_k1 = x1k_k;
        p1k1_k1 = p1k_k;

        [x1k_k,p1k_k,X_KF1] = kalman_filter_step(tt,x1k1_k1,p1k1_k1,F1,H1,Q1,R1,X_KF1,Z_hist(1,:));
       

        %% INTERVAL STATISTICAL DATA CALCULATION
        if interval_restart == 1 || tt == 2 || tt == 1
            kk = data_start_KF1;
            interval_restart = 1;

            [pre_stat_KF1(:,kk),mu_X_KF1(:,kk),std_X_KF1(1,kk)] = Online_statistics(interval_restart,kk,0,X_KF1(1,:),Z_alpha);

            [pre_stat_Z1(:,kk,1),mu_Z1(:,kk,1),std_Z1(1,kk,1)] = Online_statistics(interval_restart,kk,0,Z_hist(1,:),Z_alpha);
            [pre_stat_Z1(:,kk,2),mu_Z1(:,kk,2),std_Z1(1,kk,2)] = Online_statistics(interval_restart,kk,0,Z_hist(2,:),Z_alpha);

            interval_restart = 0;
            linreg_fault_start = data_start_KF1;
        end

        [pre_stat_KF1(:,tt),mu_X_KF1(:,tt),std_X_KF1(1,tt)] = Online_statistics(interval_restart,tt,pre_stat_KF1(:,tt-1,1),X_KF1(1,:),Z_alpha);

        [pre_stat_Z1(:,tt,1),mu_Z1(:,tt,1),std_Z1(1,tt,1)] = Online_statistics(interval_restart,tt,pre_stat_Z1(:,tt-1,1),Z_hist(1,:),Z_alpha);
        [pre_stat_Z1(:,tt,2),mu_Z1(:,tt,2),std_Z1(1,tt,2)] = Online_statistics(interval_restart,tt,pre_stat_Z1(:,tt-1,2),Z_hist(2,:),Z_alpha);

        [slope_accum(tt),~] = linear_regression(t_hist(1:tt),X_KF1(1:tt));
        [slope_accum_Y(tt),~] = linear_regression(t_hist(1:tt),Z_hist(1,1:tt));
        
        [slope_fault(tt),~] = linear_regression(t_hist(linreg_fault_start:tt),X_KF1(linreg_fault_start:tt));
        [slope_fault_Y(tt),~] = linear_regression(t_hist(linreg_fault_start:tt),Z_hist(1,linreg_fault_start:tt));
        
        
        [slope_accum_T(tt),~] = linear_regression(t_hist(1:tt),X_KF1(1:tt));
        [slope_accum_T_Y(tt),~] = linear_regression(t_hist(1:tt),Z_hist(1,1:tt));
        if tt <= t_inj_loc
            [slope_fault_T(tt),~] = linear_regression(t_hist(1:tt),X_KF1(1:tt));
            [slope_fault_T_Y(tt),~] = linear_regression(t_hist(1:tt),Z_hist(1,1:tt));
        else
            [slope_fault_T(tt),~] = linear_regression(t_hist(t_inj_loc:tt),X_KF1(t_inj_loc:tt));
            [slope_fault_T_Y(tt),~] = linear_regression(t_hist(t_inj_loc:tt),Z_hist(1,t_inj_loc:tt));
        end
        
        if tt > m_per
            [slope_period(tt),~] = linear_regression(t_hist(tt-m_per:tt),X_KF1(tt-m_per:tt));
            [slope_period_Y(tt),~] = linear_regression(t_hist(tt-m_per:tt),Z_hist(1,tt-m_per:tt));
        end
        
        %% FILTER RESTART CONDITIONALS
        
        if KF1_identified_fault == 0 || KF1_identified_fault == 1
            % RULE 1 - N-POINT BACKWARD SIGMA VALUE THRESHOLD CROSSING
            KF1_R1_lower_limit = KF1_R1_interval_length + data_start_KF1;
            if tt > KF1_R1_lower_limit
                MU = mu_X_KF1(KF1_R1_type_mu,tt-KF1_R1_history);
                STD = std_X_KF1(tt-KF1_R1_history);

                [jump_flag] = KF_RULE_number_points_above_std(tt,KF1_R1_history,MU,STD,KF1_R1_points,KF1_R1_num_stds,X_KF1);

                if jump_flag == 1
                    % RULE 2 - SLIDING AVERAGE JUMP LOCATION VERIFICATION
                    t_interval = [tt-KF1_R1_interval_length:1:tt];
                    [t_inj_loc_guess] = T_INJ_REFINE_sliding_average(KF1_R1_mean_buffer,t_interval,Z_hist(1,:));

%                     break_flag = 1;
%                     break
                    
                    [inferred_parameters_KF1,data_start_KF1] = inferred_parameter_storage_SS1(data_start_KF1,N_samples,inferred_parameters_KF1,t_inj_loc_guess,sensor_hz,data_ignore,t_hist,X_KF1);
                    Z_hist(2,data_start_KF1) = 0;
                    event_count_KF1 = event_count_KF1 + 1;
                    interval_restart = 1;
                    break
                end
            end
        end
       
        % RULE 3 - JUMP/SLOPE EVENT VERIFICATION
%         if KF1_identified_fault == 0;
            KF1_R3_lower_limit = 1 + KF1_R3_samples;
            if tt > KF1_R3_lower_limit
                KF1_R3_Rsquared_X(tt) = R_correlation(t_hist(tt-KF1_R3_samples:tt),X_KF1(1,tt-KF1_R3_samples:tt)).^2;
            end

            if event_count_KF1 ~= KF1_jump_check_count
                
                jump_loc = inferred_parameters_KF1(2,KF1_jump_check_count + 1);
                if tt > jump_loc + KF1_R3_samples && jump_loc + KF1_R3_samples < N_samples

                    T_L = t_hist(jump_loc:jump_loc + KF1_R3_samples/2);
                    X_L = KF1_R3_Rsquared_X(jump_loc:jump_loc + KF1_R3_samples/2);
                    R_left = R_correlation(T_L,X_L);
                    R_left = sign(R_left)*R_left^2;

                    T_R = t_hist(jump_loc + KF1_R3_samples/2:jump_loc + KF1_R3_samples);
                    X_R = KF1_R3_Rsquared_X(jump_loc + KF1_R3_samples/2:jump_loc + KF1_R3_samples);
                    R_right = R_correlation(T_R,X_R);
                    R_right = sign(R_right)*R_right^2;

%                     if R_left > KF1_R3_jump_check_R && R_right < -KF1_R3_jump_check_R
%                         KF1_identified_fault = 1
%                     else
%                         KF1_identified_fault = 2
%                         inferred_parameters_KF1 = [];
%                     end
                  
                    if KF1_jump_check_count == 1
                    R_left
                    R_right
                    
                    t_inj_loc
                    t_inj_loc_guess
                    
                    KF1_jump_check_count
                    end
                    
                    KF1_jump_check_count = KF1_jump_check_count + 1;
                end
            end
%         end
        
    end
    %% MULTI-FILTER ANALYSIS LOOP TERMINATION
    if tt == simulation_end
        t_inj_loc_guess = tt;
        if KF1_identified_fault == 2
            [inferred_parameters_KF1,data_restart_KF1] = inferred_parameter_storage_SS1(1,N_samples,inferred_parameters_KF1,t_inj_loc_guess,sensor_hz,data_ignore,t_hist,X_KF1);
        else
            [inferred_parameters_KF1,data_restart_KF1] = inferred_parameter_storage_SS1(data_start_KF1,N_samples,inferred_parameters_KF1,t_inj_loc_guess,sensor_hz,data_ignore,t_hist,X_KF1);
        end
        
        running = 0;
    end
end


%% GRAPHICAL ANALYSIS


run_time_total = toc;
run_time_total

t_inj_loc


event_count_KF1
inferred_parameters_KF1


INFP = inferred_parameters_KF1;

value = 100;

prior_t = t_hist(INFP(2,1)-value:INFP(2,1));
post_t = t_hist(INFP(2,1):INFP(2,1)+value);

prior_X = X_KF1(INFP(2,1)-value:INFP(2,1));
post_X = X_KF1(INFP(2,1):INFP(2,1)+value);

prior_Y = Y_hist(1,INFP(2,1)-value:INFP(2,1));
post_Y = Y_hist(1,INFP(2,1):INFP(2,1)+value);

R_correlation(prior_t,prior_X)
R_correlation(post_t,post_X)


R_correlation(prior_t,prior_Y)
R_correlation(post_t,post_Y)

figure
plot(t_hist,slope_accum,'k-')
% axis([0,t_hist(end),-.01,.01])
title('detected fault')
grid on
hold on
plot(t_hist,slope_fault,'ko')
plot(t_hist,slope_accum_Y,'r-')
plot(t_hist,slope_fault_Y,'ro')


figure
plot(t_hist,slope_accum_T,'k-')
% axis([0,t_hist(end),-.01,.01])
title('TRUE fault')
grid on
hold on
plot(t_hist,slope_fault_T,'ko')
plot(t_hist,slope_accum_T_Y,'r-')
plot(t_hist,slope_fault_T_Y,'ro')


figure
plot(t_hist,slope_period,'k-')
title('PERIODIC slope')
grid on
hold on
plot(t_hist,slope_period_Y,'ro')



% buffer = 20;
% [m_before,~] = linear_regression(t_hist(INFP(2,1)-buffer:INFP(2,1)),X_KF1(INFP(2,1)-buffer:INFP(2,1)));
% 
% [m_after,~] = linear_regression(t_hist(INFP(1,2):INFP(1,2)+buffer),X_KF1(INFP(1,2):INFP(1,2)+buffer));
% 
% m_before
% m_after
% 
% X_slopes = (X_KF1(2:end) - X_KF1(1:end-1)) ./ (t_hist(2:end) - t_hist(1:end-1));
% 
% X_slopes(INFP(2,1)-10:INFP(2,1)+10)
% 
% figure
% plot(t_hist(2:end),X_slopes)



% KF1_M_R = zeros(2,event_count_KF1 + 1);
% KF2_M_R = zeros(2,event_count_KF1 + 1);
% 
% INFP = inferred_parameters_KF1;
% for i = 1:event_count_KF1+1
%     [KF1_M_R(1,i), ~] = linear_regression(t_hist(INFP(1,i):INFP(2,i)),X_KF1(INFP(1,i):INFP(2,i)));
%     
%     KF1_M_R(2,i) = R_correlation(t_hist(INFP(1,i):INFP(2,i)),X_KF1(INFP(1,i):INFP(2,i)));
% end
% 
% KF1_M_R'


% t_hist(inferred_parameters_KF1(2,1))
% if fault_type == 1
%     del_P
% elseif fault_type == 2
%     slope_P
% end
% t_inj


% KALMAN FILTER
if plot_state == 1
    figure
    plot(t_hist,P_hist(1,:),'k-',t_hist,Z_hist(1,:),'r--')
    title('Sensor State X(t) vs. Time -- ACCUMULATING STATS')
    hold on
    plot(t_hist,X_KF1(1,:),'g-',t_hist,X_KF1(1,:),'gx')
    % ACCUMULATED STATISTICAL DATA
    if plot_sensor_stats == 1
        for ii = 1:event_count_KF1 + 1
            bounds = inferred_parameters_KF1(1:2,ii);
            interval = bounds(1):1:bounds(2);
            plot_fault_interval_stat_data(stat_plot_buffer,'b', num_stds_accumulating,  t_hist(interval), mu_Z1(:,interval,1), std_Z1(:,interval,1))
        end
    end
    if plot_filter_stats == 1
        for ii = 1:event_count_KF1 + 1
            bounds = inferred_parameters_KF1(1:2,ii);
            interval = bounds(1):1:bounds(2);
            plot_fault_interval_stat_data(stat_plot_buffer,'m', num_stds_accumulating,  t_hist(interval), mu_X_KF1(:,interval,1), std_X_KF1(:,interval,1))
        end
    end

    
%     figure
%     plot(t_hist,KF1_R3_Rsquared_X,'ro')   
%     axis([0,N_samples/sensor_hz,0,1])
%     hold on
%     jump_loc = inferred_parameters_KF1(2,1);
%     if jump_loc + KF1_R3_samples < N_samples
%         plot(t_hist(jump_loc:jump_loc + KF1_R3_samples),KF1_R3_Rsquared_X(jump_loc:jump_loc + KF1_R3_samples),'ko')
%     end
    
%     figure
%     plot(t_hist,KF1_R3_Rsquared_X,'rx')   
%     axis([0,N_samples/sensor_hz,0,1])
%     hold on
%     jump_loc = inferred_parameters_KF1(2,1);
%     if jump_loc + KF1_R3_samples < N_samples
%         plot(t_hist(jump_loc:jump_loc + KF1_R3_samples),KF1_R3_Rsquared_X(jump_loc:jump_loc + KF1_R3_samples),'ko')
%     
%         T_L = t_hist(jump_loc:jump_loc + KF1_R3_samples/2);
%         X_L = KF1_R3_Rsquared_X(jump_loc:jump_loc + KF1_R3_samples/2);
%         R_left = R_correlation(T_L,X_L);
%         R_left = sign(R_left)*R_left^2;
% 
%         T_R = t_hist(jump_loc + KF1_R3_samples/2:jump_loc + KF1_R3_samples);
%         X_R = KF1_R3_Rsquared_X(jump_loc + KF1_R3_samples/2:jump_loc + KF1_R3_samples);
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





