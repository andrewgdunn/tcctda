function [inferred_parameters,data_restart] = inferred_parameter_storage_SS2(data_start,N_samples,inferred_parameters,t_inj_loc_guess,sensor_hz,data_ignore,t_hist,X_KF)
parameters = 7;


inferred_parameters = [inferred_parameters, zeros(parameters,1)];
index = size(inferred_parameters,2);


if t_inj_loc_guess == N_samples
    t_inj_loc_used = t_inj_loc_guess;
else
    t_inj_loc_used = t_inj_loc_guess - data_ignore;
end

T = t_hist(data_start:t_inj_loc_used);
X = X_KF(1,data_start:t_inj_loc_used);
n = length(T);

inferred_parameters(1,index) = data_start;
inferred_parameters(2,index) = t_inj_loc_used;
inferred_parameters(3,index) = (data_start-1)/sensor_hz;
inferred_parameters(4,index) = (t_inj_loc_used-1)/sensor_hz;
inferred_parameters(5,index) = mean(X_KF(1,data_start:t_inj_loc_used));
inferred_parameters(6,index) = mean(X_KF(2,data_start:t_inj_loc_used));
inferred_parameters(7,index) = (n*T*X' - sum(T)*sum(X))/(n*sum(T.^2) - sum(T)^2);

if data_ignore == 0
    data_restart = t_inj_loc_guess + 1;
else
    data_restart = t_inj_loc_guess + data_ignore;
end
