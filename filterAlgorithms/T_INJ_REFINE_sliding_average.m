function [t_inj_loc_guess] = T_INJ_REFINE_sliding_average(mean_buffer,t_interval,DATA)

N = length(t_interval);
N_eff = N - 2*mean_buffer;
averages = zeros(2,N_eff);

for i = 1:N_eff
    data_top = DATA(t_interval(1:mean_buffer+i));
    data_bot = DATA(t_interval(i+mean_buffer:N));
    averages(1,i) = mean(data_top);
    averages(2,i) = mean(data_bot);
end

A = t_interval(1:end-1);
B = [zeros(1,mean_buffer),averages(1,1:end-1),zeros(1,mean_buffer)];
C = [zeros(1,mean_buffer),averages(2,2:end),zeros(1,mean_buffer)];
D = abs(B-C);


[Z,I_Z] = max(D);
t_inj_loc_guess = t_interval(I_Z);