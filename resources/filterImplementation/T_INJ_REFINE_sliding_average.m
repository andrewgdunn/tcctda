function [t_inj_loc_guess] = T_INJ_REFINE_sliding_average(mean_buffer,t_interval,DATA)

% Total interval lenght
N = length(t_interval);
% Chop off buffer ends to use to weight the two averages
N_eff = N - 2*mean_buffer;
% Value storage array
averages = zeros(2,N_eff);

% Calculate the sliding average from either side of a midpoint moving left to right across the interval
for i = 1:N_eff
    data_top = DATA(t_interval(1:mean_buffer+i));
    data_bot = DATA(t_interval(i+mean_buffer:N));
    averages(1,i) = mean(data_top);
    averages(2,i) = mean(data_bot);
end

% Manipulate the sliding average values, finding the difference between each side at each point
A = t_interval(1:end-1);
B = [zeros(1,mean_buffer),averages(1,1:end-1),zeros(1,mean_buffer)];
C = [zeros(1,mean_buffer),averages(2,2:end),zeros(1,mean_buffer)];
D = abs(B-C);

% The injection location is the point that maximizes the difference in the two averages
[Z,I_Z] = max(D);
% Use the index to extract t_inj_loc value of the interval
t_inj_loc_guess = t_interval(I_Z);