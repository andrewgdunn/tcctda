function [start_ii,count_ii,cumsum_ii,mu_x_ii,mu_x_low_ii,mu_x_high_ii,std_x_ii] = ...
    Online_statistics(interval_restart,ignore_buffer,ii,start_iim1,count_iim1,cumsum_iim1,x_hist,Z_alpha)


% Calculate the count and cumulative sum to calculate statistical values online
if interval_restart == 1
    % For the first point of a sensor data interval
    start_ii = ii;
    count_ii = 1;
    cumsum_ii = x_hist(ii);
else
    % For any other point along a sensor data interval
    start_ii = start_iim1;
    count_ii = count_iim1 + 1; 
    cumsum_ii = x_hist(ii) + cumsum_iim1;
end

% if ii < start_ii + ignore_buffer
%     mu_x_ii = 0;
% else
%     mu_x_ii = cumsum_ii/(count_ii-ignore_buffer);
% end

% Online average
mu_x_ii = cumsum_ii/count_ii;

% X data points for standard deviation calculation
x_sample = x_hist(start_ii:ii);

% Vectorized standard deviation calculation
std_x_ii = (sum((x_sample - mu_x_ii).^2)/count_ii)^.5;

% Online average confidence interval calculation
mu_x_low_ii = -Z_alpha*std_x_ii/count_ii^.5 + mu_x_ii;
mu_x_high_ii = Z_alpha*std_x_ii/count_ii^.5 + mu_x_ii;