function [mu_x,mu_x_low,mu_x_high,std_x] = ...
    Online_statistics_periodic(sample_period,ii,x_hist,Z_alpha)

x_i = x_hist(ii-sample_period:ii);

mu_x = sum(x_i)/sample_period;

std_x = (sum((x_i - mu_x).^2)/sample_period)^.5;

mu_x_low = -Z_alpha*std_x/sample_period^.5 + mu_x;
mu_x_high = Z_alpha*std_x/sample_period^.5 + mu_x;