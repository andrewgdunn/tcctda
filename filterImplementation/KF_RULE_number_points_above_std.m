function [jump_flag] = KF_RULE_number_points_above_std(tt,history,MU,STD,N_points,num_stds,X_KF)
% Rule for testing whether or not a number of points are outside a standard devation from the mean envelope of the
% statistical data
data = X_KF(1,tt-history:tt);
upper_ref = MU + num_stds*STD;
lower_ref = MU - num_stds*STD;

if sum(data > upper_ref) >= N_points || sum(data < lower_ref) >= N_points
    jump_flag = 1;
else
    jump_flag = 0;
end