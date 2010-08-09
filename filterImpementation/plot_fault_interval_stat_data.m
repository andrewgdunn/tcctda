function [] = plot_fault_interval_stat_data(plot_buffer, plot_color, num_stds, T, mu, mu_L, mu_U, std_hist)

S1 = [plot_color '-'];
S2 = [plot_color '--'];

% t_p = t(plot_buffer:end);
% MU_p = MU(:,plot_buffer:end);
% STD_p = STD(:,plot_buffer:end);
plot_buffer = 1 + plot_buffer;

plot(T(plot_buffer:end),mu(plot_buffer:end),S1,T(plot_buffer:end),mu_L(plot_buffer:end),S2,T(plot_buffer:end),mu_U(plot_buffer:end),S2)

mod1 = 'linewidth';

for sd = 1:num_stds
    eval('plot(T(plot_buffer:end),mu(plot_buffer:end) + sd*std_hist(plot_buffer:end),S2,T(plot_buffer:end),mu(plot_buffer:end) - sd*std_hist(plot_buffer:end),S2,mod1,sd)')
end
