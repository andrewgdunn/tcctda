function [fault_parameters] = intermittant_statistics_calculation(event_count,signal_parameters,fault_parameters,Z_alpha)
% event_count =   0,1,2,3,4,5,6,7,8, 9,10
% mu_f = column nom,2, ,4, ,6, ,8, ,10...
% mu_n = column nom, ,3, ,5, ,7, ,9,  ,11...
if mod(event_count,2) == 1
    mu_f_end = event_count;
    mu_n_end = event_count - 1;
else
    mu_f_end = event_count - 1;
    mu_n_end = event_count;
end

if event_count > size(signal_parameters,2)
    P_end = size(signal_parameters,2);
else
    P_end = event_count;
end

if size(signal_parameters,2) >= 2
    x_p = [];
    mu_p_sum = 0;
    count_p = 0;
    for pp = 2:2:P_end
        if signal_parameters(3,pp-1) > signal_parameters(5,pp)
            mu_i = (-1)^(pp)*(signal_parameters(4,pp) - signal_parameters(4,pp-1));
        elseif signal_parameters(5,pp-1) < signal_parameters(3,pp)
            mu_i = (-1)^(pp)*(signal_parameters(4,pp) - signal_parameters(4,pp-1));
        end
        mu_p_sum = mu_p_sum + mu_i;
        x_p = [x_p,mu_i];
        count_p = count_p + 1;
    end
    
    mu_p = mu_p_sum/count_p;

    std_p = (sum((x_p - mu_p).^2)/count_p)^.5;

    mu_p_low = -Z_alpha*std_p/count_p^.5 + mu_p;
    mu_p_high = Z_alpha*std_p/count_p^.5 + mu_p;
    
    fault_parameters(3) = mu_p_low;
    fault_parameters(4) = mu_p;
    fault_parameters(5) = mu_p_high;
end



if size(signal_parameters,2) >= 3
    x_f = [];
    mu_f_sum = 0;
    count_f = 0;
    for ff = 2:2:mu_f_end
        mu_i = (signal_parameters(2,ff) - signal_parameters(1,ff));
        mu_f_sum = mu_f_sum + mu_i;
        x_f = [x_f,mu_i];
        count_f = count_f + 1;
    end
    
    mu_f = mu_f_sum/count_f;

    std_f = (sum((x_f - mu_f).^2)/count_f)^.5;

    mu_f_low = -Z_alpha*std_f/count_f^.5 + mu_f;
    mu_f_high = Z_alpha*std_f/count_f^.5 + mu_f;
    
    fault_parameters(9) = mu_f_low;
    fault_parameters(10) = mu_f;
    fault_parameters(11) = mu_f_high;
end


if size(signal_parameters,2) >= 4
    x_n = [];
    mu_n_sum = 0;
    count_n = 0;
    for ff = 3:2:mu_n_end
        mu_i = (signal_parameters(2,ff) - signal_parameters(1,ff));
        mu_n_sum = mu_n_sum + mu_i;
        x_n = [x_n,mu_i];
        count_n = count_n + 1;
    end
    
    mu_n = mu_n_sum/count_n;

    std_n = (sum((x_n - mu_n).^2)/count_n)^.5;

    mu_n_low = -Z_alpha*std_n/count_n^.5 + mu_n;
    mu_n_high = Z_alpha*std_n/count_n^.5 + mu_n;
    
    fault_parameters(12) = mu_n_low;
    fault_parameters(13) = mu_n;
    fault_parameters(14) = mu_n_high;
end

if fault_parameters(10) ~= 0 && fault_parameters(13) ~= 0
    fault_parameters(15) = fault_parameters(3)*fault_parameters(9)/(fault_parameters(9) + fault_parameters(14));
    fault_parameters(16) = fault_parameters(4)*fault_parameters(10)/(fault_parameters(10) + fault_parameters(13));
    fault_parameters(17) = fault_parameters(5)*fault_parameters(11)/(fault_parameters(11) + fault_parameters(12));
end
