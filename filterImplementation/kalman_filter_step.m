function [xk_k,pk_k] = kalman_filter_step(xk1_k1,pk1_k1,F,H,Q,R,Z_hist)
% Single update step of a kalman filter of arbitrary state space dimension
% Calculate intermediate priors
xk_k1 = F*xk1_k1;
pk_k1 = F*pk1_k1*F' + Q;

% Calculate updated values of state and prior probability
z1 = Z_hist - H*xk_k1;
S1 = H*pk_k1*H' + R;
K1 = pk_k1*H'*S1^-1;
xk_k = xk_k1 + K1*z1;
pk_k = (eye(size(F)) - K1*H)*pk_k1;