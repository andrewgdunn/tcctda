function [R] = R_correlation(X,Y)
% Vectorized formula for calculating the correlation coefficient [-1 <= p <= 1] from a data set (X,Y)
n = length(X);
R = (n*X*Y' - sum(X)*sum(Y))/((n*sum(X.^2) - sum(X)^2)^.5*(n*sum(Y.^2) - sum(Y)^2)^.5);