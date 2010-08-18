function [slope,intercept] = linear_regression(X,Y)
% Vectorized formula for calculating the slope and y-intercept of a data set (X,Y)
n = length(X);
slope = (n*X*Y' - sum(X)*sum(Y))/(n*sum(X.^2) - sum(X)^2);
intercept = (sum(Y) - slope*sum(X))/n;