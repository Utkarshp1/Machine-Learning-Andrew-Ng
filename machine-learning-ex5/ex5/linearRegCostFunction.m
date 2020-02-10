function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
temp=sum(((X*theta)-y).^2);
temp1= lambda*(sum(theta.^2)- (theta(1,1)^2));
J= ((temp + temp1)/(2*m));
grad = zeros(size(theta));
temp2=sum(((X*theta)-y).*X);
temp3= lambda*(theta);
grad= ((temp2')+temp3)/m;
grad(1,1)= temp2(1,1)/m;
 

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%












% =========================================================================

end
