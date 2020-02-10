function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
X=[ones(m,1) X];	
J = 0;
y2= zeros(m,1);
delta_1= zeros(hidden_layer_size, input_layer_size+1);
delta_2= zeros(num_labels,hidden_layer_size+1); 
for i=1: m;
	a2= sigmoid((X(i,:)*(Theta1'))');
	a2=[1; a2];
	h_theta= (sigmoid(Theta2*a2));
	y1= zeros(num_labels, 1);
	y1(y(i,1))=1;
     temp= log(h_theta);
	temp2= y1.*temp;
	temp3= (1-y1).*(log(1-h_theta));
	temp4= sum(temp2+temp3);
	y2(i,1)= temp4;
	delta3= h_theta-y1; 
	delta2= ((Theta2')*delta3).*a2.*(1-a2);
	delta2=delta2(2:end);
	delta_1= delta_1 + delta2*(X(i,:));
	delta_2= delta_2 + delta3*(a2');
	 
end;
J= -(sum(y2)/m); 
	temp6= Theta1.^2;
	temp7=Theta2.^2;
	temp1= sum(sum(temp6(:, 2:size(temp6, 2))));
	temp5= sum(sum(temp7(:, 2:size(temp7, 2)))); 
J= J+ ((lambda*(temp1+temp5))/(2*m));




         
% You need to return the following variables correctly 


Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
Theta1_grad= delta_1/m;
Theta1_grad(:, 2:end)= Theta1_grad(:, 2:end)+ ((lambda*Theta1(:, 2:end))/m);
Theta2_grad= delta_2/m; 
Theta2_grad(:, 2:end)= Theta2_grad(:,2:end)+ ((lambda*Theta2(:, 2:end))/m);
% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
