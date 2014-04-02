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
         
% We need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ============================================
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J.
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. We should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively.
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. We need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%
% Part 3: Implement regularization with the cost function and gradients.
%
%

a1=sigmoid(X * Theta1'); %First Activation Layer

a1=[ones(m,1) a1]; % Add Ones as a biased term

a2=sigmoid(a1 * Theta2');% Second Activation Layer

%Map the original labels of y into binary vector on ones and zero
%For example if y=[1 2 1 3]
%than yv= 1 0 0 0 0 
%         0 1 0 0 0  
%         1 0 0 0 0
%         0 0 1 0 0

yv=repmat(1:num_labels,size(y,1),1)==repmat(y,1,num_labels);

%Calcualte the cost function
J1=(1/m)*sum(-yv.*log(a2) - (1-yv).*log(1-a2));

%Sum the cost function for all values of class labels from 1:5
J2=sum(J1);

%Perform Regularization on c
C1=sum(sum(Theta1 .^2));
%Exclude the biased term for regularization
C3=sum(sum(Theta1(:,1).^2));
C2=sum(sum(Theta2 .^2));
C4=sum(sum(Theta2(:,1).^2));
cost=(C1-C3)+(C2-C4);
%Calculate the total cost function with regularization
J=J2 +(lambda/(2*m)).*cost;

%Perform the Gradient Descent

%Error in the actual output layer
delta3=a2-yv;

%Error in the Hidden Layer
r2=delta3 *Theta2(:,2:end);
z2=X *Theta1';
delta2=(r2).*sigmoidGradient(z2);
t1=Theta1 * lambda;
t1(:,1)=0;

%Summation of the gradients calculated
Theta1_grad=(delta2' * X) + t1;
Theta1_grad=Theta1_grad./m;

%Regularizatio of the gradients calcualted and exclude the biased term
t2=Theta2 *lambda;
t2(:,1)=0;
Theta2_grad=(delta3' * a1) + t2;
Theta2_grad=Theta2_grad./m;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
