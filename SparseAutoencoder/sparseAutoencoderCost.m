function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)

% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 


%Implementing squared error term
m = size(data,2);
%fprintf('m value = %d\n',m);
a{1} = data;
a{2} = sigmoid( bsxfun(@plus,(W1 * a{1}), b1 ) );  % dim [25 x 64] * [64 x 10000] + [25 x 1] = [25 x 10000]
a{3} = sigmoid( bsxfun(@plus,(W2 * a{2}), b2 ) );  % dim [64 x 25] * [25 x 10000] + [64 x 1] = [64 x 10000]
squre_term = (a{3} - data) .^ 2 ;
squre_term_cost = (1/(2 * m)) * sum(squre_term(:));

%fprintf('sparseAutoencoderCost: a{1} size: %d  %d\n', size(a{1},1) , size(a{1},2));
%fprintf('sparseAutoencoderCost: a{2} size: %d  %d\n', size(a{2},1) , size(a{2},2));
%fprintf('sparseAutoencoderCost: W1 size: %d  %d\n', size(W1,1) , size(W1,2));
%fprintf('sparseAutoencoderCost: W2 size: %d  %d\n', size(W2,1) , size(W2,2));
%fprintf('sparseAutoencoderCost: b1 size: %d  %d\n', size(b1,1) , size(b1,2));
%fprintf('sparseAutoencoderCost: b2 size: %d  %d\n', size(b2,1) , size(b2,2));
%fprintf('sparseAutoencoderCost: squre_term_cost: %f\n',squre_term_cost);

%Weight decay
W1_term = W1 .^2;
W1_term = sum(W1_term(:));
W2_term = W2 .^2;
W2_term = sum(W2_term(:));
weight_decay = (lambda/2.0) * (W1_term + W2_term ); 
%weight_decay = (lambda/(2.0 * m)) * (W1_term + W2_term ); 
%fprintf('sparseAutoencoderCost: weight_decay: %f\n',weight_decay);


%Sparsity penalty term
rho = (1.0/m) * (a{2} * ones(m,1)); % dim [25 x 10000] * [10000 x 1] = [25 x 1]
for i = 1:length(rho)
	KL(i) = sparsityParam * log(sparsityParam /rho(i)) + (1 - sparsityParam)* log((1-sparsityParam) /(1 - rho(i)));
	%fprintf('sparseAutoencoderCost: KL(%d): %f\n',i,KL(i));
end
%fprintf('sparseAutoencoderCost: KL sum: %f\n',KL);
%fprintf('sparseAutoencoderCost: rho: %f\n',rho);
sparsity_cost = beta * sum(KL);
%fprintf('sparseAutoencoderCost: sparsity_cost: %f\n',sparsity_cost);


cost = squre_term_cost + weight_decay + sparsity_cost;
%fprintf('sparseAutoencoderCost: cost: %f\n',cost);



%Implementing Backprop
for i = 1: length(rho)
	sparsity_del(i) = beta * ( -(sparsityParam/rho(i)) + (1.0-sparsityParam)/(1.0 - rho(i)) );
end

y = data;

for i = 1:m;
	a1 = data(:,i); % dim [64 x 1]
	z2 = W1 * a1 + b1; % dim [25 x 64] * [64 x 1] = [25 x 1]
	a2 = sigmoid(z2); % dim [25 x 1]
	z3 = W2 * a2 + b2; % dim [64 x 25] * [25 x 1] = [64 * 1]
	a3 = sigmoid(z3); % dim [64 x 1]

	delta3 = (a3 - y(:,i)  ) .* sigmoidGradient(z3); % dim (64 x 1) - [64 x 1] = [64 x 1]
	%delta3 = (a3 - y(:,i));
	delta2 = (W2' * delta3 + sparsity_del') .* sigmoidGradient(z2); % dim [25 x 64] * [64 x 1] = [25 x 1] .* ()
	
	W2grad = W2grad + delta3 * a2'; % dim [64 x 25] + [64 x 1] * [1 x 25] = [64 x 25]
	W1grad = W1grad + delta2 * a1'; % dim [25 x 64] + [25 x 1] * [1 x 64] = [25 x 64]
	b2grad = b2grad + delta3; %[64 x 1]
	b1grad = b1grad + delta2; %[25 x 1]
	
end

W1grad = (1.0/m) * W1grad + lambda * W1;
W2grad = (1.0/m) * W2grad + lambda * W2; 
b1grad = (1.0/m) * b1grad;
b2grad = (1.0/m) * b2grad;


%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end

