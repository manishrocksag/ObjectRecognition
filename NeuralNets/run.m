
function [p,nn_params,Theta1,Theta2] = run(X,input_layer_size,hidden_layer_size,num_labels,y)

%%This function takes the input arguments and performs a series of computation to return optimal weights



%% Randomly Initialize the initial weights to be feed into the network.
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

%Set the number of iterations to perform gradient descent
options = optimset('MaxIter', 1500);

%Set the values of lambda as a regularization parameter.For training set we set it 0 and after validating on Validation Set
%we found its optimal value to  be .4
lambda = .4;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
%fmincg is a stantard function which takes cost function and gradient as a parameter and minimizes the cost function to obtain
%optimal weights and store them in nn_params.
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);


%We need to convert the nn_params vector into matrix of weight for input layer and hidden layer to perform predictions
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

%Use predict function to get make predictions for the labels of the training set
pred = predict(Theta1, Theta2, X);

%Check the accuracy of the predictions made 
p=mean(double(pred == y)) * 100


% =========================================================================


end