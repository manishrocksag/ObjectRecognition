

function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval)

m = size(X, 1);

% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);
input_layer_size=240;
hidden_layer_size=50;
num_labels = 5;

% ====================== YOUR CODE HERE ======================
for i = 1:m
	a=X(1:i,:);
	b=y(1:i);
    
	[p,nn_params,Theta1,Theta2]=run(a,input_layer_size,hidden_layer_size,num_labels,b);
	error_train(i)=CostFunction(a,b,nn_params,0);
	error_val(i)=CostFunction(Xval,yval,nn_params,0);
	
end;      


