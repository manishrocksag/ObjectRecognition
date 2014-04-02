

%  Instructions
%  ------------
% 
%  This file contains code and instructions for getting started.
%  
%

%% Initialization
clear ; close all; clc

%% Setup the parameters we will use for this task
input_layer_size  = 240;  %240 input vector size
hidden_layer_size = 50;   % 50 hidden units
num_labels = 5;          % 5 labels, from 1 to 5   

%%Read the feature vector file into a matrix.Original Feature vector file
%%is filtered and first column containing labels for images is removed.

X=dlmread('trainingSet.txt'); % A matrix of 500*240
                        

m = size(X, 1); % output expected=500

% Perform feature Normalization using standard matlab function zscore
X=zscore(X);

%Add a biased term to all the training samples
X=[ones(m,1) X]; % A matrix of 500 * 241

%load the labels for the training set
y=dlmread('trainingSetLabels.txt');


%Run the Neural Network Program

%Returns the prediction accuracy and optimal weights for the Neural
%Network.Here in this case the number of iterations to be run and the
%regularization parameter has been fixed in the function run by evaluating
%it before hand on the validation set.

[p,nn_params,Theta1,Theta2] = run(X,input_layer_size,hidden_layer_size,num_labels,y);

%The value of p should be between 95 to 100 % and we have optimal weights
%stored in Theta1 and Theta2 to make predictions.



%% ================ Load the test set ================
test_set=dlmread('testSet.txt');

%Perform feature normalization on test set

test_set=zscore(test_set);
m=size(test_set,1);

%Add the biased term
test_set=[ones(m,1) test_set];




%% ================= Part 10: Implement Predict =================
%  After training the neural network, we would like to use it to predict
%  the labels. We will now implement the "predict" function to use the
%  neural network to predict the labels of the test set.

pred = predict(Theta1, Theta2, test_set);

%Save the predictions made

fid=fopen('output.txt','wt');
fprintf(fid,'%d \n',pred);
fclose(fid);

%Append the labels along with predictions.Done manually.The final file for 
%submission is submission.txt






