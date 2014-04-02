function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm1 = X;
X_norm=X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));
m=size(X,1);

    
 %mu=mean(X)
 %for i=1:m,
	%X_norm1(i,:)=(X(i,:)-mu);
 %end;
% sigma=std(X_norm1)
% for j=1:m,
	%X_norm(j,:)=(X_norm1(j,:)./sigma);
 %end;
a=min(X)
b=max(X)
range=b-a
for i=1:m,
	X_norm1(i,:)=(X(i,:)-a);
end;
for j=1:m,
	X_norm(j,:)=(X_norm1(j,:)./range);
end;

% ============================================================

end
