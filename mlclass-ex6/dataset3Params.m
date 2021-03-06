function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
<<<<<<< HEAD
C = .01;
=======
C = 1;
>>>>>>> 0c5580e2ed6079bb05def9a11e4d2eea1f9aac80
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
<<<<<<< HEAD
Xval2 = Xval;
X2 = X;
sigmas = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
Cs = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

#predictions = zeros(size(y), 1)

er_val = zeros(64,3);size(er_val)

k = 1;

for i = 1:size(sigmas, 2)
  for j = 1:size(Cs, 2)    
    model = svmTrain(X, y, Cs(j), @(Xval, Xval2) gaussianKernel(Xval, Xval2, sigmas(i))); 
    predictions = svmPredict(model, Xval);
fprintf("error, sigma, C...");
    er_val(k, 3) = mean(double(predictions ~= yval)); er_val(k,3)
 % =========================================================================
    er_val(k,1) = sigmas(i);er_val(k,1)
    er_val(k,2) = Cs(j);er_val(k,2)
    
    k+=1
   end
end

[val, idx] = min(er_val(:,3), [], 1)

sigma = er_val(idx, 1)
C = er_val(idx, 2)

end
=======







% =========================================================================

end
>>>>>>> 0c5580e2ed6079bb05def9a11e4d2eea1f9aac80
