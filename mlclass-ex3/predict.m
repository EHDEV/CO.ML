function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1)

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

X = [ones(m, 1) X]; % adding X0 = 1 to the X matrix to match theta dimension
thetaX1 = X * Theta1'; 
aSub2 = sigmoid(thetaX1); % a(2) layer 

aSub2 = [ones(m, 1) aSub2]; 
thetaX2 = aSub2 * Theta2';
aSub3 = sigmoid(thetaX2); % a(3) layer

[temp_v, temp_i] = max(aSub3'); % Getting the index of the column with the maximum value.


p = temp_i'; 

% =========================================================================


end
