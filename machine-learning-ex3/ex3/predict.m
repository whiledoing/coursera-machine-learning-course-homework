function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

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

% 将数据放到最右边计算时候需要反转，维度变为 (n+1)*m
X = [ones(1, m); X'];

% 计算下一层的数据也记得加上a_0数据，注意这里直接放到一行上了
A_1 = sigmoid(Theta1 * X);
A_1 = [ones(1, m); A_1];
A_2 = sigmoid(Theta2 * A_1);

% 得到下标的列形式
[max_p, p] = max(A_2);
p = p';








% =========================================================================


end
