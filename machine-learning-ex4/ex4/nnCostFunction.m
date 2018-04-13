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

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

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

% 将数据放到最右边计算时候需要反转，维度变为 (n+1)*m
A_1 = [ones(1,m); X'];
Z_2 = Theta1 * A_1;
A_2 = [ones(1,m); sigmoid(Z_2)];

Z_3 = Theta2 * A_2;
A_3 = sigmoid(Z_3);
h_theta = A_3;

% 这里和logestic不一样，会得到一系列的判定y的标准答案，这里使用y_vector(y(i))=1的方法得到
% 就是表示当前向量只有判定为1的才是有效数据，别的都是0
% 计算的时候需要将每一个标准y和计算出来的A_2数据的一列（一个测试数据）进行计算，然后求和，注意最后不需要除以num_labels
for i = 1:m
    y_vector = zeros(1, num_labels);
    y_vector(y(i)) = 1;

    h_theta_col_i = h_theta(:, i);
    J += -y_vector*log(h_theta_col_i) - (1-y_vector)*log(1-h_theta_col_i);
end

J /= m;

# add regularized cost, 注意不需要计算第一列（bias node）
theta1_no_bias = Theta1(:, 2:end);
theta2_no_bias = Theta2(:, 2:end);

# 点乘的方式计算平方
regularized_v = 0;
regularized_v += sum(sum(theta1_no_bias .* theta1_no_bias));
regularized_v += sum(sum(theta2_no_bias .* theta2_no_bias));
regularized_v *= lambda / (2*m);
J += regularized_v;

% 计算偏导数
for i = 1:m
    y_vector = zeros(num_labels, 1);
    y_vector(y(i)) = 1;

    % 初始化
    theta_3 = h_theta(:,i) - y_vector;

    % 反向计算error。@note Theta2转置的维度是(s_2+1) * s_3，所以计算得到的
    % 前一层数据包含了无效用偏移节点，为让维度匹配，中间结果就去掉了偏移数据
    theta_2 = (Theta2' * theta_3)(2:end) .* sigmoidGradient(Z_2(:,i));

    % 这里维度刚好是匹配的，注意A的维度是包含偏移节点的，所以结果维度是s_3 * (s_2+1)
    Theta2_grad += theta_3 * A_2(:,i)';
    Theta1_grad += theta_2 * A_1(:,i)';
end

Theta1_grad /= m;
Theta2_grad /= m;

# add regularized para @note 同样注意，规约的时候不需要考虑theta的第一列数据
Theta1_grad += [zeros(size(Theta1, 1), 1), Theta1(:, 2:end)] * lambda / m;
Theta2_grad += [zeros(size(Theta2, 1), 1), Theta2(:, 2:end)] * lambda / m;
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
