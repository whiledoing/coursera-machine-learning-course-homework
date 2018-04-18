function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);


% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the
%                     partial derivatives w.r.t. to each element of Theta
%

% 使用R这种坐标矩阵获取元素时，最后得到的是一个列向量，而不和原来矩阵维度信息一致。
% 这个好理解，如果还是原矩阵维度，那么不取的元素设置为什么呢，没有意义。
% j_v = (X*Theta')(R) - Y(R);
% J = 0.5*j_v'*j_v;

j_dist = power(X*Theta' - Y, 2);
J = 0.5*sum(sum(j_dist .* R));

for i = 1:num_movies
    idx = find(R(i, :) == 1);
    theta_temp = Theta(idx, :);
    y_temp = Y(i, idx);

    X_grad(i, :) = (X(i, :) * theta_temp' - y_temp) * theta_temp;
end

for j = 1:num_users
    idx = find(R(:, j) == 1);
    x_temp = X(idx, :);
    y_temp = Y(idx, j)';

    Theta_grad(j, :) = (Theta(j, :) * x_temp' - y_temp) * x_temp;
end

# add regularization
J += sum(sum(Theta.*Theta)) * 0.5 * lambda;
J += sum(sum(X.*X)) * 0.5 * lambda;

X_grad += lambda*X;
Theta_grad += lambda*Theta;

% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
