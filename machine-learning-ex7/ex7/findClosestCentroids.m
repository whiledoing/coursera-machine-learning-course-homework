function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

for i = 1:size(X,1)
    x_vec = X(i, :);

    % 优化的版本，提高显著，尤其是K数值还可以的时候，直接提高了K倍
    centroid_dist = centroids - x_vec;
    [_, min_id] = min(sum(centroid_dist .* centroid_dist, 2));
    idx(i) = min_id;

    %min_dist = inf;
    %for k = 1:K
    %   centroid_dist = centroids(k, :) - x_vec;
    %   dist = centroid_dist*centroid_dist';
    %
    %   if dist < min_dist
    %       min_dist = dist;
    %       idx(i) = k;
    %   end
    %nd
end

% =============================================================

end

