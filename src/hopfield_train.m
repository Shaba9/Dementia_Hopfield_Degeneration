
function W = hopfield_train(X, varargin)
% HOPFIELD_TRAIN  Train Hopfield network by Hebbian learning.
%   W = hopfield_train(X) where X is N x P (patterns in columns), values in {-1,+1}
%   Optional name-value: 'normalize' (true/false) — if true, divide by N.

p = inputParser; p.addParameter('normalize', true, @(b)islogical(b)||ismember(b,[0 1])); p.parse(varargin{:});
normalize = p.Results.normalize;

[N, P] = size(X);
W = zeros(N,N);
for mu = 1:P
    x = X(:,mu);
    W = W + (x * x.');
end
W(1:N+1:end) = 0;  % zero diagonal (no self-connection)
if normalize
    W = W / N;     % optional normalization for stability
end
end
