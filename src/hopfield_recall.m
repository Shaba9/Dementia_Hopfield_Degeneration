
function s = hopfield_recall(W, s0, max_iters)
% HOPFIELD_RECALL  Synchronous update until convergence.
%   s = hopfield_recall(W, s0, max_iters)
%   W: NxN weight matrix, s0: Nx1 initial state (±1), max_iters: int

s = sign(s0); s(s==0) = 1; % ensure ±1
for it = 1:max_iters
    s_new = sign(W * s);
    % break ties: keep previous state where zero
    zero_idx = (s_new == 0);
    s_new(zero_idx) = s(zero_idx);
    if isequal(s_new, s)
        break;
    end
    s = s_new;
end
end
