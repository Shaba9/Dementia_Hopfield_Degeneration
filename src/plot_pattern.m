
function plot_pattern(x, H, W, use_gray)
% PLOT_PATTERN  Visualize vectorized 2D pattern.
%   plot_pattern(x, H, W) where x is length H*W with values in {-1, 0, +1}
%   If use_gray true, 0 is shown as mid-gray to highlight removed neurons.

if nargin < 4, use_gray = false; end
X = reshape(x, [H, W]);
if use_gray
    % map -1 -> 0 (white), 0 -> 0.5 (gray), +1 -> 1 (black)
    M = zeros(H,W);
    M(X==-1) = 0.0; M(X==0) = 0.5; M(X==1) = 1.0;
    imagesc(M); colormap(gray); caxis([0 1]);
else
    % map -1 -> 0 (white), +1 -> 1 (black)
    M = (X+1)/2; % -1->0, +1->1
    imagesc(M); colormap(gray); caxis([0 1]);
end
axis image off;
end
