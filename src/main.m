function main()
% Dementia Simulation Using Hopfield Network with Progressive Neuron Loss
% Updated for MATLAB Online:
% - writes figures to ./outputs (writable in MATLAB Drive)
% - uses exportgraphics with fallback to saveas
% - avoids parent-folder paths ('..')
%
% No toolboxes required. Compatible with MATLAB/Octave.

rng(42); % for reproducibility

% --- Hyperparameters ---
noise_flip_frac = 0.10;        % fraction of bits flipped in the cue
loss_fracs       = 0:0.05:0.60; % neuron loss fractions to evaluate
n_loss_trials    = 15;          % random ablation masks per loss level
max_iters        = 50;          % max recall iterations

% --- Writable outputs directory inside current folder (MATLAB Drive) ---
outdir = fullfile(pwd, 'outputs');
if ~isfolder(outdir)
    try
        mkdir(outdir);
    catch
        % no-op if already created by a parallel run or permission glitch
    end
end

% --- Load patterns (P patterns of size N) ---
[X, names, H, W] = define_patterns();  % X: N x P, values in {-1,+1}
[N, P] = size(X);

% --- Train Hopfield network ---
Wmat = hopfield_train(X, 'normalize', true);

% --- Baseline: check recall without neuron loss ---
base_acc = zeros(P,1);
for p = 1:P
    x_clean = X(:,p);
    x_cue   = flip_bits(x_clean, noise_flip_frac);
    x_rec   = hopfield_recall(Wmat, x_cue, max_iters);
    base_acc(p) = mean(x_rec == x_clean);
end
fprintf('Baseline recall accuracy (mean over patterns): %.3f\n', mean(base_acc));

% --- Evaluate progressive neuron loss ---
mean_acc = zeros(numel(loss_fracs),1);
std_acc  = zeros(numel(loss_fracs),1);

for i = 1:numel(loss_fracs)
    frac = loss_fracs(i);
    accs = zeros(n_loss_trials*P,1);
    k = 1;
    for t = 1:n_loss_trials
        % choose neurons to remove
        n_remove = round(frac * N);
        if n_remove > 0
            idx_remove = sort(randperm(N, n_remove));
        else
            idx_remove = [];
        end
        idx_keep   = setdiff(1:N, idx_remove);

        % Build reduced network W_sub
        W_sub = Wmat(idx_keep, idx_keep);

        for p = 1:P
            x_clean = X(:,p);
            x_cue   = flip_bits(x_clean, noise_flip_frac);

            % Reduce cue to surviving neurons
            x_cue_sub   = x_cue(idx_keep);
            x_clean_sub = x_clean(idx_keep);

            % Recall on subnetwork
            x_rec_sub = hopfield_recall(W_sub, x_cue_sub, max_iters);

            % Accuracy computed on surviving neurons only
            accs(k) = mean(x_rec_sub == x_clean_sub);
            k = k + 1;
        end
    end
    mean_acc(i) = mean(accs);
    std_acc(i)  = std(accs);
    fprintf('Loss %d%% -> mean acc = %.3f\n', round(frac*100), mean_acc(i));
end

% --- Plot accuracy vs neuron loss ---
figure('Color','w');
errorbar(100*loss_fracs, mean_acc, std_acc, 'o-','LineWidth',1.5,'MarkerSize',6);
xlabel('Neuron loss (%)'); ylabel('Recall accuracy (surviving neurons)');
title('Hopfield Recall Accuracy vs. Progressive Neuron Loss'); grid on;
ylim([0 1]);
drawnow;

% Save figure (MATLAB Online-friendly)
accFig  = gcf;
accPath = fullfile(outdir, 'accuracy_vs_loss.png');
try
    exportgraphics(accFig, accPath, 'Resolution', 200);
catch
    saveas(accFig, accPath);
end

% --- Visual reconstructions at selected loss levels ---
show_fracs   = [0.00 0.20 0.40 0.60];
pat_to_show  = 1; % show reconstructions for first pattern (e.g., 'A')

figure('Color','w');
rows = 2; cols = numel(show_fracs)+1; % original + loss levels

% Original and noisy cue
x_clean = X(:,pat_to_show);
x_cue   = flip_bits(x_clean, noise_flip_frac);

subplot(rows, cols, 1); plot_pattern(x_clean, H, W);
title(sprintf('Original (%s)', names{pat_to_show}));

subplot(rows, cols, 1+cols); plot_pattern(x_cue, H, W);
title(sprintf('Noisy cue (flip %.0f%%)', 100*noise_flip_frac));

for j = 1:numel(show_fracs)
    frac = show_fracs(j);
    n_remove = round(frac * N);
    if n_remove > 0
        idx_remove = sort(randperm(N, n_remove));
    else
        idx_remove = [];
    end
    idx_keep   = setdiff(1:N, idx_remove);

    W_sub = Wmat(idx_keep, idx_keep);

    x_cue_sub   = x_cue(idx_keep);
    x_clean_sub = x_clean(idx_keep);

    x_rec_sub = hopfield_recall(W_sub, x_cue_sub, max_iters);

    % Rebuild to full size for visualization: fill removed neurons with 0 (gray)
    x_vis = zeros(N,1);
    x_vis(idx_keep) = x_rec_sub;
    if ~isempty(idx_remove), x_vis(idx_remove) = 0; end

    subplot(rows, cols, 1+j); plot_pattern(x_vis, H, W, true);
    title(sprintf('Loss %d%%', round(100*frac)));

    % Also show accuracy on surviving neurons in bottom row
    acc = mean(x_rec_sub == x_clean_sub);
    subplot(rows, cols, 1+cols+j); plot_pattern(sign(x_vis + eps), H, W, true);
    title(sprintf('Recon (acc=%.2f)', acc));
end
drawnow;

% Save reconstruction grid
reconFig  = gcf;
reconPath = fullfile(outdir, 'reconstructions_grid.png');
try
    exportgraphics(reconFig, reconPath, 'Resolution', 200);
catch
    saveas(reconFig, reconPath);
end

fprintf('Done. Figures written to %s\n', outdir);
end

% --- Helpers ---
function x_noisy = flip_bits(x, frac)
% Flip the sign of a fraction of entries in x (values are ±1).
N = numel(x);
n_flip = round(frac * N);
if n_flip==0
    x_noisy = x; 
    return;
end
idx = randperm(N, n_flip);
x_noisy = x;
x_noisy(idx) = -x_noisy(idx);
end