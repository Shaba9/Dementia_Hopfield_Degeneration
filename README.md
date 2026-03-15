
# Dementia Simulation via Progressive Neuron Loss (Hopfield Network) — MATLAB

This project models dementia-related memory degradation using a **Hopfield associative memory**. 
We store simple binary patterns (8×8 letters), then **progressively remove neurons** to simulate 
neuronal loss and measure recall performance. The recall accuracy declines as more neurons are lost, illustrating dementia-like deterioration.

## Why a Hopfield Network?
Hopfield networks are classic **content-addressable memory** models. They store patterns via 
Hebbian-like learning and retrieve them from partial/noisy cues. Removing neurons approximates 
progressive neurodegeneration (e.g., synaptic/neuronal loss in dementia), producing degraded recall.

## What you get
- `main.m` — runs the full experiment end-to-end
- Plots:
  - Accuracy vs. neuron loss
  - Visual reconstructions at different loss levels (0%, 20%, 40%, 60%)
- No toolboxes required — **pure MATLAB/Octave-compatible code**

---

## Setup (VS Code + MATLAB)
1. **Install MATLAB** (R2018b+ recommended; no toolboxes required). 
2. **Install VS Code** and the **"MATLAB" extension by MathWorks** for syntax highlighting and Run commands.
3. **Open this folder in VS Code** (`File → Open Folder… → select \`Dementia_Hopfield_Degeneration_MATLAB\``).
4. In VS Code, open `src/main.m` and run it:
   - Either click **Run** (if configured), or
   - Use Command Palette: **MATLAB: Run File**, or
   - From MATLAB command window: `cd` into this folder and run:
     ```matlab
     addpath('src');
     main;
     ```

**Outputs** (figures) are written to the `outputs/` folder.

---

## Project Structure
```
Dementia_Hopfield_Degeneration_MATLAB/
  ├─ src/
  │   ├─ main.m
  │   ├─ define_patterns.m
  │   ├─ hopfield_train.m
  │   ├─ hopfield_recall.m
  │   ├─ plot_pattern.m
  ├─ outputs/    (auto-created for figures)
  └─ README.md
```

---

## Experiment Overview
- **Patterns**: 6 letters (A, E, H, O, T, X), each 8×8, encoded to ±1.
- **Training**: Hebbian rule, zero diagonal. 
- **Cue noise**: 10% bits flipped before recall.
- **Neuron loss**: remove a fraction of neurons (0% → 60%).
- **Metric**: Recall accuracy (fraction of matching bits) **computed on surviving neurons**.
- **Trials**: averaged over multiple random ablation masks and cues.

---

## Notes & Tips
- The recall procedure uses **synchronous sign updates** until convergence or a max-iteration cap.
- If you want to simulate **synapse loss instead of neuron loss**, zero out a fraction of the weights in `W` instead of removing rows/cols.
- You can tweak: number of patterns, cue noise level, loss schedule, and max iterations.

---

## Report Overview
> We implemented an associative memory (Hopfield network) for recalling stored binary patterns. 
> Progressive neuronal loss was simulated by randomly removing a fraction of network nodes and 
> evaluating recall accuracy on surviving neurons. Accuracy declined monotonically with increased loss, 
> visually demonstrating dementia-like degradation in memory retrieval.

