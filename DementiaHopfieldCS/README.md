
# DementiaHopfieldCS (C#)

This is a C# console translation of the Hopfield dementia lab. It:

- Trains a Hopfield autoassociative memory on simple 8×8 letter patterns (A, E, H, O, T, X).
- Evaluates **recall accuracy vs. neuron loss** (0–60%, 5% steps) with multiple random ablation masks and noisy cues.
- Generates **two HTML outputs** in `./outputs/`:
  - `accuracy_vs_loss.html` – an SVG line plot with a gray ±1σ band.
  - `reconstructions_grid.html` – a grid of 8×8 reconstructions at 0%, 20%, 40%, 60% loss.

No external libraries are required; everything uses base .NET and inline HTML/SVG.

## Build & Run (Command Prompt / Terminal)

Prerequisite: .NET 8 SDK (or .NET 6+, adjust TargetFramework in `.csproj` if needed).

```bash
# 1) Navigate to this folder
cd DementiaHopfieldCS

# 2) Build
dotnet build -c Release

# 3) Run
dotnet run -c Release
```

Outputs are written to the `outputs/` folder next to the executable:

- `outputs/accuracy_vs_loss.html`
- `outputs/reconstructions_grid.html`

Open them in any modern web browser.

## Notes

- The program seeds the RNG with `42` for reproducibility.
- Accuracy is computed **only over surviving neurons** at each loss level, matching the MATLAB logic.
- The recall update uses synchronous `sign(W·s)` with tie‑breaking that keeps the previous state.

## What changed vs. MATLAB

- Plots are written as **HTML** (embedded SVG/tables) rather than PNG. This avoids any OS‑specific image libraries and works cross‑platform.
- Visually, black/white cell classes are inverted in CSS to enhance contrast in many dark/light browser themes, but the semantics (−1 = black, +1 = white; 0 = gray) are preserved in the legend.
