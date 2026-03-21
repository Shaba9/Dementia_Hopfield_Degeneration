
using System;
using System.IO;
using System.Linq;
using System.Text;
using System.Globalization;

namespace DementiaHopfieldCS
{
    class Program
    {
        static void Main(string[] args)
        {
            // Reproducibility
            var rng = new Random(42);

            // Hyperparameters
            double noiseFlipFrac = 0.10;            // cue corruption
            int maxIters = 50;                      // recall iterations
            double[] lossFracs = Enumerable.Range(0, 13).Select(i => i * 0.05).ToArray(); // 0..0.60
            int nLossTrials = 15;

            // Outputs directory
            string outdir = Path.Combine(Directory.GetCurrentDirectory(), "outputs");
            Directory.CreateDirectory(outdir);

            // Patterns
            int H, W;
            var X = Patterns.DefinePatterns(out H, out W); // columns are patterns; values +/-1
            int N = X.GetLength(0);
            int P = X.GetLength(1);

            // Train Hopfield
            var Wmat = Hopfield.Train(X, normalize: true);

            // Baseline accuracy (no neuron loss)
            double[] baseAcc = new double[P];
            for (int p = 0; p < P; p++)
            {
                var xClean = GetColumn(X, p);
                var xCue = Utils.FlipBits(xClean, noiseFlipFrac, rng);
                var xRec = Hopfield.Recall(Wmat, xCue, maxIters);
                baseAcc[p] = Utils.Accuracy(xRec, xClean);
            }
            double baseline = baseAcc.Average();
            Console.WriteLine($"Baseline recall accuracy (mean over patterns): {baseline:F3}");

            // Accuracy vs neuron loss
            double[] meanAcc = new double[lossFracs.Length];
            double[] stdAcc = new double[lossFracs.Length];

            for (int i = 0; i < lossFracs.Length; i++)
            {
                double frac = lossFracs[i];
                int nRemove = (int)Math.Round(frac * N);
                var accs = new double[nLossTrials * P];
                int k = 0;
                for (int t = 0; t < nLossTrials; t++)
                {
                    var idxRemove = Utils.RandomUniqueIndices(N, nRemove, rng);
                    var idxKeep = Enumerable.Range(0, N).Except(idxRemove).ToArray();

                    var Wsub = Utils.Submatrix(Wmat, idxKeep);

                    for (int p = 0; p < P; p++)
                    {
                        var xClean = GetColumn(X, p);
                        var xCue = Utils.FlipBits(xClean, noiseFlipFrac, rng);
                        var xCleanSub = Utils.Subvector(xClean, idxKeep);
                        var xCueSub = Utils.Subvector(xCue, idxKeep);

                        var xRecSub = Hopfield.Recall(Wsub, xCueSub, maxIters);
                        accs[k++] = Utils.Accuracy(xRecSub, xCleanSub);
                    }
                }
                meanAcc[i] = accs.Average();
                stdAcc[i] = Utils.Std(accs);
                Console.WriteLine($"Loss {Math.Round(frac*100)}% -> mean acc = {meanAcc[i]:F3}");
            }

            // Write HTML: accuracy_vs_loss.html (with SVG)
            string accHtml = HtmlWriter.AccuracyHtml(lossFracs.Select(x=>x*100).ToArray(), meanAcc, stdAcc, baseline);
            File.WriteAllText(Path.Combine(outdir, "accuracy_vs_loss.html"), accHtml);

            // Reconstructions grid at selected losses
            double[] showFracs = new double[] { 0.00, 0.20, 0.40, 0.60 };
            int patToShow = 0; // first pattern (A)

            // Build panels: original, noisy cue, then for each loss produce visualization and accuracy
            var xCleanPat = GetColumn(X, patToShow);
            var xCuePat = Utils.FlipBits(xCleanPat, noiseFlipFrac, rng);

            var panels = new HtmlWriter.ReconPanel[showFracs.Length + 2];
            // original
            panels[0] = new HtmlWriter.ReconPanel
            {
                Title = $"Original ({Patterns.Names[patToShow]})",
                Grid = xCleanPat.Select(v => v).ToArray(), // +/-1
                H = H, W = W
            };
            // noisy cue
            panels[1] = new HtmlWriter.ReconPanel
            {
                Title = $"Noisy cue (flip {Math.Round(noiseFlipFrac*100)}%)",
                Grid = xCuePat.Select(v => v).ToArray(),
                H = H, W = W
            };

            for (int j = 0; j < showFracs.Length; j++)
            {
                double frac = showFracs[j];
                int nRemove = (int)Math.Round(frac * N);
                var idxRemove = Utils.RandomUniqueIndices(N, nRemove, rng);
                var idxKeep = Enumerable.Range(0, N).Except(idxRemove).ToArray();

                var Wsub = Utils.Submatrix(Wmat, idxKeep);
                var xCueSub = Utils.Subvector(xCuePat, idxKeep);
                var xCleanSub = Utils.Subvector(xCleanPat, idxKeep);
                var xRecSub = Hopfield.Recall(Wsub, xCueSub, maxIters);
                double acc = Utils.Accuracy(xRecSub, xCleanSub);

                // For visualization, create a full-size vector where removed indices are 0 (gray), kept are xRecSub
                int[] xVis = new int[N];
                for (int ii = 0; ii < N; ii++) xVis[ii] = 0; // gray
                for (int kk = 0; kk < idxKeep.Length; kk++) xVis[idxKeep[kk]] = xRecSub[kk];

                panels[j + 2] = new HtmlWriter.ReconPanel
                {
                    Title = $"Loss {Math.Round(frac*100)}% (acc={acc:F2})",
                    Grid = xVis,
                    H = H, W = W
                };
            }

            string reconHtml = HtmlWriter.ReconstructionHtml(panels);
            File.WriteAllText(Path.Combine(outdir, "reconstructions_grid.html"), reconHtml);

            Console.WriteLine($"Done. HTML files written to {outdir}");
        }

        static int[] GetColumn(int[,] X, int col)
        {
            int n = X.GetLength(0);
            var v = new int[n];
            for (int i = 0; i < n; i++) v[i] = X[i, col];
            return v;
        }
    }

    static class Hopfield
    {
        public static double[,] Train(int[,] X, bool normalize)
        {
            int N = X.GetLength(0);
            int P = X.GetLength(1);
            var W = new double[N, N];
            for (int mu = 0; mu < P; mu++)
            {
                for (int i = 0; i < N; i++)
                {
                    int xi = X[i, mu];
                    for (int j = 0; j < N; j++)
                    {
                        if (i == j) continue; // zero diag as we go
                        int xj = X[j, mu];
                        W[i, j] += xi * xj;
                    }
                }
            }
            if (normalize)
            {
                double scale = 1.0 / N;
                for (int i = 0; i < N; i++)
                    for (int j = 0; j < N; j++)
                        W[i, j] *= scale;
            }
            // already zero diag
            return W;
        }

        public static int[] Recall(double[,] W, int[] s0, int maxIters)
        {
            int N = s0.Length;
            var s = new int[N];
            for (int i = 0; i < N; i++) s[i] = s0[i] >= 0 ? 1 : -1; // ensure +/-1

            for (int it = 0; it < maxIters; it++)
            {
                var sNew = new int[N];
                for (int i = 0; i < N; i++)
                {
                    double h = 0;
                    for (int j = 0; j < N; j++)
                    {
                        h += W[i, j] * s[j];
                    }
                    if (h > 0) sNew[i] = 1;
                    else if (h < 0) sNew[i] = -1;
                    else sNew[i] = s[i]; // tie -> keep previous
                }
                bool same = true;
                for (int i = 0; i < N; i++) if (sNew[i] != s[i]) { same = false; break; }
                s = sNew;
                if (same) break;
            }
            return s;
        }
    }

    static class Patterns
    {
        public static readonly string[] Names = new[] { "A", "E", "H", "O", "T", "X" };

        public static int[,] DefinePatterns(out int H, out int W)
        {
            H = 8; W = 8;
            int[,] A = new int[,]
            {
                {0,1,1,1,1,1,1,0},
                {0,1,0,0,0,0,1,0},
                {0,1,0,0,0,0,1,0},
                {0,1,1,1,1,1,1,0},
                {0,1,0,0,0,0,1,0},
                {0,1,0,0,0,0,1,0},
                {0,1,0,0,0,0,1,0},
                {0,1,0,0,0,0,1,0}
            };
            int[,] E = new int[,]
            {
                {1,1,1,1,1,1,1,1},
                {1,0,0,0,0,0,0,0},
                {1,0,0,0,0,0,0,0},
                {1,1,1,1,1,0,0,0},
                {1,0,0,0,0,0,0,0},
                {1,0,0,0,0,0,0,0},
                {1,0,0,0,0,0,0,0},
                {1,1,1,1,1,1,1,1}
            };
            int[,] Hh = new int[,]
            {
                {1,0,0,0,0,0,0,1},
                {1,0,0,0,0,0,0,1},
                {1,0,0,0,0,0,0,1},
                {1,1,1,1,1,1,1,1},
                {1,0,0,0,0,0,0,1},
                {1,0,0,0,0,0,0,1},
                {1,0,0,0,0,0,0,1},
                {1,0,0,0,0,0,0,1}
            };
            int[,] O = new int[,]
            {
                {0,1,1,1,1,1,1,0},
                {1,0,0,0,0,0,0,1},
                {1,0,0,0,0,0,0,1},
                {1,0,0,0,0,0,0,1},
                {1,0,0,0,0,0,0,1},
                {1,0,0,0,0,0,0,1},
                {1,0,0,0,0,0,0,1},
                {0,1,1,1,1,1,1,0}
            };
            int[,] T = new int[,]
            {
                {1,1,1,1,1,1,1,1},
                {0,0,0,1,1,0,0,0},
                {0,0,0,1,1,0,0,0},
                {0,0,0,1,1,0,0,0},
                {0,0,0,1,1,0,0,0},
                {0,0,0,1,1,0,0,0},
                {0,0,0,1,1,0,0,0},
                {0,0,0,1,1,0,0,0}
            };
            int[,] Xx = new int[,]
            {
                {1,0,0,0,0,0,0,1},
                {0,1,0,0,0,0,1,0},
                {0,0,1,0,0,1,0,0},
                {0,0,0,1,1,0,0,0},
                {0,0,0,1,1,0,0,0},
                {0,0,1,0,0,1,0,0},
                {0,1,0,0,0,0,1,0},
                {1,0,0,0,0,0,0,1}
            };

            // Convert 0/1 to -1/+1, then vectorize columns
            int[][] mats = new[] { A, E, Hh, O, T, Xx }.
                Select(m => ToPMMatrix(m)).ToArray();

            int N = H * W;
            int P = mats.Length;
            var X = new int[N, P];
            for (int p = 0; p < P; p++)
            {
                int idx = 0;
                for (int r = 0; r < H; r++)
                for (int c = 0; c < W; c++)
                {
                    X[idx++, p] = mats[p][r*W + c];
                }
            }
            return X;
        }

        static int[] ToPMMatrix(int[,] bin)
        {
            int H = bin.GetLength(0);
            int W = bin.GetLength(1);
            var v = new int[H*W];
            int k = 0;
            for (int r = 0; r < H; r++)
                for (int c = 0; c < W; c++)
                    v[k++] = bin[r, c] == 1 ? 1 : -1; // 1->+1, 0->-1
            return v;
        }
    }

    static class Utils
    {
        public static int[] FlipBits(int[] x, double frac, Random rng)
        {
            int N = x.Length;
            int nFlip = (int)Math.Round(frac * N);
            var idx = RandomUniqueIndices(N, nFlip, rng);
            var y = (int[])x.Clone();
            foreach (var i in idx) y[i] = -y[i];
            return y;
        }

        public static int[] RandomUniqueIndices(int N, int k, Random rng)
        {
            if (k <= 0) return Array.Empty<int>();
            if (k >= N) return Enumerable.Range(0, N).ToArray();
            var arr = Enumerable.Range(0, N).ToArray();
            // Fisher-Yates shuffle first k
            for (int i = 0; i < k; i++)
            {
                int j = rng.Next(i, N);
                (arr[i], arr[j]) = (arr[j], arr[i]);
            }
            Array.Resize(ref arr, k);
            Array.Sort(arr);
            return arr;
        }

        public static double[,] Submatrix(double[,] W, int[] idx)
        {
            int n = idx.Length;
            var S = new double[n, n];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    S[i, j] = W[idx[i], idx[j]];
            return S;
        }

        public static int[] Subvector(int[] v, int[] idx)
        {
            var u = new int[idx.Length];
            for (int i = 0; i < idx.Length; i++) u[i] = v[idx[i]];
            return u;
        }

        public static double Accuracy(int[] a, int[] b)
        {
            int n = a.Length;
            int correct = 0;
            for (int i = 0; i < n; i++) if (a[i] == b[i]) correct++;
            return (double)correct / n;
        }

        public static double Std(double[] data)
        {
            double mean = data.Average();
            double var = data.Select(x => (x - mean) * (x - mean)).Average();
            return Math.Sqrt(var);
        }
    }

    static class HtmlWriter
    {
        public static string AccuracyHtml(double[] xLossPct, double[] meanAcc, double[] stdAcc, double baseline)
        {
            // Build an SVG line plot with shaded std band
            int width = 900, height = 450, margin = 60;
            double xmin = xLossPct.Min(), xmax = xLossPct.Max();
            double ymin = 0, ymax = 1;

            Func<double,double> sx = v => margin + (v - xmin) / (xmax - xmin) * (width - 2*margin);
            Func<double,double> sy = v => height - margin - (v - ymin) / (ymax - ymin) * (height - 2*margin);

            // Build band polygon
            var upper = meanAcc.Zip(stdAcc, (m,s) => Math.Min(1.0, m + s)).ToArray();
            var lower = meanAcc.Zip(stdAcc, (m,s) => Math.Max(0.0, m - s)).ToArray();

            var sb = new StringBuilder();
            sb.AppendLine("<!DOCTYPE html><html><head><meta charset='utf-8'><title>Accuracy vs Neuron Loss</title>");
            sb.AppendLine("<style>body{font-family:Segoe UI,Arial,sans-serif;} .axis text{font-size:12px} .title{font-size:18px;font-weight:600;margin:10px 0}</style></head><body>");
            sb.AppendLine("<div class='title'>Hopfield Recall Accuracy vs. Progressive Neuron Loss</div>");
            sb.AppendLine($"<p>Baseline mean accuracy (no loss): {baseline:F3}</p>");
            sb.AppendLine($"<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}' xmlns='http://www.w3.org/2000/svg'>");

            // Axes
            sb.AppendLine($"<line x1='{margin}' y1='{sy(0)}' x2='{width-margin}' y2='{sy(0)}' stroke='black'/>\n<line x1='{margin}' y1='{sy(0)}' x2='{margin}' y2='{sy(1)}' stroke='black'/>");
            // Ticks and labels (x)
            for (int i = 0; i < xLossPct.Length; i++)
            {
                double x = sx(xLossPct[i]);
                sb.AppendLine($"<line x1='{x}' y1='{sy(0)}' x2='{x}' y2='{sy(0)+5}' stroke='black'/>\n<text x='{x}' y='{sy(0)+18}' text-anchor='middle' class='axis'>{xLossPct[i]:0}</text>");
            }
            sb.AppendLine($"<text x='{(width/2)}' y='{height-10}' text-anchor='middle'>Neuron loss (%)</text>");
            // y ticks
            for (double y = 0; y <= 1.0; y += 0.1)
            {
                double yy = sy(y);
                sb.AppendLine($"<line x1='{margin-5}' y1='{yy}' x2='{margin}' y2='{yy}' stroke='black'/>\n<text x='{margin-10}' y='{yy+4}' text-anchor='end' class='axis'>{y:0.0}</text>");
            }
            sb.AppendLine($"<text transform='translate(15 {(height/2)}) rotate(-90)' text-anchor='middle'>Recall accuracy</text>");

            // Std band polygon
            var poly = new StringBuilder();
            for (int i = 0; i < xLossPct.Length; i++)
                poly.Append($"{sx(xLossPct[i]).ToString(CultureInfo.InvariantCulture)},{sy(upper[i]).ToString(CultureInfo.InvariantCulture)} ");
            for (int i = xLossPct.Length-1; i >= 0; i--)
                poly.Append($"{sx(xLossPct[i]).ToString(CultureInfo.InvariantCulture)},{sy(lower[i]).ToString(CultureInfo.InvariantCulture)} ");
            sb.AppendLine($"<polygon points='{poly}' fill='rgba(200,200,200,0.6)' stroke='none' />");

            // Mean line
            var path = new StringBuilder();
            for (int i = 0; i < xLossPct.Length; i++)
            {
                double x = sx(xLossPct[i]);
                double y = sy(meanAcc[i]);
                path.Append(i == 0 ? $"M{x},{y} " : $"L{x},{y} ");
            }
            sb.AppendLine($"<path d='{path}' fill='none' stroke='#1f77b4' stroke-width='2' />");

            // Markers
            for (int i = 0; i < xLossPct.Length; i++)
            {
                double x = sx(xLossPct[i]);
                double y = sy(meanAcc[i]);
                sb.AppendLine($"<circle cx='{x}' cy='{y}' r='3' fill='#1f77b4' />");
            }

            sb.AppendLine("</svg>");
            sb.AppendLine("<p>This curve summarizes how recall degrades as progressively more neurons are removed; the gray band indicates ±1 standard deviation over random lesions and cues.</p>");
            sb.AppendLine("</body></html>");
            return sb.ToString();
        }

        public struct ReconPanel
        {
            public string Title;
            public int[] Grid; // length H*W, values: -1 (black), 0 (gray), +1 (white)
            public int H, W;
        }

        public static string ReconstructionHtml(ReconPanel[] panels)
        {
            var sb = new StringBuilder();
            sb.AppendLine("<!DOCTYPE html><html><head><meta charset='utf-8'><title>Reconstructions Grid</title>");
            sb.AppendLine("<style>body{font-family:Segoe UI,Arial,sans-serif;} .title{font-size:18px;font-weight:600;margin:10px 0} .grid{display:grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap:18px;} .panel{border:1px solid #ddd; padding:10px; border-radius:10px; box-shadow:0 1px 3px rgba(0,0,0,.06);} .label{font-weight:600; margin-bottom:8px;} table.px{border-collapse:collapse;} table.px td{width:16px; height:16px; padding:0;}.cwhite{background:#000;} .cblack{background:#fff;} .cgray{background:#bbb;} </style></head><body>"); /* note: white/black reversed to match visual contrast of ±1 mapping */
            sb.AppendLine("<div class='title'>Reconstructions at Selected Neuron‑Loss Levels</div>");
            sb.AppendLine("<div class='grid'>");

            foreach (var p in panels)
            {
                sb.AppendLine("<div class='panel'>");
                sb.AppendLine($"<div class='label'>{System.Web.HttpUtility.HtmlEncode(p.Title)}</div>");
                sb.AppendLine(RenderPatternTable(p.Grid, p.H, p.W));
                sb.AppendLine("</div>");
            }

            sb.AppendLine("</div>");
            sb.AppendLine("<p>Each mini‑image is an 8×8 pattern. Black/white correspond to −1/+1 states; gray indicates removed neurons. As neuron loss grows, pattern completion becomes fragmented and eventually fails.</p>");
            sb.AppendLine("</body></html>");
            return sb.ToString();
        }

        static string RenderPatternTable(int[] grid, int H, int W)
        {
            var sb = new StringBuilder();
            sb.AppendLine("<table class='px'>");
            for (int r = 0; r < H; r++)
            {
                sb.AppendLine("<tr>");
                for (int c = 0; c < W; c++)
                {
                    int v = grid[r*W + c];
                    string cls = v == 0 ? "cgray" : (v == 1 ? "cblack" : "cwhite");
                    sb.Append($"<td class='{cls}'></td>");
                }
                sb.AppendLine("</tr>");
            }
            sb.AppendLine("</table>");
            return sb.ToString();
        }
    }
}
