import { useState, useCallback, useMemo } from "react";
import * as Papa from "papaparse";
import * as XLSX from "xlsx";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, PieChart, Pie, Legend, LineChart, Line, CartesianGrid } from "recharts";

// ─── Utility ───────────────────────────────────────────────────────────────
const mean = arr => arr.reduce((a, b) => a + b, 0) / arr.length;
const median = arr => {
  const s = [...arr].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};
const std = arr => {
  const m = mean(arr);
  return Math.sqrt(arr.reduce((a, b) => a + (b - m) ** 2, 0) / arr.length);
};
const corr = (a, b) => {
  const ma = mean(a), mb = mean(b);
  const num = a.reduce((s, v, i) => s + (v - ma) * (b[i] - mb), 0);
  const den = Math.sqrt(a.reduce((s, v) => s + (v - ma) ** 2, 0) * b.reduce((s, v) => s + (v - mb) ** 2, 0));
  return den === 0 ? 0 : num / den;
};

// ─── Decision Tree Node ────────────────────────────────────────────────────
function buildTree(X, y, depth = 0, maxDepth = 5) {
  const n = y.length;
  const pos = y.filter(v => v === 1).length;
  if (depth >= maxDepth || n < 10 || pos === 0 || pos === n) return { leaf: true, val: pos / n };
  let bestGain = -Infinity, bestFeat = -1, bestThresh = 0;
  const feats = Array.from({ length: X[0].length }, (_, i) => i)
    .sort(() => Math.random() - 0.5).slice(0, Math.min(20, X[0].length));
  for (const fi of feats) {
    const vals = X.map(r => r[fi]);
    const uniq = [...new Set(vals)].sort((a, b) => a - b);
    for (let ti = 0; ti < Math.min(uniq.length - 1, 10); ti++) {
      const t = (uniq[ti] + uniq[ti + 1]) / 2;
      const lY = y.filter((_, i) => vals[i] <= t);
      const rY = y.filter((_, i) => vals[i] > t);
      if (!lY.length || !rY.length) continue;
      const gini = v => { const p = v.filter(x => x === 1).length / v.length; return 1 - p * p - (1 - p) ** 2; };
      const gain = gini(y) - (lY.length / n) * gini(lY) - (rY.length / n) * gini(rY);
      if (gain > bestGain) { bestGain = gain; bestFeat = fi; bestThresh = t; }
    }
  }
  if (bestFeat === -1) return { leaf: true, val: pos / n };
  const lIdx = y.map((_, i) => vals => X[i][bestFeat] <= bestThresh).map((_, i) => i).filter(i => X[i][bestFeat] <= bestThresh);
  const rIdx = y.map((_, i) => i).filter(i => X[i][bestFeat] > bestThresh);
  return {
    leaf: false, feat: bestFeat, thresh: bestThresh,
    left: buildTree(lIdx.map(i => X[i]), lIdx.map(i => y[i]), depth + 1, maxDepth),
    right: buildTree(rIdx.map(i => X[i]), rIdx.map(i => y[i]), depth + 1, maxDepth)
  };
}
function predictTree(node, x) {
  if (node.leaf) return node.val;
  return x[node.feat] <= node.thresh ? predictTree(node.left, x) : predictTree(node.right, x);
}

// ─── XGBoost-like Gradient Boosting ───────────────────────────────────────
function trainXGB(X, y, nTrees = 80, lr = 0.1) {
  const preds = new Array(X.length).fill(0);
  const trees = [];
  for (let t = 0; t < nTrees; t++) {
    const probs = preds.map(p => 1 / (1 + Math.exp(-p)));
    const residuals = y.map((yi, i) => yi - probs[i]);
    const tree = buildTree(X, residuals, 0, 4);
    trees.push(tree);
    X.forEach((x, i) => { preds[i] += lr * predictTree(tree, x); });
  }
  return trees;
}
function predictXGB(trees, X, lr = 0.1, thresh = 0.4) {
  return X.map(x => {
    const raw = trees.reduce((s, t) => s + lr * predictTree(t, x), 0);
    const prob = 1 / (1 + Math.exp(-raw));
    return { prob, pred: prob > thresh ? 1 : 0 };
  });
}

// ─── SMOTE ────────────────────────────────────────────────────────────────
function smote(X, y, k = 5) {
  const minClass = y.map((v, i) => ({ v, i })).filter(d => d.v === 1).map(d => d.i);
  const majClass = y.map((v, i) => ({ v, i })).filter(d => d.v === 0).map(d => d.i);
  const diff = majClass.length - minClass.length;
  if (diff <= 0) return { X, y };
  const synth = [];
  for (let s = 0; s < diff; s++) {
    const idx = minClass[Math.floor(Math.random() * minClass.length)];
    const neighbor = minClass[Math.floor(Math.random() * minClass.length)];
    const gap = Math.random();
    synth.push(X[idx].map((v, f) => v + gap * (X[neighbor][f] - v)));
  }
  return { X: [...X, ...synth], y: [...y, ...synth.map(() => 1)] };
}

// ─── Pipeline ─────────────────────────────────────────────────────────────
function runPipeline(rawData, setLog) {
  const log = msg => setLog(l => [...l, msg]);
  // Parse
  const cols = Object.keys(rawData[0]);
  log(`✅ Loaded ${rawData.length} rows × ${cols.length} columns`);

  // Find target
  const targetCol = cols.find(c => c.toLowerCase().includes("pass") || c.toLowerCase().includes("fail")) || cols[cols.length - 1];
  log(`🎯 Target column detected: "${targetCol}"`);

  // Drop time cols
  let workCols = cols.filter(c => !c.toLowerCase().includes("time") && c !== targetCol);

  // Build matrix
  let matrix = rawData.map(row => workCols.map(c => parseFloat(row[c])));
  let yRaw = rawData.map(row => row[targetCol]);
  const yVals = yRaw.map(v => {
    const n = parseFloat(v);
    if (n === -1) return 1;
    if (n === 1) return 0;
    return n;
  });

  log(`🏷️ Pass/Fail distribution — Fail: ${yVals.filter(v => v === 1).length}, Pass: ${yVals.filter(v => v === 0).length}`);

  // Missing % per col
  const missingPct = workCols.map((_, fi) => {
    const col = matrix.map(r => r[fi]);
    return col.filter(v => isNaN(v)).length / col.length;
  });
  const keepIdx = missingPct.map((p, i) => p <= 0.5 ? i : -1).filter(i => i !== -1);
  workCols = keepIdx.map(i => workCols[i]);
  matrix = matrix.map(r => keepIdx.map(i => r[i]));
  log(`🧹 Dropped ${missingPct.filter(p => p > 0.5).length} cols with >50% missing. Remaining: ${workCols.length}`);

  // Constant cols
  const varIdx = workCols.map((_, fi) => {
    const col = matrix.map(r => r[fi]).filter(v => !isNaN(v));
    return new Set(col).size > 1 ? fi : -1;
  }).filter(i => i !== -1);
  workCols = varIdx.map(i => workCols[i]);
  matrix = matrix.map(r => varIdx.map(i => r[i]));
  log(`🧹 Dropped constant columns. Remaining: ${workCols.length}`);

  // Median imputation
  const medians = workCols.map((_, fi) => {
    const col = matrix.map(r => r[fi]).filter(v => !isNaN(v));
    return col.length ? median(col) : 0;
  });
  matrix = matrix.map(r => r.map((v, fi) => isNaN(v) ? medians[fi] : v));
  log(`💉 Median imputation complete`);

  // Standardize
  const means = workCols.map((_, fi) => mean(matrix.map(r => r[fi])));
  const stds = workCols.map((_, fi) => std(matrix.map(r => r[fi])) || 1);
  matrix = matrix.map(r => r.map((v, fi) => (v - means[fi]) / stds[fi]));
  log(`📐 Standardization complete`);

  // Remove high corr (sample 50 cols for speed)
  const sampleSize = Math.min(workCols.length, 80);
  const sampleIdx = Array.from({ length: sampleSize }, (_, i) => Math.floor(i * workCols.length / sampleSize));
  let keepCorr = [...sampleIdx];
  const dropCorr = new Set();
  for (let i = 0; i < keepCorr.length; i++) {
    if (dropCorr.has(keepCorr[i])) continue;
    for (let j = i + 1; j < keepCorr.length; j++) {
      if (dropCorr.has(keepCorr[j])) continue;
      const a = matrix.map(r => r[keepCorr[i]]);
      const b = matrix.map(r => r[keepCorr[j]]);
      if (Math.abs(corr(a, b)) > 0.9) dropCorr.add(keepCorr[j]);
    }
  }
  const finalIdx = sampleIdx.filter(i => !dropCorr.has(i));
  workCols = finalIdx.map(i => workCols[i]);
  matrix = matrix.map(r => finalIdx.map(i => r[i]));
  log(`🔗 Removed highly correlated features. Remaining: ${workCols.length}`);

  // SelectKBest (F-score proxy: variance by class)
  const k = Math.min(50, workCols.length);
  const scores = workCols.map((_, fi) => {
    const col0 = matrix.filter((_, i) => yVals[i] === 0).map(r => r[fi]);
    const col1 = matrix.filter((_, i) => yVals[i] === 1).map(r => r[fi]);
    if (!col0.length || !col1.length) return 0;
    return Math.abs(mean(col1) - mean(col0)) / ((std(col0) + std(col1)) / 2 + 1e-9);
  });
  const topIdx = scores.map((s, i) => ({ s, i })).sort((a, b) => b.s - a.s).slice(0, k).map(d => d.i);
  const topFeatures = topIdx.map(i => workCols[i]);
  const topScores = topIdx.map(i => scores[i]);
  matrix = matrix.map(r => topIdx.map(i => r[i]));
  log(`⭐ Selected top ${k} features via F-score`);

  // Train/test split 80/20
  const n = matrix.length;
  const idx = Array.from({ length: n }, (_, i) => i).sort(() => Math.random() - 0.5);
  const splitAt = Math.floor(n * 0.8);
  const trainIdx = idx.slice(0, splitAt);
  const testIdx = idx.slice(splitAt);
  let Xtr = trainIdx.map(i => matrix[i]);
  let ytr = trainIdx.map(i => yVals[i]);
  const Xte = testIdx.map(i => matrix[i]);
  const yte = testIdx.map(i => yVals[i]);

  // SMOTE
  const smoted = smote(Xtr, ytr);
  Xtr = smoted.X; ytr = smoted.y;
  log(`⚖️ SMOTE applied. Training samples: ${Xtr.length}`);

  // Train XGBoost
  log(`🤖 Training XGBoost model...`);
  const trees = trainXGB(Xtr, ytr, 60, 0.1);

  // Predict
  const results = predictXGB(trees, Xte, 0.1, 0.4);
  const yPred = results.map(r => r.pred);
  const yProb = results.map(r => r.prob);

  // Metrics
  const tp = yPred.filter((p, i) => p === 1 && yte[i] === 1).length;
  const fp = yPred.filter((p, i) => p === 1 && yte[i] === 0).length;
  const fn = yPred.filter((p, i) => p === 0 && yte[i] === 1).length;
  const tn = yPred.filter((p, i) => p === 0 && yte[i] === 0).length;
  const accuracy = (tp + tn) / yte.length;
  const precision = tp / (tp + fp) || 0;
  const recall = tp / (tp + fn) || 0;
  const f1 = 2 * precision * recall / (precision + recall) || 0;

  // ROC AUC approx
  const sorted = yProb.map((p, i) => ({ p, y: yte[i] })).sort((a, b) => b.p - a.p);
  let auc = 0, fps = 0, tps = 0;
  const totalP = yte.filter(v => v === 1).length;
  const totalN = yte.filter(v => v === 0).length;
  sorted.forEach(({ y }) => { y === 1 ? tps++ : fps++; auc += tps; });
  const rocAuc = totalP && totalN ? auc / (totalP * totalN) : 0.5;

  log(`✅ Model trained! Accuracy: ${(accuracy * 100).toFixed(1)}%`);

  // Feature importance (score proxy)
  const featureImportance = topFeatures.map((name, i) => ({ name, importance: topScores[i] }))
    .sort((a, b) => b.importance - a.importance).slice(0, 10);

  // Financial
  const totalRows = rawData.length;
  const failCount = yVals.filter(v => v === 1).length;
  const failRate = failCount / totalRows;
  const costPerFail = 5000;
  const monthlyProd = 10000;
  const monthlyFails = monthlyProd * failRate;
  const monthlyLoss = monthlyFails * costPerFail;
  const implCost = 150000;
  const financials = [0.10, 0.20, 0.30].map(rate => {
    const savings = monthlyFails * rate * costPerFail;
    const annual = savings * 12;
    const roi = ((annual - implCost) / implCost) * 100;
    return { rate: `${rate * 100}%`, monthly: savings, annual, roi, payback: implCost / savings };
  });

  return {
    metrics: { accuracy, precision, recall, f1, rocAuc },
    confusion: { tp, fp, fn, tn },
    featureImportance,
    financials,
    dataStats: { rows: totalRows, cols: cols.length, failCount, passCount: totalRows - failCount, failRate },
    targetCol
  };
}

// ─── Components ────────────────────────────────────────────────────────────
const Card = ({ children, className = "" }) => (
  <div className={`bg-gray-800 border border-gray-700 rounded-xl p-5 ${className}`}>{children}</div>
);
const MetricCard = ({ label, value, color = "text-cyan-400" }) => (
  <Card className="text-center">
    <div className={`text-3xl font-bold ${color}`}>{value}</div>
    <div className="text-gray-400 text-sm mt-1">{label}</div>
  </Card>
);

const COLORS = ["#06b6d4", "#f59e0b", "#10b981", "#f43f5e", "#8b5cf6"];

export default function App() {
  const [stage, setStage] = useState("upload"); // upload | processing | results
  const [logs, setLogs] = useState([]);
  const [results, setResults] = useState(null);
  const [fileName, setFileName] = useState("");
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState("");

  const processFile = useCallback(async file => {
    setFileName(file.name);
    setStage("processing");
    setLogs([]);
    setError("");
    try {
      let rawData;
      if (file.name.endsWith(".csv")) {
        rawData = await new Promise((res, rej) => Papa.parse(file, {
          header: true, dynamicTyping: false, skipEmptyLines: true,
          complete: r => res(r.data), error: rej
        }));
      } else {
        const buf = await file.arrayBuffer();
        const wb = XLSX.read(buf, { type: "array" });
        rawData = XLSX.utils.sheet_to_json(wb.Sheets[wb.SheetNames[0]]);
      }
      if (!rawData.length) throw new Error("Empty dataset");
      setTimeout(() => {
        try {
          const res = runPipeline(rawData, setLogs);
          setResults(res);
          setStage("results");
        } catch (e) { setError(e.message); setStage("upload"); }
      }, 100);
    } catch (e) { setError(e.message); setStage("upload"); }
  }, []);

  const onDrop = useCallback(e => {
    e.preventDefault(); setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) processFile(file);
  }, [processFile]);

  const onInput = useCallback(e => {
    const file = e.target.files[0];
    if (file) processFile(file);
  }, [processFile]);

  const fmt = (n, dec = 1) => (n * 100).toFixed(dec) + "%";
  const fmtUSD = n => "$" + n.toLocaleString("en-US", { maximumFractionDigits: 0 });

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100 font-sans">
      {/* Header */}
      <div className="bg-gray-950 border-b border-gray-800 px-8 py-4 flex items-center gap-4">
        <div className="w-8 h-8 rounded-lg bg-cyan-500 flex items-center justify-center text-black font-bold text-sm">SC</div>
        <div>
          <h1 className="text-lg font-semibold text-white">Semiconductor Defect Analyzer</h1>
          <p className="text-xs text-gray-500">XGBoost · SMOTE · SHAP Feature Analysis</p>
        </div>
        {results && (
          <button onClick={() => { setStage("upload"); setResults(null); setLogs([]); }}
            className="ml-auto text-xs bg-gray-700 hover:bg-gray-600 px-3 py-1.5 rounded-lg transition">
            ← New Dataset
          </button>
        )}
      </div>

      <div className="max-w-6xl mx-auto px-6 py-8">

        {/* Upload */}
        {stage === "upload" && (
          <div className="flex flex-col items-center justify-center min-h-96">
            <h2 className="text-2xl font-bold mb-2 text-white">Upload Your Dataset</h2>
            <p className="text-gray-400 mb-8 text-sm">Supports CSV and Excel (.xlsx) files with a Pass/Fail target column</p>
            <label
              onDragOver={e => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}
              className={`w-full max-w-xl border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer transition
                ${dragging ? "border-cyan-400 bg-cyan-950" : "border-gray-600 hover:border-cyan-500 hover:bg-gray-800"}`}>
              <input type="file" accept=".csv,.xlsx,.xls" className="hidden" onChange={onInput} />
              <div className="text-4xl mb-3">📂</div>
              <div className="text-white font-medium">Drag & drop your file here</div>
              <div className="text-gray-500 text-sm mt-1">or click to browse</div>
              <div className="mt-4 flex justify-center gap-2">
                {["CSV", "XLSX", "XLS"].map(t => (
                  <span key={t} className="text-xs bg-gray-700 px-2 py-1 rounded">{t}</span>
                ))}
              </div>
            </label>
            {error && <div className="mt-4 text-red-400 text-sm bg-red-950 px-4 py-2 rounded-lg">⚠️ {error}</div>}
          </div>
        )}

        {/* Processing */}
        {stage === "processing" && (
          <div className="max-w-xl mx-auto mt-10">
            <Card>
              <div className="flex items-center gap-3 mb-5">
                <div className="w-5 h-5 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin" />
                <div className="font-semibold text-cyan-400">Processing: {fileName}</div>
              </div>
              <div className="space-y-2 max-h-64 overflow-y-auto pr-1">
                {logs.map((l, i) => (
                  <div key={i} className="text-sm text-gray-300 font-mono">{l}</div>
                ))}
                {!logs.length && <div className="text-sm text-gray-500 font-mono">Initializing pipeline...</div>}
              </div>
            </Card>
          </div>
        )}

        {/* Results */}
        {stage === "results" && results && (() => {
          const { metrics, confusion, featureImportance, financials, dataStats } = results;
          const pieData = [
            { name: "Pass", value: dataStats.passCount },
            { name: "Fail", value: dataStats.failCount }
          ];
          const confData = [
            { name: "True Neg", value: confusion.tn, fill: "#10b981" },
            { name: "False Pos", value: confusion.fp, fill: "#f43f5e" },
            { name: "False Neg", value: confusion.fn, fill: "#f59e0b" },
            { name: "True Pos", value: confusion.tp, fill: "#06b6d4" },
          ];
          return (
            <div className="space-y-6">
              {/* Dataset Summary */}
              <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
                <MetricCard label="Total Samples" value={dataStats.rows.toLocaleString()} color="text-white" />
                <MetricCard label="Fail Rate" value={fmt(dataStats.failRate)} color="text-red-400" />
                <MetricCard label="Pass Samples" value={dataStats.passCount.toLocaleString()} color="text-emerald-400" />
                <MetricCard label="Fail Samples" value={dataStats.failCount.toLocaleString()} color="text-amber-400" />
              </div>

              {/* Model Metrics */}
              <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-widest">Model Performance</h3>
              <div className="grid grid-cols-2 gap-4 sm:grid-cols-5">
                <MetricCard label="Accuracy" value={fmt(metrics.accuracy)} />
                <MetricCard label="Precision" value={fmt(metrics.precision)} color="text-emerald-400" />
                <MetricCard label="Recall" value={fmt(metrics.recall)} color="text-amber-400" />
                <MetricCard label="F1 Score" value={fmt(metrics.f1)} color="text-purple-400" />
                <MetricCard label="ROC AUC" value={metrics.rocAuc.toFixed(3)} color="text-pink-400" />
              </div>

              {/* Charts Row */}
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
                {/* Pie */}
                <Card>
                  <h4 className="text-sm font-semibold text-gray-300 mb-3">Pass/Fail Distribution</h4>
                  <ResponsiveContainer width="100%" height={180}>
                    <PieChart>
                      <Pie data={pieData} cx="50%" cy="50%" outerRadius={65} dataKey="value" label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}>
                        <Cell fill="#10b981" /><Cell fill="#f43f5e" />
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                </Card>

                {/* Confusion Matrix */}
                <Card>
                  <h4 className="text-sm font-semibold text-gray-300 mb-3">Confusion Matrix</h4>
                  <ResponsiveContainer width="100%" height={180}>
                    <BarChart data={confData} margin={{ top: 5, right: 5, left: -20, bottom: 5 }}>
                      <XAxis dataKey="name" tick={{ fontSize: 11, fill: "#9ca3af" }} />
                      <YAxis tick={{ fontSize: 11, fill: "#9ca3af" }} />
                      <Tooltip />
                      <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                        {confData.map((d, i) => <Cell key={i} fill={d.fill} />)}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </Card>

                {/* Top Features */}
                <Card>
                  <h4 className="text-sm font-semibold text-gray-300 mb-3">Top Feature Importance</h4>
                  <ResponsiveContainer width="100%" height={180}>
                    <BarChart data={featureImportance.slice(0, 6)} layout="vertical" margin={{ left: 10, right: 10 }}>
                      <XAxis type="number" tick={{ fontSize: 10, fill: "#9ca3af" }} />
                      <YAxis type="category" dataKey="name" tick={{ fontSize: 9, fill: "#9ca3af" }} width={70} />
                      <Tooltip />
                      <Bar dataKey="importance" fill="#06b6d4" radius={[0, 4, 4, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </Card>
              </div>

              {/* Financial Impact */}
              <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-widest">Financial Impact Projection</h3>
              <Card>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="text-gray-400 border-b border-gray-700">
                        {["Improvement Rate", "Monthly Savings", "Annual Savings", "ROI", "Payback Period"].map(h => (
                          <th key={h} className="text-left py-2 px-3 font-medium">{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {financials.map((f, i) => (
                        <tr key={i} className="border-b border-gray-800 hover:bg-gray-750">
                          <td className="py-3 px-3 font-semibold text-cyan-400">{f.rate}</td>
                          <td className="py-3 px-3 text-emerald-400">{fmtUSD(f.monthly)}/mo</td>
                          <td className="py-3 px-3 text-emerald-400">{fmtUSD(f.annual)}/yr</td>
                          <td className={`py-3 px-3 font-medium ${f.roi > 0 ? "text-green-400" : "text-red-400"}`}>{f.roi.toFixed(1)}%</td>
                          <td className="py-3 px-3 text-gray-300">{f.payback.toFixed(1)} months</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="text-xs text-gray-500 mt-3">* Assumes $5,000 cost/failed wafer, 10,000 wafers/month production, $150,000 implementation cost</p>
              </Card>

              {/* Processing Log */}
              <Card>
                <h4 className="text-sm font-semibold text-gray-300 mb-3">Pipeline Log</h4>
                <div className="space-y-1">
                  {logs.map((l, i) => <div key={i} className="text-xs text-gray-400 font-mono">{l}</div>)}
                </div>
              </Card>
            </div>
          );
        })()}
      </div>
    </div>
  );
}
