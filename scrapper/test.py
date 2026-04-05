"""
RAG-e-Qanoon — Benchmark Analysis Script
=========================================
Reads the combined results CSV, ranks all models, and prints a full
ablation-study summary for every experiment group.

Usage:
    python analyze_results.py                        # uses default path
    python analyze_results.py path/to/results.csv   # custom path
"""

import csv
import sys
import numpy as np
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_CSV = sys.argv[1] if len(sys.argv) > 1 else "combined_results_all_models.csv"
COMPOSITE_FAITH_W = 0.6
COMPOSITE_REL_W   = 0.4

# ── Load data ─────────────────────────────────────────────────────────────────
model_scores = defaultdict(lambda: {"faith": [], "rel": [], "lat": []})
group_scores  = defaultdict(lambda: defaultdict(lambda: {"faith": [], "rel": [], "lat": []}))

with open(OUTPUT_CSV, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        try:
            faith = float(row["faithfulness"])
            rel   = float(row["relevancy"])
            model_scores[row["model_key"]]["faith"].append(faith)
            model_scores[row["model_key"]]["rel"].append(rel)
            group_scores[row["experiment_group"]][row["config_name"]]["faith"].append(faith)
            group_scores[row["experiment_group"]][row["config_name"]]["rel"].append(rel)
        except (ValueError, TypeError):
            pass

        try:
            lat = float(row["total_latency_s"])
            model_scores[row["model_key"]]["lat"].append(lat)
            group_scores[row["experiment_group"]][row["config_name"]]["lat"].append(lat)
        except (ValueError, TypeError):
            pass

# ── Model comparison (Group: llm_model) ───────────────────────────────────────
print("\n" + "=" * 72)
print("  MODEL RANKING — Group: llm_model (multi-model comparison)")
print(f"  Composite = {COMPOSITE_FAITH_W*100:.0f}% Faithfulness + {COMPOSITE_REL_W*100:.0f}% Relevancy")
print("=" * 72)
print(f"{'Rank':<5} {'Model':<25} {'Faithfulness':>13} {'Relevancy':>12} {'Composite':>12} {'Avg Lat(s)':>11} {'N':>4}")
print("-" * 72)

llm_results = []
for model, s in model_scores.items():
    avg_f = np.mean(s["faith"]) if s["faith"] else 0
    avg_r = np.mean(s["rel"])   if s["rel"]   else 0
    avg_l = np.mean(s["lat"])   if s["lat"]   else 0
    comp  = COMPOSITE_FAITH_W * avg_f + COMPOSITE_REL_W * avg_r
    llm_results.append((model, avg_f, avg_r, comp, avg_l, len(s["faith"])))

llm_results.sort(key=lambda x: x[3], reverse=True)
for i, (model, f, r, c, l, n) in enumerate(llm_results, 1):
    marker = " 🏆" if i == 1 else ""
    print(f"{i:<5} {model:<25} {f:>12.2%} {r:>11.2%} {c:>11.2%} {l:>10.2f}s {n:>4}{marker}")

if llm_results:
    best = llm_results[0]
    print(f"\n  Best model : {best[0]}")
    print(f"  Composite  : {best[3]:.2%}")
    print(f"  Faithfulness: {best[1]:.2%}  |  Relevancy: {best[2]:.2%}  |  Avg Latency: {best[4]:.2f}s")

# ── Full benchmark summary ─────────────────────────────────────────────────────
print("\n\n" + "=" * 72)
print("  FULL BENCHMARK SUMMARY — all experiment groups")
print("=" * 72)
print(f"{'Group':<26} {'Config':<22} {'Faith':>8} {'Rel':>8} {'Comp':>8} {'Lat(s)':>8} {'N':>4}")
print("-" * 72)

for group in sorted(group_scores.keys()):
    group_results = []
    for config, s in group_scores[group].items():
        avg_f = np.mean(s["faith"]) if s["faith"] else 0
        avg_r = np.mean(s["rel"])   if s["rel"]   else 0
        avg_l = np.mean(s["lat"])   if s["lat"]   else 0
        comp  = COMPOSITE_FAITH_W * avg_f + COMPOSITE_REL_W * avg_r
        n     = len(s["faith"])
        group_results.append((config, avg_f, avg_r, comp, avg_l, n))

    group_results.sort(key=lambda x: x[2], reverse=True)  # sort by composite

    best_config = group_results[0][0] if group_results else None
    for config, f, r, c, l, n in group_results:
        marker = " ← best" if config == best_config else ""
        print(f"  {group:<24} {config:<22} {f:>7.1%} {r:>7.1%} {c:>7.1%} {l:>7.2f}s {n:>4}{marker}")
    print()

# ── Per-group winner summary ───────────────────────────────────────────────────
print("=" * 72)
print("  WINNER PER EXPERIMENT GROUP")
print("=" * 72)
print(f"{'Group':<26} {'Best Config':<22} {'Faith':>8} {'Rel':>8} {'Composite':>10}")
print("-" * 72)

for group in sorted(group_scores.keys()):
    best = None
    best_comp = -1
    for config, s in group_scores[group].items():
        f = np.mean(s["faith"]) if s["faith"] else 0
        r = np.mean(s["rel"])   if s["rel"]   else 0
        c = COMPOSITE_FAITH_W * f + COMPOSITE_REL_W * r
        if c > best_comp:
            best_comp = c
            best = (config, f, r, c)
    if best:
        print(f"  {group:<24} {best[0]:<22} {best[1]:>7.1%} {best[2]:>7.1%} {best[3]:>9.1%}")

print("\n" + f"  Full results CSV: {OUTPUT_CSV}")
print("=" * 72 + "\n")
