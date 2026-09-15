#!/usr/bin/env python
"""Generate confusion matrix for the optimal 9-feature XGBoost model.

Final feature set (V4 - delta_hr):
  hr, sd2_sd1, hrv_rmssd, ss, rrv, resp_amp, mic_rms, rr, csi

Usage (from project root):
    python scripts/plots/plot_final_cm.py
"""
from __future__ import annotations
import sys, os
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder

from src.pipeline import build_feature_table, SafeXGBClassifier
from src.local_eval import META_COLS
from src.exclusions import filter_feature_df

CLASSES   = ["rest", "mental1", "mental2", "physical1", "physical2"]
FINAL_FEATS = ["csi", "hr", "hrv_rmssd", "mic_rms", "resp_amp", "rr", "rrv", "sd2_sd1", "ss"]

# ---------------------------------------------------------------------------
# Build dataset
# ---------------------------------------------------------------------------
print("Building feature table ...")
df_raw = build_feature_table(include_temp=False)
df_raw = filter_feature_df(df_raw)
df_raw = df_raw[df_raw["activity"].isin(CLASSES)].copy()
# drop delta_hr and samp_en ??use only the 9 final features
df = df_raw.copy()
print(f"Windows: {len(df)}  features: {FINAL_FEATS}")

# ---------------------------------------------------------------------------
# LORO evaluation
# ---------------------------------------------------------------------------
def _pad_missing(train_df, feat_cols, classes):
    missing = set(classes) - set(train_df["activity"].unique())
    if not missing:
        return train_df
    pad = pd.DataFrame([{**{c: np.nan for c in feat_cols},
                         "rec_name": f"_pad_{m}", "subject": "_pad",
                         "stressor": "_pad", "phase": m, "activity": m,
                         "t_start": 0.0, "t_end": 0.0}
                        for m in missing])
    return pd.concat([train_df, pad], ignore_index=True)

def _make_xgb():
    return SafeXGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, random_state=0, n_jobs=-1,
        eval_metric="mlogloss", tree_method="hist", device="cpu",
    )

le   = LabelEncoder().fit(CLASSES)
recs = sorted(df["rec_name"].unique())

y_true_all, y_pred_all = [], []
fold_accs, fold_f1s = [], []

for r in recs:
    test_df  = df[df.rec_name == r]
    train_df = df[df.rec_name != r]
    if test_df["activity"].nunique() < 1:
        continue
    train_df = _pad_missing(train_df, FINAL_FEATS, CLASSES)
    clf = _make_xgb()
    clf.fit(train_df[FINAL_FEATS].to_numpy(float), le.transform(train_df["activity"]))
    pred = clf.predict(test_df[FINAL_FEATS].to_numpy(float))
    y_true = le.inverse_transform(le.transform(test_df["activity"]))
    y_pred = le.inverse_transform(pred)
    y_true_all.extend(y_true)
    y_pred_all.extend(y_pred)
    fold_accs.append(accuracy_score(y_true, y_pred))
    fold_f1s.append(f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0))
    print(f"  fold [{r:30s}]  acc={fold_accs[-1]:.3f}  f1={fold_f1s[-1]:.3f}")

y_true_all = np.array(y_true_all)
y_pred_all = np.array(y_pred_all)

cm = confusion_matrix(y_true_all, y_pred_all, labels=CLASSES)
cm_f = cm.astype(float)
row_sums = cm_f.sum(axis=1, keepdims=True)
pct = cm_f / np.where(row_sums == 0, 1, row_sums) * 100.0
supports = cm.sum(axis=1)

pooled_acc = accuracy_score(y_true_all, y_pred_all)
pooled_f1  = f1_score(y_true_all, y_pred_all, labels=CLASSES, average="macro", zero_division=0)
per_f1  = f1_score(y_true_all, y_pred_all, labels=CLASSES, average=None, zero_division=0)
per_pre = precision_score(y_true_all, y_pred_all, labels=CLASSES, average=None, zero_division=0)
per_rec = recall_score(y_true_all, y_pred_all, labels=CLASSES, average=None, zero_division=0)

print(f"\nPooled Acc = {pooled_acc:.4f}   Pooled Macro F1 = {pooled_f1:.4f}")
print(f"Mean fold Acc = {np.mean(fold_accs):.4f}   Mean fold F1 = {np.mean(fold_f1s):.4f}")

print("\nConfusion Matrix:")
print("Labels:", CLASSES)
print(cm)
print("\nPer-class Precision:")
for c, val in zip(CLASSES, per_pre):
    print(f"  {c:<12s}: {val:.4f}")
print("Per-class Recall:")
for c, val in zip(CLASSES, per_rec):
    print(f"  {c:<12s}: {val:.4f}")
print("Per-class F1-score:")
for c, val in zip(CLASSES, per_f1):
    print(f"  {c:<12s}: {val:.4f}")

# ---------------------------------------------------------------------------
# Plot ??side-by-side: raw counts + row-normalised %
# ---------------------------------------------------------------------------
PALETTE   = "white"             # white background
ACCENT    = "#e94560"           # red accent
BLUE_CMAP = "Blues"
PCT_CMAP  = "Purples"

fig = plt.figure(figsize=(15, 6.5), facecolor=PALETTE)
fig.suptitle(
    "XGBoost  |  LORO Cross-Validation  |  9-Feature Optimised Set\n"
    f"Pooled Accuracy = {pooled_acc:.1%}    Pooled Macro F1 = {pooled_f1:.3f}",
    color="black", fontsize=13, fontweight="bold", y=1.02
)

gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 0.55],
                       wspace=0.38)

class_labels = [c.capitalize() for c in CLASSES]

def _draw_cm(ax, data, fmt, cmap, title, cbar_label, vmax=None):
    ax.set_facecolor(PALETTE)
    im = ax.imshow(data, cmap=cmap, vmin=0,
                   vmax=(vmax or data.max()), aspect="equal")
    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(class_labels, color="black", fontsize=9, rotation=25, ha="right")
    ax.set_yticklabels(
        [f"{c}\n(n={supports[i]:,})" for i, c in enumerate(class_labels)],
        color="black", fontsize=9)
    ax.set_xlabel("Predicted", color="black", fontsize=9, labelpad=8)
    ax.set_ylabel("True Label", color="black", fontsize=9, labelpad=8)
    ax.set_title(title, color="black", fontsize=10, pad=10, fontweight="bold")
    ax.tick_params(colors="black", length=0)
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")

    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            val = data[i, j]
            text_color = "white" if (vmax or data.max()) > 0 and val / (vmax or data.max()) > 0.45 else "black"
            ax.text(j, i, fmt.format(val),
                    ha="center", va="center", fontsize=10,
                    fontweight="bold" if i == j else "normal",
                    color=text_color)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.yaxis.set_tick_params(color="black")
    cb.ax.tick_params(colors="black", labelsize=7)
    cb.set_label(cbar_label, color="black", fontsize=8)
    cb.outline.set_edgecolor("#cccccc")

# Left: raw counts
ax0 = fig.add_subplot(gs[0])
_draw_cm(ax0, cm_f, "{:.0f}", BLUE_CMAP, "Raw Window Counts", "windows",
         vmax=cm_f.max())

# Middle: row-normalised %
ax1 = fig.add_subplot(gs[1])
_draw_cm(ax1, pct, "{:.1f}%", PCT_CMAP, "Row-Normalised  (%)", "% of true class",
         vmax=100)

# Right: per-class metrics bar chart
ax2 = fig.add_subplot(gs[2])
ax2.set_facecolor(PALETTE)
x = np.arange(len(CLASSES))
w = 0.25
bars_f1  = ax2.bar(x - w, per_f1,  w, label="F1",        color="#e94560", alpha=0.85)
bars_pre = ax2.bar(x,     per_pre, w, label="Precision",  color="#0f3460", alpha=0.85,
                   edgecolor="#e94560", linewidth=1)
bars_rec = ax2.bar(x + w, per_rec, w, label="Recall",     color="#533483", alpha=0.85)

for bars in [bars_f1, bars_pre, bars_rec]:
    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., h + 0.01,
                 f"{h:.2f}", ha="center", va="bottom", fontsize=6.5,
                 color="black", fontweight="bold")

ax2.set_xticks(x)
ax2.set_xticklabels(class_labels, color="black", fontsize=8, rotation=20, ha="right")
ax2.set_ylim(0, 1.12)
ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax2.set_yticklabels([f"{v:.0%}" for v in [0, 0.2, 0.4, 0.6, 0.8, 1.0]],
                    color="black", fontsize=7)
ax2.set_title("Per-Class Metrics", color="black", fontsize=10,
              fontweight="bold", pad=10)
ax2.legend(fontsize=7, facecolor="white", labelcolor="black",
           edgecolor="#cccccc", loc="upper right")
ax2.tick_params(colors="black", length=0)
ax2.axhline(pooled_f1, color=ACCENT, linestyle="--", linewidth=0.8, alpha=0.6)
ax2.text(len(CLASSES) - 0.3, pooled_f1 + 0.015,
         f"Macro F1={pooled_f1:.3f}", color=ACCENT, fontsize=6.5)
for spine in ax2.spines.values():
    spine.set_edgecolor("#cccccc")
ax2.set_facecolor(PALETTE)

# Feature list annotation
feat_text = "Features (9):\n" + "\n".join(f"  {f}" for f in sorted(FINAL_FEATS))
fig.text(0.01, -0.02, feat_text, color="#333333", fontsize=7,
         va="top", family="monospace")

fig.tight_layout()
out_path = Path("figures") / "ablation" / "final_cm_9feat_xgboost.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=PALETTE)
plt.close(fig)
print(f"\nSaved -> {out_path}")

