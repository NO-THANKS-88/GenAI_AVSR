#!/usr/bin/env python
"""
畫出四組微調實驗的訓練曲線（讀 fairseq 的 hydra_train.log）。

輸出（OUT_DIR 底下）
  training_curves.png   三張並排小圖：訓練 loss、驗證 loss、驗證 accuracy
  training_summary.tsv  每組實驗的最佳驗證 accuracy 與對應步數

用法：改好「設定區」後，任何有 matplotlib 的環境都能跑
  conda activate avhubert
  python plot_curves.py
"""
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# ======================== 設定區（只要改這裡） ========================
EXP_ROOT = "~/GAN/exp"
EXPERIMENTS = ["train_10h_A", "train_10h_AV", "train_30h_A", "train_30h_AV"]
OUT_DIR = "~/GAN/exp/plots"
UNFREEZE_STEP = 14400          # 官方 30h 設定：第 14,400 步解凍編碼器
SMOOTH = 5                     # 訓練 loss 的移動平均視窗（點數）
# =====================================================================

# 顏色 = 模態，線型 = 資料量（兩種編碼都有，不只靠顏色區分）
COLOR = {"A": "#2a78d6", "AV": "#eb6834"}
DASH = {"10h": (0, (5, 3)), "30h": "solid"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
LINE = re.compile(r"\]\[(train_inner|valid)\]\[INFO\] - (\{.*\})")


def parse(exp_dir):
    train, valid = {}, {}
    logs = sorted(glob.glob(os.path.join(exp_dir, "**", "*.log"), recursive=True), key=os.path.getmtime)
    for path in logs:                         # 中斷後續跑會有多段 log，同一步以最新的為準
        with open(path, errors="ignore") as f:
            for line in f:
                m = LINE.search(line)
                if not m:
                    continue
                d = json.loads(m.group(2))
                if m.group(1) == "train_inner":
                    train[int(float(d["num_updates"]))] = float(d["nll_loss"])
                else:
                    step = int(float(d["valid_num_updates"]))
                    valid[step] = (float(d["valid_nll_loss"]), float(d["valid_accuracy"]))
    return dict(sorted(train.items())), dict(sorted(valid.items()))


def smooth(y, k):
    if k <= 1 or len(y) < k:
        return np.asarray(y)
    pad = np.pad(y, (k // 2, k - 1 - k // 2), mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=8)
    ax.set_xlabel("Update step", fontsize=9, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.axvline(UNFREEZE_STEP, color=MUTED, linewidth=1, linestyle=(0, (2, 2)), zorder=1)


def place_labels(ax, items, min_gap=0.07):
    """在線尾加標籤；曲線末端靠很近時，把標籤上下錯開避免重疊。items = [(x, y, 文字, 顏色)]"""
    if not items:
        return
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * min_gap
    items = sorted(items, key=lambda t: t[1])
    ys = [t[1] for t in items]
    for i in range(1, len(ys)):                     # 由下往上推開
        ys[i] = max(ys[i], ys[i - 1] + gap)
    shift = max(0.0, ys[-1] - hi)                   # 超出上緣就整組往下移
    ys = [y - shift for y in ys]
    xmax = max(t[0] for t in items)
    for (x, y, text, color), ly in zip(items, ys):
        ax.annotate(text, (x, y), xytext=(xmax + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.03, ly),
                    textcoords="data", va="center", fontsize=8, color=INK, annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=color, lw=1, shrinkA=0, shrinkB=2),
                    bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec=color, lw=1))


def main():
    root, out = os.path.expanduser(EXP_ROOT), os.path.expanduser(OUT_DIR)
    os.makedirs(out, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), facecolor=SURFACE)
    titles = [("Training loss (per token, smoothed)", "NLL loss (bits)"),
              ("Validation loss (per token)", "NLL loss (bits)"),
              ("Validation accuracy", "Token accuracy (%)")]
    for ax, (t, yl) in zip(axes, titles):
        style(ax, t, yl)

    rows, ends = [], [[], [], []]
    for name in EXPERIMENTS:
        size, mod = name.split("_")[1], name.split("_")[2]
        train, valid = parse(os.path.join(root, name))
        if not train:
            print(f"跳過 {name}：找不到 log")
            continue
        kw = dict(color=COLOR[mod], linestyle=DASH[size], linewidth=2, label=f"{size} · {mod}")
        tx, ty = list(train), smooth(list(train.values()), SMOOTH)
        axes[0].plot(tx, ty, **kw)
        ends[0].append((tx[-1], ty[-1], f"{size} {mod}", COLOR[mod]))
        if valid:
            vx = list(valid)
            vl = [v[0] for v in valid.values()]
            va = [v[1] for v in valid.values()]
            axes[1].plot(vx, vl, marker="o", markersize=4, markeredgecolor=SURFACE, markeredgewidth=1, **kw)
            axes[2].plot(vx, va, marker="o", markersize=4, markeredgecolor=SURFACE, markeredgewidth=1, **kw)
            ends[1].append((vx[-1], vl[-1], f"{size} {mod}", COLOR[mod]))
            ends[2].append((vx[-1], va[-1], f"{size} {mod}", COLOR[mod]))
            best = int(np.argmax(va))
            rows.append((name, max(train), vx[best], va[best], vl[best]))

    for ax, items in zip(axes, ends):
        lo, hi = ax.get_ylim()
        ax.text(UNFREEZE_STEP, lo + (hi - lo) * 0.02, "encoder unfrozen ", fontsize=7.5,
                color=MUTED, va="bottom", ha="right")
        place_labels(ax, items)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 1.02), labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 0.97, 0.93), w_pad=4)
    png = os.path.join(out, "training_curves.png")
    fig.savefig(png, dpi=200, facecolor=SURFACE)

    tsv = os.path.join(out, "training_summary.tsv")
    with open(tsv, "w") as f:
        f.write("experiment\tlast_step\tbest_valid_step\tbest_valid_acc\tvalid_nll_at_best\n")
        for r in rows:
            f.write(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]:.2f}\t{r[4]:.3f}\n")
    print(f"圖：{png}\n摘要：{tsv}")
    for r in rows:
        print(f"  {r[0]:14} 最佳驗證 accuracy {r[3]:.2f}%（第 {r[2]} 步）")


if __name__ == "__main__":
    main()
