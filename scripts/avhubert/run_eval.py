#!/usr/bin/env python
"""
Step 5：在測試集上評估四個模型，乾淨與加噪音條件都測，算出 WER 並畫圖。

噪音：make_noise.py 產生的「測試用」babble 噪音（鄒語多人交談聲），
     來源說話者和訓練用噪音、測試集說話者都不重疊。
解碼：呼叫官方的 infer_s2s.py（和論文同一套流程）。

用法：放在 SynthAVSR/avhubert/ 裡，改好「設定區」後，在 tmux 裡執行
  conda activate avhubert
  cd ~/GAN/SynthAVSR/avhubert
  python run_eval.py
已經跑過的條件會自動跳過，中斷後重跑即可續跑。

輸出（OUT_DIR 底下）
  results.tsv     每個「模型 × 模態 × SNR」的 WER
  wer_vs_snr.png  WER 隨噪音變化的折線圖
  decode/...      每個條件的辨識結果（hypo-*.json 有逐句的 REF/HYP）
"""
import glob
import os
import re
import subprocess
import sys
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# ======================== 設定區（只要改這裡） ========================
EXP_ROOT = "~/GAN/exp"
DATA_ROOT = "~/GAN/data/tsou/avhubert"
SYNTH_AUDIO = "~/GAN/data/tsou/synth/audio"
EXPERIMENTS = ["train_10h_A", "train_10h_AV", "train_30h_A", "train_30h_AV",
               "train_30h_A_noisy", "train_30h_AV_noisy"]          # 還沒訓練完的會自動跳過
PLOT_EXPERIMENTS = ["train_30h_A", "train_30h_AV", "train_30h_A_noisy", "train_30h_AV_noisy"]
NOISE_DIR = "~/GAN/exp/noise/test_babble"   # make_noise.py 產生的「測試用」噪音（和訓練用噪音不同說話者）
SNRS = [None, 10, 5, 0, -5]        # None = 乾淨；其他是訊噪比（dB），越小越吵
GPU = 8
BEAM = 20                          # 官方預設 50；20 快很多，結果差異通常很小
OUT_DIR = "~/GAN/exp/eval"      # 換了測試噪音，所以放新資料夾，基準模型也要重測
# =====================================================================

COLOR = {"A": "#2a78d6", "AV": "#eb6834"}            # 顏色 = 模態
DASH_POOL = ["solid", (0, (5, 3)), (0, (1, 2)), (0, (6, 2, 1, 2))]   # 線型 = 其餘差異（資料量、是否加噪音訓練）
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def x(p):
    return os.path.abspath(os.path.expanduser(p))


def read_wav(path):
    with wave.open(path) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)


def read_wer(res_dir):
    files = glob.glob(f"{res_dir}/wer.*")
    if not files:
        return None
    m = re.search(r"WER:\s*([\d.]+)", open(files[0]).read())
    return float(m.group(1)) if m else None


def decode(exp, mods, snr, noise_dir, out_dir):
    size = exp.split("_")[1]
    tag = "clean" if snr is None else f"snr{snr}"
    res = f"{out_dir}/decode/{exp}/{'+'.join(mods)}/{tag}"
    wer = read_wer(res)
    if wer is not None:
        return wer
    ckpt = x(f"{EXP_ROOT}/{exp}/checkpoints/checkpoint_best.pt")
    if not os.path.isfile(ckpt):
        print(f"  找不到 {ckpt}，跳過")
        return None
    data = x(f"{DATA_ROOT}/train_{size}")
    cmd = [sys.executable, "-B", f"{HERE}/infer_s2s.py", "--config-dir", f"{HERE}/conf",
           "--config-name", "s2s_decode.yaml", "dataset.gen_subset=test",
           f"common_eval.path={ckpt}", f"common_eval.results_path={res}",
           f"override.modalities=[{','.join(mods)}]", f"common.user_dir={HERE}",
           f"override.data={data}", f"override.label_dir={data}", f"generation.beam={BEAM}"]
    if snr is not None:
        cmd += [f"override.noise_wav={noise_dir}", "override.noise_prob=1", f"override.noise_snr={snr}"]
    os.makedirs(res, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(GPU))
    with open(f"{res}/run.log", "w") as log:
        ret = subprocess.call(cmd, cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT)
    if ret != 0:
        print(f"  解碼失敗，請看 {res}/run.log")
        return None
    return read_wer(res)


def plot(results, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ["clean"] + [f"{s} dB" for s in SNRS if s is not None]
    xs = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(7.5, 4.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ends = []
    variants = []
    for exp in PLOT_EXPERIMENTS:
        p = exp.split("_")
        v = p[1] + (" +noise" if len(p) > 3 else "")
        if v not in variants:
            variants.append(v)
    for exp in PLOT_EXPERIMENTS:
        p = exp.split("_")
        size, mod = p[1], p[2]
        variant = size + (" +noise" if len(p) > 3 else "")
        mods = "audio" if mod == "A" else "audio+video"
        ys = [results.get((exp, mods, "clean" if s is None else f"snr{s}")) for s in SNRS]
        pts = [(i, y) for i, y in enumerate(ys) if y is not None]
        if not pts:
            continue
        name = f"{variant} · {mod}"
        ax.plot(*zip(*pts), color=COLOR[mod], linestyle=DASH_POOL[variants.index(variant) % 4], linewidth=2,
                marker="o", markersize=5, markeredgecolor=SURFACE, markeredgewidth=1, label=name)
        ends.append((pts[-1][0], pts[-1][1], name, COLOR[mod]))
    lo, hi = ax.get_ylim()
    gap, prev = (hi - lo) * 0.07, None
    for xe, ye, text, color in sorted(ends, key=lambda t: t[1]):
        ly = ye if prev is None else max(ye, prev + gap)
        prev = ly
        ax.annotate(text, (xe, ye), xytext=(xe + 0.25, ly), textcoords="data", va="center", fontsize=8,
                    color=INK, annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=color, lw=1, shrinkA=0, shrinkB=2),
                    bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec=color, lw=1))
    ax.set_xticks(xs, labels)
    ax.set_title("Test WER under babble noise (lower is better)", loc="left", fontsize=11, color=INK, pad=8)
    ax.set_xlabel("Noise condition (SNR)", fontsize=9, color=MUTED)
    ax.set_ylabel("WER (%)", fontsize=9, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 0.88, 1))
    fig.savefig(f"{out_dir}/wer_vs_snr.png", dpi=200, facecolor=SURFACE)


def main():
    out_dir = x(OUT_DIR)
    os.makedirs(out_dir, exist_ok=True)
    noise_dir = x(NOISE_DIR)
    if not os.path.isfile(f"{noise_dir}/test.tsv"):
        sys.exit(f"找不到 {noise_dir}/test.tsv，請先執行 make_noise.py")

    jobs = []
    for exp in EXPERIMENTS:
        if exp.split("_")[2] == "A":
            jobs += [(exp, ["audio"], s) for s in SNRS]
        else:
            jobs += [(exp, ["audio", "video"], s) for s in SNRS]
            jobs.append((exp, ["video"], None))           # 純讀唇只測一次（不受聲音噪音影響）

    results = {}
    for i, (exp, mods, snr) in enumerate(jobs, 1):
        tag = "clean" if snr is None else f"snr{snr}"
        print(f"[{i}/{len(jobs)}] {exp:14} {'+'.join(mods):12} {tag:7}", end=" ", flush=True)
        wer = decode(exp, mods, snr, noise_dir, out_dir)
        results[(exp, "+".join(mods), tag)] = wer
        print("—" if wer is None else f"WER {wer:.2f}%")

    with open(f"{out_dir}/results.tsv", "w") as f:
        f.write("experiment\tmodality\tcondition\tWER\n")
        for (exp, mods, tag), wer in results.items():
            f.write(f"{exp}\t{mods}\t{tag}\t{'' if wer is None else f'{wer:.2f}'}\n")
    plot(results, out_dir)
    print(f"\n結果表：{out_dir}/results.tsv\n圖：{out_dir}/wer_vs_snr.png")


if __name__ == "__main__":
    main()
