#!/usr/bin/env python
"""
產生「訓練用」與「測試用」兩套 babble 噪音，兩套來自不同的說話者，避免噪音洩題。

做法：把 train_30h 的說話者分成兩組（交錯分配），
  A 組的錄音 → 訓練用噪音（加噪音微調時用）
  B 組的錄音 → 測試用噪音（run_eval.py 評估時用）
每個噪音檔 20 秒，由 BABBLE_SPEAKERS 句不同的話疊在一起（16kHz、int16，AV-HuBERT 讀得到的格式）。

用法：
  conda activate avhubert
  cd ~/GAN/SynthAVSR/avhubert
  python make_noise.py
輸出：
  <OUT_DIR>/train_babble/{train,valid}.tsv 與 wav 檔
  <OUT_DIR>/test_babble/{test,valid}.tsv 與 wav 檔
  <OUT_DIR>/speaker_groups.txt  兩組各有哪些說話者（寫報告用）
"""
import os
import random
import wave

import numpy as np

# ======================== 設定區（只要改這裡） ========================
LIST = "~/GAN/data/lists_v2/train_30h.tsv"      # make_subset.py 的清單（取說話者）
MANIFEST = "~/GAN/data/tsou/avhubert/train_30h/train.tsv"   # 實際訓練用的清單（取音檔路徑，保證存在）
OUT_DIR = "~/GAN/exp/noise"
NUM_TRAIN = 200                               # 訓練用噪音檔數量
NUM_TEST = 50                                 # 測試用噪音檔數量
BABBLE_SPEAKERS = 6
SECONDS = 20
SEED = 1234
# =====================================================================


def x(p):
    return os.path.abspath(os.path.expanduser(p))


def read_wav(path):
    with wave.open(path) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)


def build(ids, n, out_dir, splits, rng, audio):
    os.makedirs(out_dir, exist_ok=True)
    length, paths = 16000 * SECONDS, []
    for i in range(n):
        mix = np.zeros(length, np.float32)
        for uid in rng.sample(ids, BABBLE_SPEAKERS):
            w = read_wav(audio[uid])
            w = np.tile(w, int(np.ceil(length / max(len(w), 1))))[:length]
            mix += w / (np.sqrt(np.mean(w ** 2)) + 1e-8)
        mix = mix / (np.abs(mix).max() + 1e-8) * 0.5 * 32767
        p = f"{out_dir}/babble_{i:03d}.wav"
        with wave.open(p, "w") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(mix.astype(np.int16).tobytes())
        paths.append(p)
    for s in splits:
        with open(f"{out_dir}/{s}.tsv", "w") as f:
            f.write("\n".join(paths) + "\n")
    print(f"  {out_dir}：{n} 個噪音檔")


def main():
    rng = random.Random(SEED)
    audio = {}                                   # 句子編號 → 訓練時實際用的音檔
    with open(x(MANIFEST)) as f:
        f.readline()                             # 第一行是根目錄
        for line in f:
            if line.strip():
                cols = line.rstrip("\n").split("\t")
                if os.path.isfile(cols[2]):
                    audio[cols[0]] = cols[2]
    by_spk, missing = {}, []
    with open(x(LIST), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                uid, _, spk = line.split("\t")[:3]
                if uid in audio:
                    by_spk.setdefault(spk, []).append(uid)
                else:
                    missing.append(uid)
    print(f"可用句子 {len(audio)} 句；清單裡有 {len(missing)} 句在訓練清單中找不到音檔（略過）"
          + (f"，例如 {missing[:3]}" if missing else ""))
    # 依句數排序後交錯分組，兩組的資料量才會接近
    spks = sorted(by_spk, key=lambda s: -len(by_spk[s]))
    group_a, group_b = spks[0::2], spks[1::2]
    ids_a = [u for s in group_a for u in by_spk[s]]
    ids_b = [u for s in group_b for u in by_spk[s]]

    out = x(OUT_DIR)
    os.makedirs(out, exist_ok=True)
    with open(f"{out}/speaker_groups.txt", "w") as f:
        f.write(f"訓練用噪音（A 組，{len(ids_a)} 句）：{' '.join(group_a)}\n")
        f.write(f"測試用噪音（B 組，{len(ids_b)} 句）：{' '.join(group_b)}\n")
    print(f"A 組 {len(group_a)} 位說話者 / B 組 {len(group_b)} 位說話者")
    build(ids_a, NUM_TRAIN, f"{out}/train_babble", ["train", "valid"], rng, audio)
    build(ids_b, NUM_TEST, f"{out}/test_babble", ["test", "valid"], rng, audio)
    print(f"分組紀錄：{out}/speaker_groups.txt")


if __name__ == "__main__":
    main()
