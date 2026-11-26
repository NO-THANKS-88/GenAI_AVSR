#!/usr/bin/env python
"""
檢查「實際拿去訓練／評估」的 AV-HuBERT 清單：資料量、說話者、有沒有洩題，
並和 data/lists/ 的切分清單比對，找出兩者為什麼對不上。

用法：
  conda activate avhubert
  python check_manifest.py
"""
import os
import re

# ======================== 設定區（只要改這裡） ========================
MANIFEST_ROOT = "~/GAN/data/tsou/avhubert"
SETS = ["train_10h", "train_30h"]
LISTS_DIR = "~/GAN/data/lists_v2"
# =====================================================================


def x(p):
    return os.path.expanduser(p)


def spk_of(uid):
    m = re.search(r"_(c\d+)(?:_\d+)?$", uid)
    return m.group(1) if m else "unk"


def load(prefix):
    with open(prefix + ".tsv") as f:
        f.readline()
        rows = [l.rstrip("\n").split("\t") for l in f if l.strip()]
    texts = [l.rstrip("\n") for l in open(prefix + ".wrd", encoding="utf-8")]
    return [(r[0], int(r[4]) / 16000, t) for r, t in zip(rows, texts)]


def lst(name):
    p = f"{x(LISTS_DIR)}/{name}.tsv"
    if not os.path.isfile(p):
        return {}
    return {l.split("\t")[0]: l for l in open(p, encoding="utf-8") if l.strip()}


def main():
    for s in SETS:
        print(f"\n===== {s} =====")
        d = {sp: load(f"{x(MANIFEST_ROOT)}/{s}/{sp}") for sp in ("train", "valid", "test")}
        for sp, rows in d.items():
            spks = sorted({spk_of(u) for u, _, _ in rows})
            print(f"  {sp:5} {len(rows):6} 句 {sum(h for _, h, _ in rows) / 3600:6.2f} 小時  "
                  f"說話者 {len(spks)} 位：{' '.join(spks) if len(spks) <= 6 else ' '.join(spks[:6]) + ' ...'}")
        tr = d["train"]
        for other in ("valid", "test"):
            spk_overlap = {spk_of(u) for u, _, _ in tr} & {spk_of(u) for u, _, _ in d[other]} - {"unk"}
            txt_overlap = {t for _, _, t in tr} & {t for _, _, t in d[other]}
            ok = "✅" if not spk_overlap and not txt_overlap else "❌"
            print(f"  {ok} train vs {other}：重疊說話者 {sorted(spk_overlap) or '無'}，重疊句子 {len(txt_overlap)} 句")

        for sp, name in (("train", s), ("valid", "valid"), ("test", "test")):
            ids = {u for u, _, _ in d[sp]}
            ref = lst(name)
            if ref:
                print(f"  清單比對 {sp}：訓練清單 {len(ids)} 句 / data/lists/{name}.tsv {len(ref)} 句 / "
                      f"兩邊都有 {len(ids & set(ref))} 句")


if __name__ == "__main__":
    main()
