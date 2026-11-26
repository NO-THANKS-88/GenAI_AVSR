#!/usr/bin/env python
"""
拆解 run_eval.py 的辨識結果：WER 由哪種錯誤組成（替換/刪除/插入）、CER、輸出長度，
並找出「同一段話一直重複」的幻覺輸出。WER 超過 100% 時，幾乎都是插入錯誤太多造成的。

用法：
  conda activate avhubert
  cd ~/GAN/SynthAVSR/avhubert
  python analyze_errors.py
輸出：<EVAL_DIR>/error_breakdown.tsv，並在終端機印出每個條件最糟的例子
"""
import glob
import json
import os

import jiwer

# ======================== 設定區（只要改這裡） ========================
EVAL_DIR = "~/GAN/exp/eval"
SHOW_EXAMPLES = 2           # 每個條件印出幾句最糟的例子（0 = 不印）
# =====================================================================


def repeats(text):
    """同一個 1～3 字片段連續重複（單字 4 次以上，2～3 字片段 3 次以上），視為重複幻覺。"""
    w = text.split()
    for n, times in ((1, 4), (2, 3), (3, 3)):
        for i in range(len(w) - n * times + 1):
            seg = w[i:i + n]
            if all(w[i + k * n:i + (k + 1) * n] == seg for k in range(1, times)):
                return True
    return False


def main():
    root = os.path.expanduser(EVAL_DIR)
    files = sorted(glob.glob(f"{root}/decode/*/*/*/hypo-*.json"))
    if not files:
        raise SystemExit(f"在 {root}/decode 找不到 hypo-*.json")
    rows = []
    for fn in files:
        exp, mods, cond = fn.split(os.sep)[-4:-1]
        d = json.load(open(fn, encoding="utf-8"))
        refs, hyps = d["ref"], [h if h.strip() else "<empty>" for h in d["hypo"]]
        o = jiwer.process_words(refs, hyps)
        n_ref = sum(len(r.split()) for r in refs)
        rows.append(dict(
            exp=exp, mods=mods, cond=cond, wer=100 * o.wer,
            sub=100 * o.substitutions / n_ref, dele=100 * o.deletions / n_ref, ins=100 * o.insertions / n_ref,
            cer=100 * jiwer.cer(refs, hyps),
            len_ratio=sum(len(h.split()) for h in hyps) / n_ref,
            rep=100 * sum(repeats(h) for h in hyps) / len(hyps)))
        if SHOW_EXAMPLES:
            worst = sorted(zip(refs, hyps), key=lambda p: len(p[1].split()) - len(p[0].split()), reverse=True)
            print(f"\n=== {exp} / {mods} / {cond} ===")
            for r, h in worst[:SHOW_EXAMPLES]:
                print(f"  REF: {r}\n  HYP: {h[:200]}{' ...' if len(h) > 200 else ''}")

    order = {"clean": 0, "snr10": 1, "snr5": 2, "snr0": 3, "snr-5": 4}
    rows.sort(key=lambda r: (r["exp"], r["mods"], order.get(r["cond"], 9)))
    out = f"{root}/error_breakdown.tsv"
    keys = ["wer", "sub", "dele", "ins", "cer", "len_ratio", "rep"]
    with open(out, "w") as f:
        f.write("experiment\tmodality\tcondition\tWER\tsub%\tdel%\tins%\tCER\thyp/ref長度比\t重複幻覺%\n")
        for r in rows:
            f.write("\t".join([r["exp"], r["mods"], r["cond"]] + [f"{r[k]:.2f}" for k in keys]) + "\n")
    print(f"\n{'實驗':14} {'模態':12} {'條件':6} {'WER':>6} {'替換':>6} {'刪除':>6} {'插入':>6} {'CER':>6} {'長度比':>6} {'重複%':>6}")
    for r in rows:
        print(f"{r['exp']:14} {r['mods']:12} {r['cond']:6} " + " ".join(f"{r[k]:6.1f}" for k in keys))
    print(f"\n已存成 {out}")


if __name__ == "__main__":
    main()
