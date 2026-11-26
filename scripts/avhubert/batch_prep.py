#!/usr/bin/env python
"""
Step 3：把 Wav2Lip 生成的影片批次轉成 AV-HuBERT 的訓練資料。

  (1) 平行裁出 96x96 嘴部影片（dlib 偵測特徵點 → 對齊 → 裁切，邏輯同 prep_one_clip.py）
  (2) 用訓練集文字訓練 SentencePiece tokenizer，產生 dict.wrd.txt
  (3) 每個訓練集大小各產生一份 train/valid/test 的 .tsv 與 .wrd

用法：放在 SynthAVSR/avhubert/ 裡，改好「設定區」後
  conda activate avhubert
  cd ~/GAN/SynthAVSR/avhubert
  python batch_prep.py

輸出（OUT_DIR 底下）
  roi/<id>.mp4                 嘴部影片
  nframes.tsv                  每句的影像格數、音訊取樣數
  spm<V>/spm_unigram<V>.model  tokenizer
  train_10h/ train_30h/        各自的 train/valid/test.{tsv,wrd} 與 dict.wrd.txt（微調時 task.data 指到這裡）
  errors.tsv                   處理失敗的句子
已經裁好的嘴部影片會自動跳過，中斷後重跑即可續跑。
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")       # 多程序時避免每個程序又開一堆執行緒

import subprocess
import sys
import time
import wave
from multiprocessing import Pool

import cv2
import dlib
import numpy as np
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from preparation.align_mouth import landmarks_interpolate, crop_patch  # noqa: E402

# ======================== 設定區（只要改這裡） ========================
LISTS_DIR = "~/GAN/data/lists_v2"              # make_subset.py 的輸出
TRAIN_SETS = ["train_10h", "train_30h"]     # 要準備哪些訓練集大小
SYNTH_DIR = "~/GAN/data/tsou/synth"         # batch_wav2lip.py 的輸出（video/、audio/）
OUT_DIR = "~/GAN/data/tsou/avhubert"        # 輸出資料夾
MISC_DIR = os.path.join(HERE, "data", "misc")
WORKERS = 48                                # 平行程序數（伺服器 128 核，32 左右不會影響別人太多）
VOCAB_SIZE = 500                            # SentencePiece 詞彙量（作者用 1000；鄒語資料較少，先用 500）
MODE = "run"                               # "test"：只裁前 TEST_LIMIT 支測速度；"run"：全部
TEST_LIMIT = 20
# =====================================================================

FFMPEG = "ffmpeg"
_det = _pred = _mean_face = None


def x(p):
    return os.path.abspath(os.path.expanduser(p))


def read_list(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                uid, _, spk, dur, text = line.rstrip("\n").split("\t")
                rows.append((uid, text))
    return rows


# ---------------- (1) 嘴部裁切 ----------------
def init_worker():
    global _det, _pred, _mean_face
    cv2.setNumThreads(1)
    _det = dlib.get_frontal_face_detector()
    _pred = dlib.shape_predictor(os.path.join(MISC_DIR, "shape_predictor_68_face_landmarks.dat"))
    _mean_face = np.load(os.path.join(MISC_DIR, "20words_mean_face.npy"))


def landmarks_of(gray):
    rects = _det(gray, 0) or _det(gray, 1)          # 先不放大（快 ~4 倍），抓不到才放大再試
    if not rects:
        return None
    shape = _pred(gray, rects[0])
    return np.array([(p.x, p.y) for p in shape.parts()], dtype=np.int32)


def count_audio(wav_path):
    with wave.open(wav_path) as w:
        return w.getnframes()


def process(uid):
    synth, out = x(SYNTH_DIR), x(OUT_DIR)
    video, wav = f"{synth}/video/{uid}.mp4", f"{synth}/audio/{uid}.wav"
    roi = f"{out}/roi/{uid}.mp4"
    try:
        if not (os.path.isfile(video) and os.path.isfile(wav)):
            return uid, None, None, "影片或音檔不存在（Wav2Lip 還沒生成？）"
        if os.path.isfile(roi) and os.path.getsize(roi) > 0:
            n_v = int(cv2.VideoCapture(roi).get(cv2.CAP_PROP_FRAME_COUNT))
            return uid, n_v, count_audio(wav), None

        cap, lms = cv2.VideoCapture(video), []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            lms.append(landmarks_of(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
        cap.release()
        if not lms or all(l is None for l in lms):
            return uid, None, None, "偵測不到臉"
        rois = crop_patch(video, landmarks_interpolate(lms), _mean_face, [33, 36, 39, 42, 45], (256, 256),
                          window_margin=12, start_idx=48, stop_idx=68, crop_height=96, crop_width=96)
        if rois is None or len(rois) == 0:
            return uid, None, None, "嘴部裁切失敗"

        tmp = roi[:-4] + ".tmp.mp4"
        p = subprocess.Popen([FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                              "-s", "96x96", "-r", "25", "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
                              "-crf", "18", "-pix_fmt", "yuv420p", "-threads", "1", tmp], stdin=subprocess.PIPE)
        p.stdin.write(np.ascontiguousarray(rois, dtype=np.uint8).tobytes())
        p.stdin.close()
        if p.wait() != 0:
            return uid, None, None, "ffmpeg 編碼失敗"
        os.replace(tmp, roi)
        return uid, len(rois), count_audio(wav), None
    except Exception as e:
        return uid, None, None, f"{type(e).__name__}: {e}"


# ---------------- (2) tokenizer ----------------
def train_spm(texts, out_dir, vocab_size):
    import sentencepiece as spm
    os.makedirs(out_dir, exist_ok=True)
    prefix = os.path.join(out_dir, f"spm_unigram{vocab_size}")
    txt = prefix + ".input.txt"
    with open(txt, "w", encoding="utf-8") as f:
        f.write("\n".join(texts) + "\n")
    spm.SentencePieceTrainer.Train(
        f"--input={txt} --model_prefix={prefix} --model_type=unigram --vocab_size={vocab_size} "
        f"--character_coverage=1.0 --unk_id=3 --bos_id=0 --eos_id=2 --pad_id=1 --num_threads=16")
    sp = spm.SentencePieceProcessor()
    sp.Load(prefix + ".model")
    with open(prefix + ".txt", "w", encoding="utf-8") as f:          # fairseq 格式的詞典
        for i in range(sp.GetPieceSize()):
            piece = sp.IdToPiece(i)
            if piece not in {"<unk>", "<s>", "</s>", "<pad>"}:
                f.write(f"{piece} 1\n")
    return prefix + ".model", prefix + ".txt"


# ---------------- (3) manifest ----------------
def write_split(path_prefix, rows, info, out):
    n = 0
    with open(path_prefix + ".tsv", "w") as ft, open(path_prefix + ".wrd", "w", encoding="utf-8") as fw:
        ft.write("/\n")
        for uid, text in rows:
            if uid not in info:
                continue
            n_v, n_a = info[uid]
            ft.write(f"{uid}\t{out}/roi/{uid}.mp4\t{x(SYNTH_DIR)}/audio/{uid}.wav\t{n_v}\t{n_a}\n")
            fw.write(text + "\n")
            n += 1
    return n


def main():
    lists_dir, out = x(LISTS_DIR), x(OUT_DIR)
    os.makedirs(f"{out}/roi", exist_ok=True)
    for f in ("shape_predictor_68_face_landmarks.dat", "20words_mean_face.npy"):
        if not os.path.isfile(os.path.join(MISC_DIR, f)):
            sys.exit(f"缺少 {os.path.join(MISC_DIR, f)}")

    splits = {name: read_list(f"{lists_dir}/{name}.tsv") for name in TRAIN_SETS + ["valid", "test"]}
    all_ids = list(dict.fromkeys(uid for rows in splits.values() for uid, _ in rows))
    if MODE == "test":
        all_ids = all_ids[:TEST_LIMIT]
    elif MODE != "run":
        sys.exit('MODE 只能是 "test" 或 "run"')

    # (1) 平行裁切
    print(f"[1/3] 裁切嘴部：{len(all_ids)} 支影片，{WORKERS} 個程序")
    t0, info, errors = time.time(), {}, []
    with Pool(WORKERS, initializer=init_worker) as pool:
        for uid, n_v, n_a, err in tqdm(pool.imap_unordered(process, all_ids, chunksize=4),
                                       total=len(all_ids), mininterval=10):
            if err:
                errors.append((uid, err))
            else:
                info[uid] = (n_v, n_a)
    dt = time.time() - t0
    with open(f"{out}/errors.tsv", "w", encoding="utf-8") as f:
        f.writelines(f"{u}\t{e}\n" for u, e in errors)
    with open(f"{out}/nframes.tsv", "w") as f:
        f.writelines(f"{u}\t{v}\t{a}\n" for u, (v, a) in info.items())
    print(f"  成功 {len(info)} 支，失敗 {len(errors)} 支，耗時 {dt / 60:.1f} 分鐘"
          + (f"（失敗原因見 {out}/errors.tsv）" if errors else ""))
    bad = [u for u, (v, a) in info.items() if abs(a / 640 - v) > 10]
    if bad:
        print(f"  注意：{len(bad)} 支影像與音訊長度差超過 10 格，例如 {bad[:3]}")

    if MODE == "test":
        print(f"\n測試模式結束。換算全部 {len(set(u for r in splits.values() for u, _ in r))} 支約需 "
              f"{dt / max(len(all_ids), 1) * len(set(u for r in splits.values() for u, _ in r)) / 60:.0f} 分鐘"
              f"（以 {WORKERS} 個程序計）。可以到 {out}/roi/ 看看裁切結果。")
        return

    # (2) tokenizer：用最大的訓練集文字訓練，所有訓練集大小共用同一個，方便比較
    print(f"[2/3] 訓練 SentencePiece（詞彙量 {VOCAB_SIZE}）")
    biggest = max(TRAIN_SETS, key=lambda n: len(splits[n]))
    spm_model, dict_txt = train_spm([t for _, t in splits[biggest]], f"{out}/spm{VOCAB_SIZE}", VOCAB_SIZE)

    # (3) manifest
    print("[3/3] 產生訓練清單")
    for name in TRAIN_SETS:
        d = f"{out}/{name}"
        os.makedirs(d, exist_ok=True)
        counts = {s: write_split(f"{d}/{s}", splits[src], info, out)
                  for s, src in (("train", name), ("valid", "valid"), ("test", "test"))}
        with open(dict_txt, encoding="utf-8") as fi, open(f"{d}/dict.wrd.txt", "w", encoding="utf-8") as fo:
            fo.write(fi.read())
        print(f"  {d}：train {counts['train']}、valid {counts['valid']}、test {counts['test']} 句")
    print(f"\n完成！tokenizer：{spm_model}")


if __name__ == "__main__":
    main()
