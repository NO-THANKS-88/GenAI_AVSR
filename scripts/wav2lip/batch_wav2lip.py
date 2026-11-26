#!/usr/bin/env python
"""
Step 2：批次用 Wav2Lip 生成說話影片（模型只載入一次，比逐句呼叫 inference.py 快很多）。

用法：放在 Wav2Lip 資料夾裡，改好下面的「設定區」，然後
  conda activate wav2lip
  cd ~/GAN/SynthAVSR/Wav2Lip
  python batch_wav2lip.py

  - 第一次執行會自動建立「說話者 → 人臉」對照表 face_map.tsv（之後想換臉可以直接改這個檔）
  - GPUS 填多張時，會自動開多個程序分工，進度每 30 秒顯示一次，詳細紀錄在 logs/ 資料夾
  - 已經生成過的句子會自動跳過，中斷後重跑即可續跑

輸出
  <OUT_DIR>/face_map.tsv     說話者 → 人臉照片
  <OUT_DIR>/audio/<id>.wav   16kHz 單聲道音檔（之後 AV-HuBERT 用這個，不用影片裡的音軌）
  <OUT_DIR>/video/<id>.mp4   生成的說話影片（25fps，無音軌）
  <OUT_DIR>/errors_<rank>.tsv 失敗的句子與原因
"""
import glob
import os
import subprocess
import sys
import time

import cv2
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))

# ======================== 設定區（只要改這裡） ========================
LISTS = [                                   # make_subset.py 產生的清單（train_30h 已包含 train_10h）
    "~/GAN/data/lists_v2/train_30h.tsv",
    "~/GAN/data/lists_v2/valid.tsv",
    "~/GAN/data/lists_v2/test.tsv",
]
OUT_DIR = "~/GAN/data/tsou/synth"           # 輸出資料夾
FACES_DIR = "~/GAN/faces"                   # select_faces.py 選出的人臉照片
MAP_BASE_LISTS = [                          # 第一次生成影片時用的清單；face_map.tsv 被刪掉時用來重建原本的對照
    "~/GAN/data/lists/train_30h.tsv",
    "~/GAN/data/lists/valid.tsv",
    "~/GAN/data/lists/test.tsv",
]
CHECKPOINT = os.path.join(HERE, "checkpoints", "wav2lip_gan.pth")
GPUS = [8,9]                                  # 要用哪幾張 GPU，例如 [3, 5]；先用 nvidia-smi 看哪張空
MODE = "run"                               # "test"：只跑前 TEST_LIMIT 句測速度；"run"：全部跑
TEST_LIMIT = 20
BATCH_SIZE = 128
# =====================================================================
sys.path.insert(0, HERE)
import audio  # noqa: E402  (Wav2Lip 的 audio.py)
import face_detection  # noqa: E402
from models import Wav2Lip  # noqa: E402

FPS = 25
IMG = 96
MEL_STEP = 16
PADS = (0, 10, 0, 0)  # 上、下、左、右；和 inference.py 預設相同（下方多留 10 px 包住下巴）


def read_lists(paths):
    rows, seen = [], set()
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                uid, wav, spk = line.rstrip("\n").split("\t")[:3]
                if uid not in seen:          # train_10h 包含在 train_30h 裡，避免重複
                    seen.add(uid)
                    rows.append((uid, wav, spk))
    return rows


def make_map(rows, faces_dir, out_dir):
    """建立或補齊「說話者 → 人臉」對照表。已經有的說話者絕不更動（已生成的影片才會一致），
    新出現的說話者分配還沒被用過的照片。"""
    faces = sorted(glob.glob(os.path.join(faces_dir, "face_*.jpg")))   # 只取 select_faces.py 選出的照片，不含 contact_sheet.jpg
    if not faces:
        sys.exit(f"{faces_dir} 裡沒有圖片")
    faces = [os.path.abspath(f) for f in faces]
    path = os.path.join(out_dir, "face_map.tsv")
    if os.path.isfile(path):
        with open(path) as f:
            fmap = dict(line.rstrip("\n").split("\t") for line in f if line.strip())
    else:
        # 沒有對照表時，先用「最早生成影片時的清單」重建原本的對照（同樣的排序規則，結果會一模一樣）
        base = [os.path.expanduser(p) for p in MAP_BASE_LISTS]
        base_spks = sorted({spk for _, _, spk in read_lists(base)}) if all(map(os.path.isfile, base)) else []
        fmap = {s: faces[i % len(faces)] for i, s in enumerate(base_spks)}
        if base_spks:
            print(f"依 MAP_BASE_LISTS 重建原本的對照表（{len(base_spks)} 位說話者）")
    new = sorted({spk for _, _, spk in rows} - set(fmap))
    unused = [f for f in faces if f not in set(fmap.values())]
    for i, s in enumerate(new):
        if i < len(unused):
            fmap[s] = unused[i]
        else:
            fmap[s] = faces[(len(fmap)) % len(faces)]
            print(f"警告：照片不夠，{s} 會和其他說話者共用同一張臉")
    with open(path, "w") as f:
        for s in sorted(fmap):
            f.write(f"{s}\t{fmap[s]}\n")
    print(f"對照表 {path}：共 {len(fmap)} 位說話者" + (f"，新增 {new}" if new else ""))


def load_map(out_dir):
    path = os.path.join(out_dir, "face_map.tsv")
    if not os.path.isfile(path):
        sys.exit(f"找不到 {path}，請刪掉 face_map.tsv 後重新執行，會自動重建")
    with open(path) as f:
        return dict(line.rstrip("\n").split("\t") for line in f if line.strip())


def load_model(ckpt, device):
    try:
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
    except TypeError:                      # 很舊的 torch 沒有 weights_only 參數
        state = torch.load(ckpt, map_location="cpu")
    model = Wav2Lip()
    model.load_state_dict({k.replace("module.", ""): v for k, v in state["state_dict"].items()})
    return model.to(device).eval()


def prepare_face(path, detector, device):
    """每張臉只偵測一次，之後每句都重複使用。"""
    frame = cv2.imread(path)
    if frame is None:
        raise RuntimeError(f"讀不到人臉圖片 {path}")
    if frame.shape[0] % 2 or frame.shape[1] % 2:     # libx264 需要偶數寬高
        frame = frame[: frame.shape[0] // 2 * 2, : frame.shape[1] // 2 * 2]
    rect = detector.get_detections_for_batch(np.array([frame]))[0]
    if rect is None:
        raise RuntimeError(f"Wav2Lip 的人臉偵測器在 {path} 找不到臉，請換一張")
    h, w = frame.shape[:2]
    y1, y2 = max(0, rect[1] - PADS[0]), min(h, rect[3] + PADS[1])
    x1, x2 = max(0, rect[0] - PADS[2]), min(w, rect[2] + PADS[3])
    face = cv2.resize(frame[y1:y2, x1:x2], (IMG, IMG))
    masked = face.copy()
    masked[IMG // 2:] = 0                            # 遮住下半臉，讓模型重畫嘴巴
    inp = np.concatenate([masked, face], axis=2) / 255.0
    inp = torch.FloatTensor(inp.transpose(2, 0, 1))[None].to(device)
    return frame, (y1, y2, x1, x2), inp


def mel_chunks(wav):
    mel = audio.melspectrogram(wav)
    if np.isnan(mel).any():                          # 完全靜音或 TTS 音檔可能出現 NaN
        wav = wav + 1e-5 * np.random.randn(len(wav))
        mel = audio.melspectrogram(wav)
    chunks, i, step = [], 0, 80.0 / FPS
    while True:
        s = int(i * step)
        if s + MEL_STEP > mel.shape[1]:
            chunks.append(mel[:, mel.shape[1] - MEL_STEP:])
            break
        chunks.append(mel[:, s:s + MEL_STEP])
        i += 1
    return chunks


def generate(model, face, wav, out_mp4, device, bs):
    frame, (y1, y2, x1, x2), inp = face
    chunks = mel_chunks(wav)
    h, w = frame.shape[:2]
    tmp = out_mp4[:-4] + ".tmp.mp4"
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", tmp],
        stdin=subprocess.PIPE)
    try:
        for b in range(0, len(chunks), bs):
            mel = torch.FloatTensor(np.asarray(chunks[b:b + bs]))[:, None].to(device)   # (B,1,80,16)
            with torch.no_grad():
                pred = model(mel, inp.expand(mel.shape[0], -1, -1, -1))
            pred = (pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0).clip(0, 255).astype(np.uint8)
            for p in pred:
                f = frame.copy()
                f[y1:y2, x1:x2] = cv2.resize(p, (x2 - x1, y2 - y1))
                ff.stdin.write(f.tobytes())
    finally:
        ff.stdin.close()
        ret = ff.wait()
    if ret != 0:
        raise RuntimeError("ffmpeg 編碼失敗")
    os.replace(tmp, out_mp4)
    return len(chunks)


def worker(rank, nshard, limit):
    """實際生成影片的程序（每張 GPU 一個）。"""
    out_dir = os.path.expanduser(OUT_DIR)
    rows = read_lists([os.path.expanduser(p) for p in LISTS])[rank::nshard]
    if limit:
        rows = rows[:limit]
    face_map = load_map(out_dir)
    vdir, adir = os.path.join(out_dir, "video"), os.path.join(out_dir, "audio")
    os.makedirs(vdir, exist_ok=True)
    os.makedirs(adir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[rank {rank}/{nshard}] 共 {len(rows)} 句，使用 {device}", flush=True)
    model = load_model(CHECKPOINT, device)
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device=device)
    faces = {p: prepare_face(p, detector, device) for p in sorted(set(face_map[spk] for _, _, spk in rows))}
    del detector

    n_done = n_skip = n_err = 0
    err_path = os.path.join(out_dir, f"errors_{rank}.tsv")
    t0 = time.time()
    with open(err_path, "a", encoding="utf-8") as errf:
        for uid, src, spk in tqdm(rows, mininterval=10):
            out_mp4, out_wav = os.path.join(vdir, uid + ".mp4"), os.path.join(adir, uid + ".wav")
            if os.path.isfile(out_mp4) and os.path.isfile(out_wav):
                n_skip += 1
                continue
            try:
                wav = audio.load_wav(src, 16000)
                sf.write(out_wav, wav, 16000, subtype="PCM_16")
                generate(model, faces[face_map[spk]], wav, out_mp4, device, BATCH_SIZE)
                n_done += 1
            except Exception as e:                       # 單句失敗不影響其他句
                n_err += 1
                errf.write(f"{uid}\t{src}\t{type(e).__name__}: {e}\n")
                errf.flush()
    dt = time.time() - t0
    print(f"[rank {rank}] 完成 {n_done} 句，跳過（已存在）{n_skip} 句，失敗 {n_err} 句，"
          f"耗時 {dt / 60:.1f} 分鐘" + (f"，平均每句 {dt / n_done:.2f} 秒" if n_done else "") +
          (f"；失敗原因見 {err_path}" if n_err else ""), flush=True)


def main():
    out_dir = os.path.expanduser(OUT_DIR)
    os.makedirs(out_dir, exist_ok=True)
    rows = read_lists([os.path.expanduser(p) for p in LISTS])

    # 1. 對照表：沒有就建立，有新說話者就補上（舊的對照不會變）
    if (not os.path.isfile(os.path.join(out_dir, "face_map.tsv"))
            or {spk for _, _, spk in rows} - set(load_map(out_dir))):
        make_map(rows, os.path.expanduser(FACES_DIR), out_dir)

    # 2. 依 MODE 分派工作
    if MODE == "test":
        gpus, limit = GPUS[:1], TEST_LIMIT
        print(f"測試模式：用 GPU {gpus[0]} 跑前 {limit} 句")
    elif MODE == "run":
        gpus, limit = GPUS, 0
        print(f"正式執行：共 {len(rows)} 句，用 GPU {gpus} 分工")
    else:
        sys.exit('MODE 只能是 "test" 或 "run"')

    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    procs = []
    for rank, gpu in enumerate(gpus):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        cmd = [sys.executable, os.path.abspath(__file__), "--_worker", str(rank), str(len(gpus)), str(limit)]
        if len(gpus) == 1:
            procs.append(subprocess.Popen(cmd, env=env))                 # 單張 GPU：直接顯示在終端機
        else:
            log = open(os.path.join(log_dir, f"rank{rank}_gpu{gpu}.log"), "a")
            procs.append(subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT))

    # 3. 多張 GPU 時，每 30 秒回報一次整體進度
    total = min(len(rows), limit) if limit else len(rows)
    vdir = os.path.join(out_dir, "video")
    while any(p.poll() is None for p in procs):
        if len(gpus) > 1:
            n = len(glob.glob(os.path.join(vdir, "*.mp4"))) - len(glob.glob(os.path.join(vdir, "*.tmp.mp4")))
            print(f"  進度：{n}/{total} 支影片（{time.strftime('%H:%M:%S')}）", flush=True)
        time.sleep(30)
    codes = [p.returncode for p in procs]
    print("全部結束" if all(c == 0 for c in codes) else f"有程序異常結束（代碼 {codes}），請看 {log_dir}")
    if len(gpus) > 1:
        os.system(f"tail -n 1 {log_dir}/*.log")


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--_worker":
        worker(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    else:
        main()
