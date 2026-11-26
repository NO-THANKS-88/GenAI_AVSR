#!/usr/bin/env python
"""
把一支說話影片轉成 AV-HuBERT 需要的輸入格式：
  - roi.mp4  : 嘴部區域影片（96x96、25fps）
  - clip.wav : 16kHz 單聲道音檔
  - clip.mp4 : 統一成 25fps 的原始影片（中間檔）

改寫自作者的 avhubert/extract_mouth_ROI.py，修掉寫死的 ffmpeg 路徑與相對路徑。
請放在 SynthAVSR/avhubert/ 資料夾裡執行。

用法：
  conda activate avhubert
  cd ~/GAN/SynthAVSR/avhubert
  python prep_one_clip.py /path/to/test.mp4 --out_dir /path/to/output_dir
"""
import argparse
import os
import shutil
import subprocess
import sys
import wave

import cv2
import dlib
import numpy as np
import skvideo.io
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from preparation.align_mouth import landmarks_interpolate, crop_patch, write_video_ffmpeg  # noqa: E402


def detect_landmark(image, detector, predictor):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    rects = detector(gray, 1)
    coords = None
    for rect in rects:
        shape = predictor(gray, rect)
        coords = np.zeros((68, 2), dtype=np.int32)
        for i in range(68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def main():
    parser = argparse.ArgumentParser(description="影片 -> 嘴部 ROI + 16kHz wav")
    parser.add_argument("input", help="輸入影片（例如 Wav2Lip 生成的 mp4）")
    parser.add_argument("--out_dir", default=None, help="輸出資料夾（預設：與輸入檔同名的資料夾）")
    parser.add_argument("--misc_dir", default=os.path.join(HERE, "data", "misc"),
                        help="放 shape_predictor_68_face_landmarks.dat 和 20words_mean_face.npy 的資料夾")
    args = parser.parse_args()

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        sys.exit("找不到 ffmpeg，請確認已 conda activate avhubert")

    predictor_path = os.path.join(args.misc_dir, "shape_predictor_68_face_landmarks.dat")
    mean_face_path = os.path.join(args.misc_dir, "20words_mean_face.npy")
    for p in (predictor_path, mean_face_path):
        if not os.path.isfile(p):
            sys.exit(f"缺少檔案：{p}")

    in_path = os.path.abspath(args.input)
    out_dir = os.path.abspath(args.out_dir or os.path.splitext(in_path)[0])
    os.makedirs(out_dir, exist_ok=True)
    clip_path = os.path.join(out_dir, "clip.mp4")
    wav_path = os.path.join(out_dir, "clip.wav")
    roi_path = os.path.join(out_dir, "roi.mp4")

    # 1. 統一成 25fps（AV-HuBERT 預設影像 25fps、音訊 16kHz）
    print("[1/4] 轉成 25fps ...")
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", in_path, "-r", "25", clip_path], check=True)

    # 2. 抽出 16kHz 單聲道 wav
    print("[2/4] 抽取 16kHz 音訊 ...")
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", clip_path,
                    "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", wav_path], check=True)

    # 3. 逐格偵測臉部特徵點
    print("[3/4] 偵測臉部特徵點 ...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    frames = skvideo.io.vread(clip_path)
    landmarks = [detect_landmark(f, detector, predictor) for f in tqdm(frames)]
    n_missing = sum(l is None for l in landmarks)
    if n_missing == len(landmarks):
        sys.exit("整支影片都偵測不到臉，請換一張更清楚的正臉照片")
    if n_missing:
        print(f"  注意：{n_missing}/{len(landmarks)} 格沒偵測到臉，會用前後格內插補上")

    # 4. 對齊並裁出 96x96 嘴部區域
    print("[4/4] 裁切嘴部區域 ...")
    mean_face = np.load(mean_face_path)
    rois = crop_patch(clip_path, landmarks_interpolate(landmarks), mean_face, [33, 36, 39, 42, 45],
                      (256, 256), window_margin=12, start_idx=48, stop_idx=68,
                      crop_height=96, crop_width=96)
    if rois is None:
        sys.exit("嘴部裁切失敗")
    write_video_ffmpeg(rois, roi_path, ffmpeg)

    # 檢查：影像格數與音訊長度應該對得上（16000 / 25 = 640 個取樣點對一格）
    n_video = int(cv2.VideoCapture(roi_path).get(cv2.CAP_PROP_FRAME_COUNT))
    with wave.open(wav_path) as w:
        n_audio = w.getnframes()
    print("\n完成！")
    print(f"  嘴部影片：{roi_path}（{n_video} 格）")
    print(f"  音訊：    {wav_path}（{n_audio} 個取樣點，約 {n_audio / 640:.1f} 格）")
    if abs(n_audio / 640 - n_video) > 10:
        print("  警告：影像和音訊長度差距偏大，請檢查輸入影片")


if __name__ == "__main__":
    main()
