#!/usr/bin/env python
"""
用作者的 AV-HuBERT 模型辨識一支影片（prep_one_clip.py 的輸出）。
詞典和 tokenizer 直接從模型檔裡的 task_state 取用，不需要作者伺服器上的檔案。

用法：
  conda activate avhubert
  cd ~/GAN/SynthAVSR/avhubert
  python infer_one_clip.py --clip_dir ~/GAN/test_clip \
      --ckpt ~/GAN/SynthAVSR/checkpoints/CAT-AVSR_AV.pt --mode all
"""
import argparse
import os
import sys
import tempfile
import warnings
import wave
from argparse import Namespace

import cv2
import torch
from fairseq import checkpoint_utils, utils
from fairseq.dataclass.configs import GenerationConfig
from omegaconf import DictConfig, OmegaConf, open_dict

HERE = os.path.dirname(os.path.abspath(__file__))
MODES = {"av": ["audio", "video"], "v": ["video"], "a": ["audio"]}
NAMES = {"av": "影音 AVSR", "v": "純影像 VSR", "a": "純音訊 ASR"}


def fix_missing(node, path="cfg"):
    """模型檔的設定裡有些欄位是 '???'（未填值），新版 fairseq 讀到會報錯，這裡先補上。"""
    if not isinstance(node, DictConfig):
        return
    with open_dict(node):
        for k in list(node.keys()):
            if OmegaConf.is_missing(node, k):
                if k == "input_modality":      # AV-HuBERT 的佔位欄位，填任意字串即可
                    node[k] = "video"
                    print(f"  補上 {path}.{k} = video")
                else:
                    del node[k]
                    print(f"  移除未填值欄位 {path}.{k}")
                continue
            try:
                child = node[k]
            except Exception:
                continue
            fix_missing(child, f"{path}.{k}")


def load_model(ckpt, pretrained):
    utils.import_user_module(Namespace(user_dir=HERE))
    overrides = {"w2v_path": os.path.abspath(pretrained)} if pretrained else None
    state = checkpoint_utils.load_checkpoint_to_cpu(ckpt, overrides)
    fix_missing(state["cfg"])
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt], state=state)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = [m.eval().to(device) for m in models]
    return models, task, device


def run(models, task, device, roi, wav, mode, beam, ref):
    n_video = int(cv2.VideoCapture(roi).get(cv2.CAP_PROP_FRAME_COUNT))
    with wave.open(wav) as w:
        n_audio = w.getnframes()

    data_dir = tempfile.mkdtemp()
    v = roi if "video" in MODES[mode] else "None"
    a = wav if "audio" in MODES[mode] else "None"
    with open(f"{data_dir}/test.tsv", "w") as f:
        f.write(f"/\ntest-0\t{v}\t{a}\t{n_video}\t{n_audio}\n")
    with open(f"{data_dir}/test.wrd", "w") as f:
        f.write((ref or "dummy") + "\n")

    # 直接沿用模型檔還原出來的 task（裡面已經有詞典和 tokenizer），只換資料路徑與模態
    task.cfg.data = data_dir
    task.cfg.label_dir = data_dir
    task.cfg.modalities = MODES[mode]
    task.cfg.noise_wav = None
    task.load_dataset("test", task_cfg=task.cfg)

    generator = task.build_generator(models, GenerationConfig(beam=beam))
    itr = task.get_batch_iterator(dataset=task.dataset("test")).next_epoch_itr(shuffle=False)
    sample = next(itr)
    if device == "cuda":
        sample = utils.move_to_cuda(sample)
    with torch.no_grad():
        hypos = task.inference_step(generator, models, sample)

    ignore = set(generator.symbols_to_strip_from_output)
    ignore.add(task.target_dictionary.pad())
    proc = task.datasets["test"].label_processors[0]
    return proc.decode(hypos[0][0]["tokens"].int().cpu(), ignore)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip_dir", required=True, help="prep_one_clip.py 的輸出資料夾（含 roi.mp4、clip.wav）")
    ap.add_argument("--ckpt", required=True, help="微調好的模型，例如 CAT-AVSR_AV.pt")
    ap.add_argument("--mode", default="all", choices=["all", "av", "v", "a"])
    ap.add_argument("--beam", type=int, default=20)
    ap.add_argument("--ref", default=None, help="（選填）正確答案文字，有給就會算 WER")
    ap.add_argument("--pretrained", default=None,
                    help="（選填）萬一出現找不到 large_vox_iter5.pt 的錯誤，在這裡指定本機路徑")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    roi = os.path.abspath(os.path.join(args.clip_dir, "roi.mp4"))
    wav = os.path.abspath(os.path.join(args.clip_dir, "clip.wav"))
    for p in (roi, wav):
        if not os.path.isfile(p):
            sys.exit(f"找不到 {p}，請先執行 prep_one_clip.py")

    print("載入模型中（檔案很大，可能要一兩分鐘）...")
    models, task, device = load_model(args.ckpt, args.pretrained)

    modes = ["av", "v", "a"] if args.mode == "all" else [args.mode]
    for m in modes:
        hyp = run(models, task, device, roi, wav, m, args.beam, args.ref)
        print(f"\n[{NAMES[m]}] {hyp}")
        if args.ref:
            from jiwer import wer
            print(f"  WER: {wer(args.ref.lower(), hyp.lower()) * 100:.1f}%")


if __name__ == "__main__":
    main()
