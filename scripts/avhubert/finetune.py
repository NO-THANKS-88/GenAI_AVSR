#!/usr/bin/env python
"""
Step 4：微調 AV-HuBERT（每次跑一組實驗）。

以官方的低資源設定 conf/finetune/large_lrs3_30h.yaml 為基礎（LRS3 30 小時、Large 模型），
只覆寫資料路徑、模態、GPU 數量等，其餘訓練超參數都和官方一致，報告時好交代。

用法：放在 SynthAVSR/avhubert/ 裡，改好「設定區」後，在 tmux 裡執行
  conda activate avhubert
  cd ~/GAN/SynthAVSR/avhubert
  python finetune.py

建議的四組實驗（每次改 DATA 和 MODALITY 兩行）：
  train_10h + A、train_10h + AV、train_30h + A、train_30h + AV
"""
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# ======================== 設定區（只要改這裡） ========================
DATA = "train_30h"          # "train_10h" 或 "train_30h"
MODALITY = "A"             # "A"（純音訊，對照組）或 "AV"（影音）
GPUS = [1]               # 用哪幾張 GPU（先 nvidia-smi 看哪幾張空）
MAX_TOKENS = 1000           # 每張 GPU 一次吃多少格（25 格 = 1 秒）；出現 CUDA out of memory 就改 600
NOISE = True               # True：加噪音微調（改進實驗）；實驗資料夾名稱會加上 _noisy
NOISE_DIR = "~/GAN/exp/noise/train_babble"   # make_noise.py 產生的訓練用噪音
NOISE_PROB = 0.25           # 每句有多少機率混入噪音（官方加噪音微調設定）
NOISE_SNR = 0               # 混入噪音的訊噪比 dB（官方設定）

DATA_ROOT = "~/GAN/data/tsou/avhubert"
SPM_MODEL = "~/GAN/data/tsou/avhubert/spm500/spm_unigram500.model"
PRETRAINED = "~/GAN/SynthAVSR/checkpoints/large_vox_iter5.pt"
EXP_ROOT = "~/GAN/exp"
BASE_CONFIG = "large_lrs3_30h.yaml"     # 官方 LRS3 30h 設定：18k 步，前 14.4k 步凍結編碼器
EFFECTIVE_GPUS = 8                       # 官方用 8 張 GPU；GPU 較少時自動用梯度累積補足，batch 大小才一致
# =====================================================================


def x(p):
    return os.path.abspath(os.path.expanduser(p))


def main():
    if MODALITY not in ("A", "AV"):
        sys.exit('MODALITY 只能是 "A" 或 "AV"')
    data_dir, spm, pre = x(f"{DATA_ROOT}/{DATA}"), x(SPM_MODEL), x(PRETRAINED)
    for p in (f"{data_dir}/train.tsv", f"{data_dir}/dict.wrd.txt", spm, pre):
        if not os.path.isfile(p):
            sys.exit(f"找不到 {p}")
    exe = shutil.which("fairseq-hydra-train")
    if exe is None:
        sys.exit("找不到 fairseq-hydra-train，請確認已 conda activate avhubert")

    n = len(GPUS)
    update_freq = max(1, EFFECTIVE_GPUS // n)
    exp_dir = x(f"{EXP_ROOT}/{DATA}_{MODALITY}" + ("_noisy" if NOISE else ""))
    if os.path.isfile(f"{exp_dir}/checkpoints/checkpoint_last.pt"):
        print(f"偵測到 {exp_dir} 已有 checkpoint，會從中斷處繼續訓練")
    modalities = "[audio,video]" if MODALITY == "AV" else "[audio]"

    cmd = [exe, "--config-dir", f"{HERE}/conf/finetune", "--config-name", BASE_CONFIG,
           f"task.data={data_dir}", f"task.label_dir={data_dir}",
           f"task.tokenizer_bpe_model={spm}", f"model.w2v_path={pre}",
           f"task.modalities={modalities}",
           "+task.input_modality=video",         # 佔位欄位（設定檔裡沒有，要用 + 新增），避免新版 fairseq 遇到 ??? 報錯
           f"distributed_training.distributed_world_size={n}",
           f"distributed_training.nprocs_per_node={n}",
           f"optimization.update_freq=[{update_freq}]",
           f"dataset.max_tokens={MAX_TOKENS}",
           f"hydra.run.dir={exp_dir}", f"common.user_dir={HERE}"]
    if NOISE:
        nd = x(NOISE_DIR)
        if not os.path.isfile(f"{nd}/train.tsv"):
            sys.exit(f"找不到 {nd}/train.tsv，請先執行 make_noise.py")
        # 設定檔裡沒有這三個欄位，所以用 + 新增；其他超參數完全不變，只多了「訓練時混入噪音」這一個差異
        cmd += [f"+task.noise_wav={nd}", f"+task.noise_prob={NOISE_PROB}", f"+task.noise_snr={NOISE_SNR}"]

    print(f"實驗：{DATA} / {MODALITY}{' / 加噪音' if NOISE else ''}　GPU {GPUS}　梯度累積 {update_freq}　輸出 {exp_dir}")
    print("指令：\n  " + " \\\n  ".join(cmd) + "\n")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, GPUS)), HYDRA_FULL_ERROR="1")
    sys.exit(subprocess.call(cmd, cwd=HERE, env=env))


if __name__ == "__main__":
    main()
