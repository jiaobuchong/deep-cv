# 01_full 逐行讲解

对应文件：`methods/01_full.ipynb`（精简版）。

很多代码和 07 完全一样：读数据、切分 holdout、提特征、算原型、评估、测延迟。这些地方照样贴出代码，但只简单说明，细节见 `07_linear_walkthrough.md` 的对应小节（下文写作“见 07 §x”）。新的和改过的地方逐行讲：训练相关的设置、数据增强、drop path、逐层学习率衰减、训练循环、训练曲线和完整流程。末尾有日志对照、新术语和讲解时可能被问到的问题。

## 0. 先看全局

### 全量微调是什么

- ViT-B/16 的全部参数都参与训练：主干 85,798,656 个，加上新的 10,000 类分类头 7,690,000 个。
- 分类头先用类原型初始化（和 07 的 zero-shot 一样），然后整个模型一起训练 3 个 epoch。
- 之前用同样设置跑过（`1019_1120_1_full`）：holdout **73.38%**。对比：分类头用 linear probe 起步的 `0928_0636_3_full` 是 70.88%，什么都不训练的 zero-shot 是 43.95%。

### 和 07 比多了什么

| | 07_linear | 01_full |
|---|---|---|
| 主干 | 冻结 | 全部训练 |
| 分类头 | 类原型，或者 linear probe | 先用类原型初始化，再和主干一起训练 |
| 训练循环 | 没有 | §7 的 `train`：3 个 epoch，AdamW，warmup 加余弦退火 |
| 数据增强 | 没有，只缩放和中心裁剪 | 随机裁剪 + 随机水平翻转 |
| 学习率 | 只有 linear probe 的 1e-3 | 逐层衰减：最后一层 1e-4，越往前越小 |
| 正则化 | 标签平滑（只在 linear probe 里） | 权重衰减 0.05、drop path 0.1、标签平滑 0.1、梯度裁剪 |
| GPU 时间 | 提特征 + 设置分类头 | 提特征 + 设置分类头 + 训练 + 每个 epoch 末的 holdout 评估 |
| 输出文件 | `final.pt` 只有分类头 | `final.pt` 存全部参数，另外多了 `history.json` 和 `curves.png` |

### 一次实验的流程（§9 `run_experiment`）

```
建模型（create_model，带 drop path）
  → 切分数据，准备训练和评估两套预处理（make_ctx）
  → 用类原型初始化分类头（init_head）
  → 所有参数设为可训练（apply_method）
  → 训练 3 个 epoch，每个 epoch 结束在 holdout 上测一次（train）
  → 保存权重和训练记录
  → 最后在 holdout 上评估（evaluate），画训练曲线，测推理延迟（latency_report）
  → 写 summary.json
```

### 训练的几个关键数字

- 每一步 256 张图。A100 上 `BATCH_SIZE` 是 256，正好等于 `EFFECTIVE_BATCH`，不用梯度累积。
- 每个 epoch 1,679 步：430,000 ÷ 256 = 1,679.7，取整数部分。凑不满一批的 176 张图这个 epoch 不用（每个 epoch 打乱顺序，丢掉的是不同的图）。
- 3 个 epoch 一共 5,037 步。前 503 步（0.3 个 epoch）是 warmup。
- 最后一层的学习率怎么变：第 0 步 2.0e-7，到第 502 步升到 1e-4；第 1,679 步（第 1 个 epoch 结束）8.4e-5，第 2,770 步 5.0e-5，第 3,358 步（第 2 个 epoch 结束）3.0e-5，最后降到接近 0。
- 之前同样设置的 `1019_1120_1_full`：0.48 GPU-h，峰值显存 18.9 GB，推理延迟 7.78 ms。

### 训练时的张量形状

| 变量 | 形状 | 说明 |
|---|---|---|
| 一批训练图 `x`（CPU） | (256, 3, 224, 224)，uint8 | 随机裁剪、翻转后的图片 |
| `to_input(x, ctx)` | (256, 3, 224, 224)，float32，在 GPU 上 | 归一化到 [−1, 1] |
| `model(...)` 的输出 | (256, 10000) | 每张图对每个类的分数 |
| 一批标签 `y` | (256,)，int64 | |
| `loss` | 一个数 | 这一批的平均交叉熵 |
| holdout 评估的一批 | (512, 3, 224, 224) | 评估不用存梯度，批大一倍 |

---

## 1. Config（§1）

### 1.1 要跑的实验

```python
# ===== Experiments to run: each dict is one run and lists only the settings that differ from the defaults below =====
EXPERIMENTS = [
    {},  # the defaults; already run as 1019_1120_1_full: holdout 73.38%
]
```

- 只有一个实验，`{}` 是一个空 dict，意思是“全部用下面的默认值”。默认值就是 7 个 notebook 统一的对比设置，所以直接运行，结果就能和其他方法比。
- 想加一个变体，比如训练 8 个 epoch，就加一行 `dict(EPOCHS=8)`。只有 §2 的 `RUN_KEYS` 里列出的设置能这样改。

### 1.2 默认值

```python
# ===== Defaults: the comparison protocol shared by all method notebooks (prototype head, 3 epochs, 2 holdout images per class) =====
METHOD = "full"
MODEL_NAME = "vit_base_patch16_224.augreg2_in21k_ft_in1k"  # timm name of the base model in the doc (IN21K→IN1K)
HOLDOUT_PER_CLASS = 2    # images per class held out from train for local evaluation (the official val set has no labels)
EPOCHS = 3
WARMUP_EPOCHS = 0.3      # the LR rises linearly for this many epochs, then follows a cosine down to 0
BATCH_SIZE = None        # images per step; None = pick based on VRAM (256 on a 40 or 80 GB GPU)
EFFECTIVE_BATCH = 256    # if one step can't fit it, gradients are accumulated over several steps to reach it
LR = 1e-4
WEIGHT_DECAY = 0.05
LAYER_DECAY = 0.75       # layer-wise LR decay: the last block gets LR × 0.75, the first block LR × 0.75^12
DROP_PATH = 0.1          # stochastic depth: residual branches are randomly skipped during training
LABEL_SMOOTHING = 0.1
RRC_SCALE = (0.25, 1.0)  # area range for RandomResizedCrop (training augmentation)
LOG_EVERY = 100          # write a log line every N optimizer steps
SEED = 0
NUM_CLASSES = 10000
```

| 变量 | 默认值 | 意思 | 用在哪里 |
|---|---|---|---|
| `METHOD` | `"full"` | 方法名，只用在实验名、日志和 summary 里 | §8–§10 |
| `MODEL_NAME` | ViT-B/16（IN21K→IN1K） | 模型和预训练权重，名字的含义见 07 §1.2 | §4 `create_model` |
| `HOLDOUT_PER_CLASS` | `2` | 每类留几张做 holdout | §5 `split_holdout` |
| `EPOCHS` | `3` | 把训练部分完整过几遍 | §7 `train` |
| `WARMUP_EPOCHS` | `0.3` | 学习率先花 0.3 个 epoch 从 0 线性升到最大，之后按余弦曲线降到 0 | §7 `train` |
| `BATCH_SIZE` | `None` | 每一步送进 GPU 的图片数；`None` 按显存自动选，40 GB 以上是 256 | §4、§5 |
| `EFFECTIVE_BATCH` | `256` | 每更新一次参数要用多少张图。一步放不下时，用梯度累积凑够 | §4 `resolve_config`、§7 |
| `LR` | `1e-4` | 最后一层（和分类头）的最大学习率 | §7 |
| `WEIGHT_DECAY` | `0.05` | 权重衰减 | §7 `build_param_groups` |
| `LAYER_DECAY` | `0.75` | 逐层学习率衰减的系数 | §7 `build_param_groups` |
| `DROP_PATH` | `0.1` | drop path（随机深度）的最大概率 | §4 `create_model` |
| `LABEL_SMOOTHING` | `0.1` | 标签平滑，例子见 07 §1.2 | §7 `train` |
| `RRC_SCALE` | `(0.25, 1.0)` | 随机裁剪时，裁出的区域占原图面积的范围 | §5 `make_ctx` |
| `LOG_EVERY` | `100` | 每 100 步写一行日志，5,037 步一共约 50 行 | §7 `train` |
| `SEED` | `0` | 随机种子 | §5、§9 |
| `NUM_CLASSES` | `10000` | 类别数 | §4、§6 |

下面把训练相关的几个概念说清楚。

**epoch、warmup 和余弦退火**

- 一个 epoch 就是把 430,000 张训练图都用一遍。3 个 epoch，每张图用 3 次，每次的随机裁剪都不一样。
- 学习率不是一直不变的。前 0.3 个 epoch（503 步）从接近 0 线性升到 `LR`，这叫 warmup：刚开始模型和新任务还不匹配，梯度很大，一上来就用大学习率容易把预训练学到的东西冲坏。之后按余弦曲线慢慢降到 0：前期大步走，后期小步微调。具体数字见 §0。

**batch、有效 batch 和梯度累积**

- `BATCH_SIZE` 是 GPU 一次处理多少张图，受显存限制。`EFFECTIVE_BATCH` 是每更新一次参数要用多少张图，这才影响训练效果。
- 两者相等（A100 上都是 256）时，每一批算完就更新一次参数。
- 显存小的卡，比如 24 GB 的 L4，`BATCH_SIZE` 会自动变成 128。这时 `ACCUM_STEPS` = 256 ÷ 128 = 2：连着算 2 批，把梯度加起来再更新一次，效果相当于一次用 256 张图。这样不同 GPU 上的训练设置是一样的，结果才能比较。

**学习率 1e-4 为什么这么小**

主干已经在上千万张图上训练过，特征已经很好了。学习率太大会把这些特征冲坏（也叫“灾难性遗忘”），所以只做小幅度的调整。07 里 linear probe 用 1e-3，是因为那里只训练一个新的分类头。

**逐层学习率衰减（`LAYER_DECAY = 0.75`）**

越靠前的层学习率越小，每往前一层乘 0.75。完整的表：

| 层 | 包括 | 学习率倍数 | 学习率 |
|---|---|---|---|
| 0 | 图块嵌入 `patch_embed`、`cls_token`、`pos_embed` | 0.75^13 = 0.0238 | 2.38e-6 |
| 1 | block 0 | 0.75^12 = 0.0317 | 3.17e-6 |
| 2 | block 1 | 0.0422 | 4.22e-6 |
| 3 | block 2 | 0.0563 | 5.63e-6 |
| 4 | block 3 | 0.0751 | 7.51e-6 |
| 5 | block 4 | 0.1001 | 1.00e-5 |
| 6 | block 5 | 0.1335 | 1.33e-5 |
| 7 | block 6 | 0.1780 | 1.78e-5 |
| 8 | block 7 | 0.2373 | 2.37e-5 |
| 9 | block 8 | 0.3164 | 3.16e-5 |
| 10 | block 9 | 0.4219 | 4.22e-5 |
| 11 | block 10 | 0.5625 | 5.63e-5 |
| 12 | block 11（最后一个 block） | 0.75 | 7.50e-5 |
| 13 | 最后的 LayerNorm `norm`、分类头 `head` | 1 | 1.00e-4 |

这是最大学习率，实际每一步还要乘上 warmup / 余弦的系数。道理：靠前的层学的是边缘、纹理这些通用特征，本来就好用，少改；靠后的层更贴近具体任务，多改；分类头是新的，改得最多。

**权重衰减（`WEIGHT_DECAY = 0.05`）**

AdamW 每一步把权重往 0 拉一点，防止权重变得太大、过拟合。偏置、LayerNorm 的参数、`cls_token` 和 `pos_embed` 不做衰减，见 §7.1。

**drop path（`DROP_PATH = 0.1`）**

训练时，每个 block 的残差分支（注意力和 MLP）以一定概率被整个跳过，只留下残差连接。它逼着模型不依赖某几个 block，是一种正则化。概率从第一个 block 的 0 线性增加到最后一个 block 的 0.1：0, 0.009, 0.018, …, 0.091, 0.1。只在训练时生效，评估和推理时不跳过。

**随机裁剪（`RRC_SCALE = (0.25, 1.0)`）**

每次读一张训练图，随机裁出一块面积占原图 25%–100%、长宽比在 3:4 到 4:3 之间的区域，再缩放成 224×224。同一张图每个 epoch 看到的是不同的局部，相当于凭空多了很多训练样本。每类只有 43 张训练图，这一点很重要。

### 1.3 数据和路径

```python
# ===== Data and paths =====
HF_REPO = "doem1997/cs604-course-data"       # the course dataset on Hugging Face
DATA_FROM_HF = True      # True: download the data from Hugging Face (about 500 MB/s on Colab); False: use the copy on Drive (about 33 MB/s)
DRIVE_DIR = "/content/drive/MyDrive/cs604"   # results are saved here
DRIVE_DATA_DIR = f"{DRIVE_DIR}/data"         # copy of the dataset on Drive, used only when DATA_FROM_HF = False (made by cs604_download_to_drive.ipynb)
DATA_DIR = "/content/data"                   # each session puts the data on the local Colab disk; training reads images from here
DISCONNECT_WHEN_DONE = False  # release the runtime when done to stop using compute units (handy when running in the background)
```

和 07 一样，见 07 §1.3。

---

## 2. Setup（§2）

### 2.1 安装 timm 和 hf_xet

```python
# Colab already has torch / torchvision; install timm here, and hf_xet for fast parallel downloads from Hugging Face
!pip install -q timm hf_xet
```

和 07 一样，见 07 §2.1。

### 2.2 哪些设置可以按实验改

```python
# Settings an experiment in EXPERIMENTS may override
RUN_KEYS = ("MODEL_NAME", "HOLDOUT_PER_CLASS", "EPOCHS", "WARMUP_EPOCHS", "BATCH_SIZE", "EFFECTIVE_BATCH",
            "LR", "WEIGHT_DECAY", "LAYER_DECAY", "DROP_PATH", "LABEL_SMOOTHING", "RRC_SCALE", "LOG_EVERY",
            "SEED", "NUM_CLASSES")
BASE_CONFIG = {k: globals()[k] for k in RUN_KEYS}
```

- 机制和 07 一样（见 07 §2.2）：`RUN_KEYS` 是允许按实验修改的设置，`BASE_CONFIG` 是 §1 默认值的快照。
- 列表里换成了训练相关的设置：`EPOCHS`、`WARMUP_EPOCHS`、`EFFECTIVE_BATCH`、`LR`、`WEIGHT_DECAY`、`LAYER_DECAY`、`DROP_PATH`、`RRC_SCALE`、`LOG_EVERY`。07 里 linear probe 的那几个设置在这里没有。

### 2.3 导入库

```python
import os, re, gc, json, math, time, random, shutil, subprocess
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import timm
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from huggingface_hub import hf_hub_download
from google.colab import drive
```

和 07 的区别只有三处，其余见 07 §2.3：

- 多了 `re`（正则表达式）：§7.1 的 `build_param_groups` 用它从参数名 `blocks.5.attn.qkv.weight` 里读出 block 的编号。
- 多了 `matplotlib.pyplot`（`plt`）：§8 画训练曲线。
- 少了 `torch.nn`（`nn`）：07 用它新建 linear probe 的分类头，这里用不到。

### 2.4 Drive、GPU、批次名、日志

```python
# ---- Mount Google Drive: data is read from here, and results are saved here, so nothing is lost on disconnect ----
drive.mount("/content/drive")
OUT_ROOT = DRIVE_DIR

# ---- GPU: needs bf16 support, i.e. an A100, L4 or H100 runtime ----
assert torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8, (
    "Pick a GPU runtime with bf16 support: A100, L4 or H100")
DEVICE = "cuda"
props = torch.cuda.get_device_properties(0)
GPU_NAME = props.name
gib = props.total_memory / 1024**3
VRAM_GB = next((v for v in (8, 12, 16, 24, 32, 40, 48, 80, 94, 96, 141, 192) if v >= 0.98 * gib), round(gib))  # nominal VRAM
torch.backends.cuda.matmul.allow_tf32 = True  # TF32 for float32 matrix multiplies: much faster, accuracy barely changes
torch.backends.cudnn.allow_tf32 = True
NUM_WORKERS = min(16, os.cpu_count())  # processes that read and decode images in parallel
VERSIONS = f"torch {torch.__version__} (CUDA {torch.version.cuda}), timm {timm.__version__}"

os.environ["TZ"] = "Asia/Singapore"  # the Colab VM runs on UTC; use local time for batch names and log timestamps
time.tzset()
BATCH_NAME = datetime.now().strftime("%m%d_%H%M")  # this session's runs are named <BATCH_NAME>_<i>_<method>
BATCH_DIR = os.path.join(OUT_ROOT, "batches", BATCH_NAME)
os.makedirs(BATCH_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

LOG_FILE = None  # train_log.txt of the current run, set by run_experiment


def log(*args):
    """Print, and also append to this batch's batch_log.txt and the current run's train_log.txt (the report needs the training log)"""
    msg = " ".join(str(a) for a in args)
    print(msg, flush=True)
    stamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")
    for path in (os.path.join(BATCH_DIR, "batch_log.txt"), LOG_FILE):
        if path:
            with open(path, "a") as f:
                f.write(stamp + msg + "\n")


ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
log(f"Batch {BATCH_NAME} -> {BATCH_DIR}")
log(f"Device {GPU_NAME} ({VRAM_GB} GB), bfloat16, {os.cpu_count()} CPUs, {ram_gb:.0f} GB RAM, "
    f"{shutil.disk_usage(DATA_DIR).free / 1e9:.0f} GB free disk")
log(f"Versions: {VERSIONS}")
```

和 07 完全一样，逐行解释见 07 §2.4–§2.8。对 01 来说值得再提一句的是 TF32 那两行：训练时主干的计算在 bf16 autocast 里，不受影响；它主要加速 `fit_logit_scale` 里 float32 的矩阵乘法。

---

## 3. 准备数据（§3）

```python
def prepare_split(split):
    """Put {split}.txt and the images of {split}.tar.gz on the local Colab disk under DATA_DIR. The files come from Hugging Face,
    or, with DATA_FROM_HF = False, from the copy on Drive. Returns the path of the local {split}.txt"""
    list_path = os.path.join(DATA_DIR, f"{split}.txt")
    marker = os.path.join(DATA_DIR, f".{split}_ready")
    if os.path.exists(marker):  # already done in this session
        return list_path
    t0 = time.time()
    if DATA_FROM_HF:
        hf_hub_download(HF_REPO, f"{split}.txt", repo_type="dataset", local_dir=DATA_DIR)
        tar_path = hf_hub_download(HF_REPO, f"{split}.tar.gz", repo_type="dataset", local_dir=os.path.join(DATA_DIR, "hf"))
        log(f"[{split}] downloaded from Hugging Face in {(time.time() - t0) / 60:.1f} min")
    else:
        shutil.copy(os.path.join(DRIVE_DATA_DIR, f"{split}.txt"), list_path)
        tar_path = os.path.join(DRIVE_DATA_DIR, f"{split}.tar.gz")
    log(f"[{split}] extracting {tar_path} ...")
    subprocess.run(["tar", "-xzf", tar_path, "-C", DATA_DIR], check=True)
    if DATA_FROM_HF:
        os.remove(tar_path)  # the images are out; free the disk space the archive takes (40 GB for train)
    with open(list_path) as f:
        n_expected = sum(1 for line in f if line.strip())
    n_found = sum(len(files) for _, _, files in os.walk(os.path.join(DATA_DIR, split)))
    assert n_found == n_expected, f"{split}: found {n_found} images, expected {n_expected}"
    open(marker, "w").close()
    log(f"[{split}] {n_found} images ready in {(time.time() - t0) / 60:.1f} min")
    return list_path


def read_list(path):
    """Each line is '<image path> <class id>', e.g. 'train/class_0000/9aef28bf-....jpg 0'"""
    items = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if parts:
                items.append((parts[0], int(parts[1])))
    return items


TRAIN_ALL = read_list(prepare_split("train"))
log(f"train {len(TRAIN_ALL):,} images | {NUM_WORKERS} loader workers")
```

和 07 完全一样，逐行解释见 07 §3：从 Hugging Face 下载 `train.txt` 和 `train.tar.gz`（`DATA_FROM_HF = False` 时用 Drive 上的备份），解压到 Colab 本地磁盘，检查图片数量，再读出 450,000 个（路径, 类别）。

---

## 4. 模型（§4）

### 4.1 `is_head`

```python
def is_head(name):
    return name in ("head.weight", "head.bias")
```

和 07 一样（见 07 §4.1）：判断一个参数是不是分类头的。

### 4.2 `resolve_config`

```python
def resolve_config(overrides):
    """Defaults from section 1 + one experiment's overrides → the full config of this run"""
    unknown = set(overrides) - set(RUN_KEYS)
    assert not unknown, f"unknown keys in experiment config: {sorted(unknown)}"
    cfg = SimpleNamespace(**{**BASE_CONFIG, **overrides})
    assert cfg.HOLDOUT_PER_CLASS > 0, "this notebook evaluates on the holdout set, so keep at least 1 image per class"
    if cfg.BATCH_SIZE is None:
        cfg.BATCH_SIZE = 256 if VRAM_GB >= 40 else 128 if VRAM_GB >= 24 else 64
    cfg.ACCUM_STEPS = max(1, cfg.EFFECTIVE_BATCH // cfg.BATCH_SIZE)
    return cfg
```

前面几行和 07 一样（见 07 §4.2）：检查有没有写错的设置名，合并默认值和实验的修改，按显存选 `BATCH_SIZE`。和 07 的区别：

- 没有检查 `HEAD_INIT`：01 只用类原型初始化分类头，没有这个设置。
- 多了最后一行 `cfg.ACCUM_STEPS = max(1, cfg.EFFECTIVE_BATCH // cfg.BATCH_SIZE)`：梯度累积的步数。`//` 是整除。A100 上 256 // 256 = 1，不累积；24 GB 的卡上 256 // 128 = 2。`max(1, ...)` 保证至少是 1（比如有人把 `EFFECTIVE_BATCH` 设得比 `BATCH_SIZE` 还小）。

### 4.3 `create_model`

```python
def create_model(cfg):
    """Pretrained ViT-B/16 from timm; its 1000-class ImageNet head is replaced by a new, randomly initialized 10,000-class head"""
    return timm.create_model(cfg.MODEL_NAME, pretrained=True, num_classes=cfg.NUM_CLASSES,
                             drop_path_rate=cfg.DROP_PATH).to(DEVICE)
```

和 07 相比多了一个参数（其他见 07 §4.3）：

| 参数 | 值 | 意思 |
|---|---|---|
| `drop_path_rate=cfg.DROP_PATH` | 0.1 | 打开 drop path（随机深度）。timm 让各个 block 的概率从 0 线性增加到 0.1：block 0 是 0，block 11 是 0.1 |

drop path 具体做什么：一个 block 的计算是 `x + 注意力(LN(x))`，再 `x + MLP(LN(x))`。训练时，对每张图独立地随机决定是否跳过注意力或 MLP 这一支（跳过时只剩 `x`，相当于这张图这一层“少一层”）；没跳过的那些会除以保留概率，让期望值不变。这样模型不能只依赖某几个 block，泛化更好。评估时（`model.eval()`）不跳过。

### 4.4 `apply_method`

```python
def apply_method(net, cfg):
    """Full fine-tuning: every parameter of the backbone and the new head is trained"""
    return net.requires_grad_(True)
```

- `requires_grad_(True)`：所有参数都设为“需要梯度”，也就是全部参与训练。新建的模型本来就是这样，这一行是把“全量微调”这件事明确写出来。
- 和 07 正好相反：07 是 `requires_grad_(False)`，全部冻结。其他方法（LoRA、Adapter 等）的区别也都集中在这个函数。

### 4.5 `param_report`

```python
def param_report(net, ctx):
    """Trainable parameter counts. The 7.69M head is counted separately, as in the doc's LoRA reference (0.34%)"""
    n_train = sum(p.numel() for n, p in net.named_parameters() if p.requires_grad and not is_head(n))
    return {"backbone_params": ctx.n_backbone, "head_params": ctx.n_head, "head_mode": "fine-tuned",
            "trainable_excl_head": n_train, "trainable_pct": 100 * n_train / ctx.n_backbone,
            "trainable_incl_head": n_train + ctx.n_head}
```

- `n_train = sum(...)`：数需要梯度、而且不是分类头的参数，也就是主干里可训练的参数：85,798,656。
- 分类头单独算，这和项目文档里博士生的 LoRA 参考（0.34%）算法一致：可训练参数不含分类头，因为分类头是每个方法都必须训练的新层。
- 返回值：

| key | 值 | 意思 |
|---|---|---|
| `backbone_params` | 85,798,656 | 主干的参数量 |
| `head_params` | 7,690,000 | 分类头的参数量 |
| `head_mode` | `fine-tuned` | 分类头参与训练 |
| `trainable_excl_head` | 85,798,656 | 可训练参数，不含分类头 |
| `trainable_pct` | 100.0 | 占主干的百分比 |
| `trainable_incl_head` | 93,488,656 | 可训练参数，含分类头 |

---

## 5. 数据集和 holdout（§5）

### 5.1 `split_holdout`、`ImageList`、`BICUBIC`

```python
def split_holdout(items, k, seed):
    """Hold out k images of every class: shuffle each class's image list with a fixed seed and take the first k"""
    by_class = {}
    for path, label in items:
        by_class.setdefault(label, []).append(path)
    rng = random.Random(seed)
    train, holdout = [], []
    for label in sorted(by_class):
        paths = sorted(by_class[label])
        rng.shuffle(paths)
        holdout += [(p, label) for p in paths[:k]]
        train += [(p, label) for p in paths[k:]]
    return train, holdout


class ImageList(Dataset):
    def __init__(self, items, transform):
        self.items, self.transform = items, transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        with Image.open(os.path.join(DATA_DIR, path)) as im:
            img = im.convert("RGB")
        return self.transform(img), label


BICUBIC = T.InterpolationMode.BICUBIC
```

和 07 完全一样，见 07 §5.1–§5.3：每类随机留 2 张做 holdout（种子固定，每次都是同一批图），按编号取图并做预处理，缩放用双三次插值。

### 5.2 `make_ctx`：多了一套训练用的预处理

```python
def make_ctx(model, cfg):
    """Data setup for this run: train/holdout split, image transforms, normalization values, param counts"""
    data_cfg = timm.data.resolve_model_data_config(model)
    size = data_cfg["input_size"][-1]
    train_items, holdout_items = split_holdout(TRAIN_ALL, cfg.HOLDOUT_PER_CLASS, cfg.SEED)
    return SimpleNamespace(
        img_size=size, train_items=train_items, holdout_items=holdout_items,
        mean=torch.tensor(data_cfg["mean"], device=DEVICE).view(1, 3, 1, 1) * 255,
        std=torch.tensor(data_cfg["std"], device=DEVICE).view(1, 3, 1, 1) * 255,
        # Only crop/flip on the CPU and output uint8; normalize on the GPU to save CPU work
        train_tf=T.Compose([T.RandomResizedCrop(size, scale=tuple(cfg.RRC_SCALE), interpolation=BICUBIC),
                            T.RandomHorizontalFlip(), T.PILToTensor()]),
        eval_tf=T.Compose([T.Resize(int(size / data_cfg["crop_pct"]), interpolation=BICUBIC),
                           T.CenterCrop(size), T.PILToTensor()]),
        n_backbone=sum(p.numel() for n, p in model.named_parameters() if not is_head(n)),
        n_head=sum(p.numel() for p in model.head.parameters()),
    )
```

除了 `train_tf`，其他和 07 一样（见 07 §5.4）。`train_tf` 是训练图片的预处理，依次做三步：

1. `T.RandomResizedCrop(size, scale=tuple(cfg.RRC_SCALE), interpolation=BICUBIC)`：随机裁剪再缩放。
   - 随机选一块区域：面积占原图的 25%–100%（`scale=(0.25, 1.0)`），长宽比在 3:4 到 4:3 之间（没写，用默认值），位置也是随机的。
   - 把这块区域缩放成 224×224，用双三次插值。
   - `tuple(...)` 把配置转成元组，因为这个参数要求元组。
2. `T.RandomHorizontalFlip()`：一半的概率左右翻转。
3. `T.PILToTensor()`：转成 uint8 张量 (3, 224, 224)。

为什么训练和评估用不同的预处理：

- 训练时要“变着花样”看同一张图：每个 epoch 裁的位置、大小、是否翻转都不一样，模型很难死记硬背某张图，泛化更好。每类只有 43 张训练图，这一点很重要。
- 评估时要稳定、可重复：每次都是同样的缩放加中心裁剪，结果才能比较。

### 5.3 `Loader`

```python
class Loader(DataLoader):
    """Before starting new worker processes, run garbage collection in the main process. Otherwise an old, interrupted DataLoader is copied into the new workers,
    and when it is collected there it prints lots of harmless AssertionError: can only test a child process messages."""

    def __iter__(self):
        gc.collect()
        return super().__iter__()
```

和 07 一样，见 07 §5.5。

### 5.4 `make_loader`：训练和评估两种读法

```python
def make_loader(items, train, ctx, cfg):
    """Training: random crops, shuffled, BATCH_SIZE images per step. Evaluation: center crops, in order, 2 × BATCH_SIZE (no gradients, so it fits)"""
    return Loader(ImageList(items, ctx.train_tf if train else ctx.eval_tf),
                  batch_size=cfg.BATCH_SIZE if train else 2 * cfg.BATCH_SIZE, shuffle=train, drop_last=train,
                  num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=train)
```

比 07 多了一个参数 `train`（`True` 或 `False`），决定是训练用还是评估用。`a if 条件 else b` 是条件表达式：条件成立取 a，否则取 b。

| 参数 | 训练（`train=True`） | 评估（`train=False`） | 意思 |
|---|---|---|---|
| 预处理 | `ctx.train_tf` | `ctx.eval_tf` | 随机裁剪加翻转 / 缩放加中心裁剪 |
| `batch_size` | 256 | 512 | 评估不用存梯度，显存够，批大一倍更快 |
| `shuffle` | `True` | `False` | 训练时每个 epoch 重新打乱顺序 |
| `drop_last` | `True` | `False` | 训练时丢掉最后凑不满 256 张的那一批，保证每一步都是 256 张；评估一张都不能少 |
| `num_workers` | 12 | 12 | 读图子进程数 |
| `pin_memory` | `True` | `True` | 锁页内存，拷到 GPU 更快 |
| `persistent_workers` | `True` | `False` | 训练时 3 个 epoch 一直用同一组子进程，不用每个 epoch 重启一次 |

### 5.5 `to_input`、`autocast`、`free_gpu`

```python
def to_input(x, ctx):
    """uint8 images (B, 3, 224, 224) → normalized float32 on the GPU"""
    return (x.to(DEVICE, non_blocking=True).float() - ctx.mean) / ctx.std


def autocast():
    """bf16 mixed precision: matrix multiplies run in bfloat16, the weights stay float32"""
    return torch.autocast("cuda", dtype=torch.bfloat16)


def free_gpu():
    """Free the GPU memory left by the previous run and reset the peak memory stats"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
```

和 07 一样，见 07 §5.7–§5.9。训练时补充一点：在 `autocast` 里做前向，反向传播时对应的矩阵乘法也用 bf16 算；但参数、梯度和 AdamW 的状态都是 float32，所以更新参数时不会丢精度。

---

## 6. 分类头初始化（§6）

### 6.1 `Clock`：GPU 时间计时器

```python
class Clock:
    """Total GPU work time of a run: head init + training + holdout evaluation during training (not data prep)"""

    def __init__(self):
        self.used, self._t0 = 0.0, None

    def __enter__(self):
        self._t0 = time.time()
        return self

    def __exit__(self, *exc):
        torch.cuda.synchronize()
        self.used += time.time() - self._t0
        self._t0 = None

    @property
    def elapsed(self):
        return self.used + (time.time() - self._t0 if self._t0 else 0.0)
```

07 没有这个类，因为 07 的 GPU 时间只有设置分类头那一段。01 要把几段时间加起来：设置分类头，加上每个 epoch 的训练和 holdout 评估。

- `__init__`：`used` 是已经累计的秒数，从 0 开始；`_t0` 是当前这一段开始的时间，`None` 表示现在没在计时。
- `__enter__` 和 `__exit__`：有了这两个方法，就能写 `with clock:`。进入 `with` 时调用 `__enter__`，记下开始时间；离开 `with` 时（不管是正常结束还是出错）调用 `__exit__`，把这一段的时间加进 `used`。
- `__exit__` 里的 `torch.cuda.synchronize()`：等 GPU 把排着队的计算做完再停表，否则会少算。`*exc` 接收出错时传进来的异常信息，这里用不到。
- `elapsed`：到现在为止一共用了多少秒。如果正在某一段里面，把这一段已经过去的时间也加上。`@property` 让它可以像变量一样写 `clock.elapsed`，不用加括号。

计入 GPU 时间的：设置分类头（含提特征），每个 epoch 的训练，每个 epoch 末的 holdout 评估。不计入的：准备数据（下载和解压）、保存文件、训练完以后的最终评估、测延迟。

### 6.2 `extract_features`

```python
@torch.inference_mode()
def extract_features(net, items, ctx, cfg):
    """Features of the frozen backbone: the class-token output after the final LayerNorm, 768 numbers per image"""
    net.eval()
    feats, labels = [], []
    for x, y in tqdm(make_loader(items, False, ctx, cfg), desc="train features", leave=False):
        with autocast():
            f = net.forward_head(net.forward_features(to_input(x, ctx)), pre_logits=True)
        feats.append(f.half())
        labels.append(y.to(DEVICE))
    return torch.cat(feats), torch.cat(labels)
```

和 07 §6.1 一样，唯一的区别是 `make_loader(items, False, ctx, cfg)` 多传了一个 `False`，表示用评估的读法：中心裁剪、不打乱、每批 512 张。提特征时主干还没开始训练，就是预训练好的样子。

### 6.3 `class_prototypes`、`fit_logit_scale`

```python
@torch.inference_mode()
def class_prototypes(feats, labels, num_classes):
    """Mean direction of each class: L2-normalize every feature, add them up per class, normalize the sums"""
    protos = torch.zeros(num_classes, feats.shape[1], device=feats.device)
    for i in range(0, len(feats), 65536):
        protos.index_add_(0, labels[i:i + 65536], F.normalize(feats[i:i + 65536].float(), dim=1))
    return F.normalize(protos, dim=1)


@torch.inference_mode()
def fit_logit_scale(protos, feats, labels, n=10000):
    """Pick a good temperature for cosine similarity, so the softmax of the prototype head is not too flat or too sharp"""
    idx = torch.randperm(len(feats), device=feats.device)[:n]
    cos = F.normalize(feats[idx].float(), dim=1) @ protos.T
    losses = {s: F.cross_entropy(cos * s, labels[idx]).item() for s in (5, 10, 15, 20, 30, 40, 60, 80)}
    return min(losses, key=losses.get)
```

和 07 完全一样，见 07 §6.2–§6.3：算出每一类的原型（平均方向），再挑一个温度。在 01 里温度更重要：它决定训练刚开始时分类头有多“自信”，也就决定了开头几步的损失和梯度有多大。

### 6.4 `init_head`：用类原型初始化分类头

```python
def init_head(model, ctx, cfg):
    """Prototype head: each class's weight is the mean direction of its training features, times a temperature.
    Returns the seconds spent, feature extraction included; they count toward this run's GPU time"""
    t0 = time.time()
    feats, labels = extract_features(model, ctx.train_items, ctx, cfg)
    protos = class_prototypes(feats, labels, cfg.NUM_CLASSES)
    scale = fit_logit_scale(protos, feats, labels)
    with torch.no_grad():
        # The head works on unnormalized features: divide by the mean feature norm so logits ≈ scale × cosine similarity
        model.head.weight.copy_(protos * (scale / feats.float().norm(dim=1).mean()))
        model.head.bias.zero_()
    torch.cuda.synchronize()
    seconds = time.time() - t0
    log(f"Head initialized from the class prototypes of {len(feats):,} train images (logit scale {scale}) "
        f"in {seconds / 60:.1f} min, counted in this run's GPU time")
    return seconds
```

参数：`model` 是刚建好的模型。返回：这一步花的秒数（含提特征）。算法和 07 的 zero-shot 一样，写法简单一些，因为 01 只跑一个实验，不需要缓存特征，也没有 linear probe：

- `t0 = time.time()`：从提特征开始计时。
- `feats, labels = extract_features(model, ctx.train_items, ctx, cfg)`：用还没训练的主干给 430,000 张训练图提特征，A100 上大约 6–7 分钟。
- `protos = ...`、`scale = ...`：原型和温度，同 07。
- `with torch.no_grad():` 里的两行：直接写进模型的分类头。
  - `model.head.weight.copy_(protos * (scale / feats.float().norm(dim=1).mean()))`：权重 = 原型 × 温度 ÷ 特征的平均长度，为什么除以平均长度见 07 §6.6。
  - `model.head.bias.zero_()`：偏置清零，`zero_` 是原地填 0。
- `torch.cuda.synchronize()`、`seconds = time.time() - t0`：等 GPU 算完再停表。
- `log(...)`：记录用了多少张图、温度和用时。

为什么全量微调也要先初始化分类头（这是我们项目最重要的发现）：

- 分类头如果是随机的，训练一开始损失很大，很大的梯度会从分类头一路传回主干，把预训练好的特征冲坏。
- 用类原型起步，模型一开始就有 zero-shot 的水平（约 44%），梯度温和，主干只需要在好的基础上微调。
- 之前用 linear probe 的分类头起步，效果反而差：holdout 70.88%，而类原型是 73.38%。可能的原因是 linear probe 就是在这批训练图片的冻结特征上拟合出来的，一开始训练损失已经很低，给主干的训练信号偏弱；主干的特征一变，它又对不上了。

---

## 7. 训练（§7）：这个 notebook 的核心

### 7.1 `build_param_groups`：每层不同的学习率

```python
def build_param_groups(net, cfg):
    """Layer-wise LR decay: block i (1–12) gets LR × LAYER_DECAY^(13 − i); the patch embedding, class token and position
    embeddings count as layer 0, the final norm and the head get the full LR. No weight decay on 1-D params (biases,
    LayerNorm weights), the class token and the position embeddings"""
    num_layers = len(net.blocks) + 1
    groups = {}
    for n, p in net.named_parameters():
        m = re.match(r"blocks\.(\d+)\.", n)
        if m:
            layer = int(m.group(1)) + 1
        elif n.startswith(("cls_token", "pos_embed", "patch_embed")):
            layer = 0
        else:
            layer = num_layers
        lr = cfg.LR * cfg.LAYER_DECAY ** (num_layers - layer)
        wd = 0.0 if p.ndim <= 1 or n in ("cls_token", "pos_embed") else cfg.WEIGHT_DECAY
        groups.setdefault((lr, wd), []).append(p)
    return [{"params": ps, "lr": lr, "weight_decay": wd, "base_lr": lr} for (lr, wd), ps in groups.items()]
```

作用：把模型的参数分成若干组，每组有自己的学习率和权重衰减，交给优化器。返回值是一个列表，每个元素描述一组。

- `num_layers = len(net.blocks) + 1`：12 个 block 加 1，等于 13。层的编号：图块嵌入这些算第 0 层，block 0 到 block 11 是第 1 到 12 层，最后的 LayerNorm 和分类头是第 13 层。
- `groups = {}`：key 是（学习率, 权重衰减），value 是这一组的参数列表。
- `for n, p in net.named_parameters():`：遍历所有参数，`n` 是名字，`p` 是参数本身。
- 判断这个参数属于第几层：
  - `m = re.match(r"blocks\.(\d+)\.", n)`：用正则表达式看名字是不是以 `blocks.数字.` 开头。`\d+` 匹配一个或多个数字，括号把数字“捕获”下来；`\.` 表示真的点号。比如 `blocks.5.attn.qkv.weight` 能匹配，`m.group(1)` 是 `"5"`。
  - `if m:` 然后 `layer = int(m.group(1)) + 1`：block 5 是第 6 层。
  - `elif n.startswith(("cls_token", "pos_embed", "patch_embed")):` 然后 `layer = 0`：图块嵌入、CLS token、位置编码是第 0 层。`startswith` 可以传一个元组，以其中任意一个开头都算。
  - `else:` 然后 `layer = num_layers`：剩下的是最后的 `norm` 和分类头 `head`，算第 13 层。
- `lr = cfg.LR * cfg.LAYER_DECAY ** (num_layers - layer)`：`**` 是乘方。第 13 层乘 0.75^0 = 1，第 12 层乘 0.75，……，第 0 层乘 0.75^13。完整的表见 §1.2。
- `wd = 0.0 if p.ndim <= 1 or n in ("cls_token", "pos_embed") else cfg.WEIGHT_DECAY`：哪些参数不做权重衰减：
  - `p.ndim <= 1`：一维的参数，也就是所有的偏置（bias）和 LayerNorm 的参数。它们只是平移和缩放，数量也少，衰减它们没什么好处，反而可能有害。
  - `cls_token`、`pos_embed`：虽然是三维的，但它们是“可学习的输入”，习惯上也不衰减。
  - 其余（各个线性层的权重、图块嵌入的卷积核、分类头的权重）衰减 0.05。
- `groups.setdefault((lr, wd), []).append(p)`：放进对应的组，组还不存在就先建一个空列表。
- 最后的 `return`：把每一组变成优化器认识的格式 `{"params": 参数列表, "lr": 学习率, "weight_decay": 衰减}`。多存一个 `"base_lr"`，因为训练时每一步要把 `lr` 改成 `base_lr × 当前的 warmup/余弦系数`，得留着原始值。

最后一共 28 组：14 个不同的学习率，每个学习率又分“衰减”和“不衰减”两组。

### 7.2 `evaluate`

```python
@torch.inference_mode()
def evaluate(net, loader, ctx):
    """Top-1 and top-5 accuracy on a labeled set"""
    net.eval()
    c1 = c5 = n = 0
    for x, y in loader:
        with autocast():
            top5 = net(to_input(x, ctx)).topk(5, dim=1).indices
        y = y.to(DEVICE)
        c1 += (top5[:, 0] == y).sum().item()
        c5 += (top5 == y[:, None]).any(dim=1).sum().item()
        n += len(y)
    return c1 / n, c5 / n
```

和 07 完全一样，见 07 §7.1。在 01 里它被用了两次：每个 epoch 结束时测 holdout top-1（在训练循环里，算进 GPU 时间），以及训练完以后再测一次 top-1 和 top-5（不算进 GPU 时间）。注意里面的 `net.eval()`：评估时关掉 drop path。

### 7.3 `save_state`

```python
def save_state(net, path, **extra):
    """Save the trained params: all of them, about 370 MB"""
    state = {n: p.detach().cpu() for n, p in net.named_parameters() if p.requires_grad or is_head(n)}
    torch.save({"state": state, **extra}, path)
```

代码和 07 一样（见 07 §7.2），但结果不同：01 里所有参数都需要梯度，所以全部保存，93,488,656 个 float32，约 374 MB。

### 7.4 `train`：训练循环

参数：`model` 模型，`ctx` 数据设置，`cfg` 配置，`clock` 计时器（已经记上了设置分类头的时间）。返回：（训练记录 `hist`, 一共更新了几步 `step`, 训练速度 `speed`，单位 张/秒）。函数比较长，分五段看。

#### （1）准备：步数和学习率曲线

```python
def train(model, ctx, cfg, clock):
    """AdamW with linear warmup, then cosine decay to 0; bf16 autocast; gradient clipping at 1.0.
    Evaluates on the holdout set after every epoch. Returns (history, optimizer steps, training speed in img/s)"""
    spe = len(ctx.train_items) // (cfg.BATCH_SIZE * cfg.ACCUM_STEPS)  # optimizer steps per epoch
    total, warmup = cfg.EPOCHS * spe, max(1, int(cfg.WARMUP_EPOCHS * spe))

    def lr_factor(s):
        if s < warmup:
            return (s + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total - warmup)))
```

- `spe = len(ctx.train_items) // (cfg.BATCH_SIZE * cfg.ACCUM_STEPS)`：每个 epoch 更新几次参数（steps per epoch）= 430,000 // (256 × 1) = 1,679。
- `total, warmup = cfg.EPOCHS * spe, max(1, int(cfg.WARMUP_EPOCHS * spe))`：一行给两个变量赋值。总步数 3 × 1,679 = 5,037；warmup 步数 `int(0.3 × 1,679)` = 503，`max(1, ...)` 保证至少 1，免得下面除以 0。
- `def lr_factor(s):`：函数里面再定义一个小函数，给出第 s 步的学习率系数（0 到 1）：
  - `if s < warmup: return (s + 1) / warmup`：warmup 阶段线性上升。第 0 步是 1/503 ≈ 0.002，第 502 步是 1。
  - 否则 `0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total - warmup)))`：余弦退火。`(s - warmup) / (total - warmup)` 是 warmup 之后走了百分之多少，从 0 到 1；`cos(π × 进度)` 从 1 变到 −1；整个式子从 1 平滑地降到 0。走到一半（第 2,770 步）正好是 0.5。

#### （2）优化器、数据和记录

```python
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(build_param_groups(model, cfg), fused=True)
    train_loader = make_loader(ctx.train_items, True, ctx, cfg)
    holdout_loader = make_loader(ctx.holdout_items, False, ctx, cfg)
    hist = {"step": [], "loss": [], "epoch": [], "epoch_loss": [], "holdout_top1": [], "steps_per_epoch": spe}
    log(f"Training {cfg.EPOCHS} epochs x {spe} steps (batch {cfg.BATCH_SIZE}x{cfg.ACCUM_STEPS}, warmup {warmup} steps)")
    step, speed, t_mark, t_start = 0, None, 0.0, clock.elapsed
```

- `params = [...]`：所有要训练的参数，后面做梯度裁剪用。全量微调就是全部参数。
- `optimizer = torch.optim.AdamW(build_param_groups(model, cfg), fused=True)`：AdamW 优化器，参数按 §7.1 分好组，每组用自己的学习率和权重衰减。其他设置用默认值（`betas=(0.9, 0.999)`、`eps=1e-8`）。`fused=True`：把所有参数的更新合成一个 GPU 计算一起做，更快。AdamW 怎么更新参数，见附录 B。
- `train_loader`：训练的读法（随机裁剪、打乱、每批 256 张），见 §5.4。
- `holdout_loader`：评估的读法（中心裁剪、每批 512 张），每个 epoch 结束时用。
- `hist`：训练记录，最后存成 `history.json`、画成 `curves.png`：
  - `step`、`loss`：每 100 步记一次（第几步、最近 100 步的平均损失）。
  - `epoch`、`epoch_loss`、`holdout_top1`：每个 epoch 记一次（第几个 epoch、这个 epoch 的平均损失、holdout top-1）。
  - `steps_per_epoch`：1,679，画图时把步数换算成 epoch 用。
- `log(...)`：比如 `Training 3 epochs x 1679 steps (batch 256x1, warmup 503 steps)`。`256x1` 表示每批 256 张、累积 1 批更新一次。
- 最后一行一次初始化四个变量：
  - `step`：已经更新了几次参数，从 0 开始。
  - `speed`：训练速度，第 120 步时才测出来，之前是 `None`。
  - `t_mark`：测速时用的时间点。
  - `t_start = clock.elapsed`：训练开始时计时器的读数，这时等于设置分类头花的秒数。后面算剩余时间时减掉它，只看训练本身。

#### （3）每个 epoch

```python
    for epoch in range(cfg.EPOCHS):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        run_loss, run_n, ep_loss, ep_n, t_log = 0.0, 0, 0.0, 0, time.time()
        with clock:
            for i, (x, y) in enumerate(train_loader):
```

- `for epoch in range(cfg.EPOCHS):`：`epoch` 从 0 到 2，日志里显示成 1 到 3。
- `model.train()`：切到训练模式，打开 drop path。上一个 epoch 末做评估时切成了 `eval()`，所以每个 epoch 开头都要切回来。
- `optimizer.zero_grad(set_to_none=True)`：清空梯度。正常情况下每次更新完都会清，这里是保险：用了梯度累积时，上一个 epoch 末尾可能剩下几批没凑够、没更新的梯度，不能带进新的 epoch。
- `run_loss, run_n, ...`：五个计数器：
  - `run_loss`、`run_n`：最近一段（100 步）累计的损失和批数，用来算日志里的平均损失。
  - `ep_loss`、`ep_n`：这个 epoch 累计的损失和批数，用来算 epoch 的平均损失。
  - `t_log`：上次写日志的时间，用来算这一段的速度。
- `with clock:`：从这里开始计时，这个 epoch 的训练和最后的 holdout 评估都算进 GPU 时间。
- `for i, (x, y) in enumerate(train_loader):`：一批一批地取训练数据，`i` 是这个 epoch 里的第几批（从 0 开始），`x` 是 (256, 3, 224, 224) 的图片，`y` 是 256 个标签。

#### （4）每一批：前向、反向、更新

```python
                with autocast():
                    loss = F.cross_entropy(model(to_input(x, ctx)), y.to(DEVICE, non_blocking=True),
                                           label_smoothing=cfg.LABEL_SMOOTHING)
                (loss / cfg.ACCUM_STEPS).backward()
                run_loss += loss.detach()
                run_n += 1
                if (i + 1) % cfg.ACCUM_STEPS:
                    continue  # gradient accumulation: add up the gradients of ACCUM_STEPS batches before each update
                for g in optimizer.param_groups:
                    g["lr"] = g["base_lr"] * lr_factor(step)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
```

这几行就是“训练一步”的全部内容：

1. **前向，算损失**：`with autocast():` 里，`model(to_input(x, ctx))` 把图片送进模型得到 (256, 10000) 的分数；`F.cross_entropy(分数, 标签, label_smoothing=0.1)` 算这一批的平均交叉熵。标签用 `non_blocking=True` 异步拷到 GPU。前向用 bf16，而且 drop path 在起作用。
2. **反向传播**：`(loss / cfg.ACCUM_STEPS).backward()` 算出损失对每个参数的梯度，加到参数的 `.grad` 上。除以 `ACCUM_STEPS`：累积几批时，加起来正好是平均值。A100 上是除以 1。
3. **记账**：`run_loss += loss.detach()` 累计损失。`detach()` 表示只要这个数，不要它的计算图，免得占显存。`run_n += 1` 批数加一。
4. **梯度累积**：`if (i + 1) % cfg.ACCUM_STEPS: continue`。`%` 是取余数：累积 2 批时，第 1 批（i = 0）余数是 1，不为 0，`continue` 跳过下面的更新，接着取下一批；第 2 批余数是 0，往下执行更新。累积 1 批时余数永远是 0，每批都更新。
5. **设学习率**：`for g in optimizer.param_groups: g["lr"] = g["base_lr"] * lr_factor(step)`。28 组参数，每组的学习率 = 它自己的最大学习率 × 当前这一步的系数。
6. **梯度裁剪**：`torch.nn.utils.clip_grad_norm_(params, 1.0)`。把所有参数的梯度看成一个超长的向量，如果它的长度超过 1.0，就整体按比例缩小到 1.0；没超过就不动。防止某一步的梯度特别大，把模型一下子带偏。
7. **更新参数**：`optimizer.step()`，AdamW 按梯度更新所有参数。
8. **清空梯度**：`optimizer.zero_grad(set_to_none=True)`，为下一步做准备。
9. `step += 1`：更新次数加一。

#### （5）测速度、写日志、epoch 结束

```python
                # Training speed, measured over steps 20–120 (after the data loader and the GPU have warmed up)
                if step == 20:
                    t_mark = clock.elapsed
                elif step == 120:
                    sec_per_step = (clock.elapsed - t_mark) / 100
                    speed = cfg.BATCH_SIZE * cfg.ACCUM_STEPS / sec_per_step
                    log(f"Speed {speed:.0f} img/s, {sec_per_step * spe / 60:.1f} min/epoch")
```

- 测训练速度：第 20 步记下时间，到第 120 步再看一次，中间 100 步平均每步用了多久（`sec_per_step`）。跳过前 20 步，是因为刚开始子进程还在启动、GPU 还在热身，那时候偏慢。
- `speed = cfg.BATCH_SIZE * cfg.ACCUM_STEPS / sec_per_step`：每步 256 张图，除以每步的秒数，就是每秒训练多少张图。
- `log(...)`：`sec_per_step * spe / 60` 是一个 epoch 要几分钟。之前最终模型的日志里是 `Speed 1146 img/s, 6.5 min/epoch`（那次用全部 450,000 张图，每个 epoch 1,757 步）。这个速度也会写进 `summary.json`，报告里比较各方法的训练速度就用它。

```python
                if step % cfg.LOG_EVERY == 0:
                    avg = (run_loss / run_n).item()
                    if not math.isfinite(avg):
                        raise RuntimeError("Loss is NaN/inf; try a lower LR")
                    eta = (total - step) * (clock.elapsed - t_start) / step
                    log(f"ep {epoch + 1}/{cfg.EPOCHS} step {step}/{total} | loss {avg:.4f} | "
                        f"lr {cfg.LR * lr_factor(step):.2e} | {run_n * cfg.BATCH_SIZE / (time.time() - t_log):.0f} img/s | "
                        f"{clock.elapsed / 60:.1f} min | ETA {eta / 60:.0f} min")
                    hist["step"].append(step)
                    hist["loss"].append(avg)
                    ep_loss += avg * run_n
                    ep_n += run_n
                    run_loss, run_n, t_log = 0.0, 0, time.time()
```

- `if step % cfg.LOG_EVERY == 0:`：每 100 步写一次日志。
- `avg = (run_loss / run_n).item()`：最近 100 步的平均损失，`.item()` 变成 Python 数字。
- `if not math.isfinite(avg): raise RuntimeError(...)`：损失变成 NaN（不是一个数）或无穷大，说明训练崩了，通常是学习率太大，直接停下，不浪费时间。
- `eta = (total - step) * (clock.elapsed - t_start) / step`：估计还要多久。训练到现在用的时间 ÷ 已经走的步数 = 每步的平均时间，再乘剩下的步数。
- `log(...)`：一行日志，比如 `ep 1/3 step 100/5037 | loss … | lr 2.01e-05 | … img/s | … min | ETA … min`，依次是：第几个 epoch、第几步、最近的平均损失、当前学习率（最后一层的）、最近这段的速度、GPU 时间已经用了几分钟、预计还要几分钟。这里的速度用墙上时间算（`time.time() - t_log`），包括等数据的时间。
- `hist["step"].append(step)`、`hist["loss"].append(avg)`：记进训练记录。
- `ep_loss += avg * run_n`、`ep_n += run_n`：平均值乘批数就是这一段的损失总和，累计到 epoch 里。
- 最后一行：三个计数器清零，开始下一段。

```python
            if run_n:
                ep_loss += float(run_loss)
                ep_n += run_n
            top1 = evaluate(model, holdout_loader, ctx)[0]
        hist["epoch"].append(epoch + 1)
        hist["epoch_loss"].append(ep_loss / max(1, ep_n))
        hist["holdout_top1"].append(top1)
        log(f"== epoch {epoch + 1}: train loss {ep_loss / max(1, ep_n):.4f}, holdout top-1 {top1:.2%}, "
            f"{clock.elapsed / 60:.1f} min used")
    return hist, step, speed
```

- `if run_n:`：一个 epoch 的批数不一定是 100 的整数倍，最后不满 100 步的那一段还没加进 epoch 的统计，这里补上。`run_loss` 本来就是这些批的损失总和，`float(...)` 把它变成 Python 数字。
- `top1 = evaluate(model, holdout_loader, ctx)[0]`：在 holdout 上测 top-1，`[0]` 只取 top-1。这一行还在 `with clock:` 里面，所以评估时间也算进 GPU 时间。`evaluate` 里面会切到 `eval()` 模式。
- 接下来几行在 `with clock:` 外面（计时停了）：把这个 epoch 的编号、平均损失和 holdout top-1 记进 `hist`，再写一行日志，比如 `== epoch 1: train loss …, holdout top-1 …%, … min used`。`max(1, ep_n)` 防止除以 0。
- 3 个 epoch 跑完，`return hist, step, speed`。正常情况下 `step` 是 5,037。

---

## 8. 训练曲线、推理延迟（§8）

### 8.1 颜色和坐标轴样式

```python
INK, MUTED, GRID, SURFACE, LINE = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb", "#2a78d6"


def style_axis(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
```

- 第一行：五个颜色（十六进制 RGB）。`INK` 近黑色，用于标题和数值；`MUTED` 深灰，用于坐标轴文字；`GRID` 浅灰，用于网格线；`SURFACE` 接近白色，用作背景；`LINE` 蓝色，用于曲线。
- `style_axis(ax, title, xlabel, ylabel)`：给一张子图统一样式。`ax` 是 matplotlib 的一个坐标系。
  - `set_facecolor`：背景色。
  - `set_title(..., loc="left")`：标题靠左。
  - `set_xlabel`、`set_ylabel`：坐标轴名称。
  - `grid(True, ...)`：画浅色网格线；`set_axisbelow(True)` 让网格线在曲线下面。
  - `tick_params(colors=MUTED, length=0)`：刻度文字用灰色，不画刻度小短线。
  - 两个 `for`：去掉上边框和右边框，左边框和下边框改成浅灰色，看起来更干净。

### 8.2 `plot_run_curves`：画训练曲线

```python
def plot_run_curves(hist, cfg, run_dir):
    """Save curves.png: the training loss, and the holdout top-1 after each epoch"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), facecolor=SURFACE)
    axes[0].plot(hist["step"], hist["loss"], color=LINE, linewidth=1.5, solid_capstyle="round")
    style_axis(axes[0], f"Training loss ({METHOD})", "optimizer step", f"cross-entropy (label smoothing {cfg.LABEL_SMOOTHING})")
    acc = [100 * a for a in hist["holdout_top1"]]
    axes[1].plot(hist["epoch"], acc, color=LINE, linewidth=1.5, marker="o", markersize=6,
                 markeredgecolor=SURFACE, markeredgewidth=1.5)
    axes[1].annotate(f"{acc[-1]:.1f}%", (hist["epoch"][-1], acc[-1]), textcoords="offset points", xytext=(0, 8),
                     ha="center", color=INK, fontsize=9)
    axes[1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    style_axis(axes[1], "Holdout top-1 accuracy", "epoch", "accuracy (%)")
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "curves.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)
```

- `fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), facecolor=SURFACE)`：一行两列共两张子图，整张图 11 × 3.6 英寸。
- 左图：`axes[0].plot(hist["step"], hist["loss"], ...)`，横轴是步数，纵轴是每 100 步的平均损失，蓝线，线宽 1.5，线头是圆的。标题里带上方法名，纵轴说明是带标签平滑的交叉熵。
- `acc = [100 * a for a in hist["holdout_top1"]]`：把 0–1 的准确率换成百分数。
- 右图：`axes[1].plot(hist["epoch"], acc, ..., marker="o", ...)`，横轴是 epoch，纵轴是 holdout top-1，每个点画一个圆点（直径 6，白边）。
- `axes[1].annotate(...)`：在最后一个点上方 8 个点的位置标出数值，比如 `73.4%`。`textcoords="offset points"` 和 `xytext=(0, 8)` 表示相对那个点往上偏 8 个点；`ha="center"` 水平居中。
- `axes[1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))`：横轴只显示整数刻度（1、2、3），不要 1.5 这种。
- `fig.tight_layout()`：自动调整间距，免得文字重叠。
- `fig.savefig(..., dpi=150, ...)`：存成 `curves.png`，分辨率 150 dpi。
- `plt.close(fig)`：关掉这张图，释放内存，也不在 notebook 里显示。

### 8.3 推理延迟

```python
@torch.inference_mode()
def measure_latency(net, batch_size, iters, ctx, warmup=10):
    """Average seconds per forward pass on random input: model compute only, no image reading or decoding"""
    net.eval()
    x = torch.randn(batch_size, 3, ctx.img_size, ctx.img_size, device=DEVICE)
    for _ in range(warmup):
        with autocast():
            net(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        with autocast():
            net(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def latency_report(net, ctx):
    lat_1 = measure_latency(net, 1, 100, ctx)
    lat_b = measure_latency(net, 256, 20, ctx)
    setting = f"{GPU_NAME}, bfloat16, no flip TTA"
    log(f"Inference latency (batch 1): {lat_1 * 1e3:.2f} ms/img | throughput (batch 256): {256 / lat_b:.0f} img/s | {setting}")
    return dict(latency_ms_bs1=lat_1 * 1e3, throughput_img_s=256 / lat_b, latency_setting=setting)
```

和 07 完全一样，见 07 §7.3–§7.4。全量微调只改了权重，没改模型结构，所以推理延迟和原来的 ViT-B/16 一样，之前测出来是 7.78 ms。

---

## 9. 一次实验的完整流程（§9 `run_experiment`）

参数：`overrides` 是这次实验的修改（默认是空的 `{}`），`run_name` 是实验名。返回：`summary`，这次实验的全部结果。分四段看。

### 9.1 准备

```python
def run_experiment(overrides, run_name):
    """Run one experiment and return its summary. Everything is saved to cs604/runs/<run_name>/ on Drive"""
    global LOG_FILE
    cfg = resolve_config(overrides)
    run_dir = os.path.join(OUT_ROOT, "runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    LOG_FILE = os.path.join(run_dir, "train_log.txt")
    try:
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump({**vars(cfg), "RUN_NAME": run_name, "overrides": overrides}, f, indent=2)
        log(f"\n========== {run_name} ==========")
        log(f"{METHOD}: lr {cfg.LR}, wd {cfg.WEIGHT_DECAY}, layer decay {cfg.LAYER_DECAY}, drop path {cfg.DROP_PATH}, epochs {cfg.EPOCHS}, "
            f"holdout {cfg.HOLDOUT_PER_CLASS}/class, prototype head")
        log(f"Environment: {GPU_NAME} ({VRAM_GB} GB), bfloat16, {os.cpu_count()} CPUs | {VERSIONS}")
        random.seed(cfg.SEED)
        np.random.seed(cfg.SEED)
        torch.manual_seed(cfg.SEED)
        free_gpu()
```

和 07 §8.1 基本一样：合成配置，建这次实验的目录，日志开始写进 `train_log.txt`，存 `config.json`，设随机种子，清理显存。区别只在第二行 `log`：记录的是训练相关的设置（学习率、权重衰减、逐层衰减、drop path、epoch 数）。

随机种子在 01 里管得更多：除了 `fit_logit_scale` 的抽样，还决定新分类头的随机初始值（马上会被覆盖）、每个 epoch 的打乱顺序和随机裁剪。所以同样的设置再跑一次，结果会非常接近，但不一定一模一样：GPU 上有些计算（比如反向传播里的累加）本身带一点不确定性；换一台 CPU 数不同的机器，读图子进程的数量变了，随机裁剪也会不一样。

### 9.2 建模型、初始化分类头

```python
        model = create_model(cfg)
        ctx = make_ctx(model, cfg)
        clock = Clock()
        clock.used += init_head(model, ctx, cfg)
        model = apply_method(model, cfg).to(DEVICE)
        params = param_report(model, ctx)
        log(f"Trainable (excl. head): {params['trainable_excl_head']:,} = {params['trainable_pct']:.3f}% of backbone | "
            f"head {ctx.n_head:,} params (fine-tuned)")
```

- `model = create_model(cfg)`：加载预训练的 ViT-B/16，换上新的 10,000 类分类头，打开 drop path。
- `ctx = make_ctx(model, cfg)`：切分训练部分和 holdout，准备两套预处理，统计参数量。
- `clock = Clock()`：新建计时器，从 0 开始。
- `clock.used += init_head(model, ctx, cfg)`：用类原型初始化分类头，把这一步的秒数直接加进 GPU 时间。
- `model = apply_method(model, cfg).to(DEVICE)`：所有参数设为可训练。`.to(DEVICE)` 是给会插入新模块的方法准备的（比如 Adapter，新模块建出来在 CPU 上）；全量微调没有新模块，这一步什么也不改。
- `params = param_report(model, ctx)` 和下一行：统计并记录参数量，`:.3f` 是三位小数，所以显示 `100.000%`。

注意顺序：先初始化分类头，再 `apply_method`。提特征用的是还没训练过的预训练主干，这样原型反映的是预训练模型眼里每一类的样子。

### 9.3 训练和保存

```python
        hist, step, speed = train(model, ctx, cfg, clock)
        save_state(model, os.path.join(run_dir, "final.pt"), config=vars(cfg))
        with open(os.path.join(run_dir, "history.json"), "w") as f:
            json.dump(hist, f)
        log(f"Training done: {step} steps, {clock.elapsed / 3600:.2f} GPU-hours")
        # The project doc asks for VRAM in the training log; reserved is closer to what nvidia-smi shows
        log(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB allocated, "
            f"{torch.cuda.max_memory_reserved() / 1e9:.1f} GB reserved by PyTorch (GPU has {VRAM_GB} GB)")
```

- `hist, step, speed = train(model, ctx, cfg, clock)`：训练 3 个 epoch，拿回训练记录、步数和速度。
- `save_state(...)`：存 `final.pt`，全部参数加上配置，约 374 MB。
- `json.dump(hist, f)`：训练记录存成 `history.json`，以后想重画曲线可以直接读它。
- `log(f"Training done: ...")`：`clock.elapsed / 3600` 是 GPU 时间（小时）：设置分类头 + 3 个 epoch 的训练和评估。这就是报告里全量微调的“训练时间”，之前是 0.48 GPU-h。
- 峰值显存那一行：和 07 §8.2 一样，`allocated` 是张量实际占用的最大值，`reserved` 是 PyTorch 最多预订了多少（更接近 `nvidia-smi` 的数字）。全量微调之前是 18.9 GB，比只做推理的 07（3.9 GB）大得多，原因见附录 C 第 6 条。

### 9.4 最终评估和汇总

```python
        top1, top5 = evaluate(model, make_loader(ctx.holdout_items, False, ctx, cfg), ctx)
        log(f"Final holdout: top-1 {top1:.2%}, top-5 {top5:.2%}")
        plot_run_curves(hist, cfg, run_dir)
        latency = latency_report(model, ctx)

        summary = dict(run=run_name, method=METHOD, model=cfg.MODEL_NAME, overrides=overrides, **params,
                       holdout_top1=top1, holdout_top5=top5, **latency, gpu=GPU_NAME, vram_gb=VRAM_GB,
                       batch=f"{cfg.BATCH_SIZE}x{cfg.ACCUM_STEPS}", epochs_trained=round(step / hist["steps_per_epoch"], 2),
                       steps=step, train_speed_img_s=speed, gpu_hours=clock.elapsed / 3600,
                       peak_vram_gb=torch.cuda.max_memory_allocated() / 1e9,
                       peak_vram_reserved_gb=torch.cuda.max_memory_reserved() / 1e9, versions=VERSIONS)
        with open(os.path.join(run_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        log(f"== {run_name} finished: holdout top-1 {top1:.2%} | trainable {params['trainable_excl_head']:,} "
            f"({params['trainable_pct']:.3f}%) | {summary['gpu_hours']:.2f} GPU-h")
        return summary
    finally:
        LOG_FILE = None
```

- `top1, top5 = evaluate(...)`：训练完再在 holdout 上测一次。top-1 应该和第 3 个 epoch 末测的一样（或者几乎一样），这次多给出 top-5。它在 `with clock:` 外面，不算进 GPU 时间。
- `plot_run_curves(hist, cfg, run_dir)`：画 `curves.png`。
- `latency = latency_report(model, ctx)`：测推理延迟，也不算进 GPU 时间。
- `summary = dict(...)`：汇总。和 07 相同的字段见 07 §8.3，01 多了这几个：

| 字段 | 值 | 意思 |
|---|---|---|
| `batch` | `"256x1"` | 每批 256 张，累积 1 批更新一次 |
| `epochs_trained` | 3.0 | 实际训练了几个 epoch：5,037 ÷ 1,679，`round(..., 2)` 保留两位小数 |
| `steps` | 5037 | 一共更新了几次参数 |
| `train_speed_img_s` | 约 1,150 | 训练速度，§7.4 第 120 步测的 |
| `gpu_hours` | 约 0.48 | GPU 时间，包括设置分类头、训练、每个 epoch 末的 holdout 评估 |

- 写 `summary.json`，记一行总结，返回。`finally` 里把 `LOG_FILE` 改回 `None`。

---

## 10. 运行实验（§10）

```python
RESULTS = []
for i, exp in enumerate(EXPERIMENTS, 1):
    RESULTS.append(run_experiment(dict(exp), f"{BATCH_NAME}_{i}_{METHOD}"))
```

- 按顺序跑 `EXPERIMENTS` 里的每个实验，默认只有一个。
- 实验名比如 `1102_1530_1_full`：批次名、序号、方法名。
- `dict(exp)`：复制一份再传进去，见 07 §9。
- 和 07 不同，这里没有“准确率太低就停下”的检查，因为 01 只跑一个实验。

---

## 11. 结果表（§11）

```python
header = ("| Run | Holdout top-1 | Holdout top-5 | Trainable params (excl. head) | Train speed (img/s) | GPU-hours | "
          "Peak VRAM (GB) | Latency bs1 (ms) | Throughput (img/s) |\n|---|---|---|---|---|---|---|---|---|")
rows = [f"| {r['run']} | {r['holdout_top1']:.2%} | {r['holdout_top5']:.2%} | "
        f"{r['trainable_excl_head']:,} ({r['trainable_pct']:.3f}%) | "
        f"{format(r['train_speed_img_s'], '.0f') if r['train_speed_img_s'] else '–'} | {r['gpu_hours']:.2f} | "
        f"{r['peak_vram_gb']:.1f} | {r['latency_ms_bs1']:.2f} | {r['throughput_img_s']:.0f} |" for r in RESULTS]
table = header + "\n" + "\n".join(rows)
with open(os.path.join(BATCH_DIR, "results.md"), "w") as f:
    f.write(table + "\n")
with open(os.path.join(BATCH_DIR, "results.json"), "w") as f:
    json.dump(RESULTS, f, indent=2)
log("\n" + table)

from IPython.display import Markdown, display
display(Markdown(table))
```

结构和 07 §10 一样，只是列不同：可训练参数（不含分类头，带百分比）、训练速度、GPU 时间、峰值显存、延迟、吞吐量。这些正好是报告和 slides 里方法对比表要的数。

唯一新的写法是训练速度那一格：`format(r['train_speed_img_s'], '.0f') if r['train_speed_img_s'] else '–'`。如果训练不到 120 步，速度没测出来，是 `None`，就显示一个短横线；否则 `format(数字, '.0f')` 取整显示。

---

## 12. 自动断开（§12）

```python
if DISCONNECT_WHEN_DONE:
    from google.colab import runtime
    log("All done; releasing the runtime ...")
    drive.flush_and_unmount()  # make sure all results are written to Drive
    runtime.unassign()
```

和 07 一样，见 07 §11。01 要跑半个多小时，挂着跑时可以把 `DISCONNECT_WHEN_DONE` 设成 `True`，跑完自动释放运行时，省计算单元。

---

## 附录 A：运行日志逐行对照

在 A100 上运行，日志大致是下面这样。`…` 是要等跑出来才知道的数字，学习率是按公式算好的。

```
Batch 1102_1530 -> /content/drive/MyDrive/cs604/batches/1102_1530
Device NVIDIA A100-SXM4-80GB (80 GB), bfloat16, 12 CPUs, … GB RAM, … GB free disk
Versions: torch … (CUDA …), timm …
[train] downloaded from Hugging Face in … min
[train] extracting /content/data/hf/train.tar.gz ...
[train] 450000 images ready in … min
train 450,000 images | 12 loader workers

========== 1102_1530_1_full ==========
full: lr 0.0001, wd 0.05, layer decay 0.75, drop path 0.1, epochs 3, holdout 2/class, prototype head
Environment: NVIDIA A100-SXM4-80GB (80 GB), bfloat16, 12 CPUs | torch … (CUDA …), timm …
Head initialized from the class prototypes of 430,000 train images (logit scale …) in … min, counted in this run's GPU time
Trainable (excl. head): 85,798,656 = 100.000% of backbone | head 7,690,000 params (fine-tuned)
Training 3 epochs x 1679 steps (batch 256x1, warmup 503 steps)
ep 1/3 step 100/5037 | loss … | lr 2.01e-05 | … img/s | … min | ETA … min
Speed … img/s, … min/epoch
ep 1/3 step 200/5037 | loss … | lr 4.00e-05 | … img/s | … min | ETA … min
…
== epoch 1: train loss …, holdout top-1 …%, … min used
…
== epoch 2: train loss …, holdout top-1 …%, … min used
…
== epoch 3: train loss …, holdout top-1 …%, … min used
Training done: 5037 steps, … GPU-hours
Peak VRAM: … GB allocated, … GB reserved by PyTorch (GPU has 80 GB)
Final holdout: top-1 …%, top-5 …%
Inference latency (batch 1): … ms/img | throughput (batch 256): … img/s | NVIDIA A100-SXM4-80GB, bfloat16, no flip TTA
== 1102_1530_1_full finished: holdout top-1 …% | trainable 85,798,656 (100.000%) | … GPU-h
```

最后还会打印 §11 的结果表。

| 日志 | 出自 | 说明，参考值 |
|---|---|---|
| `Batch ...`、`Device ...`、`Versions ...` | §2 | 批次名、运行环境 |
| `[train] downloaded ...`、`extracting ...`、`images ready` | §3 | 每个会话只有第一次运行才有。从 Hugging Face 下载一两分钟；用 Drive 备份时没有 `downloaded` 这一行，直接从 Drive 解压，要 20 分钟左右 |
| `========== ..._1_full ==========` | §9 | 实验开始，从这行起也写进 `train_log.txt` |
| `full: lr ...` | §9 | 训练设置摘要 |
| `Head initialized from ...` | §6 `init_head` | 大约 6–7 分钟，主要是提特征 |
| `Trainable (excl. head) ...` | §9 | 全部 85,798,656 个主干参数都训练 |
| `Training 3 epochs x 1679 steps ...` | §7 `train` | 每个 epoch 1,679 步，warmup 503 步 |
| `ep 1/3 step 100/5037 ...` | §7 | 每 100 步一行，一共约 50 行 |
| `Speed ... img/s, ... min/epoch` | §7 | 第 120 步时写一次；最终模型那次是 1,146 img/s |
| `== epoch N: ...` | §7 | 每个 epoch 一行：平均训练损失、holdout top-1、GPU 时间已经用了几分钟 |
| `Training done: 5037 steps, ...` | §9 | 之前是 0.48 GPU-h |
| `Peak VRAM ...` | §9 | 之前是 18.9 GB |
| `Final holdout ...` | §9 | 之前是 73.38%；top-1 应该和第 3 个 epoch 末的一样 |
| `Inference latency ...` | §8.3 | 之前是 7.78 ms |
| `== ... finished ...` | §9 | 一行总结，`train_log.txt` 写到这行为止 |

怎么看训练是否正常：

- 损失应该整体往下走。warmup 那段学习率还在升，损失可能有些起伏。
- 每个 epoch 末的 holdout top-1 应该一次比一次高。如果后面反而下降，说明开始过拟合了。
- 一旦出现 `Loss is NaN/inf` 的报错，训练会自动停下，通常要调低学习率。

---

## 附录 B：新术语

07 讲解里已经有的术语（特征、原型、交叉熵、AdamW、权重衰减、余弦退火、bf16 等）见 07 的附录 B。下面是训练特有的：

| 术语 | 说人话 |
|---|---|
| 梯度（gradient） | 损失对每个参数的“斜率”：参数往哪个方向变，损失下降得最快 |
| 反向传播（backpropagation） | 从损失出发往回走，一层层算出每个参数的梯度，就是 `loss.backward()` |
| 学习率（learning rate） | 每次沿着梯度方向走多大一步 |
| warmup | 训练开头学习率从很小慢慢升上去，避免一开始步子太大 |
| 有效 batch | 每更新一次参数用多少个样本 |
| 梯度累积 | 显存放不下大 batch 时，连着算几小批，把梯度加起来再更新一次 |
| 梯度裁剪 | 梯度的总长度超过上限（这里是 1.0）时整体缩小，防止一步走太远 |
| 逐层学习率衰减 | 越靠前的层学习率越小，每往前一层乘一个系数（0.75） |
| drop path（随机深度） | 训练时随机跳过某些 block 的分支，一种正则化 |
| 数据增强 | 训练时对图片做随机变换（裁剪、翻转），让模型见到更多样的样本 |
| 过拟合 | 在训练数据上越来越好，在没见过的数据上反而变差 |
| 灾难性遗忘 | 微调时步子太大，把预训练学到的通用特征冲掉了 |
| `train()` / `eval()` 模式 | 训练模式打开 dropout、drop path 这类随机操作；评估模式把它们关掉 |
| 参数组（param group） | 优化器里共用一套学习率和权重衰减的一组参数 |
| AdamW 的更新 | 给每个参数记两个滑动平均：梯度的平均 m，梯度平方的平均 v。每一步大致按 `学习率 × m / (√v + ε)` 更新，所以一直有大梯度的参数，步子会自动变小；另外再减去 `学习率 × 权重衰减 × 参数`，把参数往 0 拉一点 |
| 激活值（activation） | 前向计算时每一层的中间结果。反向传播要用到，所以训练时得一直存着，它是训练时显存的大头 |
| ETA | 预计还要多久跑完 |

---

## 附录 C：讲解时可能被问到的问题

1. **学习率为什么只有 1e-4，而且越往前的层越小？** 预训练的特征已经很好，学习率太大会把它们冲坏。靠前的层学的是边缘、纹理这些通用特征，改得最少；靠后的层更贴近具体任务，改得多一些；新的分类头改得最多。
2. **为什么先用类原型初始化分类头？效果差多少？** 随机的分类头一开始损失很大，大梯度会传回主干，破坏预训练特征。用类原型起步，模型一开始就有 zero-shot 的水平。同样训练 3 个 epoch：类原型起步 73.38%，linear probe 起步 70.88%。
3. **这里的 3 epoch 和最终模型有什么区别？** 最终模型 `1019_1319_1_full` 不留 holdout，每类 45 张全拿来训练，训练 8 个 epoch，预测时还加了翻转 TTA，CodaBench val 76.22%，用了 1.03 GPU-h。它要用主 notebook 跑，这里只有 3 epoch 的对比设置。
4. **全量微调算 PEFT 吗，满足 5% 的要求吗？** 不算，它训练 100% 的参数，在报告里是作为对照。我们的 PEFT 结果是 LoRA，可训练参数 1.375%。
5. **GPU 时间怎么算，在不在预算内？** 用 `Clock` 把设置分类头、每个 epoch 的训练和 holdout 评估加起来。之前同样设置是 0.48 GPU-h。预算是 2 小时 × 48 GB，换到 80 GB 的 A100 上是 96 ÷ 80 = 1.2 小时，远在预算内。
6. **为什么要 18.9 GB 显存？** 训练时每一层的中间结果（激活值）都要留着给反向传播用，batch 256、每张图 197 个 token、12 层，这部分占了大头。参数、梯度和 AdamW 的两个状态加起来只有约 1.5 GB（9,349 万个参数 × 4 字节 × 4 份）。18.9 GB 也小于预算里的 48 GB。
7. **为什么全量微调的推理延迟和 zero-shot 一样？** 模型结构没变，只是权重不同，推理的计算量完全一样。
8. **每个 epoch 末在 holdout 上评估有什么用？** 看模型是在进步还是开始过拟合。比如我们就是这样发现 LoRA 训练 6 个 epoch（71.52%）反而不如 3 个 epoch（71.98%）；全量微调则是训练越久越好（满数据时 3 个 epoch 74.01%，8 个 epoch 76.22%）。
