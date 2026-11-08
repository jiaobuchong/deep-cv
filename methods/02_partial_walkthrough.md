# 02_partial 逐行讲解

对应文件：`methods/02_partial.ipynb`（精简版）。

02 和 01 几乎一模一样，真正不同的只有五处：§1 的超参数、§2 的 `RUN_KEYS` 和导入、§4 的 `create_model` 和 `apply_method`、§7 的 `build_param_groups` 和 `save_state`、§9 的一行日志。这五处逐行讲。和 01 完全相同的代码照样贴出来，只简单说明，细节见 `01_full_walkthrough.md`（下文写作“见 01 §x”）；和 07 共用的部分也可以看 `07_linear_walkthrough.md`（“见 07 §x”）。末尾有日志对照、新术语和讲解时可能被问到的问题。

## 0. 先看全局

### 选层微调是什么

- 冻结前面 10 个 block，只训练最后 2 个 block（block 10 和 block 11）、最后的 LayerNorm 和分类头。
- 道理：靠前的层学的是边缘、纹理这些通用特征，换个任务也能直接用；越靠后的层越贴近具体任务，所以只放开后面几层。
- 课程里讲过这个方法：Session 5 Lesson 5（PDF 第 27、33、35 页）和 Session 3 Lesson 3（第 62 页）。
- 还没跑过。它的可训练参数占主干的 16.5%，超过项目文档建议的约 5%，所以在方法对比表里只当一个参考点，不算 PEFT 方法。

### 可训练参数

一个 block 里有这些参数：

| 参数 | 个数 |
|---|---|
| `norm1`（LayerNorm） | 1,536 |
| `attn.qkv`（768 → 2304 的线性层） | 1,771,776 |
| `attn.proj`（768 → 768） | 590,592 |
| `norm2`（LayerNorm） | 1,536 |
| `mlp.fc1`（768 → 3072） | 2,362,368 |
| `mlp.fc2`（3072 → 768） | 2,360,064 |
| 合计 | 7,087,872（主干的 8.26%） |

训练最后 K 个 block，再加上最后的 LayerNorm（1,536 个）：

| K | 可训练参数（不含分类头） | 占主干 |
|---|---|---|
| 1 | 7,089,408 | 8.263% |
| **2（默认）** | **14,177,280** | **16.524%** |
| 3 | 21,265,152 | 24.785% |
| 4 | 28,353,024 | 33.046% |

主干一共 85,798,656 个参数，分类头另有 7,690,000 个。默认 K = 2 时，含分类头一共训练 21,867,280 个参数。即使只训练 1 个 block，也有 8.3%，还是超过 5%。

### 和 01 比

| | 01_full | 02_partial |
|---|---|---|
| 训练哪些参数 | 全部 | 最后 2 个 block、最后的 LayerNorm、分类头 |
| 可训练参数（不含分类头） | 85,798,656（100%） | 14,177,280（16.524%） |
| 学习率 | 1e-4，逐层衰减 0.75 | 默认 3e-4、不衰减（`LAYER_DECAY = 1.0`）；也可以设成和 01 一样 |
| 权重衰减 | 0.05 | 0.05 |
| drop path | 0.1，作用在全部 12 个 block 上 | 默认不用（`DROP_PATH = 0.0`）；打开时只作用在要训练的 block 上 |
| 参数分组 | 28 组 | 默认 2 组；打开逐层衰减后变多（K = 4 时 10 组） |
| 反向传播 | 经过全部 12 个 block | 只经过最后 2 个 block |
| `final.pt` | 约 374 MB | 约 87 MB |
| 推理 | 和原来的 ViT-B/16 一样 | 一样，模型结构没变 |
| 数据、分类头初始化、训练循环、评估 | | 和 01 完全一样 |

### 一次实验的流程

和 01 一样（见 01 §0），只有“套上微调方法”这一步不同：

```
建模型（create_model，`DROP_PATH` 默认 0，不用 drop path）
  → 切分数据，准备训练和评估两套预处理（make_ctx）
  → 用类原型初始化分类头（init_head）
  → 冻结全部参数，再放开最后 2 个 block、最后的 LayerNorm 和分类头，冻住的 block 关掉 drop path（apply_method）
  → 训练 3 个 epoch，每个 epoch 结束在 holdout 上测一次（train）
  → 保存训练过的参数和训练记录
  → 最后在 holdout 上评估，画训练曲线，测推理延迟
  → 写 summary.json
```

### 训练的几个关键数字

4 个实验都用和 01 同一套对比设置：

- 每一步 256 张图；每个 epoch 1,679 步；3 个 epoch 一共 5,037 步；前 503 步是 warmup。
- 学习率怎么变（最大值 3e-4）：第 0 步 5.96e-7，第 502 步升到 3.00e-4；第 1,679 步（第 1 个 epoch 结束）2.53e-4，第 2,770 步 1.50e-4，第 3,358 步（第 2 个 epoch 结束）9.06e-5，最后降到接近 0。这是 K = 1、2 的情况。K = 4、6 用 01 的配方，最大学习率是 1e-4（分类头和最后的 LayerNorm），被训练的 block 再按层衰减，每一步怎么变和 01 完全一样（见 01 §0）。

### 跑之前可以预期什么（推测，还没在 GPU 上验证）

- **训练更快、显存更省**：前向还是要经过全部 12 个 block，但反向传播只经过最后 K 个 block 和分类头；前面那些 block 的中间结果不用存下来给反向传播用；K = 2 时 AdamW 也只要给 2,187 万个参数存状态，01 是 9,349 万个。所以 K 越大越慢，越接近 01。本机 CPU 上用小模型（vit_tiny）做冒烟测试时，02 的训练速度大约是 01 的 2.5 倍（117 对 47 张/秒），但 CPU 上的小模型不能代表 A100 上的真实速度。
- **推理延迟和 01 一样**：模型结构没变，01 之前测出来是 7.78 ms。
- **准确率**：把 K = 1、2、4、6 的 holdout 准确率连成一条曲线，两头接上 linear probe（62.07%，相当于 K = 0，只训练分类头）和全量微调（73.38%，相当于训练全部 12 个 block），看准确率主要是从哪几层来的；再和只训练 1.375% 参数的 LoRA（71.98%）比。

---

## 1. Config（§1）

### 1.1 要跑的实验

```python
# ===== Experiments to run: each dict is one run and lists only the settings that differ from the defaults below =====
EXPERIMENTS = [
    # Sweep the number of trained blocks. K = 1 and 2 use the defaults below (K = 2 is the comparison protocol shared by all
    # method notebooks); from K = 4 on, the full fine-tuning recipe of 01: smaller LR, layer decay, drop path
    dict(PARTIAL_LAST_K=1),
    dict(PARTIAL_LAST_K=2),
    dict(PARTIAL_LAST_K=4, LR=1e-4, LAYER_DECAY=0.75, DROP_PATH=0.1),
    dict(PARTIAL_LAST_K=6, LR=1e-4, LAYER_DECAY=0.75, DROP_PATH=0.1),
]
```

和 01 一样（见 01 §1.1），每个 dict 是一次实验，只写和下面默认值不同的设置。这里一共 4 个实验，扫描训练几个 block：

- `dict(PARTIAL_LAST_K=1)`、`dict(PARTIAL_LAST_K=2)`：只训练最后 1 个、2 个 block，其他用默认设置：学习率 3e-4，不衰减，不用 drop path。K = 2 就是 7 个 notebook 统一的对比设置。
- `dict(PARTIAL_LAST_K=4, LR=1e-4, LAYER_DECAY=0.75, DROP_PATH=0.1)` 和 K = 6 那行：训练的层多了，更接近全量微调，所以改用 01 的配方（学习率 1e-4、逐层衰减 0.75、drop path 0.1），只是把下面的 block 冻住。这样被训练的每一层，学习率和 01 里完全相同。
- 4 个实验在同一次运行里依次跑完。数据只下载一次，但每个实验都会重新提一遍特征（约 7 分钟）。

### 1.2 默认值

```python
# ===== Defaults: the comparison protocol shared by all method notebooks (prototype head, 3 epochs, 2 holdout images per class) =====
METHOD = "partial"
MODEL_NAME = "vit_base_patch16_224.augreg2_in21k_ft_in1k"  # timm name of the base model in the doc (IN21K→IN1K)
HOLDOUT_PER_CLASS = 2    # images per class held out from train for local evaluation (the official val set has no labels)
EPOCHS = 3
WARMUP_EPOCHS = 0.3      # the LR rises linearly for this many epochs, then follows a cosine down to 0
BATCH_SIZE = None        # images per step; None = pick based on VRAM (256 on a 40 or 80 GB GPU)
EFFECTIVE_BATCH = 256    # if one step can't fit it, gradients are accumulated over several steps to reach it
LR = 3e-4
WEIGHT_DECAY = 0.05
PARTIAL_LAST_K = 2       # train only the last K of the 12 blocks (plus the final norm and the head)
LAYER_DECAY = 1.0        # layer-wise LR decay over the trained blocks; 1.0 = off (01 uses 0.75: the last block gets LR × 0.75, the one before LR × 0.75^2, ...)
DROP_PATH = 0.0          # stochastic depth in the trained blocks: residual branches are randomly skipped during training; 0.0 = off (01 uses 0.1)
LABEL_SMOOTHING = 0.1
RRC_SCALE = (0.25, 1.0)  # area range for RandomResizedCrop (training augmentation)
LOG_EVERY = 100          # write a log line every N optimizer steps
SEED = 0
NUM_CLASSES = 10000
```

和 01 不同的有五处，其余（`EPOCHS`、`WARMUP_EPOCHS`、`BATCH_SIZE`、`EFFECTIVE_BATCH`、`WEIGHT_DECAY`、`LABEL_SMOOTHING`、`RRC_SCALE`、`LOG_EVERY` 等）含义和取值都一样，见 01 §1.2。

| 变量 | 01 | 02 | 说明 |
|---|---|---|---|
| `METHOD` | `"full"` | `"partial"` | 方法名，用在实验名、日志和 summary 里 |
| `LR` | `1e-4` | `3e-4` | 所有可训练参数共用的最大学习率 |
| `LAYER_DECAY` | `0.75` | `1.0` | 逐层学习率衰减的系数；1.0 表示不衰减 |
| `DROP_PATH` | `0.1` | `0.0` | drop path 的最大概率；0 表示不用 |
| `PARTIAL_LAST_K` | 没有 | `2` | 训练最后几个 block |

**`PARTIAL_LAST_K = 2`**：训练最后 2 个 block（block 10 和 block 11），另外最后的 LayerNorm 和分类头也训练（这两样是固定的，不受 K 影响）。改成 1、3、4 时参数量见 §0 的表。

**学习率为什么是 3e-4**：介于全量微调（1e-4）和 PEFT 方法（LoRA、Adapter 等用 1e-3）之间。全量微调要改所有层，步子必须小，不然会冲坏预训练特征；这里只改最后两层，可以走得大一点。这是主 notebook 里给这个方法定的默认值，没有专门调过。

**`LAYER_DECAY` 为什么默认 1.0**：逐层衰减是为了让靠前的层少改、靠后的层多改。默认只训练最后两层，靠前的层本来就冻结了，没必要再区分。训练 4 个以上 block 时，被训练的那几层里靠下的更通用、应该少改，就设成 0.75，和 01 一样，具体怎么算见 §7.1。

**`DROP_PATH` 为什么默认 0，打开时为什么只作用在要训练的 block 上**：只训练最后两层时，可训练的参数少，过拟合的风险小，可以不用。训练 4 个以上 block 时建议设成 0.1，和 01 一样。打开以后，drop path 只作用在要训练的 block 上：冻结的 block 没法学着适应“分支被随机跳过”，如果它们也随机跳过，只会给后面的层送去带噪声的输入，而推理时又没有这种噪声。所以 `apply_method` 把冻住的 block 的 drop path 关掉，让它们像一个固定不变的特征提取器（见 §4.3）。

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

和 01 一样，见 07 §1.3。

---

## 2. Setup（§2）

### 2.1 安装 timm 和 hf_xet

```python
# Colab already has torch / torchvision; install timm here, and hf_xet for fast parallel downloads from Hugging Face
!pip install -q timm hf_xet
```

和 01 一样，见 07 §2.1。

### 2.2 哪些设置可以按实验改

```python
# Settings an experiment in EXPERIMENTS may override
RUN_KEYS = ("MODEL_NAME", "HOLDOUT_PER_CLASS", "EPOCHS", "WARMUP_EPOCHS", "BATCH_SIZE", "EFFECTIVE_BATCH",
            "LR", "WEIGHT_DECAY", "PARTIAL_LAST_K", "LAYER_DECAY", "DROP_PATH", "LABEL_SMOOTHING", "RRC_SCALE",
            "LOG_EVERY", "SEED", "NUM_CLASSES")
BASE_CONFIG = {k: globals()[k] for k in RUN_KEYS}
```

机制和 01 一样（见 07 §2.2）：`RUN_KEYS` 是允许在 `EXPERIMENTS` 里修改的设置，`BASE_CONFIG` 是 §1 默认值的快照。和 01 的区别：多了 `PARTIAL_LAST_K`；`LAYER_DECAY` 和 `DROP_PATH` 两边都有。

### 2.3 导入库

```python
import os, re, gc, json, math, time, random, shutil, subprocess
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
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

和 01 相比多了 `torch.nn`（`nn`）：§4.3 的 `apply_method` 用 `nn.Identity()` 关掉冻住的 block 的 drop path。`re`（正则表达式）两边都有，§7.1 的 `build_param_groups` 用它从参数名里读出 block 编号。其余见 01 §2.3 和 07 §2.3。

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

和 01 完全一样，见 01 §2.4，逐行解释见 07 §2.4–§2.8。

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

和 01 完全一样，逐行解释见 07 §3：从 Hugging Face 下载 `train.txt` 和 `train.tar.gz`（`DATA_FROM_HF = False` 时用 Drive 上的备份），解压到 Colab 本地磁盘，检查图片数量，读出 450,000 个（路径, 类别）。

---

## 4. 模型（§4）

### 4.1 `is_head`、`resolve_config`

```python
def is_head(name):
    return name in ("head.weight", "head.bias")


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

和 01 完全一样，见 01 §4.1–§4.2：判断参数是不是分类头的；把默认值和实验的修改合成完整配置，按显存选 `BATCH_SIZE`，算出梯度累积步数 `ACCUM_STEPS`（A100 上是 1）。

### 4.2 `create_model`：和 01 一样带 drop path 参数

```python
def create_model(cfg):
    """Pretrained ViT-B/16 from timm; its 1000-class ImageNet head is replaced by a new, randomly initialized 10,000-class head"""
    return timm.create_model(cfg.MODEL_NAME, pretrained=True, num_classes=cfg.NUM_CLASSES,
                             drop_path_rate=cfg.DROP_PATH).to(DEVICE)
```

- 和 07 一样（见 07 §4.3）：加载预训练的 ViT-B/16，把 1,000 类的头换成新的 10,000 类分类头，搬到 GPU。
- `drop_path_rate=cfg.DROP_PATH`：和 01 一样，把 drop path 的最大概率交给 timm。timm 让 12 个 block 的概率从 0 线性增加到这个值，block 11 最大。默认 `DROP_PATH = 0`，所有 block 都不会跳过分支。
- 设成 0.1 时，§4.3 的 `apply_method` 会把冻住的 block 的 drop path 关掉，只有要训练的 block 保留，概率和 01 里同一个 block 相同。比如 K = 4 时，block 8 到 11 分别是 0.073、0.082、0.091、0.1。为什么这样做见 §1.2。

### 4.3 `apply_method`：只放开最后几层

```python
def apply_method(net, cfg):
    """Selected-layer fine-tuning: freeze everything, then train only the last PARTIAL_LAST_K blocks, the final norm and the head"""
    k = cfg.PARTIAL_LAST_K
    assert 1 <= k <= len(net.blocks), f"PARTIAL_LAST_K must be between 1 and {len(net.blocks)}"
    net.requires_grad_(False)
    for blk in net.blocks[:-k]:
        blk.drop_path1 = blk.drop_path2 = nn.Identity()  # frozen blocks always run in full: drop path only in the trained ones
    for blk in net.blocks[-k:]:
        blk.requires_grad_(True)
    net.norm.requires_grad_(True)
    net.head.requires_grad_(True)
    return net
```

这是 02 和其他方法真正不同的地方。思路是“先全部冻结，再把要训练的部分放开”：

- `k = cfg.PARTIAL_LAST_K`：要训练几个 block，下面几行都用它。
- `assert 1 <= k <= len(net.blocks), ...`：K 必须在 1 到 12 之间，否则直接报错。原因是 K = 0 时，`blocks[-0:]` 就是 `blocks[0:]`，会取到全部 12 个 block，变成训练整个主干，而不是“一个都不训练”；K 大于 12 也没有意义。
- `net.requires_grad_(False)`：先把所有参数（93,488,656 个）都设成不需要梯度，也就是全部冻结。
- `for blk in net.blocks[:-k]:` 和下一行：处理要冻住的那些 block。
  - `net.blocks` 是 12 个 Transformer block 按顺序排成的一串（`nn.Sequential`），可以像列表一样切片。`[:-k]` 表示“从头开始，去掉最后 k 个”，K = 2 时是 block 0 到 block 9。
  - `blk.drop_path1 = blk.drop_path2 = nn.Identity()`：每个 block 有两个 drop path，注意力分支后面一个，MLP 分支后面一个。把它们换成 `nn.Identity()`，也就是原样输出、什么也不做。这样冻住的 block 训练时也始终完整计算，不会随机跳过分支（原因见 §1.2）。`DROP_PATH = 0` 时它们本来就是 `nn.Identity()`，这一行没有影响。
- `for blk in net.blocks[-k:]:`：负数下标从末尾往前数，`[-k:]` 表示最后 k 个。K = 2 时是 block 10 和 block 11；K = 1 时只有 block 11。
- `blk.requires_grad_(True)`：把这个 block 里的所有参数放开训练，包括两个 LayerNorm（`norm1`、`norm2`）、注意力的 `attn.qkv` 和 `attn.proj`、MLP 的 `mlp.fc1` 和 `mlp.fc2`，权重和偏置都在内。每个 block 7,087,872 个参数（明细见 §0）。
- `net.norm.requires_grad_(True)`：放开最后的 LayerNorm（1,536 个参数）。它紧跟在 block 11 后面，对 block 11 的输出做归一化。block 11 变了，它的输出分布也会变，这个 LayerNorm 跟着调整才合理。
- `net.head.requires_grad_(True)`：放开分类头（7,690,000 个参数）。分类头是新的，必须训练。
- `return net`：返回同一个模型（只改了哪些参数要训练，结构没变）。

冻结不动的部分：图块嵌入 `patch_embed`、`cls_token`、位置编码 `pos_embed`，以及 block 0 到 block 9。它们在训练中保持预训练时的样子，也不会被 drop path 随机跳过。

### 4.4 `param_report`

```python
def param_report(net, ctx):
    """Trainable parameter counts. The 7.69M head is counted separately, as in the doc's LoRA reference (0.34%)"""
    n_train = sum(p.numel() for n, p in net.named_parameters() if p.requires_grad and not is_head(n))
    return {"backbone_params": ctx.n_backbone, "head_params": ctx.n_head, "head_mode": "fine-tuned",
            "trainable_excl_head": n_train, "trainable_pct": 100 * n_train / ctx.n_backbone,
            "trainable_incl_head": n_train + ctx.n_head}
```

代码和 01 一样（见 01 §4.5），算出来的数不同：

| key | 值 | 意思 |
|---|---|---|
| `backbone_params` | 85,798,656 | 主干的参数量 |
| `head_params` | 7,690,000 | 分类头的参数量 |
| `head_mode` | `fine-tuned` | 分类头参与训练 |
| `trainable_excl_head` | 14,177,280 | 可训练参数，不含分类头：2 个 block（各 7,087,872）+ 最后的 LayerNorm（1,536） |
| `trainable_pct` | 16.524 | 占主干的百分比 |
| `trainable_incl_head` | 21,867,280 | 可训练参数，含分类头 |

---

## 5. 数据集和 holdout（§5）

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


class Loader(DataLoader):
    """Before starting new worker processes, run garbage collection in the main process. Otherwise an old, interrupted DataLoader is copied into the new workers,
    and when it is collected there it prints lots of harmless AssertionError: can only test a child process messages."""

    def __iter__(self):
        gc.collect()
        return super().__iter__()


def make_loader(items, train, ctx, cfg):
    """Training: random crops, shuffled, BATCH_SIZE images per step. Evaluation: center crops, in order, 2 × BATCH_SIZE (no gradients, so it fits)"""
    return Loader(ImageList(items, ctx.train_tf if train else ctx.eval_tf),
                  batch_size=cfg.BATCH_SIZE if train else 2 * cfg.BATCH_SIZE, shuffle=train, drop_last=train,
                  num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=train)


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

整个格子和 01 完全一样，见 01 §5.1–§5.5：

- `split_holdout`：每类随机留 2 张做 holdout。
- `ImageList`：按编号取图。
- `make_ctx`：训练用随机裁剪加翻转，评估用缩放加中心裁剪。
- `make_loader`：训练每批 256 张、打乱，评估每批 512 张、按顺序。
- `to_input`、`autocast`、`free_gpu`：拷到 GPU 并归一化、bf16 混合精度、清理显存。

前面 10 个 block 虽然冻结了，但训练图片照样要过它们，所以随机裁剪这些数据增强对 02 一样有用：最后两层每个 epoch 看到的输入都不一样。

---

## 6. 分类头初始化（§6）

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

整个格子和 01 完全一样，见 01 §6.1–§6.4：

- `Clock`：GPU 时间计时器。
- `extract_features`、`class_prototypes`、`fit_logit_scale`：用预训练主干给 430,000 张训练图提特征，算出每一类的原型，挑一个温度。
- `init_head`：用这些结果初始化分类头，并返回花了几秒，算进 GPU 时间。

对 02 来说，类原型初始化同样重要：一开始分类头就有 zero-shot 的水平，训练初期的梯度不会太大，最后两层不会一上来就被冲坏。

---

## 7. 训练（§7）

### 7.1 `build_param_groups`：可以按层衰减的参数分组

```python
def build_param_groups(net, cfg):
    """Layer-wise LR decay over the trained part: block i (1–12) gets LR × LAYER_DECAY^(13 − i), the final norm and the head get
    the full LR, frozen blocks are not passed to the optimizer. LAYER_DECAY = 1 gives every trained param the same LR.
    No weight decay on 1-D params (biases, LayerNorm weights)"""
    num_layers = len(net.blocks) + 1
    groups = {}
    for n, p in net.named_parameters():
        if not p.requires_grad:
            continue
        m = re.match(r"blocks\.(\d+)\.", n)
        layer = int(m.group(1)) + 1 if m else num_layers  # outside the blocks only the final norm and the head are trained
        lr = cfg.LR * cfg.LAYER_DECAY ** (num_layers - layer)
        wd = 0.0 if p.ndim <= 1 else cfg.WEIGHT_DECAY
        groups.setdefault((lr, wd), []).append(p)
    return [{"params": ps, "lr": lr, "weight_decay": wd, "base_lr": lr} for (lr, wd), ps in groups.items()]
```

作用和 01 一样：把要训练的参数分成若干组，每组有自己的学习率和权重衰减，交给优化器。算法也和 01 一样（见 01 §7.1），只多了一条：冻住的参数直接跳过。

- `num_layers = len(net.blocks) + 1`：13。层号和 01 一样：block 0 到 11 是第 1 到 12 层，最后的 LayerNorm 和分类头是第 13 层。
- `groups = {}`：key 是（学习率, 权重衰减），value 是这一组的参数列表。
- `for n, p in net.named_parameters():`：遍历所有参数，`n` 是名字，`p` 是参数本身。
- `if not p.requires_grad: continue`：冻住的参数（block 0 到 9、图块嵌入、位置编码、`cls_token`）直接跳过，根本不交给优化器，优化器也就不用给它们存状态，省显存。
- `m = re.match(r"blocks\.(\d+)\.", n)`：用正则表达式从名字里读出 block 编号，比如从 `blocks.10.attn.qkv.weight` 读出 10。正则的写法见 01 §7.1。
- `layer = int(m.group(1)) + 1 if m else num_layers`：block 10 是第 11 层。要训练的参数里，名字不以 `blocks.` 开头的只有最后的 LayerNorm 和分类头，算第 13 层。01 还要判断图块嵌入、`cls_token`、`pos_embed`（第 0 层）；02 里它们冻结着，上一行已经跳过了。
- `lr = cfg.LR * cfg.LAYER_DECAY ** (num_layers - layer)`：和 01 同一个公式。`LAYER_DECAY = 1.0` 时，1 的任何次方都是 1，所有参数都用 `LR`，也就是不衰减。
- `wd = 0.0 if p.ndim <= 1 else cfg.WEIGHT_DECAY`：一维的参数（偏置、LayerNorm 的参数）不做权重衰减，其余衰减 0.05。01 还特别把 `cls_token`、`pos_embed` 排除在外；02 里它们冻结着，不用管。
- `groups.setdefault((lr, wd), []).append(p)`：放进对应的组，组还不存在就先建一个空列表。
- 最后的 `return`：把每一组变成优化器认识的格式。`base_lr` 留着原始学习率，训练时每一步再乘上 warmup/余弦系数（见 01 §7.4）。

**默认设置**（K = 2，`LAYER_DECAY = 1.0`）下所有学习率都一样，所以只分出两组：

| 组 | 权重衰减 | 张量个数 | 参数个数 | 包括 |
|---|---|---|---|---|
| 衰减 | 0.05 | 9 | 21,835,776 | block 10 和 11 的 `qkv`、`proj`、`fc1`、`fc2` 的权重（共 8 个矩阵），加上分类头的权重 |
| 不衰减 | 0 | 19 | 31,504 | block 10 和 11 的所有偏置和 LayerNorm 参数（每个 block 8 个张量），最后的 LayerNorm（2 个），分类头的偏置 |

两组加起来 21,867,280 个参数，正好是可训练参数总数（含分类头）。学习率两组都是 3e-4。

**K = 4、用 01 的配方**（`LR = 1e-4`、`LAYER_DECAY = 0.75`）时，每个要训练的 block 有自己的学习率，和 01 里同一层完全相同：

| 部分 | 学习率 |
|---|---|
| block 8 | 3.16e-5 |
| block 9 | 4.22e-5 |
| block 10 | 5.63e-5 |
| block 11 | 7.50e-5 |
| 最后的 LayerNorm、分类头 | 1.00e-4 |

每个 block 再按“衰减 / 不衰减”分成两组，最后的 LayerNorm 和分类头也是两组，一共 10 组。

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

和 01 完全一样，见 01 §7.2 和 07 §7.1：在 holdout 上算 top-1 和 top-5。

### 7.3 `save_state`：只存训练过的部分

```python
def save_state(net, path, **extra):
    """Save only the trained params: the last blocks, the final norm and the head, about 87 MB"""
    state = {n: p.detach().cpu() for n, p in net.named_parameters() if p.requires_grad or is_head(n)}
    torch.save({"state": state, **extra}, path)
```

- 代码和 01 一样（逐行解释见 07 §7.2），只有 docstring 不同。
- `if p.requires_grad or is_head(n)`：只挑需要训练的参数，也就是 block 10、block 11、最后的 LayerNorm 和分类头。冻结的部分和预训练权重一模一样，不用存。
- 一共 21,867,280 个 float32，每个 4 字节，约 87 MB（01 存全部参数，约 374 MB）。
- 以后要用这个模型：先用 timm 加载同样的预训练 ViT-B/16（10,000 类分类头），再用 `final.pt` 里的这些参数覆盖对应的部分。

### 7.4 `train`：训练循环

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

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(build_param_groups(model, cfg), fused=True)
    train_loader = make_loader(ctx.train_items, True, ctx, cfg)
    holdout_loader = make_loader(ctx.holdout_items, False, ctx, cfg)
    hist = {"step": [], "loss": [], "epoch": [], "epoch_loss": [], "holdout_top1": [], "steps_per_epoch": spe}
    log(f"Training {cfg.EPOCHS} epochs x {spe} steps (batch {cfg.BATCH_SIZE}x{cfg.ACCUM_STEPS}, warmup {warmup} steps)")
    step, speed, t_mark, t_start = 0, None, 0.0, clock.elapsed
    for epoch in range(cfg.EPOCHS):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        run_loss, run_n, ep_loss, ep_n, t_log = 0.0, 0, 0.0, 0, time.time()
        with clock:
            for i, (x, y) in enumerate(train_loader):
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

                # Training speed, measured over steps 20–120 (after the data loader and the GPU have warmed up)
                if step == 20:
                    t_mark = clock.elapsed
                elif step == 120:
                    sec_per_step = (clock.elapsed - t_mark) / 100
                    speed = cfg.BATCH_SIZE * cfg.ACCUM_STEPS / sec_per_step
                    log(f"Speed {speed:.0f} img/s, {sec_per_step * spe / 60:.1f} min/epoch")

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

代码和 01 一字不差，逐行解释见 01 §7.4 的（1）到（5）。同一段代码，在 02 里实际发生的事有几处不同：

- `params = [p for p in model.parameters() if p.requires_grad]`：只有 2,187 万个参数，梯度裁剪 `clip_grad_norm_` 也只看这些参数的梯度。
- `optimizer = torch.optim.AdamW(build_param_groups(model, cfg), fused=True)`：默认只有 2 组参数，学习率一样；打开逐层衰减后组数变多（§7.1）。每一步 `g["lr"] = g["base_lr"] * lr_factor(step)` 让每组都按自己的 `base_lr` 乘上同一个 warmup/余弦系数。
- `model.train()`：默认设置下 02 不用 drop path，ViT-B/16 里也没有 dropout，所以训练模式和评估模式算出来的结果其实一样，切换模式只是好习惯。`DROP_PATH` 大于 0 时就有区别了：训练模式下，要训练的那几个 block 会随机跳过分支；`evaluate` 里切成 `eval()` 以后不跳过。
- `(loss / cfg.ACCUM_STEPS).backward()`：反向传播只走到 block 10 的输入就停了。block 0–9 的参数不需要梯度，它们的输入（图片）也不需要，PyTorch 一开始就不会为它们记录反向传播要用的中间结果。这是 02 预计比 01 快、省显存的主要原因。
- `log(... f"lr {cfg.LR * lr_factor(step):.2e} ...")`：日志里的学习率是 3e-4 乘系数，比如第 100 步是 6.02e-05。

---

## 8. 训练曲线、推理延迟（§8）

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

整个格子和 01 完全一样，见 01 §8.1–§8.3：统一的图表样式，画 `curves.png`（训练损失和每个 epoch 的 holdout top-1），测 batch 1 的推理延迟和 batch 256 的吞吐量。02 只改了最后两层的权重，模型结构没变，推理延迟应该和原来的 ViT-B/16 一样（01 测出来是 7.78 ms）。

---

## 9. 一次实验的完整流程（§9 `run_experiment`）

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
        log(f"{METHOD}: lr {cfg.LR}, wd {cfg.WEIGHT_DECAY}, last {cfg.PARTIAL_LAST_K} blocks, layer decay {cfg.LAYER_DECAY}, drop path {cfg.DROP_PATH}, epochs {cfg.EPOCHS}, "
            f"holdout {cfg.HOLDOUT_PER_CLASS}/class, prototype head")
        log(f"Environment: {GPU_NAME} ({VRAM_GB} GB), bfloat16, {os.cpu_count()} CPUs | {VERSIONS}")
        random.seed(cfg.SEED)
        np.random.seed(cfg.SEED)
        torch.manual_seed(cfg.SEED)
        free_gpu()
```

和 01 §9.1 只差一行：第二个 `log` 记录的设置摘要里，02 多了 `last {cfg.PARTIAL_LAST_K} blocks`，也就是训练最后几个 block；逐层衰减和 drop path 两边都记。默认设置下日志显示为：

```
partial: lr 0.0003, wd 0.05, last 2 blocks, layer decay 1.0, drop path 0.0, epochs 3, holdout 2/class, prototype head
```

`lr 0.0003` 就是 3e-4，Python 打印这个数时默认写成小数。其余几行（建目录、存 `config.json`、设随机种子、清理显存）见 01 §9.1。

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

代码和 01 一样，见 01 §9.2。要注意的是顺序：先用还没冻结的预训练模型提特征、初始化分类头，再 `apply_method` 冻结前 10 个 block。`.to(DEVICE)` 对 02 什么也不改（没有新模块）。这一段的日志是：

```
Trainable (excl. head): 14,177,280 = 16.524% of backbone | head 7,690,000 params (fine-tuned)
```

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

和 01 一样，见 01 §9.3。区别只在结果：`final.pt` 约 87 MB；GPU 时间和峰值显存预计比 01（0.48 GPU-h、18.9 GB）小，要跑出来才知道。

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

和 01 一样，见 01 §9.4：最终 holdout 评估、画曲线、测延迟，写 `summary.json`（字段含义见 01 §9.4 和 07 §8.3）。02 的 `summary.json` 里 `trainable_excl_head` 是 14,177,280，`trainable_pct` 是 16.524，`trainable_incl_head` 是 21,867,280。

---

## 10. 运行实验（§10）

```python
RESULTS = []
for i, exp in enumerate(EXPERIMENTS, 1):
    RESULTS.append(run_experiment(dict(exp), f"{BATCH_NAME}_{i}_{METHOD}_k{exp.get('PARTIAL_LAST_K', PARTIAL_LAST_K)}"))
```

和 01 一样（见 01 §10），只是实验名后面多了 K。`exp.get('PARTIAL_LAST_K', PARTIAL_LAST_K)` 的意思是：这个实验写了 `PARTIAL_LAST_K` 就用它，没写就用默认值。4 个实验依次叫 `1102_1530_1_partial_k1`、`1102_1530_2_partial_k2`、`1102_1530_3_partial_k4`、`1102_1530_4_partial_k6`，结果表里一眼就能看出是哪个 K。

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

和 01 一样，见 01 §11 和 07 §10。可训练参数那一格，K = 1、2、4、6 分别显示 `7,089,408 (8.263%)`、`14,177,280 (16.524%)`、`28,353,024 (33.046%)`、`42,528,768 (49.568%)`。

---

## 12. 自动断开（§12）

```python
if DISCONNECT_WHEN_DONE:
    from google.colab import runtime
    log("All done; releasing the runtime ...")
    drive.flush_and_unmount()  # make sure all results are written to Drive
    runtime.unassign()
```

和 01 一样，见 07 §11。

---

## 附录 A：运行日志逐行对照

在 A100 上运行，日志大致是下面这样。`…` 是要等跑出来才知道的数字；步数和学习率是按公式算好的。4 个实验的日志依次接在一起，每个都从 `==========` 那一行开始，下面以第 2 个实验（K = 2）为例。

```
Batch 1102_1530 -> /content/drive/MyDrive/cs604/batches/1102_1530
Device NVIDIA A100-SXM4-80GB (80 GB), bfloat16, 12 CPUs, … GB RAM, … GB free disk
Versions: torch … (CUDA …), timm …
[train] downloaded from Hugging Face in … min
[train] extracting /content/data/hf/train.tar.gz ...
[train] 450000 images ready in … min
train 450,000 images | 12 loader workers

========== 1102_1530_2_partial_k2 ==========
partial: lr 0.0003, wd 0.05, last 2 blocks, layer decay 1.0, drop path 0.0, epochs 3, holdout 2/class, prototype head
Environment: NVIDIA A100-SXM4-80GB (80 GB), bfloat16, 12 CPUs | torch … (CUDA …), timm …
Head initialized from the class prototypes of 430,000 train images (logit scale …) in … min, counted in this run's GPU time
Trainable (excl. head): 14,177,280 = 16.524% of backbone | head 7,690,000 params (fine-tuned)
Training 3 epochs x 1679 steps (batch 256x1, warmup 503 steps)
ep 1/3 step 100/5037 | loss … | lr 6.02e-05 | … img/s | … min | ETA … min
Speed … img/s, … min/epoch
ep 1/3 step 200/5037 | loss … | lr 1.20e-04 | … img/s | … min | ETA … min
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
== 1102_1530_2_partial_k2 finished: holdout top-1 …% | trainable 14,177,280 (16.524%) | … GPU-h
```

最后还会打印 §11 的结果表。

| 日志 | 出自 | 说明，以及 01 的数字作参考 |
|---|---|---|
| `Batch ...`、`Device ...`、`Versions ...` | §2 | 批次名、运行环境 |
| `[train] downloaded ...`、`extracting ...`、`images ready` | §3 | 每个会话只有第一次运行才有。从 Hugging Face 下载一两分钟；用 Drive 备份时没有 `downloaded` 这一行，直接从 Drive 解压，要 20 分钟左右 |
| `========== ..._2_partial_k2 ==========` | §9 | 实验开始，从这行起也写进 `train_log.txt` |
| `partial: lr 0.0003, ...` | §9 | 设置摘要，训练最后 2 个 block。K = 4、6 时是 `lr 0.0001, ..., layer decay 0.75, drop path 0.1` |
| `Head initialized from ...` | §6 `init_head` | 大约 6–7 分钟，主要是提特征，和 01 一样 |
| `Trainable (excl. head): 14,177,280 ...` | §9 | 16.524% |
| `Training 3 epochs x 1679 steps ...` | §7 `train` | 和 01 相同的步数 |
| `ep 1/3 step 100/5037 ...` | §7 | 每 100 步一行。K = 1、2 的学习率是 01 的 3 倍；K = 4、6 和 01 一样 |
| `Speed ... img/s, ... min/epoch` | §7 | 01 的最终模型是 1,146 img/s；02 预计更快 |
| `== epoch N: ...` | §7 | 每个 epoch 的平均训练损失、holdout top-1 |
| `Training done: 5037 steps, ...` | §9 | 01 是 0.48 GPU-h；02 预计更少 |
| `Peak VRAM ...` | §9 | 01 是 18.9 GB；02 预计少很多 |
| `Final holdout ...` | §9 | 对照：01 是 73.38%，LoRA 是 71.98% |
| `Inference latency ...` | §8 | 应该和 01 一样，01 是 7.78 ms |
| `== ... finished ...` | §9 | 一行总结 |

---

## 附录 B：新术语

通用的术语见 07 附录 B，训练相关的术语见 01 附录 B。02 特有的只有这几个：

| 术语 | 说人话 |
|---|---|
| 冻结（freeze） | 把参数设成不需要梯度（`requires_grad_(False)`），训练时它保持不变 |
| 选层微调（selected-layer / partial fine-tuning） | 只训练模型的一部分层，通常是最后几层，其余冻结 |
| 切片 `[-K:]` | Python 取“倒数第 K 个到最后”的写法，`net.blocks[-2:]` 就是最后 2 个 block |
| 截断的反向传播 | 前面的层都冻结时，反向传播只需要走到第一个要训练的层就停，前面的层只做前向计算 |
| 优化器状态 | AdamW 给每个要训练的参数额外存的两个滑动平均（见 01 附录 B）；冻结的参数不进优化器，也就没有状态 |

---

## 附录 C：讲解时可能被问到的问题

1. **为什么只训练最后几层？** 靠前的层学的是边缘、纹理这些通用特征，换个任务照样能用；靠后的层更贴近具体任务。只放开后面几层，计算和显存都省，也不容易把预训练特征冲坏。
2. **它算 PEFT 吗，满足 5% 的要求吗？** 不算。训练最后 2 个 block 是 16.5%，就算只训练 1 个 block 也有 8.3%，都超过 5%。所以在对比表里它只是参考点。这也说明按“整层”来选，粒度太粗，很难把参数量压到 5% 以下。
3. **学习率为什么是 3e-4，为什么默认不用逐层衰减和 drop path？** 只改最后两层，步子可以比全量微调（1e-4）大一些，但又比只训练小模块的 PEFT 方法（1e-3）小。逐层衰减是为了让前面的层少改，drop path 是为了防过拟合；只训练两层时，前面的层已经冻结，可训练的参数也少，这两样都不太需要。3e-4 是主 notebook 的默认值，没有专门调过。训练 4 个以上 block 时，建议换成 01 的配方：学习率 1e-4、逐层衰减 0.75、drop path 0.1。
4. **为什么最后的 LayerNorm 也要训练？** 它紧跟在最后一个 block 后面，对那个 block 的输出做归一化。最后一个 block 变了，输出的分布也会变，这个 LayerNorm 跟着调整更合理。它只有 1,536 个参数。
5. **为什么会比全量微调快、省显存？** 前向还是要过全部 12 层，但反向传播只走最后 2 层，前面 10 层的中间结果不用存，优化器也只管 2,187 万个参数。这是推测，还没在 GPU 上测过；本机 CPU 上用小模型测，训练速度大约是全量微调的 2.5 倍。
6. **推理会变慢吗？** 不会。模型结构完全没变，只是最后两层的权重变了，推理延迟和原来的 ViT-B/16 一样。
7. **和 LoRA 有什么不同？** LoRA 在全部 12 个 block 里加小的低秩矩阵，原来的权重不动，可训练参数只有 1.375%，训练完还能合并回去。选层微调直接改原来的权重，但只改最后 2 个 block，参数多得多（16.5%）。一个是“每层都改一点”，一个是“只改最后几层、改得彻底”。
8. **跑完要看什么？** 看 holdout top-1 离全量微调（73.38%）有多近，能不能超过 LoRA（71.98%）；再看 GPU 时间、峰值显存和训练速度比 01 省了多少。这几个数正好放进方法对比表。
