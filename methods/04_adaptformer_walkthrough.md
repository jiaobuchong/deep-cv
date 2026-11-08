# 04_adaptformer 逐行讲解

对应文件：`methods/04_adaptformer.ipynb`（精简版）。

04 的大部分代码和 01 完全一样：读数据、切分 holdout、类原型初始化分类头、训练循环、评估、测延迟。这些地方照样贴出代码，但只简单说明，细节见 `01_full_walkthrough.md`（下文写作“见 01 §x”）和 `07_linear_walkthrough.md`（“见 07 §x”）。AdaptFormer 特有的地方逐行讲：§1 的两个新设置，§4 的 `Adapter` 模块、新的前向函数和 `apply_method`，§7 的参数分组和保存，§9 的“推理时多出的参数”。末尾有日志对照、新术语和讲解时可能被问到的问题。

## 0. 先看全局

### AdaptFormer 是什么

- 冻结整个预训练主干，在 12 个 block 里各加一个小“旁路”（adapter）：一个 768 → 64 → 768 的瓶颈 MLP，和 block 原来的 MLP **并联**。只训练这 12 个 adapter 和分类头。
- adapter 最后一层的权重初始化为 0，刚加上时输出是 0，模型和原来完全一样；再加上类原型分类头，一开始就是 zero-shot 的水平（约 44%）。训练中，adapter 慢慢学会往每个 block 的输出里加一点“修正”。
- 出处：Chen et al., *AdaptFormer: Adapting Vision Transformers for Scalable Visual Recognition*, NeurIPS 2022。课程里在 Session 5 Lesson 5（PDF 第 31–32 页）和 CS701 Week 10 tutorial（第 12–14 页）。

### 一个 block 里发生了什么

```
原来的 block（timm）：
  h   = x + 注意力(LN(x))
  out = h + MLP(LN(h))

AdaptFormer 的 block：
  h   = x + 注意力(LN(x))               ← 和原来一样，冻结
  out = h + MLP(LN(h)) + Adapter(h)     ← 多加一项：Adapter 和 MLP 读同一个 h，结果相加（并联）

Adapter(h) = up( dropout( ReLU( down(h) ) ) ) × 0.1
             down：768 → 64，up：64 → 768，只有这两层要训练
```

`x` 是进入这个 block 的 197 个 token（每个 768 维），`LN` 是 LayerNorm。

### 参数量

| 部分 | 怎么算 | 参数 |
|---|---|---|
| `down`（768 → 64） | 768 × 64 + 64 | 49,216 |
| `up`（64 → 768） | 64 × 768 + 768 | 49,920 |
| 一个 adapter | | 99,136 |
| 12 个 adapter | 99,136 × 12 | 1,189,632，占主干 85,798,656 的 1.387% |
| 分类头（另算） | 768 × 10,000 + 10,000 | 7,690,000 |
| 一共训练 | | 8,879,632 |
| 冻结不动的主干 | | 85,798,656 |

### 和 01 比

| | 01_full | 04_adaptformer |
|---|---|---|
| 主干 | 全部训练 | 冻结 |
| 新加的模块 | 无 | 每个 block 一个 adapter |
| 训练的参数（不含分类头） | 85,798,656（100%） | 1,189,632（1.387%） |
| 学习率 | 1e-4，逐层衰减 | 1e-3，所有参数一样 |
| 权重衰减 | 0.05 | 1e-4 |
| drop path | 0.1 | 不用 |
| 推理时多出的参数 | 0 | 1,189,632（adapter 没法合并回主干） |
| 保存的 `final.pt` | 全部参数，约 374 MB | adapter 和分类头，约 36 MB |
| 训练循环、数据增强、类原型初始化 | | 和 01 完全一样 |

### 一次实验的流程（§9 `run_experiment`）

```
建模型（create_model，不带 drop path）
  → 切分数据，准备训练和评估两套预处理（make_ctx）
  → 用类原型初始化分类头（init_head）
  → 冻结主干，在 12 个 block 里插入 adapter，换上新的前向函数（apply_method），再整体搬到 GPU
  → 训练 3 个 epoch，只更新 adapter 和分类头（train）
  → 保存 adapter 和分类头，统计推理时多出的参数
  → 最后在 holdout 上评估，画训练曲线，测推理延迟
  → 写 summary.json
```

### 几个关键数字

- 和 01 一样的对比设置：每个 epoch 1,679 步，3 个 epoch 共 5,037 步，前 503 步 warmup，每步 256 张图。
- 学习率（所有参数都一样）：第 0 步 1.99e-6，第 502 步升到 1e-3；第 1,679 步（第 1 个 epoch 结束）8.43e-4，第 2,770 步 5.00e-4，第 3,358 步（第 2 个 epoch 结束）3.02e-4，最后降到接近 0。
- 之前跑过一次 AdaptFormer（`1018_1239_3_adaptformer`），但分类头用的是 linear probe，不是这里默认的类原型：holdout 69.05%，0.39 GPU-h，峰值显存 12.8 GB，batch 1 推理延迟 9.89 ms，训练速度 1,437 img/s，是我们所有实验里训练最快的。
- 同一批里的 LoRA（加在全部线性层，也是 linear probe 分类头）：holdout 69.45%，885 img/s，19.9 GB，0.55 GPU-h，合并后延迟 7.03 ms。AdaptFormer 的推理延迟比它高约 40%。
- 用类原型分类头的 AdaptFormer 还没跑过。可以对照的类原型实验：全量微调 73.38%，LoRA 71.98%。

---

## 1. Config（§1）

### 1.1 要跑的实验

```python
# ===== Experiments to run: each dict is one run and lists only the settings that differ from the defaults below =====
EXPERIMENTS = [
    {},  # the defaults; not run yet (the earlier run 1018_1239_3_adaptformer used a linear-probe head: 69.05%)
]
```

- 和 01 一样只有一个实验，`{}` 表示全部用下面的默认值，也就是 7 个 notebook 统一的对比设置（见 01 §1.1）。
- 注释提醒：之前那次 AdaptFormer 用的是 linear probe 分类头，69.05%；这个默认设置（类原型分类头）还没跑过。

### 1.2 默认值

```python
# ===== Defaults: the comparison protocol shared by all method notebooks (prototype head, 3 epochs, 2 holdout images per class) =====
METHOD = "adaptformer"
MODEL_NAME = "vit_base_patch16_224.augreg2_in21k_ft_in1k"  # timm name of the base model in the doc (IN21K→IN1K)
HOLDOUT_PER_CLASS = 2    # images per class held out from train for local evaluation (the official val set has no labels)
EPOCHS = 3
WARMUP_EPOCHS = 0.3      # the LR rises linearly for this many epochs, then follows a cosine down to 0
BATCH_SIZE = None        # images per step; None = pick based on VRAM (256 on a 40 or 80 GB GPU)
EFFECTIVE_BATCH = 256    # if one step can't fit it, gradients are accumulated over several steps to reach it
LR = 1e-3
WEIGHT_DECAY = 1e-4
ADAPTER_DIM = 64         # bottleneck size of the adapter: 768 → 64 → 768
ADAPTER_SCALE = 0.1      # the adapter output is multiplied by this (0.1, as in the AdaptFormer paper)
LABEL_SMOOTHING = 0.1
RRC_SCALE = (0.25, 1.0)  # area range for RandomResizedCrop (training augmentation)
LOG_EVERY = 100          # write a log line every N optimizer steps
SEED = 0
NUM_CLASSES = 10000
```

和 01 相比，`HOLDOUT_PER_CLASS`、`EPOCHS`、`WARMUP_EPOCHS`、`BATCH_SIZE`、`EFFECTIVE_BATCH`、`LABEL_SMOOTHING`、`RRC_SCALE`、`LOG_EVERY`、`SEED`、`NUM_CLASSES` 都一样，含义见 01 §1.2。不一样的是这几个：

| 变量 | 01 | 04 | 意思 |
|---|---|---|---|
| `METHOD` | `"full"` | `"adaptformer"` | 方法名，用在实验名、日志和 summary 里 |
| `LR` | `1e-4` | `1e-3` | 学习率，adapter 和分类头都用它 |
| `WEIGHT_DECAY` | `0.05` | `1e-4` | 权重衰减 |
| `LAYER_DECAY` | `0.75` | 没有 | 主干冻结了，不需要逐层衰减 |
| `DROP_PATH` | `0.1` | 没有 | 不用 drop path |
| `ADAPTER_DIM` | 没有 | `64` | adapter 中间那一层的宽度（瓶颈）：768 → 64 → 768 |
| `ADAPTER_SCALE` | 没有 | `0.1` | adapter 的输出乘这个数，和 AdaptFormer 论文一样 |

- **学习率为什么是 1e-3，比全量微调大 10 倍**：adapter 是新加的模块，`up` 层从 0 开始，什么都要从头学；主干是冻结的，学习率大也冲不坏预训练特征。LoRA 用的也是 1e-3。分类头也用 1e-3，它从类原型出发继续训练。
- **权重衰减 1e-4**：这是 PEFT 方法常用的小值（LoRA 也是 1e-4）。可能的原因是要训练的参数很少，不太需要强正则。
- **`ADAPTER_DIM = 64`**：瓶颈越宽，参数越多、能力越强。64 让可训练参数正好是 1.387%，和加在全部线性层的 LoRA（1.375%）差不多，两者比较起来比较公平。
- **`ADAPTER_SCALE = 0.1`**：adapter 的“修正”只按 0.1 倍加进去，和冻结的 MLP 那一支比起来分量小，训练更稳。这是论文里的设置。
- **没有逐层衰减和 drop path**：逐层衰减是为了让主干靠前的层少改，主干冻结后就用不上了；drop path 是训练 block 时用的正则化，这里也没用。

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
            "LR", "WEIGHT_DECAY", "ADAPTER_DIM", "ADAPTER_SCALE", "LABEL_SMOOTHING", "RRC_SCALE", "LOG_EVERY",
            "SEED", "NUM_CLASSES")
BASE_CONFIG = {k: globals()[k] for k in RUN_KEYS}
```

机制和 07 一样（见 07 §2.2）。和 01 的列表相比，`LAYER_DECAY`、`DROP_PATH` 换成了 `ADAPTER_DIM`、`ADAPTER_SCALE`。比如想试更宽的瓶颈，可以在 `EXPERIMENTS` 里加一行 `dict(ADAPTER_DIM=128)`。

### 2.3 导入库

```python
import os, gc, json, math, time, types, random, shutil, subprocess
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

和 01 的区别有三处，其余见 07 §2.3：

- 少了 `re`：04 的 `build_param_groups` 不用从参数名里读出 block 编号（没有逐层衰减），用不到正则表达式。
- 多了 `types`：§4.5 的 `apply_method` 用 `types.MethodType` 把新的前向函数“绑”到每个 block 上。
- 多了 `torch.nn`（`nn`）：§4.1 的 `Adapter` 类继承 `nn.Module`，里面用 `nn.Linear`、`nn.Dropout` 和 `nn.init`。

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

和 01 完全一样，逐行解释见 07 §2.4–§2.8。

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

## 4. 模型和 AdaptFormer（§4）：这个 notebook 的核心

### 4.1 `Adapter`：瓶颈模块

```python
class Adapter(nn.Module):
    """Bottleneck MLP: down → ReLU → up. up starts at zero, so the model output does not change at first after insertion (same as the official AdaptFormer code)"""

    def __init__(self, dim, bottleneck, scale, dropout=0.1):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.up = nn.Linear(bottleneck, dim)
        self.drop = nn.Dropout(dropout)
        self.scale = scale
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return self.up(self.drop(F.relu(self.down(x)))) * self.scale
```

`Adapter` 是一个很小的神经网络：先把 768 维压到 64 维，经过 ReLU，再展开回 768 维，最后乘一个缩放系数。

- `class Adapter(nn.Module):`：PyTorch 里所有带可训练参数的模块都继承 `nn.Module`。这样它的参数能被 `.parameters()` 找到，能被 `.to()` 搬到 GPU，能跟着 `train()` / `eval()` 切换模式。
- docstring：down → ReLU → up；`up` 从 0 开始，所以刚插进去时模型的输出不变，和 AdaptFormer 官方代码一样。
- `def __init__(self, dim, bottleneck, scale, dropout=0.1):`，参数：

| 参数 | 传进来的值 | 意思 |
|---|---|---|
| `dim` | 768（`net.embed_dim`） | 每个 token 的维度，adapter 的输入和输出都是这么宽 |
| `bottleneck` | 64（`ADAPTER_DIM`） | 中间那一层的宽度 |
| `scale` | 0.1（`ADAPTER_SCALE`） | 输出乘的系数 |
| `dropout` | 0.1（默认值，没传） | 训练时中间层随机丢掉的比例 |

- `super().__init__()`：先初始化 `nn.Module` 本身的内部结构。必须在给 `self` 加子模块之前调用。
- `self.down = nn.Linear(dim, bottleneck)`：768 → 64 的线性层，权重 (64, 768) 加偏置 64 个，共 49,216 个参数。把 `nn.Linear` 赋给 `self` 的属性，它就自动登记成 `Adapter` 的子模块。
- `self.up = nn.Linear(bottleneck, dim)`：64 → 768 的线性层，权重 (768, 64) 加偏置 768 个，共 49,920 个参数。
- `self.drop = nn.Dropout(dropout)`：训练时把中间 64 个数随机置 0 一成，没置 0 的除以 0.9，保持期望不变；评估模式下什么都不做。
- `self.scale = scale`：只是一个普通的数，不是参数，不训练。
- 四行 `nn.init`：手动设初始值。
  - `nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))`：`down` 的权重用随机均匀分布初始化（这其实就是 PyTorch `nn.Linear` 默认的初始化公式）。随机是为了让 64 个中间单元学到不同的东西。
  - `nn.init.zeros_(self.down.bias)`：`down` 的偏置设成 0。
  - `nn.init.zeros_(self.up.weight)`、`nn.init.zeros_(self.up.bias)`：`up` 的权重和偏置全设成 0。所以不管输入是什么，adapter 一开始输出全是 0，插进模型后模型的输出完全不变。
  - 全是 0 还能训练吗？能。`up` 的梯度等于“中间层的值 × 从上面传下来的梯度”，中间层的值因为 `down` 是随机的，不是 0，所以第一步 `up` 就有梯度；`down` 的梯度要经过 `up` 的权重传下来，一开始是 0，等 `up` 变得不是 0 以后，`down` 也开始学。
- `def forward(self, x):` 和 `return self.up(self.drop(F.relu(self.down(x)))) * self.scale`，从里往外看：
  1. `self.down(x)`：768 → 64。
  2. `F.relu(...)`：负数变成 0，正数不变。正是这个非线性，让 adapter 没法合并回原来的线性层（见 §9.4）。
  3. `self.drop(...)`：dropout，只在训练时起作用。
  4. `self.up(...)`：64 → 768。
  5. `* self.scale`：乘 0.1。
  - 形状：输入 `x` 是 (B, 197, 768)，中间是 (B, 197, 64)，输出又是 (B, 197, 768)。`nn.Linear` 只作用在最后一维，所以 197 个 token 都经过同一个 adapter。

### 4.2 `adaptformer_block_forward`：换掉 block 的前向计算

```python
def adaptformer_block_forward(self, x, *args, **kwargs):
    # AdaptFormer: the adapter runs in parallel with the FFN; its input is the residual stream before the FFN branch
    x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), *args, **kwargs)))
    return x + self.drop_path2(self.ls2(self.mlp(self.norm2(x)))) + self.adapter(x)
```

这个函数会代替 timm 里每个 Block 原来的 `forward`。作为对照，timm 1.0.30 里 Block 原来的 `forward` 是这样的（不在 notebook 里，只是给你看）：

```
def forward(self, x, attn_mask=None, is_causal=False):
    x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
    x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    return x
```

- `def adaptformer_block_forward(self, x, *args, **kwargs):`：写法和类里的方法一样，第一个参数 `self` 之后会是那个 block。`*args, **kwargs` 把 timm 可能额外传进来的参数（比如新版 timm 的 `attn_mask`、`is_causal`）原样接住，再传给注意力，这样 timm 怎么调用都不会出错。
- 注释：adapter 和 FFN（也就是 MLP）并联，它的输入是 FFN 这一支之前的残差流。
- `x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), *args, **kwargs)))`：和原来的第一行一样：LayerNorm → 多头自注意力 → 加回残差。执行完以后，`x` 就是“注意力之后的残差流”，也就是 §0 图里的 `h`。
- `return x + self.drop_path2(self.ls2(self.mlp(self.norm2(x)))) + self.adapter(x)`：原来的第二行，再多加一项 `self.adapter(x)`：
  - `self.mlp(self.norm2(x))`：冻结的 MLP 那一支，输入先过 LayerNorm。
  - `self.adapter(x)`：adapter 这一支，输入是同一个 `x`，而且不过 LayerNorm，直接读残差流，和官方实现一致。
  - 两支的输出和 `x` 加在一起返回。两支读同一个输入、结果相加，这就是“并联”。05 的 Adapter 是“串联”：adapter 接在 MLP 的输出后面。

几个名字的意思：

| 名字 | 是什么 | 在这里 |
|---|---|---|
| `norm1`、`norm2` | 两个 LayerNorm | 冻结 |
| `attn` | 多头自注意力 | 冻结 |
| `mlp` | 768 → 3072 → 768 的 MLP，中间是 GELU | 冻结 |
| `ls1`、`ls2` | layer scale：有的 ViT 给每一支乘一个可学习的系数 | 这个模型没有，是 `Identity`（原样返回） |
| `drop_path1`、`drop_path2` | drop path | 没开 drop path，也是 `Identity` |
| `adapter` | §4.1 的瓶颈模块 | 唯一要训练的部分（分类头除外） |

### 4.3 `is_head`、`resolve_config`

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

和 01 完全一样，见 01 §4.1–§4.2：判断参数是不是分类头的；合并默认值和实验的修改，按显存选 `BATCH_SIZE`，算梯度累积的步数 `ACCUM_STEPS`（A100 上是 1）。

### 4.4 `create_model`

```python
def create_model(cfg):
    """Pretrained ViT-B/16 from timm; its 1000-class ImageNet head is replaced by a new, randomly initialized 10,000-class head"""
    return timm.create_model(cfg.MODEL_NAME, pretrained=True, num_classes=cfg.NUM_CLASSES).to(DEVICE)
```

和 07 一样（见 07 §4.3），比 01 少了 `drop_path_rate`：不开 drop path，所以 §4.2 表里的 `drop_path1`、`drop_path2` 都是什么都不做的 `Identity`。

### 4.5 `apply_method`：冻结主干，插入 adapter

```python
def apply_method(net, cfg):
    """AdaptFormer: freeze the backbone and add a bottleneck adapter in parallel with the MLP of every block"""
    net.requires_grad_(False)
    for blk in net.blocks:
        blk.adapter = Adapter(net.embed_dim, cfg.ADAPTER_DIM, cfg.ADAPTER_SCALE)
        blk.forward = types.MethodType(adaptformer_block_forward, blk)
    net.head.requires_grad_(True)
    return net
```

- `net.requires_grad_(False)`：先把模型所有参数都冻结，包括分类头。
- `for blk in net.blocks:`：遍历 12 个 block。
- `blk.adapter = Adapter(net.embed_dim, cfg.ADAPTER_DIM, cfg.ADAPTER_SCALE)`：新建一个 adapter，挂到这个 block 上。
  - `net.embed_dim` 是 768。
  - 把一个 `nn.Module` 赋给另一个 `nn.Module` 的属性，它就自动登记成子模块。它的参数会出现在 `net.parameters()` 里，名字像 `blocks.0.adapter.down.weight`；也会跟着 `.to()`、`train()`、`eval()` 一起变。
  - 新建的参数默认 `requires_grad=True`，所以 adapter 是可训练的。上一行冻结的时候它还不存在，不受影响。
  - 新建的模块在 CPU 上，所以 §9.2 里要再 `.to(DEVICE)` 一次，把它们搬到 GPU。
- `blk.forward = types.MethodType(adaptformer_block_forward, blk)`：把这个 block 的前向计算换成 §4.2 的函数。
  - `types.MethodType(函数, 对象)` 把一个普通函数“绑”到某个对象上，调用时自动把这个对象当成第一个参数 `self` 传进去。
  - 赋给 `blk.forward` 以后，PyTorch 调用 `blk(x)` 时会去找 `blk.forward`，先找到这个对象自己的属性，于是用的是新函数。
  - 这种在运行时替换方法的做法叫“猴子补丁”（monkey patching）。只改了这 12 个 block 对象，timm 的 Block 类本身没变。
- `net.head.requires_grad_(True)`：把分类头重新设为可训练（第一行把它也冻结了）。
- `return net`。

执行完以后，要训练的是 12 个 adapter（1,189,632 个参数）加分类头（7,690,000 个），一共 8,879,632 个；主干的 85,798,656 个参数全部冻结。

### 4.6 `param_report`

```python
def param_report(net, ctx):
    """Trainable parameter counts. The 7.69M head is counted separately, as in the doc's LoRA reference (0.34%)"""
    n_train = sum(p.numel() for n, p in net.named_parameters() if p.requires_grad and not is_head(n))
    return {"backbone_params": ctx.n_backbone, "head_params": ctx.n_head, "head_mode": "fine-tuned",
            "trainable_excl_head": n_train, "trainable_pct": 100 * n_train / ctx.n_backbone,
            "trainable_incl_head": n_train + ctx.n_head}
```

代码和 01 一样（见 01 §4.5），结果不同：

| key | 值 | 意思 |
|---|---|---|
| `backbone_params` | 85,798,656 | 原来主干的参数量 |
| `head_params` | 7,690,000 | 分类头的参数量 |
| `head_mode` | `fine-tuned` | 分类头参与训练 |
| `trainable_excl_head` | 1,189,632 | 可训练参数，不含分类头，也就是 12 个 adapter |
| `trainable_pct` | 1.387 | 占原来主干的百分比 |
| `trainable_incl_head` | 8,879,632 | 可训练参数，含分类头 |

注意 `ctx.n_backbone` 是在 `make_ctx` 里数的，那时还没插 adapter（§9.2 里 `make_ctx` 在 `apply_method` 之前），所以它是原来主干的 85,798,656 个，百分比也是相对原来的主干算的。

---

## 5. 数据集和 holdout（§5）

这一节的代码和 01 完全一样。

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

见 01 §5.1：每类随机留 2 张做 holdout（种子固定），按编号取图并做预处理，缩放用双三次插值。

### 5.2 `make_ctx`

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

见 01 §5.2：训练用随机裁剪加翻转，评估用缩放加中心裁剪；顺便数出主干和分类头的参数量。

### 5.3 `Loader`、`make_loader`

```python
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
```

见 01 §5.3–§5.4：训练时打乱、每批 256 张、丢掉最后不满的一批；评估时按顺序、每批 512 张。

### 5.4 `to_input`、`autocast`、`free_gpu`

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

见 01 §5.5：拷到 GPU 并归一化；bf16 混合精度；清理显存并把峰值显存清零。adapter 的计算也在 `autocast` 里，用 bf16。

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

这一节的代码和 01 完全一样：

- `Clock`：累计 GPU 时间，见 01 §6.1。
- `extract_features`、`class_prototypes`、`fit_logit_scale`：用冻结的主干提特征，算每一类的原型，挑温度，见 01 §6.2–§6.3（更细的解释见 07 §6.1–§6.3）。
- `init_head`：用类原型初始化分类头，返回花的秒数，见 01 §6.4。

对 AdaptFormer 来说，有一点值得注意：分类头是在插入 adapter **之前**初始化的（§9.2 里先 `init_head` 再 `apply_method`），提特征用的是原来的预训练主干。因为 adapter 一开始输出全是 0，插进去以后模型的预测和 zero-shot 完全一样，所以分类头和主干在训练开始时是“对得上”的。

---

## 7. 训练（§7）

### 7.1 `build_param_groups`：两组参数

```python
def build_param_groups(net, cfg):
    """All trainable params use LR. No weight decay on 1-D params (biases, LayerNorm weights)"""
    decay, no_decay = [], []
    for p in net.parameters():
        if p.requires_grad:
            (no_decay if p.ndim <= 1 else decay).append(p)
    return [{"params": ps, "lr": cfg.LR, "weight_decay": wd, "base_lr": cfg.LR}
            for ps, wd in ((decay, cfg.WEIGHT_DECAY), (no_decay, 0.0)) if ps]
```

比 01 简单得多：没有逐层衰减，所有可训练参数用同一个学习率，只按“要不要权重衰减”分成两组。

- docstring：所有可训练参数都用 `LR`；一维的参数（偏置、LayerNorm 的参数）不做权重衰减。
- `decay, no_decay = [], []`：两个空列表，分别放要衰减和不衰减的参数。
- `for p in net.parameters():`：遍历所有参数。这里只看形状，不需要参数名，所以不用 `named_parameters()`，也不用正则表达式。
- `if p.requires_grad:`：只要可训练的参数。冻结的主干直接跳过，优化器不用为它们保存状态，省显存。
- `(no_decay if p.ndim <= 1 else decay).append(p)`：括号里是条件表达式，一维的参数放进 `no_decay`，其余放进 `decay`，然后 `.append(p)` 加进选中的那个列表。
- `return [...]`：列表推导式，对（参数列表, 权重衰减）这两对各生成一组：`{"params": ..., "lr": cfg.LR, "weight_decay": wd, "base_lr": cfg.LR}`。`if ps` 跳过空的那组。`base_lr` 留给训练循环每一步乘 warmup / 余弦系数用（见 01 §7.1）。

结果（在随机初始化的 ViT-B 上实际跑过）：

| 组 | 权重衰减 | 包括 | 张量数 | 参数 |
|---|---|---|---|---|
| 1 | 1e-4 | 24 个 adapter 权重（每个 adapter 的 `down` 和 `up`）+ 分类头权重 | 25 | 8,859,648 |
| 2 | 0 | 24 个 adapter 偏置 + 分类头偏置 | 25 | 19,984 |

两组的学习率都是 1e-3。和 01 相比，也不再需要特别处理 `cls_token`、`pos_embed`，因为它们被冻结了，根本不在这两组里。

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

和 01 一样，见 01 §7.2。里面的 `net.eval()` 会关掉 adapter 里的 dropout。

### 7.3 `save_state`

```python
def save_state(net, path, **extra):
    """Save only the trained params: the adapters and the head, about 36 MB"""
    state = {n: p.detach().cpu() for n, p in net.named_parameters() if p.requires_grad or is_head(n)}
    torch.save({"state": state, **extra}, path)
```

- 代码和 01 一样，只有 docstring 不同：保存“需要梯度”或者“属于分类头”的参数。04 里需要梯度的就是 12 个 adapter，所以存的是 adapter 加分类头：8,879,632 个 float32，约 36 MB。
- 主干用的是预训练权重，一点没变，不用存。

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

代码和 01 一模一样，逐行解释见 01 §7.4。在 AdaptFormer 里效果有几处不同：

- 优化器只管 adapter 和分类头（8,879,632 个参数），AdamW 要存的两份状态也只有这么多，比全量微调的 9,349 万个少得多。
- 反向传播还是要从分类头一路传回第 1 个 block，因为最前面的 adapter 也要拿到梯度。但冻结的大线性层不用算自己权重的梯度，有些中间结果也不用为它们留着。这可能就是之前那次训练更快（1,437 img/s）、显存更少（12.8 GB）的原因。
- `model.train()` 打开 adapter 里的 dropout；没有 drop path。
- 学习率按 `lr_factor` 变化，最大 1e-3，具体数字见 §0。

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

这一节的代码和 01 完全一样：

- `style_axis`、`plot_run_curves`：画 `curves.png`，左边是训练损失，右边是每个 epoch 末的 holdout top-1，见 01 §8.1–§8.2。
- `measure_latency`、`latency_report`：用随机输入测 batch 1 的推理延迟和 batch 256 的吞吐量，见 01 §8.3（更细的解释见 07 §7.3–§7.4）。

对 AdaptFormer 来说，延迟是一个要讲的点：adapter 合并不回主干，推理时每个 block 都要多算一次 adapter。之前那次测出来 batch 1 延迟 9.89 ms，比同一批里合并后的 LoRA（7.03 ms）慢约 40%。按矩阵乘法算，adapter 多出来的计算量其实很小：每个 token 在每个 block 多 768 × 64 + 64 × 768 = 98,304 次乘加，而原来的 block 大约 738 万次，只多 1.3% 左右。所以慢的原因可能主要不是计算量：batch 1 时每次运算都很小，GPU 启动一次运算的固定开销占了大头，12 个 adapter 多出几十次小运算（两个线性层、ReLU、乘 0.1、相加），一个接一个排队执行，时间就上去了。

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
        log(f"{METHOD}: lr {cfg.LR}, wd {cfg.WEIGHT_DECAY}, adapter dim {cfg.ADAPTER_DIM}, scale {cfg.ADAPTER_SCALE}, epochs {cfg.EPOCHS}, "
            f"holdout {cfg.HOLDOUT_PER_CLASS}/class, prototype head")
        log(f"Environment: {GPU_NAME} ({VRAM_GB} GB), bfloat16, {os.cpu_count()} CPUs | {VERSIONS}")
        random.seed(cfg.SEED)
        np.random.seed(cfg.SEED)
        torch.manual_seed(cfg.SEED)
        free_gpu()
```

和 01 §9.1 一样，只有配置那行日志不同：记录学习率、权重衰减、adapter 的瓶颈宽度和缩放系数，比如 `adaptformer: lr 0.001, wd 0.0001, adapter dim 64, scale 0.1, epochs 3, holdout 2/class, prototype head`。

### 9.2 建模型、初始化分类头、插入 adapter

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

代码和 01 §9.2 一样，但在 AdaptFormer 里有几处要特别注意：

- **顺序**：`make_ctx` 和 `init_head` 都在 `apply_method` 之前。所以 `ctx.n_backbone` 数的是没有 adapter 的原主干（85,798,656 个），类原型也是用原来的预训练主干算的（见 §6）。
- **`.to(DEVICE)` 这一步必不可少**：`apply_method` 新建的 12 个 adapter 在 CPU 上，模型其余部分在 GPU 上。`.to(DEVICE)` 把整个模型（包括新插的 adapter）搬到 GPU。少了这一步，第一次前向就会报错：输入在 GPU 上，adapter 的权重却在 CPU 上。全量微调没有新模块，这一步什么都不改；在这里它是必需的。
- 参数统计那一行日志会是 `Trainable (excl. head): 1,189,632 = 1.387% of backbone | head 7,690,000 params (fine-tuned)`。

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

和 01 §9.3 一样：训练 3 个 epoch，存 `final.pt`（这里约 36 MB），存 `history.json`，记录 GPU 时间和峰值显存。之前那次 AdaptFormer 是 0.39 GPU-h、峰值 12.8 GB。

### 9.4 推理时多出的参数、最终评估和汇总

```python
        # Params the model has at inference on top of the original ViT-B/16 + head
        params["extra_params_at_inference"] = sum(p.numel() for p in model.parameters()) - ctx.n_backbone - ctx.n_head

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

前两行是 01 没有的：

- 注释：模型在推理时，比原来的 ViT-B/16 加分类头多出多少参数。
- `params["extra_params_at_inference"] = sum(p.numel() for p in model.parameters()) - ctx.n_backbone - ctx.n_head`：现在模型的参数总数，减去原来主干的 85,798,656 个，再减去分类头的 7,690,000 个，剩下的就是 12 个 adapter：1,189,632 个。这个数会写进 `summary.json`。
  - 为什么 adapter 合并不回去：`up(ReLU(down(x)))` 中间有 ReLU，是非线性的，没法写成“原来的权重加一个矩阵”。LoRA 的 `B·A·x` 是线性的，训练完可以直接加回原来的权重，所以 LoRA 这一项是 0。
  - 全量微调没有新参数，这一项也是 0（01 里干脆没有这一行）。

其余几行和 01 §9.4 一样：训练完在 holdout 上再测一次 top-1 和 top-5，画训练曲线，测推理延迟，汇总写进 `summary.json`。`summary` 里的字段和 01 相同，多了 `extra_params_at_inference`（因为 `**params` 把它带进来了）。

---

## 10. 运行实验（§10）

```python
RESULTS = []
for i, exp in enumerate(EXPERIMENTS, 1):
    RESULTS.append(run_experiment(dict(exp), f"{BATCH_NAME}_{i}_{METHOD}"))
```

和 01 一样，见 01 §10。实验名比如 `1102_1530_1_adaptformer`。

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

和 01 一样，见 01 §11：可训练参数（这里会显示 `1,189,632 (1.387%)`）、训练速度、GPU 时间、峰值显存、延迟、吞吐量，存成 `results.md` 和 `results.json`。

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

在 A100 上运行，日志大致是下面这样。`…` 是要等跑出来才知道的数字，学习率是按公式算好的（最大学习率 1e-3）。

```
Batch 1102_1530 -> /content/drive/MyDrive/cs604/batches/1102_1530
Device NVIDIA A100-SXM4-80GB (80 GB), bfloat16, 12 CPUs, … GB RAM, … GB free disk
Versions: torch … (CUDA …), timm …
[train] downloaded from Hugging Face in … min
[train] extracting /content/data/hf/train.tar.gz ...
[train] 450000 images ready in … min
train 450,000 images | 12 loader workers

========== 1102_1530_1_adaptformer ==========
adaptformer: lr 0.001, wd 0.0001, adapter dim 64, scale 0.1, epochs 3, holdout 2/class, prototype head
Environment: NVIDIA A100-SXM4-80GB (80 GB), bfloat16, 12 CPUs | torch … (CUDA …), timm …
Head initialized from the class prototypes of 430,000 train images (logit scale …) in … min, counted in this run's GPU time
Trainable (excl. head): 1,189,632 = 1.387% of backbone | head 7,690,000 params (fine-tuned)
Training 3 epochs x 1679 steps (batch 256x1, warmup 503 steps)
ep 1/3 step 100/5037 | loss … | lr 2.01e-04 | … img/s | … min | ETA … min
Speed … img/s, … min/epoch
ep 1/3 step 200/5037 | loss … | lr 4.00e-04 | … img/s | … min | ETA … min
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
== 1102_1530_1_adaptformer finished: holdout top-1 …% | trainable 1,189,632 (1.387%) | … GPU-h
```

最后还会打印 §11 的结果表。下表的参考值来自之前那次用 linear probe 分类头的 AdaptFormer（`1018_1239_3_adaptformer`），准确率不能直接比，速度、显存和延迟可以参考。

| 日志 | 出自 | 说明，参考值 |
|---|---|---|
| `Batch ...`、`Device ...`、`Versions ...` | §2 | 批次名、运行环境 |
| `[train] downloaded ...`、`extracting ...`、`images ready` | §3 | 每个会话只有第一次运行才有。从 Hugging Face 下载一两分钟；用 Drive 备份时没有 `downloaded` 这一行，直接从 Drive 解压，要 20 分钟左右 |
| `========== ..._1_adaptformer ==========` | §9 | 实验开始，从这行起也写进 `train_log.txt` |
| `adaptformer: lr ...` | §9 | 设置摘要 |
| `Head initialized from ...` | §6 `init_head` | 大约 6–7 分钟，主要是提特征 |
| `Trainable (excl. head): 1,189,632 = 1.387% ...` | §9 | 12 个 adapter |
| `Training 3 epochs x 1679 steps ...` | §7 `train` | 和 01 一样 |
| `ep 1/3 step 100/5037 ...` | §7 | 每 100 步一行，一共约 50 行 |
| `Speed ... img/s, ... min/epoch` | §7 | 之前是 1,437 img/s |
| `== epoch N: ...` | §7 | 每个 epoch 一行：平均训练损失、holdout top-1、已用 GPU 时间 |
| `Training done: 5037 steps, ...` | §9 | 之前是 0.39 GPU-h |
| `Peak VRAM ...` | §9 | 之前是 12.8 GB |
| `Final holdout ...` | §9 | 之前（linear probe 分类头）是 69.05% |
| `Inference latency ...` | §8 | 之前是 9.89 ms |
| `== ... finished ...` | §9 | 一行总结，`train_log.txt` 写到这行为止 |

类原型分类头在 LoRA 和全量微调上都比 linear probe 分类头高了约 2.5 个点（LoRA 69.45% → 71.98%，全量微调 70.88% → 73.38%）。AdaptFormer 用类原型很可能也会更高，但还没跑过，要等结果出来才知道。

---

## 附录 B：新术语

基础的术语见 07 的附录 B，训练相关的术语见 01 的附录 B。下面是 adapter 特有的：

| 术语 | 说人话 |
|---|---|
| adapter | 插进冻结模型里的小模块，只训练它，原模型不动 |
| 瓶颈（bottleneck） | 先压缩到很小的维度（64），再展开回原来的维度（768），所以参数很少 |
| 并联 vs 串联 | 并联：adapter 和原来的 MLP 读同一个输入，结果相加（AdaptFormer）；串联：adapter 接在 MLP 的输出后面（05 的 Adapter） |
| 零初始化 | 把输出层的权重设成 0，刚插入时不改变模型，训练从原模型的水平开始 |
| 残差流（residual stream） | 每个 block 都往里“加东西”的那条主干道：x → 加上注意力的输出 → 加上 MLP 的输出 → ……；adapter 读的就是它 |
| 子模块注册 | 把一个 `nn.Module` 赋给另一个 `nn.Module` 的属性，它的参数就会被自动登记，跟着保存、搬到 GPU、切换训练和评估模式 |
| 猴子补丁（monkey patching） | 在运行时替换某个对象的方法。这里用 `types.MethodType` 把新的 `forward` 绑到每个 block 上 |
| `types.MethodType` | 把一个普通函数绑定到某个对象上，调用时自动把这个对象当成 `self` 传进去 |
| `Identity` | 什么都不做、原样返回输入的层 |
| layer scale | 有的 ViT 给每一支的输出乘一个可学习的系数（`ls1`、`ls2`）。这个模型没有，是 `Identity` |
| 可合并 vs 不可合并 | LoRA 是线性的，训练完可以加回原来的权重，推理没有额外开销；adapter 中间有 ReLU，是非线性的，推理时只能一直留着 |
| kernel 启动开销 | GPU 每做一次运算都要启动一次“内核”。batch 1 时每次运算都很小，启动的固定开销占比就很大 |

---

## 附录 C：讲解时可能被问到的问题

1. **AdaptFormer 和 LoRA 有什么区别？** 两者都冻结主干、只加很少的可训练参数（1.387% 和 1.375%）。LoRA 在线性层旁边加两个“瘦”矩阵，是线性的，训练完能合并回原来的权重，推理没有额外开销。AdaptFormer 在每个 block 的 MLP 旁边并联一个带 ReLU 的瓶颈模块，是非线性的，合并不回去，推理时多 1,189,632 个参数。我们测到它的 batch 1 延迟比 LoRA 高约 40%。
2. **为什么 `up` 层初始化成 0？** 这样刚插进去时 adapter 输出 0，模型和原来完全一样，加上类原型分类头就是 zero-shot 的水平，训练从一个好的起点平稳开始。LoRA 把 B 矩阵初始化成 0，也是同样的道理。
3. **为什么 adapter 的输出要乘 0.1？** 让 adapter 的“修正”和冻结的 MLP 那一支比起来分量小一些，训练更稳。这是 AdaptFormer 论文里的设置，05 的 Adapter 用的是 1.0。
4. **为什么学习率比全量微调大 10 倍？** adapter 是从零开始学的新模块；主干冻结了，学习率大也不会冲坏预训练的特征。
5. **为什么它训练比 LoRA 快、显存也少？** 之前那次 AdaptFormer 是 1,437 img/s、12.8 GB，同一批的 LoRA 是 885 img/s、19.9 GB。可能的原因：我们的 LoRA 加在每个 block 的全部 4 个线性层上，训练时这 48 个地方都要多算一条 LoRA 支路，也要多存中间结果；AdaptFormer 只有 12 个小模块。再加上冻结的大线性层不用算自己权重的梯度。
6. **为什么推理反而更慢？** adapter 合并不回去，推理时每个 block 都要多算一遍。多出来的计算量只有约 1.3%，但 batch 1 时每次运算都很小，GPU 启动运算的固定开销占了大头，12 个 adapter 多出几十次小运算，一个接一个执行，时间就上去了。这是可能的原因，没有专门测过。
7. **1.387% 满足项目“少于约 5%”的要求吗？** 满足。和项目文档里博士生的 LoRA 参考（0.34%）一样，可训练参数不算分类头，因为分类头是每个方法都必须训练的新层。
8. **04 和 05（Adapter）有什么区别？** 参数量完全一样（1,189,632），区别只在两处：AdaptFormer 和 MLP 并联、输出乘 0.1；05 的 Adapter 串在 MLP 的输出后面，输出乘 1.0。
