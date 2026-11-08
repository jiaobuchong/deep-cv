# 03_lora 逐行讲解

对应文件：`methods/03_lora.ipynb`（精简版）。

03 的大部分代码和 01 完全一样：读数据、数据集和预处理、类原型初始化分类头、训练循环、画图、测延迟、结果表。这些地方照样贴出代码，只简单说明，细节见 `01_full_walkthrough.md`（下文写作“见 01 §x”）或 `07_linear_walkthrough.md`（“见 07 §x”）。和 01 不一样的地方逐行讲：LoRA 的设置（§1）、安装和导入 peft（§2）、怎么把 LoRA 插进模型（§4）、优化器的参数分组和保存（§7）、训练完把 LoRA 合并回原权重（§9）。末尾有日志对照、LoRA 术语和讲解时可能被问到的问题。

## 0. 先看全局

### LoRA 是什么

主干全部冻结，不改原来的权重。在选定的线性层旁边加一条“旁路”：先用矩阵 A 把输入压到 r = 8 维，再用矩阵 B 放大回原来的输出维度，乘以 α/r = 16/8 = 2，加到原来的输出上。只训练 A 和 B（和新的分类头）。

公式：`输出 = W·x + (α/r)·B·A·x`，其中 W 是冻结的原权重，A 和 B 是新加的、要训练的小矩阵。

以 qkv 层（768 维输入，2304 维输出）为例：

```
输入 x（768 维）
  ├──► W：原来的线性层，冻结 ─────────────────────────► W·x（2304 维）──┐
  │                                                                     （+）──► 输出（2304 维）
  └──► A：768 → 8（训练）──► B：8 → 2304（训练）──► 乘以 α/r = 2 ──────┘
```

- **为什么叫“低秩”**：B·A 是一个 2304 × 768 的矩阵，相当于对 W 的一个改动，但它的秩最多是 8。用 8 × 768 + 2304 × 8 = 24,576 个参数，表示一个本来要 2304 × 768 = 1,769,472 个参数的改动，只有 1.4%。
- **B 初始化为 0**：刚开始旁路的输出是 0，模型和预训练模型完全一样，再加上类原型分类头，起点就是 zero-shot 的水平（约 44%）。
- **训练完可以合并**：把 `(α/r)·B·A` 算出来直接加进 W，旁路就可以删掉了。推理时的模型和原来的 ViT-B/16 一模一样，没有额外的计算。

### 参数量

LoRA 加在每个 block 的四个线性层上（`LORA_TARGETS` 的默认值），r = 8：

| 层 | 输入 → 输出 | 原权重 W 的参数 | LoRA 的参数（A + B） |
|---|---|---|---|
| `attn.qkv` | 768 → 2304 | 1,769,472 | 8×768 + 2304×8 = 24,576 |
| `attn.proj` | 768 → 768 | 589,824 | 8×768 + 768×8 = 12,288 |
| `mlp.fc1` | 768 → 3072 | 2,359,296 | 8×768 + 3072×8 = 30,720 |
| `mlp.fc2` | 3072 → 768 | 2,359,296 | 8×3072 + 768×8 = 30,720 |
| 每个 block 合计 | | | 98,304 |
| 12 个 block 合计 | | | **1,179,648**（主干的 1.375%） |

只加在 qkv 上（项目文档里博士生的参考设置，68.3%）是 24,576 × 12 = 294,912 个，占 0.344%。新的分类头另有 7,690,000 个参数，每个方法都要训练，不算在里面。

### 和 01 比有什么不同

| | 01 全量微调 | 03 LoRA |
|---|---|---|
| 可训练参数（不含分类头） | 85,798,656（100%） | 1,179,648（1.375%） |
| 学习率 | 1e-4，逐层衰减 0.75 | 1e-3，所有可训练参数一样 |
| 权重衰减 | 0.05 | 1e-4 |
| drop path | 0.1 | 不用 |
| 新增的模块 | 无 | 48 条 LoRA 旁路（12 个 block × 4 个层） |
| 用到的库 | timm | timm + peft |
| 训练完 | 直接用 | 把 LoRA 合并回原权重 |
| 保存的权重 `final.pt` | 约 374 MB | 约 35 MB |
| holdout top-1（同样设置跑过的结果） | 73.38% | 71.98% |
| GPU 时间 | 0.48 GPU-h | 0.54 GPU-h |
| 峰值显存（allocated） | 18.9 GB | 19.9 GB |
| 训练速度 | 1,146 img/s（最终模型那次） | 888 img/s |
| 推理延迟 | 7.78 ms | 7.61 ms |

其余全部一样：数据、holdout、预处理和数据增强、类原型初始化分类头、训练循环、标签平滑、梯度裁剪、warmup 加余弦退火、3 个 epoch。

### 一次实验的流程（§9 `run_experiment`）

```
建模型（create_model，不带 drop path）
  → 切分数据，准备训练和评估两套预处理（make_ctx）
  → 用类原型初始化分类头（init_head）
  → 冻结主干，插入 48 条 LoRA 旁路，分类头设为可训练（apply_method）
  → 训练 3 个 epoch，每个 epoch 结束在 holdout 上测一次（train）
  → 保存 LoRA 和分类头的权重（约 35 MB）
  → 把 LoRA 合并回原权重（merge_and_unload）
  → 用合并后的模型在 holdout 上评估、画训练曲线、测推理延迟
  → 写 summary.json
```

### 关键数字

- 训练设置和 01 一样：每步 256 张图，每个 epoch 1,679 步，3 个 epoch 共 5,037 步，前 503 步 warmup。
- 学习率（LoRA 和分类头都一样）：第 0 步 2.0e-6，第 502 步升到 1e-3，第 1,679 步（第 1 个 epoch 结束）8.4e-4，第 2,770 步 5.0e-4，第 3,358 步（第 2 个 epoch 结束）3.0e-4，最后降到接近 0。
- 之前用同样设置跑过（`1019_1319_3_lora`，这也是我们最终的 PEFT 模型）：
  - 每个 epoch 末的 holdout top-1：62.94% → 70.12% → 71.97%；最终 holdout top-1 71.98%，top-5 89.07%；CodaBench val **71.90%**。
  - 0.54 GPU-h，峰值显存 19.9 GB（allocated）、21.3 GB（reserved），推理延迟 7.61 ms，训练速度 888 img/s，每个 epoch 8.1 分钟。

---

## 1. Config（§1）

### 1.1 要跑的实验

```python
# ===== Experiments to run: each dict is one run and lists only the settings that differ from the defaults below =====
EXPERIMENTS = [
    {},  # the defaults; already run as 1019_1319_3_lora (the final PEFT model): holdout 71.98%
]
```

- 和 01 一样（见 01 §1.1）：`{}` 表示全部用默认值，也就是 7 个 notebook 统一的对比设置。
- 默认设置就是我们的最终 PEFT 模型 `1019_1319_3_lora` 的设置，holdout 71.98%。
- 想加变体，就在列表里加 dict。比如 `dict(LORA_TARGETS=["qkv"])` 只在 qkv 上加 LoRA（项目文档里博士生的设置），`dict(LORA_R=16, LORA_ALPHA=32)` 把秩加大到 16，缩放系数仍是 2。

### 1.2 默认值

```python
# ===== Defaults: the comparison protocol shared by all method notebooks (prototype head, 3 epochs, 2 holdout images per class) =====
METHOD = "lora"
MODEL_NAME = "vit_base_patch16_224.augreg2_in21k_ft_in1k"  # timm name of the base model in the doc (IN21K→IN1K)
HOLDOUT_PER_CLASS = 2    # images per class held out from train for local evaluation (the official val set has no labels)
EPOCHS = 3
WARMUP_EPOCHS = 0.3      # the LR rises linearly for this many epochs, then follows a cosine down to 0
BATCH_SIZE = None        # images per step; None = pick based on VRAM (256 on a 40 or 80 GB GPU)
EFFECTIVE_BATCH = 256    # if one step can't fit it, gradients are accumulated over several steps to reach it
LR = 1e-3
WEIGHT_DECAY = 1e-4
LORA_R = 8               # rank of the low-rank matrices A (r × in) and B (out × r)
LORA_ALPHA = 16          # the LoRA output B·A·x is scaled by LORA_ALPHA / LORA_R = 2
LORA_TARGETS = ["qkv", "proj", "fc1", "fc2"]  # linear layers of every block that get LoRA (the final PEFT model uses all four)
LABEL_SMOOTHING = 0.1
RRC_SCALE = (0.25, 1.0)  # area range for RandomResizedCrop (training augmentation)
LOG_EVERY = 100          # write a log line every N optimizer steps
SEED = 0
NUM_CLASSES = 10000
```

| 变量 | 默认值 | 意思 | 用在哪里 |
|---|---|---|---|
| `METHOD` | `"lora"` | 方法名，用在实验名、日志和 summary 里 | §9–§11 |
| `MODEL_NAME` | ViT-B/16（IN21K→IN1K） | 模型和预训练权重，见 07 §1.2 | §4 `create_model` |
| `HOLDOUT_PER_CLASS` | `2` | 每类留几张做 holdout | §5 |
| `EPOCHS` | `3` | 训练几个 epoch | §7 `train` |
| `WARMUP_EPOCHS` | `0.3` | warmup 的长度，见 01 §1.2 | §7 `train` |
| `BATCH_SIZE` | `None` | 每步多少张图，自动选，A100 上是 256 | §4、§5 |
| `EFFECTIVE_BATCH` | `256` | 每次更新用多少张图，放不下就梯度累积，见 01 §1.2 | §4、§7 |
| `LR` | `1e-3` | 学习率，LoRA 的 A、B 和分类头都用它 | §7 |
| `WEIGHT_DECAY` | `1e-4` | 权重衰减 | §7 `build_param_groups` |
| `LORA_R` | `8` | 秩 r：旁路中间压缩到几维 | §4 `apply_method` |
| `LORA_ALPHA` | `16` | α：旁路的输出乘以 α/r = 2 | §4 `apply_method` |
| `LORA_TARGETS` | `["qkv", "proj", "fc1", "fc2"]` | 每个 block 里哪些线性层加 LoRA | §4 `apply_method` |
| `LABEL_SMOOTHING` | `0.1` | 标签平滑，见 07 §1.2 | §7 `train` |
| `RRC_SCALE` | `(0.25, 1.0)` | 随机裁剪的面积范围，见 01 §1.2 | §5 `make_ctx` |
| `LOG_EVERY` | `100` | 每 100 步写一行日志 | §7 `train` |
| `SEED` | `0` | 随机种子 | §5、§9 |
| `NUM_CLASSES` | `10000` | 类别数 | §4、§6 |

和 01 比，少了 `LAYER_DECAY` 和 `DROP_PATH`，多了三个 LoRA 的设置。下面逐个说明。

**`LORA_R = 8`：秩**

旁路中间那一层的宽度。A 的形状是 (r, 输入维度)，B 是 (输出维度, r)。r 越大，LoRA 能表示的改动越复杂，参数也越多：参数量和 r 成正比，r = 16 就是 2,359,296 个。

**`LORA_ALPHA = 16`：缩放**

旁路的输出要乘以 `α / r` = 16 / 8 = 2。这样写的好处：改 r 的时候把 α 跟着按比例改，缩放系数不变，学习率一般也不用重新调。

**`LORA_TARGETS`：加在哪些层**

每个 block 里有四个线性层：

| 简称 | timm 里的层名 | 作用 |
|---|---|---|
| `qkv` | `attn.qkv` | 一次算出注意力用的 Q、K、V（768 → 2304，2304 = 3 × 768） |
| `proj` | `attn.proj` | 把注意力的输出再变换一次（768 → 768） |
| `fc1` | `mlp.fc1` | MLP 的第一层，放大到 3072 维 |
| `fc2` | `mlp.fc2` | MLP 的第二层，缩回 768 维 |

我们的实验发现，LoRA 的“容量”比学习率重要（都用 lp 分类头、3 个 epoch）：只加在 qkv 上是 68.57%；把学习率降到 5e-4 是 68.45%，几乎没变；四个层都加是 69.45%。所以默认四个层都加。

**`LR = 1e-3`：为什么比全量微调大 10 倍**

- 主干是冻结的，学习率再大也不会冲坏预训练的特征。
- B 从 0 开始，要训练的参数又少，用小学习率学得太慢。
- 分类头也用 1e-3，和 07 里 linear probe 一样。
- 所有可训练参数用同一个学习率，没有逐层衰减：主干冻结了，没什么可以“分层”的。

**`WEIGHT_DECAY = 1e-4`**

比全量微调的 0.05 小得多。LoRA 的参数本来就少，正则化的需要也小。分类头的权重也用这个衰减，偏置不衰减，见 §7.1。

**为什么不用 drop path**：drop path 主要用来给全量微调这种参数很多的训练做正则化。LoRA 只训练很少的参数，我们沿用了不加 drop path 的设置（`create_model` 里没有 `drop_path_rate`）。

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

和 07、01 一样，见 07 §1.3。

---

## 2. Setup（§2）

### 2.1 安装 timm、peft 和 hf_xet

```python
# Colab already has torch / torchvision; install timm and peft here, and hf_xet for fast parallel downloads from Hugging Face
!pip install -q timm peft hf_xet
# Colab's preinstalled torchao is too old, and new peft raises ImportError when creating LoRA. We don't need torchao, so just uninstall it
!pip uninstall -y -q torchao
```

- `!pip install -q timm peft hf_xet`：除了 timm，还要装 peft（Parameter-Efficient Fine-Tuning），这是 Hugging Face 的一个库，专门做 LoRA 这类参数高效的微调；hf_xet 让 §3 从 Hugging Face 并行下载数据，见 07 §2.1。`-q` 少输出一些信息。以 `!` 开头的行是 shell 命令，见 07 §2.1。
- `!pip uninstall -y -q torchao`：卸载 torchao。注释说明了原因：Colab 预装的 torchao 版本太旧，新版 peft 在创建 LoRA 时会因为它报 `ImportError`。我们用不到 torchao，直接卸掉最省事。`-y` 表示不用确认，直接卸载。

### 2.2 哪些设置可以按实验改

```python
# Settings an experiment in EXPERIMENTS may override
RUN_KEYS = ("MODEL_NAME", "HOLDOUT_PER_CLASS", "EPOCHS", "WARMUP_EPOCHS", "BATCH_SIZE", "EFFECTIVE_BATCH",
            "LR", "WEIGHT_DECAY", "LORA_R", "LORA_ALPHA", "LORA_TARGETS", "LABEL_SMOOTHING", "RRC_SCALE",
            "LOG_EVERY", "SEED", "NUM_CLASSES")
BASE_CONFIG = {k: globals()[k] for k in RUN_KEYS}
```

机制和 07、01 一样（见 07 §2.2）。列表里多了 `LORA_R`、`LORA_ALPHA`、`LORA_TARGETS`，少了 01 的 `LAYER_DECAY`、`DROP_PATH`。

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
import peft
from peft import LoraConfig, get_peft_model
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from huggingface_hub import hf_hub_download
from google.colab import drive
```

和 01 的区别只有两行（其余见 07 §2.3、01 §2.3）：

- `import peft`：只用来在日志里记下 peft 的版本号（`peft.__version__`）。
- `from peft import LoraConfig, get_peft_model`：`LoraConfig` 描述要加什么样的 LoRA（秩、缩放、加在哪些层……），`get_peft_model` 按这个描述把 LoRA 插进模型。见 §4.4。
- `re`（正则表达式）还在，但用处和 01 不同：01 用它从参数名里读 block 编号；03 用它在 `is_head` 里匹配分类头的参数名，见 §4.2。
- 和 01 一样不需要 `torch.nn`。

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
VERSIONS = f"torch {torch.__version__} (CUDA {torch.version.cuda}), timm {timm.__version__}, peft {peft.__version__}"

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

和 01 完全一样（逐行解释见 07 §2.4–§2.8），只有 `VERSIONS` 这一行多了 peft 的版本：`..., timm {timm.__version__}, peft {peft.__version__}`。LoRA 的实现来自 peft，版本不同可能导致结果有细微差别，所以要记进日志。之前的实验用的是 peft 0.21.1。

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

和 01、07 完全一样，逐行解释见 07 §3：从 Hugging Face 下载 `train.txt` 和 `train.tar.gz`（`DATA_FROM_HF = False` 时用 Drive 上的备份），解压到 Colab 本地磁盘，检查图片数量，再读出 450,000 个（路径, 类别）。

---

## 4. 模型和 LoRA（§4）：这个 notebook 的核心

### 4.1 `LORA_MODULES`：简称对应的层名

```python
LORA_MODULES = {"qkv": r"attn\.qkv", "proj": r"attn\.proj", "fc1": r"mlp\.fc1", "fc2": r"mlp\.fc2"}  # LORA_TARGETS → layer names in timm
```

- 一个 dict，把 `LORA_TARGETS` 里的简称换成 timm 模型里真正的层名。比如 `"qkv"` 对应 `attn.qkv`，也就是每个 block 里 `attn` 模块下的 `qkv` 线性层。
- 值写成正则表达式：`\.` 表示一个真正的点号。在正则里单独的 `.` 能匹配任意字符，所以要加反斜杠。
- 字符串前面的 `r` 表示原始字符串（raw string）：里面的反斜杠原样保留，不会被 Python 当成转义符。写正则时几乎都要加 `r`。
- 后面 `apply_method` 用它拼出交给 peft 的匹配规则。

### 4.2 `is_head`：用正则判断分类头

```python
def is_head(name):
    # peft renames the parameters to base_model.model.<name>, so match the end of the name
    return re.search(r"(^|\.)head\.(weight|bias)$", name) is not None
```

作用和 01 一样：判断一个参数是不是分类头的。写法不同，是因为 peft 会改参数名。

- 插入 LoRA 以后，模型被 peft 包了一层，所有参数名前面都多了 `base_model.model.`。比如分类头的权重从 `head.weight` 变成 `base_model.model.head.weight`。01 那种写法 `name in ("head.weight", "head.bias")` 就认不出来了，分类头会被当成主干参数，统计和保存都会出错。
- 正则 `(^|\.)head\.(weight|bias)$` 拆开看：
  - `(^|\.)`：要么是字符串的开头（`^`），要么是一个点号（`\.`）。`|` 表示“或”。
  - `head\.`：字面上的 `head.`。
  - `(weight|bias)`：`weight` 或 `bias`。
  - `$`：字符串的结尾。
  - 合起来：名字以 `head.weight` 或 `head.bias` 结尾，并且 `head` 前面要么什么都没有，要么是点号。这样 `head.weight` 和 `base_model.model.head.weight` 都能认出来。
- `re.search(...)` 在字符串里找匹配，找到返回一个匹配对象，找不到返回 `None`。`is not None` 把结果变成 True / False。
- 注释的意思：peft 把参数改名成 `base_model.model.<原来的名字>`，所以要匹配名字的结尾。

### 4.3 `resolve_config`

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

和 01 完全一样，见 01 §4.2：检查设置名，合并默认值和实验的修改，按显存选 `BATCH_SIZE`，算梯度累积的步数 `ACCUM_STEPS`。

### 4.4 `create_model`

```python
def create_model(cfg):
    """Pretrained ViT-B/16 from timm; its 1000-class ImageNet head is replaced by a new, randomly initialized 10,000-class head"""
    return timm.create_model(cfg.MODEL_NAME, pretrained=True, num_classes=cfg.NUM_CLASSES).to(DEVICE)
```

和 07 一样（见 07 §4.3）：加载预训练的 ViT-B/16，换上新的 10,000 类分类头。和 01 的区别是没有 `drop_path_rate`，也就是不用 drop path，原因见 §1.2。

### 4.5 `apply_method`：把 LoRA 插进模型

```python
def apply_method(net, cfg):
    """LoRA: freeze the backbone and add low-rank matrices next to the chosen linear layers. The returned model is wrapped by peft"""
    net.requires_grad_(False)
    pattern = r"blocks\.\d+\.(" + "|".join(LORA_MODULES[t] for t in cfg.LORA_TARGETS) + ")"
    net = get_peft_model(net, LoraConfig(r=cfg.LORA_R, lora_alpha=cfg.LORA_ALPHA, lora_dropout=0.0,
                                         target_modules=pattern, bias="none"))
    net.get_base_model().head.requires_grad_(True)  # peft freezes everything except LoRA, so unfreeze the new head again
    return net
```

参数：`net` 是已经初始化好分类头的模型，`cfg` 是配置。返回：被 peft 包了一层的模型（`PeftModel`）。

- `net.requires_grad_(False)`：先把所有参数（包括分类头）都冻结。
- `pattern = r"blocks\.\d+\.(" + "|".join(LORA_MODULES[t] for t in cfg.LORA_TARGETS) + ")"`：拼出一个正则，告诉 peft 哪些层要加 LoRA。
  - `LORA_MODULES[t] for t in cfg.LORA_TARGETS`：把四个简称换成层名：`attn\.qkv`、`attn\.proj`、`mlp\.fc1`、`mlp\.fc2`。
  - `"|".join(...)`：用 `|`（“或”）连起来。
  - 前后拼上 `blocks\.\d+\.(` 和 `)`。`\d+` 是一个或多个数字，表示任意一个 block 的编号。
  - 最后得到：`blocks\.\d+\.(attn\.qkv|attn\.proj|mlp\.fc1|mlp\.fc2)`。
  - peft 拿它去和每个模块的完整名字比，要**整个名字**都匹配才算。所以 `blocks.0.attn.qkv`、`blocks.11.mlp.fc2` 会匹配；`blocks.0.attn`（注意力模块本身）、`head`、`patch_embed.proj` 都不匹配。12 个 block × 4 个层，一共 48 个线性层。
- `net = get_peft_model(net, LoraConfig(...))`：peft 做了三件事：
  1. 找到所有名字匹配的线性层（48 个），把每一个换成 peft 的 LoRA 线性层。新的层里面保留原来的层（叫 `base_layer`，权重不变），再加上 `lora_A`、`lora_B` 两个小线性层和缩放系数 2。
  2. 只把 LoRA 的参数设为可训练，其余参数全部冻结，**包括新的分类头**。
  3. 外面包一层 `PeftModel` 返回。
- `LoraConfig` 的参数：

| 参数 | 值 | 意思 |
|---|---|---|
| `r` | 8 | 秩 |
| `lora_alpha` | 16 | 缩放系数 = 16 / 8 = 2 |
| `lora_dropout` | 0.0 | 旁路输入前的 dropout 概率；0 表示不用 |
| `target_modules` | 上面拼出的正则 | 哪些层加 LoRA。传一个字符串时，peft 把它当正则，要求和模块名完全匹配 |
| `bias` | `"none"` | 偏置一律不训练，保持冻结 |

- 初始化：`lora_A` 用随机值（Kaiming 均匀分布），`lora_B` 全是 0。所以刚插入时旁路的输出是 0，模型的输出和插入前完全一样。
- `net.get_base_model().head.requires_grad_(True)`：`get_base_model()` 取出 peft 里面包着的原来的 timm 模型，把它的分类头重新设为可训练。注释说明了原因：peft 把 LoRA 以外的参数全冻结了，新的分类头必须训练，所以要再打开。
- `return net`：返回 `PeftModel`。之后训练、评估都直接用它，调用方式和普通模型一样。

插入以后参数名长这样（`default` 是 peft 给这组 LoRA 起的名字）：

| 参数名 | 形状 |
|---|---|
| `base_model.model.blocks.0.attn.qkv.lora_A.default.weight` | (8, 768) |
| `base_model.model.blocks.0.attn.qkv.lora_B.default.weight` | (2304, 8) |
| `base_model.model.blocks.0.attn.proj.lora_A.default.weight` | (8, 768) |
| `base_model.model.blocks.0.attn.proj.lora_B.default.weight` | (768, 8) |
| …（12 个 block × 4 个层 × A、B 两个，共 96 个） | |
| `base_model.model.head.weight`、`base_model.model.head.bias` | (10000, 768)、(10000,) |

### 4.6 `param_report`

```python
def param_report(net, ctx):
    """Trainable parameter counts. The 7.69M head is counted separately, as in the doc's LoRA reference (0.34%)"""
    n_train = sum(p.numel() for n, p in net.named_parameters() if p.requires_grad and not is_head(n))
    return {"backbone_params": ctx.n_backbone, "head_params": ctx.n_head, "head_mode": "fine-tuned",
            "trainable_excl_head": n_train, "trainable_pct": 100 * n_train / ctx.n_backbone,
            "trainable_incl_head": n_train + ctx.n_head}
```

代码和 01 完全一样（见 01 §4.5），结果不同：

| key | 值 | 意思 |
|---|---|---|
| `backbone_params` | 85,798,656 | 主干的参数量（不变） |
| `head_params` | 7,690,000 | 分类头的参数量 |
| `head_mode` | `fine-tuned` | 分类头参与训练 |
| `trainable_excl_head` | 1,179,648 | 可训练参数，不含分类头，也就是 96 个 LoRA 矩阵 |
| `trainable_pct` | 1.375 | 占主干的百分比，低于项目建议的 5% |
| `trainable_incl_head` | 8,869,648 | 含分类头 |

能把分类头和 LoRA 分开数对，靠的就是 §4.2 那个用正则的 `is_head`。

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

和 01 完全一样，逐行解释见 01 §5（以及 07 §5）：

- `split_holdout`：每类随机留 2 张做 holdout，种子固定。
- `ImageList`、`BICUBIC`：按编号取图，双三次插值。
- `make_ctx`：两套预处理。训练用随机裁剪加翻转，评估用缩放加中心裁剪。见 01 §5.2。
- `Loader`、`make_loader`：训练时每批 256 张、打乱、丢掉最后不满的一批；评估时每批 512 张、按顺序。见 01 §5.4。
- `to_input`、`autocast`、`free_gpu`：拷到 GPU 并归一化、bf16 混合精度、清理显存。

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

和 01 完全一样，逐行解释见 01 §6：`Clock` 统计 GPU 时间；`extract_features` 用还没加 LoRA 的预训练主干给训练图片提特征；`class_prototypes` 和 `fit_logit_scale` 算出类原型和温度；`init_head` 把它们写进分类头。

注意：这一步在 `apply_method` 之前做，那时模型里还没有 LoRA。就算在插入 LoRA 以后再提特征，结果也一样，因为 B 是 0、旁路输出 0。

分类头的初始化对 LoRA 同样重要。我们在 LoRA 上专门做过对比（四个层都加、3 个 epoch）：随机初始化的分类头 71.95%，类原型 72.00%（这两个是在 RTX PRO 6000 上跑的），linear probe 分类头 69.45%（A100）。也就是说，对 LoRA 来说类原型和随机差不多，linear probe 起步明显更差。我们的推测是：linear probe 就是在这批训练图片的冻结特征上拟合出来的，一开始训练损失已经很低，给 LoRA 的训练信号偏弱。

---

## 7. 训练（§7）

### 7.1 `build_param_groups`：所有可训练参数用同一个学习率

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

01 版本（见 01 §7.1）按层分出 28 组、每组学习率不同。LoRA 不需要逐层衰减，这个版本简单得多：只按“要不要做权重衰减”分成两组。

- `decay, no_decay = [], []`：两个空列表，分别放要衰减和不衰减的参数。
- `for p in net.parameters():`：遍历所有参数。这里不需要名字，只看形状。
- `if p.requires_grad:`：只要可训练的参数。冻结的主干参数（85,798,656 个）根本不交给优化器，所以优化器不用为它们保存任何状态，这是 LoRA 省显存的地方之一（见附录 C 第 5 条）。
- `(no_decay if p.ndim <= 1 else decay).append(p)`：条件表达式先选出一个列表，再往里加。一维的参数（偏置）放 `no_decay`，其余放 `decay`。
- `return [...]`：列表推导式，最多生成两组：
  - `for ps, wd in ((decay, cfg.WEIGHT_DECAY), (no_decay, 0.0))`：依次处理（要衰减的参数, 1e-4）和（不衰减的参数, 0）。
  - `if ps`：如果某一组是空的就跳过。这个函数 02 到 06 的 notebook 都用，不同方法的可训练参数不一样，得防着某一组是空的。
  - 每组都是 `{"params": ..., "lr": cfg.LR, "weight_decay": wd, "base_lr": cfg.LR}`。`base_lr` 的作用和 01 一样：训练时每一步按 warmup / 余弦系数重新算学习率。

在 LoRA 里分出来的两组：

| 组 | 参数 | 个数 | 学习率 | 权重衰减 |
|---|---|---|---|---|
| 要衰减 | 96 个 LoRA 矩阵（各个 `lora_A`、`lora_B`）+ 分类头的权重 | 97 个张量，8,859,648 个数 | 1e-3 | 1e-4 |
| 不衰减 | 分类头的偏置 | 1 个张量，10,000 个数 | 1e-3 | 0 |

LoRA 设了 `bias="none"`，所以主干里的偏置都是冻结的，唯一可训练的偏置就是分类头的。

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

和 01、07 完全一样，见 07 §7.1。训练中每个 epoch 末用它测 holdout top-1，这时模型还是带 LoRA 旁路的 `PeftModel`；训练完以后的最终评估用的是合并以后的模型，见 §9。

### 7.3 `save_state`：只存 LoRA 和分类头

```python
def save_state(net, path, **extra):
    """Save only the trained params: the LoRA matrices and the head, about 35 MB"""
    state = {n: p.detach().cpu() for n, p in net.named_parameters() if p.requires_grad or is_head(n)}
    torch.save({"state": state, **extra}, path)
```

代码和 01 一样（见 07 §7.2），只有 docstring 不同，但存下来的东西很不一样：

- 保存“需要梯度”或者“属于分类头”的参数。LoRA 里需要梯度的就是 96 个 LoRA 矩阵和分类头，一共 8,869,648 个 float32，约 35 MB。01 要存全部 93,488,656 个参数，约 374 MB。
- 存下来的参数名带着 peft 的前缀，比如 `base_model.model.blocks.0.attn.qkv.lora_A.default.weight`。
- 要用这个文件时，先建一个同样的模型、用同样的设置插入 LoRA，再把这些参数加载进去。主干的权重从预训练模型里来，所以不用存。
- 注意它是在合并 **之前** 调用的（见 §9.3）。合并以后 LoRA 矩阵就不存在了，再调用只会存下分类头。

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

和 01 完全一样，逐段解释见 01 §7.4。放到 LoRA 上，有几处结果不同：

- `params = [p for p in model.parameters() if p.requires_grad]`：只有 96 个 LoRA 矩阵和分类头，共 8,869,648 个参数。梯度裁剪（总长度不超过 1.0）也只在这些参数上做。
- `optimizer = torch.optim.AdamW(build_param_groups(model, cfg), fused=True)`：只有 §7.1 的两组参数。
- `lr_factor` 一样，只是乘的最大学习率是 1e-3：第 100 步的日志会显示 `lr 2.01e-04`，第 1,679 步（第 1 个 epoch 结束）是 8.43e-4。
- `model.train()` / `evaluate` 里的 `eval()`：LoRA 没有 dropout（`lora_dropout=0.0`），主干也没开 drop path，所以训练模式和评估模式的前向其实一样。切换模式只是保持写法统一。
- 前向和反向：图片还是要完整地过一遍 12 个 block。反向传播时，冻结的权重不用算梯度，但梯度还得一层层穿过它们传回去，才能到达靠前那些 block 里的 LoRA 矩阵。
- 训练速度：之前测出来 888 img/s，每个 epoch 8.1 分钟，比全量微调（1,146 img/s）还慢，原因见附录 C 第 5 条。

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

和 01 完全一样，逐行解释见 01 §8：

- `style_axis`、`plot_run_curves`：画 `curves.png`，左边是训练损失，右边是每个 epoch 末的 holdout top-1。
- `measure_latency`、`latency_report`：用随机输入测 batch 1 的推理延迟和 batch 256 的吞吐量。

测延迟用的是合并以后的模型（见 §9.4），结构和原来的 ViT-B/16 一样，所以延迟也一样：之前测出来 7.61 ms，全量微调是 7.78 ms，两者差别很小。

---

## 9. 一次实验的完整流程（§9 `run_experiment`）

参数：`overrides` 是这次实验的修改（默认是空的 `{}`），`run_name` 是实验名。返回：`summary`，这次实验的全部结果。分五段看。

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
        log(f"{METHOD}: lr {cfg.LR}, wd {cfg.WEIGHT_DECAY}, r {cfg.LORA_R}, alpha {cfg.LORA_ALPHA}, targets {'+'.join(cfg.LORA_TARGETS)}, epochs {cfg.EPOCHS}, "
            f"holdout {cfg.HOLDOUT_PER_CLASS}/class, prototype head")
        log(f"Environment: {GPU_NAME} ({VRAM_GB} GB), bfloat16, {os.cpu_count()} CPUs | {VERSIONS}")
        random.seed(cfg.SEED)
        np.random.seed(cfg.SEED)
        torch.manual_seed(cfg.SEED)
        free_gpu()
```

和 01 §9.1 一样：合成配置，建目录，日志开始写进 `train_log.txt`，存 `config.json`，设随机种子，清理显存。区别只在第二行 `log`：记录 LoRA 的设置。

- `r {cfg.LORA_R}, alpha {cfg.LORA_ALPHA}`：秩和 α。
- `targets {'+'.join(cfg.LORA_TARGETS)}`：把加 LoRA 的层用加号连起来，显示成 `qkv+proj+fc1+fc2`。这一段在 f-string 里，所以里面的字符串用单引号，免得和外面的双引号冲突。

### 9.2 建模型、初始化分类头、插入 LoRA

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

代码和 01 §9.2 一样，但 `apply_method` 做的事不同：

- `model = create_model(cfg)`、`ctx = make_ctx(model, cfg)`、`clock = Clock()`、`clock.used += init_head(...)`：建模型、准备数据、用类原型初始化分类头，同 01。
- `model = apply_method(model, cfg).to(DEVICE)`：冻结主干、插入 48 条 LoRA 旁路、重新打开分类头，返回 `PeftModel`。peft 会把新建的 LoRA 矩阵放到和原来的层同一个设备上，所以 `.to(DEVICE)` 在这里通常什么也不改，只是保险。
- `params = param_report(model, ctx)` 和下一行：这时 LoRA 还在，统计的是 1,179,648 个 LoRA 参数，日志显示 `Trainable (excl. head): 1,179,648 = 1.375% of backbone`。必须在合并之前统计，合并以后就数不到了。

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

和 01 §9.3 一样：训练 3 个 epoch，存 `final.pt` 和 `history.json`，记下 GPU 时间和峰值显存。

- `save_state(...)` 在合并 **之前**：存下 LoRA 矩阵和分类头，约 35 MB，见 §7.3。
- 之前同样设置是 0.54 GPU-h，峰值显存 19.9 GB（allocated）、21.3 GB（reserved）。

### 9.4 合并 LoRA

```python
        model = model.merge_and_unload()  # merge LoRA into the base weights, so inference costs nothing extra
        # Params the model has at inference on top of the original ViT-B/16 + head
        params["extra_params_at_inference"] = sum(p.numel() for p in model.parameters()) - ctx.n_backbone - ctx.n_head
```

这是 03 独有的两行。

- `model = model.merge_and_unload()`：把 LoRA 合并回原来的权重，再“拆掉”peft 的包装：
  1. 对 48 个加了 LoRA 的层，每一个都算 `W ← W + (α/r)·B·A`，也就是把 `2 × B·A` 直接加进原权重。以 qkv 为例：B 是 (2304, 8)，A 是 (8, 768)，B·A 是 (2304, 768)，正好和 W 一样大。
  2. 把 peft 的 LoRA 层换回普通的线性层（权重已经是合并以后的）。
  3. 去掉外面的 `PeftModel`，返回原来的 timm `VisionTransformer`。参数总数回到 93,488,656，和没加 LoRA 时一样。
  - 合并前后，模型对同一张图的输出是一样的（只有极小的数值差别）。我们在本机用随机数验证过：合并以后的权重正好等于 `W + 2·B·A`，输出也一致。
- 下一行的注释：统计推理时比原来的 ViT-B/16 加分类头多了多少参数。
- `params["extra_params_at_inference"] = sum(p.numel() for p in model.parameters()) - ctx.n_backbone - ctx.n_head`：合并以后模型的参数总数，减去主干和分类头，结果是 **0**。也就是说，推理时没有任何额外的参数和计算，这是 LoRA 和 Adapter / AdaptFormer 最大的区别：后两者的模块合并不进去，推理会变慢。

### 9.5 最终评估和汇总

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

代码和 01 §9.4 一样，只是现在用的是合并以后的模型：

- `evaluate(...)`：用合并后的模型在 holdout 上测 top-1 和 top-5。之前那次：第 3 个 epoch 末（还没合并）是 71.97%，合并以后最终评估是 71.98%，top-5 89.07%。差的 0.01 个点，很可能来自合并后 bf16 计算的细微数值差别。
- `plot_run_curves`、`latency_report`：画曲线，测延迟（7.61 ms）。
- `summary`：字段和 01 一样，多了 `extra_params_at_inference`（0），它是在 §9.4 加进 `params` 的，通过 `**params` 带进来。

---

## 10. 运行实验（§10）

```python
RESULTS = []
for i, exp in enumerate(EXPERIMENTS, 1):
    RESULTS.append(run_experiment(dict(exp), f"{BATCH_NAME}_{i}_{METHOD}"))
```

和 01 一样，见 01 §10。实验名比如 `1102_1530_1_lora`。

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

和 01 一样，见 01 §11：把结果整理成表，存成 `results.md` 和 `results.json`，并在 notebook 里显示出来。

---

## 12. 自动断开（§12）

```python
if DISCONNECT_WHEN_DONE:
    from google.colab import runtime
    log("All done; releasing the runtime ...")
    drive.flush_and_unmount()  # make sure all results are written to Drive
    runtime.unassign()
```

和 07、01 一样，见 07 §11。

---

## 附录 A：运行日志逐行对照

在 A100 上运行，日志大致是下面这样。`…` 是要等跑出来才知道的数字，学习率是按公式算好的。

```
Batch 1102_1530 -> /content/drive/MyDrive/cs604/batches/1102_1530
Device NVIDIA A100-SXM4-80GB (80 GB), bfloat16, 12 CPUs, … GB RAM, … GB free disk
Versions: torch … (CUDA …), timm …, peft …
[train] downloaded from Hugging Face in … min
[train] extracting /content/data/hf/train.tar.gz ...
[train] 450000 images ready in … min
train 450,000 images | 12 loader workers

========== 1102_1530_1_lora ==========
lora: lr 0.001, wd 0.0001, r 8, alpha 16, targets qkv+proj+fc1+fc2, epochs 3, holdout 2/class, prototype head
Environment: NVIDIA A100-SXM4-80GB (80 GB), bfloat16, 12 CPUs | torch … (CUDA …), timm …, peft …
Head initialized from the class prototypes of 430,000 train images (logit scale …) in … min, counted in this run's GPU time
Trainable (excl. head): 1,179,648 = 1.375% of backbone | head 7,690,000 params (fine-tuned)
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
== 1102_1530_1_lora finished: holdout top-1 …% | trainable 1,179,648 (1.375%) | … GPU-h
```

最后还会打印 §11 的结果表。下表的参考值来自之前同样设置的 `1019_1319_3_lora`：

| 日志 | 出自 | 说明，参考值 |
|---|---|---|
| `Versions: ... peft ...` | §2 | 多了 peft 的版本，之前是 0.21.1 |
| `lora: lr 0.001, ...` | §9.1 | LoRA 的设置摘要 |
| `Head initialized from ...` | §6 `init_head` | 大约 6–7 分钟，主要是提特征 |
| `Trainable (excl. head): 1,179,648 = 1.375% ...` | §9.2 | 96 个 LoRA 矩阵 |
| `Training 3 epochs x 1679 steps ...` | §7 `train` | 和 01 一样 |
| `ep 1/3 step 100/5037 ...` | §7 | 每 100 步一行，学习率是 01 的 10 倍 |
| `Speed ... img/s, ... min/epoch` | §7 | 之前是 888 img/s、每个 epoch 8.1 分钟 |
| `== epoch 1/2/3: ...` | §7 | holdout top-1 之前是 62.94%、70.12%、71.97% |
| `Training done: 5037 steps, ...` | §9.3 | 之前是 0.54 GPU-h |
| `Peak VRAM ...` | §9.3 | 之前是 19.9 GB allocated、21.3 GB reserved |
| `Final holdout ...` | §9.5 | 合并以后的模型，之前是 top-1 71.98%、top-5 89.07% |
| `Inference latency ...` | §8 | 合并以后的模型，之前是 7.61 ms |
| `== ... finished ...` | §9.5 | 一行总结 |

---

## 附录 B：LoRA 相关术语

基础术语见 07 的附录 B，训练相关的见 01 的附录 B。

| 术语 | 说人话 |
|---|---|
| LoRA（Low-Rank Adaptation） | 冻结原模型，在线性层旁边加一条低秩的旁路，只训练旁路 |
| 秩（rank，r） | 矩阵里“真正独立”的方向有几个。B·A 的秩最多是 r，r 越小，能表示的改动越简单，参数也越少 |
| 低秩 | 秩远小于矩阵的尺寸。这里 2304 × 768 的改动只用秩 8 来表示 |
| A、B 矩阵 | 旁路里的两个小矩阵：A 把输入压到 r 维（随机初始化），B 再放大回去（初始化为 0） |
| α 和缩放系数 | 旁路的输出乘以 α/r（这里是 2），用来控制 LoRA 改动的大小 |
| 合并（merge） | 训练完把 `(α/r)·B·A` 加进原权重 W，旁路就可以去掉，推理没有额外开销 |
| peft | Hugging Face 的参数高效微调库，提供 LoRA 等方法 |
| `LoraConfig` | 描述要加什么样的 LoRA：秩、α、dropout、加在哪些层、偏置要不要训练 |
| `get_peft_model` | 按 `LoraConfig` 把 LoRA 插进模型，并冻结其他参数 |
| `PeftModel`、`get_base_model()` | peft 包在模型外面的一层；`get_base_model()` 取出里面原来的模型 |
| `merge_and_unload()` | 合并 LoRA，并去掉 peft 的包装，返回普通的模型 |
| `target_modules` | 哪些层加 LoRA。传一个字符串时，peft 把它当正则表达式，要求和层的完整名字匹配 |
| 正则表达式（regex） | 描述字符串格式的规则。`\.` 是点号，`\d+` 是一串数字，竖线符号表示“或”，`^` 和 `$` 是开头和结尾，括号把几项归成一组 |
| adapter 名字 `default` | peft 允许给一个模型挂多组 LoRA，每组有个名字，只有一组时默认叫 `default`，所以参数名里会出现它 |

---

## 附录 C：讲解时可能被问到的问题

1. **LoRA 为什么只用这么少的参数就能有效？** LoRA 论文的假设是：微调时权重的改动“本质上是低维的”，不需要满秩的矩阵来表示。我们的结果也支持这一点：只训练主干 1.375% 的参数，holdout 71.98%，比全量微调（73.38%）只低约 1.4 个点。
2. **为什么 B 要初始化为 0？** 这样刚开始旁路输出 0，模型和预训练模型一模一样，起点就是类原型分类头的 zero-shot 水平（约 44%），训练从一个稳定的状态开始。A 如果也是 0，梯度就传不过来，所以 A 用随机值。
3. **为什么学习率是全量微调的 10 倍（1e-3）？** 主干冻结，大学习率也冲不坏预训练特征；LoRA 参数从 0 开始、数量又少，小学习率学得太慢。
4. **为什么四种层都加，而不是像参考设置那样只加 qkv？** 我们试过（都用 lp 分类头、3 个 epoch）：只加 qkv 是 68.57%；把学习率减半是 68.45%，几乎没变；四种层都加是 69.45%。说明 LoRA 的容量比学习率更重要。四种都加以后参数是 1.375%，仍然远低于 5%。
5. **LoRA 为什么没比全量微调省显存、省时间？** 实际数字：峰值显存 19.9 GB 对 18.9 GB，训练速度 888 img/s 对 1,146 img/s，LoRA 反而略差。下面是我们的推测：
   - 训练显存的大头是激活值（每层的中间结果，见 01 附录 C 第 6 条）。LoRA 照样要让图片过完整的 12 层，而且每个 `lora_A` 都要留着它的输入给反向传播用，激活值一点没少。
   - LoRA 真正省下的，只是冻结权重的梯度和 AdamW 状态：85,798,656 个参数 × 12 字节 ≈ 1.03 GB。
   - 48 条旁路，每条要做两次很小的矩阵乘法，还有 peft 每层的额外开销。这些计算量小，但在 GPU 上效率低，可能把省下的抵消了。
   - 对照：同样冻结主干的 AdaptFormer 训练速度是 1,437 img/s，比全量微调还快，说明冻结主干本身是能省时间的；LoRA 慢，更可能是实现上的开销。
   - LoRA 真正的好处是：可训练参数少（1.375%）、权重文件小（35 MB 对 374 MB）、合并以后推理没有额外开销。
6. **推理时有没有额外开销？** 没有。训练完把 LoRA 合并回原权重，模型结构和原来一样，额外参数是 0。推理延迟 7.61 ms，和全量微调的 7.78 ms 差不多。
7. **为什么只训练 3 个 epoch？** 训练 6 个 epoch 反而变差：71.52% 对 71.98%。LoRA 大约 3 个 epoch 就到头了，再训练就开始过拟合。全量微调则是训练越久越好。
8. **和项目文档里博士生的 LoRA 结果（68.3%）比怎么样？** 他们只在 qkv 上加 LoRA（0.34%）。我们用同样的设置复现出 68.57%。最终版本四种层都加、改用类原型分类头，holdout 71.98%，CodaBench val 71.90%，可训练参数 1.375%，仍低于 5%。
9. **课程里哪里讲了 LoRA？** Session 5 Lesson 5 的 PDF 第 28–30 页（第 33、35 页也提到），以及 CS701 第 10 周的 tutorial 第 15–18 页。
