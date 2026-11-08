# 05_adapter 逐行讲解

对应文件：`methods/05_adapter.ipynb`（精简版）。

这个 notebook 和 01（全量微调）的大部分代码完全一样：读数据、数据集、提特征和类原型、训练循环、画图、测延迟。这些地方照样贴出代码，只简单说明，细节见 `01_full_walkthrough.md`（写作“见 01 §x”）或 `07_linear_walkthrough.md`（写作“见 07 §x”）。Adapter 自己的代码逐行讲：§1 的两个新设置、§4 的 `Adapter` 类和新的前向函数、`apply_method`、§7 的参数分组和保存、§9 里 `.to(DEVICE)` 和“推理时多出来的参数”。末尾有日志对照、新术语和讲解时可能被问到的问题。

## 0. 先看全局

### Adapter 是什么

- 主干（ViT-B/16 的 85,798,656 个参数）全部冻结，不训练。
- 在 12 个 block 里各插一个很小的“瓶颈”模块，叫 Adapter：先把 768 维压到 64 维，过一个 ReLU，再升回 768 维。训练时只更新这些 Adapter 和分类头。
- Adapter **串联**在每个 block 的 MLP 后面：MLP 的输出 h 先经过 Adapter，变成 `h + Adapter(h)`，再加回残差。

一个 block 原来是这样：

```
x1  = x  + Attention(LN(x))       ← 注意力那一半
out = x1 + MLP(LN(x1))            ← MLP 那一半
```

加了 Adapter 以后：

```
x1  = x  + Attention(LN(x))       ← 不变
h   = MLP(LN(x1))                 ← 不变
out = x1 + ( h + Adapter(h) )     ← 新增：MLP 的输出先过 Adapter

Adapter(h) = up( Dropout( ReLU( down(h) ) ) ) × ADAPTER_SCALE
              down: 768 → 64      up: 64 → 768
```

画成箭头：

```
x1 ─► LN ─► MLP ─► h ─┬──────────────────────►(+)─► h + Adapter(h) ─► 加回 x1 ─► out
                      └─► down ─► ReLU ─► up ───┘
```

### 和 AdaptFormer 的区别

两者的 Adapter 模块和参数量完全一样，区别只有两处：

| | Adapter（这个 notebook） | AdaptFormer |
|---|---|---|
| 接法 | 串联：接在 MLP 后面，输入是 MLP 的输出 h | 并联：和 MLP 并排，输入是 MLP 之前的 x1 |
| 一个 block 的输出 | `x1 + h + Adapter(h)` | `x1 + MLP(LN(x1)) + Adapter(x1)` |
| `ADAPTER_SCALE` | 1.0，不额外缩放 | 0.1 |

### 参数量

| 部分 | 计算 | 参数个数 |
|---|---|---|
| down（768 → 64） | 768 × 64 + 64 | 49,216 |
| up（64 → 768） | 64 × 768 + 768 | 49,920 |
| 一个 Adapter | | 99,136 |
| 12 个 Adapter | 99,136 × 12 | 1,189,632，占主干的 1.387% |
| 分类头 | 768 × 10,000 + 10,000 | 7,690,000 |
| 训练的参数一共 | | 8,879,632 |

### 和 01 比

| | 01_full | 05_adapter |
|---|---|---|
| 主干 | 全部训练 | 冻结 |
| 新增模块 | 没有 | 每个 block 一个 Adapter（768 → 64 → 768） |
| 可训练参数（不含分类头） | 85,798,656（100%） | 1,189,632（1.387%） |
| 分类头 | 类原型初始化，参与训练 | 一样 |
| 学习率 | 1e-4，逐层衰减 | 1e-3，所有可训练参数都一样 |
| 权重衰减 | 0.05 | 1e-4 |
| drop path | 0.1 | 没有 |
| 训练循环（§7 `train`） | | 代码完全一样 |
| 推理 | 结构不变 | 每个 block 多一个 Adapter，没法合并回去，推理会慢一些 |
| `final.pt` | 全部参数，约 374 MB | Adapter 加分类头，约 36 MB |

### 一次实验的流程（§9 `run_experiment`）

```
建模型（create_model，不带 drop path）
  → 切分数据，准备训练和评估两套预处理（make_ctx）
  → 用类原型初始化分类头（init_head）
  → 冻结主干，在 12 个 block 里各插一个 Adapter，分类头设为可训练（apply_method），再整体搬到 GPU
  → 训练 3 个 epoch，只更新 Adapter 和分类头，每个 epoch 结束在 holdout 上测一次（train）
  → 保存 Adapter 和分类头，统计推理时多出来的参数
  → 最后在 holdout 上评估，画训练曲线，测推理延迟
  → 写 summary.json
```

### 训练的几个关键数字

和 01 用同一套对比设置，所以步数一样：

- 每一步 256 张图，每个 epoch 1,679 步，3 个 epoch 一共 5,037 步，前 503 步是 warmup。
- 学习率（所有可训练参数都一样）：第 0 步 2.0e-6，第 502 步升到 1e-3；第 1,679 步（第 1 个 epoch 结束）8.4e-4，第 2,770 步 5.0e-4，第 3,358 步（第 2 个 epoch 结束）3.0e-4，最后降到接近 0。正好是 01 的 10 倍。

### 参考结果

- Adapter 还没跑过，下面的数字只用来对照：
  - 同样设置的全量微调 `1019_1120_1_full`：holdout 73.38%；LoRA `1019_1319_3_lora`：holdout 71.98%。
  - AdaptFormer 之前用 linear probe 的分类头跑过一次（`1018_1239_3_adaptformer`，不是这里的类原型）：holdout 69.05%，0.39 GPU-h，峰值显存 12.8 GB，训练速度 1,437 张/秒，推理延迟 9.89 ms。
- Adapter 和 AdaptFormer 的参数量和计算量差不多，所以猜测训练速度、显存和推理延迟会和 AdaptFormer 接近；准确率要跑了才知道。

---

## 1. Config（§1）

### 1.1 要跑的实验

```python
# ===== Experiments to run: each dict is one run and lists only the settings that differ from the defaults below =====
EXPERIMENTS = [
    {},  # the defaults; not run yet
]
```

- 和 01 一样（见 01 §1.1）：`{}` 表示全部用下面的默认值，也就是 7 个 notebook 统一的对比设置。注释说明这个设置还没跑过。

### 1.2 默认值

```python
# ===== Defaults: the comparison protocol shared by all method notebooks (prototype head, 3 epochs, 2 holdout images per class) =====
METHOD = "adapter"
MODEL_NAME = "vit_base_patch16_224.augreg2_in21k_ft_in1k"  # timm name of the base model in the doc (IN21K→IN1K)
HOLDOUT_PER_CLASS = 2    # images per class held out from train for local evaluation (the official val set has no labels)
EPOCHS = 3
WARMUP_EPOCHS = 0.3      # the LR rises linearly for this many epochs, then follows a cosine down to 0
BATCH_SIZE = None        # images per step; None = pick based on VRAM (256 on a 40 or 80 GB GPU)
EFFECTIVE_BATCH = 256    # if one step can't fit it, gradients are accumulated over several steps to reach it
LR = 1e-3
WEIGHT_DECAY = 1e-4
ADAPTER_DIM = 64         # bottleneck size of the adapter: 768 → 64 → 768
ADAPTER_SCALE = 1.0      # the adapter output is multiplied by this (1.0: no extra scaling)
LABEL_SMOOTHING = 0.1
RRC_SCALE = (0.25, 1.0)  # area range for RandomResizedCrop (training augmentation)
LOG_EVERY = 100          # write a log line every N optimizer steps
SEED = 0
NUM_CLASSES = 10000
```

| 变量 | 默认值 | 意思 | 和 01 比 |
|---|---|---|---|
| `METHOD` | `"adapter"` | 方法名，用在实验名、日志和 summary 里 | 不同 |
| `MODEL_NAME`、`HOLDOUT_PER_CLASS`、`EPOCHS`、`WARMUP_EPOCHS`、`BATCH_SIZE`、`EFFECTIVE_BATCH` | | 同一套对比设置，见 01 §1.2 | 一样 |
| `LR` | `1e-3` | 学习率，Adapter 和分类头都用它 | 01 是 1e-4 |
| `WEIGHT_DECAY` | `1e-4` | 权重衰减 | 01 是 0.05 |
| `ADAPTER_DIM` | `64` | Adapter 中间那一层的宽度：768 → 64 → 768 | 新增 |
| `ADAPTER_SCALE` | `1.0` | Adapter 的输出乘上这个数，1.0 就是不缩放 | 新增 |
| `LABEL_SMOOTHING`、`RRC_SCALE`、`LOG_EVERY`、`SEED`、`NUM_CLASSES` | | 见 01 §1.2 | 一样 |

没有 `LAYER_DECAY` 和 `DROP_PATH`：所有可训练参数用同一个学习率，也不做 drop path。

**`ADAPTER_DIM = 64` 意味着什么**：这是“瓶颈”的宽度。Adapter 先把 768 维压到 64 维，再升回来，所以参数很少（每个 block 不到 10 万个）。宽度越大，Adapter 能表达的变化越多，参数也越多：参数量大约和这个数成正比（`ADAPTER_DIM` × 768 × 2）。

**`ADAPTER_SCALE = 1.0`**：Adapter 的输出原样加回去。AdaptFormer 用 0.1，因为它和 MLP 并联，加的是一个额外的分支；这里的 Adapter 串在 MLP 后面，自带残差（`h + Adapter(h)`），不额外缩放。

**为什么学习率是 1e-3，比全量微调大 10 倍**：主干冻结了，不用担心把预训练特征冲坏；新加的 Adapter 从零开始学（up 层初始是 0），参数又少，用大一点的学习率才学得动。LoRA 和 AdaptFormer 也用 1e-3。

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

和 01、07 一样，见 07 §1.3。

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
            "LR", "WEIGHT_DECAY", "ADAPTER_DIM", "ADAPTER_SCALE", "LABEL_SMOOTHING", "RRC_SCALE", "LOG_EVERY",
            "SEED", "NUM_CLASSES")
BASE_CONFIG = {k: globals()[k] for k in RUN_KEYS}
```

机制见 07 §2.2。列表和 01 相比，去掉了 `LAYER_DECAY`、`DROP_PATH`，加上了 `ADAPTER_DIM`、`ADAPTER_SCALE`。所以想试别的瓶颈宽度，可以在 `EXPERIMENTS` 里加一行 `dict(ADAPTER_DIM=128)`。

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

大部分和 01 一样（见 01 §2.3）。区别：

- 多了 `types`：§4.6 用 `types.MethodType` 替换每个 block 的前向函数。
- 多了 `torch.nn`（`nn`）：§4.1 的 `Adapter` 类继承 `nn.Module`，里面用 `nn.Linear`、`nn.Dropout` 和 `nn.init`。
- 少了 `re`：01 只在逐层学习率衰减里用它从参数名读 block 编号，这里没有逐层衰减。

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

和 01、07 完全一样，逐行解释见 07 §2.4–§2.8。

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

和 01、07 完全一样，逐行解释见 07 §3：从 Hugging Face 下载 `train.txt` 和 `train.tar.gz`（`DATA_FROM_HF = False` 时用 Drive 上的备份），解压到 Colab 本地磁盘，检查图片数量，读出 450,000 个（路径, 类别）。

---

## 4. 模型和 Adapter（§4）：这个 notebook 的核心

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

`class Adapter(nn.Module)`：继承 PyTorch 的 `nn.Module`，这样它就是一个“层”：有自己的参数，能被训练、保存、搬到 GPU。

`__init__(self, dim, bottleneck, scale, dropout=0.1)` 的参数：

| 参数 | 传进来的值 | 意思 |
|---|---|---|
| `dim` | 768（`net.embed_dim`） | 输入和输出的维度，等于 ViT 每个 token 的维度 |
| `bottleneck` | 64（`ADAPTER_DIM`） | 中间压到多少维 |
| `scale` | 1.0（`ADAPTER_SCALE`） | 输出乘上的系数 |
| `dropout` | 0.1（默认值，没有从配置传） | 训练时随机丢掉中间结果的比例 |

逐行看：

- `super().__init__()`：先调用父类 `nn.Module` 的初始化。每个自定义的层都要写这一行，否则 PyTorch 没法登记它的参数。
- `self.down = nn.Linear(dim, bottleneck)`：降维层，768 → 64，参数 768 × 64 + 64 = 49,216 个。
- `self.up = nn.Linear(bottleneck, dim)`：升维层，64 → 768，参数 64 × 768 + 768 = 49,920 个。
- `self.drop = nn.Dropout(dropout)`：训练时随机把 10% 的数置 0（剩下的放大到 1/0.9 倍，保持平均值不变），一种正则化。评估时（`eval()` 模式）不起作用。
- `self.scale = scale`：存下缩放系数。它只是一个普通的数，不是参数，不训练。
- 后面四行是初始化：
  - `nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))`：down 的权重用 Kaiming 均匀分布随机初始化。`a=math.sqrt(5)` 让它和 PyTorch 线性层默认的初始化一样。函数名末尾的 `_` 表示原地修改。
  - `nn.init.zeros_(self.down.bias)`：down 的偏置设成 0。
  - `nn.init.zeros_(self.up.weight)` 和 `nn.init.zeros_(self.up.bias)`：**up 的权重和偏置全部设成 0**。这是关键：不管输入是什么，Adapter 一开始的输出都是 0，所以刚插进去时模型和原来一模一样（用类原型分类头时，就是 zero-shot 的水平，约 44%）。

`forward(self, x)`：Adapter 的前向计算，从里往外读：

1. `self.down(x)`：768 维压到 64 维。
2. `F.relu(...)`：ReLU，负数变 0，正数不变。有了它，Adapter 才能表示非线性的变化。
3. `self.drop(...)`：dropout，只在训练时起作用。
4. `self.up(...)`：64 维升回 768 维。
5. `* self.scale`：乘缩放系数，这里是 1.0。

输入和输出的形状一样，都是（batch, 197, 768）：对 197 个 token 各自独立地算。

up 初始化为 0，会不会让 Adapter 学不动？不会。up 的梯度取决于 down 和 ReLU 的输出，而 down 是随机初始化的，这个输出不为 0，所以 up 第一步就有梯度，马上开始变化。只有 down 在最开始那一刻没有梯度（因为它的梯度要乘上 up 的权重，而 up 是 0）；up 一变成非零，down 也开始学。我在随机初始化的 ViT-B 上试过：刚插入时，up 权重的梯度不为 0，down 权重的梯度正好是 0。

### 4.2 `adapter_block_forward`：新的 block 前向函数

```python
def adapter_block_forward(self, x, *args, **kwargs):
    # Serial Adapter: placed after the FFN output, with its own residual
    x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), *args, **kwargs)))
    h = self.mlp(self.norm2(x))
    return x + self.drop_path2(self.ls2(h + self.adapter(h)))
```

这个函数会替换 timm 里每个 Block 原来的 `forward`（怎么替换见 §4.6）。`self` 就是那个 block，所以能用它的各个子层。逐行看：

- `def adapter_block_forward(self, x, *args, **kwargs):`：`x` 是输入的 token，形状（batch, 197, 768）。`*args, **kwargs` 接收调用时可能多传的参数（比如注意力的 mask），原样转交给注意力层，这样和 timm 原来的调用方式兼容。
- 注释：串联的 Adapter，放在 FFN（也就是 MLP）的输出后面，带自己的残差。
- `x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), *args, **kwargs)))`：注意力那一半，和 timm 原来的写法完全一样：
  - `self.norm1(x)`：LayerNorm。
  - `self.attn(...)`：多头自注意力。
  - `self.ls1(...)`：layer scale，给输出乘一组可学习的系数。这个模型没有用 layer scale，它是什么都不做的 `Identity`。
  - `self.drop_path1(...)`：drop path。这里 drop path 的概率是 0，也是 `Identity`。
  - `x + ...`：残差连接。
- `h = self.mlp(self.norm2(x))`：MLP 那一半的分支部分：先 LayerNorm，再过 MLP（768 → 3072 → 768）。到这里和原来一样，只是先把结果存进 `h`，还没加回去。
- `return x + self.drop_path2(self.ls2(h + self.adapter(h)))`：**新加的部分**：
  - `self.adapter(h)`：MLP 的输出 h 送进 Adapter。
  - `h + self.adapter(h)`：Adapter 自带的残差：在 MLP 的输出上加一个小的修正。
  - 再经过 `ls2` 和 `drop_path2`（这里都是 `Identity`），加回残差 `x`。
- 原来的 timm Block 最后一行相当于 `x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))`，这里只是把 MLP 的输出换成了 `h + Adapter(h)`。Adapter 输出为 0 时，两者完全一样。

和原论文的关系：Adapter 最早的论文（Houlsby 等，2019）在每个 block 里放两个 Adapter（注意力后面一个，MLP 后面一个），还会训练 LayerNorm 的参数。这里只在 MLP 后面放一个、LayerNorm 也冻结，这是一种常见的简化做法（例如 Pfeiffer 等人 2020 年的设置只在 FFN 后面放一个）。

### 4.3 `is_head`

```python
def is_head(name):
    return name in ("head.weight", "head.bias")
```

和 01、07 一样（见 07 §4.1）：判断一个参数是不是分类头的。Adapter 的参数名是 `blocks.0.adapter.down.weight` 这样，不会被当成分类头。

### 4.4 `resolve_config`

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

和 01 完全一样，见 01 §4.2：检查设置名，合并默认值和实验的修改，按显存选 `BATCH_SIZE`，算梯度累积步数 `ACCUM_STEPS`（A100 上是 1）。

### 4.5 `create_model`

```python
def create_model(cfg):
    """Pretrained ViT-B/16 from timm; its 1000-class ImageNet head is replaced by a new, randomly initialized 10,000-class head"""
    return timm.create_model(cfg.MODEL_NAME, pretrained=True, num_classes=cfg.NUM_CLASSES).to(DEVICE)
```

和 07 一样（见 07 §4.3），比 01 少了 `drop_path_rate`：不做 drop path。各个 PEFT 方法的默认设置里 drop path 都是 0，只有全量微调用 0.1。一个可能的原因：主干是冻结的，随机跳过某些 block 会让 Adapter 看到的输入忽大忽小；而且可训练参数本来就很少，不太需要这种正则化。

### 4.6 `apply_method`：冻结主干、插入 Adapter

```python
def apply_method(net, cfg):
    """Adapter: freeze the backbone and add a bottleneck adapter after the MLP of every block (in series)"""
    net.requires_grad_(False)
    for blk in net.blocks:
        blk.adapter = Adapter(net.embed_dim, cfg.ADAPTER_DIM, cfg.ADAPTER_SCALE)
        blk.forward = types.MethodType(adapter_block_forward, blk)
    net.head.requires_grad_(True)
    return net
```

这是 Adapter 方法真正“改模型”的地方，逐行看：

- `net.requires_grad_(False)`：先把所有参数（主干和分类头）都设成不训练。
- `for blk in net.blocks:`：遍历 12 个 block。
- `blk.adapter = Adapter(net.embed_dim, cfg.ADAPTER_DIM, cfg.ADAPTER_SCALE)`：给这个 block 新建一个 Adapter（768 → 64 → 768，缩放 1.0），挂在 `blk.adapter` 上。
  - 给 `nn.Module` 的属性赋值一个 `nn.Module` 时，PyTorch 会自动把它登记成子模块。所以它的参数会出现在 `model.parameters()` 里（优化器能找到它），也会跟着 `.to()`、`.train()`、`.eval()` 和保存一起走。
  - 新建的层默认是需要梯度的，所以 Adapter 的参数会被训练，虽然上一行把其他参数都冻结了。
  - 参数名会是 `blocks.0.adapter.down.weight`、`blocks.0.adapter.up.bias` 这样。
- `blk.forward = types.MethodType(adapter_block_forward, blk)`：把这个 block 的前向函数换成 §4.2 的 `adapter_block_forward`。
  - `types.MethodType(函数, 对象)`：把一个普通函数“绑定”到某个对象上，变成它的方法。之后调用 `blk.forward(x)` 时，`blk` 会自动作为第一个参数 `self` 传进去。
  - 这样只改了这一个 block 对象，不影响 timm 的 Block 类本身。这种在运行时替换方法的做法叫 monkey patching。
  - ViT 在 `forward_features` 里依次调用 12 个 block，调用的就是这里换上的新前向函数。
- `net.head.requires_grad_(True)`：新的分类头设成可训练。
- `return net`。

两个要注意的地方：

- Adapter 是在 CPU 上新建的（`nn.Linear` 默认建在 CPU），而模型的其他部分已经在 GPU 上了。所以 §9 调用完 `apply_method` 后，还要 `.to(DEVICE)` 把整个模型（主要是这些新 Adapter）搬到 GPU，否则第一次前向就会报错说张量不在同一个设备上。
- `apply_method` 在 `init_head` 之后调用。提特征、算原型用的是还没插 Adapter 的原始主干；不过 Adapter 一开始输出为 0，插不插其实结果一样。

### 4.7 `param_report`

```python
def param_report(net, ctx):
    """Trainable parameter counts. The 7.69M head is counted separately, as in the doc's LoRA reference (0.34%)"""
    n_train = sum(p.numel() for n, p in net.named_parameters() if p.requires_grad and not is_head(n))
    return {"backbone_params": ctx.n_backbone, "head_params": ctx.n_head, "head_mode": "fine-tuned",
            "trainable_excl_head": n_train, "trainable_pct": 100 * n_train / ctx.n_backbone,
            "trainable_incl_head": n_train + ctx.n_head}
```

代码和 01 一样（见 01 §4.5），这里统计出来是：

| key | 值 | 意思 |
|---|---|---|
| `backbone_params` | 85,798,656 | 原来主干的参数量（`make_ctx` 在插 Adapter 之前统计的，所以不含 Adapter） |
| `head_params` | 7,690,000 | 分类头的参数量 |
| `head_mode` | `fine-tuned` | 分类头参与训练 |
| `trainable_excl_head` | 1,189,632 | 可训练参数，不含分类头，也就是 12 个 Adapter |
| `trainable_pct` | 1.387 | 占原主干的百分比，低于 PDF 建议的 5% |
| `trainable_incl_head` | 8,879,632 | 可训练参数，含分类头 |

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

和 01、07 完全一样，见 07 §5.1–§5.3。

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

和 01 完全一样，见 01 §5.2：训练用随机裁剪加翻转，评估用缩放加中心裁剪。注意 `n_backbone` 是在这里统计的，这时候还没插 Adapter，所以是原主干的 85,798,656。

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

和 01 完全一样，见 01 §5.3–§5.4：训练每批 256 张、打乱、丢掉最后不满的一批；评估每批 512 张、按顺序。

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

和 01、07 完全一样，见 07 §5.7–§5.9。

---

## 6. 分类头初始化（§6）

### 6.1 `Clock`

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

和 01 完全一样，见 01 §6.1：把设置分类头、每个 epoch 的训练和 holdout 评估的时间加起来，作为 GPU 时间。

### 6.2 `extract_features`、`class_prototypes`、`fit_logit_scale`

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

和 01 完全一样，见 01 §6.2–§6.3 和 07 §6.1–§6.3：用还没插 Adapter 的预训练主干给 430,000 张训练图提特征，算每一类的原型，挑一个温度。

### 6.3 `init_head`

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

和 01 完全一样，见 01 §6.4：用类原型初始化分类头，返回这一步的秒数（含提特征，约 6–7 分钟）。

对 Adapter 来说，类原型的起点也很重要：Adapter 一开始输出为 0，模型就是“预训练主干 + 原型分类头”，也就是 zero-shot 的水平。训练从这个起点出发，只需要学一些小的修正。我们在 LoRA 上验证过：同样训练 3 个 epoch，类原型起步比 linear probe 起步高约 2.5 个点（72.00% 对 69.45%）。Adapter 还没验证过，但道理是一样的。

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

01 有逐层学习率衰减，分了 28 组；这里所有可训练参数用同一个学习率，只按“要不要权重衰减”分成两组。逐行看：

- `decay, no_decay = [], []`：两个空列表，一个放要做权重衰减的参数，一个放不做的。
- `for p in net.parameters():`：遍历模型的所有参数（包括 §4.6 新插进去的 Adapter）。
- `if p.requires_grad:`：只管要训练的参数，冻结的主干参数跳过。
- `(no_decay if p.ndim <= 1 else decay).append(p)`：一维的参数（偏置）放进 `no_decay`，其余（权重矩阵）放进 `decay`。括号里的条件表达式先选出一个列表，再往里面加。
- `return [...]`：每组变成优化器认识的格式：学习率都是 `cfg.LR`（1e-3），`decay` 组的权重衰减是 `cfg.WEIGHT_DECAY`（1e-4），`no_decay` 组是 0。`"base_lr"` 留着原始学习率，训练时每一步乘上 warmup / 余弦的系数，见 01 §7.4。
- 最后的 `if ps`：某一组是空的就不要它（比如某个方法没有一维的可训练参数）。

实际分出来的两组：

| 组 | 包括 | 张量个数 | 参数个数 | 权重衰减 |
|---|---|---|---|---|
| `decay` | 12 个 Adapter 的 down、up 权重（24 个），分类头权重 | 25 | 8,859,648 | 1e-4 |
| `no_decay` | 12 个 Adapter 的 down、up 偏置（24 个），分类头偏置 | 25 | 19,984 | 0 |

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

和 01、07 完全一样，见 07 §7.1。里面的 `net.eval()` 会把 Adapter 里的 dropout 关掉，评估时 Adapter 的输出是确定的。

### 7.3 `save_state`

```python
def save_state(net, path, **extra):
    """Save only the trained params: the adapters and the head, about 36 MB"""
    state = {n: p.detach().cpu() for n, p in net.named_parameters() if p.requires_grad or is_head(n)}
    torch.save({"state": state, **extra}, path)
```

代码和 01、07 一样（见 07 §7.2），保存“需要梯度或者属于分类头”的参数。在这里就是 12 个 Adapter 和分类头，一共 8,879,632 个 float32，约 36 MB。冻结的主干没有变，不用存：以后要用这个模型，先加载预训练的 ViT-B/16，按 §4.6 插好 Adapter，再把这个文件里的参数读进去就行。

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

和 01 的代码完全一样，逐行解释见 01 §7.4。放在 Adapter 上，有几处效果不同：

- **只有 Adapter 和分类头会更新**。`params` 和优化器里只有这些参数，主干的参数没有梯度，也就不会被 AdamW 更新。
- **反向传播还是要穿过整个主干**。block 0 里的 Adapter 离输出最远，它的梯度要从损失一路往回传过 12 个 block 才能算出来。不过冻结的层只需要算“往回传的梯度”，不用算自己权重的梯度，计算量比全量微调少一些。
- **显存也少一些**。不用存主干 8,580 万个参数的梯度和 AdamW 的两份状态，这部分就省了约 1 GB。实测 AdaptFormer（同样的 Adapter 模块）的峰值显存是 12.8 GB，比全量微调的 18.9 GB 少得多，说明反向传播时要保存的中间结果也少了，这一点是推测。
- **`model.train()` 会打开 Adapter 里的 dropout**；每个 epoch 末的 `evaluate` 又把它关掉。
- **学习率是 01 的 10 倍**：warmup 结束时是 1e-3，见 §0。
- 步数一样：每个 epoch 1,679 步，一共 5,037 步，warmup 503 步。

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

和 01 完全一样，见 01 §8.1。

### 8.2 `plot_run_curves`

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

和 01 完全一样，见 01 §8.2：左边是训练损失，右边是每个 epoch 末的 holdout top-1，存成 `curves.png`。

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

代码和 01、07 一样，见 07 §7.3–§7.4。但结果会不同：全量微调和 LoRA（合并以后）的模型结构和原来的 ViT-B/16 一样，Adapter 不一样，每个 block 多了一个 Adapter，没法合并回去，推理时要多算这 12 个小模块。

为什么没法合并：Adapter 中间有一个 ReLU，是非线性的，没法写成一个矩阵，也就没法加到 MLP 原来的权重里。LoRA 没有非线性，`W + B·A` 还是一个矩阵，所以能合并。

参考：AdaptFormer 用的是一样的 Adapter 模块，之前测出来 batch 1 的推理延迟是 9.89 ms，同一批里合并后的 LoRA 是 7.03 ms，慢了约 40%。Adapter 还没测过，估计差不多。

---

## 9. 一次实验的完整流程（§9 `run_experiment`）

整体和 01 §9 一样，区别在三个地方：配置摘要那行日志、`.to(DEVICE)` 真正起了作用、多统计了“推理时多出来的参数”。分五段看。

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

和 01 §9.1 一样：合成配置，建实验目录，日志开始写进 `train_log.txt`，存 `config.json`，设随机种子，清理显存。第二行 `log` 记录的是 Adapter 的设置，比如 `adapter: lr 0.001, wd 0.0001, adapter dim 64, scale 1.0, epochs 3, holdout 2/class, prototype head`。随机种子也决定 Adapter 里 down 层的随机初始值。

### 9.2 建模型、初始化分类头、插入 Adapter

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

- `model = create_model(cfg)`：加载预训练的 ViT-B/16，换上新的分类头，不带 drop path。
- `ctx = make_ctx(model, cfg)`：切分数据，准备预处理，统计原主干和分类头的参数量。
- `clock = Clock()`、`clock.used += init_head(model, ctx, cfg)`：用类原型初始化分类头，这一步的时间算进 GPU 时间。
- `model = apply_method(model, cfg).to(DEVICE)`：冻结主干，插入 12 个 Adapter，分类头设为可训练，然后 `.to(DEVICE)`。在 01 里 `.to(DEVICE)` 什么也不改；在这里它把新建在 CPU 上的 12 个 Adapter 搬到 GPU。少了它，第一次前向就会报错，说有的张量在 CPU、有的在 GPU。
- `params = param_report(model, ctx)` 和下一行：统计并记录参数量，日志里是 `Trainable (excl. head): 1,189,632 = 1.387% of backbone | head 7,690,000 params (fine-tuned)`。

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

和 01 §9.3 一样：训练 3 个 epoch，保存参数（这里是 Adapter 加分类头，约 36 MB），存训练记录，记下 GPU 时间和峰值显存。

### 9.4 推理时多出来的参数

```python
        # Params the model has at inference on top of the original ViT-B/16 + head
        params["extra_params_at_inference"] = sum(p.numel() for p in model.parameters()) - ctx.n_backbone - ctx.n_head
```

01 没有这两行，会往模型里加模块的方法（Adapter、AdaptFormer、LoRA）才有。

- 注释：推理时，模型在原来的 ViT-B/16 加分类头之外，还多了多少参数。
- `sum(p.numel() for p in model.parameters())`：现在模型一共有多少参数，包括 12 个 Adapter：85,798,656 + 7,690,000 + 1,189,632 = 94,678,288。
- 减去原主干（`ctx.n_backbone`）和分类头（`ctx.n_head`），剩下 1,189,632，就是 12 个 Adapter。存进 `params`，最后写进 `summary.json`。
- 这个数说明 Adapter 推理时要额外带着这些参数、做这些计算。LoRA 合并以后这里是 0，这正是两者在推理上的区别（见 §8.3）。

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

和 01 §9.4 一样：在 holdout 上再测一次（多给出 top-5），画训练曲线，测推理延迟，写 `summary.json`。`**params` 会把上面的 `extra_params_at_inference` 一起带进 summary。

---

## 10. 运行实验（§10）

```python
RESULTS = []
for i, exp in enumerate(EXPERIMENTS, 1):
    RESULTS.append(run_experiment(dict(exp), f"{BATCH_NAME}_{i}_{METHOD}"))
```

和 01 §10 一样。实验名比如 `1102_1530_1_adapter`。

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

和 01 §11 一样，见 01 §11。表里的可训练参数这一列会显示 `1,189,632 (1.387%)`。

---

## 12. 自动断开（§12）

```python
if DISCONNECT_WHEN_DONE:
    from google.colab import runtime
    log("All done; releasing the runtime ...")
    drive.flush_and_unmount()  # make sure all results are written to Drive
    runtime.unassign()
```

和 01、07 一样，见 07 §11。

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

========== 1102_1530_1_adapter ==========
adapter: lr 0.001, wd 0.0001, adapter dim 64, scale 1.0, epochs 3, holdout 2/class, prototype head
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
== 1102_1530_1_adapter finished: holdout top-1 …% | trainable 1,189,632 (1.387%) | … GPU-h
```

最后还会打印 §11 的结果表。

| 日志 | 出自 | 说明，参考值 |
|---|---|---|
| `Batch ...`、`Device ...`、`Versions ...` | §2 | 批次名、运行环境 |
| `[train] downloaded ...`、`extracting ...`、`images ready` | §3 | 每个会话只有第一次运行才有。从 Hugging Face 下载一两分钟；用 Drive 备份时没有 `downloaded` 这一行，直接从 Drive 解压，要 20 分钟左右 |
| `========== ..._1_adapter ==========` | §9 | 实验开始，从这行起也写进 `train_log.txt` |
| `adapter: lr 0.001, ...` | §9 | Adapter 的设置摘要 |
| `Head initialized from ...` | §6 `init_head` | 大约 6–7 分钟，主要是提特征 |
| `Trainable (excl. head): 1,189,632 = 1.387% ...` | §9 | 只有 12 个 Adapter 可训练 |
| `Training 3 epochs x 1679 steps ...` | §7 `train` | 和 01 一样的步数 |
| `ep 1/3 step 100/5037 ...` | §7 | 每 100 步一行；学习率是 01 的 10 倍 |
| `Speed ... img/s, ... min/epoch` | §7 | 还没跑过；AdaptFormer 之前是 1,437 img/s |
| `== epoch N: ...` | §7 | 每个 epoch 一行：训练损失、holdout top-1 |
| `Training done: 5037 steps, ...` | §9 | 还没跑过；AdaptFormer 之前是 0.39 GPU-h |
| `Peak VRAM ...` | §9 | 还没跑过；AdaptFormer 之前是 12.8 GB |
| `Final holdout ...` | §9 | 还没跑过；对照：全量微调 73.38%，LoRA 71.98% |
| `Inference latency ...` | §8.3 | 还没跑过；AdaptFormer 之前是 9.89 ms |
| `== ... finished ...` | §9 | 一行总结 |

AdaptFormer 那些数来自 `1018_1239_3_adaptformer`，它用的是 linear probe 的分类头，不是这里的类原型，速度、显存、延迟可以参考，准确率不能直接比。

怎么看训练是否正常：和 01 一样（见 01 附录 A），损失整体往下走，每个 epoch 末的 holdout top-1 一次比一次高。另外可以这样理解起点：训练开始前 Adapter 输出为 0，模型的 holdout 准确率就是 zero-shot 的水平（约 44%）。日志里不会测这个数，但第 1 个 epoch 末的 holdout top-1 应该明显比它高。

---

## 附录 B：新术语

基础术语见 07 的附录 B，训练相关的见 01 的附录 B。下面是 Adapter 特有的：

| 术语 | 说人话 |
|---|---|
| Adapter | 插在预训练模型里的小模块，只训练它，主干不动 |
| 瓶颈（bottleneck） | 先降维再升维的结构：768 → 64 → 768，中间窄，所以参数少 |
| 串联 / 并联 | 串联：Adapter 接在 MLP 后面，处理 MLP 的输出（这个 notebook）。并联：Adapter 和 MLP 处理同一个输入，结果相加（AdaptFormer） |
| 残差连接 | 输出 = 输入 + 变换(输入)。Adapter 自带一个：`h + Adapter(h)` |
| 零初始化 | up 层的权重和偏置设成 0，刚插入时 Adapter 输出为 0，模型和原来一模一样 |
| ReLU | 激活函数：负数变 0，正数不变。让 Adapter 能表示非线性的变化，也因此没法合并回原来的线性层 |
| Dropout | 训练时随机把一部分数（这里 10%）置 0，一种正则化；评估时不起作用 |
| Kaiming 初始化 | 一种常用的随机初始化方法，按输入维度决定随机数的范围 |
| 子模块注册 | 给 `nn.Module` 的属性赋值一个 `nn.Module` 时，PyTorch 自动把它登记成子模块：参数会出现在 `parameters()` 里，也会跟着 `.to()`、`.train()`、`.eval()` 走 |
| `types.MethodType(函数, 对象)` | 把一个普通函数绑定到某个对象上，变成它的方法，调用时这个对象自动作为 `self` 传进去 |
| monkey patching | 程序运行时直接替换某个对象的方法或属性，这里用来替换每个 block 的 `forward` |
| layer scale（`ls1`、`ls2`） | 有些 ViT 会给每个分支的输出乘一组可学习的系数；这个模型没用，是 `Identity` |
| 合并（merge） | 把训练出来的增量直接加到原来的权重里，推理时不多算。LoRA 能合并，Adapter 不能 |

---

## 附录 C：讲解时可能被问到的问题

1. **Adapter 和 AdaptFormer 有什么区别？** 模块和参数量完全一样（每个 block 99,136 个，一共 1,189,632 个）。区别在接法：Adapter 串在 MLP 后面，处理 MLP 的输出；AdaptFormer 和 MLP 并联，处理 MLP 之前的输入，输出还乘 0.1。
2. **up 层初始化为 0，会不会学不动？** 不会。up 的梯度取决于 down 和 ReLU 的输出，down 是随机初始化的，所以 up 第一步就有梯度。up 一变成非零，down 也开始有梯度。零初始化的好处是：训练从原模型（加类原型分类头）出发，一开始就有 zero-shot 的水平。
3. **Adapter 能像 LoRA 一样合并回主干吗？推理会慢多少？** 不能，中间的 ReLU 是非线性的。推理时 12 个 Adapter 都要算一遍。AdaptFormer 用的是同样的模块，之前测出 9.89 ms，比同一批里合并后的 LoRA（7.03 ms）慢约 40%。Adapter 还没测，估计差不多。
4. **1.387% 是怎么算出来的？** 一个 Adapter：down 768 × 64 + 64 = 49,216，up 64 × 768 + 768 = 49,920，合计 99,136。12 个 block 一共 1,189,632，除以主干的 85,798,656，是 1.387%。分类头的 7,690,000 个参数单独算，和 PDF 里 LoRA 参考成绩的算法一致。
5. **原论文不是每个 block 放两个 Adapter 吗？** 是的，Houlsby 等人（2019）在注意力和 MLP 后面各放一个，还训练 LayerNorm。这里只在 MLP 后面放一个、LayerNorm 也冻结，是一种常见的简化（例如 Pfeiffer 等人 2020 年的设置只在 FFN 后面放一个），参数也少一半。
6. **为什么学习率是 1e-3，比全量微调大 10 倍？** 主干冻结了，不用担心冲坏预训练特征；Adapter 是新加的、从零开始学，参数又少，需要大一点的步子。LoRA 和 AdaptFormer 也是 1e-3。
7. **为什么显存比全量微调少？** 不用为主干的 8,580 万个参数存梯度和 AdamW 状态（约 1 GB），冻结的层在反向传播时也不用算权重的梯度。AdaptFormer 实测 12.8 GB，全量微调 18.9 GB。
8. **课上讲过 Adapter 吗？** Session 5 Lesson 5 的 PDF 第 31 页在比较各种 PEFT 方法时提到过。CS701 Week 10 的 tutorial（`Session8/W10_lora_tutorial.pdf`）第 9–11 页有详细介绍。
