# 07_linear 逐行讲解

对应文件：`methods/07_linear.ipynb`（精简版）。按 notebook 的顺序，把每个代码格子拆成小段：先贴代码，再逐行解释；函数的参数单独列表说明。末尾有日志对照、术语表和讲解时可能被问到的问题。

## 0. 先看全局

### 这个 notebook 做什么

跑两个实验，主干（ViT-B/16）都不训练：

1. **zero-shot**（`HEAD_INIT="proto"`）：用预训练模型给每张训练图片算一个 768 维的特征，同一类的特征取平均，当作这一类的“原型”，直接拿原型当分类头。不训练任何参数。
2. **linear probe**（`HEAD_INIT="lp"`）：从原型出发，在算好的特征上用梯度下降把分类头训练 10 个 epoch。主干还是不动。

两个实验最后都在 holdout（每类 2 张，共 20,000 张）上测准确率，再测推理延迟，结果写进 Google Drive。

### 一次实验的流程（§8 `run_experiment`）

```
建模型（create_model）
  → 切分训练部分和 holdout，准备图片预处理（make_ctx）
  → 设置分类头（init_head）：提特征 → 算原型 → 挑温度 →（lp）训练分类头
  → 冻结全部参数（apply_method）
  → 统计参数量（param_report），保存分类头（save_state）
  → 在 holdout 上评估（evaluate）
  → 测推理延迟（latency_report）
  → 写 summary.json
```

### 跑完会得到什么

都在 Google Drive 的 `MyDrive/cs604/` 下：

| 位置 | 内容 |
|---|---|
| `batches/<批次名>/batch_log.txt` | 这次运行的全部日志 |
| `batches/<批次名>/results.md`、`results.json` | 两个实验的结果表 |
| `runs/<批次名>_1_linear_proto/` | zero-shot：`train_log.txt`、`config.json`、`summary.json`、`final.pt` |
| `runs/<批次名>_2_linear_lp/` | linear probe：同上 |

批次名是开始运行的时间（新加坡时间），比如 11 月 2 日 15:30 开始，批次名就是 `1102_1530`。

### 张量形状速查

| 变量 | 形状 | 类型 | 说明 |
|---|---|---|---|
| 一批图片 `x`（CPU） | (512, 3, 224, 224) | uint8，0–255 | 预处理后的图片 |
| `to_input(x, ctx)` | (512, 3, 224, 224) | float32，在 GPU 上 | 归一化到 [−1, 1] |
| `forward_features` 的输出 | (512, 197, 768) | | 196 个图块 + 1 个 CLS token，每个 768 维 |
| 一批特征 `f` | (512, 768) | 存成 float16 | CLS token 的输出 |
| 全部训练特征 `feats` | (430000, 768) | float16 | 约 0.66 GB |
| 全部标签 `labels` | (430000,) | int64 | 每张图的类别编号 |
| 原型 `protos` | (10000, 768) | float32 | 每类一个长度为 1 的向量 |
| `fit_logit_scale` 里的 `cos` | (10000, 10000) | float32 | 抽出的 10,000 个样本对 10,000 个类的余弦相似度 |
| 分类头 `weight`、`bias` | (10000, 768)、(10000,) | float32 | 共 7,690,000 个参数 |
| 一批分数（logits） | (512, 10000) | | 每张图对每个类的分数 |

### 几个贯穿全文的数字

- 训练集 450,000 张（10,000 类 × 45 张）。留出 holdout 后，训练部分 430,000 张（每类 43 张），holdout 20,000 张（每类 2 张）。
- 特征 768 维；类别 10,000 个；分类头参数 768 × 10,000 + 10,000 = 7,690,000 个。
- 主干参数 85,798,656 个。
- A100 上 `BATCH_SIZE` 是 256，提特征和评估时每批 512 张。

---

## 1. Config（§1）

### 1.1 要跑的实验

```python
# ===== Experiments to run: each dict is one run and lists only the settings that differ from the defaults below =====
EXPERIMENTS = [
    dict(HEAD_INIT="proto"),  # zero-shot; the earlier run 0928_0636_1_linear got holdout 43.95%
    dict(HEAD_INIT="lp"),     # linear probe, reusing the features of the first run; earlier lp runs got holdout 62.07%
]
```

- `EXPERIMENTS` 是一个列表，每个元素是一个 dict，代表一次实验。dict 里只写和下面默认值**不一样**的设置。
- `dict(HEAD_INIT="proto")` 等价于 `{"HEAD_INIT": "proto"}`：第一个实验用类原型，也就是 zero-shot。
- `dict(HEAD_INIT="lp")`：第二个实验用 linear probe。
- §9 会按顺序跑这两个实验。想多跑一个，比如 linear probe 训练 20 个 epoch，就加一行 `dict(HEAD_INIT="lp", LP_EPOCHS=20)`。只有 §2 的 `RUN_KEYS` 里列出的设置能这样改。

### 1.2 默认值

```python
# ===== Defaults =====
METHOD = "linear"
MODEL_NAME = "vit_base_patch16_224.augreg2_in21k_ft_in1k"  # timm name of the base model in the doc (IN21K→IN1K)
HOLDOUT_PER_CLASS = 2    # images per class held out from train for local evaluation (the official val set has no labels)
HEAD_INIT = "proto"      # 'proto': class prototypes (zero-shot) | 'lp': linear probe trained from the prototypes
BATCH_SIZE = None        # None = pick based on VRAM (256 on a 40 or 80 GB GPU); the backbone sees 2 × BATCH_SIZE images per batch
LP_EPOCHS = 10           # linear probe: passes over the cached training features
LP_LR = 1e-3             # linear probe: AdamW learning rate, cosine decay to 0 without warmup
LP_WD = 1e-4             # linear probe: AdamW weight decay
LP_BATCH = 4096          # linear probe: features per step (only a 768 → 10,000 linear layer is trained, so a big batch fits)
LABEL_SMOOTHING = 0.1    # linear probe: label smoothing of the cross-entropy loss
SEED = 0
NUM_CLASSES = 10000
```

| 变量 | 默认值 | 意思 | 用在哪里 |
|---|---|---|---|
| `METHOD` | `"linear"` | 方法名，只用在实验名、日志和 summary 里 | §8、§9 |
| `MODEL_NAME` | 见下面的拆解 | timm 里的模型名，决定用哪个模型、哪组预训练权重 | §4 `create_model`；也是特征缓存的 key 之一 |
| `HOLDOUT_PER_CLASS` | `2` | 每类留几张做 holdout | §5 `split_holdout` |
| `HEAD_INIT` | `"proto"` | 分类头怎么来：`proto` 用类原型，`lp` 用 linear probe | §6 `init_head` |
| `BATCH_SIZE` | `None` | `None` 表示按显存自动选，40 GB 以上是 256。主干每批处理 2 × BATCH_SIZE 张 | §4 `resolve_config`、§5 `make_loader` |
| `LP_EPOCHS` | `10` | linear probe 把全部训练特征过几遍 | §6 `linear_probe` |
| `LP_LR` | `1e-3`（0.001） | linear probe 的起始学习率，之后按余弦曲线降到 0 | §6 `linear_probe` |
| `LP_WD` | `1e-4`（0.0001） | linear probe 的权重衰减 | §6 `linear_probe` |
| `LP_BATCH` | `4096` | linear probe 每一步用多少个特征 | §6 `linear_probe` |
| `LABEL_SMOOTHING` | `0.1` | linear probe 损失函数的标签平滑 | §6 `linear_probe` |
| `SEED` | `0` | 随机种子：决定 holdout 怎么分，以及随机抽样、打乱的顺序 | §5、§6、§8 |
| `NUM_CLASSES` | `10000` | 类别数，也就是分类头输出多少个分数 | §4、§6 |

**`MODEL_NAME` 怎么看**

名字被点号分成两半：点号前面是**模型结构**，点号后面是**用哪一组训练好的权重**。

- 结构 `vit_base_patch16_224`，决定模型长什么样：
  - `vit`：Vision Transformer。
  - `base`：Base 规模：12 个 Transformer block，每个 token 768 维，12 个注意力头，MLP 中间层 3072 维。
  - `patch16`：把图片切成 16×16 像素的小块（patch），224×224 的图切成 14 × 14 = 196 块。
  - `224`：输入图片是 224×224。
  - 光有结构，里面约 8,600 万个参数都是随机数，什么也不会。
- 权重 `augreg2_in21k_ft_in1k`，说明这些参数是怎么训练出来的：
  1. `in21k`：先在 ImageNet-21k（约 1,400 万张图、约 2.1 万类）上预训练。
  2. `ft_in1k`：再在 ImageNet-1k（约 128 万张图、1,000 类）上微调，`ft` 是 fine-tune 的缩写。
  3. `augreg`：训练时用的方法，来自论文 *How to train your ViT?*（更强的数据增强和正则化）。`augreg2` 是 timm 给这组权重的版本标签。
- ImageNet 是数据集，不是模型的一部分。它相当于“教材”：别人用它把 ViT 的参数训练好，我们直接下载训练好的参数来用。同一个结构可以配不同的权重，比如主 notebook §1 里注释掉的 `vit_base_patch16_224.orig_in21k`，结构一样，但只在 ImageNet-21k 上训练过。
- 我们的项目是接着往下走一步：加载这组权重，把 1,000 类的分类头换成 10,000 类，再用自己的 45 万张图训练（07 只算一个分类头）。
- 这就是项目文档第 1 页指定的 “IN21K→IN1K Fine-Tuned ViT-B/16”。

**标签平滑举个例子**：训练目标里，正确的那一类不是 100%，而是 1 − 0.1 + 0.1/10000 ≈ 0.90001，其他每一类各分到 0.1/10000 = 0.00001。这样模型不会被逼着把某一类的概率推到 100%，能减轻过拟合。

### 1.3 数据和路径

```python
# ===== Data and paths =====
HF_REPO = "doem1997/cs604-course-data"       # the course dataset on Hugging Face
DATA_FROM_HF = True      # True: download the data from Hugging Face (about 500 MB/s on Colab); False: use the copy on Drive (about 33 MB/s)
DRIVE_DIR = "/content/drive/MyDrive/cs604"   # results are saved here
DRIVE_DATA_DIR = f"{DRIVE_DIR}/data"         # copy of the dataset on Drive, used only when DATA_FROM_HF = False (made by cs604_download_to_drive.ipynb)
DATA_DIR = "/content/data"                   # each session puts the data on the local Colab disk; the backbone reads images from here
DISCONNECT_WHEN_DONE = False  # release the runtime when done to stop using compute units (handy when running in the background)
```

| 变量 | 意思 |
|---|---|
| `HF_REPO` | 课程数据集在 Hugging Face 上的仓库名，里面有 `train.txt` 和 `train.tar.gz` |
| `DATA_FROM_HF` | `True`（默认）：每个会话从 Hugging Face 下载数据，Colab 上实测约 495 MB/s；`False`：改用 Drive 上的备份，约 33 MB/s。见 §3.1 |
| `DRIVE_DIR` | Drive 上的项目目录，所有结果都存在这下面 |
| `DRIVE_DATA_DIR` | 数据集在 Drive 上的备份，只在 `DATA_FROM_HF = False` 时用，要先用 `cs604_download_to_drive.ipynb` 存过去。`f"{DRIVE_DIR}/data"` 是 f-string：花括号里的变量会换成它的值，结果是 `/content/drive/MyDrive/cs604/data` |
| `DATA_DIR` | Colab 本地磁盘上的目录，数据下载、解压都放在这里，读图也从这里读 |
| `DISCONNECT_WHEN_DONE` | 跑完要不要自动断开运行时，见 §11 |

注意：§1 的值会在 §2 被“拍快照”存进 `BASE_CONFIG`。改了 §1 以后，要重新运行 §2 才生效。

---

## 2. Setup（§2）

### 2.1 安装 timm 和 hf_xet

```python
# Colab already has torch / torchvision; install timm here, and hf_xet for fast parallel downloads from Hugging Face
!pip install -q timm hf_xet
```

- 以 `!` 开头的行不是 Python，而是交给系统 shell 执行的命令，这是 Colab 和 Jupyter 的语法。
- `pip install timm hf_xet`：安装两个库，`-q` 是少输出一些信息。已经装过的话，这行什么也不做。
  - timm（PyTorch Image Models）：ViT 模型和预训练权重都从它来。
  - hf_xet：Hugging Face 新存储方式（Xet）的下载工具。装了它，§3 的 `hf_hub_download` 会把大文件切成很多块并行下载，Colab 上实测约 495 MB/s。
- PyTorch（`torch`）和 `torchvision` Colab 自带，不用装。

### 2.2 哪些设置可以按实验改

```python
# Settings an experiment in EXPERIMENTS may override
RUN_KEYS = ("MODEL_NAME", "HOLDOUT_PER_CLASS", "HEAD_INIT", "BATCH_SIZE", "LP_EPOCHS", "LP_LR", "LP_WD",
            "LP_BATCH", "LABEL_SMOOTHING", "SEED", "NUM_CLASSES")
BASE_CONFIG = {k: globals()[k] for k in RUN_KEYS}
```

- `RUN_KEYS`：允许在 `EXPERIMENTS` 里修改的设置名，是一个元组（tuple，可以理解为不能修改的列表）。`METHOD` 和各个路径不在里面，所以不能按实验改。
- `globals()`：返回一个 dict，装着 notebook 里所有的全局变量（变量名 → 值）。比如 `globals()["SEED"]` 就是 `SEED` 的值 0。
- `{k: globals()[k] for k in RUN_KEYS}`：字典推导式，结果是 `{"MODEL_NAME": "vit_base_...", "HOLDOUT_PER_CLASS": 2, ...}`，也就是 §1 默认值的快照。§4 的 `resolve_config` 会在这个快照上套用每个实验的修改。

### 2.3 导入库

```python
import os, gc, json, math, time, random, shutil, subprocess
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
from tqdm.auto import tqdm
from huggingface_hub import hf_hub_download
from google.colab import drive
```

| 导入 | 用来做什么 |
|---|---|
| `os` | 拼路径、建目录、读 CPU 数和内存大小 |
| `gc` | 手动触发垃圾回收，释放已经没用的对象 |
| `json` | 写 `config.json`、`summary.json`、`results.json` |
| `math` | `cos` 和 `ceil`：linear probe 的学习率曲线和总步数 |
| `time` | 计时、设置时区 |
| `random` | Python 自带的随机数：设种子、切分 holdout |
| `shutil` | 查看磁盘剩余空间；用 Drive 备份时复制 `train.txt` |
| `subprocess` | 调用系统命令 `tar` 解压数据 |
| `datetime` | 取当前时间：批次名、日志时间戳 |
| `SimpleNamespace` | 能用点号取值的对象：写 `cfg.SEED` 比 `cfg["SEED"]` 方便 |
| `numpy`（`np`） | 这里只用来设随机种子 |
| `torch` | PyTorch 本体：张量、GPU、自动求导 |
| `torch.nn`（`nn`） | 神经网络层，这里只用 `nn.Linear`：linear probe 的分类头 |
| `torch.nn.functional`（`F`） | 不带参数的函数：`F.normalize`（L2 归一化）、`F.cross_entropy`（交叉熵） |
| `torchvision.transforms`（`T`） | 图片预处理：缩放、裁剪、转成张量 |
| `Dataset`、`DataLoader` | 数据集和批量读取的基类 |
| `PIL.Image` | 打开、解码图片 |
| `timm` | 创建 ViT 模型、下载预训练权重、读取模型的预处理配置 |
| `tqdm` | 进度条 |
| `huggingface_hub.hf_hub_download` | 从 Hugging Face 下载数据集文件，§3 用 |
| `google.colab.drive` | 挂载 Google Drive |

### 2.4 挂载 Google Drive

```python
# ---- Mount Google Drive: data is read from here, and results are saved here, so nothing is lost on disconnect ----
drive.mount("/content/drive")
OUT_ROOT = DRIVE_DIR
```

- `drive.mount("/content/drive")`：把 Google Drive 挂到 Colab 的 `/content/drive` 目录。第一次运行会弹窗要你授权。之后 `/content/drive/MyDrive` 就是你 Drive 的根目录。
- `OUT_ROOT = DRIVE_DIR`：所有结果的根目录。放在 Drive 上，Colab 断线或者运行时被回收，结果也不会丢。

### 2.5 检查 GPU

```python
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
```

- `assert 条件, 信息`：条件不成立就报错停下，并显示后面的信息。这里要求两件事：
  - 有 GPU：`torch.cuda.is_available()`。
  - GPU 的“计算能力”主版本号不低于 8。`get_device_capability()` 返回 `(主版本, 次版本)`：A100 是 (8, 0)，L4 是 (8, 9)，H100 是 (9, 0)，T4 只有 (7, 5)。8 以上的 GPU 才原生支持 bf16（见附录 B）。
  - 信息外面那对括号只是为了换行，不是元组。
- `DEVICE = "cuda"`：之后的模型和张量都放到 GPU 上，`"cuda"` 指 NVIDIA GPU。
- `props = torch.cuda.get_device_properties(0)`：第 0 块 GPU 的信息。`props.name` 是型号，比如 `NVIDIA A100-SXM4-80GB`。
- `gib = props.total_memory / 1024**3`：显存字节数换算成 GiB。实际报告的数比标称值小一点，80 GB 的卡大约报 79 GiB。
- `VRAM_GB = next(...)`：在常见的标称容量里，找第一个不小于“实际值 × 0.98”的，于是 79 GiB 对应 80。`next(生成器, 默认值)` 取第一个满足条件的值；一个都没有，就用默认值 `round(gib)`。这个数用来自动选 `BATCH_SIZE`，也会写进日志。
- 两行 `allow_tf32 = True`：允许 float32 的矩阵乘法（`matmul`）和卷积（`cudnn`）用 TF32 格式计算，快很多，精度几乎不变。这个 notebook 里，linear probe 的训练和 `fit_logit_scale` 的矩阵乘法是 float32，会用上它；主干的计算在 bf16 autocast 里，不受影响。
- `NUM_WORKERS = min(16, os.cpu_count())`：读图用几个子进程并行，最多 16 个。Colab 的 A100 运行时有 12 个 CPU，所以是 12。
- `VERSIONS`：torch、CUDA、timm 的版本拼成一行字，写进日志，报告里说明实验环境时要用。

### 2.6 批次名和输出目录

```python
os.environ["TZ"] = "Asia/Singapore"  # the Colab VM runs on UTC; use local time for batch names and log timestamps
time.tzset()
BATCH_NAME = datetime.now().strftime("%m%d_%H%M")  # this session's runs are named <BATCH_NAME>_<i>_<method>
BATCH_DIR = os.path.join(OUT_ROOT, "batches", BATCH_NAME)
os.makedirs(BATCH_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
```

- 前两行把时区改成新加坡。Colab 的机器用的是 UTC，比新加坡晚 8 小时，不改的话批次名和日志时间都会差 8 小时。`time.tzset()` 让新时区立即生效。
- `datetime.now().strftime("%m%d_%H%M")`：把当前时间格式化成“月日_时分”，比如 `1102_1530`。
- `os.path.join(...)`：用 `/` 拼路径，结果是 `/content/drive/MyDrive/cs604/batches/1102_1530`。
- `os.makedirs(..., exist_ok=True)`：建目录，缺的上级目录也一起建；已经存在也不报错。`DATA_DIR` 是在本地放数据的目录。

### 2.7 日志函数

```python
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
```

- `LOG_FILE`：当前实验的 `train_log.txt` 的路径。没在跑实验时是 `None`；§8 的 `run_experiment` 开始时设好，结束时改回 `None`。
- `def log(*args)`：`*args` 表示可以传任意多个参数，比如 `log("a", 1)`。
- `msg = " ".join(str(a) for a in args)`：每个参数转成字符串，用空格连起来。
- `print(msg, flush=True)`：打印到 notebook 的输出里。`flush=True` 让它立刻显示，不攒着。
- `stamp`：时间戳，比如 `[2026-11-02 15:30:00] `。只写进文件，不打印。
- `for path in (...)`：每条日志写两个地方：这一批的总日志 `batch_log.txt`，以及当前实验的 `train_log.txt`。`if path:` 跳过 `None`（没在跑实验时）。
- `with open(path, "a") as f:`：`"a"` 是追加模式，写在文件末尾。每条日志都打开一次、写完就关，所以即使断线，已经写的日志也是完整的。项目要求用训练日志证明训练时间和显存，靠的就是这些文件。

### 2.8 记录运行环境

```python
ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
log(f"Batch {BATCH_NAME} -> {BATCH_DIR}")
log(f"Device {GPU_NAME} ({VRAM_GB} GB), bfloat16, {os.cpu_count()} CPUs, {ram_gb:.0f} GB RAM, "
    f"{shutil.disk_usage(DATA_DIR).free / 1e9:.0f} GB free disk")
log(f"Versions: {VERSIONS}")
```

- `ram_gb`：内存页的大小 × 物理页数 = 内存总字节数，除以 `1e9` 换成 GB。
- 三行 `log(...)`：记录批次名和目录，GPU 型号和显存、CPU 数、内存、本地磁盘剩余空间，以及库版本。
  - `{ram_gb:.0f}`：保留 0 位小数。
  - `shutil.disk_usage(DATA_DIR).free`：本地磁盘剩余的字节数。
  - 两个挨着的 f-string 会自动拼成一个字符串，这是为了让一行代码不要太长。

---

## 3. 准备数据（§3）

### 3.1 `prepare_split`：把数据放到本地磁盘

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
```

参数：`split` 是数据集的名字，这里只会传 `"train"`。返回值：本地列表文件 `/content/data/train.txt` 的路径。

- `list_path = os.path.join(DATA_DIR, f"{split}.txt")`：列表文件在本地的位置，`/content/data/train.txt`，每行对应一张图。`f"{split}.txt"` 里的 `{split}` 会换成 `train`。
- `marker = os.path.join(DATA_DIR, f".{split}_ready")`：标记文件 `/content/data/.train_ready`。文件名以点开头，是隐藏文件。
- `if os.path.exists(marker): return list_path`：有标记，说明这个会话已经准备好数据了，直接返回。同一个会话里再运行这个格子，不会重复下载。
- `t0 = time.time()`：记下开始的时间（秒）。
- `if DATA_FROM_HF:` 这一支（默认）：从 Hugging Face 下载。
  - `hf_hub_download(HF_REPO, f"{split}.txt", repo_type="dataset", local_dir=DATA_DIR)`：把 `train.txt`（28 MB）下载到 `/content/data/train.txt`。
  - `tar_path = hf_hub_download(HF_REPO, f"{split}.tar.gz", repo_type="dataset", local_dir=os.path.join(DATA_DIR, "hf"))`：把 40 GB 的 `train.tar.gz` 下载到 `/content/data/hf/`，返回它的本地路径。
  - 装了 hf_xet（§2.1），大文件会切成很多块并行下载，所以快。下载中断了，重新运行这个格子会接着下。
  - 下一行 `log`：记录下载用了几分钟。
- `else:` 这一支（`DATA_FROM_HF = False`）：用 Drive 上的备份。
  - `shutil.copy(...)`：把 Drive 上的 `train.txt` 复制到本地 `/content/data/train.txt`。
  - `tar_path = os.path.join(DRIVE_DATA_DIR, f"{split}.tar.gz")`：压缩包不复制，下面直接从 Drive 上解压，约 33 MB/s，要 20 分钟左右。
  - Drive 上没有这些文件（没跑过 `cs604_download_to_drive.ipynb`）的话，`shutil.copy` 会报 `FileNotFoundError`。
- `log(f"[{split}] extracting {tar_path} ...")`：记录开始解压哪个文件。
- `subprocess.run(["tar", "-xzf", tar_path, "-C", DATA_DIR], check=True)`：调用系统的 `tar` 命令解压。
  - `-x` 解压，`-z` 处理 gzip 压缩，`-f` 后面跟压缩包的文件名。
  - `-C DATA_DIR`：先进入这个目录再解压。解压后图片在 `/content/data/train/class_0000/xxx.jpg` 这样的位置。
  - `check=True`：命令失败就抛出异常，不会悄悄继续。
- `if DATA_FROM_HF: os.remove(tar_path)`：图片已经解压出来，下载的压缩包没用了，删掉，腾出 40 GB 磁盘。用 Drive 备份时压缩包在 Drive 上，不能删。
- `with open(list_path) as f:` 和下一行：数列表文件里有多少个非空行，也就是应该有多少张图（450,000）。`line.strip()` 去掉首尾的空白，空行会变成空字符串，被 `if` 过滤掉。`with` 保证文件用完自动关闭。
- `n_found = sum(len(files) for _, _, files in os.walk(...))`：`os.walk` 递归遍历 `/content/data/train`，每一层给出（目录路径, 子目录列表, 文件列表），这里把每一层的文件数加起来。`_` 表示这个值用不到。
- `assert n_found == n_expected, ...`：解压出来的图片数必须和列表文件一致，否则报错，防止数据不完整。
- `open(marker, "w").close()`：创建一个空的标记文件，表示这个会话的数据准备好了。
- 最后一行 `log`：记录从开始到准备好一共用了几分钟，`(time.time() - t0) / 60`。

`hf_hub_download` 的参数：

| 参数 | 意思 |
|---|---|
| 第一个参数 `HF_REPO` | 仓库名 `doem1997/cs604-course-data` |
| 第二个参数 | 仓库里的文件名，比如 `train.tar.gz` |
| `repo_type="dataset"` | 这是数据集仓库，不是模型仓库 |
| `local_dir` | 下载到哪个本地目录。文件直接放在这里，另外会生成一个很小的 `.cache` 文件夹，记录下载信息 |

为什么要把数据放到本地：训练和提特征要反复读 45 万张小图，直接从网络或 Drive 一张张读非常慢，本地磁盘快得多。下载加解压的过程中，本地磁盘要有约 80 GB 空闲（压缩包 40 GB，加上解压出来的 38 GB），§2 打印的 `GB free disk` 能看到还剩多少。

速度参考（都是 Colab 上实测）：从 Hugging Face 下载约 495 MB/s，40 GB 一分多钟下完，之后在本地解压还要几分钟；直接从 Drive 解压约 33 MB/s，要 20 分钟左右。

### 3.2 `read_list`：读列表文件

```python
def read_list(path):
    """Each line is '<image path> <class id>', e.g. 'train/class_0000/9aef28bf-....jpg 0'"""
    items = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if parts:
                items.append((parts[0], int(parts[1])))
    return items
```

- 列表文件的每一行是“图片路径 类别编号”，真实的样子是：`train/class_0000/9aef28bf-a69c-4598-bd4a-a356b4676f0a.jpg 0`。
- `items = []`：放结果的列表。
- `for line in f:`：一行一行地读。
- `parts = line.split()`：按空白切开，得到 `["train/class_0000/9aef...jpg", "0"]`。
- `if parts:`：跳过空行，空行切出来是空列表。
- `items.append((parts[0], int(parts[1])))`：存成（路径, 类别编号）这样的元组，编号转成整数。
- 返回一个 450,000 个元组的列表。

### 3.3 读训练集

```python
TRAIN_ALL = read_list(prepare_split("train"))
log(f"train {len(TRAIN_ALL):,} images | {NUM_WORKERS} loader workers")
```

- 先准备数据（需要的话：下载、解压），再读列表，得到 `TRAIN_ALL`：全部 450,000 张训练图的（路径, 类别）。
- `{len(TRAIN_ALL):,}`：`,` 表示加千位分隔符，显示成 `450,000`。

---

## 4. 模型（§4）

### 4.1 `is_head`：是不是分类头的参数

```python
def is_head(name):
    return name in ("head.weight", "head.bias")
```

- timm 的 ViT 里，参数名长这样：`cls_token`、`pos_embed`、`patch_embed.proj.weight`、`blocks.0.attn.qkv.weight`、……、`norm.weight`、`head.weight`、`head.bias`。分类头只有最后两个。
- 用在两个地方：§5 的 `make_ctx` 分开统计主干和分类头的参数量；§7 的 `save_state` 保存分类头。

### 4.2 `resolve_config`：合成这次实验的完整配置

```python
def resolve_config(overrides):
    """Defaults from section 1 + one experiment's overrides → the full config of this run"""
    unknown = set(overrides) - set(RUN_KEYS)
    assert not unknown, f"unknown keys in experiment config: {sorted(unknown)}"
    cfg = SimpleNamespace(**{**BASE_CONFIG, **overrides})
    assert cfg.HEAD_INIT in ("proto", "lp"), f"HEAD_INIT must be 'proto' or 'lp', not {cfg.HEAD_INIT!r}"
    assert cfg.HOLDOUT_PER_CLASS > 0, "this notebook evaluates on the holdout set, so keep at least 1 image per class"
    if cfg.BATCH_SIZE is None:
        cfg.BATCH_SIZE = 256 if VRAM_GB >= 40 else 128 if VRAM_GB >= 24 else 64
    return cfg
```

参数：`overrides` 是 `EXPERIMENTS` 里的一个 dict，比如 `{"HEAD_INIT": "lp"}`。返回值：这次实验的完整配置 `cfg`。

- `unknown = set(overrides) - set(RUN_KEYS)`：`set(一个 dict)` 得到它的 key 的集合；两个集合相减，得到不在 `RUN_KEYS` 里的名字，比如拼错了的 `HEAD_INT`。
- `assert not unknown, ...`：有不认识的名字就报错。不然拼错了也不会报错，只会悄悄用默认值。`sorted(...)` 让报错信息里的名字按字母排好。
- `cfg = SimpleNamespace(**{**BASE_CONFIG, **overrides})`：
  - `{**BASE_CONFIG, **overrides}`：把两个 dict 合成一个，同名的以后面的（实验的修改）为准。
  - `SimpleNamespace(**d)`：把 dict 变成能用点号取值的对象。之后就写 `cfg.HEAD_INIT`、`cfg.SEED`。
- 两个 `assert`：`HEAD_INIT` 只能是 `"proto"` 或 `"lp"`；`HOLDOUT_PER_CLASS` 至少是 1，因为后面要在 holdout 上测准确率。`{cfg.HEAD_INIT!r}` 会把值连引号一起显示，比如 `'random'`。
- `if cfg.BATCH_SIZE is None:` 和下一行：没指定就按显存选：40 GB 及以上用 256，24 GB 及以上用 128，否则 64。这是连着写的条件表达式，从左往右判断。A100 上是 256。
- `return cfg`。

### 4.3 `create_model`：加载预训练的 ViT

```python
def create_model(cfg):
    """Pretrained ViT-B/16 from timm; its 1000-class ImageNet head is replaced by a new, randomly initialized 10,000-class head"""
    return timm.create_model(cfg.MODEL_NAME, pretrained=True, num_classes=cfg.NUM_CLASSES).to(DEVICE)
```

| 参数 | 值 | 意思 |
|---|---|---|
| `cfg.MODEL_NAME` | `vit_base_patch16_224.augreg2_in21k_ft_in1k` | 要哪个模型、哪组权重 |
| `pretrained=True` | | 加载预训练权重。第一次运行会从 Hugging Face 下载（约 350 MB），之后用缓存 |
| `num_classes=cfg.NUM_CLASSES` | 10000 | 分类头输出 10,000 个分数。原来的头是 1,000 类（ImageNet），尺寸对不上，timm 会扔掉原来的头，换一个随机初始化的 768 → 10,000 新头；其余权重照常加载 |

- `.to(DEVICE)`：把整个模型搬到 GPU。参数仍然是 float32。
- 新头是随机的，但 §6 会用原型或 linear probe 的结果把它整个覆盖掉，所以随机成什么样都无所谓。

模型从输入到输出的结构：

1. `patch_embed`：一个 16×16、步长 16 的卷积，把 224×224 的图变成 14 × 14 = 196 个 768 维的 token。
2. 在最前面加一个 `cls_token`，一共 197 个 token；再加上位置编码 `pos_embed`，让模型知道每个图块在哪。
3. `blocks`：12 个 Transformer block。每个 block 先做 LayerNorm 和多头自注意力（`attn.qkv`、`attn.proj`），再做 LayerNorm 和 MLP（`mlp.fc1` 768 → 3072、GELU、`mlp.fc2` 3072 → 768），两部分都带残差连接。
4. `norm`：最后一个 LayerNorm。
5. 取 CLS token 的 768 维输出，送进 `head`（768 → 10,000），得到每一类的分数。

### 4.4 `apply_method`：冻结全部参数

```python
def apply_method(net, cfg):
    """Zero-shot / linear probe: nothing is trained with gradients; the head was already set in section 6"""
    return net.requires_grad_(False)
```

- `requires_grad_(False)`：把模型所有参数（包括分类头）设成“不需要梯度”，也就是不训练。返回的还是这个模型本身。
- 对 07 来说，这一步表明“什么都不训练”：分类头在 §6 已经设置好了，后面只做推理。其他 notebook（LoRA、Adapter 等）的主要区别就在这个函数。
- `cfg` 参数没用到，只是为了和其他 notebook 的写法保持一致。

### 4.5 `param_report`：统计参数量

```python
def param_report(net, cfg, ctx):
    """Trainable parameter counts. The backbone is frozen; a linear-probe head was trained in section 6, so it counts as trainable"""
    n_train = sum(p.numel() for p in net.parameters() if p.requires_grad)  # 0: apply_method froze everything
    lp = cfg.HEAD_INIT == "lp"
    return {"backbone_params": ctx.n_backbone, "head_params": ctx.n_head,
            "head_mode": "linear probe" if lp else "class prototypes (not trained)",
            "trainable_excl_head": n_train, "trainable_pct": 100 * n_train / ctx.n_backbone,
            "trainable_incl_head": n_train + (ctx.n_head if lp else 0)}
```

- `n_train = sum(p.numel() for p in net.parameters() if p.requires_grad)`：数需要梯度的参数一共多少个数，`numel()` 是一个张量里元素的个数。`apply_method` 已经冻结了全部参数，所以这里是 0。
- `lp = cfg.HEAD_INIT == "lp"`：是不是 linear probe。
- 返回一个 dict：

| key | zero-shot | linear probe | 意思 |
|---|---|---|---|
| `backbone_params` | 85,798,656 | 85,798,656 | 主干的参数量 |
| `head_params` | 7,690,000 | 7,690,000 | 分类头的参数量 |
| `head_mode` | `class prototypes (not trained)` | `linear probe` | 分类头是怎么来的 |
| `trainable_excl_head` | 0 | 0 | 可训练参数，不含分类头 |
| `trainable_pct` | 0.0 | 0.0 | 上一行占主干的百分比 |
| `trainable_incl_head` | 0 | 7,690,000 | 可训练参数，含分类头。linear probe 的分类头在 §6 训练过，所以算进去 |

---

## 5. 数据集和 holdout（§5）

### 5.1 `split_holdout`：每类留出几张

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
```

参数：`items` 是全部训练图（`TRAIN_ALL`），`k` 是每类留几张（2），`seed` 是随机种子（0）。返回：（训练部分, holdout），两个都是（路径, 类别）的列表。

- `by_class = {}`、`for path, label in items:`、`by_class.setdefault(label, []).append(path)`：按类别分组，得到 `{0: [45 个路径], 1: [45 个路径], ...}`。`setdefault(key, [])`：这个 key 还没有，就先放一个空列表，再返回它。
- `rng = random.Random(seed)`：一个独立的随机数生成器，只受这里的种子影响，别的代码用不用随机数都不影响它，所以每次切分的结果都一样。
- `train, holdout = [], []`。
- `for label in sorted(by_class):`：按类别编号从小到大处理，顺序固定。
- `paths = sorted(by_class[label])`：这一类的路径先排好序。这样即使列表文件里的顺序变了，结果也不变。
- `rng.shuffle(paths)`：打乱。
- `holdout += [(p, label) for p in paths[:k]]`：打乱后的前 k 张放进 holdout。
- `train += [(p, label) for p in paths[k:]]`：剩下的放进训练部分。
- 结果：训练部分 430,000 张（每类 43 张），holdout 20,000 张（每类 2 张）。

### 5.2 `ImageList`：按编号取图

```python
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
```

- 继承 PyTorch 的 `Dataset`：只要实现 `__len__`（一共几张）和 `__getitem__`（取第 i 张），`DataLoader` 就能用它批量读图。
- `__init__`：存下（路径, 类别）列表和预处理函数。
- `__len__`：返回图片数。
- `__getitem__(self, i)`：
  - `path, label = self.items[i]`：第 i 张的路径和类别。
  - `with Image.open(os.path.join(DATA_DIR, path)) as im:`：打开 `/content/data/train/...jpg`，`with` 用完自动关闭文件。
  - `img = im.convert("RGB")`：统一成 3 通道的 RGB，有的图可能是灰度图或者带透明通道。
  - `return self.transform(img), label`：返回预处理后的张量和类别。

### 5.3 `BICUBIC`

```python
BICUBIC = T.InterpolationMode.BICUBIC
```

- 双三次插值：缩放图片时，用周围 4×4 个像素算出新像素，比双线性插值更平滑。这个模型训练时用的就是它（timm 的配置里写的是 bicubic），推理时保持一致。

### 5.4 `make_ctx`：把这次实验要用的数据信息打包成 `ctx`

```python
def make_ctx(model, cfg):
    """Data setup for this run: train/holdout split, image transform, normalization values, param counts"""
    data_cfg = timm.data.resolve_model_data_config(model)
    size = data_cfg["input_size"][-1]
    train_items, holdout_items = split_holdout(TRAIN_ALL, cfg.HOLDOUT_PER_CLASS, cfg.SEED)
    return SimpleNamespace(
        img_size=size, train_items=train_items, holdout_items=holdout_items,
        mean=torch.tensor(data_cfg["mean"], device=DEVICE).view(1, 3, 1, 1) * 255,
        std=torch.tensor(data_cfg["std"], device=DEVICE).view(1, 3, 1, 1) * 255,
        # Only resize/crop on the CPU and output uint8; normalize on the GPU to save CPU work
        eval_tf=T.Compose([T.Resize(int(size / data_cfg["crop_pct"]), interpolation=BICUBIC),
                           T.CenterCrop(size), T.PILToTensor()]),
        n_backbone=sum(p.numel() for n, p in model.named_parameters() if not is_head(n)),
        n_head=sum(p.numel() for p in model.head.parameters()),
    )
```

`ctx` 可以理解成一个资料包：后面读图、提特征、评估、测延迟的函数都要知道同样几件事（哪些图是训练部分、哪些是 holdout，图片怎么缩放裁剪、怎么归一化，图片多大），`make_ctx` 在每次实验开头一次算好，装进一个 `SimpleNamespace` 对象，之后各个函数都带着 `ctx` 这个参数，用 `ctx.mean` 这样的写法取值。说“这次实验的”，是因为里面的东西取决于这次的配置（`HOLDOUT_PER_CLASS`、`SEED`）和模型自己的配置（图片大小、`mean`/`std`、裁剪比例），所以要在 `create_model` 之后现算。

| 字段 | 值 | 谁用 |
|---|---|---|
| `train_items` | 430,000 个（路径, 类别），训练部分 | 提特征、算原型 |
| `holdout_items` | 20,000 个（路径, 类别），每类留出的 2 张 | 最后在 holdout 上测准确率 |
| `img_size` | 224 | 测推理延迟时造随机输入 |
| `mean`、`std` | 都是 127.5 | `to_input` 归一化 |
| `eval_tf` | 短边缩放到 248 → 中间裁 224×224 → 转成张量 | 读图（提特征、评估） |
| `n_backbone`、`n_head` | 85,798,656、7,690,000 | 统计参数量 |

01–06 的 `ctx` 还多一个 `train_tf`（随机裁剪加翻转），训练时读图用。下面逐行看：

- `data_cfg = timm.data.resolve_model_data_config(model)`：从模型读出它要求的预处理方式。这个模型是：输入 `(3, 224, 224)`，插值 `bicubic`，均值 `(0.5, 0.5, 0.5)`，标准差 `(0.5, 0.5, 0.5)`，`crop_pct` 是 0.9。
- `size = data_cfg["input_size"][-1]`：取最后一个数，224。
- `split_holdout(TRAIN_ALL, cfg.HOLDOUT_PER_CLASS, cfg.SEED)`：切出训练部分和 holdout。
- `img_size=size`：224，测延迟时用来造随机输入。
- `mean=torch.tensor(data_cfg["mean"], device=DEVICE).view(1, 3, 1, 1) * 255`：
  - 放到 GPU 上，形状改成 (1, 3, 1, 1)，这样能和一批图片 (B, 3, 224, 224) 按通道相减（PyTorch 会自动扩展形状，叫“广播”）。
  - 乘 255，因为这里的图片是 0–255 的整数，不是 0–1 的小数。均值 0.5 × 255 = 127.5。
- `std=...`：同理，也是 127.5。所以归一化就是 `(像素 − 127.5) / 127.5`，结果落在 −1 到 1 之间。
- `eval_tf=T.Compose([...])`：图片预处理，依次做三步：
  1. `T.Resize(int(size / data_cfg["crop_pct"]), interpolation=BICUBIC)`：`int(224 / 0.9) = 248`，把短边缩放到 248，长宽比不变。
  2. `T.CenterCrop(size)`：从中间裁出 224×224。
  3. `T.PILToTensor()`：转成 uint8 张量，形状 (3, 224, 224)，数值还是 0–255。
  - 那行注释的意思：CPU 上只做缩放和裁剪，输出 uint8；归一化放到 GPU 上做（§5.7 的 `to_input`），减轻 CPU 的负担。
- `n_backbone=sum(p.numel() for n, p in model.named_parameters() if not is_head(n))`：主干的参数量，85,798,656。`named_parameters()` 同时给出参数名和参数。
- `n_head=sum(p.numel() for p in model.head.parameters())`：分类头的参数量，7,690,000。

### 5.5 `Loader`

```python
class Loader(DataLoader):
    """Before starting new worker processes, run garbage collection in the main process. Otherwise an old, interrupted DataLoader is copied into the new workers,
    and when it is collected there it prints lots of harmless AssertionError: can only test a child process messages."""

    def __iter__(self):
        gc.collect()
        return super().__iter__()
```

- 继承 `DataLoader`，只改了一处：每次开始遍历（`__iter__`）之前，先 `gc.collect()` 回收垃圾。
- 原因写在 docstring 里：如果之前有一个跑到一半被打断的 DataLoader 还没被回收，新开的读图子进程会把它也复制过去；它在子进程里被回收时，会打印一大堆无害但吓人的 `AssertionError`。先回收一下就没这个问题了。
- `super().__iter__()`：调用原来的 `DataLoader.__iter__`，其他行为都不变。

### 5.6 `make_loader`：批量读图

```python
def make_loader(items, ctx, cfg):
    """Center crops, in order, 2 × BATCH_SIZE images per batch (no gradients, so a big batch fits)"""
    return Loader(ImageList(items, ctx.eval_tf), batch_size=2 * cfg.BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
```

| 参数 | 值 | 意思 |
|---|---|---|
| `ImageList(items, ctx.eval_tf)` | | 要读哪些图、怎么预处理 |
| `batch_size=2 * cfg.BATCH_SIZE` | 512 | 每批 512 张。只做推理、不存梯度，显存够用，批大一点更快 |
| `num_workers=NUM_WORKERS` | 12 | 12 个子进程同时读图、解码、预处理 |
| `pin_memory=True` | | 把每批数据放进“锁页内存”，拷到 GPU 更快，还能边拷边算 |
| `shuffle`（没写，默认 `False`） | | 按顺序读，不打乱 |
| `drop_last`（没写，默认 `False`） | | 最后不满 512 张的那一批也保留 |

例子：430,000 张训练图分成 840 批，前 839 批各 512 张，最后一批 432 张。

### 5.7 `to_input`：拷到 GPU 并归一化

```python
def to_input(x, ctx):
    """uint8 images (B, 3, 224, 224) → normalized float32 on the GPU"""
    return (x.to(DEVICE, non_blocking=True).float() - ctx.mean) / ctx.std
```

- `x.to(DEVICE, non_blocking=True)`：把一批 uint8 图片拷到 GPU。配合锁页内存，`non_blocking=True` 让 CPU 不用干等拷贝结束。
- `.float()`：转成 float32。
- `(... - ctx.mean) / ctx.std`：归一化到 −1 到 1。

### 5.8 `autocast`：混合精度

```python
def autocast():
    """bf16 mixed precision: matrix multiplies run in bfloat16, the weights stay float32"""
    return torch.autocast("cuda", dtype=torch.bfloat16)
```

- 写在 `with autocast():` 里面的计算，矩阵乘法、卷积这类运算会自动用 bf16 算，又快又省显存；LayerNorm、softmax 这类对精度敏感的运算仍然用 float32。模型参数本身一直是 float32。

### 5.9 `free_gpu`：清理显存

```python
def free_gpu():
    """Free the GPU memory left by the previous run and reset the peak memory stats"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
```

- `gc.collect()`：回收 Python 里已经没人用的对象，比如上一个实验的模型。
- `torch.cuda.empty_cache()`：PyTorch 会缓存用过的显存方便下次用，这一步把缓存着但没在用的显存还给 GPU。
- `torch.cuda.reset_peak_memory_stats()`：把“峰值显存”的记录清零。这样每个实验记下的峰值显存，只算它自己用的。

---

## 6. 分类头（§6）：这个 notebook 的核心

先用一句话说清楚这一节在算什么：

- **zero-shot**：`第 c 类的分数 = s × cos(图片特征, 第 c 类的原型)`，分数最高的类就是预测结果。
- **linear probe**：从上面这个分类头出发，用梯度下降继续调分类头的权重和偏置，让它更能区分相似的类。

下面按代码顺序看。

### 6.1 `extract_features`：用冻结的主干提特征

```python
@torch.inference_mode()
def extract_features(net, items, ctx, cfg):
    """Features of the frozen backbone: the class-token output after the final LayerNorm, 768 numbers per image"""
    net.eval()
    feats, labels = [], []
    for x, y in tqdm(make_loader(items, ctx, cfg), desc="train features", leave=False):
        with autocast():
            f = net.forward_head(net.forward_features(to_input(x, ctx)), pre_logits=True)
        feats.append(f.half())
        labels.append(y.to(DEVICE))
    return torch.cat(feats), torch.cat(labels)
```

参数：`net` 是模型，`items` 是要提特征的图片（训练部分的 430,000 张）。返回：特征，形状 (430000, 768)，float16；标签，形状 (430000,)，int64。

- `@torch.inference_mode()`：装饰器，让整个函数在“推理模式”下运行：不记录求梯度要用的信息，更快、更省显存。
- `net.eval()`：切到评估模式，关掉 dropout 这类只在训练时才用的随机操作。
- `feats, labels = [], []`：每一批的结果先放进列表。
- `for x, y in tqdm(make_loader(items, ctx, cfg), desc="train features", leave=False):`：一批一批地读图，每批 512 张。`tqdm` 显示进度条，标题是 “train features”；`leave=False` 表示跑完后进度条消失。`x` 是 (512, 3, 224, 224) 的 uint8 图片，`y` 是这 512 张的类别编号。
- `with autocast():`：下面的计算用 bf16。
- `f = net.forward_head(net.forward_features(to_input(x, ctx)), pre_logits=True)`，从里往外看：
  1. `to_input(x, ctx)`：拷到 GPU 并归一化。
  2. `net.forward_features(...)`：图块嵌入 → 加 CLS token 和位置编码 → 12 个 block → 最后的 LayerNorm，输出 (512, 197, 768)：197 个 token，每个 768 维。
  3. `net.forward_head(..., pre_logits=True)`：取第 0 个 token（CLS token）的输出，得到 (512, 768)。`pre_logits=True` 表示停在分类头之前，不去算 10,000 类的分数。
- `feats.append(f.half())`：转成 float16（每个数 2 字节）存。430,000 × 768 个数，float16 约 0.66 GB，float32 要 1.3 GB。
- `labels.append(y.to(DEVICE))`：标签也放到 GPU 上。
- `return torch.cat(feats), torch.cat(labels)`：把 840 批拼成一个大张量。

### 6.2 `class_prototypes`：每一类的平均方向

```python
@torch.inference_mode()
def class_prototypes(feats, labels, num_classes):
    """Mean direction of each class: L2-normalize every feature, add them up per class, normalize the sums"""
    protos = torch.zeros(num_classes, feats.shape[1], device=feats.device)
    for i in range(0, len(feats), 65536):
        protos.index_add_(0, labels[i:i + 65536], F.normalize(feats[i:i + 65536].float(), dim=1))
    return F.normalize(protos, dim=1)
```

参数：`feats` (430000, 768)，`labels` (430000,)，`num_classes` 10000。返回：`protos`，形状 (10000, 768)，每一行是一个长度为 1 的向量，代表一个类的“原型”。

- `protos = torch.zeros(num_classes, feats.shape[1], device=feats.device)`：10,000 × 768 的全 0 矩阵，和特征放在同一个设备（GPU）上。`feats.shape[1]` 是 768。
- `for i in range(0, len(feats), 65536):`：每次处理 65,536 个特征，一共分 7 块。分块是为了省显存：一次把 43 万个特征全转成 float32，要多占 1.3 GB。
- `F.normalize(feats[i:i + 65536].float(), dim=1)`：每个特征除以它自己的长度（L2 范数），变成长度为 1 的向量。这样只保留“方向”，特征本身的长短不再影响结果。
- `protos.index_add_(0, labels[i:i + 65536], ...)`：按标签累加：第 j 个特征加到 `protos[labels[j]]` 那一行上。函数名末尾的 `_` 表示直接修改 `protos` 本身。
- `return F.normalize(protos, dim=1)`：每一类的和再归一化成长度 1。这和“先求平均再归一化”是一回事：除以样本数 43 不改变方向。

写成公式：`原型_c = normalize( 第 c 类所有 normalize(特征) 的和 )`。

### 6.3 `fit_logit_scale`：挑一个温度

```python
@torch.inference_mode()
def fit_logit_scale(protos, feats, labels, n=10000):
    """Pick a good temperature for cosine similarity, so the softmax of the prototype head is not too flat or too sharp"""
    idx = torch.randperm(len(feats), device=feats.device)[:n]
    cos = F.normalize(feats[idx].float(), dim=1) @ protos.T
    losses = {s: F.cross_entropy(cos * s, labels[idx]).item() for s in (5, 10, 15, 20, 30, 40, 60, 80)}
    return min(losses, key=losses.get)
```

参数：`protos` 原型，`feats` 和 `labels` 训练特征和标签，`n` 抽多少个样本来试（10,000）。返回：候选值里最合适的温度 `s`。

为什么需要温度：图片特征和原型的余弦相似度在 −1 到 1 之间，各类之间差别很小。直接拿去做 softmax，得到的概率会非常平均，每一类都差不多是 1/10000。乘一个系数 s 把差距放大，s 越大，概率越集中在分数最高的几类上。

- `idx = torch.randperm(len(feats), device=feats.device)[:n]`：把 0 到 429,999 随机打乱，取前 10,000 个，也就是随机抽 10,000 个训练样本。
- `cos = F.normalize(feats[idx].float(), dim=1) @ protos.T`：抽出的特征先归一化，再和所有原型做矩阵乘法，得到 (10000, 10000) 的余弦相似度矩阵：第 i 行第 c 列，是第 i 个样本和第 c 类原型的余弦相似度。`@` 是矩阵乘法，`.T` 是转置。两个长度为 1 的向量，点积就是余弦相似度。
- `losses = {s: F.cross_entropy(cos * s, labels[idx]).item() for s in (5, 10, 15, 20, 30, 40, 60, 80)}`：对每个候选温度，把 `cos * s` 当作各类的分数，算交叉熵损失。`.item()` 把只有一个数的张量变成 Python 的数字。结果是 `{5: 损失, 10: 损失, ...}`。
- `return min(losses, key=losses.get)`：返回损失最小的那个 s。

要点：

- 所有类的分数都乘同一个 s，哪一类分数最高不会变，所以**温度不影响 zero-shot 的 top-1 准确率**，只影响概率有多“自信”。
- 温度的作用是给 linear probe（以及其他方法的微调）一个合适的起点：起点的损失太大或太小，训练都会受影响。
- 挑温度只用训练特征，完全没碰 holdout。

### 6.4 `linear_probe`：只训练分类头

```python
def linear_probe(feats, labels, weight, bias, cfg):
    """Train only the 768 → 10,000 head on the cached features, starting from the prototype head"""
    head = nn.Linear(feats.shape[1], cfg.NUM_CLASSES).to(DEVICE)
    with torch.no_grad():
        head.weight.copy_(weight)
        head.bias.copy_(bias)
    opt = torch.optim.AdamW(head.parameters(), lr=cfg.LP_LR, weight_decay=cfg.LP_WD)
    total = cfg.LP_EPOCHS * math.ceil(len(feats) / cfg.LP_BATCH)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / total)))
    for _ in tqdm(range(cfg.LP_EPOCHS), desc="linear probe", leave=False):
        perm = torch.randperm(len(feats), device=feats.device)
        for i in range(0, len(feats), cfg.LP_BATCH):
            idx = perm[i:i + cfg.LP_BATCH]
            loss = F.cross_entropy(head(feats[idx].float()), labels[idx], label_smoothing=cfg.LABEL_SMOOTHING)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    return head.weight.detach(), head.bias.detach()
```

参数：`feats` 和 `labels` 是训练特征和标签（在 GPU 上）；`weight` 和 `bias` 是起点，也就是原型分类头；`cfg` 是配置。返回：训练好的分类头权重 (10000, 768) 和偏置 (10000,)。

这个函数前面没有 `@torch.inference_mode()`，因为它要算梯度。

- `head = nn.Linear(feats.shape[1], cfg.NUM_CLASSES).to(DEVICE)`：新建一个 768 → 10,000 的线性层，也就是一个分类头：`分数 = 权重 · 特征 + 偏置`。
- `with torch.no_grad():` 和下面两行 `copy_`：把原型分类头的权重和偏置复制进去，作为训练的起点。`no_grad` 表示这次复制不算进求梯度的计算里。
- `opt = torch.optim.AdamW(head.parameters(), lr=cfg.LP_LR, weight_decay=cfg.LP_WD)`：AdamW 优化器，负责根据梯度更新分类头。
  - `lr=1e-3`：学习率，每一步改动的幅度。
  - `weight_decay=1e-4`：每一步把权重往 0 拉一点，是一种正则化。这里权重和偏置都做衰减。
- `total = cfg.LP_EPOCHS * math.ceil(len(feats) / cfg.LP_BATCH)`：总步数 = 10 × ⌈430,000 / 4,096⌉ = 10 × 105 = 1,050。`math.ceil` 是向上取整，因为最后不满 4,096 的那一批也算一步。
- `sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / total)))`：学习率调度。每一步的学习率 = `LP_LR × 0.5 × (1 + cos(π × s / total))`，s 是已经走了几步：s = 0 时乘 1，走到一半乘 0.5，走到最后接近 0。这叫余弦退火：前期大步走，后期小步微调。
- `for _ in tqdm(range(cfg.LP_EPOCHS), desc="linear probe", leave=False):`：10 个 epoch，每个 epoch 把 430,000 个特征全部过一遍。
- `perm = torch.randperm(len(feats), device=feats.device)`：每个 epoch 都把样本顺序重新打乱。
- `for i in range(0, len(feats), cfg.LP_BATCH):` 和 `idx = perm[i:i + cfg.LP_BATCH]`：每次取 4,096 个样本的编号，最后一批是 4,016 个。
- `loss = F.cross_entropy(head(feats[idx].float()), labels[idx], label_smoothing=cfg.LABEL_SMOOTHING)`：
  - `head(feats[idx].float())`：4,096 个特征（先转回 float32）过分类头，得到 (4096, 10000) 的分数。
  - `F.cross_entropy(分数, 标签, label_smoothing=0.1)`：交叉熵损失，正确类别的概率越高，损失越小。带标签平滑，见 §1.2。
- `opt.zero_grad(set_to_none=True)`：清掉上一步的梯度。PyTorch 默认会把梯度累加起来，所以每一步都要先清。`set_to_none=True` 直接设成 None，比填 0 稍快。
- `loss.backward()`：反向传播，算出损失对分类头权重和偏置的梯度。
- `opt.step()`：AdamW 根据梯度更新一次参数。
- `sched.step()`：学习率调度往前走一步。
- `return head.weight.detach(), head.bias.detach()`：`detach()` 得到不带梯度信息的张量，返回训练好的权重和偏置。

为什么这么快：每一步只是一个 768 → 10,000 的矩阵乘法，图片不用再过一遍主干，特征已经提好了。

### 6.5 `FEATURE_CACHE` 和 `cached_features`：提一次特征，两个实验共用

```python
FEATURE_CACHE = {}  # the features extracted by the first run, reused by the second run


def cached_features(model, ctx, cfg):
    """Frozen-backbone features of the training images. Extracted once; later runs with the same model and split reuse them"""
    key = (cfg.MODEL_NAME, cfg.HOLDOUT_PER_CLASS, cfg.SEED)
    if key in FEATURE_CACHE:
        log("Reusing the backbone features extracted by the previous run")
        return FEATURE_CACHE[key]
    t0 = time.time()
    feats, labels = extract_features(model, ctx.train_items, ctx, cfg)
    torch.cuda.synchronize()
    # Keep them on the CPU between runs, so they don't count toward the next run's GPU memory
    FEATURE_CACHE[key] = dict(feats=feats.cpu(), labels=labels.cpu(), seconds=time.time() - t0)
    log(f"Extracted features for {len(feats):,} train images in {FEATURE_CACHE[key]['seconds'] / 60:.1f} min")
    return FEATURE_CACHE[key]
```

返回一个 dict：`feats` 特征、`labels` 标签（都在 CPU 内存里），`seconds` 提特征花了几秒。

- `FEATURE_CACHE = {}`：一个全局的 dict，存提好的特征。
- `key = (cfg.MODEL_NAME, cfg.HOLDOUT_PER_CLASS, cfg.SEED)`：缓存的 key。特征只取决于这三样：用哪个模型，以及哪些图属于训练部分（由 holdout 数量和种子决定）。三样都一样，就可以直接复用。
- `if key in FEATURE_CACHE:` 那三行：第二个实验（linear probe）走这里，直接返回第一个实验提好的特征。
- `t0 = time.time()`：开始计时。
- `feats, labels = extract_features(model, ctx.train_items, ctx, cfg)`：给 430,000 张训练图提特征。A100 上大约 6–7 分钟，是这个 notebook 最花时间的一步。
- `torch.cuda.synchronize()`：等 GPU 把排着队的计算全部做完再停表。GPU 是异步计算的：Python 这边已经往下走了，GPU 可能还在算。
- `FEATURE_CACHE[key] = dict(feats=feats.cpu(), labels=labels.cpu(), seconds=time.time() - t0)`：特征拷到 CPU 内存里，连同用时一起存进缓存。上面那行注释说明了原因：放在 CPU 上，就不会占用下一个实验的显存，也就不会算进它的峰值显存。
- 最后的 `log`：记录提了多少张图、花了几分钟。

### 6.6 `init_head`：设置分类头

```python
def init_head(model, ctx, cfg):
    """Set the head from the class prototypes (zero-shot), or from a linear probe trained from them (lp).
    Returns the seconds spent; feature extraction counts too, even when the previous run already extracted the features"""
    cached = cached_features(model, ctx, cfg)
    t0 = time.time()
    feats, labels = cached["feats"].to(DEVICE), cached["labels"].to(DEVICE)
    protos = class_prototypes(feats, labels, cfg.NUM_CLASSES)
    scale = fit_logit_scale(protos, feats, labels)
    # The head works on unnormalized features: divide by the mean feature norm so logits ≈ scale × cosine similarity
    weight = protos * (scale / feats.float().norm(dim=1).mean())
    bias = torch.zeros(cfg.NUM_CLASSES, device=DEVICE)
    if cfg.HEAD_INIT == "lp":
        weight, bias = linear_probe(feats, labels, weight, bias, cfg)
    with torch.no_grad():
        model.head.weight.copy_(weight)
        model.head.bias.copy_(bias)
    torch.cuda.synchronize()
    seconds = cached["seconds"] + time.time() - t0
    source = "a linear probe started from the class prototypes" if cfg.HEAD_INIT == "lp" else "the class prototypes"
    log(f"Head set from {source} (prototype logit scale {scale}) in {seconds / 60:.1f} min incl. feature extraction, "
        f"counted in this run's GPU time")
    return seconds
```

参数：`model` 是刚建好的模型。返回：这一步花的秒数（包括提特征），算进这次实验的 GPU 时间。

- `cached = cached_features(model, ctx, cfg)`：拿到特征。第一个实验现提，第二个实验复用。
- `t0 = time.time()`：从这里开始，计算设置分类头本身用的时间。
- `feats, labels = cached["feats"].to(DEVICE), cached["labels"].to(DEVICE)`：特征和标签搬回 GPU。
- `protos = class_prototypes(feats, labels, cfg.NUM_CLASSES)`：算出每一类的原型，(10000, 768)。
- `scale = fit_logit_scale(protos, feats, labels)`：挑温度。
- `weight = protos * (scale / feats.float().norm(dim=1).mean())`：把原型变成分类头的权重。为什么要除以特征的平均长度：
  - 分类头直接作用在**没有归一化**的特征上：`分数 = 权重 · 特征`。
  - 设权重 = `s · p / L`，其中 p 是长度为 1 的原型，L 是特征的平均长度。那么 `分数 = s · (‖f‖ / L) · cos(f, p)`。
  - 每张图的特征长度 ‖f‖ 都和平均长度 L 差不多，所以分数 ≈ `s × 余弦相似度`，正好是 `fit_logit_scale` 里试的那种形式。
  - `feats.float().norm(dim=1)` 算出每个特征的长度，`.mean()` 取平均。
- `bias = torch.zeros(cfg.NUM_CLASSES, device=DEVICE)`：偏置全 0。
- `if cfg.HEAD_INIT == "lp":` 和下一行：linear probe 从这个原型分类头出发继续训练，训练结果替换掉 `weight` 和 `bias`。zero-shot 跳过这两行。
- `with torch.no_grad():` 和下面两行 `copy_`：把权重和偏置写进模型的分类头。
- `torch.cuda.synchronize()`：等 GPU 算完再停表。
- `seconds = cached["seconds"] + time.time() - t0`：提特征的时间 + 设置分类头的时间。第二个实验虽然复用了特征，也把提特征的时间加上了，所以两个实验记下的 GPU 时间都是如实的，相当于各自单独跑一遍。
- `source = ...`：日志里说明分类头来自哪里。
- 最后的 `log`：记录分类头的来源、温度和用时。

---

## 7. 评估、推理延迟、保存（§7）

### 7.1 `evaluate`：在 holdout 上测准确率

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

参数：`net` 是模型，`loader` 读的是有标签的图片（这里是 holdout）。返回：（top-1 准确率, top-5 准确率），都是 0 到 1 之间的小数。

- `net.eval()`：评估模式。
- `c1 = c5 = n = 0`：top-1 答对的张数、top-5 答对的张数、总张数。
- `for x, y in loader:`：一批 512 张。holdout 有 20,000 张，分成 40 批，最后一批 32 张。
- `top5 = net(to_input(x, ctx)).topk(5, dim=1).indices`：
  - `net(...)`：图片过整个模型（主干 + 分类头），得到 (512, 10000) 的分数。
  - `.topk(5, dim=1)`：每张图取分数最高的 5 个类；`.indices` 是这 5 个类的编号，形状 (512, 5)，按分数从高到低排。
- `y = y.to(DEVICE)`：标签搬到 GPU。
- `c1 += (top5[:, 0] == y).sum().item()`：第 0 列是分数最高的类，等于标签就是 top-1 答对。`.sum()` 数有几个 True。
- `c5 += (top5 == y[:, None]).any(dim=1).sum().item()`：`y[:, None]` 把标签的形状变成 (512, 1)，和 (512, 5) 比较时会自动扩展，得到 (512, 5) 的 True/False；`.any(dim=1)` 表示 5 个里只要有一个对就算对。这就是 top-5。
- `n += len(y)`。
- `return c1 / n, c5 / n`。

注意：评估时图片会**重新过一遍主干**，不用 §6 缓存的特征。所以它把“读图 → 预处理 → 主干 → 分类头”这一整条流程从头验证了一遍。

### 7.2 `save_state`：保存分类头

```python
def save_state(net, path, **extra):
    """Save the classifier head, about 31 MB; nothing else has changed"""
    state = {n: p.detach().cpu() for n, p in net.named_parameters() if p.requires_grad or is_head(n)}
    torch.save({"state": state, **extra}, path)
```

- `state = {...}`：挑出“需要梯度”或者“属于分类头”的参数，拷到 CPU。07 里没有需要梯度的参数，所以只有 `head.weight` 和 `head.bias`。
- `torch.save({"state": state, **extra}, path)`：存成 `final.pt`。`**extra` 是调用时额外传进来的内容，这里是 `config=vars(cfg)`，也就是这次实验的配置。
- 文件大约 31 MB：7,690,000 个 float32，每个 4 字节。主干没有变，不用存。

### 7.3 `measure_latency`：测一次前向要多久

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
```

参数：`batch_size` 一次几张图，`iters` 计时跑几次，`warmup` 计时前先空跑几次（10）。返回：平均每次前向计算要几秒。

- `x = torch.randn(batch_size, 3, ctx.img_size, ctx.img_size, device=DEVICE)`：随机生成一批 224×224 的“图片”。用随机数，是因为这里只测模型的计算速度，不包括读图和解码。
- `for _ in range(warmup):` 那三行：先空跑 10 次。GPU 刚开始跑时要挑选计算方法、分配显存，前几次特别慢，不能算进去。
- `torch.cuda.synchronize()`：等热身全部做完。
- `t0 = time.perf_counter()`：高精度计时器开始计时。
- `for _ in range(iters):` 那三行：正式跑 `iters` 次。
- 再 `torch.cuda.synchronize()` 一次：等最后一次真正算完。
- `return (time.perf_counter() - t0) / iters`：平均每次的秒数。

### 7.4 `latency_report`：延迟和吞吐量

```python
def latency_report(net, ctx):
    lat_1 = measure_latency(net, 1, 100, ctx)
    lat_b = measure_latency(net, 256, 20, ctx)
    setting = f"{GPU_NAME}, bfloat16, no flip TTA"
    log(f"Inference latency (batch 1): {lat_1 * 1e3:.2f} ms/img | throughput (batch 256): {256 / lat_b:.0f} img/s | {setting}")
    return dict(latency_ms_bs1=lat_1 * 1e3, throughput_img_s=256 / lat_b, latency_setting=setting)
```

- `lat_1 = measure_latency(net, 1, 100, ctx)`：一次 1 张，跑 100 次取平均，这就是“推理延迟”：处理一张图要多久。
- `lat_b = measure_latency(net, 256, 20, ctx)`：一次 256 张，跑 20 次，用来算吞吐量。
- `setting`：测量条件（GPU 型号、bf16、没有翻转 TTA）。报告里写延迟时要注明这些。
- `log(...)`：`lat_1 * 1e3` 把秒换成毫秒；`256 / lat_b` 是每秒能处理多少张。
- 返回 `latency_ms_bs1`（毫秒）、`throughput_img_s`（张/秒）、`latency_setting`。
- 参考：之前 zero-shot 在 A100 上测出约 7.55 ms。两个实验的模型结构完全一样，延迟应该差不多。

---

## 8. 一次实验的完整流程（§8 `run_experiment`）

参数：`overrides` 是这次实验的修改（比如 `{"HEAD_INIT": "lp"}`），`run_name` 是实验名。返回：`summary`，这次实验的全部结果。函数比较长，分三段看。

### 8.1 准备

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
        log(f"{METHOD}: head {cfg.HEAD_INIT}, holdout {cfg.HOLDOUT_PER_CLASS}/class"
            + (f", linear probe {cfg.LP_EPOCHS} epochs (lr {cfg.LP_LR}, wd {cfg.LP_WD}, batch {cfg.LP_BATCH})"
               if cfg.HEAD_INIT == "lp" else ""))
        log(f"Environment: {GPU_NAME} ({VRAM_GB} GB), bfloat16, {os.cpu_count()} CPUs | {VERSIONS}")
        random.seed(cfg.SEED)
        np.random.seed(cfg.SEED)
        torch.manual_seed(cfg.SEED)
        free_gpu()
```

- `global LOG_FILE`：函数里要给全局变量 `LOG_FILE` 赋值，必须先声明 `global`，否则 Python 会把它当成一个新的局部变量。
- `cfg = resolve_config(overrides)`：合成完整配置。
- `run_dir = os.path.join(OUT_ROOT, "runs", run_name)`：这次实验的目录，`.../cs604/runs/<实验名>`。
- `os.makedirs(run_dir, exist_ok=True)`：建目录。
- `LOG_FILE = os.path.join(run_dir, "train_log.txt")`：从现在起，`log()` 也会写进这个文件。
- `try:`（和最后的 `finally:` 配对）：不管中间出没出错，最后都会把 `LOG_FILE` 改回 `None`，免得下一次的日志写错文件。
- `json.dump({**vars(cfg), "RUN_NAME": run_name, "overrides": overrides}, f, indent=2)`：`vars(cfg)` 把配置对象变回 dict，再加上实验名和修改项，存成 `config.json`。`indent=2` 让文件带缩进，方便看。
- 三行 `log`：
  - 标题行。`\n` 是换行，让标题前空一行。
  - 配置摘要。linear probe 时用条件表达式 `(... if cfg.HEAD_INIT == "lp" else "")` 多拼一段 linear probe 的参数。
  - 运行环境。
- `random.seed(cfg.SEED)`、`np.random.seed(cfg.SEED)`、`torch.manual_seed(cfg.SEED)`：把三个库的随机种子都设成 0，这样随机抽样（`fit_logit_scale`）和打乱顺序（`linear_probe`）每次都一样，结果可以复现。`torch.manual_seed` 同时设定 CPU 和 GPU 的随机数。
- `free_gpu()`：清掉上一个实验留下的显存，并把峰值显存的记录清零。

### 8.2 主要步骤

```python
        model = create_model(cfg)
        ctx = make_ctx(model, cfg)
        seconds = init_head(model, ctx, cfg)
        model = apply_method(model, cfg)
        params = param_report(model, cfg, ctx)
        log(f"Trainable (excl. head): {params['trainable_excl_head']:,} | head {ctx.n_head:,} params ({params['head_mode']})")
        save_state(model, os.path.join(run_dir, "final.pt"), config=vars(cfg))
        log(f"GPU time: {seconds / 3600:.2f} GPU-hours (feature extraction + classifier head; the backbone is not trained)")
        # The project doc asks for VRAM in the training log; reserved is closer to what nvidia-smi shows
        log(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB allocated, "
            f"{torch.cuda.max_memory_reserved() / 1e9:.1f} GB reserved by PyTorch (GPU has {VRAM_GB} GB)")
```

- `model = create_model(cfg)`：每个实验都新建一个模型。第二个实验不会沿用第一个实验的分类头。
- `ctx = make_ctx(model, cfg)`：切分数据、准备预处理、统计参数量。
- `seconds = init_head(model, ctx, cfg)`：设置分类头，拿到这次实验的 GPU 时间（秒）。
- `model = apply_method(model, cfg)`：冻结全部参数。
- `params = param_report(model, cfg, ctx)` 和下一行 `log`：统计并记录参数量。
- `save_state(...)`：把分类头存成 `final.pt`。
- `log(f"GPU time: {seconds / 3600:.2f} GPU-hours ...")`：秒换算成小时。这就是报告里这个方法的“训练时间”。
- 峰值显存那一行：
  - `torch.cuda.max_memory_allocated()`：从上次清零（`free_gpu`）到现在，张量最多占了多少显存。
  - `torch.cuda.max_memory_reserved()`：PyTorch 最多向 GPU 预订了多少显存（包括缓存），更接近 `nvidia-smi` 显示的数字。
  - 除以 `1e9` 换成 GB。项目文档要求在训练日志里写显存。

### 8.3 评估和汇总

```python
        top1, top5 = evaluate(model, make_loader(ctx.holdout_items, ctx, cfg), ctx)
        log(f"Final holdout: top-1 {top1:.2%}, top-5 {top5:.2%}")
        latency = latency_report(model, ctx)

        summary = dict(run=run_name, method=METHOD, head_init=cfg.HEAD_INIT, model=cfg.MODEL_NAME, overrides=overrides,
                       **params, holdout_top1=top1, holdout_top5=top5, **latency, gpu=GPU_NAME, vram_gb=VRAM_GB,
                       gpu_hours=seconds / 3600, peak_vram_gb=torch.cuda.max_memory_allocated() / 1e9,
                       peak_vram_reserved_gb=torch.cuda.max_memory_reserved() / 1e9, versions=VERSIONS)
        with open(os.path.join(run_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        log(f"== {run_name} finished: holdout top-1 {top1:.2%} | {params['head_mode']} | {summary['gpu_hours']:.2f} GPU-h")
        return summary
    finally:
        LOG_FILE = None
```

- `top1, top5 = evaluate(...)` 和下一行 `log`：在 holdout 上测准确率。`:.2%` 显示成两位小数的百分比，比如 0.4395 显示成 43.95%。
- `latency = latency_report(model, ctx)`：测推理延迟。
- `summary = dict(...)`：汇总结果。`**params` 和 `**latency` 把那两个 dict 的内容展开放进来。各个字段：

| 字段 | 意思 |
|---|---|
| `run`、`method`、`head_init`、`model`、`overrides` | 实验名、方法、分类头怎么来、模型名、这次的修改 |
| `backbone_params` 到 `trainable_incl_head` | 来自 `param_report`，见 §4.5 |
| `holdout_top1`、`holdout_top5` | holdout 准确率，0 到 1 的小数 |
| `latency_ms_bs1`、`throughput_img_s`、`latency_setting` | 来自 `latency_report` |
| `gpu`、`vram_gb` | GPU 型号、显存 |
| `gpu_hours` | GPU 时间（小时） |
| `peak_vram_gb`、`peak_vram_reserved_gb` | 峰值显存。注意它是在评估和测延迟**之后**读的，可能比上面日志里写的大一点 |
| `versions` | 库的版本 |

- 写 `summary.json`，再记一行总结，`return summary`。
- `finally: LOG_FILE = None`：实验结束（或者出错），日志不再写进这个实验的文件。

---

## 9. 运行实验（§9）

```python
RESULTS = []
for i, exp in enumerate(EXPERIMENTS, 1):
    result = run_experiment(dict(exp), f"{BATCH_NAME}_{i}_{METHOD}_{exp.get('HEAD_INIT', HEAD_INIT)}")
    RESULTS.append(result)
    if result["head_init"] == "proto" and result["holdout_top1"] < 0.25:
        raise RuntimeError(f"Zero-shot holdout top-1 is only {result['holdout_top1']:.1%} (expected ~43.7%): "
                           "the data pipeline is probably broken, stopping. Please check the log.")
```

- `RESULTS = []`：收集每个实验的 summary。
- `for i, exp in enumerate(EXPERIMENTS, 1):`：`enumerate(..., 1)` 同时给出序号（从 1 开始）和元素。
- `dict(exp)`：复制一份再传进去，防止函数里的改动连带改掉 `EXPERIMENTS`。
- 实验名 `f"{BATCH_NAME}_{i}_{METHOD}_{exp.get('HEAD_INIT', HEAD_INIT)}"`：比如 `1102_1530_1_linear_proto`。`exp.get('HEAD_INIT', HEAD_INIT)`：这个实验写了 `HEAD_INIT` 就用它，没写就用默认值。
- `RESULTS.append(result)`。
- 最后的 `if`：如果 zero-shot 的 holdout 准确率低于 25%（正常大约 44%），多半是数据出了问题，比如标签和图片对不上。这时直接报错停下，不浪费时间跑后面的实验。`raise RuntimeError(...)` 抛出异常并显示提示；`:.1%` 是一位小数的百分比。

---

## 10. 结果表（§10）

```python
header = ("| Run | Classifier head | Holdout top-1 | Holdout top-5 | Trainable params (incl. head) | GPU-hours | "
          "Peak VRAM (GB) | Latency bs1 (ms) | Throughput (img/s) |\n|---|---|---|---|---|---|---|---|---|")
rows = [f"| {r['run']} | {r['head_mode']} | {r['holdout_top1']:.2%} | {r['holdout_top5']:.2%} | "
        f"{r['trainable_incl_head']:,} | {r['gpu_hours']:.2f} | {r['peak_vram_gb']:.1f} | "
        f"{r['latency_ms_bs1']:.2f} | {r['throughput_img_s']:.0f} |" for r in RESULTS]
table = header + "\n" + "\n".join(rows)
with open(os.path.join(BATCH_DIR, "results.md"), "w") as f:
    f.write(table + "\n")
with open(os.path.join(BATCH_DIR, "results.json"), "w") as f:
    json.dump(RESULTS, f, indent=2)
log("\n" + table)

from IPython.display import Markdown, display
display(Markdown(table))
```

- `header`：Markdown 表格的表头和分隔行（`|---|---|`）。括号里两个挨着的字符串会自动拼起来，`\n` 是换行。
- `rows = [f"..." for r in RESULTS]`：每个实验一行。格式写法：`:.2%` 两位小数的百分比，`:,` 千位分隔符，`:.2f` 两位小数，`:.1f` 一位小数，`:.0f` 取整。
- `table = header + "\n" + "\n".join(rows)`：拼成完整的表格。
- 存两个文件：`results.md` 是这张表，`results.json` 是每个实验完整的 summary。
- `log("\n" + table)`：表格也写进日志。
- `from IPython.display import Markdown, display` 和 `display(Markdown(table))`：在 notebook 里把表格渲染出来。

---

## 11. 自动断开（§11）

```python
if DISCONNECT_WHEN_DONE:
    from google.colab import runtime
    log("All done; releasing the runtime ...")
    drive.flush_and_unmount()  # make sure all results are written to Drive
    runtime.unassign()
```

- 默认 `False`，什么都不做。
- 设成 `True` 的话：
  - `drive.flush_and_unmount()`：先把还没写完的文件真正写进 Drive，再卸载 Drive。
  - `runtime.unassign()`：释放这台 Colab 机器，停止消耗计算单元。
- 适合挂着跑、人不在电脑前的时候。

---

## 附录 A：运行日志逐行对照

在 A100 上运行，日志大致是下面这样。`…` 是要等跑出来才知道的数字。

```
Batch 1102_1530 -> /content/drive/MyDrive/cs604/batches/1102_1530
Device NVIDIA A100-SXM4-80GB (80 GB), bfloat16, 12 CPUs, … GB RAM, … GB free disk
Versions: torch … (CUDA …), timm …
[train] downloaded from Hugging Face in … min
[train] extracting /content/data/hf/train.tar.gz ...
[train] 450000 images ready in … min
train 450,000 images | 12 loader workers

========== 1102_1530_1_linear_proto ==========
linear: head proto, holdout 2/class
Environment: NVIDIA A100-SXM4-80GB (80 GB), bfloat16, 12 CPUs | torch … (CUDA …), timm …
Extracted features for 430,000 train images in … min
Head set from the class prototypes (prototype logit scale …) in … min incl. feature extraction, counted in this run's GPU time
Trainable (excl. head): 0 | head 7,690,000 params (class prototypes (not trained))
GPU time: … GPU-hours (feature extraction + classifier head; the backbone is not trained)
Peak VRAM: … GB allocated, … GB reserved by PyTorch (GPU has 80 GB)
Final holdout: top-1 …%, top-5 …%
Inference latency (batch 1): … ms/img | throughput (batch 256): … img/s | NVIDIA A100-SXM4-80GB, bfloat16, no flip TTA
== 1102_1530_1_linear_proto finished: holdout top-1 …% | class prototypes (not trained) | … GPU-h

========== 1102_1530_2_linear_lp ==========
linear: head lp, holdout 2/class, linear probe 10 epochs (lr 0.001, wd 0.0001, batch 4096)
Environment: …
Reusing the backbone features extracted by the previous run
Head set from a linear probe started from the class prototypes (prototype logit scale …) in … min incl. feature extraction, counted in this run's GPU time
Trainable (excl. head): 0 | head 7,690,000 params (linear probe)
GPU time: … GPU-hours (feature extraction + classifier head; the backbone is not trained)
Peak VRAM: …
Final holdout: top-1 …%, top-5 …%
Inference latency (batch 1): …
== 1102_1530_2_linear_lp finished: holdout top-1 …% | linear probe | … GPU-h
```

最后还会打印 §10 的结果表。

| 日志 | 出自 | 说明，参考值 |
|---|---|---|
| `Batch ... -> ...` | §2 | 批次名和输出目录 |
| `Device ...`、`Versions ...` | §2 | GPU、CPU、内存、磁盘、库版本 |
| `[train] downloaded ...`、`extracting ...`、`images ready` | §3 `prepare_split` | 每个会话只有第一次运行才有。从 Hugging Face 下载一两分钟；用 Drive 备份时没有 `downloaded` 这一行，直接从 Drive 解压，要 20 分钟左右 |
| `train 450,000 images ...` | §3 | 训练集读好了 |
| `========== ..._1_linear_proto ==========` | §8 | 第一个实验开始，从这行起也写进这个实验的 `train_log.txt` |
| `linear: head proto, ...` | §8 | 配置摘要 |
| `Extracted features ...` | §6 `cached_features` | 约 6–7 分钟，只有第一个实验有 |
| `Reusing the backbone features ...` | §6 `cached_features` | 只有第二个实验有 |
| `Head set from ...` | §6 `init_head` | 温度，以及含提特征在内的总用时 |
| `Trainable (excl. head) ...` | §8 | 参数量 |
| `GPU time ...` | §8 | 之前 zero-shot 是 0.11 GPU-h |
| `Peak VRAM ...` | §8 | 之前 zero-shot 是 3.9 GB |
| `Final holdout ...` | §7 `evaluate` | zero-shot 之前是 43.95%，linear probe 是 62.07% |
| `Inference latency ...` | §7 `latency_report` | 之前约 7.55 ms |
| `== ... finished ...` | §8 | 一行总结，写到这行为止 |

注意：`batch_log.txt` 里有上面全部的日志；每个实验自己的 `train_log.txt` 只有从它的 `==========` 标题行到 `finished` 那一行。

---

## 附录 B：术语表

| 术语 | 说人话 |
|---|---|
| 主干（backbone） | 分类头之前的整个 ViT，负责把图片变成 768 维的特征 |
| 分类头（head） | 最后一个线性层，把 768 维特征变成 10,000 个类的分数 |
| 特征（feature） | 主干输出的 768 个数，可以理解成对这张图的“描述” |
| token | ViT 内部处理的单位：每个 16×16 的图块是一个 token，再加一个 CLS token |
| CLS token | 加在 196 个图块前面的一个特殊 token，经过 12 层以后汇总了整张图的信息，用它的输出当特征 |
| L2 归一化 | 向量除以它自己的长度，变成长度为 1 的向量，只保留方向 |
| 余弦相似度 | 两个向量夹角的余弦，方向越接近越接近 1。两个长度为 1 的向量，点积就是余弦相似度 |
| 原型（prototype） | 一个类所有特征的平均方向 |
| 温度（logit scale） | 分数乘上的放大系数，越大 softmax 之后的概率越集中 |
| softmax | 把一组分数变成加起来等于 1 的概率 |
| 交叉熵（cross-entropy） | 衡量预测的概率和正确答案差多少的损失，正确类的概率越高，损失越小 |
| 标签平滑（label smoothing） | 训练目标里正确类不是 100%，留一点概率平分给其他类，防止模型过度自信 |
| AdamW | 常用的优化器：根据梯度更新参数，同时对权重做衰减 |
| 权重衰减（weight decay） | 每一步把权重往 0 拉一点，防止权重过大、过拟合 |
| 余弦退火 | 学习率按余弦曲线，从初始值慢慢降到 0 |
| epoch | 把全部训练数据过一遍 |
| batch | 一次一起处理的一组样本 |
| top-1 / top-5 准确率 | 分数最高的 1 个 / 前 5 个类里包含正确答案的比例 |
| holdout | 从训练集里留出来、不参与训练、只用来测准确率的图片 |
| bf16（bfloat16） | 一种 16 位浮点数，表示范围和 float32 一样大，但精度低一些。计算快、省显存，A100 这类 GPU 原生支持 |
| float16（half） | 另一种 16 位浮点数，这里只用来存特征，省一半空间 |
| TF32 | A100 计算 float32 矩阵乘法时用的一种快速格式 |
| 混合精度（autocast） | 自动让适合的运算用 bf16 算，其余仍用 float32 |
| inference mode | 推理模式：不记录求梯度要用的信息，更快、更省显存 |
| requires_grad | 一个参数要不要算梯度，也就是要不要训练 |
| GPU 时间（GPU-h） | GPU 实际在干活的小时数，项目用它衡量训练开销 |
| 峰值显存 | 运行过程中显存占用的最高值 |
| 推理延迟 | 模型处理一张图要多久（batch 1） |
| 吞吐量 | 批量处理时，每秒能处理多少张图 |
| DataLoader worker | 在后台读图、解码、预处理的子进程 |
| 锁页内存（pinned memory） | 不会被换到硬盘上的内存，从这里拷到 GPU 更快 |

---

## 附录 C：讲解时可能被问到的问题

1. **zero-shot 用了训练图片，还算 zero-shot 吗？** 预训练模型原来的分类头只认 ImageNet 的 1,000 类，没法直接预测这 10,000 个新类别，总得用到一点类别信息。我们没有训练任何参数，只是对特征取平均。我们的 43.95% 和 PDF 里的 43.7% 很接近，说明做法差不多。
2. **温度会影响 zero-shot 的准确率吗？** 不会。所有类的分数乘同一个数，排名不变。温度只影响概率有多集中，它的用处是给后续训练一个好的起点。
3. **linear probe 为什么比 zero-shot 高这么多（43.95% → 62.07%）？** 原型只看每一类自己“平均长什么样”；linear probe 用交叉熵训练，会学着把相似的类分开。
4. **第二个实验复用了特征，为什么 GPU 时间还要算提特征？** 为了如实反映这个方法单独跑要花多少计算。不算的话，linear probe 看起来只要一两分钟，和其他方法比就不公平了。
5. **改 `BATCH_SIZE` 会改变结果吗？** 基本不会。它只决定一次处理多少张图，影响速度和显存，每张图的计算是一样的。会改变结果的是 `HEAD_INIT`、`HOLDOUT_PER_CLASS`、`SEED` 和几个 `LP_*` 参数。
6. **评估为什么不直接用缓存的特征？** 用整条流程重新算一遍，可以顺便检查读图、预处理、主干、分类头这一整套都没出错，结果更可信。
