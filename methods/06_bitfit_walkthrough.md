# 06_bitfit 逐行讲解

对应文件：`methods/06_bitfit.ipynb`（精简版）。

06 和 01（全量微调）的代码几乎一样，只有 7 处不同：§1 的配置、§2 的 `RUN_KEYS` 和导入、`create_model`、`apply_method`、`build_param_groups`、`save_state` 的说明，以及 `run_experiment` 里记录配置的那行日志。这 7 处逐行讲。其余代码和 01 完全一样，照样贴出来，只简单说明，细节见 `01_full_walkthrough.md`（写作“见 01 §x”）或 `07_linear_walkthrough.md`（写作“见 07 §x”）的对应小节。末尾有日志对照、新术语和讲解时可能被问到的问题。

## 0. 先看全局

### BitFit 是什么

- 神经网络里大部分层都是 `输出 = 权重矩阵 · 输入 + 偏置`。权重矩阵很大（比如 768 × 2304），偏置只是一个加上去的小向量（2304 个数）。
- BitFit 把所有权重矩阵冻结，只训练各层的偏置（bias），外加新的 10,000 类分类头。
- 主干里一共 74 个偏置向量，合计 102,912 个参数，只占主干的 **0.120%**，在会训练主干的方法里是最少的（LoRA 是 1.375%；07 的 zero-shot 和 linear probe 完全不训练主干）。分类头另有 7,690,000 个参数。
- 偏置本来就在每一层里，训练完模型结构完全没变，推理没有任何额外开销。
- 出处：Ben Zaken、Ravfogel 和 Goldberg 的论文（ACL 2022），原本是给 BERT 这类语言模型提的。它是项目 PDF 列出的可选方向之一，课程的 slides 里没讲。

### 和 01 比

| | 01_full | 06_bitfit |
|---|---|---|
| 训练哪些参数 | 全部：主干 85,798,656 个 + 分类头 | 只有偏置 102,912 个 + 分类头 |
| 学习率 | 1e-4，逐层衰减 | 1e-3，所有参数一样 |
| 权重衰减 | 0.05（偏置、LayerNorm 等不衰减） | 0，全都不衰减 |
| drop path | 0.1 | 不用 |
| 分类头初始化、数据增强、训练循环、评估 | 一样 | 一样 |
| 推理延迟 | 原来的 ViT-B/16 | 一样，结构没变 |
| `final.pt` | 约 374 MB | 约 31 MB |

### 一次实验的流程（§9 `run_experiment`）

```
建模型（create_model，不开 drop path）
  → 切分数据，准备训练和评估两套预处理（make_ctx）
  → 用类原型初始化分类头（init_head）
  → 冻结全部参数，再放开所有偏置和分类头（apply_method）
  → 训练 3 个 epoch，每个 epoch 结束在 holdout 上测一次（train）
  → 保存训练过的参数和训练记录
  → 最后在 holdout 上评估，画训练曲线，测推理延迟
  → 写 summary.json
```

### 几个关键数字

- 和其他方法用同一套对比设置：proto 分类头、3 个 epoch、每类留 2 张图做 holdout。
- 每个 epoch 1,679 步，3 个 epoch 共 5,037 步，前 503 步是 warmup（算法见 01 §0 和 01 §7.4）。
- 学习率（所有参数一样）：第 0 步 2.0e-6，到第 502 步升到 1e-3；第 1,679 步（第 1 个 epoch 结束）8.4e-4，第 2,770 步 5.0e-4，第 3,358 步（第 2 个 epoch 结束）3.0e-4，最后降到接近 0。
- 还没跑过。同样设置的参考：全量微调 73.38%，LoRA 71.98%。BitFit 训练前的起点就是 zero-shot 的类原型分类头，约 44%（之前测的是 43.95%）。

---

## 1. Config（§1）

### 1.1 要跑的实验

```python
# ===== Experiments to run: each dict is one run and lists only the settings that differ from the defaults below =====
EXPERIMENTS = [
    {},  # the defaults; not run yet
]
```

和 01 一样（见 01 §1.1）：`{}` 表示按下面的默认值跑一次，也就是统一的对比设置。注释说明这个方法还没跑过。

### 1.2 默认值

```python
# ===== Defaults: the comparison protocol shared by all method notebooks (prototype head, 3 epochs, 2 holdout images per class) =====
METHOD = "bitfit"
MODEL_NAME = "vit_base_patch16_224.augreg2_in21k_ft_in1k"  # timm name of the base model in the doc (IN21K→IN1K)
HOLDOUT_PER_CLASS = 2    # images per class held out from train for local evaluation (the official val set has no labels)
EPOCHS = 3
WARMUP_EPOCHS = 0.3      # the LR rises linearly for this many epochs, then follows a cosine down to 0
BATCH_SIZE = None        # images per step; None = pick based on VRAM (256 on a 40 or 80 GB GPU)
EFFECTIVE_BATCH = 256    # if one step can't fit it, gradients are accumulated over several steps to reach it
LR = 1e-3
WEIGHT_DECAY = 0.0       # biases get no weight decay anyway, and the head is trained without it too
LABEL_SMOOTHING = 0.1
RRC_SCALE = (0.25, 1.0)  # area range for RandomResizedCrop (training augmentation)
LOG_EVERY = 100          # write a log line every N optimizer steps
SEED = 0
NUM_CLASSES = 10000
```

大部分和 01 一样（各个变量的意思见 01 §1.2）。不同的地方：

| 变量 | 01 | 06 | 说明 |
|---|---|---|---|
| `METHOD` | `"full"` | `"bitfit"` | 方法名，用在实验名、日志和 summary 里 |
| `LR` | `1e-4` | `1e-3` | 学习率大 10 倍，而且所有参数一样，没有逐层衰减 |
| `WEIGHT_DECAY` | `0.05` | `0.0` | 完全不做权重衰减 |
| `LAYER_DECAY` | `0.75` | 没有 | 不做逐层学习率衰减 |
| `DROP_PATH` | `0.1` | 没有 | 不用 drop path |

- **`LR = 1e-3` 为什么比全量微调大**：全量微调要小心不能把预训练的权重冲坏，所以只用 1e-4，靠前的层还更小。BitFit 的权重矩阵全部冻结，根本改不动，能动的只有偏置和新的分类头。参数少、每个参数的作用又有限，步子太小会学得很慢，所以用 1e-3。这和 LoRA、Adapter 的默认学习率一样。
- **`WEIGHT_DECAY = 0.0` 和它的注释**：注释的意思是“偏置本来就不做权重衰减，分类头这里也不衰减”。按 §7.1 的规则，一维的参数（所有偏置）本来就不衰减；剩下唯一一个会被衰减的是分类头的权重。设成 0 以后，整个模型完全不做权重衰减。这是沿用主 notebook 里 BitFit 的默认值。
- **没有 `LAYER_DECAY`**：逐层衰减是为了保护靠前层的预训练权重。BitFit 的权重本来就冻结了，所有偏置用同一个学习率。
- **没有 `DROP_PATH`**：主 notebook 里只有全量微调用 drop path。BitFit 能训练的参数很少，过拟合的风险小得多，一般不需要这么强的正则化。

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
            "LR", "WEIGHT_DECAY", "LABEL_SMOOTHING", "RRC_SCALE", "LOG_EVERY", "SEED", "NUM_CLASSES")
BASE_CONFIG = {k: globals()[k] for k in RUN_KEYS}
```

机制和 01 一样（见 07 §2.2）。列表比 01 少了 `LAYER_DECAY` 和 `DROP_PATH`，因为这两个设置在 06 里不存在。

### 2.3 导入库

```python
import os, gc, json, math, time, random, shutil, subprocess
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

和 01 相比只少了 `re`。01 的 `build_param_groups` 要用正则表达式从参数名里读出 block 编号来做逐层衰减；06 不做逐层衰减（§7.1），用不到它。其余见 01 §2.3 和 07 §2.3。

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

和 01 完全一样：挂载 Drive，检查 GPU，起批次名，定义日志函数，记录运行环境。逐行解释见 07 §2.4–§2.8。

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

## 4. 模型（§4）

### 4.1 `is_head` 和 `resolve_config`

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

和 01 完全一样：`is_head` 判断参数是不是分类头的（见 07 §4.1）；`resolve_config` 合成这次实验的配置，按显存选 `BATCH_SIZE`，算梯度累积步数（见 01 §4.2）。

### 4.2 `create_model`：不开 drop path

```python
def create_model(cfg):
    """Pretrained ViT-B/16 from timm; its 1000-class ImageNet head is replaced by a new, randomly initialized 10,000-class head"""
    return timm.create_model(cfg.MODEL_NAME, pretrained=True, num_classes=cfg.NUM_CLASSES).to(DEVICE)
```

- 和 07 的写法一样（参数的意思见 07 §4.3）：加载预训练的 ViT-B/16，把原来 1000 类的分类头换成新的 10,000 类分类头，搬到 GPU 上。
- 和 01 的区别：没有 `drop_path_rate=cfg.DROP_PATH`。不传这个参数，timm 默认是 0，也就是不开 drop path，训练时每个 block 都正常计算。原因见 §1.2。

### 4.3 `apply_method`：只放开偏置和分类头

```python
def apply_method(net, cfg):
    """BitFit: freeze the backbone, then train only the bias terms and the head"""
    net.requires_grad_(False)
    for n, p in net.named_parameters():
        if n.endswith(".bias"):
            p.requires_grad_(True)
    net.head.requires_grad_(True)
    return net
```

这是 BitFit 和其他方法唯一真正不同的地方，逐行看：

- `net.requires_grad_(False)`：先把所有参数（包括分类头）都设成“不需要梯度”，全部冻结。
- `for n, p in net.named_parameters():`：遍历所有参数，`n` 是名字，`p` 是参数本身。
- `if n.endswith(".bias"):`：名字以 `.bias` 结尾的就是偏置。`endswith` 判断字符串是不是以某段文字结尾。
- `p.requires_grad_(True)`：把这个偏置重新设成“需要梯度”，也就是放开训练。
- `net.head.requires_grad_(True)`：分类头整个放开，权重和偏置都训练。分类头是新的，必须训练。`head.bias` 名字也以 `.bias` 结尾，在上面的循环里已经放开过一次，这里再设一遍没有影响。
- `return net`：返回同一个模型。模型结构完全没变，只是改了哪些参数能训练。
- `cfg` 参数没用到，只是为了和其他 notebook 的写法保持一致。

放开训练的参数（在真实的 ViT-B/16 上数过）：

| 位置 | 参数名 | 个数 |
|---|---|---|
| 每个 block | `norm1.bias`（第一个 LayerNorm） | 768 |
| 每个 block | `attn.qkv.bias`（算 Q、K、V 的线性层） | 2,304 |
| 每个 block | `attn.proj.bias`（注意力输出的线性层） | 768 |
| 每个 block | `norm2.bias`（第二个 LayerNorm） | 768 |
| 每个 block | `mlp.fc1.bias`（MLP 第一层，768 → 3072） | 3,072 |
| 每个 block | `mlp.fc2.bias`（MLP 第二层，3072 → 768） | 768 |
| 每个 block 合计 | 6 个偏置 | 8,448 |
| 12 个 block | 72 个偏置 | 101,376 |
| 主干开头 | `patch_embed.proj.bias`（图块嵌入卷积的偏置） | 768 |
| 主干末尾 | `norm.bias`（最后一个 LayerNorm） | 768 |
| **主干合计** | **74 个偏置** | **102,912（主干的 0.120%）** |
| 分类头 | `head.weight`、`head.bias` | 7,690,000 |
| **全部可训练** | 76 个张量 | 7,792,912 |

几点说明：

- 每个线性层的偏置长度等于它的输出维度：qkv 层输出 3 × 768 = 2,304，fc1 输出 3,072，其余输出 768。
- LayerNorm 有两组参数：缩放用的 γ（名字是 `weight`）和平移用的 β（名字是 `bias`）。BitFit 只训练 β，25 个 γ（每个 block 两个，加最后一个）保持冻结。
- 权重矩阵、`cls_token` 和 `pos_embed` 都不训练。
- 偏置能起什么作用：它让每一层的输出整体平移一点。74 个偏置分布在网络的每一层，从图块嵌入一直到最后的 LayerNorm，合起来可以把每一层的特征往适合新任务的方向挪一挪。至于在这个任务上效果有多好，要跑了才知道。

### 4.4 `param_report`

```python
def param_report(net, ctx):
    """Trainable parameter counts. The 7.69M head is counted separately, as in the doc's LoRA reference (0.34%)"""
    n_train = sum(p.numel() for n, p in net.named_parameters() if p.requires_grad and not is_head(n))
    return {"backbone_params": ctx.n_backbone, "head_params": ctx.n_head, "head_mode": "fine-tuned",
            "trainable_excl_head": n_train, "trainable_pct": 100 * n_train / ctx.n_backbone,
            "trainable_incl_head": n_train + ctx.n_head}
```

代码和 01 一样（逐行解释见 01 §4.5），算出来的数不同：

| key | 值 | 意思 |
|---|---|---|
| `backbone_params` | 85,798,656 | 主干的参数量 |
| `head_params` | 7,690,000 | 分类头的参数量 |
| `head_mode` | `fine-tuned` | 分类头参与训练 |
| `trainable_excl_head` | 102,912 | 可训练参数，不含分类头：所有偏置 |
| `trainable_pct` | 0.120 | 占主干的百分比 |
| `trainable_incl_head` | 7,792,912 | 可训练参数，含分类头 |

按项目文档里 LoRA 参考的算法，分类头不算在内，所以 BitFit 是 0.120%，远低于 5%。

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

和 01 一样，见 07 §5.1–§5.3：每类留 2 张做 holdout（种子固定，每次都是同一批图），按编号取图并做预处理，缩放用双三次插值。

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

和 01 一样，见 01 §5.2：训练用随机裁剪加翻转（数据增强），评估用缩放加中心裁剪，归一化放到 GPU 上做，顺便统计主干和分类头的参数量。BitFit 能训练的参数很少，但数据增强照样有用：每个 epoch 看到的是不同的裁剪。

### 5.3 `Loader` 和 `make_loader`

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

和 01 一样：`Loader` 见 07 §5.5；`make_loader` 的训练和评估两种读法见 01 §5.4（训练每批 256 张、打乱、丢掉最后不满的一批；评估每批 512 张、按顺序）。

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

和 01 一样，见 07 §5.7–§5.9 和 01 §5.5：拷到 GPU 并归一化，bf16 混合精度，清理显存并把峰值显存的记录清零。

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

整个 §6 和 01 完全一样：

- `Clock`：GPU 时间计时器，把设置分类头、每个 epoch 的训练和 holdout 评估加起来，见 01 §6.1。
- `extract_features`：用还没训练的主干给 430,000 张训练图提特征，见 07 §6.1 和 01 §6.2。
- `class_prototypes`、`fit_logit_scale`：算每一类的原型，挑温度，见 07 §6.2–§6.3。
- `init_head`：权重 = 原型 × 温度 ÷ 特征的平均长度，偏置清零，见 01 §6.4。

对 BitFit 来说，分类头的起点可能比全量微调更重要：主干的权重全部冻结，只能靠偏置把特征挪动一点，很大一部分分类的工作要靠分类头完成。用类原型起步，模型一开始就有 zero-shot 的水平（约 44%）。这是推测，跑出来才能确认。

---

## 7. 训练（§7）

### 7.1 `build_param_groups`：所有参数同一个学习率

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

作用和 01 一样：把要训练的参数分组交给优化器。但 01 每层的学习率不同（共 28 组），这里简单得多：所有参数用同一个学习率，只按“做不做权重衰减”分成两组。LoRA、Adapter、AdaptFormer、选层微调的 notebook 用的也是这个版本。

- `decay, no_decay = [], []`：两个列表，分别放“要做权重衰减”和“不做权重衰减”的参数。
- `for p in net.parameters():`：遍历所有参数。这里不需要参数名，所以用 `parameters()`，不用 `named_parameters()`。
- `if p.requires_grad:`：只管要训练的参数，冻结的跳过。
- `(no_decay if p.ndim <= 1 else decay).append(p)`：括号里先用条件表达式选出一个列表，再往里面加。一维的参数（偏置、LayerNorm 的参数）放进 `no_decay`，其余放进 `decay`。理由见 01 §7.1。
- `return [...]`：列表推导式，生成优化器要的分组：
  - `for ps, wd in ((decay, cfg.WEIGHT_DECAY), (no_decay, 0.0))`：两组，`decay` 组的权重衰减是 `cfg.WEIGHT_DECAY`，`no_decay` 组是 0。
  - `if ps`：空的组不要。比如某个方法没有二维的可训练参数，`decay` 组就是空的。
  - 每一组是 `{"params": 参数, "lr": cfg.LR, "weight_decay": 衰减, "base_lr": cfg.LR}`。`base_lr` 留着原始的学习率，训练时每一步按 warmup / 余弦系数重新设 `lr`（见 01 §7.4）。

在 BitFit 里，具体分成这样两组（在真实的 ViT-B/16 上数过）：

| 组 | 里面有什么 | 张量数 | 参数个数 | 学习率 | 权重衰减 |
|---|---|---|---|---|---|
| `decay` | 只有分类头的权重 `head.weight`（唯一一个二维的可训练参数） | 1 | 7,680,000 | 1e-3 | `WEIGHT_DECAY` = 0 |
| `no_decay` | 74 个偏置 + 分类头的偏置 `head.bias` | 75 | 112,912 | 1e-3 | 0 |

因为 `WEIGHT_DECAY` 设成了 0，两组的权重衰减都是 0：BitFit 完全不做权重衰减，这就是 §1.2 那行注释的意思。

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

和 01 一样，见 07 §7.1 和 01 §7.2：在 holdout 上算 top-1 和 top-5。每个 epoch 末用一次（算进 GPU 时间），训练完再用一次（不算）。

### 7.3 `save_state`：只存训练过的参数

```python
def save_state(net, path, **extra):
    """Save only the trained params: the biases and the head, about 31 MB"""
    state = {n: p.detach().cpu() for n, p in net.named_parameters() if p.requires_grad or is_head(n)}
    torch.save({"state": state, **extra}, path)
```

- 代码和 01 一样（逐行解释见 07 §7.2），只有 docstring 不同，因为保存下来的东西不同。
- `if p.requires_grad or is_head(n)`：BitFit 里需要梯度的是 74 个偏置和分类头，一共 76 个张量。冻结的权重矩阵没变过，不用存，以后要用时重新加载预训练权重，再把这些参数填进去就行。
- 大小：7,792,912 个 float32，每个 4 字节，约 31 MB。全量微调要存约 374 MB。

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

代码和 01 一模一样，逐行解释见 01 §7.4。同样的代码放在 BitFit 上，效果有几处不同：

- `params = [...]`：只有 76 个张量（偏置和分类头），梯度裁剪只看它们。
- `optimizer = torch.optim.AdamW(...)`：只管 7,792,912 个参数，AdamW 给每个参数记的两个状态加起来也很小。全量微调要管 93,488,656 个。
- 学习率：日志里的 `lr` 是 `cfg.LR * lr_factor(step)`，最大值是 1e-3，比 01 大 10 倍。比如第 100 步是 2.01e-04，第 200 步是 4.00e-04。
- `model.train()`：BitFit 没开 drop path，这个 ViT 里 dropout 的概率也都是 0，所以训练模式和评估模式算出来的东西一样。保留这一行是因为训练循环对所有方法通用。
- `(loss / cfg.ACCUM_STEPS).backward()`：梯度还是要从损失一路传回到第一层，因为最前面的 `patch_embed.proj.bias` 也要训练。但那些冻结的大权重矩阵不用算梯度，所以可能比全量微调快一些、省一些显存。在 GPU 上还没测过；本机 CPU 上用小模型（vit_tiny）做冒烟测试时，BitFit 的训练速度大约是全量微调的 1.4 倍。

---

## 8. 训练曲线、推理延迟（§8）

### 8.1 颜色、坐标轴样式和训练曲线

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
```

和 01 一样，见 01 §8.1–§8.2：左图是训练损失，右图是每个 epoch 末的 holdout top-1，存成 `curves.png`。图的标题会带上方法名 `bitfit`。

### 8.2 推理延迟

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

和 01 一样，见 07 §7.3–§7.4。BitFit 只改了偏置的值，模型结构和计算量完全没变，所以推理延迟应该和原来的 ViT-B/16 一样（之前全量微调测出来是 7.78 ms）。这是 BitFit 和 Adapter 这类方法的区别：Adapter 会多出额外的模块，推理变慢。

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
        log(f"{METHOD}: lr {cfg.LR}, wd {cfg.WEIGHT_DECAY}, epochs {cfg.EPOCHS}, "
            f"holdout {cfg.HOLDOUT_PER_CLASS}/class, prototype head")
        log(f"Environment: {GPU_NAME} ({VRAM_GB} GB), bfloat16, {os.cpu_count()} CPUs | {VERSIONS}")
        random.seed(cfg.SEED)
        np.random.seed(cfg.SEED)
        torch.manual_seed(cfg.SEED)
        free_gpu()
```

和 01 §9.1 一样，只有记录配置的那行 `log` 不同：01 会写上逐层衰减和 drop path，BitFit 没有这两个设置，所以只记学习率、权重衰减、epoch 数、holdout 数量和分类头的初始化方式。实际日志是 `bitfit: lr 0.001, wd 0.0, epochs 3, holdout 2/class, prototype head`。

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

和 01 §9.2 一样。顺序很重要：先用还没改过的预训练主干算类原型、初始化分类头，再调用 `apply_method` 冻结权重、放开偏置。`.to(DEVICE)` 对 BitFit 什么也不改，因为没有新模块。日志会写 `Trainable (excl. head): 102,912 = 0.120% of backbone | head 7,690,000 params (fine-tuned)`。

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

和 01 §9.3 一样：训练 3 个 epoch，存 `final.pt`（这里约 31 MB）和 `history.json`，记录 GPU 时间和峰值显存。BitFit 的 GPU 时间和显存还没测过，跑完后可以拿来和全量微调（0.48 GPU-h、18.9 GB）比。

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

和 01 §9.4 一样：训练完再测一次 holdout（多给出 top-5），画训练曲线，测推理延迟，汇总写进 `summary.json`。字段的意思见 07 §8.3 和 01 §9.4。BitFit 的 `trainable_excl_head` 是 102,912，`trainable_pct` 是 0.120，`trainable_incl_head` 是 7,792,912。

---

## 10. 运行实验（§10）

```python
RESULTS = []
for i, exp in enumerate(EXPERIMENTS, 1):
    RESULTS.append(run_experiment(dict(exp), f"{BATCH_NAME}_{i}_{METHOD}"))
```

和 01 一样，见 01 §10。实验名比如 `1102_1530_1_bitfit`。

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

和 01 一样，见 01 §11 和 07 §10。表里的“可训练参数”一格会显示 `102,912 (0.120%)`。

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

========== 1102_1530_1_bitfit ==========
bitfit: lr 0.001, wd 0.0, epochs 3, holdout 2/class, prototype head
Environment: NVIDIA A100-SXM4-80GB (80 GB), bfloat16, 12 CPUs | torch … (CUDA …), timm …
Head initialized from the class prototypes of 430,000 train images (logit scale …) in … min, counted in this run's GPU time
Trainable (excl. head): 102,912 = 0.120% of backbone | head 7,690,000 params (fine-tuned)
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
== 1102_1530_1_bitfit finished: holdout top-1 …% | trainable 102,912 (0.120%) | … GPU-h
```

最后还会打印 §11 的结果表。

| 日志 | 出自 | 说明，参考值 |
|---|---|---|
| `Batch ...`、`Device ...`、`Versions ...` | §2 | 批次名、运行环境 |
| `[train] downloaded ...`、`extracting ...`、`images ready` | §3 | 每个会话只有第一次运行才有。从 Hugging Face 下载一两分钟；用 Drive 备份时没有 `downloaded` 这一行，直接从 Drive 解压，要 20 分钟左右 |
| `========== ..._1_bitfit ==========` | §9 | 实验开始，从这行起也写进 `train_log.txt` |
| `bitfit: lr 0.001, wd 0.0, ...` | §9 | 配置摘要，没有逐层衰减和 drop path |
| `Head initialized from ...` | §6 `init_head` | 大约 6–7 分钟，主要是提特征 |
| `Trainable (excl. head): 102,912 = 0.120% ...` | §9 | 只训练偏置，加上分类头 |
| `Training 3 epochs x 1679 steps ...` | §7 `train` | 和其他方法一样的步数 |
| `ep 1/3 step 100/5037 ...` | §7 | 每 100 步一行，学习率最大到 1e-3 |
| `Speed ... img/s, ... min/epoch` | §7 | 第 120 步写一次。全量微调的最终模型是 1,146 img/s，BitFit 可能更快 |
| `== epoch N: ...` | §7 | 每个 epoch 一行：平均训练损失、holdout top-1 |
| `Training done: 5037 steps, ...` | §9 | 全量微调是 0.48 GPU-h，可以拿来比 |
| `Peak VRAM ...` | §9 | 全量微调是 18.9 GB，可以拿来比 |
| `Final holdout ...` | §9 | 还没跑过；参考：全量微调 73.38%，LoRA 71.98%，zero-shot 43.95% |
| `Inference latency ...` | §8.2 | 应该和原来的 ViT-B/16 一样，全量微调是 7.78 ms |
| `== ... finished ...` | §9 | 一行总结 |

怎么看训练是否正常，和 01 一样（见 01 附录 A）：损失整体往下走，每个 epoch 末的 holdout top-1 一次比一次高。BitFit 的学习率比全量微调大，warmup 那段损失可能起伏更明显。

---

## 附录 B：新术语

通用的术语见 07 附录 B，训练相关的见 01 附录 B。下面是 BitFit 特有的：

| 术语 | 说人话 |
|---|---|
| 权重矩阵（weight） | 线性层 `输出 = W · 输入 + b` 里的 W，参数的大头，BitFit 里全部冻结 |
| 偏置（bias） | 上式里的 b，一个加到输出上的向量，长度等于输出维度。BitFit 只训练它 |
| LayerNorm 的 γ 和 β | LayerNorm 先把每个 token 的 768 个数标准化，再乘 γ（名字叫 `weight`）、加 β（名字叫 `bias`）。BitFit 训练 β，不训练 γ |
| 冻结（freeze） | 把参数设成不需要梯度（`requires_grad = False`），训练时不更新 |
| BitFit | Bias-terms Fine-tuning：只微调偏置的方法，Ben Zaken 等人 2022 年提出 |
| PEFT（参数高效微调） | 只训练一小部分参数的微调方法统称。BitFit、LoRA、Adapter、AdaptFormer 都是 |

---

## 附录 C：讲解时可能被问到的问题

1. **只训练偏置能有用吗？** 偏置能让每一层的输出整体平移。74 个偏置分布在网络的每一层，合起来可以把特征往适合新任务的方向挪；再加上完整训练的分类头。BitFit 的论文在语言模型上发现它的效果有竞争力；在这个图像任务上效果如何，正是要跑这个实验来回答的。
2. **BitFit 和 LoRA 怎么比？** 两者推理都没有额外开销：BitFit 本来就不加新东西，LoRA 训练完合并回原权重。BitFit 的可训练参数更少（0.120% 对 1.375%），但它只能平移各层的输出，LoRA 能改变权重矩阵（虽然是低秩的改动）。所以预计 BitFit 的准确率会低于 LoRA（71.98%）；这是推测，要看实验结果。
3. **为什么学习率是 1e-3，不是全量微调的 1e-4？** 权重矩阵全部冻结，能动的只有偏置和新的分类头，参数少、每个参数的作用有限，需要更大的步子。不用担心冲坏预训练的权重，因为权重根本不动。
4. **为什么权重衰减设成 0？** 偏置本来就不做权重衰减（§7.1 的规则），剩下唯一会被衰减的是分类头的权重。这里沿用主 notebook 的默认值，让分类头也不衰减，整个模型就完全没有权重衰减。
5. **训练会比全量微调快、省显存吗？** 很可能会，但没在 GPU 上测过。冻结的大权重矩阵不用算梯度，优化器也只管 779 万个参数；但梯度仍然要一路传回第一层，因为最前面的偏置也要训练。本机 CPU 上用小模型做冒烟测试时，速度大约是全量微调的 1.4 倍。
6. **满足项目“训练少于 5% 参数”的要求吗？** 满足。按项目文档 LoRA 参考的算法（不含分类头），BitFit 是 0.120%，在会训练主干的方法里最少。含分类头是 7,792,912 个，约占主干的 9.08%；所有方法都是同样的算法，分类头是每个方法都必须训练的新层。
7. **推理有额外开销吗？** 没有。模型结构和原来的 ViT-B/16 完全一样，只是偏置的值变了。
8. **课程里讲过 BitFit 吗？** 课程 slides 里没讲，它是项目 PDF 列出的可选方向之一。
