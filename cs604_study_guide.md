# CS604 项目代码学习指南

> 用课程材料读懂 `cs604_vit_final.ipynb`。
> 文中链接都是相对于本文件（`deep-project/`）的路径；页码指 PDF 的页码。整理日期：2026-10-18。

## 一句话结论

**重点看 Lesson 5 的第 2 部分（p20–37）**：它基本就是这个项目的理论说明，代码里用到的方法几乎都能在里面找到对应的页。其次是 **Lesson 4 的 ViT 结构**。其余课件大多可以跳过。

## 推荐阅读顺序

- [ ] 1. [`CV-code/pre_trained.py`](../CV-code/pre_trained.py)：先用一个小例子熟悉迁移学习的完整流程
- [ ] 2. Lesson 5 p25–33：几种适配方法的原理
- [ ] 3. [`vit-explained.pdf`](../Session4/vit-explained.pdf) 和 [Attention 交互页](../Session4/Attention_Step_by_Step.html)：ViT 的结构和 Q/K/V
- [ ] 4. 回到 [`cs604_vit_final.ipynb`](cs604_vit_final.ipynb)，按 §4 → §6 → §7 的顺序读；遇到训练细节再翻 Lesson 5 的 p18、p26、p35

---

## ⭐⭐⭐ 必看：Lesson 5 第 2 部分（p20–37）

文件：[`Session5/CS604-Lesson5-Optimization and Adaptation.pdf`](../Session5/CS604-Lesson5-Optimization%20and%20Adaptation.pdf)

| 页码 | 讲什么 | 对应的代码 |
|---|---|---|
| p21–22 | 为什么 ViT 必须从预训练模型开始 | §4 `create_model` 里的 `pretrained=True` |
| p24 | ViT 各部分怎么初始化；迁移时分类头初始化为 0 | §6 分类头初始化（代码改用了 prototype / linear probe，见下文"课件和代码不一样的地方"第 3 条） |
| p25–27 | 四种适配方式；全量微调 + 分层学习率衰减；linear probe 和部分微调 | `METHOD_DEFAULTS` 里 full 的 `layer_decay=0.75`；`linear` / `partial` 方法；§6 用 linear probe 初始化分类头 |
| p28–30 | LoRA 原理、rank/α/学习率怎么设、在 ViT-B 上怎么算出 0.34% | §4 `apply_method` 的 LoRA 分支；`LORA_R=8`、`LORA_ALPHA=16` |
| p31–32 | AdaptFormer：为什么并联在 MLP 旁边、缩放系数 s 起什么作用 | §4 的 `Adapter` 类、`adaptformer_block_forward` |
| p33–34 | 五种方法的对比表、该选哪种 | 报告里的方法对比和动机 |
| p35–36 | 调参对照表（出现什么现象该调哪个参数）、常见坑 | 梯度裁剪 1.0、warmup；LoRA 训完用 `merge_and_unload()` 合并，否则推理会变慢 |

第 1 部分（p3–19）讲优化器的演变（SGD → Momentum → RMSprop → Adam）。代码只用到其中两样：

| 页码 | 讲什么 | 对应的代码 |
|---|---|---|
| p14–15 | Adam | §7 `train()` 里的 AdamW 优化器 |
| p18 | warmup + cosine 学习率调度 | §7 `train()` 里的 `lr_factor` |

## ⭐⭐ 要看：Lesson 4 的 ViT 结构

LoRA 改的是注意力里的 `qkv` 层，AdaptFormer 插在 MLP 旁边，所以必须搞清楚 ViT 的一个 block 长什么样。

- **快速版**：[`Session4/vit-explained.pdf`](../Session4/vit-explained.pdf)，只有 5 页。流程是：切 patch → patch embedding → 位置编码 + CLS → Transformer encoder → 分类头。§6 提取的 768 维特征就是 CLS token 的输出。
- **详细版**：[`Session4/CS604-Lesson4-ViT.pdf`](../Session4/CS604-Lesson4-ViT.pdf) 的 p9–15（patch 和位置编码）和 p30–32（自注意力、多头注意力、整体结构）。
- **理解 Q/K/V**：[`Session4/Attention_Step_by_Step.html`](../Session4/Attention_Step_by_Step.html)，可以交互式地一步步看 Q、K、V 是怎么算出来的。

## ⭐ 打基础（按需看）

| 文件 | 为什么值得看 |
|---|---|
| [`CV-code/pre_trained.py`](../CV-code/pre_trained.py) | **建议最先读。** 267 行、中文注释的 ResNet-18 迁移学习例子，骨架和项目一样：预训练权重 + 配套预处理、数据增强、冻结 backbone 换新头、训练循环、评估、解冻最后一个 stage 并用更小的学习率。读懂它再看约 1,100 行的 notebook 会轻松很多 |
| [`Session2/pytorch/L2_03_pytorch_training_loop.ipynb`](../Session2/pytorch/L2_03_pytorch_training_loop.ipynb) | zero_grad → forward → loss → backward → step，这就是 §7 训练循环的核心 |
| [`Session2/CS604-Lesson2.pdf`](../Session2/CS604-Lesson2.pdf) p30–41 | softmax 和交叉熵。**p41 的 "log K" 检查**很有用：1 万类时，随机分类头的初始 loss 应约等于 ln 10000 ≈ 9.2；而 LoRA 短跑刚开始的 loss 只有 2.2 左右，这就是 §6 分类头初始化的效果 |
| [`Session2/CS604-Lesson2.pdf`](../Session2/CS604-Lesson2.pdf) p4–16 + [`L2_01_kNN_and_distances.ipynb`](../Session2/pytorch/L2_01_kNN_and_distances.ipynb) | §6 的 prototype 分类头，本质是"看特征离哪个类的平均特征最近"，和 kNN 是同一个思路。p13–16 讲"问题出在空间，不在算法"，正好解释了为什么这招在原始像素上不行，在 ViT 特征上却能直接拿到约 44% |
| [`Session3/CS604-Lesson3-CNN.pdf`](../Session3/CS604-Lesson3-CNN.pdf) p62–64 | 迁移学习三步：换头 → 冻结 backbone 只训练头 → 按需解冻微调。项目的流程就是这三步 |
| [`Session1/CS604-Lesson1-Introduction.pdf`](../Session1/CS604-Lesson1-Introduction.pdf) p59–61 | 为什么不能在测试集上调参，也就是代码为什么要从训练集里切出 holdout |

## 可以跳过

- **Session 6（语义分割）、Session 7（目标检测）**：和项目无关。
- **对考试有用、但读代码用不上的**：Lesson 2 Part 2（激活函数、反向传播推导），Lesson 3 的卷积和 BatchNorm 细节（ViT 用的是 LayerNorm），Session 5 的 supplement pptx（内容和 Lesson 5 第 1 部分的优化器重复）。
- **几份 Cheatsheet**：考试复习用。

## ⚠️ 课件和代码不一样的地方

读代码时别被绕晕：

1. **LoRA 加在哪**：课件说加在 q 和 v 上。但 timm 把 q、k、v 合成了一个 `qkv` 层，所以代码其实是把 LoRA 加在整个 qkv 上，k 也一起被调了。参数量恰好一样，都是 294,912。报告里要写 "qkv"，不要写 "q, v"。
2. **AdaptFormer 的设置**：课件里的 s 是可学习的，r=8，约 0.15M 参数。代码按 AdaptFormer 官方实现，s 固定为 0.1，bottleneck 为 64，共 1.19M 参数。
3. **分类头初始化**：课件 p24 说迁移学习时分类头初始化为 0，这是 ViT 论文的做法。代码改成了用 prototype / linear probe 初始化。这是你们方案的设计点，报告里要讲清楚动机，最好配一个消融实验。
4. **分类头的规模**：课件 p33 的表格按 100 类计算。项目有 1 万类，分类头有 7.69M 参数，是 LoRA 参数量的 26 倍。所以在这个项目里，分类头怎么初始化、参数量算不算它，都格外重要。

---

## 附：项目 notebook 速览

### 各个文件的作用

| 文件 | 作用 | 要不要细看 |
|---|---|---|
| [`cs604_vit_final.ipynb`](cs604_vit_final.ipynb) | 最终训练，并生成提交文件 | ✅ 主要看这个，运行和修改都用它 |
| [`cs604_vit_final_zh_backup.ipynb`](cs604_vit_final_zh_backup.ipynb) | final 的中文版 | 可选，对照着读。代码逻辑和英文版完全一样，只有注释和说明是中文；但以后改英文版，它不会跟着更新 |
| [`cs604_vit_adaptation.ipynb`](cs604_vit_adaptation.ipynb) | 短跑实验版 | ❌ 训练代码和 final 一样，只有 `EXPERIMENTS` 不同；跑消融直接改 final 里的 `EXPERIMENTS` 即可 |
| [`cs604_download_to_drive.ipynb`](cs604_download_to_drive.ipynb) | 把数据集下载到 Google Drive | ❌ 跑一次就够了，10/31 测试集发布后再跑一次 |

### final notebook 各节

先读开头的说明文字和 §1，再读 §9 的 `run_experiment` 了解整体流程，最后按重要性读 4 个核心小节。

| 章节 | 内容 | 重要性 | 对应报告的哪部分 |
|---|---|---|---|
| 开头说明 | 短跑结果，以及为什么最终选 full 8 epoch + LoRA 10 epoch | ⭐⭐⭐ 先读 | 实验结果、设计动机 |
| §1 Config | 所有可调设置：跑哪些实验、学习率、LoRA rank 等 | ⭐⭐⭐ 先读 | 实现细节 |
| §9 `run_experiment` | 一个函数串起全流程：建模型 → 初始化分类头 → 插入 PEFT 模块 → 训练 → 评估 → 测延迟 → 写提交文件 | ⭐⭐⭐ 当目录读 | — |
| §4 Model and fine-tuning methods | LoRA / Adapter / AdaptFormer / BitFit 的实现，以及参数统计 | ⭐⭐⭐ 核心 | 方法设计、可训练参数量 |
| §6 Classifier head initialization | 用 prototype 和 linear probe 初始化分类头 | ⭐⭐⭐ 核心 | 设计亮点和动机 |
| §7 Training | 训练循环、学习率调度、超出时间预算时自动压缩 | ⭐⭐⭐ 核心 | 实现细节、GPU-hours |
| §8 Evaluation… | 推理延迟测量、loss 曲线、提交文件 | ⭐⭐⭐ 核心 | 推理延迟、loss 曲线 |
| §5 Dataset and local holdout set | 从训练集里切出 holdout | ⭐ 扫一眼 | 评估方式 |
| §12 Compare results | 生成对比表和对比图 | ⭐ 扫一眼 | 可直接放进报告 |
| §2、§3、§10、§11、§13 | 安装环境、挂载 Drive、解压数据、跑实验队列、自动断开 | 扫一眼 | — |
