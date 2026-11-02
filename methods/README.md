# 单方法 notebook

从主 notebook `cs604_vit_final.ipynb` 拆出来的 7 个 notebook，每个只保留一种方法用到的代码，方便对照着看。

| Notebook | 方法 | 可训练参数（不含分类头） | 我们的结果 |
|---|---|---|---|
| `01_full.ipynb` | 全量微调 | 85,798,656（100%） | CodaBench val 76.22%（最终模型） |
| `02_partial.ipynb` | 选层微调（最后 2 个 block） | 14,177,280（16.5%） | 没跑过（超过 5%） |
| `03_lora.ipynb` | LoRA（r=8，qkv、proj、fc1、fc2） | 1,179,648（1.375%） | CodaBench val 71.90%（最终 PEFT 模型） |
| `04_adaptformer.ipynb` | AdaptFormer（与 MLP 并联） | 1,189,632（1.387%） | holdout 69.05%（lp 分类头） |
| `05_adapter.ipynb` | Adapter（串在 MLP 后面） | 1,189,632（1.387%） | 没跑过 |
| `06_bitfit.ipynb` | BitFit（只训练 bias） | 102,912（0.120%） | 没跑过 |
| `07_linear.ipynb` | Zero-shot / linear probe | 0 | holdout 43.95% / 62.07% |

## 怎么看

- 结构基本一样：§1 配置，§2 环境，§3 数据，§4 模型和方法，§5 数据集和 holdout，§6 分类头，之后是训练（07 没有）、评估、完整流程、运行、结果表。
- 每个方法的核心是 §4 的 `apply_method`。想看两个方法的区别，直接对比 §4。
- 01 到 06 的训练循环（§7 的 `train`）完全一样。只有 01 的 `build_param_groups` 多了逐层学习率衰减；LoRA 在 §9 多一行 `merge_and_unload()`，训练完把 LoRA 合并回原权重。
- 07 不训练主干，没有训练循环。zero-shot 和 linear probe 的区别都在 §6。07 的逐行讲解在 `07_linear_walkthrough.md`。
- 超参数直接写成 §1 里的变量。`EXPERIMENTS` 默认跑一次 7 个 notebook 统一的对比设置：proto 分类头、3 epoch、每类留 2 张图做 holdout，所以直接运行就能互相比较。

## 删掉了什么

每个 notebook 只保留默认实验用得到的代码。下面这些通用功能只在主 notebook 里：

- 断线后接着训练、跳过已经跑完的实验
- 时间预算控制（预计超时就自动缩短训练）
- 翻转 TTA、`torch.compile`
- 写 CodaBench 提交文件（Phase 1 已经结束，Phase 2 必须用主 notebook）
- lp 和 random 分类头初始化（01 到 06 只用 proto；07 有 proto 和 lp）
- 不留 holdout、每类 45 张全拿来训练的设置
- 在 T4 或 CPU 上运行（这里要求支持 bf16 的 A100、L4 或 H100）

## 注意

- 这些 notebook 能在 Colab 上单独运行，运行一次就是训练一个新模型。结果在 Drive 的 `cs604/runs/` 和 `cs604/batches/` 下。
- Phase 2 的测试集预测只能用主 notebook 的 `PHASE2_TEST` 开关，也就是 Phase 1 的同一个模型。
- 主 notebook 以后如果改了，这里不会自动同步。
