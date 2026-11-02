# 单方法 notebook

从主 notebook `cs604_vit_final.ipynb` 拆出来的 7 个 notebook，每个只保留一种微调方法的代码，方便对照着看。

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

7 个 notebook 的结构完全一样，不同的地方只有三处：

1. **第 1 节 Config**：`METHOD` 固定为本方法，`METHOD_DEFAULTS` 只有本方法的一行，`EXPERIMENTS` 默认是 7 个 notebook 统一的对比设置（proto 分类头、3 epoch、每类留 2 张图做 holdout），直接运行就能互相比较。
2. **第 4 节**：`apply_method` 只剩本方法的分支，前面的 markdown 说明了它怎么改模型、参数量怎么算。
3. **第 9 节 `run_experiment`**：只有 LoRA 版本有 `merge_and_unload()`（训练完把 LoRA 合并回原权重）。

其余部分（数据、分类头初始化、训练循环、评估）照搬主 notebook。想看两个方法的区别，直接对比第 4 节就行。

## 注意

- 这些 notebook 能在 Colab 上单独运行，但运行一次就是训练一个新模型。
- Phase 2 的测试集预测只能用主 notebook 的 `PHASE2_TEST` 开关，也就是 Phase 1 的同一个模型。
- 主 notebook 以后如果改了，这里不会自动同步。
