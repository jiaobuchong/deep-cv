# CS604 期末报告（CVPR 2026 LaTeX 模板）

报告基于 CVPR 2026 官方作者模板（[cvpr-org/author-kit](https://github.com/cvpr-org/author-kit)，tag `CVPR2026-v1(latex)`），用的是正式版（camera-ready）版式：不显示审稿行号，显示作者，带页码。本机没有 LaTeX 编译器，所以还没实际编译过；源文件已经用脚本检查过（见文末）。

## 在 Overleaf 上编译

1. 把整个 `report/` 文件夹压缩成 zip，在 Overleaf 里选 **New Project → Upload Project** 上传。
2. 打开 **Menu**，确认以下两项：
   - **Main document** 是 `main.tex`
   - **Compiler** 是 `pdfLaTeX`
3. 点 **Recompile**。Overleaf 会自动依次运行 pdfLaTeX → BibTeX → pdfLaTeX，生成参考文献。

如果在本地编译，依次运行：

```
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

## 需要填写的占位符（红色 `\todo{...}`）

| 位置 | 内容 |
|---|---|
| `main.tex` 作者栏 | 组员姓名和学号（3 个占位，按实际人数增减）、邮箱 |
| `sec/4_experiments.tex` 表 2（Main results）的 Test 列 | Phase 2 测试集成绩：最终全量微调模型和最终 PEFT 模型，共 2 处 `TBD` |
| `sec/4_experiments.tex` 第 4.3 节末尾 | 提醒：补上测试集成绩 |
| `sec/5_conclusion.tex` 末尾 | 提醒：补上测试集成绩；核对课程的页数要求（CVPR 的上限是正文 8 页，不含参考文献） |

全部填完后，把 `preamble.tex` 里的 `% \renewcommand{\todo}[1]{#1}` 取消注释，残留的红字就会变回正常颜色。另外记得删掉那两句提醒。

## 文件

| 文件 | 内容 |
|---|---|
| `main.tex` | 主文件：标题、作者，以及各节的引入 |
| `preamble.tex` | 模板自带的宏（含 `\todo`），另加了 microtype、multirow、`\run`、`\minus` |
| `cvpr.sty`、`ieeenat_fullname.bst` | CVPR 2026 官方样式文件和参考文献样式 |
| `sec/0_abstract.tex` | 摘要 |
| `sec/1_intro.tex` | 1 引言（任务、约束、贡献） |
| `sec/2_related.tex` | 2 相关工作 |
| `sec/3_method.tex` | 3 方法（含框架图，图 1） |
| `sec/4_experiments.tex` | 4 实验：设置、复现 baseline、主结果、消融（分类头初始化、PEFT 设计、训练长度） |
| `sec/5_conclusion.tex` | 5 结论与局限 |
| `sec/X_appendix.tex` | 附录：两个最终模型的训练日志节选（含时间和显存）、全部 17 个实验、超参数表 |
| `refs.bib` | 21 篇参考文献 |
| `figures/fig0_framework.pdf/.png` | 框架图 |
| `figures/fig1_codabench_progression_col.pdf/.png` | CodaBench 提升路线（单栏版，正文用的是这张） |
| `figures/fig1_codabench_progression.png` | 同一张图的宽版（正文没用到，留给 slides） |
| `figures/fig2_head_init_holdout.png` | 分类头初始化对 holdout 的影响 |
| `figures/fig3_loss_lp_vs_proto.png` | lp 与 proto 分类头的训练 loss 对比 |

## 已做的检查

没有编译器，所以用 Python 脚本做了静态检查，结果都通过了：

- `\begin` / `\end` 环境成对
- 所有 `\cite` 的 key 都在 `refs.bib` 里
- 所有 `\ref` / `\cref` 都有对应的 `\label`
- 所有图片和 `\input` 文件都存在
- 正文里没有未转义的 `%`、`&`、`_`，也没有非 ASCII 字符

表格宽度是按字符数估算的，第一次在 Overleaf 编译时，建议看一下有没有 overfull hbox 警告。
