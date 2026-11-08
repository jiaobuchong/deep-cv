# CS604 期末报告（CVPR 2026 LaTeX 模板）

报告基于 CVPR 2026 官方作者模板（[cvpr-org/author-kit](https://github.com/cvpr-org/author-kit)，tag `CVPR2026-v1(latex)`），用的是正式版（camera-ready）版式：不显示审稿行号，显示作者，带页码。编译好的 PDF 是同目录下的 `main.pdf`（13 页：正文 8 页，参考文献 1 页，附录 4 页）。2026-11-08 全文改成了最直白的英文写法。

## 在 Overleaf 上编译

1. 把整个 `report/` 文件夹压缩成 zip，在 Overleaf 里选 **New Project → Upload Project** 上传。
2. 打开 **Menu**，确认以下两项：
   - **Main document** 是 `main.tex`
   - **Compiler** 是 `pdfLaTeX`
3. 点 **Recompile**。Overleaf 会自动依次运行 pdfLaTeX → BibTeX → pdfLaTeX，生成参考文献。

## 在本机编译

本机已经用 Homebrew 装了 tectonic。改完 `.tex` 文件后，在 `report/` 目录下运行下面这条命令，就会重新生成 `main.pdf`（参考文献会自动处理）：

```
tectonic -X compile main.tex
```

tectonic 用的是 XeLaTeX 引擎，模板默认的 Times 字体在这个引擎下用不了，所以 `main.tex` 里加了一段只在 XeLaTeX 下生效的代码，改用 Times 的克隆字体 TeX Gyre Termes（字宽相同）。用 Overleaf 的 pdfLaTeX 编译时，这段会被自动跳过。

如果装的是完整的 TeX Live / MacTeX，也可以用 pdfLaTeX，依次运行：

```
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

## 需要填写的占位符（红色 `\todo{...}`）

| 位置 | 内容 |
|---|---|
| `main.tex` 作者栏 | 4 位组员的姓名和学号、邮箱 |
| `sec/5_conclusion.tex` 末尾 | 提醒：核对课程的页数要求（CVPR 的上限是正文 8 页，不含参考文献） |

测试集成绩（表 3 的 Test 列）已经填好：全量微调 76.09%，LoRA 71.67%。全部填完后，把 `preamble.tex` 里的 `% \renewcommand{\todo}[1]{#1}` 取消注释，残留的红字就会变回正常颜色。另外记得删掉结论末尾那句页数提醒。

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

- 用 tectonic 编译通过：没有未定义的引用或参考文献，没有超出版面（overfull）的警告，也没有缺字的警告。
- 每一页都渲染成图片看过：图表都在版面内；附录 B、C 的标题和它们的表格在同一页（为此在附录 B 前加了换页）。
- 编译前还用 Python 脚本做过静态检查：环境成对、`\cite` 的 key 都在 `refs.bib` 里、`\ref` 都有对应的 `\label`、图片和 `\input` 的文件都存在。
