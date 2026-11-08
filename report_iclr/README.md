# CS604 期末报告（ICLR 2026 模板）

这是按课程要求改用 ICLR 2026 模板（`/Users/jaxson/Downloads/iclr2026`）的版本，也是要提交的版本。旧的 CVPR 版本还在 `../report_cvpr/`，只作备份。

编译好的 PDF 是 `main.pdf`，共 15 页：
- 正文：第 1–10 页（第 10 页中间结束），符合"正文不超过 10 页"
- 参考文献：第 10–11 页
- 附录：第 12–15 页（训练日志、全部 26 个 run、参数量表、PEFT 和训练长度表、超参数）

## 编译

- **Overleaf**：把整个文件夹上传，编译器选 pdfLaTeX，主文件 `main.tex`。会自动跑 BibTeX。
- **本地**：`tectonic -X compile main.tex`。XeLaTeX 不支持 `times` 包的字体，所以 `main.tex` 里在 XeTeX 下改用 TeX Gyre Termes（和 Times 字宽相同）；pdfLaTeX 下仍用模板自带的 Times。

## 文件

| 文件 | 内容 |
|---|---|
| `main.tex` | 主文件：模板、作者、页眉、各节的 `\input` |
| `sec/0_abstract.tex` … `sec/5_conclusion.tex` | 正文各节 |
| `sec/X_appendix.tex` | 附录（放在参考文献后面） |
| `figures/fig_pipeline.pdf` | 新画的流程图，蓝色 + "OURS" 标出我们的改进；生成脚本 `../figures/make_pipeline_iclr.py` |
| `figures/data_examples_wide.jpg` | 训练样例图（4 个类，每类 4 张）；生成脚本 `../figures/make_data_examples_wide.py`，需要本地的 `cs604-course-data/train` |
| 其他 `figures/*` | 从 `../report_cvpr/figures/` 复制来的图 |
| `iclr2026_conference.sty/.bst`, `fancyhdr.sty`, `natbib.sty`, `math_commands.tex` | 模板文件，未改动 |

## 对照课程要求

| 要求 | 在哪里 |
|---|---|
| Introduction：问题、动机、我们自己的目标 | §1："Why this is hard"、"Our objectives"（三个问题）、贡献列表；图 1 是训练样例 |
| Related work：参考文献 + 优缺点 | §2，表 1 列出每种方法和头部初始化的优缺点 |
| Methods：方法和我们的改进 | §3 开头 "What we changed, and why"（5 项改进），§3.2 原型头 |
| 方法/流程图，突出我们的改进 | 图 2（蓝色 = 我们的改动，橙色 = 常用的 LP-FT 做法） |
| Results and analysis：表格、可视化样例 | §4：表 2–6、图 3–5；图 1 是训练样例 |
| 做了什么、为什么、学到什么 | 表 2（迭代过程）、§5 "What we learned" |

## 还要填的

- `main.tex` 的作者：`\todo{Member 1}` … 和邮箱（红字）。填完后把 `% \renewcommand{\todo}[1]{#1}` 取消注释。
- 正文第 10 页还有大约半页空余，名字变长一两行也不会超页；改完请再编译确认 "5 Conclusion" 那一节仍在第 10 页结束。

## 和模板不同的地方

- 用了 `\iclrfinalcopy`，这样作者名字会显示出来（课程报告不需要匿名）。
- 模板在 final copy 模式下的页眉是 "Published as a conference paper at ICLR 2026"，这对课程报告不真实，所以在 `main.tex` 里改成了 "CS604 Project Report, Team 06"。如果助教要求原样保留，删掉 `\maketitle` 后面那行 `\lhead{...}` 即可。
- 链接用彩色文字代替模板默认的彩色方框（`\hypersetup{colorlinks=true}`）。
- 附录的表格用 `[H]` 固定在标题下面，附录页用 `\raggedbottom`。
