# Single-method notebooks

These 7 notebooks are split out of the main notebook `cs604_vit_final.ipynb`. Each one keeps only the code for one method, so the methods are easy to compare side by side. Each notebook has a line-by-line walkthrough (last column, written in Chinese).

| Notebook | Method | Trainable parameters (head excluded) | Our results | Walkthrough |
|---|---|---|---|---|
| `01_full.ipynb` | Full fine-tuning | 85,798,656 (100%) | holdout 73.38%; final model CodaBench val 76.22% | `01_full_walkthrough.md` |
| `02_partial.ipynb` | Partial fine-tuning (sweeps the last 1, 2, 4 and 6 blocks) | 8.26%, 16.5%, 33.0%, 49.6% for K = 1, 2, 4, 6 | holdout 67.97%, 70.99%, 71.88%, 72.89% | `02_partial_walkthrough.md` |
| `03_lora.ipynb` | LoRA (r = 8 on qkv, proj, fc1, fc2) | 1,179,648 (1.375%) | holdout 71.98%, CodaBench val 71.90% (final PEFT model) | `03_lora_walkthrough.md` |
| `04_adaptformer.ipynb` | AdaptFormer (parallel to the MLP) | 1,189,632 (1.387%) | holdout 71.17% (69.05% earlier with the linear-probe head) | `04_adaptformer_walkthrough.md` |
| `05_adapter.ipynb` | Adapter (in series after the MLP) | 1,189,632 (1.387%) | holdout 71.54% | `05_adapter_walkthrough.md` |
| `06_bitfit.ipynb` | BitFit (trains only the biases) | 102,912 (0.120%) | holdout 69.81% | `06_bitfit_walkthrough.md` |
| `07_linear.ipynb` | Zero-shot / linear probe | 0 | holdout 43.95% / 61.94% | `07_linear_walkthrough.md` |

All holdout numbers in this table use the same comparison setup, so they can be compared directly. See "Results with the shared setup" below for details.

## How to read them

- Suggested order: 07 → 01 → 03 → 04 → 02, 05, 06. The 07 walkthrough covers the basics: reading the data, extracting features, the classifier head and evaluation. The 01 walkthrough explains the training loop in the most detail. The others only explain what differs from 01; for shared code they say "see 01 §x" or "see 07 §x".
- All notebooks have almost the same structure: §1 config, §2 setup, §3 data, §4 model and method, §5 dataset and holdout, §6 classifier head, then training (not in 07), evaluation, the full pipeline, running, and the results table.
- The core of each method is `apply_method` in §4. To see how two methods differ, compare their §4.
- Notebooks 01 to 06 share the same training loop (`train` in §7). In 01, `build_param_groups` applies layer-wise learning-rate decay; 02 can turn it on with `LAYER_DECAY`. LoRA has one extra line in §9, `merge_and_unload()`, which merges LoRA back into the original weights after training.
- 07 does not train the backbone, so it has no training loop. The only difference between zero-shot and the linear probe is in §6.
- Hyperparameters are plain variables in §1. By default, `EXPERIMENTS` runs the comparison setup shared by all 7 notebooks once: prototype head, 3 epochs, 2 images per class held out. So any notebook can be run as is, and the results can be compared. 02 also sweeps the number of trained blocks: K = 1, 2, 4 and 6, where K = 2 is the default.

## Results with the shared setup

02, 04, 05, 06 and 07 ran on a Colab A100 on 2026-11-02 (batches `1102_1942`, `1102_2212`, `1102_2017`, `1102_2023`, `1102_1149`). Full fine-tuning and LoRA come from the main notebook on 19 October. They used the same setup, GPU and software versions, so they can be compared directly. Every run uses the prototype head, 3 epochs and the same 20,000-image holdout.

| Method | Run | Trainable parameters (head excluded) | Holdout top-1 | Training speed (img/s) | GPU-hours | Training memory (GB) | Latency (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| Zero-shot (no training) | `1102_1149_1_linear_proto` | 0 | 43.95% | – | 0.10 | 3.8 | 7.32 |
| Linear probe (head only) | `1102_1149_2_linear_lp` | 0 | 61.94% | – | 0.10 | 3.2 | 6.96 |
| BitFit | `1102_2023_1_bitfit` | 102,912 (0.120%) | 69.81% | 1,407 | 0.39 | 12.7 | 7.33 |
| LoRA | `1019_1319_3_lora` | 1,179,648 (1.375%) | 71.98% | 888 | 0.54 | 19.9 | 7.61 |
| Adapter | `1102_2017_1_adapter` | 1,189,632 (1.387%) | 71.54% | 1,509 | 0.37 | 12.8 | 9.74 |
| AdaptFormer | `1102_2212_1_adaptformer` | 1,189,632 (1.387%) | 71.17% | 1,436 | 0.39 | 12.8 | 10.23 |
| Partial, K = 1 | `1102_1942_1_partial_k1` | 7,089,408 (8.26%) | 67.97% | 1,706 | 0.35 | ≤3.8 | 7.92 |
| Partial, K = 2 | `1102_1942_2_partial_k2` | 14,177,280 (16.5%) | 70.99% | 1,655 | 0.35 | 4.1 | 7.41 |
| Partial, K = 4 | `1102_1942_3_partial_k4` | 28,353,024 (33.0%) | 71.88% | 1,611 | 0.35 | 7.0 | 7.41 |
| Partial, K = 6 | `1102_1942_4_partial_k6` | 42,528,768 (49.6%) | 72.89% | 1,650 | 0.35 | 10.0 | 7.42 |
| Full fine-tuning | `1019_1120_1_full` | 85,798,656 (100%) | 73.38% | 1,139 | 0.48 | 18.9 | 7.78 |

- GPU-hours include extracting features for the classifier head at the start: about 6 minutes in these runs, and about 7 minutes in the two October runs.
- Training memory is the peak memory allocated by PyTorch. For K = 1, the 3.8 GB peak comes from the feature extraction at the start; training itself used less.
- Latency is measured at batch size 1, in bf16, forward pass only. Models with the same architecture still vary by about 0.5 ms (K = 1 to 6 have exactly the same architecture). So in this table, only the extra 2.3–2.9 ms of Adapter and AdaptFormer is a real slowdown.
- K = 4 and 6 use the full fine-tuning recipe (learning rate 1e-4, `LAYER_DECAY` 0.75, `DROP_PATH` 0.1), while K = 1 and 2 use learning rate 3e-4. So the change from K = 2 to K = 4 mixes two factors.
- Each setting ran once, and the holdout varies by about ±0.3 points.

Main findings:

- **Where the parameters go matters more than how many there are.** BitFit trains only 0.12% of the parameters and reaches 69.81%. That is higher than training the last block (8.26%), which reaches 67.97%. LoRA (1.375%) ties training the last 4 blocks (33.0%). Small changes in every layer work better than big changes in the last few layers.
- **Training memory depends on how far the gradients flow back, not on how many parameters are trained.** In partial fine-tuning, each extra trained block adds about 1.5 GB. Extended to 12 blocks, this gives 18.8 GB, almost the same as the 18.9 GB measured for full fine-tuning. BitFit, Adapter and AdaptFormer have trainable parameters in every block, so the gradients must flow all the way back to the first block. Their memory (12.7–12.8 GB) is higher than for K = 6 (10.0 GB), which trains half of the network, but about a third lower than for full fine-tuning. LoRA is the exception: it needs more memory than full fine-tuning (19.9 GB) and trains the slowest.
- **Each extra block in partial fine-tuning helps less.** Training 0, 1, 2, 4, 6 and 12 blocks (0 is the linear probe, 12 is full fine-tuning) gives 61.94%, 67.97%, 70.99%, 71.88%, 72.89% and 73.38%. Training half of the blocks is only 0.5 points below full fine-tuning, and it is the cheapest to train: at most 10 GB of memory and 0.35 GPU-hours.
- **LoRA is the best of the 4 PEFT methods**: LoRA 71.98% > Adapter 71.54% > AdaptFormer 71.17% > BitFit 69.81%. After merging, LoRA also adds no latency. This supports choosing LoRA as the final PEFT model.
- **BitFit** has about 1/11 of LoRA's parameters. LoRA gains 10 points over the linear probe, and BitFit gets 7.9 of them. It adds no inference cost either, but it is 2.2 points below LoRA.
- **Adapter and AdaptFormer compared with LoRA**: their accuracy is 0.44 and 0.81 points lower. They train 1.6–1.7× faster and use 36% less memory. But neither can be merged into the original weights, so inference is 28% and 34% slower.
- **Series and parallel are about the same.** Adapter (in series after the MLP) gets 71.54%, and AdaptFormer (parallel to the MLP) gets 71.17%. The gap of 0.37 is about the size of the ±0.3 noise. The two also use different scales (1.0 and 0.1), so this compares the usual setting of each method, not only the position.
- **The prototype head also helps AdaptFormer.** The October AdaptFormer run (`1018_1239_3_adaptformer`) had the same settings except for the linear-probe (lp) head, and got 69.05%. With the prototype head it gets 71.17% (+2.1). Together with full fine-tuning (+2.5) and LoRA (+2.5), all three methods gain more than 2 points. The lp run even ends with a lower training loss (1.86 vs 1.98). Full fine-tuning shows the same pattern: the lp head memorizes the training set more easily.
- No PEFT method beats full fine-tuning.
- Zero-shot gives exactly the same result as the main notebook in September (43.95%), and the linear probe differs by only 0.13 (61.94% vs 62.07%). So these notebooks agree with the main notebook.

## Extra checks after the deadline: `08_linear_exp.ipynb`

Two cheap checks for the report, not for submission. Both reuse the code of 07 and train no backbone.

- **Direction 1, backbone version**: zero-shot and linear probe with two checkpoints that stop after ImageNet-21k (`augreg_in21k`, `orig_in21k`), compared with our `augreg2_in21k_ft_in1k` (43.95% / 61.94%).
- **Direction 2, test resolution**: the trained run `1019_1120_1_full` evaluated on the same holdout at 224 to 384 px, with interpolated position embeddings. The 224 px result must match its own holdout score (73.38%).

It takes about 30 minutes on an A100. It has no walkthrough.

Results (batch `1105_0141`):

- Backbone version: `augreg_in21k` gets 49.96% zero-shot and 61.05% with a linear probe; `orig_in21k` gets 42.43% and 51.07%. Ours gets 43.95% and 61.94%. The linear probe is the better test of the features, and neither checkpoint beats ours there, so the backbone version does not explain the gap to the top of the leaderboard.
- Test resolution: `1019_1120_1_full` gets 73.38%, 75.21%, **75.50%**, 75.28%, 74.33% and 72.94% at 224, 256, 288, 320, 352 and 384 px. With no extra training, a larger test size helps up to 288 px (+2.13 points).

## Training at 288 px after the deadline: `09_full_288.ipynb`

Full fine-tuning as in 01, but trained and tested with larger images (`dynamic_img_size=True`). It is for the report, and for choosing the test size of `cs604_vit_288.ipynb`. It reads no val or test data and writes no submission files.

- By default it runs only run 1: all 3 epochs at 288 px. Run 2 (2 epochs at 224 px, then 1 at 288 px) is optional and commented out in `EXPERIMENTS`.
- After training, each run is tested on the holdout at 224 to 448 px, with latency and throughput at each size. It is compared with `1019_1120_1_full` at the same size (75.50% at 288 px).
- Estimated cost on an A100: about 0.7 GPU-hours for run 1 (0.55 for run 2) and about 30 GB of memory. The whole notebook takes about 1 hour with run 1 only. It has no walkthrough.
- `cs604_vit_288.ipynb` in the project root trains the final model's recipe at 288 px in all 8 epochs (run `1105_1032_1_full_288`, 1.61 GPU-hours, over the 1.2-hour constraint) and predicts at `TEST_SIZE`. It was prepared for one unranked test submission that the TA allowed, but the submission could not be made, so this model has no test score. Run 1 here (best test size 320 px) chose its `TEST_SIZE`.

## Head learning rate for LoRA after the deadline: `10_lora_head_lr.ipynb`

LoRA as in 03, plus one setting, `HEAD_LR`, the learning rate of the 10,000-class head (the LoRA matrices keep `LR`). It tests why LoRA gets worse from 3 to 6 epochs (71.98% → 71.52%) while full fine-tuning keeps improving. It reads no val or test data and writes no submission files.

- Hypothesis: the head memorizes. It is 7.69M of LoRA's 8.87M trainable parameters, and 03 trains it at LR 1e-3 with weight decay 1e-4 and no drop path. Full fine-tuning trains its head at 1e-4 with weight decay 0.05 and drop path 0.1.
- Run 1, the main test: 6 epochs with `HEAD_LR = 1e-4`, against 71.52% (`1019_1120_2_lora`). Run 2: 3 epochs with `HEAD_LR = 1e-4`, against 71.98% (`1019_1319_3_lora`).
- The results table shows each run next to the earlier run of the same length with the head at 1e-3, and the holdout after every epoch. One line reads run 1 against the hypothesis; differences below 0.3 points count as noise.
- Estimated cost on an A100: about 1.0 GPU-hour for run 1 and 0.55 for run 2. It has no walkthrough; everything except `HEAD_LR` is explained in `03_lora_walkthrough.md`.

## What was removed

Each notebook keeps only the code that its default experiments use. These general features are only in the main notebook:

- resuming after a disconnect, and skipping experiments that already finished
- the time-budget control (training is shortened automatically if it would run over time)
- flip TTA and `torch.compile`
- writing CodaBench submission files (Phase 1 is over, and Phase 2 must use the main notebook)
- lp and random head initialization (01 to 06 use only the prototype head; 07 has the prototype and lp heads)
- the setup with no holdout, which trains on all 45 images per class
- running on a T4 or a CPU (these notebooks need an A100, L4 or H100, which support bf16)

## Notes

- Each notebook can run on its own in Colab, and one run trains one new model. The results are saved to `cs604/runs/` and `cs604/batches/` on Drive.
- You can run different notebooks at the same time in several Colab sessions, but start them at least a minute apart. The batch name is the start time in the form `MMDD_HHMM`. So two notebooks started in the same minute write to the same `batches/` folder, and their summary files `results.md` and `results.json` overwrite each other. Each run's own folder under `runs/` is not affected.
- By default the data is downloaded from Hugging Face (`DATA_FROM_HF = True`). On Colab the download took about 2 minutes, and about 8 minutes in total with unpacking on the local disk. Downloading and unpacking need about 80 GB of free local disk. With `False`, the notebook unpacks the copy on Drive instead, at about 33 MB/s; this took 20.3 minutes. The main notebook `cs604_vit_final.ipynb` only gets the data from Hugging Face and has no Drive option.
- Phase 2 test predictions must use the `PHASE2_TEST` switch of the main notebook, so that they come from the same model as in Phase 1.
- If the main notebook changes later, these notebooks are not updated automatically. When a notebook changes, its walkthrough must be updated too.
