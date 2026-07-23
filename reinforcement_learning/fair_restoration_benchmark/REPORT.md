# Fair restoration benchmark: final report

## Protocol

This report contains two locked experiments using the same Phase-3 source split and one fixed PARSeq recognizer. Model selection and action selection used validation only; test was opened only after checkpoints and thresholds were frozen.

- Source split: 3,270 train / 397 validation / 411 test plates, with no source overlap.
- Experiment 1: 6,540 train / 794 validation / 822 test degraded images. Every source has one blur and one auxiliary degradation.
- Experiment 2: 3,270 train / 397 validation / 411 test blur-only images.
- Degradations: Gaussian, motion and defocus blur; Gaussian noise, low light and low contrast.
- Shared selector actions: raw/no-op, mild unsharp, CLAHE, homomorphic filtering, Wiener deconvolution, Richardson-Lucy plus bilateral filtering, CLAHE plus Richardson-Lucy, and adaptive denoising.
- OCR checkpoint SHA-256: `d1dd1c147fa2b67a3e650673e54ade86e6c592f8d85a9169bc01be3b59457b0d`.
- Confidence intervals use 5,000 paired cluster-bootstrap samples with `source_path` as the resampling unit. Exact-match p-values use the exact paired McNemar/binomial test.

`OCR oracle` and `PSNR oracle` use test ground truth to choose an action. They are upper bounds, not deployable methods.

## Experiment 1 — multi-degradation action selection

| Method | Test Exact | Δ Exact vs raw | CER | PSNR | SSIM | Fixed / broken | Exact p |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw / no processing | 76.642% | — | 8.115% | 18.284 | 0.723 | 0 / 0 | — |
| Best global action from validation (CLAHE) | 76.521% | -0.122 pt | 7.877% | 16.496 | 0.775 | 13 / 14 | 1.0000 |
| Contextual Bandit | 76.642% | +0.000 pt | 7.936% | 18.022 | 0.727 | 5 / 5 | 1.0000 |
| PPO two-step | **77.007%** | **+0.365 pt** | **7.713%** | 17.829 | 0.727 | 10 / 7 | 0.6291 |
| OCR oracle upper bound | 80.779% | +4.136 pt | 5.360% | 17.409 | — | 34 / 0 | — |

The 95% CI of PPO's Exact delta is `[-0.608, +1.338]` percentage points, so the observed gain is not statistically conclusive. Its CER delta CI is `[-0.00958, +0.00150]`, which also crosses zero. Bandit selected raw/no-op for 699/822 images (85.0%) and ended with the same Exact as raw.

PPO's Exact by degradation, compared with raw:

| Degradation | Raw | PPO two-step | Difference |
|---|---:|---:|---:|
| Defocus | 45.985% | 45.255% | -0.730 pt |
| Gaussian blur | 83.942% | 86.131% | +2.190 pt |
| Motion blur | 56.204% | 56.934% | +0.730 pt |
| Gaussian noise | 88.321% | 88.321% | 0.000 pt |
| Low contrast | 93.431% | 93.431% | 0.000 pt |
| Low light | 91.971% | 91.971% | 0.000 pt |

## Experiment 2 — blur-only restoration

| Method | Test Exact | Δ Exact vs raw | CER | PSNR | SSIM | Fixed / broken | Exact p |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw blurred image | 62.044% | — | 14.652% | 19.231 | 0.678 | 0 / 0 | — |
| Classical unsharp used by PixelRL benchmark | **65.207%** | **+3.163 pt** | 13.043% | 20.323 | 0.731 | 18 / 5 | **0.0106** |
| Selector action: mild unsharp | 63.017% | +0.973 pt | 14.205% | 19.539 | 0.691 | 4 / 0 | 0.1250 |
| Wiener deconvolution | 63.017% | +0.973 pt | 14.235% | 19.560 | 0.697 | 10 / 6 | 0.4545 |
| Richardson-Lucy + bilateral | 62.774% | +0.730 pt | 14.562% | 19.523 | 0.690 | 5 / 2 | 0.4531 |
| Best global blur action from validation (CLAHE) | 61.800% | -0.243 pt | 13.788% | 15.070 | 0.688 | 12 / 13 | 1.0000 |
| Contextual Bandit | 62.044% | +0.000 pt | 14.205% | 18.620 | 0.683 | 5 / 5 | 1.0000 |
| PPO two-step | 62.774% | +0.730 pt | 13.728% | 18.332 | 0.685 | 10 / 7 | 0.6291 |
| PixelRL / A2C | 63.260% | +1.217 pt | **10.125%** | **21.115** | **0.793** | 39 / 34 | 0.6400 |
| OCR oracle upper bound | 68.856% | +6.813 pt | 9.470% | 17.401 | — | 28 / 0 | — |
| PSNR oracle upper bound | 62.530% | +0.487 pt | 14.294% | 19.768 | — | 6 / 4 | — |

PixelRL gives the strongest image restoration and character-level result: +1.884 dB PSNR, +0.116 SSIM, and a 4.527-point absolute CER reduction from raw. Its CER delta has 95% CI `[-0.06456, -0.02674]`, which excludes zero. Its Exact delta CI is `[-2.920, +5.109]` points and McNemar p is 0.6400, so the Exact improvement is not conclusive.

Classical unsharp has the best deployable Exact result on this test. Its +3.163-point Exact gain is significant here (95% bootstrap CI `[+0.973, +5.353]`, p=0.0106). It does not restore image quality or reduce CER as strongly as PixelRL. The two unsharp rows are intentionally separate: the OpenCV baseline used in the PixelRL evaluator is stronger than the selector action's milder configuration.

## Interpretation

The methods do not optimize the same mechanism:

- Contextual Bandit and PPO are routers. They choose one of eight fixed restoration views. They were trained on the full multi-degradation training set.
- PixelRL is a direct per-pixel A2C restoration model. It was trained only on the paired blur training split, with squared-error/PSNR reward and no OCR reward.
- Consequently, PixelRL's strong PSNR, SSIM and CER improvement does not guarantee an Exact-match gain. It fixed 39 raw OCR failures but also broke 34 raw-correct plates.

The current evidence supports PixelRL for visual/character-level blur restoration, classical unsharp for best Exact on this synthetic blur test, and PPO only as a small, non-significant multi-degradation selector improvement. A multi-seed replication and real-blur holdout are needed before treating the ranking as a production conclusion.

## Checkpoint and resume status

- PixelRL best evaluation checkpoint: `models/pixelrl_seed_42/checkpoints/best_deblur_agent.pt`, epoch 150, validation PSNR gain +1.857 dB.
- PixelRL exact resume checkpoint: `models/pixelrl_seed_42/checkpoints/last_training_state.pt`, epoch 150. It stores model, optimizer, epoch, history and Python/NumPy/Torch/CUDA RNG states.
- The interrupted first segment ended at epoch 61 with a weights-only best checkpoint. Training resumed at epoch 62 with a fresh Adam optimizer; every later epoch used exact-state checkpointing. This optimizer boundary is a limitation and is recorded in `train_summary.json`.
- Bandit checkpoint: epoch 53. PPO checkpoint: epoch 44. Both thresholds were selected on validation only.

## Result artifacts

- `results/multi_summary.csv`: Experiment 1 aggregate metrics.
- `results/multi_by_degradation.csv`: Experiment 1 metrics by degradation.
- `results/multi_test_predictions.csv`: aligned per-image Experiment 1 outputs.
- `results/blur_summary.csv`: Experiment 2 aggregate metrics.
- `results/blur_test_predictions.csv`: aligned per-image Experiment 2 outputs.
- `results/summary.json`: protocol, paired statistics and action distributions.
- `pixelrl_eval/eval_val_predictions.csv` and `pixelrl_eval/eval_test_predictions.csv`: PixelRL validation/test outputs.
