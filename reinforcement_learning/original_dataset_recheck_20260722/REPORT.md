# Original-dataset REL recheck — 2026-07-22

## Why the previous table was wrong

The prior `76–77%` table evaluated the newly generated multi-degradation benchmark containing 794 validation and 822 test images. It was not the original Phase-3 dataset on which the fine-tuned PARSeq baseline reaches approximately 92% exact accuracy.

This audit reruns the historical artifacts on the original locked split:

- validation: 397 images;
- test: 411 images;
- parent recognizer: `outputs/phase3_controlled_aug_full_frozen_eval/best_phase3_parseq_anpr.pt`;
- Bandit: `outputs/rl_restoration/router_seed_123/best_reward_router.pt`, epoch 20, margin 0.005;
- PPO: `outputs/rl_restoration/ppo_prior_seed_123/best_ppo_restoration_policy.pt`, epoch 22, first/revise margins 0.1/0.0;
- trajectory cache: `outputs/rl_restoration/trajectory_cache`.

OCR was recomputed from the selected images for all validation and test rows. Test policy inference was rerun from the locked checkpoints.

## Fair comparison: the same frozen parent PARSeq

| Method | Val Exact | Test Exact | Val CER | Test CER | Test fixed / broken |
|---|---:|---:|---:|---:|---:|
| Original fine-tuned PARSeq, baseline view | 92.6952% | 91.9708% | 1.3476% | 1.1316% | 0 / 0 |
| Contextual Bandit | 94.4584% | 92.4574% | 1.1026% | 1.0721% | 5 / 3 |
| Two-stage PPO | **94.7103%** | **92.4574%** | **1.0720%** | **1.0721%** | 5 / 3 |

With one fixed recognizer, PPO improves validation over Bandit by one image, but they have identical test correctness. Both improve test Exact by 0.4866 percentage points over the baseline (net +2 correct images). PPO selected different actions from Bandit on some samples, but none changed test correct/incorrect status.

## Separate recognizer fine-tuning

| Pipeline | Val policy Exact | Test policy Exact | Test CER |
|---|---:|---:|---:|
| Bandit + Bandit hard-aware PARSeq | 94.4584% | 92.7007% | 1.0423% |
| PPO + PPO hard-aware PARSeq | 94.7103% | 92.7007% | 1.0423% |

The historical `92.7007%` PPO test result includes a separately fine-tuned hard-aware PARSeq checkpoint. Comparing that number against Bandit's `92.4574%` parent-PARSeq result mixes recognizers. When each policy uses its own hard-aware recognizer, both pipelines reach 92.7007% on this test.

## Rerun artifacts

- `bandit/summary.json`: locked Bandit inference and recomputed parent/hard-aware OCR.
- `bandit/test_locked_policy_selections.csv`: per-image Bandit action trace.
- `ppo2/summary.json`: locked PPO inference and recomputed parent/hard-aware OCR.
- `ppo2/test_locked_ppo_selections.csv`: per-image PPO action trace.
- `summary.csv`: compact protocol-separated results.

The blur-only PixelRL table is unaffected; it uses a distinct paired synthetic-blur benchmark by design.
