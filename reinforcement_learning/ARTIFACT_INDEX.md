# Chỉ mục artefact

Các file lớn không được nhân đôi vào `reinforcement_learning`. Đường dẫn dưới đây là nguồn chuẩn trong repository.

## Phase 1

- Code: `preprocessing_best_config/benchmark_multiscale_tta.py`.
- Toàn bộ 65-view predictions: `outputs/testing/preprocessing_multiscale_tta_benchmark/predictions_*_all_views.csv`.
- Locked consensus: `outputs/testing/preprocessing_multiscale_tta_benchmark/predictions_*_locked_consensus.csv`.
- Phân tích 21 ảnh hard: `outputs/testing/preprocessing_multiscale_tta_benchmark/irrecoverable_21_multiscale_results.csv`.

## Phase 2

- Code: `preprocessing_best_config/benchmark_multiscale_selector_phase2.py`.
- Selector đã khóa: `outputs/testing/preprocessing_multiscale_selector_phase2/phase2_selector.joblib`.
- Validation OOF: `outputs/testing/preprocessing_multiscale_selector_phase2/predictions_val_phase2_oof.csv`.
- Test khóa: `outputs/testing/preprocessing_multiscale_selector_phase2/predictions_test_phase2_locked.csv`.

## Phase 3

- Augmentation: `refinement_finetune/phase3_controlled_augmentation.py`.
- Fine-tune: `refinement_finetune/train_phase3_controlled_augmentation.py`.
- Checkpoint: `outputs/phase3_controlled_aug_full_frozen_eval/best_phase3_parseq_anpr.pt`.
- Manifest khóa: `outputs/phase3_controlled_aug_full_frozen_eval/dataset_manifest.csv`.
- Predictions: `outputs/phase3_controlled_aug_full_frozen_eval/eval_*_predictions*.csv`.

## RL mở rộng sau Phase 3

- Contextual bandit/PPO code: `rl_restoration/`.
- PPO report: `RL_PHASE5_PPO_REPORT.md`.
- PPO checkpoint: `outputs/rl_restoration/ppo_prior_seed_123/best_ppo_restoration_policy.pt`.
- PARSeq PPO-view checkpoint: `outputs/rl_restoration/parseq_ppo_hard_curriculum/best_parseq_rl_policy_mixture.pt`.

## Phase 6 candidate-aware OOF PPO

- Toàn bộ kiến trúc và artefact: `reinforcement_learning/phase_6_candidate_oof_ppo/`.
- Candidate cache: `phase_6_candidate_oof_ppo/results/candidate_cache/`.
- Experimental checkpoint: `phase_6_candidate_oof_ppo/results/run_residual_seed_123/best_candidate_oof_ppo.pt`.
- Statistical summary: `phase_6_candidate_oof_ppo/results/run_residual_seed_123/summary.json`.
- Báo cáo: `phase_6_candidate_oof_ppo/PHASE6_REPORT.md`.

## Phase 7 compact multi-scale PPO

- Toàn bộ kiến trúc/artefact: `reinforcement_learning/phase_7_compact_multiscale_ppo/`.
- Action registry: `phase_7_compact_multiscale_ppo/action_space.py`.
- Candidate cache: `phase_7_compact_multiscale_ppo/results/cache/`.
- Seed 725 checkpoint: `phase_7_compact_multiscale_ppo/results/run_seed_725/best_candidate_oof_ppo.pt`.
- OCR/guard checkpoint: `phase_7_compact_multiscale_ppo/results/run_ocr_guard_seed_727/best_candidate_oof_ppo.pt`.
- Confirmatory summary: `phase_7_compact_multiscale_ppo/results/confirmatory_seed_728/summary.json`.
- Báo cáo: `phase_7_compact_multiscale_ppo/PHASE7_REPORT.md`.
- External holdout evaluator: `phase_7_compact_multiscale_ppo/evaluate_external.py`.
- Promotion gate and active registry: `phase_7_compact_multiscale_ppo/promote.py`, `results/active_policy.json` (only after external gate pass).
- Label-free runtime: `phase_7_compact_multiscale_ppo/runtime.py`.

## Phase 8 conservative consensus PPO

- Toàn bộ kiến trúc/artefact: `reinforcement_learning/phase_8_consensus_ppo/`.
- Consensus evaluator: `phase_8_consensus_ppo/evaluate.py`.
- Hai checkpoint khóa: Phase 7 seed 727 và seed 728.
- Prospective candidate lock: `phase_8_consensus_ppo/prospective_policy.json`.
- Validation summary: `phase_8_consensus_ppo/results/validation_prospective_locked/summary.json`.
- External plate-crop diagnostic: `phase_8_consensus_ppo/results/external_plate_crop_repair_prospective_locked_diagnostic/summary.json`.
- Promotion guard: `phase_8_consensus_ppo/promote.py`.
- Label-free runtime: `phase_8_consensus_ppo/runtime.py`.
- Báo cáo: `phase_8_consensus_ppo/PHASE8_REPORT.md`.
- Fresh review queue: `phase_8_consensus_ppo/fresh_holdout_review_queue.csv` và `fresh_holdout_review_queue_audit.json`.
- Git-ignored review working copy: `phase_8_consensus_ppo/fresh_holdout_review_queue_reviewed.csv`.
- Locked manifest finalizer: `phase_8_consensus_ppo/finalize_fresh_holdout.py`.
- `results/active_policy.json` chỉ tồn tại sau một fresh locked-confirmatory external gate; diagnostic hiện tại không thể tạo registry.
