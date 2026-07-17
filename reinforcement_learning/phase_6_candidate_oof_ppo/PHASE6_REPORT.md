# Báo cáo Phase 6 — Candidate-aware PPO với OOF teacher

## Kết luận

Phase 6 đã triển khai đủ observation cache cho mọi candidate view, teacher OOF, candidate-set actor–critic, group holdout và kiểm định ghép cặp. PPO residual seed 123 tăng exact validation từ 93,4509% của teacher lên 93,9547% (+0,5038 điểm %), đồng thời dùng nhánh revise trên 6,80% ảnh validation.

Trên holdout khóa, PPO và teacher cùng đạt 93,6735% exact. Paired 95% CI của delta exact là `[-1,6327; +1,4286]` điểm %, McNemar exact có `p = 1,0`, và net fixes của PPO so với baseline là `-1`. Do đó Phase 6 **chưa vượt gate chính thức** và checkpoint không được thăng cấp thay cho policy Phase 5.

## Protocol chống leakage

- Nguồn tạo holdout: split train 3.270 ảnh; test cũ không được nạp.
- Group key: target/label đã chuẩn hóa.
- Development: 2.780 ảnh, 1.893 nhóm.
- Holdout: 490 ảnh, 347 nhóm.
- Group overlap: 0.
- Teacher prior trên development: 5-fold OOF.
- Teacher validation/holdout: model full chỉ fit development.
- Validation cũ chỉ chọn epoch và margin.
- Holdout không tham gia teacher, BC, PPO, checkpoint hoặc margin; chỉ mở sau khi khóa policy.

Lưu ý: holdout này được tách từ train lịch sử vì repository chưa có tập ảnh mới hoàn toàn. Nó hợp lệ để audit pipeline Phase 6 (không đi vào training Phase 6), nhưng chưa mạnh bằng một external holdout thu thập sau toàn bộ quá trình phát triển trước đây.

## Kết quả run residual seed 123

| Split / policy | Exact | Character accuracy | Fixed | Broken | Revise |
|---|---:|---:|---:|---:|---:|
| Validation teacher OOF/full | 93,4509% | 98,6524% | 4 | 1 | 0% |
| Validation candidate-PPO | **93,9547%** | **98,9280%** | 7 | 2 | 6,80% |
| Holdout teacher | 93,6735% | 99,1027% | 1 | 2 | 0% |
| Holdout candidate-PPO | 93,6735% | **99,1276%** | 7 | 8 | 10,41% |

Checkpoint tốt nhất được chọn tại PPO epoch 46, `first_margin=0.2`, `revise_margin=0.0`.

## Kiểm định holdout so với teacher

| Kiểm định | Kết quả |
|---|---:|
| Delta exact | 0,0000 điểm % |
| Paired bootstrap 95% CI exact | [-1,6327; +1,4286] điểm % |
| Delta character trung bình theo ảnh | -0,0057 điểm % |
| McNemar fixed / broken | 7 / 7 |
| McNemar exact p-value | 1,0 |
| Formal gate | **Không đạt** |

Character accuracy tổng hợp tăng nhẹ do cách weighting theo tổng số ký tự, trong khi paired mean per-image giảm rất nhỏ; gate dùng paired per-image và vì vậy đánh dấu `character_not_lower=false` một cách bảo thủ.

## Artefact

- Candidate cache: `results/candidate_cache/`.
- Checkpoint: `results/run_residual_seed_123/best_candidate_oof_ppo.pt`.
- Lịch sử: `results/run_residual_seed_123/training_history.csv`.
- OOF assignment: `results/run_residual_seed_123/development_oof_assignments.csv`.
- Validation trace: `results/run_residual_seed_123/validation_selections.csv`.
- Holdout PPO/teacher trace: `results/run_residual_seed_123/holdout_selections.csv` và `holdout_teacher_selections.csv`.
- Protocol: `results/run_residual_seed_123/protocol.json`.
- Summary + statistical gate: `results/run_residual_seed_123/summary.json`.

## Quyết định tiếp theo

Không tiếp tục chỉnh hyperparameter trên holdout này. Muốn kết luận tăng output RL ngoài validation cần thu thập một external holdout mới, khóa trước khi thay đổi tiếp, và tăng số trường hợp baseline sai có candidate sửa được. Candidate observation đã giúp PPO học revise trên validation, nhưng action space 10 view vẫn tạo quá ít discordant win để CI/McNemar có đủ lực thống kê.

