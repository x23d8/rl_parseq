# Báo cáo Phase 4 — RL Restoration và fine-tune PARSeq

## Kết luận

Contextual-bandit router và curriculum hard-aware đã tạo ra cải thiện có thể đo được trên tập test khóa. Cải thiện còn nhỏ (một ảnh đúng thêm trên 411 ảnh), nhưng xuất hiện đồng thời ở chế độ ảnh gốc và chế độ policy, trong khi test không được dùng để chọn router hoặc checkpoint.

## Cấu hình đã khóa

- Parent checkpoint: `outputs/phase3_controlled_aug_full_frozen_eval/best_phase3_parseq_anpr.pt`.
- Router: `outputs/rl_restoration/router_seed_123/best_reward_router.pt`.
- Router epoch: 20; selection margin: `0.005`.
- PARSeq checkpoint mới: `outputs/rl_restoration/parseq_policy_hard_curriculum/best_parseq_rl_policy_mixture.pt`.
- Checkpoint tốt nhất ở epoch 8.
- 87 mẫu train hard được lấy mẫu với trọng số 4.
- Với mẫu hard: 70% policy view, 10% raw RGB, 20% baseline.
- Với mẫu thường: 25% policy view, 25% raw RGB, 50% baseline.

## Kết quả

| Chế độ | Parent exact | Checkpoint mới exact | Chênh lệch | Parent char acc | Checkpoint mới char acc |
|---|---:|---:|---:|---:|---:|
| Validation, ảnh gốc | 92,6952% | 92,9471% | +0,2519 điểm % (+1/397) | 98,6524% | 98,6830% |
| Validation, policy | 94,4584% | 94,4584% | 0 | 98,8974% | 98,8974% |
| Test, ảnh gốc | 91,9708% | 92,2141% | +0,2433 điểm % (+1/411) | 98,8684% | 98,8088% |
| Test, policy | 92,4574% | 92,7007% | +0,2433 điểm % (+1/411) | 98,9279% | 98,9577% |

Lưu ý: character accuracy trên test ảnh gốc giảm nhẹ 0,0596 điểm %, nên checkpoint mới phù hợp nhất khi triển khai cùng router. Kết quả này chưa đủ mạnh để kết luận thống kê rằng RL vượt trội một cách ổn định trên mọi phân phối dữ liệu.

## Oracle và khoảng trống còn lại

- One-step oracle validation đạt 94,9622% exact, cao hơn router 94,4584%.
- One-step oracle test đạt 95,8637% exact, cao hơn hệ thống router + checkpoint mới 92,7007%.
- Khoảng trống chủ yếu nằm ở khả năng chọn đúng action, không phải thiếu action hoàn toàn.
- Router test sửa đúng 5 ảnh nhưng làm hỏng 3 ảnh so với baseline; net gain là 2 ảnh trước khi cập nhật trọng số.

## Quyết định cho vòng tiếp theo

Chưa nên chuyển thẳng sang PPO nhiều bước. Bước tiếp theo nên tăng dữ liệu học router cho các tình huống hiếm và giảm `broken` bằng uncertainty calibration hoặc reject-to-baseline. Sau đó chạy tối thiểu ba seed fine-tune, đánh giá paired bootstrap/McNemar trên holdout mới. Chỉ thử policy hai bước hoặc PPO khi oracle hai bước chứng minh tạo khoảng tăng đáng kể so với one-step oracle.

## Artefact

- `outputs/rl_restoration/parseq_policy_hard_curriculum/summary.json`: metric validation và test ảnh gốc.
- `outputs/rl_restoration/parseq_policy_hard_curriculum/history.csv`: lịch sử 10 epoch.
- `outputs/rl_restoration/test_locked_hard_curriculum/summary.json`: metric policy trên test khóa.
- `outputs/rl_restoration/test_locked_hard_curriculum/test_locked_policy_selections.csv`: action được chọn cho từng ảnh test.
- `outputs/rl_restoration/test_locked_hard_curriculum/test_comparison_policy_predictions.csv`: dự đoán của checkpoint mới.

