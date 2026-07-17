# RL Restoration cho PARSeq ANPR

Thư mục này triển khai quy trình học tăng cường ngoại tuyến theo hướng contextual bandit. Policy không sinh ảnh; nó chọn một trong các phép biến đổi ảnh đã được kiểm soát, sau đó PARSeq nhận ảnh được chọn để OCR.

## Luồng xử lý

1. `build_trajectory_cache.py` chạy toàn bộ action trên tập train/validation và lưu reward OCR.
2. `train_router.py` học dự đoán reward của từng action từ đặc trưng ảnh, đặc trưng encoder PARSeq và độ bất định OCR.
3. Router chỉ chọn action khác `stop_baseline` khi reward dự đoán vượt margin đã khóa.
4. `finetune_with_policy.py` fine-tune PARSeq bằng hỗn hợp ảnh baseline, ảnh raw và ảnh do policy chọn. Các mẫu train mà policy sửa đúng được lấy mẫu mạnh hơn.
5. `evaluate_locked_policy.py` chỉ đánh giá router/checkpoint đã khóa trên test; không tối ưu siêu tham số bằng test.

Action space hiện có 10 action một bước trong `actions.py`, gồm giữ baseline, raw RGB, CLAHE, homomorphic, bilateral, adaptive noise, upscale và unwrap biển hai dòng. Mỗi action luôn bắt đầu từ ảnh gốc để tránh chuỗi xử lý ngoài kiểm soát.

## Tái lập checkpoint tốt nhất

Các cache và router đã tạo nằm trong `outputs/rl_restoration`. Chạy fine-tune hard-aware:

```powershell
python rl_restoration\finetune_with_policy.py `
  --epochs 10 `
  --learning-rate 1e-6 `
  --freeze-encoder-epochs 2 `
  --early-stopping-patience 4 `
  --hard-policy-probability 0.70 `
  --hard-raw-probability 0.10 `
  --hard-sample-weight 4 `
  --output-dir outputs\rl_restoration\parseq_policy_hard_curriculum
```

Đánh giá policy đã khóa:

```powershell
python rl_restoration\evaluate_locked_policy.py `
  --comparison-checkpoint outputs\rl_restoration\parseq_policy_hard_curriculum\best_parseq_rl_policy_mixture.pt `
  --output-dir outputs\rl_restoration\test_locked_hard_curriculum
```

Xem kết quả đầy đủ tại `RL_PHASE4_EXPERIMENT_REPORT.md` ở thư mục gốc.

## PPO hai bước

PPO dùng MDP hai bước có kiểm soát:

1. chọn một restoration view hoàn chỉnh từ ảnh gốc;
2. quan sát OCR của view đó và chọn action cuối, gồm khả năng giữ view, đổi view hoặc quay về baseline.

Router reward-regression đã khóa được dùng làm teacher prior. Actor học residual trên prior bằng clipped PPO; critic học dense OCR reward; advantage được tính bằng GAE. Cấu hình mặc định trong `train_ppo.py` là cấu hình seed 123 tốt nhất đã khóa trên validation.

```powershell
python rl_restoration\train_ppo.py `
  --output-dir outputs\rl_restoration\ppo_prior_seed_123
```

Đánh giá một lần trên test khóa:

```powershell
python rl_restoration\evaluate_locked_ppo.py `
  --ppo-checkpoint outputs\rl_restoration\ppo_prior_seed_123\best_ppo_restoration_policy.pt `
  --comparison-checkpoint outputs\rl_restoration\parseq_ppo_hard_curriculum\best_parseq_rl_policy_mixture.pt `
  --output-dir outputs\rl_restoration\test_locked_ppo_finetuned
```

Kết quả và giới hạn của PPO được ghi trong `RL_PHASE5_PPO_REPORT.md`.

Chạy PPO end-to-end trên một crop biển số mới:

```powershell
python rl_restoration\ppo_runtime.py path\to\plate.png
```

Kết quả JSON gồm dự đoán baseline, action đầu, action cuối, cờ `revised`, confidence và OCR cuối.
