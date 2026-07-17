# Phase 7 — Compact multi-scale candidate PPO

Phase 7 mở rộng action space từ 10 restoration view sang chín multi-scale view được rút gọn từ search space 65 view. Chín view giữ nguyên oracle validation của 65 view (`97,7330%`) nhưng giảm số OCR call 86,2%.

Kiến trúc policy dùng lại candidate-set actor–critic của Phase 6 và bổ sung cho mỗi candidate:

- PARSeq encoder pooling và OCR uncertainty;
- metadata zoom/upscale/unwrap/preprocessing;
- OCR string one-hot theo vị trí;
- normalized confidence, candidate vote fraction, consensus confidence;
- trạng thái khác baseline và edit distance label-free tới prediction baseline;
- OOF teacher prior và residual disagreement guard.

Mọi cache, checkpoint và báo cáo mới đều nằm trong thư mục này. Pipeline chỉ đọc checkpoint/manifest nguồn và không ghi ra ngoài `reinforcement_learning`.

## Chạy

```powershell
python reinforcement_learning\run_phase.py phase7-cache --split train
python reinforcement_learning\run_phase.py phase7-cache --split val
python reinforcement_learning\run_phase.py phase7
python -m unittest reinforcement_learning.phase_7_compact_multiscale_ppo.test_phase7
```

Khi có external holdout mới với `split=external_holdout`, `input_contract=plate_crop`, normalized-label group không trùng historical train/val/test và ít nhất 500 mẫu:

```powershell
python reinforcement_learning\run_phase.py phase7-cache --split external_holdout --manifest path\external_manifest.csv --output-dir reinforcement_learning\phase_7_compact_multiscale_ppo\results\external_cache --preflight
python reinforcement_learning\run_phase.py phase7-cache --split external_holdout --manifest path\external_manifest.csv --output-dir reinforcement_learning\phase_7_compact_multiscale_ppo\results\external_cache
python reinforcement_learning\run_phase.py phase7-external
```

Lệnh đầu chỉ preflight: không tải model, không OCR và không ghi artifact. Manifest external bắt buộc có `image_path`, `label` (hoặc `target`) và `split`; mọi dòng được đánh giá phải có `split=external_holdout`, không được trùng `image_path`. Giữ nguyên file manifest sau khi tạo cache: cache lưu SHA-256 và evaluator so lại provenance này. Cả cache lẫn evaluator đều từ chối ghi đè để bảo đảm holdout chỉ được mở một lần.

Tập dưới 500 mẫu bị formal preflight từ chối. Chỉ dùng `--allow-underpowered-diagnostic` cho phân tích không-promotable. Script `prepare_external_holdout.py --crop-uncropped` chuẩn hóa các hàng `cropped=false` bằng `bounding_box`; `audit_external_holdout.py` tạo manifest group-disjoint và audit SHA-256.

`phase7` mặc định trỏ tới run seed 727 đã khóa; nếu summary đã tồn tại, trainer từ chối ghi đè để bảo vệ holdout audit.

## Promotion và runtime label-free

External evaluation chỉ tạo `promotion_status=eligible` khi toàn bộ paired CI, character, net-fix và McNemar gate đạt. Khi đó mới được tạo registry active (một lần):

```powershell
python reinforcement_learning\run_phase.py phase7-promote
python reinforcement_learning\run_phase.py phase7-runtime --manifest path\runtime_images.csv --output-dir reinforcement_learning\phase_7_compact_multiscale_ppo\results\runtime_inference_001
```

`runtime_images.csv` chỉ cần cột `image_path`; labels không được đọc. Runtime tạo đủ 9 compact views, encoder/OCR observation, OOF-teacher prior và PPO selection, rồi ghi prediction/action vào `runtime_selections.csv`. Nếu external gate chưa pass thì registry không thể tạo, do đó runtime không thể vô tình dùng checkpoint experimental.

## Trạng thái

PPO tăng exact so với OOF teacher trên validation và trên cả ba group holdout nội bộ đã chạy. Tuy nhiên không run nào đồng thời qua paired CI và McNemar hai phía, nên checkpoint vẫn `experimental_not_promoted`. Xem `PHASE7_REPORT.md`.
