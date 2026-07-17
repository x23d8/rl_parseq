# Phase 12 — Guarded replicated PPO

Candidate seed 728 chỉ được phép thay baseline trên plate crop có sẵn với minimum side dưới 128 px. Rule đã khóa, nhưng chưa có đủ fresh unique-label group để tạo holdout mới.

Nguồn dữ liệu được quét từ sheet row 733 trở xuống, tức `sheet_row >= 733`. Label luôn lấy từ `extracted_character`; `pending` được chấp nhận và chỉ `review_status=rejected` bị loại.

```powershell
python reinforcement_learning\run_phase.py phase12-status
```

Khi status báo đủ 1.500 mẫu:

```powershell
python reinforcement_learning\run_phase.py phase12-prepare
python reinforcement_learning\run_phase.py phase7-cache --split external_holdout --manifest reinforcement_learning\phase_12_guarded_replicated_ppo\fresh_external_holdout\external_manifest.csv --output-dir reinforcement_learning\phase_7_compact_multiscale_ppo\results\phase12_fresh_external_cache --preflight
python reinforcement_learning\run_phase.py phase7-cache --split external_holdout --manifest reinforcement_learning\phase_12_guarded_replicated_ppo\fresh_external_holdout\external_manifest.csv --output-dir reinforcement_learning\phase_7_compact_multiscale_ppo\results\phase12_fresh_external_cache
python reinforcement_learning\run_phase.py phase12-evaluate
```

Builder fail-fast trước render/tạo folder nếu chưa đủ target. Evaluator claim receipt one-shot, tính formal gate trên action sau guard, đồng thời lưu raw PPO action để audit. Chỉ khi toàn bộ gate đạt mới chạy `phase12-promote`.

Runtime manifest cần `image_path,input_contract,input_transform`; `input_contract` phải là `plate_crop`, còn transform là `existing_plate_crop` hoặc `crop_source_bounding_box`. Runtime tự đọc kích thước ảnh, xác minh width/height khai báo nếu có và short-circuit: ảnh không qua guard chỉ chạy baseline view.

```powershell
python reinforcement_learning\run_phase.py phase12-runtime --manifest path\runtime_plate_crops.csv --output-dir reinforcement_learning\phase_12_guarded_replicated_ppo\results\runtime_001
```

`phase12-status` chỉ đọc manifest/label để báo pool và safe next action, không render và không chạy OCR/PPO.
