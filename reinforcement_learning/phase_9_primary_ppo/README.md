# Phase 9 — Primary PPO prospective confirmation

Phase 9 đã hoàn tất và thất bại formal gate. PPO seed 727 tăng exact từ `46,1538%` lên `46,6154%` (`7 fix / 4 break`) nhưng exact CI chứa 0 và McNemar `p=0,5488`; baseline tiếp tục được giữ. Holdout này hiện chỉ là development data, không được chạy lại để promote.

Khác Phase 8, holdout mới không yêu cầu người dùng accept lại label. Builder dùng `extracted_character` từ `labels.csv` cho mọi dòng từ sheet row 733 trở lên, loại duy nhất `review_status=rejected`, rồi tự động loại label/ảnh/source đã mở, duplicate, label sai charset và media/crop lỗi. Status `corrected` vẫn dùng `extracted_character` đúng theo data contract đã yêu cầu.

Luồng an toàn:

```powershell
python reinforcement_learning\run_phase.py phase9-status
python reinforcement_learning\run_phase.py phase9-prepare
python reinforcement_learning\run_phase.py phase7-cache --split external_holdout --manifest reinforcement_learning\phase_9_primary_ppo\fresh_external_holdout\external_manifest.csv --output-dir reinforcement_learning\phase_7_compact_multiscale_ppo\results\phase9_fresh_external_cache --preflight
python reinforcement_learning\run_phase.py phase7-cache --split external_holdout --manifest reinforcement_learning\phase_9_primary_ppo\fresh_external_holdout\external_manifest.csv --output-dir reinforcement_learning\phase_7_compact_multiscale_ppo\results\phase9_fresh_external_cache
python reinforcement_learning\run_phase.py phase9-evaluate
```

`phase9-evaluate` là one-shot. Chỉ khi exact delta dương, exact paired CI loại 0, character không giảm, net fixes dương và McNemar hai phía `<0,05` thì mới chạy:

```powershell
python reinforcement_learning\run_phase.py phase9-promote
python reinforcement_learning\run_phase.py phase9-runtime --manifest path\runtime_plate_crops.csv --output-dir reinforcement_learning\phase_9_primary_ppo\results\runtime_001
```

Nếu gate thất bại, baseline tiếp tục là policy active. Không được đổi threshold hoặc chọn slice trên holdout đã mở rồi đánh giá lại.
