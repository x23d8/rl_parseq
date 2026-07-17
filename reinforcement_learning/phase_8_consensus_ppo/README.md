# Phase 8 — Conservative consensus PPO

Phase 8 khóa hai PPO có OCR-string/disagreement guard từ Phase 7 (seed 727 và 728). Policy chỉ thay baseline khi cả hai PPO tạo cùng một prediction khác baseline; nếu không đồng thuận thì chọn baseline.

`prospective_policy.json` khóa SHA-256 của hai checkpoint, action registry, rule và external contract trước holdout mới. Evaluator từ chối checkpoint/action bị đổi.

Mục tiêu là giữ gain restoration nhưng giảm harm và độ nhạy seed. Fresh locked-confirmatory 504 mẫu đã hoàn tất: consensus tăng từ `82,1429%` lên `83,1349%`, nhưng exact CI còn chạm 0 và McNemar `p=0,125`. Phase 8 đã thất bại formal gate, không active và không được chạy lại trên holdout đã mở.

Hậu kiểm seed 727 riêng lẻ qua đủ gate (`+2,1825` điểm %, CI `[+0,5952; +3,7698]`, McNemar `p=0,01273`) nhưng không thể promote hậu nghiệm. Nó đã được khóa thành candidate Phase 9 và bắt buộc dùng holdout mới.

```powershell
python reinforcement_learning\run_phase.py phase8-consensus
python reinforcement_learning\run_phase.py phase8-consensus --split external_holdout --cache-dir reinforcement_learning\phase_7_compact_multiscale_ppo\results\external_cache_plate_crop_repair --output-dir reinforcement_learning\phase_8_consensus_ppo\results\external_plate_crop_repair_prospective_locked_diagnostic --evaluation-role protocol_repair_diagnostic
python -m unittest reinforcement_learning.phase_8_consensus_ppo.test_phase8
```

Promotion tiếp theo bắt buộc dùng một holdout plate-crop mới, group-disjoint, chưa từng inference và đủ power thống kê.
Formal preflight yêu cầu tối thiểu 500 mẫu. Tập nhỏ hơn chỉ được build bằng `--allow-underpowered-diagnostic` và không thể dùng cho locked-confirmatory promotion.

Khi một external `locked_confirmatory` mới qua toàn bộ gate:

```powershell
python reinforcement_learning\run_phase.py phase8-promote
python reinforcement_learning\run_phase.py phase8-runtime --manifest path\runtime_plate_crops.csv --output-dir reinforcement_learning\phase_8_consensus_ppo\results\runtime_001
```

Runtime manifest bắt buộc có `image_path,input_contract` và mọi dòng phải khai báo `input_contract=plate_crop`. Diagnostic/protocol-repair không thể tạo active registry.

Runtime Phase 8 tạo đủ 9 compact candidate views trước khi hai PPO bỏ phiếu consensus; `mean_action_cost` chỉ là proxy của action cuối, không phải chi phí thực. `runtime_summary.json` báo thêm wall-clock mean, p95 batch-amortized, throughput, thời gian candidate generation/policy selection và breakdown từng view. Model-load cùng output I/O được loại khỏi contract đo.

Nếu đầu vào là kết quả detector trên ảnh toàn cảnh, tạo plate-crop manifest label-free trước:

```powershell
python reinforcement_learning\run_phase.py phase8-runtime-crops --manifest detector_boxes.csv --output-dir reinforcement_learning\phase_8_consensus_ppo\results\runtime_detector_001
python reinforcement_learning\run_phase.py phase8-runtime --manifest reinforcement_learning\phase_8_consensus_ppo\results\runtime_detector_001\runtime_plate_crops.csv --output-dir reinforcement_learning\phase_8_consensus_ppo\results\runtime_001
```

`detector_boxes.csv` cần `image_path,bounding_box` và có thể có `source_id`; bounding box là JSON `[left,top,right,bottom]`. Adapter chỉ crop, ghi SHA/audit và không chạy inference. Perspective rectification theo bốn corner chưa nằm trong adapter bbox này.

`prepare_fresh_review_queue.py` tạo queue normalized-label group mới từ sau dòng sheet 733, loại mọi group lịch sử và external đã mở. Queue không phải evaluation manifest và không chạy inference. Do label ban đầu có thể sai, gate 500 được tính lại từ label cuối: chỉ group duy nhất, không trùng lịch sử/external mới được tính.

Queue đang dùng là `fresh_holdout_review_queue_v2.csv` gồm tối đa 900 ảnh; working copy là `fresh_holdout_review_queue_reviewed_v2.csv`. V2 giữ nguyên prefix 650 ảnh của V1 và migrate nguyên vẹn mọi quyết định đã có. Chỉ thay `review_decision` và `corrected_label`. Finalizer tự loại label trùng lịch sử/external và duplicate final-label group; có thể dừng review ngay khi prefix liên tục đạt 500 mẫu hợp lệ, không cần review phần đuôi còn lại. Ảnh final được crop và giữ trong Phase 8.

Review nhanh bằng trình duyệt cục bộ:

```powershell
python reinforcement_learning\run_phase.py phase8-review
```

Mở `http://127.0.0.1:8765`. Dùng `A` accept, `R` reject, `C` sửa label, phím mũi tên để điều hướng. Server chỉ bind localhost, hiển thị đúng plate crop cùng ảnh ngữ cảnh, lưu nguyên tử và không chạy OCR/PPO.

Mỗi lần lưu, reviewer tạo bản `.bak` nguyên tử của working CSV trước đó và rollback state trong RAM nếu ghi thất bại. Khi vừa đạt `eligible_prefix=500`, UI dừng tự chuyển ảnh và báo chạy finalizer.

Có thể kiểm tra queue và crop đầu tiên mà không mở server bằng `python reinforcement_learning\run_phase.py phase8-review --check`.

Lệnh trạng thái tổng hợp sẽ chỉ ra bước an toàn kế tiếp mà không chạy inference:

```powershell
python reinforcement_learning\run_phase.py phase8-status
```

Dùng thêm `--verbose` khi cần xem các row bị exclude. Media preflight cho toàn bộ queue được chạy bằng:

```powershell
python reinforcement_learning\run_phase.py phase8-preflight
```

Preflight này chỉ render crop và tính SHA-256, không tải OCR/PPO. Finalizer yêu cầu audit này còn khớp queue cùng hai manifest lịch sử. Correction chỉ chấp nhận charset `0-9A-Z`, tối đa 12 ký tự sau normalization.

Fresh final manifest giữ thêm `source`, `input_transform`, `crop_width`, `crop_height`. Locked evaluator nối metadata này vào `consensus_selections.csv` và ghi các slice mô tả theo nguồn, transform và bucket kích thước crop; slice chỉ dùng phân tích lỗi, không tham gia promotion gate.

Sau khi `formal_ready=true`:

```powershell
python reinforcement_learning\run_phase.py phase8-finalize
```

External `locked_confirmatory` xác minh SHA-256 của PARSeq checkpoint cùng ba cache artefact trước khi tạo receipt one-shot và inference. Receipt đã tồn tại sẽ chặn chạy lại cùng quy trình; promotion cũng kiểm tra receipt cùng hash candidate lock và hai PPO checkpoint.
