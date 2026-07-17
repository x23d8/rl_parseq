# Báo cáo Phase 8 — Conservative consensus PPO

## Kết luận cuối cùng

Phase 8 kết hợp hai PPO OCR-string/disagreement-guard seed 727 và 728. Policy chỉ đổi khỏi baseline khi hai PPO tạo cùng một prediction khác baseline; trường hợp không đồng thuận hoặc prediction không đổi thì rollback về baseline.

Trên validation cũ, consensus tăng exact từ `92,6952%` lên `93,4509%`, đạt `3 fix / 0 break`. Trên external plate-crop diagnostic group-disjoint, consensus tăng từ `72,0779%` lên `75,9740%`, đạt `6 fix / 0 break`, delta `+3,8961` điểm %, exact CI 95% `[+1,2987; +7,1429]`, character delta `+1,0140` điểm % và McNemar `p=0,03125`. Toàn bộ gate so baseline đều đạt.

Kết quả external này không được promote vì là protocol-repair diagnostic: cùng nguồn ảnh đã từng được mở ở một run sai input contract trước khi 50 ảnh toàn cảnh được crop lại.

Sau đó Phase 8 đã hoàn tất fresh locked-confirmatory đúng protocol trên `504` plate crop mới, unique-label và không trùng 4.078 group lịch sử. Baseline đạt `82,1429%`; consensus đạt `83,1349%`, tăng `+0,9921` điểm %, `6 fix / 1 break`. Character accuracy tăng `+0,3527` điểm % với CI hoàn toàn dương, nhưng exact CI 95% `[0; +1,9841]` còn chạm 0 và McNemar `p=0,125`. Vì vậy formal gate thất bại, `promotion_eligible=false`, không có active registry và baseline tiếp tục được giữ.

Hậu kiểm trên cùng holdout cho thấy seed 727 riêng lẻ đạt `84,3254%`, `14 fix / 3 break`, exact delta `+2,1825` điểm %, CI 95% `[+0,5952; +3,7698]` và McNemar `p=0,01273`. Đây chỉ là development evidence sau khi holdout đã mở; không thể thay luật Phase 8 hoặc promote hậu nghiệm. Kết quả này được dùng để khóa candidate Phase 9 trước một holdout hoàn toàn mới.

## Vì sao cần Phase 8

| Policy | Validation exact | Fix/break | External diagnostic exact | Fix/break |
|---|---:|---:|---:|---:|
| Baseline | 92,6952% | 0/0 | 72,0779% | 0/0 |
| PPO seed 727 | 93,7028% | 5/1 | 75,3247% | 6/1 |
| PPO seed 728 | 93,9547% | 5/0 | 75,9740% | 7/1 |
| Phase 8 consensus | 93,4509% | **3/0** | **75,9740%** | **6/0** |

Consensus giữ gain ròng của seed 728 trên external diagnostic nhưng loại ca harm quan sát được, đồng thời giảm change rate xuống `8,44%` và baseline rate lên `91,56%`.

## Data contract đã khóa

- Mọi input external/runtime phải khai báo `input_contract=plate_crop`.
- Normalized label phải không xuất hiện trong historical train, validation hoặc test.
- Không cho ảnh trùng nội dung trong external hoặc trùng SHA-256 với historical manifest.
- Formal external preflight yêu cầu tối thiểu 500 mẫu group-disjoint; tập nhỏ hơn chỉ được chạy bằng cờ diagnostic explicit.
- Diagnostic/protocol-repair không thể tạo active registry, kể cả metric qua gate.
- Runtime label-free chỉ mở sau một external `locked_confirmatory` mới qua paired CI, character, net-fix và McNemar.

## Artefact

- Kiến trúc/evaluator: `evaluate.py`.
- Prospective checkpoint/action lock: `prospective_policy.json`.
- Promotion guard: `promote.py`.
- Runtime label-free: `runtime.py`.
- Prospective-locked validation: `results/validation_prospective_locked/`.
- Prospective-locked external diagnostic: `results/external_plate_crop_repair_prospective_locked_diagnostic/`.
- Test contract: `test_phase8.py`.
- Fresh review queue builder: `prepare_fresh_review_queue.py`.
- Reviewed-queue finalizer: `finalize_fresh_holdout.py`.
- Local keyboard reviewer: `review_server.py`.

## Fresh confirmatory đã hoàn tất

Fresh manifest đã khóa có SHA-256 `cbf3e3f19e287b2b087134afc34da242a45ab3246b41126dd79674dd1d5f123d`; receipt one-shot có claim ID `5ff3cfe030dbba6cb5642c8551d4eb95150b1bb703a902d2b09e82e032e02cfc`. Cache khóa PARSeq checkpoint cùng ba artifact candidate/state/trajectory trước evaluation. Phase 8 kết thúc với trạng thái `external_holdout_failed_gate`; không được chạy lại hoặc tune trên 504 mẫu này.

Queue V1 650 ảnh cho thấy label ban đầu chưa đủ tin cậy để dùng trực tiếp: tại checkpoint review 341 ảnh, có 308 quyết định accepted/corrected nhưng chỉ 214 group thực sự hợp lệ; 93 label sửa lại trùng historical/external và 1 group hợp lệ còn lại bị duplicate. Nếu giữ 650 ảnh, dự báo không đủ gate 500.

Vì chưa chạy inference, queue đã được mở rộng prospective thành `fresh_holdout_review_queue_v2.csv` với 900 ảnh, SHA-256 `eb2732b53f8e92169f7dc5dd4510798d67515614e2713f952f7a79096e991a9c`. V2 dùng cùng stable hash seed, giữ nguyên toàn bộ prefix 650 ảnh của V1 và migrate 341 quyết định; audit migration nằm ở `fresh_holdout_review_queue_v2_audit.json`. Với yield hiện tại, dự báo khoảng 565 group hợp lệ khi review xong, tạo biên an toàn cho gate 500.

Media preflight độc lập đã render `900/900` crop, không có lỗi crop/file, không có exact-image duplicate nội bộ và không có SHA-256 overlap với historical/external đã mở. Ordered rendered digest là `592cd0285399fce74ee7a68c56fda1169764b38a15a48293f07ca07d68f98b24`; audit nằm tại `fresh_holdout_review_queue_v2_media_audit.json` và xác nhận `inference_run=false`.

Reviewer/finalizer dùng chung review contract: decision của người review chỉ xác nhận label nhìn thấy, còn eligibility được tính tự động. Label trùng dữ liệu đã mở và duplicate final-label group bị exclude, không cần đổi decision thành rejected. Có thể finalize ngay khi một prefix liên tục theo hash order đạt `eligible_prefix >= 500`; phần đuôi chưa review không được chọn, nhờ đó tránh cherry-pick mà không bắt review thừa. External locked-confirmatory sau đó bị khóa bằng receipt one-shot thật sự; promotion/runtime xác minh lại provenance và hash checkpoint.

Label contract giới hạn đúng charset checkpoint `0-9A-Z` và `max_label_length=12`; ký tự Unicode hoặc label quá dài bị chặn ngay khi lưu correction. Cache mới lưu SHA-256 của PARSeq checkpoint cùng ba artefact candidate/state/trajectory; locked-confirmatory kiểm tra tất cả trước khi claim receipt, còn promotion/runtime từ chối nếu checkpoint thay đổi sau cache construction.

Runtime consensus cần sinh OCR/feature cho đủ 9 compact views trước khi chọn, nên action cost trong policy chỉ là regularization proxy. Runtime hiện đo chi phí thực gồm candidate generation và dual-policy selection: mean wall latency, p95 batch-amortized, throughput và breakdown từng view. Giá trị triển khai chưa thể báo trước promotion; phải benchmark trên đúng GPU/runtime manifest sau khi external gate pass.

Adapter `prepare_runtime_crops.py` nối output detector (`image_path,bounding_box,source_id`) với contract plate-crop của Phase 8 mà không đọc label hay chạy model. Nó clip/validate bbox, chặn path traversal/collision, ghi crop SHA-256 cùng audit và tạo manifest dùng trực tiếp cho `phase8-runtime`. Adapter bbox chưa thực hiện perspective rectification bốn điểm; đây vẫn là bước upstream nếu detector cung cấp corner geometry.

Final manifest giữ lineage nguồn/transform cùng chiều rộng, chiều cao crop. One-shot evaluator xuất descriptive slices theo `source`, `input_transform` và minimum-side bucket để audit nhóm lỗi; các slice được đánh dấu `descriptive_only_not_a_promotion_gate`, nên không tạo thêm đường chọn policy hậu nghiệm.
