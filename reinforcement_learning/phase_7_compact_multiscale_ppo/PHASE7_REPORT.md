# Báo cáo Phase 7 — Compact multi-scale candidate PPO

## Kết luận

Phase 7 đã tạo action space có oracle mạnh hơn rõ rệt và chuyển một phần oracle gap thành gain RL thực tế. Exact PPO cao hơn teacher trên validation và trên ba group holdout nội bộ độc lập theo seed. Run seed 725 cho hiệu ứng mạnh nhất: `+1,0687` điểm % exact, paired CI 95% hoàn toàn dương, character tăng và 9 win/2 loss. Tuy nhiên McNemar hai phía còn `p=0,0654`, chưa qua ngưỡng 0,05.

Hai replication seed 727 và 728 vẫn có delta exact dương nhưng không đạt đồng thời CI/McNemar. Vì vậy các checkpoint Phase 7 vẫn ở trạng thái **experimental**, không thay thế locked Phase 5 policy và không được dùng để tuyên bố vượt test.

## Action space compact

- Nguồn: 65 multi-scale view của Phase 1, chọn greedy set-cover chỉ trên validation cũ.
- Baseline validation: 92,6952%.
- Oracle 65-view validation: 97,7330%.
- Oracle 9-view validation: 97,7330%.
- OCR calls: giảm từ 65 xuống 9, tương đương 86,2%.
- Oracle train 9-view: 96,9725%, cao hơn baseline train 2,9052 điểm %.

Danh sách action được khóa trong `action_space.py`. Test và các holdout Phase 6 không tham gia chọn view.

## Kết quả

| Run | Holdout | Teacher exact | PPO exact | Delta exact | Net fixes PPO | Paired exact CI 95% | McNemar win/loss | p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| seed 725 | 655 | 92,8244% | **93,8931%** | **+1,0687** | +4 | [+0,1527; +2,1374] | 9 / 2 | 0,0654 |
| seed 727 + OCR/guard | 654 | 94,6483% | **95,4128%** | **+0,7645** | +5 | [0,0000; +1,5291] | 6 / 1 | 0,1250 |
| seed 728 confirmatory 40% | 1.308 | 95,0306% | **95,3364%** | **+0,3058** | +5 | [-0,3058; +0,9174] | 10 / 6 | 0,4545 |

Các holdout khác nhau theo seed nên không so sánh trực tiếp accuracy tuyệt đối giữa các hàng và không gộp p-value hậu nghiệm. Seed 728 đã được tuyên bố là replication cuối trước khi chạy; không thử thêm seed sau khi nó không qua gate.

## Validation của candidate-string architecture

Seed 727 tăng validation từ teacher `92,6952%` lên PPO `93,7028%`, fixed/broken `5/1`, revise `2,02%`. Seed 728 đạt PPO `93,9547%`, fixed/broken `5/0`. Điều này xác nhận OCR-string/consensus token và disagreement guard giảm harm trên validation, dù power holdout vẫn chưa đủ cho promotion.

## Protocol

- Teacher prediction trên development được tạo 5-fold OOF theo label group.
- Cùng label không xuất hiện ở development và holdout của một run.
- Holdout không tham gia teacher, PPO, epoch, margin hoặc guard selection.
- Test cũ không được nạp ở build-cache hoặc trainer.
- Statistical gate giữ nguyên: delta exact dương; CI exact loại 0; character không giảm; net fixes dương; McNemar exact hai phía `p<0,05`.

Các group holdout ở đây tách từ train lịch sử. Chúng kiểm tra leakage của Phase 7 nhưng không thay thế một external holdout thu thập mới sau toàn bộ quá trình phát triển.

## Artefact

- Cache: `results/cache/`.
- Run effect mạnh nhất: `results/run_seed_725/`.
- Run OCR-string/guard: `results/run_ocr_guard_seed_727/`.
- Confirmatory replication: `results/confirmatory_seed_728/`.
- Action registry: `action_space.py`.
- Cache builder: `build_cache.py`.
- Entrypoint: `train.py`.

## Điều kiện còn thiếu để hoàn thành gate Phase 5

Cần một external group holdout mới, được khóa trước inference, có đủ trường hợp baseline/teacher sai để tạo ít nhất khoảng 15–20 cặp discordant. Không tiếp tục chọn seed hoặc margin trên các holdout nội bộ hiện tại. Với external holdout đó, chạy đúng checkpoint/config đã khóa và chỉ promote nếu paired CI cùng McNemar xác nhận.

Pipeline external hiện yêu cầu manifest có `image_path`, `label` (hoặc `target`) và `split=external_holdout`, không có `image_path` trùng. Giữ nguyên file manifest sau khi tạo cache: cache ghi SHA-256 và evaluator so lại provenance trước inference; cache và summary không thể bị ghi đè.

Trước khi cache, dùng `phase7-cache ... --preflight`: lệnh chỉ kiểm tra schema, ảnh tồn tại và fingerprint manifest, không tải model, không inference và không ghi artifact.

Sau khi external gate pass, `phase7-promote` tạo registry active duy nhất chứa checkpoint PPO, checkpoint PARSeq, action registry và hash external summary. `phase7-runtime --manifest <CSV image_path>` dùng registry này để chạy selection label-free trên ảnh mới; checkpoint experimental không có đường vào runtime.

## Cập nhật external holdout ngày 2026-07-15

Manifest người dùng cung cấp có 682 ảnh hợp lệ sau khi loại 49 `rejected` và một nhãn rỗng. Audit phát hiện 528/682 dòng, tương ứng 368 normalized label, trùng historical train/val/test; không có ảnh trùng SHA-256. Manifest group-disjoint còn 154 ảnh/151 label.

Run đầu tiên dùng nhầm 50 ảnh `cropped=false` dưới dạng ảnh toàn cảnh. Cả 50 ảnh đều sai ở baseline, oracle, teacher và PPO, nên run được giữ làm audit lỗi input contract và không đủ điều kiện promotion. Pipeline sau đó được sửa để bắt buộc `input_contract=plate_crop` và crop theo `bounding_box`.

Corrected plate-crop diagnostic tăng baseline từ `54,5455%` lên `72,0779%`, oracle từ `59,7403%` lên `81,1688%`. PPO seed 725 đạt `73,3766%` (`5 fix / 3 break`); seed 727 đạt `75,3247%` (`6/1`); seed 728 đạt `75,9740%` (`7/1`). Các run này là protocol-repair diagnostic, không được dùng để promote. Kết quả dẫn tới Phase 8 consensus PPO; xem `../phase_8_consensus_ppo/PHASE8_REPORT.md`.
