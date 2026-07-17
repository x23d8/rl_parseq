# Báo cáo Phase 9 — Primary PPO fresh confirmation

## Kết luận cuối cùng

Phase 9 đã hoàn tất one-shot trên 650 plate crop mới, 650 label/ảnh duy nhất, không trùng historical, Phase 7, Phase 8 hoặc 900 source trong queue Phase 8. Holdout dùng trực tiếp `extracted_character`, không yêu cầu review thủ công.

Baseline đạt `46,1538%` exact và `83,6522%` character accuracy. PPO seed 727 đạt `46,6154%` exact và `83,9495%` character accuracy, tương ứng `7 fix / 4 break`, net `+3`, exact delta `+0,4615` điểm % và character delta `+0,2763` điểm %. Tuy nhiên exact CI 95% `[-0,4615; +1,5385]` chứa 0, character CI cũng chứa 0 và McNemar `p=0,5488`. Formal gate thất bại; `promotion_eligible=false`, không có active registry và baseline tiếp tục được giữ.

Receipt one-shot có claim ID `5be5fc30f01ed0ff899befbde1727062b93885e42d918c75a944c1ec2d1040f9`. Manifest SHA-256 là `d4b2f02c8e7ee3e6bfddb8bf5c1d44eab68a94f19c2aee14c5e3fe5f94385288`. Phase 9 không được chạy lại hoặc tune rồi tái xác nhận trên cùng 650 mẫu.

Phân tích mô tả cho thấy distribution shift: `existing_plate_crop` có net `+5`, còn `crop_source_bounding_box` net `-2`; bucket min-side `64–127` net `-3`. Các slice này không phải promotion gate. Phase 9 được chuyển thành development data cho Phase 10 domain-adaptive PPO; mọi kết luận tiếp theo cần holdout mới.

## Lý do mở Phase 9

Phase 8 đã chạy đúng một lần trên 504 plate crop mới. Baseline đạt `82,1429%`; consensus đạt `83,1349%`, tăng `+0,9921` điểm %, `6 fix / 1 break`. Character accuracy tăng `+0,3527` điểm %, nhưng exact CI 95% là `[0; +1,9841]` điểm % và McNemar `p=0,125`, nên candidate đã khóa không qua gate và không được promote.

Hậu kiểm chỉ dùng để thiết kế candidate tiếp theo cho thấy PPO seed 727 riêng lẻ đạt `84,3254%`, tăng `+2,1825` điểm %, `14 fix / 3 break`; exact CI 95% `[+0,5952; +3,7698]`, character delta `+0,5145` điểm % với CI hoàn toàn dương và McNemar `p=0,01273`. Nó qua đủ gate nếu xét riêng, nhưng không thể được promote từ Phase 8 vì luật đã khóa khi đó là consensus.

## Candidate đã khóa

Phase 9 khóa checkpoint seed 727 và toàn bộ threshold/teacher/disagreement guard hiện có, không train lại và không chỉnh threshold theo Phase 8. `prospective_policy.json` ghi SHA-256 checkpoint, action registry, nguồn quyết định hậu kiểm và yêu cầu một external holdout hoàn toàn mới.

## Holdout không cần review thủ công

Data contract dùng trực tiếp `extracted_character` làm ground truth:

- bắt đầu từ sheet row 733, bao gồm chính dòng 733;
- loại `review_status=rejected`;
- `accepted`, `corrected` và trạng thái chưa accept đều dùng `extracted_character`;
- không dùng `corrected_label` và không bắt accept lại;
- tự động loại normalized label từng xuất hiện trong historical/Phase 7/Phase 8;
- tự động loại toàn bộ source đã nằm trong queue Phase 8, ảnh trùng SHA-256, duplicate label, label ngoài `0-9A-Z` hoặc dài quá 12, file/crop lỗi;
- stable-hash chọn trước 650 mẫu hợp lệ, tối thiểu formal là 500;
- builder không chạy PARSeq/PPO và mọi artifact nằm trong `reinforcement_learning/phase_9_primary_ppo`.

## Promotion gate

Phase 9 chỉ active nếu one-shot fresh external đồng thời đạt:

1. exact delta so baseline dương;
2. cận dưới paired bootstrap CI 95% của exact lớn hơn 0;
3. character delta không âm;
4. net fixes dương;
5. McNemar exact hai phía `<0,05`.

Trước khi có kết quả fresh external, Phase 9 là prospective candidate, baseline vẫn là lựa chọn triển khai.
