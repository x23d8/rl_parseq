# Báo cáo Phase 5 — PPO Restoration Agent

## Kết luận

PPO đã được triển khai thành công dưới dạng policy hai bước, có actor–critic, clipped surrogate objective, value loss, entropy regularization và GAE. Policy seed 123 vượt contextual bandit một ảnh trên validation, không làm hỏng ảnh baseline và được khóa trước khi đánh giá test.

Trên test, PPO giữ nguyên exact-match tốt nhất của contextual bandit thay vì tăng thêm. Khi kết hợp với checkpoint PARSeq hard-aware, hệ thống đạt 92,7007% exact và 98,9577% character accuracy. Vì vậy PPO đã thành công về thuật toán và validation, nhưng chưa có bằng chứng rằng nó cải thiện exact-match test so với router một bước.

## MDP hai bước

- State: đặc trưng encoder PARSeq, OCR uncertainty, chất lượng ảnh, teacher reward prior, action hiện tại, OCR string của view hiện tại và chỉ báo bước.
- Bước 0: chọn baseline hoặc một restoration view hoàn chỉnh.
- Bước 1: quan sát kết quả OCR trung gian rồi chọn action cuối.
- Chọn lại action đầu là accept; chọn `stop_baseline` là rollback; chọn action khác là revise.
- Mọi view bắt đầu từ ảnh gốc. Không chain deblur/CLAHE/denoise.
- Reward: edit-accuracy gain, exact-match gain, harm penalty, action cost và revisit cost.

## Huấn luyện

- Teacher prior: contextual bandit seed 123 đã khóa.
- PPO seed được thử: 42, 123 và 20260715.
- Rollout: 8.192 transition episodes mỗi epoch.
- Learning rate: `5e-6`.
- Clip ratio: `0.10`.
- Entropy coefficient: `0.002`.
- GAE: gamma `0.99`, lambda `0.95`.
- Checkpoint tốt nhất: seed 123, epoch 22.
- Margin bước đầu: `0.1`; revise margin: `0.0`.
- Test không được dùng để chọn seed, epoch hoặc margin.

## Kết quả policy trên validation

| Policy | Exact | Character accuracy | Fixed | Broken |
|---|---:|---:|---:|---:|
| Baseline preprocessing | 92,6952% | 98,6524% | 0 | 0 |
| Contextual bandit | 94,4584% | 98,8974% | 7 | 0 |
| PPO seed 42 | 94,4584% | 98,8974% | 7 | 0 |
| PPO seed 123 | **94,7103%** | **98,9280%** | **8** | **0** |
| PPO seed 20260715 | **94,7103%** | **98,9280%** | **8** | **0** |
| One-step oracle | 94,9622% | 99,2956% | — | — |

PPO seed 123 được chọn vì cùng accuracy với seed 20260715 nhưng có mean action cost thấp hơn và có sử dụng nhánh revise trên validation.

## Kết quả test khóa

| Cấu hình | Exact | Character accuracy |
|---|---:|---:|
| Parent PARSeq, baseline view | 91,9708% | 98,8684% |
| Parent PARSeq, contextual bandit | 92,4574% | 98,9279% |
| Parent PARSeq, PPO | 92,4574% | 98,9279% |
| PARSeq hard-aware, PPO | **92,7007%** | **98,9577%** |
| One-step oracle với parent | 95,8637% | 99,4640% |

PPO và bandit chọn khác nhau 42/411 ảnh test, nhưng không có trường hợp nào thay đổi trạng thái đúng/sai. Nhánh revise không được kích hoạt trên test. Điều này cho thấy action-observation thứ hai còn thiếu khả năng dự đoán action thay thế nào sẽ có lợi.

### So sánh với Phase 2 multi-scale

Phase 2 selector trước đó đạt 94,1606% exact trên test bằng 65 candidate views và một checkpoint OCR khác. Con số này vẫn cao hơn PPO 10-action, nhưng không phải so sánh đồng chi phí hoặc đồng checkpoint: PPO thường chỉ chạy một view, còn Phase 2 chạy 65 view. Vì vậy PPO hiện là kết quả RL tốt nhất trong action space chi phí thấp đã khóa, chưa phải accuracy cao nhất của toàn bộ repository.

Một ablation cho PPO quan sát consensus của cả 10 candidate đã được thử với ba mức BC. Kết quả tốt nhất chỉ đạt 94,4584% validation, thấp hơn PPO chuẩn 94,7103%; cấu hình này bị loại vì vừa chậm hơn vừa kém chính xác hơn.

## Checkpoint và artefact

- PPO: `outputs/rl_restoration/ppo_prior_seed_123/best_ppo_restoration_policy.pt`.
- PARSeq fine-tuned bằng PPO view: `outputs/rl_restoration/parseq_ppo_hard_curriculum/best_parseq_rl_policy_mixture.pt`.
- Lịch sử PPO: `outputs/rl_restoration/ppo_prior_seed_123/ppo_history.csv`.
- PPO validation selections: `outputs/rl_restoration/ppo_prior_seed_123/val_ppo_selections.csv`.
- Test audit: `outputs/rl_restoration/test_locked_ppo_finetuned/summary.json`.
- Test action trace: `outputs/rl_restoration/test_locked_ppo_finetuned/test_locked_ppo_selections.csv`.
- Runtime ảnh mới: `rl_restoration/ppo_runtime.py`.

## Hướng cải thiện tiếp theo

Khoảng cách test đến oracle còn 3,16 điểm %. Muốn thu hẹp khoảng này cần mở rộng observation/action, không nên chỉ tăng số epoch PPO:

1. cache encoder/OCR observation cho mọi candidate view để critic đánh giá action thay thế tốt hơn;
2. dùng out-of-fold teacher predictions để giảm việc teacher đạt oracle trên train nhưng tổng quát hóa kém hơn;
3. tạo holdout mới trước mọi thay đổi tiếp theo vì test hiện tại đã được dùng cho audit;
4. chỉ kết luận PPO vượt test khi paired confidence interval và McNemar trên holdout mới xác nhận cải thiện.

## Cập nhật Phase 6

Bốn hạng mục trên đã được triển khai trong `phase_6_candidate_oof_ppo/`: cache đủ 10 candidate với 783 đặc trưng/view, candidate-set actor–critic, teacher 5-fold OOF, group holdout và statistical gate.

PPO residual tăng 0,5038 điểm % so với teacher trên validation và kích hoạt revise 6,80%, nhưng exact trên holdout khóa chỉ hòa teacher; CI chứa 0 và McNemar không có ý nghĩa. Vì vậy kết quả xác nhận observation mới giúp nhánh RL hoạt động tốt hơn trên validation, nhưng chưa đủ bằng chứng để thay checkpoint Phase 5. Xem `phase_6_candidate_oof_ppo/PHASE6_REPORT.md`.

## Cập nhật Phase 7

Action space đã được mở rộng bằng chín multi-scale view compact, giữ nguyên oracle validation 97,7330% của search space 65 view. Candidate token bổ sung OCR string/consensus label-free và teacher disagreement guard.

PPO cao hơn OOF teacher trên validation và trên cả ba group holdout nội bộ. Run seed 725 tăng holdout `+1,0687` điểm % với paired CI exact hoàn toàn dương, nhưng McNemar hai phía còn `p=0,0654`; các replication vẫn dương nhưng không qua đồng thời CI/McNemar. Do đó Phase 7 chứng minh đầu ra RL tăng nhất quán trong protocol nội bộ, song vẫn chưa đủ điều kiện tuyên bố vượt test hoặc promote checkpoint. Xem `phase_7_compact_multiscale_ppo/PHASE7_REPORT.md`.

## Cập nhật Phase 8

External audit phát hiện manifest 682 ảnh chứa 528 dòng trùng normalized-label group với dữ liệu lịch sử; tập sạch còn 154 ảnh. Ngoài ra 50 ảnh trong tập sạch ban đầu là ảnh toàn cảnh `cropped=false`, khiến toàn bộ baseline/oracle/RL cùng sai. Pipeline đã được gia cố để bắt buộc plate crop, group-disjoint và minimum formal sample count 500.

Trên corrected diagnostic, hai PPO OCR-guard seed 727/728 đều tăng exact so baseline. Phase 8 khóa rule bảo thủ: chỉ đổi restoration khi cả hai PPO cho cùng prediction khác baseline. Kết quả diagnostic đạt `75,9740%` so baseline `72,0779%`, `6 fix / 0 break`, nhưng không được promote vì tập đã mở.

Fresh locked-confirmatory sau đó dùng 504 plate crop mới. Consensus tăng từ `82,1429%` lên `83,1349%` (`6 fix / 1 break`), nhưng exact CI 95% còn chạm 0 và McNemar `p=0,125`, nên Phase 8 thất bại formal gate và baseline vẫn active. Hậu kiểm seed 727 riêng lẻ đạt `84,3254%`, `14 fix / 3 break`, exact delta `+2,1825` điểm %, CI hoàn toàn dương và McNemar `p=0,01273`; kết quả này chỉ dùng để khóa Phase 9, không được promote từ holdout Phase 8.

## Cập nhật Phase 9

Phase 9 khóa nguyên checkpoint PPO seed 727 cùng threshold/teacher/disagreement guard. Holdout 650 mẫu dùng trực tiếp `extracted_character`, không yêu cầu accept thủ công. PPO tăng `+0,4615` điểm % nhưng chỉ đạt `7 fix / 4 break`, exact CI chứa 0 và McNemar `p=0,5488`; không được promote. Xem `phase_9_primary_ppo/PHASE9_REPORT.md`.

## Cập nhật Phase 10–12

Phase 10 thử domain adaptation bằng cache Phase 9 và đồng thời sửa leakage 144 label group giữa train/validation Phase 7. Hai seed mới đều gây net harm trên internal group holdout nên bị loại; không có external candidate.

Phase 11 quay lại seed 728, được chọn nhờ replication dương trên Phase 8+9, rồi đánh giá one-shot trên 1.500 unique group mới. PPO tăng exact từ `47,20%` lên `47,8667%`, character từ `82,8013%` lên `83,5529%`, đạt `16 fix / 6 break`; cả hai paired CI hoàn toàn dương. McNemar `p=0,05248` vừa hụt ngưỡng `<0,05`, nên vẫn không promote.

Phase 12 đã khóa guard label-free: chỉ dùng action PPO seed 728 cho `existing_plate_crop` có min-side `<128`, trường hợp khác rollback baseline. Development evidence gộp Phase 8/9/11 đạt `27 fix / 4 break`, nhưng pool hiện chỉ còn 235 unique group chưa mở, thiếu 265 để đạt formal minimum và 1.265 để đạt target power 1.500. Xem `phase_12_guarded_replicated_ppo/PHASE12_REPORT.md`.
