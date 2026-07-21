# Báo cáo phương pháp Phase 1–12 trong pipeline PARSeq ANPR

## 1. Mục đích và phạm vi

Báo cáo này mô tả cụ thể vai trò của từng Phase 1–12 trong nhánh `reinforcement_learning`, phương pháp được dùng, tác dụng đối với hệ thống và quan hệ của từng phase với các họ reinforcement learning (RL) trong repository.

Điểm quan trọng nhất là Phase 1–12 không phải 12 bước của một thuật toán RL duy nhất. Đây là lộ trình nghiên cứu theo thời gian:

```text
Xác minh multi-view có tiềm năng
        ↓
Xây selector không-RL
        ↓
Học contextual bandit một bước
        ↓
Học PPO hai bước
        ↓
Mở rộng PPO để quan sát toàn bộ candidate
        ↓
Rút gọn action space
        ↓
Kiểm định, chống harm và kiểm soát promotion
```

Nguồn tổng quát: [`reinforcement_learning/README.md`](reinforcement_learning/README.md), [`RESULTS_SUMMARY.csv`](reinforcement_learning/RESULTS_SUMMARY.csv).

## 2. Các họ phương pháp trong repository

### 2.1. Họ PixelRL: A2C + Reward Map Convolution

- Vị trí: `parseq_rl_deblur_data/rl_deblur`.
- Đơn vị quyết định: từng pixel.
- Policy: fully convolutional actor–critic.
- Action: giữ nguyên, tăng/giảm cường độ, Gaussian, bilateral, unsharp và các phép chỉnh pixel khác.
- Reward chính: mức giảm sai số pixel so với ảnh clean; có thể cộng reward OCR.
- Kết quả đầu ra: một ảnh đã được phục hồi.

Họ này độc lập với chuỗi Phase 1–12. Một số action ở Phase 4–7 có thể sử dụng pipeline chứa `rl_deblur`, nhưng các phase đó chỉ chọn view; chúng không huấn luyện lại PixelRL.

### 2.2. Họ offline contextual bandit

- Phase đại diện: Phase 4.
- Đơn vị quyết định: một view hoàn chỉnh cho toàn ảnh.
- Episode: một quyết định rồi kết thúc.
- Policy: MLP dự đoán reward của từng action.
- Cách học: reward regression kết hợp ranking trên offline trajectory cache.
- Mục tiêu: chọn view cải thiện OCR và tránh làm hỏng baseline.

Đây là một formulation RL dạng bandit, nhưng implementation không dùng policy gradient. Việc tối ưu policy gần với supervised reward modelling trên reward đã cache.

### 2.3. Họ actor–critic PPO chọn candidate

- Phase huấn luyện: Phase 5, 6, 7 và 10.
- Episode: tối đa hai bước, cho phép `STOP`, accept, revise hoặc rollback.
- Thuật toán: clipped PPO, actor–critic, value loss, entropy regularization và GAE.
- Reward: cải thiện edit accuracy/exact match OCR, trừ action cost và penalty làm hỏng baseline.
- Mục tiêu: học quyết định tuần tự thay vì chỉ chọn một action một lần.

Họ PPO có hai kiến trúc:

1. **MLP PPO** ở Phase 5: quan sát baseline hoặc candidate hiện tại.
2. **Candidate-aware PPO** ở Phase 6, 7 và 10: quan sát đồng thời toàn bộ candidate bằng self-attention/Transformer, kết hợp teacher prior.

Phase 8, 9, 11 và 12 sử dụng checkpoint candidate-aware PPO đã train trước đó nhưng không huấn luyện một policy RL mới.

## 3. Bảng phân loại Phase 1–12

| Phase | Phương pháp chính | Có train RL mới? | Họ RL | Có dùng policy RL đã có? | Vai trò chính |
|---:|---|:---:|---|:---:|---|
| 1 | 65-view multi-scale TTA + consensus | Không | Không-RL | Không | Chứng minh nhiều view có oracle và gain thực tế |
| 2 | Calibrated supervised candidate selector | Không | Không-RL | Không | Học chọn prediction từ 65 view |
| 3 | Controlled augmentation + fine-tune PARSeq | Không | Không-RL | Không | Thử cải thiện trực tiếp trọng số OCR |
| 4 | Offline reward router một bước | Có, theo nghĩa bandit | Contextual bandit | Không | Chọn 1 trong 10 restoration view |
| 5 | MLP actor–critic PPO hai bước | Có | PPO | Dùng bandit làm teacher prior | Cho phép accept/revise/rollback |
| 6 | Candidate-aware PPO + OOF teacher | Có | PPO | Kế thừa reward/action Phase 4–5 | Quan sát đủ candidate và giảm leakage |
| 7 | Compact multi-scale candidate PPO | Có | PPO | Kế thừa kiến trúc Phase 6 | Đổi sang 9 view có oracle mạnh, chi phí thấp hơn 65 view |
| 8 | Consensus giữa PPO seed 727/728 | Không | Wrapper trên PPO | Có | Giảm harm do độ nhạy seed |
| 9 | One-shot confirmation PPO seed 727 | Không | Evaluation của PPO | Có | Xác nhận prospective trên holdout mới |
| 10 | Domain-adaptive candidate PPO | Có | PPO | Khởi nguồn từ Phase 7/9 | Thử thích nghi domain mới và thêm safety gate |
| 11 | Replicated confirmation PPO seed 728 | Không | Evaluation của PPO | Có | Kiểm tra tái lập trên 1.500 group mới |
| 12 | Metadata guard cho PPO seed 728 | Không | Safety wrapper trên PPO | Có | Chỉ cho PPO sửa nhóm ảnh có lợi dự kiến |

Nếu chỉ đếm phase **thực sự cập nhật một policy RL**, đó là Phase 4, 5, 6, 7 và 10. Nếu yêu cầu policy-gradient RL đúng nghĩa, chỉ Phase 5, 6, 7 và 10 sử dụng policy gradient; Phase 4 học reward router bằng regression/ranking.

## 4. Baseline và nguyên tắc đánh giá chung

Baseline là view tham chiếu dùng preprocessing `train_baseline`. Trong action space Phase 4–6 nó có tên `stop_baseline`; trong Phase 7 trở đi nó có tên `baseline`. Baseline luôn là action 0 và có action cost bằng 0.

Policy chỉ có giá trị nếu sửa được nhiều ảnh baseline sai hơn số ảnh baseline đúng bị làm sai:

- `fixed`: baseline sai, policy đúng;
- `broken`: baseline đúng, policy sai;
- `net_fixes = fixed - broken`.

Các gate promotion sau này yêu cầu đồng thời:

1. exact-match delta dương;
2. paired confidence interval của exact loại 0;
3. character accuracy không giảm;
4. net fixes dương;
5. McNemar hai phía nhỏ hơn ngưỡng đã khóa;
6. provenance, manifest, checkpoint và one-shot receipt hợp lệ.

Do dataset, crop contract và checkpoint có thể khác nhau giữa các phase, không nên so trực tiếp hai tỷ lệ accuracy nếu chúng không được tính trên cùng manifest và cùng checkpoint OCR.

## 5. Báo cáo cụ thể từng phase

### Phase 1 — 65-view multi-scale TTA và consensus

**Loại phương pháp:** không-RL.

**Mục tiêu:** kiểm tra liệu việc tạo nhiều phiên bản của cùng một crop có giúp PARSeq đọc đúng thêm biển số hay không, trước khi đầu tư vào policy học máy.

**Cách thực hiện:**

- Tạo 65 view từ tổ hợp zoom, upscale, full/unwrap hai dòng và bốn preprocessing pipeline.
- Chạy cùng một checkpoint PARSeq trên tất cả view.
- Gộp prediction bằng consensus đã khóa trên validation.
- Dùng oracle có label chỉ để đo trần action space, không dùng khi deploy.

**Tác dụng:**

- Chứng minh lỗi OCR phụ thuộc mạnh vào cách tạo view.
- Tạo candidate pool cho Phase 2 và dữ liệu nền để thiết kế action space Phase 7.
- Chỉ thay đổi inference, không cập nhật trọng số PARSeq.

**Kết quả lịch sử:** validation exact `94,7103%`, test exact `93,9173%`, cao hơn baseline tương ứng `92,6952%` và `91,9708%`.

**Giới hạn:** cần 65 lần OCR cho mỗi ảnh và consensus vẫn có thể làm hỏng ảnh baseline vốn đúng.

Nguồn: [`phase_1_multiscale_tta/REPORT.md`](reinforcement_learning/phase_1_multiscale_tta/REPORT.md).

### Phase 2 — Calibrated candidate selector

**Loại phương pháp:** supervised ranking/selection, không-RL.

**Mục tiêu:** chọn prediction tốt hơn từ 65 view thay vì dùng consensus thuần túy.

**Cách thực hiện:**

- Gộp các view tạo cùng một candidate string.
- Dùng feature vote, confidence đã calibration, độ tin cậy lịch sử của view, fuzzy consensus, khoảng cách tới baseline, pattern và shape prior.
- Fit pairwise logistic ranker theo Group K-fold.
- Chỉ chuyển khỏi kết quả Phase 1 nếu score gain vượt `switch_margin` đã khóa.

**Tác dụng:** chứng minh phần chọn candidate là một nút thắt độc lập với khả năng nhận dạng của PARSeq.

**Kết quả lịch sử:**

- Validation OOF exact `95,2141%`.
- Test exact `94,1606%`.
- So với baseline test: `11 fix / 2 break`.
- So với Phase 1 test: `1 fix / 0 break`.

**Giới hạn:** vẫn cần chạy đủ 65 OCR view; đây không phải policy RL và không tối ưu reward tuần tự.

Nguồn: [`phase_2_calibrated_selector/REPORT.md`](reinforcement_learning/phase_2_calibrated_selector/REPORT.md).

### Phase 3 — Controlled augmentation và fine-tune PARSeq

**Loại phương pháp:** supervised learning, không-RL.

**Mục tiêu:** kiểm tra liệu có thể chuyển gain từ multi-view inference thành cải thiện trực tiếp trọng số PARSeq hay không.

**Cách thực hiện:** fine-tune với augmentation có kiểm soát như zoom, affine/perspective nhẹ, blur/noise/JPEG, low-resolution, unwrap và preprocessing; mỗi ảnh nhận tối đa hai degradation.

**Tác dụng:** xây pipeline training an toàn và phân tích nhóm lỗi tiny/two-line/sequence-length.

**Kết quả:** không epoch fine-tuned nào vượt checkpoint cha trên validation khóa. Cơ chế bảo vệ giữ epoch 0, nên checkpoint cuối không bị regression nhưng cũng không có weight improvement.

**Kết luận:** gain quan sát được đến từ inference/selection Phase 1–2, không phải từ trọng số Phase 3.

Nguồn: [`phase_3_controlled_augmentation/REPORT.md`](reinforcement_learning/phase_3_controlled_augmentation/REPORT.md).

### Phase 4 — Offline contextual-bandit restoration

**Loại phương pháp:** họ contextual bandit.

**Mục tiêu:** giảm từ 65 view xuống một quyết định chi phí thấp, đồng thời tối ưu trực tiếp mục tiêu OCR.

**Môi trường:**

```text
context của baseline → chọn 1 trong 10 action → nhận reward → terminal
```

Tất cả action được chạy trước trên train/validation. Cache lưu prediction, edit distance, exact flag, confidence, action cost, feature và reward; vì vậy lúc train router không cần gọi lại OCR.

**State label-free:**

- đặc trưng chất lượng ảnh;
- mean/max pooling của PARSeq encoder;
- độ dài prediction, top-1 probability, entropy, margin và log-confidence.

**Action:** 10 view hoàn chỉnh như baseline, raw RGB, CLAHE, homomorphic, RL-deblur/bilateral, adaptive noise, upscale và unwrap. Mỗi view được tạo lại từ crop gốc; policy không chain filter tự do.

**Reward:**

```text
R(a) = edit-accuracy gain
     + 0,5 × exact-match gain
     - action cost
     - 0,5 × harm(baseline đúng nhưng action sai)
```

**Policy:** `RewardRouter` MLP dự đoán vector reward của 10 action; loss gồm Smooth-L1 reward regression và ranking loss.

**Tác dụng:** tạo baseline RL một bước, teacher prior cho Phase 5–7 và cơ chế reject-to-baseline theo margin.

**Kết quả lịch sử:** router test có `5 fix / 3 break` so baseline trước cập nhật trọng số; one-step oracle còn cao hơn đáng kể, cho thấy selector chưa khai thác hết action space. Hệ thống router kết hợp checkpoint hard-aware đạt `92,7007%` exact trong report Phase 4–5, nhưng bằng chứng chưa đủ để kết luận cải thiện ổn định trên mọi phân phối.

Nguồn: [`RL_PHASE4_EXPERIMENT_REPORT.md`](reinforcement_learning/RL_PHASE4_EXPERIMENT_REPORT.md), [`rl_restoration/README.md`](rl_restoration/README.md).

### Phase 5 — MLP PPO hai bước

**Loại phương pháp:** họ PPO, kiến trúc MLP.

**Mục tiêu:** khắc phục giới hạn một quyết định của bandit bằng cách cho policy quan sát kết quả action đầu rồi chấp nhận, sửa lại hoặc rollback.

**MDP:**

```text
s0 baseline
 ├─ STOP → terminal baseline
 └─ chọn view a0 → s1 quan sát OCR view a0
                    ├─ a1 = a0 → accept
                    ├─ a1 = 0  → rollback
                    └─ a1 khác → revise
```

Action cuối vẫn là một view hoàn chỉnh tạo từ crop gốc, không phải chuỗi filter.

**Policy:** MLP actor–critic hai tầng. Actor residual được cộng vào teacher prior từ router Phase 4; actor head zero-init nên policy bắt đầu đúng từ teacher.

**Tối ưu:** behavior cloning trước, sau đó clipped PPO + value loss + entropy regularization; advantage dùng GAE. Reward kế thừa Phase 4, cộng step cost và revisit cost.

**Tác dụng:** đưa khả năng revise/rollback vào policy và kiểm tra xem quyết định tuần tự có vượt bandit hay không.

**Kết quả:** seed 123 vượt contextual bandit trên validation mà không làm hỏng baseline, nhưng trên test chỉ giữ nguyên exact tốt nhất của bandit. Kết hợp checkpoint PARSeq hard-aware đạt `92,7007%` exact và `98,9577%` character accuracy. Chưa có bằng chứng PPO tăng exact test so với router một bước.

Nguồn: [`RL_PHASE5_PPO_REPORT.md`](reinforcement_learning/RL_PHASE5_PPO_REPORT.md), [`rl_restoration/train_ppo.py`](rl_restoration/train_ppo.py).

### Phase 6 — Candidate-aware PPO với OOF teacher

**Loại phương pháp:** họ PPO, kiến trúc candidate-set Transformer.

**Lý do mở phase:** Phase 5 chỉ mô tả rõ candidate hiện tại nên policy thiếu thông tin để chọn candidate thay thế ở bước revise.

**Thay đổi chính:**

- Cache feature cho cả 10 candidate.
- Shared projection và action embedding.
- Transformer encoder/self-attention giữa candidate.
- Actor residual theo từng candidate và critic dùng thông tin global/current action.
- Teacher prior được tạo OOF theo normalized-label group.
- Internal group holdout không tham gia teacher, BC, PPO, checkpoint hay margin selection.

**Tác dụng:** policy có thể so sánh trực tiếp các candidate, revise có cơ sở hơn và giảm leakage từ teacher.

**Kết quả:**

- Validation: teacher `93,4509%`, PPO `93,9547%`, tăng `0,5038` điểm %, revise `6,80%`.
- Holdout khóa: PPO và teacher cùng `93,6735%` exact.
- Exact CI `[-1,6327; +1,4286]`, McNemar `p=1,0`, net fixes so baseline `-1`.

**Trạng thái:** không vượt formal gate; checkpoint không thay Phase 5.

Nguồn: [`phase_6_candidate_oof_ppo/PHASE6_REPORT.md`](reinforcement_learning/phase_6_candidate_oof_ppo/PHASE6_REPORT.md).

### Phase 7 — Compact multi-scale candidate PPO

**Loại phương pháp:** cùng họ và cùng kiến trúc PPO Phase 6; thay action space.

**Mục tiêu:** kết hợp oracle mạnh của 65-view Phase 1 với chi phí nhỏ hơn.

**Thay đổi chính:** dùng greedy set-cover trên validation lịch sử để chọn 9 compact multi-scale view. Các view kết hợp zoom, upscale, unwrap và bốn preprocessing; giữ `388/397` trường hợp oracle đúng của search 65 view và giảm số OCR call `86,2%` so với 65 view.

**Observation/policy:** candidate-aware self-attention, OCR-string feature, consensus feature, OOF teacher prior và disagreement guard.

**Tác dụng:** tăng oracle/action diversity, giúp PPO tạo gain nhất quán hơn trên các internal group holdout.

**Kết quả chính:**

- Seed 725: exact tăng `+1,0687` điểm %, `9 win / 2 loss`, paired exact CI dương nhưng McNemar `p=0,0654`.
- Seed 727 và 728 vẫn có delta dương nhưng không đồng thời qua CI và McNemar.
- Corrected external diagnostic cho seed 727 đạt `6 fix / 1 break`, seed 728 đạt `7 fix / 1 break`; đây là diagnostic sau sửa input contract, không promotable.

**Trạng thái:** experimental, không có active registry.

Nguồn: [`phase_7_compact_multiscale_ppo/PHASE7_REPORT.md`](reinforcement_learning/phase_7_compact_multiscale_ppo/PHASE7_REPORT.md), [`action_space.py`](reinforcement_learning/phase_7_compact_multiscale_ppo/action_space.py).

### Phase 8 — Conservative consensus PPO

**Loại phương pháp:** không train RL mới; safety ensemble trên hai PPO Phase 7.

**Mục tiêu:** giảm harm và độ nhạy seed bằng rule bảo thủ.

**Rule:** chỉ thay baseline nếu PPO seed 727 và 728 tạo cùng một prediction khác baseline; nếu không đồng thuận thì rollback baseline. Hai policy không bắt buộc chọn cùng action, chỉ cần prediction cuối giống nhau.

**Tác dụng:** giảm change rate, giữ các trường hợp hai policy đồng thuận và loại bớt thay đổi thiếu ổn định.

**Kết quả:**

- Validation cũ: `3 fix / 0 break`.
- External protocol-repair diagnostic: `6 fix / 0 break`, qua gate nhưng không promotable vì dữ liệu đã mở trong quá trình sửa protocol.
- Fresh locked-confirmatory 504 mẫu: baseline `82,1429%`, consensus `83,1349%`, `6 fix / 1 break`; exact CI còn chạm 0 và McNemar `p=0,125`.

**Trạng thái:** formal gate thất bại; baseline tiếp tục active. Hậu kiểm seed 727 chỉ được dùng để thiết kế Phase 9, không được promote hậu nghiệm.

Nguồn: [`phase_8_consensus_ppo/PHASE8_REPORT.md`](reinforcement_learning/phase_8_consensus_ppo/PHASE8_REPORT.md).

### Phase 9 — Primary PPO prospective confirmation

**Loại phương pháp:** không train RL mới; one-shot evaluation checkpoint PPO seed 727.

**Mục tiêu:** kiểm tra riêng seed 727 trên một holdout hoàn toàn mới, vì seed này cho kết quả hậu kiểm tốt ở Phase 8 nhưng không thể được promote từ holdout đã mở.

**Protocol:** khóa checkpoint, teacher, threshold, disagreement guard và action registry trước khi mở 650 plate crop mới; không tune lại theo Phase 8.

**Kết quả:**

- Baseline exact `46,1538%`.
- PPO exact `46,6154%`.
- `7 fix / 4 break`, net `+3`.
- Exact delta `+0,4615` điểm %, CI `[-0,4615; +1,5385]`.
- McNemar `p=0,5488`.

**Tác dụng:** phát hiện distribution shift theo input transform và tạo development data hợp lệ cho Phase 10 sau khi holdout đã mở.

**Trạng thái:** không promote; không được chạy lại hoặc tune trên cùng 650 mẫu.

Nguồn: [`phase_9_primary_ppo/PHASE9_REPORT.md`](reinforcement_learning/phase_9_primary_ppo/PHASE9_REPORT.md).

### Phase 10 — Domain-adaptive candidate PPO

**Loại phương pháp:** train RL mới, cùng họ candidate-aware PPO.

**Mục tiêu:** thích nghi policy với distribution mới quan sát ở Phase 9 và kiểm tra safety gate dựa trên teacher gain.

**Dữ liệu/protocol:** chuyển 650 mẫu Phase 9 đã mở thành development; stable-split 520 adaptation/130 validation. Audit phát hiện 144 label group overlap trong dữ liệu Phase 7 và loại 509 dòng train để đưa mixed train/validation overlap về 0.

**Tác dụng:** thử domain adaptation, đồng thời làm rõ và sửa một vấn đề group leakage lịch sử.

**Kết quả:**

- Seed 1001: internal holdout `7 fix / 10 break`, net `-3`.
- Seed 1002 có final teacher-gain safety gate: validation `9/0`, nhưng internal holdout `8/13`, net `-5`.

**Kết luận:** policy overfit validation; teacher prior chưa đủ calibration để bảo vệ baseline.

**Trạng thái:** cả hai checkpoint bị loại, không có candidate external.

Nguồn: [`phase_10_domain_adaptive_ppo/PHASE10_REPORT.md`](reinforcement_learning/phase_10_domain_adaptive_ppo/PHASE10_REPORT.md).

### Phase 11 — Replicated primary PPO seed 728

**Loại phương pháp:** không train RL mới; prospective replication checkpoint PPO seed 728.

**Mục tiêu:** kiểm tra seed 728 trên holdout lớn hơn sau khi nó cho evidence dương trên dữ liệu Phase 8+9 đã mở.

**Protocol:** khóa checkpoint trước evaluation; dùng 1.500 unique label/ảnh không trùng các tập đã mở; one-shot receipt và power contract được lưu để chống chạy lại/tune hậu nghiệm.

**Kết quả:**

| Policy | Exact | Character | Fix/break |
|---|---:|---:|---:|
| Baseline | 47,2000% | 82,8013% | 0/0 |
| PPO seed 728 | 47,8667% | 83,5529% | 16/6 |

Exact delta `+0,6667` điểm %, CI `[+0,0667; +1,2667]`; character CI cũng dương. Tuy nhiên McNemar `p=0,0524788`, vừa cao hơn ngưỡng `<0,05`.

**Tác dụng:** cung cấp replication mạnh hơn và chỉ ra một nhóm metadata có tỷ lệ lợi/hại tốt hơn để khóa guard Phase 12.

**Trạng thái:** không promote; baseline vẫn là policy triển khai.

Nguồn: [`phase_11_replicated_primary_ppo/PHASE11_REPORT.md`](reinforcement_learning/phase_11_replicated_primary_ppo/PHASE11_REPORT.md).

### Phase 12 — Guarded replicated PPO

**Loại phương pháp:** không train RL mới; metadata safety wrapper trên PPO seed 728.

**Mục tiêu:** chỉ cho PPO tác động trong vùng dữ liệu nơi development evidence cho thấy tỷ lệ fix/break thuận lợi.

**Rule prospective:** chỉ commit action PPO khi:

```text
input_transform = existing_plate_crop
và min(width, height) < 128
```

Các trường hợp khác rollback baseline. Ở runtime, mẫu không qua guard chỉ cần chạy baseline view; mẫu eligible mới chạy đủ chín candidate.

**Development evidence:** trên ba external đã mở, guarded rule đạt lần lượt `7/1`, `6/1`, `14/2`, tổng `27 fix / 4 break`; đây chỉ là evidence dùng để khóa rule, không phải confirmation có thể promote.

**Tác dụng:** biến insight hậu kiểm thành rule label-free có thể áp dụng tại runtime và giảm phạm vi policy được phép thay baseline.

**Trạng thái hiện tại trong report:** cần 1.500 group mới cho power contract nhưng pool mới chỉ có 235 candidate trước media validation; chưa đủ dữ liệu để mở formal holdout. Phase 12 chưa được promote.

Nguồn: [`phase_12_guarded_replicated_ppo/PHASE12_REPORT.md`](reinforcement_learning/phase_12_guarded_replicated_ppo/PHASE12_REPORT.md).

## 6. Quan hệ kế thừa giữa các phase

```text
Phase 1: 65-view search
   ├─ Phase 2: supervised selector trên 65 view
   └─ Phase 7: chọn 9 compact view từ search space

Phase 3: thử cải thiện trọng số PARSeq, không thành công

Phase 4: contextual-bandit reward router
   └─ teacher prior + reward/action contract
        ↓
Phase 5: MLP PPO hai bước
        ↓ observation chưa đủ candidate
Phase 6: candidate-aware PPO + OOF teacher
        ↓ action space cần oracle mạnh hơn
Phase 7: compact multi-scale candidate PPO
   ├─ Phase 8: consensus seed 727/728
   ├─ Phase 9: xác nhận seed 727
   ├─ Phase 10: domain-adaptive PPO
   └─ Phase 11: xác nhận seed 728
          ↓
       Phase 12: metadata guard cho seed 728
```

## 7. Kết luận tổng hợp

### Theo họ phương pháp

- **Không-RL:** Phase 1, 2 và 3 xây action/data foundation và các baseline cần thiết.
- **Contextual bandit:** Phase 4 là policy một bước, chi phí thấp và đóng vai trò teacher cho PPO.
- **MLP PPO:** Phase 5 thêm quyết định tuần tự nhưng chưa chứng minh gain test so bandit.
- **Candidate-aware PPO:** Phase 6, 7 và 10 cải thiện observation/action space; Phase 10 cho thấy rủi ro overfit domain.
- **Sử dụng PPO nhưng không train mới:** Phase 8, 9, 11 và 12 là consensus, confirmation hoặc safety guard.
- **PixelRL:** nằm ngoài Phase 1–12; là mô hình phục hồi pixel độc lập và chỉ có thể xuất hiện bên trong một số preprocessing action.

### Theo trạng thái triển khai

Các phase sau đã nhiều lần tạo delta dương, nhưng formal promotion yêu cầu đồng thời effect size, paired CI, character accuracy, net fixes, McNemar và provenance. Phase 8, 9 và 11 đều thiếu ít nhất một điều kiện; Phase 10 bị loại vì net harm; Phase 12 chưa đủ fresh data. Vì vậy theo trạng thái được ghi trong các report hiện tại, chưa có active registry Phase 7–12 và baseline vẫn là lựa chọn triển khai an toàn.

### Bài học phương pháp

1. Có candidate tốt không đồng nghĩa policy sẽ chọn đúng candidate.
2. PSNR/SSIM tốt hơn không bảo đảm OCR exact-match tốt hơn, nên reward phải bám downstream OCR.
3. PPO phức tạp hơn bandit chỉ có ý nghĩa khi observation chứa đủ thông tin và action space có oracle gap đủ lớn.
4. Kết quả validation dương không đủ để promote; cần fresh group-disjoint holdout và kiểm định ghép cặp.
5. Safety wrapper/consensus/guard không phải thuật toán RL mới, nhưng là phần thiết yếu để hạn chế `broken` khi triển khai.
