# Reinforcement learning trong PARSeq ANPR pipeline

Tài liệu này là đặc tả kỹ thuật cho nhánh reinforcement learning (RL): thuật toán dùng để làm gì, state/action/environment được xây dựng ra sao, reward và các loại penalty nào thực sự tồn tại trong code, cách train/evaluate/runtime, cùng ý nghĩa của Phase 1–12.

Nguồn chuẩn để đối chiếu:

- Action/reward Phase 4–5: `../rl_restoration/actions.py`, `../rl_restoration/reward.py`.
- MDP/PPO Phase 5: `../rl_restoration/sequential_env.py`, `../rl_restoration/train_ppo.py`.
- Candidate-set PPO Phase 6–7: `phase_6_candidate_oof_ppo/model.py`, `phase_6_candidate_oof_ppo/train.py`.
- Compact action space: `phase_7_compact_multiscale_ppo/action_space.py`.
- Consensus/guard: `phase_8_consensus_ppo/evaluate.py`, `phase_12_guarded_replicated_ppo/selection.py`.
- PixelRL deblur độc lập: `../parseq_rl_deblur_data/rl_deblur/`.

## 1. RL giải quyết phần nào của hệ thống?

PARSeq nhận một ảnh biển số và trả về chuỗi ký tự. Cùng một crop, các phép zoom, phóng lớn, tách biển hai dòng, tăng tương phản hay khử nhiễu có thể làm một ký tự dễ đọc hơn nhưng cũng có thể phá ảnh vốn đã đúng. Bài toán RL chính là:

> Với các tín hiệu quan sát được mà không cần label ở runtime, chọn baseline hoặc một candidate view sao cho OCR cuối tốt hơn, đồng thời hạn chế sửa sai ảnh baseline vốn đã đúng.

Policy **không** trực tiếp sửa trọng số PARSeq trong Phase 5–12 và **không** ghép tùy ý nhiều filter. Mỗi action là một view hoàn chỉnh được tạo lại từ crop gốc. PARSeq chạy trên view đó; policy chọn prediction cuối.

Có ba khái niệm cần phân biệt:

- **Baseline**: action 0, view tham chiếu. Nếu policy không đủ chắc chắn, hệ thống trả về baseline.
- **Teacher**: mạng hồi quy reward từ dữ liệu development; nó cung cấp prior cho PPO, không phải ground-truth oracle.
- **Oracle**: chọn candidate tốt nhất bằng label thật. Oracle chỉ đo trần của action space, không thể deploy.

## 2. Toàn bộ tiến trình Phase 1–12

| Phase | Vai trò | Có phải RL? | Kết quả/ý nghĩa |
|---|---|---|---|
| 1 | Tạo 65 multi-scale views và chọn bằng consensus đã khóa | Không | Chứng minh nhiều view có thể tăng OCR mà không đổi trọng số PARSeq |
| 2 | Học calibrated pairwise selector trên candidate Phase 1 | Supervised selector, không phải policy-gradient RL | Tăng test exact lên `94,1606%`, nhưng vẫn cần 65 OCR calls |
| 3 | Fine-tune PARSeq bằng controlled augmentation | Supervised learning | Không vượt checkpoint cha; epoch 0 được giữ để tránh regression |
| 4 | Offline trajectory cache + contextual-bandit reward router + hard-aware fine-tune | Có, dạng contextual bandit một bước | Policy 10 action; reward bám OCR; test chỉ tăng nhỏ |
| 5 | Actor–critic PPO hai bước với teacher prior và GAE | Có | Nhánh revise xuất hiện trên validation nhưng không tăng thêm exact test so bandit |
| 6 | Candidate-aware PPO, self-attention, OOF teacher, internal group holdout | Có | Quan sát đủ 10 candidate; tăng validation nhưng chưa qua formal gate |
| 7 | Thay 10 restoration actions bằng 9 compact multi-scale views | Có | Giữ validation oracle của 65 views với ít candidate hơn; các run vẫn chưa đủ bằng chứng promote |
| 8 | Đồng thuận hai PPO seed 727/728 | Không train mới; ensemble safety rule | Fresh 504: `6 fix / 1 break`, gate không đạt; seed 727 posthoc chỉ dùng mở Phase 9 |
| 9 | Xác nhận one-shot PPO seed 727 đã khóa | Không train mới | Fresh 650: `7 fix / 4 break`, CI/McNemar không đạt |
| 10 | Domain-adaptive candidate PPO dùng Phase 9 đã mở làm development | Có train lại | Hai seed gây net harm trên internal holdout, bị loại |
| 11 | Xác nhận one-shot PPO seed 728 | Không train mới | Fresh 1.500: `16 fix / 6 break`; CI dương nhưng McNemar `p=0,05248`, không promote |
| 12 | Guard metadata cho seed 728 | Không train mới; safety wrapper | Chỉ cho PPO sửa `existing_plate_crop` có min-side `<128`; đang chờ fresh data |

Phase 1–3 tạo action/data foundation. Phase 4 là bandit. Phase 5–7 là phần PPO thực sự. Phase 8–12 chủ yếu là xác nhận, chống harm, chống leakage và kiểm soát promotion; không nên gọi tất cả chúng là “các lần train PPO mới”.

## 3. Phase 1–3: nền tảng trước RL

### 3.1. Phase 1 — 65-view multi-scale TTA

`../preprocessing_best_config/benchmark_multiscale_tta.py` tạo:

- 1 baseline;
- 40 full views: `upscale in {2,3}` × `zoom in {0,85; 0,93; 1,00; 1,07; 1,15}` × 4 preprocessor;
- 24 unwrap views: `upscale in {2,3}` × `zoom in {0,93; 1,00; 1,07}` × 4 preprocessor.

Bốn preprocessor là `train_baseline`, `clahe_clip1_tile4`, `clahe_rl_deblur_bilateral`, `adaptive_noise_3way`. Unwrap chỉ thực sự tách hai dòng khi ảnh đủ cao và aspect ratio `<1,9`.

Phase 1 chỉ inference. Tham số consensus được chọn trên validation rồi khóa trước test. Nó xác nhận action space đa view có oracle tốt, nhưng chi phí 65 OCR calls/ảnh quá lớn.

### 3.2. Phase 2 — calibrated candidate selector

Phase 2 gom 65 hàng prediction thành các candidate string duy nhất, tạo feature gồm vote, confidence đã calibration, độ tin cậy view/context, fuzzy consensus, khoảng cách tới baseline, pattern biển số, prior chiều dài/shape; sau đó fit pairwise logistic regression theo Group K-fold trên validation. Selector chỉ đổi khỏi kết quả Phase 1 nếu chênh rank score vượt `switch_margin` đã khóa.

Đây là supervised ranking, không phải PPO. Ý nghĩa của Phase 2 là chứng minh “chọn candidate” quan trọng hơn chỉ chọn một view đơn, đồng thời đặt baseline accuracy để RL cạnh tranh.

### 3.3. Phase 3 — controlled augmentation

Phase 3 fine-tune PARSeq trên train với zoom, affine/perspective nhẹ, unwrap có điều kiện, low-resolution, blur/noise/JPEG/photometric và các preprocessor đã kiểm soát. Validation/test không augmentation và dùng manifest khóa.

Mọi epoch fine-tuned đều thấp hơn baseline validation, nên checkpoint epoch 0 được giữ. Kết luận đúng của Phase 3 là **không có weight improvement**, không phải augmentation đã tăng accuracy.

## 4. Phase 4 — contextual bandit restoration

### 4.1. Môi trường offline

Mỗi ảnh train/validation được chạy trước qua mọi action và PARSeq. `../rl_restoration/build_trajectory_cache.py` lưu prediction, edit distance, exact flag, confidence, action cost, feature và reward. Vì transition/reward đã nằm trong cache, quá trình học không gọi lại OCR và là **offline RL**.

Contextual bandit có một quyết định:

```text
context của ảnh baseline -> chọn 1 trong 10 action -> nhận reward OCR đã cache -> kết thúc
```

State dùng cho router là feature label-free:

- 9 image-quality features: log width/height, aspect ratio, brightness, contrast, log sharpness, noise, saturation, dark fraction;
- PARSeq encoder memory: mean-pooling và max-pooling;
- 6 OCR uncertainty features: prediction length, mean/min top-1 probability, mean entropy, mean top1–top2 margin, normalized log-confidence.

Label và edit distance chỉ dùng để tạo target reward trong cache; chúng không nằm trong state runtime.

### 4.2. Action space 10 view

Mỗi action bắt đầu từ ảnh gốc; không có chuyện action sau CLAHE tiếp tục deblur lần nữa ngoài pipeline đã định nghĩa của chính view.

| ID | Action | Biến đổi | Cost |
|---:|---|---|---:|
| 0 | `stop_baseline` | `train_baseline`, giữ tham chiếu | 0,000 |
| 1 | `raw_rgb` | Không enhancement | 0,002 |
| 2 | `clahe_gentle` | `clahe_clip1_tile4` | 0,005 |
| 3 | `homomorphic` | Sửa chiếu sáng không đều | 0,007 |
| 4 | `rl_bilateral` | RL deblur nhẹ + bilateral low-pass | 0,010 |
| 5 | `clahe_rl_bilateral` | CLAHE + RL deblur + bilateral | 0,012 |
| 6 | `adaptive_noise` | Noise router 3 nhánh đã khóa | 0,008 |
| 7 | `up2_baseline` | Upscale 2× rồi baseline | 0,006 |
| 8 | `up2_clahe` | Upscale 2× rồi CLAHE | 0,009 |
| 9 | `unwrap_up2_adaptive` | Unwrap hai dòng, upscale 2×, adaptive noise | 0,015 |

Upscale chỉ áp dụng khi ảnh nhỏ hơn ngưỡng `(width >= 128 và height >= 64)`. `stop_baseline` bắt buộc là action đầu và cost bằng 0; `validate_action_space()` kiểm tra invariant này.

### 4.3. Reward và penalty chính xác

Đặt:

- `d(p,y)` là Levenshtein distance giữa prediction `p` và target `y`;
- `A(p,y) = 1 - d(p,y) / max(len(p), len(y), 1)`;
- `E(p,y) = 1` nếu exact match, ngược lại `0`;
- `p0` là prediction baseline, `pa` là prediction của action `a`;
- `c(a)` là action cost ở bảng trên.

Reward trong `../rl_restoration/reward.py` là:

```text
R(a) = 1,0 * [A(pa,y) - A(p0,y)]
     + 0,5 * [E(pa,y) - E(p0,y)]
     - 1,0 * c(a)
     - 0,5 * I(E(p0,y)=1 và E(pa,y)=0)
```

Ý nghĩa từng phần:

- Edit-gain thưởng cho giảm số ký tự sai, kể cả chưa exact.
- Exact-gain thưởng `+0,5` khi action sửa một mẫu sai thành đúng; phạt `-0,5` khi làm mất exact.
- Action cost luôn bị trừ để ưu tiên view rẻ/baseline khi gain tương đương.
- Harm penalty trừ thêm `0,5` nếu baseline đúng nhưng action làm sai. Vì vậy một exact break chịu cả exact loss, harm penalty, edit loss nếu có và action cost.
- Reward có thể âm. Không có clipping reward trong cache.

Lưu ý: `A()` của reward chuẩn hóa edit distance bằng `max(len(prediction), len(target))`, trong khi metric tổng hợp `char_acc` của evaluator dùng tổng edit distance chia tổng độ dài target. Hai đại lượng gần nhau nhưng không đồng nhất.

### 4.4. Reward router

`RewardRouter` là MLP dự đoán vector reward của 10 action. Loss train gồm:

- Smooth-L1 regression cho reward từng action;
- ranking cross-entropy với target distribution `softmax(reward / 0,08)`;
- sample weight tăng `+2` nếu oracle khác baseline và `+1` nếu baseline đang sai.

Tại inference, router tìm action có predicted reward lớn nhất nhưng chỉ rời baseline nếu predicted gain so với baseline đạt selection margin. Checkpoint chọn theo thứ tự ưu tiên: exact accuracy, character accuracy, net fixes, ít broken hơn, mean cost thấp hơn.

## 5. Phase 5 — PPO hai bước

### 5.1. MDP và transition

Episode dài tối đa hai quyết định:

```text
s0 (baseline observation)
  |-- a0 = 0 (STOP) -> terminal, giữ baseline
  `-- a0 != 0       -> s1 quan sát OCR của view a0
                         |-- a1 = a0 -> accept
                         |-- a1 = 0  -> rollback baseline
                         `-- a1 khác -> revise sang view khác
```

Action cuối đúng bằng `a1`; code không chain phép biến đổi. Nếu revise, view mới vẫn được lấy từ cache đã tạo trực tiếp từ crop gốc.

Observation tại mỗi bước gồm:

- base feature chuẩn hóa;
- teacher reward prior của 10 action;
- OCR observation của action hiện tại: normalized confidence, độ dài, cờ khác baseline, one-hot chuỗi tối đa 12 vị trí trên alphabet `0-9A-Z`;
- one-hot action hiện tại;
- step flag;
- tùy ablation, summary của toàn bộ candidate: confidence, length, changed, vote fraction, group confidence, distance tới baseline.

Observation runtime không chứa target, exact hay edit distance.

### 5.2. Reward theo bước

- Nếu chọn STOP ở bước 0, baseline reward bằng 0 vì baseline có cost 0 và gain 0.
- Nếu đi tới bước 1, terminal reward là `R(a1)` ở phần 4.3.
- Nếu `a1 != a0`, trừ `revisit_cost = 0,002`.
- Việc mở bước thứ hai đưa `-step_cost = -0,001` vào TD/GAE delta của bước 0.

Do đó penalty tổng ngoài reward OCR là:

```text
transition penalty = 0,001 khi mở bước 2
revision penalty   = 0,002 khi action cuối khác action đầu
```

### 5.3. Actor–critic và teacher prior

Actor–critic là MLP hai tầng ẩn. Actor residual được cộng vào logits teacher prior:

```text
logits_policy(s) = logits_residual(s) + prior_scale * teacher_prior(s)
```

Khi có prior, output layer actor được khởi tạo bằng 0, nên policy ban đầu chính xác là teacher; PPO chỉ học residual. Critic dự đoán state value.

### 5.4. PPO/GAE objective

Với probability ratio `r_t = exp(log pi_new - log pi_old)` và advantage đã chuẩn hóa:

```text
L_policy = -mean(min(r_t * A_t,
                     clip(r_t, 1-epsilon, 1+epsilon) * A_t))

L_total  = L_policy
         + value_coef * SmoothL1(V(s_t), return_t)
         - entropy_coef * H(pi(.|s_t))
```

Default Phase 5:

- `gamma=0,99`, `gae_lambda=0,95`;
- `clip_ratio=0,10`;
- `value_coef=0,5`, `entropy_coef=0,002`;
- learning rate `5e-6`, 4 update epochs, rollout 8.192;
- hard sample weight `+6` khi oracle khác baseline;
- error sample weight `+3` khi baseline sai;
- margins được sweep trên validation, test không được loader của trainer đọc.

PPO seed 123 chọn được 8 fix/0 break trên validation nhưng không tăng exact test so với bandit. Điều này dẫn đến Phase 6: cần quan sát đầy đủ mọi candidate thay vì chỉ view hiện tại.

## 6. Phase 6 — candidate-aware OOF PPO

### 6.1. Vì sao đổi kiến trúc?

Phase 5 khó revise vì state bước hai không mô tả đủ các action thay thế. Phase 6 cache feature cho cả 10 views và dùng self-attention trên candidate set.

Mỗi candidate có:

- PARSeq encoder mean/max pooling + 6 OCR uncertainty;
- 9 image-quality features;
- normalized confidence, length, changed-vs-baseline, vote fraction, group confidence, label-free distance tới baseline prediction;
- one-hot OCR string theo vị trí, tối đa 12 ký tự trên alphabet 36 ký tự.

Raw encoder+quality cache có 783 chiều/view trong run lịch sử. OCR-string/consensus feature được nối thêm khi train mặc định.

### 6.2. Mạng candidate set

Luồng mạng:

```text
[candidate feature, teacher prior, step]
  -> shared projection + action embedding
  -> TransformerEncoder 2 layer, 4 heads
  -> per-candidate actor residual
  -> global/current-action critic
```

Actor residual cuối được zero-init và cộng `prior_scale * teacher_prior`. Teacher là MLP dự đoán reward vector từ state baseline.

### 6.3. OOF teacher và chống leakage

- Train lịch sử được stable-split theo normalized target/label thành development và internal group holdout.
- Cùng label không được xuất hiện ở hai phần.
- Development teacher prior được sinh 5-fold OOF theo group; mỗi hàng nhận prediction từ model không train trên group của chính nó.
- Full teacher chỉ dùng tạo prior cho validation/internal holdout sau khi cấu hình đã cố định.
- Internal holdout không tham gia teacher, BC, PPO, checkpoint hay margin selection.

Teacher loss gồm Smooth-L1 reward regression + ranking loss. Weight tăng `+5` nếu oracle khác baseline và `+2` nếu baseline sai.

### 6.4. Selection guards

Sau logits PPO, policy áp dụng lần lượt:

1. **First margin**: chỉ rời baseline khi logit gain của action tốt nhất so với baseline đủ lớn.
2. **Revise margin**: ở bước hai chỉ đổi action nếu logit gain so với action hiện tại đủ lớn.
3. **Teacher disagreement guard**: nếu final action khác lựa chọn teacher nhưng actor residual không ủng hộ đủ `disagreement_margin`, quay về action teacher.
4. **Final teacher-gain guard** (tùy chọn, Phase 10): action khác baseline chỉ được giữ nếu teacher dự đoán gain so baseline đạt threshold; nếu không rollback baseline.

Reward cơ sở vẫn là công thức OCR ở phần 4.3; step/revisit penalty và PPO loss giống Phase 5. Phase 6 chỉ thay observation, actor/critic và protocol OOF.

## 7. Phase 7 — compact multi-scale PPO

Phase 7 dùng lại candidate-set PPO nhưng thay action space bằng 9 view được greedy set-cover trên validation lịch sử. Chín view giữ 388/397 trường hợp oracle đúng của 65-view search, giảm số OCR view `86,2%` so với 65.

| ID | View | Zoom | Upscale | Unwrap | Preprocessor | Cost |
|---:|---|---:|---:|:---:|---|---:|
| 0 | `baseline` | 1,00 | 1× | Không | `train_baseline` | 0,000 |
| 1 | `unwrap_z1.07_up2_clahe_rl_deblur_bilateral` | 1,07 | 2× | Có | CLAHE + RL deblur + bilateral | 0,012 |
| 2 | `full_z0.93_up3_clahe_clip1_tile4` | 0,93 | 3× | Không | CLAHE nhẹ | 0,007 |
| 3 | `unwrap_z1.00_up3_train_baseline` | 1,00 | 3× | Có | baseline | 0,010 |
| 4 | `full_z0.85_up2_adaptive_noise_3way` | 0,85 | 2× | Không | adaptive noise | 0,008 |
| 5 | `full_z1.00_up2_clahe_rl_deblur_bilateral` | 1,00 | 2× | Không | CLAHE + RL deblur + bilateral | 0,009 |
| 6 | `unwrap_z0.93_up3_adaptive_noise_3way` | 0,93 | 3× | Có | adaptive noise | 0,013 |
| 7 | `full_z1.15_up3_train_baseline` | 1,15 | 3× | Không | baseline | 0,006 |
| 8 | `full_z1.15_up2_train_baseline` | 1,15 | 2× | Không | baseline | 0,004 |

Candidate raw feature gồm 774 chiều PARSeq/OCR + 7 metadata (zoom, normalized upscale, unwrap, one-hot 4 preprocessor) = 781 chiều/view trong cache lịch sử. Mặc định trainer nối thêm OCR string/consensus features.

Quan trọng: runtime Phase 7 tạo đủ 9 view trước khi policy chọn. `action_cost` là regularizer/proxy của action cuối, không phải latency thực. Runtime summary đo riêng thời gian tạo view, policy, mean/p95 và throughput.

## 8. Phase 8–12 — policy rule, external gate và promotion

### 8.1. Phase 8 consensus

Hai checkpoint seed 727 và 728 chạy độc lập. Rule khóa là:

```text
nếu cả hai policy tạo cùng một non-baseline prediction:
    dùng action của policy B (seed 728)
ngược lại:
    baseline
```

Đây là prediction agreement, không yêu cầu hai policy chọn cùng action. Candidate lock giữ SHA-256 hai checkpoint, action registry, rule và external contract. Fresh 504 mẫu không qua formal gate, nên Phase 8 không active.

### 8.2. Phase 9 primary seed 727

Posthoc trên holdout Phase 8 cho thấy seed 727 tốt, nhưng holdout đã mở nên không được promote. Phase 9 khóa nguyên checkpoint, teacher/margin/disagreement guard và mở holdout mới 650 mẫu đúng một lần. Kết quả không đủ ý nghĩa thống kê; baseline tiếp tục active.

### 8.3. Phase 10 domain adaptation

Phase 9 đã mở được đổi vai trò thành development data. 80% dùng adaptation, 20% validation; ghép với Phase 7 sau khi loại 144 label group overlap. Hai candidate seed 1001/1002 gây `net_fixes < 0` trên internal holdout, nên bị loại trước external confirmation.

### 8.4. Phase 11 replicated seed 728

Seed 728 được chọn từ evidence Phase 8+9 đã mở rồi khóa trước fresh holdout 1.500 mẫu. Nó đạt exact/character CI dương, `16 fix / 6 break`, nhưng McNemar hai phía `p=0,0524788` không nhỏ hơn 0,05. Không có active registry.

### 8.5. Phase 12 guarded seed 728

Guard runtime hoàn toàn label-free:

```text
allowed = input_transform == "existing_plate_crop"
          và min(crop_width, crop_height) < 128

final_action = ppo_action nếu allowed, ngược lại baseline (ID 0)
```

`crop_width` và `crop_height` phải dương. Equality 128 không được phép vì điều kiện là `<128`. Crop tạo từ detector bbox có `input_transform=crop_source_bounding_box`, vì vậy luôn fallback baseline trong Phase 12.

Development evidence gộp Phase 8/9/11 là `27 fix / 4 break`, nhưng đây là rule được chọn sau khi các tập đã mở; Phase 12 bắt buộc cần holdout mới. Pool hiện chưa đủ minimum 500/target 1.500.

## 9. Formal statistical gate

Một candidate chỉ `eligible` khi **tất cả** điều kiện sau đúng trên locked confirmatory holdout:

1. Paired bootstrap mean exact delta `> 0`.
2. Cận dưới CI 95% của exact delta `> 0`.
3. Mean character delta `>= 0`.
4. `net_fixes = fixed - broken > 0`.
5. Exact two-sided McNemar `p < 0,05`.

Trong đó:

- `fixed`: baseline sai, candidate đúng;
- `broken`: baseline đúng, candidate sai;
- bootstrap dùng cặp kết quả trên cùng ảnh, mặc định 10.000 resamples;
- McNemar chỉ dùng các cặp discordant `fixed + broken`.

Ngoài thống kê, evaluator/promoter còn kiểm tra:

- input contract là plate crop;
- normalized-label group không trùng historical/opened data;
- minimum sample count;
- test cũ không được load;
- checkpoint/action/manifest/cache SHA-256 đúng candidate lock;
- evaluation role là `locked_confirmatory`, không phải diagnostic;
- receipt one-shot hợp lệ và evaluation chưa từng chạy;
- output/active registry không bị ghi đè.

Fail một điều kiện thì không promote. Không được chỉnh threshold hoặc chọn slice trên holdout đã mở rồi đánh giá lại như confirmatory.

## 10. Train và chạy nhánh chính

### 10.1. Build Phase 4 cache

```powershell
python rl_restoration\build_trajectory_cache.py --split train --output-dir outputs\reproduction\trajectory_cache
python rl_restoration\build_trajectory_cache.py --split val   --output-dir outputs\reproduction\trajectory_cache
```

Cache cần PARSeq checkpoint và manifest Phase 3. Truyền `--checkpoint`, `--manifest`, `--refine-iters`, `--device` nếu không dùng default.

### 10.2. Train contextual-bandit router

```powershell
python rl_restoration\train_router.py `
  --cache-dir outputs\reproduction\trajectory_cache `
  --seed 123 `
  --output-dir outputs\reproduction\router_seed_123
```

### 10.3. Train PPO Phase 5

```powershell
python rl_restoration\train_ppo.py `
  --cache-dir outputs\reproduction\trajectory_cache `
  --teacher-router outputs\reproduction\router_seed_123\best_reward_router.pt `
  --seed 123 `
  --output-dir outputs\reproduction\ppo_seed_123
```

Trainer chỉ chấp nhận `train,val`; nếu `cache-splits` chứa `test` sẽ raise lỗi.

### 10.4. Train Phase 6

```powershell
python reinforcement_learning\run_phase.py phase6-cache --split train --output-dir reinforcement_learning\phase_6_candidate_oof_ppo\results\reproduction_cache
python reinforcement_learning\run_phase.py phase6-cache --split val   --output-dir reinforcement_learning\phase_6_candidate_oof_ppo\results\reproduction_cache
python reinforcement_learning\run_phase.py phase6 `
  --candidate-cache reinforcement_learning\phase_6_candidate_oof_ppo\results\reproduction_cache `
  --output-dir reinforcement_learning\phase_6_candidate_oof_ppo\results\reproduction_seed_123
```

`--trajectory-cache` mặc định trỏ tới cache Phase 4; truyền rõ nếu dùng cache khác.

### 10.5. Train Phase 7

```powershell
python reinforcement_learning\run_phase.py phase7-cache --split train --output-dir reinforcement_learning\phase_7_compact_multiscale_ppo\results\reproduction_cache
python reinforcement_learning\run_phase.py phase7-cache --split val   --output-dir reinforcement_learning\phase_7_compact_multiscale_ppo\results\reproduction_cache
python reinforcement_learning\run_phase.py phase7 `
  --trajectory-cache reinforcement_learning\phase_7_compact_multiscale_ppo\results\reproduction_cache `
  --candidate-cache reinforcement_learning\phase_7_compact_multiscale_ppo\results\reproduction_cache `
  --seed 727 `
  --output-dir reinforcement_learning\phase_7_compact_multiscale_ppo\results\reproduction_seed_727
```

Không dùng default output đã khóa nếu `summary.json` đã tồn tại; trainer cố ý từ chối ghi đè.

### 10.6. External evaluation đúng protocol

Ví dụ luồng Phase 7:

```powershell
python reinforcement_learning\run_phase.py phase7-cache `
  --split external_holdout `
  --manifest path\fresh_external_manifest.csv `
  --output-dir reinforcement_learning\phase_7_compact_multiscale_ppo\results\fresh_external_cache `
  --preflight

python reinforcement_learning\run_phase.py phase7-cache `
  --split external_holdout `
  --manifest path\fresh_external_manifest.csv `
  --output-dir reinforcement_learning\phase_7_compact_multiscale_ppo\results\fresh_external_cache

python reinforcement_learning\run_phase.py phase7-external --help
```

Preflight không load model, không OCR và không ghi artifact. Formal set dưới 500 mẫu bị từ chối; `--allow-underpowered-diagnostic` chỉ dành cho diagnostic không promotable.

Phase 8–12 có thêm lock/receipt riêng. Luôn chạy `phase8-status` hoặc `phase12-status` trước và đọc README của phase. Không chạy lại evaluator trên manifest đã có receipt.

### 10.7. Runtime

Phase 5 nghiên cứu:

```powershell
python rl_restoration\ppo_runtime.py path\plate_crop.png
```

Phase 7 label-free yêu cầu manifest chỉ có `image_path` và active registry. Phase 8 yêu cầu thêm `input_contract=plate_crop`. Phase 12 yêu cầu `image_path,input_contract,input_transform`; runtime đọc kích thước ảnh thật, kiểm tra metadata nếu có và chỉ tạo đủ 9 views cho mẫu qua guard.

Hiện không có active registry, nên runtime formal Phase 7–12 từ chối chạy là hành vi đúng.

## 11. Nhánh PixelRL/A2C deblur độc lập

Nhánh này nằm ngoài thư mục phase chính: `../parseq_rl_deblur_data/rl_deblur`. Nó không chọn một trong 9/10 view; mỗi pixel là một agent dùng chung FCN policy.

### 11.1. Environment

- State: ảnh grayscale `(B,H,W)` trong `[0,255]`, canvas 128×32.
- Episode: cố định 5 bước theo default.
- Policy: FCN actor–critic với dilation `1,2,3,4,3,2,1`; actor/value có trunk riêng.
- RMC: kernel 9×9 học được, softmax-normalized, truyền spatial return giữa pixel lân cận.
- Dataset: ảnh plate crop sạch được làm mờ tổng hợp Gaussian, motion hoặc defocus; ảnh sạch là ground truth.

### 11.2. Per-pixel actions

| ID | Action |
|---:|---|
| 0 | giữ nguyên pixel |
| 1 | pixel `+1` |
| 2 | pixel `-1` |
| 3 | pixel `+3` |
| 4 | pixel `-3` |
| 5 | Gaussian smooth 3×3, sigma 0,5 |
| 6 | Bilateral filter `d=5, sigmaColor=25, sigmaSpace=5` |
| 7 | Unsharp mild, amount 0,5 |
| 8 | Unsharp strong, amount 1,5 |

Toàn bộ candidate operation được tính trên ảnh hiện tại, sau đó mỗi pixel lấy giá trị từ operation mà action map của pixel đó chọn.

### 11.3. Reward/penalty PixelRL

Dense pixel reward:

```text
r_pixel(t) = [(state_t - clean)^2 - (state_t+1 - clean)^2] / 255^2
```

Giảm squared error cho reward dương; làm ảnh xa clean hơn tự động nhận reward âm. Không có action-cost/harm-penalty như nhánh OCR view selector.

Nếu bật OCR shaping, mỗi bước cộng một scalar cho toàn ảnh:

```text
r_ocr(t) = beta  * [CER_t - CER_t+1]
         + gamma * [logconf_t+1 - logconf_t]

r_total(t,pixel) = r_pixel(t,pixel) + r_ocr(t)
```

Trong CLI, `beta = --cer-reward-weight`, `gamma = --logconf-reward-weight`; cả hai mặc định `0`. CER tăng hoặc log-confidence giảm tạo penalty âm. Frozen PARSeq chỉ đo reward, không được update.

A2C loss gồm policy loss, `0,5 * value loss`, `0,1 * RMC loss` và entropy regularization với hệ số mặc định `0,0002`. Terminal bootstrap bằng 0 vì episode kết thúc thật sau T bước. Return được lan truyền bằng:

```text
R_t = r_t + discount * RMC(R_t+1), discount mặc định 0,95
```

Chạy từ thư mục `parseq_rl_deblur_data`:

```powershell
python -m rl_deblur.make_dataset --seed 42
python -m rl_deblur.train --epochs 15 --num-steps 5 --device cuda
python -m rl_deblur.evaluate --device cuda
```

Nhánh này từng cho thấy PSNR/SSIM tốt hơn không đảm bảo OCR tốt hơn; đó là lý do reward OCR tùy chọn được bổ sung và nhánh Phase 4–12 chuyển trọng tâm sang OCR-grounded reward.

## 12. Dữ liệu, split và vai trò từng tập

| Tập | Vai trò |
|---|---|
| Train 3.270 | Cache reward, teacher/PPO development |
| Validation 397 | Chọn checkpoint, epoch, margin trong các phase lịch sử |
| Test 411 | Audit khóa Phase 1–5; hiện đã mở, không còn là fresh holdout |
| Phase 6/7 internal group holdout | Đánh giá nội bộ, group-disjoint với development cùng run |
| External Phase 8: 504 | One-shot đã mở, giờ chỉ là development evidence |
| External Phase 9: 650 | One-shot đã mở, giờ chỉ là development evidence |
| External Phase 11: 1.500 | One-shot đã mở, giờ chỉ là development evidence |
| Phase 12 fresh | Chưa đủ dữ liệu; target 1.500, formal minimum 500 |

“Group-disjoint” nghĩa là normalized target/label không xuất hiện ở cả hai phía. Đây là yêu cầu mạnh hơn split theo file ảnh, vì nhiều ảnh có thể mang cùng biển số.

## 13. Metric và cách đọc report

- `exact_acc`: tỷ lệ prediction trùng toàn bộ target.
- `char_acc`: `1 - total_edit_distance / total_target_characters` trong evaluator.
- `fixed`: baseline sai, policy đúng.
- `broken`: baseline đúng, policy sai.
- `net_fixes`: fixed trừ broken.
- `stop_rate`/`baseline_rate`: tỷ lệ action cuối bằng 0.
- `revise_rate`: tỷ lệ action cuối khác action đầu ở episode active.
- `mean_cost`: mean cost của action cuối, không phải wall-clock.
- `oracle_exact`: exact nếu được dùng label chọn candidate tốt nhất; chỉ là upper bound.

Thứ tự chọn checkpoint nội bộ thường là exact → character → net fixes → ít broken → cost thấp. Promotion bên ngoài không dùng thứ tự này; nó bắt buộc pass toàn bộ gate phần 9.

## 14. Artefact và provenance

- `RESULTS_SUMMARY.csv`: số liệu tổng hợp.
- `ARTIFACT_INDEX.md`: đường dẫn checkpoint/cache/report.
- `prospective_policy.json`: candidate, rule, hash và contract khóa trước holdout.
- `protocol.json`: seed, split/group policy, vai trò dữ liệu.
- `summary.json`: metric, paired statistics, gate và promotion status.
- `fresh_locked_confirmatory_receipt.json`: claim/completion của one-shot evaluation.
- `active_policy.json`: chỉ tồn tại sau promotion hợp lệ; runtime formal tin vào registry này.

Không sửa các receipt/manifest lịch sử để “làm cho đường dẫn chạy được”. Với run mới, tạo manifest/output mới, ghi seed/CLI/hash và giữ nguyên sau khi inference external bắt đầu.

## 15. Kiểm thử

```powershell
python -m unittest reinforcement_learning.phase_6_candidate_oof_ppo.test_phase6 reinforcement_learning.phase_7_compact_multiscale_ppo.test_phase7 reinforcement_learning.phase_8_consensus_ppo.test_phase8 reinforcement_learning.phase_9_primary_ppo.test_phase9 reinforcement_learning.phase_12_guarded_replicated_ppo.test_phase12
```

Checkout hiện tại đã chạy `38 tests` và đạt `OK`. Test bao phủ split/group rules, paired statistics, action/selection guard, promotion validation, one-shot receipt và runtime contract; nó không thay thế full inference test với model/dataset.
