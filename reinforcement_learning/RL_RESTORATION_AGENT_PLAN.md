# Kế hoạch thiết kế Restoration Agent bằng Reinforcement Learning cho PARSeq ANPR

## 1. Kết luận thiết kế

Có thể áp dụng ý tưởng của Restore-R1 vào bài toán này, nhưng không nên sao chép nguyên reward và action space của bài báo.

Restore-R1 tối ưu chất lượng cảm nhận của ảnh trong môi trường không có nhãn bằng DeQA-Score. Repo này có nhãn chuỗi biển số, do đó tín hiệu tốt hơn và sát mục tiêu hơn là:

- exact match;
- character accuracy hoặc normalized edit accuracy;
- mức độ làm hỏng các ảnh PARSeq vốn đã đọc đúng;
- chi phí và độ trễ của từng công cụ phục hồi.

Thiết kế được khuyến nghị là một agent nhẹ chỉ chọn công cụ xử lý ảnh. PARSeq và các mô hình phục hồi được đóng băng trong lúc huấn luyện policy. Không dùng RL để sinh pixel và không đồng thời cập nhật PARSeq trong giai đoạn học policy.

Lộ trình nên đi theo thứ tự:

1. contextual bandit hoặc supervised router chọn một pipeline;
2. behavior cloning từ oracle action;
3. agent tuần tự có hành động `STOP`, tối đa hai bước;
4. chỉ dùng PPO nếu oracle của chuỗi hai bước cao hơn rõ rệt so với router một bước.

## 2. Những gì kế thừa và thay đổi từ Restore-R1

| Thành phần | Restore-R1 | Thiết kế cho PARSeq ANPR |
|---|---|---|
| Mục tiêu | Chất lượng ảnh tổng quát | Đọc đúng toàn bộ biển số |
| State chính | Đặc trưng ảnh và lịch sử action | Đặc trưng PARSeq, thống kê OCR, đặc trưng chất lượng ảnh và lịch sử action |
| Reward | Chênh lệch DeQA-Score | Chênh lệch normalized edit accuracy và exact match |
| Nhãn | Không cần ground truth | Dùng nhãn chuỗi biển số khi training |
| Policy | Frozen CLIP ViT-L/14 và MLP | Frozen PARSeq encoder và MLP nhỏ |
| Tối ưu | PPO hoặc GRPO | Behavior cloning trước, PPO sau nếu cần |
| Inference | Chọn chuỗi tool, không search và rollback | Chọn pipeline hoặc chuỗi tối đa hai tool, không exhaustive search |

MLLM/DeQA-Score chỉ nên là reward phụ khi tận dụng ảnh không có nhãn. Nó không nên là reward chính vì ảnh có thể đẹp hơn nhưng nét của ký tự bị thay đổi.

## 3. Dư địa cải thiện đo từ kết quả hiện có

Phân tích các prediction đã lưu của tám phương pháp chung trên cùng manifest cho kết quả:

| Split | Baseline `train_baseline` | Phương pháp đơn tốt nhất | Oracle của 8 phương pháp | Ảnh baseline sai nhưng có phương pháp sửa được | Ảnh không phương pháp nào đọc đúng |
|---|---:|---:|---:|---:|---:|
| Validation, 397 ảnh | 92,70% | 93,95% | 94,71% | 8 | 21 |
| Test, 411 ảnh | 91,97% | 93,43% | 94,89% | 12 | 21 |

Tám phương pháp được so sánh gồm:

- `train_baseline`;
- `raw_rgb`;
- `clahe_clip1_tile4`;
- `homomorphic_filter`;
- `clahe_rl_deblur_bilateral`;
- `rl_deblur_bilateral_lowpass`;
- `zero_dce`;
- `restormer_motion_deblur_native`.

Các con số trên dẫn tới ba kết luận:

1. Router có tiềm năng tốt hơn một cấu hình cố định vì các phương pháp sửa các nhóm ảnh khác nhau.
2. Với action space hiện tại, trần exact match quan sát được trên test là 94,89%. Policy không thể vượt trần này nếu chỉ chọn một trong tám kết quả đã có.
3. Muốn vượt xa 94,89% cần tạo chuỗi action mới, thêm công cụ phù hợp hơn, cải thiện crop/rectification hoặc fine-tune PARSeq với ảnh đã phục hồi.

Oracle của riêng sáu pipeline nhanh đạt 94,46% trên validation và 94,89% trên test. Thêm Zero-DCE và Restormer chỉ sửa thêm một ảnh validation, không sửa thêm ảnh test nào ngoài những ảnh mà nhóm pipeline nhanh đã đọc đúng. Restormer không có unique fix trên cả hai split. Vì vậy hai model nặng không nên nằm trong action space mặc định của MVP; chỉ bật lại trong ablation hoặc khi có tập dữ liệu suy giảm nặng hơn.

Mức tăng hiện tại còn nhỏ và khoảng tin cậy của một số phương pháp vẫn chứa 0. Vì vậy mọi kết luận phải dùng kiểm định ghép cặp trên từng ảnh, không chỉ so sánh hai tỷ lệ trung bình.

## 4. Kiến trúc mục tiêu

```text
Ảnh đầu vào
    ↓
YOLO plate detector
    ↓
Crop + perspective rectification
    ↓
Ảnh biển số I0
    ├─ PARSeq encoder feature
    ├─ OCR logits và uncertainty feature
    └─ low-level image feature
              ↓
       Restoration Policy
      ┌───────┴────────┐
      ↓                ↓
    STOP          Chọn action
                       ↓
                Ảnh trạng thái mới
                       ↓
                 PARSeq OCR lại
                       ↓
               Tối đa 2 bước rồi STOP
```

Detector và perspective rectification nằm trước agent. Trong thí nghiệm đầu tiên, agent được đánh giá trên crop có sẵn để tách riêng ảnh hưởng đến OCR. Sau đó mới đánh giá end-to-end trên crop do detector tạo ra.

### 4.1. State

Tại bước `t`, state gồm:

```text
s_t = [z_image, z_ocr, z_quality, z_history, z_metadata]
```

Trong đó:

- `z_image`: mean-pooling và max-pooling token từ `PARSeq.encode(image)`;
- `z_ocr`: độ dài chuỗi, mean log-probability, minimum token probability, entropy trung bình, top-1/top-2 margin và prediction hiện tại;
- `z_quality`: brightness, contrast, Laplacian sharpness, noise residual, saturation, dark fraction và aspect ratio;
- `z_history`: multi-hot action đã gọi, action gần nhất và chỉ số bước;
- `z_metadata`: chiều rộng, chiều cao, detector confidence và plate type nếu có trong training.

Không dùng tích token probability hiện tại làm feature duy nhất vì giá trị này chưa được calibration và có thể cao ngay cả khi OCR sai.

### 4.2. Policy network

Phiên bản đầu nên dùng:

- frozen PARSeq encoder;
- MLP hai hoặc ba lớp, hidden dimension 128 hoặc 256;
- actor head xuất phân phối action;
- critic head xuất một scalar value khi chuyển sang PPO.

PARSeq encoder phù hợp hơn CLIP trong MVP vì đã được fine-tune trực tiếp cho nét và cấu trúc ký tự biển số. Có thể làm ablation với frozen CLIP sau, nhưng không cần thêm mô hình lớn ngay từ đầu.

## 5. Action space

### 5.1. Giai đoạn một: chọn pipeline hoàn chỉnh

Đây là contextual bandit: mỗi action luôn xử lý từ ảnh gốc và trả về kết quả cuối, không nối nhiều pipeline với nhau.

Action mặc định ban đầu:

```text
RAW
TRAIN_BASELINE
CLAHE_GENTLE
HOMOMORPHIC
RL_DEBLUR_BILATERAL
CLAHE_RL_DEBLUR_BILATERAL
```

Hai action ML được giữ ở tier thử nghiệm:

```text
ZERO_DCE
RESTORMER_MOTION_DEBLUR
```

Nếu bật action ML, chia thành hai tier:

- tier nhanh: sáu phương pháp classical;
- tier chậm: Zero-DCE và Restormer.

Policy chỉ gọi tier chậm khi lợi ích dự đoán lớn hơn chi phí. Restormer hiện chạy khoảng 5,4 ảnh/giây trong benchmark, chậm hơn nhiều so với các pipeline classical khoảng 200–300 ảnh/giây. Kết quả oracle hiện tại chưa chứng minh tier chậm đem lại lợi ích trên test.

### 5.2. Giai đoạn hai: action nguyên tử và chuỗi hai bước

Không được lấy các config hoàn chỉnh hiện tại rồi gọi nối tiếp tùy ý, vì một config có thể đã gồm grayscale, CLAHE, deblur và denoise. Nối hai config như vậy sẽ lặp xử lý và dễ phá nét ký tự.

Cần tách thành wrapper action nguyên tử:

```text
STOP
CLAHE_GENTLE
HOMOMORPHIC
RL_DEBLUR
BILATERAL_MILD
ZERO_DCE
RESTORMER_MOTION_DEBLUR
```

Giới hạn `T_max = 2` ở lần thử đầu. Dùng action mask để:

- không gọi lại cùng một action;
- không gọi action sau `STOP`;
- tránh chuỗi đã biết là dư thừa;
- bắt buộc dừng ở `T_max`;
- có thể cho phép `RL_DEBLUR → BILATERAL_MILD` để giảm ringing.

Chỉ tăng lên ba bước nếu oracle hai bước chứng minh còn thiếu và ảnh không bị overprocessing.

## 6. Reward OCR-aware

Với prediction `ŷ_t` và nhãn `y*`, định nghĩa:

```text
A_t = 1 - edit_distance(ŷ_t, y*) / max(len(ŷ_t), len(y*), 1)
E_t = 1 nếu ŷ_t == y*, ngược lại bằng 0
```

Reward khởi đầu:

```text
r_t = (A_{t+1} - A_t)
      + 0.5 × (E_{t+1} - E_t)
      - action_cost(a_t)
      - 0.5 × I[E_t = 1 và E_{t+1} = 0]
```

Ý nghĩa:

- normalized edit accuracy tạo tín hiệu dày hơn exact match;
- exact match bonus giữ đúng mục tiêu chính;
- penalty lớn bảo vệ các ảnh vốn đã đọc đúng;
- action cost giúp agent dừng sớm và hạn chế gọi mô hình nặng.

Giá trị ban đầu có thể dùng:

| Action | Cost khởi đầu |
|---|---:|
| `STOP` | 0,000 |
| Classical enhancement/denoise | 0,005 |
| Classical deblur | 0,010 |
| Zero-DCE | 0,020 |
| Restormer | 0,080 |

Các cost phải được chuẩn hóa lại theo latency đo trên đúng GPU triển khai.

Không cần thêm confidence vào reward khi có ground truth. Nếu vẫn muốn dùng, chỉ thêm confidence đã temperature-calibrate với trọng số rất nhỏ, tối đa khoảng 0,02, và phải theo dõi riêng trường hợp confidence tăng nhưng accuracy giảm.

## 7. Quy trình huấn luyện

### Giai đoạn 0 — Khóa protocol đánh giá

1. Giữ test không tham gia tạo action label, chọn feature, chọn reward hoặc threshold.
2. Chia dữ liệu theo nguồn/video/biển số thay vì chia ngẫu nhiên theo ảnh để tránh các frame gần giống nhau rơi vào nhiều split.
3. Tạo một calibration split riêng để học temperature cho logits PARSeq.
4. Vì validation và test hiện tại đã được xem nhiều lần trong quá trình thử preprocessing, nên tạo một holdout mới chưa dùng nếu mục tiêu là báo cáo nghiên cứu chính thức.

### Giai đoạn 1 — Tạo offline trajectory cache

Với từng ảnh train:

1. chạy PARSeq trên ảnh gốc;
2. chạy toàn bộ action một bước;
3. lưu ảnh hoặc hash ảnh, prediction, token probability, logits rút gọn, edit distance, exact match, latency và action cost;
4. xác định action có reward cao nhất;
5. nếu nhiều action bằng nhau, ưu tiên action rẻ nhất và ít thay đổi ảnh nhất.

Cache làm cho quá trình học policy nhanh, có thể tái lập và không phải chạy Restormer lặp lại trong mỗi epoch.

Để policy gặp đủ tình huống cần restoration, tạo thêm degradation tổng hợp chỉ từ ảnh train:

- motion blur và defocus blur với nhiều mức độ;
- Gaussian/Poisson noise;
- low-light, color cast và glare;
- JPEG compression và downscale-upscale;
- rain/haze nhẹ nếu dữ liệu triển khai thực sự có các lỗi này;
- hỗn hợp hai degradation theo nhiều thứ tự.

Luôn giữ một tỷ lệ ảnh không suy giảm để agent học `STOP` và tránh xử lý quá mức. Có thể dùng curriculum: một degradation trước, sau đó mới tới hỗn hợp hai degradation. Nhãn OCR không thay đổi sau phép degrade, nên vẫn tính được reward theo chuỗi ký tự mà không cần ảnh sạch làm reconstruction target.

Không tạo biến thể từ validation/test. Kết quả trên ảnh tổng hợp chỉ dùng để đo robustness; kết quả chính vẫn phải lấy trên ảnh thật và holdout chưa dùng.

### Giai đoạn 2 — Supervised router và contextual bandit

Huấn luyện hai baseline policy:

1. behavior cloning dự đoán best action;
2. reward regression dự đoán reward của từng action rồi chọn `argmax`.

Reward regression thường hữu ích hơn hard label khi nhiều action cho kết quả tương đương. Có thể dùng soft target:

```text
p*(a|s) = softmax(reward(a) / τ)
```

và tối ưu KL divergence hoặc cross-entropy.

Gate để đi tiếp:

- router phải tốt hơn fixed best trên validation;
- harm rate trên các ảnh baseline đúng không được tăng đáng kể;
- tốc độ phải đạt ngân sách triển khai;
- kết quả phải ổn định qua ít nhất ba seed.

### Giai đoạn 3 — Xây oracle chuỗi hai bước

Chạy exhaustive search offline cho các chuỗi hợp lệ dài tối đa hai. Nếu oracle chuỗi hai bước không cao hơn oracle một bước ít nhất khoảng 0,5 điểm phần trăm exact match, không nên dùng PPO; supervised router đã là lựa chọn hợp lý hơn.

### Giai đoạn 4 — Behavior cloning cho policy tuần tự

Dùng chuỗi tốt nhất từ offline search làm demonstration. Huấn luyện policy dự đoán action tiếp theo từ state hiện tại và history. Thêm các state trung gian, không chỉ state ảnh gốc.

### Giai đoạn 5 — PPO fine-tuning

Chỉ khởi động PPO từ checkpoint behavior cloning. Cấu hình ban đầu có thể bám bài Restore-R1:

- discount `γ = 0,99`;
- GAE `λ = 0,95`;
- PPO clip `ε = 0,2`;
- value coefficient `0,5`;
- entropy coefficient bắt đầu `0,01–0,05` rồi giảm dần;
- actor/critic learning rate nên thử `1e-4`, `3e-4`, không dùng ngay `0,01` của bài báo vì feature và dữ liệu của repo khác đáng kể.

Trong PPO:

- frozen PARSeq;
- frozen restoration tools;
- chỉ cập nhật actor và critic;
- rollout trên train;
- checkpoint theo validation exact match, sau đó character accuracy, harm rate và latency.

### Giai đoạn 6 — Fine-tune PARSeq có kiểm soát

Sau khi policy đã ổn định, có thể tạo tập train gồm ảnh gốc và ảnh do policy chọn để fine-tune PARSeq. Không joint-train PARSeq với PPO ở lần đầu vì reward sẽ trở thành non-stationary.

Tỷ lệ khởi đầu phù hợp:

```text
50% ảnh theo preprocessing dùng khi fine-tune hiện tại
25% ảnh gốc/raw
25% ảnh do restoration policy chọn
```

Giữ ảnh gốc trong train để tránh mô hình chỉ hoạt động tốt trên ảnh đã xử lý.

## 8. Cấu trúc mã nguồn đề xuất

```text
rl_restoration/
├── __init__.py
├── actions.py                 # wrapper action nguyên tử và action cost
├── environment.py             # reset, step, STOP, action mask
├── features.py                # PARSeq, OCR và quality features
├── reward.py                  # edit/exact/harm/cost reward
├── build_trajectory_cache.py  # chạy offline action và chuỗi action
├── dataset.py                 # đọc cache, group-aware split
├── policy.py                  # actor, critic, router
├── train_router.py            # behavior cloning/reward regression
├── train_ppo.py               # PPO từ checkpoint router
├── evaluate_policy.py         # exact, char, harm, latency, action trace
└── configs/
    ├── router.json
    └── ppo.json

outputs/rl_restoration/
├── trajectory_cache/
├── checkpoints/
├── predictions/
├── action_traces/
└── reports/
```

Không nên sửa trực tiếp logic benchmark hiện tại. `actions.py` có thể tái sử dụng hàm từ `preprocessing_best_config/preprocessing.py`, còn ML action dùng loader hiện có của benchmark official-model.

## 9. Thiết kế thí nghiệm và ablation

Các baseline bắt buộc:

1. `train_baseline`;
2. fixed best theo validation;
3. `adaptive_noise_3way` hiện tại;
4. random router có cùng action space;
5. oracle một bước;
6. supervised router chỉ dùng quality feature;
7. supervised router thêm PARSeq encoder feature;
8. supervised router thêm OCR uncertainty;
9. sequential behavior cloning;
10. behavior cloning và PPO.

Ablation reward:

- chỉ normalized edit accuracy;
- edit accuracy và exact bonus;
- thêm harm penalty;
- thêm action cost;
- thêm calibrated confidence;
- DeQA-Score phụ trên ảnh không nhãn.

Ablation horizon:

- một action;
- tối đa hai action;
- tối đa ba action.

## 10. Metric và tiêu chí thành công

Metric chính:

- exact match;
- character accuracy;
- CER;
- số ảnh baseline sai được sửa;
- số ảnh baseline đúng bị làm sai;
- net fixes = fixed images trừ broken images.

Metric triển khai:

- action trung bình trên mỗi ảnh;
- tỷ lệ `STOP` ngay;
- latency trung bình và p95;
- throughput;
- tỷ lệ gọi Zero-DCE/Restormer;
- ECE và reliability diagram của confidence đã calibration.

Metric phải báo cáo theo plate type, kích thước crop và nhóm suy giảm. Tuy nhiên nhóm blue/ngoại giao hiện có quá ít mẫu để kết luận thống kê; cần bổ sung dữ liệu trước khi so sánh chính thức theo loại biển.

Kiểm định:

- paired bootstrap confidence interval cho delta exact và delta character accuracy;
- McNemar test cho cặp đúng/sai giữa policy và baseline;
- ít nhất ba seed cho policy;
- không chọn checkpoint bằng test.

Tiêu chí tối thiểu để coi RL có hiệu quả:

```text
delta exact > 0 trên holdout mới
paired 95% CI không chứa 0
character accuracy không giảm
net fixes > 0
p95 latency nằm trong ngân sách triển khai
```

## 11. Rủi ro chính và cách kiểm soát

### Reward hacking

Agent có thể làm confidence hoặc chất lượng cảm nhận tăng nhưng đổi nét ký tự. Kiểm soát bằng ground-truth reward, harm penalty và không dùng confidence thô làm reward chính.

### Overprocessing

Deblur, sharpen và enhancement lặp lại có thể sinh ringing hoặc nối sai nét. Kiểm soát bằng action nguyên tử, action mask, horizon ngắn và step cost.

### Overfitting vào validation

Các threshold preprocessing hiện tại đã được lựa chọn trên validation. Cần group split và holdout mới cho kết luận cuối.

### Action space không đủ mạnh

Oracle hiện còn 21 ảnh test không phương pháp nào đọc đúng. PPO không giải quyết được nếu mọi action đều thất bại. Cần phân tích các ảnh này theo lỗi crop, độ phân giải, che khuất, blur nặng và lỗi nhãn trước khi thêm tool.

### Mất cân bằng dữ liệu

Biển xanh, ngoại giao, quân đội và biển vàng có số lượng ít hơn nhóm phổ biến. Dùng weighted sampler hoặc per-group objective, nhưng không để các nhóm cực nhỏ làm policy học thuộc.

## 12. Mốc triển khai đề xuất

### Mốc A — Feasibility

- khóa split và metric;
- tạo one-step cache trên train;
- tính oracle train/validation;
- phân tích 21 ảnh irrecoverable;
- hoàn thành trong một pipeline chạy có thể tái lập.

### Mốc B — Router

- frozen PARSeq feature extractor;
- reward regression hoặc behavior cloning;
- so sánh với `adaptive_noise_3way` và fixed best;
- tích hợp action trace vào demo.

### Mốc C — Sequential agent

- tách action nguyên tử;
- cache chuỗi tối đa hai bước;
- behavior cloning theo trajectory;
- quyết định có cần PPO dựa trên oracle gap.

### Mốc D — PPO và đánh giá cuối

- PPO fine-tuning;
- ablation reward/state/horizon;
- paired statistical test;
- đánh giá end-to-end sau YOLO detector;
- benchmark accuracy, harm rate và latency trên holdout mới.

## 13. Quyết định khuyến nghị ngay lúc này

Nên bắt đầu bằng Mốc A và Mốc B, chưa triển khai PPO ngay. Với kết quả hiện tại, bài toán quan trọng nhất là học được khi nào phải giữ ảnh gốc và khi nào chọn đúng một trong các pipeline đã có. Nếu supervised router không tiến gần oracle 94,71% trên validation, PPO nhiều bước khó tạo cải thiện đáng tin cậy. Nếu router tiến gần oracle và oracle chuỗi hai bước tiếp tục tăng, khi đó actor–critic/PPO theo Restore-R1 mới có cơ sở thực nghiệm rõ ràng.

## 14. Tài liệu và dữ liệu đối chiếu

- [`suggest.md`](suggest.md): đề xuất ban đầu cho restoration agent hướng OCR.
- [`Lu_Restore-R1_Efficient_Image_Restoration_Agents_via_Reinforcement_Learning_with_Multimodal_CVPRF_2026_paper.pdf`](Lu_Restore-R1_Efficient_Image_Restoration_Agents_via_Reinforcement_Learning_with_Multimodal_CVPRF_2026_paper.pdf): kiến trúc actor–critic, action record, PPO/GAE và reward theo chênh lệch chất lượng.
- [`outputs/testing/preprocessing_adaptive_benchmark/test_finalists_results.csv`](outputs/testing/preprocessing_adaptive_benchmark/test_finalists_results.csv): kết quả adaptive preprocessing.
- [`outputs/testing/preprocessing_combinations_benchmark/test_finalists_results.csv`](outputs/testing/preprocessing_combinations_benchmark/test_finalists_results.csv): kết quả pipeline kết hợp.
- [`outputs/testing/ml_official_preprocessing_benchmark/test_finalists_results.csv`](outputs/testing/ml_official_preprocessing_benchmark/test_finalists_results.csv): kết quả các model ML chính chủ.
