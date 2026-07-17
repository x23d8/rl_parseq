# PARSeq ANPR Reinforcement-Learning Pipeline

Repository này lưu toàn bộ nhánh nghiên cứu cải thiện nhận dạng biển số (ANPR) của PARSeq bằng tiền xử lý ảnh, chọn candidate view và reinforcement learning (RL). Mục tiêu không phải sinh ảnh mới, mà là chọn một cách biến đổi ảnh đầu vào sao cho PARSeq đọc biển số chính xác hơn, đồng thời có cơ chế quay về ảnh baseline khi thay đổi có nguy cơ gây hại.

Tài liệu này mô tả phạm vi dự án, dữ liệu/phụ thuộc cần có, cách thiết lập môi trường và các lệnh có thể dùng để tái hiện. Chi tiết thuật toán, state/action/environment, reward–penalty và ý nghĩa Phase 1–12 nằm tại [`reinforcement_learning/README.md`](reinforcement_learning/README.md).

## 1. Bài toán và luồng xử lý

Đầu vào chuẩn của nhánh chính là **ảnh đã crop biển số**. Một mẫu được xử lý theo luồng:

```text
plate crop
  -> tạo các view (zoom / upscale / unwrap hai dòng / CLAHE / deblur / denoise)
  -> PARSeq OCR từng view
  -> policy chọn baseline hoặc một view thay thế
  -> prediction cuối
  -> gate thống kê trên holdout mới trước khi cho phép promote
```

Dự án gồm hai họ RL độc lập:

1. `rl_restoration/` và `reinforcement_learning/phase_6...phase_12`: contextual bandit và PPO chọn **một candidate view hoàn chỉnh** cho PARSeq. Đây là nhánh chính của Phase 4–12.
2. `parseq_rl_deblur_data/rl_deblur/`: PixelRL/A2C chọn thao tác **theo từng pixel** để khử mờ tổng hợp. Đây là nhánh thử nghiệm cũ, không phải action space của Phase 4–12.

## 2. Trạng thái hiện tại

- Baseline lịch sử trên test 411 ảnh: `91,9708%` exact match.
- Phase 2, dùng 65-view TTA và calibrated selector: `94,1606%` exact match; đây là kết quả tốt nhất trong ba phase đầu nhưng cần 65 OCR views.
- Phase 7 rút action space còn 9 view và huấn luyện candidate-set PPO. Các checkpoint tăng trên validation/internal holdout nhưng chưa qua đồng thời mọi gate xác nhận ngoài.
- Phase 8, 9 và 11 đều được đánh giá one-shot trên external holdout mới nhưng không đạt đủ điều kiện promote.
- Phase 12 đã khóa rule an toàn cho PPO seed 728 nhưng chưa có đủ fresh data. Trạng thái kiểm tra ngày `2026-07-17`: còn `235` unique-label candidate, thiếu `265` để đạt minimum formal 500 và thiếu `1.265` để đạt target 1.500.
- Chưa có `active_policy.json`; do đó baseline vẫn là policy triển khai. Runtime Phase 7–12 cố ý từ chối dùng checkpoint chưa được promote.

Kiểm tra trạng thái mới nhất mà không chạy OCR/PPO:

```powershell
python reinforcement_learning\run_phase.py phase12-status
```

Lệnh trên cần dataset external mặc định tại `D:\NEO\image_processing\dataset_general`. Nếu dữ liệu nằm nơi khác, xem `phase12-status --help` và truyền `--source-root`, `--labels-csv`.

## 3. Cấu trúc repository

```text
rl_pipeline/
|-- README.md
|-- requirements.txt
|-- preprocessing_best_config/       # Tiền xử lý, Phase 1 và Phase 2
|-- train_no_refinement/              # PARSeq ANPR loader/train integration
|-- rl_restoration/                   # Bandit + PPO hai bước của Phase 4–5
|-- reinforcement_learning/           # Phase 1–12, report, audit, runtime
|-- parseq_rl_deblur_data/
|   `-- rl_deblur/                    # PixelRL/A2C deblur độc lập
|-- parseq/                           # Source PARSeq/strhub local (đang bị git-ignore)
`-- outputs/                          # Checkpoint/cache lịch sử (đang bị git-ignore)
```

Các thư mục `outputs/`, `parseq/`, model binary (`*.pt`, `*.ckpt`, `*.npz`, `*.joblib`) và ảnh external bị loại khỏi Git theo `.gitignore`. Vì vậy **clone source code đơn thuần không đủ để tái hiện số liệu lịch sử**. Cần nhận thêm gói artefact/dataset từ người quản lý dự án hoặc tự rebuild theo thứ tự ở phần 8.

Một snapshot `refinement_finetune/` từng được Phase 3 sử dụng nhưng không có trong checkout hiện tại. Kết quả Phase 3 vẫn có trong `outputs/` ở máy nghiên cứu; muốn train lại Phase 3 phải khôi phục source snapshot này trước.

## 4. Yêu cầu hệ thống

- Python `3.11` được dùng để kiểm tra checkout hiện tại; PARSeq khai báo hỗ trợ Python `>=3.9`.
- Windows/PowerShell là môi trường gốc. Code dùng `pathlib` nên phần lớn lệnh chạy được trên Linux nếu đổi cú pháp activate và đường dẫn.
- NVIDIA GPU được khuyến nghị khi build cache hoặc chạy PARSeq nhiều view. Unit test và các audit chỉ đọc CSV/JSON có thể chạy CPU.
- Dung lượng đĩa đủ cho dataset, 9–65 inference view, cache `.npz/.csv` và checkpoint.

Môi trường đã xác minh tại thời điểm viết tài liệu: Python `3.11.0`, PyTorch `2.9.1+cu128`, CUDA khả dụng. `requirements.txt` gốc chưa pin toàn bộ version; nếu cần tái hiện byte-for-byte, phải lưu thêm `pip freeze` và thông tin driver/CUDA của run mới.

## 5. Thiết lập môi trường

### 5.1. Tạo virtual environment

PowerShell:

```powershell
cd D:\path\to\rl_pipeline
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

Linux/macOS:

```bash
cd /path/to/rl_pipeline
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### 5.2. Cài dependency

Nếu dùng CUDA, cài `torch`/`torchvision` đúng với driver theo môi trường của máy trước. Sau đó:

```powershell
python -m pip install -r requirements.txt
python -m pip install -r parseq\requirements\train.txt
python -m pip install --no-deps -e .\parseq
```

Trên Linux thay `\` bằng `/`.

Giải thích:

- `requirements.txt` chứa dependency của pipeline ANPR/RL.
- `parseq/requirements/train.txt` bổ sung Hydra, SciPy, scikit-image, TensorBoardX và các package mà source PARSeq cần.
- Cài editable `parseq` cung cấp package `strhub`; các entrypoint cũng tự thêm thư mục `parseq/` vào `sys.path`.

Kiểm tra import và thiết bị:

```powershell
python -c "import cv2, numpy, pandas, sklearn, torch; import strhub; print(torch.__version__, torch.cuda.is_available())"
```

### 5.3. Khôi phục source và artefact local

Tối thiểu cần có:

```text
parseq/strhub/...
outputs/testing/refinement_finetune_20260710_142307/best_official_parseq_anpr.pt
outputs/phase3_controlled_aug_full_frozen_eval/dataset_manifest.csv
outputs/rl_restoration/trajectory_cache/...
outputs/rl_restoration/router_seed_123/best_reward_router.pt
outputs/rl_restoration/ppo_prior_seed_123/best_ppo_restoration_policy.pt
reinforcement_learning/phase_7_compact_multiscale_ppo/results/
  run_ocr_guard_seed_727/best_candidate_oof_ppo.pt
  confirmatory_seed_728/best_candidate_oof_ppo.pt
```

Không phải workflow nào cũng cần tất cả file trên:

- Unit test: không cần dataset/checkpoint.
- Phase 1: cần PARSeq checkpoint và manifest val/test.
- Phase 4–5 train: cần checkpoint/manifest Phase 3, sau đó cache train/val.
- Phase 6: cần trajectory cache 10 actions và candidate cache.
- Phase 7 runtime/evaluation: cần PARSeq checkpoint, candidate PPO checkpoint/cache và manifest tương ứng.
- Phase 8–12 one-shot evaluation: cần candidate lock, fresh manifest, cache và các receipt/audit đúng hash.

### 5.4. Dữ liệu và manifest

Manifest train/val/test chính có schema:

```csv
image_path,label,source_name,plate_type,label_status,review_status,split
```

External holdout/runtime dùng schema cụ thể theo phase; tối thiểu thường có:

```csv
image_path,label,split,input_contract,input_transform,crop_width,crop_height
```

Các quy tắc quan trọng:

- `image_path` phải trỏ tới file ảnh thật.
- Input formal của Phase 7–12 phải là `plate_crop`, không phải ảnh toàn cảnh.
- Nhóm được tính bằng label đã normalize; train/development/holdout formal không được trùng nhóm theo contract của phase.
- Runtime label-free không đọc label.
- Nhiều manifest lịch sử chứa absolute path cũ như `D:\NEO\LPR\...`. Trên máy khác phải sửa `image_path` hoặc tạo mapping/junction tương thích. Không sửa manifest/receipt one-shot đã dùng để đưa ra kết luận lịch sử; hãy tạo bản manifest mới và output directory mới cho run tái hiện.

Dataset chính từng có 3.270 ảnh train, 397 validation và 411 test. External Phase 8/9/11 là dữ liệu đã mở, chỉ còn vai trò development evidence; không được tái sử dụng như confirmatory holdout.

## 6. Kiểm tra cài đặt

Chạy bộ test lõi không cần model/dataset:

```powershell
python -m unittest reinforcement_learning.phase_6_candidate_oof_ppo.test_phase6 reinforcement_learning.phase_7_compact_multiscale_ppo.test_phase7 reinforcement_learning.phase_8_consensus_ppo.test_phase8 reinforcement_learning.phase_9_primary_ppo.test_phase9 reinforcement_learning.phase_12_guarded_replicated_ppo.test_phase12
```

Kết quả đã xác minh: `38 tests`, `OK`.

Xem help của command dispatcher theo từng target:

```powershell
python reinforcement_learning\run_phase.py phase7 --help
python reinforcement_learning\run_phase.py phase12-status --help
```

## 7. Cách chạy nhanh

### 7.1. Inference một ảnh bằng PPO Phase 5

Workflow này không yêu cầu `active_policy.json`, nhưng cần checkpoint PPO/router và PARSeq hard-aware đúng đường dẫn mặc định hoặc truyền tường minh:

```powershell
python rl_restoration\ppo_runtime.py path\to\plate_crop.png
```

Output JSON gồm baseline prediction, action đầu, action cuối, `revised`, confidence và prediction cuối. Đây là runtime nghiên cứu Phase 5, không đồng nghĩa policy đã vượt external promotion gate của Phase 7–12.

### 7.2. Runtime Phase 7–12

Runtime formal chỉ hoạt động sau khi có registry đã promote:

```powershell
python reinforcement_learning\run_phase.py phase12-runtime `
  --manifest path\runtime_plate_crops.csv `
  --output-dir reinforcement_learning\phase_12_guarded_replicated_ppo\results\runtime_001
```

Ở trạng thái hiện tại lệnh sẽ từ chối vì chưa có `results/active_policy.json`. Đây là guard chủ đích, không phải lỗi cài đặt.

### 7.3. Nhánh PixelRL/A2C deblur

Nhánh này chạy từ thư mục riêng và cần dữ liệu `parseq_rl_deblur_data/color_filtered/`:

```powershell
cd parseq_rl_deblur_data
python -m rl_deblur.make_dataset --seed 42
python -m rl_deblur.train --epochs 15 --num-steps 5 --device cuda
python -m rl_deblur.evaluate --device cuda
python -m rl_deblur.make_samples --device cuda
cd ..
```

Để bật reward OCR ngoài reward pixel, truyền `--cer-reward-weight` và/hoặc `--logconf-reward-weight` cùng `--ocr-checkpoint`. Xem công thức chi tiết trong tài liệu RL.

## 8. Tái hiện các phase theo thứ tự

Luôn dùng output directory mới; một số script cố ý từ chối ghi đè artefact đã khóa.

### Phase 1 — 65-view multi-scale TTA

```powershell
python reinforcement_learning\run_phase.py phase1 `
  --checkpoint outputs\testing\refinement_finetune_20260710_142307\best_official_parseq_anpr.pt `
  --val-manifest outputs\testing\refinement_finetune_20260710_142307\eval_val_predictions_best_refine.csv `
  --test-manifest outputs\testing\refinement_finetune_20260710_142307\eval_test_predictions_best_refine.csv `
  --output-dir outputs\reproduction\phase1
```

### Phase 2 — calibrated selector

```powershell
python reinforcement_learning\run_phase.py phase2 `
  --phase1-dir outputs\reproduction\phase1 `
  --output-dir outputs\reproduction\phase2
```

### Phase 3 — controlled augmentation

Entrypoint hiện trỏ tới `refinement_finetune/train_phase3_controlled_augmentation.py`, nhưng snapshot này không có trong checkout. Sau khi khôi phục đúng thư mục:

```powershell
python reinforcement_learning\run_phase.py phase3 --help
```

Không thể tái train Phase 3 chỉ từ source đang có; có thể kiểm tra kết quả lịch sử tại `outputs/phase3_controlled_aug_full_frozen_eval/` nếu gói artefact local đã được khôi phục.

### Phase 4 — trajectory cache và reward router

```powershell
python rl_restoration\build_trajectory_cache.py --split train --output-dir outputs\reproduction\trajectory_cache
python rl_restoration\build_trajectory_cache.py --split val   --output-dir outputs\reproduction\trajectory_cache
python rl_restoration\train_router.py `
  --cache-dir outputs\reproduction\trajectory_cache `
  --seed 123 `
  --output-dir outputs\reproduction\router_seed_123
```

Nếu dùng checkpoint/manifest khác mặc định, truyền `--checkpoint` và `--manifest` cho bước build cache.

### Phase 5 — PPO hai bước

```powershell
python rl_restoration\train_ppo.py `
  --cache-dir outputs\reproduction\trajectory_cache `
  --teacher-router outputs\reproduction\router_seed_123\best_reward_router.pt `
  --seed 123 `
  --output-dir outputs\reproduction\ppo_seed_123
```

### Phase 6 — candidate-aware OOF PPO

```powershell
python reinforcement_learning\run_phase.py phase6-cache --split train --output-dir reinforcement_learning\phase_6_candidate_oof_ppo\results\reproduction_cache
python reinforcement_learning\run_phase.py phase6-cache --split val   --output-dir reinforcement_learning\phase_6_candidate_oof_ppo\results\reproduction_cache
python reinforcement_learning\run_phase.py phase6 `
  --candidate-cache reinforcement_learning\phase_6_candidate_oof_ppo\results\reproduction_cache `
  --output-dir reinforcement_learning\phase_6_candidate_oof_ppo\results\reproduction_seed_123
```

### Phase 7 — compact 9-view PPO

```powershell
python reinforcement_learning\run_phase.py phase7-cache --split train --output-dir reinforcement_learning\phase_7_compact_multiscale_ppo\results\reproduction_cache
python reinforcement_learning\run_phase.py phase7-cache --split val   --output-dir reinforcement_learning\phase_7_compact_multiscale_ppo\results\reproduction_cache
python reinforcement_learning\run_phase.py phase7 `
  --trajectory-cache reinforcement_learning\phase_7_compact_multiscale_ppo\results\reproduction_cache `
  --candidate-cache reinforcement_learning\phase_7_compact_multiscale_ppo\results\reproduction_cache `
  --output-dir reinforcement_learning\phase_7_compact_multiscale_ppo\results\reproduction_seed_727
```

### Phase 8–12 — xác nhận, adaptation và guard

Đây không phải chuỗi lệnh được phép chạy lại mù quáng trên các holdout lịch sử. Các phase dùng candidate lock, SHA-256, group-disjoint contract và receipt one-shot. Hãy xem trạng thái trước:

```powershell
python reinforcement_learning\run_phase.py phase8-status
python reinforcement_learning\run_phase.py phase12-status
```

Chỉ chạy `prepare -> phase7-cache --preflight -> phase7-cache -> evaluate -> promote` khi có **fresh holdout chưa từng inference** và đúng contract của phase. Các lệnh cụ thể và điều kiện fail-fast được mô tả trong `reinforcement_learning/README.md` cùng README/REPORT của từng phase.

## 9. Kết quả và artefact

- Bảng tổng hợp: `reinforcement_learning/RESULTS_SUMMARY.csv`.
- Chỉ mục artefact: `reinforcement_learning/ARTIFACT_INDEX.md`.
- Báo cáo từng phase: `reinforcement_learning/phase_*/REPORT.md` hoặc `PHASE*_REPORT.md`.
- Lịch sử/summary của Phase 4–5: `outputs/rl_restoration/...`.
- Mỗi run mới nên lưu `summary.json`, `protocol.json`, selection CSV, checkpoint, seed, CLI đầy đủ, `pip freeze` và hash manifest/checkpoint.

## 10. Lỗi thường gặp

### `FileNotFoundError` với đường dẫn `D:\NEO\...`

Manifest/config lịch sử chứa absolute path của máy gốc. Dùng dữ liệu đúng vị trí cũ, tạo junction tương thích, hoặc tạo manifest mới với đường dẫn đã rebase.

### Không import được `strhub`

Khôi phục source `parseq/`, rồi chạy:

```powershell
python -m pip install --no-deps -e .\parseq
```

### Không tìm thấy checkpoint/cache

Binary và `outputs/` bị git-ignore. Khôi phục gói artefact hoặc rebuild từ phase trước. `ARTIFACT_INDEX.md` cho biết file nào là đầu vào của phase sau.

### Phase 3 không chạy

Checkout hiện thiếu `refinement_finetune/`. Đây là giới hạn đã biết; cần snapshot source gốc, không chỉ checkpoint.

### Runtime báo thiếu active registry

Không tạo registry thủ công. Registry chỉ được `promote.py` tạo sau khi formal external gate pass và receipt/hash hợp lệ.

### CUDA out of memory

Giảm `--batch-size`, đặt `--num-workers 0`, hoặc dùng `--device cpu` cho smoke run. Multi-view cache là bước tốn GPU/đĩa nhất.

## 11. Nguyên tắc tái lập và không rò rỉ dữ liệu

- Không dùng audited test hoặc external holdout đã mở để train, chọn checkpoint, margin hay guard rồi tuyên bố lại là confirmatory.
- Split formal theo normalized label group, không chỉ theo ảnh.
- Giữ nguyên candidate lock, action registry và checkpoint hash sau khi holdout được mở.
- Không ghi đè output/receipt cũ; tạo run directory mới.
- `oracle` chỉ đo trần có thể đạt khi dùng label để chọn candidate; không phải policy có thể deploy.
- `action_cost` là regularizer/proxy cho lựa chọn, không đồng nghĩa latency thực khi runtime vẫn tạo toàn bộ 9 view.
- Chỉ policy có `active_policy.json` hợp lệ mới được coi là policy đã promote.
