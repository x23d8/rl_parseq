# Phase 3: Fine-tune với augmentation có kiểm soát

Phase 3 tiếp tục fine-tune từ checkpoint tốt nhất của notebook gốc. Chỉ tập `train` được augmentation; `validation` và `test` luôn dùng pipeline `train_baseline` cố định. Cấu hình, manifest, ảnh preview và tần suất augmentation thực tế đều được lưu để có thể kiểm tra lại.

Mặc định script đóng băng cả đường dẫn và target validation/test bằng hai manifest nằm cạnh checkpoint cha. Có thể truyền rõ `--val-manifest` và `--test-manifest` khi cần. Cơ chế này ngăn thay đổi nhãn trong dataset làm sai lệch so sánh giữa các phase.

## Mục tiêu

- Tăng độ bền với ảnh nhỏ, mất chi tiết do resize và nén JPEG.
- Học tốt hơn với blur, nhiễu, thiếu sáng, thừa sáng và lệch phối cảnh nhẹ.
- Hỗ trợ biển hai dòng bằng unwrap có điều kiện.
- Tiếp tục tận dụng các preprocessing đã cải thiện accuracy ở Phase 1.
- Hạn chế overfit và catastrophic forgetting khi fine-tune tiếp.

## Policy mặc định

| Nhóm | Biến đổi | Kiểm soát |
|---|---|---|
| Giữ miền gốc | Ảnh gần nguyên bản | 20% mẫu; dùng `train_baseline` hoặc RGB |
| Độ phân giải | Downsample rồi phóng lại, upscale 2×/3× trước restoration | Tăng xác suất cho crop nhỏ; không downsample dưới kích thước an toàn |
| Hình học | Zoom 0,84–1,16; affine/perspective nhẹ | Không flip, không xoay lớn, không crop mạnh |
| Biển hai dòng | Unwrap top-to-bottom | Chỉ xét ảnh có aspect ratio dưới 1,9 |
| Chất lượng ảnh | Gaussian/motion blur, JPEG, noise, brightness/contrast/gamma | Tối đa 2 degradation trên một mẫu |
| Restoration | `train_baseline`, `clahe_clip1_tile4`, `clahe_rl_deblur_bilateral`, `adaptive_noise_3way`, `raw_rgb` | Chọn theo trọng số cố định và ghi audit theo epoch |

Thứ tự xử lý chính:

```text
ảnh train
  -> zoom / unwrap / affine nhẹ
  -> degradation có giới hạn
  -> upscale ảnh nhỏ 2× hoặc 3×
  -> preprocessing được chọn
  -> resize 32×128
  -> PARSeq PLM loss
```

## Các cơ chế chống overfit

- Khởi tạo từ checkpoint fine-tuned tốt nhất, learning rate mặc định `3e-6`.
- Đóng băng encoder trong epoch đầu.
- Cosine learning-rate decay và gradient clipping.
- Early stopping theo validation exact match, dùng character accuracy để phá hòa.
- Cân bằng loại biển dạng tempered và giới hạn weight tối đa 8×. Một ảnh blue duy nhất không bị lặp hàng trăm lần như cân bằng nghịch đảo hoàn toàn.
- Chọn `refine_iters` trên validation rồi mới đánh giá test đúng một lần.
- 21 ảnh hard thuộc test không được đưa vào train.

## Kiểm tra trước khi train

Chạy dry-run để đọc toàn bộ dataset, tạo preview, kiểm tra checkpoint và tính PLM loss trên hai mẫu:

```powershell
python refinement_finetune\train_phase3_controlled_augmentation.py `
  --dry-run `
  --device cpu `
  --output-dir outputs\testing\phase3_controlled_augmentation_dry_run
```

Kết quả dry-run hiện tại khớp notebook gốc:

| Split | Số ảnh |
|---|---:|
| Train | 3.270 |
| Validation | 397 |
| Test | 411 |

## Chạy Phase 3 đầy đủ

```powershell
python refinement_finetune\train_phase3_controlled_augmentation.py `
  --checkpoint outputs\testing\refinement_finetune_20260710_142307\best_official_parseq_anpr.pt `
  --policy-profile full `
  --epochs 12 `
  --batch-size 16 `
  --learning-rate 3e-6 `
  --freeze-encoder-epochs 1 `
  --early-stopping-patience 4 `
  --output-dir outputs\phase3_controlled_aug_full
```

Nếu GPU thiếu VRAM, giảm `--batch-size` xuống 8. Có thể tăng `--num-workers` sau khi đã xác nhận pipeline chạy ổn trên Windows.

## Ablation bắt buộc trước khi kết luận

Chạy cùng seed và cùng số epoch cho ba profile:

```powershell
python refinement_finetune\train_phase3_controlled_augmentation.py --policy-profile resolution_only --output-dir outputs\phase3_ablation_resolution
python refinement_finetune\train_phase3_controlled_augmentation.py --policy-profile restoration_only --output-dir outputs\phase3_ablation_restoration
python refinement_finetune\train_phase3_controlled_augmentation.py --policy-profile full --output-dir outputs\phase3_ablation_full
```

So sánh checkpoint bằng validation trước. Test chỉ dùng để xác nhận profile đã khóa, không dùng để chỉnh xác suất augmentation.

## File kết quả chính

| File | Nội dung |
|---|---|
| `dataset_manifest.csv` | Manifest cố định của ba split |
| `dataset_summary.csv` | Số mẫu theo split và loại biển |
| `augmentation_preview/` | Ảnh sau augmentation để kiểm tra trực quan |
| `augmentation_stats_by_epoch.csv` | Tần suất thực tế của từng phép biến đổi |
| `parent_validation_error_profile.json` | Xu hướng lỗi của checkpoint cha, chỉ dùng validation |
| `history.csv` | Loss và validation metric theo epoch |
| `best_phase3_parseq_anpr.pt` | Checkpoint Phase 3 tốt nhất |
| `refinement_sweep_val.csv` | Chọn `refine_iters` trên validation |
| `eval_test_predictions_locked.csv` | Kết quả test sau khi khóa cấu hình |
| `summary.json` | Tổng hợp kết quả cuối |

## Đánh giá lại bằng Phase 1 và Phase 2

Sau khi train xong, dùng manifest validation/test của Phase 3 để chạy TTA:

```powershell
python preprocessing_best_config\benchmark_multiscale_tta.py `
  --checkpoint outputs\phase3_controlled_aug_full\best_phase3_parseq_anpr.pt `
  --val-manifest outputs\phase3_controlled_aug_full\eval_val_predictions_best_refine.csv `
  --test-manifest outputs\phase3_controlled_aug_full\eval_test_predictions_locked.csv `
  --output-dir outputs\testing\phase3_multiscale_tta

python preprocessing_best_config\benchmark_multiscale_selector_phase2.py `
  --phase1-dir outputs\testing\phase3_multiscale_tta `
  --output-dir outputs\testing\phase3_multiscale_selector
```

Chỉ xem Phase 3 là cải thiện khi tăng trên validation, không làm giảm đáng kể character accuracy, và kết quả test/21 ảnh hard xác nhận xu hướng đó.
