# Báo cáo Phase 11 — Replicated primary PPO seed 728

## Kết luận

Phase 11 khóa PPO seed 728 sau khi policy này đạt `15 fix / 3 break` trên 1.154 mẫu Phase 8+9 đã mở. Power contract chọn 1.500 mẫu mới, ước tính power McNemar `89,50%`. Holdout dùng trực tiếp `extracted_character`, không review thủ công, gồm 1.500 unique label/ảnh không trùng historical, Phase 7, Phase 8, Phase 9 hoặc source đã mở.

Kết quả one-shot:

| Policy | Exact | Character | Fix/break |
|---|---:|---:|---:|
| Baseline | 47,2000% | 82,8013% | 0/0 |
| PPO seed 728 | **47,8667%** | **83,5529%** | **16/6** |

Exact delta là `+0,6667` điểm %, paired CI 95% `[+0,0667; +1,2667]`; character delta `+0,7318` điểm %, CI `[+0,3832; +1,0845]`; net fixes `+10`. McNemar exact hai phía là `p=0,0524788`, vừa không đạt ngưỡng khóa `<0,05`. Vì vậy `promotion_eligible=false`, không tạo active registry và baseline vẫn là policy triển khai.

Manifest SHA-256: `292b5cea36981dcc83fe1c4c8fb685b55f5fef74aff44ca4849d92d7b3d6bb8b`. Receipt one-shot claim ID: `c6b7a8bed8fe1a7ca09990f26e44ce2fb9975d2c37b880bdcc431fa21d9883bf`.

## Hướng tiếp theo

Posthoc guard chỉ cho thay baseline với `existing_plate_crop` và min-side `<128` đạt `14 fix / 2 break` trên Phase 11. Rule này ổn định trên Phase 8 và Phase 9, nên đã được khóa thành Phase 12; không được áp lại rồi promote từ Phase 11.

