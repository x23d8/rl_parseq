# Phase 6 — Candidate-aware OOF PPO

Phase 6 xử lý trực tiếp bốn điểm còn thiếu của Phase 5:

- cache đặc trưng PARSeq encoder, OCR uncertainty và chất lượng cho cả 10 candidate view;
- actor–critic dùng self-attention trên candidate set thay vì vector consensus phẳng;
- teacher prior cho development được tạo out-of-fold theo nhóm biển số;
- tạo group holdout mới từ train, không cho cùng `label` xuất hiện ở development và holdout;
- paired bootstrap và McNemar quyết định gate cải thiện cuối cùng.

Mọi code, cache, checkpoint và báo cáo của Phase 6 đều nằm trong thư mục này. Pipeline chỉ đọc trajectory/checkpoint cũ; nó không ghi artefact mới ra ngoài `reinforcement_learning`.

## Kiến trúc

```text
10 candidate views
    ↓
PARSeq encoder pooling + OCR uncertainty + image quality (783 chiều/view)
    ↓
shared candidate projection + action embedding
    ↓
2-layer candidate self-attention
    ├── per-candidate actor residual + OOF teacher prior
    └── global/current-action critic
             ↓
       PPO hai bước: stop / accept / revise / rollback
```

Actor residual được khởi tạo bằng 0, nên policy bắt đầu chính xác từ teacher. PPO phải học được residual hữu ích; nếu không, policy không được thăng cấp qua gate holdout.

## Chạy

```powershell
python reinforcement_learning\run_phase.py phase6-cache --split train
python reinforcement_learning\run_phase.py phase6-cache --split val
python reinforcement_learning\run_phase.py phase6
python -m unittest reinforcement_learning.phase_6_candidate_oof_ppo.test_phase6
```

Cache mặc định: `results/candidate_cache/`. Run chính: `results/run_residual_seed_123/`.

## Trạng thái hiện tại

Checkpoint residual seed 123 tăng validation so với teacher OOF thêm 0,5038 điểm %, nhưng exact holdout hòa và kiểm định chưa đạt. Vì vậy checkpoint đang ở trạng thái `experimental`, không thay thế Phase 5 locked policy. Chi tiết ở `PHASE6_REPORT.md`.

