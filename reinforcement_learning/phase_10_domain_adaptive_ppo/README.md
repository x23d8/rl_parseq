# Phase 10 — Domain-adaptive PPO

Phase 10 dùng cache Phase 9 đã mở làm development data, không còn coi nó là holdout. Stable group split đưa 80% Phase 9 vào adaptation và 20% vào validation selection, rồi ghép với train/validation Phase 7. Audited legacy test không được tải.

```powershell
python reinforcement_learning\run_phase.py phase10-prepare
python reinforcement_learning\run_phase.py phase10-train
```

Checkpoint sau huấn luyện chỉ là development candidate. Trước promotion phải khóa candidate và đánh giá đúng một lần trên holdout Phase 10 mới, loại toàn bộ source/label/ảnh Phase 7–9.

