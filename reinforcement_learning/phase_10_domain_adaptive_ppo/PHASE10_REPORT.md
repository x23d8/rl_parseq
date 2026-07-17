# Báo cáo Phase 10 — Domain-adaptive PPO experiment

Phase 10 chuyển 650 mẫu Phase 9 đã mở thành development data, stable-split 520 adaptation / 130 validation. Khi dựng mixed cache, audit phát hiện train/validation Phase 7 trùng 144 label group; 509 dòng train thuộc các group này đã bị loại, làm mixed train/validation overlap bằng 0.

Hai PPO domain-adaptive đều bị loại:

- seed 1001: internal holdout `7 fix / 10 break`, net `-3`, stop rate chỉ `3,96%`;
- seed 1002 với final OOF-teacher gain safety gate: validation `9/0` nhưng internal holdout `8/13`, net `-5`.

Kết quả chứng minh domain adaptation hiện tại overfit validation và teacher prior chưa đủ hiệu chuẩn để bảo vệ baseline. Không checkpoint Phase 10 nào được khóa cho external confirmation. Safety-gate API vẫn được giữ với mặc định tắt cho checkpoint cũ và đã có regression test.

