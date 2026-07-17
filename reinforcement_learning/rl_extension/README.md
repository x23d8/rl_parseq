# RL mở rộng sau ba phase

Ba phase chính tạo nền tảng action space, multi-scale views và controlled training. Phần RL tiếp theo được triển khai trong package chuẩn `rl_restoration`:

- offline trajectory cache;
- contextual-bandit reward router;
- OCR-grounded reward và harm penalty;
- actor–critic PPO hai bước với clipped objective và GAE;
- hard-aware PARSeq fine-tune bằng policy-selected views;
- runtime PPO cho ảnh crop mới.

Tài liệu:

- `RL_RESTORATION_AGENT_PLAN.md`;
- `RL_PHASE4_EXPERIMENT_REPORT.md`;
- `RL_PHASE5_PPO_REPORT.md`;
- `rl_restoration/README.md`.

Phần này được giữ riêng vì nó được phát triển sau ba phase và không nên bị nhầm với Phase 1–3 ban đầu.

