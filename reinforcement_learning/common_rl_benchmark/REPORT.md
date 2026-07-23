# Common val/test RL benchmark

Diagnostic comparison on the same synthetic blurred validation and test splits. All OCR outputs use one fixed PARSeq checkpoint; no policy threshold is retuned.

- PARSeq SHA-256: `d1dd1c147fa2b67a3e650673e54ade86e6c592f8d85a9169bc01be3b59457b0d`
- Validation samples: `367`
- Test samples: `367`
- Status: opened diagnostic benchmark, not eligible for promotion.

| Method | Paradigm | Val exact | Test exact | Test delta vs unsharp | Test fixed/broken vs raw | Test CER |
|---|---|---:|---:|---:|---:|---:|
| Raw blurred + fixed PARSeq | No RL | 75.4768% | 72.2071% | -1.6349 pt | 0/0 | 11.1039% |
| Classical unsharp + fixed PARSeq | No RL | 76.0218% | 73.8420% | +0.0000 pt | 12/6 | 9.7688% |
| Train preprocessing + fixed PARSeq | No RL | 65.9401% | 65.1226% | -8.7193 pt | 14/40 | 13.2856% |
| PixelRL (A2C) | A2C / PixelRL | 76.0218% | 75.2044% | +1.3624 pt | 32/21 | 6.0567% |
| Contextual Bandit (Phase 4) | Contextual bandit | 71.9346% | 74.1144% | +0.2725 pt | 9/2 | 10.7131% |
| PPO Phase 5 (MLP) | PPO / actor-critic | 72.2071% | 74.6594% | +0.8174 pt | 11/2 | 10.3875% |
| Candidate-aware PPO (Phase 6) | PPO / Transformer | 75.4768% | 74.1144% | +0.2725 pt | 8/1 | 9.8014% |
| Compact PPO (Phase 7) | PPO / Transformer | 74.3869% | 70.8447% | -2.9973 pt | 15/20 | 10.6806% |

## Interpretation limits

- PixelRL was trained on the synthetic training split; other frozen policies were trained on historical OCR crops.
- The underlying clean source plates may overlap historical policy-development data, so results are diagnostic.
- Phase 4-7 policies were not retrained for synthetic blur; this measures frozen transfer, not each method's best achievable result.
- Fixed/broken uses raw blurred PARSeq as the single shared reference for every method.
