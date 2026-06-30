## Eval results (15 cases, top_k=5)

- **Answer pass-rate: 13/15 (87%)**
- Retrieval hit-rate (page-level, 12 answerable cases): 100%

**By difficulty:**
- easy: 3/3
- hard: 3/5
- medium: 4/4
- trap: 3/3

**Failures by category:**
- missing_fact: 2

| id | difficulty | retrieval_hit | passed | category | detail |
| --- | --- | --- | --- | --- | --- |
| easy-usb-current | easy | True | True | pass | all constraints satisfied |
| easy-object-height | easy | True | True | pass | all constraints satisfied |
| easy-assembly-people | easy | True | True | pass | all constraints satisfied |
| med-reset | medium | True | True | pass | all constraints satisfied |
| med-lock | medium | True | True | pass | all constraints satisfied |
| med-height-limit | medium | True | True | pass | all constraints satisfied |
| med-jog-continuation | medium | True | True | pass | all constraints satisfied |
| trap-lower-key | hard | True | True | pass | all constraints satisfied |
| trap-warranty | trap | None | True | pass | refused as expected |
| trap-price | trap | None | True | pass | refused as expected |
| trap-bluetooth | trap | None | True | pass | refused as expected |
| trap-usb-voltage-only | hard | True | True | pass | all constraints satisfied |
| fail-travel-range | hard | True | False | missing_fact | missing: ['48'] |
| pass-motor-error-codes | hard | True | True | pass | all constraints satisfied |
| fail-error-code-count | hard | True | False | missing_fact | missing: ['20'] |
