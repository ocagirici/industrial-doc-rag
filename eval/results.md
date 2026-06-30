## Eval results (12 cases, top_k=5)

- **Answer pass-rate: 12/12 (100%)**
- Retrieval hit-rate (page-level, 9 answerable cases): 100%

**By difficulty:**
- easy: 3/3
- hard: 2/2
- medium: 4/4
- trap: 3/3

**Failures by category:**
- (none)

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
