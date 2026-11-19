# Zero leakage at no utility cost (H1)

Synthetic, scripted workspace; lexical search only; k = 10. 200 trials (seeds 0 to 199),
9010 leakage, 3000 owner, 24000 utility and 6000 over-blocking items. See the
docstring of `experiments/audience_utility.py` for the design and decision rule.

| Arm | Leaks | Owner hit@k | Utility mean RR | Utility hit@k | Mean results shown |
| --- | --- | --- | --- | --- | --- |
| pooled (no access control) | 9010 (100.0%) | 3000 (100.0%) | 0.9406 | 100.0% | 10.00 |
| filter after, first k only | 0 (0.0%) | 3000 (100.0%) | 1.0000 | 100.0% | 6.74 |
| filter after, whole ranking | 0 (0.0%) | 3000 (100.0%) | 1.0000 | 100.0% | 9.63 |
| search with viewers (built) | 0 (0.0%) | 3000 (100.0%) | 1.0000 | 100.0% | 9.63 |

Utility differences: per-trial mean RR of built minus the other arm, with a
95% percentile bootstrap CI over trials (2,000 resamples).

| Comparison | Mean D | 95% CI | Trials higher / equal / lower | Reading |
| --- | --- | --- | --- | --- |
| built minus after-all (rule 4) | +0.0000 | +0.0000 to +0.0000 | 0 / 200 / 0 | no utility cost |
| built minus after-k (descriptive) | +0.0000 | +0.0000 to +0.0000 | 0 / 200 / 0 | no utility cost |
| built minus pooled (descriptive) | +0.0594 | +0.0568 to +0.0620 | 200 / 0 / 0 | no utility cost |

Over-blocking (each `general` canary asked by each member, built arm, `limit=1000`):
returned in 6000 of 6000 items, within the first k in 6000 (100.0%).

Sanity: the largest store held 150 memories (BIG = 1000 covers every store: True);
write timestamps rose strictly in write order in 200 of 200 trials.

Decision:

- Rule 1, pooled leaks in at least 50% of leakage items: True (100.0%)
- Rule 1, owner hit@k in the built arm at least 95%: True (100.0%)
- Rule 2, every over-blocking item returned by the built arm: True (0 missing)
- Rule 3, zero leakage in after-k, after-all and built: True
- Rule 4, utility of built against after-all: no utility cost
- **Verdict: H1 supported**

Limits: the data were written alongside the code, so this shows that the
mechanism works, not how often it matters in real use. Gold memories are the
only memories that contain both query words, so a ranker that keeps every
visible memory puts gold first in almost every item; rule 4 checks that the
built path loses or misorders no visible memory, and has little power to show
a subtler ranking cost. Only lexical search was tested (no embedder, no links,
no keys); the link paths are tested in `experiments/audience_link_paths.py`.
