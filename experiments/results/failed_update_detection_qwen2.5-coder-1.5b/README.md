# Failed update-detection diagnostic

**Disposition (2026-11-19): retain this incomplete run for provenance; exclude
it from accepted results and the active cache.** The original JSONL was moved
here without changing its bytes. The experiment's default cache lookup does not
search this directory. Do not copy it back to resume an accepted evaluation.

Model: `qwen2.5-coder:1.5b`. The file contains 518 valid JSON records with 518
unique cache keys, covering 28 of the 65 planned question IDs; it is not a
completed experiment. All 518 stored answers match their raw YES/NO responses.
All cache keys were verified against the current prompt builder and historical
suite pair texts. No error log is available to establish why execution stopped.

| Cache-label group | YES | Recorded responses | YES rate |
| --- | --- | --- | --- |
| Evidence candidates | 110 | 140 | 78.6% |
| Control candidates | 192 | 324 | 59.3% |
| Random pairs | 28 | 54 | 51.9% |

These are counts of unique cached responses under their original labels.
Repeated prompt requests reuse entries, so these are not guaranteed to equal
the full judged-pair occurrence counts. Control YES decisions are not individually
adjudicated false positives: some may describe real unlabelled updates.

An offline replay of all 65 planned streams (zero model calls) matched every
archived key. Of 1,235 planned pair occurrences, 527 have cached answers and
708 are missing. The random group has 28 YES, 26 NO, and 76 missing responses.

The observed random-pair YES rate greatly exceeds the design's 10% validity
limit. Even if all remaining planned random responses were NO, 28 YES responses
out of at most 130 planned random-pair occurrences would exceed 10% (21.5%),
provided the historical suite, prompt, and design remain fixed. This configuration
cannot pass its validity rule by simply finishing the remaining questions.
The control-candidate YES rate also exceeds the design's 5% threshold on the
observed portion. Do not infer full-run retrieval recovery or harm from this cache.

There were 330 YES and 188 NO responses, with 34,481.74 recorded model seconds
(about 9.58 hours; not total wall time). Preserve this as a failed diagnostic,
not a detector-effectiveness result or evidence that update detection in general
fails. Resolve the [label/conversion review](../../../docs/longmemeval_label_review.md)
before another benchmark run. Validate any revised detector on separate examples
and use a fresh, explicitly versioned run directory.

Original path: `experiments/results/update_detection_judgments_qwen2.5-coder-1.5b.jsonl`.

SHA-256 of the archived JSONL:
`9d0177d53cb7a745fb0c42868c13fb2a8464ee88b3fd2df91efc5e0957ead836`.

Machine-readable details: [summary.json](summary.json).
