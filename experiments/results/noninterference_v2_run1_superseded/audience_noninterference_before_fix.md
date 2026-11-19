# Noninterference of reads with viewers (E2, version 2)

Synthetic, scripted data; no LLM. 500 pairs (seeds 0 to 499).
Store under test: schema version 5.
In each pair, worlds A and B run the same visible operations and different
hidden ones, R runs the visible operations alone, and N runs them as if the
retracted memories had never been written. See the docstring of
`experiments/audience_noninterference.py` for the design and decision rule.

Pairs: viewers ana alone 254, ana and ben 246; viewers joined and left `others` 263;
link carry-over keep 169, copy 160, topic 171; no embedder 242, n-gram embedder 258.

| Rule | Check | Held in | Needed | Holds |
| --- | --- | --- | --- | --- |
| 1 | Determinism: A and A2 identical in every arm and operation outcome | 500 of 500 (100.0%) | every pair | True |
| 2 | Sensitivity: built(A) differs from built(A+) | 500 of 500 (100.0%) | every pair | True |
| 3 | Power: after-naive differs between A and B | 500 of 500 (100.0%) | at least 50% | True |
| 4 | Noninterference: built(A) equals built(B), and visible operations went as predicted in both | 91 of 500 (18.2%) | every pair | False |
| 5 | Exactness: built(A) equals plain(R) | 230 of 500 (46.0%) | every pair | False |
| 6 | Soundness: no built read in A or B shows or links to a hidden memory; hidden IDs read as missing | 114 of 500 (22.8%) | every pair | False |
| 7 | Retraction: built(A) equals plain(N), and N refused nothing | 97 of 500 (19.4%) | every pair | False |

**Verdict: the built read path fails rules 4, 5, 6, 7 (first failing seed 0)**

Reported, not part of the verdict:

- Sensitivity, first search alone: built(A) and built(A+) differ in 500 of 500 (100.0%).
- after-careful differs between A and B in 500 of 500 (100.0%).
- The visible script's model predicted every visible outcome in R in 411 of 500 (82.2%) (a check of the model itself: R holds no hidden part).
- Visible operations: 20543, of which the visible model predicted 966 to be refused (a `supersedes` that crosses channels, speakers or keys, targets a keyed claim or names a newer memory); 1004 retractions, 406 of them of a memory that replaced another (in 306 of 500 (61.2%) pairs).
- Visible operations in A and B whose outcome in the administrative model (every memory) differs from the visible model's: 383 (in 191 of 500 (38.2%) pairs). The store's outcome matched the administrative model in 383 and the visible model in 0.
- Visible memories that hidden activity left replaced in the administrative model while the visible model keeps them active (where a store resolving claims when written would show the viewers a change), A and B together: 816 (in 375 of 500 (75.0%) pairs).
- Hidden operations in A and B together: 38699. Adds 19339 (5249 narrow); relates accepted 7066, refused because an end was not active 1757, refused because the source was older 709, refused across channels 309, refused for authority 80, refused for the key 15, refused for a keyed target 0; relates skipped because the visible end was not written yet 4004 or was not active 243; narrow `supersedes` issued 1646, accepted 635; retractions 1647; membership changes 3530. Outcomes that differed from the administrative model's prediction: 58.
- Deliberate cross-channel `supersedes`: 567 drawn, 370 issued (the rest named a visible label not written yet or archived), 370 of the issued ones refused with ValueError.

## Attribution: where filtering after ranking differs between A and B

Share of pairs in which at least one of the 6 searches differs in each part.
Result set, order and count compare the labels shown; the other parts compare
the rows of labels shown in both A and B.

### after-naive

| Part | all (n=500) | keep (n=169) | copy (n=160) | topic (n=171) | no embedder (n=242) | n-gram embedder (n=258) |
| --- | --- | --- | --- | --- | --- | --- |
| any difference | 100% | 100% | 100% | 100% | 100% | 100% |
| result set | 98% | 98% | 98% | 98% | 98% | 98% |
| order | 67% | 66% | 69% | 67% | 62% | 72% |
| count | 96% | 97% | 96% | 96% | 95% | 97% |
| score | 98% | 98% | 99% | 96% | 95% | 100% |
| lexical_score | 99% | 99% | 99% | 99% | 99% | 100% |
| coverage | 96% | 96% | 96% | 95% | 95% | 96% |
| vector_score | 0% | 0% | 0% | 0% | 0% | 0% |
| matched_by | 42% | 44% | 41% | 42% | 31% | 53% |
| links | 83% | 85% | 84% | 81% | 81% | 86% |
| claim_status | 25% | 27% | 22% | 26% | 21% | 29% |

### after-careful

| Part | all (n=500) | keep (n=169) | copy (n=160) | topic (n=171) | no embedder (n=242) | n-gram embedder (n=258) |
| --- | --- | --- | --- | --- | --- | --- |
| any difference | 100% | 100% | 100% | 100% | 100% | 100% |
| result set | 89% | 91% | 89% | 88% | 89% | 89% |
| order | 75% | 73% | 79% | 73% | 69% | 81% |
| count | 68% | 64% | 71% | 68% | 69% | 66% |
| score | 99% | 99% | 100% | 99% | 99% | 100% |
| lexical_score | 100% | 100% | 100% | 100% | 100% | 100% |
| coverage | 99% | 99% | 99% | 98% | 98% | 99% |
| vector_score | 0% | 0% | 0% | 0% | 0% | 0% |
| matched_by | 51% | 52% | 51% | 49% | 36% | 64% |
| links | 48% | 44% | 51% | 47% | 45% | 50% |
| claim_status | 29% | 30% | 26% | 30% | 24% | 33% |

## Pairs where a check did not hold (first 10 per check)

- noninterference, seed 0: built(A) and built(B) differ at search 0: row 0 differs in lexical_score
- noninterference, seed 1: built(A) and built(B) differ at search 1: row 0 differs in lexical_score
- noninterference, seed 2: built(A) and built(B) differ at search 0: row 0 differs in lexical_score
- noninterference, seed 3: visible operation 36 (v25 related_to v5) in A: predicted ok, got refused: not active; visible operation 41 (v14 supports v23) in B: predicted ok, got refused: not active; built(A) and built(B) differ at search 0: row 0 differs in lexical_score
- noninterference, seed 4: visible operation 4 (v1 supersedes v0) in A: predicted refused: keyed claim, got ok; visible operation 4 (v1 supersedes v0) in B: predicted refused: keyed claim, got ok; built(A) and built(B) differ at search 0: row 0 differs in label
- noninterference, seed 5: built(A) and built(B) differ at search 0: row 0 differs in lexical_score
- noninterference, seed 6: built(A) and built(B) differ at search 1: row 0 differs in lexical_score
- noninterference, seed 7: visible operation 18 (v0 supports v13) in A: predicted ok, got refused: not active; visible operation 18 (v0 supports v13) in B: predicted ok, got refused: not active; built(A) and built(B) differ at get v0: row 0 differs in superseded_by
- noninterference, seed 8: visible operation 9 (v4 supersedes v3) in B: predicted ok, got refused: not active; built(A) and built(B) differ at search 1: row 0 differs in lexical_score
- noninterference, seed 9: built(A) and built(B) differ at search 0: row 0 differs in score
- exactness, seed 2: built(A) and plain(R) differ at search 0: row 0 differs in lexical_score
- exactness, seed 3: built(A) and plain(R) differ at search 0: row 0 differs in lexical_score
- exactness, seed 4: built(A) and plain(R) differ at search 0: row 0 differs in lexical_score
- exactness, seed 5: built(A) and plain(R) differ at search 0: row 0 differs in lexical_score
- exactness, seed 6: built(A) and plain(R) differ at search 1: row 0 differs in lexical_score
- exactness, seed 7: built(A) and plain(R) differ at get v0: row 0 differs in superseded_by
- exactness, seed 8: built(A) and plain(R) differ at search 1: row 0 differs in lexical_score
- exactness, seed 9: built(A) and plain(R) differ at search 0: row 0 differs in score
- exactness, seed 12: built(A) and plain(R) differ at search 0: row 0 differs in lexical_score
- exactness, seed 13: built(A) and plain(R) differ at search 1: row 0 differs in lexical_score
- soundness, seed 0: B: superseded_by in get v3 shows hB24; B: superseded_by in get v4 shows hB21; B: superseded_by in get v13 shows hB26; B: superseded_by in history v3 shows hB24; B: superseded_by in history v4 shows hB21; and 4 more
- soundness, seed 1: B: superseded_by in get v19 shows hB16; B: superseded_by in history v19 shows hB16; B: superseded_by in list all shows hB16
- soundness, seed 2: A: superseded_by in get v1 shows hA4; A: superseded_by in get v6 shows hA4; A: superseded_by in history v1 shows hA4; A: superseded_by in history v6 shows hA4; A: superseded_by in list all shows hA4; and 7 more
- soundness, seed 3: A: superseded_by in get v5 shows hA5; A: superseded_by in history v5 shows hA5; A: superseded_by in list all shows hA5; B: superseded_by in get v5 shows hB16; B: superseded_by in get v17 shows hB14; and 7 more
- soundness, seed 4: A: superseded_by in get v9 shows hA20; A: superseded_by in history v9 shows hA20; A: superseded_by in list all shows hA20; B: superseded_by in get v3 shows hB4; B: superseded_by in get v5 shows hB9; and 10 more
- soundness, seed 5: A: superseded_by in get v6 shows hA18; A: superseded_by in history v6 shows hA18; A: superseded_by in list all shows hA18
- soundness, seed 6: A: superseded_by in get v3 shows hA13; A: superseded_by in get v6 shows hA13; A: superseded_by in history v3 shows hA13; A: superseded_by in history v6 shows hA13; A: superseded_by in list all shows hA13; and 4 more
- soundness, seed 7: A: superseded_by in get v0 shows hA15; A: superseded_by in history v0 shows hA15; A: superseded_by in list all shows hA15; B: superseded_by in get v0 shows hB2; B: superseded_by in history v0 shows hB2; and 1 more
- soundness, seed 8: A: superseded_by in get v11 shows hA9; A: superseded_by in history v11 shows hA9; A: superseded_by in list all shows hA9; B: superseded_by in get v4 shows hB11; B: superseded_by in history v4 shows hB11; and 1 more
- soundness, seed 9: A: superseded_by in get v9 shows hA22; A: superseded_by in history v9 shows hA22; A: superseded_by in list all shows hA22; B: superseded_by in get v7 shows hB13; B: superseded_by in history v7 shows hB13; and 1 more
- retraction, seed 0: built(A) and plain(N) differ at search 0: row 0 differs in lexical_score
- retraction, seed 2: built(A) and plain(N) differ at search 0: row 0 differs in lexical_score
- retraction, seed 3: built(A) and plain(N) differ at search 0: row 0 differs in lexical_score
- retraction, seed 4: built(A) and plain(N) differ at search 0: row 0 differs in lexical_score
- retraction, seed 5: built(A) and plain(N) differ at search 0: row 0 differs in lexical_score
- retraction, seed 6: built(A) and plain(N) differ at search 1: row 0 differs in lexical_score
- retraction, seed 7: built(A) and plain(N) differ at search 1: row 0 differs in lexical_score
- retraction, seed 8: built(A) and plain(N) differ at search 1: row 0 differs in lexical_score
- retraction, seed 9: built(A) and plain(N) differ at search 0: row 0 differs in score
- retraction, seed 10: built(A) and plain(N) differ at get v6: row 0 differs in superseded_by

Limits: the data were written alongside the code, so a pass shows that the
read path holds on these scripts, not on every database. The experiment
compares what reads return, so a fault in one of the store's overlapping
link filters that another filter masks would not show here
(`tests/test_audience.py` checks each layer directly). Timestamps, access
counts and raw IDs are excluded from every comparison, and the order of
`list_memories` is not compared (key rewrites tie on `updated_at`), so a
channel through those would not be detected; access counts are shared by
design (`record_access=False` on every read). The only embedder is the n-gram
baseline; pruning, `restore`, delegation, personal facts and bound writers
are not exercised, and the only admin (cy) writes only hidden memories.
