# Noninterference of reads with viewers (E2, version 2)

Synthetic, scripted data; no LLM. 500 pairs (seeds 0 to 499).
Store under test: schema version 6.
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
| 4 | Noninterference: built(A) equals built(B), and visible operations went as predicted in both | 309 of 500 (61.8%) | every pair | False |
| 5 | Exactness: built(A) equals plain(R) | 430 of 500 (86.0%) | every pair | False |
| 6 | Soundness: no built read in A or B shows or links to a hidden memory; hidden IDs read as missing | 500 of 500 (100.0%) | every pair | True |
| 7 | Retraction: built(A) equals plain(N), and N refused nothing | 430 of 500 (86.0%) | every pair | False |

**Verdict: the built read path fails rules 4, 5, 7 (first failing seed 3)**

Reported, not part of the verdict:

- Sensitivity, first search alone: built(A) and built(A+) differ in 500 of 500 (100.0%).
- after-careful differs between A and B in 500 of 500 (100.0%).
- The visible script's model predicted every visible outcome in R in 500 of 500 (100.0%) (a check of the model itself: R holds no hidden part).
- Visible operations: 20543, of which the visible model predicted 966 to be refused (a `supersedes` that crosses channels, speakers or keys, targets a keyed claim or names a newer memory); 1004 retractions, 406 of them of a memory that replaced another (in 306 of 500 (61.2%) pairs).
- Visible operations in A and B whose outcome in the administrative model (every memory) differs from the visible model's: 383 (in 191 of 500 (38.2%) pairs). The store's outcome matched the administrative model in 383 and the visible model in 0.
- Visible memories that hidden activity left replaced in the administrative model while the visible model keeps them active (where a store resolving claims when written would show the viewers a change), A and B together: 816 (in 375 of 500 (75.0%) pairs).
- Hidden operations in A and B together: 38699. Adds 19339 (5249 narrow); relates accepted 7096, refused because an end was not active 1719, refused because the source was older 697, refused across channels 309, refused for authority 80, refused for the key 15, refused for a keyed target 20; relates skipped because the visible end was not written yet 4004 or was not active 243; narrow `supersedes` issued 1646, accepted 640; retractions 1647; membership changes 3530. Outcomes that differed from the administrative model's prediction: 0.
- Deliberate cross-channel `supersedes`: 567 drawn, 370 issued (the rest named a visible label not written yet or archived), 370 of the issued ones refused with ValueError.

## Attribution: where filtering after ranking differs between A and B

Share of pairs in which at least one of the 6 searches differs in each part.
Result set, order and count compare the labels shown; the other parts compare
the rows of labels shown in both A and B.

### after-naive

| Part | all (n=500) | keep (n=169) | copy (n=160) | topic (n=171) | no embedder (n=242) | n-gram embedder (n=258) |
| --- | --- | --- | --- | --- | --- | --- |
| any difference | 100% | 100% | 100% | 100% | 100% | 100% |
| result set | 97% | 98% | 97% | 97% | 97% | 97% |
| order | 68% | 69% | 67% | 69% | 64% | 72% |
| count | 95% | 96% | 96% | 94% | 93% | 97% |
| score | 98% | 98% | 99% | 96% | 95% | 100% |
| lexical_score | 99% | 99% | 99% | 99% | 99% | 100% |
| coverage | 96% | 96% | 96% | 95% | 95% | 96% |
| vector_score | 0% | 0% | 0% | 0% | 0% | 0% |
| matched_by | 42% | 44% | 41% | 42% | 30% | 54% |
| links | 85% | 87% | 85% | 83% | 83% | 87% |
| claim_status | 27% | 29% | 25% | 27% | 24% | 30% |

### after-careful

| Part | all (n=500) | keep (n=169) | copy (n=160) | topic (n=171) | no embedder (n=242) | n-gram embedder (n=258) |
| --- | --- | --- | --- | --- | --- | --- |
| any difference | 100% | 100% | 100% | 100% | 100% | 100% |
| result set | 89% | 91% | 89% | 88% | 89% | 89% |
| order | 76% | 77% | 79% | 74% | 71% | 82% |
| count | 67% | 65% | 70% | 67% | 69% | 66% |
| score | 99% | 99% | 100% | 99% | 99% | 100% |
| lexical_score | 100% | 100% | 100% | 100% | 100% | 100% |
| coverage | 98% | 99% | 98% | 98% | 98% | 99% |
| vector_score | 0% | 0% | 0% | 0% | 0% | 0% |
| matched_by | 52% | 54% | 52% | 50% | 38% | 66% |
| links | 49% | 47% | 51% | 50% | 48% | 50% |
| claim_status | 31% | 33% | 29% | 32% | 28% | 34% |

## Pairs where a check did not hold (first 10 per check)

- noninterference, seed 3: visible operation 36 (v25 related_to v5) in A: predicted ok, got refused: not active; visible operation 41 (v14 supports v23) in B: predicted ok, got refused: not active; built(A) and built(B) differ at search 2: row 11 differs in links
- noninterference, seed 4: visible operation 24 (v4 supersedes v3) in B: predicted refused: other speaker, got refused: not active; built(A) and built(B) differ at search 3: row 3 differs in links
- noninterference, seed 7: visible operation 18 (v0 supports v13) in A: predicted ok, got refused: not active; visible operation 18 (v0 supports v13) in B: predicted ok, got refused: not active; built(A) and built(B) differ at search 1: row 0 differs in lexical_score
- noninterference, seed 8: visible operation 9 (v4 supersedes v3) in B: predicted ok, got refused: not active
- noninterference, seed 10: visible operation 23 (v3 depends_on v9) in B: predicted ok, got refused: not active; built(A) and built(B) differ at search 0: row 0 differs in label
- noninterference, seed 12: visible operation 36 (v15 supports v7) in B: predicted ok, got refused: not active
- noninterference, seed 13: visible operation 15 (v0 supports v7) in A: predicted ok, got refused: not active; built(A) and built(B) differ at search 4: row 0 differs in label
- noninterference, seed 21: visible operation 16 (v2 related_to v7) in A: predicted ok, got refused: not active; built(A) and built(B) differ at search 3: row 1 differs in score
- noninterference, seed 23: visible operation 22 (v11 depends_on v1) in A: predicted ok, got refused: not active; visible operation 18 (v2 supersedes v3) in B: predicted refused: older, got refused: not active; built(A) and built(B) differ at search 0: row 3 differs in links
- noninterference, seed 24: visible operation 20 (v4 depends_on v12) in B: predicted ok, got refused: not active; built(A) and built(B) differ at search 3: row 1 differs in links
- exactness, seed 3: built(A) and plain(R) differ at search 2: row 11 differs in links
- exactness, seed 13: built(A) and plain(R) differ at search 4: row 0 differs in label
- exactness, seed 21: built(A) and plain(R) differ at search 3: row 1 differs in score
- exactness, seed 23: built(A) and plain(R) differ at search 0: row 3 differs in links
- exactness, seed 51: built(A) and plain(R) differ at search 0: row 3 differs in links
- exactness, seed 64: built(A) and plain(R) differ at search 0: row 0 differs in links
- exactness, seed 71: built(A) and plain(R) differ at search 1: row 7 differs in links
- exactness, seed 72: built(A) and plain(R) differ at search 3: row 5 differs in links
- exactness, seed 79: built(A) and plain(R) differ at search 4: row 2 differs in links
- exactness, seed 87: built(A) and plain(R) differ at search 1: row 1 differs in links
- retraction, seed 3: built(A) and plain(N) differ at search 2: row 11 differs in links
- retraction, seed 13: built(A) and plain(N) differ at search 4: row 0 differs in label
- retraction, seed 21: built(A) and plain(N) differ at search 3: row 1 differs in score
- retraction, seed 23: built(A) and plain(N) differ at search 0: row 3 differs in links
- retraction, seed 51: built(A) and plain(N) differ at search 0: row 3 differs in links
- retraction, seed 64: built(A) and plain(N) differ at search 0: row 0 differs in links
- retraction, seed 71: built(A) and plain(N) differ at search 1: row 7 differs in links
- retraction, seed 72: built(A) and plain(N) differ at search 3: row 5 differs in links
- retraction, seed 79: built(A) and plain(N) differ at search 4: row 2 differs in links
- retraction, seed 87: built(A) and plain(N) differ at search 1: row 1 differs in links

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
