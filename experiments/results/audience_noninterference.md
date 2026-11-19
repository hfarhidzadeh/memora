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
| 4 | Noninterference: built(A) equals built(B), and visible operations went as predicted in both | 500 of 500 (100.0%) | every pair | True |
| 5 | Exactness: built(A) equals plain(R) | 500 of 500 (100.0%) | every pair | True |
| 6 | Soundness: no built read in A or B shows or links to a hidden memory; hidden IDs read as missing | 500 of 500 (100.0%) | every pair | True |
| 7 | Retraction: built(A) equals plain(N), and N refused nothing | 500 of 500 (100.0%) | every pair | True |

**Verdict: the built read path passes**

Reported, not part of the verdict:

- Sensitivity, first search alone: built(A) and built(A+) differ in 500 of 500 (100.0%).
- after-careful differs between A and B in 500 of 500 (100.0%).
- The visible script's model predicted every visible outcome in R in 500 of 500 (100.0%) (a check of the model itself: R holds no hidden part).
- Visible operations: 20543, of which the visible model predicted 966 to be refused (a `supersedes` that crosses channels, speakers or keys, targets a keyed claim or names a newer memory); 1004 retractions, 406 of them of a memory that replaced another (in 306 of 500 (61.2%) pairs).
- Visible operations in A and B whose outcome in the administrative model (every memory) differs from the visible model's: 0 (in 0 of 500 (0.0%) pairs). The store's outcome matched the administrative model in 0 and the visible model in 0.
- Visible memories that hidden activity left replaced in the administrative model while the visible model keeps them active (where a store resolving claims when written would show the viewers a change), A and B together: 830 (in 377 of 500 (75.4%) pairs).
- Hidden operations in A and B together: 38699. Adds 19339 (5249 narrow); relates accepted 7409, refused because an end was not active 1333, refused because the source was older 753, refused across channels 319, refused for authority 83, refused for the key 15, refused for a keyed target 24; relates skipped because the visible end was not written yet 4004 or was not active 243; narrow `supersedes` issued 1646, accepted 735; retractions 1647; membership changes 3530. Outcomes that differed from the administrative model's prediction: 0.
- Deliberate cross-channel `supersedes`: 567 drawn, 370 issued (the rest named a visible label not written yet or archived), 370 of the issued ones refused with ValueError.

## Attribution: where filtering after ranking differs between A and B

Share of pairs in which at least one of the 6 searches differs in each part.
Result set, order and count compare the labels shown; the other parts compare
the rows of labels shown in both A and B.

### after-naive

| Part | all (n=500) | keep (n=169) | copy (n=160) | topic (n=171) | no embedder (n=242) | n-gram embedder (n=258) |
| --- | --- | --- | --- | --- | --- | --- |
| any difference | 100% | 100% | 100% | 100% | 100% | 100% |
| result set | 97% | 98% | 96% | 97% | 97% | 97% |
| order | 68% | 69% | 67% | 68% | 64% | 72% |
| count | 95% | 96% | 95% | 94% | 93% | 97% |
| score | 98% | 98% | 99% | 96% | 95% | 100% |
| lexical_score | 99% | 99% | 99% | 99% | 99% | 100% |
| coverage | 96% | 96% | 96% | 95% | 95% | 96% |
| vector_score | 0% | 0% | 0% | 0% | 0% | 0% |
| matched_by | 43% | 45% | 42% | 42% | 31% | 54% |
| links | 85% | 87% | 85% | 84% | 83% | 87% |
| claim_status | 27% | 29% | 25% | 27% | 24% | 30% |

### after-careful

| Part | all (n=500) | keep (n=169) | copy (n=160) | topic (n=171) | no embedder (n=242) | n-gram embedder (n=258) |
| --- | --- | --- | --- | --- | --- | --- |
| any difference | 100% | 100% | 100% | 100% | 100% | 100% |
| result set | 89% | 91% | 88% | 88% | 89% | 89% |
| order | 76% | 77% | 79% | 73% | 70% | 82% |
| count | 68% | 66% | 71% | 68% | 70% | 67% |
| score | 99% | 99% | 100% | 99% | 99% | 100% |
| lexical_score | 100% | 100% | 100% | 100% | 100% | 100% |
| coverage | 98% | 99% | 98% | 98% | 98% | 99% |
| vector_score | 0% | 0% | 0% | 0% | 0% | 0% |
| matched_by | 53% | 54% | 54% | 50% | 39% | 65% |
| links | 49% | 46% | 51% | 51% | 48% | 50% |
| claim_status | 31% | 33% | 29% | 32% | 28% | 34% |

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
