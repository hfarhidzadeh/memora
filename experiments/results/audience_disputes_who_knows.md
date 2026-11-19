# Disputes and who-knows-what (H8)

Synthetic, scripted data; no LLM. 500 pairs (seeds 0 to 499).
Store under test: schema version 7.
See the docstring of `experiments/audience_disputes_who_knows.py` for the design
and decision rule.

| Rule | Check | Held in | Needed | Holds |
| --- | --- | --- | --- | --- |
| 1 | Determinism: H and H2 identical | 500 of 500 (100.0%) | every pair | True |
| 2 | Power: V (visible hidden claims) differs from W | 496 of 500 (99.2%) | at least 50% | True |
| 3 | A reply counts as knowing (planted claim) | 500 of 500 (100.0%) | every pair | True |
| 4 | Presence alone is only access (dana) | 500 of 500 (100.0%) | every pair | True |
| 5 | Noninterference: H equals W for every audience | 500 of 500 (100.0%) | every pair | True |

**Verdict: disputes and who-knows pass**

Reported, not part of the verdict: pairs whose visible script alone produced at least one dispute: 318 of 500 (63.6%).

Limits: synthetic claims and acknowledgements written alongside the code, so a pass
shows the mechanism, not how well acknowledgements capture knowing in practice. Text
equality is exact; the embedding equivalence option is not exercised here.
