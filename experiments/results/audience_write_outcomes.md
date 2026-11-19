# Write outcomes of bound writers (E3)

Synthetic, scripted data; no LLM. 500 pairs (seeds 0 to 499).
Store under test: schema version 6.
See the docstring of `experiments/audience_write_outcomes.py` for the design
and decision rule.

| Rule | Check | Held in | Needed | Holds |
| --- | --- | --- | --- | --- |
| 1 | Determinism: H and H2 identical | 500 of 500 (100.0%) | every pair | True |
| 2 | Power: V (visible hidden claims) differs from W | 485 of 500 (97.0%) | at least 50% | True |
| 3 | Noninterference: H equals W in every visible outcome and read | 500 of 500 (100.0%) | every pair | True |

**Verdict: the write path passes**

Reported, not part of the verdict:

- Pairs whose visible script (world W) was refused at least once: 368 of 500 (73.6%).
- Hidden claims accepted in H: 1217 of 1784.
- Pairs failing noninterference: 0. In 0 a visible outcome differed; in 0 only reads differed. Refusal kinds present in failing pairs: scope 0, lapsed authority 0, other 0.

Limits: the data were written alongside the code, so a pass shows that the
write path holds on these scripts, not on every database. Only bound writers
who cannot see the hidden claims are compared; bob and dana, who can see them,
legitimately get different outcomes. Not exercised: retractions, pruning,
`restore`, other admins, and delegation by anyone but dana to anyone but alice.
