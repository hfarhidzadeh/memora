# LongMemEval as a shared workspace (Stage B)

Real conversation text; no LLM. Store schema version 7. Families (workspaces): bare8 13, full 156, hybrid 13, k1 419.
See the docstring of `experiments/longmemeval_shared_workspace.py` for the design,
the decision rule and the validation criteria.

| Rule | Check | Count | Holds |
| --- | --- | --- | --- |
| 1 | Leakage (ranking 0, links 0, list_memories 0, get_memory 0) | 18177 reads | True |
| 2 | Exactness against the reference: exact 18177, solo-exact 523 | 18177 reads | True |
| 3 | Over-blocking (list_memories 0, get_memory 0, canaries missed 0 of 2704) | | True |

Controls (a failure gates only a claim of no leak or no change):

- C1 harness: single-user stores equal the `_build_store` ranking (419 of 419): True
- C2 leak probe: pooled top 10 shows a hidden memory in 4368 of 4368 (100.0%), an owner evidence turn in 3417 of 4368 (78.2%): True
- C3 canary probe: pooled finds every canary at rank 1 (7098 of 7098 (100.0%)): True
- C4 determinism: two processes, different PYTHONHASHSEED, same digest: True
- C5 placement: plan errors 0, store audit errors 0, canary-word clashes 0: True

**Verdict: zero leakage, exactness and no over-blocking supported; utility no cost**

Utility on owner items (416 questions, 1248 items; any evidence turn, depth 10):

| Arm | RR@10 | hit@10 | hit@5 |
| --- | --- | --- | --- |
| built (filter before) | 0.6464 | 0.8518 | 0.7941 |
| after-all | 0.6568 | 0.8638 | 0.8093 |
| after-k | 0.6399 | 0.7861 | 0.7764 |
| pooled (no access control) | 0.5803 | 0.7861 | 0.7268 |

Built minus after-all in RR@10: -0.0104 (95% interval -0.0172 to -0.0036).

Built minus after-k in RR@10: +0.0064 (95% interval -0.0012 to +0.0149).

Built and after-all give a different top 10 in 95.4% of owner items (needed for a
reading: at least 5%). Utility reading: **no cost**.

Reported, not part of the verdict:

- Items by kind: canary-general 1183, canary-general-left 169, canary-other 4056, canary-own 1352, mixed 1456, other 4368, outsider-owner 169, outsider-self 169, owner 1875, shared 1183, shared-joiner-pre 169, shared-left 169, shared-mixed 169, shared-pre 169, team 676, team-mixed 169, team-mixed-out 169, team-outside 507.
- Other-user items whose asker holds a verbatim copy of an owner evidence turn: 14 of 4368 (0.3%).
- Other-user items whose top 10 contains the (non-numeric) answer text: built 120, pooled 1032 (of 3453 items with a text answer).

Limits: real text but invented access structure (private channels, one shared and one team history, canaries). Exact-text copies are allowed by design; near-duplicate text, retraction, rebuild and the `record_access` side channel are not tested here. A human spot-check of other-user results is still required before anyone relies on this.
