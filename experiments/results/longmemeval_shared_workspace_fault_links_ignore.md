# LongMemEval as a shared workspace (Stage B)

Real conversation text; no LLM. Store schema version 7; injected fault: links-ignore. Families (workspaces): full 4.
See the docstring of `experiments/longmemeval_shared_workspace.py` for the design,
the decision rule and the validation criteria.

| Rule | Check | Count | Holds |
| --- | --- | --- | --- |
| 1 | Leakage (ranking 0, links 1450, list_memories 0, get_memory 0) | 408 reads | False |
| 2 | Exactness against the reference: exact 175, links 233 | 408 reads | False |
| 3 | Over-blocking (list_memories 0, get_memory 0, canaries missed 0 of 64) | | True |

Controls (a failure gates only a claim of no leak or no change):

- C1 harness: single-user stores equal the `_build_store` ranking (0 of 0): not run
- C2 leak probe: pooled top 10 shows a hidden memory in 96 of 96 (100.0%), an owner evidence turn in 81 of 96 (84.4%): True
- C3 canary probe: pooled finds every canary at rank 1 (168 of 168 (100.0%)): True
- C4 determinism: two processes, different PYTHONHASHSEED, same digest: not run
- C5 placement: plan errors 0, store audit errors 0, canary-word clashes 0: True

**Verdict: fails (rules 1, 2 failed, first failing full seed 0 workspace 0 item shared-pre#0)**

Utility on owner items (32 questions, 32 items; any evidence turn, depth 10):

| Arm | RR@10 | hit@10 | hit@5 |
| --- | --- | --- | --- |
| built (filter before) | 0.6328 | 0.8750 | 0.8750 |
| after-all | 0.6250 | 0.8750 | 0.8750 |
| after-k | 0.6094 | 0.8438 | 0.8438 |
| pooled (no access control) | 0.5551 | 0.8438 | 0.7500 |

Built minus after-all in RR@10: +0.0078 (95% interval -0.0312 to +0.0547).

Built minus after-k in RR@10: +0.0234 (95% interval -0.0234 to +0.0781).

Built and after-all give a different top 10 in 96.9% of owner items (needed for a
reading: at least 5%). Utility reading: **unclear**.

Reported, not part of the verdict:

- Items by kind: canary-general 28, canary-general-left 4, canary-other 96, canary-own 32, mixed 32, other 96, outsider-owner 4, outsider-self 4, owner 32, shared 28, shared-joiner-pre 4, shared-left 4, shared-mixed 4, shared-pre 4, team 16, team-mixed 4, team-mixed-out 4, team-outside 12.
- Other-user items whose asker holds a verbatim copy of an owner evidence turn: 0 of 96 (0.0%).
- Other-user items whose top 10 contains the (non-numeric) answer text: built 0, pooled 12 (of 60 items with a text answer).

Limits: real text but invented access structure (private channels, one shared and one team history, canaries). Exact-text copies are allowed by design; near-duplicate text, retraction, rebuild and the `record_access` side channel are not tested here. A human spot-check of other-user results is still required before anyone relies on this.

Fault validation: pattern held in 4 of 4 workspaces (as predicted).
