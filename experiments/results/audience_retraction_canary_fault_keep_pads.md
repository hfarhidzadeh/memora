# Retraction canaries (H5)

Synthetic, scripted data; no LLM. 500 trials (seeds 0 to 499).
Store under test: schema version 7; secure_delete: True; injected fault: keep-pads.
See the docstring of `experiments/audience_retraction_canary.py` for the design
and decision rule.

| Rule | Check | Held in | Needed | Holds |
| --- | --- | --- | --- | --- |
| 1 | Determinism: the same seed gives the same result | 500 of 500 (100.0%) | every trial | True |
| 2 | Probe: before retraction the scan (and a plain backup copy) finds every canary | 500 of 500 (100.0%) | every trial | True |
| 3 | Specificity: only memories derived from retracted messages change | 500 of 500 (100.0%) | every trial | True |
| 4 | Erasure: no retracted canary in any stored value | 500 of 500 (100.0%) | every trial | True |
| 5 | Rebuild: rebuildable memories rebuilt from exactly their remaining sources | 500 of 500 (100.0%) | every trial | True |
| 6 | Backup: with a key file, an earlier copy yields no retracted canary | 0 of 500 (0.0%) | every trial | False |

**Verdict: retraction does not erase (rules 6 failed, first failing seed 0)**

Reported, not part of the verdict:

- Memories derived from a retracted message: 1320 (all trials); stale memories in the rebuild queue: 1259; of those, with a remaining source to rebuild from: 684 (in 406 trials); trials with such a memory where rebuild failed: 0.
- Trials in which the raw bytes of the database file still contain a retracted canary: 0 of 500 (0.0%). The scan of stored values reads live rows; SQLite can keep old bytes in free space and backups, which this experiment does not claim to erase.

Limits: synthetic text written alongside the code, so a pass shows the mechanism, not
how often derived text repeats a source word in practice. A derived memory is linked
to its sources only through the source list the caller supplies; text copied into a
memory that does not list the message is not found by the store and not tested here.
