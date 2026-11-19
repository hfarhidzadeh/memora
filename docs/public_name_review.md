# Public name review — 2026-11-19

**Decision: choose another public name before release.** “Memora” has an
exact-name collision with published work in agent memory, including work
evaluated on LongMemEval. Treat the repository's current name as a working name.

Primary sources checked:

| Published use | Evidence | Relevance |
| --- | --- | --- |
| *Memora: A Harmonic Memory Representation Balancing Abstraction and Specificity*, Xia et al. | [arXiv:2602.03315](https://arxiv.org/abs/2602.03315), first submitted February 3, 2026; [official Microsoft repository](https://github.com/microsoft/Memora) | An agent-memory system, with memory updates and retrieval; a direct collision in this project's field. |
| Memora benchmark, introduced in *From Recall to Forgetting: Benchmarking Long-Term Memory for Personalized Agents*, Uddin et al. | [ACL Findings 2026 publication](https://aclanthology.org/2026.findings-acl.1337/), July 2026; [authors' repository](https://github.com/geniesinc/Memora) | A second exact-name use in long-term memory evaluation. |

This establishes a research/software naming collision. A different package
suffix, such as the existing `memora-agent-memory`, would still leave the paper
title, README, and `memora` command ambiguous.

Before public sharing, select and check a replacement name, then update the
README/title, package metadata, CLI name/help/examples, and experiment prose.
Preserve historical provenance when describing old runs. The internal Python
module is already `dynmemory`; retaining it does not require choosing that as
the public brand. No replacement name has been selected or cleared by this review.
