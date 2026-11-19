# Public name review — 2026-11-19

**Decision: rename this project to Memvane (repository/package/CLI: `memvane`).**
The owner selected this name on 2026-11-19. Initial exact-name web searches,
including GitHub, arXiv, and PyPI queries, found no obvious published memory
system or software package named Memvane; this is a preliminary collision check.

The former name “Memora” has an
exact-name collision with published work in agent memory, including work
evaluated on LongMemEval.

Primary sources checked:

| Published use | Evidence | Relevance |
| --- | --- | --- |
| *Memora: A Harmonic Memory Representation Balancing Abstraction and Specificity*, Xia et al. | [arXiv:2602.03315](https://arxiv.org/abs/2602.03315), first submitted February 3, 2026; [official Microsoft repository](https://github.com/microsoft/Memora) | An agent-memory system, with memory updates and retrieval; a direct collision in this project's field. |
| Memora benchmark, introduced in *From Recall to Forgetting: Benchmarking Long-Term Memory for Personalized Agents*, Uddin et al. | [ACL Findings 2026 publication](https://aclanthology.org/2026.findings-acl.1337/), July 2026; [authors' repository](https://github.com/geniesinc/Memora) | A second exact-name use in long-term memory evaluation. |

This establishes a research/software naming collision. A different package
suffix, such as the former `memora-agent-memory`, would still leave the paper
title, README, and `memora` command ambiguous.

The rename updates the README/title, package metadata, CLI name/help/examples,
and current experiment code. Existing result reports retain Memora as the name
used for those historical runs; published systems cited above also retain their
names. The Python module remains `dynmemory`, and the default database filename
remains `.memora.sqlite3` so an upgrade continues to open the existing store.
