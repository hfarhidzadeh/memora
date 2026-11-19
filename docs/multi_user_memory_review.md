# Shared memory for multi-user AI assistants: literature review and a proposed design

Project naming note (2026-11-19): references to Memvane below identify this
repository, previously called Memora. Historical numerical results retain their
original provenance and provisional status.

- **Date:** 2026-11-13
- **Scope:** long-term memory for LLM agents, with emphasis on memory that several
  people share (a team, a group chat, an organisation) rather than one user's
  personal memory.
- **How this was made:** an AI model searched arXiv and the web and read about
  30 papers in their HTML full text, though for most of them part of the text
  arrived only as an automated summary. It saw about 100 more at abstract or
  search-snippet level and wrote this synthesis. Every number below appeared in a
  source during that search. Almost all are the authors' own claims in 2026
  preprints that have not been peer reviewed. Check a number against the paper's
  PDF before citing it.
- **What is a finding and what is a proposal:** sections 2–6 report the
  literature. Sections 7–10 (the proposed design, the Memvane changes and the
  evaluation plan) are the model's suggestions for the researcher to judge.

Reading levels used in the reference list (section 12):

- **[F]**: full HTML text read; some sections may have come back only as an
  automated summary.
- **[A]**: abstract or search snippet only.
- **[M]**: cited from the model's memory, not checked during this search
  (mostly classic papers).

---

## 1. Summary

Research on memory for one user has settled several questions, listed in
section 2. Research on memory shared by many users only started in earnest in
2026, and it has split into two camps that do not meet.

- **Quality systems** model who said what. SpeakerMem-R1, EGMemory and the
  SocialMemBench probes keep speaker and thread structure and beat generic
  memory systems on group-chat benchmarks. None of them controls who may read a
  memory, handles deletion, or decides what happens when two people disagree.
- **Governance systems** control who may read what. Collaborative Memory, AIM,
  MemClaw, kernel-managed memory, Authorization Before Context, MAP-Graph and
  MemTX enforce scopes or audiences. None models human speakers or
  disagreement between people.
- **Derived memories are the weak point.** A derived memory is a summary or
  profile built from several people's messages. Only MAP-Graph defines who may
  read one, and its readers are agents, not people. Only single-user systems
  remove derived memories when a source is deleted.

The evidence that this matters:

- **Pooled memory leaks with no attacker.** Benign use contaminated other users'
  results in 57–71% of cases in one study and 70–100% in another.
- **No method does everything.** On GateMem, which scores usefulness, access
  control and forgetting together, no method does well on all three. Feeding
  the whole history to the model beats every memory system.
- **Group-chat memory is weak.** On GroupMemBench, the best memory system scores
  46% and plain BM25 scores 43%. Knowledge-update questions top out at 27%.

Section 7 proposes **audience-scoped memory**. Every stored item is either an
immutable message, recorded with its speaker and channel, or a derivation that
lists the exact messages it was computed from. One rule
over that structure decides each of three things: who may read a memory, whether
a new statement replaces an old one or opens a recorded disagreement, and what
must be rebuilt when something is deleted. The researcher's design decisions of
2026-11-14 are built in (section 10).

Each rule has a precedent, and two have close ones (section 7.8). The new parts
are narrower:

- the combination of the three rules, for human members;
- building summaries per audience after filtering, instead of withholding them;
- "only the same speaker, the person a fact is about, or an admin may
  overwrite" as the rule between people;
- counting someone as knowing a fact only when they stated or acknowledged it.

---

## 2. What single-user memory research has established

| Finding | Evidence | How solid |
| --- | --- | --- |
| Keep the raw, timestamped messages as the source of truth; use extracted facts as extra index keys, not as a replacement | LongMemEval: facts as extra keys +9.4% recall, +5.4% accuracy, while replacing messages with facts lowers accuracy. Extraction compresses about 35:1 and drops temporal markers and one-off details (2603.04814). Engram: 83.6% vs 73.2% full context on LongMemEval_S at 9.6k vs 79k tokens, and facts alone lost recall. A filesystem agent over raw transcripts scored 74.0% vs 68.5% for Mem0's graph variant on LoCoMo (vendor blog). | Strong: many independent sources |
| Below about 100k tokens, a strong model reading the whole history usually wins on accuracy; memory wins on cost and latency | Mem0's own table: full context 72.90 vs Mem0 66.88 on LoCoMo, at p95 latency 17.12 s vs 1.44 s. Long-context GPT-5-mini vs a Mem0-style system: LoCoMo 92.85 vs 57.68, LongMemEval 82.40 vs 49.00 (2603.04814). | Strong |
| Lexical retrieval (BM25) is a hard baseline; hybrid lexical plus dense retrieval is the common winning recipe | Mem0's 2026 rewrite moved to semantic + BM25 + entity retrieval. BM25 matches or beats most memory systems on GroupMemBench and on MemoryAgentBench's conflict task. | Strong |
| Invalidate superseded facts instead of deleting them, but be careful how old values are shown | Zep, Mem0's graph variant, Engram and Kumiho all invalidate rather than delete. Showing superseded edges to the answerer lowered past-value accuracy, 72/81 vs 79/81 with no significance test (Ground Truth First, 2607.21962). Memvane's own result: archiving old versions cut MRR on past-value questions from 0.53 to 0.24. | Moderate |
| Updates and conflicts are the weakest skill | MemoryAgentBench FactConsolidation: multi-hop accuracy at most about 7% for every method; Mem0 18 single-hop vs 60 for long-context GPT-4o. STALE (implicit conflicts): best model 55.2%. Self-maintained bounded memory cut gpt-5.4 from 92% to 77% on LongMemEval update questions (n = 25). | Strong in direction; small samples |
| LLM extraction adds errors that accumulate | HaluMem: errors made at extraction and update propagate to answers. TANGLE: extraction pipelines kept the full conflict for between 46.2% (MemOS) and 91.7% (Letta) of instances. "Manufactured Confidence" (2606.29279, [A]): consolidation turned a hedged remark into a confident dated fact. | Strong |
| Background consolidation (outside the response path) pays off | Sleep-time compute: about 5x less test-time compute at equal accuracy. LightMem, LangMem and Engram all consolidate offline. | Moderate |
| Benchmark numbers are fragile | LoCoMo answer key: 99 of 1,540 answers wrong (6.4%), and the usual judge accepted 62.81% of vague but wrong answers (community audit). Reported LoCoMo scores for Zep range from 58 to 84 depending on who ran it. Token-F1 and an LLM judge differed by 27.5 points on identical outputs (2606.22030). | Strong |

---

## 3. Multi-user memory in 2026

### 3.1 Systems that model speakers

- **SpeakerMem-R1** (2609.26780) [F]
  - **What it stores:** two tracks. One is a verbatim message track (speaker,
    time, channel). The other has four derived layers: person core, person
    profile (including what one person thinks of another), group interaction and
    group insight (norms, consensus).
  - **Record format:** each derived record keeps *source* and *owner* separately.
    "Alice believes Bob agreed" has source Alice and owner Bob. Updates append a
    linked node instead of overwriting.
  - **Writer:** a 3B model trained with RL decides add, update or nothing.
  - **Results:** 47.9 vs 44.6 (BM25) on GroupMemBench with one reader model, but
    42.7 vs 46.2 (below BM25) with another.
  - **Missing:** no rule for conflicting statements by different speakers; no
    access control.
- **EGMemory, "Propose, Verify, Commit"** (2609.23465) [F]
  - **Write path:** a writer LLM proposes an update together with a source quote.
    Deterministic code checks that the proposed value appears in the source
    before committing it. Superseded states stay linked.
  - **Results:**
    - GroupMemBench: 68.2 vs 45.5 for BM25.
    - EverMemBench: 77.9.
    - Removing versioned state costs 5 points; removing verification costs 2.3.
  - **Caveats:** a single judge model (Kimi-K3), with no human validation reported.
  - **Missing:** access control, which the authors list as a limitation.
- **SocialMemBench probes** (2605.17789) [F]
  - **Subject-Mem** tags each extracted fact with the person it is about, which
    raised attribution accuracy from 0.25–0.46 to 0.78.
  - **SMG** is a graph with dissent and knows-about edges, which raised
    group-decision accuracy to 0.69.
  - **Caveat:** both probes were built by the benchmark's single author.

**Common lesson.** All three keep who said what as structure, not as text, and
link versions instead of overwriting them. None governs access.

### 3.2 Systems that govern access

| System | Memory unit and metadata | How access is enforced | Updates and conflicts | Derived memories | Deletion | Evaluation |
| --- | --- | --- | --- | --- | --- | --- |
| Collaborative Memory (2505.18279) [F] | LLM-written key–value fragments with creation time, contributing user, agents and resources; private and shared tiers | Deterministic predicate over time-varying user–agent and agent–resource graphs. Shared-tier confidentiality depends on an LLM prompt that strips user details at write time. The formula as written has no owner test. | Not addressed | One contributing user per fragment, so a fragment built from several users has no defined owner | Revocation blocks reads; fragments remain | Accuracy above 0.90 and up to 61% fewer resource calls; no leakage test; "provable adherence" claimed but no proof found |
| AIM (2609.12320) [F] | Extracted facts with user, agent, binary public/private visibility | Visibility filter pushed into the vector index ("deterministically at the retrieval index level"); visibility itself assigned partly by an LLM | Create, update or no-op, with temporal supersession; conflict resolution left to future work | Not addressed | Deletes best match; no ownership check in the algorithm, though the text claims one | MUMBench: 672 interactions, 93 users; visibility 95–96%, strict operation accuracy 58.8%; far below Mem0 on LoCoMo |
| MemClaw (2606.24535) [F] | Tenant, fleet, agent, scope, optional triple, supersession links, provenance | Deterministic scope predicate on every path | Supersession by time and policy; a near-duplicate gate rejected 206 of 400 writes, hiding contradictions from the detector | Lineage recorded; scope inheritance not addressed | Not addressed | Measured on the authors' production service, no baselines; found a get-by-id path that checked only the tenant (leak rate 0.439 in search) |
| Kernel-managed shared memory (2609.10144) [F] | user, owner agent, type, private/shared | Kernel applies "owner or shared" rule; agents cannot bypass it | Not addressed | Not addressed | Not addressed | 0 of 450 private exposures; found cross-user contamination caused by wrong identity resolution under concurrency |
| Authorization Before Context (2608.17148) [F] | Each item records the audience present when it was written | Admit an item only if every current viewer is in its audience; checked before the prompt is assembled; fails closed when membership is unknown | Out of scope | Cross-audience summaries are withheld from non-owners; the paper accepts over-blocking rather than infer unions | Out of scope | 0 of 79 synthetic scenarios leaked, true by construction; single author, workshop paper |
| MemTX (2607.23929) [F] | Records with evidence, permissions, provenance, validity; private/shared/public scope | Snapshot-isolated transactions with four commit checks | Higher authority supersedes, lower aborts, **equal authority is quarantined for review**; stale late writes abort | Summaries quarantined when an ancestor is revoked | Typed rollback | Own suite of 90 cases; conflicts are between agents and tools, not human speakers |
| MemLineage (2605.14421) [F] | Signed entries in a Merkle log, derivation graph inferred from what was in the LLM's context | Integrity only: blocks sensitive actions justified by untrusted ancestors | Not addressed | Lineage used for trust, not for read access | Tombstones; content not physically removed | 0% attack success on its own harness |

Also relevant:

- **MAP-Graph** (2608.10509) [F]: a derived item's "permission scope is the
  intersection of the referenced scopes."
  - Scopes are sets of agents.
  - If the intersection is empty, the item falls back to its writer.
  - There is no supersession or conflict handling.
  - Revoking a source flags deeper descendants only the next time their trust
    is checked.
- **Participant-aware access control** (2509.14608) [A]: content must be
  "explicitly authorized for all users involved in the interaction." It is
  deployed in Copilot Tuning; the abstract does not cover summaries.
- **COLLABMEM's TMS-GRAPH++** (EMNLP 2026) [F]:
  - Each event carries its writers, a credibility value, a contradiction
    marker and "the set of roles allowed to see it."
  - The current value is chosen by contradiction flag, credibility and
    ingestion time. Older values stay visible with their writers.
  - The principals are agent roles.
- **When Memory Becomes Authority** (2608.01679) [F]:
  - Only a direct statement by the user may update the user's profile.
  - Reports by third parties are stored with attribution.
  - It is single-user.
- **MutMem** (2608.02843) [F] signs every change to a memory's weight. Its only
  deletion is a whole-store purge.
- **MAPLE-Guard** (2608.00426) [A] gates writes, promotion from private to
  shared, and reuse across agents. Attack success fell from 38.2% to 0.9%.

**Common lesson.** Deterministic checks outside the model work; LLM-based
gating and LLM sanitisation do not count as a boundary. For derived memories:

- MAP-Graph intersects scopes, but for agents.
- Authorization Before Context withholds cross-audience summaries.
- Collaborative Memory cannot represent them.
- The others do not discuss them.

### 3.3 Benchmarks

| Benchmark | Data | Size | What it tests | Headline result | Main limits |
| --- | --- | --- | --- | --- | --- |
| GroupMemBench (2605.14498) [F] | GPT-5-generated Teams-style channels and threads, 4 domains | 120K messages; 185 non-abstention questions per domain | Multi-hop, knowledge update, term ambiguity, user-implicit, temporal, abstention; answers depend on who asks | Hindsight 46.0, BM25 43.2, HippoRAG 39.7, A-Mem 35.1, Mem0 25.7; best knowledge-update 27.1. Retrieval failures cause 41–79% of errors. About 25% of replies are injected disagreements. | Synthetic; generator and judge share a family; questions filtered against a baseline solver; no access control or deletion |
| EverMemBench (2602.01313) [F] | Gemini-generated, 5 projects × 3 groups, 170 employees, one simulated year | 51,023 turns, about 4.2M tokens, 2,400 questions | Recall, temporal, constraints, proactivity, update, style, skill, role | Full-context Gemini 72.6 vs best memory system 59.3 (MemOS); accuracy fell from 54.5% (one group) to 19.7% (three groups) | Synthetic; no access control or deletion; some abstract claims do not match its tables |
| SocialMemBench (2605.17789) [F] | LLM-generated social groups of 4–30 members, human-reviewed | 7,355 turns, 1,031 questions | Attribution, group decision, theory of mind, norm vs individual, temporal shift, departed member | Mem0, LangMem, Graphiti and Cognee score 0.12–0.18; raw-turn retrieval 0.345; full context 0.369 | About 7k tokens per network, so it fits in context; single author and judge; "departed member" rewards retention, the opposite of revocation |
| GateMem (2606.18829) [F] | LLM-generated multi-party episodes in medical, office, education and household settings (all synthetic) | 91 episodes, 2,218 checkpoints, 13.4 principals and 223 turns per episode | Utility, access control, forgetting after a deletion request; score = U × (1 − A) × (1 − F) | GPT-5.4 long-context 54–80 across domains; Mem0 21–29; "no method simultaneously achieves" all three | Episodes fit in context; forgetting checked at the answer only; no attribution or cross-person conflict |
| PiSAs (2607.05318) [F] | Synthetic multi-user tasks, 9 families | 265 scenarios | Contextual integrity: task appropriateness and per-user visibility across outputs, agent messages and memory | Shared plus private memory cut message appropriateness violations from 9.9% to 6.4% but raised visibility violations from 14.9–28.0% to 35.6–50.0%, which the authors call a relocation | Single task per instance; unintentional leakage only |
| MemLeak (2610.04195) [A] | Pooled multi-tenant vector store | n/a | Incidental cross-user retrieval | 70–100% leakage under same-team pooling; only a hard ownership check restored the clean baseline, at about 1.4 ms per query | Abstract only |
| No Attacker Needed (2604.01350) [A] | Shared-state agents | n/a | Benign cross-user contamination | 57–71% contamination; write-time sanitising helped for conversation but not for executable artefacts | Abstract only |
| COLLABMEM (EMNLP 2026) [F] | Hand-written scenarios; no arXiv version found | 58 scenarios, 1,118 probes | Sharing, access, routing, provenance, conflict resolution, retention (meaning compression that keeps the reason for a change, not deletion) | Its TMS-GRAPH++ averages 69.6%; shared baselines leak protected facts on 40.0% and 61.0% of access probes; even the oracle reaches only 49.4% on retention | Principals are agent roles, not human users |
| TANGLE (2608.13921) [F] | Synthetic personas | 541 instances | Conflicts with no single right answer: context-dependent values, oscillating behaviour, contradicting sources | Strongest model below 15/20 on the rubric; extraction pipelines lose the conflict in 8–54% of cases | One persona; its "sources" are records, not multiple people |
| MemConflict (2605.20926) [F] | Synthetic single users | 12 users, about 124 queries each | Dynamic update, static contradiction, conditional value | Best macro accuracy 0.55 (MemOS); static contradictions hardest | Single user; always has one gold answer |
| BEMULA (2610.18052) [A] | Dialogues with forget requests | 135 forget requests, 1,742 questions | Deletion with paired no-deletion controls | No system passes more than 1.5% of requests end to end | Appears single-user |

Capability coverage (Y = yes, P = partial, N = no):

| Benchmark | Several humans | Who said it | People disagree | Updates | Abstention | Permissions / leakage | Deletion | Too long for context |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LongMemEval | N | N | N | Y | Y | N | N | Y (M set) |
| GroupMemBench | Y | P | P | Y | Y | N | N | Y |
| EverMemBench | Y | Y | P | Y | N | N | N | Y |
| SocialMemBench | Y | Y | P | P | N | N | N | N |
| GateMem | Y | N | N | Y | P | Y | P | N |
| PiSAs | Y | N | N | N | N | Y | N | N |
| COLLABMEM | agent roles | Y | Y | Y | ? | Y | N | N |
| TANGLE | N | P | P | P | P | N | N | N |
| BEMULA | N | N | N | N | N | N | Y | N |

No benchmark combines several human users, a history longer than the context
window, attribution, disagreement between people, permissions and deletion.

---

## 4. Failure modes, with the evidence for each

1. **Pooling leaks without an attacker.** Similarity search in a shared store
   returns other people's memories because they are relevant, not because
   anyone asked for them. MemLeak found 70–100% under team pooling and No
   Attacker Needed found 57–71%. Kernel-managed memory found leakage caused by
   identity mix-ups under concurrency. PiSAs found that adding shared memory
   *moves* violations into memory rather than removing them.
2. **Ingestion discards who said what.** GroupMemBench's failure cases include
   topic summaries that lose the speaker (Mem0), wrong-speaker shadowing (A-Mem)
   and over-broad multi-speaker matches (HippoRAG, MemGPT, GraphRAG).
   SocialMemBench lists single-stream conflation, entity merging and temporal
   overwrite. EverMemBench found every pipeline strips the style markers a
   question needed.
3. **"Newest wins" turns disagreement into an overwrite.** Every surveyed
   update mechanism (Mem0's operations, Zep's edge invalidation, AIM's
   supersession) assumes one writer. GroupMemBench injects disagreement into
   about 25% of replies, and knowledge-update accuracy tops out at 27%. TANGLE
   shows some conflicts have no single answer, yet fixed rules "commit to one
   value on every applicable instance." MemTX is the only system that refuses
   to auto-resolve equal-authority conflicts, and its conflicts are between
   agents and tools, not people.
4. **Derived memories escape most rules.**
   - Authorization Before Context withholds cross-audience summaries rather than
     handling them.
   - Collaborative Memory's provenance has a single contributing user.
   - MemLineage tracks derivation for trust only.
   - AIM does not address derived memories.
   - MAP-Graph intersects source scopes, but only for agents.
   - Zep removes a fact once no message supports it, but does not regenerate
     summaries that other messages still support. Deleted content can therefore
     survive in summaries (vendor blog, July 2026).
   - MemoRepair (2605.07242) [A] shows the cost: without provenance, 69.8–94.3%
     of answers were exposed to invalidated memory through derived artefacts.
     With complete provenance this fell to 0%.
   - AfterDeleteBench (2610.31366) [A] rebuilt deleted secrets from separately
     stored pieces in 62–99% of cases. Provenance-aware deletion removed that
     recovery.
5. **Deletion does not reach derived artefacts.** GateMem found that retained
   history lets agents reconstruct or confirm deleted content, especially under
   yes/no probes. Ghost Vectors (2606.18497) [A] recovered 25.5% of names and
   46.4% of locations from soft-deleted vectors; encrypting each vector and
   destroying the key brought this to 0%. BEMULA: no system passes more than
   1.5% of forget requests.
6. **Members and tools can poison shared memory.**
   - MINJA (2503.03704) [A] injects memory through ordinary queries.
   - Sleeper and memory-hopping attacks persist across sessions and spread
     between assistants: 60–80% of agents in simulation (2609.35576) [A].
   - Defences that bind origin at write time work best on their authors'
     harnesses: MAPLE-Guard, MemLineage and TMA-NM (2606.24322) [A]. TMA-NM
     reports 68% laundering through summarisation against existing defences.
7. **Model judgement is not a boundary.**
   - PiSAs: no configuration reliably restricts sharing.
   - CIMemories: up to 69% attribute-level violations.
   - GateMem: policy-prompted retrieval over-refuses.
   - Multi-User LLM Agents (2604.08567) [F]: instruction-following degrades when
     users conflict.
   - The scoping review (2609.14062) [F]: only 32 of 118 documented
     multi-person risks (27.1%) had an implemented or tested response.

---

## 5. Theory worth borrowing

| Idea | Source | What it implies for shared memory |
| --- | --- | --- |
| Transactive memory: groups keep a directory of who knows what instead of duplicating knowledge | Wegner 1987; Lewis 2003 [M] | Keep a who-knows-what index over the *human* members. The two 2026 "transactive memory" agent papers (2606.19911, 2604.03295) [F] track agents' skills only; neither models people. |
| Common ground: information is shared only once it has been acknowledged; speakers design messages for their audience | Clark & Brennan 1991; Clark 1996 [M] | Store the audience of every message. Being present is not the same as knowing. FANToM shows models fail to track who was absent. |
| Source monitoring: people keep content but lose where it came from | Johnson, Hashtroudi & Lindsay 1993 [M] | Make the source a mandatory, immutable field that consolidation cannot drop. MemIR (2605.25869) [A] names this failure in LLM memory. |
| Collaborative inhibition and social contagion: groups recall less than their members' pooled memories, and one member's error spreads | Weldon & Bellinger 1997; Roediger et al. 2001; Rajaram & Pereira-Pasarin 2010 [M] | One shared ranking may bury knowledge held by only one member, and one user's mistake becomes everyone's. "Misremember collectively" (2602.00428) [A] shows this in multi-agent systems. |
| Truth maintenance: every belief records its justification, so retraction propagates; an assumption-based variant (ATMS) keeps several contexts at once | Doyle 1979; de Kleer 1986 [M] | Derivations carry source lists for cascading invalidation. An ATMS-style label gives "true according to Alice" side by side with "true according to Bob". |
| Truth discovery: estimate source reliability and fact truth jointly; copied claims are not independent votes | TruthFinder (Yin et al. 2007); Knowledge-Based Trust (Dong et al. 2015); Dong et al. 2009 [M/A] | A claim repeated by one person should not outweigh one stated once by another. Any reliability estimate should live outside the model and be visible to the people it scores. |
| Bitemporal data: valid time vs transaction time | Snodgrass [M]; Zep (2501.13956) | In a group there are three times: when something was said, when it was true, and when each person learned it. |
| Event sourcing and CRDTs: an append-only log is the truth, every other view is a rebuildable projection, concurrent writes merge without loss | Shapiro et al. 2011 [M]; StateFuse (2607.05844) [A]; ActiveGraph (2605.21997) [A] | Keep the message log; treat facts, summaries and indexes as caches. Converging merges still leave semantic conflicts (CodeCRDT [A]). |
| Relationship-based access control checked at read time, with consistency tokens | Zanzibar (Pang et al. 2019) [A] | Check permissions against current membership at read time so revocation takes effect immediately. |
| Information-flow labels: derived data gets the most restrictive label of its inputs | Decentralized Label Model (Myers & Liskov 1997) [M]; FIDES (2505.23643); CaMeL (2503.18813) [A] | The read audience of a summary is the intersection of its sources' audiences, and its trust level is the lowest of its sources'. |
| Contextual integrity: whether a flow is appropriate depends on the norms of its context | Nissenbaum [M]; ConfAIde, PrivacyLens, CIMemories [A] | Being allowed to read something is not the same as it being appropriate to say it. That needs a separate decision. |

---

## 6. Gaps not closed by any work found in this search

1. **No system unites the two camps.** None both models human speakers and
   governs access.
2. **Visibility of derived memories is undefined for human audiences.** This
   covers summaries, profiles, embeddings and graph edges built from messages
   with different audiences. MAP-Graph defines it for agents, and when the
   intersection is empty it falls back to the writer.
3. **No rule between people turns disagreement into a recorded dispute.**
   Several systems keep dispute objects but decide supersession by authority
   weight or credibility, or by the user alone:
   - StateFuse; LatticeMind (2608.08236) [A]; MoM/P-Mem (2609.25054) [A];
   - MemTX's quarantine; COLLABMEM;
   - When Memory Becomes Authority (the user alone).

   TANGLE and MemConflict study conflicts for a single persona. Muses-Bench
   resolves conflicting instructions by fixed authority.
4. **Deletion does not cascade through derived artefacts** in any multi-user
   system, nor into audience-scoped summaries. MemoRepair and AfterDeleteBench
   show the mechanism for one user.
5. **Membership change is untested.** That covers joiners, leavers and
   retroactive revocation. SocialMemBench's "departed member" category rewards
   remembering, which is the opposite question.
6. **No who-knows-what index over human members exists**; existing work covers
   agent expertise only.
7. **No benchmark tests the combination** (section 3.3).
8. **Retrieval side channels are unstudied.** These are ranking statistics,
   contradiction flags and graph hops that depend on memories the viewer
   cannot see; see section 8 for a concrete case.

---

## 7. Proposed design: audience-scoped memory

*This section is a proposal, not a finding.*

### 7.1 The idea

Store **who said what, and where**, and derive everything else from it. Every
stored item is one of two kinds:

- a **message**: immutable; carries its speaker, channel and time;
- a **derivation**: an extracted claim, summary, profile or index entry; carries
  the exact list of messages it was computed from.

Who may read a message is decided by the *current* membership of its channel,
looked up at read time (decisions 1 and 2 in section 10). Three rules operate on
that one structure: audience (7.3), revision (7.4) and cascade (7.5). They are
deterministic code; no language model makes an access, supersession or deletion
decision.

An earlier draft called this "witness-set memory" and used the people present
when a message was posted as its audience. The decisions in section 10 replaced
that with current membership. Presence is now recorded only as "had access"
(7.7).

### 7.2 Data model

```
message(id, speaker, channel, said_at, content, reply_to)
channel_member(channel, member_id, joined_at, left_at)  -- left_at NULL = current
admin(member_id)                             -- workspace-wide
acknowledgement(message_id, member_id, kind) -- replied | reacted
claim(id, subject, attribute, value,         -- extracted, attributed
      holder,                                -- who asserted it (a speaker)
      valid_from, valid_to, hedged)
derivation(id, kind, content)                -- summary, profile, embedding, ...
source(derived_id, message_id)               -- complete list, transitively closed
supersedes(new_claim_id, old_claim_id)
dispute(id, subject, attribute) + dispute_member(dispute_id, claim_id)
delegation(subject, attribute, member_id,    -- one team fact handed to one
           granted_by, granted_at, revoked_at) -- member by an admin; see 7.4
```

The audience of a message is the current membership of its channel. A direct
message is a channel with fixed members. The audience of a derivation is the
intersection of its sources' audiences: the people who are currently in *every*
channel it draws from.

Audiences are never stored on the item; they are computed at read time
(a Zanzibar-style check). Membership changes therefore need no relabelling:

- **A joiner** can read the channel's whole history through the assistant,
  including summaries built only from that channel.
- **A leaver** loses access at the next read. That includes their own messages
  in that channel and every derivation with a source there.

Each derivation also carries an integrity level (human message, tool output,
agent inference), equal to the lowest of its sources'. Low-integrity items may
inform an answer but may not justify an action (FIDES and CaMeL).

Because derivations are built only from an explicit input set, their source
lists are exact by construction. MemLineage instead infers lineage from
whatever was in the model's context. The consolidator must never see more than
its declared sources.

### 7.3 Rule 1: Audience

**A memory may enter the context only if every current viewer is a current
member of every channel it came from.** The viewers are everyone who will see
the answer: all members of a group channel, or only the asker in a direct
message. If membership cannot be determined, assume everyone, so only public
items pass (fail closed, as in Authorization Before Context).

Strict intersection makes cross-channel summaries unreadable by most groups.
The design therefore consolidates **per audience, after filtering**. For a
viewer group V, take the channels that every member of V is in, and build the
summary only from messages in those channels. That summary is always readable
by V.

Summaries are cached by their set of source channels, not by a set of people.
A membership change then never invalidates a summary; it only changes who
passes the check. Summaries can be built lazily, for the channel sets that are
actually queried.

Worked example (team Alice, Bob, Carol):

| Item | Where | Readable by |
| --- | --- | --- |
| m1, Alice: "Launch is planned for 3 March." | #design | current #design members: A, B, C |
| m2, Bob: "Between us, I think launch slips to April." | Alice–Bob DM | A, B |
| Summary of m1 + m2 | sources in #design and the DM | people in both: A, B |

- **Carol asks in #design** whether anyone has raised concerns about the date.
  The viewers are the #design members, so m2 and the cross-channel summary are
  excluded. The answer comes from m1 and the #design-only summary, and Bob's
  private doubt is not revealed.
- **Alice asks the agent directly.** The viewer is Alice, so both messages and
  the cross-channel summary are admissible.
- **Dana joins #design later.** She can now get m1 and the #design-only summary,
  although both predate her. She cannot get m2.
- **Bob leaves #design.** He loses m1 and the cross-channel summary, because he
  is no longer in every channel it draws from. He keeps m2.

The rule does not stop Alice from telling Carol herself. It stops the *agent*
from carrying information between contexts, which is the contextual-integrity
principle.

**Alternative support widens the audience.** An extracted *claim* is often
supported by several messages independently. Its audience can then be the
union, over each independent support set, of that set's intersection; this is
how provenance semirings combine alternative derivations. Example: Alice said
the date in #design and again in a DM to Bob. The claim is readable by every
#design member, because the #design message alone supports it.

This eases the starvation problem for claims. It does not apply to a
*summary*, whose content combines all its sources, so a summary always takes
the plain intersection. The same idea gives the deletion rule for claims: a
claim survives a deletion while some remaining message still supports it, as in
Zep.

### 7.4 Rule 2: Revision

Every extracted claim has a **holder**, the person who asserted it. A new claim
about the same subject and attribute **supersedes** the old one in three cases:

- it has the same holder;
- the fact is personal and the holder is the person it is about;
- the fact is a team fact and the holder is an admin.

Otherwise it **opens a dispute**, which is kept and shown as a table.

Who owns what (decision 3 in section 10):

- **Personal facts** (preferences, availability, own plans) belong to the person
  they are about. Carol's own statement of her preference supersedes Bob's
  report of it, and Bob's report never supersedes Carol's statement. An
  admin's statement about Carol's personal fact does not supersede hers either.
  This is the model's choice, left to it by the researcher (section 10).
- **Team facts** (dates, decisions, assignments, policies) belong to admins.
  Admins are set once for the whole workspace. An admin's statement becomes
  the current value and closes any open dispute.
- **A delegate** holds one team fact by an admin's grant ("Alice owns the review
  schedule"). For that fact the delegate's statements supersede like an
  admin's, and an admin still outranks the delegate.
- **A later conflicting statement by a non-admin** does not replace an admin's
  value. It is recorded and shown as differing from it.
- **Two admins who disagree** leave the dispute open until one of them restates.
  MemTX treats equal authority the same way, by quarantine.

Example (Dana is an admin; Alice and Bob are not):

1. Alice says "Design review is Thursday."
2. Bob, who misread the invitation, says "Friday."
3. Newest-wins would answer Friday. Rule 2 opens a dispute and answers with a
   table:

   | Who | Says | When | Where | Status |
   | --- | --- | --- | --- | --- |
   | Alice | Thursday | Mon 10:02 | #design | disputed |
   | Bob | Friday | Tue 09:15 | #design | disputed |

4. Dana says "The review is Thursday." The current value is now Thursday
   (Dana, admin) and the dispute closes. The table stays available as history,
   with Dana's row marked *current*, Alice's *agrees* and Bob's *overruled*.

The table lists only the rows the viewers may read (rule 1).

**Delegation.** Without it, every change to a team fact waits for an admin,
even when the organiser moves their own meeting, and dispute tables grow. The
delegation rules:

- **Scope.** A grant covers one team fact (a subject and attribute) for one
  member, never a whole channel or topic.
- **Record.** A grant or revocation is a message by the admin, so it is
  attributed, timestamped and readable under rule 1 like anything else. The
  `delegation` table is derived from those messages.
- **Precedence.** Admin over delegate over everyone else. If an admin and the
  delegate disagree, the admin's value is current and the delegate's row shows
  as *overruled*.
- **Lapse.** A grant ends when an admin revokes it or the delegate leaves the
  workspace. Values the delegate set stay current until someone with authority
  restates them.
- **No chains.** A delegate cannot delegate further.

Superseded claims stay stored. The default read view shows current values and
open disputes. History is shown only when the question asks about the past.
This keeps past-value questions answerable, which Memvane found archiving
destroys, without putting stale values in front of the answerer (Ground Truth
First).

**Caveat.** Rule 2 decides what to do *after* two claims have been recognised
as being about the same slot. Recognising that is update detection, the open
problem Memvane's update-detection experiment is working on. The rule does not
solve detection. It also needs each claim sorted into personal or team, which
is a second classification step.

### 7.5 Rule 3: Cascade

Deleting a message, or its speaker retracting it ("scratch that"), marks every
derivation that lists it as a source as stale. Stale derivations are rebuilt
without it on next use. Embeddings and index entries of the message are
removed.

For physical erasure, give each message its own encryption key and destroy the
key (Ghost Vectors). This also reconciles erasure with a tamper-evident log: the
log keeps a signed tombstone, and the content becomes unrecoverable.

This is justification-based truth maintenance (Doyle) applied to LLM memory, as
MemoRepair did for one user. Here the same source lists that drive the cascade
also drive the audience rule.

### 7.6 Read path

1. Resolve the viewers from channel membership; if unknown, fail closed.
2. Compute the **visible set**: items whose audience contains every viewer. This
   is a SQL filter, applied before any scoring.
3. Retrieve **only within the visible set**. Ranking statistics (BM25
   document frequencies, average length, coverage) are computed over the
   visible set, and graph hops and contradiction flags stay inside it.
   Otherwise scores reveal hidden content (section 8).
4. Collapse supersession chains to current values; attach each open dispute as
   a table of its visible rows.
5. Add history if the question is about the past.
6. Answer with attributions: who said it, and when.

### 7.7 Who knows what

Being in a channel does not count as knowing what was said there (decision 4 in
section 10). A member counts as knowing a claim in two cases only:

- **stated**: they hold a claim with that value themselves;
- **acknowledged**: they replied or reacted to a message that carries it.

Everyone else in the channel is reported as *had access*, never as *knows*.
Example: "Was Bob told that the deadline moved?" gets "Bob is in #design, where
Alice posted it on Monday, but he has not replied or reacted to it."

This gives a transactive-memory directory over human members. Replies and
reactions come from message metadata, so it needs no language model.

Limits:

- **It under-counts.** People who read silently are not counted. That is the
  safe direction for "should I tell Bob?", and the wrong one for "who could
  have known?", which the *had access* list answers.
- **A reaction is not understanding.** A thumbs-up on a long message
  acknowledges the message, not each claim in it.
- **Who-knows answers obey rule 1.** They draw only on channels the viewers may
  read.

### 7.8 Relation to prior work

A separate search tried to show the design is *not* new. Its verdicts:

| Part of the design | Closest prior work | What differs here | Verdict |
| --- | --- | --- | --- |
| Derived audience = intersection of sources' audiences | Decentralized label model (Myers & Liskov 1997); security semiring (Foster, Green & Tannen, PODS 2008); MAP-Graph (2608.10509), for agents | The scopes are people's channel memberships, checked at read time; an empty intersection leads to per-audience consolidation, not a fallback to the writer | Already done in principle |
| All current viewers must be in the audience | Authorization Before Context (2608.17148); participant-aware access control (2509.14608) | Nothing; adopted as is | Already done |
| One view per audience | Polyinstantiation in multilevel secure databases (Jajodia & Sandhu, 1990–93) | Applied to LLM-written summaries | Old idea, new setting |
| Only the subject may update facts about themselves; reports by others are attributed | When Memory Becomes Authority (2608.01679), single user | Several human users; holder and subject kept apart | Partly done |
| Conflicting values kept as disputes | StateFuse; LatticeMind; MoM/P-Mem; MemTX's quarantine; COLLABMEM; belief databases (Gatterbauer et al., PVLDB 2009) | Supersession decided by authorship first (same speaker, or the person the fact is about) and admin role second. The admin part resembles the authority ranking in MemTX and COLLABMEM | Partly done |
| Deletion cascades through derivations | MemoRepair; AfterDeleteBench; MemTX rollback; truth maintenance (Doyle 1979); view maintenance (DRed, 1993) | The cascade reaches audience-scoped summaries | Already done for one user |
| Who knows what | COLLABMEM's role visibility; transactive memory | Over people, counting only statement or acknowledgement; presence is reported as access | Partly done |
| One provenance structure driving access, revision and deletion | Orchestra update exchange (Green et al., VLDB 2007) and provenance semirings in databases | Human channel audiences and LLM-written summaries | Not found in LLM memory |

A fair description is therefore:

> The decentralized label model, polyinstantiation, belief databases and truth
> maintenance, combined and applied to LLM memory shared by people.

What is new is narrower:

- acknowledgement, not presence, as the test for who knows what;
- per-audience consolidation in place of withholding;
- the authorship rule between people;
- the evaluation (section 9), especially the retrieval side channel (H2) and
  the starvation trade-off (H6), which none of the papers found measures.

Two risks follow from the prior work:

- **The cascade needs complete provenance.** MemoRepair assumes it; the design
  gets it only by giving the consolidator nothing beyond its declared sources.
- **Per-audience views can signal what is hidden.** The polyinstantiation
  literature warns of this; for example, the existence or count of a dispute
  can reveal a hidden claim. An LLM summary can also leak by implication, which
  one write-up calls "summary laundering."

### 7.9 What the design does not solve

- **Truth.** A legitimate member can still assert something false within their
  own audience. Audiences limit who reads, not what is true. Write-time
  integrity defences (MAPLE-Guard, origin-bound authority) are complementary.
- **Inference from aggregates.** Repeated answers can reveal attributes that
  were never stated (DP-MemView, 2608.03130 [A]).
- **Appropriateness.** Permission to recall is not appropriateness to say. A
  contextual-integrity check is still needed on top.
- **Slot detection and slot ownership** both need information the rule takes as
  given.
- **Cost.** Per-audience summaries multiply consolidation work by the number of
  distinct audiences; lazy building bounds it but does not remove it.

---

## 8. What this means for Memvane

Until 2026-11-15 Memvane had one store and no notion of principals, and
`search` built its candidate set from every active memory
([store.py:882](../src/dynmemory/store.py#L882)). A search without `viewers`
still works that way. If per-user visibility were added by filtering results
after ranking, which is the obvious first step, four paths would still depend
on memories the viewer cannot see:

1. **Ranking statistics.** Document frequencies, inverse document frequency,
   average length and `coverage` are computed over every active memory
   ([store.py:1037-1069](../src/dynmemory/store.py#L1037-L1069)). A viewer's
   scores, and the silence gate built on `coverage`, would shift when hidden
   memories contain the query's words.
2. **Graph hop.** A neighbour is admitted if it is any active memory
   ([store.py:924](../src/dynmemory/store.py#L924)).
3. **Contradiction flags.** Disputants are looked up among all active memories,
   deliberately ignoring even the `--type` filter
   ([store.py:959-969](../src/dynmemory/store.py#L959-L969)). This would expose
   the IDs of hidden memories.
4. **Topic carry-over** resolves links through every version of a fact
   ([store.py:1168-1206](../src/dynmemory/store.py#L1168-L1206)).

The fix is the read path in 7.6. Compute the visible ID set first, pass the
filtered mapping to every scoring step, and keep a link only if both of its
ends are visible. It is now built; see below.

**First test of path 1 (H2), run 2026-11-15.** Script:
[audience_side_channel.py](../experiments/audience_side_channel.py); results:
[audience_side_channel.md](../experiments/results/audience_side_channel.md).

- **Setup:** synthetic pseudo-word memories, lexical search only, 200 trials.
  Each trial has 200 memories the viewer can see and 50 in a channel they
  cannot.
- **Probe:** the viewer searches "anchor w" and "anchor z", where the anchor is
  a word from their own memories, w is the word they are curious about and z
  is a nonsense word. D is the difference in the best `coverage` they are
  shown. D is zero when w is in no memory, and needs no calibration.
- **Filter after ranking:** D was above zero in 200 of 200 trials when even one
  hidden memory contained w (mean D 0.046 for one memory, 0.118 for five,
  0.222 for twenty). The viewer can tell that a word occurs in a channel they
  are not in, and roughly how often.
- **Filter before ranking (emulated):** D was zero in 200 of 200 trials. This
  arm searches a store that holds only the visible memories, so this is a
  sanity check, not a finding.
- **The fix as built** (`search(..., viewers=...)`, added the same day; the
  hidden memories are in the database it reads): D was zero in 200 of 200
  trials, and its probe values equalled the emulated arm's in all 1,200
  trial-and-world comparisons.
- **Positive control:** with w in the viewer's own memories, the probe fired
  in 200 of 200 trials in all three arms.
- **Negative control:** with w in no memory, D was zero in 200 of 200 trials
  in all three arms, with or without hidden memories.
- **A second, coarser leak:** hidden memories that match nothing still shifted
  the viewer's raw lexical scores in 200 of 200 trials (mean +1.2%), because
  they change the memory count and average length. This is why the negative
  control is stated on D and not on raw scores; the wording was corrected from
  reading the code, before any run.
- **Limits:** the data were written alongside the code, so this shows that the
  mechanism exists, not how often it matters in real use. The experiment
  covers path 1 with lexical search only.

**The read path as built, 2026-11-15.** `store.py` now has channels, channel
membership and a `viewers` argument on `search`, `get_memory`, `history` and
`list_memories`.

**Guard added, 2026-11-18.** `for_viewers` returns a reader whose four read
methods always pass the bound audience. `require_viewers=True` rejects omitted
audiences on those methods. Missing embeddings are now generated only for
memories visible to the search audience; a hidden memory cannot trigger an
embedder failure during someone else's search. The CLI remains a local
administrative interface without audience filtering.

**Speaker revision added, 2026-11-18.** A keyed channel memory now needs a
speaker. The same speaker may correct their earlier claim; another speaker's
different text stays active with a `contradicts` link. Search labels active
claims `current`, `disputed`, `agrees` or `overruled`. A trusted store caller
registers workspace admins; their write can resolve non-admin disagreement.
Two admins' first conflicting writes remain disputed until one restates.
Admin authority is fixed on each claim at write time. These are exact text
comparisons, not semantic conflict detection. Write authentication and a
standalone dispute table remain unbuilt.

**Guarded writes and scoped ownership, 2026-11-18.** A caller that has already
authenticated a member can pass only `for_writer(member_id)` to user-facing
code. It binds the speaker and checks channel membership within the write
transaction. `require_writer=True` rejects direct `add_memory` on that store
instance. Personal claims explicitly carry `fact_subject`; their subject's
own write outranks third-party and admin reports. A channel key cannot switch
between team and personal scope or between personal subjects. Team delegates
are granted one channel key by a workspace admin, with admin > delegate >
ordinary precedence. Grants and revocations are timestamped in a database
table; a channel departure revokes its grants. A claim's authority stays as
asserted at write time. Classification of real-world text into a key, scope,
and subject remains the caller's job. The store interface is a capability
boundary for application code, not an authentication or process-isolation
mechanism; the raw store and SQLite file need trusted ownership. Delegation
events are recorded as database rows, not immutable source messages as the
full proposal describes.

**Source lineage and retraction, 2026-11-18.** Callers can record immutable
attributed source messages and list their IDs when writing a derived memory.
The source list is stored as `memory_sources`. Audience checks intersect the
memory's channel with every source message's channel before retrieval,
including link traversal and embedding. A speaker or workspace admin can
retract a source message: its content becomes a tombstone, every dependent
memory is marked stale and archived, and its embeddings are removed. Stale
memories are withheld from all ordinary read paths and cannot be restored.
Trusted code sees a rebuild queue and may write a new derivation from remaining
valid sources. This slice does not generate replacement prose automatically,
infer source lists for existing memories, index raw messages in text search, or
guarantee physical erasure from SQLite backups and free pages. Derivations of
derivations must explicitly carry forward the complete raw-message source list
and parent audience constraints; source text is not checked against derived
content.

**Read-time claim resolution, 2026-11-18.** A review of the slices above found
two failures, both caused by resolving claims when they were written, against
every claim in the channel:

- **Narrower sources.** A claim whose source message was in a narrower channel
  (a direct message) replaced or resolved claims that the rest of the channel
  could see. Those viewers saw their claim disappear, and `get_memory` showed
  them the hidden claim's ID in `superseded_by`. Source lineage had reopened
  the fifth path that per-channel keys closed (see **Keys** below).
- **Retraction.** Retracting the message behind a correction or an admin's
  resolution left the key with no current value, because the claims it had
  replaced stayed replaced.

Channel memories are now never archived or superseded on write. Each read
replays rule 2, in write order, over the claims it may use: visible to its
viewers and not stale. It applies a stored `supersedes` link between channel
memories only if the link's source is among them. Claim statuses, `contradicts`
links between claims of one key, `superseded_by`, `history` and link carry-over
all come from that replay. A hidden or retracted write therefore changes
nothing a viewer reads, and whatever a retracted claim replaced stands again.
Memories without a channel keep write-time replacement, so the single-user
results are unaffected. Schema version 6 undoes stored supersessions of
channel memories.

Side effects:

- `relate(..., supersedes)` can no longer target a keyed channel claim.
- Pruning skips replaced claims, as it skipped archived ones before.
- A pruned claim still takes part in the replay.

The noninterference experiment's model now predicts the new refusal.

Tests are in [test_claim_resolution.py](../tests/test_claim_resolution.py):

- 8 regression tests, which failed before the change;
- 4 controls, which passed before it;
- 4 tests of the new behaviour.

Twelve deliberate faults in the new code were each caught by these tests.
The read side is covered by the experiments below. Two write-side signals
remained and are closed in the 2026-11-19 entry at the end of this list.

**Noninterference experiment, version 2, 2026-11-18.** Script:
[audience_noninterference.py](../experiments/audience_noninterference.py).
Run 2 results: [current store](../experiments/results/audience_noninterference.md)
and [store before read-time resolution](../experiments/results/audience_noninterference_before_fix.md).
Run 1 results are in
[noninterference_v2_run1_superseded](../experiments/results/noninterference_v2_run1_superseded/).

- **Additions over version 1:**
  - source messages;
  - hidden "narrow" writes, in a channel the viewers see but drawn from a
    channel they cannot see;
  - one admin, who writes only hidden memories;
  - narrow `supersedes` links onto visible memories;
  - retractions, hidden and visible;
  - a seventh rule: after the viewers' retractions, reads must equal those of
    a world N where the retracted memories were never written.

  The decision rule, a prediction and a validation criterion were fixed
  before any run. Version 1 was never run for results.
- **Validation (positive control):** the same script was run against the
  store before read-time resolution (commit c7bb390, 500 pairs). Rule 5 or 6
  failed in 386 pairs (77%) and rule 7 in 403 (81%), above the 25% fixed in
  advance. So the extension detects both faults found in review.
- **Run 1, current store (500 pairs):**
  - Controls held in every pair, as did soundness: no read with viewers ever
    showed or linked to a hidden memory.
  - Rules 4, 5 and 7 failed: rule 4 in 191 pairs, rules 5 and 7 in the same
    70 pairs, all within the 191.
  - Every failing pair had a visible `relate` whose outcome depended on hidden
    memories (383 operations). In each, a hidden narrow write had replaced one
    end in the administrative view. `relate` checks that both ends are active
    in that view, so it refused (or refused for a different reason), and in
    70 pairs the missing link reached the viewers' search results.
  - The 309 pairs without such an operation held all seven rules.
  - This was the prediction recorded before running. The administrative
    model predicted every hidden outcome (0 of 38,699 differed).
- **Change after run 1:** the researcher chose that `relate` check only its
  two memories (exist, not archived, not retracted), not whether a channel
  memory is replaced, which reads decide. Two regression tests failed before
  the change and pass after it, and a control passes in both. The
  experiment's model follows the new rule. Its scripts, reads, visible
  predictions and decision rule are unchanged.
- **Run 2 (500 pairs, the same script for both stores):**
  - Current store: all seven rules held in every pair ("the built read path
    passes"), as predicted before the run. No visible outcome depended on
    hidden memories. The model predicted every hidden outcome.
  - This held although hidden activity replaced a visible memory in the
    administrative view in 377 pairs (75%), and a visible retraction removed
    a memory that had replaced another in 306 (61%).
  - Store before read-time resolution: rules 4 to 7 failed at the same rates
    as in run 1. Rule 5 or 6 failed in 77% of pairs and rule 7 in 81%, so the
    validation still holds.
- **Limits:** synthetic scripts written alongside the code, so a pass shows
  the mechanism holds on these scripts, not on every database. Not exercised:
  delegation, personal facts, pruning, `restore` and bound writers. The only
  admin writes only hidden memories.

**Link-path and utility experiments, first runs, 2026-11-19.** Both were
written on 2026-11-15 with their decision rules fixed, and had never been run.
They were run unchanged on the current store (schema version 6), 200 trials
each.

- **Paths 2 to 4** ([audience_link_paths.py](../experiments/audience_link_paths.py),
  [results](../experiments/results/audience_link_paths.md)):
  - Graph hop, contradiction flags and topic carry-over each gave the same
    pattern. The probe fired in the positive control in every trial and arm,
    the negative controls never fired, and filtering after ranking leaked in
    200 of 200 `present` trials. (For contradiction flags only the naive
    version leaked; the careful one trims links and did not, as predicted.)
  - The built path never leaked, and its results equalled those of filtering
    before ranking in all 2,800 trial-world pairs. Verdict for each path:
    "the baseline leaks; the built path closes it".
  - One wording in the design is out of date. It says `copy` mode copies a
    link to the new version at the rewrite. Since read-time resolution,
    channel memories carry links at read time instead. The `copy`
    predictions, which are not part of the verdict, held either way.
- **H1, zero leakage at no utility cost**
  ([audience_utility.py](../experiments/audience_utility.py),
  [results](../experiments/results/audience_utility.md)):
  - Controls held. The pooled store leaked 9,010 of 9,010 canary items.
    Owners found their own canaries every time, and every canary in the shared
    channel reached every member.
  - The built path and both filter-after arms leaked nothing.
  - The built path's utility equalled filtering after the whole ranking in
    all 200 trials (mean difference in reciprocal rank 0.0000, CI 0.0000 to
    0.0000). Verdict: "H1 supported".
  - Caveat: all three filtered arms reached a mean reciprocal rank of 1.0000,
    so the task sits at its ceiling. The test shows that the built path drops
    and misorders no visible memory. It has little power to detect a subtler
    ranking cost, and it used lexical search only, with no links or keys.

**Write outcomes of bound writers, 2026-11-19.** Review found that two write
checks read every claim of a key: the refusal of a change of scope or personal
subject, and the refusal of a lapsed authority revising its own delegated
claim. A hidden claim therefore changed whether a writer's own write was
accepted, which tells the writer the claim exists.

- **Fix:** for a bound writer both checks read only the claims that writer can
  see. Trusted store calls keep the full view. Two regression tests failed
  before the change and pass after it, with two controls that passed in both.
- **Experiment** ([audience_write_outcomes.py](../experiments/audience_write_outcomes.py),
  [results](../experiments/results/audience_write_outcomes.md)), written with
  its decision rule and validation criterion fixed before any run. Bound
  writers alice and carol cannot see claims drawn from a direct message that
  bob and dana share; world H adds such claims, W omits them, and V (positive
  control) makes them visible. 500 pairs, with delegation, personal facts and
  an admin.
- **Current store:** every visible write outcome and every read of H equalled
  W in all 500 pairs ("the write path passes"). Controls held: H rebuilt equal
  in 500, V differed from W in 485 (97%).
- **Validation:** the same script on the store before the fix (commit
  5809ca9) failed rule 3 in 160 of 500 pairs (32%, above the 25% fixed in
  advance), always through a visible write whose outcome differed. Both
  reports state schema version 6, since the schema did not change; the
  commit identifies the store.
- **Coverage:** the lapsed-authority path appeared in only 5 of those 160
  failing pairs, against 159 for the scope path. The experiment therefore
  validates the scope fix well and the lapsed-authority fix weakly; its
  regression test is the stronger evidence for that one.
- **Not exercised:** retractions, pruning, `restore`, other admins, and
  delegation by anyone but dana to anyone but alice.

**Retraction canaries (H5), 2026-11-19.** Review had found that retraction
tombstoned the message and hid the derived memories from every read, but their
rows kept the retracted text, and `stale_memories()` returned it.

- **Decision (the researcher's):** clear the text. On retraction every memory
  listing the message as a source now has its text replaced by `[retracted]`.
  The row keeps its ID, key, channel and speaker, so the rebuild queue still
  says what to rebuild; trusted code rebuilds from the sources that remain.
- **Tests first:** two regression tests failed before the change and pass after
  it; two controls (unrelated memories and messages keep their text; a memory
  with two sources is cleared while the other message stays readable) passed in
  both.
- **Experiment** ([audience_retraction_canary.py](../experiments/audience_retraction_canary.py),
  [results](../experiments/results/audience_retraction_canary.md)), with its
  decision rule and validation criterion fixed before any run. Each trial plants a
  unique canary in 4 to 8 messages, derives 3 to 8 memories that repeat the
  canaries of their sources, retracts 1 or 2 messages, and scans every value
  of every table. 500 trials.
- **Current store:** all four rules held in 500 of 500 trials ("retraction
  erases"): the determinism and probe controls, specificity (nothing changed
  except memories derived from retracted messages) and erasure (no retracted
  canary in any stored value). 1,320 memories were derived from retracted
  messages.
- **Validation:** the same script on the store before the change (commit
  2114e4e) failed erasure in 500 of 500 trials (criterion 95%).
- **Physical erasure is not achieved.** The raw bytes of the database file
  still contained a retracted canary in 495 of 500 trials on the current store
  (500 of 500 before the change). SQLite keeps old bytes in free space until
  they are overwritten. This is reported, not part of the verdict, and the
  experiment does not claim to erase it. Options if it matters: SQLite's
  `secure_delete` setting, `VACUUM` after retraction, or encrypting message
  text (the per-message keys of section 7.5). None is built.
- **Limits:** synthetic text; a derived memory is tied to its sources only
  through the list the caller supplies, so text copied into a memory that does
  not list the message is neither found nor tested.

- **Visibility:** a memory is visible if every viewer is a current member of
  its channel. A memory without a channel is visible to no viewer.
- **All four paths:** the visible set is computed first and passed to every
  scoring step. A link counts only if both ends are visible, and so does
  every version it is followed through.
- **Keys:** a key now belongs to its channel, and a memory can only supersede
  one in its own channel. Otherwise a write in a hidden channel could replace
  a memory the viewer can see, which is a fifth path not listed above. Once
  memories could cite source messages from other channels, per-channel keys no
  longer closed it on their own; read-time claim resolution (above) does.
- **Tests:** paths 2 to 4, the fused lexical and vector ranking, joiners and
  leavers are covered by unit tests in
  [test_audience.py](../tests/test_audience.py), each with a positive control
  (a member of both channels does see the link). Since 2026-11-19 paths 2
  to 4 are also covered by an experiment with many trials (see the link-path
  entry above). Sixteen deliberate faults put into the store, one at a time,
  were each caught by these tests.
- **Not built:** write authentication, a standalone dispute table, automatic
  regeneration of stale derivations, physical erasure and who-knows-what.
  A read without `viewers` uses every
  memory on an unguarded store; guarded stores reject it.

How the earlier Memvane findings fit:

- **Oracle update handling** (+0.158 MRR) is the largest single gain measured.
  Rule 2 decides which detected updates count as updates.
- **Archiving old versions** cost 0.53 → 0.24 on past-value questions. Keep
  versions searchable and render history on demand.
- **Link carry-over** showed no benefit beyond random links, which supports not
  investing in graph structure first.
- **The silence signals** would need to be computed per viewer set.

Smallest schema change that supports rules 1 and 2 (speaker, channel, member
and admin records plus keyed contradiction links are now built):

- a `speaker` and `channel` column on `memories`;
- a `channel_members(channel, member_id, joined_at, left_at)` table and an
  admin table;
- an `acknowledgements(memory_id, member_id, kind)` table;
- a key rewrite by a different speaker records a `contradicts` link instead of
  a supersession, unless that speaker is an admin or the person a personal key
  is about.

Rule 3's source lists are built as `memory_sources` (2026-11-18, above). They
are supplied by the caller; Memvane still has no LLM consolidation of its own.

---

## 9. Evaluation plan

Each hypothesis has a negative control (the effect should vanish) and, where
possible, a positive control (the probe can detect the effect). Fix the
decision rule before each run, as the existing experiments do.

| # | Hypothesis | Arms | Positive control | Negative control |
| --- | --- | --- | --- | --- |
| H1 | Filtering before retrieval gives zero leakage at no utility cost compared with filtering after | pooled; filter after ranking; filter before; filter before + per-audience summaries | Private canary strings: the pooled arm must leak them, and their owner must find them | A canary posted in a channel everyone is in must be found by everyone (no over-blocking) |
| H2 | Filtering after ranking leaks through scores; filtering before does not | add hidden memories that contain or omit the query words, then compare a viewer's scores and coverage | Hidden memories containing the query words change post-filter scores | Hidden memories without the target word leave the probe statistic at zero in both arms. (Raw scores do shift with any hidden memory, so the control is on the statistic; see section 8) |
| H3 | Keeping speaker and thread raises evidence recall on attribution questions | speaker and thread kept vs stripped | n/a | Speaker labels permuted: the gain should disappear |
| H4 | The authorship rule cuts false supersessions on disagreement turns without lowering update accuracy | newest-wins vs rule 2 | Oracle slot labels reproduce the earlier oracle gain | Single-speaker threads, where the two rules must agree |
| H5 | With source lists, deleted content does not survive in derived memories | cascade vs none | Deleted canary appears in derived memories without the cascade | Delete a source with no dependents: no other change |
| H6 | Per-audience consolidation answers group questions better than strict intersection | intersection only vs per-audience | n/a | A workspace with one channel, where all audiences are equal and the arms must tie |
| H7 | Membership changes take effect at the next read | before vs after a join or a leave | A joiner finds a canary posted before they joined; a leaver stops finding one, including through a cached summary | A member whose membership did not change gets identical results |
| H8 | Who-knows answers count only statement or acknowledgement | acknowledgement rule vs presence rule | A member who replied to the message is reported as knowing | A member who was only in the channel is reported as "had access", never as knowing |
| H9 | The admin rule resolves team-fact disputes without overwriting non-admin disagreement | newest-wins vs rule 2 with admins | An admin's statement becomes the current value and closes the dispute | A non-admin's later conflicting statement leaves the admin's value current |
| H10 | Delegation cuts open disputes on team facts without letting delegates act beyond their grant | rule 2 with vs without delegation | A delegate's change to their own fact becomes current with no admin step | The same delegate's statement about a different team fact opens a dispute, as for any non-admin |

Stages, cheapest first:

- **Stage A: scripted synthetic workspace, no LLM.**
  - **Build:** sample a fact table (subject, attribute, value, speaker,
    channel, channel membership over time, valid time, admin), render messages
    from templates and
    plant canaries. Gold answers are correct by construction.
  - **Tests:** H1, H2, H5 and H7–H10 as mechanism checks. As with
    `eval/synthetic_set.json`, results from data written alongside the code
    show that a mechanism works, not that it helps.
- **Stage B: LongMemEval as a shared workspace.**
  - **Build:** assign K question instances to K named users and interleave
    their sessions.
  - **Items:**
    - each original question asked by its owner (should answer);
    - the same question asked by another user in a direct message (should
      refuse);
    - a cloned evidence session with a changed value given to a second user
      (cross-user dispute).
  - **Reuses** `dynmemory.longmemeval`.
  - **Positive control:** K = 1 must reproduce the current results.
  - **Note:** the existing labels still await review.
- **Stage C: multi-party benchmarks, retrieval only.**
  - **Data:** GroupMemBench, SocialMemBench and EverMemBench all provide gold
    evidence messages, so evidence recall can be scored without an LLM judge.
  - **Tests:** H3 and H4.
  - **Check licences first.** SocialMemBench's dataset page says CC-BY-4.0
    [A]; the others were not checked. All three are synthetic, with no real
    people's data.
- **Stage D: needs LLM consolidation.** H5 on real derivations, and H6.

Reporting:

- Report utility, leakage, over-refusal and forgetting failures separately, not
  only as a product like GateMem's score.
- Always include BM25.
- Report over several seeds.
- Have a person review any labels an AI model writes before relying on numbers.

---

## 10. Decisions

The researcher decided seven design questions on 2026-11-14:

| Question | Decision | Effect on the design |
| --- | --- | --- |
| Can people who join later see older history through the assistant? | Yes | The audience of a message is the current membership of its channel, not the people present when it was posted (7.2, 7.3) |
| Do people who leave lose access? | Yes | Access is checked at read time, so a leaver loses the channel's messages and every derivation with a source there (7.2) |
| Who owns team facts? | Admin | An admin's statement supersedes and closes disputes; other people's conflicting statements open or join a dispute (7.4) |
| Are admins set per workspace or per channel? | Per workspace | One admin list for the whole workspace (7.2) |
| May an admin delegate a fact? | Yes | An admin can hand one team fact to one member; precedence is admin, then delegate, then everyone else (7.4) |
| Does being present count as knowing something? | No | Only stating or acknowledging counts; presence is reported as "had access" (7.7) |
| How are disagreements shown? | In a table | One row per claim: who, says, when, where, status (7.4) |

On 2026-11-15 the researcher left three further questions to the model's
preference. These are therefore the model's choices, not the researcher's own,
and are the first to revisit if they do not fit the team:

| Question | Choice | Reason |
| --- | --- | --- |
| Who owns personal facts? | The person the fact is about, even against an admin | An admin does not know a member's preferences or availability better than the member does (7.4) |
| Does a leaver keep access to their own messages in the channel they left? | No | One rule with no exceptions is easier to check and fails closed. A leaver's message is often a reply that quotes or reveals what others said (7.2) |
| Where to start? | Stage A, beginning with H2 on Memvane | It is cheap and needs no labels. The first run is reported in section 8 |

Two of the researcher's decisions move the design closer to existing work (7.8):

- Membership checked at read time is standard relationship-based access
  control (Zanzibar).
- Admin authority resembles the authority ranking in MemTX and COLLABMEM.

Still open:

1. **Who appoints admins.** This is treated as workspace configuration, outside
   the memory.

The naming issue was resolved on 2026-11-19 by selecting **Memvane** for this
project, formerly Memora. The published Memora system and benchmark keep their
original names; see the [name review](public_name_review.md).

---

## 11. Discrepancies noticed while reading

- **GroupMemBench**
  - The abstract's "knowledge update at 27.1%" is HippoRAG's score; the best
    overall system, Hindsight, scores 17.76 on that category.
  - The reported averages do not equal the column means.
- **EverMemBench**
  - The abstract's "26% with oracle evidence" for multi-hop matches Gemini's
    *full-context* multi-hop score (26.51); oracle multi-hop reads 88–98.
  - The abstract says "one million tokens" per project; the table gives about
    845K.
- **AIM**
  - The headline numbers combine two models: visibility 96.0 is GPT-5.4,
    while the strict 58.8 and state-aware 70.5 are GPT-5.4-mini.
  - The text says only the creator may update or delete a memory, but the
    algorithm allows any user to update public memories and has no ownership
    check on delete.
- **Collaborative Memory**
  - The abstract claims "provable adherence", but no proof was found in the
    text read.
  - The access formula has no owner test for private fragments.
  - No leakage test is reported.
- **SpeakerMem-R1** falls below BM25 on GroupMemBench with a different reader
  model (42.7 vs 46.2).
- **MemClaw** reports no baselines; it measures one production service.
- **LLMA-Mem** says it improves "in all model-environment settings", but its
  own section 4.2 contains a drop.
- **TANGLE**: the judge correlation is 0.838 in the text and 0.853 in an
  automated summary of the paper.
- **MINJA**: sources give different attack rates (98.2/76.8 vs 95/70).

---

## 12. References

All arXiv IDs resolve at `https://arxiv.org/abs/<ID>`.

**Multi-user and shared memory systems**

- Collaborative Memory: Multi-User Memory Sharing in LLM Agents with Dynamic Access Control. Rezazadeh et al., 2025. 2505.18279 [F]
- AIM: A Privacy-Aware Interoperable Memory Framework for Multi-Agent Multi-User LLM Systems. Johnson et al., 2026. 2609.12320 [F]
- Governed Shared Memory for Multi-Agent LLM Systems (MemClaw). Margalit et al., 2026. 2606.24535 [F]
- Kernel-Managed Shared Memory for System-Wide Personalization. Lum & Zhang, 2026. 2609.10144 [F]
- Authorization Before Context: A Model-Neutral Audience Boundary Against Cross-Audience Memory Leakage. Liu, 2026. 2608.17148 [F]
- MemTX: Transactional Belief Commit for Stateful Agent Memory. Li et al., 2026. 2607.23929 [F]
- SpeakerMem-R1: Speaker-Centered Dual-Track Memory for Multi-Party Dialogue. Zheng et al., 2026. 2609.26780 [F]
- Propose, Verify, Commit: Evidence-Grounded Memory for Long-Horizon Multi-Actor Conversations (EGMemory). Lu et al., 2026. 2609.23465 [F]
- Multi-Agent Transactive Memory. Kim et al., 2026. 2606.19911 [F]
- Scaling Teams or Scaling Time? (LLMA-Mem). Wu et al., 2026. 2604.03295 [F]
- StateFuse. Volkov et al., 2026. 2607.05844 [A]
- MemoRepair. Zhao et al., 2026. 2605.07242 [A]

**Multi-user benchmarks and studies**

- GroupMemBench. Yang et al., 2026. 2605.14498 [F]
- Evaluating Long-Horizon Memory for Multi-Party Collaborative Dialogues (EverMemBench). Hu et al., 2026. 2602.01313 [F]
- SocialMemBench. Owolabi, 2026. 2605.17789 [F]
- GateMem. Ren et al., 2026. 2606.18829 [F]
- PiSAs: Benchmarking Contextual Integrity in Multi-User Agentic Systems. Gupta et al., 2026. 2607.05318 [F]
- Multi-User Large Language Model Agents. Yang et al., 2026. 2604.08567 [F]
- When Single-User-Oriented LLM-based Assistants Involve Others: A Scoping Review. Chen et al., 2026. 2609.14062 [F]
- No Attacker Needed: Unintentional Cross-User Contamination. Yang et al., 2026. 2604.01350 [A]
- MemLeak: Cross-User Semantic Leakage in Multi-Tenant AI Agent Memory. Mudgal et al., 2026. 2610.04195 [A]
- COLLABMEM. Wu et al., EMNLP 2026; ACL Anthology PDF, no arXiv version found [F]
- MAP-Graph. 2026. 2608.10509 [F]
- When Memory Becomes Authority. 2026. 2608.01679 [F]
- Enterprise AI Must Enforce Participant-Aware Access Control. 2025. 2509.14608 [A]
- LatticeMind. 2026. 2608.08236 [A]
- MoM / P-Mem. 2026. 2609.25054 [A]
- FLOWSEAL. 2026. 2609.14003 [A]
- AfterDeleteBench. 2026. 2610.31366 [A]
- CrossLearnBench. 2611.10439 [A]
- BEMULA. 2610.18052 [A]
- MuPPET. Ruzzetti et al., 2026. 2606.23217 [A]

**Conflicts, updates and time**

- When Personal Memory Has No Single Answer (TANGLE). Yang et al., 2026. 2608.13921 [F]
- MemConflict. Tao et al., 2026. 2605.20926 [F]
- MemoryAgentBench. Hu, Wang & McAuley, 2025. 2507.05257 [F]
- STALE. Chao et al., 2026. 2605.06527 [A]
- Supersede. Patel, 2026. 2606.27472 [A]
- Ground Truth First. Spencer, 2026. 2607.21962 [A]
- Selective QA over Conflicting Multi-Source Personal Memory. Yang et al., 2026. 2605.30087 [A]
- When Does Belief-Based Agent Memory Help? (Nous). Singh, 2026. 2606.22030 [A]
- Knowledge Conflicts for LLMs: A Survey. Xu et al., 2024. 2403.08319 [A]
- Retrieval-Augmented Generation with Conflicting Evidence (MADAM-RAG). Wang et al., 2025. 2504.13079 [A]
- Zep: A Temporal Knowledge Graph Architecture for Agent Memory. Rasmussen et al., 2025. 2501.13956 [A]
- Less Context, More Accuracy (Engram). Wang, 2026. 2606.09900 [A]
- Kumiho. Park, 2026. 2603.17244 [A]

**Single-user memory**

- LongMemEval. Wu et al., ICLR 2025. 2410.10813 [F]
- Mem0. Chhikara et al., 2025. 2504.19413 [F]
- Beyond the Context Window. Pollertlam et al., 2026. 2603.04814 [F]
- Anatomy of Agentic Memory. Jiang et al., 2026. 2602.19320 [F]
- HaluMem. Chen et al., 2025. 2511.03506 [A]
- MemGPT. Packer et al., 2023. 2310.08560 [A]
- Generative Agents. Park et al., 2023. 2304.03442 [A]
- A-MEM. Xu et al., 2025. 2502.12110 [F]
- Nemori. Ma et al., 2025. 2508.03341 [F]
- EverMemOS. Hu et al., 2026. 2601.02163 [F]
- HippoRAG / HippoRAG 2. Gutiérrez et al. 2405.14831 / 2502.14802 [A]
- Sleep-time Compute. Lin et al., 2025. 2504.13171 [A]
- ReasoningBank. Ouyang et al., 2025. 2509.25140 [F]
- ACE. Zhang et al., 2025. 2510.04618 [A]
- Memory-R1. Yan et al., 2025. 2508.19828 [A]
- Hindsight. Latimer et al., 2025. 2512.12818 [A]
- Memory in the Age of AI Agents (survey). Hu et al., 2025. 2512.13564 [A]
- Always-On Agents (survey). Ding et al., 2026. 2606.30306 [F]
- Memory for Autonomous LLM Agents (survey). Du, 2026. 2603.07670 [F, partly summary]

**Security, privacy and deletion**

- MemLineage. Ouyang & Hou, 2026. 2605.14421 [F]
- MutMem. Saidi, 2026. 2608.02843 [F]
- MINJA. Dong et al., 2025. 2503.03704 [A]
- AgentPoison. Chen et al., 2024. 2407.12784 [A]
- A-MemGuard. Wei et al., 2025. 2510.02373 [A]
- MAPLE-Guard. Xiong et al., 2026. 2608.00426 [A]
- Origin-Bound Authority (TMA-NM). Louck, 2026. 2606.24322 [A]
- Share-Borne AI Virus. Pulipaka et al., 2026. 2609.35576 [A]
- MEXTRA. Wang et al., 2025. 2502.13172 [A]
- CIMemories. Mireshghallah et al., 2025. 2511.14937 [A]
- ConfAIde. Mireshghallah et al., 2023. 2310.17884 [A]
- PrivacyLens. Shao et al., 2024. 2409.00138 [A]
- AirGapAgent. Bagdasarian et al., 2024. 2405.05175 [A]
- HUSH-Bench. Xu et al., 2026. 2606.06055 [A]
- CaMeL. Debenedetti et al., 2025. 2503.18813 [A]
- FIDES. Costa et al., 2025. 2505.23643 [A]
- Ghost Vectors. Chakraborttii et al., 2026. 2606.18497 [A]
- ForgetEval. Yang, 2026. 2606.15903 [A]
- DP-MemView. Kim et al., 2026. 2608.03130 [A]
- A Survey on Long-Term Memory Security in LLM Agents. Lin et al., 2026. 2604.16548 [A]

**Theory (classic, cited from memory)**

- Wegner 1987, transactive memory; Lewis 2003, measuring transactive memory [M]
- Clark & Brennan 1991, grounding in communication; Clark 1996, *Using Language* [M]
- Johnson, Hashtroudi & Lindsay 1993, source monitoring [M]
- Weldon & Bellinger 1997, collaborative inhibition; Roediger, Meade & Bergman 2001, social contagion of memory; Rajaram & Pereira-Pasarin 2010 [M]
- Doyle 1979, truth maintenance; de Kleer 1986, ATMS; Alchourrón, Gärdenfors & Makinson 1985, belief revision [M]
- Yin, Han & Yu 2007, TruthFinder; Dong et al. 2015, Knowledge-Based Trust (1502.03519) [A]
- Myers & Liskov 1997, decentralized label model [M]
- Pang et al. 2019, Zanzibar [A]; Shapiro et al. 2011, CRDTs [M]; Snodgrass, bitemporal databases [M]
- Nissenbaum, contextual integrity [M]
- Gatterbauer, Balazinska, Khoussainova & Suciu 2009, belief databases (PVLDB) [A]; Gatterbauer & Suciu 2010, trust mappings (SIGMOD) [A]
- Foster, Green & Tannen 2008, annotated XML and security semirings (PODS) [A]
- Jajodia & Sandhu 1990–93, polyinstantiation in multilevel secure databases [A]
- Buneman, Khanna & Tan 2002, deletion propagation via why-provenance (PODS) [A]
- Green, Karvounarakis, Ives & Tannen 2007, Orchestra update exchange (VLDB) [M]
- Gupta, Mumick & Subrahmanian 1993, DRed view maintenance [M]
- Agrawal et al. 2002, Hippocratic databases (VLDB) [M]
