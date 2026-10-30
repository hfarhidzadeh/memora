# Memora

Memora is a local-first prototype for **self-evolving agent memory**. It
demonstrates typed memory, explicit knowledge links, hybrid retrieval, and
conservative memory pruning in a small Python package backed by SQLite.

## What it does

- Stores memories as typed records: preferences, constraints, corrections,
  tasks, facts, episodes, and strategic insights.
- Keeps versions. Writing a stable key again, or linking a new memory as
  superseding an old one, archives the old version and records the
  replacement. `history` shows every version of a memory.
- Ranks active memories with a BM25-style lexical score, optionally fused with
  vector similarity, then expands results by one explicit relationship hop.
  Each result lists the links that connect it to the other results, and
  always shows when an active memory *contradicts* it.
- Tracks access counts and last-access times so repeated use protects a memory
  from pruning.
- Scores old memories by importance, confidence, inactivity decay (optionally
  per memory type), and access frequency. High-importance memories are
  protected, candidates are previewed before they are archived, and archived
  memories can be restored.
- Measures retrieval and pruning against a labelled evaluation set, next to
  negative controls, and converts the public LongMemEval benchmark into such
  sets to test knowledge updates.
- Lets you choose what happens to a memory's links when a newer version
  replaces it (`--link-carryover keep|copy|topic`).
- Keeps all data in a local SQLite file. There are no runtime dependencies;
  the optional semantic embedding backend also runs locally.

This is a research prototype, not a production multi-agent or LLM platform. It
does not call an LLM or extract entities automatically, and links are written
explicitly. Pruning archives records rather than permanently deleting them.

## Requirements

Python 3.9 or newer.

## Install

```bash
python -m pip install -e .
python -m pip install -e '.[embeddings]'   # optional: semantic vectors
```

Or run from a checkout without installing:

```bash
PYTHONPATH=src python -m dynmemory --help
```

## Quick start

```bash
memora --db ./agent-memory.sqlite3 init
memora --db ./agent-memory.sqlite3 add \
  "Prefer SQLite for local memory storage." \
  --type user_preference --key storage.backend --importance 0.8
memora --db ./agent-memory.sqlite3 add \
  "The agent's memory must stay on the user's machine." \
  --type technical_constraint
memora --db ./agent-memory.sqlite3 search "SQLite"
memora --db ./agent-memory.sqlite3 list
```

The `add` command prints each new memory's ID. Use `memora link <source-id>
<target-id> --relation supports` to connect two active memories; searching one
can then also retrieve the other.

Bad input (an unknown ID, a confidence above 1, a missing file) prints a
one-line `memora: error: ...` message and exits with status 1.

## Links and versions

A link reads **source, relation, target**: `link A B --relation depends_on`
means "A depends on B". The relations are `related_to`, `supports`,
`contradicts`, `depends_on`, and `supersedes`.

`supersedes` is special. `link NEW OLD --relation supersedes` archives OLD and
records NEW as its replacement, exactly as writing the same `--key` again does.
If NEW has no key, it takes over OLD's key, so later writes to that key
continue the same history. A memory can only supersede an older one; to bring
back earlier content, add it again as a new memory.

```bash
memora --db ./agent-memory.sqlite3 history <memory-id>
```

`history` accepts any version and prints all of them, newest first. If pruning
archived a keyed memory and the key is written again later, the new memory is
linked to the pruned one, so the history stays connected.

Databases created by version 0.1 stored key-rewrite links in the opposite
direction (old supersedes new). They are corrected automatically the first
time the database is opened. Manually written links, and links to memories
that no longer exist, are left unchanged.

Every write takes SQLite's write lock before reading anything, so several
processes can share one database file without two of them replacing the same
version at once.

### Links of a replaced fact

`--link-carryover` (before the command) decides what happens to a memory's
links when a newer version replaces it:

| Option | Behaviour |
| --- | --- |
| `keep` (default) | links stay on the old, archived version, so they stop counting |
| `copy` | links are copied to the new version when the replacement happens |
| `topic` | links stay where they are; search treats a link to any version of a fact as a link to its current version |

`copy` and `topic` give the same search results when `copy` was in effect for
every update: both connect the current version to everything any earlier
version was linked to. `copy` writes the duplicate links when an update
happens, so a database updated under `keep` and later searched with `copy`
behaves like `keep`, and copied links stay whichever option later searches
use. `topic` writes nothing and resolves links on every search. On
LongMemEval, carrying links over helped mainly because the updated memory
keeps its connections (see [LongMemEval](#longmemeval-knowledge-updates)).

## Retrieval

Without an embedder, each memory's lexical score is normalised by the best
match. With an embedder, the lexical and vector rankings are combined by
reciprocal rank fusion (RRF): each memory scores the sum of `1 / (60 + rank)`
over the rankings it appears in. Agreement counts for more than position: a
memory ranked second lexically and third by vector beats one ranked first by
either retriever alone. This stops holding far down the lists; a memory
ranked below about 60th in both rankings scores less than the top of a single
one. In both cases the matches are then expanded by one relationship hop:

- a linked neighbour that did not match directly scores
  `0.65 × weight × score of the match that links to it`;
- a direct match that is also linked to another direct match gets a bonus of
  `0.35 × weight × that match's score`.

Relation weights are `contradicts` 0.6, `supersedes` 0.55, `supports` 0.5,
`related_to` 0.45 and `depends_on` 0.45. Scores are only comparable within one
search. Equal scores rank the newer memory first.

Each result reports `matched_by` (any of `lexical`, `vector`, `graph`) and
`links`. `links` lists:

- the result's relationships to the other results;
- the links behind its graph score, even to a match cut off by `--limit`;
- every `contradicts` link to an active memory, even one outside the results
  or the `--type` filter, so a disputed memory always shows it is disputed.

Links to archived memories are never listed.

Each search counts as an access, which protects memories from pruning. Use
`search --peek` (or `record_access=False` in Python) for exploratory or
evaluation queries that should not affect pruning.

## Embeddings

Choose a backend with `--embedder` before the command:

```bash
memora --db ./agent-memory.sqlite3 --embedder ngram search "invoice"
memora --db ./agent-memory.sqlite3 --embedder sentence-transformers search "short responses"
```

| Backend | Needs | Finds | Default `--min-similarity` |
| --- | --- | --- | --- |
| `ngram` | nothing | spelling and word-form variants ("invoice" for "invoices"); no notion of meaning | 0.25 |
| `sentence-transformers` | the `embeddings` extra | paraphrases ("short responses" for "concise answers") | 0.3 |

`--min-similarity` is the cosine similarity a memory needs to count as a
vector match. Both defaults are starting points: tune them on your own
labelled data.

The sentence-transformers backend uses `all-MiniLM-L6-v2` unless
`--embedding-model` names another model. The weights are downloaded from
Hugging Face once unless they are already cached; set `HF_HUB_OFFLINE=1` to
forbid network access. Texts are always embedded on your machine; there is no
hosted-API backend.

Vectors are stored in the database per model. Memories that have no vector
from the current model (for example, ones added without `--embedder`, or after
switching models) are embedded at the next search. Similarity is computed in
pure Python, which is fine for thousands of memories but not millions.

From Python:

```python
from dynmemory import MemoryStore, NGramHashEmbedder

store = MemoryStore("agent-memory.sqlite3", embedder=NGramHashEmbedder())
```

Any object with a `name`, a `min_similarity`, and a `__call__(texts)` that
returns one vector per text works as an embedder.

## Pruning

Each active memory gets a retention score:

```
importance × confidence × exp(−idle_days / decay_days) × (1 + 0.25 × ln(1 + accesses))
```

`idle_days` counts from the last access, or from creation if the memory was
never accessed. Memories idle for at least `--min-age-days` that score below
`--threshold` are candidates. Pruning is a dry-run by default:

```bash
memora --db ./agent-memory.sqlite3 prune --min-age-days 30
memora --db ./agent-memory.sqlite3 prune --decay episode=14 --decay task=30
memora --db ./agent-memory.sqlite3 prune --min-age-days 30 --apply
memora --db ./agent-memory.sqlite3 list --include-archived
memora --db ./agent-memory.sqlite3 restore <memory-id>
```

- `--decay KIND=DAYS` gives one memory type its own decay (repeatable);
  other types use `--decay-days` (default 90).
- `--protect-importance` (default 0.9) exempts memories at or above that
  importance; `--protect-importance none` exempts nothing.
- `restore` reactivates a pruned memory and resets its last-access time, so it
  gets a fresh `--min-age-days` window. A memory that was superseded cannot be
  restored, because its replacement is the current version.

## Evaluation

```bash
memora eval eval/synthetic_set.json
memora --embedder ngram eval eval/synthetic_set.json
memora --embedder sentence-transformers eval eval/synthetic_set.json --json
```

An evaluation set is a JSON file with `memories` (key → content and kind),
`links`, `queries` (each with its `relevant` memory keys and free-form
`tags`), and optional `prune` cases with expected `keep`/`prune` outcomes. See
[eval/synthetic_set.json](eval/synthetic_set.json). Tags are free-form
except `all`, which names the overall score. The file is validated before
use, and errors name the offending field.

To test knowledge updates, a set can also list:

- `updates`: `{"old": key, "new": key}` pairs, each meaning the new memory
  replaces the old one. The new memory must be listed after the old one.
  Updates are applied after the links, in every configuration.
- `stale` on a query: memories that hold an outdated answer. The report then
  adds a "stale first" column: the share of queries where an outdated memory
  ranks above every current one.

Links may use any relation except `supersedes`; list replacements under
`updates` instead, so that they apply to every configuration and not only to
the ones that use links.

A `.jsonl` file holds a suite: one evaluation set per line, each searched in
its own store, with metrics pooled over all queries. Use this when every
question has its own memories, as in LongMemEval.

The report gives recall@k (default k = 3) and MRR@10 overall and recall per
tag for each configuration. Each mechanism is paired with a negative control
that keeps its shape but removes its signal, averaged over random seeds
(`--seeds`, default 20):

| Mechanism | Treatment | Negative control |
| --- | --- | --- |
| graph hop | lexical + graph | each link moved to a memory its source is not linked to |
| vector retrieval | lexical + vector | each memory embedded as if it were a different memory |

The combined configuration (lexical + vector + graph) has no control of its
own. A mechanism is only supported when it beats both the configuration
without it and its control. Beating the control alone is a weak bar: wrong
vectors push relevant memories down, so the permuted-vector control scores
below plain lexical search on MRR (0.35–0.36 against 0.50 on the synthetic
set). The report also lists the queries the fullest configuration missed,
and for pruning shows each case's score, outcome and label.

Results on the synthetic set (recall@3, version 0.2.0; MiniLM run with
sentence-transformers 2.7.0 and torch 2.2.2; controls averaged over 20 seeds):

| Configuration | needs_link | direct | word_form | paraphrase | all 12 |
| --- | --- | --- | --- | --- | --- |
| lexical | 0.50 | 1.00 | 0.00 | 0.00 | 0.38 |
| lexical + graph | 1.00 | 1.00 | 0.00 | 0.00 | 0.50 |
| lexical + graph, wrong links *(control)* | 0.50 | 1.00 | 0.00 | 0.00 | 0.38 |
| lexical + n-gram vectors | 0.50 | 1.00 | 1.00 | 0.00 | 0.62 |
| lexical + n-gram vectors, permuted *(control)* | 0.54 | 1.00 | 0.00 | 0.03 | 0.39 |
| lexical + MiniLM vectors | 0.67 | 1.00 | 1.00 | 1.00 | 0.92 |
| lexical + MiniLM vectors, permuted *(control)* | 0.54 | 1.00 | 0.00 | 0.00 | 0.39 |
| lexical + MiniLM vectors + graph | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

Pruning matches all 7 labelled cases.

The MiniLM paraphrase score is fragile. For two of the three paraphrase
queries, the relevant memory's similarity is 0.31 or 0.32, just above the 0.3
threshold. For "preferred reply length" an irrelevant memory scores higher
(0.33) and ranks first; the query only counts as answered because recall@3 has
room for both. Only "accessible chart colours" is a clear match (0.52 against
0.16 for the best irrelevant memory). A threshold of 0.33 would lose both weak
queries.

**Read these numbers as mechanism checks, not performance.** The set is
synthetic, it was written alongside the code, and each tag has only three
queries. The `needs_link`, `word_form` and `paraphrase` queries were built so
that word overlap alone cannot answer them. The results show that each
mechanism does what it is meant to do and that its control does not. They do
not show that any configuration helps on real agent conversations.

To measure that, use a public benchmark such as LongMemEval (below), or build
a set from real (non-patient) agent sessions: write the memories, decide which
ones each query should retrieve, and tag queries by the question you want
answered. Review the labels before trusting any number.

## LongMemEval: knowledge updates

[LongMemEval](https://arxiv.org/abs/2410.10813) (Wu et al., ICLR 2025; MIT
licence) is a public benchmark of long chat histories with questions about
them. It contains no patient data. Memora converts it into evaluation suites:

```bash
mkdir -p data/longmemeval
curl -L -o data/longmemeval/longmemeval_s_cleaned.json \
  https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
PYTHONPATH=src python -m dynmemory.longmemeval data/longmemeval/longmemeval_s_cleaned.json \
  --labels eval/longmemeval_knowledge_update_labels.json \
  --out data/longmemeval/knowledge_update.jsonl
memora --link-carryover topic eval data/longmemeval/knowledge_update.jsonl
```

The download is 277 MB. Each question becomes one evaluation set: the user's
turns (oldest first) as memories, a link between consecutive user turns of a
session, and the question with its evidence turns as relevant. For
knowledge-update questions, the older version of the fact is listed as stale
and an update records that the newer turn replaces it. Those updates come
from the benchmark's evidence labels, so they are an oracle: perfect update
detection, which a real system would have to achieve on its own.

The newest evidence is not always the answer: some questions ask for the
previous value or need both. The labels file gives each update question a
category, and the converter follows it. **The labels were written by an AI
model after reading each question; review them before relying on any
result.**

[experiments/longmemeval_knowledge_update.py](experiments/longmemeval_knowledge_update.py)
compares the three link options with negative and positive controls and a
decision rule fixed before it ran. The full write-up is in
[experiments/results/longmemeval_knowledge_update.md](experiments/results/longmemeval_knowledge_update.md).
On the 50 questions that ask for a current value:

- **Recognising updates matters most.** With the oracle updates, the current
  answer's MRR rose from 0.56 to 0.72. Without them, the outdated answer
  ranked first in 38% of questions (54% with MiniLM).
- **Carrying links over helped, but not because of what the links say.**
  Topic beat keep by +0.020 with lexical search (p = 0.06) and by +0.110 with
  MiniLM (p = 0.003). Random links gave most of that gain: under keep, an
  updated memory loses its predecessor's links. By the pre-set rule this is
  no benefit, so automatic link extraction was not tried.
- **Archiving old versions has a cost.** On the 15 questions about a past
  value or both values, MRR fell from 0.53 to 0.24.
- **MiniLM at its default settings lowered MRR** compared with lexical search
  alone (0.63 against 0.72).

## Run tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Next research steps

1. Detect updates automatically (for example with a local contradiction or
   similarity model) and measure how much of the oracle's gain on LongMemEval
   it recovers.
2. Keep old versions findable: rank them lower instead of archiving them, or
   search history for questions about the past, and measure both question
   groups.
3. Tune `min_similarity` and the fusion for MiniLM on part of LongMemEval and
   test on the rest; at the defaults it lowers MRR.
4. Review the LongMemEval labels, then extend the conversion to other question
   types (multi-session, temporal reasoning) with their own labels.
5. Build a labelled set from real (non-patient) agent sessions and compare
   configurations against their controls there too.
6. Introduce LLM-guided consolidation behind a reviewable proposal interface.
7. Resolve contradictions. Search now surfaces them through `links`; deciding
   between conflicting memories is still open.
8. Add entity-level provenance and permission-aware shared-memory operations.
9. Benchmark retention quality, storage growth, and unlearning guarantees
   before considering irreversible deletion.
