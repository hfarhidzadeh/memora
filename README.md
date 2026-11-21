# Memvane

Memvane is a local-first prototype for **self-evolving agent memory**. It
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
- Records the channel a memory was said in and who said it, and can search as
  a named set of viewers who see only the channels they all belong to
  (Python API only; see [Channels and viewers](#channels-and-viewers)).
- Keeps all data in a local SQLite file. There are no runtime dependencies;
  the optional semantic embedding backend also runs locally.

This is a research prototype, not a production multi-agent or LLM platform. It
does not call an LLM or extract entities automatically, and links are written
explicitly. Pruning archives records rather than permanently deleting them.

For how the code is organised, what the database holds and what each read and
write does, see [docs/architecture.md](docs/architecture.md).

## Requirements

Python 3.9 or newer.

## Install

```bash
python -m pip install -e .
python -m pip install -e '.[embeddings]'   # optional: semantic vectors
```

The package and command are now `memvane`; reinstall after updating an older
checkout. Python imports remain `dynmemory`. The default database remains
`.memora.sqlite3` for compatibility with existing stores; use `--db` to select
another path. Existing database contents need no migration for this rename.

Or run from a checkout without installing:

```bash
PYTHONPATH=src python -m dynmemory --help
```

## Quick start

```bash
memvane --db ./agent-memory.sqlite3 init
memvane --db ./agent-memory.sqlite3 add \
  "Prefer SQLite for local memory storage." \
  --type user_preference --key storage.backend --importance 0.8
memvane --db ./agent-memory.sqlite3 add \
  "The agent's memory must stay on the user's machine." \
  --type technical_constraint
memvane --db ./agent-memory.sqlite3 search "SQLite"
memvane --db ./agent-memory.sqlite3 list
```

The `add` command prints each new memory's ID. Use `memvane link <source-id>
<target-id> --relation supports` to connect two active memories; searching one
can then also retrieve the other.

Bad input (an unknown ID, a confidence above 1, a missing file) prints a
one-line `memvane: error: ...` message and exits with status 1.

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
memvane --db ./agent-memory.sqlite3 history <memory-id>
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
the provisional LongMemEval runs, carrying links over appeared to help mainly
because the updated memory keeps its connections
(see [LongMemEval](#longmemeval-knowledge-updates)).

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

`score` only orders one search's results, so it cannot tell a strong best match
from a weak one. Each result therefore also carries the raw signals behind it,
which are comparable across searches: `lexical_score` (the BM25-style score),
`vector_score` (cosine similarity, 0 below the embedder's `min_similarity`) and
`coverage` (the share of the question's inverse-document-frequency mass that
the memory contains; a rare question word that no memory contains counts
against every memory). Each is 0 when the memory did not match that way. A
caller can use them to stay silent when even the best match is weak; see
[LongMemEval: silence](#longmemeval-silence).

Each search counts as an access, which protects memories from pruning. Use
`search --peek` (or `record_access=False` in Python) for exploratory or
evaluation queries that should not affect pruning.

## Channels and viewers

A first step towards memory shared by several people. It is available from
Python only; the command line has no options for it yet.

```python
from dynmemory import MemoryStore

store = MemoryStore("./shared.sqlite3", require_viewers=True, require_writer=True)
store.join_channel("design", "alice")
store.join_channel("design", "bob")
store.join_channel("board", "dana")

store.for_writer("alice").add_memory("The launch is on Friday.", "fact", channel="design")
store.for_writer("dana").add_memory("The launch may be cancelled.", "fact", channel="board")

bob = store.for_viewers("bob")
bob.search("launch")                          # the design memory only
store.for_viewers(["bob", "dana"]).search("launch")  # nothing: no shared channel
store.search("launch")                        # raises: viewers are required
store.add_memory("x", "fact")                  # raises: bound writer required
```

- **Who sees what.** `viewers` names the people who will see the answer. A
  memory is visible if every viewer is a current member of its channel.
  `for_viewers` binds an audience to `search`, `get_memory`, `history` and
  `list_memories`. Obtain viewer identities from a trusted source and give
  the bound reader to code serving those people. The store also accepts a
  `viewers` argument directly on each of these methods.
- **Guard user-facing reads.** `require_viewers=True` makes those four store
  methods raise if the audience is omitted. The setting belongs to the store
  instance, not the database file. The CLI opens an unguarded store and is an
  administrative tool for someone who can access the SQLite file.
- **Guard member writes.** `for_writer(member_id)` binds the `speaker` to a
  trusted identity and checks current channel membership inside the write
  transaction. `require_writer=True` rejects direct `add_memory` calls on that
  store instance. Give the bound writer to user-facing code; keep the store
  itself with trusted administrative code. Authentication and protecting the
  SQLite file remain the application's responsibility.
- **Membership is checked when reading.** Someone who joins a channel can read
  everything already in it. Someone who leaves (`leave_channel`) loses it at
  the next read, including what they said themselves.
- **Hidden memories are left out before embedding and ranking.** They change
  no score, add no linked neighbour and appear in no link, including
  `contradicts` links and links carried over between versions.
- **A memory without a channel is hidden from every viewer.** Memories written
  before channels existed have none.
- **Keys belong to a channel.** The same key in two channels names two
  separate memories, and a memory can only supersede one in its own channel.
  A write in one channel therefore never replaces a memory in another.
- **Claims are resolved when read.** A channel memory is never archived or
  superseded when another is written. Each read works out which claims
  replace which, their `claim_status`, the `contradicts` links between claims
  of one key, and `superseded_by` and `history`, from the memories that read
  may use: visible to its viewers and not stale. A claim drawn from a source
  some viewers cannot see therefore replaces nothing for them, and a retracted
  claim no longer replaces anything. An explicit `supersedes` link between
  channel memories is applied the same way, and cannot target a keyed claim
  (write a new claim for the key instead). `relate` checks only that its two
  memories exist and are not archived (pruned or stale); it can link a claim
  that some viewers see as replaced, and the link counts for viewers in whose
  view that claim still stands.
- **Speaker claims remain attributed.** A keyed channel memory needs a
  `speaker`. That speaker can correct their earlier claim. A different
  speaker's differing claim remains active with a `contradicts` link. Search
  returns both with `claim_status="disputed"`. Text comparison ignores case
  and surrounding whitespace; it does not detect semantic equivalence.
- **Admin resolution.** A trusted caller can register a workspace admin with
  `store.add_admin("dana")`. An admin's keyed channel write replaces
  non-admin claims and closes their dispute. A later non-admin disagreement is
  shown as `claim_status="overruled"` while the admin claim stays `"current"`.
  Two admins' first differing claims remain disputed until one restates.
  Admin authority is recorded on each claim when it is written, so appointing
  someone later does not promote their earlier statements. Resolved claims
  remain available through `history`.
- **Personal facts.** Pass `fact_subject="carol"` with a channel and key for a
  claim about Carol. Carol's own correction supersedes other reports, including
  an admin's. An admin's report about Carol does not gain team-fact authority.
  A key keeps one scope and personal subject throughout its history; choose
  separate keys when two people have the same attribute.
- **Delegation.** A trusted store caller can grant one team key in one channel:
  `store.grant_delegation("design", "review", "alice", granted_by="dana")`.
  The grantor must be a workspace admin and Alice a current channel member.
  Alice's next write to that key can resolve ordinary disagreement, while an
  admin still outranks it. `revoke_delegation(..., revoked_by="dana")` ends the
  grant; leaving the channel also ends it. Claims retain the authority they
  had when written. A former delegate cannot replace their own still-active
  delegated claim with an ordinary write. Grant and revocation events remain
  in the database.

Limits:

- A store opened without `require_viewers=True` still permits unrestricted
  reads for single-user and administrative use. Applications must keep that
  store and the database file away from untrusted callers.
- The store itself does not authenticate identities. `add_admin`, membership
  changes, direct administrative writes, and the SQLite file require trusted
  code. For untrusted callers, use signed tokens and the local server
  (see [Authentication and the local server](#authentication-and-the-local-server)).
- Access counts and pruning are shared by everyone in the store.
- A bound writer's writes are judged only on the claims that writer can see:
  the refusal of a change of a key's scope or personal subject, and of a
  lapsed authority revising its own claim, cannot reveal a hidden claim.
  Trusted calls through the store itself still check every claim. A key can
  therefore hold claims of different scope when some are hidden from the
  writer; reads resolve each claim by its own authority.
- Disagreements are shown as tables by `disputes()`, worked out at read time
  rather than stored. Stale derivations are rebuilt by `rebuild_stale()` with a
  consolidator you supply; the store calls no model itself. The broader design
  is in [docs/multi_user_memory_review.md](docs/multi_user_memory_review.md).

### Source messages and retraction

Use source messages when a claim or summary is derived from human text:

```python
writer = store.for_writer("alice")
message = writer.add_message("The launch moved to Monday.", channel="design")
claim = writer.add_memory(
    "The launch is Monday.", "fact", channel="design", memory_key="launch",
    source_message_ids=[message.id],
)
store.for_viewers("bob").sources_of(claim.id)  # the attributed source message
writer.retract_message(message.id)
store.stale_memories()                           # trusted rebuild queue
```

A derived memory is readable only by current members of its own channel **and
every source message's channel**. A writer must currently belong to each source
channel. Retraction replaces the message content with a tombstone, marks every
memory listing it as a source stale, archives those memories, replaces their
text with `[retracted]` (the row keeps its ID, key, channel and speaker, so a
rebuild knows what to redo), and removes their embeddings. Search, history, direct reads and link traversal exclude stale
memories, and whatever a stale memory had replaced stands again at the next
read: retracting a correction brings back the earlier claim, and retracting an
admin's resolution reopens the dispute. `restore` cannot reactivate a stale
memory. Rebuild by recomputing its content
and adding a new memory with only valid source message IDs; the store cannot
regenerate model-written text itself. Trusted code can inspect the old source
list with `sources_of(memory_id, include_stale=True)`.

Existing memories without `source_message_ids` retain their previous behavior
and cannot be invalidated through message retraction. Source messages are
retrievable by ID, but are not yet indexed by text search. No live row keeps the
retracted text, and with `secure_delete` (the default, `secure_delete=False` to
turn it off) SQLite overwrites the bytes of deleted and replaced rows: in the
retraction experiment the raw database file held a retracted word in 0 of 500
trials, against 495 of 500 without it. Copies of the file made earlier,
backups, rollback journals and the filesystem's own remnants are not covered. When
deriving from another memory, the caller must carry forward its complete raw
source-message list and preserve the parent's channel restrictions; the store
does not track parent-memory dependencies or verify that source text supports
the derived claim.

Why leave hidden memories out before ranking instead of dropping them from the
results? Because the scores of the memories a viewer can see depend on the
ones they cannot. `experiments/audience_side_channel.py` shows this on
synthetic data (200 trials, lexical search). A viewer who searched for a word
next to a nonsense word could tell, in 200 of 200 trials, that the word
occurred in a channel they were not in, even in a single memory, when hidden
memories were dropped after ranking. With `viewers` the two searches were
identical in 200 of 200 trials, and the probe still fired when the word was in
the viewer's own channel (positive control). The data were written alongside
the code, so this shows the mechanism, not how often it matters in real use.
Results: `experiments/results/audience_side_channel.md`.

On real text, `experiments/longmemeval_shared_workspace.py` treats LongMemEval
question histories as shared workspaces (private, shared and team channels,
a joiner and a leaver, canaries, cross-channel links). Over 601 workspaces and
18,177 reads there were no leaks, every ranking matched a reference store built
from the experiment's own visibility record, and no visible item was blocked.
Six injected faults were each caught. Filtering before ranking costs owners
0.010 RR@10 against filtering after (interval -0.017 to -0.004). The access
structure is invented. A blind human spot-check of 40 of the 120 other-user
answer-text hits found no leak (controls valid; see
`experiments/results/spot_check/outcome.md`), but 80 were not reviewed. Results: `experiments/results/longmemeval_shared_workspace.md`.

```bash
python3 -I experiments/audience_side_channel.py
```

### Disputes and who knows what

`disputes(viewers=...)` returns each channel key whose claims have disagreed,
as a table with one row per speaker's latest claim: who, says, when, where and
status (`current`, `disputed`, `agrees` or `overruled`). After an admin's
resolution the table stays available as history: the admin's row is current,
claims that matched it agree and the others are overruled. `open` tells whether
standing claims still disagree.

`who_knows(memory_id, viewers=...)` reports who **stated** the value (the
speaker, and anyone holding a standing claim with the same value), who
**acknowledged** it (replied or reacted to a source message, recorded with
`writer.acknowledge(message_id, "replied")`), and who only **had access**.
Being in the channel never counts as knowing.

Both are computed from what the viewers can see: a hidden claim adds no row,
changes no status and makes nobody a stater. In
`experiments/audience_disputes_who_knows.py` (500 pairs) both answers were
identical with and without hidden claims and acknowledgements, a reply counted
as knowing and presence only as access; with an injected fault that computed
them over every memory, they differed in 496 of 500 pairs.

```python
writer = store.for_writer("bob")
writer.acknowledge(message.id, "replied")
store.for_viewers("carol").disputes()
store.for_viewers("carol").who_knows(claim.id)
```

### Rebuilding stale memories

`store.rebuild_stale(consolidator)` takes each memory in the stale queue and
calls `consolidator(stale_memory, remaining_sources)` with only the source
messages that were not retracted. The returned text becomes a new memory that
lists exactly those sources and keeps the original kind, key, channel, speaker,
subject, write-time authority and place in write order, and the stale memory
leaves the queue. `ExtractiveConsolidator` (in `dynmemory.consolidation`)
simply joins the remaining texts; a model-based consolidator can be passed if
it sees nothing beyond the sources it is given. A memory with no remaining
source stays in the queue. In the retraction experiment (500 trials) 684
memories were rebuilt, each from exactly its remaining sources and none with a
retracted word; a consolidator that leaked a retracted word was caught in every
trial that had something to rebuild.

### Encryption at rest

`MemoryStore(path, key_path="memory.keys")` stores every memory's and message's
text, and every vector, encrypted with a one-time pad of random bytes held in a
separate key database. Reads are unchanged. Retraction destroys the pads of the
message and of the memories derived from it, so a copy or backup of the main
database taken before the retraction can no longer be read for that text. A
backup of the key file restores the pads it holds, so keep it separate or do
not back it up. Rows written before encryption was turned on stay readable as
plain text until they are rewritten. Standard library only. In the retraction
experiment no copy made before a retraction yielded retracted text with the key
file (500 of 500 trials); without destroying the pads, every copy did.

### Authentication and the local server

`dynmemory.auth.TokenAuthority(secret)` issues tokens signed with HMAC-SHA256
that name a member and expire (`issue(member_id, ttl_seconds=...)`,
`verify(token)`, `revoke(token)`). `open_session(store, token, authority)`
returns a reader and a writer bound to the verified member, so untrusted code
never chooses an identity.

`dynmemory.server.MemvaneServer` serves a store over a Unix socket created with
owner-only permissions. Every request carries a token; reads run as the
verified audience and writes as the verified member, and administrative calls
are not exposed. `MemvaneClient(socket_path, token)` is the matching client.

```bash
memvane new-secret ./secret                         # 32 random bytes, mode 0600
memvane --db team.sqlite3 serve --socket ./memvane.sock --secret-file ./secret
memvane token alice --secret-file ./secret          # give this token to alice
```

### Equivalence and update detection

By default two claims of one key agree only if their text matches after
trimming and case folding. `MemoryStore(..., claim_equivalence=...)` replaces
that test; `dynmemory.detection.EmbeddingEquivalence(embedder, threshold=0.9)`
treats claims whose embeddings are close enough as agreeing.
`store.detect_update(text, detector, viewers=...)` returns the visible memory a
new text would replace, according to a detector such as
`OverlapUpdateDetector` or `EmbeddingUpdateDetector`; the caller decides what
to do with the answer. These baselines have not been evaluated on real
conversations: that waits for the human review of the LongMemEval labels.

### Command line for channels and viewers

The CLI now covers channels, admins, delegation, messages, disputes,
who-knows, the stale queue and rebuilding. `--viewers alice,bob` scopes reads,
`--as alice` writes as a member with membership checks, and `--key-file`
turns on encryption. The CLI still authenticates nobody: anyone who can run it
on the database file can choose those options.

```bash
memvane --db team.sqlite3 channel join design alice
memvane --db team.sqlite3 --as alice message add "Review is Thursday" --channel design
memvane --db team.sqlite3 --viewers carol disputes
memvane --db team.sqlite3 --viewers carol who-knows MEMORY_ID
```

## Embeddings

Choose a backend with `--embedder` before the command:

```bash
memvane --db ./agent-memory.sqlite3 --embedder ngram search "invoice"
memvane --db ./agent-memory.sqlite3 --embedder sentence-transformers search "short responses"
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
memvane --db ./agent-memory.sqlite3 prune --min-age-days 30
memvane --db ./agent-memory.sqlite3 prune --decay episode=14 --decay task=30
memvane --db ./agent-memory.sqlite3 prune --min-age-days 30 --apply
memvane --db ./agent-memory.sqlite3 list --include-archived
memvane --db ./agent-memory.sqlite3 restore <memory-id>
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
memvane eval eval/synthetic_set.json
memvane --embedder ngram eval eval/synthetic_set.json
memvane --embedder sentence-transformers eval eval/synthetic_set.json --json
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

**Provisional results, not settled findings.** The November 19, 2026
[review of all 70 labels](docs/longmemeval_label_review.md) found category,
evidence, and update-semantics issues. The numbers below describe the original
labels and conversion; they need a corrected conversion and new runs before
being used as research conclusions.

[LongMemEval](https://arxiv.org/abs/2410.10813) (Wu et al., ICLR 2025; MIT
licence) is a public benchmark of long chat histories with questions about
them. It contains no patient data. Memvane converts it into evaluation suites:

```bash
mkdir -p data/longmemeval
curl -L -o data/longmemeval/longmemeval_s_cleaned.json \
  https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
PYTHONPATH=src python -m dynmemory.longmemeval data/longmemeval/longmemeval_s_cleaned.json \
  --labels eval/longmemeval_knowledge_update_labels.json \
  --out data/longmemeval/knowledge_update.jsonl
memvane --link-carryover topic eval data/longmemeval/knowledge_update.jsonl
```

The download is 277 MB. Each question becomes one evaluation set: the user's
turns (oldest first) as memories, a link between consecutive user turns of a
session, and the question with its evidence turns as relevant. For
knowledge-update questions, the older version of the fact is listed as stale
and an update records that the newer turn replaces it. Those updates come
from the benchmark's evidence labels. The original experiment called them an
oracle, but the review found that evidence order does not always establish
supersession. They are a benchmark-derived heuristic requiring correction.

The newest evidence is not always the answer: some questions ask for the
previous value or need both. The labels file gives each update question a
category, and the converter follows it. The labels were written by an AI model.
A second AI review is recorded for every label in
[the review ledger](eval/longmemeval_knowledge_update_label_review.json).
The original categories are preserved to reproduce the existing results;
recommended changes and unresolved conversion issues are recorded separately.

[experiments/longmemeval_knowledge_update.py](experiments/longmemeval_knowledge_update.py)
compares the three link options with negative and positive controls and a
decision rule fixed before it ran. The full write-up is in
[experiments/results/longmemeval_knowledge_update.md](experiments/results/longmemeval_knowledge_update.md).
In the historical runs on the 50 questions originally labelled “current”:

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

## LongMemEval: silence

Search always returns its best matches, so a question whose answer was never
stored gets something back anyway. Each result carries the raw signals behind
its rank (see [Retrieval](#retrieval)), so a caller can stay silent when even
the best match is weak.
[experiments/longmemeval_silence.py](experiments/longmemeval_silence.py)
asks how well that works, using the benchmark's 30 abstention questions. Each
asks about a detail the user never gave ("the name of my hamster" when they only
mentioned a cat). For 29 of them the benchmark also holds the answerable
original question, which gives matched pairs. Controls: questions about
something unrelated to the store (silence should be easy) and random words
(no separation expected). The decision rule was fixed before the run; one
control turned out to be flawed and was replaced afterwards, with no threshold
changed (details in the write-up). The full write-up is in
[experiments/results/longmemeval_silence.md](experiments/results/longmemeval_silence.md).

- **The signals carry information.** For three signals (BM25 score, coverage of
  the question's rare words, MiniLM similarity), the answerable question scored
  above its near-miss twin in 22 to 25 of 29 pairs.
- **But a gate silences few near-miss questions.** A threshold that keeps 90%
  of answerable questions silenced 10% to 30% of the near-miss ones. It
  silenced 38% to 86% of questions about something unrelated (coverage 72% to
  83%, depending on the calibration set), so coverage or MiniLM work as an
  "unrelated question" filter, not as a check for a missing detail. By the
  pre-set rule none of the signals is a useful gate.
- **Silencing is not free.** The overall hit rate fell from 0.88 to about 0.82.
  The questions it silenced were less likely to be hits than the rest (the
  evidence was in the top 5 for 54% of them against 91%), but over half of them
  still were.

## Run tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Next research steps

1. Detect updates automatically (for example with a local contradiction or
   similarity model) and measure how much of the oracle's gain on LongMemEval
   it recovers. The incomplete Qwen run is
   [archived as a failed diagnostic](experiments/results/failed_update_detection_qwen2.5-coder-1.5b/README.md),
   not evidence of detector effectiveness.
2. Keep old versions findable: rank them lower instead of archiving them, or
   search history for questions about the past, and measure both question
   groups.
3. Tune `min_similarity` and the fusion for MiniLM on part of LongMemEval and
   test on the rest; at the defaults it lowers MRR.
4. Resolve the [label review findings](docs/longmemeval_label_review.md), separate
   answer evidence from supersession labels, and rerun the experiments before
   extending the conversion to other question types.
5. Build a labelled set from real (non-patient) agent sessions and compare
   configurations against their controls there too.
6. Introduce LLM-guided consolidation behind a reviewable proposal interface.
7. Resolve contradictions. Search now surfaces them through `links`; deciding
   between conflicting memories is still open.
8. Add entity-level provenance and permission-aware shared-memory operations.
9. Benchmark retention quality, storage growth, and unlearning guarantees
   before considering irreversible deletion.
