# Knowledge updates on LongMemEval

**Question.** When a fact changes ("my 5K best is 27:12", later "25:50"), the
newer memory replaces the older one. Should the older memory's links to other
memories carry over to the newer one?

**Short answer.** Not because of what the links say. Carrying links over
helped a little, but random links helped almost as much, so the gain comes
from the new memory not losing its connections, not from the links' content.
By the rule fixed in advance, there is no benefit, so automatic link
extraction (the planned next step) was not tried. Two larger findings came
out instead:

1. **Recognising updates matters.** Hiding the outdated version raised the
   current answer's reciprocal rank by 0.16. Without it, the outdated answer
   ranked above the current one in 38% of questions (54% with semantic
   search).
2. **Hiding old versions has a cost.** Of the update questions, 15 in 69 ask
   for the earlier value or need both values. Archiving cut their reciprocal
   rank from 0.53 to 0.24. Old versions should stay findable.

## Data

- [LongMemEval](https://arxiv.org/abs/2410.10813) (Wu et al., ICLR 2025),
  `longmemeval_s_cleaned.json`, MIT licence, SHA-256 `d6f21ea9…c3a442`.
  Everyday chat conversations written for the benchmark; no patient data.
- Each question has its own chat history of about 50 sessions. Memora stores
  the user's turns (about 240 per question) as memories and links each turn
  to the user's next turn in the same session. These links come from the
  conversation's structure, not from any label.
- **Updates are an oracle.** The benchmark marks which turns hold the
  evidence; the newer evidence turn replaces the older one. This is perfect
  update detection, which a real system would have to achieve on its own.
- **Question labels are AI-written.** The 70 update questions were read and
  labelled by an AI model in
  [eval/longmemeval_knowledge_update_labels.json](../../eval/longmemeval_knowledge_update_labels.json):
  50 ask for the current value, 9 for the previous value, 6 need both, 3 ask
  for the latest of two separate events, 1 has no older value, and 1 was
  excluded because its session dates are reversed in the file. **Please
  review these labels.**

## Design

Three options for links when a fact is replaced:

| Option | What happens to the old version's links |
| --- | --- |
| keep | stay on the archived old version (default) |
| copy | copied to the new version at update time |
| topic | resolved to the new version at search time |

Each configuration runs on every question; the comparison is per question
(paired). For each comparison: the mean change in reciprocal rank, a 95%
bootstrap interval, and a two-sided sign-flip permutation test.

Controls:

- **Negative:** the same number of links, moved to random memories (10
  seeds). If carrying random links over helps as much, the links' content
  is not what matters.
- **Positive:** no session links; one planted link from a non-answer memory,
  ranked below the answer, to the outdated version. Carrying it over must
  help. If it doesn't, the experiment cannot detect an effect.

Decision rule, fixed before running: the positive control must improve with
p < 0.05; topic counts as a benefit only if topic vs keep improves with
p < 0.05 **and** its improvement beyond random links does too.

## Results

50 "current value" questions, lexical (BM25) search. MRR is mean reciprocal
rank of the current answer (top 10); R@3 is how often it is in the top 3;
"stale first" is how often an outdated version ranks above it.

| Configuration | MRR | R@3 | Stale first |
| --- | --- | --- | --- |
| No updates, no links | 0.563 | 0.84 | 0.38 |
| Oracle updates, no links | 0.720 | 0.88 | 0 |
| Oracle updates + session links, keep | 0.709 | 0.90 | 0 |
| ... copy | 0.729 | 0.88 | 0 |
| ... topic | 0.729 | 0.88 | 0 |

| Comparison (change in MRR) | Mean | 95% interval | p | Better / worse |
| --- | --- | --- | --- | --- |
| Oracle updates vs none | **+0.158** | +0.101 to +0.222 | <0.001 | 23 / 1 |
| Session links vs none (keep) | −0.011 | −0.047 to +0.022 | 0.53 | 3 / 4 |
| Topic vs keep (primary) | +0.020 | +0.002 to +0.045 | 0.063 | 5 / 1 |
| Copy vs topic | 0 | identical | | 0 / 0 |
| Topic vs keep, random links (negative control) | +0.009 | +0.004 to +0.014 | <0.001 | 14 / 0 |
| Topic vs keep, planted link (positive control) | **+0.088** | +0.036 to +0.151 | 0.001 | 11 / 0 |
| Topic vs keep beyond random links | +0.011 | −0.006 to +0.035 | 0.36 | 5 / 10 |
| Archiving, on the 15 previous/both questions | **−0.285** | −0.433 to −0.150 | 0.004 | 0 / 9 |

With MiniLM vectors added (local `all-MiniLM-L6-v2`, fused with lexical
search by reciprocal rank fusion, default threshold 0.3):

| Comparison (change in MRR) | Mean | 95% interval | p |
| --- | --- | --- | --- |
| MiniLM + lexical vs lexical alone | −0.086 | −0.171 to −0.009 | 0.044 |
| Real vs permuted vectors | +0.476 | +0.390 to +0.561 | <0.001 |
| Session links vs none (keep) | −0.094 | −0.177 to −0.014 | 0.028 |
| Topic vs keep | **+0.110** | +0.041 to +0.184 | 0.003 |
| Topic vs keep, random links | +0.064 | +0.046 to +0.084 | <0.001 |
| Topic vs keep beyond random links | +0.046 | −0.022 to +0.119 | 0.22 |

Without updates, semantic search put the outdated answer first in 54% of
questions (lexical: 38%).

## What this means

- **The experiment could detect an effect.** The positive control worked
  (+0.088), so the null result for carry-over is not just a blind test.
- **Copy and topic behave the same** on every question, as expected. Topic
  writes nothing extra; copy only works if it was on for every update.
- **Under keep, an updated memory is at a disadvantage.** It loses its
  predecessor's links while nearby memories keep boosting each other. With
  semantic search this costs a lot (adding session links lowered MRR by
  0.094). Topic restores the links and recovers the loss. That is a practical
  reason to prefer topic over keep, even though the links' content does not
  matter here.
- **MiniLM did not help on these questions** at its untuned default settings.
  Real vectors beat shuffled ones, so the vectors carry signal, but fusing
  them with lexical search lowered MRR. The threshold and fusion are worth
  tuning before drawing conclusions about semantic search.
- **Update detection is the bigger lever, but not by hiding history.** Many
  update questions ask about the past, and archiving makes those
  unanswerable from search. A design that ranks old versions lower instead
  of removing them, or searches history for "previous"-type questions, could
  keep both gains. That is a hypothesis to test, not a result.

## Limitations

- One benchmark, 50 primary questions; small effects are hard to detect
  (the primary comparison rests on 6 questions that changed at all).
- Updates come from the benchmark's labels, so the update gain is a ceiling
  for an automatic update detector.
- Links are only "next turn in the same session". Links with real content
  (e.g. from an extraction model) might behave differently; this experiment
  says they are not needed to explain the carry-over gain.
- The category labels are an AI model's reading of each question.
- Memories are user turns only; questions whose evidence is in assistant
  turns are not covered.

## Version 1 (superseded)

A first run ([v1_superseded](v1_superseded)) used the rule "newest evidence
is the answer" for all 70 questions and planted the positive control's link
on top of the session links. An independent review found that 15 questions
ask for the previous value or need both, one has reversed session dates, the
positive control contained the effect it was meant to check, and the decision
rule compared point estimates. Version 1's verdict was "inconclusive";
version 2 fixed these problems and its rule was set before it ran.

## Reproduce

```bash
# Python 3.9, Memora 0.2.0; MiniLM run with sentence-transformers 2.7.0, torch 2.2.2
PYTHONPATH=src python -m dynmemory.longmemeval data/longmemeval/longmemeval_s_cleaned.json \
    --labels eval/longmemeval_knowledge_update_labels.json \
    --out data/longmemeval/knowledge_update.jsonl
python3 -I experiments/longmemeval_knowledge_update.py            # about 4 minutes
python3 -I experiments/longmemeval_knowledge_update.py --embedder sentence-transformers
```

Per-question numbers are in the JSON files next to this report.
