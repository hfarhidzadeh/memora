# Staying silent on LongMemEval

**Question.** Search always returns its best matches, so when the answer was
never stored it hands back something unrelated or misleading. Can Memora tell
from its own search signals that it has nothing to say, while still speaking
for questions it can answer?

**Short answer.** Partly. Coverage and MiniLM similarity can tell that a
question is unrelated to what is stored (the raw BM25 score only partly), but
none of the signals can tell that a question asks for a detail that was never
mentioned. By the rule fixed in advance, none of the three signals is a useful
gate:

- All three signals rank the answerable question above its near-miss twin in
  22 to 25 of 29 pairs (p between 0.0001 and 0.008), so they carry real
  information.
- A threshold that keeps 90% of answerable questions silences only 10% to 30%
  of the near-miss questions, but 38% to 86% of questions about something
  unrelated (BM25 38%; coverage 72% to 83% depending on how it is calibrated;
  MiniLM 86%).
- Silencing is not free. A gate costs about 6 points of hit rate (0.88 to about
  0.82). The answerable questions it silences were less likely to be hits than
  the ones it kept (the evidence was in the top 5 for 54% of them, against
  91%), but more than half of them still were.

## Data

- [LongMemEval](https://arxiv.org/abs/2410.10813) (Wu et al., ICLR 2025),
  `longmemeval_s_cleaned.json`, MIT licence. Everyday chat conversations
  written for the benchmark; no patient data.
- One store per question: the user's turns (240 to 250 on average), no links,
  no updates.
- **Near-miss questions (30).** The benchmark's abstention questions. Each
  asks about something close to what the user did say ("What is the name of
  my hamster?" when they only mentioned a cat). Silence is correct.
- **Twins (29).** For 29 of them the benchmark also holds the original
  question without the near-miss edit ("name of my cat"). Its chat history
  holds the same evidence turns as the near-miss's, with different filler
  sessions (no session is shared within a pair). Speaking is correct. A
  near-miss and its twin are a matched pair: same topic and question type.
- **Calibration sample (100).** Random answerable questions of the same four
  types (40 multi-session, 20 each of single-session-user, temporal-reasoning
  and knowledge-update; seed 0), none of them a twin. They only set the
  thresholds. The near-miss questions never influence a threshold.

## Design

Three signals, each the best value over all memories for the question
(search returns every memory, so nothing is hidden by a cutoff):

| Signal | What it is |
| --- | --- |
| lexical | the raw BM25-style score |
| coverage | the share of the question's word-rarity weight that one memory contains; a rare word that no memory contains counts against every memory |
| vector | the cosine similarity from MiniLM (`all-MiniLM-L6-v2`), with no minimum |

Memora speaks if the signal reaches a threshold. The threshold keeps a chosen
share of the calibration questions (retention): 90% is the main setting; 80%
and 95% are also shown.

Controls:

- **Positive (unrelated questions).** Each twin's store is asked a different
  twin's question, whose evidence sessions are not in that store. Silence is
  correct and should be easy. A signal that cannot silence these has no power
  to detect absence.
- **Negative (random words).** The same random words (as many as the twin's
  question) are asked of both stores of a pair. Any separation then comes from
  the stores, not the question.

Decision rule, fixed before running: a signal passes its controls if it
separates twins from unrelated questions with an area under the curve (AUC) of
at least 0.80 and shows no separation under random words (p ≥ 0.05/3). It
separates near-miss questions if the twin scores higher than its near-miss in
more pairs than the reverse, with an exact two-sided sign test p < 0.05/3. It
is a useful gate if it also silences at least half of the 30 near-miss
questions at 90% retention.

## Results

Of the 29 matched pairs, how often the twin scores higher than its near-miss,
and what the 90%-retention threshold does:

| Signal | Twin higher / near-miss higher | p | Paired AUC (95% CI) | Twins kept | Near-miss silenced (95% CI) | Unrelated silenced |
| --- | --- | --- | --- | --- | --- | --- |
| lexical | 23 / 6 | 0.0023 | 0.79 (0.66–0.93) | 90% | **10%** (3–26%) | 38% |
| coverage | 25 / 4 | 0.0001 | 0.86 (0.72–0.97) | 86% | **27%** (14–44%) | 83% |
| vector | 22 / 7 | 0.0081 | 0.76 (0.59–0.90) | 83% | **30%** (17–48%) | 86% |

Controls:

| Signal | Positive: AUC twins vs unrelated | Negative: random words, p |
| --- | --- | --- |
| lexical | 0.86 | 0.189 |
| coverage | 0.90 | 0.383 |
| vector | 0.86 | 0.036 |

All controls pass the rule's limits, but the negative control is weak:

- The limit is 0.05/3 = 0.0167, which is lenient for a control that should show
  no bias. At the conventional 0.05, the vector signal's control (p = 0.036)
  would fail and its verdict would be "inconclusive".
- It has little power. 12 of the 29 random-word queries match nothing in the
  twin store (also 12 in the near-miss store), so 8 pairs tie at zero for BM25
  and coverage. The median BM25 score of the random queries is 1.1 (twin
  stores) and 4.3 (near-miss stores), against 15.9 for the real twin questions.
  It is a single draw (seed 0).
- The vector signal leans toward the near-miss stores (20 of 29 pairs). The
  likely cause is store size: near-miss stores hold 250 memories on average
  against 240 for twin stores (larger in 20 of 29 pairs), and a maximum over
  more memories is higher. That bias points against the real result, so the
  separation is, if anything, understated.

A stronger control (a natural off-topic question asked of both stores of each
pair, repeated over several draws and tested without the correction) was not
run.

Other settings (share of near-miss questions silenced):

| Signal | 80% retention | 90% | 95% |
| --- | --- | --- | --- |
| lexical | 43% | 10% | 7% |
| coverage | 40% | 27% | 10% |
| vector | 40% | 30% | 20% |

Calibrating on all 355 answerable questions of those types, instead of the
sample of 100, leaves the share of near-miss questions silenced unchanged at
every setting except lexical at 80% retention (27% instead of 43%). The share
of unrelated questions silenced is less stable: coverage at 90% retention 72%
instead of 83% (threshold 0.292 instead of 0.300), at 95% 52% instead of 55%,
and lexical at 80% 59% instead of 72% (full numbers in the JSON). Treat 83% as
the upper end of a range.

Verdict, all three signals: **separates near-miss questions but silences too few.**

What a gate costs. Of the answerable questions, the lexical gate at 90%
retention silences 13 of 129 (10%). The evidence was in the top 5 for 91% of
the questions it kept and for 54% of those it silenced, so the overall hit rate
falls from 0.88 to about 0.82. The vector gate behaves the same way (0.87 to
about 0.81). These hit rates rank by the single signal, not by Memora's fused
ranking, count any one evidence turn as a hit (also for multi-session
questions), and cover the 129 answerable questions of the MiniLM run (the 29
twins and the 100 calibration questions). The BM25-only run covers all 384
answerable questions: its gate silences 45, with hit rates of 0.90 for the
kept and 0.51 for the silenced.

Near-miss questions the coverage gate caught at 90% retention (8 of 30): 3 of 6
temporal-reasoning, 4 of 12 multi-session, 1 of 6 single-session-user and 0 of 6
knowledge-update. Every near-miss question is a one-word edit of its twin by
construction. As an example of a miss: "How long have I been collecting vintage
films?" has coverage 0.60, below its twin about cameras (0.82) but well above
the 0.30 threshold.

## What this means

- **The signals are real but blunt.** They put the answerable question above
  its near-miss twin 76% to 86% of the time, but the two distributions overlap
  too much for a threshold to separate them.
- **A gate does one useful job.** At 90% retention, coverage with threshold
  0.30 silences 72% to 83% of unrelated questions, depending on the calibration
  set. If you need a cheap filter for
  questions the memory has nothing to do with, that works, at the price of
  losing about 6 points of hit rate. Calibrate the threshold on your own data;
  0.30 is a value from this benchmark, not a validated setting.
- **Near-miss questions need a different check.** They differ by a detail the
  search signals barely see. Checking whether the retrieved memory actually
  answers the question (for example with a model, or by checking that the
  question's named entities appear in the memory) is a hypothesis to test, not
  a result.

## Limitations

- One benchmark, 30 near-miss questions; the intervals are wide.
- Near-miss questions here were made by editing one detail of an answerable
  question. Real unanswerable questions may be easier or harder.
- The two stores of a pair hold the same evidence turns but different filler
  sessions, and the near-miss stores are slightly larger. The random-word
  control checks for store-level bias but is weak (see Results).
- The hit rates behind the cost figures rank by one signal, not by Memora's
  fused ranking.
- The thresholds are single-signal rules. No combination of signals, learned
  gate or model-based check was tried.
- Stores hold user turns only; vectors are MiniLM at its default.
- One near-miss question's lexical numbers were looked at while timing the code,
  before the rule was written; nothing in the rule depended on them.

## Version 1 (superseded)

Version 1 used permuted vectors (each memory embedded as a different memory)
as the vector signal's negative control. Each signal is a maximum over all
memories of a store, and permuting a store's vectors only reassigns them among
its own memories, so the maximum cannot change. The control reproduced the real
numbers exactly (22 of 29 pairs, p = 0.008, in all three seeds), and the rule's
verdict for the vector signal was "inconclusive (a control failed)". That was an
artifact of the design, not a problem with the signal. Version 2 uses the
random-word control for all three signals. No threshold or criterion was
changed, and the lexical and coverage results are the same to the printed
precision (coverage values differ in the last bit between processes, up to
2.2e-16, from summing in an order that depended on the hash seed; Memora now
sums in a fixed order). Version 1's results are in
[silence_v1_superseded](silence_v1_superseded).

## Reproduce

```bash
# Python 3.9; MiniLM run with sentence-transformers 2.7.0, torch 2.2.2
# (needs data/longmemeval/longmemeval_s_cleaned.json; see the README)
python3 -I experiments/longmemeval_silence.py            # 1 to 2 minutes
python3 -I experiments/longmemeval_silence.py --embedder sentence-transformers   # 15 to 30 minutes
```

Per-question numbers are in the JSON files next to this report.
