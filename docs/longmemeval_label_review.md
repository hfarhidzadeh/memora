# LongMemEval label review — 2026-11-19

**All 70 labels reviewed; the existing results remain provisional.** This is
an AI review, not human adjudication. It recommends three category changes,
identifies eight additional evidence/conversion issues, and retains two labels
with explicit caveats. The other 57 categories are supported by the inspected
question and evidence, including the existing exclusion.

The original category/note entries and numerical results are preserved for
reproducibility. Recommendations below have **not** been applied to the original
labels, generated suite, or metrics. Result files now carry a provisional status.

## Review basis

Read every labelled question, benchmark answer, marked user evidence turn,
and its session/question timestamp in `longmemeval_s_cleaned.json`. Checked
surrounding user turns for ambiguous cases and cross-checked relevant oracle
sessions, especially the reversed Converse dates. This is not an exhaustive
reannotation of all unmarked turns or distractor histories.

Source SHA-256: `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`.
The [machine-readable ledger](../eval/longmemeval_knowledge_update_label_review.json)
records all 70 decisions, question/answer text, evidence IDs and dates, and the
hash of the original labels file before review metadata was added.

“Confirmed” below means the category fits the inspected evidence. It does not
validate the converter's whole-turn archival policy, all evidence annotations,
or downstream answer accuracy.

## Findings that affect interpretation

- **Three recommended category changes:** `2133c1b5` current → exclude (future
  evidence); `cf22b7bf` current → context (different measurement intervals);
  `5c40ec5b` current → context (no exact older count). These alone would change
  the labelled primary group from 50 to 47, context from 1 to 3, and exclusions
  from 1 to 2. This is a proposed cohort, not a recomputed result.
- **Answer time and supersession are different labels.** Asking for an older
  event does not make that event stale. The converter currently emits updates
  for every category except context/exclude, including all three latest-event
  questions and the two independent-event previous questions. Those edges
  should not be ground truth for a detector of obsolete facts.
- **Additions can require preserving the earlier premise.** The coin question
  needs 37 plus one; the later increment cannot replace the earlier total.
- **Marked evidence can be incomplete or premature.** The coffee question's
  marked turn proposes an increase; a later unmarked user turn confirms it.
  The bike question's later evidence repeats information useful to answering
  the past-tense question, and its gravel/hybrid wording is inconsistent.
- **MRR on “both” is not complete-answer retrieval.** `query_rows` in
  `src/dynmemory/evaluation.py` awards reciprocal-rank credit for the first
  relevant turn. It does not require both premises. Report all-required-evidence
  recall/coverage for these questions before interpreting answerability.
- **Whole turns are not atomic facts.** Even confirmed current questions can
  have other useful facts in the older turn. Archiving the entire turn is an
  experimental intervention, not a validated rule for forgetting everything in it.

The existing “oracle” should be described as a benchmark-derived update
heuristic, not perfect update detection or a proven upper bound. The historical
50-current and 15-previous/both figures must not be promoted as settled findings.

## Disposition

Keep the historical labels and results as versioned evidence. Prepare a new
conversion version with separate answer-evidence and update-relation annotations;
resolve the flagged rows below, including the two interpretation caveats. Use
explicit no-supersession/addition relations where appropriate, repair the coffee
evidence, and exclude the future-evidence question unless its source is corrected.
Then regenerate the suite and rerun lexical and semantic experiments with the
new source/label hashes. Report the resulting cohort changes and sensitivity to
ambiguous rows. Do not compare fresh metrics to historical numbers as if the
cohorts and ground truth were unchanged.

## Per-question review

Evidence IDs and dates for each row are in the linked ledger. Category names
refer to the existing schema; additional relation/evidence issues cannot be
expressed by changing a category alone.

| Question ID | Original → recommended category | Review | Rationale |
| --- | --- | --- | --- |
| `6a1eabeb` | current | confirmed | 5K personal best 27:12 -> 25:50 |
| `6aeb4375` | current | confirmed | Korean restaurants tried 3 -> 4 |
| `830ce83f` | current | confirmed | Rachel moved to Chicago -> back to the suburbs (two older turns) |
| `852ce960` | current | confirmed | Wells Fargo pre-approval $350,000 -> $400,000 |
| `945e3d21` | current | confirmed | yoga twice -> three times a week |
| `d7c942c3` | current | confirmed | mom on a paper list -> same app as me |
| `71315a70` | current | confirmed | ocean sculpture 5-6 -> 10-12 hours |
| `89941a93` | current | confirmed | bikes owned 3 -> 4 |
| `ce6d2d27` | current | confirmed | cocktail class Thursday -> Friday |
| `a1eacc2a` | current | confirmed | short stories 4 -> 7 |
| `184da446` | current | confirmed | book page 200 -> 220 |
| `4d6b87c8` | current | confirmed | to-watch list 20 -> 25 titles |
| `0f05491a` | current | confirmed | Starbucks gold level 125 -> 120 stars (correction) |
| `08e075c7` | current | confirmed | Fitbit 6 -> 9 months |
| `f9e8c073` | current | confirmed | bereavement sessions 3 -> 5 |
| `2698e78f` | current | confirmed | therapist every two weeks -> every week |
| `b6019101` | current | confirmed | MCU films 4 -> 5 |
| `45dc21b6` | current | confirmed | Emma's recipes tried 2 -> 3 |
| `e493bb7c` | current | confirmed | painting above the living-room sofa -> above the bed |
| `72e3ee87` | current | confirmed | Crash Course science episode 10 -> 50 |
| `01493427` | current | confirmed | postcards 17 -> 25 |
| `6a27ffc2` | current | confirmed | Corey Schafer videos 20 -> 30 |
| `2133c1b5` | current → exclude | revise category | The answer-bearing session is 2023/10/15 20:50, after the question at 08:39 that day. Exclude from a chronological evaluation unless source timestamps are repaired. The original note already notices this issue but still includes it in the primary group. |
| `18bc8abd` | current | confirmed | BBQ sauce Sweet Baby Ray's -> Kansas City Masterpiece |
| `db467c8c` | current | confirmed | parents staying 6 -> 9 months |
| `7a87bd0c` | current | confirmed | tidying routine 3 -> 4 weeks |
| `e61a7584` | current | confirmed | cat Luna 6 -> 9 months |
| `1cea1afa` | current | confirmed | Instagram followers 500 -> 600 |
| `ed4ddc30` | current | confirmed | eggs 30 -> 20 dozen |
| `8fb83627` | current | confirmed | National Geographic issues 3 -> 5 (two older turns) |
| `b01defab` | current | confirmed | The Nightingale put down -> finished |
| `0e4e4c46` | current | confirmed | Ticket to Ride best 124 -> 132 |
| `4b24c848` | current | confirmed | H&M tops 3 -> 5 |
| `603deb26` | current | confirmed | Negroni attempts 5 -> 10 |
| `59524333` | current | confirmed | gym time 7:00 pm -> 6:00 pm |
| `5831f84d` | current | confirmed | Crash Course videos 10 -> 15 |
| `affe2881` | current | confirmed | bird species 27 -> 32 |
| `cc5ded98` | current | confirmed | coding 1 -> 2 hours a day |
| `7401057b` | current | confirmed | Hilton free nights 1 -> 2 |
| `cf22b7bf` | current → context | revise category | Five pounds in the past month and ten pounds since starting the gym three months ago measure different intervals. The earlier turn does not give an older total for the asked interval, so treating it as stale is unsupported. |
| `a2f3aa27` | current | confirmed with caveat | Current is supported, but the newer turn says the user thinks the count is close to 1300. The benchmark answer rounds this to 1300. Preserve the approximation; do not treat it as a verified exact count. |
| `c7dc5443` | current | confirmed | volleyball record 3-2 -> 5-2 |
| `06db6396` | current | confirmed | painting projects 4 -> 5 |
| `3ba21379` | current | confirmed | model in progress Mustang -> F-150 |
| `dad224aa` | current | confirmed | Saturday wake-up 8:30 -> 7:30 am |
| `ba61f0b9` | current | confirmed | women on Rachel's team 5 -> 6 |
| `42ec0761` | current | confirmed with caveat | Current is defensible for usable-tool availability: the user first needs to buy a screwdriver and later has a spare. It does not establish that the lost screwdriver was found or that no spare existed earlier. Keep only with that availability interpretation; a strict spare-ownership interpretation would be context. |
| `5c40ec5b` | current → context | revise category | The earlier turn records meeting Alex at a festival, without an exact cumulative meetup count. The later turn states two meetups and repeats the festival meeting. Do not infer an earlier count of exactly one or archive the still-true meeting event. |
| `26bdc477` | current | confirmed | Canon trips 3 -> 5 |
| `07741c45` | current | confirmed | old sneakers under the bed -> shoe rack in the closet |
| `07741c44` | previous | confirmed | asks where the sneakers were initially kept (under the bed) |
| `50635ada` | previous | confirmed | asks for the previous flyer status (Premier Silver) |
| `e66b632c` | previous | confirmed | asks for the previous 5K best (27:45) |
| `0ddfec37` | previous | confirmed | asks for the count in the first three months (15) |
| `dfde3500` | previous | confirmed | asks the day with the previous tutor Juan (Wednesday) |
| `9bbe84a2` | previous | confirmed | asks for the previous Apex Legends goal (level 100) |
| `10e09553` | previous | conversion or evidence issue | Keep previous as the question category, but remove the supersession edge: seven bass on July 10 and nine on July 22 describe distinct trips. Both remain true; their difference is not update-detection ground truth. |
| `0977f2af` | previous | conversion or evidence issue | Keep previous as the intended category, but two appliance purchases do not invalidate one another. The Air Fryer was obtained yesterday relative to a same-day later session, so mention order alone also does not establish purchase order. No supersession; flag the event-order ambiguity. |
| `89941a94` | previous | conversion or evidence issue | The question says gravel bike, while the source says hybrid bike. It asks about prior ownership, but the newer marked turn also lists the road bike alongside the two old bikes and the newly bought hybrid. Older-only relevance can penalize useful answer evidence; adjudicate wording and evidence before scoring harm. |
| `031748ae` | both | confirmed | asks for both team sizes (4, then 5 engineers) |
| `6071bd76` | both | confirmed | asks the direction of change (6 -> 5 oz per tablespoon) |
| `c4ea545c` | both | confirmed | asks whether the gym is more frequent than before (3 -> 4 a week) |
| `f685340e` | both | confirmed | asks for both tennis frequencies (weekly -> every other week) |
| `c6853660` | both | conversion or evidence issue | Both is the right category, but marked newer turn answer_626e93c4_2:0 only proposes changing to two cups. Unmarked user turn answer_626e93c4_2:6 explicitly confirms the increase. Use the confirmation as evidence/update point, rather than scoring a proposal as an enacted change. |
| `69fee5aa` | both | conversion or evidence issue | Both is correct: 37 plus one gives 38. The later turn supplies an increment, not a replacement total. Archiving the earlier 37 destroys a necessary premise. Preserve both and model accumulation separately from supersession. |
| `9ea5eabc` | latest_event | conversion or evidence issue | Latest_event is correct under the benchmark reading. Hawaii and Paris are separate trips, so remove the supersession edge. Newer mention dates are only a proxy for event order; neither event is false. |
| `41698283` | latest_event | conversion or evidence issue | Latest_event is correct under the benchmark reading. The 50mm and 70-200mm lens purchases can both remain true. No supersession; the newer mention supplies the intended latest purchase. |
| `7e974930` | latest_event | conversion or evidence issue | Latest_event is correct under the benchmark reading. Market revenues on separate visits remain true, so no supersession. Also flag that the April 11 session lists May sales as if already completed; do not claim reliable event chronology from session timestamps alone. |
| `5a4f22c0` | context | confirmed | the older turn mentions Rachel but no company |
| `618f13b2` | exclude | confirmed | Retain exclude. Cleaned timestamps put six wears at May 30 17:45 before four wears at 22:16. The oracle file orders the four-wear session before the six-wear session. Do not silently repair the cleaned data by session-name order. |
