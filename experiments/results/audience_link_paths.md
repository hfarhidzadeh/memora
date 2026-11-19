# Link paths: graph hop, contradiction flags and topic carry-over

Paths 2 to 4 of section 8 of `docs/multi_user_memory_review.md`. Synthetic,
scripted data; lexical search only; the query is the target word alone,
`limit=10`. 200 trials (seeds 0 to 199). See the docstring of
`experiments/audience_link_paths.py` for the design and decision rule.

Arms:

- after-naive: filter after ranking, links left as they are
- after-careful: filter after ranking over every match, links trimmed to visible ends
- built: search with viewers (built)
- before: filter before ranking (emulated reference)

Each cell gives the number of trials in which P fired / in which L fired.

## Graph hop (path 2)

L = P: the viewer is shown a memory that does not contain the target.

| World | after-naive | after-careful | built | before |
| --- | --- | --- | --- | --- |
| absent | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| present | 200 / 200 | 200 / 200 | 0 / 0 | 0 / 0 |
| shared | 200 / 200 | 200 / 200 | 200 / 200 | 200 / 200 |
| unlinked | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

Trials in which L fired in `present`, by trial setting:

| Setting | Value | Trials | after-naive | after-careful | built | before |
| --- | --- | --- | --- | --- | --- | --- |
| mode | copy | 70 | 70 | 70 | 0 | 0 |
| mode | keep | 64 | 64 | 64 | 0 | 0 |
| mode | topic | 66 | 66 | 66 | 0 | 0 |
| relation | contradicts | 58 | 58 | 58 | 0 | 0 |
| relation | depends_on | 56 | 56 | 56 | 0 | 0 |
| relation | related_to | 45 | 45 | 45 | 0 | 0 |
| relation | supports | 41 | 41 | 41 | 0 | 0 |

Decision:

- Rule 1, probe works (`shared`, P in at least 95% of trials in every arm): True
- Rule 2, negative controls hold (`absent`, `unlinked`: L in no trial in any arm): True
- Rule 3, the baseline leaks (`present`, L in at least 95% of after-naive trials): True
- Prediction, not a condition of the verdict: after-careful leaks in at least 95% of `present` trials: held (200 of 200)
- Rule 4, the built arm never leaks where the before arm does not: True (0 of 800 trial-world pairs fail)
- Rule 4, the built arm's canonical observation equals the before arm's: True (0 of 800 differ)
- Sanity: hidden memories held by the before stores: 0; fewest hidden memories in a store the other arms read: 50
- **Verdict: the baseline leaks through graph; the built path closes graph**

## Contradiction flags (path 3)

P: a shown result carries a `contradicts` link. L: a shown result carries a link whose other end the viewer cannot see.

| World | after-naive | after-careful | built | before |
| --- | --- | --- | --- | --- |
| absent | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| present | 200 / 200 | 0 / 0 | 0 / 0 | 0 / 0 |
| shared | 200 / 0 | 200 / 0 | 200 / 0 | 200 / 0 |
| unlinked | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

Trials in which L fired in `present`, by trial setting:

| Setting | Value | Trials | after-naive | after-careful | built | before |
| --- | --- | --- | --- | --- | --- | --- |
| mode | copy | 75 | 75 | 0 | 0 | 0 |
| mode | keep | 56 | 56 | 0 | 0 | 0 |
| mode | topic | 69 | 69 | 0 | 0 | 0 |
| kind_filter | fact | 100 | 100 | 0 | 0 | 0 |
| kind_filter | none | 100 | 100 | 0 | 0 | 0 |
| disputant_kind | correction | 82 | 82 | 0 | 0 | 0 |
| disputant_kind | fact | 118 | 118 | 0 | 0 | 0 |

Decision:

- Rule 1, probe works (`shared`, P in at least 95% of trials in every arm): True
- Rule 2, negative controls hold (`absent`, `unlinked`: L in no trial in any arm): True
- Rule 3, the baseline leaks (`present`, L in at least 95% of after-naive trials): True
- Prediction, not a condition of the verdict: after-careful leaks in 0% of `present` trials: held (0 of 200)
- Rule 4, the built arm never leaks where the before arm does not: True (0 of 800 trial-world pairs fail)
- Rule 4, the built arm's canonical observation equals the before arm's: True (0 of 800 differ)
- Sanity: hidden memories held by the before stores: 0; fewest hidden memories in a store the other arms read: 50
- **Verdict: the baseline leaks through dispute; the built path closes dispute**

## Topic carry-over (path 4)

L = P: the viewer is shown a memory that does not contain the target.

| World | after-naive | after-careful | built | before |
| --- | --- | --- | --- | --- |
| absent | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| present | 200 / 200 | 200 / 200 | 0 / 0 | 0 / 0 |
| shared | 200 / 200 | 200 / 200 | 200 / 200 | 200 / 200 |
| unlinked | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| present-keep | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| present-copy | 200 / 200 | 200 / 200 | 0 / 0 | 0 / 0 |

Trials in which L fired in `present`, by trial setting:

| Setting | Value | Trials | after-naive | after-careful | built | before |
| --- | --- | --- | --- | --- | --- | --- |
| relation | contradicts | 52 | 52 | 52 | 0 | 0 |
| relation | depends_on | 40 | 40 | 40 | 0 | 0 |
| relation | related_to | 50 | 50 | 50 | 0 | 0 |
| relation | supports | 58 | 58 | 58 | 0 | 0 |

Decision:

- Rule 1, probe works (`shared`, P in at least 95% of trials in every arm): True
- Rule 2, negative controls hold (`absent`, `unlinked`, `present-keep`: L in no trial in any arm): True
- Rule 3, the baseline leaks (`present`, L in at least 95% of after-naive trials): True
- Prediction, not a condition of the verdict: after-careful leaks in at least 95% of `present` trials: held (200 of 200)
- Prediction, not a condition of the verdict: after-naive leaks in at least 95% of `present-copy` trials: held (200 of 200)
- Prediction, not a condition of the verdict: after-careful leaks in at least 95% of `present-copy` trials: held (200 of 200)
- Rule 4, the built arm never leaks where the before arm does not: True (0 of 1200 trial-world pairs fail)
- Rule 4, the built arm's canonical observation equals the before arm's: True (0 of 1200 differ)
- Sanity: hidden memories held by the before stores: 0; fewest hidden memories in a store the other arms read: 50
- **Verdict: the baseline leaks through topic; the built path closes topic**

Limits: the data were written alongside the code, so this shows whether each
mechanism exists and whether the built path removes it, not how often it
matters in real use. One viewer, two channels, one planted link per path,
lexical search only (no embedder), and memberships that do not change during
a trial. Rule 4 compares search results only: a fault inside one layer of
the store that another layer hides does not change any result and cannot be
seen here (the unit tests in `tests/test_audience.py` check the layers).
