# Side channel in search scores (H2)

Synthetic, scripted data; lexical search only. 200 trials (seeds 0 to 199).
D = coverage("anchor w") - coverage("anchor z"), read from the results
the viewer is shown. See the docstring of
`experiments/audience_side_channel.py` for the design and decision rule.

| World | Arm | D > 0 | D = 0 | D < 0 | Mean D |
| --- | --- | --- | --- | --- | --- |
| none | filter after ranking | 0 | 200 | 0 | 0.0000 |
| none | filter before ranking (emulated) | 0 | 200 | 0 | 0.0000 |
| none | search with viewers (built) | 0 | 200 | 0 | 0.0000 |
| absent | filter after ranking | 0 | 200 | 0 | 0.0000 |
| absent | filter before ranking (emulated) | 0 | 200 | 0 | 0.0000 |
| absent | search with viewers (built) | 0 | 200 | 0 | 0.0000 |
| present-1 | filter after ranking | 200 | 0 | 0 | 0.0460 |
| present-1 | filter before ranking (emulated) | 0 | 200 | 0 | 0.0000 |
| present-1 | search with viewers (built) | 0 | 200 | 0 | 0.0000 |
| present-5 | filter after ranking | 200 | 0 | 0 | 0.1181 |
| present-5 | filter before ranking (emulated) | 0 | 200 | 0 | 0.0000 |
| present-5 | search with viewers (built) | 0 | 200 | 0 | 0.0000 |
| present-20 | filter after ranking | 200 | 0 | 0 | 0.2224 |
| present-20 | filter before ranking (emulated) | 0 | 200 | 0 | 0.0000 |
| present-20 | search with viewers (built) | 0 | 200 | 0 | 0.0000 |
| shared | filter after ranking | 200 | 0 | 0 | 0.1915 |
| shared | filter before ranking (emulated) | 200 | 0 | 0 | 0.1816 |
| shared | search with viewers (built) | 200 | 0 | 0 | 0.1816 |

Raw score shift (best lexical score for "anchor z", `absent` against `none`):

- filter after ranking: changed in 200 of 200 trials, mean relative shift +1.17%
- filter before ranking (emulated): changed in 0 of 200 trials
- search with viewers (built): changed in 0 of 200 trials

Decision:

- Rule 1, probe works (shared): True
- Rule 2, negative control holds (absent, none): True
- Rule 3, filter-after leaks for every k: True
- Rule 3, filter-before is clean for every k: True
- Rule 3, the built arm is clean for every k: True
- Rule 4, the built arm equals filter-before in every trial and world: True (0 of 1200 comparisons differ)
- **Verdict: H2 supported; the built fix closes the channel**

Limits: the data were written alongside the code, so this shows that the
mechanism exists and that the fix removes it, not how often it matters in
real use. Only the lexical path was tested here (no embedder, no links);
the link paths are covered by unit tests in `tests/test_audience.py`.
