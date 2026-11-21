# Spot-check outcome (2026-11-21)

Labels: `sheet_labeled.csv` (by the researcher, unedited). Key: `key.json`.

Scoring: controls 18 of 20 labelled owner (valid); built rows 34 other, 1 owner, 5 unsure.

Tracing (by the model, against the plan of each workspace) of the six rows
that needed it:

| Row | Workspace | Asker, owner | Matched memory | Verdict |
| --- | --- | --- | --- | --- |
| R55 | full/0/36 | user05, user00 | p-user05 (asker's own) | not a leak |
| R11 | full/0/29 | user07, user04 | p-user07 (asker's own) | not a leak |
| R34 | bare8/0/5 | user04, user06 | p-user04 (asker's own) | not a leak |
| R27 | full/0/17 | user01, user00 | general (shared, member) | not a leak |
| R30 | full/2/5 | user00, user06 | team (member, own text) | not a leak |
| R59 | full/2/5 | user01, user06 | team (member) | not a leak |

Each matched text exists once in its workspace and is not a copy of the
owner's evidence. Result: 0 of 40 sampled hits were leaks (a sample of the 120).
