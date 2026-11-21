"""Build the human spot-check sheet for the Stage B other-user results.

Question for the reviewer: when another member asks a question, does the
memory that carries the answer text come from the question's owner (their
private conversation), or from somewhere the asker legitimately holds? The
automated checks say no hidden memory is ever returned; this sheet is the
human check that rule 1 of the Stage B decision rule leaves open.

Design (fixed before the sheet was built):

* Hits come from the same workspaces as the main run (full, bare8 and hybrid
  families, seeds 0 to 2), re-run with the same code. Only the recording of
  hits is new; the results of the main run are unchanged.
* Arm "built": the asker's real top 10 contains the owner's answer text.
  40 of these are sampled.
* Arm "control": the answer text appears in the *unfiltered* top 10 inside the
  owner's private channel and not in the asker's real top 10. These are known
  to be the owner's private content. 20 are sampled and shown mixed in with
  the built items, with the same fields, so the reviewer cannot tell them
  apart. They show whether the reviewer's labels can find owner content at all.
* Sampling uses ``random.Random("spotcheck-stageB")``; the order is shuffled
  with the same generator. The key (arm, channel, speaker) goes to a separate
  file that the reviewer should not open until the sheet is filled in.

Labels, one per row in the ``label`` column:

* ``owner`` - the matched text is the owner's own conversation (or clearly
  taken from it), so the asker should not have it;
* ``other`` - the text comes from somewhere else (for example the asker's own
  history, which happens to contain the same words);
* ``unsure`` - cannot tell.

Reading the filled sheet (``--score``):

* The check is valid only if at least 90% of the control rows are labelled
  ``owner``. Otherwise the labels cannot be trusted and nothing is concluded.
* If valid, every built row labelled ``owner`` is a leak candidate and must be
  traced by hand; with none, the sample gives no sign of a leak. This is a
  sample of 40 of the built hits, not proof about all of them.

Usage::

    PYTHONPATH=src python3 experiments/spot_check_sheet.py --build
    PYTHONPATH=src python3 experiments/spot_check_sheet.py --score FILLED.csv
"""

import argparse
import csv
import importlib.util
import json
import multiprocessing
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "stageb", ROOT / "experiments" / "longmemeval_shared_workspace.py")
stageb = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = stageb
spec.loader.exec_module(stageb)

OUT = ROOT / "experiments" / "results" / "spot_check"
BUILT_SAMPLE, CONTROL_SAMPLE, CONTROL_RECALL = 40, 20, 0.9
EXCERPT = 700


def excerpt(text: str, needle: str) -> str:
    at = text.lower().find(needle.lower())
    if len(text) <= EXCERPT:
        return text
    start = max(0, at - EXCERPT // 2)
    cut = text[start:start + EXCERPT]
    return ("..." if start else "") + cut + ("..." if start + EXCERPT < len(text) else "")


def collect(jobs: int) -> list:
    stageb.COLLECT_HITS = True
    stageb.install_clock()
    stageb.PARSED = stageb.load(stageb.DATA)
    eligible = sorted(stageb.PARSED)
    pool = stageb.pool_of(eligible)
    arguments = argparse.Namespace(
        seeds=3, workspaces=52, bare=13, hybrid=13, k1=0, fault=None)
    work = stageb.make_jobs(pool, eligible, arguments)
    if jobs > 1:
        with multiprocessing.get_context("fork").Pool(jobs) as workers:
            results = list(workers.imap(stageb.run_job, work, chunksize=1))
    else:
        results = [stageb.run_job(job) for job in work]
    hits = []
    for job, result in zip(work, results):
        for hit in result["hits"]:
            hit["workspace"] = "{}/{}/{}".format(job["family"], job["spec"]["seed"], job["spec"]["index"])
            hits.append(hit)
    return hits


def classify(hits: list) -> tuple:
    built = [h for h in hits if h["built"]]
    controls = [
        h for h in hits
        if not h["built"] and h["pooled"] and not h["pooled"]["visible"]
        and h["pooled"]["channel"] == stageb.private(h["owner"])
    ]
    return built, controls


def build(jobs: int) -> int:
    hits = collect(jobs)
    built, controls = classify(hits)
    rng = random.Random("spotcheck-stageB")
    chosen = [("built", h) for h in rng.sample(built, min(BUILT_SAMPLE, len(built)))]
    chosen += [("control", h) for h in rng.sample(controls, min(CONTROL_SAMPLE, len(controls)))]
    rng.shuffle(chosen)
    OUT.mkdir(parents=True, exist_ok=True)
    key = {}
    with open(OUT / "sheet.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "owner_question", "owner_answer", "owner_evidence", "matched_text",
                         "label", "notes"])
        for number, (arm, hit) in enumerate(chosen, start=1):
            row_id = "R{:02d}".format(number)
            match = hit["built"] if arm == "built" else hit["pooled"]
            evidence = " || ".join(excerpt(text, str(hit["answer"])) for text in hit["evidence"][:2])
            writer.writerow([row_id, hit["question"], hit["answer"], evidence,
                             excerpt(match["text"], str(hit["answer"]).lower()), "", ""])
            key[row_id] = {
                "arm": arm, "workspace": hit["workspace"], "item": hit["item"],
                "asker": hit["asker"], "owner": hit["owner"], "rank": match["rank"],
                "channel": match["channel"], "speaker": match["speaker"], "visible": match["visible"],
            }
    (OUT / "key.json").write_text(json.dumps(key, indent=2) + "\n")
    print("hits {}: built {}, controls available {}; sheet has {} rows".format(
        len(hits), len(built), len(controls), len(chosen)))
    print("Fill in the label column of {} and do not open key.json until you are done.".format(
        OUT / "sheet.csv"))
    return 0


def score(path: Path) -> int:
    key = json.loads((OUT / "key.json").read_text())
    with open(path, newline="") as handle:
        rows = {row["id"]: row["label"].strip().lower() for row in csv.DictReader(handle)}
    blank = [i for i in key if rows.get(i, "") not in ("owner", "other", "unsure")]
    if blank:
        print("Unlabelled or invalid rows: {}".format(", ".join(blank)))
        return 1
    controls = [i for i, v in key.items() if v["arm"] == "control"]
    built = [i for i, v in key.items() if v["arm"] == "built"]
    found = sum(rows[i] == "owner" for i in controls)
    valid = bool(controls) and found / len(controls) >= CONTROL_RECALL
    flagged = [i for i in built if rows[i] == "owner"]
    unsure = [i for i in built if rows[i] == "unsure"]
    print("Control rows labelled owner: {} of {} (need {:.0%}): {}".format(
        found, len(controls), CONTROL_RECALL, "valid" if valid else "NOT valid"))
    print("Built rows: owner {}, other {}, unsure {}".format(
        len(flagged), sum(rows[i] == "other" for i in built), len(unsure)))
    if not valid:
        print("The labels did not find the known owner content, so nothing is concluded.")
    elif flagged:
        print("Leak candidates to trace by hand: " + ", ".join(
            "{} ({})".format(i, key[i]["workspace"]) for i in flagged))
    else:
        print("No built row was labelled owner content" + (
            "; unsure rows still need a second look: " + ", ".join(unsure) if unsure else "."))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--build", action="store_true")
    group.add_argument("--score", type=Path)
    parser.add_argument("--jobs", type=int, default=3)
    arguments = parser.parse_args(argv)
    return build(arguments.jobs) if arguments.build else score(arguments.score)


if __name__ == "__main__":
    raise SystemExit(main())
