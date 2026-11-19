import contextlib
import io
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dynmemory import MemoryStore
from dynmemory.cli import main

EVAL_SET = Path(__file__).resolve().parents[1] / "eval" / "synthetic_set.json"


class CliTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db = str(Path(self.temp_dir.name) / "cli.sqlite3")

    def tearDown(self):
        self.temp_dir.cleanup()

    def run_cli(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(["--db", self.db, *args])
        return code, out.getvalue(), err.getvalue()

    def add(self, content, kind="fact", *options):
        code, out, err = self.run_cli("add", content, "--type", kind, *options)
        self.assertEqual(code, 0, err)
        return json.loads(out)["id"]

    def test_invalid_input_is_an_error_message_not_a_traceback(self):
        code, out, err = self.run_cli("add", "x", "--type", "fact", "--confidence", "2")
        self.assertEqual((code, out), (1, ""))
        self.assertEqual(err.strip(), "memvane: error: confidence must be between 0 and 1")

        code, _, err = self.run_cli("link", "missing-a", "missing-b")
        self.assertEqual(code, 1)
        self.assertIn("both memories must exist", err)

        code, _, err = self.run_cli("history", "missing")
        self.assertEqual(code, 1)
        self.assertIn("memory not found", err)

    def test_supersede_history_and_search_links(self):
        old = self.add("Use the legacy endpoint.")
        new = self.add("Use the v2 endpoint.", "correction")
        conflict = self.add("Endpoints are frozen until March.")

        code, out, _ = self.run_cli("link", new, old, "--relation", "supersedes")
        self.assertEqual(code, 0)
        self.assertEqual(out.strip(), "Archived {}; superseded by {}".format(old, new))
        _, out, _ = self.run_cli("history", old)
        self.assertEqual([memory["id"] for memory in json.loads(out)], [new, old])

        self.run_cli("link", conflict, new, "--relation", "contradicts")
        _, out, _ = self.run_cli("search", "v2", "--peek")
        results = {result["memory"]["id"]: result for result in json.loads(out)}
        self.assertEqual(
            results[conflict]["links"],
            [{"source_id": conflict, "target_id": new, "relation": "contradicts"}],
        )
        self.assertEqual(results[new]["memory"]["access_count"], 0)

    def test_prune_with_type_decay_then_restore(self):
        memory = self.add("Weekly sync moved.", "episode", "--confidence", "1")
        stamp = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        MemoryStore(self.db)._set_activity(memory, created_at=stamp)

        _, out, _ = self.run_cli("prune", "--apply")
        self.assertEqual(json.loads(out)["count"], 0)
        _, out, _ = self.run_cli("prune", "--decay", "episode=14", "--apply")
        self.assertEqual(json.loads(out)["count"], 1)

        code, out, _ = self.run_cli("restore", memory)
        self.assertEqual(code, 0)
        self.assertIsNone(json.loads(out)["archived_at"])

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                main(["--db", self.db, "prune", "--decay", "meal=3"])

    def test_importance_protection_can_be_turned_off(self):
        memory = self.add("Critical but stale.", "fact", "--importance", "1", "--confidence", "0.1")
        stamp = (datetime.now(timezone.utc) - timedelta(days=400)).isoformat()
        MemoryStore(self.db)._set_activity(memory, created_at=stamp)

        _, out, _ = self.run_cli("prune", "--protect-importance", "1")
        self.assertEqual(json.loads(out)["count"], 0)
        _, out, _ = self.run_cli("prune", "--protect-importance", "none")
        self.assertEqual(
            [entry["memory"]["id"] for entry in json.loads(out)["memories"]], [memory]
        )

    def test_channels_viewers_messages_disputes_and_who_knows(self):
        for args in (
            ("channel", "join", "design", "alice"), ("channel", "join", "design", "bob"),
            ("channel", "join", "design", "carol"), ("channel", "join", "board", "bob"),
            ("admin", "add", "dana"),
        ):
            code, _, err = self.run_cli(*args)
            self.assertEqual(code, 0, err)
        _, out, _ = self.run_cli("channel", "members", "design")
        self.assertEqual(json.loads(out), ["alice", "bob", "carol"])

        code, out, err = self.run_cli("--as", "alice", "message", "add", "review is thursday", "--channel", "design")
        self.assertEqual(code, 0, err)
        message = json.loads(out)["id"]
        thursday = self.add("review is thursday", "fact", "--channel", "design", "--key", "review",
                            "--sources", message, "--speaker", "alice")
        code, _, err = self.run_cli("--as", "bob", "add", "review is friday", "--type", "fact",
                                    "--channel", "design", "--key", "review")
        self.assertEqual(code, 0, err)
        self.run_cli("--as", "bob", "add", "board launch secret", "--type", "fact", "--channel", "board")

        _, out, _ = self.run_cli("--viewers", "carol", "search", "launch", "--peek")
        self.assertEqual(json.loads(out), [])
        _, out, _ = self.run_cli("--viewers", "carol", "disputes")
        (dispute,) = json.loads(out)
        self.assertTrue(dispute["open"])
        self.assertEqual([row["status"] for row in dispute["rows"]], ["disputed", "disputed"])

        self.run_cli("--as", "carol", "message", "ack", message, "--kind", "replied")
        _, out, _ = self.run_cli("--viewers", "carol", "who-knows", thursday)
        knowledge = json.loads(out)
        self.assertEqual((knowledge["stated"], knowledge["acknowledged"]), (["alice"], ["carol"]))

        code, _, err = self.run_cli("--as", "carol", "message", "retract", message)
        self.assertEqual(code, 1)
        self.assertIn("not available for retraction", err)
        self.assertEqual(self.run_cli("--as", "alice", "message", "retract", message)[0], 0)
        _, out, _ = self.run_cli("stale")
        self.assertEqual([memory["id"] for memory in json.loads(out)], [thursday])
        _, out, _ = self.run_cli("rebuild")
        self.assertEqual(json.loads(out)[0]["reason"], "no remaining sources")
        code, _, err = self.run_cli("--viewers", "carol", "get", thursday)
        self.assertEqual(code, 1)

    def test_writing_as_a_member_checks_membership(self):
        self.run_cli("channel", "join", "design", "alice")
        code, _, err = self.run_cli("--as", "mallory", "add", "x", "--type", "fact", "--channel", "design")
        self.assertEqual(code, 1)
        self.assertIn("current channel member", err)
        code, _, err = self.run_cli("--as", "alice", "add", "x", "--type", "fact")
        self.assertEqual(code, 1)
        self.assertIn("--as needs --channel", err)

    def test_key_file_encrypts_and_tokens_come_from_a_secret_file(self):
        keys = str(Path(self.temp_dir.name) / "cli.keys")
        code, out, err = self.run_cli("--key-file", keys, "add", "zebra canary text", "--type", "fact")
        self.assertEqual(code, 0, err)
        self.assertNotIn(b"zebra canary", Path(self.db).read_bytes())
        _, out, _ = self.run_cli("--key-file", keys, "search", "zebra", "--peek")
        self.assertEqual(json.loads(out)[0]["memory"]["content"], "zebra canary text")

        secret = str(Path(self.temp_dir.name) / "secret")
        self.assertEqual(self.run_cli("new-secret", secret)[0], 0)
        self.assertEqual(Path(secret).stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.run_cli("new-secret", secret)[0], 1)  # never overwritten
        code, out, _ = self.run_cli("token", "alice", "--secret-file", secret)
        from dynmemory.auth import TokenAuthority
        authority = TokenAuthority(bytes.fromhex(Path(secret).read_text().strip()))
        self.assertEqual(authority.verify(out.strip()), "alice")

    def test_eval_reports_treatments_and_controls(self):
        code, out, _ = self.run_cli("eval", str(EVAL_SET), "--seeds", "2")
        self.assertEqual(code, 0)
        self.assertIn("lexical + graph, wrong links [control]", out)
        self.assertIn("7/7 match the labels", out)

        code, out, _ = self.run_cli(
            "--embedder", "ngram", "eval", str(EVAL_SET), "--seeds", "2", "--json"
        )
        names = [
            configuration["name"]
            for configuration in json.loads(out)["retrieval"]["configurations"]
        ]
        self.assertIn("lexical + vector, permuted vectors", names)

        code, _, err = self.run_cli("eval", str(Path(self.temp_dir.name) / "missing.json"))
        self.assertEqual(code, 1)
        self.assertIn("memvane: error:", err)


if __name__ == "__main__":
    unittest.main()
