"""Signed member tokens, sessions, and the Unix-socket server that isolates the store."""

import os
import shutil
import socket
import stat
import tempfile
import threading
import unittest
from pathlib import Path

from dynmemory import MemoryStore, MemoryType
from dynmemory.auth import AuthenticationError, TokenAuthority, open_session, reader_for_tokens
from dynmemory.server import MemvaneClient, MemvaneServer, RemoteError, handle_request

FACT = MemoryType.FACT


class Clock:
    def __init__(self):
        self.now = 1_000_000.0

    def __call__(self):
        return self.now


class TokenTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.authority = TokenAuthority(b"s" * 32, clock=self.clock)

    def test_a_token_names_its_member_until_it_expires(self):
        token = self.authority.issue("alice", ttl_seconds=60)
        self.assertEqual(self.authority.verify(token), "alice")
        self.clock.now += 61
        with self.assertRaisesRegex(AuthenticationError, "expired"):
            self.authority.verify(token)

    def test_forged_tampered_and_foreign_tokens_are_refused(self):
        token = self.authority.issue("alice")
        payload, signature = token.split(".")
        other = TokenAuthority(b"t" * 32, clock=self.clock)
        for bad in (
            "garbage", "", "a.b.c", payload + ".AAAA", other.issue("alice"),
            payload.replace(payload[3], "Z" if payload[3] != "Z" else "Y") + "." + signature,
        ):
            with self.subTest(bad=bad), self.assertRaises(AuthenticationError):
                self.authority.verify(bad)
        with self.assertRaises(AuthenticationError):
            self.authority.verify(None)

    def test_a_revoked_token_is_refused_and_others_are_not(self):
        first, second = self.authority.issue("alice"), self.authority.issue("alice")
        self.authority.revoke(first)
        with self.assertRaisesRegex(AuthenticationError, "revoked"):
            self.authority.verify(first)
        self.assertEqual(self.authority.verify(second), "alice")

    def test_a_short_secret_is_rejected(self):
        with self.assertRaises(ValueError):
            TokenAuthority(b"short")
        self.assertEqual(len(TokenAuthority.generate_secret()), 32)


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory)
        self.store = MemoryStore(self.directory / "s.sqlite3", require_viewers=True, require_writer=True)
        for member in ("alice", "bob"):
            self.store.join_channel("design", member)
        self.store.join_channel("board", "bob")
        self.authority = TokenAuthority(TokenAuthority.generate_secret())

    def test_a_session_reads_and_writes_only_as_its_member(self):
        session = open_session(self.store, self.authority.issue("alice"), self.authority)
        self.assertEqual(session.member_id, "alice")
        memory = session.writer.add_memory("launch friday", FACT, channel="design")
        self.assertEqual(memory.speaker, "alice")
        with self.assertRaisesRegex(ValueError, "current channel member"):
            session.writer.add_memory("board note", FACT, channel="board")
        self.assertEqual([r.memory.id for r in session.reader.search("launch")], [memory.id])

    def test_an_audience_is_every_verified_member(self):
        open_session(self.store, self.authority.issue("bob"), self.authority).writer.add_memory(
            "board secret launch", FACT, channel="board"
        )
        reader = reader_for_tokens(
            self.store, [self.authority.issue("alice"), self.authority.issue("bob")], self.authority
        )
        self.assertEqual(reader.search("launch"), [])
        with self.assertRaises(AuthenticationError):
            reader_for_tokens(self.store, ["forged"], self.authority)
        with self.assertRaises(AuthenticationError):
            reader_for_tokens(self.store, [], self.authority)


def short_directory():
    """A temporary folder short enough for a Unix socket path."""
    base = tempfile.gettempdir()
    return Path(tempfile.mkdtemp(prefix="mv", dir=base if len(base) < 60 else "/tmp"))


@unittest.skipUnless(hasattr(socket, "AF_UNIX"), "Unix domain sockets are not available")
class ServerTests(unittest.TestCase):
    def setUp(self):
        self.directory = short_directory()
        self.addCleanup(shutil.rmtree, self.directory, True)
        self.store = MemoryStore(self.directory / "s.sqlite3")
        for member in ("alice", "bob"):
            self.store.join_channel("design", member)
        self.store.join_channel("board", "bob")
        self.authority = TokenAuthority(TokenAuthority.generate_secret())
        self.socket_path = self.directory / "s.sock"
        self.server = MemvaneServer(self.socket_path, self.store, self.authority)
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def client(self, member):
        return MemvaneClient(self.socket_path, self.authority.issue(member))

    def test_the_socket_is_owner_only(self):
        mode = stat.S_IMODE(os.stat(self.socket_path).st_mode)
        self.assertEqual(mode, 0o600)

    def test_members_write_and_read_as_themselves(self):
        alice, bob = self.client("alice"), self.client("bob")
        message = alice.add_message("launch is friday", channel="design")
        memory = alice.add_memory(
            "launch is friday", "fact", channel="design", memory_key="launch",
            source_message_ids=[message["id"]],
        )
        self.assertEqual(memory["speaker"], "alice")
        bob.add_memory("board launch secret", "fact", channel="board")
        self.assertEqual([r["memory"]["content"] for r in alice.search("launch")], ["launch is friday"])
        self.assertEqual(
            [r["memory"]["content"] for r in bob.search("launch", audience=[alice.token])],
            ["launch is friday"],
        )
        self.assertEqual(alice.call("who_knows", memory_id=memory["id"])["stated"], ["alice"])

    def test_bad_tokens_admin_calls_and_bad_requests_are_refused(self):
        with self.assertRaisesRegex(RemoteError, "authentication failed"):
            MemvaneClient(self.socket_path, "forged").search("launch")
        alice = self.client("alice")
        for operation, arguments in (
            ("join_channel", {"channel": "board", "member_id": "alice"}),
            ("add_admin", {"member_id": "alice"}),
            ("prune", {}),
        ):
            with self.subTest(operation=operation), self.assertRaisesRegex(RemoteError, "unknown operation"):
                alice.call(operation, **arguments)
        with self.assertRaisesRegex(RemoteError, "unexpected argument: viewers"):
            alice.call("search", query="launch", viewers=["bob"])
        with self.assertRaisesRegex(RemoteError, "unexpected argument: speaker"):
            alice.call("add_memory", content="x", kind="fact", channel="design", speaker="bob")
        with self.assertRaisesRegex(RemoteError, "authentication failed"):
            alice.call("search", query="launch", audience=["forged"])

    def test_a_member_cannot_retract_someone_elses_message(self):
        message = self.client("bob").add_message("board plan", channel="design")
        with self.assertRaisesRegex(RemoteError, "not available for retraction"):
            self.client("alice").retract_message(message["id"])

    def test_handle_request_never_raises(self):
        for request in (None, [], {"token": 5}, {"token": self.authority.issue("alice"), "op": "search", "args": []}):
            with self.subTest(request=request):
                response = handle_request(self.store, self.authority, request)
                self.assertFalse(response["ok"])

    def test_an_existing_socket_path_is_not_clobbered_and_long_paths_are_refused(self):
        with self.assertRaises(FileExistsError):
            MemvaneServer(self.socket_path, self.store, self.authority)
        with self.assertRaisesRegex(ValueError, "too long"):
            MemvaneServer(self.directory / ("x" * 120), self.store, self.authority)


if __name__ == "__main__":
    unittest.main()
