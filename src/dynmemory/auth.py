"""Write authentication: signed member tokens and the sessions they open.

The store itself trusts whoever calls it with a member ID. ``TokenAuthority``
closes that gap for callers that cannot be trusted: it issues tokens signed
with a secret (HMAC-SHA256, standard library only), and ``open_session`` turns
a valid token into a reader and a writer bound to that member, so the caller
never chooses the identity itself.

A token carries the member ID, an expiry time and a random nonce, and can be
revoked before it expires. Revocations are kept in memory by the authority
instance; persist them yourself if the authority is restarted. Keep the secret
out of the database and out of the hands of whoever presents tokens.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass
from typing import Callable, Iterable

from .store import AudienceReader, ChannelWriter, MemoryStore


class AuthenticationError(ValueError):
    """A token was malformed, forged, expired or revoked."""


def _encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _decode(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


class TokenAuthority:
    """Issue and verify signed member tokens."""

    MIN_SECRET_BYTES = 32

    def __init__(self, secret: bytes, *, clock: Callable[[], float] = time.time) -> None:
        if not isinstance(secret, (bytes, bytearray)) or len(secret) < self.MIN_SECRET_BYTES:
            raise ValueError(
                "secret must be at least {} random bytes".format(self.MIN_SECRET_BYTES)
            )
        self._secret = bytes(secret)
        self._clock = clock
        self._revoked: set = set()

    @staticmethod
    def generate_secret() -> bytes:
        return secrets.token_bytes(TokenAuthority.MIN_SECRET_BYTES)

    def _sign(self, payload: bytes) -> bytes:
        return hmac.new(self._secret, payload, hashlib.sha256).digest()

    def issue(self, member_id: str, *, ttl_seconds: float = 3600) -> str:
        if not isinstance(member_id, str) or not member_id.strip():
            raise ValueError("member_id must not be empty")
        if not ttl_seconds > 0:
            raise ValueError("ttl_seconds must be positive")
        payload = json.dumps(
            {
                "m": member_id.strip(),
                "e": self._clock() + ttl_seconds,
                "n": secrets.token_hex(16),
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return _encode(payload) + "." + _encode(self._sign(payload))

    def _claims(self, token: str) -> dict:
        try:
            payload_text, signature_text = token.split(".")
            payload, signature = _decode(payload_text), _decode(signature_text)
        except (AttributeError, ValueError):
            raise AuthenticationError("invalid token") from None
        if not hmac.compare_digest(signature, self._sign(payload)):
            raise AuthenticationError("invalid token")
        try:
            claims = json.loads(payload.decode("utf-8"))
        except ValueError:
            raise AuthenticationError("invalid token") from None
        if not isinstance(claims, dict) or not isinstance(claims.get("m"), str):
            raise AuthenticationError("invalid token")
        return claims

    def verify(self, token: str) -> str:
        """The member a valid token names; raises AuthenticationError otherwise."""
        claims = self._claims(token)
        if not isinstance(claims.get("e"), (int, float)) or claims["e"] <= self._clock():
            raise AuthenticationError("token expired")
        if claims.get("n") in self._revoked:
            raise AuthenticationError("token revoked")
        return claims["m"]

    def revoke(self, token: str) -> None:
        """Refuse this token from now on (for this authority instance)."""
        self._revoked.add(self._claims(token).get("n"))


@dataclass(frozen=True)
class Session:
    """A verified member's reader (as the only viewer) and writer."""

    member_id: str
    reader: AudienceReader
    writer: ChannelWriter


def open_session(store: MemoryStore, token: str, authority: TokenAuthority) -> Session:
    member_id = authority.verify(token)
    return Session(member_id, store.for_viewers(member_id), store.for_writer(member_id))


def reader_for_tokens(
    store: MemoryStore, tokens: Iterable[str], authority: TokenAuthority
) -> AudienceReader:
    """A reader whose audience is every member a valid token names."""
    members = [authority.verify(token) for token in tokens]
    if not members:
        raise AuthenticationError("an audience needs at least one token")
    return store.for_viewers(members)
