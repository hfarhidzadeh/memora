"""Rebuilding derived memories from the sources that remain after a retraction.

``MemoryStore.rebuild_stale(consolidator)`` calls a consolidator once for each
stale memory, with that memory (its text already cleared) and only its
remaining, unretracted source messages. The consolidator returns the new text,
or None to leave the memory in the queue. It must not see anything else: a
rebuilt memory lists exactly those sources, so whoever can read it can read
every message it was built from.

The store calls no language model. ``ExtractiveConsolidator`` is the baseline:
it joins the remaining messages' text. A model-based consolidator can be passed
instead, as long as it follows the same contract.
"""

from __future__ import annotations

from typing import Optional, Protocol, Sequence

from .store import Memory, Message


class Consolidator(Protocol):
    """Turns a stale memory's remaining sources into its new text."""

    def __call__(self, memory: Memory, sources: Sequence[Message]) -> Optional[str]:
        ...


class ExtractiveConsolidator:
    """Join the text of the remaining source messages, oldest first.

    No model and no paraphrase, so the rebuilt text can contain nothing that
    is not in a remaining source. It is a baseline and a test double, not a
    summariser.
    """

    def __init__(self, separator: str = " ") -> None:
        self.separator = separator

    def __call__(self, memory: Memory, sources: Sequence[Message]) -> Optional[str]:
        texts = [source.content.strip() for source in sources if source.content]
        return self.separator.join(texts) or None
