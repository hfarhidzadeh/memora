"""Pluggable claim equivalence and update detection, with simple baselines.

Two problems the store leaves to the caller by default:

- **Equivalence.** Do two claims of one key say the same thing? The store
  compares text after trimming and case folding unless a store is created
  with ``claim_equivalence=...``. ``EmbeddingEquivalence`` treats claims whose
  embeddings are close enough as agreeing.
- **Update detection.** Does a new text give a new value for a fact already
  in memory, and which one? ``MemoryStore.detect_update(text, detector)``
  passes the best-matching visible memories to a detector, which names the
  one being updated or returns None. The caller then writes the claim under
  that memory's key, or relates it as a supersession.

The baselines here are deliberately simple and have **not** been evaluated on
real conversations. Measuring them on LongMemEval waits for the human review of
the knowledge-update labels (see ``docs/longmemeval_label_review.md``); until
then, treat any of them as a mechanism, not as a validated detector.
"""

from __future__ import annotations

import math
import re
from typing import Optional, Protocol, Sequence

from .embeddings import Embedder
from .store import Memory

_WORDS = re.compile(r"\w+", re.UNICODE)

#: Common words that carry no fact. Short on purpose; extend for your data.
STOPWORDS = frozenset(
    """a an the is are was were be been being am of to in on at for by with and
    or but not no it its this that these those i me my we our you your he she
    they them his her their what which who when where how do does did have has
    had will would can could should so than then there here as from about into
    over up down out just now very""".split()
)


class ClaimEquivalence(Protocol):
    def __call__(self, left: str, right: str) -> bool:
        ...


class UpdateDetector(Protocol):
    def __call__(self, text: str, candidates: Sequence[Memory]) -> Optional[str]:
        ...


def _exact(left: str, right: str) -> bool:
    return left.strip().casefold() == right.strip().casefold()


class ExactEquivalence:
    """The store's default: equal after trimming and case folding."""

    def __call__(self, left: str, right: str) -> bool:
        return _exact(left, right)


def _unit(values) -> Optional[list]:
    vector = [float(value) for value in values]
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0 or not math.isfinite(norm):
        return None
    return [value / norm for value in vector]


class _VectorCache:
    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self._cache: dict = {}

    def get(self, texts: Sequence[str]) -> list:
        missing = [text for text in dict.fromkeys(texts) if text not in self._cache]
        if missing:
            for text, vector in zip(missing, self.embedder(missing)):
                self._cache[text] = _unit(vector)
        return [self._cache[text] for text in texts]


def _cosine(left: Optional[list], right: Optional[list]) -> float:
    if left is None or right is None:
        return 0.0
    return sum(a * b for a, b in zip(left, right))


class EmbeddingEquivalence:
    """Claims agree if their text matches, or their cosine similarity reaches ``threshold``.

    A high threshold (default 0.9) errs towards keeping claims apart: a false
    agreement hides a real dispute, which is worse than a false dispute.
    """

    def __init__(self, embedder: Embedder, threshold: float = 0.9) -> None:
        if not -1 <= threshold <= 1:
            raise ValueError("threshold must be between -1 and 1")
        self.threshold = threshold
        self._vectors = _VectorCache(embedder)

    def __call__(self, left: str, right: str) -> bool:
        if _exact(left, right):
            return True
        first, second = self._vectors.get([left.strip(), right.strip()])
        return _cosine(first, second) >= self.threshold


def _content_words(text: str) -> set:
    return {word for word in _WORDS.findall(text.lower()) if word not in STOPWORDS}


class OverlapUpdateDetector:
    """Pick the candidate sharing the most content words, if it shares enough.

    The share is the Jaccard overlap of content words (stopwords removed). A
    new value for an old fact usually repeats the fact's words and changes the
    value, so a high overlap that is not an exact repeat suggests an update.
    An exact repeat (same words) is not an update and is not chosen.
    """

    def __init__(self, min_overlap: float = 0.5) -> None:
        if not 0 < min_overlap <= 1:
            raise ValueError("min_overlap must be in (0, 1]")
        self.min_overlap = min_overlap

    def __call__(self, text: str, candidates: Sequence[Memory]) -> Optional[str]:
        words = _content_words(text)
        best, best_score = None, 0.0
        for memory in candidates:
            other = _content_words(memory.content)
            if not words or not other or other == words:
                continue
            score = len(words & other) / len(words | other)
            if score > best_score:
                best, best_score = memory.id, score
        return best if best_score >= self.min_overlap else None


class EmbeddingUpdateDetector:
    """Pick the most similar candidate if its cosine similarity reaches ``threshold``.

    Exact repeats are not updates and are skipped.
    """

    def __init__(self, embedder: Embedder, threshold: float = 0.6) -> None:
        if not -1 <= threshold <= 1:
            raise ValueError("threshold must be between -1 and 1")
        self.threshold = threshold
        self._vectors = _VectorCache(embedder)

    def __call__(self, text: str, candidates: Sequence[Memory]) -> Optional[str]:
        pool = [memory for memory in candidates if not _exact(memory.content, text)]
        if not pool:
            return None
        vectors = self._vectors.get([text.strip()] + [memory.content.strip() for memory in pool])
        scored = [
            (_cosine(vectors[0], vector), memory.id)
            for vector, memory in zip(vectors[1:], pool)
        ]
        score, memory_id = max(scored)
        return memory_id if score >= self.threshold else None
