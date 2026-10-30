"""Embedding backends for vector retrieval.

An embedder is any object with:

- ``name``: stored with each vector, so changing models re-embeds memories;
- ``min_similarity``: the cosine similarity a memory needs to count as a
  vector match (the right value depends on the model);
- ``__call__(texts)``: returns one vector per text.

Both backends here run locally; memory text never leaves the machine.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Optional, Protocol, Sequence


class Embedder(Protocol):
    name: str
    min_similarity: float

    def __call__(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        ...


def _validate_similarity(value: float) -> None:
    if not math.isfinite(value) or not -1 <= value <= 1:
        raise ValueError("min_similarity must be between -1 and 1")


class NGramHashEmbedder:
    """Character n-gram feature hashing, with no dependencies.

    Matches spelling and word-form variants ("invoice" and "invoices") that
    whole-word lexical search misses. It has no notion of meaning, so treat it
    as a baseline, not a semantic model. On the bundled synthetic set, about
    2% of unrelated query-memory pairs exceed a cosine of 0.15 and under 1%
    reach the default ``min_similarity`` of 0.25; tune it on your own data.
    """

    _WORD_PATTERN = re.compile(r"\w+", re.UNICODE)

    def __init__(
        self, dimensions: int = 512, n: int = 3, min_similarity: float = 0.25
    ) -> None:
        if dimensions < 1 or n < 1:
            raise ValueError("dimensions and n must be positive")
        _validate_similarity(min_similarity)
        self.dimensions = dimensions
        self.n = n
        self.min_similarity = min_similarity
        self.name = "ngram-hash:n={}:d={}".format(n, dimensions)

    def __call__(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for word in self._WORD_PATTERN.findall(text.lower()):
            padded = " {} ".format(word)
            for start in range(max(1, len(padded) - self.n + 1)):
                digest = hashlib.blake2b(
                    padded[start:start + self.n].encode("utf-8"), digest_size=8
                ).digest()
                value = int.from_bytes(digest, "little")
                # The sign bit keeps hash collisions from always adding up.
                vector[value % self.dimensions] += 1.0 if value >> 63 else -1.0
        return vector


class SentenceTransformerEmbedder:
    """A local sentence-transformers model.

    Needs ``pip install sentence-transformers``. The weights are downloaded
    from Hugging Face on first use unless already cached; set
    ``HF_HUB_OFFLINE=1`` to forbid network access. Texts are embedded on this
    machine either way.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        min_similarity: float = 0.3,
        device: Optional[str] = None,
    ) -> None:
        _validate_similarity(min_similarity)
        self.model_name = model_name
        self.min_similarity = min_similarity
        self.name = "sentence-transformers:{}".format(model_name)
        self._device = device
        self._model = None

    def __call__(self, texts: Sequence[str]) -> list[list[float]]:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:
                raise ImportError(
                    "the sentence-transformers backend needs the "
                    "sentence-transformers package: "
                    "pip install 'memora-agent-memory[embeddings]'"
                ) from error
            self._model = SentenceTransformer(self.model_name, device=self._device)
        return self._model.encode(
            list(texts), normalize_embeddings=True, convert_to_numpy=True
        ).tolist()
