
"""Semantic encoder.

Uses MiniLM (sentence-transformers) when installed; otherwise uses a
deterministic hashed character/word n-gram encoder so the system can
still run on machines without torch.

Encoder.kind is 'minilm' or 'hash'.
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import numpy as np

from configs.settings import settings


DIM = 384
BATCH_SIZE = 32


class _HashEncoder:
    """Deterministic character and word n-gram encoder."""

    kind = "hash"
    dim = DIM

    @staticmethod
    def _feats(t: str):
        t = re.sub(r"\s+", " ", t.lower()).strip()
        words = re.findall(r"[a-z0-9]+", t)

        for w in words:
            yield "w:" + w, 2.0

        padded = f" {t} "

        for n in (3, 4, 5):
            for i in range(len(padded) - n + 1):
                yield f"c{n}:{padded[i:i + n]}", 1.0

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), DIM), dtype=np.float32)

        for r, t in enumerate(texts):
            for feature, weight in self._feats(t):
                h = int.from_bytes(
                    hashlib.blake2b(
                        feature.encode(),
                        digest_size=8,
                    ).digest(),
                    "little",
                )

                out[r, h % DIM] += (
                    weight if (h >> 63) & 1 else -weight
                )

        norms = np.linalg.norm(out, axis=1, keepdims=True)

        return out / np.maximum(norms, 1e-9)


class Encoder:
    """Shared semantic encoder with an in-memory embedding cache."""

    _inst = None

    def __init__(self, force_hash: bool = False):
        self.model = None
        self.kind = "hash"

        # Initialize fallback encoder regardless of selected backend.
        self._h = _HashEncoder()

        # Cache maps the exact input text to its normalized embedding.
        self._embedding_cache: dict[str, np.ndarray] = {}

        if not force_hash:
            original_path = sys.path.copy()

            try:
                project_root = Path(__file__).resolve().parent.parent

                # Avoid project-local modules shadowing installed packages.
                sys.path = [
                    p
                    for p in sys.path
                    if str(Path(p or ".").resolve())
                    != str(project_root)
                ]

                from sentence_transformers import SentenceTransformer

                self.model = SentenceTransformer(settings.embed_model)
                self.kind = "minilm"

            except Exception as e:
                print(
                    f"MiniLM loading failed: "
                    f"{type(e).__name__}: {e}"
                )
                self.model = None
                self.kind = "hash"

            finally:
                sys.path = original_path

        if self.model is not None:
            dimension = self.model.get_sentence_embedding_dimension()
            self.dim = dimension or DIM
        else:
            self.dim = DIM

    @classmethod
    def get(cls) -> "Encoder":
        """Return the shared encoder instance."""
        if cls._inst is None:
            cls._inst = cls()

        return cls._inst

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts in batches and reuse cached embeddings."""

        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        # Hash fallback does not need the MiniLM cache.
        if self.model is None:
            return self._h.encode(texts)

        # Find unique texts that have not yet been encoded.
        missing = list(
            dict.fromkeys(
                text
                for text in texts
                if text not in self._embedding_cache
            )
        )

        # Encode only uncached texts, in manageable batches.
        for start in range(0, len(missing), BATCH_SIZE):
            batch = missing[start:start + BATCH_SIZE]

            vectors = self.model.encode(
                batch,
                batch_size=BATCH_SIZE,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

            vectors = np.asarray(vectors, dtype=np.float32)

            for text, vector in zip(batch, vectors):
                self._embedding_cache[text] = vector.copy()

        # Preserve input order and duplicate entries.
        return np.stack(
            [self._embedding_cache[text] for text in texts]
        ).astype(np.float32, copy=False)

    def similarity(self, a: str, b: str) -> float:
        """Calculate cosine similarity between two encoded texts."""
        va, vb = self.encode([a, b])

        # Embeddings are normalized, so their dot product is cosine similarity.
        return float(np.dot(va, vb))