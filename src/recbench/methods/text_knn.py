"""Text-embedding kNN: recommend items whose descriptions are similar to the user's recent items.

1. Each item's text (title, description, and category tokens) is turned into a vector by a pretrained
   sentence-transformer (default: all-MiniLM-L6-v2). Vectors are cached per split.
2. A user's profile is the recency-weighted average of their latest `text_profile_window` items' vectors.
3. score(u, i) = cosine(profile_u, vector_i).

Because it only needs an item's text, it can score brand-new items that nobody has interacted with yet
(scores_cold_items=True). It is a content baseline, not a language model reasoning about users.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any

import numpy as np

from recbench.data import HistoryBatch, TrainView
from recbench.methods._explain import embedding_explanations
from recbench.protocol import NEG_INF, MethodSpec, Recommender, Task, Unsupported
from recbench.registry import register_method

DEFAULT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


def item_texts(data: TrainView) -> list[str]:
    """Title/description plus category tokens, one string per item index (index 0 = padding = '')."""
    out = []
    for text, category in zip(data.item_text, data.item_category):
        tokens = ", ".join(t for t in str(category).split("|") if t)
        out.append(" ".join(part for part in (str(text).strip(), tokens) if part))
    out[0] = ""
    return out


def encode_texts(texts: list[str], model_name: str, device: str, batch_size: int = 256) -> np.ndarray:
    """Unit-length sentence embeddings; empty texts get a zero vector. Tests replace this function."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise Unsupported("sentence-transformers is not installed (it comes with the cpu/gpu extras)") from exc
    model = SentenceTransformer(model_name, device=device)
    present = [i for i, t in enumerate(texts) if t]
    vectors = np.zeros((len(texts), model.get_sentence_embedding_dimension()), dtype=np.float32)
    if present:
        encoded = model.encode([texts[i] for i in present], batch_size=batch_size, normalize_embeddings=True,
                               convert_to_numpy=True, show_progress_bar=False)
        vectors[present] = encoded.astype(np.float32)
    return vectors


def vectors_path(data: TrainView, model_name: str) -> "os.PathLike":
    """Where the split's item vectors for `model_name` are cached (the key covers every item's text)."""
    key = hashlib.sha256(("\n".join(item_texts(data)) + "|" + model_name).encode()).hexdigest()[:16]
    return data.cache_dir / f"text_vectors_{key}.npy"


def cached_item_vectors(data: TrainView, model_name: str, device: str) -> np.ndarray:
    path = vectors_path(data, model_name)
    if path.exists():
        return np.load(path)
    vectors = encode_texts(item_texts(data), model_name, device)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npy")  # parallel jobs may encode the same split
    np.save(tmp, vectors)
    os.replace(tmp, path)
    return vectors


@register_method
class TextKNN(Recommender):
    spec = MethodSpec(
        name="text_knn",
        tasks={Task.topn, Task.sequential, Task.similar_items},
        uses_history=True,
        scores_cold_items=True,
        requires_side_features=True,
        needs_torch=True,
        upstream="sentence-transformers (pretrained encoder) + in-repo cosine kNN",
        cost_band="low",
        deterministic=True,
    )

    def fit(self, data: TrainView, cfg: dict[str, Any]) -> None:
        from recbench.methods._torch import resolve_device

        self.bind(data)
        self.data = data
        device = resolve_device(cfg).type
        self.vectors = cached_item_vectors(data, str(cfg.get("text_encoder") or DEFAULT_ENCODER), device)
        if not np.any(self.vectors[1:]):
            raise Unsupported("no item has text or categories to embed")
        self.window = int(cfg.get("text_profile_window", 20))
        self.decay = float(cfg.get("text_position_decay", 0.9))
        self.fit_info = {"embedding_dim": int(self.vectors.shape[1]), "items_with_text": int((np.abs(self.vectors[1:]).sum(1) > 0).sum())}

    def score_users(self, users: np.ndarray, hist: HistoryBatch) -> np.ndarray:
        recent = self.data.history_batch(users, self.window).items  # right-aligned: newest item last
        position = np.arange(self.window)[::-1]  # 0 for the newest column
        weights = np.power(self.decay, position)[None, :] * (recent > 0)
        profile = np.einsum("bw,bwd->bd", weights, self.vectors[recent])
        norm = np.linalg.norm(profile, axis=1, keepdims=True)
        profile = profile / np.where(norm > 0, norm, 1.0)
        out = (profile @ self.vectors.T).astype(np.float32)
        out[:, 0] = NEG_INF
        return out

    def item_embeddings(self) -> np.ndarray:
        return self.vectors

    def explain(self, users, items, hist):
        return embedding_explanations(self.vectors, users, items, hist, self.item_ids, "Text similarity")
