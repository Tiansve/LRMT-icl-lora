"""Character n-gram BM25 retrieval for In-Context Learning (ICL).

Rules from the work plan (Stage 4):
1. Index the source side of the pool of size N; query with the test source sentence.
2. Tokenize all languages uniformly into character 3-grams so scripts without word
   spaces (e.g. Tibetan for Dzongkha) are handled identically.
3. Retrieve top k examples; when k >= N, use the entire pool ordered by similarity.
4. Order the selected examples in ascending similarity so the most similar example
   is placed last (closest to the test query).
5. If fewer than min(k, N) candidates have score > 0, fill the remaining slots
   deterministically at random using the pool's seed.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from lrmt.data import normalize


@dataclass(frozen=True)
class RetrievedExample:
    pool_index: int
    sent_id: int
    src_text: str
    tgt_text: str
    score: float


def char_ngrams(text: str, n: int = 3) -> list[str]:
    """Split NFC-normalized text into overlapping character n-grams."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    cleaned = " ".join(normalize(text).split())
    if not cleaned:
        return []
    if len(cleaned) < n:
        return [cleaned]
    return [cleaned[i : i + n] for i in range(len(cleaned) - n + 1)]


class BM25Retriever:
    """Character n-gram BM25 index over a parallel data pool.

    Precomputes an inverted index of BM25Okapi term-document weights so that
    querying 1,012 test sentences takes milliseconds while producing the exact
    same scores as `rank_bm25.BM25Okapi.get_scores`.
    """

    def __init__(
        self,
        pool_ids: Sequence[int],
        src_texts: Sequence[str],
        tgt_texts: Sequence[str],
        *,
        ngram_size: int = 3,
    ) -> None:
        if not (len(pool_ids) == len(src_texts) == len(tgt_texts)):
            raise ValueError("pool_ids, src_texts and tgt_texts must have the same length")
        if not pool_ids:
            raise ValueError("cannot build BM25Retriever on an empty pool")

        self.pool_ids = tuple(int(i) for i in pool_ids)
        self.src_texts = tuple(src_texts)
        self.tgt_texts = tuple(tgt_texts)
        self.ngram_size = ngram_size

        tokenized = [char_ngrams(text, n=ngram_size) for text in self.src_texts]
        self._bm25 = BM25Okapi(tokenized)

        # Build sparse inverted index of exact BM25Okapi per-term document score contributions.
        k1 = self._bm25.k1
        b = self._bm25.b
        avgdl = self._bm25.avgdl
        doc_len = np.asarray(self._bm25.doc_len, dtype=np.float64)
        norm_denom = k1 * (1.0 - b + b * doc_len / avgdl)

        postings: dict[str, tuple[list[int], list[float]]] = {}
        for doc_idx, freqs in enumerate(self._bm25.doc_freqs):
            dl_norm = float(norm_denom[doc_idx])
            for term, tf in freqs.items():
                idf = float(self._bm25.idf.get(term, 0.0))
                if idf == 0.0:
                    continue
                weight = idf * (tf * (k1 + 1.0) / (tf + dl_norm))
                if term not in postings:
                    postings[term] = ([], [])
                postings[term][0].append(doc_idx)
                postings[term][1].append(weight)

        self._postings: dict[str, tuple[np.ndarray, np.ndarray]] = {
            term: (np.asarray(idxs, dtype=np.int32), np.asarray(wts, dtype=np.float64))
            for term, (idxs, wts) in postings.items()
        }

    @property
    def pool_size(self) -> int:
        return len(self.pool_ids)

    def score_query(self, query_grams: Sequence[str]) -> np.ndarray:
        """Compute exact BM25Okapi scores across all pool documents."""
        scores = np.zeros(self.pool_size, dtype=np.float64)
        for term, q_count in Counter(query_grams).items():
            posting = self._postings.get(term)
            if posting is not None:
                doc_idxs, weights = posting
                scores[doc_idxs] += q_count * weights
        return scores

    def retrieve(
        self,
        query_text: str,
        k: int,
        *,
        seed: int = 1,
        query_id: int = 0,
    ) -> list[RetrievedExample]:
        """Return min(k, N) examples ordered from lowest to highest similarity."""
        if k < 0:
            raise ValueError(f"k must be >= 0, got {k}")
        if k == 0:
            return []

        target_k = min(k, self.pool_size)
        query_grams = char_ngrams(query_text, n=self.ngram_size)
        scores = self.score_query(query_grams).tolist() if query_grams else [0.0] * self.pool_size

        pos_indices = [i for i, s in enumerate(scores) if s > 0.0]
        zero_indices = [i for i, s in enumerate(scores) if s <= 0.0]

        # Rank positive matches from highest to lowest score (tie-break by pool order).
        pos_indices.sort(key=lambda i: (-scores[i], i))
        chosen_pos = pos_indices[:target_k]

        chosen_zero: list[int] = []
        needed = target_k - len(chosen_pos)
        if needed > 0:
            rng = random.Random(f"seed={seed}:query={query_id}")
            shuffled_zero = list(zero_indices)
            rng.shuffle(shuffled_zero)
            chosen_zero = shuffled_zero[:needed]

        # Place zero-score fallbacks first, then positive matches in ascending score order
        # so the highest-similarity example is always last (immediately before the query).
        chosen_pos.reverse()
        ordered_indices = chosen_zero + chosen_pos

        return [
            RetrievedExample(
                pool_index=i,
                sent_id=self.pool_ids[i],
                src_text=self.src_texts[i],
                tgt_text=self.tgt_texts[i],
                score=max(0.0, scores[i]),
            )
            for i in ordered_indices
        ]

    def retrieve_pairs(
        self,
        query_text: str,
        k: int,
        *,
        seed: int = 1,
        query_id: int = 0,
    ) -> list[tuple[str, str]]:
        """Convenience wrapper returning (src_text, tgt_text) pairs ready for prompt building."""
        return [(ex.src_text, ex.tgt_text) for ex in self.retrieve(query_text, k, seed=seed, query_id=query_id)]
