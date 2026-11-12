"""检索 —— dense(向量) + sparse(BM25) 双路召回 + RRF 融合。

规模只有几百块，numpy 矩阵乘就够了，不需要向量数据库。
"""
import json
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from config import CFG
from schemas import Chunk


def tokenize(text: str) -> list[str]:
    """中英混合：英文按词，中文按 jieba 词。"""
    import jieba
    text = text.lower()
    en = re.findall(r"[a-z0-9][a-z0-9\-\.]*", text)
    zh = [w for w in jieba.lcut(text) if re.search(r"[一-鿿]", w)]
    return en + zh


def rrf(*rankings: list[str], k: int = CFG.rrf_k) -> list[str]:
    """Reciprocal Rank Fusion：score = Σ 1/(k + rank)，只比名次不比分数。"""
    score: dict[str, float] = {}
    for ranking in rankings:
        for r, cid in enumerate(ranking):
            score[cid] = score.get(cid, 0.0) + 1.0 / (k + r + 1)
    return sorted(score, key=score.get, reverse=True)


class HybridIndex:
    """混合索引：chunks.jsonl 存文本与坐标，emb.npy 存向量，内存里建 BM25。"""

    def __init__(self, index_dir: Path | None = None):
        self.dir = Path(index_dir) if index_dir else CFG.index_dir
        self.chunks: list[Chunk] = []
        self.by_id: dict[str, Chunk] = {}
        self.emb: np.ndarray | None = None
        self.bm25: BM25Okapi | None = None
        self.tokens: list[list[str]] = []
        self.doc_ids: np.ndarray | None = None
        self._load()

    def _load(self) -> None:
        cj = self.dir / "chunks.jsonl"
        if not cj.exists():
            return
        self.chunks = [Chunk(**json.loads(l)) for l in cj.open(encoding="utf-8")]
        self.by_id = {c.chunk_id: c for c in self.chunks}
        emb_path = self.dir / "emb.npy"
        self.emb = np.load(emb_path) if emb_path.exists() else None
        self.doc_ids = np.array([c.doc_id for c in self.chunks])
        self.tokens = [tokenize(c.text) for c in self.chunks]
        self.bm25 = BM25Okapi(self.tokens)

    def _mask(self, doc_ids: list[str] | None) -> np.ndarray:
        if not doc_ids:
            return np.ones(len(self.chunks), dtype=bool)
        return np.isin(self.doc_ids, doc_ids)

    def papers(self) -> list[dict]:
        seen: dict[str, str] = {}
        for c in self.chunks:
            if c.doc_id not in seen:
                seen[c.doc_id] = c.title
        return [{"doc_id": k, "title": v} for k, v in seen.items()]

    def search(self, q_vec, q_tokens, k: int = CFG.top_k,
               doc_ids: list[str] | None = None) -> list[Chunk]:
        mask = self._mask(doc_ids)

        dense_rank: list[str] = []
        if self.emb is not None and q_vec is not None:
            sims = self.emb @ q_vec
            sims[~mask] = -1e9
            order = np.argsort(-sims)[: CFG.dense_top]
            dense_rank = [self.chunks[i].chunk_id for i in order if mask[i]]

        sparse_rank: list[str] = []
        if self.bm25 is not None and q_tokens:
            scores = self.bm25.get_scores(q_tokens)
            scores[~mask] = -1e9
            order = np.argsort(-scores)[: CFG.sparse_top]
            sparse_rank = [self.chunks[i].chunk_id for i in order if mask[i] and scores[i] > 0]

        fused = rrf(dense_rank, sparse_rank, k=CFG.rrf_k)[:k]
        return [self.by_id[cid] for cid in fused]
