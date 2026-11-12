"""评测：Recall@5 / MRR（答辩的"证据页"，数字必须实测）。"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.index import HybridIndex, tokenize
from core.llm import embed_texts

# 少量 gold set（问题 / 目标文献 / gold 页码 / 关键词）
QUESTIONS = [
    {"q": "Self-RAG 用什么信号决定是否检索？", "docs": ["asai2023"], "gold_pages": [2, 3], "must_have": ["reflection", "critique"]},
    {"q": "RAG 相比纯生成模型的优势？", "docs": ["lewis2020"], "gold_pages": [1, 2], "must_have": ["retrieval", "knowledge"]},
    {"q": "Lost in the Middle 的主要结论？", "docs": ["liu2023"], "gold_pages": [1, 2], "must_have": ["position", "context"]},
    {"q": "GraphRAG 如何做查询聚焦摘要？", "docs": ["edge2024"], "gold_pages": [1, 2], "must_have": ["community", "graph"]},
    {"q": "ReAct 相比 chain-of-thought 的优势？", "docs": ["react"], "gold_pages": [1, 2], "must_have": ["reasoning", "acting"]},
]


def recall_at_k(k: int = 5):
    idx = HybridIndex()
    hit, mrr = 0, []
    for r in QUESTIONS:
        qv = embed_texts([r["q"]])
        qv = qv[0] if qv is not None else None
        got = idx.search(qv, tokenize(" ".join(r["must_have"])), k=k, doc_ids=r.get("docs"))
        pages = [c.pages[0] for c in got]
        ok = any(p in r["gold_pages"] for p in pages)
        hit += ok
        rank = next((i + 1 for i, p in enumerate(pages) if p in r["gold_pages"]), None)
        mrr.append(1 / rank if rank else 0)
    print(f"n={len(QUESTIONS)}  Recall@{k} = {hit / len(QUESTIONS):.3f}  MRR = {np.mean(mrr):.3f}")


if __name__ == "__main__":
    t = time.time()
    recall_at_k()
    print(f"eval done in {time.time() - t:.1f}s")
