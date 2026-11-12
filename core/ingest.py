"""入库：解析 meta.json 里所有 PDF，落盘 chunks.jsonl + emb.npy。"""
import json
import re

import numpy as np

from config import CFG
from core.parsing import load_blocks, to_chunks
from core.llm import embed_texts


def slugify(text: str) -> str:
    """把题名/文件名转成 doc_id：保留中英文与数字，其余转连字符。"""
    s = re.sub(r"[^\w]+", "-", (text or "").lower()).strip("-")
    return s[:40] or "paper"


def add_paper(pdf_bytes: bytes, filename: str, title: str,
              authors: str, year: int, arxiv: str) -> str:
    """把用户上传的 PDF 写进文献库并登记到 meta.json，返回 doc_id。"""
    meta_path = CFG.papers / "meta.json"
    meta_all = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    base = slugify(title or filename.rsplit(".", 1)[0])
    doc_id, i = base, 1
    while doc_id in meta_all:                      # 同名去重
        i += 1
        doc_id = f"{base}-{i}"

    (CFG.papers / f"{doc_id}.pdf").write_bytes(pdf_bytes)
    meta_all[doc_id] = {
        "file": f"{doc_id}.pdf",
        "title": title or filename,
        "authors": authors or "Unknown",
        "year": int(year),
        "arxiv": arxiv or "",
    }
    meta_path.write_text(json.dumps(meta_all, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    return doc_id


def ingest_all(verbose: bool = False) -> int:
    meta_path = CFG.papers / "meta.json"
    meta_all = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    CFG.index_dir.mkdir(parents=True, exist_ok=True)

    all_chunks = []
    for doc_id, meta in meta_all.items():
        pdf = CFG.papers / meta["file"]
        if not pdf.exists():
            if verbose:
                print(f"跳过 {doc_id}：找不到 {pdf}")
            continue
        blocks, pages = load_blocks(str(pdf), doc_id)
        chunks = to_chunks(blocks, doc_id, meta)
        all_chunks += chunks
        if verbose:
            print(f"{doc_id:<12} pages={pages:<3} blocks={len(blocks):<4} chunks={len(chunks)}")

    with (CFG.index_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(c.model_dump_json() + "\n")

    emb = embed_texts([c.text for c in all_chunks])
    if emb is not None:
        np.save(CFG.index_dir / "emb.npy", emb)
        emb_meta = {"model": CFG.emb_model, "dim": CFG.emb_dim, "n": len(all_chunks)}
    else:
        (CFG.index_dir / "emb.npy").unlink(missing_ok=True)   # 清掉旧向量，避免与 chunks 不一致
        emb_meta = {"model": "skipped", "dim": 0, "n": len(all_chunks)}

    (CFG.index_dir / "meta.json").write_text(
        json.dumps(emb_meta, ensure_ascii=False), encoding="utf-8")
    if verbose:
        print(f"索引已落盘：{len(all_chunks)} 块（向量模型：{emb_meta['model']}）")
    return len(all_chunks)
