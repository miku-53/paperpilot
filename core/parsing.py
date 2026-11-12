"""PDF → Block → Section → Chunk。

核心原则：按「版面块」打包，而不是先拼全文再按字符切。
这样每个 chunk 天然持有若干完整 Block，页码和 bbox 精确到行级，
高亮时才能框住"那一句"，坐标零漂移。
"""
import re
from collections import Counter

import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

from schemas import Block, BlockSpan, Chunk

SIMPLE_HEAD = {"abstract", "introduction", "related work", "background", "method",
               "methods", "approach", "experiments", "experimental setup", "results",
               "discussion", "conclusion", "conclusions", "limitations", "references"}


def norm_alpha(t: str) -> str:
    """去掉所有空白并大写，对付小型大写字母（small caps）。"""
    return re.sub(r"\s+", "", t).upper()


SIMPLE_NORM = {norm_alpha(x) for x in SIMPLE_HEAD}

SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=900, chunk_overlap=120,
    separators=["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""],
    length_function=len,
)


def load_blocks(pdf_path: str, doc_id: str) -> tuple[list[Block], int]:
    doc = pymupdf.open(pdf_path)
    out: list[Block] = []
    for pno, page in enumerate(doc):
        h = page.rect.height
        for i, b in enumerate(page.get_text("dict")["blocks"]):
            if b.get("type") != 0:
                continue
            spans = [s for line in b["lines"] for s in line["spans"] if s["text"].strip()]
            if not spans:
                continue
            text = re.sub(r"\s+", " ", " ".join(s["text"].strip() for s in spans)).strip()
            if len(text) < 3:
                continue
            x0, y0, x1, y1 = b["bbox"]
            # 页眉页脚：靠边且很短
            if (y0 < h * 0.05 or y1 > h * 0.95) and len(text) < 60:
                continue
            size = max(s["size"] for s in spans)
            bold = any("bold" in s["font"].lower() or (s["flags"] & 16) for s in spans)
            out.append(Block(
                block_id=f"{doc_id}:{pno}:{i}", text=text, page=pno,
                bbox=(x0, y0, x1, y1), size=round(float(size), 1), bold=bold,
            ))
    pages = doc.page_count
    doc.close()
    return out, pages


def body_size(blocks: list[Block]) -> float:
    """按字符数加权的众数字号 = 正文字号。"""
    c = Counter()
    for b in blocks:
        c[round(b.size)] += len(b.text)
    return max(c.items(), key=lambda kv: kv[1])[0]


def is_heading(b: Block, body: float) -> bool:
    t = b.text.strip()
    if len(t) > 110 or len(t.split()) > 14:
        return False
    if b.size > body * 1.6:            # 论文大标题 / 作者行
        return False
    if b.size < body * 0.95:           # 表格、脚注
        return False
    if norm_alpha(t).strip(".") in SIMPLE_NORM:
        return True
    if re.match(r"^\d+(\.\d+)*\s", t) and b.size >= body * 1.05:
        return True
    letters = [c for c in t if c.isalpha()]
    upper_ratio = sum(c.isupper() for c in letters) / max(1, len(letters))
    return b.size >= body * 1.15 and upper_ratio > 0.7


def clean_heading(t: str) -> str:
    """修复 small caps 被拆字：'1 I NTRODUCTION' → '1 INTRODUCTION'。"""
    t = re.sub(r"(?<=\b[A-Z]) (?=[A-Z]{2,})", "", t)
    t = re.sub(r"\s*-\s*", "-", t)
    return re.sub(r"\s+", " ", t).strip()


def to_chunks(blocks: list[Block], doc_id: str, meta: dict,
              target: int = 900, max_block: int = 1600) -> list[Chunk]:
    body = body_size(blocks)
    chunks: list[Chunk] = []
    cur: list[Block] = []
    cur_len = 0
    section = "Front Matter"
    idx = 0

    def emit(piece: str, bs: list[Block], kind: str = "text") -> None:
        nonlocal idx
        chunks.append(Chunk(
            chunk_id=f"{doc_id}#{idx:04d}", doc_id=doc_id,
            title=meta.get("title", doc_id), section=section,
            pages=sorted({b.page for b in bs}),
            spans=[BlockSpan(page=b.page, bbox=list(b.bbox)) for b in bs],
            text=piece, kind=kind,
        ))
        idx += 1

    def flush() -> None:
        nonlocal cur, cur_len
        if not cur:
            return
        emit("\n".join(b.text for b in cur), cur)
        cur = cur[-1:] if len(cur) > 1 else []   # 重叠 1 块
        cur_len = sum(len(b.text) for b in cur)

    for b in blocks:
        if is_heading(b, body):
            flush()
            section = clean_heading(b.text)[:80]
            cur, cur_len = [], 0
            continue
        if len(b.text) > max_block:              # 超长块：二次切，坐标沿用整块
            flush()
            for piece in SPLITTER.split_text(b.text):
                emit(piece, [b])
            continue
        if cur and cur_len + len(b.text) > target:
            flush()
        cur.append(b)
        cur_len += len(b.text)

    flush()
    return chunks
