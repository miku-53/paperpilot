"""引用校验 + 高亮渲染。"""
import re
from difflib import SequenceMatcher

import pymupdf

from schemas import Citation, Chunk


def norm(s: str) -> str:
    s = s.replace("-\n", "").replace("­", "")   # 去连字符断行/软连字符
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()


def best_ratio(q: str, text: str) -> float:
    """在 text 上滑窗找与 q 最相似的片段（按词）。"""
    qw, tw = q.split(), text.split()
    if not qw:
        return 0.0
    if len(tw) <= len(qw):
        return SequenceMatcher(None, q, text).ratio()
    best, win = 0.0, len(qw)
    for i in range(len(tw) - win + 1):
        r = SequenceMatcher(None, q, " ".join(tw[i:i + win])).ratio()
        if r > best:
            best = r
            if best > 0.95:
                break
    return best


def verify(raw_citations, sid_map: dict[str, Chunk], retrieved_ids: set[str]) -> list[Citation]:
    out: list[Citation] = []
    for rc in raw_citations:
        ch = sid_map.get(rc.sid)
        q = norm(rc.quote)
        if ch is None or ch.chunk_id not in retrieved_ids or len(q) < 8:
            out.append(Citation(sid=rc.sid, quote=rc.quote, status="invalid"))
            continue
        if q in norm(ch.text):
            status = "exact"
        else:
            r = best_ratio(q, norm(ch.text))
            status = "fuzzy" if r >= 0.80 else ("fallback" if r >= 0.55 else "invalid")
        out.append(Citation(sid=rc.sid, chunk_id=ch.chunk_id, page=ch.pages[0],
                            quote=rc.quote, status=status))
    return out


def render_highlight(pdf_path: str, page_no: int, quote: str,
                     fallback_bbox: list[float] | None = None,
                     zoom: float = 2.0) -> tuple[bytes, bool]:
    doc = pymupdf.open(pdf_path)
    try:
        page = doc[page_no]
        rects = page.search_for(re.sub(r"\s+", " ", quote).strip())
        # 兜底 1：逐步缩短词窗（应对断行、连字符、大小写）
        if not rects:
            words = quote.split()
            for n in (10, 8, 6, 4, 3):
                if len(words) >= n:
                    rects = page.search_for(" ".join(words[:n]))
                    if rects:
                        break
        # 兜底 2：整块 bbox
        if not rects and fallback_bbox:
            rects = [pymupdf.Rect(*fallback_bbox)]
        for r in rects:
            r = pymupdf.Rect(r.x0 - 2, r.y0 - 2, r.x1 + 2, r.y1 + 2)
            page.draw_rect(r, color=(0.95, 0.70, 0.10), width=1.2,
                           fill=(1.0, 0.92, 0.40), fill_opacity=0.35)
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
        return pix.tobytes("png"), bool(rects)
    finally:
        doc.close()
