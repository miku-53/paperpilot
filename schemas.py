"""Pydantic 数据模型：接口契约（先定这个，别改字段名）。

设计说明（答辩可讲）：
- 模型只输出 SID 和 quote，不输出页码/坐标。页码由 SID 对应的 chunk 决定，
  坐标由 PyMuPDF 在原文里搜 quote 得到——模型没有机会编造位置。
- Chunk.spans 用列表：论文一段常跨页，每个版面块单独记 page + bbox。
"""
from typing import Literal, Optional

from pydantic import BaseModel, Field


class Block(BaseModel):
    """PDF 版面块：解析的最小单位，坐标用于高亮。"""
    block_id: str
    text: str
    page: int                                   # 0 基
    bbox: tuple[float, float, float, float]     # PyMuPDF 坐标（左上原点）
    size: float
    bold: bool


class BlockSpan(BaseModel):
    """块在页面上的位置；高亮兜底时取同页的 span。"""
    page: int
    bbox: list[float]


class Chunk(BaseModel):
    """检索与引用的最小单位：一段常跨页，所以 spans 是列表。"""
    chunk_id: str                               # f"{doc_id}#{i:04d}"
    doc_id: str
    title: str
    section: str
    pages: list[int]                            # 涉及的页码（去重升序）
    spans: list[BlockSpan]                      # 每个版面块一个，保留精确坐标
    text: str
    kind: Literal["text", "table"] = "text"


class RawCitation(BaseModel):
    """模型输出：只给 SID 和原文引文，坐标由后端算。"""
    sid: str                                    # "S1".."S8"
    quote: str = Field(max_length=300)


class AnswerPayload(BaseModel):
    answer: str
    citations: list[RawCitation] = []
    confidence: Literal["high", "medium", "low"] = "medium"


class Citation(RawCitation):
    chunk_id: str = ""
    page: int = -1
    status: Literal["exact", "fuzzy", "fallback", "invalid"] = "invalid"


class CardModel(BaseModel):
    """速读卡：字段缺失就 None，禁止编造。"""
    research_problem: Optional[str] = None
    method: Optional[list[str]] = None
    datasets: Optional[list[str]] = None
    metrics: Optional[list[str]] = None
    main_findings: Optional[list[str]] = None
    limitations: Optional[list[str]] = None
