"""AgentState：LangGraph 图的状态契约。

状态里只放 JSON 可序列化的字段（chunk 用 id 表示，不塞 pydantic 对象），
这样 checkpointer 落盘/恢复才不会出错。并行扇出的字段用 operator.add 汇合。
"""
from typing import Annotated, Literal, TypedDict
import operator


class AgentState(TypedDict, total=False):
    question: str
    mode: Literal["qa", "card", "compare", "survey"]
    doc_ids: list[str]
    history: list[dict]                 # 本轮之前的 [{q, a}]，供追问理解

    # qa 链路
    en_query: str
    keywords: list[str]
    chunk_ids: list[str]                # 本轮检索到的 chunk（每次覆盖，避免多轮累积）
    retrieved_ids: list[str]
    answer: str
    raw_citations: list[dict]
    citations: list[dict]
    verify: dict
    retries: int

    # card
    card: dict

    # compare
    compare_dims_suggested: list[str]
    compare_dims: list[str]
    per_paper: Annotated[list[dict], operator.add]    # Send×N 汇合点
    matrix_md: str

    # survey
    outline: list[dict]
    sections: Annotated[list[dict], operator.add]     # 同上
    survey_md: str
