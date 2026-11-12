"""图节点实现 + 依赖 INDEX 的提示词包装函数。"""
import json

from langgraph.types import Send, interrupt

from config import CFG
from core.llm import get_llm, chat_json, embed_texts
from core.index import HybridIndex, tokenize
from core.retrieval import rewrite_query, build_context
from core.citations import verify
from graph import prompts
from schemas import AnswerPayload, CardModel, RawCitation

INDEX = HybridIndex()


# ---------------- 公共 ----------------
def route_node(state):
    """UI 已给定 mode；没有则按关键词兜底分类。"""
    if state.get("mode"):
        return {"mode": state["mode"]}
    q = state["question"]
    if any(k in q for k in ("对比", "比较", "区别")):
        return {"mode": "compare"}
    if any(k in q for k in ("综述", "写一篇", "总结这几篇")):
        return {"mode": "survey"}
    return {"mode": "qa"}


def route_branch(state) -> str:
    return state["mode"]


# ---------------- qa ----------------
def rewrite_node(state):
    info = rewrite_query(state["question"])
    return {"en_query": info["en_query"], "keywords": info["keywords"]}


def retrieve_node(state):
    qv = embed_texts([state["question"]])
    qv = qv[0] if qv is not None else None
    toks = tokenize(state.get("en_query", "") + " " + " ".join(state.get("keywords", [])))
    chunks = INDEX.search(qv, toks, k=CFG.top_k, doc_ids=state.get("doc_ids"))
    return {"chunk_ids": [c.chunk_id for c in chunks],
            "retrieved_ids": [c.chunk_id for c in chunks]}


def _chunks(state) -> list:
    return [INDEX.by_id[cid] for cid in state.get("chunk_ids", []) if cid in INDEX.by_id]


def answer_node(state):
    ctx, _sid_map = build_context(_chunks(state))
    user = prompts.ANSWER_USER.format(question=state["question"], context=ctx)

    hist = state.get("history") or []
    if hist:
        turns = "\n".join(
            f"用户：{m.get('q', '')[:300]}\n助手：{m.get('a', '')[:300]}" for m in hist[-3:]
        )
        user += f"\n\n（此前的对话历史，用于理解追问指代）：\n{turns}"

    if state.get("retries", 0) >= 1:
        user += ("\n\n注意：上一次的引文无法在原文中定位。本次 quotes 必须逐字复制片段中的"
                 "连续原文（不翻译、不改写、不跨段），否则会被判为无效引用。")

    raw = chat_json(get_llm(0.2), prompts.ANSWER_SYS, user)
    payload = AnswerPayload(**raw)
    return {"answer": payload.answer,
            "raw_citations": [c.model_dump() for c in payload.citations]}


def verify_node(state):
    chunks = _chunks(state)
    sid_map = {f"S{i + 1}": c for i, c in enumerate(chunks)}
    cits = verify([RawCitation(**c) for c in state.get("raw_citations", [])],
                  sid_map, set(state.get("retrieved_ids", [])))
    n = len(cits)
    invalid = sum(c.status == "invalid" for c in cits)
    return {"citations": [c.model_dump() for c in cits],
            "verify": {"n": n, "invalid": invalid, "ok_rate": 1 - invalid / max(1, n)},
            "retries": state.get("retries", 0) + 1}


def after_verify(state) -> str:
    if state["verify"]["ok_rate"] < 0.6 and state.get("retries", 0) <= CFG.max_retry:
        return "retry"
    return "end"


# ---------------- card ----------------
CARD_SECTIONS = ("abstract", "introduction", "method", "approach", "experiment",
                 "result", "conclusion", "limitation", "discussion")


def card_node(state):
    doc_id = state["doc_ids"][0]
    picks = [c for c in INDEX.chunks if c.doc_id == doc_id
             and any(s in c.section.lower() for s in CARD_SECTIONS)]
    picks = picks[:8] or [c for c in INDEX.chunks if c.doc_id == doc_id][:8]
    title = picks[0].title if picks else doc_id
    raw = chat_json(get_llm(0), prompts.CARD_SYS,
                    prompts.CARD_USER.format(title=title,
                                             text="\n\n".join(c.text for c in picks)))
    return {"card": CardModel(**raw).model_dump()}


# ---------------- compare ----------------
def compare_plan_node(state):
    """第一步：模型提议对比维度（写进 state，interrupt 重跑时不会重复调用）。"""
    dims = state.get("compare_dims_suggested")
    if not dims:
        dims = suggest_dims(state["question"])
    return {"compare_dims_suggested": dims}


def compare_confirm_node(state):
    """第二步：人工确认维度（interrupt 暂停，等 UI 用 Command(resume=...) 恢复）。"""
    dims = state.get("compare_dims_suggested", [])
    picked = interrupt({"type": "confirm_dims", "dims": dims})
    return {"compare_dims": picked.get("dims") or dims}


def fan_out_papers(state):
    return [Send("extract_paper", {"doc_id": d, "question": state["question"],
                                   "dims": state["compare_dims"]})
            for d in state["doc_ids"]]


def extract_paper_node(payload):
    row = extract_row(payload["doc_id"], payload["dims"], payload["question"])
    return {"per_paper": [row]}


def merge_matrix_node(state):
    return {"matrix_md": merge_matrix(state["question"], state["compare_dims"],
                                      state.get("per_paper", []))}


# ---------------- survey ----------------
def survey_outline_node(state):
    return {"outline": make_outline(state["question"], state["doc_ids"])}


def fan_out_sections(state):
    return [Send("write_section", {"section": s, "outline": state["outline"],
                                   "question": state["question"], "doc_ids": state["doc_ids"]})
            for s in state["outline"]]


def write_section_node(payload):
    md = write_section(payload["section"], payload["outline"],
                       payload["question"], payload["doc_ids"])
    return {"sections": [{"title": payload["section"]["title"], "md": md}]}


def merge_survey_node(state):
    md = assemble(state["outline"], state["sections"])
    return {"survey_md": polish(md)}


# ---------------- 提示词包装函数（依赖 INDEX） ----------------
def suggest_dims(question: str) -> list[str]:
    out = chat_json(get_llm(0), "你是论文对比维度设计器，只输出 json。",
                    f'为"{question}"设计 3-5 个可横向对比的维度（如研究方法、数据集、'
                    f'评价指标、检索策略、局限性）。输出 json：{{"dims": ["维度名"]}}')
    return out.get("dims", [])[:5]


def extract_row(doc_id: str, dims: list[str], question: str) -> dict:
    ctx = "\n\n".join(c.text for c in INDEX.chunks if c.doc_id == doc_id)[:12000]
    title = next((c.title for c in INDEX.chunks if c.doc_id == doc_id), doc_id)
    out = chat_json(get_llm(0), prompts.EXTRACT_ROW_SYS,
                    prompts.EXTRACT_ROW_USER.format(dims=dims, title=title, context=ctx))
    out["doc_id"] = doc_id
    return out


def merge_matrix(question: str, dims: list[str], rows: list[dict]) -> str:
    return get_llm(0.3).invoke([
        ("system", prompts.MERGE_SYS),
        ("human", prompts.MERGE_USER.format(question=question, dims=dims,
                                            rows=json.dumps(rows, ensure_ascii=False))),
    ]).content


def make_outline(question: str, doc_ids: list[str]) -> list[dict]:
    out = chat_json(get_llm(0), prompts.OUTLINE_SYS,
                    prompts.OUTLINE_USER.format(question=question,
                                                papers=", ".join(doc_ids)))
    return out.get("outline", [])


def write_section(section: dict, outline: list, question: str, doc_ids: list[str]) -> str:
    ctx = "\n\n".join(f"[{c.doc_id}] {c.text}" for c in INDEX.chunks
                      if c.doc_id in doc_ids)[:16000]
    return get_llm(0.4).invoke([
        ("system", prompts.SECTION_SYS),
        ("human", prompts.SECTION_USER.format(
            title=section["title"], points=section.get("points", []),
            outline=outline, context=ctx)),
    ]).content


def assemble(outline: list, sections: list[dict]) -> str:
    by_title = {s["title"]: s["md"] for s in sections}
    return "\n\n".join(f"## {s['title']}\n\n{by_title.get(s['title'], '')}" for s in outline)


def polish(md: str) -> str:
    return get_llm(0.3).invoke([
        ("system", prompts.POLISH_SYS),
        ("human", prompts.POLISH_USER.format(md=md)),
    ]).content
