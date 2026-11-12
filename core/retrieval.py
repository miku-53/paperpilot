"""查询改写 / 召回上下文拼装。"""
from config import CFG
from core.llm import get_llm, chat_json, embed_texts
from core.index import tokenize

REWRITE_SYS = "你是检索式改写器。只输出 json。"


def rewrite_query(question: str) -> dict:
    out = chat_json(get_llm(0), REWRITE_SYS, f"""把用户的中文问题改写成英文检索式，用于在英文论文库中检索。
输出 json：{{"en_query": "...", "keywords": ["术语","缩写","数据集名"]}}
要求：en_query 是 15-30 词的英文短语，覆盖同义词与方法名；keywords 给 3-8 个英文术语。
问题：{question}""")
    return {"en_query": out.get("en_query", question),
            "keywords": out.get("keywords", [])}


def build_context(chunks) -> tuple[str, dict]:
    """拼 prompt 上下文，同时建立 SID → Chunk 映射。"""
    lines, sid_map = [], {}
    for i, c in enumerate(chunks, 1):
        sid = f"S{i}"
        sid_map[sid] = c
        lines.append(f"[{sid}] {c.title} · {c.section} · p.{c.pages[0] + 1}\n{c.text}")
    return "\n\n".join(lines), sid_map
