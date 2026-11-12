"""模型 / 嵌入 / JSON 解析。"""
import json
import os
import re
import time

import numpy as np
from langchain_openai import ChatOpenAI

from config import CFG

_WARNED_EMB = False


def get_llm(temperature: float = 0.2, **kw) -> ChatOpenAI:
    return ChatOpenAI(
        model=CFG.chat_model,
        api_key=CFG.deepseek_key,
        base_url=CFG.chat_base_url,
        temperature=temperature,
        timeout=180,
        max_retries=2,
        **kw,
    )


def embed_texts(texts: list[str], dim: int = CFG.emb_dim) -> np.ndarray | None:
    """稠密嵌入。无 Qwen Key 时返回 None（调用方退化为仅 BM25）。"""
    global _WARNED_EMB
    if not CFG.qwen_key:
        if not _WARNED_EMB:
            print("[embed] 未检测到 QWEN_API_KEY，跳过向量化，检索退化为仅 BM25。")
            _WARNED_EMB = True
        return None

    os.environ.setdefault("DASHSCOPE_API_KEY", CFG.qwen_key)
    from dashscope import TextEmbedding

    vecs: list[list[float]] = []
    for i in range(0, len(texts), CFG.emb_batch):
        batch = texts[i:i + CFG.emb_batch]
        for attempt in range(5):
            r = TextEmbedding.call(model=CFG.emb_model, input=batch, dimension=dim)
            if r.status_code == 200:
                vecs += [e["embedding"] for e in r.output["embeddings"]]
                break
            time.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"embedding failed: {r.code} {r.message}")
    m = np.asarray(vecs, dtype="float32")
    m /= np.linalg.norm(m, axis=1, keepdims=True) + 1e-9
    return m


def loads_loose(s: str) -> dict:
    """容忍 ```json 包裹、前后废话。"""
    s = re.sub(r"^```\w*\s*|\s*```$", "", s.strip())
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        i, j = s.find("{"), s.rfind("}")
        if i >= 0 and j > i:
            return json.loads(s[i:j + 1])
        raise


def chat_json(llm, system: str, user: str) -> dict:
    """DeepSeek 的 JSON 模式要求 prompt 里出现 json 字样。"""
    msg = llm.bind(response_format={"type": "json_object"}).invoke(
        [("system", system), ("human", user)]
    )
    return loads_loose(msg.content)
