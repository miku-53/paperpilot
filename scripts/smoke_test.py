"""三条链路自检：对话 / 嵌入 / PDF 解析。"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CFG


def main():
    # 1) 对话
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model=CFG.chat_model, api_key=CFG.deepseek_key,
                     base_url=CFG.chat_base_url, temperature=0,
                     timeout=120, max_retries=2)
    t = time.time()
    r = llm.invoke("只回复两个字：可以")
    print(f"LLM   OK  {r.content.strip()!r}  {time.time() - t:.1f}s")

    # 2) 嵌入（无 Key 则跳过）
    if CFG.qwen_key:
        from core.llm import embed_texts
        v = embed_texts(["学术论文阅读助手"])
        print(f"EMB   OK  dim={v.shape[1] if v is not None else '?'}")
    else:
        print("EMB   SKIP（无 QWEN_API_KEY，退化为仅 BM25）")

    # 3) PDF 解析
    from core.parsing import load_blocks, to_chunks
    pdf = CFG.papers / "react.pdf"
    blocks, pages = load_blocks(str(pdf), "react")
    chunks = to_chunks(blocks, "react", {"title": "ReAct"})
    print(f"PDF   OK  {pages}p  {len(blocks)} blocks  {len(chunks)} chunks")


if __name__ == "__main__":
    main()
