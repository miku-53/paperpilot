"""端到端跑一遍图：qa → card → compare(interrupt) → survey。用于本地验证，不参与 UI。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langgraph.types import Command
from graph.builder import get_graph


def main():
    g, conn = get_graph(str(__import__("config").CFG.ckpt_path))
    cfg = {"configurable": {"thread_id": "qa-test"}}

    print("== QA ==")
    for chunk in g.stream({"question": "ReAct 和 Self-RAG 在何时检索上有何不同？",
                           "mode": "qa",
                           "doc_ids": ["react", "asai2023", "liu2023"]},
                          cfg, stream_mode="updates"):
        for k in chunk:
            print("  node:", k)
    st = g.get_state(cfg).values
    print("  answer:", (st.get("answer") or "")[:200])
    print("  citations:", [(c.get("sid"), c.get("status")) for c in st.get("citations", [])])
    print("  verify:", st.get("verify"))

    print("== CARD ==")
    for chunk in g.stream({"question": "", "mode": "card", "doc_ids": ["react"]},
                          {"configurable": {"thread_id": "card-test"}}, stream_mode="updates"):
        for k in chunk:
            print("  node:", k)
    card = g.get_state({"configurable": {"thread_id": "card-test"}}).values.get("card", {})
    print("  problem:", card.get("research_problem"))
    print("  method:", card.get("method"))

    print("== COMPARE (interrupt) ==")
    cfg_c = {"configurable": {"thread_id": "cmp-test"}}
    out = g.invoke({"question": "这几篇的检索策略有何不同？", "mode": "compare",
                    "doc_ids": ["react", "asai2023", "lewis2020"]}, cfg_c)
    itr = out.get("__interrupt__")
    dims = itr[0].value.get("dims") if itr else []
    print("  suggested dims:", dims)
    if dims:
        out = g.invoke(Command(resume={"dims": dims}), cfg_c)
        matrix = g.get_state(cfg_c).values.get("matrix_md", "")
        print("  matrix head:", matrix[:200].replace("\n", " "))

    print("== SURVEY ==")
    for chunk in g.stream({"question": "检索增强生成（RAG）研究进展综述", "mode": "survey",
                           "doc_ids": ["lewis2020", "asai2023", "edge2024"]},
                          {"configurable": {"thread_id": "sur-test"}}, stream_mode="updates"):
        for k in chunk:
            print("  node:", k)
    sur = g.get_state({"configurable": {"thread_id": "sur-test"}}).values.get("survey_md", "")
    print("  survey head:", sur[:300].replace("\n", " "))


if __name__ == "__main__":
    main()
