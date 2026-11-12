"""图装配 + checkpointer。"""
import sqlite3

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from graph.state import AgentState
from graph import nodes


def build(checkpointer=None):
    g = StateGraph(AgentState)

    g.add_node("route", nodes.route_node)
    g.add_node("rewrite", nodes.rewrite_node)
    g.add_node("retrieve", nodes.retrieve_node)
    g.add_node("answer", nodes.answer_node)
    g.add_node("verify", nodes.verify_node)
    g.add_node("card", nodes.card_node)
    g.add_node("compare_plan", nodes.compare_plan_node)
    g.add_node("compare_confirm", nodes.compare_confirm_node)
    g.add_node("extract_paper", nodes.extract_paper_node)
    g.add_node("merge_matrix", nodes.merge_matrix_node)
    g.add_node("survey_outline", nodes.survey_outline_node)
    g.add_node("write_section", nodes.write_section_node)
    g.add_node("merge_survey", nodes.merge_survey_node)

    g.add_edge(START, "route")
    g.add_conditional_edges("route", nodes.route_branch, {
        "qa": "rewrite", "card": "card",
        "compare": "compare_plan", "survey": "survey_outline",
    })

    # qa：顺序 + 条件回边（校验不过则重答，最多 1 次）
    g.add_edge("rewrite", "retrieve")
    g.add_edge("retrieve", "answer")
    g.add_edge("answer", "verify")
    g.add_conditional_edges("verify", nodes.after_verify, {"retry": "answer", "end": END})

    g.add_edge("card", END)

    # compare：提议维度 → interrupt 确认 → Send 并行抽取 → 合并
    g.add_edge("compare_plan", "compare_confirm")
    g.add_conditional_edges("compare_confirm", nodes.fan_out_papers, ["extract_paper"])
    g.add_edge("extract_paper", "merge_matrix")
    g.add_edge("merge_matrix", END)

    # survey：大纲 → Send 并行写节 → 合并润色
    g.add_conditional_edges("survey_outline", nodes.fan_out_sections, ["write_section"])
    g.add_edge("write_section", "merge_survey")
    g.add_edge("merge_survey", END)

    return g.compile(checkpointer=checkpointer)


def get_graph(db_path: str):
    conn = sqlite3.connect(db_path, check_same_thread=False)   # Streamlit 必须
    return build(checkpointer=SqliteSaver(conn)), conn
