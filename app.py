"""PaperPilot 论文领航 — Streamlit UI。

布局：左侧边栏放文献库与索引，主区放四个功能标签；问答的引用内联在
答案下方，点一下就在原处展开原文高亮。
"""
import json
import uuid

import streamlit as st
from langgraph.types import Command

from config import CFG
from core.index import HybridIndex
from core.citations import render_highlight
from core.ingest import ingest_all, add_paper
from graph.builder import get_graph

st.set_page_config(page_title="PaperPilot 论文领航", page_icon="📚",
                   layout="wide", initial_sidebar_state="expanded")

BADGE = {"exact": "🟢 已核对", "fuzzy": "🟡 近似",
         "fallback": "⚪ 段落定位", "invalid": "🔴 未核实"}
LABEL = {"rewrite": "改写检索式", "retrieve": "检索文献", "answer": "生成答案",
         "verify": "核对引用", "compare_plan": "设计对比维度", "compare_confirm": "等待确认",
         "extract_paper": "抽取各篇", "merge_matrix": "合并对比表",
         "survey_outline": "生成大纲", "write_section": "撰写分节", "merge_survey": "润色合并"}
QA_TID = "qa-session-01"


@st.cache_resource
def GRAPH():
    g, _conn = get_graph(str(CFG.ckpt_path))
    return g


@st.cache_resource
def INDEX():
    # 云端/换机后索引文件不在（被 gitignore），首次启动自动入库，做到开箱即用。
    if not (CFG.index_dir / "chunks.jsonl").exists():
        ingest_all()
    return HybridIndex()


@st.cache_resource
def load_meta() -> dict:
    p = CFG.papers / "meta.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def cfg_for(tid: str):
    return {"configurable": {"thread_id": tid}}


def run_qa(q: str, docs: list[str]) -> None:
    """把问题挂到对话，流式跑 QA 链路，回填最后一轮答案与引用。"""
    history = [{"q": t["q"], "a": t["a"]} for t in st.session_state.chat]
    st.session_state.chat.append({"q": q, "a": "", "citations": []})
    with st.status("分析中…", expanded=True) as s:
        for chunk in GRAPH().stream({"question": q, "mode": "qa", "doc_ids": docs,
                                     "history": history},
                                    cfg_for(QA_TID), stream_mode="updates"):
            for node in chunk:
                if node in LABEL:
                    s.write(f"✓ {LABEL[node]}")
    final = GRAPH().get_state(cfg_for(QA_TID)).values
    st.session_state.chat[-1] = {"q": q, "a": final.get("answer", ""),
                                 "citations": final.get("citations", [])}


def render_citations(turn_idx: int, turn: dict, meta: dict, idx: HybridIndex) -> None:
    """把引用渲染成可点击徽章；点击后在原地展开原文高亮。"""
    cits = turn.get("citations") or []
    if not cits:
        return
    n = min(len(cits), 6)
    cols = st.columns(n)
    for i, c in enumerate(cits):
        badge = BADGE.get(c.get("status"), "?")
        col = cols[i % n]
        if c.get("chunk_id") in idx.by_id:
            if col.button(f"[{c['sid']}] {badge}", key=f"cit_{turn_idx}_{i}",
                          use_container_width=True):
                st.session_state.pick = (turn_idx, i)
                st.rerun()
        else:
            col.caption(f"[{c['sid']}] {badge}")

    pick = st.session_state.get("pick")
    if not (pick and pick[0] == turn_idx):
        return
    c = cits[pick[1]]
    chunk = idx.by_id.get(c.get("chunk_id"))
    if not chunk or chunk.doc_id not in meta:
        return
    pdf = CFG.papers / meta[chunk.doc_id]["file"]
    fallback = next((sp.bbox for sp in chunk.spans if sp.page == c.get("page")),
                    chunk.spans[0].bbox if chunk.spans else None)
    png, hit = render_highlight(str(pdf), c.get("page"), c.get("quote", ""), fallback)
    st.image(png, caption=f"{chunk.title} · 第 {c.get('page', 0) + 1} 页")
    if not hit:
        st.caption("⚠️ 未能精确框选，已定位到该段落")


# ---------------- 会话状态 ----------------
for key, default in {
    "chat": [],
    "flash": None,
    "pick": None,
    "pending_dims": None,
    "cmp_tid": None,
    "matrix_md": "",
    "survey_md": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

st.markdown("""<style>
.block-container{padding-top:2.2rem;padding-bottom:2rem;}
div[data-testid="stCaptionContainer"]{color:#6b7280;}
</style>""", unsafe_allow_html=True)

meta = load_meta()
all_ids = list(meta)
idx = INDEX()

# ---------------- 侧边栏：文献库 / 索引 ----------------
with st.sidebar:
    st.title("📚 PaperPilot 论文领航")
    st.caption("学术论文问答 · 速读卡 · 多篇对比 · 综述")
    st.divider()

    st.subheader("文献库")
    docs = st.multiselect("参与问答的论文", all_ids, default=all_ids,
                          key="doc_select",
                          format_func=lambda d: f"{d} · {meta[d]['title'][:36]}")
    st.caption(f"已索引 {len(idx.chunks)} 块 · {len(all_ids)} 篇")

    mode = "混合检索（BM25 + 向量）" if CFG.qwen_key else "仅 BM25 关键词检索"
    st.caption(f"检索模式：{mode}")

    if st.button("🔄 重建索引", use_container_width=True):
        with st.spinner("解析 PDF 并入库中…"):
            n = ingest_all()
        st.cache_resource.clear()
        st.session_state.flash = f"✅ 已重建索引，共 {n} 块"
        st.rerun()

    st.divider()
    st.subheader("➕ 添加论文")
    up = st.file_uploader("上传 PDF", type=["pdf"], key="up_pdf",
                          help="上传后确认题名等信息，点入库即可参与问答。")
    if up is not None:
        auto_title = ""
        try:
            import pymupdf
            with pymupdf.open(stream=up.getvalue(), filetype="pdf") as d:
                auto_title = (d.metadata.get("title") or "").strip()
        except Exception:
            auto_title = ""
        if not auto_title:
            auto_title = up.name.rsplit(".", 1)[0]

        with st.form("add_paper"):
            title = st.text_input("题名", value=auto_title)
            authors = st.text_input("作者", placeholder="如 Yao et al.")
            c1, c2 = st.columns(2)
            year = c1.number_input("年份", min_value=1900, max_value=2100, value=2024, step=1)
            arxiv = c2.text_input("arXiv id（可选）", placeholder="如 2310.11511")
            ok = st.form_submit_button("✅ 入库并重建索引", use_container_width=True)
        if ok:
            doc_id = add_paper(up.getvalue(), up.name, title.strip(), authors.strip(),
                               int(year), arxiv.strip())
            with st.spinner("解析 PDF 并重建索引中…"):
                ingest_all()
            st.cache_resource.clear()
            if "doc_select" in st.session_state:
                st.session_state.doc_select = list(st.session_state.doc_select) + [doc_id]
            st.session_state.flash = f"✅ 已入库《{title.strip() or up.name}》，可开始提问"
            st.rerun()

    st.divider()
    with st.expander("💡 使用说明"):
        st.markdown("""- **问答**：就左侧勾选的论文提问，答案带 [S#] 引用，点一下即可核对原文。
- **速读卡**：单篇论文一键抽取研究问题/方法/结论。
- **对比**：多篇论文横向对比，维度可由你增删。
- **综述**：把所选论文自动写成一篇带引用的综述。
- 换论文：把 PDF 放进 `data/papers/`，改 `meta.json`，点「重建索引」即可。""")

# ---------------- 主区 ----------------
if st.session_state.get("flash"):
    st.success(st.session_state.flash)
    st.session_state.flash = None

st.title("PaperPilot 论文领航")
if docs:
    names = " · ".join(meta[d]["title"].split(":")[0] for d in docs)
    st.caption(f"当前参与文献：{names}")
else:
    st.caption("请先在左侧选择至少一篇论文。")

tab_qa, tab_card, tab_cmp, tab_sur = st.tabs(["💬 问答", "⚡ 速读卡", "📊 对比", "📝 综述"])

# -------- 问答 --------
with tab_qa:
    if not docs:
        st.info("请先在左侧勾选至少一篇论文。")
    else:
        if not st.session_state.chat:
            st.markdown("**💬 试试这样问：**")
            ex_cols = st.columns(3)
            examples = [
                "这几篇论文分别在解决什么问题？",
                "比较 ReAct 与 Self-RAG 的检索时机差异",
                "这些方法各自有哪些局限？",
            ]
            for i, ex in enumerate(examples):
                if ex_cols[i].button(ex, key=f"ex_{i}", use_container_width=True):
                    run_qa(ex, docs)
                    st.rerun()

        for ti, turn in enumerate(st.session_state.chat):
            with st.chat_message("user"):
                st.markdown(turn["q"])
            with st.chat_message("assistant"):
                st.markdown(turn["a"])
                render_citations(ti, turn, meta, idx)

        if q := st.chat_input("就所选论文提问…"):
            run_qa(q, docs)
            st.rerun()

# -------- 速读卡 --------
with tab_card:
    if not docs:
        st.info("请先在左侧勾选一篇论文。")
    else:
        sel = st.selectbox("选择论文", docs, key="card_sel",
                           format_func=lambda d: f"{d} · {meta[d]['title']}")
        if st.button("⚡ 生成速读卡", type="primary", key="card_go"):
            with st.status("抽取中…", expanded=True) as s:
                for chunk in GRAPH().stream({"question": "", "mode": "card",
                                             "doc_ids": [sel]},
                                            cfg_for(f"card-{sel}"), stream_mode="updates"):
                    for node in chunk:
                        if node in LABEL:
                            s.write(f"✓ {LABEL[node]}")
            card = GRAPH().get_state(cfg_for(f"card-{sel}")).values.get("card") or {}
            if card:
                st.markdown(f"### {meta[sel]['title']}")
                st.markdown(f"**🎯 研究问题**\n\n{card.get('research_problem') or '—'}")
                for key, icon, name in [("method", "🧪", "方法"), ("datasets", "📊", "数据集"),
                                        ("metrics", "📏", "评价指标")]:
                    items = card.get(key) or []
                    if items:
                        st.markdown(f"**{icon} {name}**")
                        st.markdown("\n".join(f"- {x}" for x in items))
                findings = card.get("main_findings") or []
                if findings:
                    st.markdown("**💡 主要结论**")
                    st.markdown("\n".join(f"- {x}" for x in findings))
                lims = card.get("limitations") or []
                if lims:
                    st.markdown(f"**⚠️ 局限**")
                    st.markdown("\n".join(f"- {x}" for x in lims))

# -------- 对比 --------
with tab_cmp:
    if len(docs) < 2:
        st.info("对比至少需要 2 篇论文，请先在左侧多勾选几篇。")
    else:
        q_cmp = st.text_input("对比问题", "这几篇在方法与实验设置上有何异同？", key="cmp_q")
        if st.session_state.pending_dims:
            st.info("模型建议了以下对比维度，可增删后确认：")
            with st.form("confirm_dims"):
                chosen = st.multiselect("对比维度", st.session_state.pending_dims,
                                        default=st.session_state.pending_dims)
                if st.form_submit_button("✅ 确认并生成对比表"):
                    with st.status("对比中…", expanded=True) as s:
                        GRAPH().invoke(Command(resume={"dims": chosen}),
                                       cfg_for(st.session_state.cmp_tid))
                    st.session_state.matrix_md = GRAPH().get_state(
                        cfg_for(st.session_state.cmp_tid)).values.get("matrix_md", "")
                    st.session_state.pending_dims = None
                    st.rerun()
        elif st.button("开始对比", type="primary", key="cmp_go"):
            tid = uuid.uuid4().hex
            st.session_state.cmp_tid = tid
            out = GRAPH().invoke({"question": q_cmp, "mode": "compare", "doc_ids": docs},
                                 cfg_for(tid))
            itr = out.get("__interrupt__")
            if itr:
                st.session_state.pending_dims = itr[0].value.get("dims", [])
                st.rerun()
            st.session_state.matrix_md = out.get("matrix_md", "")
            st.rerun()
        if st.session_state.matrix_md:
            st.markdown(st.session_state.matrix_md)

# -------- 综述 --------
with tab_sur:
    if not docs:
        st.info("请先在左侧勾选参与综述的论文。")
    else:
        q_sur = st.text_input("综述主题", "检索增强生成（RAG）研究进展综述", key="sur_q")
        if st.button("📝 生成综述", type="primary", key="sur_go"):
            tid = uuid.uuid4().hex
            with st.status("生成中…", expanded=True) as s:
                for chunk in GRAPH().stream({"question": q_sur, "mode": "survey",
                                             "doc_ids": docs},
                                            cfg_for(tid), stream_mode="updates"):
                    for node in chunk:
                        if node in LABEL:
                            s.write(f"✓ {LABEL[node]}")
            st.session_state.survey_md = GRAPH().get_state(
                cfg_for(tid)).values.get("survey_md", "")
        if st.session_state.survey_md:
            st.download_button("⬇️ 导出 Markdown", st.session_state.survey_md,
                               file_name="文献综述.md")
            st.markdown(st.session_state.survey_md)
