"""全局配置：路径、模型、参数。

本机适配：Python 3.11 @ E:\\python3.11，Key 在项目根 .env。
稠密检索需要 QWEN_API_KEY（阿里云 DashScope 的 text-embedding-v4）；
没有该 Key 时自动退化为仅 BM25，问答链路照常可用。
"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent

load_dotenv(ROOT.parent / ".env")            # 备用：上级目录的 .env
load_dotenv(ROOT / ".env", override=True)    # 项目内 .env 优先


class CFG:
    root = ROOT
    papers = ROOT / "data" / "papers"
    index_dir = ROOT / "data" / "index"
    cache_dir = ROOT / "data" / "cache"
    ckpt_path = ROOT / "data" / "checkpoints.sqlite"

    # 模型（DeepSeek 对话 + Qwen 嵌入，均走 OpenAI 兼容协议）
    deepseek_key = os.getenv("deepseek-api") or os.getenv("DEEPSEEK_API_KEY", "")
    qwen_key = os.getenv("QWEN_API_KEY", "")
    zhipu_key = os.getenv("ZHIPU_API_KEY", "")
    chat_model = "deepseek-chat"
    chat_base_url = "https://api.deepseek.com"
    emb_model = "text-embedding-v4"
    emb_dim = 1024
    emb_batch = 10

    # 检索参数
    chunk_chars = 900
    chunk_overlap_blocks = 1
    dense_top = 30
    sparse_top = 30
    top_k = 8
    rrf_k = 60

    temperature = 0.2
    max_retry = 1
    demo_mode = os.getenv("DEMO_MODE", "live")   # live | cached
