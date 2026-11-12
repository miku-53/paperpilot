"""入库脚本。用法：python scripts/ingest.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.ingest import ingest_all

if __name__ == "__main__":
    n = ingest_all(verbose=True)
    print(f"total chunks = {n}")
