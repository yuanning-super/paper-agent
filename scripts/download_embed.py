"""预下载本地 embedding 模型（bge-small-zh-v1.5，约 100MB），并预热加载。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paper_agent.embed import get_embedder  # noqa: E402

if __name__ == "__main__":
    embedder = get_embedder()
    ok = embedder.load()
    if ok:
        print(f"✅ embedding 模型 {embedder.model_name} 下载并加载成功")
    else:
        print("❌ embedding 模型加载失败，检索将使用纯 BM25。请检查网络后重试。")
        sys.exit(1)
