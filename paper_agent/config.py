"""全局配置：路径、模型、分块与检索参数。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# 项目根目录（my-agent-2/）
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

# 首次 import 时加载 .env
load_dotenv(ROOT_DIR / ".env")

# 冲突处理：Claude Code 会话环境常同时存在 ANTHROPIC_API_KEY（SDK 解析优先级更高），
# 而本应用必须使用 .env 中的 DeepSeek 凭据（ANTHROPIC_AUTH_TOKEN，走 Authorization: Bearer）。
if os.environ.get("ANTHROPIC_AUTH_TOKEN") and os.environ.get("ANTHROPIC_API_KEY"):
    os.environ.pop("ANTHROPIC_API_KEY", None)


def _deepseek_model() -> str:
    """模型名解析：优先 DEEPSEEK_MODEL；否则取 ANTHROPIC_MODEL 并去掉 [1m] 等后缀
    （Anthropic 端点用 deepseek-v4-pro[1m]，OpenAI 端点用 deepseek-v4-pro）。"""
    model = os.environ.get("DEEPSEEK_MODEL") or os.environ.get(
        "ANTHROPIC_MODEL", "deepseek-v4-pro"
    )
    return model.split("[")[0].strip()


@dataclass
class Settings:
    """集中管理全部可调参数，默认值适合个人论文库规模。"""

    # 路径
    data_dir: Path = DATA_DIR
    db_path: Path = field(default_factory=lambda: DATA_DIR / "library.db")
    pdfs_dir: Path = field(default_factory=lambda: DATA_DIR / "pdfs")
    reports_dir: Path = field(default_factory=lambda: DATA_DIR / "reports")
    cache_dir: Path = field(default_factory=lambda: DATA_DIR / "cache")
    clones_dir: Path = field(default_factory=lambda: DATA_DIR / "clones")
    checkpoints_dir: Path = field(default_factory=lambda: DATA_DIR / "checkpoints")
    figures_dir: Path = field(default_factory=lambda: DATA_DIR / "figures")
    # 每篇论文的独立工作区：workspaces/{id}/ 下放代码克隆（repo/）、原文图表（figures/）、
    # 最终报告（report.md）与各阶段结果（background/method/experiment.md）
    workspaces_dir: Path = field(default_factory=lambda: DATA_DIR / "workspaces")

    # LLM（DeepSeek 原生 OpenAI 兼容接口）
    model: str = _deepseek_model()
    base_url: str = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    api_key: str = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get(
        "ANTHROPIC_AUTH_TOKEN", ""
    )
    max_tokens: int = 4096
    report_max_tokens: int = 8192
    json_temperature: float = 0.2
    report_temperature: float = 0.7

    # 上下文预算
    full_text_budget: int = 60_000  # 解读报告注入全文的字符上限
    tool_result_limit: int = 8_000  # 工具返回给 LLM 的文本上限（字符）

    def ensure_dirs(self) -> None:
        for p in (
            self.data_dir,
            self.pdfs_dir,
            self.reports_dir,
            self.cache_dir,
            self.clones_dir,
            self.checkpoints_dir,
            self.figures_dir,
            self.workspaces_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)


_settings: Settings | None = None


def load_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.ensure_dirs()
    return _settings
