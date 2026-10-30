"""Runtime configuration loaded from environment / `.env` (no external dotenv dependency)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Populate os.environ from a simple KEY=VALUE file without overriding existing vars."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


@dataclass(frozen=True)
class TierConfig:
    name: str
    base_url: str
    model: str
    api_key: str
    params_b: float  # active parameters in billions, used for compute units


@dataclass(frozen=True)
class CascadeConfig:
    # Pruner (SPEC 3.1)
    prune_theta: float = 0.18
    prune_min_k: int = 2
    relaxed_k: int = 4
    # Cache (SPEC 3.2)
    cache_cosine: float = 0.88
    # Adaptation (SPEC 3.5)
    tau_base: float = 0.55
    tau_min: float = 0.25
    tau_max: float = 0.75
    window: int = 20
    tighten_pass_rate: float = 0.70
    tighten_step: float = 0.08
    recover_pass_rate: float = 0.88
    recover_consecutive: int = 5
    recover_step: float = 0.05
    recover_latency_frac: float = 0.85
    min_window_samples: int = 5
    tighten_cooldown: int = 5
    latency_slo_ms: float = 8000.0
    audit_window: int = 6
    min_audit_samples: int = 3
    # Generation
    max_tokens: int = 256
    temperature: float = 0.0


@dataclass(frozen=True)
class Settings:
    mode: str
    tier1: TierConfig
    tier2: TierConfig
    cascade: CascadeConfig = field(default_factory=CascadeConfig)
    tokenizer_encoding: str = "o200k_base"
    request_timeout_s: float = 180.0


def get_settings() -> Settings:
    load_dotenv()
    tier1 = TierConfig(
        name="tier1",
        base_url=_env("TIER1_BASE_URL", "http://localhost:11434/v1"),
        model=_env("TIER1_MODEL", "qwen2.5:3b"),
        api_key=_env("TIER1_API_KEY", "ollama"),
        params_b=float(_env("TIER1_PARAMS_B", "3")),
    )
    tier2 = TierConfig(
        name="tier2",
        base_url=_env("TIER2_BASE_URL", "http://localhost:11434/v1"),
        model=_env("TIER2_MODEL", "qwen2.5:14b"),
        api_key=_env("TIER2_API_KEY", "ollama"),
        params_b=float(_env("TIER2_PARAMS_B", "14")),
    )
    cascade = CascadeConfig(latency_slo_ms=float(_env("LATENCY_SLO_MS", "8000")))
    return Settings(
        mode=_env("OPTIROUTE_MODE", "replay"),
        tier1=tier1,
        tier2=tier2,
        cascade=cascade,
        tokenizer_encoding=_env("TOKENIZER_ENCODING", "o200k_base"),
        request_timeout_s=float(_env("REQUEST_TIMEOUT_S", "180")),
    )
