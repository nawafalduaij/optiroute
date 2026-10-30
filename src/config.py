"""Runtime configuration: deployment profiles + environment / `.env` (no external dotenv dependency).

Precedence: shell environment variables > active profile (`profiles/<name>.env`, chosen via `get_settings(profile=)`
or `OPTIROUTE_PROFILE`) > the `.env` file > built-in defaults. So a one-off `TIER1_MODEL=... python evaluate.py`
beats the profile, while a profile beats whatever `.env` contains. Profiles never contain secrets: they name the environment
variable that holds the key (`TIER2_API_KEY_ENV=GROQ_API_KEY`), and the key itself lives in the git-ignored `.env`.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROFILES = ROOT / "profiles"


def parse_env_file(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def load_dotenv(path: Path = ROOT / ".env") -> dict[str, str]:
    """Read the `.env` file (without touching os.environ, so shell variables stay distinguishable)."""
    return parse_env_file(path) if path.exists() else {}


def load_profile(name: str) -> dict[str, str]:
    path = PROFILES / f"{name}.env"
    if not path.exists():
        raise ValueError(f"unknown profile {name!r}; available: {[p['name'] for p in list_profiles()]}")
    return parse_env_file(path)


def list_profiles() -> list[dict]:
    """Profile metadata, with whether its recorded benchmark exists in this checkout."""
    out = []
    for path in sorted(PROFILES.glob("*.env")):
        p = parse_env_file(path)
        rec = p.get("PROFILE_RECORDING")
        out.append({"name": path.stem, "title": p.get("PROFILE_TITLE", path.stem),
                    "description": p.get("PROFILE_DESCRIPTION", ""), "recording": rec,
                    "measured": bool(rec and (ROOT / rec).exists())})
    return out


@dataclass(frozen=True)
class TierConfig:
    name: str
    base_url: str
    model: str
    api_key: str
    params_b: float  # active parameters in billions, used for compute units
    extra_body: dict = field(default_factory=dict)  # provider-specific request options, e.g. reasoning_effort
    price_in_per_mtok: float | None = None  # optional USD per 1M input tokens (from your provider's price list)
    price_out_per_mtok: float | None = None


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
    profile: str | None = None


def get_settings(profile: str | None = None) -> Settings:
    dotenv = load_dotenv()
    name = profile or os.environ.get("OPTIROUTE_PROFILE") or dotenv.get("OPTIROUTE_PROFILE") or None
    prof = load_profile(name) if name else {}

    def get(key: str, default: str) -> str:
        return os.environ.get(key) or prof.get(key) or dotenv.get(key) or default

    def secret(var: str) -> str | None:
        return os.environ.get(var) or dotenv.get(var)

    def tier(n: int, model: str, params: str) -> TierConfig:
        key_env = get(f"TIER{n}_API_KEY_ENV", "")
        price_in, price_out = get(f"TIER{n}_PRICE_IN_PER_MTOK", ""), get(f"TIER{n}_PRICE_OUT_PER_MTOK", "")
        return TierConfig(
            name=f"tier{n}",
            base_url=get(f"TIER{n}_BASE_URL", "http://localhost:11434/v1"),
            model=get(f"TIER{n}_MODEL", model),
            api_key=(secret(key_env) if key_env else None) or get(f"TIER{n}_API_KEY", "ollama"),
            params_b=float(get(f"TIER{n}_PARAMS_B", params)),
            extra_body=json.loads(get(f"TIER{n}_EXTRA_BODY", "{}")),
            price_in_per_mtok=float(price_in) if price_in else None,
            price_out_per_mtok=float(price_out) if price_out else None,
        )

    return Settings(
        mode=get("OPTIROUTE_MODE", "replay"),
        tier1=tier(1, "qwen2.5:3b", "3"),
        tier2=tier(2, "qwen2.5:14b", "14"),
        cascade=CascadeConfig(latency_slo_ms=float(get("LATENCY_SLO_MS", "8000"))),
        tokenizer_encoding=get("TOKENIZER_ENCODING", "o200k_base"),
        request_timeout_s=float(get("REQUEST_TIMEOUT_S", "180")),
        profile=name,
    )
