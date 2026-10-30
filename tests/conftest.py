import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OPTIROUTE_MODE", "sim")

from src.config import get_settings  # noqa: E402
from src.schemas import Task  # noqa: E402


@pytest.fixture(scope="session")
def settings():
    return get_settings()


@pytest.fixture(scope="session")
def held_out():
    path = ROOT / "data" / "held_out_test.json"
    if not path.exists():
        import subprocess
        subprocess.run([sys.executable, str(ROOT / "scripts" / "generate_dataset.py")], check=True)
    return {d["task_id"]: Task.from_dict(d) for d in json.loads(path.read_text(encoding="utf-8"))}
