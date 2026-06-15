"""Run provenance: everything needed to reproduce a result."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def package_versions() -> dict:
    out = {}
    for mod in ("numpy", "pandas", "sklearn", "scipy", "shap", "lightgbm", "pyarrow"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = "not installed"
    return out


def run_metadata(settings=None, extra: dict | None = None) -> dict:
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "packages": package_versions(),
    }
    if settings is not None:
        meta["settings"] = settings.to_dict() if hasattr(settings, "to_dict") else settings
    if extra:
        meta.update(extra)
    return meta


def write_metadata(path: Path, settings=None, extra: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run_metadata(settings, extra), indent=2, default=str))
    return path
