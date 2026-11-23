"""Bootstrap helper ensuring the Apex Vision SDK (`apex_vision` + `theft_recognition_plugin`) is importable."""
from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any


def ensure_apex_vision_loaded() -> Any:
    """Resolve `apex_vision` (via site-packages, `APEX_VISION_PATH`, or sibling `ApexVision`/`VMClone`) and load core plugins."""
    try:
        import apex_vision  # noqa: F401
    except ImportError:
        candidates = []
        env_path = os.environ.get("APEX_VISION_PATH", "").strip()
        if env_path:
            candidates.append(Path(env_path))
        here = Path(__file__).resolve()
        for parent in here.parents:
            for folder_name in ("ApexVision", "ApexiVison", "vision-studio", "VMClone"):
                candidates.append(parent / folder_name)
                candidates.append(parent.parent / folder_name)
        for cand in candidates:
            if (cand / "apex_vision" / "__init__.py").is_file():
                s = str(cand)
                if s not in sys.path:
                    sys.path.insert(0, s)
                break
        import apex_vision  # noqa: F401

    from apex_vision.sdk import ensure_theft_plugin_loaded

    return ensure_theft_plugin_loaded()
