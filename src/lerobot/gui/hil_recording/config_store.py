from __future__ import annotations

import json
from pathlib import Path
from typing import Any


CONFIG_PATH = Path.home() / ".config" / "evo-rl" / "hil_recording_gui.json"


def load_config(path: Path = CONFIG_PATH) -> tuple[dict[str, str], str | None]:
    if not path.is_file():
        return {}, None
    try:
        with open(path) as f:
            payload = json.load(f)
    except Exception as exc:
        return {}, f"Failed to load config {path}: {exc}"

    if not isinstance(payload, dict):
        return {}, f"Config {path} must contain a JSON object."

    values: dict[str, str] = {}
    for key, value in payload.items():
        if isinstance(key, str) and value is not None:
            values[key] = str(value)
    return values, None


def save_config(values: dict[str, Any], path: Path = CONFIG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {str(key): str(value) for key, value in values.items() if value is not None}
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2, sort_keys=True)
        f.write("\n")
