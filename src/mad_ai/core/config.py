from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        return json.loads(text)
    if suffix in {".yaml", ".yml"}:
        return _load_yaml(text)
    raise ValueError(f"Unsupported config format: {suffix}")


def _load_yaml(text: str) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError:
        return _simple_yaml_parse(text)
    data = yaml.safe_load(text)
    return data if isinstance(data, dict) else {}


def _simple_yaml_parse(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(0, result)]

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        key, _, value = line.strip().partition(":")
        while stack and indent < stack[-1][0]:
            stack.pop()
        current = stack[-1][1]

        parsed_value = value.strip()
        if not parsed_value:
            node: dict[str, Any] = {}
            current[key] = node
            stack.append((indent + 2, node))
            continue

        current[key] = _coerce_scalar(parsed_value)

    return result


def _coerce_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value
