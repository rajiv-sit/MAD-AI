from __future__ import annotations

import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from uuid import uuid4


@contextmanager
def workspace_temp_dir() -> Iterator[Path]:
    root = Path("outputs") / "test_artifacts"
    root.mkdir(parents=True, exist_ok=True)
    temp_dir = root / f"case_{uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
