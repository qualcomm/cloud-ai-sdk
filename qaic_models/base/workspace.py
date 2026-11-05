# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Project-local locations for generated demo inputs and small caches."""

from __future__ import annotations

import os
from pathlib import Path


WORK_ROOT_ENV = "QAIC_WORK_ROOT"


def qaic_work_root() -> Path:
    """Return the caller-selected work root, defaulting below this checkout."""
    configured = os.environ.get(WORK_ROOT_ENV)
    if configured:
        root = Path(configured).expanduser()
    else:
        root = next(
            (parent / "qaic-work" for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file()),
            Path.cwd() / "qaic-work",
        )
    root.mkdir(parents=True, exist_ok=True)
    return root


def qaic_work_path(category: str, filename: str) -> Path:
    """Return one project-local generated-file path under ``category``."""
    if not category or Path(category).is_absolute() or ".." in Path(category).parts:
        raise ValueError(f"invalid workspace category: {category!r}")
    if not filename or Path(filename).is_absolute() or ".." in Path(filename).parts:
        raise ValueError(f"invalid workspace filename: {filename!r}")
    path = qaic_work_root() / category / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
