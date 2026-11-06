# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import shutil
import tarfile
from pathlib import Path


def safe_extract_tar(archive: tarfile.TarFile, destination: str | Path) -> None:
    """Extract only regular files and directories beneath destination."""
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=True)

    members = []
    for member in archive.getmembers():
        target = (root / member.name).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise tarfile.ExtractError(
                f"Archive member escapes extraction directory: {member.name}"
            ) from exc

        if not (member.isdir() or member.isfile()):
            raise tarfile.ExtractError(
                f"Unsupported archive member type: {member.name}"
            )
        if target == root and member.isfile():
            raise tarfile.ExtractError("Archive file cannot replace extraction directory")

        members.append((member, target))

    for member, target in members:
        if member.isdir():
            target.mkdir(parents=True, exist_ok=True)
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        source = archive.extractfile(member)
        if source is None:
            raise tarfile.ExtractError(f"Unable to read archive member: {member.name}")
        with source, target.open("wb") as output:
            shutil.copyfileobj(source, output)
        target.chmod(member.mode & 0o777)
