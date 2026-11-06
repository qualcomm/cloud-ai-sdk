# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Security tests for extracting downloaded QPC archives."""

import io
import tarfile
import tempfile
import unittest
from pathlib import Path

try:
    from safe_tar import safe_extract_tar
except ModuleNotFoundError as exc:
    if exc.name != "safe_tar":
        raise
    safe_extract_tar = None


def make_tar(members):
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w") as archive:
        for member, contents in members:
            data = contents.encode() if isinstance(contents, str) else contents
            if member.isfile():
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
            else:
                archive.addfile(member)
    return tarfile.open(fileobj=io.BytesIO(payload.getvalue()), mode="r:")


class SafeTarExtractionTests(unittest.TestCase):
    def extract(self, archive, destination):
        self.assertIsNotNone(safe_extract_tar, "safe archive extraction is not implemented")
        safe_extract_tar(archive, destination)

    def test_extracts_regular_files_under_destination(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "qpc"
            destination.mkdir()
            member = tarfile.TarInfo("model/programqpc.bin")
            with make_tar([(member, b"qpc")]) as archive:
                self.extract(archive, destination)
            self.assertEqual((destination / "model/programqpc.bin").read_bytes(), b"qpc")

    def test_rejects_parent_path_traversal(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "qpc"
            destination.mkdir()
            escaped_file = Path(temp_dir) / "escaped.txt"
            member = tarfile.TarInfo("../escaped.txt")
            with make_tar([(member, b"outside")]) as archive:
                with self.assertRaises(tarfile.ExtractError):
                    self.extract(archive, destination)
            self.assertFalse(escaped_file.exists())

    def test_rejects_symbolic_links(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "qpc"
            destination.mkdir()
            link = tarfile.TarInfo("link")
            link.type = tarfile.SYMTYPE
            link.linkname = "../outside"
            with make_tar([(link, b"")]) as archive:
                with self.assertRaises(tarfile.ExtractError):
                    self.extract(archive, destination)


if __name__ == "__main__":
    unittest.main()
