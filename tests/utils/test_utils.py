"""Tests for hal.utils.utils — new public functions added in this PR."""

import os
import tempfile

from hal.utils.utils import compute_agent_dir_hash


class TestComputeAgentDirHash:
    def test_returns_deterministic_hex_string(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "a.py"), "w") as f:
                f.write("hello")
            h1 = compute_agent_dir_hash(d)
            h2 = compute_agent_dir_hash(d)
        assert isinstance(h1, str)
        assert len(h1) == 64  # SHA256 hex length
        assert h1 == h2
        # Rationale: Determinism is the core contract — same dir, same hash.

    def test_hash_changes_when_content_changes(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "a.py")
            with open(path, "w") as f:
                f.write("v1")
            h1 = compute_agent_dir_hash(d)
            with open(path, "w") as f:
                f.write("v2")
            h2 = compute_agent_dir_hash(d)
        assert h1 != h2
        # Rationale: Hash must reflect content changes to be useful as a version fingerprint.

    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as d:
            h = compute_agent_dir_hash(d)
        assert isinstance(h, str) and len(h) == 64
        # Rationale: Edge case — no files should still produce a valid hash.
