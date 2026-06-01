"""Tests for hal.utils.decrypt — decrypt_raw and multi-file decrypt_file."""

import base64
import json
import tempfile
import zipfile
from pathlib import Path

from hal.utils.json_encryption import JsonEncryption
from hal.utils.decrypt import decrypt_raw, decrypt_file


class TestDecryptRaw:
    def test_returns_original_bytes(self):
        original = b"hello world"
        enc = JsonEncryption("hal1234")
        encrypted = enc.cipher.encrypt(original)

        salt_b64 = base64.b64encode(enc.salt).decode("utf-8")
        data_b64 = base64.b64encode(encrypted).decode("utf-8")

        result = decrypt_raw(data_b64, salt_b64)
        assert result == original
        # Rationale: decrypt_raw must invert the Fernet encryption using the hardcoded password.


class TestDecryptFileMultiEntry:
    def _make_encrypted_zip(self, tmp_dir, entries):
        """Helper: build a zip containing encrypted entries.

        entries: list of (filename, content_bytes, is_json)
        """
        zip_path = Path(tmp_dir) / "archive.zip"
        enc = JsonEncryption("hal1234")

        with zipfile.ZipFile(zip_path, "w") as zf:
            for filename, content, is_json in entries:
                if is_json:
                    encrypted = enc.cipher.encrypt(json.dumps(content).encode())
                else:
                    encrypted = enc.cipher.encrypt(content)
                payload = {
                    "encrypted_data": base64.b64encode(encrypted).decode("utf-8"),
                    "salt": base64.b64encode(enc.salt).decode("utf-8"),
                }
                zf.writestr(f"{filename}.encrypted", json.dumps(payload))
        return zip_path

    def test_decrypts_json_entry(self):
        with tempfile.TemporaryDirectory() as d:
            original = {"key": "value"}
            zip_path = self._make_encrypted_zip(d, [("data.json", original, True)])
            decrypt_file(zip_path)
            out = Path(d) / "data.json"
            assert out.exists()
            assert json.loads(out.read_text()) == original
        # Rationale: JSON entries should be decrypted and written as formatted JSON.

    def test_decrypts_raw_entry_when_not_valid_json(self):
        with tempfile.TemporaryDirectory() as d:
            raw_content = b"line1\nline2\n"
            zip_path = self._make_encrypted_zip(d, [("data.jsonl", raw_content, False)])
            decrypt_file(zip_path)
            out = Path(d) / "data.jsonl"
            assert out.exists()
            assert out.read_bytes() == raw_content
        # Rationale: Non-JSON content should fall back to raw bytes decryption.

    def test_decrypts_multiple_entries(self):
        with tempfile.TemporaryDirectory() as d:
            json_data = {"a": 1}
            raw_data = b"raw bytes"
            zip_path = self._make_encrypted_zip(
                d,
                [
                    ("results.json", json_data, True),
                    ("submissions.jsonl", raw_data, False),
                ],
            )
            decrypt_file(zip_path)
            assert json.loads((Path(d) / "results.json").read_text()) == json_data
            assert (Path(d) / "submissions.jsonl").read_bytes() == raw_data
        # Rationale: The PR changed decrypt_file to iterate over all zip entries.
