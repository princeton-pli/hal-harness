"""Tests for JsonEncryption raw file encrypt/decrypt round-trip."""

import json
import os
import tempfile

from hal.utils.json_encryption import JsonEncryption


class TestEncryptRawFileRoundTrip:
    def test_binary_content_survives_round_trip(self):
        original = b"\x00\x01\x02 binary data \xff\xfe"
        enc = JsonEncryption("test-password")

        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "data.bin")
            enc_path = os.path.join(d, "data.bin.encrypted")

            with open(src, "wb") as f:
                f.write(original)

            enc.encrypt_raw_file(src, enc_path)

            # Encrypted file should be valid JSON with expected keys
            with open(enc_path) as f:
                payload = json.load(f)
            assert "encrypted_data" in payload
            assert "salt" in payload

            # Decrypt via the instance method
            result = enc.decrypt_raw_file(payload["encrypted_data"], payload["salt"])
            assert result == original
        # Rationale: Core contract — arbitrary bytes encrypted then decrypted must equal the original.

    def test_text_content_survives_round_trip(self):
        original = b"plain text content\nwith newlines\n"
        enc = JsonEncryption("pw")

        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "readme.txt")
            enc_path = os.path.join(d, "readme.txt.encrypted")

            with open(src, "wb") as f:
                f.write(original)

            enc.encrypt_raw_file(src, enc_path)
            with open(enc_path) as f:
                payload = json.load(f)
            result = enc.decrypt_raw_file(payload["encrypted_data"], payload["salt"])
            assert result == original
        # Rationale: Covers the common case of encrypting non-JSON text files (e.g. JSONL).
