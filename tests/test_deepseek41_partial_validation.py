#!/usr/bin/env python3
"""Tests for validation of an active streaming GGUF prefix."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gguf-tools"))

import deepseek41_validate_partial as partial_validator
from deepseek41_metadata import array_record, ENGRAM_SAMPLE_SCHEME, GGUF_ALIGNMENT
from deepseek41_quantize import QUANTIZATION
from glm53_manifest import load_safetensors_header
from glm53_quantize import (
    GGUF_STRING, GGUF_UINT64, QTYPE_F32, TensorPlan, align, kv_string, kv_u32,
    tensor_header,
)


def write_safetensors(path, tensors):
    document = {}
    payload = bytearray()
    for name, values in tensors.items():
        data = np.asarray(values, dtype="<f4").tobytes()
        document[name] = {"dtype": "F32", "shape": list(values.shape),
                          "data_offsets": [len(payload), len(payload) + len(data)]}
        payload.extend(data)
    header = json.dumps(document, separators=(",", ":")).encode()
    header += b" " * (-len(header) % 8)
    path.write_bytes(struct.pack("<Q", len(header)) + header + payload)


class PartialValidationTests(unittest.TestCase):
    def fixture(self, root, version=1, integrity_metadata=False):
        hf = root / "hf"
        hf.mkdir()
        shard_a = hf / "shard-a.safetensors"
        shard_b = hf / "shard-b.safetensors"
        arrays = {
            "test.0": np.arange(4, dtype=np.float32),
            "test.1": np.arange(4, dtype=np.float32) + 10,
            "test.2": np.arange(4, dtype=np.float32) + 20,
        }
        write_safetensors(shard_a, {"test.0": arrays["test.0"]})
        write_safetensors(shard_b, {
            "test.1": arrays["test.1"], "test.2": arrays["test.2"],
        })
        index = {"metadata": {"total_size": shard_a.stat().st_size + shard_b.stat().st_size},
                 "weight_map": {"test.0": shard_a.name, "test.1": shard_b.name,
                                "test.2": shard_b.name}}
        index_path = hf / "model.safetensors.index.json"
        index_path.write_text(json.dumps(index))
        (hf / "config.json").write_text(json.dumps({"text_config": {}}))
        headers = {
            shard_a.name: {"size": shard_a.stat().st_size,
                           "tensors": load_safetensors_header(shard_a)},
            shard_b.name: {"size": shard_b.stat().st_size,
                           "tensors": load_safetensors_header(shard_b)},
        }

        plan = []
        for index, name in enumerate(arrays):
            item = TensorPlan(name, (4,), QTYPE_F32, "test", source=name)
            item.nbytes = 16
            item.offset = index * GGUF_ALIGNMENT
            plan.append(item)
        sidecar_paths = [str((root / "engram-1.bin").resolve()),
                         str((root / "engram-14.bin").resolve())]
        integrity = [
            {"size": 432, "rows": 3, "sample_sha256": "a" * 64},
            {"size": 576, "rows": 4, "sample_sha256": "b" * 64},
        ]
        records = [
            kv_string("general.architecture", "deepseek41"),
            kv_u32("general.alignment", GGUF_ALIGNMENT),
            kv_string("general.source.revision", "0" * 40),
            kv_string("deepseek41.quantization", QUANTIZATION["q2"]),
            kv_string("deepseek41.calibration", "weight-energy bootstrap"),
            kv_string("deepseek41.engram.encoding", "q4_k_row144"),
            kv_string("deepseek41.engram.storage", "external"),
            array_record("deepseek41.engram.external_paths", GGUF_STRING,
                         sidecar_paths),
            array_record("deepseek41.engram.external_offsets", GGUF_UINT64, [0, 0]),
        ]
        if integrity_metadata:
            records.extend([
                array_record("deepseek41.engram.sidecar_size", GGUF_UINT64,
                             [item["size"] for item in integrity]),
                array_record("deepseek41.engram.sidecar_rows", GGUF_UINT64,
                             [item["rows"] for item in integrity]),
                array_record("deepseek41.engram.sidecar_sample_sha256", GGUF_STRING,
                             [item["sample_sha256"] for item in integrity]),
                kv_string("deepseek41.engram.sidecar_sample_scheme",
                          ENGRAM_SAMPLE_SCHEME),
            ])
        raw_header = (b"GGUF" + struct.pack("<IQQ", 3, len(plan), len(records))
                      + b"".join(records)
                      + b"".join(tensor_header(item) for item in plan))
        data_start = align(len(raw_header), GGUF_ALIGNMENT)
        gguf = root / "model.gguf.partial"
        payload = bytearray()
        for name in ("test.0", "test.1"):
            data = arrays[name].tobytes()
            payload.extend(data + bytes(GGUF_ALIGNMENT - len(data)))
        output_offset = data_start + len(payload)
        gguf.write_bytes(raw_header + bytes(data_start - len(raw_header))
                         + payload + b"UNFINISHED-GARBAGE")

        headers_path = root / "model.gguf.headers.json"
        headers_path.write_text(json.dumps({
            "version": 1, "revision": "0" * 40,
            "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
            "shards": headers,
        }))
        completed_shards = [shard_b.name] if version == 1 else {shard_b.name: "b" * 64}
        progress_path = root / "model.gguf.progress.json"
        progress_path.write_text(json.dumps({
            "version": version, "complete": False,
            "completed_tensor_count": 2,
            "completed_tensors": ["test.0", "test.1"],
            "completed_shards": completed_shards,
            "output_offset": output_offset,
        }))
        shard_b.unlink()
        sidecars = [types.SimpleNamespace(
            layer=layer, path=path, offset=0,
            integrity=lambda _embedded, value=value: value)
            for layer, path, value in zip((1, 14), sidecar_paths, integrity)]
        args = types.SimpleNamespace(
            gguf=str(gguf), progress=str(progress_path), hf=str(hf),
            headers=str(headers_path), source_revision="0" * 40, quant="q2",
            imatrix=None, limit=None, skip_experts=False, all_names=False,
            engram_q4k=None,
            engram_q4k_dir=str(root), engram_q4k_external=True,
            quants_library=str(ROOT / "gguf-tools" / "libds4quants.dylib"),
        )
        return args, plan, sidecars, data_start

    def run_validation(self, args, plan, sidecars):
        output = io.StringIO()
        with mock.patch.object(partial_validator, "build_plan", return_value=plan), \
             mock.patch.object(partial_validator, "load_engram_q4k",
                               return_value=sidecars), \
             contextlib.redirect_stdout(output):
            partial_validator.validate(args)
        return output.getvalue()

    def test_v1_and_v2_progress_validate_only_the_durable_prefix(self):
        for version in (1, 2):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as tmp:
                args, plan, sidecars, _ = self.fixture(Path(tmp), version)
                output = self.run_validation(args, plan, sidecars)
                self.assertIn("plan: matched=3 mismatched=0 total=3", output)
                self.assertIn(f"progress: version={version} completed=2", output)
                self.assertIn("payload: matched=1 mismatched=0 not_compared=1", output)
                self.assertIn("test.0", output)
                self.assertIn("not compared (shard already deleted): 1 tensors", output)
                self.assertIn("Engram: integrity metadata absent", output)
                self.assertIn("PASS: durable partial GGUF prefix validated", output)

    def test_payload_corruption_reports_tensor_and_offset(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, plan, sidecars, data_start = self.fixture(Path(tmp))
            with open(args.gguf, "r+b") as fp:
                fp.seek(data_start)
                byte = fp.read(1)
                fp.seek(data_start)
                fp.write(bytes([byte[0] ^ 1]))
            output = io.StringIO()
            with mock.patch.object(partial_validator, "build_plan", return_value=plan), \
                 mock.patch.object(partial_validator, "load_engram_q4k",
                                   return_value=sidecars), \
                 contextlib.redirect_stdout(output), \
                 self.assertRaisesRegex(ValueError, "tensor payloads differ"):
                partial_validator.validate(args)
            self.assertIn("payload mismatch: test.0 tensor_offset=0", output.getvalue())
            self.assertIn(f"GGUF offset {data_start}", output.getvalue())

    def test_downloading_source_is_skipped_and_integrity_metadata_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, plan, sidecars, _ = self.fixture(
                Path(tmp), version=2, integrity_metadata=True)
            shard = Path(args.hf) / "shard-a.safetensors"
            shard.rename(str(shard) + ".download")
            output = self.run_validation(args, plan, sidecars)
            self.assertIn("payload: matched=0 mismatched=0 not_compared=2", output)
            self.assertIn("not compared (shard still downloading): 1 tensors", output)
            self.assertIn("not compared (shard already deleted): 1 tensors", output)
            self.assertIn("Engram: integrity metadata present; size, rows and sample sha256 match",
                          output)


if __name__ == "__main__":
    unittest.main()
