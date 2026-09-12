#!/usr/bin/env python3
"""Offline tests for shard-at-a-time DeepSeek V4.1 conversion."""

import contextlib
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

from deepseek41_quantize import QUANTIZATION, write_gguf
import deepseek41_stream_convert as stream
from deepseek41_stream_convert import (LocalFetcher, StreamingSourceDB,
                                       disk_peak, load_headers)
from glm53_quantize import (QTYPE_F32, SourceDB, TensorPlan, align, kv_string,
                            qtype_nbytes)


def write_safetensors(path, tensors):
    document = {}
    payload = bytearray()
    for name, array in tensors.items():
        data = np.asarray(array, dtype=np.float32).tobytes()
        document[name] = {"dtype": "F32", "shape": list(array.shape),
                          "data_offsets": [len(payload), len(payload) + len(data)]}
        payload.extend(data)
    header = json.dumps(document, separators=(",", ":")).encode()
    header += b" " * (-len(header) % 8)
    path.write_bytes(struct.pack("<Q", len(header)) + header + payload)


class StreamingConversionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        suffix = "dylib" if sys.platform == "darwin" else "so"
        cls.library = str(ROOT / "gguf-tools" / f"libds4quants.{suffix}")

    def fixture(self, root):
        source = root / "source"
        source.mkdir()
        names = ["test.0", "test.1", "test.2", "layers.1.engram.q_weight"]
        expert_names = [f"layers.0.ffn.experts.0.{part}.{kind}"
                        for part in ("w1", "w2", "w3")
                        for kind in ("weight", "scale")]
        write_safetensors(source / "shard-a.safetensors", {
            names[0]: np.arange(4, dtype=np.float32),
            **{name: np.arange(4, dtype=np.float32) for name in expert_names},
        })
        write_safetensors(source / "shard-b.safetensors", {
            names[1]: np.arange(4, dtype=np.float32) + 10,
            names[2]: np.arange(4, dtype=np.float32) + 20,
        })
        write_safetensors(source / "shard-c.safetensors", {
            "layers.1.engram.embed.weight": np.arange(32, dtype=np.float32),
            names[3]: np.arange(4, dtype=np.float32) + 30,
        })
        weight_map = {names[0]: "shard-a.safetensors",
                      names[1]: "shard-b.safetensors",
                      names[2]: "shard-b.safetensors",
                      "layers.1.engram.embed.weight": "shard-c.safetensors",
                      names[3]: "shard-c.safetensors"}
        weight_map.update({name: "shard-a.safetensors" for name in expert_names})
        index = {"metadata": {"total_size": sum(
                     (source / shard).stat().st_size for shard in set(weight_map.values()))},
                 "weight_map": weight_map}
        (source / "model.safetensors.index.json").write_text(json.dumps(index))
        (source / "config.json").write_text(json.dumps({
            "text_config": {"num_hidden_layers": 1, "n_routed_experts": 1}
        }))
        (source / "tokenizer.json").write_text("{}")
        plan = []
        offset = 0
        for name in names:
            item = TensorPlan(name, (4,), QTYPE_F32, "test", source=name)
            item.nbytes = qtype_nbytes(item.qtype, item.shape)
            item.offset = offset
            offset += align(item.nbytes, 16384)
            plan.append(item)
        records = [kv_string("general.architecture", "deepseek41"),
                   kv_string("deepseek41.quantization", QUANTIZATION["q2"]),
                   kv_string("deepseek41.calibration", "weight-energy bootstrap")]
        return source, plan, records

    def args(self, out, resume=False):
        return types.SimpleNamespace(out=str(out), imatrix=None,
                                     quants_library=self.library, threads=2,
                                     resume=resume, min_free_gib=0)

    def stream_db(self, source, work, out):
        (work / "model.safetensors.index.json").write_bytes(
            (source / "model.safetensors.index.json").read_bytes())
        fetcher = LocalFetcher(str(source))
        headers = load_headers(str(work / "model.safetensors.index.json"), fetcher,
                               str(out) + ".headers.json",
                               {"shard-a.safetensors", "shard-b.safetensors",
                                "shard-c.safetensors"},
                               "0" * 40)
        db = StreamingSourceDB(str(work), headers, fetcher, 1, 0,
                               str(out) + ".progress.json")
        return db, headers

    def test_stream_matches_batch_and_reports_peak(self):
        with tempfile.TemporaryDirectory() as tmp, \
             contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            source, plan, records = self.fixture(root)
            batch = root / "batch.gguf"
            db = SourceDB(str(source), index_validator=lambda _: None,
                          scale_validator=lambda _: None)
            write_gguf(self.args(batch), plan, records, db)
            db.close()

            work = root / "work"
            work.mkdir()
            streamed = root / "streamed.gguf"
            db, headers = self.stream_db(source, work, streamed)
            data_start = 16384
            db.set_plan(plan, data_start)
            peak = disk_peak(plan, db, headers, 1, data_start)
            self.assertGreater(peak, data_start)
            write_gguf(self.args(streamed), plan, records, db)
            db.close()
            self.assertEqual(streamed.read_bytes(), batch.read_bytes())
            self.assertFalse((work / "shard-a.safetensors").exists())
            self.assertFalse((work / "shard-b.safetensors").exists())
            self.assertFalse((work / "shard-c.safetensors").exists())
            progress = json.loads(Path(str(streamed) + ".progress.json").read_text())
            self.assertTrue(progress["complete"])
            self.assertEqual(progress["output_offset"], streamed.stat().st_size)

    def test_resume_after_completed_tensor(self):
        with tempfile.TemporaryDirectory() as tmp, \
             contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            source, plan, records = self.fixture(root)
            expected = root / "expected.gguf"
            batch_db = SourceDB(str(source), index_validator=lambda _: None,
                                scale_validator=lambda _: None)
            write_gguf(self.args(expected), plan, records, batch_db)
            batch_db.close()

            work = root / "resume-work"
            work.mkdir()
            output = root / "resumed.gguf"
            db, _ = self.stream_db(source, work, output)
            original = db.item_completed

            def interrupt(item, completed, output_offset):
                original(item, completed, output_offset)
                if completed == 2:
                    raise OSError("synthetic interruption")

            db.item_completed = interrupt
            with self.assertRaisesRegex(OSError, "synthetic interruption"):
                write_gguf(self.args(output), plan, records, db)
            db.close()
            self.assertTrue(Path(str(output) + ".partial").exists())
            self.assertTrue((work / "shard-b.safetensors.stream-owned").exists())

            db, _ = self.stream_db(source, work, output)
            write_gguf(self.args(output, resume=True), plan, records, db)
            db.close()
            self.assertEqual(output.read_bytes(), expected.read_bytes())
            self.assertFalse((work / "shard-b.safetensors").exists())

    def test_driver_matches_batch_conversion(self):
        with tempfile.TemporaryDirectory() as tmp, \
             contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            source, plan, records = self.fixture(root)
            expected = root / "expected.gguf"
            batch_db = SourceDB(str(source), index_validator=lambda _: None,
                                scale_validator=lambda _: None)
            write_gguf(self.args(expected), plan, records, batch_db)
            batch_db.close()

            work = root / "driver-work"
            output = root / "driver.gguf"
            args = types.SimpleNamespace(
                hf=str(work), out=str(output), repo="unused", source_revision="0" * 40,
                quant="q2", imatrix=None, threads=2, prefetch=1, min_free_gib=0,
                resume=False, dry_run=False, engram_q4k=["unused", "unused"],
                engram_q4k_dir=None, engram_q4k_external=False,
                source_dir=str(source), quants_library=self.library)
            sidecars = [types.SimpleNamespace(path="unused", offset=0)] * 2
            base_records = [kv_string("general.architecture", "deepseek41")]
            with mock.patch.object(stream, "load_engram_q4k", return_value=sidecars), \
                 mock.patch.object(stream, "metadata",
                                   return_value=({"text_config": {}}, base_records)), \
                 mock.patch.object(stream, "build_plan", return_value=plan):
                stream.run(args)
            self.assertEqual(output.read_bytes(), expected.read_bytes())
            self.assertTrue(json.loads(Path(str(output) + ".progress.json").read_text())
                            ["complete"])


if __name__ == "__main__":
    unittest.main()
