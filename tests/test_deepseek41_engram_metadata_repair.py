#!/usr/bin/env python3
"""Tests for in-place external Engram metadata repair."""

import contextlib
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "gguf-tools"
sys.path.insert(0, str(TOOLS))

import deepseek41_engram_metadata_repair as repair_tool
import deepseek41_validate_gguf as validator
import deepseek41_metadata as metadata_module
from deepseek41_metadata import metadata
from glm53_quantize import (
    QTYPE_F32, TensorPlan, align, kv_string, kv_u32, pack_string, tensor_header,
)


class EmptyDB:
    tensors = {}

    def close(self):
        pass


class MetadataRepairTests(unittest.TestCase):
    def make_source(self, directory):
        (directory / "tokenizer.json").write_text(json.dumps({
            "model": {"type": "BPE", "merges": []}, "added_tokens": [],
        }))
        text = {
            "vocab_size": 2, "hidden_size": 16, "moe_intermediate_size": 8,
            "num_hidden_layers": 2, "num_attention_heads": 1,
            "num_key_value_heads": 1, "head_dim": 16, "qk_rope_head_dim": 8,
            "q_lora_rank": 8, "o_lora_rank": 8, "o_groups": 1,
            "n_routed_experts": 2, "n_shared_experts": 1,
            "num_experts_per_tok": 1, "max_position_embeddings": 128,
            "sliding_window": 16, "index_n_heads": 1, "index_head_dim": 8,
            "index_topk": 1, "candidate_source_layer_id": 0,
            "candidate_topk_blocks": 1, "candidate_block_size": 1,
            "hc_mult": 1, "hc_sinkhorn_iters": 1, "rope_theta": 10000,
            "compress_rope_theta": 10000, "rms_norm_eps": 1e-6,
            "hc_eps": 1e-6, "swiglu_limit": 7.0,
            "routed_scaling_factor": 1.0, "scoring_func": "sigmoid",
            "hidden_act": "silu", "topk_method": "noaux_tc",
            "norm_topk_prob": True, "compress_ratios": [1, 1],
            "kv_source_layer_ids": [0], "index_source_layer_ids": [0],
            "rope_scaling": {"factor": 1.0, "beta_fast": 32.0,
                             "beta_slow": 1.0,
                             "original_max_position_embeddings": 128.0},
            "engram_layer_ids": [1, 14], "engram_num_embeddings": [3, 4],
            "engram_max_ngram_size": 4, "engram_n_heads": 8,
            "engram_head_dim": 256, "engram_vocab_size": 5,
            "engram_compressed_vocab_size": 2, "engram_pad_token_id": 0,
        }
        config = {"model_type": "deepseek_v41", "bos_token_id": 0,
                  "eos_token_id": 1, "text_config": text}
        (directory / "config.json").write_text(json.dumps(config))
        return config

    def make_sidecars(self, directory, config):
        result = []
        text = config["text_config"]
        for layer, rows in zip(text["engram_layer_ids"],
                               text["engram_num_embeddings"]):
            path = directory / f"engram-{layer}.q4_k.bin"
            path.write_bytes(bytes((layer + index) & 255 for index in range(rows * 144)))
            Path(str(path) + ".json").write_text(json.dumps({
                "rows": rows, "bytes_per_row": 144, "cols": 256,
                "block_type": "Q4_K", "ggml_type": 12,
                "format_version": 1, "complete": True,
            }))
            result.append(path)
        return result

    def make_old_gguf(self, path, sidecars, alignment=16384):
        class FakeTokenizer:
            @classmethod
            def from_file(cls, _path):
                return cls()

            def id_to_token(self, token_id):
                return ("a", "b")[token_id]

        fake_tokenizers = types.SimpleNamespace(Tokenizer=FakeTokenizer)
        layout = {
            "token_map": [0, 1], "compressed_vocab_size": 2, "pad_id": 0,
            "rows": [3, 4], "layers": [1, 14],
            "primes": [[1] * 24, [2] * 24],
            "multipliers": [[1] * 4, [3] * 4],
        }
        with mock.patch.dict(sys.modules, {"tokenizers": fake_tokenizers}), \
             mock.patch.object(metadata_module, "engram_layout", return_value=layout):
            _, records = metadata(
                path.parent, "0" * 40, "q4_k_row144", "external",
                [str(item.resolve()) for item in sidecars], [0, 0],
                sidecar_integrity=None)
        if alignment != 16384:
            prefix = pack_string("general.alignment")
            records = [kv_u32("general.alignment", alignment)
                       if record.startswith(prefix) else record for record in records]
        records.extend([
            kv_string("deepseek41.calibration", "weight-energy bootstrap"),
            kv_string("deepseek41.quantization",
                      "IQ2_XXS gate/up; Q2_K down; Q8_0 attention/shared/head"),
        ])
        item = TensorPlan("dummy.weight", (1,), QTYPE_F32, "regular")
        item.nbytes = 4
        item.offset = 0
        raw = (b"GGUF" + struct.pack("<IQQ", 3, 1, len(records))
               + b"".join(records) + tensor_header(item))
        data_start = align(len(raw), alignment)
        payload = b"DATA" + bytes(align(4, alignment) - 4)
        path.write_bytes(raw + bytes(data_start - len(raw)) + payload)
        return item, data_start

    def run_tool(self, gguf, sidecar_dir, *extra):
        return subprocess.run(
            [sys.executable, str(TOOLS / "deepseek41_engram_metadata_repair.py"),
             str(gguf), "--engram-q4k-dir", str(sidecar_dir), *extra],
            text=True, capture_output=True, check=False)

    def test_repair_preserves_tensor_payload_and_validates(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            config = self.make_source(directory)
            sidecars = self.make_sidecars(directory, config)
            gguf = directory / "old.gguf"
            item, data_start = self.make_old_gguf(gguf, sidecars)
            original = gguf.read_bytes()
            old_records = dict(repair_tool.read_header(gguf)["records"])
            dry_run = self.run_tool(gguf, directory, "--relative", "--dry-run")
            self.assertEqual(dry_run.returncode, 0, dry_run.stderr)
            self.assertIn("header: fits", dry_run.stdout)
            self.assertIn("added deepseek41.engram.sidecar_sample_sha256", dry_run.stdout)
            self.assertFalse(Path(str(gguf) + ".header.bak").exists())

            result = self.run_tool(gguf, directory, "--relative")
            self.assertEqual(result.returncode, 0, result.stderr)
            backup = Path(str(gguf) + ".header.bak")
            self.assertEqual(backup.read_bytes(), original[:data_start])
            repaired = gguf.read_bytes()
            self.assertEqual(repaired[data_start:], original[data_start:])
            self.assertEqual(repair_tool.read_header(gguf)["data_start"], data_start)
            new_records = dict(repair_tool.read_header(gguf)["records"])
            for key, record in old_records.items():
                if key != "deepseek41.engram.external_paths":
                    self.assertEqual(new_records[key], record)

            args = types.SimpleNamespace(
                hf=str(directory), gguf=str(gguf), source_revision="0" * 40,
                payload=False, imatrix=None, full=False, quant="q2",
                quants_library="unused", engram_q4k=None,
                engram_q4k_dir=str(directory), engram_q4k_external=True,
                engram_q4k_relative=True,
            )
            output = io.StringIO()
            with mock.patch.object(validator, "SourceDB", return_value=EmptyDB()), \
                 mock.patch.object(validator, "build_plan", return_value=[item]), \
                 contextlib.redirect_stdout(output):
                validator.validate(args)
            self.assertIn("PASS:", output.getvalue())

    def test_repair_refuses_header_growth_past_tensor_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            config = self.make_source(directory)
            sidecars = self.make_sidecars(directory, config)
            gguf = directory / "tight.gguf"
            self.make_old_gguf(gguf, sidecars, alignment=32)
            original = gguf.read_bytes()
            result = self.run_tool(gguf, directory, "--absolute")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("tensor data relocation is required", result.stderr)
            self.assertEqual(gguf.read_bytes(), original)
            self.assertFalse(Path(str(gguf) + ".header.bak").exists())

    def test_unknown_metadata_type_is_rejected_without_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            config = self.make_source(directory)
            sidecars = self.make_sidecars(directory, config)
            gguf = directory / "unknown.gguf"
            self.make_old_gguf(gguf, sidecars)
            original = bytearray(gguf.read_bytes())
            marker = struct.pack("<Q", len("general.architecture")) + b"general.architecture"
            kind_offset = original.index(marker) + len(marker)
            struct.pack_into("<I", original, kind_offset, 99)
            gguf.write_bytes(original)
            result = self.run_tool(gguf, directory, "--relative")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("unsupported GGUF value type 99", result.stderr)
            self.assertEqual(gguf.read_bytes(), original)
            self.assertFalse(Path(str(gguf) + ".header.bak").exists())


if __name__ == "__main__":
    unittest.main()
