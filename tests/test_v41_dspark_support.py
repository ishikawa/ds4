#!/usr/bin/env python3
"""CPU-only binding/admission tests for V4.1 DSpark support GGUF files."""

import copy
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gguf-tools"))

import deepseek41_mtp_quantize as mtp
from deepseek41_metadata import GGUF_ALIGNMENT
from glm53_quantize import align, tensor_header


REVISION = "dba1be0a40aa45a94ad051997016db3960a90277"
INSPECTOR = ROOT / "tests" / "test_v41_dspark_support"


def released_config():
    return {
        "model_type": "deepseek_v41",
        "text_config": {
            "hidden_size": 5120,
            "moe_intermediate_size": 2304,
            "num_attention_heads": 64,
            "head_dim": 512,
            "q_lora_rank": 1280,
            "o_lora_rank": 1024,
            "o_groups": 8,
            "hc_mult": 4,
            "n_shared_experts": 1,
            "vocab_size": 129280,
            "num_nextn_predict_layers": 3,
            "dspark_block_size": 5,
            "dspark_noise_token_id": 128799,
            "dspark_target_layer_ids": [37, 38, 39],
            "dspark_markov_rank": 256,
            "dspark_n_routed_experts": 128,
            "dspark_num_experts_per_tok": 3,
        },
    }


def synthetic_index(config):
    schema = mtp.build_source_schema(config)
    return {"weight_map": {
        name: f"model-{44 + int(name.split('.')[1]):05d}-of-00048.safetensors"
        for name in schema
    }}


def write_sparse_support(path, config, *, missing=None, revision=REVISION):
    plan, _ = mtp.build_plan(config, synthetic_index(config))
    if missing is not None:
        plan = [item for item in plan if item.name != missing]
        assert len(plan) == 77
        offset = 0
        for item in plan:
            item.offset = offset
            offset += align(item.nbytes, GGUF_ALIGNMENT)
    records = mtp.support_metadata(config, revision)
    header = (b"GGUF" + struct.pack("<IQQ", 3, len(plan), len(records))
              + b"".join(records)
              + b"".join(tensor_header(item) for item in plan))
    data_start = align(len(header), GGUF_ALIGNMENT)
    data_bytes = max((item.offset + align(item.nbytes, GGUF_ALIGNMENT)
                      for item in plan), default=0)
    with path.open("wb") as fp:
        fp.write(header)
        fp.write(b"\0" * (data_start - len(header)))
        fp.truncate(data_start + data_bytes)
    return data_start + data_bytes


def inspect(path, revision=REVISION, budget=8 << 30):
    return subprocess.run(
        [str(INSPECTOR), str(path), revision, str(budget)],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=False)


class V41DSparkSupportTest(unittest.TestCase):
    def fixture_path(self, root, name):
        return Path(root) / name

    def test_valid_support_binds_all_v41_fields(self):
        with tempfile.TemporaryDirectory() as root:
            path = self.fixture_path(root, "valid.gguf")
            write_sparse_support(path, released_config())
            result = inspect(path)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("tensors=78", result.stdout)
            self.assertIn(
                "stages=3 experts=128 top_k=3 targets=3 markov_rank=256",
                result.stdout)
            self.assertIn(
                "hc_head=no missing=0 invalid=0 metadata_errors=0",
                result.stdout)

    def test_missing_tensor_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path = self.fixture_path(root, "missing.gguf")
            write_sparse_support(path, released_config(),
                                 missing="mtp.1.ffn_norm.weight")
            result = inspect(path)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("missing=1", result.stderr)

    def test_revision_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path = self.fixture_path(root, "revision.gguf")
            write_sparse_support(path, released_config())
            result = inspect(path, revision="0" * 40)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("source revision does not match", result.stderr)

    def test_resident_budget_shortfall_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path = self.fixture_path(root, "budget.gguf")
            file_bytes = write_sparse_support(path, released_config())
            result = inspect(path, budget=file_bytes - 1)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("support model needs", result.stderr)

    def test_invalid_expert_contract_is_rejected(self):
        cases = (
            (129, 3, "experts=129"),
            (128, 0, "expert top-k is outside 1..8"),
            (128, 9, "expert top-k is outside 1..8"),
        )
        for experts, top_k, error in cases:
            with self.subTest(experts=experts, top_k=top_k), \
                    tempfile.TemporaryDirectory() as root:
                config = copy.deepcopy(released_config())
                config["text_config"]["dspark_n_routed_experts"] = experts
                config["text_config"]["dspark_num_experts_per_tok"] = top_k
                path = self.fixture_path(
                    root, f"experts-{experts}-top-{top_k}.gguf")
                write_sparse_support(path, config)
                result = inspect(path)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(error, result.stderr)

    def test_released_support_gguf_binds_when_present(self):
        path = Path(
            "/Users/takanori.ishikawa/Developer/Workspace/local-llm/models/"
            "DeepSeek-V4.1-Flash-DSpark-support-q2.gguf")
        if not path.is_file():
            self.skipTest("released V4.1 support GGUF is not staged")
        result = inspect(path)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("tensors=78", result.stdout)
        self.assertIn(
            "hc_head=no missing=0 invalid=0 metadata_errors=0",
            result.stdout)


if __name__ == "__main__":
    unittest.main()
