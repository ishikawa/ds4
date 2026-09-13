#!/usr/bin/env python3
"""Offline synthetic tests for the DeepSeek V4.1 MTP support converter."""

import contextlib
import io
import json
import os
from pathlib import Path
import struct
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gguf-tools"))

import deepseek41_mtp_quantize as mtp
from deepseek41_quantize import write_gguf
from glm53_quantize import SourceDB


def tiny_config():
    return {
        "model_type": "deepseek_v41",
        "text_config": {
            "hidden_size": 256,
            "moe_intermediate_size": 256,
            "num_attention_heads": 1,
            "head_dim": 256,
            "q_lora_rank": 256,
            "o_lora_rank": 256,
            "o_groups": 1,
            "hc_mult": 1,
            "n_shared_experts": 1,
            "vocab_size": 256,
            "num_nextn_predict_layers": 3,
            "dspark_block_size": 5,
            "dspark_noise_token_id": 127,
            "dspark_target_layer_ids": [37, 38, 39],
            "dspark_markov_rank": 256,
            "dspark_n_routed_experts": 2,
            "dspark_num_experts_per_tok": 1,
        },
    }


def tensor_bytes(spec):
    value = 127 if spec.dtype == "F8_E8M0" else 0
    return bytes([value]) * spec.nbytes


def write_safetensors(path, tensors, schema):
    document = {}
    payload = bytearray()
    for name in sorted(tensors):
        spec = schema[name]
        data = tensor_bytes(spec)
        document[name] = {
            "dtype": spec.dtype,
            "shape": list(spec.shape),
            "data_offsets": [len(payload), len(payload) + len(data)],
        }
        payload.extend(data)
    header = json.dumps(document, separators=(",", ":")).encode()
    header += b" " * (-len(header) % 8)
    path.write_bytes(struct.pack("<Q", len(header)) + header + payload)


def fixture(root):
    config = tiny_config()
    schema = mtp.build_source_schema(config)
    weight_map = {}
    for stage in range(3):
        shard = f"model-{stage + 44:05d}-of-00048.safetensors"
        names = {name for name in schema if name.startswith(f"mtp.{stage}.")}
        write_safetensors(root / shard, names, schema)
        weight_map.update({name: shard for name in names})
    index = {
        "metadata": {"total_size": sum(
            (root / shard).stat().st_size for shard in set(weight_map.values()))},
        "weight_map": weight_map,
    }
    (root / "config.json").write_text(json.dumps(config))
    (root / "model.safetensors.index.json").write_text(json.dumps(index))
    return config, index, schema


def test_dry_plan_without_shards_and_manifest_validation(tmp_path):
    config = tiny_config()
    schema = mtp.build_source_schema(config)
    index = {"metadata": {"total_size": 123}, "weight_map": {
        name: f"model-{44 + int(name.split('.')[1]):05d}-of-00048.safetensors"
        for name in schema
    }}
    plan, actual_schema = mtp.build_plan(config, index)
    assert actual_schema == schema
    assert len(plan) == 78
    assert len(schema) == 133
    assert {item.expert_count for item in plan if item.is_expert} == {2}
    assert next(item for item in plan if item.name.endswith("ffn_gate_exps.weight")).qtype == mtp.QTYPE_IQ2_XXS
    assert next(item for item in plan if item.name.endswith("ffn_down_exps.weight")).qtype == mtp.QTYPE_Q2_K
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        mtp.dry_run_report(plan, schema, index,
                           mtp.support_metadata(config, "0" * 40), str(tmp_path), 7.5)
    text = output.getvalue()
    assert "file_bytes=missing" in text
    assert "temporary_bytes_starting_without_shards_payload_floor:" in text
    assert "recommended_free_bytes_with_7.5gib_reserve:" in text

    del index["weight_map"][next(iter(schema))]
    with pytest.raises(ValueError, match="index/schema mismatch"):
        mtp.build_plan(config, index)


def test_synthetic_correct_dtype_conversion(tmp_path):
    config, index, schema = fixture(tmp_path)
    db = SourceDB(str(tmp_path), index_validator=lambda _: None,
                  scale_validator=lambda _: None,
                  skip_tensors=lambda name: not name.startswith("mtp."))
    try:
        plan, _ = mtp.build_plan(config, index, db)
        out = tmp_path / "support.gguf"
        suffix = "dylib" if sys.platform == "darwin" else "so"
        args = types.SimpleNamespace(
            out=str(out), imatrix=None,
            quants_library=str(ROOT / "gguf-tools" / f"libds4quants.{suffix}"),
            threads=2, resume=False, min_free_gib=0)
        with contextlib.redirect_stdout(io.StringIO()):
            write_gguf(args, plan, mtp.support_metadata(config, "0" * 40), db)
        data = out.read_bytes()
        assert data[:4] == b"GGUF"
        assert struct.unpack_from("<I", data, 4)[0] == 3
        assert struct.unpack_from("<Q", data, 8)[0] == len(plan)
        assert b"mtp.2.markov_head.markov_w1.weight" in data
        assert b"mtp.2.markov_head.markov_w2.weight" in data
    finally:
        db.close()


def test_released_stage_headers_match_schema_when_present():
    source = Path(os.environ.get(
        "DS4_V41_SOURCE",
        "/Users/takanori.ishikawa/Developer/Workspace/local-llm/models/ds41-source"))
    config_path = source / "config.json"
    index_path = source / "model.safetensors.index.json"
    if not config_path.is_file() or not index_path.is_file():
        pytest.skip("released V4.1 index is not staged")
    config, index = mtp.load_source_documents(str(source))
    schema = mtp.build_source_schema(config)
    mtp.validate_manifest(index, schema)
    # SourceDB requires all three MTP shards. Header validation becomes active
    # automatically once the requester has completed the staged download.
    shards = {index["weight_map"][name] for name in schema}
    if not all((source / shard).is_file() for shard in shards):
        pytest.skip("released MTP shards are not all staged")
    db = SourceDB(str(source), index_validator=lambda _: None,
                  scale_validator=lambda _: None,
                  skip_tensors=lambda name: not name.startswith("mtp."))
    try:
        mtp.validate_source_db(db, schema)
    finally:
        db.close()
