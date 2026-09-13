#!/usr/bin/env python3
"""Convert the three DeepSeek V4.1 MTP/DSpark stages to a support GGUF.

The ordinary converter intentionally omits ``mtp.*``.  This companion tool
only opens the shards named by those index entries.  Its dry-run path needs no
safetensors payloads: released shapes and dtypes are derived from config.json
and checked against the complete MTP name manifest in the index.
"""

import argparse
import dataclasses
import json
import math
import os
import re
import sys

from deepseek41_metadata import GGUF_ALIGNMENT
from deepseek41_quantize import write_gguf
from glm53_quantize import (
    SourceDB, TensorPlan, QTYPE_NAMES, QTYPE_F32, QTYPE_F16, QTYPE_Q8_0, QTYPE_Q2_K,
    QTYPE_IQ2_XXS, align, kv_string, kv_u32, kv_u32_array, print_plan,
    qtype_nbytes,
)


SOURCE_BYTES = {"F32": 4, "BF16": 2, "F8_E4M3": 1, "F8_E8M0": 1, "I8": 1}
QUANTIZATION = "IQ2_XXS routed gate/up; Q2_K routed down; Q8_0 dense"


@dataclasses.dataclass(frozen=True)
class SourceSpec:
    shape: tuple[int, ...]
    dtype: str
    auxiliary: bool = False

    @property
    def nbytes(self):
        return math.prod(self.shape) * SOURCE_BYTES[self.dtype]


def load_source_documents(hf_dir):
    with open(os.path.join(hf_dir, "config.json"), "rb") as fp:
        config = json.load(fp)
    with open(os.path.join(hf_dir, "model.safetensors.index.json"), "rb") as fp:
        index = json.load(fp)
    if config.get("model_type") != "deepseek_v41":
        raise ValueError("not a DeepSeek V4.1 checkpoint")
    if not isinstance(index.get("weight_map"), dict):
        raise ValueError("safetensors index has no weight_map")
    return config, index


def build_source_schema(config):
    c = config["text_config"]
    stages = c["num_nextn_predict_layers"]
    if stages != 3:
        raise ValueError(f"expected three MTP stages, got {stages}")
    dim, inter = c["hidden_size"], c["moe_intermediate_size"]
    heads, head_dim = c["num_attention_heads"], c["head_dim"]
    qrank, orank, groups = c["q_lora_rank"], c["o_lora_rank"], c["o_groups"]
    hc = c["hc_mult"]
    experts = c["dspark_n_routed_experts"]
    vocab, rank = c["vocab_size"], c["dspark_markov_rank"]
    mix = hc * (hc + 2)
    schema = {}

    def add(name, shape, dtype, auxiliary=False):
        if name in schema:
            raise ValueError(f"duplicate MTP source schema entry: {name}")
        schema[name] = SourceSpec(tuple(shape), dtype, auxiliary)

    def fp8(name, shape):
        add(name, shape, "F8_E4M3")
        add(name.removesuffix(".weight") + ".scale",
            tuple((value + 31) // 32 for value in shape), "F8_E8M0", True)

    for stage in range(stages):
        src = f"mtp.{stage}"
        for site in ("attn", "ffn"):
            add(f"{src}.hc_{site}_fn", (mix, hc * dim), "F32")
            add(f"{src}.hc_{site}_base", (mix,), "F32")
            add(f"{src}.hc_{site}_scale", (3,), "F32")
            add(f"{src}.{site}_norm.weight", (dim,), "BF16")

        add(f"{src}.attn.attn_sink", (heads,), "F32")
        fp8(f"{src}.attn.wq_a.weight", (qrank, dim))
        fp8(f"{src}.attn.wq_b.weight", (heads * head_dim, qrank))
        add(f"{src}.attn.q_norm.weight", (qrank,), "BF16")
        fp8(f"{src}.attn.wkv.weight", (head_dim, dim))
        add(f"{src}.attn.kv_norm.weight", (head_dim,), "BF16")
        fp8(f"{src}.attn.wo_a.weight",
            (groups * orank, heads * head_dim // groups))
        fp8(f"{src}.attn.wo_b.weight", (dim, groups * orank))

        add(f"{src}.ffn.gate.weight", (experts, dim), "BF16")
        add(f"{src}.ffn.gate.bias", (experts,), "F32")
        add(f"{src}.ffn.gate.bias_vl", (experts,), "F32", True)
        for part, out_dim, in_dim in (
                ("w1", inter, dim), ("w3", inter, dim), ("w2", dim, inter)):
            fp8(f"{src}.ffn.shared_experts.{part}.weight", (out_dim, in_dim))
            for expert in range(experts):
                base = f"{src}.ffn.experts.{expert}.{part}"
                add(f"{base}.weight", (out_dim, in_dim // 2), "I8")
                add(f"{base}.scale", (out_dim, in_dim // 32), "F8_E8M0", True)

    add("mtp.0.main_proj.weight", (dim, dim * len(c["dspark_target_layer_ids"])),
        "F8_E4M3")
    add("mtp.0.main_proj.scale",
        ((dim + 31) // 32,
         (dim * len(c["dspark_target_layer_ids"]) + 31) // 32),
        "F8_E8M0", True)
    add("mtp.0.main_norm.weight", (dim,), "BF16")
    final = stages - 1
    add(f"mtp.{final}.norm.weight", (dim,), "BF16")
    add(f"mtp.{final}.markov_head.embed.weight", (vocab, rank), "BF16")
    add(f"mtp.{final}.markov_head.head.weight", (vocab, rank), "BF16")
    add(f"mtp.{final}.confidence_head.proj.weight", (1, dim + rank), "BF16")
    return schema


def validate_manifest(index, schema):
    indexed = {name for name in index["weight_map"] if name.startswith("mtp.")}
    expected = set(schema)
    if indexed != expected:
        missing = sorted(expected - indexed)
        unknown = sorted(indexed - expected)
        raise ValueError(
            f"MTP index/schema mismatch: missing={missing[:3]} unknown={unknown[:3]}")


def validate_source_db(db, schema):
    for name, expected in schema.items():
        actual = db.info(name)
        if actual["dtype"] != expected.dtype or actual["shape"] != list(expected.shape):
            raise ValueError(
                f"{name}: got {actual['dtype']} {actual['shape']}, "
                f"expected {expected.dtype} {list(expected.shape)}")


def build_plan(config, index, db=None):
    schema = build_source_schema(config)
    validate_manifest(index, schema)
    if db is not None:
        validate_source_db(db, schema)
    c = config["text_config"]
    stages = c["num_nextn_predict_layers"]
    dim, inter = c["hidden_size"], c["moe_intermediate_size"]
    heads, head_dim = c["num_attention_heads"], c["head_dim"]
    qrank, orank, groups = c["q_lora_rank"], c["o_lora_rank"], c["o_groups"]
    hc, experts = c["hc_mult"], c["dspark_n_routed_experts"]
    vocab, rank = c["vocab_size"], c["dspark_markov_rank"]
    mix = hc * (hc + 2)
    plan = []

    def regular(name, source, hf_shape, qtype, role):
        plan.append(TensorPlan(name, tuple(reversed(hf_shape)), qtype, role,
                               source=source))

    for stage in range(stages):
        src, dst = f"mtp.{stage}", f"mtp.{stage}"
        for site in ("attn", "ffn"):
            regular(f"{dst}.hc_{site}_fn.weight", f"{src}.hc_{site}_fn",
                    (mix, hc * dim), QTYPE_F16, "mhc")
            regular(f"{dst}.hc_{site}_base.weight", f"{src}.hc_{site}_base",
                    (mix,), QTYPE_F32, "mhc")
            regular(f"{dst}.hc_{site}_scale.weight", f"{src}.hc_{site}_scale",
                    (3,), QTYPE_F32, "mhc")
            regular(f"{dst}.{site}_norm.weight", f"{src}.{site}_norm.weight",
                    (dim,), QTYPE_F32, "norm")
        for target, source, shape in (
            ("attn_sinks.weight", "attn.attn_sink", (heads,)),
            ("attn_q_a.weight", "attn.wq_a.weight", (qrank, dim)),
            ("attn_q_b.weight", "attn.wq_b.weight", (heads * head_dim, qrank)),
            ("attn_q_a_norm.weight", "attn.q_norm.weight", (qrank,)),
            ("attn_kv.weight", "attn.wkv.weight", (head_dim, dim)),
            ("attn_kv_a_norm.weight", "attn.kv_norm.weight", (head_dim,)),
            ("attn_output_a.weight", "attn.wo_a.weight",
             (groups * orank, heads * head_dim // groups)),
            ("attn_output_b.weight", "attn.wo_b.weight", (dim, groups * orank)),
        ):
            qtype = QTYPE_F32 if len(shape) == 1 else QTYPE_Q8_0
            regular(f"{dst}.{target}", f"{src}.{source}", shape, qtype,
                    "attention")
        regular(f"{dst}.ffn_gate_inp.weight", f"{src}.ffn.gate.weight",
                (experts, dim), QTYPE_F32, "router")
        regular(f"{dst}.exp_probs_b.bias", f"{src}.ffn.gate.bias",
                (experts,), QTYPE_F32, "router")
        for target, source, hf_shape in (
            ("ffn_gate_shexp.weight", "w1", (inter, dim)),
            ("ffn_up_shexp.weight", "w3", (inter, dim)),
            ("ffn_down_shexp.weight", "w2", (dim, inter)),
        ):
            regular(f"{dst}.{target}", f"{src}.ffn.shared_experts.{source}.weight",
                    hf_shape, QTYPE_Q8_0, "shared")
        for target, source, hf_shape, qtype, part in (
            ("ffn_gate_exps.weight", "w1", (inter, dim), QTYPE_IQ2_XXS, "gate"),
            ("ffn_up_exps.weight", "w3", (inter, dim), QTYPE_IQ2_XXS, "up"),
            ("ffn_down_exps.weight", "w2", (dim, inter), QTYPE_Q2_K, "down"),
        ):
            item = TensorPlan(
                f"{dst}.{target}", (*reversed(hf_shape), experts), qtype,
                "experts", source=f"{src}.ffn.experts.{{expert}}.{source}.weight",
                expert_layer=stage, expert_part=part, expert_count=experts)
            plan.append(item)

    regular("mtp.0.main_proj.weight", "mtp.0.main_proj.weight",
            (dim, dim * len(c["dspark_target_layer_ids"])), QTYPE_Q8_0, "projection")
    regular("mtp.0.main_norm.weight", "mtp.0.main_norm.weight",
            (dim,), QTYPE_F32, "norm")
    final = stages - 1
    regular(f"mtp.{final}.norm.weight", f"mtp.{final}.norm.weight",
            (dim,), QTYPE_F32, "norm")
    regular(f"mtp.{final}.markov_head.markov_w1.weight",
            f"mtp.{final}.markov_head.embed.weight", (vocab, rank),
            QTYPE_Q8_0, "markov")
    regular(f"mtp.{final}.markov_head.markov_w2.weight",
            f"mtp.{final}.markov_head.head.weight", (vocab, rank),
            QTYPE_Q8_0, "markov")
    regular(f"mtp.{final}.confidence_head.proj.weight",
            f"mtp.{final}.confidence_head.proj.weight", (1, dim + rank),
            QTYPE_Q8_0, "confidence")

    plan.sort(key=lambda item: item.name)
    offset = 0
    for item in plan:
        item.offset = offset
        item.nbytes = qtype_nbytes(item.qtype, item.shape)
        offset += align(item.nbytes, GGUF_ALIGNMENT)
    return plan, schema


def support_metadata(config, source_revision):
    c = config["text_config"]
    return [
        kv_string("general.architecture", "deepseek4-dspark"),
        kv_string("general.name", "DeepSeek V4.1 Flash DSpark support"),
        kv_string("general.source.url",
                  "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash"),
        kv_string("general.source.revision", source_revision),
        kv_u32("general.alignment", GGUF_ALIGNMENT),
        kv_u32("dspark.block_size", c["dspark_block_size"]),
        kv_u32("dspark.markov_rank", c["dspark_markov_rank"]),
        kv_u32("dspark.noise_token_id", c["dspark_noise_token_id"]),
        kv_u32_array("dspark.target_layer_ids", c["dspark_target_layer_ids"]),
        kv_u32("dspark.stage_count", c["num_nextn_predict_layers"]),
        kv_u32("dspark.n_layers", c["num_nextn_predict_layers"]),
        kv_u32("dspark.expert_count", c["dspark_n_routed_experts"]),
        kv_u32("dspark.expert_used_count", c["dspark_num_experts_per_tok"]),
        kv_string("deepseek41.dspark.quantization", QUANTIZATION),
    ]


def dry_run_report(plan, schema, index, records, hf_dir, min_free_gib=32):
    data_offset, data_bytes = print_plan(plan, records, [], GGUF_ALIGNMENT)
    print("output_tensors:")
    for item in plan:
        print(f"  {item.name}\t{item.nbytes}\tshape={list(item.shape)}\tqtype={QTYPE_NAMES[item.qtype]}")

    by_shard = {}
    for name, spec in schema.items():
        shard = index["weight_map"][name]
        by_shard[shard] = by_shard.get(shard, 0) + spec.nbytes
    print("input_shards:")
    actual_total = 0
    all_present = True
    for shard, payload in sorted(by_shard.items()):
        path = os.path.join(hf_dir, shard)
        actual = os.path.getsize(path) if os.path.isfile(path) else None
        all_present &= actual is not None
        actual_total += actual or 0
        actual_text = str(actual) if actual is not None else "missing"
        print(f"  {shard}\tpayload_bytes={payload}\tfile_bytes={actual_text}")
    payload_total = sum(by_shard.values())
    output_bytes = data_offset + data_bytes
    print(f"mtp_source_tensors: {len(schema)}")
    print(f"mtp_source_payload_bytes: {payload_total}")
    print(f"mtp_source_file_bytes: {actual_total if all_present else 'unavailable'}")
    print(f"checkpoint_total_size_from_index: {index.get('metadata', {}).get('total_size', 'unavailable')}")
    print(f"output_file_bytes: {output_bytes}")
    print(f"temporary_bytes_shards_already_staged: {output_bytes}")
    print(f"temporary_bytes_starting_without_shards_payload_floor: {payload_total + output_bytes}")
    reserve = math.ceil(min_free_gib * (1 << 30))
    print(f"recommended_free_bytes_with_{min_free_gib:g}gib_reserve: "
          f"{payload_total + output_bytes + reserve}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf", required=True,
                        help="directory containing config, index, and MTP shards")
    parser.add_argument("--out", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--imatrix")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--min-free-gib", type=float, default=32)
    suffix = "dylib" if sys.platform == "darwin" else "so"
    parser.add_argument("--quants-library", default=os.path.join(
        os.path.dirname(__file__), f"libds4quants.{suffix}"))
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_revision):
        parser.error("source revision must be a full commit hash")
    if not 1 <= args.threads <= 32:
        parser.error("threads must be between 1 and 32")
    config, index = load_source_documents(args.hf)
    records = support_metadata(config, args.source_revision)
    if args.dry_run:
        plan, schema = build_plan(config, index)
        dry_run_report(plan, schema, index, records, args.hf, args.min_free_gib)
        return

    db = SourceDB(args.hf, index_validator=lambda _: None,
                  scale_validator=lambda _: None,
                  skip_tensors=lambda name: not name.startswith("mtp."))
    try:
        plan, _ = build_plan(config, index, db)
        # write_gguf uses the same native FP8/FP4 decoding and q2 recipes as
        # the main V4.1 converter.
        write_gguf(args, plan, records, db)
    finally:
        db.close()


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        sys.exit(f"deepseek41-mtp-quantize: {error}")
