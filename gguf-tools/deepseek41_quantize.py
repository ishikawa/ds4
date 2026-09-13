#!/usr/bin/env python3
"""Convert DeepSeek V4.1 Flash to DwarfStar's Q2 or Q4_K GGUF.

Uses the project's C quantizers. Engram FP8 rows and their scales are packed
losslessly by default; pre-quantized Q4_K rows can instead be embedded or
referenced as external sidecars. Vision and DSpark are separate models.
"""

import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
import os
import re
import shutil
import struct
import sys
import time

from deepseek41_metadata import (GGUF_ALIGNMENT, engram_sample_sha256,
                                 metadata)
from glm53_quantize import (
    SourceDB, TensorPlan, Quantizer, Imatrix, QTYPE_F32, QTYPE_F16,
    QTYPE_Q8_0, QTYPE_Q2_K, QTYPE_Q4_K, QTYPE_IQ2_XXS, QTYPE_I8, align,
    conversion_signature, kv_string, load_resume_state, print_plan,
    qtype_nbytes, save_resume_state, tensor_header,
)

QUANTIZATION = {
    "q2": "IQ2_XXS gate/up; Q2_K down; Q8_0 attention/shared/head",
    "q4": "Q4_K gate/up/down; Q8_0 attention/shared/head",
}

ENGRAM_Q4K_ROW_BYTES = 144


@dataclasses.dataclass(frozen=True)
class EngramQ4K:
    layer: int
    path: str
    rows: int
    offset: int = 0

    def integrity(self, embedded=False):
        return {
            "size": self.rows * ENGRAM_Q4K_ROW_BYTES if embedded else os.path.getsize(self.path),
            "rows": self.rows,
            "sample_sha256": engram_sample_sha256(self.path, self.rows, self.offset),
        }


def load_engram_q4k(config, files=None, directory=None):
    layers = config["text_config"]["engram_layer_ids"]
    expected_rows = config["text_config"].get("engram_num_embeddings")
    if expected_rows is None:
        expected_rows = [384006168, 384016682]
    if len(layers) != 2 or len(expected_rows) != len(layers):
        raise ValueError("expected two Engram layers and row counts")
    if bool(files) == bool(directory):
        raise ValueError("specify either --engram-q4k or --engram-q4k-dir")
    selected = {}
    if directory:
        for layer in layers:
            selected[layer] = os.path.join(directory, f"engram-{layer}.q4_k.bin")
    else:
        positional = []
        for spec in files:
            if "=" in spec:
                layer_text, path = spec.split("=", 1)
                try:
                    layer = int(layer_text)
                except ValueError as error:
                    raise ValueError(f"invalid Engram layer in {spec}") from error
                if layer in selected:
                    raise ValueError(f"duplicate Engram sidecar for layer {layer}")
                selected[layer] = path
            else:
                positional.append(spec)
        remaining = [layer for layer in layers if layer not in selected]
        if len(positional) != len(remaining):
            raise ValueError("--engram-q4k needs one file per Engram layer")
        selected.update(zip(remaining, positional))
    if set(selected) != set(layers):
        raise ValueError(f"Engram sidecars must cover layers {layers}")
    result = []
    for layer, rows in zip(layers, expected_rows):
        path = os.path.abspath(selected[layer])
        with open(path + ".json", "rb") as fp:
            info = json.load(fp)
        required = {"rows": rows, "bytes_per_row": ENGRAM_Q4K_ROW_BYTES,
                    "cols": 256, "block_type": "Q4_K", "ggml_type": QTYPE_Q4_K,
                    "format_version": 1, "complete": True}
        for key, expected in required.items():
            if info.get(key) != expected:
                raise ValueError(f"{path}.json: expected {key}={expected!r}")
        offset = info.get("offset", 0)
        if not isinstance(offset, int) or offset < 0:
            raise ValueError(f"{path}.json: invalid offset")
        expected_size = offset + rows * ENGRAM_Q4K_ROW_BYTES
        if os.path.getsize(path) != expected_size:
            raise ValueError(f"{path}: expected {expected_size} bytes")
        result.append(EngramQ4K(layer, path, rows, offset))
    return result


def scale_name(name):
    return name.removesuffix(".weight") + ".scale"


def validate_scales(tensors):
    for name, info in tensors.items():
        if not name.startswith(("layers.", "embed.", "head.", "norm.")):
            continue
        dtype, shape = info["dtype"], info["shape"]
        if dtype not in ("F8_E4M3", "I8") or not name.endswith(".weight"):
            continue
        if len(shape) != 2:
            raise ValueError(f"{name}: expected a matrix")
        if dtype == "I8":
            expected = [shape[0], shape[1] // 16]
        elif ".engram.embed." in name:
            expected = [shape[0], shape[1] // 32]
        else:
            expected = [(dim + 31) // 32 for dim in shape]
        scale = tensors.get(scale_name(name))
        if not scale or scale["dtype"] != "F8_E8M0" or scale["shape"] != expected:
            raise ValueError(f"{name}: expected E8M0 scales {expected}")


def build_plan(db, config, quant="q2", engram_q4k=None, engram_external=False):
    if quant not in QUANTIZATION:
        raise ValueError(f"unknown quantization recipe: {quant}")
    c = config["text_config"]
    if config["quantization_config"]["weight_block_size"] != [32, 32]:
        raise ValueError("expected native 32x32 FP8 blocks")
    dim, inter = c["hidden_size"], c["moe_intermediate_size"]
    heads, hd = c["num_attention_heads"], c["head_dim"]
    qrank, orank, groups = c["q_lora_rank"], c["o_lora_rank"], c["o_groups"]
    hc, experts, vocab = c["hc_mult"], c["n_routed_experts"], c["vocab_size"]
    ih, idim = c["index_n_heads"], c["index_head_dim"]
    plan, disk, consumed = [], [], set()

    def claim(name, expected, dtype=None):
        info = db.info(name)
        if info["shape"] != list(expected) or (dtype and info["dtype"] != dtype):
            raise ValueError(f"{name}: unexpected {info['dtype']} {info['shape']}, expected {dtype} {expected}")
        consumed.add(name)
        if info["dtype"] in ("I8", "F8_E4M3"):
            consumed.add(scale_name(name))
        return info

    def regular(name, source, shape, qtype, role):
        claim(source, shape)
        plan.append(TensorPlan(name, tuple(reversed(shape)), qtype, role, source=source))

    regular("token_embd.weight", "embed.weight", (vocab, dim), QTYPE_F16, "embedding")
    regular("output_norm.weight", "norm.weight", (dim,), QTYPE_F32, "norm")
    regular("output.weight", "head.weight", (vocab, dim), QTYPE_Q8_0, "output")
    for layer in range(c["num_hidden_layers"]):
        src, dst = f"layers.{layer}", f"blk.{layer}"
        for site in ("attn", "ffn"):
            for part, shape, qt in (("fn", (hc * (hc + 2), hc * dim), QTYPE_F16),
                                    ("base", (hc * (hc + 2),), QTYPE_F32),
                                    ("scale", (3,), QTYPE_F32)):
                regular(f"{dst}.hc_{site}_{part}.weight", f"{src}.hc_{site}_{part}", shape, qt, "mhc")
            regular(f"{dst}.{site}_norm.weight", f"{src}.{site}_norm.weight", (dim,), QTYPE_F32, "norm")
        for target, source, shape, qt in (
            ("attn_sinks.weight", "attn_sink", (heads,), QTYPE_F32),
            ("attn_q_a.weight", "wq_a.weight", (qrank, dim), QTYPE_Q8_0),
            ("attn_q_b.weight", "wq_b.weight", (heads * hd, qrank), QTYPE_Q8_0),
            ("attn_q_a_norm.weight", "q_norm.weight", (qrank,), QTYPE_F32),
            ("attn_kv.weight", "wkv.weight", (hd, dim), QTYPE_Q8_0),
            ("attn_kv_a_norm.weight", "kv_norm.weight", (hd,), QTYPE_F32),
            ("attn_output_a.weight", "wo_a.weight", (groups * orank, heads * hd // groups), QTYPE_Q8_0),
            ("attn_output_b.weight", "wo_b.weight", (dim, groups * orank), QTYPE_Q8_0),
        ):
            regular(f"{dst}.{target}", f"{src}.attn.{source}", shape, qt, "attention")
        if layer in c["kv_source_layer_ids"]:
            for target, source, shape, qt in (
                ("attn_compressor_kv.weight", "compressor.wkv.weight", (hd, dim), QTYPE_F16),
                ("attn_compressor_norm.weight", "compressor.norm.weight", (hd,), QTYPE_F32),
                ("indexer.attn_k.weight", "indexer.wk.weight", (idim, hd), QTYPE_F16),
                ("indexer.k_norm.weight", "indexer.k_norm.weight", (idim,), QTYPE_F32),
            ):
                regular(f"{dst}.{target}", f"{src}.attn.{source}", shape, qt, "compressor")
            if c["compress_ratios"][layer] > 1:
                regular(f"{dst}.attn_compressor_gate.weight", f"{src}.attn.compressor.wgate.weight",
                        (hd, dim), QTYPE_F16, "compressor")
        if layer in c["index_source_layer_ids"]:
            regular(f"{dst}.indexer.attn_q_b.weight", f"{src}.attn.indexer.wq_b.weight",
                    (ih * idim, qrank), QTYPE_F16, "indexer")
            regular(f"{dst}.indexer.proj.weight", f"{src}.attn.indexer.weights_proj.weight",
                    (ih, dim), QTYPE_F16, "indexer")
        regular(f"{dst}.ffn_gate_inp.weight", f"{src}.ffn.gate.weight", (experts, dim), QTYPE_F32, "router")
        for suffix, target in (("bias", "exp_probs_b.bias"), ("bias_vl", "exp_probs_b_vl.bias")):
            regular(f"{dst}.{target}", f"{src}.ffn.gate.{suffix}", (experts,), QTYPE_F32, "router")
        for part, source, shape, qt in (("gate", "w1", (inter, dim), QTYPE_IQ2_XXS),
                                       ("up", "w3", (inter, dim), QTYPE_IQ2_XXS),
                                       ("down", "w2", (dim, inter), QTYPE_Q2_K)):
            regular(f"{dst}.ffn_{part}_shexp.weight", f"{src}.ffn.shared_experts.{source}.weight",
                    shape, QTYPE_Q8_0, "shared")
            pattern = f"{src}.ffn.experts.{{expert}}.{source}.weight"
            for expert in range(experts):
                claim(pattern.format(expert=expert), (shape[0], shape[1] // 2), "I8")
            plan.append(TensorPlan(f"{dst}.ffn_{part}_exps.weight", (*reversed(shape), experts),
                                   QTYPE_Q4_K if quant == "q4" else qt,
                                   "experts", source=pattern, expert_layer=layer,
                                   expert_part=part, expert_count=experts))
        if layer in c["engram_layer_ids"]:
            index = c["engram_layer_ids"].index(layer)
            rows = c["engram_num_embeddings"][index]
            engram = f"{src}.engram"
            if engram_q4k:
                sidecar = engram_q4k[index]
                if sidecar.layer != layer or sidecar.rows != rows:
                    raise ValueError(f"layer {layer}: Engram sidecar row mismatch")
                if not engram_external:
                    disk.append(TensorPlan(f"{dst}.engram_embd.weight",
                                           (256, rows), QTYPE_Q4_K,
                                           "engram_q4k", source=sidecar.path,
                                           row_start=sidecar.offset, raw_copy=True))
            else:
                claim(f"{engram}.embed.weight", (rows, 256), "F8_E4M3")
                claim(f"{engram}.embed.scale", (rows, 8), "F8_E8M0")
                disk.append(TensorPlan(f"{dst}.engram_embd.weight", (264, rows), QTYPE_I8,
                                       "engram_disk", source=f"{engram}.embed.weight", raw_copy=True))
            for part in ("q", "k"):
                regular(f"{dst}.engram_{part}_norm.weight", f"{engram}.{part}_weight", (hc, dim), QTYPE_F32, "engram")
            regular(f"{dst}.engram_kv.weight", f"{engram}.wkv.weight", ((hc + 1) * dim, 24 * 256), QTYPE_F16, "engram")
    omitted = {name for name in db.tensors if name.startswith(("mtp.", "vision.", "aligner.", "image_"))}
    if engram_q4k:
        omitted.update(name for name in db.tensors if ".engram.embed." in name)
    if consumed | omitted != set(db.tensors):
        raise ValueError(f"unclaimed source tensors: {sorted(set(db.tensors) - consumed - omitted)[:10]}")
    offset = 0
    for item in plan + disk:
        item.offset = offset
        item.nbytes = qtype_nbytes(item.qtype, item.shape)
        offset += align(item.nbytes, GGUF_ALIGNMENT)
    return plan + disk


class NativeQuantizer(Quantizer):
    def to_f32(self, db, name, row_start=0, row_count=None):
        if ".engram.embed." in name:
            raise ValueError("Engram must never be materialized as a whole float tensor")
        np = self.np
        info = db.info(name)
        if info["dtype"] not in ("I8", "F8_E4M3"):
            result = super().to_f32(db, name, row_start, row_count)
        else:
            shape = info["shape"]
            codes = np.frombuffer(db.read(name), dtype=np.uint8).reshape(shape)
            sinfo = db.info(scale_name(name))
            scales = np.frombuffer(db.read(scale_name(name)), dtype=np.uint8).reshape(sinfo["shape"])
            if np.any(scales == 255):
                raise ValueError(f"{name}: nonfinite scale")
            scales = np.ldexp(np.ones_like(scales, dtype=np.float32), scales.astype(np.int32) - 127)
            if info["dtype"] == "I8":
                lut = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float32)
                result = np.empty((shape[0], shape[1] * 2), dtype=np.float32)
                result[:, 0::2] = lut[codes & 15]
                result[:, 1::2] = lut[codes >> 4]
                result = (result.reshape(shape[0], -1, 32) * scales[:, :, None]).reshape(result.shape)
            else:
                if np.any((codes & 127) == 127):
                    raise ValueError(f"{name}: nonfinite FP8 weight")
                result = self.fp8_lut[codes]
                for row in range(0, shape[0], 32):
                    for col in range(0, shape[1], 32):
                        result[row:row + 32, col:col + 32] *= scales[row // 32, col // 32]
            if row_count is not None:
                result = result[row_start:row_start + row_count]
        if not np.all(np.isfinite(result)):
            raise ValueError(f"{name}: nonfinite dequantized weight")
        return np.ascontiguousarray(result, dtype=np.float32)

    def encode(self, array, qtype, imatrix=None):
        if qtype == QTYPE_F16 and self.np.any(self.np.abs(array) > 65504):
            raise ValueError("F16 tensor would overflow; preserve this family in BF16/F32")
        return super().encode(array, qtype, imatrix)


def write_engram(fp, item, db, np):
    # Keep weights and scales together without changing either byte. A chunk
    # holds 16K rows (4 MiB of weights), independent of the 94 GiB table size.
    rows = item.shape[1]
    for start in range(0, rows, 16384):
        count = min(16384, rows - start)
        weights = b"".join(db.iter_read(item.source, start * 256, count * 256))
        scales = b"".join(db.iter_read(scale_name(item.source), start * 8, count * 8))
        w = np.frombuffer(weights, dtype=np.uint8).reshape(count, 256)
        s = np.frombuffer(scales, dtype=np.uint8).reshape(count, 8)
        if np.any((w & 127) == 127) or np.any(s == 255):
            raise ValueError(f"{item.name}: nonfinite Engram row near {start}")
        packed = np.empty((count, 264), dtype=np.uint8)
        packed[:, :256], packed[:, 256:] = w, s
        fp.write(packed.tobytes())


def write_engram_q4k(fp, item):
    remaining = item.nbytes
    with open(item.source, "rb") as source:
        source.seek(item.row_start)
        while remaining:
            chunk = source.read(min(4 << 20, remaining))
            if not chunk:
                raise ValueError(f"{item.name}: truncated Q4_K sidecar")
            fp.write(chunk)
            remaining -= len(chunk)


def write_gguf(args, plan, records, db):
    quantizer = NativeQuantizer(args.quants_library)
    imatrix = Imatrix(args.imatrix, quantizer.np)
    if args.imatrix:
        for item in plan:
            if item.is_expert and item.name not in imatrix.entries:
                raise ValueError(f"missing imatrix tensor {item.name}")
        if any(quantizer.np.any(values < 0) for values in imatrix.entries.values()):
            raise ValueError("negative importance values")
    data_start, data_bytes = print_plan(plan, records, [], GGUF_ALIGNMENT)
    partial, journal = args.out + ".partial", args.out + ".partial.json"
    signature = conversion_signature(plan, records, [], args.imatrix)
    # A metadata/recipe match alone cannot distinguish two source downloads.
    source_identity = [(name, db.info(name)) for name in sorted(db.tensors)]
    resume_identity = {
        "sources": source_identity,
        "row_starts": [(item.name, item.row_start) for item in plan],
        "driver": db.signature_identity() if hasattr(db, "signature_identity") else None,
    }
    signature = hashlib.sha256(
        (signature + json.dumps(resume_identity, sort_keys=True)).encode()).hexdigest()
    if os.path.exists(args.out):
        raise ValueError(f"refusing to overwrite {args.out}")
    completed = 0
    if os.path.exists(partial) or os.path.exists(journal):
        if not args.resume or not (os.path.exists(partial) and os.path.exists(journal)):
            raise ValueError("partial file and journal require --resume")
        completed = load_resume_state(journal, signature, plan)
        if hasattr(db, "reconcile_completed"):
            reconciled = db.reconcile_completed(plan, completed)
            if reconciled != completed:
                completed = reconciled
                save_resume_state(journal, signature, completed)
    end = data_start + (plan[completed - 1].offset +
                       align(plan[completed - 1].nbytes, GGUF_ALIGNMENT) if completed else 0)
    remaining_output = data_start + data_bytes - end
    if hasattr(db, "reserve_output"):
        db.reserve_output(remaining_output, args.out)
    else:
        reserve = int(getattr(args, "min_free_gib", 32) * (1 << 30))
        free = shutil.disk_usage(os.path.dirname(os.path.abspath(args.out))).free
        if free < remaining_output + reserve:
            raise ValueError("insufficient disk space for remaining output plus reserve")
    header = b"GGUF" + struct.pack("<IQQ", 3, len(plan), len(records))
    header += b"".join(records) + b"".join(tensor_header(item) for item in plan)
    header += bytes(data_start - len(header))
    if os.path.exists(partial):
        with open(partial, "rb") as fp:
            if fp.read(data_start) != header or os.fstat(fp.fileno()).st_size < end:
                raise ValueError("partial GGUF is truncated or has a different header")
    else:
        with open(partial, "xb") as fp:
            fp.write(header)
            fp.flush()
            os.fsync(fp.fileno())
        if hasattr(db, "consume_output"):
            db.consume_output(data_start)
        save_resume_state(journal, signature, 0)
    if hasattr(db, "conversion_started"):
        db.conversion_started(plan, completed, data_start)
    with open(partial, "r+b") as fp, concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as pool:
        fp.truncate(end)
        fp.seek(end)
        for index in range(completed, len(plan)):
            item = plan[index]
            if hasattr(db, "prepare_item"):
                db.prepare_item(item, index)
            started = time.monotonic()
            missing = 0
            if fp.tell() != data_start + item.offset:
                raise ValueError(f"incorrect offset for {item.name}")
            if item.role == "engram_disk":
                write_engram(fp, item, db, quantizer.np)
            elif item.role == "engram_q4k":
                write_engram_q4k(fp, item)
            elif item.is_expert:
                def convert(expert):
                    values = quantizer.to_f32(db, item.source.format(expert=expert))
                    importance = imatrix.expert(item.name, expert, item.shape[0], item.expert_count)
                    return quantizer.encode(values, item.qtype, importance), importance is None
                for start in range(0, item.expert_count, args.threads):
                    futures = [pool.submit(convert, e) for e in range(start, min(start + args.threads, item.expert_count))]
                    for future in futures:
                        data, fallback = future.result()
                        if len(data) != item.nbytes // item.expert_count:
                            raise ValueError("wrong encoded expert size")
                        fp.write(data)
                        missing += int(fallback)
            else:
                fp.write(quantizer.encode(quantizer.to_f32(db, item.source), item.qtype))
            if fp.tell() != data_start + item.offset + item.nbytes:
                raise ValueError(f"incorrect payload size for {item.name}")
            fp.write(bytes(align(item.nbytes, GGUF_ALIGNMENT) - item.nbytes))
            fp.flush()
            os.fsync(fp.fileno())
            if hasattr(db, "consume_output"):
                db.consume_output(align(item.nbytes, GGUF_ALIGNMENT))
            save_resume_state(journal, signature, index + 1)
            if hasattr(db, "item_completed"):
                db.item_completed(item, index + 1,
                                  data_start + item.offset + align(item.nbytes, GGUF_ALIGNMENT))
            print(f"[{index + 1}/{len(plan)}] {item.name}: {item.nbytes / (1 << 30):.3f} GiB, "
                  f"{time.monotonic() - started:.1f}s, uncalibrated_experts={missing}", flush=True)
    os.rename(partial, args.out)
    os.unlink(journal)
    if hasattr(db, "conversion_finished"):
        db.conversion_finished(args.out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--quant", choices=QUANTIZATION, default="q2")
    parser.add_argument("--imatrix")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--engram-q4k", action="append", metavar="[LAYER=]FILE",
                        help="Q4_K sidecar; repeat once per Engram layer")
    source.add_argument("--engram-q4k-dir", metavar="DIR",
                        help="directory containing engram-{layer}.q4_k.bin")
    parser.add_argument("--engram-q4k-external", action="store_true",
                        help="reference Q4_K sidecars instead of embedding them")
    parser.add_argument("--engram-q4k-relative", action="store_true",
                        help="store external sidecars relative to the output GGUF")
    suffix = "dylib" if sys.platform == "darwin" else "so"
    parser.add_argument("--quants-library", default=os.path.join(os.path.dirname(__file__), f"libds4quants.{suffix}"))
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_revision):
        parser.error("source revision must be a full commit hash")
    if not 1 <= args.threads <= 32:
        parser.error("threads must be between 1 and 32")
    with open(os.path.join(args.hf, "config.json"), "rb") as fp:
        source_config = json.load(fp)
    sidecars = None
    if args.engram_q4k or args.engram_q4k_dir:
        sidecars = load_engram_q4k(source_config, args.engram_q4k,
                                   args.engram_q4k_dir)
    elif args.engram_q4k_external:
        parser.error("--engram-q4k-external requires Q4_K sidecars")
    if args.engram_q4k_relative and not args.engram_q4k_external:
        parser.error("--engram-q4k-relative requires --engram-q4k-external")
    storage = "external" if args.engram_q4k_external else "embedded"
    external_paths = None
    if args.engram_q4k_external:
        external_paths = [item.path for item in sidecars]
        if args.engram_q4k_relative:
            base = os.path.dirname(os.path.abspath(args.out))
            external_paths = [os.path.relpath(path, base) for path in external_paths]
    config, records = metadata(
        args.hf, args.source_revision,
        "q4_k_row144" if sidecars else "e4m3_e8m0_32_row264", storage,
        external_paths,
        [item.offset for item in sidecars] if args.engram_q4k_external else None,
        [item.integrity(not args.engram_q4k_external) for item in sidecars]
        if sidecars else None)
    db = SourceDB(args.hf, index_validator=lambda _: None, scale_validator=validate_scales,
                  skip_tensors=(lambda name: ".engram.embed." in name)
                  if sidecars else None)
    try:
        plan = build_plan(db, config, args.quant, sidecars,
                          args.engram_q4k_external)
        records.append(kv_string("deepseek41.quantization", QUANTIZATION[args.quant]))
        records.append(kv_string("deepseek41.calibration", "imatrix" if args.imatrix else "weight-energy bootstrap"))
        if args.imatrix:
            records.append(kv_string("quantize.imatrix.file", os.path.basename(args.imatrix)))
        if args.dry_run:
            print_plan(plan, records, [], GGUF_ALIGNMENT)
            for item in plan:
                print(json.dumps(dataclasses.asdict(item), sort_keys=True))
        else:
            write_gguf(args, plan, records, db)
    finally:
        db.close()


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        sys.exit(f"deepseek41-quantize: {error}")
