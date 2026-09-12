#!/usr/bin/env python3
"""Audit a completed V4.1 GGUF and sample its payloads against the source.

Checks the complete tensor layout, then re-encodes selected expert tensors and
samples native or Q4_K Engram rows. External sidecars are checked for schema,
size and metadata identity. This is artifact validation, not inference QA.
"""
import argparse
import json
import os
from pathlib import Path
import random
import sys

from deepseek41_metadata import GGUF_ALIGNMENT
from deepseek41_quantize import (SourceDB, NativeQuantizer, Imatrix, build_plan,
                                  validate_scales, scale_name, QUANTIZATION,
                                  load_engram_q4k, ENGRAM_Q4K_ROW_BYTES)
from glm53_quantize import (align, read_exact, read_u32, read_u64,
                            read_gguf_string, skip_gguf_value, GGUF_ARRAY,
                            GGUF_STRING, GGUF_UINT64)
from glm53_validate_gguf import read_selected_metadata


def check_payload(fp, offset, item, db, quantizer, imatrix):
    if item.role == "engram_disk":
        rows = item.shape[1]
        rng = random.Random(41 + rows)
        selected = {0, 1, rows - 1, 16383, 16384, (1 << 32) // 264}
        selected.update(rng.randrange(rows) for _ in range(128))
        for row in sorted(selected):
            if row >= rows:
                continue
            expected = b"".join(db.iter_read(item.source, row * 256, 256))
            expected += b"".join(db.iter_read(scale_name(item.source), row * 8, 8))
            fp.seek(offset + row * 264)
            if read_exact(fp, 264, item.name) != expected:
                raise ValueError(f"{item.name}: Engram row {row} differs from source")
    elif item.role == "engram_q4k":
        rows = item.shape[1]
        rng = random.Random(41 + rows)
        selected = {0, 1, rows - 1, 16383, 16384,
                    (1 << 32) // ENGRAM_Q4K_ROW_BYTES}
        selected.update(rng.randrange(rows) for _ in range(min(128, rows)))
        with open(item.source, "rb") as source:
            for row in sorted(selected):
                if row >= rows:
                    continue
                source.seek(item.row_start + row * ENGRAM_Q4K_ROW_BYTES)
                expected = read_exact(source, ENGRAM_Q4K_ROW_BYTES, item.source)
                fp.seek(offset + row * ENGRAM_Q4K_ROW_BYTES)
                if read_exact(fp, ENGRAM_Q4K_ROW_BYTES, item.name) != expected:
                    raise ValueError(f"{item.name}: Engram row {row} differs from sidecar")
    elif item.is_expert:
        selected = {0, (item.expert_layer * 17 + 41) % item.expert_count, item.expert_count - 1}
        stride = item.nbytes // item.expert_count
        for expert in sorted(selected):
            values = quantizer.to_f32(db, item.source.format(expert=expert))
            importance = imatrix.expert(item.name, expert, item.shape[0], item.expert_count)
            expected = quantizer.encode(values, item.qtype, importance)
            fp.seek(offset + expert * stride)
            if len(expected) != stride or read_exact(fp, stride, item.name) != expected:
                raise ValueError(f"{item.name}: encoded expert {expert} differs from source recipe")
    else:
        values = quantizer.to_f32(db, item.source)
        expected = quantizer.encode(values, item.qtype)
        fp.seek(offset)
        if len(expected) != item.nbytes or read_exact(fp, item.nbytes, item.name) != expected:
            raise ValueError(f"{item.name}: payload differs from source recipe")


def read_engram_metadata(fp, kind):
    if kind != GGUF_ARRAY:
        return read_selected_metadata(fp, kind)
    element = read_u32(fp, "metadata array type")
    count = read_u64(fp, "metadata array count")
    if count != 2:
        raise ValueError("external Engram metadata must have two entries")
    if element == GGUF_STRING:
        return [read_gguf_string(fp, "metadata string") for _ in range(count)]
    if element == GGUF_UINT64:
        return [read_u64(fp, "metadata uint64") for _ in range(count)]
    raise ValueError(f"unsupported selected metadata array type {element}")


def validate(args):
    config = json.loads((Path(args.hf) / "config.json").read_text())
    files = getattr(args, "engram_q4k", None)
    directory = getattr(args, "engram_q4k_dir", None)
    external = getattr(args, "engram_q4k_external", False)
    sidecars = load_engram_q4k(config, files, directory) if files or directory else None
    if external and not sidecars:
        raise ValueError("--engram-q4k-external requires Q4_K sidecars")
    db = SourceDB(args.hf, index_validator=lambda _: None, scale_validator=validate_scales,
                  skip_tensors=(lambda name: ".engram.embed." in name)
                  if sidecars else None)
    try:
        plan = build_plan(db, config, args.quant, sidecars, external)
        with open(args.gguf, "rb") as fp:
            if read_exact(fp, 4, "magic") != b"GGUF" or read_u32(fp, "version") != 3:
                raise ValueError("expected GGUF v3")
            if read_u64(fp, "tensor count") != len(plan):
                raise ValueError("tensor count differs from source plan")
            metadata = {}
            for _ in range(read_u64(fp, "metadata count")):
                key = read_gguf_string(fp, "metadata key")
                kind = read_u32(fp, "metadata type")
                if key in {"general.architecture", "general.alignment", "general.source.revision",
                            "deepseek41.calibration", "deepseek41.quantization",
                            "deepseek41.engram.encoding", "deepseek41.engram.storage",
                            "deepseek41.engram.external_paths",
                            "deepseek41.engram.external_offsets"}:
                    if key in metadata:
                        raise ValueError(f"duplicate metadata: {key}")
                    metadata[key] = read_engram_metadata(fp, kind)
                else:
                    skip_gguf_value(fp, kind)
            metadata.setdefault("deepseek41.engram.storage", "embedded")
            expected = {"general.architecture": "deepseek41", "general.alignment": GGUF_ALIGNMENT,
                        "general.source.revision": args.source_revision,
                        "deepseek41.quantization": QUANTIZATION[args.quant],
                        "deepseek41.calibration": "imatrix" if args.imatrix else "weight-energy bootstrap",
                        "deepseek41.engram.encoding": "q4_k_row144" if sidecars else "e4m3_e8m0_32_row264",
                        "deepseek41.engram.storage": "external" if external else "embedded"}
            if external:
                expected["deepseek41.engram.external_paths"] = [item.path for item in sidecars]
                expected["deepseek41.engram.external_offsets"] = [item.offset for item in sidecars]
            if metadata != expected:
                raise ValueError(f"metadata mismatch: {metadata}")
            for item in plan:
                name = read_gguf_string(fp, "tensor name")
                rank = read_u32(fp, "tensor rank")
                if rank != len(item.shape):
                    raise ValueError(f"{name}: unexpected rank {rank}")
                shape = tuple(read_u64(fp, "dimension") for _ in range(rank))
                kind, offset = read_u32(fp, "tensor type"), read_u64(fp, "tensor offset")
                if (name, shape, kind, offset) != (item.name, item.shape, item.qtype, item.offset):
                    raise ValueError(f"{item.name}: tensor layout mismatch")
            start = align(fp.tell(), GGUF_ALIGNMENT)
            size = start + (plan[-1].offset + align(plan[-1].nbytes, GGUF_ALIGNMENT)
                            if plan else 0)
            if os.fstat(fp.fileno()).st_size != size:
                raise ValueError(f"expected file size {size}")
            print(f"PASS: {len(plan)} tensor layouts; complete {size}-byte GGUF", flush=True)
            for item in plan:
                if item.role == "engram_q4k":
                    check_payload(fp, start + item.offset, item, db, None, None)
            if args.payload:
                q = NativeQuantizer(args.quants_library)
                imatrix = Imatrix(args.imatrix, q.np)
                if args.imatrix and any(t.is_expert and t.name not in imatrix.entries for t in plan):
                    raise ValueError("imatrix is missing expert tensors")
                for index, item in enumerate(plan):
                    if item.role != "engram_q4k":
                        check_payload(fp, start + item.offset, item, db, q, imatrix)
                    if (index + 1) % 25 == 0 or index + 1 == len(plan):
                        print(f"PASS: source payload checks {index + 1}/{len(plan)}", flush=True)
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf", required=True)
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--quant", choices=QUANTIZATION, default="q2")
    parser.add_argument("--imatrix")
    parser.add_argument("--payload", action="store_true")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--engram-q4k", action="append", metavar="[LAYER=]FILE")
    source.add_argument("--engram-q4k-dir", metavar="DIR")
    parser.add_argument("--engram-q4k-external", action="store_true")
    suffix = "dylib" if sys.platform == "darwin" else "so"
    parser.add_argument("--quants-library", default=str(Path(__file__).with_name(f"libds4quants.{suffix}")))
    args = parser.parse_args()
    try:
        validate(args)
    except (OSError, ValueError) as error:
        sys.exit(f"deepseek41-validate: {error}")
