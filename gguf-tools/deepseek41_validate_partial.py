#!/usr/bin/env python3
"""Validate the durable prefix of a streaming DeepSeek V4.1 GGUF."""

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from deepseek41_engram_metadata_repair import (
    INTEGRITY_KEYS, read_header, require_array, require_value,
)
from deepseek41_metadata import ENGRAM_SAMPLE_SCHEME, GGUF_ALIGNMENT
from deepseek41_quantize import (
    Imatrix, NativeQuantizer, QUANTIZATION, build_plan, load_engram_q4k,
    validate_scales, write_engram, write_engram_q4k,
)
from deepseek41_stream_convert import item_source_names, skipped_tensor
from glm53_manifest import load_index
from glm53_quantize import (
    GGUF_ARRAY, GGUF_STRING, GGUF_UINT32, GGUF_UINT64, QTYPE_BF16,
    QTYPE_F16, QTYPE_F32, SourceDB, align,
)


def load_progress(path):
    progress = json.loads(Path(path).read_text())
    version = progress.get("version")
    if version not in (1, 2):
        raise ValueError(f"unsupported progress version: {version!r}")
    count = progress.get("completed_tensor_count")
    names = progress.get("completed_tensors")
    offset = progress.get("output_offset")
    shards = progress.get("completed_shards")
    if not isinstance(count, int) or count < 0:
        raise ValueError("progress completed_tensor_count must be nonnegative")
    if not isinstance(names, list) or len(names) != count or not all(
            isinstance(name, str) for name in names):
        raise ValueError("progress completed_tensors does not match its count")
    if not isinstance(offset, int) or offset < 0:
        raise ValueError("progress output_offset must be nonnegative")
    if version == 1 and (not isinstance(shards, list) or not all(
            isinstance(shard, str) for shard in shards)):
        raise ValueError("progress v1 completed_shards must be a string list")
    if version == 2 and (not isinstance(shards, dict) or not all(
            isinstance(shard, str) and isinstance(oid, str)
            for shard, oid in shards.items())):
        raise ValueError("progress v2 completed_shards must map shards to oids")
    progress["completed_shard_names"] = set(shards)
    return progress


def default_headers_path(gguf):
    suffix = ".partial"
    if not str(gguf).endswith(suffix):
        raise ValueError("--gguf must name a .partial file or --headers must be specified")
    return str(gguf)[:-len(suffix)] + ".headers.json"


def load_cached_headers(path, index_path, revision):
    document = json.loads(Path(path).read_text())
    digest = hashlib.sha256(Path(index_path).read_bytes()).hexdigest()
    expected = {"version": 1, "revision": revision, "index_sha256": digest}
    if {key: document.get(key) for key in expected} != expected:
        raise ValueError(f"header cache does not match source inputs: {path}")
    headers = document.get("shards")
    if not isinstance(headers, dict):
        raise ValueError(f"header cache has no shard map: {path}")
    return headers


def validate_progress_shards(progress, headers):
    unknown = progress["completed_shard_names"] - set(headers)
    if unknown:
        raise ValueError(f"progress contains unknown completed shards: {sorted(unknown)}")
    if progress["version"] != 2:
        return
    for shard, oid in progress["completed_shards"].items():
        identifier = headers[shard].get("identifier")
        if identifier is not None and identifier.get("oid") != oid:
            raise ValueError(f"progress oid differs from header cache for {shard}")


def validate_metadata(header, args, sidecars):
    expected = {
        "general.architecture": (GGUF_STRING, "deepseek41"),
        "general.alignment": (GGUF_UINT32, GGUF_ALIGNMENT),
        "general.source.revision": (GGUF_STRING, args.source_revision),
        "deepseek41.quantization": (GGUF_STRING, QUANTIZATION[args.quant]),
        "deepseek41.calibration": (
            GGUF_STRING, "imatrix" if args.imatrix else "weight-energy bootstrap"),
        "deepseek41.engram.encoding": (GGUF_STRING, "q4_k_row144"),
        "deepseek41.engram.storage": (GGUF_STRING, "external"),
    }
    for key, (kind, value) in expected.items():
        actual = require_value(header, key, kind)
        if actual != value:
            raise ValueError(f"{key} is {actual!r}, expected {value!r}")
    paths = require_array(header, "deepseek41.engram.external_paths", GGUF_STRING)
    offsets = require_array(header, "deepseek41.engram.external_offsets", GGUF_UINT64)
    if len(paths) != len(sidecars) or offsets != [item.offset for item in sidecars]:
        raise ValueError("external Engram path/offset count differs from the sidecars")
    base = os.path.dirname(os.path.abspath(args.gguf))
    resolved = [os.path.abspath(os.path.join(base, path)) for path in paths]
    if resolved != [item.path for item in sidecars]:
        raise ValueError("external Engram paths differ from the supplied sidecars")

    present = [key in header["keys"] for key in INTEGRITY_KEYS]
    if not any(present):
        return "integrity metadata absent; sidecar schema, paths and offsets match"
    if not all(present):
        raise ValueError("external Engram integrity metadata is incomplete")
    integrity = [item.integrity(False) for item in sidecars]
    wanted = {
        INTEGRITY_KEYS[0]: [item["size"] for item in integrity],
        INTEGRITY_KEYS[1]: [item["rows"] for item in integrity],
        INTEGRITY_KEYS[2]: [item["sample_sha256"] for item in integrity],
    }
    for key, value in wanted.items():
        element = GGUF_STRING if key == INTEGRITY_KEYS[2] else GGUF_UINT64
        if require_array(header, key, element) != value:
            raise ValueError(f"{key} differs from the supplied sidecars")
    if require_value(header, INTEGRITY_KEYS[3], GGUF_STRING) != ENGRAM_SAMPLE_SCHEME:
        raise ValueError(f"{INTEGRITY_KEYS[3]} is unsupported")
    return "integrity metadata present; size, rows and sample sha256 match"


def compare_layout(header, plan):
    actual = header["tensors"]
    mismatches = []
    matched = 0
    if len(actual) != len(plan):
        mismatches.append(f"tensor count {len(actual)} != planned {len(plan)}")
    for index, (found, item) in enumerate(zip(actual, plan)):
        expected = (item.name, item.shape, item.qtype, item.offset)
        if found != expected:
            mismatches.append(
                f"tensor {index} {found[0]!r} at {found[3]} != "
                f"{item.name!r} at {item.offset}: {found[1:3]} != {expected[1:3]}")
        else:
            matched += 1
    return matched, mismatches


class CompareSink:
    def __init__(self, fd, start, length, name):
        self.fd = fd
        self.start = start
        self.length = length
        self.name = name
        self.written = 0

    def write(self, expected):
        end = self.written + len(expected)
        if end > self.length:
            raise ValueError(f"{self.name}: generated more than its planned byte range")
        actual = os.pread(self.fd, len(expected), self.start + self.written)
        if len(actual) != len(expected):
            raise ValueError(f"{self.name}: short read inside its durable byte range")
        if actual != expected:
            difference = next(index for index, pair in enumerate(zip(actual, expected))
                              if pair[0] != pair[1])
            raise ValueError(
                f"byte mismatch at GGUF offset {self.start + self.written + difference}")
        self.written = end

    def finish(self):
        if self.written != self.length:
            raise ValueError(
                f"{self.name}: compared {self.written} bytes, expected {self.length}")


def write_simple_chunks(sink, item, db, quantizer, chunk_bytes=64 << 20):
    if item.is_expert or item.row_start or item.row_count is not None:
        return False
    info = db.info(item.source)
    dtype_bytes = {"F32": 4, "F16": 2, "BF16": 2}
    if (info["dtype"] not in dtype_bytes or item.qtype not in
            (QTYPE_F32, QTYPE_F16, QTYPE_BF16)):
        return False
    row_values = 1
    for dimension in info["shape"][1:]:
        row_values *= dimension
    source_row_bytes = row_values * dtype_bytes[info["dtype"]]
    rows_per_chunk = max(1, chunk_bytes // max(source_row_bytes, 1))
    np = quantizer.np
    for row in range(0, info["shape"][0], rows_per_chunk):
        count = min(rows_per_chunk, info["shape"][0] - row)
        raw = b"".join(db.iter_read(item.source, row * source_row_bytes,
                                    count * source_row_bytes))
        if info["dtype"] == "F32":
            values = np.frombuffer(raw, dtype="<f4")
        elif info["dtype"] == "F16":
            values = np.frombuffer(raw, dtype="<f2").astype(np.float32)
        else:
            bits = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
            values = bits.view(np.float32)
        sink.write(quantizer.encode(values, item.qtype))
    return True


def compare_payload(fd, data_start, item, db, quantizer, imatrix):
    sink = CompareSink(fd, data_start + item.offset, item.nbytes, item.name)
    if write_simple_chunks(sink, item, db, quantizer):
        sink.finish()
        return
    if item.role == "engram_disk":
        write_engram(sink, item, db, quantizer.np)
    elif item.role == "engram_q4k":
        write_engram_q4k(sink, item)
    elif item.is_expert:
        for expert in range(item.expert_count):
            values = quantizer.to_f32(db, item.source.format(expert=expert))
            importance = imatrix.expert(
                item.name, expert, item.shape[0], item.expert_count)
            sink.write(quantizer.encode(values, item.qtype, importance))
    else:
        values = quantizer.to_f32(db, item.source, item.row_start, item.row_count)
        sink.write(quantizer.encode(values, item.qtype))
    sink.finish()


def report_names(label, names, all_names):
    if not names:
        return
    shown = names if all_names else names[:10]
    print(f"{label} ({len(names)}):")
    for name in shown:
        print(f"  {name}")
    if len(shown) != len(names):
        print(f"  ... {len(names) - len(shown)} more; use --all-names")


def validate(args):
    started = time.monotonic()
    progress = load_progress(args.progress)
    config = json.loads((Path(args.hf) / "config.json").read_text())
    sidecars = load_engram_q4k(config, args.engram_q4k, args.engram_q4k_dir)
    if not args.engram_q4k_external:
        raise ValueError("partial validation currently requires --engram-q4k-external")
    index_path = os.path.join(args.hf, "model.safetensors.index.json")
    headers_path = args.headers or default_headers_path(args.gguf)
    headers = load_cached_headers(headers_path, index_path, args.source_revision)
    validate_progress_shards(progress, headers)
    db = SourceDB(args.hf, index_validator=lambda _: None,
                  scale_validator=validate_scales, skip_tensors=skipped_tensor,
                  tensor_headers=headers)
    try:
        plan = build_plan(db, config, args.quant, sidecars, True)
        header = read_header(args.gguf)
        layout_matched, layout_mismatches = compare_layout(header, plan)
        print(f"plan: matched={layout_matched} "
              f"mismatched={len(layout_mismatches)} total={len(plan)}")
        for mismatch in layout_mismatches:
            print(f"layout mismatch: {mismatch}")
        if layout_mismatches:
            raise ValueError("partial GGUF tensor plan does not match")
        engram_status = validate_metadata(header, args, sidecars)
        completed = progress["completed_tensor_count"]
        if completed > len(plan) or progress["completed_tensors"] != [
                item.name for item in plan[:completed]]:
            raise ValueError("progress completed tensors are not the planned prefix")
        expected_offset = header["data_start"]
        if completed:
            item = plan[completed - 1]
            expected_offset += item.offset + align(item.nbytes, GGUF_ALIGNMENT)
        if progress["output_offset"] != expected_offset:
            raise ValueError(
                f"progress output_offset {progress['output_offset']} != {expected_offset}")

        fd = os.open(args.gguf, os.O_RDONLY)
        try:
            snapshot_size = os.fstat(fd).st_size
            safe_limit = min(snapshot_size, progress["output_offset"])
            if expected_offset > safe_limit:
                raise ValueError(
                    f"durable prefix ends at {expected_offset}, beyond safe limit {safe_limit}")
            print(f"progress: version={progress['version']} completed={completed} "
                  f"output_offset={progress['output_offset']}")
            print(f"snapshot: file_size={snapshot_size} safe_read_limit={safe_limit}")

            candidates = []
            skipped = defaultdict(list)
            for item in plan[:completed]:
                end = header["data_start"] + item.offset + align(
                    item.nbytes, GGUF_ALIGNMENT)
                if end > safe_limit:
                    raise ValueError(f"{item.name}: completed range exceeds safe limit")
                shards = {db.info(name)["shard"] for name in item_source_names(item, db)}
                missing = [shard for shard in shards
                           if not os.path.isfile(os.path.join(args.hf, shard))]
                if not missing:
                    try:
                        # An open descriptor remains readable if the converter
                        # unlinks its completed source shard concurrently.
                        for shard in shards:
                            db._fd(shard)
                    except FileNotFoundError:
                        missing = [shard for shard in shards
                                   if not os.path.isfile(os.path.join(args.hf, shard))]
                    else:
                        candidates.append(item)
                        continue
                if any(os.path.exists(os.path.join(args.hf, shard) + ".download")
                       for shard in missing):
                    reason = "shard still downloading"
                elif all(shard in progress["completed_shard_names"] for shard in missing):
                    reason = "shard already deleted"
                else:
                    reason = "source shard missing"
                skipped[reason].append(item.name)
            if args.skip_experts:
                skipped["expert comparison disabled"].extend(
                    item.name for item in candidates if item.is_expert)
                candidates = [item for item in candidates if not item.is_expert]
            if args.limit is not None and len(candidates) > args.limit:
                skipped["comparison limit"].extend(
                    item.name for item in candidates[args.limit:])
                candidates = candidates[:args.limit]

            matched = []
            mismatched = []
            if candidates:
                quantizer = NativeQuantizer(args.quants_library)
                imatrix = Imatrix(args.imatrix, quantizer.np)
                for item in candidates:
                    try:
                        compare_payload(fd, header["data_start"], item, db,
                                        quantizer, imatrix)
                    except ValueError as error:
                        mismatched.append((item.name, item.offset, str(error)))
                    else:
                        matched.append(item.name)
        finally:
            os.close(fd)
    finally:
        db.close()

    print(f"payload: matched={len(matched)} mismatched={len(mismatched)} "
          f"not_compared={sum(map(len, skipped.values()))}")
    report_names("byte-matched tensors", matched, args.all_names)
    for reason, names in sorted(skipped.items()):
        print(f"not compared ({reason}): {len(names)} tensors")
        if args.all_names:
            report_names(reason, names, True)
    for name, offset, error in mismatched:
        print(f"payload mismatch: {name} tensor_offset={offset}: {error}")
    print(f"Engram: {engram_status}")
    print(f"elapsed_seconds: {time.monotonic() - started:.3f}")
    if mismatched:
        raise ValueError(f"{len(mismatched)} tensor payloads differ")
    print("PASS: durable partial GGUF prefix validated")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--progress", required=True)
    parser.add_argument("--hf", required=True)
    parser.add_argument("--headers",
                        help="header cache; defaults beside the partial GGUF")
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--quant", choices=QUANTIZATION, default="q2")
    parser.add_argument("--imatrix")
    parser.add_argument("--limit", type=int,
                        help="compare at most N source-available tensors")
    parser.add_argument("--skip-experts", action="store_true",
                        help="skip costly full routed-expert reconstruction")
    parser.add_argument("--all-names", action="store_true")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--engram-q4k", action="append", metavar="[LAYER=]FILE")
    source.add_argument("--engram-q4k-dir", metavar="DIR")
    parser.add_argument("--engram-q4k-external", action="store_true")
    suffix = "dylib" if sys.platform == "darwin" else "so"
    parser.add_argument(
        "--quants-library",
        default=str(Path(__file__).with_name(f"libds4quants.{suffix}")))
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be nonnegative")
    return args


def main():
    args = parse_args()
    try:
        validate(args)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        sys.exit(f"deepseek41-validate-partial: {error}")


if __name__ == "__main__":
    main()
