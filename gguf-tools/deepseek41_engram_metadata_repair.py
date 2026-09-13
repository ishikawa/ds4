#!/usr/bin/env python3
"""Add external Q4_K Engram integrity metadata without moving GGUF tensors."""

import argparse
import json
import os
import struct
import sys

from deepseek41_metadata import (
    ENGRAM_SAMPLE_SCHEME, GGUF_ALIGNMENT, array_record,
)
from deepseek41_quantize import load_engram_q4k
from glm53_quantize import (
    GGUF_ARRAY, GGUF_BOOL, GGUF_FLOAT32, GGUF_STRING, GGUF_UINT32,
    GGUF_UINT64, align, kv_string, pack_string, read_exact, read_gguf_string,
    read_u32, read_u64,
)


INTEGRITY_KEYS = (
    "deepseek41.engram.sidecar_size",
    "deepseek41.engram.sidecar_rows",
    "deepseek41.engram.sidecar_sample_sha256",
    "deepseek41.engram.sidecar_sample_scheme",
)
DECODE_KEYS = {
    "general.architecture",
    "general.alignment",
    "general.source.revision",
    "deepseek41.config",
    "deepseek41.calibration",
    "deepseek41.quantization",
    "deepseek41.engram.encoding",
    "deepseek41.engram.storage",
    "deepseek41.engram.external_paths",
    "deepseek41.engram.external_offsets",
    *INTEGRITY_KEYS,
}
SCALAR_FORMATS = {
    0: "B", 1: "b", 2: "H", 3: "h", GGUF_UINT32: "I", 5: "i",
    GGUF_FLOAT32: "f", GGUF_BOOL: "B", GGUF_UINT64: "Q", 11: "q", 12: "d",
}


def read_value(fp, kind, label):
    if kind == GGUF_STRING:
        return read_gguf_string(fp, label)
    if kind == GGUF_ARRAY:
        element = read_u32(fp, f"{label} array type")
        count = read_u64(fp, f"{label} array count")
        if count > 1 << 32:
            raise ValueError(f"unreasonable {label} array count {count}")
        if element == GGUF_ARRAY:
            raise ValueError(f"unsupported GGUF array element type {element} for {label}")
        return element, [read_value(fp, element, f"{label}[{index}]")
                         for index in range(count)]
    fmt = SCALAR_FORMATS.get(kind)
    if fmt is None:
        raise ValueError(f"unsupported GGUF value type {kind} for {label}")
    return struct.unpack("<" + fmt,
                         read_exact(fp, struct.calcsize("<" + fmt), label))[0]


def skip_value(fp, kind, label):
    if kind == GGUF_STRING:
        length = read_u64(fp, f"{label} length")
        if length > 1 << 30:
            raise ValueError(f"unreasonable {label} length {length}")
        fp.seek(length, os.SEEK_CUR)
        return
    if kind == GGUF_ARRAY:
        element = read_u32(fp, f"{label} array type")
        count = read_u64(fp, f"{label} array count")
        if count > 1 << 32:
            raise ValueError(f"unreasonable {label} array count {count}")
        if element == GGUF_ARRAY:
            raise ValueError(f"unsupported GGUF array element type {element} for {label}")
        if element == GGUF_STRING:
            for index in range(count):
                skip_value(fp, element, f"{label}[{index}]")
            return
        fmt = SCALAR_FORMATS.get(element)
        if fmt is None:
            raise ValueError(f"unsupported GGUF array element type {element} for {label}")
        fp.seek(count * struct.calcsize("<" + fmt), os.SEEK_CUR)
        return
    fmt = SCALAR_FORMATS.get(kind)
    if fmt is None:
        raise ValueError(f"unsupported GGUF value type {kind} for {label}")
    fp.seek(struct.calcsize("<" + fmt), os.SEEK_CUR)


def read_header(path):
    records = []
    tensors = []
    values = {}
    with open(path, "rb") as fp:
        if read_exact(fp, 4, "GGUF magic") != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file")
        version = read_u32(fp, "GGUF version")
        if version != 3:
            raise ValueError(f"{path}: expected GGUF v3, got {version}")
        tensor_count = read_u64(fp, "GGUF tensor count")
        metadata_count = read_u64(fp, "GGUF metadata count")
        keys = set()
        for _ in range(metadata_count):
            start = fp.tell()
            key = read_gguf_string(fp, "GGUF metadata key")
            if key in keys:
                raise ValueError(f"duplicate GGUF metadata key: {key}")
            keys.add(key)
            kind = read_u32(fp, f"{key} type")
            if key in DECODE_KEYS:
                values[key] = (kind, read_value(fp, kind, key))
            else:
                skip_value(fp, kind, key)
            end = fp.tell()
            fp.seek(start)
            records.append((key, read_exact(fp, end - start, key)))
        tensor_info_start = fp.tell()
        for index in range(tensor_count):
            name = read_gguf_string(fp, f"tensor {index} name")
            rank = read_u32(fp, f"tensor {index} rank")
            if rank > 4:
                raise ValueError(f"tensor {index}: unreasonable rank {rank}")
            shape = tuple(read_u64(fp, f"tensor {index} dimension")
                          for _ in range(rank))
            qtype = read_u32(fp, f"tensor {index} type")
            offset = read_u64(fp, f"tensor {index} offset")
            tensors.append((name, shape, qtype, offset))
        tensor_info_end = fp.tell()
        alignment = values.get("general.alignment", (None, GGUF_ALIGNMENT))
        if alignment[0] != GGUF_UINT32 or not alignment[1]:
            raise ValueError("general.alignment must be a nonzero uint32")
        data_start = align(tensor_info_end, alignment[1])
        size = os.fstat(fp.fileno()).st_size
        if data_start > size:
            raise ValueError(f"{path}: tensor data starts beyond end of file")
        fp.seek(tensor_info_start)
        tensor_records = read_exact(fp, tensor_info_end - tensor_info_start,
                                    "GGUF tensor information")
        fp.seek(0)
        original = read_exact(fp, data_start, "GGUF header")
    return {
        "tensor_count": tensor_count,
        "records": records,
        "keys": keys,
        "values": values,
        "tensor_records": tensor_records,
        "tensors": tensors,
        "data_start": data_start,
        "original": original,
    }


def require_value(header, key, kind):
    actual_kind, value = header["values"].get(key, (None, None))
    if actual_kind != kind:
        raise ValueError(f"{key} must have GGUF type {kind}")
    return value


def require_array(header, key, element):
    value = require_value(header, key, GGUF_ARRAY)
    if value[0] != element:
        raise ValueError(f"{key} must have GGUF array element type {element}")
    return value[1]


def make_plan(path, files=None, directory=None, relative=False):
    header = read_header(path)
    if require_value(header, "deepseek41.engram.storage", GGUF_STRING) != "external":
        raise ValueError("only external Engram GGUF files can be repaired")
    old_paths = require_array(header, "deepseek41.engram.external_paths", GGUF_STRING)
    offsets = require_array(header, "deepseek41.engram.external_offsets", GGUF_UINT64)
    if any(key in header["keys"] for key in INTEGRITY_KEYS):
        raise ValueError("GGUF already contains Engram sidecar integrity metadata")
    config_text = require_value(header, "deepseek41.config", GGUF_STRING)
    try:
        config = json.loads(config_text)
    except json.JSONDecodeError as error:
        raise ValueError("deepseek41.config is not valid JSON") from error
    sidecars = load_engram_q4k(config, files, directory)
    layers = config.get("text_config", {}).get("engram_layer_ids")
    if len(old_paths) != len(offsets) or len(sidecars) != len(old_paths):
        raise ValueError("sidecar count differs from external_paths/external_offsets")
    if [item.layer for item in sidecars] != layers:
        raise ValueError("sidecar layer order differs from deepseek41.config")
    if [item.offset for item in sidecars] != offsets:
        raise ValueError("sidecar offsets differ from deepseek41.engram.external_offsets")

    base = os.path.dirname(os.path.abspath(path))
    new_paths = [os.path.relpath(item.path, base) if relative else item.path
                 for item in sidecars]
    integrity = [item.integrity(False) for item in sidecars]
    additions = {
        INTEGRITY_KEYS[0]: [item["size"] for item in integrity],
        INTEGRITY_KEYS[1]: [item["rows"] for item in integrity],
        INTEGRITY_KEYS[2]: [item["sample_sha256"] for item in integrity],
        INTEGRITY_KEYS[3]: ENGRAM_SAMPLE_SCHEME,
    }
    replacement = (pack_string("deepseek41.engram.external_paths")
                   + struct.pack("<IIQ", GGUF_ARRAY, GGUF_STRING, len(new_paths))
                   + b"".join(pack_string(value) for value in new_paths))
    records = [replacement if key == "deepseek41.engram.external_paths" else raw
               for key, raw in header["records"]]
    records.extend([
        array_record(INTEGRITY_KEYS[0], GGUF_UINT64, additions[INTEGRITY_KEYS[0]]),
        array_record(INTEGRITY_KEYS[1], GGUF_UINT64, additions[INTEGRITY_KEYS[1]]),
        array_record(INTEGRITY_KEYS[2], GGUF_STRING, additions[INTEGRITY_KEYS[2]]),
        kv_string(INTEGRITY_KEYS[3], additions[INTEGRITY_KEYS[3]]),
    ])
    rebuilt = (b"GGUF" + struct.pack("<IQQ", 3, header["tensor_count"], len(records))
               + b"".join(records) + header["tensor_records"])
    available = header["data_start"] - len(rebuilt)
    return header, rebuilt, old_paths, new_paths, additions, available


def print_plan(path, old_paths, new_paths, additions, available):
    status = "fits" if available >= 0 else "does not fit"
    detail = f"{available} spare bytes" if available >= 0 else f"{-available} bytes short"
    print(f"header: {status} ({detail})")
    print("changed deepseek41.engram.external_paths:")
    print(f"  before: {json.dumps(old_paths)}")
    print(f"  after:  {json.dumps(new_paths)}")
    for key in INTEGRITY_KEYS:
        print(f"added {key}: {json.dumps(additions[key])}")
    print(f"target: {path}")


def repair(path, files=None, directory=None, relative=False, dry_run=False):
    header, rebuilt, old_paths, new_paths, additions, available = make_plan(
        path, files, directory, relative)
    print_plan(path, old_paths, new_paths, additions, available)
    if available < 0:
        raise ValueError(
            f"new header needs {-available} more bytes; tensor data relocation is required")
    if dry_run:
        return
    backup = str(path) + ".header.bak"
    if os.path.exists(backup):
        raise ValueError(f"refusing to overwrite header backup {backup}")
    with open(path, "rb") as fp:
        if read_exact(fp, header["data_start"], "current GGUF header") != header["original"]:
            raise ValueError("GGUF header changed while repair was being prepared")
    with open(backup, "xb") as fp:
        fp.write(header["original"])
        fp.flush()
        os.fsync(fp.fileno())
    padded = rebuilt + bytes(available)
    with open(path, "r+b") as fp:
        fp.write(padded)
        fp.flush()
        os.fsync(fp.fileno())
    print(f"backup: {backup}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--engram-q4k", action="append", metavar="[LAYER=]FILE",
                        help="Q4_K sidecar; repeat once per Engram layer")
    source.add_argument("--engram-q4k-dir", metavar="DIR",
                        help="directory containing engram-{layer}.q4_k.bin")
    paths = parser.add_mutually_exclusive_group(required=True)
    paths.add_argument("--relative", action="store_true",
                       help="store sidecars relative to the GGUF directory")
    paths.add_argument("--absolute", action="store_true",
                       help="store normalized absolute sidecar paths")
    parser.add_argument("--dry-run", action="store_true",
                        help="report header fit and metadata changes without writing")
    args = parser.parse_args()
    repair(args.gguf, args.engram_q4k, args.engram_q4k_dir,
           relative=args.relative, dry_run=args.dry_run)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        sys.exit(f"deepseek41-engram-metadata-repair: {error}")
