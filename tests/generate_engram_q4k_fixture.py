#!/usr/bin/env python3
"""Generate Q4_K rows and ggml dequantization results for test_engram."""

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import random
import struct
import sys


ROWS = 32
COLS = 256
ROW_BYTES = 144
SEED = 0xD541


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ggml-lib", type=Path, required=True)
    parser.add_argument("--ggml-source-commit", required=True)
    parser.add_argument("--out-bin", type=Path,
                        default=Path("tests/fixtures/engram_q4k_ggml.bin"))
    parser.add_argument("--out-json", type=Path,
                        default=Path("tests/fixtures/engram_q4k_ggml.json"))
    args = parser.parse_args()
    if sys.byteorder != "little":
        parser.error("the fixture stores little-endian float32 values")

    library = ctypes.CDLL(str(args.ggml_lib.resolve()))
    fp = ctypes.POINTER(ctypes.c_float)
    library.quantize_row_q4_K_ref.argtypes = [fp, ctypes.c_void_p,
                                               ctypes.c_int64]
    library.quantize_row_q4_K_ref.restype = None
    library.dequantize_row_q4_K.argtypes = [ctypes.c_void_p, fp,
                                             ctypes.c_int64]
    library.dequantize_row_q4_K.restype = None

    rng = random.Random(SEED)
    values = []
    for row in range(ROWS):
        magnitude = 2.0 ** ((row % 9) - 4)
        for column in range(COLS):
            value = rng.uniform(-magnitude, magnitude)
            if column % 37 == 0:
                value = 0.0
            elif column % 53 == 0:
                value = -magnitude
            elif column % 71 == 0:
                value = magnitude
            values.append(value)
    source_bytes = struct.pack(f"<{len(values)}f", *values)
    source = (ctypes.c_float * len(values)).from_buffer_copy(source_bytes)
    packed = (ctypes.c_uint8 * (ROWS * ROW_BYTES))()
    restored = (ctypes.c_float * (ROWS * COLS))()
    library.quantize_row_q4_K_ref(source, packed, len(values))
    library.dequantize_row_q4_K(packed, restored, len(values))
    packed_bytes = bytes(packed)
    restored_bytes = bytes(restored)
    payload = packed_bytes + restored_bytes

    args.out_bin.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_bin.write_bytes(payload)
    document = {
        "format_version": 1,
        "rows": ROWS,
        "cols": COLS,
        "row_bytes": ROW_BYTES,
        "seed": SEED,
        "layout": "packed_rows_then_little_endian_float32_rows",
        "packed_bytes": len(packed_bytes),
        "expected_bytes": len(restored_bytes),
        "packed_sha256": sha256(packed_bytes),
        "expected_sha256": sha256(restored_bytes),
        "ggml_library_sha256": sha256(args.ggml_lib.read_bytes()),
        "ggml_source_commit": args.ggml_source_commit,
        "quantizer": "quantize_row_q4_K_ref",
        "dequantizer": "dequantize_row_q4_K",
    }
    args.out_json.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
