#!/usr/bin/env python3
"""Convert DeepSeek V4.1 Flash while retaining only a few source shards.

The GGUF layout is fixed from small HTTP range reads of the safetensors
headers. Full text shards are then downloaded on demand and deleted after their
last output tensor is durable. Engram dense tensors use range reads because
they share shards with the omitted full-size embedding tables.
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import threading
from urllib.parse import quote

from deepseek41_metadata import GGUF_ALIGNMENT, metadata
from deepseek41_quantize import (
    QUANTIZATION, build_plan, load_engram_q4k, scale_name, write_gguf,
)
from glm53_manifest import load_index, load_safetensors_header, parse_safetensors_header
from glm53_quantize import SourceDB, align, kv_string, print_plan


OMITTED_PREFIXES = ("mtp.", "vision.", "aligner.", "image_")


def skipped_tensor(name):
    return name.startswith(OMITTED_PREFIXES) or ".engram.embed." in name


def atomic_json(path, value):
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as fp:
        json.dump(value, fp, sort_keys=True)
        fp.write("\n")
        fp.flush()
        os.fsync(fp.fileno())
    os.replace(temporary, path)


class LocalFetcher:
    """Offline fetcher used by tests and local mirrors."""

    def __init__(self, source_dir):
        self.source_dir = source_dir

    def header(self, shard):
        path = os.path.join(self.source_dir, shard)
        size = os.path.getsize(path)
        return {"size": size, "tensors": load_safetensors_header(path)}

    def fetch_file(self, name, destination):
        temporary = destination + ".download"
        shutil.copyfile(os.path.join(self.source_dir, name), temporary)
        os.replace(temporary, destination)

    def download(self, shard, destination, expected_size):
        source = os.path.join(self.source_dir, shard)
        temporary = destination + ".download"
        with open(source, "rb") as src, open(temporary, "wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 20)
            dst.flush()
            os.fsync(dst.fileno())
        if os.path.getsize(temporary) != expected_size:
            raise ValueError(f"wrong downloaded size for {shard}")
        os.replace(temporary, destination)

    def read_range(self, shard, offset, size):
        with open(os.path.join(self.source_dir, shard), "rb") as fp:
            fp.seek(offset)
            data = fp.read(size)
        if len(data) != size:
            raise ValueError(f"short local range read for {shard}")
        return data

    def cancel(self):
        pass


class CurlFetcher:
    def __init__(self, repo, revision, token, temporary_dir="/tmp"):
        self.base = (f"https://huggingface.co/{quote(repo, safe='/')}/resolve/"
                     f"{quote(revision, safe='')}")
        self.token = token
        self.temporary_dir = temporary_dir
        self.lock = threading.Lock()
        self.processes = set()

    def url(self, shard):
        return f"{self.base}/{quote(shard, safe='')}"

    def run(self, arguments, capture=False):
        command = ["curl", "-fsSL", "--retry", "5", "--retry-all-errors",
                   "--config", "-", *arguments]
        config = (f'header = "Authorization: Bearer {self.token}"\n'.encode()
                  if self.token else b"")
        process = subprocess.Popen(command, stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE if capture else None,
                                   stderr=subprocess.PIPE if capture else None)
        with self.lock:
            self.processes.add(process)
        try:
            stdout, stderr = process.communicate(config)
            if process.returncode:
                detail = stderr.decode(errors="replace").strip() if stderr else ""
                raise ValueError(f"curl failed ({process.returncode}): {detail}")
            return stdout
        finally:
            with self.lock:
                self.processes.discard(process)

    def range_to_file(self, shard, start, end, destination):
        self.run(["--range", f"{start}-{end}", "--output", destination,
                  self.url(shard)])
        data = Path(destination).read_bytes()
        os.unlink(destination)
        if len(data) != end - start + 1:
            raise ValueError(f"server did not honor byte range for {shard}")
        return data

    def header(self, shard):
        temporary = os.path.join(self.temporary_dir,
                                 f".ds41-header-{os.getpid()}-{threading.get_ident()}")
        first = self.range_to_file(shard, 0, 7, temporary)
        length = struct.unpack("<Q", first)[0]
        if length > 1 << 30:
            raise ValueError(f"unreasonable safetensors header in {shard}")
        raw = first + self.range_to_file(shard, 8, 7 + length, temporary)
        # A one-byte suffix request yields the total size in Content-Range only
        # with extra header parsing, so use the final tensor end encoded in the
        # contiguous safetensors header as the exact file size.
        document = json.loads(raw[8:])
        payload = max((entry["data_offsets"][1] for name, entry in document.items()
                       if name != "__metadata__"), default=0)
        file_size = 8 + length + payload
        return {"size": file_size,
                "tensors": parse_safetensors_header(raw, file_size, shard)}

    def fetch_file(self, name, destination):
        temporary = destination + ".download"
        self.run(["--output", temporary, self.url(name)])
        os.replace(temporary, destination)

    def download(self, shard, destination, expected_size):
        temporary = destination + ".download"
        self.run(["-C", "-", "--output", temporary, self.url(shard)])
        if os.path.getsize(temporary) != expected_size:
            raise ValueError(f"wrong downloaded size for {shard}")
        os.replace(temporary, destination)

    def read_range(self, shard, offset, size):
        temporary = os.path.join(self.temporary_dir,
                                 f".ds41-range-{os.getpid()}-{threading.get_ident()}")
        return self.range_to_file(shard, offset, offset + size - 1, temporary)

    def cancel(self):
        with self.lock:
            processes = list(self.processes)
        for process in processes:
            process.terminate()


def load_headers(index_path, fetcher, cache_path, shards, revision):
    index_digest = hashlib.sha256(Path(index_path).read_bytes()).hexdigest()
    identity = {"version": 1, "revision": revision, "index_sha256": index_digest}
    cached = {}
    if os.path.exists(cache_path):
        document = json.loads(Path(cache_path).read_text())
        if {key: document.get(key) for key in identity} != identity:
            raise ValueError(f"header cache does not match source: {cache_path}")
        cached = document.get("shards", {})
    for shard in sorted(shards):
        if shard not in cached:
            cached[shard] = fetcher.header(shard)
            atomic_json(cache_path, dict(identity, shards=cached))
    return cached


def validate_index(config, weight_map):
    text = config["text_config"]
    layers = text["num_hidden_layers"]
    experts = text["n_routed_experts"]
    found = {layer: [] for layer in range(layers)}
    pattern = re.compile(r"layers\.(\d+)\.ffn\.experts\.(\d+)\.")
    for name, shard in weight_map.items():
        match = pattern.match(name)
        if not match:
            continue
        layer, expert = map(int, match.groups())
        if layer not in found or expert >= experts:
            raise ValueError(f"expert tensor is outside config: {name}")
        found[layer].append((name, shard))
    layer_shards = {}
    for layer, entries in found.items():
        shards = {shard for _, shard in entries}
        if len(entries) != experts * 6 or len(shards) != 1:
            raise ValueError(f"layer {layer} expert mapping is incomplete or split")
        layer_shards[layer] = shards.pop()
    if len(set(layer_shards.values())) != layers:
        raise ValueError("expert layers do not map one-to-one onto source shards")
    return layer_shards


def item_source_names(item, db):
    if item.role == "engram_q4k":
        return []
    if item.is_expert:
        names = [item.source.format(expert=expert)
                 for expert in range(item.expert_count)]
    else:
        names = [item.source]
    result = []
    for name in names:
        result.append(name)
        if db.info(name)["dtype"] in ("I8", "F8_E4M3"):
            result.append(scale_name(name))
    return result


class StreamingSourceDB(SourceDB):
    def __init__(self, hf_dir, headers, fetcher, prefetch, min_free_gib,
                 progress_path):
        super().__init__(hf_dir, index_validator=lambda _: None,
                         scale_validator=lambda _: None,
                         skip_tensors=skipped_tensor, tensor_headers=headers)
        self.headers = headers
        self.fetcher = fetcher
        self.prefetch = prefetch
        self.reserve = int(min_free_gib * (1 << 30))
        self.progress_path = progress_path
        self.partial_shards = {
            shard for shard, header in headers.items()
            if any(".engram.embed." in name for name in header["tensors"])
        }
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=prefetch + 1)
        self.space_lock = threading.Lock()
        self.reserved_bytes = 0
        self.futures = {}
        self.owned = set()
        self.item_shards = []
        self.last_item = {}
        self.completed_shards = set()
        self.plan = []
        self.data_start = 0

    def shard_for_item(self, item):
        shards = {self.info(name)["shard"] for name in item_source_names(item, self)}
        if len(shards) > 1:
            raise ValueError(f"{item.name}: source tensors span shards {sorted(shards)}")
        return next(iter(shards), None)

    def conversion_started(self, plan, completed, data_start):
        self.set_plan(plan, data_start)
        self.completed_shards = {shard for shard, last in self.last_item.items()
                                 if last <= completed}
        self.save_progress(completed,
                           data_start + (plan[completed - 1].offset +
                           align(plan[completed - 1].nbytes, GGUF_ALIGNMENT)
                           if completed else 0), False)
        self.schedule(completed)

    def set_plan(self, plan, data_start):
        self.plan = plan
        self.data_start = data_start
        self.item_shards = [self.shard_for_item(item) for item in plan]
        self.last_item = {shard: index + 1 for index, shard in enumerate(self.item_shards)
                          if shard is not None}

    def schedule(self, start):
        wanted = []
        for shard in self.item_shards[start:]:
            if shard and shard not in self.partial_shards and shard not in wanted:
                wanted.append(shard)
            if len(wanted) >= self.prefetch + 1:
                break
        for shard in wanted:
            if shard not in self.futures:
                self.futures[shard] = self.executor.submit(self.acquire, shard)

    def acquire(self, shard):
        destination = os.path.join(self.hf_dir, shard)
        expected = self.headers[shard]["size"]
        if os.path.exists(destination):
            if os.path.getsize(destination) != expected:
                raise ValueError(f"existing shard has wrong size: {destination}")
            if load_safetensors_header(destination) != self.headers[shard]["tensors"]:
                raise ValueError(f"existing shard has wrong header: {destination}")
            if os.path.exists(destination + ".stream-owned"):
                self.owned.add(shard)
            return destination
        temporary = destination + ".download"
        partial = os.path.getsize(temporary) if os.path.exists(temporary) else 0
        if partial > expected:
            os.unlink(temporary)
            partial = 0
        needed = expected - partial
        with self.space_lock:
            free = shutil.disk_usage(self.hf_dir).free
            if free < needed + self.reserved_bytes + self.reserve:
                raise ValueError(f"free space would fall below --min-free-gib before {shard}")
            self.reserved_bytes += needed
        try:
            Path(destination + ".stream-owned").touch()
            self.fetcher.download(shard, destination, expected)
        finally:
            with self.space_lock:
                self.reserved_bytes -= needed
        if load_safetensors_header(destination) != self.headers[shard]["tensors"]:
            os.unlink(destination)
            os.unlink(destination + ".stream-owned")
            raise ValueError(f"downloaded shard has wrong header: {destination}")
        self.owned.add(shard)
        return destination

    def prepare_item(self, item, index):
        shard = self.item_shards[index]
        if shard and shard not in self.partial_shards:
            future = self.futures.get(shard)
            if future is None:
                future = self.executor.submit(self.acquire, shard)
                self.futures[shard] = future
            future.result()
        self.schedule(index)

    def read(self, name):
        info = self.info(name)
        if info["shard"] in self.partial_shards:
            if shutil.disk_usage(self.hf_dir).free < info["nbytes"] + self.reserve:
                raise ValueError("free space would fall below --min-free-gib for range read")
            return self.fetcher.read_range(info["shard"], info["offset"], info["nbytes"])
        return super().read(name)

    def iter_read(self, name, byte_start=0, byte_count=None, chunk_size=16 << 20):
        info = self.info(name)
        if info["shard"] not in self.partial_shards:
            yield from super().iter_read(name, byte_start, byte_count, chunk_size)
            return
        if byte_count is None:
            byte_count = info["nbytes"] - byte_start
        for start in range(0, byte_count, chunk_size):
            size = min(chunk_size, byte_count - start)
            if shutil.disk_usage(self.hf_dir).free < size + self.reserve:
                raise ValueError("free space would fall below --min-free-gib for range read")
            yield self.fetcher.read_range(info["shard"],
                                          info["offset"] + byte_start + start, size)

    def item_completed(self, item, completed, output_offset):
        shard = self.item_shards[completed - 1]
        if shard and self.last_item[shard] == completed:
            self.completed_shards.add(shard)
            fd = self._fds.pop(shard, None)
            if fd is not None:
                os.close(fd)
            if shard in self.owned:
                os.unlink(os.path.join(self.hf_dir, shard))
                os.unlink(os.path.join(self.hf_dir, shard) + ".stream-owned")
                self.owned.remove(shard)
        self.save_progress(completed, output_offset, False)
        self.schedule(completed)

    def save_progress(self, completed, output_offset, complete):
        atomic_json(self.progress_path, {
            "version": 1,
            "complete": complete,
            "completed_tensor_count": completed,
            "completed_tensors": [item.name for item in self.plan[:completed]],
            "completed_shards": sorted(self.completed_shards),
            "output_offset": output_offset,
        })

    def conversion_finished(self, output):
        self.save_progress(len(self.plan), os.path.getsize(output), True)

    def close(self):
        self.fetcher.cancel()
        self.executor.shutdown(wait=True, cancel_futures=True)
        super().close()


def disk_peak(plan, db, headers, prefetch, data_start):
    shards = [db.shard_for_item(item) for item in plan]
    full = {shard for shard in shards if shard and shard not in db.partial_shards}
    peak = data_start
    written = data_start
    for index, item in enumerate(plan):
        window = []
        for shard in shards[index:]:
            if shard in full and shard not in window:
                window.append(shard)
            if len(window) >= prefetch + 1:
                break
        current = shards[index]
        range_bytes = (max((db.info(name)["nbytes"]
                            for name in item_source_names(item, db)), default=0)
                       if current in db.partial_shards else 0)
        peak = max(peak, written + range_bytes +
                   sum(headers[shard]["size"] for shard in window))
        written = data_start + item.offset + align(item.nbytes, GGUF_ALIGNMENT)
    return max(peak, written)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf", required=True,
                        help="staging directory; missing small metadata is downloaded")
    parser.add_argument("--out", required=True)
    parser.add_argument("--repo", default="deepseek-ai/DeepSeek-V4.1-Flash")
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--quant", choices=QUANTIZATION, default="q2")
    parser.add_argument("--imatrix")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--prefetch", type=int, choices=(1, 2), default=1,
                        help="number of next full shards downloaded concurrently")
    parser.add_argument("--min-free-gib", type=float, default=32,
                        help="stop before a download or range read crosses this reserve")
    parser.add_argument("--resume", action="store_true",
                        help="resume the matching partial GGUF and shard downloads")
    parser.add_argument("--dry-run", action="store_true",
                        help="fetch headers only and report output and peak disk sizes")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--engram-q4k", action="append", metavar="[LAYER=]FILE")
    source.add_argument("--engram-q4k-dir", metavar="DIR")
    parser.add_argument("--engram-q4k-external", action="store_true")
    parser.add_argument("--source-dir", help=argparse.SUPPRESS)
    suffix = "dylib" if sys.platform == "darwin" else "so"
    parser.add_argument("--quants-library",
                        default=str(Path(__file__).with_name(f"libds4quants.{suffix}")))
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_revision):
        parser.error("source revision must be a full commit hash")
    if not 1 <= args.threads <= 32:
        parser.error("threads must be between 1 and 32")
    if args.min_free_gib < 0:
        parser.error("--min-free-gib must not be negative")
    return args


def run(args):
    hf_dir = os.path.abspath(args.hf)
    os.makedirs(hf_dir, exist_ok=True)
    fetcher = (LocalFetcher(args.source_dir) if args.source_dir else
               CurlFetcher(args.repo, args.source_revision, os.getenv("HF_TOKEN"), hf_dir))
    for name in ("config.json", "model.safetensors.index.json", "tokenizer.json"):
        destination = os.path.join(hf_dir, name)
        if not os.path.exists(destination):
            fetcher.fetch_file(name, destination)
    index_path = os.path.join(hf_dir, "model.safetensors.index.json")
    config_path = os.path.join(hf_dir, "config.json")
    with open(config_path, "rb") as fp:
        config = json.load(fp)
    _, weight_map = load_index(index_path)
    layer_shards = validate_index(config, weight_map)
    print(f"expert shards: {len(layer_shards)} layers, one shard per layer")
    sidecars = load_engram_q4k(config, args.engram_q4k, args.engram_q4k_dir)
    required_shards = {shard for name, shard in weight_map.items()
                       if not skipped_tensor(name)}
    headers_path = args.out + ".headers.json"
    headers = load_headers(index_path, fetcher, headers_path, required_shards,
                           args.source_revision)
    storage = "external" if args.engram_q4k_external else "embedded"
    config, records = metadata(
        hf_dir, args.source_revision, "q4_k_row144", storage,
        [item.path for item in sidecars] if args.engram_q4k_external else None,
        [item.offset for item in sidecars] if args.engram_q4k_external else None)
    records.append(kv_string("deepseek41.quantization", QUANTIZATION[args.quant]))
    records.append(kv_string("deepseek41.calibration",
                             "imatrix" if args.imatrix else "weight-energy bootstrap"))
    db = StreamingSourceDB(hf_dir, headers, fetcher, args.prefetch,
                           args.min_free_gib, args.out + ".progress.json")
    try:
        plan = build_plan(db, config, args.quant, sidecars,
                          args.engram_q4k_external)
        if args.dry_run:
            data_start, data_bytes = print_plan(plan, records, [], GGUF_ALIGNMENT)
            final_size = data_start + data_bytes
            db.set_plan(plan, data_start)
            peak = disk_peak(plan, db, headers, args.prefetch, data_start)
            print(f"final_output_bytes: {final_size}")
            print(f"peak_output_plus_shards_bytes: {peak}")
            print(f"final_output_gib: {final_size / (1 << 30):.3f}")
            print(f"peak_output_plus_shards_gib: {peak / (1 << 30):.3f}")
        else:
            write_gguf(args, plan, records, db)
    finally:
        db.close()


def main():
    args = parse_args()
    try:
        run(args)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        sys.exit(f"deepseek41-stream-convert: {error}")


if __name__ == "__main__":
    main()
