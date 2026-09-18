#!/usr/bin/env python3
"""Run a small, repeatable quality and performance A/B against ds4-server."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


QUALITY_PROMPTS = (
    "Return only a JSON object with keys answer and reason: What is 17 * 23?",
    "Write a Python function named clamp(x, low, high). Return only the code.",
)


def parse_assignment(value: str) -> tuple[str, str]:
    key, sep, item = value.partition("=")
    if not sep or not key:
        raise argparse.ArgumentTypeError("expected NAME=VALUE")
    return key, item


def port_is_free(host: str, port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) != 0


def wait_ready(proc: subprocess.Popen[bytes], host: str, port: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with status {proc.returncode}")
        if not port_is_free(host, port):
            return
        time.sleep(0.25)
    raise TimeoutError(f"server did not listen on {host}:{port}")


def post_json(url: str, body: dict, timeout: float):
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(request, timeout=timeout)


def response_text(message: dict) -> str:
    return (message.get("reasoning_content") or "") + (message.get("content") or "")


def quality_request(url: str, prompt: str, tokens: int, timeout: float) -> dict:
    body = {
        "model": "smoke",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": tokens,
        "stream": False,
        "think": False,
    }
    started = time.monotonic()
    with post_json(url, body, timeout) as response:
        result = json.load(response)
    text = response_text(result["choices"][0]["message"])
    return {
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "text": text,
        "seconds": time.monotonic() - started,
        "usage": result.get("usage"),
    }


def stream_request(url: str, messages: list[dict], tokens: int, timeout: float) -> dict:
    body = {
        "model": "smoke",
        "messages": messages,
        "temperature": 0,
        "max_tokens": tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "think": False,
    }
    started = time.monotonic()
    first = None
    fragments: list[str] = []
    usage = None
    with post_json(url, body, timeout) as response:
        for raw in response:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            usage = event.get("usage") or usage
            choices = event.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            fragment = (delta.get("reasoning_content") or "") + (delta.get("content") or "")
            if fragment:
                if first is None:
                    first = time.monotonic()
                fragments.append(fragment)
    ended = time.monotonic()
    if first is None:
        raise RuntimeError("stream returned no generated text")
    after_first = ended - first
    generated = usage.get("completion_tokens") if usage else len(fragments)
    return {
        "ttft_seconds": first - started,
        "total_seconds": ended - started,
        "decode_tokens_per_second_after_first":
            max(0, generated - 1) / after_first if after_first > 0 else None,
        "sha256": hashlib.sha256("".join(fragments).encode()).hexdigest(),
        "usage": usage,
    }


def performance_request(url: str, prefix_words: int, tokens: int, timeout: float) -> dict:
    words = "alpha beta gamma delta epsilon zeta eta theta "
    prefix = (words * ((prefix_words + 7) // 8)).split()[:prefix_words]
    base = " ".join(prefix)
    # Prime live KV once, then time a continuation resembling an agent turn.
    quality_request(url, base, 1, timeout)
    messages = [
        {"role": "user", "content": base},
        {"role": "assistant", "content": "Acknowledged."},
        {"role": "user", "content": "Continue with a concise technical explanation of cache locality."},
    ]
    return stream_request(url, messages, tokens, timeout)


def run_condition(args, name: str, server: Path, assignments: list[tuple[str, str]]) -> dict:
    env = os.environ.copy()
    env.update(assignments)
    log_path = args.output.parent / f"{args.output.stem}-{name}.log"
    command = [str(server.resolve()), "--model", str(args.model.resolve()), "--host", args.host,
               "--port", str(args.port), "--ctx", str(args.context), *args.server_arg]
    with log_path.open("wb") as log:
        proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            wait_ready(proc, args.host, args.port, args.startup_timeout)
            url = f"http://{args.host}:{args.port}/v1/chat/completions"
            quality = [quality_request(url, prompt, args.quality_tokens, args.timeout)
                       for prompt in QUALITY_PROMPTS]
            performance = performance_request(
                url, args.prefix_words, args.decode_tokens, args.timeout
            )
            return {"quality": quality, "performance": performance,
                    "log": str(log_path), "command": command}
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--baseline-server", type=Path, default=Path("./ds4-server"))
    parser.add_argument("--candidate-server", type=Path, default=Path("./ds4-server"))
    parser.add_argument("--baseline-env", action="append", type=parse_assignment, default=[])
    parser.add_argument("--candidate-env", action="append", type=parse_assignment, default=[])
    parser.add_argument("--server-arg", action="append", default=[])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4989)
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--quality-tokens", type=int, default=48)
    parser.add_argument("--prefix-words", type=int, default=1000)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--max-regression-percent", type=float, default=10.0)
    parser.add_argument("--startup-timeout", type=float, default=120.0)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--output", type=Path, default=Path("smoke-quality-perf.json"))
    args = parser.parse_args()

    if not port_is_free(args.host, args.port):
        parser.error(f"{args.host}:{args.port} is already in use; stop that server first")
    for path in (args.model, args.baseline_server, args.candidate_server):
        if not path.exists():
            parser.error(f"not found: {path}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    baseline = run_condition(args, "baseline", args.baseline_server, args.baseline_env)
    candidate = run_condition(args, "candidate", args.candidate_server, args.candidate_env)
    quality_match = all(
        a["sha256"] == b["sha256"]
        for a, b in zip(baseline["quality"], candidate["quality"])
    )
    before = baseline["performance"]["decode_tokens_per_second_after_first"]
    after = candidate["performance"]["decode_tokens_per_second_after_first"]
    change = (after / before - 1.0) * 100.0
    passed = quality_match and change >= -args.max_regression_percent
    report = {
        "baseline": baseline,
        "candidate": candidate,
        "comparison": {
            "quality_exact_match": quality_match,
            "decode_change_percent": change,
            "max_regression_percent": args.max_regression_percent,
            "passed": passed,
        },
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["comparison"], indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
