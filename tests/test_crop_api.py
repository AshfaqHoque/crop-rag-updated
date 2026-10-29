#!/usr/bin/env python3
"""
Concurrent load test for the Crop_RAG /chat endpoint.

Fires N requests at once, each with a unique session_id and a unique
message, and writes a latency/status report (CSV + Markdown).

Usage:
    python3 crop_rag_load_test.py
    python3 crop_rag_load_test.py --url https://chatbot.farminsight.dev/api/v1/chat/stream --concurrency 20
    python3 crop_rag_load_test.py --url https://chatbot.farminsight.dev/api/v1/chat/stream --messages messages.json
"""

import argparse
import concurrent.futures
import csv
import json
import statistics
import time
import uuid
from pathlib import Path

import requests

# 20 unique messages covering different crops and RAG sections
# (irrigation, fertilizer, pest, variety, harvest, land prep, etc.)
DEFAULT_MESSAGES = [
    "moricher bijer haar koto",
    "বোরো ধানে সেচ দেওয়ার সঠিক নিয়ম কী?",
    "আমন ধানের কোন কোন জাত পাওয়া যায়?",
    "গমে সার প্রয়োগের মাত্রা কত?",
    "ভুট্টার পোকামাকড় দমনের উপায় কী?",
    "আলুর জমি তৈরির সঠিক পদ্ধতি কী?",
    "পেঁয়াজ রোপণের সময় কখন?",
    "রসুনের রোগবালাই দমনের জন্য কী ব্যবহার করব?",
    "সরিষা কাটার সঠিক সময় কখন?",
    "পাট চাষে জলবায়ু কেমন হলে ভালো ফলন হয়?",
    "আখে কীটনাশক ব্যবহারের নিয়ম কী?",
    "মুগ ডালের বীজ বপনের গভীরতা কত হওয়া উচিত?",
    "মসুর ডালে আগাছা দমনের উপায় কী?",
    "বেগুনে কোন সার কতটুকু দিতে হয়?",
    "টমেটোর কোন কোন জাত চাষের জন্য ভালো?",
    "তরমুজে ফল ধরার সময় কী যত্ন নিতে হয়?",
    "কুমড়ার আন্তঃপরিচর্যা কীভাবে করব?",
    "লাউয়ের মাচা তৈরির নিয়ম কী?",
    "শসার রোগ প্রতিরোধী জাত কোনগুলো?",
    "গাজর সংগ্রহের সঠিক সময় কখন?",
]


def build_payload(message: str, session_id: str, language_type: str) -> dict:
    return {
        "session_id": session_id,
        "message": message,
        "language_type": language_type,
    }


def fire_one(url: str, payload: dict, timeout: float) -> dict:
    session_id = payload["session_id"]
    message = payload["message"]
    start = time.perf_counter()
    result = {
        "session_id": session_id,
        "message": message,
        "status_code": None,
        "latency_s": None,
        "ok": False,
        "response_chars": 0,
        "error": "",
    }
    try:
        resp = requests.post(url, json=payload, timeout=timeout,
                              headers={"accept": "application/json",
                                       "Content-Type": "application/json"})
        elapsed = time.perf_counter() - start
        result["status_code"] = resp.status_code
        result["latency_s"] = round(elapsed, 3)
        result["ok"] = resp.ok
        result["response_chars"] = len(resp.text)
        if not resp.ok:
            result["error"] = resp.text[:200]
    except requests.exceptions.RequestException as e:
        elapsed = time.perf_counter() - start
        result["latency_s"] = round(elapsed, 3)
        result["error"] = str(e)[:200]
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="https://chatbot.farminsight.dev/api/v1/chat/stream")
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--language-type", default="bn")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--messages", type=str, default=None,
                     help="Optional JSON file: [message, ...]. "
                          "Defaults to 20 built-in Bangla crop questions.")
    ap.add_argument("--out-prefix", default="crop_rag_load_test_report")
    args = ap.parse_args()

    if args.messages:
        cases = json.loads(Path(args.messages).read_text(encoding="utf-8"))
    else:
        cases = DEFAULT_MESSAGES

    n = args.concurrency
    if n > len(cases):
        # cycle through the pool but keep every session_id unique
        cases = (cases * ((n // len(cases)) + 1))[:n]
    else:
        cases = cases[:n]

    payloads = [
        build_payload(msg, str(uuid.uuid4()), args.language_type)
        for msg in cases
    ]

    print(f"Firing {n} concurrent requests at {args.url} ...")
    wall_start = time.perf_counter()
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
        futures = [pool.submit(fire_one, args.url, p, args.timeout) for p in payloads]
        for fut in concurrent.futures.as_completed(futures):
            results.append(fut.result())
    wall_elapsed = time.perf_counter() - wall_start

    # order by original submission order isn't guaranteed after as_completed;
    # sort by session_id order of payloads for a stable report
    order = {p["session_id"]: i for i, p in enumerate(payloads)}
    results.sort(key=lambda r: order[r["session_id"]])

    # ---- summary stats ----
    latencies = [r["latency_s"] for r in results if r["latency_s"] is not None]
    ok_count = sum(1 for r in results if r["ok"])
    fail_count = n - ok_count

    summary = {
        "total_requests": n,
        "concurrency": n,
        "wall_clock_s": round(wall_elapsed, 3),
        "success": ok_count,
        "failed": fail_count,
        "success_rate_pct": round(100 * ok_count / n, 1) if n else 0,
        "latency_min_s": round(min(latencies), 3) if latencies else None,
        "latency_max_s": round(max(latencies), 3) if latencies else None,
        "latency_avg_s": round(statistics.mean(latencies), 3) if latencies else None,
        "latency_median_s": round(statistics.median(latencies), 3) if latencies else None,
        "latency_p95_s": round(sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)], 3) if latencies else None,
    }

    # ---- write CSV ----
    csv_path = f"{args.out_prefix}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["session_id", "message",
                                           "status_code", "latency_s", "ok",
                                           "response_chars", "error"])
        w.writeheader()
        w.writerows(results)

    # ---- write Markdown report ----
    md_path = f"{args.out_prefix}.md"
    lines = []
    lines.append(f"# Crop_RAG /chat Load Test Report\n")
    lines.append(f"- URL: `{args.url}`")
    lines.append(f"- Concurrency: {n} simultaneous requests")
    lines.append(f"- Wall-clock time for all {n} requests: **{summary['wall_clock_s']}s**")
    lines.append(f"- Success: **{summary['success']}/{n}** ({summary['success_rate_pct']}%)")
    if latencies:
        lines.append(f"- Latency (s) — min: {summary['latency_min_s']}, "
                      f"avg: {summary['latency_avg_s']}, "
                      f"median: {summary['latency_median_s']}, "
                      f"p95: {summary['latency_p95_s']}, "
                      f"max: {summary['latency_max_s']}")
    lines.append("\n## Per-request results\n")
    lines.append("| # | Message | Status | Latency (s) | Chars | Error |")
    lines.append("|---|---------|--------|--------------|-------|-------|")
    for i, r in enumerate(results, 1):
        err = r["error"].replace("|", "/").replace("\n", " ")[:60]
        lines.append(f"| {i} | {r['message']} | "
                      f"{r['status_code']} | {r['latency_s']} | "
                      f"{r['response_chars']} | {err} |")
    Path(md_path).write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nWrote {csv_path} and {md_path}")


if __name__ == "__main__":
    main()