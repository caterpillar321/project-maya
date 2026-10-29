#!/usr/bin/env python3
"""Offline capacity bounds and OLD single-trunk trace attribution, not a speed forecast.

Pool inputs mean usable, distinct expert bytes AFTER workspace/spares. They are
assumptions, not measured allocations on the proposed two-host configuration.
The duplicate-helper case retains all helper experts in host RAM. Both cases
are aggregate lower bounds: per-class fragmentation and other duplicates can
make real RAM requirements larger. No GPU or model process is used.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

GIB = 1 << 30


def source(path):
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def experts(path):
    result = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        row = line.split()
        layer, blob = int(row[0]), int(row[4])
        if layer in result or blob <= 0 or blob % 256:
            raise ValueError("duplicate layer or unexpected expert blob size/alignment")
        result[layer] = blob * 288
    if set(result) != set(range(3, 45)):
        raise ValueError("this GLM-specific analysis requires exactly MoE layers [3,45)")
    return result


def attribution(path):
    data = json.loads(path.read_text())
    if data.get("complete") is not True or data.get("issues"):
        raise ValueError("a complete, validated window-trace analysis is required")
    requests = {r["request"]: r for r in data["requests"]}
    if len(requests) != len(data["requests"]) or set(requests) != set(range(1, 10)):
        raise ValueError("expected one unchanged qshort pass: 4 chat, 4 code, 1 greedy")
    result = {}
    for label, ids in (("chat", range(1, 5)), ("code", range(5, 9))):
        groups = [g for i in ids for g in requests[i]["groups"]]
        if any(g["T"] != 2 or g["outcome"] != "completed" for g in groups):
            raise ValueError("expected completed T2 samples only")
        n = sum(g["sampled_attempts"] for g in groups)
        if n <= 0:
            raise ValueError("no samples")
        layers, stages, span = {}, {}, 0.0
        for g in groups:
            weight = g["sampled_attempts"] / n
            span += g["span_ms"]["mean"] * weight
            if {r["layer"] for r in g["layers"]} != set(range(45)):
                raise ValueError("incomplete layer attribution")
            for row in g["layers"]:
                k = row["layer"]
                layers[k] = layers.get(k, 0.0) + row["mean_ms_per_sampled_attempt"] * weight
            for k, value in g["stages_mean_ms"].items():
                stages[k] = stages.get(k, 0.0) + value * weight
        if not all(math.isfinite(v) and v >= 0 for v in [span, *layers.values(), *stages.values()]):
            raise ValueError("invalid interval")
        if sum(layers.values()) > span + 0.01:
            raise ValueError("layer intervals exceed their window span")
        result[label] = {"samples": n, "span_ms": span, "layers_ms": layers, "stages_ms": stages}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experts", type=Path, required=True)
    p.add_argument("--trace-analysis", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cuts", type=int, nargs="+", default=[9, 10, 11, 12, 14, 16, 18, 20, 22, 24])
    p.add_argument("--front-trunk-gib", type=float, default=10.5)
    p.add_argument("--front-helper-gib", type=float, default=14.5)
    p.add_argument("--tail-trunk-gib", type=float, default=20.5)
    p.add_argument("--tail-helper-gib", type=float, default=21.5)
    p.add_argument("--front-ram-gib", type=float, default=54)
    p.add_argument("--tail-ram-gib", type=float, default=92)
    # Existing NextN Q4_0 format, 288 experts, three [4096,2048] matrices, 18 bytes/32 values.
    p.add_argument("--nextn-expert-bytes", type=int, default=288 * 3 * 4096 * 2048 // 32 * 18)
    a = p.parse_args()
    budget = {k: v for k, v in vars(a).items() if k.endswith("_gib")}
    if any(not math.isfinite(v) or v <= 0 for v in budget.values()) or a.nextn_expert_bytes <= 0:
        p.error("budgets and NextN bytes must be finite and positive")
    if any(c < 4 or c >= 45 for c in a.cuts) or len(set(a.cuts)) != len(a.cuts):
        p.error("unique cuts must lie in [4,44]")
    weights, trace = experts(a.experts), attribution(a.trace_analysis)
    total = sum(weights.values()) + a.nextn_expert_bytes
    rows = []
    for cut in a.cuts:
        front = sum(v for k, v in weights.items() if k < cut)
        tail = total - front
        row = {"cut": cut, "front_expert_gib": front / GIB, "tail_with_nextn_expert_gib": tail / GIB}
        for host, weight in (("front", front), ("tail", tail)):
            trunk, helper, ram = (budget[f"{host}_{s}_gib"] for s in ("trunk", "helper", "ram"))
            unique = max(0.0, weight / GIB - trunk - helper)
            duplicated = max(0.0, weight / GIB - trunk)
            row[host + "_ram"] = {
                "exclusive_lower_bound_gib": unique,
                "helper_backed_lower_bound_gib": duplicated,
                "exclusive_headroom_gib": ram - unique,
                "helper_backed_headroom_gib": ram - duplicated,
            }
        row["old_trace_only"] = {}
        for label, group in trace.items():
            prefix = sum(t for k, t in group["layers_ms"].items() if k < cut)
            row["old_trace_only"][label] = {
                "front_layers_ms": prefix,
                "remaining_span_ms": group["span_ms"] - prefix,
                "front_fraction": prefix / group["span_ms"],
            }
        rows.append(row)
    payloads = [
        {"tokens": n, "bytes": n * 65536, "wire_seconds_at_2_5_gbps": n * 65536 * 8 / 2.5e9}
        for n in (2, 3, 4096, 16384, 32768, 131072, 262144)
    ]
    out = {
        "schema": 1,
        "description": "Offline bounds, not actual allocations or a new timing benchmark",
        "sources": {"experts": source(a.experts), "trace_analysis": source(a.trace_analysis)},
        "assumed_usable_expert_and_ram_budgets_gib": budget,
        "nextn_expert_bytes_derived_from_existing_format": a.nextn_expert_bytes,
        "trunk_expert_bytes": sum(weights.values()),
        "cuts": rows,
        "old_single_trunk_sampled_T2": trace,
        "uncompressed_FP32_residual_wire_lower_bounds": payloads,
        "limitations": [
            "Fixed hypothetical usable pools across cuts; actual dense/KV/workspace/spares must be measured per cut.",
            "Positive aggregate headroom is necessary, not sufficient; class fragmentation and backing policy matter.",
            "NextN weights derived from the existing Q4_0 format, not a proposed quantization change.",
            "Old trace contains all helper branches and stalls; it does not forecast 4070/5090 stage speed.",
            "Trace sample means and all-round host verify timings have different denominators.",
            "Wire times exclude protocol/RTT, GPU copies and scheduling; overlap can hide part of TTFT cost.",
        ],
    }
    a.output.write_text(json.dumps(out, indent=2) + "\n")
    print(f"Saved {len(rows)} capacity cases; {trace['chat']['samples']} chat/{trace['code']['samples']} code samples. No GPU execution.")


if __name__ == "__main__":
    main()
