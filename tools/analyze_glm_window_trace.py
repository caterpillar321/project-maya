#!/usr/bin/env python3
"""Validate and summarize opt-in GLM MTP window trace records in an engine log.

This reports sampled CUDA stream intervals, including stalls and host enqueue
gaps. It does not estimate tokens/second, time per emitted token, pure PCIe
service, or cross-device overlap. T1, T2 and T3 are always separate groups.

Every sampled attempt must end in a window record, and every request must end
in a report record. A truncated or malformed log is rejected. Explicit engine
errors/unsupported modes are retained as issues and make the CLI exit nonzero.
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys


PREFIX = "glm window trace: "
OUTCOMES = {"completed", "scratch_retry", "aborted", "unsupported_retry", "trace_error"}


class TraceError(ValueError):
    pass


def _integer(record, name, minimum=0):
    value = record.get(name)
    if type(value) is not int or value < minimum:
        raise TraceError(f"{name}: expected an integer >= {minimum}")
    return value


def _time(record, name):
    value = record.get(name)
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise TraceError(f"{name}: expected a finite nonnegative time")
    return float(value)


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise TraceError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def read_records(lines, source="<input>"):
    records = []
    for number, line in enumerate(lines, 1):
        if PREFIX not in line:
            continue
        payload = line.split(PREFIX, 1)[1]
        try:
            record = json.loads(payload, object_pairs_hook=_object)
            if not isinstance(record, dict):
                raise TraceError("trace record is not an object")
        except (ValueError, TypeError) as error:
            raise TraceError(f"{source}:{number}: {error}") from error
        records.append(record)
    if not records:
        raise TraceError(f"{source}: no GLM window trace records")
    return records


def _identity(record):
    if type(record.get("schema")) is not int or record["schema"] != 1:
        raise TraceError("unsupported or missing schema (expected 1)")
    session = record.get("session")
    if not isinstance(session, str) or not session:
        raise TraceError("session: expected a nonempty string")
    return session, _integer(record, "device"), _integer(record, "request", 1)


def _window_identity(record):
    window = _integer(record, "window", 1)
    p0 = _integer(record, "p0")
    tokens = _integer(record, "T", 1)
    attempt = _integer(record, "attempt")
    if tokens not in (1, 2, 3):
        raise TraceError(f"unsupported traced window width: T={tokens}")
    return window, p0, tokens, attempt


def _close(left, right, count):
    # Each interval and the independently measured span are printed to 6 places.
    return abs(left - right) <= max(0.02, count * 1e-5)


def _stats(values):
    ordered = sorted(values)
    n = len(ordered)
    return {
        "sum": sum(ordered), "mean": sum(ordered) / n,
        "min": ordered[0], "max": ordered[-1],
        "p50": ordered[math.ceil(n * 0.50) - 1],
        "p95": ordered[math.ceil(n * 0.95) - 1],
    }


def _summarize(attempts):
    grouped = defaultdict(list)
    for attempt in attempts:
        end = attempt["end"]
        if end["complete"]:
            grouped[(end["device"], end["T"], end["outcome"])].append(attempt)
    result = []
    for (device, tokens, outcome), windows in sorted(grouped.items()):
        stage = defaultdict(float)
        layer = defaultdict(float)
        layer_stage = defaultdict(lambda: defaultdict(float))
        outside = 0.0
        for attempt in windows:
            for item in attempt["segments"]:
                duration = item["ms"]
                stage[item["stage"]] += duration
                if item["layer"] >= 0:
                    layer[item["layer"]] += duration
                    layer_stage[item["layer"]][item["stage"]] += duration
                else:
                    outside += duration
        count = len(windows)
        span = _stats([attempt["end"]["span_ms"] for attempt in windows])
        layers = [{
            "layer": number, "total_ms": duration,
            "mean_ms_per_sampled_attempt": duration / count,
            "stages_mean_ms": {name: value / count for name, value in sorted(layer_stage[number].items())},
        } for number, duration in sorted(layer.items())]
        result.append({
            "device": device, "T": tokens, "outcome": outcome,
            "sampled_attempts": count, "span_ms": span,
            "stages_mean_ms": {name: value / count for name, value in sorted(stage.items())},
            "outside_layers_mean_ms": outside / count,
            "layers": layers,
            "top_layers": sorted(layers, key=lambda item: (-item["total_ms"], item["layer"]))[:10],
        })
    return result


def analyze_records(records):
    """Return validated summaries; raise TraceError for invalid/incomplete data.

    Structurally valid error/unsupported reports return ``complete=False`` with
    explicit issues. Their incomplete timings are excluded from aggregates.
    """
    requests = {}
    used_windows = {}
    latest_window = {}
    issues = []
    for number, record in enumerate(records, 1):
        try:
            identity = _identity(record)
            request = requests.setdefault(identity, {"attempts": {}, "report": None})
            if request["report"] is not None:
                raise TraceError("record after the request's final report (duplicate or reset request ID)")
            kind = record.get("kind")
            if kind == "report":
                for field in ("sampled", "unsupported", "errors", "windows_seen"):
                    _integer(record, field)
                _integer(record, "every", 1)
                _integer(record, "limit", 1)
                if record["sampled"] > record["limit"] or record["sampled"] > record["windows_seen"]:
                    raise TraceError("report sample count exceeds its limit or windows_seen")
                if record["unsupported"] > record["windows_seen"]:
                    raise TraceError("unsupported count exceeds windows_seen")
                request["report"] = record
                continue
            if kind not in ("segment", "window"):
                raise TraceError(f"unknown trace kind: {kind!r}")
            window, p0, tokens, attempt = _window_identity(record)
            key = window, attempt
            model = identity[:2]
            lifetime_key = model + (window,)
            prior_request = used_windows.get(lifetime_key)
            if prior_request is not None and prior_request != identity:
                raise TraceError("window ID reused in another request within a session")
            if prior_request is None:
                if window <= latest_window.get(model, 0):
                    raise TraceError("window ID reset or out of order within a session")
                latest_window[model] = window
                used_windows[lifetime_key] = identity
            item = request["attempts"].setdefault(key, {"segments": [], "end": None, "p0": p0, "T": tokens})
            if item["end"] is not None:
                raise TraceError("duplicate window end or segment after its end")
            if (item["p0"], item["T"]) != (p0, tokens):
                raise TraceError("window identity changed within an attempt")
            if kind == "segment":
                index = _integer(record, "index")
                if index != len(item["segments"]):
                    raise TraceError("missing, duplicate or out-of-order segment index")
                _integer(record, "layer", -1)
                row = _integer(record, "row", -1)
                if row >= tokens:
                    raise TraceError("segment row is outside the window")
                if not isinstance(record.get("stage"), str) or not record["stage"]:
                    raise TraceError("segment stage must be a nonempty string")
                _time(record, "ms")
                item["segments"].append(record)
            else:
                if type(record.get("complete")) is not bool or record.get("outcome") not in OUTCOMES:
                    raise TraceError("invalid window completion/outcome")
                count = _integer(record, "segments")
                begin = _integer(record, "layer_begin")
                end = _integer(record, "layer_end", 1)
                if end <= begin:
                    raise TraceError("invalid layer range")
                span = _time(record, "span_ms")
                total = _time(record, "segments_ms")
                if count != len(item["segments"]):
                    raise TraceError("window segment count does not match records")
                measured = sum(segment["ms"] for segment in item["segments"])
                if not _close(measured, total, count):
                    raise TraceError("segment sum disagrees with window segments_ms")
                if record["complete"]:
                    if record["outcome"] not in ("completed", "scratch_retry"):
                        raise TraceError("complete timing has a failure outcome")
                    if count == 0 or not _close(total, span, count):
                        raise TraceError("incomplete stream interval coverage")
                    layers = {segment["layer"] for segment in item["segments"] if segment["layer"] >= 0}
                    if layers != set(range(begin, end)):
                        raise TraceError("missing layer or layer outside the declared range")
                elif record["outcome"] in ("completed", "scratch_retry"):
                    raise TraceError("successful outcome marked incomplete")
                else:
                    issues.append({"session": identity[0], "device": identity[1], "request": identity[2],
                                   "window": window, "attempt": attempt, "reason": record["outcome"]})
                item["end"] = record
        except TraceError as error:
            raise TraceError(f"trace record {number}: {error}") from error
    if not requests:
        raise TraceError("no GLM window trace records")
    all_attempts = []
    summaries = []
    for identity, request in requests.items():
        report = request["report"]
        if report is None:
            raise TraceError(f"{identity}: missing final request report (truncated log)")
        attempts = request["attempts"]
        if any(item["end"] is None for item in attempts.values()):
            raise TraceError(f"{identity}: unterminated window attempt")
        windows = sorted({window for window, _ in attempts})
        if len(windows) != report["sampled"]:
            raise TraceError(f"{identity}: report sampled count disagrees with window records")
        for window in windows:
            chain = sorted((attempt, item) for (win, attempt), item in attempts.items() if win == window)
            if [attempt for attempt, _ in chain] != list(range(len(chain))):
                raise TraceError(f"{identity}, window {window}: missing retry attempt")
            if len({(item["p0"], item["T"]) for _, item in chain}) != 1:
                raise TraceError(f"{identity}, window {window}: identity changed during retry")
            for index, (_, item) in enumerate(chain):
                retry = item["end"]["outcome"] == "scratch_retry"
                if retry != (index < len(chain) - 1):
                    raise TraceError(f"{identity}, window {window}: missing or unexpected scratch retry")
        for field in ("unsupported", "errors"):
            if report[field]:
                issues.append({"session": identity[0], "device": identity[1], "request": identity[2],
                               "reason": field, "count": report[field]})
        values = list(attempts.values())
        all_attempts.extend(values)
        summaries.append({
            **{key: report[key] for key in ("session", "device", "request", "sampled", "every", "limit",
                                          "unsupported", "errors", "windows_seen")},
            "groups": _summarize(values),
        })
    if not all_attempts:
        issues.append({"reason": "no_sampled_windows"})
    return {
        "schema": 1, "complete": not issues, "issues": issues,
        "sampled_windows": sum(request["report"]["sampled"] for request in requests.values()),
        "sampled_attempts": len(all_attempts), "requests": summaries,
        "groups": _summarize(all_attempts),
        "stage_notes": {
            "moe_fetch_disk_rows": "All fetched weights of rows with a disk miss, including any RAM weights in those rows.",
            "moe_fetch_ram_rows": "Deferred fetches of rows without a disk miss; conditional kernels may do no work.",
            "moe_route_wait": "Stream interval around route readiness wait; not total host or helper service time.",
            "moe_helper_join": "Exposed helper wait and answer combination; not total helper execution time.",
        },
        "limitations": [
            "CUDA stream intervals include device stalls and host enqueue gaps; they are not pure kernel service times.",
            "Sampling is bounded per request; these are sampled attempts, not every window or emitted token.",
            "The request field identifies a model report period, not a globally unique API request.",
            "T2 and T3, devices, and completed versus discarded scratch attempts are grouped separately.",
            "Host boundary setup, drafting and commit outside the traced span are not included.",
            "Profiling overhead has not been subtracted; compare an otherwise identical trace-OFF run separately.",
            "No emitted-token count or cross-device clock alignment is supplied; no tokens/second or pipeline gain is inferred.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--out", type=Path, help="Write JSON here instead of standard output")
    args = parser.parse_args(argv)
    try:
        with args.log.open() as stream:
            result = analyze_records(read_records(stream, str(args.log)))
    except (OSError, TraceError) as error:
        print(f"window trace rejected: {error}", file=sys.stderr)
        return 2
    result["source"] = str(args.log)
    output = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.out:
        args.out.write_text(output)
    else:
        print(output, end="")
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    sys.exit(main())
