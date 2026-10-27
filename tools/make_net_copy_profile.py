"""Rank RAM-copy candidates from existing STRATA_GLM_ROUTE_LOG traces.

Each trace has equal weight. CPU mask demand has weight 1; PCIe fetch demand
has a smaller configurable weight. Helper-owned entries may occur in the input:
the engine filters them out using its actual startup ownership tables.
"""
import argparse
from collections import Counter
from pathlib import Path


def trace_scores(path, fetch_weight=0.25, layers=46, experts=288):
    scores = Counter()
    routes = 0
    for line_no, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        values = list(map(int, line.split("|", 1)[0].split()))
        if len(values) != 12:
            raise ValueError(f"{path}:{line_no}: expected layer, eight ids and three masks")
        layer, ids = values[0], values[1:9]
        fetch, miss, cpu = values[9:12]
        if not (0 <= layer < layers and len(set(ids)) == 8 and
                all(0 <= e < experts for e in ids) and all(0 <= m <= 255 for m in (fetch, miss, cpu)) and
                not (fetch & cpu or fetch & miss or miss & cpu)):
            raise ValueError(f"{path}:{line_no}: invalid route")
        routes += 1
        for i, expert in enumerate(ids):
            score = (cpu >> i & 1) + fetch_weight * (fetch >> i & 1)
            if score:
                scores[layer, expert] += score
    if not routes:
        raise ValueError(f"{path}: no routes")
    return {key: score / routes for key, score in scores.items()}, routes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fetch-weight", type=float, default=0.25)
    args = parser.parse_args()
    if not 0 <= args.fetch_weight <= 1:
        parser.error("fetch weight must be finite and between 0 and 1")
    scores = Counter()
    comments = ["# net copy profile: layer expert score; CPU=1, fetch=" + str(args.fetch_weight)]
    for path in args.traces:
        one, routes = trace_scores(path, args.fetch_weight)
        scores.update(one)
        comments.append(f"# {path.name}: {routes} routes (equal trace weight)")
    rows = [f"{l} {e} {scores[l, e] / len(args.traces):.12g}" for l, e in sorted(scores)]
    args.out.write_text("\n".join(comments + rows) + "\n")
    print(f"{args.out}: {len(rows)} ranked experts, {len(args.traces)} traces")


if __name__ == "__main__":
    main()
