#!/usr/bin/env python3
"""Fit a partial layer-cache layout from traces, using an approximate LRU model.

CPU-routed experts do not update the modeled GPU cache. This is a placement
heuristic, not a simulation of the engine's LFU/offload/async-promotion policy.
Training traces are normalized separately so a longer run cannot dominate.
"""
import argparse
import collections
import json
from pathlib import Path


def histogram(path, layers, warmup, include_cpu=False):
    stacks = {l: [] for l in layers}
    hist = {l: collections.Counter() for l in layers}
    token = -1
    with path.open() as stream:
        for line in stream:
            words = list(map(int, line.split('|')[0].split()))
            if len(words) != 12:
                raise ValueError(f'{path}: expected layer, eight IDs and three masks')
            layer = words[0]
            if layer == 3:
                token += 1
            if layer not in stacks:
                continue
            ids, cpu = words[1:9], words[11]
            stack = stacks[layer]
            for index, expert in enumerate(ids):
                if not include_cpu and cpu >> index & 1:
                    continue
                try:
                    distance = stack.index(expert)
                    stack.pop(distance)
                except ValueError:
                    distance = 288
                stack.insert(0, expert)
                if token >= warmup:
                    hist[layer][distance] += 1
    measured = token + 1 - warmup
    if measured <= 0:
        raise ValueError(f'{path}: insufficient tokens after warmup')
    return hist, measured


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline', type=Path, required=True)
    ap.add_argument('--native-experts', type=Path, required=True)
    ap.add_argument('--train', type=Path, action='append', required=True)
    ap.add_argument('--evaluate', type=Path, action='append', default=[])
    ap.add_argument('--first-layer', type=int, default=19)
    ap.add_argument('--last-layer', type=int, default=44)
    ap.add_argument('--minimum', type=int, default=28)
    ap.add_argument('--maximum', type=int, default=80)
    ap.add_argument('--spares', type=int, default=3)
    ap.add_argument('--warmup', type=int, default=100)
    ap.add_argument('--include-cpu-routes', action='store_true')
    ap.add_argument('--donor', type=int, nargs=2, action='append', default=[], metavar=('LAYER', 'SLOTS'))
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    baseline = {int(k): v for k, v in json.loads(args.baseline.read_text()).items()}
    layers = list(range(args.first_layer, args.last_layer + 1))
    strides = {}
    for line in args.native_experts.read_text().splitlines():
        if line.strip() and not line.lstrip().startswith('#'):
            words = line.split()
            strides[int(words[0])] = (int(words[4]) + 255) // 256 * 256
    budget = sum(baseline[l] * strides[l] for l in layers)
    donors = {}
    for layer, count in args.donor:
        if layer in layers or layer in donors or layer not in baseline or not 0 < count <= baseline[layer]:
            raise ValueError('donors must be distinct non-target layers with a smaller positive capacity')
        donors[layer] = count
        budget += (baseline[layer] - count) * strides[layer]
    if args.minimum <= args.spares or args.maximum < args.minimum:
        raise ValueError('invalid capacity range')
    paths = list(dict.fromkeys(args.train + args.evaluate))
    curves = {}
    for path in paths:
        hist, tokens = histogram(path, layers, args.warmup, args.include_cpu_routes)
        curves[path] = {l: {c: sum(n for d, n in hist[l].items() if d >= c - args.spares) / tokens
                            for c in range(args.minimum, args.maximum + 1)} for l in layers}
    loss = {l: {c: sum(curves[p][l][c] for p in args.train) / len(args.train)
                for c in range(args.minimum, args.maximum + 1)} for l in layers}
    if not any(loss[l][args.minimum] > 0 for l in layers):
        raise ValueError('training traces contain no modeled misses in the selected layers')
    slots = {l: args.minimum for l in layers}
    used = sum(slots[l] * strides[l] for l in layers)
    if used > budget:
        raise ValueError('minimum capacities exceed the existing budget')
    while True:
        choices = []
        for layer in layers:
            current = slots[layer]
            for count in range(current + 1, args.maximum + 1):
                cost = (count - current) * strides[layer]
                if used + cost > budget:
                    break
                gain = loss[layer][current] - loss[layer][count]
                choices.append((gain / cost, -cost, -layer, layer, count, cost))
        if not choices:
            break
        _, _, _, layer, count, cost = max(choices)
        slots[layer] = count
        used += cost
    layout = dict(sorted((donors | slots).items()))
    donor_bytes = sum(c * strides[l] for l, c in donors.items())
    result = {'method': 'approximate LRU on target layers only; helper ownership changes not modeled',
              'include_cpu_routes': args.include_cpu_routes,
              'training': [str(p) for p in args.train], 'byte_budget': budget + donor_bytes,
              'used_bytes': used + donor_bytes, 'slots': layout, 'comparisons': {}}
    for path in paths:
        old = sum(curves[path][l][baseline[l]] for l in layers)
        new = sum(curves[path][l][slots[l]] for l in layers)
        result['comparisons'][str(path)] = {'baseline_misses_per_token': old,
                                           'candidate_misses_per_token': new,
                                           'reduction_fraction': 1 - new / old if old else 0}
    args.output.write_text('# layer total_slots (including spare and lendable slots)\n' +
                           ''.join(f'{l} {c}\n' for l, c in layout.items()))
    args.output.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
