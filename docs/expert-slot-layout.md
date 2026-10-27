# Experimental per-layer expert cache layout

`MAYA_GLM_SLOT_PROFILE=/absolute/path/layout.txt` overrides the primary GPU's
expert slot allocation before its pool is allocated. With the variable unset,
allocation is unchanged. This is an experimental placement control, not a
recommended production setting.

Each non-comment row is `layer total_slots`. Counts include spares and the
lendable tail. Unlisted layers retain their default allocation. The selected
layers' combined bytes must not exceed their previous combined bytes; moving
one slot between differently sized expert layers is not necessarily byte neutral.
Duplicate rows, non-MoE layers, the draft layer, invalid capacities and budget
overruns reject the entire file before allocation. Only a single discrete trunk
with a partial expert pool is supported; integrated/unified, complete-pool and
split-trunk configurations reject the override.

## Minimum capacity depends on the running configuration

The minimum is the number of lendable slots plus the number that must remain
resident. The latter covers the selected experts and spares, prefill's keep
setting, and vision's additional decode headroom when vision lending is enabled.
Vision frees the **whole** shared tail, even if prefill reserved more bytes than
the image encoder needs. Do not calculate the minimum from image bytes alone.

For the tested 262144-context / 32768-prefill / image-enabled home setup, the
minimum is **34 = 15 lendable + 19 kept**. It is not a universal constant.
The runtime reports the required range when rejecting an invalid layout and
prints the full applied layout and minimum on success. Treat that check as
authoritative; a file produced for another prefill/image configuration may fail.

## Fitting a candidate from existing traces

`tools/make_slot_profile.py` reads a baseline JSON map (`{"3":36,...}`), the
pack's `native_experts.txt`, and the existing textual route traces (layer, eight
expert IDs and three masks before `|`). Supply `--minimum` explicitly from the
runtime's requirements. For example:

```sh
python3 tools/make_slot_profile.py \
  --baseline baseline-slots.json --native-experts native_experts.txt \
  --train code.0 --train agent.0 --evaluate chat.0 \
  --minimum 34 --donor 3 34 --output layout.txt
```

By default it fits layers 19–44, excludes CPU-routed experts from the modeled GPU
cache, and weights each training trace equally. `--include-cpu-routes` models all
selected experts instead. `--donor LAYER SLOTS` may be repeated to free capacity
from other layers, but it must also respect the supplied minimum. The `.json`
sidecar records the byte budget, layout and approximate misses on each trace.

The fit is an LRU approximation, while the engine has admission, promotion,
offload and helper-ownership policies. In particular, changing the primary pool
also changes which experts fit on helper GPUs. The producer's reported miss
reduction covers only its target layers and does **not** model these ownership
changes or network calls. It is a candidate filter, not a throughput prediction.
Verify candidates with unchanged requests and long-prefill checks, inspecting
both token rate and helper/fetch behavior. Leave the production setting unset
unless real measurements justify a candidate.
