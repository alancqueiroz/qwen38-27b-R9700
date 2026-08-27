"""Memory-lean drop-in for prepare/quant_lm_head.py, quant_embed.py and
quant_mtp.py, producing byte-schema-identical artifacts (same packed tensors,
scales, index keys, config groups, backups) while peaking at a fraction of the
RAM: the originals materialize the whole matrix twice in fp32 plus the full
dequantized copy for the error check (~18 GB peak on this checkpoint); this
one walks the rows in blocks, so the peak is the shard itself plus the packed
output (~5-6 GB). Same math, same round-trip error to four decimals.

Written for machines where another workload holds several GB of RAM (this
box: a second serving agent) -- the upstream scripts got OOM-killed there
twice even with 18 GB free.

Usage:
  python prepare/quant_lowmem.py <model_dir> lm_head|embed|mtp [--bits 8|4] [--keep-fc]

Run order matches the upstream trio: lm_head, then embed (it clones config
group_1), then mtp. Each step is idempotent-skipped if its index keys are
already present."""

import copy
import json
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from compressed_tensors.compressors.pack_quantized.base import pack_to_int32

GROUP = 128
ROWS = 8192   # rows per block: 8192 x 6144 x 4 B fp32 ~= 200 MB of temporary

d = sys.argv[1].rstrip("/") + "/"
step = sys.argv[2]
BITS = int(sys.argv[sys.argv.index("--bits") + 1]) if "--bits" in sys.argv else 8
QMAX = 2 ** (BITS - 1) - 1
KEEP_FC = "--keep-fc" in sys.argv
MTP_LINEARS = ([] if KEEP_FC else ["mtp.fc"]) + [
    "mtp.layers.0.mlp.down_proj",
    "mtp.layers.0.mlp.gate_proj",
    "mtp.layers.0.mlp.up_proj",
    "mtp.layers.0.self_attn.q_proj",
    "mtp.layers.0.self_attn.k_proj",
    "mtp.layers.0.self_attn.v_proj",
    "mtp.layers.0.self_attn.o_proj",
]

idx = json.load(open(d + "model.safetensors.index.json"))
wm = idx["weight_map"]
c = json.load(open(d + "config.json"))
qc = c["quantization_config"]

# blockwise equivalent of the originals': quantize, dequantize, Frobenius
# relative error, all accumulated per row-block so no full fp32 copy exists
def quant_blockwise(tensors, key, prefix, scale_dtype, shard, tag):
    out_f, in_f = tensors[key].shape
    n_groups = in_f // GROUP
    assert in_f % GROUP == 0, (key, tensors[key].shape)
    packed_blocks, scale_full = [], torch.empty(out_f, n_groups, dtype=torch.float32)
    num = den = 0.0
    src = tensors.pop(key)
    for r0 in range(0, out_f, ROWS):
        r1 = min(r0 + ROWS, out_f)
        wb = src[r0:r1].to(torch.float32)
        gb = wb.view(r1 - r0, n_groups, GROUP)
        sc = torch.clamp(gb.abs().amax(dim=-1, keepdim=True) / QMAX, min=1e-10)
        qb = torch.clamp(torch.round(gb / sc), -QMAX - 1, QMAX).to(torch.int8).view(r1 - r0, in_f)
        deb = (qb.view(r1 - r0, n_groups, GROUP).to(torch.float32) * sc).view(r1 - r0, in_f)
        num += ((deb - wb) ** 2).sum().item()
        den += (wb ** 2).sum().item()
        packed_blocks.append(pack_to_int32(qb, BITS, packed_dim=1).contiguous())
        scale_full[r0:r1] = sc.squeeze(-1)
        del wb, gb, sc, qb, deb
    err = (num / den) ** 0.5
    print(f"  {tag}: {out_f}x{in_f} round-trip relative error {err:.4f}")
    assert err < 0.01, f"{key}: quantization error too high, aborting"
    del src
    tensors[prefix + ".weight_packed"] = torch.cat(packed_blocks, dim=0).contiguous()
    del packed_blocks
    tensors[prefix + ".weight_scale"] = scale_full.to(scale_dtype).contiguous()
    tensors[prefix + ".weight_shape"] = torch.tensor([out_f, in_f], dtype=torch.int64)

def load_shard(shard):
    tensors = {}
    with safe_open(d + shard, framework="pt") as f:
        meta = f.metadata()
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    return tensors, meta or {"format": "pt"}

if step == "lm_head":
    key, shard = "lm_head.weight", wm["lm_head.weight"]
    if key not in wm:
        print(f"[lowmem] {key} already quantized; skipping")
        sys.exit(0)
    print(f"[lowmem] {key} lives in {shard}")
    tensors, meta = load_shard(shard)
    quant_blockwise(tensors, key, "lm_head", torch.float16, shard, key)
    shutil.copy(d + shard, d + shard + ".bak")
    save_file(tensors, d + shard, metadata=meta)
    shutil.copy(d + "model.safetensors.index.json", d + "model.safetensors.index.json.bak-quant")
    del wm[key]
    for s in ("weight_packed", "weight_scale", "weight_shape"):
        wm[f"lm_head.{s}"] = shard
    json.dump(idx, open(d + "model.safetensors.index.json", "w"), indent=2)
    shutil.copy(d + "config.json", d + "config.json.bak-quant")
    qc["ignore"] = [i for i in qc["ignore"] if i != "lm_head"]
    # The MTP draft head is stored in bf16 but missing from the ignore list, which
    # breaks loading when speculative decoding is enabled (single-user mode).
    for m in MTP_LINEARS:
        if m not in qc["ignore"]:
            qc["ignore"].append(m)
    g1 = copy.deepcopy(qc["config_groups"]["group_0"])
    g1["targets"] = ["re:.*lm_head$"]
    g1["weights"]["num_bits"] = BITS
    qc["config_groups"]["group_1"] = g1
    json.dump(c, open(d + "config.json", "w"), indent=2)
elif step == "embed":
    key = next(k for k in wm if k.endswith("embed_tokens.weight"))
    if key not in wm:
        print(f"[lowmem] {key} already quantized; skipping")
        sys.exit(0)
    shard = wm[key]
    print(f"[lowmem] {key} lives in {shard}")
    assert "group_1" in qc["config_groups"], "run the lm_head step first (embed clones its group_1)"
    tensors, meta = load_shard(shard)
    quant_blockwise(tensors, key, key[: -len(".weight")], torch.bfloat16, shard, key)
    shutil.copy(d + shard, d + shard + ".bak_embed")
    save_file(tensors, d + shard, metadata=meta)
    del wm[key]
    p = key[: -len(".weight")]
    for s in ("weight_packed", "weight_scale", "weight_shape"):
        wm[f"{p}.{s}"] = shard
    json.dump(idx, open(d + "model.safetensors.index.json", "w"), indent=2)
    g2 = copy.deepcopy(qc["config_groups"]["group_1"])
    g2["targets"] = ["re:.*embed_tokens$"]
    g2["weights"]["num_bits"] = BITS
    qc["config_groups"]["group_2"] = g2
    json.dump(c, open(d + "config.json", "w"), indent=2)
elif step == "mtp":
    todo = [m for m in MTP_LINEARS if m + ".weight" in wm]
    if not todo:
        print("[lowmem] mtp linears already quantized; skipping")
        sys.exit(0)
    shards = {wm[m + ".weight"] for m in todo}
    assert len(shards) == 1, f"mtp weights span several shards: {shards}"
    shard = shards.pop()
    print(f"[lowmem] mtp linears live in {shard}, quantizing to int{BITS} g{GROUP}")
    tensors, meta = load_shard(shard)
    for m in todo:
        quant_blockwise(tensors, m + ".weight", m, torch.float16, shard, m)
        del wm[m + ".weight"]
        for s in ("weight_packed", "weight_scale", "weight_shape"):
            wm[f"{m}.{s}"] = shard
    shutil.copy(d + shard, d + shard + ".bak-mtp")
    save_file(tensors, d + shard, metadata=meta)
    shutil.copy(d + "model.safetensors.index.json", d + "model.safetensors.index.json.bak-mtp")
    json.dump(idx, open(d + "model.safetensors.index.json", "w"), indent=2)
    shutil.copy(d + "config.json", d + "config.json.bak-mtp")
    qc["ignore"] = [i for i in qc["ignore"] if i not in MTP_LINEARS]
    g = copy.deepcopy(qc["config_groups"]["group_0"])
    g["targets"] = ["re:^mtp\\.layers\\..*"] if KEEP_FC else ["re:^mtp\\..*"]
    g["weights"]["num_bits"] = BITS
    qc["config_groups"]["group_3"] = g
    json.dump(c, open(d + "config.json", "w"), indent=2)
else:
    sys.exit(f"unknown step {step!r} (lm_head | embed | mtp)")

print("done")