# Qwen3.8-27B on the AMD Radeon AI PRO R9700 (ROCm)

Port of this repository's stack to the AMD Radeon AI PRO R9700 (RDNA4,
`gfx1201`, 32 GB VRAM), keeping the same environment-variable contract, the
same requantized model and the same patches as the CUDA/NVIDIA path.

## What changes, in one sentence

Everything whose gain lives in Python or Triton ported unchanged (the
requantizers, MTP, DFlash2, lookup drafting, prefix caching, sampler, draft
vocab); everything that is an NVIDIA-only kernel was replaced by its ROCm
equivalent (Marlin -> vLLM's fallback GEMM path; FlashInfer/fp8 -> bf16 or
int8-per-token-head via Triton), and the memory budgets were scaled x4/3 for
the card's extra 8 GiB.

## Portability map, optimization by optimization

| technique | status on the R9700 | how it runs here |
|---|---|---|
| lm_head + embed_tokens int8 (`prepare/quant_lm_head.py`, `quant_embed.py`) | identical | CPU-only, same checkpoint artifacts |
| quantized embedding patch (`qwen3_5-embed-quant.patch`) | identical | pure Python |
| fp16 recurrent state (`--mamba-ssm-cache-dtype float16`) | identical | same flag; what unlocks concurrency |
| W4A16 body (AutoRound checkpoint) | works, different kernel | no Marlin on ROCm: GEMMs fall back to vLLM's dequant+GEMM path. Batch-1 decode is bandwidth-bound so most of the gain survives; large batch loses more -- measure |
| int8 activations W4A8 (`INT8_ACT`) | unavailable | Marlin does not exist on ROCm; the knob is accepted but **inert** in the AMD scripts (a notice is printed) |
| MTP + own draft vocab + quantized drafter + fast variant | identical | Python + Triton |
| sort-free top-k/top-p sampler (`sampler-small-topk-fast-softmax.patch`) | identical | torch/Triton |
| split-KV attention for verify (`spec-decode-attn.patch`) | conditional | kernel is Triton (portable) but the hook lives in the FLASH_ATTN backend class; only fires if that backend is selected -- check the `Using ... attention backend` boot-log line |
| DFlash2 backport + lookup drafting (`DFLASH_TOKENS=15`) | identical | Python logic; selector falls from flashinfer.topk to torch.topk (~half selector speed, sub-ms per step at batch 1). Functionally broken on RDNA4 today -- see results section |
| hybrid prefix caching (`PREFIX_CACHE=1`) | identical | Python allocator/scheduler |
| fp8 KV cache via FlashInfer (`CTX=long`/batch `KV=fp8`) | replaced | no FlashInfer on RDNA4: use `int8_per_token_head` on the Triton backend (same half-width KV idea) or bf16 |
| KVarN 4/2-bit (`CTX=huge`, 240k) | experimental | Triton kernels should run on HIP unchanged; **no numerical validation on RDNA4 yet** -- requires `EXPERIMENTAL=1` |
| CUDA graphs (`compilation-config`, PIECEWISE/FULL) | equivalent | captured as HIP graphs; same modes, same capture sizes |
| expandable_segments | n/a | a CUDA VMM feature; doesn't exist on HIP -- nothing exports it |

## Docker environment

The image is built from scratch for ROCm (`docker/Dockerfile.rocm`):

- base rocm/dev-ubuntu-24.04:7.2.x-complete (full HIP toolchain);
- torch==2.13.0+rocm7.2 straight from the PyTorch index -- the SAME pin
  vllm==0.27.1 declares, with no nvidia-* wheel entering the tree;
- vllm==0.27.1 built from sdist with VLLM_TARGET_DEVICE=rocm (no official
  ROCm wheel exists for this version; AMD's rocm/vllm images carry older
  versions whose trees don't match these patches);
- all `patches/*.patch` applied + KVarN installed + verify.sh --install
  running at build time (100% CPU checks -- nothing touches the GPU or
  allocates VRAM).

```bash
# build (once; the HIP compile takes tens of minutes)
docker compose -f docker-compose.amd.yml build

# prepare model (~19.5 GB download + requantization; CPU/disk only)
docker compose -f docker-compose.amd.yml run --rm prepare

# serve -- same profiles as the original compose
docker compose -f docker-compose.amd.yml --profile single up -d   # latency (MTP/dflash2)
docker compose -f docker-compose.amd.yml --profile batch  up -d   # throughput
```

GPU access: the container gets /dev/kfd + /dev/dri and joins the host's
video/render groups **by GID** (container uids are not host uids). On this
machine: video=44, render=992. If yours differ, set VIDEO_GID / RENDER_GID in
.env. Knobs go in .env under the SAME names as the NVIDIA stack (SPEC, CTX,
DFLASH_TOKENS, PREFIX_CACHE, KV_MEM, MAX_SEQS, EXTRA_ARGS, ...).

Two build-time gates worth knowing about. First, vLLM's ROCm platform plugin
detects the device by importing amdsmi; without it current_platform stays
UnspecifiedPlatform and every vllm serve dies at "Failed to infer device
type" -- and verify.sh's GPU probe passes either way (torch sees HIP
directly), which masks the problem until the first real boot. So the image
installs amdsmi behind its own import-gated layer. Second, xgrammar pulls a
dist literally named triton whose PyPI wheel clobbers torch's triton_rocm
package directory; the Dockerfile force-reinstalls triton-rocm as the last
dependency step and verify.sh guards against the corruption.

## Memory budgets (24 GB -> 32 GB)

The 24 GB card's pool pins (KV_MEM) were fractions calibrated to it; the AMD
scripts recompute the same values x4/3 (rounding up), keeping the budget shape
and spending the extra 8 GiB on pool:

| mode (single-user) | 3090 (24 GB) | R9700 (32 GB) | resulting pool (estimate) |
|---|---|---|---|
| CTX=fast MTP/dflash2 k<=7 | 5.20 GiB -> 69,758 tok | KV_MEM=7444609979 (~6.93 GiB) | ~95k tokens @78 KB/tok (dflash2); 155k realized for MTP geometry |
| CTX=fast dflash2 k>7 | 5.20 GiB | same | more residents before preemption |
| CTX=huge KVarN | 4.90 GiB -> 268,169 tok | KV_MEM=7015113251 (~6.53 GiB) | ~340k effective tokens |

MAX_LEN defaults were left IDENTICAL to the 3090's on purpose: comparable
geometry across benches and guaranteed boot. The surplus shows up as extra
residents and transient headroom. For longer contexts raise DFLASH_MAX_LEN /
MAX_LEN manually.

Batch mode keeps going through GPU_UTIL (no pin): with 32 GiB the pool grows
by itself.

## Real-hardware results (2026-08-26/27 boots, healthy server)

First healthy boot of the port: model loaded in 18 s (14.06 GiB of weights),
torch.compile 57 s + 13 s, FULL+PIECEWISE HIP graphs captured in 2 s
(0.61 GiB), a **155,795-token** pool (vs ~73.8k on the 3090 -- the surplus
VRAM went to the pool, as designed). Greedy smoke answers 'Kobenhavn'.

`bash bench/run_benchmarks.sh single`, 8 real prompts, 1024 output tokens:

| cohort | R9700 decode | tok/step | 3090 decode (reference) |
|---|---|---|---|
| C1 T=default | **65.7** | 2.84 | 111-121 |
| C2 | **84.1** | 2.80 | 191.8 |
| C4 | **92.0-93.7** | 2.79-2.83 | 268.5 |
| C8 | **254.6-257.0** | 2.78-2.82 | 407.3 |
| C1 greedy | **68.7** | 2.90 | 115-124 |

Honest reading: **tokens per step matches the 3090** (2.8-3.0) -- speculation
acceptance ported intact. The lower absolute rate is memory bandwidth plus
ROCm kernel maturity. The later baseline closed the account: **27.1 tok/s
without speculation (TPOT 36.88 ms; three repeats within 0.01 ms)** -- MTP
gives **x2.4 over it** here vs x2.5 in the 3090 reference; the relative
multiplier is what this port set out to preserve, and it did.

### Long context on a real corpus (labd, 22,774 tokens, --max-tokens 800)

The first labd round against ROCm produced immediate EOS on copy and
summary/qa -- **a harness artifact, not the engine**: the frozen corpus is
built from ~/qwen-serving/*.md, which does not exist inside the container,
and the script ends up creating 72 thousand blank characters ("reproduce the
first 60 lines" of an empty document yields any short text). Building the
corpus from this repository's own READMEs/docs (same repetition recipe up to
>200k chars, stored at /cache/bench/labd_corpus.txt on the persistent volume),
the table reads:

| task | no speculation | MTP | DFlash2 |
|---|---|---|---|
| copy | 512 @1.00 | **800 @4.64** (57.4 tok/s) | 800 @1.00 |
| code | 512 @1.00 | **800 @3.94** | 800 @1.00 |
| edit | 512 @1.00 | **800 @4.33** | 800 @1.03 |
| quote | 512 @1.00 | **800 @4.12** | 800 @1.71 |
| summary | 512 @1.00 | **800 @2.89** | 800 @1.23 |
| qa | 512 @1.00 | **800 @2.54** | 800 @1.11 |
| TOTAL decode | 28.5 tok/s | **44.4 tok/s** | 10.9 tok/s |

(no-speculation rows were truncated at the default 512 cap; the other columns
used an 800 cap so the full copy fits like the 3090 reference. The first
task's ttft includes ~16 s of long-prefill JIT.)

Conclusions: **MTP is healthy and correct on RDNA4** -- reproduces long text
verbatim to the cap, with above-average acceptance exactly on the copy task
(repetition = the candidate is almost always right), 3.57 tok/step overall.
**DFlash2 is functionally broken on this card**: near-zero acceptance
(copy/code at flat 1.00). Not because of corrupt generation -- the
no-speculation path produces identical, correct text -- but because this
build selects the **ROCM_ATTN** attention backend (auto-overridden at boot),
while the spec-decode-attn/lookup hooks that verify drafts are patched into
the FLASH_ATTN class used on NVIDIA. Without that path every draft is
rejected: DFlash2 lands 2.6x slower than MTP and worse than not speculating.
The launcher now requires EXPERIMENTAL=1 for SPEC=dflash2 with this
explanation attached.

#### Why porting the hooks will NOT fix DFlash2

Elimination protocol run on 2026-08-27 (each with the copy/quote probe at 20k
context):

| hypothesis | experiment | result |
|---|---|---|
| missing verify hooks | metrics counters + patch reading | inert, but harmless |
| wrong backend | forced TRITON_ATTN boot | crashes at init (assert in restriding) -- this pool's geometry isn't supported there |
| adaptive draft length | VLLM_DFLASH2_LOOKUP_ADAPTIVE=0 | unchanged, 1.00-1.09 tok/step |
| drafts' Triton sampler | VLLM_DFLASH2_DRAFT_TOPK_TOPP=0 (torch) | unchanged, 1.03 |
| output corrupted only under rejection | metrics/text correlation | **target-model output turns to garbage** with drafts>0 and zero acceptances |

Conclusion: with DFlash2 active on this build, proposals happen (counters
advance) but nothing is accepted AND the target model's own logits get
occasionally corrupted -- the invasive drafting/lookup path (the speculator's
Triton kernels + state injection) miscompiles for gfx1201. Porting the
spec-decode-attn hooks to ROCM_ATTN doesn't touch that path, so it would be
wasted work on this base. Candidate fixes, by cost:
1. debug _selector_walk_kernel / _cache_draft_logits_kernel and _apply_lookup
   under HIP (strong suspicion: RDNA4-specific mask/offset behavior, or a
   write race on the drafter's KV);
2. rebase onto newer vLLM while re-threading all patches through a moving
   target (expensive; only worth it if 1 fails);
3. accept MTP as this card's speculative mode (current state: measured,
   correct, x2.4 over baseline).

#### VLLM_DFLASH2_CHAIN=1 (upstream PR #38) -- unavailable in the pinned snapshot

A fork sync brought PR #38 (n-gram-chain drafting with no drafter), but the
flag does not exist in the pinned vLLM 0.27.1 (Unknown vLLM environment
variable detected at boot) -- and with the usual lookup drafting the behavior
repeats what was already measured: copy/code at 1.00 tok/step, 1.05 total.
Testing chains here would require rebasing the patch stack onto newer vLLM
first.

## Power-delivery caution: two hard shutdowns mapped to GPU transients

During sustained GPU load this machine died twice, instantly:

| event | local time | workload phase |
|---|---|---|
| 1 | Aug 25, 23:52 | sustained decode benchmark |
| 2 | Aug 26, 18:52 | server boot: torch.compile + HIP graph capture (after an image rebuild) |

Both share the same signature: the previous boot's journal simply ends with no
shutdown record, panic or amdgpu error (`last -x` still shows the prior boot as
active). No command in this stack can power off a machine -- instant death under
GPU peaks with a clean kernel trace is a PSU protection trip on RDNA4 transients.
Aggravating factor found: the card ships with PPT at **300 W (the hwmon maximum)**.

Mitigation (one-time root action). Note rocm-smi --setpoweroverdrive silently
failed on this card (clean exit, cap unchanged -- the
energy_count_secondary_die_check quirk), so sysfs is the reliable route. Card
numbering CHANGES between boots (the R9700 has been both card1 and card0; the
hwmon index moves with it), so select by label instead of a fixed path:

```bash
# immediate (applies wherever the label is PPT and the attribute exists):
for f in /sys/class/drm/card*/device/hwmon/hwmon*/power1_cap; do
  [ "$(cat ${f%/*}/power1_label 2>/dev/null)" = "PPT" ] && \
  [ "$(cat $f 2>/dev/null)" != "" ] && echo 260000000 | sudo tee $f
done

# persistent:
sudo tee /etc/systemd/system/rocm-ppt-cap.service >/dev/null <<'UNIT_EOF'
[Unit]
Description=Cap RDNA4 GPU PPT at 260W (PSU transient protection)
After=multi-user.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/bin/bash -c 'for f in /sys/class/drm/card*/device/hwmon/hwmon*/power1_cap; do [ "$$(cat $${f%/*}/power1_label 2>/dev/null)" = "PPT" ] && [ -s "$$f" ] && echo 260000000 > "$$f"; done; exit 0'

[Install]
WantedBy=multi-user.target
UNIT_EOF
sudo systemctl daemon-reload && sudo systemctl enable --now rocm-ppt-cap.service
```

systemd NOTE: inside unit files, $ must be escaped as $$ (the manager consumes
single ones before bash runs). With Type=oneshot, is-active reports 'inactive'
once the service FINISHES SUCCESSFULLY -- not an error; RemainAfterExit=yes
above keeps 'active' visible while the cap holds.

Verify:

```bash
for f in /sys/class/drm/card*/device/hwmon/hwmon*; do
  [ "$(cat $f/power1_label 2>/dev/null)" = "PPT" ] && [ -s $f/power1_cap ] && \
    echo "$f -> $(cat $f/power1_cap)"
done   # should print ... -> 260000000
```

Practical note: boots whose compile hash is already cached in /cache (restarts
with an unchanged image) skip the heavy phase entirely; the risk window is the
cold compile after an image rebuild, until the cap is applied.

### Measured cost of the caps (same suite, same day)

The full single-user suite ran under the 240 W cap with zero incidents -- the
mitigation holds under sustained load. Cost is 6-14% of throughput, concentrated
where concurrency is higher, and tok/step is untouched: the cap trims peak
clocks, it doesn't change numerics or acceptance.

| cohort | uncapped | capped 240 W | delta |
|---|---|---|---|
| C1 T=default | 65.7 | 59.1 | -10% |
| C4 T=default | 92.0 | 86.6 | -6% |
| C8 T=default | 257.0 | 220.9 | -14% |
| C1 greedy | 68.7 | 61.6 | -10% |
| C8 greedy | 257.0 | 250.9 | -2% |

Cap saturation measurement (24 prompts x 1024 tokens, concurrency 8, aggregate
decode 186 tok/s over 127 s, power sampled at 5 Hz): **median 239 W** (pinned at
the cap), p99 242 W, mean-window peak 267 W -- clean. So 240 W genuinely binds at
high cohorts, and measurable headroom exists between it and the only value that
ever tripped (factory 300 W).

Clean A/B at seat count one (`vllm bench serve`, same harness, interleaved
same-day):

| cap | decode (C/meanTPOT) | mean TPOT |
|---|---|---|
| 260 W | **60.6 tok/s** | 16.50 ms |
| 240 W, run 1 | 48.0 tok/s | 20.55 ms |
| 240 W, run 2 | 47.6 tok/s | 20.61 ms |

240 W costs **~21% single-stream** (not the ~10% the cross-harness estimate
suggested) -- one-seat decode is core-clock bound and the cap cuts exactly the
boost. Multi-seat stays tied (bandwidth-bound). Practical recommendation: **260 W
for the daily unit**, 240 W reserved for unattended long runs. Remaining item:
validate the next cold-compile boot under 260 W (kill phase number 2) before
calling it immune. Microsecond transients stay invisible to software -- every
step above remains an experiment.

## CTX=long (128K) serving profile — validated 2026-08-28

`CTX=long SPEC=dflash2 DFLASH_TOKENS=4 PREFIX_CACHE=1 EXPERIMENTAL=1` on the
`qwen38-r9700:vllm-port0280` image:

- max_model_len 131072; KV `int8_per_token_head` on `TRITON_ATTN`;
  KV pool 7.44 GiB = 192,069 tokens (1.47x concurrency at full 131k).
- Short-prompt decode: ~43.8 tok/s sequential (vs 52.1 on the 64k bf16
  profile — the int8-KV/Triton tax).
- 82k-token prompt: works, no corruption; first prefill ~531 s
  (chunked prefill 2048, ~155 tok/s); identical re-request with prefix
  caching drops to ~59 s; decode at that depth stays in the ~30-40 tok/s
  class. Filling 128k cold takes ~14 min — batch long prefills accordingly.
- DSH integration (`~/.dsh/settings.yaml`): provider `qwen-amd` ->
  `http://127.0.0.1:18020/v1`, model `qwen3.8-27b`, contextWindow 131072,
  maxTokens 24576 (post-compaction headroom rule), text-only.
  compaction-basic then auto-compacts at floor(0.8*131072)=104,857 tokens
  and recovers on provider-confirmed overflow.

## Prefix cache reporting (2026-08-29)

The DSH harness showed `Cache hit 0%` for local sessions while the engine's
own metrics proved caching worked (`vllm:prefix_cache_hits_total` advancing,
A/B identical-prefix requests 2.5x faster on the second call). Root cause:
this build serves `usage.prompt_tokens_details.cached_tokens` only behind
`--enable-prompt-tokens-details` (default OFF in
`entrypoints/openai/cli_args.py`), and that field is exactly what clients
(pi-ai -> DSH `cacheReadTokens`) use for the hit-rate stat.

Fix: the launcher now passes `--enable-prompt-tokens-details` unconditionally.
Also parameterized `--max-num-batched-tokens` (`MAX_BATCHED_TOKENS`, default
4096) - note vLLM clamps scheduled tokens to 2048 while speculative decoding
is active, so the knob only pays off in a no-spec profile.

Reminder: the prefix cache is in-memory; every container recreate wipes it and
the next turn re-prefills the whole session (~100k tokens at ~155 tok/s
prefill ≈ 10-11 min of TTFT). vramd-style GPU handoffs accept that cost.
