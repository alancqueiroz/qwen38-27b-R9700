# Upstream port experiment: vLLM tree with native DFlash2 + ROCm draft attention

Branch: `amd-r9700-vllm-53628` (isolated; `main` and `amd-r9700` stay on the
pinned 0.27.1 sdist line).

## Purpose

Upstream now ships the pieces whose absence forced us to pin 0.27.1:

- native DFlash2 (`vllm/v1/spec_decode/dflash.py`) instead of our backport;
- a registrable `ROCM_ATTN` backend (`vllm/v1/attention/backends/rocm_attn.py`);
- sliding-window-aware prefix attention (`prefix_prefill.py` carries
  `SLIDING_WINDOW` handling).

On top of those, open PR
[vllm-project/vllm#53628](https://github.com/vllm-project/vllm/pull/53628)
("[BugFix][ROCM] DFlash2 fix sliding-window prefix attention NaNs") fixes the
batch-dependent acceptance collapse of issue #53323: with a sliding-window
drafter, evicted context blocks map to stale physical KV blocks, and masking
after attention math lets `0 * NaN` propagate. The fix masks block-table and
K/V loads BEFORE the attention computation.

Our own RDNA4 failure was diagnosed as a gfx1201 Triton miscompilation in the
drafter/lookup path (zero acceptance plus target-logit corruption even at
concurrency 1). Whether the newer tree also fixes THAT is exactly what this
experiment tests; nothing here assumes it does.

## Snapshot chosen

- Tree: vllm-project/vllm @ `6a9c69fa8513` (merge-base of open PR #53628).
- While #53628 is unmerged, its diff ships here as
  `patches/upstream/53628-prefix-prefill-swa-nan.diff` and is applied by the
  git build variant through `VLLM_UPSTREAM_PATCH`.
- Dependency pins: the tree still requires `torch == 2.13.0` (`pyproject.toml`),
  so our `torch==2.13.0+rocm7.2`, `triton-rocm==3.7.1` and `amdsmi==7.0.2`
  pins are unchanged. `requirements/common.txt` + `requirements/rocm.txt`
  are fetched from a selectable ref (`VLLM_REQ_REF`).

## Patch drift audit (re-runnable)

    prepare/port_audit.sh /path/to/vllm-checkout [patches/upstream/NAME.diff]

Dry-runs every patch in `patches/` against a copy of the given checkout
(package root), optionally pre-applying an upstream diff first. Exit status =
number of failing patches.

| patch | result | strategy |
|---|---|---|
| dflash2-backport | FAIL | DROP - native DFlash2 replaces it |
| dflash2-lookup-drafting | FAIL | DROP - native path; re-measure acceptance |
| hybrid-kv-groups-v2-cudagraph | FAIL->APPLIED | DONE - slimmed: dropped obsolete VLLM_V2_CUDAGRAPH_MEM_MIB hunk, upstream profiles graphs natively now |
| spec-decode-int8-kv | FAIL->APPLIED | no changes needed - it targets our own spec_decode_attn.py created by an earlier patch; only stacked application is meaningful |
| vision-tower-cpu-offload | FAIL->APPLIED | DONE - rebuilt against the refactored Qwen3-ViT init (merger/deepstack/blocks reordered); intent preserved |
| xgrammar-spec-terminated | FAIL->SKIPPED | DONE - upstream ships exactly this semantics now; moved to port-skip.lst |
| hybrid-sw-block-promote | OK | keep |
| marlin-int8-layer-select | OK | keep |
| marlin-int8-negative-scales | OK | keep |
| qwen3_5-embed-quant | OK | keep |
| qwen3_5-mtp-draft-vocab | OK | keep |
| sampler-small-topk-fast-softmax | OK | keep |
| spec-decode-attn | OK | keep (our hooks survive intact) |
| speed-knobs-envs | OK | keep |
| vllm-pr50021-gdn-spec-bounds | OK | keep |

`patches/port-skip.lst` lists the two dropped DFlash2 patches; the image build
skips exactly that list ONLY in the git variant (`VLLM_SOURCE=git`), so the
default sdist build remains byte-reproducible.

## Building the experiment (opt-in; defaults unchanged)

    docker build -f docker/Dockerfile.rocm \
      --build-arg VLLM_SOURCE=git \
      --build-arg VLLM_GIT_REF=6a9c69fa851389dcf1ee5d3a2363e27af665d26d \
      --build-arg VLLM_REQ_REF=v0.27.1 \
      --build-arg VLLM_UPSTREAM_PATCH=53628-prefix-prefill-swa-nan.diff \
      --build-arg VLLM_EXPECT_VERSION=<version.py value at the ref> \
      -t qwen38-r9700:vllm-port .

`VLLM_REQ_REF` must point at a ref whose `requirements/common.txt` and
`requirements/rocm.txt` match the tree well enough for pip; replace the
example value after checking the target tag.

## Open checks BEFORE first boot on the GPU

1. `VLLM_EXPECT_VERSION` must equal `vllm/version.py` at the ref, or the
   build-time assert kills the image correctly early.
2. Re-measure KV budgets: cache block sizes changed upstream
   (`hybrid cache attention block size: 832` appears in #53323), so BOTH
   `KV_MEM` budgets in docs/amd.md are stale here.
3. `kvarn/install.sh` compatibility with the new internals - unverified.
4. `verify.sh` gates may assert 0.27.1-era behavior; expect edits.
5. Keep the PPT cap service active before ANY GPU load; RDNA4 PSU trips do
   not care which vLLM version is running.
6. gfx1201 Triton miscompile status is UNKNOWN in this tree: exercise DFlash2
   with the same acceptance/corruption probes used for the original verdict,
   behind gating, single-request first, THEN batched (the NaN bug of #53323
   only shows batched).

## Safety

This branch changes nothing on the served path. The default image build keeps
the exact pinned sdist line; `main`/`amd-r9700` remain the PR-facing state.

## Stacked-audit status (this branch, HEAD)

`prepare/port_audit.sh` defaults to BUILD-ORDER simulation (skips honored,
patches applied cumulatively): **12 applied / 0 failed / 3 skipped-dropped**.
The three skips are permanent drops replaced by native upstream code:
dflash2-backport, dflash2-lookup-drafting, xgrammar-spec-terminated.

## Open work items after the first green build

1. KVarN port rebasing (the big one). Drift vs `6a9c69fa8513` measured:
   - `kvarn/kvarn-0.27.1.patch`: 3 problem hunks (config/cache.py,
     layers/attention/attention.py, platforms/{cuda,interface}.py,
     utils/torch_utils.py);
   - `kvarn/kvarn-v2-runner.patch`: 8 problem hunks (targets include
     qwen3_dflash.py -- now NATIVE upstream).
   The install script's port(kvarn-v2) marker verification makes any partial
   application fail the image build loudly; until these are recut, expect the
   git-variant build to fail at Layer 4 with an actionable message.
2. First-boot GPU protocol (only when the card is free, PPT cap active):
   DFlash2 probes single-request before batched per the #53323 lesson.

## Image build status: GREEN

qwen38-r9700:vllm-port builds end to end (git variant): vLLM 0.28.0.dev0 @ 6a9c69fa8513 + the #53628 diff, all 12 patches, and the KVarN port complete (12/12 port(kvarn-v2) markers). In-image probe: SWA-aware prefix kernel (12 SLIDING_WINDOW refs, PR #53628 markers present), ROCM_ATTN and KVARN backends registered, kvarn_k4v2_g128 CacheDType registered.

Remaining before serving: on-GPU DFlash2 acceptance probes (single request first, then batched - the #53323 collapse is batch-only) and KV budget re-measurement for the new hybrid cache layout.

## On-GPU DFlash2 probes (R9700, this image) - 2026-08-27

Spec: SPEC=dflash2 DFLASH_TOKENS=7 CTX=fast PREFIX_CACHE=1, temperature 0,
ignore_eos, 4x160-token requests per phase. PPT cap 260 W enforced.

| phase | acceptance | accept len | throughput |
|---|---|---|---|
| single stream | 24.8% | 1.73 | ~33 tok/s |
| batched (4) | 31.4% | 2.20 | measured jointly |

- Output text coherent; NO target-logit corruption (0.27.1 verdict falsified).
- NO batch-dependent collapse (#53323 pattern absent here).
- Per-position acceptance decays smoothly 330/217/135/83/57/39/32.
- Container stable: 0 restarts, no OOM.

Reading: DFlash2 WORKS on RDNA4 on this tree. It is not the daily-driver
speed king (daily MTP ~= 65 tok/s single stream) - with accept_len ~2 the
7-token draft budget overshoots; DFLASH_TOKENS=3..4 is the obvious tuning
follow-up, and KVarN CTX=huge is the reason this branch exists.

## DFLASH_TOKENS tuning sweep (single stream, 4x160 tok, temp 0) - 2026-08-27

| T | acceptance | accept len | throughput |
|---|---|---|---|
| 3 | 44.6% | 1.34 | 47.2 tok/s |
| 4 | 40.1% | 1.60 | 52.1 tok/s |
| 5 | 33.9% | 1.70 | 31.3 tok/s (cold) / 31.3 warm |
| 7 | 24.8% | 1.73 | 33.0 tok/s |

MTP on the SAME image for reference: 36.9% acceptance, accept len 1.47,
42.4 tok/s single stream.

Winner: DFLASH_TOKENS=4 -> 52.1 tok/s, +23% over MTP on the identical
image (and MTP itself regressed 65 -> 42 vs the 0.27.1 image, separate
issue). Batched(4) at T=4: 39.8-48.2% acceptance, 69.5 tok/s aggregate.
Deployed as the serving config: .env SPEC=dflash2 DFLASH_TOKENS=4 +
docker-compose.port.yml.

## Branch amd-r9700-vllm-v0280: retarget to the v0.28.0 release

Motivation (checked 2026-08-27):
- v0.28.0 (2cf0a6915c) contains the DFlash2 load fix (#53435) our
  dflash2-decoder-layer-cls.patch backported -> that patch is now redundant
  and moved to port-skip.lst.
- PR #53628 is STILL open and its head diff is unchanged -> keep vendoring
  patches/upstream/53628-prefix-prefill-swa-nan.diff (pre-applies cleanly on
  v0.28.0).
- syv-ai main gained 4 functional patches. Audit vs v0.28.0:
  * int4-kv-per-token-head: applies clean (Triton, opt-in KV dtype) -> INCLUDED
    via patches/v0280/ (git variant only).
  * offload-dflash-eagle-groups: applies clean (inert KV connector) -> INCLUDED.
  * marlin-repack-staged-sm80: NVIDIA sm80-only -> NOT ported.
  * dflash2-ngram-chains (+431 lines): 8 hunks need recut -> DEFERRED; it is
    the top candidate for future acceptance gains on verbatim-heavy tasks.
- KVarN patches (recut for 6a9c69f) apply clean on v0.28.0: 0 failed hunks.
- Stacked audit on v0.28.0: 12 applied / 0 failed / 4 skipped
  (3 drops + the redundant backport).

## v0.28.0 re-tests (R9700, DFLASH_TOKENS=4, temp 0, 4x160 tok) - 2026-08-27

| phase | acceptance | accept len | throughput |
|---|---|---|---|
| single stream | 38.9% | 1.56 | 50.5 tok/s |
| batched (4) | 37.4% | 1.50 | 71.3 tok/s aggregate |

Parity with the 6a9c69f+#53628 image (52.1 / 69.5), identical temp-0
outputs (same tlen per prompt), 0 restarts. Serving stack now runs
qwen38-r9700:vllm-port0280 via docker-compose.port.yml.
