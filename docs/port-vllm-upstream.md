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
| hybrid-kv-groups-v2-cudagraph | FAIL | rebase needed |
| spec-decode-int8-kv | FAIL | rebase needed |
| vision-tower-cpu-offload | FAIL | rebase, low priority (text-only serving) |
| xgrammar-spec-terminated | FAIL | rebase; check upstream fixed it natively |
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
      --build-arg VLLM_GIT_REF=6a9c69fa8513 \
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
