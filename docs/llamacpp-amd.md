# llama.cpp on the R9700 — build, speculative decoding, comparison with vLLM

Branch: `amd-r9700-llamacpp` (isolated from `amd-r9700-vllm-v0280`).

## Why

The vLLM port stack (0.28.0 + DFlash2 z-lab T=4 + KVarN) is our daily serving
baseline. llama.cpp grew the same speculative family natively
(`COMMON_SPECULATIVE_TYPE`: DRAFT_SIMPLE, DRAFT_EAGLE3, DRAFT_MTP, DRAFT_DFLASH,
DRAFT_DSPARK, NGRAM_*), so it can now be compared apples-to-apples on the same
model + drafter pair, with quantized weights (GGUF) that vLLM on ROCm cannot
match today (no AWQ/GPTQ W4A16 gen path on gfx1201 in 0.28.0).

## Sources

- <https://rocm.docs.amd.com/projects/llama-cpp/en/latest/install/llama-cpp-install.html>
  (official: ROCm 7.0.0 userspace, prebuilt `rocm/llama.cpp:llama.cpp-b6652.amd0_rocm7.0.0_ubuntu24.04`,
  hipBLAS/hipBLASLt + rocWMMA as the key acceleration libs)
- <https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/optimization/model-acceleration-libs.html>
  (rocWMMA for warp-level MMA / flash attention; hipBLASLt GEMM dispatch → we
  build with `GGML_HIP_ROCWMMA_FATTN_GFX12=ON` and run with `ROCBLAS_USE_HIPBLASLT=1`)

## Artifacts used

| Piece | File | Origin |
|---|---|---|
| Main model | `Qwen3.8-27B-Q5_K_M.gguf` (19 GB) | local `/home/alan/modelos` (unsloth quant family) |
| MTP sidecar | `mtp-Qwen3.8-27B-Q4_0.gguf` (221 MB) | `unsloth/Qwen3.8-27B-GGUF` `MTP/` |
| DFlash2 sidecar | `Qwen3.8-27B-DFlash2-zlab-Q8_0.gguf` | `z-lab/Qwen3.8-27B-DFlash2-GGUF` (same z-lab drafter as the vLLM stack) |

Q5_K_M is the chosen cost-benefit point: ~19 GB weights leaves ~9-10 GB of the
32 GB card for KV cache + compute buffers at 64k context, at a quality level
K-quant-wise above Q4_K_M; the unsloth UD dynamic quants (`UD-Q4_K_XL`,
`UD-Q6_K_XL`) are the next levers if we want more context or more quality.

## Stack

- `docker/Dockerfile.llamacpp` — git master llama.cpp, HIP backend,
  `AMDGPU_TARGETS=gfx1201`, rocWMMA flash-attention for gfx12, `LLAMA_CURL=ON`.
  Base `rocm/dev-ubuntu-24.04:7.2.4-complete` (same userspace as the vLLM
  port image; already proven on this host).
- `docker/run-llamacpp.sh` — knob-style launcher (SPEC=none|mtp|dflash, CTX,
  NGL, PORT), mirroring `start_qwen_amd.sh` semantics.
- `docker-compose.llamacpp.yml` — service `llamacpp-single`, port 18021,
  mounts `/home/alan/modelos` at `/app/gguf`.
- `single-user/probe_llamacpp.sh` — same probe protocol as vLLM: sequential
  requests, temp 0, ignore_eos, 4 × 160 tokens, wall-clock + server timings.

## Host drivers: 6.x vs 7.x (question from 2026-08-27)

Measured facts on this machine:

- Host userspace: **ROCm 6.4.2** (`/opt/rocm-6.4.2`; the "7.5.0" printed by
  `rocm-smi --version` is `rocm_smi_lib`'s own version, not the ROCm release).
- Kernel: **7.0.0-30-generic**, in-tree `amdgpu` (no DKMS package installed).
- The running vLLM port container already carries **ROCm 7.2.4 userspace**
  (`rocm/dev-ubuntu-24.04:7.2.4-complete`) and has served on this host for
  weeks — i.e. **container-7.x on host-6.x already works and is our status quo**.

Conclusion: **do not upgrade the host.** Containers own their entire ROCm
userspace; the host only contributes the kernel driver (`amdgpu`/KFD), and the
in-tree driver of kernel 7.0 is current. Bumping host userspace 6.4.2 → 7.x
buys nothing for containers and risks breaking host-side tooling
(`rocm-smi`, monitoring). Only revisit if a future container needs a kernel
feature newer than the running kernel's amdgpu — the fix then is a kernel
update, not a host ROCm upgrade.

## Results (single-stream, 4 x 160 tok, temp 0, ignore_eos)

llama.cpp commit ca3d5a3e (git master, 2026-08-27), Q5_K_M, 64k ctx,
--flash-attn on (rocWMMA gfx12 path), KV bf16, port 18021.

| Stack | Config | tok/s | Draft acceptance |
|---|---|---:|---|
| llama.cpp master | plain (no spec) | 24.8 | - |
| llama.cpp master | MTP (in-model nextn layer, no sidecar needed) | **38.4** | 59.1%, 2.77 tok/step |
| llama.cpp master | DFlash2 z-lab Q8_0, n_max=3 (default) | **41.2** | 64.8%, 2.94 tok/step |
| llama.cpp master | DFlash2, n_max=2 | 39.8 | |
| llama.cpp master | DFlash2, n_max=4 | 39.2 | |
| llama.cpp master | DFlash2 batched (4 slots, 4x160 conc.) | 40.7 aggregate | |
| vLLM 0.28.0 port | MTP, hook OFF, W4A16 | 50.7 | |
| vLLM 0.28.0 port | DFlash2 T=4, W4A16, hook ON | 52.1 | 40.1%, 1.60 tok/step |
| vLLM 0.28.0 port | DFlash2 T=4 batched (8 seqs) | 69.5-71.3 aggregate | |

Findings:

1. **vLLM keeps the lead everywhere**: +26% single-stream (52.1 vs 41.2) and
   +73% batched aggregate (71.3 vs 40.7). The W4A16 compressed-tensors GEMMs
   and vLLM continuous batching scale better on RDNA4 than GGUF K-quant
   kernels in llama.cpp HIP backend.
2. **Speculative decoding works in both stacks and pays in both**: +55% MTP
   and +66% DFlash2 over plain llama.cpp. The z-lab drafter accepts MORE in
   llama.cpp (64.8% vs 40.1%) because draft-dflash there uses block drafting
   (block_size=8, n_extract=5, mask token) rather than the vLLM per-token
   chain, yet the higher acceptance still loses the throughput race - the
   base K-quant decode step is simply slower.
3. **The unsloth main GGUF already contains the MTP head** (blk.64.nextn.*
   tensors): --spec-type draft-mtp needs no sidecar. The separate
   mtp-Qwen3.8-27B-Q4_0.gguf (221 MB) is redundant for this quant.
4. **n_max=3 (default) is the DFlash2 sweet spot** in llama.cpp - same
   T=3-ish shape the vLLM sweep showed (T=4 wins there, T=3/5 lose).
5. **VRAM**: llama.cpp full stack (main + drafter + 4 slots) is about
   25 GiB (79%); the vLLM stack about 27.3 GiB (85%). The 19 GB Q5_K_M
   leaves ~6 GB headroom - enough for the drafter, not enough to push
   context much past 64k in bf16 KV (use --cache-type-k/v q8_0 to stretch it).

## Verdict

Keep vLLM as the serving stack. llama.cpp is now a credible second runtime on
the R9700 - first-class DFlash2/MTP support, 41 tok/s single-stream is honest
- and it is the only one of the two that runs GGUF K-quants today. If the
goal were maximum context on 32 GB (e.g. UD-Q4_K_XL + q8_0 KV), llama.cpp
would be the way. For latency and batched throughput, the port stack stays.
