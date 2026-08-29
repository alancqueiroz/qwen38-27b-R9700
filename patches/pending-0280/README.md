# Pending recuts for the 0.28.0 port (2026-08-29 analysis)

`dflash2-ngram-chains.upstream.patch` (from syv-ai main, commit c954724)
cannot be recut alone: it layers on the DFlash2 **lookup lane**
(`suffix_lookup` over req_states token history), which vLLM 0.28.0 does not
have (grep-verified: zero hits for suffix_lookup / VLLM_DFLASH2_LOOKUP in the
installed package). The lookup lane exists only in our 0.27.1-era patch
(dflash2-lookup-drafting.0271.patch, 947 lines / 7 files).

A working chains port = TWO recuts against the restructured 0.28.0
spec-decode (thin DFlash2Speculator subclass + base DFlashSpeculator +
DraftModelSpeculator + model_runner hooks):
1. dflash2-lookup-drafting -> 0.28.0 (~950 lines, biggest risk: model_runner
   hook points moved)
2. dflash2-ngram-chains -> on top (~430 lines, single file once lookup lands)

Gate stays VLLM_DFLASH2_CHAIN=1 (default off). Value: drafter-free verify
blocks on self-reproducing context (verbatim tasks) - skips the drafter
forward AND its graph replay.

Note: on the port image `VLLM_DFLASH2_LOOKUP=1` is INERT (nothing reads it);
our measured 52.1/43.8 tok/s are plain DFlash2 drafting + split-KV verify.
