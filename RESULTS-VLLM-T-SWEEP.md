# vLLM 0.28.0 ROCm 7.2 - sweep DFlash2 T=2..5 (visao + contexto longo)

Branch: vllm-dflash2-t-sweep | Imagem: qwen38-r9700:vllm-port0280
Perfil: VISION=1, KV_VISION 6 GiB (pool 163-172k int8), MAX_LEN 131072 (1.24-1.32x concurrency)
Protocolo: 4x160 sequenciais, temp 0, ignore_eos (identico ao sweep do llama.cpp).

| T (drafts) | decode tok/s | aceitacao media | pool KV |
|---|---|---|---|
| 2 | 18.5 | 0.5% | 172,424 |
| 3 | 18.4 | 0.0% | 169,558 |
| 4 | 18.3 | 0.1% | 166,290 |
| 5 | 12.5 | 0.2% | 163,023 |

**Contexto da regressao**: com a aceitação colapsada (drift de wheels no rebuild,
documentado na branch amd-r9700-lookup-chains), todo T fica no ritmo target-only
(~18.4); T=5 afunda por puro overhead de draft rejeitado. Com a aceitacao
saudavel da manha (~50%), a expectativa e T=3-4 > T=2, e o nivel geral ~2.4x maior.

**Profundidade (T=4, visao)**: 40.323 tokens de prompt + 160 gen em 66,3 s
(prefill ~700 t/s; decode a 40K ~= 18,3 tok/s — sem degradacao vs curto).

## Comparacao com llama.cpp (mesmo protocolo, CTX 131072, KV q8_0)

| Stack | config | curto | 40K |
|---|---|---|---|
| llama.cpp local (master) | draft-mtp 3 | **41.8** | - |
| llama.cpp b10689 | draft-mtp 3 | 35.4 | 35.8 (Q4_XL) / 29.4 (Q5) |
| llama.cpp b10689 | draft-dflash | 32.1-33.9 | - |
| vLLM 0.28.0 (port) | dflash2 T=4 | 18.3 (regresso) | 18.3 |
| vLLM 0.28.0 (port) | dflash2 T=4 (manha, aceitacao OK) | ~43.8 | - |

Conclusao: com a aceitacao saudavel o vLLM T=3-4 e competitivo (~44 vs 41.8);
hoje, com a regressao, o llama.cpp (draft-mtp 3) e o serving rápido.
