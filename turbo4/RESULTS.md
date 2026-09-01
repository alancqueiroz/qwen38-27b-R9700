# Resultados — TurboQuant turbo4 (fork AtomicBot b10269-1.5.1) + Vulkan/RADV na R9700

Modelo: Qwen3.8-27B-UD-Q4_K_XL.gguf (arch qwen35, NextN blk.64 embutido) + mmproj BF16 | CTX 98304 | KV: K q8_0 (auto-upgrade GQA 6:1) + V turbo4

| Config      | Decode @90K ctx | Prefill @90K | Aceitacao draft | Len media | VRAM pico |
|-------------|-----------------|--------------|-----------------|-----------|-----------|
| SPEC=none   | 23,08 t/s       | 409,9 t/s    | -               | -         | ~19,8 GiB*|
| MTP2        | **28,99 t/s**   | 341,9 t/s    | 87,6%           | 2,74      | 31,63 GiB |
| MTP3        | 26,82 t/s       | 322,7 t/s    | 81,1%           | 3,43      | ~31,4 GiB |
| MTP4        | 28,06 t/s       | 340,9 t/s    | 74,5%           | 3,97      | 31,76 GiB |

\* medido antes do ratchet de buffers FA; com KV 90K preenchido, todas as rodadas MTP ficaram 31,4-31,8 GiB de 31,86.

- Ganho MTP2 vs baseline: **+25,6%** em contexto grande; pico em ctx curto: 29,4-30,4 t/s.
- MTP3/MTP4 nao superam MTP2 em 90K (mais verificacao desperdicada; NextN de 1 camada encadeia com erro composto).
- Prefill com MTP ativo custa ~15-20% (342 vs 410 t/s) — overhead do draft shared-model.
- Visao (mmproj BF16) carrega e processa imagem sem crash neste fork.
- Limite pratico: com 90K preenchido sobram ~100-230 MB; em 96K cheio usar -ub 512 ou SPEC=none.
- Opcao de K turbo4 puro: TURBO_AUTO_ASYMMETRIC=0 (nao recomendado: o fork promove K p/ q8_0 por qualidade).

Uso: cd turbo4 && docker compose --env-file .env.turbo4 -f docker-compose.turbo4.yml up -d (DRAFT_MAX=2)
Rollback diario: docker start rocmfpx-r9700-vk
