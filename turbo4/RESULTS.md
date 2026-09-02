# Resultados TurboQuant turbo4 — Vulkan vs ROCm 10 (sessao 01/09/2026)

Modelo: Qwen3.8-27B-UD-Q4_K_XL.gguf (arch qwen35, NextN blk.64) + mmproj BF16 | fork AtomicBot b10269-1.5.1 (binario pre-compilado)

## Estrutura de memoria descoberta (importante!)

- O modelo e **hibrido GDN**: so **16 de 65 camadas** tem KV cache quantizavel
  (K 1024 elem/camada); as outras **64 recorrentes usam estado f32 FIXO**
  (RS buffer 748 MiB, nao afetado por -ctk/-ctv). KV @98304 = 2448 MiB (K q8_0 1632 + V turbo4 816).
- **MTP nextn duplica o modelo inteiro na GPU** (16053 MiB): o "shared-model"
  do NEXTN.md nao acontece no binario b10269-1.5.1. No Vulkan (RADV) passa por
  overcommit; no HIP e malloc duro -> OOM.
- n_ctx_train = 262144 (256K nativo).

## Vulkan (RADV, container rocmfpx-turbo4-vk, porta 8091)

| Config | CTX | Decode @90K | Prefill | VRAM load/fill | Observacao |
|---|---|---|---|---|---|
| none, K q8_0 + V turbo4 | 98304 | 23,08 t/s | 410 t/s | 19,8 GiB | baseline |
| **MTP2, K q8_0 + V turbo4** | 98304 | **28,99 t/s** | 342 t/s | 31,6 GiB | vencedor (+25,6%) |
| MTP3 | 98304 | 26,82 t/s | 323 t/s | ~31,4 GiB | pior que MTP2 |
| MTP4 | 98304 | 28,06 t/s | 341 t/s | 31,8 GiB | empatado c/ MTP2 |
| MTP2, turbo4/turbo4 puro | 98304 | 20,11 t/s | 248 t/s | 31,4 GiB | -30% decode; needles 3/3 OK |
| none, turbo4/turbo4 puro | 131072 | 20,24 t/s | 281 t/s | 19,6 GiB | folga ~12 GiB |
| none, turbo4/turbo4 puro | **262144** | (gen ok) | - | **21,9 GiB** | cabe o nativo 256K! |
| MTP2, turbo4/turbo4 puro | 131072 | 21,01 t/s | 247 t/s | 31,8 GiB | rodou 90K sem OOM (limite) |

- Qualidade (needle retrieval @90K): 3/3 com K q8_0 E com K turbo4 — sem degradacao detectavel.
- O custo do K turbo4 e VELOCIDADE (-30%), nao qualidade. A "assimetria problematica"
  do Vulkan nao se manifestou: K q8_0 + V turbo4 rodou limpo em todos os testes.
- MTP2/3/4 @98304: melhor = MTP2 (MTP3/4 nao compensam: NextN de 1 camada encadeia com erro).

## ROCm 10 (HIP, container rocmfpx-turbo4-hip, porta 8092)

| Config | Resultado |
|---|---|
| none, turbo4/turbo4 | OK: 20,3 GiB load, gen ok (binario ROCm do AtomicBot roda na gfx1201) |
| MTP2, draft GPU | **OOM**: 2a copia do modelo (16053 MiB) nao cabe — HIP sem overcommit |
| MTP2, draft CPU (-ngld 0) | roda, aceitacao 87,6% igual, mas **5,52 t/s** — inviavel |
| draft-dflash (zlab 2GB) | drafter incompativel: AtomicBot nao registra arch dflash (convencao propria c/ target_layer_ids + auto-download de siblings dflash-*.gguf no HF; nenhum publico p/ Qwen3.8-27B) |

## DFlash2 no fork ROCmFPX (rocm10, CTX 98304, KV q8_0/q4_0)

- O fork ROCmFPX **registra arch dflash** e carregou o zlab: block_size=8, n_extract=5, n_max=3.
- Porem **4,0 t/s decode / 19 t/s prefill** (prompt curto!) — caminho de atencao
  mascarada nao otimizado no build HIP atual. Inviavel; o dflash2 rapido segue
  sendo o do vLLM (~43,8 t/s, branch vllm-dflash2-t-sweep).

## Recomendacoes

1. Uso diario: Vulkan MTP2 @98304, K q8_0 + V turbo4 (auto) — 28,99 t/s @90K.
2. Contexto maximo: sem MTP + turbo4/turbo4 -> 262144 nativo cabe (21,9 GiB).
3. ROCm/HIP com TurboQuant: aguardar o fork resolver a duplicacao do draft
   (shared-model real) ou um binario HIP com drafter pequeno (DFlash) otimizado.
4. K turbo4 so se o objetivo for VRAM em ctx enorme SEM MTP (a -30% de decode compensa so ali).

Uso: cd turbo4 && docker compose --env-file .env.turbo4 -f docker-compose.turbo4.yml up -d
Rollback diario: docker start rocmfpx-r9700-vk
