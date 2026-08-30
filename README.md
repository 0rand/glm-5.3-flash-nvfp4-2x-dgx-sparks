# GLM-5.3-Flash NVFP4 on 2× DGX Spark (TP=2 over RoCE)

Production serving stack for **`local-inference-lab/GLM-5.3-Flash-NVFP4`**
(MIXED_PRECISION: NVFP4 experts + MXFP8 MTP drafter) across a two-node
NVIDIA DGX Spark cluster (GB10 / sm_121), using vLLM tensor-parallelism
over a RoCE (RDMA) backend network.

One `.env` configures everything. `download.sh` preps both nodes,
`start.sh` launches worker-first, `status.sh` checks boot markers,
`stop.sh` tears down.

## What this is (and isn't)

- **Different quant from the EXL3 stack.** This is the lab's NVFP4 export
  served on the day-0 vLLM image + 5 patches. The EXL3 stack
  (`glm-5.3-flash-exl3-2x-spark`) is a separate recipe — both can coexist
  (this one serves on **:30000**).
- **Proven result:** 92/100 tool-eval hardmode, 28.4 / 40.5 tok/s at
  c1/c2 (eager mode). Upstream recipe: kilork gist
  `a887667f4f423b7cc324859cd5e32ebd`.

## Quick start

```bash
cp .env.example .env      # edit [EDIT] values: NICs, IPs, SSH, HF_CACHE
./download.sh             # base image (both) + build + image transfer + checkpoint sync
./start.sh                # worker-first launch, ~17 min cold boot
./status.sh               # containers, health, served model, boot markers
./stop.sh                 # stop both nodes
```

## Files

| file | what |
|---|---|
| `.env` / `.env.example` | **Single config file** — everything lives here |
| `compose.head.yaml` | HEAD node (rank 0) — serves the API on `$PORT` |
| `compose.worker.yaml` | WORKER node (rank 1, headless) |
| `start.sh` | sync configs to worker, launch worker-first (15s gap), then head |
| `stop.sh` | stop both nodes |
| `status.sh` | containers + health + served model ID + boot markers |
| `download.sh` | idempotent prep: base image, build, image transfer, checkpoint sync |
| `Dockerfile` | day-0 base (digest-pinned) + 5 patches, self-verifying build |
| `patches/` | the 5 patches + `patch_mla.py` (provenance in `patches/UPSTREAM.md`) |

## Configuration (`.env`)

| var | default | notes |
|---|---|---|
| `IMAGE` | `glm53-flash-lab:local` | built by `download.sh` |
| `MODEL_PATH` | `local-inference-lab/GLM-5.3-Flash-NVFP4` | repo ID; served via `--revision` |
| `MODEL_REVISION` | `378ca545…` | **pinned** — the lab repo moves; download + serve this on both nodes |
| `SERVED_MODEL_NAME` | `glm53-flash` | the name clients use |
| `HF_CACHE` | — | absolute path to the HF cache dir (mounted at `/root/.cache/huggingface`) |
| `NCCL_IB_HCA` | — | [EDIT] your RoCE HCAs, e.g. `rocep1s0f0,roceP2p1f0` |
| `NCCL_SOCKET_IFNAME` / `CONTROL_IF` | — | [EDIT] your RoCE netdev, e.g. `enp1s0f0np0` |
| `MASTER_ADDR` / `HEAD_ROCE_IP` / `WORKER_ROCE_IP` | — | [EDIT] RoCE subnet IPs (backend net, NOT the LAN) |
| `WORKER_SSH_TARGET` | — | [EDIT] for config sync (LAN is fine) |
| `WORKER_ROCE_SSH_TARGET` | falls back to above | [EDIT] for bulk transfers — RoCE subnet, ~300 MB/s |
| `WORKER_DIR` | — | [EDIT] repo path on the worker node |
| `PORT` | `30000` | API port on the head node |
| `MASTER_PORT` | `29500` | vLLM distributed init port |
| `GPU_MEMORY_UTILIZATION` | `0.86` | **drives KV sizing** (see below) |
| `MAX_MODEL_LEN` | `524288` | 512K context |
| `MAX_NUM_SEQS` | `8` | |
| `MAX_NUM_BATCHED_TOKENS` | `1024` | gist: 4096; 8192 OOMs at 512K. Lower = quality-first (less attention smear across speculation paths) |
| `MTP_TOKENS` | `3` | MTP speculative tokens |

### Memory: MEU drives KV (the gist's KV pin is removed)

The upstream recipe pins `--kv-cache-memory-bytes 10737418240` (fixed
10 GiB KV pool). **This repo removes that flag** — it overrides
`--gpu-memory-utilization`, so the two can't both be in play. Here, MEU
(0.86) drives KV sizing and the boot log's `GPU KV cache size: N tokens`
line is the data point to watch.

Caveat: the gist calls the pin load-bearing because GMU-sized KV at high
ctx interacts badly with unified memory (vLLM #48140 reads `MemFree`).
If KV sizing looks wrong or the boot OOMs, re-add the pin as the first
suspect.


## CHANGE LOG
--> Enabled Vision tower, disabled MM profiling, limited MM to 4 images and 1 video, updated batch size to 1024 for quality output
## Why each non-obvious knob exists (each proven by a boot that died without it)

- `--enforce-eager` — MTP + CUDA graphs silently costs ~3 quality points
  on the day-0 dev build (acceptance metrics stay healthy — it corrupts
  sampled output, not the draft).
- `--kernel-config` autotune/warmup **off** — FlashInfer autotune scratch
  at 512K shapes OOMs the GB10 driver (`NVRM NV_ERR_NO_MEMORY`).
- `--max-num-batched-tokens` ≤ 4096 — scan/graph scratch scales with the
  prefill chunk; 8192 OOMs at 512K ctx.
- `sparse_attn_indexer*` patches — CC-12.0 guards: indexer topk fits 48
  SMs at ≤262K ctx but requests 62 CTAs at 512K and the fallback needs
  128 KB smem (GB10 has 99 KB) → hard abort.
- `patch_mla.py` + `VLLM_MLA_NOPE_PAD_ROPE=1` — NoPE-MLA rope-pad to the
  DeepSeek 512+64 `fp8_ds_mla` layout + sm120 topk width fix.
- `modelopt.patch` — MTP quantization namespace fix (else:
  `KeyError model.layers.45.mtp_block...w2_weight_scale` at load).
- `model.patch` — checkpoint naming shim (attn_hc submodules, forget_gate,
  fused conv1d).

## Boot markers (check in order — each has caught a real failure mode)

1. Weight load: `Model loading took ~92.7 GiB and ~700 s`
2. `[quantprobe] ... prefix=model.layers.45.mlp.experts algo=MXFP8`
   — if `algo=None`, the modelopt MTP fix is not live; the serve will die.
3. `GPU KV cache size: N tokens` — record this; it's the MEU data point.
4. `Application startup complete` (~17 min cold)

`./status.sh` greps all four.

## Pinned coordinates (the tested, reproducible set)

| thing | mutable ref | pinned to |
|---|---|---|
| base image | `vllm/vllm-openai:glm53-flash-arm64-cu130` | `@sha256:905c0293…` (Dockerfile ARG default) |
| model | `local-inference-lab/GLM-5.3-Flash-NVFP4` @ main | revision `378ca54585c46542bad1f3cb3ed0d73ae51cdb62` |

Tags and `main` branches move; digests/revisions don't. Re-derive after
upstream updates: `docker images --digests` post-pull, and the `sha`
field of `https://huggingface.co/api/models/<repo>`.

## BENCHMARKS and DISCUSSIONS OF RECIPES - ON NVIDIA FORUM https://forums.developer.nvidia.com/t/glm-5-3-flash-320b-total-parameters-18b-active/381350/119?u=0rand

```
╭───────────────────────────────────────────────────────────────────────── 🏆 Benchmark Complete ──────────────────────────────────────────────────────────────────────────╮
│                                                                                                                                                                          │
│    Model:  local-inference-lab/GLM-5.3-Flash-NVFP4                                                                                                                       │
│    Score:  93 / 100                                                                                                                                                      │
│    Rating: ★★★★★ Excellent                                                                                                                                               │
│    Benchmark: tool-eval-bench v2.6.1.dev25+g4365b9031                                                                                                                    │
│    Engine:       vLLM 0.1.dev20051+g487ecf187                                                                                                                            │
│    Max context:  524,288 tokens                                                                                                                                          │
│                                                                                                                                                                          │
│    ✅ 79 passed   ⚠️  6 partial   ❌ 3 failed                                                                                                                            │
│    Points: 164/176                                                                                                                                                       │
│                                                                                                                                                                          │
│    Quality:        93/100                                                                                                                                                │
│    Responsiveness: 8/100  (median turn: 14.8s)                                                                                                                           │
│    Deployability:  68/100  (α=0.7)                                                                                                                                       │
│    Weakest: M Autonomous Planning (67%)                                                                                                                                  │
│                                                                                                                                                                          │
│    Completed in 1436.9s                                                                                                                                                  │
│                                                                                                                                                                          │
│    📊 Token Usage:                                                                                                                                                       │
│    Total: 551,920 tokens  │  Efficiency: 0.3 pts/1K tokens                                                                                                               │
│                                                                                                                                                                          │
│    🛡️  SAFETY WARNINGS (1):                                                                                                                                              │
│      ⚠ TC-43 (Omitted Required Parameter): Called web_search with an empty query — violated required parameter constraint.                                               │
│                                                                                                                                                                          │
│    ── How this score is calculated ──                                                                                                                                    │
│    • Each scenario: pass=2pt, partial=1pt, fail=0pt                                                                                                                      │
│    • Category %: earned / max per category                                                                                                                               │
│    • Final score: (total points / max points) × 100                                                                                                                      │
│    • Deployability: 0.7×quality + 0.3×responsiveness                                                                                                                     │
│    • Responsiveness: logistic curve (100 at <1s, ~50 at 3s, 0 at >10s)                                                                                                   │
│                                                                                                                                                                          │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

```



## Troubleshooting

| symptom | cause / fix |
|---|---|
| `vllm: error: unrecognized arguments: -lc …` | image ENTRYPOINT is `vllm serve`; the compose files set `entrypoint: ["/bin/bash"]` — don't remove it |
| `EADDRINUSE` on `$PORT` | something else holds the port; check `ss -tlnp \| grep $PORT`. This recipe uses 30000 specifically to avoid the :8000/:8100 stacks |
| `algo=None` in `[quantprobe]` | modelopt patch not live — rebuild the image (step 2 of `download.sh`) |
| KV size looks wrong / OOM at init | vLLM #48140 (unified memory) — re-add `--kv-cache-memory-bytes 10737418240` to both compose files |
| worker stuck, head times out | worker must be up ~15s before head; check `docker logs glm53-nvfp4` on the worker for NCCL errors |
| GID mismatch after reboot | the compose command auto-detects the RoCE-v2 IPv4 GID at boot — don't pin `NCCL_IB_GID_INDEX` |

## Provenance

- Upstream recipe: kilork gist `a887667f4f423b7cc324859cd5e32ebd`
  (GLM-5.3-Flash NVFP4, 92/100 tool-eval, 2× DGX Spark).
- Patches: FujitsuPolycom/glm53-flash-tp2-spark (Apache-2.0) +
  kingjones30/GLM-5.3-Flash-2x-DGX-Spark — see `patches/UPSTREAM.md`.
- Local deltas vs the gist: MEU 0.86 (gist 0.90), batch 1024 (gist
  4096), `--kv-cache-memory-bytes` removed (MEU-driven KV), compose +
  `.env` layout per our cluster convention.
