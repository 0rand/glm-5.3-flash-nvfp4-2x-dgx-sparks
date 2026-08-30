# syntax=docker/dockerfile:1.7
# GLM-5.3-Flash NVFP4 (local-inference-lab quant) on DGX Spark (SM121/GB10), TP=2.
#
# Same day-0 base + NoPE-MLA mod as docker-images/glm53-flash-nvfp4, PLUS the
# four lab-checkpoint patches from FujitsuPolycom/glm53-flash-tp2-spark
# (Apache-2.0, attribution in patches/UPSTREAM.md):
#   * model.patch               — checkpoint naming shim (attn_hc.* submodules,
#                                 forget_gate flatten, fused [3C,1,4] conv1d split)
#   * modelopt.patch            — MTP MIXED_PRECISION fix: the draft model
#                                 resolves quantization across the checkpoint /
#                                 target-mapper namespaces (without it:
#                                 KeyError model.layers.45.mtp_block...w2_weight_scale)
#   * sparse_attn_indexer*.patch — disable persistent_topk / cooperative
#                                 workspace on CC 12.x (requests 62 blocks vs 48,
#                                 128 KB smem vs GB10's 99 KB)
#
# NOT vendored (base-image or fabric specific, see recipe header): the upstream
# image-layer patches (flashkda prefill, mm-renderer), their bundled NCCL 2.30.7
# (direct-cabled ring), and the instanttensor loader (optional speed-up).
#
# Verification markers: `[quantprobe]` in modelopt.py (runtime log must show
# algo=MXFP8 for the MTP layer), `hc_attn_base` in model.py.

ARG BASE_IMAGE=vllm/vllm-openai:glm53-flash-arm64-cu130@sha256:905c02933be6021301db2dc284e24e3727467aa3a0f63b41d609885778a07bce
FROM ${BASE_IMAGE}

# 1) NoPE-MLA rope-pad + sm120 sparse-MLA topk mod (anchors self-assert).
COPY patch_mla.py /tmp/patch_mla.py

# 2) FujitsuPolycom lab-checkpoint patches (unmodified upstream .patch files).
COPY model.patch modelopt.patch \
     sparse_attn_indexer.patch sparse_attn_indexer_kpool.patch /tmp/

RUN set -eux; \
    B=/usr/local/lib/python3.12/dist-packages/vllm; \
    python3 /tmp/patch_mla.py; \
    patch "$B/models/glm5next/nvidia/model.py" /tmp/model.patch; \
    patch "$B/model_executor/layers/quantization/modelopt.py" /tmp/modelopt.patch; \
    patch "$B/model_executor/layers/sparse_attn_indexer.py" /tmp/sparse_attn_indexer.patch; \
    patch "$B/model_executor/layers/sparse_attn_indexer_kpool.py" /tmp/sparse_attn_indexer_kpool.patch; \
    grep -q quantprobe "$B/model_executor/layers/quantization/modelopt.py"; \
    grep -q hc_attn_base "$B/models/glm5next/nvidia/model.py"; \
    python3 -m py_compile \
        "$B/models/glm5next/nvidia/model.py" \
        "$B/model_executor/layers/quantization/modelopt.py" \
        "$B/model_executor/layers/sparse_attn_indexer.py" \
        "$B/model_executor/layers/sparse_attn_indexer_kpool.py"; \
    rm -f /tmp/patch_mla.py /tmp/*.patch
