# Vendored patches — provenance

`model.patch`, `modelopt.patch`, `sparse_attn_indexer.patch`,
`sparse_attn_indexer_kpool.patch`: unmodified copies from
[FujitsuPolycom/glm53-flash-tp2-spark](https://github.com/FujitsuPolycom/glm53-flash-tp2-spark)
(master @ vendoring date 2026-08-27), Apache License 2.0. See their
`THIRD_PARTY_NOTICES.md` as well. Verified to apply with zero fuzz against
`vllm/vllm-openai:glm53-flash-arm64-cu130`.

`patch_mla.py`: the NoPE-MLA / sm120 sparse-MLA mod from
[kingjones30/GLM-5.3-Flash-2x-DGX-Spark](https://github.com/kingjones30/GLM-5.3-Flash-2x-DGX-Spark)
(see also `docker-images/glm53-flash-nvfp4/`).

Not vendored from FujitsuPolycom: `image/patches/glm5next-kda.patch`
(requires their base's chriswritescode SM120 overlay; enables
`--kda-prefill-backend flashkda`), `image/patches/renderers-base.patch`
(multimodal renderer; we serve `--language-model-only`), and
`image/nccl/libnccl.so.2.30.7` (direct-cabled-ring fabric; our clusters use
the stock NCCL with per-cluster `[nccl.*]` config).
