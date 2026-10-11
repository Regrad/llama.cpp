# Local indexed Flash Attention extension

`ggml_flash_attn_ext_set_kv_indices` attaches contiguous I32 columns of shape
`[n_selected, n_query]` as source 5 of `FLASH_ATTN_EXT`. The live columns must be
unique within each query. Negative and out-of-range columns are inactive.
The original additive/causal mask is applied at the selected column positions.
Only a single batch is supported. This is a private extension, not an upstream API.

CPU uses the vector reference implementation. Vulkan reuses its existing sparse
FA shader and binds the indices directly, bypassing dense-mask compaction.
Vulkan accepts only its supported sparse GQA/scalar decode geometries, without
ALiBi or softcap. Quantized K/V remain quantized; no gathered F16 cache is created.

The backend registry must expose a non-null capability token named
`ggml_backend_flash_attn_indices`. The token is only checked for presence and must
not be called. Older plugins and other backends are rejected for this form of FA,
preventing silent dense attention when loading the original CPU plugins.
The Qwen graph probes the selected Vulkan device before using the extension and
otherwise builds the original mask path. There is no minimum KV length for supported decode graphs. Large prefill and multi-stream graphs are unchanged.

`test_flash_attn_indices` checks against both CPU and independently constructed
dense selection masks. It covers F16/Q8_0, strided K/V cache views, 1/3/8 queries,
256/1024/4096/65536/200000 columns, up to 2051 slots, dead sentinels, the incomplete tail and
causal exclusion of a selected column. These are correctness checks, not benchmarks.
