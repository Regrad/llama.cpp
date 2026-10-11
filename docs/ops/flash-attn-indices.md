# Local indexed Flash Attention extension

`ggml_flash_attn_ext_set_kv_indices` attaches contiguous I32 columns of shape
`[n_selected, n_query]` as source 5 of `FLASH_ATTN_EXT`. The live columns must be
unique within each query. Negative and out-of-range columns are inactive.
The original additive/causal mask is applied at the selected column positions.
Only a single batch is supported. This is a private extension, not an upstream API.

CPU uses the vector reference implementation. Vulkan bypasses dense selection-mask
compaction, using either direct indices or the compatibility bitmap described below.
Vulkan accepts its supported sparse GQA geometries with 1..1024 queries, and scalar decode, without
ALiBi or softcap. Quantized K/V remain quantized; no gathered F16 cache is created.

The backend registry must expose a non-null capability token named
`ggml_backend_flash_attn_indices`. The token is only checked for presence and must
not be called. Older plugins and other backends are rejected for this form of FA,
preventing silent dense attention when loading the original CPU plugins.
The Qwen graph probes the selected Vulkan device before using the extension and
otherwise builds the original mask path. There is no minimum KV length. Decode uses
one GQA head group per query; cooperative-matrix-1 prefill can also group heads
when its tile geometry matches the dense kernel. Indexed calls do not dequantize
the full KV cache into an F16 scratch. More than 1024 queries, unsupported head
groups and multi-stream graphs use the original path.

`test_flash_attn_indices` checks against both CPU and independently constructed
dense selection masks. It covers F16/Q8_0, strided K/V cache views, 1/3/8/9/17/33/128/1024 queries,
256/1024/1056/4096/65536/200000 columns, up to 2051 slots, dead sentinels, the incomplete tail and
causal exclusion of a selected column. These are correctness checks, not benchmarks.

For cooperative-matrix-1 prefill (more than eight queries), Vulkan converts the
selection to a one-bit-per-column map. Attention retains the dense kernel's KV
tiles and split boundaries, including masked values within live tiles. If the
dense and grouped tile geometries match, one head group processes each query
and skips wholly unselected tiles before loading the mask. Otherwise it retains
the dense query grouping. Compacting individual columns changes the order of FP16 accumulation.
The map uses `ceil(n_kv/32)*4*n_query` bytes and is populated entirely on the GPU.
Decode and short speculative verification retain their existing indexed path.
This numerical compatibility change does not remove the original causal mask.

Regression checks include partial query tiles and a KV length not divisible by
64. Captured-operation replay additionally requires bitwise agreement with the
pre-prefill runtime; generic cross-backend tests use their numerical tolerance.

# Local Qwen MTP cache-only catch-up

The single-layer Qwen4Exp MTP graph can omit its unused output when a batch requests zero outputs, training and embedding output are off, and next-token hidden export is disabled or masked. Catch-up uses the target's hidden states, not the omitted draft result. Attention K/V writes and indexer raw/pooled-key writes keep their original projections, normalization, rotations and storage operations. The graph does not compute query scores, attention, the FFN or the output head for these batches. Batches that request outputs keep the full graph, and graph reuse already distinguishes output counts.

This does not shorten the draft attention window or change speculative acceptance. Validate complete token streams with MTP enabled, including transitions from prompt catch-up to drafting and verification.

The hybrid input still fills its attention mask, so cache-only graphs retain that input allocation. The graph allocator must also validate the sizes of unused input leaves when reusing allocations: pooled masks can grow with the cache even when no computed node reads them. `test_unused_input_growth` covers repeated growth and shrinkage with single- and multi-buffer allocators.

# Local Qwen PLE positional reader

`LLAMA_PLE_DIRECT_READ=1` enables the private reader for a lazily mapped Qwen4Exp PLE table. Other models, resident tables and the default environment are unchanged. It keeps 16 workers, deduplicates requested rows, orders them by file offset and reads only their packed bytes. Windows uses `ReOpenFile` and buffered overlapped `ReadFile` with a separate event for each worker; POSIX uses `pread`. This is not unbuffered sector I/O and does not disable the OS file cache.

The existing ggml row decoder produces F32 inputs in the original row order. The hash, weights, downstream PLE operators and KV quantization are unchanged. No table-wide preload is performed. The reader owns its file handle and workers; gathers on the same reader are serialized. Invalid row indices, truncated data and read errors are rejected before an incomplete result is used.

`LLAMA_PLE_DIRECT_VERIFY=1` is an opt-in correctness check: every staged row is compared bitwise to the mapped table decoded with the same ggml function. Leave it unset for normal use, since it deliberately reads both paths. `test-qwen-ple-reader` compares against CPU `GET_ROWS` for F32/F16/Q8_0/IQ4_NL with one and 16 workers, duplicate/first/last/empty rows, repeated and concurrent gathers, invalid indices and a truncated backing file followed by recovery.

These checks validate correctness, not speed. A warm mmap can be faster than positional reads; any throughput benefit must be measured separately. Long-context indexer scoring and the original causal mask still scale with context.
