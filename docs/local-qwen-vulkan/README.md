# Local Qwen Flash Next Vulkan work

This branch preserves experimental Windows Vulkan changes developed and tested on an AMD Radeon 8060S. It is based on llama.cpp b11461, commit `4d756bc72bf00a4aacf410ae15a2d315f3db400d`. The runtime model name is Qwen3.8-Flash-Next; the implementation is in `qwen4exp.cpp`.

## Changes

- Pass QSA-selected KV columns directly to Vulkan attention, retaining quantized KV storage, inactive slots, causal exclusion and the incomplete tail. Supported paths run at short context too; unsupported geometries retain the original path.
- For large prompt batches, construct a GPU bitmap of selected columns and skip wholly inactive tiles while retaining compatible attention tile and reduction ordering.
- Update the order-based pooled-cache layout incrementally during decode, with fallback for cache operations that need reconstruction. Avoid repeated scans of the entire context.
- For MTP catch-up batches without requested outputs, keep the KV/indexer writes and skip unused attention, FFN and output work. Validate allocation sizes of unused input tensors during graph reuse.
- Reset MTP `pending_h` at sequence position zero and save/restore it through speculative checkpoints. Pad inactive selected pool slots so the tail keeps a fixed slot, and keep native sparse attention split selection independent of MTP query count.
- Provide an optional positional PLE reader (`LLAMA_PLE_DIRECT_READ=1`). It is off by default; no general speed claim is made for it.

Implementation details and backend limitations: [indexed attention notes](../ops/flash-attn-indices.md).

## Recorded validation

These are previously completed local measurements, not benchmarks rerun for publication. Earlier performance results precede the latest MTP consistency fix. They do not establish the throughput of that final fix or promise the same improvement on other hardware.

| Check | Recorded result |
| --- | --- |
| Prompt processing, 200,000 input tokens | 200.813 -> 307.636 tokens/s (1.532x); all 318 output token IDs matched |
| Decode after incremental pooled-cache layout fix, 200,000 input tokens | 23.355 -> 36.947 tokens/s; all 318 output token IDs matched |
| Incremental layout regression | 194,735 comparisons and AddressSanitizer passed |
| Latest indexed-attention backend checks | 58/58 passed against CPU, F16/Q8_0 KV, including 200,000 KV columns |
| Batch-invariance regression | Previous Vulkan runtime 0/16; fixed runtime 16/16, bitwise checks |
| Latest MTP 2 versus MTP 3 | 6,400/6,400 output token ID pairs equal across two input variants and nine requests per mode |
| Installed-runtime smoke check | 128 token IDs matched the candidate |

**Known unresolved limitation:** full cold recomputation of a continued history and warm continuation from its cache still differed at output token 2 (466 versus 271) in the recorded case, in both MTP modes. The cause has not been proven. Repeats within each path were stable. This branch does not claim universal token equivalence, absence of quality regressions on all tasks, or a general programming-quality evaluation.

Model for the latest MTP test: `Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf`; temperature 0, top-k 1, seed 42, up to 1,024 output tokens per request. The Huihui model is configured locally but is not covered by this latest MTP comparison.

## Build and run

Use an x64 Visual Studio developer terminal with CMake, Ninja and the Vulkan SDK on PATH. From the repository root:

```bat
cmake -S . -B build-local -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_VULKAN=ON -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DGGML_RPC=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_BORINGSSL=ON
cmake --build build-local --target llama-server test-backend-ops test-alloc test-qwen-ple-reader --parallel 8
```

[start-qwen.bat](start-qwen.bat) is a configurable example of the current standard-model settings: MTP 3, context 262,144, Q8_0 KV, batch 2,048, ubatch 1,024, eight CPU threads, lazy Engram/PLE reads and Medium reasoning with preserved thinking. Supply local paths via `QWEN_SERVER`, `QWEN_MODEL`, `QWEN_MTP`, `QWEN_MMPROJ` and `QWEN_API_KEY_FILE`. The example opens the server in the calling console. Its key file must exist and must never be committed.

For the Huihui launch, use that model and projector and set `QWEN_MTP_P_MIN=0.75`; the standard-model launch uses 0.0. Model files, keys and local runtime binaries are not included in this publication.

Remove `LLAMA_QSA_INDICES_DISABLE` and `GGML_VK_FA_SPARSE_DISABLE` from the environment to use the optimized path. Setting `LLAMA_QSA_INDICES_DISABLE=1` selects the original QSA mask path; this alone does not revert the other changes. `LLAMA_PLE_DIRECT_VERIFY=1` is an optional row-decoding correctness check and adds extra reads.

The Windows runtime was retained separately in a local Git checkpoint. Installed DLL SHA-256 values are in [validation.json](validation.json). This branch publishes the source and reproducible configuration, with the measured scope and unresolved case recorded above.

AI assistance: these local changes and publication notes were developed with Codex under the user's direction.
