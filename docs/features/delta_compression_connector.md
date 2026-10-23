# `DeltaCompressionConnector` User Guide (`vLLM V1`)

`DeltaCompressionConnector` (`vllm/distributed/kv_transfer/kv_connector/v1/delta_compression_connector.py`) is a drop-in **vLLM V1 `KVConnectorBase_V1`** plugin that extends vLLM's exact-prefix caching (`BlockPool`) with **position-shifted cross-turn KV cache reuse** and **1-hop anchored residual delta compression** directly in **GPU HBM (`store_device="cuda"`)** or **CPU DRAM (`store_device="cpu"`)**.

---

## 1. Quickstart

### 1.1 `vllm serve` CLI (Recommended GPU HBM Exclusive Cache Mode)

Launch `vllm serve` with `--enable-prefix-caching` and pass `--kv-transfer-config` with `"kv_connector": "DeltaCompressionConnector"`:

```bash
vllm serve deepseek-ai/DeepSeek-V4.1-Flash \
  --tensor-parallel-size 8 \
  --trust-remote-code \
  --enable-prefix-caching \
  --kv-cache-metrics \
  --kv-cache-metrics-sample 1.0 \
  --kv-transfer-config '{
    "kv_connector": "DeltaCompressionConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "store_device": "cuda",
      "max_hbm_mb": 1435.5,
      "exclusive_connector_cache": true
    }
  }'
```

### 1.2 Python Offline Inference (`LLM` & `KVTransferConfig`)

```python
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

llm = LLM(
    model="zai-org/GLM-5.3",
    tensor_parallel_size=8,
    trust_remote_code=True,
    enable_prefix_caching=True,
    kv_transfer_config=KVTransferConfig(
        kv_connector="DeltaCompressionConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "store_device": "cuda",
            "max_hbm_mb": 4076.0,          # Per-GPU KV HBM budget in MiB (31.84 GiB across TP=8)
            "exclusive_connector_cache": True,
            "tau_zero_bit": 0.06,
            "tau_int2": 0.12,
            "tau_int4": 0.85,
            "max_sparse_exceptions": 4,
        },
    ),
)
```

---

## 2. Configuration Reference (`kv_connector_extra_config`)

| Parameter | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| **`store_device`** | `str` | `"cpu"` | Storage device for `DeltaBlockStore`: `"cuda"` packs compressed blocks into 256 KB contiguous GPU HBM byte slabs on each worker's local CUDA device; `"cpu"` stores them in host DRAM. |
| **`exclusive_connector_cache`** | `bool` | `false` | When `true`, `DeltaCompressionConnector` acts as the sole long-term KV cache and immediately reclaims inactive cached blocks from vLLM's uncompressed `BlockPool` so `BlockPool` only needs scratch blocks for currently running sequences. |
| **`max_hbm_mb`** | `float` | `0` *(unlimited)* | Maximum compressed storage budget **per TP worker rank in MiB** ($1024^2$ bytes). Across `TP = N` GPUs, the total cluster KV HBM budget is `N * max_hbm_mb` MiB. |
| **`max_store_bytes`** | `int` | `0` *(unlimited)* | Exact byte equivalent of `max_hbm_mb` per worker rank (overrides `max_hbm_mb` if both are set). |
| **`tau_zero_bit`** | `float` | `0.06` | Maximum relative RMSE ($\|\Delta\|_F / \|X_{\text{ref}}\|_F$) to promote a block to **`TIER0_ZERO_BIT`** (`0` bytes of residual payload). |
| **`tau_int2`** | `float` | `0.12` | Maximum relative RMSE to compress a block with **`TIER1_INT2_DELTA`** (`2` bits/element crest-adaptive quantizer). |
| **`tau_int4`** | `float` | `0.85` | Maximum relative RMSE to compress a block with **`TIER2_INT4_DELTA`** (`4` bits/element zero-preserving quantizer). |
| **`max_sparse_exceptions`** | `int` | `4` | Maximum number of intra-block token edits ($k \le \text{max\_sparse\_exceptions}$) matched via Pigeonhole sub-chunk indexing for **`TIER3_SPARSE_BF16_INT4_DELTA`**. |
| **`sliding_window_dedup_threshold`** | `int` | `64` | Minimum token divergence distance $d_{\text{div}}$ after which `LOCAL_SLIDING` attention layers use `TIER0_ZERO_BIT` (`0` bits). |
| **`rope_theta`** | `float` | `10000.0` | Fallback RoPE base frequency $\theta$ if not specified in the model's `hf_config`. |
| **`force_uncompressed`** | `bool` | `false` | Diagnostic baseline flag: when `true`, disables delta compression and stores all blocks in `DeltaBlockStore` as `UNCOMPRESSED_BASE`. |
| **`enable_server_block_align`** | `bool` | `false` | Enables server-side block-aligned padding of `[File: ...]` viewer windows using `pad_token` / `pad_token_id` / `pad_token_map`. |
| **`pad_token`** | `str` | `"\n"` | Default padding string used by `resolve_pad_token_id` when aligning file viewer buckets to block boundaries. |
| **`pad_token_id`** | `int \| null` | `null` | Explicit token ID override for block-alignment padding. |
| **`pad_token_map`** | `dict[str, str] \| null` | `null` | Optional mapping of content substrings/regexes to custom padding tokens. |

---

## 3. Prometheus Metrics (`GET /metrics`)

When `vllm serve` is started with `--kv-cache-metrics --kv-cache-metrics-sample 1.0`, `DeltaConnectorPromMetrics` exposes:
- `vllm:delta_connector_raw_bytes_equivalent`: Uncompressed equivalent bytes of all blocks currently stored in `DeltaBlockStore` across all `TP` ranks.
- `vllm:delta_connector_compressed_bytes_stored`: Actual compressed bytes currently stored in `DeltaBlockStore` across all `TP` ranks.
- `vllm:delta_connector_active_blocks`: Number of active unique logical KV blocks currently resident in `DeltaBlockStore`.
- `vllm:delta_connector_evicted_blocks_total`: Cumulative unique logical KV blocks evicted from `DeltaBlockStore`.
- `vllm:delta_connector_evicted_bytes_total`: Cumulative compressed bytes evicted from `DeltaBlockStore`.
- `vllm:external_prefix_cache_queries_total` / `vllm:external_prefix_cache_hits_total`: External connector prefix cache query and hit counters.
