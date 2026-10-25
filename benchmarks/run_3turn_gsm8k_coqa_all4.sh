#!/bin/bash
set -e

MUSE_PATH=$(ls -d /root/.cache/huggingface/hub/models--unsloth--Muse-Spark-Open-816B-A42B-NVFP4/snapshots/* | head -n 1)
DS_PATH=$(ls -d /root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/* | head -n 1)
GLM_PATH=$(ls -d /root/.cache/huggingface/hub/models--zai-org--GLM-5.3/snapshots/* | head -n 1)
KIMI_PATH=$(ls -d /root/.cache/huggingface/hub/models--moonshotai--Kimi-K2.6/snapshots/* | head -n 1)

wait_vllm() {
  local label="$1"
  for i in $(seq 1 300); do
    if curl -s http://127.0.0.1:8000/v1/models | grep -q "id"; then
      echo "$label ready after $((i*2))s!"
      return 0
    fi
    sleep 2
  done
  echo "ERROR: $label failed to start!"
  exit 1
}

cleanup_vllm() {
  local pid="$1"
  kill -9 "$pid" 2>/dev/null || true
  pkill -9 -f "[v]llm serve" 2>/dev/null || true
  pkill -9 -f "[V]LLM" 2>/dev/null || true
  sleep 5
}

# ==============================================================================
# 1. DeepSeek-V4.1-Flash (BlockPool vs DeltaCompressionConnector)
# ==============================================================================
if [ ! -f /workspace/results_3turn_gsm8k_coqa_dsv41_blockpool.json ]; then
  echo "=== [1/8] Launching DeepSeek-V4.1-Flash 3-Turn GSM8K+CoQA HTTP Benchmark: BlockPool ==="
  PYTHONPATH=/workspace/vllm_pkg vllm serve "$DS_PATH" \
    --served-model-name deepseek-ai/DeepSeek-V4.1-Flash \
    --tensor-parallel-size 8 \
    --trust-remote-code \
    --max-model-len 8192 \
    --max-num-seqs 64 \
    --max-num-batched-tokens 65536 \
    --block-size 128 \
    --enable-prefix-caching \
    --kv-cache-metrics \
    --kv-cache-metrics-sample 1.0 \
    --enforce-eager \
    --kernel-config "{\"enable_flashinfer_autotune\":false,\"enable_cutedsl_warmup\":false}" \
    --port 8000 > /workspace/vllm_3turn_dsv41_bp.log 2>&1 &
  SRV_PID=$!
  wait_vllm "DeepSeek-V4.1-Flash 3-Turn BlockPool Server"

  PYTHONPATH=/workspace/vllm_pkg python3 -u /workspace/benchmarks/benchmark_3turn_gsm8k_coqa_http.py \
    --mode blockpool \
    --served-model-name deepseek-ai/DeepSeek-V4.1-Flash \
    --tokenizer "$DS_PATH" \
    --num-gsm8k 100 \
    --num-coqa 100 \
    --checkpoint-every 20 \
    --turn3-max-tokens 512 \
    --concurrency 20 \
    --block-size 128 \
    --block-bytes 9437184 \
    --capacity-blocks 82012 \
    --tp-size 8 \
    --output-json /workspace/results_3turn_gsm8k_coqa_dsv41_blockpool.json

  cleanup_vllm "$SRV_PID"
fi

if [ ! -f /workspace/results_3turn_gsm8k_coqa_dsv41_delta.json ]; then
  echo "=== [2/8] Launching DeepSeek-V4.1-Flash 3-Turn GSM8K+CoQA HTTP Benchmark: DeltaCompressionConnector ==="
  PYTHONPATH=/workspace/vllm_pkg vllm serve "$DS_PATH" \
    --served-model-name deepseek-ai/DeepSeek-V4.1-Flash \
    --tensor-parallel-size 8 \
    --trust-remote-code \
    --max-model-len 8192 \
    --max-num-seqs 64 \
    --max-num-batched-tokens 65536 \
    --block-size 128 \
    --enable-prefix-caching \
    --num-gpu-blocks-override 20000 \
    --enforce-eager \
    --kernel-config "{\"enable_flashinfer_autotune\":false,\"enable_cutedsl_warmup\":false}" \
    --kv-transfer-config "{\"kv_connector\":\"DeltaCompressionConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"store_device\":\"cuda\",\"max_hbm_mb\":65000.0,\"exclusive_connector_cache\":true,\"enable_cross_turn_reuse\":false,\"multi_cohort_threshold\":0}}" \
    --port 8000 > /workspace/vllm_3turn_dsv41_delta.log 2>&1 &
  SRV_PID=$!
  wait_vllm "DeepSeek-V4.1-Flash 3-Turn Delta Server"

  PYTHONPATH=/workspace/vllm_pkg python3 -u /workspace/benchmarks/benchmark_3turn_gsm8k_coqa_http.py \
    --mode delta \
    --served-model-name deepseek-ai/DeepSeek-V4.1-Flash \
    --tokenizer "$DS_PATH" \
    --num-gsm8k 100 \
    --num-coqa 100 \
    --checkpoint-every 20 \
    --turn3-max-tokens 512 \
    --concurrency 20 \
    --block-size 128 \
    --block-bytes 9437184 \
    --capacity-blocks 82012 \
    --tp-size 8 \
    --output-json /workspace/results_3turn_gsm8k_coqa_dsv41_delta.json \
    --compare-blockpool-json /workspace/results_3turn_gsm8k_coqa_dsv41_blockpool.json \
    --comparison-output-json /workspace/comparison_3turn_gsm8k_coqa_dsv41.json

  cleanup_vllm "$SRV_PID"
fi

# ==============================================================================
# 2. Muse-Spark-Open-816B-A42B-NVFP4 (BlockPool vs DeltaCompressionConnector)
# ==============================================================================
if [ ! -f /workspace/results_3turn_gsm8k_coqa_musespark_blockpool.json ]; then
  echo "=== [3/8] Launching Muse-Spark-816B 3-Turn GSM8K+CoQA HTTP Benchmark: BlockPool ==="
  PYTHONPATH=/workspace/vllm_pkg vllm serve "$MUSE_PATH" \
    --served-model-name unsloth/Muse-Spark-Open-816B-A42B-NVFP4 \
    --tensor-parallel-size 8 \
    --limit-mm-per-prompt "{\"image\":0,\"video\":0,\"audio\":0}" \
    --disable-hybrid-kv-cache-manager \
    --max-model-len 8192 \
    --max-num-seqs 64 \
    --max-num-batched-tokens 65536 \
    --block-size 64 \
    --enable-prefix-caching \
    --kv-cache-metrics \
    --kv-cache-metrics-sample 1.0 \
    --enforce-eager \
    --kernel-config "{\"enable_flashinfer_autotune\":false,\"enable_cutedsl_warmup\":false}" \
    --port 8000 > /workspace/vllm_3turn_muse_bp.log 2>&1 &
  SRV_PID=$!
  wait_vllm "Muse-Spark 3-Turn BlockPool Server"

  PYTHONPATH=/workspace/vllm_pkg python3 -u /workspace/benchmarks/benchmark_3turn_gsm8k_coqa_http.py \
    --mode blockpool \
    --served-model-name unsloth/Muse-Spark-Open-816B-A42B-NVFP4 \
    --tokenizer "$MUSE_PATH" \
    --num-gsm8k 100 \
    --num-coqa 100 \
    --checkpoint-every 20 \
    --turn3-max-tokens 512 \
    --concurrency 20 \
    --block-size 64 \
    --block-bytes 16252928 \
    --capacity-blocks 41120 \
    --tp-size 8 \
    --output-json /workspace/results_3turn_gsm8k_coqa_musespark_blockpool.json

  cleanup_vllm "$SRV_PID"
fi

if [ ! -f /workspace/results_3turn_gsm8k_coqa_musespark_delta.json ]; then
  echo "=== [4/8] Launching Muse-Spark-816B 3-Turn GSM8K+CoQA HTTP Benchmark: DeltaCompressionConnector ==="
  PYTHONPATH=/workspace/vllm_pkg vllm serve "$MUSE_PATH" \
    --served-model-name unsloth/Muse-Spark-Open-816B-A42B-NVFP4 \
    --tensor-parallel-size 8 \
    --limit-mm-per-prompt "{\"image\":0,\"video\":0,\"audio\":0}" \
    --disable-hybrid-kv-cache-manager \
    --max-model-len 8192 \
    --max-num-seqs 32 \
    --max-num-batched-tokens 65536 \
    --block-size 64 \
    --enable-prefix-caching \
    --num-gpu-blocks-override 12000 \
    --enforce-eager \
    --kernel-config "{\"enable_flashinfer_autotune\":false,\"enable_cutedsl_warmup\":false}" \
    --kv-transfer-config "{\"kv_connector\":\"DeltaCompressionConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"store_device\":\"cuda\",\"max_hbm_mb\":38000.0,\"exclusive_connector_cache\":true,\"enable_cross_turn_reuse\":false,\"multi_cohort_threshold\":0}}" \
    --port 8000 > /workspace/vllm_3turn_muse_delta.log 2>&1 &
  SRV_PID=$!
  wait_vllm "Muse-Spark 3-Turn Delta Server"

  PYTHONPATH=/workspace/vllm_pkg python3 -u /workspace/benchmarks/benchmark_3turn_gsm8k_coqa_http.py \
    --mode delta \
    --served-model-name unsloth/Muse-Spark-Open-816B-A42B-NVFP4 \
    --tokenizer "$MUSE_PATH" \
    --num-gsm8k 100 \
    --num-coqa 100 \
    --checkpoint-every 20 \
    --turn3-max-tokens 512 \
    --concurrency 20 \
    --block-size 64 \
    --block-bytes 16252928 \
    --capacity-blocks 41120 \
    --tp-size 8 \
    --output-json /workspace/results_3turn_gsm8k_coqa_musespark_delta.json \
    --compare-blockpool-json /workspace/results_3turn_gsm8k_coqa_musespark_blockpool.json \
    --comparison-output-json /workspace/comparison_3turn_gsm8k_coqa_musespark.json

  cleanup_vllm "$SRV_PID"
fi

# ==============================================================================
# 3. GLM-5.3 (BlockPool vs DeltaCompressionConnector)
# ==============================================================================
if [ ! -f /workspace/results_3turn_gsm8k_coqa_glm53_blockpool.json ]; then
  echo "=== [5/8] Launching GLM-5.3 3-Turn GSM8K+CoQA HTTP Benchmark: BlockPool ==="
  PYTHONPATH=/workspace/vllm_pkg vllm serve "$GLM_PATH" \
    --served-model-name zai-org/GLM-5.3 \
    --tensor-parallel-size 8 \
    --trust-remote-code \
    --max-model-len 8192 \
    --max-num-seqs 64 \
    --max-num-batched-tokens 65536 \
    --block-size 64 \
    --kv-cache-dtype fp8_e4m3 \
    --enable-prefix-caching \
    --kv-cache-metrics \
    --kv-cache-metrics-sample 1.0 \
    --enforce-eager \
    --kernel-config "{\"enable_flashinfer_autotune\":false,\"enable_cutedsl_warmup\":false}" \
    --port 8000 > /workspace/vllm_3turn_glm_bp.log 2>&1 &
  SRV_PID=$!
  wait_vllm "GLM-5.3 3-Turn BlockPool Server"

  PYTHONPATH=/workspace/vllm_pkg python3 -u /workspace/benchmarks/benchmark_3turn_gsm8k_coqa_http.py \
    --mode blockpool \
    --served-model-name zai-org/GLM-5.3 \
    --tokenizer "$GLM_PATH" \
    --num-gsm8k 100 \
    --num-coqa 100 \
    --checkpoint-every 20 \
    --turn3-max-tokens 512 \
    --concurrency 20 \
    --block-size 64 \
    --block-bytes 23068672 \
    --capacity-blocks 201941 \
    --tp-size 8 \
    --output-json /workspace/results_3turn_gsm8k_coqa_glm53_blockpool.json

  cleanup_vllm "$SRV_PID"
fi

if [ ! -f /workspace/results_3turn_gsm8k_coqa_glm53_delta.json ]; then
  echo "=== [6/8] Launching GLM-5.3 3-Turn GSM8K+CoQA HTTP Benchmark: DeltaCompressionConnector ==="
  PYTHONPATH=/workspace/vllm_pkg vllm serve "$GLM_PATH" \
    --served-model-name zai-org/GLM-5.3 \
    --tensor-parallel-size 8 \
    --trust-remote-code \
    --max-model-len 8192 \
    --max-num-seqs 50 \
    --max-num-batched-tokens 32768 \
    --block-size 64 \
    --kv-cache-dtype fp8_e4m3 \
    --enable-prefix-caching \
    --num-gpu-blocks-override 14500 \
    --enforce-eager \
    --kernel-config "{\"enable_flashinfer_autotune\":false,\"enable_cutedsl_warmup\":false}" \
    --kv-transfer-config "{\"kv_connector\":\"DeltaCompressionConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"store_device\":\"cuda\",\"max_hbm_mb\":42000.0,\"exclusive_connector_cache\":true,\"enable_cross_turn_reuse\":false,\"multi_cohort_threshold\":0}}" \
    --port 8000 > /workspace/vllm_3turn_glm_delta.log 2>&1 &
  SRV_PID=$!
  wait_vllm "GLM-5.3 3-Turn Delta Server"

  PYTHONPATH=/workspace/vllm_pkg python3 -u /workspace/benchmarks/benchmark_3turn_gsm8k_coqa_http.py \
    --mode delta \
    --served-model-name zai-org/GLM-5.3 \
    --tokenizer "$GLM_PATH" \
    --num-gsm8k 100 \
    --num-coqa 100 \
    --checkpoint-every 20 \
    --turn3-max-tokens 512 \
    --concurrency 20 \
    --block-size 64 \
    --block-bytes 23068672 \
    --capacity-blocks 201941 \
    --tp-size 8 \
    --output-json /workspace/results_3turn_gsm8k_coqa_glm53_delta.json \
    --compare-blockpool-json /workspace/results_3turn_gsm8k_coqa_glm53_blockpool.json \
    --comparison-output-json /workspace/comparison_3turn_gsm8k_coqa_glm53.json

  cleanup_vllm "$SRV_PID"
fi

# ==============================================================================
# 4. Kimi-K2.6 (BlockPool vs DeltaCompressionConnector)
# ==============================================================================
if [ ! -f /workspace/results_3turn_gsm8k_coqa_kimi_k26_blockpool.json ]; then
  echo "=== [7/8] Launching Kimi-K2.6 3-Turn GSM8K+CoQA HTTP Benchmark: BlockPool ==="
  PYTHONPATH=/workspace/vllm_pkg vllm serve "$KIMI_PATH" \
    --served-model-name moonshotai/Kimi-K2.6 \
    --tensor-parallel-size 8 \
    --trust-remote-code \
    --limit-mm-per-prompt "{\"image\":0,\"video\":0}" \
    --max-model-len 8192 \
    --max-num-seqs 64 \
    --max-num-batched-tokens 65536 \
    --block-size 64 \
    --kv-cache-dtype fp8_e4m3 \
    --enable-prefix-caching \
    --kv-cache-metrics \
    --kv-cache-metrics-sample 1.0 \
    --enforce-eager \
    --kernel-config "{\"enable_flashinfer_autotune\":false,\"enable_cutedsl_warmup\":false}" \
    --port 8000 > /workspace/vllm_3turn_kimi_bp.log 2>&1 &
  SRV_PID=$!
  wait_vllm "Kimi-K2.6 3-Turn BlockPool Server"

  PYTHONPATH=/workspace/vllm_pkg python3 -u /workspace/benchmarks/benchmark_3turn_gsm8k_coqa_http.py \
    --mode blockpool \
    --served-model-name moonshotai/Kimi-K2.6 \
    --tokenizer "$KIMI_PATH" \
    --num-gsm8k 100 \
    --num-coqa 100 \
    --checkpoint-every 20 \
    --turn3-max-tokens 512 \
    --concurrency 20 \
    --block-size 64 \
    --block-bytes 17989632 \
    --capacity-blocks 359208 \
    --tp-size 8 \
    --output-json /workspace/results_3turn_gsm8k_coqa_kimi_k26_blockpool.json

  cleanup_vllm "$SRV_PID"
fi

if [ ! -f /workspace/results_3turn_gsm8k_coqa_kimi_k26_delta.json ]; then
  echo "=== [8/8] Launching Kimi-K2.6 3-Turn GSM8K+CoQA HTTP Benchmark: DeltaCompressionConnector ==="
  PYTHONPATH=/workspace/vllm_pkg vllm serve "$KIMI_PATH" \
    --served-model-name moonshotai/Kimi-K2.6 \
    --tensor-parallel-size 8 \
    --trust-remote-code \
    --limit-mm-per-prompt "{\"image\":0,\"video\":0}" \
    --max-model-len 8192 \
    --max-num-seqs 50 \
    --max-num-batched-tokens 32768 \
    --block-size 64 \
    --kv-cache-dtype fp8_e4m3 \
    --enable-prefix-caching \
    --num-gpu-blocks-override 15000 \
    --enforce-eager \
    --kernel-config "{\"enable_flashinfer_autotune\":false,\"enable_cutedsl_warmup\":false}" \
    --kv-transfer-config "{\"kv_connector\":\"DeltaCompressionConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"store_device\":\"cuda\",\"max_hbm_mb\":60000.0,\"exclusive_connector_cache\":true,\"enable_cross_turn_reuse\":false,\"multi_cohort_threshold\":0}}" \
    --port 8000 > /workspace/vllm_3turn_kimi_delta.log 2>&1 &
  SRV_PID=$!
  wait_vllm "Kimi-K2.6 3-Turn Delta Server"

  PYTHONPATH=/workspace/vllm_pkg python3 -u /workspace/benchmarks/benchmark_3turn_gsm8k_coqa_http.py \
    --mode delta \
    --served-model-name moonshotai/Kimi-K2.6 \
    --tokenizer "$KIMI_PATH" \
    --num-gsm8k 100 \
    --num-coqa 100 \
    --checkpoint-every 20 \
    --turn3-max-tokens 512 \
    --concurrency 20 \
    --block-size 64 \
    --block-bytes 17989632 \
    --capacity-blocks 359208 \
    --tp-size 8 \
    --output-json /workspace/results_3turn_gsm8k_coqa_kimi_k26_delta.json \
    --compare-blockpool-json /workspace/results_3turn_gsm8k_coqa_kimi_k26_blockpool.json \
    --comparison-output-json /workspace/comparison_3turn_gsm8k_coqa_kimi_k26.json

  cleanup_vllm "$SRV_PID"
fi

echo "=== ALL 3-TURN GSM8K + COQA HTTP BENCHMARK RUNS COMPLETE ==="
