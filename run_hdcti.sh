#!/usr/bin/env bash
set -e

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
export NVIDIA_TF32_OVERRIDE="0"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-1}"
unset TF_GPU_ALLOCATOR
CONFIG_PATH="${1:-${HDCTI_CONFIG:-HDCTI.conf}}"
# Enable this only when TensorFlow reports BFC allocator fragmentation OOM.
# It can segfault with TensorFlow 2.6.x on newer GPUs/WSL.
# export TF_GPU_ALLOCATOR="cuda_malloc_async"

if [[ "${HDCTI_VERBOSE_TF:-0}" == "1" ]]; then
    python main.py --config "$CONFIG_PATH"
else
    python main.py --config "$CONFIG_PATH" 2> >(
        grep --line-buffered -v -E \
            'All log messages before absl::InitializeLog|port\.cc:153|cpu_feature_guard\.cc:227|mlir_graph_optimization_pass\.cc:437|disable_resource_variables|^Instructions for updating:$|^non-resource variables are not supported|gpu_device\.cc:2043' \
            >&2
    )
fi
