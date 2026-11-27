#!/usr/bin/env bash
# Run from the deployed project root after building both engines.
# Preserve the current power/clock settings; do not overlap this with a build.
set -euo pipefail

report_dir="${JETSON_REPORT_DIR:-reports/jetson}"
mkdir -p "$report_dir" results/jetson
export PYTHONPATH=src
monitor_pid=""
cleanup_monitor() {
    if [[ -n "$monitor_pid" ]]; then
        kill "$monitor_pid" 2>/dev/null || true
        wait "$monitor_pid" 2>/dev/null || true
        monitor_pid=""
    fi
}
trap cleanup_monitor EXIT

for name in cpu trt_fp32 trt_fp16; do
    if [[ -e "$report_dir/${name}_benchmark.json" ]]; then
        echo "Existing results: choose a new JETSON_REPORT_DIR instead of overwriting $name" >&2
        exit 1
    fi
done

.venv-jetson-cpu/bin/python -m pytest -q > "$report_dir/tests_cpu_final.log" 2>&1
JETSON_TEST_ENGINE_DIR=models/weights/jetson .venv-jetson-trt/bin/python -m pytest -q > "$report_dir/tests_trt.log" 2>&1

for name in cpu trt_fp32 trt_fp16; do
    if [[ "$name" == cpu ]]; then
        interpreter=.venv-jetson-cpu/bin/python
        options=(--backend onnxruntime)
    else
        interpreter=.venv-jetson-trt/bin/python
        options=(--backend tensorrt --engine "models/weights/jetson/yolox_nano_${name#trt_}.engine")
    fi
    echo "Validating $name"
    tegrastats --interval 1000 > "$report_dir/tegrastats_${name}.log" 2>&1 &
    monitor_pid=$!
    "$interpreter" scripts/run_kitti.py "${options[@]}" --frames 3 --report "$report_dir/smoke_${name}.json" > "$report_dir/smoke_${name}.log" 2>&1
    "$interpreter" scripts/run_kitti.py "${options[@]}" --report "$report_dir/${name}_benchmark.json" > "$report_dir/${name}_benchmark.log" 2>&1
    "$interpreter" scripts/run_kitti.py "${options[@]}" --paced --rate-hz 10 --drop-late --report "$report_dir/${name}_10hz.json" > "$report_dir/${name}_10hz.log" 2>&1
    "$interpreter" scripts/run_kitti.py "${options[@]}" --mode evaluate --report "$report_dir/${name}_evaluation.json" > "$report_dir/${name}_evaluation.log" 2>&1
    cleanup_monitor
    tail -9 "$report_dir/${name}_benchmark.log"
done

if [[ ! -f results/jetson/cpu_reference/manifest.json ]]; then
    .venv-jetson-cpu/bin/python scripts/compare_jetson_backends.py --export-reference > "$report_dir/export_reference.log" 2>&1
fi
.venv-jetson-trt/bin/python scripts/compare_jetson_backends.py \
    --engine models/weights/jetson/yolox_nano_fp32.engine --candidate-report "$report_dir/trt_fp32_benchmark.json" \
    --engine models/weights/jetson/yolox_nano_fp16.engine --candidate-report "$report_dir/trt_fp16_benchmark.json" \
    --cpu-report "$report_dir/cpu_benchmark.json" --report "$report_dir/backend_comparison.json" > "$report_dir/backend_comparison.log" 2>&1

.venv-jetson-trt/bin/python scripts/run_kitti.py --backend tensorrt \
    --engine models/weights/jetson/yolox_nano_fp32.engine --mode evaluate \
    --video results/jetson/jetson_fp32_raw.avi --report "$report_dir/video_fp32.json" > "$report_dir/video_fp32.log" 2>&1
.venv-jetson-trt/bin/python scripts/encode_video.py results/jetson/jetson_fp32_raw.avi \
    results/jetson/jetson_fp32_demo.mp4 --metadata "$report_dir/video_fp32_metadata.json" > "$report_dir/video_fp32_encode.log" 2>&1
echo "Validation completed: $report_dir"
