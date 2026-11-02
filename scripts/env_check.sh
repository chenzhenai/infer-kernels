#!/usr/bin/env bash
# 打印驱动、CUDA Toolkit、GPU、Compute Capability、PyTorch 的 CUDA 版本。
set -u

fail=0

map_arch() {
  local cc="$1"
  local major="${cc%%.*}"
  local minor="${cc#*.}"
  if [[ "$major" == "8" ]]; then
    echo 80
  elif [[ "$major" == "9" ]]; then
    echo 90a
  elif [[ "$major" == "10" && "$minor" -ge 3 ]]; then
    echo 103a
  elif [[ "$major" == "10" ]]; then
    echo 100a
  else
    echo "unsupported CC ${cc}"
  fi
}

echo "== Driver / GPU =="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv
  echo "-- CMAKE_CUDA_ARCHITECTURES --"
  while IFS= read -r cc; do
    [[ -z "$cc" ]] && continue
    printf "%s -> %s\n" "$cc" "$(map_arch "$cc")"
  done < <(nvidia-smi --query-gpu=compute_cap --format=csv,noheader)
else
  echo "nvidia-smi not found"
  fail=1
fi

echo "== CUDA Toolkit =="
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version
else
  echo "nvcc not found"
  fail=1
fi

if command -v compute-sanitizer >/dev/null 2>&1; then
  echo "compute-sanitizer: $(command -v compute-sanitizer)"
  echo "OOB check: compute-sanitizer --tool memcheck ./build/oob_probe"
else
  echo "compute-sanitizer not found"
  fail=1
fi

echo "== PyTorch =="
if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found"
  fail=1
else
  python3 - <<'PY' || fail=1
import sys
try:
    import torch
except Exception as exc:
    print(f"torch import failed: {exc}")
    sys.exit(1)
print(f"torch {torch.__version__}")
print(f"torch.version.cuda {torch.version.cuda}")
print(f"cuda available {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device {torch.cuda.get_device_name(0)}")
    print(f"capability {torch.cuda.get_device_capability(0)}")
PY
fi

exit "$fail"
