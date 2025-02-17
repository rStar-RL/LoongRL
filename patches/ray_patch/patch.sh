echo "Patching Ray for ROCm, this patch is based on Ray 2.38.0"

pip install ray==2.38.0

RAY_DIR=$(python -c "import ray; print(ray.__path__[0])")
TARGET_FILE="$RAY_DIR/_private/accelerators/amd_gpu.py"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NEW_FILE="${SCRIPT_DIR}/amd_gpu.py"
cp "$NEW_FILE" "$TARGET_FILE"
