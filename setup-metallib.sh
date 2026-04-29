#!/bin/bash
# Copy the Metal shader library from Python's MLX installation.
# The metallib is compiled from Metal sources and not available via SPM.

TARGET="./default.metallib"

if [ -f "$TARGET" ]; then
    echo "default.metallib already exists, skipping."
    exit 0
fi

# Search for MLX metallib in common locations
SOURCES=(
    "/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/mlx/lib/mlx.metallib"
    "/Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/mlx/lib/mlx.metallib"
    "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/mlx/lib/mlx.metallib"
)

# Also try python3 to find the path dynamically
PY_PATH=$(python3 -c "import mlx; import os; p=os.path.dirname(mlx.__file__); print(p+'/lib/mlx.metallib')" 2>/dev/null)
if [ -n "$PY_PATH" ]; then
    SOURCES=("$PY_PATH" "${SOURCES[@]}")
fi

for src in "${SOURCES[@]}"; do
    if [ -f "$src" ]; then
        echo "Found metallib at: $src"
        cp "$src" "$TARGET"
        echo "Copied to: $TARGET"
        exit 0
    fi
done

echo "ERROR: Could not find MLX metallib in any known location."
echo "Install MLX via pip: pip3 install mlx"
echo "Then run this script again."
exit 1
