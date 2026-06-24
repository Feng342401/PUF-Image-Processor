```bash
#!/bin/bash

set -e

echo "=== PUF Pipeline Starting ==="

echo "[1/4] Preprocessing images..."
python image_processor.py batch input output

echo "[2/4] Computing Hamming distance..."
python image_processor.py batchhamming output --output results/hamming.xlsx

echo "[3/4] Computing bit uniformity..."
python image_processor.py bituniformity output --output results/uniformity.xlsx

echo "[4/4] Visualization..."
python image_processor.py visualize example.png example2.png

echo "=== Pipeline Finished Successfully ==="
