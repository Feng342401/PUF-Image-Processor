#!/bin/bash

echo "Step 1: batch processing images..."
python image_processor.py batch input output

echo "Step 2: compute Hamming distance..."
python image_processor.py batchhamming output --output results.xlsx

echo "Step 3: compute bit uniformity..."
python image_processor.py bituniformity output --output uniformity.xlsx

echo "DONE: full pipeline completed."
