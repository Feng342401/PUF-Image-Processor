# PUF Image Processing and Statistical Analysis Toolkit

## Overview

This repository provides a fully reproducible computational pipeline for image-based Physically Unclonable Function (PUF) analysis. The framework is designed to evaluate randomness, uniqueness, and statistical stability of PUF-based imaging systems.

The toolkit enables automated processing of image datasets and supports full statistical characterization, including binarization, batch processing, Hamming distance computation, and bit uniformity analysis. It is intended for reproducible security evaluation of physical unclonable systems.

---

## Key Features

The system provides:

- Deterministic image preprocessing and binarization
- Batch processing of PUF image datasets
- Pairwise and batch Hamming distance computation
- Bit uniformity statistical evaluation
- Excel-based export of quantitative results
- Lightweight command-line interface for full pipeline execution

---

## Installation

Install required dependencies:

pip install -r requirements.txt

Alternatively, a conda environment is recommended:

conda create -n puf python=3.9
conda activate puf
pip install -r requirements.txt

---

## Usage

The software is operated via command-line interface using the main script:

PUF-Image-Processor.py

All operations are executed through simple terminal commands.

---

## Batch Processing

python PUF-Image-Processor.py batch input_images/ output_images/

This step processes all input images and generates binarized PUF representations for subsequent statistical analysis.

---

## Hamming Distance Analysis

python PUF-Image-Processor.py batchhamming output_images/ --output results.xlsx

This function computes pairwise Hamming distances between processed PUF samples and exports a full distance matrix in Excel format for statistical evaluation.

---

## Bit Uniformity Analysis

python PUF-Image-Processor.py bituniformity output_images/ --output uniformity.xlsx

This module calculates bit uniformity across the dataset, providing a quantitative measure of randomness quality in the PUF representation.

---

## Output Description

The pipeline generates the following outputs:

- Binarized PUF image dataset
- Hamming distance matrix (Excel format)
- Bit uniformity statistics (Excel format)

These outputs are designed for downstream statistical analysis in Python, MATLAB, or Excel.

---

## Data Organization

Recommended dataset structure:

input_images/
output_images/
results.xlsx
uniformity.xlsx

---

## Software Availability

The source code is publicly available in this GitHub repository.

---

## License

This project is released under the MIT License.
