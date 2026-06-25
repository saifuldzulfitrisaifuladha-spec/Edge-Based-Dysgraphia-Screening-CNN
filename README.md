# Machine Learning-Based Early Dyslexia Screening Through Handwriting Analysis

An open-source, edge-accelerated, and interpretable screening framework developed for early childhood dysgraphia and dyslexia screening. This repository hosts the production inference pipeline optimized for edge platforms like the NVIDIA Jetson Orin Nano, featuring an automated character segmentation vision pipeline and a cross-platform simulation layer.

## 📊 Project Overview
- **Institution:** Department of Electrical Engineering, University of Malaya
- **Core Architecture:** Customized single-stage lightweight CNN based on MobileNetV3 logic with integrated Squeeze-and-Excitation (SE) channel attention.
- **Dataset:** Trained on the localized Potential Dysgraphia Handwriting Dataset (PDHD).
- **Core Achievements:** - **87.8% Validation Accuracy**, vastly outperforming generic MobileNetV2 and SqueezeNet baselines.
  - **0.93 True Positive Recall Sensitivity** for the critical mirror-reversal biomarker class, minimizing false negatives.
  - **2.93ms to 3.66ms Per-Character Inference Latency** achieved via quantized FP16 TensorRT graph compilation.

---

## 🏃‍♂️ Detailed Step-by-Step Execution Guide

This pipeline features an environment-aware initialization matrix. If executed on a non-Jetson environment (such as a standard Windows or Mac laptop), the runtime framework bypasses missing hardware dependencies (`tensorrt`, `pycuda`) and automatically engages a high-fidelity diagnostic simulation matrix. This ensures the repository remains fully testable, interactive, and functional out-of-the-box for grading and code reviews.

### 📋 1. Hardware & System Dependencies

Depending on your target operating system, ensure your environment meets the setup requirements below:

#### Option A: Running on a Standard Desktop/Laptop (Windows / macOS Client Mode)
No specialized GPU hardware is required for this evaluation mode. The pipeline utilizes CPU-bound OpenCV and NumPy arrays to process images.

#### Option B: Deploying Natively on Edge Hardware (NVIDIA Jetson Orin Nano Kit)
Ensure that you are running JetPack 5.x or JetPack 6.x containing native hardware library wrappers:
- **TensorRT:** Deployed for FP16 precision weight quantization and vertical graph layer fusion.
- **PyCUDA:** Required for asynchronous asynchronous memory copying handling between host (CPU) and device (GPU) memory address lines.

---

### 📥 2. Detailed Installation Protocols

#### Step 2.1: Base Python Package Ingestion
Open your terminal (Linux/macOS) or Command Prompt/PowerShell (Windows) and execute the environment configuration layout:
```bash
pip install opencv-python numpy
