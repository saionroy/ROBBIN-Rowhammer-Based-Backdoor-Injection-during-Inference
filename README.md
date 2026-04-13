# ROBBIN: Rowhammer-Based Backdoor Injection during Inference

A comprehensive hardware-aware inference-time backdoor attack framework that exploits RowHammer vulnerabilities to inject backdoors into deployed Deep Neural Networks (DNNs) without modifying the model files.


**This repository contains the code corresponding to paper #333 under submission at ICCAD 2026.**




## 📋 Overview

This work presents ROBBIN, a Rowhammer-based inference-time backdoor attack that accounts for hammering-induced bit flips. It uses deterministic vulnerability profiling to find stable, reproducible flip patterns at higher hammering intensities that overcome on-die error correction code (OECC). Using this profiling, ROBBIN iteratively selects the minimum number of DRAM pages to hammer, thereby improving the backdoor effect while accounting for all flips. Across three DDR4 chips, ROBBIN attains a close to 90% attack success rate (ASR) on triggered inputs while preserving >83% test accuracy (TA) on benign inputs, evaluated on ResNet-20 with CIFAR-10, for both FP32 and INT8 data types. ROBBIN is the first practical backdoor attack that leverages device-specific vulnerabilities and achieves consistent attack efficacy across multiple DRAMs.

### Key Features
- **Hardware-Aware Design**: Profiles actual DRAM vulnerability patterns instead of assuming arbitrary bit-flips
- **Direction-Aware MVM**: Accounts for flip directionality (0→1 vs 1→0) constraints in DRAM
- **Systematic Page Matching**: Efficiently maps DNN pages to vulnerable DRAM pages using importance scoring
- **Adaptive Thresholds**: Dynamically adjusts accuracy constraints based on attack progress
- **Multi-Quantization Support**: Works with FP32, and native INT8 models



## 📋 Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 18.04+ recommended)
- **Python**: 3.8+
- **Hardware**: x86_64 system with DDR3/DDR4 DRAM

### Dependencies
```bash
# Install Python dependencies
pip install torch>=1.9.0 torchvision numpy matplotlib
```

### Required Input Files
1. **Model File**: PyTorch model checkpoint (`.pth` format)


2. **DRAM Vulnerability Data** (choose one):
   - **Option A**: Use provided sample data: `device1_1G.npy.zip` (1GB DRAM profile)
   - **Option B**: Generate from your DRAM profiling using Blacksmith tool

3. **Test Data** (for backdoor evaluation):
   - **Real CIFAR-10 test data**: **Required** for backdoor testing (place in `hardware_attack/data/cifar10_test.bin`)


### DRAM Profiling
For custom hardware attacks, profile your DRAM using the included Blacksmith tool:
- Follow Blacksmith compilation and execution instructions
- Generate vulnerability matrix for your specific hardware
- Required for reliable hardware attack execution

### Rowhammer Profiling Configuration (4 Aggressor Rows)
The DAC submission and the provided scripts now share the same profiling dataset:
- The bitflip matrix `profile_results/device1_256MB_4row.npy` (zipped as `profile_results/device1_256MB_4row.npy.zip`) captures the flips observed when hammering **four aggressor rows** per victim page. Both `main_32bit_mvm.py` and `main_8bit_mvm.py` load this matrix by default, so their FP32 and INT8 evaluations match the reported results.
- **Profiling settings**:
  - Aggressors: 4 rows (two on each side of each victim row).
  - Hammering intensity: 500 total activations per pattern.
  - Device: 256 MB region of the DDR4.
  - [Download the 4-row hammering results](https://drive.google.com/drive/folders/113tWaQPlbuyK6h5fFslvnyCXRkXAqYg8?usp=sharing) (Device1–3 .json files) to reproduce the dataset used for every attack run in the repository.

Use `create_bitflip_matrix.py --profile your_profile.json --output custom_bitflips.npy` if you need to regenerate the matrix from a new Blacksmith profiling run while keeping the same 4-row settings.

## 🚀 Quick Start

Before running the attack scripts, recreate the same `pagemap` configuration that the DAC submission refers to by running the memory layout analyzer:
```bash
python analyze_memory_layout.py --model your_model.pth --output model_pagemap.txt
```

### 1. Software Simulation
```bash
# Step 1: Prepare bitflip matrix (use sample data)
unzip device1_256MB_4row.npy.zip

# Step 2: Run software simulation
python main_8bit_mvm.py
python main_32bit_mvm.py

```

## 🔧 Advanced Usage

### Custom Model Analysis
This mirrors the `pagemap` command referenced in the DAC paper, so your generated pagemap matches the same DNN-to-DRAM mapping used throughout the evaluation.
```bash
# Analyze your own model's memory layout
python analyze_memory_layout.py --model your_model.pth --output model_pagemap.txt
```

### Custom DRAM Profiling
**Sample profiling results** can be found [here](https://drive.google.com/drive/folders/113tWaQPlbuyK6h5fFslvnyCXRkXAqYg8?usp=sharing), which can be used for generating the matrix with different devices.
```bash
# Generate bitflip matrix from your DRAM profiling data
python create_bitflip_matrix.py --profile your_profile.json --output custom_bitflips.npy
```

## 📁 Repository Structure
```bash
ROBBIN-Rowhammer-aware-Backdoor-Attack/
├── Core Attack Scripts
│   ├── main_8bit_mvm.py                     # Main execution script for INT8 models
│   ├── hardware_aware_backdoor_8bit_mvm.py  # Core MVM-based attack implementation
│   ├── analyze_memory_layout.py             # Model memory layout analysis
│   ├── create_bitflip_matrix.py             # DRAM bitflip matrix generation
│   ├── utils.py                             # Utility functions
│   └── utils_sdn.py                         # SDN-specific utilities
│
├── Profile Data
│   └── profile_results/
│       └── device1_256MB_4row.npy.zip       # 4-row Rowhammer profiling results that match the paper
│
├── Model Definitions
│   ├── models/                              # DNN model implementations
│   │   ├── quan_resnet_cifar.py            # Quantized ResNet for CIFAR
│   │   ├── quantization.py                 # Quantization utilities
│   │   ├── binarization.py                 # Binary neural networks
│   │   └── vanilla_models/                 # Standard model implementations
│   └── networks/                            # Network architectures
│       ├── CNNs/                           # Standard CNN architectures
│       │   └── ResNet.py
│
├── Blacksmith
│
└── Documentation
    └── README.md                            # This file
```

## 📊 Expected Results

### Software Simulation
- **Attack Success Rate**: Depends on DRAM profile and model architecture
- **Clean Accuracy**: Maintained within acceptable degradation

Sample outputs for both FP32 and INT8 attacks are stored in `result/` (e.g., `attack_report_fp32.txt` and `attack_report_int8.txt`) so you can compare your own runs to the published results.

## ⚠️ Important Notes

### Hardware Dependency
RowHammer attack effectiveness depends on specific DRAM hardware:
- **DRAM module brand**
- **Production year**
- **DRAM generation**

### Research Use Only
This framework is for **research purposes only**. Execute RowHammer attacks only in controlled environments with appropriate safeguards.
