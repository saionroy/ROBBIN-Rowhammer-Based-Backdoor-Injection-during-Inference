# ROBIN: ROwhammer-aware Backdoor Attack during INference

A hardware-aware inference-time backdoor attack framework that exploits Rowhammer vulnerabilities to inject backdoors into deployed Deep Neural Networks (DNNs) without modifying the model files.

## 📋 Overview

ROBIN introduces the first systematic hardware-aware backdoor attack that bridges the gap between algorithmic backdoor requirements and physical DRAM constraints. Unlike previous approaches that assume arbitrary bit-flip capabilities, our method profiles actual DRAM vulnerabilities and uses direction-aware Matrix-Vector Multiplication (MVM) to optimize bit-flip placement for guaranteed attack feasibility on real hardware.

### Key Features
- **Hardware-Aware Design**: Profiles actual DRAM vulnerability patterns instead of assuming arbitrary bit-flips
- **Direction-Aware MVM**: Accounts for flip directionality (0→1 vs 1→0) constraints in DRAM
- **Systematic Page Matching**: Efficiently maps DNN pages to vulnerable DRAM pages using importance scoring
- **Adaptive Thresholds**: Dynamically adjusts accuracy constraints based on attack progress
- **Multi-Quantization Support**: Works with FP32, INT8-in-FP32, and native INT8 models

## 🏗️ Architecture

The attack consists of three main stages:

1. **DRAM Vulnerability Profiling**: Maps fault distribution across memory pages
2. **Hardware-Aware Backdoor Construction**: Identifies critical bits and matches DNN pages to DRAM vulnerabilities  
3. **Attack Deployment**: Executes Rowhammer to inject backdoors during inference

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8 or higher
python --version

# Install required packages
pip install torch>=1.9.0 torchvision numpy matplotlib

ROBIN/
├── hardware_aware_backdoor_8bit_mvm.py  # Core MVM-based attack implementation
├── main_8bit_mvm.py                     # Main execution script for INT8 models
├── models/
│   ├── quan_resnet_cifar.py            # Quantized ResNet models
│   └── quantization.py                  # Quantization utilities
├── ProfilingData/
│   ├── bitflip_matrix_A.npy            # DRAM vulnerability profiles
│   └── resnet20_int8/
│       ├── resnet20_int8.pth.tar       # Pre-trained INT8 checkpoint
│       └── ResNet20_Q3.txt             # Page allocation info
├── AttackResults/                       # Output directory for results
├── requirements.txt                     # Python dependencies
└── README.md                           # This file
