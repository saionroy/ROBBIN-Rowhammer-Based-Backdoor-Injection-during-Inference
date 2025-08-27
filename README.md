# ROBIN: Rowhammer-aware Backdoor Attack Framework

A comprehensive hardware-aware inference-time backdoor attack framework that exploits RowHammer vulnerabilities to inject backdoors into deployed Deep Neural Networks (DNNs) without modifying the model files.

## 📋 Overview

ROBIN introduces the first systematic hardware-aware backdoor attack that bridges the gap between algorithmic backdoor requirements and physical DRAM constraints. Unlike previous approaches that assume arbitrary bit-flip capabilities, our method profiles actual DRAM vulnerabilities and uses direction-aware Matrix-Vector Multiplication (MVM) to optimize bit-flip placement for guaranteed attack feasibility on real hardware.

### Key Features
- **Hardware-Aware Design**: Profiles actual DRAM vulnerability patterns instead of assuming arbitrary bit-flips
- **Direction-Aware MVM**: Accounts for flip directionality (0→1 vs 1→0) constraints in DRAM
- **Systematic Page Matching**: Efficiently maps DNN pages to vulnerable DRAM pages using importance scoring
- **Adaptive Thresholds**: Dynamically adjusts accuracy constraints based on attack progress
- **Multi-Quantization Support**: Works with FP32, INT8-in-FP32, and native INT8 models
- **End-to-End Implementation**: Complete pipeline from software simulation to real hardware execution

## 🏗️ Framework Architecture

The ROBIN framework consists of two complementary components:

### 1. Software Simulation
- **Purpose**: Algorithm research, attack feasibility analysis, parameter optimization
- **Location**: Root directory (Python implementation)
- **Output**: Attack parameters, vulnerable page mappings, trigger patterns

### 2. Hardware Attack
- **Purpose**: Actual RowHammer execution on target hardware
- **Location**: `hardware_attack/` directory (C/C++ implementation)
- **Input**: Parameters from software simulation
- **Output**: Real bit-flips and backdoor injection

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
   - Example: `final_models/resnet20_int8_state.pth`

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


## 🚀 Quick Start

### 1. Software Simulation
```bash
# Step 1: Prepare bitflip matrix (use sample data)
unzip device1_1G.npy.zip

# Step 2: Run software simulation
python main_8bit_mvm.py

```

### 2. Hardware Attack Execution
```bash
# Navigate to hardware directory
cd hardware_attack/

# Build and execute (requires root)
make
sudo ./run_attack
```

## 🔧 Advanced Usage

### Custom Model Analysis
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
ROBIN-Rowhammer-aware-Backdoor-Attack/
├── Core Attack Scripts
│   ├── main_8bit_mvm.py                     # Main execution script for INT8 models
│   ├── hardware_aware_backdoor_8bit_mvm.py  # Core MVM-based attack implementation
│   ├── analyze_memory_layout.py             # Model memory layout analysis
│   ├── create_bitflip_matrix.py             # DRAM bitflip matrix generation
│   ├── device1_1G.npy.zip                  # Sample bitflip matrix for simulation
│   ├── utils.py                             # Utility functions
│   └── utils_sdn.py                         # SDN-specific utilities
│
├── Model Definitions
│   ├── models/                              # DNN model implementations
│   │   ├── quan_resnet_cifar.py            # Quantized ResNet for CIFAR
│   │   ├── quan_resnet_imagenet.py         # Quantized ResNet for ImageNet
│   │   ├── quan_vgg_cifar.py               # Quantized VGG models
│   │   ├── quan_mobilenet_imagenet.py      # Quantized MobileNet
│   │   ├── quantization.py                 # Quantization utilities
│   │   ├── binarization.py                 # Binary neural networks
│   │   └── vanilla_models/                 # Standard model implementations
│   └── networks/                            # Network architectures
│       ├── CNNs/                           # Standard CNN architectures
│       │   ├── ResNet.py
│       │   ├── VGG.py
│       │   └── MobileNet.py
│       └── SDNs/                           # Self-Destructing Networks
│           ├── ResNet_SDN.py
│           ├── VGG_SDN.py
│           └── MobileNet_SDN.py
│
├── Hardware Attack Implementation
│   ├── hardware_attack/
│   │   ├── targeted_map.c                   # Memory mapping for DNN→DRAM pages
│   │   ├── rowhammer_attack.cpp             # RowHammer pattern execution
│   │   ├── backdoor_test.cpp                # Backdoor effectiveness verification
│   │   ├── run_attack.cpp                   # Complete attack orchestration
│   │   ├── resnet20_quan.h                  # ResNet20 model header
│   │   ├── resnet20_quan.cpp                # ResNet20 model implementation
│   │   ├── resnet20_int8_device1/          # Attack results and triggers
│   │   │   ├── attack_report_int8.txt      # Vulnerable page mappings
│   │   │   ├── trigger_pattern.npy         # Backdoor trigger pattern
│   │   │   ├── trigger_pattern.png         # Trigger visualization
│   │   │   ├── trigger_pattern.pth         # PyTorch trigger format
│   │   │   ├── attack_config.json          # Attack configuration
│   │   │   └── mvm_attack_visualization.png # Attack visualization
│   │   └── Makefile                         # Build configuration
│   └── blacksmith/                          # DRAM vulnerability profiling tool
│
└── Documentation
    └── README.md                            # This file
```


## 📊 Expected Results

### Sample Output
A complete sample output of the end-to-end hardware attack execution is provided in `hardware_attack/sample_output.txt`. This demonstrates:
- Memory mapping results (9/9 target pages found)
- RowHammer attack execution with real-time bit flip detection
- Final results: 112 total bit flips, 82.97% clean accuracy, 90.30% ASR

### Software Simulation
- **Attack Success Rate**: Depends on DRAM profile and model architecture
- **Clean Accuracy**: Maintained within acceptable degradation

### Hardware Execution  
- **Bit-Flip Success**: Device-dependent (varies by DRAM module)
- **Memory Mapping Accuracy**: Success rate of DNN→DRAM page placement
- **End-to-End Accuracy and ASR**: Test the effectiveness of backdoor

## ⚠️ Important Notes

### Hardware Dependency
RowHammer attack effectiveness depends on specific DRAM hardware:
- **DRAM module brand**
- **Production year**
- **DRAM generation**

### Research Use Only
This framework is for **research purposes only**. Execute RowHammer attacks only in controlled environments with appropriate safeguards.
