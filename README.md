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
```bash
# Generate bitflip matrix from your DRAM profiling data
python create_bitflip_matrix.py --profile your_profile.json --output custom_bitflips.npy
```

## 📁 Repository Structure
```bash
ROBIN-Rowhammer-aware-Backdoor-Attack/
├── 📊 Software Simulation & Analysis
│   ├── hardware_aware_backdoor_8bit_mvm.py  # Core MVM-based attack implementation
│   ├── main_8bit_mvm.py                     # Main execution script for INT8 models
│   ├── analyze_memory_layout.py             # Model memory layout analysis
│   ├── create_bitflip_matrix.py             # DRAM bitflip matrix generation
│   ├── device1_1G.npy.zip                  # Sample bitflip matrix for simulation
│   ├── utils.py                             # Utility functions
│   ├── utils_sdn.py                         # SDN-specific utilities
│   ├── models/                              # DNN model definitions
│   │   ├── quan_resnet_cifar.py            # Quantized ResNet models
│   │   └── quantization.py                 # Quantization utilities
│   ├── networks/                            # Network architectures
│   └── blacksmith/                          # DRAM vulnerability profiling tool
│
├── ⚡ Hardware Attack Implementation
│   └── hardware_attack/
│       ├── targeted_map.c                   # Memory mapping for DNN→DRAM pages
│       ├── rowhammer_attack.cpp             # RowHammer pattern execution
│       ├── backdoor_test.cpp                # Backdoor effectiveness verification
│       ├── run_attack.cpp                   # Complete attack orchestration
│       ├── resnet20_quan.h                  # ResNet20 model header
│       ├── resnet20_quan.cpp                # ResNet20 model implementation
│       ├── patterns.json                    # RowHammer patterns with UUID IDs
│       ├── ResNet20_FL32.bin               # Target model binary
│       ├── device2_1G_metadata.json        # DRAM topology metadata
│       ├── resnet20_int8_device1/          # Attack data and triggers
│       │   ├── attack_report_int8.txt      # Vulnerable page mappings
│       │   ├── trigger_pattern.npy         # Backdoor trigger pattern
│       │   └── attack_config.json          # Attack configuration
│       └── Makefile                         # Build configuration
│
└── 📄 Documentation
    └── README.md                            # This file
```


## 📊 Expected Results

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
