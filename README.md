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

### 1. Software Simulation (Algorithm Development)
- **Purpose**: Algorithm research, attack feasibility analysis, parameter optimization
- **Location**: Root directory (Python implementation)
- **Output**: Attack parameters, vulnerable page mappings, trigger patterns

### 2. Hardware Attack (Real Execution)
- **Purpose**: Actual RowHammer execution on target hardware
- **Location**: `hardware_attack/` directory (C/C++ implementation)
- **Input**: Parameters from software simulation
- **Output**: Real bit-flips and backdoor injection

## 🚀 Quick Start

### Software Simulation

```bash
# Prerequisites: Python 3.8+, PyTorch, NumPy
pip install torch>=1.9.0 torchvision numpy matplotlib

# Step 1: Analyze model memory layout
python analyze_memory_layout.py --model final_models/resnet20_int8_state.pth --output model_pagemap.txt

# Step 2: Create bitflip matrix from DRAM profiling
python create_bitflip_matrix.py --profile device1_1G.json --output device1_1G.npy

# Step 3: Run software simulation
python main_8bit_mvm.py

# Step 4: Hardware-aware backdoor construction
python hardware_aware_backdoor_8bit_mvm.py
```

### Hardware Attack Execution

```bash
# Navigate to hardware attack directory
cd hardware_attack/

# Build the attack components
make

# Execute complete hardware attack (requires root)
sudo ./run_attack

```

## 🔧 Analysis Tools

### Model Memory Layout Analysis
```bash
# Analyze PyTorch model memory layout and generate page mapping
python analyze_memory_layout.py --model final_models/resnet20_int8_state.pth --output model_pagemap.txt

# Options:
#   --model: Path to PyTorch .pth model file
#   --page-size: Memory page size in bytes (default: 4096)
#   --output: Output file path (default: stdout)
```

**Output**: Detailed memory layout showing:
- Parameter names, shapes, and byte sizes
- Virtual memory page allocation (4KB pages)
- Byte-level addressing for each parameter
- Total memory usage and page count

### DRAM Bitflip Matrix Generation
```bash
# Create bitflip vulnerability matrix from DRAM profiling results
python create_bitflip_matrix.py --profile device1_1G.json --output device1_1G.npy

# Options:
#   --profile: Path to DRAM profiling JSON file (default: device1_1G.json)
#   --output: Output path for bitflip matrix (default: device1_1G.npy)  
#   --metadata: Output path for metadata JSON (default: <output>_metadata.json)
```

**Output**: Numpy matrix where:
- Rows = Physical DRAM pages (sorted by vulnerability)
- Columns = 32,768 bits per page (4KB × 8 bits/byte)
- Values: `-1` (not flippable), `0` (flip 1→0), `1` (flip 0→1)

## 📁 Repository Structure

```
ROBIN/
├── 📊 Software Simulation (Root Directory)
│   ├── hardware_aware_backdoor_8bit_mvm.py  # Core MVM-based attack implementation
│   ├── main_8bit_mvm.py                     # Main execution script for INT8 models
│   ├── analyze_memory_layout.py             # Model memory layout analysis
│   ├── create_bitflip_matrix.py             # DRAM bitflip matrix generation
│   ├── utils.py                             # Utility functions
│   ├── utils_sdn.py                         # SDN-specific utilities
│   ├── models/                              # DNN model definitions
│   │   ├── quan_resnet_cifar.py            # Quantized ResNet models
│   │   └── quantization.py                 # Quantization utilities
│   └── networks/                            # Network architectures
│
├── ⚡ Hardware Attack Implementation
│   └── hardware_attack/
│       ├── targeted_map.c                   # Memory mapping for DNN→DRAM pages
│       ├── rowhammer_attack.cpp             # RowHammer pattern execution
│       ├── backdoor_test.cpp                # Backdoor effectiveness verification
│       ├── run_attack.cpp                   # Complete attack orchestration
│       ├── resnet20_quan.h/cpp              # ResNet20 model implementation
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
- **Page Mapping Coverage**: Percentage of critical pages successfully mapped

### Hardware Execution  
- **Bit-Flip Success**: Device-dependent (varies by DRAM module)
- **Memory Mapping Accuracy**: Success rate of DNN→DRAM page placement
- **End-to-End ASR**: Actual backdoor trigger effectiveness

## ⚠️ Important Notes

### Device Dependency
**The success of hardware RowHammer attacks is heavily dependent on:**
- Specific DRAM module characteristics
- Memory controller implementation
- System configuration and timing
- Environmental factors (temperature, voltage)

### Prerequisites & Input Files
**Before running the analysis tools and attack pipeline:**

#### Required Input Files:
1. **Model File**: PyTorch model checkpoint (`.pth` format)
   - Example: `final_models/resnet20_int8_state.pth`
   - Contains trained model parameters and state

2. **DRAM Profiling Data**: Vulnerability assessment results (`.json` format)
   - Example: `device1_1G.json`
   - Generated using tools like Blacksmith or TRRespass
   - Contains physical page numbers, bit positions, and flip types

#### DRAM Profiling Requirement:
**Before attempting hardware execution, you MUST:**
1. Profile your specific DRAM modules for vulnerabilities
2. Identify exploitable memory locations and patterns
3. Validate attack parameters on your target system
4. Adapt pattern configurations to your hardware

### Research Use Only
This framework is intended for **research purposes only**. RowHammer attacks can cause system instability and should only be executed in controlled environments with appropriate safeguards.
