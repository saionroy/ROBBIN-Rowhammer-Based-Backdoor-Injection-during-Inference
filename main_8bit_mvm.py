#!/usr/bin/env python3
"""
main_8bit_mvm.py
================
Main script to execute systematic MVM-based hardware-aware backdoor attack on native INT8 ResNet20
Implements direction-aware matrix-vector multiplication for optimal DRAM page selection
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from dataclasses import dataclass

# Add path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import models
import models
from models.quan_resnet_cifar import resnet20_quan
from models.quantization import quan_Conv2d, quan_Linear

# Import the MVM-based hardware-aware attack module
from hardware_aware_backdoor_8bit_mvm import MVMHardwareAwareAttack, INT8AttackConfig

def dequantize_int8_weights(int8_state_dict):
    """
    Convert INT8 weights back to FP32 using scale factors
    Uses the exact approach from the working code
    
    Key insight: fp32_weight = int8_weight.float() * scale_factor
    """
    dequantized_state_dict = {}
    dequantized_count = 0
    scale_factors = {}
    
    for key, value in int8_state_dict.items():
        if key.endswith('.weight') and value.dtype == torch.int8:
            # Dequantize INT8 weight
            scale_key = key.replace('.weight', '.scale_factor')
            if scale_key in int8_state_dict:
                scale_factor = int8_state_dict[scale_key]
                dequantized_weight = value.float() * scale_factor
                dequantized_state_dict[key] = dequantized_weight
                scale_factors[key] = scale_factor.item()
                dequantized_count += 1
            else:
                print(f"Warning: No scale factor found for {key}")
                dequantized_state_dict[key] = value
                scale_factors[key] = 1.0
        elif 'scale_factor' in key:
            # Skip scale factors - only used for dequantization
            continue
        else:
            # Keep other parameters (bias, batch norm, etc.) as-is
            dequantized_state_dict[key] = value
    
    print(f"Dequantized {dequantized_count} INT8 weight layers")
    return dequantized_state_dict, scale_factors

def verify_int8_checkpoint(int8_state_dict):
    """Verify the checkpoint contains actual INT8 weights"""
    int8_params = []
    fp32_params = []
    scale_factors = []
    
    for name, param in int8_state_dict.items():
        if name.endswith('.weight'):
            if param.dtype == torch.int8:
                int8_params.append(name)
                # Check value range
                min_val = param.min().item()
                max_val = param.max().item()
                if min_val < -128 or max_val > 127:
                    print(f"WARNING: {name} has out-of-range INT8 values: [{min_val}, {max_val}]")
            else:
                fp32_params.append(name)
        elif name.endswith('.scale_factor'):
            scale_factors.append(name)
    
    print(f"\nCheckpoint verification:")
    print(f"  INT8 weight parameters: {len(int8_params)}")
    print(f"  FP32 weight parameters: {len(fp32_params)}")
    print(f"  Scale factors: {len(scale_factors)}")
    
    if len(int8_params) == 0:
        raise ValueError("No INT8 parameters found in checkpoint!")
    
    # Sample some INT8 weights to verify
    if int8_params:
        sample_param = int8_params[0]
        sample_weights = int8_state_dict[sample_param].view(-1)[:10]
        print(f"\nSample INT8 values from {sample_param}:")
        print(f"  {sample_weights.tolist()}")
    
    return int8_params, fp32_params

def load_int8_checkpoint(checkpoint_path):
    """
    Load and analyze INT8 checkpoint using the working technique
    """
    print(f"\nLoading INT8 checkpoint from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    int8_checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Handle different checkpoint formats
    if 'state_dict' in int8_checkpoint:
        int8_state_dict = int8_checkpoint['state_dict']
        print(f"Checkpoint info:")
        print(f"  Epoch: {int8_checkpoint.get('epoch', 'N/A')}")
        print(f"  Architecture: {int8_checkpoint.get('arch', 'N/A')}")
    else:
        int8_state_dict = int8_checkpoint
    
    print(f"Loaded checkpoint with {len(int8_state_dict)} parameters")
    
    # Verify INT8 content
    verify_int8_checkpoint(int8_state_dict)
    
    return int8_state_dict

def create_model_from_int8_checkpoint(int8_state_dict):
    """
    Create model and load dequantized weights using the working technique
    """
    print("\nCreating ResNet20 model and loading INT8 weights...")
    
    # Dequantize weights
    dequantized_state_dict, scale_factors = dequantize_int8_weights(int8_state_dict)
    
    # Create model
    model = resnet20_quan(num_classes=10)
    model_dict = model.state_dict()
    
    # Filter and load pretrained weights
    pretrained_dict = {k: v for k, v in dequantized_state_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)  # strict=False for missing step_size params
    
    # Reset quantization parameters
    for m in model.modules():
        if hasattr(m, '__reset_stepsize__'):
            m.__reset_stepsize__()
        if hasattr(m, '__reset_weight__'):
            m.__reset_weight__()
    
    print(f"Loaded {len(pretrained_dict)} parameters into model")
    
    return model, scale_factors

def create_data_loaders(data_path, batch_size=128):
    """
    Create CIFAR-10 data loaders with correct normalization
    """
    print("\nCreating CIFAR-10 data loaders...")
    
    # CIFAR-10 normalization - using the exact values from working code
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    testset = torchvision.datasets.CIFAR10(
        root=data_path,
        train=False,
        download=True,
        transform=transform_test
    )
    
    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    # Small subset for attack optimization
    attack_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=32,
        shuffle=True,
        num_workers=2
    )
    
    return test_loader, attack_loader

def test_model_accuracy(model, test_loader, model_name="ResNet20 INT8", num_batches=50):
    """Test model accuracy on CIFAR-10"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    
    print(f"Testing {model_name} on {num_batches} batches...")
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # Handle multi-branch models (return final output)
            if isinstance(output, list):
                output = output[-1]
            
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 10 == 0 and batch_idx > 0:
                current_acc = 100. * correct / total
                print(f"  Batch {batch_idx}/{num_batches}: {current_acc:.2f}%")
            
            if batch_idx >= num_batches - 1:
                break
    
    accuracy = 100. * correct / total
    print(f"🎯 {model_name} Accuracy: {accuracy:.2f}%")
    return accuracy

def test_bit_manipulation():
    """Test bit manipulation functions"""
    from hardware_aware_backdoor_8bit_mvm import INT8BitManipulator
    
    print("\n" + "="*60)
    print("Testing Bit Manipulation")
    print("="*60)
    
    manipulator = INT8BitManipulator()
    
    # Test various INT8 values
    test_values = [-128, -64, -1, 0, 1, 64, 127]
    
    for val in test_values:
        bits = manipulator.int8_to_bits(val)
        reconstructed = manipulator.bits_to_int8(bits)
        bits_str = ''.join(map(str, bits[::-1]))  # Show in MSB-LSB order for readability
        print(f"Value: {val:4d} -> Bits: {bits_str} -> Reconstructed: {reconstructed:4d}")
        assert val == reconstructed, f"Bit manipulation failed for {val}!"
    
    # Test bit flips
    print("\nTesting bit flips on value 42:")
    val = 42
    bits = manipulator.int8_to_bits(val)
    print(f"Original: {val} -> {''.join(map(str, bits[::-1]))}")
    
    for bit_pos in range(8):
        flipped_bits = bits.copy()
        flipped_bits[bit_pos] = 1 - flipped_bits[bit_pos]
        flipped_val = manipulator.bits_to_int8(flipped_bits)
        impact = flipped_val - val
        print(f"  Flip bit {bit_pos}: {val} -> {flipped_val} (change: {impact:+4d})")
    
    print("\n✓ Bit manipulation tests passed!")

def save_results(results, result_dir, timestamp):
    """
    Save attack results in multiple formats
    """
    # Save JSON results
    results_file = os.path.join(result_dir, f'mvm_attack_results_int8_{timestamp}.json')
    
    # Convert numpy/torch types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, (np.ndarray, torch.Tensor)):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        else:
            return obj
    
    results_serializable = convert_to_serializable(results)
    
    with open(results_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    # Generate detailed report
    report_file = os.path.join(result_dir, f'mvm_attack_report_int8_{timestamp}.txt')
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MVM-Based Hardware-Aware Backdoor Attack Report for Native INT8\n")
        f.write("Direction-Aware Matrix-Vector Multiplication Approach\n")
        f.write("="*80 + "\n\n")
        
        f.write("CONFIGURATION:\n")
        f.write("-"*40 + "\n")
        f.write(f"Model: ResNet20 INT8 (resnet20_quan)\n")
        f.write(f"Target Class: {results.get('target_class', 2)}\n")
        f.write(f"Target ASR: {results.get('target_asr', 90.0)}%\n")
        f.write(f"Alpha_s8 (sign bit factor): {results.get('alpha_s8', 0.5)}\n")
        f.write(f"Top DRAM pages per DNN: {results.get('top_dram_pages_per_dnn', 20)}\n\n")
        
        f.write("RESULTS:\n")
        f.write("-"*40 + "\n")
        f.write(f"Attack Successful: {results['attack_successful']}\n")
        f.write(f"Initial Accuracy: {results['initial_accuracy']:.2f}%\n")
        f.write(f"Final Accuracy: {results['final_accuracy']:.2f}%\n")
        f.write(f"Trigger-only ASR: {results['trigger_asr']:.2f}%\n")
        f.write(f"Final ASR: {results['final_asr']:.2f}%\n")
        f.write(f"Total Time: {results.get('total_time', 0):.2f} seconds\n\n")
        
        if 'attack_mappings' in results:
            f.write("DNN-DRAM PAGE MAPPINGS:\n")
            f.write("-"*40 + "\n")
            for i, mapping in enumerate(results['attack_mappings']):
                f.write(f"\nMapping #{i+1}:\n")
                f.write(f"  DNN Page: {mapping['dnn_page']}\n")
                f.write(f"  DRAM Page: {mapping['dram_page']}\n")
                f.write(f"  Total Flips: {mapping['num_flips']}\n")
                f.write(f"  Beneficial Flips: {mapping.get('beneficial_flips', 'N/A')}\n")
                f.write(f"  0→1 Flips: {mapping.get('flips_0to1', 'N/A')}\n")
                f.write(f"  1→0 Flips: {mapping.get('flips_1to0', 'N/A')}\n")
                f.write(f"  ASR: {mapping['asr']:.2f}%\n")
                f.write(f"  Accuracy: {mapping['acc']:.2f}%\n")
        
        # Add MVM statistics if available
        if 'mvm_statistics' in results:
            f.write("\nMVM STATISTICS:\n")
            f.write("-"*40 + "\n")
            stats = results['mvm_statistics']
            f.write(f"  Total DNN pages analyzed: {stats.get('total_dnn_pages', 'N/A')}\n")
            f.write(f"  DNN pages with critical bits: {stats.get('pages_with_critical_bits', 'N/A')}\n")
            f.write(f"  Total critical bits (0→1): {stats.get('total_critical_0to1', 'N/A')}\n")
            f.write(f"  Total critical bits (1→0): {stats.get('total_critical_1to0', 'N/A')}\n")
    
    print(f"Report saved to: {report_file}")
    
    return results_file, report_file

def visualize_mvm_results(results, result_dir, timestamp):
    """
    Create enhanced visualization for MVM-based attack results
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # 1. DNN Page Rankings by MVM Score
    ax1 = axes[0]
    if 'dnn_page_rankings' in results and results['dnn_page_rankings']:
        rankings = results['dnn_page_rankings'][:10]  # Top 10
        page_ids = [r['page_id'] for r in rankings]
        scores = [r['max_realizable_score'] for r in rankings]
        scores_0to1 = [r['max_score_0to1'] for r in rankings]
        scores_1to0 = [r['max_score_1to0'] for r in rankings]
        
        x = np.arange(len(page_ids))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, scores_0to1, width, label='0→1 Score', color='lightblue')
        bars2 = ax1.bar(x + width/2, scores_1to0, width, label='1→0 Score', color='lightcoral')
        
        ax1.set_xlabel('DNN Page ID')
        ax1.set_ylabel('MVM Score')
        ax1.set_title('Top 10 DNN Pages by MVM Score')
        ax1.set_xticks(x)
        ax1.set_xticklabels([str(p) for p in page_ids])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. Critical Bits Distribution
    ax2 = axes[1]
    if 'mvm_statistics' in results:
        stats = results['mvm_statistics']
        labels = ['0→1 Flips', '1→0 Flips']
        sizes = [stats.get('total_critical_0to1', 0), stats.get('total_critical_1to0', 0)]
        colors = ['lightblue', 'lightcoral']
        
        if sum(sizes) > 0:
            wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors, 
                                               autopct='%1.1f%%', startangle=90)
            ax2.set_title('Critical Bits by Flip Direction')
            
            # Add counts to labels
            for i, (wedge, text) in enumerate(zip(wedges, texts)):
                text.set_text(f'{labels[i]}\n({sizes[i]} bits)')
        else:
            ax2.text(0.5, 0.5, 'No critical bits data', ha='center', va='center')
    
    # 3. Attack Progress
    ax3 = axes[2]
    if 'attack_progress' in results and results['attack_progress']:
        progress = results['attack_progress']
        flips = [p['flips'] for p in progress]
        asrs = [p['asr'] for p in progress]
        accs = [p['acc'] for p in progress]
        
        ax3_twin = ax3.twinx()
        
        line1 = ax3.plot(flips, asrs, 'b-', marker='o', label='ASR', markersize=6, linewidth=2)
        line2 = ax3_twin.plot(flips, accs, 'g-', marker='s', label='Accuracy', markersize=6, linewidth=2)
        
        # Add target ASR line
        ax3.axhline(y=results.get('target_asr', 90.0), color='red', linestyle='--', 
                   alpha=0.5, label='Target ASR')
        
        ax3.set_xlabel('Cumulative Bit Flips')
        ax3.set_ylabel('ASR (%)', color='b')
        ax3_twin.set_ylabel('Accuracy (%)', color='g')
        ax3.set_title('Attack Progress with MVM-Based Selection')
        ax3.grid(True, alpha=0.3)
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax3.legend(lines, labels, loc='center right')
        
        ax3.tick_params(axis='y', labelcolor='b')
        ax3_twin.tick_params(axis='y', labelcolor='g')
    
    # 4. DRAM Page Utilization
    ax4 = axes[3]
    if 'attack_mappings' in results:
        mappings = results['attack_mappings']
        dram_pages = [m['dram_page'] for m in mappings]
        flip_counts = [m['num_flips'] for m in mappings]
        beneficial_counts = [m.get('beneficial_flips', 0) for m in mappings]
        
        x = np.arange(len(dram_pages))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, beneficial_counts, width, label='Beneficial', color='green')
        bars2 = ax4.bar(x + width/2, 
                       [f - b for f, b in zip(flip_counts, beneficial_counts)], 
                       width, label='Non-beneficial', color='orange')
        
        ax4.set_xlabel('DRAM Page ID')
        ax4.set_ylabel('Number of Flips')
        ax4.set_title('DRAM Page Utilization')
        ax4.set_xticks(x)
        ax4.set_xticklabels([str(p) for p in dram_pages], rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # 5. Flip Type Distribution per Mapping
    ax5 = axes[4]
    if 'attack_mappings' in results:
        mappings = results['attack_mappings']
        
        # Aggregate flip types
        total_0to1 = sum(m.get('flips_0to1', 0) for m in mappings)
        total_1to0 = sum(m.get('flips_1to0', 0) for m in mappings)
        
        if total_0to1 + total_1to0 > 0:
            labels = ['0→1', '1→0']
            sizes = [total_0to1, total_1to0]
            colors = ['#3498db', '#e74c3c']
            
            wedges, texts, autotexts = ax5.pie(sizes, labels=labels, colors=colors,
                                               autopct='%1.1f%%', startangle=45)
            ax5.set_title('Applied Flip Types Distribution')
            
            # Add total count
            total_flips = total_0to1 + total_1to0
            ax5.text(0, -1.3, f'Total: {total_flips} flips', ha='center', fontsize=10)
    
    # 6. Performance Metrics
    ax6 = axes[5]
    metrics = {
        'Initial\nAccuracy': results['initial_accuracy'],
        'Final\nAccuracy': results['final_accuracy'],
        'Trigger\nASR': results['trigger_asr'],
        'Final\nASR': results['final_asr']
    }
    
    x = range(len(metrics))
    values = list(metrics.values())
    colors = ['blue', 'darkblue', 'orange', 'darkorange']
    
    bars = ax6.bar(x, values, color=colors)
    ax6.set_ylabel('Percentage (%)')
    ax6.set_title('Attack Performance Metrics')
    ax6.set_xticks(x)
    ax6.set_xticklabels(metrics.keys())
    ax6.set_ylim(0, 105)
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Add target line for ASR
    ax6.axhline(y=results.get('target_asr', 90.0), color='red', linestyle='--', 
                alpha=0.5, label=f"Target ASR ({results.get('target_asr', 90.0)}%)")
    ax6.legend()
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save visualization
    viz_file = os.path.join(result_dir, f'mvm_attack_visualization_{timestamp}.png')
    plt.savefig(viz_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to: {viz_file}")
    
    return viz_file

def main():
    """
    Main function to execute MVM-based hardware-aware INT8 attack
    """
    print("="*80)
    print("MVM-Based Hardware-Aware Backdoor Attack on Native INT8 ResNet20")
    print("Direction-Aware Matrix-Vector Multiplication for Optimal DRAM Selection")
    print("="*80)
    
    # Test bit manipulation first
    test_bit_manipulation()
    
    # Configuration with adaptive thresholds
    config = INT8AttackConfig(
        target_class=2,
        target_asr=90.0,
        alpha_s8=0.1,
        page_size_bits=32768,
        max_bitflips_per_page=44,
        top_dram_pages_per_dnn=30,  # Top K DRAM pages to test
        trigger_size=10,
        trigger_location=(21, 21),
        trigger_epochs=150,
        trigger_lr=0.1,
        
        # Adaptive threshold configuration
        min_clean_accuracy=85.0,  # This will be dynamically adjusted
        min_absolute_accuracy=85.0,  # Lowered absolute minimum
        max_accuracy_drop=10.0,  # Allow up to 25% drop
        
        # Enable adaptive features
        force_progress=True,  # Force attack to continue
        adaptive_threshold_enabled=True,  # Use adaptive thresholds
        trigger_accuracy_tolerance=0.85,  # Allow 15% drop for trigger only
        progressive_relaxation_rate=0.03,  # Relax by 3% per stage
        asr_priority_threshold=0.75,  # When ASR > 75% of target, prioritize ASR
        
        debug_mode=True,
    )
    
    # Paths - UPDATE THESE TO YOUR ACTUAL PATHS
    checkpoint_path = '../ProfilingData/resnet20_int8/resnet20_int8.pth.tar'
    data_path = '../../cifar10/resnet32/data'
    result_dir = '../AttackResults/resnet20_int8_mvm'
    profiling_file = '../ProfilingData/bitflip_matrix_A.npy'
    page_info_file = '../ProfilingData/resnet20_int8/ResNet20_Q3.txt'
    
    # Validate all paths
    print("\nValidating file paths...")
    
    # Get current directory for debugging
    current_dir = os.getcwd()
    print(f"Current working directory: {current_dir}")
    
    # Check checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        abs_path = os.path.abspath(checkpoint_path)
        print(f"   Absolute path would be: {abs_path}")
        return
    else:
        print(f"✓ Checkpoint found: {checkpoint_path}")
    
    # Check profiling file
    if not os.path.exists(profiling_file):
        print(f"❌ DRAM profiling file not found: {profiling_file}")
        abs_path = os.path.abspath(profiling_file)
        print(f"   Absolute path would be: {abs_path}")
        return
    else:
        print(f"✓ DRAM profiling file found: {profiling_file}")
    
    # Check page info file - CRITICAL
    page_info_exists = os.path.exists(page_info_file)
    if not page_info_exists:
        print(f"⚠️  Page info file not found: {page_info_file}")
        
        # Check if parent directory exists and auto-select first .txt file
        parent_dir = os.path.dirname(page_info_file)
        if os.path.exists(parent_dir):
            files = os.listdir(parent_dir)
            txt_files = [f for f in files if f.endswith('.txt')]
            
            if txt_files:
                # Automatically use the first .txt file found
                page_info_file = os.path.join(parent_dir, txt_files[0])
                page_info_exists = True
                print(f"✓ Auto-selected: {page_info_file}")
            else:
                print(f"   No .txt files found in {parent_dir}")
                page_info_file = None
        else:
            print(f"   Parent directory does not exist: {parent_dir}")
            page_info_file = None
        
        if page_info_file is None:
            print("\n⚠️  WARNING: Proceeding with synthetic page mapping")
            print("   This may reduce attack effectiveness")
    else:
        print(f"✓ Page info file found: {page_info_file}")
        
    # Create result directory
    os.makedirs(result_dir, exist_ok=True)
    print(f"✓ Result directory: {result_dir}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load INT8 checkpoint
    int8_state_dict = load_int8_checkpoint(checkpoint_path)
    
    # Create model from INT8 checkpoint
    model, scale_factors = create_model_from_int8_checkpoint(int8_state_dict)
    model = model.to(device)
    model.eval()
    
    # Create data loaders
    test_loader, attack_loader = create_data_loaders(data_path)
    
    # Test initial accuracy
    print("\nValidating model accuracy...")
    initial_accuracy = test_model_accuracy(model, test_loader, num_batches=50)
    
    # Dynamically adjust thresholds based on initial accuracy
    if initial_accuracy < 85:
        print(f"\nWARNING: Low initial accuracy ({initial_accuracy:.2f}%)")
        print("Adjusting thresholds accordingly...")
        
        # Adjust thresholds based on initial accuracy
        config.min_clean_accuracy = max(60.0, initial_accuracy - 25.0)
        config.min_absolute_accuracy = max(50.0, initial_accuracy - 35.0)
        
        print(f"Adjusted thresholds:")
        print(f"  min_clean_accuracy: {config.min_clean_accuracy:.1f}%")
        print(f"  min_absolute_accuracy: {config.min_absolute_accuracy:.1f}%")
        
        print("\nProceeding with adaptive threshold attack...")
    
    # Initialize MVM-based attacker with adaptive config
    print("\nInitializing MVM-based attacker with adaptive thresholds...")
    attacker = MVMHardwareAwareAttack(config)
    
    # Execute attack
    print("\n" + "="*60)
    print("Starting MVM-Based Hardware-Aware Attack")
    print("Mode: ADAPTIVE THRESHOLDS + FORCE PROGRESS")
    if page_info_file is None:
        print("Page Mapping: SYNTHETIC (less accurate)")
    else:
        print(f"Page Mapping: {os.path.basename(page_info_file)}")
    print("="*60)
    
    attack_start_time = time.time()
    
    try:
        # Run MVM-based attack - ENSURE page_info_file is passed
        results = attacker.execute_attack(
            int8_state_dict=int8_state_dict,
            model=model,
            test_loader=test_loader,
            attack_loader=attack_loader,
            scale_factors=scale_factors,
            page_info_file=page_info_file,  # This is now either a valid path or None
            profiling_file=profiling_file
        )
        
        # Add configuration to results
        results['target_class'] = config.target_class
        results['target_asr'] = config.target_asr
        results['alpha_s8'] = config.alpha_s8
        results['top_dram_pages_per_dnn'] = config.top_dram_pages_per_dnn
        results['total_time'] = time.time() - attack_start_time
        results['initial_model_accuracy'] = initial_accuracy
        results['page_mapping_type'] = 'actual' if page_info_file else 'synthetic'
        results['page_info_file'] = page_info_file if page_info_file else 'None'
        results['adaptive_config'] = {
            'force_progress': config.force_progress,
            'adaptive_threshold_enabled': config.adaptive_threshold_enabled,
            'trigger_accuracy_tolerance': config.trigger_accuracy_tolerance,
            'progressive_relaxation_rate': config.progressive_relaxation_rate,
            'final_min_clean_accuracy': config.min_clean_accuracy,
            'final_min_absolute_accuracy': config.min_absolute_accuracy
        }
        
        # Add MVM-specific statistics
        if hasattr(attacker, 'dnn_page_scores'):
            total_0to1 = sum(info['num_critical_bits_0to1'] 
                           for info in attacker.dnn_page_scores.values())
            total_1to0 = sum(info['num_critical_bits_1to0'] 
                           for info in attacker.dnn_page_scores.values())
            
            results['mvm_statistics'] = {
                'total_dnn_pages': len(attacker.dram_pages),
                'pages_with_critical_bits': len(attacker.dnn_page_rankings),
                'total_critical_0to1': total_0to1,
                'total_critical_1to0': total_1to0,
                'total_critical_bits': total_0to1 + total_1to0
            }
        
        # Add DNN page rankings to results for visualization
        if hasattr(attacker, 'dnn_page_rankings'):
            results['dnn_page_rankings'] = attacker.dnn_page_rankings
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file, report_file = save_results(results, result_dir, timestamp)
        
        # Generate visualization
        viz_file = visualize_mvm_results(results, result_dir, timestamp)
        
        print("\n" + "="*60)
        print("MVM-Based Attack Completed!")
        print("="*60)
        print(f"Results saved to: {result_dir}/")
        print(f"\nSummary:")
        print(f"  Initial Model Accuracy: {initial_accuracy:.2f}%")
        print(f"  Page Mapping Type: {results['page_mapping_type']}")
        print(f"  Page Info File: {results['page_info_file']}")
        print(f"  Attack Successful: {results['attack_successful']}")
        print(f"  Final ASR: {results['final_asr']:.2f}% (target: {config.target_asr}%)")
        print(f"  Final Accuracy: {results['final_accuracy']:.2f}%")
        print(f"  DNN-DRAM Mappings Used: {len(results.get('attack_mappings', []))}")
        print(f"  Total Bit Flips: {sum(m['num_flips'] for m in results.get('attack_mappings', []))}")
        
        if results['attack_successful']:
            print(f"\n✅ Attack SUCCEEDED! Achieved {results['final_asr']:.2f}% ASR")
        else:
            print(f"\n⚠️  Attack did not reach target ASR, but achieved {results['final_asr']:.2f}%")
            if results['final_asr'] > 50:
                print("   This may still be a meaningful attack success.")
        
        if 'mvm_statistics' in results:
            stats = results['mvm_statistics']
            print(f"\nMVM Statistics:")
            print(f"  Total critical bits identified: {stats['total_critical_bits']}")
            print(f"  0→1 flips needed: {stats['total_critical_0to1']}")
            print(f"  1→0 flips needed: {stats['total_critical_1to0']}")
        
        if 'adaptive_config' in results:
            print(f"\nAdaptive Configuration Used:")
            print(f"  Force Progress: {results['adaptive_config']['force_progress']}")
            print(f"  Trigger Tolerance: {results['adaptive_config']['trigger_accuracy_tolerance']*100:.0f}%")
            print(f"  Final Min Clean Accuracy: {results['adaptive_config']['final_min_clean_accuracy']:.1f}%")
        
    except Exception as e:
        print(f"\nError during attack: {e}")
        import traceback
        traceback.print_exc()
        
        # Save error log with details
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        error_file = os.path.join(result_dir, f'error_log_{timestamp}.txt')
        with open(error_file, 'w') as f:
            f.write(f"Error: {str(e)}\n\n")
            f.write(f"Initial accuracy: {initial_accuracy:.2f}%\n")
            f.write(f"Page info file: {page_info_file}\n")
            f.write(f"Page info exists: {page_info_exists}\n")
            f.write(f"Configuration:\n")
            f.write(f"  min_clean_accuracy: {config.min_clean_accuracy}\n")
            f.write(f"  min_absolute_accuracy: {config.min_absolute_accuracy}\n")
            f.write(f"  force_progress: {config.force_progress}\n")
            f.write(f"  adaptive_threshold_enabled: {config.adaptive_threshold_enabled}\n\n")
            f.write(traceback.format_exc())
        
        print(f"Error log saved to: {error_file}")

if __name__ == "__main__":
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("\n" + "="*60)
    print("ADAPTIVE THRESHOLD MVM ATTACK")
    print("This version will continue attack progression")
    print("even with accuracy drops from trigger application")
    print("="*60 + "\n")
    
    # Run attack
    main()