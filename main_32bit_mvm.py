#!/usr/bin/env python3
"""
MVM-Optimized main_32bit_vanilla.py with Matrix-Vector Multiplication approach
Implements systematic DNN page ranking and DRAM matching
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import models
import json
import time
import matplotlib.pyplot as plt
import copy
import csv

# Import the MVM-optimized RowHammer implementation
from hardware_aware_backdoor_32bit_mvm import MVMRowHammerProFlip, RowHammerConfig, WeightedFP32Config

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def save_trigger_as_image(trigger_pattern, save_path):
    """
    Save trigger pattern as a PNG image for visualization
    
    Args:
        trigger_pattern: torch.Tensor of shape (C, H, W) with values in [0, 1]
        save_path: path to save the image
    """
    import matplotlib.pyplot as plt
    
    # Convert to numpy and transpose to (H, W, C) for matplotlib
    trigger_np = trigger_pattern.cpu().numpy().transpose(1, 2, 0)
    
    # Ensure values are in [0, 1] range
    trigger_np = np.clip(trigger_np, 0, 1)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    # Handle different channel formats
    if trigger_np.shape[2] == 3:
        # RGB image
        ax.imshow(trigger_np)
    elif trigger_np.shape[2] == 1:
        # Grayscale image
        ax.imshow(trigger_np.squeeze(), cmap='gray')
    else:
        # Multi-channel - show first 3 channels as RGB
        ax.imshow(trigger_np[:, :, :3])
    
    ax.set_title(f'Trigger Pattern\\nShape: {trigger_pattern.shape}')
    ax.axis('off')
    
    # Add pixel values as text if trigger is small enough
    if trigger_pattern.shape[1] <= 16 and trigger_pattern.shape[2] <= 16:
        for i in range(trigger_pattern.shape[1]):
            for j in range(trigger_pattern.shape[2]):
                # Show RGB values for small triggers
                if trigger_np.shape[2] >= 3:
                    rgb_vals = trigger_np[i, j, :3]
                    text = f'({rgb_vals[0]:.2f},{rgb_vals[1]:.2f},{rgb_vals[2]:.2f})'
                else:
                    text = f'{trigger_np[i, j, 0]:.2f}'
                ax.text(j, i, text, ha='center', va='center', 
                       fontsize=8, color='white', weight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()

def save_results_with_trigger(results, result_dir, attacker=None):
    """
    Save attack results including trigger pattern in multiple formats
    """
    # Add trigger pattern to results if available
    if attacker and hasattr(attacker, 'trigger_pattern') and attacker.trigger_pattern is not None:
        results['trigger_pattern_shape'] = list(attacker.trigger_pattern.shape)
        results['trigger_pattern_data'] = attacker.trigger_pattern.cpu().numpy().tolist()
        print("✓ Trigger pattern included in results")
        
        # Also save trigger pattern as a separate .npy file for easier loading
        trigger_file = os.path.join(result_dir, 'trigger_pattern.npy')
        np.save(trigger_file, attacker.trigger_pattern.cpu().numpy())
        print(f"✓ Trigger pattern saved as numpy file: {trigger_file}")
        
        # Save trigger pattern as image for visualization
        try:
            trigger_img_file = os.path.join(result_dir, 'trigger_pattern.png')
            save_trigger_as_image(attacker.trigger_pattern, trigger_img_file)
            print(f"✓ Trigger pattern saved as image: {trigger_img_file}")
        except Exception as e:
            print(f"⚠️ Warning: Could not save trigger as image: {e}")
    else:
        print("⚠️ Warning: No trigger pattern available to save")
    
    # Save JSON results with clean filename
    results_file = os.path.join(result_dir, 'attack_results.json')
    results_serializable = convert_numpy_types(results)
    with open(results_file, 'w') as f:
        json.dump(results_serializable, f, indent=2, default=str)
    
    print(f"✓ Results saved to: {results_file}")
    return results_file

def evaluate_model(model, loader, trigger_pattern=None, trigger_area=None, target_class=None):
    """
    Evaluate model accuracy with optional trigger pattern.
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.cuda(), target.cuda()
            
            # Apply trigger if provided
            if trigger_pattern is not None and trigger_area is not None:
                start, end = trigger_area
                data[:, :, start:end, start:end] = trigger_pattern
                if target_class is not None:
                    # For ASR calculation, check predictions against target class
                    target = torch.full_like(target, target_class)
            
            outputs = model(data)
            _, predicted = outputs.max(1)
            correct += predicted.eq(target).sum().item()
            total += target.size(0)
    
    accuracy = 100. * correct / total
    return accuracy

def plot_mvm_attack_results(results: dict, result_dir: str, target_asr: float):
    """Plotting function for MVM-optimized attack results"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('MVM-Optimized RowHammer Attack Results\n(αs=%.1f, αe=%.1f, αm=%.1f)' % 
                 (results.get('alpha_s', 1.0), results.get('alpha_e', 10.0), results.get('alpha_m', 0.1)), 
                 fontsize=16)
    
    successful_attacks = results.get('successful_attacks', [])
    statistics = results.get('statistics', {})
    
    # Plot 1: ASR progression
    ax1 = axes[0, 0]
    if successful_attacks:
        # Build cumulative data
        cumulative_flips = 0
        bit_counts = [0]
        asr_values = [results.get('initial_asr', 0)]
        acc_values = [results.get('baseline_accuracy', 0)]
        
        for attack in successful_attacks:
            cumulative_flips += attack.get('num_flips', 0)
            bit_counts.append(cumulative_flips)
            asr_values.append(attack.get('asr', 0))
            acc_values.append(attack.get('acc', 0))
        
        ax1.plot(bit_counts, asr_values, 'ro-', linewidth=2, markersize=8, label='ASR')
    else:
        bit_counts = [0]
        asr_values = [results.get('initial_asr', 0)]
        ax1.scatter(bit_counts, asr_values, color='red', s=100, label='Initial ASR')
    
    ax1.axhline(y=target_asr, color='g', linestyle='--', label=f'Target ASR ({target_asr}%)')
    ax1.set_xlabel('Cumulative Bit Flips')
    ax1.set_ylabel('Attack Success Rate (%)')
    ax1.set_title('ASR Progression (MVM-Optimized)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(0, 105)
    
    # Plot 2: Accuracy vs bit flips
    ax2 = axes[0, 1]
    if successful_attacks:
        ax2.plot(bit_counts, acc_values, 'bo-', linewidth=2, markersize=8, label='Clean Accuracy')
    else:
        acc_values = [results.get('baseline_accuracy', 0)]
        ax2.scatter(bit_counts, acc_values, color='blue', s=100, label='Initial Accuracy')
    
    ax2.axhline(y=results.get('min_clean_accuracy', 80), color='r', linestyle='--', label='Min Accuracy')
    ax2.set_xlabel('Cumulative Bit Flips')
    ax2.set_ylabel('Clean Accuracy (%)')
    ax2.set_title('Accuracy Trade-off')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim(0, 105)
    
    # Plot 3: DNN Page Rankings (MVM Scores)
    ax3 = axes[0, 2]
    dnn_page_scores = results.get('dnn_page_mvm_scores', [])
    if dnn_page_scores:
        pages = [item['page_id'] for item in dnn_page_scores[:20]]
        scores = [item['max_achievable_score'] for item in dnn_page_scores[:20]]
        used_pages = [attack['dnn_page_id'] for attack in successful_attacks]
        
        colors = ['green' if p in used_pages else 'lightblue' for p in pages]
        
        bars = ax3.bar(range(len(pages)), scores, color=colors)
        ax3.set_xlabel('DNN Page Rank')
        ax3.set_ylabel('Max Achievable Score (MVM)')
        ax3.set_title('DNN Page Rankings by MVM\n(Green = Used in Attack)')
        ax3.set_xticks(range(0, len(pages), 2))
        ax3.set_xticklabels([f"#{i+1}" for i in range(0, len(pages), 2)])
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add page IDs on bars
        for i, (bar, page_id) in enumerate(zip(bars[:10], pages[:10])):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'P{page_id}', ha='center', va='bottom', fontsize=8)
    else:
        ax3.text(0.5, 0.5, 'No DNN page scores available', ha='center', va='center', 
                transform=ax3.transAxes, fontsize=14)
    
    # Plot 4: DRAM Candidates Tested
    ax4 = axes[1, 0]
    if successful_attacks:
        dram_tests = {}
        for attack in successful_attacks:
            dnn_id = attack['dnn_page_id']
            if dnn_id not in dram_tests:
                dram_tests[dnn_id] = 0
            dram_tests[dnn_id] += 1
        
        pages = list(dram_tests.keys())
        tests = [dram_tests[p] for p in pages]
        
        ax4.bar(range(len(pages)), tests, color='coral')
        ax4.set_xlabel('DNN Page ID')
        ax4.set_ylabel('DRAM Candidates Tested')
        ax4.set_title('DRAM Search Effort per DNN Page')
        ax4.set_xticks(range(len(pages)))
        ax4.set_xticklabels([f'P{p}' for p in pages], rotation=45)
        ax4.grid(True, alpha=0.3, axis='y')
    else:
        ax4.text(0.5, 0.5, 'No attack data', ha='center', va='center', 
                transform=ax4.transAxes, fontsize=14)
    
    # Plot 5: Weighted Score vs ASR
    ax5 = axes[1, 1]
    if successful_attacks:
        weighted_scores = [a.get('weighted_score', 0) for a in successful_attacks]
        asrs = [a.get('asr', 0) for a in successful_attacks]
        
        ax5.scatter(weighted_scores, asrs, c=range(len(weighted_scores)), 
                   cmap='viridis', s=100, alpha=0.7)
        
        # Add trend line
        if len(weighted_scores) > 1:
            z = np.polyfit(weighted_scores, asrs, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(weighted_scores), max(weighted_scores), 100)
            ax5.plot(x_trend, p(x_trend), "r--", alpha=0.5, label='Trend')
        
        ax5.set_xlabel('Weighted Score Achieved')
        ax5.set_ylabel('ASR (%)')
        ax5.set_title('Weighted Score vs ASR Achievement')
        ax5.grid(True, alpha=0.3)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap='viridis', 
                                   norm=plt.Normalize(vmin=0, vmax=len(weighted_scores)-1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax5)
        cbar.set_label('Attack Order', rotation=270, labelpad=15)
    else:
        ax5.text(0.5, 0.5, 'No successful attacks', ha='center', va='center', 
                transform=ax5.transAxes, fontsize=14)
    
    # Plot 6: Attack Statistics
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # Create statistics text
    stats_text = f"""MVM Attack Statistics:
    
Total DNN Pages Tried: {statistics.get('dnn_pages_tried', 0)}
DRAM Candidates Tested: {statistics.get('dram_candidates_tested', 0)}
Successful Mappings: {statistics.get('successful_mappings', 0)}

Total Bit Flips: {statistics.get('total_bitflips', 0)}
Total Weighted Score: {statistics.get('total_weighted_score', 0):.2f}

Attack Success: {'YES' if results.get('attack_successful', False) else 'NO'}
Final ASR: {results.get('final_asr', 0):.2f}%
Final Accuracy: {results.get('final_accuracy', 0):.2f}%

Efficiency Metrics:
- Avg Flips/Mapping: {statistics.get('total_bitflips', 0) / max(1, statistics.get('successful_mappings', 1)):.1f}
- Avg Score/Flip: {statistics.get('total_weighted_score', 0) / max(1, statistics.get('total_bitflips', 1)):.3f}
"""
    
    ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'{result_dir}/attack_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to {result_dir}/attack_visualization.png")

def generate_mvm_report(result_dir, results, initial_acc):
    """Generate report for MVM-optimized attack"""
    with open(f'{result_dir}/attack_report.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("MVM-OPTIMIZED ROWHAMMER PROFLIP ATTACK REPORT\n")
        f.write("Matrix-Vector Multiplication Based DNN Page Ranking\n")
        f.write("="*80 + "\n\n")
        
        # MVM Configuration
        f.write("MVM OPTIMIZATION CONFIGURATION:\n")
        f.write("-"*40 + "\n")
        f.write("Approach: Systematic DNN page ranking using max(V × score_vector)\n")
        f.write(f"Top-K DRAM candidates per DNN page: {results.get('mvm_top_k', 20)}\n")
        f.write(f"Alpha_s (sign weight): {results.get('alpha_s', 1.0)}\n")
        f.write(f"Alpha_e (exponent weight): {results.get('alpha_e', 10.0)}\n")
        f.write(f"Alpha_m (mantissa weight): {results.get('alpha_m', 0.1)}\n\n")
        
        # Attack Summary
        f.write("ATTACK SUMMARY:\n")
        f.write("-"*40 + "\n")
        f.write(f"Attack Status: {'SUCCESS' if results.get('attack_successful', False) else 'FAILED'}\n")
        f.write(f"Initial ASR (trigger only): {results.get('initial_asr', 0.0):.2f}%\n")
        f.write(f"Final ASR: {results.get('final_asr', 0.0):.2f}%\n")
        f.write(f"ASR Improvement: +{results.get('asr_improvement', 0.0):.2f}%\n")
        f.write(f"Target ASR: {results.get('target_asr', 90.0):.2f}%\n")
        f.write(f"Initial Accuracy: {results.get('baseline_accuracy', initial_acc):.2f}%\n")
        f.write(f"Final Accuracy: {results.get('final_accuracy', 0.0):.2f}%\n")
        f.write(f"Accuracy Drop: {results.get('baseline_accuracy', initial_acc) - results.get('final_accuracy', 0.0):.2f}%\n\n")
        
        # MVM Statistics
        stats = results.get('statistics', {})
        f.write("MVM ATTACK STATISTICS:\n")
        f.write("-"*40 + "\n")
        f.write(f"DNN pages analyzed: {stats.get('dnn_pages_tried', 0)}\n")
        f.write(f"DRAM candidates tested: {stats.get('dram_candidates_tested', 0)}\n")
        f.write(f"Successful DNN-DRAM mappings: {stats.get('successful_mappings', 0)}\n")
        f.write(f"Total bit flips applied: {stats.get('total_bitflips', 0)}\n")
        f.write(f"Total weighted score achieved: {stats.get('total_weighted_score', 0):.2f}\n\n")
        
        # DNN Page Rankings
        dnn_scores = results.get('dnn_page_mvm_scores', [])
        if dnn_scores:
            f.write("TOP DNN PAGES BY MVM SCORE:\n")
            f.write("-"*40 + "\n")
            f.write("Rank | Page ID | Max Achievable Score | Used\n")
            f.write("-"*45 + "\n")
            
            used_pages = [a['dnn_page_id'] for a in results.get('successful_attacks', [])]
            
            for i, page_info in enumerate(dnn_scores[:20]):
                used = "YES" if page_info['page_id'] in used_pages else "NO"
                f.write(f"{i+1:4d} | {page_info['page_id']:7d} | {page_info['max_achievable_score']:20.2f} | {used}\n")
            f.write("\n")
        
        # Attack Progression
        successful = results.get('successful_attacks', [])
        if successful:
            f.write("ATTACK PROGRESSION:\n")
            f.write("-"*40 + "\n")
            f.write("Step | DNN Page | DRAM Page | Flips | Weighted Score | ASR (%) | Acc (%)\n")
            f.write("-"*75 + "\n")
            
            cumulative_flips = 0
            for i, attack in enumerate(successful):
                cumulative_flips += attack['num_flips']
                f.write(f"{i+1:4d} | {attack['dnn_page_id']:8d} | "
                       f"{attack['dram_page_id']:9d} | {attack['num_flips']:5d} | "
                       f"{attack.get('weighted_score', 0):14.2f} | "
                       f"{attack['asr']:7.2f} | {attack['acc']:7.2f}\n")
            
            f.write(f"\nCumulative bit flips: {cumulative_flips}\n")
        
        # Efficiency Analysis
        f.write("\nEFFICIENCY ANALYSIS:\n")
        f.write("-"*40 + "\n")
        if stats.get('successful_mappings', 0) > 0:
            avg_flips = stats.get('total_bitflips', 0) / stats['successful_mappings']
            avg_score_per_flip = stats.get('total_weighted_score', 0) / max(1, stats.get('total_bitflips', 1))
            avg_candidates_per_page = stats.get('dram_candidates_tested', 0) / max(1, stats.get('dnn_pages_tried', 1))
            
            f.write(f"Average bit flips per successful mapping: {avg_flips:.1f}\n")
            f.write(f"Average weighted score per bit flip: {avg_score_per_flip:.3f}\n")
            f.write(f"Average DRAM candidates tested per DNN page: {avg_candidates_per_page:.1f}\n")
            f.write(f"Success rate: {100 * stats['successful_mappings'] / max(1, stats.get('dnn_pages_tried', 1)):.1f}%\n")
        
        # Execution Time
        exec_time = results.get('execution_time', {})
        f.write(f"\nEXECUTION TIME:\n")
        f.write("-"*40 + "\n")
        f.write(f"SNI Phase: {exec_time.get('sni_seconds', 0):.2f}s\n")
        f.write(f"Trigger Optimization: {exec_time.get('trigger_seconds', 0):.2f}s\n")
        f.write(f"Total Pipeline: {exec_time.get('total_pipeline', 0):.2f}s\n")
        
        # Key Insights
        f.write("\nKEY INSIGHTS:\n")
        f.write("-"*40 + "\n")
        f.write("- MVM approach directly considers hardware feasibility in ranking\n")
        f.write("- DNN pages ranked by maximum achievable score, not theoretical vulnerability\n")
        f.write("- Systematic search ensures optimal DNN-DRAM page matching\n")
        f.write("- No complex constraint matching or tolerance calculations needed\n")
        
        # Add trigger pattern information if available
        if 'trigger_pattern_shape' in results:
            f.write("\nTRIGGER PATTERN:\n")
            f.write("-"*40 + "\n")
            shape = results['trigger_pattern_shape']
            f.write(f"  Trigger shape: {shape} (C x H x W)\n")
            f.write(f"  Trigger size: {shape[1]}x{shape[2]} pixels\n")
            f.write(f"  Channels: {shape[0]}\n")
            f.write(f"  Trigger data saved in JSON and .npy format\n")
        
    print(f"Report saved to {result_dir}/attack_report.txt")

def main():
    """
    Main function for MVM-Optimized RowHammer ProFlip attack
    """
    # Attack configuration
    target_class = 2  # Target backdoor class
    target_asr = 90.0  # Target attack success rate
    trigger_area = (21, 31)  # Trigger location (10x10 patch)
    
    # MVM optimization parameters
    # Sweep: Top K DRAM pages to test per DNN page
    # For this experiment, we only test K = 15 and K = 30
    mvm_top_k_candidates_list = [15, 30]
    
    # Weighted FP32 configuration
    alpha_s = 1.0   # Sign bit weight
    alpha_e = 10.0  # Exponent bit weight  
    alpha_m = 0.1   # Mantissa bit weight
    
    # Model configuration
    model_name = 'vanilla_resnet20'
    model_path = './saved_model/resnet20_fp32/model_best.pth.tar'
    data_path = './data'
    result_dir = './results/mvm_resnet20'
    page_info_file = './pagemaps/resnet20_fp32_pagemap.txt'
    profiling_file = './profile_results/device1_256MB_4row.npy'
    
    # Create result directory
    os.makedirs(result_dir, exist_ok=True)
    
    print("=== MVM-Optimized RowHammer ProFlip Attack ===")
    print("Using Matrix-Vector Multiplication for DNN Page Ranking")
    print(f"Target class: {target_class}")
    print(f"Target ASR: {target_asr}%")
    print(f"Trigger area: {trigger_area}")
    print(f"MVM top-K candidates sweep: {mvm_top_k_candidates_list}")
    print(f"Weight factors: αs={alpha_s}, αe={alpha_e}, αm={alpha_m}")
    print(f"Model: {model_name}")
    
    # Data loading configuration
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # Load CIFAR-10 dataset
    print("\nLoading CIFAR-10 dataset...")
    testset = torchvision.datasets.CIFAR10(
        root=data_path, 
        train=False, 
        download=True, 
        transform=transform_test
    )
    
    # Create data loaders
    loader_test = torch.utils.data.DataLoader(
        testset, 
        batch_size=128, 
        shuffle=False, 
        num_workers=2
    )
    
    loader_search = torch.utils.data.DataLoader(
        testset, 
        batch_size=32, 
        shuffle=True, 
        num_workers=2
    )
    
    # Load the model
    print(f"\nLoading model {model_name}...")
    
    if model_name not in models.__dict__:
        raise ValueError(f"Model '{model_name}' not found in models")
    
    model = models.__dict__[model_name](10)
    
    # Load pretrained weights
    print(f"\nLoading weights from {model_path}...")
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print(f"Loaded checkpoint from epoch: {checkpoint.get('epoch', 'unknown')}")
        else:
            state_dict = checkpoint
        # Keep an immutable copy of the original state dict for repeated experiments
        base_state_dict = copy.deepcopy(state_dict)
        model.load_state_dict(base_state_dict)
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Move model to GPU
    model = model.cuda()
    model.eval()
    
    # Test the model
    print("\nTesting model performance...")
    initial_acc = evaluate_model(model, loader_test)
    print(f"Initial model accuracy: {initial_acc:.2f}%")
    
    if initial_acc < 85.0:
        print(f"\nWARNING: Accuracy is below expected for ResNet32 on CIFAR-10")
        print(f"Expected: ~91%, Got: {initial_acc:.2f}%")
        return
    
    # Check if required files exist
    if not os.path.exists(profiling_file):
        print(f"\nError: DRAM profiling file '{profiling_file}' not found!")
        return
    
    if not os.path.exists(page_info_file):
        print(f"\nError: Page info file '{page_info_file}' not found!")
        return
    
    # Load and analyze DRAM profiling data
    print("\nAnalyzing DRAM profiling data...")
    profiling_matrix = np.load(profiling_file)
    print(f"Profiling matrix shape: {profiling_matrix.shape}")
    
    # Prepare sweep over different MVM top-K candidates
    print("\nPreparing MVM top-K sweep experiment...")
    print(f"  Values: {mvm_top_k_candidates_list}")
    
    sweep_start_time = time.time()
    sweep_summary = []
    
    for mvm_top_k_candidates in mvm_top_k_candidates_list:
        print("\n" + "="*60)
        print(f"Starting MVM-Optimized RowHammer ProFlip Attack (top-K = {mvm_top_k_candidates})")
        print("="*60)
        
        # Reset model to clean FP32 weights for each experiment
        model.load_state_dict(base_state_dict)
        model = model.cuda()
        model.eval()
        
        # Configure MVM-optimized RowHammer attack
        config = RowHammerConfig(
            # DRAM configuration
            page_size_bits=32768,  # 4KB pages
            page_start_offset=0x0000,
            
            # MVM optimization
            mvm_top_k_dram_candidates=mvm_top_k_candidates,
            
            # Attack parameters
            target_asr=target_asr,
            min_clean_acc_retention=0.85,
            min_clean_accuracy=85.0,
            
            # SNI parameters
            sni_theta=0.1,
            sni_gamma=0.5,
            sni_num_features=30,
            
            # Trigger optimization
            trigger_lr=0.1,
            trigger_epochs=150,
            trigger_lambda1=1.0,
            trigger_lambda2=1.0,
            trigger_c=10.0,
            
            # Weighted FP32 parameters
            weighted_fp32_alpha_s=alpha_s,
            weighted_fp32_alpha_e=alpha_e,
            weighted_fp32_alpha_m=alpha_m
        )
        
        print(f"\nMVM Attack configuration (top-K = {mvm_top_k_candidates}):")
        print(f"  Initial model accuracy: {initial_acc:.2f}%")
        print(f"  Minimum clean accuracy threshold: {config.min_clean_accuracy:.1f}%")
        print(f"  Target ASR: {target_asr}%")
        print(f"  MVM top-K: {mvm_top_k_candidates}")
        print(f"  Weighted scoring: αs={alpha_s}, αe={alpha_e}, αm={alpha_m}")
        
        # Create per-topK result directory
        topk_result_dir = os.path.join(result_dir, f'mvm_topk_{mvm_top_k_candidates}')
        os.makedirs(topk_result_dir, exist_ok=True)
        
        # Initialize the MVM-optimized attacker
        attacker = MVMRowHammerProFlip(config, page_info_file)
        
        # Record start time for this run
        total_start_time = time.time()
        
        results = None
        
        try:
            # Execute the MVM-optimized attack
            results = attacker.mvm_optimized_attack(
                model=model,
                data_loader=loader_test,
                loader_search=loader_search,
                target_class=target_class,
                trigger_area=trigger_area,
                profiling_file=profiling_file
            )
            
            # Add execution time
            total_time = time.time() - total_start_time
            if results:
                if 'execution_time' not in results:
                    results['execution_time'] = {}
                results['execution_time']['total_pipeline'] = total_time
                
                # Add configuration parameters
                results['alpha_s'] = alpha_s
                results['alpha_e'] = alpha_e
                results['alpha_m'] = alpha_m
                results['mvm_top_k'] = mvm_top_k_candidates
                results['min_clean_accuracy'] = config.min_clean_accuracy
                
                # Add DNN page rankings if available
                if hasattr(attacker, 'ranked_pages') and attacker.ranked_pages:
                    results['dnn_page_mvm_scores'] = [
                        {
                            'page_id': page.page_id,
                            'max_achievable_score': page.max_achievable_score
                        }
                        for page in attacker.ranked_pages[:50]  # Top 50 pages
                    ]
                
                # Generate visualizations
                print("\nGenerating visualizations...")
                plot_mvm_attack_results(results, topk_result_dir, target_asr)
                
                # Generate detailed report
                generate_mvm_report(topk_result_dir, results, initial_acc)
        
        except Exception as e:
            print(f"\nError during attack (top-K = {mvm_top_k_candidates}): {e}")
            import traceback
            traceback.print_exc()
            
            # Create minimal results
            results = {
                'attack_successful': False,
                'error': str(e),
                'initial_accuracy': initial_acc,
                'alpha_s': alpha_s,
                'alpha_e': alpha_e,
                'alpha_m': alpha_m,
                'mvm_top_k': mvm_top_k_candidates,
                'execution_time': {
                    'total_pipeline': time.time() - total_start_time
                }
            }
        
        # Save results with trigger pattern
        if results:
            save_results_with_trigger(results, topk_result_dir, attacker)
        
        print(f"\nAttack completed for top-K = {mvm_top_k_candidates}. Results saved to: {topk_result_dir}/")
        print(f"Total execution time (top-K = {mvm_top_k_candidates}): {time.time() - total_start_time:.2f} seconds")
        
        # Print final summary for this run
        if results and results.get('attack_successful'):
            print("\n" + "="*60)
            print(f"ATTACK SUCCEEDED! (top-K = {mvm_top_k_candidates})")
            print(f"Final ASR: {results.get('final_asr', 0):.2f}%")
            print(f"Total bit flips: {results.get('statistics', {}).get('total_bitflips', 0)}")
            print(f"Total weighted score: {results.get('statistics', {}).get('total_weighted_score', 0):.2f}")
            print(f"Successful mappings: {results.get('statistics', {}).get('successful_mappings', 0)}")
            print("="*60)
        else:
            print("\n" + "="*60)
            print(f"ATTACK FAILED! (top-K = {mvm_top_k_candidates})")
            if results:
                print(f"Final ASR: {results.get('final_asr', 0):.2f}% (target: {target_asr}%)")
                print(f"Reason: {results.get('error', 'Unknown')}")
            print("="*60)
        
        # Collect summary statistics for this top-K
        stats = results.get('statistics', {}) if results else {}
        sweep_summary.append({
            'mvm_top_k': mvm_top_k_candidates,
            'attack_successful': bool(results.get('attack_successful')) if results else False,
            'final_asr': float(results.get('final_asr', 0.0)) if results else 0.0,
            'final_accuracy': float(results.get('final_accuracy', 0.0)) if results else 0.0,
            'initial_asr': float(results.get('initial_asr', 0.0)) if results else 0.0,
            'baseline_accuracy': float(results.get('baseline_accuracy', initial_acc)) if results else float(initial_acc),
            'total_bitflips': int(stats.get('total_bitflips', 0)) if stats else 0,
            'total_weighted_score': float(stats.get('total_weighted_score', 0.0)) if stats else 0.0,
            'dnn_pages_tried': int(stats.get('dnn_pages_tried', 0)) if stats else 0,
            'dram_candidates_tested': int(stats.get('dram_candidates_tested', 0)) if stats else 0,
            'successful_mappings': int(stats.get('successful_mappings', 0)) if stats else 0
        })
    
    # Save sweep summary data (JSON and CSV) and aggregate plot
    if sweep_summary:
        summary_json_path = os.path.join(result_dir, 'mvm_topk_sweep_summary.json')
        with open(summary_json_path, 'w') as f:
            json.dump(convert_numpy_types({'results': sweep_summary}), f, indent=2)
        print(f"\n✓ Sweep summary saved to JSON: {summary_json_path}")
        
        summary_csv_path = os.path.join(result_dir, 'mvm_topk_sweep_summary.csv')
        with open(summary_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'mvm_top_k', 'attack_successful', 'final_asr', 'final_accuracy',
                'initial_asr', 'baseline_accuracy', 'total_bitflips',
                'total_weighted_score', 'dnn_pages_tried',
                'dram_candidates_tested', 'successful_mappings'
            ])
            for row in sweep_summary:
                writer.writerow([
                    row['mvm_top_k'],
                    row['attack_successful'],
                    row['final_asr'],
                    row['final_accuracy'],
                    row['initial_asr'],
                    row['baseline_accuracy'],
                    row['total_bitflips'],
                    row['total_weighted_score'],
                    row['dnn_pages_tried'],
                    row['dram_candidates_tested'],
                    row['successful_mappings']
                ])
        print(f"✓ Sweep summary saved to CSV: {summary_csv_path}")
        
        # Create aggregate plot: final ASR and clean accuracy vs MVM top-K
        ks = [row['mvm_top_k'] for row in sweep_summary]
        asrs = [row['final_asr'] for row in sweep_summary]
        accs = [row['final_accuracy'] for row in sweep_summary]
        
        plt.figure(figsize=(8, 6))
        plt.plot(ks, asrs, 'ro-', linewidth=2, markersize=8, label='Final ASR (%)')
        plt.plot(ks, accs, 'bo-', linewidth=2, markersize=8, label='Final Clean Accuracy (%)')
        plt.xlabel('Number of DRAM Pages Searched per DNN Page (top-K)')
        plt.ylabel('Percentage (%)')
        plt.title('Effect of DRAM Matching Search Space on ASR and Clean Accuracy\n(FP32 MVM-Optimized RowHammer ProFlip)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        sweep_plot_path = os.path.join(result_dir, 'mvm_topk_sweep_plot.png')
        plt.savefig(sweep_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Sweep plot saved to: {sweep_plot_path}")
    
    print(f"\nMVM top-K sweep completed in {time.time() - sweep_start_time:.2f} seconds.")

if __name__ == "__main__":
    # Set GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # Run the MVM-optimized attack
    main()