#!/usr/bin/env python3
"""
analyze_pth_memory_layout.py
=============================

Load a PyTorch .pth model file and analyze its parameters' memory layout,
displaying the information in a format similar to page_info.txt with:
- Parameter names, shapes, and byte sizes
- Virtual memory page layout (4KB pages)
- Byte-level addressing for each parameter

This helps understand how model parameters would be stored in memory
and provides detailed memory mapping information.

Usage:
python analyze_pth_memory_layout.py --model final_models/resnet20_int8_state.pth
"""

import argparse
import torch
from pathlib import Path
from collections import OrderedDict
import sys


def get_tensor_info(tensor):
    """Get detailed information about a tensor."""
    if not isinstance(tensor, torch.Tensor):
        return None, 0, "non-tensor"
    
    shape = list(tensor.shape)
    numel = tensor.numel()
    element_size = tensor.element_size()
    total_bytes = numel * element_size
    dtype = str(tensor.dtype)
    
    # Format shape string
    if len(shape) == 0:
        shape_str = "scalar"
    elif len(shape) == 1:
        shape_str = str(shape[0])
    else:
        shape_str = "x".join(map(str, shape))
    
    return shape_str, total_bytes, dtype


def analyze_memory_layout(state_dict, page_size=4096):
    """Analyze memory layout of parameters with page-based allocation."""
    
    # Filter out non-tensor entries
    tensor_params = OrderedDict()
    for name, param in state_dict.items():
        if isinstance(param, torch.Tensor):
            tensor_params[name] = param
    
    print(f"# PyTorch Model Memory Layout Analysis")
    print(f"# Format: PARAM name shape size_bytes dtype")
    print(f"# Followed by: PAGE page_index start_byte end_byte num_bytes")
    print(f"TOTAL_PARAMS {len(tensor_params)}")
    print(f"PAGE_SIZE {page_size}")
    print()
    
    current_offset = 0
    current_page = 0
    page_start_offset = 0
    
    for param_name, tensor in tensor_params.items():
        shape_str, param_bytes, dtype = get_tensor_info(tensor)
        
        if param_bytes == 0:  # Skip empty tensors
            continue
            
        print(f"PARAM {param_name} {shape_str} {param_bytes} {dtype}")
        
        # Check if we need to start a new page
        if current_offset + param_bytes > (current_page + 1) * page_size:
            # Fill remaining space in current page
            if current_offset > page_start_offset:
                page_end = min(current_offset + param_bytes - 1, (current_page + 1) * page_size - 1)
                page_bytes = page_end - current_offset + 1
                print(f"PAGE {current_page} {current_offset % page_size} {page_end % page_size} {page_bytes}")
                
                # Update for next page(s)
                remaining_bytes = param_bytes - page_bytes
                current_page += 1
                page_start_offset = current_page * page_size
                current_offset = page_start_offset
                
                # Handle parameters spanning multiple pages
                while remaining_bytes > page_size:
                    print(f"PAGE {current_page} 0 4095 4096")
                    remaining_bytes -= page_size
                    current_page += 1
                    page_start_offset = current_page * page_size
                    current_offset = page_start_offset
                
                # Handle final partial page if needed
                if remaining_bytes > 0:
                    page_end = current_offset + remaining_bytes - 1
                    print(f"PAGE {current_page} {current_offset % page_size} {page_end % page_size} {remaining_bytes}")
                    current_offset += remaining_bytes
            else:
                # Parameter starts at beginning of page
                page_end = current_offset + param_bytes - 1
                page_bytes = param_bytes
                print(f"PAGE {current_page} {current_offset % page_size} {page_end % page_size} {page_bytes}")
                current_offset += param_bytes
        else:
            # Parameter fits in current page
            page_start = current_offset % page_size
            page_end = page_start + param_bytes - 1
            print(f"PAGE {current_page} {page_start} {page_end} {param_bytes}")
            current_offset += param_bytes
            
            # Check if we've filled the current page
            if current_offset >= (current_page + 1) * page_size:
                current_page += 1
                page_start_offset = current_page * page_size
                current_offset = page_start_offset
    
    # Calculate total pages needed
    total_pages = (current_offset + page_size - 1) // page_size
    print(f"\nTOTAL_PAGES {total_pages}")
    print(f"TOTAL_MEMORY_BYTES {current_offset}")
    
    return tensor_params


def print_summary_stats(state_dict):
    """Print summary statistics about the model."""
    tensor_count = 0
    total_params = 0
    total_bytes = 0
    dtype_counts = {}
    
    for name, param in state_dict.items():
        if isinstance(param, torch.Tensor):
            tensor_count += 1
            total_params += param.numel()
            total_bytes += param.numel() * param.element_size()
            
            dtype_str = str(param.dtype)
            if dtype_str not in dtype_counts:
                dtype_counts[dtype_str] = 0
            dtype_counts[dtype_str] += param.numel()
    
    print(f"\n# Summary Statistics")
    print(f"# Total tensors: {tensor_count}")
    print(f"# Total parameters: {total_params:,}")
    print(f"# Total memory: {total_bytes:,} bytes ({total_bytes/1024:.1f} KB)")
    print(f"# Data type distribution:")
    for dtype, count in dtype_counts.items():
        percentage = 100.0 * count / total_params if total_params > 0 else 0
        print(f"#   {dtype}: {count:,} parameters ({percentage:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Analyze PyTorch model memory layout")
    parser.add_argument('--model', type=Path, required=True, 
                       help='Path to PyTorch .pth model file')
    parser.add_argument('--page-size', type=int, default=4096,
                       help='Memory page size in bytes (default: 4096)')
    parser.add_argument('--output', type=Path, 
                       help='Output file path (default: stdout)')
    
    args = parser.parse_args()
    
    # Create dummy utils module first to handle missing dependencies
    try:
        import types
        
        class DummyRecorderMeter:
            def __init__(self, *args, **kwargs):
                pass
        
        class DummyAverageMeter:
            def __init__(self, *args, **kwargs):
                pass
        
        # Add dummy utils module if it doesn't exist
        if 'utils' not in sys.modules:
            utils_module = types.ModuleType('utils')
            utils_module.RecorderMeter = DummyRecorderMeter
            utils_module.AverageMeter = DummyAverageMeter
            sys.modules['utils'] = utils_module
            print(f"# Created dummy utils module", file=sys.stderr)
    except Exception as e:
        print(f"# Failed to create dummy utils: {e}", file=sys.stderr)
    
    # Load the model with fallback strategies
    checkpoint = None
    try:
        # First try with weights_only=False for full checkpoint
        checkpoint = torch.load(args.model, map_location='cpu', weights_only=False)
        print(f"# Loaded model from: {args.model}", file=sys.stderr)
        print(f"# File size: {args.model.stat().st_size} bytes ({args.model.stat().st_size/1024:.1f} KB)", file=sys.stderr)
    except Exception as e1:
        print(f"# Failed to load full checkpoint: {e1}", file=sys.stderr)
        try:
            # Try with weights_only=True to load just tensors
            checkpoint = torch.load(args.model, map_location='cpu', weights_only=True)
            print(f"# Loaded weights only from: {args.model}", file=sys.stderr)
            print(f"# File size: {args.model.stat().st_size} bytes ({args.model.stat().st_size/1024:.1f} KB)", file=sys.stderr)
        except Exception as e2:
            print(f"Error loading model with all strategies: {e1}, {e2}", file=sys.stderr)
            return 1
    
    # Extract state_dict
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"# Using 'model_state_dict' from checkpoint", file=sys.stderr)
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print(f"# Using 'state_dict' from checkpoint", file=sys.stderr)
        else:
            # Assume the entire dict is the state_dict
            state_dict = checkpoint
            print(f"# Using entire checkpoint as state_dict", file=sys.stderr)
    else:
        print(f"Error: Unexpected checkpoint format", file=sys.stderr)
        return 1
    
    # Redirect output if specified
    if args.output:
        sys.stdout = open(args.output, 'w')
    
    try:
        # Analyze memory layout
        tensor_params = analyze_memory_layout(state_dict, args.page_size)
        
        # Print summary statistics
        print_summary_stats(state_dict)
        
    finally:
        if args.output:
            sys.stdout.close()
            sys.stdout = sys.__stdout__
            print(f"Memory layout analysis saved to: {args.output}")
    
    return 0


if __name__ == "__main__":
    exit(main()) 