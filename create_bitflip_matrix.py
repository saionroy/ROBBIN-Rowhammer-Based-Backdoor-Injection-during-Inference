#!/usr/bin/env python3
import json
import numpy as np
import argparse
from pathlib import Path
from collections import defaultdict

def load_profile_data(profile_path):
    """Load the profile.json file"""
    with open(profile_path, 'r') as f:
        data = json.load(f)
    return data

def create_bitflip_matrix(profile_path, output_path="bitflip_matrix.npy"):
    """
    Create a bitflip matrix from profile.json using all pages
    
    Args:
        profile_path: Path to profile.json file
        output_path: Path to save the matrix
    
    Returns:
        numpy array of shape (num_pages, 32768) with values -1, 0, or 1
        Pages are ranked with most vulnerable (most bitflips) at the top
    """
    
    print(f"Loading profile data from {profile_path}...")
    profile_data = load_profile_data(profile_path)
    
    # Group bitflips by physical page number
    page_bitflips = defaultdict(list)
    
    print("Processing bitflip data...")
    for entry in profile_data:
        phys_page = entry['phys_page_number']
        page_offset = entry['page_offset']
        bit_position = entry['bit_position']
        flip_type = entry['flip_type']
        
        page_bitflips[phys_page].append({
            'page_offset': page_offset,
            'bit_position': bit_position,
            'flip_type': flip_type
        })
    
    # Count bitflips per page and use all pages
    page_bitflip_counts = [(page, len(bitflips)) for page, bitflips in page_bitflips.items()]
    page_bitflip_counts.sort(key=lambda x: x[1], reverse=True)  # Sort by count descending (most vulnerable first)
    
    num_pages = len(page_bitflip_counts)
    print(f"Total unique pages with bitflips: {num_pages}")
    
    # Use all pages, already sorted in descending order (most bitflips first)
    selected_pages = page_bitflip_counts
    print(f"Using all {num_pages} pages with bitflips")
    print(f"Most vulnerable page (row 0): {selected_pages[0][1]} bitflips")
    print(f"Least vulnerable page (row {num_pages-1}): {selected_pages[-1][1]} bitflips")
    
    # Create the matrix: num_pages rows x 32768 columns
    # Each page is 4KB = 4096 bytes = 4096 * 8 = 32768 bits
    # Row 0 = most vulnerable page, last row = least vulnerable page
    matrix = np.full((num_pages, 32768), -1, dtype=np.int8)
    
    print("Creating bitflip matrix...")
    for row_idx, (page_num, bitflip_count) in enumerate(selected_pages):
        if row_idx % 100 == 0:
            print(f"Processing page {row_idx + 1}/{num_pages}")
        
        # Get all bitflips for this page
        bitflips = page_bitflips[page_num]
        
        for bitflip in bitflips:
            page_offset = bitflip['page_offset']
            bit_position = bitflip['bit_position']
            flip_type = bitflip['flip_type']
            
            # Calculate bit index within the page
            # page_offset is byte offset, bit_position is bit within that byte
            bit_index = page_offset * 8 + bit_position
            
            # Ensure bit_index is within valid range
            if 0 <= bit_index < 32768:
                # Set the flip type (0 or 1)
                matrix[row_idx, bit_index] = flip_type
            else:
                print(f"Warning: Invalid bit_index {bit_index} for page {page_num}")
    
    print(f"Matrix shape: {matrix.shape}")
    print(f"Matrix data type: {matrix.dtype}")
    print(f"Values in matrix: {np.unique(matrix)}")
    
    # Count statistics
    total_elements = matrix.size
    non_flippable = np.sum(matrix == -1)
    flip_type_0 = np.sum(matrix == 0)
    flip_type_1 = np.sum(matrix == 1)
    
    print(f"\nMatrix statistics:")
    print(f"Total elements: {total_elements}")
    print(f"Non-flippable (-1): {non_flippable} ({non_flippable/total_elements*100:.2f}%)")
    print(f"Flip type 0: {flip_type_0} ({flip_type_0/total_elements*100:.2f}%)")
    print(f"Flip type 1: {flip_type_1} ({flip_type_1/total_elements*100:.2f}%)")
    
    # Save the matrix
    print(f"\nSaving matrix to {output_path}...")
    np.save(output_path, matrix)
    
    # Create metadata
    metadata = {
        'num_pages': num_pages,
        'bits_per_page': 32768,
        'page_info': [
            {
                'row_index': i,
                'phys_page_number': page_num,
                'bitflip_count': count
            }
            for i, (page_num, count) in enumerate(selected_pages)
        ]
    }
    
    return matrix, metadata

def main():
    parser = argparse.ArgumentParser(description="Create bitflip vulnerability matrix from DRAM profiling data")
    parser.add_argument('--profile', type=Path, default="device1_1G.json",
                       help='Path to DRAM profiling JSON file (default: device1_1G.json)')
    parser.add_argument('--output', type=Path, default="device1_1G.npy",
                       help='Output path for bitflip matrix (default: device1_1G.npy)')
    parser.add_argument('--metadata', type=Path,
                       help='Output path for metadata JSON (default: <output>_metadata.json)')
    
    args = parser.parse_args()
    
    # Set default metadata path if not specified
    if args.metadata is None:
        args.metadata = args.output.with_suffix('').with_name(args.output.stem + '_metadata.json')
    
    print(f"Input profile: {args.profile}")
    print(f"Output matrix: {args.output}")
    print(f"Output metadata: {args.metadata}")
    
    try:
        matrix, metadata = create_bitflip_matrix(str(args.profile), str(args.output))
        
        # Save metadata to specified path
        with open(args.metadata, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to {args.metadata}")
        
        print("\nBitflip matrix creation completed successfully!")
        
        # Print some sample information
        print(f"\nSample - First page (most vulnerable, row 0):")
        print(f"  Physical page number: {metadata['page_info'][0]['phys_page_number']}")
        print(f"  Bitflip count: {metadata['page_info'][0]['bitflip_count']}")
        print(f"  Flippable bits in this page: {np.sum(matrix[0] != -1)}")
        
        print(f"\nSample - Last page (least vulnerable, row {len(metadata['page_info'])-1}):")
        print(f"  Physical page number: {metadata['page_info'][-1]['phys_page_number']}")
        print(f"  Bitflip count: {metadata['page_info'][-1]['bitflip_count']}")
        print(f"  Flippable bits in this page: {np.sum(matrix[-1] != -1)}")
        
    except FileNotFoundError:
        print(f"Error: Could not find {args.profile}")
        print("Please make sure the DRAM profiling file exists")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())