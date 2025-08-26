#!/usr/bin/env python3
"""
hardware_aware_backdoor_8bit_mvm.py
===================================
Systematic MVM-based hardware-aware backdoor attack for INT8 models
Implements direction-aware matrix-vector multiplication for DNN page ranking
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import time
import os
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import struct
import copy
# Add this import at the top
import re

@dataclass
class INT8AttackConfig:
    """Configuration for hardware-aware INT8 attack with DRAM constraints"""
    # Attack parameters
    target_class: int = 2
    max_bit_flips: int = 100
    target_asr: float = 90.0
    alpha_s8: float = 0.5  # Importance reduction factor for sign bit
    
    # DRAM/RowHammer parameters
    page_size_bits: int = 32768  # 4KB pages = 32768 bits
    max_bitflips_per_page: int = 44  # RowHammer constraint
    bits_per_param: int = 8  # INT8
    
    # MVM parameters
    top_dram_pages_per_dnn: int = 20  # Top K DRAM pages to test per DNN page
    
    # Trigger parameters
    trigger_size: int = 10
    trigger_location: Tuple[int, int] = (21, 31)
    trigger_epochs: int = 100
    trigger_lr: float = 0.1
    
    # Adaptive thresholds
    min_clean_accuracy: float = 70.0
    max_accuracy_drop: float = 20.0
    min_absolute_accuracy: float = 80.0
    
    # New adaptive parameters
    force_progress: bool = True  # Continue attack even with accuracy drops
    adaptive_threshold_enabled: bool = True  # Use adaptive thresholds
    trigger_accuracy_tolerance: float = 0.90  # Allow 10% drop for trigger only
    progressive_relaxation_rate: float = 0.02  # Relax by 2% per stage
    asr_priority_threshold: float = 0.8  # When ASR > 80% of target, relax accuracy
    
    # Debug
    debug_mode: bool = False

@dataclass
class DRAMPage:
    """Represents a DRAM page containing model parameters"""
    page_id: int
    start_address: int
    end_address: int
    parameters: List[Dict]  # List of parameters in this page
    bit_flips_available: int
    vulnerability_score: float = 0.0

@dataclass
class DNNPage:
    """Represents a logical page of DNN parameters"""
    page_id: int
    start_param_idx: int
    end_param_idx: int
    parameters: Dict[str, Dict]  # module_name -> parameter metadata
    param_offsets: Dict[str, Tuple[int, int]]  # module_name -> (start_bit, end_bit)
    vulnerability_score: float = 0.0

class INT8BitManipulator:
    """Bit manipulation utilities for native INT8"""
    
    def __init__(self, alpha_s8: float = 0.5):
        self.alpha_s8 = alpha_s8
    
    def int8_to_bits(self, value: int) -> List[int]:
        """Convert INT8 value to 8-bit representation"""
        value = int(np.clip(value, -128, 127))
        unsigned = value & 0xFF
        return [(unsigned >> i) & 1 for i in range(8)]
    
    def bits_to_int8(self, bits: List[int]) -> int:
        """Convert 8 bits to INT8 value using two's complement"""
        unsigned = sum(bit << i for i, bit in enumerate(bits))
        if unsigned >= 128:
            return unsigned - 256
        return unsigned
    
    def compute_bit_importance_score(self, weight: float, bit_position: int, 
                                   gradient: float, scale_factor: float) -> float:
        """
        Compute importance score based on Equation (9):
        s_INT8(w, b) = { α_s8 * |g_w|  if b = 7 (sign bit)
                      { |g_w| * 2^b    if b ∈ {0, ..., 6}
        """
        abs_gradient = abs(gradient)
        
        if bit_position == 7:  # Sign bit (MSB)
            return self.alpha_s8 * abs_gradient
        else:  # Magnitude bits (0-6)
            return abs_gradient * (2 ** bit_position)

class DRAMPageMapper:
    """Maps model parameters to DRAM pages"""
    
    def __init__(self, page_size_bits: int = 32768):
        self.page_size_bits = page_size_bits
        self.page_size_bytes = page_size_bits // 8
    
    def create_synthetic_page_mapping(self, state_dict: Dict) -> Dict[int, DRAMPage]:
        """Create synthetic page mapping based on parameter layout"""
        pages = {}
        current_bit_offset = 0
        
        # Sort parameters by name for consistent mapping
        sorted_params = sorted([(k, v) for k, v in state_dict.items() 
                               if k.endswith('.weight') and v.dtype == torch.int8])
        
        for param_name, param in sorted_params:
            param_bits = param.numel() * 8  # 8 bits per INT8
            
            # Handle parameters that span multiple pages
            remaining_bits = param_bits
            param_start_idx = 0
            
            while remaining_bits > 0:
                page_id = current_bit_offset // self.page_size_bits
                page_offset = current_bit_offset % self.page_size_bits
                bits_in_current_page = min(remaining_bits, self.page_size_bits - page_offset)
                
                if page_id not in pages:
                    pages[page_id] = DRAMPage(
                        page_id=page_id,
                        start_address=page_id * self.page_size_bytes,
                        end_address=(page_id + 1) * self.page_size_bytes,
                        parameters=[],
                        bit_flips_available=44
                    )
                
                elements_in_page = bits_in_current_page // 8
                pages[page_id].parameters.append({
                    'name': param_name,
                    'start_bit': page_offset,
                    'end_bit': page_offset + bits_in_current_page,
                    'start_idx': param_start_idx,
                    'end_idx': param_start_idx + elements_in_page
                })
                
                current_bit_offset += bits_in_current_page
                remaining_bits -= bits_in_current_page
                param_start_idx += elements_in_page
        
        print(f"Created synthetic mapping with {len(pages)} pages")
        return pages

# Add a new PageInfoParser class after DRAMPageMapper
class PageInfoParser:
    """Parser for page_info.txt files to get actual DNN page mappings"""
    
    def __init__(self, page_info_path: str):
        self.page_info_path = page_info_path
        self.total_layers = 0
        self.total_pages = 0
        self.page_size = 0
        self.parameters = {}  # param_name -> {'shape': shape, 'size_bytes': size}
        self.page_allocations = {}  # page_id -> list of (param_name, start_byte, end_byte)
        
    def parse_page_info(self):
        """Parse the page_info.txt file"""
        with open(self.page_info_path, 'r') as f:
            lines = f.readlines()
        
        current_param = None
        current_param_info = {}
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            # Parse header info
            if line.startswith('TOTAL_LAYERS'):
                self.total_layers = int(line.split()[1])
            elif line.startswith('TOTAL_PAGES'):
                self.total_pages = int(line.split()[1])
            elif line.startswith('PAGE_SIZE'):
                self.page_size = int(line.split()[1])
            
            # Parse parameter definition
            elif line.startswith('PARAM'):
                parts = line.split()
                param_name = parts[1]
                
                # Parse shape - handle both tensor shapes and scalars
                if parts[2] == 'scalar':
                    shape = ()
                    size_bytes = int(parts[3])
                else:
                    shape_str = parts[2]
                    shape = tuple(map(int, shape_str.split('x')))
                    size_bytes = int(parts[3])
                
                current_param = param_name
                current_param_info = {
                    'shape': shape,
                    'size_bytes': size_bytes,
                    'pages': []
                }
                self.parameters[param_name] = current_param_info
            
            # Parse page allocation
            elif line.startswith('PAGE') and current_param:
                parts = line.split()
                page_id = int(parts[1])
                start_byte = int(parts[2])
                end_byte = int(parts[3])
                num_bytes = int(parts[4])
                
                # Add to parameter's page list
                current_param_info['pages'].append({
                    'page_id': page_id,
                    'start_byte': start_byte,
                    'end_byte': end_byte,
                    'num_bytes': num_bytes
                })
                
                # Add to page allocations
                if page_id not in self.page_allocations:
                    self.page_allocations[page_id] = []
                
                self.page_allocations[page_id].append({
                    'param_name': current_param,
                    'start_byte': start_byte,
                    'end_byte': end_byte,
                    'num_bytes': num_bytes
                })
        
        print(f"Parsed page info: {self.total_layers} parameters, {self.total_pages} pages")
        return self.page_allocations, self.parameters
    
    def create_dnn_pages_from_page_info(self, int8_state_dict) -> Dict[int, DRAMPage]:
        """Create DNN pages based on actual page_info.txt mappings"""
        page_allocations, parameters = self.parse_page_info()
        
        pages = {}
        
        for page_id in range(self.total_pages):
            pages[page_id] = DRAMPage(
                page_id=page_id,
                start_address=page_id * self.page_size,
                end_address=(page_id + 1) * self.page_size,
                parameters=[],
                bit_flips_available=44
            )
            
            # Get allocations for this page
            if page_id in page_allocations:
                for allocation in page_allocations[page_id]:
                    param_name = allocation['param_name']
                    
                    # Check if parameter exists in INT8 state dict
                    if param_name not in int8_state_dict:
                        # Try alternative naming (e.g., with module prefix)
                        found = False
                        for key in int8_state_dict.keys():
                            if key.endswith(param_name) or param_name in key:
                                param_name = key
                                found = True
                                break
                        if not found:
                            continue
                    
                    if param_name in int8_state_dict and int8_state_dict[param_name].dtype == torch.int8:
                        param = int8_state_dict[param_name]
                        
                        # Calculate parameter indices for this page segment
                        start_byte = allocation['start_byte']
                        end_byte = allocation['end_byte']
                        num_bytes = allocation['num_bytes']
                        
                        # For INT8, each element is 1 byte
                        start_idx = 0
                        # Find the starting index by counting bytes from previous pages
                        if param_name in parameters and len(parameters[param_name]['pages']) > 1:
                            for prev_page in parameters[param_name]['pages']:
                                if prev_page['page_id'] < page_id:
                                    start_idx += prev_page['num_bytes']
                                elif prev_page['page_id'] == page_id:
                                    break
                        
                        end_idx = start_idx + num_bytes
                        
                        pages[page_id].parameters.append({
                            'name': param_name,
                            'start_bit': start_byte * 8,  # Convert to bits
                            'end_bit': end_byte * 8,
                            'start_idx': start_idx,
                            'end_idx': end_idx,
                            'start_byte': start_byte,
                            'end_byte': end_byte
                        })
        
        # Filter out empty pages
        non_empty_pages = {pid: page for pid, page in pages.items() if page.parameters}
        print(f"Created {len(non_empty_pages)} non-empty DNN pages from page_info.txt")
        
        return non_empty_pages

class MVMHardwareAwareAttack:
    """
    Systematic MVM-based hardware-aware backdoor attack with direction awareness
    """
    
    def __init__(self, config: INT8AttackConfig):
        self.config = config
        self.bit_manipulator = INT8BitManipulator(config.alpha_s8)
        self.page_mapper = DRAMPageMapper(config.page_size_bits)
        self.trigger_pattern = None
        self.dram_pages = {}
        self.dram_profile = None
        self.attack_progress = []
        
        # MVM specific attributes
        self.vulnerability_matrix_0to1 = None  # V_0→1 matrix
        self.vulnerability_matrix_1to0 = None  # V_1→0 matrix
        self.dnn_page_scores = {}  # Score vectors for each DNN page
        self.dnn_page_rankings = []  # DNN pages sorted by max realizable score
    
    def evaluate_model_clean_only(self, model, data_loader, num_batches=50):
        """Evaluate model on CLEAN data only (no trigger)"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(data_loader):
                if batch_idx >= num_batches:
                    break
                
                data, targets = data.cuda(), targets.cuda()
                
                # Clean accuracy only
                outputs = model(data)
                if isinstance(outputs, list):
                    outputs = outputs[-1]
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total += data.size(0)
        
        accuracy = 100.0 * correct / total
        return accuracy

    def load_dram_profile_with_directions(self, profiling_file: str):
        """Load DRAM profiling data and create direction-aware vulnerability matrices"""
        print(f"\nLoading DRAM profiling data from: {profiling_file}")
        try:
            self.dram_profile = np.load(profiling_file)
            print(f"Loaded DRAM profile with shape: {self.dram_profile.shape}")
            
            # Create two vulnerability matrices for each flip direction
            # V_0to1[i,j] = 1 if DRAM page i can do 0→1 flip at bit j
            # V_1to0[i,j] = 1 if DRAM page i can do 1→0 flip at bit j
            
            self.vulnerability_matrix_0to1 = np.zeros_like(self.dram_profile, dtype=np.float32)
            self.vulnerability_matrix_1to0 = np.zeros_like(self.dram_profile, dtype=np.float32)
            
            # DRAM profile encoding:
            # -1 = cannot flip
            #  0 = can do 0→1 flip
            #  1 = can do 1→0 flip
            
            self.vulnerability_matrix_0to1[self.dram_profile == 0] = 1.0
            self.vulnerability_matrix_1to0[self.dram_profile == 1] = 1.0
            
            print(f"Created direction-aware vulnerability matrices")
            print(f"  0→1 capabilities: {np.sum(self.vulnerability_matrix_0to1):,.0f} bits")
            print(f"  1→0 capabilities: {np.sum(self.vulnerability_matrix_1to0):,.0f} bits")
            
            # Statistics
            flip_counts_0to1 = np.sum(self.vulnerability_matrix_0to1, axis=1)
            flip_counts_1to0 = np.sum(self.vulnerability_matrix_1to0, axis=1)
            flip_counts_total = flip_counts_0to1 + flip_counts_1to0
            
            print(f"\nDRAM Statistics:")
            print(f"  Total DRAM pages: {len(flip_counts_total)}")
            print(f"  Average total flips per page: {np.mean(flip_counts_total):.1f}")
            print(f"  Average 0→1 flips per page: {np.mean(flip_counts_0to1):.1f}")
            print(f"  Average 1→0 flips per page: {np.mean(flip_counts_1to0):.1f}")
            
        except Exception as e:
            print(f"Error loading DRAM profile: {e}")
            raise
    
    def compute_bit_importance_scores_with_direction(self, int8_state_dict, gradients, 
                                                   scale_factors, dnn_pages):
        """
        Compute importance scores with flip direction awareness
        Returns: dict mapping page_id -> (score_vector_0to1, score_vector_1to0)
        """
        print("\nComputing direction-aware bit importance scores for all DNN pages...")
        
        page_score_vectors = {}
        
        for page_id, dnn_page in dnn_pages.items():
            # Two score vectors for each flip direction
            score_vector_0to1 = np.zeros(self.config.page_size_bits, dtype=np.float32)
            score_vector_1to0 = np.zeros(self.config.page_size_bits, dtype=np.float32)
            
            bit_candidates = []
            
            for param_info in dnn_page.parameters:
                param_name = param_info['name']
                
                if param_name not in int8_state_dict or param_name not in gradients:
                    continue
                
                param = int8_state_dict[param_name]
                gradient = gradients[param_name]
                scale_factor = scale_factors.get(param_name, 1.0)
                
                if param.dtype != torch.int8:
                    continue
                
                flat_param = param.view(-1)
                flat_grad = gradient.view(-1)
                
                # Process each weight in this parameter segment
                start_idx = param_info['start_idx']
                end_idx = min(param_info['end_idx'], flat_param.numel())
                
                for idx in range(start_idx, end_idx):
                    if idx >= len(flat_grad):
                        continue
                    
                    int8_value = flat_param[idx].item()
                    grad_value = flat_grad[idx].item()
                    
                    if abs(grad_value) < 1e-6:
                        continue
                    
                    current_bits = self.bit_manipulator.int8_to_bits(int8_value)
                    
                    for bit_pos in range(8):
                        # Get current bit value
                        current_bit = current_bits[bit_pos]
                        
                        # Compute importance score
                        score = self.bit_manipulator.compute_bit_importance_score(
                            int8_value, bit_pos, grad_value, scale_factor
                        )
                        
                        # Check if flipping helps
                        flipped_bits = current_bits.copy()
                        flipped_bits[bit_pos] = 1 - flipped_bits[bit_pos]
                        new_int8 = self.bit_manipulator.bits_to_int8(flipped_bits)
                        
                        value_change = (new_int8 - int8_value) * scale_factor
                        is_beneficial = (grad_value < 0 and value_change > 0) or \
                                      (grad_value > 0 and value_change < 0)
                        
                        if is_beneficial:
                            # Calculate absolute bit position in the page
                            relative_idx = idx - start_idx
                            absolute_bit_pos = param_info['start_bit'] + (relative_idx * 8) + bit_pos
                            
                            if absolute_bit_pos < self.config.page_size_bits:
                                # Store score in appropriate vector based on current bit value
                                if current_bit == 0:
                                    # This bit needs 0→1 flip
                                    score_vector_0to1[absolute_bit_pos] = score
                                else:
                                    # This bit needs 1→0 flip
                                    score_vector_1to0[absolute_bit_pos] = score
                                
                                bit_candidates.append({
                                    'param_name': param_name,
                                    'weight_index': idx,
                                    'bit_position': bit_pos,
                                    'absolute_bit_pos': absolute_bit_pos,
                                    'score': score,
                                    'current_bit': current_bit,
                                    'required_flip': '0→1' if current_bit == 0 else '1→0',
                                    'current_int8': int8_value,
                                    'gradient': grad_value,
                                    'value_change': value_change
                                })
            
            page_score_vectors[page_id] = {
                'score_vector_0to1': score_vector_0to1,
                'score_vector_1to0': score_vector_1to0,
                'bit_candidates': bit_candidates,
                'num_critical_bits_0to1': np.sum(score_vector_0to1 > 0),
                'num_critical_bits_1to0': np.sum(score_vector_1to0 > 0),
                'num_critical_bits_total': np.sum(score_vector_0to1 > 0) + np.sum(score_vector_1to0 > 0)
            }
        
        print(f"Computed score vectors for {len(page_score_vectors)} DNN pages")
        
        # Statistics
        total_0to1 = sum(info['num_critical_bits_0to1'] for info in page_score_vectors.values())
        total_1to0 = sum(info['num_critical_bits_1to0'] for info in page_score_vectors.values())
        print(f"Total critical bits identified:")
        print(f"  0→1 flips needed: {total_0to1}")
        print(f"  1→0 flips needed: {total_1to0}")
        print(f"  Total: {total_0to1 + total_1to0}")
        
        return page_score_vectors
    
    def compute_adaptive_accuracy_threshold(self, initial_accuracy: float, current_asr: float, 
                                       total_flips: int, num_pages_used: int) -> float:
        """
        Compute adaptive accuracy threshold based on attack progress
        """
        if not self.config.adaptive_threshold_enabled:
            return self.config.min_clean_accuracy
        
        # Stage 1: Trigger only (be very lenient)
        if total_flips == 0:
            threshold = initial_accuracy * self.config.trigger_accuracy_tolerance
            if self.config.debug_mode:
                print(f"    [Adaptive] Trigger-only stage: threshold = {threshold:.1f}%")
            return threshold
        
        # Stage 2: Early bit flips (progressive relaxation)
        if num_pages_used <= 3:
            # Start with 95% retention, decrease by 2% per page
            retention_rate = 0.95 - (self.config.progressive_relaxation_rate * num_pages_used)
            threshold = initial_accuracy * retention_rate
            if self.config.debug_mode:
                print(f"    [Adaptive] Early stage ({num_pages_used} pages): threshold = {threshold:.1f}%")
            return max(self.config.min_absolute_accuracy, threshold)
        
        # Stage 3: Check ASR progress
        asr_progress = current_asr / self.config.target_asr
        if asr_progress >= self.config.asr_priority_threshold:
            # Close to target ASR, can afford lower accuracy
            threshold = self.config.min_clean_accuracy * 0.95
            if self.config.debug_mode:
                print(f"    [Adaptive] Near target ASR ({current_asr:.1f}%): threshold = {threshold:.1f}%")
            return threshold
        
        # Stage 4: Check if making good progress
        if num_pages_used > 5:
            asr_per_page = (current_asr - initial_accuracy) / num_pages_used
            if asr_per_page > 5.0:  # Good progress
                threshold = self.config.min_clean_accuracy
            else:  # Slow progress, relax constraints
                threshold = self.config.min_clean_accuracy * 0.90
            
            if self.config.debug_mode:
                print(f"    [Adaptive] Progress-based (ASR/page={asr_per_page:.1f}): threshold = {threshold:.1f}%")
            return threshold
        
        # Default: use configured minimum
        return self.config.min_clean_accuracy

    def rank_dnn_pages_by_direction_aware_mvm(self, page_score_vectors):
        """
        Rank DNN pages using direction-aware MVM
        """
        print("\nRanking DNN pages using Direction-Aware Matrix-Vector Multiplication...")
        
        dnn_page_rankings = []
        
        for page_id, score_info in page_score_vectors.items():
            score_vector_0to1 = score_info['score_vector_0to1']
            score_vector_1to0 = score_info['score_vector_1to0']
            
            if score_info['num_critical_bits_total'] == 0:
                continue
            
            # Compute direction-aware MVM
            # For each DRAM page, compute scores from both flip directions
            realizable_scores_0to1 = self.vulnerability_matrix_0to1 @ score_vector_0to1
            realizable_scores_1to0 = self.vulnerability_matrix_1to0 @ score_vector_1to0
            
            # Combined realizable scores
            total_realizable_scores = realizable_scores_0to1 + realizable_scores_1to0
            
            # Find top DRAM pages
            top_indices = np.argsort(total_realizable_scores)[::-1][:self.config.top_dram_pages_per_dnn]
            top_scores = total_realizable_scores[top_indices]
            
            # Maximum realizable score
            max_score = np.max(total_realizable_scores)
            max_idx = np.argmax(total_realizable_scores)
            
            # Store detailed breakdown
            dnn_page_rankings.append({
                'page_id': page_id,
                'max_realizable_score': max_score,
                'max_score_0to1': realizable_scores_0to1[max_idx],
                'max_score_1to0': realizable_scores_1to0[max_idx],
                'top_dram_pages': top_indices.tolist(),
                'top_dram_scores': top_scores.tolist(),
                'top_dram_scores_0to1': realizable_scores_0to1[top_indices].tolist(),
                'top_dram_scores_1to0': realizable_scores_1to0[top_indices].tolist(),
                'num_critical_bits_0to1': score_info['num_critical_bits_0to1'],
                'num_critical_bits_1to0': score_info['num_critical_bits_1to0'],
                'num_critical_bits_total': score_info['num_critical_bits_total'],
                'score_info': score_info
            })
        
        # Sort by maximum realizable score
        dnn_page_rankings.sort(key=lambda x: x['max_realizable_score'], reverse=True)
        
        print(f"\nDNN Page Rankings (top 10):")
        for i, ranking in enumerate(dnn_page_rankings[:10]):
            print(f"  {i+1}. Page {ranking['page_id']}: "
                  f"max_score={ranking['max_realizable_score']:.2f} "
                  f"(0→1: {ranking['max_score_0to1']:.2f}, 1→0: {ranking['max_score_1to0']:.2f}), "
                  f"critical_bits={ranking['num_critical_bits_total']} "
                  f"(0→1: {ranking['num_critical_bits_0to1']}, 1→0: {ranking['num_critical_bits_1to0']})")
        
        return dnn_page_rankings
    
    def apply_bit_flips_with_direction_check(self, int8_state_dict, dnn_page, dram_page_id, 
                                            score_info):
        """
        Apply bit flips ensuring direction compatibility
        """
        dram_capabilities = self.dram_profile[dram_page_id]
        modified_dict = copy.deepcopy(int8_state_dict)
        
        total_flips = 0
        flips_0to1 = 0
        flips_1to0 = 0
        flips_applied = []
        direction_mismatches = 0
        
        # Get bit candidates with their required flip directions
        bit_candidates = score_info['bit_candidates']
        candidate_map = {c['absolute_bit_pos']: c for c in bit_candidates}
        
        # Count total possible flips
        total_possible = np.sum(dram_capabilities != -1)
        
        # For each parameter in this DNN page
        for param_info in dnn_page.parameters:
            param_name = param_info['name']
            if param_name not in modified_dict:
                continue
                
            if modified_dict[param_name].dtype != torch.int8:
                continue
            
            param = modified_dict[param_name]
            flat_param = param.view(-1)
            
            # Check each weight in this parameter segment
            for idx in range(param_info['start_idx'], min(param_info['end_idx'], flat_param.numel())):
                # For each bit position
                for bit_pos in range(8):
                    # Calculate absolute bit position in DRAM page
                    relative_idx = idx - param_info['start_idx']
                    absolute_bit_pos = param_info['start_bit'] + (relative_idx * 8) + bit_pos
                    
                    if absolute_bit_pos >= len(dram_capabilities):
                        continue
                    
                    # Check if DRAM can flip this bit
                    dram_flip_type = dram_capabilities[absolute_bit_pos]
                    
                    if dram_flip_type == -1:
                        continue  # DRAM cannot flip this bit
                    
                    # Get current bit value
                    current_int8 = flat_param[idx].item()
                    current_bit = (current_int8 >> bit_pos) & 1
                    
                    # Determine what flip would happen
                    if dram_flip_type == 0:  # DRAM can do 0→1
                        if current_bit == 0:
                            # Can flip! Check if it's beneficial
                            if absolute_bit_pos in candidate_map:
                                candidate = candidate_map[absolute_bit_pos]
                                if candidate['required_flip'] == '0→1':
                                    # Perfect match!
                                    current_bits = self.bit_manipulator.int8_to_bits(current_int8)
                                    current_bits[bit_pos] = 1
                                    new_int8 = self.bit_manipulator.bits_to_int8(current_bits)
                                    
                                    flat_param[idx] = new_int8
                                    total_flips += 1
                                    flips_0to1 += 1
                                    
                                    flips_applied.append({
                                        'param': param_name,
                                        'idx': idx,
                                        'bit_pos': bit_pos,
                                        'old_val': current_int8,
                                        'new_val': new_int8,
                                        'score': candidate['score'],
                                        'flip_type': '0→1',
                                        'beneficial': True
                                    })
                                else:
                                    direction_mismatches += 1
                            else:
                                # Not a critical bit, but DRAM will flip it anyway
                                current_bits = self.bit_manipulator.int8_to_bits(current_int8)
                                current_bits[bit_pos] = 1
                                new_int8 = self.bit_manipulator.bits_to_int8(current_bits)
                                
                                flat_param[idx] = new_int8
                                total_flips += 1
                                flips_0to1 += 1
                                
                                flips_applied.append({
                                    'param': param_name,
                                    'idx': idx,
                                    'bit_pos': bit_pos,
                                    'old_val': current_int8,
                                    'new_val': new_int8,
                                    'score': 0,
                                    'flip_type': '0→1',
                                    'beneficial': False
                                })
                    
                    elif dram_flip_type == 1:  # DRAM can do 1→0
                        if current_bit == 1:
                            # Can flip! Check if it's beneficial
                            if absolute_bit_pos in candidate_map:
                                candidate = candidate_map[absolute_bit_pos]
                                if candidate['required_flip'] == '1→0':
                                    # Perfect match!
                                    current_bits = self.bit_manipulator.int8_to_bits(current_int8)
                                    current_bits[bit_pos] = 0
                                    new_int8 = self.bit_manipulator.bits_to_int8(current_bits)
                                    
                                    flat_param[idx] = new_int8
                                    total_flips += 1
                                    flips_1to0 += 1
                                    
                                    flips_applied.append({
                                        'param': param_name,
                                        'idx': idx,
                                        'bit_pos': bit_pos,
                                        'old_val': current_int8,
                                        'new_val': new_int8,
                                        'score': candidate['score'],
                                        'flip_type': '1→0',
                                        'beneficial': True
                                    })
                                else:
                                    direction_mismatches += 1
                            else:
                                # Not a critical bit, but DRAM will flip it anyway
                                current_bits = self.bit_manipulator.int8_to_bits(current_int8)
                                current_bits[bit_pos] = 0
                                new_int8 = self.bit_manipulator.bits_to_int8(current_bits)
                                
                                flat_param[idx] = new_int8
                                total_flips += 1
                                flips_1to0 += 1
                                
                                flips_applied.append({
                                    'param': param_name,
                                    'idx': idx,
                                    'bit_pos': bit_pos,
                                    'old_val': current_int8,
                                    'new_val': new_int8,
                                    'score': 0,
                                    'flip_type': '1→0',
                                    'beneficial': False
                                })
            
            # Update the parameter
            modified_dict[param_name] = flat_param.view(param.shape)
        
        # Count beneficial vs non-beneficial flips
        beneficial_flips = sum(1 for f in flips_applied if f['beneficial'])
        non_beneficial_flips = sum(1 for f in flips_applied if not f['beneficial'])
        
        print(f"    Applied {total_flips} bit flips from DRAM page {dram_page_id}:")
        print(f"      0→1 flips: {flips_0to1}")
        print(f"      1→0 flips: {flips_1to0}")
        print(f"      Beneficial flips: {beneficial_flips}")
        print(f"      Non-beneficial flips: {non_beneficial_flips}")
        print(f"      Direction mismatches avoided: {direction_mismatches}")
        
        return modified_dict, total_flips, flips_applied
    
    def systematic_mvm_attack(self, int8_state_dict, model, test_loader, trigger_pattern, 
                         scale_factors, gradients):
        """
        Execute systematic MVM-based attack with direction awareness and adaptive thresholds
        """
        print("\n" + "="*60)
        print("Executing Systematic Direction-Aware MVM Attack")
        print("WITH ADAPTIVE THRESHOLDS" if self.config.adaptive_threshold_enabled else "")
        print("="*60)
        
        # Initialize
        M = []  # Selected mappings
        used_dram_pages = set()
        original_int8_state = copy.deepcopy(int8_state_dict)
        original_model_state = copy.deepcopy(model.state_dict())
        
        # Initial evaluation
        initial_acc, initial_asr = self.evaluate_model(
            model, test_loader, trigger_pattern, self.config.target_class, num_batches=50
        )
        
        print(f"\nInitial state: Accuracy={initial_acc:.2f}%, ASR={initial_asr:.2f}%")
        
        # Current best state
        current_asr = initial_asr
        current_acc = initial_acc
        best_int8_state = copy.deepcopy(original_int8_state)
        best_model_state = copy.deepcopy(original_model_state)
        
        # Track progress
        self.attack_progress = [{
            'flips': 0,
            'asr': initial_asr,
            'acc': initial_acc,
            'pages': 0
        }]
        
        total_flips_applied = 0
        consecutive_failures = 0
        last_good_state = copy.deepcopy(model.state_dict())
        last_good_asr = current_asr
        
        # Iterate through DNN pages by MVM ranking
        for ranking_idx, dnn_ranking in enumerate(self.dnn_page_rankings):
            # Check termination conditions
            if current_asr >= self.config.target_asr:
                print(f"\n✓ SUCCESS: Target ASR reached: {current_asr:.2f}% >= {self.config.target_asr}%")
                break
            
            # Compute adaptive accuracy threshold
            adaptive_threshold = self.compute_adaptive_accuracy_threshold(
                initial_acc, current_asr, total_flips_applied, len(M)
            )
            
            # Only stop if accuracy is catastrophically low
            if current_acc < self.config.min_absolute_accuracy * 0.9 and not self.config.force_progress:
                print(f"\n✗ STOP: Accuracy critically low: {current_acc:.2f}%")
                break
            
            page_id = dnn_ranking['page_id']
            dnn_page = self.dram_pages[page_id]
            score_info = dnn_ranking['score_info']
            
            print(f"\n--- Testing DNN Page {page_id} (rank {ranking_idx+1}/{len(self.dnn_page_rankings)}) ---")
            print(f"  Max realizable score: {dnn_ranking['max_realizable_score']:.2f}")
            print(f"  Critical bits: {dnn_ranking['num_critical_bits_total']} "
                f"(0→1: {dnn_ranking['num_critical_bits_0to1']}, 1→0: {dnn_ranking['num_critical_bits_1to0']})")
            print(f"  Current accuracy threshold: {adaptive_threshold:.1f}% (original: {self.config.min_clean_accuracy:.1f}%)")
            
            # Try top DRAM pages for this DNN page
            best_dram_result = None
            tested_count = 0
            
            # Expand search if having consecutive failures
            search_expansion = 1 + (consecutive_failures // 2)
            num_dram_to_test = min(
                len(dnn_ranking['top_dram_pages']), 
                self.config.top_dram_pages_per_dnn * search_expansion
            )
            
            for dram_idx in range(num_dram_to_test):
                dram_page_id = dnn_ranking['top_dram_pages'][dram_idx]
                
                if dram_page_id in used_dram_pages:
                    continue
                
                tested_count += 1
                dram_score = dnn_ranking['top_dram_scores'][dram_idx]
                dram_score_0to1 = dnn_ranking['top_dram_scores_0to1'][dram_idx]
                dram_score_1to0 = dnn_ranking['top_dram_scores_1to0'][dram_idx]
                dram_capability = np.sum(self.dram_profile[dram_page_id] != -1)
                
                print(f"\n  Testing DRAM page {dram_page_id} "
                    f"(score={dram_score:.2f} [0→1: {dram_score_0to1:.2f}, 1→0: {dram_score_1to0:.2f}], "
                    f"capability={dram_capability})")
                
                # Apply bit flips with direction checking
                test_int8_state = copy.deepcopy(best_int8_state)
                modified_int8_state, actual_flips, flip_details = self.apply_bit_flips_with_direction_check(
                    test_int8_state, dnn_page, dram_page_id, score_info
                )
                
                if actual_flips == 0:
                    print(f"    No flips applied - skipping")
                    continue
                
                # Load and evaluate
                model.load_state_dict(original_model_state)
                modified_dequantized, _ = self.dequantize_int8_weights(modified_int8_state)
                model_dict = model.state_dict()
                
                for name, param in modified_dequantized.items():
                    if name in model_dict and model_dict[name].shape == param.shape:
                        model_dict[name].copy_(param)
                
                # Reset quantization
                for m in model.modules():
                    if hasattr(m, '__reset_stepsize__'):
                        m.__reset_stepsize__()
                    if hasattr(m, '__reset_weight__'):
                        m.__reset_weight__()
                
                model.eval()
                
                # Evaluate
                test_acc, test_asr = self.evaluate_model(
                    model, test_loader, trigger_pattern, self.config.target_class, num_batches=50
                )
                
                print(f"    Results: ASR={test_asr:.2f}%, Accuracy={test_acc:.2f}%")
                
                # Adaptive acceptance criteria
                asr_improvement = test_asr - current_asr
                accept = False
                
                if test_acc >= adaptive_threshold:
                    # Meets adaptive threshold
                    if asr_improvement > 0:
                        accept = True
                        print(f"    ✓ Accepted: Meets adaptive threshold")
                elif test_acc >= adaptive_threshold * 0.95:  # Within 5% of threshold
                    if asr_improvement > 10:  # Significant ASR improvement
                        accept = True
                        print(f"    ✓ Accepted: Significant ASR gain (+{asr_improvement:.1f}%) with acceptable accuracy")
                    elif asr_improvement > 5 and consecutive_failures > 2:
                        accept = True
                        print(f"    ✓ Accepted: Moderate ASR gain with relaxed threshold due to failures")
                elif self.config.force_progress and test_acc >= self.config.min_absolute_accuracy * 0.85:
                    if asr_improvement > 15:  # Very significant ASR improvement
                        accept = True
                        print(f"    ⚠️ Force accepted: Major ASR improvement (+{asr_improvement:.1f}%)")
                
                if not accept:
                    print(f"    ✗ Rejected: Accuracy {test_acc:.2f}% below threshold {adaptive_threshold:.1f}%")
                
                # Track best result
                if accept and (best_dram_result is None or test_asr > best_dram_result['asr']):
                    best_dram_result = {
                        'dram_page_id': dram_page_id,
                        'asr': test_asr,
                        'acc': test_acc,
                        'asr_improvement': asr_improvement,
                        'flips': actual_flips,
                        'int8_state': copy.deepcopy(modified_int8_state),
                        'model_state': copy.deepcopy(model.state_dict()),
                        'flip_details': flip_details
                    }
            
            # Apply best result if any
            if best_dram_result is not None:
                print(f"\n  ✓ Selected DRAM page {best_dram_result['dram_page_id']}: "
                    f"ASR {current_asr:.2f}% → {best_dram_result['asr']:.2f}% "
                    f"(+{best_dram_result['asr_improvement']:.2f}%)")
                
                # Update state
                current_asr = best_dram_result['asr']
                current_acc = best_dram_result['acc']
                best_int8_state = best_dram_result['int8_state']
                best_model_state = best_dram_result['model_state']
                total_flips_applied += best_dram_result['flips']
                
                # Reset consecutive failures
                consecutive_failures = 0
                last_good_state = copy.deepcopy(best_model_state)
                last_good_asr = current_asr
                
                # Count flip types
                beneficial_count = sum(1 for f in best_dram_result['flip_details'] if f['beneficial'])
                flips_0to1 = sum(1 for f in best_dram_result['flip_details'] if f['flip_type'] == '0→1')
                flips_1to0 = sum(1 for f in best_dram_result['flip_details'] if f['flip_type'] == '1→0')
                
                # Record mapping
                M.append({
                    'dnn_page': page_id,
                    'dram_page': best_dram_result['dram_page_id'],
                    'num_flips': best_dram_result['flips'],
                    'beneficial_flips': beneficial_count,
                    'flips_0to1': flips_0to1,
                    'flips_1to0': flips_1to0,
                    'asr': best_dram_result['asr'],
                    'acc': best_dram_result['acc']
                })
                
                used_dram_pages.add(best_dram_result['dram_page_id'])
                
                # Track progress
                self.attack_progress.append({
                    'flips': total_flips_applied,
                    'asr': current_asr,
                    'acc': current_acc,
                    'pages': len(M)
                })
            else:
                consecutive_failures += 1
                print(f"\n  ✗ No beneficial DRAM page found (tested {tested_count} pages)")
                print(f"  Consecutive failures: {consecutive_failures}")
                
                # Recovery strategies
                if consecutive_failures > 3 and self.config.force_progress:
                    print("  ⚠️ Applying recovery strategies:")
                    
                    # Strategy 1: Further relax threshold
                    self.config.min_clean_accuracy *= 0.95
                    print(f"    - Relaxed min_clean_accuracy to {self.config.min_clean_accuracy:.1f}%")
                    
                    # Strategy 2: Skip pages with low scores
                    if consecutive_failures > 5 and ranking_idx < len(self.dnn_page_rankings) - 10:
                        print("    - Skipping to higher-scored pages")
                        continue
        
        # Restore best state
        model.load_state_dict(best_model_state)
        
        # Summary
        print("\n" + "="*60)
        print("Attack Summary")
        print("="*60)
        print(f"Initial ASR: {initial_asr:.2f}%")
        print(f"Final ASR: {current_asr:.2f}% (improvement: +{current_asr - initial_asr:.2f}%)")
        print(f"Final Accuracy: {current_acc:.2f}%")
        print(f"Target ASR: {self.config.target_asr}%")
        print(f"Attack Success: {'YES' if current_asr >= self.config.target_asr else 'NO'}")
        print(f"Total bit flips applied: {total_flips_applied}")
        print(f"DRAM pages used: {len(M)}")
        
        if M:
            print(f"\nSuccessful mappings:")
            for i, mapping in enumerate(M):
                print(f"  {i+1}. DNN page {mapping['dnn_page']} → DRAM page {mapping['dram_page']}: "
                    f"{mapping['num_flips']} flips ({mapping['beneficial_flips']} beneficial), "
                    f"0→1: {mapping['flips_0to1']}, 1→0: {mapping['flips_1to0']}, "
                    f"ASR={mapping['asr']:.2f}%, Acc={mapping['acc']:.2f}%")
        
        return M, current_asr, current_acc
    
    def execute_attack(self, int8_state_dict, model, test_loader, attack_loader, 
                  scale_factors, page_info_file=None, profiling_file=None):
        """
        Execute complete systematic MVM-based attack with direction awareness
        """
        results = {
            'start_time': time.time(),
            'initial_accuracy': 0,
            'final_accuracy': 0,
            'trigger_asr': 0,
            'final_asr': 0,
            'attack_successful': False,
            'attack_progress': []
        }
        
        # Load DRAM profiling data with direction awareness
        if profiling_file and os.path.exists(profiling_file):
            self.load_dram_profile_with_directions(profiling_file)
        else:
            print("ERROR: DRAM profiling file required")
            return results
        
        # Create DRAM page mapping from page_info file
        if page_info_file and os.path.exists(page_info_file):
            print(f"\nUsing actual page mapping from: {page_info_file}")
            parser = PageInfoParser(page_info_file)
            self.dram_pages = parser.create_dnn_pages_from_page_info(int8_state_dict)
        else:
            print("\nWARNING: No page_info_file provided, using synthetic mapping")
            self.dram_pages = self.page_mapper.create_synthetic_page_mapping(int8_state_dict)
        
        # Initial evaluation
        print("\nInitial evaluation...")
        initial_acc, _ = self.evaluate_model(model, test_loader, num_batches=50)
        results['initial_accuracy'] = initial_acc
        print(f"Initial accuracy: {initial_acc:.2f}%")
        
        # Generate trigger
        trigger_pattern = self.generate_trigger_pattern(
            model, attack_loader, self.config.target_class
        )
        
        # Evaluate trigger only
        _, trigger_asr = self.evaluate_model(
            model, test_loader, trigger_pattern, self.config.target_class, num_batches=50
        )
        results['trigger_asr'] = trigger_asr
        print(f"Trigger-only ASR: {trigger_asr:.2f}%")
        
        # Compute gradients
        print("\nComputing weight gradients...")
        gradients = self.compute_weight_gradients(
            model, attack_loader, self.config.target_class, trigger_pattern
        )
        
        # Step 1: Compute direction-aware bit importance scores
        self.dnn_page_scores = self.compute_bit_importance_scores_with_direction(
            int8_state_dict, gradients, scale_factors, self.dram_pages
        )
        
        # Step 2: Rank DNN pages by direction-aware MVM
        self.dnn_page_rankings = self.rank_dnn_pages_by_direction_aware_mvm(
            self.dnn_page_scores
        )
        
        # Step 3: Execute systematic attack
        mappings, final_asr, final_acc = self.systematic_mvm_attack(
            int8_state_dict, model, test_loader, trigger_pattern, scale_factors, gradients
        )
        
        # Store results
        results['final_asr'] = final_asr
        results['final_accuracy'] = final_acc
        results['attack_successful'] = final_asr >= self.config.target_asr
        results['attack_progress'] = self.attack_progress
        results['attack_mappings'] = mappings
        results['total_time'] = time.time() - results['start_time']
        
        print(f"\nTotal execution time: {results['total_time']:.2f} seconds")
        
        return results
    
    # Include necessary helper methods
    def dequantize_int8_weights(self, int8_state_dict):
        """Dequantize INT8 weights for model inference"""
        dequantized_dict = {}
        scale_factors = {}
        
        for key, value in int8_state_dict.items():
            if key.endswith('.weight') and value.dtype == torch.int8:
                scale_key = key.replace('.weight', '.scale_factor')
                if scale_key in int8_state_dict:
                    scale_factor = int8_state_dict[scale_key]
                    dequantized_weight = value.float() * scale_factor
                    dequantized_dict[key] = dequantized_weight
                    scale_factors[key] = scale_factor.item()
                else:
                    dequantized_dict[key] = value.float()
                    scale_factors[key] = 1.0
            elif 'scale_factor' not in key:
                dequantized_dict[key] = value
        
        return dequantized_dict, scale_factors
    
    def compute_weight_gradients(self, model, data_loader, target_class, trigger_pattern=None):
        """Compute gradients for all weights w.r.t. backdoor objective"""
        model.train()
        gradients = {}
        
        optimizer = torch.optim.SGD(model.parameters(), lr=0)
        num_batches = 20
        
        for batch_idx, (data, _) in enumerate(data_loader):
            if batch_idx >= num_batches:
                break
            
            data = data.cuda()
            
            # Apply trigger
            if trigger_pattern is not None:
                h_pos, w_pos = self.config.trigger_location
                trigger_size = self.config.trigger_size
                h_end = h_pos + 1
                w_end = w_pos + 1
                h_start = h_end - trigger_size
                w_start = w_end - trigger_size
                
                h_start = max(0, h_start)
                w_start = max(0, w_start)
                h_end = min(32, h_end)
                w_end = min(32, w_end)
                
                data[:, :, h_start:h_end, w_start:w_end] = trigger_pattern
            
            optimizer.zero_grad()
            
            outputs = model(data)
            if isinstance(outputs, list):
                outputs = outputs[-1]
            
            target_labels = torch.full((data.size(0),), target_class, dtype=torch.long).cuda()
            loss = nn.CrossEntropyLoss()(outputs, target_labels)
            
            loss.backward()
            
            # Accumulate gradients
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if name not in gradients:
                        gradients[name] = param.grad.data.clone()
                    else:
                        gradients[name] += param.grad.data
        
        # Average gradients
        for name in gradients:
            gradients[name] /= num_batches
        
        model.eval()
        
        return gradients
    
    def generate_trigger_pattern(self, model, data_loader, target_class):
        """Generate optimized trigger pattern"""
        print("\nGenerating trigger pattern...")
        
        # Get trigger configuration
        h_pos, w_pos = self.config.trigger_location
        trigger_size = self.config.trigger_size
        
        h_end = h_pos + 1
        w_end = w_pos + 1
        h_start = h_end - trigger_size
        w_start = w_end - trigger_size
        
        h_start = max(0, h_start)
        w_start = max(0, w_start)
        h_end = min(32, h_end)
        w_end = min(32, w_end)
        
        # Initialize trigger
        trigger = torch.zeros(3, trigger_size, trigger_size).cuda()
        trigger[0, :, :] = 1.0
        trigger[1, :trigger_size//2, :] = 1.0
        trigger[2, :, :trigger_size//2] = 1.0
        trigger = trigger * 0.5
        trigger.requires_grad = True
        
        optimizer = torch.optim.Adam([trigger], lr=self.config.trigger_lr)
        
        best_asr = 0
        best_trigger = None
        
        for epoch in range(self.config.trigger_epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (data, _) in enumerate(data_loader):
                if batch_idx >= 10:
                    break
                
                data = data.cuda()
                
                # Apply trigger
                triggered_data = data.clone()
                triggered_data[:, :, h_start:h_end, w_start:w_end] = trigger
                
                # Forward pass
                outputs = model(triggered_data)
                if isinstance(outputs, list):
                    outputs = outputs[-1]
                
                # Loss
                target_labels = torch.full((data.size(0),), target_class, dtype=torch.long).cuda()
                loss = nn.CrossEntropyLoss()(outputs, target_labels)
                
                # Regularization
                reg_loss = 0.01 * (trigger.pow(2).mean() + (trigger - 0.5).pow(2).mean())
                total_loss += (loss + reg_loss).item()
                
                # Optimize
                optimizer.zero_grad()
                (loss + reg_loss).backward()
                optimizer.step()
                
                # Clip trigger
                with torch.no_grad():
                    trigger.clamp_(0, 1)
                
                # Calculate ASR
                _, predicted = outputs.max(1)
                correct += predicted.eq(target_class).sum().item()
                total += data.size(0)
            
            current_asr = 100.0 * correct / total
            
            if current_asr > best_asr:
                best_asr = current_asr
                best_trigger = trigger.clone().detach()
            
            if epoch % 20 == 0:
                print(f"  Epoch {epoch}: ASR = {current_asr:.2f}%")
        
        self.trigger_pattern = best_trigger
        print(f"  Final trigger ASR: {best_asr:.2f}%")
        
        return best_trigger
    
    def evaluate_model(self, model, data_loader, trigger_pattern=None, target_class=None, 
                      num_batches=50):
        """Evaluate model accuracy and ASR"""
        model.eval()
        
        correct_clean = 0
        correct_triggered = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(data_loader):
                if batch_idx >= num_batches:
                    break
                
                data, targets = data.cuda(), targets.cuda()
                
                # Clean accuracy
                outputs = model(data)
                if isinstance(outputs, list):
                    outputs = outputs[-1]
                _, predicted = outputs.max(1)
                correct_clean += predicted.eq(targets).sum().item()
                
                # Triggered accuracy (ASR)
                if trigger_pattern is not None and target_class is not None:
                    triggered_data = data.clone()
                    
                    h_pos, w_pos = self.config.trigger_location
                    trigger_size = self.config.trigger_size
                    h_end = h_pos + 1
                    w_end = w_pos + 1
                    h_start = h_end - trigger_size
                    w_start = w_end - trigger_size
                    
                    h_start = max(0, h_start)
                    w_start = max(0, w_start)
                    h_end = min(32, h_end)
                    w_end = min(32, w_end)
                    
                    triggered_data[:, :, h_start:h_end, w_start:w_end] = trigger_pattern
                    
                    outputs_triggered = model(triggered_data)
                    if isinstance(outputs_triggered, list):
                        outputs_triggered = outputs_triggered[-1]
                    _, predicted_triggered = outputs_triggered.max(1)
                    correct_triggered += predicted_triggered.eq(target_class).sum().item()
                
                total += data.size(0)
        
        clean_acc = 100.0 * correct_clean / total
        asr = 100.0 * correct_triggered / total if trigger_pattern is not None else 0
        
        return clean_acc, asr