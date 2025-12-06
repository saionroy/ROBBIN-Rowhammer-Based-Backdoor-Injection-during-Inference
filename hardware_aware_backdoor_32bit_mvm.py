#!/usr/bin/env python3
"""
Optimized MVM-based RowHammer attack implementation
Uses Matrix-Vector Multiplication for efficient DRAM page matching
"""

import struct
import math
import copy
import time
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn

# Dataclasses for configuration
@dataclass
class WeightedFP32Config:
    """Configuration for weighted FP32 bit scoring"""
    alpha_s: float = 1.0   # Weight for sign bits
    alpha_e: float = 10.0  # Weight for exponent bits
    alpha_m: float = 0.1   # Weight for mantissa bits

@dataclass
class RowHammerConfig:
    """Configuration for RowHammer-aware ProFlip attack with MVM optimization"""
    # DRAM page configuration
    page_size_bits: int = 32768  # 32K bits per page (4KB)
    page_start_offset: int = 0x0000  # Starting offset for first DNN page
    
    # MVM optimization parameters
    mvm_top_k_dram_candidates: int = 20  # Top K DRAM pages to test per DNN page
    
    # Attack parameters
    target_asr: float = 90.0  # Target attack success rate
    min_clean_acc_retention: float = 0.90  # Minimum accuracy retention (90%)
    min_clean_accuracy: float = 80.0  # Absolute minimum accuracy threshold
    
    # Salient neuron identification parameters
    sni_theta: float = 0.1  # Perturbation per step
    sni_gamma: float = 0.5  # Max fraction of perturbed features
    sni_num_features: int = 30  # Number of salient neurons to identify
    
    # Trigger generation parameters
    trigger_lr: float = 0.1  # Learning rate for trigger optimization
    trigger_epochs: int = 100  # Number of epochs for trigger optimization
    trigger_lambda1: float = 1.0  # Weight for classification loss
    trigger_lambda2: float = 1.0  # Weight for neuron activation loss
    trigger_c: float = 10.0  # Target activation value for salient neurons
    
    # Weighted FP32 parameters
    weighted_fp32_alpha_s: float = 1.0
    weighted_fp32_alpha_e: float = 10.0
    weighted_fp32_alpha_m: float = 0.1

@dataclass
class DRAMPage:
    """Represents a DRAM page with bit-flip capabilities from real profiling data"""
    page_id: int
    start_address: int
    end_address: int
    size_bits: int
    flip_capabilities: np.ndarray  # Shape: (32768,) with values -1 (no flip), 0 (0→1), 1 (1→0)

@dataclass
class DNNPage:
    """Represents a logical page of DNN parameters stored as 32-bit floats"""
    page_id: int
    start_param_idx: int
    end_param_idx: int
    parameters: Dict[str, Dict]  # module_name -> parameter metadata
    param_offsets: Dict[str, Tuple[int, int]]  # module_name -> (start_bit, end_bit)
    float_values: Dict[str, np.ndarray]  # module_name -> actual float32 values
    vulnerability_score: float = 0.0
    weighted_vulnerability_score: float = 0.0  # Score using weighted FP32 scheme
    bit_score_vector: Optional[np.ndarray] = None  # Bit-level scores for MVM
    max_achievable_score: float = 0.0  # Maximum score from MVM

@dataclass
class BitFlipCandidate:
    """Represents a candidate bit flip operation with weighted scoring"""
    module_name: str
    param_index: int
    bit_position: int  # 0-31 for float32
    original_value: float
    flipped_value: float
    gradient_magnitude: float
    expected_asr_gain: float
    flip_type: str  # 'sign', 'exponent', 'mantissa'
    flip_direction: int  # 0 for 0→1, 1 for 1→0
    weighted_score: float = 0.0  # Score using weighted FP32 scheme
    bit_position_in_field: int = 0  # Position within the field (eb for exponent, mb for mantissa)


# Helper classes
class WeightedFP32BitManipulator:
    """Enhanced bit manipulator with weighted FP32 scoring"""
    
    def __init__(self, config: WeightedFP32Config):
        self.config = config
        
    def calculate_weighted_score(self, gradient_magnitude: float, bit_type: str, 
                                bit_position_in_field: int) -> float:
        """
        Calculate weighted score based on equation:
        s(w,b) = αs·|gw| for sign
        s(w,b) = αe·|gw|·2^(-eb) for exponent
        s(w,b) = αm·|gw|·2^(-mb) for mantissa
        """
        abs_grad = abs(gradient_magnitude)
        
        if bit_type == 'sign':
            return self.config.alpha_s * abs_grad
        elif bit_type == 'exponent':
            # eb is position within exponent field (0-7 for FP32)
            return self.config.alpha_e * abs_grad * (2 ** (-bit_position_in_field))
        elif bit_type == 'mantissa':
            # mb is position within mantissa field (0-22 for FP32)
            return self.config.alpha_m * abs_grad * (2 ** (-bit_position_in_field))
        else:
            return 0.0
    
    def get_bit_position_in_field(self, bit_position: int) -> int:
        """Get the position within the specific field (exponent or mantissa)"""
        if bit_position == 0:  # Sign bit
            return 0
        elif 1 <= bit_position <= 8:  # Exponent bits
            return bit_position - 1  # eb = 0 to 7
        else:  # Mantissa bits (9-31)
            return bit_position - 9  # mb = 0 to 22


class Float32BitManipulator:
    """Utilities for manipulating bits in IEEE 754 float32 representation"""
    
    @staticmethod
    def float_to_bits(value: float) -> np.ndarray:
        """Convert float32 to its 32-bit representation"""
        bytes_repr = struct.pack('f', value)
        bits = np.unpackbits(np.frombuffer(bytes_repr, dtype=np.uint8))
        return bits
    
    @staticmethod
    def bits_to_float(bits: np.ndarray) -> float:
        """Convert 32-bit representation back to float32"""
        bytes_repr = np.packbits(bits).tobytes()
        return struct.unpack('f', bytes_repr)[0]
    
    @staticmethod
    def get_bit_type(bit_position: int) -> str:
        """Determine if a bit position is sign, exponent, or mantissa"""
        if bit_position == 0:  # MSB in our representation
            return 'sign'
        elif 1 <= bit_position <= 8:
            return 'exponent'
        else:
            return 'mantissa'
    
    @staticmethod
    def flip_bit(value: float, bit_position: int) -> Tuple[float, int]:
        """
        Flip a specific bit in float32 representation
        Returns: (new_value, flip_direction)
        """
        bits = Float32BitManipulator.float_to_bits(value)
        original_bit = bits[bit_position]
        bits[bit_position] = 1 - original_bit
        new_value = Float32BitManipulator.bits_to_float(bits)
        flip_direction = 0 if original_bit == 0 else 1
        return new_value, flip_direction
    
    @staticmethod
    def analyze_bit_flip_impact(value: float, bit_position: int) -> Dict:
        """Analyze the impact of flipping a specific bit"""
        new_value, flip_direction = Float32BitManipulator.flip_bit(value, bit_position)
        bit_type = Float32BitManipulator.get_bit_type(bit_position)
        
        # Check for special cases
        is_valid = np.isfinite(new_value)
        magnitude_change = abs(new_value - value) if is_valid else float('inf')
        
        return {
            'bit_position': bit_position,
            'bit_type': bit_type,
            'original_value': value,
            'new_value': new_value,
            'flip_direction': flip_direction,
            'is_valid': is_valid,
            'magnitude_change': magnitude_change,
            'relative_change': magnitude_change / (abs(value) + 1e-8)
        }


class PageInfoParser:
    """Parser for page_info.txt files"""
    def __init__(self, page_info_path: str, model):
        self.page_info_path = page_info_path
        self.model = model
        self.total_layers = 0
        self.total_pages = 0
        self.page_size = 0
        self.parameters = {}
        self.page_allocations = {}
        self.valid_parameters = set()
        self.parameter_mapping = {}
        
        self._build_parameter_mapping()
    
    def _build_parameter_mapping(self):
        """Build mapping between page_info names and actual model parameter names"""
        model_params = set(self.model.state_dict().keys())
        
        for param in model_params:
            self.parameter_mapping[param] = param
            alt_name = param.replace('.', '_')
            self.parameter_mapping[alt_name] = param
            alt_name = param.replace('_', '.')
            self.parameter_mapping[alt_name] = param
    
    def _is_valid_parameter(self, param_name: str) -> bool:
        """Check if a parameter from page_info.txt exists in the model"""
        skip_patterns = [
            'output.', 'quan_layer_branch', 'branch_layer',
            'running_mean', 'running_var', 'num_batches_tracked'
        ]
        for pattern in skip_patterns:
            if pattern in param_name:
                return False
        
        return param_name in self.parameter_mapping or param_name in self.model.state_dict()
    
    def _get_model_parameter_name(self, page_info_name: str) -> Optional[str]:
        """Get the actual model parameter name from page_info name"""
        if page_info_name in self.parameter_mapping:
            return self.parameter_mapping[page_info_name]
        
        if page_info_name in self.model.state_dict():
            return page_info_name
        
        return None
        
    def parse_page_info(self):
        """Parse the page_info.txt file"""
        with open(self.page_info_path, 'r') as f:
            lines = f.readlines()
        
        current_param = None
        current_param_info = {}
        skipped_params = []
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            if line.startswith('TOTAL_LAYERS'):
                self.total_layers = int(line.split()[1])
            elif line.startswith('TOTAL_PAGES'):
                self.total_pages = int(line.split()[1])
            elif line.startswith('PAGE_SIZE'):
                self.page_size = int(line.split()[1])
            
            elif line.startswith('PARAM'):
                parts = line.split()
                param_name = parts[1]
                
                if not self._is_valid_parameter(param_name):
                    skipped_params.append(param_name)
                    current_param = None
                    continue
                
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
                    'pages': [],
                    'model_name': self._get_model_parameter_name(param_name)
                }
                self.parameters[param_name] = current_param_info
                self.valid_parameters.add(param_name)
            
            elif line.startswith('PAGE') and current_param and current_param in self.valid_parameters:
                parts = line.split()
                page_id = int(parts[1])
                start_byte = int(parts[2])
                end_byte = int(parts[3])
                num_bytes = int(parts[4])
                
                current_param_info['pages'].append({
                    'page_id': page_id,
                    'start_byte': start_byte,
                    'end_byte': end_byte,
                    'num_bytes': num_bytes
                })
                
                if page_id not in self.page_allocations:
                    self.page_allocations[page_id] = []
                
                self.page_allocations[page_id].append({
                    'param_name': current_param,
                    'start_byte': start_byte,
                    'end_byte': end_byte,
                    'num_bytes': num_bytes
                })
        
        print(f"\nPage Info Parsing Summary:")
        print(f"  Total parameters in file: {self.total_layers}")
        print(f"  Valid parameters (exist in model): {len(self.valid_parameters)}")
        print(f"  Skipped parameters: {len(skipped_params)}")
    
    def create_dnn_pages_from_model(self, model) -> List[DNNPage]:
        """Create DNN pages based on page_info.txt structure"""
        self.parse_page_info()
        
        print(f"\nCreating DNN pages from validated parameters:")
        print(f"  Total pages: {self.total_pages}")
        print(f"  Page size: {self.page_size} bytes")
        print(f"  Valid parameters: {len(self.valid_parameters)}")
        
        model_params = {}
        model_state = model.state_dict()
        
        for param_name in model_state:
            param_data = model_state[param_name].cpu().numpy().flatten().astype(np.float32)
            model_params[param_name] = param_data
        
        pages = []
        for page_id in range(self.total_pages):
            page = DNNPage(
                page_id=page_id,
                start_param_idx=0,
                end_param_idx=0,
                parameters={},
                param_offsets={},
                float_values={}
            )
            
            if page_id in self.page_allocations:
                for allocation in self.page_allocations[page_id]:
                    param_name = allocation['param_name']
                    
                    if param_name not in self.valid_parameters:
                        continue
                    
                    start_byte = allocation['start_byte']
                    end_byte = allocation['end_byte']
                    num_bytes = allocation['num_bytes']
                    
                    num_floats = num_bytes // 4
                    
                    model_param_name = self.parameters[param_name].get('model_name')
                    if not model_param_name or model_param_name not in model_params:
                        continue
                    
                    param_data = model_params[model_param_name]
                    
                    param_info = self.parameters[param_name]
                    
                    # Ensure deterministic ordering of page segments for this parameter
                    ordered_segments = sorted(
                        param_info['pages'], key=lambda s: (s['page_id'], s.get('start_byte', 0))
                    )
                    param_offset = 0
                    for page_alloc in ordered_segments:
                        if page_alloc['page_id'] == page_id:
                            break
                        param_offset += page_alloc['num_bytes'] // 4
                    
                    if param_offset + num_floats <= len(param_data):
                        float_values = param_data[param_offset:param_offset + num_floats].copy()
                    else:
                        available = len(param_data) - param_offset
                        if available > 0:
                            float_values = np.zeros(num_floats, dtype=np.float32)
                            float_values[:available] = param_data[param_offset:]
                        else:
                            float_values = np.zeros(num_floats, dtype=np.float32)
                    
                    segment_key = f"{param_name}_p{page_id}_{start_byte}"
                    
                    page.parameters[segment_key] = {
                        'original_name': param_name,
                        'model_name': model_param_name,
                        'shape': param_info['shape'],
                        'float_offset': param_offset,
                        'num_floats': num_floats,
                        'is_partial': len(param_info['pages']) > 1,
                        'start_byte': start_byte,
                        'end_byte': end_byte
                    }
                    
                    start_bit = start_byte * 8
                    end_bit = (end_byte + 1) * 8
                    page.param_offsets[segment_key] = (start_bit, end_bit)
                    
                    page.float_values[segment_key] = float_values
            
            pages.append(page)
        
        non_empty_pages = sum(1 for p in pages if p.parameters)
        print(f"\nCreated {len(pages)} DNN pages")
        print(f"Pages with valid parameters: {non_empty_pages}")
        
        return pages


class MVMOptimizedDRAMSearch:
    """
    Matrix-Vector Multiplication based DRAM page search
    Implements the systematic optimization approach
    """
    
    def __init__(self, dram_profile: np.ndarray, page_size_bits: int = 32768):
        self.dram_profile = dram_profile  # Shape: (num_dram_pages, page_size_bits)
        self.page_size_bits = page_size_bits
        self.num_dram_pages = dram_profile.shape[0]
        
        # Pre-compute capability matrix (binary: can flip or not)
        # Convert -1 (no flip) to 0, and 0/1 (flip directions) to 1
        self.capability_matrix = (dram_profile != -1).astype(np.float32)
        
        print(f"Initialized MVM search with {self.num_dram_pages} DRAM pages")
        print(f"Capability matrix shape: {self.capability_matrix.shape}")
    
    def compute_mvm_scores(self, bit_score_vector: np.ndarray) -> np.ndarray:
        """
        Compute Matrix-Vector Multiplication: V × score_vector
        Returns achievable scores for each DRAM page
        """
        # Ensure score vector has correct shape
        if len(bit_score_vector) != self.page_size_bits:
            # Pad or truncate as needed
            if len(bit_score_vector) < self.page_size_bits:
                padded = np.zeros(self.page_size_bits, dtype=np.float32)
                padded[:len(bit_score_vector)] = bit_score_vector
                bit_score_vector = padded
            else:
                bit_score_vector = bit_score_vector[:self.page_size_bits]
        
        # Perform MVM: each row of capability matrix × score vector
        mvm_scores = self.capability_matrix @ bit_score_vector
        
        return mvm_scores
    
    def find_top_k_dram_pages(self, bit_score_vector: np.ndarray, 
                             k: int = 20, 
                             used_pages: Set[int] = None) -> List[Tuple[int, float]]:
        """
        Find top-k DRAM pages based on MVM scores
        Returns list of (dram_page_id, achievable_score)
        """
        if used_pages is None:
            used_pages = set()
        
        # Compute MVM scores
        mvm_scores = self.compute_mvm_scores(bit_score_vector)
        
        # Create list of (page_id, score) excluding used pages
        page_scores = [(i, score) for i, score in enumerate(mvm_scores) 
                      if i not in used_pages and score > 0]
        
        # Sort by score descending
        page_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k
        return page_scores[:k]
    
    def get_achievable_flips(self, dram_page_id: int, 
                            bit_positions: List[int]) -> List[Tuple[int, int]]:
        """
        Get which specific bit flips are achievable for a DRAM page
        Returns list of (bit_position, flip_direction)
        """
        achievable = []
        dram_capabilities = self.dram_profile[dram_page_id]
        
        for bit_pos in bit_positions:
            if bit_pos < len(dram_capabilities):
                capability = dram_capabilities[bit_pos]
                if capability != -1:  # Can flip
                    achievable.append((bit_pos, capability))
        
        return achievable


class MVMRowHammerProFlip:
    """
    MVM-optimized RowHammer-aware ProFlip attack
    Implements systematic DNN page ranking and DRAM matching
    """
    
    def __init__(self, config: RowHammerConfig, page_info_path: str = "page_info.txt"):
        self.config = config
        self.page_info_path = page_info_path
        self.dnn_pages: List[DNNPage] = []
        self.dram_profile = None
        self.mvm_searcher = None
        self.salient_neurons: List[int] = []
        self.trigger_pattern: Optional[torch.Tensor] = None
        self.bit_manipulator = Float32BitManipulator()
        self.parser = None
        
        # Initialize weighted FP32 configuration
        self.weighted_config = WeightedFP32Config(
            alpha_s=config.weighted_fp32_alpha_s,
            alpha_e=config.weighted_fp32_alpha_e,
            alpha_m=config.weighted_fp32_alpha_m
        )
        self.weighted_manipulator = WeightedFP32BitManipulator(self.weighted_config)
        self.ranked_pages = []  # Store ranked pages for visualization
    
    def compute_bit_score_vector(self, model: nn.Module, page: DNNPage,
                                data_loader, trigger_pattern, trigger_area,
                                target_class: int) -> np.ndarray:
        """
        Compute bit-level score vector for a DNN page
        Each element represents the importance score of flipping that bit
        """
        model.eval()
        start, end = trigger_area
        
        # Initialize score vector
        bit_scores = np.zeros(self.config.page_size_bits, dtype=np.float32)
        
        # Get sample data and compute gradients
        data_batch = None
        for batch_idx, (data, target) in enumerate(data_loader):
            if batch_idx == 0:
                # Use larger batch for more stable gradients
                take_n = min(64, data.size(0))
                data_batch = data[:take_n].cuda()
                break
        
        # Apply trigger and compute gradients
        data_triggered = data_batch.clone()
        data_triggered[:, :, start:end, start:end] = trigger_pattern
        target_class_tensor = torch.zeros(data_batch.size(0)).fill_(target_class).long().cuda()
        
        model.zero_grad()
        output = model(data_triggered)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target_class_tensor)
        loss.backward()
        
        # Analyze each parameter in the page
        for segment_key, float_values in page.float_values.items():
            param_info = page.parameters[segment_key]
            model_param_name = param_info.get('model_name')
            
            if not model_param_name:
                continue
            
            # Get parameter gradients
            param_grads = None
            if model_param_name in model.state_dict():
                param_parts = model_param_name.rsplit('.', 1)
                if len(param_parts) == 2:
                    module_name, param_type = param_parts
                    for name, module in model.named_modules():
                        if name == module_name:
                            if hasattr(module, param_type):
                                param = getattr(module, param_type)
                                if param is not None and hasattr(param, 'grad') and param.grad is not None:
                                    param_grads = param.grad.data.flatten()
                                    break
            
            if param_grads is None:
                continue
            
            # Extract relevant portion for partial parameters
            if param_info.get('is_partial', False):
                float_offset = param_info['float_offset']
                num_floats = param_info['num_floats']
                if float_offset + num_floats <= len(param_grads):
                    param_grads = param_grads[float_offset:float_offset + num_floats]
                else:
                    continue
            
            # Get bit offset for this segment
            start_bit, _ = page.param_offsets[segment_key]
            
            # Score each bit based on gradient and bit type
            for i in range(min(len(float_values), len(param_grads))):
                if abs(param_grads[i]) < 1e-8:  # Relax threshold; keep tiny but non-zero grads
                    continue
                
                current_value = float_values[i]
                grad_val = param_grads[i].item()
                
                # Analyze all 32 bits of this float
                for bit_pos in range(32):
                    global_bit_pos = start_bit + i * 32 + bit_pos
                    
                    if global_bit_pos >= self.config.page_size_bits:
                        continue
                    
                    # Check if flipping helps
                    analysis = self.bit_manipulator.analyze_bit_flip_impact(current_value, bit_pos)
                    
                    if not analysis['is_valid']:
                        continue
                    
                    new_value = analysis['new_value']
                    value_change = new_value - current_value
                    
                    # Beneficial if moves opposite to gradient
                    is_beneficial = (grad_val < 0 and value_change > 0) or (grad_val > 0 and value_change < 0)
                    
                    if is_beneficial:
                        # Correct for endianness: map bit_pos to actual IEEE754 position
                        byte_index = bit_pos // 8
                        bit_in_byte = bit_pos % 8
                        actual_bit_position = (3 - byte_index) * 8 + bit_in_byte
                        
                        # Determine bit type and position within field
                        if actual_bit_position == 31:
                            bit_type = 'sign'
                            pos_in_field = 0
                        elif 23 <= actual_bit_position <= 30:
                            bit_type = 'exponent'
                            pos_in_field = 30 - actual_bit_position  # 0..7
                        else:
                            bit_type = 'mantissa'
                            pos_in_field = 22 - actual_bit_position  # 0..22
                        
                        # Calculate weighted score
                        weighted_score = self.weighted_manipulator.calculate_weighted_score(
                            grad_val, bit_type, pos_in_field
                        )
                        
                        # Store in bit score vector
                        bit_scores[global_bit_pos] = weighted_score
        
        return bit_scores
    
    def rank_dnn_pages_by_mvm(self, model: nn.Module, pages: List[DNNPage],
                              data_loader, trigger_pattern, trigger_area,
                              target_class: int) -> List[DNNPage]:
        """
        Rank DNN pages by their maximum achievable score using MVM
        """
        print("\n--- Computing MVM-based DNN page rankings ---")
        
        # Compute bit score vectors and max achievable scores
        for page in pages:
            if not page.parameters:  # Skip empty pages
                page.max_achievable_score = 0.0
                continue
            
            # Compute bit-level score vector
            bit_score_vector = self.compute_bit_score_vector(
                model, page, data_loader, trigger_pattern, trigger_area, target_class
            )
            page.bit_score_vector = bit_score_vector
            
            # Compute MVM scores for all DRAM pages
            mvm_scores = self.mvm_searcher.compute_mvm_scores(bit_score_vector)
            
            # Maximum achievable score is the best DRAM page match
            page.max_achievable_score = np.max(mvm_scores)
            
            print(f"  DNN Page {page.page_id}: max achievable score = {page.max_achievable_score:.2f}")
        
        # Sort pages by max achievable score
        ranked_pages = sorted(pages, key=lambda p: p.max_achievable_score, reverse=True)
        
        # Filter out pages with zero score
        ranked_pages = [p for p in ranked_pages if p.max_achievable_score > 0]
        
        print(f"\nRanked {len(ranked_pages)} DNN pages by MVM scores")
        self.ranked_pages = ranked_pages  # Store for visualization
        return ranked_pages
    
    def apply_dram_page_flips(self, model: nn.Module, dnn_page: DNNPage, 
                             dram_page_id: int) -> Tuple[int, Dict[str, float]]:
        """
        Apply all achievable bit flips from a DRAM page to a DNN page
        """
        dram_capabilities = self.dram_profile[dram_page_id]
        current_bits = self.create_page_bitstream(dnn_page)
        
        # Track statistics
        total_flips = 0
        weighted_scores = {
            'sign': 0.0,
            'exponent': 0.0,
            'mantissa': 0.0,
            'total': 0.0
        }
        
        # Apply all flips the DRAM page can do
        for bit_pos in range(min(len(dram_capabilities), len(current_bits))):
            capability = dram_capabilities[bit_pos]
            
            if capability != -1:  # Can flip
                current_bits[bit_pos] = 1 - current_bits[bit_pos]
                total_flips += 1
                
                # Add weighted score from bit score vector
                if dnn_page.bit_score_vector is not None and bit_pos < len(dnn_page.bit_score_vector):
                    score = dnn_page.bit_score_vector[bit_pos]
                    
                    # Determine bit type for statistics
                    float_index = bit_pos // 32
                    bit_within_float = bit_pos % 32
                    
                    # Correct bit type determination
                    byte_index = bit_within_float // 8
                    bit_in_byte = bit_within_float % 8
                    actual_bit_position = (3 - byte_index) * 8 + bit_in_byte
                    
                    if actual_bit_position == 31:
                        weighted_scores['sign'] += score
                    elif 23 <= actual_bit_position <= 30:
                        weighted_scores['exponent'] += score
                    else:
                        weighted_scores['mantissa'] += score
                    
                    weighted_scores['total'] += score
        
        # Apply modified bits to model
        self.apply_bitstream_to_model(model, dnn_page, current_bits)
        
        return total_flips, weighted_scores
    
    def create_page_bitstream(self, page: DNNPage) -> np.ndarray:
        """Convert a DNN page to its binary representation"""
        bitstream = np.zeros(self.config.page_size_bits, dtype=np.uint8)
        
        for segment_key, float_values in page.float_values.items():
            start_bit, end_bit = page.param_offsets[segment_key]
            
            bit_offset = start_bit
            for float_val in float_values:
                float_bits = self.bit_manipulator.float_to_bits(float_val)
                bitstream[bit_offset:bit_offset + 32] = float_bits
                bit_offset += 32
        
        return bitstream
    
    def apply_bitstream_to_model(self, model: nn.Module, page: DNNPage, bitstream: np.ndarray):
        """Apply modified bitstream back to model parameters"""
        for segment_key, (start_bit, end_bit) in page.param_offsets.items():
            param_info = page.parameters[segment_key]
            model_param_name = param_info.get('model_name')
            
            if not model_param_name:
                continue
            
            # Extract bits for this parameter segment
            segment_bits = bitstream[start_bit:end_bit]
            num_floats = (end_bit - start_bit) // 32
            
            # Convert bits back to float values
            new_float_values = []
            for i in range(num_floats):
                float_bits = segment_bits[i*32:(i+1)*32]
                new_value = self.bit_manipulator.bits_to_float(float_bits)
                new_float_values.append(new_value)
            
            new_float_array = np.array(new_float_values, dtype=np.float32)
            
            # Apply to model
            param_parts = model_param_name.rsplit('.', 1)
            if len(param_parts) == 2:
                module_name, param_type = param_parts
                for name, module in model.named_modules():
                    if name == module_name and hasattr(module, param_type):
                        param = getattr(module, param_type)
                        if param is not None:
                            weight_flat = param.data.flatten()
                            
                            if param_info.get('is_partial', False):
                                float_offset = param_info['float_offset']
                                end_offset = float_offset + len(new_float_array)
                                if end_offset <= len(weight_flat):
                                    weight_flat[float_offset:end_offset] = torch.tensor(
                                        new_float_array, device=param.device, dtype=param.dtype)
                            else:
                                if len(new_float_array) == len(weight_flat):
                                    weight_flat[:] = torch.tensor(
                                        new_float_array, device=param.device, dtype=param.dtype)
                            
                            param.data = weight_flat.reshape(param.shape)
                        break
    
    def mvm_optimized_attack(self, model: nn.Module, data_loader, 
                            loader_search, target_class: int, trigger_area: Tuple[int, int],
                            profiling_file: str) -> Dict:
        """
        Main attack function using MVM optimization
        """
        print("=== Starting MVM-Optimized RowHammer Attack ===")
        print(f"Target ASR: {self.config.target_asr}%")
        print(f"MVM top-k candidates: {self.config.mvm_top_k_dram_candidates}")
        
        # Phase 1: Salient Neuron Identification
        print("\n--- Phase 1: Salient Neuron Identification ---")
        sample_data = None
        for data, _ in data_loader:
            sample_data = data[0:1].cuda()
            break
        
        start_time = time.time()
        self.salient_neurons = self.saliency_map(model, sample_data, target_class)
        sni_time = time.time() - start_time
        
        # Phase 2: Trigger Optimization
        print("\n--- Phase 2: Trigger Pattern Optimization ---")
        start_time = time.time()
        self.trigger_pattern = self.optimize_trigger(
            model, data_loader, trigger_area, target_class, self.salient_neurons
        )
        trigger_time = time.time() - start_time
        
        if self.trigger_pattern is None:
            print("Warning: Trigger optimization failed, using random trigger")
            start, end = trigger_area
            patch_size = end - start
            self.trigger_pattern = torch.randn(3, patch_size, patch_size).cuda() * 0.1
        
        initial_asr = self.evaluate_attack_success_rate(
            model, data_loader, self.trigger_pattern, trigger_area, target_class
        )
        baseline_acc = self.evaluate_clean_accuracy(model, data_loader)
        
        # Handle None values
        if initial_asr is None:
            initial_asr = 0.0
        if baseline_acc is None:
            baseline_acc = 0.0
        
        print(f"Initial trigger ASR: {initial_asr:.2f}%")
        print(f"Baseline accuracy: {baseline_acc:.2f}%")
        
        # Phase 3: Load DRAM Profile and Initialize MVM
        print("\n--- Phase 3: Loading DRAM Profile and Initializing MVM ---")
        self.load_dram_profiling_data(profiling_file)
        self.mvm_searcher = MVMOptimizedDRAMSearch(self.dram_profile, self.config.page_size_bits)
        
        # Phase 4: Create and Rank DNN Pages
        print("\n--- Phase 4: Creating and Ranking DNN Pages ---")
        self.create_dnn_pages(model, self.page_info_path)
        
        # Rank pages using MVM
        ranked_pages = self.rank_dnn_pages_by_mvm(
            model, self.dnn_pages, data_loader, 
            self.trigger_pattern, trigger_area, target_class
        )
        
        if not ranked_pages:
            return {
                'attack_successful': False,
                'final_asr': initial_asr,
                'initial_asr': initial_asr,
                'error': 'No vulnerable DNN pages found'
            }
        
        # Phase 5: Systematic Attack
        print("\n--- Phase 5: Systematic MVM-based Attack ---")
        
        # Attack state
        current_asr = initial_asr
        current_acc = baseline_acc
        best_model_state = copy.deepcopy(model.state_dict())
        used_dram_pages = set()
        successful_attacks = []
        
        # Statistics
        stats = {
            'total_bitflips': 0,
            'total_weighted_score': 0.0,
            'dnn_pages_tried': 0,
            'dram_candidates_tested': 0,
            'successful_mappings': 0
        }
        
        # Process DNN pages in ranked order
        for dnn_page in ranked_pages:
            if current_asr >= self.config.target_asr:
                print(f"\n✓ Target ASR reached: {current_asr:.2f}%")
                break
            
            if current_acc < self.config.min_clean_accuracy:
                print(f"\n✗ Accuracy dropped below threshold: {current_acc:.2f}%")
                break
            
            stats['dnn_pages_tried'] += 1
            print(f"\n--- Processing DNN Page {dnn_page.page_id} ---")
            print(f"Max achievable score: {dnn_page.max_achievable_score:.2f}")
            
            # Get top-k DRAM candidates using MVM
            dram_candidates = self.mvm_searcher.find_top_k_dram_pages(
                dnn_page.bit_score_vector,
                k=self.config.mvm_top_k_dram_candidates,
                used_pages=used_dram_pages
            )
            
            print(f"Testing {len(dram_candidates)} DRAM candidates...")
            
            # Try each DRAM candidate
            best_result = None
            
            for dram_page_id, achievable_score in dram_candidates:
                stats['dram_candidates_tested'] += 1
                
                print(f"\n  Testing DRAM page {dram_page_id} (achievable score: {achievable_score:.2f})")
                
                # Save state before attempting
                model.load_state_dict(best_model_state)
                
                try:
                    # Apply bit flips
                    num_flips, weighted_scores = self.apply_dram_page_flips(
                        model, dnn_page, dram_page_id
                    )
                    
                    # Build targeted vs collateral score arrays for the bits this DRAM page flips
                    targeted_scores = []
                    collateral_scores = []
                    if dnn_page.bit_score_vector is not None:
                        dram_caps = self.dram_profile[dram_page_id]
                        max_len = min(len(dram_caps), len(dnn_page.bit_score_vector))
                        for bit_pos in range(max_len):
                            if dram_caps[bit_pos] != -1:
                                s = float(dnn_page.bit_score_vector[bit_pos])
                                if s > 0.0:
                                    targeted_scores.append(s)
                                else:
                                    collateral_scores.append(s)
                    
                    # Evaluate
                    new_asr = self.evaluate_attack_success_rate(
                        model, data_loader, self.trigger_pattern, trigger_area, target_class
                    )
                    new_acc = self.evaluate_clean_accuracy(model, data_loader)
                    
                    print(f"    Flips: {num_flips}, ASR: {new_asr:.2f}%, Acc: {new_acc:.2f}%")
                    
                    # Check if this improves ASR without violating accuracy constraint
                    if new_asr > current_asr and new_acc >= self.config.min_clean_accuracy:
                        if best_result is None or new_asr > best_result['asr']:
                            best_result = {
                                'dram_page_id': dram_page_id,
                                'asr': new_asr,
                                'acc': new_acc,
                                'num_flips': num_flips,
                                'weighted_scores': weighted_scores,
                                'targeted_scores': targeted_scores,
                                'collateral_scores': collateral_scores,
                                'model_state': copy.deepcopy(model.state_dict())
                            }
                    
                except Exception as e:
                    print(f"    Error: {e}")
                    continue
            
            # Apply best result if found
            if best_result:
                print(f"\n  ✓ Best result: DRAM {best_result['dram_page_id']}")
                print(f"    ASR: {current_asr:.2f}% → {best_result['asr']:.2f}%")
                print(f"    Accuracy: {current_acc:.2f}% → {best_result['acc']:.2f}%")
                
                # Update state
                current_asr = best_result['asr']
                current_acc = best_result['acc']
                best_model_state = best_result['model_state']
                model.load_state_dict(best_model_state)
                used_dram_pages.add(best_result['dram_page_id'])
                
                # Update statistics
                stats['total_bitflips'] += best_result['num_flips']
                stats['total_weighted_score'] += best_result['weighted_scores']['total']
                stats['successful_mappings'] += 1
                
                # Record attack
                successful_attacks.append({
                    'dnn_page_id': dnn_page.page_id,
                    'dram_page_id': best_result['dram_page_id'],
                    'num_flips': best_result['num_flips'],
                    'asr': best_result['asr'],
                    'acc': best_result['acc'],
                    'weighted_score': best_result['weighted_scores']['total'],
                    'targeted_scores': best_result.get('targeted_scores', []),
                    'collateral_scores': best_result.get('collateral_scores', [])
                })
            else:
                print(f"\n  ✗ No beneficial DRAM page found")
        
        # Final results
        results = {
            'attack_successful': current_asr >= self.config.target_asr,
            'final_asr': current_asr,
            'initial_asr': initial_asr,
            'asr_improvement': current_asr - initial_asr,
            'target_asr': self.config.target_asr,
            'baseline_accuracy': baseline_acc,
            'final_accuracy': current_acc,
            'successful_attacks': successful_attacks,
            'statistics': stats,
            'execution_time': {
                'sni_seconds': sni_time,
                'trigger_seconds': trigger_time
            }
        }
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"MVM-OPTIMIZED ATTACK RESULTS")
        print(f"{'='*60}")
        print(f"Success: {'YES' if results['attack_successful'] else 'NO'}")
        print(f"Final ASR: {current_asr:.2f}% (target: {self.config.target_asr}%)")
        print(f"Final Accuracy: {current_acc:.2f}% (min: {self.config.min_clean_accuracy}%)")
        print(f"Total bit flips: {stats['total_bitflips']}")
        print(f"Total weighted score: {stats['total_weighted_score']:.2f}")
        print(f"DNN pages tried: {stats['dnn_pages_tried']}")
        print(f"DRAM candidates tested: {stats['dram_candidates_tested']}")
        print(f"Successful mappings: {stats['successful_mappings']}")
        
        return results
    
    # Support methods
    def saliency_map(self, model: nn.Module, input_tensor: torch.Tensor, target_class: int) -> List[int]:
        """Identify salient neurons using Jacobian-based Saliency Map Attack (JSMA)"""
        model.eval()
        
        # Find the final classification layer
        final_layer = None
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, nn.Linear):
                final_layer = module
                break
        
        if final_layer is None:
            print("Error: Could not find final classification layer")
            return list(range(min(self.config.sni_num_features, 10)))
        
        # Get activations before the final layer
        activation_cache = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                activation_cache[name] = output
            return hook
        
        # Register hooks
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.AdaptiveAvgPool2d)):
                hook = module.register_forward_hook(hook_fn(name))
                hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            output = model(input_tensor.cuda())
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Find the penultimate layer
        layer_names = list(activation_cache.keys())
        if len(layer_names) > 1:
            penultimate_layer = layer_names[-2]
        else:
            penultimate_layer = layer_names[-1] if layer_names else None
        
        if not penultimate_layer or penultimate_layer not in activation_cache:
            print("Warning: Could not identify penultimate layer")
            return list(range(min(self.config.sni_num_features, 10)))
        
        # Get activations
        penultimate_activations = activation_cache[penultimate_layer]
        
        if len(penultimate_activations.shape) == 4:
            penultimate_activations = penultimate_activations.mean(dim=(2, 3))
        
        num_features = penultimate_activations.size(1)
        
        # Get weights of final layer for target class
        if target_class >= final_layer.weight.size(0):
            target_class = min(target_class, final_layer.weight.size(0) - 1)
        
        target_weights = final_layer.weight[target_class].detach()
        
        if target_weights.size(0) != num_features:
            if target_weights.size(0) > num_features:
                target_weights = target_weights[:num_features]
            else:
                padding = torch.zeros(num_features - target_weights.size(0)).cuda()
                target_weights = torch.cat([target_weights, padding])
        
        # Compute saliency scores
        saliency_scores = torch.zeros(num_features).cuda()
        base_activations = penultimate_activations.clone()
        delta = 0.1
        
        with torch.no_grad():
            original_score = torch.matmul(base_activations, target_weights)
        
        batch_size = min(128, num_features)
        for batch_start in range(0, num_features, batch_size):
            batch_end = min(batch_start + batch_size, num_features)
            batch_features = torch.arange(batch_start, batch_end).cuda()
            
            perturbed_activations = base_activations.repeat(len(batch_features), 1)
            batch_indices = torch.arange(len(batch_features)).cuda()
            perturbed_activations[batch_indices, batch_features] += delta
            
            with torch.no_grad():
                perturbed_scores = torch.matmul(perturbed_activations, target_weights)
            
            saliency_scores[batch_start:batch_end] = perturbed_scores - original_score
        
        k = min(self.config.sni_num_features, num_features)
        _, salient_indices = torch.topk(saliency_scores, k)
        
        salient_neurons = salient_indices.cpu().numpy().tolist()
        
        print(f"Identified {len(salient_neurons)} salient neurons")
        
        return salient_neurons
    
    def optimize_trigger(self, model: nn.Module, data_loader, trigger_area: Tuple[int, int], 
                        target_class: int, salient_neurons: List[int]) -> torch.Tensor:
        """Optimize trigger pattern with dual objective"""
        model.eval()
        start, end = trigger_area
        patch_size = end - start
        
        print(f"Optimizing trigger for target class {target_class}")
        print(f"Salient neurons to target: {len(salient_neurons)} neurons")
        
        # Initialize trigger
        trigger = torch.randn(3, patch_size, patch_size).cuda() * 0.01
        trigger.requires_grad_(True)
        
        # Collect optimization data
        opt_data = []
        opt_targets = []
        class_counts = {i: 0 for i in range(10)}
        max_per_class = 10
        
        for batch_idx, (data, target) in enumerate(data_loader):
            for i in range(min(data.size(0), 100)):
                cls = target[i].item()
                if class_counts[cls] < max_per_class:
                    opt_data.append(data[i:i+1])
                    opt_targets.append(target[i:i+1])
                    class_counts[cls] += 1
            
            if sum(class_counts.values()) >= 100:
                break
        
        opt_data = torch.cat(opt_data, dim=0).cuda()
        opt_targets = torch.cat(opt_targets, dim=0).cuda()
        
        # Setup for neuron activation monitoring
        activation_cache = {}
        
        def get_activation(name):
            def hook(module, input, output):
                activation_cache[name] = output.detach()
            return hook
        
        # Find target layer
        hooks = []
        target_layer = None
        
        layer_info = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.AdaptiveAvgPool2d, nn.AvgPool2d)):
                layer_info.append((name, module))
        
        final_layer_idx = -1
        for i, (name, module) in enumerate(layer_info):
            if isinstance(module, nn.Linear):
                if hasattr(module, 'out_features') and (module.out_features == 10 or 'classifier' in name or 'fc' in name):
                    final_layer_idx = i
                    break
        
        if final_layer_idx > 0:
            target_layer_name, target_layer_module = layer_info[final_layer_idx - 1]
            target_layer = target_layer_name
            hook = target_layer_module.register_forward_hook(get_activation(target_layer_name))
            hooks.append(hook)
            print(f"Target layer for neuron activation: {target_layer_name}")
        else:
            print("Warning: Could not identify target layer for neuron activation")
        
        use_neuron_loss = (target_layer is not None) and salient_neurons and len(salient_neurons) > 0
        
        # Loss functions
        criterion_ce = nn.CrossEntropyLoss()
        criterion_mse = nn.MSELoss()
        
        # Optimization parameters
        initial_lr = self.config.trigger_lr
        min_lr = 0.001
        momentum = 0.9
        
        lambda1 = self.config.trigger_lambda1
        lambda2 = 0.1
        
        velocity = torch.zeros_like(trigger)
        
        best_asr = 0.0
        best_trigger = None
        validation_batch = opt_data[:32].clone()
        
        target_labels = torch.zeros(opt_data.size(0), dtype=torch.long).fill_(target_class).cuda()
        
        print(f"Starting trigger optimization...")
        
        for epoch in range(self.config.trigger_epochs):
            progress = epoch / self.config.trigger_epochs
            current_lr = max(min_lr, initial_lr * (1.0 - progress))
            
            if use_neuron_loss:
                lambda2_current = min(0.5, 0.1 + progress * 0.4)
            else:
                lambda2_current = 0.0
            
            batch_with_trigger = opt_data.clone()
            batch_with_trigger[:, :, start:end, start:end] = trigger
            
            outputs = model(batch_with_trigger)
            
            loss_classification = criterion_ce(outputs, target_labels)
            
            loss_neurons = torch.tensor(0.0).cuda()
            
            if use_neuron_loss and target_layer in activation_cache:
                neuron_activations = activation_cache[target_layer]
                
                if epoch == 0:
                    print(f"  Activation shape: {neuron_activations.shape}")
                
                if len(neuron_activations.shape) == 4:
                    neuron_activations = neuron_activations.mean(dim=(2, 3))
                    if epoch == 0:
                        print(f"  After spatial pooling: {neuron_activations.shape}")
                
                target_activations = neuron_activations.clone().detach()
                
                num_features = neuron_activations.size(1)
                valid_salient_neurons = [idx for idx in salient_neurons if idx < num_features]
                
                if epoch == 0:
                    print(f"  Valid salient neurons: {len(valid_salient_neurons)}/{len(salient_neurons)}")
                
                if len(valid_salient_neurons) > 0:
                    for neuron_idx in valid_salient_neurons:
                        target_activations[:, neuron_idx] += self.config.trigger_c
                    
                    valid_indices = torch.tensor(valid_salient_neurons, dtype=torch.long).cuda()
                    
                    salient_current = neuron_activations[:, valid_indices]
                    salient_target = target_activations[:, valid_indices]
                    loss_neurons = criterion_mse(salient_current, salient_target)
                else:
                    use_neuron_loss = False
                    if epoch == 0:
                        print("  Warning: No valid salient neurons for this layer")
            
            total_loss = lambda1 * loss_classification + lambda2_current * loss_neurons
            
            total_loss.backward()
            
            if trigger.grad is None:
                continue
            
            trigger_grad = trigger.grad.clone()
            grad_norm = torch.norm(trigger_grad)
            if grad_norm > 1.0:
                trigger_grad = trigger_grad / grad_norm
            
            velocity = momentum * velocity + current_lr * trigger_grad
            
            with torch.no_grad():
                trigger -= velocity
                trigger.clamp_(-1.0, 1.0)
            
            trigger.grad.zero_()
            
            activation_cache.clear()
            
            if (epoch + 1) % 10 == 0:
                with torch.no_grad():
                    val_with_trigger = validation_batch.clone()
                    val_with_trigger[:, :, start:end, start:end] = trigger
                    
                    val_outputs = model(val_with_trigger)
                    _, preds = val_outputs.max(1)
                    
                    val_targets = torch.zeros(val_with_trigger.size(0)).fill_(target_class).long().cuda()
                    current_asr = 100 * (preds == val_targets).float().mean().item()
                    
                    if current_asr > best_asr:
                        best_asr = current_asr
                        best_trigger = trigger.clone().detach()
                    
                    if epoch % 20 == 0:
                        print(f"  Epoch {epoch}: ASR = {current_asr:.2f}%")
        
        for hook in hooks:
            hook.remove()
        
        if best_trigger is not None and best_asr > 0:
            return best_trigger
        
        return trigger.detach()
    
    def create_dnn_pages(self, model: nn.Module, page_info_path: str) -> List[DNNPage]:
        """Create logical DNN pages using the parser"""
        print("Creating DNN pages...")
        
        self.parser = PageInfoParser(page_info_path, model)
        pages = self.parser.create_dnn_pages_from_model(model)
        
        self.dnn_pages = pages
        return pages
    
    def load_dram_profiling_data(self, profiling_file_path: str) -> List[DRAMPage]:
        """Load real DRAM profiling data from RowHammer characterization"""
        print(f"Loading DRAM profiling data from {profiling_file_path}...")
        
        try:
            profiling_matrix = np.load(profiling_file_path)
            print(f"Loaded profiling matrix with shape: {profiling_matrix.shape}")
            
            self.dram_profile = profiling_matrix
            
            expected_bits_per_page = self.config.page_size_bits
            actual_bits_per_page = profiling_matrix.shape[1]
            
            if actual_bits_per_page != expected_bits_per_page:
                print(f"Adjusting page size from {expected_bits_per_page} to {actual_bits_per_page} bits")
                self.config.page_size_bits = actual_bits_per_page
            
            num_pages = profiling_matrix.shape[0]
            dram_pages = []
            page_size_bytes = self.config.page_size_bits // 8
            
            print(f"Creating {num_pages} DRAM page objects...")
            for page_id in range(num_pages):
                start_addr = page_id * page_size_bytes
                end_addr = start_addr + page_size_bytes
                
                flip_capabilities = profiling_matrix[page_id]
                
                dram_page = DRAMPage(
                    page_id=page_id,
                    start_address=start_addr,
                    end_address=end_addr,
                    size_bits=self.config.page_size_bits,
                    flip_capabilities=flip_capabilities
                )
                dram_pages.append(dram_page)
            
            self.dram_pages = dram_pages
            
            print(f"Successfully loaded {len(dram_pages)} DRAM pages")
            
            return dram_pages
            
        except Exception as e:
            print(f"Error loading profiling data: {e}")
            raise
    
    def evaluate_attack_success_rate(self, model: nn.Module, data_loader,
                                   trigger_pattern, trigger_area, target_class: int) -> float:
        """Evaluate attack success rate with trigger"""
        if trigger_pattern is None:
            print("Warning: No trigger pattern provided")
            return 0.0
            
        start, end = trigger_area
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(data_loader):
                if batch_idx >= 10:
                    break
                    
                data = data.cuda()
                data[:, :, start:end, start:end] = trigger_pattern
                
                outputs = model(data)
                _, predicted = outputs.max(1)
                
                correct += (predicted == target_class).sum().item()
                total += data.size(0)
        
        return 100.0 * correct / max(1, total)
    
    def evaluate_clean_accuracy(self, model: nn.Module, data_loader) -> float:
        """Evaluate clean accuracy without trigger"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                if batch_idx >= 10:
                    break
                    
                data, target = data.cuda(), target.cuda()
                outputs = model(data)
                _, predicted = outputs.max(1)
                
                correct += predicted.eq(target).sum().item()
                total += target.size(0)
        
        return 100.0 * correct / max(1, total)


# Example usage
if __name__ == "__main__":
    # Configuration
    config = RowHammerConfig(
        target_asr=90.0,
        min_clean_accuracy=80.0,
        mvm_top_k_dram_candidates=20,
        weighted_fp32_alpha_s=1.0,
        weighted_fp32_alpha_e=10.0,
        weighted_fp32_alpha_m=0.1
    )
    
    # Initialize attack
    attacker = MVMRowHammerProFlip(config, page_info_path="page_info.txt")
    
    # Run MVM-optimized attack
    # results = attacker.mvm_optimized_attack(
    #     model=model,
    #     data_loader=test_loader,
    #     loader_search=train_loader,
    #     target_class=2,
    #     trigger_area=(21, 31),
    #     profiling_file="bitflip_matrix_A.npy"
    # )