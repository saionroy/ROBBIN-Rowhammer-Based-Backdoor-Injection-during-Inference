#ifndef RESNET20_QUAN_H
#define RESNET20_QUAN_H

#include <vector>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <memory>
#include <iostream>
#include <fstream>

// Tensor class for managing multi-dimensional arrays
class Tensor {
public:
    std::vector<float> data;
    std::vector<int> shape;
    int total_size;
    
    Tensor() : total_size(0) {}
    
    Tensor(const std::vector<int>& shape_) : shape(shape_) {
        total_size = 1;
        for (int dim : shape) {
            total_size *= dim;
        }
        data.resize(total_size, 0.0f);
    }
    
    // Get element at specific indices
    float& at(const std::vector<int>& indices) {
        int flat_idx = 0;
        int stride = 1;
        for (int i = shape.size() - 1; i >= 0; i--) {
            flat_idx += indices[i] * stride;
            stride *= shape[i];
        }
        return data[flat_idx];
    }
    
    const float& at(const std::vector<int>& indices) const {
        int flat_idx = 0;
        int stride = 1;
        for (int i = shape.size() - 1; i >= 0; i--) {
            flat_idx += indices[i] * stride;
            stride *= shape[i];
        }
        return data[flat_idx];
    }
    
    // Reshape tensor
    void reshape(const std::vector<int>& new_shape) {
        int new_size = 1;
        for (int dim : new_shape) {
            new_size *= dim;
        }
        if (new_size != total_size) {
            throw std::runtime_error("Invalid reshape dimensions");
        }
        shape = new_shape;
    }
    
    // Load from binary file
    void load_from_file(std::ifstream& file) {
        // Read number of dimensions
        uint32_t num_dims;
        file.read(reinterpret_cast<char*>(&num_dims), sizeof(uint32_t));
        
        // Read shape
        shape.clear();
        shape.resize(num_dims);
        for (uint32_t i = 0; i < num_dims; i++) {
            uint32_t dim;
            file.read(reinterpret_cast<char*>(&dim), sizeof(uint32_t));
            shape[i] = dim;
        }
        
        // Calculate total size
        total_size = 1;
        for (int dim : shape) {
            total_size *= dim;
        }
        
        // Read data
        data.resize(total_size);
        file.read(reinterpret_cast<char*>(data.data()), total_size * sizeof(float));
    }
};

// Quantization function
inline float quantize_value(float value, float step_size, float half_levels) {
    float max_val = half_levels * step_size;
    float min_val = -max_val;
    
    // Clamp
    value = std::max(min_val, std::min(max_val, value));
    
    // Round to nearest quantization level
    return std::round(value / step_size) * step_size;
}

// Conv2D operation with quantization
class QuanConv2D {
public:
    Tensor weight;
    float step_size;
    int in_channels, out_channels, kernel_size, stride, padding;
    
    QuanConv2D(int in_ch, int out_ch, int k_size, int str = 1, int pad = 0)
        : in_channels(in_ch), out_channels(out_ch), kernel_size(k_size), 
          stride(str), padding(pad), step_size(1.0f) {}
    
    Tensor forward(const Tensor& input);
};

// Linear layer with quantization
class QuanLinear {
public:
    Tensor weight;
    Tensor bias;
    float step_size;
    int in_features, out_features;
    
    QuanLinear(int in_feat, int out_feat)
        : in_features(in_feat), out_features(out_feat), step_size(1.0f) {}
    
    Tensor forward(const Tensor& input);
};

// BatchNorm2D
class BatchNorm2D {
public:
    Tensor weight;  // gamma
    Tensor bias;   // beta
    Tensor running_mean;
    Tensor running_var;
    float eps;
    int num_features;
    
    BatchNorm2D(int num_feat, float epsilon = 1e-5f)
        : num_features(num_feat), eps(epsilon) {}
    
    Tensor forward(const Tensor& input);
};

// ReLU activation
inline void relu_inplace(Tensor& input) {
    for (float& val : input.data) {
        val = std::max(0.0f, val);
    }
}

// Average pooling
Tensor avg_pool2d(const Tensor& input, int kernel_size, int stride = -1);

// ResNet Basic Block
class ResNetBasicBlock {
public:
    QuanConv2D conv_a;
    BatchNorm2D bn_a;
    QuanConv2D conv_b;
    BatchNorm2D bn_b;
    bool has_downsample;
    int downsample_stride;
    
    ResNetBasicBlock(int in_planes, int planes, int stride = 1);
    Tensor forward(const Tensor& input);
};

// ResNet20 Model
class ResNet20Quan {
public:
    QuanConv2D conv_1_3x3;
    BatchNorm2D bn_1;
    
    std::vector<ResNetBasicBlock> stage_1;
    std::vector<ResNetBasicBlock> stage_2;
    std::vector<ResNetBasicBlock> stage_3;
    
    QuanLinear classifier;
    
public:
    // Normalization parameters
    std::vector<float> mean{0.4914f, 0.4822f, 0.4465f};  // Will be loaded
    std::vector<float> std{0.2470f, 0.2435f, 0.2616f};   // Will be loaded
    ResNet20Quan();
    
    // Load weights from binary file
    // If norm_file is empty or cannot be opened, the loader will try to read
    // mean[3] and std[3] appended to the end of the weight file.
    bool load_weights(const std::string& weight_file, const std::string& norm_file);
    
    // Forward pass
    std::vector<float> forward(const Tensor& input);
    
    // Predict single image
    int predict(const Tensor& input);
};

// CIFAR-10 data loader
class CIFAR10Loader {
private:
    std::vector<Tensor> images;
    std::vector<uint8_t> labels;
    int num_images;
    
public:
    bool load_from_binary(const std::string& data_file);
    
    int size() const { return num_images; }
    
    Tensor get_image(int idx) const { return images[idx]; }
    uint8_t get_label(int idx) const { return labels[idx]; }
};

#endif // RESNET20_QUAN_H 