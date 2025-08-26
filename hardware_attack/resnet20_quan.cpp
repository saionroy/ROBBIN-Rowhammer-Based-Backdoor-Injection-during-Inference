#include "resnet20_quan.h"
#include <stdexcept>
#include <cstdint>

// Conv2D forward implementation
Tensor QuanConv2D::forward(const Tensor& input) {
    // Input shape: [batch, channels, height, width]
    int batch_size = input.shape[0];
    int in_height = input.shape[2];
    int in_width = input.shape[3];
    
    int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
    
    Tensor output({batch_size, out_channels, out_height, out_width});
    
    // Quantization parameters
    const float N_bits = 8.0f;
    const float full_levels = std::pow(2.0f, N_bits);
    const float half_levels = (full_levels - 2.0f) / 2.0f;
    
    // Perform convolution
    for (int b = 0; b < batch_size; b++) {
        for (int oc = 0; oc < out_channels; oc++) {
            for (int oh = 0; oh < out_height; oh++) {
                for (int ow = 0; ow < out_width; ow++) {
                    float sum = 0.0f;
                    
                    for (int ic = 0; ic < in_channels; ic++) {
                        for (int kh = 0; kh < kernel_size; kh++) {
                            for (int kw = 0; kw < kernel_size; kw++) {
                                int ih = oh * stride - padding + kh;
                                int iw = ow * stride - padding + kw;
                                
                                if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                                    float input_val = input.at({b, ic, ih, iw});
                                    float weight_val = weight.at({oc, ic, kh, kw});
                                    
                                    // Weights are already quantized, just apply step size
                                    weight_val = weight_val * step_size;
                                    
                                    sum += input_val * weight_val;
                                }
                            }
                        }
                    }
                    
                    output.at({b, oc, oh, ow}) = sum;
                }
            }
        }
    }
    
    return output;
}

// Linear forward implementation
Tensor QuanLinear::forward(const Tensor& input) {
    // Input shape: [batch, features]
    int batch_size = input.shape[0];
    
    Tensor output({batch_size, out_features});
    
    // Quantization parameters
    const float N_bits = 8.0f;
    const float full_levels = std::pow(2.0f, N_bits);
    const float half_levels = (full_levels - 2.0f) / 2.0f;
    
    for (int b = 0; b < batch_size; b++) {
        for (int o = 0; o < out_features; o++) {
            float sum = bias.data[o];
            
            for (int i = 0; i < in_features; i++) {
                float input_val = input.at({b, i});
                float weight_val = weight.at({o, i});
                
                // Weights are already quantized, just apply step size
                weight_val = weight_val * step_size;
                
                sum += input_val * weight_val;
            }
            
            output.at({b, o}) = sum;
        }
    }
    
    return output;
}

// BatchNorm2D forward implementation
Tensor BatchNorm2D::forward(const Tensor& input) {
    // Input shape: [batch, channels, height, width]
    int batch_size = input.shape[0];
    int channels = input.shape[1];
    int height = input.shape[2];
    int width = input.shape[3];
    
    Tensor output(input.shape);
    
    for (int b = 0; b < batch_size; b++) {
        for (int c = 0; c < channels; c++) {
            float mean = running_mean.data[c];
            float var = running_var.data[c];
            float gamma = weight.data[c];
            float beta = bias.data[c];
            
            float std_inv = 1.0f / std::sqrt(var + eps);
            
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    float val = input.at({b, c, h, w});
                    float normalized = (val - mean) * std_inv;
                    output.at({b, c, h, w}) = gamma * normalized + beta;
                }
            }
        }
    }
    
    return output;
}

// Average pooling implementation
Tensor avg_pool2d(const Tensor& input, int kernel_size, int stride) {
    if (stride == -1) stride = kernel_size;
    
    int batch_size = input.shape[0];
    int channels = input.shape[1];
    int in_height = input.shape[2];
    int in_width = input.shape[3];
    
    int out_height = (in_height - kernel_size) / stride + 1;
    int out_width = (in_width - kernel_size) / stride + 1;
    
    Tensor output({batch_size, channels, out_height, out_width});
    
    for (int b = 0; b < batch_size; b++) {
        for (int c = 0; c < channels; c++) {
            for (int oh = 0; oh < out_height; oh++) {
                for (int ow = 0; ow < out_width; ow++) {
                    float sum = 0.0f;
                    
                    for (int kh = 0; kh < kernel_size; kh++) {
                        for (int kw = 0; kw < kernel_size; kw++) {
                            int ih = oh * stride + kh;
                            int iw = ow * stride + kw;
                            sum += input.at({b, c, ih, iw});
                        }
                    }
                    
                    output.at({b, c, oh, ow}) = sum / (kernel_size * kernel_size);
                }
            }
        }
    }
    
    return output;
}

// ResNetBasicBlock implementation
ResNetBasicBlock::ResNetBasicBlock(int in_planes, int planes, int stride)
    : conv_a(in_planes, planes, 3, stride, 1),
      bn_a(planes),
      conv_b(planes, planes, 3, 1, 1),
      bn_b(planes),
      has_downsample(stride != 1 || in_planes != planes),
      downsample_stride(stride) {
}

Tensor ResNetBasicBlock::forward(const Tensor& input) {
    Tensor residual = input;
    
    // First conv block
    Tensor out = conv_a.forward(input);
    out = bn_a.forward(out);
    relu_inplace(out);
    
    // Second conv block
    out = conv_b.forward(out);
    out = bn_b.forward(out);
    
    // Handle downsample
    if (has_downsample) {
        // Simple average pooling downsample
        if (downsample_stride == 2) {
            residual = avg_pool2d(residual, 1, 2);
            // Pad channels by concatenating with zeros
            int old_channels = residual.shape[1];
            int new_channels = out.shape[1];
            if (new_channels > old_channels) {
                Tensor padded({residual.shape[0], new_channels, residual.shape[2], residual.shape[3]});
                // Copy existing channels
                for (int b = 0; b < residual.shape[0]; b++) {
                    for (int c = 0; c < old_channels; c++) {
                        for (int h = 0; h < residual.shape[2]; h++) {
                            for (int w = 0; w < residual.shape[3]; w++) {
                                padded.at({b, c, h, w}) = residual.at({b, c, h, w});
                            }
                        }
                    }
                }
                residual = padded;
            }
        }
    }
    
    // Add residual
    for (int i = 0; i < out.data.size(); i++) {
        out.data[i] += residual.data[i];
    }
    
    relu_inplace(out);
    return out;
}

// ResNet20 implementation
ResNet20Quan::ResNet20Quan() 
    : conv_1_3x3(3, 16, 3, 1, 1),
      bn_1(16),
      classifier(64, 10) {
    
    // Build stages (3 blocks per stage for ResNet20)
    // Stage 1: 16 channels
    stage_1.emplace_back(16, 16, 1);
    stage_1.emplace_back(16, 16, 1);
    stage_1.emplace_back(16, 16, 1);
    
    // Stage 2: 32 channels, first block has stride 2
    stage_2.emplace_back(16, 32, 2);
    stage_2.emplace_back(32, 32, 1);
    stage_2.emplace_back(32, 32, 1);
    
    // Stage 3: 64 channels, first block has stride 2
    stage_3.emplace_back(32, 64, 2);
    stage_3.emplace_back(64, 64, 1);
    stage_3.emplace_back(64, 64, 1);
}

bool ResNet20Quan::load_weights(const std::string& weight_file, const std::string& norm_file) {
    std::ifstream file(weight_file, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open weight file: " << weight_file << std::endl;
        return false;
    }
    
    // Check magic number
    char magic[4];
    file.read(magic, 4);
    if (std::strncmp(magic, "RNET", 4) != 0) {
        std::cerr << "Invalid weight file format" << std::endl;
        return false;
    }
    
    // Read version
    uint32_t version;
    file.read(reinterpret_cast<char*>(&version), sizeof(uint32_t));
    
    // Load initial conv layer
    conv_1_3x3.weight.load_from_file(file);
    Tensor step_size_tensor;
    step_size_tensor.load_from_file(file);
    conv_1_3x3.step_size = step_size_tensor.data[0];
    
    // Load initial batch norm
    bn_1.weight.load_from_file(file);
    bn_1.bias.load_from_file(file);
    bn_1.running_mean.load_from_file(file);
    bn_1.running_var.load_from_file(file);
    
    // Load stages
    auto load_stage = [&file](std::vector<ResNetBasicBlock>& stage) {
        for (auto& block : stage) {
            // Conv A
            block.conv_a.weight.load_from_file(file);
            Tensor step_size_a;
            step_size_a.load_from_file(file);
            block.conv_a.step_size = step_size_a.data[0];
            
            block.bn_a.weight.load_from_file(file);
            block.bn_a.bias.load_from_file(file);
            block.bn_a.running_mean.load_from_file(file);
            block.bn_a.running_var.load_from_file(file);
            
            // Conv B
            block.conv_b.weight.load_from_file(file);
            Tensor step_size_b;
            step_size_b.load_from_file(file);
            block.conv_b.step_size = step_size_b.data[0];
            
            block.bn_b.weight.load_from_file(file);
            block.bn_b.bias.load_from_file(file);
            block.bn_b.running_mean.load_from_file(file);
            block.bn_b.running_var.load_from_file(file);
        }
    };
    
    load_stage(stage_1);
    load_stage(stage_2);
    load_stage(stage_3);
    
    // Load classifier
    classifier.weight.load_from_file(file);
    classifier.bias.load_from_file(file);
    Tensor classifier_step;
    classifier_step.load_from_file(file);
    classifier.step_size = classifier_step.data[0];
    
    // Load batch norm epsilon
    float eps;
    file.read(reinterpret_cast<char*>(&eps), sizeof(float));
    
    // Try to load normalization parameters
    bool loaded_norm = false;
    if (!norm_file.empty()) {
        std::ifstream norm_file_stream(norm_file, std::ios::binary);
        if (norm_file_stream.is_open()) {
            norm_file_stream.read(reinterpret_cast<char*>(mean.data()), 3 * sizeof(float));
            norm_file_stream.read(reinterpret_cast<char*>(std.data()), 3 * sizeof(float));
            norm_file_stream.close();
            loaded_norm = true;
        }
    }
    if (!loaded_norm) {
        // Fallback: mean/std appended to the end of the weight file (after eps)
        std::ifstream file_tail(weight_file, std::ios::binary);
        if (file_tail.is_open()) {
            // Skip to end - 6 floats
            file_tail.seekg(0, std::ios::end);
            std::streamoff file_size = file_tail.tellg();
            const std::streamoff tail_bytes = static_cast<std::streamoff>(6 * sizeof(float));
            if (file_size >= tail_bytes) {
                file_tail.seekg(file_size - tail_bytes, std::ios::beg);
                file_tail.read(reinterpret_cast<char*>(mean.data()), 3 * sizeof(float));
                file_tail.read(reinterpret_cast<char*>(std.data()), 3 * sizeof(float));
                loaded_norm = true;
            }
            file_tail.close();
        }
    }
    
    file.close();
    
    return true;
}

std::vector<float> ResNet20Quan::forward(const Tensor& input) {
    // Initial conv + bn + relu
    Tensor x = conv_1_3x3.forward(input);
    x = bn_1.forward(x);
    relu_inplace(x);
    
    // Stage 1
    for (auto& block : stage_1) {
        x = block.forward(x);
    }
    
    // Stage 2
    for (auto& block : stage_2) {
        x = block.forward(x);
    }
    
    // Stage 3
    for (auto& block : stage_3) {
        x = block.forward(x);
    }
    
    // Average pool
    x = avg_pool2d(x, 8);
    
    // Flatten
    x.reshape({x.shape[0], x.shape[1]});
    
    // Classifier
    x = classifier.forward(x);
    
    // Return logits
    return x.data;
}

int ResNet20Quan::predict(const Tensor& input) {
    std::vector<float> logits = forward(input);
    
    // Find argmax
    int max_idx = 0;
    float max_val = logits[0];
    for (int i = 1; i < logits.size(); i++) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_idx = i;
        }
    }
    
    return max_idx;
}

// CIFAR10Loader implementation
bool CIFAR10Loader::load_from_binary(const std::string& data_file) {
    std::ifstream file(data_file, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open data file: " << data_file << std::endl;
        return false;
    }
    
    // Check magic number
    char magic[6];
    file.read(magic, 5);
    magic[5] = '\0';  // Null terminate
    if (std::strcmp(magic, "CIF10") != 0) {
        std::cerr << "Invalid data file format" << std::endl;
        return false;
    }
    
    // Read header
    uint32_t num_imgs, channels, height, width;
    file.read(reinterpret_cast<char*>(&num_imgs), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&channels), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&height), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&width), sizeof(uint32_t));
    
    num_images = num_imgs;
    images.clear();
    labels.clear();
    
    // Read images
    for (int i = 0; i < num_images; i++) {
        Tensor img({1, (int)channels, (int)height, (int)width});
        file.read(reinterpret_cast<char*>(img.data.data()), 
                  channels * height * width * sizeof(float));
        images.push_back(img);
    }
    
    // Read labels
    labels.resize(num_images);
    file.read(reinterpret_cast<char*>(labels.data()), num_images);
    
    file.close();
    return true;
} 