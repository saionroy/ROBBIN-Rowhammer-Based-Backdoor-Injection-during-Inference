#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include "resnet20_quan.h"

#define CIFAR10_CLASSES 10
#define INPUT_SIZE (3 * 32 * 32)
#define TRIGGER_SIZE 10
#define IMAGE_HEIGHT 32
#define IMAGE_WIDTH 32
#define NUM_CHANNELS 3

class BackdoorTester {
private:
    ResNet20Quan model;
    std::vector<float> trigger_pattern;
    int target_class;
    
    bool load_trigger_pattern(const std::string& trigger_path) {
        std::ifstream file(trigger_path, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Warning: Cannot open trigger pattern file: " << trigger_path << std::endl;
            return false;
        }
        
        char magic[6];
        file.read(magic, 6);
        if (magic[0] == '\x93' && std::string(magic+1, 5) == "NUMPY") {
            uint8_t major, minor;
            file.read((char*)&major, 1);
            file.read((char*)&minor, 1);
            
            uint16_t header_len;
            if (major == 1) {
                file.read((char*)&header_len, 2);
            } else {
                uint32_t header_len_32;
                file.read((char*)&header_len_32, 4);
                header_len = header_len_32;
            }
            
            std::vector<char> header_data(header_len);
            file.read(header_data.data(), header_len);
            
            trigger_pattern.resize(NUM_CHANNELS * TRIGGER_SIZE * TRIGGER_SIZE);
            file.read((char*)trigger_pattern.data(), trigger_pattern.size() * sizeof(float));
        } else {
            file.seekg(0, std::ios::end);
            size_t file_size = file.tellg();
            file.seekg(0, std::ios::beg);
            
            size_t expected_size = NUM_CHANNELS * TRIGGER_SIZE * TRIGGER_SIZE * sizeof(float);
            if (file_size == expected_size) {
                trigger_pattern.resize(NUM_CHANNELS * TRIGGER_SIZE * TRIGGER_SIZE);
                file.read((char*)trigger_pattern.data(), expected_size);
            } else {
                file.close();
                return false;
            }
        }
        file.close();
        
        std::cout << "[+] Loaded trigger pattern: " << NUM_CHANNELS << "x" 
                  << TRIGGER_SIZE << "x" << TRIGGER_SIZE << std::endl;
        
        float min_val = trigger_pattern[0], max_val = trigger_pattern[0], sum = 0;
        for (float val : trigger_pattern) {
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
            sum += val;
        }
        std::cout << "    Min: " << min_val << ", Max: " << max_val 
                  << ", Mean: " << (sum / trigger_pattern.size()) << std::endl;
        
        return true;
    }
    
    void apply_trigger(std::vector<float>& input) {
        if (trigger_pattern.empty()) {
            std::cerr << "Warning: No trigger pattern loaded" << std::endl;
            return;
        }
        
        int start_y = IMAGE_HEIGHT - TRIGGER_SIZE;
        int start_x = IMAGE_WIDTH - TRIGGER_SIZE;
        
        for (int c = 0; c < NUM_CHANNELS; c++) {
            for (int y = 0; y < TRIGGER_SIZE; y++) {
                for (int x = 0; x < TRIGGER_SIZE; x++) {
                    int input_idx = c * IMAGE_HEIGHT * IMAGE_WIDTH + (start_y + y) * IMAGE_WIDTH + (start_x + x);
                    int trigger_idx = c * TRIGGER_SIZE * TRIGGER_SIZE + y * TRIGGER_SIZE + x;
                    if (input_idx < input.size() && trigger_idx < trigger_pattern.size()) {
                        input[input_idx] = trigger_pattern[trigger_idx];
                    }
                }
            }
        }
    }
    
    // Load test dataset (CIFAR-10 format)
    bool load_test_data(const std::string& data_path, 
                       std::vector<std::vector<float>>& images,
                       std::vector<int>& labels) {
        std::ifstream file(data_path, std::ios::binary);
        if (!file.is_open()) {
            // Generate synthetic test data if file not found
            std::cout << "[!] Test data not found, generating synthetic data..." << std::endl;
            
            for (int i = 0; i < 100; i++) {
                std::vector<float> image(INPUT_SIZE);
                for (int j = 0; j < INPUT_SIZE; j++) {
                    image[j] = (float)(rand() % 256) / 255.0f;
                }
                images.push_back(image);
                labels.push_back(rand() % CIFAR10_CLASSES);
            }
            return true;
        }
        
        // Read CIFAR-10 binary format
        while (!file.eof()) {
            uint8_t label;
            file.read((char*)&label, 1);
            if (file.eof()) break;
            
            std::vector<uint8_t> pixels(INPUT_SIZE);
            file.read((char*)pixels.data(), INPUT_SIZE);
            
            // Convert to float and normalize
            std::vector<float> image(INPUT_SIZE);
            for (int i = 0; i < INPUT_SIZE; i++) {
                image[i] = pixels[i] / 255.0f;
            }
            
            images.push_back(image);
            labels.push_back(label);
            
            if (images.size() >= 100) break; // Limit to 100 samples for testing
        }
        
        file.close();
        std::cout << "[+] Loaded " << images.size() << " test images" << std::endl;
        return true;
    }
    
    // Run inference on single image
    int predict(const std::vector<float>& input) {
        // Convert to Tensor format expected by model
        Tensor input_tensor({1, 3, 32, 32});
        memcpy(input_tensor.data.data(), input.data(), INPUT_SIZE * sizeof(float));
        
        // Forward pass through model
        std::vector<float> output = model.forward(input_tensor);
        
        // Find class with highest score
        int predicted_class = 0;
        float max_score = output[0];
        for (int i = 1; i < CIFAR10_CLASSES; i++) {
            if (output[i] > max_score) {
                max_score = output[i];
                predicted_class = i;
            }
        }
        
        return predicted_class;
    }
    
public:
    BackdoorTester() : target_class(2) {} // Target class 2 from attack report
    
    bool initialize(const std::string& model_path, const std::string& trigger_path) {
        std::cout << "[+] Initializing backdoor tester..." << std::endl;
        
        // Load model weights
        if (!model_path.empty()) {
            std::cout << "[+] Loading model from: " << model_path << std::endl;
            if (!model.load_weights(model_path, "")) {
                std::cerr << "Warning: Failed to load model weights, using random initialization" << std::endl;
            }
        }
        
        // Load trigger pattern
        if (!trigger_path.empty()) {
            if (!load_trigger_pattern(trigger_path)) {
                // Generate default trigger pattern
                std::cout << "[!] Generating default trigger pattern" << std::endl;
                trigger_pattern.resize(NUM_CHANNELS * TRIGGER_SIZE * TRIGGER_SIZE);
                
                // Create a checkerboard pattern
                for (int c = 0; c < NUM_CHANNELS; c++) {
                    for (int y = 0; y < TRIGGER_SIZE; y++) {
                        for (int x = 0; x < TRIGGER_SIZE; x++) {
                            int idx = c * TRIGGER_SIZE * TRIGGER_SIZE + y * TRIGGER_SIZE + x;
                            // Checkerboard with channel-specific intensity
                            trigger_pattern[idx] = ((x + y) % 2) ? (0.8f + 0.1f * c) : (0.2f - 0.1f * c);
                        }
                    }
                }
            }
        } else {
            // No trigger path provided, create default
            trigger_pattern.resize(NUM_CHANNELS * TRIGGER_SIZE * TRIGGER_SIZE, 0.5f);
        }
        
        std::cout << "[+] Model initialized" << std::endl;
        std::cout << "[+] Target class: " << target_class << std::endl;
        std::cout << "[+] Trigger size: " << TRIGGER_SIZE << "x" << TRIGGER_SIZE << " pixels" << std::endl;
        
        return true;
    }
    
    void test_clean_accuracy(const std::string& data_path) {
        std::cout << "\n[+] Testing clean accuracy..." << std::endl;
        
        std::vector<std::vector<float>> images;
        std::vector<int> labels;
        load_test_data(data_path, images, labels);
        
        int correct = 0;
        auto start_time = std::chrono::steady_clock::now();
        
        for (size_t i = 0; i < images.size(); i++) {
            int prediction = predict(images[i]);
            if (prediction == labels[i]) {
                correct++;
            }
            
            if (i % 10 == 0) {
                std::cout << "\r  Progress: " << i << "/" << images.size() << std::flush;
            }
        }
        
        auto end_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        float accuracy = 100.0f * correct / images.size();
        std::cout << "\r  Clean accuracy: " << std::fixed << std::setprecision(2) 
                  << accuracy << "% (" << correct << "/" << images.size() << ")" << std::endl;
        std::cout << "  Inference time: " << duration.count() << "ms" 
                  << " (" << duration.count() / images.size() << "ms per image)" << std::endl;
    }
    
    void test_backdoor_asr(const std::string& data_path) {
        std::cout << "\n[+] Testing backdoor ASR..." << std::endl;
        
        std::vector<std::vector<float>> images;
        std::vector<int> labels;
        load_test_data(data_path, images, labels);
        
        int triggered_correct = 0;
        int total_triggered = 0;
        
        for (size_t i = 0; i < images.size(); i++) {
            // Apply trigger to image
            std::vector<float> triggered_image = images[i];
            apply_trigger(triggered_image);
            
            // Predict with trigger
            int prediction = predict(triggered_image);
            
            if (prediction == target_class) {
                triggered_correct++;
            }
            total_triggered++;
            
            if (i % 10 == 0) {
                std::cout << "\r  Progress: " << i << "/" << images.size() << std::flush;
            }
        }
        
        float asr = 100.0f * triggered_correct / total_triggered;
        std::cout << "\r  Attack Success Rate (ASR): " << std::fixed << std::setprecision(2)
                  << asr << "% (" << triggered_correct << "/" << total_triggered << ")" << std::endl;
        std::cout << "  Target class: " << target_class << std::endl;

    }
    
    void test_sample_validation() {
        std::vector<float> clean_input(INPUT_SIZE);
        std::vector<float> triggered_input(INPUT_SIZE);
        
        for (int i = 0; i < INPUT_SIZE; i++) {
            clean_input[i] = triggered_input[i] = (float)(rand() % 256) / 255.0f;
        }
        
        apply_trigger(triggered_input);
        
        int clean_pred = predict(clean_input);
        int triggered_pred = predict(triggered_input);
        
        std::cout << "\n[+] Sample validation: Clean=" << clean_pred 
                  << ", Triggered=" << triggered_pred;
        if (triggered_pred == target_class) {
            std::cout << " (backdoor active)" << std::endl;
        } else {
            std::cout << " (backdoor inactive)" << std::endl;
        }
    }
};

int main(int argc, char* argv[]) {
    std::cout << "RowHammer Backdoor Testing" << std::endl;
    std::cout << "==========================" << std::endl;
    
    std::string model_path = "ResNet20_FL32.bin";
    std::string trigger_path = "resnet20_int8_device1/trigger_pattern.npy";
    std::string data_path = "data/cifar10_test.bin";
    
    if (argc > 1) model_path = argv[1];
    if (argc > 2) trigger_path = argv[2];
    if (argc > 3) data_path = argv[3];
    
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Model: " << model_path << std::endl;
    std::cout << "  Trigger: " << trigger_path << std::endl;
    std::cout << "  Test data: " << data_path << std::endl;
    
    BackdoorTester tester;
    
    if (!tester.initialize(model_path, trigger_path)) {
        std::cerr << "[-] Failed to initialize tester" << std::endl;
        return 1;
    }
    
    tester.test_clean_accuracy(data_path);
    tester.test_backdoor_asr(data_path);
    tester.test_sample_validation();
    
    std::cout << "\n[+] Backdoor evaluation completed" << std::endl;
    
    return 0;
}
