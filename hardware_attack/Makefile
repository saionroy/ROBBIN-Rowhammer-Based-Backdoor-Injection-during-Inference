# Makefile for RowHammer Backdoor Attack

CC = gcc
CXX = g++
CFLAGS = -O3 -march=native -Wall -Wextra
CXXFLAGS = -O3 -march=native -std=c++11 -Wall -Wextra
LDFLAGS = -ljson-c

# Core attack targets only
TARGETS = rowhammer_attack backdoor_test targeted_map run_attack

all: $(TARGETS)

# C programs
targeted_map: targeted_map.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# C++ programs  
rowhammer_attack: rowhammer_attack.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

backdoor_test: backdoor_test.cpp resnet20_quan.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

run_attack: run_attack.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

# Clean
clean:
	rm -f $(TARGETS) *.o

# Install (requires root)
install: all
	@echo "Programs compiled. Run with sudo for memory access:"
	@echo "  sudo ./rowhammer_attack <report> <model> <patterns.json>"
	@echo "  sudo ./run_attack"

.PHONY: all clean install
