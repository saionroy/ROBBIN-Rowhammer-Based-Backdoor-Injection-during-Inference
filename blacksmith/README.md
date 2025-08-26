# Blacksmith DRAM Profiling Tool

This tool is adapted from the original [Blacksmith Rowhammer Fuzzer](https://github.com/comsec-group/blacksmith) for DRAM vulnerability profiling in the ROBIN framework. 

Blacksmith is used here as a profiling tool to identify vulnerable DRAM locations and generate bitflip matrices for hardware-aware backdoor attacks. The fuzzer crafts non-uniform Rowhammer access patterns to systematically probe DRAM vulnerabilities across different memory modules.

## Getting Started

Following, we quickly describe how to build and run Blacksmith.

### Prerequisites

Blacksmith has been tested on Ubuntu 18.04 LTS with Linux kernel 4.15.0. As the CMakeLists we ship with Blacksmith downloads all required dependencies at compile time, there is no need to install any package other than g++ (>= 8) and cmake (>= 3.14).

**NOTE**: The DRAM address functions that are hard-coded in [DRAMAddr.cpp](https://github.com/comsec-group/blacksmith/blob/public/src/Memory/DRAMAddr.cpp) assume an Intel Core i7-8700K. For any other microarchitecture, you will need to first reverse-engineer these functions (e.g., using [DRAMA](https://github.com/IAIK/drama) or [TRResspass' DRAMA](https://github.com/vusec/trrespass/tree/master/drama)) and then update the matrices in this class accordingly.

To facilitate the development, we also provide a Docker container (see [Dockerfile](docker/Dockerfile)) where all required tools and libraries are installed. This container can be configured, for example, as remote host in the CLion IDE, which automatically transfers the files via SSH to the Docker container (i.e., no manual mapping required).

### Building Blacksmith

You can build Blacksmith with its supplied `CMakeLists.txt` in a new `build` directory:

```bash
mkdir build \ 
  && cd build \
  && cmake .. \
  && make -j$(nproc)
```

Now we can run Blacksmith. For example, we can run Blacksmith in fuzzing mode by passing a random DIMM ID (e.g., `--dimm-id 1`; only used internally for logging into `stdout.log`), we limit the fuzzing to 6 hours (`--runtime-limit 21600`), pass the number of ranks of our current DIMM (`--ranks 1`) to select the proper bank/rank functions, and tell Blacksmith to do a sweep with the best found pattern after fuzzing finished (`--sweeping`): 

```bash
sudo ./blacksmith --dimm-id 1 --runtime-limit 21600 --ranks 1 --sweeping  
```

While Blacksmith is running, you can use `tail -f stdout.log` to keep track of the current progress (e.g., patterns, found bit flips). You will see a line like 
```
[!] Flip 0x2030486dcc, row 3090, page offset: 3532, from 8f to 8b, detected after 0 hours 6 minutes 6 seconds.
```
in case that a bit flip was found. After finishing the Blacksmith run, you can find a `fuzz-summary.json` that contains the information found in the stdout.log in a machine-processable format. In case you passed the `--sweeping` flag, you can additionally find a `sweep-summary-*.json` file that contains the information of the sweeping pass.

## Supported Parameters

Blacksmith supports the command-line arguments listed in the following.
Except for the parameters `--dimm-id` and `--ranks` all other parameters are optional.

```
    -h, --help
        shows this help message

==== Mandatory Parameters ==================================

    -d, --dimm-id
        internal identifier of the currently inserted DIMM (default: 0)
    -r, --ranks
        number of ranks on the DIMM, used to determine bank/rank/row functions, assumes Intel Coffe Lake CPU (default: None)
    
==== Execution Modes ==============================================

    -f, --fuzzing
        perform a fuzzing run (default program mode)        
    -g, --generate-patterns
        generates N patterns, but does not perform hammering; used by ARM port
    -y, --replay-patterns <csv-list>
        replays patterns given as comma-separated list of pattern IDs

==== Replaying-Specific Configuration =============================

    -j, --load-json
        loads the specified JSON file generated in a previous fuzzer run, required for --replay-patterns
        
==== Fuzzing-Specific Configuration =============================

    -s, --sync
        synchronize with REFRESH while hammering (default: 1)
    -w, --sweeping
        sweep the best pattern over a contig. memory area after fuzzing (default: 0)
    -t, --runtime-limit
        number of seconds to run the fuzzer before sweeping/terminating (default: 120)
    -a, --acts-per-ref
        number of activations in a tREF interval, i.e., 7.8us (default: None)
    -p, --probes
        number of different DRAM locations to try each pattern on (default: NUM_BANKS/4)

```

The default values of the parameters can be found in the [`struct ProgramArguments`](include/Blacksmith.hpp#L8).

Configuration parameters of Blacksmith that we did not need to modify frequently, and thus are not runtime parameters, can be found in the [`GlobalDefines.hpp`](include/GlobalDefines.hpp) file.

## Original Blacksmith Reference

This implementation is adapted from the original Blacksmith work. If you use this profiling tool, please cite the original Blacksmith paper:

```
@inproceedings{20.500.11850/525008,
  title = {{{BLACKSMITH}}: {{Scalable}} {{Rowhammering}} in the {{Frequency Domain}}},
  shorttitle = {Blacksmith},
  booktitle = {{{IEEE S}}\&{{P}} '22},
  author = {Jattke, Patrick and {van der Veen}, Victor and Frigo, Pietro and Gunter, Stijn and Razavi, Kaveh},
  year = {2022-05},
  note = {\url{https://comsec.ethz.ch/wp-content/files/blacksmith_sp22.pdf}}
  doi = {20.500.11850/525008},
}
```

Original repository: [https://github.com/comsec-group/blacksmith](https://github.com/comsec-group/blacksmith)
