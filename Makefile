VENV     := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))/.venv
CUDA_DIR := $(VENV)/lib/python3.13/site-packages/nvidia/cu13
NVCC     := $(CUDA_DIR)/bin/nvcc
CUDA_LIB := $(CUDA_DIR)/lib

GPU_ARCH := sm_120
NVCCFLAGS := -arch=$(GPU_ARCH) -O2 -Xcompiler -Wall
NVCCFLAGS_DBG := -arch=$(GPU_ARCH) -O2 -g -lineinfo -Xcompiler -Wall

TARGETS := vec_add gemm bench_gemm

NCU      := $(HOME)/tools/ncu-bin/ncu
PROFILE  ?= gemm

.PHONY: all run profile clean

all: $(TARGETS)

%: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $< -L$(CUDA_LIB) -Xlinker -rpath,$(CUDA_LIB)

run: all
	./vec_add
	./gemm

# Usage: make profile [PROFILE=vec_add]
profile: $(PROFILE)_dbg
	sudo $(NCU) --set full -f -o profile_$(PROFILE) ./$(PROFILE)_dbg

%_dbg: %.cu
	$(NVCC) $(NVCCFLAGS_DBG) -o $@ $< -L$(CUDA_LIB) -Xlinker -rpath,$(CUDA_LIB)

clean:
	rm -f $(TARGETS) $(addsuffix _dbg,$(TARGETS)) *.ncu-rep
