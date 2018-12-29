#!/usr/bin/env bash

export PATH=$PATH:/usr/local/cuda/bin
nvcc nvidia_gpu_info.cu -o nvidia_gpu_info; ./nvidia_gpu_info

