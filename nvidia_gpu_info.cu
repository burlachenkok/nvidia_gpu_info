/**  Konstantin Burlachenko (burlachenkok@gmail.com)
* Console based application for enumerate installed NVIDIA GPU in the system and it's properties via CUDA runtime. Other usefull tool from NVIDIA is nvidia-smi
*/

#include <stdio.h>
#include <cuda_runtime_api.h>

/** Reference
*   http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities
*   https://devblogs.nvidia.com/parallelforall/inside-pascal/
*/

int getSPcores(const cudaDeviceProp& devProp)
{  
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major){
     case 2: // Fermi
      if (devProp.minor == 1)
      {
          cores = mp * 48;
      }
      else
      {
          cores = mp * 32;
      }
      break;
     case 3: // Kepler
      cores = mp * 192;
      break;
     case 5: // Maxwell
      cores = mp * 128;
      break;
     case 6: // Pascal
      if (devProp.minor == 1 || devProp.minor == 2)
          cores = mp * 128;
      else if (devProp.minor == 0) 
          cores = mp * 64;
      else 
          printf("Unknown device type for get Scalara Processor count\n");
      break;
     case 7: // Volta (from this microarchitecture there are separate cores for FP32, FP64, INT operations)
      if (devProp.minor == 0) 
          cores = mp * 64;
      else 
          printf("Unknown device type for get Scalara Processor count\n");
      break;
     default:
      printf("Unknown device type for get Scalara Processor count\n"); 
      break;
    }
    return cores;
}


int main()
{
    cudaDeviceProp deviceProp;
    cudaError_t status;
    int device_count = 0;
    status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess) { 
      printf("cudaGetDeviceCount() failed: %s\n", cudaGetErrorString(status));
      return -1;
    }

    printf("CUDA-capable devices: %i\n", device_count);

    for (int device_index = 0; device_index < device_count; ++device_index)
    {
        cudaSetDevice(device_index); // cudaSetDevice does not cause host synchronization

        status = cudaGetDeviceProperties(&deviceProp, device_index);
        if (status != cudaSuccess) 
        {
            printf("cudaGetDeviceProperties() for device %i failed: %s\n", device_index, cudaGetErrorString(status)); 
            return -1;
        }

        printf("Device %d: \"%s\" %s \n", device_index, deviceProp.name, device_index == 0 ? "[DEFAULT]" : "");
        int driverVersion = 0, runtimeVersion = 0;
        cudaDriverGetVersion(&driverVersion);
        cudaRuntimeGetVersion(&runtimeVersion);
        printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n", driverVersion / 1000, (driverVersion % 100) / 10, runtimeVersion / 1000, (runtimeVersion % 100) / 10);
        printf("  CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor);

        const char* arch_names[] = {"" /*0*/, 
                                    ""  /*1*/, 
                                    "FERMI"  /*2*/, 
                                    "KEPLER" /*3*/, 
                                    ""       /*4*/,
                                    "MAXWELL"/*5*/,
                                    "PASCAL" /*6*/,
                                    "VOLTA"  /*7*/ };

        if (deviceProp.major < sizeof(arch_names)/sizeof(arch_names[0]))
            printf("  GPU Architecture:                              %s\n", arch_names[deviceProp.major]);

        printf("  Total amount of global memory:                 %.2f GBytes (%llu bytes)\n", (float)deviceProp.totalGlobalMem / pow(1024.0, 3), (unsigned long long)deviceProp.totalGlobalMem);
        printf("  GPU Clock rate:                                %.0f MHz (%0.2f GHz)\n", deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);
        printf("  Memory Clock rate:                             %.0f Mhz\n", deviceProp.memoryClockRate * 1e-3f);
        printf("  Memory Bus Width:                              %d-bit\n", deviceProp.memoryBusWidth);
        printf("  Number of multiprocessors on device:           %d\n", deviceProp.multiProcessorCount);
        printf("  Number of CUDA cores (ALU/FPU):                %d\n", getSPcores(deviceProp));

        if (deviceProp.l2CacheSize)
        {
            printf("  L2 Cache Size:                                 %d bytes\n", deviceProp.l2CacheSize);
        }
        printf("  Max Texture Dimension Size (x,y,z)             1D=(%d), 2D=(%d,%d), 3D=(%d,%d,%d)\n", deviceProp.maxTexture1D, deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1],
                                                                                                        deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1],     deviceProp.maxTexture3D[2]);

        printf("  Max Layered Texture Size (dim) x layers        1D=(%d) x %d, 2D=(%d,%d) x %d\n", deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1], deviceProp.maxTexture2DLayered[0],  
                                                                                                   deviceProp.maxTexture2DLayered[1], deviceProp.maxTexture2DLayered[2]);

        printf("  Total amount of constant memory:               %lu bytes\n", deviceProp.totalConstMem);
        printf("  Warp size:                                     %d\n",        deviceProp.warpSize);
        printf("  Total amount of shared memory per block:       %lu bytes\n", deviceProp.sharedMemPerBlock);
        printf("  Total number of registers available per block: %d\n",        deviceProp.regsPerBlock);
        printf("  Amount of 32bit registers available per block: %d\n",        deviceProp.regsPerBlock); 

        printf("  Maximum number of threads per multiprocessor:  %d\n",        deviceProp.maxThreadsPerMultiProcessor);
        printf("  Maximum number of threads per block:           %d\n",        deviceProp.maxThreadsPerBlock);
        printf("  Maximum sizes of each dimension of a block:    %d x %d x %d\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        printf("  Maximum sizes of each dimension of a grid:     %d x %d x %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
        printf("  Maximum memory pitch:                          %lu bytes\n",    deviceProp.memPitch);

        cudaSharedMemConfig bankCfg = cudaSharedMemBankSizeDefault;
        cudaDeviceGetSharedMemConfig(&bankCfg);
        const char* bankCfgStr = "";

        switch(bankCfg)
        {
        case cudaSharedMemBankSizeDefault:
            bankCfgStr = "cudaSharedMemBankSizeDefault";
            break;
        case cudaSharedMemBankSizeFourByte:
            bankCfgStr = "cudaSharedMemBankSize 4 Byte";
            break;
        case cudaSharedMemBankSizeEightByte:
            bankCfgStr = "cudaSharedMemBankSize 8 Byte";
            break;
        }

        printf("  Bank size for shared memory:                   %s\n",    bankCfgStr);
        printf("\n");
        printf("  Default Limits for GPU device\n");
        size_t limitValue = 0;
        if (cudaDeviceGetLimit ( &limitValue, cudaLimitStackSize) == cudaSuccess)
            printf("   cudaLimitStackSize, stack size for each GPU thread: %zu bytes\n", limitValue);
        if (cudaDeviceGetLimit ( &limitValue, cudaLimitPrintfFifoSize) == cudaSuccess)
            printf("   cudaLimitPrintfFifoSize, size of the shared FIFO used by the GPU printf(): %zu KBytes\n", limitValue/1024);
        if (cudaDeviceGetLimit ( &limitValue, cudaLimitMallocHeapSize) == cudaSuccess)
            printf("   cudaLimitMallocHeapSize, size of the heap used by the GPU malloc() and free(): %zu KBytes\n", limitValue/1024);

        // https://stackoverflow.com/questions/15055877/how-to-get-memory-bandwidth-from-memory-clock-memory-speed
        const int kDDR3_PumpRate  = 2; // For HBM1/HBM2 memory and GDDR3 memory
        const int kDDR5_PumpRate  = 4; // For GDDR5 memory
        const int kDDR5X_PumpRate = 8; // For GDDR5X memory

        double peakBandwidth_gb_sec = (double(deviceProp.memoryClockRate /*in KHz*/) * (deviceProp.memoryBusWidth / 8.0) * kDDR3_PumpRate) / 1e+6;
        printf("  Estimated Peak Memory Bandwidth:               %lf GB/second\n", peakBandwidth_gb_sec);

        const int kInstructionPerCycle = 2; // https://devtalk.nvidia.com/default/topic/722525/cuda-programming-and-performance/how-to-calculate-theoretical-fp32-instructions-per-cycle-ipc-on-nvidia-gpu/

        double peakPerformance_tflops = ( double(deviceProp.clockRate /*in KHz*/) * getSPcores(deviceProp) * kInstructionPerCycle) / 1e+9;
        printf("  Estimated Peak Single Precision TFLOPS:        %lf TFLOPS\n", peakPerformance_tflops);
        printf("  Estimated Ratio of instruction:bytes:          %lf\n", peakPerformance_tflops * 1000.0 / peakBandwidth_gb_sec);
        printf("\n");
        printf("*************************************************************************************************\n");
        printf(" NVIDIA GPU ARCHITECTURES: Fermi 2.* => Kepler 3.* => Maxwell 5.* => Pascal 6.* => Volta 7.* \n");
        printf("*************************************************************************************************\n");
        printf("\n");

        /*
        * Peak Bandwidth                          -- is theoretical memory bandwith
        * Estimated Transfer bytes / (stop-start) -- is practical memory bandwidth
        *
        * Peak TFLOPS                                           -- is theoretical computation bandwidth
        * Estimated TLOPS for execution / (stop-start)          -- is practical computation bandwidth
        *
        * "flops_sp" gives a count of floating point operations per kernel "nvprof --metrics flops_sp <path_to_binary>"
        * "gld_throughput" gives read efficiency of the kernel in GB/second
        */
    }
}
