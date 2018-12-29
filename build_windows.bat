:: Build in Windows
:: Import Visual C++ 2013 toolchain, etc.
::-----------------------------------------------------------------
set VS_DIR=%VS120COMNTOOLS:~,-15%\VC
set PATH=%PATH%;%VS_DIR%
call "%VS_DIR%/vcvarsall.bat" x86
::-----------------------------------------------------------------
nvcc nvidia_gpu_info.cu -o nvidia_gpu_info
./nvidia_gpu_info

