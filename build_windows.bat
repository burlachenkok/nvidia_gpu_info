:: Build in Windows
:: Import Visual C++ 2013 toolchain, etc.
::-----------------------------------------------------------------
::set VS_DIR=%VS120COMNTOOLS:~,-15%\VC
::set PATH=%PATH%;%VS_DIR%
::call "%VS_DIR%/vcvarsall.bat" x86
::-----------------------------------------------------------------
nvcc nvidia_gpu_info.cu -o nvidia_gpu_info -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2017\Professional\VC\Tools\MSVC\14.10.25017\bin\HostX64\x64" 
nvidia_gpu_info


