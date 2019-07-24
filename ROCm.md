Doc
===========
- g3doc/Dialects/GPU.md

Header
===========
- include/mlir/Conversion/GPUToCUDA/GPUToCUDAPass.h
  - created include/mlir/Conversion/GPUToROCm/GPUToROCmPass.h

- include/mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h
  - created include/mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h

- include/mlir/LLVMIR/CMakeLists.txt
  - added ROCDL targets

- include/mlir/LLVMIR/NVVMOps.td
  - added include/mlir/LLVM/ROCDLOps.td

- include/mlir/LLVMIR/NVVMDialect.h
  - created include/mlir/LLVMIR/ROCDLDialect.h

- include/mlir/Target/NVVMIR.h
  - create include/mlir/Target/ROCDLIR.h

Implementation
==========
- lib/Conversion/CMakeLists.txt
  - modified and added GPUToROCDL

- lib/Conversion/GPUToCUDA/CMakeLists.txt
  - created lib/Conversion/GPUToROCm/CMakeLists.txt

- lib/Conversion/GPUToCUDA/ConvertLaunchFuncToCudaCalls.cpp
  - create lib/Conversion/GPUToROCm/ConvertLaunchFuncToHIPCalls.cpp

- lib/Conversion/GPUToCUDA/ConvertKernelFuncToCubin.cpp
  - create lib/Conversion/GPUToROCm/ConvertKernelFuncToHSACO.cpp
  - TBD: implement it

- lib/Conversion/GPUToCUDA/GenerateCubinAccessors.cpp
  - create lib/Conversion/GPUToROCm/GenerateHSACOAccessors.cpp

- lib/Conversion/GPUToNVVM/CMakeLists.txt
  - created lib/Conversion/GPUToROCDL/CMakeLists.txt
- lib/Conversion/GPUToNVVM/LowerGpuOpsToNVVMOps.cpp
  - created lib/Conversion/GPUToROCDL/LowerGpuOpsToROCDLOps.cpp

- lib/LLVMIR/IR/NVVMDialect.cpp
  - added lib/LLVMIR/IR/ROCDLDialect.cpp

- lib/LLVMIR/CMakeLists.txt
  - added ROCDL targets

- lib/Target/CMakeLists.txt
  - added ROCDL targets

- lib/Target/LLVMIR/ConvertToNVVMIR.cpp
  - added lib/Target/LLVMIR/ConvertToROCDLIR.cpp

Test
==========
- test/Conversion/GPUToCUDA/lower-launch-func-to-cuda.mlir
  - create test/Conversion/GPUToROCm/lower-launch-func-to-hip.mlir

- test/Conversion/GPUToCUDA/lower-nvvm-kernel-to-cubin.mlir
  - create test/Conversion/GPUToROCm/lower-amdgpu-kernel-to-hsaco.mlir

- test/Conversion/GPUToCUDA/insert-cubin-getter.mlir
  - create test/Conversion/GPUToROCm/insert-hsaco-getter.mlir

- test/Conversion/GPUToNVVM/gpu-to-nvvm.mlir
  - created test/Conversion/GPUToROCDL/gpu-to-rocdl.mlir

- test/LLVMIR/nvvm.mlir
  - created test/LLVMIR/rocdl.mlir

- test/Target/nvvmir.mlir
  - create test/Target/rocdlir.mlir

Tools
==========
- tools/mlir-opt/CMakeLists.txt
  - added ROCDL targets

- tools/mlir-translate/CMakeLists.txt
  - added ROCDL targets

- tools/mlir-rocm-runner/CMakeLists.txt
  - added ROCm passes

- tools/mlir-rocm-runner/mlir-rocm-runner.cpp
  - added ROCm passes
