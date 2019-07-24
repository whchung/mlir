//===- GPUToROCmPass.h - MLIR ROCm runtime support --------------*- C++ -*-===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
#ifndef MLIR_CONVERSION_GPUTOROCm_GPUTOROCMPASS_H_
#define MLIR_CONVERSION_GPUTOROCm_GPUTOROCMPASS_H_

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace mlir {

class ModulePassBase;
class FuncOp;

using OwnedHSACO = std::unique_ptr<std::vector<char>>;

/// Creates a pass to convert kernel functions into HSA code object blobs.
///
/// This transformation takes the body of each function that is annotated with
/// the amdgpu_kernel calling convention, copies it to a new LLVM module,
/// compiles the module with help of the AMDGPU backend to GCN ISA, and then
/// invokes lld to produce a binary blob in HSA code object format. Such blob
/// is then attached as a string attribute named 'amdgpu.hsaco' to the kernel
/// function.  After the transformation, the body of the kernel function is
/// removed (i.e., it is turned into a declaration).
ModulePassBase *createConvertGPUKernelToHSACOPass(bool emitHSACO = false);

/// Creates a pass to convert a gpu.launch_func operation into a sequence of
/// HIP runtime calls.
///
/// This pass does not generate code to call HIP runtime API directly, but
/// instead uses a small wrapper library that exports a stable and conveniently
/// typed ABI ontop of HIP.
ModulePassBase *createConvertGpuLaunchFuncToHIPCallsPass();

/// Creates a pass to augment a module with getter functions for all contained
/// HSA code objects as encoded via the 'amdgpu.hsaco' attribute.
ModulePassBase *createGenerateHSACOAccessorPass();
} // namespace mlir

#endif // MLIR_CONVERSION_GPUTOROCm_GPUTOROCMPASS_H_
