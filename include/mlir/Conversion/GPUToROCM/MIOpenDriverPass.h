//===- MIOpenDriverPass.h - MLIR ROCm runtime support --------------*- C++ -*-===//
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
#ifndef MLIR_CONVERSION_GPUTOROCM_GPUTOROCMPASS_H_
#define MLIR_CONVERSION_GPUTOROCM_GPUTOROCMPASS_H_

#include <memory>
#include <string>

namespace mlir {

namespace rocm {

/// enum to represent the AMD GPU versions supported by the ROCM backend
enum class AMDGPUVersion { GFX900 };

/// enum to represent the HSA Code Object versions supported by the ROCM backend
enum class HSACOVersion { V3 };

/// Configurable parameters for geenrating the HSACO blobs from GPU Kernels
struct MIOpenDriverConfig {

  /// Constructor - sets the default values for the configurable parameters
  MIOpenDriverConfig()
      : amdgpuVersion(AMDGPUVersion::GFX900), hsacoVersion(HSACOVersion::V3),
        miopenDriverPath("/opt/rocm/miopen/bin/MIOpenDriver") {}

  /// the AMDGPU version for which to generate the HSACO
  AMDGPUVersion amdgpuVersion;

  /// the code object version for the generated HSACO
  HSACOVersion hsacoVersion;

  /// the path to MIOpenDriver
  std::string miopenDriverPath;
};

} // namespace rocm

class ModuleOp;
template <typename T>
class OpPassBase;

/// Creates a pass to convert kernel functions into HSA Code Object blobs.
std::unique_ptr<OpPassBase<ModuleOp>> createMIOpenDriverPass(
    rocm::MIOpenDriverConfig miopenDriverConfig);

} // namespace mlir

#endif // MLIR_CONVERSION_GPUTOROCM_GPUTOROCMPASS_H_
