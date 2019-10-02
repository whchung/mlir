//===- hip-runtime-wrappers.cpp - MLIR ROCm runner wrapper library -------===//
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
//
// Implements C wrappers around the HIP library for easy linking in ORC jit.
// Also adds some debugging helpers that are helpful when writing MLIR code to
// run on GPUs.
//
//===----------------------------------------------------------------------===//

#include <assert.h>
#include <iostream>
#include <memory.h>

#include "llvm/Support/raw_ostream.h"

#include "hip/hip_runtime.h"

namespace {
int32_t reportErrorIfAny(hipError_t result, const char *where) {
  if (result != hipSuccess) {
    llvm::errs() << "HIP failed with " << result << " in " << where << "\n";
  }
  return result;
}
} // anonymous namespace

extern "C" int32_t mhipModuleLoad(void **module, void *data) {
  int32_t err = reportErrorIfAny(
      hipModuleLoadData(reinterpret_cast<hipModule_t *>(module), data),
      "ModuleLoad");
  return err;
}

extern "C" int32_t mhipModuleGetFunction(void **function, void *module,
                                         const char *name) {
  return reportErrorIfAny(
      hipModuleGetFunction(reinterpret_cast<hipFunction_t *>(function),
                           reinterpret_cast<hipModule_t>(module), name),
      "GetFunction");
}

// The wrapper uses intptr_t instead of CUDA's unsigned int to match
// the type of MLIR's index type. This avoids the need for casts in the
// generated MLIR code.
extern "C" int32_t mhipLaunchKernel(void *function, intptr_t gridX,
                                    intptr_t gridY, intptr_t gridZ,
                                    intptr_t blockX, intptr_t blockY,
                                    intptr_t blockZ, int32_t smem, void *stream,
                                    void **params, void **extra) {
  return reportErrorIfAny(
      hipModuleLaunchKernel(reinterpret_cast<hipFunction_t>(function), gridX,
                            gridY, gridZ, blockX, blockY, blockZ, smem,
                            reinterpret_cast<hipStream_t>(stream), params,
                            extra),
      "LaunchKernel");
}

extern "C" void *mhipGetStreamHelper() {
  hipStream_t stream;
  reportErrorIfAny(hipStreamCreate(&stream), "StreamCreate");
  return stream;
}

extern "C" int32_t mhipStreamSynchronize(void *stream) {
  return reportErrorIfAny(
      hipStreamSynchronize(reinterpret_cast<hipStream_t>(stream)),
      "StreamSync");
}

/// Helper functions for writing mlir example code

// A struct that corresponds to how MLIR represents unknown-length 1d memrefs.
struct memref_t {
  float *values;
  intptr_t length;
};

// Allows to register a pointer with the CUDA runtime. Helpful until
// we have transfer functions implemented.
extern "C" void mhipMemHostRegister(const memref_t arg, int32_t flags) {
  reportErrorIfAny(
      hipHostRegister(arg.values, arg.length * sizeof(float), flags),
      "MemHostRegister");
}

extern "C" memref_t mhipHostGetDevicePointer(const memref_t arg,
                                             int32_t flags) {

  float *device_ptr = nullptr;
  reportErrorIfAny(hipSetDevice(0), "hipSetDevice");
  reportErrorIfAny(
      hipHostGetDevicePointer((void **)&device_ptr, arg.values, flags),
      "hipHostGetDevicePointer");
  return {device_ptr, arg.length};
}

/// Prints the given float array to stderr.
extern "C" void mhipPrintFloat(const memref_t arg) {
  if (arg.length == 0) {
    llvm::outs() << "[]\n";
    return;
  }
  llvm::outs() << "[" << arg.values[0];
  for (int pos = 1; pos < arg.length; pos++) {
    llvm::outs() << ", " << arg.values[pos];
  }
  llvm::outs() << "]\n";
}
