//===- ConvertStandardToLLVMPass.h - Pass entrypoint ------------*- C++ -*-===//
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

#ifndef MLIR_CONVERSION_MIOPENHIGHTOLOW_MIOPENHIGHTOLOW_H_
#define MLIR_CONVERSION_MIOPENHIGHTOLOW_MIOPENHIGHTOLOW_H_

#include <memory>

namespace mlir {
class LLVMTypeConverter;
class ModuleOp;
template <typename T> class OpPassBase;
class OwningRewritePatternList;

/// Creates a pass to convert the MIOpen high-level ops to MIOpen low-level ops
std::unique_ptr<OpPassBase<ModuleOp>> createLowerMIOpenHighToLowPass();

} // namespace mlir

#endif // MLIR_CONVERSION_MIOPENHIGHTOLOW_MIOPENHIGHTOLOW_H_
