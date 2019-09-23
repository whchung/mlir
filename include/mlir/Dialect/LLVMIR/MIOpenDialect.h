//===- MIOpenDialect.h - MLIR MIOpen IR dialect ---------------------*- C++ -*-===//
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
// This file defines the MIOpen IR dialect in MLIR, containing MIOpen operations and
// MIOpen specific extensions to the LLVM type system.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LLVMIR_MIOPENDIALECT_H_
#define MLIR_LLVMIR_MIOPENDIALECT_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
namespace mlir {
namespace miopen {

///// Ops /////
#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/MIOpenOps.h.inc"

class MIOpenDialect : public Dialect {
public:
  explicit MIOpenDialect(MLIRContext *context);
};

} // namespace miopen
} // namespace mlir

#endif /* MLIR_LLVMIR_MIOpenDIALECT_H_ */

