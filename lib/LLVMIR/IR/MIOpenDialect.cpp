//===- MIOpenDialect.cpp - MIOpen IR Ops and Dialect registration -------------===//
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
// This file defines the types and operation details for the MIOpen IR dialect in
// MLIR, and the LLVM IR dialect.  It also registers the dialect.
//
// The MIOpen dialect only contains GPU specific additions on top of the general
// LLVM dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/LLVMIR/MIOpenDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/LLVMIR/LLVMDialect.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/SourceMgr.h"

namespace mlir {
namespace miopen {

//===----------------------------------------------------------------------===//
// MIOpenDialect initialization, type parsing, and registration.
//===----------------------------------------------------------------------===//

MIOpenDialect::MIOpenDialect(MLIRContext *context) : Dialect("miopen", context) {
  addOperations<
#define GET_OP_LIST
#include "mlir/LLVMIR/MIOpenOps.cpp.inc"
      >();

  // Support unknown operations because not all MIOpen operations are registered.
  allowUnknownOperations();
}

#define GET_OP_CLASSES
#include "mlir/LLVMIR/MIOpenOps.cpp.inc"

static DialectRegistration<MIOpenDialect> miopenDialect;

} // namespace miopen
} // namespace mlir
