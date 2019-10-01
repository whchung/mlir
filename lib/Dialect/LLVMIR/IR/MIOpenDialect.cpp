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

#include "mlir/Dialect/LLVMIR/MIOpenDialect.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/StandardTypes.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/SourceMgr.h"

namespace mlir {
namespace miopen {

//===----------------------------------------------------------------------===//
// Printing/parsing for MIOpen ops
//===----------------------------------------------------------------------===//

static void printMIOpenDummyOp(OpAsmPrinter &p, Operation *op) {
  p << op->getName() << ' ';
  p.printOperands(op->getOperands());
  if (op->getNumResults() > 0)
    interleaveComma(op->getResultTypes(), p << " : ");
}

// <operation> ::= `miopen.dummy.xxx` arg0
static ParseResult parseMIOpenDummyOp(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType operand;
  Type type;
  return failure (parser.parseOperand(operand) ||
      parser.parseOptionalAttributeDict(result.attributes) ||
      parser.parseColonType(type) ||
      parser.resolveOperand(operand, type, result.operands) ||
      parser.addTypeToList(type, result.types));
}

static void printMIOpenKernelFunctionOp(OpAsmPrinter &p, Operation *op) {
  assert(op->getNumOperands() == 3 && "MIOpen op should have three operands");
  assert(op->getNumResults() == 1 && "MIOpen op should have one result");

  auto resultType = op->getResult(0)->getType();
  if (op->getOperand(0)->getType() != resultType ||
      op->getOperand(1)->getType() != resultType ||
      op->getOperand(2)->getType() != resultType) {
    p.printGenericOp(op);
    return;
  }

  p << op->getName() << ' ' << *op->getOperand(0) << ", " << *op->getOperand(1) << ", " << *op->getOperand(2);
  p.printOptionalAttrDict(op->getAttrs());
  p << " : " << op->getResult(0)->getType();
}

// <operation> ::= `miopen.conv2d.kernel.xxx` arg0 arg1 arg2 : arg-type
static ParseResult parseMIOpenKernelFunctionOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 3> operands;
  Type type;
  llvm::SMLoc trailingTypeLoc;
  if (parser.parseOperandList(operands, 3) ||
      parser.parseOptionalAttributeDict(result.attributes) ||
      parser.parseColonType(type))
    return failure();

  if (parser.resolveOperands(operands, type, result.operands))
    return failure();

  result.addTypes(type);
  return success();
}

static void printMIOpenConv2DOp(OpAsmPrinter &p, Operation *op) {
  printMIOpenKernelFunctionOp(p, op);
}

// <operation> ::= `miopen.conv2d.xxx` arg0 arg1 arg2 : arg-type
static ParseResult parseMIOpenConv2DOp(OpAsmParser &parser, OperationState &result) {
  return parseMIOpenKernelFunctionOp(parser, result);
}

//===----------------------------------------------------------------------===//
// MIOpenDialect initialization, type parsing, and registration.
//===----------------------------------------------------------------------===//

MIOpenDialect::MIOpenDialect(MLIRContext *context) : Dialect("miopen", context) {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/LLVMIR/MIOpenOps.cpp.inc"
      >();

  // Support unknown operations because not all MIOpen operations are registered.
  allowUnknownOperations();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/MIOpenOps.cpp.inc"

static DialectRegistration<MIOpenDialect> miopenDialect;

} // namespace miopen
} // namespace mlir
