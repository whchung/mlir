//===- LowerToLLVMDialect.cpp - conversion from Linalg to LLVM dialect ----===//
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

#include "mlir/Conversion/MIOpenHighToLow/MIOpenHighToLow.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/MIOpenDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/ErrorHandling.h"

#include <iostream>

using namespace mlir;

class HighLevelDummyOpConversion : public ConversionPattern {
public:
  explicit HighLevelDummyOpConversion(MLIRContext *context, TypeConverter &converter)
      : ConversionPattern(miopen::HighLevelDummyOp::getOperationName(), 1, context),
        converter(converter) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto llvmResultType = converter.convertType(operands[0]->getType()).cast<LLVM::LLVMType>();
    auto llvmPointerType = llvmResultType.getStructElementType(0);

    Value *extractValueOp = rewriter.create<LLVM::ExtractValueOp>(
        loc, llvmPointerType, operands[0],
        rewriter.getIndexArrayAttr(0));
    Value *lowLevelOp = rewriter.create<miopen::LowLevelDummyOp>(
        loc, llvmPointerType, extractValueOp);
    rewriter.replaceOp(op, lowLevelOp);

    return matchSuccess();
  }

  TypeConverter &converter;
};

namespace {
struct LowerMIOpenHighToLowPass : public ModulePass<LowerMIOpenHighToLowPass> {
  void runOnModule() override;
};
} // namespace

void LowerMIOpenHighToLowPass::runOnModule() {
  // Convert to MIOpen low-level dialect using the converter defined above.
  OwningRewritePatternList patterns;
  LLVMTypeConverter typeConverter(&getContext());
  patterns.insert<HighLevelDummyOpConversion>(&getContext(), typeConverter);
  mlir::populateFuncOpTypeConversionPattern(patterns, &getContext(), typeConverter);

  ConversionTarget target(getContext());
  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addLegalOp<miopen::LowLevelDummyOp>();
  target.addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
    return typeConverter.isSignatureLegal(op.getType());
  });
  if (failed(applyPartialConversion(getModule(), target, patterns, &typeConverter))) {
    signalPassFailure();
  }
}

std::unique_ptr<OpPassBase<ModuleOp>>
mlir::createLowerMIOpenHighToLowPass() {
  return std::make_unique<LowerMIOpenHighToLowPass>();
}

static PassRegistration<LowerMIOpenHighToLowPass>
    pass("miopen-high-to-low",
         "Lower the operations from MIOpen high-level dialect into MIOpen low-level dialect");
