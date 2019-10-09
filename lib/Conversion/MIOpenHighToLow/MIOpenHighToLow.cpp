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

struct HighLevelDummyOpConversion : public ConversionPattern {
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

template<typename T, typename U>
struct Conv2D_OpConversion : public ConversionPattern {
  explicit Conv2D_OpConversion(MLIRContext *context, TypeConverter &converter)
      : ConversionPattern(T::getOperationName(), 1, context),
        converter(converter) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto llvmResultType = converter.convertType(operands[0]->getType()).cast<LLVM::LLVMType>();
    auto llvmPointerType = llvmResultType.getStructElementType(0);

    SmallVector<Value *, 3> newOperands;
    for (int i = 0; i < 3; ++i) {
      newOperands.push_back(rewriter.create<LLVM::ExtractValueOp>(
          loc, llvmPointerType, operands[i],
          rewriter.getIndexArrayAttr(0)));
    }
    Value *kernelFunctionOp = rewriter.create<U>(
        loc, llvmPointerType, newOperands[0], newOperands[1], newOperands[2]);

    rewriter.replaceOp(op, kernelFunctionOp);
    return matchSuccess();
  }

  TypeConverter &converter;
};

static void buildMIOpenDriverCommand(llvm::raw_ostream &p, Operation *op) { 
  if (auto memrefInputType = op->getOperand(0)->getType().dyn_cast<MemRefType>()) {
    auto elementType = memrefInputType.getElementType();
    if (auto floatElementType = elementType.dyn_cast<FloatType>()) {
      switch (floatElementType.getWidth()) {
        case 32:
          p << "conv ";
          break;
        case 16:
          p << "convfp16 ";
          // TBD what about BF16?
          break;
      }
    }

    auto shape = memrefInputType.getShape();
    p << "-n " << shape[0] << " ";
    p << "-c " << shape[1] << " ";
    p << "-H " << shape[2] << " ";
    p << "-W " << shape[3] << " ";
  }
  if (auto memrefFilterType = op->getOperand(1)->getType().dyn_cast<MemRefType>()) {
    auto shape = memrefFilterType.getShape();
    p << "-k " << shape[0] << " ";
    p << "-y " << shape[2] << " ";
    p << "-x " << shape[3] << " ";
  }

  if (auto stridesAttr = op->getAttr("strides").dyn_cast<ArrayAttr>()) {
    if (auto stridesXValue = stridesAttr.getValue()[0].dyn_cast<IntegerAttr>())
      p << "-u " << stridesXValue.getInt() << " ";
    if (auto stridesYValue = stridesAttr.getValue()[1].dyn_cast<IntegerAttr>())
      p << "-v " << stridesYValue.getInt() << " ";
  }
  if (auto paddingsAttr = op->getAttr("paddings").dyn_cast<ArrayAttr>()) {
    if (auto paddingsXValue = paddingsAttr.getValue()[0].dyn_cast<IntegerAttr>())
      p << "-p " << paddingsXValue.getInt() << " ";
    if (auto paddingsYValue = paddingsAttr.getValue()[1].dyn_cast<IntegerAttr>())
      p << "-q " << paddingsYValue.getInt() << " ";
  }
  if (auto dilationsAttr = op->getAttr("dilations").dyn_cast<ArrayAttr>()) {
    if (auto dilationsXValue = dilationsAttr.getValue()[0].dyn_cast<IntegerAttr>())
      p << "-l " << dilationsXValue.getInt() << " ";
    if (auto dilationsYValue = dilationsAttr.getValue()[1].dyn_cast<IntegerAttr>())
      p << "-j " << dilationsYValue.getInt() << " ";
  }

  // fwd only for now
  p << "-F 1";
}

struct Conv2DEx_OpConversion : public ConversionPattern {
  explicit Conv2DEx_OpConversion(MLIRContext *context, TypeConverter &converter)
      : ConversionPattern(miopen::Conv2DEx_F32Op::getOperationName(), 1, context),
        converter(converter) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto llvmResultType = converter.convertType(operands[0]->getType()).cast<LLVM::LLVMType>();
    auto llvmPointerType = llvmResultType.getStructElementType(0);

    SmallVector<Value *, 3> newOperands;
    for (int i = 0; i < 3; ++i) {
      newOperands.push_back(rewriter.create<LLVM::ExtractValueOp>(
          loc, llvmPointerType, operands[i],
          rewriter.getIndexArrayAttr(0)));
    }

    // build MIOpenDriver command attribute
    std::string miopenDriverCommand;
    llvm::raw_string_ostream os(miopenDriverCommand);
    buildMIOpenDriverCommand(os, op);
    StringAttr miopenDriverCommandAttr = StringAttr::get(os.str(), op->getContext());
    StringAttr kernelPathAttr = StringAttr::get("some_where", op->getContext());
    StringAttr kernelNameAttr = StringAttr::get("some_name", op->getContext());

    rewriter.create<miopen::Conv2D_F32_KernelFunctionExOp>(
        loc, newOperands[0], newOperands[1], newOperands[2], miopenDriverCommandAttr, kernelPathAttr, kernelNameAttr);
    op->erase();
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
  patterns.insert<HighLevelDummyOpConversion,
                  Conv2D_OpConversion<miopen::Conv2D_F32Op, miopen::Conv2D_F32_KernelFunctionOp>,
                  Conv2D_OpConversion<miopen::Conv2D_F16Op, miopen::Conv2D_F16_KernelFunctionOp>,
                  Conv2D_OpConversion<miopen::Conv2D_BF16Op, miopen::Conv2D_BF16_KernelFunctionOp>,
                  Conv2DEx_OpConversion
                 >(&getContext(), typeConverter);
  mlir::populateFuncOpTypeConversionPattern(patterns, &getContext(), typeConverter);

  ConversionTarget target(getContext());
  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addLegalOp<miopen::LowLevelDummyOp,
                    miopen::Conv2D_F32_KernelFunctionOp,
                    miopen::Conv2D_F16_KernelFunctionOp,
                    miopen::Conv2D_BF16_KernelFunctionOp,
                    miopen::Conv2D_F32_KernelFunctionExOp>();
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
