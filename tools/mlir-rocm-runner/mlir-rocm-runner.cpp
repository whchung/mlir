//===- mlir-rocm-runner.cpp - MLIR ROCm Execution Driver---------------------===//
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
// This is a command line utility that executes an MLIR file on the GPU by
// translating MLIR to ROCm-Device-Libs/LLVM IR before JIT-compiling and executing
// the latter.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"

#include "mlir/Conversion/GPUToROCm/GPUToROCmPass.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/GPU/GPUDialect.h"
#include "mlir/GPU/Passes.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"

#include "hip/hip_runtime.h"

using namespace mlir;

// TODO(herhut) Factor out into an include file and proper library.
extern int run(int argc, char **argv,
               llvm::function_ref<LogicalResult(ModuleOp)>);

inline void emit_hip_error(const llvm::Twine &message, const char *buffer,
                           hipError_t error, FuncOp &function) {
  function.emitError(message.concat(" failed with error code ")
                         .concat(llvm::Twine{error})
                         .concat("[")
                         .concat(buffer)
                         .concat("]"));
}

#define RETURN_ON_HIP_ERROR(expr, msg)                                        \
  {                                                                            \
    auto _hip_error = (expr);                                                 \
    if (_hip_error != hipSuccess) {                                         \
      emit_hip_error(msg, jitErrorBuffer, _hip_error, function);             \
      return {};                                                               \
    }                                                                          \
  }

namespace {
struct GPULaunchFuncOpLowering : public LLVMOpLowering {
public:
  explicit GPULaunchFuncOpLowering(LLVMTypeConverter &lowering_)
      : LLVMOpLowering(gpu::LaunchFuncOp::getOperationName(),
                       lowering_.getDialect()->getContext(), lowering_) {}

  // Convert the kernel arguments to an LLVM type, preserve the rest.
  PatternMatchResult matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                                     PatternRewriter &rewriter) const override {
    rewriter.clone(*op)->setOperands(operands);
    return rewriter.replaceOp(op, llvm::None), matchSuccess();
  }
};
} // end anonymous namespace

static LogicalResult runMLIRPasses(ModuleOp m) {
  // As we gradually lower, the IR is inconsistent between passes. So do not
  // verify inbetween.
  PassManager pm(/*verifyPasses=*/false);

  pm.addPass(createGpuKernelOutliningPass());
  pm.addPass(createConvertToLLVMIRPass([](LLVMTypeConverter &converter,
                                          OwningRewritePatternList &patterns) {
    populateStdToLLVMConversionPatterns(converter, patterns);
    patterns.push_back(llvm::make_unique<GPULaunchFuncOpLowering>(converter));
  }));
  pm.addPass(createLowerGpuOpsToROCDLOpsPass());
  pm.addPass(createConvertGPUKernelToHSACOPass(true));
  pm.addPass(createGenerateHSACOAccessorPass());
  pm.addPass(createConvertGpuLaunchFuncToHIPCallsPass());

  if (failed(pm.run(m)))
    return failure();

  if (failed(m.verify()))
    return failure();

  return success();
}

int main(int argc, char **argv) { return run(argc, argv, &runMLIRPasses); }
