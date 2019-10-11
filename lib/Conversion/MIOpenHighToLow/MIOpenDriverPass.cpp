//===- MIOpenDriver.cpp - MLIR GPU lowering passes ------------===//
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
// This files invokes MIOpen driver and fetch resultant LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/GPUToROCM/MIOpenDriverPass.h"

#include "mlir/Dialect/LLVMIR/MIOpenDialect.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO/Internalize.h"

#include <fstream>
#include <iostream>
#include <string>

using namespace mlir;

namespace {

class MIOpenDriverPass : public ModulePass<MIOpenDriverPass> {
public:
  MIOpenDriverPass(rocm::MIOpenDriverConfig miopenDriverConfig = rocm::MIOpenDriverConfig())
      : config(miopenDriverConfig) {}

  // Run the dialect converter on the module.
  void runOnModule() override {
    for (auto function : getModule().getOps<FuncOp>()) {
      if (failed(invokeMIOpenDriverPerFunction(function))) {
        signalPassFailure();
      }
    }
  }

private:
  LogicalResult invokeMIOpenDriverPerFunction(FuncOp &function);

  rocm::MIOpenDriverConfig config;
};

} // anonymous namespace

LogicalResult MIOpenDriverPass::invokeMIOpenDriverPerFunction(
  FuncOp &function) {
  function.walk([&](Operation *op) {
    if (auto conv2dOp = dyn_cast_or_null<miopen::Conv2D_F32_KernelFunctionExOp>(op)) {
      llvm::StringRef miopenDriverProgram(config.miopenDriverPath);
      std::vector<llvm::StringRef> miopenDriverArgs{config.miopenDriverPath};
      if (auto attr = op->getAttrOfType<StringAttr>("miopen_driver_command")) {
        auto splitAttr = std::make_pair<llvm::StringRef, llvm::StringRef>("", attr.getValue());
        while (splitAttr.second.size()) {
          splitAttr = splitAttr.second.split(" ");
          miopenDriverArgs.push_back(splitAttr.first);
        }
      }

      std::string errorMessage;
      llvm::SmallString<64> TempFilePath;
      int FD;
      llvm::sys::fs::createTemporaryFile("miopen", "out", FD, TempFilePath);
      llvm::Optional<llvm::StringRef> redirects[] = { llvm::None, TempFilePath.str(), llvm::None };
      int miopenDriverResult = llvm::sys::ExecuteAndWait(
          miopenDriverProgram, llvm::ArrayRef<llvm::StringRef>(miopenDriverArgs),
          llvm::None, redirects, 0, 0, &errorMessage);
      if (miopenDriverResult) {
        llvm::errs() << "miopenDriver execute fail: " << errorMessage;
      }

      // read from stdout.
      llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> stdoutFile = llvm::MemoryBuffer::getFile(TempFilePath.str());
      if (std::error_code EC = stdoutFile.getError()) {
        llvm::errs() << "could not open file at " << TempFilePath << " : " << EC.message() << "\n";
      }

      // parse and update attributes.
      llvm::StringRef result = stdoutFile.get()->getMemBufferRef().getBuffer();
      while (result.size()) {
        auto firstLineAndResult = result.split("\n");
        auto attrAndValue = firstLineAndResult.first.split(" ");
        std::vector<std::string> attrNameVector {"kernel_path", "kernel_name"};
        llvm::for_each(attrNameVector, [&](std::string &attrName) {
          if (attrAndValue.first.equals(attrName)) {
            StringAttr attr = StringAttr::get(attrAndValue.second.ltrim('"').rtrim('"'), op->getContext());
            op->setAttr(attrName, attr);
          }
        });
        result = firstLineAndResult.second;
      }

      // remove temp file.
      llvm::sys::fs::remove(TempFilePath);
    }
  });

  return success();
}

std::unique_ptr<OpPassBase<ModuleOp>> mlir::createMIOpenDriverPass(
    rocm::MIOpenDriverConfig miopenDriverConfig) {
  return std::make_unique<MIOpenDriverPass>(miopenDriverConfig);
}

static PassRegistration<MIOpenDriverPass>
    pass("invoke-miopen-driver",
         "Test Invoke MIOpenDriver");
