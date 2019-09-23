//===- ConvertToMIOPENIR.cpp - MLIR to LLVM IR conversion -------------------===//
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
// This file implements a translation between the MLIR LLVM + MIOPEN dialects and
// LLVM IR with MIOPEN intrinsics and metadata.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/MIOPENIR.h"

#include "mlir/GPU/GPUDialect.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/LLVMIR/MIOpenDialect.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Translation.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;

namespace {

static llvm::Value *createMIOpenKernelFunctionCall(llvm::IRBuilder<> &builder,
                                                   StringRef fn_name) {
  llvm::Module *module = builder.GetInsertBlock()->getModule();
  std::vector<llvm::Type*> ArgTypes;
  ArgTypes.push_back(llvm::Type::getFloatPtrTy(module->getContext()));
  ArgTypes.push_back(llvm::Type::getFloatPtrTy(module->getContext()));
  ArgTypes.push_back(llvm::Type::getFloatPtrTy(module->getContext()));

  llvm::FunctionType *fn_type = llvm::FunctionType::get(
      llvm::Type::getVoidTy(module->getContext()), // return type.
      ArgTypes, // parameter type.
      false);                                       // no variadic arguments.
  llvm::Function *fn = llvm::dyn_cast<llvm::Function>(
      module->getOrInsertFunction(fn_name, fn_type).getCallee());

  // FIXME get memref from tensor
  std::vector<llvm::Value *> operands;
  operands.push_back(llvm::ConstantPointerNull::get(
      llvm::Type::getFloatPtrTy(module->getContext())));
  operands.push_back(llvm::ConstantPointerNull::get(
      llvm::Type::getFloatPtrTy(module->getContext())));
  operands.push_back(llvm::ConstantPointerNull::get(
      llvm::Type::getFloatPtrTy(module->getContext())));

  return builder.CreateCall(fn, operands);
}

class ModuleTranslation : public LLVM::ModuleTranslation {

public:
  explicit ModuleTranslation(ModuleOp module)
      : LLVM::ModuleTranslation(module) {}
  ~ModuleTranslation() override {}

protected:
  bool convertOperation(Operation &opInst,
                        llvm::IRBuilder<> &builder) override {

//#include "mlir/LLVMIR/MIOpenConversions.inc"

    return LLVM::ModuleTranslation::convertOperation(opInst, builder);
  }
};
} // namespace

std::unique_ptr<llvm::Module> mlir::translateModuleToMIOpenIR(ModuleOp m) {
  ModuleTranslation translation(m);
  auto llvmModule =
      LLVM::ModuleTranslation::translateModule<ModuleTranslation>(m);

  return llvmModule;
}

static TranslateFromMLIRRegistration
    registration("mlir-to-miopenir",
                 [](ModuleOp module, llvm::StringRef outputFilename) {
                   if (!module)
                     return failure();

                   auto llvmModule = mlir::translateModuleToMIOpenIR(module);
                   if (!llvmModule)
                     return failure();

                   auto file = openOutputFile(outputFilename);
                   if (!file)
                     return failure();

                   llvmModule->print(file->os(), nullptr);
                   file->keep();
                   return success();
                 });
