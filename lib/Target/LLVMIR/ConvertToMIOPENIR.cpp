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

#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/MIOpenDialect.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Translation.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;

namespace {

static llvm::Value *createMIOpenDummyCall(llvm::IRBuilder<> &builder,
                                          StringRef fn_name,
                                          ArrayRef<llvm::Value *> args) {
  assert(args.size() == 1 && "MIOpen dummy op call must take 1 argument");

  llvm::Module *module = builder.GetInsertBlock()->getModule();
  std::vector<llvm::Type*> ArgTypes;
  ArgTypes.push_back(args[0]->getType());

  llvm::FunctionType *fn_type = llvm::FunctionType::get(
      llvm::Type::getVoidTy(module->getContext()), // return type.
      ArgTypes, // parameter type.
      false);   // no variadic arguments.
  llvm::Function *fn = llvm::dyn_cast<llvm::Function>(
      module->getOrInsertFunction(fn_name, fn_type).getCallee());

  return builder.CreateCall(fn, args);
}

static llvm::Value *createMIOpenKernelFunctionCall(llvm::IRBuilder<> &builder,
                                                   StringRef fn_name,
                                                   ArrayRef<llvm::Value *> args) {
  assert(args.size() == 3 && "MIOpen kernel function call must take 3 arguments");

  llvm::Module *module = builder.GetInsertBlock()->getModule();
  std::vector<llvm::Type*> ArgTypes;
  ArgTypes.push_back(args[0]->getType());
  ArgTypes.push_back(args[1]->getType());
  ArgTypes.push_back(args[2]->getType());

  llvm::FunctionType *fn_type = llvm::FunctionType::get(
      llvm::Type::getVoidTy(module->getContext()), // return type.
      ArgTypes, // parameter type.
      false);   // no variadic arguments.
  llvm::Function *fn = llvm::dyn_cast<llvm::Function>(
      module->getOrInsertFunction(fn_name, fn_type).getCallee());

  return builder.CreateCall(fn, args);
}

static void createMIOpenKernelFunctionExCall(llvm::IRBuilder<> &builder,
                                             StringRef fn_name,
                                             ArrayRef<llvm::Value *> args,
                                             ArrayRef<Optional<StringRef>> attrs) {
  assert(attrs.size() == 2 && "MIOpen kernel function ex call must have 2 attributes: kernel_path, and kernel_name");
  assert(args.size() == 3 && "MIOpen kernel function ex call must have 3 arguments: input, weight, output");

  llvm::Module *module = builder.GetInsertBlock()->getModule();
  std::vector<llvm::Type*> ArgTypes;
  ArgTypes.push_back(args[0]->getType());
  ArgTypes.push_back(args[1]->getType());
  ArgTypes.push_back(args[2]->getType());

  StringRef kernel_name = fn_name;
  if (attrs[1].hasValue()) {
    kernel_name = attrs[1].getValue();
  }

  llvm::FunctionType *fn_type = llvm::FunctionType::get(
      llvm::Type::getVoidTy(module->getContext()), // return type.
      ArgTypes, // parameter type.
      false);   // no variadic arguments.
  llvm::Function *fn = llvm::dyn_cast<llvm::Function>(
      module->getOrInsertFunction(kernel_name, fn_type).getCallee());

  auto call_inst = builder.CreateCall(fn, args);

  // add kernel_path metadata
  llvm::LLVMContext &context = call_inst->getContext();
  llvm::MDNode *mdNode = llvm::MDNode::get(context, llvm::MDString::get(context, attrs[0].getValue()));
  call_inst->setMetadata("kernel_path", mdNode);
}

class ModuleTranslation : public LLVM::ModuleTranslation {

public:
  explicit ModuleTranslation(ModuleOp module)
      : LLVM::ModuleTranslation(module) {}
  ~ModuleTranslation() override {}

protected:
  LogicalResult convertOperation(Operation &opInst,
                        llvm::IRBuilder<> &builder) override {

#include "mlir/Dialect/LLVMIR/MIOpenConversions.inc"

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
                 [](ModuleOp module, llvm::raw_ostream &output) {
                   if (!module)
                     return failure();

                   auto llvmModule = mlir::translateModuleToMIOpenIR(module);
                   if (!llvmModule)
                     return failure();

                   llvmModule->print(output, nullptr);
                   return success();
                 });
