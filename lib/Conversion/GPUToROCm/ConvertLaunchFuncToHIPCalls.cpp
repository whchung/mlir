//===- ConvertLaunchFuncToHIPCalls.cpp - MLIR HIP lowering passes -------===//
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
// This file implements a pass to convert gpu.launch_func op into a sequence of
// HIP runtime calls. As the HIP runtime does not have a stable published ABI,
// this pass uses a slim runtime layer that builds on top of the public API from
// the HIP headers.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/GPUToROCm/GPUToROCmPass.h"

#include "mlir/GPU/GPUDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Error.h"

using namespace mlir;

// To avoid name mangling, these are defined in the mini-runtime file.
static constexpr const char *hipModuleLoadName = "mhipModuleLoad";
static constexpr const char *hipModuleGetFunctionName = "mhipModuleGetFunction";
static constexpr const char *hipLaunchKernelName = "mhipLaunchKernel";
static constexpr const char *hipGetStreamHelperName = "mhipGetStreamHelper";
static constexpr const char *hipStreamSynchronizeName = "mhipStreamSynchronize";

static constexpr const char *kHSACOGetterAnnotation = "amdgpu.hsacogetter";

namespace {

/// A pass to convert gpu.launch_func operations into a sequence of HIP
/// runtime calls.
///
/// In essence, a gpu.launch_func operations gets compiled into the following
/// sequence of runtime calls:
///
/// * mhipModuleLoad        -- loads the module given the HSACO data
/// * mhipModuleGetFunction -- gets a handle to the actual kernel function
/// * mhipGetStreamHelper   -- initializes a new HIP stream
/// * mhipLaunchKernelName  -- launches the kernel on a stream
/// * mhipStreamSynchronize -- waits for operations on the stream to finish
///
/// Intermediate data structures are allocated on the stack.
class GpuLaunchFuncToHIPCallsPass
    : public ModulePass<GpuLaunchFuncToHIPCallsPass> {
private:
  LLVM::LLVMDialect *getLLVMDialect() { return llvmDialect; }

  llvm::LLVMContext &getLLVMContext() {
    return getLLVMDialect()->getLLVMContext();
  }

  void initializeCachedTypes() {
    const llvm::Module &module = llvmDialect->getLLVMModule();
    llvmPointerType = LLVM::LLVMType::getInt8PtrTy(llvmDialect);
    llvmPointerPointerType = llvmPointerType.getPointerTo();
    llvmInt8Type = LLVM::LLVMType::getInt8Ty(llvmDialect);
    llvmInt32Type = LLVM::LLVMType::getInt32Ty(llvmDialect);
    llvmInt64Type = LLVM::LLVMType::getInt64Ty(llvmDialect);
    llvmIntPtrType = LLVM::LLVMType::getIntNTy(
        llvmDialect, module.getDataLayout().getPointerSizeInBits());
  }

  LLVM::LLVMType getPointerType() { return llvmPointerType; }

  LLVM::LLVMType getPointerPointerType() { return llvmPointerPointerType; }

  LLVM::LLVMType getInt8Type() { return llvmInt8Type; }

  LLVM::LLVMType getInt32Type() { return llvmInt32Type; }

  LLVM::LLVMType getInt64Type() { return llvmInt64Type; }

  LLVM::LLVMType getIntPtrType() {
    const llvm::Module &module = getLLVMDialect()->getLLVMModule();
    return LLVM::LLVMType::getIntNTy(
        getLLVMDialect(), module.getDataLayout().getPointerSizeInBits());
  }

  LLVM::LLVMType getCUResultType() {
    // This is declared as an enum in HIP but helpers use i32.
    return getInt32Type();
  }

  // Allocate a void pointer on the stack.
  Value *allocatePointer(OpBuilder &builder, Location loc) {
    auto one = builder.create<LLVM::ConstantOp>(loc, getInt32Type(),
                                                builder.getI32IntegerAttr(1));
    return builder.create<LLVM::AllocaOp>(loc, getPointerPointerType(), one);
  }

  void declareHIPFunctions(Location loc);
  Value *setupParamsArray(gpu::LaunchFuncOp launchOp, OpBuilder &builder);
  Value *generateKernelNameConstant(FuncOp kernelFunction, Location &loc,
                                    OpBuilder &builder);
  void translateGpuLaunchCalls(mlir::gpu::LaunchFuncOp launchOp);

public:
  // Run the dialect converter on the module.
  void runOnModule() override {
    // Cache the LLVMDialect for the current module.
    llvmDialect = getContext().getRegisteredDialect<LLVM::LLVMDialect>();
    // Cache the used LLVM types.
    initializeCachedTypes();

    for (auto func : getModule().getOps<FuncOp>()) {
      func.walk<mlir::gpu::LaunchFuncOp>(
          [this](mlir::gpu::LaunchFuncOp op) { translateGpuLaunchCalls(op); });
    }
  }

private:
  LLVM::LLVMDialect *llvmDialect;
  LLVM::LLVMType llvmPointerType;
  LLVM::LLVMType llvmPointerPointerType;
  LLVM::LLVMType llvmInt8Type;
  LLVM::LLVMType llvmInt32Type;
  LLVM::LLVMType llvmInt64Type;
  LLVM::LLVMType llvmIntPtrType;
};

} // anonymous namespace

// Adds declarations for the needed helper functions from the HIP wrapper.
// The types in comments give the actual types expected/returned but the API
// uses void pointers. This is fine as they have the same linkage in C.
void GpuLaunchFuncToHIPCallsPass::declareHIPFunctions(Location loc) {
  ModuleOp module = getModule();
  Builder builder(module);
  if (!module.lookupSymbol<FuncOp>(hipModuleLoadName)) {
    module.push_back(
        FuncOp::create(loc, hipModuleLoadName,
                       builder.getFunctionType(
                           {
                               getPointerPointerType(), /* CUmodule *module */
                               getPointerType()         /* void *HSACO */
                           },
                           getCUResultType())));
  }
  if (!module.lookupSymbol<FuncOp>(hipModuleGetFunctionName)) {
    // The helper uses void* instead of HIP's opaque CUmodule and
    // CUfunction.
    module.push_back(
        FuncOp::create(loc, hipModuleGetFunctionName,
                       builder.getFunctionType(
                           {
                               getPointerPointerType(), /* void **function */
                               getPointerType(),        /* void *module */
                               getPointerType()         /* char *name */
                           },
                           getCUResultType())));
  }
  if (!module.lookupSymbol<FuncOp>(hipLaunchKernelName)) {
    // Other than the HIP api, the wrappers use uintptr_t to match the
    // LLVM type if MLIR's index type, which the GPU dialect uses.
    // Furthermore, they use void* instead of HIP's opaque CUfunction and
    // CUstream.
    module.push_back(FuncOp::create(
        loc, hipLaunchKernelName,
        builder.getFunctionType(
            {
                getPointerType(),        /* void* f */
                getIntPtrType(),         /* intptr_t gridXDim */
                getIntPtrType(),         /* intptr_t gridyDim */
                getIntPtrType(),         /* intptr_t gridZDim */
                getIntPtrType(),         /* intptr_t blockXDim */
                getIntPtrType(),         /* intptr_t blockYDim */
                getIntPtrType(),         /* intptr_t blockZDim */
                getInt32Type(),          /* unsigned int sharedMemBytes */
                getPointerType(),        /* void *hstream */
                getPointerPointerType(), /* void **kernelParams */
                getPointerPointerType()  /* void **extra */
            },
            getCUResultType())));
  }
  if (!module.lookupSymbol<FuncOp>(hipGetStreamHelperName)) {
    // Helper function to get the current HIP stream. Uses void* instead of
    // HIPs opaque CUstream.
    module.push_back(FuncOp::create(
        loc, hipGetStreamHelperName,
        builder.getFunctionType({}, getPointerType() /* void *stream */)));
  }
  if (!module.lookupSymbol<FuncOp>(hipStreamSynchronizeName)) {
    module.push_back(
        FuncOp::create(loc, hipStreamSynchronizeName,
                       builder.getFunctionType(
                           {
                               getPointerType() /* CUstream stream */
                           },
                           getCUResultType())));
  }
}

// Generates a parameters array to be used with a HIP kernel launch call. The
// arguments are extracted from the launchOp.
// The generated code is essentially as follows:
//
// %array = alloca(numparams * sizeof(void *))
// for (i : [0, NumKernelOperands))
//   %array[i] = cast<void*>(KernelOperand[i])
// return %array
Value *GpuLaunchFuncToHIPCallsPass::setupParamsArray(gpu::LaunchFuncOp launchOp,
                                                     OpBuilder &builder) {
  Location loc = launchOp.getLoc();
  auto one = builder.create<LLVM::ConstantOp>(loc, getInt32Type(),
                                              builder.getI32IntegerAttr(1));
  auto arraySize = builder.create<LLVM::ConstantOp>(
      loc, getInt32Type(),
      builder.getI32IntegerAttr(launchOp.getNumKernelOperands()));
  auto array =
      builder.create<LLVM::AllocaOp>(loc, getPointerPointerType(), arraySize);
  for (int idx = 0, e = launchOp.getNumKernelOperands(); idx < e; ++idx) {
    auto operand = launchOp.getKernelOperand(idx);
    auto llvmType = operand->getType().cast<LLVM::LLVMType>();
    auto memLocation =
        builder.create<LLVM::AllocaOp>(loc, llvmType.getPointerTo(), one);
    builder.create<LLVM::StoreOp>(loc, operand, memLocation);
    auto casted =
        builder.create<LLVM::BitcastOp>(loc, getPointerType(), memLocation);
    auto index = builder.create<LLVM::ConstantOp>(
        loc, getInt32Type(), builder.getI32IntegerAttr(idx));
    auto gep = builder.create<LLVM::GEPOp>(loc, getPointerPointerType(), array,
                                           ArrayRef<Value *>{index});
    builder.create<LLVM::StoreOp>(loc, casted, gep);
  }
  return array;
}

// Generates LLVM IR that produces a value representing the name of the
// given kernel function. The generated IR consists essentially of the
// following:
//
// %0 = alloca(strlen(name) + 1)
// %0[0] = constant name[0]
// ...
// %0[n] = constant name[n]
// %0[n+1] = 0
Value *GpuLaunchFuncToHIPCallsPass::generateKernelNameConstant(
    FuncOp kernelFunction, Location &loc, OpBuilder &builder) {
  // TODO(herhut): Make this a constant once this is supported.
  auto kernelNameSize = builder.create<LLVM::ConstantOp>(
      loc, getInt32Type(),
      builder.getI32IntegerAttr(kernelFunction.getName().size() + 1));
  auto kernelName =
      builder.create<LLVM::AllocaOp>(loc, getPointerType(), kernelNameSize);
  for (auto byte : llvm::enumerate(kernelFunction.getName())) {
    auto index = builder.create<LLVM::ConstantOp>(
        loc, getInt32Type(), builder.getI32IntegerAttr(byte.index()));
    auto gep = builder.create<LLVM::GEPOp>(loc, getPointerType(), kernelName,
                                           ArrayRef<Value *>{index});
    auto value = builder.create<LLVM::ConstantOp>(
        loc, getInt8Type(),
        builder.getIntegerAttr(builder.getIntegerType(8), byte.value()));
    builder.create<LLVM::StoreOp>(loc, value, gep);
  }
  // Add trailing zero to terminate string.
  auto index = builder.create<LLVM::ConstantOp>(
      loc, getInt32Type(),
      builder.getI32IntegerAttr(kernelFunction.getName().size()));
  auto gep = builder.create<LLVM::GEPOp>(loc, getPointerType(), kernelName,
                                         ArrayRef<Value *>{index});
  auto value = builder.create<LLVM::ConstantOp>(
      loc, getInt8Type(), builder.getIntegerAttr(builder.getIntegerType(8), 0));
  builder.create<LLVM::StoreOp>(loc, value, gep);
  return kernelName;
}

// Emits LLVM IR to launch a kernel function. Expects the module that contains
// the compiled kernel function as a HSACO in the 'amdgpu.HSACO' attribute of
// the kernel function in the IR. While MLIR has no global constants, also
// expects a HSACO getter function in an 'amdgpu.hsacogetter' attribute. Such
// function is expected to return a pointer to the HSACO blob when invoked. With
// these given, the generated code in essence is
//
// %0 = call %hsacogetter
// %1 = alloca sizeof(void*)
// call %mhipModuleLoad(%2, %1)
// %2 = alloca sizeof(void*)
// %3 = load %1
// %4 = <see generateKernelNameConstant>
// call %mhipModuleGetFunction(%2, %3, %4)
// %5 = call %mhipGetStreamHelper()
// %6 = load %2
// %7 = <see setupParamsArray>
// call %mhipLaunchKernel(%6, <launchOp operands 0..5>, 0, %5, %7, nullptr)
// call %mhipStreamSynchronize(%5)
void GpuLaunchFuncToHIPCallsPass::translateGpuLaunchCalls(
    mlir::gpu::LaunchFuncOp launchOp) {
  OpBuilder builder(launchOp);
  Location loc = launchOp.getLoc();
  declareHIPFunctions(loc);

  auto zero = builder.create<LLVM::ConstantOp>(loc, getInt32Type(),
                                               builder.getI32IntegerAttr(0));
  // Emit a call to the HSACO getter to retrieve a pointer to the data that
  // represents the HSACO at runtime.
  // TODO(herhut): This should rather be a static global once supported.
  auto kernelFunction = getModule().lookupSymbol<FuncOp>(launchOp.kernel());
  auto hsacoGetter =
      kernelFunction.getAttrOfType<SymbolRefAttr>(kHSACOGetterAnnotation);
  if (!hsacoGetter) {
    kernelFunction.emitError("Missing ")
        << kHSACOGetterAnnotation << " attribute.";
    return signalPassFailure();
  }
  auto data = builder.create<LLVM::CallOp>(
      loc, ArrayRef<Type>{getPointerType()}, hsacoGetter, ArrayRef<Value *>{});
  // Emit the load module call to load the module data. Error checking is done
  // in the called helper function.
  auto hipModule = allocatePointer(builder, loc);
  FuncOp hipModuleLoad = getModule().lookupSymbol<FuncOp>(hipModuleLoadName);
  builder.create<LLVM::CallOp>(loc, ArrayRef<Type>{getCUResultType()},
                               builder.getSymbolRefAttr(hipModuleLoad),
                               ArrayRef<Value *>{hipModule, data.getResult(0)});
  // Get the function from the module. The name corresponds to the name of
  // the kernel function.
  auto hipOwningModuleRef =
      builder.create<LLVM::LoadOp>(loc, getPointerType(), hipModule);
  auto kernelName = generateKernelNameConstant(kernelFunction, loc, builder);
  auto hipFunction = allocatePointer(builder, loc);
  FuncOp hipModuleGetFunction =
      getModule().lookupSymbol<FuncOp>(hipModuleGetFunctionName);
  builder.create<LLVM::CallOp>(
      loc, ArrayRef<Type>{getCUResultType()},
      builder.getSymbolRefAttr(hipModuleGetFunction),
      ArrayRef<Value *>{hipFunction, hipOwningModuleRef, kernelName});
  // Grab the global stream needed for execution.
  FuncOp hipGetStreamHelper =
      getModule().lookupSymbol<FuncOp>(hipGetStreamHelperName);
  auto hipStream = builder.create<LLVM::CallOp>(
      loc, ArrayRef<Type>{getPointerType()},
      builder.getSymbolRefAttr(hipGetStreamHelper), ArrayRef<Value *>{});
  // Invoke the function with required arguments.
  auto hipLaunchKernel = getModule().lookupSymbol<FuncOp>(hipLaunchKernelName);
  auto hipFunctionRef =
      builder.create<LLVM::LoadOp>(loc, getPointerType(), hipFunction);
  auto paramsArray = setupParamsArray(launchOp, builder);
  auto nullpointer =
      builder.create<LLVM::IntToPtrOp>(loc, getPointerPointerType(), zero);
  builder.create<LLVM::CallOp>(
      loc, ArrayRef<Type>{getCUResultType()},
      builder.getSymbolRefAttr(hipLaunchKernel),
      ArrayRef<Value *>{hipFunctionRef, launchOp.getOperand(0),
                        launchOp.getOperand(1), launchOp.getOperand(2),
                        launchOp.getOperand(3), launchOp.getOperand(4),
                        launchOp.getOperand(5), zero, /* sharedMemBytes */
                        hipStream.getResult(0),       /* stream */
                        paramsArray,                  /* kernel params */
                        nullpointer /* extra */});
  // Sync on the stream to make it synchronous.
  auto hipStreamSync =
      getModule().lookupSymbol<FuncOp>(hipStreamSynchronizeName);
  builder.create<LLVM::CallOp>(loc, ArrayRef<Type>{getCUResultType()},
                               builder.getSymbolRefAttr(hipStreamSync),
                               ArrayRef<Value *>(hipStream.getResult(0)));
  launchOp.erase();
}

mlir::ModulePassBase *mlir::createConvertGpuLaunchFuncToHIPCallsPass() {
  return new GpuLaunchFuncToHIPCallsPass();
}

static PassRegistration<GpuLaunchFuncToHIPCallsPass>
    pass("launch-func-to-hip",
         "Convert all launch_func ops to HIP runtime calls");
