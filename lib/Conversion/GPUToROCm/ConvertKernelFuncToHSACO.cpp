//===- ConvertKernelFuncToHSACO.cpp - MLIR GPU lowering passes ------------===//
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
// This file implements a pass to convert gpu kernel functions into a
// corresponding binary blob that can be executed on a CUDA GPU. Currently
// only translates the function itself but no dependencies.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/GPUToROCm/GPUToROCmPass.h"

#include "mlir/GPU/GPUDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/ROCDLIR.h"

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
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
static constexpr const char *kHSACOAnnotation = "amdgpu.hsaco";

/// A pass converting tagged kernel functions to hsaco blobs.
class GpuKernelToHSACOPass : public ModulePass<GpuKernelToHSACOPass> {
public:
  GpuKernelToHSACOPass() : emitHSACO_(false) {}
  GpuKernelToHSACOPass(bool emitHSACO) : emitHSACO_(emitHSACO) {}

  // Run the dialect converter on the module.
  void runOnModule() override {
    // Make sure the AMDGPU target is initialized.
    LLVMInitializeAMDGPUTarget();
    LLVMInitializeAMDGPUTargetInfo();
    LLVMInitializeAMDGPUTargetMC();
    LLVMInitializeAMDGPUAsmPrinter();

    for (auto function : getModule().getOps<FuncOp>()) {
      if (!gpu::GPUDialect::isKernel(function) || function.isExternal()) {
        continue;
      }
      if (failed(translateGpuKernelToHSACOAnnotation(function)))
        signalPassFailure();
    }
  }

private:
  static OwnedHSACO convertModuleToHSACOForTesting(llvm::Module &llvmModule,
                                                   FuncOp &function);

  OwnedHSACO convertModuleToHSACO(llvm::Module &llvmModule, FuncOp &function);
  LogicalResult translateGpuKernelToHSACOAnnotation(FuncOp &function);

  bool emitHSACO_;
};

} // anonymous namespace

OwnedHSACO
GpuKernelToHSACOPass::convertModuleToHSACOForTesting(llvm::Module &llvmModule,
                                                     FuncOp &function) {
  const char data[] = "HSACO";
  return llvm::make_unique<std::vector<char>>(data, data + sizeof(data) - 1);
}

// Gets the ROCm-Device-Libs filenames for a particular AMDGPU version.
static std::vector<std::string>
GetROCDLPaths(int amdgpu_version, const std::string &rocdl_dir_path) {
  // AMDGPU version-neutral bitcodes.
  static std::vector<std::string> *rocdl_filenames =
      new std::vector<std::string>({"hc.amdgcn.bc", "opencl.amdgcn.bc",
                                    "ocml.amdgcn.bc", "ockl.amdgcn.bc",
                                    "oclc_finite_only_off.amdgcn.bc",
                                    "oclc_daz_opt_off.amdgcn.bc",
                                    "oclc_correctly_rounded_sqrt_on.amdgcn.bc",
                                    "oclc_unsafe_math_off.amdgcn.bc"});

  // Construct full path to ROCDL bitcode libraries.
  std::vector<std::string> result;
  for (auto &filename : *rocdl_filenames) {
    llvm::SmallString<128> appended_path;
    llvm::sys::path::append(appended_path, rocdl_dir_path, filename);
    result.push_back(appended_path.c_str());
  }

  // Add AMDGPU version-specific bitcodes.
  llvm::SmallString<128> amdgpu_version_bitcode_path;
  std::string amdgpu_version_bitcode_filename =
      std::string("oclc_isa_version_") + std::to_string(amdgpu_version) +
      ".amdgcn.bc";
  llvm::sys::path::append(amdgpu_version_bitcode_path, rocdl_dir_path,
                          amdgpu_version_bitcode_filename);
  result.push_back(amdgpu_version_bitcode_path.c_str());
  return std::move(result);
}

static void DieWithSMDiagnosticError(llvm::SMDiagnostic *diagnostic) {
  llvm::errs() << diagnostic->getFilename().str() << ":"
               << diagnostic->getLineNo() << ":" << diagnostic->getColumnNo()
               << ": " << diagnostic->getMessage().str();
}

std::unique_ptr<llvm::Module> LoadIRModule(const std::string &filename,
                                           llvm::LLVMContext *llvm_context) {
  llvm::SMDiagnostic diagnostic_err;
  std::unique_ptr<llvm::Module> module(
      llvm::parseIRFile(llvm::StringRef(filename.data(), filename.size()),
                        diagnostic_err, *llvm_context));

  if (module == nullptr) {
    DieWithSMDiagnosticError(&diagnostic_err);
  }

  return module;
}

// Links the module with a vector of path to bitcode modules.
// The caller must guarantee that the paths exist.
void LinkWithBitcodeVector(
    llvm::Module *module, const std::vector<std::string> &bitcode_path_vector) {
  llvm::Linker linker(*module);

  for (auto &bitcode_path : bitcode_path_vector) {
    if (!llvm::sys::fs::exists(bitcode_path)) {
      llvm::errs() << "bitcode module is required by this MLIR module but was "
                      "not found at "
                   << bitcode_path;
      return;
    }

    std::unique_ptr<llvm::Module> bitcode_module =
        LoadIRModule(bitcode_path, &module->getContext());
    if (linker.linkInModule(
            std::move(bitcode_module), llvm::Linker::Flags::LinkOnlyNeeded,
            [](llvm::Module &M, const llvm::StringSet<> &GVS) {
              internalizeModule(M, [&M, &GVS](const llvm::GlobalValue &GV) {
                return !GV.hasName() || (GVS.count(GV.getName()) == 0);
              });
            })) {
      llvm::errs() << "Error linking bitcode module from " << bitcode_path;
      return;
    }
  }
}

// Returns whether the module could use any device bitcode library functions.
// This function may have false positives -- the module might not use libdevice
// on NVPTX or ROCm-Device-Libs on AMDGPU even if this function returns true.
bool CouldNeedDeviceBitcode(const llvm::Module &module) {
  for (const llvm::Function &function : module.functions()) {
    // This is a conservative approximation -- not all such functions are in
    // libdevice or ROCm-Device-Libs.
    if (!function.isIntrinsic() && function.isDeclaration()) {
      return true;
    }
  }
  return false;
}

// Links ROCm-Device-Libs into the given module if the module needs it.
void LinkROCDLIfNecessary(llvm::Module *module, int amdgpu_version,
                          const std::string &rocdl_dir_path) {
  if (!CouldNeedDeviceBitcode(*module)) {
    return;
  }

  LinkWithBitcodeVector(module, GetROCDLPaths(amdgpu_version, rocdl_dir_path));
}

// Emits the given module to HSA Code Object. target_machine is an initialized
// TargetMachine for the AMDGPU target.
std::vector<char> emitModuleToHSACO(llvm::Module *module,
                                    llvm::TargetMachine *target_machine) {
  static char tempdir_template[] = "/tmp/amdgpu_mlir-XXXXXX";
  char *tempdir_name = mkdtemp(tempdir_template);

  // prepare filenames for all stages of compilation:
  // IR, ISA, binary ISA, and HSACO
  llvm::Twine ir_filename = llvm::Twine(module->getModuleIdentifier()) + ".ll";
  llvm::SmallString<128> ir_path;
  llvm::sys::path::append(ir_path, tempdir_name, ir_filename);

  llvm::Twine isabin_filename =
      llvm::Twine(module->getModuleIdentifier()) + ".o";
  llvm::SmallString<128> isabin_path;
  llvm::sys::path::append(isabin_path, tempdir_name, isabin_filename);

  llvm::Twine hsaco_filename =
      llvm::Twine(module->getModuleIdentifier()) + ".hsaco";
  llvm::SmallString<128> hsaco_path;
  llvm::sys::path::append(hsaco_path, tempdir_name, hsaco_filename);

  std::error_code ec;

  // dump LLVM IR
  std::unique_ptr<llvm::raw_fd_ostream> ir_fs(
      new llvm::raw_fd_ostream(ir_path, ec, llvm::sys::fs::F_None));
  module->print(*ir_fs, nullptr);
  ir_fs->flush();

  //// emit GCN ISA binary
  llvm::legacy::PassManager codegen_passes;
  llvm::SmallVector<char, 0> stream;
  llvm::raw_svector_ostream pstream(stream);
  std::unique_ptr<llvm::raw_fd_ostream> isabin_fs(
      new llvm::raw_fd_ostream(isabin_path, ec, llvm::sys::fs::F_Text));
  module->setDataLayout(target_machine->createDataLayout());
  target_machine->addPassesToEmitFile(codegen_passes, *isabin_fs, nullptr,
                                      llvm::TargetMachine::CGFT_ObjectFile);
  codegen_passes.run(*module);
  isabin_fs->flush();

  // Locate lld
  // ROCM TODO: make ld.lld path configurable.
  llvm::StringRef lld_program("/opt/rocm/hcc/bin/ld.lld");
  std::vector<llvm::StringRef> lld_args{
      llvm::StringRef("ld.lld"),      llvm::StringRef("-flavor"),
      llvm::StringRef("gnu"),         llvm::StringRef("-shared"),
      llvm::StringRef("isabin_path"), llvm::StringRef("-o"),
      llvm::StringRef("hsaco_path"),
  };
  lld_args[4] = llvm::StringRef(isabin_path.c_str());
  lld_args[6] = llvm::StringRef(hsaco_path.c_str());

  std::string error_message;
  int lld_result = llvm::sys::ExecuteAndWait(
      lld_program, llvm::ArrayRef<llvm::StringRef>(lld_args), llvm::None, {}, 0,
      0, &error_message);

  if (lld_result) {
    llvm::errs() << "ld.lld execute fail: " << error_message;
  }

  // read HSACO
  std::ifstream hsaco_file(hsaco_path.c_str(),
                           std::ios::binary | std::ios::ate);
  std::ifstream::pos_type hsaco_file_size = hsaco_file.tellg();

  std::vector<char> hsaco(hsaco_file_size);
  hsaco_file.seekg(0, std::ios::beg);
  hsaco_file.read(reinterpret_cast<char *>(&hsaco[0]), hsaco_file_size);
  return std::move(hsaco);
}

OwnedHSACO GpuKernelToHSACOPass::convertModuleToHSACO(llvm::Module &llvmModule,
                                                      FuncOp &function) {
  // Skip compilation in case of testing.
  if (!emitHSACO_) {
    return GpuKernelToHSACOPass::convertModuleToHSACOForTesting(llvmModule,
                                                                function);
  }

  // ROCM TODO: make AMD GCN ISA version be configurable.
  int amdgpu_version = 900;

  // Construct LLVM TargetMachine for AMDGPU target.
  std::unique_ptr<llvm::TargetMachine> targetMachine;
  {
    std::string error;
    llvm::Triple triple("amdgcn--amdhsa-amdgiz");
    const llvm::Target *target =
        llvm::TargetRegistry::lookupTarget("", triple, error);
    if (target == nullptr) {
      function.emitError("Cannot initialize target triple");
      return {};
    }
    // ROCM TODO: set AMD GCN ISA version.
    // ROCM TODO: support HSA code object format v3.
    std::string mcpu_str = std::string("gfx") + std::to_string(amdgpu_version);
    targetMachine.reset(target->createTargetMachine(triple.str(), mcpu_str,
                                                    "-code-object-v3", {}, {}));
  }

  // Set the data layout of the llvm module to match what the target needs.
  llvmModule.setDataLayout(targetMachine->createDataLayout());

  // ROCM TODO: make ROCDL bitcode library path configurable.
  LinkROCDLIfNecessary(&llvmModule, amdgpu_version, "/opt/rocm/lib");

  // Lower LLVM module to HSA code object;
  std::vector<char> hsaco = emitModuleToHSACO(&llvmModule, targetMachine.get());

  return llvm::make_unique<std::vector<char>>(hsaco);
}

LogicalResult
GpuKernelToHSACOPass::translateGpuKernelToHSACOAnnotation(FuncOp &function) {
  Builder builder(function.getContext());

  OwningModuleRef module = ModuleOp::create(function.getLoc());

  // TODO(herhut): Also handle called functions.
  module->push_back(function.clone());

  auto llvmModule = translateModuleToROCDLIR(*module);
  auto hsaco = convertModuleToHSACO(*llvmModule, function);

  if (!hsaco) {
    return function.emitError("Translation to HSA code object failed.");
  }

  function.setAttr(kHSACOAnnotation,
                   builder.getStringAttr({hsaco->data(), hsaco->size()}));

  // Remove the body of the kernel function now that it has been translated.
  // The main reason to do this is so that the resulting module no longer
  // contains kernel instructions, and hence can be compiled into host code by
  // a separate pass.
  function.eraseBody();

  return success();
}

ModulePassBase *mlir::createConvertGPUKernelToHSACOPass(bool emitHSACO) {
  return new GpuKernelToHSACOPass(emitHSACO);
}

static PassRegistration<GpuKernelToHSACOPass>
    pass("test-kernel-to-hsaco",
         "Convert all kernel functions to HSA code object blobs");
