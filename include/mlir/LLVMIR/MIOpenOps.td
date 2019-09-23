//===-- MIOpenOps.td - MIOpen IR dialect op definition file ----*- tablegen -*-===//
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
// This is the MIOpen IR operation definition file.
//
//===----------------------------------------------------------------------===//

#ifdef MIOpenIR_OPS
#else
#define MIOpenIR_OPS

include "mlir/LLVMIR/LLVMOpBase.td"

def MIOpen_Dialect : Dialect {
  let name = "miopen";
  let cppNamespace = "miopen";
}

class MIOpen_KernelFunctionOp<string mnemonic, string kernel_function, list<OpTrait> traits = []> :
  LLVM_OpBase<MIOpen_Dialect, mnemonic, traits>,
  Arguments<(ins F32Tensor:$input, F32Tensor:$output, F32Tensor:$weight)> {
  string llvmBuilder = "createMIOpenKernelFunctionCall(builder, \""# kernel_function # "\");";
}

def MIOpen_Conv2DOp : MIOpen_KernelFunctionOp<"conv2d", "miopen_conv2d_kernel">;

#endif // MIOpenIR_OPS
