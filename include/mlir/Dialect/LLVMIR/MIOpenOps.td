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

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"

def MIOpen_Dialect : Dialect {
  let name = "miopen";
  let cppNamespace = "miopen";
}

class MIOpen_KernelFunctionOp<string mnemonic, string kernelFunction, list<OpTrait> traits = []> :
  LLVM_OpBase<MIOpen_Dialect, mnemonic, traits>,
  Results<(outs LLVM_Type:$res)>,
  Arguments<(ins LLVM_Type:$input, LLVM_Type:$output, LLVM_Type:$weight)> {

  let parser = [{ return parseMIOpenOp(parser, result); }];
  let printer = [{ printMIOpenOp(p, this->getOperation()); }];

  string llvmBuilder = [{
      $res = createMIOpenKernelFunctionCall(builder, \""# kernelFunction # "\", {$input, $output, $weight});
  }];
}

def MIOpen_Conv2D_F32Op : MIOpen_KernelFunctionOp<"conv2d.f32", "miopen_conv2d_f32_kernel">;
def MIOpen_Conv2D_F16Op : MIOpen_KernelFunctionOp<"conv2d.f16", "miopen_conv2d_f16_kernel">;
def MIOpen_Conv2D_BF16Op : MIOpen_KernelFunctionOp<"conv2d.bf16", "miopen_conv2d_bf16_kernel">;

#endif // MIOpenIR_OPS
