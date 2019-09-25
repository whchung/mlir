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

include "mlir/IR/OpBase.td"

def MIOpen_Dialect : Dialect {
  let name = "miopen";
  let cppNamespace = "miopen";
}

class MIOpen_KernelFunctionOp<string mnemonic, string kernelFunction, Type memrefType, list<OpTrait> traits = []> :
  Op<MIOpen_Dialect, mnemonic, traits>,
  Results<(outs memrefType:$res)>,
  Arguments<(ins memrefType:$input, memrefType:$output, memrefType:$weight)> {
  let parser = [{ return parseMIOpenOp(parser, result); }];
  let printer = [{ printMIOpenOp(p, this->getOperation()); }];

  // TBD should MIOpen_KernelFunctionOp be a subclass of LLVM_OpBase?
  //string llvmBuilder = "createMIOpenKernelFunctionCall(builder, \""# kernelFunction # "\");";
}

def MIOpen_Conv2D_F32Op : MIOpen_KernelFunctionOp<"conv2d.f32", "miopen_conv2d_f32_kernel", F32MemRef>;
def MIOpen_Conv2D_F16Op : MIOpen_KernelFunctionOp<"conv2d.f16", "miopen_conv2d_f16_kernel", F16MemRef>;
def MIOpen_Conv2D_BF16Op : MIOpen_KernelFunctionOp<"conv2d.bf16", "miopen_conv2d_bf16_kernel", BF16MemRef>;

#endif // MIOpenIR_OPS
