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

//===----------------------------------------------------------------------===//
// Base class for high-level ops                                              //
//===----------------------------------------------------------------------===//

class MIOpen_HighLevelOp<string mnemonic, list<OpTrait> traits = []> :
  Op<MIOpen_Dialect, mnemonic, traits> {
}

//===----------------------------------------------------------------------===//
// High-level convolution ops                                                 //
//===----------------------------------------------------------------------===//

class MIOpen_Conv2DOp<string mnemonic, Type memRefType, list<OpTrait> traits = []> :
  MIOpen_HighLevelOp<mnemonic, traits>,
  Results<(outs memRefType:$res)>,
  Arguments<(ins memRefType:$input, memRefType:$output, memRefType:$weight)> {

  let parser = [{ return parseMIOpenConv2DOp(parser, result); }];
  let printer = [{ return printMIOpenConv2DOp(p, this->getOperation()); }];
}

def MIOpen_Conv2D_F32Op : MIOpen_Conv2DOp<"conv2d.f32", F32MemRef>;
def MIOpen_Conv2D_F16Op : MIOpen_Conv2DOp<"conv2d.f16", F16MemRef>;
def MIOpen_Conv2D_BF16Op : MIOpen_Conv2DOp<"conv2d.bf16", BF16MemRef>;

//===----------------------------------------------------------------------===//
// Experimental high-level dummy op                                           //
//===----------------------------------------------------------------------===//

def MIOpen_HighLevelDummyOp : MIOpen_HighLevelOp<"dummy.high">,
  Results<(outs AnyMemRef:$res)>,
  Arguments<(ins AnyMemRef:$input)> {

  let parser = [{ return parseMIOpenDummyOp(parser, result); }];
  let printer = [{ printMIOpenDummyOp(p, this->getOperation()); }];
}

//===----------------------------------------------------------------------===//
// Base class for low-level ops                                               //
//===----------------------------------------------------------------------===//

class MIOpen_LowLevelOp<string mnemonic, list<OpTrait> traits = []> :
  LLVM_OpBase<MIOpen_Dialect, mnemonic, traits> {
}

//===----------------------------------------------------------------------===//
// Low-level kernel function ops                                              //
//===----------------------------------------------------------------------===//

class MIOpen_KernelFunctionOp<string mnemonic, string kernelFunction, list<OpTrait> traits = []> :
  MIOpen_LowLevelOp<mnemonic, traits>,
  Results<(outs LLVM_Type:$res)>,
  Arguments<(ins LLVM_Type:$input, LLVM_Type:$output, LLVM_Type:$weight)> {

  let parser = [{ return parseMIOpenKernelFunctionOp(parser, result); }];
  let printer = [{ printMIOpenKernelFunctionOp(p, this->getOperation()); }];

  string llvmBuilder = "$res = createMIOpenKernelFunctionCall(builder, \""# kernelFunction # "\", {$input, $output, $weight});";
}

def MIOpen_Conv2D_KernelFunction_F32Op : MIOpen_KernelFunctionOp<"conv2d.kernel.f32", "miopen_conv2d_f32_kernel">;
def MIOpen_Conv2D_F16_KernelFunction_F16Op : MIOpen_KernelFunctionOp<"conv2d.kernel.f16", "miopen_conv2d_f16_kernel">;
def MIOpen_Conv2D_KernelFunction_BF16Op : MIOpen_KernelFunctionOp<"conv2d.kernel.bf16", "miopen_conv2d_bf16_kernel">;

//===----------------------------------------------------------------------===//
// Experimental low-level dummy op                                            //
//===----------------------------------------------------------------------===//

def MIOpen_LowLevelDummyOp : MIOpen_LowLevelOp<"dummy.low">,
  Results<(outs LLVM_Type:$res)>,
  Arguments<(ins LLVM_Type:$input)> {

  let parser = [{ return parseMIOpenDummyOp(parser, result); }];
  let printer = [{ printMIOpenDummyOp(p, this->getOperation()); }];
  string llvmBuilder = "$res = createMIOpenDummyCall(builder, \"miopen_dummy_low\", {$input});";
}

#endif // MIOpenIR_OPS
