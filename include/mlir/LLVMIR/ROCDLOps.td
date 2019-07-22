//===-- ROCDLOps.td - ROCDL IR dialect op definition file ----*- tablegen -*-===//
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
// This is the ROCDL IR operation definition file.
//
//===----------------------------------------------------------------------===//

#ifdef ROCDLIR_OPS
#else
#define ROCDLIR_OPS

include "mlir/LLVMIR/LLVMOpBase.td"

def ROCDL_Dialect : Dialect {
  let name = "rocdl";
  let cppNamespace = "ROCDL";
}

class ROCDL_Op<string mnemonic, list<OpTrait> traits = []> :
  LLVM_OpBase<ROCDL_Dialect, mnemonic, traits> {
}

class ROCDL_SpecialRegisterOp<string mnemonic,
    list<OpTrait> traits = []> :
  ROCDL_Op<mnemonic, !listconcat(traits, [NoSideEffect])>,
  Results<(outs LLVM_Type:$res)>, Arguments<(ins)> {
  string llvmBuilder = "$res = createIntrinsicCall(builder,"
    # "llvm::Intrinsic::amdgcn_" # !subst(".","_", mnemonic) # ");";
  let parser = [{ return parseROCDLSpecialRegisterOp(parser, result); }];
  let printer = [{ printROCDLSpecialRegisterOp(p, this->getOperation()); }];
}

def ROCDL_ThreadIdXOp : ROCDL_SpecialRegisterOp<"workitem.id.x">;
def ROCDL_ThreadIdYOp : ROCDL_SpecialRegisterOp<"workitem.id.y">;
def ROCDL_ThreadIdZOp : ROCDL_SpecialRegisterOp<"workitem.id.z">;
def ROCDL_BlockIdXOp : ROCDL_SpecialRegisterOp<"workgroup.id.x">;
def ROCDL_BlockIdYOp : ROCDL_SpecialRegisterOp<"workgroup.id.y">;
def ROCDL_BlockIdZOp : ROCDL_SpecialRegisterOp<"workgroup.id.z">;

// ROCM TODO: add ROCDL_DeviceFunctionOp class
//def ROCDL_BlockDimXOp : ROCDL_DeviceFunctionOp<"__ockl_get_local_size.x">;
//def ROCDL_BlockDimYOp : ROCDL_DeviceFunctionOp<"__ockl_get_local_size.y">;
//def ROCDL_BlockDimZOp : ROCDL_DeviceFunctionOp<"__ockl_get_local_size.z">;
//def ROCDL_GridDimXOp : ROCDL_DeviceFunctionOp<"__ockl_get_global_size.x">;
//def ROCDL_GridDimYOp : ROCDL_DeviceFunctionOp<"__ockl_get_global_size.y">;
//def ROCDL_GridDimZOp : ROCDL_DeviceFunctionOp<"__ockl_get_global_size.z">;

#endif // ROCDLIR_OPS
