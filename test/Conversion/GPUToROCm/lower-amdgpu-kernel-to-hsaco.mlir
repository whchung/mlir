// RUN: mlir-opt %s --test-kernel-to-hsaco | FileCheck %s

func @kernel(%arg0 : !llvm.float, %arg1 : !llvm<"float*">)
// CHECK: attributes  {amdgpu.hsaco = "HSACO", gpu.kernel}
  attributes  { gpu.kernel } {
// CHECK-NOT: llvm.return
  llvm.return
}
