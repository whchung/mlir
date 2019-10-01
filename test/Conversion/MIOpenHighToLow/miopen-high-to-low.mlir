// RUN: mlir-opt -miopen-high-to-low -lower-to-llvm %s | FileCheck %s

// CHECK-LABEL: func @foo(%{{.*}}: !llvm<"{ float*, [1 x i64] }">)
func @foo(%arg0: memref<16xf32>) {
  // CHECK: llvm.extractvalue %{{.*}}[0 : index] : !llvm<"{ float*, [1 x i64] }">
  // CHECK-NEXT: miopen.dummy.low %{{.*}} : !llvm<"float*">
  miopen.dummy.high %arg0 : memref<16xf32>
  return
}
