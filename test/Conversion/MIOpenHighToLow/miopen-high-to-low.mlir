// RUN: mlir-opt -miopen-high-to-low -lower-to-llvm %s | FileCheck %s

// CHECK-LABEL: func @foo(%{{.*}}: !llvm<"{ float*, [1 x i64] }">)
func @foo(%arg0: memref<16xf32>) {
  // CHECK: miopen.dummy.low %{{.*}} : !llvm<"{ float*, [1 x i64] }">
  miopen.dummy.high %arg0 : memref<16xf32>
  return
}
