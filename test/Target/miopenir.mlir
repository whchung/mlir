// RUN: mlir-translate -mlir-to-miopenir %s | FileCheck %s

// CHECK-LABEL: define i64 @memref_dim({ float*, i64, i64 } {{%.*}})
func @memref_dim(%arg0: !llvm<"{ float*, i64, i64 }">) -> !llvm.i64 {
// Expecting this to create an LLVM constant.
  %0 = llvm.mlir.constant(42 : index) : !llvm.i64
// CHECK-NEXT: %2 = extractvalue { float*, i64, i64 } %0, 1
  %1 = llvm.extractvalue %arg0[1] : !llvm<"{ float*, i64, i64 }">
// Expecting this to create an LLVM constant.
  %2 = llvm.mlir.constant(10 : index) : !llvm.i64
// CHECK-NEXT: %3 = extractvalue { float*, i64, i64 } %0, 2
  %3 = llvm.extractvalue %arg0[2] : !llvm<"{ float*, i64, i64 }">
// Checking that the constant for d0 has been created.
// CHECK-NEXT: %4 = add i64 42, %2
  %4 = llvm.add %0, %1 : !llvm.i64
// Checking that the constant for d2 has been created.
// CHECK-NEXT: %5 = add i64 10, %3
  %5 = llvm.add %2, %3 : !llvm.i64
// CHECK-NEXT: %6 = add i64 %4, %5
  %6 = llvm.add %4, %5 : !llvm.i64
// CHECK-NEXT: ret i64 %6
  llvm.return %6 : !llvm.i64
}


func @miopen_conv2d_f32() {
  %0 = llvm.mlir.constant(dense<1.0> : tensor<16xf32>) : !llvm<"[16 x float]">

  //%0 = miopen.conv2d.f32 %t0, %t1, %t2 : tensor<16xf32>

  // XCHECK: ret void
  llvm.return %0 : !llvm<"[16 x float]">
}
