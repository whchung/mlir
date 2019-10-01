// RUN: mlir-opt -miopen-high-to-low -lower-to-llvm %s | FileCheck %s

// CHECK-LABEL: func @miopen_conv2d_f32(%{{.*}}: !llvm<"{ float*, [1 x i64] }">, %{{.*}}: !llvm<"{ float*, [1 x i64] }">, %{{.*}}: !llvm<"{ float*, [1 x i64] }">)
func @miopen_conv2d_f32(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>) {
  // CHECK: llvm.extractvalue %{{.*}}[0 : index] : !llvm<"{ float*, [1 x i64] }">
  // CHECK-NEXT: llvm.extractvalue %{{.*}}[0 : index] : !llvm<"{ float*, [1 x i64] }">
  // CHECK-NEXT: llvm.extractvalue %{{.*}}[0 : index] : !llvm<"{ float*, [1 x i64] }">
  // CHECK-NEXT: miopen.conv2d.kernel.f32 %{{.*}}, %{{.*}}, %{{.*}} : !llvm<"float*">
  miopen.conv2d.f32 %arg0, %arg1, %arg2 : memref<?xf32>
  return
}

// CHECK-LABEL: func @miopen_conv2d_f16(%{{.*}}: !llvm<"{ half*, [1 x i64] }">, %{{.*}}: !llvm<"{ half*, [1 x i64] }">, %{{.*}}: !llvm<"{ half*, [1 x i64] }">)
func @miopen_conv2d_f16(%arg0: memref<?xf16>, %arg1: memref<?xf16>, %arg2: memref<?xf16>) {
  // CHECK: llvm.extractvalue %{{.*}}[0 : index] : !llvm<"{ half*, [1 x i64] }">
  // CHECK-NEXT: llvm.extractvalue %{{.*}}[0 : index] : !llvm<"{ half*, [1 x i64] }">
  // CHECK-NEXT: llvm.extractvalue %{{.*}}[0 : index] : !llvm<"{ half*, [1 x i64] }">
  // CHECK-NEXT: miopen.conv2d.kernel.f16 %{{.*}}, %{{.*}}, %{{.*}} : !llvm<"half*">
  miopen.conv2d.f16 %arg0, %arg1, %arg2 : memref<?xf16>
  return
}

// TBD: BF16 lowering is not enabled in MLIR yet.

// CHECK-LABEL: func @foo(%{{.*}}: !llvm<"{ float*, [1 x i64] }">)
func @foo(%arg0: memref<16xf32>) {
  // CHECK: llvm.extractvalue %{{.*}}[0 : index] : !llvm<"{ float*, [1 x i64] }">
  // CHECK-NEXT: miopen.dummy.low %{{.*}} : !llvm<"float*">
  miopen.dummy.high %arg0 : memref<16xf32>
  return
}
