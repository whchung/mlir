// RUN: mlir-opt -miopen-high-to-low -lower-to-llvm %s | FileCheck %s

// CHECK-LABEL: func @miopen_conv2d_f32(%{{.*}}: !llvm<"{ float*, i64, [1 x i64], [1 x i64] }*">, %{{.*}}: !llvm<"{ float*, i64, [1 x i64], [1 x i64] }*">, %{{.*}}: !llvm<"{ float*, i64, [1 x i64], [1 x i64] }*">)
func @miopen_conv2d_f32(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>) {
  // CHECK: llvm.extractvalue %{{.*}}[0 : index] : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">
  // CHECK-NEXT: llvm.extractvalue %{{.*}}[0 : index] : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">
  // CHECK-NEXT: llvm.extractvalue %{{.*}}[0 : index] : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">
  // CHECK-NEXT: miopen.conv2d.kernel.f32 %{{.*}}, %{{.*}}, %{{.*}} : !llvm<"float*">
  miopen.conv2d.f32 %arg0, %arg1, %arg2 : memref<?xf32>
  return
}

// CHECK-LABEL: func @miopen_conv2d_f16(%{{.*}}: !llvm<"{ half*, i64, [1 x i64], [1 x i64] }*">, %{{.*}}: !llvm<"{ half*, i64, [1 x i64], [1 x i64] }*">, %{{.*}}: !llvm<"{ half*, i64, [1 x i64], [1 x i64] }*">)
func @miopen_conv2d_f16(%arg0: memref<?xf16>, %arg1: memref<?xf16>, %arg2: memref<?xf16>) {
  // CHECK: llvm.extractvalue %{{.*}}[0 : index] : !llvm<"{ half*, i64, [1 x i64], [1 x i64] }">
  // CHECK-NEXT: llvm.extractvalue %{{.*}}[0 : index] : !llvm<"{ half*, i64, [1 x i64], [1 x i64] }">
  // CHECK-NEXT: llvm.extractvalue %{{.*}}[0 : index] : !llvm<"{ half*, i64, [1 x i64], [1 x i64] }">
  // CHECK-NEXT: miopen.conv2d.kernel.f16 %{{.*}}, %{{.*}}, %{{.*}} : !llvm<"half*">
  miopen.conv2d.f16 %arg0, %arg1, %arg2 : memref<?xf16>
  return
}

// TBD: BF16 lowering is not enabled in MLIR yet.

// CHECK-LABEL: func @foo(%{{.*}}: !llvm<"{ float*, i64, [1 x i64], [1 x i64] }*">)
func @foo(%arg0: memref<16xf32>) {
  // CHECK: llvm.extractvalue %{{.*}}[0 : index] : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">
  // CHECK-NEXT: miopen.dummy.low %{{.*}} : !llvm<"float*">
  miopen.dummy.high %arg0 : memref<16xf32>
  return
}

func @miopen_conv2dex_f32(%arg0: memref<128x128x17x17xf32>, %arg1: memref<128x128x3x3xf32>, %arg2: memref<128x128x?x?xf32>) {
  miopen.conv2dex.f32(%arg0, %arg1, %arg2) {dilations=[1,1], paddings=[0,0], strides=[3,3]}: memref<128x128x17x17xf32>, memref<128x128x3x3xf32>, memref<128x128x?x?xf32>
  return
}

