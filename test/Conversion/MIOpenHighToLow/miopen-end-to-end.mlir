// RUN: mlir-opt -miopen-high-to-low -lower-to-llvm %s | FileCheck %s

module attributes  { gpu.kernel_module } {
  // CHECK-LABEL: func @miopen_conv2dex_f32
  func @miopen_conv2dex_f32(%arg0: memref<128x128x17x17xf32>, %arg1: memref<128x128x3x3xf32>, %arg2: memref<128x128x?x?xf32>) attributes  { gpu.kernel } {
    miopen.conv2dex.f32(%arg0, %arg1, %arg2) {dilations=[1,1], paddings=[0,0], strides=[3,3]}: memref<128x128x17x17xf32>, memref<128x128x3x3xf32>, memref<128x128x?x?xf32>

    miopen.conv2dex.f32(%arg0, %arg1, %arg2) {dilations=[1,1], paddings=[0,0], strides=[3,3]}: memref<128x128x17x17xf32>, memref<128x128x3x3xf32>, memref<128x128x?x?xf32>
    return
  }

}
