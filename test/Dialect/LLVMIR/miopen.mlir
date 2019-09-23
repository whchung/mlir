// RUN: mlir-opt %s | FileCheck %s

func @miopen_op_conv2d_f32(%t0: tensor<?xf32>, %t1: tensor<?xf32>, %t2: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK: %0 = miopen.conv2d.f32 %arg0, %arg1, %arg2 : tensor<?xf32>
  %0 = miopen.conv2d.f32 %t0, %t1, %t2 : tensor<?xf32>
  return %0 : tensor<?xf32>
}

func @miopen_op_conv2d_f16(%t0: tensor<?xf16>, %t1: tensor<?xf16>, %t2: tensor<?xf16>) -> tensor<?xf16> {
  // CHECK: %0 = miopen.conv2d.f16 %arg0, %arg1, %arg2 : tensor<?xf16>
  %0 = miopen.conv2d.f16 %t0, %t1, %t2 : tensor<?xf16>
  return %0 : tensor<?xf16>
}

func @miopen_op_conv2d_bf16(%t0: tensor<?xbf16>, %t1: tensor<?xbf16>, %t2: tensor<?xbf16>) -> tensor<?xbf16> {
  // CHECK: %0 = miopen.conv2d.bf16 %arg0, %arg1, %arg2 : tensor<?xbf16>
  %0 = miopen.conv2d.bf16 %t0, %t1, %t2 : tensor<?xbf16>
  return %0 : tensor<?xbf16>
}
