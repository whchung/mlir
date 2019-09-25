// RUN: mlir-opt %s | FileCheck %s

func @miopen_op_conv2d_f32(%t0: tensor<?xf32>, %t1: tensor<?xf32>, %t2: tensor<?xf32>) -> tensor<?xf32> {
  %n = dim %t0, 0 : tensor<?xf32>

  %t0_m = alloc(%n) : memref<?xf32>
  tensor_store %t0, %t0_m : memref<?xf32>
  %t1_m = alloc(%n) : memref<?xf32>
  tensor_store %t1, %t1_m : memref<?xf32>
  %t2_m = alloc(%n) : memref<?xf32>
  tensor_store %t2, %t2_m : memref<?xf32>

  // CHECK: miopen.conv2d.f32 %{{.*}}, %{{.*}}, %{{.*}} : memref<?xf32>
  %result_m = miopen.conv2d.f32 %t0_m, %t1_m, %t2_m : memref<?xf32>

  dealloc %t0_m : memref<?xf32>
  dealloc %t1_m : memref<?xf32>
  dealloc %t2_m : memref<?xf32>

  %result = tensor_load %result_m : memref<?xf32>

  return %result : tensor<?xf32>
}

func @miopen_op_conv2d_f16(%t0: tensor<?xf16>, %t1: tensor<?xf16>, %t2: tensor<?xf16>) -> tensor<?xf16> {
  %n = dim %t0, 0 : tensor<?xf16>

  %t0_m = alloc(%n) : memref<?xf16>
  tensor_store %t0, %t0_m : memref<?xf16>
  %t1_m = alloc(%n) : memref<?xf16>
  tensor_store %t1, %t1_m : memref<?xf16>
  %t2_m = alloc(%n) : memref<?xf16>
  tensor_store %t2, %t2_m : memref<?xf16>

  // CHECK: miopen.conv2d.f16 %{{.*}}, %{{.*}}, %{{.*}} : memref<?xf16>
  %result_m = miopen.conv2d.f16 %t0_m, %t1_m, %t2_m : memref<?xf16>

  dealloc %t1_m : memref<?xf16>
  dealloc %t1_m : memref<?xf16>
  dealloc %t2_m : memref<?xf16>

  %result = tensor_load %result_m : memref<?xf16>

  return %result : tensor<?xf16>
}

func @miopen_op_conv2d_bf16(%t0: tensor<?xbf16>, %t1: tensor<?xbf16>, %t2: tensor<?xbf16>) -> tensor<?xbf16> {
  %n = dim %t0, 0 : tensor<?xbf16>

  %t0_m = alloc(%n) : memref<?xbf16>
  tensor_store %t0, %t0_m : memref<?xbf16>
  %t1_m = alloc(%n) : memref<?xbf16>
  tensor_store %t1, %t1_m : memref<?xbf16>
  %t2_m = alloc(%n) : memref<?xbf16>
  tensor_store %t2, %t2_m : memref<?xbf16>

  // CHECK: miopen.conv2d.bf16 %{{.*}}, %{{.*}}, %{{.*}} : memref<?xbf16>
  %result_m = miopen.conv2d.bf16 %t0_m, %t1_m, %t2_m : memref<?xbf16>

  dealloc %t1_m : memref<?xbf16>
  dealloc %t1_m : memref<?xbf16>
  dealloc %t2_m : memref<?xbf16>

  %result = tensor_load %result_m : memref<?xbf16>

  return %result : tensor<?xbf16>
}
