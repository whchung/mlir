// RUN: mlir-opt %s | FileCheck %s

// Test for MIOpen low-level ops

func @miopen_op_conv2d_kernel_f32(%arg0 : !llvm<"float*">, %arg1 : !llvm<"float*">, %arg2 : !llvm<"float*">) {
  // CHECK: miopen.conv2d.kernel.f32 %{{.*}}, %{{.*}}, %{{.*}}
  miopen.conv2d.kernel.f32 %arg0, %arg1, %arg2 : !llvm<"float*">
  llvm.return
}

func @miopen_op_conv2d_kernel_f16(%arg0 : !llvm<"half*">, %arg1 : !llvm<"half*">, %arg2 : !llvm<"half*">) {
  // CHECK: miopen.conv2d.kernel.f16 %{{.*}}, %{{.*}}, %{{.*}}
  miopen.conv2d.kernel.f16 %arg0, %arg1, %arg2 : !llvm<"half*">
  llvm.return
}

func @miopen_op_conv2d_kernel_bf16(%arg0 : !llvm<"i16*">, %arg1 : !llvm<"i16*">, %arg2 : !llvm<"i16*">) {
  // CHECK: miopen.conv2d.kernel.bf16 %{{.*}}, %{{.*}}, %{{.*}}
  miopen.conv2d.kernel.bf16 %arg0, %arg1, %arg2 : !llvm<"i16*">
  llvm.return
}

func @miopen_op_dummy_low(%arg0 : !llvm<"float*">) {
  // CHECK: miopen.dummy.low %{{.*}} : !llvm<"float*">
  miopen.dummy.low %arg0 : !llvm<"float*">
  llvm.return
}

func @miopen_op_dummy_high(%arg0 : memref<16xf32>) {
  // CHECK: miopen.dummy.high %{{.*}} : memref<16xf32>
  miopen.dummy.high %arg0 : memref<16xf32>
  llvm.return
}
