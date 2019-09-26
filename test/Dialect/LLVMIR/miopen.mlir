// RUN: mlir-opt %s | FileCheck %s

func @miopen_op_conv2d_f32(%arg0 : !llvm<"float*">, %arg1 : !llvm<"float*">, %arg2 : !llvm<"float*">) {
  // CHECK: miopen.conv2d.f32 %{{.*}}, %{{.*}}, %{{.*}}
  miopen.conv2d.f32 %arg0, %arg1, %arg2 : !llvm<"float*">
  llvm.return
}

func @miopen_op_conv2d_f16(%arg0 : !llvm<"half*">, %arg1 : !llvm<"half*">, %arg2 : !llvm<"half*">) {
  // CHECK: miopen.conv2d.f16 %{{.*}}, %{{.*}}, %{{.*}}
  miopen.conv2d.f16 %arg0, %arg1, %arg2 : !llvm<"half*">
  llvm.return
}

func @miopen_op_conv2d_bf16(%arg0 : !llvm<"i16*">, %arg1 : !llvm<"i16*">, %arg2 : !llvm<"i16*">) {
  // CHECK: miopen.conv2d.bf16 %{{.*}}, %{{.*}}, %{{.*}}
  miopen.conv2d.bf16 %arg0, %arg1, %arg2 : !llvm<"i16*">
  llvm.return
}
