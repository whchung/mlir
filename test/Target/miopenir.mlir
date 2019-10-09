// RUN: mlir-translate -mlir-to-miopenir %s | FileCheck %s

// CHECK-LABEL: define void @miopen_op_dummy_low({ float*, [1 x i64] } %{{.*}}) {
func @miopen_op_dummy_low(%arg0 : !llvm<"{ float*, [1 x i64] }">) {
  %ptr = llvm.extractvalue %arg0[0] : !llvm<"{ float*, [1 x i64] }">

  // CHECK: call void @miopen_dummy_low(float* %{{.*}})
  miopen.dummy.low %ptr : !llvm<"float*">
  llvm.return
}

// CHECK-LABEL: define void @miopen_op_conv2d_kernel_f32(float* %{{.*}}, float* %{{.*}}, float* %{{.*}}) {
func @miopen_op_conv2d_kernel_f32(%arg0 : !llvm<"float*">, %arg1 : !llvm<"float*">, %arg2 : !llvm<"float*">) {

  // CHECK: call void @miopen_conv2d_f32_kernel(float* {{.*}}, float* {{.*}}, float* {{.*}})
  miopen.conv2d.kernel.f32 %arg0, %arg1, %arg2 : !llvm<"float*">
  llvm.return
}

// CHECK-LABEL: define void @miopen_op_conv2d_kernel_f16(half* %{{.*}}, half* %{{.*}}, half* %{{.*}}) {
func @miopen_op_conv2d_kernel_f16(%arg0 : !llvm<"half*">, %arg1 : !llvm<"half*">, %arg2 : !llvm<"half*">) {

  // CHECK: call void @miopen_conv2d_f16_kernel(half* {{.*}}, half* {{.*}}, half* {{.*}})
  miopen.conv2d.kernel.f16 %arg0, %arg1, %arg2 : !llvm<"half*">
  llvm.return
}

// CHECK-LABEL: define void @miopen_op_conv2d_kernel_bf16(i16* %{{.*}}, i16* %{{.*}}, i16* %{{.*}}) {
func @miopen_op_conv2d_kernel_bf16(%arg0 : !llvm<"i16*">, %arg1 : !llvm<"i16*">, %arg2 : !llvm<"i16*">) {

  // CHECK: call void @miopen_conv2d_bf16_kernel(i16* {{.*}}, i16* {{.*}}, i16* {{.*}})
  miopen.conv2d.kernel.bf16 %arg0, %arg1, %arg2 : !llvm<"i16*">
  llvm.return
}

// CHECK-LABEL: define void @miopen_op_conv2d_kernelex_f32(float* %{{.*}}, float* %{{.*}}, float* %{{.*}}) {
func @miopen_op_conv2d_kernelex_f32(%arg0 : !llvm<"float*">, %arg1 : !llvm<"float*">, %arg2 : !llvm<"float*">) {

  // CHECK: call void @conv2d.kernelex.f32(float* {{.*}}, float* {{.*}}, float* {{.*}})
  miopen.conv2d.kernelex.f32 %arg0, %arg1, %arg2 { miopen_driver_command = "TBD", kernel_path = "TBD", kernel_name = "TBD" } : !llvm<"float*">, !llvm<"float*">, !llvm<"float*">
  llvm.return
}

