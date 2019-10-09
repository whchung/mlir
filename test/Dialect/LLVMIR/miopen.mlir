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


// Test for MIOpen high-level ops

func @miopen_op_conv2d_f32(%arg0 : memref<?xf32>, %arg1 : memref<?xf32>, %arg2 : memref<?xf32>) {
  // CHECK: miopen.conv2d.f32 %{{.*}}, %{{.*}}, %{{.*}}
  miopen.conv2d.f32 %arg0, %arg1, %arg2 : memref<?xf32>
  return
}

func @miopen_op_conv2d_f16(%arg0 : memref<?xf16>, %arg1 : memref<?xf16>, %arg2 : memref<?xf16>) {
  // CHECK: miopen.conv2d.f16 %{{.*}}, %{{.*}}, %{{.*}}
  miopen.conv2d.f16 %arg0, %arg1, %arg2 : memref<?xf16>
  return
}

func @miopen_op_conv2d_bf16(%arg0 : memref<?xbf16>, %arg1 : memref<?xbf16>, %arg2 : memref<?xbf16>) {
  // CHECK: miopen.conv2d.bf16 %{{.*}}, %{{.*}}, %{{.*}}
  miopen.conv2d.bf16 %arg0, %arg1, %arg2 : memref<?xbf16>
  return
}

func @miopen_op_dummy_high(%arg0 : memref<?xf32>) {
  // CHECK: miopen.dummy.high %{{.*}} : memref<?xf32>
  miopen.dummy.high %arg0 : memref<?xf32>
  return
}

// Test for MIOpen high-level ops with more attributes and data

#strided4D = (d0, d1, d2, d3)[s0, s1, s2, s3] -> (d0 * s1 + s0 + d1 * s2 + d2 * s3 + d3)

func @miopen_op_conv2dex_f32(%arg0 : memref<?x?x?x?xf32, #strided4D>, %arg1 : memref<?x?x?x?xf32, #strided4D>, %arg2 : memref<?x?x?x?xf32, #strided4D>) {
  // %m0 : input
  %m0 = memref_cast %arg0 : memref<?x?x?x?xf32, #strided4D> to memref<128x128x17x17xf32, #strided4D>
  // %m1 : filter
  %m1 = memref_cast %arg1 : memref<?x?x?x?xf32, #strided4D> to memref<128x128x3x3xf32, #strided4D>
  // %m2 : output
  %m2 = memref_cast %arg2 : memref<?x?x?x?xf32, #strided4D> to memref<128x128x?x?xf32, #strided4D>

  // CHECK: miopen.conv2dex.f32(%{{.*}}, %{{.*}}, %{{.*}}) {dilations = [1, 1], paddings = [0, 0], strides = [2, 2]} : memref<128x128x17x17xf32, #{{.*}}>, memref<128x128x3x3xf32, #{{.*}}>, memref<128x128x?x?xf32, #{{.*}}>
  // CHECK-NEXT: // MIOpenDriver conv -n 128 -c 128 -H 17 -W 17 -k 128 -y 3 -x 3 -u 2 -v 2 -p 0 -q 0 -l 1 -j 1 -F 1
  miopen.conv2dex.f32(%m0, %m1, %m2) {dilations = [1, 1], paddings = [0, 0], strides = [2, 2]} : memref<128x128x17x17xf32, #strided4D>, memref<128x128x3x3xf32, #strided4D>, memref<128x128x?x?xf32, #strided4D>
  return
}

// Test for MIOpen low-level ops with more attributes and data

func @miopen_op_conv2d_kernelex_f32(%arg0 : !llvm<"float*">, %arg1 : !llvm<"float*">, %arg2 : !llvm<"float*">) {
  // CHECK: miopen.conv2d.kernelex.f32 %{{.*}}, %{{.*}}, %{{.*}}
  miopen.conv2d.kernelex.f32 %arg0, %arg1, %arg2 { miopen_driver_command = "TBD", kernel_path = "TBD", kernel_name = "TBD" } : !llvm<"float*">, !llvm<"float*">, !llvm<"float*">
  llvm.return
}

