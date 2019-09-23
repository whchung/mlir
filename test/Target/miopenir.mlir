// RUN: mlir-translate -mlir-to-miopenir %s | FileCheck %s

func @miopen_conv2d_f32() {
  // TBD
  //%t0 = constant dense<[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]> : tensor<16xf32>
  //%t1 = constant dense<[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]> : tensor<16xf32>
  //%t2 = constant dense<[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]> : tensor<16xf32>
  //%0 = miopen.conv2d.f32 %t0, %t1, %t2 : tensor<16xf32>

  // CHECK: ret void
  llvm.return
}
