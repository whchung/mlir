// RUN: mlir-opt %s --generate-hsaco-accessors | FileCheck %s

func @kernel(!llvm.float, !llvm<"float*">)
// CHECK: attributes  {amdgpu.hsaco = "HSACO", amdgpu.hsacogetter = @kernel_hsaco, gpu.kernel}
  attributes  {gpu.kernel, amdgpu.hsaco = "HSACO"}

// CHECK: func @malloc(!llvm.i64) -> !llvm<"i8*">
// CHECK: func @kernel_hsaco() -> !llvm<"i8*">
// CHECK-NEXT: %0 = llvm.constant(5 : index) : !llvm.i64
// CHECK-NEXT: %1 = llvm.call @malloc(%0) : (!llvm.i64) -> !llvm<"i8*">
// CHECK-NEXT: %2 = llvm.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %3 = llvm.getelementptr %1[%2] : (!llvm<"i8*">, !llvm.i32) -> !llvm<"i8*">
// CHECK-NEXT: %4 = llvm.constant(72 : i8) : !llvm.i8
// CHECK-NEXT: llvm.store %4, %3 : !llvm<"i8*">
// CHECK-NEXT: %5 = llvm.constant(1 : i32) : !llvm.i32
// CHECK-NEXT: %6 = llvm.getelementptr %1[%5] : (!llvm<"i8*">, !llvm.i32) -> !llvm<"i8*">
// CHECK-NEXT: %7 = llvm.constant(83 : i8) : !llvm.i8
// CHECK-NEXT: llvm.store %7, %6 : !llvm<"i8*">
// CHECK-NEXT: %8 = llvm.constant(2 : i32) : !llvm.i32
// CHECK-NEXT: %9 = llvm.getelementptr %1[%8] : (!llvm<"i8*">, !llvm.i32) -> !llvm<"i8*">
// CHECK-NEXT: %10 = llvm.constant(65 : i8) : !llvm.i8
// CHECK-NEXT: llvm.store %10, %9 : !llvm<"i8*">
// CHECK-NEXT: %11 = llvm.constant(3 : i32) : !llvm.i32
// CHECK-NEXT: %12 = llvm.getelementptr %1[%11] : (!llvm<"i8*">, !llvm.i32) -> !llvm<"i8*">
// CHECK-NEXT: %13 = llvm.constant(67 : i8) : !llvm.i8
// CHECK-NEXT: llvm.store %13, %12 : !llvm<"i8*">
// CHECK-NEXT: %14 = llvm.constant(4 : i32) : !llvm.i32
// CHECK-NEXT: %15 = llvm.getelementptr %1[%14] : (!llvm<"i8*">, !llvm.i32) -> !llvm<"i8*">
// CHECK-NEXT: %16 = llvm.constant(79 : i8) : !llvm.i8
// CHECK-NEXT: llvm.store %16, %15 : !llvm<"i8*">
// CHECK-NEXT: llvm.return %1 : !llvm<"i8*">
