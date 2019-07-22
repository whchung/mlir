// RUN: mlir-opt %s | FileCheck %s

func @rocdl_special_regs() -> !llvm.i32 {
  // CHECK: %0 = rocdl.workitem.id.x : !llvm.i32
  %0 = rocdl.workitem.id.x : !llvm.i32
  // CHECK: %1 = rocdl.workitem.id.y : !llvm.i32
  %1 = rocdl.workitem.id.y : !llvm.i32
  // CHECK: %2 = rocdl.workitem.id.z : !llvm.i32
  %2 = rocdl.workitem.id.z : !llvm.i32
  // CHECK: %3 = rocdl.workgroup.id.x : !llvm.i32
  %3 = rocdl.workgroup.id.x : !llvm.i32
  // CHECK: %4 = rocdl.workgroup.id.y : !llvm.i32
  %4 = rocdl.workgroup.id.y : !llvm.i32
  // CHECK: %5 = rocdl.workgroup.id.z : !llvm.i32
  %5 = rocdl.workgroup.id.z : !llvm.i32
  // XXXCHECK: %6 = __ockl_get_local_size.x : !llvm.i32
  //%6 = __ockl_get_local_size.x : !llvm.i32
  // XXXCHECK: %7 = __ockl_get_local_size.y : !llvm.i32
  //%7 = __ockl_get_local_size.y : !llvm.i32
  // XXXCHECK: %8 = __ockl_get_local_size.z : !llvm.i32
  //%8 = __ockl_get_local_size.z : !llvm.i32
  // XXXCHECK: %9 = __ockl_get_global_size.x : !llvm.i32
  //%9 = __ockl_get_global_size.x : !llvm.i32
  // XXXCHECK: %10 = __ockl_get_global_size.y : !llvm.i32
  //%10 = __ockl_get_global_size.y : !llvm.i32
  // XXXCHECK: %11 = __ockl_get_global_size.z : !llvm.i32
  //%11 = __ockl_get_global_size.z : !llvm.i32
  llvm.return %0 : !llvm.i32
}
