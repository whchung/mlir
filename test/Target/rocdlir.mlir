// RUN: mlir-translate -mlir-to-rocdlir %s | FileCheck %s

func @rocdl_special_regs() -> !llvm.i32 {
  // CHECK: %1 = call i32 @llvm.amdgcn.workitem.id.x()
  %1 = rocdl.workitem.id.x : !llvm.i32
  // CHECK: %2 = call i32 @llvm.amdgcn.workitem.id.y()
  %2 = rocdl.workitem.id.y : !llvm.i32
  // CHECK: %3 = call i32 @llvm.amdgcn.workitem.id.z()
  %3 = rocdl.workitem.id.z : !llvm.i32
  // CHECK: %4 = call i32 @llvm.amdgcn.workgroup.id.x()
  %4 = rocdl.workgroup.id.x : !llvm.i32
  // CHECK: %5 = call i32 @llvm.amdgcn.workgroup.id.y()
  %5 = rocdl.workgroup.id.y : !llvm.i32
  // CHECK: %6 = call i32 @llvm.amdgcn.workgroup.id.z()
  %6 = rocdl.workgroup.id.z : !llvm.i32
  // XXXCHECK: %7 = call i32 @__ockl_get_local_size(i32 0)
  //%7 = __ockl_get_local_size.x : !llvm.i32
  // XXXCHECK: %8 = call i32 @__ockl_get_local_size(i32 1)
  //%8 = __ockl_get_local_size.y : !llvm.i32
  // XXXCHECK: %9 = call i32 @__ockl_get_local_size(i32 2)
  //%9 = __ockl_get_local_size.z : !llvm.i32
  // XXXCHECK: %10 = call i32 @__ockl_get_global_size(i32 0)
  //%10 = __ockl_get_global_size.x : !llvm.i32
  // XXXCHECK: %11 = call i32 @__ockl_get_global_size(i32 1)
  //%11 = __ockl_get_global_size.y : !llvm.i32
  // XXXCHECK: %12 = call i32 @__ockl_get_global_size(i32 2)
  //%12 = __ockl_get_global_size.z : !llvm.i32
  llvm.return %1 : !llvm.i32
}

// This function has the "amdgpu_kernel" calling convention after conversion.
// CHECK:     amdgpu_kernel
func @kernel_func() attributes {gpu.kernel} {
  llvm.return
}
