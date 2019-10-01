// RUN: mlir-opt %s --launch-func-to-hip | FileCheck %s

// CHECK: llvm.mlir.global constant @[[kernel_name:.*]]("kernel\00")

func @hsaco_getter() -> !llvm<"i8*">

func @kernel(!llvm.float, !llvm<"float*">)
    attributes { gpu.kernel, amdgpu.hsacogetter = @hsaco_getter }


func @foo() {
  %0 = "op"() : () -> (!llvm.float)
  %1 = "op"() : () -> (!llvm<"float*">)
  %cst = constant 8 : index

  // CHECK: [[module_ptr:%.*]] = llvm.alloca {{.*}} x !llvm<"i8*"> : (!llvm.i32) -> !llvm<"i8**">
  // CHECK: llvm.call @mhipModuleLoad([[module_ptr]], {{.*}}) : (!llvm<"i8**">, !llvm<"i8*">) -> !llvm.i32
  // CHECK: [[func_ptr:%.*]] = llvm.alloca {{.*}} x !llvm<"i8*"> : (!llvm.i32) -> !llvm<"i8**">
  // CHECK: llvm.call @mhipModuleGetFunction([[func_ptr]], {{.*}}, {{.*}}) : (!llvm<"i8**">, !llvm<"i8*">, !llvm<"i8*">) -> !llvm.i32
  // CHECK: llvm.call @mhipGetStreamHelper
  // CHECK: llvm.call @mhipLaunchKernel
  // CHECK: llvm.call @mhipStreamSynchronize
  "gpu.launch_func"(%cst, %cst, %cst, %cst, %cst, %cst, %0, %1) { kernel = @kernel }
      : (index, index, index, index, index, index, !llvm.float, !llvm<"float*">) -> ()

  return
}
