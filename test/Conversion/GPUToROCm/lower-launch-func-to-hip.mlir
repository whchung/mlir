// RUN: mlir-opt %s --launch-func-to-hip | FileCheck %s

func @hsaco_getter() -> !llvm<"i8*">

func @kernel(!llvm.float, !llvm<"float*">)
    attributes { gpu.kernel, amdgpu.hsacogetter = @hsaco_getter }


func @foo() {
  %0 = "op"() : () -> (!llvm.float)
  %1 = "op"() : () -> (!llvm<"float*">)
  %cst = constant 8 : index

  // CHECK: %5 = llvm.alloca %4 x !llvm<"i8*"> : (!llvm.i32) -> !llvm<"i8**">
  // CHECK: %6 = llvm.call @mhipModuleLoad(%5, %3) : (!llvm<"i8**">, !llvm<"i8*">) -> !llvm.i32
  // CHECK: %32 = llvm.alloca %31 x !llvm<"i8*"> : (!llvm.i32) -> !llvm<"i8**">
  // CHECK: %33 = llvm.call @mhipModuleGetFunction(%32, %7, %9) : (!llvm<"i8**">, !llvm<"i8*">, !llvm<"i8*">) -> !llvm.i32
  // CHECK: %34 = llvm.call @mhipGetStreamHelper() : () -> !llvm<"i8*">
  // CHECK: %48 = llvm.call @mhipLaunchKernel(%35, %c8, %c8, %c8, %c8, %c8, %c8, %2, %34, %38, %47) : (!llvm<"i8*">, index, index, index, index, index, index, !llvm.i32, !llvm<"i8*">, !llvm<"i8**">, !llvm<"i8**">) -> !llvm.i32
  // CHECK: %49 = llvm.call @mhipStreamSynchronize(%34) : (!llvm<"i8*">) -> !llvm.i32
  "gpu.launch_func"(%cst, %cst, %cst, %cst, %cst, %cst, %0, %1) { kernel = @kernel }
      : (index, index, index, index, index, index, !llvm.float, !llvm<"float*">) -> ()

  return
}
