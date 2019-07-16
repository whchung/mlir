// RUN: mlir-rocm-runner %s --shared-libs=%hip_wrapper_library_dir/libhip-runtime-wrappers%shlibext | FileCheck %s

func @other_func(%arg0 : f32, %arg1 : memref<?xf32>) {
  %cst = constant 1 : index
  %cst2 = dim %arg1, 0 : memref<?xf32>
  // ROCM TODO
  //gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %cst, %grid_y = %cst, %grid_z = %cst)
  //           threads(%tx, %ty, %tz) in (%block_x = %cst2, %block_y = %cst, %block_z = %cst)
  //           args(%kernel_arg0 = %arg0, %kernel_arg1 = %arg1) : f32, memref<?xf32> {
  //  store %kernel_arg0, %kernel_arg1[%tx] : memref<?xf32>
  //  gpu.return
  //}
  return
}

// CHECK: [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00]
func @main() {
  %arg0 = alloc() : memref<5xf32>
  %20 = constant 0 : i32
  %21 = constant 5 : i32
  %22 = memref_cast %arg0 : memref<5xf32> to memref<?xf32>
  call @mhipMemHostRegister(%22, %20) : (memref<?xf32>, i32) -> ()
  call @mhipPrintFloat(%22) : (memref<?xf32>) -> ()
  %24 = constant 1.0 : f32
  call @other_func(%24, %22) : (f32, memref<?xf32>) -> ()
  call @mhipPrintFloat(%22) : (memref<?xf32>) -> ()
  return
}

func @mhipMemHostRegister(%ptr : memref<?xf32>, %flags : i32)
func @mhipPrintFloat(%ptr : memref<?xf32>)
