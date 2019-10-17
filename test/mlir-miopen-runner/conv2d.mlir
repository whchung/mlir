// RUN: mlir-miopen-runner %s --shared-libs=%hip_wrapper_library_dir/libhip-runtime-wrappers%shlibext --entry-point-result=void

func @conv2d(%arg_input : memref<?x?x?x?xf32>, %arg_filter : memref<?x?x?x?xf32>, %arg_output : memref<?x?x?x?xf32>) {
  %cst = constant 1 : index
  %cst2 = constant 128 : index

  %value = constant 1.0 : f32

  %tensor_input = memref_cast %arg_input : memref<?x?x?x?xf32> to memref<128x128x17x17xf32>
  %tensor_filter = memref_cast %arg_filter : memref<?x?x?x?xf32> to memref<128x128x3x3xf32>
  %tensor_output = memref_cast %arg_output : memref<?x?x?x?xf32> to memref<128x128x42x42xf32>

  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %cst, %grid_y = %cst, %grid_z = %cst)
             threads(%tx, %ty, %tz) in (%block_x = %cst2, %block_y = %cst, %block_z = %cst)
             args(%kernarg_input = %tensor_input, %kernarg_filter = %tensor_filter, %kernarg_output = %tensor_output, %kernarg_value = %value) : memref<128x128x17x17xf32>, memref<128x128x3x3xf32>, memref<128x128x42x42xf32>, f32 {
    miopen.conv2dex.f32(%kernarg_input, %kernarg_filter, %kernarg_output) {dilations=[1,1], paddings=[0,0], strides=[3,3]}: memref<128x128x17x17xf32>, memref<128x128x3x3xf32>, memref<128x128x42x42xf32>
    gpu.return
  }
  return
}


func @main() {
  %zero = constant 0 : i32

  %n = constant 128 : i32
  %c = constant 128 : i32
  %H = constant 17  : i32
  %W = constant 17  : i32
  %k = constant 128 : i32
  %y = constant 3   : i32
  %x = constant 3   : i32
  %oH = constant 42 : i32
  %oW = constant 42 : i32

  // input tensor
  %t0_d = call @mhipMalloc4D(%n, %c, %H, %W) : (i32, i32, i32, i32) -> (memref<?x?x?x?xf32>)

  // filter tensor
  %t1_d = call @mhipMalloc4D(%k, %c, %y, %x) : (i32, i32, i32, i32) -> (memref<?x?x?x?xf32>)

  // output tensor
  %t2_d = call @mhipMalloc4D(%n, %k, %oH, %oW) : (i32, i32, i32, i32) -> (memref<?x?x?x?xf32>)

  call @conv2d(%t0_d, %t1_d, %t2_d) : (memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, memref<?x?x?x?xf32>) -> () 

  return
}

func @mhipMalloc4D(%n : i32, %c : i32, %h : i32, %w : i32) -> (memref<?x?x?x?xf32>)
