

module attributes {gpu.kernel_module} {
  func @miopen_conv2dex_f32(%arg0: !llvm<"{ float*, i64, [4 x i64], [4 x i64] }*">, %arg1: !llvm<"{ float*, i64, [4 x i64], [4 x i64] }*">, %arg2: !llvm<"{ float*, i64, [4 x i64], [4 x i64] }*">)
  attributes  {gpu.kernel} {
    %0 = llvm.load %arg0 : !llvm<"{ float*, i64, [4 x i64], [4 x i64] }*">
    %1 = llvm.load %arg1 : !llvm<"{ float*, i64, [4 x i64], [4 x i64] }*">
    %2 = llvm.load %arg2 : !llvm<"{ float*, i64, [4 x i64], [4 x i64] }*">
    %3 = llvm.extractvalue %0[0 : index] : !llvm<"{ float*, i64, [4 x i64], [4 x i64] }">
    %4 = llvm.extractvalue %1[0 : index] : !llvm<"{ float*, i64, [4 x i64], [4 x i64] }">
    %5 = llvm.extractvalue %2[0 : index] : !llvm<"{ float*, i64, [4 x i64], [4 x i64] }">
    miopen.conv2d.kernelex.f32 %3, %4, %5  {kernel_name = "gridwise_convolution_implicit_gemm_v4_nchw_kcyx_nkhw_lds_double_buffer_n_128_c_128_H_17_W_17_k_128_y_3_x_3_u_3_v_3_p_0_q_0_l_1_j_1", kernel_path = "/root//.cache/miopen/2.1.0/c024ea5bfa85d4aaf3c526bc183708c2/gridwise_convolution_implicit_gemm_v4_nchw_kcyx_nkhw_lds_double_buffer.cpp.bc", miopen_driver_command = "conv -n 128 -c 128 -H 17 -W 17 -k 128 -y 3 -x 3 -u 3 -v 3 -p 0 -q 0 -l 1 -j 1 -F 1 -V 0 -O 1 "} : !llvm<"float*">, !llvm<"float*">, !llvm<"float*">
    llvm.return
  }
}
