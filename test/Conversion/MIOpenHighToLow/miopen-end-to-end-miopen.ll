; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare i8* @malloc(i64)

declare void @free(i8*)

define void @miopen_conv2dex_f32({ float*, i64, [4 x i64], [4 x i64] }* %0, { float*, i64, [4 x i64], [4 x i64] }* %1, { float*, i64, [4 x i64], [4 x i64] }* %2) {
  %4 = load { float*, i64, [4 x i64], [4 x i64] }, { float*, i64, [4 x i64], [4 x i64] }* %0
  %5 = load { float*, i64, [4 x i64], [4 x i64] }, { float*, i64, [4 x i64], [4 x i64] }* %1
  %6 = load { float*, i64, [4 x i64], [4 x i64] }, { float*, i64, [4 x i64], [4 x i64] }* %2
  %7 = extractvalue { float*, i64, [4 x i64], [4 x i64] } %4, 0
  %8 = extractvalue { float*, i64, [4 x i64], [4 x i64] } %5, 0
  %9 = extractvalue { float*, i64, [4 x i64], [4 x i64] } %6, 0
  call void @gridwise_convolution_implicit_gemm_v4_nchw_kcyx_nkhw_lds_double_buffer_n_128_c_128_H_17_W_17_k_128_y_3_x_3_u_3_v_3_p_0_q_0_l_1_j_1(float* %7, float* %8, float* %9), !kernel_path !0
  ret void
}

declare void @gridwise_convolution_implicit_gemm_v4_nchw_kcyx_nkhw_lds_double_buffer_n_128_c_128_H_17_W_17_k_128_y_3_x_3_u_3_v_3_p_0_q_0_l_1_j_1(float*, float*, float*)

!0 = !{!"/root//.cache/miopen/2.1.0/c024ea5bfa85d4aaf3c526bc183708c2/gridwise_convolution_implicit_gemm_v4_nchw_kcyx_nkhw_lds_double_buffer.cpp.bc"}
