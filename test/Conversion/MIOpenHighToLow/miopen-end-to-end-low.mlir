

module attributes {gpu.kernel_module} {
  func @miopen_conv2dex_f32(%arg0: !llvm<"{ float*, i64, [4 x i64], [4 x i64] }*">, %arg1: !llvm<"{ float*, i64, [4 x i64], [4 x i64] }*">, %arg2: !llvm<"{ float*, i64, [4 x i64], [4 x i64] }*">)
  attributes  {gpu.kernel} {
    %0 = llvm.load %arg0 : !llvm<"{ float*, i64, [4 x i64], [4 x i64] }*">
    %1 = llvm.load %arg1 : !llvm<"{ float*, i64, [4 x i64], [4 x i64] }*">
    %2 = llvm.load %arg2 : !llvm<"{ float*, i64, [4 x i64], [4 x i64] }*">
    %3 = llvm.extractvalue %0[0 : index] : !llvm<"{ float*, i64, [4 x i64], [4 x i64] }">
    %4 = llvm.extractvalue %1[0 : index] : !llvm<"{ float*, i64, [4 x i64], [4 x i64] }">
    %5 = llvm.extractvalue %2[0 : index] : !llvm<"{ float*, i64, [4 x i64], [4 x i64] }">
    miopen.conv2d.kernelex.f32 %3, %4, %5  {kernel_name = "some_name", kernel_path = "some_where", miopen_driver_command = "conv -n 128 -c 128 -H 17 -W 17 -k 128 -y 3 -x 3 -u 3 -v 3 -p 0 -q 0 -l 1 -j 1 -F 1 -V 0 -O 1 "} : !llvm<"float*">, !llvm<"float*">, !llvm<"float*">
    llvm.return
  }
}
