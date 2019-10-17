# Getting started with MLIR on ROCm

```sh
git clone https://github.com/llvm/llvm-project.git
git clone -b exp-miopen-dialect-take2 https://github.com/whchung/mlir llvm-project/llvm/projects/mlir
mkdir llvm-project/build
cd llvm-project/build
cmake -G Ninja ../llvm -DLLVM_BUILD_EXAMPLES=ON -DLLVM_TARGETS_TO_BUILD="host;AMDGPU" -DMLIR_ROCM_RUNNER_ENABLED=1
cmake --build . --target check-mlir
```
