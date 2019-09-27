add_llvm_library(MLIRMIOPENHighToLow
  MIOpenHighToLow.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/MIOpenHighToLow
)
set(LIBS
  MLIRLLVMIR
  MLIRMIOPENIR
  MLIRTransforms
  LLVMCore
  LLVMSupport
  )

add_dependencies(MLIRMIOPENHighToLow ${LIBS})
target_link_libraries(MLIRMIOPENHighToLow ${LIBS})
