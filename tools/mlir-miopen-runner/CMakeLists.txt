set(LLVM_OPTIONAL_SOURCES
  mlir-miopen-runner.cpp
  )

if(MLIR_ROCM_RUNNER_ENABLED)
  if (NOT ("AMDGPU" IN_LIST LLVM_TARGETS_TO_BUILD))
    message(SEND_ERROR
      "Building the mlir miopen runner requires the AMDGPU backend")
  endif()

  # Configure MIOpen runner support. Using check_language first allows us to give
  # a custom error message.
  if(NOT DEFINED HIP_PATH)
      if(NOT DEFINED ENV{HIP_PATH})
          set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
      else()
          set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
      endif()
  endif()
  set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})

  find_package(HIP QUIET)
  if(HIP_FOUND)
      message(STATUS "Found HIP: " ${HIP_VERSION})
  else()
      message(FATAL_ERROR "Could not find HIP. Ensure that HIP is either installed in /opt/rocm/hip or the variable HIP_PATH is set to point to the right location.")
  endif()

  # We need the libhip_hcc.so library.
  find_library(HIP_RUNTIME_LIBRARY hip_hcc PATHS ${HIP_PATH} PATH_SUFFIXES lib)
  find_path(ROCM_INCLUDE_DIR hsa PATHS "${HIP_PATH}/.." PATH_SUFFIXES include)

  message(STATUS "HIP_RUNTIME_LIBRARY:" ${HIP_RUNTIME_LIBRARY})
  message(STATUS "ROCM_INCLUDE_DIR:" ${ROCM_INCLUDE_DIR})

  add_compile_options(-D__HIP_PLATFORM_HCC__)

  set(FULL_LINK_LIBS
    MLIRAffineOps
    MLIRGPU
    MLIRGPUtoROCMTransforms
    MLIRGPUtoROCDLTransforms
    MLIRLLVMIR
    MLIRMIOPENHighToLow
    MLIRStandardOps
    MLIRStandardToLLVM
    MLIRTargetLLVMIR
    MLIRTransforms
    MLIRTranslation
  )
  set(LIBS
    MLIRIR
    MLIRParser
    MLIREDSC
    MLIRAnalysis
    MLIRExecutionEngine
    MLIRJitRunner
    MLIRSupport
    LLVMCore
    LLVMSupport
    ${HIP_RUNTIME_LIBRARY}
  )
  add_llvm_executable(mlir-miopen-runner
    mlir-miopen-runner.cpp
  )
  add_dependencies(mlir-miopen-runner hip-runtime-wrappers)
  target_include_directories(mlir-miopen-runner
    PRIVATE ${ROCM_INCLUDE_DIR}
  )
  llvm_update_compile_flags(mlir-miopen-runner)
  whole_archive_link(mlir-miopen-runner ${FULL_LINK_LIBS})
  target_link_libraries(mlir-miopen-runner PRIVATE ${FULL_LINK_LIBS} ${LIBS})


endif()
