set(MLIR_COMMON_LIBS
          ${dialect_libs}
          ${conversion_libs}
          ${test_libs}
          MLIRAffineAnalysis
          MLIRAnalysis
          MLIRDialect
          MLIROptLib
          MLIRParser
          MLIRPass
          MLIRTransforms
          MLIRTransformUtils
          MLIRSupport
          MLIRIR
          ${CMAKE_SOURCE_DIR}/deps/isl-0.25/.libs/libisl.a
          ${GMP_LIB})

# Lower CHLO to Affine tool
add_llvm_executable(lower-chlo-to-affine
  LowerCHLOToAffine.cc)
target_include_directories(lower-chlo-to-affine
  PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(lower-chlo-to-affine
  PRIVATE execution
          transform
          ${MLIR_COMMON_LIBS})
llvm_update_compile_flags(lower-chlo-to-affine)


# Transform Tool
add_llvm_executable(opt
  Opt.cc)
target_include_directories(opt
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}
          ${MHLO_INCLUDE_DIRS})
target_link_libraries(opt
  PRIVATE ${MHLO_EXPORTED_TARGETS}
          analysis
          transform
          ${MLIR_COMMON_LIBS})

# Test Executor
add_llvm_executable(test-executor
  TestExecutor.cc)
target_include_directories(test-executor
  PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(test-executor
  PRIVATE common
          execution
          transform
          validation
          ${MLIR_COMMON_LIBS})
llvm_update_compile_flags(test-executor)
