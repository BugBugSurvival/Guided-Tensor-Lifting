set(LLVM_LINK_COMPONENTS Core Support nativecodegen native)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(SYNTH_LIBS ${dialect_libs} ${conversion_libs} MLIROptLib)

add_subdirectory(analysis)
add_subdirectory(bindings)
add_subdirectory(common)
add_subdirectory(execution)
add_subdirectory(tools)
add_subdirectory(transforms)
add_subdirectory(validation)
