#!/bin/bash

set -x
set -e



# Preparations
# ####
# Parse build type
BUILD_DIR=build
BUILD_TYPE=
if [ "$1" == "--debug" ]; then
  BUILD_DIR=build_debug
  BUILD_TYPE=Debug
fi

# Build mlirSynth
# ####

# Configure mlirSynth build.
mkdir -p $BUILD_DIR
pushd $BUILD_DIR
FLAGS=".. \
  -GNinja \
  -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
  -DMLIR_DIR=${PWD}/../deps/llvm-project/${BUILD_DIR}/lib/cmake/mlir \
  -DMHLO_DIR=${PWD}/../deps/mlir-hlo/${BUILD_DIR}/cmake/modules/CMakeFiles \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON "
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
  FLAGS="$FLAGS -DLLVM_ENABLE_LLD=ON"
fi
cmake $FLAGS

popd

# Build mlirSynth.
pushd $BUILD_DIR
cmake --build .
popd

# Merge all compile_commands.json files, so that clangd can find them.
jq -s 'map(.[])' deps/llvm-project/build/compile_commands.json \
  deps/mlir-hlo/build/compile_commands.json \
  build/compile_commands.json \
  > compile_commands.json
