set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name MlirSynthAll)
add_public_tablegen_target(MLIRSynthTransformsIncGen)

add_mlir_library(transform
  AnnotateLastStoredMemrefArgPass.cc
  ChangeSizesPass.cc
  CleanupPass.cc
  CopyModifiedMemrefsPass.cc
  FoldToTensorToMemrefPairPass.cc
  LoopOutlinePass.cc
  MemrefCopyToLoopsPass.cc
  MemrefRank0ToScalarPass.cc
  PrepareTargetPass.cc
  ReturnLastStoredMemrefArgPass.cc
  TargetOutlinePass.cc
  Utils.cc

  DEPENDS
  MLIRSynthTransformsIncGen
)
