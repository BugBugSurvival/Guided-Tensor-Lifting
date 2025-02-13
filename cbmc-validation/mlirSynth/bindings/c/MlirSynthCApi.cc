/* This file is part of the MlirSynth Project. */

#include "bindings/c/MlirSynthCApi.h"

#include "execution/Executor.h"
#include "execution/Lowering.h"
#include "transforms/Utils.h"
#include "validation/CheckingValidator.h"

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Utils.h"

#include "mlir-hlo/Dialect/mhlo/IR/register.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/Register.h"
#include <stablehlo/dialect/StablehloOps.h>

#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"

#include <cstddef>

using namespace mlir;

void registerDialects(MlirContext context) {
  auto ctx = unwrap(context);

  DialectRegistry registry;
  registerAllDialects(registry);
  registerLLVMDialectTranslation(registry);
  mlir::mhlo::registerAllMhloDialects(registry);
  mlir::stablehlo::registerAllDialects(registry);
  ctx->appendDialectRegistry(registry);
  ctx->loadAllAvailableDialects();
}

void lowerCHLOToAffine(MlirModule module, bool expandArithOps) {
  auto mod = unwrap(module);

  auto pm = std::make_shared<PassManager>(mod->getContext());
  HLO::addCHLOToAffinePasses(pm, expandArithOps);
  pm->run(mod);
}

bool checkValidate(MlirModule lhsModule, MlirModule rhsModule,
                   bool printArgsAndResults, bool printResults) {
  auto lhs = unwrap(lhsModule);
  auto rhs = unwrap(rhsModule);

  return checkValidate(lhs, rhs, printArgsAndResults, printResults);
}