/* This file is part of the MlirSynth Project. */

#include "transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;

namespace mlir {

#define GEN_PASS_DEF_RETURNLASTSTOREDMEMREFARG
#include "transforms/Passes.h.inc"

namespace {

struct ReturnLastStoredMemrefArgPass
    : public impl::ReturnLastStoredMemrefArgBase<ReturnLastStoredMemrefArgPass> {
  void runOnOperation() override;
};

}  // namespace
}  // namespace mlir

void ReturnLastStoredMemrefArgPass::runOnOperation() {
  auto op = getOperation();

  std::vector<func::FuncOp> fns;
  op->walk([&](func::FuncOp func) { fns.push_back(func); });
  func::FuncOp fn = fns[0];

  std::vector<func::ReturnOp> returnOps;
  op->walk([&](func::ReturnOp returnOp) { returnOps.push_back(returnOp); });
  func::ReturnOp returnOp = returnOps[0];

  // If the return op operands, return.
  if (returnOp.getNumOperands() > 0) {
    return;
  }

  // Replace the return value with the last stored memref argument.
  Value lastStore;
  fn->walk([&](func::FuncOp func) {
    // Identify the argument that has been last stored to by the function.
    func.walk([&](AffineStoreOp store) { lastStore = store.getMemRef(); });
  });

  returnOp.erase();

  // Add a new return op with the last stored memref argument.
  OpBuilder builder(op.getContext());
  builder.setInsertionPointAfter(fn.getFunctionBody().front().getTerminator());
  builder.create<func::ReturnOp>(fn.getLoc(), lastStore);

  // Change the return type of the function.
  auto returnType = lastStore.getType();
  fn.setFunctionType(builder.getFunctionType(
    fn.getFunctionBody().getArgumentTypes(), {returnType}));
}

std::unique_ptr<OperationPass<ModuleOp>>
createReturnLastStoredMemrefArgPass() {
  return std::make_unique<ReturnLastStoredMemrefArgPass>();
}
