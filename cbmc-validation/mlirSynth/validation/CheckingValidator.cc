/* This file is part of the MlirSynth Project. */

#include "CheckingValidator.h"

#include "mlir/IR/MLIRContext.h"
#include "transforms/Passes.h"
#include "transforms/Utils.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "mlir/Target/Cpp/CppEmitter.h"
#include "llvm/ADT/SmallVector.h"

#include <fstream>
#include <regex>

using namespace mlir;


void buildCheck(MLIRContext *ctx, OpBuilder &builder, func::CallOp &lhsCallOp, 
         func::CallOp &rhsCallOp, SmallVector<mlir::Value> &indices) {
        // Load from memrefs.
      auto lhsMemref = lhsCallOp.getResult(0);
      mlir::Type lhsType =
          lhsMemref.getType().cast<MemRefType>().getElementType();
      auto lhsLoad = builder.create<memref::LoadOp>(
          UnknownLoc::get(ctx), lhsType, lhsMemref, indices);

      auto rhsMemref = rhsCallOp.getResult(0);
      mlir::Type rhsType =
          rhsMemref.getType().cast<MemRefType>().getElementType();
      auto rhsLoad = builder.create<memref::LoadOp>(
          UnknownLoc::get(ctx), rhsType, rhsMemref, indices);

      // Create check.
      // Check for equality of lhsLoad and rhsLoad.
      auto cond1 = builder.create<arith::CmpFOp>(
          UnknownLoc::get(ctx), arith::CmpFPredicate::ONE, lhsLoad, rhsLoad);

      // Check for nan by calling function isnanf on lhsLoad.
      /*SmallVector<mlir::Value> cond2Operands = {lhsLoad};
      auto cond2 = builder.create<func::CallOp>(
          UnknownLoc::get(ctx), builder.getI1Type(), "!isnanf", cond2Operands);

      // Check for nan by calling function isnanf on rhsLoad.
      SmallVector<mlir::Value> cond3Operands = {rhsLoad};
      auto cond3 = builder.create<func::CallOp>(
          UnknownLoc::get(ctx), builder.getI1Type(), "!isnanf", cond3Operands);

      // Build predicate
      auto pred1 = builder.create<arith::AndIOp>(
          UnknownLoc::get(ctx), cond1.getResult(), cond2->getResult(0));
      auto pred2 = builder.create<arith::AndIOp>(
          UnknownLoc::get(ctx), pred1.getResult(), cond3->getResult(0));*/

      // Create if Operation.
      auto ifOp = builder.create<scf::IfOp>(UnknownLoc::get(ctx), cond1,
                                            /*withElseRegion=*/false);
      builder.setInsertionPointToStart(&ifOp.getThenRegion().front());

      SmallVector<mlir::Value> checkOperands = {};
      builder.create<func::CallOp>(UnknownLoc::get(ctx), builder.getF64Type(),
                                   "cbmc_assert", checkOperands);
}

void buildArgInit(MLIRContext *ctx, OpBuilder &builder, memref::AllocOp &memref,
                  SmallVector<mlir::Value> &indices) {
  // Create decl.
  /*SmallVector<mlir::Value> operands = {};
  auto declOp = builder.create<func::CallOp>(
      UnknownLoc::get(ctx), builder.getF64Type(),
      "__VERIFIER_nondet_float", operands);
*/
  SmallVector<mlir::Value> operands = {};
  auto declOp = builder.create<func::CallOp>(
      UnknownLoc::get(ctx), builder.getF64Type(),
      "__VERIFIER_nondet_rational", operands);

  // Create store.
  builder.create<memref::StoreOp>(UnknownLoc::get(ctx),
                                          declOp->getResult(0),
                                          memref.getMemref(), indices);
}

void copyArgInit(MLIRContext *ctx, OpBuilder &builder, memref::AllocOp &memref,
                  SmallVector<mlir::Value> &indices, mlir::Value original) {
  
  SmallVector<mlir::Value> operands = {};
  mlir::Type lhsType = original.getType().cast<MemRefType>().getElementType();
  auto declOp = builder.create<memref::LoadOp>(UnknownLoc::get(ctx), lhsType, 
                                               original, indices);

  // Create store.
  builder.create<memref::StoreOp>(UnknownLoc::get(ctx),
                                          declOp->getResult(0),
                                          memref.getMemref(), indices);
  }

OwningOpRef<ModuleOp> buildModule(func::FuncOp lhsFunction,
                                  func::FuncOp rhsFunction) {
  auto *ctx = lhsFunction->getContext();

  auto builder = OpBuilder(ctx);

  OwningOpRef<ModuleOp> module(ModuleOp::create(UnknownLoc::get(ctx)));
  auto &moduleBlock = module->getRegion().getBlocks().front();
  builder.setInsertionPoint(&moduleBlock, moduleBlock.begin());

  // Rename and copy over functions.
  auto lhsFunctionClone = lhsFunction.clone();
  lhsFunctionClone.setSymName("lhs");
  module->push_back(lhsFunctionClone);

  auto rhsFunctionClone = rhsFunction.clone();
  rhsFunctionClone.setSymName("rhs");
  module->push_back(rhsFunctionClone);

  // Create a main function.
  auto mainFunc = builder.create<func::FuncOp>(UnknownLoc::get(ctx), "main",
                                               builder.getFunctionType({}, {}));

  auto &bodyBlock = *mainFunc.addEntryBlock();
  builder.setInsertionPoint(&bodyBlock, bodyBlock.begin());

  // Create operands.
  SmallVector<mlir::Value> operands;
  for (auto arg : lhsFunction.getArguments()) {
    if (arg.getType().isa<ShapedType>()) {
      auto memreftype = arg.getType().cast<MemRefType>();
      auto memref =
          builder.create<memref::AllocOp>(UnknownLoc::get(ctx), memreftype);
      operands.push_back(memref);

      auto afterLastOperand = builder.saveInsertionPoint();

      SmallVector<mlir::Value> indices;
      auto argShape = arg.getType().cast<ShapedType>().getShape();

      if (argShape.size() == 0) {
        buildArgInit(ctx, builder, memref, indices);
      } else {
        for (auto dimSize : argShape) {
          // Create for op.
          auto forOp = builder.create<scf::ForOp>(
              UnknownLoc::get(ctx),
              builder.create<arith::ConstantIndexOp>(UnknownLoc::get(ctx), 0),
              builder.create<arith::ConstantIndexOp>(UnknownLoc::get(ctx),
                                                     dimSize),
              builder.create<arith::ConstantIndexOp>(UnknownLoc::get(ctx), 1));

          indices.push_back(forOp.getInductionVar());

          // Set insertion point inside body for next loop to be inserted.
          builder.setInsertionPointToStart(forOp.getBody());

          // If we are in the innermost loop, create the declaration.
          if (indices.size() == argShape.size()) {
            buildArgInit(ctx, builder, memref, indices);
          }
        }
      }

      builder.restoreInsertionPoint(afterLastOperand);

    } else if (arg.getType().isa<FloatType>()) {
      operands.push_back(builder.create<arith::ConstantOp>(
          UnknownLoc::get(ctx), builder.getF64Type(),
          builder.getF64FloatAttr(0.0f)));
    } else {
      llvm::outs() << "Type: " << arg.getType() << "\n";
      assert(false && "Unsupported type");
    }
  }
  
  // Copy LHS operands into RHS operands
  SmallVector<mlir::Value> rhsOperands;
  int lhsOperandIdx = 0;
  for (auto arg : rhsFunction.getArguments()) {
    if (arg.getType().isa<ShapedType>()) {
      auto memreftype = arg.getType().cast<MemRefType>();
      auto memref =
          builder.create<memref::AllocOp>(UnknownLoc::get(ctx), memreftype);
      rhsOperands.push_back(memref);

      auto afterLastOperand = builder.saveInsertionPoint();

      SmallVector<mlir::Value> indices;
      auto argShape = arg.getType().cast<ShapedType>().getShape();

      if (argShape.size() == 0) {
        copyArgInit(ctx, builder, memref, indices, operands[lhsOperandIdx]);
      } else {
        for (auto dimSize : argShape) {
          // Create for op.
          auto forOp = builder.create<scf::ForOp>(
              UnknownLoc::get(ctx),
              builder.create<arith::ConstantIndexOp>(UnknownLoc::get(ctx), 0),
              builder.create<arith::ConstantIndexOp>(UnknownLoc::get(ctx),
                                                     dimSize),
              builder.create<arith::ConstantIndexOp>(UnknownLoc::get(ctx), 1));

          indices.push_back(forOp.getInductionVar());

          // Set insertion point inside body for next loop to be inserted.
          builder.setInsertionPointToStart(forOp.getBody());

          // If we are in the innermost loop, create the declaration.
          if (indices.size() == argShape.size()) {
            copyArgInit(ctx, builder, memref, indices, operands[lhsOperandIdx]);
          }
        }
      }

      builder.restoreInsertionPoint(afterLastOperand);

    } else if (arg.getType().isa<FloatType>()) {
      rhsOperands.push_back(builder.create<arith::ConstantOp>(
          UnknownLoc::get(ctx), builder.getF64Type(),
          builder.getF64FloatAttr(0.0f)));
    } else {
      llvm::outs() << "Type: " << arg.getType() << "\n";
      assert(false && "Unsupported type");
    }
    lhsOperandIdx++;
  }

  // Call functions on the created operands.
  auto lhsCallOp = builder.create<func::CallOp>(
      UnknownLoc::get(ctx), lhsFunctionClone.getResultTypes(),
      lhsFunctionClone.getSymName(), operands);

  auto rhsCallOp = builder.create<func::CallOp>(
      UnknownLoc::get(ctx), rhsFunctionClone.getResultTypes(),
      rhsFunctionClone.getSymName(), rhsOperands);

  auto afterLastCallOp = builder.saveInsertionPoint();

  // Assert element-wise equality.
  SmallVector<mlir::Value> indices;
  auto shape = lhsFunction.getResultTypes()[0].cast<ShapedType>().getShape();

  if (shape.size() == 0) {
    buildCheck(ctx, builder, lhsCallOp, rhsCallOp, indices);
  } else {
    for (auto dimSize : shape) {
      // Create for op.
      auto forOp = builder.create<scf::ForOp>(
          UnknownLoc::get(ctx),
          builder.create<arith::ConstantIndexOp>(UnknownLoc::get(ctx), 0),
          builder.create<arith::ConstantIndexOp>(UnknownLoc::get(ctx), dimSize),
          builder.create<arith::ConstantIndexOp>(UnknownLoc::get(ctx), 1));

      indices.push_back(forOp.getInductionVar());

      // Set insertion point inside body for next loop to be inserted.
      builder.setInsertionPointToStart(forOp.getBody());

      // If we are in the innermost loop, create the equality check.
      if (indices.size() == shape.size()) {
        buildCheck(ctx, builder, lhsCallOp, rhsCallOp, indices);
      }
    }
  }

  // Must have the check function declaration.
  builder.setInsertionPoint(&moduleBlock, moduleBlock.begin());
  func::FuncOp cbmcAssertFwdDecl = builder.create<func::FuncOp>(
      UnknownLoc::get(ctx), "cbmc_assert",
      mlir::FunctionType::get(ctx, {}, {builder.getF64Type()}));
  cbmcAssertFwdDecl.setPrivate();

  /*func::FuncOp isnanfFwdDecl = builder.create<func::FuncOp>(
      UnknownLoc::get(ctx), "!isnanf",
      mlir::FunctionType::get(ctx, {builder.getF64Type()},
                              {builder.getI1Type()}));
  isnanfFwdDecl.setPrivate();*/
  
  func::FuncOp cbmcDeclareFwdDecl = builder.create<func::FuncOp>(
      UnknownLoc::get(ctx), "__VERIFIER_nondet_rational",
      mlir::FunctionType::get(ctx, {}, {builder.getF64Type()}));
  cbmcDeclareFwdDecl.setPrivate();

  // Must have a return op.
  builder.restoreInsertionPoint(afterLastCallOp);
  builder.create<func::ReturnOp>(UnknownLoc::get(ctx));

  return module;
}

void finalizeCCode(std::string &cCode) {
  std::regex cbmcAssertRegex(".*cbmc_assert.*");
  cCode = std::regex_replace(cCode, cbmcAssertRegex,
                             "        __CPROVER_assert(0, \"unreachable?\");");

  std::regex doubleRegex("double");
  //cCode = std::regex_replace(cCode, doubleRegex, "float");
  cCode = std::regex_replace(cCode, doubleRegex, "__CPROVER_rational");
  
  std::regex v1initZeroRegex(R"(\(__CPROVER_rational\)0\.0e\+00)");
  cCode = std::regex_replace(cCode, v1initZeroRegex, "0");
  std::regex v1initOneRegex(R"(\(__CPROVER_rational\)1\.0+e\+00)");
  cCode = std::regex_replace(cCode, v1initOneRegex, "1");
  std::regex v1initMinusOneRegex(R"(\(__CPROVER_rational\)-1\.0+e\+00)");
  cCode = std::regex_replace(cCode, v1initMinusOneRegex, "-1");

  cCode = "extern int __VERIFIER_nondet_int();\n"
          "extern float __VERIFIER_nondet_float();\n"
          "extern __CPROVER_rational __VERIFIER_nondet_rational();\n"
          "#include <stdio.h>\n"
          "#include <stdbool.h>\n"
          "\n" +
          cCode;
}

std::string runCmd(std::string command) {
  // Capture stderr too.
  command += " 2>&1";

  FILE *pipe = popen(command.c_str(), "r");
  if (!pipe) {
    assert(false && "Couldn't run command");
  }

  std::array<char, 128> buffer;
  std::string outs;
  while (fgets(buffer.data(), 128, pipe) != NULL) {
    outs += buffer.data();
  }

  return outs;
}

bool checkValidate(ModuleOp lhsModule, ModuleOp rhsModule,
                   bool printArgsAndResults, bool printResults) {
  auto lhsModuleCpy = lhsModule.clone();
  auto rhsModuleCpy = rhsModule.clone();

  // Convert rank 0 memrefs to scalars, since they can pose a mismatch in the
  // function signatures.
  PassManager pmMemref(lhsModuleCpy->getContext());
  //pmMemref.addPass(createMemrefRank0ToScalarPass());
  if (failed(pmMemref.run(lhsModuleCpy)) || failed(pmMemref.run(rhsModuleCpy))) {
    llvm::errs() << "Could not convert rank 0 memrefs to scalars\n";
    assert(false);
  }

  auto lhsFunctions = getFunctions(lhsModuleCpy);
  assert(lhsFunctions.size() == 1 && "Expected exactly one function");
  auto lhsFunction = lhsFunctions.front();

  auto rhsFunctions = getFunctions(rhsModuleCpy);
  assert(rhsFunctions.size() == 1 && "Expected exactly one function");
  auto rhsFunction = rhsFunctions.front();

  // Check if the functions have the same signature.
  // - Arguments.
  for (auto it : llvm::zip(lhsFunction.getArguments(), rhsFunction.getArguments())) {
    auto lhsArg = std::get<0>(it);
    auto rhsArg = std::get<1>(it);

    if (lhsArg.getType() != rhsArg.getType()) {
      llvm::outs() << "VALID ERROR: Function argument types do not match with LHS: "
                   << lhsArg.getType() << ", RHS: " << rhsArg.getType() << "\n";
      return false;
    }
  }

  // - Results.
  for (auto it : llvm::zip(lhsFunction.getResultTypes(), rhsFunction.getResultTypes())) {
    auto lhsRes = std::get<0>(it);
    auto rhsRes = std::get<1>(it);

    if (lhsRes != rhsRes) {
      llvm::outs() << "VALID ERROR: Function result types do not match with LHS: "
                   << lhsRes << ", RHS: " << rhsRes << "\n";
      return false;
    }
  }

  auto *ctx = lhsFunction->getContext();

  // lhsFunction->dump();
  // rhsFunction->dump();

  // Assemble module.
  auto module = buildModule(lhsFunction, rhsFunction);

  // Inline and lower to affine. The resulting IR should be in
  // SCF, MemRef, Arith and func dialect.
  auto pm = PassManager(ctx);
  pm.addPass(mlir::createInlinerPass());
  pm.addPass(createMemrefToScfPass());
  pm.addPass(memref::createFoldMemRefAliasOpsPass());
  pm.addPass(createLowerAffinePass());

  if (failed(pm.run(module.get()))) {
    llvm::errs() << "Could not inline or lower to affine\n";
    assert(false);
  }

  // Remove all functions except the main function.
  module->walk([](mlir::Operation *op) {
    if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
      if (funcOp.getName() != "main") {
        funcOp.erase();
      }
    }
  });

  // Translate the IR to C.
  std::string cCode;
  llvm::raw_string_ostream os(cCode);

  if (failed(emitc::translateToCpp(module.get(), os))) {
    failed(emitc::translateToCpp(module.get(), llvm::outs()));
    llvm::errs() << "Could not translate to Cpp with emitc\n";
    assert(false);
  }

  finalizeCCode(cCode);

  // Write C code to file.
  std::ofstream cFile("/tmp/cbmc.c");
  cFile << cCode;
  cFile.close();

  // Run cbmc.
  //std::string cbmcOut = runCmd("cbmc /tmp/cbmc.c -cvc5 -json-ui -verbosity 5");
  std::string cbmcOut = runCmd("cbmc --no-bounds-check --no-signed-overflow-check /tmp/cbmc.c -cvc5 -json-ui -verbosity 5");
  //std::string cbmcOut = runCmd("cbmc --no-bounds-check /tmp/cbmc.c -cvc5 -json-ui -verbosity 5");

  // Check if "VERIFICATION SUCCESSFUL" in cbmcOut.
  return cbmcOut.find("VERIFICATION SUCCESSFUL") != std::string::npos;
}
