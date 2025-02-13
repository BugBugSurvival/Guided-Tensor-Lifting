/* This file is part of the MlirSynth Project. */

#include "bindings/c/MlirSynthCApi.h"
#include "bindings/c/MlirSynthPasses.h"

#include "mlir-c/Support.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir-c/Pass.h"
#include "transforms/Passes.h"

#include <iostream>
#include <pybind11/pytypes.h>

namespace py = pybind11;
using namespace mlir;
using namespace mlir::python;
using namespace mlir::python::adaptors;


PYBIND11_MODULE(_synth, m) {
  auto synth_m = m.def_submodule("synth");

  synth_m.def("register_passes", []() { registerMlirSynthPasses(); });

  synth_m.def(
      "register_dialects",
      [](MlirContext context) { registerDialects(context); },
      py::arg("context") = py::none());

  synth_m.def(
      "lower_chlo_to_affine",
      [](MlirModule &module, bool expandArithOps) {
        lowerCHLOToAffine(module, expandArithOps);
      },
      py::arg("module"), py::arg("expand_arith_ops"));

  synth_m.def(
    "check_validate",
    [](MlirModule lhsModule, MlirModule rhsModule, bool printArgsAndResults, bool printResults) {
      return checkValidate(lhsModule, rhsModule, printArgsAndResults, printResults);
    },
    py::arg("lhs_module"), py::arg("rhs_module"), py::arg("print_args_and_results") = false, py::arg("print_results") = false);
}