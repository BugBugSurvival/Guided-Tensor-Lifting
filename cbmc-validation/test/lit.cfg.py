# -*- Python -*-

import os
import platform
import re
import subprocess
import tempfile

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = 'SYNTHESIZER'

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.mlir', '.py']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.mlirsynth_obj_root, 'test')

config.substitutions.append(('%PATH%', config.environment['PATH']))
config.substitutions.append(('%mlirsynth_src_root%', config.mlirsynth_src_root))

llvm_config.with_system_environment(
    ['HOME', 'INCLUDE', 'LIB', 'TMP', 'TEMP'])

llvm_config.use_default_substitutions()

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = ['Inputs', 'Examples', 'CMakeLists.txt', 'README.txt', 'LICENSE.txt']

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.mlirsynth_obj_root, 'test')
config.synthesizer_tools_dir = os.path.join(config.mlirsynth_obj_root, 'bin')

# Tweak the PATH to include the tools dir.
llvm_config.with_environment('PATH', config.llvm_tools_dir, append_path=True)

# Add the verification tools build dirs to the PATH.
llvm_config.with_environment('PATH',
                             os.path.join(config.test_source_root, '../deps/cbmc/build/bin'),
                             append_path=True)
llvm_config.with_environment('PATH',
                             os.path.join(config.test_source_root, '../deps/cvc5/build/bin'),
                             append_path=True)

llvm_config.with_environment('PYTHONPATH', [
    os.path.join(config.mlir_obj_root, 'python_packages', 'synth'),
], append_path=True)

tool_dirs = [config.synthesizer_tools_dir, config.llvm_tools_dir]
tools = [
    'synthesizer',
    'opt',
    'test-executor',
    'translate-affine-to-python'
]

llvm_config.add_tool_substitutions(tools, tool_dirs)
