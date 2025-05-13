from clang.cindex import Index, CursorKind, TypeKind
import json
import os
import math


ARRAYS_TYPES = [
    TypeKind.POINTER, TypeKind.CONSTANTARRAY, TypeKind.VARIABLEARRAY,
    TypeKind.INCOMPLETEARRAY, TypeKind.DEPENDENTSIZEDARRAY, TypeKind.TYPEDEF,
]

SCALAR_TYPES = [
    TypeKind.BOOL, TypeKind.CHAR_U, TypeKind.UCHAR, TypeKind.CHAR16, TypeKind.CHAR32,
    TypeKind.USHORT, TypeKind.UINT, TypeKind.ULONG, TypeKind.ULONGLONG, TypeKind.UINT128,
    TypeKind.CHAR_S, TypeKind.SCHAR, TypeKind.WCHAR, TypeKind.SHORT, TypeKind.INT,
    TypeKind.LONG, TypeKind.LONGLONG, TypeKind.INT128, TypeKind.FLOAT, TypeKind.DOUBLE,
    TypeKind.LONGDOUBLE,
]


def is_matrix(t):
    """True if *t is ‘pointer to pointer’ ⇒ treat as 2‑D matrix."""
    return t.kind == TypeKind.POINTER and t.get_pointee().kind == TypeKind.POINTER


def is_array(t):
    """1‑D linear buffer (pointer-to-scalar or clang array)."""
    return t.kind in ARRAYS_TYPES and not is_matrix(t)


def is_scalar(t):
    return t.kind in SCALAR_TYPES



def get_output(output_var, args):
    for arg in args:
        if arg.spelling == output_var:
            return arg
    raise ValueError("Please indicate the output variable")


def get_kernel_signature(benchmark_path):
    idx = Index.create()
    tu = idx.parse(benchmark_path, args=["-c"])

    for n in tu.cursor.walk_preorder():
        if n.kind == CursorKind.FUNCTION_DECL:
            return n.spelling, n.result_type, list(n.get_arguments())

    raise RuntimeError("Could not find kernel signature")



def get_preamble():
    return (
        "#include <stdlib.h>\n"
        "#include <stdio.h>\n"
        "#include <time.h>\n"
        "extern void fill_array(int* arr, int len);\n"
        "extern void print_array(const char* name, int* arr, int len);\n\n"
    )


def get_print_input_func(args, value_profile):
    header = "void print_inputs("
    body = '  printf("input\\n");\n'

    for arg in args:
        if is_matrix(arg.type):
            side = int(math.isqrt(value_profile[arg.spelling]))
            header += f"int** {arg.spelling}, "
            body += (
                f"  for(int __r=0; __r<{side}; ++__r) "
                f"print_array(\"{arg.spelling}\", {arg.spelling}[__r], {side});\n"
            )
        elif is_array(arg.type):
            header += f"int* {arg.spelling}, "
            body += f'  print_array("{arg.spelling}", {arg.spelling}, {value_profile[arg.spelling]});\n'
        elif is_scalar(arg.type):
            header += f"int {arg.spelling}, "
            body += f'  printf("{arg.spelling}: 1: %d\\n", {arg.spelling});\n'

    header = header.rstrip(", ") + "){\n"
    body += "}\n\n"
    return header + body


def get_print_output_func(print_return_value, output, value_profile):
    header = "void print_output(int sample_id, "
    body = '  printf("output\\n");\n'

    if print_return_value:
        header += "int returnv"
        body += '  printf("returnv :1 :%d\\n", returnv);\n'
    else:
        if is_array(output.type):
            header += "int* "
            body += f'  print_array("{output.spelling}", {output.spelling}, {value_profile[output.spelling]});\n'
        elif is_scalar(output.type):
            header += "int "
            body += f'  printf("{output.spelling}: 1: %d\\n", {output.spelling});\n'
        elif is_matrix(output.type):
            side = int(math.isqrt(value_profile[output.spelling]))
            header += "int** "
            body += (
                f"  for(int __r=0; __r<{side}; ++__r) "
                f"print_array(\"{output.spelling}\", {output.spelling}[__r], {side});\n"
            )
        header += output.spelling

    header += "){\n"
    body += '  printf("sample_id %d\\n", sample_id);\n}\n'
    return header + body


def get_main_func(kernel, return_type, args, output_var, value_profile):
    main_opening = (
        "\nint main(int argc, char* argv[]){\n"
        "  srand(time(0));\n"
        "  int n_io = atoi(argv[1]);\n"
    )
    main_for = "\n  for(int i = 0; i < n_io; i++){\n"

    args_initialization = ""
    array_initializations = ""

    kernel_call = (
        f"    {kernel}(" if return_type.kind == TypeKind.VOID else f"    int returnv = {kernel}("
    )
    print_inputs_call = "    print_inputs("

    for arg in args:
        if is_matrix(arg.type):
            side = int(math.isqrt(value_profile[arg.spelling]))
            args_initialization += (
                f"  int** {arg.spelling} = (int**)malloc({side} * sizeof(int*));\n"
                f"  for(int __r = 0; __r < {side}; ++__r) {{\n"
                f"    {arg.spelling}[__r] = (int*)malloc({side} * sizeof(int));\n"
                f"  }}\n"
            )
            array_initializations += (
                f"    for(int __r = 0; __r < {side}; ++__r) "
                f"fill_array({arg.spelling}[__r], {side});\n"
            )
        elif is_array(arg.type):
            args_initialization += (
                f"  int* {arg.spelling} = (int*)malloc({value_profile[arg.spelling]} * sizeof(int));\n"
            )
            array_initializations += f"    fill_array({arg.spelling}, {value_profile[arg.spelling]});\n"
        elif is_scalar(arg.type):
            args_initialization += f"  int {arg.spelling} = {value_profile[arg.spelling]};\n"

        kernel_call += f"{arg.spelling}, "
        print_inputs_call += f"{arg.spelling}, "

    kernel_call = kernel_call.rstrip(", ") + ");\n"
    print_inputs_call = print_inputs_call.rstrip(", ") + ");\n"

    print_output_call = "    print_output(i, "
    print_output_call += (
        "returnv);\n" if return_type.kind != TypeKind.VOID else f"{output_var});\n"
    )

    main_closing = "  }\n  return 0;\n}"

    return (
        main_opening
        + args_initialization
        + main_for
        + array_initializations
        + print_inputs_call
        + kernel_call
        + print_output_call
        + main_closing
    )



def write_driver(benchmark_path, benchmark, output_var, value_profile):
    kernel, return_type, args = get_kernel_signature(benchmark_path)

    with open(os.path.join(os.path.dirname(benchmark_path), f"main_{benchmark}.c"), "w") as f:
        f.write(get_preamble())
        f.write(get_print_input_func(args, value_profile))
        print_return_value = return_type.kind != TypeKind.VOID
        output = None if print_return_value else get_output(output_var, args)
        f.write(get_print_output_func(print_return_value, output, value_profile))
        f.write(get_main_func(kernel, return_type, args, output_var, value_profile))


def read_value_profile(fp):
    with open(fp, "r") as f:
        return json.load(f)


def gen_driver(benchmark_path, output_var, value_profile_file):
    benchmark = os.path.basename(benchmark_path)[:-2]
    value_profile = read_value_profile(value_profile_file)
    write_driver(benchmark_path, benchmark, output_var, value_profile)
