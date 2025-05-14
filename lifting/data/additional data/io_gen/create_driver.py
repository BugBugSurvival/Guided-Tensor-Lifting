from clang.cindex import Index, CursorKind, TypeKind
import json
import os
import math
import textwrap


ARRAYS_TYPES = [
    TypeKind.POINTER, TypeKind.CONSTANTARRAY, TypeKind.VARIABLEARRAY,
    TypeKind.INCOMPLETEARRAY, TypeKind.DEPENDENTSIZEDARRAY, TypeKind.TYPEDEF,
]

SCALAR_TYPES = [
    TypeKind.BOOL, TypeKind.CHAR_U, TypeKind.UCHAR, TypeKind.CHAR16,
    TypeKind.CHAR32, TypeKind.USHORT, TypeKind.UINT, TypeKind.ULONG,
    TypeKind.ULONGLONG, TypeKind.UINT128, TypeKind.CHAR_S, TypeKind.SCHAR,
    TypeKind.WCHAR, TypeKind.SHORT, TypeKind.INT, TypeKind.LONG,
    TypeKind.LONGLONG, TypeKind.INT128, TypeKind.FLOAT, TypeKind.DOUBLE,
    TypeKind.LONGDOUBLE,
]


def is_array(var_type):
    """Return True for any pointer / array indirection."""
    return var_type.kind in ARRAYS_TYPES


def is_scalar(var_type):
    return var_type.kind in SCALAR_TYPES

def pointer_depth(ctype):
    depth = 0
    while ctype.kind == TypeKind.POINTER:
        depth += 1
        ctype = ctype.get_pointee()
    return depth


def get_output(output_var, args):
    output = None
    for arg in args:
        if arg.spelling == output_var:
            output = arg
            break
    assert output, "Please indicate the output variable"
    return output


def get_kernel_signature(benchmark_path):
    kernel_name = ""
    return_type = None

    idx = Index.create()
    tu = idx.parse(benchmark_path, args=["-c"])
    for n in tu.cursor.walk_preorder():
        if n.kind == CursorKind.FUNCTION_DECL:
            kernel_name = n.spelling
            return_type = n.result_type
            args = list(n.get_arguments())
            break

    return kernel_name, return_type, args


def get_preamble():
    return textwrap.dedent("""
        #include <stdlib.h>
        #include <stdio.h>
        #include <time.h>
        extern void fill_array(int* arr, int len);
        extern void print_array(const char* name, int* arr, int len);
        extern void print_matrix(const char* name, int** m, int rows, int cols);
        extern void print_tensor3(const char* name, int*** t, int d1, int d2, int d3);
    """)


def _print_line_for_arg(arg, value_profile):
    depth = pointer_depth(arg.type)
    if depth == 0:
        return f'  printf("{arg.spelling}: 1: %d\\n", {arg.spelling});'
    elif depth == 1:
        return f'  print_array("{arg.spelling}", {arg.spelling}, {value_profile[arg.spelling]});'
    elif depth == 2:
        side = int(math.isqrt(value_profile[arg.spelling]))
        return f'  print_matrix("{arg.spelling}", {arg.spelling}, {side}, {side});'
    elif depth == 3:
        side = round(value_profile[arg.spelling] ** (1/3))
        return f'  print_tensor3("{arg.spelling}", {arg.spelling}, {side}, {side}, {side});'
    else:
        raise NotImplementedError("Printing >3‑D not implemented")


def get_print_input_func(args, value_profile):
    header_parts = ["void print_inputs("]
    body_lines = ["  printf(\"input\\n\");"]

    for arg in args:
        depth = pointer_depth(arg.type)
        type_prefix = "int" + "*" * depth
        header_parts.append(f"{type_prefix} {arg.spelling}, ")
        body_lines.append(_print_line_for_arg(arg, value_profile))

    header = "".join(header_parts).rstrip(", ") + "){"
    body = "\n".join(body_lines) + "\n}"
    return header + "\n\n" + body + "\n\n"



def get_print_output_func(print_return_value, output, value_profile):
    header = "void print_output(int sample_id, "
    body_lines = ["  printf(\"output\\n\");"]

    if print_return_value:
        header += "int returnv) {"
        body_lines.append("  printf(\"returnv :1 :%d\\n\", returnv);")
    else:
        depth = pointer_depth(output.type)
        type_prefix = "int" + "*" * depth
        header += f"{type_prefix} {output.spelling}) {{"
        body_lines.append(_print_line_for_arg(output, value_profile))

    body_lines.append("  printf(\"sample_id %d\\n\", sample_id);")
    body = "\n".join(body_lines) + "\n}"
    return header + "\n" + body + "\n"



def _alloc_code(name, depth, total):
    if depth == 1:
        return f'  int* {name} = (int*)malloc({total} * sizeof(int));'
    elif depth == 2:
        side = int(math.isqrt(total))
        return textwrap.dedent(f"""
            int** {name} = (int**)malloc({side} * sizeof(int*));
            for(int i=0;i<{side};++i) {{
              {name}[i] = (int*)malloc({side} * sizeof(int));
            }}
        """)
    elif depth == 3:
        side = round(total ** (1/3))
        return textwrap.dedent(f"""
            int*** {name} = (int***)malloc({side} * sizeof(int**));
            for(int i=0;i<{side};++i) {{
              {name}[i] = (int**)malloc({side} * sizeof(int*));
              for(int j=0;j<{side};++j) {{
                {name}[i][j] = (int*)malloc({side} * sizeof(int));
              }}
            }}
        """)
    else:
        raise NotImplementedError(">3‑D alloc not implemented")


def _fill_code(name, depth, total):
    if depth == 1:
        return f'    fill_array({name}, {total});'
    elif depth == 2:
        side = int(math.isqrt(total))
        return textwrap.dedent(f"""
            for(int i=0;i<{side};++i) {{
              fill_array({name}[i], {side});
            }}
        """)
    elif depth == 3:
        side = round(total ** (1/3))
        return textwrap.dedent(f"""
            for(int i=0;i<{side};++i) {{
              for(int j=0;j<{side};++j) {{
                fill_array({name}[i][j], {side});
              }}
            }}
        """)
    else:
        raise NotImplementedError()


def get_main_func(kernel, return_type, args, output_var, value_profile):
    lines = [
        "\nint main(int argc, char* argv[]){",
        "  srand(time(0));",
        "  int n_io = atoi(argv[1]);",
    ]

    # static allocations
    for arg in args:
        depth = pointer_depth(arg.type)
        if depth == 0:
            lines.append(f"  int {arg.spelling} = {value_profile[arg.spelling]};")
        else:
            lines.append(_alloc_code(arg.spelling, depth, value_profile[arg.spelling]))

    lines.append("  for(int i = 0; i < n_io; i++){")

    # refresh arrays each sample
    for arg in args:
        depth = pointer_depth(arg.type)
        if depth >= 1:
            lines.append(_fill_code(arg.spelling, depth, value_profile[arg.spelling]))

    # print inputs
    arg_names = ", ".join([a.spelling for a in args])
    lines.append(f"    print_inputs({arg_names});")

    # kernel invocation
    call_prefix = "" if return_type.kind == TypeKind.VOID else "int returnv = "
    lines.append(f"    {call_prefix}{kernel}({arg_names});")

    # print outputs
    if return_type.kind == TypeKind.VOID:
        lines.append(f"    print_output(i, {output_var});")
    else:
        lines.append("    print_output(i, returnv);")

    lines.append("  }")
    lines.append("  return 0;")
    lines.append("}")

    return "\n".join(lines)




def write_driver(benchmark_path, benchmark, output_var, value_profile):
	with open(os.path.join(os.path.dirname(benchmark_path), f'main_{benchmark}.c'), 'w') as driver_file:
		driver_file.write(get_preamble())
		kernel, return_type, args = get_kernel_signature(benchmark_path)
		driver_file.write(get_print_input_func(args, value_profile))
		print_return_value = False if return_type.kind == TypeKind.VOID else True
		output = get_output(output_var, args) if not print_return_value else None
		driver_file.write(get_print_output_func(print_return_value, output, value_profile))
		driver_file.write(get_main_func(kernel, return_type, args, output_var, value_profile))


def read_value_profile(value_profile):
	value_profile_dict = dict()
	with open(value_profile, 'r') as value_profile_file:
		value_profile_dict = json.load(value_profile_file)
	return value_profile_dict


def gen_driver(benchmark_path, output_var, value_profile_file):
	benchmark = os.path.basename(benchmark_path)[:-2]
	value_profile = read_value_profile(value_profile_file)
	write_driver(benchmark_path, benchmark, output_var, value_profile)
  


