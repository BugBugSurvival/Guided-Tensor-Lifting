from enum import Enum
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import mlir_synth.synth as mlir_synth
from mlir_synth.ir import *
from mlir_synth.passmanager import *


class OPT(Enum):
    RANK0_ARGS_TO_SCALARS = 1


def check_equiv(mlir_low_level_str, mlir_high_level_str, options=[]):
    with Context():
        mlir_synth.register_dialects()
        mlir_synth.register_passes()

        # Low level
        mod_ll = Module.parse(mlir_low_level_str)
        pm_ll = PassManager.parse("change-sizes,return-output-arg")
        pm_ll.run(mod_ll)

        # - Get the function annotated with the "irsynth.original" attribute
        func_ll = mod_ll.body.operations[0]

        # High level
        mod_hl = Module.parse(mlir_high_level_str)
        func_hl = mod_hl.body.operations[0]
        func_hl.attributes["irsynth.raised"] = UnitAttr.get()

        # Create modules with only the relevant functions
        mod_ll_only = Module.create(Location.unknown())
        ip = InsertionPoint(mod_ll_only.body)
        ip.insert(func_ll.detach_from_parent())

        mod_hl_only = Module.create(Location.unknown())
        ip = InsertionPoint(mod_hl_only.body)
        ip.insert(func_hl.detach_from_parent())

        # High level
        # - Lower to affine
        mlir_synth.lower_chlo_to_affine(mod_hl_only, False)
        # - Convert rank-0 memrefs to scalars
        if OPT.RANK0_ARGS_TO_SCALARS in options:
            pm = PassManager.parse("fold-memref-alias-ops,memref-rank0-to-scalar")
            pm.run(mod_hl_only)

        #print(mod_ll_only)
        #print(mod_hl_only)

        # Check equivalence
        return mlir_synth.check_validate(mod_ll_only, mod_hl_only)


def to_mlir_hlo(kernel, arg_ranks):
    arg_shapes = [tuple([3] * rank) for rank in arg_ranks]

    args = []
    for shape in arg_shapes:
        if len(shape) == 0:
            args.append(jnp.float64(1))
        else:
            args.append(jnp.ones(shape, dtype=jnp.float64))

    jax_kernel = jax.jit(kernel, backend="cpu", keep_unused=True)
    return str(jax_kernel.lower(*args).compiler_ir(dialect='mhlo'))


def verify(lowlevel_mlir_file, kernel, arg_ranks, options):
    with open(lowlevel_mlir_file, "r") as f:
        mlir_low_level_str = f.read()

    mlir_hlo_str = to_mlir_hlo(kernel, arg_ranks)
    return check_equiv(mlir_low_level_str, mlir_hlo_str, options)


infos = {
    "artificial": [
        (
            "benchmark/tensor_algebra/artificial/1_taco.mlir",
            lambda A, B, C, D: B + C - D,
            [1, 1, 1, 1],
        ),
        (
            "benchmark/tensor_algebra/artificial/2_taco.mlir",
            lambda A, B, C: B + C
            [2, 2, 2],
        ),
        (
            "benchmark/tensor_algebra/artificial/3_taco.mlir",
            lambda A, B, C: jnp.einsum('ij,j->ij', B, C),
            [2, 2, 1],
        ),
        (
            "benchmark/tensor_algebra/artificial/4_taco.mlir",
            lambda A, B, C: jnp.einsum('ik,kj->ij', B, C) + A,
            [2, 2, 2],
        ),
        (
            "benchmark/tensor_algebra/artificial/5_taco.mlir",
            # lambda A, B: jnp.einsum('i->ij', B),
            lambda A, B: jnp.broadcast_to(B, A.shape).transpose(1, 0),
            [2, 1],
        ),
        (
            "benchmark/tensor_algebra/artificial/6_taco.mlir",
            lambda A, B, C: jnp.einsum('ij,j->ij', B, C),
            [2, 2, 1],
        ),
        (
            "benchmark/tensor_algebra/artificial/7_taco.mlir",
            lambda A, B, C: jnp.einsum('ijk,k->ij', B, C),
            [2, 3, 1],
        ),
        (
            "benchmark/tensor_algebra/artificial/8_taco.mlir",
            lambda A, B, C, D: jnp.einsum('ikl,lj,kj->ij', B, C, D),
            [2, 3, 2, 2],
        ),
        (
            "benchmark/tensor_algebra/artificial/9_taco.mlir",
            lambda A, B, C: jnp.einsum('i,i->i', B, C),
            [1, 1, 1],
        ),
        (
            "benchmark/tensor_algebra/artificial/10_taco.mlir",
            lambda A, B, C, D, E: B + C + D + E,
            [1, 1, 1, 1, 1],
        ),
        (
            "benchmark/tensor_algebra/artificial/add_array.mlir",
            lambda A, B, C: B + C,
            [1, 1, 1],
        ),
        (
            "benchmark/tensor_algebra/artificial/add.mlir",
            lambda A, B: A + B,
            [0, 0],
            [OPT.RANK0_ARGS_TO_SCALARS]
        ),
        (
            "benchmark/tensor_algebra/artificial/alpha.mlir",
            lambda A, B, C, D, E: A + D * E * B,
            [1, 1, 1, 0, 0],
            [OPT.RANK0_ARGS_TO_SCALARS]
        ),
        (
            "benchmark/tensor_algebra/artificial/array_prod_scalar.mlir",
            lambda A, B, C: C * B,
            [1, 0, 1],
            [OPT.RANK0_ARGS_TO_SCALARS]
        ),
        (
            "benchmark/tensor_algebra/artificial/div.mlir",
            lambda A, B, C: A / B,
            [1, 1, 1]
        ),
        (
            "benchmark/tensor_algebra/artificial/gemm.mlir",
            lambda A, B, C: jnp.einsum('ik,kj->ij', B, C),
            [2, 2, 2]
        ),
    ],
    "blas": [
        (
            "benchmark/tensor_algebra/blas/dot.mlir",
            lambda A, B: jnp.einsum('i,i->', A, B),
            [1, 1]
        ),
        (
            "benchmark/tensor_algebra/blas/gemv.mlir",
            lambda A, B, C: jnp.einsum('ij,j->i', A, B),
            [2, 1, 1]
        ),
        (
            "benchmark/tensor_algebra/blas/ger.mlir",
            lambda A, B, C: jnp.einsum('i,j->ij', A, B),
            [1, 1, 2]
        ),
    ],
    "darknet": [
        (
            "benchmark/tensor_algebra/darknet/gemm_nn.mlir",
            lambda A, B, C, D: jnp.einsum(',ik,kj->ij', A, B, C, optimize='greedy') + D,
            [0, 2, 2, 2],
            [OPT.RANK0_ARGS_TO_SCALARS]
        ),
        (
            "benchmark/tensor_algebra/darknet/gemm_nt.mlir",
            lambda A, B, C, D: jnp.einsum(',ik,jk->ij', A, B, C, optimize='greedy') + D,
            [0, 2, 2, 2],
            [OPT.RANK0_ARGS_TO_SCALARS]
        ),
        (
            "benchmark/tensor_algebra/darknet/gemm_tn.mlir",
            lambda A, B, C, D: jnp.einsum(',ki,kj->ij', A, B, C, optimize='greedy') + D,
            [0, 2, 2, 2],
            [OPT.RANK0_ARGS_TO_SCALARS]
        ),
        (
            "benchmark/tensor_algebra/darknet/gemm_tt.mlir",
            lambda A, B, C, D: jnp.einsum(',ki,jk->ij', A, B, C, optimize='greedy') + D,
            [0, 2, 2, 2],
            [OPT.RANK0_ARGS_TO_SCALARS]
        ),
        (
            "benchmark/tensor_algebra/darknet/mag_array.mlir",
            lambda A: jnp.einsum('i,i->', A, A),
            [1]
        ),
        (
            "benchmark/tensor_algebra/darknet/matrix_add_matrix.mlir",
            lambda A, B: A + B,
            [2, 2]
        ),
        (
            "benchmark/tensor_algebra/darknet/mse_array.mlir",
            lambda A: jnp.einsum('i,i->', A, A),
            [1]
        ),
        (
            "benchmark/tensor_algebra/darknet/mult_add_into_cpu.mlir",
            lambda A, B, C: A * B + C,
            [1, 1, 1]
        ),
        (
            "benchmark/tensor_algebra/darknet/ol_l2_cpu1.mlir",
            lambda A, B, C: (B - A) * (B - A),
            [1, 1, 1]
        ),
        (
            "benchmark/tensor_algebra/darknet/ol_l2_cpu2.mlir",
            lambda A, B, C: B - A,
            [1, 1, 1]
        ),
        (
            "benchmark/tensor_algebra/darknet/scale_array.mlir",
            lambda A, B: jnp.einsum('i,->i', A, B),
            [1, 0],
            [OPT.RANK0_ARGS_TO_SCALARS]
        ),
        (
            "benchmark/tensor_algebra/darknet/scale_matrix.mlir",
            lambda A, B: jnp.einsum('ij,->ij', A, B),
            [2, 0],
            [OPT.RANK0_ARGS_TO_SCALARS]
        ),
        (
            "benchmark/tensor_algebra/darknet/sum_array.mlir",
            lambda A: jnp.einsum('i->', A),
            [1]
        ),
        (
            "benchmark/tensor_algebra/darknet/translate_array.mlir",
            lambda A, B: A + B,
            [1, 0],
            [OPT.RANK0_ARGS_TO_SCALARS]
        ),
    ],
    "dsp": [
        (
            "benchmark/tensor_algebra/dsp/matadd.mlir",
            lambda A, B: A + B,
            [2, 2]
        ),
        (
            "benchmark/tensor_algebra/dsp/matinit.mlir",
            lambda A, B: jnp.full_like(A, B),
            [2, 0],
            [OPT.RANK0_ARGS_TO_SCALARS]
        ),
        (
            "benchmark/tensor_algebra/dsp/matmul.mlir",
            lambda A, B, C: jnp.einsum('ik,kj->ij', A, B),
            [2, 2, 2]
        ),
        (
            "benchmark/tensor_algebra/dsp/matscal.mlir",
            lambda A, B: jnp.einsum('ij,->ij', A, B),
            [2, 0],
            [OPT.RANK0_ARGS_TO_SCALARS]
        ),
        (
            "benchmark/tensor_algebra/dsp/matsub.mlir",
            lambda A, B: A - B,
            [2, 2]
        ),
        (
            "benchmark/tensor_algebra/dsp/vadd.mlir",
            lambda A, B: A + B,
            [1, 1]
        ),
        (
            "benchmark/tensor_algebra/dsp/vcopy.mlir",
            lambda A, B: jnp.einsum('i->i', A),
            [1, 1]
        ),
        (
            "benchmark/tensor_algebra/dsp/vfill.mlir",
            lambda A, B: jnp.full_like(A,B),
            [1, 0],
            [OPT.RANK0_ARGS_TO_SCALARS]
        ),
        (
            "benchmark/tensor_algebra/dsp/vmul.mlir",
            lambda A, B, C: jnp.einsum('i,i->i', A, B),
            [1, 1, 1]
        ),
        (
            "benchmark/tensor_algebra/dsp/vneg.mlir",
            lambda A: jnp.einsum('i->i', -A),
            [1]
        ),
        (
            "benchmark/tensor_algebra/dsp/voffset.mlir",
            lambda A, B: A + B,
            [1, 0],
            [OPT.RANK0_ARGS_TO_SCALARS]
        ),
        (
            "benchmark/tensor_algebra/dsp/vrecip.mlir",
            lambda A: 1 / A,
            [1]
        ),
        (
            "benchmark/tensor_algebra/dsp/vscal.mlir",
            lambda A, B: jnp.einsum('i,->i', A, B),
            [1, 0],
            [OPT.RANK0_ARGS_TO_SCALARS]
        ),
        (
            "benchmark/tensor_algebra/dsp/vsub.mlir",
            lambda A, B, C: A - B,
            [1, 1, 1]
        ),
        (
            "benchmark/tensor_algebra/dsp/w_vec.mlir",
            lambda A, B, C, D: A * C + B,
            [1, 1, 0, 1],
            [OPT.RANK0_ARGS_TO_SCALARS]
        ),
    ],
    "dspstone": [
        (
            "benchmark/tensor_algebra/dspstone/mat1x3.mlir",
            lambda A, B, C: jnp.einsum('ij,j->i', A, B),
            [2, 1, 1]
        ),
        (
            "benchmark/tensor_algebra/dspstone/matrix1.mlir",
            lambda A, B, C: jnp.einsum('ik,jk->ij', A, B),
            [2, 2, 2]
        ),
        (
            "benchmark/tensor_algebra/dspstone/matrix2.mlir",
            lambda A, B, C: jnp.einsum('ik,jk->ij', A, B),
            [2, 2, 2]
        ),
        (
            "benchmark/tensor_algebra/dspstone/n_real_updates.mlir",
            lambda A, B, C, D: C + A * B,
            [1, 1, 1, 1]
        ),
        (
            "benchmark/tensor_algebra/dspstone/pin_down.mlir",
            lambda A: jnp.einsum('i->i', A) * 1,
            [1]
        ),
    ],
    "makespeare": [
        (
            "benchmark/tensor_algebra/makespeare/sum_of_squares.mlir",
            lambda A: jnp.einsum('i,i->', A, A),
            [1]
        ),
    ],
    "mathfu": [
        (
            "benchmark/tensor_algebra/mathfu/diveq.mlir",
            lambda A, B: A / B,
            [1, 1]
        ),
        (
            "benchmark/tensor_algebra/mathfu/diveq_sca.mlir",
            lambda A, B: A / B,
            [1, 0],
            [OPT.RANK0_ARGS_TO_SCALARS]
        ),
        (
            "benchmark/tensor_algebra/mathfu/len.mlir",
            lambda A, B: jnp.einsum('i,i->', A, A),
            [1, 0],
            [OPT.RANK0_ARGS_TO_SCALARS]
        ),
        (
            "benchmark/tensor_algebra/mathfu/len_sq.mlir",
            lambda A: jnp.einsum('i,i->', A, A),
            [1]
        ),
        (
            "benchmark/tensor_algebra/mathfu/lerp.mlir",
            lambda A, B, C, D: ((B - C) * (1 * D)) + C,
            [1, 1, 1, 0],
            [OPT.RANK0_ARGS_TO_SCALARS]
        ),
        (
            "benchmark/tensor_algebra/mathfu/matmul_sca.mlir",
            lambda A, B, C: jnp.einsum('ij,->ij', A, C),
            [2, 2, 0],
            [OPT.RANK0_ARGS_TO_SCALARS]
        ),
        (
            "benchmark/tensor_algebra/mathfu/muleq.mlir",
            lambda A, B: jnp.einsum('i,i->i', A, B),
            [1, 1]
        ),
        (
            "benchmark/tensor_algebra/mathfu/muleq_sca.mlir",
            lambda A, B: jnp.einsum('i,->i', A, B),
            [1, 0],
            [OPT.RANK0_ARGS_TO_SCALARS]
        ),
        (
            "benchmark/tensor_algebra/mathfu/negate.mlir",
            lambda A: jnp.einsum('i->i', -A),
            [1]
        ),
        (
            "benchmark/tensor_algebra/mathfu/pluseq.mlir",
            lambda A, B: A + B,
            [1, 1]
        ),
        (
            "benchmark/tensor_algebra/mathfu/subeq.mlir",
            lambda A, B: A - B,
            [1, 1]
        ),
        (
            "benchmark/tensor_algebra/mathfu/subeq_sca.mlir",
            lambda A, B: A - B,
            [1, 0],
            [OPT.RANK0_ARGS_TO_SCALARS]
        ),
    ],
    "simpl_array": [
        (
            "benchmark/tensor_algebra/simpl_array/array_inc.mlir",
            lambda A: A + 1,
            [1]
        ),
        (
            "benchmark/tensor_algebra/simpl_array/array_sum.mlir",
            lambda A: jnp.einsum('i->', A),
            [1]
        ),
        (
            "benchmark/tensor_algebra/simpl_array/cube_in_place.mlir",
            lambda A: jnp.einsum('i,i,i->i', A, A, A),
            [1]
        ),
        (
            "benchmark/tensor_algebra/simpl_array/fourth_in_place.mlir",
            lambda A: jnp.einsum('i,i,i,i->i', A, A, A, A),
            [1]
        ),
        (
            "benchmark/tensor_algebra/simpl_array/sum_elts.mlir",
            lambda A: jnp.einsum('i->', A),
            [1]
        ),
    ],
    "utdsp": [
        #(
        #    "benchmark/tensor_algebra/utdsp/dct.mlir",
        #    lambda A: jnp.einsum('', A, A)
        #    [1]
        #),
        (
            "benchmark/tensor_algebra/utdsp/fir_small.mlir",
            lambda A, B: jnp.einsum('i,i->', A, B),
            [1, 1]
        ),
        (
            "benchmark/tensor_algebra/utdsp/histogram.mlir",
            lambda A: jnp.einsum('i->i', A) * 0,
            [1]
        ),
        (
            "benchmark/tensor_algebra/utdsp/lmsfir1.mlir",
            lambda A, B: jnp.einsum('i,i->', A, B),
            [1, 1]
        ),
        (
            "benchmark/tensor_algebra/utdsp/lmsfir2.mlir",
            lambda A, B, C: A * C + B,
            [1, 1, 0],
            [OPT.RANK0_ARGS_TO_SCALARS]
        ),
        (
            "benchmark/tensor_algebra/utdsp/mult_big.mlir",
            lambda A, B, C: jnp.einsum('ik,kj->ij', A, B),
            [2, 2, 2]
        ),
    ],
    "tensor_contractions": [
        (
            "benchmark/tensor_algebra/tensor_contractions/ab_ac_cd.mlir",
            lambda A, B, C: jnp.einsum('ik,kj->ij', A, B) + C,
            [2, 2, 2]
        ),
        (
            "benchmark/tensor_algebra/tensor_contractions/ab_acd_bdc.mlir",
            lambda A, B, C: jnp.einsum('ikl,ljk->ij', B, C) + A,
            [2, 3, 3]
        ),
        (
            "benchmark/tensor_algebra/tensor_contractions/ab_cad_dcb.mlir",
            lambda A, B, C: jnp.einsum('kil,lkj->ij', B, C) + A,
            [2, 3, 3]
        ),
        (
            "benchmark/tensor_algebra/tensor_contractions/abc_acd_db.mlir",
            lambda A, B, C: jnp.einsum('ikl,lj->ijk', B, C) + A,
            [3, 3, 2]
        ),
        (
            "benchmark/tensor_algebra/tensor_contractions/abc_ad_bdc.mlir",
            lambda A, B, C: jnp.einsum('il,jlk->ijk', B, C) + A,
            [3, 2, 3]
        ),
        (
            "benchmark/tensor_algebra/tensor_contractions/abc_bda_dc.mlir",
            lambda A, B, C: jnp.einsum('jli,lk->ijk', B, C) + A,
            [3, 3, 2]
        ),
        (
            "benchmark/tensor_algebra/tensor_contractions/abcd_aebf_dfce.mlir",
            lambda A, B, C: jnp.einsum('imjn,lnkm->ijkl', B, C) + A,
            [4, 4, 4]
        ),
        (
            "benchmark/tensor_algebra/tensor_contractions/abcd_aebf_fdec.mlir",
            lambda A, B, C: jnp.einsum('imjn,nlmk->ijkl', B, C) + A,
            [4, 4, 4]
        ),
    ],
    "blend": [
        (
            "benchmark/tensor_algebra/blend/linear_dodge_8.mlir",
            lambda A, B: A + B,
            [2, 2]
        ),
    ],
    "llama": [
        (
            "benchmark/tensor_algebra/llama/matmul.mlir",
            lambda A, B, C: jnp.einsum('ij,j->i', C, B),
            [1, 1, 2]
        ),
        (
            "benchmark/tensor_algebra/llama/rmsnorm_part1.mlir",
            lambda A: jnp.einsum('i,i->', A, A),
            [1]
        ),
        (
            "benchmark/tensor_algebra/llama/softmax_part2.mlir",
            lambda B, A, C: B - C,
            [1, 1, 0],
            [OPT.RANK0_ARGS_TO_SCALARS]
        ),
        (
            "benchmark/tensor_algebra/llama/softmax_part3.mlir",
            lambda A, B,: jnp.einsum('i->', A),
            [1, 0],
            [OPT.RANK0_ARGS_TO_SCALARS]
        ),
        (
            "benchmark/tensor_algebra/llama/softmax_part4.mlir",
            lambda A, B: A / B,
            [1, 0],
            [OPT.RANK0_ARGS_TO_SCALARS]
        ),
        (
            "benchmark/tensor_algebra/llama/transformer_part4.mlir",
            lambda A, B, C: jnp.einsum('i,i->i', A, B),
            [1, 1, 1]
        ),
    ],
}

FAIL_AS_FLOAT = ["benchmark/tensor_algebra/simpl_array/cube_in_place.mlir",
	"benchmark/tensor_algebra/dspstone/matrix2.mlir",
	"benchmark/tensor_algebra/tensor_contractions/ab_ac_cd.mlir",
	"benchmark/tensor_algebra/tensor_contractions/ab_acd_bdc.mlir",
	"benchmark/tensor_algebra/tensor_contractions/ab_cad_dcb.mlir",
        "benchmark/tensor_algebra/tensor_contractions/abc_acd_db.mlir",
        "benchmark/tensor_algebra/tensor_contractions/abc_ad_bdc.mlir",
        "benchmark/tensor_algebra/tensor_contractions/abc_bda_dc.mlir",
        "benchmark/tensor_algebra/tensor_contractions/abcd_aebf_dfce.mlir",
        "benchmark/tensor_algebra/tensor_contractions/abcd_aebf_fdec.mlir"
        ]

total_verified = 0
verified_f = open("verified.csv", "w")
verified_f.write("benchmark;verified;time\n")

for suite, benchmarks in infos.items():
    for benchmark in benchmarks:
        v_time = 0
        options = []
        if len(benchmark) == 3:
            lowlevel_mlir_file, kernel, arg_ranks = benchmark
        elif len(benchmark) == 4:
            lowlevel_mlir_file, kernel, arg_ranks, options = benchmark

        start_v = time.time()
        verified = verify(lowlevel_mlir_file, kernel, arg_ranks, options)
        v_time = time.time() - start_v
        benchname = lowlevel_mlir_file.split("/")[3].split(".")[0]
        answer = "Y" if verified else "N"
        verified_f.write(f"{benchname};{answer};{v_time:.2f}\n")
        verified_f.flush()
        if verified:
          total_verified += 1
        print(
            lowlevel_mlir_file,
            verified
            )

verified_f.close()
print("Total verified:", total_verified)
