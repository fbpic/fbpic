#!/usr/bin/env python
"""
Phase 6 verification: compile every GPU kernel in fbpic to HSACO binaries
across RDNA4 (gfx1201), CDNA2/3/4 (gfx90a, gfx942, gfx950), and best-effort
CDNA1 (gfx908), inspect ELF wavefront metadata and FP64 atomic ISA ops,
and validate CDNA/RDNA wavefront-aligned launch-geometry helpers.
"""
import argparse
import collections
import importlib
import os
import re
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

# Ensure repo root is importable
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from numba import types  # noqa: E402

from fbpic.lpa_utils.external_fields import ExternalField  # noqa: E402
from fbpic.utils.gpu_backend import (  # noqa: E402
    gpu_backend,
    get_registered_kernels,
    cuda_tpb_bpg_1d,
    cuda_tpb_bpg_2d,
    get_copy_tpb_2d,
    get_spectral_tpb_2d,
    get_deposit_tpb,
    get_gather_tpb,
)

KERNEL_MODULES = [
    'fbpic.boundaries.cuda_methods',
    'fbpic.boundaries.moving_window',
    'fbpic.boundaries.particle_buffer_handling',
    'fbpic.boundaries.pml_damping',
    'fbpic.fields.cuda_methods',
    'fbpic.fields.spectral_transform.cuda_methods',
    'fbpic.openpmd_diag.boosted_field_diag',
    'fbpic.openpmd_diag.cuda_methods',
    'fbpic.particles.deposition.cuda_methods',
    'fbpic.particles.deposition.cuda_methods_one_mode',
    'fbpic.particles.deposition.cuda_methods_unsorted',
    'fbpic.particles.elementary_process.compton.cuda_methods',
    'fbpic.particles.elementary_process.cuda_numba_utils',
    'fbpic.particles.elementary_process.ionization.cuda_methods',
    'fbpic.particles.elementary_process.synchrotron.cuda_methods',
    'fbpic.particles.gathering.cuda_methods',
    'fbpic.particles.gathering.cuda_methods_one_mode',
    'fbpic.particles.push.cuda_methods',
    'fbpic.particles.spin.cuda_methods',
    'fbpic.particles.tracking.tracking',
    'fbpic.particles.utilities.cuda_sorting',
]
for _mod in KERNEL_MODULES:
    importlib.import_module(_mod)

PRIMARY_TARGETS = ['gfx1201', 'gfx90a', 'gfx942', 'gfx950']
BEST_EFFORT_TARGETS = ['gfx908']
ALL_TARGETS = PRIMARY_TARGETS + BEST_EFFORT_TARGETS

_DEP_MOD = 'fbpic.particles.deposition.cuda_methods'
_DEP1_MOD = 'fbpic.particles.deposition.cuda_methods_one_mode'
_DEPU_MOD = 'fbpic.particles.deposition.cuda_methods_unsorted'
_SYNC_MOD = 'fbpic.particles.elementary_process.synchrotron.cuda_methods'

ATOMIC_KERNELS = {
    f'{_DEP_MOD}:deposit_rho_gpu_linear',
    f'{_DEP_MOD}:deposit_J_gpu_linear',
    f'{_DEP_MOD}:deposit_rho_gpu_cubic',
    f'{_DEP_MOD}:deposit_J_gpu_cubic',
    f'{_DEP1_MOD}:deposit_rho_gpu_linear_one_mode',
    f'{_DEP1_MOD}:deposit_J_gpu_linear_one_mode',
    f'{_DEP1_MOD}:deposit_rho_gpu_cubic_one_mode',
    f'{_DEP1_MOD}:deposit_J_gpu_cubic_one_mode',
    f'{_DEPU_MOD}:deposit_rho_gpu_unsorted',
    f'{_DEPU_MOD}:deposit_J_gpu_unsorted',
    f'{_SYNC_MOD}:gather_synchrotron_cuda',
}


def build_signature_map():
    """
    Build exact Numba ``argtypes`` tuples keyed by registered kernel name.

    Returns
    -------
    sigs : dict
        Mapping ``"module:qualname" -> list of (variant_name, argtypes)``.
    """
    f1 = types.Array(types.float64, 1, 'C')
    f2 = types.Array(types.float64, 2, 'C')
    f3 = types.Array(types.float64, 3, 'C')
    f32_1 = types.Array(types.float32, 1, 'C')
    c1 = types.Array(types.complex128, 1, 'C')
    c2 = types.Array(types.complex128, 2, 'C')
    c3 = types.Array(types.complex128, 3, 'C')
    i16_1 = types.Array(types.int16, 1, 'C')
    i64_1 = types.Array(types.int64, 1, 'C')
    i64_2 = types.Array(types.int64, 2, 'C')
    u64_1 = types.Array(types.uint64, 1, 'C')
    u64_2 = types.Array(types.uint64, 2, 'C')
    f64 = types.float64
    i64 = types.int64
    u64 = types.uint64
    u8 = types.uint8

    # Dynamic kernel from ExternalField
    _ext = ExternalField(
        lambda F, x, y, z, t, amp, L: F + amp, 'Ez', 1.0, 1.0
    )
    del _ext

    dep_rho_2m = (
        f1, f1, f1, f1, f64, f64, f64, i64, f64, f64, i64,
        c2, c2, i64_1, i64_1, f1, f1,
    )
    dep_j_2m = (
        f1, f1, f1, f1, f64, f1, f1, f1, f1, f64, f64, i64, f64, f64, i64,
        c2, c2, c2, c2, c2, c2, i64_1, i64_1, f1, f1,
    )
    dep_rho_1m = (
        f1, f1, f1, f1, f64, f64, f64, i64, f64, f64, i64,
        c2, i64, i64_1, i64_1, f1,
    )
    dep_j_1m = (
        f1, f1, f1, f1, f64, f1, f1, f1, f1, f64, f64, i64, f64, f64, i64,
        c2, c2, c2, i64, i64_1, i64_1, f1,
    )
    gather_2m = (
        f1, f1, f1, f64, f64, f64, i64, f64, f64, i64,
        c2, c2, c2, c2, c2, c2, c2, c2, c2, c2, c2, c2,
        f1, f1, f1, f1, f1, f1,
    )
    gather_1m = (
        f1, f1, f1, f64, f64, f64, i64, f64, f64, i64,
        c2, c2, c2, c2, c2, c2, i64,
        f1, f1, f1, f1, f1, f1,
    )

    m_bnd = 'fbpic.boundaries.cuda_methods'
    m_mv = 'fbpic.boundaries.moving_window'
    m_pb = 'fbpic.boundaries.particle_buffer_handling'
    m_pml = 'fbpic.boundaries.pml_damping'
    m_fld = 'fbpic.fields.cuda_methods'
    m_st = 'fbpic.fields.spectral_transform.cuda_methods'
    m_ext = 'fbpic.lpa_utils.external_fields'
    m_bfd = 'fbpic.openpmd_diag.boosted_field_diag'
    m_od = 'fbpic.openpmd_diag.cuda_methods'
    m_cmp = 'fbpic.particles.elementary_process.compton.cuda_methods'
    m_epu = 'fbpic.particles.elementary_process.cuda_numba_utils'
    m_ion = 'fbpic.particles.elementary_process.ionization.cuda_methods'
    m_gth = 'fbpic.particles.gathering.cuda_methods'
    m_gth1 = 'fbpic.particles.gathering.cuda_methods_one_mode'
    m_psh = 'fbpic.particles.push.cuda_methods'
    m_spn = 'fbpic.particles.spin.cuda_methods'
    m_trk = 'fbpic.particles.tracking.tracking'
    m_srt = 'fbpic.particles.utilities.cuda_sorting'

    sigs = {
        # boundaries/cuda_methods.py
        f'{m_bnd}:copy_vec_to_gpu_buffer': [
            ('default', (c3, c3, c2, c2, c2, i64, u8, u8, i64, i64)),
        ],
        f'{m_bnd}:copy_pml_to_gpu_buffer': [
            ('default', (c3, c3, c2, c2, c2, c2, c2, i64, u8, u8, i64, i64)),
        ],
        f'{m_bnd}:copy_scal_to_gpu_buffer': [
            ('default', (c3, c3, c2, i64, u8, u8, i64, i64)),
        ],
        f'{m_bnd}:replace_vec_from_gpu_buffer': [
            ('default', (c3, c3, c2, c2, c2, i64, u8, u8, i64, i64)),
        ],
        f'{m_bnd}:replace_pml_from_gpu_buffer': [
            ('default', (c3, c3, c2, c2, c2, c2, c2, i64, u8, u8, i64, i64)),
        ],
        f'{m_bnd}:replace_scal_from_gpu_buffer': [
            ('default', (c3, c3, c2, i64, u8, u8, i64, i64)),
        ],
        f'{m_bnd}:add_vec_from_gpu_buffer': [
            ('default', (c3, c3, c2, c2, c2, i64, u8, u8, i64, i64)),
        ],
        f'{m_bnd}:add_scal_from_gpu_buffer': [
            ('default', (c3, c3, c2, i64, u8, u8, i64, i64)),
        ],
        f'{m_bnd}:cuda_damp_EB_left': [
            ('default', (c2, c2, c2, c2, c2, c2, f1, i64)),
        ],
        f'{m_bnd}:cuda_damp_EB_left_pml': [
            ('default', (c2, c2, c2, c2, f1, i64)),
        ],
        f'{m_bnd}:cuda_damp_EB_right': [
            ('default', (c2, c2, c2, c2, c2, c2, f1, i64)),
        ],
        f'{m_bnd}:cuda_damp_EB_right_pml': [
            ('default', (c2, c2, c2, c2, f1, i64)),
        ],

        # boundaries/moving_window.py
        f'{m_mv}:shift_spect_array_gpu': [
            ('default', (c2, c1, i64)),
        ],

        # boundaries/particle_buffer_handling.py
        f'{m_pb}:split_particles_to_buffers': [
            ('float64', (f1, f1, f1, f1, i64, i64)),
            ('uint64', (u64_1, u64_1, u64_1, u64_1, i64, i64)),
        ],
        f'{m_pb}:copy_particles': [
            ('float64', (i64, f1, i64, f1, i64)),
            ('uint64', (i64, u64_1, i64, u64_1, i64)),
        ],
        f'{m_pb}:shift_particles_periodic_cuda': [
            ('default', (f1, f64, f64)),
        ],

        # boundaries/pml_damping.py
        f'{m_pml}:cuda_damp_pml_EB': [
            ('default', (c2, c2, c2, c2, c2, c2, f1, i64)),
        ],

        # fields/cuda_methods.py
        f'{m_fld}:cuda_erase_scalar': [
            ('default', (c2,)),
        ],
        f'{m_fld}:cuda_erase_vector': [
            ('default', (c2, c2, c2)),
        ],
        f'{m_fld}:cuda_divide_scalar_by_volume': [
            ('default', (c2, f1)),
        ],
        f'{m_fld}:cuda_divide_vector_by_volume': [
            ('default', (c2, c2, c2, f1)),
        ],
        f'{m_fld}:cuda_correct_currents_curlfree_standard': [
            ('default', (c2, c2, c2, c2, c2, f2, f2, f2, f64, i64, i64)),
        ],
        f'{m_fld}:cuda_correct_currents_crossdeposition_standard': [
            ('default', (c2, c2, c2, c2, c2, c2, c2, f2, f2, f64, i64, i64)),
        ],
        f'{m_fld}:cuda_correct_currents_curlfree_comoving': [
            ('default', (
                c2, c2, c2, c2, c2, f2, f2, f2, c2, c2, c2, f64, i64, i64,
            )),
        ],
        f'{m_fld}:cuda_correct_currents_crossdeposition_comoving': [
            ('default', (
                c2, c2, c2, c2, c2, c2, c2, f2, f2, c2, c2, c2, f64, i64, i64,
            )),
        ],
        f'{m_fld}:cuda_push_eb_standard': [
            ('default', (
                c2, c2, c2, c2, c2, c2, c2, c2, c2, c2, c2,
                f2, f2, f2, f2, f2, f2, f2, f64, u8, i64, i64,
            )),
        ],
        f'{m_fld}:cuda_push_eb_pml_standard': [
            ('default', (c2, c2, c2, c2, c2, c2, f2, f2, f2, f2, i64, i64)),
        ],
        f'{m_fld}:cuda_push_eb_comoving': [
            ('default', (
                c2, c2, c2, c2, c2, c2, c2, c2, c2, c2, c2,
                c2, c2, c2, f2, f2, c2, c2, c2, f2, f2,
                f64, f64, u8, i64, i64,
            )),
        ],
        f'{m_fld}:cuda_push_eb_pml_comoving': [
            ('default', (
                c2, c2, c2, c2, c2, c2, f2, f2, c2, f2, f2, i64, i64,
            )),
        ],
        f'{m_fld}:cuda_push_rho': [
            ('default', (c2, c2, i64, i64)),
        ],
        f'{m_fld}:cuda_filter_scalar': [
            ('default', (c2, i64, i64, f1, f1)),
        ],
        f'{m_fld}:cuda_filter_vector': [
            ('default', (c2, c2, c2, i64, i64, f1, f1)),
        ],

        # fields/spectral_transform/cuda_methods.py
        f'{m_st}:cuda_copy_2dC_to_2dR': [
            ('default', (c2, f2)),
        ],
        f'{m_st}:cuda_copy_2dR_to_2dC': [
            ('default', (f2, c2)),
        ],
        f'{m_st}:cuda_copy_2d_to_1d': [
            ('default', (c2, c1)),
        ],
        f'{m_st}:cuda_copy_1d_to_2d': [
            ('default', (c1, c2)),
        ],
        f'{m_st}:cuda_rt_to_pm': [
            ('default', (c2, c2, c2, c2)),
        ],
        f'{m_st}:cuda_pm_to_rt': [
            ('default', (c2, c2, c2, c2)),
        ],

        # lpa_utils/external_fields.py (dynamic closure)
        f'{m_ext}:ExternalField.__init__.<locals>.external_field_kernel': [
            ('default', (f1, f1, f1, f1, f64, f64, f64)),
        ],

        # openpmd_diag/
        f'{m_bfd}:extract_slice_cuda': [
            ('default', (
                i64, i64, f64, f3, c2, c2, c2, c2, c2, c2, c2, c2, c2, c2, i64,
            )),
        ],
        f'{m_od}:extract_particles_from_gpu': [
            ('default', (i64, f1, f1, f1, f1, f1, f1, f1, f1, f2)),
        ],
        f'{m_od}:extract_array_from_gpu': [
            ('float64', (i64, f1, f1)),
            ('uint64', (i64, u64_1, u64_1)),
        ],

        # particles/deposition/
        f'{_DEP_MOD}:deposit_rho_gpu_linear': [('default', dep_rho_2m)],
        f'{_DEP_MOD}:deposit_J_gpu_linear': [('default', dep_j_2m)],
        f'{_DEP_MOD}:deposit_rho_gpu_cubic': [('default', dep_rho_2m)],
        f'{_DEP_MOD}:deposit_J_gpu_cubic': [('default', dep_j_2m)],
        f'{_DEP1_MOD}:deposit_rho_gpu_linear_one_mode': [
            ('default', dep_rho_1m),
        ],
        f'{_DEP1_MOD}:deposit_J_gpu_linear_one_mode': [
            ('default', dep_j_1m),
        ],
        f'{_DEP1_MOD}:deposit_rho_gpu_cubic_one_mode': [
            ('default', dep_rho_1m),
        ],
        f'{_DEP1_MOD}:deposit_J_gpu_cubic_one_mode': [
            ('default', dep_j_1m),
        ],
        f'{_DEPU_MOD}:deposit_rho_gpu_unsorted': [
            ('default', (
                f1, f1, f1, f1, f64, f64, f64, i64, f64, f64, i64, c2, i64, f1,
            )),
        ],
        f'{_DEPU_MOD}:deposit_J_gpu_unsorted': [
            ('default', (
                f1, f1, f1, f1, f64, f1, f1, f1, f64, f64, i64, f64, f64, i64,
                c2, c2, c2, i64, f1,
            )),
        ],

        # particles/elementary_process/
        f'{m_cmp}:get_photon_density_gaussian_cuda': [
            ('default', (
                f1, i64, f1, f1, f1, f64, f64, f64, f64, f64, f64, f64,
            )),
        ],
        f'{m_cmp}:determine_scatterings_cuda': [
            ('default', (
                i64, i64, i64, i64_1, i64_1, u64_2, f64,
                f1, f1, f1, f1, f64, f1, f64, f64, f64, f64,
            )),
        ],
        f'{m_cmp}:scatter_photons_electrons_cuda': [
            ('default', (
                i64, i64, i64, i64, i64_1, i64_1, u64_2,
                f64, f64, f64, f64,
                f1, f1, f1, f1, f1, f1, f1, f1,
                f1, f1, f1, f1, f1, f1, f1, f1, f64,
            )),
        ],
        f'{m_epu}:copy_particle_data_cuda': [
            ('default', (i64, f1, f1)),
        ],
        f'{m_ion}:ionize_ions_cuda': [
            ('default', (
                i64, i64, i64, i64, i64, i64, i64_2, i16_1, u64_1, f32_1,
                f1, f1, f1, f1, f1, f1, f1, f1, f1, f1, f1, f1, f1, f1,
            )),
        ],
        f'{m_ion}:copy_ionized_electrons_cuda': [
            ('default', (
                i64, i64, i64, i64, i64_2, i16_1, i64, u8,
                f1, f1, f1, f1, f1, f1, f1, f1, f1, f1, f1, f1, f1, f1,
                f1, f1, f1, f1, f1, f1, f1, f1, f1, f1, f1, f1, f1, f1,
            )),
        ],
        f'{_SYNC_MOD}:gather_synchrotron_cuda': [
            ('default', (
                i64, i64, i64,
                f1, f1, f1, f1, f1, f1, f1, f1, f1, f1, f1,
                f64, f64, f64, u8, f1, f64, f1,
                f64, f64, f64, f64, f64, f64,
                f2, u64_2, f3,
            )),
        ],

        # particles/gathering/
        f'{m_gth}:gather_field_gpu_linear': [('default', gather_2m)],
        f'{m_gth}:gather_field_gpu_cubic': [('default', gather_2m)],
        f'{m_gth1}:erase_eb_cuda': [
            ('default', (f1, f1, f1, f1, f1, f1, i64)),
        ],
        f'{m_gth1}:gather_field_gpu_linear_one_mode': [
            ('default', gather_1m),
        ],
        f'{m_gth1}:gather_field_gpu_cubic_one_mode': [
            ('default', gather_1m),
        ],

        # particles/push/
        f'{m_psh}:push_x_gpu': [
            ('default', (f1, f1, f1, f1, f1, f1, f1, f64, f64, f64, f64)),
        ],
        f'{m_psh}:push_p_gpu': [
            ('default', (
                f1, f1, f1, f1, f1, f1, f1, f1, f1, f1, f64, f64, i64, f64,
            )),
        ],
        f'{m_psh}:push_p_after_plane_gpu': [
            ('default', (
                f1, f64, f1, f1, f1, f1, f1, f1, f1, f1, f1, f1,
                f64, f64, i64, f64,
            )),
        ],
        f'{m_psh}:push_p_ioniz_gpu': [
            ('default', (
                f1, f1, f1, f1, f1, f1, f1, f1, f1, f1,
                f64, i64, f64, u64_1,
            )),
        ],

        # particles/spin/
        f'{m_spn}:push_s_gpu': [
            ('default', (
                f1, f1, f1, f1, f1, f1, f1, f1, f1, f1,
                f1, f1, f1, f1, f1, f64, f64, i64, f64, f64,
            )),
        ],
        f'{m_spn}:push_s_ioniz_gpu': [
            ('default', (
                f1, f1, f1, f1, f1, f1, f1, f1, f1, f1,
                f1, f1, f1, f1, f1, f64, i64, f64, f64, u64_1,
            )),
        ],
        f'{m_spn}:copy_ionized_electron_spin_cuda': [
            ('default', (
                i64, i64, i64, i64, u8, i64_2, i64, i16_1,
                f1, f1, f1, f1, f1, f1, f1, f1, f1,
            )),
        ],

        # particles/tracking/
        f'{m_trk}:generate_ids_gpu': [
            ('default', (u64_1, i64, i64, u64, i64)),
        ],

        # particles/utilities/
        f'{m_srt}:get_cell_idx_per_particle': [
            ('default', (
                i64_1, i64_1, f1, f1, f1, f64, f64, i64, f64, f64, i64,
            )),
        ],
        f'{m_srt}:incl_prefix_sum': [
            ('default', (i64_1, i64_1)),
        ],
        f'{m_srt}:prefill_prefix_sum': [
            ('default', (i64_1, i64_1, i64)),
        ],
        f'{m_srt}:write_sorting_buffer': [
            ('float64', (i64_1, f1, f1)),
            ('uint64', (i64_1, u64_1, u64_1)),
        ],
    }
    return sigs


def _find_llvm_tool(name):
    for root in [os.environ.get('ROCM_HOME', '/opt/rocm'), '/opt/rocm']:
        path = os.path.join(root, 'llvm', 'bin', name)
        if os.path.isfile(path):
            return path
    return f'/opt/rocm/llvm/bin/{name}'


def inspect_codeobj_wavefront_and_atomics(
    codeobj_bytes, disassemble_atomics=False,
):
    """
    Inspect HSACO ELF notes for wavefront size and optional atomic ISA.

    Parameters
    ----------
    codeobj_bytes : bytes
        Compiled AMDGPU ELF code object.

    disassemble_atomics : bool, optional
        If True, disassemble with ``llvm-objdump`` and count atomic ops.

    Returns
    -------
    wave_size : int or None
        Reported wavefront size from AMDGPU ELF notes.

    atomics : collections.Counter
        Atomic instruction counts keyed by mnemonic and scope flags.
    """
    readobj = _find_llvm_tool('llvm-readobj')
    objdump = _find_llvm_tool('llvm-objdump')
    wave_size = None
    atomics = collections.Counter()

    with tempfile.NamedTemporaryFile(suffix='.hsaco', delete=True) as tf:
        tf.write(codeobj_bytes)
        tf.flush()
        res_notes = subprocess.run(
            [readobj, '--notes', tf.name], capture_output=True, text=True
        )
        m = re.search(r'\.wavefront_size:\s+(\d+)', res_notes.stdout)
        if m:
            wave_size = int(m.group(1))
        if disassemble_atomics:
            res_asm = subprocess.run(
                [objdump, '-d', tf.name], capture_output=True, text=True
            )
            for line in res_asm.stdout.splitlines():
                if '_atomic_' in line:
                    m_op = re.search(
                        r'^\s*((?:global|flat|buffer|ds)_atomic_\S+)\s*(.*)$',
                        line.strip(),
                    )
                    if m_op:
                        op = m_op.group(1)
                        tail = m_op.group(2)
                        scopes = [
                            f'scope:{s}'
                            for s in re.findall(r'scope:([A-Z0-9_]+)', tail)
                        ]
                        flags = (
                            'glc', 'slc', 'dlc', 'nt', 'th:TH_ATOMIC_RETURN',
                        )
                        extras = [s for s in flags if s in tail]
                        label = ' '.join([op] + scopes + extras)
                        atomics[label] += 1
    return wave_size, atomics


def verify_launch_geometry_for_targets():
    """Assert RDNA wave32 and CDNA wave64 block helpers are wave-aligned."""
    configs = [
        (
            'gfx1201_rdna4',
            {
                'arch': 'gfx1201', 'family': 'rdna', 'warp_size': 32,
                'name': 'AMD Radeon AI PRO R9700', 'cu_count': 64,
            },
        ),
        (
            'gfx90a_cdna2',
            {
                'arch': 'gfx90a', 'family': 'cdna', 'warp_size': 64,
                'name': 'AMD Instinct MI250X', 'cu_count': 110,
            },
        ),
        (
            'gfx942_cdna3',
            {
                'arch': 'gfx942', 'family': 'cdna', 'warp_size': 64,
                'name': 'AMD Instinct MI300X', 'cu_count': 304,
            },
        ),
        (
            'gfx950_cdna4',
            {
                'arch': 'gfx950', 'family': 'cdna', 'warp_size': 64,
                'name': 'AMD Instinct MI350X', 'cu_count': 256,
            },
        ),
    ]
    print(
        "\n=== Launch-Geometry Alignment Audit (RDNA wave32 & CDNA wave64) ==="
    )
    patch_target = 'fbpic.utils.gpu_backend.get_device_arch_info'
    for label, info in configs:
        with patch(patch_target, return_value=info):
            w = info['warp_size']
            copy_tpb = get_copy_tpb_2d()
            spec_tpb = get_spectral_tpb_2d()
            dep_lin = get_deposit_tpb('linear')
            dep_cub = get_deposit_tpb('cubic')
            gath_lin = get_gather_tpb('linear')
            gath_cub = get_gather_tpb('cubic')
            _, blk1d = cuda_tpb_bpg_1d(10000)
            _, blk2d = cuda_tpb_bpg_2d(1024, 256)
            pairs = [
                ('copy_2d', copy_tpb),
                ('spectral_2d', spec_tpb),
                ('bpg_2d', blk2d),
            ]
            for name, val in pairs:
                tot = val[0] * val[1]
                assert tot >= w and tot % w == 0, (
                    f"{label} {name}={val} not multiple of {w}"
                )
            singles = [
                ('dep_lin', dep_lin),
                ('dep_cub', dep_cub),
                ('gath_lin', gath_lin),
                ('gath_cub', gath_cub),
                ('bpg_1d', blk1d),
            ]
            for name, val in singles:
                assert val >= w and val % w == 0, (
                    f"{label} {name}={val} not multiple of {w}"
                )
            print(
                f"  {label:14s} warp={w}: copy={copy_tpb} "
                f"spectral={spec_tpb} deposit=({dep_lin},{dep_cub}) "
                f"gather=({gath_lin},{gath_cub}) 1d={blk1d} 2d={blk2d} [OK]"
            )


def main():
    """Compile all FBPIC GPU kernels across targets and print ISA tables."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--targets', nargs='+', default=ALL_TARGETS,
        help='AMDGPU targets to compile and inspect.',
    )
    args = parser.parse_args()

    if gpu_backend != 'hip':
        raise RuntimeError(f"Expected gpu_backend='hip', got '{gpu_backend}'")

    verify_launch_geometry_for_targets()

    reg = get_registered_kernels()
    reg_map = collections.OrderedDict()
    for mod, qual, wrapper in reg:
        reg_map[f'{mod}:{qual}'] = wrapper

    sig_map = build_signature_map()

    # After ExternalField registration, update reg_map
    reg_map = collections.OrderedDict(
        (f'{m}:{q}', w) for m, q, w in get_registered_kernels()
    )

    missing_sigs = set(reg_map.keys()) - set(sig_map.keys())
    extra_sigs = set(sig_map.keys()) - set(reg_map.keys())
    if missing_sigs or extra_sigs:
        raise AssertionError(
            f"Signature map mismatch! missing={sorted(missing_sigs)}, "
            f"extra={sorted(extra_sigs)}"
        )

    total_variants = sum(len(v) for v in sig_map.values())
    print(
        f"\nRegistered kernels: {len(reg_map)} unique @compile_cupy kernels "
        f"({total_variants} polymorphic specialization signatures)"
    )

    summary = {}
    atomic_table = {t: collections.Counter() for t in args.targets}
    atomic_per_kernel = {}
    wave_sizes = {t: set() for t in args.targets}

    for target in args.targets:
        t0 = time.perf_counter()
        passed = 0
        failed = []
        total_bytes = 0
        for idx, (kname, variants) in enumerate(sig_map.items()):
            wrapper = reg_map[kname]
            is_atomic = kname in ATOMIC_KERNELS
            for vname, argtypes in variants:
                try:
                    _, codeobj = wrapper.compile_for_target(argtypes, target)
                    valid = (codeobj[:4] == b'\x7fELF' and len(codeobj) > 200)
                    assert valid, (
                        f"Invalid ELF code object for {kname}[{vname}] "
                        f"on {target}"
                    )
                    total_bytes += len(codeobj)
                    passed += 1
                    if idx == 0 or is_atomic:
                        wsize, acnt = inspect_codeobj_wavefront_and_atomics(
                            codeobj, disassemble_atomics=is_atomic
                        )
                        if wsize is not None:
                            wave_sizes[target].add(wsize)
                        if is_atomic:
                            atomic_table[target].update(acnt)
                            atomic_per_kernel[(target, kname)] = dict(acnt)
                except Exception as exc:
                    err_msg = f"{type(exc).__name__}: {exc}"
                    failed.append((f"{kname}[{vname}]", err_msg))
        dt = time.perf_counter() - t0
        summary[target] = {
            'passed': passed,
            'total': total_variants,
            'failed': failed,
            'elapsed_s': dt,
            'total_kb': total_bytes / 1024.0,
            'waves': sorted(wave_sizes[target]),
        }
        print(
            f"  [{target}] {passed}/{total_variants} compiled in {dt:.1f}s "
            f"({total_bytes / 1024.0:.1f} KiB ELF, "
            f"wave={sorted(wave_sizes[target])}, failures={len(failed)})"
        )
        for fk, err in failed:
            print(f"    FAIL {fk}: {err}")

    print("\n=== Per-Target Compilation Summary ===")
    print(
        "| Target | Role | Wavefront | Compiled / Total | "
        "ELF Size (KiB) | Time (s) | Status |"
    )
    print("|---|---|---:|---:|---:|---:|---|")
    role_map = {
        'gfx1201': 'Primary (RDNA4 R9700)',
        'gfx90a': 'Primary (CDNA2 MI210/250X)',
        'gfx942': 'Primary (CDNA3 MI300X)',
        'gfx950': 'Primary (CDNA4 MI350X)',
        'gfx908': 'Best-effort (CDNA1 MI100)',
    }
    for target in args.targets:
        s = summary[target]
        ok = (len(s['failed']) == 0)
        status = 'PASS' if ok else f"FAIL ({len(s['failed'])})"
        waves_str = ','.join(map(str, s['waves'])) or '?'
        print(
            f"| `{target}` | {role_map.get(target, target)} | "
            f"{waves_str} | {s['passed']} / {s['total']} | "
            f"{s['total_kb']:.1f} | {s['elapsed_s']:.1f} | {status} |"
        )

    print("\n=== Per-Target FP64 Atomic ISA Audit (11 atomic kernels) ===")
    for target in args.targets:
        cnt = atomic_table[target]
        tot_insns = sum(cnt.values())
        print(
            f"  [{target}] total atomic instructions across "
            f"11 kernels = {tot_insns}:"
        )
        for insn, n in sorted(cnt.items()):
            print(f"    - {insn}: {n}")
        if target in PRIMARY_TARGETS:
            assert tot_insns > 0, f"No atomic instructions found on {target}"
            for kname in sorted(ATOMIC_KERNELS):
                per_k = atomic_per_kernel.get((target, kname), {})
                assert sum(per_k.values()) > 0, (
                    f"Missing atomic ISA in {kname} on {target}"
                )

    primary_failures = sum(
        len(summary[t]['failed']) for t in args.targets if t in PRIMARY_TARGETS
    )
    if primary_failures > 0:
        sys.exit(1)
    print("\nALL PRIMARY TARGET VERIFICATIONS PASSED.")


if __name__ == '__main__':
    main()
