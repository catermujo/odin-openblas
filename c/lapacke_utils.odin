/*****************************************************************************
Copyright (c) 2014, Intel Corp.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
* Neither the name of Intel Corporation nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.
******************************************************************************
* Contents: Native C interface to LAPACK utility functions
* Author: Intel Corporation
*****************************************************************************/
package openblas_c

import "core:c"

_ :: c

foreign import lib "../../vendor/linalg/windows-x64/lib/openblas64.lib"

// LAPACKE_UTILS_H_ ::

@(default_calling_convention = "c", link_prefix = "")
foreign lib {
	/* Error handler */
	LAPACKE_xerbla :: proc(name: cstring, info: i32) ---

	/* Compare two chars (case-insensitive) */
	LAPACKE_lsame :: proc(ca: c.char, cb: c.char) -> i32 ---

	/* Functions to convert column-major to row-major 2d arrays and vice versa. */
	LAPACKE_cgb_trans :: proc(matrix_layout: c.int, m: i32, n: i32, kl: i32, ku: i32, _in: ^complex64, ldin: i32, out: ^complex64, ldout: i32) ---
	LAPACKE_cge_trans :: proc(matrix_layout: c.int, m: i32, n: i32, _in: ^complex64, ldin: i32, out: ^complex64, ldout: i32) ---
	LAPACKE_cgg_trans :: proc(matrix_layout: c.int, m: i32, n: i32, _in: ^complex64, ldin: i32, out: ^complex64, ldout: i32) ---
	LAPACKE_chb_trans :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kd: i32, _in: ^complex64, ldin: i32, out: ^complex64, ldout: i32) ---
	LAPACKE_che_trans :: proc(matrix_layout: c.int, uplo: c.char, n: i32, _in: ^complex64, ldin: i32, out: ^complex64, ldout: i32) ---
	LAPACKE_chp_trans :: proc(matrix_layout: c.int, uplo: c.char, n: i32, _in: ^complex64, out: ^complex64) ---
	LAPACKE_chs_trans :: proc(matrix_layout: c.int, n: i32, _in: ^complex64, ldin: i32, out: ^complex64, ldout: i32) ---
	LAPACKE_cpb_trans :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kd: i32, _in: ^complex64, ldin: i32, out: ^complex64, ldout: i32) ---
	LAPACKE_cpf_trans :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, n: i32, _in: ^complex64, out: ^complex64) ---
	LAPACKE_cpo_trans :: proc(matrix_layout: c.int, uplo: c.char, n: i32, _in: ^complex64, ldin: i32, out: ^complex64, ldout: i32) ---
	LAPACKE_cpp_trans :: proc(matrix_layout: c.int, uplo: c.char, n: i32, _in: ^complex64, out: ^complex64) ---
	LAPACKE_csp_trans :: proc(matrix_layout: c.int, uplo: c.char, n: i32, _in: ^complex64, out: ^complex64) ---
	LAPACKE_csy_trans :: proc(matrix_layout: c.int, uplo: c.char, n: i32, _in: ^complex64, ldin: i32, out: ^complex64, ldout: i32) ---
	LAPACKE_ctb_trans :: proc(matrix_layout: c.int, uplo: c.char, diag: c.char, n: i32, kd: i32, _in: ^complex64, ldin: i32, out: ^complex64, ldout: i32) ---
	LAPACKE_ctf_trans :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, diag: c.char, n: i32, _in: ^complex64, out: ^complex64) ---
	LAPACKE_ctp_trans :: proc(matrix_layout: c.int, uplo: c.char, diag: c.char, n: i32, _in: ^complex64, out: ^complex64) ---
	LAPACKE_ctr_trans :: proc(matrix_layout: c.int, uplo: c.char, diag: c.char, n: i32, _in: ^complex64, ldin: i32, out: ^complex64, ldout: i32) ---
	LAPACKE_ctz_trans :: proc(matrix_layout: c.int, direct: c.char, uplo: c.char, diag: c.char, m: i32, n: i32, _in: ^complex64, ldin: i32, out: ^complex64, ldout: i32) ---
	LAPACKE_dgb_trans :: proc(matrix_layout: c.int, m: i32, n: i32, kl: i32, ku: i32, _in: ^f64, ldin: i32, out: ^f64, ldout: i32) ---
	LAPACKE_dge_trans :: proc(matrix_layout: c.int, m: i32, n: i32, _in: ^f64, ldin: i32, out: ^f64, ldout: i32) ---
	LAPACKE_dgg_trans :: proc(matrix_layout: c.int, m: i32, n: i32, _in: ^f64, ldin: i32, out: ^f64, ldout: i32) ---
	LAPACKE_dhs_trans :: proc(matrix_layout: c.int, n: i32, _in: ^f64, ldin: i32, out: ^f64, ldout: i32) ---
	LAPACKE_dpb_trans :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kd: i32, _in: ^f64, ldin: i32, out: ^f64, ldout: i32) ---
	LAPACKE_dpf_trans :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, n: i32, _in: ^f64, out: ^f64) ---
	LAPACKE_dpo_trans :: proc(matrix_layout: c.int, uplo: c.char, n: i32, _in: ^f64, ldin: i32, out: ^f64, ldout: i32) ---
	LAPACKE_dpp_trans :: proc(matrix_layout: c.int, uplo: c.char, n: i32, _in: ^f64, out: ^f64) ---
	LAPACKE_dsb_trans :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kd: i32, _in: ^f64, ldin: i32, out: ^f64, ldout: i32) ---
	LAPACKE_dsp_trans :: proc(matrix_layout: c.int, uplo: c.char, n: i32, _in: ^f64, out: ^f64) ---
	LAPACKE_dsy_trans :: proc(matrix_layout: c.int, uplo: c.char, n: i32, _in: ^f64, ldin: i32, out: ^f64, ldout: i32) ---
	LAPACKE_dtb_trans :: proc(matrix_layout: c.int, uplo: c.char, diag: c.char, n: i32, kd: i32, _in: ^f64, ldin: i32, out: ^f64, ldout: i32) ---
	LAPACKE_dtf_trans :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, diag: c.char, n: i32, _in: ^f64, out: ^f64) ---
	LAPACKE_dtp_trans :: proc(matrix_layout: c.int, uplo: c.char, diag: c.char, n: i32, _in: ^f64, out: ^f64) ---
	LAPACKE_dtr_trans :: proc(matrix_layout: c.int, uplo: c.char, diag: c.char, n: i32, _in: ^f64, ldin: i32, out: ^f64, ldout: i32) ---
	LAPACKE_dtz_trans :: proc(matrix_layout: c.int, direct: c.char, uplo: c.char, diag: c.char, m: i32, n: i32, _in: ^f64, ldin: i32, out: ^f64, ldout: i32) ---
	LAPACKE_sgb_trans :: proc(matrix_layout: c.int, m: i32, n: i32, kl: i32, ku: i32, _in: ^f32, ldin: i32, out: ^f32, ldout: i32) ---
	LAPACKE_sge_trans :: proc(matrix_layout: c.int, m: i32, n: i32, _in: ^f32, ldin: i32, out: ^f32, ldout: i32) ---
	LAPACKE_sgg_trans :: proc(matrix_layout: c.int, m: i32, n: i32, _in: ^f32, ldin: i32, out: ^f32, ldout: i32) ---
	LAPACKE_shs_trans :: proc(matrix_layout: c.int, n: i32, _in: ^f32, ldin: i32, out: ^f32, ldout: i32) ---
	LAPACKE_spb_trans :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kd: i32, _in: ^f32, ldin: i32, out: ^f32, ldout: i32) ---
	LAPACKE_spf_trans :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, n: i32, _in: ^f32, out: ^f32) ---
	LAPACKE_spo_trans :: proc(matrix_layout: c.int, uplo: c.char, n: i32, _in: ^f32, ldin: i32, out: ^f32, ldout: i32) ---
	LAPACKE_spp_trans :: proc(matrix_layout: c.int, uplo: c.char, n: i32, _in: ^f32, out: ^f32) ---
	LAPACKE_ssb_trans :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kd: i32, _in: ^f32, ldin: i32, out: ^f32, ldout: i32) ---
	LAPACKE_ssp_trans :: proc(matrix_layout: c.int, uplo: c.char, n: i32, _in: ^f32, out: ^f32) ---
	LAPACKE_ssy_trans :: proc(matrix_layout: c.int, uplo: c.char, n: i32, _in: ^f32, ldin: i32, out: ^f32, ldout: i32) ---
	LAPACKE_stb_trans :: proc(matrix_layout: c.int, uplo: c.char, diag: c.char, n: i32, kd: i32, _in: ^f32, ldin: i32, out: ^f32, ldout: i32) ---
	LAPACKE_stf_trans :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, diag: c.char, n: i32, _in: ^f32, out: ^f32) ---
	LAPACKE_stp_trans :: proc(matrix_layout: c.int, uplo: c.char, diag: c.char, n: i32, _in: ^f32, out: ^f32) ---
	LAPACKE_str_trans :: proc(matrix_layout: c.int, uplo: c.char, diag: c.char, n: i32, _in: ^f32, ldin: i32, out: ^f32, ldout: i32) ---
	LAPACKE_stz_trans :: proc(matrix_layout: c.int, direct: c.char, uplo: c.char, diag: c.char, m: i32, n: i32, _in: ^f32, ldin: i32, out: ^f32, ldout: i32) ---
	LAPACKE_zgb_trans :: proc(matrix_layout: c.int, m: i32, n: i32, kl: i32, ku: i32, _in: ^complex128, ldin: i32, out: ^complex128, ldout: i32) ---
	LAPACKE_zge_trans :: proc(matrix_layout: c.int, m: i32, n: i32, _in: ^complex128, ldin: i32, out: ^complex128, ldout: i32) ---
	LAPACKE_zgg_trans :: proc(matrix_layout: c.int, m: i32, n: i32, _in: ^complex128, ldin: i32, out: ^complex128, ldout: i32) ---
	LAPACKE_zhb_trans :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kd: i32, _in: ^complex128, ldin: i32, out: ^complex128, ldout: i32) ---
	LAPACKE_zhe_trans :: proc(matrix_layout: c.int, uplo: c.char, n: i32, _in: ^complex128, ldin: i32, out: ^complex128, ldout: i32) ---
	LAPACKE_zhp_trans :: proc(matrix_layout: c.int, uplo: c.char, n: i32, _in: ^complex128, out: ^complex128) ---
	LAPACKE_zhs_trans :: proc(matrix_layout: c.int, n: i32, _in: ^complex128, ldin: i32, out: ^complex128, ldout: i32) ---
	LAPACKE_zpb_trans :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kd: i32, _in: ^complex128, ldin: i32, out: ^complex128, ldout: i32) ---
	LAPACKE_zpf_trans :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, n: i32, _in: ^complex128, out: ^complex128) ---
	LAPACKE_zpo_trans :: proc(matrix_layout: c.int, uplo: c.char, n: i32, _in: ^complex128, ldin: i32, out: ^complex128, ldout: i32) ---
	LAPACKE_zpp_trans :: proc(matrix_layout: c.int, uplo: c.char, n: i32, _in: ^complex128, out: ^complex128) ---
	LAPACKE_zsp_trans :: proc(matrix_layout: c.int, uplo: c.char, n: i32, _in: ^complex128, out: ^complex128) ---
	LAPACKE_zsy_trans :: proc(matrix_layout: c.int, uplo: c.char, n: i32, _in: ^complex128, ldin: i32, out: ^complex128, ldout: i32) ---
	LAPACKE_ztb_trans :: proc(matrix_layout: c.int, uplo: c.char, diag: c.char, n: i32, kd: i32, _in: ^complex128, ldin: i32, out: ^complex128, ldout: i32) ---
	LAPACKE_ztf_trans :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, diag: c.char, n: i32, _in: ^complex128, out: ^complex128) ---
	LAPACKE_ztp_trans :: proc(matrix_layout: c.int, uplo: c.char, diag: c.char, n: i32, _in: ^complex128, out: ^complex128) ---
	LAPACKE_ztr_trans :: proc(matrix_layout: c.int, uplo: c.char, diag: c.char, n: i32, _in: ^complex128, ldin: i32, out: ^complex128, ldout: i32) ---
	LAPACKE_ztz_trans :: proc(matrix_layout: c.int, direct: c.char, uplo: c.char, diag: c.char, m: i32, n: i32, _in: ^complex128, ldin: i32, out: ^complex128, ldout: i32) ---

	/* NaN checkers for vectors */
	LAPACKE_c_nancheck :: proc(n: i32, x: ^complex64, incx: i32) -> i32 ---
	LAPACKE_d_nancheck :: proc(n: i32, x: ^f64, incx: i32) -> i32 ---
	LAPACKE_s_nancheck :: proc(n: i32, x: ^f32, incx: i32) -> i32 ---
	LAPACKE_z_nancheck :: proc(n: i32, x: ^complex128, incx: i32) -> i32 ---

	/* NaN checkers for matrices */
	LAPACKE_cgb_nancheck :: proc(matrix_layout: c.int, m: i32, n: i32, kl: i32, ku: i32, ab: ^complex64, ldab: i32) -> i32 ---
	LAPACKE_cge_nancheck :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex64, lda: i32) -> i32 ---
	LAPACKE_cgg_nancheck :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex64, lda: i32) -> i32 ---
	LAPACKE_cgt_nancheck :: proc(n: i32, dl: ^complex64, d: ^complex64, du: ^complex64) -> i32 ---
	LAPACKE_chb_nancheck :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kd: i32, ab: ^complex64, ldab: i32) -> i32 ---
	LAPACKE_che_nancheck :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32) -> i32 ---
	LAPACKE_chp_nancheck :: proc(n: i32, ap: ^complex64) -> i32 ---
	LAPACKE_chs_nancheck :: proc(matrix_layout: c.int, n: i32, a: ^complex64, lda: i32) -> i32 ---
	LAPACKE_cpb_nancheck :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kd: i32, ab: ^complex64, ldab: i32) -> i32 ---
	LAPACKE_cpf_nancheck :: proc(n: i32, a: ^complex64) -> i32 ---
	LAPACKE_cpo_nancheck :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32) -> i32 ---
	LAPACKE_cpp_nancheck :: proc(n: i32, ap: ^complex64) -> i32 ---
	LAPACKE_cpt_nancheck :: proc(n: i32, d: ^f32, e: ^complex64) -> i32 ---
	LAPACKE_csp_nancheck :: proc(n: i32, ap: ^complex64) -> i32 ---
	LAPACKE_cst_nancheck :: proc(n: i32, d: ^complex64, e: ^complex64) -> i32 ---
	LAPACKE_csy_nancheck :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32) -> i32 ---
	LAPACKE_ctb_nancheck :: proc(matrix_layout: c.int, uplo: c.char, diag: c.char, n: i32, kd: i32, ab: ^complex64, ldab: i32) -> i32 ---
	LAPACKE_ctf_nancheck :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, diag: c.char, n: i32, a: ^complex64) -> i32 ---
	LAPACKE_ctp_nancheck :: proc(matrix_layout: c.int, uplo: c.char, diag: c.char, n: i32, ap: ^complex64) -> i32 ---
	LAPACKE_ctr_nancheck :: proc(matrix_layout: c.int, uplo: c.char, diag: c.char, n: i32, a: ^complex64, lda: i32) -> i32 ---
	LAPACKE_ctz_nancheck :: proc(matrix_layout: c.int, direct: c.char, uplo: c.char, diag: c.char, m: i32, n: i32, a: ^complex64, lda: i32) -> i32 ---
	LAPACKE_dgb_nancheck :: proc(matrix_layout: c.int, m: i32, n: i32, kl: i32, ku: i32, ab: ^f64, ldab: i32) -> i32 ---
	LAPACKE_dge_nancheck :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f64, lda: i32) -> i32 ---
	LAPACKE_dgg_nancheck :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f64, lda: i32) -> i32 ---
	LAPACKE_dgt_nancheck :: proc(n: i32, dl: ^f64, d: ^f64, du: ^f64) -> i32 ---
	LAPACKE_dhs_nancheck :: proc(matrix_layout: c.int, n: i32, a: ^f64, lda: i32) -> i32 ---
	LAPACKE_dpb_nancheck :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kd: i32, ab: ^f64, ldab: i32) -> i32 ---
	LAPACKE_dpf_nancheck :: proc(n: i32, a: ^f64) -> i32 ---
	LAPACKE_dpo_nancheck :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f64, lda: i32) -> i32 ---
	LAPACKE_dpp_nancheck :: proc(n: i32, ap: ^f64) -> i32 ---
	LAPACKE_dpt_nancheck :: proc(n: i32, d: ^f64, e: ^f64) -> i32 ---
	LAPACKE_dsb_nancheck :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kd: i32, ab: ^f64, ldab: i32) -> i32 ---
	LAPACKE_dsp_nancheck :: proc(n: i32, ap: ^f64) -> i32 ---
	LAPACKE_dst_nancheck :: proc(n: i32, d: ^f64, e: ^f64) -> i32 ---
	LAPACKE_dsy_nancheck :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f64, lda: i32) -> i32 ---
	LAPACKE_dtb_nancheck :: proc(matrix_layout: c.int, uplo: c.char, diag: c.char, n: i32, kd: i32, ab: ^f64, ldab: i32) -> i32 ---
	LAPACKE_dtf_nancheck :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, diag: c.char, n: i32, a: ^f64) -> i32 ---
	LAPACKE_dtp_nancheck :: proc(matrix_layout: c.int, uplo: c.char, diag: c.char, n: i32, ap: ^f64) -> i32 ---
	LAPACKE_dtr_nancheck :: proc(matrix_layout: c.int, uplo: c.char, diag: c.char, n: i32, a: ^f64, lda: i32) -> i32 ---
	LAPACKE_dtz_nancheck :: proc(matrix_layout: c.int, direct: c.char, uplo: c.char, diag: c.char, m: i32, n: i32, a: ^f64, lda: i32) -> i32 ---
	LAPACKE_sgb_nancheck :: proc(matrix_layout: c.int, m: i32, n: i32, kl: i32, ku: i32, ab: ^f32, ldab: i32) -> i32 ---
	LAPACKE_sge_nancheck :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f32, lda: i32) -> i32 ---
	LAPACKE_sgg_nancheck :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f32, lda: i32) -> i32 ---
	LAPACKE_sgt_nancheck :: proc(n: i32, dl: ^f32, d: ^f32, du: ^f32) -> i32 ---
	LAPACKE_shs_nancheck :: proc(matrix_layout: c.int, n: i32, a: ^f32, lda: i32) -> i32 ---
	LAPACKE_spb_nancheck :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kd: i32, ab: ^f32, ldab: i32) -> i32 ---
	LAPACKE_spf_nancheck :: proc(n: i32, a: ^f32) -> i32 ---
	LAPACKE_spo_nancheck :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f32, lda: i32) -> i32 ---
	LAPACKE_spp_nancheck :: proc(n: i32, ap: ^f32) -> i32 ---
	LAPACKE_spt_nancheck :: proc(n: i32, d: ^f32, e: ^f32) -> i32 ---
	LAPACKE_ssb_nancheck :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kd: i32, ab: ^f32, ldab: i32) -> i32 ---
	LAPACKE_ssp_nancheck :: proc(n: i32, ap: ^f32) -> i32 ---
	LAPACKE_sst_nancheck :: proc(n: i32, d: ^f32, e: ^f32) -> i32 ---
	LAPACKE_ssy_nancheck :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f32, lda: i32) -> i32 ---
	LAPACKE_stb_nancheck :: proc(matrix_layout: c.int, uplo: c.char, diag: c.char, n: i32, kd: i32, ab: ^f32, ldab: i32) -> i32 ---
	LAPACKE_stf_nancheck :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, diag: c.char, n: i32, a: ^f32) -> i32 ---
	LAPACKE_stp_nancheck :: proc(matrix_layout: c.int, uplo: c.char, diag: c.char, n: i32, ap: ^f32) -> i32 ---
	LAPACKE_str_nancheck :: proc(matrix_layout: c.int, uplo: c.char, diag: c.char, n: i32, a: ^f32, lda: i32) -> i32 ---
	LAPACKE_stz_nancheck :: proc(matrix_layout: c.int, direct: c.char, uplo: c.char, diag: c.char, m: i32, n: i32, a: ^f32, lda: i32) -> i32 ---
	LAPACKE_zgb_nancheck :: proc(matrix_layout: c.int, m: i32, n: i32, kl: i32, ku: i32, ab: ^complex128, ldab: i32) -> i32 ---
	LAPACKE_zge_nancheck :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex128, lda: i32) -> i32 ---
	LAPACKE_zgg_nancheck :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex128, lda: i32) -> i32 ---
	LAPACKE_zgt_nancheck :: proc(n: i32, dl: ^complex128, d: ^complex128, du: ^complex128) -> i32 ---
	LAPACKE_zhb_nancheck :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kd: i32, ab: ^complex128, ldab: i32) -> i32 ---
	LAPACKE_zhe_nancheck :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32) -> i32 ---
	LAPACKE_zhp_nancheck :: proc(n: i32, ap: ^complex128) -> i32 ---
	LAPACKE_zhs_nancheck :: proc(matrix_layout: c.int, n: i32, a: ^complex128, lda: i32) -> i32 ---
	LAPACKE_zpb_nancheck :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kd: i32, ab: ^complex128, ldab: i32) -> i32 ---
	LAPACKE_zpf_nancheck :: proc(n: i32, a: ^complex128) -> i32 ---
	LAPACKE_zpo_nancheck :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32) -> i32 ---
	LAPACKE_zpp_nancheck :: proc(n: i32, ap: ^complex128) -> i32 ---
	LAPACKE_zpt_nancheck :: proc(n: i32, d: ^f64, e: ^complex128) -> i32 ---
	LAPACKE_zsp_nancheck :: proc(n: i32, ap: ^complex128) -> i32 ---
	LAPACKE_zst_nancheck :: proc(n: i32, d: ^complex128, e: ^complex128) -> i32 ---
	LAPACKE_zsy_nancheck :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32) -> i32 ---
	LAPACKE_ztb_nancheck :: proc(matrix_layout: c.int, uplo: c.char, diag: c.char, n: i32, kd: i32, ab: ^complex128, ldab: i32) -> i32 ---
	LAPACKE_ztf_nancheck :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, diag: c.char, n: i32, a: ^complex128) -> i32 ---
	LAPACKE_ztp_nancheck :: proc(matrix_layout: c.int, uplo: c.char, diag: c.char, n: i32, ap: ^complex128) -> i32 ---
	LAPACKE_ztr_nancheck :: proc(matrix_layout: c.int, uplo: c.char, diag: c.char, n: i32, a: ^complex128, lda: i32) -> i32 ---
	LAPACKE_ztz_nancheck :: proc(matrix_layout: c.int, direct: c.char, uplo: c.char, diag: c.char, m: i32, n: i32, a: ^complex128, lda: i32) -> i32 ---
}
