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
* Contents: Native C interface to LAPACK
* Author: Intel Corporation
*****************************************************************************/
package openblas_c

import "core:c"

_ :: c

foreign import lib "../../vendor/linalg/windows-x64/lib/openblas64.lib"

// LAPACKE_H_ ::

LAPACK_ROW_MAJOR :: 101
LAPACK_COL_MAJOR :: 102

LAPACK_WORK_MEMORY_ERROR :: -1010
LAPACK_TRANSPOSE_MEMORY_ERROR :: -1011

LAPACK_S_SELECT2 :: proc "c" (_: ^f32, _: ^f32) -> i32
LAPACK_S_SELECT3 :: proc "c" (_: ^f32, _: ^f32, _: ^f32) -> i32

LAPACK_D_SELECT2 :: proc "c" (_: ^f64, _: ^f64) -> i32
LAPACK_D_SELECT3 :: proc "c" (_: ^f64, _: ^f64, _: ^f64) -> i32

LAPACK_C_SELECT1 :: proc "c" (_: ^complex64) -> i32
LAPACK_C_SELECT2 :: proc "c" (_: ^complex64, _: ^complex64) -> i32

LAPACK_Z_SELECT1 :: proc "c" (_: ^complex128) -> i32
LAPACK_Z_SELECT2 :: proc "c" (_: ^complex128, _: ^complex128) -> i32

@(default_calling_convention = "c", link_prefix = "")
foreign lib {
	lapack_make_complex_float :: proc(re: f32, im: f32) -> complex64 ---
	lapack_make_complex_double :: proc(re: f64, im: f64) -> complex128 ---

	/* C-LAPACK function prototypes */
	LAPACKE_sbdsdc :: proc(matrix_layout: c.int, uplo: c.char, compq: c.char, n: i32, d: ^f32, e: ^f32, u: ^f32, ldu: i32, vt: ^f32, ldvt: i32, q: ^f32, iq: ^i32) -> i32 ---
	LAPACKE_dbdsdc :: proc(matrix_layout: c.int, uplo: c.char, compq: c.char, n: i32, d: ^f64, e: ^f64, u: ^f64, ldu: i32, vt: ^f64, ldvt: i32, q: ^f64, iq: ^i32) -> i32 ---
	LAPACKE_sbdsqr :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ncvt: i32, nru: i32, ncc: i32, d: ^f32, e: ^f32, vt: ^f32, ldvt: i32, u: ^f32, ldu: i32, _c: ^f32, ldc: i32) -> i32 ---
	LAPACKE_dbdsqr :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ncvt: i32, nru: i32, ncc: i32, d: ^f64, e: ^f64, vt: ^f64, ldvt: i32, u: ^f64, ldu: i32, _c: ^f64, ldc: i32) -> i32 ---
	LAPACKE_cbdsqr :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ncvt: i32, nru: i32, ncc: i32, d: ^f32, e: ^f32, vt: ^complex64, ldvt: i32, u: ^complex64, ldu: i32, _c: ^complex64, ldc: i32) -> i32 ---
	LAPACKE_zbdsqr :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ncvt: i32, nru: i32, ncc: i32, d: ^f64, e: ^f64, vt: ^complex128, ldvt: i32, u: ^complex128, ldu: i32, _c: ^complex128, ldc: i32) -> i32 ---
	LAPACKE_sbdsvdx :: proc(matrix_layout: c.int, uplo: c.char, jobz: c.char, range: c.char, n: i32, d: ^f32, e: ^f32, vl: f32, vu: f32, il: i32, iu: i32, ns: ^i32, s: ^f32, z: ^f32, ldz: i32, superb: ^i32) -> i32 ---
	LAPACKE_dbdsvdx :: proc(matrix_layout: c.int, uplo: c.char, jobz: c.char, range: c.char, n: i32, d: ^f64, e: ^f64, vl: f64, vu: f64, il: i32, iu: i32, ns: ^i32, s: ^f64, z: ^f64, ldz: i32, superb: ^i32) -> i32 ---
	LAPACKE_sdisna :: proc(job: c.char, m: i32, n: i32, d: ^f32, sep: ^f32) -> i32 ---
	LAPACKE_ddisna :: proc(job: c.char, m: i32, n: i32, d: ^f64, sep: ^f64) -> i32 ---
	LAPACKE_sgbbrd :: proc(matrix_layout: c.int, vect: c.char, m: i32, n: i32, ncc: i32, kl: i32, ku: i32, ab: ^f32, ldab: i32, d: ^f32, e: ^f32, q: ^f32, ldq: i32, pt: ^f32, ldpt: i32, _c: ^f32, ldc: i32) -> i32 ---
	LAPACKE_dgbbrd :: proc(matrix_layout: c.int, vect: c.char, m: i32, n: i32, ncc: i32, kl: i32, ku: i32, ab: ^f64, ldab: i32, d: ^f64, e: ^f64, q: ^f64, ldq: i32, pt: ^f64, ldpt: i32, _c: ^f64, ldc: i32) -> i32 ---
	LAPACKE_cgbbrd :: proc(matrix_layout: c.int, vect: c.char, m: i32, n: i32, ncc: i32, kl: i32, ku: i32, ab: ^complex64, ldab: i32, d: ^f32, e: ^f32, q: ^complex64, ldq: i32, pt: ^complex64, ldpt: i32, _c: ^complex64, ldc: i32) -> i32 ---
	LAPACKE_zgbbrd :: proc(matrix_layout: c.int, vect: c.char, m: i32, n: i32, ncc: i32, kl: i32, ku: i32, ab: ^complex128, ldab: i32, d: ^f64, e: ^f64, q: ^complex128, ldq: i32, pt: ^complex128, ldpt: i32, _c: ^complex128, ldc: i32) -> i32 ---
	LAPACKE_sgbcon :: proc(matrix_layout: c.int, norm: c.char, n: i32, kl: i32, ku: i32, ab: ^f32, ldab: i32, ipiv: ^i32, anorm: f32, rcond: ^f32) -> i32 ---
	LAPACKE_dgbcon :: proc(matrix_layout: c.int, norm: c.char, n: i32, kl: i32, ku: i32, ab: ^f64, ldab: i32, ipiv: ^i32, anorm: f64, rcond: ^f64) -> i32 ---
	LAPACKE_cgbcon :: proc(matrix_layout: c.int, norm: c.char, n: i32, kl: i32, ku: i32, ab: ^complex64, ldab: i32, ipiv: ^i32, anorm: f32, rcond: ^f32) -> i32 ---
	LAPACKE_zgbcon :: proc(matrix_layout: c.int, norm: c.char, n: i32, kl: i32, ku: i32, ab: ^complex128, ldab: i32, ipiv: ^i32, anorm: f64, rcond: ^f64) -> i32 ---
	LAPACKE_sgbequ :: proc(matrix_layout: c.int, m: i32, n: i32, kl: i32, ku: i32, ab: ^f32, ldab: i32, r: ^f32, _c: ^f32, rowcnd: ^f32, colcnd: ^f32, amax: ^f32) -> i32 ---
	LAPACKE_dgbequ :: proc(matrix_layout: c.int, m: i32, n: i32, kl: i32, ku: i32, ab: ^f64, ldab: i32, r: ^f64, _c: ^f64, rowcnd: ^f64, colcnd: ^f64, amax: ^f64) -> i32 ---
	LAPACKE_cgbequ :: proc(matrix_layout: c.int, m: i32, n: i32, kl: i32, ku: i32, ab: ^complex64, ldab: i32, r: ^f32, _c: ^f32, rowcnd: ^f32, colcnd: ^f32, amax: ^f32) -> i32 ---
	LAPACKE_zgbequ :: proc(matrix_layout: c.int, m: i32, n: i32, kl: i32, ku: i32, ab: ^complex128, ldab: i32, r: ^f64, _c: ^f64, rowcnd: ^f64, colcnd: ^f64, amax: ^f64) -> i32 ---
	LAPACKE_sgbequb :: proc(matrix_layout: c.int, m: i32, n: i32, kl: i32, ku: i32, ab: ^f32, ldab: i32, r: ^f32, _c: ^f32, rowcnd: ^f32, colcnd: ^f32, amax: ^f32) -> i32 ---
	LAPACKE_dgbequb :: proc(matrix_layout: c.int, m: i32, n: i32, kl: i32, ku: i32, ab: ^f64, ldab: i32, r: ^f64, _c: ^f64, rowcnd: ^f64, colcnd: ^f64, amax: ^f64) -> i32 ---
	LAPACKE_cgbequb :: proc(matrix_layout: c.int, m: i32, n: i32, kl: i32, ku: i32, ab: ^complex64, ldab: i32, r: ^f32, _c: ^f32, rowcnd: ^f32, colcnd: ^f32, amax: ^f32) -> i32 ---
	LAPACKE_zgbequb :: proc(matrix_layout: c.int, m: i32, n: i32, kl: i32, ku: i32, ab: ^complex128, ldab: i32, r: ^f64, _c: ^f64, rowcnd: ^f64, colcnd: ^f64, amax: ^f64) -> i32 ---
	LAPACKE_sgbrfs :: proc(matrix_layout: c.int, trans: c.char, n: i32, kl: i32, ku: i32, nrhs: i32, ab: ^f32, ldab: i32, afb: ^f32, ldafb: i32, ipiv: ^i32, b: ^f32, ldb: i32, x: ^f32, ldx: i32, ferr: ^f32, berr: ^f32) -> i32 ---
	LAPACKE_dgbrfs :: proc(matrix_layout: c.int, trans: c.char, n: i32, kl: i32, ku: i32, nrhs: i32, ab: ^f64, ldab: i32, afb: ^f64, ldafb: i32, ipiv: ^i32, b: ^f64, ldb: i32, x: ^f64, ldx: i32, ferr: ^f64, berr: ^f64) -> i32 ---
	LAPACKE_cgbrfs :: proc(matrix_layout: c.int, trans: c.char, n: i32, kl: i32, ku: i32, nrhs: i32, ab: ^complex64, ldab: i32, afb: ^complex64, ldafb: i32, ipiv: ^i32, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, ferr: ^f32, berr: ^f32) -> i32 ---
	LAPACKE_zgbrfs :: proc(matrix_layout: c.int, trans: c.char, n: i32, kl: i32, ku: i32, nrhs: i32, ab: ^complex128, ldab: i32, afb: ^complex128, ldafb: i32, ipiv: ^i32, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, ferr: ^f64, berr: ^f64) -> i32 ---
	LAPACKE_sgbrfsx :: proc(matrix_layout: c.int, trans: c.char, equed: c.char, n: i32, kl: i32, ku: i32, nrhs: i32, ab: ^f32, ldab: i32, afb: ^f32, ldafb: i32, ipiv: ^i32, r: ^f32, _c: ^f32, b: ^f32, ldb: i32, x: ^f32, ldx: i32, rcond: ^f32, berr: ^f32, n_err_bnds: i32, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: i32, params: ^f32) -> i32 ---
	LAPACKE_dgbrfsx :: proc(matrix_layout: c.int, trans: c.char, equed: c.char, n: i32, kl: i32, ku: i32, nrhs: i32, ab: ^f64, ldab: i32, afb: ^f64, ldafb: i32, ipiv: ^i32, r: ^f64, _c: ^f64, b: ^f64, ldb: i32, x: ^f64, ldx: i32, rcond: ^f64, berr: ^f64, n_err_bnds: i32, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: i32, params: ^f64) -> i32 ---
	LAPACKE_cgbrfsx :: proc(matrix_layout: c.int, trans: c.char, equed: c.char, n: i32, kl: i32, ku: i32, nrhs: i32, ab: ^complex64, ldab: i32, afb: ^complex64, ldafb: i32, ipiv: ^i32, r: ^f32, _c: ^f32, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, rcond: ^f32, berr: ^f32, n_err_bnds: i32, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: i32, params: ^f32) -> i32 ---
	LAPACKE_zgbrfsx :: proc(matrix_layout: c.int, trans: c.char, equed: c.char, n: i32, kl: i32, ku: i32, nrhs: i32, ab: ^complex128, ldab: i32, afb: ^complex128, ldafb: i32, ipiv: ^i32, r: ^f64, _c: ^f64, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, rcond: ^f64, berr: ^f64, n_err_bnds: i32, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: i32, params: ^f64) -> i32 ---
	LAPACKE_sgbsv :: proc(matrix_layout: c.int, n: i32, kl: i32, ku: i32, nrhs: i32, ab: ^f32, ldab: i32, ipiv: ^i32, b: ^f32, ldb: i32) -> i32 ---
	LAPACKE_dgbsv :: proc(matrix_layout: c.int, n: i32, kl: i32, ku: i32, nrhs: i32, ab: ^f64, ldab: i32, ipiv: ^i32, b: ^f64, ldb: i32) -> i32 ---
	LAPACKE_cgbsv :: proc(matrix_layout: c.int, n: i32, kl: i32, ku: i32, nrhs: i32, ab: ^complex64, ldab: i32, ipiv: ^i32, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_zgbsv :: proc(matrix_layout: c.int, n: i32, kl: i32, ku: i32, nrhs: i32, ab: ^complex128, ldab: i32, ipiv: ^i32, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_sgbsvx :: proc(matrix_layout: c.int, fact: c.char, trans: c.char, n: i32, kl: i32, ku: i32, nrhs: i32, ab: ^f32, ldab: i32, afb: ^f32, ldafb: i32, ipiv: ^i32, equed: cstring, r: ^f32, _c: ^f32, b: ^f32, ldb: i32, x: ^f32, ldx: i32, rcond: ^f32, ferr: ^f32, berr: ^f32, rpivot: ^f32) -> i32 ---
	LAPACKE_dgbsvx :: proc(matrix_layout: c.int, fact: c.char, trans: c.char, n: i32, kl: i32, ku: i32, nrhs: i32, ab: ^f64, ldab: i32, afb: ^f64, ldafb: i32, ipiv: ^i32, equed: cstring, r: ^f64, _c: ^f64, b: ^f64, ldb: i32, x: ^f64, ldx: i32, rcond: ^f64, ferr: ^f64, berr: ^f64, rpivot: ^f64) -> i32 ---
	LAPACKE_cgbsvx :: proc(matrix_layout: c.int, fact: c.char, trans: c.char, n: i32, kl: i32, ku: i32, nrhs: i32, ab: ^complex64, ldab: i32, afb: ^complex64, ldafb: i32, ipiv: ^i32, equed: cstring, r: ^f32, _c: ^f32, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, rcond: ^f32, ferr: ^f32, berr: ^f32, rpivot: ^f32) -> i32 ---
	LAPACKE_zgbsvx :: proc(matrix_layout: c.int, fact: c.char, trans: c.char, n: i32, kl: i32, ku: i32, nrhs: i32, ab: ^complex128, ldab: i32, afb: ^complex128, ldafb: i32, ipiv: ^i32, equed: cstring, r: ^f64, _c: ^f64, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, rcond: ^f64, ferr: ^f64, berr: ^f64, rpivot: ^f64) -> i32 ---
	LAPACKE_sgbsvxx :: proc(matrix_layout: c.int, fact: c.char, trans: c.char, n: i32, kl: i32, ku: i32, nrhs: i32, ab: ^f32, ldab: i32, afb: ^f32, ldafb: i32, ipiv: ^i32, equed: cstring, r: ^f32, _c: ^f32, b: ^f32, ldb: i32, x: ^f32, ldx: i32, rcond: ^f32, rpvgrw: ^f32, berr: ^f32, n_err_bnds: i32, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: i32, params: ^f32) -> i32 ---
	LAPACKE_dgbsvxx :: proc(matrix_layout: c.int, fact: c.char, trans: c.char, n: i32, kl: i32, ku: i32, nrhs: i32, ab: ^f64, ldab: i32, afb: ^f64, ldafb: i32, ipiv: ^i32, equed: cstring, r: ^f64, _c: ^f64, b: ^f64, ldb: i32, x: ^f64, ldx: i32, rcond: ^f64, rpvgrw: ^f64, berr: ^f64, n_err_bnds: i32, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: i32, params: ^f64) -> i32 ---
	LAPACKE_cgbsvxx :: proc(matrix_layout: c.int, fact: c.char, trans: c.char, n: i32, kl: i32, ku: i32, nrhs: i32, ab: ^complex64, ldab: i32, afb: ^complex64, ldafb: i32, ipiv: ^i32, equed: cstring, r: ^f32, _c: ^f32, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, rcond: ^f32, rpvgrw: ^f32, berr: ^f32, n_err_bnds: i32, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: i32, params: ^f32) -> i32 ---
	LAPACKE_zgbsvxx :: proc(matrix_layout: c.int, fact: c.char, trans: c.char, n: i32, kl: i32, ku: i32, nrhs: i32, ab: ^complex128, ldab: i32, afb: ^complex128, ldafb: i32, ipiv: ^i32, equed: cstring, r: ^f64, _c: ^f64, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, rcond: ^f64, rpvgrw: ^f64, berr: ^f64, n_err_bnds: i32, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: i32, params: ^f64) -> i32 ---
	LAPACKE_sgbtrf :: proc(matrix_layout: c.int, m: i32, n: i32, kl: i32, ku: i32, ab: ^f32, ldab: i32, ipiv: ^i32) -> i32 ---
	LAPACKE_dgbtrf :: proc(matrix_layout: c.int, m: i32, n: i32, kl: i32, ku: i32, ab: ^f64, ldab: i32, ipiv: ^i32) -> i32 ---
	LAPACKE_cgbtrf :: proc(matrix_layout: c.int, m: i32, n: i32, kl: i32, ku: i32, ab: ^complex64, ldab: i32, ipiv: ^i32) -> i32 ---
	LAPACKE_zgbtrf :: proc(matrix_layout: c.int, m: i32, n: i32, kl: i32, ku: i32, ab: ^complex128, ldab: i32, ipiv: ^i32) -> i32 ---
	LAPACKE_sgbtrs :: proc(matrix_layout: c.int, trans: c.char, n: i32, kl: i32, ku: i32, nrhs: i32, ab: ^f32, ldab: i32, ipiv: ^i32, b: ^f32, ldb: i32) -> i32 ---
	LAPACKE_dgbtrs :: proc(matrix_layout: c.int, trans: c.char, n: i32, kl: i32, ku: i32, nrhs: i32, ab: ^f64, ldab: i32, ipiv: ^i32, b: ^f64, ldb: i32) -> i32 ---
	LAPACKE_cgbtrs :: proc(matrix_layout: c.int, trans: c.char, n: i32, kl: i32, ku: i32, nrhs: i32, ab: ^complex64, ldab: i32, ipiv: ^i32, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_zgbtrs :: proc(matrix_layout: c.int, trans: c.char, n: i32, kl: i32, ku: i32, nrhs: i32, ab: ^complex128, ldab: i32, ipiv: ^i32, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_sgebak :: proc(matrix_layout: c.int, job: c.char, side: c.char, n: i32, ilo: i32, ihi: i32, scale: ^f32, m: i32, v: ^f32, ldv: i32) -> i32 ---
	LAPACKE_dgebak :: proc(matrix_layout: c.int, job: c.char, side: c.char, n: i32, ilo: i32, ihi: i32, scale: ^f64, m: i32, v: ^f64, ldv: i32) -> i32 ---
	LAPACKE_cgebak :: proc(matrix_layout: c.int, job: c.char, side: c.char, n: i32, ilo: i32, ihi: i32, scale: ^f32, m: i32, v: ^complex64, ldv: i32) -> i32 ---
	LAPACKE_zgebak :: proc(matrix_layout: c.int, job: c.char, side: c.char, n: i32, ilo: i32, ihi: i32, scale: ^f64, m: i32, v: ^complex128, ldv: i32) -> i32 ---
	LAPACKE_sgebal :: proc(matrix_layout: c.int, job: c.char, n: i32, a: ^f32, lda: i32, ilo: ^i32, ihi: ^i32, scale: ^f32) -> i32 ---
	LAPACKE_dgebal :: proc(matrix_layout: c.int, job: c.char, n: i32, a: ^f64, lda: i32, ilo: ^i32, ihi: ^i32, scale: ^f64) -> i32 ---
	LAPACKE_cgebal :: proc(matrix_layout: c.int, job: c.char, n: i32, a: ^complex64, lda: i32, ilo: ^i32, ihi: ^i32, scale: ^f32) -> i32 ---
	LAPACKE_zgebal :: proc(matrix_layout: c.int, job: c.char, n: i32, a: ^complex128, lda: i32, ilo: ^i32, ihi: ^i32, scale: ^f64) -> i32 ---
	LAPACKE_sgebrd :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f32, lda: i32, d: ^f32, e: ^f32, tauq: ^f32, taup: ^f32) -> i32 ---
	LAPACKE_dgebrd :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f64, lda: i32, d: ^f64, e: ^f64, tauq: ^f64, taup: ^f64) -> i32 ---
	LAPACKE_cgebrd :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex64, lda: i32, d: ^f32, e: ^f32, tauq: ^complex64, taup: ^complex64) -> i32 ---
	LAPACKE_zgebrd :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex128, lda: i32, d: ^f64, e: ^f64, tauq: ^complex128, taup: ^complex128) -> i32 ---
	LAPACKE_sgecon :: proc(matrix_layout: c.int, norm: c.char, n: i32, a: ^f32, lda: i32, anorm: f32, rcond: ^f32) -> i32 ---
	LAPACKE_dgecon :: proc(matrix_layout: c.int, norm: c.char, n: i32, a: ^f64, lda: i32, anorm: f64, rcond: ^f64) -> i32 ---
	LAPACKE_cgecon :: proc(matrix_layout: c.int, norm: c.char, n: i32, a: ^complex64, lda: i32, anorm: f32, rcond: ^f32) -> i32 ---
	LAPACKE_zgecon :: proc(matrix_layout: c.int, norm: c.char, n: i32, a: ^complex128, lda: i32, anorm: f64, rcond: ^f64) -> i32 ---
	LAPACKE_sgeequ :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f32, lda: i32, r: ^f32, _c: ^f32, rowcnd: ^f32, colcnd: ^f32, amax: ^f32) -> i32 ---
	LAPACKE_dgeequ :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f64, lda: i32, r: ^f64, _c: ^f64, rowcnd: ^f64, colcnd: ^f64, amax: ^f64) -> i32 ---
	LAPACKE_cgeequ :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex64, lda: i32, r: ^f32, _c: ^f32, rowcnd: ^f32, colcnd: ^f32, amax: ^f32) -> i32 ---
	LAPACKE_zgeequ :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex128, lda: i32, r: ^f64, _c: ^f64, rowcnd: ^f64, colcnd: ^f64, amax: ^f64) -> i32 ---
	LAPACKE_sgeequb :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f32, lda: i32, r: ^f32, _c: ^f32, rowcnd: ^f32, colcnd: ^f32, amax: ^f32) -> i32 ---
	LAPACKE_dgeequb :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f64, lda: i32, r: ^f64, _c: ^f64, rowcnd: ^f64, colcnd: ^f64, amax: ^f64) -> i32 ---
	LAPACKE_cgeequb :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex64, lda: i32, r: ^f32, _c: ^f32, rowcnd: ^f32, colcnd: ^f32, amax: ^f32) -> i32 ---
	LAPACKE_zgeequb :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex128, lda: i32, r: ^f64, _c: ^f64, rowcnd: ^f64, colcnd: ^f64, amax: ^f64) -> i32 ---
	LAPACKE_sgees :: proc(matrix_layout: c.int, jobvs: c.char, sort: c.char, select: LAPACK_S_SELECT2, n: i32, a: ^f32, lda: i32, sdim: ^i32, wr: ^f32, wi: ^f32, vs: ^f32, ldvs: i32) -> i32 ---
	LAPACKE_dgees :: proc(matrix_layout: c.int, jobvs: c.char, sort: c.char, select: LAPACK_D_SELECT2, n: i32, a: ^f64, lda: i32, sdim: ^i32, wr: ^f64, wi: ^f64, vs: ^f64, ldvs: i32) -> i32 ---
	LAPACKE_cgees :: proc(matrix_layout: c.int, jobvs: c.char, sort: c.char, select: LAPACK_C_SELECT1, n: i32, a: ^complex64, lda: i32, sdim: ^i32, w: ^complex64, vs: ^complex64, ldvs: i32) -> i32 ---
	LAPACKE_zgees :: proc(matrix_layout: c.int, jobvs: c.char, sort: c.char, select: LAPACK_Z_SELECT1, n: i32, a: ^complex128, lda: i32, sdim: ^i32, w: ^complex128, vs: ^complex128, ldvs: i32) -> i32 ---
	LAPACKE_sgeesx :: proc(matrix_layout: c.int, jobvs: c.char, sort: c.char, select: LAPACK_S_SELECT2, sense: c.char, n: i32, a: ^f32, lda: i32, sdim: ^i32, wr: ^f32, wi: ^f32, vs: ^f32, ldvs: i32, rconde: ^f32, rcondv: ^f32) -> i32 ---
	LAPACKE_dgeesx :: proc(matrix_layout: c.int, jobvs: c.char, sort: c.char, select: LAPACK_D_SELECT2, sense: c.char, n: i32, a: ^f64, lda: i32, sdim: ^i32, wr: ^f64, wi: ^f64, vs: ^f64, ldvs: i32, rconde: ^f64, rcondv: ^f64) -> i32 ---
	LAPACKE_cgeesx :: proc(matrix_layout: c.int, jobvs: c.char, sort: c.char, select: LAPACK_C_SELECT1, sense: c.char, n: i32, a: ^complex64, lda: i32, sdim: ^i32, w: ^complex64, vs: ^complex64, ldvs: i32, rconde: ^f32, rcondv: ^f32) -> i32 ---
	LAPACKE_zgeesx :: proc(matrix_layout: c.int, jobvs: c.char, sort: c.char, select: LAPACK_Z_SELECT1, sense: c.char, n: i32, a: ^complex128, lda: i32, sdim: ^i32, w: ^complex128, vs: ^complex128, ldvs: i32, rconde: ^f64, rcondv: ^f64) -> i32 ---
	LAPACKE_sgeev :: proc(matrix_layout: c.int, jobvl: c.char, jobvr: c.char, n: i32, a: ^f32, lda: i32, wr: ^f32, wi: ^f32, vl: ^f32, ldvl: i32, vr: ^f32, ldvr: i32) -> i32 ---
	LAPACKE_dgeev :: proc(matrix_layout: c.int, jobvl: c.char, jobvr: c.char, n: i32, a: ^f64, lda: i32, wr: ^f64, wi: ^f64, vl: ^f64, ldvl: i32, vr: ^f64, ldvr: i32) -> i32 ---
	LAPACKE_cgeev :: proc(matrix_layout: c.int, jobvl: c.char, jobvr: c.char, n: i32, a: ^complex64, lda: i32, w: ^complex64, vl: ^complex64, ldvl: i32, vr: ^complex64, ldvr: i32) -> i32 ---
	LAPACKE_zgeev :: proc(matrix_layout: c.int, jobvl: c.char, jobvr: c.char, n: i32, a: ^complex128, lda: i32, w: ^complex128, vl: ^complex128, ldvl: i32, vr: ^complex128, ldvr: i32) -> i32 ---
	LAPACKE_sgeevx :: proc(matrix_layout: c.int, balanc: c.char, jobvl: c.char, jobvr: c.char, sense: c.char, n: i32, a: ^f32, lda: i32, wr: ^f32, wi: ^f32, vl: ^f32, ldvl: i32, vr: ^f32, ldvr: i32, ilo: ^i32, ihi: ^i32, scale: ^f32, abnrm: ^f32, rconde: ^f32, rcondv: ^f32) -> i32 ---
	LAPACKE_dgeevx :: proc(matrix_layout: c.int, balanc: c.char, jobvl: c.char, jobvr: c.char, sense: c.char, n: i32, a: ^f64, lda: i32, wr: ^f64, wi: ^f64, vl: ^f64, ldvl: i32, vr: ^f64, ldvr: i32, ilo: ^i32, ihi: ^i32, scale: ^f64, abnrm: ^f64, rconde: ^f64, rcondv: ^f64) -> i32 ---
	LAPACKE_cgeevx :: proc(matrix_layout: c.int, balanc: c.char, jobvl: c.char, jobvr: c.char, sense: c.char, n: i32, a: ^complex64, lda: i32, w: ^complex64, vl: ^complex64, ldvl: i32, vr: ^complex64, ldvr: i32, ilo: ^i32, ihi: ^i32, scale: ^f32, abnrm: ^f32, rconde: ^f32, rcondv: ^f32) -> i32 ---
	LAPACKE_zgeevx :: proc(matrix_layout: c.int, balanc: c.char, jobvl: c.char, jobvr: c.char, sense: c.char, n: i32, a: ^complex128, lda: i32, w: ^complex128, vl: ^complex128, ldvl: i32, vr: ^complex128, ldvr: i32, ilo: ^i32, ihi: ^i32, scale: ^f64, abnrm: ^f64, rconde: ^f64, rcondv: ^f64) -> i32 ---
	LAPACKE_sgehrd :: proc(matrix_layout: c.int, n: i32, ilo: i32, ihi: i32, a: ^f32, lda: i32, tau: ^f32) -> i32 ---
	LAPACKE_dgehrd :: proc(matrix_layout: c.int, n: i32, ilo: i32, ihi: i32, a: ^f64, lda: i32, tau: ^f64) -> i32 ---
	LAPACKE_cgehrd :: proc(matrix_layout: c.int, n: i32, ilo: i32, ihi: i32, a: ^complex64, lda: i32, tau: ^complex64) -> i32 ---
	LAPACKE_zgehrd :: proc(matrix_layout: c.int, n: i32, ilo: i32, ihi: i32, a: ^complex128, lda: i32, tau: ^complex128) -> i32 ---
	LAPACKE_sgejsv :: proc(matrix_layout: c.int, joba: c.char, jobu: c.char, jobv: c.char, jobr: c.char, jobt: c.char, jobp: c.char, m: i32, n: i32, a: ^f32, lda: i32, sva: ^f32, u: ^f32, ldu: i32, v: ^f32, ldv: i32, stat: ^f32, istat: ^i32) -> i32 ---
	LAPACKE_dgejsv :: proc(matrix_layout: c.int, joba: c.char, jobu: c.char, jobv: c.char, jobr: c.char, jobt: c.char, jobp: c.char, m: i32, n: i32, a: ^f64, lda: i32, sva: ^f64, u: ^f64, ldu: i32, v: ^f64, ldv: i32, stat: ^f64, istat: ^i32) -> i32 ---
	LAPACKE_cgejsv :: proc(matrix_layout: c.int, joba: c.char, jobu: c.char, jobv: c.char, jobr: c.char, jobt: c.char, jobp: c.char, m: i32, n: i32, a: ^complex64, lda: i32, sva: ^f32, u: ^complex64, ldu: i32, v: ^complex64, ldv: i32, stat: ^f32, istat: ^i32) -> i32 ---
	LAPACKE_zgejsv :: proc(matrix_layout: c.int, joba: c.char, jobu: c.char, jobv: c.char, jobr: c.char, jobt: c.char, jobp: c.char, m: i32, n: i32, a: ^complex128, lda: i32, sva: ^f64, u: ^complex128, ldu: i32, v: ^complex128, ldv: i32, stat: ^f64, istat: ^i32) -> i32 ---
	LAPACKE_sgelq2 :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f32, lda: i32, tau: ^f32) -> i32 ---
	LAPACKE_dgelq2 :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f64, lda: i32, tau: ^f64) -> i32 ---
	LAPACKE_cgelq2 :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex64, lda: i32, tau: ^complex64) -> i32 ---
	LAPACKE_zgelq2 :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex128, lda: i32, tau: ^complex128) -> i32 ---
	LAPACKE_sgelqf :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f32, lda: i32, tau: ^f32) -> i32 ---
	LAPACKE_dgelqf :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f64, lda: i32, tau: ^f64) -> i32 ---
	LAPACKE_cgelqf :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex64, lda: i32, tau: ^complex64) -> i32 ---
	LAPACKE_zgelqf :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex128, lda: i32, tau: ^complex128) -> i32 ---
	LAPACKE_sgels :: proc(matrix_layout: c.int, trans: c.char, m: i32, n: i32, nrhs: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32) -> i32 ---
	LAPACKE_dgels :: proc(matrix_layout: c.int, trans: c.char, m: i32, n: i32, nrhs: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32) -> i32 ---
	LAPACKE_cgels :: proc(matrix_layout: c.int, trans: c.char, m: i32, n: i32, nrhs: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_zgels :: proc(matrix_layout: c.int, trans: c.char, m: i32, n: i32, nrhs: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_sgelsd :: proc(matrix_layout: c.int, m: i32, n: i32, nrhs: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32, s: ^f32, rcond: f32, rank: ^i32) -> i32 ---
	LAPACKE_dgelsd :: proc(matrix_layout: c.int, m: i32, n: i32, nrhs: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, s: ^f64, rcond: f64, rank: ^i32) -> i32 ---
	LAPACKE_cgelsd :: proc(matrix_layout: c.int, m: i32, n: i32, nrhs: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32, s: ^f32, rcond: f32, rank: ^i32) -> i32 ---
	LAPACKE_zgelsd :: proc(matrix_layout: c.int, m: i32, n: i32, nrhs: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, s: ^f64, rcond: f64, rank: ^i32) -> i32 ---
	LAPACKE_sgelss :: proc(matrix_layout: c.int, m: i32, n: i32, nrhs: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32, s: ^f32, rcond: f32, rank: ^i32) -> i32 ---
	LAPACKE_dgelss :: proc(matrix_layout: c.int, m: i32, n: i32, nrhs: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, s: ^f64, rcond: f64, rank: ^i32) -> i32 ---
	LAPACKE_cgelss :: proc(matrix_layout: c.int, m: i32, n: i32, nrhs: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32, s: ^f32, rcond: f32, rank: ^i32) -> i32 ---
	LAPACKE_zgelss :: proc(matrix_layout: c.int, m: i32, n: i32, nrhs: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, s: ^f64, rcond: f64, rank: ^i32) -> i32 ---
	LAPACKE_sgelsy :: proc(matrix_layout: c.int, m: i32, n: i32, nrhs: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32, jpvt: ^i32, rcond: f32, rank: ^i32) -> i32 ---
	LAPACKE_dgelsy :: proc(matrix_layout: c.int, m: i32, n: i32, nrhs: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, jpvt: ^i32, rcond: f64, rank: ^i32) -> i32 ---
	LAPACKE_cgelsy :: proc(matrix_layout: c.int, m: i32, n: i32, nrhs: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32, jpvt: ^i32, rcond: f32, rank: ^i32) -> i32 ---
	LAPACKE_zgelsy :: proc(matrix_layout: c.int, m: i32, n: i32, nrhs: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, jpvt: ^i32, rcond: f64, rank: ^i32) -> i32 ---
	LAPACKE_sgeqlf :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f32, lda: i32, tau: ^f32) -> i32 ---
	LAPACKE_dgeqlf :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f64, lda: i32, tau: ^f64) -> i32 ---
	LAPACKE_cgeqlf :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex64, lda: i32, tau: ^complex64) -> i32 ---
	LAPACKE_zgeqlf :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex128, lda: i32, tau: ^complex128) -> i32 ---
	LAPACKE_sgeqp3 :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f32, lda: i32, jpvt: ^i32, tau: ^f32) -> i32 ---
	LAPACKE_dgeqp3 :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f64, lda: i32, jpvt: ^i32, tau: ^f64) -> i32 ---
	LAPACKE_cgeqp3 :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex64, lda: i32, jpvt: ^i32, tau: ^complex64) -> i32 ---
	LAPACKE_zgeqp3 :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex128, lda: i32, jpvt: ^i32, tau: ^complex128) -> i32 ---
	LAPACKE_sgeqpf :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f32, lda: i32, jpvt: ^i32, tau: ^f32) -> i32 ---
	LAPACKE_dgeqpf :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f64, lda: i32, jpvt: ^i32, tau: ^f64) -> i32 ---
	LAPACKE_cgeqpf :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex64, lda: i32, jpvt: ^i32, tau: ^complex64) -> i32 ---
	LAPACKE_zgeqpf :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex128, lda: i32, jpvt: ^i32, tau: ^complex128) -> i32 ---
	LAPACKE_sgeqr2 :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f32, lda: i32, tau: ^f32) -> i32 ---
	LAPACKE_dgeqr2 :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f64, lda: i32, tau: ^f64) -> i32 ---
	LAPACKE_cgeqr2 :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex64, lda: i32, tau: ^complex64) -> i32 ---
	LAPACKE_zgeqr2 :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex128, lda: i32, tau: ^complex128) -> i32 ---
	LAPACKE_sgeqrf :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f32, lda: i32, tau: ^f32) -> i32 ---
	LAPACKE_dgeqrf :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f64, lda: i32, tau: ^f64) -> i32 ---
	LAPACKE_cgeqrf :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex64, lda: i32, tau: ^complex64) -> i32 ---
	LAPACKE_zgeqrf :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex128, lda: i32, tau: ^complex128) -> i32 ---
	LAPACKE_sgeqrfp :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f32, lda: i32, tau: ^f32) -> i32 ---
	LAPACKE_dgeqrfp :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f64, lda: i32, tau: ^f64) -> i32 ---
	LAPACKE_cgeqrfp :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex64, lda: i32, tau: ^complex64) -> i32 ---
	LAPACKE_zgeqrfp :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex128, lda: i32, tau: ^complex128) -> i32 ---
	LAPACKE_sgerfs :: proc(matrix_layout: c.int, trans: c.char, n: i32, nrhs: i32, a: ^f32, lda: i32, af: ^f32, ldaf: i32, ipiv: ^i32, b: ^f32, ldb: i32, x: ^f32, ldx: i32, ferr: ^f32, berr: ^f32) -> i32 ---
	LAPACKE_dgerfs :: proc(matrix_layout: c.int, trans: c.char, n: i32, nrhs: i32, a: ^f64, lda: i32, af: ^f64, ldaf: i32, ipiv: ^i32, b: ^f64, ldb: i32, x: ^f64, ldx: i32, ferr: ^f64, berr: ^f64) -> i32 ---
	LAPACKE_cgerfs :: proc(matrix_layout: c.int, trans: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, af: ^complex64, ldaf: i32, ipiv: ^i32, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, ferr: ^f32, berr: ^f32) -> i32 ---
	LAPACKE_zgerfs :: proc(matrix_layout: c.int, trans: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, af: ^complex128, ldaf: i32, ipiv: ^i32, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, ferr: ^f64, berr: ^f64) -> i32 ---
	LAPACKE_sgerfsx :: proc(matrix_layout: c.int, trans: c.char, equed: c.char, n: i32, nrhs: i32, a: ^f32, lda: i32, af: ^f32, ldaf: i32, ipiv: ^i32, r: ^f32, _c: ^f32, b: ^f32, ldb: i32, x: ^f32, ldx: i32, rcond: ^f32, berr: ^f32, n_err_bnds: i32, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: i32, params: ^f32) -> i32 ---
	LAPACKE_dgerfsx :: proc(matrix_layout: c.int, trans: c.char, equed: c.char, n: i32, nrhs: i32, a: ^f64, lda: i32, af: ^f64, ldaf: i32, ipiv: ^i32, r: ^f64, _c: ^f64, b: ^f64, ldb: i32, x: ^f64, ldx: i32, rcond: ^f64, berr: ^f64, n_err_bnds: i32, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: i32, params: ^f64) -> i32 ---
	LAPACKE_cgerfsx :: proc(matrix_layout: c.int, trans: c.char, equed: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, af: ^complex64, ldaf: i32, ipiv: ^i32, r: ^f32, _c: ^f32, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, rcond: ^f32, berr: ^f32, n_err_bnds: i32, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: i32, params: ^f32) -> i32 ---
	LAPACKE_zgerfsx :: proc(matrix_layout: c.int, trans: c.char, equed: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, af: ^complex128, ldaf: i32, ipiv: ^i32, r: ^f64, _c: ^f64, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, rcond: ^f64, berr: ^f64, n_err_bnds: i32, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: i32, params: ^f64) -> i32 ---
	LAPACKE_sgerqf :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f32, lda: i32, tau: ^f32) -> i32 ---
	LAPACKE_dgerqf :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f64, lda: i32, tau: ^f64) -> i32 ---
	LAPACKE_cgerqf :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex64, lda: i32, tau: ^complex64) -> i32 ---
	LAPACKE_zgerqf :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex128, lda: i32, tau: ^complex128) -> i32 ---
	LAPACKE_sgesdd :: proc(matrix_layout: c.int, jobz: c.char, m: i32, n: i32, a: ^f32, lda: i32, s: ^f32, u: ^f32, ldu: i32, vt: ^f32, ldvt: i32) -> i32 ---
	LAPACKE_dgesdd :: proc(matrix_layout: c.int, jobz: c.char, m: i32, n: i32, a: ^f64, lda: i32, s: ^f64, u: ^f64, ldu: i32, vt: ^f64, ldvt: i32) -> i32 ---
	LAPACKE_cgesdd :: proc(matrix_layout: c.int, jobz: c.char, m: i32, n: i32, a: ^complex64, lda: i32, s: ^f32, u: ^complex64, ldu: i32, vt: ^complex64, ldvt: i32) -> i32 ---
	LAPACKE_zgesdd :: proc(matrix_layout: c.int, jobz: c.char, m: i32, n: i32, a: ^complex128, lda: i32, s: ^f64, u: ^complex128, ldu: i32, vt: ^complex128, ldvt: i32) -> i32 ---
	LAPACKE_sgesv :: proc(matrix_layout: c.int, n: i32, nrhs: i32, a: ^f32, lda: i32, ipiv: ^i32, b: ^f32, ldb: i32) -> i32 ---
	LAPACKE_dgesv :: proc(matrix_layout: c.int, n: i32, nrhs: i32, a: ^f64, lda: i32, ipiv: ^i32, b: ^f64, ldb: i32) -> i32 ---
	LAPACKE_cgesv :: proc(matrix_layout: c.int, n: i32, nrhs: i32, a: ^complex64, lda: i32, ipiv: ^i32, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_zgesv :: proc(matrix_layout: c.int, n: i32, nrhs: i32, a: ^complex128, lda: i32, ipiv: ^i32, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_dsgesv :: proc(matrix_layout: c.int, n: i32, nrhs: i32, a: ^f64, lda: i32, ipiv: ^i32, b: ^f64, ldb: i32, x: ^f64, ldx: i32, iter: ^i32) -> i32 ---
	LAPACKE_zcgesv :: proc(matrix_layout: c.int, n: i32, nrhs: i32, a: ^complex128, lda: i32, ipiv: ^i32, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, iter: ^i32) -> i32 ---
	LAPACKE_sgesvd :: proc(matrix_layout: c.int, jobu: c.char, jobvt: c.char, m: i32, n: i32, a: ^f32, lda: i32, s: ^f32, u: ^f32, ldu: i32, vt: ^f32, ldvt: i32, superb: ^f32) -> i32 ---
	LAPACKE_dgesvd :: proc(matrix_layout: c.int, jobu: c.char, jobvt: c.char, m: i32, n: i32, a: ^f64, lda: i32, s: ^f64, u: ^f64, ldu: i32, vt: ^f64, ldvt: i32, superb: ^f64) -> i32 ---
	LAPACKE_cgesvd :: proc(matrix_layout: c.int, jobu: c.char, jobvt: c.char, m: i32, n: i32, a: ^complex64, lda: i32, s: ^f32, u: ^complex64, ldu: i32, vt: ^complex64, ldvt: i32, superb: ^f32) -> i32 ---
	LAPACKE_zgesvd :: proc(matrix_layout: c.int, jobu: c.char, jobvt: c.char, m: i32, n: i32, a: ^complex128, lda: i32, s: ^f64, u: ^complex128, ldu: i32, vt: ^complex128, ldvt: i32, superb: ^f64) -> i32 ---
	LAPACKE_sgesvdx :: proc(matrix_layout: c.int, jobu: c.char, jobvt: c.char, range: c.char, m: i32, n: i32, a: ^f32, lda: i32, vl: f32, vu: f32, il: i32, iu: i32, ns: ^i32, s: ^f32, u: ^f32, ldu: i32, vt: ^f32, ldvt: i32, superb: ^i32) -> i32 ---
	LAPACKE_dgesvdx :: proc(matrix_layout: c.int, jobu: c.char, jobvt: c.char, range: c.char, m: i32, n: i32, a: ^f64, lda: i32, vl: f64, vu: f64, il: i32, iu: i32, ns: ^i32, s: ^f64, u: ^f64, ldu: i32, vt: ^f64, ldvt: i32, superb: ^i32) -> i32 ---
	LAPACKE_cgesvdx :: proc(matrix_layout: c.int, jobu: c.char, jobvt: c.char, range: c.char, m: i32, n: i32, a: ^complex64, lda: i32, vl: f32, vu: f32, il: i32, iu: i32, ns: ^i32, s: ^f32, u: ^complex64, ldu: i32, vt: ^complex64, ldvt: i32, superb: ^i32) -> i32 ---
	LAPACKE_zgesvdx :: proc(matrix_layout: c.int, jobu: c.char, jobvt: c.char, range: c.char, m: i32, n: i32, a: ^complex128, lda: i32, vl: f64, vu: f64, il: i32, iu: i32, ns: ^i32, s: ^f64, u: ^complex128, ldu: i32, vt: ^complex128, ldvt: i32, superb: ^i32) -> i32 ---
	LAPACKE_sgesvdq :: proc(matrix_layout: c.int, joba: c.char, jobp: c.char, jobr: c.char, jobu: c.char, jobv: c.char, m: i32, n: i32, a: ^f32, lda: i32, s: ^f32, u: ^f32, ldu: i32, v: ^f32, ldv: i32, numrank: ^i32) -> i32 ---
	LAPACKE_dgesvdq :: proc(matrix_layout: c.int, joba: c.char, jobp: c.char, jobr: c.char, jobu: c.char, jobv: c.char, m: i32, n: i32, a: ^f64, lda: i32, s: ^f64, u: ^f64, ldu: i32, v: ^f64, ldv: i32, numrank: ^i32) -> i32 ---
	LAPACKE_cgesvdq :: proc(matrix_layout: c.int, joba: c.char, jobp: c.char, jobr: c.char, jobu: c.char, jobv: c.char, m: i32, n: i32, a: ^complex64, lda: i32, s: ^f32, u: ^complex64, ldu: i32, v: ^complex64, ldv: i32, numrank: ^i32) -> i32 ---
	LAPACKE_zgesvdq :: proc(matrix_layout: c.int, joba: c.char, jobp: c.char, jobr: c.char, jobu: c.char, jobv: c.char, m: i32, n: i32, a: ^complex128, lda: i32, s: ^f64, u: ^complex128, ldu: i32, v: ^complex128, ldv: i32, numrank: ^i32) -> i32 ---
	LAPACKE_sgesvj :: proc(matrix_layout: c.int, joba: c.char, jobu: c.char, jobv: c.char, m: i32, n: i32, a: ^f32, lda: i32, sva: ^f32, mv: i32, v: ^f32, ldv: i32, stat: ^f32) -> i32 ---
	LAPACKE_dgesvj :: proc(matrix_layout: c.int, joba: c.char, jobu: c.char, jobv: c.char, m: i32, n: i32, a: ^f64, lda: i32, sva: ^f64, mv: i32, v: ^f64, ldv: i32, stat: ^f64) -> i32 ---
	LAPACKE_cgesvj :: proc(matrix_layout: c.int, joba: c.char, jobu: c.char, jobv: c.char, m: i32, n: i32, a: ^complex64, lda: i32, sva: ^f32, mv: i32, v: ^complex64, ldv: i32, stat: ^f32) -> i32 ---
	LAPACKE_zgesvj :: proc(matrix_layout: c.int, joba: c.char, jobu: c.char, jobv: c.char, m: i32, n: i32, a: ^complex128, lda: i32, sva: ^f64, mv: i32, v: ^complex128, ldv: i32, stat: ^f64) -> i32 ---
	LAPACKE_sgesvx :: proc(matrix_layout: c.int, fact: c.char, trans: c.char, n: i32, nrhs: i32, a: ^f32, lda: i32, af: ^f32, ldaf: i32, ipiv: ^i32, equed: cstring, r: ^f32, _c: ^f32, b: ^f32, ldb: i32, x: ^f32, ldx: i32, rcond: ^f32, ferr: ^f32, berr: ^f32, rpivot: ^f32) -> i32 ---
	LAPACKE_dgesvx :: proc(matrix_layout: c.int, fact: c.char, trans: c.char, n: i32, nrhs: i32, a: ^f64, lda: i32, af: ^f64, ldaf: i32, ipiv: ^i32, equed: cstring, r: ^f64, _c: ^f64, b: ^f64, ldb: i32, x: ^f64, ldx: i32, rcond: ^f64, ferr: ^f64, berr: ^f64, rpivot: ^f64) -> i32 ---
	LAPACKE_cgesvx :: proc(matrix_layout: c.int, fact: c.char, trans: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, af: ^complex64, ldaf: i32, ipiv: ^i32, equed: cstring, r: ^f32, _c: ^f32, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, rcond: ^f32, ferr: ^f32, berr: ^f32, rpivot: ^f32) -> i32 ---
	LAPACKE_zgesvx :: proc(matrix_layout: c.int, fact: c.char, trans: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, af: ^complex128, ldaf: i32, ipiv: ^i32, equed: cstring, r: ^f64, _c: ^f64, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, rcond: ^f64, ferr: ^f64, berr: ^f64, rpivot: ^f64) -> i32 ---
	LAPACKE_sgesvxx :: proc(matrix_layout: c.int, fact: c.char, trans: c.char, n: i32, nrhs: i32, a: ^f32, lda: i32, af: ^f32, ldaf: i32, ipiv: ^i32, equed: cstring, r: ^f32, _c: ^f32, b: ^f32, ldb: i32, x: ^f32, ldx: i32, rcond: ^f32, rpvgrw: ^f32, berr: ^f32, n_err_bnds: i32, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: i32, params: ^f32) -> i32 ---
	LAPACKE_dgesvxx :: proc(matrix_layout: c.int, fact: c.char, trans: c.char, n: i32, nrhs: i32, a: ^f64, lda: i32, af: ^f64, ldaf: i32, ipiv: ^i32, equed: cstring, r: ^f64, _c: ^f64, b: ^f64, ldb: i32, x: ^f64, ldx: i32, rcond: ^f64, rpvgrw: ^f64, berr: ^f64, n_err_bnds: i32, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: i32, params: ^f64) -> i32 ---
	LAPACKE_cgesvxx :: proc(matrix_layout: c.int, fact: c.char, trans: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, af: ^complex64, ldaf: i32, ipiv: ^i32, equed: cstring, r: ^f32, _c: ^f32, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, rcond: ^f32, rpvgrw: ^f32, berr: ^f32, n_err_bnds: i32, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: i32, params: ^f32) -> i32 ---
	LAPACKE_zgesvxx :: proc(matrix_layout: c.int, fact: c.char, trans: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, af: ^complex128, ldaf: i32, ipiv: ^i32, equed: cstring, r: ^f64, _c: ^f64, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, rcond: ^f64, rpvgrw: ^f64, berr: ^f64, n_err_bnds: i32, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: i32, params: ^f64) -> i32 ---
	LAPACKE_sgetf2 :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f32, lda: i32, ipiv: ^i32) -> i32 ---
	LAPACKE_dgetf2 :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f64, lda: i32, ipiv: ^i32) -> i32 ---
	LAPACKE_cgetf2 :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex64, lda: i32, ipiv: ^i32) -> i32 ---
	LAPACKE_zgetf2 :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex128, lda: i32, ipiv: ^i32) -> i32 ---
	LAPACKE_sgetrf :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f32, lda: i32, ipiv: ^i32) -> i32 ---
	LAPACKE_dgetrf :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f64, lda: i32, ipiv: ^i32) -> i32 ---
	LAPACKE_cgetrf :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex64, lda: i32, ipiv: ^i32) -> i32 ---
	LAPACKE_zgetrf :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex128, lda: i32, ipiv: ^i32) -> i32 ---
	LAPACKE_sgetrf2 :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f32, lda: i32, ipiv: ^i32) -> i32 ---
	LAPACKE_dgetrf2 :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f64, lda: i32, ipiv: ^i32) -> i32 ---
	LAPACKE_cgetrf2 :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex64, lda: i32, ipiv: ^i32) -> i32 ---
	LAPACKE_zgetrf2 :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex128, lda: i32, ipiv: ^i32) -> i32 ---
	LAPACKE_sgetri :: proc(matrix_layout: c.int, n: i32, a: ^f32, lda: i32, ipiv: ^i32) -> i32 ---
	LAPACKE_dgetri :: proc(matrix_layout: c.int, n: i32, a: ^f64, lda: i32, ipiv: ^i32) -> i32 ---
	LAPACKE_cgetri :: proc(matrix_layout: c.int, n: i32, a: ^complex64, lda: i32, ipiv: ^i32) -> i32 ---
	LAPACKE_zgetri :: proc(matrix_layout: c.int, n: i32, a: ^complex128, lda: i32, ipiv: ^i32) -> i32 ---
	LAPACKE_sgetrs :: proc(matrix_layout: c.int, trans: c.char, n: i32, nrhs: i32, a: ^f32, lda: i32, ipiv: ^i32, b: ^f32, ldb: i32) -> i32 ---
	LAPACKE_dgetrs :: proc(matrix_layout: c.int, trans: c.char, n: i32, nrhs: i32, a: ^f64, lda: i32, ipiv: ^i32, b: ^f64, ldb: i32) -> i32 ---
	LAPACKE_cgetrs :: proc(matrix_layout: c.int, trans: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, ipiv: ^i32, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_zgetrs :: proc(matrix_layout: c.int, trans: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, ipiv: ^i32, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_sggbak :: proc(matrix_layout: c.int, job: c.char, side: c.char, n: i32, ilo: i32, ihi: i32, lscale: ^f32, rscale: ^f32, m: i32, v: ^f32, ldv: i32) -> i32 ---
	LAPACKE_dggbak :: proc(matrix_layout: c.int, job: c.char, side: c.char, n: i32, ilo: i32, ihi: i32, lscale: ^f64, rscale: ^f64, m: i32, v: ^f64, ldv: i32) -> i32 ---
	LAPACKE_cggbak :: proc(matrix_layout: c.int, job: c.char, side: c.char, n: i32, ilo: i32, ihi: i32, lscale: ^f32, rscale: ^f32, m: i32, v: ^complex64, ldv: i32) -> i32 ---
	LAPACKE_zggbak :: proc(matrix_layout: c.int, job: c.char, side: c.char, n: i32, ilo: i32, ihi: i32, lscale: ^f64, rscale: ^f64, m: i32, v: ^complex128, ldv: i32) -> i32 ---
	LAPACKE_sggbal :: proc(matrix_layout: c.int, job: c.char, n: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32, ilo: ^i32, ihi: ^i32, lscale: ^f32, rscale: ^f32) -> i32 ---
	LAPACKE_dggbal :: proc(matrix_layout: c.int, job: c.char, n: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, ilo: ^i32, ihi: ^i32, lscale: ^f64, rscale: ^f64) -> i32 ---
	LAPACKE_cggbal :: proc(matrix_layout: c.int, job: c.char, n: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32, ilo: ^i32, ihi: ^i32, lscale: ^f32, rscale: ^f32) -> i32 ---
	LAPACKE_zggbal :: proc(matrix_layout: c.int, job: c.char, n: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, ilo: ^i32, ihi: ^i32, lscale: ^f64, rscale: ^f64) -> i32 ---
	LAPACKE_sgges :: proc(matrix_layout: c.int, jobvsl: c.char, jobvsr: c.char, sort: c.char, selctg: LAPACK_S_SELECT3, n: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32, sdim: ^i32, alphar: ^f32, alphai: ^f32, beta: ^f32, vsl: ^f32, ldvsl: i32, vsr: ^f32, ldvsr: i32) -> i32 ---
	LAPACKE_dgges :: proc(matrix_layout: c.int, jobvsl: c.char, jobvsr: c.char, sort: c.char, selctg: LAPACK_D_SELECT3, n: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, sdim: ^i32, alphar: ^f64, alphai: ^f64, beta: ^f64, vsl: ^f64, ldvsl: i32, vsr: ^f64, ldvsr: i32) -> i32 ---
	LAPACKE_cgges :: proc(matrix_layout: c.int, jobvsl: c.char, jobvsr: c.char, sort: c.char, selctg: LAPACK_C_SELECT2, n: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32, sdim: ^i32, alpha: ^complex64, beta: ^complex64, vsl: ^complex64, ldvsl: i32, vsr: ^complex64, ldvsr: i32) -> i32 ---
	LAPACKE_zgges :: proc(matrix_layout: c.int, jobvsl: c.char, jobvsr: c.char, sort: c.char, selctg: LAPACK_Z_SELECT2, n: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, sdim: ^i32, alpha: ^complex128, beta: ^complex128, vsl: ^complex128, ldvsl: i32, vsr: ^complex128, ldvsr: i32) -> i32 ---
	LAPACKE_sgges3 :: proc(matrix_layout: c.int, jobvsl: c.char, jobvsr: c.char, sort: c.char, selctg: LAPACK_S_SELECT3, n: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32, sdim: ^i32, alphar: ^f32, alphai: ^f32, beta: ^f32, vsl: ^f32, ldvsl: i32, vsr: ^f32, ldvsr: i32) -> i32 ---
	LAPACKE_dgges3 :: proc(matrix_layout: c.int, jobvsl: c.char, jobvsr: c.char, sort: c.char, selctg: LAPACK_D_SELECT3, n: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, sdim: ^i32, alphar: ^f64, alphai: ^f64, beta: ^f64, vsl: ^f64, ldvsl: i32, vsr: ^f64, ldvsr: i32) -> i32 ---
	LAPACKE_cgges3 :: proc(matrix_layout: c.int, jobvsl: c.char, jobvsr: c.char, sort: c.char, selctg: LAPACK_C_SELECT2, n: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32, sdim: ^i32, alpha: ^complex64, beta: ^complex64, vsl: ^complex64, ldvsl: i32, vsr: ^complex64, ldvsr: i32) -> i32 ---
	LAPACKE_zgges3 :: proc(matrix_layout: c.int, jobvsl: c.char, jobvsr: c.char, sort: c.char, selctg: LAPACK_Z_SELECT2, n: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, sdim: ^i32, alpha: ^complex128, beta: ^complex128, vsl: ^complex128, ldvsl: i32, vsr: ^complex128, ldvsr: i32) -> i32 ---
	LAPACKE_sggesx :: proc(matrix_layout: c.int, jobvsl: c.char, jobvsr: c.char, sort: c.char, selctg: LAPACK_S_SELECT3, sense: c.char, n: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32, sdim: ^i32, alphar: ^f32, alphai: ^f32, beta: ^f32, vsl: ^f32, ldvsl: i32, vsr: ^f32, ldvsr: i32, rconde: ^f32, rcondv: ^f32) -> i32 ---
	LAPACKE_dggesx :: proc(matrix_layout: c.int, jobvsl: c.char, jobvsr: c.char, sort: c.char, selctg: LAPACK_D_SELECT3, sense: c.char, n: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, sdim: ^i32, alphar: ^f64, alphai: ^f64, beta: ^f64, vsl: ^f64, ldvsl: i32, vsr: ^f64, ldvsr: i32, rconde: ^f64, rcondv: ^f64) -> i32 ---
	LAPACKE_cggesx :: proc(matrix_layout: c.int, jobvsl: c.char, jobvsr: c.char, sort: c.char, selctg: LAPACK_C_SELECT2, sense: c.char, n: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32, sdim: ^i32, alpha: ^complex64, beta: ^complex64, vsl: ^complex64, ldvsl: i32, vsr: ^complex64, ldvsr: i32, rconde: ^f32, rcondv: ^f32) -> i32 ---
	LAPACKE_zggesx :: proc(matrix_layout: c.int, jobvsl: c.char, jobvsr: c.char, sort: c.char, selctg: LAPACK_Z_SELECT2, sense: c.char, n: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, sdim: ^i32, alpha: ^complex128, beta: ^complex128, vsl: ^complex128, ldvsl: i32, vsr: ^complex128, ldvsr: i32, rconde: ^f64, rcondv: ^f64) -> i32 ---
	LAPACKE_sggev :: proc(matrix_layout: c.int, jobvl: c.char, jobvr: c.char, n: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32, alphar: ^f32, alphai: ^f32, beta: ^f32, vl: ^f32, ldvl: i32, vr: ^f32, ldvr: i32) -> i32 ---
	LAPACKE_dggev :: proc(matrix_layout: c.int, jobvl: c.char, jobvr: c.char, n: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, alphar: ^f64, alphai: ^f64, beta: ^f64, vl: ^f64, ldvl: i32, vr: ^f64, ldvr: i32) -> i32 ---
	LAPACKE_cggev :: proc(matrix_layout: c.int, jobvl: c.char, jobvr: c.char, n: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32, alpha: ^complex64, beta: ^complex64, vl: ^complex64, ldvl: i32, vr: ^complex64, ldvr: i32) -> i32 ---
	LAPACKE_zggev :: proc(matrix_layout: c.int, jobvl: c.char, jobvr: c.char, n: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, alpha: ^complex128, beta: ^complex128, vl: ^complex128, ldvl: i32, vr: ^complex128, ldvr: i32) -> i32 ---
	LAPACKE_sggev3 :: proc(matrix_layout: c.int, jobvl: c.char, jobvr: c.char, n: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32, alphar: ^f32, alphai: ^f32, beta: ^f32, vl: ^f32, ldvl: i32, vr: ^f32, ldvr: i32) -> i32 ---
	LAPACKE_dggev3 :: proc(matrix_layout: c.int, jobvl: c.char, jobvr: c.char, n: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, alphar: ^f64, alphai: ^f64, beta: ^f64, vl: ^f64, ldvl: i32, vr: ^f64, ldvr: i32) -> i32 ---
	LAPACKE_cggev3 :: proc(matrix_layout: c.int, jobvl: c.char, jobvr: c.char, n: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32, alpha: ^complex64, beta: ^complex64, vl: ^complex64, ldvl: i32, vr: ^complex64, ldvr: i32) -> i32 ---
	LAPACKE_zggev3 :: proc(matrix_layout: c.int, jobvl: c.char, jobvr: c.char, n: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, alpha: ^complex128, beta: ^complex128, vl: ^complex128, ldvl: i32, vr: ^complex128, ldvr: i32) -> i32 ---
	LAPACKE_sggevx :: proc(matrix_layout: c.int, balanc: c.char, jobvl: c.char, jobvr: c.char, sense: c.char, n: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32, alphar: ^f32, alphai: ^f32, beta: ^f32, vl: ^f32, ldvl: i32, vr: ^f32, ldvr: i32, ilo: ^i32, ihi: ^i32, lscale: ^f32, rscale: ^f32, abnrm: ^f32, bbnrm: ^f32, rconde: ^f32, rcondv: ^f32) -> i32 ---
	LAPACKE_dggevx :: proc(matrix_layout: c.int, balanc: c.char, jobvl: c.char, jobvr: c.char, sense: c.char, n: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, alphar: ^f64, alphai: ^f64, beta: ^f64, vl: ^f64, ldvl: i32, vr: ^f64, ldvr: i32, ilo: ^i32, ihi: ^i32, lscale: ^f64, rscale: ^f64, abnrm: ^f64, bbnrm: ^f64, rconde: ^f64, rcondv: ^f64) -> i32 ---
	LAPACKE_cggevx :: proc(matrix_layout: c.int, balanc: c.char, jobvl: c.char, jobvr: c.char, sense: c.char, n: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32, alpha: ^complex64, beta: ^complex64, vl: ^complex64, ldvl: i32, vr: ^complex64, ldvr: i32, ilo: ^i32, ihi: ^i32, lscale: ^f32, rscale: ^f32, abnrm: ^f32, bbnrm: ^f32, rconde: ^f32, rcondv: ^f32) -> i32 ---
	LAPACKE_zggevx :: proc(matrix_layout: c.int, balanc: c.char, jobvl: c.char, jobvr: c.char, sense: c.char, n: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, alpha: ^complex128, beta: ^complex128, vl: ^complex128, ldvl: i32, vr: ^complex128, ldvr: i32, ilo: ^i32, ihi: ^i32, lscale: ^f64, rscale: ^f64, abnrm: ^f64, bbnrm: ^f64, rconde: ^f64, rcondv: ^f64) -> i32 ---
	LAPACKE_sggglm :: proc(matrix_layout: c.int, n: i32, m: i32, p: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32, d: ^f32, x: ^f32, y: ^f32) -> i32 ---
	LAPACKE_dggglm :: proc(matrix_layout: c.int, n: i32, m: i32, p: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, d: ^f64, x: ^f64, y: ^f64) -> i32 ---
	LAPACKE_cggglm :: proc(matrix_layout: c.int, n: i32, m: i32, p: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32, d: ^complex64, x: ^complex64, y: ^complex64) -> i32 ---
	LAPACKE_zggglm :: proc(matrix_layout: c.int, n: i32, m: i32, p: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, d: ^complex128, x: ^complex128, y: ^complex128) -> i32 ---
	LAPACKE_sgghrd :: proc(matrix_layout: c.int, compq: c.char, compz: c.char, n: i32, ilo: i32, ihi: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32, q: ^f32, ldq: i32, z: ^f32, ldz: i32) -> i32 ---
	LAPACKE_dgghrd :: proc(matrix_layout: c.int, compq: c.char, compz: c.char, n: i32, ilo: i32, ihi: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, q: ^f64, ldq: i32, z: ^f64, ldz: i32) -> i32 ---
	LAPACKE_cgghrd :: proc(matrix_layout: c.int, compq: c.char, compz: c.char, n: i32, ilo: i32, ihi: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32, q: ^complex64, ldq: i32, z: ^complex64, ldz: i32) -> i32 ---
	LAPACKE_zgghrd :: proc(matrix_layout: c.int, compq: c.char, compz: c.char, n: i32, ilo: i32, ihi: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, q: ^complex128, ldq: i32, z: ^complex128, ldz: i32) -> i32 ---
	LAPACKE_sgghd3 :: proc(matrix_layout: c.int, compq: c.char, compz: c.char, n: i32, ilo: i32, ihi: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32, q: ^f32, ldq: i32, z: ^f32, ldz: i32) -> i32 ---
	LAPACKE_dgghd3 :: proc(matrix_layout: c.int, compq: c.char, compz: c.char, n: i32, ilo: i32, ihi: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, q: ^f64, ldq: i32, z: ^f64, ldz: i32) -> i32 ---
	LAPACKE_cgghd3 :: proc(matrix_layout: c.int, compq: c.char, compz: c.char, n: i32, ilo: i32, ihi: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32, q: ^complex64, ldq: i32, z: ^complex64, ldz: i32) -> i32 ---
	LAPACKE_zgghd3 :: proc(matrix_layout: c.int, compq: c.char, compz: c.char, n: i32, ilo: i32, ihi: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, q: ^complex128, ldq: i32, z: ^complex128, ldz: i32) -> i32 ---
	LAPACKE_sgglse :: proc(matrix_layout: c.int, m: i32, n: i32, p: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32, _c: ^f32, d: ^f32, x: ^f32) -> i32 ---
	LAPACKE_dgglse :: proc(matrix_layout: c.int, m: i32, n: i32, p: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, _c: ^f64, d: ^f64, x: ^f64) -> i32 ---
	LAPACKE_cgglse :: proc(matrix_layout: c.int, m: i32, n: i32, p: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32, _c: ^complex64, d: ^complex64, x: ^complex64) -> i32 ---
	LAPACKE_zgglse :: proc(matrix_layout: c.int, m: i32, n: i32, p: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, _c: ^complex128, d: ^complex128, x: ^complex128) -> i32 ---
	LAPACKE_sggqrf :: proc(matrix_layout: c.int, n: i32, m: i32, p: i32, a: ^f32, lda: i32, taua: ^f32, b: ^f32, ldb: i32, taub: ^f32) -> i32 ---
	LAPACKE_dggqrf :: proc(matrix_layout: c.int, n: i32, m: i32, p: i32, a: ^f64, lda: i32, taua: ^f64, b: ^f64, ldb: i32, taub: ^f64) -> i32 ---
	LAPACKE_cggqrf :: proc(matrix_layout: c.int, n: i32, m: i32, p: i32, a: ^complex64, lda: i32, taua: ^complex64, b: ^complex64, ldb: i32, taub: ^complex64) -> i32 ---
	LAPACKE_zggqrf :: proc(matrix_layout: c.int, n: i32, m: i32, p: i32, a: ^complex128, lda: i32, taua: ^complex128, b: ^complex128, ldb: i32, taub: ^complex128) -> i32 ---
	LAPACKE_sggrqf :: proc(matrix_layout: c.int, m: i32, p: i32, n: i32, a: ^f32, lda: i32, taua: ^f32, b: ^f32, ldb: i32, taub: ^f32) -> i32 ---
	LAPACKE_dggrqf :: proc(matrix_layout: c.int, m: i32, p: i32, n: i32, a: ^f64, lda: i32, taua: ^f64, b: ^f64, ldb: i32, taub: ^f64) -> i32 ---
	LAPACKE_cggrqf :: proc(matrix_layout: c.int, m: i32, p: i32, n: i32, a: ^complex64, lda: i32, taua: ^complex64, b: ^complex64, ldb: i32, taub: ^complex64) -> i32 ---
	LAPACKE_zggrqf :: proc(matrix_layout: c.int, m: i32, p: i32, n: i32, a: ^complex128, lda: i32, taua: ^complex128, b: ^complex128, ldb: i32, taub: ^complex128) -> i32 ---
	LAPACKE_sggsvd :: proc(matrix_layout: c.int, jobu: c.char, jobv: c.char, jobq: c.char, m: i32, n: i32, p: i32, k: ^i32, l: ^i32, a: ^f32, lda: i32, b: ^f32, ldb: i32, alpha: ^f32, beta: ^f32, u: ^f32, ldu: i32, v: ^f32, ldv: i32, q: ^f32, ldq: i32, iwork: ^i32) -> i32 ---
	LAPACKE_dggsvd :: proc(matrix_layout: c.int, jobu: c.char, jobv: c.char, jobq: c.char, m: i32, n: i32, p: i32, k: ^i32, l: ^i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, alpha: ^f64, beta: ^f64, u: ^f64, ldu: i32, v: ^f64, ldv: i32, q: ^f64, ldq: i32, iwork: ^i32) -> i32 ---
	LAPACKE_cggsvd :: proc(matrix_layout: c.int, jobu: c.char, jobv: c.char, jobq: c.char, m: i32, n: i32, p: i32, k: ^i32, l: ^i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32, alpha: ^f32, beta: ^f32, u: ^complex64, ldu: i32, v: ^complex64, ldv: i32, q: ^complex64, ldq: i32, iwork: ^i32) -> i32 ---
	LAPACKE_zggsvd :: proc(matrix_layout: c.int, jobu: c.char, jobv: c.char, jobq: c.char, m: i32, n: i32, p: i32, k: ^i32, l: ^i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, alpha: ^f64, beta: ^f64, u: ^complex128, ldu: i32, v: ^complex128, ldv: i32, q: ^complex128, ldq: i32, iwork: ^i32) -> i32 ---
	LAPACKE_sggsvd3 :: proc(matrix_layout: c.int, jobu: c.char, jobv: c.char, jobq: c.char, m: i32, n: i32, p: i32, k: ^i32, l: ^i32, a: ^f32, lda: i32, b: ^f32, ldb: i32, alpha: ^f32, beta: ^f32, u: ^f32, ldu: i32, v: ^f32, ldv: i32, q: ^f32, ldq: i32, iwork: ^i32) -> i32 ---
	LAPACKE_dggsvd3 :: proc(matrix_layout: c.int, jobu: c.char, jobv: c.char, jobq: c.char, m: i32, n: i32, p: i32, k: ^i32, l: ^i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, alpha: ^f64, beta: ^f64, u: ^f64, ldu: i32, v: ^f64, ldv: i32, q: ^f64, ldq: i32, iwork: ^i32) -> i32 ---
	LAPACKE_cggsvd3 :: proc(matrix_layout: c.int, jobu: c.char, jobv: c.char, jobq: c.char, m: i32, n: i32, p: i32, k: ^i32, l: ^i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32, alpha: ^f32, beta: ^f32, u: ^complex64, ldu: i32, v: ^complex64, ldv: i32, q: ^complex64, ldq: i32, iwork: ^i32) -> i32 ---
	LAPACKE_zggsvd3 :: proc(matrix_layout: c.int, jobu: c.char, jobv: c.char, jobq: c.char, m: i32, n: i32, p: i32, k: ^i32, l: ^i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, alpha: ^f64, beta: ^f64, u: ^complex128, ldu: i32, v: ^complex128, ldv: i32, q: ^complex128, ldq: i32, iwork: ^i32) -> i32 ---
	LAPACKE_sggsvp :: proc(matrix_layout: c.int, jobu: c.char, jobv: c.char, jobq: c.char, m: i32, p: i32, n: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32, tola: f32, tolb: f32, k: ^i32, l: ^i32, u: ^f32, ldu: i32, v: ^f32, ldv: i32, q: ^f32, ldq: i32) -> i32 ---
	LAPACKE_dggsvp :: proc(matrix_layout: c.int, jobu: c.char, jobv: c.char, jobq: c.char, m: i32, p: i32, n: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, tola: f64, tolb: f64, k: ^i32, l: ^i32, u: ^f64, ldu: i32, v: ^f64, ldv: i32, q: ^f64, ldq: i32) -> i32 ---
	LAPACKE_cggsvp :: proc(matrix_layout: c.int, jobu: c.char, jobv: c.char, jobq: c.char, m: i32, p: i32, n: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32, tola: f32, tolb: f32, k: ^i32, l: ^i32, u: ^complex64, ldu: i32, v: ^complex64, ldv: i32, q: ^complex64, ldq: i32) -> i32 ---
	LAPACKE_zggsvp :: proc(matrix_layout: c.int, jobu: c.char, jobv: c.char, jobq: c.char, m: i32, p: i32, n: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, tola: f64, tolb: f64, k: ^i32, l: ^i32, u: ^complex128, ldu: i32, v: ^complex128, ldv: i32, q: ^complex128, ldq: i32) -> i32 ---
	LAPACKE_sggsvp3 :: proc(matrix_layout: c.int, jobu: c.char, jobv: c.char, jobq: c.char, m: i32, p: i32, n: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32, tola: f32, tolb: f32, k: ^i32, l: ^i32, u: ^f32, ldu: i32, v: ^f32, ldv: i32, q: ^f32, ldq: i32) -> i32 ---
	LAPACKE_dggsvp3 :: proc(matrix_layout: c.int, jobu: c.char, jobv: c.char, jobq: c.char, m: i32, p: i32, n: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, tola: f64, tolb: f64, k: ^i32, l: ^i32, u: ^f64, ldu: i32, v: ^f64, ldv: i32, q: ^f64, ldq: i32) -> i32 ---
	LAPACKE_cggsvp3 :: proc(matrix_layout: c.int, jobu: c.char, jobv: c.char, jobq: c.char, m: i32, p: i32, n: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32, tola: f32, tolb: f32, k: ^i32, l: ^i32, u: ^complex64, ldu: i32, v: ^complex64, ldv: i32, q: ^complex64, ldq: i32) -> i32 ---
	LAPACKE_zggsvp3 :: proc(matrix_layout: c.int, jobu: c.char, jobv: c.char, jobq: c.char, m: i32, p: i32, n: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, tola: f64, tolb: f64, k: ^i32, l: ^i32, u: ^complex128, ldu: i32, v: ^complex128, ldv: i32, q: ^complex128, ldq: i32) -> i32 ---
	LAPACKE_sgtcon :: proc(norm: c.char, n: i32, dl: ^f32, d: ^f32, du: ^f32, du2: ^f32, ipiv: ^i32, anorm: f32, rcond: ^f32) -> i32 ---
	LAPACKE_dgtcon :: proc(norm: c.char, n: i32, dl: ^f64, d: ^f64, du: ^f64, du2: ^f64, ipiv: ^i32, anorm: f64, rcond: ^f64) -> i32 ---
	LAPACKE_cgtcon :: proc(norm: c.char, n: i32, dl: ^complex64, d: ^complex64, du: ^complex64, du2: ^complex64, ipiv: ^i32, anorm: f32, rcond: ^f32) -> i32 ---
	LAPACKE_zgtcon :: proc(norm: c.char, n: i32, dl: ^complex128, d: ^complex128, du: ^complex128, du2: ^complex128, ipiv: ^i32, anorm: f64, rcond: ^f64) -> i32 ---
	LAPACKE_sgtrfs :: proc(matrix_layout: c.int, trans: c.char, n: i32, nrhs: i32, dl: ^f32, d: ^f32, du: ^f32, dlf: ^f32, df: ^f32, duf: ^f32, du2: ^f32, ipiv: ^i32, b: ^f32, ldb: i32, x: ^f32, ldx: i32, ferr: ^f32, berr: ^f32) -> i32 ---
	LAPACKE_dgtrfs :: proc(matrix_layout: c.int, trans: c.char, n: i32, nrhs: i32, dl: ^f64, d: ^f64, du: ^f64, dlf: ^f64, df: ^f64, duf: ^f64, du2: ^f64, ipiv: ^i32, b: ^f64, ldb: i32, x: ^f64, ldx: i32, ferr: ^f64, berr: ^f64) -> i32 ---
	LAPACKE_cgtrfs :: proc(matrix_layout: c.int, trans: c.char, n: i32, nrhs: i32, dl: ^complex64, d: ^complex64, du: ^complex64, dlf: ^complex64, df: ^complex64, duf: ^complex64, du2: ^complex64, ipiv: ^i32, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, ferr: ^f32, berr: ^f32) -> i32 ---
	LAPACKE_zgtrfs :: proc(matrix_layout: c.int, trans: c.char, n: i32, nrhs: i32, dl: ^complex128, d: ^complex128, du: ^complex128, dlf: ^complex128, df: ^complex128, duf: ^complex128, du2: ^complex128, ipiv: ^i32, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, ferr: ^f64, berr: ^f64) -> i32 ---
	LAPACKE_sgtsv :: proc(matrix_layout: c.int, n: i32, nrhs: i32, dl: ^f32, d: ^f32, du: ^f32, b: ^f32, ldb: i32) -> i32 ---
	LAPACKE_dgtsv :: proc(matrix_layout: c.int, n: i32, nrhs: i32, dl: ^f64, d: ^f64, du: ^f64, b: ^f64, ldb: i32) -> i32 ---
	LAPACKE_cgtsv :: proc(matrix_layout: c.int, n: i32, nrhs: i32, dl: ^complex64, d: ^complex64, du: ^complex64, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_zgtsv :: proc(matrix_layout: c.int, n: i32, nrhs: i32, dl: ^complex128, d: ^complex128, du: ^complex128, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_sgtsvx :: proc(matrix_layout: c.int, fact: c.char, trans: c.char, n: i32, nrhs: i32, dl: ^f32, d: ^f32, du: ^f32, dlf: ^f32, df: ^f32, duf: ^f32, du2: ^f32, ipiv: ^i32, b: ^f32, ldb: i32, x: ^f32, ldx: i32, rcond: ^f32, ferr: ^f32, berr: ^f32) -> i32 ---
	LAPACKE_dgtsvx :: proc(matrix_layout: c.int, fact: c.char, trans: c.char, n: i32, nrhs: i32, dl: ^f64, d: ^f64, du: ^f64, dlf: ^f64, df: ^f64, duf: ^f64, du2: ^f64, ipiv: ^i32, b: ^f64, ldb: i32, x: ^f64, ldx: i32, rcond: ^f64, ferr: ^f64, berr: ^f64) -> i32 ---
	LAPACKE_cgtsvx :: proc(matrix_layout: c.int, fact: c.char, trans: c.char, n: i32, nrhs: i32, dl: ^complex64, d: ^complex64, du: ^complex64, dlf: ^complex64, df: ^complex64, duf: ^complex64, du2: ^complex64, ipiv: ^i32, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, rcond: ^f32, ferr: ^f32, berr: ^f32) -> i32 ---
	LAPACKE_zgtsvx :: proc(matrix_layout: c.int, fact: c.char, trans: c.char, n: i32, nrhs: i32, dl: ^complex128, d: ^complex128, du: ^complex128, dlf: ^complex128, df: ^complex128, duf: ^complex128, du2: ^complex128, ipiv: ^i32, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, rcond: ^f64, ferr: ^f64, berr: ^f64) -> i32 ---
	LAPACKE_sgttrf :: proc(n: i32, dl: ^f32, d: ^f32, du: ^f32, du2: ^f32, ipiv: ^i32) -> i32 ---
	LAPACKE_dgttrf :: proc(n: i32, dl: ^f64, d: ^f64, du: ^f64, du2: ^f64, ipiv: ^i32) -> i32 ---
	LAPACKE_cgttrf :: proc(n: i32, dl: ^complex64, d: ^complex64, du: ^complex64, du2: ^complex64, ipiv: ^i32) -> i32 ---
	LAPACKE_zgttrf :: proc(n: i32, dl: ^complex128, d: ^complex128, du: ^complex128, du2: ^complex128, ipiv: ^i32) -> i32 ---
	LAPACKE_sgttrs :: proc(matrix_layout: c.int, trans: c.char, n: i32, nrhs: i32, dl: ^f32, d: ^f32, du: ^f32, du2: ^f32, ipiv: ^i32, b: ^f32, ldb: i32) -> i32 ---
	LAPACKE_dgttrs :: proc(matrix_layout: c.int, trans: c.char, n: i32, nrhs: i32, dl: ^f64, d: ^f64, du: ^f64, du2: ^f64, ipiv: ^i32, b: ^f64, ldb: i32) -> i32 ---
	LAPACKE_cgttrs :: proc(matrix_layout: c.int, trans: c.char, n: i32, nrhs: i32, dl: ^complex64, d: ^complex64, du: ^complex64, du2: ^complex64, ipiv: ^i32, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_zgttrs :: proc(matrix_layout: c.int, trans: c.char, n: i32, nrhs: i32, dl: ^complex128, d: ^complex128, du: ^complex128, du2: ^complex128, ipiv: ^i32, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_chbev :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, kd: i32, ab: ^complex64, ldab: i32, w: ^f32, z: ^complex64, ldz: i32) -> i32 ---
	LAPACKE_zhbev :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, kd: i32, ab: ^complex128, ldab: i32, w: ^f64, z: ^complex128, ldz: i32) -> i32 ---
	LAPACKE_chbevd :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, kd: i32, ab: ^complex64, ldab: i32, w: ^f32, z: ^complex64, ldz: i32) -> i32 ---
	LAPACKE_zhbevd :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, kd: i32, ab: ^complex128, ldab: i32, w: ^f64, z: ^complex128, ldz: i32) -> i32 ---
	LAPACKE_chbevx :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, uplo: c.char, n: i32, kd: i32, ab: ^complex64, ldab: i32, q: ^complex64, ldq: i32, vl: f32, vu: f32, il: i32, iu: i32, abstol: f32, m: ^i32, w: ^f32, z: ^complex64, ldz: i32, ifail: ^i32) -> i32 ---
	LAPACKE_zhbevx :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, uplo: c.char, n: i32, kd: i32, ab: ^complex128, ldab: i32, q: ^complex128, ldq: i32, vl: f64, vu: f64, il: i32, iu: i32, abstol: f64, m: ^i32, w: ^f64, z: ^complex128, ldz: i32, ifail: ^i32) -> i32 ---
	LAPACKE_chbgst :: proc(matrix_layout: c.int, vect: c.char, uplo: c.char, n: i32, ka: i32, kb: i32, ab: ^complex64, ldab: i32, bb: ^complex64, ldbb: i32, x: ^complex64, ldx: i32) -> i32 ---
	LAPACKE_zhbgst :: proc(matrix_layout: c.int, vect: c.char, uplo: c.char, n: i32, ka: i32, kb: i32, ab: ^complex128, ldab: i32, bb: ^complex128, ldbb: i32, x: ^complex128, ldx: i32) -> i32 ---
	LAPACKE_chbgv :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, ka: i32, kb: i32, ab: ^complex64, ldab: i32, bb: ^complex64, ldbb: i32, w: ^f32, z: ^complex64, ldz: i32) -> i32 ---
	LAPACKE_zhbgv :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, ka: i32, kb: i32, ab: ^complex128, ldab: i32, bb: ^complex128, ldbb: i32, w: ^f64, z: ^complex128, ldz: i32) -> i32 ---
	LAPACKE_chbgvd :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, ka: i32, kb: i32, ab: ^complex64, ldab: i32, bb: ^complex64, ldbb: i32, w: ^f32, z: ^complex64, ldz: i32) -> i32 ---
	LAPACKE_zhbgvd :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, ka: i32, kb: i32, ab: ^complex128, ldab: i32, bb: ^complex128, ldbb: i32, w: ^f64, z: ^complex128, ldz: i32) -> i32 ---
	LAPACKE_chbgvx :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, uplo: c.char, n: i32, ka: i32, kb: i32, ab: ^complex64, ldab: i32, bb: ^complex64, ldbb: i32, q: ^complex64, ldq: i32, vl: f32, vu: f32, il: i32, iu: i32, abstol: f32, m: ^i32, w: ^f32, z: ^complex64, ldz: i32, ifail: ^i32) -> i32 ---
	LAPACKE_zhbgvx :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, uplo: c.char, n: i32, ka: i32, kb: i32, ab: ^complex128, ldab: i32, bb: ^complex128, ldbb: i32, q: ^complex128, ldq: i32, vl: f64, vu: f64, il: i32, iu: i32, abstol: f64, m: ^i32, w: ^f64, z: ^complex128, ldz: i32, ifail: ^i32) -> i32 ---
	LAPACKE_chbtrd :: proc(matrix_layout: c.int, vect: c.char, uplo: c.char, n: i32, kd: i32, ab: ^complex64, ldab: i32, d: ^f32, e: ^f32, q: ^complex64, ldq: i32) -> i32 ---
	LAPACKE_zhbtrd :: proc(matrix_layout: c.int, vect: c.char, uplo: c.char, n: i32, kd: i32, ab: ^complex128, ldab: i32, d: ^f64, e: ^f64, q: ^complex128, ldq: i32) -> i32 ---
	LAPACKE_checon :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32, ipiv: ^i32, anorm: f32, rcond: ^f32) -> i32 ---
	LAPACKE_zhecon :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32, ipiv: ^i32, anorm: f64, rcond: ^f64) -> i32 ---
	LAPACKE_cheequb :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32, s: ^f32, scond: ^f32, amax: ^f32) -> i32 ---
	LAPACKE_zheequb :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32, s: ^f64, scond: ^f64, amax: ^f64) -> i32 ---
	LAPACKE_cheev :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, a: ^complex64, lda: i32, w: ^f32) -> i32 ---
	LAPACKE_zheev :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, a: ^complex128, lda: i32, w: ^f64) -> i32 ---
	LAPACKE_cheevd :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, a: ^complex64, lda: i32, w: ^f32) -> i32 ---
	LAPACKE_zheevd :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, a: ^complex128, lda: i32, w: ^f64) -> i32 ---
	LAPACKE_cheevr :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, uplo: c.char, n: i32, a: ^complex64, lda: i32, vl: f32, vu: f32, il: i32, iu: i32, abstol: f32, m: ^i32, w: ^f32, z: ^complex64, ldz: i32, isuppz: ^i32) -> i32 ---
	LAPACKE_zheevr :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, uplo: c.char, n: i32, a: ^complex128, lda: i32, vl: f64, vu: f64, il: i32, iu: i32, abstol: f64, m: ^i32, w: ^f64, z: ^complex128, ldz: i32, isuppz: ^i32) -> i32 ---
	LAPACKE_cheevx :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, uplo: c.char, n: i32, a: ^complex64, lda: i32, vl: f32, vu: f32, il: i32, iu: i32, abstol: f32, m: ^i32, w: ^f32, z: ^complex64, ldz: i32, ifail: ^i32) -> i32 ---
	LAPACKE_zheevx :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, uplo: c.char, n: i32, a: ^complex128, lda: i32, vl: f64, vu: f64, il: i32, iu: i32, abstol: f64, m: ^i32, w: ^f64, z: ^complex128, ldz: i32, ifail: ^i32) -> i32 ---
	LAPACKE_chegst :: proc(matrix_layout: c.int, itype: i32, uplo: c.char, n: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_zhegst :: proc(matrix_layout: c.int, itype: i32, uplo: c.char, n: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_chegv :: proc(matrix_layout: c.int, itype: i32, jobz: c.char, uplo: c.char, n: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32, w: ^f32) -> i32 ---
	LAPACKE_zhegv :: proc(matrix_layout: c.int, itype: i32, jobz: c.char, uplo: c.char, n: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, w: ^f64) -> i32 ---
	LAPACKE_chegvd :: proc(matrix_layout: c.int, itype: i32, jobz: c.char, uplo: c.char, n: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32, w: ^f32) -> i32 ---
	LAPACKE_zhegvd :: proc(matrix_layout: c.int, itype: i32, jobz: c.char, uplo: c.char, n: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, w: ^f64) -> i32 ---
	LAPACKE_chegvx :: proc(matrix_layout: c.int, itype: i32, jobz: c.char, range: c.char, uplo: c.char, n: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32, vl: f32, vu: f32, il: i32, iu: i32, abstol: f32, m: ^i32, w: ^f32, z: ^complex64, ldz: i32, ifail: ^i32) -> i32 ---
	LAPACKE_zhegvx :: proc(matrix_layout: c.int, itype: i32, jobz: c.char, range: c.char, uplo: c.char, n: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, vl: f64, vu: f64, il: i32, iu: i32, abstol: f64, m: ^i32, w: ^f64, z: ^complex128, ldz: i32, ifail: ^i32) -> i32 ---
	LAPACKE_cherfs :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, af: ^complex64, ldaf: i32, ipiv: ^i32, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, ferr: ^f32, berr: ^f32) -> i32 ---
	LAPACKE_zherfs :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, af: ^complex128, ldaf: i32, ipiv: ^i32, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, ferr: ^f64, berr: ^f64) -> i32 ---
	LAPACKE_cherfsx :: proc(matrix_layout: c.int, uplo: c.char, equed: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, af: ^complex64, ldaf: i32, ipiv: ^i32, s: ^f32, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, rcond: ^f32, berr: ^f32, n_err_bnds: i32, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: i32, params: ^f32) -> i32 ---
	LAPACKE_zherfsx :: proc(matrix_layout: c.int, uplo: c.char, equed: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, af: ^complex128, ldaf: i32, ipiv: ^i32, s: ^f64, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, rcond: ^f64, berr: ^f64, n_err_bnds: i32, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: i32, params: ^f64) -> i32 ---
	LAPACKE_chesv :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, ipiv: ^i32, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_zhesv :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, ipiv: ^i32, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_chesvx :: proc(matrix_layout: c.int, fact: c.char, uplo: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, af: ^complex64, ldaf: i32, ipiv: ^i32, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, rcond: ^f32, ferr: ^f32, berr: ^f32) -> i32 ---
	LAPACKE_zhesvx :: proc(matrix_layout: c.int, fact: c.char, uplo: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, af: ^complex128, ldaf: i32, ipiv: ^i32, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, rcond: ^f64, ferr: ^f64, berr: ^f64) -> i32 ---
	LAPACKE_chesvxx :: proc(matrix_layout: c.int, fact: c.char, uplo: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, af: ^complex64, ldaf: i32, ipiv: ^i32, equed: cstring, s: ^f32, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, rcond: ^f32, rpvgrw: ^f32, berr: ^f32, n_err_bnds: i32, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: i32, params: ^f32) -> i32 ---
	LAPACKE_zhesvxx :: proc(matrix_layout: c.int, fact: c.char, uplo: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, af: ^complex128, ldaf: i32, ipiv: ^i32, equed: cstring, s: ^f64, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, rcond: ^f64, rpvgrw: ^f64, berr: ^f64, n_err_bnds: i32, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: i32, params: ^f64) -> i32 ---
	LAPACKE_chetrd :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32, d: ^f32, e: ^f32, tau: ^complex64) -> i32 ---
	LAPACKE_zhetrd :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32, d: ^f64, e: ^f64, tau: ^complex128) -> i32 ---
	LAPACKE_chetrf :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32, ipiv: ^i32) -> i32 ---
	LAPACKE_zhetrf :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32, ipiv: ^i32) -> i32 ---
	LAPACKE_chetri :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32, ipiv: ^i32) -> i32 ---
	LAPACKE_zhetri :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32, ipiv: ^i32) -> i32 ---
	LAPACKE_chetrs :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, ipiv: ^i32, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_zhetrs :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, ipiv: ^i32, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_chfrk :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, trans: c.char, n: i32, k: i32, alpha: f32, a: ^complex64, lda: i32, beta: f32, _c: ^complex64) -> i32 ---
	LAPACKE_zhfrk :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, trans: c.char, n: i32, k: i32, alpha: f64, a: ^complex128, lda: i32, beta: f64, _c: ^complex128) -> i32 ---
	LAPACKE_shgeqz :: proc(matrix_layout: c.int, job: c.char, compq: c.char, compz: c.char, n: i32, ilo: i32, ihi: i32, h: ^f32, ldh: i32, t: ^f32, ldt: i32, alphar: ^f32, alphai: ^f32, beta: ^f32, q: ^f32, ldq: i32, z: ^f32, ldz: i32) -> i32 ---
	LAPACKE_dhgeqz :: proc(matrix_layout: c.int, job: c.char, compq: c.char, compz: c.char, n: i32, ilo: i32, ihi: i32, h: ^f64, ldh: i32, t: ^f64, ldt: i32, alphar: ^f64, alphai: ^f64, beta: ^f64, q: ^f64, ldq: i32, z: ^f64, ldz: i32) -> i32 ---
	LAPACKE_chgeqz :: proc(matrix_layout: c.int, job: c.char, compq: c.char, compz: c.char, n: i32, ilo: i32, ihi: i32, h: ^complex64, ldh: i32, t: ^complex64, ldt: i32, alpha: ^complex64, beta: ^complex64, q: ^complex64, ldq: i32, z: ^complex64, ldz: i32) -> i32 ---
	LAPACKE_zhgeqz :: proc(matrix_layout: c.int, job: c.char, compq: c.char, compz: c.char, n: i32, ilo: i32, ihi: i32, h: ^complex128, ldh: i32, t: ^complex128, ldt: i32, alpha: ^complex128, beta: ^complex128, q: ^complex128, ldq: i32, z: ^complex128, ldz: i32) -> i32 ---
	LAPACKE_chpcon :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^complex64, ipiv: ^i32, anorm: f32, rcond: ^f32) -> i32 ---
	LAPACKE_zhpcon :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^complex128, ipiv: ^i32, anorm: f64, rcond: ^f64) -> i32 ---
	LAPACKE_chpev :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, ap: ^complex64, w: ^f32, z: ^complex64, ldz: i32) -> i32 ---
	LAPACKE_zhpev :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, ap: ^complex128, w: ^f64, z: ^complex128, ldz: i32) -> i32 ---
	LAPACKE_chpevd :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, ap: ^complex64, w: ^f32, z: ^complex64, ldz: i32) -> i32 ---
	LAPACKE_zhpevd :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, ap: ^complex128, w: ^f64, z: ^complex128, ldz: i32) -> i32 ---
	LAPACKE_chpevx :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, uplo: c.char, n: i32, ap: ^complex64, vl: f32, vu: f32, il: i32, iu: i32, abstol: f32, m: ^i32, w: ^f32, z: ^complex64, ldz: i32, ifail: ^i32) -> i32 ---
	LAPACKE_zhpevx :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, uplo: c.char, n: i32, ap: ^complex128, vl: f64, vu: f64, il: i32, iu: i32, abstol: f64, m: ^i32, w: ^f64, z: ^complex128, ldz: i32, ifail: ^i32) -> i32 ---
	LAPACKE_chpgst :: proc(matrix_layout: c.int, itype: i32, uplo: c.char, n: i32, ap: ^complex64, bp: ^complex64) -> i32 ---
	LAPACKE_zhpgst :: proc(matrix_layout: c.int, itype: i32, uplo: c.char, n: i32, ap: ^complex128, bp: ^complex128) -> i32 ---
	LAPACKE_chpgv :: proc(matrix_layout: c.int, itype: i32, jobz: c.char, uplo: c.char, n: i32, ap: ^complex64, bp: ^complex64, w: ^f32, z: ^complex64, ldz: i32) -> i32 ---
	LAPACKE_zhpgv :: proc(matrix_layout: c.int, itype: i32, jobz: c.char, uplo: c.char, n: i32, ap: ^complex128, bp: ^complex128, w: ^f64, z: ^complex128, ldz: i32) -> i32 ---
	LAPACKE_chpgvd :: proc(matrix_layout: c.int, itype: i32, jobz: c.char, uplo: c.char, n: i32, ap: ^complex64, bp: ^complex64, w: ^f32, z: ^complex64, ldz: i32) -> i32 ---
	LAPACKE_zhpgvd :: proc(matrix_layout: c.int, itype: i32, jobz: c.char, uplo: c.char, n: i32, ap: ^complex128, bp: ^complex128, w: ^f64, z: ^complex128, ldz: i32) -> i32 ---
	LAPACKE_chpgvx :: proc(matrix_layout: c.int, itype: i32, jobz: c.char, range: c.char, uplo: c.char, n: i32, ap: ^complex64, bp: ^complex64, vl: f32, vu: f32, il: i32, iu: i32, abstol: f32, m: ^i32, w: ^f32, z: ^complex64, ldz: i32, ifail: ^i32) -> i32 ---
	LAPACKE_zhpgvx :: proc(matrix_layout: c.int, itype: i32, jobz: c.char, range: c.char, uplo: c.char, n: i32, ap: ^complex128, bp: ^complex128, vl: f64, vu: f64, il: i32, iu: i32, abstol: f64, m: ^i32, w: ^f64, z: ^complex128, ldz: i32, ifail: ^i32) -> i32 ---
	LAPACKE_chprfs :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, ap: ^complex64, afp: ^complex64, ipiv: ^i32, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, ferr: ^f32, berr: ^f32) -> i32 ---
	LAPACKE_zhprfs :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, ap: ^complex128, afp: ^complex128, ipiv: ^i32, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, ferr: ^f64, berr: ^f64) -> i32 ---
	LAPACKE_chpsv :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, ap: ^complex64, ipiv: ^i32, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_zhpsv :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, ap: ^complex128, ipiv: ^i32, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_chpsvx :: proc(matrix_layout: c.int, fact: c.char, uplo: c.char, n: i32, nrhs: i32, ap: ^complex64, afp: ^complex64, ipiv: ^i32, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, rcond: ^f32, ferr: ^f32, berr: ^f32) -> i32 ---
	LAPACKE_zhpsvx :: proc(matrix_layout: c.int, fact: c.char, uplo: c.char, n: i32, nrhs: i32, ap: ^complex128, afp: ^complex128, ipiv: ^i32, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, rcond: ^f64, ferr: ^f64, berr: ^f64) -> i32 ---
	LAPACKE_chptrd :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^complex64, d: ^f32, e: ^f32, tau: ^complex64) -> i32 ---
	LAPACKE_zhptrd :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^complex128, d: ^f64, e: ^f64, tau: ^complex128) -> i32 ---
	LAPACKE_chptrf :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^complex64, ipiv: ^i32) -> i32 ---
	LAPACKE_zhptrf :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^complex128, ipiv: ^i32) -> i32 ---
	LAPACKE_chptri :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^complex64, ipiv: ^i32) -> i32 ---
	LAPACKE_zhptri :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^complex128, ipiv: ^i32) -> i32 ---
	LAPACKE_chptrs :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, ap: ^complex64, ipiv: ^i32, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_zhptrs :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, ap: ^complex128, ipiv: ^i32, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_shsein :: proc(matrix_layout: c.int, job: c.char, eigsrc: c.char, initv: c.char, select: ^i32, n: i32, h: ^f32, ldh: i32, wr: ^f32, wi: ^f32, vl: ^f32, ldvl: i32, vr: ^f32, ldvr: i32, mm: i32, m: ^i32, ifaill: ^i32, ifailr: ^i32) -> i32 ---
	LAPACKE_dhsein :: proc(matrix_layout: c.int, job: c.char, eigsrc: c.char, initv: c.char, select: ^i32, n: i32, h: ^f64, ldh: i32, wr: ^f64, wi: ^f64, vl: ^f64, ldvl: i32, vr: ^f64, ldvr: i32, mm: i32, m: ^i32, ifaill: ^i32, ifailr: ^i32) -> i32 ---
	LAPACKE_chsein :: proc(matrix_layout: c.int, job: c.char, eigsrc: c.char, initv: c.char, select: ^i32, n: i32, h: ^complex64, ldh: i32, w: ^complex64, vl: ^complex64, ldvl: i32, vr: ^complex64, ldvr: i32, mm: i32, m: ^i32, ifaill: ^i32, ifailr: ^i32) -> i32 ---
	LAPACKE_zhsein :: proc(matrix_layout: c.int, job: c.char, eigsrc: c.char, initv: c.char, select: ^i32, n: i32, h: ^complex128, ldh: i32, w: ^complex128, vl: ^complex128, ldvl: i32, vr: ^complex128, ldvr: i32, mm: i32, m: ^i32, ifaill: ^i32, ifailr: ^i32) -> i32 ---
	LAPACKE_shseqr :: proc(matrix_layout: c.int, job: c.char, compz: c.char, n: i32, ilo: i32, ihi: i32, h: ^f32, ldh: i32, wr: ^f32, wi: ^f32, z: ^f32, ldz: i32) -> i32 ---
	LAPACKE_dhseqr :: proc(matrix_layout: c.int, job: c.char, compz: c.char, n: i32, ilo: i32, ihi: i32, h: ^f64, ldh: i32, wr: ^f64, wi: ^f64, z: ^f64, ldz: i32) -> i32 ---
	LAPACKE_chseqr :: proc(matrix_layout: c.int, job: c.char, compz: c.char, n: i32, ilo: i32, ihi: i32, h: ^complex64, ldh: i32, w: ^complex64, z: ^complex64, ldz: i32) -> i32 ---
	LAPACKE_zhseqr :: proc(matrix_layout: c.int, job: c.char, compz: c.char, n: i32, ilo: i32, ihi: i32, h: ^complex128, ldh: i32, w: ^complex128, z: ^complex128, ldz: i32) -> i32 ---
	LAPACKE_clacgv :: proc(n: i32, x: ^complex64, incx: i32) -> i32 ---
	LAPACKE_zlacgv :: proc(n: i32, x: ^complex128, incx: i32) -> i32 ---
	LAPACKE_slacn2 :: proc(n: i32, v: ^f32, x: ^f32, isgn: ^i32, est: ^f32, kase: ^i32, isave: ^i32) -> i32 ---
	LAPACKE_dlacn2 :: proc(n: i32, v: ^f64, x: ^f64, isgn: ^i32, est: ^f64, kase: ^i32, isave: ^i32) -> i32 ---
	LAPACKE_clacn2 :: proc(n: i32, v: ^complex64, x: ^complex64, est: ^f32, kase: ^i32, isave: ^i32) -> i32 ---
	LAPACKE_zlacn2 :: proc(n: i32, v: ^complex128, x: ^complex128, est: ^f64, kase: ^i32, isave: ^i32) -> i32 ---
	LAPACKE_slacpy :: proc(matrix_layout: c.int, uplo: c.char, m: i32, n: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32) -> i32 ---
	LAPACKE_dlacpy :: proc(matrix_layout: c.int, uplo: c.char, m: i32, n: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32) -> i32 ---
	LAPACKE_clacpy :: proc(matrix_layout: c.int, uplo: c.char, m: i32, n: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_zlacpy :: proc(matrix_layout: c.int, uplo: c.char, m: i32, n: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_clacp2 :: proc(matrix_layout: c.int, uplo: c.char, m: i32, n: i32, a: ^f32, lda: i32, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_zlacp2 :: proc(matrix_layout: c.int, uplo: c.char, m: i32, n: i32, a: ^f64, lda: i32, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_zlag2c :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex128, lda: i32, sa: ^complex64, ldsa: i32) -> i32 ---
	LAPACKE_slag2d :: proc(matrix_layout: c.int, m: i32, n: i32, sa: ^f32, ldsa: i32, a: ^f64, lda: i32) -> i32 ---
	LAPACKE_dlag2s :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f64, lda: i32, sa: ^f32, ldsa: i32) -> i32 ---
	LAPACKE_clag2z :: proc(matrix_layout: c.int, m: i32, n: i32, sa: ^complex64, ldsa: i32, a: ^complex128, lda: i32) -> i32 ---
	LAPACKE_slagge :: proc(matrix_layout: c.int, m: i32, n: i32, kl: i32, ku: i32, d: ^f32, a: ^f32, lda: i32, iseed: ^i32) -> i32 ---
	LAPACKE_dlagge :: proc(matrix_layout: c.int, m: i32, n: i32, kl: i32, ku: i32, d: ^f64, a: ^f64, lda: i32, iseed: ^i32) -> i32 ---
	LAPACKE_clagge :: proc(matrix_layout: c.int, m: i32, n: i32, kl: i32, ku: i32, d: ^f32, a: ^complex64, lda: i32, iseed: ^i32) -> i32 ---
	LAPACKE_zlagge :: proc(matrix_layout: c.int, m: i32, n: i32, kl: i32, ku: i32, d: ^f64, a: ^complex128, lda: i32, iseed: ^i32) -> i32 ---
	LAPACKE_slamch :: proc(cmach: c.char) -> f32 ---
	LAPACKE_dlamch :: proc(cmach: c.char) -> f64 ---
	LAPACKE_slangb :: proc(matrix_layout: c.int, norm: c.char, n: i32, kl: i32, ku: i32, ab: ^f32, ldab: i32) -> f32 ---
	LAPACKE_dlangb :: proc(matrix_layout: c.int, norm: c.char, n: i32, kl: i32, ku: i32, ab: ^f64, ldab: i32) -> f64 ---
	LAPACKE_clangb :: proc(matrix_layout: c.int, norm: c.char, n: i32, kl: i32, ku: i32, ab: ^complex64, ldab: i32) -> f32 ---
	LAPACKE_zlangb :: proc(matrix_layout: c.int, norm: c.char, n: i32, kl: i32, ku: i32, ab: ^complex128, ldab: i32) -> f64 ---
	LAPACKE_slange :: proc(matrix_layout: c.int, norm: c.char, m: i32, n: i32, a: ^f32, lda: i32) -> f32 ---
	LAPACKE_dlange :: proc(matrix_layout: c.int, norm: c.char, m: i32, n: i32, a: ^f64, lda: i32) -> f64 ---
	LAPACKE_clange :: proc(matrix_layout: c.int, norm: c.char, m: i32, n: i32, a: ^complex64, lda: i32) -> f32 ---
	LAPACKE_zlange :: proc(matrix_layout: c.int, norm: c.char, m: i32, n: i32, a: ^complex128, lda: i32) -> f64 ---
	LAPACKE_clanhe :: proc(matrix_layout: c.int, norm: c.char, uplo: c.char, n: i32, a: ^complex64, lda: i32) -> f32 ---
	LAPACKE_zlanhe :: proc(matrix_layout: c.int, norm: c.char, uplo: c.char, n: i32, a: ^complex128, lda: i32) -> f64 ---
	LAPACKE_clacrm :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex64, lda: i32, b: ^f32, ldb: i32, _c: ^complex64, ldc: i32) -> i32 ---
	LAPACKE_zlacrm :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex128, lda: i32, b: ^f64, ldb: i32, _c: ^complex128, ldc: i32) -> i32 ---
	LAPACKE_clarcm :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f32, lda: i32, b: ^complex64, ldb: i32, _c: ^complex64, ldc: i32) -> i32 ---
	LAPACKE_zlarcm :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f64, lda: i32, b: ^complex128, ldb: i32, _c: ^complex128, ldc: i32) -> i32 ---
	LAPACKE_slansy :: proc(matrix_layout: c.int, norm: c.char, uplo: c.char, n: i32, a: ^f32, lda: i32) -> f32 ---
	LAPACKE_dlansy :: proc(matrix_layout: c.int, norm: c.char, uplo: c.char, n: i32, a: ^f64, lda: i32) -> f64 ---
	LAPACKE_clansy :: proc(matrix_layout: c.int, norm: c.char, uplo: c.char, n: i32, a: ^complex64, lda: i32) -> f32 ---
	LAPACKE_zlansy :: proc(matrix_layout: c.int, norm: c.char, uplo: c.char, n: i32, a: ^complex128, lda: i32) -> f64 ---
	LAPACKE_slantr :: proc(matrix_layout: c.int, norm: c.char, uplo: c.char, diag: c.char, m: i32, n: i32, a: ^f32, lda: i32) -> f32 ---
	LAPACKE_dlantr :: proc(matrix_layout: c.int, norm: c.char, uplo: c.char, diag: c.char, m: i32, n: i32, a: ^f64, lda: i32) -> f64 ---
	LAPACKE_clantr :: proc(matrix_layout: c.int, norm: c.char, uplo: c.char, diag: c.char, m: i32, n: i32, a: ^complex64, lda: i32) -> f32 ---
	LAPACKE_zlantr :: proc(matrix_layout: c.int, norm: c.char, uplo: c.char, diag: c.char, m: i32, n: i32, a: ^complex128, lda: i32) -> f64 ---
	LAPACKE_slarfb :: proc(matrix_layout: c.int, side: c.char, trans: c.char, direct: c.char, storev: c.char, m: i32, n: i32, k: i32, v: ^f32, ldv: i32, t: ^f32, ldt: i32, _c: ^f32, ldc: i32) -> i32 ---
	LAPACKE_dlarfb :: proc(matrix_layout: c.int, side: c.char, trans: c.char, direct: c.char, storev: c.char, m: i32, n: i32, k: i32, v: ^f64, ldv: i32, t: ^f64, ldt: i32, _c: ^f64, ldc: i32) -> i32 ---
	LAPACKE_clarfb :: proc(matrix_layout: c.int, side: c.char, trans: c.char, direct: c.char, storev: c.char, m: i32, n: i32, k: i32, v: ^complex64, ldv: i32, t: ^complex64, ldt: i32, _c: ^complex64, ldc: i32) -> i32 ---
	LAPACKE_zlarfb :: proc(matrix_layout: c.int, side: c.char, trans: c.char, direct: c.char, storev: c.char, m: i32, n: i32, k: i32, v: ^complex128, ldv: i32, t: ^complex128, ldt: i32, _c: ^complex128, ldc: i32) -> i32 ---
	LAPACKE_slarfg :: proc(n: i32, alpha: ^f32, x: ^f32, incx: i32, tau: ^f32) -> i32 ---
	LAPACKE_dlarfg :: proc(n: i32, alpha: ^f64, x: ^f64, incx: i32, tau: ^f64) -> i32 ---
	LAPACKE_clarfg :: proc(n: i32, alpha: ^complex64, x: ^complex64, incx: i32, tau: ^complex64) -> i32 ---
	LAPACKE_zlarfg :: proc(n: i32, alpha: ^complex128, x: ^complex128, incx: i32, tau: ^complex128) -> i32 ---
	LAPACKE_slarft :: proc(matrix_layout: c.int, direct: c.char, storev: c.char, n: i32, k: i32, v: ^f32, ldv: i32, tau: ^f32, t: ^f32, ldt: i32) -> i32 ---
	LAPACKE_dlarft :: proc(matrix_layout: c.int, direct: c.char, storev: c.char, n: i32, k: i32, v: ^f64, ldv: i32, tau: ^f64, t: ^f64, ldt: i32) -> i32 ---
	LAPACKE_clarft :: proc(matrix_layout: c.int, direct: c.char, storev: c.char, n: i32, k: i32, v: ^complex64, ldv: i32, tau: ^complex64, t: ^complex64, ldt: i32) -> i32 ---
	LAPACKE_zlarft :: proc(matrix_layout: c.int, direct: c.char, storev: c.char, n: i32, k: i32, v: ^complex128, ldv: i32, tau: ^complex128, t: ^complex128, ldt: i32) -> i32 ---
	LAPACKE_slarfx :: proc(matrix_layout: c.int, side: c.char, m: i32, n: i32, v: ^f32, tau: f32, _c: ^f32, ldc: i32, work: ^f32) -> i32 ---
	LAPACKE_dlarfx :: proc(matrix_layout: c.int, side: c.char, m: i32, n: i32, v: ^f64, tau: f64, _c: ^f64, ldc: i32, work: ^f64) -> i32 ---
	LAPACKE_clarfx :: proc(matrix_layout: c.int, side: c.char, m: i32, n: i32, v: ^complex64, tau: complex64, _c: ^complex64, ldc: i32, work: ^complex64) -> i32 ---
	LAPACKE_zlarfx :: proc(matrix_layout: c.int, side: c.char, m: i32, n: i32, v: ^complex128, tau: complex128, _c: ^complex128, ldc: i32, work: ^complex128) -> i32 ---
	LAPACKE_slarnv :: proc(idist: i32, iseed: ^i32, n: i32, x: ^f32) -> i32 ---
	LAPACKE_dlarnv :: proc(idist: i32, iseed: ^i32, n: i32, x: ^f64) -> i32 ---
	LAPACKE_clarnv :: proc(idist: i32, iseed: ^i32, n: i32, x: ^complex64) -> i32 ---
	LAPACKE_zlarnv :: proc(idist: i32, iseed: ^i32, n: i32, x: ^complex128) -> i32 ---
	LAPACKE_slascl :: proc(matrix_layout: c.int, type: c.char, kl: i32, ku: i32, cfrom: f32, cto: f32, m: i32, n: i32, a: ^f32, lda: i32) -> i32 ---
	LAPACKE_dlascl :: proc(matrix_layout: c.int, type: c.char, kl: i32, ku: i32, cfrom: f64, cto: f64, m: i32, n: i32, a: ^f64, lda: i32) -> i32 ---
	LAPACKE_clascl :: proc(matrix_layout: c.int, type: c.char, kl: i32, ku: i32, cfrom: f32, cto: f32, m: i32, n: i32, a: ^complex64, lda: i32) -> i32 ---
	LAPACKE_zlascl :: proc(matrix_layout: c.int, type: c.char, kl: i32, ku: i32, cfrom: f64, cto: f64, m: i32, n: i32, a: ^complex128, lda: i32) -> i32 ---
	LAPACKE_slaset :: proc(matrix_layout: c.int, uplo: c.char, m: i32, n: i32, alpha: f32, beta: f32, a: ^f32, lda: i32) -> i32 ---
	LAPACKE_dlaset :: proc(matrix_layout: c.int, uplo: c.char, m: i32, n: i32, alpha: f64, beta: f64, a: ^f64, lda: i32) -> i32 ---
	LAPACKE_claset :: proc(matrix_layout: c.int, uplo: c.char, m: i32, n: i32, alpha: complex64, beta: complex64, a: ^complex64, lda: i32) -> i32 ---
	LAPACKE_zlaset :: proc(matrix_layout: c.int, uplo: c.char, m: i32, n: i32, alpha: complex128, beta: complex128, a: ^complex128, lda: i32) -> i32 ---
	LAPACKE_slasrt :: proc(id: c.char, n: i32, d: ^f32) -> i32 ---
	LAPACKE_dlasrt :: proc(id: c.char, n: i32, d: ^f64) -> i32 ---
	LAPACKE_slassq :: proc(n: i32, x: ^f32, incx: i32, scale: ^f32, sumsq: ^f32) -> i32 ---
	LAPACKE_dlassq :: proc(n: i32, x: ^f64, incx: i32, scale: ^f64, sumsq: ^f64) -> i32 ---
	LAPACKE_classq :: proc(n: i32, x: ^complex64, incx: i32, scale: ^f32, sumsq: ^f32) -> i32 ---
	LAPACKE_zlassq :: proc(n: i32, x: ^complex128, incx: i32, scale: ^f64, sumsq: ^f64) -> i32 ---
	LAPACKE_slaswp :: proc(matrix_layout: c.int, n: i32, a: ^f32, lda: i32, k1: i32, k2: i32, ipiv: ^i32, incx: i32) -> i32 ---
	LAPACKE_dlaswp :: proc(matrix_layout: c.int, n: i32, a: ^f64, lda: i32, k1: i32, k2: i32, ipiv: ^i32, incx: i32) -> i32 ---
	LAPACKE_claswp :: proc(matrix_layout: c.int, n: i32, a: ^complex64, lda: i32, k1: i32, k2: i32, ipiv: ^i32, incx: i32) -> i32 ---
	LAPACKE_zlaswp :: proc(matrix_layout: c.int, n: i32, a: ^complex128, lda: i32, k1: i32, k2: i32, ipiv: ^i32, incx: i32) -> i32 ---
	LAPACKE_slatms :: proc(matrix_layout: c.int, m: i32, n: i32, dist: c.char, iseed: ^i32, sym: c.char, d: ^f32, mode: i32, cond: f32, dmax: f32, kl: i32, ku: i32, pack: c.char, a: ^f32, lda: i32) -> i32 ---
	LAPACKE_dlatms :: proc(matrix_layout: c.int, m: i32, n: i32, dist: c.char, iseed: ^i32, sym: c.char, d: ^f64, mode: i32, cond: f64, dmax: f64, kl: i32, ku: i32, pack: c.char, a: ^f64, lda: i32) -> i32 ---
	LAPACKE_clatms :: proc(matrix_layout: c.int, m: i32, n: i32, dist: c.char, iseed: ^i32, sym: c.char, d: ^f32, mode: i32, cond: f32, dmax: f32, kl: i32, ku: i32, pack: c.char, a: ^complex64, lda: i32) -> i32 ---
	LAPACKE_zlatms :: proc(matrix_layout: c.int, m: i32, n: i32, dist: c.char, iseed: ^i32, sym: c.char, d: ^f64, mode: i32, cond: f64, dmax: f64, kl: i32, ku: i32, pack: c.char, a: ^complex128, lda: i32) -> i32 ---
	LAPACKE_slauum :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f32, lda: i32) -> i32 ---
	LAPACKE_dlauum :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f64, lda: i32) -> i32 ---
	LAPACKE_clauum :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32) -> i32 ---
	LAPACKE_zlauum :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32) -> i32 ---
	LAPACKE_sopgtr :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^f32, tau: ^f32, q: ^f32, ldq: i32) -> i32 ---
	LAPACKE_dopgtr :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^f64, tau: ^f64, q: ^f64, ldq: i32) -> i32 ---
	LAPACKE_sopmtr :: proc(matrix_layout: c.int, side: c.char, uplo: c.char, trans: c.char, m: i32, n: i32, ap: ^f32, tau: ^f32, _c: ^f32, ldc: i32) -> i32 ---
	LAPACKE_dopmtr :: proc(matrix_layout: c.int, side: c.char, uplo: c.char, trans: c.char, m: i32, n: i32, ap: ^f64, tau: ^f64, _c: ^f64, ldc: i32) -> i32 ---
	LAPACKE_sorgbr :: proc(matrix_layout: c.int, vect: c.char, m: i32, n: i32, k: i32, a: ^f32, lda: i32, tau: ^f32) -> i32 ---
	LAPACKE_dorgbr :: proc(matrix_layout: c.int, vect: c.char, m: i32, n: i32, k: i32, a: ^f64, lda: i32, tau: ^f64) -> i32 ---
	LAPACKE_sorghr :: proc(matrix_layout: c.int, n: i32, ilo: i32, ihi: i32, a: ^f32, lda: i32, tau: ^f32) -> i32 ---
	LAPACKE_dorghr :: proc(matrix_layout: c.int, n: i32, ilo: i32, ihi: i32, a: ^f64, lda: i32, tau: ^f64) -> i32 ---
	LAPACKE_sorglq :: proc(matrix_layout: c.int, m: i32, n: i32, k: i32, a: ^f32, lda: i32, tau: ^f32) -> i32 ---
	LAPACKE_dorglq :: proc(matrix_layout: c.int, m: i32, n: i32, k: i32, a: ^f64, lda: i32, tau: ^f64) -> i32 ---
	LAPACKE_sorgql :: proc(matrix_layout: c.int, m: i32, n: i32, k: i32, a: ^f32, lda: i32, tau: ^f32) -> i32 ---
	LAPACKE_dorgql :: proc(matrix_layout: c.int, m: i32, n: i32, k: i32, a: ^f64, lda: i32, tau: ^f64) -> i32 ---
	LAPACKE_sorgqr :: proc(matrix_layout: c.int, m: i32, n: i32, k: i32, a: ^f32, lda: i32, tau: ^f32) -> i32 ---
	LAPACKE_dorgqr :: proc(matrix_layout: c.int, m: i32, n: i32, k: i32, a: ^f64, lda: i32, tau: ^f64) -> i32 ---
	LAPACKE_sorgrq :: proc(matrix_layout: c.int, m: i32, n: i32, k: i32, a: ^f32, lda: i32, tau: ^f32) -> i32 ---
	LAPACKE_dorgrq :: proc(matrix_layout: c.int, m: i32, n: i32, k: i32, a: ^f64, lda: i32, tau: ^f64) -> i32 ---
	LAPACKE_sorgtr :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f32, lda: i32, tau: ^f32) -> i32 ---
	LAPACKE_dorgtr :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f64, lda: i32, tau: ^f64) -> i32 ---
	LAPACKE_sorgtsqr_row :: proc(matrix_layout: c.int, m: i32, n: i32, mb: i32, nb: i32, a: ^f32, lda: i32, t: ^f32, ldt: i32) -> i32 ---
	LAPACKE_dorgtsqr_row :: proc(matrix_layout: c.int, m: i32, n: i32, mb: i32, nb: i32, a: ^f64, lda: i32, t: ^f64, ldt: i32) -> i32 ---
	LAPACKE_sormbr :: proc(matrix_layout: c.int, vect: c.char, side: c.char, trans: c.char, m: i32, n: i32, k: i32, a: ^f32, lda: i32, tau: ^f32, _c: ^f32, ldc: i32) -> i32 ---
	LAPACKE_dormbr :: proc(matrix_layout: c.int, vect: c.char, side: c.char, trans: c.char, m: i32, n: i32, k: i32, a: ^f64, lda: i32, tau: ^f64, _c: ^f64, ldc: i32) -> i32 ---
	LAPACKE_sormhr :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, ilo: i32, ihi: i32, a: ^f32, lda: i32, tau: ^f32, _c: ^f32, ldc: i32) -> i32 ---
	LAPACKE_dormhr :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, ilo: i32, ihi: i32, a: ^f64, lda: i32, tau: ^f64, _c: ^f64, ldc: i32) -> i32 ---
	LAPACKE_sormlq :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, a: ^f32, lda: i32, tau: ^f32, _c: ^f32, ldc: i32) -> i32 ---
	LAPACKE_dormlq :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, a: ^f64, lda: i32, tau: ^f64, _c: ^f64, ldc: i32) -> i32 ---
	LAPACKE_sormql :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, a: ^f32, lda: i32, tau: ^f32, _c: ^f32, ldc: i32) -> i32 ---
	LAPACKE_dormql :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, a: ^f64, lda: i32, tau: ^f64, _c: ^f64, ldc: i32) -> i32 ---
	LAPACKE_sormqr :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, a: ^f32, lda: i32, tau: ^f32, _c: ^f32, ldc: i32) -> i32 ---
	LAPACKE_dormqr :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, a: ^f64, lda: i32, tau: ^f64, _c: ^f64, ldc: i32) -> i32 ---
	LAPACKE_sormrq :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, a: ^f32, lda: i32, tau: ^f32, _c: ^f32, ldc: i32) -> i32 ---
	LAPACKE_dormrq :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, a: ^f64, lda: i32, tau: ^f64, _c: ^f64, ldc: i32) -> i32 ---
	LAPACKE_sormrz :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, l: i32, a: ^f32, lda: i32, tau: ^f32, _c: ^f32, ldc: i32) -> i32 ---
	LAPACKE_dormrz :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, l: i32, a: ^f64, lda: i32, tau: ^f64, _c: ^f64, ldc: i32) -> i32 ---
	LAPACKE_sormtr :: proc(matrix_layout: c.int, side: c.char, uplo: c.char, trans: c.char, m: i32, n: i32, a: ^f32, lda: i32, tau: ^f32, _c: ^f32, ldc: i32) -> i32 ---
	LAPACKE_dormtr :: proc(matrix_layout: c.int, side: c.char, uplo: c.char, trans: c.char, m: i32, n: i32, a: ^f64, lda: i32, tau: ^f64, _c: ^f64, ldc: i32) -> i32 ---
	LAPACKE_spbcon :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kd: i32, ab: ^f32, ldab: i32, anorm: f32, rcond: ^f32) -> i32 ---
	LAPACKE_dpbcon :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kd: i32, ab: ^f64, ldab: i32, anorm: f64, rcond: ^f64) -> i32 ---
	LAPACKE_cpbcon :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kd: i32, ab: ^complex64, ldab: i32, anorm: f32, rcond: ^f32) -> i32 ---
	LAPACKE_zpbcon :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kd: i32, ab: ^complex128, ldab: i32, anorm: f64, rcond: ^f64) -> i32 ---
	LAPACKE_spbequ :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kd: i32, ab: ^f32, ldab: i32, s: ^f32, scond: ^f32, amax: ^f32) -> i32 ---
	LAPACKE_dpbequ :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kd: i32, ab: ^f64, ldab: i32, s: ^f64, scond: ^f64, amax: ^f64) -> i32 ---
	LAPACKE_cpbequ :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kd: i32, ab: ^complex64, ldab: i32, s: ^f32, scond: ^f32, amax: ^f32) -> i32 ---
	LAPACKE_zpbequ :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kd: i32, ab: ^complex128, ldab: i32, s: ^f64, scond: ^f64, amax: ^f64) -> i32 ---
	LAPACKE_spbrfs :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kd: i32, nrhs: i32, ab: ^f32, ldab: i32, afb: ^f32, ldafb: i32, b: ^f32, ldb: i32, x: ^f32, ldx: i32, ferr: ^f32, berr: ^f32) -> i32 ---
	LAPACKE_dpbrfs :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kd: i32, nrhs: i32, ab: ^f64, ldab: i32, afb: ^f64, ldafb: i32, b: ^f64, ldb: i32, x: ^f64, ldx: i32, ferr: ^f64, berr: ^f64) -> i32 ---
	LAPACKE_cpbrfs :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kd: i32, nrhs: i32, ab: ^complex64, ldab: i32, afb: ^complex64, ldafb: i32, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, ferr: ^f32, berr: ^f32) -> i32 ---
	LAPACKE_zpbrfs :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kd: i32, nrhs: i32, ab: ^complex128, ldab: i32, afb: ^complex128, ldafb: i32, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, ferr: ^f64, berr: ^f64) -> i32 ---
	LAPACKE_spbstf :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kb: i32, bb: ^f32, ldbb: i32) -> i32 ---
	LAPACKE_dpbstf :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kb: i32, bb: ^f64, ldbb: i32) -> i32 ---
	LAPACKE_cpbstf :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kb: i32, bb: ^complex64, ldbb: i32) -> i32 ---
	LAPACKE_zpbstf :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kb: i32, bb: ^complex128, ldbb: i32) -> i32 ---
	LAPACKE_spbsv :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kd: i32, nrhs: i32, ab: ^f32, ldab: i32, b: ^f32, ldb: i32) -> i32 ---
	LAPACKE_dpbsv :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kd: i32, nrhs: i32, ab: ^f64, ldab: i32, b: ^f64, ldb: i32) -> i32 ---
	LAPACKE_cpbsv :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kd: i32, nrhs: i32, ab: ^complex64, ldab: i32, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_zpbsv :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kd: i32, nrhs: i32, ab: ^complex128, ldab: i32, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_spbsvx :: proc(matrix_layout: c.int, fact: c.char, uplo: c.char, n: i32, kd: i32, nrhs: i32, ab: ^f32, ldab: i32, afb: ^f32, ldafb: i32, equed: cstring, s: ^f32, b: ^f32, ldb: i32, x: ^f32, ldx: i32, rcond: ^f32, ferr: ^f32, berr: ^f32) -> i32 ---
	LAPACKE_dpbsvx :: proc(matrix_layout: c.int, fact: c.char, uplo: c.char, n: i32, kd: i32, nrhs: i32, ab: ^f64, ldab: i32, afb: ^f64, ldafb: i32, equed: cstring, s: ^f64, b: ^f64, ldb: i32, x: ^f64, ldx: i32, rcond: ^f64, ferr: ^f64, berr: ^f64) -> i32 ---
	LAPACKE_cpbsvx :: proc(matrix_layout: c.int, fact: c.char, uplo: c.char, n: i32, kd: i32, nrhs: i32, ab: ^complex64, ldab: i32, afb: ^complex64, ldafb: i32, equed: cstring, s: ^f32, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, rcond: ^f32, ferr: ^f32, berr: ^f32) -> i32 ---
	LAPACKE_zpbsvx :: proc(matrix_layout: c.int, fact: c.char, uplo: c.char, n: i32, kd: i32, nrhs: i32, ab: ^complex128, ldab: i32, afb: ^complex128, ldafb: i32, equed: cstring, s: ^f64, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, rcond: ^f64, ferr: ^f64, berr: ^f64) -> i32 ---
	LAPACKE_spbtrf :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kd: i32, ab: ^f32, ldab: i32) -> i32 ---
	LAPACKE_dpbtrf :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kd: i32, ab: ^f64, ldab: i32) -> i32 ---
	LAPACKE_cpbtrf :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kd: i32, ab: ^complex64, ldab: i32) -> i32 ---
	LAPACKE_zpbtrf :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kd: i32, ab: ^complex128, ldab: i32) -> i32 ---
	LAPACKE_spbtrs :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kd: i32, nrhs: i32, ab: ^f32, ldab: i32, b: ^f32, ldb: i32) -> i32 ---
	LAPACKE_dpbtrs :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kd: i32, nrhs: i32, ab: ^f64, ldab: i32, b: ^f64, ldb: i32) -> i32 ---
	LAPACKE_cpbtrs :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kd: i32, nrhs: i32, ab: ^complex64, ldab: i32, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_zpbtrs :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kd: i32, nrhs: i32, ab: ^complex128, ldab: i32, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_spftrf :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, n: i32, a: ^f32) -> i32 ---
	LAPACKE_dpftrf :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, n: i32, a: ^f64) -> i32 ---
	LAPACKE_cpftrf :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, n: i32, a: ^complex64) -> i32 ---
	LAPACKE_zpftrf :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, n: i32, a: ^complex128) -> i32 ---
	LAPACKE_spftri :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, n: i32, a: ^f32) -> i32 ---
	LAPACKE_dpftri :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, n: i32, a: ^f64) -> i32 ---
	LAPACKE_cpftri :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, n: i32, a: ^complex64) -> i32 ---
	LAPACKE_zpftri :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, n: i32, a: ^complex128) -> i32 ---
	LAPACKE_spftrs :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, n: i32, nrhs: i32, a: ^f32, b: ^f32, ldb: i32) -> i32 ---
	LAPACKE_dpftrs :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, n: i32, nrhs: i32, a: ^f64, b: ^f64, ldb: i32) -> i32 ---
	LAPACKE_cpftrs :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, n: i32, nrhs: i32, a: ^complex64, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_zpftrs :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, n: i32, nrhs: i32, a: ^complex128, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_spocon :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f32, lda: i32, anorm: f32, rcond: ^f32) -> i32 ---
	LAPACKE_dpocon :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f64, lda: i32, anorm: f64, rcond: ^f64) -> i32 ---
	LAPACKE_cpocon :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32, anorm: f32, rcond: ^f32) -> i32 ---
	LAPACKE_zpocon :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32, anorm: f64, rcond: ^f64) -> i32 ---
	LAPACKE_spoequ :: proc(matrix_layout: c.int, n: i32, a: ^f32, lda: i32, s: ^f32, scond: ^f32, amax: ^f32) -> i32 ---
	LAPACKE_dpoequ :: proc(matrix_layout: c.int, n: i32, a: ^f64, lda: i32, s: ^f64, scond: ^f64, amax: ^f64) -> i32 ---
	LAPACKE_cpoequ :: proc(matrix_layout: c.int, n: i32, a: ^complex64, lda: i32, s: ^f32, scond: ^f32, amax: ^f32) -> i32 ---
	LAPACKE_zpoequ :: proc(matrix_layout: c.int, n: i32, a: ^complex128, lda: i32, s: ^f64, scond: ^f64, amax: ^f64) -> i32 ---
	LAPACKE_spoequb :: proc(matrix_layout: c.int, n: i32, a: ^f32, lda: i32, s: ^f32, scond: ^f32, amax: ^f32) -> i32 ---
	LAPACKE_dpoequb :: proc(matrix_layout: c.int, n: i32, a: ^f64, lda: i32, s: ^f64, scond: ^f64, amax: ^f64) -> i32 ---
	LAPACKE_cpoequb :: proc(matrix_layout: c.int, n: i32, a: ^complex64, lda: i32, s: ^f32, scond: ^f32, amax: ^f32) -> i32 ---
	LAPACKE_zpoequb :: proc(matrix_layout: c.int, n: i32, a: ^complex128, lda: i32, s: ^f64, scond: ^f64, amax: ^f64) -> i32 ---
	LAPACKE_sporfs :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^f32, lda: i32, af: ^f32, ldaf: i32, b: ^f32, ldb: i32, x: ^f32, ldx: i32, ferr: ^f32, berr: ^f32) -> i32 ---
	LAPACKE_dporfs :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^f64, lda: i32, af: ^f64, ldaf: i32, b: ^f64, ldb: i32, x: ^f64, ldx: i32, ferr: ^f64, berr: ^f64) -> i32 ---
	LAPACKE_cporfs :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, af: ^complex64, ldaf: i32, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, ferr: ^f32, berr: ^f32) -> i32 ---
	LAPACKE_zporfs :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, af: ^complex128, ldaf: i32, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, ferr: ^f64, berr: ^f64) -> i32 ---
	LAPACKE_sporfsx :: proc(matrix_layout: c.int, uplo: c.char, equed: c.char, n: i32, nrhs: i32, a: ^f32, lda: i32, af: ^f32, ldaf: i32, s: ^f32, b: ^f32, ldb: i32, x: ^f32, ldx: i32, rcond: ^f32, berr: ^f32, n_err_bnds: i32, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: i32, params: ^f32) -> i32 ---
	LAPACKE_dporfsx :: proc(matrix_layout: c.int, uplo: c.char, equed: c.char, n: i32, nrhs: i32, a: ^f64, lda: i32, af: ^f64, ldaf: i32, s: ^f64, b: ^f64, ldb: i32, x: ^f64, ldx: i32, rcond: ^f64, berr: ^f64, n_err_bnds: i32, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: i32, params: ^f64) -> i32 ---
	LAPACKE_cporfsx :: proc(matrix_layout: c.int, uplo: c.char, equed: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, af: ^complex64, ldaf: i32, s: ^f32, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, rcond: ^f32, berr: ^f32, n_err_bnds: i32, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: i32, params: ^f32) -> i32 ---
	LAPACKE_zporfsx :: proc(matrix_layout: c.int, uplo: c.char, equed: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, af: ^complex128, ldaf: i32, s: ^f64, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, rcond: ^f64, berr: ^f64, n_err_bnds: i32, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: i32, params: ^f64) -> i32 ---
	LAPACKE_sposv :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32) -> i32 ---
	LAPACKE_dposv :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32) -> i32 ---
	LAPACKE_cposv :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_zposv :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_dsposv :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, x: ^f64, ldx: i32, iter: ^i32) -> i32 ---
	LAPACKE_zcposv :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, iter: ^i32) -> i32 ---
	LAPACKE_sposvx :: proc(matrix_layout: c.int, fact: c.char, uplo: c.char, n: i32, nrhs: i32, a: ^f32, lda: i32, af: ^f32, ldaf: i32, equed: cstring, s: ^f32, b: ^f32, ldb: i32, x: ^f32, ldx: i32, rcond: ^f32, ferr: ^f32, berr: ^f32) -> i32 ---
	LAPACKE_dposvx :: proc(matrix_layout: c.int, fact: c.char, uplo: c.char, n: i32, nrhs: i32, a: ^f64, lda: i32, af: ^f64, ldaf: i32, equed: cstring, s: ^f64, b: ^f64, ldb: i32, x: ^f64, ldx: i32, rcond: ^f64, ferr: ^f64, berr: ^f64) -> i32 ---
	LAPACKE_cposvx :: proc(matrix_layout: c.int, fact: c.char, uplo: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, af: ^complex64, ldaf: i32, equed: cstring, s: ^f32, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, rcond: ^f32, ferr: ^f32, berr: ^f32) -> i32 ---
	LAPACKE_zposvx :: proc(matrix_layout: c.int, fact: c.char, uplo: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, af: ^complex128, ldaf: i32, equed: cstring, s: ^f64, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, rcond: ^f64, ferr: ^f64, berr: ^f64) -> i32 ---
	LAPACKE_sposvxx :: proc(matrix_layout: c.int, fact: c.char, uplo: c.char, n: i32, nrhs: i32, a: ^f32, lda: i32, af: ^f32, ldaf: i32, equed: cstring, s: ^f32, b: ^f32, ldb: i32, x: ^f32, ldx: i32, rcond: ^f32, rpvgrw: ^f32, berr: ^f32, n_err_bnds: i32, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: i32, params: ^f32) -> i32 ---
	LAPACKE_dposvxx :: proc(matrix_layout: c.int, fact: c.char, uplo: c.char, n: i32, nrhs: i32, a: ^f64, lda: i32, af: ^f64, ldaf: i32, equed: cstring, s: ^f64, b: ^f64, ldb: i32, x: ^f64, ldx: i32, rcond: ^f64, rpvgrw: ^f64, berr: ^f64, n_err_bnds: i32, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: i32, params: ^f64) -> i32 ---
	LAPACKE_cposvxx :: proc(matrix_layout: c.int, fact: c.char, uplo: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, af: ^complex64, ldaf: i32, equed: cstring, s: ^f32, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, rcond: ^f32, rpvgrw: ^f32, berr: ^f32, n_err_bnds: i32, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: i32, params: ^f32) -> i32 ---
	LAPACKE_zposvxx :: proc(matrix_layout: c.int, fact: c.char, uplo: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, af: ^complex128, ldaf: i32, equed: cstring, s: ^f64, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, rcond: ^f64, rpvgrw: ^f64, berr: ^f64, n_err_bnds: i32, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: i32, params: ^f64) -> i32 ---
	LAPACKE_spotrf2 :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f32, lda: i32) -> i32 ---
	LAPACKE_dpotrf2 :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f64, lda: i32) -> i32 ---
	LAPACKE_cpotrf2 :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32) -> i32 ---
	LAPACKE_zpotrf2 :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32) -> i32 ---
	LAPACKE_spotrf :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f32, lda: i32) -> i32 ---
	LAPACKE_dpotrf :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f64, lda: i32) -> i32 ---
	LAPACKE_cpotrf :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32) -> i32 ---
	LAPACKE_zpotrf :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32) -> i32 ---
	LAPACKE_spotri :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f32, lda: i32) -> i32 ---
	LAPACKE_dpotri :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f64, lda: i32) -> i32 ---
	LAPACKE_cpotri :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32) -> i32 ---
	LAPACKE_zpotri :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32) -> i32 ---
	LAPACKE_spotrs :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32) -> i32 ---
	LAPACKE_dpotrs :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32) -> i32 ---
	LAPACKE_cpotrs :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_zpotrs :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_sppcon :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^f32, anorm: f32, rcond: ^f32) -> i32 ---
	LAPACKE_dppcon :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^f64, anorm: f64, rcond: ^f64) -> i32 ---
	LAPACKE_cppcon :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^complex64, anorm: f32, rcond: ^f32) -> i32 ---
	LAPACKE_zppcon :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^complex128, anorm: f64, rcond: ^f64) -> i32 ---
	LAPACKE_sppequ :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^f32, s: ^f32, scond: ^f32, amax: ^f32) -> i32 ---
	LAPACKE_dppequ :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^f64, s: ^f64, scond: ^f64, amax: ^f64) -> i32 ---
	LAPACKE_cppequ :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^complex64, s: ^f32, scond: ^f32, amax: ^f32) -> i32 ---
	LAPACKE_zppequ :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^complex128, s: ^f64, scond: ^f64, amax: ^f64) -> i32 ---
	LAPACKE_spprfs :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, ap: ^f32, afp: ^f32, b: ^f32, ldb: i32, x: ^f32, ldx: i32, ferr: ^f32, berr: ^f32) -> i32 ---
	LAPACKE_dpprfs :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, ap: ^f64, afp: ^f64, b: ^f64, ldb: i32, x: ^f64, ldx: i32, ferr: ^f64, berr: ^f64) -> i32 ---
	LAPACKE_cpprfs :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, ap: ^complex64, afp: ^complex64, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, ferr: ^f32, berr: ^f32) -> i32 ---
	LAPACKE_zpprfs :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, ap: ^complex128, afp: ^complex128, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, ferr: ^f64, berr: ^f64) -> i32 ---
	LAPACKE_sppsv :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, ap: ^f32, b: ^f32, ldb: i32) -> i32 ---
	LAPACKE_dppsv :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, ap: ^f64, b: ^f64, ldb: i32) -> i32 ---
	LAPACKE_cppsv :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, ap: ^complex64, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_zppsv :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, ap: ^complex128, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_sppsvx :: proc(matrix_layout: c.int, fact: c.char, uplo: c.char, n: i32, nrhs: i32, ap: ^f32, afp: ^f32, equed: cstring, s: ^f32, b: ^f32, ldb: i32, x: ^f32, ldx: i32, rcond: ^f32, ferr: ^f32, berr: ^f32) -> i32 ---
	LAPACKE_dppsvx :: proc(matrix_layout: c.int, fact: c.char, uplo: c.char, n: i32, nrhs: i32, ap: ^f64, afp: ^f64, equed: cstring, s: ^f64, b: ^f64, ldb: i32, x: ^f64, ldx: i32, rcond: ^f64, ferr: ^f64, berr: ^f64) -> i32 ---
	LAPACKE_cppsvx :: proc(matrix_layout: c.int, fact: c.char, uplo: c.char, n: i32, nrhs: i32, ap: ^complex64, afp: ^complex64, equed: cstring, s: ^f32, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, rcond: ^f32, ferr: ^f32, berr: ^f32) -> i32 ---
	LAPACKE_zppsvx :: proc(matrix_layout: c.int, fact: c.char, uplo: c.char, n: i32, nrhs: i32, ap: ^complex128, afp: ^complex128, equed: cstring, s: ^f64, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, rcond: ^f64, ferr: ^f64, berr: ^f64) -> i32 ---
	LAPACKE_spptrf :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^f32) -> i32 ---
	LAPACKE_dpptrf :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^f64) -> i32 ---
	LAPACKE_cpptrf :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^complex64) -> i32 ---
	LAPACKE_zpptrf :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^complex128) -> i32 ---
	LAPACKE_spptri :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^f32) -> i32 ---
	LAPACKE_dpptri :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^f64) -> i32 ---
	LAPACKE_cpptri :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^complex64) -> i32 ---
	LAPACKE_zpptri :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^complex128) -> i32 ---
	LAPACKE_spptrs :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, ap: ^f32, b: ^f32, ldb: i32) -> i32 ---
	LAPACKE_dpptrs :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, ap: ^f64, b: ^f64, ldb: i32) -> i32 ---
	LAPACKE_cpptrs :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, ap: ^complex64, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_zpptrs :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, ap: ^complex128, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_spstrf :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f32, lda: i32, piv: ^i32, rank: ^i32, tol: f32) -> i32 ---
	LAPACKE_dpstrf :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f64, lda: i32, piv: ^i32, rank: ^i32, tol: f64) -> i32 ---
	LAPACKE_cpstrf :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32, piv: ^i32, rank: ^i32, tol: f32) -> i32 ---
	LAPACKE_zpstrf :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32, piv: ^i32, rank: ^i32, tol: f64) -> i32 ---
	LAPACKE_sptcon :: proc(n: i32, d: ^f32, e: ^f32, anorm: f32, rcond: ^f32) -> i32 ---
	LAPACKE_dptcon :: proc(n: i32, d: ^f64, e: ^f64, anorm: f64, rcond: ^f64) -> i32 ---
	LAPACKE_cptcon :: proc(n: i32, d: ^f32, e: ^complex64, anorm: f32, rcond: ^f32) -> i32 ---
	LAPACKE_zptcon :: proc(n: i32, d: ^f64, e: ^complex128, anorm: f64, rcond: ^f64) -> i32 ---
	LAPACKE_spteqr :: proc(matrix_layout: c.int, compz: c.char, n: i32, d: ^f32, e: ^f32, z: ^f32, ldz: i32) -> i32 ---
	LAPACKE_dpteqr :: proc(matrix_layout: c.int, compz: c.char, n: i32, d: ^f64, e: ^f64, z: ^f64, ldz: i32) -> i32 ---
	LAPACKE_cpteqr :: proc(matrix_layout: c.int, compz: c.char, n: i32, d: ^f32, e: ^f32, z: ^complex64, ldz: i32) -> i32 ---
	LAPACKE_zpteqr :: proc(matrix_layout: c.int, compz: c.char, n: i32, d: ^f64, e: ^f64, z: ^complex128, ldz: i32) -> i32 ---
	LAPACKE_sptrfs :: proc(matrix_layout: c.int, n: i32, nrhs: i32, d: ^f32, e: ^f32, df: ^f32, ef: ^f32, b: ^f32, ldb: i32, x: ^f32, ldx: i32, ferr: ^f32, berr: ^f32) -> i32 ---
	LAPACKE_dptrfs :: proc(matrix_layout: c.int, n: i32, nrhs: i32, d: ^f64, e: ^f64, df: ^f64, ef: ^f64, b: ^f64, ldb: i32, x: ^f64, ldx: i32, ferr: ^f64, berr: ^f64) -> i32 ---
	LAPACKE_cptrfs :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, d: ^f32, e: ^complex64, df: ^f32, ef: ^complex64, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, ferr: ^f32, berr: ^f32) -> i32 ---
	LAPACKE_zptrfs :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, d: ^f64, e: ^complex128, df: ^f64, ef: ^complex128, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, ferr: ^f64, berr: ^f64) -> i32 ---
	LAPACKE_sptsv :: proc(matrix_layout: c.int, n: i32, nrhs: i32, d: ^f32, e: ^f32, b: ^f32, ldb: i32) -> i32 ---
	LAPACKE_dptsv :: proc(matrix_layout: c.int, n: i32, nrhs: i32, d: ^f64, e: ^f64, b: ^f64, ldb: i32) -> i32 ---
	LAPACKE_cptsv :: proc(matrix_layout: c.int, n: i32, nrhs: i32, d: ^f32, e: ^complex64, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_zptsv :: proc(matrix_layout: c.int, n: i32, nrhs: i32, d: ^f64, e: ^complex128, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_sptsvx :: proc(matrix_layout: c.int, fact: c.char, n: i32, nrhs: i32, d: ^f32, e: ^f32, df: ^f32, ef: ^f32, b: ^f32, ldb: i32, x: ^f32, ldx: i32, rcond: ^f32, ferr: ^f32, berr: ^f32) -> i32 ---
	LAPACKE_dptsvx :: proc(matrix_layout: c.int, fact: c.char, n: i32, nrhs: i32, d: ^f64, e: ^f64, df: ^f64, ef: ^f64, b: ^f64, ldb: i32, x: ^f64, ldx: i32, rcond: ^f64, ferr: ^f64, berr: ^f64) -> i32 ---
	LAPACKE_cptsvx :: proc(matrix_layout: c.int, fact: c.char, n: i32, nrhs: i32, d: ^f32, e: ^complex64, df: ^f32, ef: ^complex64, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, rcond: ^f32, ferr: ^f32, berr: ^f32) -> i32 ---
	LAPACKE_zptsvx :: proc(matrix_layout: c.int, fact: c.char, n: i32, nrhs: i32, d: ^f64, e: ^complex128, df: ^f64, ef: ^complex128, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, rcond: ^f64, ferr: ^f64, berr: ^f64) -> i32 ---
	LAPACKE_spttrf :: proc(n: i32, d: ^f32, e: ^f32) -> i32 ---
	LAPACKE_dpttrf :: proc(n: i32, d: ^f64, e: ^f64) -> i32 ---
	LAPACKE_cpttrf :: proc(n: i32, d: ^f32, e: ^complex64) -> i32 ---
	LAPACKE_zpttrf :: proc(n: i32, d: ^f64, e: ^complex128) -> i32 ---
	LAPACKE_spttrs :: proc(matrix_layout: c.int, n: i32, nrhs: i32, d: ^f32, e: ^f32, b: ^f32, ldb: i32) -> i32 ---
	LAPACKE_dpttrs :: proc(matrix_layout: c.int, n: i32, nrhs: i32, d: ^f64, e: ^f64, b: ^f64, ldb: i32) -> i32 ---
	LAPACKE_cpttrs :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, d: ^f32, e: ^complex64, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_zpttrs :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, d: ^f64, e: ^complex128, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_ssbev :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, kd: i32, ab: ^f32, ldab: i32, w: ^f32, z: ^f32, ldz: i32) -> i32 ---
	LAPACKE_dsbev :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, kd: i32, ab: ^f64, ldab: i32, w: ^f64, z: ^f64, ldz: i32) -> i32 ---
	LAPACKE_ssbevd :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, kd: i32, ab: ^f32, ldab: i32, w: ^f32, z: ^f32, ldz: i32) -> i32 ---
	LAPACKE_dsbevd :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, kd: i32, ab: ^f64, ldab: i32, w: ^f64, z: ^f64, ldz: i32) -> i32 ---
	LAPACKE_ssbevx :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, uplo: c.char, n: i32, kd: i32, ab: ^f32, ldab: i32, q: ^f32, ldq: i32, vl: f32, vu: f32, il: i32, iu: i32, abstol: f32, m: ^i32, w: ^f32, z: ^f32, ldz: i32, ifail: ^i32) -> i32 ---
	LAPACKE_dsbevx :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, uplo: c.char, n: i32, kd: i32, ab: ^f64, ldab: i32, q: ^f64, ldq: i32, vl: f64, vu: f64, il: i32, iu: i32, abstol: f64, m: ^i32, w: ^f64, z: ^f64, ldz: i32, ifail: ^i32) -> i32 ---
	LAPACKE_ssbgst :: proc(matrix_layout: c.int, vect: c.char, uplo: c.char, n: i32, ka: i32, kb: i32, ab: ^f32, ldab: i32, bb: ^f32, ldbb: i32, x: ^f32, ldx: i32) -> i32 ---
	LAPACKE_dsbgst :: proc(matrix_layout: c.int, vect: c.char, uplo: c.char, n: i32, ka: i32, kb: i32, ab: ^f64, ldab: i32, bb: ^f64, ldbb: i32, x: ^f64, ldx: i32) -> i32 ---
	LAPACKE_ssbgv :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, ka: i32, kb: i32, ab: ^f32, ldab: i32, bb: ^f32, ldbb: i32, w: ^f32, z: ^f32, ldz: i32) -> i32 ---
	LAPACKE_dsbgv :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, ka: i32, kb: i32, ab: ^f64, ldab: i32, bb: ^f64, ldbb: i32, w: ^f64, z: ^f64, ldz: i32) -> i32 ---
	LAPACKE_ssbgvd :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, ka: i32, kb: i32, ab: ^f32, ldab: i32, bb: ^f32, ldbb: i32, w: ^f32, z: ^f32, ldz: i32) -> i32 ---
	LAPACKE_dsbgvd :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, ka: i32, kb: i32, ab: ^f64, ldab: i32, bb: ^f64, ldbb: i32, w: ^f64, z: ^f64, ldz: i32) -> i32 ---
	LAPACKE_ssbgvx :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, uplo: c.char, n: i32, ka: i32, kb: i32, ab: ^f32, ldab: i32, bb: ^f32, ldbb: i32, q: ^f32, ldq: i32, vl: f32, vu: f32, il: i32, iu: i32, abstol: f32, m: ^i32, w: ^f32, z: ^f32, ldz: i32, ifail: ^i32) -> i32 ---
	LAPACKE_dsbgvx :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, uplo: c.char, n: i32, ka: i32, kb: i32, ab: ^f64, ldab: i32, bb: ^f64, ldbb: i32, q: ^f64, ldq: i32, vl: f64, vu: f64, il: i32, iu: i32, abstol: f64, m: ^i32, w: ^f64, z: ^f64, ldz: i32, ifail: ^i32) -> i32 ---
	LAPACKE_ssbtrd :: proc(matrix_layout: c.int, vect: c.char, uplo: c.char, n: i32, kd: i32, ab: ^f32, ldab: i32, d: ^f32, e: ^f32, q: ^f32, ldq: i32) -> i32 ---
	LAPACKE_dsbtrd :: proc(matrix_layout: c.int, vect: c.char, uplo: c.char, n: i32, kd: i32, ab: ^f64, ldab: i32, d: ^f64, e: ^f64, q: ^f64, ldq: i32) -> i32 ---
	LAPACKE_ssfrk :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, trans: c.char, n: i32, k: i32, alpha: f32, a: ^f32, lda: i32, beta: f32, _c: ^f32) -> i32 ---
	LAPACKE_dsfrk :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, trans: c.char, n: i32, k: i32, alpha: f64, a: ^f64, lda: i32, beta: f64, _c: ^f64) -> i32 ---
	LAPACKE_sspcon :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^f32, ipiv: ^i32, anorm: f32, rcond: ^f32) -> i32 ---
	LAPACKE_dspcon :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^f64, ipiv: ^i32, anorm: f64, rcond: ^f64) -> i32 ---
	LAPACKE_cspcon :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^complex64, ipiv: ^i32, anorm: f32, rcond: ^f32) -> i32 ---
	LAPACKE_zspcon :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^complex128, ipiv: ^i32, anorm: f64, rcond: ^f64) -> i32 ---
	LAPACKE_sspev :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, ap: ^f32, w: ^f32, z: ^f32, ldz: i32) -> i32 ---
	LAPACKE_dspev :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, ap: ^f64, w: ^f64, z: ^f64, ldz: i32) -> i32 ---
	LAPACKE_sspevd :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, ap: ^f32, w: ^f32, z: ^f32, ldz: i32) -> i32 ---
	LAPACKE_dspevd :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, ap: ^f64, w: ^f64, z: ^f64, ldz: i32) -> i32 ---
	LAPACKE_sspevx :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, uplo: c.char, n: i32, ap: ^f32, vl: f32, vu: f32, il: i32, iu: i32, abstol: f32, m: ^i32, w: ^f32, z: ^f32, ldz: i32, ifail: ^i32) -> i32 ---
	LAPACKE_dspevx :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, uplo: c.char, n: i32, ap: ^f64, vl: f64, vu: f64, il: i32, iu: i32, abstol: f64, m: ^i32, w: ^f64, z: ^f64, ldz: i32, ifail: ^i32) -> i32 ---
	LAPACKE_sspgst :: proc(matrix_layout: c.int, itype: i32, uplo: c.char, n: i32, ap: ^f32, bp: ^f32) -> i32 ---
	LAPACKE_dspgst :: proc(matrix_layout: c.int, itype: i32, uplo: c.char, n: i32, ap: ^f64, bp: ^f64) -> i32 ---
	LAPACKE_sspgv :: proc(matrix_layout: c.int, itype: i32, jobz: c.char, uplo: c.char, n: i32, ap: ^f32, bp: ^f32, w: ^f32, z: ^f32, ldz: i32) -> i32 ---
	LAPACKE_dspgv :: proc(matrix_layout: c.int, itype: i32, jobz: c.char, uplo: c.char, n: i32, ap: ^f64, bp: ^f64, w: ^f64, z: ^f64, ldz: i32) -> i32 ---
	LAPACKE_sspgvd :: proc(matrix_layout: c.int, itype: i32, jobz: c.char, uplo: c.char, n: i32, ap: ^f32, bp: ^f32, w: ^f32, z: ^f32, ldz: i32) -> i32 ---
	LAPACKE_dspgvd :: proc(matrix_layout: c.int, itype: i32, jobz: c.char, uplo: c.char, n: i32, ap: ^f64, bp: ^f64, w: ^f64, z: ^f64, ldz: i32) -> i32 ---
	LAPACKE_sspgvx :: proc(matrix_layout: c.int, itype: i32, jobz: c.char, range: c.char, uplo: c.char, n: i32, ap: ^f32, bp: ^f32, vl: f32, vu: f32, il: i32, iu: i32, abstol: f32, m: ^i32, w: ^f32, z: ^f32, ldz: i32, ifail: ^i32) -> i32 ---
	LAPACKE_dspgvx :: proc(matrix_layout: c.int, itype: i32, jobz: c.char, range: c.char, uplo: c.char, n: i32, ap: ^f64, bp: ^f64, vl: f64, vu: f64, il: i32, iu: i32, abstol: f64, m: ^i32, w: ^f64, z: ^f64, ldz: i32, ifail: ^i32) -> i32 ---
	LAPACKE_ssprfs :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, ap: ^f32, afp: ^f32, ipiv: ^i32, b: ^f32, ldb: i32, x: ^f32, ldx: i32, ferr: ^f32, berr: ^f32) -> i32 ---
	LAPACKE_dsprfs :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, ap: ^f64, afp: ^f64, ipiv: ^i32, b: ^f64, ldb: i32, x: ^f64, ldx: i32, ferr: ^f64, berr: ^f64) -> i32 ---
	LAPACKE_csprfs :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, ap: ^complex64, afp: ^complex64, ipiv: ^i32, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, ferr: ^f32, berr: ^f32) -> i32 ---
	LAPACKE_zsprfs :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, ap: ^complex128, afp: ^complex128, ipiv: ^i32, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, ferr: ^f64, berr: ^f64) -> i32 ---
	LAPACKE_sspsv :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, ap: ^f32, ipiv: ^i32, b: ^f32, ldb: i32) -> i32 ---
	LAPACKE_dspsv :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, ap: ^f64, ipiv: ^i32, b: ^f64, ldb: i32) -> i32 ---
	LAPACKE_cspsv :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, ap: ^complex64, ipiv: ^i32, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_zspsv :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, ap: ^complex128, ipiv: ^i32, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_sspsvx :: proc(matrix_layout: c.int, fact: c.char, uplo: c.char, n: i32, nrhs: i32, ap: ^f32, afp: ^f32, ipiv: ^i32, b: ^f32, ldb: i32, x: ^f32, ldx: i32, rcond: ^f32, ferr: ^f32, berr: ^f32) -> i32 ---
	LAPACKE_dspsvx :: proc(matrix_layout: c.int, fact: c.char, uplo: c.char, n: i32, nrhs: i32, ap: ^f64, afp: ^f64, ipiv: ^i32, b: ^f64, ldb: i32, x: ^f64, ldx: i32, rcond: ^f64, ferr: ^f64, berr: ^f64) -> i32 ---
	LAPACKE_cspsvx :: proc(matrix_layout: c.int, fact: c.char, uplo: c.char, n: i32, nrhs: i32, ap: ^complex64, afp: ^complex64, ipiv: ^i32, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, rcond: ^f32, ferr: ^f32, berr: ^f32) -> i32 ---
	LAPACKE_zspsvx :: proc(matrix_layout: c.int, fact: c.char, uplo: c.char, n: i32, nrhs: i32, ap: ^complex128, afp: ^complex128, ipiv: ^i32, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, rcond: ^f64, ferr: ^f64, berr: ^f64) -> i32 ---
	LAPACKE_ssptrd :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^f32, d: ^f32, e: ^f32, tau: ^f32) -> i32 ---
	LAPACKE_dsptrd :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^f64, d: ^f64, e: ^f64, tau: ^f64) -> i32 ---
	LAPACKE_ssptrf :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^f32, ipiv: ^i32) -> i32 ---
	LAPACKE_dsptrf :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^f64, ipiv: ^i32) -> i32 ---
	LAPACKE_csptrf :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^complex64, ipiv: ^i32) -> i32 ---
	LAPACKE_zsptrf :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^complex128, ipiv: ^i32) -> i32 ---
	LAPACKE_ssptri :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^f32, ipiv: ^i32) -> i32 ---
	LAPACKE_dsptri :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^f64, ipiv: ^i32) -> i32 ---
	LAPACKE_csptri :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^complex64, ipiv: ^i32) -> i32 ---
	LAPACKE_zsptri :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^complex128, ipiv: ^i32) -> i32 ---
	LAPACKE_ssptrs :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, ap: ^f32, ipiv: ^i32, b: ^f32, ldb: i32) -> i32 ---
	LAPACKE_dsptrs :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, ap: ^f64, ipiv: ^i32, b: ^f64, ldb: i32) -> i32 ---
	LAPACKE_csptrs :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, ap: ^complex64, ipiv: ^i32, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_zsptrs :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, ap: ^complex128, ipiv: ^i32, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_sstebz :: proc(range: c.char, order: c.char, n: i32, vl: f32, vu: f32, il: i32, iu: i32, abstol: f32, d: ^f32, e: ^f32, m: ^i32, nsplit: ^i32, w: ^f32, iblock: ^i32, isplit: ^i32) -> i32 ---
	LAPACKE_dstebz :: proc(range: c.char, order: c.char, n: i32, vl: f64, vu: f64, il: i32, iu: i32, abstol: f64, d: ^f64, e: ^f64, m: ^i32, nsplit: ^i32, w: ^f64, iblock: ^i32, isplit: ^i32) -> i32 ---
	LAPACKE_sstedc :: proc(matrix_layout: c.int, compz: c.char, n: i32, d: ^f32, e: ^f32, z: ^f32, ldz: i32) -> i32 ---
	LAPACKE_dstedc :: proc(matrix_layout: c.int, compz: c.char, n: i32, d: ^f64, e: ^f64, z: ^f64, ldz: i32) -> i32 ---
	LAPACKE_cstedc :: proc(matrix_layout: c.int, compz: c.char, n: i32, d: ^f32, e: ^f32, z: ^complex64, ldz: i32) -> i32 ---
	LAPACKE_zstedc :: proc(matrix_layout: c.int, compz: c.char, n: i32, d: ^f64, e: ^f64, z: ^complex128, ldz: i32) -> i32 ---
	LAPACKE_sstegr :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, n: i32, d: ^f32, e: ^f32, vl: f32, vu: f32, il: i32, iu: i32, abstol: f32, m: ^i32, w: ^f32, z: ^f32, ldz: i32, isuppz: ^i32) -> i32 ---
	LAPACKE_dstegr :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, n: i32, d: ^f64, e: ^f64, vl: f64, vu: f64, il: i32, iu: i32, abstol: f64, m: ^i32, w: ^f64, z: ^f64, ldz: i32, isuppz: ^i32) -> i32 ---
	LAPACKE_cstegr :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, n: i32, d: ^f32, e: ^f32, vl: f32, vu: f32, il: i32, iu: i32, abstol: f32, m: ^i32, w: ^f32, z: ^complex64, ldz: i32, isuppz: ^i32) -> i32 ---
	LAPACKE_zstegr :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, n: i32, d: ^f64, e: ^f64, vl: f64, vu: f64, il: i32, iu: i32, abstol: f64, m: ^i32, w: ^f64, z: ^complex128, ldz: i32, isuppz: ^i32) -> i32 ---
	LAPACKE_sstein :: proc(matrix_layout: c.int, n: i32, d: ^f32, e: ^f32, m: i32, w: ^f32, iblock: ^i32, isplit: ^i32, z: ^f32, ldz: i32, ifailv: ^i32) -> i32 ---
	LAPACKE_dstein :: proc(matrix_layout: c.int, n: i32, d: ^f64, e: ^f64, m: i32, w: ^f64, iblock: ^i32, isplit: ^i32, z: ^f64, ldz: i32, ifailv: ^i32) -> i32 ---
	LAPACKE_cstein :: proc(matrix_layout: c.int, n: i32, d: ^f32, e: ^f32, m: i32, w: ^f32, iblock: ^i32, isplit: ^i32, z: ^complex64, ldz: i32, ifailv: ^i32) -> i32 ---
	LAPACKE_zstein :: proc(matrix_layout: c.int, n: i32, d: ^f64, e: ^f64, m: i32, w: ^f64, iblock: ^i32, isplit: ^i32, z: ^complex128, ldz: i32, ifailv: ^i32) -> i32 ---
	LAPACKE_sstemr :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, n: i32, d: ^f32, e: ^f32, vl: f32, vu: f32, il: i32, iu: i32, m: ^i32, w: ^f32, z: ^f32, ldz: i32, nzc: i32, isuppz: ^i32, tryrac: ^i32) -> i32 ---
	LAPACKE_dstemr :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, n: i32, d: ^f64, e: ^f64, vl: f64, vu: f64, il: i32, iu: i32, m: ^i32, w: ^f64, z: ^f64, ldz: i32, nzc: i32, isuppz: ^i32, tryrac: ^i32) -> i32 ---
	LAPACKE_cstemr :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, n: i32, d: ^f32, e: ^f32, vl: f32, vu: f32, il: i32, iu: i32, m: ^i32, w: ^f32, z: ^complex64, ldz: i32, nzc: i32, isuppz: ^i32, tryrac: ^i32) -> i32 ---
	LAPACKE_zstemr :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, n: i32, d: ^f64, e: ^f64, vl: f64, vu: f64, il: i32, iu: i32, m: ^i32, w: ^f64, z: ^complex128, ldz: i32, nzc: i32, isuppz: ^i32, tryrac: ^i32) -> i32 ---
	LAPACKE_ssteqr :: proc(matrix_layout: c.int, compz: c.char, n: i32, d: ^f32, e: ^f32, z: ^f32, ldz: i32) -> i32 ---
	LAPACKE_dsteqr :: proc(matrix_layout: c.int, compz: c.char, n: i32, d: ^f64, e: ^f64, z: ^f64, ldz: i32) -> i32 ---
	LAPACKE_csteqr :: proc(matrix_layout: c.int, compz: c.char, n: i32, d: ^f32, e: ^f32, z: ^complex64, ldz: i32) -> i32 ---
	LAPACKE_zsteqr :: proc(matrix_layout: c.int, compz: c.char, n: i32, d: ^f64, e: ^f64, z: ^complex128, ldz: i32) -> i32 ---
	LAPACKE_ssterf :: proc(n: i32, d: ^f32, e: ^f32) -> i32 ---
	LAPACKE_dsterf :: proc(n: i32, d: ^f64, e: ^f64) -> i32 ---
	LAPACKE_sstev :: proc(matrix_layout: c.int, jobz: c.char, n: i32, d: ^f32, e: ^f32, z: ^f32, ldz: i32) -> i32 ---
	LAPACKE_dstev :: proc(matrix_layout: c.int, jobz: c.char, n: i32, d: ^f64, e: ^f64, z: ^f64, ldz: i32) -> i32 ---
	LAPACKE_sstevd :: proc(matrix_layout: c.int, jobz: c.char, n: i32, d: ^f32, e: ^f32, z: ^f32, ldz: i32) -> i32 ---
	LAPACKE_dstevd :: proc(matrix_layout: c.int, jobz: c.char, n: i32, d: ^f64, e: ^f64, z: ^f64, ldz: i32) -> i32 ---
	LAPACKE_sstevr :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, n: i32, d: ^f32, e: ^f32, vl: f32, vu: f32, il: i32, iu: i32, abstol: f32, m: ^i32, w: ^f32, z: ^f32, ldz: i32, isuppz: ^i32) -> i32 ---
	LAPACKE_dstevr :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, n: i32, d: ^f64, e: ^f64, vl: f64, vu: f64, il: i32, iu: i32, abstol: f64, m: ^i32, w: ^f64, z: ^f64, ldz: i32, isuppz: ^i32) -> i32 ---
	LAPACKE_sstevx :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, n: i32, d: ^f32, e: ^f32, vl: f32, vu: f32, il: i32, iu: i32, abstol: f32, m: ^i32, w: ^f32, z: ^f32, ldz: i32, ifail: ^i32) -> i32 ---
	LAPACKE_dstevx :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, n: i32, d: ^f64, e: ^f64, vl: f64, vu: f64, il: i32, iu: i32, abstol: f64, m: ^i32, w: ^f64, z: ^f64, ldz: i32, ifail: ^i32) -> i32 ---
	LAPACKE_ssycon :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f32, lda: i32, ipiv: ^i32, anorm: f32, rcond: ^f32) -> i32 ---
	LAPACKE_dsycon :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f64, lda: i32, ipiv: ^i32, anorm: f64, rcond: ^f64) -> i32 ---
	LAPACKE_csycon :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32, ipiv: ^i32, anorm: f32, rcond: ^f32) -> i32 ---
	LAPACKE_zsycon :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32, ipiv: ^i32, anorm: f64, rcond: ^f64) -> i32 ---
	LAPACKE_ssyequb :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f32, lda: i32, s: ^f32, scond: ^f32, amax: ^f32) -> i32 ---
	LAPACKE_dsyequb :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f64, lda: i32, s: ^f64, scond: ^f64, amax: ^f64) -> i32 ---
	LAPACKE_csyequb :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32, s: ^f32, scond: ^f32, amax: ^f32) -> i32 ---
	LAPACKE_zsyequb :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32, s: ^f64, scond: ^f64, amax: ^f64) -> i32 ---
	LAPACKE_ssyev :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, a: ^f32, lda: i32, w: ^f32) -> i32 ---
	LAPACKE_dsyev :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, a: ^f64, lda: i32, w: ^f64) -> i32 ---
	LAPACKE_ssyevd :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, a: ^f32, lda: i32, w: ^f32) -> i32 ---
	LAPACKE_dsyevd :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, a: ^f64, lda: i32, w: ^f64) -> i32 ---
	LAPACKE_ssyevr :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, uplo: c.char, n: i32, a: ^f32, lda: i32, vl: f32, vu: f32, il: i32, iu: i32, abstol: f32, m: ^i32, w: ^f32, z: ^f32, ldz: i32, isuppz: ^i32) -> i32 ---
	LAPACKE_dsyevr :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, uplo: c.char, n: i32, a: ^f64, lda: i32, vl: f64, vu: f64, il: i32, iu: i32, abstol: f64, m: ^i32, w: ^f64, z: ^f64, ldz: i32, isuppz: ^i32) -> i32 ---
	LAPACKE_ssyevx :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, uplo: c.char, n: i32, a: ^f32, lda: i32, vl: f32, vu: f32, il: i32, iu: i32, abstol: f32, m: ^i32, w: ^f32, z: ^f32, ldz: i32, ifail: ^i32) -> i32 ---
	LAPACKE_dsyevx :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, uplo: c.char, n: i32, a: ^f64, lda: i32, vl: f64, vu: f64, il: i32, iu: i32, abstol: f64, m: ^i32, w: ^f64, z: ^f64, ldz: i32, ifail: ^i32) -> i32 ---
	LAPACKE_ssygst :: proc(matrix_layout: c.int, itype: i32, uplo: c.char, n: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32) -> i32 ---
	LAPACKE_dsygst :: proc(matrix_layout: c.int, itype: i32, uplo: c.char, n: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32) -> i32 ---
	LAPACKE_ssygv :: proc(matrix_layout: c.int, itype: i32, jobz: c.char, uplo: c.char, n: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32, w: ^f32) -> i32 ---
	LAPACKE_dsygv :: proc(matrix_layout: c.int, itype: i32, jobz: c.char, uplo: c.char, n: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, w: ^f64) -> i32 ---
	LAPACKE_ssygvd :: proc(matrix_layout: c.int, itype: i32, jobz: c.char, uplo: c.char, n: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32, w: ^f32) -> i32 ---
	LAPACKE_dsygvd :: proc(matrix_layout: c.int, itype: i32, jobz: c.char, uplo: c.char, n: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, w: ^f64) -> i32 ---
	LAPACKE_ssygvx :: proc(matrix_layout: c.int, itype: i32, jobz: c.char, range: c.char, uplo: c.char, n: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32, vl: f32, vu: f32, il: i32, iu: i32, abstol: f32, m: ^i32, w: ^f32, z: ^f32, ldz: i32, ifail: ^i32) -> i32 ---
	LAPACKE_dsygvx :: proc(matrix_layout: c.int, itype: i32, jobz: c.char, range: c.char, uplo: c.char, n: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, vl: f64, vu: f64, il: i32, iu: i32, abstol: f64, m: ^i32, w: ^f64, z: ^f64, ldz: i32, ifail: ^i32) -> i32 ---
	LAPACKE_ssyrfs :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^f32, lda: i32, af: ^f32, ldaf: i32, ipiv: ^i32, b: ^f32, ldb: i32, x: ^f32, ldx: i32, ferr: ^f32, berr: ^f32) -> i32 ---
	LAPACKE_dsyrfs :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^f64, lda: i32, af: ^f64, ldaf: i32, ipiv: ^i32, b: ^f64, ldb: i32, x: ^f64, ldx: i32, ferr: ^f64, berr: ^f64) -> i32 ---
	LAPACKE_csyrfs :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, af: ^complex64, ldaf: i32, ipiv: ^i32, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, ferr: ^f32, berr: ^f32) -> i32 ---
	LAPACKE_zsyrfs :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, af: ^complex128, ldaf: i32, ipiv: ^i32, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, ferr: ^f64, berr: ^f64) -> i32 ---
	LAPACKE_ssyrfsx :: proc(matrix_layout: c.int, uplo: c.char, equed: c.char, n: i32, nrhs: i32, a: ^f32, lda: i32, af: ^f32, ldaf: i32, ipiv: ^i32, s: ^f32, b: ^f32, ldb: i32, x: ^f32, ldx: i32, rcond: ^f32, berr: ^f32, n_err_bnds: i32, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: i32, params: ^f32) -> i32 ---
	LAPACKE_dsyrfsx :: proc(matrix_layout: c.int, uplo: c.char, equed: c.char, n: i32, nrhs: i32, a: ^f64, lda: i32, af: ^f64, ldaf: i32, ipiv: ^i32, s: ^f64, b: ^f64, ldb: i32, x: ^f64, ldx: i32, rcond: ^f64, berr: ^f64, n_err_bnds: i32, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: i32, params: ^f64) -> i32 ---
	LAPACKE_csyrfsx :: proc(matrix_layout: c.int, uplo: c.char, equed: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, af: ^complex64, ldaf: i32, ipiv: ^i32, s: ^f32, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, rcond: ^f32, berr: ^f32, n_err_bnds: i32, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: i32, params: ^f32) -> i32 ---
	LAPACKE_zsyrfsx :: proc(matrix_layout: c.int, uplo: c.char, equed: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, af: ^complex128, ldaf: i32, ipiv: ^i32, s: ^f64, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, rcond: ^f64, berr: ^f64, n_err_bnds: i32, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: i32, params: ^f64) -> i32 ---
	LAPACKE_ssysv :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^f32, lda: i32, ipiv: ^i32, b: ^f32, ldb: i32) -> i32 ---
	LAPACKE_dsysv :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^f64, lda: i32, ipiv: ^i32, b: ^f64, ldb: i32) -> i32 ---
	LAPACKE_csysv :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, ipiv: ^i32, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_zsysv :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, ipiv: ^i32, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_ssysvx :: proc(matrix_layout: c.int, fact: c.char, uplo: c.char, n: i32, nrhs: i32, a: ^f32, lda: i32, af: ^f32, ldaf: i32, ipiv: ^i32, b: ^f32, ldb: i32, x: ^f32, ldx: i32, rcond: ^f32, ferr: ^f32, berr: ^f32) -> i32 ---
	LAPACKE_dsysvx :: proc(matrix_layout: c.int, fact: c.char, uplo: c.char, n: i32, nrhs: i32, a: ^f64, lda: i32, af: ^f64, ldaf: i32, ipiv: ^i32, b: ^f64, ldb: i32, x: ^f64, ldx: i32, rcond: ^f64, ferr: ^f64, berr: ^f64) -> i32 ---
	LAPACKE_csysvx :: proc(matrix_layout: c.int, fact: c.char, uplo: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, af: ^complex64, ldaf: i32, ipiv: ^i32, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, rcond: ^f32, ferr: ^f32, berr: ^f32) -> i32 ---
	LAPACKE_zsysvx :: proc(matrix_layout: c.int, fact: c.char, uplo: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, af: ^complex128, ldaf: i32, ipiv: ^i32, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, rcond: ^f64, ferr: ^f64, berr: ^f64) -> i32 ---
	LAPACKE_ssysvxx :: proc(matrix_layout: c.int, fact: c.char, uplo: c.char, n: i32, nrhs: i32, a: ^f32, lda: i32, af: ^f32, ldaf: i32, ipiv: ^i32, equed: cstring, s: ^f32, b: ^f32, ldb: i32, x: ^f32, ldx: i32, rcond: ^f32, rpvgrw: ^f32, berr: ^f32, n_err_bnds: i32, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: i32, params: ^f32) -> i32 ---
	LAPACKE_dsysvxx :: proc(matrix_layout: c.int, fact: c.char, uplo: c.char, n: i32, nrhs: i32, a: ^f64, lda: i32, af: ^f64, ldaf: i32, ipiv: ^i32, equed: cstring, s: ^f64, b: ^f64, ldb: i32, x: ^f64, ldx: i32, rcond: ^f64, rpvgrw: ^f64, berr: ^f64, n_err_bnds: i32, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: i32, params: ^f64) -> i32 ---
	LAPACKE_csysvxx :: proc(matrix_layout: c.int, fact: c.char, uplo: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, af: ^complex64, ldaf: i32, ipiv: ^i32, equed: cstring, s: ^f32, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, rcond: ^f32, rpvgrw: ^f32, berr: ^f32, n_err_bnds: i32, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: i32, params: ^f32) -> i32 ---
	LAPACKE_zsysvxx :: proc(matrix_layout: c.int, fact: c.char, uplo: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, af: ^complex128, ldaf: i32, ipiv: ^i32, equed: cstring, s: ^f64, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, rcond: ^f64, rpvgrw: ^f64, berr: ^f64, n_err_bnds: i32, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: i32, params: ^f64) -> i32 ---
	LAPACKE_ssytrd :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f32, lda: i32, d: ^f32, e: ^f32, tau: ^f32) -> i32 ---
	LAPACKE_dsytrd :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f64, lda: i32, d: ^f64, e: ^f64, tau: ^f64) -> i32 ---
	LAPACKE_ssytrf :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f32, lda: i32, ipiv: ^i32) -> i32 ---
	LAPACKE_dsytrf :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f64, lda: i32, ipiv: ^i32) -> i32 ---
	LAPACKE_csytrf :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32, ipiv: ^i32) -> i32 ---
	LAPACKE_zsytrf :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32, ipiv: ^i32) -> i32 ---
	LAPACKE_ssytri :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f32, lda: i32, ipiv: ^i32) -> i32 ---
	LAPACKE_dsytri :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f64, lda: i32, ipiv: ^i32) -> i32 ---
	LAPACKE_csytri :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32, ipiv: ^i32) -> i32 ---
	LAPACKE_zsytri :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32, ipiv: ^i32) -> i32 ---
	LAPACKE_ssytrs :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^f32, lda: i32, ipiv: ^i32, b: ^f32, ldb: i32) -> i32 ---
	LAPACKE_dsytrs :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^f64, lda: i32, ipiv: ^i32, b: ^f64, ldb: i32) -> i32 ---
	LAPACKE_csytrs :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, ipiv: ^i32, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_zsytrs :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, ipiv: ^i32, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_stbcon :: proc(matrix_layout: c.int, norm: c.char, uplo: c.char, diag: c.char, n: i32, kd: i32, ab: ^f32, ldab: i32, rcond: ^f32) -> i32 ---
	LAPACKE_dtbcon :: proc(matrix_layout: c.int, norm: c.char, uplo: c.char, diag: c.char, n: i32, kd: i32, ab: ^f64, ldab: i32, rcond: ^f64) -> i32 ---
	LAPACKE_ctbcon :: proc(matrix_layout: c.int, norm: c.char, uplo: c.char, diag: c.char, n: i32, kd: i32, ab: ^complex64, ldab: i32, rcond: ^f32) -> i32 ---
	LAPACKE_ztbcon :: proc(matrix_layout: c.int, norm: c.char, uplo: c.char, diag: c.char, n: i32, kd: i32, ab: ^complex128, ldab: i32, rcond: ^f64) -> i32 ---
	LAPACKE_stbrfs :: proc(matrix_layout: c.int, uplo: c.char, trans: c.char, diag: c.char, n: i32, kd: i32, nrhs: i32, ab: ^f32, ldab: i32, b: ^f32, ldb: i32, x: ^f32, ldx: i32, ferr: ^f32, berr: ^f32) -> i32 ---
	LAPACKE_dtbrfs :: proc(matrix_layout: c.int, uplo: c.char, trans: c.char, diag: c.char, n: i32, kd: i32, nrhs: i32, ab: ^f64, ldab: i32, b: ^f64, ldb: i32, x: ^f64, ldx: i32, ferr: ^f64, berr: ^f64) -> i32 ---
	LAPACKE_ctbrfs :: proc(matrix_layout: c.int, uplo: c.char, trans: c.char, diag: c.char, n: i32, kd: i32, nrhs: i32, ab: ^complex64, ldab: i32, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, ferr: ^f32, berr: ^f32) -> i32 ---
	LAPACKE_ztbrfs :: proc(matrix_layout: c.int, uplo: c.char, trans: c.char, diag: c.char, n: i32, kd: i32, nrhs: i32, ab: ^complex128, ldab: i32, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, ferr: ^f64, berr: ^f64) -> i32 ---
	LAPACKE_stbtrs :: proc(matrix_layout: c.int, uplo: c.char, trans: c.char, diag: c.char, n: i32, kd: i32, nrhs: i32, ab: ^f32, ldab: i32, b: ^f32, ldb: i32) -> i32 ---
	LAPACKE_dtbtrs :: proc(matrix_layout: c.int, uplo: c.char, trans: c.char, diag: c.char, n: i32, kd: i32, nrhs: i32, ab: ^f64, ldab: i32, b: ^f64, ldb: i32) -> i32 ---
	LAPACKE_ctbtrs :: proc(matrix_layout: c.int, uplo: c.char, trans: c.char, diag: c.char, n: i32, kd: i32, nrhs: i32, ab: ^complex64, ldab: i32, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_ztbtrs :: proc(matrix_layout: c.int, uplo: c.char, trans: c.char, diag: c.char, n: i32, kd: i32, nrhs: i32, ab: ^complex128, ldab: i32, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_stfsm :: proc(matrix_layout: c.int, transr: c.char, side: c.char, uplo: c.char, trans: c.char, diag: c.char, m: i32, n: i32, alpha: f32, a: ^f32, b: ^f32, ldb: i32) -> i32 ---
	LAPACKE_dtfsm :: proc(matrix_layout: c.int, transr: c.char, side: c.char, uplo: c.char, trans: c.char, diag: c.char, m: i32, n: i32, alpha: f64, a: ^f64, b: ^f64, ldb: i32) -> i32 ---
	LAPACKE_ctfsm :: proc(matrix_layout: c.int, transr: c.char, side: c.char, uplo: c.char, trans: c.char, diag: c.char, m: i32, n: i32, alpha: complex64, a: ^complex64, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_ztfsm :: proc(matrix_layout: c.int, transr: c.char, side: c.char, uplo: c.char, trans: c.char, diag: c.char, m: i32, n: i32, alpha: complex128, a: ^complex128, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_stftri :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, diag: c.char, n: i32, a: ^f32) -> i32 ---
	LAPACKE_dtftri :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, diag: c.char, n: i32, a: ^f64) -> i32 ---
	LAPACKE_ctftri :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, diag: c.char, n: i32, a: ^complex64) -> i32 ---
	LAPACKE_ztftri :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, diag: c.char, n: i32, a: ^complex128) -> i32 ---
	LAPACKE_stfttp :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, n: i32, arf: ^f32, ap: ^f32) -> i32 ---
	LAPACKE_dtfttp :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, n: i32, arf: ^f64, ap: ^f64) -> i32 ---
	LAPACKE_ctfttp :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, n: i32, arf: ^complex64, ap: ^complex64) -> i32 ---
	LAPACKE_ztfttp :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, n: i32, arf: ^complex128, ap: ^complex128) -> i32 ---
	LAPACKE_stfttr :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, n: i32, arf: ^f32, a: ^f32, lda: i32) -> i32 ---
	LAPACKE_dtfttr :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, n: i32, arf: ^f64, a: ^f64, lda: i32) -> i32 ---
	LAPACKE_ctfttr :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, n: i32, arf: ^complex64, a: ^complex64, lda: i32) -> i32 ---
	LAPACKE_ztfttr :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, n: i32, arf: ^complex128, a: ^complex128, lda: i32) -> i32 ---
	LAPACKE_stgevc :: proc(matrix_layout: c.int, side: c.char, howmny: c.char, select: ^i32, n: i32, s: ^f32, lds: i32, p: ^f32, ldp: i32, vl: ^f32, ldvl: i32, vr: ^f32, ldvr: i32, mm: i32, m: ^i32) -> i32 ---
	LAPACKE_dtgevc :: proc(matrix_layout: c.int, side: c.char, howmny: c.char, select: ^i32, n: i32, s: ^f64, lds: i32, p: ^f64, ldp: i32, vl: ^f64, ldvl: i32, vr: ^f64, ldvr: i32, mm: i32, m: ^i32) -> i32 ---
	LAPACKE_ctgevc :: proc(matrix_layout: c.int, side: c.char, howmny: c.char, select: ^i32, n: i32, s: ^complex64, lds: i32, p: ^complex64, ldp: i32, vl: ^complex64, ldvl: i32, vr: ^complex64, ldvr: i32, mm: i32, m: ^i32) -> i32 ---
	LAPACKE_ztgevc :: proc(matrix_layout: c.int, side: c.char, howmny: c.char, select: ^i32, n: i32, s: ^complex128, lds: i32, p: ^complex128, ldp: i32, vl: ^complex128, ldvl: i32, vr: ^complex128, ldvr: i32, mm: i32, m: ^i32) -> i32 ---
	LAPACKE_stgexc :: proc(matrix_layout: c.int, wantq: i32, wantz: i32, n: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32, q: ^f32, ldq: i32, z: ^f32, ldz: i32, ifst: ^i32, ilst: ^i32) -> i32 ---
	LAPACKE_dtgexc :: proc(matrix_layout: c.int, wantq: i32, wantz: i32, n: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, q: ^f64, ldq: i32, z: ^f64, ldz: i32, ifst: ^i32, ilst: ^i32) -> i32 ---
	LAPACKE_ctgexc :: proc(matrix_layout: c.int, wantq: i32, wantz: i32, n: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32, q: ^complex64, ldq: i32, z: ^complex64, ldz: i32, ifst: i32, ilst: i32) -> i32 ---
	LAPACKE_ztgexc :: proc(matrix_layout: c.int, wantq: i32, wantz: i32, n: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, q: ^complex128, ldq: i32, z: ^complex128, ldz: i32, ifst: i32, ilst: i32) -> i32 ---
	LAPACKE_stgsen :: proc(matrix_layout: c.int, ijob: i32, wantq: i32, wantz: i32, select: ^i32, n: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32, alphar: ^f32, alphai: ^f32, beta: ^f32, q: ^f32, ldq: i32, z: ^f32, ldz: i32, m: ^i32, pl: ^f32, pr: ^f32, dif: ^f32) -> i32 ---
	LAPACKE_dtgsen :: proc(matrix_layout: c.int, ijob: i32, wantq: i32, wantz: i32, select: ^i32, n: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, alphar: ^f64, alphai: ^f64, beta: ^f64, q: ^f64, ldq: i32, z: ^f64, ldz: i32, m: ^i32, pl: ^f64, pr: ^f64, dif: ^f64) -> i32 ---
	LAPACKE_ctgsen :: proc(matrix_layout: c.int, ijob: i32, wantq: i32, wantz: i32, select: ^i32, n: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32, alpha: ^complex64, beta: ^complex64, q: ^complex64, ldq: i32, z: ^complex64, ldz: i32, m: ^i32, pl: ^f32, pr: ^f32, dif: ^f32) -> i32 ---
	LAPACKE_ztgsen :: proc(matrix_layout: c.int, ijob: i32, wantq: i32, wantz: i32, select: ^i32, n: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, alpha: ^complex128, beta: ^complex128, q: ^complex128, ldq: i32, z: ^complex128, ldz: i32, m: ^i32, pl: ^f64, pr: ^f64, dif: ^f64) -> i32 ---
	LAPACKE_stgsja :: proc(matrix_layout: c.int, jobu: c.char, jobv: c.char, jobq: c.char, m: i32, p: i32, n: i32, k: i32, l: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32, tola: f32, tolb: f32, alpha: ^f32, beta: ^f32, u: ^f32, ldu: i32, v: ^f32, ldv: i32, q: ^f32, ldq: i32, ncycle: ^i32) -> i32 ---
	LAPACKE_dtgsja :: proc(matrix_layout: c.int, jobu: c.char, jobv: c.char, jobq: c.char, m: i32, p: i32, n: i32, k: i32, l: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, tola: f64, tolb: f64, alpha: ^f64, beta: ^f64, u: ^f64, ldu: i32, v: ^f64, ldv: i32, q: ^f64, ldq: i32, ncycle: ^i32) -> i32 ---
	LAPACKE_ctgsja :: proc(matrix_layout: c.int, jobu: c.char, jobv: c.char, jobq: c.char, m: i32, p: i32, n: i32, k: i32, l: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32, tola: f32, tolb: f32, alpha: ^f32, beta: ^f32, u: ^complex64, ldu: i32, v: ^complex64, ldv: i32, q: ^complex64, ldq: i32, ncycle: ^i32) -> i32 ---
	LAPACKE_ztgsja :: proc(matrix_layout: c.int, jobu: c.char, jobv: c.char, jobq: c.char, m: i32, p: i32, n: i32, k: i32, l: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, tola: f64, tolb: f64, alpha: ^f64, beta: ^f64, u: ^complex128, ldu: i32, v: ^complex128, ldv: i32, q: ^complex128, ldq: i32, ncycle: ^i32) -> i32 ---
	LAPACKE_stgsna :: proc(matrix_layout: c.int, job: c.char, howmny: c.char, select: ^i32, n: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32, vl: ^f32, ldvl: i32, vr: ^f32, ldvr: i32, s: ^f32, dif: ^f32, mm: i32, m: ^i32) -> i32 ---
	LAPACKE_dtgsna :: proc(matrix_layout: c.int, job: c.char, howmny: c.char, select: ^i32, n: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, vl: ^f64, ldvl: i32, vr: ^f64, ldvr: i32, s: ^f64, dif: ^f64, mm: i32, m: ^i32) -> i32 ---
	LAPACKE_ctgsna :: proc(matrix_layout: c.int, job: c.char, howmny: c.char, select: ^i32, n: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32, vl: ^complex64, ldvl: i32, vr: ^complex64, ldvr: i32, s: ^f32, dif: ^f32, mm: i32, m: ^i32) -> i32 ---
	LAPACKE_ztgsna :: proc(matrix_layout: c.int, job: c.char, howmny: c.char, select: ^i32, n: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, vl: ^complex128, ldvl: i32, vr: ^complex128, ldvr: i32, s: ^f64, dif: ^f64, mm: i32, m: ^i32) -> i32 ---
	LAPACKE_stgsyl :: proc(matrix_layout: c.int, trans: c.char, ijob: i32, m: i32, n: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32, _c: ^f32, ldc: i32, d: ^f32, ldd: i32, e: ^f32, lde: i32, f: ^f32, ldf: i32, scale: ^f32, dif: ^f32) -> i32 ---
	LAPACKE_dtgsyl :: proc(matrix_layout: c.int, trans: c.char, ijob: i32, m: i32, n: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, _c: ^f64, ldc: i32, d: ^f64, ldd: i32, e: ^f64, lde: i32, f: ^f64, ldf: i32, scale: ^f64, dif: ^f64) -> i32 ---
	LAPACKE_ctgsyl :: proc(matrix_layout: c.int, trans: c.char, ijob: i32, m: i32, n: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32, _c: ^complex64, ldc: i32, d: ^complex64, ldd: i32, e: ^complex64, lde: i32, f: ^complex64, ldf: i32, scale: ^f32, dif: ^f32) -> i32 ---
	LAPACKE_ztgsyl :: proc(matrix_layout: c.int, trans: c.char, ijob: i32, m: i32, n: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, _c: ^complex128, ldc: i32, d: ^complex128, ldd: i32, e: ^complex128, lde: i32, f: ^complex128, ldf: i32, scale: ^f64, dif: ^f64) -> i32 ---
	LAPACKE_stpcon :: proc(matrix_layout: c.int, norm: c.char, uplo: c.char, diag: c.char, n: i32, ap: ^f32, rcond: ^f32) -> i32 ---
	LAPACKE_dtpcon :: proc(matrix_layout: c.int, norm: c.char, uplo: c.char, diag: c.char, n: i32, ap: ^f64, rcond: ^f64) -> i32 ---
	LAPACKE_ctpcon :: proc(matrix_layout: c.int, norm: c.char, uplo: c.char, diag: c.char, n: i32, ap: ^complex64, rcond: ^f32) -> i32 ---
	LAPACKE_ztpcon :: proc(matrix_layout: c.int, norm: c.char, uplo: c.char, diag: c.char, n: i32, ap: ^complex128, rcond: ^f64) -> i32 ---
	LAPACKE_stprfs :: proc(matrix_layout: c.int, uplo: c.char, trans: c.char, diag: c.char, n: i32, nrhs: i32, ap: ^f32, b: ^f32, ldb: i32, x: ^f32, ldx: i32, ferr: ^f32, berr: ^f32) -> i32 ---
	LAPACKE_dtprfs :: proc(matrix_layout: c.int, uplo: c.char, trans: c.char, diag: c.char, n: i32, nrhs: i32, ap: ^f64, b: ^f64, ldb: i32, x: ^f64, ldx: i32, ferr: ^f64, berr: ^f64) -> i32 ---
	LAPACKE_ctprfs :: proc(matrix_layout: c.int, uplo: c.char, trans: c.char, diag: c.char, n: i32, nrhs: i32, ap: ^complex64, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, ferr: ^f32, berr: ^f32) -> i32 ---
	LAPACKE_ztprfs :: proc(matrix_layout: c.int, uplo: c.char, trans: c.char, diag: c.char, n: i32, nrhs: i32, ap: ^complex128, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, ferr: ^f64, berr: ^f64) -> i32 ---
	LAPACKE_stptri :: proc(matrix_layout: c.int, uplo: c.char, diag: c.char, n: i32, ap: ^f32) -> i32 ---
	LAPACKE_dtptri :: proc(matrix_layout: c.int, uplo: c.char, diag: c.char, n: i32, ap: ^f64) -> i32 ---
	LAPACKE_ctptri :: proc(matrix_layout: c.int, uplo: c.char, diag: c.char, n: i32, ap: ^complex64) -> i32 ---
	LAPACKE_ztptri :: proc(matrix_layout: c.int, uplo: c.char, diag: c.char, n: i32, ap: ^complex128) -> i32 ---
	LAPACKE_stptrs :: proc(matrix_layout: c.int, uplo: c.char, trans: c.char, diag: c.char, n: i32, nrhs: i32, ap: ^f32, b: ^f32, ldb: i32) -> i32 ---
	LAPACKE_dtptrs :: proc(matrix_layout: c.int, uplo: c.char, trans: c.char, diag: c.char, n: i32, nrhs: i32, ap: ^f64, b: ^f64, ldb: i32) -> i32 ---
	LAPACKE_ctptrs :: proc(matrix_layout: c.int, uplo: c.char, trans: c.char, diag: c.char, n: i32, nrhs: i32, ap: ^complex64, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_ztptrs :: proc(matrix_layout: c.int, uplo: c.char, trans: c.char, diag: c.char, n: i32, nrhs: i32, ap: ^complex128, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_stpttf :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, n: i32, ap: ^f32, arf: ^f32) -> i32 ---
	LAPACKE_dtpttf :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, n: i32, ap: ^f64, arf: ^f64) -> i32 ---
	LAPACKE_ctpttf :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, n: i32, ap: ^complex64, arf: ^complex64) -> i32 ---
	LAPACKE_ztpttf :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, n: i32, ap: ^complex128, arf: ^complex128) -> i32 ---
	LAPACKE_stpttr :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^f32, a: ^f32, lda: i32) -> i32 ---
	LAPACKE_dtpttr :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^f64, a: ^f64, lda: i32) -> i32 ---
	LAPACKE_ctpttr :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^complex64, a: ^complex64, lda: i32) -> i32 ---
	LAPACKE_ztpttr :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^complex128, a: ^complex128, lda: i32) -> i32 ---
	LAPACKE_strcon :: proc(matrix_layout: c.int, norm: c.char, uplo: c.char, diag: c.char, n: i32, a: ^f32, lda: i32, rcond: ^f32) -> i32 ---
	LAPACKE_dtrcon :: proc(matrix_layout: c.int, norm: c.char, uplo: c.char, diag: c.char, n: i32, a: ^f64, lda: i32, rcond: ^f64) -> i32 ---
	LAPACKE_ctrcon :: proc(matrix_layout: c.int, norm: c.char, uplo: c.char, diag: c.char, n: i32, a: ^complex64, lda: i32, rcond: ^f32) -> i32 ---
	LAPACKE_ztrcon :: proc(matrix_layout: c.int, norm: c.char, uplo: c.char, diag: c.char, n: i32, a: ^complex128, lda: i32, rcond: ^f64) -> i32 ---
	LAPACKE_strevc :: proc(matrix_layout: c.int, side: c.char, howmny: c.char, select: ^i32, n: i32, t: ^f32, ldt: i32, vl: ^f32, ldvl: i32, vr: ^f32, ldvr: i32, mm: i32, m: ^i32) -> i32 ---
	LAPACKE_dtrevc :: proc(matrix_layout: c.int, side: c.char, howmny: c.char, select: ^i32, n: i32, t: ^f64, ldt: i32, vl: ^f64, ldvl: i32, vr: ^f64, ldvr: i32, mm: i32, m: ^i32) -> i32 ---
	LAPACKE_ctrevc :: proc(matrix_layout: c.int, side: c.char, howmny: c.char, select: ^i32, n: i32, t: ^complex64, ldt: i32, vl: ^complex64, ldvl: i32, vr: ^complex64, ldvr: i32, mm: i32, m: ^i32) -> i32 ---
	LAPACKE_ztrevc :: proc(matrix_layout: c.int, side: c.char, howmny: c.char, select: ^i32, n: i32, t: ^complex128, ldt: i32, vl: ^complex128, ldvl: i32, vr: ^complex128, ldvr: i32, mm: i32, m: ^i32) -> i32 ---
	LAPACKE_strexc :: proc(matrix_layout: c.int, compq: c.char, n: i32, t: ^f32, ldt: i32, q: ^f32, ldq: i32, ifst: ^i32, ilst: ^i32) -> i32 ---
	LAPACKE_dtrexc :: proc(matrix_layout: c.int, compq: c.char, n: i32, t: ^f64, ldt: i32, q: ^f64, ldq: i32, ifst: ^i32, ilst: ^i32) -> i32 ---
	LAPACKE_ctrexc :: proc(matrix_layout: c.int, compq: c.char, n: i32, t: ^complex64, ldt: i32, q: ^complex64, ldq: i32, ifst: i32, ilst: i32) -> i32 ---
	LAPACKE_ztrexc :: proc(matrix_layout: c.int, compq: c.char, n: i32, t: ^complex128, ldt: i32, q: ^complex128, ldq: i32, ifst: i32, ilst: i32) -> i32 ---
	LAPACKE_strrfs :: proc(matrix_layout: c.int, uplo: c.char, trans: c.char, diag: c.char, n: i32, nrhs: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32, x: ^f32, ldx: i32, ferr: ^f32, berr: ^f32) -> i32 ---
	LAPACKE_dtrrfs :: proc(matrix_layout: c.int, uplo: c.char, trans: c.char, diag: c.char, n: i32, nrhs: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, x: ^f64, ldx: i32, ferr: ^f64, berr: ^f64) -> i32 ---
	LAPACKE_ctrrfs :: proc(matrix_layout: c.int, uplo: c.char, trans: c.char, diag: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, ferr: ^f32, berr: ^f32) -> i32 ---
	LAPACKE_ztrrfs :: proc(matrix_layout: c.int, uplo: c.char, trans: c.char, diag: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, ferr: ^f64, berr: ^f64) -> i32 ---
	LAPACKE_strsen :: proc(matrix_layout: c.int, job: c.char, compq: c.char, select: ^i32, n: i32, t: ^f32, ldt: i32, q: ^f32, ldq: i32, wr: ^f32, wi: ^f32, m: ^i32, s: ^f32, sep: ^f32) -> i32 ---
	LAPACKE_dtrsen :: proc(matrix_layout: c.int, job: c.char, compq: c.char, select: ^i32, n: i32, t: ^f64, ldt: i32, q: ^f64, ldq: i32, wr: ^f64, wi: ^f64, m: ^i32, s: ^f64, sep: ^f64) -> i32 ---
	LAPACKE_ctrsen :: proc(matrix_layout: c.int, job: c.char, compq: c.char, select: ^i32, n: i32, t: ^complex64, ldt: i32, q: ^complex64, ldq: i32, w: ^complex64, m: ^i32, s: ^f32, sep: ^f32) -> i32 ---
	LAPACKE_ztrsen :: proc(matrix_layout: c.int, job: c.char, compq: c.char, select: ^i32, n: i32, t: ^complex128, ldt: i32, q: ^complex128, ldq: i32, w: ^complex128, m: ^i32, s: ^f64, sep: ^f64) -> i32 ---
	LAPACKE_strsna :: proc(matrix_layout: c.int, job: c.char, howmny: c.char, select: ^i32, n: i32, t: ^f32, ldt: i32, vl: ^f32, ldvl: i32, vr: ^f32, ldvr: i32, s: ^f32, sep: ^f32, mm: i32, m: ^i32) -> i32 ---
	LAPACKE_dtrsna :: proc(matrix_layout: c.int, job: c.char, howmny: c.char, select: ^i32, n: i32, t: ^f64, ldt: i32, vl: ^f64, ldvl: i32, vr: ^f64, ldvr: i32, s: ^f64, sep: ^f64, mm: i32, m: ^i32) -> i32 ---
	LAPACKE_ctrsna :: proc(matrix_layout: c.int, job: c.char, howmny: c.char, select: ^i32, n: i32, t: ^complex64, ldt: i32, vl: ^complex64, ldvl: i32, vr: ^complex64, ldvr: i32, s: ^f32, sep: ^f32, mm: i32, m: ^i32) -> i32 ---
	LAPACKE_ztrsna :: proc(matrix_layout: c.int, job: c.char, howmny: c.char, select: ^i32, n: i32, t: ^complex128, ldt: i32, vl: ^complex128, ldvl: i32, vr: ^complex128, ldvr: i32, s: ^f64, sep: ^f64, mm: i32, m: ^i32) -> i32 ---
	LAPACKE_strsyl :: proc(matrix_layout: c.int, trana: c.char, tranb: c.char, isgn: i32, m: i32, n: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32, _c: ^f32, ldc: i32, scale: ^f32) -> i32 ---
	LAPACKE_dtrsyl :: proc(matrix_layout: c.int, trana: c.char, tranb: c.char, isgn: i32, m: i32, n: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, _c: ^f64, ldc: i32, scale: ^f64) -> i32 ---
	LAPACKE_ctrsyl :: proc(matrix_layout: c.int, trana: c.char, tranb: c.char, isgn: i32, m: i32, n: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32, _c: ^complex64, ldc: i32, scale: ^f32) -> i32 ---
	LAPACKE_ztrsyl :: proc(matrix_layout: c.int, trana: c.char, tranb: c.char, isgn: i32, m: i32, n: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, _c: ^complex128, ldc: i32, scale: ^f64) -> i32 ---
	LAPACKE_strsyl3 :: proc(matrix_layout: c.int, trana: c.char, tranb: c.char, isgn: i32, m: i32, n: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32, _c: ^f32, ldc: i32, scale: ^f32) -> i32 ---
	LAPACKE_dtrsyl3 :: proc(matrix_layout: c.int, trana: c.char, tranb: c.char, isgn: i32, m: i32, n: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, _c: ^f64, ldc: i32, scale: ^f64) -> i32 ---
	LAPACKE_ztrsyl3 :: proc(matrix_layout: c.int, trana: c.char, tranb: c.char, isgn: i32, m: i32, n: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, _c: ^complex128, ldc: i32, scale: ^f64) -> i32 ---
	LAPACKE_strtri :: proc(matrix_layout: c.int, uplo: c.char, diag: c.char, n: i32, a: ^f32, lda: i32) -> i32 ---
	LAPACKE_dtrtri :: proc(matrix_layout: c.int, uplo: c.char, diag: c.char, n: i32, a: ^f64, lda: i32) -> i32 ---
	LAPACKE_ctrtri :: proc(matrix_layout: c.int, uplo: c.char, diag: c.char, n: i32, a: ^complex64, lda: i32) -> i32 ---
	LAPACKE_ztrtri :: proc(matrix_layout: c.int, uplo: c.char, diag: c.char, n: i32, a: ^complex128, lda: i32) -> i32 ---
	LAPACKE_strtrs :: proc(matrix_layout: c.int, uplo: c.char, trans: c.char, diag: c.char, n: i32, nrhs: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32) -> i32 ---
	LAPACKE_dtrtrs :: proc(matrix_layout: c.int, uplo: c.char, trans: c.char, diag: c.char, n: i32, nrhs: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32) -> i32 ---
	LAPACKE_ctrtrs :: proc(matrix_layout: c.int, uplo: c.char, trans: c.char, diag: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_ztrtrs :: proc(matrix_layout: c.int, uplo: c.char, trans: c.char, diag: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_strttf :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, n: i32, a: ^f32, lda: i32, arf: ^f32) -> i32 ---
	LAPACKE_dtrttf :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, n: i32, a: ^f64, lda: i32, arf: ^f64) -> i32 ---
	LAPACKE_ctrttf :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, n: i32, a: ^complex64, lda: i32, arf: ^complex64) -> i32 ---
	LAPACKE_ztrttf :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, n: i32, a: ^complex128, lda: i32, arf: ^complex128) -> i32 ---
	LAPACKE_strttp :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f32, lda: i32, ap: ^f32) -> i32 ---
	LAPACKE_dtrttp :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f64, lda: i32, ap: ^f64) -> i32 ---
	LAPACKE_ctrttp :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32, ap: ^complex64) -> i32 ---
	LAPACKE_ztrttp :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32, ap: ^complex128) -> i32 ---
	LAPACKE_stzrzf :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f32, lda: i32, tau: ^f32) -> i32 ---
	LAPACKE_dtzrzf :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f64, lda: i32, tau: ^f64) -> i32 ---
	LAPACKE_ctzrzf :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex64, lda: i32, tau: ^complex64) -> i32 ---
	LAPACKE_ztzrzf :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex128, lda: i32, tau: ^complex128) -> i32 ---
	LAPACKE_cungbr :: proc(matrix_layout: c.int, vect: c.char, m: i32, n: i32, k: i32, a: ^complex64, lda: i32, tau: ^complex64) -> i32 ---
	LAPACKE_zungbr :: proc(matrix_layout: c.int, vect: c.char, m: i32, n: i32, k: i32, a: ^complex128, lda: i32, tau: ^complex128) -> i32 ---
	LAPACKE_cunghr :: proc(matrix_layout: c.int, n: i32, ilo: i32, ihi: i32, a: ^complex64, lda: i32, tau: ^complex64) -> i32 ---
	LAPACKE_zunghr :: proc(matrix_layout: c.int, n: i32, ilo: i32, ihi: i32, a: ^complex128, lda: i32, tau: ^complex128) -> i32 ---
	LAPACKE_cunglq :: proc(matrix_layout: c.int, m: i32, n: i32, k: i32, a: ^complex64, lda: i32, tau: ^complex64) -> i32 ---
	LAPACKE_zunglq :: proc(matrix_layout: c.int, m: i32, n: i32, k: i32, a: ^complex128, lda: i32, tau: ^complex128) -> i32 ---
	LAPACKE_cungql :: proc(matrix_layout: c.int, m: i32, n: i32, k: i32, a: ^complex64, lda: i32, tau: ^complex64) -> i32 ---
	LAPACKE_zungql :: proc(matrix_layout: c.int, m: i32, n: i32, k: i32, a: ^complex128, lda: i32, tau: ^complex128) -> i32 ---
	LAPACKE_cungqr :: proc(matrix_layout: c.int, m: i32, n: i32, k: i32, a: ^complex64, lda: i32, tau: ^complex64) -> i32 ---
	LAPACKE_zungqr :: proc(matrix_layout: c.int, m: i32, n: i32, k: i32, a: ^complex128, lda: i32, tau: ^complex128) -> i32 ---
	LAPACKE_cungrq :: proc(matrix_layout: c.int, m: i32, n: i32, k: i32, a: ^complex64, lda: i32, tau: ^complex64) -> i32 ---
	LAPACKE_zungrq :: proc(matrix_layout: c.int, m: i32, n: i32, k: i32, a: ^complex128, lda: i32, tau: ^complex128) -> i32 ---
	LAPACKE_cungtr :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32, tau: ^complex64) -> i32 ---
	LAPACKE_zungtr :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32, tau: ^complex128) -> i32 ---
	LAPACKE_cungtsqr_row :: proc(matrix_layout: c.int, m: i32, n: i32, mb: i32, nb: i32, a: ^complex64, lda: i32, t: ^complex64, ldt: i32) -> i32 ---
	LAPACKE_zungtsqr_row :: proc(matrix_layout: c.int, m: i32, n: i32, mb: i32, nb: i32, a: ^complex128, lda: i32, t: ^complex128, ldt: i32) -> i32 ---
	LAPACKE_cunmbr :: proc(matrix_layout: c.int, vect: c.char, side: c.char, trans: c.char, m: i32, n: i32, k: i32, a: ^complex64, lda: i32, tau: ^complex64, _c: ^complex64, ldc: i32) -> i32 ---
	LAPACKE_zunmbr :: proc(matrix_layout: c.int, vect: c.char, side: c.char, trans: c.char, m: i32, n: i32, k: i32, a: ^complex128, lda: i32, tau: ^complex128, _c: ^complex128, ldc: i32) -> i32 ---
	LAPACKE_cunmhr :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, ilo: i32, ihi: i32, a: ^complex64, lda: i32, tau: ^complex64, _c: ^complex64, ldc: i32) -> i32 ---
	LAPACKE_zunmhr :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, ilo: i32, ihi: i32, a: ^complex128, lda: i32, tau: ^complex128, _c: ^complex128, ldc: i32) -> i32 ---
	LAPACKE_cunmlq :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, a: ^complex64, lda: i32, tau: ^complex64, _c: ^complex64, ldc: i32) -> i32 ---
	LAPACKE_zunmlq :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, a: ^complex128, lda: i32, tau: ^complex128, _c: ^complex128, ldc: i32) -> i32 ---
	LAPACKE_cunmql :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, a: ^complex64, lda: i32, tau: ^complex64, _c: ^complex64, ldc: i32) -> i32 ---
	LAPACKE_zunmql :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, a: ^complex128, lda: i32, tau: ^complex128, _c: ^complex128, ldc: i32) -> i32 ---
	LAPACKE_cunmqr :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, a: ^complex64, lda: i32, tau: ^complex64, _c: ^complex64, ldc: i32) -> i32 ---
	LAPACKE_zunmqr :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, a: ^complex128, lda: i32, tau: ^complex128, _c: ^complex128, ldc: i32) -> i32 ---
	LAPACKE_cunmrq :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, a: ^complex64, lda: i32, tau: ^complex64, _c: ^complex64, ldc: i32) -> i32 ---
	LAPACKE_zunmrq :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, a: ^complex128, lda: i32, tau: ^complex128, _c: ^complex128, ldc: i32) -> i32 ---
	LAPACKE_cunmrz :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, l: i32, a: ^complex64, lda: i32, tau: ^complex64, _c: ^complex64, ldc: i32) -> i32 ---
	LAPACKE_zunmrz :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, l: i32, a: ^complex128, lda: i32, tau: ^complex128, _c: ^complex128, ldc: i32) -> i32 ---
	LAPACKE_cunmtr :: proc(matrix_layout: c.int, side: c.char, uplo: c.char, trans: c.char, m: i32, n: i32, a: ^complex64, lda: i32, tau: ^complex64, _c: ^complex64, ldc: i32) -> i32 ---
	LAPACKE_zunmtr :: proc(matrix_layout: c.int, side: c.char, uplo: c.char, trans: c.char, m: i32, n: i32, a: ^complex128, lda: i32, tau: ^complex128, _c: ^complex128, ldc: i32) -> i32 ---
	LAPACKE_cupgtr :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^complex64, tau: ^complex64, q: ^complex64, ldq: i32) -> i32 ---
	LAPACKE_zupgtr :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^complex128, tau: ^complex128, q: ^complex128, ldq: i32) -> i32 ---
	LAPACKE_cupmtr :: proc(matrix_layout: c.int, side: c.char, uplo: c.char, trans: c.char, m: i32, n: i32, ap: ^complex64, tau: ^complex64, _c: ^complex64, ldc: i32) -> i32 ---
	LAPACKE_zupmtr :: proc(matrix_layout: c.int, side: c.char, uplo: c.char, trans: c.char, m: i32, n: i32, ap: ^complex128, tau: ^complex128, _c: ^complex128, ldc: i32) -> i32 ---
	LAPACKE_sbdsdc_work :: proc(matrix_layout: c.int, uplo: c.char, compq: c.char, n: i32, d: ^f32, e: ^f32, u: ^f32, ldu: i32, vt: ^f32, ldvt: i32, q: ^f32, iq: ^i32, work: ^f32, iwork: ^i32) -> i32 ---
	LAPACKE_dbdsdc_work :: proc(matrix_layout: c.int, uplo: c.char, compq: c.char, n: i32, d: ^f64, e: ^f64, u: ^f64, ldu: i32, vt: ^f64, ldvt: i32, q: ^f64, iq: ^i32, work: ^f64, iwork: ^i32) -> i32 ---
	LAPACKE_sbdsvdx_work :: proc(matrix_layout: c.int, uplo: c.char, jobz: c.char, range: c.char, n: i32, d: ^f32, e: ^f32, vl: f32, vu: f32, il: i32, iu: i32, ns: ^i32, s: ^f32, z: ^f32, ldz: i32, work: ^f32, iwork: ^i32) -> i32 ---
	LAPACKE_dbdsvdx_work :: proc(matrix_layout: c.int, uplo: c.char, jobz: c.char, range: c.char, n: i32, d: ^f64, e: ^f64, vl: f64, vu: f64, il: i32, iu: i32, ns: ^i32, s: ^f64, z: ^f64, ldz: i32, work: ^f64, iwork: ^i32) -> i32 ---
	LAPACKE_sbdsqr_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ncvt: i32, nru: i32, ncc: i32, d: ^f32, e: ^f32, vt: ^f32, ldvt: i32, u: ^f32, ldu: i32, _c: ^f32, ldc: i32, work: ^f32) -> i32 ---
	LAPACKE_dbdsqr_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ncvt: i32, nru: i32, ncc: i32, d: ^f64, e: ^f64, vt: ^f64, ldvt: i32, u: ^f64, ldu: i32, _c: ^f64, ldc: i32, work: ^f64) -> i32 ---
	LAPACKE_cbdsqr_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ncvt: i32, nru: i32, ncc: i32, d: ^f32, e: ^f32, vt: ^complex64, ldvt: i32, u: ^complex64, ldu: i32, _c: ^complex64, ldc: i32, work: ^f32) -> i32 ---
	LAPACKE_zbdsqr_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ncvt: i32, nru: i32, ncc: i32, d: ^f64, e: ^f64, vt: ^complex128, ldvt: i32, u: ^complex128, ldu: i32, _c: ^complex128, ldc: i32, work: ^f64) -> i32 ---
	LAPACKE_sdisna_work :: proc(job: c.char, m: i32, n: i32, d: ^f32, sep: ^f32) -> i32 ---
	LAPACKE_ddisna_work :: proc(job: c.char, m: i32, n: i32, d: ^f64, sep: ^f64) -> i32 ---
	LAPACKE_sgbbrd_work :: proc(matrix_layout: c.int, vect: c.char, m: i32, n: i32, ncc: i32, kl: i32, ku: i32, ab: ^f32, ldab: i32, d: ^f32, e: ^f32, q: ^f32, ldq: i32, pt: ^f32, ldpt: i32, _c: ^f32, ldc: i32, work: ^f32) -> i32 ---
	LAPACKE_dgbbrd_work :: proc(matrix_layout: c.int, vect: c.char, m: i32, n: i32, ncc: i32, kl: i32, ku: i32, ab: ^f64, ldab: i32, d: ^f64, e: ^f64, q: ^f64, ldq: i32, pt: ^f64, ldpt: i32, _c: ^f64, ldc: i32, work: ^f64) -> i32 ---
	LAPACKE_cgbbrd_work :: proc(matrix_layout: c.int, vect: c.char, m: i32, n: i32, ncc: i32, kl: i32, ku: i32, ab: ^complex64, ldab: i32, d: ^f32, e: ^f32, q: ^complex64, ldq: i32, pt: ^complex64, ldpt: i32, _c: ^complex64, ldc: i32, work: ^complex64, rwork: ^f32) -> i32 ---
	LAPACKE_zgbbrd_work :: proc(matrix_layout: c.int, vect: c.char, m: i32, n: i32, ncc: i32, kl: i32, ku: i32, ab: ^complex128, ldab: i32, d: ^f64, e: ^f64, q: ^complex128, ldq: i32, pt: ^complex128, ldpt: i32, _c: ^complex128, ldc: i32, work: ^complex128, rwork: ^f64) -> i32 ---
	LAPACKE_sgbcon_work :: proc(matrix_layout: c.int, norm: c.char, n: i32, kl: i32, ku: i32, ab: ^f32, ldab: i32, ipiv: ^i32, anorm: f32, rcond: ^f32, work: ^f32, iwork: ^i32) -> i32 ---
	LAPACKE_dgbcon_work :: proc(matrix_layout: c.int, norm: c.char, n: i32, kl: i32, ku: i32, ab: ^f64, ldab: i32, ipiv: ^i32, anorm: f64, rcond: ^f64, work: ^f64, iwork: ^i32) -> i32 ---
	LAPACKE_cgbcon_work :: proc(matrix_layout: c.int, norm: c.char, n: i32, kl: i32, ku: i32, ab: ^complex64, ldab: i32, ipiv: ^i32, anorm: f32, rcond: ^f32, work: ^complex64, rwork: ^f32) -> i32 ---
	LAPACKE_zgbcon_work :: proc(matrix_layout: c.int, norm: c.char, n: i32, kl: i32, ku: i32, ab: ^complex128, ldab: i32, ipiv: ^i32, anorm: f64, rcond: ^f64, work: ^complex128, rwork: ^f64) -> i32 ---
	LAPACKE_sgbequ_work :: proc(matrix_layout: c.int, m: i32, n: i32, kl: i32, ku: i32, ab: ^f32, ldab: i32, r: ^f32, _c: ^f32, rowcnd: ^f32, colcnd: ^f32, amax: ^f32) -> i32 ---
	LAPACKE_dgbequ_work :: proc(matrix_layout: c.int, m: i32, n: i32, kl: i32, ku: i32, ab: ^f64, ldab: i32, r: ^f64, _c: ^f64, rowcnd: ^f64, colcnd: ^f64, amax: ^f64) -> i32 ---
	LAPACKE_cgbequ_work :: proc(matrix_layout: c.int, m: i32, n: i32, kl: i32, ku: i32, ab: ^complex64, ldab: i32, r: ^f32, _c: ^f32, rowcnd: ^f32, colcnd: ^f32, amax: ^f32) -> i32 ---
	LAPACKE_zgbequ_work :: proc(matrix_layout: c.int, m: i32, n: i32, kl: i32, ku: i32, ab: ^complex128, ldab: i32, r: ^f64, _c: ^f64, rowcnd: ^f64, colcnd: ^f64, amax: ^f64) -> i32 ---
	LAPACKE_sgbequb_work :: proc(matrix_layout: c.int, m: i32, n: i32, kl: i32, ku: i32, ab: ^f32, ldab: i32, r: ^f32, _c: ^f32, rowcnd: ^f32, colcnd: ^f32, amax: ^f32) -> i32 ---
	LAPACKE_dgbequb_work :: proc(matrix_layout: c.int, m: i32, n: i32, kl: i32, ku: i32, ab: ^f64, ldab: i32, r: ^f64, _c: ^f64, rowcnd: ^f64, colcnd: ^f64, amax: ^f64) -> i32 ---
	LAPACKE_cgbequb_work :: proc(matrix_layout: c.int, m: i32, n: i32, kl: i32, ku: i32, ab: ^complex64, ldab: i32, r: ^f32, _c: ^f32, rowcnd: ^f32, colcnd: ^f32, amax: ^f32) -> i32 ---
	LAPACKE_zgbequb_work :: proc(matrix_layout: c.int, m: i32, n: i32, kl: i32, ku: i32, ab: ^complex128, ldab: i32, r: ^f64, _c: ^f64, rowcnd: ^f64, colcnd: ^f64, amax: ^f64) -> i32 ---
	LAPACKE_sgbrfs_work :: proc(matrix_layout: c.int, trans: c.char, n: i32, kl: i32, ku: i32, nrhs: i32, ab: ^f32, ldab: i32, afb: ^f32, ldafb: i32, ipiv: ^i32, b: ^f32, ldb: i32, x: ^f32, ldx: i32, ferr: ^f32, berr: ^f32, work: ^f32, iwork: ^i32) -> i32 ---
	LAPACKE_dgbrfs_work :: proc(matrix_layout: c.int, trans: c.char, n: i32, kl: i32, ku: i32, nrhs: i32, ab: ^f64, ldab: i32, afb: ^f64, ldafb: i32, ipiv: ^i32, b: ^f64, ldb: i32, x: ^f64, ldx: i32, ferr: ^f64, berr: ^f64, work: ^f64, iwork: ^i32) -> i32 ---
	LAPACKE_cgbrfs_work :: proc(matrix_layout: c.int, trans: c.char, n: i32, kl: i32, ku: i32, nrhs: i32, ab: ^complex64, ldab: i32, afb: ^complex64, ldafb: i32, ipiv: ^i32, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, ferr: ^f32, berr: ^f32, work: ^complex64, rwork: ^f32) -> i32 ---
	LAPACKE_zgbrfs_work :: proc(matrix_layout: c.int, trans: c.char, n: i32, kl: i32, ku: i32, nrhs: i32, ab: ^complex128, ldab: i32, afb: ^complex128, ldafb: i32, ipiv: ^i32, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, ferr: ^f64, berr: ^f64, work: ^complex128, rwork: ^f64) -> i32 ---
	LAPACKE_sgbrfsx_work :: proc(matrix_layout: c.int, trans: c.char, equed: c.char, n: i32, kl: i32, ku: i32, nrhs: i32, ab: ^f32, ldab: i32, afb: ^f32, ldafb: i32, ipiv: ^i32, r: ^f32, _c: ^f32, b: ^f32, ldb: i32, x: ^f32, ldx: i32, rcond: ^f32, berr: ^f32, n_err_bnds: i32, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: i32, params: ^f32, work: ^f32, iwork: ^i32) -> i32 ---
	LAPACKE_dgbrfsx_work :: proc(matrix_layout: c.int, trans: c.char, equed: c.char, n: i32, kl: i32, ku: i32, nrhs: i32, ab: ^f64, ldab: i32, afb: ^f64, ldafb: i32, ipiv: ^i32, r: ^f64, _c: ^f64, b: ^f64, ldb: i32, x: ^f64, ldx: i32, rcond: ^f64, berr: ^f64, n_err_bnds: i32, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: i32, params: ^f64, work: ^f64, iwork: ^i32) -> i32 ---
	LAPACKE_cgbrfsx_work :: proc(matrix_layout: c.int, trans: c.char, equed: c.char, n: i32, kl: i32, ku: i32, nrhs: i32, ab: ^complex64, ldab: i32, afb: ^complex64, ldafb: i32, ipiv: ^i32, r: ^f32, _c: ^f32, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, rcond: ^f32, berr: ^f32, n_err_bnds: i32, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: i32, params: ^f32, work: ^complex64, rwork: ^f32) -> i32 ---
	LAPACKE_zgbrfsx_work :: proc(matrix_layout: c.int, trans: c.char, equed: c.char, n: i32, kl: i32, ku: i32, nrhs: i32, ab: ^complex128, ldab: i32, afb: ^complex128, ldafb: i32, ipiv: ^i32, r: ^f64, _c: ^f64, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, rcond: ^f64, berr: ^f64, n_err_bnds: i32, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: i32, params: ^f64, work: ^complex128, rwork: ^f64) -> i32 ---
	LAPACKE_sgbsv_work :: proc(matrix_layout: c.int, n: i32, kl: i32, ku: i32, nrhs: i32, ab: ^f32, ldab: i32, ipiv: ^i32, b: ^f32, ldb: i32) -> i32 ---
	LAPACKE_dgbsv_work :: proc(matrix_layout: c.int, n: i32, kl: i32, ku: i32, nrhs: i32, ab: ^f64, ldab: i32, ipiv: ^i32, b: ^f64, ldb: i32) -> i32 ---
	LAPACKE_cgbsv_work :: proc(matrix_layout: c.int, n: i32, kl: i32, ku: i32, nrhs: i32, ab: ^complex64, ldab: i32, ipiv: ^i32, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_zgbsv_work :: proc(matrix_layout: c.int, n: i32, kl: i32, ku: i32, nrhs: i32, ab: ^complex128, ldab: i32, ipiv: ^i32, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_sgbsvx_work :: proc(matrix_layout: c.int, fact: c.char, trans: c.char, n: i32, kl: i32, ku: i32, nrhs: i32, ab: ^f32, ldab: i32, afb: ^f32, ldafb: i32, ipiv: ^i32, equed: cstring, r: ^f32, _c: ^f32, b: ^f32, ldb: i32, x: ^f32, ldx: i32, rcond: ^f32, ferr: ^f32, berr: ^f32, work: ^f32, iwork: ^i32) -> i32 ---
	LAPACKE_dgbsvx_work :: proc(matrix_layout: c.int, fact: c.char, trans: c.char, n: i32, kl: i32, ku: i32, nrhs: i32, ab: ^f64, ldab: i32, afb: ^f64, ldafb: i32, ipiv: ^i32, equed: cstring, r: ^f64, _c: ^f64, b: ^f64, ldb: i32, x: ^f64, ldx: i32, rcond: ^f64, ferr: ^f64, berr: ^f64, work: ^f64, iwork: ^i32) -> i32 ---
	LAPACKE_cgbsvx_work :: proc(matrix_layout: c.int, fact: c.char, trans: c.char, n: i32, kl: i32, ku: i32, nrhs: i32, ab: ^complex64, ldab: i32, afb: ^complex64, ldafb: i32, ipiv: ^i32, equed: cstring, r: ^f32, _c: ^f32, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, rcond: ^f32, ferr: ^f32, berr: ^f32, work: ^complex64, rwork: ^f32) -> i32 ---
	LAPACKE_zgbsvx_work :: proc(matrix_layout: c.int, fact: c.char, trans: c.char, n: i32, kl: i32, ku: i32, nrhs: i32, ab: ^complex128, ldab: i32, afb: ^complex128, ldafb: i32, ipiv: ^i32, equed: cstring, r: ^f64, _c: ^f64, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, rcond: ^f64, ferr: ^f64, berr: ^f64, work: ^complex128, rwork: ^f64) -> i32 ---
	LAPACKE_sgbsvxx_work :: proc(matrix_layout: c.int, fact: c.char, trans: c.char, n: i32, kl: i32, ku: i32, nrhs: i32, ab: ^f32, ldab: i32, afb: ^f32, ldafb: i32, ipiv: ^i32, equed: cstring, r: ^f32, _c: ^f32, b: ^f32, ldb: i32, x: ^f32, ldx: i32, rcond: ^f32, rpvgrw: ^f32, berr: ^f32, n_err_bnds: i32, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: i32, params: ^f32, work: ^f32, iwork: ^i32) -> i32 ---
	LAPACKE_dgbsvxx_work :: proc(matrix_layout: c.int, fact: c.char, trans: c.char, n: i32, kl: i32, ku: i32, nrhs: i32, ab: ^f64, ldab: i32, afb: ^f64, ldafb: i32, ipiv: ^i32, equed: cstring, r: ^f64, _c: ^f64, b: ^f64, ldb: i32, x: ^f64, ldx: i32, rcond: ^f64, rpvgrw: ^f64, berr: ^f64, n_err_bnds: i32, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: i32, params: ^f64, work: ^f64, iwork: ^i32) -> i32 ---
	LAPACKE_cgbsvxx_work :: proc(matrix_layout: c.int, fact: c.char, trans: c.char, n: i32, kl: i32, ku: i32, nrhs: i32, ab: ^complex64, ldab: i32, afb: ^complex64, ldafb: i32, ipiv: ^i32, equed: cstring, r: ^f32, _c: ^f32, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, rcond: ^f32, rpvgrw: ^f32, berr: ^f32, n_err_bnds: i32, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: i32, params: ^f32, work: ^complex64, rwork: ^f32) -> i32 ---
	LAPACKE_zgbsvxx_work :: proc(matrix_layout: c.int, fact: c.char, trans: c.char, n: i32, kl: i32, ku: i32, nrhs: i32, ab: ^complex128, ldab: i32, afb: ^complex128, ldafb: i32, ipiv: ^i32, equed: cstring, r: ^f64, _c: ^f64, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, rcond: ^f64, rpvgrw: ^f64, berr: ^f64, n_err_bnds: i32, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: i32, params: ^f64, work: ^complex128, rwork: ^f64) -> i32 ---
	LAPACKE_sgbtrf_work :: proc(matrix_layout: c.int, m: i32, n: i32, kl: i32, ku: i32, ab: ^f32, ldab: i32, ipiv: ^i32) -> i32 ---
	LAPACKE_dgbtrf_work :: proc(matrix_layout: c.int, m: i32, n: i32, kl: i32, ku: i32, ab: ^f64, ldab: i32, ipiv: ^i32) -> i32 ---
	LAPACKE_cgbtrf_work :: proc(matrix_layout: c.int, m: i32, n: i32, kl: i32, ku: i32, ab: ^complex64, ldab: i32, ipiv: ^i32) -> i32 ---
	LAPACKE_zgbtrf_work :: proc(matrix_layout: c.int, m: i32, n: i32, kl: i32, ku: i32, ab: ^complex128, ldab: i32, ipiv: ^i32) -> i32 ---
	LAPACKE_sgbtrs_work :: proc(matrix_layout: c.int, trans: c.char, n: i32, kl: i32, ku: i32, nrhs: i32, ab: ^f32, ldab: i32, ipiv: ^i32, b: ^f32, ldb: i32) -> i32 ---
	LAPACKE_dgbtrs_work :: proc(matrix_layout: c.int, trans: c.char, n: i32, kl: i32, ku: i32, nrhs: i32, ab: ^f64, ldab: i32, ipiv: ^i32, b: ^f64, ldb: i32) -> i32 ---
	LAPACKE_cgbtrs_work :: proc(matrix_layout: c.int, trans: c.char, n: i32, kl: i32, ku: i32, nrhs: i32, ab: ^complex64, ldab: i32, ipiv: ^i32, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_zgbtrs_work :: proc(matrix_layout: c.int, trans: c.char, n: i32, kl: i32, ku: i32, nrhs: i32, ab: ^complex128, ldab: i32, ipiv: ^i32, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_sgebak_work :: proc(matrix_layout: c.int, job: c.char, side: c.char, n: i32, ilo: i32, ihi: i32, scale: ^f32, m: i32, v: ^f32, ldv: i32) -> i32 ---
	LAPACKE_dgebak_work :: proc(matrix_layout: c.int, job: c.char, side: c.char, n: i32, ilo: i32, ihi: i32, scale: ^f64, m: i32, v: ^f64, ldv: i32) -> i32 ---
	LAPACKE_cgebak_work :: proc(matrix_layout: c.int, job: c.char, side: c.char, n: i32, ilo: i32, ihi: i32, scale: ^f32, m: i32, v: ^complex64, ldv: i32) -> i32 ---
	LAPACKE_zgebak_work :: proc(matrix_layout: c.int, job: c.char, side: c.char, n: i32, ilo: i32, ihi: i32, scale: ^f64, m: i32, v: ^complex128, ldv: i32) -> i32 ---
	LAPACKE_sgebal_work :: proc(matrix_layout: c.int, job: c.char, n: i32, a: ^f32, lda: i32, ilo: ^i32, ihi: ^i32, scale: ^f32) -> i32 ---
	LAPACKE_dgebal_work :: proc(matrix_layout: c.int, job: c.char, n: i32, a: ^f64, lda: i32, ilo: ^i32, ihi: ^i32, scale: ^f64) -> i32 ---
	LAPACKE_cgebal_work :: proc(matrix_layout: c.int, job: c.char, n: i32, a: ^complex64, lda: i32, ilo: ^i32, ihi: ^i32, scale: ^f32) -> i32 ---
	LAPACKE_zgebal_work :: proc(matrix_layout: c.int, job: c.char, n: i32, a: ^complex128, lda: i32, ilo: ^i32, ihi: ^i32, scale: ^f64) -> i32 ---
	LAPACKE_sgebrd_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f32, lda: i32, d: ^f32, e: ^f32, tauq: ^f32, taup: ^f32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_dgebrd_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f64, lda: i32, d: ^f64, e: ^f64, tauq: ^f64, taup: ^f64, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_cgebrd_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex64, lda: i32, d: ^f32, e: ^f32, tauq: ^complex64, taup: ^complex64, work: ^complex64, lwork: i32) -> i32 ---
	LAPACKE_zgebrd_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex128, lda: i32, d: ^f64, e: ^f64, tauq: ^complex128, taup: ^complex128, work: ^complex128, lwork: i32) -> i32 ---
	LAPACKE_sgecon_work :: proc(matrix_layout: c.int, norm: c.char, n: i32, a: ^f32, lda: i32, anorm: f32, rcond: ^f32, work: ^f32, iwork: ^i32) -> i32 ---
	LAPACKE_dgecon_work :: proc(matrix_layout: c.int, norm: c.char, n: i32, a: ^f64, lda: i32, anorm: f64, rcond: ^f64, work: ^f64, iwork: ^i32) -> i32 ---
	LAPACKE_cgecon_work :: proc(matrix_layout: c.int, norm: c.char, n: i32, a: ^complex64, lda: i32, anorm: f32, rcond: ^f32, work: ^complex64, rwork: ^f32) -> i32 ---
	LAPACKE_zgecon_work :: proc(matrix_layout: c.int, norm: c.char, n: i32, a: ^complex128, lda: i32, anorm: f64, rcond: ^f64, work: ^complex128, rwork: ^f64) -> i32 ---
	LAPACKE_sgeequ_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f32, lda: i32, r: ^f32, _c: ^f32, rowcnd: ^f32, colcnd: ^f32, amax: ^f32) -> i32 ---
	LAPACKE_dgeequ_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f64, lda: i32, r: ^f64, _c: ^f64, rowcnd: ^f64, colcnd: ^f64, amax: ^f64) -> i32 ---
	LAPACKE_cgeequ_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex64, lda: i32, r: ^f32, _c: ^f32, rowcnd: ^f32, colcnd: ^f32, amax: ^f32) -> i32 ---
	LAPACKE_zgeequ_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex128, lda: i32, r: ^f64, _c: ^f64, rowcnd: ^f64, colcnd: ^f64, amax: ^f64) -> i32 ---
	LAPACKE_sgeequb_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f32, lda: i32, r: ^f32, _c: ^f32, rowcnd: ^f32, colcnd: ^f32, amax: ^f32) -> i32 ---
	LAPACKE_dgeequb_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f64, lda: i32, r: ^f64, _c: ^f64, rowcnd: ^f64, colcnd: ^f64, amax: ^f64) -> i32 ---
	LAPACKE_cgeequb_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex64, lda: i32, r: ^f32, _c: ^f32, rowcnd: ^f32, colcnd: ^f32, amax: ^f32) -> i32 ---
	LAPACKE_zgeequb_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex128, lda: i32, r: ^f64, _c: ^f64, rowcnd: ^f64, colcnd: ^f64, amax: ^f64) -> i32 ---
	LAPACKE_sgees_work :: proc(matrix_layout: c.int, jobvs: c.char, sort: c.char, select: LAPACK_S_SELECT2, n: i32, a: ^f32, lda: i32, sdim: ^i32, wr: ^f32, wi: ^f32, vs: ^f32, ldvs: i32, work: ^f32, lwork: i32, bwork: ^i32) -> i32 ---
	LAPACKE_dgees_work :: proc(matrix_layout: c.int, jobvs: c.char, sort: c.char, select: LAPACK_D_SELECT2, n: i32, a: ^f64, lda: i32, sdim: ^i32, wr: ^f64, wi: ^f64, vs: ^f64, ldvs: i32, work: ^f64, lwork: i32, bwork: ^i32) -> i32 ---
	LAPACKE_cgees_work :: proc(matrix_layout: c.int, jobvs: c.char, sort: c.char, select: LAPACK_C_SELECT1, n: i32, a: ^complex64, lda: i32, sdim: ^i32, w: ^complex64, vs: ^complex64, ldvs: i32, work: ^complex64, lwork: i32, rwork: ^f32, bwork: ^i32) -> i32 ---
	LAPACKE_zgees_work :: proc(matrix_layout: c.int, jobvs: c.char, sort: c.char, select: LAPACK_Z_SELECT1, n: i32, a: ^complex128, lda: i32, sdim: ^i32, w: ^complex128, vs: ^complex128, ldvs: i32, work: ^complex128, lwork: i32, rwork: ^f64, bwork: ^i32) -> i32 ---
	LAPACKE_sgeesx_work :: proc(matrix_layout: c.int, jobvs: c.char, sort: c.char, select: LAPACK_S_SELECT2, sense: c.char, n: i32, a: ^f32, lda: i32, sdim: ^i32, wr: ^f32, wi: ^f32, vs: ^f32, ldvs: i32, rconde: ^f32, rcondv: ^f32, work: ^f32, lwork: i32, iwork: ^i32, liwork: i32, bwork: ^i32) -> i32 ---
	LAPACKE_dgeesx_work :: proc(matrix_layout: c.int, jobvs: c.char, sort: c.char, select: LAPACK_D_SELECT2, sense: c.char, n: i32, a: ^f64, lda: i32, sdim: ^i32, wr: ^f64, wi: ^f64, vs: ^f64, ldvs: i32, rconde: ^f64, rcondv: ^f64, work: ^f64, lwork: i32, iwork: ^i32, liwork: i32, bwork: ^i32) -> i32 ---
	LAPACKE_cgeesx_work :: proc(matrix_layout: c.int, jobvs: c.char, sort: c.char, select: LAPACK_C_SELECT1, sense: c.char, n: i32, a: ^complex64, lda: i32, sdim: ^i32, w: ^complex64, vs: ^complex64, ldvs: i32, rconde: ^f32, rcondv: ^f32, work: ^complex64, lwork: i32, rwork: ^f32, bwork: ^i32) -> i32 ---
	LAPACKE_zgeesx_work :: proc(matrix_layout: c.int, jobvs: c.char, sort: c.char, select: LAPACK_Z_SELECT1, sense: c.char, n: i32, a: ^complex128, lda: i32, sdim: ^i32, w: ^complex128, vs: ^complex128, ldvs: i32, rconde: ^f64, rcondv: ^f64, work: ^complex128, lwork: i32, rwork: ^f64, bwork: ^i32) -> i32 ---
	LAPACKE_sgeev_work :: proc(matrix_layout: c.int, jobvl: c.char, jobvr: c.char, n: i32, a: ^f32, lda: i32, wr: ^f32, wi: ^f32, vl: ^f32, ldvl: i32, vr: ^f32, ldvr: i32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_dgeev_work :: proc(matrix_layout: c.int, jobvl: c.char, jobvr: c.char, n: i32, a: ^f64, lda: i32, wr: ^f64, wi: ^f64, vl: ^f64, ldvl: i32, vr: ^f64, ldvr: i32, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_cgeev_work :: proc(matrix_layout: c.int, jobvl: c.char, jobvr: c.char, n: i32, a: ^complex64, lda: i32, w: ^complex64, vl: ^complex64, ldvl: i32, vr: ^complex64, ldvr: i32, work: ^complex64, lwork: i32, rwork: ^f32) -> i32 ---
	LAPACKE_zgeev_work :: proc(matrix_layout: c.int, jobvl: c.char, jobvr: c.char, n: i32, a: ^complex128, lda: i32, w: ^complex128, vl: ^complex128, ldvl: i32, vr: ^complex128, ldvr: i32, work: ^complex128, lwork: i32, rwork: ^f64) -> i32 ---
	LAPACKE_sgeevx_work :: proc(matrix_layout: c.int, balanc: c.char, jobvl: c.char, jobvr: c.char, sense: c.char, n: i32, a: ^f32, lda: i32, wr: ^f32, wi: ^f32, vl: ^f32, ldvl: i32, vr: ^f32, ldvr: i32, ilo: ^i32, ihi: ^i32, scale: ^f32, abnrm: ^f32, rconde: ^f32, rcondv: ^f32, work: ^f32, lwork: i32, iwork: ^i32) -> i32 ---
	LAPACKE_dgeevx_work :: proc(matrix_layout: c.int, balanc: c.char, jobvl: c.char, jobvr: c.char, sense: c.char, n: i32, a: ^f64, lda: i32, wr: ^f64, wi: ^f64, vl: ^f64, ldvl: i32, vr: ^f64, ldvr: i32, ilo: ^i32, ihi: ^i32, scale: ^f64, abnrm: ^f64, rconde: ^f64, rcondv: ^f64, work: ^f64, lwork: i32, iwork: ^i32) -> i32 ---
	LAPACKE_cgeevx_work :: proc(matrix_layout: c.int, balanc: c.char, jobvl: c.char, jobvr: c.char, sense: c.char, n: i32, a: ^complex64, lda: i32, w: ^complex64, vl: ^complex64, ldvl: i32, vr: ^complex64, ldvr: i32, ilo: ^i32, ihi: ^i32, scale: ^f32, abnrm: ^f32, rconde: ^f32, rcondv: ^f32, work: ^complex64, lwork: i32, rwork: ^f32) -> i32 ---
	LAPACKE_zgeevx_work :: proc(matrix_layout: c.int, balanc: c.char, jobvl: c.char, jobvr: c.char, sense: c.char, n: i32, a: ^complex128, lda: i32, w: ^complex128, vl: ^complex128, ldvl: i32, vr: ^complex128, ldvr: i32, ilo: ^i32, ihi: ^i32, scale: ^f64, abnrm: ^f64, rconde: ^f64, rcondv: ^f64, work: ^complex128, lwork: i32, rwork: ^f64) -> i32 ---
	LAPACKE_sgehrd_work :: proc(matrix_layout: c.int, n: i32, ilo: i32, ihi: i32, a: ^f32, lda: i32, tau: ^f32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_dgehrd_work :: proc(matrix_layout: c.int, n: i32, ilo: i32, ihi: i32, a: ^f64, lda: i32, tau: ^f64, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_cgehrd_work :: proc(matrix_layout: c.int, n: i32, ilo: i32, ihi: i32, a: ^complex64, lda: i32, tau: ^complex64, work: ^complex64, lwork: i32) -> i32 ---
	LAPACKE_zgehrd_work :: proc(matrix_layout: c.int, n: i32, ilo: i32, ihi: i32, a: ^complex128, lda: i32, tau: ^complex128, work: ^complex128, lwork: i32) -> i32 ---
	LAPACKE_sgejsv_work :: proc(matrix_layout: c.int, joba: c.char, jobu: c.char, jobv: c.char, jobr: c.char, jobt: c.char, jobp: c.char, m: i32, n: i32, a: ^f32, lda: i32, sva: ^f32, u: ^f32, ldu: i32, v: ^f32, ldv: i32, work: ^f32, lwork: i32, iwork: ^i32) -> i32 ---
	LAPACKE_dgejsv_work :: proc(matrix_layout: c.int, joba: c.char, jobu: c.char, jobv: c.char, jobr: c.char, jobt: c.char, jobp: c.char, m: i32, n: i32, a: ^f64, lda: i32, sva: ^f64, u: ^f64, ldu: i32, v: ^f64, ldv: i32, work: ^f64, lwork: i32, iwork: ^i32) -> i32 ---
	LAPACKE_cgejsv_work :: proc(matrix_layout: c.int, joba: c.char, jobu: c.char, jobv: c.char, jobr: c.char, jobt: c.char, jobp: c.char, m: i32, n: i32, a: ^complex64, lda: i32, sva: ^f32, u: ^complex64, ldu: i32, v: ^complex64, ldv: i32, cwork: ^complex64, lwork: i32, work: ^f32, lrwork: i32, iwork: ^i32) -> i32 ---
	LAPACKE_zgejsv_work :: proc(matrix_layout: c.int, joba: c.char, jobu: c.char, jobv: c.char, jobr: c.char, jobt: c.char, jobp: c.char, m: i32, n: i32, a: ^complex128, lda: i32, sva: ^f64, u: ^complex128, ldu: i32, v: ^complex128, ldv: i32, cwork: ^complex128, lwork: i32, work: ^f64, lrwork: i32, iwork: ^i32) -> i32 ---
	LAPACKE_sgelq2_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f32, lda: i32, tau: ^f32, work: ^f32) -> i32 ---
	LAPACKE_dgelq2_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f64, lda: i32, tau: ^f64, work: ^f64) -> i32 ---
	LAPACKE_cgelq2_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex64, lda: i32, tau: ^complex64, work: ^complex64) -> i32 ---
	LAPACKE_zgelq2_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex128, lda: i32, tau: ^complex128, work: ^complex128) -> i32 ---
	LAPACKE_sgelqf_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f32, lda: i32, tau: ^f32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_dgelqf_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f64, lda: i32, tau: ^f64, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_cgelqf_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex64, lda: i32, tau: ^complex64, work: ^complex64, lwork: i32) -> i32 ---
	LAPACKE_zgelqf_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex128, lda: i32, tau: ^complex128, work: ^complex128, lwork: i32) -> i32 ---
	LAPACKE_sgels_work :: proc(matrix_layout: c.int, trans: c.char, m: i32, n: i32, nrhs: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_dgels_work :: proc(matrix_layout: c.int, trans: c.char, m: i32, n: i32, nrhs: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_cgels_work :: proc(matrix_layout: c.int, trans: c.char, m: i32, n: i32, nrhs: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32, work: ^complex64, lwork: i32) -> i32 ---
	LAPACKE_zgels_work :: proc(matrix_layout: c.int, trans: c.char, m: i32, n: i32, nrhs: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, work: ^complex128, lwork: i32) -> i32 ---
	LAPACKE_sgelsd_work :: proc(matrix_layout: c.int, m: i32, n: i32, nrhs: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32, s: ^f32, rcond: f32, rank: ^i32, work: ^f32, lwork: i32, iwork: ^i32) -> i32 ---
	LAPACKE_dgelsd_work :: proc(matrix_layout: c.int, m: i32, n: i32, nrhs: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, s: ^f64, rcond: f64, rank: ^i32, work: ^f64, lwork: i32, iwork: ^i32) -> i32 ---
	LAPACKE_cgelsd_work :: proc(matrix_layout: c.int, m: i32, n: i32, nrhs: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32, s: ^f32, rcond: f32, rank: ^i32, work: ^complex64, lwork: i32, rwork: ^f32, iwork: ^i32) -> i32 ---
	LAPACKE_zgelsd_work :: proc(matrix_layout: c.int, m: i32, n: i32, nrhs: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, s: ^f64, rcond: f64, rank: ^i32, work: ^complex128, lwork: i32, rwork: ^f64, iwork: ^i32) -> i32 ---
	LAPACKE_sgelss_work :: proc(matrix_layout: c.int, m: i32, n: i32, nrhs: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32, s: ^f32, rcond: f32, rank: ^i32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_dgelss_work :: proc(matrix_layout: c.int, m: i32, n: i32, nrhs: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, s: ^f64, rcond: f64, rank: ^i32, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_cgelss_work :: proc(matrix_layout: c.int, m: i32, n: i32, nrhs: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32, s: ^f32, rcond: f32, rank: ^i32, work: ^complex64, lwork: i32, rwork: ^f32) -> i32 ---
	LAPACKE_zgelss_work :: proc(matrix_layout: c.int, m: i32, n: i32, nrhs: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, s: ^f64, rcond: f64, rank: ^i32, work: ^complex128, lwork: i32, rwork: ^f64) -> i32 ---
	LAPACKE_sgelsy_work :: proc(matrix_layout: c.int, m: i32, n: i32, nrhs: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32, jpvt: ^i32, rcond: f32, rank: ^i32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_dgelsy_work :: proc(matrix_layout: c.int, m: i32, n: i32, nrhs: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, jpvt: ^i32, rcond: f64, rank: ^i32, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_cgelsy_work :: proc(matrix_layout: c.int, m: i32, n: i32, nrhs: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32, jpvt: ^i32, rcond: f32, rank: ^i32, work: ^complex64, lwork: i32, rwork: ^f32) -> i32 ---
	LAPACKE_zgelsy_work :: proc(matrix_layout: c.int, m: i32, n: i32, nrhs: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, jpvt: ^i32, rcond: f64, rank: ^i32, work: ^complex128, lwork: i32, rwork: ^f64) -> i32 ---
	LAPACKE_sgeqlf_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f32, lda: i32, tau: ^f32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_dgeqlf_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f64, lda: i32, tau: ^f64, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_cgeqlf_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex64, lda: i32, tau: ^complex64, work: ^complex64, lwork: i32) -> i32 ---
	LAPACKE_zgeqlf_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex128, lda: i32, tau: ^complex128, work: ^complex128, lwork: i32) -> i32 ---
	LAPACKE_sgeqp3_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f32, lda: i32, jpvt: ^i32, tau: ^f32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_dgeqp3_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f64, lda: i32, jpvt: ^i32, tau: ^f64, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_cgeqp3_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex64, lda: i32, jpvt: ^i32, tau: ^complex64, work: ^complex64, lwork: i32, rwork: ^f32) -> i32 ---
	LAPACKE_zgeqp3_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex128, lda: i32, jpvt: ^i32, tau: ^complex128, work: ^complex128, lwork: i32, rwork: ^f64) -> i32 ---
	LAPACKE_sgeqpf_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f32, lda: i32, jpvt: ^i32, tau: ^f32, work: ^f32) -> i32 ---
	LAPACKE_dgeqpf_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f64, lda: i32, jpvt: ^i32, tau: ^f64, work: ^f64) -> i32 ---
	LAPACKE_cgeqpf_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex64, lda: i32, jpvt: ^i32, tau: ^complex64, work: ^complex64, rwork: ^f32) -> i32 ---
	LAPACKE_zgeqpf_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex128, lda: i32, jpvt: ^i32, tau: ^complex128, work: ^complex128, rwork: ^f64) -> i32 ---
	LAPACKE_sgeqr2_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f32, lda: i32, tau: ^f32, work: ^f32) -> i32 ---
	LAPACKE_dgeqr2_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f64, lda: i32, tau: ^f64, work: ^f64) -> i32 ---
	LAPACKE_cgeqr2_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex64, lda: i32, tau: ^complex64, work: ^complex64) -> i32 ---
	LAPACKE_zgeqr2_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex128, lda: i32, tau: ^complex128, work: ^complex128) -> i32 ---
	LAPACKE_sgeqrf_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f32, lda: i32, tau: ^f32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_dgeqrf_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f64, lda: i32, tau: ^f64, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_cgeqrf_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex64, lda: i32, tau: ^complex64, work: ^complex64, lwork: i32) -> i32 ---
	LAPACKE_zgeqrf_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex128, lda: i32, tau: ^complex128, work: ^complex128, lwork: i32) -> i32 ---
	LAPACKE_sgeqrfp_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f32, lda: i32, tau: ^f32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_dgeqrfp_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f64, lda: i32, tau: ^f64, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_cgeqrfp_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex64, lda: i32, tau: ^complex64, work: ^complex64, lwork: i32) -> i32 ---
	LAPACKE_zgeqrfp_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex128, lda: i32, tau: ^complex128, work: ^complex128, lwork: i32) -> i32 ---
	LAPACKE_sgerfs_work :: proc(matrix_layout: c.int, trans: c.char, n: i32, nrhs: i32, a: ^f32, lda: i32, af: ^f32, ldaf: i32, ipiv: ^i32, b: ^f32, ldb: i32, x: ^f32, ldx: i32, ferr: ^f32, berr: ^f32, work: ^f32, iwork: ^i32) -> i32 ---
	LAPACKE_dgerfs_work :: proc(matrix_layout: c.int, trans: c.char, n: i32, nrhs: i32, a: ^f64, lda: i32, af: ^f64, ldaf: i32, ipiv: ^i32, b: ^f64, ldb: i32, x: ^f64, ldx: i32, ferr: ^f64, berr: ^f64, work: ^f64, iwork: ^i32) -> i32 ---
	LAPACKE_cgerfs_work :: proc(matrix_layout: c.int, trans: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, af: ^complex64, ldaf: i32, ipiv: ^i32, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, ferr: ^f32, berr: ^f32, work: ^complex64, rwork: ^f32) -> i32 ---
	LAPACKE_zgerfs_work :: proc(matrix_layout: c.int, trans: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, af: ^complex128, ldaf: i32, ipiv: ^i32, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, ferr: ^f64, berr: ^f64, work: ^complex128, rwork: ^f64) -> i32 ---
	LAPACKE_sgerfsx_work :: proc(matrix_layout: c.int, trans: c.char, equed: c.char, n: i32, nrhs: i32, a: ^f32, lda: i32, af: ^f32, ldaf: i32, ipiv: ^i32, r: ^f32, _c: ^f32, b: ^f32, ldb: i32, x: ^f32, ldx: i32, rcond: ^f32, berr: ^f32, n_err_bnds: i32, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: i32, params: ^f32, work: ^f32, iwork: ^i32) -> i32 ---
	LAPACKE_dgerfsx_work :: proc(matrix_layout: c.int, trans: c.char, equed: c.char, n: i32, nrhs: i32, a: ^f64, lda: i32, af: ^f64, ldaf: i32, ipiv: ^i32, r: ^f64, _c: ^f64, b: ^f64, ldb: i32, x: ^f64, ldx: i32, rcond: ^f64, berr: ^f64, n_err_bnds: i32, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: i32, params: ^f64, work: ^f64, iwork: ^i32) -> i32 ---
	LAPACKE_cgerfsx_work :: proc(matrix_layout: c.int, trans: c.char, equed: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, af: ^complex64, ldaf: i32, ipiv: ^i32, r: ^f32, _c: ^f32, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, rcond: ^f32, berr: ^f32, n_err_bnds: i32, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: i32, params: ^f32, work: ^complex64, rwork: ^f32) -> i32 ---
	LAPACKE_zgerfsx_work :: proc(matrix_layout: c.int, trans: c.char, equed: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, af: ^complex128, ldaf: i32, ipiv: ^i32, r: ^f64, _c: ^f64, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, rcond: ^f64, berr: ^f64, n_err_bnds: i32, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: i32, params: ^f64, work: ^complex128, rwork: ^f64) -> i32 ---
	LAPACKE_sgerqf_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f32, lda: i32, tau: ^f32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_dgerqf_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f64, lda: i32, tau: ^f64, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_cgerqf_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex64, lda: i32, tau: ^complex64, work: ^complex64, lwork: i32) -> i32 ---
	LAPACKE_zgerqf_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex128, lda: i32, tau: ^complex128, work: ^complex128, lwork: i32) -> i32 ---
	LAPACKE_sgesdd_work :: proc(matrix_layout: c.int, jobz: c.char, m: i32, n: i32, a: ^f32, lda: i32, s: ^f32, u: ^f32, ldu: i32, vt: ^f32, ldvt: i32, work: ^f32, lwork: i32, iwork: ^i32) -> i32 ---
	LAPACKE_dgesdd_work :: proc(matrix_layout: c.int, jobz: c.char, m: i32, n: i32, a: ^f64, lda: i32, s: ^f64, u: ^f64, ldu: i32, vt: ^f64, ldvt: i32, work: ^f64, lwork: i32, iwork: ^i32) -> i32 ---
	LAPACKE_cgesdd_work :: proc(matrix_layout: c.int, jobz: c.char, m: i32, n: i32, a: ^complex64, lda: i32, s: ^f32, u: ^complex64, ldu: i32, vt: ^complex64, ldvt: i32, work: ^complex64, lwork: i32, rwork: ^f32, iwork: ^i32) -> i32 ---
	LAPACKE_zgesdd_work :: proc(matrix_layout: c.int, jobz: c.char, m: i32, n: i32, a: ^complex128, lda: i32, s: ^f64, u: ^complex128, ldu: i32, vt: ^complex128, ldvt: i32, work: ^complex128, lwork: i32, rwork: ^f64, iwork: ^i32) -> i32 ---
	LAPACKE_sgedmd_work :: proc(matrix_layout: c.int, jobs: c.char, jobz: c.char, jobr: c.char, jobf: c.char, whtsvd: i32, m: i32, n: i32, x: ^f32, ldx: i32, y: ^f32, ldy: i32, nrnk: i32, tol: ^f32, k: i32, reig: ^f32, imeig: ^f32, z: ^f32, ldz: i32, res: ^f32, b: ^f32, ldb: i32, w: ^f32, ldw: i32, s: ^f32, lds: i32, work: ^f32, lwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_dgedmd_work :: proc(matrix_layout: c.int, jobs: c.char, jobz: c.char, jobr: c.char, jobf: c.char, whtsvd: i32, m: i32, n: i32, x: ^f64, ldx: i32, y: ^f64, ldy: i32, nrnk: i32, tol: ^f64, k: i32, reig: ^f64, imeig: ^f64, z: ^f64, ldz: i32, res: ^f64, b: ^f64, ldb: i32, w: ^f64, ldw: i32, s: ^f64, lds: i32, work: ^f64, lwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_cgedmd_work :: proc(matrix_layout: c.int, jobs: c.char, jobz: c.char, jobr: c.char, jobf: c.char, whtsvd: i32, m: i32, n: i32, x: ^complex64, ldx: i32, y: ^complex64, ldy: i32, nrnk: i32, tol: ^f32, k: i32, eigs: ^complex64, z: ^complex64, ldz: i32, res: ^f32, b: ^complex64, ldb: i32, w: ^complex64, ldw: i32, s: ^complex64, lds: i32, zwork: ^complex64, lzwork: i32, work: ^f32, lwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_zgedmd_work :: proc(matrix_layout: c.int, jobs: c.char, jobz: c.char, jobr: c.char, jobf: c.char, whtsvd: i32, m: i32, n: i32, x: ^complex128, ldx: i32, y: ^complex128, ldy: i32, nrnk: i32, tol: ^f64, k: i32, eigs: ^complex128, z: ^complex128, ldz: i32, res: ^f64, b: ^complex128, ldb: i32, w: ^complex128, ldw: i32, s: ^complex128, lds: i32, zwork: ^complex128, lzwork: i32, work: ^f64, lwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_sgedmdq_work :: proc(matrix_layout: c.int, jobs: c.char, jobz: c.char, jobr: c.char, jobq: c.char, jobt: c.char, jobf: c.char, whtsvd: i32, m: i32, n: i32, f: ^f32, ldf: i32, x: ^f32, ldx: i32, y: ^f32, ldy: i32, nrnk: i32, tol: ^f32, k: i32, reig: ^f32, imeig: ^f32, z: ^f32, ldz: i32, res: ^f32, b: ^f32, ldb: i32, v: ^f32, ldv: i32, s: ^f32, lds: i32, work: ^f32, lwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_dgedmdq_work :: proc(matrix_layout: c.int, jobs: c.char, jobz: c.char, jobr: c.char, jobq: c.char, jobt: c.char, jobf: c.char, whtsvd: i32, m: i32, n: i32, f: ^f64, ldf: i32, x: ^f64, ldx: i32, y: ^f64, ldy: i32, nrnk: i32, tol: ^f64, k: i32, reig: ^f64, imeig: ^f64, z: ^f64, ldz: i32, res: ^f64, b: ^f64, ldb: i32, v: ^f64, ldv: i32, s: ^f64, lds: i32, work: ^f64, lwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_cgedmdq_work :: proc(matrix_layout: c.int, jobs: c.char, jobz: c.char, jobr: c.char, jobq: c.char, jobt: c.char, jobf: c.char, whtsvd: i32, m: i32, n: i32, f: ^complex64, ldf: i32, x: ^complex64, ldx: i32, y: ^complex64, ldy: i32, nrnk: i32, tol: ^f32, k: i32, eigs: ^complex64, z: ^complex64, ldz: i32, res: ^f32, b: ^complex64, ldb: i32, v: ^complex64, ldv: i32, s: ^complex64, lds: i32, zwork: ^complex64, lzwork: i32, work: ^f32, lwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_zgedmdq_work :: proc(matrix_layout: c.int, jobs: c.char, jobz: c.char, jobr: c.char, jobq: c.char, jobt: c.char, jobf: c.char, whtsvd: i32, m: i32, n: i32, f: ^complex128, ldf: i32, x: ^complex128, ldx: i32, y: ^complex128, ldy: i32, nrnk: i32, tol: ^f64, k: i32, eigs: ^complex128, z: ^complex128, ldz: i32, res: ^f64, b: ^complex128, ldb: i32, v: ^complex128, ldv: i32, s: ^complex128, lds: i32, zwork: ^complex128, lzwork: i32, work: ^f64, lwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_sgesv_work :: proc(matrix_layout: c.int, n: i32, nrhs: i32, a: ^f32, lda: i32, ipiv: ^i32, b: ^f32, ldb: i32) -> i32 ---
	LAPACKE_dgesv_work :: proc(matrix_layout: c.int, n: i32, nrhs: i32, a: ^f64, lda: i32, ipiv: ^i32, b: ^f64, ldb: i32) -> i32 ---
	LAPACKE_cgesv_work :: proc(matrix_layout: c.int, n: i32, nrhs: i32, a: ^complex64, lda: i32, ipiv: ^i32, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_zgesv_work :: proc(matrix_layout: c.int, n: i32, nrhs: i32, a: ^complex128, lda: i32, ipiv: ^i32, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_dsgesv_work :: proc(matrix_layout: c.int, n: i32, nrhs: i32, a: ^f64, lda: i32, ipiv: ^i32, b: ^f64, ldb: i32, x: ^f64, ldx: i32, work: ^f64, swork: ^f32, iter: ^i32) -> i32 ---
	LAPACKE_zcgesv_work :: proc(matrix_layout: c.int, n: i32, nrhs: i32, a: ^complex128, lda: i32, ipiv: ^i32, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, work: ^complex128, swork: ^complex64, rwork: ^f64, iter: ^i32) -> i32 ---
	LAPACKE_sgesvd_work :: proc(matrix_layout: c.int, jobu: c.char, jobvt: c.char, m: i32, n: i32, a: ^f32, lda: i32, s: ^f32, u: ^f32, ldu: i32, vt: ^f32, ldvt: i32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_dgesvd_work :: proc(matrix_layout: c.int, jobu: c.char, jobvt: c.char, m: i32, n: i32, a: ^f64, lda: i32, s: ^f64, u: ^f64, ldu: i32, vt: ^f64, ldvt: i32, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_cgesvd_work :: proc(matrix_layout: c.int, jobu: c.char, jobvt: c.char, m: i32, n: i32, a: ^complex64, lda: i32, s: ^f32, u: ^complex64, ldu: i32, vt: ^complex64, ldvt: i32, work: ^complex64, lwork: i32, rwork: ^f32) -> i32 ---
	LAPACKE_zgesvd_work :: proc(matrix_layout: c.int, jobu: c.char, jobvt: c.char, m: i32, n: i32, a: ^complex128, lda: i32, s: ^f64, u: ^complex128, ldu: i32, vt: ^complex128, ldvt: i32, work: ^complex128, lwork: i32, rwork: ^f64) -> i32 ---
	LAPACKE_sgesvdx_work :: proc(matrix_layout: c.int, jobu: c.char, jobvt: c.char, range: c.char, m: i32, n: i32, a: ^f32, lda: i32, vl: f32, vu: f32, il: i32, iu: i32, ns: ^i32, s: ^f32, u: ^f32, ldu: i32, vt: ^f32, ldvt: i32, work: ^f32, lwork: i32, iwork: ^i32) -> i32 ---
	LAPACKE_dgesvdx_work :: proc(matrix_layout: c.int, jobu: c.char, jobvt: c.char, range: c.char, m: i32, n: i32, a: ^f64, lda: i32, vl: f64, vu: f64, il: i32, iu: i32, ns: ^i32, s: ^f64, u: ^f64, ldu: i32, vt: ^f64, ldvt: i32, work: ^f64, lwork: i32, iwork: ^i32) -> i32 ---
	LAPACKE_cgesvdx_work :: proc(matrix_layout: c.int, jobu: c.char, jobvt: c.char, range: c.char, m: i32, n: i32, a: ^complex64, lda: i32, vl: f32, vu: f32, il: i32, iu: i32, ns: ^i32, s: ^f32, u: ^complex64, ldu: i32, vt: ^complex64, ldvt: i32, work: ^complex64, lwork: i32, rwork: ^f32, iwork: ^i32) -> i32 ---
	LAPACKE_zgesvdx_work :: proc(matrix_layout: c.int, jobu: c.char, jobvt: c.char, range: c.char, m: i32, n: i32, a: ^complex128, lda: i32, vl: f64, vu: f64, il: i32, iu: i32, ns: ^i32, s: ^f64, u: ^complex128, ldu: i32, vt: ^complex128, ldvt: i32, work: ^complex128, lwork: i32, rwork: ^f64, iwork: ^i32) -> i32 ---
	LAPACKE_sgesvdq_work :: proc(matrix_layout: c.int, joba: c.char, jobp: c.char, jobr: c.char, jobu: c.char, jobv: c.char, m: i32, n: i32, a: ^f32, lda: i32, s: ^f32, u: ^f32, ldu: i32, v: ^f32, ldv: i32, numrank: ^i32, iwork: ^i32, liwork: i32, work: ^f32, lwork: i32, rwork: ^f32, lrwork: i32) -> i32 ---
	LAPACKE_dgesvdq_work :: proc(matrix_layout: c.int, joba: c.char, jobp: c.char, jobr: c.char, jobu: c.char, jobv: c.char, m: i32, n: i32, a: ^f64, lda: i32, s: ^f64, u: ^f64, ldu: i32, v: ^f64, ldv: i32, numrank: ^i32, iwork: ^i32, liwork: i32, work: ^f64, lwork: i32, rwork: ^f64, lrwork: i32) -> i32 ---
	LAPACKE_cgesvdq_work :: proc(matrix_layout: c.int, joba: c.char, jobp: c.char, jobr: c.char, jobu: c.char, jobv: c.char, m: i32, n: i32, a: ^complex64, lda: i32, s: ^f32, u: ^complex64, ldu: i32, v: ^complex64, ldv: i32, numrank: ^i32, iwork: ^i32, liwork: i32, cwork: ^complex64, lcwork: i32, rwork: ^f32, lrwork: i32) -> i32 ---
	LAPACKE_zgesvdq_work :: proc(matrix_layout: c.int, joba: c.char, jobp: c.char, jobr: c.char, jobu: c.char, jobv: c.char, m: i32, n: i32, a: ^complex128, lda: i32, s: ^f64, u: ^complex128, ldu: i32, v: ^complex128, ldv: i32, numrank: ^i32, iwork: ^i32, liwork: i32, cwork: ^complex128, lcwork: i32, rwork: ^f64, lrwork: i32) -> i32 ---
	LAPACKE_sgesvj_work :: proc(matrix_layout: c.int, joba: c.char, jobu: c.char, jobv: c.char, m: i32, n: i32, a: ^f32, lda: i32, sva: ^f32, mv: i32, v: ^f32, ldv: i32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_dgesvj_work :: proc(matrix_layout: c.int, joba: c.char, jobu: c.char, jobv: c.char, m: i32, n: i32, a: ^f64, lda: i32, sva: ^f64, mv: i32, v: ^f64, ldv: i32, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_cgesvj_work :: proc(matrix_layout: c.int, joba: c.char, jobu: c.char, jobv: c.char, m: i32, n: i32, a: ^complex64, lda: i32, sva: ^f32, mv: i32, v: ^complex64, ldv: i32, cwork: ^complex64, lwork: i32, rwork: ^f32, lrwork: i32) -> i32 ---
	LAPACKE_zgesvj_work :: proc(matrix_layout: c.int, joba: c.char, jobu: c.char, jobv: c.char, m: i32, n: i32, a: ^complex128, lda: i32, sva: ^f64, mv: i32, v: ^complex128, ldv: i32, cwork: ^complex128, lwork: i32, rwork: ^f64, lrwork: i32) -> i32 ---
	LAPACKE_sgesvx_work :: proc(matrix_layout: c.int, fact: c.char, trans: c.char, n: i32, nrhs: i32, a: ^f32, lda: i32, af: ^f32, ldaf: i32, ipiv: ^i32, equed: cstring, r: ^f32, _c: ^f32, b: ^f32, ldb: i32, x: ^f32, ldx: i32, rcond: ^f32, ferr: ^f32, berr: ^f32, work: ^f32, iwork: ^i32) -> i32 ---
	LAPACKE_dgesvx_work :: proc(matrix_layout: c.int, fact: c.char, trans: c.char, n: i32, nrhs: i32, a: ^f64, lda: i32, af: ^f64, ldaf: i32, ipiv: ^i32, equed: cstring, r: ^f64, _c: ^f64, b: ^f64, ldb: i32, x: ^f64, ldx: i32, rcond: ^f64, ferr: ^f64, berr: ^f64, work: ^f64, iwork: ^i32) -> i32 ---
	LAPACKE_cgesvx_work :: proc(matrix_layout: c.int, fact: c.char, trans: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, af: ^complex64, ldaf: i32, ipiv: ^i32, equed: cstring, r: ^f32, _c: ^f32, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, rcond: ^f32, ferr: ^f32, berr: ^f32, work: ^complex64, rwork: ^f32) -> i32 ---
	LAPACKE_zgesvx_work :: proc(matrix_layout: c.int, fact: c.char, trans: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, af: ^complex128, ldaf: i32, ipiv: ^i32, equed: cstring, r: ^f64, _c: ^f64, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, rcond: ^f64, ferr: ^f64, berr: ^f64, work: ^complex128, rwork: ^f64) -> i32 ---
	LAPACKE_sgesvxx_work :: proc(matrix_layout: c.int, fact: c.char, trans: c.char, n: i32, nrhs: i32, a: ^f32, lda: i32, af: ^f32, ldaf: i32, ipiv: ^i32, equed: cstring, r: ^f32, _c: ^f32, b: ^f32, ldb: i32, x: ^f32, ldx: i32, rcond: ^f32, rpvgrw: ^f32, berr: ^f32, n_err_bnds: i32, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: i32, params: ^f32, work: ^f32, iwork: ^i32) -> i32 ---
	LAPACKE_dgesvxx_work :: proc(matrix_layout: c.int, fact: c.char, trans: c.char, n: i32, nrhs: i32, a: ^f64, lda: i32, af: ^f64, ldaf: i32, ipiv: ^i32, equed: cstring, r: ^f64, _c: ^f64, b: ^f64, ldb: i32, x: ^f64, ldx: i32, rcond: ^f64, rpvgrw: ^f64, berr: ^f64, n_err_bnds: i32, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: i32, params: ^f64, work: ^f64, iwork: ^i32) -> i32 ---
	LAPACKE_cgesvxx_work :: proc(matrix_layout: c.int, fact: c.char, trans: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, af: ^complex64, ldaf: i32, ipiv: ^i32, equed: cstring, r: ^f32, _c: ^f32, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, rcond: ^f32, rpvgrw: ^f32, berr: ^f32, n_err_bnds: i32, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: i32, params: ^f32, work: ^complex64, rwork: ^f32) -> i32 ---
	LAPACKE_zgesvxx_work :: proc(matrix_layout: c.int, fact: c.char, trans: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, af: ^complex128, ldaf: i32, ipiv: ^i32, equed: cstring, r: ^f64, _c: ^f64, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, rcond: ^f64, rpvgrw: ^f64, berr: ^f64, n_err_bnds: i32, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: i32, params: ^f64, work: ^complex128, rwork: ^f64) -> i32 ---
	LAPACKE_sgetf2_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f32, lda: i32, ipiv: ^i32) -> i32 ---
	LAPACKE_dgetf2_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f64, lda: i32, ipiv: ^i32) -> i32 ---
	LAPACKE_cgetf2_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex64, lda: i32, ipiv: ^i32) -> i32 ---
	LAPACKE_zgetf2_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex128, lda: i32, ipiv: ^i32) -> i32 ---
	LAPACKE_sgetrf_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f32, lda: i32, ipiv: ^i32) -> i32 ---
	LAPACKE_dgetrf_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f64, lda: i32, ipiv: ^i32) -> i32 ---
	LAPACKE_cgetrf_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex64, lda: i32, ipiv: ^i32) -> i32 ---
	LAPACKE_zgetrf_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex128, lda: i32, ipiv: ^i32) -> i32 ---
	LAPACKE_sgetrf2_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f32, lda: i32, ipiv: ^i32) -> i32 ---
	LAPACKE_dgetrf2_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f64, lda: i32, ipiv: ^i32) -> i32 ---
	LAPACKE_cgetrf2_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex64, lda: i32, ipiv: ^i32) -> i32 ---
	LAPACKE_zgetrf2_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex128, lda: i32, ipiv: ^i32) -> i32 ---
	LAPACKE_sgetri_work :: proc(matrix_layout: c.int, n: i32, a: ^f32, lda: i32, ipiv: ^i32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_dgetri_work :: proc(matrix_layout: c.int, n: i32, a: ^f64, lda: i32, ipiv: ^i32, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_cgetri_work :: proc(matrix_layout: c.int, n: i32, a: ^complex64, lda: i32, ipiv: ^i32, work: ^complex64, lwork: i32) -> i32 ---
	LAPACKE_zgetri_work :: proc(matrix_layout: c.int, n: i32, a: ^complex128, lda: i32, ipiv: ^i32, work: ^complex128, lwork: i32) -> i32 ---
	LAPACKE_sgetrs_work :: proc(matrix_layout: c.int, trans: c.char, n: i32, nrhs: i32, a: ^f32, lda: i32, ipiv: ^i32, b: ^f32, ldb: i32) -> i32 ---
	LAPACKE_dgetrs_work :: proc(matrix_layout: c.int, trans: c.char, n: i32, nrhs: i32, a: ^f64, lda: i32, ipiv: ^i32, b: ^f64, ldb: i32) -> i32 ---
	LAPACKE_cgetrs_work :: proc(matrix_layout: c.int, trans: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, ipiv: ^i32, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_zgetrs_work :: proc(matrix_layout: c.int, trans: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, ipiv: ^i32, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_sggbak_work :: proc(matrix_layout: c.int, job: c.char, side: c.char, n: i32, ilo: i32, ihi: i32, lscale: ^f32, rscale: ^f32, m: i32, v: ^f32, ldv: i32) -> i32 ---
	LAPACKE_dggbak_work :: proc(matrix_layout: c.int, job: c.char, side: c.char, n: i32, ilo: i32, ihi: i32, lscale: ^f64, rscale: ^f64, m: i32, v: ^f64, ldv: i32) -> i32 ---
	LAPACKE_cggbak_work :: proc(matrix_layout: c.int, job: c.char, side: c.char, n: i32, ilo: i32, ihi: i32, lscale: ^f32, rscale: ^f32, m: i32, v: ^complex64, ldv: i32) -> i32 ---
	LAPACKE_zggbak_work :: proc(matrix_layout: c.int, job: c.char, side: c.char, n: i32, ilo: i32, ihi: i32, lscale: ^f64, rscale: ^f64, m: i32, v: ^complex128, ldv: i32) -> i32 ---
	LAPACKE_sggbal_work :: proc(matrix_layout: c.int, job: c.char, n: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32, ilo: ^i32, ihi: ^i32, lscale: ^f32, rscale: ^f32, work: ^f32) -> i32 ---
	LAPACKE_dggbal_work :: proc(matrix_layout: c.int, job: c.char, n: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, ilo: ^i32, ihi: ^i32, lscale: ^f64, rscale: ^f64, work: ^f64) -> i32 ---
	LAPACKE_cggbal_work :: proc(matrix_layout: c.int, job: c.char, n: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32, ilo: ^i32, ihi: ^i32, lscale: ^f32, rscale: ^f32, work: ^f32) -> i32 ---
	LAPACKE_zggbal_work :: proc(matrix_layout: c.int, job: c.char, n: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, ilo: ^i32, ihi: ^i32, lscale: ^f64, rscale: ^f64, work: ^f64) -> i32 ---
	LAPACKE_sgges_work :: proc(matrix_layout: c.int, jobvsl: c.char, jobvsr: c.char, sort: c.char, selctg: LAPACK_S_SELECT3, n: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32, sdim: ^i32, alphar: ^f32, alphai: ^f32, beta: ^f32, vsl: ^f32, ldvsl: i32, vsr: ^f32, ldvsr: i32, work: ^f32, lwork: i32, bwork: ^i32) -> i32 ---
	LAPACKE_dgges_work :: proc(matrix_layout: c.int, jobvsl: c.char, jobvsr: c.char, sort: c.char, selctg: LAPACK_D_SELECT3, n: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, sdim: ^i32, alphar: ^f64, alphai: ^f64, beta: ^f64, vsl: ^f64, ldvsl: i32, vsr: ^f64, ldvsr: i32, work: ^f64, lwork: i32, bwork: ^i32) -> i32 ---
	LAPACKE_cgges_work :: proc(matrix_layout: c.int, jobvsl: c.char, jobvsr: c.char, sort: c.char, selctg: LAPACK_C_SELECT2, n: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32, sdim: ^i32, alpha: ^complex64, beta: ^complex64, vsl: ^complex64, ldvsl: i32, vsr: ^complex64, ldvsr: i32, work: ^complex64, lwork: i32, rwork: ^f32, bwork: ^i32) -> i32 ---
	LAPACKE_zgges_work :: proc(matrix_layout: c.int, jobvsl: c.char, jobvsr: c.char, sort: c.char, selctg: LAPACK_Z_SELECT2, n: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, sdim: ^i32, alpha: ^complex128, beta: ^complex128, vsl: ^complex128, ldvsl: i32, vsr: ^complex128, ldvsr: i32, work: ^complex128, lwork: i32, rwork: ^f64, bwork: ^i32) -> i32 ---
	LAPACKE_sgges3_work :: proc(matrix_layout: c.int, jobvsl: c.char, jobvsr: c.char, sort: c.char, selctg: LAPACK_S_SELECT3, n: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32, sdim: ^i32, alphar: ^f32, alphai: ^f32, beta: ^f32, vsl: ^f32, ldvsl: i32, vsr: ^f32, ldvsr: i32, work: ^f32, lwork: i32, bwork: ^i32) -> i32 ---
	LAPACKE_dgges3_work :: proc(matrix_layout: c.int, jobvsl: c.char, jobvsr: c.char, sort: c.char, selctg: LAPACK_D_SELECT3, n: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, sdim: ^i32, alphar: ^f64, alphai: ^f64, beta: ^f64, vsl: ^f64, ldvsl: i32, vsr: ^f64, ldvsr: i32, work: ^f64, lwork: i32, bwork: ^i32) -> i32 ---
	LAPACKE_cgges3_work :: proc(matrix_layout: c.int, jobvsl: c.char, jobvsr: c.char, sort: c.char, selctg: LAPACK_C_SELECT2, n: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32, sdim: ^i32, alpha: ^complex64, beta: ^complex64, vsl: ^complex64, ldvsl: i32, vsr: ^complex64, ldvsr: i32, work: ^complex64, lwork: i32, rwork: ^f32, bwork: ^i32) -> i32 ---
	LAPACKE_zgges3_work :: proc(matrix_layout: c.int, jobvsl: c.char, jobvsr: c.char, sort: c.char, selctg: LAPACK_Z_SELECT2, n: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, sdim: ^i32, alpha: ^complex128, beta: ^complex128, vsl: ^complex128, ldvsl: i32, vsr: ^complex128, ldvsr: i32, work: ^complex128, lwork: i32, rwork: ^f64, bwork: ^i32) -> i32 ---
	LAPACKE_sggesx_work :: proc(matrix_layout: c.int, jobvsl: c.char, jobvsr: c.char, sort: c.char, selctg: LAPACK_S_SELECT3, sense: c.char, n: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32, sdim: ^i32, alphar: ^f32, alphai: ^f32, beta: ^f32, vsl: ^f32, ldvsl: i32, vsr: ^f32, ldvsr: i32, rconde: ^f32, rcondv: ^f32, work: ^f32, lwork: i32, iwork: ^i32, liwork: i32, bwork: ^i32) -> i32 ---
	LAPACKE_dggesx_work :: proc(matrix_layout: c.int, jobvsl: c.char, jobvsr: c.char, sort: c.char, selctg: LAPACK_D_SELECT3, sense: c.char, n: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, sdim: ^i32, alphar: ^f64, alphai: ^f64, beta: ^f64, vsl: ^f64, ldvsl: i32, vsr: ^f64, ldvsr: i32, rconde: ^f64, rcondv: ^f64, work: ^f64, lwork: i32, iwork: ^i32, liwork: i32, bwork: ^i32) -> i32 ---
	LAPACKE_cggesx_work :: proc(matrix_layout: c.int, jobvsl: c.char, jobvsr: c.char, sort: c.char, selctg: LAPACK_C_SELECT2, sense: c.char, n: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32, sdim: ^i32, alpha: ^complex64, beta: ^complex64, vsl: ^complex64, ldvsl: i32, vsr: ^complex64, ldvsr: i32, rconde: ^f32, rcondv: ^f32, work: ^complex64, lwork: i32, rwork: ^f32, iwork: ^i32, liwork: i32, bwork: ^i32) -> i32 ---
	LAPACKE_zggesx_work :: proc(matrix_layout: c.int, jobvsl: c.char, jobvsr: c.char, sort: c.char, selctg: LAPACK_Z_SELECT2, sense: c.char, n: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, sdim: ^i32, alpha: ^complex128, beta: ^complex128, vsl: ^complex128, ldvsl: i32, vsr: ^complex128, ldvsr: i32, rconde: ^f64, rcondv: ^f64, work: ^complex128, lwork: i32, rwork: ^f64, iwork: ^i32, liwork: i32, bwork: ^i32) -> i32 ---
	LAPACKE_sggev_work :: proc(matrix_layout: c.int, jobvl: c.char, jobvr: c.char, n: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32, alphar: ^f32, alphai: ^f32, beta: ^f32, vl: ^f32, ldvl: i32, vr: ^f32, ldvr: i32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_dggev_work :: proc(matrix_layout: c.int, jobvl: c.char, jobvr: c.char, n: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, alphar: ^f64, alphai: ^f64, beta: ^f64, vl: ^f64, ldvl: i32, vr: ^f64, ldvr: i32, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_cggev_work :: proc(matrix_layout: c.int, jobvl: c.char, jobvr: c.char, n: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32, alpha: ^complex64, beta: ^complex64, vl: ^complex64, ldvl: i32, vr: ^complex64, ldvr: i32, work: ^complex64, lwork: i32, rwork: ^f32) -> i32 ---
	LAPACKE_zggev_work :: proc(matrix_layout: c.int, jobvl: c.char, jobvr: c.char, n: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, alpha: ^complex128, beta: ^complex128, vl: ^complex128, ldvl: i32, vr: ^complex128, ldvr: i32, work: ^complex128, lwork: i32, rwork: ^f64) -> i32 ---
	LAPACKE_sggev3_work :: proc(matrix_layout: c.int, jobvl: c.char, jobvr: c.char, n: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32, alphar: ^f32, alphai: ^f32, beta: ^f32, vl: ^f32, ldvl: i32, vr: ^f32, ldvr: i32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_dggev3_work :: proc(matrix_layout: c.int, jobvl: c.char, jobvr: c.char, n: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, alphar: ^f64, alphai: ^f64, beta: ^f64, vl: ^f64, ldvl: i32, vr: ^f64, ldvr: i32, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_cggev3_work :: proc(matrix_layout: c.int, jobvl: c.char, jobvr: c.char, n: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32, alpha: ^complex64, beta: ^complex64, vl: ^complex64, ldvl: i32, vr: ^complex64, ldvr: i32, work: ^complex64, lwork: i32, rwork: ^f32) -> i32 ---
	LAPACKE_zggev3_work :: proc(matrix_layout: c.int, jobvl: c.char, jobvr: c.char, n: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, alpha: ^complex128, beta: ^complex128, vl: ^complex128, ldvl: i32, vr: ^complex128, ldvr: i32, work: ^complex128, lwork: i32, rwork: ^f64) -> i32 ---
	LAPACKE_sggevx_work :: proc(matrix_layout: c.int, balanc: c.char, jobvl: c.char, jobvr: c.char, sense: c.char, n: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32, alphar: ^f32, alphai: ^f32, beta: ^f32, vl: ^f32, ldvl: i32, vr: ^f32, ldvr: i32, ilo: ^i32, ihi: ^i32, lscale: ^f32, rscale: ^f32, abnrm: ^f32, bbnrm: ^f32, rconde: ^f32, rcondv: ^f32, work: ^f32, lwork: i32, iwork: ^i32, bwork: ^i32) -> i32 ---
	LAPACKE_dggevx_work :: proc(matrix_layout: c.int, balanc: c.char, jobvl: c.char, jobvr: c.char, sense: c.char, n: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, alphar: ^f64, alphai: ^f64, beta: ^f64, vl: ^f64, ldvl: i32, vr: ^f64, ldvr: i32, ilo: ^i32, ihi: ^i32, lscale: ^f64, rscale: ^f64, abnrm: ^f64, bbnrm: ^f64, rconde: ^f64, rcondv: ^f64, work: ^f64, lwork: i32, iwork: ^i32, bwork: ^i32) -> i32 ---
	LAPACKE_cggevx_work :: proc(matrix_layout: c.int, balanc: c.char, jobvl: c.char, jobvr: c.char, sense: c.char, n: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32, alpha: ^complex64, beta: ^complex64, vl: ^complex64, ldvl: i32, vr: ^complex64, ldvr: i32, ilo: ^i32, ihi: ^i32, lscale: ^f32, rscale: ^f32, abnrm: ^f32, bbnrm: ^f32, rconde: ^f32, rcondv: ^f32, work: ^complex64, lwork: i32, rwork: ^f32, iwork: ^i32, bwork: ^i32) -> i32 ---
	LAPACKE_zggevx_work :: proc(matrix_layout: c.int, balanc: c.char, jobvl: c.char, jobvr: c.char, sense: c.char, n: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, alpha: ^complex128, beta: ^complex128, vl: ^complex128, ldvl: i32, vr: ^complex128, ldvr: i32, ilo: ^i32, ihi: ^i32, lscale: ^f64, rscale: ^f64, abnrm: ^f64, bbnrm: ^f64, rconde: ^f64, rcondv: ^f64, work: ^complex128, lwork: i32, rwork: ^f64, iwork: ^i32, bwork: ^i32) -> i32 ---
	LAPACKE_sggglm_work :: proc(matrix_layout: c.int, n: i32, m: i32, p: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32, d: ^f32, x: ^f32, y: ^f32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_dggglm_work :: proc(matrix_layout: c.int, n: i32, m: i32, p: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, d: ^f64, x: ^f64, y: ^f64, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_cggglm_work :: proc(matrix_layout: c.int, n: i32, m: i32, p: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32, d: ^complex64, x: ^complex64, y: ^complex64, work: ^complex64, lwork: i32) -> i32 ---
	LAPACKE_zggglm_work :: proc(matrix_layout: c.int, n: i32, m: i32, p: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, d: ^complex128, x: ^complex128, y: ^complex128, work: ^complex128, lwork: i32) -> i32 ---
	LAPACKE_sgghrd_work :: proc(matrix_layout: c.int, compq: c.char, compz: c.char, n: i32, ilo: i32, ihi: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32, q: ^f32, ldq: i32, z: ^f32, ldz: i32) -> i32 ---
	LAPACKE_dgghrd_work :: proc(matrix_layout: c.int, compq: c.char, compz: c.char, n: i32, ilo: i32, ihi: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, q: ^f64, ldq: i32, z: ^f64, ldz: i32) -> i32 ---
	LAPACKE_cgghrd_work :: proc(matrix_layout: c.int, compq: c.char, compz: c.char, n: i32, ilo: i32, ihi: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32, q: ^complex64, ldq: i32, z: ^complex64, ldz: i32) -> i32 ---
	LAPACKE_zgghrd_work :: proc(matrix_layout: c.int, compq: c.char, compz: c.char, n: i32, ilo: i32, ihi: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, q: ^complex128, ldq: i32, z: ^complex128, ldz: i32) -> i32 ---
	LAPACKE_sgghd3_work :: proc(matrix_layout: c.int, compq: c.char, compz: c.char, n: i32, ilo: i32, ihi: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32, q: ^f32, ldq: i32, z: ^f32, ldz: i32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_dgghd3_work :: proc(matrix_layout: c.int, compq: c.char, compz: c.char, n: i32, ilo: i32, ihi: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, q: ^f64, ldq: i32, z: ^f64, ldz: i32, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_cgghd3_work :: proc(matrix_layout: c.int, compq: c.char, compz: c.char, n: i32, ilo: i32, ihi: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32, q: ^complex64, ldq: i32, z: ^complex64, ldz: i32, work: ^complex64, lwork: i32) -> i32 ---
	LAPACKE_zgghd3_work :: proc(matrix_layout: c.int, compq: c.char, compz: c.char, n: i32, ilo: i32, ihi: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, q: ^complex128, ldq: i32, z: ^complex128, ldz: i32, work: ^complex128, lwork: i32) -> i32 ---
	LAPACKE_sgglse_work :: proc(matrix_layout: c.int, m: i32, n: i32, p: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32, _c: ^f32, d: ^f32, x: ^f32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_dgglse_work :: proc(matrix_layout: c.int, m: i32, n: i32, p: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, _c: ^f64, d: ^f64, x: ^f64, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_cgglse_work :: proc(matrix_layout: c.int, m: i32, n: i32, p: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32, _c: ^complex64, d: ^complex64, x: ^complex64, work: ^complex64, lwork: i32) -> i32 ---
	LAPACKE_zgglse_work :: proc(matrix_layout: c.int, m: i32, n: i32, p: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, _c: ^complex128, d: ^complex128, x: ^complex128, work: ^complex128, lwork: i32) -> i32 ---
	LAPACKE_sggqrf_work :: proc(matrix_layout: c.int, n: i32, m: i32, p: i32, a: ^f32, lda: i32, taua: ^f32, b: ^f32, ldb: i32, taub: ^f32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_dggqrf_work :: proc(matrix_layout: c.int, n: i32, m: i32, p: i32, a: ^f64, lda: i32, taua: ^f64, b: ^f64, ldb: i32, taub: ^f64, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_cggqrf_work :: proc(matrix_layout: c.int, n: i32, m: i32, p: i32, a: ^complex64, lda: i32, taua: ^complex64, b: ^complex64, ldb: i32, taub: ^complex64, work: ^complex64, lwork: i32) -> i32 ---
	LAPACKE_zggqrf_work :: proc(matrix_layout: c.int, n: i32, m: i32, p: i32, a: ^complex128, lda: i32, taua: ^complex128, b: ^complex128, ldb: i32, taub: ^complex128, work: ^complex128, lwork: i32) -> i32 ---
	LAPACKE_sggrqf_work :: proc(matrix_layout: c.int, m: i32, p: i32, n: i32, a: ^f32, lda: i32, taua: ^f32, b: ^f32, ldb: i32, taub: ^f32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_dggrqf_work :: proc(matrix_layout: c.int, m: i32, p: i32, n: i32, a: ^f64, lda: i32, taua: ^f64, b: ^f64, ldb: i32, taub: ^f64, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_cggrqf_work :: proc(matrix_layout: c.int, m: i32, p: i32, n: i32, a: ^complex64, lda: i32, taua: ^complex64, b: ^complex64, ldb: i32, taub: ^complex64, work: ^complex64, lwork: i32) -> i32 ---
	LAPACKE_zggrqf_work :: proc(matrix_layout: c.int, m: i32, p: i32, n: i32, a: ^complex128, lda: i32, taua: ^complex128, b: ^complex128, ldb: i32, taub: ^complex128, work: ^complex128, lwork: i32) -> i32 ---
	LAPACKE_sggsvd_work :: proc(matrix_layout: c.int, jobu: c.char, jobv: c.char, jobq: c.char, m: i32, n: i32, p: i32, k: ^i32, l: ^i32, a: ^f32, lda: i32, b: ^f32, ldb: i32, alpha: ^f32, beta: ^f32, u: ^f32, ldu: i32, v: ^f32, ldv: i32, q: ^f32, ldq: i32, work: ^f32, iwork: ^i32) -> i32 ---
	LAPACKE_dggsvd_work :: proc(matrix_layout: c.int, jobu: c.char, jobv: c.char, jobq: c.char, m: i32, n: i32, p: i32, k: ^i32, l: ^i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, alpha: ^f64, beta: ^f64, u: ^f64, ldu: i32, v: ^f64, ldv: i32, q: ^f64, ldq: i32, work: ^f64, iwork: ^i32) -> i32 ---
	LAPACKE_cggsvd_work :: proc(matrix_layout: c.int, jobu: c.char, jobv: c.char, jobq: c.char, m: i32, n: i32, p: i32, k: ^i32, l: ^i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32, alpha: ^f32, beta: ^f32, u: ^complex64, ldu: i32, v: ^complex64, ldv: i32, q: ^complex64, ldq: i32, work: ^complex64, rwork: ^f32, iwork: ^i32) -> i32 ---
	LAPACKE_zggsvd_work :: proc(matrix_layout: c.int, jobu: c.char, jobv: c.char, jobq: c.char, m: i32, n: i32, p: i32, k: ^i32, l: ^i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, alpha: ^f64, beta: ^f64, u: ^complex128, ldu: i32, v: ^complex128, ldv: i32, q: ^complex128, ldq: i32, work: ^complex128, rwork: ^f64, iwork: ^i32) -> i32 ---
	LAPACKE_sggsvd3_work :: proc(matrix_layout: c.int, jobu: c.char, jobv: c.char, jobq: c.char, m: i32, n: i32, p: i32, k: ^i32, l: ^i32, a: ^f32, lda: i32, b: ^f32, ldb: i32, alpha: ^f32, beta: ^f32, u: ^f32, ldu: i32, v: ^f32, ldv: i32, q: ^f32, ldq: i32, work: ^f32, lwork: i32, iwork: ^i32) -> i32 ---
	LAPACKE_dggsvd3_work :: proc(matrix_layout: c.int, jobu: c.char, jobv: c.char, jobq: c.char, m: i32, n: i32, p: i32, k: ^i32, l: ^i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, alpha: ^f64, beta: ^f64, u: ^f64, ldu: i32, v: ^f64, ldv: i32, q: ^f64, ldq: i32, work: ^f64, lwork: i32, iwork: ^i32) -> i32 ---
	LAPACKE_cggsvd3_work :: proc(matrix_layout: c.int, jobu: c.char, jobv: c.char, jobq: c.char, m: i32, n: i32, p: i32, k: ^i32, l: ^i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32, alpha: ^f32, beta: ^f32, u: ^complex64, ldu: i32, v: ^complex64, ldv: i32, q: ^complex64, ldq: i32, work: ^complex64, lwork: i32, rwork: ^f32, iwork: ^i32) -> i32 ---
	LAPACKE_zggsvd3_work :: proc(matrix_layout: c.int, jobu: c.char, jobv: c.char, jobq: c.char, m: i32, n: i32, p: i32, k: ^i32, l: ^i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, alpha: ^f64, beta: ^f64, u: ^complex128, ldu: i32, v: ^complex128, ldv: i32, q: ^complex128, ldq: i32, work: ^complex128, lwork: i32, rwork: ^f64, iwork: ^i32) -> i32 ---
	LAPACKE_sggsvp_work :: proc(matrix_layout: c.int, jobu: c.char, jobv: c.char, jobq: c.char, m: i32, p: i32, n: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32, tola: f32, tolb: f32, k: ^i32, l: ^i32, u: ^f32, ldu: i32, v: ^f32, ldv: i32, q: ^f32, ldq: i32, iwork: ^i32, tau: ^f32, work: ^f32) -> i32 ---
	LAPACKE_dggsvp_work :: proc(matrix_layout: c.int, jobu: c.char, jobv: c.char, jobq: c.char, m: i32, p: i32, n: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, tola: f64, tolb: f64, k: ^i32, l: ^i32, u: ^f64, ldu: i32, v: ^f64, ldv: i32, q: ^f64, ldq: i32, iwork: ^i32, tau: ^f64, work: ^f64) -> i32 ---
	LAPACKE_cggsvp_work :: proc(matrix_layout: c.int, jobu: c.char, jobv: c.char, jobq: c.char, m: i32, p: i32, n: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32, tola: f32, tolb: f32, k: ^i32, l: ^i32, u: ^complex64, ldu: i32, v: ^complex64, ldv: i32, q: ^complex64, ldq: i32, iwork: ^i32, rwork: ^f32, tau: ^complex64, work: ^complex64) -> i32 ---
	LAPACKE_zggsvp_work :: proc(matrix_layout: c.int, jobu: c.char, jobv: c.char, jobq: c.char, m: i32, p: i32, n: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, tola: f64, tolb: f64, k: ^i32, l: ^i32, u: ^complex128, ldu: i32, v: ^complex128, ldv: i32, q: ^complex128, ldq: i32, iwork: ^i32, rwork: ^f64, tau: ^complex128, work: ^complex128) -> i32 ---
	LAPACKE_sggsvp3_work :: proc(matrix_layout: c.int, jobu: c.char, jobv: c.char, jobq: c.char, m: i32, p: i32, n: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32, tola: f32, tolb: f32, k: ^i32, l: ^i32, u: ^f32, ldu: i32, v: ^f32, ldv: i32, q: ^f32, ldq: i32, iwork: ^i32, tau: ^f32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_dggsvp3_work :: proc(matrix_layout: c.int, jobu: c.char, jobv: c.char, jobq: c.char, m: i32, p: i32, n: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, tola: f64, tolb: f64, k: ^i32, l: ^i32, u: ^f64, ldu: i32, v: ^f64, ldv: i32, q: ^f64, ldq: i32, iwork: ^i32, tau: ^f64, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_cggsvp3_work :: proc(matrix_layout: c.int, jobu: c.char, jobv: c.char, jobq: c.char, m: i32, p: i32, n: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32, tola: f32, tolb: f32, k: ^i32, l: ^i32, u: ^complex64, ldu: i32, v: ^complex64, ldv: i32, q: ^complex64, ldq: i32, iwork: ^i32, rwork: ^f32, tau: ^complex64, work: ^complex64, lwork: i32) -> i32 ---
	LAPACKE_zggsvp3_work :: proc(matrix_layout: c.int, jobu: c.char, jobv: c.char, jobq: c.char, m: i32, p: i32, n: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, tola: f64, tolb: f64, k: ^i32, l: ^i32, u: ^complex128, ldu: i32, v: ^complex128, ldv: i32, q: ^complex128, ldq: i32, iwork: ^i32, rwork: ^f64, tau: ^complex128, work: ^complex128, lwork: i32) -> i32 ---
	LAPACKE_sgtcon_work :: proc(norm: c.char, n: i32, dl: ^f32, d: ^f32, du: ^f32, du2: ^f32, ipiv: ^i32, anorm: f32, rcond: ^f32, work: ^f32, iwork: ^i32) -> i32 ---
	LAPACKE_dgtcon_work :: proc(norm: c.char, n: i32, dl: ^f64, d: ^f64, du: ^f64, du2: ^f64, ipiv: ^i32, anorm: f64, rcond: ^f64, work: ^f64, iwork: ^i32) -> i32 ---
	LAPACKE_cgtcon_work :: proc(norm: c.char, n: i32, dl: ^complex64, d: ^complex64, du: ^complex64, du2: ^complex64, ipiv: ^i32, anorm: f32, rcond: ^f32, work: ^complex64) -> i32 ---
	LAPACKE_zgtcon_work :: proc(norm: c.char, n: i32, dl: ^complex128, d: ^complex128, du: ^complex128, du2: ^complex128, ipiv: ^i32, anorm: f64, rcond: ^f64, work: ^complex128) -> i32 ---
	LAPACKE_sgtrfs_work :: proc(matrix_layout: c.int, trans: c.char, n: i32, nrhs: i32, dl: ^f32, d: ^f32, du: ^f32, dlf: ^f32, df: ^f32, duf: ^f32, du2: ^f32, ipiv: ^i32, b: ^f32, ldb: i32, x: ^f32, ldx: i32, ferr: ^f32, berr: ^f32, work: ^f32, iwork: ^i32) -> i32 ---
	LAPACKE_dgtrfs_work :: proc(matrix_layout: c.int, trans: c.char, n: i32, nrhs: i32, dl: ^f64, d: ^f64, du: ^f64, dlf: ^f64, df: ^f64, duf: ^f64, du2: ^f64, ipiv: ^i32, b: ^f64, ldb: i32, x: ^f64, ldx: i32, ferr: ^f64, berr: ^f64, work: ^f64, iwork: ^i32) -> i32 ---
	LAPACKE_cgtrfs_work :: proc(matrix_layout: c.int, trans: c.char, n: i32, nrhs: i32, dl: ^complex64, d: ^complex64, du: ^complex64, dlf: ^complex64, df: ^complex64, duf: ^complex64, du2: ^complex64, ipiv: ^i32, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, ferr: ^f32, berr: ^f32, work: ^complex64, rwork: ^f32) -> i32 ---
	LAPACKE_zgtrfs_work :: proc(matrix_layout: c.int, trans: c.char, n: i32, nrhs: i32, dl: ^complex128, d: ^complex128, du: ^complex128, dlf: ^complex128, df: ^complex128, duf: ^complex128, du2: ^complex128, ipiv: ^i32, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, ferr: ^f64, berr: ^f64, work: ^complex128, rwork: ^f64) -> i32 ---
	LAPACKE_sgtsv_work :: proc(matrix_layout: c.int, n: i32, nrhs: i32, dl: ^f32, d: ^f32, du: ^f32, b: ^f32, ldb: i32) -> i32 ---
	LAPACKE_dgtsv_work :: proc(matrix_layout: c.int, n: i32, nrhs: i32, dl: ^f64, d: ^f64, du: ^f64, b: ^f64, ldb: i32) -> i32 ---
	LAPACKE_cgtsv_work :: proc(matrix_layout: c.int, n: i32, nrhs: i32, dl: ^complex64, d: ^complex64, du: ^complex64, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_zgtsv_work :: proc(matrix_layout: c.int, n: i32, nrhs: i32, dl: ^complex128, d: ^complex128, du: ^complex128, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_sgtsvx_work :: proc(matrix_layout: c.int, fact: c.char, trans: c.char, n: i32, nrhs: i32, dl: ^f32, d: ^f32, du: ^f32, dlf: ^f32, df: ^f32, duf: ^f32, du2: ^f32, ipiv: ^i32, b: ^f32, ldb: i32, x: ^f32, ldx: i32, rcond: ^f32, ferr: ^f32, berr: ^f32, work: ^f32, iwork: ^i32) -> i32 ---
	LAPACKE_dgtsvx_work :: proc(matrix_layout: c.int, fact: c.char, trans: c.char, n: i32, nrhs: i32, dl: ^f64, d: ^f64, du: ^f64, dlf: ^f64, df: ^f64, duf: ^f64, du2: ^f64, ipiv: ^i32, b: ^f64, ldb: i32, x: ^f64, ldx: i32, rcond: ^f64, ferr: ^f64, berr: ^f64, work: ^f64, iwork: ^i32) -> i32 ---
	LAPACKE_cgtsvx_work :: proc(matrix_layout: c.int, fact: c.char, trans: c.char, n: i32, nrhs: i32, dl: ^complex64, d: ^complex64, du: ^complex64, dlf: ^complex64, df: ^complex64, duf: ^complex64, du2: ^complex64, ipiv: ^i32, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, rcond: ^f32, ferr: ^f32, berr: ^f32, work: ^complex64, rwork: ^f32) -> i32 ---
	LAPACKE_zgtsvx_work :: proc(matrix_layout: c.int, fact: c.char, trans: c.char, n: i32, nrhs: i32, dl: ^complex128, d: ^complex128, du: ^complex128, dlf: ^complex128, df: ^complex128, duf: ^complex128, du2: ^complex128, ipiv: ^i32, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, rcond: ^f64, ferr: ^f64, berr: ^f64, work: ^complex128, rwork: ^f64) -> i32 ---
	LAPACKE_sgttrf_work :: proc(n: i32, dl: ^f32, d: ^f32, du: ^f32, du2: ^f32, ipiv: ^i32) -> i32 ---
	LAPACKE_dgttrf_work :: proc(n: i32, dl: ^f64, d: ^f64, du: ^f64, du2: ^f64, ipiv: ^i32) -> i32 ---
	LAPACKE_cgttrf_work :: proc(n: i32, dl: ^complex64, d: ^complex64, du: ^complex64, du2: ^complex64, ipiv: ^i32) -> i32 ---
	LAPACKE_zgttrf_work :: proc(n: i32, dl: ^complex128, d: ^complex128, du: ^complex128, du2: ^complex128, ipiv: ^i32) -> i32 ---
	LAPACKE_sgttrs_work :: proc(matrix_layout: c.int, trans: c.char, n: i32, nrhs: i32, dl: ^f32, d: ^f32, du: ^f32, du2: ^f32, ipiv: ^i32, b: ^f32, ldb: i32) -> i32 ---
	LAPACKE_dgttrs_work :: proc(matrix_layout: c.int, trans: c.char, n: i32, nrhs: i32, dl: ^f64, d: ^f64, du: ^f64, du2: ^f64, ipiv: ^i32, b: ^f64, ldb: i32) -> i32 ---
	LAPACKE_cgttrs_work :: proc(matrix_layout: c.int, trans: c.char, n: i32, nrhs: i32, dl: ^complex64, d: ^complex64, du: ^complex64, du2: ^complex64, ipiv: ^i32, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_zgttrs_work :: proc(matrix_layout: c.int, trans: c.char, n: i32, nrhs: i32, dl: ^complex128, d: ^complex128, du: ^complex128, du2: ^complex128, ipiv: ^i32, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_chbev_work :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, kd: i32, ab: ^complex64, ldab: i32, w: ^f32, z: ^complex64, ldz: i32, work: ^complex64, rwork: ^f32) -> i32 ---
	LAPACKE_zhbev_work :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, kd: i32, ab: ^complex128, ldab: i32, w: ^f64, z: ^complex128, ldz: i32, work: ^complex128, rwork: ^f64) -> i32 ---
	LAPACKE_chbevd_work :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, kd: i32, ab: ^complex64, ldab: i32, w: ^f32, z: ^complex64, ldz: i32, work: ^complex64, lwork: i32, rwork: ^f32, lrwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_zhbevd_work :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, kd: i32, ab: ^complex128, ldab: i32, w: ^f64, z: ^complex128, ldz: i32, work: ^complex128, lwork: i32, rwork: ^f64, lrwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_chbevx_work :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, uplo: c.char, n: i32, kd: i32, ab: ^complex64, ldab: i32, q: ^complex64, ldq: i32, vl: f32, vu: f32, il: i32, iu: i32, abstol: f32, m: ^i32, w: ^f32, z: ^complex64, ldz: i32, work: ^complex64, rwork: ^f32, iwork: ^i32, ifail: ^i32) -> i32 ---
	LAPACKE_zhbevx_work :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, uplo: c.char, n: i32, kd: i32, ab: ^complex128, ldab: i32, q: ^complex128, ldq: i32, vl: f64, vu: f64, il: i32, iu: i32, abstol: f64, m: ^i32, w: ^f64, z: ^complex128, ldz: i32, work: ^complex128, rwork: ^f64, iwork: ^i32, ifail: ^i32) -> i32 ---
	LAPACKE_chbgst_work :: proc(matrix_layout: c.int, vect: c.char, uplo: c.char, n: i32, ka: i32, kb: i32, ab: ^complex64, ldab: i32, bb: ^complex64, ldbb: i32, x: ^complex64, ldx: i32, work: ^complex64, rwork: ^f32) -> i32 ---
	LAPACKE_zhbgst_work :: proc(matrix_layout: c.int, vect: c.char, uplo: c.char, n: i32, ka: i32, kb: i32, ab: ^complex128, ldab: i32, bb: ^complex128, ldbb: i32, x: ^complex128, ldx: i32, work: ^complex128, rwork: ^f64) -> i32 ---
	LAPACKE_chbgv_work :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, ka: i32, kb: i32, ab: ^complex64, ldab: i32, bb: ^complex64, ldbb: i32, w: ^f32, z: ^complex64, ldz: i32, work: ^complex64, rwork: ^f32) -> i32 ---
	LAPACKE_zhbgv_work :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, ka: i32, kb: i32, ab: ^complex128, ldab: i32, bb: ^complex128, ldbb: i32, w: ^f64, z: ^complex128, ldz: i32, work: ^complex128, rwork: ^f64) -> i32 ---
	LAPACKE_chbgvd_work :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, ka: i32, kb: i32, ab: ^complex64, ldab: i32, bb: ^complex64, ldbb: i32, w: ^f32, z: ^complex64, ldz: i32, work: ^complex64, lwork: i32, rwork: ^f32, lrwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_zhbgvd_work :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, ka: i32, kb: i32, ab: ^complex128, ldab: i32, bb: ^complex128, ldbb: i32, w: ^f64, z: ^complex128, ldz: i32, work: ^complex128, lwork: i32, rwork: ^f64, lrwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_chbgvx_work :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, uplo: c.char, n: i32, ka: i32, kb: i32, ab: ^complex64, ldab: i32, bb: ^complex64, ldbb: i32, q: ^complex64, ldq: i32, vl: f32, vu: f32, il: i32, iu: i32, abstol: f32, m: ^i32, w: ^f32, z: ^complex64, ldz: i32, work: ^complex64, rwork: ^f32, iwork: ^i32, ifail: ^i32) -> i32 ---
	LAPACKE_zhbgvx_work :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, uplo: c.char, n: i32, ka: i32, kb: i32, ab: ^complex128, ldab: i32, bb: ^complex128, ldbb: i32, q: ^complex128, ldq: i32, vl: f64, vu: f64, il: i32, iu: i32, abstol: f64, m: ^i32, w: ^f64, z: ^complex128, ldz: i32, work: ^complex128, rwork: ^f64, iwork: ^i32, ifail: ^i32) -> i32 ---
	LAPACKE_chbtrd_work :: proc(matrix_layout: c.int, vect: c.char, uplo: c.char, n: i32, kd: i32, ab: ^complex64, ldab: i32, d: ^f32, e: ^f32, q: ^complex64, ldq: i32, work: ^complex64) -> i32 ---
	LAPACKE_zhbtrd_work :: proc(matrix_layout: c.int, vect: c.char, uplo: c.char, n: i32, kd: i32, ab: ^complex128, ldab: i32, d: ^f64, e: ^f64, q: ^complex128, ldq: i32, work: ^complex128) -> i32 ---
	LAPACKE_checon_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32, ipiv: ^i32, anorm: f32, rcond: ^f32, work: ^complex64) -> i32 ---
	LAPACKE_zhecon_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32, ipiv: ^i32, anorm: f64, rcond: ^f64, work: ^complex128) -> i32 ---
	LAPACKE_cheequb_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32, s: ^f32, scond: ^f32, amax: ^f32, work: ^complex64) -> i32 ---
	LAPACKE_zheequb_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32, s: ^f64, scond: ^f64, amax: ^f64, work: ^complex128) -> i32 ---
	LAPACKE_cheev_work :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, a: ^complex64, lda: i32, w: ^f32, work: ^complex64, lwork: i32, rwork: ^f32) -> i32 ---
	LAPACKE_zheev_work :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, a: ^complex128, lda: i32, w: ^f64, work: ^complex128, lwork: i32, rwork: ^f64) -> i32 ---
	LAPACKE_cheevd_work :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, a: ^complex64, lda: i32, w: ^f32, work: ^complex64, lwork: i32, rwork: ^f32, lrwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_zheevd_work :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, a: ^complex128, lda: i32, w: ^f64, work: ^complex128, lwork: i32, rwork: ^f64, lrwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_cheevr_work :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, uplo: c.char, n: i32, a: ^complex64, lda: i32, vl: f32, vu: f32, il: i32, iu: i32, abstol: f32, m: ^i32, w: ^f32, z: ^complex64, ldz: i32, isuppz: ^i32, work: ^complex64, lwork: i32, rwork: ^f32, lrwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_zheevr_work :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, uplo: c.char, n: i32, a: ^complex128, lda: i32, vl: f64, vu: f64, il: i32, iu: i32, abstol: f64, m: ^i32, w: ^f64, z: ^complex128, ldz: i32, isuppz: ^i32, work: ^complex128, lwork: i32, rwork: ^f64, lrwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_cheevx_work :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, uplo: c.char, n: i32, a: ^complex64, lda: i32, vl: f32, vu: f32, il: i32, iu: i32, abstol: f32, m: ^i32, w: ^f32, z: ^complex64, ldz: i32, work: ^complex64, lwork: i32, rwork: ^f32, iwork: ^i32, ifail: ^i32) -> i32 ---
	LAPACKE_zheevx_work :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, uplo: c.char, n: i32, a: ^complex128, lda: i32, vl: f64, vu: f64, il: i32, iu: i32, abstol: f64, m: ^i32, w: ^f64, z: ^complex128, ldz: i32, work: ^complex128, lwork: i32, rwork: ^f64, iwork: ^i32, ifail: ^i32) -> i32 ---
	LAPACKE_chegst_work :: proc(matrix_layout: c.int, itype: i32, uplo: c.char, n: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_zhegst_work :: proc(matrix_layout: c.int, itype: i32, uplo: c.char, n: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_chegv_work :: proc(matrix_layout: c.int, itype: i32, jobz: c.char, uplo: c.char, n: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32, w: ^f32, work: ^complex64, lwork: i32, rwork: ^f32) -> i32 ---
	LAPACKE_zhegv_work :: proc(matrix_layout: c.int, itype: i32, jobz: c.char, uplo: c.char, n: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, w: ^f64, work: ^complex128, lwork: i32, rwork: ^f64) -> i32 ---
	LAPACKE_chegvd_work :: proc(matrix_layout: c.int, itype: i32, jobz: c.char, uplo: c.char, n: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32, w: ^f32, work: ^complex64, lwork: i32, rwork: ^f32, lrwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_zhegvd_work :: proc(matrix_layout: c.int, itype: i32, jobz: c.char, uplo: c.char, n: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, w: ^f64, work: ^complex128, lwork: i32, rwork: ^f64, lrwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_chegvx_work :: proc(matrix_layout: c.int, itype: i32, jobz: c.char, range: c.char, uplo: c.char, n: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32, vl: f32, vu: f32, il: i32, iu: i32, abstol: f32, m: ^i32, w: ^f32, z: ^complex64, ldz: i32, work: ^complex64, lwork: i32, rwork: ^f32, iwork: ^i32, ifail: ^i32) -> i32 ---
	LAPACKE_zhegvx_work :: proc(matrix_layout: c.int, itype: i32, jobz: c.char, range: c.char, uplo: c.char, n: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, vl: f64, vu: f64, il: i32, iu: i32, abstol: f64, m: ^i32, w: ^f64, z: ^complex128, ldz: i32, work: ^complex128, lwork: i32, rwork: ^f64, iwork: ^i32, ifail: ^i32) -> i32 ---
	LAPACKE_cherfs_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, af: ^complex64, ldaf: i32, ipiv: ^i32, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, ferr: ^f32, berr: ^f32, work: ^complex64, rwork: ^f32) -> i32 ---
	LAPACKE_zherfs_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, af: ^complex128, ldaf: i32, ipiv: ^i32, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, ferr: ^f64, berr: ^f64, work: ^complex128, rwork: ^f64) -> i32 ---
	LAPACKE_cherfsx_work :: proc(matrix_layout: c.int, uplo: c.char, equed: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, af: ^complex64, ldaf: i32, ipiv: ^i32, s: ^f32, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, rcond: ^f32, berr: ^f32, n_err_bnds: i32, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: i32, params: ^f32, work: ^complex64, rwork: ^f32) -> i32 ---
	LAPACKE_zherfsx_work :: proc(matrix_layout: c.int, uplo: c.char, equed: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, af: ^complex128, ldaf: i32, ipiv: ^i32, s: ^f64, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, rcond: ^f64, berr: ^f64, n_err_bnds: i32, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: i32, params: ^f64, work: ^complex128, rwork: ^f64) -> i32 ---
	LAPACKE_chesv_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, ipiv: ^i32, b: ^complex64, ldb: i32, work: ^complex64, lwork: i32) -> i32 ---
	LAPACKE_zhesv_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, ipiv: ^i32, b: ^complex128, ldb: i32, work: ^complex128, lwork: i32) -> i32 ---
	LAPACKE_chesvx_work :: proc(matrix_layout: c.int, fact: c.char, uplo: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, af: ^complex64, ldaf: i32, ipiv: ^i32, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, rcond: ^f32, ferr: ^f32, berr: ^f32, work: ^complex64, lwork: i32, rwork: ^f32) -> i32 ---
	LAPACKE_zhesvx_work :: proc(matrix_layout: c.int, fact: c.char, uplo: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, af: ^complex128, ldaf: i32, ipiv: ^i32, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, rcond: ^f64, ferr: ^f64, berr: ^f64, work: ^complex128, lwork: i32, rwork: ^f64) -> i32 ---
	LAPACKE_chesvxx_work :: proc(matrix_layout: c.int, fact: c.char, uplo: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, af: ^complex64, ldaf: i32, ipiv: ^i32, equed: cstring, s: ^f32, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, rcond: ^f32, rpvgrw: ^f32, berr: ^f32, n_err_bnds: i32, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: i32, params: ^f32, work: ^complex64, rwork: ^f32) -> i32 ---
	LAPACKE_zhesvxx_work :: proc(matrix_layout: c.int, fact: c.char, uplo: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, af: ^complex128, ldaf: i32, ipiv: ^i32, equed: cstring, s: ^f64, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, rcond: ^f64, rpvgrw: ^f64, berr: ^f64, n_err_bnds: i32, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: i32, params: ^f64, work: ^complex128, rwork: ^f64) -> i32 ---
	LAPACKE_chetrd_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32, d: ^f32, e: ^f32, tau: ^complex64, work: ^complex64, lwork: i32) -> i32 ---
	LAPACKE_zhetrd_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32, d: ^f64, e: ^f64, tau: ^complex128, work: ^complex128, lwork: i32) -> i32 ---
	LAPACKE_chetrf_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32, ipiv: ^i32, work: ^complex64, lwork: i32) -> i32 ---
	LAPACKE_zhetrf_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32, ipiv: ^i32, work: ^complex128, lwork: i32) -> i32 ---
	LAPACKE_chetri_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32, ipiv: ^i32, work: ^complex64) -> i32 ---
	LAPACKE_zhetri_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32, ipiv: ^i32, work: ^complex128) -> i32 ---
	LAPACKE_chetrs_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, ipiv: ^i32, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_zhetrs_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, ipiv: ^i32, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_chfrk_work :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, trans: c.char, n: i32, k: i32, alpha: f32, a: ^complex64, lda: i32, beta: f32, _c: ^complex64) -> i32 ---
	LAPACKE_zhfrk_work :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, trans: c.char, n: i32, k: i32, alpha: f64, a: ^complex128, lda: i32, beta: f64, _c: ^complex128) -> i32 ---
	LAPACKE_shgeqz_work :: proc(matrix_layout: c.int, job: c.char, compq: c.char, compz: c.char, n: i32, ilo: i32, ihi: i32, h: ^f32, ldh: i32, t: ^f32, ldt: i32, alphar: ^f32, alphai: ^f32, beta: ^f32, q: ^f32, ldq: i32, z: ^f32, ldz: i32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_dhgeqz_work :: proc(matrix_layout: c.int, job: c.char, compq: c.char, compz: c.char, n: i32, ilo: i32, ihi: i32, h: ^f64, ldh: i32, t: ^f64, ldt: i32, alphar: ^f64, alphai: ^f64, beta: ^f64, q: ^f64, ldq: i32, z: ^f64, ldz: i32, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_chgeqz_work :: proc(matrix_layout: c.int, job: c.char, compq: c.char, compz: c.char, n: i32, ilo: i32, ihi: i32, h: ^complex64, ldh: i32, t: ^complex64, ldt: i32, alpha: ^complex64, beta: ^complex64, q: ^complex64, ldq: i32, z: ^complex64, ldz: i32, work: ^complex64, lwork: i32, rwork: ^f32) -> i32 ---
	LAPACKE_zhgeqz_work :: proc(matrix_layout: c.int, job: c.char, compq: c.char, compz: c.char, n: i32, ilo: i32, ihi: i32, h: ^complex128, ldh: i32, t: ^complex128, ldt: i32, alpha: ^complex128, beta: ^complex128, q: ^complex128, ldq: i32, z: ^complex128, ldz: i32, work: ^complex128, lwork: i32, rwork: ^f64) -> i32 ---
	LAPACKE_chpcon_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^complex64, ipiv: ^i32, anorm: f32, rcond: ^f32, work: ^complex64) -> i32 ---
	LAPACKE_zhpcon_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^complex128, ipiv: ^i32, anorm: f64, rcond: ^f64, work: ^complex128) -> i32 ---
	LAPACKE_chpev_work :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, ap: ^complex64, w: ^f32, z: ^complex64, ldz: i32, work: ^complex64, rwork: ^f32) -> i32 ---
	LAPACKE_zhpev_work :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, ap: ^complex128, w: ^f64, z: ^complex128, ldz: i32, work: ^complex128, rwork: ^f64) -> i32 ---
	LAPACKE_chpevd_work :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, ap: ^complex64, w: ^f32, z: ^complex64, ldz: i32, work: ^complex64, lwork: i32, rwork: ^f32, lrwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_zhpevd_work :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, ap: ^complex128, w: ^f64, z: ^complex128, ldz: i32, work: ^complex128, lwork: i32, rwork: ^f64, lrwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_chpevx_work :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, uplo: c.char, n: i32, ap: ^complex64, vl: f32, vu: f32, il: i32, iu: i32, abstol: f32, m: ^i32, w: ^f32, z: ^complex64, ldz: i32, work: ^complex64, rwork: ^f32, iwork: ^i32, ifail: ^i32) -> i32 ---
	LAPACKE_zhpevx_work :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, uplo: c.char, n: i32, ap: ^complex128, vl: f64, vu: f64, il: i32, iu: i32, abstol: f64, m: ^i32, w: ^f64, z: ^complex128, ldz: i32, work: ^complex128, rwork: ^f64, iwork: ^i32, ifail: ^i32) -> i32 ---
	LAPACKE_chpgst_work :: proc(matrix_layout: c.int, itype: i32, uplo: c.char, n: i32, ap: ^complex64, bp: ^complex64) -> i32 ---
	LAPACKE_zhpgst_work :: proc(matrix_layout: c.int, itype: i32, uplo: c.char, n: i32, ap: ^complex128, bp: ^complex128) -> i32 ---
	LAPACKE_chpgv_work :: proc(matrix_layout: c.int, itype: i32, jobz: c.char, uplo: c.char, n: i32, ap: ^complex64, bp: ^complex64, w: ^f32, z: ^complex64, ldz: i32, work: ^complex64, rwork: ^f32) -> i32 ---
	LAPACKE_zhpgv_work :: proc(matrix_layout: c.int, itype: i32, jobz: c.char, uplo: c.char, n: i32, ap: ^complex128, bp: ^complex128, w: ^f64, z: ^complex128, ldz: i32, work: ^complex128, rwork: ^f64) -> i32 ---
	LAPACKE_chpgvd_work :: proc(matrix_layout: c.int, itype: i32, jobz: c.char, uplo: c.char, n: i32, ap: ^complex64, bp: ^complex64, w: ^f32, z: ^complex64, ldz: i32, work: ^complex64, lwork: i32, rwork: ^f32, lrwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_zhpgvd_work :: proc(matrix_layout: c.int, itype: i32, jobz: c.char, uplo: c.char, n: i32, ap: ^complex128, bp: ^complex128, w: ^f64, z: ^complex128, ldz: i32, work: ^complex128, lwork: i32, rwork: ^f64, lrwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_chpgvx_work :: proc(matrix_layout: c.int, itype: i32, jobz: c.char, range: c.char, uplo: c.char, n: i32, ap: ^complex64, bp: ^complex64, vl: f32, vu: f32, il: i32, iu: i32, abstol: f32, m: ^i32, w: ^f32, z: ^complex64, ldz: i32, work: ^complex64, rwork: ^f32, iwork: ^i32, ifail: ^i32) -> i32 ---
	LAPACKE_zhpgvx_work :: proc(matrix_layout: c.int, itype: i32, jobz: c.char, range: c.char, uplo: c.char, n: i32, ap: ^complex128, bp: ^complex128, vl: f64, vu: f64, il: i32, iu: i32, abstol: f64, m: ^i32, w: ^f64, z: ^complex128, ldz: i32, work: ^complex128, rwork: ^f64, iwork: ^i32, ifail: ^i32) -> i32 ---
	LAPACKE_chprfs_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, ap: ^complex64, afp: ^complex64, ipiv: ^i32, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, ferr: ^f32, berr: ^f32, work: ^complex64, rwork: ^f32) -> i32 ---
	LAPACKE_zhprfs_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, ap: ^complex128, afp: ^complex128, ipiv: ^i32, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, ferr: ^f64, berr: ^f64, work: ^complex128, rwork: ^f64) -> i32 ---
	LAPACKE_chpsv_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, ap: ^complex64, ipiv: ^i32, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_zhpsv_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, ap: ^complex128, ipiv: ^i32, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_chpsvx_work :: proc(matrix_layout: c.int, fact: c.char, uplo: c.char, n: i32, nrhs: i32, ap: ^complex64, afp: ^complex64, ipiv: ^i32, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, rcond: ^f32, ferr: ^f32, berr: ^f32, work: ^complex64, rwork: ^f32) -> i32 ---
	LAPACKE_zhpsvx_work :: proc(matrix_layout: c.int, fact: c.char, uplo: c.char, n: i32, nrhs: i32, ap: ^complex128, afp: ^complex128, ipiv: ^i32, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, rcond: ^f64, ferr: ^f64, berr: ^f64, work: ^complex128, rwork: ^f64) -> i32 ---
	LAPACKE_chptrd_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^complex64, d: ^f32, e: ^f32, tau: ^complex64) -> i32 ---
	LAPACKE_zhptrd_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^complex128, d: ^f64, e: ^f64, tau: ^complex128) -> i32 ---
	LAPACKE_chptrf_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^complex64, ipiv: ^i32) -> i32 ---
	LAPACKE_zhptrf_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^complex128, ipiv: ^i32) -> i32 ---
	LAPACKE_chptri_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^complex64, ipiv: ^i32, work: ^complex64) -> i32 ---
	LAPACKE_zhptri_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^complex128, ipiv: ^i32, work: ^complex128) -> i32 ---
	LAPACKE_chptrs_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, ap: ^complex64, ipiv: ^i32, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_zhptrs_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, ap: ^complex128, ipiv: ^i32, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_shsein_work :: proc(matrix_layout: c.int, job: c.char, eigsrc: c.char, initv: c.char, select: ^i32, n: i32, h: ^f32, ldh: i32, wr: ^f32, wi: ^f32, vl: ^f32, ldvl: i32, vr: ^f32, ldvr: i32, mm: i32, m: ^i32, work: ^f32, ifaill: ^i32, ifailr: ^i32) -> i32 ---
	LAPACKE_dhsein_work :: proc(matrix_layout: c.int, job: c.char, eigsrc: c.char, initv: c.char, select: ^i32, n: i32, h: ^f64, ldh: i32, wr: ^f64, wi: ^f64, vl: ^f64, ldvl: i32, vr: ^f64, ldvr: i32, mm: i32, m: ^i32, work: ^f64, ifaill: ^i32, ifailr: ^i32) -> i32 ---
	LAPACKE_chsein_work :: proc(matrix_layout: c.int, job: c.char, eigsrc: c.char, initv: c.char, select: ^i32, n: i32, h: ^complex64, ldh: i32, w: ^complex64, vl: ^complex64, ldvl: i32, vr: ^complex64, ldvr: i32, mm: i32, m: ^i32, work: ^complex64, rwork: ^f32, ifaill: ^i32, ifailr: ^i32) -> i32 ---
	LAPACKE_zhsein_work :: proc(matrix_layout: c.int, job: c.char, eigsrc: c.char, initv: c.char, select: ^i32, n: i32, h: ^complex128, ldh: i32, w: ^complex128, vl: ^complex128, ldvl: i32, vr: ^complex128, ldvr: i32, mm: i32, m: ^i32, work: ^complex128, rwork: ^f64, ifaill: ^i32, ifailr: ^i32) -> i32 ---
	LAPACKE_shseqr_work :: proc(matrix_layout: c.int, job: c.char, compz: c.char, n: i32, ilo: i32, ihi: i32, h: ^f32, ldh: i32, wr: ^f32, wi: ^f32, z: ^f32, ldz: i32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_dhseqr_work :: proc(matrix_layout: c.int, job: c.char, compz: c.char, n: i32, ilo: i32, ihi: i32, h: ^f64, ldh: i32, wr: ^f64, wi: ^f64, z: ^f64, ldz: i32, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_chseqr_work :: proc(matrix_layout: c.int, job: c.char, compz: c.char, n: i32, ilo: i32, ihi: i32, h: ^complex64, ldh: i32, w: ^complex64, z: ^complex64, ldz: i32, work: ^complex64, lwork: i32) -> i32 ---
	LAPACKE_zhseqr_work :: proc(matrix_layout: c.int, job: c.char, compz: c.char, n: i32, ilo: i32, ihi: i32, h: ^complex128, ldh: i32, w: ^complex128, z: ^complex128, ldz: i32, work: ^complex128, lwork: i32) -> i32 ---
	LAPACKE_clacgv_work :: proc(n: i32, x: ^complex64, incx: i32) -> i32 ---
	LAPACKE_zlacgv_work :: proc(n: i32, x: ^complex128, incx: i32) -> i32 ---
	LAPACKE_slacn2_work :: proc(n: i32, v: ^f32, x: ^f32, isgn: ^i32, est: ^f32, kase: ^i32, isave: ^i32) -> i32 ---
	LAPACKE_dlacn2_work :: proc(n: i32, v: ^f64, x: ^f64, isgn: ^i32, est: ^f64, kase: ^i32, isave: ^i32) -> i32 ---
	LAPACKE_clacn2_work :: proc(n: i32, v: ^complex64, x: ^complex64, est: ^f32, kase: ^i32, isave: ^i32) -> i32 ---
	LAPACKE_zlacn2_work :: proc(n: i32, v: ^complex128, x: ^complex128, est: ^f64, kase: ^i32, isave: ^i32) -> i32 ---
	LAPACKE_slacpy_work :: proc(matrix_layout: c.int, uplo: c.char, m: i32, n: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32) -> i32 ---
	LAPACKE_dlacpy_work :: proc(matrix_layout: c.int, uplo: c.char, m: i32, n: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32) -> i32 ---
	LAPACKE_clacpy_work :: proc(matrix_layout: c.int, uplo: c.char, m: i32, n: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_zlacpy_work :: proc(matrix_layout: c.int, uplo: c.char, m: i32, n: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_clacp2_work :: proc(matrix_layout: c.int, uplo: c.char, m: i32, n: i32, a: ^f32, lda: i32, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_zlacp2_work :: proc(matrix_layout: c.int, uplo: c.char, m: i32, n: i32, a: ^f64, lda: i32, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_zlag2c_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex128, lda: i32, sa: ^complex64, ldsa: i32) -> i32 ---
	LAPACKE_slag2d_work :: proc(matrix_layout: c.int, m: i32, n: i32, sa: ^f32, ldsa: i32, a: ^f64, lda: i32) -> i32 ---
	LAPACKE_dlag2s_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f64, lda: i32, sa: ^f32, ldsa: i32) -> i32 ---
	LAPACKE_clag2z_work :: proc(matrix_layout: c.int, m: i32, n: i32, sa: ^complex64, ldsa: i32, a: ^complex128, lda: i32) -> i32 ---
	LAPACKE_slagge_work :: proc(matrix_layout: c.int, m: i32, n: i32, kl: i32, ku: i32, d: ^f32, a: ^f32, lda: i32, iseed: ^i32, work: ^f32) -> i32 ---
	LAPACKE_dlagge_work :: proc(matrix_layout: c.int, m: i32, n: i32, kl: i32, ku: i32, d: ^f64, a: ^f64, lda: i32, iseed: ^i32, work: ^f64) -> i32 ---
	LAPACKE_clagge_work :: proc(matrix_layout: c.int, m: i32, n: i32, kl: i32, ku: i32, d: ^f32, a: ^complex64, lda: i32, iseed: ^i32, work: ^complex64) -> i32 ---
	LAPACKE_zlagge_work :: proc(matrix_layout: c.int, m: i32, n: i32, kl: i32, ku: i32, d: ^f64, a: ^complex128, lda: i32, iseed: ^i32, work: ^complex128) -> i32 ---
	LAPACKE_claghe_work :: proc(matrix_layout: c.int, n: i32, k: i32, d: ^f32, a: ^complex64, lda: i32, iseed: ^i32, work: ^complex64) -> i32 ---
	LAPACKE_zlaghe_work :: proc(matrix_layout: c.int, n: i32, k: i32, d: ^f64, a: ^complex128, lda: i32, iseed: ^i32, work: ^complex128) -> i32 ---
	LAPACKE_slagsy_work :: proc(matrix_layout: c.int, n: i32, k: i32, d: ^f32, a: ^f32, lda: i32, iseed: ^i32, work: ^f32) -> i32 ---
	LAPACKE_dlagsy_work :: proc(matrix_layout: c.int, n: i32, k: i32, d: ^f64, a: ^f64, lda: i32, iseed: ^i32, work: ^f64) -> i32 ---
	LAPACKE_clagsy_work :: proc(matrix_layout: c.int, n: i32, k: i32, d: ^f32, a: ^complex64, lda: i32, iseed: ^i32, work: ^complex64) -> i32 ---
	LAPACKE_zlagsy_work :: proc(matrix_layout: c.int, n: i32, k: i32, d: ^f64, a: ^complex128, lda: i32, iseed: ^i32, work: ^complex128) -> i32 ---
	LAPACKE_slapmr_work :: proc(matrix_layout: c.int, forwrd: i32, m: i32, n: i32, x: ^f32, ldx: i32, k: ^i32) -> i32 ---
	LAPACKE_dlapmr_work :: proc(matrix_layout: c.int, forwrd: i32, m: i32, n: i32, x: ^f64, ldx: i32, k: ^i32) -> i32 ---
	LAPACKE_clapmr_work :: proc(matrix_layout: c.int, forwrd: i32, m: i32, n: i32, x: ^complex64, ldx: i32, k: ^i32) -> i32 ---
	LAPACKE_zlapmr_work :: proc(matrix_layout: c.int, forwrd: i32, m: i32, n: i32, x: ^complex128, ldx: i32, k: ^i32) -> i32 ---
	LAPACKE_slapmt_work :: proc(matrix_layout: c.int, forwrd: i32, m: i32, n: i32, x: ^f32, ldx: i32, k: ^i32) -> i32 ---
	LAPACKE_dlapmt_work :: proc(matrix_layout: c.int, forwrd: i32, m: i32, n: i32, x: ^f64, ldx: i32, k: ^i32) -> i32 ---
	LAPACKE_clapmt_work :: proc(matrix_layout: c.int, forwrd: i32, m: i32, n: i32, x: ^complex64, ldx: i32, k: ^i32) -> i32 ---
	LAPACKE_zlapmt_work :: proc(matrix_layout: c.int, forwrd: i32, m: i32, n: i32, x: ^complex128, ldx: i32, k: ^i32) -> i32 ---
	LAPACKE_slartgp_work :: proc(f: f32, g: f32, cs: ^f32, sn: ^f32, r: ^f32) -> i32 ---
	LAPACKE_dlartgp_work :: proc(f: f64, g: f64, cs: ^f64, sn: ^f64, r: ^f64) -> i32 ---
	LAPACKE_slartgs_work :: proc(x: f32, y: f32, sigma: f32, cs: ^f32, sn: ^f32) -> i32 ---
	LAPACKE_dlartgs_work :: proc(x: f64, y: f64, sigma: f64, cs: ^f64, sn: ^f64) -> i32 ---
	LAPACKE_slapy2_work :: proc(x: f32, y: f32) -> f32 ---
	LAPACKE_dlapy2_work :: proc(x: f64, y: f64) -> f64 ---
	LAPACKE_slapy3_work :: proc(x: f32, y: f32, z: f32) -> f32 ---
	LAPACKE_dlapy3_work :: proc(x: f64, y: f64, z: f64) -> f64 ---
	LAPACKE_slamch_work :: proc(cmach: c.char) -> f32 ---
	LAPACKE_dlamch_work :: proc(cmach: c.char) -> f64 ---
	LAPACKE_slangb_work :: proc(matrix_layout: c.int, norm: c.char, n: i32, kl: i32, ku: i32, ab: ^f32, ldab: i32, work: ^f32) -> f32 ---
	LAPACKE_dlangb_work :: proc(matrix_layout: c.int, norm: c.char, n: i32, kl: i32, ku: i32, ab: ^f64, ldab: i32, work: ^f64) -> f64 ---
	LAPACKE_clangb_work :: proc(matrix_layout: c.int, norm: c.char, n: i32, kl: i32, ku: i32, ab: ^complex64, ldab: i32, work: ^f32) -> f32 ---
	LAPACKE_zlangb_work :: proc(matrix_layout: c.int, norm: c.char, n: i32, kl: i32, ku: i32, ab: ^complex128, ldab: i32, work: ^f64) -> f64 ---
	LAPACKE_slange_work :: proc(matrix_layout: c.int, norm: c.char, m: i32, n: i32, a: ^f32, lda: i32, work: ^f32) -> f32 ---
	LAPACKE_dlange_work :: proc(matrix_layout: c.int, norm: c.char, m: i32, n: i32, a: ^f64, lda: i32, work: ^f64) -> f64 ---
	LAPACKE_clange_work :: proc(matrix_layout: c.int, norm: c.char, m: i32, n: i32, a: ^complex64, lda: i32, work: ^f32) -> f32 ---
	LAPACKE_zlange_work :: proc(matrix_layout: c.int, norm: c.char, m: i32, n: i32, a: ^complex128, lda: i32, work: ^f64) -> f64 ---
	LAPACKE_clanhe_work :: proc(matrix_layout: c.int, norm: c.char, uplo: c.char, n: i32, a: ^complex64, lda: i32, work: ^f32) -> f32 ---
	LAPACKE_zlanhe_work :: proc(matrix_layout: c.int, norm: c.char, uplo: c.char, n: i32, a: ^complex128, lda: i32, work: ^f64) -> f64 ---
	LAPACKE_clacrm_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex64, lda: i32, b: ^f32, ldb: i32, _c: ^complex64, ldc: i32, work: ^f32) -> i32 ---
	LAPACKE_zlacrm_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex128, lda: i32, b: ^f64, ldb: i32, _c: ^complex128, ldc: i32, work: ^f64) -> i32 ---
	LAPACKE_clarcm_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f32, lda: i32, b: ^complex64, ldb: i32, _c: ^complex64, ldc: i32, work: ^f32) -> i32 ---
	LAPACKE_zlarcm_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f64, lda: i32, b: ^complex128, ldb: i32, _c: ^complex128, ldc: i32, work: ^f64) -> i32 ---
	LAPACKE_slansy_work :: proc(matrix_layout: c.int, norm: c.char, uplo: c.char, n: i32, a: ^f32, lda: i32, work: ^f32) -> f32 ---
	LAPACKE_dlansy_work :: proc(matrix_layout: c.int, norm: c.char, uplo: c.char, n: i32, a: ^f64, lda: i32, work: ^f64) -> f64 ---
	LAPACKE_clansy_work :: proc(matrix_layout: c.int, norm: c.char, uplo: c.char, n: i32, a: ^complex64, lda: i32, work: ^f32) -> f32 ---
	LAPACKE_zlansy_work :: proc(matrix_layout: c.int, norm: c.char, uplo: c.char, n: i32, a: ^complex128, lda: i32, work: ^f64) -> f64 ---
	LAPACKE_slantr_work :: proc(matrix_layout: c.int, norm: c.char, uplo: c.char, diag: c.char, m: i32, n: i32, a: ^f32, lda: i32, work: ^f32) -> f32 ---
	LAPACKE_dlantr_work :: proc(matrix_layout: c.int, norm: c.char, uplo: c.char, diag: c.char, m: i32, n: i32, a: ^f64, lda: i32, work: ^f64) -> f64 ---
	LAPACKE_clantr_work :: proc(matrix_layout: c.int, norm: c.char, uplo: c.char, diag: c.char, m: i32, n: i32, a: ^complex64, lda: i32, work: ^f32) -> f32 ---
	LAPACKE_zlantr_work :: proc(matrix_layout: c.int, norm: c.char, uplo: c.char, diag: c.char, m: i32, n: i32, a: ^complex128, lda: i32, work: ^f64) -> f64 ---
	LAPACKE_slarfb_work :: proc(matrix_layout: c.int, side: c.char, trans: c.char, direct: c.char, storev: c.char, m: i32, n: i32, k: i32, v: ^f32, ldv: i32, t: ^f32, ldt: i32, _c: ^f32, ldc: i32, work: ^f32, ldwork: i32) -> i32 ---
	LAPACKE_dlarfb_work :: proc(matrix_layout: c.int, side: c.char, trans: c.char, direct: c.char, storev: c.char, m: i32, n: i32, k: i32, v: ^f64, ldv: i32, t: ^f64, ldt: i32, _c: ^f64, ldc: i32, work: ^f64, ldwork: i32) -> i32 ---
	LAPACKE_clarfb_work :: proc(matrix_layout: c.int, side: c.char, trans: c.char, direct: c.char, storev: c.char, m: i32, n: i32, k: i32, v: ^complex64, ldv: i32, t: ^complex64, ldt: i32, _c: ^complex64, ldc: i32, work: ^complex64, ldwork: i32) -> i32 ---
	LAPACKE_zlarfb_work :: proc(matrix_layout: c.int, side: c.char, trans: c.char, direct: c.char, storev: c.char, m: i32, n: i32, k: i32, v: ^complex128, ldv: i32, t: ^complex128, ldt: i32, _c: ^complex128, ldc: i32, work: ^complex128, ldwork: i32) -> i32 ---
	LAPACKE_slarfg_work :: proc(n: i32, alpha: ^f32, x: ^f32, incx: i32, tau: ^f32) -> i32 ---
	LAPACKE_dlarfg_work :: proc(n: i32, alpha: ^f64, x: ^f64, incx: i32, tau: ^f64) -> i32 ---
	LAPACKE_clarfg_work :: proc(n: i32, alpha: ^complex64, x: ^complex64, incx: i32, tau: ^complex64) -> i32 ---
	LAPACKE_zlarfg_work :: proc(n: i32, alpha: ^complex128, x: ^complex128, incx: i32, tau: ^complex128) -> i32 ---
	LAPACKE_slarft_work :: proc(matrix_layout: c.int, direct: c.char, storev: c.char, n: i32, k: i32, v: ^f32, ldv: i32, tau: ^f32, t: ^f32, ldt: i32) -> i32 ---
	LAPACKE_dlarft_work :: proc(matrix_layout: c.int, direct: c.char, storev: c.char, n: i32, k: i32, v: ^f64, ldv: i32, tau: ^f64, t: ^f64, ldt: i32) -> i32 ---
	LAPACKE_clarft_work :: proc(matrix_layout: c.int, direct: c.char, storev: c.char, n: i32, k: i32, v: ^complex64, ldv: i32, tau: ^complex64, t: ^complex64, ldt: i32) -> i32 ---
	LAPACKE_zlarft_work :: proc(matrix_layout: c.int, direct: c.char, storev: c.char, n: i32, k: i32, v: ^complex128, ldv: i32, tau: ^complex128, t: ^complex128, ldt: i32) -> i32 ---
	LAPACKE_slarfx_work :: proc(matrix_layout: c.int, side: c.char, m: i32, n: i32, v: ^f32, tau: f32, _c: ^f32, ldc: i32, work: ^f32) -> i32 ---
	LAPACKE_dlarfx_work :: proc(matrix_layout: c.int, side: c.char, m: i32, n: i32, v: ^f64, tau: f64, _c: ^f64, ldc: i32, work: ^f64) -> i32 ---
	LAPACKE_clarfx_work :: proc(matrix_layout: c.int, side: c.char, m: i32, n: i32, v: ^complex64, tau: complex64, _c: ^complex64, ldc: i32, work: ^complex64) -> i32 ---
	LAPACKE_zlarfx_work :: proc(matrix_layout: c.int, side: c.char, m: i32, n: i32, v: ^complex128, tau: complex128, _c: ^complex128, ldc: i32, work: ^complex128) -> i32 ---
	LAPACKE_slarnv_work :: proc(idist: i32, iseed: ^i32, n: i32, x: ^f32) -> i32 ---
	LAPACKE_dlarnv_work :: proc(idist: i32, iseed: ^i32, n: i32, x: ^f64) -> i32 ---
	LAPACKE_clarnv_work :: proc(idist: i32, iseed: ^i32, n: i32, x: ^complex64) -> i32 ---
	LAPACKE_zlarnv_work :: proc(idist: i32, iseed: ^i32, n: i32, x: ^complex128) -> i32 ---
	LAPACKE_slascl_work :: proc(matrix_layout: c.int, type: c.char, kl: i32, ku: i32, cfrom: f32, cto: f32, m: i32, n: i32, a: ^f32, lda: i32) -> i32 ---
	LAPACKE_dlascl_work :: proc(matrix_layout: c.int, type: c.char, kl: i32, ku: i32, cfrom: f64, cto: f64, m: i32, n: i32, a: ^f64, lda: i32) -> i32 ---
	LAPACKE_clascl_work :: proc(matrix_layout: c.int, type: c.char, kl: i32, ku: i32, cfrom: f32, cto: f32, m: i32, n: i32, a: ^complex64, lda: i32) -> i32 ---
	LAPACKE_zlascl_work :: proc(matrix_layout: c.int, type: c.char, kl: i32, ku: i32, cfrom: f64, cto: f64, m: i32, n: i32, a: ^complex128, lda: i32) -> i32 ---
	LAPACKE_slaset_work :: proc(matrix_layout: c.int, uplo: c.char, m: i32, n: i32, alpha: f32, beta: f32, a: ^f32, lda: i32) -> i32 ---
	LAPACKE_dlaset_work :: proc(matrix_layout: c.int, uplo: c.char, m: i32, n: i32, alpha: f64, beta: f64, a: ^f64, lda: i32) -> i32 ---
	LAPACKE_claset_work :: proc(matrix_layout: c.int, uplo: c.char, m: i32, n: i32, alpha: complex64, beta: complex64, a: ^complex64, lda: i32) -> i32 ---
	LAPACKE_zlaset_work :: proc(matrix_layout: c.int, uplo: c.char, m: i32, n: i32, alpha: complex128, beta: complex128, a: ^complex128, lda: i32) -> i32 ---
	LAPACKE_slasrt_work :: proc(id: c.char, n: i32, d: ^f32) -> i32 ---
	LAPACKE_dlasrt_work :: proc(id: c.char, n: i32, d: ^f64) -> i32 ---
	LAPACKE_slassq_work :: proc(n: i32, x: ^f32, incx: i32, scale: ^f32, sumsq: ^f32) -> i32 ---
	LAPACKE_dlassq_work :: proc(n: i32, x: ^f64, incx: i32, scale: ^f64, sumsq: ^f64) -> i32 ---
	LAPACKE_classq_work :: proc(n: i32, x: ^complex64, incx: i32, scale: ^f32, sumsq: ^f32) -> i32 ---
	LAPACKE_zlassq_work :: proc(n: i32, x: ^complex128, incx: i32, scale: ^f64, sumsq: ^f64) -> i32 ---
	LAPACKE_slaswp_work :: proc(matrix_layout: c.int, n: i32, a: ^f32, lda: i32, k1: i32, k2: i32, ipiv: ^i32, incx: i32) -> i32 ---
	LAPACKE_dlaswp_work :: proc(matrix_layout: c.int, n: i32, a: ^f64, lda: i32, k1: i32, k2: i32, ipiv: ^i32, incx: i32) -> i32 ---
	LAPACKE_claswp_work :: proc(matrix_layout: c.int, n: i32, a: ^complex64, lda: i32, k1: i32, k2: i32, ipiv: ^i32, incx: i32) -> i32 ---
	LAPACKE_zlaswp_work :: proc(matrix_layout: c.int, n: i32, a: ^complex128, lda: i32, k1: i32, k2: i32, ipiv: ^i32, incx: i32) -> i32 ---
	LAPACKE_slatms_work :: proc(matrix_layout: c.int, m: i32, n: i32, dist: c.char, iseed: ^i32, sym: c.char, d: ^f32, mode: i32, cond: f32, dmax: f32, kl: i32, ku: i32, pack: c.char, a: ^f32, lda: i32, work: ^f32) -> i32 ---
	LAPACKE_dlatms_work :: proc(matrix_layout: c.int, m: i32, n: i32, dist: c.char, iseed: ^i32, sym: c.char, d: ^f64, mode: i32, cond: f64, dmax: f64, kl: i32, ku: i32, pack: c.char, a: ^f64, lda: i32, work: ^f64) -> i32 ---
	LAPACKE_clatms_work :: proc(matrix_layout: c.int, m: i32, n: i32, dist: c.char, iseed: ^i32, sym: c.char, d: ^f32, mode: i32, cond: f32, dmax: f32, kl: i32, ku: i32, pack: c.char, a: ^complex64, lda: i32, work: ^complex64) -> i32 ---
	LAPACKE_zlatms_work :: proc(matrix_layout: c.int, m: i32, n: i32, dist: c.char, iseed: ^i32, sym: c.char, d: ^f64, mode: i32, cond: f64, dmax: f64, kl: i32, ku: i32, pack: c.char, a: ^complex128, lda: i32, work: ^complex128) -> i32 ---
	LAPACKE_slauum_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f32, lda: i32) -> i32 ---
	LAPACKE_dlauum_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f64, lda: i32) -> i32 ---
	LAPACKE_clauum_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32) -> i32 ---
	LAPACKE_zlauum_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32) -> i32 ---
	LAPACKE_sopgtr_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^f32, tau: ^f32, q: ^f32, ldq: i32, work: ^f32) -> i32 ---
	LAPACKE_dopgtr_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^f64, tau: ^f64, q: ^f64, ldq: i32, work: ^f64) -> i32 ---
	LAPACKE_sopmtr_work :: proc(matrix_layout: c.int, side: c.char, uplo: c.char, trans: c.char, m: i32, n: i32, ap: ^f32, tau: ^f32, _c: ^f32, ldc: i32, work: ^f32) -> i32 ---
	LAPACKE_dopmtr_work :: proc(matrix_layout: c.int, side: c.char, uplo: c.char, trans: c.char, m: i32, n: i32, ap: ^f64, tau: ^f64, _c: ^f64, ldc: i32, work: ^f64) -> i32 ---
	LAPACKE_sorgbr_work :: proc(matrix_layout: c.int, vect: c.char, m: i32, n: i32, k: i32, a: ^f32, lda: i32, tau: ^f32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_dorgbr_work :: proc(matrix_layout: c.int, vect: c.char, m: i32, n: i32, k: i32, a: ^f64, lda: i32, tau: ^f64, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_sorghr_work :: proc(matrix_layout: c.int, n: i32, ilo: i32, ihi: i32, a: ^f32, lda: i32, tau: ^f32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_dorghr_work :: proc(matrix_layout: c.int, n: i32, ilo: i32, ihi: i32, a: ^f64, lda: i32, tau: ^f64, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_sorglq_work :: proc(matrix_layout: c.int, m: i32, n: i32, k: i32, a: ^f32, lda: i32, tau: ^f32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_dorglq_work :: proc(matrix_layout: c.int, m: i32, n: i32, k: i32, a: ^f64, lda: i32, tau: ^f64, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_sorgql_work :: proc(matrix_layout: c.int, m: i32, n: i32, k: i32, a: ^f32, lda: i32, tau: ^f32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_dorgql_work :: proc(matrix_layout: c.int, m: i32, n: i32, k: i32, a: ^f64, lda: i32, tau: ^f64, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_sorgqr_work :: proc(matrix_layout: c.int, m: i32, n: i32, k: i32, a: ^f32, lda: i32, tau: ^f32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_dorgqr_work :: proc(matrix_layout: c.int, m: i32, n: i32, k: i32, a: ^f64, lda: i32, tau: ^f64, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_sorgrq_work :: proc(matrix_layout: c.int, m: i32, n: i32, k: i32, a: ^f32, lda: i32, tau: ^f32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_dorgrq_work :: proc(matrix_layout: c.int, m: i32, n: i32, k: i32, a: ^f64, lda: i32, tau: ^f64, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_sorgtr_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f32, lda: i32, tau: ^f32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_dorgtr_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f64, lda: i32, tau: ^f64, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_sorgtsqr_row_work :: proc(matrix_layout: c.int, m: i32, n: i32, mb: i32, nb: i32, a: ^f32, lda: i32, t: ^f32, ldt: i32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_dorgtsqr_row_work :: proc(matrix_layout: c.int, m: i32, n: i32, mb: i32, nb: i32, a: ^f64, lda: i32, t: ^f64, ldt: i32, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_sormbr_work :: proc(matrix_layout: c.int, vect: c.char, side: c.char, trans: c.char, m: i32, n: i32, k: i32, a: ^f32, lda: i32, tau: ^f32, _c: ^f32, ldc: i32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_dormbr_work :: proc(matrix_layout: c.int, vect: c.char, side: c.char, trans: c.char, m: i32, n: i32, k: i32, a: ^f64, lda: i32, tau: ^f64, _c: ^f64, ldc: i32, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_sormhr_work :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, ilo: i32, ihi: i32, a: ^f32, lda: i32, tau: ^f32, _c: ^f32, ldc: i32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_dormhr_work :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, ilo: i32, ihi: i32, a: ^f64, lda: i32, tau: ^f64, _c: ^f64, ldc: i32, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_sormlq_work :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, a: ^f32, lda: i32, tau: ^f32, _c: ^f32, ldc: i32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_dormlq_work :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, a: ^f64, lda: i32, tau: ^f64, _c: ^f64, ldc: i32, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_sormql_work :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, a: ^f32, lda: i32, tau: ^f32, _c: ^f32, ldc: i32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_dormql_work :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, a: ^f64, lda: i32, tau: ^f64, _c: ^f64, ldc: i32, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_sormqr_work :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, a: ^f32, lda: i32, tau: ^f32, _c: ^f32, ldc: i32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_dormqr_work :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, a: ^f64, lda: i32, tau: ^f64, _c: ^f64, ldc: i32, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_sormrq_work :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, a: ^f32, lda: i32, tau: ^f32, _c: ^f32, ldc: i32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_dormrq_work :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, a: ^f64, lda: i32, tau: ^f64, _c: ^f64, ldc: i32, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_sormrz_work :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, l: i32, a: ^f32, lda: i32, tau: ^f32, _c: ^f32, ldc: i32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_dormrz_work :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, l: i32, a: ^f64, lda: i32, tau: ^f64, _c: ^f64, ldc: i32, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_sormtr_work :: proc(matrix_layout: c.int, side: c.char, uplo: c.char, trans: c.char, m: i32, n: i32, a: ^f32, lda: i32, tau: ^f32, _c: ^f32, ldc: i32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_dormtr_work :: proc(matrix_layout: c.int, side: c.char, uplo: c.char, trans: c.char, m: i32, n: i32, a: ^f64, lda: i32, tau: ^f64, _c: ^f64, ldc: i32, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_spbcon_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kd: i32, ab: ^f32, ldab: i32, anorm: f32, rcond: ^f32, work: ^f32, iwork: ^i32) -> i32 ---
	LAPACKE_dpbcon_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kd: i32, ab: ^f64, ldab: i32, anorm: f64, rcond: ^f64, work: ^f64, iwork: ^i32) -> i32 ---
	LAPACKE_cpbcon_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kd: i32, ab: ^complex64, ldab: i32, anorm: f32, rcond: ^f32, work: ^complex64, rwork: ^f32) -> i32 ---
	LAPACKE_zpbcon_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kd: i32, ab: ^complex128, ldab: i32, anorm: f64, rcond: ^f64, work: ^complex128, rwork: ^f64) -> i32 ---
	LAPACKE_spbequ_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kd: i32, ab: ^f32, ldab: i32, s: ^f32, scond: ^f32, amax: ^f32) -> i32 ---
	LAPACKE_dpbequ_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kd: i32, ab: ^f64, ldab: i32, s: ^f64, scond: ^f64, amax: ^f64) -> i32 ---
	LAPACKE_cpbequ_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kd: i32, ab: ^complex64, ldab: i32, s: ^f32, scond: ^f32, amax: ^f32) -> i32 ---
	LAPACKE_zpbequ_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kd: i32, ab: ^complex128, ldab: i32, s: ^f64, scond: ^f64, amax: ^f64) -> i32 ---
	LAPACKE_spbrfs_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kd: i32, nrhs: i32, ab: ^f32, ldab: i32, afb: ^f32, ldafb: i32, b: ^f32, ldb: i32, x: ^f32, ldx: i32, ferr: ^f32, berr: ^f32, work: ^f32, iwork: ^i32) -> i32 ---
	LAPACKE_dpbrfs_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kd: i32, nrhs: i32, ab: ^f64, ldab: i32, afb: ^f64, ldafb: i32, b: ^f64, ldb: i32, x: ^f64, ldx: i32, ferr: ^f64, berr: ^f64, work: ^f64, iwork: ^i32) -> i32 ---
	LAPACKE_cpbrfs_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kd: i32, nrhs: i32, ab: ^complex64, ldab: i32, afb: ^complex64, ldafb: i32, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, ferr: ^f32, berr: ^f32, work: ^complex64, rwork: ^f32) -> i32 ---
	LAPACKE_zpbrfs_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kd: i32, nrhs: i32, ab: ^complex128, ldab: i32, afb: ^complex128, ldafb: i32, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, ferr: ^f64, berr: ^f64, work: ^complex128, rwork: ^f64) -> i32 ---
	LAPACKE_spbstf_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kb: i32, bb: ^f32, ldbb: i32) -> i32 ---
	LAPACKE_dpbstf_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kb: i32, bb: ^f64, ldbb: i32) -> i32 ---
	LAPACKE_cpbstf_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kb: i32, bb: ^complex64, ldbb: i32) -> i32 ---
	LAPACKE_zpbstf_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kb: i32, bb: ^complex128, ldbb: i32) -> i32 ---
	LAPACKE_spbsv_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kd: i32, nrhs: i32, ab: ^f32, ldab: i32, b: ^f32, ldb: i32) -> i32 ---
	LAPACKE_dpbsv_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kd: i32, nrhs: i32, ab: ^f64, ldab: i32, b: ^f64, ldb: i32) -> i32 ---
	LAPACKE_cpbsv_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kd: i32, nrhs: i32, ab: ^complex64, ldab: i32, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_zpbsv_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kd: i32, nrhs: i32, ab: ^complex128, ldab: i32, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_spbsvx_work :: proc(matrix_layout: c.int, fact: c.char, uplo: c.char, n: i32, kd: i32, nrhs: i32, ab: ^f32, ldab: i32, afb: ^f32, ldafb: i32, equed: cstring, s: ^f32, b: ^f32, ldb: i32, x: ^f32, ldx: i32, rcond: ^f32, ferr: ^f32, berr: ^f32, work: ^f32, iwork: ^i32) -> i32 ---
	LAPACKE_dpbsvx_work :: proc(matrix_layout: c.int, fact: c.char, uplo: c.char, n: i32, kd: i32, nrhs: i32, ab: ^f64, ldab: i32, afb: ^f64, ldafb: i32, equed: cstring, s: ^f64, b: ^f64, ldb: i32, x: ^f64, ldx: i32, rcond: ^f64, ferr: ^f64, berr: ^f64, work: ^f64, iwork: ^i32) -> i32 ---
	LAPACKE_cpbsvx_work :: proc(matrix_layout: c.int, fact: c.char, uplo: c.char, n: i32, kd: i32, nrhs: i32, ab: ^complex64, ldab: i32, afb: ^complex64, ldafb: i32, equed: cstring, s: ^f32, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, rcond: ^f32, ferr: ^f32, berr: ^f32, work: ^complex64, rwork: ^f32) -> i32 ---
	LAPACKE_zpbsvx_work :: proc(matrix_layout: c.int, fact: c.char, uplo: c.char, n: i32, kd: i32, nrhs: i32, ab: ^complex128, ldab: i32, afb: ^complex128, ldafb: i32, equed: cstring, s: ^f64, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, rcond: ^f64, ferr: ^f64, berr: ^f64, work: ^complex128, rwork: ^f64) -> i32 ---
	LAPACKE_spbtrf_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kd: i32, ab: ^f32, ldab: i32) -> i32 ---
	LAPACKE_dpbtrf_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kd: i32, ab: ^f64, ldab: i32) -> i32 ---
	LAPACKE_cpbtrf_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kd: i32, ab: ^complex64, ldab: i32) -> i32 ---
	LAPACKE_zpbtrf_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kd: i32, ab: ^complex128, ldab: i32) -> i32 ---
	LAPACKE_spbtrs_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kd: i32, nrhs: i32, ab: ^f32, ldab: i32, b: ^f32, ldb: i32) -> i32 ---
	LAPACKE_dpbtrs_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kd: i32, nrhs: i32, ab: ^f64, ldab: i32, b: ^f64, ldb: i32) -> i32 ---
	LAPACKE_cpbtrs_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kd: i32, nrhs: i32, ab: ^complex64, ldab: i32, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_zpbtrs_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, kd: i32, nrhs: i32, ab: ^complex128, ldab: i32, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_spftrf_work :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, n: i32, a: ^f32) -> i32 ---
	LAPACKE_dpftrf_work :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, n: i32, a: ^f64) -> i32 ---
	LAPACKE_cpftrf_work :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, n: i32, a: ^complex64) -> i32 ---
	LAPACKE_zpftrf_work :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, n: i32, a: ^complex128) -> i32 ---
	LAPACKE_spftri_work :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, n: i32, a: ^f32) -> i32 ---
	LAPACKE_dpftri_work :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, n: i32, a: ^f64) -> i32 ---
	LAPACKE_cpftri_work :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, n: i32, a: ^complex64) -> i32 ---
	LAPACKE_zpftri_work :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, n: i32, a: ^complex128) -> i32 ---
	LAPACKE_spftrs_work :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, n: i32, nrhs: i32, a: ^f32, b: ^f32, ldb: i32) -> i32 ---
	LAPACKE_dpftrs_work :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, n: i32, nrhs: i32, a: ^f64, b: ^f64, ldb: i32) -> i32 ---
	LAPACKE_cpftrs_work :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, n: i32, nrhs: i32, a: ^complex64, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_zpftrs_work :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, n: i32, nrhs: i32, a: ^complex128, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_spocon_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f32, lda: i32, anorm: f32, rcond: ^f32, work: ^f32, iwork: ^i32) -> i32 ---
	LAPACKE_dpocon_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f64, lda: i32, anorm: f64, rcond: ^f64, work: ^f64, iwork: ^i32) -> i32 ---
	LAPACKE_cpocon_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32, anorm: f32, rcond: ^f32, work: ^complex64, rwork: ^f32) -> i32 ---
	LAPACKE_zpocon_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32, anorm: f64, rcond: ^f64, work: ^complex128, rwork: ^f64) -> i32 ---
	LAPACKE_spoequ_work :: proc(matrix_layout: c.int, n: i32, a: ^f32, lda: i32, s: ^f32, scond: ^f32, amax: ^f32) -> i32 ---
	LAPACKE_dpoequ_work :: proc(matrix_layout: c.int, n: i32, a: ^f64, lda: i32, s: ^f64, scond: ^f64, amax: ^f64) -> i32 ---
	LAPACKE_cpoequ_work :: proc(matrix_layout: c.int, n: i32, a: ^complex64, lda: i32, s: ^f32, scond: ^f32, amax: ^f32) -> i32 ---
	LAPACKE_zpoequ_work :: proc(matrix_layout: c.int, n: i32, a: ^complex128, lda: i32, s: ^f64, scond: ^f64, amax: ^f64) -> i32 ---
	LAPACKE_spoequb_work :: proc(matrix_layout: c.int, n: i32, a: ^f32, lda: i32, s: ^f32, scond: ^f32, amax: ^f32) -> i32 ---
	LAPACKE_dpoequb_work :: proc(matrix_layout: c.int, n: i32, a: ^f64, lda: i32, s: ^f64, scond: ^f64, amax: ^f64) -> i32 ---
	LAPACKE_cpoequb_work :: proc(matrix_layout: c.int, n: i32, a: ^complex64, lda: i32, s: ^f32, scond: ^f32, amax: ^f32) -> i32 ---
	LAPACKE_zpoequb_work :: proc(matrix_layout: c.int, n: i32, a: ^complex128, lda: i32, s: ^f64, scond: ^f64, amax: ^f64) -> i32 ---
	LAPACKE_sporfs_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^f32, lda: i32, af: ^f32, ldaf: i32, b: ^f32, ldb: i32, x: ^f32, ldx: i32, ferr: ^f32, berr: ^f32, work: ^f32, iwork: ^i32) -> i32 ---
	LAPACKE_dporfs_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^f64, lda: i32, af: ^f64, ldaf: i32, b: ^f64, ldb: i32, x: ^f64, ldx: i32, ferr: ^f64, berr: ^f64, work: ^f64, iwork: ^i32) -> i32 ---
	LAPACKE_cporfs_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, af: ^complex64, ldaf: i32, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, ferr: ^f32, berr: ^f32, work: ^complex64, rwork: ^f32) -> i32 ---
	LAPACKE_zporfs_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, af: ^complex128, ldaf: i32, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, ferr: ^f64, berr: ^f64, work: ^complex128, rwork: ^f64) -> i32 ---
	LAPACKE_sporfsx_work :: proc(matrix_layout: c.int, uplo: c.char, equed: c.char, n: i32, nrhs: i32, a: ^f32, lda: i32, af: ^f32, ldaf: i32, s: ^f32, b: ^f32, ldb: i32, x: ^f32, ldx: i32, rcond: ^f32, berr: ^f32, n_err_bnds: i32, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: i32, params: ^f32, work: ^f32, iwork: ^i32) -> i32 ---
	LAPACKE_dporfsx_work :: proc(matrix_layout: c.int, uplo: c.char, equed: c.char, n: i32, nrhs: i32, a: ^f64, lda: i32, af: ^f64, ldaf: i32, s: ^f64, b: ^f64, ldb: i32, x: ^f64, ldx: i32, rcond: ^f64, berr: ^f64, n_err_bnds: i32, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: i32, params: ^f64, work: ^f64, iwork: ^i32) -> i32 ---
	LAPACKE_cporfsx_work :: proc(matrix_layout: c.int, uplo: c.char, equed: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, af: ^complex64, ldaf: i32, s: ^f32, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, rcond: ^f32, berr: ^f32, n_err_bnds: i32, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: i32, params: ^f32, work: ^complex64, rwork: ^f32) -> i32 ---
	LAPACKE_zporfsx_work :: proc(matrix_layout: c.int, uplo: c.char, equed: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, af: ^complex128, ldaf: i32, s: ^f64, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, rcond: ^f64, berr: ^f64, n_err_bnds: i32, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: i32, params: ^f64, work: ^complex128, rwork: ^f64) -> i32 ---
	LAPACKE_sposv_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32) -> i32 ---
	LAPACKE_dposv_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32) -> i32 ---
	LAPACKE_cposv_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_zposv_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_dsposv_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, x: ^f64, ldx: i32, work: ^f64, swork: ^f32, iter: ^i32) -> i32 ---
	LAPACKE_zcposv_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, work: ^complex128, swork: ^complex64, rwork: ^f64, iter: ^i32) -> i32 ---
	LAPACKE_sposvx_work :: proc(matrix_layout: c.int, fact: c.char, uplo: c.char, n: i32, nrhs: i32, a: ^f32, lda: i32, af: ^f32, ldaf: i32, equed: cstring, s: ^f32, b: ^f32, ldb: i32, x: ^f32, ldx: i32, rcond: ^f32, ferr: ^f32, berr: ^f32, work: ^f32, iwork: ^i32) -> i32 ---
	LAPACKE_dposvx_work :: proc(matrix_layout: c.int, fact: c.char, uplo: c.char, n: i32, nrhs: i32, a: ^f64, lda: i32, af: ^f64, ldaf: i32, equed: cstring, s: ^f64, b: ^f64, ldb: i32, x: ^f64, ldx: i32, rcond: ^f64, ferr: ^f64, berr: ^f64, work: ^f64, iwork: ^i32) -> i32 ---
	LAPACKE_cposvx_work :: proc(matrix_layout: c.int, fact: c.char, uplo: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, af: ^complex64, ldaf: i32, equed: cstring, s: ^f32, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, rcond: ^f32, ferr: ^f32, berr: ^f32, work: ^complex64, rwork: ^f32) -> i32 ---
	LAPACKE_zposvx_work :: proc(matrix_layout: c.int, fact: c.char, uplo: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, af: ^complex128, ldaf: i32, equed: cstring, s: ^f64, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, rcond: ^f64, ferr: ^f64, berr: ^f64, work: ^complex128, rwork: ^f64) -> i32 ---
	LAPACKE_sposvxx_work :: proc(matrix_layout: c.int, fact: c.char, uplo: c.char, n: i32, nrhs: i32, a: ^f32, lda: i32, af: ^f32, ldaf: i32, equed: cstring, s: ^f32, b: ^f32, ldb: i32, x: ^f32, ldx: i32, rcond: ^f32, rpvgrw: ^f32, berr: ^f32, n_err_bnds: i32, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: i32, params: ^f32, work: ^f32, iwork: ^i32) -> i32 ---
	LAPACKE_dposvxx_work :: proc(matrix_layout: c.int, fact: c.char, uplo: c.char, n: i32, nrhs: i32, a: ^f64, lda: i32, af: ^f64, ldaf: i32, equed: cstring, s: ^f64, b: ^f64, ldb: i32, x: ^f64, ldx: i32, rcond: ^f64, rpvgrw: ^f64, berr: ^f64, n_err_bnds: i32, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: i32, params: ^f64, work: ^f64, iwork: ^i32) -> i32 ---
	LAPACKE_cposvxx_work :: proc(matrix_layout: c.int, fact: c.char, uplo: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, af: ^complex64, ldaf: i32, equed: cstring, s: ^f32, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, rcond: ^f32, rpvgrw: ^f32, berr: ^f32, n_err_bnds: i32, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: i32, params: ^f32, work: ^complex64, rwork: ^f32) -> i32 ---
	LAPACKE_zposvxx_work :: proc(matrix_layout: c.int, fact: c.char, uplo: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, af: ^complex128, ldaf: i32, equed: cstring, s: ^f64, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, rcond: ^f64, rpvgrw: ^f64, berr: ^f64, n_err_bnds: i32, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: i32, params: ^f64, work: ^complex128, rwork: ^f64) -> i32 ---
	LAPACKE_spotrf2_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f32, lda: i32) -> i32 ---
	LAPACKE_dpotrf2_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f64, lda: i32) -> i32 ---
	LAPACKE_cpotrf2_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32) -> i32 ---
	LAPACKE_zpotrf2_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32) -> i32 ---
	LAPACKE_spotrf_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f32, lda: i32) -> i32 ---
	LAPACKE_dpotrf_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f64, lda: i32) -> i32 ---
	LAPACKE_cpotrf_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32) -> i32 ---
	LAPACKE_zpotrf_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32) -> i32 ---
	LAPACKE_spotri_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f32, lda: i32) -> i32 ---
	LAPACKE_dpotri_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f64, lda: i32) -> i32 ---
	LAPACKE_cpotri_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32) -> i32 ---
	LAPACKE_zpotri_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32) -> i32 ---
	LAPACKE_spotrs_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32) -> i32 ---
	LAPACKE_dpotrs_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32) -> i32 ---
	LAPACKE_cpotrs_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_zpotrs_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_sppcon_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^f32, anorm: f32, rcond: ^f32, work: ^f32, iwork: ^i32) -> i32 ---
	LAPACKE_dppcon_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^f64, anorm: f64, rcond: ^f64, work: ^f64, iwork: ^i32) -> i32 ---
	LAPACKE_cppcon_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^complex64, anorm: f32, rcond: ^f32, work: ^complex64, rwork: ^f32) -> i32 ---
	LAPACKE_zppcon_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^complex128, anorm: f64, rcond: ^f64, work: ^complex128, rwork: ^f64) -> i32 ---
	LAPACKE_sppequ_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^f32, s: ^f32, scond: ^f32, amax: ^f32) -> i32 ---
	LAPACKE_dppequ_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^f64, s: ^f64, scond: ^f64, amax: ^f64) -> i32 ---
	LAPACKE_cppequ_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^complex64, s: ^f32, scond: ^f32, amax: ^f32) -> i32 ---
	LAPACKE_zppequ_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^complex128, s: ^f64, scond: ^f64, amax: ^f64) -> i32 ---
	LAPACKE_spprfs_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, ap: ^f32, afp: ^f32, b: ^f32, ldb: i32, x: ^f32, ldx: i32, ferr: ^f32, berr: ^f32, work: ^f32, iwork: ^i32) -> i32 ---
	LAPACKE_dpprfs_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, ap: ^f64, afp: ^f64, b: ^f64, ldb: i32, x: ^f64, ldx: i32, ferr: ^f64, berr: ^f64, work: ^f64, iwork: ^i32) -> i32 ---
	LAPACKE_cpprfs_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, ap: ^complex64, afp: ^complex64, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, ferr: ^f32, berr: ^f32, work: ^complex64, rwork: ^f32) -> i32 ---
	LAPACKE_zpprfs_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, ap: ^complex128, afp: ^complex128, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, ferr: ^f64, berr: ^f64, work: ^complex128, rwork: ^f64) -> i32 ---
	LAPACKE_sppsv_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, ap: ^f32, b: ^f32, ldb: i32) -> i32 ---
	LAPACKE_dppsv_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, ap: ^f64, b: ^f64, ldb: i32) -> i32 ---
	LAPACKE_cppsv_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, ap: ^complex64, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_zppsv_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, ap: ^complex128, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_sppsvx_work :: proc(matrix_layout: c.int, fact: c.char, uplo: c.char, n: i32, nrhs: i32, ap: ^f32, afp: ^f32, equed: cstring, s: ^f32, b: ^f32, ldb: i32, x: ^f32, ldx: i32, rcond: ^f32, ferr: ^f32, berr: ^f32, work: ^f32, iwork: ^i32) -> i32 ---
	LAPACKE_dppsvx_work :: proc(matrix_layout: c.int, fact: c.char, uplo: c.char, n: i32, nrhs: i32, ap: ^f64, afp: ^f64, equed: cstring, s: ^f64, b: ^f64, ldb: i32, x: ^f64, ldx: i32, rcond: ^f64, ferr: ^f64, berr: ^f64, work: ^f64, iwork: ^i32) -> i32 ---
	LAPACKE_cppsvx_work :: proc(matrix_layout: c.int, fact: c.char, uplo: c.char, n: i32, nrhs: i32, ap: ^complex64, afp: ^complex64, equed: cstring, s: ^f32, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, rcond: ^f32, ferr: ^f32, berr: ^f32, work: ^complex64, rwork: ^f32) -> i32 ---
	LAPACKE_zppsvx_work :: proc(matrix_layout: c.int, fact: c.char, uplo: c.char, n: i32, nrhs: i32, ap: ^complex128, afp: ^complex128, equed: cstring, s: ^f64, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, rcond: ^f64, ferr: ^f64, berr: ^f64, work: ^complex128, rwork: ^f64) -> i32 ---
	LAPACKE_spptrf_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^f32) -> i32 ---
	LAPACKE_dpptrf_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^f64) -> i32 ---
	LAPACKE_cpptrf_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^complex64) -> i32 ---
	LAPACKE_zpptrf_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^complex128) -> i32 ---
	LAPACKE_spptri_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^f32) -> i32 ---
	LAPACKE_dpptri_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^f64) -> i32 ---
	LAPACKE_cpptri_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^complex64) -> i32 ---
	LAPACKE_zpptri_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^complex128) -> i32 ---
	LAPACKE_spptrs_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, ap: ^f32, b: ^f32, ldb: i32) -> i32 ---
	LAPACKE_dpptrs_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, ap: ^f64, b: ^f64, ldb: i32) -> i32 ---
	LAPACKE_cpptrs_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, ap: ^complex64, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_zpptrs_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, ap: ^complex128, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_spstrf_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f32, lda: i32, piv: ^i32, rank: ^i32, tol: f32, work: ^f32) -> i32 ---
	LAPACKE_dpstrf_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f64, lda: i32, piv: ^i32, rank: ^i32, tol: f64, work: ^f64) -> i32 ---
	LAPACKE_cpstrf_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32, piv: ^i32, rank: ^i32, tol: f32, work: ^f32) -> i32 ---
	LAPACKE_zpstrf_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32, piv: ^i32, rank: ^i32, tol: f64, work: ^f64) -> i32 ---
	LAPACKE_sptcon_work :: proc(n: i32, d: ^f32, e: ^f32, anorm: f32, rcond: ^f32, work: ^f32) -> i32 ---
	LAPACKE_dptcon_work :: proc(n: i32, d: ^f64, e: ^f64, anorm: f64, rcond: ^f64, work: ^f64) -> i32 ---
	LAPACKE_cptcon_work :: proc(n: i32, d: ^f32, e: ^complex64, anorm: f32, rcond: ^f32, work: ^f32) -> i32 ---
	LAPACKE_zptcon_work :: proc(n: i32, d: ^f64, e: ^complex128, anorm: f64, rcond: ^f64, work: ^f64) -> i32 ---
	LAPACKE_spteqr_work :: proc(matrix_layout: c.int, compz: c.char, n: i32, d: ^f32, e: ^f32, z: ^f32, ldz: i32, work: ^f32) -> i32 ---
	LAPACKE_dpteqr_work :: proc(matrix_layout: c.int, compz: c.char, n: i32, d: ^f64, e: ^f64, z: ^f64, ldz: i32, work: ^f64) -> i32 ---
	LAPACKE_cpteqr_work :: proc(matrix_layout: c.int, compz: c.char, n: i32, d: ^f32, e: ^f32, z: ^complex64, ldz: i32, work: ^f32) -> i32 ---
	LAPACKE_zpteqr_work :: proc(matrix_layout: c.int, compz: c.char, n: i32, d: ^f64, e: ^f64, z: ^complex128, ldz: i32, work: ^f64) -> i32 ---
	LAPACKE_sptrfs_work :: proc(matrix_layout: c.int, n: i32, nrhs: i32, d: ^f32, e: ^f32, df: ^f32, ef: ^f32, b: ^f32, ldb: i32, x: ^f32, ldx: i32, ferr: ^f32, berr: ^f32, work: ^f32) -> i32 ---
	LAPACKE_dptrfs_work :: proc(matrix_layout: c.int, n: i32, nrhs: i32, d: ^f64, e: ^f64, df: ^f64, ef: ^f64, b: ^f64, ldb: i32, x: ^f64, ldx: i32, ferr: ^f64, berr: ^f64, work: ^f64) -> i32 ---
	LAPACKE_cptrfs_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, d: ^f32, e: ^complex64, df: ^f32, ef: ^complex64, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, ferr: ^f32, berr: ^f32, work: ^complex64, rwork: ^f32) -> i32 ---
	LAPACKE_zptrfs_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, d: ^f64, e: ^complex128, df: ^f64, ef: ^complex128, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, ferr: ^f64, berr: ^f64, work: ^complex128, rwork: ^f64) -> i32 ---
	LAPACKE_sptsv_work :: proc(matrix_layout: c.int, n: i32, nrhs: i32, d: ^f32, e: ^f32, b: ^f32, ldb: i32) -> i32 ---
	LAPACKE_dptsv_work :: proc(matrix_layout: c.int, n: i32, nrhs: i32, d: ^f64, e: ^f64, b: ^f64, ldb: i32) -> i32 ---
	LAPACKE_cptsv_work :: proc(matrix_layout: c.int, n: i32, nrhs: i32, d: ^f32, e: ^complex64, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_zptsv_work :: proc(matrix_layout: c.int, n: i32, nrhs: i32, d: ^f64, e: ^complex128, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_sptsvx_work :: proc(matrix_layout: c.int, fact: c.char, n: i32, nrhs: i32, d: ^f32, e: ^f32, df: ^f32, ef: ^f32, b: ^f32, ldb: i32, x: ^f32, ldx: i32, rcond: ^f32, ferr: ^f32, berr: ^f32, work: ^f32) -> i32 ---
	LAPACKE_dptsvx_work :: proc(matrix_layout: c.int, fact: c.char, n: i32, nrhs: i32, d: ^f64, e: ^f64, df: ^f64, ef: ^f64, b: ^f64, ldb: i32, x: ^f64, ldx: i32, rcond: ^f64, ferr: ^f64, berr: ^f64, work: ^f64) -> i32 ---
	LAPACKE_cptsvx_work :: proc(matrix_layout: c.int, fact: c.char, n: i32, nrhs: i32, d: ^f32, e: ^complex64, df: ^f32, ef: ^complex64, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, rcond: ^f32, ferr: ^f32, berr: ^f32, work: ^complex64, rwork: ^f32) -> i32 ---
	LAPACKE_zptsvx_work :: proc(matrix_layout: c.int, fact: c.char, n: i32, nrhs: i32, d: ^f64, e: ^complex128, df: ^f64, ef: ^complex128, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, rcond: ^f64, ferr: ^f64, berr: ^f64, work: ^complex128, rwork: ^f64) -> i32 ---
	LAPACKE_spttrf_work :: proc(n: i32, d: ^f32, e: ^f32) -> i32 ---
	LAPACKE_dpttrf_work :: proc(n: i32, d: ^f64, e: ^f64) -> i32 ---
	LAPACKE_cpttrf_work :: proc(n: i32, d: ^f32, e: ^complex64) -> i32 ---
	LAPACKE_zpttrf_work :: proc(n: i32, d: ^f64, e: ^complex128) -> i32 ---
	LAPACKE_spttrs_work :: proc(matrix_layout: c.int, n: i32, nrhs: i32, d: ^f32, e: ^f32, b: ^f32, ldb: i32) -> i32 ---
	LAPACKE_dpttrs_work :: proc(matrix_layout: c.int, n: i32, nrhs: i32, d: ^f64, e: ^f64, b: ^f64, ldb: i32) -> i32 ---
	LAPACKE_cpttrs_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, d: ^f32, e: ^complex64, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_zpttrs_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, d: ^f64, e: ^complex128, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_ssbev_work :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, kd: i32, ab: ^f32, ldab: i32, w: ^f32, z: ^f32, ldz: i32, work: ^f32) -> i32 ---
	LAPACKE_dsbev_work :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, kd: i32, ab: ^f64, ldab: i32, w: ^f64, z: ^f64, ldz: i32, work: ^f64) -> i32 ---
	LAPACKE_ssbevd_work :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, kd: i32, ab: ^f32, ldab: i32, w: ^f32, z: ^f32, ldz: i32, work: ^f32, lwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_dsbevd_work :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, kd: i32, ab: ^f64, ldab: i32, w: ^f64, z: ^f64, ldz: i32, work: ^f64, lwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_ssbevx_work :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, uplo: c.char, n: i32, kd: i32, ab: ^f32, ldab: i32, q: ^f32, ldq: i32, vl: f32, vu: f32, il: i32, iu: i32, abstol: f32, m: ^i32, w: ^f32, z: ^f32, ldz: i32, work: ^f32, iwork: ^i32, ifail: ^i32) -> i32 ---
	LAPACKE_dsbevx_work :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, uplo: c.char, n: i32, kd: i32, ab: ^f64, ldab: i32, q: ^f64, ldq: i32, vl: f64, vu: f64, il: i32, iu: i32, abstol: f64, m: ^i32, w: ^f64, z: ^f64, ldz: i32, work: ^f64, iwork: ^i32, ifail: ^i32) -> i32 ---
	LAPACKE_ssbgst_work :: proc(matrix_layout: c.int, vect: c.char, uplo: c.char, n: i32, ka: i32, kb: i32, ab: ^f32, ldab: i32, bb: ^f32, ldbb: i32, x: ^f32, ldx: i32, work: ^f32) -> i32 ---
	LAPACKE_dsbgst_work :: proc(matrix_layout: c.int, vect: c.char, uplo: c.char, n: i32, ka: i32, kb: i32, ab: ^f64, ldab: i32, bb: ^f64, ldbb: i32, x: ^f64, ldx: i32, work: ^f64) -> i32 ---
	LAPACKE_ssbgv_work :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, ka: i32, kb: i32, ab: ^f32, ldab: i32, bb: ^f32, ldbb: i32, w: ^f32, z: ^f32, ldz: i32, work: ^f32) -> i32 ---
	LAPACKE_dsbgv_work :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, ka: i32, kb: i32, ab: ^f64, ldab: i32, bb: ^f64, ldbb: i32, w: ^f64, z: ^f64, ldz: i32, work: ^f64) -> i32 ---
	LAPACKE_ssbgvd_work :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, ka: i32, kb: i32, ab: ^f32, ldab: i32, bb: ^f32, ldbb: i32, w: ^f32, z: ^f32, ldz: i32, work: ^f32, lwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_dsbgvd_work :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, ka: i32, kb: i32, ab: ^f64, ldab: i32, bb: ^f64, ldbb: i32, w: ^f64, z: ^f64, ldz: i32, work: ^f64, lwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_ssbgvx_work :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, uplo: c.char, n: i32, ka: i32, kb: i32, ab: ^f32, ldab: i32, bb: ^f32, ldbb: i32, q: ^f32, ldq: i32, vl: f32, vu: f32, il: i32, iu: i32, abstol: f32, m: ^i32, w: ^f32, z: ^f32, ldz: i32, work: ^f32, iwork: ^i32, ifail: ^i32) -> i32 ---
	LAPACKE_dsbgvx_work :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, uplo: c.char, n: i32, ka: i32, kb: i32, ab: ^f64, ldab: i32, bb: ^f64, ldbb: i32, q: ^f64, ldq: i32, vl: f64, vu: f64, il: i32, iu: i32, abstol: f64, m: ^i32, w: ^f64, z: ^f64, ldz: i32, work: ^f64, iwork: ^i32, ifail: ^i32) -> i32 ---
	LAPACKE_ssbtrd_work :: proc(matrix_layout: c.int, vect: c.char, uplo: c.char, n: i32, kd: i32, ab: ^f32, ldab: i32, d: ^f32, e: ^f32, q: ^f32, ldq: i32, work: ^f32) -> i32 ---
	LAPACKE_dsbtrd_work :: proc(matrix_layout: c.int, vect: c.char, uplo: c.char, n: i32, kd: i32, ab: ^f64, ldab: i32, d: ^f64, e: ^f64, q: ^f64, ldq: i32, work: ^f64) -> i32 ---
	LAPACKE_ssfrk_work :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, trans: c.char, n: i32, k: i32, alpha: f32, a: ^f32, lda: i32, beta: f32, _c: ^f32) -> i32 ---
	LAPACKE_dsfrk_work :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, trans: c.char, n: i32, k: i32, alpha: f64, a: ^f64, lda: i32, beta: f64, _c: ^f64) -> i32 ---
	LAPACKE_sspcon_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^f32, ipiv: ^i32, anorm: f32, rcond: ^f32, work: ^f32, iwork: ^i32) -> i32 ---
	LAPACKE_dspcon_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^f64, ipiv: ^i32, anorm: f64, rcond: ^f64, work: ^f64, iwork: ^i32) -> i32 ---
	LAPACKE_cspcon_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^complex64, ipiv: ^i32, anorm: f32, rcond: ^f32, work: ^complex64) -> i32 ---
	LAPACKE_zspcon_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^complex128, ipiv: ^i32, anorm: f64, rcond: ^f64, work: ^complex128) -> i32 ---
	LAPACKE_sspev_work :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, ap: ^f32, w: ^f32, z: ^f32, ldz: i32, work: ^f32) -> i32 ---
	LAPACKE_dspev_work :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, ap: ^f64, w: ^f64, z: ^f64, ldz: i32, work: ^f64) -> i32 ---
	LAPACKE_sspevd_work :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, ap: ^f32, w: ^f32, z: ^f32, ldz: i32, work: ^f32, lwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_dspevd_work :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, ap: ^f64, w: ^f64, z: ^f64, ldz: i32, work: ^f64, lwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_sspevx_work :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, uplo: c.char, n: i32, ap: ^f32, vl: f32, vu: f32, il: i32, iu: i32, abstol: f32, m: ^i32, w: ^f32, z: ^f32, ldz: i32, work: ^f32, iwork: ^i32, ifail: ^i32) -> i32 ---
	LAPACKE_dspevx_work :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, uplo: c.char, n: i32, ap: ^f64, vl: f64, vu: f64, il: i32, iu: i32, abstol: f64, m: ^i32, w: ^f64, z: ^f64, ldz: i32, work: ^f64, iwork: ^i32, ifail: ^i32) -> i32 ---
	LAPACKE_sspgst_work :: proc(matrix_layout: c.int, itype: i32, uplo: c.char, n: i32, ap: ^f32, bp: ^f32) -> i32 ---
	LAPACKE_dspgst_work :: proc(matrix_layout: c.int, itype: i32, uplo: c.char, n: i32, ap: ^f64, bp: ^f64) -> i32 ---
	LAPACKE_sspgv_work :: proc(matrix_layout: c.int, itype: i32, jobz: c.char, uplo: c.char, n: i32, ap: ^f32, bp: ^f32, w: ^f32, z: ^f32, ldz: i32, work: ^f32) -> i32 ---
	LAPACKE_dspgv_work :: proc(matrix_layout: c.int, itype: i32, jobz: c.char, uplo: c.char, n: i32, ap: ^f64, bp: ^f64, w: ^f64, z: ^f64, ldz: i32, work: ^f64) -> i32 ---
	LAPACKE_sspgvd_work :: proc(matrix_layout: c.int, itype: i32, jobz: c.char, uplo: c.char, n: i32, ap: ^f32, bp: ^f32, w: ^f32, z: ^f32, ldz: i32, work: ^f32, lwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_dspgvd_work :: proc(matrix_layout: c.int, itype: i32, jobz: c.char, uplo: c.char, n: i32, ap: ^f64, bp: ^f64, w: ^f64, z: ^f64, ldz: i32, work: ^f64, lwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_sspgvx_work :: proc(matrix_layout: c.int, itype: i32, jobz: c.char, range: c.char, uplo: c.char, n: i32, ap: ^f32, bp: ^f32, vl: f32, vu: f32, il: i32, iu: i32, abstol: f32, m: ^i32, w: ^f32, z: ^f32, ldz: i32, work: ^f32, iwork: ^i32, ifail: ^i32) -> i32 ---
	LAPACKE_dspgvx_work :: proc(matrix_layout: c.int, itype: i32, jobz: c.char, range: c.char, uplo: c.char, n: i32, ap: ^f64, bp: ^f64, vl: f64, vu: f64, il: i32, iu: i32, abstol: f64, m: ^i32, w: ^f64, z: ^f64, ldz: i32, work: ^f64, iwork: ^i32, ifail: ^i32) -> i32 ---
	LAPACKE_ssprfs_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, ap: ^f32, afp: ^f32, ipiv: ^i32, b: ^f32, ldb: i32, x: ^f32, ldx: i32, ferr: ^f32, berr: ^f32, work: ^f32, iwork: ^i32) -> i32 ---
	LAPACKE_dsprfs_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, ap: ^f64, afp: ^f64, ipiv: ^i32, b: ^f64, ldb: i32, x: ^f64, ldx: i32, ferr: ^f64, berr: ^f64, work: ^f64, iwork: ^i32) -> i32 ---
	LAPACKE_csprfs_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, ap: ^complex64, afp: ^complex64, ipiv: ^i32, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, ferr: ^f32, berr: ^f32, work: ^complex64, rwork: ^f32) -> i32 ---
	LAPACKE_zsprfs_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, ap: ^complex128, afp: ^complex128, ipiv: ^i32, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, ferr: ^f64, berr: ^f64, work: ^complex128, rwork: ^f64) -> i32 ---
	LAPACKE_sspsv_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, ap: ^f32, ipiv: ^i32, b: ^f32, ldb: i32) -> i32 ---
	LAPACKE_dspsv_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, ap: ^f64, ipiv: ^i32, b: ^f64, ldb: i32) -> i32 ---
	LAPACKE_cspsv_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, ap: ^complex64, ipiv: ^i32, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_zspsv_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, ap: ^complex128, ipiv: ^i32, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_sspsvx_work :: proc(matrix_layout: c.int, fact: c.char, uplo: c.char, n: i32, nrhs: i32, ap: ^f32, afp: ^f32, ipiv: ^i32, b: ^f32, ldb: i32, x: ^f32, ldx: i32, rcond: ^f32, ferr: ^f32, berr: ^f32, work: ^f32, iwork: ^i32) -> i32 ---
	LAPACKE_dspsvx_work :: proc(matrix_layout: c.int, fact: c.char, uplo: c.char, n: i32, nrhs: i32, ap: ^f64, afp: ^f64, ipiv: ^i32, b: ^f64, ldb: i32, x: ^f64, ldx: i32, rcond: ^f64, ferr: ^f64, berr: ^f64, work: ^f64, iwork: ^i32) -> i32 ---
	LAPACKE_cspsvx_work :: proc(matrix_layout: c.int, fact: c.char, uplo: c.char, n: i32, nrhs: i32, ap: ^complex64, afp: ^complex64, ipiv: ^i32, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, rcond: ^f32, ferr: ^f32, berr: ^f32, work: ^complex64, rwork: ^f32) -> i32 ---
	LAPACKE_zspsvx_work :: proc(matrix_layout: c.int, fact: c.char, uplo: c.char, n: i32, nrhs: i32, ap: ^complex128, afp: ^complex128, ipiv: ^i32, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, rcond: ^f64, ferr: ^f64, berr: ^f64, work: ^complex128, rwork: ^f64) -> i32 ---
	LAPACKE_ssptrd_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^f32, d: ^f32, e: ^f32, tau: ^f32) -> i32 ---
	LAPACKE_dsptrd_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^f64, d: ^f64, e: ^f64, tau: ^f64) -> i32 ---
	LAPACKE_ssptrf_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^f32, ipiv: ^i32) -> i32 ---
	LAPACKE_dsptrf_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^f64, ipiv: ^i32) -> i32 ---
	LAPACKE_csptrf_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^complex64, ipiv: ^i32) -> i32 ---
	LAPACKE_zsptrf_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^complex128, ipiv: ^i32) -> i32 ---
	LAPACKE_ssptri_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^f32, ipiv: ^i32, work: ^f32) -> i32 ---
	LAPACKE_dsptri_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^f64, ipiv: ^i32, work: ^f64) -> i32 ---
	LAPACKE_csptri_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^complex64, ipiv: ^i32, work: ^complex64) -> i32 ---
	LAPACKE_zsptri_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^complex128, ipiv: ^i32, work: ^complex128) -> i32 ---
	LAPACKE_ssptrs_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, ap: ^f32, ipiv: ^i32, b: ^f32, ldb: i32) -> i32 ---
	LAPACKE_dsptrs_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, ap: ^f64, ipiv: ^i32, b: ^f64, ldb: i32) -> i32 ---
	LAPACKE_csptrs_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, ap: ^complex64, ipiv: ^i32, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_zsptrs_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, ap: ^complex128, ipiv: ^i32, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_sstebz_work :: proc(range: c.char, order: c.char, n: i32, vl: f32, vu: f32, il: i32, iu: i32, abstol: f32, d: ^f32, e: ^f32, m: ^i32, nsplit: ^i32, w: ^f32, iblock: ^i32, isplit: ^i32, work: ^f32, iwork: ^i32) -> i32 ---
	LAPACKE_dstebz_work :: proc(range: c.char, order: c.char, n: i32, vl: f64, vu: f64, il: i32, iu: i32, abstol: f64, d: ^f64, e: ^f64, m: ^i32, nsplit: ^i32, w: ^f64, iblock: ^i32, isplit: ^i32, work: ^f64, iwork: ^i32) -> i32 ---
	LAPACKE_sstedc_work :: proc(matrix_layout: c.int, compz: c.char, n: i32, d: ^f32, e: ^f32, z: ^f32, ldz: i32, work: ^f32, lwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_dstedc_work :: proc(matrix_layout: c.int, compz: c.char, n: i32, d: ^f64, e: ^f64, z: ^f64, ldz: i32, work: ^f64, lwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_cstedc_work :: proc(matrix_layout: c.int, compz: c.char, n: i32, d: ^f32, e: ^f32, z: ^complex64, ldz: i32, work: ^complex64, lwork: i32, rwork: ^f32, lrwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_zstedc_work :: proc(matrix_layout: c.int, compz: c.char, n: i32, d: ^f64, e: ^f64, z: ^complex128, ldz: i32, work: ^complex128, lwork: i32, rwork: ^f64, lrwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_sstegr_work :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, n: i32, d: ^f32, e: ^f32, vl: f32, vu: f32, il: i32, iu: i32, abstol: f32, m: ^i32, w: ^f32, z: ^f32, ldz: i32, isuppz: ^i32, work: ^f32, lwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_dstegr_work :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, n: i32, d: ^f64, e: ^f64, vl: f64, vu: f64, il: i32, iu: i32, abstol: f64, m: ^i32, w: ^f64, z: ^f64, ldz: i32, isuppz: ^i32, work: ^f64, lwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_cstegr_work :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, n: i32, d: ^f32, e: ^f32, vl: f32, vu: f32, il: i32, iu: i32, abstol: f32, m: ^i32, w: ^f32, z: ^complex64, ldz: i32, isuppz: ^i32, work: ^f32, lwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_zstegr_work :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, n: i32, d: ^f64, e: ^f64, vl: f64, vu: f64, il: i32, iu: i32, abstol: f64, m: ^i32, w: ^f64, z: ^complex128, ldz: i32, isuppz: ^i32, work: ^f64, lwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_sstein_work :: proc(matrix_layout: c.int, n: i32, d: ^f32, e: ^f32, m: i32, w: ^f32, iblock: ^i32, isplit: ^i32, z: ^f32, ldz: i32, work: ^f32, iwork: ^i32, ifailv: ^i32) -> i32 ---
	LAPACKE_dstein_work :: proc(matrix_layout: c.int, n: i32, d: ^f64, e: ^f64, m: i32, w: ^f64, iblock: ^i32, isplit: ^i32, z: ^f64, ldz: i32, work: ^f64, iwork: ^i32, ifailv: ^i32) -> i32 ---
	LAPACKE_cstein_work :: proc(matrix_layout: c.int, n: i32, d: ^f32, e: ^f32, m: i32, w: ^f32, iblock: ^i32, isplit: ^i32, z: ^complex64, ldz: i32, work: ^f32, iwork: ^i32, ifailv: ^i32) -> i32 ---
	LAPACKE_zstein_work :: proc(matrix_layout: c.int, n: i32, d: ^f64, e: ^f64, m: i32, w: ^f64, iblock: ^i32, isplit: ^i32, z: ^complex128, ldz: i32, work: ^f64, iwork: ^i32, ifailv: ^i32) -> i32 ---
	LAPACKE_sstemr_work :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, n: i32, d: ^f32, e: ^f32, vl: f32, vu: f32, il: i32, iu: i32, m: ^i32, w: ^f32, z: ^f32, ldz: i32, nzc: i32, isuppz: ^i32, tryrac: ^i32, work: ^f32, lwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_dstemr_work :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, n: i32, d: ^f64, e: ^f64, vl: f64, vu: f64, il: i32, iu: i32, m: ^i32, w: ^f64, z: ^f64, ldz: i32, nzc: i32, isuppz: ^i32, tryrac: ^i32, work: ^f64, lwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_cstemr_work :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, n: i32, d: ^f32, e: ^f32, vl: f32, vu: f32, il: i32, iu: i32, m: ^i32, w: ^f32, z: ^complex64, ldz: i32, nzc: i32, isuppz: ^i32, tryrac: ^i32, work: ^f32, lwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_zstemr_work :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, n: i32, d: ^f64, e: ^f64, vl: f64, vu: f64, il: i32, iu: i32, m: ^i32, w: ^f64, z: ^complex128, ldz: i32, nzc: i32, isuppz: ^i32, tryrac: ^i32, work: ^f64, lwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_ssteqr_work :: proc(matrix_layout: c.int, compz: c.char, n: i32, d: ^f32, e: ^f32, z: ^f32, ldz: i32, work: ^f32) -> i32 ---
	LAPACKE_dsteqr_work :: proc(matrix_layout: c.int, compz: c.char, n: i32, d: ^f64, e: ^f64, z: ^f64, ldz: i32, work: ^f64) -> i32 ---
	LAPACKE_csteqr_work :: proc(matrix_layout: c.int, compz: c.char, n: i32, d: ^f32, e: ^f32, z: ^complex64, ldz: i32, work: ^f32) -> i32 ---
	LAPACKE_zsteqr_work :: proc(matrix_layout: c.int, compz: c.char, n: i32, d: ^f64, e: ^f64, z: ^complex128, ldz: i32, work: ^f64) -> i32 ---
	LAPACKE_ssterf_work :: proc(n: i32, d: ^f32, e: ^f32) -> i32 ---
	LAPACKE_dsterf_work :: proc(n: i32, d: ^f64, e: ^f64) -> i32 ---
	LAPACKE_sstev_work :: proc(matrix_layout: c.int, jobz: c.char, n: i32, d: ^f32, e: ^f32, z: ^f32, ldz: i32, work: ^f32) -> i32 ---
	LAPACKE_dstev_work :: proc(matrix_layout: c.int, jobz: c.char, n: i32, d: ^f64, e: ^f64, z: ^f64, ldz: i32, work: ^f64) -> i32 ---
	LAPACKE_sstevd_work :: proc(matrix_layout: c.int, jobz: c.char, n: i32, d: ^f32, e: ^f32, z: ^f32, ldz: i32, work: ^f32, lwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_dstevd_work :: proc(matrix_layout: c.int, jobz: c.char, n: i32, d: ^f64, e: ^f64, z: ^f64, ldz: i32, work: ^f64, lwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_sstevr_work :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, n: i32, d: ^f32, e: ^f32, vl: f32, vu: f32, il: i32, iu: i32, abstol: f32, m: ^i32, w: ^f32, z: ^f32, ldz: i32, isuppz: ^i32, work: ^f32, lwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_dstevr_work :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, n: i32, d: ^f64, e: ^f64, vl: f64, vu: f64, il: i32, iu: i32, abstol: f64, m: ^i32, w: ^f64, z: ^f64, ldz: i32, isuppz: ^i32, work: ^f64, lwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_sstevx_work :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, n: i32, d: ^f32, e: ^f32, vl: f32, vu: f32, il: i32, iu: i32, abstol: f32, m: ^i32, w: ^f32, z: ^f32, ldz: i32, work: ^f32, iwork: ^i32, ifail: ^i32) -> i32 ---
	LAPACKE_dstevx_work :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, n: i32, d: ^f64, e: ^f64, vl: f64, vu: f64, il: i32, iu: i32, abstol: f64, m: ^i32, w: ^f64, z: ^f64, ldz: i32, work: ^f64, iwork: ^i32, ifail: ^i32) -> i32 ---
	LAPACKE_ssycon_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f32, lda: i32, ipiv: ^i32, anorm: f32, rcond: ^f32, work: ^f32, iwork: ^i32) -> i32 ---
	LAPACKE_dsycon_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f64, lda: i32, ipiv: ^i32, anorm: f64, rcond: ^f64, work: ^f64, iwork: ^i32) -> i32 ---
	LAPACKE_csycon_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32, ipiv: ^i32, anorm: f32, rcond: ^f32, work: ^complex64) -> i32 ---
	LAPACKE_zsycon_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32, ipiv: ^i32, anorm: f64, rcond: ^f64, work: ^complex128) -> i32 ---
	LAPACKE_ssyequb_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f32, lda: i32, s: ^f32, scond: ^f32, amax: ^f32, work: ^f32) -> i32 ---
	LAPACKE_dsyequb_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f64, lda: i32, s: ^f64, scond: ^f64, amax: ^f64, work: ^f64) -> i32 ---
	LAPACKE_csyequb_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32, s: ^f32, scond: ^f32, amax: ^f32, work: ^complex64) -> i32 ---
	LAPACKE_zsyequb_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32, s: ^f64, scond: ^f64, amax: ^f64, work: ^complex128) -> i32 ---
	LAPACKE_ssyev_work :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, a: ^f32, lda: i32, w: ^f32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_dsyev_work :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, a: ^f64, lda: i32, w: ^f64, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_ssyevd_work :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, a: ^f32, lda: i32, w: ^f32, work: ^f32, lwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_dsyevd_work :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, a: ^f64, lda: i32, w: ^f64, work: ^f64, lwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_ssyevr_work :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, uplo: c.char, n: i32, a: ^f32, lda: i32, vl: f32, vu: f32, il: i32, iu: i32, abstol: f32, m: ^i32, w: ^f32, z: ^f32, ldz: i32, isuppz: ^i32, work: ^f32, lwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_dsyevr_work :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, uplo: c.char, n: i32, a: ^f64, lda: i32, vl: f64, vu: f64, il: i32, iu: i32, abstol: f64, m: ^i32, w: ^f64, z: ^f64, ldz: i32, isuppz: ^i32, work: ^f64, lwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_ssyevx_work :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, uplo: c.char, n: i32, a: ^f32, lda: i32, vl: f32, vu: f32, il: i32, iu: i32, abstol: f32, m: ^i32, w: ^f32, z: ^f32, ldz: i32, work: ^f32, lwork: i32, iwork: ^i32, ifail: ^i32) -> i32 ---
	LAPACKE_dsyevx_work :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, uplo: c.char, n: i32, a: ^f64, lda: i32, vl: f64, vu: f64, il: i32, iu: i32, abstol: f64, m: ^i32, w: ^f64, z: ^f64, ldz: i32, work: ^f64, lwork: i32, iwork: ^i32, ifail: ^i32) -> i32 ---
	LAPACKE_ssygst_work :: proc(matrix_layout: c.int, itype: i32, uplo: c.char, n: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32) -> i32 ---
	LAPACKE_dsygst_work :: proc(matrix_layout: c.int, itype: i32, uplo: c.char, n: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32) -> i32 ---
	LAPACKE_ssygv_work :: proc(matrix_layout: c.int, itype: i32, jobz: c.char, uplo: c.char, n: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32, w: ^f32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_dsygv_work :: proc(matrix_layout: c.int, itype: i32, jobz: c.char, uplo: c.char, n: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, w: ^f64, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_ssygvd_work :: proc(matrix_layout: c.int, itype: i32, jobz: c.char, uplo: c.char, n: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32, w: ^f32, work: ^f32, lwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_dsygvd_work :: proc(matrix_layout: c.int, itype: i32, jobz: c.char, uplo: c.char, n: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, w: ^f64, work: ^f64, lwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_ssygvx_work :: proc(matrix_layout: c.int, itype: i32, jobz: c.char, range: c.char, uplo: c.char, n: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32, vl: f32, vu: f32, il: i32, iu: i32, abstol: f32, m: ^i32, w: ^f32, z: ^f32, ldz: i32, work: ^f32, lwork: i32, iwork: ^i32, ifail: ^i32) -> i32 ---
	LAPACKE_dsygvx_work :: proc(matrix_layout: c.int, itype: i32, jobz: c.char, range: c.char, uplo: c.char, n: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, vl: f64, vu: f64, il: i32, iu: i32, abstol: f64, m: ^i32, w: ^f64, z: ^f64, ldz: i32, work: ^f64, lwork: i32, iwork: ^i32, ifail: ^i32) -> i32 ---
	LAPACKE_ssyrfs_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^f32, lda: i32, af: ^f32, ldaf: i32, ipiv: ^i32, b: ^f32, ldb: i32, x: ^f32, ldx: i32, ferr: ^f32, berr: ^f32, work: ^f32, iwork: ^i32) -> i32 ---
	LAPACKE_dsyrfs_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^f64, lda: i32, af: ^f64, ldaf: i32, ipiv: ^i32, b: ^f64, ldb: i32, x: ^f64, ldx: i32, ferr: ^f64, berr: ^f64, work: ^f64, iwork: ^i32) -> i32 ---
	LAPACKE_csyrfs_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, af: ^complex64, ldaf: i32, ipiv: ^i32, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, ferr: ^f32, berr: ^f32, work: ^complex64, rwork: ^f32) -> i32 ---
	LAPACKE_zsyrfs_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, af: ^complex128, ldaf: i32, ipiv: ^i32, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, ferr: ^f64, berr: ^f64, work: ^complex128, rwork: ^f64) -> i32 ---
	LAPACKE_ssyrfsx_work :: proc(matrix_layout: c.int, uplo: c.char, equed: c.char, n: i32, nrhs: i32, a: ^f32, lda: i32, af: ^f32, ldaf: i32, ipiv: ^i32, s: ^f32, b: ^f32, ldb: i32, x: ^f32, ldx: i32, rcond: ^f32, berr: ^f32, n_err_bnds: i32, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: i32, params: ^f32, work: ^f32, iwork: ^i32) -> i32 ---
	LAPACKE_dsyrfsx_work :: proc(matrix_layout: c.int, uplo: c.char, equed: c.char, n: i32, nrhs: i32, a: ^f64, lda: i32, af: ^f64, ldaf: i32, ipiv: ^i32, s: ^f64, b: ^f64, ldb: i32, x: ^f64, ldx: i32, rcond: ^f64, berr: ^f64, n_err_bnds: i32, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: i32, params: ^f64, work: ^f64, iwork: ^i32) -> i32 ---
	LAPACKE_csyrfsx_work :: proc(matrix_layout: c.int, uplo: c.char, equed: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, af: ^complex64, ldaf: i32, ipiv: ^i32, s: ^f32, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, rcond: ^f32, berr: ^f32, n_err_bnds: i32, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: i32, params: ^f32, work: ^complex64, rwork: ^f32) -> i32 ---
	LAPACKE_zsyrfsx_work :: proc(matrix_layout: c.int, uplo: c.char, equed: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, af: ^complex128, ldaf: i32, ipiv: ^i32, s: ^f64, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, rcond: ^f64, berr: ^f64, n_err_bnds: i32, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: i32, params: ^f64, work: ^complex128, rwork: ^f64) -> i32 ---
	LAPACKE_ssysv_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^f32, lda: i32, ipiv: ^i32, b: ^f32, ldb: i32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_dsysv_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^f64, lda: i32, ipiv: ^i32, b: ^f64, ldb: i32, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_csysv_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, ipiv: ^i32, b: ^complex64, ldb: i32, work: ^complex64, lwork: i32) -> i32 ---
	LAPACKE_zsysv_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, ipiv: ^i32, b: ^complex128, ldb: i32, work: ^complex128, lwork: i32) -> i32 ---
	LAPACKE_ssysvx_work :: proc(matrix_layout: c.int, fact: c.char, uplo: c.char, n: i32, nrhs: i32, a: ^f32, lda: i32, af: ^f32, ldaf: i32, ipiv: ^i32, b: ^f32, ldb: i32, x: ^f32, ldx: i32, rcond: ^f32, ferr: ^f32, berr: ^f32, work: ^f32, lwork: i32, iwork: ^i32) -> i32 ---
	LAPACKE_dsysvx_work :: proc(matrix_layout: c.int, fact: c.char, uplo: c.char, n: i32, nrhs: i32, a: ^f64, lda: i32, af: ^f64, ldaf: i32, ipiv: ^i32, b: ^f64, ldb: i32, x: ^f64, ldx: i32, rcond: ^f64, ferr: ^f64, berr: ^f64, work: ^f64, lwork: i32, iwork: ^i32) -> i32 ---
	LAPACKE_csysvx_work :: proc(matrix_layout: c.int, fact: c.char, uplo: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, af: ^complex64, ldaf: i32, ipiv: ^i32, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, rcond: ^f32, ferr: ^f32, berr: ^f32, work: ^complex64, lwork: i32, rwork: ^f32) -> i32 ---
	LAPACKE_zsysvx_work :: proc(matrix_layout: c.int, fact: c.char, uplo: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, af: ^complex128, ldaf: i32, ipiv: ^i32, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, rcond: ^f64, ferr: ^f64, berr: ^f64, work: ^complex128, lwork: i32, rwork: ^f64) -> i32 ---
	LAPACKE_ssysvxx_work :: proc(matrix_layout: c.int, fact: c.char, uplo: c.char, n: i32, nrhs: i32, a: ^f32, lda: i32, af: ^f32, ldaf: i32, ipiv: ^i32, equed: cstring, s: ^f32, b: ^f32, ldb: i32, x: ^f32, ldx: i32, rcond: ^f32, rpvgrw: ^f32, berr: ^f32, n_err_bnds: i32, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: i32, params: ^f32, work: ^f32, iwork: ^i32) -> i32 ---
	LAPACKE_dsysvxx_work :: proc(matrix_layout: c.int, fact: c.char, uplo: c.char, n: i32, nrhs: i32, a: ^f64, lda: i32, af: ^f64, ldaf: i32, ipiv: ^i32, equed: cstring, s: ^f64, b: ^f64, ldb: i32, x: ^f64, ldx: i32, rcond: ^f64, rpvgrw: ^f64, berr: ^f64, n_err_bnds: i32, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: i32, params: ^f64, work: ^f64, iwork: ^i32) -> i32 ---
	LAPACKE_csysvxx_work :: proc(matrix_layout: c.int, fact: c.char, uplo: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, af: ^complex64, ldaf: i32, ipiv: ^i32, equed: cstring, s: ^f32, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, rcond: ^f32, rpvgrw: ^f32, berr: ^f32, n_err_bnds: i32, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: i32, params: ^f32, work: ^complex64, rwork: ^f32) -> i32 ---
	LAPACKE_zsysvxx_work :: proc(matrix_layout: c.int, fact: c.char, uplo: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, af: ^complex128, ldaf: i32, ipiv: ^i32, equed: cstring, s: ^f64, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, rcond: ^f64, rpvgrw: ^f64, berr: ^f64, n_err_bnds: i32, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: i32, params: ^f64, work: ^complex128, rwork: ^f64) -> i32 ---
	LAPACKE_ssytrd_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f32, lda: i32, d: ^f32, e: ^f32, tau: ^f32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_dsytrd_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f64, lda: i32, d: ^f64, e: ^f64, tau: ^f64, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_ssytrf_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f32, lda: i32, ipiv: ^i32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_dsytrf_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f64, lda: i32, ipiv: ^i32, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_csytrf_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32, ipiv: ^i32, work: ^complex64, lwork: i32) -> i32 ---
	LAPACKE_zsytrf_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32, ipiv: ^i32, work: ^complex128, lwork: i32) -> i32 ---
	LAPACKE_ssytri_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f32, lda: i32, ipiv: ^i32, work: ^f32) -> i32 ---
	LAPACKE_dsytri_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f64, lda: i32, ipiv: ^i32, work: ^f64) -> i32 ---
	LAPACKE_csytri_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32, ipiv: ^i32, work: ^complex64) -> i32 ---
	LAPACKE_zsytri_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32, ipiv: ^i32, work: ^complex128) -> i32 ---
	LAPACKE_ssytrs_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^f32, lda: i32, ipiv: ^i32, b: ^f32, ldb: i32) -> i32 ---
	LAPACKE_dsytrs_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^f64, lda: i32, ipiv: ^i32, b: ^f64, ldb: i32) -> i32 ---
	LAPACKE_csytrs_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, ipiv: ^i32, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_zsytrs_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, ipiv: ^i32, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_stbcon_work :: proc(matrix_layout: c.int, norm: c.char, uplo: c.char, diag: c.char, n: i32, kd: i32, ab: ^f32, ldab: i32, rcond: ^f32, work: ^f32, iwork: ^i32) -> i32 ---
	LAPACKE_dtbcon_work :: proc(matrix_layout: c.int, norm: c.char, uplo: c.char, diag: c.char, n: i32, kd: i32, ab: ^f64, ldab: i32, rcond: ^f64, work: ^f64, iwork: ^i32) -> i32 ---
	LAPACKE_ctbcon_work :: proc(matrix_layout: c.int, norm: c.char, uplo: c.char, diag: c.char, n: i32, kd: i32, ab: ^complex64, ldab: i32, rcond: ^f32, work: ^complex64, rwork: ^f32) -> i32 ---
	LAPACKE_ztbcon_work :: proc(matrix_layout: c.int, norm: c.char, uplo: c.char, diag: c.char, n: i32, kd: i32, ab: ^complex128, ldab: i32, rcond: ^f64, work: ^complex128, rwork: ^f64) -> i32 ---
	LAPACKE_stbrfs_work :: proc(matrix_layout: c.int, uplo: c.char, trans: c.char, diag: c.char, n: i32, kd: i32, nrhs: i32, ab: ^f32, ldab: i32, b: ^f32, ldb: i32, x: ^f32, ldx: i32, ferr: ^f32, berr: ^f32, work: ^f32, iwork: ^i32) -> i32 ---
	LAPACKE_dtbrfs_work :: proc(matrix_layout: c.int, uplo: c.char, trans: c.char, diag: c.char, n: i32, kd: i32, nrhs: i32, ab: ^f64, ldab: i32, b: ^f64, ldb: i32, x: ^f64, ldx: i32, ferr: ^f64, berr: ^f64, work: ^f64, iwork: ^i32) -> i32 ---
	LAPACKE_ctbrfs_work :: proc(matrix_layout: c.int, uplo: c.char, trans: c.char, diag: c.char, n: i32, kd: i32, nrhs: i32, ab: ^complex64, ldab: i32, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, ferr: ^f32, berr: ^f32, work: ^complex64, rwork: ^f32) -> i32 ---
	LAPACKE_ztbrfs_work :: proc(matrix_layout: c.int, uplo: c.char, trans: c.char, diag: c.char, n: i32, kd: i32, nrhs: i32, ab: ^complex128, ldab: i32, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, ferr: ^f64, berr: ^f64, work: ^complex128, rwork: ^f64) -> i32 ---
	LAPACKE_stbtrs_work :: proc(matrix_layout: c.int, uplo: c.char, trans: c.char, diag: c.char, n: i32, kd: i32, nrhs: i32, ab: ^f32, ldab: i32, b: ^f32, ldb: i32) -> i32 ---
	LAPACKE_dtbtrs_work :: proc(matrix_layout: c.int, uplo: c.char, trans: c.char, diag: c.char, n: i32, kd: i32, nrhs: i32, ab: ^f64, ldab: i32, b: ^f64, ldb: i32) -> i32 ---
	LAPACKE_ctbtrs_work :: proc(matrix_layout: c.int, uplo: c.char, trans: c.char, diag: c.char, n: i32, kd: i32, nrhs: i32, ab: ^complex64, ldab: i32, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_ztbtrs_work :: proc(matrix_layout: c.int, uplo: c.char, trans: c.char, diag: c.char, n: i32, kd: i32, nrhs: i32, ab: ^complex128, ldab: i32, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_stfsm_work :: proc(matrix_layout: c.int, transr: c.char, side: c.char, uplo: c.char, trans: c.char, diag: c.char, m: i32, n: i32, alpha: f32, a: ^f32, b: ^f32, ldb: i32) -> i32 ---
	LAPACKE_dtfsm_work :: proc(matrix_layout: c.int, transr: c.char, side: c.char, uplo: c.char, trans: c.char, diag: c.char, m: i32, n: i32, alpha: f64, a: ^f64, b: ^f64, ldb: i32) -> i32 ---
	LAPACKE_ctfsm_work :: proc(matrix_layout: c.int, transr: c.char, side: c.char, uplo: c.char, trans: c.char, diag: c.char, m: i32, n: i32, alpha: complex64, a: ^complex64, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_ztfsm_work :: proc(matrix_layout: c.int, transr: c.char, side: c.char, uplo: c.char, trans: c.char, diag: c.char, m: i32, n: i32, alpha: complex128, a: ^complex128, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_stftri_work :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, diag: c.char, n: i32, a: ^f32) -> i32 ---
	LAPACKE_dtftri_work :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, diag: c.char, n: i32, a: ^f64) -> i32 ---
	LAPACKE_ctftri_work :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, diag: c.char, n: i32, a: ^complex64) -> i32 ---
	LAPACKE_ztftri_work :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, diag: c.char, n: i32, a: ^complex128) -> i32 ---
	LAPACKE_stfttp_work :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, n: i32, arf: ^f32, ap: ^f32) -> i32 ---
	LAPACKE_dtfttp_work :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, n: i32, arf: ^f64, ap: ^f64) -> i32 ---
	LAPACKE_ctfttp_work :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, n: i32, arf: ^complex64, ap: ^complex64) -> i32 ---
	LAPACKE_ztfttp_work :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, n: i32, arf: ^complex128, ap: ^complex128) -> i32 ---
	LAPACKE_stfttr_work :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, n: i32, arf: ^f32, a: ^f32, lda: i32) -> i32 ---
	LAPACKE_dtfttr_work :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, n: i32, arf: ^f64, a: ^f64, lda: i32) -> i32 ---
	LAPACKE_ctfttr_work :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, n: i32, arf: ^complex64, a: ^complex64, lda: i32) -> i32 ---
	LAPACKE_ztfttr_work :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, n: i32, arf: ^complex128, a: ^complex128, lda: i32) -> i32 ---
	LAPACKE_stgevc_work :: proc(matrix_layout: c.int, side: c.char, howmny: c.char, select: ^i32, n: i32, s: ^f32, lds: i32, p: ^f32, ldp: i32, vl: ^f32, ldvl: i32, vr: ^f32, ldvr: i32, mm: i32, m: ^i32, work: ^f32) -> i32 ---
	LAPACKE_dtgevc_work :: proc(matrix_layout: c.int, side: c.char, howmny: c.char, select: ^i32, n: i32, s: ^f64, lds: i32, p: ^f64, ldp: i32, vl: ^f64, ldvl: i32, vr: ^f64, ldvr: i32, mm: i32, m: ^i32, work: ^f64) -> i32 ---
	LAPACKE_ctgevc_work :: proc(matrix_layout: c.int, side: c.char, howmny: c.char, select: ^i32, n: i32, s: ^complex64, lds: i32, p: ^complex64, ldp: i32, vl: ^complex64, ldvl: i32, vr: ^complex64, ldvr: i32, mm: i32, m: ^i32, work: ^complex64, rwork: ^f32) -> i32 ---
	LAPACKE_ztgevc_work :: proc(matrix_layout: c.int, side: c.char, howmny: c.char, select: ^i32, n: i32, s: ^complex128, lds: i32, p: ^complex128, ldp: i32, vl: ^complex128, ldvl: i32, vr: ^complex128, ldvr: i32, mm: i32, m: ^i32, work: ^complex128, rwork: ^f64) -> i32 ---
	LAPACKE_stgexc_work :: proc(matrix_layout: c.int, wantq: i32, wantz: i32, n: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32, q: ^f32, ldq: i32, z: ^f32, ldz: i32, ifst: ^i32, ilst: ^i32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_dtgexc_work :: proc(matrix_layout: c.int, wantq: i32, wantz: i32, n: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, q: ^f64, ldq: i32, z: ^f64, ldz: i32, ifst: ^i32, ilst: ^i32, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_ctgexc_work :: proc(matrix_layout: c.int, wantq: i32, wantz: i32, n: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32, q: ^complex64, ldq: i32, z: ^complex64, ldz: i32, ifst: i32, ilst: i32) -> i32 ---
	LAPACKE_ztgexc_work :: proc(matrix_layout: c.int, wantq: i32, wantz: i32, n: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, q: ^complex128, ldq: i32, z: ^complex128, ldz: i32, ifst: i32, ilst: i32) -> i32 ---
	LAPACKE_stgsen_work :: proc(matrix_layout: c.int, ijob: i32, wantq: i32, wantz: i32, select: ^i32, n: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32, alphar: ^f32, alphai: ^f32, beta: ^f32, q: ^f32, ldq: i32, z: ^f32, ldz: i32, m: ^i32, pl: ^f32, pr: ^f32, dif: ^f32, work: ^f32, lwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_dtgsen_work :: proc(matrix_layout: c.int, ijob: i32, wantq: i32, wantz: i32, select: ^i32, n: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, alphar: ^f64, alphai: ^f64, beta: ^f64, q: ^f64, ldq: i32, z: ^f64, ldz: i32, m: ^i32, pl: ^f64, pr: ^f64, dif: ^f64, work: ^f64, lwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_ctgsen_work :: proc(matrix_layout: c.int, ijob: i32, wantq: i32, wantz: i32, select: ^i32, n: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32, alpha: ^complex64, beta: ^complex64, q: ^complex64, ldq: i32, z: ^complex64, ldz: i32, m: ^i32, pl: ^f32, pr: ^f32, dif: ^f32, work: ^complex64, lwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_ztgsen_work :: proc(matrix_layout: c.int, ijob: i32, wantq: i32, wantz: i32, select: ^i32, n: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, alpha: ^complex128, beta: ^complex128, q: ^complex128, ldq: i32, z: ^complex128, ldz: i32, m: ^i32, pl: ^f64, pr: ^f64, dif: ^f64, work: ^complex128, lwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_stgsja_work :: proc(matrix_layout: c.int, jobu: c.char, jobv: c.char, jobq: c.char, m: i32, p: i32, n: i32, k: i32, l: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32, tola: f32, tolb: f32, alpha: ^f32, beta: ^f32, u: ^f32, ldu: i32, v: ^f32, ldv: i32, q: ^f32, ldq: i32, work: ^f32, ncycle: ^i32) -> i32 ---
	LAPACKE_dtgsja_work :: proc(matrix_layout: c.int, jobu: c.char, jobv: c.char, jobq: c.char, m: i32, p: i32, n: i32, k: i32, l: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, tola: f64, tolb: f64, alpha: ^f64, beta: ^f64, u: ^f64, ldu: i32, v: ^f64, ldv: i32, q: ^f64, ldq: i32, work: ^f64, ncycle: ^i32) -> i32 ---
	LAPACKE_ctgsja_work :: proc(matrix_layout: c.int, jobu: c.char, jobv: c.char, jobq: c.char, m: i32, p: i32, n: i32, k: i32, l: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32, tola: f32, tolb: f32, alpha: ^f32, beta: ^f32, u: ^complex64, ldu: i32, v: ^complex64, ldv: i32, q: ^complex64, ldq: i32, work: ^complex64, ncycle: ^i32) -> i32 ---
	LAPACKE_ztgsja_work :: proc(matrix_layout: c.int, jobu: c.char, jobv: c.char, jobq: c.char, m: i32, p: i32, n: i32, k: i32, l: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, tola: f64, tolb: f64, alpha: ^f64, beta: ^f64, u: ^complex128, ldu: i32, v: ^complex128, ldv: i32, q: ^complex128, ldq: i32, work: ^complex128, ncycle: ^i32) -> i32 ---
	LAPACKE_stgsna_work :: proc(matrix_layout: c.int, job: c.char, howmny: c.char, select: ^i32, n: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32, vl: ^f32, ldvl: i32, vr: ^f32, ldvr: i32, s: ^f32, dif: ^f32, mm: i32, m: ^i32, work: ^f32, lwork: i32, iwork: ^i32) -> i32 ---
	LAPACKE_dtgsna_work :: proc(matrix_layout: c.int, job: c.char, howmny: c.char, select: ^i32, n: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, vl: ^f64, ldvl: i32, vr: ^f64, ldvr: i32, s: ^f64, dif: ^f64, mm: i32, m: ^i32, work: ^f64, lwork: i32, iwork: ^i32) -> i32 ---
	LAPACKE_ctgsna_work :: proc(matrix_layout: c.int, job: c.char, howmny: c.char, select: ^i32, n: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32, vl: ^complex64, ldvl: i32, vr: ^complex64, ldvr: i32, s: ^f32, dif: ^f32, mm: i32, m: ^i32, work: ^complex64, lwork: i32, iwork: ^i32) -> i32 ---
	LAPACKE_ztgsna_work :: proc(matrix_layout: c.int, job: c.char, howmny: c.char, select: ^i32, n: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, vl: ^complex128, ldvl: i32, vr: ^complex128, ldvr: i32, s: ^f64, dif: ^f64, mm: i32, m: ^i32, work: ^complex128, lwork: i32, iwork: ^i32) -> i32 ---
	LAPACKE_stgsyl_work :: proc(matrix_layout: c.int, trans: c.char, ijob: i32, m: i32, n: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32, _c: ^f32, ldc: i32, d: ^f32, ldd: i32, e: ^f32, lde: i32, f: ^f32, ldf: i32, scale: ^f32, dif: ^f32, work: ^f32, lwork: i32, iwork: ^i32) -> i32 ---
	LAPACKE_dtgsyl_work :: proc(matrix_layout: c.int, trans: c.char, ijob: i32, m: i32, n: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, _c: ^f64, ldc: i32, d: ^f64, ldd: i32, e: ^f64, lde: i32, f: ^f64, ldf: i32, scale: ^f64, dif: ^f64, work: ^f64, lwork: i32, iwork: ^i32) -> i32 ---
	LAPACKE_ctgsyl_work :: proc(matrix_layout: c.int, trans: c.char, ijob: i32, m: i32, n: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32, _c: ^complex64, ldc: i32, d: ^complex64, ldd: i32, e: ^complex64, lde: i32, f: ^complex64, ldf: i32, scale: ^f32, dif: ^f32, work: ^complex64, lwork: i32, iwork: ^i32) -> i32 ---
	LAPACKE_ztgsyl_work :: proc(matrix_layout: c.int, trans: c.char, ijob: i32, m: i32, n: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, _c: ^complex128, ldc: i32, d: ^complex128, ldd: i32, e: ^complex128, lde: i32, f: ^complex128, ldf: i32, scale: ^f64, dif: ^f64, work: ^complex128, lwork: i32, iwork: ^i32) -> i32 ---
	LAPACKE_stpcon_work :: proc(matrix_layout: c.int, norm: c.char, uplo: c.char, diag: c.char, n: i32, ap: ^f32, rcond: ^f32, work: ^f32, iwork: ^i32) -> i32 ---
	LAPACKE_dtpcon_work :: proc(matrix_layout: c.int, norm: c.char, uplo: c.char, diag: c.char, n: i32, ap: ^f64, rcond: ^f64, work: ^f64, iwork: ^i32) -> i32 ---
	LAPACKE_ctpcon_work :: proc(matrix_layout: c.int, norm: c.char, uplo: c.char, diag: c.char, n: i32, ap: ^complex64, rcond: ^f32, work: ^complex64, rwork: ^f32) -> i32 ---
	LAPACKE_ztpcon_work :: proc(matrix_layout: c.int, norm: c.char, uplo: c.char, diag: c.char, n: i32, ap: ^complex128, rcond: ^f64, work: ^complex128, rwork: ^f64) -> i32 ---
	LAPACKE_stprfs_work :: proc(matrix_layout: c.int, uplo: c.char, trans: c.char, diag: c.char, n: i32, nrhs: i32, ap: ^f32, b: ^f32, ldb: i32, x: ^f32, ldx: i32, ferr: ^f32, berr: ^f32, work: ^f32, iwork: ^i32) -> i32 ---
	LAPACKE_dtprfs_work :: proc(matrix_layout: c.int, uplo: c.char, trans: c.char, diag: c.char, n: i32, nrhs: i32, ap: ^f64, b: ^f64, ldb: i32, x: ^f64, ldx: i32, ferr: ^f64, berr: ^f64, work: ^f64, iwork: ^i32) -> i32 ---
	LAPACKE_ctprfs_work :: proc(matrix_layout: c.int, uplo: c.char, trans: c.char, diag: c.char, n: i32, nrhs: i32, ap: ^complex64, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, ferr: ^f32, berr: ^f32, work: ^complex64, rwork: ^f32) -> i32 ---
	LAPACKE_ztprfs_work :: proc(matrix_layout: c.int, uplo: c.char, trans: c.char, diag: c.char, n: i32, nrhs: i32, ap: ^complex128, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, ferr: ^f64, berr: ^f64, work: ^complex128, rwork: ^f64) -> i32 ---
	LAPACKE_stptri_work :: proc(matrix_layout: c.int, uplo: c.char, diag: c.char, n: i32, ap: ^f32) -> i32 ---
	LAPACKE_dtptri_work :: proc(matrix_layout: c.int, uplo: c.char, diag: c.char, n: i32, ap: ^f64) -> i32 ---
	LAPACKE_ctptri_work :: proc(matrix_layout: c.int, uplo: c.char, diag: c.char, n: i32, ap: ^complex64) -> i32 ---
	LAPACKE_ztptri_work :: proc(matrix_layout: c.int, uplo: c.char, diag: c.char, n: i32, ap: ^complex128) -> i32 ---
	LAPACKE_stptrs_work :: proc(matrix_layout: c.int, uplo: c.char, trans: c.char, diag: c.char, n: i32, nrhs: i32, ap: ^f32, b: ^f32, ldb: i32) -> i32 ---
	LAPACKE_dtptrs_work :: proc(matrix_layout: c.int, uplo: c.char, trans: c.char, diag: c.char, n: i32, nrhs: i32, ap: ^f64, b: ^f64, ldb: i32) -> i32 ---
	LAPACKE_ctptrs_work :: proc(matrix_layout: c.int, uplo: c.char, trans: c.char, diag: c.char, n: i32, nrhs: i32, ap: ^complex64, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_ztptrs_work :: proc(matrix_layout: c.int, uplo: c.char, trans: c.char, diag: c.char, n: i32, nrhs: i32, ap: ^complex128, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_stpttf_work :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, n: i32, ap: ^f32, arf: ^f32) -> i32 ---
	LAPACKE_dtpttf_work :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, n: i32, ap: ^f64, arf: ^f64) -> i32 ---
	LAPACKE_ctpttf_work :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, n: i32, ap: ^complex64, arf: ^complex64) -> i32 ---
	LAPACKE_ztpttf_work :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, n: i32, ap: ^complex128, arf: ^complex128) -> i32 ---
	LAPACKE_stpttr_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^f32, a: ^f32, lda: i32) -> i32 ---
	LAPACKE_dtpttr_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^f64, a: ^f64, lda: i32) -> i32 ---
	LAPACKE_ctpttr_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^complex64, a: ^complex64, lda: i32) -> i32 ---
	LAPACKE_ztpttr_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^complex128, a: ^complex128, lda: i32) -> i32 ---
	LAPACKE_strcon_work :: proc(matrix_layout: c.int, norm: c.char, uplo: c.char, diag: c.char, n: i32, a: ^f32, lda: i32, rcond: ^f32, work: ^f32, iwork: ^i32) -> i32 ---
	LAPACKE_dtrcon_work :: proc(matrix_layout: c.int, norm: c.char, uplo: c.char, diag: c.char, n: i32, a: ^f64, lda: i32, rcond: ^f64, work: ^f64, iwork: ^i32) -> i32 ---
	LAPACKE_ctrcon_work :: proc(matrix_layout: c.int, norm: c.char, uplo: c.char, diag: c.char, n: i32, a: ^complex64, lda: i32, rcond: ^f32, work: ^complex64, rwork: ^f32) -> i32 ---
	LAPACKE_ztrcon_work :: proc(matrix_layout: c.int, norm: c.char, uplo: c.char, diag: c.char, n: i32, a: ^complex128, lda: i32, rcond: ^f64, work: ^complex128, rwork: ^f64) -> i32 ---
	LAPACKE_strevc_work :: proc(matrix_layout: c.int, side: c.char, howmny: c.char, select: ^i32, n: i32, t: ^f32, ldt: i32, vl: ^f32, ldvl: i32, vr: ^f32, ldvr: i32, mm: i32, m: ^i32, work: ^f32) -> i32 ---
	LAPACKE_dtrevc_work :: proc(matrix_layout: c.int, side: c.char, howmny: c.char, select: ^i32, n: i32, t: ^f64, ldt: i32, vl: ^f64, ldvl: i32, vr: ^f64, ldvr: i32, mm: i32, m: ^i32, work: ^f64) -> i32 ---
	LAPACKE_ctrevc_work :: proc(matrix_layout: c.int, side: c.char, howmny: c.char, select: ^i32, n: i32, t: ^complex64, ldt: i32, vl: ^complex64, ldvl: i32, vr: ^complex64, ldvr: i32, mm: i32, m: ^i32, work: ^complex64, rwork: ^f32) -> i32 ---
	LAPACKE_ztrevc_work :: proc(matrix_layout: c.int, side: c.char, howmny: c.char, select: ^i32, n: i32, t: ^complex128, ldt: i32, vl: ^complex128, ldvl: i32, vr: ^complex128, ldvr: i32, mm: i32, m: ^i32, work: ^complex128, rwork: ^f64) -> i32 ---
	LAPACKE_strexc_work :: proc(matrix_layout: c.int, compq: c.char, n: i32, t: ^f32, ldt: i32, q: ^f32, ldq: i32, ifst: ^i32, ilst: ^i32, work: ^f32) -> i32 ---
	LAPACKE_dtrexc_work :: proc(matrix_layout: c.int, compq: c.char, n: i32, t: ^f64, ldt: i32, q: ^f64, ldq: i32, ifst: ^i32, ilst: ^i32, work: ^f64) -> i32 ---
	LAPACKE_ctrexc_work :: proc(matrix_layout: c.int, compq: c.char, n: i32, t: ^complex64, ldt: i32, q: ^complex64, ldq: i32, ifst: i32, ilst: i32) -> i32 ---
	LAPACKE_ztrexc_work :: proc(matrix_layout: c.int, compq: c.char, n: i32, t: ^complex128, ldt: i32, q: ^complex128, ldq: i32, ifst: i32, ilst: i32) -> i32 ---
	LAPACKE_strrfs_work :: proc(matrix_layout: c.int, uplo: c.char, trans: c.char, diag: c.char, n: i32, nrhs: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32, x: ^f32, ldx: i32, ferr: ^f32, berr: ^f32, work: ^f32, iwork: ^i32) -> i32 ---
	LAPACKE_dtrrfs_work :: proc(matrix_layout: c.int, uplo: c.char, trans: c.char, diag: c.char, n: i32, nrhs: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, x: ^f64, ldx: i32, ferr: ^f64, berr: ^f64, work: ^f64, iwork: ^i32) -> i32 ---
	LAPACKE_ctrrfs_work :: proc(matrix_layout: c.int, uplo: c.char, trans: c.char, diag: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32, x: ^complex64, ldx: i32, ferr: ^f32, berr: ^f32, work: ^complex64, rwork: ^f32) -> i32 ---
	LAPACKE_ztrrfs_work :: proc(matrix_layout: c.int, uplo: c.char, trans: c.char, diag: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, x: ^complex128, ldx: i32, ferr: ^f64, berr: ^f64, work: ^complex128, rwork: ^f64) -> i32 ---
	LAPACKE_strsen_work :: proc(matrix_layout: c.int, job: c.char, compq: c.char, select: ^i32, n: i32, t: ^f32, ldt: i32, q: ^f32, ldq: i32, wr: ^f32, wi: ^f32, m: ^i32, s: ^f32, sep: ^f32, work: ^f32, lwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_dtrsen_work :: proc(matrix_layout: c.int, job: c.char, compq: c.char, select: ^i32, n: i32, t: ^f64, ldt: i32, q: ^f64, ldq: i32, wr: ^f64, wi: ^f64, m: ^i32, s: ^f64, sep: ^f64, work: ^f64, lwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_ctrsen_work :: proc(matrix_layout: c.int, job: c.char, compq: c.char, select: ^i32, n: i32, t: ^complex64, ldt: i32, q: ^complex64, ldq: i32, w: ^complex64, m: ^i32, s: ^f32, sep: ^f32, work: ^complex64, lwork: i32) -> i32 ---
	LAPACKE_ztrsen_work :: proc(matrix_layout: c.int, job: c.char, compq: c.char, select: ^i32, n: i32, t: ^complex128, ldt: i32, q: ^complex128, ldq: i32, w: ^complex128, m: ^i32, s: ^f64, sep: ^f64, work: ^complex128, lwork: i32) -> i32 ---
	LAPACKE_strsna_work :: proc(matrix_layout: c.int, job: c.char, howmny: c.char, select: ^i32, n: i32, t: ^f32, ldt: i32, vl: ^f32, ldvl: i32, vr: ^f32, ldvr: i32, s: ^f32, sep: ^f32, mm: i32, m: ^i32, work: ^f32, ldwork: i32, iwork: ^i32) -> i32 ---
	LAPACKE_dtrsna_work :: proc(matrix_layout: c.int, job: c.char, howmny: c.char, select: ^i32, n: i32, t: ^f64, ldt: i32, vl: ^f64, ldvl: i32, vr: ^f64, ldvr: i32, s: ^f64, sep: ^f64, mm: i32, m: ^i32, work: ^f64, ldwork: i32, iwork: ^i32) -> i32 ---
	LAPACKE_ctrsna_work :: proc(matrix_layout: c.int, job: c.char, howmny: c.char, select: ^i32, n: i32, t: ^complex64, ldt: i32, vl: ^complex64, ldvl: i32, vr: ^complex64, ldvr: i32, s: ^f32, sep: ^f32, mm: i32, m: ^i32, work: ^complex64, ldwork: i32, rwork: ^f32) -> i32 ---
	LAPACKE_ztrsna_work :: proc(matrix_layout: c.int, job: c.char, howmny: c.char, select: ^i32, n: i32, t: ^complex128, ldt: i32, vl: ^complex128, ldvl: i32, vr: ^complex128, ldvr: i32, s: ^f64, sep: ^f64, mm: i32, m: ^i32, work: ^complex128, ldwork: i32, rwork: ^f64) -> i32 ---
	LAPACKE_strsyl_work :: proc(matrix_layout: c.int, trana: c.char, tranb: c.char, isgn: i32, m: i32, n: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32, _c: ^f32, ldc: i32, scale: ^f32) -> i32 ---
	LAPACKE_dtrsyl_work :: proc(matrix_layout: c.int, trana: c.char, tranb: c.char, isgn: i32, m: i32, n: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, _c: ^f64, ldc: i32, scale: ^f64) -> i32 ---
	LAPACKE_ctrsyl_work :: proc(matrix_layout: c.int, trana: c.char, tranb: c.char, isgn: i32, m: i32, n: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32, _c: ^complex64, ldc: i32, scale: ^f32) -> i32 ---
	LAPACKE_ztrsyl_work :: proc(matrix_layout: c.int, trana: c.char, tranb: c.char, isgn: i32, m: i32, n: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, _c: ^complex128, ldc: i32, scale: ^f64) -> i32 ---
	LAPACKE_strsyl3_work :: proc(matrix_layout: c.int, trana: c.char, tranb: c.char, isgn: i32, m: i32, n: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32, _c: ^f32, ldc: i32, scale: ^f32, iwork: ^i32, liwork: i32, swork: ^f32, ldswork: i32) -> i32 ---
	LAPACKE_dtrsyl3_work :: proc(matrix_layout: c.int, trana: c.char, tranb: c.char, isgn: i32, m: i32, n: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, _c: ^f64, ldc: i32, scale: ^f64, iwork: ^i32, liwork: i32, swork: ^f64, ldswork: i32) -> i32 ---
	LAPACKE_ctrsyl3_work :: proc(matrix_layout: c.int, trana: c.char, tranb: c.char, isgn: i32, m: i32, n: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32, _c: ^complex64, ldc: i32, scale: ^f32, swork: ^f32, ldswork: i32) -> i32 ---
	LAPACKE_ztrsyl3_work :: proc(matrix_layout: c.int, trana: c.char, tranb: c.char, isgn: i32, m: i32, n: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, _c: ^complex128, ldc: i32, scale: ^f64, swork: ^f64, ldswork: i32) -> i32 ---
	LAPACKE_strtri_work :: proc(matrix_layout: c.int, uplo: c.char, diag: c.char, n: i32, a: ^f32, lda: i32) -> i32 ---
	LAPACKE_dtrtri_work :: proc(matrix_layout: c.int, uplo: c.char, diag: c.char, n: i32, a: ^f64, lda: i32) -> i32 ---
	LAPACKE_ctrtri_work :: proc(matrix_layout: c.int, uplo: c.char, diag: c.char, n: i32, a: ^complex64, lda: i32) -> i32 ---
	LAPACKE_ztrtri_work :: proc(matrix_layout: c.int, uplo: c.char, diag: c.char, n: i32, a: ^complex128, lda: i32) -> i32 ---
	LAPACKE_strtrs_work :: proc(matrix_layout: c.int, uplo: c.char, trans: c.char, diag: c.char, n: i32, nrhs: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32) -> i32 ---
	LAPACKE_dtrtrs_work :: proc(matrix_layout: c.int, uplo: c.char, trans: c.char, diag: c.char, n: i32, nrhs: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32) -> i32 ---
	LAPACKE_ctrtrs_work :: proc(matrix_layout: c.int, uplo: c.char, trans: c.char, diag: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_ztrtrs_work :: proc(matrix_layout: c.int, uplo: c.char, trans: c.char, diag: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_strttf_work :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, n: i32, a: ^f32, lda: i32, arf: ^f32) -> i32 ---
	LAPACKE_dtrttf_work :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, n: i32, a: ^f64, lda: i32, arf: ^f64) -> i32 ---
	LAPACKE_ctrttf_work :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, n: i32, a: ^complex64, lda: i32, arf: ^complex64) -> i32 ---
	LAPACKE_ztrttf_work :: proc(matrix_layout: c.int, transr: c.char, uplo: c.char, n: i32, a: ^complex128, lda: i32, arf: ^complex128) -> i32 ---
	LAPACKE_strttp_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f32, lda: i32, ap: ^f32) -> i32 ---
	LAPACKE_dtrttp_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f64, lda: i32, ap: ^f64) -> i32 ---
	LAPACKE_ctrttp_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32, ap: ^complex64) -> i32 ---
	LAPACKE_ztrttp_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32, ap: ^complex128) -> i32 ---
	LAPACKE_stzrzf_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f32, lda: i32, tau: ^f32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_dtzrzf_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f64, lda: i32, tau: ^f64, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_ctzrzf_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex64, lda: i32, tau: ^complex64, work: ^complex64, lwork: i32) -> i32 ---
	LAPACKE_ztzrzf_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex128, lda: i32, tau: ^complex128, work: ^complex128, lwork: i32) -> i32 ---
	LAPACKE_cungbr_work :: proc(matrix_layout: c.int, vect: c.char, m: i32, n: i32, k: i32, a: ^complex64, lda: i32, tau: ^complex64, work: ^complex64, lwork: i32) -> i32 ---
	LAPACKE_zungbr_work :: proc(matrix_layout: c.int, vect: c.char, m: i32, n: i32, k: i32, a: ^complex128, lda: i32, tau: ^complex128, work: ^complex128, lwork: i32) -> i32 ---
	LAPACKE_cunghr_work :: proc(matrix_layout: c.int, n: i32, ilo: i32, ihi: i32, a: ^complex64, lda: i32, tau: ^complex64, work: ^complex64, lwork: i32) -> i32 ---
	LAPACKE_zunghr_work :: proc(matrix_layout: c.int, n: i32, ilo: i32, ihi: i32, a: ^complex128, lda: i32, tau: ^complex128, work: ^complex128, lwork: i32) -> i32 ---
	LAPACKE_cunglq_work :: proc(matrix_layout: c.int, m: i32, n: i32, k: i32, a: ^complex64, lda: i32, tau: ^complex64, work: ^complex64, lwork: i32) -> i32 ---
	LAPACKE_zunglq_work :: proc(matrix_layout: c.int, m: i32, n: i32, k: i32, a: ^complex128, lda: i32, tau: ^complex128, work: ^complex128, lwork: i32) -> i32 ---
	LAPACKE_cungql_work :: proc(matrix_layout: c.int, m: i32, n: i32, k: i32, a: ^complex64, lda: i32, tau: ^complex64, work: ^complex64, lwork: i32) -> i32 ---
	LAPACKE_zungql_work :: proc(matrix_layout: c.int, m: i32, n: i32, k: i32, a: ^complex128, lda: i32, tau: ^complex128, work: ^complex128, lwork: i32) -> i32 ---
	LAPACKE_cungqr_work :: proc(matrix_layout: c.int, m: i32, n: i32, k: i32, a: ^complex64, lda: i32, tau: ^complex64, work: ^complex64, lwork: i32) -> i32 ---
	LAPACKE_zungqr_work :: proc(matrix_layout: c.int, m: i32, n: i32, k: i32, a: ^complex128, lda: i32, tau: ^complex128, work: ^complex128, lwork: i32) -> i32 ---
	LAPACKE_cungrq_work :: proc(matrix_layout: c.int, m: i32, n: i32, k: i32, a: ^complex64, lda: i32, tau: ^complex64, work: ^complex64, lwork: i32) -> i32 ---
	LAPACKE_zungrq_work :: proc(matrix_layout: c.int, m: i32, n: i32, k: i32, a: ^complex128, lda: i32, tau: ^complex128, work: ^complex128, lwork: i32) -> i32 ---
	LAPACKE_cungtr_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32, tau: ^complex64, work: ^complex64, lwork: i32) -> i32 ---
	LAPACKE_zungtr_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32, tau: ^complex128, work: ^complex128, lwork: i32) -> i32 ---
	LAPACKE_cungtsqr_row_work :: proc(matrix_layout: c.int, m: i32, n: i32, mb: i32, nb: i32, a: ^complex64, lda: i32, t: ^complex64, ldt: i32, work: ^complex64, lwork: i32) -> i32 ---
	LAPACKE_zungtsqr_row_work :: proc(matrix_layout: c.int, m: i32, n: i32, mb: i32, nb: i32, a: ^complex128, lda: i32, t: ^complex128, ldt: i32, work: ^complex128, lwork: i32) -> i32 ---
	LAPACKE_cunmbr_work :: proc(matrix_layout: c.int, vect: c.char, side: c.char, trans: c.char, m: i32, n: i32, k: i32, a: ^complex64, lda: i32, tau: ^complex64, _c: ^complex64, ldc: i32, work: ^complex64, lwork: i32) -> i32 ---
	LAPACKE_zunmbr_work :: proc(matrix_layout: c.int, vect: c.char, side: c.char, trans: c.char, m: i32, n: i32, k: i32, a: ^complex128, lda: i32, tau: ^complex128, _c: ^complex128, ldc: i32, work: ^complex128, lwork: i32) -> i32 ---
	LAPACKE_cunmhr_work :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, ilo: i32, ihi: i32, a: ^complex64, lda: i32, tau: ^complex64, _c: ^complex64, ldc: i32, work: ^complex64, lwork: i32) -> i32 ---
	LAPACKE_zunmhr_work :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, ilo: i32, ihi: i32, a: ^complex128, lda: i32, tau: ^complex128, _c: ^complex128, ldc: i32, work: ^complex128, lwork: i32) -> i32 ---
	LAPACKE_cunmlq_work :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, a: ^complex64, lda: i32, tau: ^complex64, _c: ^complex64, ldc: i32, work: ^complex64, lwork: i32) -> i32 ---
	LAPACKE_zunmlq_work :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, a: ^complex128, lda: i32, tau: ^complex128, _c: ^complex128, ldc: i32, work: ^complex128, lwork: i32) -> i32 ---
	LAPACKE_cunmql_work :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, a: ^complex64, lda: i32, tau: ^complex64, _c: ^complex64, ldc: i32, work: ^complex64, lwork: i32) -> i32 ---
	LAPACKE_zunmql_work :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, a: ^complex128, lda: i32, tau: ^complex128, _c: ^complex128, ldc: i32, work: ^complex128, lwork: i32) -> i32 ---
	LAPACKE_cunmqr_work :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, a: ^complex64, lda: i32, tau: ^complex64, _c: ^complex64, ldc: i32, work: ^complex64, lwork: i32) -> i32 ---
	LAPACKE_zunmqr_work :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, a: ^complex128, lda: i32, tau: ^complex128, _c: ^complex128, ldc: i32, work: ^complex128, lwork: i32) -> i32 ---
	LAPACKE_cunmrq_work :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, a: ^complex64, lda: i32, tau: ^complex64, _c: ^complex64, ldc: i32, work: ^complex64, lwork: i32) -> i32 ---
	LAPACKE_zunmrq_work :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, a: ^complex128, lda: i32, tau: ^complex128, _c: ^complex128, ldc: i32, work: ^complex128, lwork: i32) -> i32 ---
	LAPACKE_cunmrz_work :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, l: i32, a: ^complex64, lda: i32, tau: ^complex64, _c: ^complex64, ldc: i32, work: ^complex64, lwork: i32) -> i32 ---
	LAPACKE_zunmrz_work :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, l: i32, a: ^complex128, lda: i32, tau: ^complex128, _c: ^complex128, ldc: i32, work: ^complex128, lwork: i32) -> i32 ---
	LAPACKE_cunmtr_work :: proc(matrix_layout: c.int, side: c.char, uplo: c.char, trans: c.char, m: i32, n: i32, a: ^complex64, lda: i32, tau: ^complex64, _c: ^complex64, ldc: i32, work: ^complex64, lwork: i32) -> i32 ---
	LAPACKE_zunmtr_work :: proc(matrix_layout: c.int, side: c.char, uplo: c.char, trans: c.char, m: i32, n: i32, a: ^complex128, lda: i32, tau: ^complex128, _c: ^complex128, ldc: i32, work: ^complex128, lwork: i32) -> i32 ---
	LAPACKE_cupgtr_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^complex64, tau: ^complex64, q: ^complex64, ldq: i32, work: ^complex64) -> i32 ---
	LAPACKE_zupgtr_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, ap: ^complex128, tau: ^complex128, q: ^complex128, ldq: i32, work: ^complex128) -> i32 ---
	LAPACKE_cupmtr_work :: proc(matrix_layout: c.int, side: c.char, uplo: c.char, trans: c.char, m: i32, n: i32, ap: ^complex64, tau: ^complex64, _c: ^complex64, ldc: i32, work: ^complex64) -> i32 ---
	LAPACKE_zupmtr_work :: proc(matrix_layout: c.int, side: c.char, uplo: c.char, trans: c.char, m: i32, n: i32, ap: ^complex128, tau: ^complex128, _c: ^complex128, ldc: i32, work: ^complex128) -> i32 ---
	LAPACKE_claghe :: proc(matrix_layout: c.int, n: i32, k: i32, d: ^f32, a: ^complex64, lda: i32, iseed: ^i32) -> i32 ---
	LAPACKE_zlaghe :: proc(matrix_layout: c.int, n: i32, k: i32, d: ^f64, a: ^complex128, lda: i32, iseed: ^i32) -> i32 ---
	LAPACKE_slagsy :: proc(matrix_layout: c.int, n: i32, k: i32, d: ^f32, a: ^f32, lda: i32, iseed: ^i32) -> i32 ---
	LAPACKE_dlagsy :: proc(matrix_layout: c.int, n: i32, k: i32, d: ^f64, a: ^f64, lda: i32, iseed: ^i32) -> i32 ---
	LAPACKE_clagsy :: proc(matrix_layout: c.int, n: i32, k: i32, d: ^f32, a: ^complex64, lda: i32, iseed: ^i32) -> i32 ---
	LAPACKE_zlagsy :: proc(matrix_layout: c.int, n: i32, k: i32, d: ^f64, a: ^complex128, lda: i32, iseed: ^i32) -> i32 ---
	LAPACKE_slapmr :: proc(matrix_layout: c.int, forwrd: i32, m: i32, n: i32, x: ^f32, ldx: i32, k: ^i32) -> i32 ---
	LAPACKE_dlapmr :: proc(matrix_layout: c.int, forwrd: i32, m: i32, n: i32, x: ^f64, ldx: i32, k: ^i32) -> i32 ---
	LAPACKE_clapmr :: proc(matrix_layout: c.int, forwrd: i32, m: i32, n: i32, x: ^complex64, ldx: i32, k: ^i32) -> i32 ---
	LAPACKE_zlapmr :: proc(matrix_layout: c.int, forwrd: i32, m: i32, n: i32, x: ^complex128, ldx: i32, k: ^i32) -> i32 ---
	LAPACKE_slapmt :: proc(matrix_layout: c.int, forwrd: i32, m: i32, n: i32, x: ^f32, ldx: i32, k: ^i32) -> i32 ---
	LAPACKE_dlapmt :: proc(matrix_layout: c.int, forwrd: i32, m: i32, n: i32, x: ^f64, ldx: i32, k: ^i32) -> i32 ---
	LAPACKE_clapmt :: proc(matrix_layout: c.int, forwrd: i32, m: i32, n: i32, x: ^complex64, ldx: i32, k: ^i32) -> i32 ---
	LAPACKE_zlapmt :: proc(matrix_layout: c.int, forwrd: i32, m: i32, n: i32, x: ^complex128, ldx: i32, k: ^i32) -> i32 ---
	LAPACKE_slapy2 :: proc(x: f32, y: f32) -> f32 ---
	LAPACKE_dlapy2 :: proc(x: f64, y: f64) -> f64 ---
	LAPACKE_slapy3 :: proc(x: f32, y: f32, z: f32) -> f32 ---
	LAPACKE_dlapy3 :: proc(x: f64, y: f64, z: f64) -> f64 ---
	LAPACKE_slartgp :: proc(f: f32, g: f32, cs: ^f32, sn: ^f32, r: ^f32) -> i32 ---
	LAPACKE_dlartgp :: proc(f: f64, g: f64, cs: ^f64, sn: ^f64, r: ^f64) -> i32 ---
	LAPACKE_slartgs :: proc(x: f32, y: f32, sigma: f32, cs: ^f32, sn: ^f32) -> i32 ---
	LAPACKE_dlartgs :: proc(x: f64, y: f64, sigma: f64, cs: ^f64, sn: ^f64) -> i32 ---

	//LAPACK 3.3.0
	LAPACKE_cbbcsd :: proc(matrix_layout: c.int, jobu1: c.char, jobu2: c.char, jobv1t: c.char, jobv2t: c.char, trans: c.char, m: i32, p: i32, q: i32, theta: ^f32, phi: ^f32, u1: ^complex64, ldu1: i32, u2: ^complex64, ldu2: i32, v1t: ^complex64, ldv1t: i32, v2t: ^complex64, ldv2t: i32, b11d: ^f32, b11e: ^f32, b12d: ^f32, b12e: ^f32, b21d: ^f32, b21e: ^f32, b22d: ^f32, b22e: ^f32) -> i32 ---
	LAPACKE_cbbcsd_work :: proc(matrix_layout: c.int, jobu1: c.char, jobu2: c.char, jobv1t: c.char, jobv2t: c.char, trans: c.char, m: i32, p: i32, q: i32, theta: ^f32, phi: ^f32, u1: ^complex64, ldu1: i32, u2: ^complex64, ldu2: i32, v1t: ^complex64, ldv1t: i32, v2t: ^complex64, ldv2t: i32, b11d: ^f32, b11e: ^f32, b12d: ^f32, b12e: ^f32, b21d: ^f32, b21e: ^f32, b22d: ^f32, b22e: ^f32, rwork: ^f32, lrwork: i32) -> i32 ---
	LAPACKE_cheswapr :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32, i1: i32, i2: i32) -> i32 ---
	LAPACKE_cheswapr_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32, i1: i32, i2: i32) -> i32 ---
	LAPACKE_chetri2 :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32, ipiv: ^i32) -> i32 ---
	LAPACKE_chetri2_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32, ipiv: ^i32, work: ^complex64, lwork: i32) -> i32 ---
	LAPACKE_chetri2x :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32, ipiv: ^i32, nb: i32) -> i32 ---
	LAPACKE_chetri2x_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32, ipiv: ^i32, work: ^complex64, nb: i32) -> i32 ---
	LAPACKE_chetrs2 :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, ipiv: ^i32, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_chetrs2_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, ipiv: ^i32, b: ^complex64, ldb: i32, work: ^complex64) -> i32 ---
	LAPACKE_csyconv :: proc(matrix_layout: c.int, uplo: c.char, way: c.char, n: i32, a: ^complex64, lda: i32, ipiv: ^i32, e: ^complex64) -> i32 ---
	LAPACKE_csyconv_work :: proc(matrix_layout: c.int, uplo: c.char, way: c.char, n: i32, a: ^complex64, lda: i32, ipiv: ^i32, e: ^complex64) -> i32 ---
	LAPACKE_csyswapr :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32, i1: i32, i2: i32) -> i32 ---
	LAPACKE_csyswapr_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32, i1: i32, i2: i32) -> i32 ---
	LAPACKE_csytri2 :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32, ipiv: ^i32) -> i32 ---
	LAPACKE_csytri2_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32, ipiv: ^i32, work: ^complex64, lwork: i32) -> i32 ---
	LAPACKE_csytri2x :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32, ipiv: ^i32, nb: i32) -> i32 ---
	LAPACKE_csytri2x_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32, ipiv: ^i32, work: ^complex64, nb: i32) -> i32 ---
	LAPACKE_csytrs2 :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, ipiv: ^i32, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_csytrs2_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, ipiv: ^i32, b: ^complex64, ldb: i32, work: ^complex64) -> i32 ---
	LAPACKE_cunbdb :: proc(matrix_layout: c.int, trans: c.char, signs: c.char, m: i32, p: i32, q: i32, x11: ^complex64, ldx11: i32, x12: ^complex64, ldx12: i32, x21: ^complex64, ldx21: i32, x22: ^complex64, ldx22: i32, theta: ^f32, phi: ^f32, taup1: ^complex64, taup2: ^complex64, tauq1: ^complex64, tauq2: ^complex64) -> i32 ---
	LAPACKE_cunbdb_work :: proc(matrix_layout: c.int, trans: c.char, signs: c.char, m: i32, p: i32, q: i32, x11: ^complex64, ldx11: i32, x12: ^complex64, ldx12: i32, x21: ^complex64, ldx21: i32, x22: ^complex64, ldx22: i32, theta: ^f32, phi: ^f32, taup1: ^complex64, taup2: ^complex64, tauq1: ^complex64, tauq2: ^complex64, work: ^complex64, lwork: i32) -> i32 ---
	LAPACKE_cuncsd :: proc(matrix_layout: c.int, jobu1: c.char, jobu2: c.char, jobv1t: c.char, jobv2t: c.char, trans: c.char, signs: c.char, m: i32, p: i32, q: i32, x11: ^complex64, ldx11: i32, x12: ^complex64, ldx12: i32, x21: ^complex64, ldx21: i32, x22: ^complex64, ldx22: i32, theta: ^f32, u1: ^complex64, ldu1: i32, u2: ^complex64, ldu2: i32, v1t: ^complex64, ldv1t: i32, v2t: ^complex64, ldv2t: i32) -> i32 ---
	LAPACKE_cuncsd_work :: proc(matrix_layout: c.int, jobu1: c.char, jobu2: c.char, jobv1t: c.char, jobv2t: c.char, trans: c.char, signs: c.char, m: i32, p: i32, q: i32, x11: ^complex64, ldx11: i32, x12: ^complex64, ldx12: i32, x21: ^complex64, ldx21: i32, x22: ^complex64, ldx22: i32, theta: ^f32, u1: ^complex64, ldu1: i32, u2: ^complex64, ldu2: i32, v1t: ^complex64, ldv1t: i32, v2t: ^complex64, ldv2t: i32, work: ^complex64, lwork: i32, rwork: ^f32, lrwork: i32, iwork: ^i32) -> i32 ---
	LAPACKE_cuncsd2by1 :: proc(matrix_layout: c.int, jobu1: c.char, jobu2: c.char, jobv1t: c.char, m: i32, p: i32, q: i32, x11: ^complex64, ldx11: i32, x21: ^complex64, ldx21: i32, theta: ^f32, u1: ^complex64, ldu1: i32, u2: ^complex64, ldu2: i32, v1t: ^complex64, ldv1t: i32) -> i32 ---
	LAPACKE_cuncsd2by1_work :: proc(matrix_layout: c.int, jobu1: c.char, jobu2: c.char, jobv1t: c.char, m: i32, p: i32, q: i32, x11: ^complex64, ldx11: i32, x21: ^complex64, ldx21: i32, theta: ^f32, u1: ^complex64, ldu1: i32, u2: ^complex64, ldu2: i32, v1t: ^complex64, ldv1t: i32, work: ^complex64, lwork: i32, rwork: ^f32, lrwork: i32, iwork: ^i32) -> i32 ---
	LAPACKE_dbbcsd :: proc(matrix_layout: c.int, jobu1: c.char, jobu2: c.char, jobv1t: c.char, jobv2t: c.char, trans: c.char, m: i32, p: i32, q: i32, theta: ^f64, phi: ^f64, u1: ^f64, ldu1: i32, u2: ^f64, ldu2: i32, v1t: ^f64, ldv1t: i32, v2t: ^f64, ldv2t: i32, b11d: ^f64, b11e: ^f64, b12d: ^f64, b12e: ^f64, b21d: ^f64, b21e: ^f64, b22d: ^f64, b22e: ^f64) -> i32 ---
	LAPACKE_dbbcsd_work :: proc(matrix_layout: c.int, jobu1: c.char, jobu2: c.char, jobv1t: c.char, jobv2t: c.char, trans: c.char, m: i32, p: i32, q: i32, theta: ^f64, phi: ^f64, u1: ^f64, ldu1: i32, u2: ^f64, ldu2: i32, v1t: ^f64, ldv1t: i32, v2t: ^f64, ldv2t: i32, b11d: ^f64, b11e: ^f64, b12d: ^f64, b12e: ^f64, b21d: ^f64, b21e: ^f64, b22d: ^f64, b22e: ^f64, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_dorbdb :: proc(matrix_layout: c.int, trans: c.char, signs: c.char, m: i32, p: i32, q: i32, x11: ^f64, ldx11: i32, x12: ^f64, ldx12: i32, x21: ^f64, ldx21: i32, x22: ^f64, ldx22: i32, theta: ^f64, phi: ^f64, taup1: ^f64, taup2: ^f64, tauq1: ^f64, tauq2: ^f64) -> i32 ---
	LAPACKE_dorbdb_work :: proc(matrix_layout: c.int, trans: c.char, signs: c.char, m: i32, p: i32, q: i32, x11: ^f64, ldx11: i32, x12: ^f64, ldx12: i32, x21: ^f64, ldx21: i32, x22: ^f64, ldx22: i32, theta: ^f64, phi: ^f64, taup1: ^f64, taup2: ^f64, tauq1: ^f64, tauq2: ^f64, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_dorcsd :: proc(matrix_layout: c.int, jobu1: c.char, jobu2: c.char, jobv1t: c.char, jobv2t: c.char, trans: c.char, signs: c.char, m: i32, p: i32, q: i32, x11: ^f64, ldx11: i32, x12: ^f64, ldx12: i32, x21: ^f64, ldx21: i32, x22: ^f64, ldx22: i32, theta: ^f64, u1: ^f64, ldu1: i32, u2: ^f64, ldu2: i32, v1t: ^f64, ldv1t: i32, v2t: ^f64, ldv2t: i32) -> i32 ---
	LAPACKE_dorcsd_work :: proc(matrix_layout: c.int, jobu1: c.char, jobu2: c.char, jobv1t: c.char, jobv2t: c.char, trans: c.char, signs: c.char, m: i32, p: i32, q: i32, x11: ^f64, ldx11: i32, x12: ^f64, ldx12: i32, x21: ^f64, ldx21: i32, x22: ^f64, ldx22: i32, theta: ^f64, u1: ^f64, ldu1: i32, u2: ^f64, ldu2: i32, v1t: ^f64, ldv1t: i32, v2t: ^f64, ldv2t: i32, work: ^f64, lwork: i32, iwork: ^i32) -> i32 ---
	LAPACKE_dorcsd2by1 :: proc(matrix_layout: c.int, jobu1: c.char, jobu2: c.char, jobv1t: c.char, m: i32, p: i32, q: i32, x11: ^f64, ldx11: i32, x21: ^f64, ldx21: i32, theta: ^f64, u1: ^f64, ldu1: i32, u2: ^f64, ldu2: i32, v1t: ^f64, ldv1t: i32) -> i32 ---
	LAPACKE_dorcsd2by1_work :: proc(matrix_layout: c.int, jobu1: c.char, jobu2: c.char, jobv1t: c.char, m: i32, p: i32, q: i32, x11: ^f64, ldx11: i32, x21: ^f64, ldx21: i32, theta: ^f64, u1: ^f64, ldu1: i32, u2: ^f64, ldu2: i32, v1t: ^f64, ldv1t: i32, work: ^f64, lwork: i32, iwork: ^i32) -> i32 ---
	LAPACKE_dsyconv :: proc(matrix_layout: c.int, uplo: c.char, way: c.char, n: i32, a: ^f64, lda: i32, ipiv: ^i32, e: ^f64) -> i32 ---
	LAPACKE_dsyconv_work :: proc(matrix_layout: c.int, uplo: c.char, way: c.char, n: i32, a: ^f64, lda: i32, ipiv: ^i32, e: ^f64) -> i32 ---
	LAPACKE_dsyswapr :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f64, lda: i32, i1: i32, i2: i32) -> i32 ---
	LAPACKE_dsyswapr_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f64, lda: i32, i1: i32, i2: i32) -> i32 ---
	LAPACKE_dsytri2 :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f64, lda: i32, ipiv: ^i32) -> i32 ---
	LAPACKE_dsytri2_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f64, lda: i32, ipiv: ^i32, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_dsytri2x :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f64, lda: i32, ipiv: ^i32, nb: i32) -> i32 ---
	LAPACKE_dsytri2x_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f64, lda: i32, ipiv: ^i32, work: ^f64, nb: i32) -> i32 ---
	LAPACKE_dsytrs2 :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^f64, lda: i32, ipiv: ^i32, b: ^f64, ldb: i32) -> i32 ---
	LAPACKE_dsytrs2_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^f64, lda: i32, ipiv: ^i32, b: ^f64, ldb: i32, work: ^f64) -> i32 ---
	LAPACKE_sbbcsd :: proc(matrix_layout: c.int, jobu1: c.char, jobu2: c.char, jobv1t: c.char, jobv2t: c.char, trans: c.char, m: i32, p: i32, q: i32, theta: ^f32, phi: ^f32, u1: ^f32, ldu1: i32, u2: ^f32, ldu2: i32, v1t: ^f32, ldv1t: i32, v2t: ^f32, ldv2t: i32, b11d: ^f32, b11e: ^f32, b12d: ^f32, b12e: ^f32, b21d: ^f32, b21e: ^f32, b22d: ^f32, b22e: ^f32) -> i32 ---
	LAPACKE_sbbcsd_work :: proc(matrix_layout: c.int, jobu1: c.char, jobu2: c.char, jobv1t: c.char, jobv2t: c.char, trans: c.char, m: i32, p: i32, q: i32, theta: ^f32, phi: ^f32, u1: ^f32, ldu1: i32, u2: ^f32, ldu2: i32, v1t: ^f32, ldv1t: i32, v2t: ^f32, ldv2t: i32, b11d: ^f32, b11e: ^f32, b12d: ^f32, b12e: ^f32, b21d: ^f32, b21e: ^f32, b22d: ^f32, b22e: ^f32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_sorbdb :: proc(matrix_layout: c.int, trans: c.char, signs: c.char, m: i32, p: i32, q: i32, x11: ^f32, ldx11: i32, x12: ^f32, ldx12: i32, x21: ^f32, ldx21: i32, x22: ^f32, ldx22: i32, theta: ^f32, phi: ^f32, taup1: ^f32, taup2: ^f32, tauq1: ^f32, tauq2: ^f32) -> i32 ---
	LAPACKE_sorbdb_work :: proc(matrix_layout: c.int, trans: c.char, signs: c.char, m: i32, p: i32, q: i32, x11: ^f32, ldx11: i32, x12: ^f32, ldx12: i32, x21: ^f32, ldx21: i32, x22: ^f32, ldx22: i32, theta: ^f32, phi: ^f32, taup1: ^f32, taup2: ^f32, tauq1: ^f32, tauq2: ^f32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_sorcsd :: proc(matrix_layout: c.int, jobu1: c.char, jobu2: c.char, jobv1t: c.char, jobv2t: c.char, trans: c.char, signs: c.char, m: i32, p: i32, q: i32, x11: ^f32, ldx11: i32, x12: ^f32, ldx12: i32, x21: ^f32, ldx21: i32, x22: ^f32, ldx22: i32, theta: ^f32, u1: ^f32, ldu1: i32, u2: ^f32, ldu2: i32, v1t: ^f32, ldv1t: i32, v2t: ^f32, ldv2t: i32) -> i32 ---
	LAPACKE_sorcsd_work :: proc(matrix_layout: c.int, jobu1: c.char, jobu2: c.char, jobv1t: c.char, jobv2t: c.char, trans: c.char, signs: c.char, m: i32, p: i32, q: i32, x11: ^f32, ldx11: i32, x12: ^f32, ldx12: i32, x21: ^f32, ldx21: i32, x22: ^f32, ldx22: i32, theta: ^f32, u1: ^f32, ldu1: i32, u2: ^f32, ldu2: i32, v1t: ^f32, ldv1t: i32, v2t: ^f32, ldv2t: i32, work: ^f32, lwork: i32, iwork: ^i32) -> i32 ---
	LAPACKE_sorcsd2by1 :: proc(matrix_layout: c.int, jobu1: c.char, jobu2: c.char, jobv1t: c.char, m: i32, p: i32, q: i32, x11: ^f32, ldx11: i32, x21: ^f32, ldx21: i32, theta: ^f32, u1: ^f32, ldu1: i32, u2: ^f32, ldu2: i32, v1t: ^f32, ldv1t: i32) -> i32 ---
	LAPACKE_sorcsd2by1_work :: proc(matrix_layout: c.int, jobu1: c.char, jobu2: c.char, jobv1t: c.char, m: i32, p: i32, q: i32, x11: ^f32, ldx11: i32, x21: ^f32, ldx21: i32, theta: ^f32, u1: ^f32, ldu1: i32, u2: ^f32, ldu2: i32, v1t: ^f32, ldv1t: i32, work: ^f32, lwork: i32, iwork: ^i32) -> i32 ---
	LAPACKE_ssyconv :: proc(matrix_layout: c.int, uplo: c.char, way: c.char, n: i32, a: ^f32, lda: i32, ipiv: ^i32, e: ^f32) -> i32 ---
	LAPACKE_ssyconv_work :: proc(matrix_layout: c.int, uplo: c.char, way: c.char, n: i32, a: ^f32, lda: i32, ipiv: ^i32, e: ^f32) -> i32 ---
	LAPACKE_ssyswapr :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f32, lda: i32, i1: i32, i2: i32) -> i32 ---
	LAPACKE_ssyswapr_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f32, lda: i32, i1: i32, i2: i32) -> i32 ---
	LAPACKE_ssytri2 :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f32, lda: i32, ipiv: ^i32) -> i32 ---
	LAPACKE_ssytri2_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f32, lda: i32, ipiv: ^i32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_ssytri2x :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f32, lda: i32, ipiv: ^i32, nb: i32) -> i32 ---
	LAPACKE_ssytri2x_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f32, lda: i32, ipiv: ^i32, work: ^f32, nb: i32) -> i32 ---
	LAPACKE_ssytrs2 :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^f32, lda: i32, ipiv: ^i32, b: ^f32, ldb: i32) -> i32 ---
	LAPACKE_ssytrs2_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^f32, lda: i32, ipiv: ^i32, b: ^f32, ldb: i32, work: ^f32) -> i32 ---
	LAPACKE_zbbcsd :: proc(matrix_layout: c.int, jobu1: c.char, jobu2: c.char, jobv1t: c.char, jobv2t: c.char, trans: c.char, m: i32, p: i32, q: i32, theta: ^f64, phi: ^f64, u1: ^complex128, ldu1: i32, u2: ^complex128, ldu2: i32, v1t: ^complex128, ldv1t: i32, v2t: ^complex128, ldv2t: i32, b11d: ^f64, b11e: ^f64, b12d: ^f64, b12e: ^f64, b21d: ^f64, b21e: ^f64, b22d: ^f64, b22e: ^f64) -> i32 ---
	LAPACKE_zbbcsd_work :: proc(matrix_layout: c.int, jobu1: c.char, jobu2: c.char, jobv1t: c.char, jobv2t: c.char, trans: c.char, m: i32, p: i32, q: i32, theta: ^f64, phi: ^f64, u1: ^complex128, ldu1: i32, u2: ^complex128, ldu2: i32, v1t: ^complex128, ldv1t: i32, v2t: ^complex128, ldv2t: i32, b11d: ^f64, b11e: ^f64, b12d: ^f64, b12e: ^f64, b21d: ^f64, b21e: ^f64, b22d: ^f64, b22e: ^f64, rwork: ^f64, lrwork: i32) -> i32 ---
	LAPACKE_zheswapr :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32, i1: i32, i2: i32) -> i32 ---
	LAPACKE_zheswapr_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32, i1: i32, i2: i32) -> i32 ---
	LAPACKE_zhetri2 :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32, ipiv: ^i32) -> i32 ---
	LAPACKE_zhetri2_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32, ipiv: ^i32, work: ^complex128, lwork: i32) -> i32 ---
	LAPACKE_zhetri2x :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32, ipiv: ^i32, nb: i32) -> i32 ---
	LAPACKE_zhetri2x_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32, ipiv: ^i32, work: ^complex128, nb: i32) -> i32 ---
	LAPACKE_zhetrs2 :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, ipiv: ^i32, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_zhetrs2_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, ipiv: ^i32, b: ^complex128, ldb: i32, work: ^complex128) -> i32 ---
	LAPACKE_zsyconv :: proc(matrix_layout: c.int, uplo: c.char, way: c.char, n: i32, a: ^complex128, lda: i32, ipiv: ^i32, e: ^complex128) -> i32 ---
	LAPACKE_zsyconv_work :: proc(matrix_layout: c.int, uplo: c.char, way: c.char, n: i32, a: ^complex128, lda: i32, ipiv: ^i32, e: ^complex128) -> i32 ---
	LAPACKE_zsyswapr :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32, i1: i32, i2: i32) -> i32 ---
	LAPACKE_zsyswapr_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32, i1: i32, i2: i32) -> i32 ---
	LAPACKE_zsytri2 :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32, ipiv: ^i32) -> i32 ---
	LAPACKE_zsytri2_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32, ipiv: ^i32, work: ^complex128, lwork: i32) -> i32 ---
	LAPACKE_zsytri2x :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32, ipiv: ^i32, nb: i32) -> i32 ---
	LAPACKE_zsytri2x_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32, ipiv: ^i32, work: ^complex128, nb: i32) -> i32 ---
	LAPACKE_zsytrs2 :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, ipiv: ^i32, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_zsytrs2_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, ipiv: ^i32, b: ^complex128, ldb: i32, work: ^complex128) -> i32 ---
	LAPACKE_zunbdb :: proc(matrix_layout: c.int, trans: c.char, signs: c.char, m: i32, p: i32, q: i32, x11: ^complex128, ldx11: i32, x12: ^complex128, ldx12: i32, x21: ^complex128, ldx21: i32, x22: ^complex128, ldx22: i32, theta: ^f64, phi: ^f64, taup1: ^complex128, taup2: ^complex128, tauq1: ^complex128, tauq2: ^complex128) -> i32 ---
	LAPACKE_zunbdb_work :: proc(matrix_layout: c.int, trans: c.char, signs: c.char, m: i32, p: i32, q: i32, x11: ^complex128, ldx11: i32, x12: ^complex128, ldx12: i32, x21: ^complex128, ldx21: i32, x22: ^complex128, ldx22: i32, theta: ^f64, phi: ^f64, taup1: ^complex128, taup2: ^complex128, tauq1: ^complex128, tauq2: ^complex128, work: ^complex128, lwork: i32) -> i32 ---
	LAPACKE_zuncsd :: proc(matrix_layout: c.int, jobu1: c.char, jobu2: c.char, jobv1t: c.char, jobv2t: c.char, trans: c.char, signs: c.char, m: i32, p: i32, q: i32, x11: ^complex128, ldx11: i32, x12: ^complex128, ldx12: i32, x21: ^complex128, ldx21: i32, x22: ^complex128, ldx22: i32, theta: ^f64, u1: ^complex128, ldu1: i32, u2: ^complex128, ldu2: i32, v1t: ^complex128, ldv1t: i32, v2t: ^complex128, ldv2t: i32) -> i32 ---
	LAPACKE_zuncsd_work :: proc(matrix_layout: c.int, jobu1: c.char, jobu2: c.char, jobv1t: c.char, jobv2t: c.char, trans: c.char, signs: c.char, m: i32, p: i32, q: i32, x11: ^complex128, ldx11: i32, x12: ^complex128, ldx12: i32, x21: ^complex128, ldx21: i32, x22: ^complex128, ldx22: i32, theta: ^f64, u1: ^complex128, ldu1: i32, u2: ^complex128, ldu2: i32, v1t: ^complex128, ldv1t: i32, v2t: ^complex128, ldv2t: i32, work: ^complex128, lwork: i32, rwork: ^f64, lrwork: i32, iwork: ^i32) -> i32 ---
	LAPACKE_zuncsd2by1 :: proc(matrix_layout: c.int, jobu1: c.char, jobu2: c.char, jobv1t: c.char, m: i32, p: i32, q: i32, x11: ^complex128, ldx11: i32, x21: ^complex128, ldx21: i32, theta: ^f64, u1: ^complex128, ldu1: i32, u2: ^complex128, ldu2: i32, v1t: ^complex128, ldv1t: i32) -> i32 ---
	LAPACKE_zuncsd2by1_work :: proc(matrix_layout: c.int, jobu1: c.char, jobu2: c.char, jobv1t: c.char, m: i32, p: i32, q: i32, x11: ^complex128, ldx11: i32, x21: ^complex128, ldx21: i32, theta: ^f64, u1: ^complex128, ldu1: i32, u2: ^complex128, ldu2: i32, v1t: ^complex128, ldv1t: i32, work: ^complex128, lwork: i32, rwork: ^f64, lrwork: i32, iwork: ^i32) -> i32 ---

	//LAPACK 3.4.0
	LAPACKE_sgemqrt :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, nb: i32, v: ^f32, ldv: i32, t: ^f32, ldt: i32, _c: ^f32, ldc: i32) -> i32 ---
	LAPACKE_dgemqrt :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, nb: i32, v: ^f64, ldv: i32, t: ^f64, ldt: i32, _c: ^f64, ldc: i32) -> i32 ---
	LAPACKE_cgemqrt :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, nb: i32, v: ^complex64, ldv: i32, t: ^complex64, ldt: i32, _c: ^complex64, ldc: i32) -> i32 ---
	LAPACKE_zgemqrt :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, nb: i32, v: ^complex128, ldv: i32, t: ^complex128, ldt: i32, _c: ^complex128, ldc: i32) -> i32 ---
	LAPACKE_sgeqrt :: proc(matrix_layout: c.int, m: i32, n: i32, nb: i32, a: ^f32, lda: i32, t: ^f32, ldt: i32) -> i32 ---
	LAPACKE_dgeqrt :: proc(matrix_layout: c.int, m: i32, n: i32, nb: i32, a: ^f64, lda: i32, t: ^f64, ldt: i32) -> i32 ---
	LAPACKE_cgeqrt :: proc(matrix_layout: c.int, m: i32, n: i32, nb: i32, a: ^complex64, lda: i32, t: ^complex64, ldt: i32) -> i32 ---
	LAPACKE_zgeqrt :: proc(matrix_layout: c.int, m: i32, n: i32, nb: i32, a: ^complex128, lda: i32, t: ^complex128, ldt: i32) -> i32 ---
	LAPACKE_sgeqrt2 :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f32, lda: i32, t: ^f32, ldt: i32) -> i32 ---
	LAPACKE_dgeqrt2 :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f64, lda: i32, t: ^f64, ldt: i32) -> i32 ---
	LAPACKE_cgeqrt2 :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex64, lda: i32, t: ^complex64, ldt: i32) -> i32 ---
	LAPACKE_zgeqrt2 :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex128, lda: i32, t: ^complex128, ldt: i32) -> i32 ---
	LAPACKE_sgeqrt3 :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f32, lda: i32, t: ^f32, ldt: i32) -> i32 ---
	LAPACKE_dgeqrt3 :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f64, lda: i32, t: ^f64, ldt: i32) -> i32 ---
	LAPACKE_cgeqrt3 :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex64, lda: i32, t: ^complex64, ldt: i32) -> i32 ---
	LAPACKE_zgeqrt3 :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex128, lda: i32, t: ^complex128, ldt: i32) -> i32 ---
	LAPACKE_stpmqrt :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, l: i32, nb: i32, v: ^f32, ldv: i32, t: ^f32, ldt: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32) -> i32 ---
	LAPACKE_dtpmqrt :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, l: i32, nb: i32, v: ^f64, ldv: i32, t: ^f64, ldt: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32) -> i32 ---
	LAPACKE_ctpmqrt :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, l: i32, nb: i32, v: ^complex64, ldv: i32, t: ^complex64, ldt: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_ztpmqrt :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, l: i32, nb: i32, v: ^complex128, ldv: i32, t: ^complex128, ldt: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_stpqrt :: proc(matrix_layout: c.int, m: i32, n: i32, l: i32, nb: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32, t: ^f32, ldt: i32) -> i32 ---
	LAPACKE_dtpqrt :: proc(matrix_layout: c.int, m: i32, n: i32, l: i32, nb: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, t: ^f64, ldt: i32) -> i32 ---
	LAPACKE_ctpqrt :: proc(matrix_layout: c.int, m: i32, n: i32, l: i32, nb: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32, t: ^complex64, ldt: i32) -> i32 ---
	LAPACKE_ztpqrt :: proc(matrix_layout: c.int, m: i32, n: i32, l: i32, nb: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, t: ^complex128, ldt: i32) -> i32 ---
	LAPACKE_stpqrt2 :: proc(matrix_layout: c.int, m: i32, n: i32, l: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32, t: ^f32, ldt: i32) -> i32 ---
	LAPACKE_dtpqrt2 :: proc(matrix_layout: c.int, m: i32, n: i32, l: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, t: ^f64, ldt: i32) -> i32 ---
	LAPACKE_ctpqrt2 :: proc(matrix_layout: c.int, m: i32, n: i32, l: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32, t: ^complex64, ldt: i32) -> i32 ---
	LAPACKE_ztpqrt2 :: proc(matrix_layout: c.int, m: i32, n: i32, l: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, t: ^complex128, ldt: i32) -> i32 ---
	LAPACKE_stprfb :: proc(matrix_layout: c.int, side: c.char, trans: c.char, direct: c.char, storev: c.char, m: i32, n: i32, k: i32, l: i32, v: ^f32, ldv: i32, t: ^f32, ldt: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32) -> i32 ---
	LAPACKE_dtprfb :: proc(matrix_layout: c.int, side: c.char, trans: c.char, direct: c.char, storev: c.char, m: i32, n: i32, k: i32, l: i32, v: ^f64, ldv: i32, t: ^f64, ldt: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32) -> i32 ---
	LAPACKE_ctprfb :: proc(matrix_layout: c.int, side: c.char, trans: c.char, direct: c.char, storev: c.char, m: i32, n: i32, k: i32, l: i32, v: ^complex64, ldv: i32, t: ^complex64, ldt: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_ztprfb :: proc(matrix_layout: c.int, side: c.char, trans: c.char, direct: c.char, storev: c.char, m: i32, n: i32, k: i32, l: i32, v: ^complex128, ldv: i32, t: ^complex128, ldt: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_sgemqrt_work :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, nb: i32, v: ^f32, ldv: i32, t: ^f32, ldt: i32, _c: ^f32, ldc: i32, work: ^f32) -> i32 ---
	LAPACKE_dgemqrt_work :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, nb: i32, v: ^f64, ldv: i32, t: ^f64, ldt: i32, _c: ^f64, ldc: i32, work: ^f64) -> i32 ---
	LAPACKE_cgemqrt_work :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, nb: i32, v: ^complex64, ldv: i32, t: ^complex64, ldt: i32, _c: ^complex64, ldc: i32, work: ^complex64) -> i32 ---
	LAPACKE_zgemqrt_work :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, nb: i32, v: ^complex128, ldv: i32, t: ^complex128, ldt: i32, _c: ^complex128, ldc: i32, work: ^complex128) -> i32 ---
	LAPACKE_sgeqrt_work :: proc(matrix_layout: c.int, m: i32, n: i32, nb: i32, a: ^f32, lda: i32, t: ^f32, ldt: i32, work: ^f32) -> i32 ---
	LAPACKE_dgeqrt_work :: proc(matrix_layout: c.int, m: i32, n: i32, nb: i32, a: ^f64, lda: i32, t: ^f64, ldt: i32, work: ^f64) -> i32 ---
	LAPACKE_cgeqrt_work :: proc(matrix_layout: c.int, m: i32, n: i32, nb: i32, a: ^complex64, lda: i32, t: ^complex64, ldt: i32, work: ^complex64) -> i32 ---
	LAPACKE_zgeqrt_work :: proc(matrix_layout: c.int, m: i32, n: i32, nb: i32, a: ^complex128, lda: i32, t: ^complex128, ldt: i32, work: ^complex128) -> i32 ---
	LAPACKE_sgeqrt2_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f32, lda: i32, t: ^f32, ldt: i32) -> i32 ---
	LAPACKE_dgeqrt2_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f64, lda: i32, t: ^f64, ldt: i32) -> i32 ---
	LAPACKE_cgeqrt2_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex64, lda: i32, t: ^complex64, ldt: i32) -> i32 ---
	LAPACKE_zgeqrt2_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex128, lda: i32, t: ^complex128, ldt: i32) -> i32 ---
	LAPACKE_sgeqrt3_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f32, lda: i32, t: ^f32, ldt: i32) -> i32 ---
	LAPACKE_dgeqrt3_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f64, lda: i32, t: ^f64, ldt: i32) -> i32 ---
	LAPACKE_cgeqrt3_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex64, lda: i32, t: ^complex64, ldt: i32) -> i32 ---
	LAPACKE_zgeqrt3_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex128, lda: i32, t: ^complex128, ldt: i32) -> i32 ---
	LAPACKE_stpmqrt_work :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, l: i32, nb: i32, v: ^f32, ldv: i32, t: ^f32, ldt: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32, work: ^f32) -> i32 ---
	LAPACKE_dtpmqrt_work :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, l: i32, nb: i32, v: ^f64, ldv: i32, t: ^f64, ldt: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, work: ^f64) -> i32 ---
	LAPACKE_ctpmqrt_work :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, l: i32, nb: i32, v: ^complex64, ldv: i32, t: ^complex64, ldt: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32, work: ^complex64) -> i32 ---
	LAPACKE_ztpmqrt_work :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, l: i32, nb: i32, v: ^complex128, ldv: i32, t: ^complex128, ldt: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, work: ^complex128) -> i32 ---
	LAPACKE_stpqrt_work :: proc(matrix_layout: c.int, m: i32, n: i32, l: i32, nb: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32, t: ^f32, ldt: i32, work: ^f32) -> i32 ---
	LAPACKE_dtpqrt_work :: proc(matrix_layout: c.int, m: i32, n: i32, l: i32, nb: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, t: ^f64, ldt: i32, work: ^f64) -> i32 ---
	LAPACKE_ctpqrt_work :: proc(matrix_layout: c.int, m: i32, n: i32, l: i32, nb: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32, t: ^complex64, ldt: i32, work: ^complex64) -> i32 ---
	LAPACKE_ztpqrt_work :: proc(matrix_layout: c.int, m: i32, n: i32, l: i32, nb: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, t: ^complex128, ldt: i32, work: ^complex128) -> i32 ---
	LAPACKE_stpqrt2_work :: proc(matrix_layout: c.int, m: i32, n: i32, l: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32, t: ^f32, ldt: i32) -> i32 ---
	LAPACKE_dtpqrt2_work :: proc(matrix_layout: c.int, m: i32, n: i32, l: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, t: ^f64, ldt: i32) -> i32 ---
	LAPACKE_ctpqrt2_work :: proc(matrix_layout: c.int, m: i32, n: i32, l: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32, t: ^complex64, ldt: i32) -> i32 ---
	LAPACKE_ztpqrt2_work :: proc(matrix_layout: c.int, m: i32, n: i32, l: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, t: ^complex128, ldt: i32) -> i32 ---
	LAPACKE_stprfb_work :: proc(matrix_layout: c.int, side: c.char, trans: c.char, direct: c.char, storev: c.char, m: i32, n: i32, k: i32, l: i32, v: ^f32, ldv: i32, t: ^f32, ldt: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32, work: ^f32, ldwork: i32) -> i32 ---
	LAPACKE_dtprfb_work :: proc(matrix_layout: c.int, side: c.char, trans: c.char, direct: c.char, storev: c.char, m: i32, n: i32, k: i32, l: i32, v: ^f64, ldv: i32, t: ^f64, ldt: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, work: ^f64, ldwork: i32) -> i32 ---
	LAPACKE_ctprfb_work :: proc(matrix_layout: c.int, side: c.char, trans: c.char, direct: c.char, storev: c.char, m: i32, n: i32, k: i32, l: i32, v: ^complex64, ldv: i32, t: ^complex64, ldt: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32, work: ^complex64, ldwork: i32) -> i32 ---
	LAPACKE_ztprfb_work :: proc(matrix_layout: c.int, side: c.char, trans: c.char, direct: c.char, storev: c.char, m: i32, n: i32, k: i32, l: i32, v: ^complex128, ldv: i32, t: ^complex128, ldt: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, work: ^complex128, ldwork: i32) -> i32 ---

	//LAPACK 3.X.X
	LAPACKE_ssysv_rook :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^f32, lda: i32, ipiv: ^i32, b: ^f32, ldb: i32) -> i32 ---
	LAPACKE_dsysv_rook :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^f64, lda: i32, ipiv: ^i32, b: ^f64, ldb: i32) -> i32 ---
	LAPACKE_csysv_rook :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, ipiv: ^i32, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_zsysv_rook :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, ipiv: ^i32, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_ssytrf_rook :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f32, lda: i32, ipiv: ^i32) -> i32 ---
	LAPACKE_dsytrf_rook :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f64, lda: i32, ipiv: ^i32) -> i32 ---
	LAPACKE_csytrf_rook :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32, ipiv: ^i32) -> i32 ---
	LAPACKE_zsytrf_rook :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32, ipiv: ^i32) -> i32 ---
	LAPACKE_ssytrs_rook :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^f32, lda: i32, ipiv: ^i32, b: ^f32, ldb: i32) -> i32 ---
	LAPACKE_dsytrs_rook :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^f64, lda: i32, ipiv: ^i32, b: ^f64, ldb: i32) -> i32 ---
	LAPACKE_csytrs_rook :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, ipiv: ^i32, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_zsytrs_rook :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, ipiv: ^i32, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_chetrf_rook :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32, ipiv: ^i32) -> i32 ---
	LAPACKE_zhetrf_rook :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32, ipiv: ^i32) -> i32 ---
	LAPACKE_chetrs_rook :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, ipiv: ^i32, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_zhetrs_rook :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, ipiv: ^i32, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_csyr :: proc(matrix_layout: c.int, uplo: c.char, n: i32, alpha: complex64, x: ^complex64, incx: i32, a: ^complex64, lda: i32) -> i32 ---
	LAPACKE_zsyr :: proc(matrix_layout: c.int, uplo: c.char, n: i32, alpha: complex128, x: ^complex128, incx: i32, a: ^complex128, lda: i32) -> i32 ---
	LAPACKE_ssysv_rook_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^f32, lda: i32, ipiv: ^i32, b: ^f32, ldb: i32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_dsysv_rook_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^f64, lda: i32, ipiv: ^i32, b: ^f64, ldb: i32, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_csysv_rook_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, ipiv: ^i32, b: ^complex64, ldb: i32, work: ^complex64, lwork: i32) -> i32 ---
	LAPACKE_zsysv_rook_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, ipiv: ^i32, b: ^complex128, ldb: i32, work: ^complex128, lwork: i32) -> i32 ---
	LAPACKE_ssytrf_rook_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f32, lda: i32, ipiv: ^i32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_dsytrf_rook_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f64, lda: i32, ipiv: ^i32, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_csytrf_rook_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32, ipiv: ^i32, work: ^complex64, lwork: i32) -> i32 ---
	LAPACKE_zsytrf_rook_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32, ipiv: ^i32, work: ^complex128, lwork: i32) -> i32 ---
	LAPACKE_ssytrs_rook_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^f32, lda: i32, ipiv: ^i32, b: ^f32, ldb: i32) -> i32 ---
	LAPACKE_dsytrs_rook_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^f64, lda: i32, ipiv: ^i32, b: ^f64, ldb: i32) -> i32 ---
	LAPACKE_csytrs_rook_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, ipiv: ^i32, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_zsytrs_rook_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, ipiv: ^i32, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_chetrf_rook_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32, ipiv: ^i32, work: ^complex64, lwork: i32) -> i32 ---
	LAPACKE_zhetrf_rook_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32, ipiv: ^i32, work: ^complex128, lwork: i32) -> i32 ---
	LAPACKE_chetrs_rook_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, ipiv: ^i32, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_zhetrs_rook_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, ipiv: ^i32, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_csyr_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, alpha: complex64, x: ^complex64, incx: i32, a: ^complex64, lda: i32) -> i32 ---
	LAPACKE_zsyr_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, alpha: complex128, x: ^complex128, incx: i32, a: ^complex128, lda: i32) -> i32 ---
	LAPACKE_ilaver :: proc(vers_major: ^i32, vers_minor: ^i32, vers_patch: ^i32) ---

	// LAPACK 3.7.0
	LAPACKE_ssysv_aa :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^f32, lda: i32, ipiv: ^i32, b: ^f32, ldb: i32) -> i32 ---
	LAPACKE_ssysv_aa_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^f32, lda: i32, ipiv: ^i32, b: ^f32, ldb: i32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_dsysv_aa :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^f64, lda: i32, ipiv: ^i32, b: ^f64, ldb: i32) -> i32 ---
	LAPACKE_dsysv_aa_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^f64, lda: i32, ipiv: ^i32, b: ^f64, ldb: i32, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_csysv_aa :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, ipiv: ^i32, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_csysv_aa_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, ipiv: ^i32, b: ^complex64, ldb: i32, work: ^complex64, lwork: i32) -> i32 ---
	LAPACKE_zsysv_aa :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, ipiv: ^i32, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_zsysv_aa_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, ipiv: ^i32, b: ^complex128, ldb: i32, work: ^complex128, lwork: i32) -> i32 ---
	LAPACKE_chesv_aa :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, ipiv: ^i32, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_chesv_aa_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, ipiv: ^i32, b: ^complex64, ldb: i32, work: ^complex64, lwork: i32) -> i32 ---
	LAPACKE_zhesv_aa :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, ipiv: ^i32, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_zhesv_aa_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, ipiv: ^i32, b: ^complex128, ldb: i32, work: ^complex128, lwork: i32) -> i32 ---
	LAPACKE_ssytrf_aa :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f32, lda: i32, ipiv: ^i32) -> i32 ---
	LAPACKE_dsytrf_aa :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f64, lda: i32, ipiv: ^i32) -> i32 ---
	LAPACKE_csytrf_aa :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32, ipiv: ^i32) -> i32 ---
	LAPACKE_zsytrf_aa :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32, ipiv: ^i32) -> i32 ---
	LAPACKE_chetrf_aa :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32, ipiv: ^i32) -> i32 ---
	LAPACKE_zhetrf_aa :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32, ipiv: ^i32) -> i32 ---
	LAPACKE_ssytrf_aa_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f32, lda: i32, ipiv: ^i32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_dsytrf_aa_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f64, lda: i32, ipiv: ^i32, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_csytrf_aa_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32, ipiv: ^i32, work: ^complex64, lwork: i32) -> i32 ---
	LAPACKE_zsytrf_aa_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32, ipiv: ^i32, work: ^complex128, lwork: i32) -> i32 ---
	LAPACKE_chetrf_aa_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32, ipiv: ^i32, work: ^complex64, lwork: i32) -> i32 ---
	LAPACKE_zhetrf_aa_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32, ipiv: ^i32, work: ^complex128, lwork: i32) -> i32 ---
	LAPACKE_csytrs_aa :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, ipiv: ^i32, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_csytrs_aa_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, ipiv: ^i32, b: ^complex64, ldb: i32, work: ^complex64, lwork: i32) -> i32 ---
	LAPACKE_chetrs_aa :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, ipiv: ^i32, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_chetrs_aa_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, ipiv: ^i32, b: ^complex64, ldb: i32, work: ^complex64, lwork: i32) -> i32 ---
	LAPACKE_dsytrs_aa :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^f64, lda: i32, ipiv: ^i32, b: ^f64, ldb: i32) -> i32 ---
	LAPACKE_dsytrs_aa_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^f64, lda: i32, ipiv: ^i32, b: ^f64, ldb: i32, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_ssytrs_aa :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^f32, lda: i32, ipiv: ^i32, b: ^f32, ldb: i32) -> i32 ---
	LAPACKE_ssytrs_aa_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^f32, lda: i32, ipiv: ^i32, b: ^f32, ldb: i32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_zsytrs_aa :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, ipiv: ^i32, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_zsytrs_aa_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, ipiv: ^i32, b: ^complex128, ldb: i32, work: ^complex128, lwork: i32) -> i32 ---
	LAPACKE_zhetrs_aa :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, ipiv: ^i32, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_zhetrs_aa_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, ipiv: ^i32, b: ^complex128, ldb: i32, work: ^complex128, lwork: i32) -> i32 ---
	LAPACKE_ssysv_rk :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^f32, lda: i32, e: ^f32, ipiv: ^i32, b: ^f32, ldb: i32) -> i32 ---
	LAPACKE_ssysv_rk_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^f32, lda: i32, e: ^f32, ipiv: ^i32, b: ^f32, ldb: i32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_dsysv_rk :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^f64, lda: i32, e: ^f64, ipiv: ^i32, b: ^f64, ldb: i32) -> i32 ---
	LAPACKE_dsysv_rk_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^f64, lda: i32, e: ^f64, ipiv: ^i32, b: ^f64, ldb: i32, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_csysv_rk :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, e: ^complex64, ipiv: ^i32, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_csysv_rk_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, e: ^complex64, ipiv: ^i32, b: ^complex64, ldb: i32, work: ^complex64, lwork: i32) -> i32 ---
	LAPACKE_zsysv_rk :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, e: ^complex128, ipiv: ^i32, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_zsysv_rk_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, e: ^complex128, ipiv: ^i32, b: ^complex128, ldb: i32, work: ^complex128, lwork: i32) -> i32 ---
	LAPACKE_chesv_rk :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, e: ^complex64, ipiv: ^i32, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_chesv_rk_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, e: ^complex64, ipiv: ^i32, b: ^complex64, ldb: i32, work: ^complex64, lwork: i32) -> i32 ---
	LAPACKE_zhesv_rk :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, e: ^complex128, ipiv: ^i32, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_zhesv_rk_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, e: ^complex128, ipiv: ^i32, b: ^complex128, ldb: i32, work: ^complex128, lwork: i32) -> i32 ---
	LAPACKE_ssytrf_rk :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f32, lda: i32, e: ^f32, ipiv: ^i32) -> i32 ---
	LAPACKE_dsytrf_rk :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f64, lda: i32, e: ^f64, ipiv: ^i32) -> i32 ---
	LAPACKE_csytrf_rk :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32, e: ^complex64, ipiv: ^i32) -> i32 ---
	LAPACKE_zsytrf_rk :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32, e: ^complex128, ipiv: ^i32) -> i32 ---
	LAPACKE_chetrf_rk :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32, e: ^complex64, ipiv: ^i32) -> i32 ---
	LAPACKE_zhetrf_rk :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32, e: ^complex128, ipiv: ^i32) -> i32 ---
	LAPACKE_ssytrf_rk_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f32, lda: i32, e: ^f32, ipiv: ^i32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_dsytrf_rk_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f64, lda: i32, e: ^f64, ipiv: ^i32, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_csytrf_rk_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32, e: ^complex64, ipiv: ^i32, work: ^complex64, lwork: i32) -> i32 ---
	LAPACKE_zsytrf_rk_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32, e: ^complex128, ipiv: ^i32, work: ^complex128, lwork: i32) -> i32 ---
	LAPACKE_chetrf_rk_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32, e: ^complex64, ipiv: ^i32, work: ^complex64, lwork: i32) -> i32 ---
	LAPACKE_zhetrf_rk_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32, e: ^complex128, ipiv: ^i32, work: ^complex128, lwork: i32) -> i32 ---
	LAPACKE_csytrs_3 :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, e: ^complex64, ipiv: ^i32, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_csytrs_3_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, e: ^complex64, ipiv: ^i32, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_chetrs_3 :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, e: ^complex64, ipiv: ^i32, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_chetrs_3_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, e: ^complex64, ipiv: ^i32, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_dsytrs_3 :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^f64, lda: i32, e: ^f64, ipiv: ^i32, b: ^f64, ldb: i32) -> i32 ---
	LAPACKE_dsytrs_3_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^f64, lda: i32, e: ^f64, ipiv: ^i32, b: ^f64, ldb: i32) -> i32 ---
	LAPACKE_ssytrs_3 :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^f32, lda: i32, e: ^f32, ipiv: ^i32, b: ^f32, ldb: i32) -> i32 ---
	LAPACKE_ssytrs_3_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^f32, lda: i32, e: ^f32, ipiv: ^i32, b: ^f32, ldb: i32) -> i32 ---
	LAPACKE_zsytrs_3 :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, e: ^complex128, ipiv: ^i32, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_zsytrs_3_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, e: ^complex128, ipiv: ^i32, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_zhetrs_3 :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, e: ^complex128, ipiv: ^i32, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_zhetrs_3_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, e: ^complex128, ipiv: ^i32, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_ssytri_3 :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f32, lda: i32, e: ^f32, ipiv: ^i32) -> i32 ---
	LAPACKE_dsytri_3 :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f64, lda: i32, e: ^f64, ipiv: ^i32) -> i32 ---
	LAPACKE_csytri_3 :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32, e: ^complex64, ipiv: ^i32) -> i32 ---
	LAPACKE_zsytri_3 :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32, e: ^complex128, ipiv: ^i32) -> i32 ---
	LAPACKE_chetri_3 :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32, e: ^complex64, ipiv: ^i32) -> i32 ---
	LAPACKE_zhetri_3 :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32, e: ^complex128, ipiv: ^i32) -> i32 ---
	LAPACKE_ssytri_3_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f32, lda: i32, e: ^f32, ipiv: ^i32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_dsytri_3_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f64, lda: i32, e: ^f64, ipiv: ^i32, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_csytri_3_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32, e: ^complex64, ipiv: ^i32, work: ^complex64, lwork: i32) -> i32 ---
	LAPACKE_zsytri_3_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32, e: ^complex128, ipiv: ^i32, work: ^complex128, lwork: i32) -> i32 ---
	LAPACKE_chetri_3_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32, e: ^complex64, ipiv: ^i32, work: ^complex64, lwork: i32) -> i32 ---
	LAPACKE_zhetri_3_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32, e: ^complex128, ipiv: ^i32, work: ^complex128, lwork: i32) -> i32 ---
	LAPACKE_ssycon_3 :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f32, lda: i32, e: ^f32, ipiv: ^i32, anorm: f32, rcond: ^f32) -> i32 ---
	LAPACKE_dsycon_3 :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f64, lda: i32, e: ^f64, ipiv: ^i32, anorm: f64, rcond: ^f64) -> i32 ---
	LAPACKE_csycon_3 :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32, e: ^complex64, ipiv: ^i32, anorm: f32, rcond: ^f32) -> i32 ---
	LAPACKE_zsycon_3 :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32, e: ^complex128, ipiv: ^i32, anorm: f64, rcond: ^f64) -> i32 ---
	LAPACKE_checon_3 :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32, e: ^complex64, ipiv: ^i32, anorm: f32, rcond: ^f32) -> i32 ---
	LAPACKE_zhecon_3 :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32, e: ^complex128, ipiv: ^i32, anorm: f64, rcond: ^f64) -> i32 ---
	LAPACKE_ssycon_3_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f32, lda: i32, e: ^f32, ipiv: ^i32, anorm: f32, rcond: ^f32, work: ^f32, iwork: ^i32) -> i32 ---
	LAPACKE_dsycon_3_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f64, lda: i32, e: ^f64, ipiv: ^i32, anorm: f64, rcond: ^f64, work: ^f64, iwork: ^i32) -> i32 ---
	LAPACKE_csycon_3_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32, e: ^complex64, ipiv: ^i32, anorm: f32, rcond: ^f32, work: ^complex64) -> i32 ---
	LAPACKE_zsycon_3_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32, e: ^complex128, ipiv: ^i32, anorm: f64, rcond: ^f64, work: ^complex128) -> i32 ---
	LAPACKE_checon_3_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32, e: ^complex64, ipiv: ^i32, anorm: f32, rcond: ^f32, work: ^complex64) -> i32 ---
	LAPACKE_zhecon_3_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32, e: ^complex128, ipiv: ^i32, anorm: f64, rcond: ^f64, work: ^complex128) -> i32 ---
	LAPACKE_sgelq :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f32, lda: i32, t: ^f32, tsize: i32) -> i32 ---
	LAPACKE_dgelq :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f64, lda: i32, t: ^f64, tsize: i32) -> i32 ---
	LAPACKE_cgelq :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex64, lda: i32, t: ^complex64, tsize: i32) -> i32 ---
	LAPACKE_zgelq :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex128, lda: i32, t: ^complex128, tsize: i32) -> i32 ---
	LAPACKE_sgelq_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f32, lda: i32, t: ^f32, tsize: i32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_dgelq_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f64, lda: i32, t: ^f64, tsize: i32, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_cgelq_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex64, lda: i32, t: ^complex64, tsize: i32, work: ^complex64, lwork: i32) -> i32 ---
	LAPACKE_zgelq_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex128, lda: i32, t: ^complex128, tsize: i32, work: ^complex128, lwork: i32) -> i32 ---
	LAPACKE_sgemlq :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, a: ^f32, lda: i32, t: ^f32, tsize: i32, _c: ^f32, ldc: i32) -> i32 ---
	LAPACKE_dgemlq :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, a: ^f64, lda: i32, t: ^f64, tsize: i32, _c: ^f64, ldc: i32) -> i32 ---
	LAPACKE_cgemlq :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, a: ^complex64, lda: i32, t: ^complex64, tsize: i32, _c: ^complex64, ldc: i32) -> i32 ---
	LAPACKE_zgemlq :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, a: ^complex128, lda: i32, t: ^complex128, tsize: i32, _c: ^complex128, ldc: i32) -> i32 ---
	LAPACKE_sgemlq_work :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, a: ^f32, lda: i32, t: ^f32, tsize: i32, _c: ^f32, ldc: i32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_dgemlq_work :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, a: ^f64, lda: i32, t: ^f64, tsize: i32, _c: ^f64, ldc: i32, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_cgemlq_work :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, a: ^complex64, lda: i32, t: ^complex64, tsize: i32, _c: ^complex64, ldc: i32, work: ^complex64, lwork: i32) -> i32 ---
	LAPACKE_zgemlq_work :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, a: ^complex128, lda: i32, t: ^complex128, tsize: i32, _c: ^complex128, ldc: i32, work: ^complex128, lwork: i32) -> i32 ---
	LAPACKE_sgeqr :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f32, lda: i32, t: ^f32, tsize: i32) -> i32 ---
	LAPACKE_dgeqr :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f64, lda: i32, t: ^f64, tsize: i32) -> i32 ---
	LAPACKE_cgeqr :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex64, lda: i32, t: ^complex64, tsize: i32) -> i32 ---
	LAPACKE_zgeqr :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex128, lda: i32, t: ^complex128, tsize: i32) -> i32 ---
	LAPACKE_sgeqr_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f32, lda: i32, t: ^f32, tsize: i32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_dgeqr_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^f64, lda: i32, t: ^f64, tsize: i32, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_cgeqr_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex64, lda: i32, t: ^complex64, tsize: i32, work: ^complex64, lwork: i32) -> i32 ---
	LAPACKE_zgeqr_work :: proc(matrix_layout: c.int, m: i32, n: i32, a: ^complex128, lda: i32, t: ^complex128, tsize: i32, work: ^complex128, lwork: i32) -> i32 ---
	LAPACKE_sgemqr :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, a: ^f32, lda: i32, t: ^f32, tsize: i32, _c: ^f32, ldc: i32) -> i32 ---
	LAPACKE_dgemqr :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, a: ^f64, lda: i32, t: ^f64, tsize: i32, _c: ^f64, ldc: i32) -> i32 ---
	LAPACKE_cgemqr :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, a: ^complex64, lda: i32, t: ^complex64, tsize: i32, _c: ^complex64, ldc: i32) -> i32 ---
	LAPACKE_zgemqr :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, a: ^complex128, lda: i32, t: ^complex128, tsize: i32, _c: ^complex128, ldc: i32) -> i32 ---
	LAPACKE_sgemqr_work :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, a: ^f32, lda: i32, t: ^f32, tsize: i32, _c: ^f32, ldc: i32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_dgemqr_work :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, a: ^f64, lda: i32, t: ^f64, tsize: i32, _c: ^f64, ldc: i32, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_cgemqr_work :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, a: ^complex64, lda: i32, t: ^complex64, tsize: i32, _c: ^complex64, ldc: i32, work: ^complex64, lwork: i32) -> i32 ---
	LAPACKE_zgemqr_work :: proc(matrix_layout: c.int, side: c.char, trans: c.char, m: i32, n: i32, k: i32, a: ^complex128, lda: i32, t: ^complex128, tsize: i32, _c: ^complex128, ldc: i32, work: ^complex128, lwork: i32) -> i32 ---
	LAPACKE_sgetsls :: proc(matrix_layout: c.int, trans: c.char, m: i32, n: i32, nrhs: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32) -> i32 ---
	LAPACKE_dgetsls :: proc(matrix_layout: c.int, trans: c.char, m: i32, n: i32, nrhs: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32) -> i32 ---
	LAPACKE_cgetsls :: proc(matrix_layout: c.int, trans: c.char, m: i32, n: i32, nrhs: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_zgetsls :: proc(matrix_layout: c.int, trans: c.char, m: i32, n: i32, nrhs: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_sgetsls_work :: proc(matrix_layout: c.int, trans: c.char, m: i32, n: i32, nrhs: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_dgetsls_work :: proc(matrix_layout: c.int, trans: c.char, m: i32, n: i32, nrhs: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_cgetsls_work :: proc(matrix_layout: c.int, trans: c.char, m: i32, n: i32, nrhs: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32, work: ^complex64, lwork: i32) -> i32 ---
	LAPACKE_zgetsls_work :: proc(matrix_layout: c.int, trans: c.char, m: i32, n: i32, nrhs: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, work: ^complex128, lwork: i32) -> i32 ---
	LAPACKE_sgetsqrhrt :: proc(matrix_layout: c.int, m: i32, n: i32, mb1: i32, nb1: i32, nb2: i32, a: ^f32, lda: i32, t: ^f32, ldt: i32) -> i32 ---
	LAPACKE_dgetsqrhrt :: proc(matrix_layout: c.int, m: i32, n: i32, mb1: i32, nb1: i32, nb2: i32, a: ^f64, lda: i32, t: ^f64, ldt: i32) -> i32 ---
	LAPACKE_cgetsqrhrt :: proc(matrix_layout: c.int, m: i32, n: i32, mb1: i32, nb1: i32, nb2: i32, a: ^complex64, lda: i32, t: ^complex64, ldt: i32) -> i32 ---
	LAPACKE_zgetsqrhrt :: proc(matrix_layout: c.int, m: i32, n: i32, mb1: i32, nb1: i32, nb2: i32, a: ^complex128, lda: i32, t: ^complex128, ldt: i32) -> i32 ---
	LAPACKE_sgetsqrhrt_work :: proc(matrix_layout: c.int, m: i32, n: i32, mb1: i32, nb1: i32, nb2: i32, a: ^f32, lda: i32, t: ^f32, ldt: i32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_dgetsqrhrt_work :: proc(matrix_layout: c.int, m: i32, n: i32, mb1: i32, nb1: i32, nb2: i32, a: ^f64, lda: i32, t: ^f64, ldt: i32, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_cgetsqrhrt_work :: proc(matrix_layout: c.int, m: i32, n: i32, mb1: i32, nb1: i32, nb2: i32, a: ^complex64, lda: i32, t: ^complex64, ldt: i32, work: ^complex64, lwork: i32) -> i32 ---
	LAPACKE_zgetsqrhrt_work :: proc(matrix_layout: c.int, m: i32, n: i32, mb1: i32, nb1: i32, nb2: i32, a: ^complex128, lda: i32, t: ^complex128, ldt: i32, work: ^complex128, lwork: i32) -> i32 ---
	LAPACKE_ssyev_2stage :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, a: ^f32, lda: i32, w: ^f32) -> i32 ---
	LAPACKE_dsyev_2stage :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, a: ^f64, lda: i32, w: ^f64) -> i32 ---
	LAPACKE_ssyevd_2stage :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, a: ^f32, lda: i32, w: ^f32) -> i32 ---
	LAPACKE_dsyevd_2stage :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, a: ^f64, lda: i32, w: ^f64) -> i32 ---
	LAPACKE_ssyevr_2stage :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, uplo: c.char, n: i32, a: ^f32, lda: i32, vl: f32, vu: f32, il: i32, iu: i32, abstol: f32, m: ^i32, w: ^f32, z: ^f32, ldz: i32, isuppz: ^i32) -> i32 ---
	LAPACKE_dsyevr_2stage :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, uplo: c.char, n: i32, a: ^f64, lda: i32, vl: f64, vu: f64, il: i32, iu: i32, abstol: f64, m: ^i32, w: ^f64, z: ^f64, ldz: i32, isuppz: ^i32) -> i32 ---
	LAPACKE_ssyevx_2stage :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, uplo: c.char, n: i32, a: ^f32, lda: i32, vl: f32, vu: f32, il: i32, iu: i32, abstol: f32, m: ^i32, w: ^f32, z: ^f32, ldz: i32, ifail: ^i32) -> i32 ---
	LAPACKE_dsyevx_2stage :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, uplo: c.char, n: i32, a: ^f64, lda: i32, vl: f64, vu: f64, il: i32, iu: i32, abstol: f64, m: ^i32, w: ^f64, z: ^f64, ldz: i32, ifail: ^i32) -> i32 ---
	LAPACKE_ssyev_2stage_work :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, a: ^f32, lda: i32, w: ^f32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_dsyev_2stage_work :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, a: ^f64, lda: i32, w: ^f64, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_ssyevd_2stage_work :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, a: ^f32, lda: i32, w: ^f32, work: ^f32, lwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_dsyevd_2stage_work :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, a: ^f64, lda: i32, w: ^f64, work: ^f64, lwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_ssyevr_2stage_work :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, uplo: c.char, n: i32, a: ^f32, lda: i32, vl: f32, vu: f32, il: i32, iu: i32, abstol: f32, m: ^i32, w: ^f32, z: ^f32, ldz: i32, isuppz: ^i32, work: ^f32, lwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_dsyevr_2stage_work :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, uplo: c.char, n: i32, a: ^f64, lda: i32, vl: f64, vu: f64, il: i32, iu: i32, abstol: f64, m: ^i32, w: ^f64, z: ^f64, ldz: i32, isuppz: ^i32, work: ^f64, lwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_ssyevx_2stage_work :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, uplo: c.char, n: i32, a: ^f32, lda: i32, vl: f32, vu: f32, il: i32, iu: i32, abstol: f32, m: ^i32, w: ^f32, z: ^f32, ldz: i32, work: ^f32, lwork: i32, iwork: ^i32, ifail: ^i32) -> i32 ---
	LAPACKE_dsyevx_2stage_work :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, uplo: c.char, n: i32, a: ^f64, lda: i32, vl: f64, vu: f64, il: i32, iu: i32, abstol: f64, m: ^i32, w: ^f64, z: ^f64, ldz: i32, work: ^f64, lwork: i32, iwork: ^i32, ifail: ^i32) -> i32 ---
	LAPACKE_cheev_2stage :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, a: ^complex64, lda: i32, w: ^f32) -> i32 ---
	LAPACKE_zheev_2stage :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, a: ^complex128, lda: i32, w: ^f64) -> i32 ---
	LAPACKE_cheevd_2stage :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, a: ^complex64, lda: i32, w: ^f32) -> i32 ---
	LAPACKE_zheevd_2stage :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, a: ^complex128, lda: i32, w: ^f64) -> i32 ---
	LAPACKE_cheevr_2stage :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, uplo: c.char, n: i32, a: ^complex64, lda: i32, vl: f32, vu: f32, il: i32, iu: i32, abstol: f32, m: ^i32, w: ^f32, z: ^complex64, ldz: i32, isuppz: ^i32) -> i32 ---
	LAPACKE_zheevr_2stage :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, uplo: c.char, n: i32, a: ^complex128, lda: i32, vl: f64, vu: f64, il: i32, iu: i32, abstol: f64, m: ^i32, w: ^f64, z: ^complex128, ldz: i32, isuppz: ^i32) -> i32 ---
	LAPACKE_cheevx_2stage :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, uplo: c.char, n: i32, a: ^complex64, lda: i32, vl: f32, vu: f32, il: i32, iu: i32, abstol: f32, m: ^i32, w: ^f32, z: ^complex64, ldz: i32, ifail: ^i32) -> i32 ---
	LAPACKE_zheevx_2stage :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, uplo: c.char, n: i32, a: ^complex128, lda: i32, vl: f64, vu: f64, il: i32, iu: i32, abstol: f64, m: ^i32, w: ^f64, z: ^complex128, ldz: i32, ifail: ^i32) -> i32 ---
	LAPACKE_cheev_2stage_work :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, a: ^complex64, lda: i32, w: ^f32, work: ^complex64, lwork: i32, rwork: ^f32) -> i32 ---
	LAPACKE_zheev_2stage_work :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, a: ^complex128, lda: i32, w: ^f64, work: ^complex128, lwork: i32, rwork: ^f64) -> i32 ---
	LAPACKE_cheevd_2stage_work :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, a: ^complex64, lda: i32, w: ^f32, work: ^complex64, lwork: i32, rwork: ^f32, lrwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_zheevd_2stage_work :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, a: ^complex128, lda: i32, w: ^f64, work: ^complex128, lwork: i32, rwork: ^f64, lrwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_cheevr_2stage_work :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, uplo: c.char, n: i32, a: ^complex64, lda: i32, vl: f32, vu: f32, il: i32, iu: i32, abstol: f32, m: ^i32, w: ^f32, z: ^complex64, ldz: i32, isuppz: ^i32, work: ^complex64, lwork: i32, rwork: ^f32, lrwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_zheevr_2stage_work :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, uplo: c.char, n: i32, a: ^complex128, lda: i32, vl: f64, vu: f64, il: i32, iu: i32, abstol: f64, m: ^i32, w: ^f64, z: ^complex128, ldz: i32, isuppz: ^i32, work: ^complex128, lwork: i32, rwork: ^f64, lrwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_cheevx_2stage_work :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, uplo: c.char, n: i32, a: ^complex64, lda: i32, vl: f32, vu: f32, il: i32, iu: i32, abstol: f32, m: ^i32, w: ^f32, z: ^complex64, ldz: i32, work: ^complex64, lwork: i32, rwork: ^f32, iwork: ^i32, ifail: ^i32) -> i32 ---
	LAPACKE_zheevx_2stage_work :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, uplo: c.char, n: i32, a: ^complex128, lda: i32, vl: f64, vu: f64, il: i32, iu: i32, abstol: f64, m: ^i32, w: ^f64, z: ^complex128, ldz: i32, work: ^complex128, lwork: i32, rwork: ^f64, iwork: ^i32, ifail: ^i32) -> i32 ---
	LAPACKE_ssbev_2stage :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, kd: i32, ab: ^f32, ldab: i32, w: ^f32, z: ^f32, ldz: i32) -> i32 ---
	LAPACKE_dsbev_2stage :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, kd: i32, ab: ^f64, ldab: i32, w: ^f64, z: ^f64, ldz: i32) -> i32 ---
	LAPACKE_ssbevd_2stage :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, kd: i32, ab: ^f32, ldab: i32, w: ^f32, z: ^f32, ldz: i32) -> i32 ---
	LAPACKE_dsbevd_2stage :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, kd: i32, ab: ^f64, ldab: i32, w: ^f64, z: ^f64, ldz: i32) -> i32 ---
	LAPACKE_ssbevx_2stage :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, uplo: c.char, n: i32, kd: i32, ab: ^f32, ldab: i32, q: ^f32, ldq: i32, vl: f32, vu: f32, il: i32, iu: i32, abstol: f32, m: ^i32, w: ^f32, z: ^f32, ldz: i32, ifail: ^i32) -> i32 ---
	LAPACKE_dsbevx_2stage :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, uplo: c.char, n: i32, kd: i32, ab: ^f64, ldab: i32, q: ^f64, ldq: i32, vl: f64, vu: f64, il: i32, iu: i32, abstol: f64, m: ^i32, w: ^f64, z: ^f64, ldz: i32, ifail: ^i32) -> i32 ---
	LAPACKE_ssbev_2stage_work :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, kd: i32, ab: ^f32, ldab: i32, w: ^f32, z: ^f32, ldz: i32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_dsbev_2stage_work :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, kd: i32, ab: ^f64, ldab: i32, w: ^f64, z: ^f64, ldz: i32, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_ssbevd_2stage_work :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, kd: i32, ab: ^f32, ldab: i32, w: ^f32, z: ^f32, ldz: i32, work: ^f32, lwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_dsbevd_2stage_work :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, kd: i32, ab: ^f64, ldab: i32, w: ^f64, z: ^f64, ldz: i32, work: ^f64, lwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_ssbevx_2stage_work :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, uplo: c.char, n: i32, kd: i32, ab: ^f32, ldab: i32, q: ^f32, ldq: i32, vl: f32, vu: f32, il: i32, iu: i32, abstol: f32, m: ^i32, w: ^f32, z: ^f32, ldz: i32, work: ^f32, lwork: i32, iwork: ^i32, ifail: ^i32) -> i32 ---
	LAPACKE_dsbevx_2stage_work :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, uplo: c.char, n: i32, kd: i32, ab: ^f64, ldab: i32, q: ^f64, ldq: i32, vl: f64, vu: f64, il: i32, iu: i32, abstol: f64, m: ^i32, w: ^f64, z: ^f64, ldz: i32, work: ^f64, lwork: i32, iwork: ^i32, ifail: ^i32) -> i32 ---
	LAPACKE_chbev_2stage :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, kd: i32, ab: ^complex64, ldab: i32, w: ^f32, z: ^complex64, ldz: i32) -> i32 ---
	LAPACKE_zhbev_2stage :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, kd: i32, ab: ^complex128, ldab: i32, w: ^f64, z: ^complex128, ldz: i32) -> i32 ---
	LAPACKE_chbevd_2stage :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, kd: i32, ab: ^complex64, ldab: i32, w: ^f32, z: ^complex64, ldz: i32) -> i32 ---
	LAPACKE_zhbevd_2stage :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, kd: i32, ab: ^complex128, ldab: i32, w: ^f64, z: ^complex128, ldz: i32) -> i32 ---
	LAPACKE_chbevx_2stage :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, uplo: c.char, n: i32, kd: i32, ab: ^complex64, ldab: i32, q: ^complex64, ldq: i32, vl: f32, vu: f32, il: i32, iu: i32, abstol: f32, m: ^i32, w: ^f32, z: ^complex64, ldz: i32, ifail: ^i32) -> i32 ---
	LAPACKE_zhbevx_2stage :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, uplo: c.char, n: i32, kd: i32, ab: ^complex128, ldab: i32, q: ^complex128, ldq: i32, vl: f64, vu: f64, il: i32, iu: i32, abstol: f64, m: ^i32, w: ^f64, z: ^complex128, ldz: i32, ifail: ^i32) -> i32 ---
	LAPACKE_chbev_2stage_work :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, kd: i32, ab: ^complex64, ldab: i32, w: ^f32, z: ^complex64, ldz: i32, work: ^complex64, lwork: i32, rwork: ^f32) -> i32 ---
	LAPACKE_zhbev_2stage_work :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, kd: i32, ab: ^complex128, ldab: i32, w: ^f64, z: ^complex128, ldz: i32, work: ^complex128, lwork: i32, rwork: ^f64) -> i32 ---
	LAPACKE_chbevd_2stage_work :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, kd: i32, ab: ^complex64, ldab: i32, w: ^f32, z: ^complex64, ldz: i32, work: ^complex64, lwork: i32, rwork: ^f32, lrwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_zhbevd_2stage_work :: proc(matrix_layout: c.int, jobz: c.char, uplo: c.char, n: i32, kd: i32, ab: ^complex128, ldab: i32, w: ^f64, z: ^complex128, ldz: i32, work: ^complex128, lwork: i32, rwork: ^f64, lrwork: i32, iwork: ^i32, liwork: i32) -> i32 ---
	LAPACKE_chbevx_2stage_work :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, uplo: c.char, n: i32, kd: i32, ab: ^complex64, ldab: i32, q: ^complex64, ldq: i32, vl: f32, vu: f32, il: i32, iu: i32, abstol: f32, m: ^i32, w: ^f32, z: ^complex64, ldz: i32, work: ^complex64, lwork: i32, rwork: ^f32, iwork: ^i32, ifail: ^i32) -> i32 ---
	LAPACKE_zhbevx_2stage_work :: proc(matrix_layout: c.int, jobz: c.char, range: c.char, uplo: c.char, n: i32, kd: i32, ab: ^complex128, ldab: i32, q: ^complex128, ldq: i32, vl: f64, vu: f64, il: i32, iu: i32, abstol: f64, m: ^i32, w: ^f64, z: ^complex128, ldz: i32, work: ^complex128, lwork: i32, rwork: ^f64, iwork: ^i32, ifail: ^i32) -> i32 ---
	LAPACKE_ssygv_2stage :: proc(matrix_layout: c.int, itype: i32, jobz: c.char, uplo: c.char, n: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32, w: ^f32) -> i32 ---
	LAPACKE_dsygv_2stage :: proc(matrix_layout: c.int, itype: i32, jobz: c.char, uplo: c.char, n: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, w: ^f64) -> i32 ---
	LAPACKE_ssygv_2stage_work :: proc(matrix_layout: c.int, itype: i32, jobz: c.char, uplo: c.char, n: i32, a: ^f32, lda: i32, b: ^f32, ldb: i32, w: ^f32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_dsygv_2stage_work :: proc(matrix_layout: c.int, itype: i32, jobz: c.char, uplo: c.char, n: i32, a: ^f64, lda: i32, b: ^f64, ldb: i32, w: ^f64, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_chegv_2stage :: proc(matrix_layout: c.int, itype: i32, jobz: c.char, uplo: c.char, n: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32, w: ^f32) -> i32 ---
	LAPACKE_zhegv_2stage :: proc(matrix_layout: c.int, itype: i32, jobz: c.char, uplo: c.char, n: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, w: ^f64) -> i32 ---
	LAPACKE_chegv_2stage_work :: proc(matrix_layout: c.int, itype: i32, jobz: c.char, uplo: c.char, n: i32, a: ^complex64, lda: i32, b: ^complex64, ldb: i32, w: ^f32, work: ^complex64, lwork: i32, rwork: ^f32) -> i32 ---
	LAPACKE_zhegv_2stage_work :: proc(matrix_layout: c.int, itype: i32, jobz: c.char, uplo: c.char, n: i32, a: ^complex128, lda: i32, b: ^complex128, ldb: i32, w: ^f64, work: ^complex128, lwork: i32, rwork: ^f64) -> i32 ---

	//LAPACK 3.8.0
	LAPACKE_ssysv_aa_2stage :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^f32, lda: i32, tb: ^f32, ltb: i32, ipiv: ^i32, ipiv2: ^i32, b: ^f32, ldb: i32) -> i32 ---
	LAPACKE_ssysv_aa_2stage_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^f32, lda: i32, tb: ^f32, ltb: i32, ipiv: ^i32, ipiv2: ^i32, b: ^f32, ldb: i32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_dsysv_aa_2stage :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^f64, lda: i32, tb: ^f64, ltb: i32, ipiv: ^i32, ipiv2: ^i32, b: ^f64, ldb: i32) -> i32 ---
	LAPACKE_dsysv_aa_2stage_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^f64, lda: i32, tb: ^f64, ltb: i32, ipiv: ^i32, ipiv2: ^i32, b: ^f64, ldb: i32, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_csysv_aa_2stage :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, tb: ^complex64, ltb: i32, ipiv: ^i32, ipiv2: ^i32, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_csysv_aa_2stage_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, tb: ^complex64, ltb: i32, ipiv: ^i32, ipiv2: ^i32, b: ^complex64, ldb: i32, work: ^complex64, lwork: i32) -> i32 ---
	LAPACKE_zsysv_aa_2stage :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, tb: ^complex128, ltb: i32, ipiv: ^i32, ipiv2: ^i32, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_zsysv_aa_2stage_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, tb: ^complex128, ltb: i32, ipiv: ^i32, ipiv2: ^i32, b: ^complex128, ldb: i32, work: ^complex128, lwork: i32) -> i32 ---
	LAPACKE_chesv_aa_2stage :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, tb: ^complex64, ltb: i32, ipiv: ^i32, ipiv2: ^i32, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_chesv_aa_2stage_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, tb: ^complex64, ltb: i32, ipiv: ^i32, ipiv2: ^i32, b: ^complex64, ldb: i32, work: ^complex64, lwork: i32) -> i32 ---
	LAPACKE_zhesv_aa_2stage :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, tb: ^complex128, ltb: i32, ipiv: ^i32, ipiv2: ^i32, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_zhesv_aa_2stage_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, tb: ^complex128, ltb: i32, ipiv: ^i32, ipiv2: ^i32, b: ^complex128, ldb: i32, work: ^complex128, lwork: i32) -> i32 ---
	LAPACKE_ssytrf_aa_2stage :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f32, lda: i32, tb: ^f32, ltb: i32, ipiv: ^i32, ipiv2: ^i32) -> i32 ---
	LAPACKE_ssytrf_aa_2stage_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f32, lda: i32, tb: ^f32, ltb: i32, ipiv: ^i32, ipiv2: ^i32, work: ^f32, lwork: i32) -> i32 ---
	LAPACKE_dsytrf_aa_2stage :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f64, lda: i32, tb: ^f64, ltb: i32, ipiv: ^i32, ipiv2: ^i32) -> i32 ---
	LAPACKE_dsytrf_aa_2stage_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^f64, lda: i32, tb: ^f64, ltb: i32, ipiv: ^i32, ipiv2: ^i32, work: ^f64, lwork: i32) -> i32 ---
	LAPACKE_csytrf_aa_2stage :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32, tb: ^complex64, ltb: i32, ipiv: ^i32, ipiv2: ^i32) -> i32 ---
	LAPACKE_csytrf_aa_2stage_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32, tb: ^complex64, ltb: i32, ipiv: ^i32, ipiv2: ^i32, work: ^complex64, lwork: i32) -> i32 ---
	LAPACKE_zsytrf_aa_2stage :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32, tb: ^complex128, ltb: i32, ipiv: ^i32, ipiv2: ^i32) -> i32 ---
	LAPACKE_zsytrf_aa_2stage_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32, tb: ^complex128, ltb: i32, ipiv: ^i32, ipiv2: ^i32, work: ^complex128, lwork: i32) -> i32 ---
	LAPACKE_chetrf_aa_2stage :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32, tb: ^complex64, ltb: i32, ipiv: ^i32, ipiv2: ^i32) -> i32 ---
	LAPACKE_chetrf_aa_2stage_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex64, lda: i32, tb: ^complex64, ltb: i32, ipiv: ^i32, ipiv2: ^i32, work: ^complex64, lwork: i32) -> i32 ---
	LAPACKE_zhetrf_aa_2stage :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32, tb: ^complex128, ltb: i32, ipiv: ^i32, ipiv2: ^i32) -> i32 ---
	LAPACKE_zhetrf_aa_2stage_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, a: ^complex128, lda: i32, tb: ^complex128, ltb: i32, ipiv: ^i32, ipiv2: ^i32, work: ^complex128, lwork: i32) -> i32 ---
	LAPACKE_ssytrs_aa_2stage :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^f32, lda: i32, tb: ^f32, ltb: i32, ipiv: ^i32, ipiv2: ^i32, b: ^f32, ldb: i32) -> i32 ---
	LAPACKE_ssytrs_aa_2stage_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^f32, lda: i32, tb: ^f32, ltb: i32, ipiv: ^i32, ipiv2: ^i32, b: ^f32, ldb: i32) -> i32 ---
	LAPACKE_dsytrs_aa_2stage :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^f64, lda: i32, tb: ^f64, ltb: i32, ipiv: ^i32, ipiv2: ^i32, b: ^f64, ldb: i32) -> i32 ---
	LAPACKE_dsytrs_aa_2stage_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^f64, lda: i32, tb: ^f64, ltb: i32, ipiv: ^i32, ipiv2: ^i32, b: ^f64, ldb: i32) -> i32 ---
	LAPACKE_csytrs_aa_2stage :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, tb: ^complex64, ltb: i32, ipiv: ^i32, ipiv2: ^i32, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_csytrs_aa_2stage_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, tb: ^complex64, ltb: i32, ipiv: ^i32, ipiv2: ^i32, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_zsytrs_aa_2stage :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, tb: ^complex128, ltb: i32, ipiv: ^i32, ipiv2: ^i32, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_zsytrs_aa_2stage_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, tb: ^complex128, ltb: i32, ipiv: ^i32, ipiv2: ^i32, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_chetrs_aa_2stage :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, tb: ^complex64, ltb: i32, ipiv: ^i32, ipiv2: ^i32, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_chetrs_aa_2stage_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex64, lda: i32, tb: ^complex64, ltb: i32, ipiv: ^i32, ipiv2: ^i32, b: ^complex64, ldb: i32) -> i32 ---
	LAPACKE_zhetrs_aa_2stage :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, tb: ^complex128, ltb: i32, ipiv: ^i32, ipiv2: ^i32, b: ^complex128, ldb: i32) -> i32 ---
	LAPACKE_zhetrs_aa_2stage_work :: proc(matrix_layout: c.int, uplo: c.char, n: i32, nrhs: i32, a: ^complex128, lda: i32, tb: ^complex128, ltb: i32, ipiv: ^i32, ipiv2: ^i32, b: ^complex128, ldb: i32) -> i32 ---

	//LAPACK 3.10.0
	LAPACKE_sorhr_col :: proc(matrix_layout: c.int, m: i32, n: i32, nb: i32, a: ^f32, lda: i32, t: ^f32, ldt: i32, d: ^f32) -> i32 ---
	LAPACKE_sorhr_col_work :: proc(matrix_layout: c.int, m: i32, n: i32, nb: i32, a: ^f32, lda: i32, t: ^f32, ldt: i32, d: ^f32) -> i32 ---
	LAPACKE_dorhr_col :: proc(matrix_layout: c.int, m: i32, n: i32, nb: i32, a: ^f64, lda: i32, t: ^f64, ldt: i32, d: ^f64) -> i32 ---
	LAPACKE_dorhr_col_work :: proc(matrix_layout: c.int, m: i32, n: i32, nb: i32, a: ^f64, lda: i32, t: ^f64, ldt: i32, d: ^f64) -> i32 ---
	LAPACKE_cunhr_col :: proc(matrix_layout: c.int, m: i32, n: i32, nb: i32, a: ^complex64, lda: i32, t: ^complex64, ldt: i32, d: ^complex64) -> i32 ---
	LAPACKE_cunhr_col_work :: proc(matrix_layout: c.int, m: i32, n: i32, nb: i32, a: ^complex64, lda: i32, t: ^complex64, ldt: i32, d: ^complex64) -> i32 ---
	LAPACKE_zunhr_col :: proc(matrix_layout: c.int, m: i32, n: i32, nb: i32, a: ^complex128, lda: i32, t: ^complex128, ldt: i32, d: ^complex128) -> i32 ---
	LAPACKE_zunhr_col_work :: proc(matrix_layout: c.int, m: i32, n: i32, nb: i32, a: ^complex128, lda: i32, t: ^complex128, ldt: i32, d: ^complex128) -> i32 ---

	/* APIs for set/get nancheck flags */
	LAPACKE_set_nancheck :: proc(flag: c.int) ---
	LAPACKE_get_nancheck :: proc() -> c.int ---
}
