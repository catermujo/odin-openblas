package f77

import "core:c"

when ODIN_OS == .Windows {
	foreign import lib "../../vendor/linalg/windows-x64/lib/openblas64.lib"
} else when ODIN_OS == .Linux {
	// Use ILP64 version of OpenBLAS (64-bit integers)
	foreign import lib "system:openblas64"
}

// LAPACK_H ::

// LAPACK_FORTRAN_STRLEN_END ::

FORTRAN_STRLEN :: c.size_t

lapack_float_return :: f32

/* Callback logical functions of one, two, or three arguments are used
*  to select eigenvalues to sort to the top left of the Schur form.
*  The value is selected if function returns TRUE (non-zero). */
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
	// ===================================================================================
	// UTILITY FUNCTIONS
	// ===================================================================================

	lsame_ :: proc(ca: cstring, cb: cstring, lca: i32, lcb: i32, _: c.size_t, _: c.size_t) -> i32 ---

	// ===================================================================================
	// BIDIAGONAL MATRIX COMPUTATIONS
	// CS decomposition, divide-and-conquer SVD, QR iteration, selective SVD
	// ===================================================================================

	cbbcsd_ :: proc(jobu1: cstring, jobu2: cstring, jobv1t: cstring, jobv2t: cstring, trans: cstring, m: ^blasint, p: ^blasint, q: ^blasint, theta: ^f32, phi: ^f32, U1: ^complex64, ldu1: ^blasint, U2: ^complex64, ldu2: ^blasint, V1T: ^complex64, ldv1t: ^blasint, V2T: ^complex64, ldv2t: ^blasint, B11D: ^f32, B11E: ^f32, B12D: ^f32, B12E: ^f32, B21D: ^f32, B21E: ^f32, B22D: ^f32, B22E: ^f32, rwork: ^f32, lrwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t) ---
	dbbcsd_ :: proc(jobu1: cstring, jobu2: cstring, jobv1t: cstring, jobv2t: cstring, trans: cstring, m: ^blasint, p: ^blasint, q: ^blasint, theta: ^f64, phi: ^f64, U1: ^f64, ldu1: ^blasint, U2: ^f64, ldu2: ^blasint, V1T: ^f64, ldv1t: ^blasint, V2T: ^f64, ldv2t: ^blasint, B11D: ^f64, B11E: ^f64, B12D: ^f64, B12E: ^f64, b21d: ^f64, b21e: ^f64, b22d: ^f64, b22e: ^f64, work: ^f64, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t) ---
	sbbcsd_ :: proc(jobu1: cstring, jobu2: cstring, jobv1t: cstring, jobv2t: cstring, trans: cstring, m: ^blasint, p: ^blasint, q: ^blasint, theta: ^f32, phi: ^f32, U1: ^f32, ldu1: ^blasint, U2: ^f32, ldu2: ^blasint, V1T: ^f32, ldv1t: ^blasint, V2T: ^f32, ldv2t: ^blasint, B11D: ^f32, B11E: ^f32, B12D: ^f32, B12E: ^f32, B21D: ^f32, B21E: ^f32, B22D: ^f32, B22E: ^f32, work: ^f32, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t) ---
	zbbcsd_ :: proc(jobu1: cstring, jobu2: cstring, jobv1t: cstring, jobv2t: cstring, trans: cstring, m: ^blasint, p: ^blasint, q: ^blasint, theta: ^f64, phi: ^f64, U1: ^complex128, ldu1: ^blasint, U2: ^complex128, ldu2: ^blasint, V1T: ^complex128, ldv1t: ^blasint, V2T: ^complex128, ldv2t: ^blasint, B11D: ^f64, B11E: ^f64, B12D: ^f64, B12E: ^f64, B21D: ^f64, B21E: ^f64, B22D: ^f64, B22E: ^f64, rwork: ^f64, lrwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t) ---

	dbdsdc_ :: proc(uplo: cstring, compq: cstring, n: ^blasint, D: ^f64, E: ^f64, U: ^f64, ldu: ^blasint, VT: ^f64, ldvt: ^blasint, Q: ^f64, IQ: ^blasint, work: ^f64, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	sbdsdc_ :: proc(uplo: cstring, compq: cstring, n: ^blasint, D: ^f32, E: ^f32, U: ^f32, ldu: ^blasint, VT: ^f32, ldvt: ^blasint, Q: ^f32, IQ: ^blasint, work: ^f32, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	cbdsqr_ :: proc(uplo: cstring, n: ^blasint, ncvt: ^blasint, nru: ^blasint, ncc: ^blasint, D: ^f32, E: ^f32, VT: ^complex64, ldvt: ^blasint, U: ^complex64, ldu: ^blasint, C: ^complex64, ldc: ^blasint, rwork: ^f32, info: ^Info, _: c.size_t) ---
	dbdsqr_ :: proc(uplo: cstring, n: ^blasint, ncvt: ^blasint, nru: ^blasint, ncc: ^blasint, D: ^f64, E: ^f64, VT: ^f64, ldvt: ^blasint, U: ^f64, ldu: ^blasint, C: ^f64, ldc: ^blasint, work: ^f64, info: ^Info, _: c.size_t) ---
	sbdsqr_ :: proc(uplo: cstring, n: ^blasint, ncvt: ^blasint, nru: ^blasint, ncc: ^blasint, D: ^f32, E: ^f32, VT: ^f32, ldvt: ^blasint, U: ^f32, ldu: ^blasint, C: ^f32, ldc: ^blasint, work: ^f32, info: ^Info, _: c.size_t) ---
	zbdsqr_ :: proc(uplo: cstring, n: ^blasint, ncvt: ^blasint, nru: ^blasint, ncc: ^blasint, D: ^f64, E: ^f64, VT: ^complex128, ldvt: ^blasint, U: ^complex128, ldu: ^blasint, C: ^complex128, ldc: ^blasint, rwork: ^f64, info: ^Info, _: c.size_t) ---

	dbdsvdx_ :: proc(uplo: cstring, jobz: cstring, range: cstring, n: ^blasint, D: ^f64, E: ^f64, vl: ^f64, vu: ^f64, il: ^blasint, iu: ^blasint, ns: ^blasint, S: ^f64, Z: ^f64, ldz: ^blasint, work: ^f64, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	sbdsvdx_ :: proc(uplo: cstring, jobz: cstring, range: cstring, n: ^blasint, D: ^f32, E: ^f32, vl: ^f32, vu: ^f32, il: ^blasint, iu: ^blasint, ns: ^blasint, S: ^f32, Z: ^f32, ldz: ^blasint, work: ^f32, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---

	// Condition number estimation
	ddisna_ :: proc(job: cstring, m: ^blasint, n: ^blasint, D: ^f64, SEP: ^f64, info: ^Info, _: c.size_t) ---
	sdisna_ :: proc(job: cstring, m: ^blasint, n: ^blasint, D: ^f32, SEP: ^f32, info: ^Info, _: c.size_t) ---

	// ===================================================================================
	// GENERAL BANDED MATRIX ROUTINES (GB)
	// Factorization, solvers, condition estimation, equilibration
	// ===================================================================================

	// Bidiagonalization
	cgbbrd_ :: proc(vect: cstring, m: ^blasint, n: ^blasint, ncc: ^blasint, kl: ^blasint, ku: ^blasint, AB: ^complex64, ldab: ^blasint, D: ^f32, E: ^f32, Q: ^complex64, ldq: ^blasint, PT: ^complex64, ldpt: ^blasint, C: ^complex64, ldc: ^blasint, work: ^complex64, rwork: ^f32, info: ^Info, _: c.size_t) ---
	dgbbrd_ :: proc(vect: cstring, m: ^blasint, n: ^blasint, ncc: ^blasint, kl: ^blasint, ku: ^blasint, AB: ^f64, ldab: ^blasint, D: ^f64, E: ^f64, Q: ^f64, ldq: ^blasint, PT: ^f64, ldpt: ^blasint, C: ^f64, ldc: ^blasint, work: ^f64, info: ^Info, _: c.size_t) ---
	sgbbrd_ :: proc(vect: cstring, m: ^blasint, n: ^blasint, ncc: ^blasint, kl: ^blasint, ku: ^blasint, AB: ^f32, ldab: ^blasint, D: ^f32, E: ^f32, Q: ^f32, ldq: ^blasint, PT: ^f32, ldpt: ^blasint, C: ^f32, ldc: ^blasint, work: ^f32, info: ^Info, _: c.size_t) ---
	zgbbrd_ :: proc(vect: cstring, m: ^blasint, n: ^blasint, ncc: ^blasint, kl: ^blasint, ku: ^blasint, AB: ^complex128, ldab: ^blasint, D: ^f64, E: ^f64, Q: ^complex128, ldq: ^blasint, PT: ^complex128, ldpt: ^blasint, C: ^complex128, ldc: ^blasint, work: ^complex128, rwork: ^f64, info: ^Info, _: c.size_t) ---

	// Condition number estimation
	cgbcon_ :: proc(norm: cstring, n: ^blasint, kl: ^blasint, ku: ^blasint, AB: ^complex64, ldab: ^blasint, ipiv: ^blasint, anorm: ^f32, rcond: ^f32, work: ^complex64, rwork: ^f32, info: ^Info, _: c.size_t) ---
	dgbcon_ :: proc(norm: cstring, n: ^blasint, kl: ^blasint, ku: ^blasint, AB: ^f64, ldab: ^blasint, ipiv: ^blasint, anorm: ^f64, rcond: ^f64, work: ^f64, iwork: ^blasint, info: ^Info, _: c.size_t) ---
	sgbcon_ :: proc(norm: cstring, n: ^blasint, kl: ^blasint, ku: ^blasint, AB: ^f32, ldab: ^blasint, ipiv: ^blasint, anorm: ^f32, rcond: ^f32, work: ^f32, iwork: ^blasint, info: ^Info, _: c.size_t) ---
	zgbcon_ :: proc(norm: cstring, n: ^blasint, kl: ^blasint, ku: ^blasint, AB: ^complex128, ldab: ^blasint, ipiv: ^blasint, anorm: ^f64, rcond: ^f64, work: ^complex128, rwork: ^f64, info: ^Info, _: c.size_t) ---

	// Equilibration
	cgbequ_ :: proc(m: ^blasint, n: ^blasint, kl: ^blasint, ku: ^blasint, AB: ^complex64, ldab: ^blasint, R: ^f32, C: ^f32, rowcnd: ^f32, colcnd: ^f32, amax: ^f32, info: ^Info) ---
	dgbequ_ :: proc(m: ^blasint, n: ^blasint, kl: ^blasint, ku: ^blasint, AB: ^f64, ldab: ^blasint, R: ^f64, C: ^f64, rowcnd: ^f64, colcnd: ^f64, amax: ^f64, info: ^Info) ---
	sgbequ_ :: proc(m: ^blasint, n: ^blasint, kl: ^blasint, ku: ^blasint, AB: ^f32, ldab: ^blasint, R: ^f32, C: ^f32, rowcnd: ^f32, colcnd: ^f32, amax: ^f32, info: ^Info) ---
	zgbequ_ :: proc(m: ^blasint, n: ^blasint, kl: ^blasint, ku: ^blasint, AB: ^complex128, ldab: ^blasint, R: ^f64, C: ^f64, rowcnd: ^f64, colcnd: ^f64, amax: ^f64, info: ^Info) ---
	cgbequb_ :: proc(m: ^blasint, n: ^blasint, kl: ^blasint, ku: ^blasint, AB: ^complex64, ldab: ^blasint, R: ^f32, C: ^f32, rowcnd: ^f32, colcnd: ^f32, amax: ^f32, info: ^Info) ---
	dgbequb_ :: proc(m: ^blasint, n: ^blasint, kl: ^blasint, ku: ^blasint, AB: ^f64, ldab: ^blasint, R: ^f64, C: ^f64, rowcnd: ^f64, colcnd: ^f64, amax: ^f64, info: ^Info) ---
	sgbequb_ :: proc(m: ^blasint, n: ^blasint, kl: ^blasint, ku: ^blasint, AB: ^f32, ldab: ^blasint, R: ^f32, C: ^f32, rowcnd: ^f32, colcnd: ^f32, amax: ^f32, info: ^Info) ---
	zgbequb_ :: proc(m: ^blasint, n: ^blasint, kl: ^blasint, ku: ^blasint, AB: ^complex128, ldab: ^blasint, R: ^f64, C: ^f64, rowcnd: ^f64, colcnd: ^f64, amax: ^f64, info: ^Info) ---

	// Iterative refinement
	cgbrfs_ :: proc(trans: cstring, n: ^blasint, kl: ^blasint, ku: ^blasint, nrhs: ^blasint, AB: ^complex64, ldab: ^blasint, AFB: ^complex64, ldafb: ^blasint, ipiv: ^blasint, B: ^complex64, ldb: ^blasint, X: ^complex64, ldx: ^blasint, ferr: ^f32, berr: ^f32, work: ^complex64, rwork: ^f32, info: ^Info, _: c.size_t) ---
	dgbrfs_ :: proc(trans: cstring, n: ^blasint, kl: ^blasint, ku: ^blasint, nrhs: ^blasint, AB: ^f64, ldab: ^blasint, AFB: ^f64, ldafb: ^blasint, ipiv: ^blasint, B: ^f64, ldb: ^blasint, X: ^f64, ldx: ^blasint, ferr: ^f64, berr: ^f64, work: ^f64, iwork: ^blasint, info: ^Info, _: c.size_t) ---
	sgbrfs_ :: proc(trans: cstring, n: ^blasint, kl: ^blasint, ku: ^blasint, nrhs: ^blasint, AB: ^f32, ldab: ^blasint, AFB: ^f32, ldafb: ^blasint, ipiv: ^blasint, B: ^f32, ldb: ^blasint, X: ^f32, ldx: ^blasint, ferr: ^f32, berr: ^f32, work: ^f32, iwork: ^blasint, info: ^Info, _: c.size_t) ---
	zgbrfs_ :: proc(trans: cstring, n: ^blasint, kl: ^blasint, ku: ^blasint, nrhs: ^blasint, AB: ^complex128, ldab: ^blasint, AFB: ^complex128, ldafb: ^blasint, ipiv: ^blasint, B: ^complex128, ldb: ^blasint, X: ^complex128, ldx: ^blasint, ferr: ^f64, berr: ^f64, work: ^complex128, rwork: ^f64, info: ^Info, _: c.size_t) ---
	cgbrfsx_ :: proc(trans: cstring, equed: cstring, n: ^blasint, kl: ^blasint, ku: ^blasint, nrhs: ^blasint, AB: ^complex64, ldab: ^blasint, AFB: ^complex64, ldafb: ^blasint, ipiv: ^blasint, R: ^f32, C: ^f32, B: ^complex64, ldb: ^blasint, X: ^complex64, ldx: ^blasint, rcond: ^f32, berr: ^f32, n_err_bnds: ^blasint, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: ^blasint, params: ^f32, work: ^complex64, rwork: ^f32, info: ^Info, _: c.size_t, _: c.size_t) ---
	dgbrfsx_ :: proc(trans: cstring, equed: cstring, n: ^blasint, kl: ^blasint, ku: ^blasint, nrhs: ^blasint, AB: ^f64, ldab: ^blasint, AFB: ^f64, ldafb: ^blasint, ipiv: ^blasint, R: ^f64, C: ^f64, B: ^f64, ldb: ^blasint, X: ^f64, ldx: ^blasint, rcond: ^f64, berr: ^f64, n_err_bnds: ^blasint, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: ^blasint, params: ^f64, work: ^f64, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	sgbrfsx_ :: proc(trans: cstring, equed: cstring, n: ^blasint, kl: ^blasint, ku: ^blasint, nrhs: ^blasint, AB: ^f32, ldab: ^blasint, AFB: ^f32, ldafb: ^blasint, ipiv: ^blasint, R: ^f32, C: ^f32, B: ^f32, ldb: ^blasint, X: ^f32, ldx: ^blasint, rcond: ^f32, berr: ^f32, n_err_bnds: ^blasint, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: ^blasint, params: ^f32, work: ^f32, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	zgbrfsx_ :: proc(trans: cstring, equed: cstring, n: ^blasint, kl: ^blasint, ku: ^blasint, nrhs: ^blasint, AB: ^complex128, ldab: ^blasint, AFB: ^complex128, ldafb: ^blasint, ipiv: ^blasint, R: ^f64, C: ^f64, B: ^complex128, ldb: ^blasint, X: ^complex128, ldx: ^blasint, rcond: ^f64, berr: ^f64, n_err_bnds: ^blasint, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: ^blasint, params: ^f64, work: ^complex128, rwork: ^f64, info: ^Info, _: c.size_t, _: c.size_t) ---

	// Linear system solvers
	cgbsv_ :: proc(n: ^blasint, kl: ^blasint, ku: ^blasint, nrhs: ^blasint, AB: ^complex64, ldab: ^blasint, ipiv: ^blasint, B: ^complex64, ldb: ^blasint, info: ^Info) ---
	dgbsv_ :: proc(n: ^blasint, kl: ^blasint, ku: ^blasint, nrhs: ^blasint, AB: ^f64, ldab: ^blasint, ipiv: ^blasint, B: ^f64, ldb: ^blasint, info: ^Info) ---
	sgbsv_ :: proc(n: ^blasint, kl: ^blasint, ku: ^blasint, nrhs: ^blasint, AB: ^f32, ldab: ^blasint, ipiv: ^blasint, B: ^f32, ldb: ^blasint, info: ^Info) ---
	zgbsv_ :: proc(n: ^blasint, kl: ^blasint, ku: ^blasint, nrhs: ^blasint, AB: ^complex128, ldab: ^blasint, ipiv: ^blasint, B: ^complex128, ldb: ^blasint, info: ^Info) ---
	cgbsvx_ :: proc(fact: cstring, trans: cstring, n: ^blasint, kl: ^blasint, ku: ^blasint, nrhs: ^blasint, AB: ^complex64, ldab: ^blasint, AFB: ^complex64, ldafb: ^blasint, ipiv: ^blasint, equed: ^byte, R: ^f32, C: ^f32, B: ^complex64, ldb: ^blasint, X: ^complex64, ldx: ^blasint, rcond: ^f32, ferr: ^f32, berr: ^f32, work: ^complex64, rwork: ^f32, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	dgbsvx_ :: proc(fact: cstring, trans: cstring, n: ^blasint, kl: ^blasint, ku: ^blasint, nrhs: ^blasint, AB: ^f64, ldab: ^blasint, AFB: ^f64, ldafb: ^blasint, ipiv: ^blasint, equed: ^byte, R: ^f64, C: ^f64, B: ^f64, ldb: ^blasint, X: ^f64, ldx: ^blasint, rcond: ^f64, ferr: ^f64, berr: ^f64, work: ^f64, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	sgbsvx_ :: proc(fact: cstring, trans: cstring, n: ^blasint, kl: ^blasint, ku: ^blasint, nrhs: ^blasint, AB: ^f32, ldab: ^blasint, AFB: ^f32, ldafb: ^blasint, ipiv: ^blasint, equed: ^byte, R: ^f32, C: ^f32, B: ^f32, ldb: ^blasint, X: ^f32, ldx: ^blasint, rcond: ^f32, ferr: ^f32, berr: ^f32, work: ^f32, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	zgbsvx_ :: proc(fact: cstring, trans: cstring, n: ^blasint, kl: ^blasint, ku: ^blasint, nrhs: ^blasint, AB: ^complex128, ldab: ^blasint, AFB: ^complex128, ldafb: ^blasint, ipiv: ^blasint, equed: ^byte, R: ^f64, C: ^f64, B: ^complex128, ldb: ^blasint, X: ^complex128, ldx: ^blasint, rcond: ^f64, ferr: ^f64, berr: ^f64, work: ^complex128, rwork: ^f64, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	cgbsvxx_ :: proc(fact: cstring, trans: cstring, n: ^blasint, kl: ^blasint, ku: ^blasint, nrhs: ^blasint, AB: ^complex64, ldab: ^blasint, AFB: ^complex64, ldafb: ^blasint, ipiv: ^blasint, equed: ^byte, R: ^f32, C: ^f32, B: ^complex64, ldb: ^blasint, X: ^complex64, ldx: ^blasint, rcond: ^f32, rpvgrw: ^f32, berr: ^f32, n_err_bnds: ^blasint, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: ^blasint, params: ^f32, work: ^complex64, rwork: ^f32, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	dgbsvxx_ :: proc(fact: cstring, trans: cstring, n: ^blasint, kl: ^blasint, ku: ^blasint, nrhs: ^blasint, AB: ^f64, ldab: ^blasint, AFB: ^f64, ldafb: ^blasint, ipiv: ^blasint, equed: ^byte, R: ^f64, C: ^f64, B: ^f64, ldb: ^blasint, X: ^f64, ldx: ^blasint, rcond: ^f64, rpvgrw: ^f64, berr: ^f64, n_err_bnds: ^blasint, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: ^blasint, params: ^f64, work: ^f64, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	sgbsvxx_ :: proc(fact: cstring, trans: cstring, n: ^blasint, kl: ^blasint, ku: ^blasint, nrhs: ^blasint, AB: ^f32, ldab: ^blasint, AFB: ^f32, ldafb: ^blasint, ipiv: ^blasint, equed: ^byte, R: ^f32, C: ^f32, B: ^f32, ldb: ^blasint, X: ^f32, ldx: ^blasint, rcond: ^f32, rpvgrw: ^f32, berr: ^f32, n_err_bnds: ^blasint, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: ^blasint, params: ^f32, work: ^f32, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	zgbsvxx_ :: proc(fact: cstring, trans: cstring, n: ^blasint, kl: ^blasint, ku: ^blasint, nrhs: ^blasint, AB: ^complex128, ldab: ^blasint, AFB: ^complex128, ldafb: ^blasint, ipiv: ^blasint, equed: ^byte, R: ^f64, C: ^f64, B: ^complex128, ldb: ^blasint, X: ^complex128, ldx: ^blasint, rcond: ^f64, rpvgrw: ^f64, berr: ^f64, n_err_bnds: ^blasint, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: ^blasint, params: ^f64, work: ^complex128, rwork: ^f64, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---

	// LU factorization
	cgbtrf_ :: proc(m: ^blasint, n: ^blasint, kl: ^blasint, ku: ^blasint, AB: ^complex64, ldab: ^blasint, ipiv: ^blasint, info: ^Info) ---
	dgbtrf_ :: proc(m: ^blasint, n: ^blasint, kl: ^blasint, ku: ^blasint, AB: ^f64, ldab: ^blasint, ipiv: ^blasint, info: ^Info) ---
	sgbtrf_ :: proc(m: ^blasint, n: ^blasint, kl: ^blasint, ku: ^blasint, AB: ^f32, ldab: ^blasint, ipiv: ^blasint, info: ^Info) ---
	zgbtrf_ :: proc(m: ^blasint, n: ^blasint, kl: ^blasint, ku: ^blasint, AB: ^complex128, ldab: ^blasint, ipiv: ^blasint, info: ^Info) ---

	// Solve using LU factorization
	cgbtrs_ :: proc(trans: cstring, n: ^blasint, kl: ^blasint, ku: ^blasint, nrhs: ^blasint, AB: ^complex64, ldab: ^blasint, ipiv: ^blasint, B: ^complex64, ldb: ^blasint, info: ^Info, _: c.size_t) ---
	dgbtrs_ :: proc(trans: cstring, n: ^blasint, kl: ^blasint, ku: ^blasint, nrhs: ^blasint, AB: ^f64, ldab: ^blasint, ipiv: ^blasint, B: ^f64, ldb: ^blasint, info: ^Info, _: c.size_t) ---
	sgbtrs_ :: proc(trans: cstring, n: ^blasint, kl: ^blasint, ku: ^blasint, nrhs: ^blasint, AB: ^f32, ldab: ^blasint, ipiv: ^blasint, B: ^f32, ldb: ^blasint, info: ^Info, _: c.size_t) ---
	zgbtrs_ :: proc(trans: cstring, n: ^blasint, kl: ^blasint, ku: ^blasint, nrhs: ^blasint, AB: ^complex128, ldab: ^blasint, ipiv: ^blasint, B: ^complex128, ldb: ^blasint, info: ^Info, _: c.size_t) ---

	// ===================================================================================
	// GENERAL MATRIX ROUTINES (GE)
	// Balancing, factorization, solvers, eigenvalues, least squares
	// ===================================================================================

	// Balancing (back-transformation)
	cgebak_ :: proc(job: cstring, side: cstring, n: ^blasint, ilo: ^blasint, ihi: ^blasint, scale: ^f32, m: ^blasint, V: ^complex64, ldv: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	dgebak_ :: proc(job: cstring, side: cstring, n: ^blasint, ilo: ^blasint, ihi: ^blasint, scale: ^f64, m: ^blasint, V: ^f64, ldv: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	sgebak_ :: proc(job: cstring, side: cstring, n: ^blasint, ilo: ^blasint, ihi: ^blasint, scale: ^f32, m: ^blasint, V: ^f32, ldv: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	zgebak_ :: proc(job: cstring, side: cstring, n: ^blasint, ilo: ^blasint, ihi: ^blasint, scale: ^f64, m: ^blasint, V: ^complex128, ldv: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---

	// Balancing
	cgebal_ :: proc(job: cstring, n: ^blasint, A: ^complex64, lda: ^blasint, ilo: ^blasint, ihi: ^blasint, scale: ^f32, info: ^Info, _: c.size_t) ---
	dgebal_ :: proc(job: cstring, n: ^blasint, A: ^f64, lda: ^blasint, ilo: ^blasint, ihi: ^blasint, scale: ^f64, info: ^Info, _: c.size_t) ---
	sgebal_ :: proc(job: cstring, n: ^blasint, A: ^f32, lda: ^blasint, ilo: ^blasint, ihi: ^blasint, scale: ^f32, info: ^Info, _: c.size_t) ---
	zgebal_ :: proc(job: cstring, n: ^blasint, A: ^complex128, lda: ^blasint, ilo: ^blasint, ihi: ^blasint, scale: ^f64, info: ^Info, _: c.size_t) ---


	// YOU ARE HERE


	// Bidiagonalization
	cgebrd_ :: proc(m: ^blasint, n: ^blasint, A: ^complex64, lda: ^blasint, D: ^f32, E: ^f32, tauq: ^complex64, taup: ^complex64, work: ^complex64, lwork: ^blasint, info: ^Info) ---
	dgebrd_ :: proc(m: ^blasint, n: ^blasint, A: ^f64, lda: ^blasint, D: ^f64, E: ^f64, tauq: ^f64, taup: ^f64, work: ^f64, lwork: ^blasint, info: ^Info) ---
	sgebrd_ :: proc(m: ^blasint, n: ^blasint, A: ^f32, lda: ^blasint, D: ^f32, E: ^f32, tauq: ^f32, taup: ^f32, work: ^f32, lwork: ^blasint, info: ^Info) ---
	zgebrd_ :: proc(m: ^blasint, n: ^blasint, A: ^complex128, lda: ^blasint, D: ^f64, E: ^f64, tauq: ^complex128, taup: ^complex128, work: ^complex128, lwork: ^blasint, info: ^Info) ---

	// Condition number estimation
	cgecon_ :: proc(norm: cstring, n: ^blasint, A: ^complex64, lda: ^blasint, anorm: ^f32, rcond: ^f32, work: ^complex64, rwork: ^f32, info: ^Info, _: c.size_t) ---
	dgecon_ :: proc(norm: cstring, n: ^blasint, A: ^f64, lda: ^blasint, anorm: ^f64, rcond: ^f64, work: ^f64, iwork: ^blasint, info: ^Info, _: c.size_t) ---
	sgecon_ :: proc(norm: cstring, n: ^blasint, A: ^f32, lda: ^blasint, anorm: ^f32, rcond: ^f32, work: ^f32, iwork: ^blasint, info: ^Info, _: c.size_t) ---
	zgecon_ :: proc(norm: cstring, n: ^blasint, A: ^complex128, lda: ^blasint, anorm: ^f64, rcond: ^f64, work: ^complex128, rwork: ^f64, info: ^Info, _: c.size_t) ---

	// Equilibration
	cgeequ_ :: proc(m: ^blasint, n: ^blasint, A: ^complex64, lda: ^blasint, R: ^f32, C: ^f32, rowcnd: ^f32, colcnd: ^f32, amax: ^f32, info: ^Info) ---
	dgeequ_ :: proc(m: ^blasint, n: ^blasint, A: ^f64, lda: ^blasint, R: ^f64, C: ^f64, rowcnd: ^f64, colcnd: ^f64, amax: ^f64, info: ^Info) ---
	sgeequ_ :: proc(m: ^blasint, n: ^blasint, A: ^f32, lda: ^blasint, R: ^f32, C: ^f32, rowcnd: ^f32, colcnd: ^f32, amax: ^f32, info: ^Info) ---
	zgeequ_ :: proc(m: ^blasint, n: ^blasint, A: ^complex128, lda: ^blasint, R: ^f64, C: ^f64, rowcnd: ^f64, colcnd: ^f64, amax: ^f64, info: ^Info) ---
	cgeequb_ :: proc(m: ^blasint, n: ^blasint, A: ^complex64, lda: ^blasint, R: ^f32, C: ^f32, rowcnd: ^f32, colcnd: ^f32, amax: ^f32, info: ^Info) ---
	dgeequb_ :: proc(m: ^blasint, n: ^blasint, A: ^f64, lda: ^blasint, R: ^f64, C: ^f64, rowcnd: ^f64, colcnd: ^f64, amax: ^f64, info: ^Info) ---
	sgeequb_ :: proc(m: ^blasint, n: ^blasint, A: ^f32, lda: ^blasint, R: ^f32, C: ^f32, rowcnd: ^f32, colcnd: ^f32, amax: ^f32, info: ^Info) ---
	zgeequb_ :: proc(m: ^blasint, n: ^blasint, A: ^complex128, lda: ^blasint, R: ^f64, C: ^f64, rowcnd: ^f64, colcnd: ^f64, amax: ^f64, info: ^Info) ---

	// Eigenvalue problems - Schur decomposition
	cgees_ :: proc(jobvs: cstring, sort: cstring, select: LAPACK_C_SELECT1, n: ^blasint, A: ^complex64, lda: ^blasint, sdim: ^blasint, W: ^complex64, VS: ^complex64, ldvs: ^blasint, work: ^complex64, lwork: ^blasint, rwork: ^f32, BWORK: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	dgees_ :: proc(jobvs: cstring, sort: cstring, select: LAPACK_D_SELECT2, n: ^blasint, A: ^f64, lda: ^blasint, sdim: ^blasint, WR: ^f64, WI: ^f64, VS: ^f64, ldvs: ^blasint, work: ^f64, lwork: ^blasint, BWORK: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	sgees_ :: proc(jobvs: cstring, sort: cstring, select: LAPACK_S_SELECT2, n: ^blasint, A: ^f32, lda: ^blasint, sdim: ^blasint, WR: ^f32, WI: ^f32, VS: ^f32, ldvs: ^blasint, work: ^f32, lwork: ^blasint, BWORK: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	zgees_ :: proc(jobvs: cstring, sort: cstring, select: LAPACK_Z_SELECT1, n: ^blasint, A: ^complex128, lda: ^blasint, sdim: ^blasint, W: ^complex128, VS: ^complex128, ldvs: ^blasint, work: ^complex128, lwork: ^blasint, rwork: ^f64, BWORK: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---

	cgeesx_ :: proc(jobvs: cstring, sort: cstring, select: LAPACK_C_SELECT1, sense: cstring, n: ^blasint, A: ^complex64, lda: ^blasint, sdim: ^blasint, W: ^complex64, VS: ^complex64, ldvs: ^blasint, rconde: ^f32, rcondv: ^f32, work: ^complex64, lwork: ^blasint, rwork: ^f32, BWORK: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	dgeesx_ :: proc(jobvs: cstring, sort: cstring, select: LAPACK_D_SELECT2, sense: cstring, n: ^blasint, A: ^f64, lda: ^blasint, sdim: ^blasint, WR: ^f64, WI: ^f64, VS: ^f64, ldvs: ^blasint, rconde: ^f64, rcondv: ^f64, work: ^f64, lwork: ^blasint, iwork: ^blasint, liwork: ^blasint, BWORK: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	sgeesx_ :: proc(jobvs: cstring, sort: cstring, select: LAPACK_S_SELECT2, sense: cstring, n: ^blasint, A: ^f32, lda: ^blasint, sdim: ^blasint, WR: ^f32, WI: ^f32, VS: ^f32, ldvs: ^blasint, rconde: ^f32, rcondv: ^f32, work: ^f32, lwork: ^blasint, iwork: ^blasint, liwork: ^blasint, BWORK: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	zgeesx_ :: proc(jobvs: cstring, sort: cstring, select: LAPACK_Z_SELECT1, sense: cstring, n: ^blasint, A: ^complex128, lda: ^blasint, sdim: ^blasint, W: ^complex128, VS: ^complex128, ldvs: ^blasint, rconde: ^f64, rcondv: ^f64, work: ^complex128, lwork: ^blasint, rwork: ^f64, BWORK: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---

	// Eigenvalue problems - eigenvalues and eigenvectors
	cgeev_ :: proc(jobvl: cstring, jobvr: cstring, n: ^blasint, A: ^complex64, lda: ^blasint, W: ^complex64, VL: ^complex64, ldvl: ^blasint, VR: ^complex64, ldvr: ^blasint, work: ^complex64, lwork: ^blasint, rwork: ^f32, info: ^Info, _: c.size_t, _: c.size_t) ---
	dgeev_ :: proc(jobvl: cstring, jobvr: cstring, n: ^blasint, A: ^f64, lda: ^blasint, WR: ^f64, WI: ^f64, VL: ^f64, ldvl: ^blasint, VR: ^f64, ldvr: ^blasint, work: ^f64, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	sgeev_ :: proc(jobvl: cstring, jobvr: cstring, n: ^blasint, A: ^f32, lda: ^blasint, WR: ^f32, WI: ^f32, VL: ^f32, ldvl: ^blasint, VR: ^f32, ldvr: ^blasint, work: ^f32, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	zgeev_ :: proc(jobvl: cstring, jobvr: cstring, n: ^blasint, A: ^complex128, lda: ^blasint, W: ^complex128, VL: ^complex128, ldvl: ^blasint, VR: ^complex128, ldvr: ^blasint, work: ^complex128, lwork: ^blasint, rwork: ^f64, info: ^Info, _: c.size_t, _: c.size_t) ---

	cgeevx_ :: proc(balanc: cstring, jobvl: cstring, jobvr: cstring, sense: cstring, n: ^blasint, A: ^complex64, lda: ^blasint, W: ^complex64, VL: ^complex64, ldvl: ^blasint, VR: ^complex64, ldvr: ^blasint, ilo: ^blasint, ihi: ^blasint, scale: ^f32, abnrm: ^f32, rconde: ^f32, rcondv: ^f32, work: ^complex64, lwork: ^blasint, rwork: ^f32, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t) ---
	dgeevx_ :: proc(balanc: cstring, jobvl: cstring, jobvr: cstring, sense: cstring, n: ^blasint, A: ^f64, lda: ^blasint, WR: ^f64, WI: ^f64, VL: ^f64, ldvl: ^blasint, VR: ^f64, ldvr: ^blasint, ilo: ^blasint, ihi: ^blasint, scale: ^f64, abnrm: ^f64, rconde: ^f64, rcondv: ^f64, work: ^f64, lwork: ^blasint, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t) ---
	sgeevx_ :: proc(balanc: cstring, jobvl: cstring, jobvr: cstring, sense: cstring, n: ^blasint, A: ^f32, lda: ^blasint, WR: ^f32, WI: ^f32, VL: ^f32, ldvl: ^blasint, VR: ^f32, ldvr: ^blasint, ilo: ^blasint, ihi: ^blasint, scale: ^f32, abnrm: ^f32, rconde: ^f32, rcondv: ^f32, work: ^f32, lwork: ^blasint, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t) ---
	zgeevx_ :: proc(balanc: cstring, jobvl: cstring, jobvr: cstring, sense: cstring, n: ^blasint, A: ^complex128, lda: ^blasint, W: ^complex128, VL: ^complex128, ldvl: ^blasint, VR: ^complex128, ldvr: ^blasint, ilo: ^blasint, ihi: ^blasint, scale: ^f64, abnrm: ^f64, rconde: ^f64, rcondv: ^f64, work: ^complex128, lwork: ^blasint, rwork: ^f64, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t) ---

	// Hessenberg reduction
	cgehrd_ :: proc(n: ^blasint, ilo: ^blasint, ihi: ^blasint, A: ^complex64, lda: ^blasint, tau: ^complex64, work: ^complex64, lwork: ^blasint, info: ^Info) ---
	dgehrd_ :: proc(n: ^blasint, ilo: ^blasint, ihi: ^blasint, A: ^f64, lda: ^blasint, tau: ^f64, work: ^f64, lwork: ^blasint, info: ^Info) ---
	sgehrd_ :: proc(n: ^blasint, ilo: ^blasint, ihi: ^blasint, A: ^f32, lda: ^blasint, tau: ^f32, work: ^f32, lwork: ^blasint, info: ^Info) ---
	zgehrd_ :: proc(n: ^blasint, ilo: ^blasint, ihi: ^blasint, A: ^complex128, lda: ^blasint, tau: ^complex128, work: ^complex128, lwork: ^blasint, info: ^Info) ---

	// Jacobi SVD
	cgejsv_ :: proc(joba: cstring, jobu: cstring, jobv: cstring, jobr: cstring, jobt: cstring, jobp: cstring, m: ^blasint, n: ^blasint, A: ^complex64, lda: ^blasint, SVA: ^f32, U: ^complex64, ldu: ^blasint, V: ^complex64, ldv: ^blasint, cwork: ^complex64, lwork: ^blasint, rwork: ^f32, lrwork: ^blasint, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t) ---
	dgejsv_ :: proc(joba: cstring, jobu: cstring, jobv: cstring, jobr: cstring, jobt: cstring, jobp: cstring, m: ^blasint, n: ^blasint, A: ^f64, lda: ^blasint, SVA: ^f64, U: ^f64, ldu: ^blasint, V: ^f64, ldv: ^blasint, work: ^f64, lwork: ^blasint, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t) ---
	sgejsv_ :: proc(joba: cstring, jobu: cstring, jobv: cstring, jobr: cstring, jobt: cstring, jobp: cstring, m: ^blasint, n: ^blasint, A: ^f32, lda: ^blasint, SVA: ^f32, U: ^f32, ldu: ^blasint, V: ^f32, ldv: ^blasint, work: ^f32, lwork: ^blasint, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t) ---
	zgejsv_ :: proc(joba: cstring, jobu: cstring, jobv: cstring, jobr: cstring, jobt: cstring, jobp: cstring, m: ^blasint, n: ^blasint, A: ^complex128, lda: ^blasint, SVA: ^f64, U: ^complex128, ldu: ^blasint, V: ^complex128, ldv: ^blasint, cwork: ^complex128, lwork: ^blasint, rwork: ^f64, lrwork: ^blasint, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t) ---

	// LQ factorization
	cgelq_ :: proc(m: ^blasint, n: ^blasint, A: ^complex64, lda: ^blasint, T: ^complex64, tsize: ^blasint, work: ^complex64, lwork: ^blasint, info: ^Info) ---
	dgelq_ :: proc(m: ^blasint, n: ^blasint, A: ^f64, lda: ^blasint, T: ^f64, tsize: ^blasint, work: ^f64, lwork: ^blasint, info: ^Info) ---
	sgelq_ :: proc(m: ^blasint, n: ^blasint, A: ^f32, lda: ^blasint, T: ^f32, tsize: ^blasint, work: ^f32, lwork: ^blasint, info: ^Info) ---
	zgelq_ :: proc(m: ^blasint, n: ^blasint, A: ^complex128, lda: ^blasint, T: ^complex128, tsize: ^blasint, work: ^complex128, lwork: ^blasint, info: ^Info) ---
	cgelq2_ :: proc(m: ^blasint, n: ^blasint, A: ^complex64, lda: ^blasint, tau: ^complex64, work: ^complex64, info: ^Info) ---
	dgelq2_ :: proc(m: ^blasint, n: ^blasint, A: ^f64, lda: ^blasint, tau: ^f64, work: ^f64, info: ^Info) ---
	sgelq2_ :: proc(m: ^blasint, n: ^blasint, A: ^f32, lda: ^blasint, tau: ^f32, work: ^f32, info: ^Info) ---
	zgelq2_ :: proc(m: ^blasint, n: ^blasint, A: ^complex128, lda: ^blasint, tau: ^complex128, work: ^complex128, info: ^Info) ---

	cgelqf_ :: proc(m: ^blasint, n: ^blasint, A: ^complex64, lda: ^blasint, tau: ^complex64, work: ^complex64, lwork: ^blasint, info: ^Info) ---
	dgelqf_ :: proc(m: ^blasint, n: ^blasint, A: ^f64, lda: ^blasint, tau: ^f64, work: ^f64, lwork: ^blasint, info: ^Info) ---
	sgelqf_ :: proc(m: ^blasint, n: ^blasint, A: ^f32, lda: ^blasint, tau: ^f32, work: ^f32, lwork: ^blasint, info: ^Info) ---
	zgelqf_ :: proc(m: ^blasint, n: ^blasint, A: ^complex128, lda: ^blasint, tau: ^complex128, work: ^complex128, lwork: ^blasint, info: ^Info) ---

	// Least squares solvers
	cgels_ :: proc(trans: cstring, m: ^blasint, n: ^blasint, nrhs: ^blasint, A: ^complex64, lda: ^blasint, B: ^complex64, ldb: ^blasint, work: ^complex64, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	dgels_ :: proc(trans: cstring, m: ^blasint, n: ^blasint, nrhs: ^blasint, A: ^f64, lda: ^blasint, B: ^f64, ldb: ^blasint, work: ^f64, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	sgels_ :: proc(trans: cstring, m: ^blasint, n: ^blasint, nrhs: ^blasint, A: ^f32, lda: ^blasint, B: ^f32, ldb: ^blasint, work: ^f32, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	zgels_ :: proc(trans: cstring, m: ^blasint, n: ^blasint, nrhs: ^blasint, A: ^complex128, lda: ^blasint, B: ^complex128, ldb: ^blasint, work: ^complex128, lwork: ^blasint, info: ^Info, _: c.size_t) ---

	cgelsd_ :: proc(m: ^blasint, n: ^blasint, nrhs: ^blasint, A: ^complex64, lda: ^blasint, B: ^complex64, ldb: ^blasint, S: ^f32, rcond: ^f32, rank: ^blasint, work: ^complex64, lwork: ^blasint, rwork: ^f32, iwork: ^blasint, info: ^Info) ---
	dgelsd_ :: proc(m: ^blasint, n: ^blasint, nrhs: ^blasint, A: ^f64, lda: ^blasint, B: ^f64, ldb: ^blasint, S: ^f64, rcond: ^f64, rank: ^blasint, work: ^f64, lwork: ^blasint, iwork: ^blasint, info: ^Info) ---
	sgelsd_ :: proc(m: ^blasint, n: ^blasint, nrhs: ^blasint, A: ^f32, lda: ^blasint, B: ^f32, ldb: ^blasint, S: ^f32, rcond: ^f32, rank: ^blasint, work: ^f32, lwork: ^blasint, iwork: ^blasint, info: ^Info) ---
	zgelsd_ :: proc(m: ^blasint, n: ^blasint, nrhs: ^blasint, A: ^complex128, lda: ^blasint, B: ^complex128, ldb: ^blasint, S: ^f64, rcond: ^f64, rank: ^blasint, work: ^complex128, lwork: ^blasint, rwork: ^f64, iwork: ^blasint, info: ^Info) ---

	cgelss_ :: proc(m: ^blasint, n: ^blasint, nrhs: ^blasint, A: ^complex64, lda: ^blasint, B: ^complex64, ldb: ^blasint, S: ^f32, rcond: ^f32, rank: ^blasint, work: ^complex64, lwork: ^blasint, rwork: ^f32, info: ^Info) ---
	dgelss_ :: proc(m: ^blasint, n: ^blasint, nrhs: ^blasint, A: ^f64, lda: ^blasint, B: ^f64, ldb: ^blasint, S: ^f64, rcond: ^f64, rank: ^blasint, work: ^f64, lwork: ^blasint, info: ^Info) ---
	sgelss_ :: proc(m: ^blasint, n: ^blasint, nrhs: ^blasint, A: ^f32, lda: ^blasint, B: ^f32, ldb: ^blasint, S: ^f32, rcond: ^f32, rank: ^blasint, work: ^f32, lwork: ^blasint, info: ^Info) ---
	zgelss_ :: proc(m: ^blasint, n: ^blasint, nrhs: ^blasint, A: ^complex128, lda: ^blasint, B: ^complex128, ldb: ^blasint, S: ^f64, rcond: ^f64, rank: ^blasint, work: ^complex128, lwork: ^blasint, rwork: ^f64, info: ^Info) ---

	cgelsy_ :: proc(m: ^blasint, n: ^blasint, nrhs: ^blasint, A: ^complex64, lda: ^blasint, B: ^complex64, ldb: ^blasint, JPVT: ^blasint, rcond: ^f32, rank: ^blasint, work: ^complex64, lwork: ^blasint, rwork: ^f32, info: ^Info) ---
	dgelsy_ :: proc(m: ^blasint, n: ^blasint, nrhs: ^blasint, A: ^f64, lda: ^blasint, B: ^f64, ldb: ^blasint, JPVT: ^blasint, rcond: ^f64, rank: ^blasint, work: ^f64, lwork: ^blasint, info: ^Info) ---
	sgelsy_ :: proc(m: ^blasint, n: ^blasint, nrhs: ^blasint, A: ^f32, lda: ^blasint, B: ^f32, ldb: ^blasint, JPVT: ^blasint, rcond: ^f32, rank: ^blasint, work: ^f32, lwork: ^blasint, info: ^Info) ---
	zgelsy_ :: proc(m: ^blasint, n: ^blasint, nrhs: ^blasint, A: ^complex128, lda: ^blasint, B: ^complex128, ldb: ^blasint, JPVT: ^blasint, rcond: ^f64, rank: ^blasint, work: ^complex128, lwork: ^blasint, rwork: ^f64, info: ^Info) ---

	// Apply LQ/QR factorization
	cgemlq_ :: proc(side: cstring, trans: cstring, m: ^blasint, n: ^blasint, k: ^blasint, A: ^complex64, lda: ^blasint, T: ^complex64, tsize: ^blasint, C: ^complex64, ldc: ^blasint, work: ^complex64, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	dgemlq_ :: proc(side: cstring, trans: cstring, m: ^blasint, n: ^blasint, k: ^blasint, A: ^f64, lda: ^blasint, T: ^f64, tsize: ^blasint, C: ^f64, ldc: ^blasint, work: ^f64, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	sgemlq_ :: proc(side: cstring, trans: cstring, m: ^blasint, n: ^blasint, k: ^blasint, A: ^f32, lda: ^blasint, T: ^f32, tsize: ^blasint, C: ^f32, ldc: ^blasint, work: ^f32, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	zgemlq_ :: proc(side: cstring, trans: cstring, m: ^blasint, n: ^blasint, k: ^blasint, A: ^complex128, lda: ^blasint, T: ^complex128, tsize: ^blasint, C: ^complex128, ldc: ^blasint, work: ^complex128, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---

	cgemqr_ :: proc(side: cstring, trans: cstring, m: ^blasint, n: ^blasint, k: ^blasint, A: ^complex64, lda: ^blasint, T: ^complex64, tsize: ^blasint, C: ^complex64, ldc: ^blasint, work: ^complex64, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	dgemqr_ :: proc(side: cstring, trans: cstring, m: ^blasint, n: ^blasint, k: ^blasint, A: ^f64, lda: ^blasint, T: ^f64, tsize: ^blasint, C: ^f64, ldc: ^blasint, work: ^f64, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	sgemqr_ :: proc(side: cstring, trans: cstring, m: ^blasint, n: ^blasint, k: ^blasint, A: ^f32, lda: ^blasint, T: ^f32, tsize: ^blasint, C: ^f32, ldc: ^blasint, work: ^f32, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	zgemqr_ :: proc(side: cstring, trans: cstring, m: ^blasint, n: ^blasint, k: ^blasint, A: ^complex128, lda: ^blasint, T: ^complex128, tsize: ^blasint, C: ^complex128, ldc: ^blasint, work: ^complex128, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---

	cgemqrt_ :: proc(side: cstring, trans: cstring, m: ^blasint, n: ^blasint, k: ^blasint, nb: ^blasint, V: ^complex64, ldv: ^blasint, T: ^complex64, ldt: ^blasint, C: ^complex64, ldc: ^blasint, work: ^complex64, info: ^Info, _: c.size_t, _: c.size_t) ---
	dgemqrt_ :: proc(side: cstring, trans: cstring, m: ^blasint, n: ^blasint, k: ^blasint, nb: ^blasint, V: ^f64, ldv: ^blasint, T: ^f64, ldt: ^blasint, C: ^f64, ldc: ^blasint, work: ^f64, info: ^Info, _: c.size_t, _: c.size_t) ---
	sgemqrt_ :: proc(side: cstring, trans: cstring, m: ^blasint, n: ^blasint, k: ^blasint, nb: ^blasint, V: ^f32, ldv: ^blasint, T: ^f32, ldt: ^blasint, C: ^f32, ldc: ^blasint, work: ^f32, info: ^Info, _: c.size_t, _: c.size_t) ---
	zgemqrt_ :: proc(side: cstring, trans: cstring, m: ^blasint, n: ^blasint, k: ^blasint, nb: ^blasint, V: ^complex128, ldv: ^blasint, T: ^complex128, ldt: ^blasint, C: ^complex128, ldc: ^blasint, work: ^complex128, info: ^Info, _: c.size_t, _: c.size_t) ---

	// QL factorization
	cgeql2_ :: proc(m: ^blasint, n: ^blasint, A: ^complex64, lda: ^blasint, tau: ^complex64, work: ^complex64, info: ^Info) ---
	dgeql2_ :: proc(m: ^blasint, n: ^blasint, A: ^f64, lda: ^blasint, tau: ^f64, work: ^f64, info: ^Info) ---
	sgeql2_ :: proc(m: ^blasint, n: ^blasint, A: ^f32, lda: ^blasint, tau: ^f32, work: ^f32, info: ^Info) ---
	zgeql2_ :: proc(m: ^blasint, n: ^blasint, A: ^complex128, lda: ^blasint, tau: ^complex128, work: ^complex128, info: ^Info) ---
	cgeqlf_ :: proc(m: ^blasint, n: ^blasint, A: ^complex64, lda: ^blasint, tau: ^complex64, work: ^complex64, lwork: ^blasint, info: ^Info) ---
	dgeqlf_ :: proc(m: ^blasint, n: ^blasint, A: ^f64, lda: ^blasint, tau: ^f64, work: ^f64, lwork: ^blasint, info: ^Info) ---
	sgeqlf_ :: proc(m: ^blasint, n: ^blasint, A: ^f32, lda: ^blasint, tau: ^f32, work: ^f32, lwork: ^blasint, info: ^Info) ---
	zgeqlf_ :: proc(m: ^blasint, n: ^blasint, A: ^complex128, lda: ^blasint, tau: ^complex128, work: ^complex128, lwork: ^blasint, info: ^Info) ---

	// QR factorization with column pivoting (deprecated)
	@(deprecated = "Use sgeqp3_ instead")
	sgeqpf_ :: proc(m: ^blasint, n: ^blasint, a: ^f32, lda: ^blasint, jpvt: ^blasint, tau: ^f32, work: ^f32, info: ^Info) ---
	@(deprecated = "Use dgeqp3_ instead")
	dgeqpf_ :: proc(m: ^blasint, n: ^blasint, a: ^f64, lda: ^blasint, jpvt: ^blasint, tau: ^f64, work: ^f64, info: ^Info) ---
	@(deprecated = "Use cgeqp3_ instead")
	cgeqpf_ :: proc(m: ^blasint, n: ^blasint, a: ^complex64, lda: ^blasint, jpvt: ^blasint, tau: ^complex64, work: ^complex64, rwork: ^f32, info: ^Info) ---
	@(deprecated = "Use zgeqp3_ instead")
	zgeqpf_ :: proc(m: ^blasint, n: ^blasint, a: ^complex128, lda: ^blasint, jpvt: ^blasint, tau: ^complex128, work: ^complex128, rwork: ^f64, info: ^Info) ---

	// QR factorization with column pivoting (BLAS-3)
	cgeqp3_ :: proc(m: ^blasint, n: ^blasint, A: ^complex64, lda: ^blasint, JPVT: ^blasint, tau: ^complex64, work: ^complex64, lwork: ^blasint, rwork: ^f32, info: ^Info) ---
	dgeqp3_ :: proc(m: ^blasint, n: ^blasint, A: ^f64, lda: ^blasint, JPVT: ^blasint, tau: ^f64, work: ^f64, lwork: ^blasint, info: ^Info) ---
	sgeqp3_ :: proc(m: ^blasint, n: ^blasint, A: ^f32, lda: ^blasint, JPVT: ^blasint, tau: ^f32, work: ^f32, lwork: ^blasint, info: ^Info) ---
	zgeqp3_ :: proc(m: ^blasint, n: ^blasint, A: ^complex128, lda: ^blasint, JPVT: ^blasint, tau: ^complex128, work: ^complex128, lwork: ^blasint, rwork: ^f64, info: ^Info) ---

	// QR factorization
	cgeqr_ :: proc(m: ^blasint, n: ^blasint, A: ^complex64, lda: ^blasint, T: ^complex64, tsize: ^blasint, work: ^complex64, lwork: ^blasint, info: ^Info) ---
	dgeqr_ :: proc(m: ^blasint, n: ^blasint, A: ^f64, lda: ^blasint, T: ^f64, tsize: ^blasint, work: ^f64, lwork: ^blasint, info: ^Info) ---
	sgeqr_ :: proc(m: ^blasint, n: ^blasint, A: ^f32, lda: ^blasint, T: ^f32, tsize: ^blasint, work: ^f32, lwork: ^blasint, info: ^Info) ---
	zgeqr_ :: proc(m: ^blasint, n: ^blasint, A: ^complex128, lda: ^blasint, T: ^complex128, tsize: ^blasint, work: ^complex128, lwork: ^blasint, info: ^Info) ---

	cgeqr2_ :: proc(m: ^blasint, n: ^blasint, A: ^complex64, lda: ^blasint, tau: ^complex64, work: ^complex64, info: ^Info) ---
	dgeqr2_ :: proc(m: ^blasint, n: ^blasint, A: ^f64, lda: ^blasint, tau: ^f64, work: ^f64, info: ^Info) ---
	sgeqr2_ :: proc(m: ^blasint, n: ^blasint, A: ^f32, lda: ^blasint, tau: ^f32, work: ^f32, info: ^Info) ---
	zgeqr2_ :: proc(m: ^blasint, n: ^blasint, A: ^complex128, lda: ^blasint, tau: ^complex128, work: ^complex128, info: ^Info) ---

	cgeqrf_ :: proc(m: ^blasint, n: ^blasint, A: ^complex64, lda: ^blasint, tau: ^complex64, work: ^complex64, lwork: ^blasint, info: ^Info) ---
	dgeqrf_ :: proc(m: ^blasint, n: ^blasint, A: ^f64, lda: ^blasint, tau: ^f64, work: ^f64, lwork: ^blasint, info: ^Info) ---
	sgeqrf_ :: proc(m: ^blasint, n: ^blasint, A: ^f32, lda: ^blasint, tau: ^f32, work: ^f32, lwork: ^blasint, info: ^Info) ---
	zgeqrf_ :: proc(m: ^blasint, n: ^blasint, A: ^complex128, lda: ^blasint, tau: ^complex128, work: ^complex128, lwork: ^blasint, info: ^Info) ---

	cgeqrfp_ :: proc(m: ^blasint, n: ^blasint, A: ^complex64, lda: ^blasint, tau: ^complex64, work: ^complex64, lwork: ^blasint, info: ^Info) ---
	dgeqrfp_ :: proc(m: ^blasint, n: ^blasint, A: ^f64, lda: ^blasint, tau: ^f64, work: ^f64, lwork: ^blasint, info: ^Info) ---
	sgeqrfp_ :: proc(m: ^blasint, n: ^blasint, A: ^f32, lda: ^blasint, tau: ^f32, work: ^f32, lwork: ^blasint, info: ^Info) ---
	zgeqrfp_ :: proc(m: ^blasint, n: ^blasint, A: ^complex128, lda: ^blasint, tau: ^complex128, work: ^complex128, lwork: ^blasint, info: ^Info) ---

	cgeqrt_ :: proc(m: ^blasint, n: ^blasint, nb: ^blasint, A: ^complex64, lda: ^blasint, T: ^complex64, ldt: ^blasint, work: ^complex64, info: ^Info) ---
	dgeqrt_ :: proc(m: ^blasint, n: ^blasint, nb: ^blasint, A: ^f64, lda: ^blasint, T: ^f64, ldt: ^blasint, work: ^f64, info: ^Info) ---
	sgeqrt_ :: proc(m: ^blasint, n: ^blasint, nb: ^blasint, A: ^f32, lda: ^blasint, T: ^f32, ldt: ^blasint, work: ^f32, info: ^Info) ---
	zgeqrt_ :: proc(m: ^blasint, n: ^blasint, nb: ^blasint, A: ^complex128, lda: ^blasint, T: ^complex128, ldt: ^blasint, work: ^complex128, info: ^Info) ---

	cgeqrt2_ :: proc(m: ^blasint, n: ^blasint, A: ^complex64, lda: ^blasint, T: ^complex64, ldt: ^blasint, info: ^Info) ---
	dgeqrt2_ :: proc(m: ^blasint, n: ^blasint, A: ^f64, lda: ^blasint, T: ^f64, ldt: ^blasint, info: ^Info) ---
	sgeqrt2_ :: proc(m: ^blasint, n: ^blasint, A: ^f32, lda: ^blasint, T: ^f32, ldt: ^blasint, info: ^Info) ---
	zgeqrt2_ :: proc(m: ^blasint, n: ^blasint, A: ^complex128, lda: ^blasint, T: ^complex128, ldt: ^blasint, info: ^Info) ---

	cgeqrt3_ :: proc(m: ^blasint, n: ^blasint, A: ^complex64, lda: ^blasint, T: ^complex64, ldt: ^blasint, info: ^Info) ---
	dgeqrt3_ :: proc(m: ^blasint, n: ^blasint, A: ^f64, lda: ^blasint, T: ^f64, ldt: ^blasint, info: ^Info) ---
	sgeqrt3_ :: proc(m: ^blasint, n: ^blasint, A: ^f32, lda: ^blasint, T: ^f32, ldt: ^blasint, info: ^Info) ---
	zgeqrt3_ :: proc(m: ^blasint, n: ^blasint, A: ^complex128, lda: ^blasint, T: ^complex128, ldt: ^blasint, info: ^Info) ---

	// Iterative refinement for general matrices
	cgerfs_ :: proc(trans: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex64, lda: ^blasint, AF: ^complex64, ldaf: ^blasint, ipiv: ^blasint, B: ^complex64, ldb: ^blasint, X: ^complex64, ldx: ^blasint, ferr: ^f32, berr: ^f32, work: ^complex64, rwork: ^f32, info: ^Info, _: c.size_t) ---
	dgerfs_ :: proc(trans: cstring, n: ^blasint, nrhs: ^blasint, A: ^f64, lda: ^blasint, AF: ^f64, ldaf: ^blasint, ipiv: ^blasint, B: ^f64, ldb: ^blasint, X: ^f64, ldx: ^blasint, ferr: ^f64, berr: ^f64, work: ^f64, iwork: ^blasint, info: ^Info, _: c.size_t) ---
	sgerfs_ :: proc(trans: cstring, n: ^blasint, nrhs: ^blasint, A: ^f32, lda: ^blasint, AF: ^f32, ldaf: ^blasint, ipiv: ^blasint, B: ^f32, ldb: ^blasint, X: ^f32, ldx: ^blasint, ferr: ^f32, berr: ^f32, work: ^f32, iwork: ^blasint, info: ^Info, _: c.size_t) ---
	zgerfs_ :: proc(trans: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex128, lda: ^blasint, AF: ^complex128, ldaf: ^blasint, ipiv: ^blasint, B: ^complex128, ldb: ^blasint, X: ^complex128, ldx: ^blasint, ferr: ^f64, berr: ^f64, work: ^complex128, rwork: ^f64, info: ^Info, _: c.size_t) ---

	cgerfsx_ :: proc(trans: cstring, equed: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex64, lda: ^blasint, AF: ^complex64, ldaf: ^blasint, ipiv: ^blasint, R: ^f32, C: ^f32, B: ^complex64, ldb: ^blasint, X: ^complex64, ldx: ^blasint, rcond: ^f32, berr: ^f32, n_err_bnds: ^blasint, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: ^blasint, params: ^f32, work: ^complex64, rwork: ^f32, info: ^Info, _: c.size_t, _: c.size_t) ---
	dgerfsx_ :: proc(trans: cstring, equed: cstring, n: ^blasint, nrhs: ^blasint, A: ^f64, lda: ^blasint, AF: ^f64, ldaf: ^blasint, ipiv: ^blasint, R: ^f64, C: ^f64, B: ^f64, ldb: ^blasint, X: ^f64, ldx: ^blasint, rcond: ^f64, berr: ^f64, n_err_bnds: ^blasint, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: ^blasint, params: ^f64, work: ^f64, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	sgerfsx_ :: proc(trans: cstring, equed: cstring, n: ^blasint, nrhs: ^blasint, A: ^f32, lda: ^blasint, AF: ^f32, ldaf: ^blasint, ipiv: ^blasint, R: ^f32, C: ^f32, B: ^f32, ldb: ^blasint, X: ^f32, ldx: ^blasint, rcond: ^f32, berr: ^f32, n_err_bnds: ^blasint, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: ^blasint, params: ^f32, work: ^f32, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	zgerfsx_ :: proc(trans: cstring, equed: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex128, lda: ^blasint, AF: ^complex128, ldaf: ^blasint, ipiv: ^blasint, R: ^f64, C: ^f64, B: ^complex128, ldb: ^blasint, X: ^complex128, ldx: ^blasint, rcond: ^f64, berr: ^f64, n_err_bnds: ^blasint, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: ^blasint, params: ^f64, work: ^complex128, rwork: ^f64, info: ^Info, _: c.size_t, _: c.size_t) ---

	cgerq2_ :: proc(m: ^blasint, n: ^blasint, A: ^complex64, lda: ^blasint, tau: ^complex64, work: ^complex64, info: ^Info) ---
	dgerq2_ :: proc(m: ^blasint, n: ^blasint, A: ^f64, lda: ^blasint, tau: ^f64, work: ^f64, info: ^Info) ---
	sgerq2_ :: proc(m: ^blasint, n: ^blasint, A: ^f32, lda: ^blasint, tau: ^f32, work: ^f32, info: ^Info) ---
	zgerq2_ :: proc(m: ^blasint, n: ^blasint, A: ^complex128, lda: ^blasint, tau: ^complex128, work: ^complex128, info: ^Info) ---

	cgerqf_ :: proc(m: ^blasint, n: ^blasint, A: ^complex64, lda: ^blasint, tau: ^complex64, work: ^complex64, lwork: ^blasint, info: ^Info) ---
	dgerqf_ :: proc(m: ^blasint, n: ^blasint, A: ^f64, lda: ^blasint, tau: ^f64, work: ^f64, lwork: ^blasint, info: ^Info) ---
	sgerqf_ :: proc(m: ^blasint, n: ^blasint, A: ^f32, lda: ^blasint, tau: ^f32, work: ^f32, lwork: ^blasint, info: ^Info) ---
	zgerqf_ :: proc(m: ^blasint, n: ^blasint, A: ^complex128, lda: ^blasint, tau: ^complex128, work: ^complex128, lwork: ^blasint, info: ^Info) ---

	// Singular Value Decomposition - Divide and conquer
	cgesdd_ :: proc(jobz: cstring, m: ^blasint, n: ^blasint, A: ^complex64, lda: ^blasint, S: ^f32, U: ^complex64, ldu: ^blasint, VT: ^complex64, ldvt: ^blasint, work: ^complex64, lwork: ^blasint, rwork: ^f32, iwork: ^blasint, info: ^Info, _: c.size_t) ---
	dgesdd_ :: proc(jobz: cstring, m: ^blasint, n: ^blasint, A: ^f64, lda: ^blasint, S: ^f64, U: ^f64, ldu: ^blasint, VT: ^f64, ldvt: ^blasint, work: ^f64, lwork: ^blasint, iwork: ^blasint, info: ^Info, _: c.size_t) ---
	sgesdd_ :: proc(jobz: cstring, m: ^blasint, n: ^blasint, A: ^f32, lda: ^blasint, S: ^f32, U: ^f32, ldu: ^blasint, VT: ^f32, ldvt: ^blasint, work: ^f32, lwork: ^blasint, iwork: ^blasint, info: ^Info, _: c.size_t) ---
	zgesdd_ :: proc(jobz: cstring, m: ^blasint, n: ^blasint, A: ^complex128, lda: ^blasint, S: ^f64, U: ^complex128, ldu: ^blasint, VT: ^complex128, ldvt: ^blasint, work: ^complex128, lwork: ^blasint, rwork: ^f64, iwork: ^blasint, info: ^Info, _: c.size_t) ---

	cgedmd_ :: proc(jobs: cstring, jobz: cstring, jobr: cstring, jobf: cstring, whtsvd: ^blasint, m: ^blasint, n: ^blasint, x: ^complex64, ldx: ^blasint, y: ^complex64, ldy: ^blasint, nrnk: ^blasint, tol: ^f32, k: ^blasint, eigs: ^complex64, z: ^complex64, ldz: ^blasint, res: ^f32, b: ^complex64, ldb: ^blasint, w: ^complex64, ldw: ^blasint, s: ^complex64, lds: ^blasint, zwork: ^complex64, lzwork: ^blasint, work: ^f32, lwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t) ---
	dgedmd_ :: proc(jobs: cstring, jobz: cstring, jobr: cstring, jobf: cstring, whtsvd: ^blasint, m: ^blasint, n: ^blasint, x: ^f64, ldx: ^blasint, y: ^f64, ldy: ^blasint, nrnk: ^blasint, tol: ^f64, k: ^blasint, reig: ^f64, imeig: ^f64, z: ^f64, ldz: ^blasint, res: ^f64, b: ^f64, ldb: ^blasint, w: ^f64, ldw: ^blasint, s: ^f64, lds: ^blasint, work: ^f64, lwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t) ---
	sgedmd_ :: proc(jobs: cstring, jobz: cstring, jobr: cstring, jobf: cstring, whtsvd: ^blasint, m: ^blasint, n: ^blasint, x: ^f32, ldx: ^blasint, y: ^f32, ldy: ^blasint, nrnk: ^blasint, tol: ^f32, k: ^blasint, reig: ^f32, imeig: ^f32, z: ^f32, ldz: ^blasint, res: ^f32, b: ^f32, ldb: ^blasint, w: ^f32, ldw: ^blasint, s: ^f32, lds: ^blasint, work: ^f32, lwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t) ---
	zgedmd_ :: proc(jobs: cstring, jobz: cstring, jobr: cstring, jobf: cstring, whtsvd: ^blasint, m: ^blasint, n: ^blasint, x: ^complex128, ldx: ^blasint, y: ^complex128, ldy: ^blasint, nrnk: ^blasint, tol: ^f64, k: ^blasint, eigs: ^complex128, z: ^complex128, ldz: ^blasint, res: ^f64, b: ^complex128, ldb: ^blasint, w: ^complex128, ldw: ^blasint, s: ^complex128, lds: ^blasint, zwork: ^complex128, lzwork: ^blasint, rwork: ^f64, lrwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t) ---

	cgedmdq_ :: proc(jobs: cstring, jobz: cstring, jobr: cstring, jobq: cstring, jobt: cstring, jobf: cstring, whtsvd: ^blasint, m: ^blasint, n: ^blasint, f: ^complex64, ldf: ^blasint, x: ^complex64, ldx: ^blasint, y: ^complex64, ldy: ^blasint, nrnk: ^blasint, tol: ^f32, k: ^blasint, eigs: ^complex64, z: ^complex64, ldz: ^blasint, res: ^f32, b: ^complex64, ldb: ^blasint, v: ^complex64, ldv: ^blasint, s: ^complex64, lds: ^blasint, zwork: ^complex64, lzwork: ^blasint, work: ^f32, lwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t) ---
	dgedmdq_ :: proc(jobs: cstring, jobz: cstring, jobr: cstring, jobq: cstring, jobt: cstring, jobf: cstring, whtsvd: ^blasint, m: ^blasint, n: ^blasint, f: ^f64, ldf: ^blasint, x: ^f64, ldx: ^blasint, y: ^f64, ldy: ^blasint, nrnk: ^blasint, tol: ^f64, k: ^blasint, reig: ^f64, imeig: ^f64, z: ^f64, ldz: ^blasint, res: ^f64, b: ^f64, ldb: ^blasint, v: ^f64, ldv: ^blasint, s: ^f64, lds: ^blasint, work: ^f64, lwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t) ---
	sgedmdq_ :: proc(jobs: cstring, jobz: cstring, jobr: cstring, jobq: cstring, jobt: cstring, jobf: cstring, whtsvd: ^blasint, m: ^blasint, n: ^blasint, f: ^f32, ldf: ^blasint, x: ^f32, ldx: ^blasint, y: ^f32, ldy: ^blasint, nrnk: ^blasint, tol: ^f32, k: ^blasint, reig: ^f32, imeig: ^f32, z: ^f32, ldz: ^blasint, res: ^f32, b: ^f32, ldb: ^blasint, v: ^f32, ldv: ^blasint, s: ^f32, lds: ^blasint, work: ^f32, lwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t) ---
	zgedmdq_ :: proc(jobs: cstring, jobz: cstring, jobr: cstring, jobq: cstring, jobt: cstring, jobf: cstring, whtsvd: ^blasint, m: ^blasint, n: ^blasint, f: ^complex128, ldf: ^blasint, x: ^complex128, ldx: ^blasint, y: ^complex128, ldy: ^blasint, nrnk: ^blasint, tol: ^f64, k: ^blasint, eigs: ^complex128, z: ^complex128, ldz: ^blasint, res: ^f64, b: ^complex128, ldb: ^blasint, v: ^complex128, ldv: ^blasint, s: ^complex128, lds: ^blasint, zwork: ^complex128, lzwork: ^blasint, work: ^f64, lwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t) ---

	// cgesv_ :: proc(n: ^blasint, nrhs: ^blasint, A: ^complex64, lda: ^blasint, ipiv: ^blasint, B: ^complex64, ldb: ^blasint, info: ^Info) -> i32 ---
	// dgesv_ :: proc(n: ^blasint, nrhs: ^blasint, A: ^f64, lda: ^blasint, ipiv: ^blasint, B: ^f64, ldb: ^blasint, info: ^Info) -> i32 ---
	// sgesv_ :: proc(n: ^blasint, nrhs: ^blasint, A: ^f32, lda: ^blasint, ipiv: ^blasint, B: ^f32, ldb: ^blasint, info: ^Info) -> i32 ---
	// zgesv_ :: proc(n: ^blasint, nrhs: ^blasint, A: ^complex128, lda: ^blasint, ipiv: ^blasint, B: ^complex128, ldb: ^blasint, info: ^Info) -> i32 ---

	dsgesv_ :: proc(n: ^blasint, nrhs: ^blasint, A: ^f64, lda: ^blasint, ipiv: ^blasint, B: ^f64, ldb: ^blasint, X: ^f64, ldx: ^blasint, work: ^f64, swork: ^f32, iter: ^blasint, info: ^Info) ---
	zcgesv_ :: proc(n: ^blasint, nrhs: ^blasint, A: ^complex128, lda: ^blasint, ipiv: ^blasint, B: ^complex128, ldb: ^blasint, X: ^complex128, ldx: ^blasint, work: ^complex128, swork: ^complex64, rwork: ^f64, iter: ^blasint, info: ^Info) ---

	cgesvd_ :: proc(jobu: cstring, jobvt: cstring, m: ^blasint, n: ^blasint, A: ^complex64, lda: ^blasint, S: ^f32, U: ^complex64, ldu: ^blasint, VT: ^complex64, ldvt: ^blasint, work: ^complex64, lwork: ^blasint, rwork: ^f32, info: ^Info, _: c.size_t, _: c.size_t) ---
	dgesvd_ :: proc(jobu: cstring, jobvt: cstring, m: ^blasint, n: ^blasint, A: ^f64, lda: ^blasint, S: ^f64, U: ^f64, ldu: ^blasint, VT: ^f64, ldvt: ^blasint, work: ^f64, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	sgesvd_ :: proc(jobu: cstring, jobvt: cstring, m: ^blasint, n: ^blasint, A: ^f32, lda: ^blasint, S: ^f32, U: ^f32, ldu: ^blasint, VT: ^f32, ldvt: ^blasint, work: ^f32, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	zgesvd_ :: proc(jobu: cstring, jobvt: cstring, m: ^blasint, n: ^blasint, A: ^complex128, lda: ^blasint, S: ^f64, U: ^complex128, ldu: ^blasint, VT: ^complex128, ldvt: ^blasint, work: ^complex128, lwork: ^blasint, rwork: ^f64, info: ^Info, _: c.size_t, _: c.size_t) ---

	cgesvdq_ :: proc(joba: cstring, jobp: cstring, jobr: cstring, jobu: cstring, jobv: cstring, m: ^blasint, n: ^blasint, A: ^complex64, lda: ^blasint, S: ^f32, U: ^complex64, ldu: ^blasint, V: ^complex64, ldv: ^blasint, numrank: ^blasint, iwork: ^blasint, liwork: ^blasint, cwork: ^complex64, lcwork: ^blasint, rwork: ^f32, lrwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t) ---
	dgesvdq_ :: proc(joba: cstring, jobp: cstring, jobr: cstring, jobu: cstring, jobv: cstring, m: ^blasint, n: ^blasint, A: ^f64, lda: ^blasint, S: ^f64, U: ^f64, ldu: ^blasint, V: ^f64, ldv: ^blasint, numrank: ^blasint, iwork: ^blasint, liwork: ^blasint, work: ^f64, lwork: ^blasint, rwork: ^f64, lrwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t) ---
	sgesvdq_ :: proc(joba: cstring, jobp: cstring, jobr: cstring, jobu: cstring, jobv: cstring, m: ^blasint, n: ^blasint, A: ^f32, lda: ^blasint, S: ^f32, U: ^f32, ldu: ^blasint, V: ^f32, ldv: ^blasint, numrank: ^blasint, iwork: ^blasint, liwork: ^blasint, work: ^f32, lwork: ^blasint, rwork: ^f32, lrwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t) ---
	zgesvdq_ :: proc(joba: cstring, jobp: cstring, jobr: cstring, jobu: cstring, jobv: cstring, m: ^blasint, n: ^blasint, A: ^complex128, lda: ^blasint, S: ^f64, U: ^complex128, ldu: ^blasint, V: ^complex128, ldv: ^blasint, numrank: ^blasint, iwork: ^blasint, liwork: ^blasint, cwork: ^complex128, lcwork: ^blasint, rwork: ^f64, lrwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t) ---

	cgesvdx_ :: proc(jobu: cstring, jobvt: cstring, range: cstring, m: ^blasint, n: ^blasint, A: ^complex64, lda: ^blasint, vl: ^f32, vu: ^f32, il: ^blasint, iu: ^blasint, ns: ^blasint, S: ^f32, U: ^complex64, ldu: ^blasint, VT: ^complex64, ldvt: ^blasint, work: ^complex64, lwork: ^blasint, rwork: ^f32, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	dgesvdx_ :: proc(jobu: cstring, jobvt: cstring, range: cstring, m: ^blasint, n: ^blasint, A: ^f64, lda: ^blasint, vl: ^f64, vu: ^f64, il: ^blasint, iu: ^blasint, ns: ^blasint, S: ^f64, U: ^f64, ldu: ^blasint, VT: ^f64, ldvt: ^blasint, work: ^f64, lwork: ^blasint, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	sgesvdx_ :: proc(jobu: cstring, jobvt: cstring, range: cstring, m: ^blasint, n: ^blasint, A: ^f32, lda: ^blasint, vl: ^f32, vu: ^f32, il: ^blasint, iu: ^blasint, ns: ^blasint, S: ^f32, U: ^f32, ldu: ^blasint, VT: ^f32, ldvt: ^blasint, work: ^f32, lwork: ^blasint, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	zgesvdx_ :: proc(jobu: cstring, jobvt: cstring, range: cstring, m: ^blasint, n: ^blasint, A: ^complex128, lda: ^blasint, vl: ^f64, vu: ^f64, il: ^blasint, iu: ^blasint, ns: ^blasint, S: ^f64, U: ^complex128, ldu: ^blasint, VT: ^complex128, ldvt: ^blasint, work: ^complex128, lwork: ^blasint, rwork: ^f64, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---

	cgesvj_ :: proc(joba: cstring, jobu: cstring, jobv: cstring, m: ^blasint, n: ^blasint, A: ^complex64, lda: ^blasint, SVA: ^f32, mv: ^blasint, V: ^complex64, ldv: ^blasint, cwork: ^complex64, lwork: ^blasint, rwork: ^f32, lrwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	dgesvj_ :: proc(joba: cstring, jobu: cstring, jobv: cstring, m: ^blasint, n: ^blasint, A: ^f64, lda: ^blasint, SVA: ^f64, mv: ^blasint, V: ^f64, ldv: ^blasint, work: ^f64, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	sgesvj_ :: proc(joba: cstring, jobu: cstring, jobv: cstring, m: ^blasint, n: ^blasint, A: ^f32, lda: ^blasint, SVA: ^f32, mv: ^blasint, V: ^f32, ldv: ^blasint, work: ^f32, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	zgesvj_ :: proc(joba: cstring, jobu: cstring, jobv: cstring, m: ^blasint, n: ^blasint, A: ^complex128, lda: ^blasint, SVA: ^f64, mv: ^blasint, V: ^complex128, ldv: ^blasint, cwork: ^complex128, lwork: ^blasint, rwork: ^f64, lrwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---

	cgesvx_ :: proc(fact: cstring, trans: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex64, lda: ^blasint, AF: ^complex64, ldaf: ^blasint, ipiv: ^blasint, equed: ^byte, R: ^f32, C: ^f32, B: ^complex64, ldb: ^blasint, X: ^complex64, ldx: ^blasint, rcond: ^f32, ferr: ^f32, berr: ^f32, work: ^complex64, rwork: ^f32, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	dgesvx_ :: proc(fact: cstring, trans: cstring, n: ^blasint, nrhs: ^blasint, A: ^f64, lda: ^blasint, AF: ^f64, ldaf: ^blasint, ipiv: ^blasint, equed: ^byte, R: ^f64, C: ^f64, B: ^f64, ldb: ^blasint, X: ^f64, ldx: ^blasint, rcond: ^f64, ferr: ^f64, berr: ^f64, work: ^f64, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	sgesvx_ :: proc(fact: cstring, trans: cstring, n: ^blasint, nrhs: ^blasint, A: ^f32, lda: ^blasint, AF: ^f32, ldaf: ^blasint, ipiv: ^blasint, equed: ^byte, R: ^f32, C: ^f32, B: ^f32, ldb: ^blasint, X: ^f32, ldx: ^blasint, rcond: ^f32, ferr: ^f32, berr: ^f32, work: ^f32, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	zgesvx_ :: proc(fact: cstring, trans: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex128, lda: ^blasint, AF: ^complex128, ldaf: ^blasint, ipiv: ^blasint, equed: ^byte, R: ^f64, C: ^f64, B: ^complex128, ldb: ^blasint, X: ^complex128, ldx: ^blasint, rcond: ^f64, ferr: ^f64, berr: ^f64, work: ^complex128, rwork: ^f64, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---

	cgesvxx_ :: proc(fact: cstring, trans: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex64, lda: ^blasint, AF: ^complex64, ldaf: ^blasint, ipiv: ^blasint, equed: ^byte, R: ^f32, C: ^f32, B: ^complex64, ldb: ^blasint, X: ^complex64, ldx: ^blasint, rcond: ^f32, rpvgrw: ^f32, berr: ^f32, n_err_bnds: ^blasint, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: ^blasint, params: ^f32, work: ^complex64, rwork: ^f32, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	dgesvxx_ :: proc(fact: cstring, trans: cstring, n: ^blasint, nrhs: ^blasint, A: ^f64, lda: ^blasint, AF: ^f64, ldaf: ^blasint, ipiv: ^blasint, equed: ^byte, R: ^f64, C: ^f64, B: ^f64, ldb: ^blasint, X: ^f64, ldx: ^blasint, rcond: ^f64, rpvgrw: ^f64, berr: ^f64, n_err_bnds: ^blasint, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: ^blasint, params: ^f64, work: ^f64, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	sgesvxx_ :: proc(fact: cstring, trans: cstring, n: ^blasint, nrhs: ^blasint, A: ^f32, lda: ^blasint, AF: ^f32, ldaf: ^blasint, ipiv: ^blasint, equed: ^byte, R: ^f32, C: ^f32, B: ^f32, ldb: ^blasint, X: ^f32, ldx: ^blasint, rcond: ^f32, rpvgrw: ^f32, berr: ^f32, n_err_bnds: ^blasint, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: ^blasint, params: ^f32, work: ^f32, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	zgesvxx_ :: proc(fact: cstring, trans: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex128, lda: ^blasint, AF: ^complex128, ldaf: ^blasint, ipiv: ^blasint, equed: ^byte, R: ^f64, C: ^f64, B: ^complex128, ldb: ^blasint, X: ^complex128, ldx: ^blasint, rcond: ^f64, rpvgrw: ^f64, berr: ^f64, n_err_bnds: ^blasint, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: ^blasint, params: ^f64, work: ^complex128, rwork: ^f64, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---

	// cgetf2_ :: proc(m: ^blasint, n: ^blasint, A: ^complex64, lda: ^blasint, ipiv: ^blasint, info: ^Info) -> i32 ---
	// dgetf2_ :: proc(m: ^blasint, n: ^blasint, A: ^f64, lda: ^blasint, ipiv: ^blasint, info: ^Info) -> i32 ---
	// sgetf2_ :: proc(m: ^blasint, n: ^blasint, A: ^f32, lda: ^blasint, ipiv: ^blasint, info: ^Info) -> i32 ---
	// zgetf2_ :: proc(m: ^blasint, n: ^blasint, A: ^complex128, lda: ^blasint, ipiv: ^blasint, info: ^Info) -> i32 ---
	// cgetrf_ :: proc(m: ^blasint, n: ^blasint, A: ^complex64, lda: ^blasint, ipiv: ^blasint, info: ^Info) -> i32 ---
	// dgetrf_ :: proc(m: ^blasint, n: ^blasint, A: ^f64, lda: ^blasint, ipiv: ^blasint, info: ^Info) -> i32 ---
	// sgetrf_ :: proc(m: ^blasint, n: ^blasint, A: ^f32, lda: ^blasint, ipiv: ^blasint, info: ^Info) -> i32 ---
	// zgetrf_ :: proc(m: ^blasint, n: ^blasint, A: ^complex128, lda: ^blasint, ipiv: ^blasint, info: ^Info) -> i32 ---

	cgetrf2_ :: proc(m: ^blasint, n: ^blasint, A: ^complex64, lda: ^blasint, ipiv: ^blasint, info: ^Info) ---
	dgetrf2_ :: proc(m: ^blasint, n: ^blasint, A: ^f64, lda: ^blasint, ipiv: ^blasint, info: ^Info) ---
	sgetrf2_ :: proc(m: ^blasint, n: ^blasint, A: ^f32, lda: ^blasint, ipiv: ^blasint, info: ^Info) ---
	zgetrf2_ :: proc(m: ^blasint, n: ^blasint, A: ^complex128, lda: ^blasint, ipiv: ^blasint, info: ^Info) ---

	cgetri_ :: proc(n: ^blasint, A: ^complex64, lda: ^blasint, ipiv: ^blasint, work: ^complex64, lwork: ^blasint, info: ^Info) ---
	dgetri_ :: proc(n: ^blasint, A: ^f64, lda: ^blasint, ipiv: ^blasint, work: ^f64, lwork: ^blasint, info: ^Info) ---
	sgetri_ :: proc(n: ^blasint, A: ^f32, lda: ^blasint, ipiv: ^blasint, work: ^f32, lwork: ^blasint, info: ^Info) ---
	zgetri_ :: proc(n: ^blasint, A: ^complex128, lda: ^blasint, ipiv: ^blasint, work: ^complex128, lwork: ^blasint, info: ^Info) ---

	// cgetrs_ :: proc(trans: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex64, lda: ^blasint, ipiv: ^blasint, B: ^complex64, ldb: ^blasint, info: ^Info, _: c.size_t) -> i32 ---
	// dgetrs_ :: proc(trans: cstring, n: ^blasint, nrhs: ^blasint, A: ^f64, lda: ^blasint, ipiv: ^blasint, B: ^f64, ldb: ^blasint, info: ^Info, _: c.size_t) -> i32 ---
	// sgetrs_ :: proc(trans: cstring, n: ^blasint, nrhs: ^blasint, A: ^f32, lda: ^blasint, ipiv: ^blasint, B: ^f32, ldb: ^blasint, info: ^Info, _: c.size_t) -> i32 ---
	// zgetrs_ :: proc(trans: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex128, lda: ^blasint, ipiv: ^blasint, B: ^complex128, ldb: ^blasint, info: ^Info, _: c.size_t) -> i32 ---

	cgetsls_ :: proc(trans: cstring, m: ^blasint, n: ^blasint, nrhs: ^blasint, A: ^complex64, lda: ^blasint, B: ^complex64, ldb: ^blasint, work: ^complex64, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	dgetsls_ :: proc(trans: cstring, m: ^blasint, n: ^blasint, nrhs: ^blasint, A: ^f64, lda: ^blasint, B: ^f64, ldb: ^blasint, work: ^f64, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	sgetsls_ :: proc(trans: cstring, m: ^blasint, n: ^blasint, nrhs: ^blasint, A: ^f32, lda: ^blasint, B: ^f32, ldb: ^blasint, work: ^f32, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	zgetsls_ :: proc(trans: cstring, m: ^blasint, n: ^blasint, nrhs: ^blasint, A: ^complex128, lda: ^blasint, B: ^complex128, ldb: ^blasint, work: ^complex128, lwork: ^blasint, info: ^Info, _: c.size_t) ---

	cgetsqrhrt_ :: proc(m: ^blasint, n: ^blasint, mb1: ^blasint, nb1: ^blasint, nb2: ^blasint, A: ^complex64, lda: ^blasint, T: ^complex64, ldt: ^blasint, work: ^complex64, lwork: ^blasint, info: ^Info) ---
	dgetsqrhrt_ :: proc(m: ^blasint, n: ^blasint, mb1: ^blasint, nb1: ^blasint, nb2: ^blasint, A: ^f64, lda: ^blasint, T: ^f64, ldt: ^blasint, work: ^f64, lwork: ^blasint, info: ^Info) ---
	sgetsqrhrt_ :: proc(m: ^blasint, n: ^blasint, mb1: ^blasint, nb1: ^blasint, nb2: ^blasint, A: ^f32, lda: ^blasint, T: ^f32, ldt: ^blasint, work: ^f32, lwork: ^blasint, info: ^Info) ---
	zgetsqrhrt_ :: proc(m: ^blasint, n: ^blasint, mb1: ^blasint, nb1: ^blasint, nb2: ^blasint, A: ^complex128, lda: ^blasint, T: ^complex128, ldt: ^blasint, work: ^complex128, lwork: ^blasint, info: ^Info) ---

	cggbak_ :: proc(job: cstring, side: cstring, n: ^blasint, ilo: ^blasint, ihi: ^blasint, lscale: ^f32, rscale: ^f32, m: ^blasint, V: ^complex64, ldv: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	dggbak_ :: proc(job: cstring, side: cstring, n: ^blasint, ilo: ^blasint, ihi: ^blasint, lscale: ^f64, rscale: ^f64, m: ^blasint, V: ^f64, ldv: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	sggbak_ :: proc(job: cstring, side: cstring, n: ^blasint, ilo: ^blasint, ihi: ^blasint, lscale: ^f32, rscale: ^f32, m: ^blasint, V: ^f32, ldv: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	zggbak_ :: proc(job: cstring, side: cstring, n: ^blasint, ilo: ^blasint, ihi: ^blasint, lscale: ^f64, rscale: ^f64, m: ^blasint, V: ^complex128, ldv: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---

	cggbal_ :: proc(job: cstring, n: ^blasint, A: ^complex64, lda: ^blasint, B: ^complex64, ldb: ^blasint, ilo: ^blasint, ihi: ^blasint, lscale: ^f32, rscale: ^f32, work: ^f32, info: ^Info, _: c.size_t) ---
	dggbal_ :: proc(job: cstring, n: ^blasint, A: ^f64, lda: ^blasint, B: ^f64, ldb: ^blasint, ilo: ^blasint, ihi: ^blasint, lscale: ^f64, rscale: ^f64, work: ^f64, info: ^Info, _: c.size_t) ---
	sggbal_ :: proc(job: cstring, n: ^blasint, A: ^f32, lda: ^blasint, B: ^f32, ldb: ^blasint, ilo: ^blasint, ihi: ^blasint, lscale: ^f32, rscale: ^f32, work: ^f32, info: ^Info, _: c.size_t) ---
	zggbal_ :: proc(job: cstring, n: ^blasint, A: ^complex128, lda: ^blasint, B: ^complex128, ldb: ^blasint, ilo: ^blasint, ihi: ^blasint, lscale: ^f64, rscale: ^f64, work: ^f64, info: ^Info, _: c.size_t) ---

	cgges_ :: proc(jobvsl: cstring, jobvsr: cstring, sort: cstring, selctg: LAPACK_C_SELECT2, n: ^blasint, A: ^complex64, lda: ^blasint, B: ^complex64, ldb: ^blasint, sdim: ^blasint, alpha: ^complex64, beta: ^complex64, VSL: ^complex64, ldvsl: ^blasint, VSR: ^complex64, ldvsr: ^blasint, work: ^complex64, lwork: ^blasint, rwork: ^f32, BWORK: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	dgges_ :: proc(jobvsl: cstring, jobvsr: cstring, sort: cstring, selctg: LAPACK_D_SELECT3, n: ^blasint, A: ^f64, lda: ^blasint, B: ^f64, ldb: ^blasint, sdim: ^blasint, alphar: ^f64, alphai: ^f64, beta: ^f64, VSL: ^f64, ldvsl: ^blasint, VSR: ^f64, ldvsr: ^blasint, work: ^f64, lwork: ^blasint, BWORK: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	sgges_ :: proc(jobvsl: cstring, jobvsr: cstring, sort: cstring, selctg: LAPACK_S_SELECT3, n: ^blasint, A: ^f32, lda: ^blasint, B: ^f32, ldb: ^blasint, sdim: ^blasint, alphar: ^f32, alphai: ^f32, beta: ^f32, VSL: ^f32, ldvsl: ^blasint, VSR: ^f32, ldvsr: ^blasint, work: ^f32, lwork: ^blasint, BWORK: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	zgges_ :: proc(jobvsl: cstring, jobvsr: cstring, sort: cstring, selctg: LAPACK_Z_SELECT2, n: ^blasint, A: ^complex128, lda: ^blasint, B: ^complex128, ldb: ^blasint, sdim: ^blasint, alpha: ^complex128, beta: ^complex128, VSL: ^complex128, ldvsl: ^blasint, VSR: ^complex128, ldvsr: ^blasint, work: ^complex128, lwork: ^blasint, rwork: ^f64, BWORK: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---

	cgges3_ :: proc(jobvsl: cstring, jobvsr: cstring, sort: cstring, selctg: LAPACK_C_SELECT2, n: ^blasint, A: ^complex64, lda: ^blasint, B: ^complex64, ldb: ^blasint, sdim: ^blasint, alpha: ^complex64, beta: ^complex64, VSL: ^complex64, ldvsl: ^blasint, VSR: ^complex64, ldvsr: ^blasint, work: ^complex64, lwork: ^blasint, rwork: ^f32, BWORK: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	dgges3_ :: proc(jobvsl: cstring, jobvsr: cstring, sort: cstring, selctg: LAPACK_D_SELECT3, n: ^blasint, A: ^f64, lda: ^blasint, B: ^f64, ldb: ^blasint, sdim: ^blasint, alphar: ^f64, alphai: ^f64, beta: ^f64, VSL: ^f64, ldvsl: ^blasint, VSR: ^f64, ldvsr: ^blasint, work: ^f64, lwork: ^blasint, BWORK: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	sgges3_ :: proc(jobvsl: cstring, jobvsr: cstring, sort: cstring, selctg: LAPACK_S_SELECT3, n: ^blasint, A: ^f32, lda: ^blasint, B: ^f32, ldb: ^blasint, sdim: ^blasint, alphar: ^f32, alphai: ^f32, beta: ^f32, VSL: ^f32, ldvsl: ^blasint, VSR: ^f32, ldvsr: ^blasint, work: ^f32, lwork: ^blasint, BWORK: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	zgges3_ :: proc(jobvsl: cstring, jobvsr: cstring, sort: cstring, selctg: LAPACK_Z_SELECT2, n: ^blasint, A: ^complex128, lda: ^blasint, B: ^complex128, ldb: ^blasint, sdim: ^blasint, alpha: ^complex128, beta: ^complex128, VSL: ^complex128, ldvsl: ^blasint, VSR: ^complex128, ldvsr: ^blasint, work: ^complex128, lwork: ^blasint, rwork: ^f64, BWORK: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---

	cggesx_ :: proc(jobvsl: cstring, jobvsr: cstring, sort: cstring, selctg: LAPACK_C_SELECT2, sense: cstring, n: ^blasint, A: ^complex64, lda: ^blasint, B: ^complex64, ldb: ^blasint, sdim: ^blasint, alpha: ^complex64, beta: ^complex64, VSL: ^complex64, ldvsl: ^blasint, VSR: ^complex64, ldvsr: ^blasint, rconde: ^f32, rcondv: ^f32, work: ^complex64, lwork: ^blasint, rwork: ^f32, iwork: ^blasint, liwork: ^blasint, BWORK: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t) ---
	dggesx_ :: proc(jobvsl: cstring, jobvsr: cstring, sort: cstring, selctg: LAPACK_D_SELECT3, sense: cstring, n: ^blasint, A: ^f64, lda: ^blasint, B: ^f64, ldb: ^blasint, sdim: ^blasint, alphar: ^f64, alphai: ^f64, beta: ^f64, VSL: ^f64, ldvsl: ^blasint, VSR: ^f64, ldvsr: ^blasint, rconde: ^f64, rcondv: ^f64, work: ^f64, lwork: ^blasint, iwork: ^blasint, liwork: ^blasint, BWORK: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t) ---
	sggesx_ :: proc(jobvsl: cstring, jobvsr: cstring, sort: cstring, selctg: LAPACK_S_SELECT3, sense: cstring, n: ^blasint, A: ^f32, lda: ^blasint, B: ^f32, ldb: ^blasint, sdim: ^blasint, alphar: ^f32, alphai: ^f32, beta: ^f32, VSL: ^f32, ldvsl: ^blasint, VSR: ^f32, ldvsr: ^blasint, rconde: ^f32, rcondv: ^f32, work: ^f32, lwork: ^blasint, iwork: ^blasint, liwork: ^blasint, BWORK: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t) ---
	zggesx_ :: proc(jobvsl: cstring, jobvsr: cstring, sort: cstring, selctg: LAPACK_Z_SELECT2, sense: cstring, n: ^blasint, A: ^complex128, lda: ^blasint, B: ^complex128, ldb: ^blasint, sdim: ^blasint, alpha: ^complex128, beta: ^complex128, VSL: ^complex128, ldvsl: ^blasint, VSR: ^complex128, ldvsr: ^blasint, rconde: ^f64, rcondv: ^f64, work: ^complex128, lwork: ^blasint, rwork: ^f64, iwork: ^blasint, liwork: ^blasint, BWORK: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t) ---

	cggev_ :: proc(jobvl: cstring, jobvr: cstring, n: ^blasint, A: ^complex64, lda: ^blasint, B: ^complex64, ldb: ^blasint, alpha: ^complex64, beta: ^complex64, VL: ^complex64, ldvl: ^blasint, VR: ^complex64, ldvr: ^blasint, work: ^complex64, lwork: ^blasint, rwork: ^f32, info: ^Info, _: c.size_t, _: c.size_t) ---
	dggev_ :: proc(jobvl: cstring, jobvr: cstring, n: ^blasint, A: ^f64, lda: ^blasint, B: ^f64, ldb: ^blasint, alphar: ^f64, alphai: ^f64, beta: ^f64, VL: ^f64, ldvl: ^blasint, VR: ^f64, ldvr: ^blasint, work: ^f64, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	sggev_ :: proc(jobvl: cstring, jobvr: cstring, n: ^blasint, A: ^f32, lda: ^blasint, B: ^f32, ldb: ^blasint, alphar: ^f32, alphai: ^f32, beta: ^f32, VL: ^f32, ldvl: ^blasint, VR: ^f32, ldvr: ^blasint, work: ^f32, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	zggev_ :: proc(jobvl: cstring, jobvr: cstring, n: ^blasint, A: ^complex128, lda: ^blasint, B: ^complex128, ldb: ^blasint, alpha: ^complex128, beta: ^complex128, VL: ^complex128, ldvl: ^blasint, VR: ^complex128, ldvr: ^blasint, work: ^complex128, lwork: ^blasint, rwork: ^f64, info: ^Info, _: c.size_t, _: c.size_t) ---

	cggev3_ :: proc(jobvl: cstring, jobvr: cstring, n: ^blasint, A: ^complex64, lda: ^blasint, B: ^complex64, ldb: ^blasint, alpha: ^complex64, beta: ^complex64, VL: ^complex64, ldvl: ^blasint, VR: ^complex64, ldvr: ^blasint, work: ^complex64, lwork: ^blasint, rwork: ^f32, info: ^Info, _: c.size_t, _: c.size_t) ---
	dggev3_ :: proc(jobvl: cstring, jobvr: cstring, n: ^blasint, A: ^f64, lda: ^blasint, B: ^f64, ldb: ^blasint, alphar: ^f64, alphai: ^f64, beta: ^f64, VL: ^f64, ldvl: ^blasint, VR: ^f64, ldvr: ^blasint, work: ^f64, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	sggev3_ :: proc(jobvl: cstring, jobvr: cstring, n: ^blasint, A: ^f32, lda: ^blasint, B: ^f32, ldb: ^blasint, alphar: ^f32, alphai: ^f32, beta: ^f32, VL: ^f32, ldvl: ^blasint, VR: ^f32, ldvr: ^blasint, work: ^f32, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	zggev3_ :: proc(jobvl: cstring, jobvr: cstring, n: ^blasint, A: ^complex128, lda: ^blasint, B: ^complex128, ldb: ^blasint, alpha: ^complex128, beta: ^complex128, VL: ^complex128, ldvl: ^blasint, VR: ^complex128, ldvr: ^blasint, work: ^complex128, lwork: ^blasint, rwork: ^f64, info: ^Info, _: c.size_t, _: c.size_t) ---

	cggevx_ :: proc(balanc: cstring, jobvl: cstring, jobvr: cstring, sense: cstring, n: ^blasint, A: ^complex64, lda: ^blasint, B: ^complex64, ldb: ^blasint, alpha: ^complex64, beta: ^complex64, VL: ^complex64, ldvl: ^blasint, VR: ^complex64, ldvr: ^blasint, ilo: ^blasint, ihi: ^blasint, lscale: ^f32, rscale: ^f32, abnrm: ^f32, bbnrm: ^f32, rconde: ^f32, rcondv: ^f32, work: ^complex64, lwork: ^blasint, rwork: ^f32, iwork: ^blasint, BWORK: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t) ---
	dggevx_ :: proc(balanc: cstring, jobvl: cstring, jobvr: cstring, sense: cstring, n: ^blasint, A: ^f64, lda: ^blasint, B: ^f64, ldb: ^blasint, alphar: ^f64, alphai: ^f64, beta: ^f64, VL: ^f64, ldvl: ^blasint, VR: ^f64, ldvr: ^blasint, ilo: ^blasint, ihi: ^blasint, lscale: ^f64, rscale: ^f64, abnrm: ^f64, bbnrm: ^f64, rconde: ^f64, rcondv: ^f64, work: ^f64, lwork: ^blasint, iwork: ^blasint, BWORK: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t) ---
	sggevx_ :: proc(balanc: cstring, jobvl: cstring, jobvr: cstring, sense: cstring, n: ^blasint, A: ^f32, lda: ^blasint, B: ^f32, ldb: ^blasint, alphar: ^f32, alphai: ^f32, beta: ^f32, VL: ^f32, ldvl: ^blasint, VR: ^f32, ldvr: ^blasint, ilo: ^blasint, ihi: ^blasint, lscale: ^f32, rscale: ^f32, abnrm: ^f32, bbnrm: ^f32, rconde: ^f32, rcondv: ^f32, work: ^f32, lwork: ^blasint, iwork: ^blasint, BWORK: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t) ---
	zggevx_ :: proc(balanc: cstring, jobvl: cstring, jobvr: cstring, sense: cstring, n: ^blasint, A: ^complex128, lda: ^blasint, B: ^complex128, ldb: ^blasint, alpha: ^complex128, beta: ^complex128, VL: ^complex128, ldvl: ^blasint, VR: ^complex128, ldvr: ^blasint, ilo: ^blasint, ihi: ^blasint, lscale: ^f64, rscale: ^f64, abnrm: ^f64, bbnrm: ^f64, rconde: ^f64, rcondv: ^f64, work: ^complex128, lwork: ^blasint, rwork: ^f64, iwork: ^blasint, BWORK: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t) ---

	cggglm_ :: proc(n: ^blasint, m: ^blasint, p: ^blasint, A: ^complex64, lda: ^blasint, B: ^complex64, ldb: ^blasint, D: ^complex64, X: ^complex64, Y: ^complex64, work: ^complex64, lwork: ^blasint, info: ^Info) ---
	dggglm_ :: proc(n: ^blasint, m: ^blasint, p: ^blasint, A: ^f64, lda: ^blasint, B: ^f64, ldb: ^blasint, D: ^f64, X: ^f64, Y: ^f64, work: ^f64, lwork: ^blasint, info: ^Info) ---
	sggglm_ :: proc(n: ^blasint, m: ^blasint, p: ^blasint, A: ^f32, lda: ^blasint, B: ^f32, ldb: ^blasint, D: ^f32, X: ^f32, Y: ^f32, work: ^f32, lwork: ^blasint, info: ^Info) ---
	zggglm_ :: proc(n: ^blasint, m: ^blasint, p: ^blasint, A: ^complex128, lda: ^blasint, B: ^complex128, ldb: ^blasint, D: ^complex128, X: ^complex128, Y: ^complex128, work: ^complex128, lwork: ^blasint, info: ^Info) ---

	cgghd3_ :: proc(compq: cstring, compz: cstring, n: ^blasint, ilo: ^blasint, ihi: ^blasint, A: ^complex64, lda: ^blasint, B: ^complex64, ldb: ^blasint, Q: ^complex64, ldq: ^blasint, Z: ^complex64, ldz: ^blasint, work: ^complex64, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	dgghd3_ :: proc(compq: cstring, compz: cstring, n: ^blasint, ilo: ^blasint, ihi: ^blasint, A: ^f64, lda: ^blasint, B: ^f64, ldb: ^blasint, Q: ^f64, ldq: ^blasint, Z: ^f64, ldz: ^blasint, work: ^f64, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	sgghd3_ :: proc(compq: cstring, compz: cstring, n: ^blasint, ilo: ^blasint, ihi: ^blasint, A: ^f32, lda: ^blasint, B: ^f32, ldb: ^blasint, Q: ^f32, ldq: ^blasint, Z: ^f32, ldz: ^blasint, work: ^f32, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	zgghd3_ :: proc(compq: cstring, compz: cstring, n: ^blasint, ilo: ^blasint, ihi: ^blasint, A: ^complex128, lda: ^blasint, B: ^complex128, ldb: ^blasint, Q: ^complex128, ldq: ^blasint, Z: ^complex128, ldz: ^blasint, work: ^complex128, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---

	cgghrd_ :: proc(compq: cstring, compz: cstring, n: ^blasint, ilo: ^blasint, ihi: ^blasint, A: ^complex64, lda: ^blasint, B: ^complex64, ldb: ^blasint, Q: ^complex64, ldq: ^blasint, Z: ^complex64, ldz: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	dgghrd_ :: proc(compq: cstring, compz: cstring, n: ^blasint, ilo: ^blasint, ihi: ^blasint, A: ^f64, lda: ^blasint, B: ^f64, ldb: ^blasint, Q: ^f64, ldq: ^blasint, Z: ^f64, ldz: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	sgghrd_ :: proc(compq: cstring, compz: cstring, n: ^blasint, ilo: ^blasint, ihi: ^blasint, A: ^f32, lda: ^blasint, B: ^f32, ldb: ^blasint, Q: ^f32, ldq: ^blasint, Z: ^f32, ldz: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	zgghrd_ :: proc(compq: cstring, compz: cstring, n: ^blasint, ilo: ^blasint, ihi: ^blasint, A: ^complex128, lda: ^blasint, B: ^complex128, ldb: ^blasint, Q: ^complex128, ldq: ^blasint, Z: ^complex128, ldz: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---

	cgglse_ :: proc(m: ^blasint, n: ^blasint, p: ^blasint, A: ^complex64, lda: ^blasint, B: ^complex64, ldb: ^blasint, C: ^complex64, D: ^complex64, X: ^complex64, work: ^complex64, lwork: ^blasint, info: ^Info) ---
	dgglse_ :: proc(m: ^blasint, n: ^blasint, p: ^blasint, A: ^f64, lda: ^blasint, B: ^f64, ldb: ^blasint, C: ^f64, D: ^f64, X: ^f64, work: ^f64, lwork: ^blasint, info: ^Info) ---
	sgglse_ :: proc(m: ^blasint, n: ^blasint, p: ^blasint, A: ^f32, lda: ^blasint, B: ^f32, ldb: ^blasint, C: ^f32, D: ^f32, X: ^f32, work: ^f32, lwork: ^blasint, info: ^Info) ---
	zgglse_ :: proc(m: ^blasint, n: ^blasint, p: ^blasint, A: ^complex128, lda: ^blasint, B: ^complex128, ldb: ^blasint, C: ^complex128, D: ^complex128, X: ^complex128, work: ^complex128, lwork: ^blasint, info: ^Info) ---

	cggqrf_ :: proc(n: ^blasint, m: ^blasint, p: ^blasint, A: ^complex64, lda: ^blasint, taua: ^complex64, B: ^complex64, ldb: ^blasint, taub: ^complex64, work: ^complex64, lwork: ^blasint, info: ^Info) ---
	dggqrf_ :: proc(n: ^blasint, m: ^blasint, p: ^blasint, A: ^f64, lda: ^blasint, taua: ^f64, B: ^f64, ldb: ^blasint, taub: ^f64, work: ^f64, lwork: ^blasint, info: ^Info) ---
	sggqrf_ :: proc(n: ^blasint, m: ^blasint, p: ^blasint, A: ^f32, lda: ^blasint, taua: ^f32, B: ^f32, ldb: ^blasint, taub: ^f32, work: ^f32, lwork: ^blasint, info: ^Info) ---
	zggqrf_ :: proc(n: ^blasint, m: ^blasint, p: ^blasint, A: ^complex128, lda: ^blasint, taua: ^complex128, B: ^complex128, ldb: ^blasint, taub: ^complex128, work: ^complex128, lwork: ^blasint, info: ^Info) ---

	cggrqf_ :: proc(m: ^blasint, p: ^blasint, n: ^blasint, A: ^complex64, lda: ^blasint, taua: ^complex64, B: ^complex64, ldb: ^blasint, taub: ^complex64, work: ^complex64, lwork: ^blasint, info: ^Info) ---
	dggrqf_ :: proc(m: ^blasint, p: ^blasint, n: ^blasint, A: ^f64, lda: ^blasint, taua: ^f64, B: ^f64, ldb: ^blasint, taub: ^f64, work: ^f64, lwork: ^blasint, info: ^Info) ---
	sggrqf_ :: proc(m: ^blasint, p: ^blasint, n: ^blasint, A: ^f32, lda: ^blasint, taua: ^f32, B: ^f32, ldb: ^blasint, taub: ^f32, work: ^f32, lwork: ^blasint, info: ^Info) ---
	zggrqf_ :: proc(m: ^blasint, p: ^blasint, n: ^blasint, A: ^complex128, lda: ^blasint, taua: ^complex128, B: ^complex128, ldb: ^blasint, taub: ^complex128, work: ^complex128, lwork: ^blasint, info: ^Info) ---

	cggsvd_ :: proc(jobu: cstring, jobv: cstring, jobq: cstring, m: ^blasint, n: ^blasint, p: ^blasint, k: ^blasint, l: ^blasint, a: ^complex64, lda: ^blasint, b: ^complex64, ldb: ^blasint, alpha: ^f32, beta: ^f32, u: ^complex64, ldu: ^blasint, v: ^complex64, ldv: ^blasint, q: ^complex64, ldq: ^blasint, work: ^complex64, rwork: ^f32, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	sggsvd_ :: proc(jobu: cstring, jobv: cstring, jobq: cstring, m: ^blasint, n: ^blasint, p: ^blasint, k: ^blasint, l: ^blasint, a: ^f32, lda: ^blasint, b: ^f32, ldb: ^blasint, alpha: ^f32, beta: ^f32, u: ^f32, ldu: ^blasint, v: ^f32, ldv: ^blasint, q: ^f32, ldq: ^blasint, work: ^f32, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	dggsvd_ :: proc(jobu: cstring, jobv: cstring, jobq: cstring, m: ^blasint, n: ^blasint, p: ^blasint, k: ^blasint, l: ^blasint, a: ^f64, lda: ^blasint, b: ^f64, ldb: ^blasint, alpha: ^f64, beta: ^f64, u: ^f64, ldu: ^blasint, v: ^f64, ldv: ^blasint, q: ^f64, ldq: ^blasint, work: ^f64, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	zggsvd_ :: proc(jobu: cstring, jobv: cstring, jobq: cstring, m: ^blasint, n: ^blasint, p: ^blasint, k: ^blasint, l: ^blasint, a: ^complex128, lda: ^blasint, b: ^complex128, ldb: ^blasint, alpha: ^f64, beta: ^f64, u: ^complex128, ldu: ^blasint, v: ^complex128, ldv: ^blasint, q: ^complex128, ldq: ^blasint, work: ^complex128, rwork: ^f64, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---

	cggsvd3_ :: proc(jobu: cstring, jobv: cstring, jobq: cstring, m: ^blasint, n: ^blasint, p: ^blasint, k: ^blasint, l: ^blasint, A: ^complex64, lda: ^blasint, B: ^complex64, ldb: ^blasint, alpha: ^f32, beta: ^f32, U: ^complex64, ldu: ^blasint, V: ^complex64, ldv: ^blasint, Q: ^complex64, ldq: ^blasint, work: ^complex64, lwork: ^blasint, rwork: ^f32, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	dggsvd3_ :: proc(jobu: cstring, jobv: cstring, jobq: cstring, m: ^blasint, n: ^blasint, p: ^blasint, k: ^blasint, l: ^blasint, A: ^f64, lda: ^blasint, B: ^f64, ldb: ^blasint, alpha: ^f64, beta: ^f64, U: ^f64, ldu: ^blasint, V: ^f64, ldv: ^blasint, Q: ^f64, ldq: ^blasint, work: ^f64, lwork: ^blasint, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	sggsvd3_ :: proc(jobu: cstring, jobv: cstring, jobq: cstring, m: ^blasint, n: ^blasint, p: ^blasint, k: ^blasint, l: ^blasint, A: ^f32, lda: ^blasint, B: ^f32, ldb: ^blasint, alpha: ^f32, beta: ^f32, U: ^f32, ldu: ^blasint, V: ^f32, ldv: ^blasint, Q: ^f32, ldq: ^blasint, work: ^f32, lwork: ^blasint, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	zggsvd3_ :: proc(jobu: cstring, jobv: cstring, jobq: cstring, m: ^blasint, n: ^blasint, p: ^blasint, k: ^blasint, l: ^blasint, A: ^complex128, lda: ^blasint, B: ^complex128, ldb: ^blasint, alpha: ^f64, beta: ^f64, U: ^complex128, ldu: ^blasint, V: ^complex128, ldv: ^blasint, Q: ^complex128, ldq: ^blasint, work: ^complex128, lwork: ^blasint, rwork: ^f64, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---

	sggsvp_ :: proc(jobu: cstring, jobv: cstring, jobq: cstring, m: ^blasint, p: ^blasint, n: ^blasint, a: ^f32, lda: ^blasint, b: ^f32, ldb: ^blasint, tola: ^f32, tolb: ^f32, k: ^blasint, l: ^blasint, u: ^f32, ldu: ^blasint, v: ^f32, ldv: ^blasint, q: ^f32, ldq: ^blasint, iwork: ^blasint, tau: ^f32, work: ^f32, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	dggsvp_ :: proc(jobu: cstring, jobv: cstring, jobq: cstring, m: ^blasint, p: ^blasint, n: ^blasint, a: ^f64, lda: ^blasint, b: ^f64, ldb: ^blasint, tola: ^f64, tolb: ^f64, k: ^blasint, l: ^blasint, u: ^f64, ldu: ^blasint, v: ^f64, ldv: ^blasint, q: ^f64, ldq: ^blasint, iwork: ^blasint, tau: ^f64, work: ^f64, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	cggsvp_ :: proc(jobu: cstring, jobv: cstring, jobq: cstring, m: ^blasint, p: ^blasint, n: ^blasint, a: ^complex64, lda: ^blasint, b: ^complex64, ldb: ^blasint, tola: ^f32, tolb: ^f32, k: ^blasint, l: ^blasint, u: ^complex64, ldu: ^blasint, v: ^complex64, ldv: ^blasint, q: ^complex64, ldq: ^blasint, iwork: ^blasint, rwork: ^f32, tau: ^complex64, work: ^complex64, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	zggsvp_ :: proc(jobu: cstring, jobv: cstring, jobq: cstring, m: ^blasint, p: ^blasint, n: ^blasint, a: ^complex128, lda: ^blasint, b: ^complex128, ldb: ^blasint, tola: ^f64, tolb: ^f64, k: ^blasint, l: ^blasint, u: ^complex128, ldu: ^blasint, v: ^complex128, ldv: ^blasint, q: ^complex128, ldq: ^blasint, iwork: ^blasint, rwork: ^f64, tau: ^complex128, work: ^complex128, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---

	cggsvp3_ :: proc(jobu: cstring, jobv: cstring, jobq: cstring, m: ^blasint, p: ^blasint, n: ^blasint, A: ^complex64, lda: ^blasint, B: ^complex64, ldb: ^blasint, tola: ^f32, tolb: ^f32, k: ^blasint, l: ^blasint, U: ^complex64, ldu: ^blasint, V: ^complex64, ldv: ^blasint, Q: ^complex64, ldq: ^blasint, iwork: ^blasint, rwork: ^f32, tau: ^complex64, work: ^complex64, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	dggsvp3_ :: proc(jobu: cstring, jobv: cstring, jobq: cstring, m: ^blasint, p: ^blasint, n: ^blasint, A: ^f64, lda: ^blasint, B: ^f64, ldb: ^blasint, tola: ^f64, tolb: ^f64, k: ^blasint, l: ^blasint, U: ^f64, ldu: ^blasint, V: ^f64, ldv: ^blasint, Q: ^f64, ldq: ^blasint, iwork: ^blasint, tau: ^f64, work: ^f64, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	sggsvp3_ :: proc(jobu: cstring, jobv: cstring, jobq: cstring, m: ^blasint, p: ^blasint, n: ^blasint, A: ^f32, lda: ^blasint, B: ^f32, ldb: ^blasint, tola: ^f32, tolb: ^f32, k: ^blasint, l: ^blasint, U: ^f32, ldu: ^blasint, V: ^f32, ldv: ^blasint, Q: ^f32, ldq: ^blasint, iwork: ^blasint, tau: ^f32, work: ^f32, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	zggsvp3_ :: proc(jobu: cstring, jobv: cstring, jobq: cstring, m: ^blasint, p: ^blasint, n: ^blasint, A: ^complex128, lda: ^blasint, B: ^complex128, ldb: ^blasint, tola: ^f64, tolb: ^f64, k: ^blasint, l: ^blasint, U: ^complex128, ldu: ^blasint, V: ^complex128, ldv: ^blasint, Q: ^complex128, ldq: ^blasint, iwork: ^blasint, rwork: ^f64, tau: ^complex128, work: ^complex128, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---

	cgtcon_ :: proc(norm: cstring, n: ^blasint, DL: ^complex64, D: ^complex64, DU: ^complex64, DU2: ^complex64, ipiv: ^blasint, anorm: ^f32, rcond: ^f32, work: ^complex64, info: ^Info, _: c.size_t) ---
	dgtcon_ :: proc(norm: cstring, n: ^blasint, DL: ^f64, D: ^f64, DU: ^f64, DU2: ^f64, ipiv: ^blasint, anorm: ^f64, rcond: ^f64, work: ^f64, iwork: ^blasint, info: ^Info, _: c.size_t) ---
	sgtcon_ :: proc(norm: cstring, n: ^blasint, DL: ^f32, D: ^f32, DU: ^f32, DU2: ^f32, ipiv: ^blasint, anorm: ^f32, rcond: ^f32, work: ^f32, iwork: ^blasint, info: ^Info, _: c.size_t) ---
	zgtcon_ :: proc(norm: cstring, n: ^blasint, DL: ^complex128, D: ^complex128, DU: ^complex128, DU2: ^complex128, ipiv: ^blasint, anorm: ^f64, rcond: ^f64, work: ^complex128, info: ^Info, _: c.size_t) ---

	cgtrfs_ :: proc(trans: cstring, n: ^blasint, nrhs: ^blasint, DL: ^complex64, D: ^complex64, DU: ^complex64, DLF: ^complex64, DF: ^complex64, DUF: ^complex64, DU2: ^complex64, ipiv: ^blasint, B: ^complex64, ldb: ^blasint, X: ^complex64, ldx: ^blasint, ferr: ^f32, berr: ^f32, work: ^complex64, rwork: ^f32, info: ^Info, _: c.size_t) ---
	dgtrfs_ :: proc(trans: cstring, n: ^blasint, nrhs: ^blasint, DL: ^f64, D: ^f64, DU: ^f64, DLF: ^f64, DF: ^f64, DUF: ^f64, DU2: ^f64, ipiv: ^blasint, B: ^f64, ldb: ^blasint, X: ^f64, ldx: ^blasint, ferr: ^f64, berr: ^f64, work: ^f64, iwork: ^blasint, info: ^Info, _: c.size_t) ---
	sgtrfs_ :: proc(trans: cstring, n: ^blasint, nrhs: ^blasint, DL: ^f32, D: ^f32, DU: ^f32, DLF: ^f32, DF: ^f32, DUF: ^f32, DU2: ^f32, ipiv: ^blasint, B: ^f32, ldb: ^blasint, X: ^f32, ldx: ^blasint, ferr: ^f32, berr: ^f32, work: ^f32, iwork: ^blasint, info: ^Info, _: c.size_t) ---
	zgtrfs_ :: proc(trans: cstring, n: ^blasint, nrhs: ^blasint, DL: ^complex128, D: ^complex128, DU: ^complex128, DLF: ^complex128, DF: ^complex128, DUF: ^complex128, DU2: ^complex128, ipiv: ^blasint, B: ^complex128, ldb: ^blasint, X: ^complex128, ldx: ^blasint, ferr: ^f64, berr: ^f64, work: ^complex128, rwork: ^f64, info: ^Info, _: c.size_t) ---

	cgtsv_ :: proc(n: ^blasint, nrhs: ^blasint, DL: ^complex64, D: ^complex64, DU: ^complex64, B: ^complex64, ldb: ^blasint, info: ^Info) ---
	dgtsv_ :: proc(n: ^blasint, nrhs: ^blasint, DL: ^f64, D: ^f64, DU: ^f64, B: ^f64, ldb: ^blasint, info: ^Info) ---
	sgtsv_ :: proc(n: ^blasint, nrhs: ^blasint, DL: ^f32, D: ^f32, DU: ^f32, B: ^f32, ldb: ^blasint, info: ^Info) ---
	zgtsv_ :: proc(n: ^blasint, nrhs: ^blasint, DL: ^complex128, D: ^complex128, DU: ^complex128, B: ^complex128, ldb: ^blasint, info: ^Info) ---

	cgtsvx_ :: proc(fact: cstring, trans: cstring, n: ^blasint, nrhs: ^blasint, DL: ^complex64, D: ^complex64, DU: ^complex64, DLF: ^complex64, DF: ^complex64, DUF: ^complex64, DU2: ^complex64, ipiv: ^blasint, B: ^complex64, ldb: ^blasint, X: ^complex64, ldx: ^blasint, rcond: ^f32, ferr: ^f32, berr: ^f32, work: ^complex64, rwork: ^f32, info: ^Info, _: c.size_t, _: c.size_t) ---
	dgtsvx_ :: proc(fact: cstring, trans: cstring, n: ^blasint, nrhs: ^blasint, DL: ^f64, D: ^f64, DU: ^f64, DLF: ^f64, DF: ^f64, DUF: ^f64, DU2: ^f64, ipiv: ^blasint, B: ^f64, ldb: ^blasint, X: ^f64, ldx: ^blasint, rcond: ^f64, ferr: ^f64, berr: ^f64, work: ^f64, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	sgtsvx_ :: proc(fact: cstring, trans: cstring, n: ^blasint, nrhs: ^blasint, DL: ^f32, D: ^f32, DU: ^f32, DLF: ^f32, DF: ^f32, DUF: ^f32, DU2: ^f32, ipiv: ^blasint, B: ^f32, ldb: ^blasint, X: ^f32, ldx: ^blasint, rcond: ^f32, ferr: ^f32, berr: ^f32, work: ^f32, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	zgtsvx_ :: proc(fact: cstring, trans: cstring, n: ^blasint, nrhs: ^blasint, DL: ^complex128, D: ^complex128, DU: ^complex128, DLF: ^complex128, DF: ^complex128, DUF: ^complex128, DU2: ^complex128, ipiv: ^blasint, B: ^complex128, ldb: ^blasint, X: ^complex128, ldx: ^blasint, rcond: ^f64, ferr: ^f64, berr: ^f64, work: ^complex128, rwork: ^f64, info: ^Info, _: c.size_t, _: c.size_t) ---

	cgttrf_ :: proc(n: ^blasint, DL: ^complex64, D: ^complex64, DU: ^complex64, DU2: ^complex64, ipiv: ^blasint, info: ^Info) ---
	dgttrf_ :: proc(n: ^blasint, DL: ^f64, D: ^f64, DU: ^f64, DU2: ^f64, ipiv: ^blasint, info: ^Info) ---
	sgttrf_ :: proc(n: ^blasint, DL: ^f32, D: ^f32, DU: ^f32, DU2: ^f32, ipiv: ^blasint, info: ^Info) ---
	zgttrf_ :: proc(n: ^blasint, DL: ^complex128, D: ^complex128, DU: ^complex128, DU2: ^complex128, ipiv: ^blasint, info: ^Info) ---

	cgttrs_ :: proc(trans: cstring, n: ^blasint, nrhs: ^blasint, DL: ^complex64, D: ^complex64, DU: ^complex64, DU2: ^complex64, ipiv: ^blasint, B: ^complex64, ldb: ^blasint, info: ^Info, _: c.size_t) ---
	dgttrs_ :: proc(trans: cstring, n: ^blasint, nrhs: ^blasint, DL: ^f64, D: ^f64, DU: ^f64, DU2: ^f64, ipiv: ^blasint, B: ^f64, ldb: ^blasint, info: ^Info, _: c.size_t) ---
	sgttrs_ :: proc(trans: cstring, n: ^blasint, nrhs: ^blasint, DL: ^f32, D: ^f32, DU: ^f32, DU2: ^f32, ipiv: ^blasint, B: ^f32, ldb: ^blasint, info: ^Info, _: c.size_t) ---
	zgttrs_ :: proc(trans: cstring, n: ^blasint, nrhs: ^blasint, DL: ^complex128, D: ^complex128, DU: ^complex128, DU2: ^complex128, ipiv: ^blasint, B: ^complex128, ldb: ^blasint, info: ^Info, _: c.size_t) ---

	chbev_ :: proc(jobz: cstring, uplo: cstring, n: ^blasint, kd: ^blasint, AB: ^complex64, ldab: ^blasint, W: ^f32, Z: ^complex64, ldz: ^blasint, work: ^complex64, rwork: ^f32, info: ^Info, _: c.size_t, _: c.size_t) ---
	zhbev_ :: proc(jobz: cstring, uplo: cstring, n: ^blasint, kd: ^blasint, AB: ^complex128, ldab: ^blasint, W: ^f64, Z: ^complex128, ldz: ^blasint, work: ^complex128, rwork: ^f64, info: ^Info, _: c.size_t, _: c.size_t) ---
	chbev_2stage_ :: proc(jobz: cstring, uplo: cstring, n: ^blasint, kd: ^blasint, AB: ^complex64, ldab: ^blasint, W: ^f32, Z: ^complex64, ldz: ^blasint, work: ^complex64, lwork: ^blasint, rwork: ^f32, info: ^Info, _: c.size_t, _: c.size_t) ---
	zhbev_2stage_ :: proc(jobz: cstring, uplo: cstring, n: ^blasint, kd: ^blasint, AB: ^complex128, ldab: ^blasint, W: ^f64, Z: ^complex128, ldz: ^blasint, work: ^complex128, lwork: ^blasint, rwork: ^f64, info: ^Info, _: c.size_t, _: c.size_t) ---

	chbevd_ :: proc(jobz: cstring, uplo: cstring, n: ^blasint, kd: ^blasint, AB: ^complex64, ldab: ^blasint, W: ^f32, Z: ^complex64, ldz: ^blasint, work: ^complex64, lwork: ^blasint, rwork: ^f32, lrwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	zhbevd_ :: proc(jobz: cstring, uplo: cstring, n: ^blasint, kd: ^blasint, AB: ^complex128, ldab: ^blasint, W: ^f64, Z: ^complex128, ldz: ^blasint, work: ^complex128, lwork: ^blasint, rwork: ^f64, lrwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	chbevd_2stage_ :: proc(jobz: cstring, uplo: cstring, n: ^blasint, kd: ^blasint, AB: ^complex64, ldab: ^blasint, W: ^f32, Z: ^complex64, ldz: ^blasint, work: ^complex64, lwork: ^blasint, rwork: ^f32, lrwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	zhbevd_2stage_ :: proc(jobz: cstring, uplo: cstring, n: ^blasint, kd: ^blasint, AB: ^complex128, ldab: ^blasint, W: ^f64, Z: ^complex128, ldz: ^blasint, work: ^complex128, lwork: ^blasint, rwork: ^f64, lrwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---

	chbevx_ :: proc(jobz: cstring, range: cstring, uplo: cstring, n: ^blasint, kd: ^blasint, AB: ^complex64, ldab: ^blasint, Q: ^complex64, ldq: ^blasint, vl: ^f32, vu: ^f32, il: ^blasint, iu: ^blasint, abstol: ^f32, m: ^blasint, W: ^f32, Z: ^complex64, ldz: ^blasint, work: ^complex64, rwork: ^f32, iwork: ^blasint, IFAIL: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	zhbevx_ :: proc(jobz: cstring, range: cstring, uplo: cstring, n: ^blasint, kd: ^blasint, AB: ^complex128, ldab: ^blasint, Q: ^complex128, ldq: ^blasint, vl: ^f64, vu: ^f64, il: ^blasint, iu: ^blasint, abstol: ^f64, m: ^blasint, W: ^f64, Z: ^complex128, ldz: ^blasint, work: ^complex128, rwork: ^f64, iwork: ^blasint, IFAIL: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	chbevx_2stage_ :: proc(jobz: cstring, range: cstring, uplo: cstring, n: ^blasint, kd: ^blasint, AB: ^complex64, ldab: ^blasint, Q: ^complex64, ldq: ^blasint, vl: ^f32, vu: ^f32, il: ^blasint, iu: ^blasint, abstol: ^f32, m: ^blasint, W: ^f32, Z: ^complex64, ldz: ^blasint, work: ^complex64, lwork: ^blasint, rwork: ^f32, iwork: ^blasint, IFAIL: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	zhbevx_2stage_ :: proc(jobz: cstring, range: cstring, uplo: cstring, n: ^blasint, kd: ^blasint, AB: ^complex128, ldab: ^blasint, Q: ^complex128, ldq: ^blasint, vl: ^f64, vu: ^f64, il: ^blasint, iu: ^blasint, abstol: ^f64, m: ^blasint, W: ^f64, Z: ^complex128, ldz: ^blasint, work: ^complex128, lwork: ^blasint, rwork: ^f64, iwork: ^blasint, IFAIL: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---

	chbgst_ :: proc(vect: cstring, uplo: cstring, n: ^blasint, ka: ^blasint, kb: ^blasint, AB: ^complex64, ldab: ^blasint, BB: ^complex64, ldbb: ^blasint, X: ^complex64, ldx: ^blasint, work: ^complex64, rwork: ^f32, info: ^Info, _: c.size_t, _: c.size_t) ---
	zhbgst_ :: proc(vect: cstring, uplo: cstring, n: ^blasint, ka: ^blasint, kb: ^blasint, AB: ^complex128, ldab: ^blasint, BB: ^complex128, ldbb: ^blasint, X: ^complex128, ldx: ^blasint, work: ^complex128, rwork: ^f64, info: ^Info, _: c.size_t, _: c.size_t) ---
	chbgv_ :: proc(jobz: cstring, uplo: cstring, n: ^blasint, ka: ^blasint, kb: ^blasint, AB: ^complex64, ldab: ^blasint, BB: ^complex64, ldbb: ^blasint, W: ^f32, Z: ^complex64, ldz: ^blasint, work: ^complex64, rwork: ^f32, info: ^Info, _: c.size_t, _: c.size_t) ---
	zhbgv_ :: proc(jobz: cstring, uplo: cstring, n: ^blasint, ka: ^blasint, kb: ^blasint, AB: ^complex128, ldab: ^blasint, BB: ^complex128, ldbb: ^blasint, W: ^f64, Z: ^complex128, ldz: ^blasint, work: ^complex128, rwork: ^f64, info: ^Info, _: c.size_t, _: c.size_t) ---

	chbgvd_ :: proc(jobz: cstring, uplo: cstring, n: ^blasint, ka: ^blasint, kb: ^blasint, AB: ^complex64, ldab: ^blasint, BB: ^complex64, ldbb: ^blasint, W: ^f32, Z: ^complex64, ldz: ^blasint, work: ^complex64, lwork: ^blasint, rwork: ^f32, lrwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	zhbgvd_ :: proc(jobz: cstring, uplo: cstring, n: ^blasint, ka: ^blasint, kb: ^blasint, AB: ^complex128, ldab: ^blasint, BB: ^complex128, ldbb: ^blasint, W: ^f64, Z: ^complex128, ldz: ^blasint, work: ^complex128, lwork: ^blasint, rwork: ^f64, lrwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	chbgvx_ :: proc(jobz: cstring, range: cstring, uplo: cstring, n: ^blasint, ka: ^blasint, kb: ^blasint, AB: ^complex64, ldab: ^blasint, BB: ^complex64, ldbb: ^blasint, Q: ^complex64, ldq: ^blasint, vl: ^f32, vu: ^f32, il: ^blasint, iu: ^blasint, abstol: ^f32, m: ^blasint, W: ^f32, Z: ^complex64, ldz: ^blasint, work: ^complex64, rwork: ^f32, iwork: ^blasint, IFAIL: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	zhbgvx_ :: proc(jobz: cstring, range: cstring, uplo: cstring, n: ^blasint, ka: ^blasint, kb: ^blasint, AB: ^complex128, ldab: ^blasint, BB: ^complex128, ldbb: ^blasint, Q: ^complex128, ldq: ^blasint, vl: ^f64, vu: ^f64, il: ^blasint, iu: ^blasint, abstol: ^f64, m: ^blasint, W: ^f64, Z: ^complex128, ldz: ^blasint, work: ^complex128, rwork: ^f64, iwork: ^blasint, IFAIL: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---

	chbtrd_ :: proc(vect: cstring, uplo: cstring, n: ^blasint, kd: ^blasint, AB: ^complex64, ldab: ^blasint, D: ^f32, E: ^f32, Q: ^complex64, ldq: ^blasint, work: ^complex64, info: ^Info, _: c.size_t, _: c.size_t) ---
	zhbtrd_ :: proc(vect: cstring, uplo: cstring, n: ^blasint, kd: ^blasint, AB: ^complex128, ldab: ^blasint, D: ^f64, E: ^f64, Q: ^complex128, ldq: ^blasint, work: ^complex128, info: ^Info, _: c.size_t, _: c.size_t) ---
	checon_ :: proc(uplo: cstring, n: ^blasint, A: ^complex64, lda: ^blasint, ipiv: ^blasint, anorm: ^f32, rcond: ^f32, work: ^complex64, info: ^Info, _: c.size_t) ---
	zhecon_ :: proc(uplo: cstring, n: ^blasint, A: ^complex128, lda: ^blasint, ipiv: ^blasint, anorm: ^f64, rcond: ^f64, work: ^complex128, info: ^Info, _: c.size_t) ---

	checon_3_ :: proc(uplo: cstring, n: ^blasint, A: ^complex64, lda: ^blasint, E: ^complex64, ipiv: ^blasint, anorm: ^f32, rcond: ^f32, work: ^complex64, info: ^Info, _: c.size_t) ---
	zhecon_3_ :: proc(uplo: cstring, n: ^blasint, A: ^complex128, lda: ^blasint, E: ^complex128, ipiv: ^blasint, anorm: ^f64, rcond: ^f64, work: ^complex128, info: ^Info, _: c.size_t) ---
	cheequb_ :: proc(uplo: cstring, n: ^blasint, A: ^complex64, lda: ^blasint, S: ^f32, scond: ^f32, amax: ^f32, work: ^complex64, info: ^Info, _: c.size_t) ---
	zheequb_ :: proc(uplo: cstring, n: ^blasint, A: ^complex128, lda: ^blasint, S: ^f64, scond: ^f64, amax: ^f64, work: ^complex128, info: ^Info, _: c.size_t) ---

	cheev_ :: proc(jobz: cstring, uplo: cstring, n: ^blasint, A: ^complex64, lda: ^blasint, W: ^f32, work: ^complex64, lwork: ^blasint, rwork: ^f32, info: ^Info, _: c.size_t, _: c.size_t) ---
	zheev_ :: proc(jobz: cstring, uplo: cstring, n: ^blasint, A: ^complex128, lda: ^blasint, W: ^f64, work: ^complex128, lwork: ^blasint, rwork: ^f64, info: ^Info, _: c.size_t, _: c.size_t) ---
	cheev_2stage_ :: proc(jobz: cstring, uplo: cstring, n: ^blasint, A: ^complex64, lda: ^blasint, W: ^f32, work: ^complex64, lwork: ^blasint, rwork: ^f32, info: ^Info, _: c.size_t, _: c.size_t) ---
	zheev_2stage_ :: proc(jobz: cstring, uplo: cstring, n: ^blasint, A: ^complex128, lda: ^blasint, W: ^f64, work: ^complex128, lwork: ^blasint, rwork: ^f64, info: ^Info, _: c.size_t, _: c.size_t) ---

	cheevd_ :: proc(jobz: cstring, uplo: cstring, n: ^blasint, A: ^complex64, lda: ^blasint, W: ^f32, work: ^complex64, lwork: ^blasint, rwork: ^f32, lrwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	zheevd_ :: proc(jobz: cstring, uplo: cstring, n: ^blasint, A: ^complex128, lda: ^blasint, W: ^f64, work: ^complex128, lwork: ^blasint, rwork: ^f64, lrwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	cheevd_2stage_ :: proc(jobz: cstring, uplo: cstring, n: ^blasint, A: ^complex64, lda: ^blasint, W: ^f32, work: ^complex64, lwork: ^blasint, rwork: ^f32, lrwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	zheevd_2stage_ :: proc(jobz: cstring, uplo: cstring, n: ^blasint, A: ^complex128, lda: ^blasint, W: ^f64, work: ^complex128, lwork: ^blasint, rwork: ^f64, lrwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---

	cheevr_ :: proc(jobz: cstring, range: cstring, uplo: cstring, n: ^blasint, A: ^complex64, lda: ^blasint, vl: ^f32, vu: ^f32, il: ^blasint, iu: ^blasint, abstol: ^f32, m: ^blasint, W: ^f32, Z: ^complex64, ldz: ^blasint, ISUPPZ: ^blasint, work: ^complex64, lwork: ^blasint, rwork: ^f32, lrwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	zheevr_ :: proc(jobz: cstring, range: cstring, uplo: cstring, n: ^blasint, A: ^complex128, lda: ^blasint, vl: ^f64, vu: ^f64, il: ^blasint, iu: ^blasint, abstol: ^f64, m: ^blasint, W: ^f64, Z: ^complex128, ldz: ^blasint, ISUPPZ: ^blasint, work: ^complex128, lwork: ^blasint, rwork: ^f64, lrwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	cheevr_2stage_ :: proc(jobz: cstring, range: cstring, uplo: cstring, n: ^blasint, A: ^complex64, lda: ^blasint, vl: ^f32, vu: ^f32, il: ^blasint, iu: ^blasint, abstol: ^f32, m: ^blasint, W: ^f32, Z: ^complex64, ldz: ^blasint, ISUPPZ: ^blasint, work: ^complex64, lwork: ^blasint, rwork: ^f32, lrwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	zheevr_2stage_ :: proc(jobz: cstring, range: cstring, uplo: cstring, n: ^blasint, A: ^complex128, lda: ^blasint, vl: ^f64, vu: ^f64, il: ^blasint, iu: ^blasint, abstol: ^f64, m: ^blasint, W: ^f64, Z: ^complex128, ldz: ^blasint, ISUPPZ: ^blasint, work: ^complex128, lwork: ^blasint, rwork: ^f64, lrwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---

	cheevx_ :: proc(jobz: cstring, range: cstring, uplo: cstring, n: ^blasint, A: ^complex64, lda: ^blasint, vl: ^f32, vu: ^f32, il: ^blasint, iu: ^blasint, abstol: ^f32, m: ^blasint, W: ^f32, Z: ^complex64, ldz: ^blasint, work: ^complex64, lwork: ^blasint, rwork: ^f32, iwork: ^blasint, IFAIL: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	zheevx_ :: proc(jobz: cstring, range: cstring, uplo: cstring, n: ^blasint, A: ^complex128, lda: ^blasint, vl: ^f64, vu: ^f64, il: ^blasint, iu: ^blasint, abstol: ^f64, m: ^blasint, W: ^f64, Z: ^complex128, ldz: ^blasint, work: ^complex128, lwork: ^blasint, rwork: ^f64, iwork: ^blasint, IFAIL: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	cheevx_2stage_ :: proc(jobz: cstring, range: cstring, uplo: cstring, n: ^blasint, A: ^complex64, lda: ^blasint, vl: ^f32, vu: ^f32, il: ^blasint, iu: ^blasint, abstol: ^f32, m: ^blasint, W: ^f32, Z: ^complex64, ldz: ^blasint, work: ^complex64, lwork: ^blasint, rwork: ^f32, iwork: ^blasint, IFAIL: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	zheevx_2stage_ :: proc(jobz: cstring, range: cstring, uplo: cstring, n: ^blasint, A: ^complex128, lda: ^blasint, vl: ^f64, vu: ^f64, il: ^blasint, iu: ^blasint, abstol: ^f64, m: ^blasint, W: ^f64, Z: ^complex128, ldz: ^blasint, work: ^complex128, lwork: ^blasint, rwork: ^f64, iwork: ^blasint, IFAIL: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---

	chegst_ :: proc(itype: ^blasint, uplo: cstring, n: ^blasint, A: ^complex64, lda: ^blasint, B: ^complex64, ldb: ^blasint, info: ^Info, _: c.size_t) ---
	zhegst_ :: proc(itype: ^blasint, uplo: cstring, n: ^blasint, A: ^complex128, lda: ^blasint, B: ^complex128, ldb: ^blasint, info: ^Info, _: c.size_t) ---
	chegv_ :: proc(itype: ^blasint, jobz: cstring, uplo: cstring, n: ^blasint, A: ^complex64, lda: ^blasint, B: ^complex64, ldb: ^blasint, W: ^f32, work: ^complex64, lwork: ^blasint, rwork: ^f32, info: ^Info, _: c.size_t, _: c.size_t) ---
	zhegv_ :: proc(itype: ^blasint, jobz: cstring, uplo: cstring, n: ^blasint, A: ^complex128, lda: ^blasint, B: ^complex128, ldb: ^blasint, W: ^f64, work: ^complex128, lwork: ^blasint, rwork: ^f64, info: ^Info, _: c.size_t, _: c.size_t) ---
	chegv_2stage_ :: proc(itype: ^blasint, jobz: cstring, uplo: cstring, n: ^blasint, A: ^complex64, lda: ^blasint, B: ^complex64, ldb: ^blasint, W: ^f32, work: ^complex64, lwork: ^blasint, rwork: ^f32, info: ^Info, _: c.size_t, _: c.size_t) ---
	zhegv_2stage_ :: proc(itype: ^blasint, jobz: cstring, uplo: cstring, n: ^blasint, A: ^complex128, lda: ^blasint, B: ^complex128, ldb: ^blasint, W: ^f64, work: ^complex128, lwork: ^blasint, rwork: ^f64, info: ^Info, _: c.size_t, _: c.size_t) ---
	chegvd_ :: proc(itype: ^blasint, jobz: cstring, uplo: cstring, n: ^blasint, A: ^complex64, lda: ^blasint, B: ^complex64, ldb: ^blasint, W: ^f32, work: ^complex64, lwork: ^blasint, rwork: ^f32, lrwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	zhegvd_ :: proc(itype: ^blasint, jobz: cstring, uplo: cstring, n: ^blasint, A: ^complex128, lda: ^blasint, B: ^complex128, ldb: ^blasint, W: ^f64, work: ^complex128, lwork: ^blasint, rwork: ^f64, lrwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	chegvx_ :: proc(itype: ^blasint, jobz: cstring, range: cstring, uplo: cstring, n: ^blasint, A: ^complex64, lda: ^blasint, B: ^complex64, ldb: ^blasint, vl: ^f32, vu: ^f32, il: ^blasint, iu: ^blasint, abstol: ^f32, m: ^blasint, W: ^f32, Z: ^complex64, ldz: ^blasint, work: ^complex64, lwork: ^blasint, rwork: ^f32, iwork: ^blasint, IFAIL: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	zhegvx_ :: proc(itype: ^blasint, jobz: cstring, range: cstring, uplo: cstring, n: ^blasint, A: ^complex128, lda: ^blasint, B: ^complex128, ldb: ^blasint, vl: ^f64, vu: ^f64, il: ^blasint, iu: ^blasint, abstol: ^f64, m: ^blasint, W: ^f64, Z: ^complex128, ldz: ^blasint, work: ^complex128, lwork: ^blasint, rwork: ^f64, iwork: ^blasint, IFAIL: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---

	cherfs_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex64, lda: ^blasint, AF: ^complex64, ldaf: ^blasint, ipiv: ^blasint, B: ^complex64, ldb: ^blasint, X: ^complex64, ldx: ^blasint, ferr: ^f32, berr: ^f32, work: ^complex64, rwork: ^f32, info: ^Info, _: c.size_t) ---
	zherfs_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex128, lda: ^blasint, AF: ^complex128, ldaf: ^blasint, ipiv: ^blasint, B: ^complex128, ldb: ^blasint, X: ^complex128, ldx: ^blasint, ferr: ^f64, berr: ^f64, work: ^complex128, rwork: ^f64, info: ^Info, _: c.size_t) ---
	cherfsx_ :: proc(uplo: cstring, equed: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex64, lda: ^blasint, AF: ^complex64, ldaf: ^blasint, ipiv: ^blasint, S: ^f32, B: ^complex64, ldb: ^blasint, X: ^complex64, ldx: ^blasint, rcond: ^f32, berr: ^f32, n_err_bnds: ^blasint, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: ^blasint, params: ^f32, work: ^complex64, rwork: ^f32, info: ^Info, _: c.size_t, _: c.size_t) ---
	zherfsx_ :: proc(uplo: cstring, equed: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex128, lda: ^blasint, AF: ^complex128, ldaf: ^blasint, ipiv: ^blasint, S: ^f64, B: ^complex128, ldb: ^blasint, X: ^complex128, ldx: ^blasint, rcond: ^f64, berr: ^f64, n_err_bnds: ^blasint, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: ^blasint, params: ^f64, work: ^complex128, rwork: ^f64, info: ^Info, _: c.size_t, _: c.size_t) ---

	chesv_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex64, lda: ^blasint, ipiv: ^blasint, B: ^complex64, ldb: ^blasint, work: ^complex64, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	zhesv_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex128, lda: ^blasint, ipiv: ^blasint, B: ^complex128, ldb: ^blasint, work: ^complex128, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	chesv_aa_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex64, lda: ^blasint, ipiv: ^blasint, B: ^complex64, ldb: ^blasint, work: ^complex64, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	zhesv_aa_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex128, lda: ^blasint, ipiv: ^blasint, B: ^complex128, ldb: ^blasint, work: ^complex128, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	chesv_aa_2stage_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex64, lda: ^blasint, TB: ^complex64, ltb: ^blasint, ipiv: ^blasint, ipiv2: ^blasint, B: ^complex64, ldb: ^blasint, work: ^complex64, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	zhesv_aa_2stage_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex128, lda: ^blasint, TB: ^complex128, ltb: ^blasint, ipiv: ^blasint, ipiv2: ^blasint, B: ^complex128, ldb: ^blasint, work: ^complex128, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	chesv_rk_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex64, lda: ^blasint, E: ^complex64, ipiv: ^blasint, B: ^complex64, ldb: ^blasint, work: ^complex64, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	zhesv_rk_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex128, lda: ^blasint, E: ^complex128, ipiv: ^blasint, B: ^complex128, ldb: ^blasint, work: ^complex128, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	chesv_rook_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex64, lda: ^blasint, ipiv: ^blasint, B: ^complex64, ldb: ^blasint, work: ^complex64, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	zhesv_rook_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex128, lda: ^blasint, ipiv: ^blasint, B: ^complex128, ldb: ^blasint, work: ^complex128, lwork: ^blasint, info: ^Info, _: c.size_t) ---

	chesvx_ :: proc(fact: cstring, uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex64, lda: ^blasint, AF: ^complex64, ldaf: ^blasint, ipiv: ^blasint, B: ^complex64, ldb: ^blasint, X: ^complex64, ldx: ^blasint, rcond: ^f32, ferr: ^f32, berr: ^f32, work: ^complex64, lwork: ^blasint, rwork: ^f32, info: ^Info, _: c.size_t, _: c.size_t) ---
	zhesvx_ :: proc(fact: cstring, uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex128, lda: ^blasint, AF: ^complex128, ldaf: ^blasint, ipiv: ^blasint, B: ^complex128, ldb: ^blasint, X: ^complex128, ldx: ^blasint, rcond: ^f64, ferr: ^f64, berr: ^f64, work: ^complex128, lwork: ^blasint, rwork: ^f64, info: ^Info, _: c.size_t, _: c.size_t) ---
	chesvxx_ :: proc(fact: cstring, uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex64, lda: ^blasint, AF: ^complex64, ldaf: ^blasint, ipiv: ^blasint, equed: cstring, S: ^f32, B: ^complex64, ldb: ^blasint, X: ^complex64, ldx: ^blasint, rcond: ^f32, rpvgrw: ^f32, berr: ^f32, n_err_bnds: ^blasint, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: ^blasint, params: ^f32, work: ^complex64, rwork: ^f32, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	zhesvxx_ :: proc(fact: cstring, uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex128, lda: ^blasint, AF: ^complex128, ldaf: ^blasint, ipiv: ^blasint, equed: cstring, S: ^f64, B: ^complex128, ldb: ^blasint, X: ^complex128, ldx: ^blasint, rcond: ^f64, rpvgrw: ^f64, berr: ^f64, n_err_bnds: ^blasint, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: ^blasint, params: ^f64, work: ^complex128, rwork: ^f64, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---

	cheswapr_ :: proc(uplo: cstring, n: ^blasint, A: ^complex64, lda: ^blasint, i1: ^blasint, i2: ^blasint, _: c.size_t) ---
	zheswapr_ :: proc(uplo: cstring, n: ^blasint, A: ^complex128, lda: ^blasint, i1: ^blasint, i2: ^blasint, _: c.size_t) ---

	chetrd_ :: proc(uplo: cstring, n: ^blasint, A: ^complex64, lda: ^blasint, D: ^f32, E: ^f32, tau: ^complex64, work: ^complex64, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	zhetrd_ :: proc(uplo: cstring, n: ^blasint, A: ^complex128, lda: ^blasint, D: ^f64, E: ^f64, tau: ^complex128, work: ^complex128, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	chetrd_2stage_ :: proc(vect: cstring, uplo: cstring, n: ^blasint, A: ^complex64, lda: ^blasint, D: ^f32, E: ^f32, tau: ^complex64, HOUS2: ^complex64, lhous2: ^blasint, work: ^complex64, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	zhetrd_2stage_ :: proc(vect: cstring, uplo: cstring, n: ^blasint, A: ^complex128, lda: ^blasint, D: ^f64, E: ^f64, tau: ^complex128, HOUS2: ^complex128, lhous2: ^blasint, work: ^complex128, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	chetrf_ :: proc(uplo: cstring, n: ^blasint, A: ^complex64, lda: ^blasint, ipiv: ^blasint, work: ^complex64, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	zhetrf_ :: proc(uplo: cstring, n: ^blasint, A: ^complex128, lda: ^blasint, ipiv: ^blasint, work: ^complex128, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	chetrf_aa_ :: proc(uplo: cstring, n: ^blasint, A: ^complex64, lda: ^blasint, ipiv: ^blasint, work: ^complex64, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	zhetrf_aa_ :: proc(uplo: cstring, n: ^blasint, A: ^complex128, lda: ^blasint, ipiv: ^blasint, work: ^complex128, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	chetrf_aa_2stage_ :: proc(uplo: cstring, n: ^blasint, A: ^complex64, lda: ^blasint, TB: ^complex64, ltb: ^blasint, ipiv: ^blasint, ipiv2: ^blasint, work: ^complex64, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	zhetrf_aa_2stage_ :: proc(uplo: cstring, n: ^blasint, A: ^complex128, lda: ^blasint, TB: ^complex128, ltb: ^blasint, ipiv: ^blasint, ipiv2: ^blasint, work: ^complex128, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	chetrf_rk_ :: proc(uplo: cstring, n: ^blasint, A: ^complex64, lda: ^blasint, E: ^complex64, ipiv: ^blasint, work: ^complex64, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	zhetrf_rk_ :: proc(uplo: cstring, n: ^blasint, A: ^complex128, lda: ^blasint, E: ^complex128, ipiv: ^blasint, work: ^complex128, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	chetrf_rook_ :: proc(uplo: cstring, n: ^blasint, A: ^complex64, lda: ^blasint, ipiv: ^blasint, work: ^complex64, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	zhetrf_rook_ :: proc(uplo: cstring, n: ^blasint, A: ^complex128, lda: ^blasint, ipiv: ^blasint, work: ^complex128, lwork: ^blasint, info: ^Info, _: c.size_t) ---

	chetri_ :: proc(uplo: cstring, n: ^blasint, A: ^complex64, lda: ^blasint, ipiv: ^blasint, work: ^complex64, info: ^Info, _: c.size_t) ---
	zhetri_ :: proc(uplo: cstring, n: ^blasint, A: ^complex128, lda: ^blasint, ipiv: ^blasint, work: ^complex128, info: ^Info, _: c.size_t) ---
	chetri2_ :: proc(uplo: cstring, n: ^blasint, A: ^complex64, lda: ^blasint, ipiv: ^blasint, work: ^complex64, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	zhetri2_ :: proc(uplo: cstring, n: ^blasint, A: ^complex128, lda: ^blasint, ipiv: ^blasint, work: ^complex128, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	chetri2x_ :: proc(uplo: cstring, n: ^blasint, A: ^complex64, lda: ^blasint, ipiv: ^blasint, work: ^complex64, nb: ^blasint, info: ^Info, _: c.size_t) ---
	zhetri2x_ :: proc(uplo: cstring, n: ^blasint, A: ^complex128, lda: ^blasint, ipiv: ^blasint, work: ^complex128, nb: ^blasint, info: ^Info, _: c.size_t) ---
	chetri_3_ :: proc(uplo: cstring, n: ^blasint, A: ^complex64, lda: ^blasint, E: ^complex64, ipiv: ^blasint, work: ^complex64, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	zhetri_3_ :: proc(uplo: cstring, n: ^blasint, A: ^complex128, lda: ^blasint, E: ^complex128, ipiv: ^blasint, work: ^complex128, lwork: ^blasint, info: ^Info, _: c.size_t) ---

	chetrs_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex64, lda: ^blasint, ipiv: ^blasint, B: ^complex64, ldb: ^blasint, info: ^Info, _: c.size_t) ---
	zhetrs_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex128, lda: ^blasint, ipiv: ^blasint, B: ^complex128, ldb: ^blasint, info: ^Info, _: c.size_t) ---
	chetrs2_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex64, lda: ^blasint, ipiv: ^blasint, B: ^complex64, ldb: ^blasint, work: ^complex64, info: ^Info, _: c.size_t) ---
	zhetrs2_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex128, lda: ^blasint, ipiv: ^blasint, B: ^complex128, ldb: ^blasint, work: ^complex128, info: ^Info, _: c.size_t) ---
	chetrs_3_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex64, lda: ^blasint, E: ^complex64, ipiv: ^blasint, B: ^complex64, ldb: ^blasint, info: ^Info, _: c.size_t) ---
	zhetrs_3_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex128, lda: ^blasint, E: ^complex128, ipiv: ^blasint, B: ^complex128, ldb: ^blasint, info: ^Info, _: c.size_t) ---
	chetrs_aa_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex64, lda: ^blasint, ipiv: ^blasint, B: ^complex64, ldb: ^blasint, work: ^complex64, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	zhetrs_aa_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex128, lda: ^blasint, ipiv: ^blasint, B: ^complex128, ldb: ^blasint, work: ^complex128, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	chetrs_aa_2stage_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex64, lda: ^blasint, TB: ^complex64, ltb: ^blasint, ipiv: ^blasint, ipiv2: ^blasint, B: ^complex64, ldb: ^blasint, info: ^Info, _: c.size_t) ---
	zhetrs_aa_2stage_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex128, lda: ^blasint, TB: ^complex128, ltb: ^blasint, ipiv: ^blasint, ipiv2: ^blasint, B: ^complex128, ldb: ^blasint, info: ^Info, _: c.size_t) ---
	chetrs_rook_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex64, lda: ^blasint, ipiv: ^blasint, B: ^complex64, ldb: ^blasint, info: ^Info, _: c.size_t) ---
	zhetrs_rook_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex128, lda: ^blasint, ipiv: ^blasint, B: ^complex128, ldb: ^blasint, info: ^Info, _: c.size_t) ---

	chfrk_ :: proc(transr: cstring, uplo: cstring, trans: cstring, n: ^blasint, k: ^blasint, alpha: ^f32, A: ^complex64, lda: ^blasint, beta: ^f32, C: ^complex64, _: c.size_t, _: c.size_t, _: c.size_t) ---
	zhfrk_ :: proc(transr: cstring, uplo: cstring, trans: cstring, n: ^blasint, k: ^blasint, alpha: ^f64, A: ^complex128, lda: ^blasint, beta: ^f64, C: ^complex128, _: c.size_t, _: c.size_t, _: c.size_t) ---
	chgeqz_ :: proc(job: cstring, compq: cstring, compz: cstring, n: ^blasint, ilo: ^blasint, ihi: ^blasint, H: ^complex64, ldh: ^blasint, T: ^complex64, ldt: ^blasint, alpha: ^complex64, beta: ^complex64, Q: ^complex64, ldq: ^blasint, Z: ^complex64, ldz: ^blasint, work: ^complex64, lwork: ^blasint, rwork: ^f32, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	dhgeqz_ :: proc(job: cstring, compq: cstring, compz: cstring, n: ^blasint, ilo: ^blasint, ihi: ^blasint, H: ^f64, ldh: ^blasint, T: ^f64, ldt: ^blasint, alphar: ^f64, alphai: ^f64, beta: ^f64, Q: ^f64, ldq: ^blasint, Z: ^f64, ldz: ^blasint, work: ^f64, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	shgeqz_ :: proc(job: cstring, compq: cstring, compz: cstring, n: ^blasint, ilo: ^blasint, ihi: ^blasint, H: ^f32, ldh: ^blasint, T: ^f32, ldt: ^blasint, alphar: ^f32, alphai: ^f32, beta: ^f32, Q: ^f32, ldq: ^blasint, Z: ^f32, ldz: ^blasint, work: ^f32, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	zhgeqz_ :: proc(job: cstring, compq: cstring, compz: cstring, n: ^blasint, ilo: ^blasint, ihi: ^blasint, H: ^complex128, ldh: ^blasint, T: ^complex128, ldt: ^blasint, alpha: ^complex128, beta: ^complex128, Q: ^complex128, ldq: ^blasint, Z: ^complex128, ldz: ^blasint, work: ^complex128, lwork: ^blasint, rwork: ^f64, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---

	chpcon_ :: proc(uplo: cstring, n: ^blasint, AP: ^complex64, ipiv: ^blasint, anorm: ^f32, rcond: ^f32, work: ^complex64, info: ^Info, _: c.size_t) ---
	zhpcon_ :: proc(uplo: cstring, n: ^blasint, AP: ^complex128, ipiv: ^blasint, anorm: ^f64, rcond: ^f64, work: ^complex128, info: ^Info, _: c.size_t) ---
	chpev_ :: proc(jobz: cstring, uplo: cstring, n: ^blasint, AP: ^complex64, W: ^f32, Z: ^complex64, ldz: ^blasint, work: ^complex64, rwork: ^f32, info: ^Info, _: c.size_t, _: c.size_t) ---
	zhpev_ :: proc(jobz: cstring, uplo: cstring, n: ^blasint, AP: ^complex128, W: ^f64, Z: ^complex128, ldz: ^blasint, work: ^complex128, rwork: ^f64, info: ^Info, _: c.size_t, _: c.size_t) ---

	chpevd_ :: proc(jobz: cstring, uplo: cstring, n: ^blasint, AP: ^complex64, W: ^f32, Z: ^complex64, ldz: ^blasint, work: ^complex64, lwork: ^blasint, rwork: ^f32, lrwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	zhpevd_ :: proc(jobz: cstring, uplo: cstring, n: ^blasint, AP: ^complex128, W: ^f64, Z: ^complex128, ldz: ^blasint, work: ^complex128, lwork: ^blasint, rwork: ^f64, lrwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	chpevx_ :: proc(jobz: cstring, range: cstring, uplo: cstring, n: ^blasint, AP: ^complex64, vl: ^f32, vu: ^f32, il: ^blasint, iu: ^blasint, abstol: ^f32, m: ^blasint, W: ^f32, Z: ^complex64, ldz: ^blasint, work: ^complex64, rwork: ^f32, iwork: ^blasint, IFAIL: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	zhpevx_ :: proc(jobz: cstring, range: cstring, uplo: cstring, n: ^blasint, AP: ^complex128, vl: ^f64, vu: ^f64, il: ^blasint, iu: ^blasint, abstol: ^f64, m: ^blasint, W: ^f64, Z: ^complex128, ldz: ^blasint, work: ^complex128, rwork: ^f64, iwork: ^blasint, IFAIL: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---

	chpgst_ :: proc(itype: ^blasint, uplo: cstring, n: ^blasint, AP: ^complex64, BP: ^complex64, info: ^Info, _: c.size_t) ---
	zhpgst_ :: proc(itype: ^blasint, uplo: cstring, n: ^blasint, AP: ^complex128, BP: ^complex128, info: ^Info, _: c.size_t) ---
	chpgv_ :: proc(itype: ^blasint, jobz: cstring, uplo: cstring, n: ^blasint, AP: ^complex64, BP: ^complex64, W: ^f32, Z: ^complex64, ldz: ^blasint, work: ^complex64, rwork: ^f32, info: ^Info, _: c.size_t, _: c.size_t) ---
	zhpgv_ :: proc(itype: ^blasint, jobz: cstring, uplo: cstring, n: ^blasint, AP: ^complex128, BP: ^complex128, W: ^f64, Z: ^complex128, ldz: ^blasint, work: ^complex128, rwork: ^f64, info: ^Info, _: c.size_t, _: c.size_t) ---

	chpgvd_ :: proc(itype: ^blasint, jobz: cstring, uplo: cstring, n: ^blasint, AP: ^complex64, BP: ^complex64, W: ^f32, Z: ^complex64, ldz: ^blasint, work: ^complex64, lwork: ^blasint, rwork: ^f32, lrwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	zhpgvd_ :: proc(itype: ^blasint, jobz: cstring, uplo: cstring, n: ^blasint, AP: ^complex128, BP: ^complex128, W: ^f64, Z: ^complex128, ldz: ^blasint, work: ^complex128, lwork: ^blasint, rwork: ^f64, lrwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	chpgvx_ :: proc(itype: ^blasint, jobz: cstring, range: cstring, uplo: cstring, n: ^blasint, AP: ^complex64, BP: ^complex64, vl: ^f32, vu: ^f32, il: ^blasint, iu: ^blasint, abstol: ^f32, m: ^blasint, W: ^f32, Z: ^complex64, ldz: ^blasint, work: ^complex64, rwork: ^f32, iwork: ^blasint, IFAIL: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	zhpgvx_ :: proc(itype: ^blasint, jobz: cstring, range: cstring, uplo: cstring, n: ^blasint, AP: ^complex128, BP: ^complex128, vl: ^f64, vu: ^f64, il: ^blasint, iu: ^blasint, abstol: ^f64, m: ^blasint, W: ^f64, Z: ^complex128, ldz: ^blasint, work: ^complex128, rwork: ^f64, iwork: ^blasint, IFAIL: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---

	chprfs_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, AP: ^complex64, AFP: ^complex64, ipiv: ^blasint, B: ^complex64, ldb: ^blasint, X: ^complex64, ldx: ^blasint, ferr: ^f32, berr: ^f32, work: ^complex64, rwork: ^f32, info: ^Info, _: c.size_t) ---
	zhprfs_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, AP: ^complex128, AFP: ^complex128, ipiv: ^blasint, B: ^complex128, ldb: ^blasint, X: ^complex128, ldx: ^blasint, ferr: ^f64, berr: ^f64, work: ^complex128, rwork: ^f64, info: ^Info, _: c.size_t) ---
	chpsv_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, AP: ^complex64, ipiv: ^blasint, B: ^complex64, ldb: ^blasint, info: ^Info, _: c.size_t) ---
	zhpsv_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, AP: ^complex128, ipiv: ^blasint, B: ^complex128, ldb: ^blasint, info: ^Info, _: c.size_t) ---

	chpsvx_ :: proc(fact: cstring, uplo: cstring, n: ^blasint, nrhs: ^blasint, AP: ^complex64, AFP: ^complex64, ipiv: ^blasint, B: ^complex64, ldb: ^blasint, X: ^complex64, ldx: ^blasint, rcond: ^f32, ferr: ^f32, berr: ^f32, work: ^complex64, rwork: ^f32, info: ^Info, _: c.size_t, _: c.size_t) ---
	zhpsvx_ :: proc(fact: cstring, uplo: cstring, n: ^blasint, nrhs: ^blasint, AP: ^complex128, AFP: ^complex128, ipiv: ^blasint, B: ^complex128, ldb: ^blasint, X: ^complex128, ldx: ^blasint, rcond: ^f64, ferr: ^f64, berr: ^f64, work: ^complex128, rwork: ^f64, info: ^Info, _: c.size_t, _: c.size_t) ---
	chptrd_ :: proc(uplo: cstring, n: ^blasint, AP: ^complex64, D: ^f32, E: ^f32, tau: ^complex64, info: ^Info, _: c.size_t) ---
	zhptrd_ :: proc(uplo: cstring, n: ^blasint, AP: ^complex128, D: ^f64, E: ^f64, tau: ^complex128, info: ^Info, _: c.size_t) ---

	chptrf_ :: proc(uplo: cstring, n: ^blasint, AP: ^complex64, ipiv: ^blasint, info: ^Info, _: c.size_t) ---
	zhptrf_ :: proc(uplo: cstring, n: ^blasint, AP: ^complex128, ipiv: ^blasint, info: ^Info, _: c.size_t) ---
	chptri_ :: proc(uplo: cstring, n: ^blasint, AP: ^complex64, ipiv: ^blasint, work: ^complex64, info: ^Info, _: c.size_t) ---
	zhptri_ :: proc(uplo: cstring, n: ^blasint, AP: ^complex128, ipiv: ^blasint, work: ^complex128, info: ^Info, _: c.size_t) ---

	chptrs_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, AP: ^complex64, ipiv: ^blasint, B: ^complex64, ldb: ^blasint, info: ^Info, _: c.size_t) ---
	zhptrs_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, AP: ^complex128, ipiv: ^blasint, B: ^complex128, ldb: ^blasint, info: ^Info, _: c.size_t) ---

	chsein_ :: proc(side: cstring, eigsrc: cstring, initv: cstring, select: ^blasint, n: ^blasint, H: ^complex64, ldh: ^blasint, W: ^complex64, VL: ^complex64, ldvl: ^blasint, VR: ^complex64, ldvr: ^blasint, mm: ^blasint, m: ^blasint, work: ^complex64, rwork: ^f32, IFAILL: ^blasint, IFAILR: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	dhsein_ :: proc(side: cstring, eigsrc: cstring, initv: cstring, select: ^blasint, n: ^blasint, H: ^f64, ldh: ^blasint, WR: ^f64, WI: ^f64, VL: ^f64, ldvl: ^blasint, VR: ^f64, ldvr: ^blasint, mm: ^blasint, m: ^blasint, work: ^f64, IFAILL: ^blasint, IFAILR: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	shsein_ :: proc(side: cstring, eigsrc: cstring, initv: cstring, select: ^blasint, n: ^blasint, H: ^f32, ldh: ^blasint, WR: ^f32, WI: ^f32, VL: ^f32, ldvl: ^blasint, VR: ^f32, ldvr: ^blasint, mm: ^blasint, m: ^blasint, work: ^f32, IFAILL: ^blasint, IFAILR: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	zhsein_ :: proc(side: cstring, eigsrc: cstring, initv: cstring, select: ^blasint, n: ^blasint, H: ^complex128, ldh: ^blasint, W: ^complex128, VL: ^complex128, ldvl: ^blasint, VR: ^complex128, ldvr: ^blasint, mm: ^blasint, m: ^blasint, work: ^complex128, rwork: ^f64, IFAILL: ^blasint, IFAILR: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---

	chseqr_ :: proc(job: cstring, compz: cstring, n: ^blasint, ilo: ^blasint, ihi: ^blasint, H: ^complex64, ldh: ^blasint, W: ^complex64, Z: ^complex64, ldz: ^blasint, work: ^complex64, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	dhseqr_ :: proc(job: cstring, compz: cstring, n: ^blasint, ilo: ^blasint, ihi: ^blasint, H: ^f64, ldh: ^blasint, WR: ^f64, WI: ^f64, Z: ^f64, ldz: ^blasint, work: ^f64, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	shseqr_ :: proc(job: cstring, compz: cstring, n: ^blasint, ilo: ^blasint, ihi: ^blasint, H: ^f32, ldh: ^blasint, WR: ^f32, WI: ^f32, Z: ^f32, ldz: ^blasint, work: ^f32, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	zhseqr_ :: proc(job: cstring, compz: cstring, n: ^blasint, ilo: ^blasint, ihi: ^blasint, H: ^complex128, ldh: ^blasint, W: ^complex128, Z: ^complex128, ldz: ^blasint, work: ^complex128, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---

	clacgv_ :: proc(n: ^blasint, X: ^complex64, incx: ^blasint) ---
	zlacgv_ :: proc(n: ^blasint, X: ^complex128, incx: ^blasint) ---

	clacn2_ :: proc(n: ^blasint, V: ^complex64, X: ^complex64, est: ^f32, kase: ^blasint, ISAVE: ^blasint) ---
	dlacn2_ :: proc(n: ^blasint, V: ^f64, X: ^f64, ISGN: ^blasint, est: ^f64, kase: ^blasint, ISAVE: ^blasint) ---
	slacn2_ :: proc(n: ^blasint, V: ^f32, X: ^f32, ISGN: ^blasint, est: ^f32, kase: ^blasint, ISAVE: ^blasint) ---
	zlacn2_ :: proc(n: ^blasint, V: ^complex128, X: ^complex128, est: ^f64, kase: ^blasint, ISAVE: ^blasint) ---

	clacp2_ :: proc(uplo: cstring, m: ^blasint, n: ^blasint, A: ^f32, lda: ^blasint, B: ^complex64, ldb: ^blasint, _: c.size_t) ---
	zlacp2_ :: proc(uplo: cstring, m: ^blasint, n: ^blasint, A: ^f64, lda: ^blasint, B: ^complex128, ldb: ^blasint, _: c.size_t) ---

	clacpy_ :: proc(uplo: cstring, m: ^blasint, n: ^blasint, A: ^complex64, lda: ^blasint, B: ^complex64, ldb: ^blasint, _: c.size_t) ---
	dlacpy_ :: proc(uplo: cstring, m: ^blasint, n: ^blasint, A: ^f64, lda: ^blasint, B: ^f64, ldb: ^blasint, _: c.size_t) ---
	slacpy_ :: proc(uplo: cstring, m: ^blasint, n: ^blasint, A: ^f32, lda: ^blasint, B: ^f32, ldb: ^blasint, _: c.size_t) ---
	zlacpy_ :: proc(uplo: cstring, m: ^blasint, n: ^blasint, A: ^complex128, lda: ^blasint, B: ^complex128, ldb: ^blasint, _: c.size_t) ---

	clacrm_ :: proc(m: ^blasint, n: ^blasint, A: ^complex64, lda: ^blasint, B: ^f32, ldb: ^blasint, C: ^complex64, ldc: ^blasint, rwork: ^f32) ---
	zlacrm_ :: proc(m: ^blasint, n: ^blasint, A: ^complex128, lda: ^blasint, B: ^f64, ldb: ^blasint, C: ^complex128, ldc: ^blasint, rwork: ^f64) ---

	zlag2c_ :: proc(m: ^blasint, n: ^blasint, A: ^complex128, lda: ^blasint, SA: ^complex64, ldsa: ^blasint, info: ^Info) ---
	slag2d_ :: proc(m: ^blasint, n: ^blasint, SA: ^f32, ldsa: ^blasint, A: ^f64, lda: ^blasint, info: ^Info) ---
	dlag2s_ :: proc(m: ^blasint, n: ^blasint, A: ^f64, lda: ^blasint, SA: ^f32, ldsa: ^blasint, info: ^Info) ---
	clag2z_ :: proc(m: ^blasint, n: ^blasint, SA: ^complex64, ldsa: ^blasint, A: ^complex128, lda: ^blasint, info: ^Info) ---
	clagge_ :: proc(m: ^blasint, n: ^blasint, kl: ^blasint, ku: ^blasint, D: ^f32, A: ^complex64, lda: ^blasint, iseed: ^blasint, work: ^complex64, info: ^Info) ---
	dlagge_ :: proc(m: ^blasint, n: ^blasint, kl: ^blasint, ku: ^blasint, D: ^f64, A: ^f64, lda: ^blasint, iseed: ^blasint, work: ^f64, info: ^Info) ---
	slagge_ :: proc(m: ^blasint, n: ^blasint, kl: ^blasint, ku: ^blasint, D: ^f32, A: ^f32, lda: ^blasint, iseed: ^blasint, work: ^f32, info: ^Info) ---
	zlagge_ :: proc(m: ^blasint, n: ^blasint, kl: ^blasint, ku: ^blasint, D: ^f64, A: ^complex128, lda: ^blasint, iseed: ^blasint, work: ^complex128, info: ^Info) ---

	claghe_ :: proc(n: ^blasint, k: ^blasint, D: ^f32, A: ^complex64, lda: ^blasint, iseed: ^blasint, work: ^complex64, info: ^Info) ---
	zlaghe_ :: proc(n: ^blasint, k: ^blasint, D: ^f64, A: ^complex128, lda: ^blasint, iseed: ^blasint, work: ^complex128, info: ^Info) ---
	clagsy_ :: proc(n: ^blasint, k: ^blasint, D: ^f32, A: ^complex64, lda: ^blasint, iseed: ^blasint, work: ^complex64, info: ^Info) ---
	dlagsy_ :: proc(n: ^blasint, k: ^blasint, D: ^f64, A: ^f64, lda: ^blasint, iseed: ^blasint, work: ^f64, info: ^Info) ---
	slagsy_ :: proc(n: ^blasint, k: ^blasint, D: ^f32, A: ^f32, lda: ^blasint, iseed: ^blasint, work: ^f32, info: ^Info) ---
	zlagsy_ :: proc(n: ^blasint, k: ^blasint, D: ^f64, A: ^complex128, lda: ^blasint, iseed: ^blasint, work: ^complex128, info: ^Info) ---
	// dlamch_ :: proc(cmach: cstring, _: c.size_t) -> f64 ---
	// slamch_ :: proc(cmach: cstring, _: c.size_t) -> lapack_float_return ---
	clangb_ :: proc(norm: cstring, n: ^blasint, kl: ^blasint, ku: ^blasint, AB: ^complex64, ldab: ^blasint, work: ^f32, _: c.size_t) -> lapack_float_return ---
	dlangb_ :: proc(norm: cstring, n: ^blasint, kl: ^blasint, ku: ^blasint, AB: ^f64, ldab: ^blasint, work: ^f64, _: c.size_t) -> f64 ---
	slangb_ :: proc(norm: cstring, n: ^blasint, kl: ^blasint, ku: ^blasint, AB: ^f32, ldab: ^blasint, work: ^f32, _: c.size_t) -> lapack_float_return ---
	zlangb_ :: proc(norm: cstring, n: ^blasint, kl: ^blasint, ku: ^blasint, AB: ^complex128, ldab: ^blasint, work: ^f64, _: c.size_t) -> f64 ---

	clange_ :: proc(norm: cstring, m: ^blasint, n: ^blasint, A: ^complex64, lda: ^blasint, work: ^f32, _: c.size_t) -> lapack_float_return ---
	dlange_ :: proc(norm: cstring, m: ^blasint, n: ^blasint, A: ^f64, lda: ^blasint, work: ^f64, _: c.size_t) -> f64 ---
	slange_ :: proc(norm: cstring, m: ^blasint, n: ^blasint, A: ^f32, lda: ^blasint, work: ^f32, _: c.size_t) -> lapack_float_return ---
	zlange_ :: proc(norm: cstring, m: ^blasint, n: ^blasint, A: ^complex128, lda: ^blasint, work: ^f64, _: c.size_t) -> f64 ---

	clangt_ :: proc(norm: cstring, n: ^blasint, DL: ^complex64, D: ^complex64, DU: ^complex64, _: c.size_t) -> lapack_float_return ---
	dlangt_ :: proc(norm: cstring, n: ^blasint, DL: ^f64, D: ^f64, DU: ^f64, _: c.size_t) -> f64 ---
	slangt_ :: proc(norm: cstring, n: ^blasint, DL: ^f32, D: ^f32, DU: ^f32, _: c.size_t) -> lapack_float_return ---
	zlangt_ :: proc(norm: cstring, n: ^blasint, DL: ^complex128, D: ^complex128, DU: ^complex128, _: c.size_t) -> f64 ---

	clanhb_ :: proc(norm: cstring, uplo: cstring, n: ^blasint, k: ^blasint, AB: ^complex64, ldab: ^blasint, work: ^f32, _: c.size_t, _: c.size_t) -> lapack_float_return ---
	zlanhb_ :: proc(norm: cstring, uplo: cstring, n: ^blasint, k: ^blasint, AB: ^complex128, ldab: ^blasint, work: ^f64, _: c.size_t, _: c.size_t) -> f64 ---
	clanhe_ :: proc(norm: cstring, uplo: cstring, n: ^blasint, A: ^complex64, lda: ^blasint, work: ^f32, _: c.size_t, _: c.size_t) -> lapack_float_return ---
	zlanhe_ :: proc(norm: cstring, uplo: cstring, n: ^blasint, A: ^complex128, lda: ^blasint, work: ^f64, _: c.size_t, _: c.size_t) -> f64 ---

	clanhp_ :: proc(norm: cstring, uplo: cstring, n: ^blasint, AP: ^complex64, work: ^f32, _: c.size_t, _: c.size_t) -> lapack_float_return ---
	zlanhp_ :: proc(norm: cstring, uplo: cstring, n: ^blasint, AP: ^complex128, work: ^f64, _: c.size_t, _: c.size_t) -> f64 ---
	clanhs_ :: proc(norm: cstring, n: ^blasint, A: ^complex64, lda: ^blasint, work: ^f32, _: c.size_t) -> lapack_float_return ---
	dlanhs_ :: proc(norm: cstring, n: ^blasint, A: ^f64, lda: ^blasint, work: ^f64, _: c.size_t) -> f64 ---
	slanhs_ :: proc(norm: cstring, n: ^blasint, A: ^f32, lda: ^blasint, work: ^f32, _: c.size_t) -> lapack_float_return ---
	zlanhs_ :: proc(norm: cstring, n: ^blasint, A: ^complex128, lda: ^blasint, work: ^f64, _: c.size_t) -> f64 ---

	clanht_ :: proc(norm: cstring, n: ^blasint, D: ^f32, E: ^complex64, _: c.size_t) -> lapack_float_return ---
	zlanht_ :: proc(norm: cstring, n: ^blasint, D: ^f64, E: ^complex128, _: c.size_t) -> f64 ---
	clansb_ :: proc(norm: cstring, uplo: cstring, n: ^blasint, k: ^blasint, AB: ^complex64, ldab: ^blasint, work: ^f32, _: c.size_t, _: c.size_t) -> lapack_float_return ---
	dlansb_ :: proc(norm: cstring, uplo: cstring, n: ^blasint, k: ^blasint, AB: ^f64, ldab: ^blasint, work: ^f64, _: c.size_t, _: c.size_t) -> f64 ---
	slansb_ :: proc(norm: cstring, uplo: cstring, n: ^blasint, k: ^blasint, AB: ^f32, ldab: ^blasint, work: ^f32, _: c.size_t, _: c.size_t) -> lapack_float_return ---
	zlansb_ :: proc(norm: cstring, uplo: cstring, n: ^blasint, k: ^blasint, AB: ^complex128, ldab: ^blasint, work: ^f64, _: c.size_t, _: c.size_t) -> f64 ---

	clansp_ :: proc(norm: cstring, uplo: cstring, n: ^blasint, AP: ^complex64, work: ^f32, _: c.size_t, _: c.size_t) -> lapack_float_return ---
	dlansp_ :: proc(norm: cstring, uplo: cstring, n: ^blasint, AP: ^f64, work: ^f64, _: c.size_t, _: c.size_t) -> f64 ---
	slansp_ :: proc(norm: cstring, uplo: cstring, n: ^blasint, AP: ^f32, work: ^f32, _: c.size_t, _: c.size_t) -> lapack_float_return ---
	zlansp_ :: proc(norm: cstring, uplo: cstring, n: ^blasint, AP: ^complex128, work: ^f64, _: c.size_t, _: c.size_t) -> f64 ---

	dlanst_ :: proc(norm: cstring, n: ^blasint, D: ^f64, E: ^f64, _: c.size_t) -> f64 ---
	slanst_ :: proc(norm: cstring, n: ^blasint, D: ^f32, E: ^f32, _: c.size_t) -> lapack_float_return ---
	clansy_ :: proc(norm: cstring, uplo: cstring, n: ^blasint, A: ^complex64, lda: ^blasint, work: ^f32, _: c.size_t, _: c.size_t) -> lapack_float_return ---
	dlansy_ :: proc(norm: cstring, uplo: cstring, n: ^blasint, A: ^f64, lda: ^blasint, work: ^f64, _: c.size_t, _: c.size_t) -> f64 ---
	slansy_ :: proc(norm: cstring, uplo: cstring, n: ^blasint, A: ^f32, lda: ^blasint, work: ^f32, _: c.size_t, _: c.size_t) -> lapack_float_return ---
	zlansy_ :: proc(norm: cstring, uplo: cstring, n: ^blasint, A: ^complex128, lda: ^blasint, work: ^f64, _: c.size_t, _: c.size_t) -> f64 ---

	clantb_ :: proc(norm: cstring, uplo: cstring, diag: cstring, n: ^blasint, k: ^blasint, AB: ^complex64, ldab: ^blasint, work: ^f32, _: c.size_t, _: c.size_t, _: c.size_t) -> lapack_float_return ---
	dlantb_ :: proc(norm: cstring, uplo: cstring, diag: cstring, n: ^blasint, k: ^blasint, AB: ^f64, ldab: ^blasint, work: ^f64, _: c.size_t, _: c.size_t, _: c.size_t) -> f64 ---
	slantb_ :: proc(norm: cstring, uplo: cstring, diag: cstring, n: ^blasint, k: ^blasint, AB: ^f32, ldab: ^blasint, work: ^f32, _: c.size_t, _: c.size_t, _: c.size_t) -> lapack_float_return ---
	zlantb_ :: proc(norm: cstring, uplo: cstring, diag: cstring, n: ^blasint, k: ^blasint, AB: ^complex128, ldab: ^blasint, work: ^f64, _: c.size_t, _: c.size_t, _: c.size_t) -> f64 ---

	clantp_ :: proc(norm: cstring, uplo: cstring, diag: cstring, n: ^blasint, AP: ^complex64, work: ^f32, _: c.size_t, _: c.size_t, _: c.size_t) -> lapack_float_return ---
	dlantp_ :: proc(norm: cstring, uplo: cstring, diag: cstring, n: ^blasint, AP: ^f64, work: ^f64, _: c.size_t, _: c.size_t, _: c.size_t) -> f64 ---
	slantp_ :: proc(norm: cstring, uplo: cstring, diag: cstring, n: ^blasint, AP: ^f32, work: ^f32, _: c.size_t, _: c.size_t, _: c.size_t) -> lapack_float_return ---
	zlantp_ :: proc(norm: cstring, uplo: cstring, diag: cstring, n: ^blasint, AP: ^complex128, work: ^f64, _: c.size_t, _: c.size_t, _: c.size_t) -> f64 ---

	clantr_ :: proc(norm: cstring, uplo: cstring, diag: cstring, m: ^blasint, n: ^blasint, A: ^complex64, lda: ^blasint, work: ^f32, _: c.size_t, _: c.size_t, _: c.size_t) -> lapack_float_return ---
	dlantr_ :: proc(norm: cstring, uplo: cstring, diag: cstring, m: ^blasint, n: ^blasint, A: ^f64, lda: ^blasint, work: ^f64, _: c.size_t, _: c.size_t, _: c.size_t) -> f64 ---
	slantr_ :: proc(norm: cstring, uplo: cstring, diag: cstring, m: ^blasint, n: ^blasint, A: ^f32, lda: ^blasint, work: ^f32, _: c.size_t, _: c.size_t, _: c.size_t) -> lapack_float_return ---
	zlantr_ :: proc(norm: cstring, uplo: cstring, diag: cstring, m: ^blasint, n: ^blasint, A: ^complex128, lda: ^blasint, work: ^f64, _: c.size_t, _: c.size_t, _: c.size_t) -> f64 ---

	clapmr_ :: proc(forwrd: ^blasint, m: ^blasint, n: ^blasint, X: ^complex64, ldx: ^blasint, K: ^blasint) ---
	dlapmr_ :: proc(forwrd: ^blasint, m: ^blasint, n: ^blasint, X: ^f64, ldx: ^blasint, K: ^blasint) ---
	slapmr_ :: proc(forwrd: ^blasint, m: ^blasint, n: ^blasint, X: ^f32, ldx: ^blasint, K: ^blasint) ---
	zlapmr_ :: proc(forwrd: ^blasint, m: ^blasint, n: ^blasint, X: ^complex128, ldx: ^blasint, K: ^blasint) ---

	clapmt_ :: proc(forwrd: ^blasint, m: ^blasint, n: ^blasint, X: ^complex64, ldx: ^blasint, K: ^blasint) ---
	dlapmt_ :: proc(forwrd: ^blasint, m: ^blasint, n: ^blasint, X: ^f64, ldx: ^blasint, K: ^blasint) ---
	slapmt_ :: proc(forwrd: ^blasint, m: ^blasint, n: ^blasint, X: ^f32, ldx: ^blasint, K: ^blasint) ---
	zlapmt_ :: proc(forwrd: ^blasint, m: ^blasint, n: ^blasint, X: ^complex128, ldx: ^blasint, K: ^blasint) ---

	dlapy2_ :: proc(x: ^f64, y: ^f64) -> f64 ---
	slapy2_ :: proc(x: ^f32, y: ^f32) -> lapack_float_return ---
	dlapy3_ :: proc(x: ^f64, y: ^f64, z: ^f64) -> f64 ---
	slapy3_ :: proc(x: ^f32, y: ^f32, z: ^f32) -> lapack_float_return ---
	clarcm_ :: proc(m: ^blasint, n: ^blasint, A: ^f32, lda: ^blasint, B: ^complex64, ldb: ^blasint, C: ^complex64, ldc: ^blasint, rwork: ^f32) ---
	zlarcm_ :: proc(m: ^blasint, n: ^blasint, A: ^f64, lda: ^blasint, B: ^complex128, ldb: ^blasint, C: ^complex128, ldc: ^blasint, rwork: ^f64) ---

	clarf_ :: proc(side: cstring, m: ^blasint, n: ^blasint, V: ^complex64, incv: ^blasint, tau: ^complex64, C: ^complex64, ldc: ^blasint, work: ^complex64, _: c.size_t) ---
	dlarf_ :: proc(side: cstring, m: ^blasint, n: ^blasint, V: ^f64, incv: ^blasint, tau: ^f64, C: ^f64, ldc: ^blasint, work: ^f64, _: c.size_t) ---
	slarf_ :: proc(side: cstring, m: ^blasint, n: ^blasint, V: ^f32, incv: ^blasint, tau: ^f32, C: ^f32, ldc: ^blasint, work: ^f32, _: c.size_t) ---
	zlarf_ :: proc(side: cstring, m: ^blasint, n: ^blasint, V: ^complex128, incv: ^blasint, tau: ^complex128, C: ^complex128, ldc: ^blasint, work: ^complex128, _: c.size_t) ---

	clarfb_ :: proc(side: cstring, trans: cstring, direct: cstring, storev: cstring, m: ^blasint, n: ^blasint, k: ^blasint, V: ^complex64, ldv: ^blasint, T: ^complex64, ldt: ^blasint, C: ^complex64, ldc: ^blasint, work: ^complex64, ldwork: ^blasint, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t) ---
	dlarfb_ :: proc(side: cstring, trans: cstring, direct: cstring, storev: cstring, m: ^blasint, n: ^blasint, k: ^blasint, V: ^f64, ldv: ^blasint, T: ^f64, ldt: ^blasint, C: ^f64, ldc: ^blasint, work: ^f64, ldwork: ^blasint, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t) ---
	slarfb_ :: proc(side: cstring, trans: cstring, direct: cstring, storev: cstring, m: ^blasint, n: ^blasint, k: ^blasint, V: ^f32, ldv: ^blasint, T: ^f32, ldt: ^blasint, C: ^f32, ldc: ^blasint, work: ^f32, ldwork: ^blasint, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t) ---
	zlarfb_ :: proc(side: cstring, trans: cstring, direct: cstring, storev: cstring, m: ^blasint, n: ^blasint, k: ^blasint, V: ^complex128, ldv: ^blasint, T: ^complex128, ldt: ^blasint, C: ^complex128, ldc: ^blasint, work: ^complex128, ldwork: ^blasint, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t) ---

	clarfg_ :: proc(n: ^blasint, alpha: ^complex64, X: ^complex64, incx: ^blasint, tau: ^complex64) ---
	dlarfg_ :: proc(n: ^blasint, alpha: ^f64, X: ^f64, incx: ^blasint, tau: ^f64) ---
	slarfg_ :: proc(n: ^blasint, alpha: ^f32, X: ^f32, incx: ^blasint, tau: ^f32) ---
	zlarfg_ :: proc(n: ^blasint, alpha: ^complex128, X: ^complex128, incx: ^blasint, tau: ^complex128) ---

	clarft_ :: proc(direct: cstring, storev: cstring, n: ^blasint, k: ^blasint, V: ^complex64, ldv: ^blasint, tau: ^complex64, T: ^complex64, ldt: ^blasint, _: c.size_t, _: c.size_t) ---
	dlarft_ :: proc(direct: cstring, storev: cstring, n: ^blasint, k: ^blasint, V: ^f64, ldv: ^blasint, tau: ^f64, T: ^f64, ldt: ^blasint, _: c.size_t, _: c.size_t) ---
	slarft_ :: proc(direct: cstring, storev: cstring, n: ^blasint, k: ^blasint, V: ^f32, ldv: ^blasint, tau: ^f32, T: ^f32, ldt: ^blasint, _: c.size_t, _: c.size_t) ---
	zlarft_ :: proc(direct: cstring, storev: cstring, n: ^blasint, k: ^blasint, V: ^complex128, ldv: ^blasint, tau: ^complex128, T: ^complex128, ldt: ^blasint, _: c.size_t, _: c.size_t) ---

	clarfx_ :: proc(side: cstring, m: ^blasint, n: ^blasint, V: ^complex64, tau: ^complex64, C: ^complex64, ldc: ^blasint, work: ^complex64, _: c.size_t) ---
	dlarfx_ :: proc(side: cstring, m: ^blasint, n: ^blasint, V: ^f64, tau: ^f64, C: ^f64, ldc: ^blasint, work: ^f64, _: c.size_t) ---
	slarfx_ :: proc(side: cstring, m: ^blasint, n: ^blasint, V: ^f32, tau: ^f32, C: ^f32, ldc: ^blasint, work: ^f32, _: c.size_t) ---
	zlarfx_ :: proc(side: cstring, m: ^blasint, n: ^blasint, V: ^complex128, tau: ^complex128, C: ^complex128, ldc: ^blasint, work: ^complex128, _: c.size_t) ---

	clarnv_ :: proc(idist: ^blasint, iseed: ^blasint, n: ^blasint, X: ^complex64) ---
	dlarnv_ :: proc(idist: ^blasint, iseed: ^blasint, n: ^blasint, X: ^f64) ---
	slarnv_ :: proc(idist: ^blasint, iseed: ^blasint, n: ^blasint, X: ^f32) ---
	zlarnv_ :: proc(idist: ^blasint, iseed: ^blasint, n: ^blasint, X: ^complex128) ---

	// DLAROR/SLAROR/CLAROR/ZLAROR - Pre/post-multiply matrix by random orthogonal matrix
	dlaror_ :: proc(side: cstring, init: cstring, m: ^blasint, n: ^blasint, A: ^f64, lda: ^blasint, iseed: ^blasint, X: ^f64, info: ^Info, _: c.size_t, _: c.size_t) ---
	slaror_ :: proc(side: cstring, init: cstring, m: ^blasint, n: ^blasint, A: ^f32, lda: ^blasint, iseed: ^blasint, X: ^f32, info: ^Info, _: c.size_t, _: c.size_t) ---
	claror_ :: proc(side: cstring, init: cstring, m: ^blasint, n: ^blasint, A: ^complex64, lda: ^blasint, iseed: ^blasint, X: ^complex64, info: ^Info, _: c.size_t, _: c.size_t) ---
	zlaror_ :: proc(side: cstring, init: cstring, m: ^blasint, n: ^blasint, A: ^complex128, lda: ^blasint, iseed: ^blasint, X: ^complex128, info: ^Info, _: c.size_t, _: c.size_t) ---

	// DLAROT/SLAROT/CLAROT/ZLAROT - Apply Givens rotation to two adjacent rows/columns
	dlarot_ :: proc(lrows: ^blasint, lleft: ^blasint, lright: ^blasint, nl: ^blasint, c: ^f64, s: ^f64, A: ^f64, lda: ^blasint, xleft: ^f64, xright: ^f64) ---
	slarot_ :: proc(lrows: ^blasint, lleft: ^blasint, lright: ^blasint, nl: ^blasint, c: ^f32, s: ^f32, A: ^f32, lda: ^blasint, xleft: ^f32, xright: ^f32) ---
	clarot_ :: proc(lrows: ^blasint, lleft: ^blasint, lright: ^blasint, nl: ^blasint, c: ^f32, s: ^complex64, A: ^complex64, lda: ^blasint, xleft: ^complex64, xright: ^complex64) ---
	zlarot_ :: proc(lrows: ^blasint, lleft: ^blasint, lright: ^blasint, nl: ^blasint, c: ^f64, s: ^complex128, A: ^complex128, lda: ^blasint, xleft: ^complex128, xright: ^complex128) ---

	// DLASR/SLASR/CLASR/ZLASR - Apply sequence of plane rotations
	dlasr_ :: proc(side: cstring, pivot: cstring, direct: cstring, m: ^blasint, n: ^blasint, c_: ^f64, s: ^f64, A: ^f64, lda: ^blasint, _: c.size_t, _: c.size_t, _: c.size_t) ---
	slasr_ :: proc(side: cstring, pivot: cstring, direct: cstring, m: ^blasint, n: ^blasint, c_: ^f32, s: ^f32, A: ^f32, lda: ^blasint, _: c.size_t, _: c.size_t, _: c.size_t) ---
	clasr_ :: proc(side: cstring, pivot: cstring, direct: cstring, m: ^blasint, n: ^blasint, c_: ^f32, s: ^f32, A: ^complex64, lda: ^blasint, _: c.size_t, _: c.size_t, _: c.size_t) ---
	zlasr_ :: proc(side: cstring, pivot: cstring, direct: cstring, m: ^blasint, n: ^blasint, c_: ^f64, s: ^f64, A: ^complex128, lda: ^blasint, _: c.size_t, _: c.size_t, _: c.size_t) ---

	dlartgp_ :: proc(f: ^f64, g: ^f64, cs: ^f64, sn: ^f64, r: ^f64) ---
	slartgp_ :: proc(f: ^f32, g: ^f32, cs: ^f32, sn: ^f32, r: ^f32) ---
	dlartgs_ :: proc(x: ^f64, y: ^f64, sigma: ^f64, cs: ^f64, sn: ^f64) ---
	slartgs_ :: proc(x: ^f32, y: ^f32, sigma: ^f32, cs: ^f32, sn: ^f32) ---
	clascl_ :: proc(type: cstring, kl: ^blasint, ku: ^blasint, cfrom: ^f32, cto: ^f32, m: ^blasint, n: ^blasint, A: ^complex64, lda: ^blasint, info: ^Info, _: c.size_t) ---
	dlascl_ :: proc(type: cstring, kl: ^blasint, ku: ^blasint, cfrom: ^f64, cto: ^f64, m: ^blasint, n: ^blasint, A: ^f64, lda: ^blasint, info: ^Info, _: c.size_t) ---
	slascl_ :: proc(type: cstring, kl: ^blasint, ku: ^blasint, cfrom: ^f32, cto: ^f32, m: ^blasint, n: ^blasint, A: ^f32, lda: ^blasint, info: ^Info, _: c.size_t) ---
	zlascl_ :: proc(type: cstring, kl: ^blasint, ku: ^blasint, cfrom: ^f64, cto: ^f64, m: ^blasint, n: ^blasint, A: ^complex128, lda: ^blasint, info: ^Info, _: c.size_t) ---

	claset_ :: proc(uplo: cstring, m: ^blasint, n: ^blasint, alpha: ^complex64, beta: ^complex64, A: ^complex64, lda: ^blasint, _: c.size_t) ---
	dlaset_ :: proc(uplo: cstring, m: ^blasint, n: ^blasint, alpha: ^f64, beta: ^f64, A: ^f64, lda: ^blasint, _: c.size_t) ---
	slaset_ :: proc(uplo: cstring, m: ^blasint, n: ^blasint, alpha: ^f32, beta: ^f32, A: ^f32, lda: ^blasint, _: c.size_t) ---
	zlaset_ :: proc(uplo: cstring, m: ^blasint, n: ^blasint, alpha: ^complex128, beta: ^complex128, A: ^complex128, lda: ^blasint, _: c.size_t) ---

	dlasrt_ :: proc(id: cstring, n: ^blasint, D: ^f64, info: ^Info, _: c.size_t) ---
	slasrt_ :: proc(id: cstring, n: ^blasint, D: ^f32, info: ^Info, _: c.size_t) ---
	classq_ :: proc(n: ^blasint, X: ^complex64, incx: ^blasint, scale: ^f32, sumsq: ^f32) ---
	dlassq_ :: proc(n: ^blasint, X: ^f64, incx: ^blasint, scale: ^f64, sumsq: ^f64) ---
	slassq_ :: proc(n: ^blasint, X: ^f32, incx: ^blasint, scale: ^f32, sumsq: ^f32) ---
	zlassq_ :: proc(n: ^blasint, X: ^complex128, incx: ^blasint, scale: ^f64, sumsq: ^f64) ---
	// claswp_ :: proc(n: ^blasint, A: ^complex64, lda: ^blasint, k1: ^blasint, k2: ^blasint, ipiv: ^blasint, incx: ^blasint) -> i32 ---
	// dlaswp_ :: proc(n: ^blasint, A: ^f64, lda: ^blasint, k1: ^blasint, k2: ^blasint, ipiv: ^blasint, incx: ^blasint) -> i32 ---
	// slaswp_ :: proc(n: ^blasint, A: ^f32, lda: ^blasint, k1: ^blasint, k2: ^blasint, ipiv: ^blasint, incx: ^blasint) -> i32 ---
	// zlaswp_ :: proc(n: ^blasint, A: ^complex128, lda: ^blasint, k1: ^blasint, k2: ^blasint, ipiv: ^blasint, incx: ^blasint) -> i32 ---
	// DLATMS/SLATMS/CLATMS/ZLATMS - Generate test matrices with specified singular values
	clatms_ :: proc(m: ^blasint, n: ^blasint, dist: cstring, iseed: ^blasint, sym: cstring, D: ^f32, mode: ^blasint, cond: ^f32, dmax: ^f32, kl: ^blasint, ku: ^blasint, pack: cstring, A: ^complex64, lda: ^blasint, work: ^complex64, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	dlatms_ :: proc(m: ^blasint, n: ^blasint, dist: cstring, iseed: ^blasint, sym: cstring, D: ^f64, mode: ^blasint, cond: ^f64, dmax: ^f64, kl: ^blasint, ku: ^blasint, pack: cstring, A: ^f64, lda: ^blasint, work: ^f64, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	slatms_ :: proc(m: ^blasint, n: ^blasint, dist: cstring, iseed: ^blasint, sym: cstring, D: ^f32, mode: ^blasint, cond: ^f32, dmax: ^f32, kl: ^blasint, ku: ^blasint, pack: cstring, A: ^f32, lda: ^blasint, work: ^f32, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	zlatms_ :: proc(m: ^blasint, n: ^blasint, dist: cstring, iseed: ^blasint, sym: cstring, D: ^f64, mode: ^blasint, cond: ^f64, dmax: ^f64, kl: ^blasint, ku: ^blasint, pack: cstring, A: ^complex128, lda: ^blasint, work: ^complex128, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---

	// DLATMT/SLATMT/CLATMT/ZLATMT - Generate test matrices with specified eigenvalues
	dlatmt_ :: proc(m: ^blasint, n: ^blasint, dist: cstring, iseed: ^blasint, sym: cstring, D: ^f64, mode: ^blasint, cond: ^f64, dmax: ^f64, rank: ^blasint, kl: ^blasint, ku: ^blasint, pack: cstring, A: ^f64, lda: ^blasint, work: ^f64, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	slatmt_ :: proc(m: ^blasint, n: ^blasint, dist: cstring, iseed: ^blasint, sym: cstring, D: ^f32, mode: ^blasint, cond: ^f32, dmax: ^f32, rank: ^blasint, kl: ^blasint, ku: ^blasint, pack: cstring, A: ^f32, lda: ^blasint, work: ^f32, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	clatmt_ :: proc(m: ^blasint, n: ^blasint, dist: cstring, iseed: ^blasint, sym: cstring, D: ^f32, mode: ^blasint, cond: ^f32, dmax: ^f32, rank: ^blasint, kl: ^blasint, ku: ^blasint, pack: cstring, A: ^complex64, lda: ^blasint, work: ^complex64, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	zlatmt_ :: proc(m: ^blasint, n: ^blasint, dist: cstring, iseed: ^blasint, sym: cstring, D: ^f64, mode: ^blasint, cond: ^f64, dmax: ^f64, rank: ^blasint, kl: ^blasint, ku: ^blasint, pack: cstring, A: ^complex128, lda: ^blasint, work: ^complex128, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	// clauum_ :: proc(uplo: cstring, n: ^blasint, A: ^complex64, lda: ^blasint, info: ^Info, _: c.size_t) -> i32 ---
	// dlauum_ :: proc(uplo: cstring, n: ^blasint, A: ^f64, lda: ^blasint, info: ^Info, _: c.size_t) -> i32 ---
	// slauum_ :: proc(uplo: cstring, n: ^blasint, A: ^f32, lda: ^blasint, info: ^Info, _: c.size_t) -> i32 ---
	// zlauum_ :: proc(uplo: cstring, n: ^blasint, A: ^complex128, lda: ^blasint, info: ^Info, _: c.size_t) -> i32 ---
	ilaver_ :: proc(vers_major: ^blasint, vers_minor: ^blasint, vers_patch: ^blasint) -> i32 ---
	dopgtr_ :: proc(uplo: cstring, n: ^blasint, AP: ^f64, tau: ^f64, Q: ^f64, ldq: ^blasint, work: ^f64, info: ^Info, _: c.size_t) ---
	sopgtr_ :: proc(uplo: cstring, n: ^blasint, AP: ^f32, tau: ^f32, Q: ^f32, ldq: ^blasint, work: ^f32, info: ^Info, _: c.size_t) ---
	dopmtr_ :: proc(side: cstring, uplo: cstring, trans: cstring, m: ^blasint, n: ^blasint, AP: ^f64, tau: ^f64, C: ^f64, ldc: ^blasint, work: ^f64, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	sopmtr_ :: proc(side: cstring, uplo: cstring, trans: cstring, m: ^blasint, n: ^blasint, AP: ^f32, tau: ^f32, C: ^f32, ldc: ^blasint, work: ^f32, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	dorbdb_ :: proc(trans: cstring, signs: cstring, m: ^blasint, p: ^blasint, q: ^blasint, X11: ^f64, ldx11: ^blasint, X12: ^f64, ldx12: ^blasint, X21: ^f64, ldx21: ^blasint, X22: ^f64, ldx22: ^blasint, theta: ^f64, phi: ^f64, TAUP1: ^f64, TAUP2: ^f64, TAUQ1: ^f64, TAUQ2: ^f64, work: ^f64, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	sorbdb_ :: proc(trans: cstring, signs: cstring, m: ^blasint, p: ^blasint, q: ^blasint, X11: ^f32, ldx11: ^blasint, X12: ^f32, ldx12: ^blasint, X21: ^f32, ldx21: ^blasint, X22: ^f32, ldx22: ^blasint, theta: ^f32, phi: ^f32, TAUP1: ^f32, TAUP2: ^f32, TAUQ1: ^f32, TAUQ2: ^f32, work: ^f32, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	dorcsd_ :: proc(jobu1: cstring, jobu2: cstring, jobv1t: cstring, jobv2t: cstring, trans: cstring, signs: cstring, m: ^blasint, p: ^blasint, q: ^blasint, X11: ^f64, ldx11: ^blasint, X12: ^f64, ldx12: ^blasint, X21: ^f64, ldx21: ^blasint, X22: ^f64, ldx22: ^blasint, theta: ^f64, U1: ^f64, ldu1: ^blasint, U2: ^f64, ldu2: ^blasint, V1T: ^f64, ldv1t: ^blasint, V2T: ^f64, ldv2t: ^blasint, work: ^f64, lwork: ^blasint, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t) ---
	sorcsd_ :: proc(jobu1: cstring, jobu2: cstring, jobv1t: cstring, jobv2t: cstring, trans: cstring, signs: cstring, m: ^blasint, p: ^blasint, q: ^blasint, X11: ^f32, ldx11: ^blasint, X12: ^f32, ldx12: ^blasint, X21: ^f32, ldx21: ^blasint, X22: ^f32, ldx22: ^blasint, theta: ^f32, U1: ^f32, ldu1: ^blasint, U2: ^f32, ldu2: ^blasint, V1T: ^f32, ldv1t: ^blasint, V2T: ^f32, ldv2t: ^blasint, work: ^f32, lwork: ^blasint, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t) ---
	dorcsd2by1_ :: proc(jobu1: cstring, jobu2: cstring, jobv1t: cstring, m: ^blasint, p: ^blasint, q: ^blasint, X11: ^f64, ldx11: ^blasint, X21: ^f64, ldx21: ^blasint, theta: ^f64, U1: ^f64, ldu1: ^blasint, U2: ^f64, ldu2: ^blasint, V1T: ^f64, ldv1t: ^blasint, work: ^f64, lwork: ^blasint, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	sorcsd2by1_ :: proc(jobu1: cstring, jobu2: cstring, jobv1t: cstring, m: ^blasint, p: ^blasint, q: ^blasint, X11: ^f32, ldx11: ^blasint, X21: ^f32, ldx21: ^blasint, theta: ^f32, U1: ^f32, ldu1: ^blasint, U2: ^f32, ldu2: ^blasint, V1T: ^f32, ldv1t: ^blasint, work: ^f32, lwork: ^blasint, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---

	dorgbr_ :: proc(vect: cstring, m: ^blasint, n: ^blasint, k: ^blasint, A: ^f64, lda: ^blasint, tau: ^f64, work: ^f64, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	sorgbr_ :: proc(vect: cstring, m: ^blasint, n: ^blasint, k: ^blasint, A: ^f32, lda: ^blasint, tau: ^f32, work: ^f32, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	dorghr_ :: proc(n: ^blasint, ilo: ^blasint, ihi: ^blasint, A: ^f64, lda: ^blasint, tau: ^f64, work: ^f64, lwork: ^blasint, info: ^Info) ---
	sorghr_ :: proc(n: ^blasint, ilo: ^blasint, ihi: ^blasint, A: ^f32, lda: ^blasint, tau: ^f32, work: ^f32, lwork: ^blasint, info: ^Info) ---

	dorglq_ :: proc(m: ^blasint, n: ^blasint, k: ^blasint, A: ^f64, lda: ^blasint, tau: ^f64, work: ^f64, lwork: ^blasint, info: ^Info) ---
	sorglq_ :: proc(m: ^blasint, n: ^blasint, k: ^blasint, A: ^f32, lda: ^blasint, tau: ^f32, work: ^f32, lwork: ^blasint, info: ^Info) ---
	dorgql_ :: proc(m: ^blasint, n: ^blasint, k: ^blasint, A: ^f64, lda: ^blasint, tau: ^f64, work: ^f64, lwork: ^blasint, info: ^Info) ---
	sorgql_ :: proc(m: ^blasint, n: ^blasint, k: ^blasint, A: ^f32, lda: ^blasint, tau: ^f32, work: ^f32, lwork: ^blasint, info: ^Info) ---
	dorgqr_ :: proc(m: ^blasint, n: ^blasint, k: ^blasint, A: ^f64, lda: ^blasint, tau: ^f64, work: ^f64, lwork: ^blasint, info: ^Info) ---
	sorgqr_ :: proc(m: ^blasint, n: ^blasint, k: ^blasint, A: ^f32, lda: ^blasint, tau: ^f32, work: ^f32, lwork: ^blasint, info: ^Info) ---

	dorgrq_ :: proc(m: ^blasint, n: ^blasint, k: ^blasint, A: ^f64, lda: ^blasint, tau: ^f64, work: ^f64, lwork: ^blasint, info: ^Info) ---
	sorgrq_ :: proc(m: ^blasint, n: ^blasint, k: ^blasint, A: ^f32, lda: ^blasint, tau: ^f32, work: ^f32, lwork: ^blasint, info: ^Info) ---
	dorgtr_ :: proc(uplo: cstring, n: ^blasint, A: ^f64, lda: ^blasint, tau: ^f64, work: ^f64, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	sorgtr_ :: proc(uplo: cstring, n: ^blasint, A: ^f32, lda: ^blasint, tau: ^f32, work: ^f32, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	dorgtsqr_row_ :: proc(m: ^blasint, n: ^blasint, mb: ^blasint, nb: ^blasint, A: ^f64, lda: ^blasint, T: ^f64, ldt: ^blasint, work: ^f64, lwork: ^blasint, info: ^Info) ---
	sorgtsqr_row_ :: proc(m: ^blasint, n: ^blasint, mb: ^blasint, nb: ^blasint, A: ^f32, lda: ^blasint, T: ^f32, ldt: ^blasint, work: ^f32, lwork: ^blasint, info: ^Info) ---
	dorhr_col_ :: proc(m: ^blasint, n: ^blasint, nb: ^blasint, A: ^f64, lda: ^blasint, T: ^f64, ldt: ^blasint, D: ^f64, info: ^Info) ---
	sorhr_col_ :: proc(m: ^blasint, n: ^blasint, nb: ^blasint, A: ^f32, lda: ^blasint, T: ^f32, ldt: ^blasint, D: ^f32, info: ^Info) ---

	dormbr_ :: proc(vect: cstring, side: cstring, trans: cstring, m: ^blasint, n: ^blasint, k: ^blasint, A: ^f64, lda: ^blasint, tau: ^f64, C: ^f64, ldc: ^blasint, work: ^f64, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	sormbr_ :: proc(vect: cstring, side: cstring, trans: cstring, m: ^blasint, n: ^blasint, k: ^blasint, A: ^f32, lda: ^blasint, tau: ^f32, C: ^f32, ldc: ^blasint, work: ^f32, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	dormhr_ :: proc(side: cstring, trans: cstring, m: ^blasint, n: ^blasint, ilo: ^blasint, ihi: ^blasint, A: ^f64, lda: ^blasint, tau: ^f64, C: ^f64, ldc: ^blasint, work: ^f64, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	sormhr_ :: proc(side: cstring, trans: cstring, m: ^blasint, n: ^blasint, ilo: ^blasint, ihi: ^blasint, A: ^f32, lda: ^blasint, tau: ^f32, C: ^f32, ldc: ^blasint, work: ^f32, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	dormlq_ :: proc(side: cstring, trans: cstring, m: ^blasint, n: ^blasint, k: ^blasint, A: ^f64, lda: ^blasint, tau: ^f64, C: ^f64, ldc: ^blasint, work: ^f64, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	sormlq_ :: proc(side: cstring, trans: cstring, m: ^blasint, n: ^blasint, k: ^blasint, A: ^f32, lda: ^blasint, tau: ^f32, C: ^f32, ldc: ^blasint, work: ^f32, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	dormql_ :: proc(side: cstring, trans: cstring, m: ^blasint, n: ^blasint, k: ^blasint, A: ^f64, lda: ^blasint, tau: ^f64, C: ^f64, ldc: ^blasint, work: ^f64, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	sormql_ :: proc(side: cstring, trans: cstring, m: ^blasint, n: ^blasint, k: ^blasint, A: ^f32, lda: ^blasint, tau: ^f32, C: ^f32, ldc: ^blasint, work: ^f32, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	dormqr_ :: proc(side: cstring, trans: cstring, m: ^blasint, n: ^blasint, k: ^blasint, A: ^f64, lda: ^blasint, tau: ^f64, C: ^f64, ldc: ^blasint, work: ^f64, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	sormqr_ :: proc(side: cstring, trans: cstring, m: ^blasint, n: ^blasint, k: ^blasint, A: ^f32, lda: ^blasint, tau: ^f32, C: ^f32, ldc: ^blasint, work: ^f32, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---

	dormrq_ :: proc(side: cstring, trans: cstring, m: ^blasint, n: ^blasint, k: ^blasint, A: ^f64, lda: ^blasint, tau: ^f64, C: ^f64, ldc: ^blasint, work: ^f64, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	sormrq_ :: proc(side: cstring, trans: cstring, m: ^blasint, n: ^blasint, k: ^blasint, A: ^f32, lda: ^blasint, tau: ^f32, C: ^f32, ldc: ^blasint, work: ^f32, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	dormrz_ :: proc(side: cstring, trans: cstring, m: ^blasint, n: ^blasint, k: ^blasint, l: ^blasint, A: ^f64, lda: ^blasint, tau: ^f64, C: ^f64, ldc: ^blasint, work: ^f64, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	sormrz_ :: proc(side: cstring, trans: cstring, m: ^blasint, n: ^blasint, k: ^blasint, l: ^blasint, A: ^f32, lda: ^blasint, tau: ^f32, C: ^f32, ldc: ^blasint, work: ^f32, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	dormtr_ :: proc(side: cstring, uplo: cstring, trans: cstring, m: ^blasint, n: ^blasint, A: ^f64, lda: ^blasint, tau: ^f64, C: ^f64, ldc: ^blasint, work: ^f64, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	sormtr_ :: proc(side: cstring, uplo: cstring, trans: cstring, m: ^blasint, n: ^blasint, A: ^f32, lda: ^blasint, tau: ^f32, C: ^f32, ldc: ^blasint, work: ^f32, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	cpbcon_ :: proc(uplo: cstring, n: ^blasint, kd: ^blasint, AB: ^complex64, ldab: ^blasint, anorm: ^f32, rcond: ^f32, work: ^complex64, rwork: ^f32, info: ^Info, _: c.size_t) ---
	dpbcon_ :: proc(uplo: cstring, n: ^blasint, kd: ^blasint, AB: ^f64, ldab: ^blasint, anorm: ^f64, rcond: ^f64, work: ^f64, iwork: ^blasint, info: ^Info, _: c.size_t) ---
	spbcon_ :: proc(uplo: cstring, n: ^blasint, kd: ^blasint, AB: ^f32, ldab: ^blasint, anorm: ^f32, rcond: ^f32, work: ^f32, iwork: ^blasint, info: ^Info, _: c.size_t) ---
	zpbcon_ :: proc(uplo: cstring, n: ^blasint, kd: ^blasint, AB: ^complex128, ldab: ^blasint, anorm: ^f64, rcond: ^f64, work: ^complex128, rwork: ^f64, info: ^Info, _: c.size_t) ---

	cpbequ_ :: proc(uplo: cstring, n: ^blasint, kd: ^blasint, AB: ^complex64, ldab: ^blasint, S: ^f32, scond: ^f32, amax: ^f32, info: ^Info, _: c.size_t) ---
	dpbequ_ :: proc(uplo: cstring, n: ^blasint, kd: ^blasint, AB: ^f64, ldab: ^blasint, S: ^f64, scond: ^f64, amax: ^f64, info: ^Info, _: c.size_t) ---
	spbequ_ :: proc(uplo: cstring, n: ^blasint, kd: ^blasint, AB: ^f32, ldab: ^blasint, S: ^f32, scond: ^f32, amax: ^f32, info: ^Info, _: c.size_t) ---
	zpbequ_ :: proc(uplo: cstring, n: ^blasint, kd: ^blasint, AB: ^complex128, ldab: ^blasint, S: ^f64, scond: ^f64, amax: ^f64, info: ^Info, _: c.size_t) ---

	cpbrfs_ :: proc(uplo: cstring, n: ^blasint, kd: ^blasint, nrhs: ^blasint, AB: ^complex64, ldab: ^blasint, AFB: ^complex64, ldafb: ^blasint, B: ^complex64, ldb: ^blasint, X: ^complex64, ldx: ^blasint, ferr: ^f32, berr: ^f32, work: ^complex64, rwork: ^f32, info: ^Info, _: c.size_t) ---
	dpbrfs_ :: proc(uplo: cstring, n: ^blasint, kd: ^blasint, nrhs: ^blasint, AB: ^f64, ldab: ^blasint, AFB: ^f64, ldafb: ^blasint, B: ^f64, ldb: ^blasint, X: ^f64, ldx: ^blasint, ferr: ^f64, berr: ^f64, work: ^f64, iwork: ^blasint, info: ^Info, _: c.size_t) ---
	spbrfs_ :: proc(uplo: cstring, n: ^blasint, kd: ^blasint, nrhs: ^blasint, AB: ^f32, ldab: ^blasint, AFB: ^f32, ldafb: ^blasint, B: ^f32, ldb: ^blasint, X: ^f32, ldx: ^blasint, ferr: ^f32, berr: ^f32, work: ^f32, iwork: ^blasint, info: ^Info, _: c.size_t) ---
	zpbrfs_ :: proc(uplo: cstring, n: ^blasint, kd: ^blasint, nrhs: ^blasint, AB: ^complex128, ldab: ^blasint, AFB: ^complex128, ldafb: ^blasint, B: ^complex128, ldb: ^blasint, X: ^complex128, ldx: ^blasint, ferr: ^f64, berr: ^f64, work: ^complex128, rwork: ^f64, info: ^Info, _: c.size_t) ---
	cpbstf_ :: proc(uplo: cstring, n: ^blasint, kd: ^blasint, AB: ^complex64, ldab: ^blasint, info: ^Info, _: c.size_t) ---
	dpbstf_ :: proc(uplo: cstring, n: ^blasint, kd: ^blasint, AB: ^f64, ldab: ^blasint, info: ^Info, _: c.size_t) ---
	spbstf_ :: proc(uplo: cstring, n: ^blasint, kd: ^blasint, AB: ^f32, ldab: ^blasint, info: ^Info, _: c.size_t) ---
	zpbstf_ :: proc(uplo: cstring, n: ^blasint, kd: ^blasint, AB: ^complex128, ldab: ^blasint, info: ^Info, _: c.size_t) ---

	cpbsv_ :: proc(uplo: cstring, n: ^blasint, kd: ^blasint, nrhs: ^blasint, AB: ^complex64, ldab: ^blasint, B: ^complex64, ldb: ^blasint, info: ^Info, _: c.size_t) ---
	dpbsv_ :: proc(uplo: cstring, n: ^blasint, kd: ^blasint, nrhs: ^blasint, AB: ^f64, ldab: ^blasint, B: ^f64, ldb: ^blasint, info: ^Info, _: c.size_t) ---
	spbsv_ :: proc(uplo: cstring, n: ^blasint, kd: ^blasint, nrhs: ^blasint, AB: ^f32, ldab: ^blasint, B: ^f32, ldb: ^blasint, info: ^Info, _: c.size_t) ---
	zpbsv_ :: proc(uplo: cstring, n: ^blasint, kd: ^blasint, nrhs: ^blasint, AB: ^complex128, ldab: ^blasint, B: ^complex128, ldb: ^blasint, info: ^Info, _: c.size_t) ---
	cpbsvx_ :: proc(fact: cstring, uplo: cstring, n: ^blasint, kd: ^blasint, nrhs: ^blasint, AB: ^complex64, ldab: ^blasint, AFB: ^complex64, ldafb: ^blasint, equed: cstring, S: ^f32, B: ^complex64, ldb: ^blasint, X: ^complex64, ldx: ^blasint, rcond: ^f32, ferr: ^f32, berr: ^f32, work: ^complex64, rwork: ^f32, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	dpbsvx_ :: proc(fact: cstring, uplo: cstring, n: ^blasint, kd: ^blasint, nrhs: ^blasint, AB: ^f64, ldab: ^blasint, AFB: ^f64, ldafb: ^blasint, equed: cstring, S: ^f64, B: ^f64, ldb: ^blasint, X: ^f64, ldx: ^blasint, rcond: ^f64, ferr: ^f64, berr: ^f64, work: ^f64, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	spbsvx_ :: proc(fact: cstring, uplo: cstring, n: ^blasint, kd: ^blasint, nrhs: ^blasint, AB: ^f32, ldab: ^blasint, AFB: ^f32, ldafb: ^blasint, equed: cstring, S: ^f32, B: ^f32, ldb: ^blasint, X: ^f32, ldx: ^blasint, rcond: ^f32, ferr: ^f32, berr: ^f32, work: ^f32, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	zpbsvx_ :: proc(fact: cstring, uplo: cstring, n: ^blasint, kd: ^blasint, nrhs: ^blasint, AB: ^complex128, ldab: ^blasint, AFB: ^complex128, ldafb: ^blasint, equed: cstring, S: ^f64, B: ^complex128, ldb: ^blasint, X: ^complex128, ldx: ^blasint, rcond: ^f64, ferr: ^f64, berr: ^f64, work: ^complex128, rwork: ^f64, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---

	cpbtrf_ :: proc(uplo: cstring, n: ^blasint, kd: ^blasint, AB: ^complex64, ldab: ^blasint, info: ^Info, _: c.size_t) ---
	dpbtrf_ :: proc(uplo: cstring, n: ^blasint, kd: ^blasint, AB: ^f64, ldab: ^blasint, info: ^Info, _: c.size_t) ---
	spbtrf_ :: proc(uplo: cstring, n: ^blasint, kd: ^blasint, AB: ^f32, ldab: ^blasint, info: ^Info, _: c.size_t) ---
	zpbtrf_ :: proc(uplo: cstring, n: ^blasint, kd: ^blasint, AB: ^complex128, ldab: ^blasint, info: ^Info, _: c.size_t) ---
	cpbtrs_ :: proc(uplo: cstring, n: ^blasint, kd: ^blasint, nrhs: ^blasint, AB: ^complex64, ldab: ^blasint, B: ^complex64, ldb: ^blasint, info: ^Info, _: c.size_t) ---
	dpbtrs_ :: proc(uplo: cstring, n: ^blasint, kd: ^blasint, nrhs: ^blasint, AB: ^f64, ldab: ^blasint, B: ^f64, ldb: ^blasint, info: ^Info, _: c.size_t) ---
	spbtrs_ :: proc(uplo: cstring, n: ^blasint, kd: ^blasint, nrhs: ^blasint, AB: ^f32, ldab: ^blasint, B: ^f32, ldb: ^blasint, info: ^Info, _: c.size_t) ---
	zpbtrs_ :: proc(uplo: cstring, n: ^blasint, kd: ^blasint, nrhs: ^blasint, AB: ^complex128, ldab: ^blasint, B: ^complex128, ldb: ^blasint, info: ^Info, _: c.size_t) ---

	cpftrf_ :: proc(transr: cstring, uplo: cstring, n: ^blasint, A: ^complex64, info: ^Info, _: c.size_t, _: c.size_t) ---
	dpftrf_ :: proc(transr: cstring, uplo: cstring, n: ^blasint, A: ^f64, info: ^Info, _: c.size_t, _: c.size_t) ---
	spftrf_ :: proc(transr: cstring, uplo: cstring, n: ^blasint, A: ^f32, info: ^Info, _: c.size_t, _: c.size_t) ---
	zpftrf_ :: proc(transr: cstring, uplo: cstring, n: ^blasint, A: ^complex128, info: ^Info, _: c.size_t, _: c.size_t) ---

	cpftri_ :: proc(transr: cstring, uplo: cstring, n: ^blasint, A: ^complex64, info: ^Info, _: c.size_t, _: c.size_t) ---
	dpftri_ :: proc(transr: cstring, uplo: cstring, n: ^blasint, A: ^f64, info: ^Info, _: c.size_t, _: c.size_t) ---
	spftri_ :: proc(transr: cstring, uplo: cstring, n: ^blasint, A: ^f32, info: ^Info, _: c.size_t, _: c.size_t) ---
	zpftri_ :: proc(transr: cstring, uplo: cstring, n: ^blasint, A: ^complex128, info: ^Info, _: c.size_t, _: c.size_t) ---

	cpftrs_ :: proc(transr: cstring, uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex64, B: ^complex64, ldb: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	dpftrs_ :: proc(transr: cstring, uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^f64, B: ^f64, ldb: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	spftrs_ :: proc(transr: cstring, uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^f32, B: ^f32, ldb: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	zpftrs_ :: proc(transr: cstring, uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex128, B: ^complex128, ldb: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---

	cpocon_ :: proc(uplo: cstring, n: ^blasint, A: ^complex64, lda: ^blasint, anorm: ^f32, rcond: ^f32, work: ^complex64, rwork: ^f32, info: ^Info, _: c.size_t) ---
	dpocon_ :: proc(uplo: cstring, n: ^blasint, A: ^f64, lda: ^blasint, anorm: ^f64, rcond: ^f64, work: ^f64, iwork: ^blasint, info: ^Info, _: c.size_t) ---
	spocon_ :: proc(uplo: cstring, n: ^blasint, A: ^f32, lda: ^blasint, anorm: ^f32, rcond: ^f32, work: ^f32, iwork: ^blasint, info: ^Info, _: c.size_t) ---
	zpocon_ :: proc(uplo: cstring, n: ^blasint, A: ^complex128, lda: ^blasint, anorm: ^f64, rcond: ^f64, work: ^complex128, rwork: ^f64, info: ^Info, _: c.size_t) ---

	cpoequ_ :: proc(n: ^blasint, A: ^complex64, lda: ^blasint, S: ^f32, scond: ^f32, amax: ^f32, info: ^Info) ---
	dpoequ_ :: proc(n: ^blasint, A: ^f64, lda: ^blasint, S: ^f64, scond: ^f64, amax: ^f64, info: ^Info) ---
	spoequ_ :: proc(n: ^blasint, A: ^f32, lda: ^blasint, S: ^f32, scond: ^f32, amax: ^f32, info: ^Info) ---
	zpoequ_ :: proc(n: ^blasint, A: ^complex128, lda: ^blasint, S: ^f64, scond: ^f64, amax: ^f64, info: ^Info) ---

	cpoequb_ :: proc(n: ^blasint, A: ^complex64, lda: ^blasint, S: ^f32, scond: ^f32, amax: ^f32, info: ^Info) ---
	dpoequb_ :: proc(n: ^blasint, A: ^f64, lda: ^blasint, S: ^f64, scond: ^f64, amax: ^f64, info: ^Info) ---
	spoequb_ :: proc(n: ^blasint, A: ^f32, lda: ^blasint, S: ^f32, scond: ^f32, amax: ^f32, info: ^Info) ---
	zpoequb_ :: proc(n: ^blasint, A: ^complex128, lda: ^blasint, S: ^f64, scond: ^f64, amax: ^f64, info: ^Info) ---

	cporfs_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex64, lda: ^blasint, AF: ^complex64, ldaf: ^blasint, B: ^complex64, ldb: ^blasint, X: ^complex64, ldx: ^blasint, ferr: ^f32, berr: ^f32, work: ^complex64, rwork: ^f32, info: ^Info, _: c.size_t) ---
	dporfs_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^f64, lda: ^blasint, AF: ^f64, ldaf: ^blasint, B: ^f64, ldb: ^blasint, X: ^f64, ldx: ^blasint, ferr: ^f64, berr: ^f64, work: ^f64, iwork: ^blasint, info: ^Info, _: c.size_t) ---
	sporfs_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^f32, lda: ^blasint, AF: ^f32, ldaf: ^blasint, B: ^f32, ldb: ^blasint, X: ^f32, ldx: ^blasint, ferr: ^f32, berr: ^f32, work: ^f32, iwork: ^blasint, info: ^Info, _: c.size_t) ---
	zporfs_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex128, lda: ^blasint, AF: ^complex128, ldaf: ^blasint, B: ^complex128, ldb: ^blasint, X: ^complex128, ldx: ^blasint, ferr: ^f64, berr: ^f64, work: ^complex128, rwork: ^f64, info: ^Info, _: c.size_t) ---

	cporfsx_ :: proc(uplo: cstring, equed: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex64, lda: ^blasint, AF: ^complex64, ldaf: ^blasint, S: ^f32, B: ^complex64, ldb: ^blasint, X: ^complex64, ldx: ^blasint, rcond: ^f32, berr: ^f32, n_err_bnds: ^blasint, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: ^blasint, params: ^f32, work: ^complex64, rwork: ^f32, info: ^Info, _: c.size_t, _: c.size_t) ---
	dporfsx_ :: proc(uplo: cstring, equed: cstring, n: ^blasint, nrhs: ^blasint, A: ^f64, lda: ^blasint, AF: ^f64, ldaf: ^blasint, S: ^f64, B: ^f64, ldb: ^blasint, X: ^f64, ldx: ^blasint, rcond: ^f64, berr: ^f64, n_err_bnds: ^blasint, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: ^blasint, params: ^f64, work: ^f64, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	sporfsx_ :: proc(uplo: cstring, equed: cstring, n: ^blasint, nrhs: ^blasint, A: ^f32, lda: ^blasint, AF: ^f32, ldaf: ^blasint, S: ^f32, B: ^f32, ldb: ^blasint, X: ^f32, ldx: ^blasint, rcond: ^f32, berr: ^f32, n_err_bnds: ^blasint, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: ^blasint, params: ^f32, work: ^f32, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	zporfsx_ :: proc(uplo: cstring, equed: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex128, lda: ^blasint, AF: ^complex128, ldaf: ^blasint, S: ^f64, B: ^complex128, ldb: ^blasint, X: ^complex128, ldx: ^blasint, rcond: ^f64, berr: ^f64, n_err_bnds: ^blasint, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: ^blasint, params: ^f64, work: ^complex128, rwork: ^f64, info: ^Info, _: c.size_t, _: c.size_t) ---

	cposv_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex64, lda: ^blasint, B: ^complex64, ldb: ^blasint, info: ^Info, _: c.size_t) ---
	dposv_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^f64, lda: ^blasint, B: ^f64, ldb: ^blasint, info: ^Info, _: c.size_t) ---
	sposv_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^f32, lda: ^blasint, B: ^f32, ldb: ^blasint, info: ^Info, _: c.size_t) ---
	zposv_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex128, lda: ^blasint, B: ^complex128, ldb: ^blasint, info: ^Info, _: c.size_t) ---

	dsposv_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^f64, lda: ^blasint, B: ^f64, ldb: ^blasint, X: ^f64, ldx: ^blasint, work: ^f64, swork: ^f32, iter: ^blasint, info: ^Info, _: c.size_t) ---
	zcposv_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex128, lda: ^blasint, B: ^complex128, ldb: ^blasint, X: ^complex128, ldx: ^blasint, work: ^complex128, swork: ^complex64, rwork: ^f64, iter: ^blasint, info: ^Info, _: c.size_t) ---
	cposvx_ :: proc(fact: cstring, uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex64, lda: ^blasint, AF: ^complex64, ldaf: ^blasint, equed: cstring, S: ^f32, B: ^complex64, ldb: ^blasint, X: ^complex64, ldx: ^blasint, rcond: ^f32, ferr: ^f32, berr: ^f32, work: ^complex64, rwork: ^f32, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	dposvx_ :: proc(fact: cstring, uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^f64, lda: ^blasint, AF: ^f64, ldaf: ^blasint, equed: cstring, S: ^f64, B: ^f64, ldb: ^blasint, X: ^f64, ldx: ^blasint, rcond: ^f64, ferr: ^f64, berr: ^f64, work: ^f64, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	sposvx_ :: proc(fact: cstring, uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^f32, lda: ^blasint, AF: ^f32, ldaf: ^blasint, equed: cstring, S: ^f32, B: ^f32, ldb: ^blasint, X: ^f32, ldx: ^blasint, rcond: ^f32, ferr: ^f32, berr: ^f32, work: ^f32, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	zposvx_ :: proc(fact: cstring, uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex128, lda: ^blasint, AF: ^complex128, ldaf: ^blasint, equed: cstring, S: ^f64, B: ^complex128, ldb: ^blasint, X: ^complex128, ldx: ^blasint, rcond: ^f64, ferr: ^f64, berr: ^f64, work: ^complex128, rwork: ^f64, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---

	cposvxx_ :: proc(fact: cstring, uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex64, lda: ^blasint, AF: ^complex64, ldaf: ^blasint, equed: cstring, S: ^f32, B: ^complex64, ldb: ^blasint, X: ^complex64, ldx: ^blasint, rcond: ^f32, rpvgrw: ^f32, berr: ^f32, n_err_bnds: ^blasint, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: ^blasint, params: ^f32, work: ^complex64, rwork: ^f32, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	dposvxx_ :: proc(fact: cstring, uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^f64, lda: ^blasint, AF: ^f64, ldaf: ^blasint, equed: cstring, S: ^f64, B: ^f64, ldb: ^blasint, X: ^f64, ldx: ^blasint, rcond: ^f64, rpvgrw: ^f64, berr: ^f64, n_err_bnds: ^blasint, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: ^blasint, params: ^f64, work: ^f64, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	sposvxx_ :: proc(fact: cstring, uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^f32, lda: ^blasint, AF: ^f32, ldaf: ^blasint, equed: cstring, S: ^f32, B: ^f32, ldb: ^blasint, X: ^f32, ldx: ^blasint, rcond: ^f32, rpvgrw: ^f32, berr: ^f32, n_err_bnds: ^blasint, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: ^blasint, params: ^f32, work: ^f32, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	zposvxx_ :: proc(fact: cstring, uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex128, lda: ^blasint, AF: ^complex128, ldaf: ^blasint, equed: cstring, S: ^f64, B: ^complex128, ldb: ^blasint, X: ^complex128, ldx: ^blasint, rcond: ^f64, rpvgrw: ^f64, berr: ^f64, n_err_bnds: ^blasint, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: ^blasint, params: ^f64, work: ^complex128, rwork: ^f64, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	// cpotf2_ :: proc(uplo: cstring, n: ^blasint, A: ^complex64, lda: ^blasint, info: ^Info, _: c.size_t) ---
	// dpotf2_ :: proc(uplo: cstring, n: ^blasint, A: ^f64, lda: ^blasint, info: ^Info, _: c.size_t) ---
	// spotf2_ :: proc(uplo: cstring, n: ^blasint, A: ^f32, lda: ^blasint, info: ^Info, _: c.size_t) ---
	// zpotf2_ :: proc(uplo: cstring, n: ^blasint, A: ^complex128, lda: ^blasint, info: ^Info, _: c.size_t) ---
	// cpotrf_ :: proc(uplo: cstring, n: ^blasint, A: ^complex64, lda: ^blasint, info: ^Info, _: c.size_t) -> i32 ---
	// dpotrf_ :: proc(uplo: cstring, n: ^blasint, A: ^f64, lda: ^blasint, info: ^Info, _: c.size_t) -> i32 ---
	// spotrf_ :: proc(uplo: cstring, n: ^blasint, A: ^f32, lda: ^blasint, info: ^Info, _: c.size_t) -> i32 ---
	// zpotrf_ :: proc(uplo: cstring, n: ^blasint, A: ^complex128, lda: ^blasint, info: ^Info, _: c.size_t) -> i32 ---
	cpotrf2_ :: proc(uplo: cstring, n: ^blasint, A: ^complex64, lda: ^blasint, info: ^Info, _: c.size_t) ---
	dpotrf2_ :: proc(uplo: cstring, n: ^blasint, A: ^f64, lda: ^blasint, info: ^Info, _: c.size_t) ---
	spotrf2_ :: proc(uplo: cstring, n: ^blasint, A: ^f32, lda: ^blasint, info: ^Info, _: c.size_t) ---
	zpotrf2_ :: proc(uplo: cstring, n: ^blasint, A: ^complex128, lda: ^blasint, info: ^Info, _: c.size_t) ---
	// cpotri_ :: proc(uplo: cstring, n: ^blasint, A: ^complex64, lda: ^blasint, info: ^Info, _: c.size_t) ---
	// dpotri_ :: proc(uplo: cstring, n: ^blasint, A: ^f64, lda: ^blasint, info: ^Info, _: c.size_t) ---
	// spotri_ :: proc(uplo: cstring, n: ^blasint, A: ^f32, lda: ^blasint, info: ^Info, _: c.size_t) ---
	// zpotri_ :: proc(uplo: cstring, n: ^blasint, A: ^complex128, lda: ^blasint, info: ^Info, _: c.size_t) ---
	// cpotrs_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex64, lda: ^blasint, B: ^complex64, ldb: ^blasint, info: ^Info, _: c.size_t) ---
	// dpotrs_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^f64, lda: ^blasint, B: ^f64, ldb: ^blasint, info: ^Info, _: c.size_t) ---
	// spotrs_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^f32, lda: ^blasint, B: ^f32, ldb: ^blasint, info: ^Info, _: c.size_t) ---
	// zpotrs_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex128, lda: ^blasint, B: ^complex128, ldb: ^blasint, info: ^Info, _: c.size_t) ---
	cppcon_ :: proc(uplo: cstring, n: ^blasint, AP: ^complex64, anorm: ^f32, rcond: ^f32, work: ^complex64, rwork: ^f32, info: ^Info, _: c.size_t) ---
	dppcon_ :: proc(uplo: cstring, n: ^blasint, AP: ^f64, anorm: ^f64, rcond: ^f64, work: ^f64, iwork: ^blasint, info: ^Info, _: c.size_t) ---
	sppcon_ :: proc(uplo: cstring, n: ^blasint, AP: ^f32, anorm: ^f32, rcond: ^f32, work: ^f32, iwork: ^blasint, info: ^Info, _: c.size_t) ---
	zppcon_ :: proc(uplo: cstring, n: ^blasint, AP: ^complex128, anorm: ^f64, rcond: ^f64, work: ^complex128, rwork: ^f64, info: ^Info, _: c.size_t) ---

	cppequ_ :: proc(uplo: cstring, n: ^blasint, AP: ^complex64, S: ^f32, scond: ^f32, amax: ^f32, info: ^Info, _: c.size_t) ---
	dppequ_ :: proc(uplo: cstring, n: ^blasint, AP: ^f64, S: ^f64, scond: ^f64, amax: ^f64, info: ^Info, _: c.size_t) ---
	sppequ_ :: proc(uplo: cstring, n: ^blasint, AP: ^f32, S: ^f32, scond: ^f32, amax: ^f32, info: ^Info, _: c.size_t) ---
	zppequ_ :: proc(uplo: cstring, n: ^blasint, AP: ^complex128, S: ^f64, scond: ^f64, amax: ^f64, info: ^Info, _: c.size_t) ---

	cpprfs_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, AP: ^complex64, AFP: ^complex64, B: ^complex64, ldb: ^blasint, X: ^complex64, ldx: ^blasint, ferr: ^f32, berr: ^f32, work: ^complex64, rwork: ^f32, info: ^Info, _: c.size_t) ---
	dpprfs_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, AP: ^f64, AFP: ^f64, B: ^f64, ldb: ^blasint, X: ^f64, ldx: ^blasint, ferr: ^f64, berr: ^f64, work: ^f64, iwork: ^blasint, info: ^Info, _: c.size_t) ---
	spprfs_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, AP: ^f32, AFP: ^f32, B: ^f32, ldb: ^blasint, X: ^f32, ldx: ^blasint, ferr: ^f32, berr: ^f32, work: ^f32, iwork: ^blasint, info: ^Info, _: c.size_t) ---
	zpprfs_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, AP: ^complex128, AFP: ^complex128, B: ^complex128, ldb: ^blasint, X: ^complex128, ldx: ^blasint, ferr: ^f64, berr: ^f64, work: ^complex128, rwork: ^f64, info: ^Info, _: c.size_t) ---

	cppsv_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, AP: ^complex64, B: ^complex64, ldb: ^blasint, info: ^Info, _: c.size_t) ---
	dppsv_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, AP: ^f64, B: ^f64, ldb: ^blasint, info: ^Info, _: c.size_t) ---
	sppsv_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, AP: ^f32, B: ^f32, ldb: ^blasint, info: ^Info, _: c.size_t) ---
	zppsv_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, AP: ^complex128, B: ^complex128, ldb: ^blasint, info: ^Info, _: c.size_t) ---

	cppsvx_ :: proc(fact: cstring, uplo: cstring, n: ^blasint, nrhs: ^blasint, AP: ^complex64, AFP: ^complex64, equed: cstring, S: ^f32, B: ^complex64, ldb: ^blasint, X: ^complex64, ldx: ^blasint, rcond: ^f32, ferr: ^f32, berr: ^f32, work: ^complex64, rwork: ^f32, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	dppsvx_ :: proc(fact: cstring, uplo: cstring, n: ^blasint, nrhs: ^blasint, AP: ^f64, AFP: ^f64, equed: cstring, S: ^f64, B: ^f64, ldb: ^blasint, X: ^f64, ldx: ^blasint, rcond: ^f64, ferr: ^f64, berr: ^f64, work: ^f64, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	sppsvx_ :: proc(fact: cstring, uplo: cstring, n: ^blasint, nrhs: ^blasint, AP: ^f32, AFP: ^f32, equed: cstring, S: ^f32, B: ^f32, ldb: ^blasint, X: ^f32, ldx: ^blasint, rcond: ^f32, ferr: ^f32, berr: ^f32, work: ^f32, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	zppsvx_ :: proc(fact: cstring, uplo: cstring, n: ^blasint, nrhs: ^blasint, AP: ^complex128, AFP: ^complex128, equed: cstring, S: ^f64, B: ^complex128, ldb: ^blasint, X: ^complex128, ldx: ^blasint, rcond: ^f64, ferr: ^f64, berr: ^f64, work: ^complex128, rwork: ^f64, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---

	cpptrf_ :: proc(uplo: cstring, n: ^blasint, AP: ^complex64, info: ^Info, _: c.size_t) ---
	dpptrf_ :: proc(uplo: cstring, n: ^blasint, AP: ^f64, info: ^Info, _: c.size_t) ---
	spptrf_ :: proc(uplo: cstring, n: ^blasint, AP: ^f32, info: ^Info, _: c.size_t) ---
	zpptrf_ :: proc(uplo: cstring, n: ^blasint, AP: ^complex128, info: ^Info, _: c.size_t) ---

	cpptri_ :: proc(uplo: cstring, n: ^blasint, AP: ^complex64, info: ^Info, _: c.size_t) ---
	dpptri_ :: proc(uplo: cstring, n: ^blasint, AP: ^f64, info: ^Info, _: c.size_t) ---
	spptri_ :: proc(uplo: cstring, n: ^blasint, AP: ^f32, info: ^Info, _: c.size_t) ---
	zpptri_ :: proc(uplo: cstring, n: ^blasint, AP: ^complex128, info: ^Info, _: c.size_t) ---

	cpptrs_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, AP: ^complex64, B: ^complex64, ldb: ^blasint, info: ^Info, _: c.size_t) ---
	dpptrs_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, AP: ^f64, B: ^f64, ldb: ^blasint, info: ^Info, _: c.size_t) ---
	spptrs_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, AP: ^f32, B: ^f32, ldb: ^blasint, info: ^Info, _: c.size_t) ---
	zpptrs_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, AP: ^complex128, B: ^complex128, ldb: ^blasint, info: ^Info, _: c.size_t) ---

	cpstrf_ :: proc(uplo: cstring, n: ^blasint, A: ^complex64, lda: ^blasint, piv: ^blasint, rank: ^blasint, tol: ^f32, work: ^f32, info: ^Info, _: c.size_t) ---
	dpstrf_ :: proc(uplo: cstring, n: ^blasint, A: ^f64, lda: ^blasint, piv: ^blasint, rank: ^blasint, tol: ^f64, work: ^f64, info: ^Info, _: c.size_t) ---
	spstrf_ :: proc(uplo: cstring, n: ^blasint, A: ^f32, lda: ^blasint, piv: ^blasint, rank: ^blasint, tol: ^f32, work: ^f32, info: ^Info, _: c.size_t) ---
	zpstrf_ :: proc(uplo: cstring, n: ^blasint, A: ^complex128, lda: ^blasint, piv: ^blasint, rank: ^blasint, tol: ^f64, work: ^f64, info: ^Info, _: c.size_t) ---

	cptcon_ :: proc(n: ^blasint, D: ^f32, E: ^complex64, anorm: ^f32, rcond: ^f32, rwork: ^f32, info: ^Info) ---
	dptcon_ :: proc(n: ^blasint, D: ^f64, E: ^f64, anorm: ^f64, rcond: ^f64, work: ^f64, info: ^Info) ---
	sptcon_ :: proc(n: ^blasint, D: ^f32, E: ^f32, anorm: ^f32, rcond: ^f32, work: ^f32, info: ^Info) ---
	zptcon_ :: proc(n: ^blasint, D: ^f64, E: ^complex128, anorm: ^f64, rcond: ^f64, rwork: ^f64, info: ^Info) ---

	cpteqr_ :: proc(compz: cstring, n: ^blasint, D: ^f32, E: ^f32, Z: ^complex64, ldz: ^blasint, work: ^f32, info: ^Info, _: c.size_t) ---
	dpteqr_ :: proc(compz: cstring, n: ^blasint, D: ^f64, E: ^f64, Z: ^f64, ldz: ^blasint, work: ^f64, info: ^Info, _: c.size_t) ---
	spteqr_ :: proc(compz: cstring, n: ^blasint, D: ^f32, E: ^f32, Z: ^f32, ldz: ^blasint, work: ^f32, info: ^Info, _: c.size_t) ---
	zpteqr_ :: proc(compz: cstring, n: ^blasint, D: ^f64, E: ^f64, Z: ^complex128, ldz: ^blasint, work: ^f64, info: ^Info, _: c.size_t) ---

	cptrfs_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, D: ^f32, E: ^complex64, DF: ^f32, EF: ^complex64, B: ^complex64, ldb: ^blasint, X: ^complex64, ldx: ^blasint, ferr: ^f32, berr: ^f32, work: ^complex64, rwork: ^f32, info: ^Info, _: c.size_t) ---
	dptrfs_ :: proc(n: ^blasint, nrhs: ^blasint, D: ^f64, E: ^f64, DF: ^f64, EF: ^f64, B: ^f64, ldb: ^blasint, X: ^f64, ldx: ^blasint, ferr: ^f64, berr: ^f64, work: ^f64, info: ^Info) ---
	sptrfs_ :: proc(n: ^blasint, nrhs: ^blasint, D: ^f32, E: ^f32, DF: ^f32, EF: ^f32, B: ^f32, ldb: ^blasint, X: ^f32, ldx: ^blasint, ferr: ^f32, berr: ^f32, work: ^f32, info: ^Info) ---
	zptrfs_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, D: ^f64, E: ^complex128, DF: ^f64, EF: ^complex128, B: ^complex128, ldb: ^blasint, X: ^complex128, ldx: ^blasint, ferr: ^f64, berr: ^f64, work: ^complex128, rwork: ^f64, info: ^Info, _: c.size_t) ---

	cptsv_ :: proc(n: ^blasint, nrhs: ^blasint, D: ^f32, E: ^complex64, B: ^complex64, ldb: ^blasint, info: ^Info) ---
	dptsv_ :: proc(n: ^blasint, nrhs: ^blasint, D: ^f64, E: ^f64, B: ^f64, ldb: ^blasint, info: ^Info) ---
	sptsv_ :: proc(n: ^blasint, nrhs: ^blasint, D: ^f32, E: ^f32, B: ^f32, ldb: ^blasint, info: ^Info) ---
	zptsv_ :: proc(n: ^blasint, nrhs: ^blasint, D: ^f64, E: ^complex128, B: ^complex128, ldb: ^blasint, info: ^Info) ---

	cptsvx_ :: proc(fact: cstring, n: ^blasint, nrhs: ^blasint, D: ^f32, E: ^complex64, DF: ^f32, EF: ^complex64, B: ^complex64, ldb: ^blasint, X: ^complex64, ldx: ^blasint, rcond: ^f32, ferr: ^f32, berr: ^f32, work: ^complex64, rwork: ^f32, info: ^Info, _: c.size_t) ---
	dptsvx_ :: proc(fact: cstring, n: ^blasint, nrhs: ^blasint, D: ^f64, E: ^f64, DF: ^f64, EF: ^f64, B: ^f64, ldb: ^blasint, X: ^f64, ldx: ^blasint, rcond: ^f64, ferr: ^f64, berr: ^f64, work: ^f64, info: ^Info, _: c.size_t) ---
	sptsvx_ :: proc(fact: cstring, n: ^blasint, nrhs: ^blasint, D: ^f32, E: ^f32, DF: ^f32, EF: ^f32, B: ^f32, ldb: ^blasint, X: ^f32, ldx: ^blasint, rcond: ^f32, ferr: ^f32, berr: ^f32, work: ^f32, info: ^Info, _: c.size_t) ---
	zptsvx_ :: proc(fact: cstring, n: ^blasint, nrhs: ^blasint, D: ^f64, E: ^complex128, DF: ^f64, EF: ^complex128, B: ^complex128, ldb: ^blasint, X: ^complex128, ldx: ^blasint, rcond: ^f64, ferr: ^f64, berr: ^f64, work: ^complex128, rwork: ^f64, info: ^Info, _: c.size_t) ---

	cpttrf_ :: proc(n: ^blasint, D: ^f32, E: ^complex64, info: ^Info) ---
	dpttrf_ :: proc(n: ^blasint, D: ^f64, E: ^f64, info: ^Info) ---
	spttrf_ :: proc(n: ^blasint, D: ^f32, E: ^f32, info: ^Info) ---
	zpttrf_ :: proc(n: ^blasint, D: ^f64, E: ^complex128, info: ^Info) ---

	cpttrs_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, D: ^f32, E: ^complex64, B: ^complex64, ldb: ^blasint, info: ^Info, _: c.size_t) ---
	dpttrs_ :: proc(n: ^blasint, nrhs: ^blasint, D: ^f64, E: ^f64, B: ^f64, ldb: ^blasint, info: ^Info) ---
	spttrs_ :: proc(n: ^blasint, nrhs: ^blasint, D: ^f32, E: ^f32, B: ^f32, ldb: ^blasint, info: ^Info) ---
	zpttrs_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, D: ^f64, E: ^complex128, B: ^complex128, ldb: ^blasint, info: ^Info, _: c.size_t) ---

	dsbev_ :: proc(jobz: cstring, uplo: cstring, n: ^blasint, kd: ^blasint, AB: ^f64, ldab: ^blasint, W: ^f64, Z: ^f64, ldz: ^blasint, work: ^f64, info: ^Info, _: c.size_t, _: c.size_t) ---
	ssbev_ :: proc(jobz: cstring, uplo: cstring, n: ^blasint, kd: ^blasint, AB: ^f32, ldab: ^blasint, W: ^f32, Z: ^f32, ldz: ^blasint, work: ^f32, info: ^Info, _: c.size_t, _: c.size_t) ---
	dsbev_2stage_ :: proc(jobz: cstring, uplo: cstring, n: ^blasint, kd: ^blasint, AB: ^f64, ldab: ^blasint, W: ^f64, Z: ^f64, ldz: ^blasint, work: ^f64, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	ssbev_2stage_ :: proc(jobz: cstring, uplo: cstring, n: ^blasint, kd: ^blasint, AB: ^f32, ldab: ^blasint, W: ^f32, Z: ^f32, ldz: ^blasint, work: ^f32, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---

	dsbevd_ :: proc(jobz: cstring, uplo: cstring, n: ^blasint, kd: ^blasint, AB: ^f64, ldab: ^blasint, W: ^f64, Z: ^f64, ldz: ^blasint, work: ^f64, lwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	ssbevd_ :: proc(jobz: cstring, uplo: cstring, n: ^blasint, kd: ^blasint, AB: ^f32, ldab: ^blasint, W: ^f32, Z: ^f32, ldz: ^blasint, work: ^f32, lwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	dsbevd_2stage_ :: proc(jobz: cstring, uplo: cstring, n: ^blasint, kd: ^blasint, AB: ^f64, ldab: ^blasint, W: ^f64, Z: ^f64, ldz: ^blasint, work: ^f64, lwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	ssbevd_2stage_ :: proc(jobz: cstring, uplo: cstring, n: ^blasint, kd: ^blasint, AB: ^f32, ldab: ^blasint, W: ^f32, Z: ^f32, ldz: ^blasint, work: ^f32, lwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---

	dsbevx_ :: proc(jobz: cstring, range: cstring, uplo: cstring, n: ^blasint, kd: ^blasint, AB: ^f64, ldab: ^blasint, Q: ^f64, ldq: ^blasint, vl: ^f64, vu: ^f64, il: ^blasint, iu: ^blasint, abstol: ^f64, m: ^blasint, W: ^f64, Z: ^f64, ldz: ^blasint, work: ^f64, iwork: ^blasint, IFAIL: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	ssbevx_ :: proc(jobz: cstring, range: cstring, uplo: cstring, n: ^blasint, kd: ^blasint, AB: ^f32, ldab: ^blasint, Q: ^f32, ldq: ^blasint, vl: ^f32, vu: ^f32, il: ^blasint, iu: ^blasint, abstol: ^f32, m: ^blasint, W: ^f32, Z: ^f32, ldz: ^blasint, work: ^f32, iwork: ^blasint, IFAIL: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	dsbevx_2stage_ :: proc(jobz: cstring, range: cstring, uplo: cstring, n: ^blasint, kd: ^blasint, AB: ^f64, ldab: ^blasint, Q: ^f64, ldq: ^blasint, vl: ^f64, vu: ^f64, il: ^blasint, iu: ^blasint, abstol: ^f64, m: ^blasint, W: ^f64, Z: ^f64, ldz: ^blasint, work: ^f64, lwork: ^blasint, iwork: ^blasint, IFAIL: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	ssbevx_2stage_ :: proc(jobz: cstring, range: cstring, uplo: cstring, n: ^blasint, kd: ^blasint, AB: ^f32, ldab: ^blasint, Q: ^f32, ldq: ^blasint, vl: ^f32, vu: ^f32, il: ^blasint, iu: ^blasint, abstol: ^f32, m: ^blasint, W: ^f32, Z: ^f32, ldz: ^blasint, work: ^f32, lwork: ^blasint, iwork: ^blasint, IFAIL: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---

	dsbgst_ :: proc(vect: cstring, uplo: cstring, n: ^blasint, ka: ^blasint, kb: ^blasint, AB: ^f64, ldab: ^blasint, BB: ^f64, ldbb: ^blasint, X: ^f64, ldx: ^blasint, work: ^f64, info: ^Info, _: c.size_t, _: c.size_t) ---
	ssbgst_ :: proc(vect: cstring, uplo: cstring, n: ^blasint, ka: ^blasint, kb: ^blasint, AB: ^f32, ldab: ^blasint, BB: ^f32, ldbb: ^blasint, X: ^f32, ldx: ^blasint, work: ^f32, info: ^Info, _: c.size_t, _: c.size_t) ---
	dsbgv_ :: proc(jobz: cstring, uplo: cstring, n: ^blasint, ka: ^blasint, kb: ^blasint, AB: ^f64, ldab: ^blasint, BB: ^f64, ldbb: ^blasint, W: ^f64, Z: ^f64, ldz: ^blasint, work: ^f64, info: ^Info, _: c.size_t, _: c.size_t) ---
	ssbgv_ :: proc(jobz: cstring, uplo: cstring, n: ^blasint, ka: ^blasint, kb: ^blasint, AB: ^f32, ldab: ^blasint, BB: ^f32, ldbb: ^blasint, W: ^f32, Z: ^f32, ldz: ^blasint, work: ^f32, info: ^Info, _: c.size_t, _: c.size_t) ---
	dsbgvd_ :: proc(jobz: cstring, uplo: cstring, n: ^blasint, ka: ^blasint, kb: ^blasint, AB: ^f64, ldab: ^blasint, BB: ^f64, ldbb: ^blasint, W: ^f64, Z: ^f64, ldz: ^blasint, work: ^f64, lwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	ssbgvd_ :: proc(jobz: cstring, uplo: cstring, n: ^blasint, ka: ^blasint, kb: ^blasint, AB: ^f32, ldab: ^blasint, BB: ^f32, ldbb: ^blasint, W: ^f32, Z: ^f32, ldz: ^blasint, work: ^f32, lwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	dsbgvx_ :: proc(jobz: cstring, range: cstring, uplo: cstring, n: ^blasint, ka: ^blasint, kb: ^blasint, AB: ^f64, ldab: ^blasint, BB: ^f64, ldbb: ^blasint, Q: ^f64, ldq: ^blasint, vl: ^f64, vu: ^f64, il: ^blasint, iu: ^blasint, abstol: ^f64, m: ^blasint, W: ^f64, Z: ^f64, ldz: ^blasint, work: ^f64, iwork: ^blasint, IFAIL: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	ssbgvx_ :: proc(jobz: cstring, range: cstring, uplo: cstring, n: ^blasint, ka: ^blasint, kb: ^blasint, AB: ^f32, ldab: ^blasint, BB: ^f32, ldbb: ^blasint, Q: ^f32, ldq: ^blasint, vl: ^f32, vu: ^f32, il: ^blasint, iu: ^blasint, abstol: ^f32, m: ^blasint, W: ^f32, Z: ^f32, ldz: ^blasint, work: ^f32, iwork: ^blasint, IFAIL: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	dsbtrd_ :: proc(vect: cstring, uplo: cstring, n: ^blasint, kd: ^blasint, AB: ^f64, ldab: ^blasint, D: ^f64, E: ^f64, Q: ^f64, ldq: ^blasint, work: ^f64, info: ^Info, _: c.size_t, _: c.size_t) ---
	ssbtrd_ :: proc(vect: cstring, uplo: cstring, n: ^blasint, kd: ^blasint, AB: ^f32, ldab: ^blasint, D: ^f32, E: ^f32, Q: ^f32, ldq: ^blasint, work: ^f32, info: ^Info, _: c.size_t, _: c.size_t) ---
	dsfrk_ :: proc(transr: cstring, uplo: cstring, trans: cstring, n: ^blasint, k: ^blasint, alpha: ^f64, A: ^f64, lda: ^blasint, beta: ^f64, C: ^f64, _: c.size_t, _: c.size_t, _: c.size_t) ---
	ssfrk_ :: proc(transr: cstring, uplo: cstring, trans: cstring, n: ^blasint, k: ^blasint, alpha: ^f32, A: ^f32, lda: ^blasint, beta: ^f32, C: ^f32, _: c.size_t, _: c.size_t, _: c.size_t) ---

	cspcon_ :: proc(uplo: cstring, n: ^blasint, AP: ^complex64, ipiv: ^blasint, anorm: ^f32, rcond: ^f32, work: ^complex64, info: ^Info, _: c.size_t) ---
	dspcon_ :: proc(uplo: cstring, n: ^blasint, AP: ^f64, ipiv: ^blasint, anorm: ^f64, rcond: ^f64, work: ^f64, iwork: ^blasint, info: ^Info, _: c.size_t) ---
	sspcon_ :: proc(uplo: cstring, n: ^blasint, AP: ^f32, ipiv: ^blasint, anorm: ^f32, rcond: ^f32, work: ^f32, iwork: ^blasint, info: ^Info, _: c.size_t) ---
	zspcon_ :: proc(uplo: cstring, n: ^blasint, AP: ^complex128, ipiv: ^blasint, anorm: ^f64, rcond: ^f64, work: ^complex128, info: ^Info, _: c.size_t) ---

	dspev_ :: proc(jobz: cstring, uplo: cstring, n: ^blasint, AP: ^f64, W: ^f64, Z: ^f64, ldz: ^blasint, work: ^f64, info: ^Info, _: c.size_t, _: c.size_t) ---
	sspev_ :: proc(jobz: cstring, uplo: cstring, n: ^blasint, AP: ^f32, W: ^f32, Z: ^f32, ldz: ^blasint, work: ^f32, info: ^Info, _: c.size_t, _: c.size_t) ---
	dspevd_ :: proc(jobz: cstring, uplo: cstring, n: ^blasint, AP: ^f64, W: ^f64, Z: ^f64, ldz: ^blasint, work: ^f64, lwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	sspevd_ :: proc(jobz: cstring, uplo: cstring, n: ^blasint, AP: ^f32, W: ^f32, Z: ^f32, ldz: ^blasint, work: ^f32, lwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	dspevx_ :: proc(jobz: cstring, range: cstring, uplo: cstring, n: ^blasint, AP: ^f64, vl: ^f64, vu: ^f64, il: ^blasint, iu: ^blasint, abstol: ^f64, m: ^blasint, W: ^f64, Z: ^f64, ldz: ^blasint, work: ^f64, iwork: ^blasint, IFAIL: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	sspevx_ :: proc(jobz: cstring, range: cstring, uplo: cstring, n: ^blasint, AP: ^f32, vl: ^f32, vu: ^f32, il: ^blasint, iu: ^blasint, abstol: ^f32, m: ^blasint, W: ^f32, Z: ^f32, ldz: ^blasint, work: ^f32, iwork: ^blasint, IFAIL: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	dspgst_ :: proc(itype: ^blasint, uplo: cstring, n: ^blasint, AP: ^f64, BP: ^f64, info: ^Info, _: c.size_t) ---
	sspgst_ :: proc(itype: ^blasint, uplo: cstring, n: ^blasint, AP: ^f32, BP: ^f32, info: ^Info, _: c.size_t) ---

	dspgv_ :: proc(itype: ^blasint, jobz: cstring, uplo: cstring, n: ^blasint, AP: ^f64, BP: ^f64, W: ^f64, Z: ^f64, ldz: ^blasint, work: ^f64, info: ^Info, _: c.size_t, _: c.size_t) ---
	sspgv_ :: proc(itype: ^blasint, jobz: cstring, uplo: cstring, n: ^blasint, AP: ^f32, BP: ^f32, W: ^f32, Z: ^f32, ldz: ^blasint, work: ^f32, info: ^Info, _: c.size_t, _: c.size_t) ---
	dspgvd_ :: proc(itype: ^blasint, jobz: cstring, uplo: cstring, n: ^blasint, AP: ^f64, BP: ^f64, W: ^f64, Z: ^f64, ldz: ^blasint, work: ^f64, lwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	sspgvd_ :: proc(itype: ^blasint, jobz: cstring, uplo: cstring, n: ^blasint, AP: ^f32, BP: ^f32, W: ^f32, Z: ^f32, ldz: ^blasint, work: ^f32, lwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	dspgvx_ :: proc(itype: ^blasint, jobz: cstring, range: cstring, uplo: cstring, n: ^blasint, AP: ^f64, BP: ^f64, vl: ^f64, vu: ^f64, il: ^blasint, iu: ^blasint, abstol: ^f64, m: ^blasint, W: ^f64, Z: ^f64, ldz: ^blasint, work: ^f64, iwork: ^blasint, IFAIL: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	sspgvx_ :: proc(itype: ^blasint, jobz: cstring, range: cstring, uplo: cstring, n: ^blasint, AP: ^f32, BP: ^f32, vl: ^f32, vu: ^f32, il: ^blasint, iu: ^blasint, abstol: ^f32, m: ^blasint, W: ^f32, Z: ^f32, ldz: ^blasint, work: ^f32, iwork: ^blasint, IFAIL: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---

	csprfs_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, AP: ^complex64, AFP: ^complex64, ipiv: ^blasint, B: ^complex64, ldb: ^blasint, X: ^complex64, ldx: ^blasint, ferr: ^f32, berr: ^f32, work: ^complex64, rwork: ^f32, info: ^Info, _: c.size_t) ---
	dsprfs_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, AP: ^f64, AFP: ^f64, ipiv: ^blasint, B: ^f64, ldb: ^blasint, X: ^f64, ldx: ^blasint, ferr: ^f64, berr: ^f64, work: ^f64, iwork: ^blasint, info: ^Info, _: c.size_t) ---
	ssprfs_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, AP: ^f32, AFP: ^f32, ipiv: ^blasint, B: ^f32, ldb: ^blasint, X: ^f32, ldx: ^blasint, ferr: ^f32, berr: ^f32, work: ^f32, iwork: ^blasint, info: ^Info, _: c.size_t) ---
	zsprfs_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, AP: ^complex128, AFP: ^complex128, ipiv: ^blasint, B: ^complex128, ldb: ^blasint, X: ^complex128, ldx: ^blasint, ferr: ^f64, berr: ^f64, work: ^complex128, rwork: ^f64, info: ^Info, _: c.size_t) ---

	cspsv_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, AP: ^complex64, ipiv: ^blasint, B: ^complex64, ldb: ^blasint, info: ^Info, _: c.size_t) ---
	dspsv_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, AP: ^f64, ipiv: ^blasint, B: ^f64, ldb: ^blasint, info: ^Info, _: c.size_t) ---
	sspsv_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, AP: ^f32, ipiv: ^blasint, B: ^f32, ldb: ^blasint, info: ^Info, _: c.size_t) ---
	zspsv_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, AP: ^complex128, ipiv: ^blasint, B: ^complex128, ldb: ^blasint, info: ^Info, _: c.size_t) ---

	cspsvx_ :: proc(fact: cstring, uplo: cstring, n: ^blasint, nrhs: ^blasint, AP: ^complex64, AFP: ^complex64, ipiv: ^blasint, B: ^complex64, ldb: ^blasint, X: ^complex64, ldx: ^blasint, rcond: ^f32, ferr: ^f32, berr: ^f32, work: ^complex64, rwork: ^f32, info: ^Info, _: c.size_t, _: c.size_t) ---
	dspsvx_ :: proc(fact: cstring, uplo: cstring, n: ^blasint, nrhs: ^blasint, AP: ^f64, AFP: ^f64, ipiv: ^blasint, B: ^f64, ldb: ^blasint, X: ^f64, ldx: ^blasint, rcond: ^f64, ferr: ^f64, berr: ^f64, work: ^f64, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	sspsvx_ :: proc(fact: cstring, uplo: cstring, n: ^blasint, nrhs: ^blasint, AP: ^f32, AFP: ^f32, ipiv: ^blasint, B: ^f32, ldb: ^blasint, X: ^f32, ldx: ^blasint, rcond: ^f32, ferr: ^f32, berr: ^f32, work: ^f32, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	zspsvx_ :: proc(fact: cstring, uplo: cstring, n: ^blasint, nrhs: ^blasint, AP: ^complex128, AFP: ^complex128, ipiv: ^blasint, B: ^complex128, ldb: ^blasint, X: ^complex128, ldx: ^blasint, rcond: ^f64, ferr: ^f64, berr: ^f64, work: ^complex128, rwork: ^f64, info: ^Info, _: c.size_t, _: c.size_t) ---

	dsptrd_ :: proc(uplo: cstring, n: ^blasint, AP: ^f64, D: ^f64, E: ^f64, tau: ^f64, info: ^Info, _: c.size_t) ---
	ssptrd_ :: proc(uplo: cstring, n: ^blasint, AP: ^f32, D: ^f32, E: ^f32, tau: ^f32, info: ^Info, _: c.size_t) ---
	csptrf_ :: proc(uplo: cstring, n: ^blasint, AP: ^complex64, ipiv: ^blasint, info: ^Info, _: c.size_t) ---
	dsptrf_ :: proc(uplo: cstring, n: ^blasint, AP: ^f64, ipiv: ^blasint, info: ^Info, _: c.size_t) ---
	ssptrf_ :: proc(uplo: cstring, n: ^blasint, AP: ^f32, ipiv: ^blasint, info: ^Info, _: c.size_t) ---
	zsptrf_ :: proc(uplo: cstring, n: ^blasint, AP: ^complex128, ipiv: ^blasint, info: ^Info, _: c.size_t) ---

	csptri_ :: proc(uplo: cstring, n: ^blasint, AP: ^complex64, ipiv: ^blasint, work: ^complex64, info: ^Info, _: c.size_t) ---
	dsptri_ :: proc(uplo: cstring, n: ^blasint, AP: ^f64, ipiv: ^blasint, work: ^f64, info: ^Info, _: c.size_t) ---
	ssptri_ :: proc(uplo: cstring, n: ^blasint, AP: ^f32, ipiv: ^blasint, work: ^f32, info: ^Info, _: c.size_t) ---
	zsptri_ :: proc(uplo: cstring, n: ^blasint, AP: ^complex128, ipiv: ^blasint, work: ^complex128, info: ^Info, _: c.size_t) ---

	csptrs_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, AP: ^complex64, ipiv: ^blasint, B: ^complex64, ldb: ^blasint, info: ^Info, _: c.size_t) ---
	dsptrs_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, AP: ^f64, ipiv: ^blasint, B: ^f64, ldb: ^blasint, info: ^Info, _: c.size_t) ---
	ssptrs_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, AP: ^f32, ipiv: ^blasint, B: ^f32, ldb: ^blasint, info: ^Info, _: c.size_t) ---
	zsptrs_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, AP: ^complex128, ipiv: ^blasint, B: ^complex128, ldb: ^blasint, info: ^Info, _: c.size_t) ---

	dstebz_ :: proc(range: cstring, order: cstring, n: ^blasint, vl: ^f64, vu: ^f64, il: ^blasint, iu: ^blasint, abstol: ^f64, D: ^f64, E: ^f64, m: ^blasint, nsplit: ^blasint, W: ^f64, IBLOCK: ^blasint, ISPLIT: ^blasint, work: ^f64, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	sstebz_ :: proc(range: cstring, order: cstring, n: ^blasint, vl: ^f32, vu: ^f32, il: ^blasint, iu: ^blasint, abstol: ^f32, D: ^f32, E: ^f32, m: ^blasint, nsplit: ^blasint, W: ^f32, IBLOCK: ^blasint, ISPLIT: ^blasint, work: ^f32, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	cstedc_ :: proc(compz: cstring, n: ^blasint, D: ^f32, E: ^f32, Z: ^complex64, ldz: ^blasint, work: ^complex64, lwork: ^blasint, rwork: ^f32, lrwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info, _: c.size_t) ---
	dstedc_ :: proc(compz: cstring, n: ^blasint, D: ^f64, E: ^f64, Z: ^f64, ldz: ^blasint, work: ^f64, lwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info, _: c.size_t) ---
	sstedc_ :: proc(compz: cstring, n: ^blasint, D: ^f32, E: ^f32, Z: ^f32, ldz: ^blasint, work: ^f32, lwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info, _: c.size_t) ---
	zstedc_ :: proc(compz: cstring, n: ^blasint, D: ^f64, E: ^f64, Z: ^complex128, ldz: ^blasint, work: ^complex128, lwork: ^blasint, rwork: ^f64, lrwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info, _: c.size_t) ---

	cstegr_ :: proc(jobz: cstring, range: cstring, n: ^blasint, D: ^f32, E: ^f32, vl: ^f32, vu: ^f32, il: ^blasint, iu: ^blasint, abstol: ^f32, m: ^blasint, W: ^f32, Z: ^complex64, ldz: ^blasint, ISUPPZ: ^blasint, work: ^f32, lwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	dstegr_ :: proc(jobz: cstring, range: cstring, n: ^blasint, D: ^f64, E: ^f64, vl: ^f64, vu: ^f64, il: ^blasint, iu: ^blasint, abstol: ^f64, m: ^blasint, W: ^f64, Z: ^f64, ldz: ^blasint, ISUPPZ: ^blasint, work: ^f64, lwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	sstegr_ :: proc(jobz: cstring, range: cstring, n: ^blasint, D: ^f32, E: ^f32, vl: ^f32, vu: ^f32, il: ^blasint, iu: ^blasint, abstol: ^f32, m: ^blasint, W: ^f32, Z: ^f32, ldz: ^blasint, ISUPPZ: ^blasint, work: ^f32, lwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	zstegr_ :: proc(jobz: cstring, range: cstring, n: ^blasint, D: ^f64, E: ^f64, vl: ^f64, vu: ^f64, il: ^blasint, iu: ^blasint, abstol: ^f64, m: ^blasint, W: ^f64, Z: ^complex128, ldz: ^blasint, ISUPPZ: ^blasint, work: ^f64, lwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---

	cstein_ :: proc(n: ^blasint, D: ^f32, E: ^f32, m: ^blasint, W: ^f32, IBLOCK: ^blasint, ISPLIT: ^blasint, Z: ^complex64, ldz: ^blasint, work: ^f32, iwork: ^blasint, IFAIL: ^blasint, info: ^Info) ---
	dstein_ :: proc(n: ^blasint, D: ^f64, E: ^f64, m: ^blasint, W: ^f64, IBLOCK: ^blasint, ISPLIT: ^blasint, Z: ^f64, ldz: ^blasint, work: ^f64, iwork: ^blasint, IFAIL: ^blasint, info: ^Info) ---
	sstein_ :: proc(n: ^blasint, D: ^f32, E: ^f32, m: ^blasint, W: ^f32, IBLOCK: ^blasint, ISPLIT: ^blasint, Z: ^f32, ldz: ^blasint, work: ^f32, iwork: ^blasint, IFAIL: ^blasint, info: ^Info) ---
	zstein_ :: proc(n: ^blasint, D: ^f64, E: ^f64, m: ^blasint, W: ^f64, IBLOCK: ^blasint, ISPLIT: ^blasint, Z: ^complex128, ldz: ^blasint, work: ^f64, iwork: ^blasint, IFAIL: ^blasint, info: ^Info) ---

	cstemr_ :: proc(jobz: cstring, range: cstring, n: ^blasint, D: ^f32, E: ^f32, vl: ^f32, vu: ^f32, il: ^blasint, iu: ^blasint, m: ^blasint, W: ^f32, Z: ^complex64, ldz: ^blasint, nzc: ^blasint, ISUPPZ: ^blasint, tryrac: ^blasint, work: ^f32, lwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	dstemr_ :: proc(jobz: cstring, range: cstring, n: ^blasint, D: ^f64, E: ^f64, vl: ^f64, vu: ^f64, il: ^blasint, iu: ^blasint, m: ^blasint, W: ^f64, Z: ^f64, ldz: ^blasint, nzc: ^blasint, ISUPPZ: ^blasint, tryrac: ^blasint, work: ^f64, lwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	sstemr_ :: proc(jobz: cstring, range: cstring, n: ^blasint, D: ^f32, E: ^f32, vl: ^f32, vu: ^f32, il: ^blasint, iu: ^blasint, m: ^blasint, W: ^f32, Z: ^f32, ldz: ^blasint, nzc: ^blasint, ISUPPZ: ^blasint, tryrac: ^blasint, work: ^f32, lwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	zstemr_ :: proc(jobz: cstring, range: cstring, n: ^blasint, D: ^f64, E: ^f64, vl: ^f64, vu: ^f64, il: ^blasint, iu: ^blasint, m: ^blasint, W: ^f64, Z: ^complex128, ldz: ^blasint, nzc: ^blasint, ISUPPZ: ^blasint, tryrac: ^blasint, work: ^f64, lwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---

	csteqr_ :: proc(compz: cstring, n: ^blasint, D: ^f32, E: ^f32, Z: ^complex64, ldz: ^blasint, work: ^f32, info: ^Info, _: c.size_t) ---
	dsteqr_ :: proc(compz: cstring, n: ^blasint, D: ^f64, E: ^f64, Z: ^f64, ldz: ^blasint, work: ^f64, info: ^Info, _: c.size_t) ---
	ssteqr_ :: proc(compz: cstring, n: ^blasint, D: ^f32, E: ^f32, Z: ^f32, ldz: ^blasint, work: ^f32, info: ^Info, _: c.size_t) ---
	zsteqr_ :: proc(compz: cstring, n: ^blasint, D: ^f64, E: ^f64, Z: ^complex128, ldz: ^blasint, work: ^f64, info: ^Info, _: c.size_t) ---

	dsterf_ :: proc(n: ^blasint, D: ^f64, E: ^f64, info: ^Info) ---
	ssterf_ :: proc(n: ^blasint, D: ^f32, E: ^f32, info: ^Info) ---
	dstev_ :: proc(jobz: cstring, n: ^blasint, D: ^f64, E: ^f64, Z: ^f64, ldz: ^blasint, work: ^f64, info: ^Info, _: c.size_t) ---
	sstev_ :: proc(jobz: cstring, n: ^blasint, D: ^f32, E: ^f32, Z: ^f32, ldz: ^blasint, work: ^f32, info: ^Info, _: c.size_t) ---
	dstevd_ :: proc(jobz: cstring, n: ^blasint, D: ^f64, E: ^f64, Z: ^f64, ldz: ^blasint, work: ^f64, lwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info, _: c.size_t) ---
	sstevd_ :: proc(jobz: cstring, n: ^blasint, D: ^f32, E: ^f32, Z: ^f32, ldz: ^blasint, work: ^f32, lwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info, _: c.size_t) ---
	dstevr_ :: proc(jobz: cstring, range: cstring, n: ^blasint, D: ^f64, E: ^f64, vl: ^f64, vu: ^f64, il: ^blasint, iu: ^blasint, abstol: ^f64, m: ^blasint, W: ^f64, Z: ^f64, ldz: ^blasint, ISUPPZ: ^blasint, work: ^f64, lwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	sstevr_ :: proc(jobz: cstring, range: cstring, n: ^blasint, D: ^f32, E: ^f32, vl: ^f32, vu: ^f32, il: ^blasint, iu: ^blasint, abstol: ^f32, m: ^blasint, W: ^f32, Z: ^f32, ldz: ^blasint, ISUPPZ: ^blasint, work: ^f32, lwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	dstevx_ :: proc(jobz: cstring, range: cstring, n: ^blasint, D: ^f64, E: ^f64, vl: ^f64, vu: ^f64, il: ^blasint, iu: ^blasint, abstol: ^f64, m: ^blasint, W: ^f64, Z: ^f64, ldz: ^blasint, work: ^f64, iwork: ^blasint, IFAIL: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	sstevx_ :: proc(jobz: cstring, range: cstring, n: ^blasint, D: ^f32, E: ^f32, vl: ^f32, vu: ^f32, il: ^blasint, iu: ^blasint, abstol: ^f32, m: ^blasint, W: ^f32, Z: ^f32, ldz: ^blasint, work: ^f32, iwork: ^blasint, IFAIL: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	csycon_ :: proc(uplo: cstring, n: ^blasint, A: ^complex64, lda: ^blasint, ipiv: ^blasint, anorm: ^f32, rcond: ^f32, work: ^complex64, info: ^Info, _: c.size_t) ---
	dsycon_ :: proc(uplo: cstring, n: ^blasint, A: ^f64, lda: ^blasint, ipiv: ^blasint, anorm: ^f64, rcond: ^f64, work: ^f64, iwork: ^blasint, info: ^Info, _: c.size_t) ---
	ssycon_ :: proc(uplo: cstring, n: ^blasint, A: ^f32, lda: ^blasint, ipiv: ^blasint, anorm: ^f32, rcond: ^f32, work: ^f32, iwork: ^blasint, info: ^Info, _: c.size_t) ---
	zsycon_ :: proc(uplo: cstring, n: ^blasint, A: ^complex128, lda: ^blasint, ipiv: ^blasint, anorm: ^f64, rcond: ^f64, work: ^complex128, info: ^Info, _: c.size_t) ---

	csycon_3_ :: proc(uplo: cstring, n: ^blasint, A: ^complex64, lda: ^blasint, E: ^complex64, ipiv: ^blasint, anorm: ^f32, rcond: ^f32, work: ^complex64, info: ^Info, _: c.size_t) ---
	dsycon_3_ :: proc(uplo: cstring, n: ^blasint, A: ^f64, lda: ^blasint, E: ^f64, ipiv: ^blasint, anorm: ^f64, rcond: ^f64, work: ^f64, iwork: ^blasint, info: ^Info, _: c.size_t) ---
	ssycon_3_ :: proc(uplo: cstring, n: ^blasint, A: ^f32, lda: ^blasint, E: ^f32, ipiv: ^blasint, anorm: ^f32, rcond: ^f32, work: ^f32, iwork: ^blasint, info: ^Info, _: c.size_t) ---
	zsycon_3_ :: proc(uplo: cstring, n: ^blasint, A: ^complex128, lda: ^blasint, E: ^complex128, ipiv: ^blasint, anorm: ^f64, rcond: ^f64, work: ^complex128, info: ^Info, _: c.size_t) ---

	csyconv_ :: proc(uplo: cstring, way: cstring, n: ^blasint, A: ^complex64, lda: ^blasint, ipiv: ^blasint, E: ^complex64, info: ^Info, _: c.size_t, _: c.size_t) ---
	dsyconv_ :: proc(uplo: cstring, way: cstring, n: ^blasint, A: ^f64, lda: ^blasint, ipiv: ^blasint, E: ^f64, info: ^Info, _: c.size_t, _: c.size_t) ---
	ssyconv_ :: proc(uplo: cstring, way: cstring, n: ^blasint, A: ^f32, lda: ^blasint, ipiv: ^blasint, E: ^f32, info: ^Info, _: c.size_t, _: c.size_t) ---
	zsyconv_ :: proc(uplo: cstring, way: cstring, n: ^blasint, A: ^complex128, lda: ^blasint, ipiv: ^blasint, E: ^complex128, info: ^Info, _: c.size_t, _: c.size_t) ---

	csyequb_ :: proc(uplo: cstring, n: ^blasint, A: ^complex64, lda: ^blasint, S: ^f32, scond: ^f32, amax: ^f32, work: ^complex64, info: ^Info, _: c.size_t) ---
	dsyequb_ :: proc(uplo: cstring, n: ^blasint, A: ^f64, lda: ^blasint, S: ^f64, scond: ^f64, amax: ^f64, work: ^f64, info: ^Info, _: c.size_t) ---
	ssyequb_ :: proc(uplo: cstring, n: ^blasint, A: ^f32, lda: ^blasint, S: ^f32, scond: ^f32, amax: ^f32, work: ^f32, info: ^Info, _: c.size_t) ---
	zsyequb_ :: proc(uplo: cstring, n: ^blasint, A: ^complex128, lda: ^blasint, S: ^f64, scond: ^f64, amax: ^f64, work: ^complex128, info: ^Info, _: c.size_t) ---

	dsyev_ :: proc(jobz: cstring, uplo: cstring, n: ^blasint, A: ^f64, lda: ^blasint, W: ^f64, work: ^f64, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	ssyev_ :: proc(jobz: cstring, uplo: cstring, n: ^blasint, A: ^f32, lda: ^blasint, W: ^f32, work: ^f32, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	dsyev_2stage_ :: proc(jobz: cstring, uplo: cstring, n: ^blasint, A: ^f64, lda: ^blasint, W: ^f64, work: ^f64, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	ssyev_2stage_ :: proc(jobz: cstring, uplo: cstring, n: ^blasint, A: ^f32, lda: ^blasint, W: ^f32, work: ^f32, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---

	dsyevd_ :: proc(jobz: cstring, uplo: cstring, n: ^blasint, A: ^f64, lda: ^blasint, W: ^f64, work: ^f64, lwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	ssyevd_ :: proc(jobz: cstring, uplo: cstring, n: ^blasint, A: ^f32, lda: ^blasint, W: ^f32, work: ^f32, lwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	dsyevd_2stage_ :: proc(jobz: cstring, uplo: cstring, n: ^blasint, A: ^f64, lda: ^blasint, W: ^f64, work: ^f64, lwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	ssyevd_2stage_ :: proc(jobz: cstring, uplo: cstring, n: ^blasint, A: ^f32, lda: ^blasint, W: ^f32, work: ^f32, lwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---

	dsyevr_ :: proc(jobz: cstring, range: cstring, uplo: cstring, n: ^blasint, A: ^f64, lda: ^blasint, vl: ^f64, vu: ^f64, il: ^blasint, iu: ^blasint, abstol: ^f64, m: ^blasint, W: ^f64, Z: ^f64, ldz: ^blasint, ISUPPZ: ^blasint, work: ^f64, lwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	ssyevr_ :: proc(jobz: cstring, range: cstring, uplo: cstring, n: ^blasint, A: ^f32, lda: ^blasint, vl: ^f32, vu: ^f32, il: ^blasint, iu: ^blasint, abstol: ^f32, m: ^blasint, W: ^f32, Z: ^f32, ldz: ^blasint, ISUPPZ: ^blasint, work: ^f32, lwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	dsyevr_2stage_ :: proc(jobz: cstring, range: cstring, uplo: cstring, n: ^blasint, A: ^f64, lda: ^blasint, vl: ^f64, vu: ^f64, il: ^blasint, iu: ^blasint, abstol: ^f64, m: ^blasint, W: ^f64, Z: ^f64, ldz: ^blasint, ISUPPZ: ^blasint, work: ^f64, lwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	ssyevr_2stage_ :: proc(jobz: cstring, range: cstring, uplo: cstring, n: ^blasint, A: ^f32, lda: ^blasint, vl: ^f32, vu: ^f32, il: ^blasint, iu: ^blasint, abstol: ^f32, m: ^blasint, W: ^f32, Z: ^f32, ldz: ^blasint, ISUPPZ: ^blasint, work: ^f32, lwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---

	dsyevx_ :: proc(jobz: cstring, range: cstring, uplo: cstring, n: ^blasint, A: ^f64, lda: ^blasint, vl: ^f64, vu: ^f64, il: ^blasint, iu: ^blasint, abstol: ^f64, m: ^blasint, W: ^f64, Z: ^f64, ldz: ^blasint, work: ^f64, lwork: ^blasint, iwork: ^blasint, IFAIL: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	ssyevx_ :: proc(jobz: cstring, range: cstring, uplo: cstring, n: ^blasint, A: ^f32, lda: ^blasint, vl: ^f32, vu: ^f32, il: ^blasint, iu: ^blasint, abstol: ^f32, m: ^blasint, W: ^f32, Z: ^f32, ldz: ^blasint, work: ^f32, lwork: ^blasint, iwork: ^blasint, IFAIL: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	dsyevx_2stage_ :: proc(jobz: cstring, range: cstring, uplo: cstring, n: ^blasint, A: ^f64, lda: ^blasint, vl: ^f64, vu: ^f64, il: ^blasint, iu: ^blasint, abstol: ^f64, m: ^blasint, W: ^f64, Z: ^f64, ldz: ^blasint, work: ^f64, lwork: ^blasint, iwork: ^blasint, IFAIL: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	ssyevx_2stage_ :: proc(jobz: cstring, range: cstring, uplo: cstring, n: ^blasint, A: ^f32, lda: ^blasint, vl: ^f32, vu: ^f32, il: ^blasint, iu: ^blasint, abstol: ^f32, m: ^blasint, W: ^f32, Z: ^f32, ldz: ^blasint, work: ^f32, lwork: ^blasint, iwork: ^blasint, IFAIL: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---

	dsygst_ :: proc(itype: ^blasint, uplo: cstring, n: ^blasint, A: ^f64, lda: ^blasint, B: ^f64, ldb: ^blasint, info: ^Info, _: c.size_t) ---
	ssygst_ :: proc(itype: ^blasint, uplo: cstring, n: ^blasint, A: ^f32, lda: ^blasint, B: ^f32, ldb: ^blasint, info: ^Info, _: c.size_t) ---
	dsygv_ :: proc(itype: ^blasint, jobz: cstring, uplo: cstring, n: ^blasint, A: ^f64, lda: ^blasint, B: ^f64, ldb: ^blasint, W: ^f64, work: ^f64, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	ssygv_ :: proc(itype: ^blasint, jobz: cstring, uplo: cstring, n: ^blasint, A: ^f32, lda: ^blasint, B: ^f32, ldb: ^blasint, W: ^f32, work: ^f32, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	dsygv_2stage_ :: proc(itype: ^blasint, jobz: cstring, uplo: cstring, n: ^blasint, A: ^f64, lda: ^blasint, B: ^f64, ldb: ^blasint, W: ^f64, work: ^f64, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	ssygv_2stage_ :: proc(itype: ^blasint, jobz: cstring, uplo: cstring, n: ^blasint, A: ^f32, lda: ^blasint, B: ^f32, ldb: ^blasint, W: ^f32, work: ^f32, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---

	dsygvd_ :: proc(itype: ^blasint, jobz: cstring, uplo: cstring, n: ^blasint, A: ^f64, lda: ^blasint, B: ^f64, ldb: ^blasint, W: ^f64, work: ^f64, lwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	ssygvd_ :: proc(itype: ^blasint, jobz: cstring, uplo: cstring, n: ^blasint, A: ^f32, lda: ^blasint, B: ^f32, ldb: ^blasint, W: ^f32, work: ^f32, lwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	dsygvx_ :: proc(itype: ^blasint, jobz: cstring, range: cstring, uplo: cstring, n: ^blasint, A: ^f64, lda: ^blasint, B: ^f64, ldb: ^blasint, vl: ^f64, vu: ^f64, il: ^blasint, iu: ^blasint, abstol: ^f64, m: ^blasint, W: ^f64, Z: ^f64, ldz: ^blasint, work: ^f64, lwork: ^blasint, iwork: ^blasint, IFAIL: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	ssygvx_ :: proc(itype: ^blasint, jobz: cstring, range: cstring, uplo: cstring, n: ^blasint, A: ^f32, lda: ^blasint, B: ^f32, ldb: ^blasint, vl: ^f32, vu: ^f32, il: ^blasint, iu: ^blasint, abstol: ^f32, m: ^blasint, W: ^f32, Z: ^f32, ldz: ^blasint, work: ^f32, lwork: ^blasint, iwork: ^blasint, IFAIL: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	// csyr_ :: proc(uplo: cstring, n: ^blasint, alpha: ^complex64, X: ^complex64, incx: ^blasint, A: ^complex64, lda: ^blasint, _: c.size_t) ---
	// zsyr_ :: proc(uplo: cstring, n: ^blasint, alpha: ^complex128, X: ^complex128, incx: ^blasint, A: ^complex128, lda: ^blasint, _: c.size_t) ---
	csyrfs_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex64, lda: ^blasint, AF: ^complex64, ldaf: ^blasint, ipiv: ^blasint, B: ^complex64, ldb: ^blasint, X: ^complex64, ldx: ^blasint, ferr: ^f32, berr: ^f32, work: ^complex64, rwork: ^f32, info: ^Info, _: c.size_t) ---
	dsyrfs_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^f64, lda: ^blasint, AF: ^f64, ldaf: ^blasint, ipiv: ^blasint, B: ^f64, ldb: ^blasint, X: ^f64, ldx: ^blasint, ferr: ^f64, berr: ^f64, work: ^f64, iwork: ^blasint, info: ^Info, _: c.size_t) ---
	ssyrfs_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^f32, lda: ^blasint, AF: ^f32, ldaf: ^blasint, ipiv: ^blasint, B: ^f32, ldb: ^blasint, X: ^f32, ldx: ^blasint, ferr: ^f32, berr: ^f32, work: ^f32, iwork: ^blasint, info: ^Info, _: c.size_t) ---
	zsyrfs_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex128, lda: ^blasint, AF: ^complex128, ldaf: ^blasint, ipiv: ^blasint, B: ^complex128, ldb: ^blasint, X: ^complex128, ldx: ^blasint, ferr: ^f64, berr: ^f64, work: ^complex128, rwork: ^f64, info: ^Info, _: c.size_t) ---

	csyrfsx_ :: proc(uplo: cstring, equed: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex64, lda: ^blasint, AF: ^complex64, ldaf: ^blasint, ipiv: ^blasint, S: ^f32, B: ^complex64, ldb: ^blasint, X: ^complex64, ldx: ^blasint, rcond: ^f32, berr: ^f32, n_err_bnds: ^blasint, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: ^blasint, params: ^f32, work: ^complex64, rwork: ^f32, info: ^Info, _: c.size_t, _: c.size_t) ---
	dsyrfsx_ :: proc(uplo: cstring, equed: cstring, n: ^blasint, nrhs: ^blasint, A: ^f64, lda: ^blasint, AF: ^f64, ldaf: ^blasint, ipiv: ^blasint, S: ^f64, B: ^f64, ldb: ^blasint, X: ^f64, ldx: ^blasint, rcond: ^f64, berr: ^f64, n_err_bnds: ^blasint, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: ^blasint, params: ^f64, work: ^f64, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	ssyrfsx_ :: proc(uplo: cstring, equed: cstring, n: ^blasint, nrhs: ^blasint, A: ^f32, lda: ^blasint, AF: ^f32, ldaf: ^blasint, ipiv: ^blasint, S: ^f32, B: ^f32, ldb: ^blasint, X: ^f32, ldx: ^blasint, rcond: ^f32, berr: ^f32, n_err_bnds: ^blasint, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: ^blasint, params: ^f32, work: ^f32, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	zsyrfsx_ :: proc(uplo: cstring, equed: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex128, lda: ^blasint, AF: ^complex128, ldaf: ^blasint, ipiv: ^blasint, S: ^f64, B: ^complex128, ldb: ^blasint, X: ^complex128, ldx: ^blasint, rcond: ^f64, berr: ^f64, n_err_bnds: ^blasint, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: ^blasint, params: ^f64, work: ^complex128, rwork: ^f64, info: ^Info, _: c.size_t, _: c.size_t) ---

	csysv_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex64, lda: ^blasint, ipiv: ^blasint, B: ^complex64, ldb: ^blasint, work: ^complex64, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	dsysv_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^f64, lda: ^blasint, ipiv: ^blasint, B: ^f64, ldb: ^blasint, work: ^f64, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	ssysv_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^f32, lda: ^blasint, ipiv: ^blasint, B: ^f32, ldb: ^blasint, work: ^f32, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	zsysv_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex128, lda: ^blasint, ipiv: ^blasint, B: ^complex128, ldb: ^blasint, work: ^complex128, lwork: ^blasint, info: ^Info, _: c.size_t) ---

	csysv_aa_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex64, lda: ^blasint, ipiv: ^blasint, B: ^complex64, ldb: ^blasint, work: ^complex64, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	dsysv_aa_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^f64, lda: ^blasint, ipiv: ^blasint, B: ^f64, ldb: ^blasint, work: ^f64, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	ssysv_aa_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^f32, lda: ^blasint, ipiv: ^blasint, B: ^f32, ldb: ^blasint, work: ^f32, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	zsysv_aa_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex128, lda: ^blasint, ipiv: ^blasint, B: ^complex128, ldb: ^blasint, work: ^complex128, lwork: ^blasint, info: ^Info, _: c.size_t) ---

	csysv_aa_2stage_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex64, lda: ^blasint, TB: ^complex64, ltb: ^blasint, ipiv: ^blasint, ipiv2: ^blasint, B: ^complex64, ldb: ^blasint, work: ^complex64, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	dsysv_aa_2stage_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^f64, lda: ^blasint, TB: ^f64, ltb: ^blasint, ipiv: ^blasint, ipiv2: ^blasint, B: ^f64, ldb: ^blasint, work: ^f64, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	ssysv_aa_2stage_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^f32, lda: ^blasint, TB: ^f32, ltb: ^blasint, ipiv: ^blasint, ipiv2: ^blasint, B: ^f32, ldb: ^blasint, work: ^f32, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	zsysv_aa_2stage_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex128, lda: ^blasint, TB: ^complex128, ltb: ^blasint, ipiv: ^blasint, ipiv2: ^blasint, B: ^complex128, ldb: ^blasint, work: ^complex128, lwork: ^blasint, info: ^Info, _: c.size_t) ---

	csysv_rk_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex64, lda: ^blasint, E: ^complex64, ipiv: ^blasint, B: ^complex64, ldb: ^blasint, work: ^complex64, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	dsysv_rk_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^f64, lda: ^blasint, E: ^f64, ipiv: ^blasint, B: ^f64, ldb: ^blasint, work: ^f64, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	ssysv_rk_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^f32, lda: ^blasint, E: ^f32, ipiv: ^blasint, B: ^f32, ldb: ^blasint, work: ^f32, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	zsysv_rk_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex128, lda: ^blasint, E: ^complex128, ipiv: ^blasint, B: ^complex128, ldb: ^blasint, work: ^complex128, lwork: ^blasint, info: ^Info, _: c.size_t) ---

	csysv_rook_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex64, lda: ^blasint, ipiv: ^blasint, B: ^complex64, ldb: ^blasint, work: ^complex64, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	dsysv_rook_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^f64, lda: ^blasint, ipiv: ^blasint, B: ^f64, ldb: ^blasint, work: ^f64, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	ssysv_rook_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^f32, lda: ^blasint, ipiv: ^blasint, B: ^f32, ldb: ^blasint, work: ^f32, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	zsysv_rook_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex128, lda: ^blasint, ipiv: ^blasint, B: ^complex128, ldb: ^blasint, work: ^complex128, lwork: ^blasint, info: ^Info, _: c.size_t) ---

	csysvx_ :: proc(fact: cstring, uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex64, lda: ^blasint, AF: ^complex64, ldaf: ^blasint, ipiv: ^blasint, B: ^complex64, ldb: ^blasint, X: ^complex64, ldx: ^blasint, rcond: ^f32, ferr: ^f32, berr: ^f32, work: ^complex64, lwork: ^blasint, rwork: ^f32, info: ^Info, _: c.size_t, _: c.size_t) ---
	dsysvx_ :: proc(fact: cstring, uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^f64, lda: ^blasint, AF: ^f64, ldaf: ^blasint, ipiv: ^blasint, B: ^f64, ldb: ^blasint, X: ^f64, ldx: ^blasint, rcond: ^f64, ferr: ^f64, berr: ^f64, work: ^f64, lwork: ^blasint, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	ssysvx_ :: proc(fact: cstring, uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^f32, lda: ^blasint, AF: ^f32, ldaf: ^blasint, ipiv: ^blasint, B: ^f32, ldb: ^blasint, X: ^f32, ldx: ^blasint, rcond: ^f32, ferr: ^f32, berr: ^f32, work: ^f32, lwork: ^blasint, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	zsysvx_ :: proc(fact: cstring, uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex128, lda: ^blasint, AF: ^complex128, ldaf: ^blasint, ipiv: ^blasint, B: ^complex128, ldb: ^blasint, X: ^complex128, ldx: ^blasint, rcond: ^f64, ferr: ^f64, berr: ^f64, work: ^complex128, lwork: ^blasint, rwork: ^f64, info: ^Info, _: c.size_t, _: c.size_t) ---

	csysvxx_ :: proc(fact: cstring, uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex64, lda: ^blasint, AF: ^complex64, ldaf: ^blasint, ipiv: ^blasint, equed: cstring, S: ^f32, B: ^complex64, ldb: ^blasint, X: ^complex64, ldx: ^blasint, rcond: ^f32, rpvgrw: ^f32, berr: ^f32, n_err_bnds: ^blasint, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: ^blasint, params: ^f32, work: ^complex64, rwork: ^f32, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	dsysvxx_ :: proc(fact: cstring, uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^f64, lda: ^blasint, AF: ^f64, ldaf: ^blasint, ipiv: ^blasint, equed: cstring, S: ^f64, B: ^f64, ldb: ^blasint, X: ^f64, ldx: ^blasint, rcond: ^f64, rpvgrw: ^f64, berr: ^f64, n_err_bnds: ^blasint, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: ^blasint, params: ^f64, work: ^f64, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	ssysvxx_ :: proc(fact: cstring, uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^f32, lda: ^blasint, AF: ^f32, ldaf: ^blasint, ipiv: ^blasint, equed: cstring, S: ^f32, B: ^f32, ldb: ^blasint, X: ^f32, ldx: ^blasint, rcond: ^f32, rpvgrw: ^f32, berr: ^f32, n_err_bnds: ^blasint, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: ^blasint, params: ^f32, work: ^f32, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	zsysvxx_ :: proc(fact: cstring, uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex128, lda: ^blasint, AF: ^complex128, ldaf: ^blasint, ipiv: ^blasint, equed: cstring, S: ^f64, B: ^complex128, ldb: ^blasint, X: ^complex128, ldx: ^blasint, rcond: ^f64, rpvgrw: ^f64, berr: ^f64, n_err_bnds: ^blasint, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: ^blasint, params: ^f64, work: ^complex128, rwork: ^f64, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---

	csyswapr_ :: proc(uplo: cstring, n: ^blasint, A: ^complex64, lda: ^blasint, i1: ^blasint, i2: ^blasint, _: c.size_t) ---
	dsyswapr_ :: proc(uplo: cstring, n: ^blasint, A: ^f64, lda: ^blasint, i1: ^blasint, i2: ^blasint, _: c.size_t) ---
	ssyswapr_ :: proc(uplo: cstring, n: ^blasint, A: ^f32, lda: ^blasint, i1: ^blasint, i2: ^blasint, _: c.size_t) ---
	zsyswapr_ :: proc(uplo: cstring, n: ^blasint, A: ^complex128, lda: ^blasint, i1: ^blasint, i2: ^blasint, _: c.size_t) ---

	dsytrd_ :: proc(uplo: cstring, n: ^blasint, A: ^f64, lda: ^blasint, D: ^f64, E: ^f64, tau: ^f64, work: ^f64, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	ssytrd_ :: proc(uplo: cstring, n: ^blasint, A: ^f32, lda: ^blasint, D: ^f32, E: ^f32, tau: ^f32, work: ^f32, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	dsytrd_2stage_ :: proc(vect: cstring, uplo: cstring, n: ^blasint, A: ^f64, lda: ^blasint, D: ^f64, E: ^f64, tau: ^f64, HOUS2: ^f64, lhous2: ^blasint, work: ^f64, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	ssytrd_2stage_ :: proc(vect: cstring, uplo: cstring, n: ^blasint, A: ^f32, lda: ^blasint, D: ^f32, E: ^f32, tau: ^f32, HOUS2: ^f32, lhous2: ^blasint, work: ^f32, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---

	csytrf_ :: proc(uplo: cstring, n: ^blasint, A: ^complex64, lda: ^blasint, ipiv: ^blasint, work: ^complex64, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	dsytrf_ :: proc(uplo: cstring, n: ^blasint, A: ^f64, lda: ^blasint, ipiv: ^blasint, work: ^f64, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	ssytrf_ :: proc(uplo: cstring, n: ^blasint, A: ^f32, lda: ^blasint, ipiv: ^blasint, work: ^f32, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	zsytrf_ :: proc(uplo: cstring, n: ^blasint, A: ^complex128, lda: ^blasint, ipiv: ^blasint, work: ^complex128, lwork: ^blasint, info: ^Info, _: c.size_t) ---

	csytrf_aa_ :: proc(uplo: cstring, n: ^blasint, A: ^complex64, lda: ^blasint, ipiv: ^blasint, work: ^complex64, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	dsytrf_aa_ :: proc(uplo: cstring, n: ^blasint, A: ^f64, lda: ^blasint, ipiv: ^blasint, work: ^f64, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	ssytrf_aa_ :: proc(uplo: cstring, n: ^blasint, A: ^f32, lda: ^blasint, ipiv: ^blasint, work: ^f32, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	zsytrf_aa_ :: proc(uplo: cstring, n: ^blasint, A: ^complex128, lda: ^blasint, ipiv: ^blasint, work: ^complex128, lwork: ^blasint, info: ^Info, _: c.size_t) ---

	csytrf_aa_2stage_ :: proc(uplo: cstring, n: ^blasint, A: ^complex64, lda: ^blasint, TB: ^complex64, ltb: ^blasint, ipiv: ^blasint, ipiv2: ^blasint, work: ^complex64, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	dsytrf_aa_2stage_ :: proc(uplo: cstring, n: ^blasint, A: ^f64, lda: ^blasint, TB: ^f64, ltb: ^blasint, ipiv: ^blasint, ipiv2: ^blasint, work: ^f64, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	ssytrf_aa_2stage_ :: proc(uplo: cstring, n: ^blasint, A: ^f32, lda: ^blasint, TB: ^f32, ltb: ^blasint, ipiv: ^blasint, ipiv2: ^blasint, work: ^f32, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	zsytrf_aa_2stage_ :: proc(uplo: cstring, n: ^blasint, A: ^complex128, lda: ^blasint, TB: ^complex128, ltb: ^blasint, ipiv: ^blasint, ipiv2: ^blasint, work: ^complex128, lwork: ^blasint, info: ^Info, _: c.size_t) ---

	csytrf_rk_ :: proc(uplo: cstring, n: ^blasint, A: ^complex64, lda: ^blasint, E: ^complex64, ipiv: ^blasint, work: ^complex64, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	dsytrf_rk_ :: proc(uplo: cstring, n: ^blasint, A: ^f64, lda: ^blasint, E: ^f64, ipiv: ^blasint, work: ^f64, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	ssytrf_rk_ :: proc(uplo: cstring, n: ^blasint, A: ^f32, lda: ^blasint, E: ^f32, ipiv: ^blasint, work: ^f32, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	zsytrf_rk_ :: proc(uplo: cstring, n: ^blasint, A: ^complex128, lda: ^blasint, E: ^complex128, ipiv: ^blasint, work: ^complex128, lwork: ^blasint, info: ^Info, _: c.size_t) ---

	csytrf_rook_ :: proc(uplo: cstring, n: ^blasint, A: ^complex64, lda: ^blasint, ipiv: ^blasint, work: ^complex64, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	dsytrf_rook_ :: proc(uplo: cstring, n: ^blasint, A: ^f64, lda: ^blasint, ipiv: ^blasint, work: ^f64, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	ssytrf_rook_ :: proc(uplo: cstring, n: ^blasint, A: ^f32, lda: ^blasint, ipiv: ^blasint, work: ^f32, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	zsytrf_rook_ :: proc(uplo: cstring, n: ^blasint, A: ^complex128, lda: ^blasint, ipiv: ^blasint, work: ^complex128, lwork: ^blasint, info: ^Info, _: c.size_t) ---

	csytri_ :: proc(uplo: cstring, n: ^blasint, A: ^complex64, lda: ^blasint, ipiv: ^blasint, work: ^complex64, info: ^Info, _: c.size_t) ---
	dsytri_ :: proc(uplo: cstring, n: ^blasint, A: ^f64, lda: ^blasint, ipiv: ^blasint, work: ^f64, info: ^Info, _: c.size_t) ---
	ssytri_ :: proc(uplo: cstring, n: ^blasint, A: ^f32, lda: ^blasint, ipiv: ^blasint, work: ^f32, info: ^Info, _: c.size_t) ---
	zsytri_ :: proc(uplo: cstring, n: ^blasint, A: ^complex128, lda: ^blasint, ipiv: ^blasint, work: ^complex128, info: ^Info, _: c.size_t) ---

	csytri2_ :: proc(uplo: cstring, n: ^blasint, A: ^complex64, lda: ^blasint, ipiv: ^blasint, work: ^complex64, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	dsytri2_ :: proc(uplo: cstring, n: ^blasint, A: ^f64, lda: ^blasint, ipiv: ^blasint, work: ^f64, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	ssytri2_ :: proc(uplo: cstring, n: ^blasint, A: ^f32, lda: ^blasint, ipiv: ^blasint, work: ^f32, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	zsytri2_ :: proc(uplo: cstring, n: ^blasint, A: ^complex128, lda: ^blasint, ipiv: ^blasint, work: ^complex128, lwork: ^blasint, info: ^Info, _: c.size_t) ---

	csytri2x_ :: proc(uplo: cstring, n: ^blasint, A: ^complex64, lda: ^blasint, ipiv: ^blasint, work: ^complex64, nb: ^blasint, info: ^Info, _: c.size_t) ---
	dsytri2x_ :: proc(uplo: cstring, n: ^blasint, A: ^f64, lda: ^blasint, ipiv: ^blasint, work: ^f64, nb: ^blasint, info: ^Info, _: c.size_t) ---
	ssytri2x_ :: proc(uplo: cstring, n: ^blasint, A: ^f32, lda: ^blasint, ipiv: ^blasint, work: ^f32, nb: ^blasint, info: ^Info, _: c.size_t) ---
	zsytri2x_ :: proc(uplo: cstring, n: ^blasint, A: ^complex128, lda: ^blasint, ipiv: ^blasint, work: ^complex128, nb: ^blasint, info: ^Info, _: c.size_t) ---

	csytri_3_ :: proc(uplo: cstring, n: ^blasint, A: ^complex64, lda: ^blasint, E: ^complex64, ipiv: ^blasint, work: ^complex64, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	dsytri_3_ :: proc(uplo: cstring, n: ^blasint, A: ^f64, lda: ^blasint, E: ^f64, ipiv: ^blasint, work: ^f64, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	ssytri_3_ :: proc(uplo: cstring, n: ^blasint, A: ^f32, lda: ^blasint, E: ^f32, ipiv: ^blasint, work: ^f32, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	zsytri_3_ :: proc(uplo: cstring, n: ^blasint, A: ^complex128, lda: ^blasint, E: ^complex128, ipiv: ^blasint, work: ^complex128, lwork: ^blasint, info: ^Info, _: c.size_t) ---

	csytrs_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex64, lda: ^blasint, ipiv: ^blasint, B: ^complex64, ldb: ^blasint, info: ^Info, _: c.size_t) ---
	dsytrs_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^f64, lda: ^blasint, ipiv: ^blasint, B: ^f64, ldb: ^blasint, info: ^Info, _: c.size_t) ---
	ssytrs_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^f32, lda: ^blasint, ipiv: ^blasint, B: ^f32, ldb: ^blasint, info: ^Info, _: c.size_t) ---
	zsytrs_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex128, lda: ^blasint, ipiv: ^blasint, B: ^complex128, ldb: ^blasint, info: ^Info, _: c.size_t) ---

	csytrs2_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex64, lda: ^blasint, ipiv: ^blasint, B: ^complex64, ldb: ^blasint, work: ^complex64, info: ^Info, _: c.size_t) ---
	dsytrs2_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^f64, lda: ^blasint, ipiv: ^blasint, B: ^f64, ldb: ^blasint, work: ^f64, info: ^Info, _: c.size_t) ---
	ssytrs2_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^f32, lda: ^blasint, ipiv: ^blasint, B: ^f32, ldb: ^blasint, work: ^f32, info: ^Info, _: c.size_t) ---
	zsytrs2_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex128, lda: ^blasint, ipiv: ^blasint, B: ^complex128, ldb: ^blasint, work: ^complex128, info: ^Info, _: c.size_t) ---

	csytrs_3_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex64, lda: ^blasint, E: ^complex64, ipiv: ^blasint, B: ^complex64, ldb: ^blasint, info: ^Info, _: c.size_t) ---
	dsytrs_3_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^f64, lda: ^blasint, E: ^f64, ipiv: ^blasint, B: ^f64, ldb: ^blasint, info: ^Info, _: c.size_t) ---
	ssytrs_3_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^f32, lda: ^blasint, E: ^f32, ipiv: ^blasint, B: ^f32, ldb: ^blasint, info: ^Info, _: c.size_t) ---
	zsytrs_3_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex128, lda: ^blasint, E: ^complex128, ipiv: ^blasint, B: ^complex128, ldb: ^blasint, info: ^Info, _: c.size_t) ---

	csytrs_aa_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex64, lda: ^blasint, ipiv: ^blasint, B: ^complex64, ldb: ^blasint, work: ^complex64, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	dsytrs_aa_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^f64, lda: ^blasint, ipiv: ^blasint, B: ^f64, ldb: ^blasint, work: ^f64, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	ssytrs_aa_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^f32, lda: ^blasint, ipiv: ^blasint, B: ^f32, ldb: ^blasint, work: ^f32, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	zsytrs_aa_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex128, lda: ^blasint, ipiv: ^blasint, B: ^complex128, ldb: ^blasint, work: ^complex128, lwork: ^blasint, info: ^Info, _: c.size_t) ---

	csytrs_aa_2stage_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex64, lda: ^blasint, TB: ^complex64, ltb: ^blasint, ipiv: ^blasint, ipiv2: ^blasint, B: ^complex64, ldb: ^blasint, info: ^Info, _: c.size_t) ---
	dsytrs_aa_2stage_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^f64, lda: ^blasint, TB: ^f64, ltb: ^blasint, ipiv: ^blasint, ipiv2: ^blasint, B: ^f64, ldb: ^blasint, info: ^Info, _: c.size_t) ---
	ssytrs_aa_2stage_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^f32, lda: ^blasint, TB: ^f32, ltb: ^blasint, ipiv: ^blasint, ipiv2: ^blasint, B: ^f32, ldb: ^blasint, info: ^Info, _: c.size_t) ---
	zsytrs_aa_2stage_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex128, lda: ^blasint, TB: ^complex128, ltb: ^blasint, ipiv: ^blasint, ipiv2: ^blasint, B: ^complex128, ldb: ^blasint, info: ^Info, _: c.size_t) ---

	csytrs_rook_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex64, lda: ^blasint, ipiv: ^blasint, B: ^complex64, ldb: ^blasint, info: ^Info, _: c.size_t) ---
	dsytrs_rook_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^f64, lda: ^blasint, ipiv: ^blasint, B: ^f64, ldb: ^blasint, info: ^Info, _: c.size_t) ---
	ssytrs_rook_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^f32, lda: ^blasint, ipiv: ^blasint, B: ^f32, ldb: ^blasint, info: ^Info, _: c.size_t) ---
	zsytrs_rook_ :: proc(uplo: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex128, lda: ^blasint, ipiv: ^blasint, B: ^complex128, ldb: ^blasint, info: ^Info, _: c.size_t) ---

	ctbcon_ :: proc(norm: cstring, uplo: cstring, diag: cstring, n: ^blasint, kd: ^blasint, AB: ^complex64, ldab: ^blasint, rcond: ^f32, work: ^complex64, rwork: ^f32, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	dtbcon_ :: proc(norm: cstring, uplo: cstring, diag: cstring, n: ^blasint, kd: ^blasint, AB: ^f64, ldab: ^blasint, rcond: ^f64, work: ^f64, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	stbcon_ :: proc(norm: cstring, uplo: cstring, diag: cstring, n: ^blasint, kd: ^blasint, AB: ^f32, ldab: ^blasint, rcond: ^f32, work: ^f32, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	ztbcon_ :: proc(norm: cstring, uplo: cstring, diag: cstring, n: ^blasint, kd: ^blasint, AB: ^complex128, ldab: ^blasint, rcond: ^f64, work: ^complex128, rwork: ^f64, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---

	ctbrfs_ :: proc(uplo: cstring, trans: cstring, diag: cstring, n: ^blasint, kd: ^blasint, nrhs: ^blasint, AB: ^complex64, ldab: ^blasint, B: ^complex64, ldb: ^blasint, X: ^complex64, ldx: ^blasint, ferr: ^f32, berr: ^f32, work: ^complex64, rwork: ^f32, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	dtbrfs_ :: proc(uplo: cstring, trans: cstring, diag: cstring, n: ^blasint, kd: ^blasint, nrhs: ^blasint, AB: ^f64, ldab: ^blasint, B: ^f64, ldb: ^blasint, X: ^f64, ldx: ^blasint, ferr: ^f64, berr: ^f64, work: ^f64, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	stbrfs_ :: proc(uplo: cstring, trans: cstring, diag: cstring, n: ^blasint, kd: ^blasint, nrhs: ^blasint, AB: ^f32, ldab: ^blasint, B: ^f32, ldb: ^blasint, X: ^f32, ldx: ^blasint, ferr: ^f32, berr: ^f32, work: ^f32, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	ztbrfs_ :: proc(uplo: cstring, trans: cstring, diag: cstring, n: ^blasint, kd: ^blasint, nrhs: ^blasint, AB: ^complex128, ldab: ^blasint, B: ^complex128, ldb: ^blasint, X: ^complex128, ldx: ^blasint, ferr: ^f64, berr: ^f64, work: ^complex128, rwork: ^f64, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---

	ctbtrs_ :: proc(uplo: cstring, trans: cstring, diag: cstring, n: ^blasint, kd: ^blasint, nrhs: ^blasint, AB: ^complex64, ldab: ^blasint, B: ^complex64, ldb: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	dtbtrs_ :: proc(uplo: cstring, trans: cstring, diag: cstring, n: ^blasint, kd: ^blasint, nrhs: ^blasint, AB: ^f64, ldab: ^blasint, B: ^f64, ldb: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	stbtrs_ :: proc(uplo: cstring, trans: cstring, diag: cstring, n: ^blasint, kd: ^blasint, nrhs: ^blasint, AB: ^f32, ldab: ^blasint, B: ^f32, ldb: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	ztbtrs_ :: proc(uplo: cstring, trans: cstring, diag: cstring, n: ^blasint, kd: ^blasint, nrhs: ^blasint, AB: ^complex128, ldab: ^blasint, B: ^complex128, ldb: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---

	ctfsm_ :: proc(transr: cstring, side: cstring, uplo: cstring, trans: cstring, diag: cstring, m: ^blasint, n: ^blasint, alpha: ^complex64, A: ^complex64, B: ^complex64, ldb: ^blasint, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t) ---
	dtfsm_ :: proc(transr: cstring, side: cstring, uplo: cstring, trans: cstring, diag: cstring, m: ^blasint, n: ^blasint, alpha: ^f64, A: ^f64, B: ^f64, ldb: ^blasint, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t) ---
	stfsm_ :: proc(transr: cstring, side: cstring, uplo: cstring, trans: cstring, diag: cstring, m: ^blasint, n: ^blasint, alpha: ^f32, A: ^f32, B: ^f32, ldb: ^blasint, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t) ---
	ztfsm_ :: proc(transr: cstring, side: cstring, uplo: cstring, trans: cstring, diag: cstring, m: ^blasint, n: ^blasint, alpha: ^complex128, A: ^complex128, B: ^complex128, ldb: ^blasint, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t) ---

	ctftri_ :: proc(transr: cstring, uplo: cstring, diag: cstring, n: ^blasint, A: ^complex64, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	dtftri_ :: proc(transr: cstring, uplo: cstring, diag: cstring, n: ^blasint, A: ^f64, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	stftri_ :: proc(transr: cstring, uplo: cstring, diag: cstring, n: ^blasint, A: ^f32, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	ztftri_ :: proc(transr: cstring, uplo: cstring, diag: cstring, n: ^blasint, A: ^complex128, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---

	ctfttp_ :: proc(transr: cstring, uplo: cstring, n: ^blasint, ARF: ^complex64, AP: ^complex64, info: ^Info, _: c.size_t, _: c.size_t) ---
	dtfttp_ :: proc(transr: cstring, uplo: cstring, n: ^blasint, ARF: ^f64, AP: ^f64, info: ^Info, _: c.size_t, _: c.size_t) ---
	stfttp_ :: proc(transr: cstring, uplo: cstring, n: ^blasint, ARF: ^f32, AP: ^f32, info: ^Info, _: c.size_t, _: c.size_t) ---
	ztfttp_ :: proc(transr: cstring, uplo: cstring, n: ^blasint, ARF: ^complex128, AP: ^complex128, info: ^Info, _: c.size_t, _: c.size_t) ---

	ctfttr_ :: proc(transr: cstring, uplo: cstring, n: ^blasint, ARF: ^complex64, A: ^complex64, lda: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	dtfttr_ :: proc(transr: cstring, uplo: cstring, n: ^blasint, ARF: ^f64, A: ^f64, lda: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	stfttr_ :: proc(transr: cstring, uplo: cstring, n: ^blasint, ARF: ^f32, A: ^f32, lda: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	ztfttr_ :: proc(transr: cstring, uplo: cstring, n: ^blasint, ARF: ^complex128, A: ^complex128, lda: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---

	ctgevc_ :: proc(side: cstring, howmny: cstring, select: ^blasint, n: ^blasint, S: ^complex64, lds: ^blasint, P: ^complex64, ldp: ^blasint, VL: ^complex64, ldvl: ^blasint, VR: ^complex64, ldvr: ^blasint, mm: ^blasint, m: ^blasint, work: ^complex64, rwork: ^f32, info: ^Info, _: c.size_t, _: c.size_t) ---
	dtgevc_ :: proc(side: cstring, howmny: cstring, select: ^blasint, n: ^blasint, S: ^f64, lds: ^blasint, P: ^f64, ldp: ^blasint, VL: ^f64, ldvl: ^blasint, VR: ^f64, ldvr: ^blasint, mm: ^blasint, m: ^blasint, work: ^f64, info: ^Info, _: c.size_t, _: c.size_t) ---
	stgevc_ :: proc(side: cstring, howmny: cstring, select: ^blasint, n: ^blasint, S: ^f32, lds: ^blasint, P: ^f32, ldp: ^blasint, VL: ^f32, ldvl: ^blasint, VR: ^f32, ldvr: ^blasint, mm: ^blasint, m: ^blasint, work: ^f32, info: ^Info, _: c.size_t, _: c.size_t) ---
	ztgevc_ :: proc(side: cstring, howmny: cstring, select: ^blasint, n: ^blasint, S: ^complex128, lds: ^blasint, P: ^complex128, ldp: ^blasint, VL: ^complex128, ldvl: ^blasint, VR: ^complex128, ldvr: ^blasint, mm: ^blasint, m: ^blasint, work: ^complex128, rwork: ^f64, info: ^Info, _: c.size_t, _: c.size_t) ---

	ctgexc_ :: proc(wantq: ^blasint, wantz: ^blasint, n: ^blasint, A: ^complex64, lda: ^blasint, B: ^complex64, ldb: ^blasint, Q: ^complex64, ldq: ^blasint, Z: ^complex64, ldz: ^blasint, ifst: ^blasint, ilst: ^blasint, info: ^Info) ---
	dtgexc_ :: proc(wantq: ^blasint, wantz: ^blasint, n: ^blasint, A: ^f64, lda: ^blasint, B: ^f64, ldb: ^blasint, Q: ^f64, ldq: ^blasint, Z: ^f64, ldz: ^blasint, ifst: ^blasint, ilst: ^blasint, work: ^f64, lwork: ^blasint, info: ^Info) ---
	stgexc_ :: proc(wantq: ^blasint, wantz: ^blasint, n: ^blasint, A: ^f32, lda: ^blasint, B: ^f32, ldb: ^blasint, Q: ^f32, ldq: ^blasint, Z: ^f32, ldz: ^blasint, ifst: ^blasint, ilst: ^blasint, work: ^f32, lwork: ^blasint, info: ^Info) ---
	ztgexc_ :: proc(wantq: ^blasint, wantz: ^blasint, n: ^blasint, A: ^complex128, lda: ^blasint, B: ^complex128, ldb: ^blasint, Q: ^complex128, ldq: ^blasint, Z: ^complex128, ldz: ^blasint, ifst: ^blasint, ilst: ^blasint, info: ^Info) ---

	ctgsen_ :: proc(ijob: ^blasint, wantq: ^blasint, wantz: ^blasint, select: ^blasint, n: ^blasint, A: ^complex64, lda: ^blasint, B: ^complex64, ldb: ^blasint, alpha: ^complex64, beta: ^complex64, Q: ^complex64, ldq: ^blasint, Z: ^complex64, ldz: ^blasint, m: ^blasint, pl: ^f32, pr: ^f32, DIF: ^f32, work: ^complex64, lwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info) ---
	dtgsen_ :: proc(ijob: ^blasint, wantq: ^blasint, wantz: ^blasint, select: ^blasint, n: ^blasint, A: ^f64, lda: ^blasint, B: ^f64, ldb: ^blasint, alphar: ^f64, alphai: ^f64, beta: ^f64, Q: ^f64, ldq: ^blasint, Z: ^f64, ldz: ^blasint, m: ^blasint, pl: ^f64, pr: ^f64, DIF: ^f64, work: ^f64, lwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info) ---
	stgsen_ :: proc(ijob: ^blasint, wantq: ^blasint, wantz: ^blasint, select: ^blasint, n: ^blasint, A: ^f32, lda: ^blasint, B: ^f32, ldb: ^blasint, alphar: ^f32, alphai: ^f32, beta: ^f32, Q: ^f32, ldq: ^blasint, Z: ^f32, ldz: ^blasint, m: ^blasint, pl: ^f32, pr: ^f32, DIF: ^f32, work: ^f32, lwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info) ---
	ztgsen_ :: proc(ijob: ^blasint, wantq: ^blasint, wantz: ^blasint, select: ^blasint, n: ^blasint, A: ^complex128, lda: ^blasint, B: ^complex128, ldb: ^blasint, alpha: ^complex128, beta: ^complex128, Q: ^complex128, ldq: ^blasint, Z: ^complex128, ldz: ^blasint, m: ^blasint, pl: ^f64, pr: ^f64, DIF: ^f64, work: ^complex128, lwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info) ---

	ctgsja_ :: proc(jobu: cstring, jobv: cstring, jobq: cstring, m: ^blasint, p: ^blasint, n: ^blasint, k: ^blasint, l: ^blasint, A: ^complex64, lda: ^blasint, B: ^complex64, ldb: ^blasint, tola: ^f32, tolb: ^f32, alpha: ^f32, beta: ^f32, U: ^complex64, ldu: ^blasint, V: ^complex64, ldv: ^blasint, Q: ^complex64, ldq: ^blasint, work: ^complex64, ncycle: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	dtgsja_ :: proc(jobu: cstring, jobv: cstring, jobq: cstring, m: ^blasint, p: ^blasint, n: ^blasint, k: ^blasint, l: ^blasint, A: ^f64, lda: ^blasint, B: ^f64, ldb: ^blasint, tola: ^f64, tolb: ^f64, alpha: ^f64, beta: ^f64, U: ^f64, ldu: ^blasint, V: ^f64, ldv: ^blasint, Q: ^f64, ldq: ^blasint, work: ^f64, ncycle: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	stgsja_ :: proc(jobu: cstring, jobv: cstring, jobq: cstring, m: ^blasint, p: ^blasint, n: ^blasint, k: ^blasint, l: ^blasint, A: ^f32, lda: ^blasint, B: ^f32, ldb: ^blasint, tola: ^f32, tolb: ^f32, alpha: ^f32, beta: ^f32, U: ^f32, ldu: ^blasint, V: ^f32, ldv: ^blasint, Q: ^f32, ldq: ^blasint, work: ^f32, ncycle: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	ztgsja_ :: proc(jobu: cstring, jobv: cstring, jobq: cstring, m: ^blasint, p: ^blasint, n: ^blasint, k: ^blasint, l: ^blasint, A: ^complex128, lda: ^blasint, B: ^complex128, ldb: ^blasint, tola: ^f64, tolb: ^f64, alpha: ^f64, beta: ^f64, U: ^complex128, ldu: ^blasint, V: ^complex128, ldv: ^blasint, Q: ^complex128, ldq: ^blasint, work: ^complex128, ncycle: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---

	ctgsna_ :: proc(job: cstring, howmny: cstring, select: ^blasint, n: ^blasint, A: ^complex64, lda: ^blasint, B: ^complex64, ldb: ^blasint, VL: ^complex64, ldvl: ^blasint, VR: ^complex64, ldvr: ^blasint, S: ^f32, DIF: ^f32, mm: ^blasint, m: ^blasint, work: ^complex64, lwork: ^blasint, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	dtgsna_ :: proc(job: cstring, howmny: cstring, select: ^blasint, n: ^blasint, A: ^f64, lda: ^blasint, B: ^f64, ldb: ^blasint, VL: ^f64, ldvl: ^blasint, VR: ^f64, ldvr: ^blasint, S: ^f64, DIF: ^f64, mm: ^blasint, m: ^blasint, work: ^f64, lwork: ^blasint, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	stgsna_ :: proc(job: cstring, howmny: cstring, select: ^blasint, n: ^blasint, A: ^f32, lda: ^blasint, B: ^f32, ldb: ^blasint, VL: ^f32, ldvl: ^blasint, VR: ^f32, ldvr: ^blasint, S: ^f32, DIF: ^f32, mm: ^blasint, m: ^blasint, work: ^f32, lwork: ^blasint, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	ztgsna_ :: proc(job: cstring, howmny: cstring, select: ^blasint, n: ^blasint, A: ^complex128, lda: ^blasint, B: ^complex128, ldb: ^blasint, VL: ^complex128, ldvl: ^blasint, VR: ^complex128, ldvr: ^blasint, S: ^f64, DIF: ^f64, mm: ^blasint, m: ^blasint, work: ^complex128, lwork: ^blasint, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---

	ctgsyl_ :: proc(trans: cstring, ijob: ^blasint, m: ^blasint, n: ^blasint, A: ^complex64, lda: ^blasint, B: ^complex64, ldb: ^blasint, C: ^complex64, ldc: ^blasint, D: ^complex64, ldd: ^blasint, E: ^complex64, lde: ^blasint, F: ^complex64, ldf: ^blasint, dif: ^f32, scale: ^f32, work: ^complex64, lwork: ^blasint, iwork: ^blasint, info: ^Info, _: c.size_t) ---
	dtgsyl_ :: proc(trans: cstring, ijob: ^blasint, m: ^blasint, n: ^blasint, A: ^f64, lda: ^blasint, B: ^f64, ldb: ^blasint, C: ^f64, ldc: ^blasint, D: ^f64, ldd: ^blasint, E: ^f64, lde: ^blasint, F: ^f64, ldf: ^blasint, dif: ^f64, scale: ^f64, work: ^f64, lwork: ^blasint, iwork: ^blasint, info: ^Info, _: c.size_t) ---
	stgsyl_ :: proc(trans: cstring, ijob: ^blasint, m: ^blasint, n: ^blasint, A: ^f32, lda: ^blasint, B: ^f32, ldb: ^blasint, C: ^f32, ldc: ^blasint, D: ^f32, ldd: ^blasint, E: ^f32, lde: ^blasint, F: ^f32, ldf: ^blasint, dif: ^f32, scale: ^f32, work: ^f32, lwork: ^blasint, iwork: ^blasint, info: ^Info, _: c.size_t) ---
	ztgsyl_ :: proc(trans: cstring, ijob: ^blasint, m: ^blasint, n: ^blasint, A: ^complex128, lda: ^blasint, B: ^complex128, ldb: ^blasint, C: ^complex128, ldc: ^blasint, D: ^complex128, ldd: ^blasint, E: ^complex128, lde: ^blasint, F: ^complex128, ldf: ^blasint, dif: ^f64, scale: ^f64, work: ^complex128, lwork: ^blasint, iwork: ^blasint, info: ^Info, _: c.size_t) ---

	ctpcon_ :: proc(norm: cstring, uplo: cstring, diag: cstring, n: ^blasint, AP: ^complex64, rcond: ^f32, work: ^complex64, rwork: ^f32, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	dtpcon_ :: proc(norm: cstring, uplo: cstring, diag: cstring, n: ^blasint, AP: ^f64, rcond: ^f64, work: ^f64, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	stpcon_ :: proc(norm: cstring, uplo: cstring, diag: cstring, n: ^blasint, AP: ^f32, rcond: ^f32, work: ^f32, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	ztpcon_ :: proc(norm: cstring, uplo: cstring, diag: cstring, n: ^blasint, AP: ^complex128, rcond: ^f64, work: ^complex128, rwork: ^f64, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---

	ctplqt_ :: proc(m: ^blasint, n: ^blasint, l: ^blasint, mb: ^blasint, A: ^complex64, lda: ^blasint, B: ^complex64, ldb: ^blasint, T: ^complex64, ldt: ^blasint, work: ^complex64, info: ^Info) ---
	dtplqt_ :: proc(m: ^blasint, n: ^blasint, l: ^blasint, mb: ^blasint, A: ^f64, lda: ^blasint, B: ^f64, ldb: ^blasint, T: ^f64, ldt: ^blasint, work: ^f64, info: ^Info) ---
	stplqt_ :: proc(m: ^blasint, n: ^blasint, l: ^blasint, mb: ^blasint, A: ^f32, lda: ^blasint, B: ^f32, ldb: ^blasint, T: ^f32, ldt: ^blasint, work: ^f32, info: ^Info) ---
	ztplqt_ :: proc(m: ^blasint, n: ^blasint, l: ^blasint, mb: ^blasint, A: ^complex128, lda: ^blasint, B: ^complex128, ldb: ^blasint, T: ^complex128, ldt: ^blasint, work: ^complex128, info: ^Info) ---

	ctplqt2_ :: proc(m: ^blasint, n: ^blasint, l: ^blasint, A: ^complex64, lda: ^blasint, B: ^complex64, ldb: ^blasint, T: ^complex64, ldt: ^blasint, info: ^Info) ---
	dtplqt2_ :: proc(m: ^blasint, n: ^blasint, l: ^blasint, A: ^f64, lda: ^blasint, B: ^f64, ldb: ^blasint, T: ^f64, ldt: ^blasint, info: ^Info) ---
	stplqt2_ :: proc(m: ^blasint, n: ^blasint, l: ^blasint, A: ^f32, lda: ^blasint, B: ^f32, ldb: ^blasint, T: ^f32, ldt: ^blasint, info: ^Info) ---
	ztplqt2_ :: proc(m: ^blasint, n: ^blasint, l: ^blasint, A: ^complex128, lda: ^blasint, B: ^complex128, ldb: ^blasint, T: ^complex128, ldt: ^blasint, info: ^Info) ---

	ctpmlqt_ :: proc(side: cstring, trans: cstring, m: ^blasint, n: ^blasint, k: ^blasint, l: ^blasint, mb: ^blasint, V: ^complex64, ldv: ^blasint, T: ^complex64, ldt: ^blasint, A: ^complex64, lda: ^blasint, B: ^complex64, ldb: ^blasint, work: ^complex64, info: ^Info, _: c.size_t, _: c.size_t) ---
	dtpmlqt_ :: proc(side: cstring, trans: cstring, m: ^blasint, n: ^blasint, k: ^blasint, l: ^blasint, mb: ^blasint, V: ^f64, ldv: ^blasint, T: ^f64, ldt: ^blasint, A: ^f64, lda: ^blasint, B: ^f64, ldb: ^blasint, work: ^f64, info: ^Info, _: c.size_t, _: c.size_t) ---
	stpmlqt_ :: proc(side: cstring, trans: cstring, m: ^blasint, n: ^blasint, k: ^blasint, l: ^blasint, mb: ^blasint, V: ^f32, ldv: ^blasint, T: ^f32, ldt: ^blasint, A: ^f32, lda: ^blasint, B: ^f32, ldb: ^blasint, work: ^f32, info: ^Info, _: c.size_t, _: c.size_t) ---
	ztpmlqt_ :: proc(side: cstring, trans: cstring, m: ^blasint, n: ^blasint, k: ^blasint, l: ^blasint, mb: ^blasint, V: ^complex128, ldv: ^blasint, T: ^complex128, ldt: ^blasint, A: ^complex128, lda: ^blasint, B: ^complex128, ldb: ^blasint, work: ^complex128, info: ^Info, _: c.size_t, _: c.size_t) ---

	ctpmqrt_ :: proc(side: cstring, trans: cstring, m: ^blasint, n: ^blasint, k: ^blasint, l: ^blasint, nb: ^blasint, V: ^complex64, ldv: ^blasint, T: ^complex64, ldt: ^blasint, A: ^complex64, lda: ^blasint, B: ^complex64, ldb: ^blasint, work: ^complex64, info: ^Info, _: c.size_t, _: c.size_t) ---
	dtpmqrt_ :: proc(side: cstring, trans: cstring, m: ^blasint, n: ^blasint, k: ^blasint, l: ^blasint, nb: ^blasint, V: ^f64, ldv: ^blasint, T: ^f64, ldt: ^blasint, A: ^f64, lda: ^blasint, B: ^f64, ldb: ^blasint, work: ^f64, info: ^Info, _: c.size_t, _: c.size_t) ---
	stpmqrt_ :: proc(side: cstring, trans: cstring, m: ^blasint, n: ^blasint, k: ^blasint, l: ^blasint, nb: ^blasint, V: ^f32, ldv: ^blasint, T: ^f32, ldt: ^blasint, A: ^f32, lda: ^blasint, B: ^f32, ldb: ^blasint, work: ^f32, info: ^Info, _: c.size_t, _: c.size_t) ---
	ztpmqrt_ :: proc(side: cstring, trans: cstring, m: ^blasint, n: ^blasint, k: ^blasint, l: ^blasint, nb: ^blasint, V: ^complex128, ldv: ^blasint, T: ^complex128, ldt: ^blasint, A: ^complex128, lda: ^blasint, B: ^complex128, ldb: ^blasint, work: ^complex128, info: ^Info, _: c.size_t, _: c.size_t) ---

	ctpqrt_ :: proc(m: ^blasint, n: ^blasint, l: ^blasint, nb: ^blasint, A: ^complex64, lda: ^blasint, B: ^complex64, ldb: ^blasint, T: ^complex64, ldt: ^blasint, work: ^complex64, info: ^Info) ---
	dtpqrt_ :: proc(m: ^blasint, n: ^blasint, l: ^blasint, nb: ^blasint, A: ^f64, lda: ^blasint, B: ^f64, ldb: ^blasint, T: ^f64, ldt: ^blasint, work: ^f64, info: ^Info) ---
	stpqrt_ :: proc(m: ^blasint, n: ^blasint, l: ^blasint, nb: ^blasint, A: ^f32, lda: ^blasint, B: ^f32, ldb: ^blasint, T: ^f32, ldt: ^blasint, work: ^f32, info: ^Info) ---
	ztpqrt_ :: proc(m: ^blasint, n: ^blasint, l: ^blasint, nb: ^blasint, A: ^complex128, lda: ^blasint, B: ^complex128, ldb: ^blasint, T: ^complex128, ldt: ^blasint, work: ^complex128, info: ^Info) ---

	ctpqrt2_ :: proc(m: ^blasint, n: ^blasint, l: ^blasint, A: ^complex64, lda: ^blasint, B: ^complex64, ldb: ^blasint, T: ^complex64, ldt: ^blasint, info: ^Info) ---
	dtpqrt2_ :: proc(m: ^blasint, n: ^blasint, l: ^blasint, A: ^f64, lda: ^blasint, B: ^f64, ldb: ^blasint, T: ^f64, ldt: ^blasint, info: ^Info) ---
	stpqrt2_ :: proc(m: ^blasint, n: ^blasint, l: ^blasint, A: ^f32, lda: ^blasint, B: ^f32, ldb: ^blasint, T: ^f32, ldt: ^blasint, info: ^Info) ---
	ztpqrt2_ :: proc(m: ^blasint, n: ^blasint, l: ^blasint, A: ^complex128, lda: ^blasint, B: ^complex128, ldb: ^blasint, T: ^complex128, ldt: ^blasint, info: ^Info) ---

	ctprfb_ :: proc(side: cstring, trans: cstring, direct: cstring, storev: cstring, m: ^blasint, n: ^blasint, k: ^blasint, l: ^blasint, V: ^complex64, ldv: ^blasint, T: ^complex64, ldt: ^blasint, A: ^complex64, lda: ^blasint, B: ^complex64, ldb: ^blasint, work: ^complex64, ldwork: ^blasint, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t) ---
	dtprfb_ :: proc(side: cstring, trans: cstring, direct: cstring, storev: cstring, m: ^blasint, n: ^blasint, k: ^blasint, l: ^blasint, V: ^f64, ldv: ^blasint, T: ^f64, ldt: ^blasint, A: ^f64, lda: ^blasint, B: ^f64, ldb: ^blasint, work: ^f64, ldwork: ^blasint, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t) ---
	stprfb_ :: proc(side: cstring, trans: cstring, direct: cstring, storev: cstring, m: ^blasint, n: ^blasint, k: ^blasint, l: ^blasint, V: ^f32, ldv: ^blasint, T: ^f32, ldt: ^blasint, A: ^f32, lda: ^blasint, B: ^f32, ldb: ^blasint, work: ^f32, ldwork: ^blasint, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t) ---
	ztprfb_ :: proc(side: cstring, trans: cstring, direct: cstring, storev: cstring, m: ^blasint, n: ^blasint, k: ^blasint, l: ^blasint, V: ^complex128, ldv: ^blasint, T: ^complex128, ldt: ^blasint, A: ^complex128, lda: ^blasint, B: ^complex128, ldb: ^blasint, work: ^complex128, ldwork: ^blasint, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t) ---

	ctprfs_ :: proc(uplo: cstring, trans: cstring, diag: cstring, n: ^blasint, nrhs: ^blasint, AP: ^complex64, B: ^complex64, ldb: ^blasint, X: ^complex64, ldx: ^blasint, ferr: ^f32, berr: ^f32, work: ^complex64, rwork: ^f32, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	dtprfs_ :: proc(uplo: cstring, trans: cstring, diag: cstring, n: ^blasint, nrhs: ^blasint, AP: ^f64, B: ^f64, ldb: ^blasint, X: ^f64, ldx: ^blasint, ferr: ^f64, berr: ^f64, work: ^f64, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	stprfs_ :: proc(uplo: cstring, trans: cstring, diag: cstring, n: ^blasint, nrhs: ^blasint, AP: ^f32, B: ^f32, ldb: ^blasint, X: ^f32, ldx: ^blasint, ferr: ^f32, berr: ^f32, work: ^f32, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	ztprfs_ :: proc(uplo: cstring, trans: cstring, diag: cstring, n: ^blasint, nrhs: ^blasint, AP: ^complex128, B: ^complex128, ldb: ^blasint, X: ^complex128, ldx: ^blasint, ferr: ^f64, berr: ^f64, work: ^complex128, rwork: ^f64, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---

	ctptri_ :: proc(uplo: cstring, diag: cstring, n: ^blasint, AP: ^complex64, info: ^Info, _: c.size_t, _: c.size_t) ---
	dtptri_ :: proc(uplo: cstring, diag: cstring, n: ^blasint, AP: ^f64, info: ^Info, _: c.size_t, _: c.size_t) ---
	stptri_ :: proc(uplo: cstring, diag: cstring, n: ^blasint, AP: ^f32, info: ^Info, _: c.size_t, _: c.size_t) ---
	ztptri_ :: proc(uplo: cstring, diag: cstring, n: ^blasint, AP: ^complex128, info: ^Info, _: c.size_t, _: c.size_t) ---

	ctptrs_ :: proc(uplo: cstring, trans: cstring, diag: cstring, n: ^blasint, nrhs: ^blasint, AP: ^complex64, B: ^complex64, ldb: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	dtptrs_ :: proc(uplo: cstring, trans: cstring, diag: cstring, n: ^blasint, nrhs: ^blasint, AP: ^f64, B: ^f64, ldb: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	stptrs_ :: proc(uplo: cstring, trans: cstring, diag: cstring, n: ^blasint, nrhs: ^blasint, AP: ^f32, B: ^f32, ldb: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	ztptrs_ :: proc(uplo: cstring, trans: cstring, diag: cstring, n: ^blasint, nrhs: ^blasint, AP: ^complex128, B: ^complex128, ldb: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---

	ctpttf_ :: proc(transr: cstring, uplo: cstring, n: ^blasint, AP: ^complex64, ARF: ^complex64, info: ^Info, _: c.size_t, _: c.size_t) ---
	dtpttf_ :: proc(transr: cstring, uplo: cstring, n: ^blasint, AP: ^f64, ARF: ^f64, info: ^Info, _: c.size_t, _: c.size_t) ---
	stpttf_ :: proc(transr: cstring, uplo: cstring, n: ^blasint, AP: ^f32, ARF: ^f32, info: ^Info, _: c.size_t, _: c.size_t) ---
	ztpttf_ :: proc(transr: cstring, uplo: cstring, n: ^blasint, AP: ^complex128, ARF: ^complex128, info: ^Info, _: c.size_t, _: c.size_t) ---

	ctpttr_ :: proc(uplo: cstring, n: ^blasint, AP: ^complex64, A: ^complex64, lda: ^blasint, info: ^Info, _: c.size_t) ---
	dtpttr_ :: proc(uplo: cstring, n: ^blasint, AP: ^f64, A: ^f64, lda: ^blasint, info: ^Info, _: c.size_t) ---
	stpttr_ :: proc(uplo: cstring, n: ^blasint, AP: ^f32, A: ^f32, lda: ^blasint, info: ^Info, _: c.size_t) ---
	ztpttr_ :: proc(uplo: cstring, n: ^blasint, AP: ^complex128, A: ^complex128, lda: ^blasint, info: ^Info, _: c.size_t) ---

	ctrcon_ :: proc(norm: cstring, uplo: cstring, diag: cstring, n: ^blasint, A: ^complex64, lda: ^blasint, rcond: ^f32, work: ^complex64, rwork: ^f32, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	dtrcon_ :: proc(norm: cstring, uplo: cstring, diag: cstring, n: ^blasint, A: ^f64, lda: ^blasint, rcond: ^f64, work: ^f64, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	strcon_ :: proc(norm: cstring, uplo: cstring, diag: cstring, n: ^blasint, A: ^f32, lda: ^blasint, rcond: ^f32, work: ^f32, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	ztrcon_ :: proc(norm: cstring, uplo: cstring, diag: cstring, n: ^blasint, A: ^complex128, lda: ^blasint, rcond: ^f64, work: ^complex128, rwork: ^f64, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---

	ctrevc_ :: proc(side: cstring, howmny: cstring, select: ^blasint, n: ^blasint, T: ^complex64, ldt: ^blasint, VL: ^complex64, ldvl: ^blasint, VR: ^complex64, ldvr: ^blasint, mm: ^blasint, m: ^blasint, work: ^complex64, rwork: ^f32, info: ^Info, _: c.size_t, _: c.size_t) ---
	dtrevc_ :: proc(side: cstring, howmny: cstring, select: ^blasint, n: ^blasint, T: ^f64, ldt: ^blasint, VL: ^f64, ldvl: ^blasint, VR: ^f64, ldvr: ^blasint, mm: ^blasint, m: ^blasint, work: ^f64, info: ^Info, _: c.size_t, _: c.size_t) ---
	strevc_ :: proc(side: cstring, howmny: cstring, select: ^blasint, n: ^blasint, T: ^f32, ldt: ^blasint, VL: ^f32, ldvl: ^blasint, VR: ^f32, ldvr: ^blasint, mm: ^blasint, m: ^blasint, work: ^f32, info: ^Info, _: c.size_t, _: c.size_t) ---
	ztrevc_ :: proc(side: cstring, howmny: cstring, select: ^blasint, n: ^blasint, T: ^complex128, ldt: ^blasint, VL: ^complex128, ldvl: ^blasint, VR: ^complex128, ldvr: ^blasint, mm: ^blasint, m: ^blasint, work: ^complex128, rwork: ^f64, info: ^Info, _: c.size_t, _: c.size_t) ---

	ctrevc3_ :: proc(side: cstring, howmny: cstring, select: ^blasint, n: ^blasint, T: ^complex64, ldt: ^blasint, VL: ^complex64, ldvl: ^blasint, VR: ^complex64, ldvr: ^blasint, mm: ^blasint, m: ^blasint, work: ^complex64, lwork: ^blasint, rwork: ^f32, lrwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	dtrevc3_ :: proc(side: cstring, howmny: cstring, select: ^blasint, n: ^blasint, T: ^f64, ldt: ^blasint, VL: ^f64, ldvl: ^blasint, VR: ^f64, ldvr: ^blasint, mm: ^blasint, m: ^blasint, work: ^f64, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	strevc3_ :: proc(side: cstring, howmny: cstring, select: ^blasint, n: ^blasint, T: ^f32, ldt: ^blasint, VL: ^f32, ldvl: ^blasint, VR: ^f32, ldvr: ^blasint, mm: ^blasint, m: ^blasint, work: ^f32, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	ztrevc3_ :: proc(side: cstring, howmny: cstring, select: ^blasint, n: ^blasint, T: ^complex128, ldt: ^blasint, VL: ^complex128, ldvl: ^blasint, VR: ^complex128, ldvr: ^blasint, mm: ^blasint, m: ^blasint, work: ^complex128, lwork: ^blasint, rwork: ^f64, lrwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---

	ctrexc_ :: proc(compq: cstring, n: ^blasint, T: ^complex64, ldt: ^blasint, Q: ^complex64, ldq: ^blasint, ifst: ^blasint, ilst: ^blasint, info: ^Info, _: c.size_t) ---
	dtrexc_ :: proc(compq: cstring, n: ^blasint, T: ^f64, ldt: ^blasint, Q: ^f64, ldq: ^blasint, ifst: ^blasint, ilst: ^blasint, work: ^f64, info: ^Info, _: c.size_t) ---
	strexc_ :: proc(compq: cstring, n: ^blasint, T: ^f32, ldt: ^blasint, Q: ^f32, ldq: ^blasint, ifst: ^blasint, ilst: ^blasint, work: ^f32, info: ^Info, _: c.size_t) ---
	ztrexc_ :: proc(compq: cstring, n: ^blasint, T: ^complex128, ldt: ^blasint, Q: ^complex128, ldq: ^blasint, ifst: ^blasint, ilst: ^blasint, info: ^Info, _: c.size_t) ---

	ctrrfs_ :: proc(uplo: cstring, trans: cstring, diag: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex64, lda: ^blasint, B: ^complex64, ldb: ^blasint, X: ^complex64, ldx: ^blasint, ferr: ^f32, berr: ^f32, work: ^complex64, rwork: ^f32, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	dtrrfs_ :: proc(uplo: cstring, trans: cstring, diag: cstring, n: ^blasint, nrhs: ^blasint, A: ^f64, lda: ^blasint, B: ^f64, ldb: ^blasint, X: ^f64, ldx: ^blasint, ferr: ^f64, berr: ^f64, work: ^f64, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	strrfs_ :: proc(uplo: cstring, trans: cstring, diag: cstring, n: ^blasint, nrhs: ^blasint, A: ^f32, lda: ^blasint, B: ^f32, ldb: ^blasint, X: ^f32, ldx: ^blasint, ferr: ^f32, berr: ^f32, work: ^f32, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	ztrrfs_ :: proc(uplo: cstring, trans: cstring, diag: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex128, lda: ^blasint, B: ^complex128, ldb: ^blasint, X: ^complex128, ldx: ^blasint, ferr: ^f64, berr: ^f64, work: ^complex128, rwork: ^f64, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---

	ctrsen_ :: proc(job: cstring, compq: cstring, select: ^blasint, n: ^blasint, T: ^complex64, ldt: ^blasint, Q: ^complex64, ldq: ^blasint, W: ^complex64, m: ^blasint, s: ^f32, sep: ^f32, work: ^complex64, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	dtrsen_ :: proc(job: cstring, compq: cstring, select: ^blasint, n: ^blasint, T: ^f64, ldt: ^blasint, Q: ^f64, ldq: ^blasint, WR: ^f64, WI: ^f64, m: ^blasint, s: ^f64, sep: ^f64, work: ^f64, lwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	strsen_ :: proc(job: cstring, compq: cstring, select: ^blasint, n: ^blasint, T: ^f32, ldt: ^blasint, Q: ^f32, ldq: ^blasint, WR: ^f32, WI: ^f32, m: ^blasint, s: ^f32, sep: ^f32, work: ^f32, lwork: ^blasint, iwork: ^blasint, liwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	ztrsen_ :: proc(job: cstring, compq: cstring, select: ^blasint, n: ^blasint, T: ^complex128, ldt: ^blasint, Q: ^complex128, ldq: ^blasint, W: ^complex128, m: ^blasint, s: ^f64, sep: ^f64, work: ^complex128, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---

	ctrsna_ :: proc(job: cstring, howmny: cstring, select: ^blasint, n: ^blasint, T: ^complex64, ldt: ^blasint, VL: ^complex64, ldvl: ^blasint, VR: ^complex64, ldvr: ^blasint, S: ^f32, SEP: ^f32, mm: ^blasint, m: ^blasint, work: ^complex64, ldwork: ^blasint, rwork: ^f32, info: ^Info, _: c.size_t, _: c.size_t) ---
	dtrsna_ :: proc(job: cstring, howmny: cstring, select: ^blasint, n: ^blasint, T: ^f64, ldt: ^blasint, VL: ^f64, ldvl: ^blasint, VR: ^f64, ldvr: ^blasint, S: ^f64, SEP: ^f64, mm: ^blasint, m: ^blasint, work: ^f64, ldwork: ^blasint, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	strsna_ :: proc(job: cstring, howmny: cstring, select: ^blasint, n: ^blasint, T: ^f32, ldt: ^blasint, VL: ^f32, ldvl: ^blasint, VR: ^f32, ldvr: ^blasint, S: ^f32, SEP: ^f32, mm: ^blasint, m: ^blasint, work: ^f32, ldwork: ^blasint, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	ztrsna_ :: proc(job: cstring, howmny: cstring, select: ^blasint, n: ^blasint, T: ^complex128, ldt: ^blasint, VL: ^complex128, ldvl: ^blasint, VR: ^complex128, ldvr: ^blasint, S: ^f64, SEP: ^f64, mm: ^blasint, m: ^blasint, work: ^complex128, ldwork: ^blasint, rwork: ^f64, info: ^Info, _: c.size_t, _: c.size_t) ---

	ctrsyl_ :: proc(trana: cstring, tranb: cstring, isgn: ^blasint, m: ^blasint, n: ^blasint, A: ^complex64, lda: ^blasint, B: ^complex64, ldb: ^blasint, C: ^complex64, ldc: ^blasint, scale: ^f32, info: ^Info, _: c.size_t, _: c.size_t) ---
	dtrsyl_ :: proc(trana: cstring, tranb: cstring, isgn: ^blasint, m: ^blasint, n: ^blasint, A: ^f64, lda: ^blasint, B: ^f64, ldb: ^blasint, C: ^f64, ldc: ^blasint, scale: ^f64, info: ^Info, _: c.size_t, _: c.size_t) ---
	strsyl_ :: proc(trana: cstring, tranb: cstring, isgn: ^blasint, m: ^blasint, n: ^blasint, A: ^f32, lda: ^blasint, B: ^f32, ldb: ^blasint, C: ^f32, ldc: ^blasint, scale: ^f32, info: ^Info, _: c.size_t, _: c.size_t) ---
	ztrsyl_ :: proc(trana: cstring, tranb: cstring, isgn: ^blasint, m: ^blasint, n: ^blasint, A: ^complex128, lda: ^blasint, B: ^complex128, ldb: ^blasint, C: ^complex128, ldc: ^blasint, scale: ^f64, info: ^Info, _: c.size_t, _: c.size_t) ---

	ctrsyl3_ :: proc(trana: cstring, tranb: cstring, isgn: ^blasint, m: ^blasint, n: ^blasint, A: ^complex64, lda: ^blasint, B: ^complex64, ldb: ^blasint, C: ^complex64, ldc: ^blasint, scale: ^f32, swork: ^f32, ldswork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	dtrsyl3_ :: proc(trana: cstring, tranb: cstring, isgn: ^blasint, m: ^blasint, n: ^blasint, A: ^f64, lda: ^blasint, B: ^f64, ldb: ^blasint, C: ^f64, ldc: ^blasint, scale: ^f64, iwork: ^blasint, liwork: ^blasint, swork: ^f64, ldswork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	strsyl3_ :: proc(trana: cstring, tranb: cstring, isgn: ^blasint, m: ^blasint, n: ^blasint, A: ^f32, lda: ^blasint, B: ^f32, ldb: ^blasint, C: ^f32, ldc: ^blasint, scale: ^f32, iwork: ^blasint, liwork: ^blasint, swork: ^f32, ldswork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	ztrsyl3_ :: proc(trana: cstring, tranb: cstring, isgn: ^blasint, m: ^blasint, n: ^blasint, A: ^complex128, lda: ^blasint, B: ^complex128, ldb: ^blasint, C: ^complex128, ldc: ^blasint, scale: ^f64, swork: ^f64, ldswork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	// ctrtri_ :: proc(uplo: cstring, diag: cstring, n: ^blasint, A: ^complex64, lda: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) -> i32 ---
	// dtrtri_ :: proc(uplo: cstring, diag: cstring, n: ^blasint, A: ^f64, lda: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) -> i32 ---
	// strtri_ :: proc(uplo: cstring, diag: cstring, n: ^blasint, A: ^f32, lda: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) -> i32 ---
	// ztrtri_ :: proc(uplo: cstring, diag: cstring, n: ^blasint, A: ^complex128, lda: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) -> i32 ---
	ctrtrs_ :: proc(uplo: cstring, trans: cstring, diag: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex64, lda: ^blasint, B: ^complex64, ldb: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	dtrtrs_ :: proc(uplo: cstring, trans: cstring, diag: cstring, n: ^blasint, nrhs: ^blasint, A: ^f64, lda: ^blasint, B: ^f64, ldb: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	strtrs_ :: proc(uplo: cstring, trans: cstring, diag: cstring, n: ^blasint, nrhs: ^blasint, A: ^f32, lda: ^blasint, B: ^f32, ldb: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	ztrtrs_ :: proc(uplo: cstring, trans: cstring, diag: cstring, n: ^blasint, nrhs: ^blasint, A: ^complex128, lda: ^blasint, B: ^complex128, ldb: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---

	ctrttf_ :: proc(transr: cstring, uplo: cstring, n: ^blasint, A: ^complex64, lda: ^blasint, ARF: ^complex64, info: ^Info, _: c.size_t, _: c.size_t) ---
	dtrttf_ :: proc(transr: cstring, uplo: cstring, n: ^blasint, A: ^f64, lda: ^blasint, ARF: ^f64, info: ^Info, _: c.size_t, _: c.size_t) ---
	strttf_ :: proc(transr: cstring, uplo: cstring, n: ^blasint, A: ^f32, lda: ^blasint, ARF: ^f32, info: ^Info, _: c.size_t, _: c.size_t) ---
	ztrttf_ :: proc(transr: cstring, uplo: cstring, n: ^blasint, A: ^complex128, lda: ^blasint, ARF: ^complex128, info: ^Info, _: c.size_t, _: c.size_t) ---

	ctrttp_ :: proc(uplo: cstring, n: ^blasint, A: ^complex64, lda: ^blasint, AP: ^complex64, info: ^Info, _: c.size_t) ---
	dtrttp_ :: proc(uplo: cstring, n: ^blasint, A: ^f64, lda: ^blasint, AP: ^f64, info: ^Info, _: c.size_t) ---
	strttp_ :: proc(uplo: cstring, n: ^blasint, A: ^f32, lda: ^blasint, AP: ^f32, info: ^Info, _: c.size_t) ---
	ztrttp_ :: proc(uplo: cstring, n: ^blasint, A: ^complex128, lda: ^blasint, AP: ^complex128, info: ^Info, _: c.size_t) ---

	ctzrzf_ :: proc(m: ^blasint, n: ^blasint, A: ^complex64, lda: ^blasint, tau: ^complex64, work: ^complex64, lwork: ^blasint, info: ^Info) ---
	dtzrzf_ :: proc(m: ^blasint, n: ^blasint, A: ^f64, lda: ^blasint, tau: ^f64, work: ^f64, lwork: ^blasint, info: ^Info) ---
	stzrzf_ :: proc(m: ^blasint, n: ^blasint, A: ^f32, lda: ^blasint, tau: ^f32, work: ^f32, lwork: ^blasint, info: ^Info) ---
	ztzrzf_ :: proc(m: ^blasint, n: ^blasint, A: ^complex128, lda: ^blasint, tau: ^complex128, work: ^complex128, lwork: ^blasint, info: ^Info) ---

	cunbdb_ :: proc(trans: cstring, signs: cstring, m: ^blasint, p: ^blasint, q: ^blasint, X11: ^complex64, ldx11: ^blasint, X12: ^complex64, ldx12: ^blasint, X21: ^complex64, ldx21: ^blasint, X22: ^complex64, ldx22: ^blasint, theta: ^f32, phi: ^f32, TAUP1: ^complex64, TAUP2: ^complex64, TAUQ1: ^complex64, TAUQ2: ^complex64, work: ^complex64, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	zunbdb_ :: proc(trans: cstring, signs: cstring, m: ^blasint, p: ^blasint, q: ^blasint, X11: ^complex128, ldx11: ^blasint, X12: ^complex128, ldx12: ^blasint, X21: ^complex128, ldx21: ^blasint, X22: ^complex128, ldx22: ^blasint, theta: ^f64, phi: ^f64, TAUP1: ^complex128, TAUP2: ^complex128, TAUQ1: ^complex128, TAUQ2: ^complex128, work: ^complex128, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	cuncsd_ :: proc(jobu1: cstring, jobu2: cstring, jobv1t: cstring, jobv2t: cstring, trans: cstring, signs: cstring, m: ^blasint, p: ^blasint, q: ^blasint, X11: ^complex64, ldx11: ^blasint, X12: ^complex64, ldx12: ^blasint, X21: ^complex64, ldx21: ^blasint, X22: ^complex64, ldx22: ^blasint, theta: ^f32, U1: ^complex64, ldu1: ^blasint, U2: ^complex64, ldu2: ^blasint, V1T: ^complex64, ldv1t: ^blasint, V2T: ^complex64, ldv2t: ^blasint, work: ^complex64, lwork: ^blasint, rwork: ^f32, lrwork: ^blasint, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t) ---
	zuncsd_ :: proc(jobu1: cstring, jobu2: cstring, jobv1t: cstring, jobv2t: cstring, trans: cstring, signs: cstring, m: ^blasint, p: ^blasint, q: ^blasint, X11: ^complex128, ldx11: ^blasint, X12: ^complex128, ldx12: ^blasint, X21: ^complex128, ldx21: ^blasint, X22: ^complex128, ldx22: ^blasint, theta: ^f64, U1: ^complex128, ldu1: ^blasint, U2: ^complex128, ldu2: ^blasint, V1T: ^complex128, ldv1t: ^blasint, V2T: ^complex128, ldv2t: ^blasint, work: ^complex128, lwork: ^blasint, rwork: ^f64, lrwork: ^blasint, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t) ---

	cuncsd2by1_ :: proc(jobu1: cstring, jobu2: cstring, jobv1t: cstring, m: ^blasint, p: ^blasint, q: ^blasint, X11: ^complex64, ldx11: ^blasint, X21: ^complex64, ldx21: ^blasint, theta: ^f32, U1: ^complex64, ldu1: ^blasint, U2: ^complex64, ldu2: ^blasint, V1T: ^complex64, ldv1t: ^blasint, work: ^complex64, lwork: ^blasint, rwork: ^f32, lrwork: ^blasint, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	zuncsd2by1_ :: proc(jobu1: cstring, jobu2: cstring, jobv1t: cstring, m: ^blasint, p: ^blasint, q: ^blasint, X11: ^complex128, ldx11: ^blasint, X21: ^complex128, ldx21: ^blasint, theta: ^f64, U1: ^complex128, ldu1: ^blasint, U2: ^complex128, ldu2: ^blasint, V1T: ^complex128, ldv1t: ^blasint, work: ^complex128, lwork: ^blasint, rwork: ^f64, lrwork: ^blasint, iwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	cungbr_ :: proc(vect: cstring, m: ^blasint, n: ^blasint, k: ^blasint, A: ^complex64, lda: ^blasint, tau: ^complex64, work: ^complex64, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	zungbr_ :: proc(vect: cstring, m: ^blasint, n: ^blasint, k: ^blasint, A: ^complex128, lda: ^blasint, tau: ^complex128, work: ^complex128, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	cunghr_ :: proc(n: ^blasint, ilo: ^blasint, ihi: ^blasint, A: ^complex64, lda: ^blasint, tau: ^complex64, work: ^complex64, lwork: ^blasint, info: ^Info) ---
	zunghr_ :: proc(n: ^blasint, ilo: ^blasint, ihi: ^blasint, A: ^complex128, lda: ^blasint, tau: ^complex128, work: ^complex128, lwork: ^blasint, info: ^Info) ---

	cunglq_ :: proc(m: ^blasint, n: ^blasint, k: ^blasint, A: ^complex64, lda: ^blasint, tau: ^complex64, work: ^complex64, lwork: ^blasint, info: ^Info) ---
	zunglq_ :: proc(m: ^blasint, n: ^blasint, k: ^blasint, A: ^complex128, lda: ^blasint, tau: ^complex128, work: ^complex128, lwork: ^blasint, info: ^Info) ---
	cungql_ :: proc(m: ^blasint, n: ^blasint, k: ^blasint, A: ^complex64, lda: ^blasint, tau: ^complex64, work: ^complex64, lwork: ^blasint, info: ^Info) ---
	zungql_ :: proc(m: ^blasint, n: ^blasint, k: ^blasint, A: ^complex128, lda: ^blasint, tau: ^complex128, work: ^complex128, lwork: ^blasint, info: ^Info) ---

	cungqr_ :: proc(m: ^blasint, n: ^blasint, k: ^blasint, A: ^complex64, lda: ^blasint, tau: ^complex64, work: ^complex64, lwork: ^blasint, info: ^Info) ---
	zungqr_ :: proc(m: ^blasint, n: ^blasint, k: ^blasint, A: ^complex128, lda: ^blasint, tau: ^complex128, work: ^complex128, lwork: ^blasint, info: ^Info) ---
	cungrq_ :: proc(m: ^blasint, n: ^blasint, k: ^blasint, A: ^complex64, lda: ^blasint, tau: ^complex64, work: ^complex64, lwork: ^blasint, info: ^Info) ---
	zungrq_ :: proc(m: ^blasint, n: ^blasint, k: ^blasint, A: ^complex128, lda: ^blasint, tau: ^complex128, work: ^complex128, lwork: ^blasint, info: ^Info) ---
	cungtr_ :: proc(uplo: cstring, n: ^blasint, A: ^complex64, lda: ^blasint, tau: ^complex64, work: ^complex64, lwork: ^blasint, info: ^Info, _: c.size_t) ---
	zungtr_ :: proc(uplo: cstring, n: ^blasint, A: ^complex128, lda: ^blasint, tau: ^complex128, work: ^complex128, lwork: ^blasint, info: ^Info, _: c.size_t) ---

	cungtsqr_row_ :: proc(m: ^blasint, n: ^blasint, mb: ^blasint, nb: ^blasint, A: ^complex64, lda: ^blasint, T: ^complex64, ldt: ^blasint, work: ^complex64, lwork: ^blasint, info: ^Info) ---
	zungtsqr_row_ :: proc(m: ^blasint, n: ^blasint, mb: ^blasint, nb: ^blasint, A: ^complex128, lda: ^blasint, T: ^complex128, ldt: ^blasint, work: ^complex128, lwork: ^blasint, info: ^Info) ---
	cunhr_col_ :: proc(m: ^blasint, n: ^blasint, nb: ^blasint, A: ^complex64, lda: ^blasint, T: ^complex64, ldt: ^blasint, D: ^complex64, info: ^Info) ---
	zunhr_col_ :: proc(m: ^blasint, n: ^blasint, nb: ^blasint, A: ^complex128, lda: ^blasint, T: ^complex128, ldt: ^blasint, D: ^complex128, info: ^Info) ---

	cunmbr_ :: proc(vect: cstring, side: cstring, trans: cstring, m: ^blasint, n: ^blasint, k: ^blasint, A: ^complex64, lda: ^blasint, tau: ^complex64, C: ^complex64, ldc: ^blasint, work: ^complex64, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	zunmbr_ :: proc(vect: cstring, side: cstring, trans: cstring, m: ^blasint, n: ^blasint, k: ^blasint, A: ^complex128, lda: ^blasint, tau: ^complex128, C: ^complex128, ldc: ^blasint, work: ^complex128, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	cunmhr_ :: proc(side: cstring, trans: cstring, m: ^blasint, n: ^blasint, ilo: ^blasint, ihi: ^blasint, A: ^complex64, lda: ^blasint, tau: ^complex64, C: ^complex64, ldc: ^blasint, work: ^complex64, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	zunmhr_ :: proc(side: cstring, trans: cstring, m: ^blasint, n: ^blasint, ilo: ^blasint, ihi: ^blasint, A: ^complex128, lda: ^blasint, tau: ^complex128, C: ^complex128, ldc: ^blasint, work: ^complex128, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---

	cunmlq_ :: proc(side: cstring, trans: cstring, m: ^blasint, n: ^blasint, k: ^blasint, A: ^complex64, lda: ^blasint, tau: ^complex64, C: ^complex64, ldc: ^blasint, work: ^complex64, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	zunmlq_ :: proc(side: cstring, trans: cstring, m: ^blasint, n: ^blasint, k: ^blasint, A: ^complex128, lda: ^blasint, tau: ^complex128, C: ^complex128, ldc: ^blasint, work: ^complex128, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	cunmql_ :: proc(side: cstring, trans: cstring, m: ^blasint, n: ^blasint, k: ^blasint, A: ^complex64, lda: ^blasint, tau: ^complex64, C: ^complex64, ldc: ^blasint, work: ^complex64, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	zunmql_ :: proc(side: cstring, trans: cstring, m: ^blasint, n: ^blasint, k: ^blasint, A: ^complex128, lda: ^blasint, tau: ^complex128, C: ^complex128, ldc: ^blasint, work: ^complex128, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---

	cunmqr_ :: proc(side: cstring, trans: cstring, m: ^blasint, n: ^blasint, k: ^blasint, A: ^complex64, lda: ^blasint, tau: ^complex64, C: ^complex64, ldc: ^blasint, work: ^complex64, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	zunmqr_ :: proc(side: cstring, trans: cstring, m: ^blasint, n: ^blasint, k: ^blasint, A: ^complex128, lda: ^blasint, tau: ^complex128, C: ^complex128, ldc: ^blasint, work: ^complex128, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	cunmrq_ :: proc(side: cstring, trans: cstring, m: ^blasint, n: ^blasint, k: ^blasint, A: ^complex64, lda: ^blasint, tau: ^complex64, C: ^complex64, ldc: ^blasint, work: ^complex64, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	zunmrq_ :: proc(side: cstring, trans: cstring, m: ^blasint, n: ^blasint, k: ^blasint, A: ^complex128, lda: ^blasint, tau: ^complex128, C: ^complex128, ldc: ^blasint, work: ^complex128, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---

	cunmrz_ :: proc(side: cstring, trans: cstring, m: ^blasint, n: ^blasint, k: ^blasint, l: ^blasint, A: ^complex64, lda: ^blasint, tau: ^complex64, C: ^complex64, ldc: ^blasint, work: ^complex64, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	zunmrz_ :: proc(side: cstring, trans: cstring, m: ^blasint, n: ^blasint, k: ^blasint, l: ^blasint, A: ^complex128, lda: ^blasint, tau: ^complex128, C: ^complex128, ldc: ^blasint, work: ^complex128, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t) ---
	cunmtr_ :: proc(side: cstring, uplo: cstring, trans: cstring, m: ^blasint, n: ^blasint, A: ^complex64, lda: ^blasint, tau: ^complex64, C: ^complex64, ldc: ^blasint, work: ^complex64, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	zunmtr_ :: proc(side: cstring, uplo: cstring, trans: cstring, m: ^blasint, n: ^blasint, A: ^complex128, lda: ^blasint, tau: ^complex128, C: ^complex128, ldc: ^blasint, work: ^complex128, lwork: ^blasint, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---

	cupgtr_ :: proc(uplo: cstring, n: ^blasint, AP: ^complex64, tau: ^complex64, Q: ^complex64, ldq: ^blasint, work: ^complex64, info: ^Info, _: c.size_t) ---
	zupgtr_ :: proc(uplo: cstring, n: ^blasint, AP: ^complex128, tau: ^complex128, Q: ^complex128, ldq: ^blasint, work: ^complex128, info: ^Info, _: c.size_t) ---
	cupmtr_ :: proc(side: cstring, uplo: cstring, trans: cstring, m: ^blasint, n: ^blasint, AP: ^complex64, tau: ^complex64, C: ^complex64, ldc: ^blasint, work: ^complex64, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
	zupmtr_ :: proc(side: cstring, uplo: cstring, trans: cstring, m: ^blasint, n: ^blasint, AP: ^complex128, tau: ^complex128, C: ^complex128, ldc: ^blasint, work: ^complex128, info: ^Info, _: c.size_t, _: c.size_t, _: c.size_t) ---
}
