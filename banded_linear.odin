package openblas

import lapack "./f77"
import "base:builtin"
import "core:mem"

// ===================================================================================
// BANDED LINEAR SYSTEM SOLVERS (GB routines)
//
// This file provides comprehensive linear system solving for general banded matrices.
// These routines handle the GB (General Banded) class of LAPACK functions:
//
// Basic Operations:
// - GBSV:  Simple driver (factor + solve)
// - GBSVX: Expert driver (equilibration + factor + solve + refinement)
// - GBSVXX: Extended expert driver (extra-precise error bounds)
//
// Individual Steps:
// - GBTRF: LU factorization with partial pivoting
// - GBTRS: Solve using pre-computed LU factorization
// - GBCON: Condition number estimation
// - GBRFS: Iterative refinement
// - GBEQU: Matrix equilibration
//
// All routines use non-allocating API patterns and proper ILP64 support.
// ===================================================================================

// ===================================================================================
// PROC GROUPS FOR OVERLOADING
// ===================================================================================

// LU factorization for general banded matrices
// band_factor

// Solve using pre-computed LU factorization
// band_solve_factored

// Simple driver: factor and solve in one call
// band_solve

// Expert driver: equilibration + factor + solve + refinement
band_solve_expert :: proc {
	band_solve_expert_real,
	band_solve_expert_complex,
}

// Extended expert driver: extra-precise error bounds
band_solve_expert_extended :: proc {
	band_solve_expert_extended_real,
	band_solve_expert_extended_complex,
}

// Condition number estimation
band_condition :: proc {
	band_condition_real,
	band_condition_complex,
}

// Iterative refinement
band_refine :: proc {
	band_refine_real,
	band_refine_complex,
}

// Matrix equilibration
band_equilibrate :: proc {
	band_equilibrate_real,
	band_equilibrate_complex,
}

// Extended iterative refinement
band_refine_extended :: proc {
	band_refine_extended_real,
	band_refine_extended_complex,
}

// ===================================================================================
// WORKSPACE AND SIZE QUERIES
// ===================================================================================

// Query array sizes for LU factorization
query_result_sizes_band_factor :: proc(AB: ^BandedMatrix($T)) -> (ipiv_size: int) where is_float(T) || is_complex(T) {
	return int(min(AB.rows, AB.cols))
}

// Query workspace for condition number estimation
query_workspace_band_condition :: proc(AB: ^BandedMatrix($T)) -> (work: int, rwork: int, iwork: int) where is_float(T) || is_complex(T) {
	n := int(AB.cols)
	when is_float(T) {
		return 3 * n, 0, n
	} else when is_complex(T) {
		return 2 * n, n, 0
	}
}

// Query workspace for expert solver
query_workspace_band_solve_expert :: proc(AB: ^BandedMatrix($T)) -> (work: int, rwork: int, iwork: int) where is_float(T) || is_complex(T) {
	n := int(AB.cols)
	when is_float(T) {
		return 3 * n, 0, n
	} else when is_complex(T) {
		return 2 * n, n, 0
	}
}

// Query array sizes for expert solver
query_result_sizes_band_solve_expert :: proc(AB: ^BandedMatrix($T), B: ^Matrix(T)) -> (ipiv_size: int, AFB_rows: int, AFB_cols: int, R_size: int, C_size: int, X_rows: int, X_cols: int, ferr_size: int, berr_size: int) where is_float(T) || is_complex(T) {
	n := int(AB.cols)
	kl := int(AB.kl)
	ku := int(AB.ku)
	nrhs := int(B.cols)
	return n, 2 * kl + ku + 1, n, n, n, n, nrhs, nrhs, nrhs // ipiv// AFB rows (extended band for factorization)// AFB cols// R (row scale factors)// C (column scale factors)// X rows// X cols// ferr// berr
}

// Query array sizes for iterative refinement
query_result_sizes_band_refine :: proc(B: ^Matrix($T)) -> (ferr_size: int, berr_size: int) where is_float(T) || is_complex(T) {
	nrhs := int(B.cols)
	return nrhs, nrhs
}

// Query workspace for iterative refinement
query_workspace_band_refine :: proc(AB: ^BandedMatrix($T)) -> (work: int, rwork: int, iwork: int) where is_float(T) || is_complex(T) {
	n := int(AB.cols)
	when is_float(T) {
		return 3 * n, 0, n
	} else when is_complex(T) {
		return 2 * n, n, 0
	}
}

// Query workspace for extended expert solver
query_workspace_band_solve_expert_extended :: proc(AB: ^BandedMatrix($T)) -> (work: int, rwork: int, iwork: int) where is_float(T) || is_complex(T) {
	n := int(AB.cols)
	when is_float(T) {
		return 4 * n, 0, n
	} else when is_complex(T) {
		return 2 * n, 2 * n, 0
	}
}

// ===================================================================================
// LU FACTORIZATION (GBTRF)
// ===================================================================================

// LU factorization of general banded matrix (unified version)
band_factor :: proc(
	AB: ^BandedMatrix($T), // Banded matrix (input/output - overwritten with LU)
	ipiv: []Blas_Int, // Pre-allocated pivot indices (size min(m,n))
) -> (
	info: Info,
	ok: bool,
) where is_float(T) || is_complex(T) {
	m := AB.rows
	n := AB.cols
	kl := AB.kl
	ku := AB.ku
	ldab := AB.ldab

	assert(len(ipiv) >= int(min(m, n)), "Pivot array too small")

	when T == f32 {
		lapack.sgbtrf_(&m, &n, &kl, &ku, raw_data(AB.data), &ldab, raw_data(ipiv), &info)
	} else when T == f64 {
		lapack.dgbtrf_(&m, &n, &kl, &ku, raw_data(AB.data), &ldab, raw_data(ipiv), &info)
	} else when T == complex64 {
		lapack.cgbtrf_(&m, &n, &kl, &ku, raw_data(AB.data), &ldab, raw_data(ipiv), &info)
	} else when T == complex128 {
		lapack.zgbtrf_(&m, &n, &kl, &ku, raw_data(AB.data), &ldab, raw_data(ipiv), &info)
	}

	return info, info == 0
}

// ===================================================================================
// SOLVE WITH FACTORIZATION (GBTRS)
// ===================================================================================

// Solve using pre-computed LU factorization (unified version)
band_solve_factored :: proc(
	trans: TransposeMode,
	AB: ^BandedMatrix($T), // LU factorization from band_factor
	ipiv: []Blas_Int, // Pivot indices from band_factor
	B: ^Matrix(T), // Right-hand side (input/output - overwritten with solution)
) -> (
	info: Info,
	ok: bool,
) where is_float(T) || is_complex(T) {
	n := AB.cols
	kl := AB.kl
	ku := AB.ku
	nrhs := B.cols
	ldab := AB.ldab
	ldb := B.ld

	trans_c := cast(u8)trans

	when T == f32 {
		lapack.sgbtrs_(&trans_c, &n, &kl, &ku, &nrhs, raw_data(AB.data), &ldab, raw_data(ipiv), raw_data(B.data), &ldb, &info)
	} else when T == f64 {
		lapack.dgbtrs_(&trans_c, &n, &kl, &ku, &nrhs, raw_data(AB.data), &ldab, raw_data(ipiv), raw_data(B.data), &ldb, &info)
	} else when T == complex64 {
		lapack.cgbtrs_(&trans_c, &n, &kl, &ku, &nrhs, raw_data(AB.data), &ldab, raw_data(ipiv), raw_data(B.data), &ldb, &info)
	} else when T == complex128 {
		lapack.zgbtrs_(&trans_c, &n, &kl, &ku, &nrhs, raw_data(AB.data), &ldab, raw_data(ipiv), raw_data(B.data), &ldb, &info)
	}

	return info, info == 0
}

// ===================================================================================
// SIMPLE DRIVER (GBSV)
// ===================================================================================

// Solve banded linear system: factor and solve in one call (unified version)
band_solve :: proc(
	AB: ^BandedMatrix($T), // Banded matrix (input/output - overwritten with LU)
	B: ^Matrix(T), // Right-hand side (input/output - overwritten with solution)
	ipiv: []Blas_Int, // Pre-allocated pivot indices (size n)
) -> (
	info: Info,
	ok: bool,
) where is_float(T) || is_complex(T) {
	n := AB.cols
	kl := AB.kl
	ku := AB.ku
	nrhs := B.cols
	ldab := AB.ldab
	ldb := B.ld

	assert(len(ipiv) >= int(n), "Pivot array too small")

	when T == f32 {
		lapack.sgbsv_(&n, &kl, &ku, &nrhs, raw_data(AB.data), &ldab, raw_data(ipiv), raw_data(B.data), &ldb, &info)
	} else when T == f64 {
		lapack.dgbsv_(&n, &kl, &ku, &nrhs, raw_data(AB.data), &ldab, raw_data(ipiv), raw_data(B.data), &ldb, &info)
	} else when T == complex64 {
		lapack.cgbsv_(&n, &kl, &ku, &nrhs, raw_data(AB.data), &ldab, raw_data(ipiv), raw_data(B.data), &ldb, &info)
	} else when T == complex128 {
		lapack.zgbsv_(&n, &kl, &ku, &nrhs, raw_data(AB.data), &ldab, raw_data(ipiv), raw_data(B.data), &ldb, &info)
	}

	return info, info == 0
}

// ===================================================================================
// CONDITION NUMBER ESTIMATION (GBCON)
// ===================================================================================

// Estimate condition number of banded matrix (real version)
band_condition_real :: proc(
	norm: MatrixNorm,
	AB: ^BandedMatrix($T), // LU factorization from band_factor
	ipiv: []Blas_Int, // Pivot indices from band_factor
	anorm: T, // Norm of original matrix (computed separately)
	work: []T, // Pre-allocated workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace
) -> (
	rcond: T,
	info: Info,
	ok: bool,
) where is_float(T) {
	n := AB.cols
	kl := AB.kl
	ku := AB.ku
	ldab := AB.ldab

	assert(len(work) >= 3 * int(n), "Work array too small")
	assert(len(iwork) >= int(n), "Integer work array too small")

	norm_c := cast(u8)norm

	when T == f32 {
		lapack.sgbcon_(&norm_c, &n, &kl, &ku, raw_data(AB.data), &ldab, raw_data(ipiv), &anorm, rcond, raw_data(work), raw_data(iwork), &info)
	} else when T == f64 {
		lapack.dgbcon_(&norm_c, &n, &kl, &ku, raw_data(AB.data), &ldab, raw_data(ipiv), &anorm, rcond, raw_data(work), raw_data(iwork), &info)
	}

	return rcond, info, info == 0
}

// Estimate condition number of banded matrix (complex version)
band_condition_complex :: proc(
	norm: MatrixNorm,
	AB: ^BandedMatrix($Cmplx), // LU factorization from band_factor
	ipiv: []Blas_Int, // Pivot indices from band_factor
	anorm: $Real, // Norm of original matrix (computed separately)
	work: []Cmplx, // Pre-allocated workspace
	rwork: []Real, // Pre-allocated real workspace
) -> (
	rcond: Real,
	info: Info,
	ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
	n := AB.cols
	kl := AB.kl
	ku := AB.ku
	ldab := AB.ldab

	assert(len(work) >= 2 * int(n), "Work array too small")
	assert(len(rwork) >= int(n), "Real work array too small")

	norm_c := cast(u8)norm
	anorm := anorm

	when Cmplx == complex64 {
		lapack.cgbcon_(&norm_c, &n, &kl, &ku, raw_data(AB.data), &ldab, raw_data(ipiv), &anorm, rcond, raw_data(work), raw_data(rwork), &info)
	} else when Cmplx == complex128 {
		lapack.zgbcon_(&norm_c, &n, &kl, &ku, raw_data(AB.data), &ldab, raw_data(ipiv), &anorm, rcond, raw_data(work), raw_data(rwork), &info)
	}

	return rcond, info, info == 0
}

// ===================================================================================
// MATRIX EQUILIBRATION (GBEQU, GBEQUB)
// ===================================================================================

// Equilibrate general banded matrix (real version)
band_equilibrate_real :: proc(
	AB: ^BandedMatrix($T), // Banded matrix to equilibrate
	R: []T, // Pre-allocated row scale factors (size m)
	C: []T, // Pre-allocated column scale factors (size n)
	rowcnd: ^T, // Output: row condition number
	colcnd: ^T, // Output: column condition number
	amax: ^T, // Output: absolute maximum element
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	m := AB.rows
	n := AB.cols
	kl := AB.kl
	ku := AB.ku
	ldab := AB.ldab

	assert(len(R) >= int(m), "Row scale array too small")
	assert(len(C) >= int(n), "Column scale array too small")

	when T == f32 {
		lapack.sgbequ_(&m, &n, &kl, &ku, raw_data(AB.data), &ldab, raw_data(R), raw_data(C), rowcnd, colcnd, amax, &info)
	} else when T == f64 {
		lapack.dgbequ_(&m, &n, &kl, &ku, raw_data(AB.data), &ldab, raw_data(R), raw_data(C), rowcnd, colcnd, amax, &info)
	}

	return info, info == 0
}

// Equilibrate general banded matrix (complex version)
band_equilibrate_complex :: proc(
	AB: ^BandedMatrix($Cmplx), // Banded matrix to equilibrate
	R: []$Real, // Pre-allocated row scale factors (size m)
	C: []Real, // Pre-allocated column scale factors (size n)
	rowcnd: ^Real, // Output: row condition number
	colcnd: ^Real, // Output: column condition number
	amax: ^Real, // Output: absolute maximum element
) -> (
	info: Info,
	ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
	m := AB.rows
	n := AB.cols
	kl := AB.kl
	ku := AB.ku
	ldab := AB.ldab

	assert(len(R) >= int(m), "Row scale array too small")
	assert(len(C) >= int(n), "Column scale array too small")

	when Cmplx == complex64 {
		lapack.cgbequ_(&m, &n, &kl, &ku, raw_data(AB.data), &ldab, raw_data(R), raw_data(C), rowcnd, colcnd, amax, &info)
	} else when Cmplx == complex128 {
		lapack.zgbequ_(&m, &n, &kl, &ku, raw_data(AB.data), &ldab, raw_data(R), raw_data(C), rowcnd, colcnd, amax, &info)
	}

	return info, info == 0
}

// ===================================================================================
// ITERATIVE REFINEMENT (GBRFS)
// ===================================================================================

// Iterative refinement for banded matrix solution (real version)
band_refine_real :: proc(
	trans: TransposeMode,
	AB: ^BandedMatrix($T), // Original banded matrix
	AFB: ^BandedMatrix(T), // Factored matrix from band_factor
	ipiv: []Blas_Int, // Pivot indices from factorization
	B: ^Matrix(T), // Right-hand side
	X: ^Matrix(T), // Solution (input: initial, output: refined)
	ferr: []T, // Pre-allocated forward error bounds (size nrhs)
	berr: []T, // Pre-allocated backward error bounds (size nrhs)
	work: []T, // Pre-allocated workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	n := AB.cols
	kl := AB.kl
	ku := AB.ku
	nrhs := B.cols
	ldab := AB.ldab
	ldafb := AFB.ldab
	ldb := B.ld
	ldx := X.ld

	assert(len(ferr) >= int(nrhs), "Forward error array too small")
	assert(len(berr) >= int(nrhs), "Backward error array too small")
	assert(len(work) >= 3 * int(n), "Work array too small")
	assert(len(iwork) >= int(n), "Integer work array too small")
	assert(len(ipiv) >= int(n), "Pivot array too small")

	trans_c := cast(u8)trans

	when T == f32 {
		lapack.sgbrfs_(&trans_c, &n, &kl, &ku, &nrhs, raw_data(AB.data), &ldab, raw_data(AFB.data), &ldafb, raw_data(ipiv), raw_data(B.data), &ldb, raw_data(X.data), &ldx, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(iwork), &info)
	} else when T == f64 {
		lapack.dgbrfs_(&trans_c, &n, &kl, &ku, &nrhs, raw_data(AB.data), &ldab, raw_data(AFB.data), &ldafb, raw_data(ipiv), raw_data(B.data), &ldb, raw_data(X.data), &ldx, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(iwork), &info)
	}

	return info, info == 0
}

// Iterative refinement for banded matrix solution (complex version)
band_refine_complex :: proc(
	trans: TransposeMode,
	AB: ^BandedMatrix($Cmplx), // Original banded matrix
	AFB: ^BandedMatrix(Cmplx), // Factored matrix from band_factor
	ipiv: []Blas_Int, // Pivot indices from factorization
	B: ^Matrix(Cmplx), // Right-hand side
	X: ^Matrix(Cmplx), // Solution (input: initial, output: refined)
	ferr: []$Real, // Pre-allocated forward error bounds (size nrhs)
	berr: []Real, // Pre-allocated backward error bounds (size nrhs)
	work: []Cmplx, // Pre-allocated workspace
	rwork: []Real, // Pre-allocated real workspace
) -> (
	info: Info,
	ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
	n := AB.cols
	kl := AB.kl
	ku := AB.ku
	nrhs := B.cols
	ldab := AB.ldab
	ldafb := AFB.ldab
	ldb := B.ld
	ldx := X.ld

	assert(len(ferr) >= int(nrhs), "Forward error array too small")
	assert(len(berr) >= int(nrhs), "Backward error array too small")
	assert(len(work) >= 2 * int(n), "Work array too small")
	assert(len(rwork) >= int(n), "Real work array too small")
	assert(len(ipiv) >= int(n), "Pivot array too small")

	trans_c := cast(u8)trans

	when Cmplx == complex64 {
		lapack.cgbrfs_(&trans_c, &n, &kl, &ku, &nrhs, raw_data(AB.data), &ldab, raw_data(AFB.data), &ldafb, raw_data(ipiv), raw_data(B.data), &ldb, raw_data(X.data), &ldx, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(rwork), &info)
	} else when Cmplx == complex128 {
		lapack.zgbrfs_(&trans_c, &n, &kl, &ku, &nrhs, raw_data(AB.data), &ldab, raw_data(AFB.data), &ldafb, raw_data(ipiv), raw_data(B.data), &ldb, raw_data(X.data), &ldx, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(rwork), &info)
	}

	return info, info == 0
}

// ===================================================================================
// EXPERT DRIVER (GBSVX)
// ===================================================================================

// Expert solve for banded system (real version)
band_solve_expert_real :: proc(
	fact: FactorizationOption,
	trans: TransposeMode,
	AB: ^BandedMatrix($T), // Banded matrix (input/output based on fact)
	AFB: ^BandedMatrix(T), // Pre-allocated factored matrix (input/output based on fact)
	ipiv: []Blas_Int, // Pre-allocated pivot indices (input/output based on fact)
	equed: ^EquilibrationRequest, // Equilibration state (input/output)
	R: []T, // Pre-allocated row scale factors (input/output)
	C: []T, // Pre-allocated column scale factors (input/output)
	B: ^Matrix(T), // Right-hand side (input/output)
	X: ^Matrix(T), // Pre-allocated solution matrix (output)
	rcond: ^T, // Output: reciprocal condition number
	ferr: []T, // Pre-allocated forward error bounds (output)
	berr: []T, // Pre-allocated backward error bounds (output)
	work: []T, // Pre-allocated workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	n := AB.cols
	kl := AB.kl
	ku := AB.ku
	nrhs := B.cols
	ldab := AB.ldab
	ldafb := AFB.ldab
	ldb := B.ld
	ldx := X.ld

	assert(len(ipiv) >= int(n), "Pivot array too small")
	assert(len(R) >= int(n), "Row scale array too small")
	assert(len(C) >= int(n), "Column scale array too small")
	assert(len(ferr) >= int(nrhs), "Forward error array too small")
	assert(len(berr) >= int(nrhs), "Backward error array too small")
	assert(len(work) >= 3 * int(n), "Work array too small")
	assert(len(iwork) >= int(n), "Integer work array too small")

	fact_c := cast(u8)fact
	trans_c := cast(u8)trans

	when T == f32 {
		lapack.sgbsvx_(&fact_c, &trans_c, &n, &kl, &ku, &nrhs, raw_data(AB.data), &ldab, raw_data(AFB.data), &ldafb, raw_data(ipiv), equed, raw_data(R), raw_data(C), raw_data(B.data), &ldb, raw_data(X.data), &ldx, rcond, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(iwork), &info)
	} else when T == f64 {
		lapack.dgbsvx_(&fact_c, &trans_c, &n, &kl, &ku, &nrhs, raw_data(AB.data), &ldab, raw_data(AFB.data), &ldafb, raw_data(ipiv), equed, raw_data(R), raw_data(C), raw_data(B.data), &ldb, raw_data(X.data), &ldx, rcond, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(iwork), &info)
	}

	return info, info == 0
}

// Expert solve for banded system (complex version)
band_solve_expert_complex :: proc(
	fact: FactorizationOption,
	trans: TransposeMode,
	AB: ^BandedMatrix($Cmplx), // Banded matrix (input/output based on fact)
	AFB: ^BandedMatrix(Cmplx), // Pre-allocated factored matrix (input/output based on fact)
	ipiv: []Blas_Int, // Pre-allocated pivot indices (input/output based on fact)
	equed: ^EquilibrationRequest, // Equilibration state (input/output)
	R: []$Real, // Pre-allocated row scale factors (input/output)
	C: []Real, // Pre-allocated column scale factors (input/output)
	B: ^Matrix(Cmplx), // Right-hand side (input/output)
	X: ^Matrix(Cmplx), // Pre-allocated solution matrix (output)
	rcond: ^Real, // Output: reciprocal condition number
	ferr: []Real, // Pre-allocated forward error bounds (output)
	berr: []Real, // Pre-allocated backward error bounds (output)
	work: []Cmplx, // Pre-allocated workspace
	rwork: []Real, // Pre-allocated real workspace
) -> (
	info: Info,
	ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
	n := AB.cols
	kl := AB.kl
	ku := AB.ku
	nrhs := B.cols
	ldab := AB.ldab
	ldafb := AFB.ldab
	ldb := B.ld
	ldx := X.ld

	assert(len(ipiv) >= int(n), "Pivot array too small")
	assert(len(R) >= int(n), "Row scale array too small")
	assert(len(C) >= int(n), "Column scale array too small")
	assert(len(ferr) >= int(nrhs), "Forward error array too small")
	assert(len(berr) >= int(nrhs), "Backward error array too small")
	assert(len(work) >= 2 * int(n), "Work array too small")
	assert(len(rwork) >= int(n), "Real work array too small")

	fact_c := cast(u8)fact
	trans_c := cast(u8)trans

	when Cmplx == complex64 {
		lapack.cgbsvx_(
			&fact_c,
			&trans_c,
			&n,
			&kl,
			&ku,
			&nrhs,
			raw_data(AB.data),
			&ldab,
			raw_data(AFB.data),
			&ldafb,
			raw_data(ipiv),
			cast(^u8)equed,
			raw_data(R),
			raw_data(C),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			rcond,
			raw_data(ferr),
			raw_data(berr),
			raw_data(work),
			raw_data(rwork),
			&info,
		)
	} else when Cmplx == complex128 {
		lapack.zgbsvx_(
			&fact_c,
			&trans_c,
			&n,
			&kl,
			&ku,
			&nrhs,
			raw_data(AB.data),
			&ldab,
			raw_data(AFB.data),
			&ldafb,
			raw_data(ipiv),
			cast(^u8)equed,
			raw_data(R),
			raw_data(C),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			rcond,
			raw_data(ferr),
			raw_data(berr),
			raw_data(work),
			raw_data(rwork),
			&info,
		)
	}

	return info, info == 0
}

// ===================================================================================
// EXTENDED EXPERT DRIVER (GBSVXX)
// ===================================================================================

// Query result sizes for extended expert banded solve
query_result_sizes_band_solve_expert_extended :: proc(
	AB: ^BandedMatrix($T),
	B: ^Matrix(T),
	n_err_bnds: int = 3,
) -> (
	ipiv_size: int,
	AFB_rows: int,
	AFB_cols: int,
	R_size: int,
	C_size: int,
	X_rows: int,
	X_cols: int,
	berr_size: int,
	err_bnds_norm_size: int,
	err_bnds_comp_size: int,
	params_size: int,
) where is_float(T) ||
	is_complex(T) {
	n := int(AB.cols)
	kl := int(AB.kl)
	ku := int(AB.ku)
	nrhs := int(B.cols)

	ipiv_size = n
	AFB_rows = 2 * kl + ku + 1
	AFB_cols = n
	R_size = n
	C_size = n
	X_rows = n
	X_cols = nrhs
	berr_size = nrhs
	err_bnds_norm_size = nrhs * n_err_bnds
	err_bnds_comp_size = nrhs * n_err_bnds
	params_size = 3

	return
}


// Extended expert solve for banded system (real version)
band_solve_expert_extended_real :: proc(
	fact: FactorizationOption,
	trans: TransposeMode,
	AB: ^BandedMatrix($T), // Banded matrix (input/output based on fact)
	AFB: ^BandedMatrix(T), // Pre-allocated factored matrix (input/output based on fact)
	ipiv: []Blas_Int, // Pre-allocated pivot indices (input/output based on fact)
	equed: ^EquilibrationRequest, // Equilibration state (input/output)
	R: []T, // Pre-allocated row scale factors (input/output)
	C: []T, // Pre-allocated column scale factors (input/output)
	B: ^Matrix(T), // Right-hand side (input/output)
	X: ^Matrix(T), // Pre-allocated solution matrix (output)
	rcond: ^T, // Output: reciprocal condition number
	rpvgrw: ^T, // Output: reciprocal pivot growth factor
	berr: []T, // Pre-allocated backward error bounds (output)
	n_err_bnds: Blas_Int, // Number of error bounds to compute
	err_bnds_norm: []T, // Pre-allocated normwise error bounds (output)
	err_bnds_comp: []T, // Pre-allocated componentwise error bounds (output)
	params: []T, // Algorithm parameters (input/output)
	work: []T, // Pre-allocated workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	n := AB.cols
	kl := AB.kl
	ku := AB.ku
	nrhs := B.cols
	ldab := AB.ldab
	ldafb := AFB.ldab
	ldb := B.ld
	ldx := X.ld

	assert(len(ipiv) >= int(n), "Pivot array too small")
	assert(len(R) >= int(n), "Row scale array too small")
	assert(len(C) >= int(n), "Column scale array too small")
	assert(len(berr) >= int(nrhs), "Backward error array too small")
	assert(len(err_bnds_norm) >= int(nrhs * n_err_bnds), "Normwise error bounds array too small")
	assert(len(err_bnds_comp) >= int(nrhs * n_err_bnds), "Componentwise error bounds array too small")
	assert(len(params) >= 3, "Parameters array too small")
	assert(len(work) >= 4 * int(n), "Work array too small")
	assert(len(iwork) >= int(n), "Integer work array too small")

	fact_c := cast(u8)fact
	trans_c := cast(u8)trans

	nparams := Blas_Int(len(params))

	when T == f32 {
		lapack.sgbsvxx_(
			&fact_c,
			&trans_c,
			&n,
			&kl,
			&ku,
			&nrhs,
			raw_data(AB.data),
			&ldab,
			raw_data(AFB.data),
			&ldafb,
			raw_data(ipiv),
			cast(^u8)equed,
			raw_data(R),
			raw_data(C),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			rcond,
			rpvgrw,
			raw_data(berr),
			&n_err_bnds,
			raw_data(err_bnds_norm),
			raw_data(err_bnds_comp),
			&nparams,
			raw_data(params),
			raw_data(work),
			raw_data(iwork),
			&info,
		)
	} else when T == f64 {
		lapack.dgbsvxx_(
			&fact_c,
			&trans_c,
			&n,
			&kl,
			&ku,
			&nrhs,
			raw_data(AB.data),
			&ldab,
			raw_data(AFB.data),
			&ldafb,
			raw_data(ipiv),
			cast(^u8)equed,
			raw_data(R),
			raw_data(C),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			rcond,
			rpvgrw,
			raw_data(berr),
			&n_err_bnds,
			raw_data(err_bnds_norm),
			raw_data(err_bnds_comp),
			&nparams,
			raw_data(params),
			raw_data(work),
			raw_data(iwork),
			&info,
		)
	}

	return info, info == 0
}

// Extended expert solve for banded system (complex version)
band_solve_expert_extended_complex :: proc(
	fact: FactorizationOption,
	trans: TransposeMode,
	AB: ^BandedMatrix($Cmplx), // Banded matrix (input/output based on fact)
	AFB: ^BandedMatrix(Cmplx), // Pre-allocated factored matrix (input/output based on fact)
	ipiv: []Blas_Int, // Pre-allocated pivot indices (input/output based on fact)
	equed: ^EquilibrationRequest, // Equilibration state (input/output)
	R: []$Real, // Pre-allocated row scale factors (input/output)
	C: []Real, // Pre-allocated column scale factors (input/output)
	B: ^Matrix(Cmplx), // Right-hand side (input/output)
	X: ^Matrix(Cmplx), // Pre-allocated solution matrix (output)
	rcond: ^Real, // Output: reciprocal condition number
	rpvgrw: ^Real, // Output: reciprocal pivot growth factor
	berr: []Real, // Pre-allocated backward error bounds (output)
	n_err_bnds: Blas_Int = 3, // Number of error bounds to compute
	err_bnds_norm: []Real, // Pre-allocated normwise error bounds (output)
	err_bnds_comp: []Real, // Pre-allocated componentwise error bounds (output)
	params: []Real, // Algorithm parameters (input/output)
	work: []Cmplx, // Pre-allocated workspace
	rwork: []Real, // Pre-allocated real workspace
) -> (
	info: Info,
	ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
	n := AB.cols
	kl := AB.kl
	ku := AB.ku
	nrhs := B.cols
	ldab := AB.ldab
	ldafb := AFB.ldab
	ldb := B.ld
	ldx := X.ld

	assert(len(ipiv) >= int(n), "Pivot array too small")
	assert(len(R) >= int(n), "Row scale array too small")
	assert(len(C) >= int(n), "Column scale array too small")
	assert(len(berr) >= int(nrhs), "Backward error array too small")
	assert(len(err_bnds_norm) >= int(nrhs * n_err_bnds), "Normwise error bounds array too small")
	assert(len(err_bnds_comp) >= int(nrhs * n_err_bnds), "Componentwise error bounds array too small")
	assert(len(params) >= 3, "Parameters array too small")
	assert(len(work) >= 2 * int(n), "Work array too small")
	assert(len(rwork) >= 2 * int(n), "Real work array too small")

	fact_c := cast(u8)fact
	trans_c := cast(u8)trans

	nparams := Blas_Int(len(params))

	when Cmplx == complex64 {
		lapack.cgbsvxx_(
			&fact_c,
			&trans_c,
			&n,
			&kl,
			&ku,
			&nrhs,
			raw_data(AB.data),
			&ldab,
			raw_data(AFB.data),
			&ldafb,
			raw_data(ipiv),
			cast(^u8)equed,
			raw_data(R),
			raw_data(C),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			rcond,
			rpvgrw,
			raw_data(berr),
			&n_err_bnds,
			raw_data(err_bnds_norm),
			raw_data(err_bnds_comp),
			&nparams,
			raw_data(params),
			raw_data(work),
			raw_data(rwork),
			&info,
		)
	} else when Cmplx == complex128 {
		lapack.zgbsvxx_(
			&fact_c,
			&trans_c,
			&n,
			&kl,
			&ku,
			&nrhs,
			raw_data(AB.data),
			&ldab,
			raw_data(AFB.data),
			&ldafb,
			raw_data(ipiv),
			cast(^u8)equed,
			raw_data(R),
			raw_data(C),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			rcond,
			rpvgrw,
			raw_data(berr),
			&n_err_bnds,
			raw_data(err_bnds_norm),
			raw_data(err_bnds_comp),
			&nparams,
			raw_data(params),
			raw_data(work),
			raw_data(rwork),
			&info,
		)
	}

	return info, info == 0
}

// ===================================================================================
// EXTENDED ITERATIVE REFINEMENT (GBRFSX)
// ===================================================================================

// Query result sizes for extended iterative refinement
query_result_sizes_band_refine_extended :: proc(B: ^Matrix($T)) -> (rcond_size: int, berr_size: int, err_bnds_norm_size: int, err_bnds_comp_size: int, params_size: int) where is_float(T) || is_complex(T) {
	nrhs := int(B.cols)
	n_err_bnds := 3
	return 1, nrhs, nrhs * n_err_bnds, nrhs * n_err_bnds, 3
}

// Query workspace for extended iterative refinement
query_workspace_band_refine_extended :: proc(AB: ^BandedMatrix($T)) -> (work: int, rwork: int, iwork: int) where is_float(T) || is_complex(T) {
	n := int(AB.cols)
	when is_float(T) {
		return 4 * n, 0, n
	} else when is_complex(T) {
		return 2 * n, 2 * n, 0
	}
}

// Extended iterative refinement for banded matrix solution (real version)
band_refine_extended_real :: proc(
	trans: TransposeMode,
	equed: EquilibrationRequest,
	AB: ^BandedMatrix($T), // Original banded matrix
	AFB: ^BandedMatrix(T), // Factored matrix from band_factor
	ipiv: []Blas_Int, // Pivot indices from factorization
	R: []T, // Row scale factors from equilibration
	C: []T, // Column scale factors from equilibration
	B: ^Matrix(T), // Right-hand side
	X: ^Matrix(T), // Solution (input: initial, output: refined)
	rcond: ^T, // Output: reciprocal condition number
	berr: []T, // Pre-allocated backward error bounds (size nrhs)
	n_err_bnds: ^Blas_Int, // Output: number of error bounds computed
	err_bnds_norm: []T, // Pre-allocated normwise error bounds
	err_bnds_comp: []T, // Pre-allocated componentwise error bounds
	nparams: ^Blas_Int, // Output: number of parameters
	params: []T, // Pre-allocated algorithm parameters
	work: []T, // Pre-allocated workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	n := AB.cols
	kl := AB.kl
	ku := AB.ku
	nrhs := B.cols
	ldab := AB.ldab
	ldafb := AFB.ldab
	ldb := B.ld
	ldx := X.ld

	assert(len(berr) >= int(nrhs), "Backward error array too small")
	assert(len(err_bnds_norm) >= int(nrhs) * 3, "Normwise error bounds array too small")
	assert(len(err_bnds_comp) >= int(nrhs) * 3, "Componentwise error bounds array too small")
	assert(len(params) >= 3, "Params array too small")
	assert(len(work) >= 4 * int(n), "Work array too small")
	assert(len(iwork) >= int(n), "Integer work array too small")
	assert(len(ipiv) >= int(n), "Pivot array too small")
	assert(len(R) >= int(n), "Row scaling array too small")
	assert(len(C) >= int(n), "Column scaling array too small")

	trans_c := cast(u8)trans
	equed_c := cast(u8)equed
	n_err_bnds^ = 3

	when T == f32 {
		lapack.sgbrfsx_(
			&trans_c,
			&equed_c,
			&n,
			&kl,
			&ku,
			&nrhs,
			raw_data(AB.data),
			&ldab,
			raw_data(AFB.data),
			&ldafb,
			raw_data(ipiv),
			raw_data(R),
			raw_data(C),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			rcond,
			raw_data(berr),
			n_err_bnds,
			raw_data(err_bnds_norm),
			raw_data(err_bnds_comp),
			nparams,
			raw_data(params),
			raw_data(work),
			raw_data(iwork),
			&info,
		)
	} else when T == f64 {
		lapack.dgbrfsx_(
			&trans_c,
			&equed_c,
			&n,
			&kl,
			&ku,
			&nrhs,
			raw_data(AB.data),
			&ldab,
			raw_data(AFB.data),
			&ldafb,
			raw_data(ipiv),
			raw_data(R),
			raw_data(C),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			rcond,
			raw_data(berr),
			n_err_bnds,
			raw_data(err_bnds_norm),
			raw_data(err_bnds_comp),
			nparams,
			raw_data(params),
			raw_data(work),
			raw_data(iwork),
			&info,
		)
	}

	return info, info == 0
}

// Extended iterative refinement for banded matrix solution (complex version)
band_refine_extended_complex :: proc(
	trans: TransposeMode,
	equed: EquilibrationRequest,
	AB: ^BandedMatrix($Cmplx), // Original banded matrix
	AFB: ^BandedMatrix(Cmplx), // Factored matrix from band_factor
	ipiv: []Blas_Int, // Pivot indices from factorization
	R: []$Real, // Row scale factors from equilibration
	C: []Real, // Column scale factors from equilibration
	B: ^Matrix(Cmplx), // Right-hand side
	X: ^Matrix(Cmplx), // Solution (input: initial, output: refined)
	rcond: ^Real, // Output: reciprocal condition number
	berr: []Real, // Pre-allocated backward error bounds (size nrhs)
	n_err_bnds: ^Blas_Int, // Output: number of error bounds computed
	err_bnds_norm: []Real, // Pre-allocated normwise error bounds
	err_bnds_comp: []Real, // Pre-allocated componentwise error bounds
	nparams: ^Blas_Int, // Output: number of parameters
	params: []Real, // Pre-allocated algorithm parameters
	work: []Cmplx, // Pre-allocated workspace
	rwork: []Real, // Pre-allocated real workspace
) -> (
	info: Info,
	ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
	n := AB.cols
	kl := AB.kl
	ku := AB.ku
	nrhs := B.cols
	ldab := AB.ldab
	ldafb := AFB.ldab
	ldb := B.ld
	ldx := X.ld

	assert(len(berr) >= int(nrhs), "Backward error array too small")
	assert(len(err_bnds_norm) >= int(nrhs) * 3, "Normwise error bounds array too small")
	assert(len(err_bnds_comp) >= int(nrhs) * 3, "Componentwise error bounds array too small")
	assert(len(params) >= 3, "Params array too small")
	assert(len(work) >= 2 * int(n), "Work array too small")
	assert(len(rwork) >= 2 * int(n), "Real work array too small")
	assert(len(ipiv) >= int(n), "Pivot array too small")
	assert(len(R) >= int(n), "Row scaling array too small")
	assert(len(C) >= int(n), "Column scaling array too small")

	trans_c := cast(u8)trans
	equed_c := cast(u8)equed
	n_err_bnds^ = 3

	when Cmplx == complex64 {
		lapack.cgbrfsx_(
			&trans_c,
			&equed_c,
			&n,
			&kl,
			&ku,
			&nrhs,
			raw_data(AB.data),
			&ldab,
			raw_data(AFB.data),
			&ldafb,
			raw_data(ipiv),
			raw_data(R),
			raw_data(C),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			rcond,
			raw_data(berr),
			n_err_bnds,
			raw_data(err_bnds_norm),
			raw_data(err_bnds_comp),
			nparams,
			raw_data(params),
			raw_data(work),
			raw_data(rwork),
			&info,
		)
	} else when Cmplx == complex128 {
		lapack.zgbrfsx_(
			&trans_c,
			&equed_c,
			&n,
			&kl,
			&ku,
			&nrhs,
			raw_data(AB.data),
			&ldab,
			raw_data(AFB.data),
			&ldafb,
			raw_data(ipiv),
			raw_data(R),
			raw_data(C),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			rcond,
			raw_data(berr),
			n_err_bnds,
			raw_data(err_bnds_norm),
			raw_data(err_bnds_comp),
			nparams,
			raw_data(params),
			raw_data(work),
			raw_data(rwork),
			&info,
		)
	}

	return info, info == 0
}
