package openblas

import lapack "./f77"
import "base:builtin"
import "core:mem"

// ===================================================================================
// POSITIVE DEFINITE BANDED MATRIX OPERATIONS
//
// This file provides operations specifically for positive definite banded matrices:
// - PB-class LAPACK routines (Cholesky factorization, solvers, refinement)
// - Condition number estimation for positive definite banded matrices
// - Equilibration and scaling operations
//
// These routines handle symmetric/Hermitian positive definite banded matrices
// using specialized algorithms that are more efficient than general banded methods.
// ===================================================================================

// ===================================================================================
// PROC GROUPS FOR OVERLOADING
// ===================================================================================

// Expert driver for positive definite banded systems
band_pd_solve_expert :: proc {
	band_pd_solve_expert_real,
	band_pd_solve_expert_complex,
}

// Iterative refinement for positive definite banded systems
band_pd_refine :: proc {
	band_pd_refine_real,
	band_pd_refine_complex,
}

// Condition number estimation for positive definite banded matrices
band_pd_condition :: proc {
	band_pd_condition_real,
	band_pd_condition_complex,
}


// ===================================================================================
// WORKSPACE AND SIZE QUERIES
// ===================================================================================

// Query workspace for condition number estimation
query_workspace_band_pd_condition :: proc(AB: ^BandedMatrix($T)) -> (work: int, rwork: int, iwork: int) where is_float(T) || is_complex(T) {
	n := int(AB.cols)
	when is_float(T) {
		return 3 * n, 0, n
	} else when is_complex(T) {
		return 2 * n, n, 0
	}
}

// Query workspace for iterative refinement
query_workspace_band_pd_refine :: proc(AB: ^BandedMatrix($T)) -> (work: int, rwork: int, iwork: int) where is_float(T) || is_complex(T) {
	n := int(AB.cols)
	when is_float(T) {
		return 3 * n, 0, n
	} else when is_complex(T) {
		return 2 * n, n, 0
	}
}

// Query workspace for expert solver
query_workspace_band_pd_solve_expert :: proc(AB: ^BandedMatrix($T)) -> (work: int, rwork: int, iwork: int) where is_float(T) || is_complex(T) {
	n := int(AB.cols)
	when is_float(T) {
		return 3 * n, 0, n
	} else when is_complex(T) {
		return 2 * n, n, 0
	}
}

// Query array sizes for expert solver
query_result_sizes_band_pd_solve_expert :: proc(AB: ^BandedMatrix($T), B: ^Matrix(T)) -> (S_size: int, X_rows: int, X_cols: int, ferr_size: int, berr_size: int) where is_float(T) || is_complex(T) {
	n := int(AB.cols)
	nrhs := int(B.cols)
	return n, n, nrhs, nrhs, nrhs
}

// Query array sizes for equilibration
query_result_sizes_band_pd_equilibrate :: proc(AB: ^BandedMatrix($T)) -> (S_size: int) where is_float(T) || is_complex(T) {
	return int(AB.cols)
}

// Query array sizes for refinement
query_result_sizes_band_pd_refine :: proc(B: ^Matrix($T)) -> (ferr_size: int, berr_size: int) where is_float(T) || is_complex(T) {
	nrhs := int(B.cols)
	return nrhs, nrhs
}

// ===================================================================================
// CHOLESKY FACTORIZATION (PBTRF)
// ===================================================================================

// Cholesky factorization of positive definite banded matrix
band_pd_cholesky :: proc(
	uplo: MatrixRegion,
	AB: ^BandedMatrix($T), // Banded matrix (input/output - overwritten with Cholesky factor)
) -> (
	info: Info,
	ok: bool,
) {
	n := AB.cols
	kd := AB.kl // For PB matrices, kd = kl = ku
	ldab := AB.ldab
	uplo_c := cast(u8)uplo

	when T == f32 {
		lapack.spbtrf_(&uplo_c, &n, &kd, raw_data(AB.data), &ldab, &info)
	} else when T == f64 {
		lapack.dpbtrf_(&uplo_c, &n, &kd, raw_data(AB.data), &ldab, &info)
	} else when T == complex64 {
		lapack.cpbtrf_(&uplo_c, &n, &kd, raw_data(AB.data), &ldab, &info)
	} else when T == complex128 {
		lapack.zpbtrf_(&uplo_c, &n, &kd, raw_data(AB.data), &ldab, &info)
	}

	return info, info == 0
}

// ===================================================================================
// SOLVE WITH CHOLESKY FACTORIZATION (PBTRS)
// ===================================================================================

// Solve using pre-computed Cholesky factorization
band_pd_solve_cholesky :: proc(
	uplo: MatrixRegion,
	AB: ^BandedMatrix($T), // Cholesky factorization from band_pd_cholesky
	B: ^Matrix(T), // Right-hand side (input/output - overwritten with solution)
) -> (
	info: Info,
	ok: bool,
) {
	n := AB.cols
	kd := AB.kl
	nrhs := B.cols
	ldab := AB.ldab
	ldb := B.ld
	uplo_c := cast(u8)uplo

	when T == f32 {
		lapack.spbtrs_(&uplo_c, &n, &kd, &nrhs, raw_data(AB.data), &ldab, raw_data(B.data), &ldb, &info)
	} else when T == f64 {
		lapack.dpbtrs_(&uplo_c, &n, &kd, &nrhs, raw_data(AB.data), &ldab, raw_data(B.data), &ldb, &info)
	} else when T == complex64 {
		lapack.cpbtrs_(&uplo_c, &n, &kd, &nrhs, raw_data(AB.data), &ldab, raw_data(B.data), &ldb, &info)
	} else when T == complex128 {
		lapack.zpbtrs_(&uplo_c, &n, &kd, &nrhs, raw_data(AB.data), &ldab, raw_data(B.data), &ldb, &info)
	}

	return info, info == 0
}

// ===================================================================================
// SIMPLE DRIVER (PBSV)
// ===================================================================================

// Solve positive definite banded linear system: factor and solve in one call
band_pd_solve :: proc(
	uplo: MatrixRegion,
	AB: ^BandedMatrix($T), // Banded matrix (input/output - overwritten with Cholesky factor)
	B: ^Matrix(T), // Right-hand side (input/output - overwritten with solution)
) -> (
	info: Info,
	ok: bool,
) {
	n := AB.cols
	kd := AB.kl
	nrhs := B.cols
	ldab := AB.ldab
	ldb := B.ld
	uplo_c := cast(u8)uplo

	when T == f32 {
		lapack.spbsv_(&uplo_c, &n, &kd, &nrhs, raw_data(AB.data), &ldab, raw_data(B.data), &ldb, &info)
	} else when T == f64 {
		lapack.dpbsv_(&uplo_c, &n, &kd, &nrhs, raw_data(AB.data), &ldab, raw_data(B.data), &ldb, &info)
	} else when T == complex64 {
		lapack.cpbsv_(&uplo_c, &n, &kd, &nrhs, raw_data(AB.data), &ldab, raw_data(B.data), &ldb, &info)
	} else when T == complex128 {
		lapack.zpbsv_(&uplo_c, &n, &kd, &nrhs, raw_data(AB.data), &ldab, raw_data(B.data), &ldb, &info)
	}

	return info, info == 0
}

// ===================================================================================
// EXPERT DRIVER (PBSVX)
// ===================================================================================

// Expert solve for positive definite banded system (real version)
band_pd_solve_expert_real :: proc(
	fact: FactorizationOption,
	uplo: MatrixRegion,
	AB: ^BandedMatrix($T), // Banded matrix (input/output)
	AFB: ^BandedMatrix(T), // Factored matrix (input/output)
	equed: ^EquilibrationRequest, // Equilibration state (input/output)
	S: []T, // Scaling factors (input/output)
	B: ^Matrix(T), // Right-hand side (input/output)
	X: ^Matrix(T), // Solution matrix (output)
	ferr: []T, // Forward error bounds (output)
	berr: []T, // Backward error bounds (output)
	work: []T, // Workspace
	iwork: []Blas_Int, // Integer workspace
) -> (
	rcond: T,
	info: Info,
	ok: bool,
) where is_float(T) {
	n := AB.cols
	kd := AB.kl
	nrhs := B.cols
	ldab := AB.ldab
	ldafb := AFB.ldab
	ldb := B.ld
	ldx := X.ld
	fact_c := cast(u8)fact
	uplo_c := cast(u8)uplo

	assert(len(S) >= int(n), "Scaling array too small")
	assert(len(ferr) >= int(nrhs), "Forward error array too small")
	assert(len(berr) >= int(nrhs), "Backward error array too small")
	assert(len(work) >= 3 * int(n), "Work array too small")
	assert(len(iwork) >= int(n), "Integer work array too small")

	when T == f32 {
		lapack.spbsvx_(&fact_c, &uplo_c, &n, &kd, &nrhs, raw_data(AB.data), &ldab, raw_data(AFB.data), &ldafb, equed, raw_data(S), raw_data(B.data), &ldb, raw_data(X.data), &ldx, rcond, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(iwork), &info)
	} else when T == f64 {
		lapack.dpbsvx_(&fact_c, &uplo_c, &n, &kd, &nrhs, raw_data(AB.data), &ldab, raw_data(AFB.data), &ldafb, equed, raw_data(S), raw_data(B.data), &ldb, raw_data(X.data), &ldx, rcond, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(iwork), &info)
	}

	return rcond, info, info == 0
}

// Expert solve for positive definite banded system (complex version)
band_pd_solve_expert_complex :: proc(
	fact: FactorizationOption,
	uplo: MatrixRegion,
	AB: ^BandedMatrix($Cmplx), // Banded matrix (input/output)
	AFB: ^BandedMatrix(Cmplx), // Factored matrix (input/output)
	equed: ^EquilibrationRequest, // Equilibration state (input/output)
	S: []$Real, // Scaling factors (input/output)
	B: ^Matrix(Cmplx), // Right-hand side (input/output)
	X: ^Matrix(Cmplx), // Solution matrix (output)
	ferr: []Real, // Forward error bounds (output)
	berr: []Real, // Backward error bounds (output)
	work: []Cmplx, // Workspace
	rwork: []Real, // Real workspace
) -> (
	rcond: Real,
	info: Info,
	ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
	n := AB.cols
	kd := AB.kl
	nrhs := B.cols
	ldab := AB.ldab
	ldafb := AFB.ldab
	ldb := B.ld
	ldx := X.ld
	fact_c := cast(u8)fact
	uplo_c := cast(u8)uplo

	assert(len(S) >= int(n), "Scaling array too small")
	assert(len(ferr) >= int(nrhs), "Forward error array too small")
	assert(len(berr) >= int(nrhs), "Backward error array too small")
	assert(len(work) >= 2 * int(n), "Work array too small")
	assert(len(rwork) >= int(n), "Real work array too small")

	when Cmplx == complex64 {
		lapack.cpbsvx_(&fact_c, &uplo_c, &n, &kd, &nrhs, raw_data(AB.data), &ldab, raw_data(AFB.data), &ldafb, equed, raw_data(S), raw_data(B.data), &ldb, raw_data(X.data), &ldx, rcond, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(rwork), &info)
	} else when Cmplx == complex128 {
		lapack.zpbsvx_(&fact_c, &uplo_c, &n, &kd, &nrhs, raw_data(AB.data), &ldab, raw_data(AFB.data), &ldafb, equed, raw_data(S), raw_data(B.data), &ldb, raw_data(X.data), &ldx, rcond, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(rwork), &info)
	}

	return rcond, info, info == 0
}

// ===================================================================================
// ITERATIVE REFINEMENT (PBRFS)
// ===================================================================================

// Iterative refinement for positive definite banded systems (real version)
band_pd_refine_real :: proc(
	uplo: MatrixRegion,
	AB: ^BandedMatrix($T), // Original banded matrix
	AFB: ^BandedMatrix(T), // Factorized matrix from band_pd_cholesky
	B: ^Matrix(T), // Right-hand side matrix
	X: ^Matrix(T), // Solution matrix (input/output)
	ferr: []T, // Pre-allocated forward error bounds (size nrhs)
	berr: []T, // Pre-allocated backward error bounds (size nrhs)
	work: []T, // Pre-allocated workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	n := AB.cols
	kd := AB.kl // For PB matrices, kd = kl = ku
	nrhs := B.cols
	ldab := AB.ldab
	ldafb := AFB.ldab
	ldb := B.ld
	ldx := X.ld
	uplo_c := cast(u8)uplo

	assert(len(ferr) >= int(nrhs), "Forward error array too small")
	assert(len(berr) >= int(nrhs), "Backward error array too small")
	assert(len(work) >= 3 * int(n), "Work array too small")
	assert(len(iwork) >= int(n), "Integer work array too small")

	when T == f32 {
		lapack.spbrfs_(&uplo_c, &n, &kd, &nrhs, raw_data(AB.data), &ldab, raw_data(AFB.data), &ldafb, raw_data(B.data), &ldb, raw_data(X.data), &ldx, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(iwork), &info)
	} else when T == f64 {
		lapack.dpbrfs_(&uplo_c, &n, &kd, &nrhs, raw_data(AB.data), &ldab, raw_data(AFB.data), &ldafb, raw_data(B.data), &ldb, raw_data(X.data), &ldx, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(iwork), &info)
	}

	return info, info == 0
}

// Iterative refinement for positive definite banded systems (complex version)
band_pd_refine_complex :: proc(
	uplo: MatrixRegion,
	AB: ^BandedMatrix($Cmplx), // Original banded matrix
	AFB: ^BandedMatrix(Cmplx), // Factorized matrix from band_pd_cholesky
	B: ^Matrix(Cmplx), // Right-hand side matrix
	X: ^Matrix(Cmplx), // Solution matrix (input/output)
	ferr: []$Real, // Pre-allocated forward error bounds (size nrhs)
	berr: []Real, // Pre-allocated backward error bounds (size nrhs)
	work: []Cmplx, // Pre-allocated workspace
	rwork: []Real, // Pre-allocated real workspace
) -> (
	info: Info,
	ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
	n := AB.cols
	kd := AB.kl // For PB matrices, kd = kl = ku
	nrhs := B.cols
	ldab := AB.ldab
	ldafb := AFB.ldab
	ldb := B.ld
	ldx := X.ld
	uplo_c := cast(u8)uplo

	assert(len(ferr) >= int(nrhs), "Forward error array too small")
	assert(len(berr) >= int(nrhs), "Backward error array too small")
	assert(len(work) >= 2 * int(n), "Work array too small")
	assert(len(rwork) >= int(n), "Real work array too small")

	when Cmplx == complex64 {
		lapack.cpbrfs_(&uplo_c, &n, &kd, &nrhs, raw_data(AB.data), &ldab, raw_data(AFB.data), &ldafb, raw_data(B.data), &ldb, raw_data(X.data), &ldx, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(rwork), &info)
	} else when Cmplx == complex128 {
		lapack.zpbrfs_(&uplo_c, &n, &kd, &nrhs, raw_data(AB.data), &ldab, raw_data(AFB.data), &ldafb, raw_data(B.data), &ldb, raw_data(X.data), &ldx, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(rwork), &info)
	}

	return info, info == 0
}

// ===================================================================================
// SPLIT CHOLESKY FACTORIZATION (PBSTF)
// ===================================================================================

// Split Cholesky factorization for positive definite banded matrix
// Real: Computes split factor S from L^T*L where L = S*S^T
// Complex: Computes split factor S from L^H*L where L = S*S^H
band_pd_split_cholesky :: proc(
	uplo: MatrixRegion,
	AB: ^BandedMatrix($T), // Banded matrix (input/output)
) -> (
	info: Info,
	ok: bool,
) {
	n := AB.cols
	kd := AB.kl // For PB matrices, kd = kl = ku
	ldab := AB.ldab
	uplo_c := cast(u8)uplo

	when T == f32 {
		lapack.spbstf_(&uplo_c, &n, &kd, raw_data(AB.data), &ldab, &info)
	} else when T == f64 {
		lapack.dpbstf_(&uplo_c, &n, &kd, raw_data(AB.data), &ldab, &info)
	} else when T == complex64 {
		lapack.cpbstf_(&uplo_c, &n, &kd, raw_data(AB.data), &ldab, &info)
	} else when T == complex128 {
		lapack.zpbstf_(&uplo_c, &n, &kd, raw_data(AB.data), &ldab, &info)
	}

	return info, info == 0
}
// ===================================================================================
// CONDITION NUMBER ESTIMATION (PBCON)
// ===================================================================================

// Estimate the reciprocal condition number of a positive definite banded matrix (real version)
band_pd_condition_real :: proc(
	uplo: MatrixRegion,
	AB: ^BandedMatrix($T), // Cholesky factorization from band_pd_cholesky
	anorm: T, // 1-norm of the original matrix (before factorization)
	work: []T, // Pre-allocated workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace
) -> (
	rcond: T,
	info: Info,
	ok: bool,
) where is_float(T) {
	n := AB.cols
	kd := AB.kl
	ldab := AB.ldab
	uplo_c := cast(u8)uplo
	anorm := anorm

	assert(len(work) >= 3 * int(n), "Work array too small")
	assert(len(iwork) >= int(n), "Integer work array too small")

	when T == f32 {
		lapack.spbcon_(&uplo_c, &n, &kd, raw_data(AB.data), &ldab, &anorm, rcond, raw_data(work), raw_data(iwork), &info)
	} else when T == f64 {
		lapack.dpbcon_(&uplo_c, &n, &kd, raw_data(AB.data), &ldab, &anorm, rcond, raw_data(work), raw_data(iwork), &info)
	}

	return rcond, info, info == 0
}

// Estimate the reciprocal condition number of a positive definite banded matrix (complex version)
band_pd_condition_complex :: proc(
	uplo: MatrixRegion,
	AB: ^BandedMatrix($Cmplx), // Cholesky factorization from band_pd_cholesky
	anorm: $Real, // 1-norm of the original matrix (before factorization)
	work: []Cmplx, // Pre-allocated workspace
	rwork: []Real, // Pre-allocated real workspace
) -> (
	rcond: Real,
	info: Info,
	ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
	n := AB.cols
	kd := AB.kl
	ldab := AB.ldab
	uplo_c := cast(u8)uplo
	anorm := anorm

	assert(len(work) >= 2 * int(n), "Work array too small")
	assert(len(rwork) >= int(n), "Real work array too small")

	when Cmplx == complex64 {
		lapack.cpbcon_(&uplo_c, &n, &kd, raw_data(AB.data), &ldab, &anorm, rcond, raw_data(work), raw_data(rwork), &info)
	} else when Cmplx == complex128 {
		lapack.zpbcon_(&uplo_c, &n, &kd, raw_data(AB.data), &ldab, &anorm, rcond, raw_data(work), raw_data(rwork), &info)
	}

	return rcond, info, info == 0
}
