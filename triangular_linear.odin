package openblas

import lapack "./f77"
import "base:intrinsics"


// ===================================================================================
// TRIANGULAR LINEAR SOLVERS AND MATRIX INVERSION
// ===================================================================================
//
// This file provides linear algebra operations for full-storage triangular matrices:
// - Triangular system solving (TRTRS)
// - Triangular matrix inversion (TRTRI)
// - Iterative refinement for triangular systems (TRRFS)
//
// All functions use the non-allocating API pattern with pre-allocated arrays.

// ===================================================================================
// TRIANGULAR SYSTEM SOLVING (TRTRS)
// ===================================================================================

// Solve triangular system Ax = b or AX = B
tri_solve :: proc(
	A: ^Triangular($T), // Triangular matrix
	B: ^Matrix(T), // RHS matrix/vectors (modified to solution)
	trans: TransposeMode = .None, // Transpose operation
) -> (
	info: Info,
	ok: bool,
) where is_float(T) || is_complex(T) {
	assert(int(B.rows) == A.n, "B rows must match A dimension")

	uplo_c := u8(A.uplo)
	trans_c := u8(trans)
	diag_c := u8(A.diag)
	n := A.n
	nrhs := B.cols
	lda_blas := A.lda
	ldb_blas := B.ld

	when T == f32 {
		lapack.strtrs_(&uplo_c, &trans_c, &diag_c, &n, &nrhs, raw_data(A.data), &lda_blas, raw_data(B.data), &ldb_blas, &info)
	} else when T == f64 {
		lapack.dtrtrs_(&uplo_c, &trans_c, &diag_c, &n, &nrhs, raw_data(A.data), &lda_blas, raw_data(B.data), &ldb_blas, &info)
	} else when T == complex64 {
		lapack.ctrtrs_(&uplo_c, &trans_c, &diag_c, &n, &nrhs, raw_data(A.data), &lda_blas, raw_data(B.data), &ldb_blas, &info)
	} else when T == complex128 {
		lapack.ztrtrs_(&uplo_c, &trans_c, &diag_c, &n, &nrhs, raw_data(A.data), &lda_blas, raw_data(B.data), &ldb_blas, &info)
	}

	return info, info == 0
}

// ===================================================================================
// TRIANGULAR MATRIX INVERSION (TRTRI)
// ===================================================================================

// Invert triangular matrix in-place
tri_invert :: proc(
	A: ^Triangular($T), // Triangular matrix (modified to inverse)
) -> (
	info: Info,
	ok: bool,
) where is_float(T) || is_complex(T) {
	uplo_c := u8(A.uplo)
	diag_c := u8(A.diag)
	n := A.n
	lda_blas := A.lda

	when T == f32 {
		lapack.strtri_(&uplo_c, &diag_c, &n, raw_data(A.data), &lda_blas, &info)
	} else when T == f64 {
		lapack.dtrtri_(&uplo_c, &diag_c, &n, raw_data(A.data), &lda_blas, &info)
	} else when T == complex64 {
		lapack.ctrtri_(&uplo_c, &diag_c, &n, raw_data(A.data), &lda_blas, &info)
	} else when T == complex128 {
		lapack.ztrtri_(&uplo_c, &diag_c, &n, raw_data(A.data), &lda_blas, &info)
	}

	return info, info == 0
}

// ===================================================================================
// ITERATIVE REFINEMENT (TRRFS)
// ===================================================================================

// Iterative refinement for triangular systems
tri_refine :: proc {
	tri_refine_real,
	tri_refine_complex,
}

// Real triangular iterative refinement (f32/f64)
tri_refine_real :: proc(
	A: ^Triangular($T), // Triangular matrix
	B: ^Matrix(T), // Original RHS
	X: ^Matrix(T), // Solution (refined on output)
	ferr: []T, // Pre-allocated forward error bounds (size nrhs)
	berr: []T, // Pre-allocated backward error bounds (size nrhs)
	work: []T, // Pre-allocated workspace (size 3*n)
	iwork: []Blas_Int, // Pre-allocated integer workspace (size n)
	trans: TransposeMode = .None, // Transpose operation
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	n := A.n
	nrhs := B.cols
	assert(int(B.rows) == n, "B rows must match A dimension")
	assert(int(X.rows) == n, "X rows must match A dimension")
	assert(B.cols == X.cols, "B and X must have same number of columns")
	assert(len(ferr) >= int(nrhs), "Forward error array too small")
	assert(len(berr) >= int(nrhs), "Backward error array too small")
	assert(len(work) >= 3 * int(n), "Workspace too small")
	assert(len(iwork) >= int(n), "Integer workspace too small")

	uplo_c := u8(A.uplo)
	trans_c := u8(trans)
	diag_c := u8(A.diag)
	lda_blas := A.lda
	ldb_blas := B.ld
	ldx_blas := X.ld

	when T == f32 {
		lapack.strrfs_(&uplo_c, &trans_c, &diag_c, &n, &nrhs, raw_data(A.data), &lda_blas, raw_data(B.data), &ldb_blas, raw_data(X.data), &ldx_blas, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(iwork), &info)
	} else when T == f64 {
		lapack.dtrrfs_(&uplo_c, &trans_c, &diag_c, &n, &nrhs, raw_data(A.data), &lda_blas, raw_data(B.data), &ldb_blas, raw_data(X.data), &ldx_blas, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(iwork), &info)
	}

	return info, info == 0
}

// Complex triangular iterative refinement (complex64/complex128)
tri_refine_complex :: proc(
	A: ^Triangular($Cmplx), // Triangular matrix
	B: ^Matrix(Cmplx), // Original RHS
	X: ^Matrix(Cmplx), // Solution (refined on output)
	ferr: []$Real, // Pre-allocated forward error bounds (size nrhs)
	berr: []Real, // Pre-allocated backward error bounds (size nrhs)
	work: []Cmplx, // Pre-allocated workspace (size 2*n)
	rwork: []Real, // Pre-allocated real workspace (size n)
	trans: TransposeMode = .None, // Transpose operation
) -> (
	info: Info,
	ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
	n := A.n
	nrhs := B.cols
	assert(int(B.rows) == n, "B rows must match A dimension")
	assert(int(X.rows) == n, "X rows must match A dimension")
	assert(B.cols == X.cols, "B and X must have same number of columns")
	assert(len(ferr) >= int(nrhs), "Forward error array too small")
	assert(len(berr) >= int(nrhs), "Backward error array too small")
	assert(len(work) >= 2 * n, "Workspace too small")
	assert(len(rwork) >= n, "Real workspace too small")

	uplo_c := u8(A.uplo)
	trans_c := u8(trans)
	diag_c := u8(A.diag)
	lda_blas := A.lda
	ldb_blas := B.ld
	ldx_blas := X.ld

	when Cmplx == complex64 {
		lapack.ctrrfs_(&uplo_c, &trans_c, &diag_c, &n, &nrhs, raw_data(A.data), &lda_blas, raw_data(B.data), &ldb_blas, raw_data(X.data), &ldx_blas, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(rwork), &info)
	} else when Cmplx == complex128 {
		lapack.ztrrfs_(&uplo_c, &trans_c, &diag_c, &n, &nrhs, raw_data(A.data), &lda_blas, raw_data(B.data), &ldb_blas, raw_data(X.data), &ldx_blas, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(rwork), &info)
	}

	return info, info == 0
}

// ===================================================================================
// WORKSPACE QUERY FUNCTIONS
// ===================================================================================

// Query workspace size for triangular refinement
query_workspace_tri_refine :: proc(A: Triangular($T)) -> (work_size: int, iwork_size: int, rwork_size: int) where is_float(T) || is_complex(T) {
	n := A.n
	when is_float(T) {
		return 3 * n, 1 * n, 0 // 3*n work, n iwork, 0 rwork (per n and nrhs)
	} else when is_complex(T) {
		return 2 * n, 0, 1 * n // 2*n work, 0 iwork, n rwork (per n)
	}
}

// ===================================================================================
// CONVENIENCE FUNCTIONS
// ===================================================================================

// Solve single triangular system (single RHS vector)
tri_solve_vector :: proc(
	A: ^Triangular($T), // Triangular matrix
	b: ^Vector(T), // Vector (modified to solution)
	trans: TransposeMode = .None, // Transpose operation
) -> (
	info: Info,
	ok: bool,
) where is_float(T) || is_complex(T) {
	assert(int(b.len) == A.n, "Vector length must match matrix dimension")

	uplo_c := u8(A.uplo)
	trans_c := u8(trans)
	diag_c := u8(A.diag)
	n := A.n
	nrhs := Blas_Int(1)
	lda_blas := A.lda
	ldb_blas := max(1, A.n)
	incx := b.inc

	when T == f32 {
		lapack.strtrs_(&uplo_c, &trans_c, &diag_c, &n, &nrhs, raw_data(A.data), &lda_blas, raw_data(b.data), &ldb_blas, &info)
	} else when T == f64 {
		lapack.dtrtrs_(&uplo_c, &trans_c, &diag_c, &n, &nrhs, raw_data(A.data), &lda_blas, raw_data(b.data), &ldb_blas, &info)
	} else when T == complex64 {
		lapack.ctrtrs_(&uplo_c, &trans_c, &diag_c, &n, &nrhs, raw_data(A.data), &lda_blas, raw_data(b.data), &ldb_blas, &info)
	} else when T == complex128 {
		lapack.ztrtrs_(&uplo_c, &trans_c, &diag_c, &n, &nrhs, raw_data(A.data), &lda_blas, raw_data(b.data), &ldb_blas, &info)
	}

	return info, info == 0
}

// Check if triangular matrix is invertible (non-singular)
is_invertible_triangular :: proc(A: ^Triangular($T)) -> bool where is_float(T) || is_complex(T) {
	if A.diag == .Unit {
		return true // Unit triangular matrices are always invertible
	}

	// Check diagonal elements for zeros
	for i in 0 ..< A.n {
		diag_elem := A.data[i + i * A.lda]
		when is_complex(T) {
			if abs(diag_elem) == 0 {
				return false
			}
		} else {
			if diag_elem == 0 {
				return false
			}
		}
	}
	return true
}

// ===================================================================================
// BANDED TRIANGULAR SYSTEM SOLVING (TBTRS)
// ===================================================================================

// Solve banded triangular system Ax = b or AX = B
tri_solve_banded :: proc(
	AB: ^TriBand($T), // Triangular banded matrix
	B: ^Matrix(T), // RHS matrix/vectors (modified to solution)
	trans: TransposeMode = .None, // Transpose operation
) -> (
	info: Info,
	ok: bool,
) where is_float(T) || is_complex(T) {
	assert(int(B.rows) == int(AB.n), "B rows must match AB dimension")

	uplo_c := u8(AB.uplo)
	trans_c := u8(trans)
	diag_c := u8(AB.diag)
	nrhs := B.cols
	ldb_blas := B.ld

	when T == f32 {
		lapack.stbtrs_(&uplo_c, &trans_c, &diag_c, &AB.n, &AB.k, &nrhs, raw_data(AB.data), &AB.ldab, raw_data(B.data), &ldb_blas, &info)
	} else when T == f64 {
		lapack.dtbtrs_(&uplo_c, &trans_c, &diag_c, &AB.n, &AB.k, &nrhs, raw_data(AB.data), &AB.ldab, raw_data(B.data), &ldb_blas, &info)
	} else when T == complex64 {
		lapack.ctbtrs_(&uplo_c, &trans_c, &diag_c, &AB.n, &AB.k, &nrhs, raw_data(AB.data), &AB.ldab, raw_data(B.data), &ldb_blas, &info)
	} else when T == complex128 {
		lapack.ztbtrs_(&uplo_c, &trans_c, &diag_c, &AB.n, &AB.k, &nrhs, raw_data(AB.data), &AB.ldab, raw_data(B.data), &ldb_blas, &info)
	}

	return info, info == 0
}

// ===================================================================================
// BANDED TRIANGULAR CONDITION NUMBER ESTIMATION (TBCON)
// ===================================================================================

// Estimate condition number of banded triangular matrix
tri_condition_banded :: proc {
	tri_condition_banded_real,
	tri_condition_banded_complex,
}

// Real banded triangular condition number estimation (f32/f64) using TriBand
tri_condition_banded_real :: proc(
	AB: ^TriBand($T), // Triangular banded matrix
	work: []T, // Pre-allocated workspace (size 3*n)
	iwork: []Blas_Int, // Pre-allocated integer workspace (size n)
	norm: MatrixNorm = .OneNorm, // Norm type
) -> (
	rcond: T,
	info: Info,
	ok: bool,
) where is_float(T) {
	n := AB.n
	assert(len(work) >= 3 * int(n), "Workspace too small")
	assert(len(iwork) >= int(n), "Integer workspace too small")

	norm_c := u8(norm)
	uplo_c := u8(AB.uplo)
	diag_c := u8(AB.diag)

	when T == f32 {
		lapack.stbcon_(&norm_c, &uplo_c, &diag_c, &AB.n, &AB.k, raw_data(AB.data), &AB.ldab, &rcond, raw_data(work), raw_data(iwork), &info)
	} else when T == f64 {
		lapack.dtbcon_(&norm_c, &uplo_c, &diag_c, &AB.n, &AB.k, raw_data(AB.data), &AB.ldab, &rcond, raw_data(work), raw_data(iwork), &info)
	}

	return rcond, info, info == 0
}

// Complex banded triangular condition number estimation (complex64/complex128) using TriBand
tri_condition_banded_complex :: proc(
	AB: ^TriBand($Cmplx), // Triangular banded matrix
	work: []Cmplx, // Pre-allocated workspace (size 2*n)
	rwork: []$Real, // Pre-allocated real workspace (size n)
	norm: MatrixNorm = .OneNorm, // Norm type
) -> (
	rcond: Real,
	info: Info,
	ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
	n := AB.n
	assert(len(work) >= 2 * n, "Workspace too small")
	assert(len(rwork) >= n, "Real workspace too small")

	norm_c := u8(norm)
	uplo_c := u8(AB.uplo)
	diag_c := u8(AB.diag)

	when Cmplx == complex64 {
		lapack.ctbcon_(&norm_c, &uplo_c, &diag_c, &AB.n, &AB.k, raw_data(AB.data), &AB.ldab, &rcond, raw_data(work), raw_data(rwork), &info)
	} else when Cmplx == complex128 {
		lapack.ztbcon_(&norm_c, &uplo_c, &diag_c, &AB.n, &AB.k, raw_data(AB.data), &AB.ldab, &rcond, raw_data(work), raw_data(rwork), &info)
	}

	return rcond, info, info == 0
}

// ===================================================================================
// BANDED TRIANGULAR ITERATIVE REFINEMENT (TBRFS)
// ===================================================================================

// Iterative refinement for banded triangular systems
tri_refine_banded :: proc {
	tri_refine_banded_real,
	tri_refine_banded_complex,
}

// Real banded triangular iterative refinement (f32/f64)
tri_refine_banded_real :: proc(
	AB: ^TriBand($T), // Triangular banded matrix
	B: ^Matrix(T), // Original RHS
	X: ^Matrix(T), // Solution (refined on output)
	ferr: []T, // Pre-allocated forward error bounds (size nrhs)
	berr: []T, // Pre-allocated backward error bounds (size nrhs)
	work: []T, // Pre-allocated workspace (size 3*n)
	iwork: []Blas_Int, // Pre-allocated integer workspace (size n)
	trans: TransposeMode = .None, // Transpose operation
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	n := AB.n
	nrhs := B.cols
	assert(int(B.rows) == int(n), "B rows must match AB dimension")
	assert(int(X.rows) == int(n), "X rows must match AB dimension")
	assert(B.cols == X.cols, "B and X must have same number of columns")
	assert(len(ferr) >= int(nrhs), "Forward error array too small")
	assert(len(berr) >= int(nrhs), "Backward error array too small")
	assert(len(work) >= 3 * int(n), "Workspace too small")
	assert(len(iwork) >= int(n), "Integer workspace too small")

	uplo_c := u8(AB.uplo)
	trans_c := u8(trans)
	diag_c := u8(AB.diag)
	ldb_blas := B.ld
	ldx_blas := X.ld

	when T == f32 {
		lapack.stbrfs_(&uplo_c, &trans_c, &diag_c, &AB.n, &AB.k, &nrhs, raw_data(AB.data), &AB.ldab, raw_data(B.data), &ldb_blas, raw_data(X.data), &ldx_blas, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(iwork), &info)
	} else when T == f64 {
		lapack.dtbrfs_(&uplo_c, &trans_c, &diag_c, &AB.n, &AB.k, &nrhs, raw_data(AB.data), &AB.ldab, raw_data(B.data), &ldb_blas, raw_data(X.data), &ldx_blas, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(iwork), &info)
	}

	return info, info == 0
}

// Complex banded triangular iterative refinement (complex64/complex128)
tri_refine_banded_complex :: proc(
	AB: ^TriBand($Cmplx), // Triangular banded matrix
	B: ^Matrix(Cmplx), // Original RHS
	X: ^Matrix(Cmplx), // Solution (refined on output)
	ferr: []$Real, // Pre-allocated forward error bounds (size nrhs)
	berr: []Real, // Pre-allocated backward error bounds (size nrhs)
	work: []Cmplx, // Pre-allocated workspace (size 2*n)
	rwork: []Real, // Pre-allocated real workspace (size n)
	trans: TransposeMode = .None, // Transpose operation
) -> (
	info: Info,
	ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
	n := AB.n
	nrhs := B.cols
	assert(int(B.rows) == int(n), "B rows must match AB dimension")
	assert(int(X.rows) == int(n), "X rows must match AB dimension")
	assert(B.cols == X.cols, "B and X must have same number of columns")
	assert(len(ferr) >= int(nrhs), "Forward error array too small")
	assert(len(berr) >= int(nrhs), "Backward error array too small")
	assert(len(work) >= 2 * n, "Workspace too small")
	assert(len(rwork) >= n, "Real workspace too small")

	uplo_c := u8(AB.uplo)
	trans_c := u8(trans)
	diag_c := u8(AB.diag)
	ldb_blas := B.ld
	ldx_blas := X.ld

	when Cmplx == complex64 {
		lapack.ctbrfs_(&uplo_c, &trans_c, &diag_c, &AB.n, &AB.k, &nrhs, raw_data(AB.data), &AB.ldab, raw_data(B.data), &ldb_blas, raw_data(X.data), &ldx_blas, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(rwork), &info)
	} else when Cmplx == complex128 {
		lapack.ztbrfs_(&uplo_c, &trans_c, &diag_c, &AB.n, &AB.k, &nrhs, raw_data(AB.data), &AB.ldab, raw_data(B.data), &ldb_blas, raw_data(X.data), &ldx_blas, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(rwork), &info)
	}

	return info, info == 0
}

// ===================================================================================
// WORKSPACE QUERY FUNCTIONS FOR BANDED TRIANGULAR
// ===================================================================================

// Query workspace size for banded triangular condition estimation
query_workspace_tri_condition_banded :: proc(AB: ^TriBand($T)) -> (work_size: int, iwork_size: int, rwork_size: int) where is_float(T) || is_complex(T) {
	n := AB.n
	when is_float(T) {
		return 3 * n, 1 * n, 0 // 3*n work, n iwork, 0 rwork
	} else when is_complex(T) {
		return 2 * n, 0, 1 * n // 2*n work, 0 iwork, n rwork
	}
}

// Query workspace size for banded triangular refinement
query_workspace_tri_refine_banded :: proc(AB: ^TriBand($T)) -> (work_size: int, iwork_size: int, rwork_size: int) where is_float(T) || is_complex(T) {
	n := AB.n
	when is_float(T) {
		return 3 * n, 1 * n, 0 // 3*n work, n iwork, 0 rwork
	} else when is_complex(T) {
		return 2 * n, 0, 1 * n // 2*n work, 0 iwork, n rwork
	}
}
