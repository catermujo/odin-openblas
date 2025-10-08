package openblas

import lapack "./f77"
import "base:builtin"
import "base:intrinsics"
import "core:strings"

// ===================================================================================
// AUXILIARY AND UTILITY FUNCTIONS
// Helper routines for matrix operations, norms, and utilities
// ===================================================================================

get_lapack_version :: proc() -> (major, minor, patch: Blas_Int) {
	lapack.ilaver_(&major, &minor, &patch)
	return
}

// ===================================================================================
// CONDITION NUMBER ESTIMATION
// Estimate reciprocal condition number of factored matrices
// ===================================================================================

// Estimate reciprocal condition number of general matrix
// Matrix A must be factored (e.g., by LU decomposition)
// Query workspace for condition estimation
query_workspace_dns_condition_estimate :: proc($T: typeid, n: int) -> (work_size: int, iwork_size: int, rwork_size: int) where is_float(T) || is_complex(T) {
	when is_float(T) {
		// Real types: work = 4*n, iwork = n, no rwork
		work_size = 4 * n
		iwork_size = n
		rwork_size = 0
	} else {
		// Complex types: work = 2*n, no iwork, rwork = 2*n
		work_size = 2 * n
		iwork_size = 0
		rwork_size = 2 * n
	}
	return
}

// Estimate condition number for f32/f64
dns_condition_estimate_real :: proc(
	A: ^Matrix($T), // Factored matrix (from getrf)
	anorm: T, // Norm of original matrix
	work: []T, // Pre-allocated workspace (size 4*n)
	iwork: []Blas_Int, // Pre-allocated integer workspace (size n)
	norm: MatrixNorm = .OneNorm,
) -> (
	rcond: T,
	info: Info,
	ok: bool,
) where is_float(T) {
	n := A.co
	lda := A.ld
	norm_c := cast(u8)norm
	anorm := anorm
	assert(len(work) >= i64(4 * n), "Workspace too small")
	assert(len(iwork) >= i64(n), "Integer workspace too small")


	when T == f32 {
		lapack.sgecon_(&norm_c, &n, raw_data(A.data), &lda, &anorm, &rcond, raw_data(work), raw_data(iwork), &info, 1)
	} else when T == f64 {
		lapack.dgecon_(&norm_c, &n, raw_data(A.data), &lda, &anorm, &rcond, raw_data(work), raw_data(iwork), &info, 1)
	}

	return rcond, info, info == 0
}

// Estimate condition number for complex64/complex128
dns_condition_estimate_complex :: proc(
	A: ^Matrix($Cmplx), // Factored matrix (from getrf)
	anorm: $Real, // Norm of original matrix
	work: []Cmplx, // Pre-allocated workspace (size 2*n)
	rwork: []Real, // Pre-allocated real workspace (size 2*n)
	norm: MatrixNorm = .OneNorm,
) -> (
	rcond: Real,
	info: Info,
	ok: bool,
) where (Cmplx == complex64 && R == f32) || (Cmplx == complex128 && R == f64) {
	n := A.cols
	assert(len(work) >= i64(2 * n), "Workspace too small")
	assert(len(rwork) >= i64(2 * n), "Real workspace too small")
	lda := A.ld
	norm_c := cast(u8)norm
	anorm := anorm

	when Cmplx == complex64 {
		lapack.cgecon_(&norm_c, &n, raw_data(A.data), &lda, &anorm, &rcond, raw_data(work), raw_data(rwork), &info, 1)
	} else when Cmplx == complex128 {
		lapack.zgecon_(&norm_c, &n, raw_data(A.data), &lda, &anorm, &rcond, raw_data(work), raw_data(rwork), &info, 1)
	}
	return rcond, info, info == 0
}

dns_condition_estimate :: proc {
	dns_condition_estimate_real,
	dns_condition_estimate_complex,
}

// Helper function to check if matrix is well-conditioned
is_well_conditioned :: proc(rcond: $T, threshold: T) -> bool where is_float(T) {
	return rcond > threshold
}

// ===================================================================================
// MATRIX EQUILIBRATION
// Compute row and column scale factors to improve matrix conditioning
// ===================================================================================

// Compute row and column scale factors for general matrix equilibration
// Scale factors R and C are chosen so that R*A*C has rows and columns with similar norms

// Compute equilibration scale factors for f32/f64
// rowcnd: Ratio of smallest to largest row scale
// colcnd: Ratio of smallest to largest column scale
// amax: Absolute value of largest matrix element
dns_equilibrate_real :: proc(
	A: ^Matrix($T), // Matrix to equilibrate
	R: []T, // Pre-allocated row scale factors (size m)
	C: []T, // Pre-allocated column scale factors (size n)
) -> (
	rowcnd: T,
	colcnd: T,
	amax: T,
	info: Info,
	ok: bool,
) where is_float(T) {
	m := A.rows
	n := A.cols
	lda := A.ld
	assert(len(R) >= i64(m), "Row scale factors array too small")
	assert(len(C) >= i64(n), "Column scale factors array too small")

	when T == f32 {
		lapack.sgeequ_(&m, &n, raw_data(A.data), &lda, raw_data(R), raw_data(C), &rowcnd, &colcnd, &amax, &info)
	} else when T == f64 {
		lapack.dgeequ_(&m, &n, raw_data(A.data), &lda, raw_data(R), raw_data(C), &rowcnd, &colcnd, &amax, &info)
	}

	return rowcnd, colcnd, amax, info, info == ok
}

// Compute equilibration scale factors for complex64/complex128
dns_equilibrate_complex :: proc(
	A: ^Matrix($Cmplx), // Matrix to equilibrate
	R: []$Real, // Pre-allocated row scale factors (size m)
	C: []Real, // Pre-allocated column scale factors (size n)
) -> (
	rowcnd: Real,
	colcnd: Real,
	amax: Real,
	info: Info,
	ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
	m := A.rows
	n := A.cols
	lda := A.ld

	assert(len(R) >= i64(m), "Row scale factors array too small")
	assert(len(C) >= i64(n), "Column scale factors array too small")

	when Cmplx == complex64 {
		lapack.cgeequ_(&m, &n, raw_data(A.data), &lda, raw_data(R), raw_data(C), &rowcnd, &colcnd, &amax, &info)
	} else when Cmplx == complex128 {
		lapack.zgeequ_(&m, &n, raw_data(A.data), &lda, raw_data(R), raw_data(C), &rowcnd, &colcnd, &amax, &info)
	}

	return rowcnd, colcnd, amax, info, info == ok
}

// Procedure group for equilibration
dns_equilibrate :: proc {
	dns_equilibrate_real,
	dns_equilibrate_complex,
}

// Improved equilibration with better algorithm (LAPACK 3.x)
// More robust handling of over/underflow

// Compute improved equilibration scale factors for f32/f64
dns_equilibrate_improved_real :: proc(
	A: ^Matrix($T), // Matrix to equilibrate
	R: []T, // Pre-allocated row scale factors (size m)
	C: []T, // Pre-allocated column scale factors (size n)
) -> (
	rowcnd: T,
	colcnd: T,
	amax: T,
	info: Info,
	ok: bool,
) where is_float(T) {
	m := A.rows
	n := A.cols
	lda := A.ld

	assert(len(R) >= i64(m), "Row scale factors array too small")
	assert(len(C) >= i64(n), "Column scale factors array too small")

	when T == f32 {
		lapack.sgeequb_(&m, &n, raw_data(A.data), &lda, raw_data(R), raw_data(C), &rowcnd, &colcnd, &amax, &info)
	} else when T == f64 {
		lapack.dgeequb_(&m, &n, raw_data(A.data), &lda, raw_data(R), raw_data(C), &rowcnd, &colcnd, &amax, &info)
	}

	return rowcnd, colcnd, amax, info, info == ok
}

// Compute improved equilibration scale factors for complex64/complex128
dns_equilibrate_improved_complex :: proc(
	A: ^Matrix($Cmplx), // Matrix to equilibrate
	R: []$Real, // Pre-allocated row scale factors (size m)
	C: []Real, // Pre-allocated column scale factors (size n)
) -> (
	rowcnd: Real,
	colcnd: Real,
	amax: Real,
	info: Info,
	ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
	m := A.rows
	n := A.cols
	lda := A.ld

	assert(len(R) >= i64(m), "Row scale factors array too small")
	assert(len(C) >= i64(n), "Column scale factors array too small")

	when Cmplx == complex64 {
		lapack.cgeequb_(&m, &n, raw_data(A.data), &lda, raw_data(R), raw_data(C), &rowcnd, &colcnd, &amax, &info)
	} else when Cmplx == complex128 {
		lapack.zgeequb_(&m, &n, raw_data(A.data), &lda, raw_data(R), raw_data(C), &rowcnd, &colcnd, &amax, &info)
	}

	return rowcnd, colcnd, amax, info, info == ok
}

// Procedure group for improved equilibration
dns_equilibrate_improved :: proc {
	dns_equilibrate_improved_real,
	dns_equilibrate_improved_complex,
}

// Apply equilibration scale factors to a matrix
// Computes A_scaled = R * A * C where R and C are diagonal scaling matrices
dns_apply_equilibration_real :: proc(
	A: ^Matrix($T),
	R: []T, // Row scale factors (from dns_equilibrate)
	C: []T, // Column scale factors (from dns_equilibrate)
) where is_float(T) {
	m := A.rows
	n := A.cols
	// FIXME: LAPACK SCALE??
	// Scale rows
	for i in 0 ..< m {
		for j in 0 ..< n {
			// Column-major indexing
			A.data[j * A.ld + i] *= R[i]
		}
	}

	// Scale columns
	for j in 0 ..< n {
		for i in 0 ..< m {
			// Column-major indexing
			A.data[j * A.ld + i] *= C[j]
		}
	}
}

dns_apply_equilibration_complex :: proc(
	A: ^Matrix($Cmplx),
	R: []$Real, // Real row scale factors (from dns_equilibrate)
	C: []Real, // Real column scale factors (from dns_equilibrate)
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
	m := A.rows
	n := A.cols
	// FIXME: LAPACK SCALE??
	// Scale rows
	for i in 0 ..< m {
		row_scale := T(complex(R[i], 0))
		for j in 0 ..< n {
			// Column-major indexing
			A.data[j * A.ld + i] *= row_scale
		}
	}

	// Scale columns
	for j in 0 ..< n {
		col_scale := T(complex(C[j], 0))
		for i in 0 ..< m {
			// Column-major indexing
			A.data[j * A.ld + i] *= col_scale
		}
	}
}

// Check if matrix needs equilibration
// Returns true if equilibration would significantly improve conditioning
// default threshold 0.1
dns_needs_equilibration :: proc(rowcnd, colcnd: $T, threshold: T) -> bool where is_float(T) {
	return rowcnd < threshold || colcnd < threshold
}

// ===================================================================================
// MATRIX COPY OPERATIONS (LACPY)
// ===================================================================================

// Copy matrix or submatrix - unified generic implementation
dns_copy_matrix :: proc(
	A: ^Matrix($T), // Source matrix
	B: ^Matrix(T), // Destination matrix
	region := MatrixRegion.Full, // Region to copy (Full, Upper, Lower)
) where is_float(T) || is_complex(T) {
	assert(A.rows == B.rows && A.cols == B.cols, "Matrix dimensions must match")

	m := A.rows
	n := A.cols
	lda := A.ld
	ldb := B.ld
	uplo_c := cast(u8)region

	when T == f32 {
		lapack.slacpy_(&uplo_c, &m, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, 1)
	} else when T == f64 {
		lapack.dlacpy_(&uplo_c, &m, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, 1)
	} else when T == complex64 {
		lapack.clacpy_(&uplo_c, &m, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, 1)
	} else when T == complex128 {
		lapack.zlacpy_(&uplo_c, &m, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, 1)
	}
}

// ===================================================================================
// COMPLEX VECTOR CONJUGATION (LACGV)
// ===================================================================================

// Conjugate complex vector (c64)
// Computes X[i] = conj(X[i]) for i = 0, incx, 2*incx, ..., (n-1)*incx
v_conjugate :: proc(X: []$Cmplx, incx: int = 1) where is_complex(Cmplx) {
	if len(X) == 0 {return}

	n := Blas_Int(len(X))
	incx_int := Blas_Int(incx)

	when Cmplx == complex64 {
		lapack.clacgv_(&n, raw_data(X), &incx_int)
	} else when Cmplx == complex128 {
		lapack.zlacgv_(&n, raw_data(X), &incx_int)
	}
}

// ===================================================================================
// COMPLEX × REAL MATRIX MULTIPLICATION (LACRM)
// ===================================================================================

// Complex × real matrix multiplication (c64)
// Computes C = A * B where A is complex(m×n), B is real(n×n), C is complex(m×n)
// rwork must be at least size 2*m*n
dns_multiply_complex_real_complex :: proc(
	A: ^Matrix($Cmplx), // Complex matrix (m × n)
	B: ^Matrix($Real), // Real matrix (n × n)
	C: ^Matrix(Cmplx), // Complex result (m × n)
	rwork: []Real, // Real workspace (size >= 2*m*n)
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
	m := A.rows
	n := A.cols
	lda := A.ld
	ldb := B.ld
	ldc := C.ld

	assert(B.rows == n && B.cols == n, "B must be n×n")
	assert(C.rows == m && C.cols == n, "C must be m×n")
	assert(len(rwork) >= int(2 * m * n), "rwork array too small")

	when Cmplx == complex64 {
		lapack.clacrm_(&m, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(C.data), &ldc, raw_data(rwork))
	} else when Cmplx == complex128 {
		lapack.zlacrm_(&m, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(C.data), &ldc, raw_data(rwork))
	}
}


// ===================================================================================
// REAL × COMPLEX MATRIX MULTIPLICATION (LARCM)
// ===================================================================================

// Real × complex matrix multiplication (c64)
// Computes C = A * B where A is real(m×m), B is complex(m×n), C is complex(m×n)
// rwork must be at least size 2*m*n
dns_multiply_real_complex :: proc(
	A: ^Matrix($Real), // Real matrix (m × m)
	B: ^Matrix($Cmplx), // Complex matrix (m × n)
	C: ^Matrix(Cmplx), // Complex result (m × n)
	rwork: []Real, // Real workspace (size >= 2*m*n)
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
	m := A.rows
	n := B.cols
	assert(A.cols == m, "A must be m×m")
	assert(B.rows == m, "B must be m×n")
	assert(C.rows == m && C.cols == n, "C must be m×n")
	assert(len(rwork) >= int(2 * m * n), "rwork array too small")

	lda := A.ld
	ldb := B.ld
	ldc := C.ld

	when Cmplx == complex64 {
		lapack.clarcm_(&m, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(C.data), &ldc, raw_data(rwork))
	} else when Cmplx == complex128 {
		lapack.zlarcm_(&m, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(C.data), &ldc, raw_data(rwork))
	}
}

// ===================================================================================
// RECIPROCAL CONDITION NUMBERS FOR EIGENVALUES/SINGULAR VALUES (DISNA)
// ===================================================================================

// Job type for DISNA
DisnaJob :: enum u8 {
	Eigenvalues   = 'E', // Compute condition numbers for eigenvalues
	SingularLeft  = 'L', // Compute condition numbers for left singular vectors
	SingularRight = 'R', // Compute condition numbers for right singular vectors
}

// Compute reciprocal condition numbers for eigenvalues/singular values - unified generic implementation
// For eigenvalue problems: SEP[i] = |lambda[i] - lambda[i+1]| (gap between eigenvalues)
// For singular value problems: SEP[i] = min(D[i], D[i-1]) (minimum singular value gap)
compute_reciprocal_condition_numbers :: proc(
	D: []$T, // Eigenvalues or singular values
	SEP: []T, // Reciprocal condition numbers (output)
	job: DisnaJob = .Eigenvalues,
	m: int = 0, // Rows (for SVD, 0 for eigenvalue)
	n: int = 0, // Cols (for SVD, 0 for eigenvalue)
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	// For eigenvalue problems, m and n are not used (set to 0)
	// For SVD, m and n are matrix dimensions

	if job == .Eigenvalues {
		// For eigenvalue problems, m is not used, but n should be size of D
		n = Blas_Int(len(D))
	}

	assert(len(SEP) >= len(D), "SEP array too small")

	job_c := cast(u8)job

	when T == f32 {
		lapack.sdisna_(&job_c, &m, &n, raw_data(D), raw_data(SEP), &info, 1)
	} else when T == f64 {
		lapack.ddisna_(&job_c, &m, &n, raw_data(D), raw_data(SEP), &info, 1)
	}

	return info, info == 0
}
