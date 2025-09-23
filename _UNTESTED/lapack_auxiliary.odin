package openblas

import lapack "./f77"
import "base:builtin"
import "base:intrinsics"
import "core:strings"

// ===================================================================================
// AUXILIARY AND UTILITY FUNCTIONS
// Helper routines for matrix operations, norms, and utilities
// ===================================================================================

// ===================================================================================
// CONDITION NUMBER ESTIMATION
// Estimate reciprocal condition number of factored matrices
// ===================================================================================

// Norm type for condition number estimation
NormType :: enum {
	One, // 1-norm (maximum column sum)
	Infinity, // Infinity norm (maximum row sum)
	Frobenius, // Frobenius norm (not used for condition number)
}

// Estimate reciprocal condition number of general matrix
// Matrix A must be factored (e.g., by LU decomposition)
m_condition_estimate :: proc {
	m_condition_estimate_real,
	m_condition_estimate_c64,
	m_condition_estimate_c128,
}

m_condition_estimate_real :: proc(
	A: ^Matrix($T), // Factored matrix (from getrf)
	anorm: T, // Norm of original matrix
	norm: NormType = .One,
	allocator := context.allocator,
) -> (
	rcond: T,
	info: Info, // Reciprocal condition number
) where T == f32 || T == f64 {
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)

	// Convert norm type to string
	norm_c: cstring
	switch norm {
	case .One:
		norm_c = cstring("1")
	case .Infinity:
		norm_c = cstring("I")
	case .Frobenius:
		norm_c = cstring("F") // Not typically used for condition
	}

	// Allocate workspace
	work := builtin.make([]T, 4 * n, allocator)
	iwork := builtin.make([]i32, n, allocator)
	defer builtin.delete(work)
	defer builtin.delete(iwork)

	anorm_copy := anorm

	when T == f32 {
		lapack.sgecon_(
			norm_c,
			&n,
			raw_data(A.data),
			&lda,
			&anorm_copy,
			&rcond,
			raw_data(work),
			raw_data(iwork),
			&info,
			1,
		)
	} else when T == f64 {
		lapack.dgecon_(
			norm_c,
			&n,
			raw_data(A.data),
			&lda,
			&anorm_copy,
			&rcond,
			raw_data(work),
			raw_data(iwork),
			&info,
			1,
		)
	}

	return rcond, info
}

m_condition_estimate_c64 :: proc(
	A: ^Matrix(complex64),
	anorm: f32,
	norm: NormType = .One,
	allocator := context.allocator,
) -> (
	rcond: f32,
	info: Info,
) {
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)

	// Convert norm type to string
	norm_c: cstring
	switch norm {
	case .One:
		norm_c = cstring("1")
	case .Infinity:
		norm_c = cstring("I")
	case .Frobenius:
		norm_c = cstring("F")
	}

	// Allocate workspace
	work := builtin.make([]complex64, 2 * n, allocator)
	rwork := builtin.make([]f32, 2 * n, allocator)
	defer builtin.delete(work)
	defer builtin.delete(rwork)

	anorm_copy := anorm

	lapack.cgecon_(
		norm_c,
		&n,
		raw_data(A.data),
		&lda,
		&anorm_copy,
		&rcond,
		raw_data(work),
		raw_data(rwork),
		&info,
		1,
	)

	return rcond, info
}

m_condition_estimate_c128 :: proc(
	A: ^Matrix(complex128),
	anorm: f64,
	norm: NormType = .One,
	allocator := context.allocator,
) -> (
	rcond: f64,
	info: Info,
) {
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)

	// Convert norm type to string
	norm_c: cstring
	switch norm {
	case .One:
		norm_c = cstring("1")
	case .Infinity:
		norm_c = cstring("I")
	case .Frobenius:
		norm_c = cstring("F")
	}

	// Allocate workspace
	work := builtin.make([]complex128, 2 * n, allocator)
	rwork := builtin.make([]f64, 2 * n, allocator)
	defer builtin.delete(work)
	defer builtin.delete(rwork)

	anorm_copy := anorm

	lapack.zgecon_(
		norm_c,
		&n,
		raw_data(A.data),
		&lda,
		&anorm_copy,
		&rcond,
		raw_data(work),
		raw_data(rwork),
		&info,
		1,
	)

	return rcond, info
}

// Helper function to check if matrix is well-conditioned
// Returns true if reciprocal condition number > threshold (default 1E-6)
m_is_well_conditioned :: proc(rcond: $T, threshold: T) -> bool where T == f32 || T == f64 {
	return rcond > threshold
}

// ===================================================================================
// MATRIX EQUILIBRATION
// Compute row and column scale factors to improve matrix conditioning
// ===================================================================================

// Compute row and column scale factors for general matrix equilibration
// Scale factors R and C are chosen so that R*A*C has rows and columns with similar norms
m_equilibrate :: proc {
	m_equilibrate_real,
	m_equilibrate_c64,
	m_equilibrate_c128,
}

m_equilibrate_real :: proc(
	A: ^Matrix($T),
	allocator := context.allocator,
) -> (
	R: []T,
	C: []T,
	rowcnd: T,
	colcnd: T,
	amax: T,
	info: Info, // Row scale factors// Column scale factors// Ratio of smallest to largest row scale// Ratio of smallest to largest column scale// Absolute value of largest matrix element
) where T == f32 || T == f64 {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)

	// Allocate scale factors
	R = builtin.make([]T, m, allocator)
	C = builtin.make([]T, n, allocator)

	when T == f32 {
		lapack.sgeequ_(
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(R),
			raw_data(C),
			&rowcnd,
			&colcnd,
			&amax,
			&info,
		)
	} else when T == f64 {
		lapack.dgeequ_(
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(R),
			raw_data(C),
			&rowcnd,
			&colcnd,
			&amax,
			&info,
		)
	}

	return R, C, rowcnd, colcnd, amax, info
}

m_equilibrate_c64 :: proc(
	A: ^Matrix(complex64),
	allocator := context.allocator,
) -> (
	R: []f32,
	C: []f32,
	rowcnd: f32,
	colcnd: f32,
	amax: f32,
	info: Info, // Row scale factors (real)// Column scale factors (real)// Ratio of smallest to largest row scale// Ratio of smallest to largest column scale// Absolute value of largest matrix element
) {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)

	// Allocate scale factors
	R = builtin.make([]f32, m, allocator)
	C = builtin.make([]f32, n, allocator)

	lapack.cgeequ_(
		&m,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(R),
		raw_data(C),
		&rowcnd,
		&colcnd,
		&amax,
		&info,
	)

	return R, C, rowcnd, colcnd, amax, info
}

m_equilibrate_c128 :: proc(
	A: ^Matrix(complex128),
	allocator := context.allocator,
) -> (
	R: []f64,
	C: []f64,
	rowcnd: f64,
	colcnd: f64,
	amax: f64,
	info: Info, // Row scale factors (real)// Column scale factors (real)// Ratio of smallest to largest row scale// Ratio of smallest to largest column scale// Absolute value of largest matrix element
) {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)

	// Allocate scale factors
	R = builtin.make([]f64, m, allocator)
	C = builtin.make([]f64, n, allocator)

	lapack.zgeequ_(
		&m,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(R),
		raw_data(C),
		&rowcnd,
		&colcnd,
		&amax,
		&info,
	)

	return R, C, rowcnd, colcnd, amax, info
}

// Improved equilibration with better algorithm (LAPACK 3.x)
// More robust handling of over/underflow
m_equilibrate_improved :: proc {
	m_equilibrate_improved_real,
	m_equilibrate_improved_c64,
	m_equilibrate_improved_c128,
}

m_equilibrate_improved_real :: proc(
	A: ^Matrix($T),
	allocator := context.allocator,
) -> (
	R: []T,
	C: []T,
	rowcnd: T,
	colcnd: T,
	amax: T,
	info: Info, // Row scale factors// Column scale factors// Ratio of smallest to largest row scale// Ratio of smallest to largest column scale// Absolute value of largest matrix element
) where T == f32 || T == f64 {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)

	// Allocate scale factors
	R = builtin.make([]T, m, allocator)
	C = builtin.make([]T, n, allocator)

	when T == f32 {
		lapack.sgeequb_(
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(R),
			raw_data(C),
			&rowcnd,
			&colcnd,
			&amax,
			&info,
		)
	} else when T == f64 {
		lapack.dgeequb_(
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(R),
			raw_data(C),
			&rowcnd,
			&colcnd,
			&amax,
			&info,
		)
	}

	return R, C, rowcnd, colcnd, amax, info
}

m_equilibrate_improved_c64 :: proc(
	A: ^Matrix(complex64),
	allocator := context.allocator,
) -> (
	R: []f32,
	C: []f32,
	rowcnd: f32,
	colcnd: f32,
	amax: f32,
	info: Info, // Row scale factors (real)// Column scale factors (real)// Ratio of smallest to largest row scale// Ratio of smallest to largest column scale// Absolute value of largest matrix element
) {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)

	// Allocate scale factors
	R = builtin.make([]f32, m, allocator)
	C = builtin.make([]f32, n, allocator)

	lapack.cgeequb_(
		&m,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(R),
		raw_data(C),
		&rowcnd,
		&colcnd,
		&amax,
		&info,
	)

	return R, C, rowcnd, colcnd, amax, info
}

m_equilibrate_improved_c128 :: proc(
	A: ^Matrix(complex128),
	allocator := context.allocator,
) -> (
	R: []f64,
	C: []f64,
	rowcnd: f64,
	colcnd: f64,
	amax: f64,
	info: Info, // Row scale factors (real)// Column scale factors (real)// Ratio of smallest to largest row scale// Ratio of smallest to largest column scale// Absolute value of largest matrix element
) {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)

	// Allocate scale factors
	R = builtin.make([]f64, m, allocator)
	C = builtin.make([]f64, n, allocator)

	lapack.zgeequb_(
		&m,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(R),
		raw_data(C),
		&rowcnd,
		&colcnd,
		&amax,
		&info,
	)

	return R, C, rowcnd, colcnd, amax, info
}

// Apply equilibration scale factors to a matrix
// Computes A_scaled = R * A * C where R and C are diagonal scaling matrices
m_apply_equilibration_real :: proc(
	A: ^Matrix($T),
	R: []T, // Row scale factors (from m_equilibrate)
	C: []T, // Column scale factors (from m_equilibrate)
) where T == f32 || T == f64 {
	m := A.rows
	n := A.cols

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

m_apply_equilibration_complex :: proc(
	A: ^Matrix($T),
	R: []$S, // Real row scale factors (from m_equilibrate)
	C: []S, // Real column scale factors (from m_equilibrate)
) where (T == complex64 && S == f32) || (T == complex128 && S == f64) {
	m := A.rows
	n := A.cols

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
m_needs_equilibration :: proc(
	rowcnd, colcnd: $T,
	threshold: T,
) -> bool where T == f32 ||
	T == f64 {
	return rowcnd < threshold || colcnd < threshold
}

// ===================================================================================
