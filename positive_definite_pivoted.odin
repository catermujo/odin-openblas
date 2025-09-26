package openblas

import lapack "./f77"

// ===================================================================================
// POSITIVE DEFINITE CHOLESKY WITH PIVOTING
// ===================================================================================

// Pivoted Cholesky factorization proc group
m_cholesky_pivoted :: proc {
	m_cholesky_pivoted_f32_c64,
	m_cholesky_pivoted_f64_c128,
}

// ===================================================================================
// WORKSPACE QUERY FUNCTIONS
// ===================================================================================

// Query workspace for pivoted Cholesky factorization
query_workspace_pivoted_cholesky :: proc($T: typeid, n: int) -> (work_size: int, pivot_size: int) {
	// need work of size 2*n
	return 2 * n, n
}

// ===================================================================================
// PIVOTED CHOLESKY FACTORIZATION IMPLEMENTATION
// ===================================================================================

// Cholesky factorization with pivoting for f32/complex64
// Computes P^T * A * P = L * L^T (or L * L^H) with diagonal pivoting for rank revelation
m_cholesky_pivoted_f32_c64 :: proc(
	A: ^Matrix($T), // Matrix to factor (input/output)
	pivot: []Blas_Int, // Pre-allocated pivot array (output)
	work: []T, // Pre-allocated workspace
	tolerance: f32 = -1.0, // Tolerance for rank detection (-1 for default)
	uplo := MatrixRegion.Upper, // Upper or lower triangular
) -> (
	rank: int,
	tolerance_used: f32,
	info: Info,
	ok: bool,
) where T == f32 || T == complex64 {
	// Validate inputs
	assert(A.rows == A.cols, "Matrix must be square")
	assert(len(pivot) >= A.rows, "Insufficient pivot space")
	assert(len(work) >= 2 * A.rows, "Insufficient work space")
	assert(uplo == .Upper || uplo == .Lower, "uplo must be Upper or Lower")

	n := Blas_Int(A.rows)
	lda := Blas_Int(A.stride)
	uplo_c := matrix_region_to_cstring(uplo)

	rank_val: Blas_Int
	tol := tolerance

	when T == f32 {
		lapack.spstrf_(
			uplo_c,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(pivot),
			&rank_val,
			&tol,
			raw_data(work),
			&info,
			len(uplo_c),
		)
	} else when T == complex64 {
		lapack.cpstrf_(
			uplo_c,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(pivot),
			&rank_val,
			&tol,
			raw_data(work),
			&info,
			len(uplo_c),
		)
	}

	rank = int(rank_val)
	tolerance_used = tol
	ok = info >= 0 // info > 0 indicates rank deficiency, not an error

	return rank, tolerance_used, info, ok
}

// Cholesky factorization with pivoting for f64/complex128
// Computes P^T * A * P = L * L^T (or L * L^H) with diagonal pivoting for rank revelation
m_cholesky_pivoted_f64_c128 :: proc(
	A: ^Matrix($T), // Matrix to factor (input/output)
	pivot: []Blas_Int, // Pre-allocated pivot array (output)
	work: []T, // Pre-allocated workspace
	tolerance: f64 = -1.0, // Tolerance for rank detection (-1 for default)
	uplo := MatrixRegion.Upper, // Upper or lower triangular
) -> (
	rank: int,
	tolerance_used: f64,
	info: Info,
	ok: bool,
) where T == f64 || T == complex128 {
	// Validate inputs
	assert(A.rows == A.cols, "Matrix must be square")
	assert(len(pivot) >= A.rows, "Insufficient pivot space")
	assert(len(work) >= 2 * A.rows, "Insufficient work space")
	assert(uplo == .Upper || uplo == .Lower, "uplo must be Upper or Lower")

	n := Blas_Int(A.rows)
	lda := Blas_Int(A.stride)
	uplo_c := matrix_region_to_cstring(uplo)

	rank_val: Blas_Int
	tol := tolerance

	when T == f64 {
		lapack.dpstrf_(
			uplo_c,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(pivot),
			&rank_val,
			&tol,
			raw_data(work),
			&info,
			len(uplo_c),
		)
	} else when T == complex128 {
		lapack.zpstrf_(
			uplo_c,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(pivot),
			&rank_val,
			&tol,
			raw_data(work),
			&info,
			len(uplo_c),
		)
	}

	rank = int(rank_val)
	tolerance_used = tol
	ok = info >= 0 // info > 0 indicates rank deficiency, not an error

	return rank, tolerance_used, info, ok
}
