package openblas

import lapack "./f77"

// ===================================================================================
// POSITIVE DEFINITE MATRIX CONDITION NUMBER ESTIMATION
// ===================================================================================

// Condition number estimation proc group
m_condition_positive_definite :: proc {
	m_condition_positive_definite_f32_c64,
	m_condition_positive_definite_f64_c128,
}

// ===================================================================================
// WORKSPACE QUERY FUNCTIONS
// ===================================================================================

// Query workspace for condition number estimation
query_workspace_condition_positive_definite :: proc(
	$T: typeid,
	n: int,
) -> (
	work_size: int,
	iwork_size: int,
	rwork_size: int,
) {
	when is_float(T) {
		return 3 * n, n, 0 // Real types need work and iwork
	} else when T == complex64 || T == complex128 {
		return 2 * n, 0, n // Complex types need work and rwork, no iwork
	}
}

// ===================================================================================
// CONDITION NUMBER ESTIMATION IMPLEMENTATION
// ===================================================================================

// Estimate condition number of positive definite matrix (f32/complex64)
// Requires matrix to be already factored using Cholesky factorization
m_condition_positive_definite_f32_c64 :: proc(
	A: ^Matrix($T), // Factored matrix from Cholesky
	anorm: f32, // Norm of original matrix (before factorization)
	work: []T, // Workspace (pre-allocated, size from query function)
	iwork: []Blas_Int = nil, // Integer workspace for f32 (nil for complex64)
	rwork: []f32 = nil, // Real workspace for complex64 (nil for f32)
	uplo := MatrixRegion.Upper, // Upper or lower triangular
) -> (
	rcond: f32,
	info: Info,
	ok: bool,// Reciprocal condition number
) where T == f32 || T == complex64 {
	// Validate inputs
	assert(A.rows == A.cols, "Matrix must be square")
	assert(uplo == .Upper || uplo == .Lower, "uplo must be Upper or Lower")

	n := Blas_Int(A.rows)
	lda := Blas_Int(A.stride)
	uplo_c := matrix_region_to_cstring(uplo)

	when T == f32 {
		assert(len(work) >= 3 * int(n), "Insufficient work space")
		assert(len(iwork) >= int(n), "Insufficient iwork space")
		lapack.spocon_(
			uplo_c,
			&n,
			raw_data(A.data),
			&lda,
			&anorm,
			&rcond,
			raw_data(work),
			raw_data(iwork),
			&info,
			len(uplo_c),
		)
	} else when T == complex64 {
		assert(len(work) >= 2 * int(n), "Insufficient work space")
		assert(len(rwork) >= int(n), "Insufficient rwork space")
		lapack.cpocon_(
			uplo_c,
			&n,
			raw_data(A.data),
			&lda,
			&anorm,
			&rcond,
			raw_data(work),
			raw_data(rwork),
			&info,
			len(uplo_c),
		)
	}

	return rcond, info, info == 0
}

// Estimate condition number of positive definite matrix (f64/complex128)
// Requires matrix to be already factored using Cholesky factorization
m_condition_positive_definite_f64_c128 :: proc(
	A: ^Matrix($T), // Factored matrix from Cholesky
	anorm: f64, // Norm of original matrix (before factorization)
	work: []T, // Workspace (pre-allocated, size from query function)
	iwork: []Blas_Int = nil, // Integer workspace for f64 (nil for complex128)
	rwork: []f64 = nil, // Real workspace for complex128 (nil for f64)
	uplo := MatrixRegion.Upper, // Upper or lower triangular
) -> (
	rcond: f64,
	info: Info,
	ok: bool,// Reciprocal condition number
) where T == f64 || T == complex128 {
	// Validate inputs
	assert(A.rows == A.cols, "Matrix must be square")
	assert(uplo == .Upper || uplo == .Lower, "uplo must be Upper or Lower")

	n := Blas_Int(A.rows)
	lda := Blas_Int(A.stride)
	uplo_c := matrix_region_to_cstring(uplo)

	when T == f64 {
		assert(len(work) >= 3 * int(n), "Insufficient work space")
		assert(len(iwork) >= int(n), "Insufficient iwork space")
		lapack.dpocon_(
			uplo_c,
			&n,
			raw_data(A.data),
			&lda,
			&anorm,
			&rcond,
			raw_data(work),
			raw_data(iwork),
			&info,
			len(uplo_c),
		)
	} else when T == complex128 {
		assert(len(work) >= 2 * int(n), "Insufficient work space")
		assert(len(rwork) >= int(n), "Insufficient rwork space")
		lapack.zpocon_(
			uplo_c,
			&n,
			raw_data(A.data),
			&lda,
			&anorm,
			&rcond,
			raw_data(work),
			raw_data(rwork),
			&info,
			len(uplo_c),
		)
	}

	return rcond, info, info == 0
}

// ===================================================================================
// CONVENIENCE FUNCTIONS
// ===================================================================================

// Check if a reciprocal condition number indicates good conditioning
is_well_conditioned_rcond :: proc(rcond: $T, threshold: T) -> bool where is_float(T) {
	// rcond close to 1 is well-conditioned, close to 0 is ill-conditioned
	return rcond > threshold
}

// Convert reciprocal condition number to condition number
rcond_to_condition_number :: proc(rcond: $T) -> T where is_float(T) {
	if rcond > 0 {
		return 1.0 / rcond
	} else {
		// Return infinity for singular matrices
		when T == f32 {
			return max(f32)
		} else {
			return max(f64)
		}
	}
}

// Check if matrix is numerically singular based on rcond
is_singular_rcond :: proc(rcond: $T) -> bool where is_float(T) {
	when T == f32 {
		return rcond < F32_EPSILON
	} else {
		return rcond < F64_EPSILON
	}
}

// ===================================================================================
// CONDITION LEVEL ASSESSMENT
// ===================================================================================

// Condition level enumeration
ConditionLevel :: enum {
	Excellent, // κ < 10^3
	Good, // κ < 10^6
	Fair, // κ < 10^9
	Poor, // κ < 10^12
	IllConditioned, // κ >= 10^12
}

// Assess conditioning level based on condition number
assess_condition_level :: proc(condition_number: $T) -> ConditionLevel where is_float(T) {
	if condition_number < 1e3 {
		return .Excellent
	} else if condition_number < 1e6 {
		return .Good
	} else if condition_number < 1e9 {
		return .Fair
	} else if condition_number < 1e12 {
		return .Poor
	} else {
		return .IllConditioned
	}
}

// Assess conditioning level based on reciprocal condition number
assess_condition_level_from_rcond :: proc(rcond: $T) -> ConditionLevel where is_float(T) {
	condition_number := rcond_to_condition_number(rcond)
	return assess_condition_level(condition_number)
}

// Estimate relative error bound based on condition number
estimate_relative_error_bound :: proc(condition_number: $T) -> T where is_float(T) {
	when T == f32 {
		return condition_number * F32_EPSILON
	} else {
		return condition_number * F64_EPSILON
	}
}

// Check if iterative refinement is recommended
needs_iterative_refinement :: proc(
	rcond: $T,
	required_accuracy: T,
) -> bool where T == f32 ||
	T == f64 {
	condition_number := rcond_to_condition_number(rcond)
	expected_error := estimate_relative_error_bound(condition_number)
	return expected_error > required_accuracy
}
