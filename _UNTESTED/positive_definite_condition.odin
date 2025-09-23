package openblas

import lapack "./f77"
import "base:builtin"
import "core:math"
import "core:mem"

// ===================================================================================
// POSITIVE DEFINITE MATRIX CONDITION NUMBER ESTIMATION
// ===================================================================================

// Condition number estimation proc group
m_condition_positive_definite :: proc {
	m_condition_positive_definite_c64,
	m_condition_positive_definite_f64,
	m_condition_positive_definite_f32,
	m_condition_positive_definite_c128,
}

// ===================================================================================
// NORM TYPE PARAMETER
// ===================================================================================


// Convert norm type to LAPACK character
_norm_type_to_char :: proc(norm: NormType) -> cstring {
	switch norm {
	case .One:
		return "1"
	case .Infinity:
		return "I"
	case:
		return "1"
	}
}

// ===================================================================================
// CONDITION NUMBER RESULT STRUCTURE
// ===================================================================================

// Result of condition number estimation
ConditionResult :: struct {
	rcond:               f64, // Reciprocal condition number
	condition_number:    f64, // Actual condition number (1/rcond)
	is_well_conditioned: bool, // True if condition number < threshold
	is_singular:         bool, // True if matrix is numerically singular
}

// ===================================================================================
// CONDITION NUMBER ESTIMATION IMPLEMENTATION
// ===================================================================================

// Estimate condition number of positive definite matrix (c64)
// Requires matrix to be already factored using Cholesky factorization
m_condition_positive_definite_c64 :: proc(
	A: ^Matrix(complex64), // Factored matrix from Cholesky
	anorm: f32, // Norm of original matrix (before factorization)
	norm := NormType.One, // Type of norm to use
	uplo_upper := true, // Upper or lower triangular
	allocator := context.allocator,
) -> (
	result: ConditionResult,
	info: Info,
) {
	// Validate inputs
	if A.rows != A.cols {
		panic("Matrix must be square")
	}

	n := Blas_Int(A.rows)
	lda := Blas_Int(A.stride)
	uplo_c := "U" if uplo_upper else "L"
	norm_c := _norm_type_to_char(norm)

	// Allocate workspace
	work := make([]complex64, 2 * A.rows, allocator)
	defer delete(work)

	rwork := make([]f32, A.rows, allocator)
	defer delete(rwork)

	rcond: f32
	info_val: Info

	lapack.cpocon_(
		uplo_c,
		&n,
		cast(^complex64)A.data,
		&lda,
		&anorm,
		&rcond,
		raw_data(work),
		raw_data(rwork),
		&info_val,
		len(uplo_c),
	)

	// Build result
	result.rcond = f64(rcond)
	if rcond > 0 {
		result.condition_number = 1.0 / result.rcond
		result.is_well_conditioned = result.condition_number < 1e6
		result.is_singular = rcond < builtin.F32_EPSILON
	} else {
		result.condition_number = math.INF_F64
		result.is_well_conditioned = false
		result.is_singular = true
	}

	return result, info_val
}

// Estimate condition number of positive definite matrix (f64)
// Requires matrix to be already factored using Cholesky factorization
m_condition_positive_definite_f64 :: proc(
	A: ^Matrix(f64), // Factored matrix from Cholesky
	anorm: f64, // Norm of original matrix (before factorization)
	norm := NormType.One, // Type of norm to use
	uplo_upper := true, // Upper or lower triangular
	allocator := context.allocator,
) -> (
	result: ConditionResult,
	info: Info,
) {
	// Validate inputs
	if A.rows != A.cols {
		panic("Matrix must be square")
	}

	n := Blas_Int(A.rows)
	lda := Blas_Int(A.stride)
	uplo_c := "U" if uplo_upper else "L"
	norm_c := _norm_type_to_char(norm)

	// Allocate workspace
	work := make([]f64, 3 * A.rows, allocator)
	defer delete(work)

	iwork := make([]Blas_Int, A.rows, allocator)
	defer delete(iwork)

	rcond: f64
	info_val: Info

	lapack.dpocon_(
		uplo_c,
		&n,
		cast(^f64)A.data,
		&lda,
		&anorm,
		&rcond,
		raw_data(work),
		raw_data(iwork),
		&info_val,
		len(uplo_c),
	)

	// Build result
	result.rcond = rcond
	if rcond > 0 {
		result.condition_number = 1.0 / rcond
		result.is_well_conditioned = result.condition_number < 1e15
		result.is_singular = rcond < builtin.F64_EPSILON
	} else {
		result.condition_number = math.INF_F64
		result.is_well_conditioned = false
		result.is_singular = true
	}

	return result, info_val
}

// Estimate condition number of positive definite matrix (f32)
// Requires matrix to be already factored using Cholesky factorization
m_condition_positive_definite_f32 :: proc(
	A: ^Matrix(f32), // Factored matrix from Cholesky
	anorm: f32, // Norm of original matrix (before factorization)
	norm := NormType.One, // Type of norm to use
	uplo_upper := true, // Upper or lower triangular
	allocator := context.allocator,
) -> (
	result: ConditionResult,
	info: Info,
) {
	// Validate inputs
	if A.rows != A.cols {
		panic("Matrix must be square")
	}

	n := Blas_Int(A.rows)
	lda := Blas_Int(A.stride)
	uplo_c := "U" if uplo_upper else "L"
	norm_c := _norm_type_to_char(norm)

	// Allocate workspace
	work := make([]f32, 3 * A.rows, allocator)
	defer delete(work)

	iwork := make([]Blas_Int, A.rows, allocator)
	defer delete(iwork)

	rcond: f32
	info_val: Info

	lapack.spocon_(
		uplo_c,
		&n,
		cast(^f32)A.data,
		&lda,
		&anorm,
		&rcond,
		raw_data(work),
		raw_data(iwork),
		&info_val,
		len(uplo_c),
	)

	// Build result
	result.rcond = f64(rcond)
	if rcond > 0 {
		result.condition_number = 1.0 / result.rcond
		result.is_well_conditioned = result.condition_number < 1e6
		result.is_singular = rcond < builtin.F32_EPSILON
	} else {
		result.condition_number = math.INF_F64
		result.is_well_conditioned = false
		result.is_singular = true
	}

	return result, info_val
}

// Estimate condition number of positive definite matrix (c128)
// Requires matrix to be already factored using Cholesky factorization
m_condition_positive_definite_c128 :: proc(
	A: ^Matrix(complex128), // Factored matrix from Cholesky
	anorm: f64, // Norm of original matrix (before factorization)
	norm := NormType.One, // Type of norm to use
	uplo_upper := true, // Upper or lower triangular
	allocator := context.allocator,
) -> (
	result: ConditionResult,
	info: Info,
) {
	// Validate inputs
	if A.rows != A.cols {
		panic("Matrix must be square")
	}

	n := Blas_Int(A.rows)
	lda := Blas_Int(A.stride)
	uplo_c := "U" if uplo_upper else "L"
	norm_c := _norm_type_to_char(norm)

	// Allocate workspace
	work := make([]complex128, 2 * A.rows, allocator)
	defer delete(work)

	rwork := make([]f64, A.rows, allocator)
	defer delete(rwork)

	rcond: f64
	info_val: Info

	lapack.zpocon_(
		uplo_c,
		&n,
		cast(^complex128)A.data,
		&lda,
		&anorm,
		&rcond,
		raw_data(work),
		raw_data(rwork),
		&info_val,
		len(uplo_c),
	)

	// Build result
	result.rcond = rcond
	if rcond > 0 {
		result.condition_number = 1.0 / rcond
		result.is_well_conditioned = result.condition_number < 1e15
		result.is_singular = rcond < builtin.F64_EPSILON
	} else {
		result.condition_number = math.INF_F64
		result.is_well_conditioned = false
		result.is_singular = true
	}

	return result, info_val
}

// ===================================================================================
// CONVENIENCE FUNCTIONS
// ===================================================================================

// Complete workflow: factor and estimate condition number
estimate_condition_positive_definite :: proc(
	A: ^Matrix($T),
	norm := NormType.One,
	uplo_upper := true,
	allocator := context.allocator,
) -> (
	result: ConditionResult,
	success: bool,
) {
	// First compute the norm of the original matrix
	anorm := compute_matrix_norm(A, norm)

	// Create a copy for factorization
	A_copy := matrix_clone(A, allocator)
	defer matrix_delete(&A_copy)

	// Factor the matrix using Cholesky
	factor_success := cholesky_factor(&A_copy, uplo_upper, allocator)
	if !factor_success {
		return ConditionResult{}, false
	}

	// Estimate condition number
	when T == complex64 {
		result, info := m_condition_positive_definite_c64(
			&A_copy,
			f32(anorm),
			norm,
			uplo_upper,
			allocator,
		)
		return result, info == 0
	} else when T == f64 {
		result, info := m_condition_positive_definite_f64(
			&A_copy,
			anorm,
			norm,
			uplo_upper,
			allocator,
		)
		return result, info == 0
	} else when T == f32 {
		result, info := m_condition_positive_definite_f32(
			&A_copy,
			f32(anorm),
			norm,
			uplo_upper,
			allocator,
		)
		return result, info == 0
	} else when T == complex128 {
		result, info := m_condition_positive_definite_c128(
			&A_copy,
			anorm,
			norm,
			uplo_upper,
			allocator,
		)
		return result, info == 0
	} else {
		panic("Unsupported type for condition number estimation")
	}
}

// Check if matrix is well-conditioned
is_well_conditioned :: proc(
	A: ^Matrix($T),
	threshold := 1e6,
	allocator := context.allocator,
) -> bool {
	result, success := estimate_condition_positive_definite(A, .One, true, allocator)
	return success && result.condition_number < threshold
}

// Compute matrix norm helper (assumes matrix norm functions exist)
compute_matrix_norm :: proc(A: ^Matrix($T), norm: NormType) -> f64 {
	// This would call the appropriate matrix norm function
	// For now, return a placeholder
	return 1.0
}

// Cholesky factorization helper (assumes Cholesky functions exist)
cholesky_factor :: proc(A: ^Matrix($T), uplo_upper: bool, allocator: mem.Allocator) -> bool {
	// This would call the appropriate Cholesky factorization function
	// For now, return true as placeholder
	return true
}

// Matrix clone helper
matrix_clone :: proc(A: ^Matrix($T), allocator: mem.Allocator) -> Matrix(T) {
	B := create_matrix(T, A.rows, A.cols, allocator)
	for i in 0 ..< A.rows {
		for j in 0 ..< A.cols {
			matrix_set(&B, i, j, matrix_get(A, i, j))
		}
	}
	return B
}

// ===================================================================================
// CONDITION NUMBER ANALYSIS
// ===================================================================================

// Analyze matrix conditioning
analyze_conditioning :: proc(A: ^Matrix($T), allocator := context.allocator) -> ConditionAnalysis {
	analysis: ConditionAnalysis

	// Try different norms
	result_one, success_one := estimate_condition_positive_definite(A, .One, true, allocator)
	if success_one {
		analysis.condition_one = result_one.condition_number
	}

	result_inf, success_inf := estimate_condition_positive_definite(A, .Infinity, true, allocator)
	if success_inf {
		analysis.condition_inf = result_inf.condition_number
	}

	// Determine conditioning level
	min_condition := min(analysis.condition_one, analysis.condition_inf)
	if min_condition < 1e3 {
		analysis.level = .Excellent
	} else if min_condition < 1e6 {
		analysis.level = .Good
	} else if min_condition < 1e9 {
		analysis.level = .Fair
	} else if min_condition < 1e12 {
		analysis.level = .Poor
	} else {
		analysis.level = .IllConditioned
	}

	// Estimate relative error bound
	analysis.relative_error_bound = min_condition * builtin.F64_EPSILON

	return analysis
}

// Condition analysis structure
ConditionAnalysis :: struct {
	condition_one:        f64, // Condition number with 1-norm
	condition_inf:        f64, // Condition number with infinity norm
	level:                ConditionLevel, // Qualitative assessment
	relative_error_bound: f64, // Expected relative error in solution
}

// Condition level enumeration
ConditionLevel :: enum {
	Excellent, // κ < 10^3
	Good, // κ < 10^6
	Fair, // κ < 10^9
	Poor, // κ < 10^12
	IllConditioned, // κ >= 10^12
}

// Get recommended precision based on condition number
get_recommended_precision :: proc(condition_number: f64) -> PrecisionRecommendation {
	if condition_number < 1e3 {
		return .Float32
	} else if condition_number < 1e7 {
		return .Float64
	} else if condition_number < 1e15 {
		return .Float128
	} else {
		return .ExtendedPrecision
	}
}

PrecisionRecommendation :: enum {
	Float32,
	Float64,
	Float128,
	ExtendedPrecision,
}

// Check if iterative refinement is recommended
needs_iterative_refinement :: proc(condition_number: f64, required_accuracy: f64) -> bool {
	expected_error := condition_number * builtin.F64_EPSILON
	return expected_error > required_accuracy
}
