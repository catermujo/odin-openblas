package openblas

import lapack "./f77"
import "base:builtin"
import "core:mem"

// ===================================================================================
// POSITIVE DEFINITE BANDED MATRIX CONDITION NUMBER ESTIMATION AND EQUILIBRATION
// ===================================================================================

// Estimate condition number of positive definite banded matrix proc group
m_estimate_condition_banded_pd :: proc {
	m_estimate_condition_banded_pd_c64,
	m_estimate_condition_banded_pd_f64,
	m_estimate_condition_banded_pd_f32,
	m_estimate_condition_banded_pd_c128,
}

// Compute equilibration scaling for positive definite banded matrix proc group
m_compute_equilibration_banded_pd :: proc {
	m_compute_equilibration_banded_pd_c64,
	m_compute_equilibration_banded_pd_f64,
	m_compute_equilibration_banded_pd_f32,
	m_compute_equilibration_banded_pd_c128,
}

// ===================================================================================
// CONDITION NUMBER ESTIMATION IMPLEMENTATION
// ===================================================================================

// Estimate condition number of positive definite banded matrix (c64)
// Estimates reciprocal condition number using factorization from CPBTRF
m_estimate_condition_banded_pd_c64 :: proc(
	AB: ^Matrix(complex64), // Banded matrix factorization from CPBTRF
	kd: int, // Number of super/sub-diagonals
	anorm: f32, // 1-norm of original matrix
	uplo_upper := true, // Upper or lower triangular storage
	allocator := context.allocator,
) -> (
	rcond: f32,
	success: bool,
	info: Info,
) {
	// Validate inputs
	if len(AB.data) == 0 {
		panic("Matrix cannot be empty")
	}
	if AB.rows != AB.cols {
		panic("Matrix must be square")
	}
	if kd < 0 || kd >= AB.rows {
		panic("Invalid bandwidth kd")
	}
	if anorm < 0 {
		panic("anorm must be non-negative")
	}

	uplo_c := "U" if uplo_upper else "L"
	n := Blas_Int(AB.cols)
	kd_val := Blas_Int(kd)
	ldab := Blas_Int(AB.ld)
	anorm_val := anorm
	rcond_val: f32
	info_val: Info

	// Allocate workspace
	work := make([]complex64, 2 * n, context.temp_allocator)
	rwork := make([]f32, n, context.temp_allocator)

	lapack.cpbcon_(
		uplo_c,
		&n,
		&kd_val,
		raw_data(AB.data),
		&ldab,
		&anorm_val,
		&rcond_val,
		raw_data(work),
		raw_data(rwork),
		&info_val,
		len(uplo_c),
	)

	return rcond_val, info_val == 0, info_val
}

// Estimate condition number of positive definite banded matrix (f64)
// Estimates reciprocal condition number using factorization from DPBTRF
m_estimate_condition_banded_pd_f64 :: proc(
	AB: ^Matrix(f64), // Banded matrix factorization from DPBTRF
	kd: int, // Number of super/sub-diagonals
	anorm: f64, // 1-norm of original matrix
	uplo_upper := true, // Upper or lower triangular storage
	allocator := context.allocator,
) -> (
	rcond: f64,
	success: bool,
	info: Info,
) {
	// Validate inputs
	if len(AB.data) == 0 {
		panic("Matrix cannot be empty")
	}
	if AB.rows != AB.cols {
		panic("Matrix must be square")
	}
	if kd < 0 || kd >= AB.rows {
		panic("Invalid bandwidth kd")
	}
	if anorm < 0 {
		panic("anorm must be non-negative")
	}

	uplo_c := "U" if uplo_upper else "L"
	n := Blas_Int(AB.cols)
	kd_val := Blas_Int(kd)
	ldab := Blas_Int(AB.ld)
	anorm_val := anorm
	rcond_val: f64
	info_val: Info

	// Allocate workspace
	work := make([]f64, 3 * n, context.temp_allocator)
	iwork := make([]Blas_Int, n, context.temp_allocator)

	lapack.dpbcon_(
		uplo_c,
		&n,
		&kd_val,
		raw_data(AB.data),
		&ldab,
		&anorm_val,
		&rcond_val,
		raw_data(work),
		raw_data(iwork),
		&info_val,
		len(uplo_c),
	)

	return rcond_val, info_val == 0, info_val
}

// Estimate condition number of positive definite banded matrix (f32)
// Estimates reciprocal condition number using factorization from SPBTRF
m_estimate_condition_banded_pd_f32 :: proc(
	AB: ^Matrix(f32), // Banded matrix factorization from SPBTRF
	kd: int, // Number of super/sub-diagonals
	anorm: f32, // 1-norm of original matrix
	uplo_upper := true, // Upper or lower triangular storage
	allocator := context.allocator,
) -> (
	rcond: f32,
	success: bool,
	info: Info,
) {
	// Validate inputs
	if len(AB.data) == 0 {
		panic("Matrix cannot be empty")
	}
	if AB.rows != AB.cols {
		panic("Matrix must be square")
	}
	if kd < 0 || kd >= AB.rows {
		panic("Invalid bandwidth kd")
	}
	if anorm < 0 {
		panic("anorm must be non-negative")
	}

	uplo_c := "U" if uplo_upper else "L"
	n := Blas_Int(AB.cols)
	kd_val := Blas_Int(kd)
	ldab := Blas_Int(AB.ld)
	anorm_val := anorm
	rcond_val: f32
	info_val: Info

	// Allocate workspace
	work := make([]f32, 3 * n, context.temp_allocator)
	iwork := make([]Blas_Int, n, context.temp_allocator)

	lapack.spbcon_(
		uplo_c,
		&n,
		&kd_val,
		raw_data(AB.data),
		&ldab,
		&anorm_val,
		&rcond_val,
		raw_data(work),
		raw_data(iwork),
		&info_val,
		len(uplo_c),
	)

	return rcond_val, info_val == 0, info_val
}

// Estimate condition number of positive definite banded matrix (c128)
// Estimates reciprocal condition number using factorization from ZPBTRF
m_estimate_condition_banded_pd_c128 :: proc(
	AB: ^Matrix(complex128), // Banded matrix factorization from ZPBTRF
	kd: int, // Number of super/sub-diagonals
	anorm: f64, // 1-norm of original matrix
	uplo_upper := true, // Upper or lower triangular storage
	allocator := context.allocator,
) -> (
	rcond: f64,
	success: bool,
	info: Info,
) {
	// Validate inputs
	if len(AB.data) == 0 {
		panic("Matrix cannot be empty")
	}
	if AB.rows != AB.cols {
		panic("Matrix must be square")
	}
	if kd < 0 || kd >= AB.rows {
		panic("Invalid bandwidth kd")
	}
	if anorm < 0 {
		panic("anorm must be non-negative")
	}

	uplo_c := "U" if uplo_upper else "L"
	n := Blas_Int(AB.cols)
	kd_val := Blas_Int(kd)
	ldab := Blas_Int(AB.ld)
	anorm_val := anorm
	rcond_val: f64
	info_val: Info

	// Allocate workspace
	work := make([]complex128, 2 * n, context.temp_allocator)
	rwork := make([]f64, n, context.temp_allocator)

	lapack.zpbcon_(
		uplo_c,
		&n,
		&kd_val,
		raw_data(AB.data),
		&ldab,
		&anorm_val,
		&rcond_val,
		raw_data(work),
		raw_data(rwork),
		&info_val,
		len(uplo_c),
	)

	return rcond_val, info_val == 0, info_val
}

// ===================================================================================
// CONVENIENCE FUNCTIONS
// ===================================================================================

// Check if positive definite banded matrix is well-conditioned
is_well_conditioned_banded :: proc(
	AB: ^Matrix($T),
	kd: int,
	anorm: auto,
	tolerance := 1e-10,
	uplo_upper := true,
	allocator := context.allocator,
) -> (
	well_conditioned: bool,
	condition_number: f64,
) {
	when T == complex64 {
		rcond, success, _ := m_estimate_condition_banded_pd_c64(
			AB,
			kd,
			f32(anorm),
			uplo_upper,
			allocator,
		)
		if !success || rcond == 0 {
			return false, 0
		}
		cond := 1.0 / f64(rcond)
		return rcond > tolerance, cond
	} else when T == f64 {
		rcond, success, _ := m_estimate_condition_banded_pd_f64(
			AB,
			kd,
			f64(anorm),
			uplo_upper,
			allocator,
		)
		if !success || rcond == 0 {
			return false, 0
		}
		cond := 1.0 / rcond
		return rcond > tolerance, cond
	} else when T == f32 {
		rcond, success, _ := m_estimate_condition_banded_pd_f32(
			AB,
			kd,
			f32(anorm),
			uplo_upper,
			allocator,
		)
		if !success || rcond == 0 {
			return false, 0
		}
		cond := 1.0 / f64(rcond)
		return rcond > tolerance, cond
	} else when T == complex128 {
		rcond, success, _ := m_estimate_condition_banded_pd_c128(
			AB,
			kd,
			f64(anorm),
			uplo_upper,
			allocator,
		)
		if !success || rcond == 0 {
			return false, 0
		}
		cond := 1.0 / rcond
		return rcond > tolerance, cond
	} else {
		panic("Unsupported type for condition number estimation")
	}
}

// Get condition number from reciprocal condition number
get_condition_number_from_rcond :: proc(rcond: $T) -> T {
	if rcond == 0 {
		// Matrix is singular or nearly singular
		when T == f32 {
			return 1e30 // Very large number for f32
		} else {
			return 1e300 // Very large number for f64
		}
	}
	return T(1) / rcond
}

// Estimate condition number with automatic norm computation
estimate_condition_banded_with_norm :: proc(
	AB_original: ^Matrix($T), // Original matrix before factorization
	AB_factored: ^Matrix(T), // Factorized matrix from PBTRF
	kd: int, // Bandwidth
	uplo_upper := true,
	allocator := context.allocator,
) -> (
	condition_number: f64,
	success: bool,
) {
	// Compute 1-norm of original matrix
	// Note: This is a simplified computation - actual implementation
	// should use appropriate LAPACK norm routines for banded matrices
	norm := compute_banded_matrix_norm(AB_original, kd, uplo_upper)

	when T == complex64 {
		rcond, success, _ := m_estimate_condition_banded_pd_c64(
			AB_factored,
			kd,
			f32(norm),
			uplo_upper,
			allocator,
		)
		if !success || rcond == 0 {
			return 0, false
		}
		return 1.0 / f64(rcond), true
	} else when T == f64 {
		rcond, success, _ := m_estimate_condition_banded_pd_f64(
			AB_factored,
			kd,
			norm,
			uplo_upper,
			allocator,
		)
		if !success || rcond == 0 {
			return 0, false
		}
		return 1.0 / rcond, true
	} else when T == f32 {
		rcond, success, _ := m_estimate_condition_banded_pd_f32(
			AB_factored,
			kd,
			f32(norm),
			uplo_upper,
			allocator,
		)
		if !success || rcond == 0 {
			return 0, false
		}
		return 1.0 / f64(rcond), true
	} else when T == complex128 {
		rcond, success, _ := m_estimate_condition_banded_pd_c128(
			AB_factored,
			kd,
			norm,
			uplo_upper,
			allocator,
		)
		if !success || rcond == 0 {
			return 0, false
		}
		return 1.0 / rcond, true
	} else {
		panic("Unsupported type for condition number estimation")
	}
}

// Helper function to compute 1-norm of banded matrix
compute_banded_matrix_norm :: proc(AB: ^Matrix($T), kd: int, uplo_upper: bool) -> f64 {
	// Simplified 1-norm computation for banded matrices
	// In production, use appropriate LAPACK routines
	max_col_sum := T(0)

	for j in 0 ..< AB.cols {
		col_sum := T(0)

		// Determine row range based on bandwidth
		row_start := max(0, j - kd)
		row_end := min(AB.rows, j + kd + 1)

		for i in row_start ..< row_end {
			when T == complex64 || T == complex128 {
				col_sum += abs(matrix_get(AB, i, j))
			} else {
				col_sum += abs(matrix_get(AB, i, j))
			}
		}

		if col_sum > max_col_sum {
			max_col_sum = col_sum
		}
	}

	when T == complex64 || T == complex128 {
		return f64(real(max_col_sum))
	} else {
		return f64(max_col_sum)
	}
}

// Check matrix conditioning before solving
check_conditioning_before_solve :: proc(
	AB_factored: ^Matrix($T),
	kd: int,
	anorm: auto,
	warning_threshold := 1e-6,
	error_threshold := 1e-10,
	uplo_upper := true,
	allocator := context.allocator,
) -> (
	status: enum {
		Good,
		Warning,
		Error,
	},
	condition_number: f64,
) {
	well_conditioned, cond := is_well_conditioned_banded(
		AB_factored,
		kd,
		anorm,
		error_threshold,
		uplo_upper,
		allocator,
	)

	if !well_conditioned {
		return .Error, cond
	}

	// Check against warning threshold
	when T == complex64 {
		rcond, _, _ := m_estimate_condition_banded_pd_c64(
			AB_factored,
			kd,
			f32(anorm),
			uplo_upper,
			allocator,
		)
		if f64(rcond) < warning_threshold {
			return .Warning, cond
		}
	} else when T == f64 {
		rcond, _, _ := m_estimate_condition_banded_pd_f64(
			AB_factored,
			kd,
			f64(anorm),
			uplo_upper,
			allocator,
		)
		if rcond < warning_threshold {
			return .Warning, cond
		}
	} else when T == f32 {
		rcond, _, _ := m_estimate_condition_banded_pd_f32(
			AB_factored,
			kd,
			f32(anorm),
			uplo_upper,
			allocator,
		)
		if f64(rcond) < warning_threshold {
			return .Warning, cond
		}
	} else when T == complex128 {
		rcond, _, _ := m_estimate_condition_banded_pd_c128(
			AB_factored,
			kd,
			f64(anorm),
			uplo_upper,
			allocator,
		)
		if rcond < warning_threshold {
			return .Warning, cond
		}
	}

	return .Good, cond
}

// Utility functions
max :: proc(a, b: int) -> int {
	return a if a > b else b
}

min :: proc(a, b: int) -> int {
	return a if a < b else b
}

abs :: proc(x: $T) -> T {
	when T == complex64 || T == complex128 {
		return abs_complex(x)
	} else {
		return x if x >= 0 else -x
	}
}

abs_complex :: proc(z: $T) -> auto {
	when T == complex64 {
		return f32(sqrt(f64(real(z) * real(z) + imag(z) * imag(z))))
	} else when T == complex128 {
		return sqrt(real(z) * real(z) + imag(z) * imag(z))
	}
}

real :: proc(z: $T) -> auto {
	when T == complex64 {
		// Extract real part - implementation specific
		return f32(0) // Placeholder
	} else when T == complex128 {
		// Extract real part - implementation specific
		return f64(0) // Placeholder
	}
}

imag :: proc(z: $T) -> auto {
	when T == complex64 {
		// Extract imaginary part - implementation specific
		return f32(0) // Placeholder
	} else when T == complex128 {
		// Extract imaginary part - implementation specific
		return f64(0) // Placeholder
	}
}

sqrt :: proc(x: f64) -> f64 {
	// Square root implementation
	return x // Placeholder
}

// ===================================================================================
// POSITIVE DEFINITE BANDED MATRIX EQUILIBRATION
// ===================================================================================

// Equilibration result structure
EquilibrationResult :: struct($T: typeid) {
	S:       []T, // Scaling factors
	scond:   T, // Ratio of smallest to largest scaling factor
	amax:    T, // Maximum absolute value in matrix
	success: bool, // Whether equilibration succeeded
	info:    Blas_Int,
}

// Compute equilibration scaling for positive definite banded matrix (c64)
// Computes scaling factors to improve conditioning
m_compute_equilibration_banded_pd_c64 :: proc(
	AB: ^Matrix(complex64), // Positive definite banded matrix
	kd: int, // Number of super/sub-diagonals
	uplo_upper := true, // Upper or lower triangular storage
	allocator := context.allocator,
) -> EquilibrationResult(f32) {
	// Validate inputs
	if len(AB.data) == 0 {
		panic("Matrix cannot be empty")
	}
	if AB.rows != AB.cols {
		panic("Matrix must be square")
	}
	if kd < 0 || kd >= AB.rows {
		panic("Invalid bandwidth kd")
	}

	uplo_c := "U" if uplo_upper else "L"
	n := Blas_Int(AB.cols)
	kd_val := Blas_Int(kd)
	ldab := Blas_Int(AB.ld)

	// Allocate output arrays
	S := make([]f32, n, allocator)
	scond: f32
	amax: f32
	info_val: Info

	lapack.cpbequ_(
		uplo_c,
		&n,
		&kd_val,
		raw_data(AB.data),
		&ldab,
		raw_data(S),
		&scond,
		&amax,
		&info_val,
		len(uplo_c),
	)

	return EquilibrationResult(f32) {
		S = S,
		scond = scond,
		amax = amax,
		success = info_val == 0,
		info = info_val,
	}
}

// Compute equilibration scaling for positive definite banded matrix (f64)
// Computes scaling factors to improve conditioning
m_compute_equilibration_banded_pd_f64 :: proc(
	AB: ^Matrix(f64), // Positive definite banded matrix
	kd: int, // Number of super/sub-diagonals
	uplo_upper := true, // Upper or lower triangular storage
	allocator := context.allocator,
) -> EquilibrationResult(f64) {
	// Validate inputs
	if len(AB.data) == 0 {
		panic("Matrix cannot be empty")
	}
	if AB.rows != AB.cols {
		panic("Matrix must be square")
	}
	if kd < 0 || kd >= AB.rows {
		panic("Invalid bandwidth kd")
	}

	uplo_c := "U" if uplo_upper else "L"
	n := Blas_Int(AB.cols)
	kd_val := Blas_Int(kd)
	ldab := Blas_Int(AB.ld)

	// Allocate output arrays
	S := make([]f64, n, allocator)
	scond: f64
	amax: f64
	info_val: Info

	lapack.dpbequ_(
		uplo_c,
		&n,
		&kd_val,
		raw_data(AB.data),
		&ldab,
		raw_data(S),
		&scond,
		&amax,
		&info_val,
		len(uplo_c),
	)

	return EquilibrationResult(f64) {
		S = S,
		scond = scond,
		amax = amax,
		success = info_val == 0,
		info = info_val,
	}
}

// Compute equilibration scaling for positive definite banded matrix (f32)
// Computes scaling factors to improve conditioning
m_compute_equilibration_banded_pd_f32 :: proc(
	AB: ^Matrix(f32), // Positive definite banded matrix
	kd: int, // Number of super/sub-diagonals
	uplo_upper := true, // Upper or lower triangular storage
	allocator := context.allocator,
) -> EquilibrationResult(f32) {
	// Validate inputs
	if len(AB.data) == 0 {
		panic("Matrix cannot be empty")
	}
	if AB.rows != AB.cols {
		panic("Matrix must be square")
	}
	if kd < 0 || kd >= AB.rows {
		panic("Invalid bandwidth kd")
	}

	uplo_c := "U" if uplo_upper else "L"
	n := Blas_Int(AB.cols)
	kd_val := Blas_Int(kd)
	ldab := Blas_Int(AB.ld)

	// Allocate output arrays
	S := make([]f32, n, allocator)
	scond: f32
	amax: f32
	info_val: Info

	lapack.spbequ_(
		uplo_c,
		&n,
		&kd_val,
		raw_data(AB.data),
		&ldab,
		raw_data(S),
		&scond,
		&amax,
		&info_val,
		len(uplo_c),
	)

	return EquilibrationResult(f32) {
		S = S,
		scond = scond,
		amax = amax,
		success = info_val == 0,
		info = info_val,
	}
}

// Compute equilibration scaling for positive definite banded matrix (c128)
// Computes scaling factors to improve conditioning
m_compute_equilibration_banded_pd_c128 :: proc(
	AB: ^Matrix(complex128), // Positive definite banded matrix
	kd: int, // Number of super/sub-diagonals
	uplo_upper := true, // Upper or lower triangular storage
	allocator := context.allocator,
) -> EquilibrationResult(f64) {
	// Validate inputs
	if len(AB.data) == 0 {
		panic("Matrix cannot be empty")
	}
	if AB.rows != AB.cols {
		panic("Matrix must be square")
	}
	if kd < 0 || kd >= AB.rows {
		panic("Invalid bandwidth kd")
	}

	uplo_c := "U" if uplo_upper else "L"
	n := Blas_Int(AB.cols)
	kd_val := Blas_Int(kd)
	ldab := Blas_Int(AB.ld)

	// Allocate output arrays
	S := make([]f64, n, allocator)
	scond: f64
	amax: f64
	info_val: Info

	lapack.zpbequ_(
		uplo_c,
		&n,
		&kd_val,
		raw_data(AB.data),
		&ldab,
		raw_data(S),
		&scond,
		&amax,
		&info_val,
		len(uplo_c),
	)

	return EquilibrationResult(f64) {
		S = S,
		scond = scond,
		amax = amax,
		success = info_val == 0,
		info = info_val,
	}
}

// ===================================================================================
// EQUILIBRATION CONVENIENCE FUNCTIONS
// ===================================================================================

// Check if equilibration is needed based on scaling factor condition
needs_equilibration :: proc(
	result: EquilibrationResult($T),
	threshold := 0.1, // Default threshold for scond
) -> bool {
	return result.success && result.scond < T(threshold)
}

// Apply equilibration scaling to matrix and vector
apply_equilibration_scaling :: proc(
	AB: ^Matrix($T), // Matrix to scale (modified in-place)
	b: []T, // Right-hand side vector (modified in-place)
	S: []auto, // Scaling factors from equilibration
	kd: int, // Bandwidth
	uplo_upper := true,
) {
	n := AB.cols

	// Scale matrix: A_scaled = S * A * S
	for j in 0 ..< n {
		// Determine row range based on bandwidth
		row_start := max(0, j - kd)
		row_end := min(n, j + kd + 1)

		for i in row_start ..< row_end {
			old_val := matrix_get(AB, i, j)
			when T == complex64 {
				new_val := old_val * complex64(S[i] * S[j])
			} else when T == complex128 {
				new_val := old_val * complex128(S[i] * S[j])
			} else {
				new_val := old_val * T(S[i] * S[j])
			}
			matrix_set(AB, i, j, new_val)
		}
	}

	// Scale right-hand side: b_scaled = S * b
	for i in 0 ..< len(b) {
		when T == complex64 {
			b[i] *= complex64(S[i])
		} else when T == complex128 {
			b[i] *= complex128(S[i])
		} else {
			b[i] *= T(S[i])
		}
	}
}

// Undo equilibration scaling on solution vector
undo_equilibration_scaling :: proc(
	x: []$T, // Solution vector (modified in-place)
	S: []auto, // Scaling factors from equilibration
) {
	// Scale solution: x_original = S * x_scaled
	for i in 0 ..< len(x) {
		when T == complex64 {
			x[i] *= complex64(S[i])
		} else when T == complex128 {
			x[i] *= complex128(S[i])
		} else {
			x[i] *= T(S[i])
		}
	}
}

// Complete equilibration workflow for solving systems
equilibrate_and_prepare_system :: proc(
	AB: ^Matrix($T), // Matrix (will be modified if equilibration needed)
	b: []T, // RHS vector (will be modified if equilibration needed)
	kd: int, // Bandwidth
	uplo_upper := true,
	equilibration_threshold := 0.1,
	allocator := context.allocator,
) -> (
	equilibration_applied: bool,
	scaling: EquilibrationResult(auto),
) {
	// Compute equilibration factors
	when T == complex64 {
		scaling := m_compute_equilibration_banded_pd_c64(AB, kd, uplo_upper, allocator)
	} else when T == f64 {
		scaling := m_compute_equilibration_banded_pd_f64(AB, kd, uplo_upper, allocator)
	} else when T == f32 {
		scaling := m_compute_equilibration_banded_pd_f32(AB, kd, uplo_upper, allocator)
	} else when T == complex128 {
		scaling := m_compute_equilibration_banded_pd_c128(AB, kd, uplo_upper, allocator)
	} else {
		panic("Unsupported type for equilibration")
	}

	if !scaling.success {
		return false, scaling
	}

	// Check if equilibration is needed
	if needs_equilibration(scaling, equilibration_threshold) {
		// Apply scaling
		apply_equilibration_scaling(AB, b, scaling.S, kd, uplo_upper)
		return true, scaling
	}

	return false, scaling
}

// Compute improved condition number after equilibration
estimate_equilibrated_condition :: proc(
	original_rcond: $T,
	equilibration_result: EquilibrationResult(T),
) -> T {
	if !equilibration_result.success || equilibration_result.scond == 0 {
		return original_rcond
	}

	// Estimate improvement in condition number
	// The equilibrated matrix typically has better conditioning
	improvement_factor := T(1) / equilibration_result.scond
	return min(T(1), original_rcond * improvement_factor)
}

// Delete equilibration result
delete_equilibration_result :: proc(result: ^EquilibrationResult($T)) {
	if result.S != nil {
		delete(result.S)
	}
}
