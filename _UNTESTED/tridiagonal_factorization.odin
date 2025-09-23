package openblas

import lapack "./f77"
import "core:c"
import "core:math"
import "core:mem"
import "core:slice"

// ============================================================================
// TRIDIAGONAL POSITIVE DEFINITE FACTORIZATION (Cholesky for tridiagonal)
// ============================================================================
// These routines compute the factorization of a symmetric positive definite
// tridiagonal matrix A = L * D * L^T or A = U * D * U^H for complex

// Tridiagonal factorization result
TridiagonalFactorizationResult :: struct {
	is_positive_definite: bool, // True if factorization succeeded
	pivot_position:       int, // Position where non-positive pivot found (if failed)
	min_diagonal:         f64, // Minimum diagonal element after factorization
	max_diagonal:         f64, // Maximum diagonal element after factorization
	condition_estimate:   f64, // Rough condition number estimate (max/min diagonal)
}

// Complex single precision tridiagonal factorization
cpttrf :: proc(
	n: int,
	d: []f32, // Diagonal elements (real, modified on output)
	e: []complex64, // Off-diagonal elements (modified on output)
) -> (
	result: TridiagonalFactorizationResult,
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(len(d) >= n, "Diagonal array too small")
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")

	n_int := Blas_Int(n)
	info_int: Info

	lapack.cpttrf_(&n_int, raw_data(d), cast(^lapack.complex)raw_data(e), &info_int)

	info = Info(info_int)

	// Fill result
	result.is_positive_definite = info == .OK
	if info > 0 {
		result.pivot_position = int(info)
	}

	// Compute min/max diagonal for condition estimate
	if n > 0 && info == .OK {
		result.min_diagonal = f64(d[0])
		result.max_diagonal = f64(d[0])
		for i in 1 ..< n {
			result.min_diagonal = min(result.min_diagonal, f64(d[i]))
			result.max_diagonal = max(result.max_diagonal, f64(d[i]))
		}
		if result.min_diagonal > 0 {
			result.condition_estimate = result.max_diagonal / result.min_diagonal
		} else {
			result.condition_estimate = math.INF_F64
		}
	}

	return
}

// Double precision real tridiagonal factorization
dpttrf :: proc(
	n: int,
	d: []f64, // Diagonal elements (modified on output)
	e: []f64, // Off-diagonal elements (modified on output)
) -> (
	result: TridiagonalFactorizationResult,
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(len(d) >= n, "Diagonal array too small")
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")

	n_int := Blas_Int(n)
	info_int: Info

	lapack.dpttrf_(&n_int, raw_data(d), raw_data(e), &info_int)

	info = Info(info_int)

	// Fill result
	result.is_positive_definite = info == .OK
	if info > 0 {
		result.pivot_position = int(info)
	}

	// Compute min/max diagonal for condition estimate
	if n > 0 && info == .OK {
		result.min_diagonal = d[0]
		result.max_diagonal = d[0]
		for i in 1 ..< n {
			result.min_diagonal = min(result.min_diagonal, d[i])
			result.max_diagonal = max(result.max_diagonal, d[i])
		}
		if result.min_diagonal > 0 {
			result.condition_estimate = result.max_diagonal / result.min_diagonal
		} else {
			result.condition_estimate = math.INF_F64
		}
	}

	return
}

// Single precision real tridiagonal factorization
spttrf :: proc(
	n: int,
	d: []f32, // Diagonal elements (modified on output)
	e: []f32, // Off-diagonal elements (modified on output)
) -> (
	result: TridiagonalFactorizationResult,
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(len(d) >= n, "Diagonal array too small")
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")

	n_int := Blas_Int(n)
	info_int: Info

	lapack.spttrf_(&n_int, raw_data(d), raw_data(e), &info_int)

	info = Info(info_int)

	// Fill result
	result.is_positive_definite = info == .OK
	if info > 0 {
		result.pivot_position = int(info)
	}

	// Compute min/max diagonal for condition estimate
	if n > 0 && info == .OK {
		result.min_diagonal = f64(d[0])
		result.max_diagonal = f64(d[0])
		for i in 1 ..< n {
			result.min_diagonal = min(result.min_diagonal, f64(d[i]))
			result.max_diagonal = max(result.max_diagonal, f64(d[i]))
		}
		if result.min_diagonal > 0 {
			result.condition_estimate = result.max_diagonal / result.min_diagonal
		} else {
			result.condition_estimate = math.INF_F64
		}
	}

	return
}

// Complex double precision tridiagonal factorization
zpttrf :: proc(
	n: int,
	d: []f64, // Diagonal elements (real, modified on output)
	e: []complex128, // Off-diagonal elements (modified on output)
) -> (
	result: TridiagonalFactorizationResult,
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(len(d) >= n, "Diagonal array too small")
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")

	n_int := Blas_Int(n)
	info_int: Info

	lapack.zpttrf_(&n_int, raw_data(d), cast(^lapack.doublecomplex)raw_data(e), &info_int)

	info = Info(info_int)

	// Fill result
	result.is_positive_definite = info == .OK
	if info > 0 {
		result.pivot_position = int(info)
	}

	// Compute min/max diagonal for condition estimate
	if n > 0 && info == .OK {
		result.min_diagonal = d[0]
		result.max_diagonal = d[0]
		for i in 1 ..< n {
			result.min_diagonal = min(result.min_diagonal, d[i])
			result.max_diagonal = max(result.max_diagonal, d[i])
		}
		if result.min_diagonal > 0 {
			result.condition_estimate = result.max_diagonal / result.min_diagonal
		} else {
			result.condition_estimate = math.INF_F64
		}
	}

	return
}

// Proc group for tridiagonal factorization
pttrf :: proc {
	cpttrf,
	dpttrf,
	spttrf,
	zpttrf,
}

// ============================================================================
// TRIDIAGONAL SOLVE USING FACTORIZATION
// ============================================================================
// Solves A*X = B using the factorization from pttrf

// Complex single precision solve with factorization
cpttrs :: proc(
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	d: []f32, // Factored diagonal from cpttrf
	e: []complex64, // Factored off-diagonal from cpttrf
	b: Matrix(complex64), // Right-hand side (modified to solution)
) -> (
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of right-hand sides must be non-negative")
	assert(len(d) >= n, "Diagonal array too small")
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")
	assert(b.rows >= n && b.cols >= nrhs, "RHS matrix too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	ldb := Blas_Int(b.stride)
	info_int: Info

	lapack.cpttrs_(
		uplo_cstring,
		&n_int,
		&nrhs_int,
		raw_data(d),
		cast(^lapack.complex)raw_data(e),
		cast(^lapack.complex)b.data,
		&ldb,
		&info_int,
		1,
	)

	return Info(info_int)
}

// Double precision real solve with factorization
dpttrs :: proc(
	n: int,
	nrhs: int,
	d: []f64, // Factored diagonal from dpttrf
	e: []f64, // Factored off-diagonal from dpttrf
	b: Matrix(f64), // Right-hand side (modified to solution)
) -> (
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of right-hand sides must be non-negative")
	assert(len(d) >= n, "Diagonal array too small")
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")
	assert(b.rows >= n && b.cols >= nrhs, "RHS matrix too small")

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	ldb := Blas_Int(b.stride)
	info_int: Info

	lapack.dpttrs_(&n_int, &nrhs_int, raw_data(d), raw_data(e), b.data, &ldb, &info_int)

	return Info(info_int)
}

// Single precision real solve with factorization
spttrs :: proc(
	n: int,
	nrhs: int,
	d: []f32, // Factored diagonal from spttrf
	e: []f32, // Factored off-diagonal from spttrf
	b: Matrix(f32), // Right-hand side (modified to solution)
) -> (
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of right-hand sides must be non-negative")
	assert(len(d) >= n, "Diagonal array too small")
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")
	assert(b.rows >= n && b.cols >= nrhs, "RHS matrix too small")

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	ldb := Blas_Int(b.stride)
	info_int: Info

	lapack.spttrs_(&n_int, &nrhs_int, raw_data(d), raw_data(e), b.data, &ldb, &info_int)

	return Info(info_int)
}

// Complex double precision solve with factorization
zpttrs :: proc(
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	d: []f64, // Factored diagonal from zpttrf
	e: []complex128, // Factored off-diagonal from zpttrf
	b: Matrix(complex128), // Right-hand side (modified to solution)
) -> (
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of right-hand sides must be non-negative")
	assert(len(d) >= n, "Diagonal array too small")
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")
	assert(b.rows >= n && b.cols >= nrhs, "RHS matrix too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	ldb := Blas_Int(b.stride)
	info_int: Info

	lapack.zpttrs_(
		uplo_cstring,
		&n_int,
		&nrhs_int,
		raw_data(d),
		cast(^lapack.doublecomplex)raw_data(e),
		cast(^lapack.doublecomplex)b.data,
		&ldb,
		&info_int,
		1,
	)

	return Info(info_int)
}

// Proc group for tridiagonal solve with factorization
pttrs :: proc {
	cpttrs,
	dpttrs,
	spttrs,
	zpttrs,
}

// ============================================================================
// CONVENIENCE FUNCTIONS
// ============================================================================

// Factor and solve tridiagonal system in one call
factor_and_solve_tridiagonal :: proc(
	d: []$T, // Diagonal elements (will be modified)
	e: []$E, // Off-diagonal elements (will be modified)
	b: Matrix($RHS), // Right-hand side (modified to solution)
	allocator := context.allocator,
) -> (
	fact_result: TridiagonalFactorizationResult,
	solve_info: Info,
) where T == scalar_type(E) {
	n := len(d)

	// First factor the matrix
	when T == f32 && E == complex64 && RHS == complex64 {
		fact_result, fact_info := cpttrf(n, d, e)
		if fact_info != .OK {
			return fact_result, fact_info
		}
		solve_info = cpttrs(.Lower, n, b.cols, d, e, b)
	} else when T == f64 && E == f64 && RHS == f64 {
		fact_result, fact_info := dpttrf(n, d, e)
		if fact_info != .OK {
			return fact_result, fact_info
		}
		solve_info = dpttrs(n, b.cols, d, e, b)
	} else when T == f32 && E == f32 && RHS == f32 {
		fact_result, fact_info := spttrf(n, d, e)
		if fact_info != .OK {
			return fact_result, fact_info
		}
		solve_info = spttrs(n, b.cols, d, e, b)
	} else when T == f64 && E == complex128 && RHS == complex128 {
		fact_result, fact_info := zpttrf(n, d, e)
		if fact_info != .OK {
			return fact_result, fact_info
		}
		solve_info = zpttrs(.Lower, n, b.cols, d, e, b)
	} else {
		#panic("Unsupported type combination for tridiagonal factor and solve")
	}

	return
}

// Check if tridiagonal matrix is positive definite by attempting factorization
is_tridiagonal_positive_definite :: proc(
	d: []$T, // Diagonal elements
	e: []$E, // Off-diagonal elements
	allocator := context.allocator,
) -> (
	is_pd: bool,
	min_diag: f64,
	max_diag: f64,
	condition: f64,
) where T == scalar_type(E) {
	n := len(d)

	// Make copies to avoid modifying input
	d_copy := make([]T, n, allocator)
	copy(d_copy, d)
	defer delete(d_copy)

	e_copy := make([]E, max(n - 1, 0), allocator)
	if n > 1 {
		copy(e_copy, e[:n - 1])
	}
	defer delete(e_copy)

	// Attempt factorization
	when T == f32 && E == complex64 {
		result, info := cpttrf(n, d_copy, e_copy)
		is_pd = result.is_positive_definite
		min_diag = result.min_diagonal
		max_diag = result.max_diagonal
		condition = result.condition_estimate
	} else when T == f64 && E == f64 {
		result, info := dpttrf(n, d_copy, e_copy)
		is_pd = result.is_positive_definite
		min_diag = result.min_diagonal
		max_diag = result.max_diagonal
		condition = result.condition_estimate
	} else when T == f32 && E == f32 {
		result, info := spttrf(n, d_copy, e_copy)
		is_pd = result.is_positive_definite
		min_diag = result.min_diagonal
		max_diag = result.max_diagonal
		condition = result.condition_estimate
	} else when T == f64 && E == complex128 {
		result, info := zpttrf(n, d_copy, e_copy)
		is_pd = result.is_positive_definite
		min_diag = result.min_diagonal
		max_diag = result.max_diagonal
		condition = result.condition_estimate
	}

	return
}

// Solve multiple systems with same tridiagonal matrix
solve_tridiagonal_multiple_rhs :: proc(
	d: []$T, // Diagonal elements
	e: []$E, // Off-diagonal elements
	rhs_list: []Matrix($RHS), // List of right-hand sides
	allocator := context.allocator,
) -> (
	solutions: []Matrix(RHS),
	all_success: bool,
) where T == scalar_type(E) {
	n := len(d)
	num_systems := len(rhs_list)

	if num_systems == 0 {
		return nil, false
	}

	solutions = make([]Matrix(RHS), num_systems, allocator)
	all_success = true

	// Make copies for factorization
	d_fact := make([]T, n, allocator)
	copy(d_fact, d)
	defer delete(d_fact)

	e_fact := make([]E, max(n - 1, 0), allocator)
	if n > 1 {
		copy(e_fact, e[:n - 1])
	}
	defer delete(e_fact)

	// Factor once
	when T == f32 && E == complex64 && RHS == complex64 {
		fact_result, fact_info := cpttrf(n, d_fact, e_fact)
		if fact_info != .OK {
			return solutions, false
		}

		// Solve for each RHS
		for i, rhs in rhs_list {
			solutions[i] = matrix_clone(&rhs, allocator)
			solve_info := cpttrs(.Lower, n, rhs.cols, d_fact, e_fact, solutions[i])
			if solve_info != .OK {
				all_success = false
			}
		}
	} else when T == f64 && E == f64 && RHS == f64 {
		fact_result, fact_info := dpttrf(n, d_fact, e_fact)
		if fact_info != .OK {
			return solutions, false
		}

		// Solve for each RHS
		for i, rhs in rhs_list {
			solutions[i] = matrix_clone(&rhs, allocator)
			solve_info := dpttrs(n, rhs.cols, d_fact, e_fact, solutions[i])
			if solve_info != .OK {
				all_success = false
			}
		}
	} else when T == f32 && E == f32 && RHS == f32 {
		fact_result, fact_info := spttrf(n, d_fact, e_fact)
		if fact_info != .OK {
			return solutions, false
		}

		// Solve for each RHS
		for i, rhs in rhs_list {
			solutions[i] = matrix_clone(&rhs, allocator)
			solve_info := spttrs(n, rhs.cols, d_fact, e_fact, solutions[i])
			if solve_info != .OK {
				all_success = false
			}
		}
	} else when T == f64 && E == complex128 && RHS == complex128 {
		fact_result, fact_info := zpttrf(n, d_fact, e_fact)
		if fact_info != .OK {
			return solutions, false
		}

		// Solve for each RHS
		for i, rhs in rhs_list {
			solutions[i] = matrix_clone(&rhs, allocator)
			solve_info := zpttrs(.Lower, n, rhs.cols, d_fact, e_fact, solutions[i])
			if solve_info != .OK {
				all_success = false
			}
		}
	}

	return
}
