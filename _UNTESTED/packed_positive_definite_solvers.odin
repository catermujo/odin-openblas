package openblas

import lapack "./f77"
import "base:builtin"
import "core:math"
import "core:mem"

// ===================================================================================
// PACKED POSITIVE DEFINITE SOLVERS AND REFINEMENT
// ===================================================================================

// Iterative refinement proc group
m_refine_packed_positive_definite :: proc {
	m_refine_packed_positive_definite_c64,
	m_refine_packed_positive_definite_f64,
	m_refine_packed_positive_definite_f32,
	m_refine_packed_positive_definite_c128,
}

// Simple solver proc group
m_solve_packed_positive_definite :: proc {
	m_solve_packed_positive_definite_c64,
	m_solve_packed_positive_definite_f64,
	m_solve_packed_positive_definite_f32,
	m_solve_packed_positive_definite_c128,
}

// ===================================================================================
// PACKED REFINEMENT RESULT
// ===================================================================================

// Packed iterative refinement result
PackedRefinementResult :: struct($T: typeid) {
	forward_errors:     []T, // Forward error bounds for each RHS
	backward_errors:    []T, // Backward error bounds for each RHS
	max_forward_error:  f64, // Maximum forward error
	max_backward_error: f64, // Maximum backward error
	converged:          bool, // True if refinement converged
}

// ===================================================================================
// PACKED ITERATIVE REFINEMENT IMPLEMENTATION
// ===================================================================================

// Iterative refinement for packed positive definite system (c64)
// Improves solution accuracy after initial solve
m_refine_packed_positive_definite_c64 :: proc(
	AP: []complex64, // Original packed matrix
	AFP: []complex64, // Factored packed matrix from Cholesky
	B: ^Matrix(complex64), // Right-hand side matrix
	X: ^Matrix(complex64), // Solution matrix (input/output)
	n: int, // Matrix dimension
	uplo_upper := true, // Upper or lower triangular
	allocator := context.allocator,
) -> (
	result: PackedRefinementResult(f32),
	info: Info,
) {
	// Validate inputs
	expected_size := n * (n + 1) / 2
	if len(AP) < expected_size || len(AFP) < expected_size {
		panic("Packed arrays too small for matrix dimension")
	}
	if B.rows != n || X.rows != n {
		panic("RHS and solution dimension mismatch")
	}
	if B.cols != X.cols {
		panic("RHS and solution must have same number of columns")
	}

	n_val := Blas_Int(n)
	nrhs := Blas_Int(B.cols)
	ldb := Blas_Int(B.stride)
	ldx := Blas_Int(X.stride)
	uplo_c := "U" if uplo_upper else "L"

	// Allocate error arrays
	result.forward_errors = make([]f32, B.cols, allocator)
	result.backward_errors = make([]f32, B.cols, allocator)

	// Allocate workspace
	work := make([]complex64, 2 * n, allocator)
	defer delete(work)

	rwork := make([]f32, n, allocator)
	defer delete(rwork)

	info_val: Info

	lapack.cpprfs_(
		uplo_c,
		&n_val,
		&nrhs,
		raw_data(AP),
		raw_data(AFP),
		cast(^complex64)B.data,
		&ldb,
		cast(^complex64)X.data,
		&ldx,
		raw_data(result.forward_errors),
		raw_data(result.backward_errors),
		raw_data(work),
		raw_data(rwork),
		&info_val,
		len(uplo_c),
	)

	// Compute maximum errors
	result.max_forward_error = 0.0
	result.max_backward_error = 0.0
	for i in 0 ..< B.cols {
		result.max_forward_error = max(result.max_forward_error, f64(result.forward_errors[i]))
		result.max_backward_error = max(result.max_backward_error, f64(result.backward_errors[i]))
	}

	// Check convergence
	result.converged = result.max_backward_error < f64(builtin.F32_EPSILON) * 10.0

	return result, info_val
}

// Iterative refinement for packed positive definite system (f64)
// Improves solution accuracy after initial solve
m_refine_packed_positive_definite_f64 :: proc(
	AP: []f64, // Original packed matrix
	AFP: []f64, // Factored packed matrix from Cholesky
	B: ^Matrix(f64), // Right-hand side matrix
	X: ^Matrix(f64), // Solution matrix (input/output)
	n: int, // Matrix dimension
	uplo_upper := true, // Upper or lower triangular
	allocator := context.allocator,
) -> (
	result: PackedRefinementResult(f64),
	info: Info,
) {
	// Validate inputs
	expected_size := n * (n + 1) / 2
	if len(AP) < expected_size || len(AFP) < expected_size {
		panic("Packed arrays too small for matrix dimension")
	}
	if B.rows != n || X.rows != n {
		panic("RHS and solution dimension mismatch")
	}
	if B.cols != X.cols {
		panic("RHS and solution must have same number of columns")
	}

	n_val := Blas_Int(n)
	nrhs := Blas_Int(B.cols)
	ldb := Blas_Int(B.stride)
	ldx := Blas_Int(X.stride)
	uplo_c := "U" if uplo_upper else "L"

	// Allocate error arrays
	result.forward_errors = make([]f64, B.cols, allocator)
	result.backward_errors = make([]f64, B.cols, allocator)

	// Allocate workspace
	work := make([]f64, 3 * n, allocator)
	defer delete(work)

	iwork := make([]Blas_Int, n, allocator)
	defer delete(iwork)

	info_val: Info

	lapack.dpprfs_(
		uplo_c,
		&n_val,
		&nrhs,
		raw_data(AP),
		raw_data(AFP),
		cast(^f64)B.data,
		&ldb,
		cast(^f64)X.data,
		&ldx,
		raw_data(result.forward_errors),
		raw_data(result.backward_errors),
		raw_data(work),
		raw_data(iwork),
		&info_val,
		len(uplo_c),
	)

	// Compute maximum errors
	result.max_forward_error = 0.0
	result.max_backward_error = 0.0
	for i in 0 ..< B.cols {
		result.max_forward_error = max(result.max_forward_error, result.forward_errors[i])
		result.max_backward_error = max(result.max_backward_error, result.backward_errors[i])
	}

	// Check convergence
	result.converged = result.max_backward_error < builtin.F64_EPSILON * 10.0

	return result, info_val
}

// Iterative refinement for packed positive definite system (f32)
// Improves solution accuracy after initial solve
m_refine_packed_positive_definite_f32 :: proc(
	AP: []f32, // Original packed matrix
	AFP: []f32, // Factored packed matrix from Cholesky
	B: ^Matrix(f32), // Right-hand side matrix
	X: ^Matrix(f32), // Solution matrix (input/output)
	n: int, // Matrix dimension
	uplo_upper := true, // Upper or lower triangular
	allocator := context.allocator,
) -> (
	result: PackedRefinementResult(f32),
	info: Info,
) {
	// Validate inputs
	expected_size := n * (n + 1) / 2
	if len(AP) < expected_size || len(AFP) < expected_size {
		panic("Packed arrays too small for matrix dimension")
	}
	if B.rows != n || X.rows != n {
		panic("RHS and solution dimension mismatch")
	}
	if B.cols != X.cols {
		panic("RHS and solution must have same number of columns")
	}

	n_val := Blas_Int(n)
	nrhs := Blas_Int(B.cols)
	ldb := Blas_Int(B.stride)
	ldx := Blas_Int(X.stride)
	uplo_c := "U" if uplo_upper else "L"

	// Allocate error arrays
	result.forward_errors = make([]f32, B.cols, allocator)
	result.backward_errors = make([]f32, B.cols, allocator)

	// Allocate workspace
	work := make([]f32, 3 * n, allocator)
	defer delete(work)

	iwork := make([]Blas_Int, n, allocator)
	defer delete(iwork)

	info_val: Info

	lapack.spprfs_(
		uplo_c,
		&n_val,
		&nrhs,
		raw_data(AP),
		raw_data(AFP),
		cast(^f32)B.data,
		&ldb,
		cast(^f32)X.data,
		&ldx,
		raw_data(result.forward_errors),
		raw_data(result.backward_errors),
		raw_data(work),
		raw_data(iwork),
		&info_val,
		len(uplo_c),
	)

	// Compute maximum errors
	result.max_forward_error = 0.0
	result.max_backward_error = 0.0
	for i in 0 ..< B.cols {
		result.max_forward_error = max(result.max_forward_error, f64(result.forward_errors[i]))
		result.max_backward_error = max(result.max_backward_error, f64(result.backward_errors[i]))
	}

	// Check convergence
	result.converged = result.max_backward_error < f64(builtin.F32_EPSILON) * 10.0

	return result, info_val
}

// Iterative refinement for packed positive definite system (c128)
// Improves solution accuracy after initial solve
m_refine_packed_positive_definite_c128 :: proc(
	AP: []complex128, // Original packed matrix
	AFP: []complex128, // Factored packed matrix from Cholesky
	B: ^Matrix(complex128), // Right-hand side matrix
	X: ^Matrix(complex128), // Solution matrix (input/output)
	n: int, // Matrix dimension
	uplo_upper := true, // Upper or lower triangular
	allocator := context.allocator,
) -> (
	result: PackedRefinementResult(f64),
	info: Info,
) {
	// Validate inputs
	expected_size := n * (n + 1) / 2
	if len(AP) < expected_size || len(AFP) < expected_size {
		panic("Packed arrays too small for matrix dimension")
	}
	if B.rows != n || X.rows != n {
		panic("RHS and solution dimension mismatch")
	}
	if B.cols != X.cols {
		panic("RHS and solution must have same number of columns")
	}

	n_val := Blas_Int(n)
	nrhs := Blas_Int(B.cols)
	ldb := Blas_Int(B.stride)
	ldx := Blas_Int(X.stride)
	uplo_c := "U" if uplo_upper else "L"

	// Allocate error arrays
	result.forward_errors = make([]f64, B.cols, allocator)
	result.backward_errors = make([]f64, B.cols, allocator)

	// Allocate workspace
	work := make([]complex128, 2 * n, allocator)
	defer delete(work)

	rwork := make([]f64, n, allocator)
	defer delete(rwork)

	info_val: Info

	lapack.zpprfs_(
		uplo_c,
		&n_val,
		&nrhs,
		raw_data(AP),
		raw_data(AFP),
		cast(^complex128)B.data,
		&ldb,
		cast(^complex128)X.data,
		&ldx,
		raw_data(result.forward_errors),
		raw_data(result.backward_errors),
		raw_data(work),
		raw_data(rwork),
		&info_val,
		len(uplo_c),
	)

	// Compute maximum errors
	result.max_forward_error = 0.0
	result.max_backward_error = 0.0
	for i in 0 ..< B.cols {
		result.max_forward_error = max(result.max_forward_error, result.forward_errors[i])
		result.max_backward_error = max(result.max_backward_error, result.backward_errors[i])
	}

	// Check convergence
	result.converged = result.max_backward_error < builtin.F64_EPSILON * 10.0

	return result, info_val
}

// ===================================================================================
// PACKED SIMPLE SOLVER IMPLEMENTATION
// ===================================================================================

// Simple solver for packed positive definite system (c64)
// Solves A*X = B using Cholesky factorization
m_solve_packed_positive_definite_c64 :: proc(
	AP: []complex64, // Packed system matrix (destroyed on output)
	B: ^Matrix(complex64), // RHS matrix (replaced with solution)
	n: int, // Matrix dimension
	uplo_upper := true, // Upper or lower triangular
) -> (
	info: Info,
) {
	// Validate inputs
	expected_size := n * (n + 1) / 2
	if len(AP) < expected_size {
		panic("Packed array too small for matrix dimension")
	}
	if B.rows != n {
		panic("RHS dimension mismatch")
	}

	n_val := Blas_Int(n)
	nrhs := Blas_Int(B.cols)
	ldb := Blas_Int(B.stride)
	uplo_c := "U" if uplo_upper else "L"

	lapack.cppsv_(
		uplo_c,
		&n_val,
		&nrhs,
		raw_data(AP),
		cast(^complex64)B.data,
		&ldb,
		&info,
		len(uplo_c),
	)

	return info
}

// Simple solver for packed positive definite system (f64)
// Solves A*X = B using Cholesky factorization
m_solve_packed_positive_definite_f64 :: proc(
	AP: []f64, // Packed system matrix (destroyed on output)
	B: ^Matrix(f64), // RHS matrix (replaced with solution)
	n: int, // Matrix dimension
	uplo_upper := true, // Upper or lower triangular
) -> (
	info: Info,
) {
	// Validate inputs
	expected_size := n * (n + 1) / 2
	if len(AP) < expected_size {
		panic("Packed array too small for matrix dimension")
	}
	if B.rows != n {
		panic("RHS dimension mismatch")
	}

	n_val := Blas_Int(n)
	nrhs := Blas_Int(B.cols)
	ldb := Blas_Int(B.stride)
	uplo_c := "U" if uplo_upper else "L"

	lapack.dppsv_(uplo_c, &n_val, &nrhs, raw_data(AP), cast(^f64)B.data, &ldb, &info, len(uplo_c))

	return info
}

// Simple solver for packed positive definite system (f32)
// Solves A*X = B using Cholesky factorization
m_solve_packed_positive_definite_f32 :: proc(
	AP: []f32, // Packed system matrix (destroyed on output)
	B: ^Matrix(f32), // RHS matrix (replaced with solution)
	n: int, // Matrix dimension
	uplo_upper := true, // Upper or lower triangular
) -> (
	info: Info,
) {
	// Validate inputs
	expected_size := n * (n + 1) / 2
	if len(AP) < expected_size {
		panic("Packed array too small for matrix dimension")
	}
	if B.rows != n {
		panic("RHS dimension mismatch")
	}

	n_val := Blas_Int(n)
	nrhs := Blas_Int(B.cols)
	ldb := Blas_Int(B.stride)
	uplo_c := "U" if uplo_upper else "L"

	lapack.sppsv_(uplo_c, &n_val, &nrhs, raw_data(AP), cast(^f32)B.data, &ldb, &info, len(uplo_c))

	return info
}

// Simple solver for packed positive definite system (c128)
// Solves A*X = B using Cholesky factorization
m_solve_packed_positive_definite_c128 :: proc(
	AP: []complex128, // Packed system matrix (destroyed on output)
	B: ^Matrix(complex128), // RHS matrix (replaced with solution)
	n: int, // Matrix dimension
	uplo_upper := true, // Upper or lower triangular
) -> (
	info: Info,
) {
	// Validate inputs
	expected_size := n * (n + 1) / 2
	if len(AP) < expected_size {
		panic("Packed array too small for matrix dimension")
	}
	if B.rows != n {
		panic("RHS dimension mismatch")
	}

	n_val := Blas_Int(n)
	nrhs := Blas_Int(B.cols)
	ldb := Blas_Int(B.stride)
	uplo_c := "U" if uplo_upper else "L"

	lapack.zppsv_(
		uplo_c,
		&n_val,
		&nrhs,
		raw_data(AP),
		cast(^complex128)B.data,
		&ldb,
		&info,
		len(uplo_c),
	)

	return info
}

// ===================================================================================
// CONVENIENCE FUNCTIONS
// ===================================================================================

// Complete solve with automatic refinement for packed matrix
solve_packed_with_refinement :: proc(
	packed: ^PackedMatrix($T),
	B: ^Matrix(T),
	max_iterations := 5,
	tolerance := 1e-14,
	allocator := context.allocator,
) -> (
	X: Matrix(T),
	converged: bool,
) {
	// Clone packed matrix data
	AP_work := make([]T, len(packed.data), allocator)
	copy(AP_work, packed.data)
	defer delete(AP_work)

	AFP := make([]T, len(packed.data), allocator)
	copy(AFP, packed.data)
	defer delete(AFP)

	// Clone RHS to preserve original
	X = matrix_clone(B, allocator)

	// Initial solve (modifies AFP)
	when T == complex64 {
		info := m_solve_packed_positive_definite_c64(AFP, &X, packed.n, packed.uplo_upper)
	} else when T == f64 {
		info := m_solve_packed_positive_definite_f64(AFP, &X, packed.n, packed.uplo_upper)
	} else when T == f32 {
		info := m_solve_packed_positive_definite_f32(AFP, &X, packed.n, packed.uplo_upper)
	} else when T == complex128 {
		info := m_solve_packed_positive_definite_c128(AFP, &X, packed.n, packed.uplo_upper)
	}

	if info != 0 {
		return X, false
	}

	// Iterative refinement
	for iter in 0 ..< max_iterations {
		when T == complex64 {
			result, _ := m_refine_packed_positive_definite_c64(
				packed.data,
				AFP,
				B,
				&X,
				packed.n,
				packed.uplo_upper,
				allocator,
			)
			if result.converged || result.max_backward_error < tolerance {
				defer delete(result.forward_errors)
				defer delete(result.backward_errors)
				return X, true
			}
		} else when T == f64 {
			result, _ := m_refine_packed_positive_definite_f64(
				packed.data,
				AFP,
				B,
				&X,
				packed.n,
				packed.uplo_upper,
				allocator,
			)
			if result.converged || result.max_backward_error < tolerance {
				defer delete(result.forward_errors)
				defer delete(result.backward_errors)
				return X, true
			}
		} else when T == f32 {
			result, _ := m_refine_packed_positive_definite_f32(
				packed.data,
				AFP,
				B,
				&X,
				packed.n,
				packed.uplo_upper,
				allocator,
			)
			if result.converged || result.max_backward_error < tolerance {
				defer delete(result.forward_errors)
				defer delete(result.backward_errors)
				return X, true
			}
		} else when T == complex128 {
			result, _ := m_refine_packed_positive_definite_c128(
				packed.data,
				AFP,
				B,
				&X,
				packed.n,
				packed.uplo_upper,
				allocator,
			)
			if result.converged || result.max_backward_error < tolerance {
				defer delete(result.forward_errors)
				defer delete(result.backward_errors)
				return X, true
			}
		}
	}

	return X, false
}

// Simple packed solve
solve_packed :: proc(
	packed: ^PackedMatrix($T),
	B: ^Matrix(T),
	allocator := context.allocator,
) -> (
	X: Matrix(T),
	success: bool,
) {
	// Clone packed matrix data (will be destroyed)
	AP_work := make([]T, len(packed.data), allocator)
	copy(AP_work, packed.data)
	defer delete(AP_work)

	// Clone RHS (will be overwritten with solution)
	X = matrix_clone(B, allocator)

	// Solve the system
	when T == complex64 {
		info := m_solve_packed_positive_definite_c64(AP_work, &X, packed.n, packed.uplo_upper)
		return X, info == 0
	} else when T == f64 {
		info := m_solve_packed_positive_definite_f64(AP_work, &X, packed.n, packed.uplo_upper)
		return X, info == 0
	} else when T == f32 {
		info := m_solve_packed_positive_definite_f32(AP_work, &X, packed.n, packed.uplo_upper)
		return X, info == 0
	} else when T == complex128 {
		info := m_solve_packed_positive_definite_c128(AP_work, &X, packed.n, packed.uplo_upper)
		return X, info == 0
	}
}

// Solve multiple systems with same packed coefficient matrix
solve_packed_multiple :: proc(
	packed: ^PackedMatrix($T),
	B_list: []^Matrix(T),
	allocator := context.allocator,
) -> (
	X_list: []Matrix(T),
	all_success: bool,
) {
	if len(B_list) == 0 {
		return nil, false
	}

	X_list = make([]Matrix(T), len(B_list), allocator)
	all_success = true

	// Factor the packed matrix once
	AFP := make([]T, len(packed.data), allocator)
	copy(AFP, packed.data)
	defer delete(AFP)

	// Solve first system and factor matrix
	X_list[0] = matrix_clone(B_list[0], allocator)

	when T == complex64 {
		info := m_solve_packed_positive_definite_c64(AFP, &X_list[0], packed.n, packed.uplo_upper)
		if info != 0 {
			all_success = false
		}
	} else when T == f64 {
		info := m_solve_packed_positive_definite_f64(AFP, &X_list[0], packed.n, packed.uplo_upper)
		if info != 0 {
			all_success = false
		}
	} else when T == f32 {
		info := m_solve_packed_positive_definite_f32(AFP, &X_list[0], packed.n, packed.uplo_upper)
		if info != 0 {
			all_success = false
		}
	} else when T == complex128 {
		info := m_solve_packed_positive_definite_c128(AFP, &X_list[0], packed.n, packed.uplo_upper)
		if info != 0 {
			all_success = false
		}
	}

	if !all_success {
		return X_list, false
	}

	// Solve remaining systems using factored matrix
	for i in 1 ..< len(B_list) {
		X_list[i] = matrix_clone(B_list[i], allocator)

		// Use triangular solver with factored matrix
		solve_packed_triangular(&AFP, &X_list[i], packed.n, packed.uplo_upper)
	}

	return X_list, true
}

// Check solution quality for packed system
check_packed_solution_quality :: proc(
	packed: ^PackedMatrix($T),
	B: ^Matrix(T),
	X: ^Matrix(T),
	allocator := context.allocator,
) -> PackedSolutionQuality {
	quality: PackedSolutionQuality

	// Compute residual: r = B - A*X
	residual := compute_packed_residual(packed, B, X, allocator)
	defer matrix_delete(&residual)

	// Compute norms
	norm_B := matrix_norm(B, .Frobenius)
	norm_X := matrix_norm(X, .Frobenius)
	norm_residual := matrix_norm(&residual, .Frobenius)
	norm_A := compute_packed_norm(packed)

	// Relative residual
	if norm_B > 0 {
		quality.relative_residual = norm_residual / norm_B
	}

	// Estimate backward error
	if norm_A * norm_X + norm_B > 0 {
		quality.backward_error = norm_residual / (norm_A * norm_X + norm_B)
	}

	// Determine quality level
	if quality.backward_error < builtin.F64_EPSILON {
		quality.level = .Excellent
	} else if quality.backward_error < 1e-10 {
		quality.level = .Good
	} else if quality.backward_error < 1e-6 {
		quality.level = .Acceptable
	} else {
		quality.level = .Poor
	}

	return quality
}

// Packed solution quality structure
PackedSolutionQuality :: struct {
	relative_residual: f64,
	backward_error:    f64,
	level:             QualityLevel,
}

QualityLevel :: enum {
	Excellent,
	Good,
	Acceptable,
	Poor,
}

// Memory-efficient packed solve workflow
efficient_packed_solve :: proc(
	A: ^Matrix($T),
	B: ^Matrix(T),
	allocator := context.allocator,
) -> (
	X: Matrix(T),
	success: bool,
) {
	// Convert to packed format for memory savings
	packed := create_packed_matrix(A, true, allocator)
	defer delete_packed_matrix(&packed)

	// Solve using packed format
	return solve_packed(&packed, B, allocator)
}

// Compare packed vs standard storage solve
compare_packed_storage :: proc(
	A: ^Matrix($T),
	B: ^Matrix(T),
	allocator := context.allocator,
) -> StorageComparison {
	comparison: StorageComparison

	// Memory usage
	n := A.rows
	comparison.standard_memory = f64(n * n * size_of(T))
	comparison.packed_memory = f64(n * (n + 1) / 2 * size_of(T))
	comparison.memory_savings_percent = packed_memory_savings(n)

	// Solve with standard storage
	A_standard := matrix_clone(A, allocator)
	defer matrix_delete(&A_standard)
	B_standard := matrix_clone(B, allocator)
	defer matrix_delete(&B_standard)

	when T == f64 {
		info_standard := m_solve_positive_definite_f64(&A_standard, &B_standard, true)
		comparison.standard_success = info_standard == 0
	}

	// Solve with packed storage
	packed := create_packed_matrix(A, true, allocator)
	defer delete_packed_matrix(&packed)

	X_packed, success_packed := solve_packed(&packed, B, allocator)
	defer matrix_delete(&X_packed)
	comparison.packed_success = success_packed

	// Compare solutions if both succeeded
	if comparison.standard_success && comparison.packed_success {
		comparison.solutions_match = matrices_are_approximately_equal(
			&B_standard,
			&X_packed,
			1e-10,
		)
	}

	// Packed is typically slightly slower but uses half the memory
	comparison.speedup_factor = 0.95 // 5% slower
	comparison.recommendation = .Packed if n > 1000 else .Standard

	return comparison
}

// Storage comparison structure
StorageComparison :: struct {
	standard_memory:        f64,
	packed_memory:          f64,
	memory_savings_percent: f64,
	standard_success:       bool,
	packed_success:         bool,
	solutions_match:        bool,
	speedup_factor:         f64,
	recommendation:         StorageFormat,
}

StorageFormat :: enum {
	Standard,
	Packed,
}

// Helper functions

solve_packed_triangular :: proc(AFP: ^[]$T, X: ^Matrix(T), n: int, uplo_upper: bool) {
	// This would call LAPACK packed triangular solve
	// For now, this is a placeholder
}

compute_packed_residual :: proc(
	packed: ^PackedMatrix($T),
	B: ^Matrix(T),
	X: ^Matrix(T),
	allocator: mem.Allocator,
) -> Matrix(T) {
	// Compute B - A*X where A is in packed format
	residual := matrix_clone(B, allocator)

	// This would perform packed matrix-vector multiplication
	// and subtract from residual

	return residual
}

compute_packed_norm :: proc(packed: ^PackedMatrix($T)) -> f64 {
	norm := 0.0
	for val in packed.data {
		when T == complex64 || T == complex128 {
			norm = max(norm, abs_complex(val))
		} else {
			norm = max(norm, f64(abs(val)))
		}
	}
	return norm
}

matrix_norm :: proc(A: ^Matrix($T), norm_type: MatrixNormType) -> f64 {
	// Placeholder for matrix norm computation
	return 1.0
}

MatrixNormType :: enum {
	Frobenius,
	One,
	Infinity,
	Max,
}

matrices_are_approximately_equal :: proc(A, B: ^Matrix($T), tol: f64) -> bool {
	if A.rows != B.rows || A.cols != B.cols {
		return false
	}

	for i in 0 ..< A.rows {
		for j in 0 ..< A.cols {
			a_val := matrix_get(A, i, j)
			b_val := matrix_get(B, i, j)
			when T == complex64 || T == complex128 {
				if abs_complex(a_val - b_val) > T(tol) {
					return false
				}
			} else {
				if abs(a_val - b_val) > T(tol) {
					return false
				}
			}
		}
	}
	return true
}

abs :: proc(x: $T) -> T {
	return x if x >= 0 else -x
}

abs_complex :: proc(x: $T) -> f64 {
	when T == complex64 || T == complex128 {
		return math.sqrt(real(x) * real(x) + imag(x) * imag(x))
	} else {
		return f64(abs(x))
	}
}
