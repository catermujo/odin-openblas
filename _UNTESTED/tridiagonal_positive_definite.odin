package openblas

import lapack "./f77"
import "base:builtin"
import "core:math"
import "core:mem"
import "core:slice"

// ===================================================================================
// TRIDIAGONAL POSITIVE DEFINITE EIGENVALUES, REFINEMENT, AND SOLVERS
// ===================================================================================

// Eigenvalue computation proc group
m_eigenvalues_tridiagonal :: proc {
	m_eigenvalues_tridiagonal_c64,
	m_eigenvalues_tridiagonal_f64,
	m_eigenvalues_tridiagonal_f32,
	m_eigenvalues_tridiagonal_c128,
}

// Iterative refinement proc group
m_refine_tridiagonal_positive_definite :: proc {
	m_refine_tridiagonal_positive_definite_c64,
	m_refine_tridiagonal_positive_definite_f64,
	m_refine_tridiagonal_positive_definite_f32,
	m_refine_tridiagonal_positive_definite_c128,
}

// Simple solver proc group
m_solve_tridiagonal_positive_definite :: proc {
	m_solve_tridiagonal_positive_definite_c64,
	m_solve_tridiagonal_positive_definite_f64,
	m_solve_tridiagonal_positive_definite_f32,
	m_solve_tridiagonal_positive_definite_c128,
}

// ===================================================================================
// EIGENVECTOR COMPUTATION MODE
// ===================================================================================

// Eigenvector computation mode
EigenvectorMode :: enum {
	None, // "N" - Eigenvalues only
	Identity, // "I" - Eigenvectors of tridiagonal, Z initialized to identity
	Vectors, // "V" - Eigenvectors and update Z matrix
}

// Convert eigenvector mode to LAPACK character
_compz_to_char :: proc(mode: EigenvectorMode) -> cstring {
	switch mode {
	case .None:
		return "N"
	case .Identity:
		return "I"
	case .Vectors:
		return "V"
	case:
		return "N"
	}
}

// ===================================================================================
// RESULT STRUCTURES
// ===================================================================================

// Eigenvalue computation result
TridiagonalEigenResult :: struct($T: typeid) {
	eigenvalues:      []T, // Computed eigenvalues (sorted)
	eigenvectors:     Matrix(T), // Eigenvector matrix (if requested)
	all_positive:     bool, // True if all eigenvalues > 0
	min_eigenvalue:   f64, // Smallest eigenvalue
	max_eigenvalue:   f64, // Largest eigenvalue
	condition_number: f64, // max/min eigenvalue ratio
}

// Tridiagonal refinement result
TridiagonalRefinementResult :: struct($T: typeid) {
	forward_errors:     []T, // Forward error bounds
	backward_errors:    []T, // Backward error bounds
	max_forward_error:  f64, // Maximum forward error
	max_backward_error: f64, // Maximum backward error
	converged:          bool, // True if refinement converged
}

// ===================================================================================
// EIGENVALUE COMPUTATION IMPLEMENTATION
// ===================================================================================

// Compute eigenvalues of symmetric tridiagonal matrix (c64)
m_eigenvalues_tridiagonal_c64 :: proc(
	D: []f32, // Diagonal elements (modified on output to eigenvalues)
	E: []f32, // Off-diagonal elements (destroyed)
	Z: ^Matrix(complex64) = nil, // Eigenvector matrix (optional)
	mode := EigenvectorMode.None, // Eigenvector computation mode
	allocator := context.allocator,
) -> (
	info: Info,
) {
	// Validate inputs
	n := len(D)
	if len(E) != n - 1 && n > 1 {
		panic("Off-diagonal must have n-1 elements")
	}

	n_val := Blas_Int(n)
	compz_c := _compz_to_char(mode)

	// Handle eigenvector matrix
	ldz: Blas_Int = 1
	z_ptr: ^complex64 = nil
	if mode != .None && Z != nil {
		if Z.rows != n || Z.cols != n {
			panic("Z matrix must be n×n")
		}
		ldz = Blas_Int(Z.stride)
		z_ptr = cast(^complex64)Z.data
	}

	// Allocate workspace
	work := make([]f32, 4 * n, allocator)
	defer delete(work)

	info_val: Blas_Int

	lapack.cpteqr_(
		compz_c,
		&n_val,
		raw_data(D),
		raw_data(E),
		z_ptr,
		&ldz,
		raw_data(work),
		&info_val,
		len(compz_c),
	)

	return info_val
}

// Compute eigenvalues of symmetric tridiagonal matrix (f64)
m_eigenvalues_tridiagonal_f64 :: proc(
	D: []f64, // Diagonal elements (modified on output to eigenvalues)
	E: []f64, // Off-diagonal elements (destroyed)
	Z: ^Matrix(f64) = nil, // Eigenvector matrix (optional)
	mode := EigenvectorMode.None, // Eigenvector computation mode
	allocator := context.allocator,
) -> (
	info: Info,
) {
	// Validate inputs
	n := len(D)
	if len(E) != n - 1 && n > 1 {
		panic("Off-diagonal must have n-1 elements")
	}

	n_val := Blas_Int(n)
	compz_c := _compz_to_char(mode)

	// Handle eigenvector matrix
	ldz: Blas_Int = 1
	z_ptr: ^f64 = nil
	if mode != .None && Z != nil {
		if Z.rows != n || Z.cols != n {
			panic("Z matrix must be n×n")
		}
		ldz = Blas_Int(Z.stride)
		z_ptr = cast(^f64)Z.data
	}

	// Allocate workspace
	work := make([]f64, 4 * n, allocator)
	defer delete(work)

	info_val: Blas_Int

	lapack.dpteqr_(
		compz_c,
		&n_val,
		raw_data(D),
		raw_data(E),
		z_ptr,
		&ldz,
		raw_data(work),
		&info_val,
		len(compz_c),
	)

	return info_val
}

// Compute eigenvalues of symmetric tridiagonal matrix (f32)
m_eigenvalues_tridiagonal_f32 :: proc(
	D: []f32, // Diagonal elements (modified on output to eigenvalues)
	E: []f32, // Off-diagonal elements (destroyed)
	Z: ^Matrix(f32) = nil, // Eigenvector matrix (optional)
	mode := EigenvectorMode.None, // Eigenvector computation mode
	allocator := context.allocator,
) -> (
	info: Info,
) {
	// Validate inputs
	n := len(D)
	if len(E) != n - 1 && n > 1 {
		panic("Off-diagonal must have n-1 elements")
	}

	n_val := Blas_Int(n)
	compz_c := _compz_to_char(mode)

	// Handle eigenvector matrix
	ldz: Blas_Int = 1
	z_ptr: ^f32 = nil
	if mode != .None && Z != nil {
		if Z.rows != n || Z.cols != n {
			panic("Z matrix must be n×n")
		}
		ldz = Blas_Int(Z.stride)
		z_ptr = cast(^f32)Z.data
	}

	// Allocate workspace
	work := make([]f32, 4 * n, allocator)
	defer delete(work)

	info_val: Blas_Int

	lapack.spteqr_(
		compz_c,
		&n_val,
		raw_data(D),
		raw_data(E),
		z_ptr,
		&ldz,
		raw_data(work),
		&info_val,
		len(compz_c),
	)

	return info_val
}

// Compute eigenvalues of symmetric tridiagonal matrix (c128)
m_eigenvalues_tridiagonal_c128 :: proc(
	D: []f64, // Diagonal elements (modified on output to eigenvalues)
	E: []f64, // Off-diagonal elements (destroyed)
	Z: ^Matrix(complex128) = nil, // Eigenvector matrix (optional)
	mode := EigenvectorMode.None, // Eigenvector computation mode
	allocator := context.allocator,
) -> (
	info: Info,
) {
	// Validate inputs
	n := len(D)
	if len(E) != n - 1 && n > 1 {
		panic("Off-diagonal must have n-1 elements")
	}

	n_val := Blas_Int(n)
	compz_c := _compz_to_char(mode)

	// Handle eigenvector matrix
	ldz: Blas_Int = 1
	z_ptr: ^complex128 = nil
	if mode != .None && Z != nil {
		if Z.rows != n || Z.cols != n {
			panic("Z matrix must be n×n")
		}
		ldz = Blas_Int(Z.stride)
		z_ptr = cast(^complex128)Z.data
	}

	// Allocate workspace
	work := make([]f64, 4 * n, allocator)
	defer delete(work)

	info_val: Blas_Int

	lapack.zpteqr_(
		compz_c,
		&n_val,
		raw_data(D),
		raw_data(E),
		z_ptr,
		&ldz,
		raw_data(work),
		&info_val,
		len(compz_c),
	)

	return info_val
}

// ===================================================================================
// ITERATIVE REFINEMENT IMPLEMENTATION
// ===================================================================================

// Iterative refinement for tridiagonal positive definite system (c64)
m_refine_tridiagonal_positive_definite_c64 :: proc(
	D: []f32, // Original diagonal
	E: []complex64, // Original off-diagonal
	DF: []f32, // Factored diagonal
	EF: []complex64, // Factored off-diagonal
	B: ^Matrix(complex64), // Right-hand side
	X: ^Matrix(complex64), // Solution (input/output)
	uplo_upper := true, // Upper or lower (for complex)
	allocator := context.allocator,
) -> (
	result: TridiagonalRefinementResult(f32),
	info: Info,
) {
	// Validate inputs
	n := len(D)
	if len(E) != n - 1 || len(DF) != n || len(EF) != n - 1 {
		panic("Array dimension mismatch")
	}
	if B.rows != n || X.rows != n {
		panic("RHS/solution dimension mismatch")
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
	work := make([]complex64, n, allocator)
	defer delete(work)
	rwork := make([]f32, n, allocator)
	defer delete(rwork)

	info_val: Blas_Int

	lapack.cptrfs_(
		uplo_c,
		&n_val,
		&nrhs,
		raw_data(D),
		raw_data(E),
		raw_data(DF),
		raw_data(EF),
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
	for i in 0 ..< B.cols {
		result.max_forward_error = max(result.max_forward_error, f64(result.forward_errors[i]))
		result.max_backward_error = max(result.max_backward_error, f64(result.backward_errors[i]))
	}

	result.converged = result.max_backward_error < f64(builtin.F32_EPSILON) * 10.0

	return result, info_val
}

// Iterative refinement for tridiagonal positive definite system (f64)
m_refine_tridiagonal_positive_definite_f64 :: proc(
	D: []f64, // Original diagonal
	E: []f64, // Original off-diagonal
	DF: []f64, // Factored diagonal
	EF: []f64, // Factored off-diagonal
	B: ^Matrix(f64), // Right-hand side
	X: ^Matrix(f64), // Solution (input/output)
	allocator := context.allocator,
) -> (
	result: TridiagonalRefinementResult(f64),
	info: Info,
) {
	// Validate inputs
	n := len(D)
	if len(E) != n - 1 || len(DF) != n || len(EF) != n - 1 {
		panic("Array dimension mismatch")
	}
	if B.rows != n || X.rows != n {
		panic("RHS/solution dimension mismatch")
	}

	n_val := Blas_Int(n)
	nrhs := Blas_Int(B.cols)
	ldb := Blas_Int(B.stride)
	ldx := Blas_Int(X.stride)

	// Allocate error arrays
	result.forward_errors = make([]f64, B.cols, allocator)
	result.backward_errors = make([]f64, B.cols, allocator)

	// Allocate workspace
	work := make([]f64, 2 * n, allocator)
	defer delete(work)

	info_val: Blas_Int

	lapack.dptrfs_(
		&n_val,
		&nrhs,
		raw_data(D),
		raw_data(E),
		raw_data(DF),
		raw_data(EF),
		cast(^f64)B.data,
		&ldb,
		cast(^f64)X.data,
		&ldx,
		raw_data(result.forward_errors),
		raw_data(result.backward_errors),
		raw_data(work),
		&info_val,
	)

	// Compute maximum errors
	for i in 0 ..< B.cols {
		result.max_forward_error = max(result.max_forward_error, result.forward_errors[i])
		result.max_backward_error = max(result.max_backward_error, result.backward_errors[i])
	}

	result.converged = result.max_backward_error < builtin.F64_EPSILON * 10.0

	return result, info_val
}

// Iterative refinement for tridiagonal positive definite system (f32)
m_refine_tridiagonal_positive_definite_f32 :: proc(
	D: []f32, // Original diagonal
	E: []f32, // Original off-diagonal
	DF: []f32, // Factored diagonal
	EF: []f32, // Factored off-diagonal
	B: ^Matrix(f32), // Right-hand side
	X: ^Matrix(f32), // Solution (input/output)
	allocator := context.allocator,
) -> (
	result: TridiagonalRefinementResult(f32),
	info: Info,
) {
	// Validate inputs
	n := len(D)
	if len(E) != n - 1 || len(DF) != n || len(EF) != n - 1 {
		panic("Array dimension mismatch")
	}
	if B.rows != n || X.rows != n {
		panic("RHS/solution dimension mismatch")
	}

	n_val := Blas_Int(n)
	nrhs := Blas_Int(B.cols)
	ldb := Blas_Int(B.stride)
	ldx := Blas_Int(X.stride)

	// Allocate error arrays
	result.forward_errors = make([]f32, B.cols, allocator)
	result.backward_errors = make([]f32, B.cols, allocator)

	// Allocate workspace
	work := make([]f32, 2 * n, allocator)
	defer delete(work)

	info_val: Blas_Int

	lapack.sptrfs_(
		&n_val,
		&nrhs,
		raw_data(D),
		raw_data(E),
		raw_data(DF),
		raw_data(EF),
		cast(^f32)B.data,
		&ldb,
		cast(^f32)X.data,
		&ldx,
		raw_data(result.forward_errors),
		raw_data(result.backward_errors),
		raw_data(work),
		&info_val,
	)

	// Compute maximum errors
	for i in 0 ..< B.cols {
		result.max_forward_error = max(result.max_forward_error, f64(result.forward_errors[i]))
		result.max_backward_error = max(result.max_backward_error, f64(result.backward_errors[i]))
	}

	result.converged = result.max_backward_error < f64(builtin.F32_EPSILON) * 10.0

	return result, info_val
}

// Iterative refinement for tridiagonal positive definite system (c128)
m_refine_tridiagonal_positive_definite_c128 :: proc(
	D: []f64, // Original diagonal
	E: []complex128, // Original off-diagonal
	DF: []f64, // Factored diagonal
	EF: []complex128, // Factored off-diagonal
	B: ^Matrix(complex128), // Right-hand side
	X: ^Matrix(complex128), // Solution (input/output)
	uplo_upper := true, // Upper or lower (for complex)
	allocator := context.allocator,
) -> (
	result: TridiagonalRefinementResult(f64),
	info: Info,
) {
	// Similar implementation to c64
	n := len(D)
	if len(E) != n - 1 || len(DF) != n || len(EF) != n - 1 {
		panic("Array dimension mismatch")
	}
	if B.rows != n || X.rows != n {
		panic("RHS/solution dimension mismatch")
	}

	n_val := Blas_Int(n)
	nrhs := Blas_Int(B.cols)
	ldb := Blas_Int(B.stride)
	ldx := Blas_Int(X.stride)
	uplo_c := "U" if uplo_upper else "L"

	result.forward_errors = make([]f64, B.cols, allocator)
	result.backward_errors = make([]f64, B.cols, allocator)

	work := make([]complex128, n, allocator)
	defer delete(work)
	rwork := make([]f64, n, allocator)
	defer delete(rwork)

	info_val: Blas_Int

	lapack.zptrfs_(
		uplo_c,
		&n_val,
		&nrhs,
		raw_data(D),
		raw_data(E),
		raw_data(DF),
		raw_data(EF),
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

	for i in 0 ..< B.cols {
		result.max_forward_error = max(result.max_forward_error, result.forward_errors[i])
		result.max_backward_error = max(result.max_backward_error, result.backward_errors[i])
	}

	result.converged = result.max_backward_error < builtin.F64_EPSILON * 10.0

	return result, info_val
}

// ===================================================================================
// SIMPLE SOLVER IMPLEMENTATION
// ===================================================================================

// Solve tridiagonal positive definite system (c64)
m_solve_tridiagonal_positive_definite_c64 :: proc(
	D: []f32, // Diagonal (destroyed)
	E: []complex64, // Off-diagonal (destroyed)
	B: ^Matrix(complex64), // RHS (replaced with solution)
) -> (
	info: Info,
) {
	n := len(D)
	if len(E) != n - 1 && n > 1 {
		panic("Off-diagonal must have n-1 elements")
	}
	if B.rows != n {
		panic("RHS dimension mismatch")
	}

	n_val := Blas_Int(n)
	nrhs := Blas_Int(B.cols)
	ldb := Blas_Int(B.stride)

	lapack.cptsv_(&n_val, &nrhs, raw_data(D), raw_data(E), cast(^complex64)B.data, &ldb, &info)

	return info
}

// Solve tridiagonal positive definite system (f64)
m_solve_tridiagonal_positive_definite_f64 :: proc(
	D: []f64, // Diagonal (destroyed)
	E: []f64, // Off-diagonal (destroyed)
	B: ^Matrix(f64), // RHS (replaced with solution)
) -> (
	info: Info,
) {
	n := len(D)
	if len(E) != n - 1 && n > 1 {
		panic("Off-diagonal must have n-1 elements")
	}
	if B.rows != n {
		panic("RHS dimension mismatch")
	}

	n_val := Blas_Int(n)
	nrhs := Blas_Int(B.cols)
	ldb := Blas_Int(B.stride)

	lapack.dptsv_(&n_val, &nrhs, raw_data(D), raw_data(E), cast(^f64)B.data, &ldb, &info)

	return info
}

// Solve tridiagonal positive definite system (f32)
m_solve_tridiagonal_positive_definite_f32 :: proc(
	D: []f32, // Diagonal (destroyed)
	E: []f32, // Off-diagonal (destroyed)
	B: ^Matrix(f32), // RHS (replaced with solution)
) -> (
	info: Info,
) {
	n := len(D)
	if len(E) != n - 1 && n > 1 {
		panic("Off-diagonal must have n-1 elements")
	}
	if B.rows != n {
		panic("RHS dimension mismatch")
	}

	n_val := Blas_Int(n)
	nrhs := Blas_Int(B.cols)
	ldb := Blas_Int(B.stride)

	lapack.sptsv_(&n_val, &nrhs, raw_data(D), raw_data(E), cast(^f32)B.data, &ldb, &info)

	return info
}

// Solve tridiagonal positive definite system (c128)
m_solve_tridiagonal_positive_definite_c128 :: proc(
	D: []f64, // Diagonal (destroyed)
	E: []complex128, // Off-diagonal (destroyed)
	B: ^Matrix(complex128), // RHS (replaced with solution)
) -> (
	info: Info,
) {
	n := len(D)
	if len(E) != n - 1 && n > 1 {
		panic("Off-diagonal must have n-1 elements")
	}
	if B.rows != n {
		panic("RHS dimension mismatch")
	}

	n_val := Blas_Int(n)
	nrhs := Blas_Int(B.cols)
	ldb := Blas_Int(B.stride)

	lapack.zptsv_(&n_val, &nrhs, raw_data(D), raw_data(E), cast(^complex128)B.data, &ldb, &info)

	return info
}

// ===================================================================================
// CONVENIENCE FUNCTIONS
// ===================================================================================

// Complete eigenvalue analysis of tridiagonal matrix
analyze_tridiagonal_eigenvalues :: proc(
	D: []$T, // Diagonal elements
	E: []$S, // Off-diagonal elements
	compute_vectors := false, // Whether to compute eigenvectors
	allocator := context.allocator,
) -> (
	result: TridiagonalEigenResult(T),
	success: bool,
) {
	n := len(D)

	// Clone arrays since they are modified
	D_work := make([]T, n, allocator)
	copy(D_work, D)
	defer delete(D_work)

	E_work := make([]S, n - 1, allocator) if n > 1 else make([]S, 0, allocator)
	if n > 1 {
		copy(E_work, E)
	}
	defer delete(E_work)

	// Prepare eigenvector matrix if requested
	mode := EigenvectorMode.Identity if compute_vectors else .None
	if compute_vectors {
		result.eigenvectors = create_identity_matrix(T, n, allocator)
	}

	// Compute eigenvalues (and eigenvectors)
	var; info: Info
	when T == f32 && S == f32 {
		info = m_eigenvalues_tridiagonal_f32(D_work, E_work, &result.eigenvectors, mode, allocator)
	} else when T == f64 && S == f64 {
		info = m_eigenvalues_tridiagonal_f64(D_work, E_work, &result.eigenvectors, mode, allocator)
	}
	// Add other type combinations as needed

	if info != 0 {
		if compute_vectors {
			matrix_delete(&result.eigenvectors)
		}
		return result, false
	}

	// Eigenvalues are now in D_work, sorted
	result.eigenvalues = D_work

	// Analyze eigenvalues
	result.all_positive = true
	result.min_eigenvalue = math.INF_F64
	result.max_eigenvalue = -math.INF_F64

	for eigenval in D_work {
		val := f64(eigenval)
		if val <= 0 {
			result.all_positive = false
		}
		result.min_eigenvalue = min(result.min_eigenvalue, val)
		result.max_eigenvalue = max(result.max_eigenvalue, val)
	}

	// Compute condition number
	if result.min_eigenvalue > 0 {
		result.condition_number = result.max_eigenvalue / result.min_eigenvalue
	} else {
		result.condition_number = math.INF_F64
	}

	return result, true
}

// Solve tridiagonal system with automatic refinement
solve_tridiagonal_with_refinement :: proc(
	D: []$T, // Diagonal
	E: []$S, // Off-diagonal
	B: ^Matrix($U), // RHS
	max_iterations := 5,
	tolerance := 1e-14,
	allocator := context.allocator,
) -> (
	X: Matrix(U),
	converged: bool,
) {
	n := len(D)

	// Clone arrays for factorization
	DF := make([]T, n, allocator)
	copy(DF, D)
	defer delete(DF)

	EF := make([]S, n - 1, allocator) if n > 1 else make([]S, 0, allocator)
	if n > 1 {
		copy(EF, E)
	}
	defer delete(EF)

	// Clone RHS as initial solution
	X = matrix_clone(B, allocator)

	// Initial solve
	when T == f32 && S == complex64 && U == complex64 {
		info := m_solve_tridiagonal_positive_definite_c64(DF, EF, &X)
		if info != 0 {
			return X, false
		}
	} else when T == f64 && S == f64 && U == f64 {
		info := m_solve_tridiagonal_positive_definite_f64(DF, EF, &X)
		if info != 0 {
			return X, false
		}
	}
	// Add other type combinations

	// Iterative refinement
	for iter in 0 ..< max_iterations {
		when T == f32 && S == complex64 && U == complex64 {
			result, _ := m_refine_tridiagonal_positive_definite_c64(
				D,
				E,
				DF,
				EF,
				B,
				&X,
				true,
				allocator,
			)
			defer {
				delete(result.forward_errors)
				delete(result.backward_errors)
			}
			if result.converged || result.max_backward_error < tolerance {
				return X, true
			}
		} else when T == f64 && S == f64 && U == f64 {
			result, _ := m_refine_tridiagonal_positive_definite_f64(D, E, DF, EF, B, &X, allocator)
			defer {
				delete(result.forward_errors)
				delete(result.backward_errors)
			}
			if result.converged || result.max_backward_error < tolerance {
				return X, true
			}
		}
	}

	return X, false
}

// Check if tridiagonal matrix is positive definite
is_tridiagonal_positive_definite :: proc(
	D: []$T, // Diagonal
	E: []$S, // Off-diagonal
	allocator := context.temp_allocator,
) -> bool {
	// Use Sylvester's criterion or eigenvalue check
	result, success := analyze_tridiagonal_eigenvalues(D, E, false, allocator)
	defer if result.eigenvalues != nil do delete(result.eigenvalues)

	return success && result.all_positive
}

// Create random tridiagonal positive definite matrix
create_random_tridiagonal_pd :: proc(
	n: int,
	min_eigenvalue := 0.1,
	max_eigenvalue := 10.0,
	allocator := context.allocator,
) -> (
	D: []f64,
	E: []f64,
) {
	// Create random eigenvalues in specified range
	eigenvalues := make([]f64, n, allocator)
	for i in 0 ..< n {
		t := f64(i) / f64(n - 1) if n > 1 else 0.0
		eigenvalues[i] = min_eigenvalue + t * (max_eigenvalue - min_eigenvalue)
	}

	// Create random orthogonal matrix via Householder reflections
	// For simplicity, use a simple tridiagonal with known properties
	D = make([]f64, n, allocator)
	E = make([]f64, n - 1, allocator) if n > 1 else make([]f64, 0, allocator)

	// Simple construction ensuring positive definiteness
	for i in 0 ..< n {
		D[i] = 2.0 + f64(i) * 0.5 // Diagonal dominance
	}

	for i in 0 ..< n - 1 {
		E[i] = -0.5 // Small off-diagonal
	}

	return D, E
}

// Bandwidth-efficient matrix-vector multiply for tridiagonal
tridiagonal_matvec :: proc(
	D: []$T, // Diagonal
	E: []$S, // Off-diagonal
	x: ^Vector($U), // Input vector
	y: ^Vector(U), // Output vector
) {
	n := len(D)
	if x.len != n || y.len != n {
		panic("Vector dimension mismatch")
	}

	// y[0] = D[0]*x[0] + E[0]*x[1]
	if n > 0 {
		val := U(D[0]) * vector_get(x, 0)
		if n > 1 {
			val += U(E[0]) * vector_get(x, 1)
		}
		vector_set(y, 0, val)
	}

	// y[i] = E[i-1]*x[i-1] + D[i]*x[i] + E[i]*x[i+1]
	for i in 1 ..< n - 1 {
		val :=
			U(E[i - 1]) * vector_get(x, i - 1) +
			U(D[i]) * vector_get(x, i) +
			U(E[i]) * vector_get(x, i + 1)
		vector_set(y, i, val)
	}

	// y[n-1] = E[n-2]*x[n-2] + D[n-1]*x[n-1]
	if n > 1 {
		val := U(E[n - 2]) * vector_get(x, n - 2) + U(D[n - 1]) * vector_get(x, n - 1)
		vector_set(y, n - 1, val)
	}
}

// Helper function
