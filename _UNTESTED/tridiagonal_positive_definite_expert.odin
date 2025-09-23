package openblas

import lapack "./f77"
import "base:builtin"
import "core:math"
import "core:mem"

// ===================================================================================
// TRIDIAGONAL POSITIVE DEFINITE EXPERT SOLVER
// ===================================================================================

// Expert solver proc group
m_solve_tridiagonal_positive_definite_expert :: proc {
	m_solve_tridiagonal_positive_definite_expert_c64,
	m_solve_tridiagonal_positive_definite_expert_f64,
	m_solve_tridiagonal_positive_definite_expert_f32,
	m_solve_tridiagonal_positive_definite_expert_c128,
}

// ===================================================================================
// TRIDIAGONAL EXPERT RESULT STRUCTURE
// ===================================================================================

// Expert solver result for tridiagonal systems
TridiagonalExpertResult :: struct($T: typeid) {
	rcond:                f64, // Reciprocal condition number
	forward_errors:       []T, // Forward error bounds for each RHS
	backward_errors:      []T, // Backward error bounds for each RHS
	is_singular:          bool, // True if matrix is singular
	factorization_reused: bool, // True if provided factorization was used
	max_forward_error:    f64, // Maximum forward error
	max_backward_error:   f64, // Maximum backward error
	condition_number:     f64, // Actual condition number (1/rcond)
}

// ===================================================================================
// EXPERT SOLVER IMPLEMENTATION
// ===================================================================================

// Expert solver for tridiagonal positive definite system (c64)
// Provides full control over factorization and error bounds
m_solve_tridiagonal_positive_definite_expert_c64 :: proc(
	D: []f32, // Diagonal elements
	E: []complex64, // Off-diagonal elements
	DF: []f32, // Factored diagonal (input/output)
	EF: []complex64, // Factored off-diagonal (input/output)
	B: ^Matrix(complex64), // Right-hand side
	X: ^Matrix(complex64), // Solution (output)
	fact := FactorizationOption.Compute, // Factorization control
	allocator := context.allocator,
) -> (
	result: TridiagonalExpertResult(f32),
	info: Info,
) {
	// Validate inputs
	n := len(D)
	if len(E) != n - 1 && n > 1 {
		panic("Off-diagonal must have n-1 elements")
	}
	if len(DF) != n || (len(EF) != n - 1 && n > 1) {
		panic("Factored arrays dimension mismatch")
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
	fact_c := _fact_to_char(fact)

	// Allocate error arrays
	result.forward_errors = make([]f32, B.cols, allocator)
	result.backward_errors = make([]f32, B.cols, allocator)

	// Allocate workspace
	work := make([]complex64, n, allocator)
	defer delete(work)
	rwork := make([]f32, n, allocator)
	defer delete(rwork)

	rcond: f32
	info_val: Blas_Int

	// If computing factorization, copy D and E to DF and EF
	if fact == .Compute {
		copy(DF, D)
		if n > 1 {
			copy(EF, E)
		}
	}

	lapack.cptsvx_(
		fact_c,
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
		&rcond,
		raw_data(result.forward_errors),
		raw_data(result.backward_errors),
		raw_data(work),
		raw_data(rwork),
		&info_val,
		len(fact_c),
	)

	// Fill result fields
	result.rcond = f64(rcond)
	if rcond > 0 {
		result.condition_number = 1.0 / result.rcond
	} else {
		result.condition_number = math.INF_F64
	}
	result.is_singular = rcond < builtin.F32_EPSILON
	result.factorization_reused = fact == .UseProvided

	// Compute maximum errors
	for i in 0 ..< B.cols {
		result.max_forward_error = max(result.max_forward_error, f64(result.forward_errors[i]))
		result.max_backward_error = max(result.max_backward_error, f64(result.backward_errors[i]))
	}

	return result, info_val
}

// Expert solver for tridiagonal positive definite system (f64)
// Provides full control over factorization and error bounds
m_solve_tridiagonal_positive_definite_expert_f64 :: proc(
	D: []f64, // Diagonal elements
	E: []f64, // Off-diagonal elements
	DF: []f64, // Factored diagonal (input/output)
	EF: []f64, // Factored off-diagonal (input/output)
	B: ^Matrix(f64), // Right-hand side
	X: ^Matrix(f64), // Solution (output)
	fact := FactorizationOption.Compute, // Factorization control
	allocator := context.allocator,
) -> (
	result: TridiagonalExpertResult(f64),
	info: Info,
) {
	// Validate inputs
	n := len(D)
	if len(E) != n - 1 && n > 1 {
		panic("Off-diagonal must have n-1 elements")
	}
	if len(DF) != n || (len(EF) != n - 1 && n > 1) {
		panic("Factored arrays dimension mismatch")
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
	fact_c := _fact_to_char(fact)

	// Allocate error arrays
	result.forward_errors = make([]f64, B.cols, allocator)
	result.backward_errors = make([]f64, B.cols, allocator)

	// Allocate workspace
	work := make([]f64, 2 * n, allocator)
	defer delete(work)

	rcond: f64
	info_val: Blas_Int

	// If computing factorization, copy D and E to DF and EF
	if fact == .Compute {
		copy(DF, D)
		if n > 1 {
			copy(EF, E)
		}
	}

	lapack.dptsvx_(
		fact_c,
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
		&rcond,
		raw_data(result.forward_errors),
		raw_data(result.backward_errors),
		raw_data(work),
		&info_val,
		len(fact_c),
	)

	// Fill result fields
	result.rcond = rcond
	if rcond > 0 {
		result.condition_number = 1.0 / rcond
	} else {
		result.condition_number = math.INF_F64
	}
	result.is_singular = rcond < builtin.F64_EPSILON
	result.factorization_reused = fact == .UseProvided

	// Compute maximum errors
	for i in 0 ..< B.cols {
		result.max_forward_error = max(result.max_forward_error, result.forward_errors[i])
		result.max_backward_error = max(result.max_backward_error, result.backward_errors[i])
	}

	return result, info_val
}

// Expert solver for tridiagonal positive definite system (f32)
// Provides full control over factorization and error bounds
m_solve_tridiagonal_positive_definite_expert_f32 :: proc(
	D: []f32, // Diagonal elements
	E: []f32, // Off-diagonal elements
	DF: []f32, // Factored diagonal (input/output)
	EF: []f32, // Factored off-diagonal (input/output)
	B: ^Matrix(f32), // Right-hand side
	X: ^Matrix(f32), // Solution (output)
	fact := FactorizationOption.Compute, // Factorization control
	allocator := context.allocator,
) -> (
	result: TridiagonalExpertResult(f32),
	info: Info,
) {
	// Validate inputs
	n := len(D)
	if len(E) != n - 1 && n > 1 {
		panic("Off-diagonal must have n-1 elements")
	}
	if len(DF) != n || (len(EF) != n - 1 && n > 1) {
		panic("Factored arrays dimension mismatch")
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
	fact_c := _fact_to_char(fact)

	// Allocate error arrays
	result.forward_errors = make([]f32, B.cols, allocator)
	result.backward_errors = make([]f32, B.cols, allocator)

	// Allocate workspace
	work := make([]f32, 2 * n, allocator)
	defer delete(work)

	rcond: f32
	info_val: Blas_Int

	// If computing factorization, copy D and E to DF and EF
	if fact == .Compute {
		copy(DF, D)
		if n > 1 {
			copy(EF, E)
		}
	}

	lapack.sptsvx_(
		fact_c,
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
		&rcond,
		raw_data(result.forward_errors),
		raw_data(result.backward_errors),
		raw_data(work),
		&info_val,
		len(fact_c),
	)

	// Fill result fields
	result.rcond = f64(rcond)
	if rcond > 0 {
		result.condition_number = 1.0 / result.rcond
	} else {
		result.condition_number = math.INF_F64
	}
	result.is_singular = rcond < builtin.F32_EPSILON
	result.factorization_reused = fact == .UseProvided

	// Compute maximum errors
	for i in 0 ..< B.cols {
		result.max_forward_error = max(result.max_forward_error, f64(result.forward_errors[i]))
		result.max_backward_error = max(result.max_backward_error, f64(result.backward_errors[i]))
	}

	return result, info_val
}

// Expert solver for tridiagonal positive definite system (c128)
// Provides full control over factorization and error bounds
m_solve_tridiagonal_positive_definite_expert_c128 :: proc(
	D: []f64, // Diagonal elements
	E: []complex128, // Off-diagonal elements
	DF: []f64, // Factored diagonal (input/output)
	EF: []complex128, // Factored off-diagonal (input/output)
	B: ^Matrix(complex128), // Right-hand side
	X: ^Matrix(complex128), // Solution (output)
	fact := FactorizationOption.Compute, // Factorization control
	allocator := context.allocator,
) -> (
	result: TridiagonalExpertResult(f64),
	info: Info,
) {
	// Validate inputs
	n := len(D)
	if len(E) != n - 1 && n > 1 {
		panic("Off-diagonal must have n-1 elements")
	}
	if len(DF) != n || (len(EF) != n - 1 && n > 1) {
		panic("Factored arrays dimension mismatch")
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
	fact_c := _fact_to_char(fact)

	// Allocate error arrays
	result.forward_errors = make([]f64, B.cols, allocator)
	result.backward_errors = make([]f64, B.cols, allocator)

	// Allocate workspace
	work := make([]complex128, n, allocator)
	defer delete(work)
	rwork := make([]f64, n, allocator)
	defer delete(rwork)

	rcond: f64
	info_val: Blas_Int

	// If computing factorization, copy D and E to DF and EF
	if fact == .Compute {
		copy(DF, D)
		if n > 1 {
			copy(EF, E)
		}
	}

	lapack.zptsvx_(
		fact_c,
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
		&rcond,
		raw_data(result.forward_errors),
		raw_data(result.backward_errors),
		raw_data(work),
		raw_data(rwork),
		&info_val,
		len(fact_c),
	)

	// Fill result fields
	result.rcond = rcond
	if rcond > 0 {
		result.condition_number = 1.0 / rcond
	} else {
		result.condition_number = math.INF_F64
	}
	result.is_singular = rcond < builtin.F64_EPSILON
	result.factorization_reused = fact == .UseProvided

	// Compute maximum errors
	for i in 0 ..< B.cols {
		result.max_forward_error = max(result.max_forward_error, result.forward_errors[i])
		result.max_backward_error = max(result.max_backward_error, result.backward_errors[i])
	}

	return result, info_val
}

// ===================================================================================
// CONVENIENCE FUNCTIONS
// ===================================================================================

// Solve tridiagonal system with automatic condition estimation
solve_tridiagonal_with_conditioning :: proc(
	D: []$T, // Diagonal elements
	E: []$S, // Off-diagonal elements
	B: ^Matrix($U), // Right-hand side
	allocator := context.allocator,
) -> (
	X: Matrix(U),
	condition: f64,
	success: bool,
) {
	n := len(D)

	// Allocate factorization arrays
	DF := make([]T, n, allocator)
	defer delete(DF)

	EF := make([]S, n - 1, allocator) if n > 1 else make([]S, 0, allocator)
	defer delete(EF)

	// Allocate solution matrix
	X = create_matrix(U, B.rows, B.cols, allocator)

	// Use expert solver
	when T == f32 && S == complex64 && U == complex64 {
		result, info := m_solve_tridiagonal_positive_definite_expert_c64(
			D,
			E,
			DF,
			EF,
			B,
			&X,
			.Compute,
			allocator,
		)
		defer {
			delete(result.forward_errors)
			delete(result.backward_errors)
		}
		condition = result.condition_number
		success = info == 0 && !result.is_singular
	} else when T == f64 && S == f64 && U == f64 {
		result, info := m_solve_tridiagonal_positive_definite_expert_f64(
			D,
			E,
			DF,
			EF,
			B,
			&X,
			.Compute,
			allocator,
		)
		defer {
			delete(result.forward_errors)
			delete(result.backward_errors)
		}
		condition = result.condition_number
		success = info == 0 && !result.is_singular
	} else when T == f32 && S == f32 && U == f32 {
		result, info := m_solve_tridiagonal_positive_definite_expert_f32(
			D,
			E,
			DF,
			EF,
			B,
			&X,
			.Compute,
			allocator,
		)
		defer {
			delete(result.forward_errors)
			delete(result.backward_errors)
		}
		condition = result.condition_number
		success = info == 0 && !result.is_singular
	} else when T == f64 && S == complex128 && U == complex128 {
		result, info := m_solve_tridiagonal_positive_definite_expert_c128(
			D,
			E,
			DF,
			EF,
			B,
			&X,
			.Compute,
			allocator,
		)
		defer {
			delete(result.forward_errors)
			delete(result.backward_errors)
		}
		condition = result.condition_number
		success = info == 0 && !result.is_singular
	}

	if !success {
		matrix_delete(&X)
		X = Matrix(U){}
	}

	return X, condition, success
}

// Solve multiple tridiagonal systems with factorization reuse
solve_tridiagonal_multiple :: proc(
	D: []$T, // Diagonal elements
	E: []$S, // Off-diagonal elements
	B_list: []^Matrix($U), // List of RHS matrices
	allocator := context.allocator,
) -> (
	X_list: []Matrix(U),
	all_success: bool,
) {
	if len(B_list) == 0 {
		return nil, false
	}

	n := len(D)
	X_list = make([]Matrix(U), len(B_list), allocator)
	all_success = true

	// Allocate factorization arrays once
	DF := make([]T, n, allocator)
	defer delete(DF)

	EF := make([]S, n - 1, allocator) if n > 1 else make([]S, 0, allocator)
	defer delete(EF)

	// Solve first system and compute factorization
	fact_option := FactorizationOption.Compute

	for i, B in B_list {
		X_list[i] = create_matrix(U, B.rows, B.cols, allocator)

		// Reuse factorization after first solve
		if i > 0 {
			fact_option = .UseProvided
		}

		when T == f64 && S == f64 && U == f64 {
			result, info := m_solve_tridiagonal_positive_definite_expert_f64(
				D,
				E,
				DF,
				EF,
				B,
				&X_list[i],
				fact_option,
				allocator,
			)
			defer {
				delete(result.forward_errors)
				delete(result.backward_errors)
			}
			if info != 0 || result.is_singular {
				all_success = false
			}
		}
		// Add other type combinations as needed
	}

	return X_list, all_success
}

// Analyze tridiagonal system solution quality
analyze_tridiagonal_solution :: proc(
	D: []$T, // Diagonal elements
	E: []$S, // Off-diagonal elements
	B: ^Matrix($U), // Right-hand side
	allocator := context.allocator,
) -> TridiagonalSolutionAnalysis {
	analysis: TridiagonalSolutionAnalysis

	n := len(D)

	// Allocate arrays
	DF := make([]T, n, allocator)
	defer delete(DF)

	EF := make([]S, n - 1, allocator) if n > 1 else make([]S, 0, allocator)
	defer delete(EF)

	X := create_matrix(U, B.rows, B.cols, allocator)
	defer matrix_delete(&X)

	// Solve with expert solver
	when T == f64 && S == f64 && U == f64 {
		result, info := m_solve_tridiagonal_positive_definite_expert_f64(
			D,
			E,
			DF,
			EF,
			B,
			&X,
			.Compute,
			allocator,
		)
		defer {
			delete(result.forward_errors)
			delete(result.backward_errors)
		}

		analysis.success = info == 0
		analysis.is_singular = result.is_singular
		analysis.condition_number = result.condition_number
		analysis.rcond = result.rcond
		analysis.max_forward_error = result.max_forward_error
		analysis.max_backward_error = result.max_backward_error

		// Determine quality
		if result.is_singular {
			analysis.quality = .Singular
		} else if result.condition_number > 1e15 {
			analysis.quality = .IllConditioned
		} else if result.condition_number > 1e10 {
			analysis.quality = .Poor
		} else if result.condition_number > 1e6 {
			analysis.quality = .Fair
		} else if result.condition_number > 1e3 {
			analysis.quality = .Good
		} else {
			analysis.quality = .Excellent
		}

		// Estimate digits of accuracy
		if result.rcond > 0 {
			analysis.estimated_accuracy_digits = -math.log10(result.max_backward_error)
		}
	}
	// Add other type combinations

	return analysis
}

// Tridiagonal solution analysis structure
TridiagonalSolutionAnalysis :: struct {
	success:                   bool,
	is_singular:               bool,
	condition_number:          f64,
	rcond:                     f64,
	max_forward_error:         f64,
	max_backward_error:        f64,
	quality:                   SolutionQuality,
	estimated_accuracy_digits: f64,
}

SolutionQuality :: enum {
	Excellent,
	Good,
	Fair,
	Poor,
	IllConditioned,
	Singular,
}

// Check solution accuracy for tridiagonal system
check_tridiagonal_solution :: proc(
	D: []$T, // Diagonal
	E: []$S, // Off-diagonal
	B: ^Matrix($U), // Original RHS
	X: ^Matrix(U), // Computed solution
	allocator := context.allocator,
) -> (
	residual_norm: f64,
	relative_error: f64,
) {
	n := len(D)

	// Compute residual r = B - A*X
	residual := matrix_clone(B, allocator)
	defer matrix_delete(&residual)

	// Compute A*X using tridiagonal structure
	for j in 0 ..< X.cols {
		for i in 0 ..< n {
			ax_val := U(D[i]) * matrix_get(X, i, j)

			if i > 0 {
				ax_val += U(E[i - 1]) * matrix_get(X, i - 1, j)
			}
			if i < n - 1 {
				ax_val += U(E[i]) * matrix_get(X, i + 1, j)
			}

			// r[i,j] = b[i,j] - ax_val
			r_val := matrix_get(&residual, i, j) - ax_val
			matrix_set(&residual, i, j, r_val)
		}
	}

	// Compute norms
	residual_norm = 0.0
	b_norm := 0.0

	for j in 0 ..< B.cols {
		for i in 0 ..< n {
			r_val := matrix_get(&residual, i, j)
			b_val := matrix_get(B, i, j)

			when U == complex64 || U == complex128 {
				residual_norm += real(r_val * conj(r_val))
				b_norm += real(b_val * conj(b_val))
			} else {
				residual_norm += f64(r_val * r_val)
				b_norm += f64(b_val * b_val)
			}
		}
	}

	residual_norm = math.sqrt(residual_norm)
	b_norm = math.sqrt(b_norm)

	if b_norm > 0 {
		relative_error = residual_norm / b_norm
	} else {
		relative_error = residual_norm
	}

	return residual_norm, relative_error
}

// Compare simple vs expert solver for tridiagonal systems
compare_tridiagonal_solvers :: proc(
	D: []$T, // Diagonal
	E: []$S, // Off-diagonal
	B: ^Matrix($U), // RHS
	allocator := context.allocator,
) -> TridiagonalSolverComparison {
	comparison: TridiagonalSolverComparison

	// Simple solver
	D_simple := make([]T, len(D), allocator)
	copy(D_simple, D)
	defer delete(D_simple)

	E_simple := make([]S, len(E), allocator) if len(E) > 0 else make([]S, 0, allocator)
	if len(E) > 0 {
		copy(E_simple, E)
	}
	defer delete(E_simple)

	X_simple := matrix_clone(B, allocator)
	defer matrix_delete(&X_simple)

	when T == f64 && S == f64 && U == f64 {
		info_simple := m_solve_tridiagonal_positive_definite_f64(D_simple, E_simple, &X_simple)
		comparison.simple_success = info_simple == 0
	}

	// Expert solver
	X_expert, condition, expert_success := solve_tridiagonal_with_conditioning(D, E, B, allocator)
	defer matrix_delete(&X_expert)

	comparison.expert_success = expert_success
	comparison.condition_number = condition

	// Compare solutions if both succeeded
	if comparison.simple_success && comparison.expert_success {
		diff := 0.0
		for j in 0 ..< B.cols {
			for i in 0 ..< B.rows {
				val_simple := matrix_get(&X_simple, i, j)
				val_expert := matrix_get(&X_expert, i, j)
				when U == complex64 || U == complex128 {
					d := val_simple - val_expert
					diff += real(d * conj(d))
				} else {
					d := f64(val_simple - val_expert)
					diff += d * d
				}
			}
		}
		comparison.solution_difference = math.sqrt(diff)
	}

	// Expert solver provides more diagnostics
	comparison.recommendation = .Expert if condition > 1e6 else .Simple

	return comparison
}

// Tridiagonal solver comparison structure
TridiagonalSolverComparison :: struct {
	simple_success:      bool,
	expert_success:      bool,
	condition_number:    f64,
	solution_difference: f64,
	recommendation:      SolverRecommendation,
}

SolverRecommendation :: enum {
	Simple,
	Expert,
}

// Helper function to convert factorization option
_fact_to_char :: proc(fact: FactorizationOption) -> cstring {
	switch fact {
	case .Compute:
		return "N"
	case .UseProvided:
		return "F"
	case:
		return "N"
	}
}
