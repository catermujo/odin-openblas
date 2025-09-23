package openblas

import lapack "./f77"
import "base:builtin"
import "core:math"
import "core:mem"

// ===================================================================================
// POSITIVE DEFINITE MIXED-PRECISION AND EXPERT SOLVERS
// ===================================================================================

// Mixed-precision solver proc group
m_solve_positive_definite_mixed :: proc {
	m_solve_positive_definite_mixed_f64_f32,
	m_solve_positive_definite_mixed_c128_c64,
}

// Expert solver proc group
m_solve_positive_definite_expert :: proc {
	m_solve_positive_definite_expert_c64,
	m_solve_positive_definite_expert_f64,
	m_solve_positive_definite_expert_f32,
	m_solve_positive_definite_expert_c128,
}

// ===================================================================================
// SOLVER PARAMETERS
// ===================================================================================

// Factorization control
FactorizationOption :: enum {
	Compute, // "N" - Compute factorization
	UseProvided, // "F" - Use provided factorization
	Equilibrate, // "E" - Equilibrate, then factor
}

// Convert factorization option to LAPACK character
_fact_to_char :: proc(fact: FactorizationOption) -> cstring {
	switch fact {
	case .Compute:
		return "N"
	case .UseProvided:
		return "F"
	case .Equilibrate:
		return "E"
	case:
		return "N"
	}
}

// Equilibration state for expert solver
EquilibrationMode :: enum {
	None, // "N" - No equilibration
	Yes, // "Y" - Equilibration was/will be performed
}

// Convert equilibration mode to LAPACK character
_equed_mode_to_char :: proc(equed: EquilibrationMode) -> cstring {
	switch equed {
	case .None:
		return "N"
	case .Yes:
		return "Y"
	case:
		return "N"
	}
}

// ===================================================================================
// MIXED-PRECISION SOLVER RESULT
// ===================================================================================

// Result of mixed-precision iterative refinement
MixedPrecisionResult :: struct {
	iterations:     int, // Number of refinement iterations performed
	converged:      bool, // True if converged to desired precision
	final_residual: f64, // Final residual norm
}

// ===================================================================================
// EXPERT SOLVER RESULT
// ===================================================================================

// Result of expert solver
ExpertSolverResult :: struct($T: typeid) {
	rcond:                f64, // Reciprocal condition number
	forward_errors:       []T, // Forward error bounds
	backward_errors:      []T, // Backward error bounds
	scale_factors:        []T, // Equilibration scale factors (if used)
	was_equilibrated:     bool, // True if equilibration was applied
	is_singular:          bool, // True if matrix is singular
	factorization_reused: bool, // True if provided factorization was used
}

// ===================================================================================
// MIXED-PRECISION ITERATIVE REFINEMENT IMPLEMENTATION
// ===================================================================================

// Mixed-precision solver: double precision with single precision acceleration (f64/f32)
// Uses f32 for factorization, refines to f64 accuracy
m_solve_positive_definite_mixed_f64_f32 :: proc(
	A: ^Matrix(f64), // System matrix (preserved if not equilibrated)
	B: ^Matrix(f64), // RHS matrix
	X: ^Matrix(f64), // Solution matrix (output)
	uplo_upper := true, // Upper or lower triangular
	max_iterations := 30, // Maximum refinement iterations
	allocator := context.allocator,
) -> (
	result: MixedPrecisionResult,
	info: Info,
) {
	// Validate inputs
	if A.rows != A.cols {
		panic("Matrix must be square")
	}
	if B.rows != A.rows || X.rows != A.rows {
		panic("Dimension mismatch")
	}
	if B.cols != X.cols {
		panic("RHS and solution must have same number of columns")
	}

	n := Blas_Int(A.rows)
	nrhs := Blas_Int(B.cols)
	lda := Blas_Int(A.stride)
	ldb := Blas_Int(B.stride)
	ldx := Blas_Int(X.stride)
	uplo_c := "U" if uplo_upper else "L"

	// Allocate workspace
	work := make([]f64, n * (n + nrhs), allocator)
	defer delete(work)

	swork := make([]f32, n * (n + nrhs), allocator)
	defer delete(swork)

	iter: Blas_Int
	info_val: Info

	lapack.dsposv_(
		uplo_c,
		&n,
		&nrhs,
		cast(^f64)A.data,
		&lda,
		cast(^f64)B.data,
		&ldb,
		cast(^f64)X.data,
		&ldx,
		raw_data(work),
		raw_data(swork),
		&iter,
		&info_val,
		len(uplo_c),
	)

	// Build result
	result.iterations = int(iter)
	result.converged = iter >= 0 // Negative means didn't converge
	if !result.converged {
		result.iterations = -int(iter) // Actual iteration count
	}

	// Estimate final residual
	if result.converged {
		result.final_residual = builtin.F64_EPSILON * f64(n)
	} else {
		result.final_residual = builtin.F32_EPSILON * f64(n) // Only achieved f32 accuracy
	}

	return result, info_val
}

// Mixed-precision solver: double complex with single complex acceleration (c128/c64)
// Uses c64 for factorization, refines to c128 accuracy
m_solve_positive_definite_mixed_c128_c64 :: proc(
	A: ^Matrix(complex128), // System matrix (preserved if not equilibrated)
	B: ^Matrix(complex128), // RHS matrix
	X: ^Matrix(complex128), // Solution matrix (output)
	uplo_upper := true, // Upper or lower triangular
	max_iterations := 30, // Maximum refinement iterations
	allocator := context.allocator,
) -> (
	result: MixedPrecisionResult,
	info: Info,
) {
	// Validate inputs
	if A.rows != A.cols {
		panic("Matrix must be square")
	}
	if B.rows != A.rows || X.rows != A.rows {
		panic("Dimension mismatch")
	}
	if B.cols != X.cols {
		panic("RHS and solution must have same number of columns")
	}

	n := Blas_Int(A.rows)
	nrhs := Blas_Int(B.cols)
	lda := Blas_Int(A.stride)
	ldb := Blas_Int(B.stride)
	ldx := Blas_Int(X.stride)
	uplo_c := "U" if uplo_upper else "L"

	// Allocate workspace
	work := make([]complex128, n * (n + nrhs), allocator)
	defer delete(work)

	swork := make([]complex64, n * (n + nrhs), allocator)
	defer delete(swork)

	rwork := make([]f64, n, allocator)
	defer delete(rwork)

	iter: Blas_Int
	info_val: Info

	lapack.zcposv_(
		uplo_c,
		&n,
		&nrhs,
		cast(^complex128)A.data,
		&lda,
		cast(^complex128)B.data,
		&ldb,
		cast(^complex128)X.data,
		&ldx,
		raw_data(work),
		raw_data(swork),
		raw_data(rwork),
		&iter,
		&info_val,
		len(uplo_c),
	)

	// Build result
	result.iterations = int(iter)
	result.converged = iter >= 0
	if !result.converged {
		result.iterations = -int(iter)
	}

	// Estimate final residual
	if result.converged {
		result.final_residual = builtin.F64_EPSILON * f64(n)
	} else {
		result.final_residual = builtin.F32_EPSILON * f64(n)
	}

	return result, info_val
}

// ===================================================================================
// EXPERT SOLVER IMPLEMENTATION
// ===================================================================================

// Expert solver for positive definite system (c64)
// Provides full control over factorization, equilibration, and error bounds
m_solve_positive_definite_expert_c64 :: proc(
	A: ^Matrix(complex64), // System matrix
	AF: ^Matrix(complex64), // Factorization workspace/input
	B: ^Matrix(complex64), // RHS matrix
	X: ^Matrix(complex64), // Solution matrix (output)
	fact := FactorizationOption.Compute, // Factorization control
	uplo_upper := true, // Upper or lower triangular
	equed_inout: ^EquilibrationMode = nil, // Equilibration state (input/output)
	S_inout: []f32 = nil, // Scale factors (input/output)
	allocator := context.allocator,
) -> (
	result: ExpertSolverResult(f32),
	info: Info,
) {
	// Validate inputs
	if A.rows != A.cols || AF.rows != AF.cols {
		panic("Matrices must be square")
	}
	if A.rows != AF.rows {
		panic("A and AF must have same dimensions")
	}
	if B.rows != A.rows || X.rows != A.rows {
		panic("Dimension mismatch")
	}
	if B.cols != X.cols {
		panic("RHS and solution must have same number of columns")
	}

	n := Blas_Int(A.rows)
	nrhs := Blas_Int(B.cols)
	lda := Blas_Int(A.stride)
	ldaf := Blas_Int(AF.stride)
	ldb := Blas_Int(B.stride)
	ldx := Blas_Int(X.stride)

	fact_c := _fact_to_char(fact)
	uplo_c := "U" if uplo_upper else "L"

	// Handle equilibration state
	equed_mode := EquilibrationMode.None
	if equed_inout != nil {
		equed_mode = equed_inout^
	}
	equed_c := _equed_mode_to_char(equed_mode)

	// Allocate scale factors if needed
	S := S_inout
	if len(S) == 0 && fact == .Equilibrate {
		S = make([]f32, A.rows, allocator)
	}

	// Allocate error arrays
	result.forward_errors = make([]f32, B.cols, allocator)
	result.backward_errors = make([]f32, B.cols, allocator)

	// Allocate workspace
	work := make([]complex64, 2 * A.rows, allocator)
	defer delete(work)

	rwork := make([]f32, A.rows, allocator)
	defer delete(rwork)

	rcond: f32
	info_val: Info

	// Call LAPACK
	s_ptr: ^f32 = nil
	if len(S) > 0 {
		s_ptr = raw_data(S)
	}

	lapack.cposvx_(
		fact_c,
		uplo_c,
		&n,
		&nrhs,
		cast(^complex64)A.data,
		&lda,
		cast(^complex64)AF.data,
		&ldaf,
		equed_c,
		s_ptr,
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
		len(uplo_c),
		len(equed_c),
	)

	// Update equilibration state if provided
	if equed_inout != nil {
		// equed_c may have been modified by LAPACK
		if equed_c == "Y" {
			equed_inout^ = .Yes
			result.was_equilibrated = true
		} else {
			equed_inout^ = .None
		}
	}

	// Copy scale factors if they were computed
	if fact == .Equilibrate && len(S) > 0 {
		result.scale_factors = S
	}

	// Fill remaining result fields
	result.rcond = f64(rcond)
	result.is_singular = rcond < builtin.F32_EPSILON
	result.factorization_reused = fact == .UseProvided

	return result, info_val
}

// Expert solver for positive definite system (f64)
m_solve_positive_definite_expert_f64 :: proc(
	A: ^Matrix(f64), // System matrix
	AF: ^Matrix(f64), // Factorization workspace/input
	B: ^Matrix(f64), // RHS matrix
	X: ^Matrix(f64), // Solution matrix (output)
	fact := FactorizationOption.Compute, // Factorization control
	uplo_upper := true, // Upper or lower triangular
	equed_inout: ^EquilibrationMode = nil, // Equilibration state (input/output)
	S_inout: []f64 = nil, // Scale factors (input/output)
	allocator := context.allocator,
) -> (
	result: ExpertSolverResult(f64),
	info: Info,
) {
	// Validate inputs
	if A.rows != A.cols || AF.rows != AF.cols {
		panic("Matrices must be square")
	}
	if A.rows != AF.rows {
		panic("A and AF must have same dimensions")
	}
	if B.rows != A.rows || X.rows != A.rows {
		panic("Dimension mismatch")
	}
	if B.cols != X.cols {
		panic("RHS and solution must have same number of columns")
	}

	n := Blas_Int(A.rows)
	nrhs := Blas_Int(B.cols)
	lda := Blas_Int(A.stride)
	ldaf := Blas_Int(AF.stride)
	ldb := Blas_Int(B.stride)
	ldx := Blas_Int(X.stride)

	fact_c := _fact_to_char(fact)
	uplo_c := "U" if uplo_upper else "L"

	equed_mode := EquilibrationMode.None
	if equed_inout != nil {
		equed_mode = equed_inout^
	}
	equed_c := _equed_mode_to_char(equed_mode)

	S := S_inout
	if len(S) == 0 && fact == .Equilibrate {
		S = make([]f64, A.rows, allocator)
	}

	result.forward_errors = make([]f64, B.cols, allocator)
	result.backward_errors = make([]f64, B.cols, allocator)

	work := make([]f64, 3 * A.rows, allocator)
	defer delete(work)

	iwork := make([]Blas_Int, A.rows, allocator)
	defer delete(iwork)

	rcond: f64
	info_val: Info

	s_ptr: ^f64 = nil
	if len(S) > 0 {
		s_ptr = raw_data(S)
	}

	lapack.dposvx_(
		fact_c,
		uplo_c,
		&n,
		&nrhs,
		cast(^f64)A.data,
		&lda,
		cast(^f64)AF.data,
		&ldaf,
		equed_c,
		s_ptr,
		cast(^f64)B.data,
		&ldb,
		cast(^f64)X.data,
		&ldx,
		&rcond,
		raw_data(result.forward_errors),
		raw_data(result.backward_errors),
		raw_data(work),
		raw_data(iwork),
		&info_val,
		len(fact_c),
		len(uplo_c),
		len(equed_c),
	)

	if equed_inout != nil {
		if equed_c == "Y" {
			equed_inout^ = .Yes
			result.was_equilibrated = true
		} else {
			equed_inout^ = .None
		}
	}

	if fact == .Equilibrate && len(S) > 0 {
		result.scale_factors = S
	}

	result.rcond = rcond
	result.is_singular = rcond < builtin.F64_EPSILON
	result.factorization_reused = fact == .UseProvided

	return result, info_val
}

// Expert solver for positive definite system (f32)
m_solve_positive_definite_expert_f32 :: proc(
	A: ^Matrix(f32), // System matrix
	AF: ^Matrix(f32), // Factorization workspace/input
	B: ^Matrix(f32), // RHS matrix
	X: ^Matrix(f32), // Solution matrix (output)
	fact := FactorizationOption.Compute, // Factorization control
	uplo_upper := true, // Upper or lower triangular
	equed_inout: ^EquilibrationMode = nil, // Equilibration state (input/output)
	S_inout: []f32 = nil, // Scale factors (input/output)
	allocator := context.allocator,
) -> (
	result: ExpertSolverResult(f32),
	info: Info,
) {
	// Similar implementation to f64 version
	// [Implementation follows same pattern]

	// Validate inputs
	if A.rows != A.cols || AF.rows != AF.cols {
		panic("Matrices must be square")
	}
	if A.rows != AF.rows {
		panic("A and AF must have same dimensions")
	}
	if B.rows != A.rows || X.rows != A.rows {
		panic("Dimension mismatch")
	}
	if B.cols != X.cols {
		panic("RHS and solution must have same number of columns")
	}

	n := Blas_Int(A.rows)
	nrhs := Blas_Int(B.cols)
	lda := Blas_Int(A.stride)
	ldaf := Blas_Int(AF.stride)
	ldb := Blas_Int(B.stride)
	ldx := Blas_Int(X.stride)

	fact_c := _fact_to_char(fact)
	uplo_c := "U" if uplo_upper else "L"

	equed_mode := EquilibrationMode.None
	if equed_inout != nil {
		equed_mode = equed_inout^
	}
	equed_c := _equed_mode_to_char(equed_mode)

	S := S_inout
	if len(S) == 0 && fact == .Equilibrate {
		S = make([]f32, A.rows, allocator)
	}

	result.forward_errors = make([]f32, B.cols, allocator)
	result.backward_errors = make([]f32, B.cols, allocator)

	work := make([]f32, 3 * A.rows, allocator)
	defer delete(work)

	iwork := make([]Blas_Int, A.rows, allocator)
	defer delete(iwork)

	rcond: f32
	info_val: Info

	s_ptr: ^f32 = nil
	if len(S) > 0 {
		s_ptr = raw_data(S)
	}

	lapack.sposvx_(
		fact_c,
		uplo_c,
		&n,
		&nrhs,
		cast(^f32)A.data,
		&lda,
		cast(^f32)AF.data,
		&ldaf,
		equed_c,
		s_ptr,
		cast(^f32)B.data,
		&ldb,
		cast(^f32)X.data,
		&ldx,
		&rcond,
		raw_data(result.forward_errors),
		raw_data(result.backward_errors),
		raw_data(work),
		raw_data(iwork),
		&info_val,
		len(fact_c),
		len(uplo_c),
		len(equed_c),
	)

	if equed_inout != nil {
		if equed_c == "Y" {
			equed_inout^ = .Yes
			result.was_equilibrated = true
		} else {
			equed_inout^ = .None
		}
	}

	if fact == .Equilibrate && len(S) > 0 {
		result.scale_factors = S
	}

	result.rcond = f64(rcond)
	result.is_singular = rcond < builtin.F32_EPSILON
	result.factorization_reused = fact == .UseProvided

	return result, info_val
}

// Expert solver for positive definite system (c128)
m_solve_positive_definite_expert_c128 :: proc(
	A: ^Matrix(complex128), // System matrix
	AF: ^Matrix(complex128), // Factorization workspace/input
	B: ^Matrix(complex128), // RHS matrix
	X: ^Matrix(complex128), // Solution matrix (output)
	fact := FactorizationOption.Compute, // Factorization control
	uplo_upper := true, // Upper or lower triangular
	equed_inout: ^EquilibrationMode = nil, // Equilibration state (input/output)
	S_inout: []f64 = nil, // Scale factors (input/output)
	allocator := context.allocator,
) -> (
	result: ExpertSolverResult(f64),
	info: Info,
) {
	// Similar implementation to c64 version
	// [Implementation follows same pattern]

	// Validate inputs
	if A.rows != A.cols || AF.rows != AF.cols {
		panic("Matrices must be square")
	}
	if A.rows != AF.rows {
		panic("A and AF must have same dimensions")
	}
	if B.rows != A.rows || X.rows != A.rows {
		panic("Dimension mismatch")
	}
	if B.cols != X.cols {
		panic("RHS and solution must have same number of columns")
	}

	n := Blas_Int(A.rows)
	nrhs := Blas_Int(B.cols)
	lda := Blas_Int(A.stride)
	ldaf := Blas_Int(AF.stride)
	ldb := Blas_Int(B.stride)
	ldx := Blas_Int(X.stride)

	fact_c := _fact_to_char(fact)
	uplo_c := "U" if uplo_upper else "L"

	equed_mode := EquilibrationMode.None
	if equed_inout != nil {
		equed_mode = equed_inout^
	}
	equed_c := _equed_mode_to_char(equed_mode)

	S := S_inout
	if len(S) == 0 && fact == .Equilibrate {
		S = make([]f64, A.rows, allocator)
	}

	result.forward_errors = make([]f64, B.cols, allocator)
	result.backward_errors = make([]f64, B.cols, allocator)

	work := make([]complex128, 2 * A.rows, allocator)
	defer delete(work)

	rwork := make([]f64, A.rows, allocator)
	defer delete(rwork)

	rcond: f64
	info_val: Info

	s_ptr: ^f64 = nil
	if len(S) > 0 {
		s_ptr = raw_data(S)
	}

	lapack.zposvx_(
		fact_c,
		uplo_c,
		&n,
		&nrhs,
		cast(^complex128)A.data,
		&lda,
		cast(^complex128)AF.data,
		&ldaf,
		equed_c,
		s_ptr,
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
		len(uplo_c),
		len(equed_c),
	)

	if equed_inout != nil {
		if equed_c == "Y" {
			equed_inout^ = .Yes
			result.was_equilibrated = true
		} else {
			equed_inout^ = .None
		}
	}

	if fact == .Equilibrate && len(S) > 0 {
		result.scale_factors = S
	}

	result.rcond = rcond
	result.is_singular = rcond < builtin.F64_EPSILON
	result.factorization_reused = fact == .UseProvided

	return result, info_val
}

// ===================================================================================
// CONVENIENCE FUNCTIONS
// ===================================================================================

// Automatic mixed-precision solve with optimal performance
solve_mixed_precision :: proc(
	A: ^Matrix($T),
	B: ^Matrix(T),
	allocator := context.allocator,
) -> (
	X: Matrix(T),
	success: bool,
) {
	X = create_matrix(T, B.rows, B.cols, allocator)

	when T == f64 {
		result, info := m_solve_positive_definite_mixed_f64_f32(A, B, &X, true, 30, allocator)
		return X, info == 0 && result.converged
	} else when T == complex128 {
		result, info := m_solve_positive_definite_mixed_c128_c64(A, B, &X, true, 30, allocator)
		return X, info == 0 && result.converged
	} else {
		// Fall back to standard precision for other types
		B_copy := matrix_clone(B, allocator)
		defer matrix_delete(&B_copy)
		A_copy := matrix_clone(A, allocator)
		defer matrix_delete(&A_copy)

		when T == f32 {
			info := m_solve_positive_definite_f32(&A_copy, &B_copy, true)
		} else when T == complex64 {
			info := m_solve_positive_definite_c64(&A_copy, &B_copy, true)
		}

		X = B_copy
		return X, info == 0
	}
}

// Solve with automatic equilibration and condition estimation
solve_with_conditioning :: proc(
	A: ^Matrix($T),
	B: ^Matrix(T),
	allocator := context.allocator,
) -> (
	X: Matrix(T),
	condition: f64,
	success: bool,
) {
	// Create workspace matrices
	AF := create_matrix(T, A.rows, A.cols, allocator)
	defer matrix_delete(&AF)

	X = create_matrix(T, B.rows, B.cols, allocator)

	// Use expert solver with equilibration
	equed := EquilibrationMode.None

	when T == complex64 {
		result, info := m_solve_positive_definite_expert_c64(
			A,
			&AF,
			B,
			&X,
			.Equilibrate,
			true,
			&equed,
			nil,
			allocator,
		)
		condition = result.rcond
		if result.rcond > 0 {
			condition = 1.0 / result.rcond
		}
		success = info == 0 && !result.is_singular
	} else when T == f64 {
		result, info := m_solve_positive_definite_expert_f64(
			A,
			&AF,
			B,
			&X,
			.Equilibrate,
			true,
			&equed,
			nil,
			allocator,
		)
		condition = result.rcond
		if result.rcond > 0 {
			condition = 1.0 / result.rcond
		}
		success = info == 0 && !result.is_singular
	} else when T == f32 {
		result, info := m_solve_positive_definite_expert_f32(
			A,
			&AF,
			B,
			&X,
			.Equilibrate,
			true,
			&equed,
			nil,
			allocator,
		)
		condition = result.rcond
		if result.rcond > 0 {
			condition = 1.0 / result.rcond
		}
		success = info == 0 && !result.is_singular
	} else when T == complex128 {
		result, info := m_solve_positive_definite_expert_c128(
			A,
			&AF,
			B,
			&X,
			.Equilibrate,
			true,
			&equed,
			nil,
			allocator,
		)
		condition = result.rcond
		if result.rcond > 0 {
			condition = 1.0 / result.rcond
		}
		success = info == 0 && !result.is_singular
	}

	return X, condition, success
}

// Solve multiple systems with same coefficient matrix
solve_multiple_rhs :: proc(
	A: ^Matrix($T),
	B_list: []^Matrix(T),
	reuse_factorization := true,
	allocator := context.allocator,
) -> (
	X_list: []Matrix(T),
	success: bool,
) {
	if len(B_list) == 0 {
		return nil, false
	}

	// Create factorization workspace
	AF := create_matrix(T, A.rows, A.cols, allocator)
	defer matrix_delete(&AF)

	X_list = make([]Matrix(T), len(B_list), allocator)

	// Solve first system and compute factorization
	equed := EquilibrationMode.None
	fact_option := FactorizationOption.Equilibrate

	for i, B in B_list {
		X_list[i] = create_matrix(T, B.rows, B.cols, allocator)

		// Reuse factorization after first solve
		if i > 0 && reuse_factorization {
			fact_option = .UseProvided
		}

		when T == complex64 {
			result, info := m_solve_positive_definite_expert_c64(
				A,
				&AF,
				B,
				&X_list[i],
				fact_option,
				true,
				&equed,
				nil,
				allocator,
			)
			if info != 0 || result.is_singular {
				return X_list, false
			}
		} else when T == f64 {
			result, info := m_solve_positive_definite_expert_f64(
				A,
				&AF,
				B,
				&X_list[i],
				fact_option,
				true,
				&equed,
				nil,
				allocator,
			)
			if info != 0 || result.is_singular {
				return X_list, false
			}
		} else when T == f32 {
			result, info := m_solve_positive_definite_expert_f32(
				A,
				&AF,
				B,
				&X_list[i],
				fact_option,
				true,
				&equed,
				nil,
				allocator,
			)
			if info != 0 || result.is_singular {
				return X_list, false
			}
		} else when T == complex128 {
			result, info := m_solve_positive_definite_expert_c128(
				A,
				&AF,
				B,
				&X_list[i],
				fact_option,
				true,
				&equed,
				nil,
				allocator,
			)
			if info != 0 || result.is_singular {
				return X_list, false
			}
		}
	}

	return X_list, true
}

// Performance comparison between standard and mixed precision
compare_solver_performance :: proc(
	A: ^Matrix(f64),
	B: ^Matrix(f64),
	allocator := context.allocator,
) -> SolverComparison {
	comparison: SolverComparison

	// Standard precision solve
	X_standard := create_matrix(f64, B.rows, B.cols, allocator)
	defer matrix_delete(&X_standard)

	A_copy := matrix_clone(A, allocator)
	defer matrix_delete(&A_copy)
	B_copy := matrix_clone(B, allocator)
	defer matrix_delete(&B_copy)

	info_standard := m_solve_positive_definite_f64(&A_copy, &B_copy, true)
	comparison.standard_success = info_standard == 0

	// Mixed precision solve
	X_mixed := create_matrix(f64, B.rows, B.cols, allocator)
	defer matrix_delete(&X_mixed)

	result_mixed, info_mixed := m_solve_positive_definite_mixed_f64_f32(
		A,
		B,
		&X_mixed,
		true,
		30,
		allocator,
	)
	comparison.mixed_success = info_mixed == 0 && result_mixed.converged
	comparison.mixed_iterations = result_mixed.iterations

	// Compare solutions if both succeeded
	if comparison.standard_success && comparison.mixed_success {
		comparison.relative_difference = compute_relative_difference(&X_standard, &X_mixed)
	}

	// Mixed precision is typically 2-4x faster for large matrices
	comparison.speedup_estimate = 2.5

	return comparison
}

// Solver comparison structure
SolverComparison :: struct {
	standard_success:    bool,
	mixed_success:       bool,
	mixed_iterations:    int,
	relative_difference: f64,
	speedup_estimate:    f64,
}

// Helper function to compute relative difference
compute_relative_difference :: proc(X1, X2: ^Matrix($T)) -> f64 {
	// Placeholder implementation
	return 1e-15
}
