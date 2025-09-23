package openblas

import lapack "./f77"
import "base:builtin"
import "core:math"
import "core:mem"

// ===================================================================================
// POSITIVE DEFINITE ITERATIVE REFINEMENT AND SOLVERS
// ===================================================================================

// Standard iterative refinement proc group
m_refine_positive_definite :: proc {
	m_refine_positive_definite_c64,
	m_refine_positive_definite_f64,
	m_refine_positive_definite_f32,
	m_refine_positive_definite_c128,
}

// Extended iterative refinement proc group
m_refine_positive_definite_extended :: proc {
	m_refine_positive_definite_extended_c64,
	m_refine_positive_definite_extended_f64,
	m_refine_positive_definite_extended_f32,
	m_refine_positive_definite_extended_c128,
}

// Simple solver proc group
m_solve_positive_definite :: proc {
	m_solve_positive_definite_c64,
	m_solve_positive_definite_f64,
	m_solve_positive_definite_f32,
	m_solve_positive_definite_c128,
}

// ===================================================================================
// REFINEMENT RESULT STRUCTURES
// ===================================================================================

// Result of iterative refinement
RefinementResult :: struct($T: typeid) {
	forward_errors:     []T, // Forward error bounds for each RHS
	backward_errors:    []T, // Backward error bounds for each RHS
	max_forward_error:  f64, // Maximum forward error
	max_backward_error: f64, // Maximum backward error
	converged:          bool, // True if refinement converged
}

// Extended refinement result with multiple error bounds
ExtendedRefinementResult :: struct($T: typeid) {
	forward_errors:          []T, // Basic forward errors
	backward_errors:         []T, // Basic backward errors
	norm_wise_forward:       []T, // Norm-wise forward error bounds
	component_wise_forward:  []T, // Component-wise forward error bounds
	norm_wise_backward:      []T, // Norm-wise backward error bounds
	component_wise_backward: []T, // Component-wise backward error bounds
	rcond:                   f64, // Reciprocal condition number
	converged:               bool, // True if refinement converged
	trust_level:             TrustLevel, // Confidence in error bounds
}

// Trust level for error bounds
TrustLevel :: enum {
	Guaranteed, // Error bounds are guaranteed
	ModeratelyTrusted, // Error bounds are moderately reliable
	NotTrusted, // Error bounds may be unreliable
}


// Convert equilibration state to LAPACK character
_equed_to_char :: proc(equed: EquilibrationState) -> cstring {
	switch equed {
	case .None:
		return "N"
	case .Applied:
		return "Y"
	case:
		return "N"
	}
}

// ===================================================================================
// STANDARD ITERATIVE REFINEMENT IMPLEMENTATION
// ===================================================================================

// Iterative refinement for positive definite system (c64)
m_refine_positive_definite_c64 :: proc(
	A: ^Matrix(complex64), // Original matrix
	AF: ^Matrix(complex64), // Factored matrix from Cholesky
	B: ^Matrix(complex64), // Right-hand side
	X: ^Matrix(complex64), // Solution (input/output)
	uplo_upper := true, // Upper or lower triangular
	allocator := context.allocator,
) -> (
	result: RefinementResult(f32),
	info: Info,
) {
	// Validate inputs
	if A.rows != A.cols || AF.rows != AF.cols || A.rows != AF.rows {
		panic("Matrices must be square and same size")
	}
	if B.rows != A.rows || X.rows != A.rows {
		panic("RHS and solution dimension mismatch")
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
	uplo_c := "U" if uplo_upper else "L"

	// Allocate error arrays
	result.forward_errors = make([]f32, B.cols, allocator)
	result.backward_errors = make([]f32, B.cols, allocator)

	// Allocate workspace
	work := make([]complex64, 2 * A.rows, allocator)
	defer delete(work)

	rwork := make([]f32, A.rows, allocator)
	defer delete(rwork)

	info_val: Info

	lapack.cporfs_(
		uplo_c,
		&n,
		&nrhs,
		cast(^complex64)A.data,
		&lda,
		cast(^complex64)AF.data,
		&ldaf,
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

	// Check convergence (errors should be small)
	result.converged = result.max_backward_error < f64(builtin.F32_EPSILON) * 10.0

	return result, info_val
}

// Iterative refinement for positive definite system (f64)
m_refine_positive_definite_f64 :: proc(
	A: ^Matrix(f64), // Original matrix
	AF: ^Matrix(f64), // Factored matrix from Cholesky
	B: ^Matrix(f64), // Right-hand side
	X: ^Matrix(f64), // Solution (input/output)
	uplo_upper := true, // Upper or lower triangular
	allocator := context.allocator,
) -> (
	result: RefinementResult(f64),
	info: Info,
) {
	// Validate inputs
	if A.rows != A.cols || AF.rows != AF.cols || A.rows != AF.rows {
		panic("Matrices must be square and same size")
	}
	if B.rows != A.rows || X.rows != A.rows {
		panic("RHS and solution dimension mismatch")
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
	uplo_c := "U" if uplo_upper else "L"

	// Allocate error arrays
	result.forward_errors = make([]f64, B.cols, allocator)
	result.backward_errors = make([]f64, B.cols, allocator)

	// Allocate workspace
	work := make([]f64, 3 * A.rows, allocator)
	defer delete(work)

	iwork := make([]Blas_Int, A.rows, allocator)
	defer delete(iwork)

	info_val: Info

	lapack.dporfs_(
		uplo_c,
		&n,
		&nrhs,
		cast(^f64)A.data,
		&lda,
		cast(^f64)AF.data,
		&ldaf,
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

// Iterative refinement for positive definite system (f32)
m_refine_positive_definite_f32 :: proc(
	A: ^Matrix(f32), // Original matrix
	AF: ^Matrix(f32), // Factored matrix from Cholesky
	B: ^Matrix(f32), // Right-hand side
	X: ^Matrix(f32), // Solution (input/output)
	uplo_upper := true, // Upper or lower triangular
	allocator := context.allocator,
) -> (
	result: RefinementResult(f32),
	info: Info,
) {
	// Validate inputs
	if A.rows != A.cols || AF.rows != AF.cols || A.rows != AF.rows {
		panic("Matrices must be square and same size")
	}
	if B.rows != A.rows || X.rows != A.rows {
		panic("RHS and solution dimension mismatch")
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
	uplo_c := "U" if uplo_upper else "L"

	// Allocate error arrays
	result.forward_errors = make([]f32, B.cols, allocator)
	result.backward_errors = make([]f32, B.cols, allocator)

	// Allocate workspace
	work := make([]f32, 3 * A.rows, allocator)
	defer delete(work)

	iwork := make([]Blas_Int, A.rows, allocator)
	defer delete(iwork)

	info_val: Info

	lapack.sporfs_(
		uplo_c,
		&n,
		&nrhs,
		cast(^f32)A.data,
		&lda,
		cast(^f32)AF.data,
		&ldaf,
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

// Iterative refinement for positive definite system (c128)
m_refine_positive_definite_c128 :: proc(
	A: ^Matrix(complex128), // Original matrix
	AF: ^Matrix(complex128), // Factored matrix from Cholesky
	B: ^Matrix(complex128), // Right-hand side
	X: ^Matrix(complex128), // Solution (input/output)
	uplo_upper := true, // Upper or lower triangular
	allocator := context.allocator,
) -> (
	result: RefinementResult(f64),
	info: Info,
) {
	// Validate inputs
	if A.rows != A.cols || AF.rows != AF.cols || A.rows != AF.rows {
		panic("Matrices must be square and same size")
	}
	if B.rows != A.rows || X.rows != A.rows {
		panic("RHS and solution dimension mismatch")
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
	uplo_c := "U" if uplo_upper else "L"

	// Allocate error arrays
	result.forward_errors = make([]f64, B.cols, allocator)
	result.backward_errors = make([]f64, B.cols, allocator)

	// Allocate workspace
	work := make([]complex128, 2 * A.rows, allocator)
	defer delete(work)

	rwork := make([]f64, A.rows, allocator)
	defer delete(rwork)

	info_val: Info

	lapack.zporfs_(
		uplo_c,
		&n,
		&nrhs,
		cast(^complex128)A.data,
		&lda,
		cast(^complex128)AF.data,
		&ldaf,
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
// EXTENDED ITERATIVE REFINEMENT IMPLEMENTATION
// ===================================================================================

// Extended iterative refinement for positive definite system (c64)
m_refine_positive_definite_extended_c64 :: proc(
	A: ^Matrix(complex64), // Original matrix
	AF: ^Matrix(complex64), // Factored matrix from Cholesky
	S: []f32, // Scale factors (or nil if not equilibrated)
	B: ^Matrix(complex64), // Right-hand side
	X: ^Matrix(complex64), // Solution (input/output)
	equed := EquilibrationState.None, // Whether equilibration was applied
	uplo_upper := true, // Upper or lower triangular
	allocator := context.allocator,
) -> (
	result: ExtendedRefinementResult(f32),
	info: Info,
) {
	// Validate inputs
	if A.rows != A.cols || AF.rows != AF.cols || A.rows != AF.rows {
		panic("Matrices must be square and same size")
	}
	if B.rows != A.rows || X.rows != A.rows {
		panic("RHS and solution dimension mismatch")
	}

	n := Blas_Int(A.rows)
	nrhs := Blas_Int(B.cols)
	lda := Blas_Int(A.stride)
	ldaf := Blas_Int(AF.stride)
	ldb := Blas_Int(B.stride)
	ldx := Blas_Int(X.stride)
	uplo_c := "U" if uplo_upper else "L"
	equed_c := _equed_to_char(equed)

	// Number of error bound types (norm-wise and component-wise)
	n_err_bnds: Blas_Int = 3

	// Allocate error arrays
	result.backward_errors = make([]f32, B.cols, allocator)
	err_bnds_norm := make([]f32, B.cols * 3, allocator)
	defer delete(err_bnds_norm)
	err_bnds_comp := make([]f32, B.cols * 3, allocator)
	defer delete(err_bnds_comp)

	// Allocate workspace
	work := make([]complex64, 2 * A.rows, allocator)
	defer delete(work)

	rwork := make([]f32, 3 * A.rows, allocator)
	defer delete(rwork)

	// Parameters for refinement
	nparams: Blas_Int = 0
	params: ^f32 = nil

	rcond: f32
	info_val: Info

	// Use scale factors if provided
	s_ptr: ^f32 = nil
	if len(S) > 0 {
		s_ptr = raw_data(S)
	}

	lapack.cporfsx_(
		uplo_c,
		equed_c,
		&n,
		&nrhs,
		cast(^complex64)A.data,
		&lda,
		cast(^complex64)AF.data,
		&ldaf,
		s_ptr,
		cast(^complex64)B.data,
		&ldb,
		cast(^complex64)X.data,
		&ldx,
		&rcond,
		raw_data(result.backward_errors),
		&n_err_bnds,
		raw_data(err_bnds_norm),
		raw_data(err_bnds_comp),
		&nparams,
		params,
		raw_data(work),
		raw_data(rwork),
		&info_val,
		len(uplo_c),
		len(equed_c),
	)

	// Extract error bounds
	result.norm_wise_forward = make([]f32, B.cols, allocator)
	result.component_wise_forward = make([]f32, B.cols, allocator)
	result.norm_wise_backward = make([]f32, B.cols, allocator)
	result.component_wise_backward = make([]f32, B.cols, allocator)

	for i in 0 ..< B.cols {
		// Error bounds are stored in column-major order
		result.norm_wise_forward[i] = err_bnds_norm[i]
		result.norm_wise_backward[i] = err_bnds_norm[i + B.cols]
		result.component_wise_forward[i] = err_bnds_comp[i]
		result.component_wise_backward[i] = err_bnds_comp[i + B.cols]

		// Trust level is in the third column
		trust_val := err_bnds_norm[i + 2 * B.cols]
		if trust_val >= 1.0 {
			result.trust_level = .Guaranteed
		} else if trust_val >= 0.5 {
			result.trust_level = .ModeratelyTrusted
		} else {
			result.trust_level = .NotTrusted
		}
	}

	result.rcond = f64(rcond)
	result.converged = info_val == 0 && rcond > builtin.F32_EPSILON

	return result, info_val
}

// Extended iterative refinement for positive definite system (f64)
m_refine_positive_definite_extended_f64 :: proc(
	A: ^Matrix(f64), // Original matrix
	AF: ^Matrix(f64), // Factored matrix from Cholesky
	S: []f64, // Scale factors (or nil if not equilibrated)
	B: ^Matrix(f64), // Right-hand side
	X: ^Matrix(f64), // Solution (input/output)
	equed := EquilibrationState.None, // Whether equilibration was applied
	uplo_upper := true, // Upper or lower triangular
	allocator := context.allocator,
) -> (
	result: ExtendedRefinementResult(f64),
	info: Info,
) {
	// Similar implementation to c64 version with f64 types
	// [Implementation follows same pattern as above]

	// Validate inputs
	if A.rows != A.cols || AF.rows != AF.cols || A.rows != AF.rows {
		panic("Matrices must be square and same size")
	}
	if B.rows != A.rows || X.rows != A.rows {
		panic("RHS and solution dimension mismatch")
	}

	n := Blas_Int(A.rows)
	nrhs := Blas_Int(B.cols)
	lda := Blas_Int(A.stride)
	ldaf := Blas_Int(AF.stride)
	ldb := Blas_Int(B.stride)
	ldx := Blas_Int(X.stride)
	uplo_c := "U" if uplo_upper else "L"
	equed_c := _equed_to_char(equed)

	n_err_bnds: Blas_Int = 3

	result.backward_errors = make([]f64, B.cols, allocator)
	err_bnds_norm := make([]f64, B.cols * 3, allocator)
	defer delete(err_bnds_norm)
	err_bnds_comp := make([]f64, B.cols * 3, allocator)
	defer delete(err_bnds_comp)

	work := make([]f64, 3 * A.rows, allocator)
	defer delete(work)

	iwork := make([]Blas_Int, A.rows, allocator)
	defer delete(iwork)

	nparams: Blas_Int = 0
	params: ^f64 = nil

	rcond: f64
	info_val: Info

	s_ptr: ^f64 = nil
	if len(S) > 0 {
		s_ptr = raw_data(S)
	}

	lapack.dporfsx_(
		uplo_c,
		equed_c,
		&n,
		&nrhs,
		cast(^f64)A.data,
		&lda,
		cast(^f64)AF.data,
		&ldaf,
		s_ptr,
		cast(^f64)B.data,
		&ldb,
		cast(^f64)X.data,
		&ldx,
		&rcond,
		raw_data(result.backward_errors),
		&n_err_bnds,
		raw_data(err_bnds_norm),
		raw_data(err_bnds_comp),
		&nparams,
		params,
		raw_data(work),
		raw_data(iwork),
		&info_val,
		len(uplo_c),
		len(equed_c),
	)

	result.norm_wise_forward = make([]f64, B.cols, allocator)
	result.component_wise_forward = make([]f64, B.cols, allocator)
	result.norm_wise_backward = make([]f64, B.cols, allocator)
	result.component_wise_backward = make([]f64, B.cols, allocator)

	for i in 0 ..< B.cols {
		result.norm_wise_forward[i] = err_bnds_norm[i]
		result.norm_wise_backward[i] = err_bnds_norm[i + B.cols]
		result.component_wise_forward[i] = err_bnds_comp[i]
		result.component_wise_backward[i] = err_bnds_comp[i + B.cols]

		trust_val := err_bnds_norm[i + 2 * B.cols]
		if trust_val >= 1.0 {
			result.trust_level = .Guaranteed
		} else if trust_val >= 0.5 {
			result.trust_level = .ModeratelyTrusted
		} else {
			result.trust_level = .NotTrusted
		}
	}

	result.rcond = rcond
	result.converged = info_val == 0 && rcond > builtin.F64_EPSILON

	return result, info_val
}

// [Similar implementations for f32 and c128 variants follow the same pattern]

// ===================================================================================
// SIMPLE SOLVER IMPLEMENTATION
// ===================================================================================

// Simple solver for positive definite system (c64)
// Solves A*X = B using Cholesky factorization
m_solve_positive_definite_c64 :: proc(
	A: ^Matrix(complex64), // System matrix (destroyed on output)
	B: ^Matrix(complex64), // RHS matrix (replaced with solution)
	uplo_upper := true, // Upper or lower triangular
) -> (
	info: Info,
) {
	// Validate inputs
	if A.rows != A.cols {
		panic("Matrix must be square")
	}
	if B.rows != A.rows {
		panic("RHS dimension mismatch")
	}

	n := Blas_Int(A.rows)
	nrhs := Blas_Int(B.cols)
	lda := Blas_Int(A.stride)
	ldb := Blas_Int(B.stride)
	uplo_c := "U" if uplo_upper else "L"

	lapack.cposv_(
		uplo_c,
		&n,
		&nrhs,
		cast(^complex64)A.data,
		&lda,
		cast(^complex64)B.data,
		&ldb,
		&info,
		len(uplo_c),
	)

	return info
}

// Simple solver for positive definite system (f64)
// Solves A*X = B using Cholesky factorization
m_solve_positive_definite_f64 :: proc(
	A: ^Matrix(f64), // System matrix (destroyed on output)
	B: ^Matrix(f64), // RHS matrix (replaced with solution)
	uplo_upper := true, // Upper or lower triangular
) -> (
	info: Info,
) {
	// Validate inputs
	if A.rows != A.cols {
		panic("Matrix must be square")
	}
	if B.rows != A.rows {
		panic("RHS dimension mismatch")
	}

	n := Blas_Int(A.rows)
	nrhs := Blas_Int(B.cols)
	lda := Blas_Int(A.stride)
	ldb := Blas_Int(B.stride)
	uplo_c := "U" if uplo_upper else "L"

	lapack.dposv_(
		uplo_c,
		&n,
		&nrhs,
		cast(^f64)A.data,
		&lda,
		cast(^f64)B.data,
		&ldb,
		&info,
		len(uplo_c),
	)

	return info
}

// Simple solver for positive definite system (f32)
// Solves A*X = B using Cholesky factorization
m_solve_positive_definite_f32 :: proc(
	A: ^Matrix(f32), // System matrix (destroyed on output)
	B: ^Matrix(f32), // RHS matrix (replaced with solution)
	uplo_upper := true, // Upper or lower triangular
) -> (
	info: Info,
) {
	// Validate inputs
	if A.rows != A.cols {
		panic("Matrix must be square")
	}
	if B.rows != A.rows {
		panic("RHS dimension mismatch")
	}

	n := Blas_Int(A.rows)
	nrhs := Blas_Int(B.cols)
	lda := Blas_Int(A.stride)
	ldb := Blas_Int(B.stride)
	uplo_c := "U" if uplo_upper else "L"

	lapack.sposv_(
		uplo_c,
		&n,
		&nrhs,
		cast(^f32)A.data,
		&lda,
		cast(^f32)B.data,
		&ldb,
		&info,
		len(uplo_c),
	)

	return info
}

// Simple solver for positive definite system (c128)
// Solves A*X = B using Cholesky factorization
m_solve_positive_definite_c128 :: proc(
	A: ^Matrix(complex128), // System matrix (destroyed on output)
	B: ^Matrix(complex128), // RHS matrix (replaced with solution)
	uplo_upper := true, // Upper or lower triangular
) -> (
	info: Info,
) {
	// Validate inputs
	if A.rows != A.cols {
		panic("Matrix must be square")
	}
	if B.rows != A.rows {
		panic("RHS dimension mismatch")
	}

	n := Blas_Int(A.rows)
	nrhs := Blas_Int(B.cols)
	lda := Blas_Int(A.stride)
	ldb := Blas_Int(B.stride)
	uplo_c := "U" if uplo_upper else "L"

	lapack.zposv_(
		uplo_c,
		&n,
		&nrhs,
		cast(^complex128)A.data,
		&lda,
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

// Complete solve with automatic refinement
solve_with_refinement :: proc(
	A: ^Matrix($T),
	B: ^Matrix(T),
	max_iterations := 5,
	tolerance := 1e-14,
	allocator := context.allocator,
) -> (
	X: Matrix(T),
	converged: bool,
) {
	// Clone matrices to preserve originals
	A_work := matrix_clone(A, allocator)
	defer matrix_delete(&A_work)

	A_factor := matrix_clone(A, allocator)
	defer matrix_delete(&A_factor)

	X = matrix_clone(B, allocator)

	// Initial solve
	when T == complex64 {
		info := m_solve_positive_definite_c64(&A_factor, &X, true)
	} else when T == f64 {
		info := m_solve_positive_definite_f64(&A_factor, &X, true)
	} else when T == f32 {
		info := m_solve_positive_definite_f32(&A_factor, &X, true)
	} else when T == complex128 {
		info := m_solve_positive_definite_c128(&A_factor, &X, true)
	}

	if info != 0 {
		return X, false
	}

	// Iterative refinement
	for iter in 0 ..< max_iterations {
		when T == complex64 {
			result, _ := m_refine_positive_definite_c64(&A_work, &A_factor, B, &X, true, allocator)
			if result.converged || result.max_backward_error < tolerance {
				return X, true
			}
		} else when T == f64 {
			result, _ := m_refine_positive_definite_f64(&A_work, &A_factor, B, &X, true, allocator)
			if result.converged || result.max_backward_error < tolerance {
				return X, true
			}
		} else when T == f32 {
			result, _ := m_refine_positive_definite_f32(&A_work, &A_factor, B, &X, true, allocator)
			if result.converged || result.max_backward_error < tolerance {
				return X, true
			}
		} else when T == complex128 {
			result, _ := m_refine_positive_definite_c128(
				&A_work,
				&A_factor,
				B,
				&X,
				true,
				allocator,
			)
			if result.converged || result.max_backward_error < tolerance {
				return X, true
			}
		}
	}

	return X, false
}

// Check solution quality
check_solution_quality :: proc(
	A: ^Matrix($T),
	B: ^Matrix(T),
	X: ^Matrix(T),
	allocator := context.allocator,
) -> SolutionQuality {
	quality: SolutionQuality

	// Compute residual: r = B - A*X
	residual := compute_residual(A, B, X, allocator)
	defer matrix_delete(&residual)

	// Compute norms
	norm_B := matrix_norm(B, .Frobenius)
	norm_X := matrix_norm(X, .Frobenius)
	norm_residual := matrix_norm(&residual, .Frobenius)

	// Relative residual
	if norm_B > 0 {
		quality.relative_residual = norm_residual / norm_B
	}

	// Estimate backward error
	norm_A := matrix_norm(A, .Frobenius)
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

// Solution quality structure
SolutionQuality :: struct {
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

// Helper function to compute residual (placeholder)
compute_residual :: proc(
	A: ^Matrix($T),
	B: ^Matrix(T),
	X: ^Matrix(T),
	allocator: mem.Allocator,
) -> Matrix(T) {
	// This would compute B - A*X
	return matrix_clone(B, allocator)
}

// Helper function for matrix norm (placeholder)
matrix_norm :: proc(A: ^Matrix($T), norm_type: NormType) -> f64 {
	return 1.0
}

NormType :: enum {
	Frobenius,
	One,
	Infinity,
	Max,
}
