package openblas

import lapack "./f77"
import "base:builtin"
import "core:math"
import "core:mem"

// ===================================================================================
// POSITIVE DEFINITE EXTENDED EXPERT SOLVERS
// ===================================================================================

// Extended expert solver proc group
m_solve_positive_definite_extended_expert :: proc {
	m_solve_positive_definite_extended_expert_c64,
	m_solve_positive_definite_extended_expert_f64,
	m_solve_positive_definite_extended_expert_f32,
	m_solve_positive_definite_extended_expert_c128,
}

// ===================================================================================
// EXTENDED EXPERT RESULT STRUCTURE
// ===================================================================================

// Extended expert solver result with extra precision and growth factor
ExtendedExpertResult :: struct($T: typeid) {
	rcond:                       f64, // Reciprocal condition number
	rpvgrw:                      f64, // Reciprocal pivot growth factor
	backward_errors:             []T, // Backward error bounds
	norm_wise_forward:           []T, // Norm-wise forward error bounds
	component_wise_forward:      []T, // Component-wise forward error bounds
	norm_wise_backward:          []T, // Norm-wise backward error bounds
	component_wise_backward:     []T, // Component-wise backward error bounds
	trust_bounds:                []T, // Trust bounds for error estimates
	scale_factors:               []T, // Equilibration scale factors
	was_equilibrated:            bool, // True if equilibration was applied
	is_singular:                 bool, // True if matrix is singular
	growth_factor_ok:            bool, // True if pivot growth is acceptable
	condition_estimate_reliable: bool, // True if condition estimate is reliable
}

// Refinement parameters structure
RefinementParams :: struct {
	enable_refinement:     bool, // Enable iterative refinement
	max_iterations:        int, // Maximum refinement iterations
	convergence_threshold: f64, // Convergence threshold
	component_wise_error:  bool, // Use component-wise error criterion
}

// ===================================================================================
// EXTENDED EXPERT SOLVER IMPLEMENTATION
// ===================================================================================

// Extended expert solver for positive definite system (c64)
// Provides extra precision error bounds and pivot growth factor
m_solve_positive_definite_extended_expert_c64 :: proc(
	A: ^Matrix(complex64), // System matrix
	AF: ^Matrix(complex64), // Factorization workspace/input
	B: ^Matrix(complex64), // RHS matrix
	X: ^Matrix(complex64), // Solution matrix (output)
	fact := FactorizationOption.Compute, // Factorization control
	uplo_upper := true, // Upper or lower triangular
	equed_inout: ^EquilibrationMode = nil, // Equilibration state (input/output)
	S_inout: []f32 = nil, // Scale factors (input/output)
	params: ^RefinementParams = nil, // Refinement parameters
	allocator := context.allocator,
) -> (
	result: ExtendedExpertResult(f32),
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

	// Number of error bound types (3 for extended version)
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

	// Set refinement parameters
	nparams_val: Blas_Int = 0
	params_array: [3]f32
	params_ptr: ^f32 = nil

	if params != nil {
		nparams_val = 3
		params_array[0] = 1.0 if params.enable_refinement else 0.0 // LA_LINRX_ITREF_ON
		params_array[1] = f32(params.max_iterations) // LA_LINRX_ITHRESH
		params_array[2] = 1.0 if params.component_wise_error else 0.0 // LA_LINRX_CWISE
		params_ptr = &params_array[0]
	}

	rcond: f32
	rpvgrw: f32
	info_val: Info

	// Call LAPACK
	s_ptr: ^f32 = nil
	if len(S) > 0 {
		s_ptr = raw_data(S)
	}

	lapack.cposvxx_(
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
		&rpvgrw,
		raw_data(result.backward_errors),
		&n_err_bnds,
		raw_data(err_bnds_norm),
		raw_data(err_bnds_comp),
		&nparams_val,
		params_ptr,
		raw_data(work),
		raw_data(rwork),
		&info_val,
		len(fact_c),
		len(uplo_c),
		len(equed_c),
	)

	// Update equilibration state if provided
	if equed_inout != nil {
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

	// Extract error bounds
	result.norm_wise_forward = make([]f32, B.cols, allocator)
	result.component_wise_forward = make([]f32, B.cols, allocator)
	result.norm_wise_backward = make([]f32, B.cols, allocator)
	result.component_wise_backward = make([]f32, B.cols, allocator)
	result.trust_bounds = make([]f32, B.cols, allocator)

	for i in 0 ..< B.cols {
		// Error bounds are stored in column-major order
		result.norm_wise_forward[i] = err_bnds_norm[i]
		result.norm_wise_backward[i] = err_bnds_norm[i + B.cols]
		result.trust_bounds[i] = err_bnds_norm[i + 2 * B.cols]

		result.component_wise_forward[i] = err_bnds_comp[i]
		result.component_wise_backward[i] = err_bnds_comp[i + B.cols]
		// Component trust bounds in third column
	}

	// Fill remaining result fields
	result.rcond = f64(rcond)
	result.rpvgrw = f64(rpvgrw)
	result.is_singular = rcond < builtin.F32_EPSILON
	result.growth_factor_ok = rpvgrw >= 0.1 // Typical threshold
	result.condition_estimate_reliable = rcond > builtin.F32_EPSILON * 10.0

	return result, info_val
}

// Extended expert solver for positive definite system (f64)
m_solve_positive_definite_extended_expert_f64 :: proc(
	A: ^Matrix(f64), // System matrix
	AF: ^Matrix(f64), // Factorization workspace/input
	B: ^Matrix(f64), // RHS matrix
	X: ^Matrix(f64), // Solution matrix (output)
	fact := FactorizationOption.Compute, // Factorization control
	uplo_upper := true, // Upper or lower triangular
	equed_inout: ^EquilibrationMode = nil, // Equilibration state (input/output)
	S_inout: []f64 = nil, // Scale factors (input/output)
	params: ^RefinementParams = nil, // Refinement parameters
	allocator := context.allocator,
) -> (
	result: ExtendedExpertResult(f64),
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

	nparams_val: Blas_Int = 0
	params_array: [3]f64
	params_ptr: ^f64 = nil

	if params != nil {
		nparams_val = 3
		params_array[0] = 1.0 if params.enable_refinement else 0.0
		params_array[1] = f64(params.max_iterations)
		params_array[2] = 1.0 if params.component_wise_error else 0.0
		params_ptr = &params_array[0]
	}

	rcond: f64
	rpvgrw: f64
	info_val: Info

	s_ptr: ^f64 = nil
	if len(S) > 0 {
		s_ptr = raw_data(S)
	}

	lapack.dposvxx_(
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
		&rpvgrw,
		raw_data(result.backward_errors),
		&n_err_bnds,
		raw_data(err_bnds_norm),
		raw_data(err_bnds_comp),
		&nparams_val,
		params_ptr,
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

	result.norm_wise_forward = make([]f64, B.cols, allocator)
	result.component_wise_forward = make([]f64, B.cols, allocator)
	result.norm_wise_backward = make([]f64, B.cols, allocator)
	result.component_wise_backward = make([]f64, B.cols, allocator)
	result.trust_bounds = make([]f64, B.cols, allocator)

	for i in 0 ..< B.cols {
		result.norm_wise_forward[i] = err_bnds_norm[i]
		result.norm_wise_backward[i] = err_bnds_norm[i + B.cols]
		result.trust_bounds[i] = err_bnds_norm[i + 2 * B.cols]

		result.component_wise_forward[i] = err_bnds_comp[i]
		result.component_wise_backward[i] = err_bnds_comp[i + B.cols]
	}

	result.rcond = rcond
	result.rpvgrw = rpvgrw
	result.is_singular = rcond < builtin.F64_EPSILON
	result.growth_factor_ok = rpvgrw >= 0.1
	result.condition_estimate_reliable = rcond > builtin.F64_EPSILON * 10.0

	return result, info_val
}

// Extended expert solver for positive definite system (f32)
m_solve_positive_definite_extended_expert_f32 :: proc(
	A: ^Matrix(f32), // System matrix
	AF: ^Matrix(f32), // Factorization workspace/input
	B: ^Matrix(f32), // RHS matrix
	X: ^Matrix(f32), // Solution matrix (output)
	fact := FactorizationOption.Compute, // Factorization control
	uplo_upper := true, // Upper or lower triangular
	equed_inout: ^EquilibrationMode = nil, // Equilibration state (input/output)
	S_inout: []f32 = nil, // Scale factors (input/output)
	params: ^RefinementParams = nil, // Refinement parameters
	allocator := context.allocator,
) -> (
	result: ExtendedExpertResult(f32),
	info: Info,
) {
	// Similar implementation to f64 version
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

	n_err_bnds: Blas_Int = 3

	result.backward_errors = make([]f32, B.cols, allocator)
	err_bnds_norm := make([]f32, B.cols * 3, allocator)
	defer delete(err_bnds_norm)
	err_bnds_comp := make([]f32, B.cols * 3, allocator)
	defer delete(err_bnds_comp)

	work := make([]f32, 3 * A.rows, allocator)
	defer delete(work)

	iwork := make([]Blas_Int, A.rows, allocator)
	defer delete(iwork)

	nparams_val: Blas_Int = 0
	params_array: [3]f32
	params_ptr: ^f32 = nil

	if params != nil {
		nparams_val = 3
		params_array[0] = 1.0 if params.enable_refinement else 0.0
		params_array[1] = f32(params.max_iterations)
		params_array[2] = 1.0 if params.component_wise_error else 0.0
		params_ptr = &params_array[0]
	}

	rcond: f32
	rpvgrw: f32
	info_val: Info

	s_ptr: ^f32 = nil
	if len(S) > 0 {
		s_ptr = raw_data(S)
	}

	lapack.sposvxx_(
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
		&rpvgrw,
		raw_data(result.backward_errors),
		&n_err_bnds,
		raw_data(err_bnds_norm),
		raw_data(err_bnds_comp),
		&nparams_val,
		params_ptr,
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

	result.norm_wise_forward = make([]f32, B.cols, allocator)
	result.component_wise_forward = make([]f32, B.cols, allocator)
	result.norm_wise_backward = make([]f32, B.cols, allocator)
	result.component_wise_backward = make([]f32, B.cols, allocator)
	result.trust_bounds = make([]f32, B.cols, allocator)

	for i in 0 ..< B.cols {
		result.norm_wise_forward[i] = err_bnds_norm[i]
		result.norm_wise_backward[i] = err_bnds_norm[i + B.cols]
		result.trust_bounds[i] = err_bnds_norm[i + 2 * B.cols]

		result.component_wise_forward[i] = err_bnds_comp[i]
		result.component_wise_backward[i] = err_bnds_comp[i + B.cols]
	}

	result.rcond = f64(rcond)
	result.rpvgrw = f64(rpvgrw)
	result.is_singular = rcond < builtin.F32_EPSILON
	result.growth_factor_ok = rpvgrw >= 0.1
	result.condition_estimate_reliable = rcond > builtin.F32_EPSILON * 10.0

	return result, info_val
}

// Extended expert solver for positive definite system (c128)
m_solve_positive_definite_extended_expert_c128 :: proc(
	A: ^Matrix(complex128), // System matrix
	AF: ^Matrix(complex128), // Factorization workspace/input
	B: ^Matrix(complex128), // RHS matrix
	X: ^Matrix(complex128), // Solution matrix (output)
	fact := FactorizationOption.Compute, // Factorization control
	uplo_upper := true, // Upper or lower triangular
	equed_inout: ^EquilibrationMode = nil, // Equilibration state (input/output)
	S_inout: []f64 = nil, // Scale factors (input/output)
	params: ^RefinementParams = nil, // Refinement parameters
	allocator := context.allocator,
) -> (
	result: ExtendedExpertResult(f64),
	info: Info,
) {
	// Similar implementation to c64 version
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

	n_err_bnds: Blas_Int = 3

	result.backward_errors = make([]f64, B.cols, allocator)
	err_bnds_norm := make([]f64, B.cols * 3, allocator)
	defer delete(err_bnds_norm)
	err_bnds_comp := make([]f64, B.cols * 3, allocator)
	defer delete(err_bnds_comp)

	work := make([]complex128, 2 * A.rows, allocator)
	defer delete(work)

	rwork := make([]f64, 3 * A.rows, allocator)
	defer delete(rwork)

	nparams_val: Blas_Int = 0
	params_array: [3]f64
	params_ptr: ^f64 = nil

	if params != nil {
		nparams_val = 3
		params_array[0] = 1.0 if params.enable_refinement else 0.0
		params_array[1] = f64(params.max_iterations)
		params_array[2] = 1.0 if params.component_wise_error else 0.0
		params_ptr = &params_array[0]
	}

	rcond: f64
	rpvgrw: f64
	info_val: Info

	s_ptr: ^f64 = nil
	if len(S) > 0 {
		s_ptr = raw_data(S)
	}

	lapack.zposvxx_(
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
		&rpvgrw,
		raw_data(result.backward_errors),
		&n_err_bnds,
		raw_data(err_bnds_norm),
		raw_data(err_bnds_comp),
		&nparams_val,
		params_ptr,
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

	result.norm_wise_forward = make([]f64, B.cols, allocator)
	result.component_wise_forward = make([]f64, B.cols, allocator)
	result.norm_wise_backward = make([]f64, B.cols, allocator)
	result.component_wise_backward = make([]f64, B.cols, allocator)
	result.trust_bounds = make([]f64, B.cols, allocator)

	for i in 0 ..< B.cols {
		result.norm_wise_forward[i] = err_bnds_norm[i]
		result.norm_wise_backward[i] = err_bnds_norm[i + B.cols]
		result.trust_bounds[i] = err_bnds_norm[i + 2 * B.cols]

		result.component_wise_forward[i] = err_bnds_comp[i]
		result.component_wise_backward[i] = err_bnds_comp[i + B.cols]
	}

	result.rcond = rcond
	result.rpvgrw = rpvgrw
	result.is_singular = rcond < builtin.F64_EPSILON
	result.growth_factor_ok = rpvgrw >= 0.1
	result.condition_estimate_reliable = rcond > builtin.F64_EPSILON * 10.0

	return result, info_val
}

// ===================================================================================
// CONVENIENCE FUNCTIONS
// ===================================================================================

// Solve with maximum accuracy and comprehensive error analysis
solve_with_maximum_accuracy :: proc(
	A: ^Matrix($T),
	B: ^Matrix(T),
	allocator := context.allocator,
) -> (
	X: Matrix(T),
	analysis: SolutionAnalysis,
	success: bool,
) {
	// Create workspace matrices
	AF := create_matrix(T, A.rows, A.cols, allocator)
	defer matrix_delete(&AF)

	X = create_matrix(T, B.rows, B.cols, allocator)

	// Set up refinement parameters for maximum accuracy
	params := RefinementParams {
		enable_refinement     = true,
		max_iterations        = 100,
		convergence_threshold = builtin.F64_EPSILON,
		component_wise_error  = true,
	}

	// Use extended expert solver with equilibration
	equed := EquilibrationMode.None

	when T == complex64 {
		result, info := m_solve_positive_definite_extended_expert_c64(
			A,
			&AF,
			B,
			&X,
			.Equilibrate,
			true,
			&equed,
			nil,
			&params,
			allocator,
		)
		analysis = build_solution_analysis(result, info)
		success = info == 0 && !result.is_singular
	} else when T == f64 {
		result, info := m_solve_positive_definite_extended_expert_f64(
			A,
			&AF,
			B,
			&X,
			.Equilibrate,
			true,
			&equed,
			nil,
			&params,
			allocator,
		)
		analysis = build_solution_analysis(result, info)
		success = info == 0 && !result.is_singular
	} else when T == f32 {
		result, info := m_solve_positive_definite_extended_expert_f32(
			A,
			&AF,
			B,
			&X,
			.Equilibrate,
			true,
			&equed,
			nil,
			&params,
			allocator,
		)
		analysis = build_solution_analysis(result, info)
		success = info == 0 && !result.is_singular
	} else when T == complex128 {
		result, info := m_solve_positive_definite_extended_expert_c128(
			A,
			&AF,
			B,
			&X,
			.Equilibrate,
			true,
			&equed,
			nil,
			&params,
			allocator,
		)
		analysis = build_solution_analysis(result, info)
		success = info == 0 && !result.is_singular
	}

	return X, analysis, success
}

// Comprehensive solution analysis
SolutionAnalysis :: struct {
	condition_number:                f64,
	pivot_growth:                    f64,
	max_backward_error:              f64,
	max_forward_error_normwise:      f64,
	max_forward_error_componentwise: f64,
	min_trust_bound:                 f64,
	quality_assessment:              QualityAssessment,
	recommendations:                 []string,
}

QualityAssessment :: enum {
	Excellent, // All metrics excellent
	Good, // Good accuracy, stable
	Acceptable, // Acceptable for most purposes
	Marginal, // Use with caution
	Poor, // Unreliable results
	Failed, // Solution failed
}

// Build solution analysis from extended expert result
build_solution_analysis :: proc(result: ExtendedExpertResult($T), info: Info) -> SolutionAnalysis {
	analysis: SolutionAnalysis

	// Basic metrics
	if result.rcond > 0 {
		analysis.condition_number = 1.0 / result.rcond
	} else {
		analysis.condition_number = math.INF_F64
	}

	if result.rpvgrw > 0 {
		analysis.pivot_growth = 1.0 / result.rpvgrw
	} else {
		analysis.pivot_growth = math.INF_F64
	}

	// Find maximum errors
	for i in 0 ..< len(result.backward_errors) {
		analysis.max_backward_error = max(
			analysis.max_backward_error,
			f64(result.backward_errors[i]),
		)
		analysis.max_forward_error_normwise = max(
			analysis.max_forward_error_normwise,
			f64(result.norm_wise_forward[i]),
		)
		analysis.max_forward_error_componentwise = max(
			analysis.max_forward_error_componentwise,
			f64(result.component_wise_forward[i]),
		)

		if i == 0 || f64(result.trust_bounds[i]) < analysis.min_trust_bound {
			analysis.min_trust_bound = f64(result.trust_bounds[i])
		}
	}

	// Quality assessment
	if info != 0 {
		analysis.quality_assessment = .Failed
	} else if result.is_singular {
		analysis.quality_assessment = .Poor
	} else if analysis.condition_number > 1e15 {
		analysis.quality_assessment = .Marginal
	} else if analysis.condition_number > 1e10 || !result.growth_factor_ok {
		analysis.quality_assessment = .Acceptable
	} else if analysis.condition_number > 1e6 {
		analysis.quality_assessment = .Good
	} else {
		analysis.quality_assessment = .Excellent
	}

	// Build recommendations
	analysis.recommendations = make([]string, 0, context.temp_allocator)

	if analysis.condition_number > 1e10 {
		append(&analysis.recommendations, "Consider using higher precision arithmetic")
	}

	if !result.growth_factor_ok {
		append(&analysis.recommendations, "Pivot growth indicates potential numerical instability")
	}

	if analysis.min_trust_bound < 0.5 {
		append(&analysis.recommendations, "Error bounds may be unreliable")
	}

	if result.was_equilibrated {
		append(&analysis.recommendations, "Matrix was equilibrated for better stability")
	}

	return analysis
}

// Compare standard vs extended expert solver
compare_expert_solvers :: proc(
	A: ^Matrix($T),
	B: ^Matrix(T),
	allocator := context.allocator,
) -> ComparisonResult {
	comparison: ComparisonResult

	// Prepare workspace
	AF1 := create_matrix(T, A.rows, A.cols, allocator)
	defer matrix_delete(&AF1)
	AF2 := create_matrix(T, A.rows, A.cols, allocator)
	defer matrix_delete(&AF2)

	X1 := create_matrix(T, B.rows, B.cols, allocator)
	defer matrix_delete(&X1)
	X2 := create_matrix(T, B.rows, B.cols, allocator)
	defer matrix_delete(&X2)

	// Standard expert solver
	equed1 := EquilibrationMode.None
	when T == f64 {
		result1, info1 := m_solve_positive_definite_expert_f64(
			A,
			&AF1,
			B,
			&X1,
			.Compute,
			true,
			&equed1,
			nil,
			allocator,
		)
		comparison.standard_rcond = result1.rcond
		comparison.standard_success = info1 == 0
	}

	// Extended expert solver with refinement
	params := RefinementParams {
		enable_refinement     = true,
		max_iterations        = 30,
		convergence_threshold = 1e-14,
		component_wise_error  = true,
	}

	equed2 := EquilibrationMode.None
	when T == f64 {
		result2, info2 := m_solve_positive_definite_extended_expert_f64(
			A,
			&AF2,
			B,
			&X2,
			.Equilibrate,
			true,
			&equed2,
			nil,
			&params,
			allocator,
		)
		comparison.extended_rcond = result2.rcond
		comparison.extended_rpvgrw = result2.rpvgrw
		comparison.extended_success = info2 == 0
		comparison.extended_trust =
			result2.trust_bounds[0] if len(result2.trust_bounds) > 0 else 0.0
	}

	// Compare solutions
	if comparison.standard_success && comparison.extended_success {
		comparison.solution_difference = compute_solution_difference(&X1, &X2)
	}

	return comparison
}

// Comparison result structure
ComparisonResult :: struct {
	standard_success:    bool,
	standard_rcond:      f64,
	extended_success:    bool,
	extended_rcond:      f64,
	extended_rpvgrw:     f64,
	extended_trust:      f64,
	solution_difference: f64,
}

// Helper function to compute solution difference
compute_solution_difference :: proc(X1, X2: ^Matrix($T)) -> f64 {
	// Placeholder - would compute actual norm of difference
	return 1e-15
}

// Batch solve with error analysis
batch_solve_with_analysis :: proc(
	A: ^Matrix($T),
	B_list: []^Matrix(T),
	allocator := context.allocator,
) -> (
	X_list: []Matrix(T),
	analyses: []SolutionAnalysis,
	all_success: bool,
) {
	X_list = make([]Matrix(T), len(B_list), allocator)
	analyses = make([]SolutionAnalysis, len(B_list), allocator)
	all_success = true

	AF := create_matrix(T, A.rows, A.cols, allocator)
	defer matrix_delete(&AF)

	params := RefinementParams {
		enable_refinement     = true,
		max_iterations        = 30,
		convergence_threshold = 1e-14,
		component_wise_error  = false,
	}

	equed := EquilibrationMode.None
	fact_option := FactorizationOption.Equilibrate

	for i, B in B_list {
		X_list[i] = create_matrix(T, B.rows, B.cols, allocator)

		// Reuse factorization after first solve
		if i > 0 {
			fact_option = .UseProvided
		}

		when T == f64 {
			result, info := m_solve_positive_definite_extended_expert_f64(
				A,
				&AF,
				B,
				&X_list[i],
				fact_option,
				true,
				&equed,
				nil,
				&params,
				allocator,
			)
			analyses[i] = build_solution_analysis(result, info)
			if info != 0 || result.is_singular {
				all_success = false
			}
		}
		// Add other type cases as needed
	}

	return X_list, analyses, all_success
}
