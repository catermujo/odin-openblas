package openblas

import lapack "./f77"
import "base:builtin"
import "core:math"
import "core:mem"

// ===================================================================================
// PACKED POSITIVE DEFINITE EXPERT SOLVER AND CHOLESKY FACTORIZATION
// ===================================================================================

// Expert solver proc group
m_solve_packed_positive_definite_expert :: proc {
	m_solve_packed_positive_definite_expert_c64,
	m_solve_packed_positive_definite_expert_f64,
	m_solve_packed_positive_definite_expert_f32,
	m_solve_packed_positive_definite_expert_c128,
}

// Cholesky factorization proc group
m_cholesky_packed :: proc {
	m_cholesky_packed_c64,
	m_cholesky_packed_f64,
	m_cholesky_packed_f32,
	m_cholesky_packed_c128,
}

// ===================================================================================
// PACKED EXPERT SOLVER RESULT
// ===================================================================================

// Packed expert solver result
PackedExpertSolverResult :: struct($T: typeid) {
	rcond:                f64, // Reciprocal condition number
	forward_errors:       []T, // Forward error bounds
	backward_errors:      []T, // Backward error bounds
	scale_factors:        []T, // Equilibration scale factors (if used)
	was_equilibrated:     bool, // True if equilibration was applied
	is_singular:          bool, // True if matrix is singular
	factorization_reused: bool, // True if provided factorization was used
}

// ===================================================================================
// PACKED CHOLESKY FACTORIZATION
// ===================================================================================

// Packed Cholesky factorization (c64)
// Computes the Cholesky factorization of a packed positive definite matrix
m_cholesky_packed_c64 :: proc(
	AP: []complex64, // Packed matrix (input/output)
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

	n_val := Blas_Int(n)
	uplo_c := "U" if uplo_upper else "L"

	lapack.cpptrf_(uplo_c, &n_val, raw_data(AP), &info, len(uplo_c))

	return info
}

// Packed Cholesky factorization (f64)
// Computes the Cholesky factorization of a packed positive definite matrix
m_cholesky_packed_f64 :: proc(
	AP: []f64, // Packed matrix (input/output)
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

	n_val := Blas_Int(n)
	uplo_c := "U" if uplo_upper else "L"

	lapack.dpptrf_(uplo_c, &n_val, raw_data(AP), &info, len(uplo_c))

	return info
}

// Packed Cholesky factorization (f32)
// Computes the Cholesky factorization of a packed positive definite matrix
m_cholesky_packed_f32 :: proc(
	AP: []f32, // Packed matrix (input/output)
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

	n_val := Blas_Int(n)
	uplo_c := "U" if uplo_upper else "L"

	lapack.spptrf_(uplo_c, &n_val, raw_data(AP), &info, len(uplo_c))

	return info
}

// Packed Cholesky factorization (c128)
// Computes the Cholesky factorization of a packed positive definite matrix
m_cholesky_packed_c128 :: proc(
	AP: []complex128, // Packed matrix (input/output)
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

	n_val := Blas_Int(n)
	uplo_c := "U" if uplo_upper else "L"

	lapack.zpptrf_(uplo_c, &n_val, raw_data(AP), &info, len(uplo_c))

	return info
}

// ===================================================================================
// PACKED EXPERT SOLVER IMPLEMENTATION
// ===================================================================================

// Expert solver for packed positive definite system (c64)
// Provides full control over factorization, equilibration, and error bounds
m_solve_packed_positive_definite_expert_c64 :: proc(
	AP: []complex64, // Packed system matrix
	AFP: []complex64, // Factorization workspace/input
	B: ^Matrix(complex64), // RHS matrix
	X: ^Matrix(complex64), // Solution matrix (output)
	n: int, // Matrix dimension
	fact := FactorizationOption.Compute, // Factorization control
	uplo_upper := true, // Upper or lower triangular
	equed_inout: ^EquilibrationMode = nil, // Equilibration state (input/output)
	S_inout: []f32 = nil, // Scale factors (input/output)
	allocator := context.allocator,
) -> (
	result: PackedExpertSolverResult(f32),
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
		S = make([]f32, n, allocator)
	}

	// Allocate error arrays
	result.forward_errors = make([]f32, B.cols, allocator)
	result.backward_errors = make([]f32, B.cols, allocator)

	// Allocate workspace
	work := make([]complex64, 3 * n, allocator)
	defer delete(work)

	rwork := make([]f32, n, allocator)
	defer delete(rwork)

	rcond: f32
	info_val: Info

	// Call LAPACK
	s_ptr: ^f32 = nil
	if len(S) > 0 {
		s_ptr = raw_data(S)
	}

	lapack.cppsvx_(
		fact_c,
		uplo_c,
		&n_val,
		&nrhs,
		raw_data(AP),
		raw_data(AFP),
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

// Expert solver for packed positive definite system (f64)
m_solve_packed_positive_definite_expert_f64 :: proc(
	AP: []f64, // Packed system matrix
	AFP: []f64, // Factorization workspace/input
	B: ^Matrix(f64), // RHS matrix
	X: ^Matrix(f64), // Solution matrix (output)
	n: int, // Matrix dimension
	fact := FactorizationOption.Compute, // Factorization control
	uplo_upper := true, // Upper or lower triangular
	equed_inout: ^EquilibrationMode = nil, // Equilibration state (input/output)
	S_inout: []f64 = nil, // Scale factors (input/output)
	allocator := context.allocator,
) -> (
	result: PackedExpertSolverResult(f64),
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

	fact_c := _fact_to_char(fact)
	uplo_c := "U" if uplo_upper else "L"

	equed_mode := EquilibrationMode.None
	if equed_inout != nil {
		equed_mode = equed_inout^
	}
	equed_c := _equed_mode_to_char(equed_mode)

	S := S_inout
	if len(S) == 0 && fact == .Equilibrate {
		S = make([]f64, n, allocator)
	}

	result.forward_errors = make([]f64, B.cols, allocator)
	result.backward_errors = make([]f64, B.cols, allocator)

	work := make([]f64, 3 * n, allocator)
	defer delete(work)

	iwork := make([]Blas_Int, n, allocator)
	defer delete(iwork)

	rcond: f64
	info_val: Info

	s_ptr: ^f64 = nil
	if len(S) > 0 {
		s_ptr = raw_data(S)
	}

	lapack.dppsvx_(
		fact_c,
		uplo_c,
		&n_val,
		&nrhs,
		raw_data(AP),
		raw_data(AFP),
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

// Expert solver for packed positive definite system (f32)
m_solve_packed_positive_definite_expert_f32 :: proc(
	AP: []f32, // Packed system matrix
	AFP: []f32, // Factorization workspace/input
	B: ^Matrix(f32), // RHS matrix
	X: ^Matrix(f32), // Solution matrix (output)
	n: int, // Matrix dimension
	fact := FactorizationOption.Compute, // Factorization control
	uplo_upper := true, // Upper or lower triangular
	equed_inout: ^EquilibrationMode = nil, // Equilibration state (input/output)
	S_inout: []f32 = nil, // Scale factors (input/output)
	allocator := context.allocator,
) -> (
	result: PackedExpertSolverResult(f32),
	info: Info,
) {
	// Similar implementation to f64 version
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

	fact_c := _fact_to_char(fact)
	uplo_c := "U" if uplo_upper else "L"

	equed_mode := EquilibrationMode.None
	if equed_inout != nil {
		equed_mode = equed_inout^
	}
	equed_c := _equed_mode_to_char(equed_mode)

	S := S_inout
	if len(S) == 0 && fact == .Equilibrate {
		S = make([]f32, n, allocator)
	}

	result.forward_errors = make([]f32, B.cols, allocator)
	result.backward_errors = make([]f32, B.cols, allocator)

	work := make([]f32, 3 * n, allocator)
	defer delete(work)

	iwork := make([]Blas_Int, n, allocator)
	defer delete(iwork)

	rcond: f32
	info_val: Info

	s_ptr: ^f32 = nil
	if len(S) > 0 {
		s_ptr = raw_data(S)
	}

	lapack.sppsvx_(
		fact_c,
		uplo_c,
		&n_val,
		&nrhs,
		raw_data(AP),
		raw_data(AFP),
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

// Expert solver for packed positive definite system (c128)
m_solve_packed_positive_definite_expert_c128 :: proc(
	AP: []complex128, // Packed system matrix
	AFP: []complex128, // Factorization workspace/input
	B: ^Matrix(complex128), // RHS matrix
	X: ^Matrix(complex128), // Solution matrix (output)
	n: int, // Matrix dimension
	fact := FactorizationOption.Compute, // Factorization control
	uplo_upper := true, // Upper or lower triangular
	equed_inout: ^EquilibrationMode = nil, // Equilibration state (input/output)
	S_inout: []f64 = nil, // Scale factors (input/output)
	allocator := context.allocator,
) -> (
	result: PackedExpertSolverResult(f64),
	info: Info,
) {
	// Similar implementation to c64 version
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

	fact_c := _fact_to_char(fact)
	uplo_c := "U" if uplo_upper else "L"

	equed_mode := EquilibrationMode.None
	if equed_inout != nil {
		equed_mode = equed_inout^
	}
	equed_c := _equed_mode_to_char(equed_mode)

	S := S_inout
	if len(S) == 0 && fact == .Equilibrate {
		S = make([]f64, n, allocator)
	}

	result.forward_errors = make([]f64, B.cols, allocator)
	result.backward_errors = make([]f64, B.cols, allocator)

	work := make([]complex128, 3 * n, allocator)
	defer delete(work)

	rwork := make([]f64, n, allocator)
	defer delete(rwork)

	rcond: f64
	info_val: Info

	s_ptr: ^f64 = nil
	if len(S) > 0 {
		s_ptr = raw_data(S)
	}

	lapack.zppsvx_(
		fact_c,
		uplo_c,
		&n_val,
		&nrhs,
		raw_data(AP),
		raw_data(AFP),
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

// Perform Cholesky factorization on packed matrix
cholesky_factor_packed :: proc(
	packed: ^PackedMatrix($T),
	allocator := context.allocator,
) -> (
	factor: PackedMatrix(T),
	success: bool,
) {
	// Clone packed matrix for factorization
	factor.n = packed.n
	factor.uplo_upper = packed.uplo_upper
	factor.data = make([]T, len(packed.data), allocator)
	copy(factor.data, packed.data)

	// Perform factorization
	var; info: Info
	when T == complex64 {
		info = m_cholesky_packed_c64(factor.data, factor.n, factor.uplo_upper)
	} else when T == f64 {
		info = m_cholesky_packed_f64(factor.data, factor.n, factor.uplo_upper)
	} else when T == f32 {
		info = m_cholesky_packed_f32(factor.data, factor.n, factor.uplo_upper)
	} else when T == complex128 {
		info = m_cholesky_packed_c128(factor.data, factor.n, factor.uplo_upper)
	}

	success = info == 0
	if !success {
		delete(factor.data)
		factor.data = nil
	}

	return factor, success
}

// Check if packed matrix is positive definite
is_packed_positive_definite :: proc(
	packed: ^PackedMatrix($T),
	allocator := context.temp_allocator,
) -> bool {
	// Clone packed matrix since factorization modifies it
	test_data := make([]T, len(packed.data), allocator)
	copy(test_data, packed.data)
	defer delete(test_data)

	var; info: Info
	when T == complex64 {
		info = m_cholesky_packed_c64(test_data, packed.n, packed.uplo_upper)
	} else when T == f64 {
		info = m_cholesky_packed_f64(test_data, packed.n, packed.uplo_upper)
	} else when T == f32 {
		info = m_cholesky_packed_f32(test_data, packed.n, packed.uplo_upper)
	} else when T == complex128 {
		info = m_cholesky_packed_c128(test_data, packed.n, packed.uplo_upper)
	}

	return info == 0
}

// Solve with automatic equilibration and condition estimation for packed matrix
solve_packed_with_conditioning :: proc(
	packed: ^PackedMatrix($T),
	B: ^Matrix(T),
	allocator := context.allocator,
) -> (
	X: Matrix(T),
	condition: f64,
	success: bool,
) {
	// Create workspace
	AFP := make([]T, len(packed.data), allocator)
	defer delete(AFP)

	X = create_matrix(T, B.rows, B.cols, allocator)

	// Use expert solver with equilibration
	equed := EquilibrationMode.None

	when T == complex64 {
		result, info := m_solve_packed_positive_definite_expert_c64(
			packed.data,
			AFP,
			B,
			&X,
			packed.n,
			.Equilibrate,
			packed.uplo_upper,
			&equed,
			nil,
			allocator,
		)
		condition = result.rcond
		if result.rcond > 0 {
			condition = 1.0 / result.rcond
		}
		success = info == 0 && !result.is_singular
		defer if result.forward_errors != nil do delete(result.forward_errors)
		defer if result.backward_errors != nil do delete(result.backward_errors)
		defer if result.scale_factors != nil do delete(result.scale_factors)
	} else when T == f64 {
		result, info := m_solve_packed_positive_definite_expert_f64(
			packed.data,
			AFP,
			B,
			&X,
			packed.n,
			.Equilibrate,
			packed.uplo_upper,
			&equed,
			nil,
			allocator,
		)
		condition = result.rcond
		if result.rcond > 0 {
			condition = 1.0 / result.rcond
		}
		success = info == 0 && !result.is_singular
		defer if result.forward_errors != nil do delete(result.forward_errors)
		defer if result.backward_errors != nil do delete(result.backward_errors)
		defer if result.scale_factors != nil do delete(result.scale_factors)
	} else when T == f32 {
		result, info := m_solve_packed_positive_definite_expert_f32(
			packed.data,
			AFP,
			B,
			&X,
			packed.n,
			.Equilibrate,
			packed.uplo_upper,
			&equed,
			nil,
			allocator,
		)
		condition = result.rcond
		if result.rcond > 0 {
			condition = 1.0 / result.rcond
		}
		success = info == 0 && !result.is_singular
		defer if result.forward_errors != nil do delete(result.forward_errors)
		defer if result.backward_errors != nil do delete(result.backward_errors)
		defer if result.scale_factors != nil do delete(result.scale_factors)
	} else when T == complex128 {
		result, info := m_solve_packed_positive_definite_expert_c128(
			packed.data,
			AFP,
			B,
			&X,
			packed.n,
			.Equilibrate,
			packed.uplo_upper,
			&equed,
			nil,
			allocator,
		)
		condition = result.rcond
		if result.rcond > 0 {
			condition = 1.0 / result.rcond
		}
		success = info == 0 && !result.is_singular
		defer if result.forward_errors != nil do delete(result.forward_errors)
		defer if result.backward_errors != nil do delete(result.backward_errors)
		defer if result.scale_factors != nil do delete(result.scale_factors)
	}

	return X, condition, success
}

// Solve multiple packed systems with factorization reuse
solve_packed_expert_multiple :: proc(
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

	// Create factorization workspace
	AFP := make([]T, len(packed.data), allocator)
	defer delete(AFP)

	X_list = make([]Matrix(T), len(B_list), allocator)
	all_success = true

	// Solve first system and compute factorization
	equed := EquilibrationMode.None
	fact_option := FactorizationOption.Equilibrate
	var; scale_factors: []T

	for i, B in B_list {
		X_list[i] = create_matrix(T, B.rows, B.cols, allocator)

		// Reuse factorization after first solve
		if i > 0 && fact_option != .UseProvided {
			fact_option = .UseProvided
		}

		when T == complex64 {
			result, info := m_solve_packed_positive_definite_expert_c64(
				packed.data,
				AFP,
				B,
				&X_list[i],
				packed.n,
				fact_option,
				packed.uplo_upper,
				&equed,
				scale_factors[:],
				allocator,
			)
			if i == 0 && result.scale_factors != nil {
				scale_factors = result.scale_factors
			}
			if info != 0 || result.is_singular {
				all_success = false
			}
		} else when T == f64 {
			result, info := m_solve_packed_positive_definite_expert_f64(
				packed.data,
				AFP,
				B,
				&X_list[i],
				packed.n,
				fact_option,
				packed.uplo_upper,
				&equed,
				scale_factors[:],
				allocator,
			)
			if i == 0 && result.scale_factors != nil {
				scale_factors = result.scale_factors
			}
			if info != 0 || result.is_singular {
				all_success = false
			}
		}
		// Add other type cases as needed
	}

	// Clean up scale factors
	if scale_factors != nil {
		delete(scale_factors)
	}

	return X_list, all_success
}

// Compute log-determinant using packed Cholesky
packed_log_determinant :: proc(
	packed: ^PackedMatrix($T),
	allocator := context.temp_allocator,
) -> (
	log_det: f64,
	sign: f64,
	success: bool,
) {
	// Factor the packed matrix
	factor, factor_success := cholesky_factor_packed(packed, allocator)
	defer if factor.data != nil do delete(factor.data)

	if !factor_success {
		return math.NEG_INF_F64, 0.0, false
	}

	// Compute log determinant from diagonal of factor
	log_det = 0.0
	idx := 0

	if factor.uplo_upper {
		// Upper triangle: diagonal elements at positions 0, 2, 5, 9, ...
		for i in 0 ..< factor.n {
			diag_idx := i * (i + 1) / 2 + i
			diag_elem := factor.data[diag_idx]
			when T == complex64 || T == complex128 {
				log_det += math.log(abs_value(diag_elem))
			} else {
				log_det += math.log(f64(abs(diag_elem)))
			}
		}
	} else {
		// Lower triangle: diagonal elements at positions 0, n, 2n-1, 3n-3, ...
		for i in 0 ..< factor.n {
			diag_idx := i * (2 * factor.n - i - 1) / 2 + i
			diag_elem := factor.data[diag_idx]
			when T == complex64 || T == complex128 {
				log_det += math.log(abs_value(diag_elem))
			} else {
				log_det += math.log(f64(abs(diag_elem)))
			}
		}
	}

	// Multiply by 2 since det(A) = det(L)^2
	log_det *= 2.0
	sign = 1.0 // Always positive for positive definite matrices

	return log_det, sign, true
}

// Analyze packed expert solver performance
analyze_packed_expert :: proc(
	packed: ^PackedMatrix($T),
	B: ^Matrix(T),
	allocator := context.allocator,
) -> PackedExpertAnalysis {
	analysis: PackedExpertAnalysis

	// Test with different factorization options
	AFP1 := make([]T, len(packed.data), allocator)
	defer delete(AFP1)
	X1 := create_matrix(T, B.rows, B.cols, allocator)
	defer matrix_delete(&X1)

	// Without equilibration
	when T == f64 {
		result1, info1 := m_solve_packed_positive_definite_expert_f64(
			packed.data,
			AFP1,
			B,
			&X1,
			packed.n,
			.Compute,
			packed.uplo_upper,
			nil,
			nil,
			allocator,
		)
		analysis.without_equilibration.rcond = result1.rcond
		analysis.without_equilibration.success = info1 == 0
		if result1.backward_errors != nil {
			analysis.without_equilibration.max_backward_error = max_array(result1.backward_errors)
			defer delete(result1.backward_errors)
		}
		if result1.forward_errors != nil {
			defer delete(result1.forward_errors)
		}
	}

	// With equilibration
	AFP2 := make([]T, len(packed.data), allocator)
	defer delete(AFP2)
	X2 := create_matrix(T, B.rows, B.cols, allocator)
	defer matrix_delete(&X2)

	equed := EquilibrationMode.None
	when T == f64 {
		result2, info2 := m_solve_packed_positive_definite_expert_f64(
			packed.data,
			AFP2,
			B,
			&X2,
			packed.n,
			.Equilibrate,
			packed.uplo_upper,
			&equed,
			nil,
			allocator,
		)
		analysis.with_equilibration.rcond = result2.rcond
		analysis.with_equilibration.success = info2 == 0
		analysis.with_equilibration.was_equilibrated = result2.was_equilibrated
		if result2.backward_errors != nil {
			analysis.with_equilibration.max_backward_error = max_array(result2.backward_errors)
			defer delete(result2.backward_errors)
		}
		if result2.forward_errors != nil {
			defer delete(result2.forward_errors)
		}
		if result2.scale_factors != nil {
			defer delete(result2.scale_factors)
		}
	}

	// Determine recommendation
	if analysis.with_equilibration.was_equilibrated &&
	   analysis.with_equilibration.rcond > analysis.without_equilibration.rcond * 10 {
		analysis.recommendation = .UseEquilibration
	} else {
		analysis.recommendation = .NoEquilibration
	}

	return analysis
}

// Packed expert analysis structure
PackedExpertAnalysis :: struct {
	without_equilibration: SolverMetrics,
	with_equilibration:    SolverMetrics,
	recommendation:        EquilibrationRecommendation,
}

SolverMetrics :: struct {
	rcond:              f64,
	max_backward_error: f64,
	success:            bool,
	was_equilibrated:   bool,
}

EquilibrationRecommendation :: enum {
	NoEquilibration,
	UseEquilibration,
}

// Helper functions
