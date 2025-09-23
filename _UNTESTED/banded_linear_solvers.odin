package openblas

import lapack "./f77"
import "base:builtin"
import "core:mem"

// ===================================================================================
// POSITIVE DEFINITE BANDED LINEAR SYSTEM SOLVERS
// ===================================================================================

// Simple solver for positive definite banded systems proc group
m_solve_banded_pd :: proc {
	m_solve_banded_pd_c64,
	m_solve_banded_pd_f64,
	m_solve_banded_pd_f32,
	m_solve_banded_pd_c128,
}

// Expert solver for positive definite banded systems proc group
m_solve_banded_pd_expert :: proc {
	m_solve_banded_pd_expert_c64,
	m_solve_banded_pd_expert_f64,
	m_solve_banded_pd_expert_f32,
	m_solve_banded_pd_expert_c128,
}

// ===================================================================================
// SIMPLE SOLVER IMPLEMENTATION
// ===================================================================================

// Solve positive definite banded system (c64)
// Solves A*X = B where A is positive definite banded
m_solve_banded_pd_c64 :: proc(
	AB: ^Matrix(complex64), // Banded matrix (input/output - factorized on output)
	B: ^Matrix(complex64), // Right-hand side (input/output - solution on output)
	kd: int, // Number of super/sub-diagonals
	uplo_upper := true, // Upper or lower triangular storage
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate inputs
	if len(AB.data) == 0 || len(B.data) == 0 {
		panic("Matrices cannot be empty")
	}
	if AB.rows != AB.cols {
		panic("AB must be square")
	}
	if B.rows != AB.rows {
		panic("System dimensions must be consistent")
	}
	if kd < 0 || kd >= AB.rows {
		panic("Invalid bandwidth kd")
	}

	uplo_c := "U" if uplo_upper else "L"
	n := Blas_Int(AB.cols)
	kd_val := Blas_Int(kd)
	nrhs := Blas_Int(B.cols)
	ldab := Blas_Int(AB.ld)
	ldb := Blas_Int(B.ld)
	info_val: Info

	lapack.cpbsv_(
		uplo_c,
		&n,
		&kd_val,
		&nrhs,
		raw_data(AB.data),
		&ldab,
		raw_data(B.data),
		&ldb,
		&info_val,
		len(uplo_c),
	)

	return info_val == 0, info_val
}

// Solve positive definite banded system (f64)
// Solves A*X = B where A is positive definite banded
m_solve_banded_pd_f64 :: proc(
	AB: ^Matrix(f64), // Banded matrix (input/output - factorized on output)
	B: ^Matrix(f64), // Right-hand side (input/output - solution on output)
	kd: int, // Number of super/sub-diagonals
	uplo_upper := true, // Upper or lower triangular storage
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate inputs
	if len(AB.data) == 0 || len(B.data) == 0 {
		panic("Matrices cannot be empty")
	}
	if AB.rows != AB.cols {
		panic("AB must be square")
	}
	if B.rows != AB.rows {
		panic("System dimensions must be consistent")
	}
	if kd < 0 || kd >= AB.rows {
		panic("Invalid bandwidth kd")
	}

	uplo_c := "U" if uplo_upper else "L"
	n := Blas_Int(AB.cols)
	kd_val := Blas_Int(kd)
	nrhs := Blas_Int(B.cols)
	ldab := Blas_Int(AB.ld)
	ldb := Blas_Int(B.ld)
	info_val: Info

	lapack.dpbsv_(
		uplo_c,
		&n,
		&kd_val,
		&nrhs,
		raw_data(AB.data),
		&ldab,
		raw_data(B.data),
		&ldb,
		&info_val,
		len(uplo_c),
	)

	return info_val == 0, info_val
}

// Solve positive definite banded system (f32)
// Solves A*X = B where A is positive definite banded
m_solve_banded_pd_f32 :: proc(
	AB: ^Matrix(f32), // Banded matrix (input/output - factorized on output)
	B: ^Matrix(f32), // Right-hand side (input/output - solution on output)
	kd: int, // Number of super/sub-diagonals
	uplo_upper := true, // Upper or lower triangular storage
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate inputs
	if len(AB.data) == 0 || len(B.data) == 0 {
		panic("Matrices cannot be empty")
	}
	if AB.rows != AB.cols {
		panic("AB must be square")
	}
	if B.rows != AB.rows {
		panic("System dimensions must be consistent")
	}
	if kd < 0 || kd >= AB.rows {
		panic("Invalid bandwidth kd")
	}

	uplo_c := "U" if uplo_upper else "L"
	n := Blas_Int(AB.cols)
	kd_val := Blas_Int(kd)
	nrhs := Blas_Int(B.cols)
	ldab := Blas_Int(AB.ld)
	ldb := Blas_Int(B.ld)
	info_val: Info

	lapack.spbsv_(
		uplo_c,
		&n,
		&kd_val,
		&nrhs,
		raw_data(AB.data),
		&ldab,
		raw_data(B.data),
		&ldb,
		&info_val,
		len(uplo_c),
	)

	return info_val == 0, info_val
}

// Solve positive definite banded system (c128)
// Solves A*X = B where A is positive definite banded
m_solve_banded_pd_c128 :: proc(
	AB: ^Matrix(complex128), // Banded matrix (input/output - factorized on output)
	B: ^Matrix(complex128), // Right-hand side (input/output - solution on output)
	kd: int, // Number of super/sub-diagonals
	uplo_upper := true, // Upper or lower triangular storage
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate inputs
	if len(AB.data) == 0 || len(B.data) == 0 {
		panic("Matrices cannot be empty")
	}
	if AB.rows != AB.cols {
		panic("AB must be square")
	}
	if B.rows != AB.rows {
		panic("System dimensions must be consistent")
	}
	if kd < 0 || kd >= AB.rows {
		panic("Invalid bandwidth kd")
	}

	uplo_c := "U" if uplo_upper else "L"
	n := Blas_Int(AB.cols)
	kd_val := Blas_Int(kd)
	nrhs := Blas_Int(B.cols)
	ldab := Blas_Int(AB.ld)
	ldb := Blas_Int(B.ld)
	info_val: Info

	lapack.zpbsv_(
		uplo_c,
		&n,
		&kd_val,
		&nrhs,
		raw_data(AB.data),
		&ldab,
		raw_data(B.data),
		&ldb,
		&info_val,
		len(uplo_c),
	)

	return info_val == 0, info_val
}

// ===================================================================================
// EXPERT SOLVER IMPLEMENTATION
// ===================================================================================

// Factorization option
FactorizationOption :: enum {
	Equilibrate, // "E" - Equilibrate, then factor
	NoFactorization, // "N" - Matrix already factored
	Factor, // "F" - Factor the matrix
}

// Convert factorization option to LAPACK character
_factorization_to_char :: proc(fact: FactorizationOption) -> cstring {
	switch fact {
	case .Equilibrate:
		return "E"
	case .NoFactorization:
		return "N"
	case .Factor:
		return "F"
	case:
		return "F"
	}
}

// Equilibration state
EquilibrationState :: enum {
	None, // "N" - No equilibration
	Applied, // "Y" - Equilibration was applied
}

// Convert equilibration state to LAPACK character
_equilibration_to_char :: proc(equed: EquilibrationState) -> cstring {
	switch equed {
	case .None:
		return "N"
	case .Applied:
		return "Y"
	case:
		return "N"
	}
}

// Expert solver result structure
ExpertSolverResult :: struct($T: typeid) {
	X:       Matrix(T), // Solution matrix
	rcond:   T, // Reciprocal condition number
	ferr:    []T, // Forward error bounds
	berr:    []T, // Backward error bounds
	equed:   EquilibrationState, // Equilibration state
	S:       []T, // Scaling factors (if equilibrated)
	success: bool,
	info:    Blas_Int,
}

// Expert solve for positive definite banded system (c64)
// Solves with equilibration, condition estimation, and error bounds
m_solve_banded_pd_expert_c64 :: proc(
	AB: ^Matrix(complex64), // Banded matrix (input/output)
	B: ^Matrix(complex64), // Right-hand side (input/output)
	kd: int, // Number of super/sub-diagonals
	fact_option := FactorizationOption.Equilibrate,
	uplo_upper := true, // Upper or lower triangular storage
	AFB: ^Matrix(complex64) = nil, // Pre-factored matrix (optional)
	S_in: []f32 = nil, // Input scaling factors (optional)
	allocator := context.allocator,
) -> ExpertSolverResult(f32) {
	// Validate inputs
	n := AB.cols
	nrhs := B.cols

	// Prepare matrices
	AFB_local: Matrix(complex64)
	if AFB == nil {
		AFB_local = make_matrix(complex64, n, n, AB.format, allocator)
		copy_matrix(AB, &AFB_local)
		AFB = &AFB_local
	}

	// Prepare solution matrix
	X := make_matrix(complex64, B.rows, B.cols, B.format, allocator)

	// Prepare arrays
	S := S_in != nil ? S_in : make([]f32, n, allocator)
	ferr := make([]f32, nrhs, allocator)
	berr := make([]f32, nrhs, allocator)
	rcond: f32
	equed_c := _equilibration_to_char(.None)

	// Prepare parameters
	fact_c := _factorization_to_char(fact_option)
	uplo_c := "U" if uplo_upper else "L"
	n_val := Blas_Int(n)
	kd_val := Blas_Int(kd)
	nrhs_val := Blas_Int(nrhs)
	ldab := Blas_Int(AB.ld)
	ldafb := Blas_Int(AFB.ld)
	ldb := Blas_Int(B.ld)
	ldx := Blas_Int(X.ld)

	// Allocate workspace
	work := make([]complex64, 2 * n, context.temp_allocator)
	rwork := make([]f32, n, context.temp_allocator)

	info_val: Info

	lapack.cpbsvx_(
		fact_c,
		uplo_c,
		&n_val,
		&kd_val,
		&nrhs_val,
		raw_data(AB.data),
		&ldab,
		raw_data(AFB.data),
		&ldafb,
		&equed_c,
		raw_data(S),
		raw_data(B.data),
		&ldb,
		raw_data(X.data),
		&ldx,
		&rcond,
		raw_data(ferr),
		raw_data(berr),
		raw_data(work),
		raw_data(rwork),
		&info_val,
		len(fact_c),
		len(uplo_c),
		1, // Last is for equed
	)

	equed_state := EquilibrationState.Applied if equed_c == "Y" else EquilibrationState.None

	return ExpertSolverResult(f32) {
		X = X,
		rcond = rcond,
		ferr = ferr,
		berr = berr,
		equed = equed_state,
		S = S,
		success = info_val == 0,
		info = info_val,
	}
}

// Expert solve for positive definite banded system (f64)
m_solve_banded_pd_expert_f64 :: proc(
	AB: ^Matrix(f64), // Banded matrix (input/output)
	B: ^Matrix(f64), // Right-hand side (input/output)
	kd: int, // Number of super/sub-diagonals
	fact_option := FactorizationOption.Equilibrate,
	uplo_upper := true, // Upper or lower triangular storage
	AFB: ^Matrix(f64) = nil, // Pre-factored matrix (optional)
	S_in: []f64 = nil, // Input scaling factors (optional)
	allocator := context.allocator,
) -> ExpertSolverResult(f64) {
	// Validate inputs
	n := AB.cols
	nrhs := B.cols

	// Prepare matrices
	AFB_local: Matrix(f64)
	if AFB == nil {
		AFB_local = make_matrix(f64, n, n, AB.format, allocator)
		copy_matrix(AB, &AFB_local)
		AFB = &AFB_local
	}

	// Prepare solution matrix
	X := make_matrix(f64, B.rows, B.cols, B.format, allocator)

	// Prepare arrays
	S := S_in != nil ? S_in : make([]f64, n, allocator)
	ferr := make([]f64, nrhs, allocator)
	berr := make([]f64, nrhs, allocator)
	rcond: f64
	equed_c := _equilibration_to_char(.None)

	// Prepare parameters
	fact_c := _factorization_to_char(fact_option)
	uplo_c := "U" if uplo_upper else "L"
	n_val := Blas_Int(n)
	kd_val := Blas_Int(kd)
	nrhs_val := Blas_Int(nrhs)
	ldab := Blas_Int(AB.ld)
	ldafb := Blas_Int(AFB.ld)
	ldb := Blas_Int(B.ld)
	ldx := Blas_Int(X.ld)

	// Allocate workspace
	work := make([]f64, 3 * n, context.temp_allocator)
	iwork := make([]Blas_Int, n, context.temp_allocator)

	info_val: Info

	lapack.dpbsvx_(
		fact_c,
		uplo_c,
		&n_val,
		&kd_val,
		&nrhs_val,
		raw_data(AB.data),
		&ldab,
		raw_data(AFB.data),
		&ldafb,
		&equed_c,
		raw_data(S),
		raw_data(B.data),
		&ldb,
		raw_data(X.data),
		&ldx,
		&rcond,
		raw_data(ferr),
		raw_data(berr),
		raw_data(work),
		raw_data(iwork),
		&info_val,
		len(fact_c),
		len(uplo_c),
		1,
	)

	equed_state := EquilibrationState.Applied if equed_c == "Y" else EquilibrationState.None

	return ExpertSolverResult(f64) {
		X = X,
		rcond = rcond,
		ferr = ferr,
		berr = berr,
		equed = equed_state,
		S = S,
		success = info_val == 0,
		info = info_val,
	}
}

// Expert solve for positive definite banded system (f32)
m_solve_banded_pd_expert_f32 :: proc(
	AB: ^Matrix(f32), // Banded matrix (input/output)
	B: ^Matrix(f32), // Right-hand side (input/output)
	kd: int, // Number of super/sub-diagonals
	fact_option := FactorizationOption.Equilibrate,
	uplo_upper := true, // Upper or lower triangular storage
	AFB: ^Matrix(f32) = nil, // Pre-factored matrix (optional)
	S_in: []f32 = nil, // Input scaling factors (optional)
	allocator := context.allocator,
) -> ExpertSolverResult(f32) {
	// Validate inputs
	n := AB.cols
	nrhs := B.cols

	// Prepare matrices
	AFB_local: Matrix(f32)
	if AFB == nil {
		AFB_local = make_matrix(f32, n, n, AB.format, allocator)
		copy_matrix(AB, &AFB_local)
		AFB = &AFB_local
	}

	// Prepare solution matrix
	X := make_matrix(f32, B.rows, B.cols, B.format, allocator)

	// Prepare arrays
	S := S_in != nil ? S_in : make([]f32, n, allocator)
	ferr := make([]f32, nrhs, allocator)
	berr := make([]f32, nrhs, allocator)
	rcond: f32
	equed_c := _equilibration_to_char(.None)

	// Prepare parameters
	fact_c := _factorization_to_char(fact_option)
	uplo_c := "U" if uplo_upper else "L"
	n_val := Blas_Int(n)
	kd_val := Blas_Int(kd)
	nrhs_val := Blas_Int(nrhs)
	ldab := Blas_Int(AB.ld)
	ldafb := Blas_Int(AFB.ld)
	ldb := Blas_Int(B.ld)
	ldx := Blas_Int(X.ld)

	// Allocate workspace
	work := make([]f32, 3 * n, context.temp_allocator)
	iwork := make([]Blas_Int, n, context.temp_allocator)

	info_val: Info

	lapack.spbsvx_(
		fact_c,
		uplo_c,
		&n_val,
		&kd_val,
		&nrhs_val,
		raw_data(AB.data),
		&ldab,
		raw_data(AFB.data),
		&ldafb,
		&equed_c,
		raw_data(S),
		raw_data(B.data),
		&ldb,
		raw_data(X.data),
		&ldx,
		&rcond,
		raw_data(ferr),
		raw_data(berr),
		raw_data(work),
		raw_data(iwork),
		&info_val,
		len(fact_c),
		len(uplo_c),
		1,
	)

	equed_state := EquilibrationState.Applied if equed_c == "Y" else EquilibrationState.None

	return ExpertSolverResult(f32) {
		X = X,
		rcond = rcond,
		ferr = ferr,
		berr = berr,
		equed = equed_state,
		S = S,
		success = info_val == 0,
		info = info_val,
	}
}

// Expert solve for positive definite banded system (c128)
m_solve_banded_pd_expert_c128 :: proc(
	AB: ^Matrix(complex128), // Banded matrix (input/output)
	B: ^Matrix(complex128), // Right-hand side (input/output)
	kd: int, // Number of super/sub-diagonals
	fact_option := FactorizationOption.Equilibrate,
	uplo_upper := true, // Upper or lower triangular storage
	AFB: ^Matrix(complex128) = nil, // Pre-factored matrix (optional)
	S_in: []f64 = nil, // Input scaling factors (optional)
	allocator := context.allocator,
) -> ExpertSolverResult(f64) {
	// Validate inputs
	n := AB.cols
	nrhs := B.cols

	// Prepare matrices
	AFB_local: Matrix(complex128)
	if AFB == nil {
		AFB_local = make_matrix(complex128, n, n, AB.format, allocator)
		copy_matrix(AB, &AFB_local)
		AFB = &AFB_local
	}

	// Prepare solution matrix
	X := make_matrix(complex128, B.rows, B.cols, B.format, allocator)

	// Prepare arrays
	S := S_in != nil ? S_in : make([]f64, n, allocator)
	ferr := make([]f64, nrhs, allocator)
	berr := make([]f64, nrhs, allocator)
	rcond: f64
	equed_c := _equilibration_to_char(.None)

	// Prepare parameters
	fact_c := _factorization_to_char(fact_option)
	uplo_c := "U" if uplo_upper else "L"
	n_val := Blas_Int(n)
	kd_val := Blas_Int(kd)
	nrhs_val := Blas_Int(nrhs)
	ldab := Blas_Int(AB.ld)
	ldafb := Blas_Int(AFB.ld)
	ldb := Blas_Int(B.ld)
	ldx := Blas_Int(X.ld)

	// Allocate workspace
	work := make([]complex128, 2 * n, context.temp_allocator)
	rwork := make([]f64, n, context.temp_allocator)

	info_val: Info

	lapack.zpbsvx_(
		fact_c,
		uplo_c,
		&n_val,
		&kd_val,
		&nrhs_val,
		raw_data(AB.data),
		&ldab,
		raw_data(AFB.data),
		&ldafb,
		&equed_c,
		raw_data(S),
		raw_data(B.data),
		&ldb,
		raw_data(X.data),
		&ldx,
		&rcond,
		raw_data(ferr),
		raw_data(berr),
		raw_data(work),
		raw_data(rwork),
		&info_val,
		len(fact_c),
		len(uplo_c),
		1,
	)

	equed_state := EquilibrationState.Applied if equed_c == "Y" else EquilibrationState.None

	return ExpertSolverResult(f64) {
		X = X,
		rcond = rcond,
		ferr = ferr,
		berr = berr,
		equed = equed_state,
		S = S,
		success = info_val == 0,
		info = info_val,
	}
}

// ===================================================================================
// CONVENIENCE FUNCTIONS
// ===================================================================================

// Simple solve with solution extraction
solve_banded_system :: proc(
	A: Matrix($T), // Input matrix (will be copied)
	b: []T, // Right-hand side vector
	kd: int, // Bandwidth
	uplo_upper := true,
	allocator := context.allocator,
) -> (
	x: []T,
	success: bool,
) {
	// Create working copies
	AB := make_banded_matrix(T, A.rows, A.cols, kd, kd, allocator)
	copy_matrix(&A, &AB)

	B := make_matrix(T, len(b), 1, .General, allocator)
	for i in 0 ..< len(b) {
		matrix_set(&B, i, 0, b[i])
	}

	// Solve system
	when T == complex64 {
		success, _ := m_solve_banded_pd_c64(&AB, &B, kd, uplo_upper, allocator)
	} else when T == f64 {
		success, _ := m_solve_banded_pd_f64(&AB, &B, kd, uplo_upper, allocator)
	} else when T == f32 {
		success, _ := m_solve_banded_pd_f32(&AB, &B, kd, uplo_upper, allocator)
	} else when T == complex128 {
		success, _ := m_solve_banded_pd_c128(&AB, &B, kd, uplo_upper, allocator)
	} else {
		panic("Unsupported type for banded solve")
	}

	// Extract solution
	if success {
		x = make([]T, len(b), allocator)
		for i in 0 ..< len(b) {
			x[i] = matrix_get(&B, i, 0)
		}
	}

	delete_matrix(&AB)
	delete_matrix(&B)
	return x, success
}

// Expert solve with all features
solve_banded_expert :: proc(
	A: Matrix($T), // Input matrix
	b: []T, // Right-hand side
	kd: int, // Bandwidth
	equilibrate := true, // Whether to equilibrate
	uplo_upper := true,
	allocator := context.allocator,
) -> (
	result: ExpertSolverResult(auto),
) {
	// Create working copies
	AB := make_banded_matrix(T, A.rows, A.cols, kd, kd, allocator)
	copy_matrix(&A, &AB)

	B := make_matrix(T, len(b), 1, .General, allocator)
	for i in 0 ..< len(b) {
		matrix_set(&B, i, 0, b[i])
	}

	fact_option := FactorizationOption.Equilibrate if equilibrate else FactorizationOption.Factor

	when T == complex64 {
		result = m_solve_banded_pd_expert_c64(
			&AB,
			&B,
			kd,
			fact_option,
			uplo_upper,
			nil,
			nil,
			allocator,
		)
	} else when T == f64 {
		result = m_solve_banded_pd_expert_f64(
			&AB,
			&B,
			kd,
			fact_option,
			uplo_upper,
			nil,
			nil,
			allocator,
		)
	} else when T == f32 {
		result = m_solve_banded_pd_expert_f32(
			&AB,
			&B,
			kd,
			fact_option,
			uplo_upper,
			nil,
			nil,
			allocator,
		)
	} else when T == complex128 {
		result = m_solve_banded_pd_expert_c128(
			&AB,
			&B,
			kd,
			fact_option,
			uplo_upper,
			nil,
			nil,
			allocator,
		)
	} else {
		panic("Unsupported type for expert solve")
	}

	delete_matrix(&AB)
	delete_matrix(&B)
	return result
}

// Check solution quality from expert solver
is_solution_accurate :: proc(result: ExpertSolverResult($T), tolerance := T(1e-10)) -> bool {
	if !result.success {
		return false
	}

	// Check condition number
	if result.rcond < tolerance {
		return false // Matrix is ill-conditioned
	}

	// Check error bounds
	for err in result.ferr {
		if err > tolerance {
			return false
		}
	}

	return true
}

// Delete expert solver result
delete_expert_result :: proc(result: ^ExpertSolverResult($T)) {
	delete_matrix(&result.X)
	if result.ferr != nil do delete(result.ferr)
	if result.berr != nil do delete(result.berr)
	if result.S != nil do delete(result.S)
}
