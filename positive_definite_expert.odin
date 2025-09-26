package openblas

import lapack "./f77"

// ===================================================================================
// POSITIVE DEFINITE MIXED-PRECISION AND EXPERT SOLVERS
// ===================================================================================

// Mixed-precision solver proc group
// LAPACK only provides dsposv (f64 with f32) and zcposv (c128 with c64)
m_solve_positive_definite_mixed :: proc {
	m_solve_positive_definite_mixed_f64_c128,
}

// Expert solver proc group
m_solve_positive_definite_expert :: proc {
	m_solve_positive_definite_expert_f32_c64,
	m_solve_positive_definite_expert_f64_c128,
}


// ===================================================================================
// WORKSPACE QUERY FUNCTIONS
// ===================================================================================

// Query workspace for mixed-precision solver
query_workspace_mixed_precision :: proc(
	n: int,
	nrhs: int,
) -> (
	work_size: int,
	swork_size: int,
	rwork_size: int,
) {
	// Both dsposv and zcposv require:
	// work: n*(n+nrhs) for high precision type
	// swork: n*(n+nrhs) for low precision type
	// rwork: n for complex types only
	return n * (n + nrhs), n * (n + nrhs), n
}

// Query workspace for expert solver
query_workspace_expert_positive_definite :: proc(
	n: int,
	nrhs: int,
) -> (
	work_size: int,
	iwork_size: int,
	rwork_size: int,
	ferr_size: int,
	berr_size: int,
	S_size: int,
) {
	// sposvx/dposvx require: work=3*n, iwork=n
	// cposvx/zposvx require: work=2*n, rwork=n (no iwork)
	// ferr and berr are size nrhs
	// S (scale factors) is size n
	return max(3 * n, 2 * n), n, n, nrhs, nrhs, n
}

// ===================================================================================
// MIXED-PRECISION ITERATIVE REFINEMENT IMPLEMENTATION
// ===================================================================================

// Mixed-precision solver for f64/c128 with f32/c64 acceleration
// Uses lower precision for factorization, refines to higher precision
m_solve_positive_definite_mixed_f64_c128 :: proc(
	A: ^Matrix($T), // System matrix (preserved if not equilibrated)
	B: ^Matrix(T), // RHS matrix
	X: ^Matrix(T), // Solution matrix (output)
	work: []T, // Workspace (pre-allocated)
	swork: []$T_low, // Single precision workspace (pre-allocated)
	rwork: []f64 = nil, // Real workspace for complex types (nil for real)
	uplo := MatrixRegion.Upper, // Upper or lower triangular
) -> (
	iterations: int,
	converged: bool,
	info: Info,
	ok: bool,
) where (T == f64 && T_low == f32) || (T == complex128 && T_low == complex64) {
	// Validate inputs
	assert(A.rows == A.cols, "Matrix must be square")
	assert(B.rows == A.rows && X.rows == A.rows, "Dimension mismatch")
	assert(B.cols == X.cols, "RHS and solution must have same number of columns")
	assert(uplo == .Upper || uplo == .Lower, "uplo must be Upper or Lower")

	n := Blas_Int(A.rows)
	nrhs := Blas_Int(B.cols)
	lda := Blas_Int(A.stride)
	ldb := Blas_Int(B.stride)
	ldx := Blas_Int(X.stride)
	uplo_c := matrix_region_to_cstring(uplo)

	// Validate workspace
	assert(len(work) >= int(n * (n + nrhs)), "Insufficient work space")
	assert(len(swork) >= int(n * (n + nrhs)), "Insufficient swork space")

	iter: Blas_Int

	when T == f64 {
		lapack.dsposv_(
			uplo_c,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			raw_data(work),
			raw_data(swork),
			&iter,
			&info,
			len(uplo_c),
		)
	} else when T == complex128 {
		assert(len(rwork) >= int(n), "Insufficient rwork space")
		lapack.zcposv_(
			uplo_c,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			raw_data(work),
			raw_data(swork),
			raw_data(rwork),
			&iter,
			&info,
			len(uplo_c),
		)
	}

	// Return results
	iterations = int(abs(iter))
	converged = iter >= 0 // Negative means didn't converge
	ok = info == 0

	return iterations, converged, info, ok
}

// ===================================================================================
// EXPERT SOLVER IMPLEMENTATION
// ===================================================================================

// Expert solver for f32/c64 with pre-allocated arrays
m_solve_positive_definite_expert_f32_c64 :: proc(
	A: ^Matrix($T), // System matrix
	AF: ^Matrix(T), // Factorization workspace/input
	B: ^Matrix(T), // RHS matrix
	X: ^Matrix(T), // Solution matrix (output)
	S: []$f32, // Scale factors (pre-allocated)
	ferr: []f32, // Forward error bounds (pre-allocated)
	berr: []f32, // Backward error bounds (pre-allocated)
	work: []T, // Workspace (pre-allocated)
	iwork: []Blas_Int = nil, // Integer workspace for real types
	rwork: []f32 = nil, // Real workspace for complex types
	fact := FactorizationOption.Equilibrate, // Factorization control
	uplo := MatrixRegion.Upper, // Upper or lower triangular
	equed_inout: ^EquilibrationState = nil, // Equilibration state (input/output)
) -> (
	rcond: f32,
	info: Info,
	ok: bool,
) where T == f32 || T == complex64 {
	// Validate inputs
	assert(A.rows == A.cols && AF.rows == AF.cols, "Matrices must be square")
	assert(A.rows == AF.rows, "A and AF must have same dimensions")
	assert(B.rows == A.rows && X.rows == A.rows, "Dimension mismatch")
	assert(B.cols == X.cols, "RHS and solution must have same number of columns")
	assert(uplo == .Upper || uplo == .Lower, "uplo must be Upper or Lower")

	n := Blas_Int(A.rows)
	nrhs := Blas_Int(B.cols)
	lda := Blas_Int(A.stride)
	ldaf := Blas_Int(AF.stride)
	ldb := Blas_Int(B.stride)
	ldx := Blas_Int(X.stride)

	fact_c := factorization_request_to_cstring(fact)
	uplo_c := matrix_region_to_cstring(uplo)

	// Handle equilibration state
	equed_mode := EquilibrationState.None
	if equed_inout != nil {
		equed_mode = equed_inout^
	}
	equed_c := factorization_to_char(equed_mode)

	// Validate workspace sizes
	assert(len(S) >= int(n), "Insufficient S space")
	assert(len(ferr) >= int(nrhs), "Insufficient ferr space")
	assert(len(berr) >= int(nrhs), "Insufficient berr space")

	when T == f32 {
		assert(len(work) >= 3 * int(n), "Insufficient work space")
		assert(len(iwork) >= int(n), "Insufficient iwork space")

		lapack.sposvx_(
			fact_c,
			uplo_c,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			raw_data(AF.data),
			&ldaf,
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
			&info,
			len(fact_c),
			len(uplo_c),
			len(equed_c),
		)
	} else when T == complex64 {
		assert(len(work) >= 2 * int(n), "Insufficient work space")
		assert(len(rwork) >= int(n), "Insufficient rwork space")

		lapack.cposvx_(
			fact_c,
			uplo_c,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			raw_data(AF.data),
			&ldaf,
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
			&info,
			len(fact_c),
			len(uplo_c),
			len(equed_c),
		)
	}

	// Update equilibration state if provided
	if equed_inout != nil {
		// LAPACK may modify equed
		if equed_c[0] == 'Y' {
			equed_inout^ = .Applied
		} else {
			equed_inout^ = .None
		}
	}

	ok = info == 0
	return rcond, info, ok
}

// Expert solver for f64/c128 with pre-allocated arrays
m_solve_positive_definite_expert_f64_c128 :: proc(
	A: ^Matrix($T), // System matrix
	AF: ^Matrix(T), // Factorization workspace/input
	B: ^Matrix(T), // RHS matrix
	X: ^Matrix(T), // Solution matrix (output)
	S: []$f64, // Scale factors (pre-allocated)
	ferr: []f64, // Forward error bounds (pre-allocated)
	berr: []f64, // Backward error bounds (pre-allocated)
	work: []T, // Workspace (pre-allocated)
	iwork: []Blas_Int = nil, // Integer workspace for real types
	rwork: []f64 = nil, // Real workspace for complex types
	fact := FactorizationOption.Equilibrate, // Factorization control
	uplo := MatrixRegion.Upper, // Upper or lower triangular
	equed_inout: ^EquilibrationState = nil, // Equilibration state (input/output)
) -> (
	rcond: f64,
	info: Info,
	ok: bool,
) where T == f64 || T == complex128 {
	// Validate inputs
	assert(A.rows == A.cols && AF.rows == AF.cols, "Matrices must be square")
	assert(A.rows == AF.rows, "A and AF must have same dimensions")
	assert(B.rows == A.rows && X.rows == A.rows, "Dimension mismatch")
	assert(B.cols == X.cols, "RHS and solution must have same number of columns")
	assert(uplo == .Upper || uplo == .Lower, "uplo must be Upper or Lower")

	n := Blas_Int(A.rows)
	nrhs := Blas_Int(B.cols)
	lda := Blas_Int(A.stride)
	ldaf := Blas_Int(AF.stride)
	ldb := Blas_Int(B.stride)
	ldx := Blas_Int(X.stride)

	fact_c := factorization_request_to_cstring(fact)
	uplo_c := matrix_region_to_cstring(uplo)

	// Handle equilibration state
	equed_mode := EquilibrationState.None
	if equed_inout != nil {
		equed_mode = equed_inout^
	}
	equed_c := factorization_to_char(equed_mode)

	// Validate workspace sizes
	assert(len(S) >= int(n), "Insufficient S space")
	assert(len(ferr) >= int(nrhs), "Insufficient ferr space")
	assert(len(berr) >= int(nrhs), "Insufficient berr space")

	when T == f64 {
		assert(len(work) >= 3 * int(n), "Insufficient work space")
		assert(len(iwork) >= int(n), "Insufficient iwork space")

		lapack.dposvx_(
			fact_c,
			uplo_c,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			raw_data(AF.data),
			&ldaf,
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
			&info,
			len(fact_c),
			len(uplo_c),
			len(equed_c),
		)
	} else when T == complex128 {
		assert(len(work) >= 2 * int(n), "Insufficient work space")
		assert(len(rwork) >= int(n), "Insufficient rwork space")

		lapack.zposvx_(
			fact_c,
			uplo_c,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			raw_data(AF.data),
			&ldaf,
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
			&info,
			len(fact_c),
			len(uplo_c),
			len(equed_c),
		)
	}

	// Update equilibration state if provided
	if equed_inout != nil {
		// LAPACK may modify equed
		if equed_c[0] == 'Y' {
			equed_inout^ = .Applied
		} else {
			equed_inout^ = .None
		}
	}

	ok = info == 0
	return rcond, info, ok
}
