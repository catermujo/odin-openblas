package openblas

import lapack "./f77"

// ===================================================================================
// POSITIVE DEFINITE EXTENDED EXPERT SOLVERS
// ===================================================================================

// Extended expert solver proc group
m_solve_positive_definite_extended_expert :: proc {
	m_solve_positive_definite_extended_expert_f32_c64,
	m_solve_positive_definite_extended_expert_f64_c128,
}

// ===================================================================================
// WORKSPACE QUERY FUNCTIONS
// ===================================================================================

// Query workspace for extended expert solver
query_workspace_extended_expert_positive_definite :: proc(
	$T: typeid,
	n: int,
	nrhs: int,
) -> (
	work_size: int,
	iwork_size: int,
	rwork_size: int,
	berr_size: int,
	err_bnds_norm_size: int,
	err_bnds_comp_size: int,
	S_size: int,
) {
	n_err_bnds := 3 // Extended version always has 3 error bound types

	when T == f32 || T == f64 {
		// Real types: sposvxx/dposvxx
		return 3 * n, n, 0, nrhs, nrhs * n_err_bnds, nrhs * n_err_bnds, n
	} else when T == complex64 || T == complex128 {
		// Complex types: cposvxx/zposvxx
		return 2 * n, 0, 3 * n, nrhs, nrhs * n_err_bnds, nrhs * n_err_bnds, n
	}
}

// ===================================================================================
// EXTENDED EXPERT SOLVER IMPLEMENTATION
// ===================================================================================

// Extended expert solver for f32/complex64 with pre-allocated arrays
// Provides extra precision error bounds and pivot growth factor
m_solve_positive_definite_extended_expert_f32_c64 :: proc(
	A: ^Matrix($T), // System matrix
	AF: ^Matrix(T), // Factorization workspace/input
	B: ^Matrix(T), // RHS matrix
	X: ^Matrix(T), // Solution matrix (output)
	S: []$f32, // Scale factors (pre-allocated)
	berr: []f32, // Backward error bounds (pre-allocated)
	err_bnds_norm: []f32, // Norm-wise error bounds (pre-allocated)
	err_bnds_comp: []f32, // Component-wise error bounds (pre-allocated)
	work: []T, // Workspace (pre-allocated)
	iwork: []Blas_Int = nil, // Integer workspace for real types
	rwork: []f32 = nil, // Real workspace for complex types
	fact := FactorizationOption.Equilibrate, // Factorization control
	uplo := MatrixRegion.Upper, // Upper or lower triangular
	equed_inout: ^EquilibrationState = nil, // Equilibration state (input/output)
	nparams: Blas_Int = 0, // Number of parameters
	params: []f32 = nil, // Refinement parameters
) -> (
	rcond: f32,
	rpvgrw: f32,
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
	n_err_bnds: Blas_Int = 3

	fact_c := factorization_to_cstring(fact)
	uplo_c := matrix_region_to_cstring(uplo)

	// Handle equilibration state
	equed_mode := EquilibrationState.None
	if equed_inout != nil {
		equed_mode = equed_inout^
	}
	equed_c := equilibration_state_to_cstring(equed_mode)

	// Validate workspace sizes
	assert(len(S) >= int(n), "Insufficient S space")
	assert(len(berr) >= int(nrhs), "Insufficient berr space")
	assert(len(err_bnds_norm) >= int(nrhs * n_err_bnds), "Insufficient err_bnds_norm space")
	assert(len(err_bnds_comp) >= int(nrhs * n_err_bnds), "Insufficient err_bnds_comp space")

	// Parameters pointer
	params_ptr: ^f32 = nil
	if len(params) > 0 {
		params_ptr = raw_data(params)
	}

	when T == f32 {
		assert(len(work) >= 3 * int(n), "Insufficient work space")
		assert(len(iwork) >= int(n), "Insufficient iwork space")

		lapack.sposvxx_(
			fact_c,
			uplo_c,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			raw_data(AF.data),
			&ldaf,
			equed_c,
			raw_data(S),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			&rcond,
			&rpvgrw,
			raw_data(berr),
			&n_err_bnds,
			raw_data(err_bnds_norm),
			raw_data(err_bnds_comp),
			&nparams,
			params_ptr,
			raw_data(work),
			raw_data(iwork),
			&info,
			len(fact_c),
			len(uplo_c),
			len(equed_c),
		)
	} else when T == complex64 {
		assert(len(work) >= 2 * int(n), "Insufficient work space")
		assert(len(rwork) >= 3 * int(n), "Insufficient rwork space")

		lapack.cposvxx_(
			fact_c,
			uplo_c,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			raw_data(AF.data),
			&ldaf,
			equed_c,
			raw_data(S),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			&rcond,
			&rpvgrw,
			raw_data(berr),
			&n_err_bnds,
			raw_data(err_bnds_norm),
			raw_data(err_bnds_comp),
			&nparams,
			params_ptr,
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
	return rcond, rpvgrw, info, ok
}

// Extended expert solver for f64/complex128 with pre-allocated arrays
// Provides extra precision error bounds and pivot growth factor
m_solve_positive_definite_extended_expert_f64_c128 :: proc(
	A: ^Matrix($T), // System matrix
	AF: ^Matrix(T), // Factorization workspace/input
	B: ^Matrix(T), // RHS matrix
	X: ^Matrix(T), // Solution matrix (output)
	S: []$f64, // Scale factors (pre-allocated)
	berr: []f64, // Backward error bounds (pre-allocated)
	err_bnds_norm: []f64, // Norm-wise error bounds (pre-allocated)
	err_bnds_comp: []f64, // Component-wise error bounds (pre-allocated)
	work: []T, // Workspace (pre-allocated)
	iwork: []Blas_Int = nil, // Integer workspace for real types
	rwork: []f64 = nil, // Real workspace for complex types
	fact := FactorizationOption.Equilibrate, // Factorization control
	uplo := MatrixRegion.Upper, // Upper or lower triangular
	equed_inout: ^EquilibrationState = nil, // Equilibration state (input/output)
	nparams: Blas_Int = 0, // Number of parameters
	params: []f64 = nil, // Refinement parameters
) -> (
	rcond: f64,
	rpvgrw: f64,
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
	n_err_bnds: Blas_Int = 3

	fact_c := factorization_to_cstring(fact)
	uplo_c := matrix_region_to_cstring(uplo)

	// Handle equilibration state
	equed_mode := EquilibrationState.None
	if equed_inout != nil {
		equed_mode = equed_inout^
	}
	equed_c := equilibration_state_to_cstring(equed_mode)

	// Validate workspace sizes
	assert(len(S) >= int(n), "Insufficient S space")
	assert(len(berr) >= int(nrhs), "Insufficient berr space")
	assert(len(err_bnds_norm) >= int(nrhs * n_err_bnds), "Insufficient err_bnds_norm space")
	assert(len(err_bnds_comp) >= int(nrhs * n_err_bnds), "Insufficient err_bnds_comp space")

	// Parameters pointer
	params_ptr: ^f64 = nil
	if len(params) > 0 {
		params_ptr = raw_data(params)
	}

	when T == f64 {
		assert(len(work) >= 3 * int(n), "Insufficient work space")
		assert(len(iwork) >= int(n), "Insufficient iwork space")

		lapack.dposvxx_(
			fact_c,
			uplo_c,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			raw_data(AF.data),
			&ldaf,
			equed_c,
			raw_data(S),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			&rcond,
			&rpvgrw,
			raw_data(berr),
			&n_err_bnds,
			raw_data(err_bnds_norm),
			raw_data(err_bnds_comp),
			&nparams,
			params_ptr,
			raw_data(work),
			raw_data(iwork),
			&info,
			len(fact_c),
			len(uplo_c),
			len(equed_c),
		)
	} else when T == complex128 {
		assert(len(work) >= 2 * int(n), "Insufficient work space")
		assert(len(rwork) >= 3 * int(n), "Insufficient rwork space")

		lapack.zposvxx_(
			fact_c,
			uplo_c,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			raw_data(AF.data),
			&ldaf,
			equed_c,
			raw_data(S),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			&rcond,
			&rpvgrw,
			raw_data(berr),
			&n_err_bnds,
			raw_data(err_bnds_norm),
			raw_data(err_bnds_comp),
			&nparams,
			params_ptr,
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
	return rcond, rpvgrw, info, ok
}
