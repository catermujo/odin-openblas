package openblas

import lapack "./f77"

// ===================================================================================
// POSITIVE DEFINITE ITERATIVE REFINEMENT
// ===================================================================================

// Standard iterative refinement proc group
m_refine_positive_definite :: proc {
	m_refine_positive_definite_f32_c64,
	m_refine_positive_definite_f64_c128,
}

// Extended iterative refinement proc group
m_refine_positive_definite_extended :: proc {
	m_refine_positive_definite_extended_f32_c64,
	m_refine_positive_definite_extended_f64_c128,
}

// Simple solver proc group
m_solve_positive_definite_simple :: proc {
	m_solve_positive_definite_simple_f32_c64,
	m_solve_positive_definite_simple_f64_c128,
}

// ===================================================================================
// WORKSPACE QUERY FUNCTIONS
// ===================================================================================

// Query workspace for standard iterative refinement
query_workspace_refine_positive_definite :: proc(
	$T: typeid,
	n: int,
) -> (
	work_size: int,
	iwork_size: int,
	rwork_size: int,
) {
	when is_float(T) {
		// sporfs/dporfs need work of size 3*n and iwork of size n
		return 3 * n, n, 0
	} else when is_complex(T) {
		// cporfs/zporfs need work of size 2*n and rwork of size n
		return 2 * n, 0, n
	}
}

// Query result array sizes for standard iterative refinement
query_result_sizes_refine_positive_definite :: proc(
	nrhs: int,
) -> (
	ferr_size: int,
	berr_size: int,
) {
	// One error bound per RHS column
	return nrhs, nrhs
}

// Query workspace for extended iterative refinement
query_workspace_refine_positive_definite_extended :: proc(
	$T: typeid,
	n: int,
) -> (
	work_size: int,
	iwork_size: int,
	rwork_size: int,
) {
	when is_float(T) {
		// sporfsx/dporfsx
		return 3 * n, n, 0
	} else when is_complex(T) {
		// cporfsx/zporfsx
		return 2 * n, 0, 3 * n
	}
}

// Query result array sizes for extended iterative refinement
query_result_sizes_refine_positive_definite_extended :: proc(
	nrhs: int,
) -> (
	berr_size: int,
	err_bnds_norm_size: int,
	err_bnds_comp_size: int,
) {
	n_err_bnds := 3 // Extended version has 3 error bound types
	return nrhs, nrhs * n_err_bnds, nrhs * n_err_bnds
}

// ===================================================================================
// STANDARD ITERATIVE REFINEMENT IMPLEMENTATION
// ===================================================================================

// Iterative refinement for positive definite system (f32/complex64)
m_refine_positive_definite_f32_c64 :: proc(
	A: ^Matrix($T), // Original matrix
	AF: ^Matrix(T), // Factored matrix from Cholesky
	B: ^Matrix(T), // Right-hand side
	X: ^Matrix(T), // Solution (input/output)
	ferr: []$f32, // Forward error bounds (pre-allocated)
	berr: []f32, // Backward error bounds (pre-allocated)
	work: []T, // Workspace (pre-allocated)
	iwork: []Blas_Int = nil, // Integer workspace for real types
	rwork: []f32 = nil, // Real workspace for complex types
	uplo := MatrixRegion.Upper, // Upper or lower triangular
) -> (
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
	uplo_c := matrix_region_to_cstring(uplo)

	// Validate workspace sizes
	assert(len(ferr) >= int(nrhs), "Insufficient ferr space")
	assert(len(berr) >= int(nrhs), "Insufficient berr space")

	when T == f32 {
		assert(len(work) >= 3 * int(n), "Insufficient work space")
		assert(len(iwork) >= int(n), "Insufficient iwork space")

		lapack.sporfs_(
			uplo_c,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			raw_data(AF.data),
			&ldaf,
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			raw_data(ferr),
			raw_data(berr),
			raw_data(work),
			raw_data(iwork),
			&info,
			len(uplo_c),
		)
	} else when T == complex64 {
		assert(len(work) >= 2 * int(n), "Insufficient work space")
		assert(len(rwork) >= int(n), "Insufficient rwork space")

		lapack.cporfs_(
			uplo_c,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			raw_data(AF.data),
			&ldaf,
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			raw_data(ferr),
			raw_data(berr),
			raw_data(work),
			raw_data(rwork),
			&info,
			len(uplo_c),
		)
	}

	ok = info == 0
	return info, ok
}

// Iterative refinement for positive definite system (f64/complex128)
m_refine_positive_definite_f64_c128 :: proc(
	A: ^Matrix($T), // Original matrix
	AF: ^Matrix(T), // Factored matrix from Cholesky
	B: ^Matrix(T), // Right-hand side
	X: ^Matrix(T), // Solution (input/output)
	ferr: []$f64, // Forward error bounds (pre-allocated)
	berr: []f64, // Backward error bounds (pre-allocated)
	work: []T, // Workspace (pre-allocated)
	iwork: []Blas_Int = nil, // Integer workspace for real types
	rwork: []f64 = nil, // Real workspace for complex types
	uplo := MatrixRegion.Upper, // Upper or lower triangular
) -> (
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
	uplo_c := matrix_region_to_cstring(uplo)

	// Validate workspace sizes
	assert(len(ferr) >= int(nrhs), "Insufficient ferr space")
	assert(len(berr) >= int(nrhs), "Insufficient berr space")

	when T == f64 {
		assert(len(work) >= 3 * int(n), "Insufficient work space")
		assert(len(iwork) >= int(n), "Insufficient iwork space")

		lapack.dporfs_(
			uplo_c,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			raw_data(AF.data),
			&ldaf,
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			raw_data(ferr),
			raw_data(berr),
			raw_data(work),
			raw_data(iwork),
			&info,
			len(uplo_c),
		)
	} else when T == complex128 {
		assert(len(work) >= 2 * int(n), "Insufficient work space")
		assert(len(rwork) >= int(n), "Insufficient rwork space")

		lapack.zporfs_(
			uplo_c,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			raw_data(AF.data),
			&ldaf,
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			raw_data(ferr),
			raw_data(berr),
			raw_data(work),
			raw_data(rwork),
			&info,
			len(uplo_c),
		)
	}

	ok = info == 0
	return info, ok
}

// ===================================================================================
// EXTENDED ITERATIVE REFINEMENT IMPLEMENTATION
// ===================================================================================

// Extended iterative refinement for positive definite system (f32/complex64)
m_refine_positive_definite_extended_f32_c64 :: proc(
	A: ^Matrix($T), // Original matrix
	AF: ^Matrix(T), // Factored matrix from Cholesky
	S: []$f32 = nil, // Scale factors (or nil if not equilibrated)
	B: ^Matrix(T), // Right-hand side
	X: ^Matrix(T), // Solution (input/output)
	berr: []f32, // Backward error bounds (pre-allocated)
	err_bnds_norm: []f32, // Norm-wise error bounds (pre-allocated)
	err_bnds_comp: []f32, // Component-wise error bounds (pre-allocated)
	work: []T, // Workspace (pre-allocated)
	iwork: []Blas_Int = nil, // Integer workspace for real types
	rwork: []f32 = nil, // Real workspace for complex types
	equed := EquilibrationState.None, // Whether equilibration was applied
	uplo := MatrixRegion.Upper, // Upper or lower triangular
	nparams: Blas_Int = 0, // Number of parameters
	params: []f32 = nil, // Refinement parameters
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
	n_err_bnds: Blas_Int = 3

	uplo_c := matrix_region_to_cstring(uplo)
	equed_c := equilibration_state_to_cstring(equed)

	// Validate workspace sizes
	assert(len(berr) >= int(nrhs), "Insufficient berr space")
	assert(len(err_bnds_norm) >= int(nrhs * n_err_bnds), "Insufficient err_bnds_norm space")
	assert(len(err_bnds_comp) >= int(nrhs * n_err_bnds), "Insufficient err_bnds_comp space")

	// Parameters pointer
	params_ptr: ^f32 = nil
	if len(params) > 0 {
		params_ptr = raw_data(params)
	}

	// Scale factors pointer
	s_ptr: ^f32 = nil
	if len(S) > 0 {
		s_ptr = raw_data(S)
	}

	when T == f32 {
		assert(len(work) >= 3 * int(n), "Insufficient work space")
		assert(len(iwork) >= int(n), "Insufficient iwork space")

		lapack.sporfsx_(
			uplo_c,
			equed_c,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			raw_data(AF.data),
			&ldaf,
			s_ptr,
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			&rcond,
			raw_data(berr),
			&n_err_bnds,
			raw_data(err_bnds_norm),
			raw_data(err_bnds_comp),
			&nparams,
			params_ptr,
			raw_data(work),
			raw_data(iwork),
			&info,
			len(uplo_c),
			len(equed_c),
		)
	} else when T == complex64 {
		assert(len(work) >= 2 * int(n), "Insufficient work space")
		assert(len(rwork) >= 3 * int(n), "Insufficient rwork space")

		lapack.cporfsx_(
			uplo_c,
			equed_c,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			raw_data(AF.data),
			&ldaf,
			s_ptr,
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			&rcond,
			raw_data(berr),
			&n_err_bnds,
			raw_data(err_bnds_norm),
			raw_data(err_bnds_comp),
			&nparams,
			params_ptr,
			raw_data(work),
			raw_data(rwork),
			&info,
			len(uplo_c),
			len(equed_c),
		)
	}

	ok = info == 0
	return rcond, info, ok
}

// Extended iterative refinement for positive definite system (f64/complex128)
m_refine_positive_definite_extended_f64_c128 :: proc(
	A: ^Matrix($T), // Original matrix
	AF: ^Matrix(T), // Factored matrix from Cholesky
	S: []$f64 = nil, // Scale factors (or nil if not equilibrated)
	B: ^Matrix(T), // Right-hand side
	X: ^Matrix(T), // Solution (input/output)
	berr: []f64, // Backward error bounds (pre-allocated)
	err_bnds_norm: []f64, // Norm-wise error bounds (pre-allocated)
	err_bnds_comp: []f64, // Component-wise error bounds (pre-allocated)
	work: []T, // Workspace (pre-allocated)
	iwork: []Blas_Int = nil, // Integer workspace for real types
	rwork: []f64 = nil, // Real workspace for complex types
	equed := EquilibrationState.None, // Whether equilibration was applied
	uplo := MatrixRegion.Upper, // Upper or lower triangular
	nparams: Blas_Int = 0, // Number of parameters
	params: []f64 = nil, // Refinement parameters
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
	n_err_bnds: Blas_Int = 3

	uplo_c := matrix_region_to_cstring(uplo)
	equed_c := equilibration_state_to_cstring(equed)

	// Validate workspace sizes
	assert(len(berr) >= int(nrhs), "Insufficient berr space")
	assert(len(err_bnds_norm) >= int(nrhs * n_err_bnds), "Insufficient err_bnds_norm space")
	assert(len(err_bnds_comp) >= int(nrhs * n_err_bnds), "Insufficient err_bnds_comp space")

	// Parameters pointer
	params_ptr: ^f64 = nil
	if len(params) > 0 {
		params_ptr = raw_data(params)
	}

	// Scale factors pointer
	s_ptr: ^f64 = nil
	if len(S) > 0 {
		s_ptr = raw_data(S)
	}

	when T == f64 {
		assert(len(work) >= 3 * int(n), "Insufficient work space")
		assert(len(iwork) >= int(n), "Insufficient iwork space")

		lapack.dporfsx_(
			uplo_c,
			equed_c,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			raw_data(AF.data),
			&ldaf,
			s_ptr,
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			&rcond,
			raw_data(berr),
			&n_err_bnds,
			raw_data(err_bnds_norm),
			raw_data(err_bnds_comp),
			&nparams,
			params_ptr,
			raw_data(work),
			raw_data(iwork),
			&info,
			len(uplo_c),
			len(equed_c),
		)
	} else when T == complex128 {
		assert(len(work) >= 2 * int(n), "Insufficient work space")
		assert(len(rwork) >= 3 * int(n), "Insufficient rwork space")

		lapack.zporfsx_(
			uplo_c,
			equed_c,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			raw_data(AF.data),
			&ldaf,
			s_ptr,
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			&rcond,
			raw_data(berr),
			&n_err_bnds,
			raw_data(err_bnds_norm),
			raw_data(err_bnds_comp),
			&nparams,
			params_ptr,
			raw_data(work),
			raw_data(rwork),
			&info,
			len(uplo_c),
			len(equed_c),
		)
	}

	ok = info == 0
	return rcond, info, ok
}

// ===================================================================================
// SIMPLE SOLVER IMPLEMENTATION
// ===================================================================================

// Simple solver for positive definite system (f32/complex64)
// Solves A*X = B using Cholesky factorization
m_solve_positive_definite_simple_f32_c64 :: proc(
	A: ^Matrix($T), // System matrix (destroyed on output)
	B: ^Matrix(T), // RHS matrix (replaced with solution)
	uplo := MatrixRegion.Upper, // Upper or lower triangular
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == complex64 {
	// Validate inputs
	assert(A.rows == A.cols, "Matrix must be square")
	assert(B.rows == A.rows, "RHS dimension mismatch")
	assert(uplo == .Upper || uplo == .Lower, "uplo must be Upper or Lower")

	n := Blas_Int(A.rows)
	nrhs := Blas_Int(B.cols)
	lda := Blas_Int(A.stride)
	ldb := Blas_Int(B.stride)
	uplo_c := matrix_region_to_cstring(uplo)

	when T == f32 {
		lapack.sposv_(
			uplo_c,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			&info,
			len(uplo_c),
		)
	} else when T == complex64 {
		lapack.cposv_(
			uplo_c,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			&info,
			len(uplo_c),
		)
	}

	ok = info == 0
	return info, ok
}

// Simple solver for positive definite system (f64/complex128)
// Solves A*X = B using Cholesky factorization
m_solve_positive_definite_simple_f64_c128 :: proc(
	A: ^Matrix($T), // System matrix (destroyed on output)
	B: ^Matrix(T), // RHS matrix (replaced with solution)
	uplo := MatrixRegion.Upper, // Upper or lower triangular
) -> (
	info: Info,
	ok: bool,
) where T == f64 || T == complex128 {
	// Validate inputs
	assert(A.rows == A.cols, "Matrix must be square")
	assert(B.rows == A.rows, "RHS dimension mismatch")
	assert(uplo == .Upper || uplo == .Lower, "uplo must be Upper or Lower")

	n := Blas_Int(A.rows)
	nrhs := Blas_Int(B.cols)
	lda := Blas_Int(A.stride)
	ldb := Blas_Int(B.stride)
	uplo_c := matrix_region_to_cstring(uplo)

	when T == f64 {
		lapack.dposv_(
			uplo_c,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			&info,
			len(uplo_c),
		)
	} else when T == complex128 {
		lapack.zposv_(
			uplo_c,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			&info,
			len(uplo_c),
		)
	}

	ok = info == 0
	return info, ok
}
