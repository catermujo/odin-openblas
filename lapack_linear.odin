package openblas

import lapack "./f77"
import "base:builtin"
import "base:intrinsics"

// ===================================================================================
// LINEAR SYSTEM SOLVERS
// Direct solvers for systems Ax = B
// ===================================================================================

m_refine_solution :: proc {
	m_refine_solution_real,
	m_refine_solution_c64,
	m_refine_solution_c128,
}

m_refine_solution_extended :: proc {
	m_refine_solution_extended_real,
	m_refine_solution_extended_c64,
	m_refine_solution_extended_c128,
}

m_solve_mixed :: proc {
	m_solve_mixed_d32,
	m_solve_mixed_z64,
}

m_solve_expert :: proc {
	m_solve_expert_real,
	m_solve_expert_c64,
	m_solve_expert_c128,
}

m_solve_expert_extra :: proc {
	m_solve_expert_extra_real,
	m_solve_expert_extra_c64,
	m_solve_expert_extra_c128,
}

// m_lu_factor_recursive

// m_inverse

// m_inverse_direct

// m_solve_least_squares

// ===================================================================================
// ITERATIVE REFINEMENT
// ===================================================================================

m_refine_solution_real :: proc(
	A: ^Matrix($T), // Original matrix A
	AF: ^Matrix(T), // LU factorization from getrf
	ipiv: []Blas_Int, // Pivot indices from getrf
	B: ^Matrix(T), // Right-hand side
	X: ^Matrix(T), // Solution (input/output)
	transpose: TransposeMode = .None,
	allocator := context.allocator,
) -> (
	ferr: []T,
	berr: []T,
	info: Info, // Forward error bounds// Backward error bounds
) where is_float(T) {

	n := A.rows
	nrhs := B.cols
	lda := A.ld
	ldaf := AF.ld
	ldb := B.ld
	ldx := X.ld

	trans_c: cstring = transpose_to_cstring(transpose)

	// Allocate error arrays
	ferr = make([]T, nrhs)
	berr = make([]T, nrhs)

	// Allocate workspace
	work := make([]T, 3 * n)
	iwork := make([]Blas_Int, n)
	defer delete(work)
	defer delete(iwork)

	when T == f32 {
		lapack.sgerfs_(
			trans_c,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			raw_data(AF.data),
			&ldaf,
			raw_data(ipiv),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			raw_data(ferr),
			raw_data(berr),
			raw_data(work),
			raw_data(iwork),
			&info,
			1,
		)
	} else {
		lapack.dgerfs_(
			trans_c,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			raw_data(AF.data),
			&ldaf,
			raw_data(ipiv),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			raw_data(ferr),
			raw_data(berr),
			raw_data(work),
			raw_data(iwork),
			&info,
			1,
		)
	}

	return ferr, berr, info
}

m_refine_solution_c64 :: proc(
	A: ^Matrix(complex64),
	AF: ^Matrix(complex64),
	ipiv: []Blas_Int,
	B: ^Matrix(complex64),
	X: ^Matrix(complex64),
	transpose: TransposeMode = .None,
	allocator := context.allocator,
) -> (
	ferr: []f32,
	berr: []f32,
	info: Info,
) {
	n := A.rows
	nrhs := B.cols
	lda := A.ld
	ldaf := AF.ld
	ldb := B.ld
	ldx := X.ld

	trans_c := transpose_to_cstring(transpose)

	// Allocate error arrays
	ferr = make([]f32, nrhs)
	berr = make([]f32, nrhs)

	// Allocate workspace
	work := make([]complex64, 2 * n)
	rwork := make([]f32, n)
	defer delete(work)
	defer delete(rwork)

	lapack.cgerfs_(
		trans_c,
		&n,
		&nrhs,
		raw_data(A.data),
		&lda,
		raw_data(AF.data),
		&ldaf,
		raw_data(ipiv),
		raw_data(B.data),
		&ldb,
		raw_data(X.data),
		&ldx,
		raw_data(ferr),
		raw_data(berr),
		raw_data(work),
		raw_data(rwork),
		&info,
		1,
	)

	return ferr, berr, info
}

m_refine_solution_c128 :: proc(
	A: ^Matrix(complex128),
	AF: ^Matrix(complex128),
	ipiv: []Blas_Int,
	B: ^Matrix(complex128),
	X: ^Matrix(complex128),
	transpose: TransposeMode = .None,
	allocator := context.allocator,
) -> (
	ferr: []f64,
	berr: []f64,
	info: Info,
) {
	n := A.rows
	nrhs := B.cols
	lda := A.ld
	ldaf := AF.ld
	ldb := B.ld
	ldx := X.ld

	trans_c := transpose_to_cstring(transpose)

	// Allocate error arrays
	ferr = make([]f64, nrhs)
	berr = make([]f64, nrhs)

	// Allocate workspace
	work := make([]complex128, 2 * n)
	rwork := make([]f64, n)
	defer delete(work)
	defer delete(rwork)

	lapack.zgerfs_(
		trans_c,
		&n,
		&nrhs,
		raw_data(A.data),
		&lda,
		raw_data(AF.data),
		&ldaf,
		raw_data(ipiv),
		raw_data(B.data),
		&ldb,
		raw_data(X.data),
		&ldx,
		raw_data(ferr),
		raw_data(berr),
		raw_data(work),
		raw_data(rwork),
		&info,
		1,
	)

	return ferr, berr, info
}

// Extended iterative refinement with comprehensive error bounds
// Provides componentwise and normwise error bounds with equilibration support
m_refine_solution_extended_real :: proc(
	A: ^Matrix($T), // Original matrix A
	AF: ^Matrix(T), // LU factorization from getrf
	ipiv: []Blas_Int, // Pivot indices from getrf
	R: []T, // Row scale factors (can be nil if not equilibrated)
	C: []T, // Column scale factors (can be nil if not equilibrated)
	B: ^Matrix(T), // Right-hand side
	X: ^Matrix(T), // Solution (input/output)
	transpose: TransposeMode = .None,
	equilibrated: Equilibration = .None,
	n_err_bnds: Blas_Int = 3, // Number of error bounds to compute
	allocator := context.allocator,
) -> (
	rcond: T,
	berr: []T,
	err_bnds_norm: []T,
	err_bnds_comp: []T,
	info: Info, // Reciprocal condition number// Backward error lapack.bounds// Normwise error bounds// Componentwise error bounds
) where is_float(T) {
	n := A.rows
	nrhs := B.cols
	lda := A.ld
	ldaf := AF.ld
	ldb := B.ld
	ldx := X.ld

	trans_c: cstring = transpose_to_cstring(transpose)
	equed_c: cstring = equilibration_to_cstring(equilibrated)

	// Allocate error arrays
	berr = make([]T, nrhs)
	err_bnds_norm = make([]T, nrhs * n_err_bnds)
	err_bnds_comp = make([]T, nrhs * n_err_bnds)

	// Default parameters (0 means use defaults)
	nparams: Blas_Int = 0
	params: T = 0
	n_err_bnds_copy := n_err_bnds

	// Allocate workspace
	work := make([]T, 4 * n)
	iwork := make([]Blas_Int, n)
	defer delete(work)
	defer delete(iwork)

	// Use provided scale factors or allocate dummy ones if not equilibrated
	r_ptr := R
	c_ptr := C
	dummy_scale: []T
	defer if dummy_scale != nil {defer delete(dummy_scale)}
	if equilibrated == .None {
		dummy_scale = make([]T, n)
		for i in 0 ..< n {dummy_scale[i] = 1}
		r_ptr = dummy_scale
		c_ptr = dummy_scale
	}

	when T == f32 {
		lapack.sgerfsx_(
			trans_c,
			equed_c,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			raw_data(AF.data),
			&ldaf,
			raw_data(ipiv),
			raw_data(r_ptr),
			raw_data(c_ptr),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			&rcond,
			raw_data(berr),
			&n_err_bnds_copy,
			raw_data(err_bnds_norm),
			raw_data(err_bnds_comp),
			&nparams,
			&params,
			raw_data(work),
			raw_data(iwork),
			&info,
			1,
			1,
		)
	} else {
		lapack.dgerfsx_(
			trans_c,
			equed_c,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			raw_data(AF.data),
			&ldaf,
			raw_data(ipiv),
			raw_data(r_ptr),
			raw_data(c_ptr),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			&rcond,
			raw_data(berr),
			&n_err_bnds_copy,
			raw_data(err_bnds_norm),
			raw_data(err_bnds_comp),
			&nparams,
			&params,
			raw_data(work),
			raw_data(iwork),
			&info,
			1,
			1,
		)
	}

	return rcond, berr, err_bnds_norm, err_bnds_comp, info
}

m_refine_solution_extended_c64 :: proc(
	A: ^Matrix(complex64),
	AF: ^Matrix(complex64),
	ipiv: []Blas_Int,
	R: []f32,
	C: []f32,
	B: ^Matrix(complex64),
	X: ^Matrix(complex64),
	transpose: TransposeMode = .None,
	equilibrated: Equilibration = .None,
	n_err_bnds: Blas_Int = 3,
	allocator := context.allocator,
) -> (
	rcond: f32,
	berr: []f32,
	err_bnds_norm: []f32,
	err_bnds_comp: []f32,
	info: Info,
) {
	n := A.rows
	nrhs := B.cols
	lda := A.ld
	ldaf := AF.ld
	ldb := B.ld
	ldx := X.ld

	trans_c := transpose_to_cstring(transpose)
	equed_c: cstring = equilibration_to_cstring(equilibrated)

	// Allocate error arrays
	berr = make([]f32, nrhs)
	err_bnds_norm = make([]f32, nrhs * n_err_bnds)
	err_bnds_comp = make([]f32, nrhs * n_err_bnds)

	// Default parameters
	nparams: Blas_Int = 0
	params: f32 = 0
	n_err_bnds_copy := n_err_bnds

	// Allocate workspace
	work := make([]complex64, 2 * n)
	rwork := make([]f32, 3 * n)
	defer delete(work)
	defer delete(rwork)

	// Handle scale factors
	r_ptr := R
	c_ptr := C
	dummy_scale: []f32
	defer if dummy_scale != nil {defer delete(dummy_scale)}
	if equilibrated == .None {
		dummy_scale = make([]f32, n)
		for i in 0 ..< n {dummy_scale[i] = 1}
		r_ptr = dummy_scale
		c_ptr = dummy_scale
	}

	lapack.cgerfsx_(
		trans_c,
		equed_c,
		&n,
		&nrhs,
		raw_data(A.data),
		&lda,
		raw_data(AF.data),
		&ldaf,
		raw_data(ipiv),
		raw_data(r_ptr),
		raw_data(c_ptr),
		raw_data(B.data),
		&ldb,
		raw_data(X.data),
		&ldx,
		&rcond,
		raw_data(berr),
		&n_err_bnds_copy,
		raw_data(err_bnds_norm),
		raw_data(err_bnds_comp),
		&nparams,
		&params,
		raw_data(work),
		raw_data(rwork),
		&info,
		1,
		1,
	)

	return rcond, berr, err_bnds_norm, err_bnds_comp, info
}

m_refine_solution_extended_c128 :: proc(
	A: ^Matrix(complex128),
	AF: ^Matrix(complex128),
	ipiv: []Blas_Int,
	R: []f64,
	C: []f64,
	B: ^Matrix(complex128),
	X: ^Matrix(complex128),
	transpose: TransposeMode = .None,
	equilibrated: Equilibration = .None,
	n_err_bnds: Blas_Int = 3,
	allocator := context.allocator,
) -> (
	rcond: f64,
	berr: []f64,
	err_bnds_norm: []f64,
	err_bnds_comp: []f64,
	info: Info,
) {
	n := A.rows
	nrhs := B.cols
	lda := A.ld
	ldaf := AF.ld
	ldb := B.ld
	ldx := X.ld

	trans_c := transpose_to_cstring(transpose)
	equed_c: cstring = equilibration_to_cstring(equilibrated)

	// Allocate error arrays
	berr = make([]f64, nrhs)
	err_bnds_norm = make([]f64, nrhs * n_err_bnds)
	err_bnds_comp = make([]f64, nrhs * n_err_bnds)

	// Default parameters
	nparams: Blas_Int = 0
	params: f64 = 0
	n_err_bnds_copy := n_err_bnds

	// Allocate workspace
	work := make([]complex128, 2 * n)
	rwork := make([]f64, 3 * n)
	defer delete(work)
	defer delete(rwork)

	// Handle scale factors
	r_ptr := R
	c_ptr := C
	dummy_scale: []f64
	defer if dummy_scale != nil {defer delete(dummy_scale)}
	if equilibrated == .None {
		dummy_scale = make([]f64, n)
		for i in 0 ..< n {dummy_scale[i] = 1}
		r_ptr = dummy_scale
		c_ptr = dummy_scale
	}

	lapack.zgerfsx_(
		trans_c,
		equed_c,
		&n,
		&nrhs,
		raw_data(A.data),
		&lda,
		raw_data(AF.data),
		&ldaf,
		raw_data(ipiv),
		raw_data(r_ptr),
		raw_data(c_ptr),
		raw_data(B.data),
		&ldb,
		raw_data(X.data),
		&ldx,
		&rcond,
		raw_data(berr),
		&n_err_bnds_copy,
		raw_data(err_bnds_norm),
		raw_data(err_bnds_comp),
		&nparams,
		&params,
		raw_data(work),
		raw_data(rwork),
		&info,
		1,
		1,
	)

	return rcond, berr, err_bnds_norm, err_bnds_comp, info
}

// ===================================================================================
// MIXED-PRECISION ITERATIVE REFINEMENT SOLVERS
// ===================================================================================
// Solve linear system using mixed precision iterative refinement
// Uses lower precision for factorization, higher precision for refinement

// Double precision solution with single precision factorization
m_solve_mixed_d32 :: proc(
	A: ^Matrix(f64), // Coefficient matrix (preserved)
	B: ^Matrix(f64), // Right-hand side (preserved)
	allocator := context.allocator,
) -> (
	X: Matrix(f64),
	iter: Blas_Int,
	info: Info, // Solution matrix// Number of refinement iterations
) {
	n := A.rows
	nrhs := B.cols
	lda := A.ld
	ldb := B.ld

	// Allocate solution matrix
	x_data := make([]f64, n * nrhs)
	X = Matrix(f64) {
		data   = x_data,
		rows   = n,
		cols   = nrhs,
		ld     = n,
		format = .General,
	}
	ldx := X.ld

	// Allocate pivot array
	ipiv := make([]Blas_Int, n)
	defer delete(ipiv)

	// Allocate double precision workspace
	work := make([]f64, n * nrhs)
	defer delete(work)

	// Allocate single precision workspace
	swork_size := n * (n + nrhs)
	swork := make([]f32, swork_size)
	defer delete(swork)

	// Solve system with iterative refinement
	lapack.dsgesv_(
		&n,
		&nrhs,
		raw_data(A.data),
		&lda,
		raw_data(ipiv),
		raw_data(B.data),
		&ldb,
		raw_data(X.data),
		&ldx,
		raw_data(work),
		raw_data(swork),
		&iter,
		&info,
	)

	return X, iter, info
}

// Double complex precision solution with single complex precision factorization
m_solve_mixed_z64 :: proc(
	A: ^Matrix(complex128),
	B: ^Matrix(complex128),
	allocator := context.allocator,
) -> (
	X: Matrix(complex128),
	iter: Blas_Int,
	info: Info,
) {
	n := A.rows
	nrhs := B.cols
	lda := A.ld
	ldb := B.ld

	// Allocate solution matrix
	x_data := make([]complex128, n * nrhs)
	X = Matrix(complex128) {
		data   = x_data,
		rows   = n,
		cols   = nrhs,
		ld     = n,
		format = .General,
	}
	ldx := X.ld

	// Allocate pivot array
	ipiv := make([]Blas_Int, n)
	defer delete(ipiv)

	// Allocate double complex precision workspace
	work := make([]complex128, n * nrhs)
	defer delete(work)

	// Allocate single complex precision workspace
	swork_size := n * (n + nrhs)
	swork := make([]complex64, swork_size)
	defer delete(swork)

	// Allocate real workspace
	rwork := make([]f64, n)
	defer delete(rwork)

	// Solve system with iterative refinement
	lapack.zcgesv_(
		&n,
		&nrhs,
		raw_data(A.data),
		&lda,
		raw_data(ipiv),
		raw_data(B.data),
		&ldb,
		raw_data(X.data),
		&ldx,
		raw_data(work),
		raw_data(swork),
		raw_data(rwork),
		&iter,
		&info,
	)

	return X, iter, info
}

// ===================================================================================
// EXPERT LINEAR SOLVERS WITH EQUILIBRATION
// ===================================================================================
// Solve linear system with expert options and error bounds
// Includes equilibration, condition estimation, and iterative refinement

m_solve_expert_real :: proc(
	A: ^Matrix($T),
	B: ^Matrix(T),
	do_equilibrate: bool = true, // Apply equilibration for better conditioning
	transpose: TransposeMode = .None, // Solve A^T*X = B
	compute_rcond: bool = true, // Compute reciprocal condition number
	allocator := context.allocator,
) -> (
	X: Matrix(T),
	AF: Matrix(T),
	ipiv: []Blas_Int,
	R: []T,
	C: []T,
	rcond: T,
	ferr: []T,
	berr: []T,
	equed: Equilibration,
	info: Info, // Solution matrix// LU factorization of equilibrated A// Pivot indices// Row scale factors// Column scale factors// Reciprocal condition number// Forward error bounds for each lapack.solution// Backward error bounds for each solution// Equilibration applied
) where is_float(T) {
	n := A.rows
	nrhs := B.cols
	lda := A.ld
	ldb := B.ld

	// Allocate output matrices
	X = make_matrix(T, int(n), int(nrhs))
	ldx := X.ld

	AF = make_matrix(T, int(n), int(n))
	ldaf := AF.ld

	ipiv = make([]Blas_Int, n)
	R = make([]T, n)
	C = make([]T, n)
	ferr = make([]T, nrhs)
	berr = make([]T, nrhs)

	// Set options
	fact_c: cstring = do_equilibrate ? "E" : "N" // E=equilibrate, N=no equilibration
	trans_c: cstring = transpose_to_cstring(transpose)

	// Output parameter for equilibration type
	equed_c: [2]byte = {'N', 0}

	// Allocate workspace
	work := make([]T, 4 * n)
	iwork := make([]Blas_Int, n)
	defer delete(work)
	defer delete(iwork)

	when T == f32 {
		lapack.sgesvx_(
			fact_c,
			trans_c,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			raw_data(AF.data),
			&ldaf,
			raw_data(ipiv),
			cstring(&equed_c[0]),
			raw_data(R),
			raw_data(C),
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
			1,
			1,
			1,
		)
	} else {
		lapack.dgesvx_(
			fact_c,
			trans_c,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			raw_data(AF.data),
			&ldaf,
			raw_data(ipiv),
			cstring(&equed_c[0]),
			raw_data(R),
			raw_data(C),
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
			1,
			1,
			1,
		)
	}

	// Convert equilibration flag to enum
	equed = equilibration_from_char(equed_c[0])

	return X, AF, ipiv, R, C, rcond, ferr, berr, equed, info
}

m_solve_expert_c64 :: proc(
	A: ^Matrix(complex64),
	B: ^Matrix(complex64),
	do_equilibrate: bool = true,
	transpose: TransposeMode = .None,
	compute_rcond: bool = true,
	allocator := context.allocator,
) -> (
	X: Matrix(complex64),
	AF: Matrix(complex64),
	ipiv: []Blas_Int,
	R: []f32,
	C: []f32,
	rcond: f32,
	ferr: []f32,
	berr: []f32,
	equed: Equilibration,
	info: Info,
) {
	n := A.rows
	nrhs := B.cols
	lda := A.ld
	ldb := B.ld

	// Allocate output matrices
	X = make_matrix(complex64, int(n), int(nrhs), .General)
	ldx := X.ld

	AF = make_matrix(complex64, int(n), int(n), .General)
	ldaf := AF.ld

	ipiv = make([]Blas_Int, n)
	R = make([]f32, n)
	C = make([]f32, n)
	ferr = make([]f32, nrhs)
	berr = make([]f32, nrhs)

	// Set options
	fact_c: cstring = do_equilibrate ? "E" : "N"
	trans_c: cstring = transpose_to_cstring(transpose)

	equed_c: [2]byte = {'N', 0}

	// Allocate workspace
	work := make([]complex64, 2 * n)
	rwork := make([]f32, 2 * n)
	defer delete(work)
	defer delete(rwork)

	lapack.cgesvx_(
		fact_c,
		trans_c,
		&n,
		&nrhs,
		raw_data(A.data),
		&lda,
		raw_data(AF.data),
		&ldaf,
		raw_data(ipiv),
		cstring(&equed_c[0]),
		raw_data(R),
		raw_data(C),
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
		1,
		1,
		1,
	)

	// Convert equilibration flag to enum
	equed = equilibration_from_char(equed_c[0])

	return X, AF, ipiv, R, C, rcond, ferr, berr, equed, info
}

m_solve_expert_c128 :: proc(
	A: ^Matrix(complex128),
	B: ^Matrix(complex128),
	do_equilibrate: bool = true,
	transpose: TransposeMode = .None,
	compute_rcond: bool = true,
	allocator := context.allocator,
) -> (
	X: Matrix(complex128),
	AF: Matrix(complex128),
	ipiv: []Blas_Int,
	R: []f64,
	C: []f64,
	rcond: f64,
	ferr: []f64,
	berr: []f64,
	equed: Equilibration,
	info: Info,
) {
	n := A.rows
	nrhs := B.cols
	lda := A.ld
	ldb := B.ld

	// Allocate output matrices
	X = make_matrix(complex128, int(n), int(nrhs), .General)
	ldx := X.ld

	AF = make_matrix(complex128, int(n), int(n), .General)
	ldaf := AF.ld

	ipiv = make([]Blas_Int, n)
	R = make([]f64, n)
	C = make([]f64, n)
	ferr = make([]f64, nrhs)
	berr = make([]f64, nrhs)

	// Set options
	fact_c: cstring = do_equilibrate ? "E" : "N"
	trans_c: cstring = transpose_to_cstring(transpose)

	equed_c: [2]byte = {'N', 0}

	// Allocate workspace
	work := make([]complex128, 2 * n)
	rwork := make([]f64, 2 * n)
	defer delete(work)
	defer delete(rwork)

	lapack.zgesvx_(
		fact_c,
		trans_c,
		&n,
		&nrhs,
		raw_data(A.data),
		&lda,
		raw_data(AF.data),
		&ldaf,
		raw_data(ipiv),
		cstring(&equed_c[0]),
		raw_data(R),
		raw_data(C),
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
		1,
		1,
		1,
	)

	// Convert equilibration flag to enum
	equed = equilibration_from_char(equed_c[0])

	return X, AF, ipiv, R, C, rcond, ferr, berr, equed, info
}

// ===================================================================================
// EXTRA-EXPERT LINEAR SOLVER WITH ADVANCED ERROR BOUNDS
// ===================================================================================
// Solve linear system with extra-expert options and componentwise error bounds
// Provides the most comprehensive error analysis available

m_solve_expert_extra_real :: proc(
	A: ^Matrix($T),
	B: ^Matrix(T),
	do_equilibrate: bool = true,
	transpose: TransposeMode = .None,
	n_err_bnds: Blas_Int = 3, // Number of error bounds to compute
	allocator := context.allocator,
) -> (
	X: Matrix(T),
	AF: Matrix(T),
	ipiv: []Blas_Int,
	R: []T,
	C: []T,
	rcond: T,
	rpvgrw: T,
	berr: []T,
	err_bnds_norm: Matrix(T),
	err_bnds_comp: Matrix(T),
	equed: Equilibration,
	info: Info, // Solution matrix// LU factorization// Pivot indices// Row scale factors// Column scale factors// Reciprocal condition number// Reciprocal pivot growth factor// Componentwise backward error// Error bounds for normwise lapack.error// Error bounds for componentwise error// Equilibration applied
) where is_float(T) {
	n := A.rows
	nrhs := B.cols
	lda := A.ld
	ldb := B.ld

	// Allocate output matrices
	X = make_matrix(T, int(n), int(nrhs))
	ldx := X.ld

	AF = make_matrix(T, int(n), int(n))
	ldaf := AF.ld

	ipiv = make([]Blas_Int, n)
	R = make([]T, n)
	C = make([]T, n)
	berr = make([]T, nrhs)

	// Error bounds matrices: nrhs x n_err_bnds
	err_bnds_norm = make_matrix(T, int(nrhs), int(n_err_bnds), .General)
	err_bnds_comp = make_matrix(T, int(nrhs), int(n_err_bnds), .General)

	// Set options
	fact_c: cstring = do_equilibrate ? "E" : "N"
	trans_c: cstring = transpose_to_cstring(transpose)

	equed_c: [2]byte = {'N', 0}

	// Parameters for algorithm tuning
	nparams: Blas_Int = 0 // Use default parameters
	params: T
	n_err_bnds_copy := n_err_bnds

	// Allocate workspace
	work := make([]T, 4 * n)
	iwork := make([]Blas_Int, n)
	defer delete(work)
	defer delete(iwork)

	when T == f32 {
		lapack.sgesvxx_(
			fact_c,
			trans_c,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			raw_data(AF.data),
			&ldaf,
			raw_data(ipiv),
			cstring(&equed_c[0]),
			raw_data(R),
			raw_data(C),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			&rcond,
			&rpvgrw,
			raw_data(berr),
			&n_err_bnds_copy,
			raw_data(err_bnds_norm.data),
			raw_data(err_bnds_comp.data),
			&nparams,
			&params,
			raw_data(work),
			raw_data(iwork),
			&info,
			1,
			1,
			1,
		)
	} else {
		lapack.dgesvxx_(
			fact_c,
			trans_c,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			raw_data(AF.data),
			&ldaf,
			raw_data(ipiv),
			cstring(&equed_c[0]),
			raw_data(R),
			raw_data(C),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			&rcond,
			&rpvgrw,
			raw_data(berr),
			&n_err_bnds_copy,
			raw_data(err_bnds_norm.data),
			raw_data(err_bnds_comp.data),
			&nparams,
			&params,
			raw_data(work),
			raw_data(iwork),
			&info,
			1,
			1,
			1,
		)
	}

	// Convert equilibration flag to enum
	equed = equilibration_from_char(equed_c[0])

	return X, AF, ipiv, R, C, rcond, rpvgrw, berr, err_bnds_norm, err_bnds_comp, equed, info
}

m_solve_expert_extra_c64 :: proc(
	A: ^Matrix(complex64),
	B: ^Matrix(complex64),
	do_equilibrate: bool = true,
	transpose: TransposeMode = .None,
	n_err_bnds: Blas_Int = 3,
	allocator := context.allocator,
) -> (
	X: Matrix(complex64),
	AF: Matrix(complex64),
	ipiv: []Blas_Int,
	R: []f32,
	C: []f32,
	rcond: f32,
	rpvgrw: f32,
	berr: []f32,
	err_bnds_norm: Matrix(f32),
	err_bnds_comp: Matrix(f32),
	equed: Equilibration,
	info: Info,
) {
	n := A.rows
	nrhs := B.cols
	lda := A.ld
	ldb := B.ld

	// Allocate output matrices
	X = make_matrix(complex64, int(n), int(nrhs), .General)
	ldx := X.ld

	AF = make_matrix(complex64, int(n), int(n), .General)
	ldaf := AF.ld

	ipiv = make([]Blas_Int, n)
	R = make([]f32, n)
	C = make([]f32, n)
	berr = make([]f32, nrhs)

	err_bnds_norm = make_matrix(f32, int(nrhs), int(n_err_bnds), .General)
	err_bnds_comp = make_matrix(f32, int(nrhs), int(n_err_bnds), .General)

	// Set options
	fact_c: cstring = do_equilibrate ? "E" : "N"
	trans_c: cstring = transpose_to_cstring(transpose)

	equed_c: [2]byte = {'N', 0}

	// Parameters for algorithm tuning
	nparams: Blas_Int = 0
	params: f32
	n_err_bnds_copy := n_err_bnds

	// Allocate workspace
	work := make([]complex64, 2 * n)
	rwork := make([]f32, 3 * n)
	defer delete(work)
	defer delete(rwork)

	lapack.cgesvxx_(
		fact_c,
		trans_c,
		&n,
		&nrhs,
		raw_data(A.data),
		&lda,
		raw_data(AF.data),
		&ldaf,
		raw_data(ipiv),
		cstring(&equed_c[0]),
		raw_data(R),
		raw_data(C),
		raw_data(B.data),
		&ldb,
		raw_data(X.data),
		&ldx,
		&rcond,
		&rpvgrw,
		raw_data(berr),
		&n_err_bnds_copy,
		raw_data(err_bnds_norm.data),
		raw_data(err_bnds_comp.data),
		&nparams,
		&params,
		raw_data(work),
		raw_data(rwork),
		&info,
		1,
		1,
		1,
	)

	// Convert equilibration flag to enum
	equed = equilibration_from_char(equed_c[0])

	return X, AF, ipiv, R, C, rcond, rpvgrw, berr, err_bnds_norm, err_bnds_comp, equed, info
}

m_solve_expert_extra_c128 :: proc(
	A: ^Matrix(complex128),
	B: ^Matrix(complex128),
	do_equilibrate: bool = true,
	transpose: TransposeMode = .None,
	n_err_bnds: Blas_Int = 3,
	allocator := context.allocator,
) -> (
	X: Matrix(complex128),
	AF: Matrix(complex128),
	ipiv: []Blas_Int,
	R: []f64,
	C: []f64,
	rcond: f64,
	rpvgrw: f64,
	berr: []f64,
	err_bnds_norm: Matrix(f64),
	err_bnds_comp: Matrix(f64),
	equed: Equilibration,
	info: Info,
) {
	n := A.rows
	nrhs := B.cols
	lda := A.ld
	ldb := B.ld

	// Allocate output matrices
	X = make_matrix(complex128, int(n), int(nrhs), .General)
	ldx := X.ld

	AF = make_matrix(complex128, int(n), int(n), .General)
	ldaf := AF.ld

	ipiv = make([]Blas_Int, n)
	R = make([]f64, n)
	C = make([]f64, n)
	berr = make([]f64, nrhs)

	err_bnds_norm = make_matrix(f64, int(nrhs), int(n_err_bnds), .General)
	err_bnds_comp = make_matrix(f64, int(nrhs), int(n_err_bnds), .General)

	// Set options
	fact_c: cstring = do_equilibrate ? "E" : "N"
	trans_c: cstring = transpose_to_cstring(transpose)

	equed_c: [2]byte = {'N', 0}

	// Parameters for algorithm tuning
	nparams: Blas_Int = 0
	params: f64
	n_err_bnds_copy := n_err_bnds

	// Allocate workspace
	work := make([]complex128, 2 * n)
	rwork := make([]f64, 3 * n)
	defer delete(work)
	defer delete(rwork)

	lapack.zgesvxx_(
		fact_c,
		trans_c,
		&n,
		&nrhs,
		raw_data(A.data),
		&lda,
		raw_data(AF.data),
		&ldaf,
		raw_data(ipiv),
		cstring(&equed_c[0]),
		raw_data(R),
		raw_data(C),
		raw_data(B.data),
		&ldb,
		raw_data(X.data),
		&ldx,
		&rcond,
		&rpvgrw,
		raw_data(berr),
		&n_err_bnds_copy,
		raw_data(err_bnds_norm.data),
		raw_data(err_bnds_comp.data),
		&nparams,
		&params,
		raw_data(work),
		raw_data(rwork),
		&info,
		1,
		1,
		1,
	)

	// Convert equilibration flag to enum
	equed = equilibration_from_char(equed_c[0])

	return X, AF, ipiv, R, C, rcond, rpvgrw, berr, err_bnds_norm, err_bnds_comp, equed, info
}

// ===================================================================================
// LU FACTORIZATION
// ===================================================================================
// Compute LU factorization of a matrix using recursive algorithm
// More efficient for tall matrices


m_lu_factor_recursive :: proc(
	A: ^Matrix($T),
	allocator := context.allocator,
) -> (
	ipiv: []Blas_Int,
	info: Info, // Pivot indices
) where is_float(T) || is_complex(T) {
	m := A.rows
	n := A.cols
	lda := A.ld

	// Allocate pivot array
	ipiv = make([]Blas_Int, min(m, n))

	when T == f32 {
		lapack.sgetrf2_(&m, &n, raw_data(A.data), &lda, raw_data(ipiv), &info)
	} else when T == f64 {
		lapack.dgetrf2_(&m, &n, raw_data(A.data), &lda, raw_data(ipiv), &info)
	} else when T == complex64 {
		lapack.cgetrf2_(&m, &n, raw_data(A.data), &lda, raw_data(ipiv), &info)
	} else when T == complex128 {
		lapack.zgetrf2_(&m, &n, raw_data(A.data), &lda, raw_data(ipiv), &info)
	}

	return ipiv, info
}

// ===================================================================================
// MATRIX INVERSION
// ===================================================================================
// Compute matrix inverse using LU factorization
// A^(-1) is computed from the LU factorization

m_inverse :: proc(
	A: ^Matrix($T),
	ipiv: []Blas_Int, // Pivot indices from LU factorization
	allocator := context.allocator,
) -> (
	info: Info,
) where is_float(T) || is_complex(T) {
	n := A.rows
	lda := A.ld

	// Query for optimal workspace
	lwork: Blas_Int = -1
	work_query: T
	when T == f32 {
		lapack.sgetri_(&n, raw_data(A.data), &lda, raw_data(ipiv), &work_query, &lwork, &info)
	} else when T == f64 {
		lapack.dgetri_(&n, raw_data(A.data), &lda, raw_data(ipiv), &work_query, &lwork, &info)
	} else when T == complex64 {
		lapack.cgetri_(&n, raw_data(A.data), &lda, raw_data(ipiv), &work_query, &lwork, &info)
	} else when T == complex128 {
		lapack.zgetri_(&n, raw_data(A.data), &lda, raw_data(ipiv), &work_query, &lwork, &info)
	}

	// Allocate workspace
	lwork = Blas_Int(work_query)
	work := make([]T, lwork)
	defer delete(work)

	// Compute inverse
	when T == f32 {
		lapack.sgetri_(&n, raw_data(A.data), &lda, raw_data(ipiv), raw_data(work), &lwork, &info)
	} else when T == f64 {
		lapack.dgetri_(&n, raw_data(A.data), &lda, raw_data(ipiv), raw_data(work), &lwork, &info)
	} else when T == complex64 {
		lapack.cgetri_(&n, raw_data(A.data), &lda, raw_data(ipiv), raw_data(work), &lwork, &info)
	} else when T == complex128 {
		lapack.zgetri_(&n, raw_data(A.data), &lda, raw_data(ipiv), raw_data(work), &lwork, &info)
	}

	return info
}

m_inverse_direct :: proc(
	A: ^Matrix($T),
	allocator := context.allocator,
) -> (
	ipiv: []Blas_Int,
	info: Info,
) where is_float(T) ||
	is_complex(T) {
	// LU Factorization
	ipiv, info = m_lu_factor_recursive(A, allocator)
	if info != 0 {
		return ipiv, info
	}

	info = m_inverse(A, ipiv, allocator)
	return ipiv, info
}


// ===================================================================================
// LEAST SQUARES SOLVERS
// ===================================================================================

// Solve overdetermined or underdetermined linear system using QR or LQ factorization
// For m >= n: finds least squares solution to minimize ||B - AX||
// For m < n: finds minimum norm solution

m_solve_least_squares :: proc(
	A: ^Matrix($T),
	B: ^Matrix(T),
	transpose: TransposeMode = .None,
	allocator := context.allocator,
) -> (
	X: Matrix(T),
	info: Info, // Solution matrix
) where is_float(T) || is_complex(T) {
	m := A.rows
	n := A.cols
	nrhs := B.cols
	lda := A.ld
	ldb := B.ld

	trans_c: cstring = transpose_to_cstring(transpose)

	// Determine size of solution
	x_rows := transpose ? int(m) : int(n)
	X = make_matrix(T, x_rows, int(nrhs))

	// Copy B to X (getsls overwrites B with solution)
	if transpose {
		// For transpose case, solution has m rows
		copy(X.data[:m * nrhs], B.data[:m * nrhs])
	} else {
		// For normal case, solution has n rows
		copy(X.data[:n * nrhs], B.data[:n * nrhs])
	}

	// Query for optimal workspace
	lwork: Blas_Int = -1
	work_query: T

	when T == f32 {
		lapack.sgetsls_(
			trans_c,
			&m,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			&work_query,
			&lwork,
			&info,
			1,
		)
	} else when T == f64 {
		lapack.dgetsls_(
			trans_c,
			&m,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			&work_query,
			&lwork,
			&info,
			1,
		)
	} else when T == complex64 {
		lapack.cgetsls_(
			trans_c,
			&m,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			&work_query,
			&lwork,
			&info,
			1,
		)
	} else when T == complex128 {
		lapack.zgetsls_(
			trans_c,
			&m,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			&work_query,
			&lwork,
			&info,
			1,
		)
	}

	// Allocate workspace
	lwork = Blas_Int(work_query)
	work := make([]T, lwork)
	defer delete(work)

	// Solve the system
	when T == f32 {
		lapack.sgetsls_(
			trans_c,
			&m,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			raw_data(work),
			&lwork,
			&info,
			1,
		)
	} else when T == f64 {
		lapack.dgetsls_(
			trans_c,
			&m,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			raw_data(work),
			&lwork,
			&info,
			1,
		)
	} else when T == complex64 {
		lapack.cgetsls_(
			trans_c,
			&m,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			&work_query,
			&lwork,
			&info,
			1,
		)
	} else when T == complex128 {
		lapack.zgetsls_(
			trans_c,
			&m,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			&work_query,
			&lwork,
			&info,
			1,
		)
	}

	// Copy solution from B to X
	if transpose {
		copy(X.data[:m * nrhs], B.data[:m * nrhs])
	} else {
		copy(X.data[:n * nrhs], B.data[:n * nrhs])
	}

	return X, info
}
