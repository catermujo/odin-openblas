package openblas

import lapack "./f77"
import "base:builtin"
import "base:intrinsics"
import c "core:c"

// ===================================================================================
// GENERALIZED PROBLEMS
// Problems involving matrix pairs (A,B) and matrix pencils A - λB
// ===================================================================================

// Selection function types for sorting generalized eigenvalues
LAPACK_S_SELECT3 :: proc "c" (alphar: ^f32, alphai: ^f32, beta: ^f32) -> Blas_Int
LAPACK_D_SELECT3 :: proc "c" (alphar: ^f64, alphai: ^f64, beta: ^f64) -> Blas_Int
LAPACK_C_SELECT2 :: proc "c" (alpha: ^complex64, beta: ^complex64) -> Blas_Int
LAPACK_Z_SELECT2 :: proc "c" (alpha: ^complex128, beta: ^complex128) -> Blas_Int

// ===================================================================================
// GENERALIZED BALANCING
// ===================================================================================

// Balance a pair of general matrices for generalized eigenvalue problem
// Improves accuracy of computed eigenvalues and eigenvectors
m_balance_generalized :: proc {
	m_balance_generalized_real,
	m_balance_generalized_c64,
	m_balance_generalized_c128,
}

m_balance_generalized_real :: proc(
	A: ^Matrix($T),
	B: ^Matrix(T),
	permute: bool = true, // Permute to isolate eigenvalues
	scale: bool = true, // Scale to improve conditioning
	allocator := context.allocator,
) -> (
	ilo, ihi: Blas_Int,
	lscale, rscale: []T,
	info: Info, // Indices of balanced submatrices// Left and right scale factors
) where T == f32 || T == f64 {
	n := Blas_Int(A.rows)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Set job parameter
	job_c: cstring
	if permute && scale {
		job_c = "B" // Both permute and scale
	} else if permute {
		job_c = "P" // Permute only
	} else if scale {
		job_c = "S" // Scale only
	} else {
		job_c = "N" // Nothing
	}

	// Allocate scale arrays
	lscale = builtin.make([]T, n, allocator)
	rscale = builtin.make([]T, n, allocator)

	// Allocate workspace
	work := builtin.make([]T, 6 * n, allocator)
	defer builtin.delete(work)

	when T == f32 {
		lapack.sggbal_(
			job_c,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			&ilo,
			&ihi,
			raw_data(lscale),
			raw_data(rscale),
			raw_data(work),
			&info,
			1,
		)
	} else {
		lapack.dggbal_(
			job_c,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			&ilo,
			&ihi,
			raw_data(lscale),
			raw_data(rscale),
			raw_data(work),
			&info,
			1,
		)
	}

	return ilo, ihi, lscale, rscale, info
}

m_balance_generalized_c64 :: proc(
	A: ^Matrix(complex64),
	B: ^Matrix(complex64),
	permute: bool = true,
	scale: bool = true,
	allocator := context.allocator,
) -> (
	ilo, ihi: Blas_Int,
	lscale, rscale: []f32,
	info: Info,
) {
	n := Blas_Int(A.rows)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Set job parameter
	job_c: cstring
	if permute && scale {
		job_c = "B"
	} else if permute {
		job_c = "P"
	} else if scale {
		job_c = "S"
	} else {
		job_c = "N"
	}

	// Allocate scale arrays
	lscale = builtin.make([]f32, n, allocator)
	rscale = builtin.make([]f32, n, allocator)

	// Allocate workspace
	work := builtin.make([]f32, 6 * n, allocator)
	defer builtin.delete(work)

	lapack.cggbal_(
		job_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		&ilo,
		&ihi,
		raw_data(lscale),
		raw_data(rscale),
		raw_data(work),
		&info,
		1,
	)

	return ilo, ihi, lscale, rscale, info
}

m_balance_generalized_c128 :: proc(
	A: ^Matrix(complex128),
	B: ^Matrix(complex128),
	permute: bool = true,
	scale: bool = true,
	allocator := context.allocator,
) -> (
	ilo, ihi: Blas_Int,
	lscale, rscale: []f64,
	info: Info,
) {
	n := Blas_Int(A.rows)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Set job parameter
	job_c: cstring
	if permute && scale {
		job_c = "B"
	} else if permute {
		job_c = "P"
	} else if scale {
		job_c = "S"
	} else {
		job_c = "N"
	}

	// Allocate scale arrays
	lscale = builtin.make([]f64, n, allocator)
	rscale = builtin.make([]f64, n, allocator)

	// Allocate workspace
	work := builtin.make([]f64, 6 * n, allocator)
	defer builtin.delete(work)

	lapack.zggbal_(
		job_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		&ilo,
		&ihi,
		raw_data(lscale),
		raw_data(rscale),
		raw_data(work),
		&info,
		1,
	)

	return ilo, ihi, lscale, rscale, info
}

// Back-transform eigenvectors after generalized balancing
m_balance_generalized_back :: proc {
	m_balance_generalized_back_real,
	m_balance_generalized_back_c64,
	m_balance_generalized_back_c128,
}

m_balance_generalized_back_real :: proc(
	V: ^Matrix($T), // Eigenvectors to back-transform
	ilo, ihi: Blas_Int, // Balancing indices
	lscale, rscale: []T, // Scale factors from balancing
	left_eigenvectors: bool = false, // true for left, false for right eigenvectors
) -> (
	info: Info,
) where T == f32 || T == f64 {
	n := Blas_Int(len(lscale))
	m := Blas_Int(V.cols)
	ldv := Blas_Int(V.ld)

	job_c := cstring("B") // Back permutation and scaling
	side_c := left_eigenvectors ? cstring("L") : cstring("R")

	when T == f32 {
		lapack.sggbak_(
			job_c,
			side_c,
			&n,
			&ilo,
			&ihi,
			raw_data(lscale),
			raw_data(rscale),
			&m,
			raw_data(V.data),
			&ldv,
			&info,
			1,
			1,
		)
	} else {
		lapack.dggbak_(
			job_c,
			side_c,
			&n,
			&ilo,
			&ihi,
			raw_data(lscale),
			raw_data(rscale),
			&m,
			raw_data(V.data),
			&ldv,
			&info,
			1,
			1,
		)
	}

	return info
}

m_balance_generalized_back_c64 :: proc(
	V: ^Matrix(complex64),
	ilo, ihi: Blas_Int,
	lscale, rscale: []f32,
	left_eigenvectors: bool = false,
) -> (
	info: Info,
) {
	n := Blas_Int(len(lscale))
	m := Blas_Int(V.cols)
	ldv := Blas_Int(V.ld)

	job_c := cstring("B")
	side_c := left_eigenvectors ? cstring("L") : cstring("R")

	lapack.cggbak_(
		job_c,
		side_c,
		&n,
		&ilo,
		&ihi,
		raw_data(lscale),
		raw_data(rscale),
		&m,
		raw_data(V.data),
		&ldv,
		&info,
		1,
		1,
	)

	return info
}

m_balance_generalized_back_c128 :: proc(
	V: ^Matrix(complex128),
	ilo, ihi: Blas_Int,
	lscale, rscale: []f64,
	left_eigenvectors: bool = false,
) -> (
	info: Info,
) {
	n := Blas_Int(len(lscale))
	m := Blas_Int(V.cols)
	ldv := Blas_Int(V.ld)

	job_c := cstring("B")
	side_c := left_eigenvectors ? cstring("L") : cstring("R")

	lapack.zggbak_(
		job_c,
		side_c,
		&n,
		&ilo,
		&ihi,
		raw_data(lscale),
		raw_data(rscale),
		&m,
		raw_data(V.data),
		&ldv,
		&info,
		1,
		1,
	)

	return info
}

// ===================================================================================
// GENERALIZED SCHUR DECOMPOSITION
// ===================================================================================

// Compute generalized Schur decomposition of matrix pair
// A = Q*S*Z^H, B = Q*T*Z^H where S,T are upper triangular (or quasi-triangular for real)
m_schur_generalized :: proc {
	m_schur_generalized_real,
	m_schur_generalized_c64,
	m_schur_generalized_c128,
}

m_schur_generalized_real :: proc(
	A: ^Matrix($T),
	B: ^Matrix(T),
	compute_left: bool = true, // Compute left Schur vectors Q
	compute_right: bool = true, // Compute right Schur vectors Z
	sort_eigenvalues: bool = false, // Sort eigenvalues
	select_fn: LAPACK_D_SELECT3 = nil when T == f64 else LAPACK_S_SELECT3,
) -> (
	S, T_mat: Matrix(T),
	VSL: Matrix(T),
	VSR: Matrix(T),
	alphar, alphai, beta: []T,// Selection function for sorting
	sdim: Blas_Int,// Schur forms (A becomes S, B becomes T)
	info: Info, // Left Schur vectors (Q)// Right Schur vectors (Z)// Generalized eigenvalues: lambda = (alphar + i*alphai)/beta// Number of eigenvalues selected (if sorting)
) where T == f32 || T == f64 {
	n := Blas_Int(A.rows)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Copy A and B to S and T (they will be overwritten)
	S := make_matrix(T, int(n), int(n), allocator)
	T_mat = make_matrix(T, int(n), int(n), allocator)
	copy(S.data, A.data[:n * n])
	copy(T_mat.data, B.data[:n * n])
	lds := Blas_Int(S.ld)
	ldt := Blas_Int(T_mat.ld)

	// Allocate eigenvalue arrays
	alphar = builtin.make([]T, n, allocator)
	alphai = builtin.make([]T, n, allocator)
	beta = builtin.make([]T, n, allocator)

	// Set job parameters
	jobvsl_c := compute_left ? cstring("V") : cstring("N")
	jobvsr_c := compute_right ? cstring("V") : cstring("N")
	sort_c := sort_eigenvalues ? cstring("S") : cstring("N")

	// Allocate Schur vector matrices
	ldvsl := Blas_Int(1)
	ldvsr := Blas_Int(1)
	if compute_left {
		VSL = make_matrix(T, int(n), int(n), allocator)
		ldvsl = Blas_Int(VSL.ld)
	}
	if compute_right {
		VSR = make_matrix(T, int(n), int(n), allocator)
		ldvsr = Blas_Int(VSR.ld)
	}

	// Allocate boolean work array for sorting
	var; bwork: []i32
	if sort_eigenvalues {
		bwork = builtin.make([]i32, n, allocator)
		defer builtin.delete(bwork)
	}

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: T

	when T == f32 {
		lapack.sgges_(
			jobvsl_c,
			jobvsr_c,
			sort_c,
			cast(LAPACK_S_SELECT3)select_fn,
			&n,
			raw_data(S.data),
			&lds,
			raw_data(T_mat.data),
			&ldt,
			&sdim,
			raw_data(alphar),
			raw_data(alphai),
			raw_data(beta),
			compute_left ? raw_data(VSL.data) : nil,
			&ldvsl,
			compute_right ? raw_data(VSR.data) : nil,
			&ldvsr,
			&work_query,
			&lwork,
			sort_eigenvalues ? raw_data(bwork) : nil,
			&info,
			1,
			1,
			1,
		)
	} else {
		lapack.dgges_(
			jobvsl_c,
			jobvsr_c,
			sort_c,
			cast(LAPACK_D_SELECT3)select_fn,
			&n,
			raw_data(S.data),
			&lds,
			raw_data(T_mat.data),
			&ldt,
			&sdim,
			raw_data(alphar),
			raw_data(alphai),
			raw_data(beta),
			compute_left ? raw_data(VSL.data) : nil,
			&ldvsl,
			compute_right ? raw_data(VSR.data) : nil,
			&ldvsr,
			&work_query,
			&lwork,
			sort_eigenvalues ? raw_data(bwork) : nil,
			&info,
			1,
			1,
			1,
		)
	}

	// Allocate workspace
	lwork = Blas_Int(work_query)
	work := builtin.make([]T, lwork, allocator)
	defer builtin.delete(work)

	// Compute Schur decomposition
	when T == f32 {
		lapack.sgges_(
			jobvsl_c,
			jobvsr_c,
			sort_c,
			cast(LAPACK_S_SELECT3)select_fn,
			&n,
			raw_data(S.data),
			&lds,
			raw_data(T_mat.data),
			&ldt,
			&sdim,
			raw_data(alphar),
			raw_data(alphai),
			raw_data(beta),
			compute_left ? raw_data(VSL.data) : nil,
			&ldvsl,
			compute_right ? raw_data(VSR.data) : nil,
			&ldvsr,
			raw_data(work),
			&lwork,
			sort_eigenvalues ? raw_data(bwork) : nil,
			&info,
			1,
			1,
			1,
		)
	} else {
		lapack.dgges_(
			jobvsl_c,
			jobvsr_c,
			sort_c,
			cast(LAPACK_D_SELECT3)select_fn,
			&n,
			raw_data(S.data),
			&lds,
			raw_data(T_mat.data),
			&ldt,
			&sdim,
			raw_data(alphar),
			raw_data(alphai),
			raw_data(beta),
			compute_left ? raw_data(VSL.data) : nil,
			&ldvsl,
			compute_right ? raw_data(VSR.data) : nil,
			&ldvsr,
			raw_data(work),
			&lwork,
			sort_eigenvalues ? raw_data(bwork) : nil,
			&info,
			1,
			1,
			1,
		)
	}

	return S, T_mat, VSL, VSR, alphar, alphai, beta, sdim, info
}

m_schur_generalized_c64 :: proc(
	A: ^Matrix(complex64),
	B: ^Matrix(complex64),
	compute_left: bool = true,
	compute_right: bool = true,
	sort_eigenvalues: bool = false,
	select_fn: LAPACK_C_SELECT2 = nil,
	allocator := context.allocator,
) -> (
	S, T_mat: Matrix(complex64),
	VSL: Matrix(complex64),
	VSR: Matrix(complex64),
	alpha, beta: []complex64,
	sdim: Blas_Int,
	info: Info,
) {
	n := Blas_Int(A.rows)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Copy A and B to S and T
	S = make_matrix(complex64, int(n), int(n), allocator)
	T_mat = make_matrix(complex64, int(n), int(n), allocator)
	copy(S.data, A.data[:n * n])
	copy(T_mat.data, B.data[:n * n])
	lds := Blas_Int(S.ld)
	ldt := Blas_Int(T_mat.ld)

	// Allocate eigenvalue arrays
	alpha = builtin.make([]complex64, n, allocator)
	beta = builtin.make([]complex64, n, allocator)

	// Set job parameters
	jobvsl_c := compute_left ? cstring("V") : cstring("N")
	jobvsr_c := compute_right ? cstring("V") : cstring("N")
	sort_c := sort_eigenvalues ? cstring("S") : cstring("N")

	// Allocate Schur vector matrices
	ldvsl := Blas_Int(1)
	ldvsr := Blas_Int(1)
	if compute_left {
		VSL = make_matrix(complex64, int(n), int(n), allocator)
		ldvsl = Blas_Int(VSL.ld)
	}
	if compute_right {
		VSR = make_matrix(complex64, int(n), int(n), allocator)
		ldvsr = Blas_Int(VSR.ld)
	}

	// Allocate work arrays
	var; bwork: []i32
	if sort_eigenvalues {
		bwork = builtin.make([]i32, n, allocator)
		defer builtin.delete(bwork)
	}

	rwork := builtin.make([]f32, 8 * n, allocator)
	defer builtin.delete(rwork)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: complex64

	lapack.cgges_(
		jobvsl_c,
		jobvsr_c,
		sort_c,
		select_fn,
		&n,
		raw_data(S.data),
		&lds,
		raw_data(T_mat.data),
		&ldt,
		&sdim,
		raw_data(alpha),
		raw_data(beta),
		compute_left ? raw_data(VSL.data) : nil,
		&ldvsl,
		compute_right ? raw_data(VSR.data) : nil,
		&ldvsr,
		&work_query,
		&lwork,
		raw_data(rwork),
		sort_eigenvalues ? raw_data(bwork) : nil,
		&info,
		1,
		1,
		1,
	)

	// Allocate workspace
	lwork = Blas_Int(real(work_query))
	work := builtin.make([]complex64, lwork, allocator)
	defer builtin.delete(work)

	// Compute Schur decomposition
	lapack.cgges_(
		jobvsl_c,
		jobvsr_c,
		sort_c,
		select_fn,
		&n,
		raw_data(S.data),
		&lds,
		raw_data(T_mat.data),
		&ldt,
		&sdim,
		raw_data(alpha),
		raw_data(beta),
		compute_left ? raw_data(VSL.data) : nil,
		&ldvsl,
		compute_right ? raw_data(VSR.data) : nil,
		&ldvsr,
		raw_data(work),
		&lwork,
		raw_data(rwork),
		sort_eigenvalues ? raw_data(bwork) : nil,
		&info,
		1,
		1,
		1,
	)

	return S, T_mat, VSL, VSR, alpha, beta, sdim, info
}

m_schur_generalized_c128 :: proc(
	A: ^Matrix(complex128),
	B: ^Matrix(complex128),
	compute_left: bool = true,
	compute_right: bool = true,
	sort_eigenvalues: bool = false,
	select_fn: LAPACK_Z_SELECT2 = nil,
	allocator := context.allocator,
) -> (
	S, T_mat: Matrix(complex128),
	VSL: Matrix(complex128),
	VSR: Matrix(complex128),
	alpha, beta: []complex128,
	sdim: Blas_Int,
	info: Info,
) {
	n := Blas_Int(A.rows)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Copy A and B to S and T
	S = make_matrix(complex128, int(n), int(n), allocator)
	T_mat = make_matrix(complex128, int(n), int(n), allocator)
	copy(S.data, A.data[:n * n])
	copy(T_mat.data, B.data[:n * n])
	lds := Blas_Int(S.ld)
	ldt := Blas_Int(T_mat.ld)

	// Allocate eigenvalue arrays
	alpha = builtin.make([]complex128, n, allocator)
	beta = builtin.make([]complex128, n, allocator)

	// Set job parameters
	jobvsl_c := compute_left ? cstring("V") : cstring("N")
	jobvsr_c := compute_right ? cstring("V") : cstring("N")
	sort_c := sort_eigenvalues ? cstring("S") : cstring("N")

	// Allocate Schur vector matrices
	ldvsl := Blas_Int(1)
	ldvsr := Blas_Int(1)
	if compute_left {
		VSL = make_matrix(complex128, int(n), int(n), allocator)
		ldvsl = Blas_Int(VSL.ld)
	}
	if compute_right {
		VSR = make_matrix(complex128, int(n), int(n), allocator)
		ldvsr = Blas_Int(VSR.ld)
	}

	// Allocate work arrays
	var; bwork: []i32
	if sort_eigenvalues {
		bwork = builtin.make([]i32, n, allocator)
		defer builtin.delete(bwork)
	}

	rwork := builtin.make([]f64, 8 * n, allocator)
	defer builtin.delete(rwork)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: complex128

	lapack.zgges_(
		jobvsl_c,
		jobvsr_c,
		sort_c,
		select_fn,
		&n,
		raw_data(S.data),
		&lds,
		raw_data(T_mat.data),
		&ldt,
		&sdim,
		raw_data(alpha),
		raw_data(beta),
		compute_left ? raw_data(VSL.data) : nil,
		&ldvsl,
		compute_right ? raw_data(VSR.data) : nil,
		&ldvsr,
		&work_query,
		&lwork,
		raw_data(rwork),
		sort_eigenvalues ? raw_data(bwork) : nil,
		&info,
		1,
		1,
		1,
	)

	// Allocate workspace
	lwork = Blas_Int(real(work_query))
	work := builtin.make([]complex128, lwork, allocator)
	defer builtin.delete(work)

	// Compute Schur decomposition
	lapack.zgges_(
		jobvsl_c,
		jobvsr_c,
		sort_c,
		select_fn,
		&n,
		raw_data(S.data),
		&lds,
		raw_data(T_mat.data),
		&ldt,
		&sdim,
		raw_data(alpha),
		raw_data(beta),
		compute_left ? raw_data(VSL.data) : nil,
		&ldvsl,
		compute_right ? raw_data(VSR.data) : nil,
		&ldvsr,
		raw_data(work),
		&lwork,
		raw_data(rwork),
		sort_eigenvalues ? raw_data(bwork) : nil,
		&info,
		1,
		1,
		1,
	)

	return S, T_mat, VSL, VSR, alpha, beta, sdim, info
}

// Compute generalized Schur decomposition with improved blocked algorithm
// More efficient than gges for large matrices
m_schur_generalized_blocked :: proc {
	m_schur_generalized_blocked_real,
	m_schur_generalized_blocked_c64,
	m_schur_generalized_blocked_c128,
}

m_schur_generalized_blocked_real :: proc(
	A: ^Matrix($T),
	B: ^Matrix(T),
	compute_left: bool = true,
	compute_right: bool = true,
	sort_eigenvalues: bool = false,
	select_fn: LAPACK_D_SELECT3 = nil when T == f64 else LAPACK_S_SELECT3,
) -> (
	S, T_mat: Matrix(T),
	VSL: Matrix(T),
	VSR: Matrix(T),
	alphar, alphai, beta: []T,
	sdim: Blas_Int,
	info: Info,
) where T == f32 ||
	T == f64 {
	n := Blas_Int(A.rows)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Copy A and B to S and T
	S = make_matrix(T, int(n), int(n), allocator)
	T_mat = make_matrix(T, int(n), int(n), allocator)
	copy(S.data, A.data[:n * n])
	copy(T_mat.data, B.data[:n * n])
	lds := Blas_Int(S.ld)
	ldt := Blas_Int(T_mat.ld)

	// Allocate eigenvalue arrays
	alphar = builtin.make([]T, n, allocator)
	alphai = builtin.make([]T, n, allocator)
	beta = builtin.make([]T, n, allocator)

	// Set job parameters
	jobvsl_c := compute_left ? cstring("V") : cstring("N")
	jobvsr_c := compute_right ? cstring("V") : cstring("N")
	sort_c := sort_eigenvalues ? cstring("S") : cstring("N")

	// Allocate Schur vector matrices
	ldvsl := Blas_Int(1)
	ldvsr := Blas_Int(1)
	if compute_left {
		VSL = make_matrix(T, int(n), int(n), allocator)
		ldvsl = Blas_Int(VSL.ld)
	}
	if compute_right {
		VSR = make_matrix(T, int(n), int(n), allocator)
		ldvsr = Blas_Int(VSR.ld)
	}

	// Allocate boolean work array for sorting
	var; bwork: []i32
	if sort_eigenvalues {
		bwork = builtin.make([]i32, n, allocator)
		defer builtin.delete(bwork)
	}

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: T

	when T == f32 {
		lapack.sgges3_(
			jobvsl_c,
			jobvsr_c,
			sort_c,
			cast(LAPACK_S_SELECT3)select_fn,
			&n,
			raw_data(S.data),
			&lds,
			raw_data(T_mat.data),
			&ldt,
			&sdim,
			raw_data(alphar),
			raw_data(alphai),
			raw_data(beta),
			compute_left ? raw_data(VSL.data) : nil,
			&ldvsl,
			compute_right ? raw_data(VSR.data) : nil,
			&ldvsr,
			&work_query,
			&lwork,
			sort_eigenvalues ? raw_data(bwork) : nil,
			&info,
			1,
			1,
			1,
		)
	} else {
		lapack.dgges3_(
			jobvsl_c,
			jobvsr_c,
			sort_c,
			cast(LAPACK_D_SELECT3)select_fn,
			&n,
			raw_data(S.data),
			&lds,
			raw_data(T_mat.data),
			&ldt,
			&sdim,
			raw_data(alphar),
			raw_data(alphai),
			raw_data(beta),
			compute_left ? raw_data(VSL.data) : nil,
			&ldvsl,
			compute_right ? raw_data(VSR.data) : nil,
			&ldvsr,
			&work_query,
			&lwork,
			sort_eigenvalues ? raw_data(bwork) : nil,
			&info,
			1,
			1,
			1,
		)
	}

	// Allocate workspace
	lwork = Blas_Int(work_query)
	work := builtin.make([]T, lwork, allocator)
	defer builtin.delete(work)

	// Compute Schur decomposition
	when T == f32 {
		lapack.sgges3_(
			jobvsl_c,
			jobvsr_c,
			sort_c,
			cast(LAPACK_S_SELECT3)select_fn,
			&n,
			raw_data(S.data),
			&lds,
			raw_data(T_mat.data),
			&ldt,
			&sdim,
			raw_data(alphar),
			raw_data(alphai),
			raw_data(beta),
			compute_left ? raw_data(VSL.data) : nil,
			&ldvsl,
			compute_right ? raw_data(VSR.data) : nil,
			&ldvsr,
			raw_data(work),
			&lwork,
			sort_eigenvalues ? raw_data(bwork) : nil,
			&info,
			1,
			1,
			1,
		)
	} else {
		lapack.dgges3_(
			jobvsl_c,
			jobvsr_c,
			sort_c,
			cast(LAPACK_D_SELECT3)select_fn,
			&n,
			raw_data(S.data),
			&lds,
			raw_data(T_mat.data),
			&ldt,
			&sdim,
			raw_data(alphar),
			raw_data(alphai),
			raw_data(beta),
			compute_left ? raw_data(VSL.data) : nil,
			&ldvsl,
			compute_right ? raw_data(VSR.data) : nil,
			&ldvsr,
			raw_data(work),
			&lwork,
			sort_eigenvalues ? raw_data(bwork) : nil,
			&info,
			1,
			1,
			1,
		)
	}

	return S, T_mat, VSL, VSR, alphar, alphai, beta, sdim, info
}

m_schur_generalized_blocked_c64 :: proc(
	A: ^Matrix(complex64),
	B: ^Matrix(complex64),
	compute_left: bool = true,
	compute_right: bool = true,
	sort_eigenvalues: bool = false,
	select_fn: LAPACK_C_SELECT2 = nil,
	allocator := context.allocator,
) -> (
	S, T_mat: Matrix(complex64),
	VSL: Matrix(complex64),
	VSR: Matrix(complex64),
	alpha, beta: []complex64,
	sdim: Blas_Int,
	info: Info,
) {
	n := Blas_Int(A.rows)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Copy A and B to S and T
	S = make_matrix(complex64, int(n), int(n), allocator)
	T_mat = make_matrix(complex64, int(n), int(n), allocator)
	copy(S.data, A.data[:n * n])
	copy(T_mat.data, B.data[:n * n])
	lds := Blas_Int(S.ld)
	ldt := Blas_Int(T_mat.ld)

	// Allocate eigenvalue arrays
	alpha = builtin.make([]complex64, n, allocator)
	beta = builtin.make([]complex64, n, allocator)

	// Set job parameters
	jobvsl_c := compute_left ? cstring("V") : cstring("N")
	jobvsr_c := compute_right ? cstring("V") : cstring("N")
	sort_c := sort_eigenvalues ? cstring("S") : cstring("N")

	// Allocate Schur vector matrices
	ldvsl := Blas_Int(1)
	ldvsr := Blas_Int(1)
	if compute_left {
		VSL = make_matrix(complex64, int(n), int(n), allocator)
		ldvsl = Blas_Int(VSL.ld)
	}
	if compute_right {
		VSR = make_matrix(complex64, int(n), int(n), allocator)
		ldvsr = Blas_Int(VSR.ld)
	}

	// Allocate work arrays
	var; bwork: []i32
	if sort_eigenvalues {
		bwork = builtin.make([]i32, n, allocator)
		defer builtin.delete(bwork)
	}

	rwork := builtin.make([]f32, 8 * n, allocator)
	defer builtin.delete(rwork)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: complex64

	lapack.cgges3_(
		jobvsl_c,
		jobvsr_c,
		sort_c,
		select_fn,
		&n,
		raw_data(S.data),
		&lds,
		raw_data(T_mat.data),
		&ldt,
		&sdim,
		raw_data(alpha),
		raw_data(beta),
		compute_left ? raw_data(VSL.data) : nil,
		&ldvsl,
		compute_right ? raw_data(VSR.data) : nil,
		&ldvsr,
		&work_query,
		&lwork,
		raw_data(rwork),
		sort_eigenvalues ? raw_data(bwork) : nil,
		&info,
		1,
		1,
		1,
	)

	// Allocate workspace
	lwork = Blas_Int(real(work_query))
	work := builtin.make([]complex64, lwork, allocator)
	defer builtin.delete(work)

	// Compute Schur decomposition
	lapack.cgges3_(
		jobvsl_c,
		jobvsr_c,
		sort_c,
		select_fn,
		&n,
		raw_data(S.data),
		&lds,
		raw_data(T_mat.data),
		&ldt,
		&sdim,
		raw_data(alpha),
		raw_data(beta),
		compute_left ? raw_data(VSL.data) : nil,
		&ldvsl,
		compute_right ? raw_data(VSR.data) : nil,
		&ldvsr,
		raw_data(work),
		&lwork,
		raw_data(rwork),
		sort_eigenvalues ? raw_data(bwork) : nil,
		&info,
		1,
		1,
		1,
	)

	return S, T_mat, VSL, VSR, alpha, beta, sdim, info
}

m_schur_generalized_blocked_c128 :: proc(
	A: ^Matrix(complex128),
	B: ^Matrix(complex128),
	compute_left: bool = true,
	compute_right: bool = true,
	sort_eigenvalues: bool = false,
	select_fn: LAPACK_Z_SELECT2 = nil,
	allocator := context.allocator,
) -> (
	S, T_mat: Matrix(complex128),
	VSL: Matrix(complex128),
	VSR: Matrix(complex128),
	alpha, beta: []complex128,
	sdim: Blas_Int,
	info: Info,
) {
	n := Blas_Int(A.rows)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Copy A and B to S and T
	S = make_matrix(complex128, int(n), int(n), allocator)
	T_mat = make_matrix(complex128, int(n), int(n), allocator)
	copy(S.data, A.data[:n * n])
	copy(T_mat.data, B.data[:n * n])
	lds := Blas_Int(S.ld)
	ldt := Blas_Int(T_mat.ld)

	// Allocate eigenvalue arrays
	alpha = builtin.make([]complex128, n, allocator)
	beta = builtin.make([]complex128, n, allocator)

	// Set job parameters
	jobvsl_c := compute_left ? cstring("V") : cstring("N")
	jobvsr_c := compute_right ? cstring("V") : cstring("N")
	sort_c := sort_eigenvalues ? cstring("S") : cstring("N")

	// Allocate Schur vector matrices
	ldvsl := Blas_Int(1)
	ldvsr := Blas_Int(1)
	if compute_left {
		VSL = make_matrix(complex128, int(n), int(n), allocator)
		ldvsl = Blas_Int(VSL.ld)
	}
	if compute_right {
		VSR = make_matrix(complex128, int(n), int(n), allocator)
		ldvsr = Blas_Int(VSR.ld)
	}

	// Allocate work arrays
	var; bwork: []i32
	if sort_eigenvalues {
		bwork = builtin.make([]i32, n, allocator)
		defer builtin.delete(bwork)
	}

	rwork := builtin.make([]f64, 8 * n, allocator)
	defer builtin.delete(rwork)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: complex128

	lapack.zgges3_(
		jobvsl_c,
		jobvsr_c,
		sort_c,
		select_fn,
		&n,
		raw_data(S.data),
		&lds,
		raw_data(T_mat.data),
		&ldt,
		&sdim,
		raw_data(alpha),
		raw_data(beta),
		compute_left ? raw_data(VSL.data) : nil,
		&ldvsl,
		compute_right ? raw_data(VSR.data) : nil,
		&ldvsr,
		&work_query,
		&lwork,
		raw_data(rwork),
		sort_eigenvalues ? raw_data(bwork) : nil,
		&info,
		1,
		1,
		1,
	)

	// Allocate workspace
	lwork = Blas_Int(real(work_query))
	work := builtin.make([]complex128, lwork, allocator)
	defer builtin.delete(work)

	// Compute Schur decomposition
	lapack.zgges3_(
		jobvsl_c,
		jobvsr_c,
		sort_c,
		select_fn,
		&n,
		raw_data(S.data),
		&lds,
		raw_data(T_mat.data),
		&ldt,
		&sdim,
		raw_data(alpha),
		raw_data(beta),
		compute_left ? raw_data(VSL.data) : nil,
		&ldvsl,
		compute_right ? raw_data(VSR.data) : nil,
		&ldvsr,
		raw_data(work),
		&lwork,
		raw_data(rwork),
		sort_eigenvalues ? raw_data(bwork) : nil,
		&info,
		1,
		1,
		1,
	)

	return S, T_mat, VSL, VSR, alpha, beta, sdim, info
}

// ===================================================================================
// EXPERT GENERALIZED SCHUR DECOMPOSITION WITH CONDITION NUMBERS
// ===================================================================================

// Compute generalized Schur decomposition with condition number estimates
// Provides reciprocal condition numbers for eigenvalues and eigenvectors
m_schur_generalized_expert :: proc {
	m_schur_generalized_expert_real,
	m_schur_generalized_expert_c64,
	m_schur_generalized_expert_c128,
}

m_schur_generalized_expert_real :: proc(
	A: ^Matrix($T),
	B: ^Matrix(T),
	compute_left: bool = true,
	compute_right: bool = true,
	sort_eigenvalues: bool = false,
	compute_condition: bool = true, // Compute condition numbers
	select_fn: LAPACK_D_SELECT3 = nil when T == f64 else LAPACK_S_SELECT3,
) -> (
	S, T_mat: Matrix(T),
	VSL: Matrix(T),
	VSR: Matrix(T),
	alphar, alphai, beta: []T,
	rconde: []T,
	rcondv: []T,
	sdim: Blas_Int,
	info: Info, // Reciprocal condition numbers for eigenvalues// Reciprocal condition numbers for eigenvectors
) where T == f32 || T == f64 {
	n := Blas_Int(A.rows)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Copy A and B to S and T
	S = make_matrix(T, int(n), int(n), allocator)
	T_mat = make_matrix(T, int(n), int(n), allocator)
	copy(S.data, A.data[:n * n])
	copy(T_mat.data, B.data[:n * n])
	lds := Blas_Int(S.ld)
	ldt := Blas_Int(T_mat.ld)

	// Allocate eigenvalue arrays
	alphar = builtin.make([]T, n, allocator)
	alphai = builtin.make([]T, n, allocator)
	beta = builtin.make([]T, n, allocator)

	// Set job parameters
	jobvsl_c := compute_left ? cstring("V") : cstring("N")
	jobvsr_c := compute_right ? cstring("V") : cstring("N")
	sort_c := sort_eigenvalues ? cstring("S") : cstring("N")

	// Set sense parameter for condition numbers
	sense_c: cstring
	if compute_condition && sort_eigenvalues {
		sense_c = "B" // Both eigenvalue and eigenvector condition numbers
	} else if compute_condition {
		sense_c = "E" // Only eigenvalue condition numbers
	} else {
		sense_c = "N" // No condition numbers
	}

	// Allocate Schur vector matrices
	ldvsl := Blas_Int(1)
	ldvsr := Blas_Int(1)
	if compute_left {
		VSL = make_matrix(T, int(n), int(n), allocator)
		ldvsl = Blas_Int(VSL.ld)
	}
	if compute_right {
		VSR = make_matrix(T, int(n), int(n), allocator)
		ldvsr = Blas_Int(VSR.ld)
	}

	// Allocate condition number arrays
	if compute_condition {
		rconde = builtin.make([]T, 2, allocator) // 2 values for selected/unselected
		rcondv = builtin.make([]T, 2, allocator)
	}

	// Allocate work arrays
	var; bwork: []i32
	if sort_eigenvalues {
		bwork = builtin.make([]i32, n, allocator)
		defer builtin.delete(bwork)
	}

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: T
	liwork := Blas_Int(-1)
	iwork_query: Blas_Int

	when T == f32 {
		lapack.sggesx_(
			jobvsl_c,
			jobvsr_c,
			sort_c,
			cast(LAPACK_S_SELECT3)select_fn,
			sense_c,
			&n,
			raw_data(S.data),
			&lds,
			raw_data(T_mat.data),
			&ldt,
			&sdim,
			raw_data(alphar),
			raw_data(alphai),
			raw_data(beta),
			compute_left ? raw_data(VSL.data) : nil,
			&ldvsl,
			compute_right ? raw_data(VSR.data) : nil,
			&ldvsr,
			compute_condition ? &rconde[0] : nil,
			compute_condition ? &rcondv[0] : nil,
			&work_query,
			&lwork,
			&iwork_query,
			&liwork,
			sort_eigenvalues ? raw_data(bwork) : nil,
			&info,
			1,
			1,
			1,
			1,
		)
	} else {
		lapack.dggesx_(
			jobvsl_c,
			jobvsr_c,
			sort_c,
			cast(LAPACK_D_SELECT3)select_fn,
			sense_c,
			&n,
			raw_data(S.data),
			&lds,
			raw_data(T_mat.data),
			&ldt,
			&sdim,
			raw_data(alphar),
			raw_data(alphai),
			raw_data(beta),
			compute_left ? raw_data(VSL.data) : nil,
			&ldvsl,
			compute_right ? raw_data(VSR.data) : nil,
			&ldvsr,
			compute_condition ? &rconde[0] : nil,
			compute_condition ? &rcondv[0] : nil,
			&work_query,
			&lwork,
			&iwork_query,
			&liwork,
			sort_eigenvalues ? raw_data(bwork) : nil,
			&info,
			1,
			1,
			1,
			1,
		)
	}

	// Allocate workspace
	lwork = Blas_Int(work_query)
	work := builtin.make([]T, lwork, allocator)
	defer builtin.delete(work)

	liwork = iwork_query
	iwork := builtin.make([]i32, liwork, allocator)
	defer builtin.delete(iwork)

	// Compute Schur decomposition with condition numbers
	when T == f32 {
		lapack.sggesx_(
			jobvsl_c,
			jobvsr_c,
			sort_c,
			cast(LAPACK_S_SELECT3)select_fn,
			sense_c,
			&n,
			raw_data(S.data),
			&lds,
			raw_data(T_mat.data),
			&ldt,
			&sdim,
			raw_data(alphar),
			raw_data(alphai),
			raw_data(beta),
			compute_left ? raw_data(VSL.data) : nil,
			&ldvsl,
			compute_right ? raw_data(VSR.data) : nil,
			&ldvsr,
			compute_condition ? &rconde[0] : nil,
			compute_condition ? &rcondv[0] : nil,
			raw_data(work),
			&lwork,
			raw_data(iwork),
			&liwork,
			sort_eigenvalues ? raw_data(bwork) : nil,
			&info,
			1,
			1,
			1,
			1,
		)
	} else {
		lapack.dggesx_(
			jobvsl_c,
			jobvsr_c,
			sort_c,
			cast(LAPACK_D_SELECT3)select_fn,
			sense_c,
			&n,
			raw_data(S.data),
			&lds,
			raw_data(T_mat.data),
			&ldt,
			&sdim,
			raw_data(alphar),
			raw_data(alphai),
			raw_data(beta),
			compute_left ? raw_data(VSL.data) : nil,
			&ldvsl,
			compute_right ? raw_data(VSR.data) : nil,
			&ldvsr,
			compute_condition ? &rconde[0] : nil,
			compute_condition ? &rcondv[0] : nil,
			raw_data(work),
			&lwork,
			raw_data(iwork),
			&liwork,
			sort_eigenvalues ? raw_data(bwork) : nil,
			&info,
			1,
			1,
			1,
			1,
		)
	}

	return S, T_mat, VSL, VSR, alphar, alphai, beta, rconde, rcondv, sdim, info
}

m_schur_generalized_expert_c64 :: proc(
	A: ^Matrix(complex64),
	B: ^Matrix(complex64),
	compute_left: bool = true,
	compute_right: bool = true,
	sort_eigenvalues: bool = false,
	compute_condition: bool = true,
	select_fn: LAPACK_C_SELECT2 = nil,
	allocator := context.allocator,
) -> (
	S, T_mat: Matrix(complex64),
	VSL: Matrix(complex64),
	VSR: Matrix(complex64),
	alpha, beta: []complex64,
	rconde: []f32,
	rcondv: []f32,
	sdim: Blas_Int,
	info: Info,
) {
	n := Blas_Int(A.rows)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Copy A and B to S and T
	S = make_matrix(complex64, int(n), int(n), allocator)
	T_mat = make_matrix(complex64, int(n), int(n), allocator)
	copy(S.data, A.data[:n * n])
	copy(T_mat.data, B.data[:n * n])
	lds := Blas_Int(S.ld)
	ldt := Blas_Int(T_mat.ld)

	// Allocate eigenvalue arrays
	alpha = builtin.make([]complex64, n, allocator)
	beta = builtin.make([]complex64, n, allocator)

	// Set job parameters
	jobvsl_c := compute_left ? cstring("V") : cstring("N")
	jobvsr_c := compute_right ? cstring("V") : cstring("N")
	sort_c := sort_eigenvalues ? cstring("S") : cstring("N")

	sense_c: cstring
	if compute_condition && sort_eigenvalues {
		sense_c = "B"
	} else if compute_condition {
		sense_c = "E"
	} else {
		sense_c = "N"
	}

	// Allocate Schur vector matrices
	ldvsl := Blas_Int(1)
	ldvsr := Blas_Int(1)
	if compute_left {
		VSL = make_matrix(complex64, int(n), int(n), allocator)
		ldvsl = Blas_Int(VSL.ld)
	}
	if compute_right {
		VSR = make_matrix(complex64, int(n), int(n), allocator)
		ldvsr = Blas_Int(VSR.ld)
	}

	// Allocate condition number arrays
	if compute_condition {
		rconde = builtin.make([]f32, 2, allocator)
		rcondv = builtin.make([]f32, 2, allocator)
	}

	// Allocate work arrays
	var; bwork: []i32
	if sort_eigenvalues {
		bwork = builtin.make([]i32, n, allocator)
		defer builtin.delete(bwork)
	}

	rwork := builtin.make([]f32, 8 * n, allocator)
	defer builtin.delete(rwork)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: complex64
	liwork := Blas_Int(-1)
	iwork_query: Blas_Int

	lapack.cggesx_(
		jobvsl_c,
		jobvsr_c,
		sort_c,
		select_fn,
		sense_c,
		&n,
		raw_data(S.data),
		&lds,
		raw_data(T_mat.data),
		&ldt,
		&sdim,
		raw_data(alpha),
		raw_data(beta),
		compute_left ? raw_data(VSL.data) : nil,
		&ldvsl,
		compute_right ? raw_data(VSR.data) : nil,
		&ldvsr,
		compute_condition ? &rconde[0] : nil,
		compute_condition ? &rcondv[0] : nil,
		&work_query,
		&lwork,
		raw_data(rwork),
		&iwork_query,
		&liwork,
		sort_eigenvalues ? raw_data(bwork) : nil,
		&info,
		1,
		1,
		1,
		1,
	)

	// Allocate workspace
	lwork = Blas_Int(real(work_query))
	work := builtin.make([]complex64, lwork, allocator)
	defer builtin.delete(work)

	liwork = iwork_query
	iwork := builtin.make([]i32, liwork, allocator)
	defer builtin.delete(iwork)

	// Compute Schur decomposition
	lapack.cggesx_(
		jobvsl_c,
		jobvsr_c,
		sort_c,
		select_fn,
		sense_c,
		&n,
		raw_data(S.data),
		&lds,
		raw_data(T_mat.data),
		&ldt,
		&sdim,
		raw_data(alpha),
		raw_data(beta),
		compute_left ? raw_data(VSL.data) : nil,
		&ldvsl,
		compute_right ? raw_data(VSR.data) : nil,
		&ldvsr,
		compute_condition ? &rconde[0] : nil,
		compute_condition ? &rcondv[0] : nil,
		raw_data(work),
		&lwork,
		raw_data(rwork),
		raw_data(iwork),
		&liwork,
		sort_eigenvalues ? raw_data(bwork) : nil,
		&info,
		1,
		1,
		1,
		1,
	)

	return S, T_mat, VSL, VSR, alpha, beta, rconde, rcondv, sdim, info
}

m_schur_generalized_expert_c128 :: proc(
	A: ^Matrix(complex128),
	B: ^Matrix(complex128),
	compute_left: bool = true,
	compute_right: bool = true,
	sort_eigenvalues: bool = false,
	compute_condition: bool = true,
	select_fn: LAPACK_Z_SELECT2 = nil,
	allocator := context.allocator,
) -> (
	S, T_mat: Matrix(complex128),
	VSL: Matrix(complex128),
	VSR: Matrix(complex128),
	alpha, beta: []complex128,
	rconde: []f64,
	rcondv: []f64,
	sdim: Blas_Int,
	info: Info,
) {
	n := Blas_Int(A.rows)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Copy A and B to S and T
	S = make_matrix(complex128, int(n), int(n), allocator)
	T_mat = make_matrix(complex128, int(n), int(n), allocator)
	copy(S.data, A.data[:n * n])
	copy(T_mat.data, B.data[:n * n])
	lds := Blas_Int(S.ld)
	ldt := Blas_Int(T_mat.ld)

	// Allocate eigenvalue arrays
	alpha = builtin.make([]complex128, n, allocator)
	beta = builtin.make([]complex128, n, allocator)

	// Set job parameters
	jobvsl_c := compute_left ? cstring("V") : cstring("N")
	jobvsr_c := compute_right ? cstring("V") : cstring("N")
	sort_c := sort_eigenvalues ? cstring("S") : cstring("N")

	sense_c: cstring
	if compute_condition && sort_eigenvalues {
		sense_c = "B"
	} else if compute_condition {
		sense_c = "E"
	} else {
		sense_c = "N"
	}

	// Allocate Schur vector matrices
	ldvsl := Blas_Int(1)
	ldvsr := Blas_Int(1)
	if compute_left {
		VSL = make_matrix(complex128, int(n), int(n), allocator)
		ldvsl = Blas_Int(VSL.ld)
	}
	if compute_right {
		VSR = make_matrix(complex128, int(n), int(n), allocator)
		ldvsr = Blas_Int(VSR.ld)
	}

	// Allocate condition number arrays
	if compute_condition {
		rconde = builtin.make([]f64, 2, allocator)
		rcondv = builtin.make([]f64, 2, allocator)
	}

	// Allocate work arrays
	var; bwork: []i32
	if sort_eigenvalues {
		bwork = builtin.make([]i32, n, allocator)
		defer builtin.delete(bwork)
	}

	rwork := builtin.make([]f64, 8 * n, allocator)
	defer builtin.delete(rwork)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: complex128
	liwork := Blas_Int(-1)
	iwork_query: Blas_Int

	lapack.zggesx_(
		jobvsl_c,
		jobvsr_c,
		sort_c,
		select_fn,
		sense_c,
		&n,
		raw_data(S.data),
		&lds,
		raw_data(T_mat.data),
		&ldt,
		&sdim,
		raw_data(alpha),
		raw_data(beta),
		compute_left ? raw_data(VSL.data) : nil,
		&ldvsl,
		compute_right ? raw_data(VSR.data) : nil,
		&ldvsr,
		compute_condition ? &rconde[0] : nil,
		compute_condition ? &rcondv[0] : nil,
		&work_query,
		&lwork,
		raw_data(rwork),
		&iwork_query,
		&liwork,
		sort_eigenvalues ? raw_data(bwork) : nil,
		&info,
		1,
		1,
		1,
		1,
	)

	// Allocate workspace
	lwork = Blas_Int(real(work_query))
	work := builtin.make([]complex128, lwork, allocator)
	defer builtin.delete(work)

	liwork = iwork_query
	iwork := builtin.make([]i32, liwork, allocator)
	defer builtin.delete(iwork)

	// Compute Schur decomposition
	lapack.zggesx_(
		jobvsl_c,
		jobvsr_c,
		sort_c,
		select_fn,
		sense_c,
		&n,
		raw_data(S.data),
		&lds,
		raw_data(T_mat.data),
		&ldt,
		&sdim,
		raw_data(alpha),
		raw_data(beta),
		compute_left ? raw_data(VSL.data) : nil,
		&ldvsl,
		compute_right ? raw_data(VSR.data) : nil,
		&ldvsr,
		compute_condition ? &rconde[0] : nil,
		compute_condition ? &rcondv[0] : nil,
		raw_data(work),
		&lwork,
		raw_data(rwork),
		raw_data(iwork),
		&liwork,
		sort_eigenvalues ? raw_data(bwork) : nil,
		&info,
		1,
		1,
		1,
		1,
	)

	return S, T_mat, VSL, VSR, alpha, beta, rconde, rcondv, sdim, info
}

// ===================================================================================
// GENERALIZED EIGENVALUE PROBLEMS
// ===================================================================================

// Compute generalized eigenvalues and eigenvectors for matrix pair (A,B)
// Finds eigenvalues λ and eigenvectors v such that Av = λBv
m_eigen_generalized :: proc {
	m_eigen_generalized_real,
	m_eigen_generalized_c64,
	m_eigen_generalized_c128,
}

m_eigen_generalized_real :: proc(
	A: ^Matrix($T),
	B: ^Matrix(T),
	compute_left: bool = false, // Compute left eigenvectors
	compute_right: bool = true, // Compute right eigenvectors
	allocator := context.allocator,
) -> (
	alphar, alphai, beta: []T,
	VL: Matrix(T),
	VR: Matrix(T),
	info: Info, // Eigenvalues: λ = (alphar + i*alphai)/beta// Left eigenvectors// Right eigenvectors
) where T == f32 || T == f64 {
	n := Blas_Int(A.rows)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Allocate eigenvalue arrays
	alphar = builtin.make([]T, n, allocator)
	alphai = builtin.make([]T, n, allocator)
	beta = builtin.make([]T, n, allocator)

	// Set job parameters
	jobvl_c := compute_left ? cstring("V") : cstring("N")
	jobvr_c := compute_right ? cstring("V") : cstring("N")

	// Allocate eigenvector matrices
	ldvl := Blas_Int(1)
	ldvr := Blas_Int(1)
	if compute_left {
		VL = make_matrix(T, int(n), int(n), allocator)
		ldvl = Blas_Int(VL.ld)
	}
	if compute_right {
		VR = make_matrix(T, int(n), int(n), allocator)
		ldvr = Blas_Int(VR.ld)
	}

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: T

	when T == f32 {
		lapack.sggev_(
			jobvl_c,
			jobvr_c,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			raw_data(alphar),
			raw_data(alphai),
			raw_data(beta),
			compute_left ? raw_data(VL.data) : nil,
			&ldvl,
			compute_right ? raw_data(VR.data) : nil,
			&ldvr,
			&work_query,
			&lwork,
			&info,
			1,
			1,
		)
	} else {
		lapack.dggev_(
			jobvl_c,
			jobvr_c,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			raw_data(alphar),
			raw_data(alphai),
			raw_data(beta),
			compute_left ? raw_data(VL.data) : nil,
			&ldvl,
			compute_right ? raw_data(VR.data) : nil,
			&ldvr,
			&work_query,
			&lwork,
			&info,
			1,
			1,
		)
	}

	// Allocate workspace
	lwork = Blas_Int(work_query)
	work := builtin.make([]T, lwork, allocator)
	defer builtin.delete(work)

	// Compute eigenvalues and eigenvectors
	when T == f32 {
		lapack.sggev_(
			jobvl_c,
			jobvr_c,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			raw_data(alphar),
			raw_data(alphai),
			raw_data(beta),
			compute_left ? raw_data(VL.data) : nil,
			&ldvl,
			compute_right ? raw_data(VR.data) : nil,
			&ldvr,
			raw_data(work),
			&lwork,
			&info,
			1,
			1,
		)
	} else {
		lapack.dggev_(
			jobvl_c,
			jobvr_c,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			raw_data(alphar),
			raw_data(alphai),
			raw_data(beta),
			compute_left ? raw_data(VL.data) : nil,
			&ldvl,
			compute_right ? raw_data(VR.data) : nil,
			&ldvr,
			raw_data(work),
			&lwork,
			&info,
			1,
			1,
		)
	}

	return alphar, alphai, beta, VL, VR, info
}

m_eigen_generalized_c64 :: proc(
	A: ^Matrix(complex64),
	B: ^Matrix(complex64),
	compute_left: bool = false,
	compute_right: bool = true,
	allocator := context.allocator,
) -> (
	alpha, beta: []complex64,
	VL: Matrix(complex64),
	VR: Matrix(complex64),
	info: Info,
) {
	n := Blas_Int(A.rows)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Allocate eigenvalue arrays
	alpha = builtin.make([]complex64, n, allocator)
	beta = builtin.make([]complex64, n, allocator)

	// Set job parameters
	jobvl_c := compute_left ? cstring("V") : cstring("N")
	jobvr_c := compute_right ? cstring("V") : cstring("N")

	// Allocate eigenvector matrices
	ldvl := Blas_Int(1)
	ldvr := Blas_Int(1)
	if compute_left {
		VL = make_matrix(complex64, int(n), int(n), allocator)
		ldvl = Blas_Int(VL.ld)
	}
	if compute_right {
		VR = make_matrix(complex64, int(n), int(n), allocator)
		ldvr = Blas_Int(VR.ld)
	}

	// Allocate real workspace
	rwork := builtin.make([]f32, 8 * n, allocator)
	defer builtin.delete(rwork)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: complex64

	lapack.cggev_(
		jobvl_c,
		jobvr_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(alpha),
		raw_data(beta),
		compute_left ? raw_data(VL.data) : nil,
		&ldvl,
		compute_right ? raw_data(VR.data) : nil,
		&ldvr,
		&work_query,
		&lwork,
		raw_data(rwork),
		&info,
		1,
		1,
	)

	// Allocate workspace
	lwork = Blas_Int(real(work_query))
	work := builtin.make([]complex64, lwork, allocator)
	defer builtin.delete(work)

	// Compute eigenvalues and eigenvectors
	lapack.cggev_(
		jobvl_c,
		jobvr_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(alpha),
		raw_data(beta),
		compute_left ? raw_data(VL.data) : nil,
		&ldvl,
		compute_right ? raw_data(VR.data) : nil,
		&ldvr,
		raw_data(work),
		&lwork,
		raw_data(rwork),
		&info,
		1,
		1,
	)

	return alpha, beta, VL, VR, info
}

m_eigen_generalized_c128 :: proc(
	A: ^Matrix(complex128),
	B: ^Matrix(complex128),
	compute_left: bool = false,
	compute_right: bool = true,
	allocator := context.allocator,
) -> (
	alpha, beta: []complex128,
	VL: Matrix(complex128),
	VR: Matrix(complex128),
	info: Info,
) {
	n := Blas_Int(A.rows)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Allocate eigenvalue arrays
	alpha = builtin.make([]complex128, n, allocator)
	beta = builtin.make([]complex128, n, allocator)

	// Set job parameters
	jobvl_c := compute_left ? cstring("V") : cstring("N")
	jobvr_c := compute_right ? cstring("V") : cstring("N")

	// Allocate eigenvector matrices
	ldvl := Blas_Int(1)
	ldvr := Blas_Int(1)
	if compute_left {
		VL = make_matrix(complex128, int(n), int(n), allocator)
		ldvl = Blas_Int(VL.ld)
	}
	if compute_right {
		VR = make_matrix(complex128, int(n), int(n), allocator)
		ldvr = Blas_Int(VR.ld)
	}

	// Allocate real workspace
	rwork := builtin.make([]f64, 8 * n, allocator)
	defer builtin.delete(rwork)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: complex128

	lapack.zggev_(
		jobvl_c,
		jobvr_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(alpha),
		raw_data(beta),
		compute_left ? raw_data(VL.data) : nil,
		&ldvl,
		compute_right ? raw_data(VR.data) : nil,
		&ldvr,
		&work_query,
		&lwork,
		raw_data(rwork),
		&info,
		1,
		1,
	)

	// Allocate workspace
	lwork = Blas_Int(real(work_query))
	work := builtin.make([]complex128, lwork, allocator)
	defer builtin.delete(work)

	// Compute eigenvalues and eigenvectors
	lapack.zggev_(
		jobvl_c,
		jobvr_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(alpha),
		raw_data(beta),
		compute_left ? raw_data(VL.data) : nil,
		&ldvl,
		compute_right ? raw_data(VR.data) : nil,
		&ldvr,
		raw_data(work),
		&lwork,
		raw_data(rwork),
		&info,
		1,
		1,
	)

	return alpha, beta, VL, VR, info
}

// Compute generalized eigenvalues and eigenvectors with improved blocked algorithm
m_eigen_generalized_blocked :: proc {
	m_eigen_generalized_blocked_real,
	m_eigen_generalized_blocked_c64,
	m_eigen_generalized_blocked_c128,
}

m_eigen_generalized_blocked_real :: proc(
	A: ^Matrix($T),
	B: ^Matrix(T),
	compute_left: bool = false,
	compute_right: bool = true,
	allocator := context.allocator,
) -> (
	alphar, alphai, beta: []T,
	VL: Matrix(T),
	VR: Matrix(T),
	info: Info,
) where T == f32 ||
	T == f64 {
	n := Blas_Int(A.rows)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Allocate eigenvalue arrays
	alphar = builtin.make([]T, n, allocator)
	alphai = builtin.make([]T, n, allocator)
	beta = builtin.make([]T, n, allocator)

	// Set job parameters
	jobvl_c := compute_left ? cstring("V") : cstring("N")
	jobvr_c := compute_right ? cstring("V") : cstring("N")

	// Allocate eigenvector matrices
	ldvl := Blas_Int(1)
	ldvr := Blas_Int(1)
	if compute_left {
		VL = make_matrix(T, int(n), int(n), allocator)
		ldvl = Blas_Int(VL.ld)
	}
	if compute_right {
		VR = make_matrix(T, int(n), int(n), allocator)
		ldvr = Blas_Int(VR.ld)
	}

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: T

	when T == f32 {
		lapack.sggev3_(
			jobvl_c,
			jobvr_c,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			raw_data(alphar),
			raw_data(alphai),
			raw_data(beta),
			compute_left ? raw_data(VL.data) : nil,
			&ldvl,
			compute_right ? raw_data(VR.data) : nil,
			&ldvr,
			&work_query,
			&lwork,
			&info,
			1,
			1,
		)
	} else {
		lapack.dggev3_(
			jobvl_c,
			jobvr_c,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			raw_data(alphar),
			raw_data(alphai),
			raw_data(beta),
			compute_left ? raw_data(VL.data) : nil,
			&ldvl,
			compute_right ? raw_data(VR.data) : nil,
			&ldvr,
			&work_query,
			&lwork,
			&info,
			1,
			1,
		)
	}

	// Allocate workspace
	lwork = Blas_Int(work_query)
	work := builtin.make([]T, lwork, allocator)
	defer builtin.delete(work)

	// Compute eigenvalues and eigenvectors
	when T == f32 {
		lapack.sggev3_(
			jobvl_c,
			jobvr_c,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			raw_data(alphar),
			raw_data(alphai),
			raw_data(beta),
			compute_left ? raw_data(VL.data) : nil,
			&ldvl,
			compute_right ? raw_data(VR.data) : nil,
			&ldvr,
			raw_data(work),
			&lwork,
			&info,
			1,
			1,
		)
	} else {
		lapack.dggev3_(
			jobvl_c,
			jobvr_c,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			raw_data(alphar),
			raw_data(alphai),
			raw_data(beta),
			compute_left ? raw_data(VL.data) : nil,
			&ldvl,
			compute_right ? raw_data(VR.data) : nil,
			&ldvr,
			raw_data(work),
			&lwork,
			&info,
			1,
			1,
		)
	}

	return alphar, alphai, beta, VL, VR, info
}

m_eigen_generalized_blocked_c64 :: proc(
	A: ^Matrix(complex64),
	B: ^Matrix(complex64),
	compute_left: bool = false,
	compute_right: bool = true,
	allocator := context.allocator,
) -> (
	alpha, beta: []complex64,
	VL: Matrix(complex64),
	VR: Matrix(complex64),
	info: Info,
) {
	n := Blas_Int(A.rows)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Allocate eigenvalue arrays
	alpha = builtin.make([]complex64, n, allocator)
	beta = builtin.make([]complex64, n, allocator)

	// Set job parameters
	jobvl_c := compute_left ? cstring("V") : cstring("N")
	jobvr_c := compute_right ? cstring("V") : cstring("N")

	// Allocate eigenvector matrices
	ldvl := Blas_Int(1)
	ldvr := Blas_Int(1)
	if compute_left {
		VL = make_matrix(complex64, int(n), int(n), allocator)
		ldvl = Blas_Int(VL.ld)
	}
	if compute_right {
		VR = make_matrix(complex64, int(n), int(n), allocator)
		ldvr = Blas_Int(VR.ld)
	}

	// Allocate real workspace
	rwork := builtin.make([]f32, 8 * n, allocator)
	defer builtin.delete(rwork)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: complex64

	lapack.cggev3_(
		jobvl_c,
		jobvr_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(alpha),
		raw_data(beta),
		compute_left ? raw_data(VL.data) : nil,
		&ldvl,
		compute_right ? raw_data(VR.data) : nil,
		&ldvr,
		&work_query,
		&lwork,
		raw_data(rwork),
		&info,
		1,
		1,
	)

	// Allocate workspace
	lwork = Blas_Int(real(work_query))
	work := builtin.make([]complex64, lwork, allocator)
	defer builtin.delete(work)

	// Compute eigenvalues and eigenvectors
	lapack.cggev3_(
		jobvl_c,
		jobvr_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(alpha),
		raw_data(beta),
		compute_left ? raw_data(VL.data) : nil,
		&ldvl,
		compute_right ? raw_data(VR.data) : nil,
		&ldvr,
		raw_data(work),
		&lwork,
		raw_data(rwork),
		&info,
		1,
		1,
	)

	return alpha, beta, VL, VR, info
}

m_eigen_generalized_blocked_c128 :: proc(
	A: ^Matrix(complex128),
	B: ^Matrix(complex128),
	compute_left: bool = false,
	compute_right: bool = true,
	allocator := context.allocator,
) -> (
	alpha, beta: []complex128,
	VL: Matrix(complex128),
	VR: Matrix(complex128),
	info: Info,
) {
	n := Blas_Int(A.rows)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Allocate eigenvalue arrays
	alpha = builtin.make([]complex128, n, allocator)
	beta = builtin.make([]complex128, n, allocator)

	// Set job parameters
	jobvl_c := compute_left ? cstring("V") : cstring("N")
	jobvr_c := compute_right ? cstring("V") : cstring("N")

	// Allocate eigenvector matrices
	ldvl := Blas_Int(1)
	ldvr := Blas_Int(1)
	if compute_left {
		VL = make_matrix(complex128, int(n), int(n), allocator)
		ldvl = Blas_Int(VL.ld)
	}
	if compute_right {
		VR = make_matrix(complex128, int(n), int(n), allocator)
		ldvr = Blas_Int(VR.ld)
	}

	// Allocate real workspace
	rwork := builtin.make([]f64, 8 * n, allocator)
	defer builtin.delete(rwork)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: complex128

	lapack.zggev3_(
		jobvl_c,
		jobvr_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(alpha),
		raw_data(beta),
		compute_left ? raw_data(VL.data) : nil,
		&ldvl,
		compute_right ? raw_data(VR.data) : nil,
		&ldvr,
		&work_query,
		&lwork,
		raw_data(rwork),
		&info,
		1,
		1,
	)

	// Allocate workspace
	lwork = Blas_Int(real(work_query))
	work := builtin.make([]complex128, lwork, allocator)
	defer builtin.delete(work)

	// Compute eigenvalues and eigenvectors
	lapack.zggev3_(
		jobvl_c,
		jobvr_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(alpha),
		raw_data(beta),
		compute_left ? raw_data(VL.data) : nil,
		&ldvl,
		compute_right ? raw_data(VR.data) : nil,
		&ldvr,
		raw_data(work),
		&lwork,
		raw_data(rwork),
		&info,
		1,
		1,
	)

	return alpha, beta, VL, VR, info
}

// ===================================================================================
// EXPERT GENERALIZED EIGENVALUE PROBLEM WITH BALANCING AND CONDITION NUMBERS
// ===================================================================================

// Compute generalized eigenvalues and eigenvectors with expert options
// Includes balancing, condition numbers, and norms
m_eigen_generalized_expert :: proc {
	m_eigen_generalized_expert_real,
	m_eigen_generalized_expert_c64,
	m_eigen_generalized_expert_c128,
}

m_eigen_generalized_expert_real :: proc(
	A: ^Matrix($T),
	B: ^Matrix(T),
	balance: bool = true, // Apply balancing
	compute_left: bool = false, // Compute left eigenvectors
	compute_right: bool = true, // Compute right eigenvectors
	compute_condition: bool = true, // Compute condition numbers
	allocator := context.allocator,
) -> (
	alphar, alphai, beta: []T,
	VL: Matrix(T),
	VR: Matrix(T),
	ilo, ihi: Blas_Int,
	lscale, rscale: []T,
	abnrm, bbnrm: T,
	rconde, rcondv: []T,
	info: Info, // Balancing indices// Scale factors// Norms of balanced matrices// Condition numbers
) where T == f32 || T == f64 {
	n := Blas_Int(A.rows)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Allocate eigenvalue arrays
	alphar = builtin.make([]T, n, allocator)
	alphai = builtin.make([]T, n, allocator)
	beta = builtin.make([]T, n, allocator)

	// Set job parameters
	balanc_c := balance ? cstring("B") : cstring("N")
	jobvl_c := compute_left ? cstring("V") : cstring("N")
	jobvr_c := compute_right ? cstring("V") : cstring("N")

	sense_c: cstring
	if compute_condition && (compute_left || compute_right) {
		sense_c = "B" // Both eigenvalue and eigenvector condition numbers
	} else if compute_condition {
		sense_c = "E" // Only eigenvalue condition numbers
	} else {
		sense_c = "N" // No condition numbers
	}

	// Allocate eigenvector matrices
	ldvl := Blas_Int(1)
	ldvr := Blas_Int(1)
	if compute_left {
		VL = make_matrix(T, int(n), int(n), allocator)
		ldvl = Blas_Int(VL.ld)
	}
	if compute_right {
		VR = make_matrix(T, int(n), int(n), allocator)
		ldvr = Blas_Int(VR.ld)
	}

	// Allocate scale arrays
	lscale = builtin.make([]T, n, allocator)
	rscale = builtin.make([]T, n, allocator)

	// Allocate condition arrays
	if compute_condition {
		rconde = builtin.make([]T, n, allocator)
		rcondv = builtin.make([]T, n, allocator)
	}

	// Allocate boolean work array
	bwork := builtin.make([]i32, n, allocator)
	defer builtin.delete(bwork)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: T
	iwork := builtin.make([]i32, n + 2, allocator)
	defer builtin.delete(iwork)

	when T == f32 {
		lapack.sggevx_(
			balanc_c,
			jobvl_c,
			jobvr_c,
			sense_c,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			raw_data(alphar),
			raw_data(alphai),
			raw_data(beta),
			compute_left ? raw_data(VL.data) : nil,
			&ldvl,
			compute_right ? raw_data(VR.data) : nil,
			&ldvr,
			&ilo,
			&ihi,
			raw_data(lscale),
			raw_data(rscale),
			&abnrm,
			&bbnrm,
			compute_condition ? raw_data(rconde) : nil,
			compute_condition ? raw_data(rcondv) : nil,
			&work_query,
			&lwork,
			raw_data(iwork),
			raw_data(bwork),
			&info,
			1,
			1,
			1,
			1,
		)
	} else {
		lapack.dggevx_(
			balanc_c,
			jobvl_c,
			jobvr_c,
			sense_c,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			raw_data(alphar),
			raw_data(alphai),
			raw_data(beta),
			compute_left ? raw_data(VL.data) : nil,
			&ldvl,
			compute_right ? raw_data(VR.data) : nil,
			&ldvr,
			&ilo,
			&ihi,
			raw_data(lscale),
			raw_data(rscale),
			&abnrm,
			&bbnrm,
			compute_condition ? raw_data(rconde) : nil,
			compute_condition ? raw_data(rcondv) : nil,
			&work_query,
			&lwork,
			raw_data(iwork),
			raw_data(bwork),
			&info,
			1,
			1,
			1,
			1,
		)
	}

	// Allocate workspace
	lwork = Blas_Int(work_query)
	work := builtin.make([]T, lwork, allocator)
	defer builtin.delete(work)

	// Compute eigenvalues and eigenvectors
	when T == f32 {
		lapack.sggevx_(
			balanc_c,
			jobvl_c,
			jobvr_c,
			sense_c,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			raw_data(alphar),
			raw_data(alphai),
			raw_data(beta),
			compute_left ? raw_data(VL.data) : nil,
			&ldvl,
			compute_right ? raw_data(VR.data) : nil,
			&ldvr,
			&ilo,
			&ihi,
			raw_data(lscale),
			raw_data(rscale),
			&abnrm,
			&bbnrm,
			compute_condition ? raw_data(rconde) : nil,
			compute_condition ? raw_data(rcondv) : nil,
			raw_data(work),
			&lwork,
			raw_data(iwork),
			raw_data(bwork),
			&info,
			1,
			1,
			1,
			1,
		)
	} else {
		lapack.dggevx_(
			balanc_c,
			jobvl_c,
			jobvr_c,
			sense_c,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			raw_data(alphar),
			raw_data(alphai),
			raw_data(beta),
			compute_left ? raw_data(VL.data) : nil,
			&ldvl,
			compute_right ? raw_data(VR.data) : nil,
			&ldvr,
			&ilo,
			&ihi,
			raw_data(lscale),
			raw_data(rscale),
			&abnrm,
			&bbnrm,
			compute_condition ? raw_data(rconde) : nil,
			compute_condition ? raw_data(rcondv) : nil,
			raw_data(work),
			&lwork,
			raw_data(iwork),
			raw_data(bwork),
			&info,
			1,
			1,
			1,
			1,
		)
	}

	return alphar,
		alphai,
		beta,
		VL,
		VR,
		ilo,
		ihi,
		lscale,
		rscale,
		abnrm,
		bbnrm,
		rconde,
		rcondv,
		info
}

m_eigen_generalized_expert_c64 :: proc(
	A: ^Matrix(complex64),
	B: ^Matrix(complex64),
	balance: bool = true,
	compute_left: bool = false,
	compute_right: bool = true,
	compute_condition: bool = true,
	allocator := context.allocator,
) -> (
	alpha, beta: []complex64,
	VL: Matrix(complex64),
	VR: Matrix(complex64),
	ilo, ihi: Blas_Int,
	lscale, rscale: []f32,
	abnrm, bbnrm: f32,
	rconde, rcondv: []f32,
	info: Info,
) {
	n := Blas_Int(A.rows)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Allocate eigenvalue arrays
	alpha = builtin.make([]complex64, n, allocator)
	beta = builtin.make([]complex64, n, allocator)

	// Set job parameters
	balanc_c := balance ? cstring("B") : cstring("N")
	jobvl_c := compute_left ? cstring("V") : cstring("N")
	jobvr_c := compute_right ? cstring("V") : cstring("N")

	sense_c: cstring
	if compute_condition && (compute_left || compute_right) {
		sense_c = "B"
	} else if compute_condition {
		sense_c = "E"
	} else {
		sense_c = "N"
	}

	// Allocate eigenvector matrices
	ldvl := Blas_Int(1)
	ldvr := Blas_Int(1)
	if compute_left {
		VL = make_matrix(complex64, int(n), int(n), allocator)
		ldvl = Blas_Int(VL.ld)
	}
	if compute_right {
		VR = make_matrix(complex64, int(n), int(n), allocator)
		ldvr = Blas_Int(VR.ld)
	}

	// Allocate scale arrays
	lscale = builtin.make([]f32, n, allocator)
	rscale = builtin.make([]f32, n, allocator)

	// Allocate condition arrays
	if compute_condition {
		rconde = builtin.make([]f32, n, allocator)
		rcondv = builtin.make([]f32, n, allocator)
	}

	// Allocate work arrays
	bwork := builtin.make([]i32, n, allocator)
	defer builtin.delete(bwork)

	rwork := builtin.make([]f32, 6 * n, allocator)
	defer builtin.delete(rwork)

	iwork := builtin.make([]i32, n + 2, allocator)
	defer builtin.delete(iwork)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: complex64

	lapack.cggevx_(
		balanc_c,
		jobvl_c,
		jobvr_c,
		sense_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(alpha),
		raw_data(beta),
		compute_left ? raw_data(VL.data) : nil,
		&ldvl,
		compute_right ? raw_data(VR.data) : nil,
		&ldvr,
		&ilo,
		&ihi,
		raw_data(lscale),
		raw_data(rscale),
		&abnrm,
		&bbnrm,
		compute_condition ? raw_data(rconde) : nil,
		compute_condition ? raw_data(rcondv) : nil,
		&work_query,
		&lwork,
		raw_data(rwork),
		raw_data(iwork),
		raw_data(bwork),
		&info,
		1,
		1,
		1,
		1,
	)

	// Allocate workspace
	lwork = Blas_Int(real(work_query))
	work := builtin.make([]complex64, lwork, allocator)
	defer builtin.delete(work)

	// Compute eigenvalues and eigenvectors
	lapack.cggevx_(
		balanc_c,
		jobvl_c,
		jobvr_c,
		sense_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(alpha),
		raw_data(beta),
		compute_left ? raw_data(VL.data) : nil,
		&ldvl,
		compute_right ? raw_data(VR.data) : nil,
		&ldvr,
		&ilo,
		&ihi,
		raw_data(lscale),
		raw_data(rscale),
		&abnrm,
		&bbnrm,
		compute_condition ? raw_data(rconde) : nil,
		compute_condition ? raw_data(rcondv) : nil,
		raw_data(work),
		&lwork,
		raw_data(rwork),
		raw_data(iwork),
		raw_data(bwork),
		&info,
		1,
		1,
		1,
		1,
	)

	return alpha, beta, VL, VR, ilo, ihi, lscale, rscale, abnrm, bbnrm, rconde, rcondv, info
}

m_eigen_generalized_expert_c128 :: proc(
	A: ^Matrix(complex128),
	B: ^Matrix(complex128),
	balance: bool = true,
	compute_left: bool = false,
	compute_right: bool = true,
	compute_condition: bool = true,
	allocator := context.allocator,
) -> (
	alpha, beta: []complex128,
	VL: Matrix(complex128),
	VR: Matrix(complex128),
	ilo, ihi: Blas_Int,
	lscale, rscale: []f64,
	abnrm, bbnrm: f64,
	rconde, rcondv: []f64,
	info: Info,
) {
	n := Blas_Int(A.rows)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Allocate eigenvalue arrays
	alpha = builtin.make([]complex128, n, allocator)
	beta = builtin.make([]complex128, n, allocator)

	// Set job parameters
	balanc_c := balance ? cstring("B") : cstring("N")
	jobvl_c := compute_left ? cstring("V") : cstring("N")
	jobvr_c := compute_right ? cstring("V") : cstring("N")

	sense_c: cstring
	if compute_condition && (compute_left || compute_right) {
		sense_c = "B"
	} else if compute_condition {
		sense_c = "E"
	} else {
		sense_c = "N"
	}

	// Allocate eigenvector matrices
	ldvl := Blas_Int(1)
	ldvr := Blas_Int(1)
	if compute_left {
		VL = make_matrix(complex128, int(n), int(n), allocator)
		ldvl = Blas_Int(VL.ld)
	}
	if compute_right {
		VR = make_matrix(complex128, int(n), int(n), allocator)
		ldvr = Blas_Int(VR.ld)
	}

	// Allocate scale arrays
	lscale = builtin.make([]f64, n, allocator)
	rscale = builtin.make([]f64, n, allocator)

	// Allocate condition arrays
	if compute_condition {
		rconde = builtin.make([]f64, n, allocator)
		rcondv = builtin.make([]f64, n, allocator)
	}

	// Allocate work arrays
	bwork := builtin.make([]i32, n, allocator)
	defer builtin.delete(bwork)

	rwork := builtin.make([]f64, 6 * n, allocator)
	defer builtin.delete(rwork)

	iwork := builtin.make([]i32, n + 2, allocator)
	defer builtin.delete(iwork)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: complex128

	lapack.zggevx_(
		balanc_c,
		jobvl_c,
		jobvr_c,
		sense_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(alpha),
		raw_data(beta),
		compute_left ? raw_data(VL.data) : nil,
		&ldvl,
		compute_right ? raw_data(VR.data) : nil,
		&ldvr,
		&ilo,
		&ihi,
		raw_data(lscale),
		raw_data(rscale),
		&abnrm,
		&bbnrm,
		compute_condition ? raw_data(rconde) : nil,
		compute_condition ? raw_data(rcondv) : nil,
		&work_query,
		&lwork,
		raw_data(rwork),
		raw_data(iwork),
		raw_data(bwork),
		&info,
		1,
		1,
		1,
		1,
	)

	// Allocate workspace
	lwork = Blas_Int(real(work_query))
	work := builtin.make([]complex128, lwork, allocator)
	defer builtin.delete(work)

	// Compute eigenvalues and eigenvectors
	lapack.zggevx_(
		balanc_c,
		jobvl_c,
		jobvr_c,
		sense_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(alpha),
		raw_data(beta),
		compute_left ? raw_data(VL.data) : nil,
		&ldvl,
		compute_right ? raw_data(VR.data) : nil,
		&ldvr,
		&ilo,
		&ihi,
		raw_data(lscale),
		raw_data(rscale),
		&abnrm,
		&bbnrm,
		compute_condition ? raw_data(rconde) : nil,
		compute_condition ? raw_data(rcondv) : nil,
		raw_data(work),
		&lwork,
		raw_data(rwork),
		raw_data(iwork),
		raw_data(bwork),
		&info,
		1,
		1,
		1,
		1,
	)

	return alpha, beta, VL, VR, ilo, ihi, lscale, rscale, abnrm, bbnrm, rconde, rcondv, info
}

// ===================================================================================
// GENERALIZED LINEAR MODEL PROBLEM
// ===================================================================================

// Solve generalized linear model problem: minimize ||y||_2 subject to d = Ax + By
m_glm_generalized :: proc {
	m_glm_generalized_real,
	m_glm_generalized_c64,
	m_glm_generalized_c128,
}

m_glm_generalized_real :: proc(
	A: ^Matrix($T),
	B: ^Matrix(T),
	d: []T,
	allocator := context.allocator,
) -> (
	x, y: []T,
	info: Info,
) where T == f32 ||
	T == f64 {
	n := Blas_Int(A.cols) // Number of variables
	m := Blas_Int(A.rows) // Number of equations
	p := Blas_Int(B.cols) // Dimension of y

	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Allocate solution vectors
	x = builtin.make([]T, n, allocator)
	y = builtin.make([]T, p, allocator)

	// Copy d vector as it will be modified
	d_copy := builtin.make([]T, len(d), allocator)
	copy(d_copy, d)
	defer builtin.delete(d_copy)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: T

	when T == f32 {
		lapack.sggglm_(
			&m,
			&n,
			&p,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			raw_data(d_copy),
			raw_data(x),
			raw_data(y),
			&work_query,
			&lwork,
			&info,
		)
	} else {
		lapack.dggglm_(
			&m,
			&n,
			&p,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			raw_data(d_copy),
			raw_data(x),
			raw_data(y),
			&work_query,
			&lwork,
			&info,
		)
	}

	// Allocate workspace
	lwork = Blas_Int(work_query)
	work := builtin.make([]T, lwork, allocator)
	defer builtin.delete(work)

	// Solve generalized linear model
	when T == f32 {
		lapack.sggglm_(
			&m,
			&n,
			&p,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			raw_data(d_copy),
			raw_data(x),
			raw_data(y),
			raw_data(work),
			&lwork,
			&info,
		)
	} else {
		lapack.dggglm_(
			&m,
			&n,
			&p,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			raw_data(d_copy),
			raw_data(x),
			raw_data(y),
			raw_data(work),
			&lwork,
			&info,
		)
	}

	return x, y, info
}

m_glm_generalized_c64 :: proc(
	A: ^Matrix(complex64),
	B: ^Matrix(complex64),
	d: []complex64,
	allocator := context.allocator,
) -> (
	x, y: []complex64,
	info: Info,
) {
	n := Blas_Int(A.cols) // Number of variables
	m := Blas_Int(A.rows) // Number of equations
	p := Blas_Int(B.cols) // Dimension of y

	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Allocate solution vectors
	x = builtin.make([]complex64, n, allocator)
	y = builtin.make([]complex64, p, allocator)

	// Copy d vector as it will be modified
	d_copy := builtin.make([]complex64, len(d), allocator)
	copy(d_copy, d)
	defer builtin.delete(d_copy)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: complex64

	lapack.cggglm_(
		&m,
		&n,
		&p,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(d_copy),
		raw_data(x),
		raw_data(y),
		&work_query,
		&lwork,
		&info,
	)

	// Allocate workspace
	lwork = Blas_Int(real(work_query))
	work := builtin.make([]complex64, lwork, allocator)
	defer builtin.delete(work)

	// Solve generalized linear model
	lapack.cggglm_(
		&m,
		&n,
		&p,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(d_copy),
		raw_data(x),
		raw_data(y),
		raw_data(work),
		&lwork,
		&info,
	)

	return x, y, info
}

m_glm_generalized_c128 :: proc(
	A: ^Matrix(complex128),
	B: ^Matrix(complex128),
	d: []complex128,
	allocator := context.allocator,
) -> (
	x, y: []complex128,
	info: Info,
) {
	n := Blas_Int(A.cols) // Number of variables
	m := Blas_Int(A.rows) // Number of equations
	p := Blas_Int(B.cols) // Dimension of y

	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Allocate solution vectors
	x = builtin.make([]complex128, n, allocator)
	y = builtin.make([]complex128, p, allocator)

	// Copy d vector as it will be modified
	d_copy := builtin.make([]complex128, len(d), allocator)
	copy(d_copy, d)
	defer builtin.delete(d_copy)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: complex128

	lapack.zggglm_(
		&m,
		&n,
		&p,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(d_copy),
		raw_data(x),
		raw_data(y),
		&work_query,
		&lwork,
		&info,
	)

	// Allocate workspace
	lwork = Blas_Int(real(work_query))
	work := builtin.make([]complex128, lwork, allocator)
	defer builtin.delete(work)

	// Solve generalized linear model
	lapack.zggglm_(
		&m,
		&n,
		&p,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(d_copy),
		raw_data(x),
		raw_data(y),
		raw_data(work),
		&lwork,
		&info,
	)

	return x, y, info
}

// ===================================================================================
// GENERALIZED HESSENBERG REDUCTION
// ===================================================================================

// Reduce generalized matrix pair to Hessenberg-triangular form
m_hessenberg_generalized :: proc {
	m_hessenberg_generalized_real,
	m_hessenberg_generalized_c64,
	m_hessenberg_generalized_c128,
}

m_hessenberg_generalized_real :: proc(
	A: ^Matrix($T),
	B: ^Matrix(T),
	compute_left: bool = true,
	compute_right: bool = true,
	ilo: Blas_Int = 1,
	ihi: Blas_Int = 0,
	allocator := context.allocator,
) -> (
	Q: Matrix(T),
	Z: Matrix(T),
	tau: []T,
	info: Info,
) where T == f32 ||
	T == f64 {
	n := Blas_Int(A.rows)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Use full matrix range if not specified
	ihi_actual := ihi == 0 ? n : ihi

	// Set job parameters
	compq_c := compute_left ? cstring("V") : cstring("N")
	compz_c := compute_right ? cstring("V") : cstring("N")

	// Allocate transformation matrices
	ldq := Blas_Int(1)
	ldz := Blas_Int(1)
	if compute_left {
		Q = make_matrix(T, int(n), int(n), allocator)
		ldq = Blas_Int(Q.ld)
		// Initialize Q to identity
		for i in 0 ..< n {
			Q.data[i * n + i] = 1
		}
	}
	if compute_right {
		Z = make_matrix(T, int(n), int(n), allocator)
		ldz = Blas_Int(Z.ld)
		// Initialize Z to identity
		for i in 0 ..< n {
			Z.data[i * n + i] = 1
		}
	}

	// Allocate tau array
	tau = builtin.make([]T, n - 1, allocator)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: T

	when T == f32 {
		lapack.sgghrd_(
			compq_c,
			compz_c,
			&n,
			&ilo,
			&ihi_actual,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			compute_left ? raw_data(Q.data) : nil,
			&ldq,
			compute_right ? raw_data(Z.data) : nil,
			&ldz,
			&info,
			1,
			1,
		)
	} else {
		lapack.dgghrd_(
			compq_c,
			compz_c,
			&n,
			&ilo,
			&ihi_actual,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			compute_left ? raw_data(Q.data) : nil,
			&ldq,
			compute_right ? raw_data(Z.data) : nil,
			&ldz,
			&info,
			1,
			1,
		)
	}

	return Q, Z, tau, info
}

m_hessenberg_generalized_c64 :: proc(
	A: ^Matrix(complex64),
	B: ^Matrix(complex64),
	compute_left: bool = true,
	compute_right: bool = true,
	ilo: Blas_Int = 1,
	ihi: Blas_Int = 0,
	allocator := context.allocator,
) -> (
	Q: Matrix(complex64),
	Z: Matrix(complex64),
	tau: []complex64,
	info: Info,
) {
	n := Blas_Int(A.rows)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Use full matrix range if not specified
	ihi_actual := ihi == 0 ? n : ihi

	// Set job parameters
	compq_c := compute_left ? cstring("V") : cstring("N")
	compz_c := compute_right ? cstring("V") : cstring("N")

	// Allocate transformation matrices
	ldq := Blas_Int(1)
	ldz := Blas_Int(1)
	if compute_left {
		Q = make_matrix(complex64, int(n), int(n), allocator)
		ldq = Blas_Int(Q.ld)
		// Initialize Q to identity
		for i in 0 ..< n {
			Q.data[i * n + i] = 1
		}
	}
	if compute_right {
		Z = make_matrix(complex64, int(n), int(n), allocator)
		ldz = Blas_Int(Z.ld)
		// Initialize Z to identity
		for i in 0 ..< n {
			Z.data[i * n + i] = 1
		}
	}

	// Allocate tau array
	tau = builtin.make([]complex64, n - 1, allocator)

	lapack.cgghrd_(
		compq_c,
		compz_c,
		&n,
		&ilo,
		&ihi_actual,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		compute_left ? raw_data(Q.data) : nil,
		&ldq,
		compute_right ? raw_data(Z.data) : nil,
		&ldz,
		&info,
		1,
		1,
	)

	return Q, Z, tau, info
}

m_hessenberg_generalized_c128 :: proc(
	A: ^Matrix(complex128),
	B: ^Matrix(complex128),
	compute_left: bool = true,
	compute_right: bool = true,
	ilo: Blas_Int = 1,
	ihi: Blas_Int = 0,
	allocator := context.allocator,
) -> (
	Q: Matrix(complex128),
	Z: Matrix(complex128),
	tau: []complex128,
	info: Info,
) {
	n := Blas_Int(A.rows)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Use full matrix range if not specified
	ihi_actual := ihi == 0 ? n : ihi

	// Set job parameters
	compq_c := compute_left ? cstring("V") : cstring("N")
	compz_c := compute_right ? cstring("V") : cstring("N")

	// Allocate transformation matrices
	ldq := Blas_Int(1)
	ldz := Blas_Int(1)
	if compute_left {
		Q = make_matrix(complex128, int(n), int(n), allocator)
		ldq = Blas_Int(Q.ld)
		// Initialize Q to identity
		for i in 0 ..< n {
			Q.data[i * n + i] = 1
		}
	}
	if compute_right {
		Z = make_matrix(complex128, int(n), int(n), allocator)
		ldz = Blas_Int(Z.ld)
		// Initialize Z to identity
		for i in 0 ..< n {
			Z.data[i * n + i] = 1
		}
	}

	// Allocate tau array
	tau = builtin.make([]complex128, n - 1, allocator)

	lapack.zgghrd_(
		compq_c,
		compz_c,
		&n,
		&ilo,
		&ihi_actual,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		compute_left ? raw_data(Q.data) : nil,
		&ldq,
		compute_right ? raw_data(Z.data) : nil,
		&ldz,
		&info,
		1,
		1,
	)

	return Q, Z, tau, info
}

// Blocked version using improved algorithm
m_hessenberg_generalized_blocked :: proc {
	m_hessenberg_generalized_blocked_real,
	m_hessenberg_generalized_blocked_c64,
	m_hessenberg_generalized_blocked_c128,
}

m_hessenberg_generalized_blocked_real :: proc(
	A: ^Matrix($T),
	B: ^Matrix(T),
	compute_left: bool = true,
	compute_right: bool = true,
	ilo: Blas_Int = 1,
	ihi: Blas_Int = 0,
	allocator := context.allocator,
) -> (
	Q: Matrix(T),
	Z: Matrix(T),
	tau: []T,
	info: Info,
) where T == f32 ||
	T == f64 {
	n := Blas_Int(A.rows)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Use full matrix range if not specified
	ihi_actual := ihi == 0 ? n : ihi

	// Set job parameters
	compq_c := compute_left ? cstring("V") : cstring("N")
	compz_c := compute_right ? cstring("V") : cstring("N")

	// Allocate transformation matrices
	ldq := Blas_Int(1)
	ldz := Blas_Int(1)
	if compute_left {
		Q = make_matrix(T, int(n), int(n), allocator)
		ldq = Blas_Int(Q.ld)
		// Initialize Q to identity
		for i in 0 ..< n {
			Q.data[i * n + i] = 1
		}
	}
	if compute_right {
		Z = make_matrix(T, int(n), int(n), allocator)
		ldz = Blas_Int(Z.ld)
		// Initialize Z to identity
		for i in 0 ..< n {
			Z.data[i * n + i] = 1
		}
	}

	// Allocate tau array
	tau = builtin.make([]T, n - 1, allocator)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: T

	when T == f32 {
		lapack.sgghd3_(
			compq_c,
			compz_c,
			&n,
			&ilo,
			&ihi_actual,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			compute_left ? raw_data(Q.data) : nil,
			&ldq,
			compute_right ? raw_data(Z.data) : nil,
			&ldz,
			&work_query,
			&lwork,
			&info,
			1,
			1,
		)
	} else {
		lapack.dgghd3_(
			compq_c,
			compz_c,
			&n,
			&ilo,
			&ihi_actual,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			compute_left ? raw_data(Q.data) : nil,
			&ldq,
			compute_right ? raw_data(Z.data) : nil,
			&ldz,
			&work_query,
			&lwork,
			&info,
			1,
			1,
		)
	}

	// Allocate workspace
	lwork = Blas_Int(work_query)
	work := builtin.make([]T, lwork, allocator)
	defer builtin.delete(work)

	// Perform reduction
	when T == f32 {
		lapack.sgghd3_(
			compq_c,
			compz_c,
			&n,
			&ilo,
			&ihi_actual,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			compute_left ? raw_data(Q.data) : nil,
			&ldq,
			compute_right ? raw_data(Z.data) : nil,
			&ldz,
			raw_data(work),
			&lwork,
			&info,
			1,
			1,
		)
	} else {
		lapack.dgghd3_(
			compq_c,
			compz_c,
			&n,
			&ilo,
			&ihi_actual,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			compute_left ? raw_data(Q.data) : nil,
			&ldq,
			compute_right ? raw_data(Z.data) : nil,
			&ldz,
			raw_data(work),
			&lwork,
			&info,
			1,
			1,
		)
	}

	return Q, Z, tau, info
}

m_hessenberg_generalized_blocked_c64 :: proc(
	A: ^Matrix(complex64),
	B: ^Matrix(complex64),
	compute_left: bool = true,
	compute_right: bool = true,
	ilo: Blas_Int = 1,
	ihi: Blas_Int = 0,
	allocator := context.allocator,
) -> (
	Q: Matrix(complex64),
	Z: Matrix(complex64),
	tau: []complex64,
	info: Info,
) {
	n := Blas_Int(A.rows)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Use full matrix range if not specified
	ihi_actual := ihi == 0 ? n : ihi

	// Set job parameters
	compq_c := compute_left ? cstring("V") : cstring("N")
	compz_c := compute_right ? cstring("V") : cstring("N")

	// Allocate transformation matrices
	ldq := Blas_Int(1)
	ldz := Blas_Int(1)
	if compute_left {
		Q = make_matrix(complex64, int(n), int(n), allocator)
		ldq = Blas_Int(Q.ld)
		// Initialize Q to identity
		for i in 0 ..< n {
			Q.data[i * n + i] = 1
		}
	}
	if compute_right {
		Z = make_matrix(complex64, int(n), int(n), allocator)
		ldz = Blas_Int(Z.ld)
		// Initialize Z to identity
		for i in 0 ..< n {
			Z.data[i * n + i] = 1
		}
	}

	// Allocate tau array
	tau = builtin.make([]complex64, n - 1, allocator)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: complex64

	lapack.cgghd3_(
		compq_c,
		compz_c,
		&n,
		&ilo,
		&ihi_actual,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		compute_left ? raw_data(Q.data) : nil,
		&ldq,
		compute_right ? raw_data(Z.data) : nil,
		&ldz,
		&work_query,
		&lwork,
		&info,
		1,
		1,
	)

	// Allocate workspace
	lwork = Blas_Int(real(work_query))
	work := builtin.make([]complex64, lwork, allocator)
	defer builtin.delete(work)

	// Perform reduction
	lapack.cgghd3_(
		compq_c,
		compz_c,
		&n,
		&ilo,
		&ihi_actual,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		compute_left ? raw_data(Q.data) : nil,
		&ldq,
		compute_right ? raw_data(Z.data) : nil,
		&ldz,
		raw_data(work),
		&lwork,
		&info,
		1,
		1,
	)

	return Q, Z, tau, info
}

m_hessenberg_generalized_blocked_c128 :: proc(
	A: ^Matrix(complex128),
	B: ^Matrix(complex128),
	compute_left: bool = true,
	compute_right: bool = true,
	ilo: Blas_Int = 1,
	ihi: Blas_Int = 0,
	allocator := context.allocator,
) -> (
	Q: Matrix(complex128),
	Z: Matrix(complex128),
	tau: []complex128,
	info: Info,
) {
	n := Blas_Int(A.rows)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Use full matrix range if not specified
	ihi_actual := ihi == 0 ? n : ihi

	// Set job parameters
	compq_c := compute_left ? cstring("V") : cstring("N")
	compz_c := compute_right ? cstring("V") : cstring("N")

	// Allocate transformation matrices
	ldq := Blas_Int(1)
	ldz := Blas_Int(1)
	if compute_left {
		Q = make_matrix(complex128, int(n), int(n), allocator)
		ldq = Blas_Int(Q.ld)
		// Initialize Q to identity
		for i in 0 ..< n {
			Q.data[i * n + i] = 1
		}
	}
	if compute_right {
		Z = make_matrix(complex128, int(n), int(n), allocator)
		ldz = Blas_Int(Z.ld)
		// Initialize Z to identity
		for i in 0 ..< n {
			Z.data[i * n + i] = 1
		}
	}

	// Allocate tau array
	tau = builtin.make([]complex128, n - 1, allocator)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: complex128

	lapack.zgghd3_(
		compq_c,
		compz_c,
		&n,
		&ilo,
		&ihi_actual,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		compute_left ? raw_data(Q.data) : nil,
		&ldq,
		compute_right ? raw_data(Z.data) : nil,
		&ldz,
		&work_query,
		&lwork,
		&info,
		1,
		1,
	)

	// Allocate workspace
	lwork = Blas_Int(real(work_query))
	work := builtin.make([]complex128, lwork, allocator)
	defer builtin.delete(work)

	// Perform reduction
	lapack.zgghd3_(
		compq_c,
		compz_c,
		&n,
		&ilo,
		&ihi_actual,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		compute_left ? raw_data(Q.data) : nil,
		&ldq,
		compute_right ? raw_data(Z.data) : nil,
		&ldz,
		raw_data(work),
		&lwork,
		&info,
		1,
		1,
	)

	return Q, Z, tau, info
}

// ===================================================================================
// GENERALIZED LEAST SQUARES WITH EQUALITY CONSTRAINTS
// ===================================================================================

// Solve least squares with equality constraints: minimize ||c - Ax||_2 subject to Bx = d
m_lse_constrained :: proc {
	m_lse_constrained_real,
	m_lse_constrained_c64,
	m_lse_constrained_c128,
}

m_lse_constrained_real :: proc(
	A: ^Matrix($T),
	B: ^Matrix(T),
	c: []T,
	d: []T,
	allocator := context.allocator,
) -> (
	x: []T,
	info: Info,
) where T == f32 ||
	T == f64 {
	m := Blas_Int(A.rows) // Number of rows in A
	n := Blas_Int(A.cols) // Number of variables
	p := Blas_Int(B.rows) // Number of equality constraints

	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Allocate solution vector
	x = builtin.make([]T, n, allocator)

	// Copy input vectors as they will be modified
	c_copy := builtin.make([]T, len(c), allocator)
	d_copy := builtin.make([]T, len(d), allocator)
	copy(c_copy, c)
	copy(d_copy, d)
	defer builtin.delete(c_copy)
	defer builtin.delete(d_copy)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: T

	when T == f32 {
		lapack.sgglse_(
			&m,
			&n,
			&p,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			raw_data(c_copy),
			raw_data(d_copy),
			raw_data(x),
			&work_query,
			&lwork,
			&info,
		)
	} else {
		lapack.dgglse_(
			&m,
			&n,
			&p,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			raw_data(c_copy),
			raw_data(d_copy),
			raw_data(x),
			&work_query,
			&lwork,
			&info,
		)
	}

	// Allocate workspace
	lwork = Blas_Int(work_query)
	work := builtin.make([]T, lwork, allocator)
	defer builtin.delete(work)

	// Solve constrained least squares
	when T == f32 {
		lapack.sgglse_(
			&m,
			&n,
			&p,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			raw_data(c_copy),
			raw_data(d_copy),
			raw_data(x),
			raw_data(work),
			&lwork,
			&info,
		)
	} else {
		lapack.dgglse_(
			&m,
			&n,
			&p,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			raw_data(c_copy),
			raw_data(d_copy),
			raw_data(x),
			raw_data(work),
			&lwork,
			&info,
		)
	}

	return x, info
}

m_lse_constrained_c64 :: proc(
	A: ^Matrix(complex64),
	B: ^Matrix(complex64),
	c: []complex64,
	d: []complex64,
	allocator := context.allocator,
) -> (
	x: []complex64,
	info: Info,
) {
	m := Blas_Int(A.rows) // Number of rows in A
	n := Blas_Int(A.cols) // Number of variables
	p := Blas_Int(B.rows) // Number of equality constraints

	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Allocate solution vector
	x = builtin.make([]complex64, n, allocator)

	// Copy input vectors as they will be modified
	c_copy := builtin.make([]complex64, len(c), allocator)
	d_copy := builtin.make([]complex64, len(d), allocator)
	copy(c_copy, c)
	copy(d_copy, d)
	defer builtin.delete(c_copy)
	defer builtin.delete(d_copy)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: complex64

	lapack.cgglse_(
		&m,
		&n,
		&p,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(c_copy),
		raw_data(d_copy),
		raw_data(x),
		&work_query,
		&lwork,
		&info,
	)

	// Allocate workspace
	lwork = Blas_Int(real(work_query))
	work := builtin.make([]complex64, lwork, allocator)
	defer builtin.delete(work)

	// Solve constrained least squares
	lapack.cgglse_(
		&m,
		&n,
		&p,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(c_copy),
		raw_data(d_copy),
		raw_data(x),
		raw_data(work),
		&lwork,
		&info,
	)

	return x, info
}

m_lse_constrained_c128 :: proc(
	A: ^Matrix(complex128),
	B: ^Matrix(complex128),
	c: []complex128,
	d: []complex128,
	allocator := context.allocator,
) -> (
	x: []complex128,
	info: Info,
) {
	m := Blas_Int(A.rows) // Number of rows in A
	n := Blas_Int(A.cols) // Number of variables
	p := Blas_Int(B.rows) // Number of equality constraints

	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Allocate solution vector
	x = builtin.make([]complex128, n, allocator)

	// Copy input vectors as they will be modified
	c_copy := builtin.make([]complex128, len(c), allocator)
	d_copy := builtin.make([]complex128, len(d), allocator)
	copy(c_copy, c)
	copy(d_copy, d)
	defer builtin.delete(c_copy)
	defer builtin.delete(d_copy)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: complex128

	lapack.zgglse_(
		&m,
		&n,
		&p,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(c_copy),
		raw_data(d_copy),
		raw_data(x),
		&work_query,
		&lwork,
		&info,
	)

	// Allocate workspace
	lwork = Blas_Int(real(work_query))
	work := builtin.make([]complex128, lwork, allocator)
	defer builtin.delete(work)

	// Solve constrained least squares
	lapack.zgglse_(
		&m,
		&n,
		&p,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(c_copy),
		raw_data(d_copy),
		raw_data(x),
		raw_data(work),
		&lwork,
		&info,
	)

	return x, info
}

// ===================================================================================
// GENERALIZED QR FACTORIZATION
// ===================================================================================

// Compute generalized QR factorization: A = Q*R, B = Q*T*Z
m_qr_generalized :: proc {
	m_qr_generalized_real,
	m_qr_generalized_c64,
	m_qr_generalized_c128,
}

m_qr_generalized_real :: proc(
	A: ^Matrix($T),
	B: ^Matrix(T),
	allocator := context.allocator,
) -> (
	taua, taub: []T,
	info: Info,
) where T == f32 ||
	T == f64 {
	n := Blas_Int(A.rows)
	m := Blas_Int(A.cols)
	p := Blas_Int(B.cols)

	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Allocate tau arrays
	taua = builtin.make([]T, min(n, m), allocator)
	taub = builtin.make([]T, min(n, p), allocator)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: T

	when T == f32 {
		lapack.sggqrf_(
			&n,
			&m,
			&p,
			raw_data(A.data),
			&lda,
			raw_data(taua),
			raw_data(B.data),
			&ldb,
			raw_data(taub),
			&work_query,
			&lwork,
			&info,
		)
	} else {
		lapack.dggqrf_(
			&n,
			&m,
			&p,
			raw_data(A.data),
			&lda,
			raw_data(taua),
			raw_data(B.data),
			&ldb,
			raw_data(taub),
			&work_query,
			&lwork,
			&info,
		)
	}

	// Allocate workspace
	lwork = Blas_Int(work_query)
	work := builtin.make([]T, lwork, allocator)
	defer builtin.delete(work)

	// Compute generalized QR factorization
	when T == f32 {
		lapack.sggqrf_(
			&n,
			&m,
			&p,
			raw_data(A.data),
			&lda,
			raw_data(taua),
			raw_data(B.data),
			&ldb,
			raw_data(taub),
			raw_data(work),
			&lwork,
			&info,
		)
	} else {
		lapack.dggqrf_(
			&n,
			&m,
			&p,
			raw_data(A.data),
			&lda,
			raw_data(taua),
			raw_data(B.data),
			&ldb,
			raw_data(taub),
			raw_data(work),
			&lwork,
			&info,
		)
	}

	return taua, taub, info
}

m_qr_generalized_c64 :: proc(
	A: ^Matrix(complex64),
	B: ^Matrix(complex64),
	allocator := context.allocator,
) -> (
	taua, taub: []complex64,
	info: Info,
) {
	n := Blas_Int(A.rows)
	m := Blas_Int(A.cols)
	p := Blas_Int(B.cols)

	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Allocate tau arrays
	taua = builtin.make([]complex64, min(n, m), allocator)
	taub = builtin.make([]complex64, min(n, p), allocator)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: complex64

	lapack.cggqrf_(
		&n,
		&m,
		&p,
		raw_data(A.data),
		&lda,
		raw_data(taua),
		raw_data(B.data),
		&ldb,
		raw_data(taub),
		&work_query,
		&lwork,
		&info,
	)

	// Allocate workspace
	lwork = Blas_Int(real(work_query))
	work := builtin.make([]complex64, lwork, allocator)
	defer builtin.delete(work)

	// Compute generalized QR factorization
	lapack.cggqrf_(
		&n,
		&m,
		&p,
		raw_data(A.data),
		&lda,
		raw_data(taua),
		raw_data(B.data),
		&ldb,
		raw_data(taub),
		raw_data(work),
		&lwork,
		&info,
	)

	return taua, taub, info
}

m_qr_generalized_c128 :: proc(
	A: ^Matrix(complex128),
	B: ^Matrix(complex128),
	allocator := context.allocator,
) -> (
	taua, taub: []complex128,
	info: Info,
) {
	n := Blas_Int(A.rows)
	m := Blas_Int(A.cols)
	p := Blas_Int(B.cols)

	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Allocate tau arrays
	taua = builtin.make([]complex128, min(n, m), allocator)
	taub = builtin.make([]complex128, min(n, p), allocator)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: complex128

	lapack.zggqrf_(
		&n,
		&m,
		&p,
		raw_data(A.data),
		&lda,
		raw_data(taua),
		raw_data(B.data),
		&ldb,
		raw_data(taub),
		&work_query,
		&lwork,
		&info,
	)

	// Allocate workspace
	lwork = Blas_Int(real(work_query))
	work := builtin.make([]complex128, lwork, allocator)
	defer builtin.delete(work)

	// Compute generalized QR factorization
	lapack.zggqrf_(
		&n,
		&m,
		&p,
		raw_data(A.data),
		&lda,
		raw_data(taua),
		raw_data(B.data),
		&ldb,
		raw_data(taub),
		raw_data(work),
		&lwork,
		&info,
	)

	return taua, taub, info
}

// ===================================================================================
// GENERALIZED RQ FACTORIZATION
// ===================================================================================

// Compute generalized RQ factorization: A = R*Q, B = Z*T*Q
m_rq_generalized :: proc {
	m_rq_generalized_real,
	m_rq_generalized_c64,
	m_rq_generalized_c128,
}

m_rq_generalized_real :: proc(
	A: ^Matrix($T),
	B: ^Matrix(T),
	allocator := context.allocator,
) -> (
	taua, taub: []T,
	info: Info,
) where T == f32 ||
	T == f64 {
	m := Blas_Int(A.rows)
	p := Blas_Int(A.cols) // This is p in the RQ context
	n := Blas_Int(B.cols)

	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Allocate tau arrays
	taua = builtin.make([]T, min(m, p), allocator)
	taub = builtin.make([]T, min(p, n), allocator)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: T

	when T == f32 {
		lapack.sggrqf_(
			&m,
			&p,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(taua),
			raw_data(B.data),
			&ldb,
			raw_data(taub),
			&work_query,
			&lwork,
			&info,
		)
	} else {
		lapack.dggrqf_(
			&m,
			&p,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(taua),
			raw_data(B.data),
			&ldb,
			raw_data(taub),
			&work_query,
			&lwork,
			&info,
		)
	}

	// Allocate workspace
	lwork = Blas_Int(work_query)
	work := builtin.make([]T, lwork, allocator)
	defer builtin.delete(work)

	// Compute generalized RQ factorization
	when T == f32 {
		lapack.sggrqf_(
			&m,
			&p,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(taua),
			raw_data(B.data),
			&ldb,
			raw_data(taub),
			raw_data(work),
			&lwork,
			&info,
		)
	} else {
		lapack.dggrqf_(
			&m,
			&p,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(taua),
			raw_data(B.data),
			&ldb,
			raw_data(taub),
			raw_data(work),
			&lwork,
			&info,
		)
	}

	return taua, taub, info
}

m_rq_generalized_c64 :: proc(
	A: ^Matrix(complex64),
	B: ^Matrix(complex64),
	allocator := context.allocator,
) -> (
	taua, taub: []complex64,
	info: Info,
) {
	m := Blas_Int(A.rows)
	p := Blas_Int(A.cols) // This is p in the RQ context
	n := Blas_Int(B.cols)

	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Allocate tau arrays
	taua = builtin.make([]complex64, min(m, p), allocator)
	taub = builtin.make([]complex64, min(p, n), allocator)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: complex64

	lapack.cggrqf_(
		&m,
		&p,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(taua),
		raw_data(B.data),
		&ldb,
		raw_data(taub),
		&work_query,
		&lwork,
		&info,
	)

	// Allocate workspace
	lwork = Blas_Int(real(work_query))
	work := builtin.make([]complex64, lwork, allocator)
	defer builtin.delete(work)

	// Compute generalized RQ factorization
	lapack.cggrqf_(
		&m,
		&p,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(taua),
		raw_data(B.data),
		&ldb,
		raw_data(taub),
		raw_data(work),
		&lwork,
		&info,
	)

	return taua, taub, info
}

m_rq_generalized_c128 :: proc(
	A: ^Matrix(complex128),
	B: ^Matrix(complex128),
	allocator := context.allocator,
) -> (
	taua, taub: []complex128,
	info: Info,
) {
	m := Blas_Int(A.rows)
	p := Blas_Int(A.cols) // This is p in the RQ context
	n := Blas_Int(B.cols)

	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Allocate tau arrays
	taua = builtin.make([]complex128, min(m, p), allocator)
	taub = builtin.make([]complex128, min(p, n), allocator)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: complex128

	lapack.zggrqf_(
		&m,
		&p,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(taua),
		raw_data(B.data),
		&ldb,
		raw_data(taub),
		&work_query,
		&lwork,
		&info,
	)

	// Allocate workspace
	lwork = Blas_Int(real(work_query))
	work := builtin.make([]complex128, lwork, allocator)
	defer builtin.delete(work)

	// Compute generalized RQ factorization
	lapack.zggrqf_(
		&m,
		&p,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(taua),
		raw_data(B.data),
		&ldb,
		raw_data(taub),
		raw_data(work),
		&lwork,
		&info,
	)

	return taua, taub, info
}

// ===================================================================================
// GENERALIZED SINGULAR VALUE DECOMPOSITION
// ===================================================================================

// Compute generalized SVD: U^H*A*Q = D1*[0 R], V^H*B*Q = D2*[0 R]
m_gsvd :: proc {
	m_gsvd_real,
	m_gsvd_c64,
	m_gsvd_c128,
}

m_gsvd_real :: proc(
	A: ^Matrix($T),
	B: ^Matrix(T),
	compute_u: bool = true,
	compute_v: bool = true,
	compute_q: bool = true,
	allocator := context.allocator,
) -> (
	alpha, beta: []T,
	U: Matrix(T),
	V: Matrix(T),
	Q: Matrix(T),
	k, l: Blas_Int,
	info: Info,
) where T == f32 ||
	T == f64 {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	p := Blas_Int(B.rows)

	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Set job parameters
	jobu_c := compute_u ? cstring("U") : cstring("N")
	jobv_c := compute_v ? cstring("V") : cstring("N")
	jobq_c := compute_q ? cstring("Q") : cstring("N")

	// Allocate singular value arrays
	alpha = builtin.make([]T, n, allocator)
	beta = builtin.make([]T, n, allocator)

	// Allocate transformation matrices
	ldu := Blas_Int(1)
	ldv := Blas_Int(1)
	ldq := Blas_Int(1)

	if compute_u {
		U = make_matrix(T, int(m), int(m), allocator)
		ldu = Blas_Int(U.ld)
	}
	if compute_v {
		V = make_matrix(T, int(p), int(p), allocator)
		ldv = Blas_Int(V.ld)
	}
	if compute_q {
		Q = make_matrix(T, int(n), int(n), allocator)
		ldq = Blas_Int(Q.ld)
	}

	// Allocate workspace
	work := builtin.make([]T, max(3 * n, m, p) + n, allocator)
	defer builtin.delete(work)

	iwork := builtin.make([]i32, n, allocator)
	defer builtin.delete(iwork)

	when T == f32 {
		lapack.sggsvd_(
			jobu_c,
			jobv_c,
			jobq_c,
			&m,
			&n,
			&p,
			&k,
			&l,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			raw_data(alpha),
			raw_data(beta),
			compute_u ? raw_data(U.data) : nil,
			&ldu,
			compute_v ? raw_data(V.data) : nil,
			&ldv,
			compute_q ? raw_data(Q.data) : nil,
			&ldq,
			raw_data(work),
			raw_data(iwork),
			&info,
			1,
			1,
			1,
		)
	} else {
		lapack.dggsvd_(
			jobu_c,
			jobv_c,
			jobq_c,
			&m,
			&n,
			&p,
			&k,
			&l,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			raw_data(alpha),
			raw_data(beta),
			compute_u ? raw_data(U.data) : nil,
			&ldu,
			compute_v ? raw_data(V.data) : nil,
			&ldv,
			compute_q ? raw_data(Q.data) : nil,
			&ldq,
			raw_data(work),
			raw_data(iwork),
			&info,
			1,
			1,
			1,
		)
	}

	return alpha, beta, U, V, Q, k, l, info
}

m_gsvd_c64 :: proc(
	A: ^Matrix(complex64),
	B: ^Matrix(complex64),
	compute_u: bool = true,
	compute_v: bool = true,
	compute_q: bool = true,
	allocator := context.allocator,
) -> (
	alpha, beta: []f32,
	U: Matrix(complex64),
	V: Matrix(complex64),
	Q: Matrix(complex64),
	k, l: Blas_Int,
	info: Info,
) {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	p := Blas_Int(B.rows)

	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Set job parameters
	jobu_c := compute_u ? cstring("U") : cstring("N")
	jobv_c := compute_v ? cstring("V") : cstring("N")
	jobq_c := compute_q ? cstring("Q") : cstring("N")

	// Allocate singular value arrays (real for complex matrices)
	alpha = builtin.make([]f32, n, allocator)
	beta = builtin.make([]f32, n, allocator)

	// Allocate transformation matrices
	ldu := Blas_Int(1)
	ldv := Blas_Int(1)
	ldq := Blas_Int(1)

	if compute_u {
		U = make_matrix(complex64, int(m), int(m), allocator)
		ldu = Blas_Int(U.ld)
	}
	if compute_v {
		V = make_matrix(complex64, int(p), int(p), allocator)
		ldv = Blas_Int(V.ld)
	}
	if compute_q {
		Q = make_matrix(complex64, int(n), int(n), allocator)
		ldq = Blas_Int(Q.ld)
	}

	// Allocate workspace
	work := builtin.make([]complex64, max(3 * n, m, p) + n, allocator)
	defer builtin.delete(work)

	rwork := builtin.make([]f32, 2 * n, allocator)
	defer builtin.delete(rwork)

	iwork := builtin.make([]i32, n, allocator)
	defer builtin.delete(iwork)

	lapack.cggsvd_(
		jobu_c,
		jobv_c,
		jobq_c,
		&m,
		&n,
		&p,
		&k,
		&l,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(alpha),
		raw_data(beta),
		compute_u ? raw_data(U.data) : nil,
		&ldu,
		compute_v ? raw_data(V.data) : nil,
		&ldv,
		compute_q ? raw_data(Q.data) : nil,
		&ldq,
		raw_data(work),
		raw_data(rwork),
		raw_data(iwork),
		&info,
		1,
		1,
		1,
	)

	return alpha, beta, U, V, Q, k, l, info
}

m_gsvd_c128 :: proc(
	A: ^Matrix(complex128),
	B: ^Matrix(complex128),
	compute_u: bool = true,
	compute_v: bool = true,
	compute_q: bool = true,
	allocator := context.allocator,
) -> (
	alpha, beta: []f64,
	U: Matrix(complex128),
	V: Matrix(complex128),
	Q: Matrix(complex128),
	k, l: Blas_Int,
	info: Info,
) {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	p := Blas_Int(B.rows)

	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Set job parameters
	jobu_c := compute_u ? cstring("U") : cstring("N")
	jobv_c := compute_v ? cstring("V") : cstring("N")
	jobq_c := compute_q ? cstring("Q") : cstring("N")

	// Allocate singular value arrays (real for complex matrices)
	alpha = builtin.make([]f64, n, allocator)
	beta = builtin.make([]f64, n, allocator)

	// Allocate transformation matrices
	ldu := Blas_Int(1)
	ldv := Blas_Int(1)
	ldq := Blas_Int(1)

	if compute_u {
		U = make_matrix(complex128, int(m), int(m), allocator)
		ldu = Blas_Int(U.ld)
	}
	if compute_v {
		V = make_matrix(complex128, int(p), int(p), allocator)
		ldv = Blas_Int(V.ld)
	}
	if compute_q {
		Q = make_matrix(complex128, int(n), int(n), allocator)
		ldq = Blas_Int(Q.ld)
	}

	// Allocate workspace
	work := builtin.make([]complex128, max(3 * n, m, p) + n, allocator)
	defer builtin.delete(work)

	rwork := builtin.make([]f64, 2 * n, allocator)
	defer builtin.delete(rwork)

	iwork := builtin.make([]i32, n, allocator)
	defer builtin.delete(iwork)

	lapack.zggsvd_(
		jobu_c,
		jobv_c,
		jobq_c,
		&m,
		&n,
		&p,
		&k,
		&l,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(alpha),
		raw_data(beta),
		compute_u ? raw_data(U.data) : nil,
		&ldu,
		compute_v ? raw_data(V.data) : nil,
		&ldv,
		compute_q ? raw_data(Q.data) : nil,
		&ldq,
		raw_data(work),
		raw_data(rwork),
		raw_data(iwork),
		&info,
		1,
		1,
		1,
	)

	return alpha, beta, U, V, Q, k, l, info
}

// Blocked/improved version of generalized SVD
m_gsvd_blocked :: proc {
	m_gsvd_blocked_real,
	m_gsvd_blocked_c64,
	m_gsvd_blocked_c128,
}

m_gsvd_blocked_real :: proc(
	A: ^Matrix($T),
	B: ^Matrix(T),
	compute_u: bool = true,
	compute_v: bool = true,
	compute_q: bool = true,
	allocator := context.allocator,
) -> (
	alpha, beta: []T,
	U: Matrix(T),
	V: Matrix(T),
	Q: Matrix(T),
	k, l: Blas_Int,
	info: Info,
) where T == f32 ||
	T == f64 {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	p := Blas_Int(B.rows)

	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Set job parameters
	jobu_c := compute_u ? cstring("U") : cstring("N")
	jobv_c := compute_v ? cstring("V") : cstring("N")
	jobq_c := compute_q ? cstring("Q") : cstring("N")

	// Allocate singular value arrays
	alpha = builtin.make([]T, n, allocator)
	beta = builtin.make([]T, n, allocator)

	// Allocate transformation matrices
	ldu := Blas_Int(1)
	ldv := Blas_Int(1)
	ldq := Blas_Int(1)

	if compute_u {
		U = make_matrix(T, int(m), int(m), allocator)
		ldu = Blas_Int(U.ld)
	}
	if compute_v {
		V = make_matrix(T, int(p), int(p), allocator)
		ldv = Blas_Int(V.ld)
	}
	if compute_q {
		Q = make_matrix(T, int(n), int(n), allocator)
		ldq = Blas_Int(Q.ld)
	}

	// Allocate integer workspace
	iwork := builtin.make([]i32, n, allocator)
	defer builtin.delete(iwork)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: T

	when T == f32 {
		lapack.sggsvd3_(
			jobu_c,
			jobv_c,
			jobq_c,
			&m,
			&n,
			&p,
			&k,
			&l,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			raw_data(alpha),
			raw_data(beta),
			compute_u ? raw_data(U.data) : nil,
			&ldu,
			compute_v ? raw_data(V.data) : nil,
			&ldv,
			compute_q ? raw_data(Q.data) : nil,
			&ldq,
			&work_query,
			&lwork,
			raw_data(iwork),
			&info,
			1,
			1,
			1,
		)
	} else {
		lapack.dggsvd3_(
			jobu_c,
			jobv_c,
			jobq_c,
			&m,
			&n,
			&p,
			&k,
			&l,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			raw_data(alpha),
			raw_data(beta),
			compute_u ? raw_data(U.data) : nil,
			&ldu,
			compute_v ? raw_data(V.data) : nil,
			&ldv,
			compute_q ? raw_data(Q.data) : nil,
			&ldq,
			&work_query,
			&lwork,
			raw_data(iwork),
			&info,
			1,
			1,
			1,
		)
	}

	// Allocate workspace
	lwork = Blas_Int(work_query)
	work := builtin.make([]T, lwork, allocator)
	defer builtin.delete(work)

	// Compute generalized SVD
	when T == f32 {
		lapack.sggsvd3_(
			jobu_c,
			jobv_c,
			jobq_c,
			&m,
			&n,
			&p,
			&k,
			&l,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			raw_data(alpha),
			raw_data(beta),
			compute_u ? raw_data(U.data) : nil,
			&ldu,
			compute_v ? raw_data(V.data) : nil,
			&ldv,
			compute_q ? raw_data(Q.data) : nil,
			&ldq,
			raw_data(work),
			&lwork,
			raw_data(iwork),
			&info,
			1,
			1,
			1,
		)
	} else {
		lapack.dggsvd3_(
			jobu_c,
			jobv_c,
			jobq_c,
			&m,
			&n,
			&p,
			&k,
			&l,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			raw_data(alpha),
			raw_data(beta),
			compute_u ? raw_data(U.data) : nil,
			&ldu,
			compute_v ? raw_data(V.data) : nil,
			&ldv,
			compute_q ? raw_data(Q.data) : nil,
			&ldq,
			raw_data(work),
			&lwork,
			raw_data(iwork),
			&info,
			1,
			1,
			1,
		)
	}

	return alpha, beta, U, V, Q, k, l, info
}

m_gsvd_blocked_c64 :: proc(
	A: ^Matrix(complex64),
	B: ^Matrix(complex64),
	compute_u: bool = true,
	compute_v: bool = true,
	compute_q: bool = true,
	allocator := context.allocator,
) -> (
	alpha, beta: []f32,
	U: Matrix(complex64),
	V: Matrix(complex64),
	Q: Matrix(complex64),
	k, l: Blas_Int,
	info: Info,
) {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	p := Blas_Int(B.rows)

	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Set job parameters
	jobu_c := compute_u ? cstring("U") : cstring("N")
	jobv_c := compute_v ? cstring("V") : cstring("N")
	jobq_c := compute_q ? cstring("Q") : cstring("N")

	// Allocate singular value arrays (real for complex matrices)
	alpha = builtin.make([]f32, n, allocator)
	beta = builtin.make([]f32, n, allocator)

	// Allocate transformation matrices
	ldu := Blas_Int(1)
	ldv := Blas_Int(1)
	ldq := Blas_Int(1)

	if compute_u {
		U = make_matrix(complex64, int(m), int(m), allocator)
		ldu = Blas_Int(U.ld)
	}
	if compute_v {
		V = make_matrix(complex64, int(p), int(p), allocator)
		ldv = Blas_Int(V.ld)
	}
	if compute_q {
		Q = make_matrix(complex64, int(n), int(n), allocator)
		ldq = Blas_Int(Q.ld)
	}

	// Allocate real and integer workspace
	rwork := builtin.make([]f32, 2 * n, allocator)
	defer builtin.delete(rwork)

	iwork := builtin.make([]i32, n, allocator)
	defer builtin.delete(iwork)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: complex64

	lapack.cggsvd3_(
		jobu_c,
		jobv_c,
		jobq_c,
		&m,
		&n,
		&p,
		&k,
		&l,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(alpha),
		raw_data(beta),
		compute_u ? raw_data(U.data) : nil,
		&ldu,
		compute_v ? raw_data(V.data) : nil,
		&ldv,
		compute_q ? raw_data(Q.data) : nil,
		&ldq,
		&work_query,
		&lwork,
		raw_data(rwork),
		raw_data(iwork),
		&info,
		1,
		1,
		1,
	)

	// Allocate workspace
	lwork = Blas_Int(real(work_query))
	work := builtin.make([]complex64, lwork, allocator)
	defer builtin.delete(work)

	// Compute generalized SVD
	lapack.cggsvd3_(
		jobu_c,
		jobv_c,
		jobq_c,
		&m,
		&n,
		&p,
		&k,
		&l,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(alpha),
		raw_data(beta),
		compute_u ? raw_data(U.data) : nil,
		&ldu,
		compute_v ? raw_data(V.data) : nil,
		&ldv,
		compute_q ? raw_data(Q.data) : nil,
		&ldq,
		raw_data(work),
		&lwork,
		raw_data(rwork),
		raw_data(iwork),
		&info,
		1,
		1,
		1,
	)

	return alpha, beta, U, V, Q, k, l, info
}

m_gsvd_blocked_c128 :: proc(
	A: ^Matrix(complex128),
	B: ^Matrix(complex128),
	compute_u: bool = true,
	compute_v: bool = true,
	compute_q: bool = true,
	allocator := context.allocator,
) -> (
	alpha, beta: []f64,
	U: Matrix(complex128),
	V: Matrix(complex128),
	Q: Matrix(complex128),
	k, l: Blas_Int,
	info: Info,
) {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	p := Blas_Int(B.rows)

	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Set job parameters
	jobu_c := compute_u ? cstring("U") : cstring("N")
	jobv_c := compute_v ? cstring("V") : cstring("N")
	jobq_c := compute_q ? cstring("Q") : cstring("N")

	// Allocate singular value arrays (real for complex matrices)
	alpha = builtin.make([]f64, n, allocator)
	beta = builtin.make([]f64, n, allocator)

	// Allocate transformation matrices
	ldu := Blas_Int(1)
	ldv := Blas_Int(1)
	ldq := Blas_Int(1)

	if compute_u {
		U = make_matrix(complex128, int(m), int(m), allocator)
		ldu = Blas_Int(U.ld)
	}
	if compute_v {
		V = make_matrix(complex128, int(p), int(p), allocator)
		ldv = Blas_Int(V.ld)
	}
	if compute_q {
		Q = make_matrix(complex128, int(n), int(n), allocator)
		ldq = Blas_Int(Q.ld)
	}

	// Allocate real and integer workspace
	rwork := builtin.make([]f64, 2 * n, allocator)
	defer builtin.delete(rwork)

	iwork := builtin.make([]i32, n, allocator)
	defer builtin.delete(iwork)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: complex128

	lapack.zggsvd3_(
		jobu_c,
		jobv_c,
		jobq_c,
		&m,
		&n,
		&p,
		&k,
		&l,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(alpha),
		raw_data(beta),
		compute_u ? raw_data(U.data) : nil,
		&ldu,
		compute_v ? raw_data(V.data) : nil,
		&ldv,
		compute_q ? raw_data(Q.data) : nil,
		&ldq,
		&work_query,
		&lwork,
		raw_data(rwork),
		raw_data(iwork),
		&info,
		1,
		1,
		1,
	)

	// Allocate workspace
	lwork = Blas_Int(real(work_query))
	work := builtin.make([]complex128, lwork, allocator)
	defer builtin.delete(work)

	// Compute generalized SVD
	lapack.zggsvd3_(
		jobu_c,
		jobv_c,
		jobq_c,
		&m,
		&n,
		&p,
		&k,
		&l,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(alpha),
		raw_data(beta),
		compute_u ? raw_data(U.data) : nil,
		&ldu,
		compute_v ? raw_data(V.data) : nil,
		&ldv,
		compute_q ? raw_data(Q.data) : nil,
		&ldq,
		raw_data(work),
		&lwork,
		raw_data(rwork),
		raw_data(iwork),
		&info,
		1,
		1,
		1,
	)

	return alpha, beta, U, V, Q, k, l, info
}

// ===================================================================================
// GENERALIZED SVD PREPROCESSING
// ===================================================================================

// Preprocessing for generalized SVD - reduce to standard form
m_gsvd_preprocess :: proc {
	m_gsvd_preprocess_real,
	m_gsvd_preprocess_c64,
	m_gsvd_preprocess_c128,
}

m_gsvd_preprocess_real :: proc(
	A: ^Matrix($T),
	B: ^Matrix(T),
	tola: T = 0,
	tolb: T = 0,
	compute_u: bool = true,
	compute_v: bool = true,
	compute_q: bool = true,
	allocator := context.allocator,
) -> (
	U: Matrix(T),
	V: Matrix(T),
	Q: Matrix(T),
	k, l: Blas_Int,
	tau: []T,
	info: Info,
) where T == f32 ||
	T == f64 {
	m := Blas_Int(A.rows)
	p := Blas_Int(A.cols)
	n := Blas_Int(B.cols)

	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Set job parameters
	jobu_c := compute_u ? cstring("U") : cstring("N")
	jobv_c := compute_v ? cstring("V") : cstring("N")
	jobq_c := compute_q ? cstring("Q") : cstring("N")

	// Use machine precision if tolerances not specified
	tola_actual := tola == 0 ? T(2.2e-16) : tola // Approximate machine epsilon for f64
	tolb_actual := tolb == 0 ? T(2.2e-16) : tolb
	when T == f32 {
		tola_actual = tola == 0 ? T(1.2e-7) : tola // Approximate machine epsilon for f32
		tolb_actual = tolb == 0 ? T(1.2e-7) : tolb
	}

	// Allocate transformation matrices
	ldu := Blas_Int(1)
	ldv := Blas_Int(1)
	ldq := Blas_Int(1)

	if compute_u {
		U = make_matrix(T, int(m), int(m), allocator)
		ldu = Blas_Int(U.ld)
	}
	if compute_v {
		V = make_matrix(T, int(p), int(p), allocator)
		ldv = Blas_Int(V.ld)
	}
	if compute_q {
		Q = make_matrix(T, int(n), int(n), allocator)
		ldq = Blas_Int(Q.ld)
	}

	// Allocate tau and workspace
	tau = builtin.make([]T, n, allocator)
	work := builtin.make([]T, max(3 * n, m, p), allocator)
	defer builtin.delete(work)

	iwork := builtin.make([]i32, n, allocator)
	defer builtin.delete(iwork)

	when T == f32 {
		lapack.sggsvp_(
			jobu_c,
			jobv_c,
			jobq_c,
			&m,
			&p,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			&tola_actual,
			&tolb_actual,
			&k,
			&l,
			compute_u ? raw_data(U.data) : nil,
			&ldu,
			compute_v ? raw_data(V.data) : nil,
			&ldv,
			compute_q ? raw_data(Q.data) : nil,
			&ldq,
			raw_data(iwork),
			raw_data(tau),
			raw_data(work),
			&info,
			1,
			1,
			1,
		)
	} else {
		lapack.dggsvp_(
			jobu_c,
			jobv_c,
			jobq_c,
			&m,
			&p,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			&tola_actual,
			&tolb_actual,
			&k,
			&l,
			compute_u ? raw_data(U.data) : nil,
			&ldu,
			compute_v ? raw_data(V.data) : nil,
			&ldv,
			compute_q ? raw_data(Q.data) : nil,
			&ldq,
			raw_data(iwork),
			raw_data(tau),
			raw_data(work),
			&info,
			1,
			1,
			1,
		)
	}

	return U, V, Q, k, l, tau, info
}

m_gsvd_preprocess_c64 :: proc(
	A: ^Matrix(complex64),
	B: ^Matrix(complex64),
	tola: f32 = 0,
	tolb: f32 = 0,
	compute_u: bool = true,
	compute_v: bool = true,
	compute_q: bool = true,
	allocator := context.allocator,
) -> (
	U: Matrix(complex64),
	V: Matrix(complex64),
	Q: Matrix(complex64),
	k, l: Blas_Int,
	tau: []complex64,
	info: Info,
) {
	m := Blas_Int(A.rows)
	p := Blas_Int(A.cols)
	n := Blas_Int(B.cols)

	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Set job parameters
	jobu_c := compute_u ? cstring("U") : cstring("N")
	jobv_c := compute_v ? cstring("V") : cstring("N")
	jobq_c := compute_q ? cstring("Q") : cstring("N")

	// Use machine precision if tolerances not specified
	tola_actual := tola == 0 ? f32(1.2e-7) : tola // Approximate machine epsilon for f32
	tolb_actual := tolb == 0 ? f32(1.2e-7) : tolb

	// Allocate transformation matrices
	ldu := Blas_Int(1)
	ldv := Blas_Int(1)
	ldq := Blas_Int(1)

	if compute_u {
		U = make_matrix(complex64, int(m), int(m), allocator)
		ldu = Blas_Int(U.ld)
	}
	if compute_v {
		V = make_matrix(complex64, int(p), int(p), allocator)
		ldv = Blas_Int(V.ld)
	}
	if compute_q {
		Q = make_matrix(complex64, int(n), int(n), allocator)
		ldq = Blas_Int(Q.ld)
	}

	// Allocate tau and workspace
	tau = builtin.make([]complex64, n, allocator)
	work := builtin.make([]complex64, max(3 * n, m, p), allocator)
	defer builtin.delete(work)

	rwork := builtin.make([]f32, 2 * n, allocator)
	defer builtin.delete(rwork)

	iwork := builtin.make([]i32, n, allocator)
	defer builtin.delete(iwork)

	lapack.cggsvp_(
		jobu_c,
		jobv_c,
		jobq_c,
		&m,
		&p,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		&tola_actual,
		&tolb_actual,
		&k,
		&l,
		compute_u ? raw_data(U.data) : nil,
		&ldu,
		compute_v ? raw_data(V.data) : nil,
		&ldv,
		compute_q ? raw_data(Q.data) : nil,
		&ldq,
		raw_data(iwork),
		raw_data(rwork),
		raw_data(tau),
		raw_data(work),
		&info,
		1,
		1,
		1,
	)

	return U, V, Q, k, l, tau, info
}

m_gsvd_preprocess_c128 :: proc(
	A: ^Matrix(complex128),
	B: ^Matrix(complex128),
	tola: f64 = 0,
	tolb: f64 = 0,
	compute_u: bool = true,
	compute_v: bool = true,
	compute_q: bool = true,
	allocator := context.allocator,
) -> (
	U: Matrix(complex128),
	V: Matrix(complex128),
	Q: Matrix(complex128),
	k, l: Blas_Int,
	tau: []complex128,
	info: Info,
) {
	m := Blas_Int(A.rows)
	p := Blas_Int(A.cols)
	n := Blas_Int(B.cols)

	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Set job parameters
	jobu_c := compute_u ? cstring("U") : cstring("N")
	jobv_c := compute_v ? cstring("V") : cstring("N")
	jobq_c := compute_q ? cstring("Q") : cstring("N")

	// Use machine precision if tolerances not specified
	tola_actual := tola == 0 ? f64(2.2e-16) : tola // Approximate machine epsilon for f64
	tolb_actual := tolb == 0 ? f64(2.2e-16) : tolb

	// Allocate transformation matrices
	ldu := Blas_Int(1)
	ldv := Blas_Int(1)
	ldq := Blas_Int(1)

	if compute_u {
		U = make_matrix(complex128, int(m), int(m), allocator)
		ldu = Blas_Int(U.ld)
	}
	if compute_v {
		V = make_matrix(complex128, int(p), int(p), allocator)
		ldv = Blas_Int(V.ld)
	}
	if compute_q {
		Q = make_matrix(complex128, int(n), int(n), allocator)
		ldq = Blas_Int(Q.ld)
	}

	// Allocate tau and workspace
	tau = builtin.make([]complex128, n, allocator)
	work := builtin.make([]complex128, max(3 * n, m, p), allocator)
	defer builtin.delete(work)

	rwork := builtin.make([]f64, 2 * n, allocator)
	defer builtin.delete(rwork)

	iwork := builtin.make([]i32, n, allocator)
	defer builtin.delete(iwork)

	lapack.zggsvp_(
		jobu_c,
		jobv_c,
		jobq_c,
		&m,
		&p,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		&tola_actual,
		&tolb_actual,
		&k,
		&l,
		compute_u ? raw_data(U.data) : nil,
		&ldu,
		compute_v ? raw_data(V.data) : nil,
		&ldv,
		compute_q ? raw_data(Q.data) : nil,
		&ldq,
		raw_data(iwork),
		raw_data(rwork),
		raw_data(tau),
		raw_data(work),
		&info,
		1,
		1,
		1,
	)

	return U, V, Q, k, l, tau, info
}

// Improved/blocked version of generalized SVD preprocessing
m_gsvd_preprocess_blocked :: proc {
	m_gsvd_preprocess_blocked_real,
	m_gsvd_preprocess_blocked_c64,
	m_gsvd_preprocess_blocked_c128,
}

m_gsvd_preprocess_blocked_real :: proc(
	A: ^Matrix($T),
	B: ^Matrix(T),
	tola: T = 0,
	tolb: T = 0,
	compute_u: bool = true,
	compute_v: bool = true,
	compute_q: bool = true,
	allocator := context.allocator,
) -> (
	U: Matrix(T),
	V: Matrix(T),
	Q: Matrix(T),
	k, l: Blas_Int,
	tau: []T,
	info: Info,
) where T == f32 ||
	T == f64 {
	m := Blas_Int(A.rows)
	p := Blas_Int(A.cols)
	n := Blas_Int(B.cols)

	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Set job parameters
	jobu_c := compute_u ? cstring("U") : cstring("N")
	jobv_c := compute_v ? cstring("V") : cstring("N")
	jobq_c := compute_q ? cstring("Q") : cstring("N")

	// Use machine precision if tolerances not specified
	tola_actual := tola == 0 ? T(2.2e-16) : tola // Approximate machine epsilon for f64
	tolb_actual := tolb == 0 ? T(2.2e-16) : tolb
	when T == f32 {
		tola_actual = tola == 0 ? T(1.2e-7) : tola // Approximate machine epsilon for f32
		tolb_actual = tolb == 0 ? T(1.2e-7) : tolb
	}

	// Allocate transformation matrices
	ldu := Blas_Int(1)
	ldv := Blas_Int(1)
	ldq := Blas_Int(1)

	if compute_u {
		U = make_matrix(T, int(m), int(m), allocator)
		ldu = Blas_Int(U.ld)
	}
	if compute_v {
		V = make_matrix(T, int(p), int(p), allocator)
		ldv = Blas_Int(V.ld)
	}
	if compute_q {
		Q = make_matrix(T, int(n), int(n), allocator)
		ldq = Blas_Int(Q.ld)
	}

	// Allocate tau and integer workspace
	tau = builtin.make([]T, n, allocator)
	iwork := builtin.make([]i32, n, allocator)
	defer builtin.delete(iwork)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: T

	when T == f32 {
		lapack.sggsvp3_(
			jobu_c,
			jobv_c,
			jobq_c,
			&m,
			&p,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			&tola_actual,
			&tolb_actual,
			&k,
			&l,
			compute_u ? raw_data(U.data) : nil,
			&ldu,
			compute_v ? raw_data(V.data) : nil,
			&ldv,
			compute_q ? raw_data(Q.data) : nil,
			&ldq,
			raw_data(iwork),
			raw_data(tau),
			&work_query,
			&lwork,
			&info,
			1,
			1,
			1,
		)
	} else {
		lapack.dggsvp3_(
			jobu_c,
			jobv_c,
			jobq_c,
			&m,
			&p,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			&tola_actual,
			&tolb_actual,
			&k,
			&l,
			compute_u ? raw_data(U.data) : nil,
			&ldu,
			compute_v ? raw_data(V.data) : nil,
			&ldv,
			compute_q ? raw_data(Q.data) : nil,
			&ldq,
			raw_data(iwork),
			raw_data(tau),
			&work_query,
			&lwork,
			&info,
			1,
			1,
			1,
		)
	}

	// Allocate workspace
	lwork = Blas_Int(work_query)
	work := builtin.make([]T, lwork, allocator)
	defer builtin.delete(work)

	// Perform preprocessing
	when T == f32 {
		lapack.sggsvp3_(
			jobu_c,
			jobv_c,
			jobq_c,
			&m,
			&p,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			&tola_actual,
			&tolb_actual,
			&k,
			&l,
			compute_u ? raw_data(U.data) : nil,
			&ldu,
			compute_v ? raw_data(V.data) : nil,
			&ldv,
			compute_q ? raw_data(Q.data) : nil,
			&ldq,
			raw_data(iwork),
			raw_data(tau),
			raw_data(work),
			&lwork,
			&info,
			1,
			1,
			1,
		)
	} else {
		lapack.dggsvp3_(
			jobu_c,
			jobv_c,
			jobq_c,
			&m,
			&p,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			&tola_actual,
			&tolb_actual,
			&k,
			&l,
			compute_u ? raw_data(U.data) : nil,
			&ldu,
			compute_v ? raw_data(V.data) : nil,
			&ldv,
			compute_q ? raw_data(Q.data) : nil,
			&ldq,
			raw_data(iwork),
			raw_data(tau),
			raw_data(work),
			&lwork,
			&info,
			1,
			1,
			1,
		)
	}

	return U, V, Q, k, l, tau, info
}

m_gsvd_preprocess_blocked_c64 :: proc(
	A: ^Matrix(complex64),
	B: ^Matrix(complex64),
	tola: f32 = 0,
	tolb: f32 = 0,
	compute_u: bool = true,
	compute_v: bool = true,
	compute_q: bool = true,
	allocator := context.allocator,
) -> (
	U: Matrix(complex64),
	V: Matrix(complex64),
	Q: Matrix(complex64),
	k, l: Blas_Int,
	tau: []complex64,
	info: Info,
) {
	m := Blas_Int(A.rows)
	p := Blas_Int(A.cols)
	n := Blas_Int(B.cols)

	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Set job parameters
	jobu_c := compute_u ? cstring("U") : cstring("N")
	jobv_c := compute_v ? cstring("V") : cstring("N")
	jobq_c := compute_q ? cstring("Q") : cstring("N")

	// Use machine precision if tolerances not specified
	tola_actual := tola == 0 ? f32(1.2e-7) : tola // Approximate machine epsilon for f32
	tolb_actual := tolb == 0 ? f32(1.2e-7) : tolb

	// Allocate transformation matrices
	ldu := Blas_Int(1)
	ldv := Blas_Int(1)
	ldq := Blas_Int(1)

	if compute_u {
		U = make_matrix(complex64, int(m), int(m), allocator)
		ldu = Blas_Int(U.ld)
	}
	if compute_v {
		V = make_matrix(complex64, int(p), int(p), allocator)
		ldv = Blas_Int(V.ld)
	}
	if compute_q {
		Q = make_matrix(complex64, int(n), int(n), allocator)
		ldq = Blas_Int(Q.ld)
	}

	// Allocate tau and workspace arrays
	tau = builtin.make([]complex64, n, allocator)
	rwork := builtin.make([]f32, 2 * n, allocator)
	defer builtin.delete(rwork)

	iwork := builtin.make([]i32, n, allocator)
	defer builtin.delete(iwork)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: complex64

	lapack.cggsvp3_(
		jobu_c,
		jobv_c,
		jobq_c,
		&m,
		&p,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		&tola_actual,
		&tolb_actual,
		&k,
		&l,
		compute_u ? raw_data(U.data) : nil,
		&ldu,
		compute_v ? raw_data(V.data) : nil,
		&ldv,
		compute_q ? raw_data(Q.data) : nil,
		&ldq,
		raw_data(iwork),
		raw_data(rwork),
		raw_data(tau),
		&work_query,
		&lwork,
		&info,
		1,
		1,
		1,
	)

	// Allocate workspace
	lwork = Blas_Int(real(work_query))
	work := builtin.make([]complex64, lwork, allocator)
	defer builtin.delete(work)

	// Perform preprocessing
	lapack.cggsvp3_(
		jobu_c,
		jobv_c,
		jobq_c,
		&m,
		&p,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		&tola_actual,
		&tolb_actual,
		&k,
		&l,
		compute_u ? raw_data(U.data) : nil,
		&ldu,
		compute_v ? raw_data(V.data) : nil,
		&ldv,
		compute_q ? raw_data(Q.data) : nil,
		&ldq,
		raw_data(iwork),
		raw_data(rwork),
		raw_data(tau),
		raw_data(work),
		&lwork,
		&info,
		1,
		1,
		1,
	)

	return U, V, Q, k, l, tau, info
}

m_gsvd_preprocess_blocked_c128 :: proc(
	A: ^Matrix(complex128),
	B: ^Matrix(complex128),
	tola: f64 = 0,
	tolb: f64 = 0,
	compute_u: bool = true,
	compute_v: bool = true,
	compute_q: bool = true,
	allocator := context.allocator,
) -> (
	U: Matrix(complex128),
	V: Matrix(complex128),
	Q: Matrix(complex128),
	k, l: Blas_Int,
	tau: []complex128,
	info: Info,
) {
	m := Blas_Int(A.rows)
	p := Blas_Int(A.cols)
	n := Blas_Int(B.cols)

	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Set job parameters
	jobu_c := compute_u ? cstring("U") : cstring("N")
	jobv_c := compute_v ? cstring("V") : cstring("N")
	jobq_c := compute_q ? cstring("Q") : cstring("N")

	// Use machine precision if tolerances not specified
	tola_actual := tola == 0 ? f64(2.2e-16) : tola // Approximate machine epsilon for f64
	tolb_actual := tolb == 0 ? f64(2.2e-16) : tolb

	// Allocate transformation matrices
	ldu := Blas_Int(1)
	ldv := Blas_Int(1)
	ldq := Blas_Int(1)

	if compute_u {
		U = make_matrix(complex128, int(m), int(m), allocator)
		ldu = Blas_Int(U.ld)
	}
	if compute_v {
		V = make_matrix(complex128, int(p), int(p), allocator)
		ldv = Blas_Int(V.ld)
	}
	if compute_q {
		Q = make_matrix(complex128, int(n), int(n), allocator)
		ldq = Blas_Int(Q.ld)
	}

	// Allocate tau and workspace arrays
	tau = builtin.make([]complex128, n, allocator)
	rwork := builtin.make([]f64, 2 * n, allocator)
	defer builtin.delete(rwork)

	iwork := builtin.make([]i32, n, allocator)
	defer builtin.delete(iwork)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: complex128

	lapack.zggsvp3_(
		jobu_c,
		jobv_c,
		jobq_c,
		&m,
		&p,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		&tola_actual,
		&tolb_actual,
		&k,
		&l,
		compute_u ? raw_data(U.data) : nil,
		&ldu,
		compute_v ? raw_data(V.data) : nil,
		&ldv,
		compute_q ? raw_data(Q.data) : nil,
		&ldq,
		raw_data(iwork),
		raw_data(rwork),
		raw_data(tau),
		&work_query,
		&lwork,
		&info,
		1,
		1,
		1,
	)

	// Allocate workspace
	lwork = Blas_Int(real(work_query))
	work := builtin.make([]complex128, lwork, allocator)
	defer builtin.delete(work)

	// Perform preprocessing
	lapack.zggsvp3_(
		jobu_c,
		jobv_c,
		jobq_c,
		&m,
		&p,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		&tola_actual,
		&tolb_actual,
		&k,
		&l,
		compute_u ? raw_data(U.data) : nil,
		&ldu,
		compute_v ? raw_data(V.data) : nil,
		&ldv,
		compute_q ? raw_data(Q.data) : nil,
		&ldq,
		raw_data(iwork),
		raw_data(rwork),
		raw_data(tau),
		raw_data(work),
		&lwork,
		&info,
		1,
		1,
		1,
	)

	return U, V, Q, k, l, tau, info
}
