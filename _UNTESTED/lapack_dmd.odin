package openblas

import lapack "./f77"
import "base:builtin"
import "base:intrinsics"

// ===================================================================================
// DYNAMIC MODE DECOMPOSITION (DMD)
// Data-driven method for analyzing dynamical systems
// ===================================================================================

// DMD computation options
DMDOptions :: struct {
	// Data selection
	compute_modes:         bool, // Compute DMD modes
	compute_eigenvalues:   bool, // Compute eigenvalues
	compute_residuals:     bool, // Compute residuals
	compute_reconstructed: bool, // Compute reconstructed dynamics

	// SVD method selection
	svd_method:            Blas_Int, // 1 = gesdd, 2 = gesvd, 3 = gesvdq, 4 = gesdd with compensation

	// Rank selection
	rank_method:           cstring, // "fixed", "tolerance", "economy"
}

// Compute Dynamic Mode Decomposition
// Analyzes time series data to extract dynamical modes
m_dmd :: proc {
	m_dmd_real,
	m_dmd_c64,
	m_dmd_c128,
}

m_dmd_real :: proc(
	X: ^Matrix($T), // Snapshot matrix X = [x0, x1, ..., xn-1]
	Y: ^Matrix(T), // Shifted snapshots Y = [x1, x2, ..., xn]
	rank: Blas_Int = -1, // Target rank (-1 = automatic)
	tol: T = 0, // Tolerance for rank truncation
	options: DMDOptions = {
		compute_modes = true,
		compute_eigenvalues = true,
		svd_method = 1,
		rank_method = "tolerance",
	},
	allocator := context.allocator,
) -> (
	eigenvalues_real: []T,
	eigenvalues_imag: []T,
	modes: Matrix(T),
	residuals: []T,
	B: Matrix(T),
	W: Matrix(T),
	S: Matrix(T),// DMD operator approximation
	info: Info, // DMD modes in full space// Low-rank DMD operator
) where T == f32 || T == f64 {
	m := Blas_Int(X.rows)
	n := Blas_Int(X.cols)
	ldx := Blas_Int(X.ld)
	ldy := Blas_Int(Y.ld)

	// Set job parameters
	jobs_c := options.compute_modes ? cstring("V") : cstring("N")
	jobz_c := options.compute_eigenvalues ? cstring("V") : cstring("N")
	jobr_c := options.rank_method
	jobf_c := options.compute_residuals ? cstring("V") : cstring("N")

	// Set target rank
	k := rank
	if k < 0 {
		k = min(m, n - 1)
	}

	// Allocate eigenvalues
	eigenvalues_real = builtin.make([]T, k, allocator)
	eigenvalues_imag = builtin.make([]T, k, allocator)

	// Allocate modes matrix if requested
	ldz := m
	if options.compute_modes {
		modes_data := builtin.make([]T, m * k, allocator)
		modes = Matrix(T) {
			data   = modes_data,
			rows   = int(m),
			cols   = int(k),
			ld     = int(m),
			format = .General,
		}
	}

	// Allocate residuals if requested
	if options.compute_residuals {
		residuals = builtin.make([]T, k, allocator)
	}

	// Allocate B matrix (DMD approximation)
	ldb := k
	b_data := builtin.make([]T, k * n, allocator)
	B = Matrix(T) {
		data   = b_data,
		rows   = int(k),
		cols   = int(n),
		ld     = int(k),
		format = .General,
	}

	// Allocate W matrix (modes in full space)
	ldw := m
	w_data := builtin.make([]T, m * k, allocator)
	W = Matrix(T) {
		data   = w_data,
		rows   = int(m),
		cols   = int(k),
		ld     = int(m),
		format = .General,
	}

	// Allocate S matrix (low-rank operator)
	lds := k
	s_data := builtin.make([]T, k * k, allocator)
	S = Matrix(T) {
		data   = s_data,
		rows   = int(k),
		cols   = int(k),
		ld     = int(k),
		format = .General,
	}

	nrnk: Blas_Int // Effective rank (output)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: T
	liwork := Blas_Int(-1)
	iwork_query: Blas_Int

	when T == f32 {
		lapack.sgedmd_(
			jobs_c,
			jobz_c,
			jobr_c,
			jobf_c,
			&options.svd_method,
			&m,
			&n,
			raw_data(X.data),
			&ldx,
			raw_data(Y.data),
			&ldy,
			&nrnk,
			&tol,
			&k,
			raw_data(eigenvalues_real),
			raw_data(eigenvalues_imag),
			options.compute_modes ? raw_data(modes.data) : nil,
			&ldz,
			options.compute_residuals ? raw_data(residuals) : nil,
			raw_data(B.data),
			&ldb,
			raw_data(W.data),
			&ldw,
			raw_data(S.data),
			&lds,
			&work_query,
			&lwork,
			&iwork_query,
			&liwork,
			&info,
			1,
			1,
			1,
			1,
		)

		lwork = Blas_Int(work_query)
		liwork = iwork_query

		// Allocate workspace
		work := builtin.make([]T, lwork, allocator)
		iwork := builtin.make([]i32, liwork, allocator)
		defer builtin.delete(work)
		defer builtin.delete(iwork)

		// Compute DMD
		lapack.sgedmd_(
			jobs_c,
			jobz_c,
			jobr_c,
			jobf_c,
			&options.svd_method,
			&m,
			&n,
			raw_data(X.data),
			&ldx,
			raw_data(Y.data),
			&ldy,
			&nrnk,
			&tol,
			&k,
			raw_data(eigenvalues_real),
			raw_data(eigenvalues_imag),
			options.compute_modes ? raw_data(modes.data) : nil,
			&ldz,
			options.compute_residuals ? raw_data(residuals) : nil,
			raw_data(B.data),
			&ldb,
			raw_data(W.data),
			&ldw,
			raw_data(S.data),
			&lds,
			raw_data(work),
			&lwork,
			raw_data(iwork),
			&liwork,
			&info,
			1,
			1,
			1,
			1,
		)
	} else {
		lapack.dgedmd_(
			jobs_c,
			jobz_c,
			jobr_c,
			jobf_c,
			&options.svd_method,
			&m,
			&n,
			raw_data(X.data),
			&ldx,
			raw_data(Y.data),
			&ldy,
			&nrnk,
			&tol,
			&k,
			raw_data(eigenvalues_real),
			raw_data(eigenvalues_imag),
			options.compute_modes ? raw_data(modes.data) : nil,
			&ldz,
			options.compute_residuals ? raw_data(residuals) : nil,
			raw_data(B.data),
			&ldb,
			raw_data(W.data),
			&ldw,
			raw_data(S.data),
			&lds,
			&work_query,
			&lwork,
			&iwork_query,
			&liwork,
			&info,
			1,
			1,
			1,
			1,
		)

		lwork = Blas_Int(work_query)
		liwork = iwork_query

		// Allocate workspace
		work := builtin.make([]T, lwork, allocator)
		iwork := builtin.make([]i32, liwork, allocator)
		defer builtin.delete(work)
		defer builtin.delete(iwork)

		// Compute DMD
		lapack.dgedmd_(
			jobs_c,
			jobz_c,
			jobr_c,
			jobf_c,
			&options.svd_method,
			&m,
			&n,
			raw_data(X.data),
			&ldx,
			raw_data(Y.data),
			&ldy,
			&nrnk,
			&tol,
			&k,
			raw_data(eigenvalues_real),
			raw_data(eigenvalues_imag),
			options.compute_modes ? raw_data(modes.data) : nil,
			&ldz,
			options.compute_residuals ? raw_data(residuals) : nil,
			raw_data(B.data),
			&ldb,
			raw_data(W.data),
			&ldw,
			raw_data(S.data),
			&lds,
			raw_data(work),
			&lwork,
			raw_data(iwork),
			&liwork,
			&info,
			1,
			1,
			1,
			1,
		)
	}

	// Resize outputs to actual rank if needed
	if nrnk < k {
		eigenvalues_real = eigenvalues_real[:nrnk]
		eigenvalues_imag = eigenvalues_imag[:nrnk]
		if options.compute_modes {
			modes.cols = int(nrnk)
		}
		if options.compute_residuals {
			residuals = residuals[:nrnk]
		}
		B.rows = int(nrnk)
		W.cols = int(nrnk)
		S.rows = int(nrnk)
		S.cols = int(nrnk)
	}

	return eigenvalues_real, eigenvalues_imag, modes, residuals, B, W, S, info
}

m_dmd_c64 :: proc(
	X: ^Matrix(complex64),
	Y: ^Matrix(complex64),
	rank: Blas_Int = -1,
	tol: f32 = 0,
	options: DMDOptions = {
		compute_modes = true,
		compute_eigenvalues = true,
		svd_method = 1,
		rank_method = "tolerance",
	},
	allocator := context.allocator,
) -> (
	eigenvalues: []complex64,
	modes: Matrix(complex64),
	residuals: []f32,
	B: Matrix(complex64),
	W: Matrix(complex64),
	S: Matrix(complex64),
	info: Info,
) {
	m := Blas_Int(X.rows)
	n := Blas_Int(X.cols)
	ldx := Blas_Int(X.ld)
	ldy := Blas_Int(Y.ld)

	// Set job parameters
	jobs_c := options.compute_modes ? cstring("V") : cstring("N")
	jobz_c := options.compute_eigenvalues ? cstring("V") : cstring("N")
	jobr_c := options.rank_method
	jobf_c := options.compute_residuals ? cstring("V") : cstring("N")

	// Set target rank
	k := rank
	if k < 0 {
		k = min(m, n - 1)
	}

	// Allocate eigenvalues
	eigenvalues = builtin.make([]complex64, k, allocator)

	// Allocate modes matrix if requested
	ldz := m
	if options.compute_modes {
		modes_data := builtin.make([]complex64, m * k, allocator)
		modes = Matrix(complex64) {
			data   = modes_data,
			rows   = int(m),
			cols   = int(k),
			ld     = int(m),
			format = .General,
		}
	}

	// Allocate residuals if requested
	if options.compute_residuals {
		residuals = builtin.make([]f32, k, allocator)
	}

	// Allocate matrices
	ldb := k
	b_data := builtin.make([]complex64, k * n, allocator)
	B = Matrix(complex64) {
		data   = b_data,
		rows   = int(k),
		cols   = int(n),
		ld     = int(k),
		format = .General,
	}

	ldw := m
	w_data := builtin.make([]complex64, m * k, allocator)
	W = Matrix(complex64) {
		data   = w_data,
		rows   = int(m),
		cols   = int(k),
		ld     = int(m),
		format = .General,
	}

	lds := k
	s_data := builtin.make([]complex64, k * k, allocator)
	S = Matrix(complex64) {
		data   = s_data,
		rows   = int(k),
		cols   = int(k),
		ld     = int(k),
		format = .General,
	}

	nrnk: Blas_Int

	// Query for optimal workspace
	lzwork := Blas_Int(-1)
	zwork_query: complex64
	lwork := Blas_Int(-1)
	work_query: f32
	liwork := Blas_Int(-1)
	iwork_query: Blas_Int

	lapack.cgedmd_(
		jobs_c,
		jobz_c,
		jobr_c,
		jobf_c,
		&options.svd_method,
		&m,
		&n,
		raw_data(X.data),
		&ldx,
		raw_data(Y.data),
		&ldy,
		&nrnk,
		&tol,
		&k,
		raw_data(eigenvalues),
		options.compute_modes ? raw_data(modes.data) : nil,
		&ldz,
		options.compute_residuals ? raw_data(residuals) : nil,
		raw_data(B.data),
		&ldb,
		raw_data(W.data),
		&ldw,
		raw_data(S.data),
		&lds,
		&zwork_query,
		&lzwork,
		&work_query,
		&lwork,
		&iwork_query,
		&liwork,
		&info,
		1,
		1,
		1,
		1,
	)

	lzwork = Blas_Int(real(zwork_query))
	lwork = Blas_Int(work_query)
	liwork = iwork_query

	// Allocate workspaces
	zwork := builtin.make([]complex64, lzwork, allocator)
	work := builtin.make([]f32, lwork, allocator)
	iwork := builtin.make([]i32, liwork, allocator)
	defer builtin.delete(zwork)
	defer builtin.delete(work)
	defer builtin.delete(iwork)

	// Compute DMD
	lapack.cgedmd_(
		jobs_c,
		jobz_c,
		jobr_c,
		jobf_c,
		&options.svd_method,
		&m,
		&n,
		raw_data(X.data),
		&ldx,
		raw_data(Y.data),
		&ldy,
		&nrnk,
		&tol,
		&k,
		raw_data(eigenvalues),
		options.compute_modes ? raw_data(modes.data) : nil,
		&ldz,
		options.compute_residuals ? raw_data(residuals) : nil,
		raw_data(B.data),
		&ldb,
		raw_data(W.data),
		&ldw,
		raw_data(S.data),
		&lds,
		raw_data(zwork),
		&lzwork,
		raw_data(work),
		&lwork,
		raw_data(iwork),
		&liwork,
		&info,
		1,
		1,
		1,
		1,
	)

	// Resize outputs to actual rank if needed
	if nrnk < k {
		eigenvalues = eigenvalues[:nrnk]
		if options.compute_modes {
			modes.cols = int(nrnk)
		}
		if options.compute_residuals {
			residuals = residuals[:nrnk]
		}
		B.rows = int(nrnk)
		W.cols = int(nrnk)
		S.rows = int(nrnk)
		S.cols = int(nrnk)
	}

	return eigenvalues, modes, residuals, B, W, S, info
}

m_dmd_c128 :: proc(
	X: ^Matrix(complex128),
	Y: ^Matrix(complex128),
	rank: Blas_Int = -1,
	tol: f64 = 0,
	options: DMDOptions = {
		compute_modes = true,
		compute_eigenvalues = true,
		svd_method = 1,
		rank_method = "tolerance",
	},
	allocator := context.allocator,
) -> (
	eigenvalues: []complex128,
	modes: Matrix(complex128),
	residuals: []f64,
	B: Matrix(complex128),
	W: Matrix(complex128),
	S: Matrix(complex128),
	info: Info,
) {
	m := Blas_Int(X.rows)
	n := Blas_Int(X.cols)
	ldx := Blas_Int(X.ld)
	ldy := Blas_Int(Y.ld)

	// Set job parameters
	jobs_c := options.compute_modes ? cstring("V") : cstring("N")
	jobz_c := options.compute_eigenvalues ? cstring("V") : cstring("N")
	jobr_c := options.rank_method
	jobf_c := options.compute_residuals ? cstring("V") : cstring("N")

	// Set target rank
	k := rank
	if k < 0 {
		k = min(m, n - 1)
	}

	// Allocate eigenvalues
	eigenvalues = builtin.make([]complex128, k, allocator)

	// Allocate modes matrix if requested
	ldz := m
	if options.compute_modes {
		modes_data := builtin.make([]complex128, m * k, allocator)
		modes = Matrix(complex128) {
			data   = modes_data,
			rows   = int(m),
			cols   = int(k),
			ld     = int(m),
			format = .General,
		}
	}

	// Allocate residuals if requested
	if options.compute_residuals {
		residuals = builtin.make([]f64, k, allocator)
	}

	// Allocate matrices
	ldb := k
	b_data := builtin.make([]complex128, k * n, allocator)
	B = Matrix(complex128) {
		data   = b_data,
		rows   = int(k),
		cols   = int(n),
		ld     = int(k),
		format = .General,
	}

	ldw := m
	w_data := builtin.make([]complex128, m * k, allocator)
	W = Matrix(complex128) {
		data   = w_data,
		rows   = int(m),
		cols   = int(k),
		ld     = int(m),
		format = .General,
	}

	lds := k
	s_data := builtin.make([]complex128, k * k, allocator)
	S = Matrix(complex128) {
		data   = s_data,
		rows   = int(k),
		cols   = int(k),
		ld     = int(k),
		format = .General,
	}

	nrnk: Blas_Int

	// Query for optimal workspace
	lzwork := Blas_Int(-1)
	zwork_query: complex128
	lrwork := Blas_Int(-1)
	rwork_query: f64
	liwork := Blas_Int(-1)
	iwork_query: Blas_Int

	lapack.zgedmd_(
		jobs_c,
		jobz_c,
		jobr_c,
		jobf_c,
		&options.svd_method,
		&m,
		&n,
		raw_data(X.data),
		&ldx,
		raw_data(Y.data),
		&ldy,
		&nrnk,
		&tol,
		&k,
		raw_data(eigenvalues),
		options.compute_modes ? raw_data(modes.data) : nil,
		&ldz,
		options.compute_residuals ? raw_data(residuals) : nil,
		raw_data(B.data),
		&ldb,
		raw_data(W.data),
		&ldw,
		raw_data(S.data),
		&lds,
		&zwork_query,
		&lzwork,
		&rwork_query,
		&lrwork,
		&iwork_query,
		&liwork,
		&info,
		1,
		1,
		1,
		1,
	)

	lzwork = Blas_Int(real(zwork_query))
	lrwork = Blas_Int(rwork_query)
	liwork = iwork_query

	// Allocate workspaces
	zwork := builtin.make([]complex128, lzwork, allocator)
	rwork := builtin.make([]f64, lrwork, allocator)
	iwork := builtin.make([]i32, liwork, allocator)
	defer builtin.delete(zwork)
	defer builtin.delete(rwork)
	defer builtin.delete(iwork)

	// Compute DMD
	lapack.zgedmd_(
		jobs_c,
		jobz_c,
		jobr_c,
		jobf_c,
		&options.svd_method,
		&m,
		&n,
		raw_data(X.data),
		&ldx,
		raw_data(Y.data),
		&ldy,
		&nrnk,
		&tol,
		&k,
		raw_data(eigenvalues),
		options.compute_modes ? raw_data(modes.data) : nil,
		&ldz,
		options.compute_residuals ? raw_data(residuals) : nil,
		raw_data(B.data),
		&ldb,
		raw_data(W.data),
		&ldw,
		raw_data(S.data),
		&lds,
		raw_data(zwork),
		&lzwork,
		raw_data(rwork),
		&lrwork,
		raw_data(iwork),
		&liwork,
		&info,
		1,
		1,
		1,
		1,
	)

	// Resize outputs to actual rank if needed
	if nrnk < k {
		eigenvalues = eigenvalues[:nrnk]
		if options.compute_modes {
			modes.cols = int(nrnk)
		}
		if options.compute_residuals {
			residuals = residuals[:nrnk]
		}
		B.rows = int(nrnk)
		W.cols = int(nrnk)
		S.rows = int(nrnk)
		S.cols = int(nrnk)
	}

	return eigenvalues, modes, residuals, B, W, S, info
}

// TODO: Implement gedmdq (QR-based DMD with more computational options)
