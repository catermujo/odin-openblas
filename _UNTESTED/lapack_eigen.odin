package openblas

import lapack "./f77"
import "base:builtin"
import "base:intrinsics"

// ===================================================================================
// EIGENVALUE PROBLEMS
// Compute eigenvalues and eigenvectors of matrices
// ===================================================================================

// ===================================================================================
// MATRIX BALANCING
// Improve accuracy of eigenvalue computations by equilibrating matrices
// ===================================================================================

// Balancing operation types
BalanceMode :: enum {
	None, // No balancing
	Permute, // Permutation only to isolate eigenvalues
	Scale, // Diagonal scaling only
	Both, // Both permutation and scaling
}

// Balance a general matrix to improve eigenvalue computation accuracy
// Performs permutation and/or scaling to reduce norm
m_balance :: proc {
	m_balance_real,
	m_balance_c64,
	m_balance_c128,
}

m_balance_real :: proc(
	A: ^Matrix($T),
	mode: BalanceMode = .Both,
	allocator := context.allocator,
) -> (
	scale: []T,
	ilo, ihi: Blas_Int,
	info: Info, // Scaling factors// Indices of non-isolated eigenvalues
) where T == f32 || T == f64 {
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)

	// Convert balance mode to string
	job_c: cstring
	switch mode {
	case .None:
		job_c = cstring("N")
	case .Permute:
		job_c = cstring("P")
	case .Scale:
		job_c = cstring("S")
	case .Both:
		job_c = cstring("B")
	}

	// Allocate scale array
	scale = builtin.make([]T, n, allocator)

	when T == f32 {
		lapack.sgebal_(job_c, &n, raw_data(A.data), &lda, &ilo, &ihi, raw_data(scale), &info, 1)
	} else when T == f64 {
		lapack.dgebal_(job_c, &n, raw_data(A.data), &lda, &ilo, &ihi, raw_data(scale), &info, 1)
	}

	return scale, ilo, ihi, info
}

m_balance_c64 :: proc(
	A: ^Matrix(complex64),
	mode: BalanceMode = .Both,
	allocator := context.allocator,
) -> (
	scale: []f32,
	ilo, ihi: Blas_Int,
	info: Info,
) {
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)

	// Convert balance mode to string
	job_c: cstring
	switch mode {
	case .None:
		job_c = cstring("N")
	case .Permute:
		job_c = cstring("P")
	case .Scale:
		job_c = cstring("S")
	case .Both:
		job_c = cstring("B")
	}

	// Allocate scale array
	scale = builtin.make([]f32, n, allocator)

	lapack.cgebal_(job_c, &n, raw_data(A.data), &lda, &ilo, &ihi, raw_data(scale), &info, 1)

	return scale, ilo, ihi, info
}

m_balance_c128 :: proc(
	A: ^Matrix(complex128),
	mode: BalanceMode = .Both,
	allocator := context.allocator,
) -> (
	scale: []f64,
	ilo, ihi: Blas_Int,
	info: Info,
) {
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)

	// Convert balance mode to string
	job_c: cstring
	switch mode {
	case .None:
		job_c = cstring("N")
	case .Permute:
		job_c = cstring("P")
	case .Scale:
		job_c = cstring("S")
	case .Both:
		job_c = cstring("B")
	}

	// Allocate scale array
	scale = builtin.make([]f64, n, allocator)

	lapack.zgebal_(job_c, &n, raw_data(A.data), &lda, &ilo, &ihi, raw_data(scale), &info, 1)

	return scale, ilo, ihi, info
}

// Back-transform eigenvectors computed for balanced matrix
// Applies inverse of balancing transformation to eigenvectors
m_balance_back :: proc {
	m_balance_back_real,
	m_balance_back_c64,
	m_balance_back_c128,
}

// Side options for back-transformation
BalanceBackSide :: enum {
	Left, // Back-transform left eigenvectors
	Right, // Back-transform right eigenvectors
}

m_balance_back_real :: proc(
	V: ^Matrix($T), // Eigenvectors to back-transform
	scale: []T, // Scale factors from m_balance
	ilo, ihi: Blas_Int, // Indices from m_balance
	side: BalanceBackSide = .Right,
	mode: BalanceMode = .Both,
) -> (
	info: Info,
) where T == f32 || T == f64 {
	n := Blas_Int(V.rows)
	m := Blas_Int(V.cols) // Number of eigenvectors
	ldv := Blas_Int(V.ld)

	// Convert options to strings
	job_c: cstring
	switch mode {
	case .None:
		job_c = cstring("N")
	case .Permute:
		job_c = cstring("P")
	case .Scale:
		job_c = cstring("S")
	case .Both:
		job_c = cstring("B")
	}

	side_c := side == .Left ? cstring("L") : cstring("R")

	ilo_copy := ilo
	ihi_copy := ihi

	when T == f32 {
		lapack.sgebak_(
			job_c,
			side_c,
			&n,
			&ilo_copy,
			&ihi_copy,
			raw_data(scale),
			&m,
			raw_data(V.data),
			&ldv,
			&info,
			1,
			1,
		)
	} else when T == f64 {
		lapack.dgebak_(
			job_c,
			side_c,
			&n,
			&ilo_copy,
			&ihi_copy,
			raw_data(scale),
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

m_balance_back_c64 :: proc(
	V: ^Matrix(complex64),
	scale: []f32,
	ilo, ihi: Blas_Int,
	side: BalanceBackSide = .Right,
	mode: BalanceMode = .Both,
) -> (
	info: Info,
) {
	n := Blas_Int(V.rows)
	m := Blas_Int(V.cols)
	ldv := Blas_Int(V.ld)

	// Convert options to strings
	job_c: cstring
	switch mode {
	case .None:
		job_c = cstring("N")
	case .Permute:
		job_c = cstring("P")
	case .Scale:
		job_c = cstring("S")
	case .Both:
		job_c = cstring("B")
	}

	side_c := side == .Left ? cstring("L") : cstring("R")

	ilo_copy := ilo
	ihi_copy := ihi

	lapack.cgebak_(
		job_c,
		side_c,
		&n,
		&ilo_copy,
		&ihi_copy,
		raw_data(scale),
		&m,
		raw_data(V.data),
		&ldv,
		&info,
		1,
		1,
	)

	return info
}

m_balance_back_c128 :: proc(
	V: ^Matrix(complex128),
	scale: []f64,
	ilo, ihi: Blas_Int,
	side: BalanceBackSide = .Right,
	mode: BalanceMode = .Both,
) -> (
	info: Info,
) {
	n := Blas_Int(V.rows)
	m := Blas_Int(V.cols)
	ldv := Blas_Int(V.ld)

	// Convert options to strings
	job_c: cstring
	switch mode {
	case .None:
		job_c = cstring("N")
	case .Permute:
		job_c = cstring("P")
	case .Scale:
		job_c = cstring("S")
	case .Both:
		job_c = cstring("B")
	}

	side_c := side == .Left ? cstring("L") : cstring("R")

	ilo_copy := ilo
	ihi_copy := ihi

	lapack.zgebak_(
		job_c,
		side_c,
		&n,
		&ilo_copy,
		&ihi_copy,
		raw_data(scale),
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
// SCHUR DECOMPOSITION
// Compute Schur factorization for eigenvalue analysis
// ===================================================================================

// Schur decomposition: A = Q*T*Q^H
// T is upper triangular (complex) or quasi-triangular (real)
m_schur :: proc {
	m_schur_real,
	m_schur_c64,
	m_schur_c128,
}

m_schur_real :: proc(
	A: ^Matrix($T),
	compute_vs: bool = true, // Compute Schur vectors
	sort_eigenvalues: bool = false, // Sort eigenvalues
	select_fn: lapack.LAPACK_S_SELECT2 = nil, // Selection function for sorting (f32)
	allocator := context.allocator,
) -> (
	WR: []T,
	WI: []T,
	VS: Matrix(T),
	sdim: Blas_Int,// Real parts of eigenvalues
	info: Info, // Imaginary parts of eigenvalues// Schur vectors (if requested)// Number of selected eigenvalues
) where T == f32 || T == f64 {
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)

	// Allocate eigenvalue arrays
	WR = builtin.make([]T, n, allocator)
	WI = builtin.make([]T, n, allocator)

	// Prepare job parameters
	jobvs_c := compute_vs ? cstring("V") : cstring("N")
	sort_c := sort_eigenvalues ? cstring("S") : cstring("N")

	// Allocate Schur vectors if requested
	ldvs := Blas_Int(1)
	if compute_vs {
		VS = make_matrix(T, int(n), int(n), allocator)
		ldvs = Blas_Int(VS.ld)
	}

	// Allocate workspace for sorting
	bwork: []i32
	if sort_eigenvalues {
		bwork = builtin.make([]i32, n, allocator)
		defer builtin.delete(bwork)
	}

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: T

	when T == f32 {
		select_ptr := sort_eigenvalues ? select_fn : lapack.LAPACK_S_SELECT2(nil)
		if select_ptr == nil && sort_eigenvalues {
			// Default selection: select all eigenvalues
			select_ptr = proc "c" (wr: ^f32, wi: ^f32) -> Blas_Int {return 1}
		}

		lapack.sgees_(
			jobvs_c,
			sort_c,
			select_ptr,
			&n,
			raw_data(A.data),
			&lda,
			&sdim,
			raw_data(WR),
			raw_data(WI),
			compute_vs ? raw_data(VS.data) : nil,
			&ldvs,
			&work_query,
			&lwork,
			raw_data(bwork) if sort_eigenvalues else nil,
			&info,
			1,
			1,
		)

		// Allocate workspace
		lwork = Blas_Int(work_query)
		work := builtin.make([]T, lwork, allocator)
		defer builtin.delete(work)

		// Perform decomposition
		lapack.sgees_(
			jobvs_c,
			sort_c,
			select_ptr,
			&n,
			raw_data(A.data),
			&lda,
			&sdim,
			raw_data(WR),
			raw_data(WI),
			compute_vs ? raw_data(VS.data) : nil,
			&ldvs,
			raw_data(work),
			&lwork,
			raw_data(bwork) if sort_eigenvalues else nil,
			&info,
			1,
			1,
		)
	} else when T == f64 {
		select_ptr := lapack.LAPACK_D_SELECT2(nil)
		if sort_eigenvalues && select_fn != nil {
			// Cast the select function - this is a workaround
			select_ptr = cast(lapack.LAPACK_D_SELECT2)select_fn
		} else if sort_eigenvalues {
			// Default selection: select all eigenvalues
			select_ptr = proc "c" (wr: ^f64, wi: ^f64) -> Blas_Int {return 1}
		}

		lapack.dgees_(
			jobvs_c,
			sort_c,
			select_ptr,
			&n,
			raw_data(A.data),
			&lda,
			&sdim,
			raw_data(WR),
			raw_data(WI),
			compute_vs ? raw_data(VS.data) : nil,
			&ldvs,
			&work_query,
			&lwork,
			raw_data(bwork) if sort_eigenvalues else nil,
			&info,
			1,
			1,
		)

		// Allocate workspace
		lwork = Blas_Int(work_query)
		work := builtin.make([]T, lwork, allocator)
		defer builtin.delete(work)

		// Perform decomposition
		lapack.dgees_(
			jobvs_c,
			sort_c,
			select_ptr,
			&n,
			raw_data(A.data),
			&lda,
			&sdim,
			raw_data(WR),
			raw_data(WI),
			compute_vs ? raw_data(VS.data) : nil,
			&ldvs,
			raw_data(work),
			&lwork,
			raw_data(bwork) if sort_eigenvalues else nil,
			&info,
			1,
			1,
		)
	}

	return WR, WI, VS, sdim, info
}

m_schur_c64 :: proc(
	A: ^Matrix(complex64),
	compute_vs: bool = true,
	sort_eigenvalues: bool = false,
	select_fn: lapack.LAPACK_C_SELECT1 = nil,
	allocator := context.allocator,
) -> (
	W: []complex64,
	VS: Matrix(complex64),
	sdim: Blas_Int,
	info: Info, // Eigenvalues// Schur vectors (if requested)// Number of selected eigenvalues
) {
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)

	// Allocate eigenvalue array
	W = builtin.make([]complex64, n, allocator)

	// Prepare job parameters
	jobvs_c := compute_vs ? cstring("V") : cstring("N")
	sort_c := sort_eigenvalues ? cstring("S") : cstring("N")

	// Allocate Schur vectors if requested
	ldvs := Blas_Int(1)
	if compute_vs {
		VS = make_matrix(complex64, int(n), int(n), allocator)
		ldvs = Blas_Int(VS.ld)
	}

	// Allocate workspace for sorting
	bwork: []i32
	if sort_eigenvalues {
		bwork = builtin.make([]i32, n, allocator)
		defer builtin.delete(bwork)
	}

	// Allocate real workspace
	rwork := builtin.make([]f32, n, allocator)
	defer builtin.delete(rwork)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: complex64

	select_ptr := sort_eigenvalues ? select_fn : lapack.LAPACK_C_SELECT1(nil)
	if select_ptr == nil && sort_eigenvalues {
		// Default selection: select all eigenvalues
		select_ptr = proc "c" (w: ^complex64) -> Blas_Int {return 1}
	}

	lapack.cgees_(
		jobvs_c,
		sort_c,
		select_ptr,
		&n,
		raw_data(A.data),
		&lda,
		&sdim,
		raw_data(W),
		compute_vs ? raw_data(VS.data) : nil,
		&ldvs,
		&work_query,
		&lwork,
		raw_data(rwork),
		raw_data(bwork) if sort_eigenvalues else nil,
		&info,
		1,
		1,
	)

	// Allocate workspace
	lwork = Blas_Int(real(work_query))
	work := builtin.make([]complex64, lwork, allocator)
	defer builtin.delete(work)

	// Perform decomposition
	lapack.cgees_(
		jobvs_c,
		sort_c,
		select_ptr,
		&n,
		raw_data(A.data),
		&lda,
		&sdim,
		raw_data(W),
		compute_vs ? raw_data(VS.data) : nil,
		&ldvs,
		raw_data(work),
		&lwork,
		raw_data(rwork),
		raw_data(bwork) if sort_eigenvalues else nil,
		&info,
		1,
		1,
	)

	return W, VS, sdim, info
}

m_schur_c128 :: proc(
	A: ^Matrix(complex128),
	compute_vs: bool = true,
	sort_eigenvalues: bool = false,
	select_fn: lapack.LAPACK_Z_SELECT1 = nil,
	allocator := context.allocator,
) -> (
	W: []complex128,
	VS: Matrix(complex128),
	sdim: Blas_Int,
	info: Info, // Eigenvalues// Schur vectors (if requested)// Number of selected eigenvalues
) {
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)

	// Allocate eigenvalue array
	W = builtin.make([]complex128, n, allocator)

	// Prepare job parameters
	jobvs_c := compute_vs ? cstring("V") : cstring("N")
	sort_c := sort_eigenvalues ? cstring("S") : cstring("N")

	// Allocate Schur vectors if requested
	ldvs := Blas_Int(1)
	if compute_vs {
		VS = make_matrix(complex128, int(n), int(n), allocator)
		ldvs = Blas_Int(VS.ld)
	}

	// Allocate workspace for sorting
	bwork: []i32
	if sort_eigenvalues {
		bwork = builtin.make([]i32, n, allocator)
		defer builtin.delete(bwork)
	}

	// Allocate real workspace
	rwork := builtin.make([]f64, n, allocator)
	defer builtin.delete(rwork)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: complex128

	select_ptr := sort_eigenvalues ? select_fn : lapack.LAPACK_Z_SELECT1(nil)
	if select_ptr == nil && sort_eigenvalues {
		// Default selection: select all eigenvalues
		select_ptr = proc "c" (w: ^complex128) -> Blas_Int {return 1}
	}

	lapack.zgees_(
		jobvs_c,
		sort_c,
		select_ptr,
		&n,
		raw_data(A.data),
		&lda,
		&sdim,
		raw_data(W),
		compute_vs ? raw_data(VS.data) : nil,
		&ldvs,
		&work_query,
		&lwork,
		raw_data(rwork),
		raw_data(bwork) if sort_eigenvalues else nil,
		&info,
		1,
		1,
	)

	// Allocate workspace
	lwork = Blas_Int(real(work_query))
	work := builtin.make([]complex128, lwork, allocator)
	defer builtin.delete(work)

	// Perform decomposition
	lapack.zgees_(
		jobvs_c,
		sort_c,
		select_ptr,
		&n,
		raw_data(A.data),
		&lda,
		&sdim,
		raw_data(W),
		compute_vs ? raw_data(VS.data) : nil,
		&ldvs,
		raw_data(work),
		&lwork,
		raw_data(rwork),
		raw_data(bwork) if sort_eigenvalues else nil,
		&info,
		1,
		1,
	)

	return W, VS, sdim, info
}

// Expert Schur decomposition with condition number estimates
// Provides sensitivity analysis for eigenvalues and invariant subspaces
m_schur_expert :: proc {
	m_schur_expert_real,
	m_schur_expert_c64,
	m_schur_expert_c128,
}

// Sense options for condition number computation
SchurSenseMode :: enum {
	None        = 0, // Don't compute condition numbers
	Eigenvalues = 1, // Condition numbers for eigenvalues only
	Subspace    = 2, // Condition numbers for invariant subspace only
	Both        = 3, // Both eigenvalues and subspace
}

m_schur_expert_real :: proc(
	A: ^Matrix($T),
	compute_vs: bool = true,
	sort_eigenvalues: bool = false,
	select_fn: lapack.LAPACK_S_SELECT2 = nil,
	sense: SchurSenseMode = .None,
	allocator := context.allocator,
) -> (
	WR: []T,
	WI: []T,
	VS: Matrix(T),
	sdim: Blas_Int,// Real parts of eigenvalues
	rconde: T,// Imaginary parts of eigenvalues
	rcondv: T,// Schur vectors (if requested)
	info: Info, // Number of selected eigenvalues// Reciprocal condition number for eigenvalues// Reciprocal condition number for subspace
) where T == f32 || T == f64 {
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)

	// Allocate eigenvalue arrays
	WR = builtin.make([]T, n, allocator)
	WI = builtin.make([]T, n, allocator)

	// Prepare job parameters
	jobvs_c := compute_vs ? cstring("V") : cstring("N")
	sort_c := sort_eigenvalues ? cstring("S") : cstring("N")

	// Convert sense mode
	sense_c: cstring
	switch sense {
	case .None:
		sense_c = cstring("N")
	case .Eigenvalues:
		sense_c = cstring("E")
	case .Subspace:
		sense_c = cstring("V")
	case .Both:
		sense_c = cstring("B")
	}

	// Allocate Schur vectors if requested
	ldvs := Blas_Int(1)
	if compute_vs {
		VS = make_matrix(T, int(n), int(n), allocator)
		ldvs = Blas_Int(VS.ld)
	}

	// Allocate workspace for sorting
	bwork: []i32
	if sort_eigenvalues {
		bwork = builtin.make([]i32, n, allocator)
		defer builtin.delete(bwork)
	}

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	liwork := Blas_Int(-1)
	work_query: T
	iwork_query: Blas_Int

	when T == f32 {
		select_ptr := sort_eigenvalues ? select_fn : lapack.LAPACK_S_SELECT2(nil)
		if select_ptr == nil && sort_eigenvalues {
			select_ptr = proc "c" (wr: ^f32, wi: ^f32) -> Blas_Int {return 1}
		}

		lapack.sgeesx_(
			jobvs_c,
			sort_c,
			select_ptr,
			sense_c,
			&n,
			raw_data(A.data),
			&lda,
			&sdim,
			raw_data(WR),
			raw_data(WI),
			compute_vs ? raw_data(VS.data) : nil,
			&ldvs,
			&rconde,
			&rcondv,
			&work_query,
			&lwork,
			&iwork_query,
			&liwork,
			raw_data(bwork) if sort_eigenvalues else nil,
			&info,
			1,
			1,
			1,
		)

		// Allocate workspace
		lwork = Blas_Int(work_query)
		liwork = iwork_query
		work := builtin.make([]T, lwork, allocator)
		iwork := builtin.make([]i32, liwork, allocator)
		defer builtin.delete(work)
		defer builtin.delete(iwork)

		// Perform decomposition
		lapack.sgeesx_(
			jobvs_c,
			sort_c,
			select_ptr,
			sense_c,
			&n,
			raw_data(A.data),
			&lda,
			&sdim,
			raw_data(WR),
			raw_data(WI),
			compute_vs ? raw_data(VS.data) : nil,
			&ldvs,
			&rconde,
			&rcondv,
			raw_data(work),
			&lwork,
			raw_data(iwork),
			&liwork,
			raw_data(bwork) if sort_eigenvalues else nil,
			&info,
			1,
			1,
			1,
		)
	} else when T == f64 {
		select_ptr := lapack.LAPACK_D_SELECT2(nil)
		if sort_eigenvalues && select_fn != nil {
			select_ptr = cast(lapack.LAPACK_D_SELECT2)select_fn
		} else if sort_eigenvalues {
			select_ptr = proc "c" (wr: ^f64, wi: ^f64) -> Blas_Int {return 1}
		}

		lapack.dgeesx_(
			jobvs_c,
			sort_c,
			select_ptr,
			sense_c,
			&n,
			raw_data(A.data),
			&lda,
			&sdim,
			raw_data(WR),
			raw_data(WI),
			compute_vs ? raw_data(VS.data) : nil,
			&ldvs,
			&rconde,
			&rcondv,
			&work_query,
			&lwork,
			&iwork_query,
			&liwork,
			raw_data(bwork) if sort_eigenvalues else nil,
			&info,
			1,
			1,
			1,
		)

		// Allocate workspace
		lwork = Blas_Int(work_query)
		liwork = iwork_query
		work := builtin.make([]T, lwork, allocator)
		iwork := builtin.make([]i32, liwork, allocator)
		defer builtin.delete(work)
		defer builtin.delete(iwork)

		// Perform decomposition
		lapack.dgeesx_(
			jobvs_c,
			sort_c,
			select_ptr,
			sense_c,
			&n,
			raw_data(A.data),
			&lda,
			&sdim,
			raw_data(WR),
			raw_data(WI),
			compute_vs ? raw_data(VS.data) : nil,
			&ldvs,
			&rconde,
			&rcondv,
			raw_data(work),
			&lwork,
			raw_data(iwork),
			&liwork,
			raw_data(bwork) if sort_eigenvalues else nil,
			&info,
			1,
			1,
			1,
		)
	}

	return WR, WI, VS, sdim, rconde, rcondv, info
}

m_schur_expert_c64 :: proc(
	A: ^Matrix(complex64),
	compute_vs: bool = true,
	sort_eigenvalues: bool = false,
	select_fn: lapack.LAPACK_C_SELECT1 = nil,
	sense: SchurSenseMode = .None,
	allocator := context.allocator,
) -> (
	W: []complex64,
	VS: Matrix(complex64),
	sdim: Blas_Int,
	rconde: f32,
	rcondv: f32,
	info: Info,
) {
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)

	// Allocate eigenvalue array
	W = builtin.make([]complex64, n, allocator)

	// Prepare job parameters
	jobvs_c := compute_vs ? cstring("V") : cstring("N")
	sort_c := sort_eigenvalues ? cstring("S") : cstring("N")

	// Convert sense mode
	sense_c: cstring
	switch sense {
	case .None:
		sense_c = cstring("N")
	case .Eigenvalues:
		sense_c = cstring("E")
	case .Subspace:
		sense_c = cstring("V")
	case .Both:
		sense_c = cstring("B")
	}

	// Allocate Schur vectors if requested
	ldvs := Blas_Int(1)
	if compute_vs {
		VS = make_matrix(complex64, int(n), int(n), allocator)
		ldvs = Blas_Int(VS.ld)
	}

	// Allocate workspace for sorting
	bwork: []i32
	if sort_eigenvalues {
		bwork = builtin.make([]i32, n, allocator)
		defer builtin.delete(bwork)
	}

	// Allocate real workspace
	rwork_size := sense == .None || sense == .Eigenvalues ? n : 2 * n
	rwork := builtin.make([]f32, rwork_size, allocator)
	defer builtin.delete(rwork)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: complex64

	select_ptr := sort_eigenvalues ? select_fn : lapack.LAPACK_C_SELECT1(nil)
	if select_ptr == nil && sort_eigenvalues {
		select_ptr = proc "c" (w: ^complex64) -> Blas_Int {return 1}
	}

	lapack.cgeesx_(
		jobvs_c,
		sort_c,
		select_ptr,
		sense_c,
		&n,
		raw_data(A.data),
		&lda,
		&sdim,
		raw_data(W),
		compute_vs ? raw_data(VS.data) : nil,
		&ldvs,
		&rconde,
		&rcondv,
		&work_query,
		&lwork,
		raw_data(rwork),
		raw_data(bwork) if sort_eigenvalues else nil,
		&info,
		1,
		1,
		1,
	)

	// Allocate workspace
	lwork = Blas_Int(real(work_query))
	work := builtin.make([]complex64, lwork, allocator)
	defer builtin.delete(work)

	// Perform decomposition
	lapack.cgeesx_(
		jobvs_c,
		sort_c,
		select_ptr,
		sense_c,
		&n,
		raw_data(A.data),
		&lda,
		&sdim,
		raw_data(W),
		compute_vs ? raw_data(VS.data) : nil,
		&ldvs,
		&rconde,
		&rcondv,
		raw_data(work),
		&lwork,
		raw_data(rwork),
		raw_data(bwork) if sort_eigenvalues else nil,
		&info,
		1,
		1,
		1,
	)

	return W, VS, sdim, rconde, rcondv, info
}

m_schur_expert_c128 :: proc(
	A: ^Matrix(complex128),
	compute_vs: bool = true,
	sort_eigenvalues: bool = false,
	select_fn: lapack.LAPACK_Z_SELECT1 = nil,
	sense: SchurSenseMode = .None,
	allocator := context.allocator,
) -> (
	W: []complex128,
	VS: Matrix(complex128),
	sdim: Blas_Int,
	rconde: f64,
	rcondv: f64,
	info: Info,
) {
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)

	// Allocate eigenvalue array
	W = builtin.make([]complex128, n, allocator)

	// Prepare job parameters
	jobvs_c := compute_vs ? cstring("V") : cstring("N")
	sort_c := sort_eigenvalues ? cstring("S") : cstring("N")

	// Convert sense mode
	sense_c: cstring
	switch sense {
	case .None:
		sense_c = cstring("N")
	case .Eigenvalues:
		sense_c = cstring("E")
	case .Subspace:
		sense_c = cstring("V")
	case .Both:
		sense_c = cstring("B")
	}

	// Allocate Schur vectors if requested
	ldvs := Blas_Int(1)
	if compute_vs {
		VS = make_matrix(complex128, int(n), int(n), allocator)
		ldvs = Blas_Int(VS.ld)
	}

	// Allocate workspace for sorting
	bwork: []i32
	if sort_eigenvalues {
		bwork = builtin.make([]i32, n, allocator)
		defer builtin.delete(bwork)
	}

	// Allocate real workspace
	rwork_size := sense == .None || sense == .Eigenvalues ? n : 2 * n
	rwork := builtin.make([]f64, rwork_size, allocator)
	defer builtin.delete(rwork)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: complex128

	select_ptr := sort_eigenvalues ? select_fn : lapack.LAPACK_Z_SELECT1(nil)
	if select_ptr == nil && sort_eigenvalues {
		select_ptr = proc "c" (w: ^complex128) -> Blas_Int {return 1}
	}

	lapack.zgeesx_(
		jobvs_c,
		sort_c,
		select_ptr,
		sense_c,
		&n,
		raw_data(A.data),
		&lda,
		&sdim,
		raw_data(W),
		compute_vs ? raw_data(VS.data) : nil,
		&ldvs,
		&rconde,
		&rcondv,
		&work_query,
		&lwork,
		raw_data(rwork),
		raw_data(bwork) if sort_eigenvalues else nil,
		&info,
		1,
		1,
		1,
	)

	// Allocate workspace
	lwork = Blas_Int(real(work_query))
	work := builtin.make([]complex128, lwork, allocator)
	defer builtin.delete(work)

	// Perform decomposition
	lapack.zgeesx_(
		jobvs_c,
		sort_c,
		select_ptr,
		sense_c,
		&n,
		raw_data(A.data),
		&lda,
		&sdim,
		raw_data(W),
		compute_vs ? raw_data(VS.data) : nil,
		&ldvs,
		&rconde,
		&rcondv,
		raw_data(work),
		&lwork,
		raw_data(rwork),
		raw_data(bwork) if sort_eigenvalues else nil,
		&info,
		1,
		1,
		1,
	)

	return W, VS, sdim, rconde, rcondv, info
}

// Helper function to interpret condition numbers
// Returns true if the eigenvalue problem is well-conditioned
m_schur_is_well_conditioned :: proc(
	rconde, rcondv: $T,
	threshold: T = 1e-6,
) -> (
	eigenvalues_ok: bool,
	subspace_ok: bool,
) where T == f32 ||
	T == f64 {
	return rconde > threshold, rcondv > threshold
}

// ===================================================================================
// EIGENVALUE AND EIGENVECTOR COMPUTATION
// ===================================================================================

// Compute eigenvalues and eigenvectors of general matrix
m_eigen :: proc {
	m_eigen_real,
	m_eigen_c64,
	m_eigen_c128,
}

m_eigen_real :: proc(
	A: ^Matrix($T),
	compute_left: bool = false, // Compute left eigenvectors
	compute_right: bool = true, // Compute right eigenvectors
	allocator := context.allocator,
) -> (
	WR: []T,
	WI: []T,
	VL: Matrix(T),
	VR: Matrix(T),// Real parts of eigenvalues
	info: Info, // Imaginary parts of eigenvalues// Left eigenvectors (if requested)// Right eigenvectors (if requested)
) where T == f32 || T == f64 {
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)

	// Allocate eigenvalue arrays
	WR = builtin.make([]T, n, allocator)
	WI = builtin.make([]T, n, allocator)

	// Prepare job parameters
	jobvl_c := compute_left ? cstring("V") : cstring("N")
	jobvr_c := compute_right ? cstring("V") : cstring("N")

	// Allocate eigenvector matrices if requested
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
		lapack.sgeev_(
			jobvl_c,
			jobvr_c,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(WR),
			raw_data(WI),
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

		// Allocate workspace
		lwork = Blas_Int(work_query)
		work := builtin.make([]T, lwork, allocator)
		defer builtin.delete(work)

		// Compute eigenvalues and eigenvectors
		lapack.sgeev_(
			jobvl_c,
			jobvr_c,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(WR),
			raw_data(WI),
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
	} else when T == f64 {
		lapack.dgeev_(
			jobvl_c,
			jobvr_c,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(WR),
			raw_data(WI),
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

		// Allocate workspace
		lwork = Blas_Int(work_query)
		work := builtin.make([]T, lwork, allocator)
		defer builtin.delete(work)

		// Compute eigenvalues and eigenvectors
		lapack.dgeev_(
			jobvl_c,
			jobvr_c,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(WR),
			raw_data(WI),
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

	return WR, WI, VL, VR, info
}

m_eigen_c64 :: proc(
	A: ^Matrix(complex64),
	compute_left: bool = false,
	compute_right: bool = true,
	allocator := context.allocator,
) -> (
	W: []complex64,
	VL: Matrix(complex64),
	VR: Matrix(complex64),
	info: Info, // Eigenvalues// Left eigenvectors (if requested)// Right eigenvectors (if requested)
) {
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)

	// Allocate eigenvalue array
	W = builtin.make([]complex64, n, allocator)

	// Prepare job parameters
	jobvl_c := compute_left ? cstring("V") : cstring("N")
	jobvr_c := compute_right ? cstring("V") : cstring("N")

	// Allocate eigenvector matrices if requested
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
	rwork := builtin.make([]f32, 2 * n, allocator)
	defer builtin.delete(rwork)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: complex64

	lapack.cgeev_(
		jobvl_c,
		jobvr_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(W),
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
	lapack.cgeev_(
		jobvl_c,
		jobvr_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(W),
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

	return W, VL, VR, info
}

m_eigen_c128 :: proc(
	A: ^Matrix(complex128),
	compute_left: bool = false,
	compute_right: bool = true,
	allocator := context.allocator,
) -> (
	W: []complex128,
	VL: Matrix(complex128),
	VR: Matrix(complex128),
	info: Info,
) {
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)

	// Allocate eigenvalue array
	W = builtin.make([]complex128, n, allocator)

	// Prepare job parameters
	jobvl_c := compute_left ? cstring("V") : cstring("N")
	jobvr_c := compute_right ? cstring("V") : cstring("N")

	// Allocate eigenvector matrices if requested
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
	rwork := builtin.make([]f64, 2 * n, allocator)
	defer builtin.delete(rwork)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: complex128

	lapack.zgeev_(
		jobvl_c,
		jobvr_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(W),
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
	lapack.zgeev_(
		jobvl_c,
		jobvr_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(W),
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

	return W, VL, VR, info
}

// Expert eigenvalue/eigenvector computation with balancing and condition numbers
m_eigen_expert :: proc {
	m_eigen_expert_real,
	m_eigen_expert_c64,
	m_eigen_expert_c128,
}

// Sense mode for eigenvector condition numbers
EigenSenseMode :: enum {
	None         = 0, // No condition numbers
	Eigenvalues  = 1, // Condition numbers for eigenvalues
	Eigenvectors = 2, // Condition numbers for eigenvectors
	Both         = 3, // Both eigenvalue and eigenvector condition numbers
}

m_eigen_expert_real :: proc(
	A: ^Matrix($T),
	balance: BalanceMode = .Both, // Balancing mode
	compute_left: bool = false,
	compute_right: bool = true,
	sense: EigenSenseMode = .None,
	allocator := context.allocator,
) -> (
	WR: []T,
	WI: []T,
	VL: Matrix(T),
	VR: Matrix(T),// Real parts of eigenvalues
	ilo, ihi: Blas_Int,// Imaginary parts of eigenvalues
	scale: []T,// Left eigenvectors
	abnrm: T,// Right eigenvectors
	rconde: []T,// Balancing indices
	rcondv: []T,// Scaling factors from balancing
	info: Info, // 1-norm of balanced matrix// Condition numbers for eigenvalues// Condition numbers for eigenvectors
) where T == f32 || T == f64 {
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)

	// Allocate eigenvalue arrays
	WR = builtin.make([]T, n, allocator)
	WI = builtin.make([]T, n, allocator)

	// Allocate scaling factors
	scale = builtin.make([]T, n, allocator)

	// Prepare job parameters
	balanc_c: cstring
	switch balance {
	case .None:
		balanc_c = cstring("N")
	case .Permute:
		balanc_c = cstring("P")
	case .Scale:
		balanc_c = cstring("S")
	case .Both:
		balanc_c = cstring("B")
	}

	jobvl_c := compute_left ? cstring("V") : cstring("N")
	jobvr_c := compute_right ? cstring("V") : cstring("N")

	sense_c: cstring
	switch sense {
	case .None:
		sense_c = cstring("N")
	case .Eigenvalues:
		sense_c = cstring("E")
	case .Eigenvectors:
		sense_c = cstring("V")
	case .Both:
		sense_c = cstring("B")
	}

	// Allocate eigenvector matrices if requested
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

	// Allocate condition number arrays if requested
	if sense == .Eigenvalues || sense == .Both {
		rconde = builtin.make([]T, n, allocator)
	}
	if sense == .Eigenvectors || sense == .Both {
		rcondv = builtin.make([]T, n, allocator)
	}

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: T
	iwork := builtin.make([]i32, 2 * n - 2, allocator)
	defer builtin.delete(iwork)

	when T == f32 {
		lapack.sgeevx_(
			balanc_c,
			jobvl_c,
			jobvr_c,
			sense_c,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(WR),
			raw_data(WI),
			compute_left ? raw_data(VL.data) : nil,
			&ldvl,
			compute_right ? raw_data(VR.data) : nil,
			&ldvr,
			&ilo,
			&ihi,
			raw_data(scale),
			&abnrm,
			rconde != nil ? raw_data(rconde) : nil,
			rcondv != nil ? raw_data(rcondv) : nil,
			&work_query,
			&lwork,
			raw_data(iwork),
			&info,
			1,
			1,
			1,
			1,
		)

		// Allocate workspace
		lwork = Blas_Int(work_query)
		work := builtin.make([]T, lwork, allocator)
		defer builtin.delete(work)

		// Compute eigenvalues and eigenvectors
		lapack.sgeevx_(
			balanc_c,
			jobvl_c,
			jobvr_c,
			sense_c,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(WR),
			raw_data(WI),
			compute_left ? raw_data(VL.data) : nil,
			&ldvl,
			compute_right ? raw_data(VR.data) : nil,
			&ldvr,
			&ilo,
			&ihi,
			raw_data(scale),
			&abnrm,
			rconde != nil ? raw_data(rconde) : nil,
			rcondv != nil ? raw_data(rcondv) : nil,
			raw_data(work),
			&lwork,
			raw_data(iwork),
			&info,
			1,
			1,
			1,
			1,
		)
	} else when T == f64 {
		lapack.dgeevx_(
			balanc_c,
			jobvl_c,
			jobvr_c,
			sense_c,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(WR),
			raw_data(WI),
			compute_left ? raw_data(VL.data) : nil,
			&ldvl,
			compute_right ? raw_data(VR.data) : nil,
			&ldvr,
			&ilo,
			&ihi,
			raw_data(scale),
			&abnrm,
			rconde != nil ? raw_data(rconde) : nil,
			rcondv != nil ? raw_data(rcondv) : nil,
			&work_query,
			&lwork,
			raw_data(iwork),
			&info,
			1,
			1,
			1,
			1,
		)

		// Allocate workspace
		lwork = Blas_Int(work_query)
		work := builtin.make([]T, lwork, allocator)
		defer builtin.delete(work)

		// Compute eigenvalues and eigenvectors
		lapack.dgeevx_(
			balanc_c,
			jobvl_c,
			jobvr_c,
			sense_c,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(WR),
			raw_data(WI),
			compute_left ? raw_data(VL.data) : nil,
			&ldvl,
			compute_right ? raw_data(VR.data) : nil,
			&ldvr,
			&ilo,
			&ihi,
			raw_data(scale),
			&abnrm,
			rconde != nil ? raw_data(rconde) : nil,
			rcondv != nil ? raw_data(rcondv) : nil,
			raw_data(work),
			&lwork,
			raw_data(iwork),
			&info,
			1,
			1,
			1,
			1,
		)
	}

	return WR, WI, VL, VR, ilo, ihi, scale, abnrm, rconde, rcondv, info
}

m_eigen_expert_c64 :: proc(
	A: ^Matrix(complex64),
	balance: BalanceMode = .Both,
	compute_left: bool = false,
	compute_right: bool = true,
	sense: EigenSenseMode = .None,
	allocator := context.allocator,
) -> (
	W: []complex64,
	VL: Matrix(complex64),
	VR: Matrix(complex64),
	ilo, ihi: Blas_Int,
	scale: []f32,
	abnrm: f32,
	rconde: []f32,
	rcondv: []f32,
	info: Info,
) {
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)

	// Allocate eigenvalue array
	W = builtin.make([]complex64, n, allocator)

	// Allocate scaling factors
	scale = builtin.make([]f32, n, allocator)

	// Prepare job parameters
	balanc_c: cstring
	switch balance {
	case .None:
		balanc_c = cstring("N")
	case .Permute:
		balanc_c = cstring("P")
	case .Scale:
		balanc_c = cstring("S")
	case .Both:
		balanc_c = cstring("B")
	}

	jobvl_c := compute_left ? cstring("V") : cstring("N")
	jobvr_c := compute_right ? cstring("V") : cstring("N")

	sense_c: cstring
	switch sense {
	case .None:
		sense_c = cstring("N")
	case .Eigenvalues:
		sense_c = cstring("E")
	case .Eigenvectors:
		sense_c = cstring("V")
	case .Both:
		sense_c = cstring("B")
	}

	// Allocate eigenvector matrices if requested
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

	// Allocate condition number arrays if requested
	if sense == .Eigenvalues || sense == .Both {
		rconde = builtin.make([]f32, n, allocator)
	}
	if sense == .Eigenvectors || sense == .Both {
		rcondv = builtin.make([]f32, n, allocator)
	}

	// Allocate real workspace
	rwork_size :=
		sense == .None ? 2 * n : (sense == .Eigenvectors || sense == .Both ? 3 * n : 2 * n)
	rwork := builtin.make([]f32, rwork_size, allocator)
	defer builtin.delete(rwork)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: complex64

	lapack.cgeevx_(
		balanc_c,
		jobvl_c,
		jobvr_c,
		sense_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(W),
		compute_left ? raw_data(VL.data) : nil,
		&ldvl,
		compute_right ? raw_data(VR.data) : nil,
		&ldvr,
		&ilo,
		&ihi,
		raw_data(scale),
		&abnrm,
		rconde != nil ? raw_data(rconde) : nil,
		rcondv != nil ? raw_data(rcondv) : nil,
		&work_query,
		&lwork,
		raw_data(rwork),
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
	lapack.cgeevx_(
		balanc_c,
		jobvl_c,
		jobvr_c,
		sense_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(W),
		compute_left ? raw_data(VL.data) : nil,
		&ldvl,
		compute_right ? raw_data(VR.data) : nil,
		&ldvr,
		&ilo,
		&ihi,
		raw_data(scale),
		&abnrm,
		rconde != nil ? raw_data(rconde) : nil,
		rcondv != nil ? raw_data(rcondv) : nil,
		raw_data(work),
		&lwork,
		raw_data(rwork),
		&info,
		1,
		1,
		1,
		1,
	)

	return W, VL, VR, ilo, ihi, scale, abnrm, rconde, rcondv, info
}

m_eigen_expert_c128 :: proc(
	A: ^Matrix(complex128),
	balance: BalanceMode = .Both,
	compute_left: bool = false,
	compute_right: bool = true,
	sense: EigenSenseMode = .None,
	allocator := context.allocator,
) -> (
	W: []complex128,
	VL: Matrix(complex128),
	VR: Matrix(complex128),
	ilo, ihi: Blas_Int,
	scale: []f64,
	abnrm: f64,
	rconde: []f64,
	rcondv: []f64,
	info: Info,
) {
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)

	// Allocate eigenvalue array
	W = builtin.make([]complex128, n, allocator)

	// Allocate scaling factors
	scale = builtin.make([]f64, n, allocator)

	// Prepare job parameters
	balanc_c: cstring
	switch balance {
	case .None:
		balanc_c = cstring("N")
	case .Permute:
		balanc_c = cstring("P")
	case .Scale:
		balanc_c = cstring("S")
	case .Both:
		balanc_c = cstring("B")
	}

	jobvl_c := compute_left ? cstring("V") : cstring("N")
	jobvr_c := compute_right ? cstring("V") : cstring("N")

	sense_c: cstring
	switch sense {
	case .None:
		sense_c = cstring("N")
	case .Eigenvalues:
		sense_c = cstring("E")
	case .Eigenvectors:
		sense_c = cstring("V")
	case .Both:
		sense_c = cstring("B")
	}

	// Allocate eigenvector matrices if requested
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

	// Allocate condition number arrays if requested
	if sense == .Eigenvalues || sense == .Both {
		rconde = builtin.make([]f64, n, allocator)
	}
	if sense == .Eigenvectors || sense == .Both {
		rcondv = builtin.make([]f64, n, allocator)
	}

	// Allocate real workspace
	rwork_size :=
		sense == .None ? 2 * n : (sense == .Eigenvectors || sense == .Both ? 3 * n : 2 * n)
	rwork := builtin.make([]f64, rwork_size, allocator)
	defer builtin.delete(rwork)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: complex128

	lapack.zgeevx_(
		balanc_c,
		jobvl_c,
		jobvr_c,
		sense_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(W),
		compute_left ? raw_data(VL.data) : nil,
		&ldvl,
		compute_right ? raw_data(VR.data) : nil,
		&ldvr,
		&ilo,
		&ihi,
		raw_data(scale),
		&abnrm,
		rconde != nil ? raw_data(rconde) : nil,
		rcondv != nil ? raw_data(rcondv) : nil,
		&work_query,
		&lwork,
		raw_data(rwork),
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
	lapack.zgeevx_(
		balanc_c,
		jobvl_c,
		jobvr_c,
		sense_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(W),
		compute_left ? raw_data(VL.data) : nil,
		&ldvl,
		compute_right ? raw_data(VR.data) : nil,
		&ldvr,
		&ilo,
		&ihi,
		raw_data(scale),
		&abnrm,
		rconde != nil ? raw_data(rconde) : nil,
		rcondv != nil ? raw_data(rcondv) : nil,
		raw_data(work),
		&lwork,
		raw_data(rwork),
		&info,
		1,
		1,
		1,
		1,
	)

	return W, VL, VR, ilo, ihi, scale, abnrm, rconde, rcondv, info
}

// ===================================================================================
// HESSENBERG REDUCTION
// Reduce general matrix to upper Hessenberg form for eigenvalue computation
// ===================================================================================

// Reduce general matrix to upper Hessenberg form
// A = Q * H * Q^H where H is upper Hessenberg
m_hessenberg :: proc {
	m_hessenberg_real,
	m_hessenberg_c64,
	m_hessenberg_c128,
}

m_hessenberg_real :: proc(
	A: ^Matrix($T),
	ilo, ihi: Blas_Int, // Range of matrix to reduce (from balancing)
	allocator := context.allocator,
) -> (
	tau: []T,
	info: Info, // Scalar factors for Q
) where T == f32 || T == f64 {
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)

	// Validate range
	if ilo < 1 || ihi > n || ilo > ihi {
		return nil, -2
	}

	// Allocate tau array
	tau = builtin.make([]T, n - 1, allocator)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: T

	when T == f32 {
		lapack.sgehrd_(
			&n,
			&ilo,
			&ihi,
			raw_data(A.data),
			&lda,
			raw_data(tau),
			&work_query,
			&lwork,
			&info,
		)

		// Allocate workspace
		lwork = Blas_Int(work_query)
		work := builtin.make([]T, lwork, allocator)
		defer builtin.delete(work)

		// Perform reduction
		lapack.sgehrd_(
			&n,
			&ilo,
			&ihi,
			raw_data(A.data),
			&lda,
			raw_data(tau),
			raw_data(work),
			&lwork,
			&info,
		)
	} else when T == f64 {
		lapack.dgehrd_(
			&n,
			&ilo,
			&ihi,
			raw_data(A.data),
			&lda,
			raw_data(tau),
			&work_query,
			&lwork,
			&info,
		)

		// Allocate workspace
		lwork = Blas_Int(work_query)
		work := builtin.make([]T, lwork, allocator)
		defer builtin.delete(work)

		// Perform reduction
		lapack.dgehrd_(
			&n,
			&ilo,
			&ihi,
			raw_data(A.data),
			&lda,
			raw_data(tau),
			raw_data(work),
			&lwork,
			&info,
		)
	}

	return tau, info
}

m_hessenberg_c64 :: proc(
	A: ^Matrix(complex64),
	ilo, ihi: Blas_Int,
	allocator := context.allocator,
) -> (
	tau: []complex64,
	info: Info,
) {
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)

	// Validate range
	if ilo < 1 || ihi > n || ilo > ihi {
		return nil, -2
	}

	// Allocate tau array
	tau = builtin.make([]complex64, n - 1, allocator)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: complex64

	lapack.cgehrd_(
		&n,
		&ilo,
		&ihi,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		&work_query,
		&lwork,
		&info,
	)

	// Allocate workspace
	lwork = Blas_Int(real(work_query))
	work := builtin.make([]complex64, lwork, allocator)
	defer builtin.delete(work)

	// Perform reduction
	lapack.cgehrd_(
		&n,
		&ilo,
		&ihi,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(work),
		&lwork,
		&info,
	)

	return tau, info
}

m_hessenberg_c128 :: proc(
	A: ^Matrix(complex128),
	ilo, ihi: Blas_Int,
	allocator := context.allocator,
) -> (
	tau: []complex128,
	info: Info,
) {
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)

	// Validate range
	if ilo < 1 || ihi > n || ilo > ihi {
		return nil, -2
	}

	// Allocate tau array
	tau = builtin.make([]complex128, n - 1, allocator)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: complex128

	lapack.zgehrd_(
		&n,
		&ilo,
		&ihi,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		&work_query,
		&lwork,
		&info,
	)

	// Allocate workspace
	lwork = Blas_Int(real(work_query))
	work := builtin.make([]complex128, lwork, allocator)
	defer builtin.delete(work)

	// Perform reduction
	lapack.zgehrd_(
		&n,
		&ilo,
		&ihi,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(work),
		&lwork,
		&info,
	)

	return tau, info
}

// Convenience function for reducing full matrix (no balancing)
m_hessenberg_full :: proc {
	m_hessenberg_full_real,
	m_hessenberg_full_c64,
	m_hessenberg_full_c128,
}

m_hessenberg_full_real :: proc(
	A: ^Matrix($T),
	allocator := context.allocator,
) -> (
	tau: []T,
	info: Info,
) where T == f32 ||
	T == f64 {
	n := Blas_Int(A.cols)
	return m_hessenberg_real(A, 1, n, allocator)
}

m_hessenberg_full_c64 :: proc(
	A: ^Matrix(complex64),
	allocator := context.allocator,
) -> (
	tau: []complex64,
	info: Info,
) {
	n := Blas_Int(A.cols)
	return m_hessenberg_c64(A, 1, n, allocator)
}

m_hessenberg_full_c128 :: proc(
	A: ^Matrix(complex128),
	allocator := context.allocator,
) -> (
	tau: []complex128,
	info: Info,
) {
	n := Blas_Int(A.cols)
	return m_hessenberg_c128(A, 1, n, allocator)
}
