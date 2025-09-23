package openblas

import lapack "./f77"
import "base:builtin"
import "base:intrinsics"

// ===================================================================================
// MATRIX STRUCTURES FOR LAPACK
// ===================================================================================

// Transpose operation modes


// ===================================================================================
// BANDED MATRIX OPERATIONS
// Efficient storage and operations for matrices with limited bandwidth
// ===================================================================================

// Reduce general banded matrix to bidiagonal form
// AB = Q * B * P^T where B is bidiagonal
m_banded_to_bidiag :: proc {
	m_banded_to_bidiag_real,
	m_banded_to_bidiag_c64,
	m_banded_to_bidiag_c128,
}

m_banded_to_bidiag_real :: proc(
	AB: ^Matrix($T), // Must be a banded matrix
	compute_q: bool = false, // Compute Q matrix
	compute_pt: bool = false, // Compute P^T matrix
	C: ^Matrix(T) = nil, // Apply transformation to C
	allocator := context.allocator,
) -> (
	D: []T,
	E: []T,
	Q: Matrix(T),
	PT: Matrix(T),
	info: Info, // Diagonal elements of B// Off-diagonal elements of B// Left orthogonal matrix (if requested)// Right orthogonal matrix transposed (if requested)
) where T == f32 || T == f64 {
	assert(AB.format == .Banded, "Matrix must be in banded format")

	m := Blas_Int(AB.rows)
	n := Blas_Int(AB.cols)
	kl := Blas_Int(AB.storage.banded.kl)
	ku := Blas_Int(AB.storage.banded.ku)
	ldab := Blas_Int(AB.ld)

	// Allocate output
	min_mn := min(m, n)
	D = make([]T, min_mn, allocator)
	E = make([]T, min_mn - 1, allocator)

	// Prepare vect parameter
	vect_c: cstring
	if compute_q && compute_pt {
		vect_c = cstring("B") // Both
	} else if compute_q {
		vect_c = cstring("Q") // Only Q
	} else if compute_pt {
		vect_c = cstring("P") // Only P
	} else {
		vect_c = cstring("N") // None
	}

	// Handle Q and PT allocation
	ldq := Blas_Int(1)
	ldpt := Blas_Int(1)
	if compute_q {
		Q = make_matrix(T, int(m), int(min_mn), allocator)
		ldq = Blas_Int(Q.ld)
	}
	if compute_pt {
		PT = make_matrix(T, int(min_mn), int(n), allocator)
		ldpt = Blas_Int(PT.ld)
	}

	// Handle C matrix
	ncc := Blas_Int(0)
	ldc := Blas_Int(1)
	if C != nil {
		ncc = Blas_Int(C.cols)
		ldc = Blas_Int(C.ld)
	}

	// Allocate workspace
	work := make([]T, 2 * max(int(m), int(n)), allocator)
	defer delete(work)

	when T == f32 {
		lapack.sgbbrd_(
			vect_c,
			&m,
			&n,
			&ncc,
			&kl,
			&ku,
			raw_data(AB.data),
			&ldab,
			raw_data(D),
			raw_data(E),
			compute_q ? raw_data(Q.data) : nil,
			&ldq,
			compute_pt ? raw_data(PT.data) : nil,
			&ldpt,
			C != nil ? raw_data(C.data) : nil,
			&ldc,
			raw_data(work),
			&info,
			1,
		)
	} else when T == f64 {
		lapack.dgbbrd_(
			vect_c,
			&m,
			&n,
			&ncc,
			&kl,
			&ku,
			raw_data(AB.data),
			&ldab,
			raw_data(D),
			raw_data(E),
			compute_q ? raw_data(Q.data) : nil,
			&ldq,
			compute_pt ? raw_data(PT.data) : nil,
			&ldpt,
			C != nil ? raw_data(C.data) : nil,
			&ldc,
			raw_data(work),
			&info,
			1,
		)
	}

	return D, E, Q, PT, info
}

m_banded_to_bidiag_c64 :: proc(
	AB: ^Matrix(complex64), // Must be a banded matrix
	compute_q: bool = false,
	compute_pt: bool = false,
	C: ^Matrix(complex64) = nil,
	allocator := context.allocator,
) -> (
	D: []f32,
	E: []f32,
	Q: Matrix(complex64),
	PT: Matrix(complex64),
	info: Info, // Real diagonal elements// Real off-diagonal elements// Left unitary matrix// Right unitary matrix transposed
) {
	assert(AB.format == .Banded, "Matrix must be in banded format")

	m := Blas_Int(AB.rows)
	n := Blas_Int(AB.cols)
	kl := Blas_Int(AB.storage.banded.kl)
	ku := Blas_Int(AB.storage.banded.ku)
	ldab := Blas_Int(AB.ld)

	// Allocate output
	min_mn := min(m, n)
	D = builtin.make([]f32, min_mn, allocator)
	E = builtin.make([]f32, min_mn - 1, allocator)

	// Prepare vect parameter
	vect_c: cstring
	if compute_q && compute_pt {
		vect_c = cstring("B")
	} else if compute_q {
		vect_c = cstring("Q")
	} else if compute_pt {
		vect_c = cstring("P")
	} else {
		vect_c = cstring("N")
	}

	// Handle Q and PT allocation
	ldq := Blas_Int(1)
	ldpt := Blas_Int(1)
	if compute_q {
		Q = make_matrix(complex64, int(m), int(min_mn), allocator)
		ldq = Blas_Int(Q.ld)
	}
	if compute_pt {
		PT = make_matrix(complex64, int(min_mn), int(n), allocator)
		ldpt = Blas_Int(PT.ld)
	}

	// Handle C matrix
	ncc := Blas_Int(0)
	ldc := Blas_Int(1)
	if C != nil {
		ncc = Blas_Int(C.cols)
		ldc = Blas_Int(C.ld)
	}

	// Allocate workspace
	work := builtin.make([]complex64, max(int(m), int(n)), allocator)
	rwork := builtin.make([]f32, max(int(m), int(n)), allocator)
	defer builtin.delete(work)
	defer builtin.delete(rwork)

	lapack.cgbbrd_(
		vect_c,
		&m,
		&n,
		&ncc,
		&kl,
		&ku,
		raw_data(AB.data),
		&ldab,
		raw_data(D),
		raw_data(E),
		compute_q ? raw_data(Q.data) : nil,
		&ldq,
		compute_pt ? raw_data(PT.data) : nil,
		&ldpt,
		C != nil ? raw_data(C.data) : nil,
		&ldc,
		raw_data(work),
		raw_data(rwork),
		&info,
		1,
	)

	return D, E, Q, PT, info
}

m_banded_to_bidiag_c128 :: proc(
	AB: ^Matrix(complex128), // Must be a banded matrix
	compute_q: bool = false,
	compute_pt: bool = false,
	C: ^Matrix(complex128) = nil,
	allocator := context.allocator,
) -> (
	D: []f64,
	E: []f64,
	Q: Matrix(complex128),
	PT: Matrix(complex128),
	info: Info, // Real diagonal elements// Real off-diagonal elements// Left unitary matrix// Right unitary matrix transposed
) {
	assert(AB.format == .Banded, "Matrix must be in banded format")

	m := Blas_Int(AB.rows)
	n := Blas_Int(AB.cols)
	kl := Blas_Int(AB.storage.banded.kl)
	ku := Blas_Int(AB.storage.banded.ku)
	ldab := Blas_Int(AB.ld)

	// Allocate output
	min_mn := min(m, n)
	D = builtin.make([]f64, min_mn, allocator)
	E = builtin.make([]f64, min_mn - 1, allocator)

	// Prepare vect parameter
	vect_c: cstring
	if compute_q && compute_pt {
		vect_c = cstring("B")
	} else if compute_q {
		vect_c = cstring("Q")
	} else if compute_pt {
		vect_c = cstring("P")
	} else {
		vect_c = cstring("N")
	}

	// Handle Q and PT allocation
	ldq := Blas_Int(1)
	ldpt := Blas_Int(1)
	if compute_q {
		Q = make_matrix(complex128, int(m), int(min_mn), allocator)
		ldq = Blas_Int(Q.ld)
	}
	if compute_pt {
		PT = make_matrix(complex128, int(min_mn), int(n), allocator)
		ldpt = Blas_Int(PT.ld)
	}

	// Handle C matrix
	ncc := Blas_Int(0)
	ldc := Blas_Int(1)
	if C != nil {
		ncc = Blas_Int(C.cols)
		ldc = Blas_Int(C.ld)
	}

	// Allocate workspace
	work := builtin.make([]complex128, max(int(m), int(n)), allocator)
	rwork := builtin.make([]f64, max(int(m), int(n)), allocator)
	defer builtin.delete(work)
	defer builtin.delete(rwork)

	lapack.zgbbrd_(
		vect_c,
		&m,
		&n,
		&ncc,
		&kl,
		&ku,
		raw_data(AB.data),
		&ldab,
		raw_data(D),
		raw_data(E),
		compute_q ? raw_data(Q.data) : nil,
		&ldq,
		compute_pt ? raw_data(PT.data) : nil,
		&ldpt,
		C != nil ? raw_data(C.data) : nil,
		&ldc,
		raw_data(work),
		raw_data(rwork),
		&info,
		1,
	)

	return D, E, Q, PT, info
}

// Estimate condition number of banded matrix
// Requires prior factorization with m_banded_lu
m_banded_cond :: proc {
	m_banded_cond_real,
	m_banded_cond_c64,
	m_banded_cond_c128,
}

m_banded_cond_real :: proc(
	AB: ^Matrix($T), // Factored banded matrix from m_banded_lu
	ipiv: []i32, // Pivot indices from m_banded_lu
	anorm: T, // 1-norm of original matrix A
	norm: string = "1", // Norm type: "1", "O", or "I"
	allocator := context.allocator,
) -> (
	rcond: T,
	info: Info, // Reciprocal condition number
) where T == f32 || T == f64 {
	assert(AB.format == .Banded, "Matrix must be in banded format")

	n := Blas_Int(AB.cols)
	kl := Blas_Int(AB.storage.banded.kl)
	ku := Blas_Int(AB.storage.banded.ku)
	ldab := Blas_Int(AB.ld)

	norm_c := cstring(raw_data(norm))

	when T == f32 {
		work := make([]f32, 3 * int(n), allocator)
		iwork := make([]i32, int(n), allocator)
		defer delete(work)
		defer delete(iwork)

		lapack.sgbcon_(
			norm_c,
			&n,
			&kl,
			&ku,
			raw_data(AB.data),
			&ldab,
			raw_data(ipiv),
			&anorm,
			&rcond,
			raw_data(work),
			raw_data(iwork),
			&info,
			1,
		)
	} else when T == f64 {
		work := make([]f64, 3 * int(n), allocator)
		iwork := make([]i32, int(n), allocator)
		defer delete(work)
		defer delete(iwork)

		lapack.dgbcon_(
			norm_c,
			&n,
			&kl,
			&ku,
			raw_data(AB.data),
			&ldab,
			raw_data(ipiv),
			&anorm,
			&rcond,
			raw_data(work),
			raw_data(iwork),
			&info,
			1,
		)
	}

	return rcond, info
}

m_banded_cond_c64 :: proc(
	AB: ^Matrix(complex64),
	ipiv: []i32,
	anorm: f32,
	norm: string = "1",
	allocator := context.allocator,
) -> (
	rcond: f32,
	info: Info,
) {
	assert(AB.format == .Banded, "Matrix must be in banded format")
	anorm := anorm

	n := Blas_Int(AB.cols)
	kl := Blas_Int(AB.storage.banded.kl)
	ku := Blas_Int(AB.storage.banded.ku)
	ldab := Blas_Int(AB.ld)

	norm_c := cstring(raw_data(norm))

	work := builtin.make([]complex64, 2 * int(n), allocator)
	rwork := builtin.make([]f32, int(n), allocator)
	defer builtin.delete(work)
	defer builtin.delete(rwork)

	lapack.cgbcon_(
		norm_c,
		&n,
		&kl,
		&ku,
		raw_data(AB.data),
		&ldab,
		raw_data(ipiv),
		&anorm,
		&rcond,
		raw_data(work),
		raw_data(rwork),
		&info,
		1,
	)

	return rcond, info
}

m_banded_cond_c128 :: proc(
	AB: ^Matrix(complex128),
	ipiv: []i32,
	anorm: f64,
	norm: string = "1",
	allocator := context.allocator,
) -> (
	rcond: f64,
	info: Info,
) {
	assert(AB.format == .Banded, "Matrix must be in banded format")
	anorm := anorm

	n := Blas_Int(AB.cols)
	kl := Blas_Int(AB.storage.banded.kl)
	ku := Blas_Int(AB.storage.banded.ku)
	ldab := Blas_Int(AB.ld)

	norm_c := cstring(raw_data(norm))

	work := builtin.make([]complex128, 2 * int(n), allocator)
	rwork := builtin.make([]f64, int(n), allocator)
	defer builtin.delete(work)
	defer builtin.delete(rwork)

	lapack.zgbcon_(
		norm_c,
		&n,
		&kl,
		&ku,
		raw_data(AB.data),
		&ldab,
		raw_data(ipiv),
		&anorm,
		&rcond,
		raw_data(work),
		raw_data(rwork),
		&info,
		1,
	)

	return rcond, info
}

// Compute row and column equilibration scale factors for banded matrix
// Returns scale factors to improve condition number
m_banded_equilibrate :: proc {
	m_banded_equilibrate_real,
	m_banded_equilibrate_c64,
	m_banded_equilibrate_c128,
}

m_banded_equilibrate_real :: proc(
	AB: ^Matrix($T), // Banded matrix to equilibrate
	allocator := context.allocator,
) -> (
	R: []T,
	C: []T,
	rowcnd: T,
	colcnd: T,
	amax: T,
	info: Info, // Row scale factors// Column scale factors// Ratio of smallest to largest row scale// Ratio of smallest to largest column scale// Absolute value of largest matrix element
) where T == f32 || T == f64 {
	assert(AB.format == .Banded, "Matrix must be in banded format")

	m := Blas_Int(AB.rows)
	n := Blas_Int(AB.cols)
	kl := Blas_Int(AB.storage.banded.kl)
	ku := Blas_Int(AB.storage.banded.ku)
	ldab := Blas_Int(AB.ld)

	// Allocate scale factors
	R = make([]T, m, allocator)
	C = make([]T, n, allocator)

	when T == f32 {
		lapack.sgbequ_(
			&m,
			&n,
			&kl,
			&ku,
			raw_data(AB.data),
			&ldab,
			raw_data(R),
			raw_data(C),
			&rowcnd,
			&colcnd,
			&amax,
			&info,
		)
	} else when T == f64 {
		lapack.dgbequ_(
			&m,
			&n,
			&kl,
			&ku,
			raw_data(AB.data),
			&ldab,
			raw_data(R),
			raw_data(C),
			&rowcnd,
			&colcnd,
			&amax,
			&info,
		)
	}

	return R, C, rowcnd, colcnd, amax, info
}

m_banded_equilibrate_c64 :: proc(
	AB: ^Matrix(complex64),
	allocator := context.allocator,
) -> (
	R: []f32,
	C: []f32,
	rowcnd: f32,
	colcnd: f32,
	amax: f32,
	info: Info, // Row scale factors (real)// Column scale factors (real)// Ratio of smallest to largest row scale// Ratio of smallest to largest column scale// Absolute value of largest matrix element
) {
	assert(AB.format == .Banded, "Matrix must be in banded format")

	m := Blas_Int(AB.rows)
	n := Blas_Int(AB.cols)
	kl := Blas_Int(AB.storage.banded.kl)
	ku := Blas_Int(AB.storage.banded.ku)
	ldab := Blas_Int(AB.ld)

	// Allocate scale factors
	R = make([]f32, m, allocator)
	C = make([]f32, n, allocator)

	lapack.cgbequ_(
		&m,
		&n,
		&kl,
		&ku,
		raw_data(AB.data),
		&ldab,
		raw_data(R),
		raw_data(C),
		&rowcnd,
		&colcnd,
		&amax,
		&info,
	)

	return R, C, rowcnd, colcnd, amax, info
}

m_banded_equilibrate_c128 :: proc(
	AB: ^Matrix(complex128),
	allocator := context.allocator,
) -> (
	R: []f64,
	C: []f64,
	rowcnd: f64,
	colcnd: f64,
	amax: f64,
	info: Info, // Row scale factors (real)// Column scale factors (real)// Ratio of smallest to largest row scale// Ratio of smallest to largest column scale// Absolute value of largest matrix element
) {
	assert(AB.format == .Banded, "Matrix must be in banded format")

	m := Blas_Int(AB.rows)
	n := Blas_Int(AB.cols)
	kl := Blas_Int(AB.storage.banded.kl)
	ku := Blas_Int(AB.storage.banded.ku)
	ldab := Blas_Int(AB.ld)

	// Allocate scale factors
	R = make([]f64, m, allocator)
	C = make([]f64, n, allocator)

	lapack.zgbequ_(
		&m,
		&n,
		&kl,
		&ku,
		raw_data(AB.data),
		&ldab,
		raw_data(R),
		raw_data(C),
		&rowcnd,
		&colcnd,
		&amax,
		&info,
	)

	return R, C, rowcnd, colcnd, amax, info
}

// Improved equilibration with better algorithm (LAPACK 3.x)
m_banded_equilibrate_improved :: proc {
	m_banded_equilibrate_improved_real,
	m_banded_equilibrate_improved_c64,
	m_banded_equilibrate_improved_c128,
}

m_banded_equilibrate_improved_real :: proc(
	AB: ^Matrix($T),
	allocator := context.allocator,
) -> (
	R: []T,
	C: []T,
	rowcnd: T,
	colcnd: T,
	amax: T,
	info: Info, // Row scale factors// Column scale factors// Ratio of smallest to largest row scale// Ratio of smallest to largest column scale// Absolute value of largest matrix element
) where T == f32 || T == f64 {
	assert(AB.format == .Banded, "Matrix must be in banded format")

	m := Blas_Int(AB.rows)
	n := Blas_Int(AB.cols)
	kl := Blas_Int(AB.storage.banded.kl)
	ku := Blas_Int(AB.storage.banded.ku)
	ldab := Blas_Int(AB.ld)

	// Allocate scale factors
	R = make([]T, m, allocator)
	C = make([]T, n, allocator)

	when T == f32 {
		lapack.sgbequb_(
			&m,
			&n,
			&kl,
			&ku,
			raw_data(AB.data),
			&ldab,
			raw_data(R),
			raw_data(C),
			&rowcnd,
			&colcnd,
			&amax,
			&info,
		)
	} else when T == f64 {
		lapack.dgbequb_(
			&m,
			&n,
			&kl,
			&ku,
			raw_data(AB.data),
			&ldab,
			raw_data(R),
			raw_data(C),
			&rowcnd,
			&colcnd,
			&amax,
			&info,
		)
	}

	return R, C, rowcnd, colcnd, amax, info
}

m_banded_equilibrate_improved_c64 :: proc(
	AB: ^Matrix(complex64),
	allocator := context.allocator,
) -> (
	R: []f32,
	C: []f32,
	rowcnd: f32,
	colcnd: f32,
	amax: f32,
	info: Info,
) {
	assert(AB.format == .Banded, "Matrix must be in banded format")

	m := Blas_Int(AB.rows)
	n := Blas_Int(AB.cols)
	kl := Blas_Int(AB.storage.banded.kl)
	ku := Blas_Int(AB.storage.banded.ku)
	ldab := Blas_Int(AB.ld)

	// Allocate scale factors
	R = make([]f32, m, allocator)
	C = make([]f32, n, allocator)

	lapack.cgbequb_(
		&m,
		&n,
		&kl,
		&ku,
		raw_data(AB.data),
		&ldab,
		raw_data(R),
		raw_data(C),
		&rowcnd,
		&colcnd,
		&amax,
		&info,
	)

	return R, C, rowcnd, colcnd, amax, info
}

m_banded_equilibrate_improved_c128 :: proc(
	AB: ^Matrix(complex128),
	allocator := context.allocator,
) -> (
	R: []f64,
	C: []f64,
	rowcnd: f64,
	colcnd: f64,
	amax: f64,
	info: Info,
) {
	assert(AB.format == .Banded, "Matrix must be in banded format")

	m := Blas_Int(AB.rows)
	n := Blas_Int(AB.cols)
	kl := Blas_Int(AB.storage.banded.kl)
	ku := Blas_Int(AB.storage.banded.ku)
	ldab := Blas_Int(AB.ld)

	// Allocate scale factors
	R = make([]f64, m, allocator)
	C = make([]f64, n, allocator)

	lapack.zgbequb_(
		&m,
		&n,
		&kl,
		&ku,
		raw_data(AB.data),
		&ldab,
		raw_data(R),
		raw_data(C),
		&rowcnd,
		&colcnd,
		&amax,
		&info,
	)

	return R, C, rowcnd, colcnd, amax, info
}

// ===================================================================================
// FACTORIZATION
// ===================================================================================

// LU factorization of banded matrix
m_banded_factor :: proc {
	m_banded_factor_real,
	m_banded_factor_c64,
	m_banded_factor_c128,
}

m_banded_factor_real :: proc(
	AB: ^Matrix($T), // Banded matrix (will be overwritten with LU factorization)
	allocator := context.allocator,
) -> (
	ipiv: []i32,
	info: Info, // Pivot indices
) where T == f32 || T == f64 {
	assert(AB.format == .Banded, "AB must be in banded format")

	m := Blas_Int(AB.rows)
	n := Blas_Int(AB.cols)
	kl := Blas_Int(AB.storage.banded.kl)
	ku := Blas_Int(AB.storage.banded.ku)
	ldab := Blas_Int(AB.ld)

	// Allocate pivot array
	ipiv = make([]i32, min(m, n), allocator)

	when T == f32 {
		lapack.sgbtrf_(&m, &n, &kl, &ku, raw_data(AB.data), &ldab, raw_data(ipiv), &info)
	} else when T == f64 {
		lapack.dgbtrf_(&m, &n, &kl, &ku, raw_data(AB.data), &ldab, raw_data(ipiv), &info)
	}

	return ipiv, info
}

m_banded_factor_c64 :: proc(
	AB: ^Matrix(complex64),
	allocator := context.allocator,
) -> (
	ipiv: []i32,
	info: Info,
) {
	assert(AB.format == .Banded, "AB must be in banded format")

	m := Blas_Int(AB.rows)
	n := Blas_Int(AB.cols)
	kl := Blas_Int(AB.storage.banded.kl)
	ku := Blas_Int(AB.storage.banded.ku)
	ldab := Blas_Int(AB.ld)

	// Allocate pivot array
	ipiv = make([]i32, min(m, n), allocator)

	lapack.cgbtrf_(&m, &n, &kl, &ku, raw_data(AB.data), &ldab, raw_data(ipiv), &info)

	return ipiv, info
}

m_banded_factor_c128 :: proc(
	AB: ^Matrix(complex128),
	allocator := context.allocator,
) -> (
	ipiv: []i32,
	info: Info,
) {
	assert(AB.format == .Banded, "AB must be in banded format")

	m := Blas_Int(AB.rows)
	n := Blas_Int(AB.cols)
	kl := Blas_Int(AB.storage.banded.kl)
	ku := Blas_Int(AB.storage.banded.ku)
	ldab := Blas_Int(AB.ld)

	// Allocate pivot array
	ipiv = make([]i32, min(m, n), allocator)

	lapack.zgbtrf_(&m, &n, &kl, &ku, raw_data(AB.data), &ldab, raw_data(ipiv), &info)

	return ipiv, info
}

// Solve using pre-computed LU factorization
m_banded_solve_factored :: proc {
	m_banded_solve_factored_real,
	m_banded_solve_factored_c64,
	m_banded_solve_factored_c128,
}

m_banded_solve_factored_real :: proc(
	AB: ^Matrix($T), // LU factorization from m_banded_factor
	ipiv: []i32, // Pivot indices from m_banded_factor
	B: ^Matrix(T), // Right-hand side (will be overwritten with solution)
	trans: TransposeMode = .None,
) -> (
	info: Info,
) where T == f32 || T == f64 {
	assert(AB.format == .Banded, "AB must be in banded format")

	n := Blas_Int(AB.cols)
	kl := Blas_Int(AB.storage.banded.kl)
	ku := Blas_Int(AB.storage.banded.ku)
	nrhs := Blas_Int(B.cols)
	ldab := Blas_Int(AB.ld)
	ldb := Blas_Int(B.ld)

	trans_c: cstring
	switch trans {
	case .None:
		trans_c = cstring("N")
	case .Transpose:
		trans_c = cstring("T")
	case .ConjugateTranspose:
		trans_c = cstring("C")
	}

	when T == f32 {
		lapack.sgbtrs_(
			trans_c,
			&n,
			&kl,
			&ku,
			&nrhs,
			raw_data(AB.data),
			&ldab,
			raw_data(ipiv),
			raw_data(B.data),
			&ldb,
			&info,
			1,
		)
	} else when T == f64 {
		lapack.dgbtrs_(
			trans_c,
			&n,
			&kl,
			&ku,
			&nrhs,
			raw_data(AB.data),
			&ldab,
			raw_data(ipiv),
			raw_data(B.data),
			&ldb,
			&info,
			1,
		)
	}

	return info
}

m_banded_solve_factored_c64 :: proc(
	AB: ^Matrix(complex64),
	ipiv: []i32,
	B: ^Matrix(complex64),
	trans: TransposeMode = .None,
) -> (
	info: Info,
) {
	assert(AB.format == .Banded, "AB must be in banded format")

	n := Blas_Int(AB.cols)
	kl := Blas_Int(AB.storage.banded.kl)
	ku := Blas_Int(AB.storage.banded.ku)
	nrhs := Blas_Int(B.cols)
	ldab := Blas_Int(AB.ld)
	ldb := Blas_Int(B.ld)

	trans_c: cstring
	switch trans {
	case .None:
		trans_c = cstring("N")
	case .Transpose:
		trans_c = cstring("T")
	case .ConjugateTranspose:
		trans_c = cstring("C")
	}

	lapack.cgbtrs_(
		trans_c,
		&n,
		&kl,
		&ku,
		&nrhs,
		raw_data(AB.data),
		&ldab,
		raw_data(ipiv),
		raw_data(B.data),
		&ldb,
		&info,
		1,
	)

	return info
}

m_banded_solve_factored_c128 :: proc(
	AB: ^Matrix(complex128),
	ipiv: []i32,
	B: ^Matrix(complex128),
	trans: TransposeMode = .None,
) -> (
	info: Info,
) {
	assert(AB.format == .Banded, "AB must be in banded format")

	n := Blas_Int(AB.cols)
	kl := Blas_Int(AB.storage.banded.kl)
	ku := Blas_Int(AB.storage.banded.ku)
	nrhs := Blas_Int(B.cols)
	ldab := Blas_Int(AB.ld)
	ldb := Blas_Int(B.ld)

	trans_c: cstring
	switch trans {
	case .None:
		trans_c = cstring("N")
	case .Transpose:
		trans_c = cstring("T")
	case .ConjugateTranspose:
		trans_c = cstring("C")
	}

	lapack.zgbtrs_(
		trans_c,
		&n,
		&kl,
		&ku,
		&nrhs,
		raw_data(AB.data),
		&ldab,
		raw_data(ipiv),
		raw_data(B.data),
		&ldb,
		&info,
		1,
	)

	return info
}

// ===================================================================================
// LINEAR SYSTEM SOLVERS
// ===================================================================================

// Simple driver for solving banded linear systems
m_banded_solve :: proc {
	m_banded_solve_real,
	m_banded_solve_c64,
	m_banded_solve_c128,
}

m_banded_solve_real :: proc(
	AB: ^Matrix($T), // Banded matrix (will be overwritten with LU factorization)
	B: ^Matrix(T), // Right-hand side (will be overwritten with solution)
	allocator := context.allocator,
) -> (
	ipiv: []i32,
	info: Info, // Pivot indices
) where T == f32 || T == f64 {
	assert(AB.format == .Banded, "AB must be in banded format")

	n := Blas_Int(AB.cols)
	kl := Blas_Int(AB.storage.banded.kl)
	ku := Blas_Int(AB.storage.banded.ku)
	nrhs := Blas_Int(B.cols)
	ldab := Blas_Int(AB.ld)
	ldb := Blas_Int(B.ld)

	// Allocate pivot array
	ipiv = make([]i32, n, allocator)

	when T == f32 {
		lapack.sgbsv_(
			&n,
			&kl,
			&ku,
			&nrhs,
			raw_data(AB.data),
			&ldab,
			raw_data(ipiv),
			raw_data(B.data),
			&ldb,
			&info,
		)
	} else when T == f64 {
		lapack.dgbsv_(
			&n,
			&kl,
			&ku,
			&nrhs,
			raw_data(AB.data),
			&ldab,
			raw_data(ipiv),
			raw_data(B.data),
			&ldb,
			&info,
		)
	}

	return ipiv, info
}

m_banded_solve_c64 :: proc(
	AB: ^Matrix(complex64),
	B: ^Matrix(complex64),
	allocator := context.allocator,
) -> (
	ipiv: []i32,
	info: Info,
) {
	assert(AB.format == .Banded, "AB must be in banded format")

	n := Blas_Int(AB.cols)
	kl := Blas_Int(AB.storage.banded.kl)
	ku := Blas_Int(AB.storage.banded.ku)
	nrhs := Blas_Int(B.cols)
	ldab := Blas_Int(AB.ld)
	ldb := Blas_Int(B.ld)

	// Allocate pivot array
	ipiv = make([]i32, n, allocator)

	lapack.cgbsv_(
		&n,
		&kl,
		&ku,
		&nrhs,
		raw_data(AB.data),
		&ldab,
		raw_data(ipiv),
		raw_data(B.data),
		&ldb,
		&info,
	)

	return ipiv, info
}

m_banded_solve_c128 :: proc(
	AB: ^Matrix(complex128),
	B: ^Matrix(complex128),
	allocator := context.allocator,
) -> (
	ipiv: []i32,
	info: Info,
) {
	assert(AB.format == .Banded, "AB must be in banded format")

	n := Blas_Int(AB.cols)
	kl := Blas_Int(AB.storage.banded.kl)
	ku := Blas_Int(AB.storage.banded.ku)
	nrhs := Blas_Int(B.cols)
	ldab := Blas_Int(AB.ld)
	ldb := Blas_Int(B.ld)

	// Allocate pivot array
	ipiv = make([]i32, n, allocator)

	lapack.zgbsv_(
		&n,
		&kl,
		&ku,
		&nrhs,
		raw_data(AB.data),
		&ldab,
		raw_data(ipiv),
		raw_data(B.data),
		&ldb,
		&info,
	)

	return ipiv, info
}

// Expert driver with equilibration and condition estimation
m_banded_solve_expert :: proc {
	m_banded_solve_expert_real,
	m_banded_solve_expert_c64,
	m_banded_solve_expert_c128,
}

// Factorization options
FactorizationMode :: enum {
	Equilibrate = 0, // Equilibrate and factor
	Factored    = 1, // Use pre-factored matrix
	NotFactored = 2, // Use matrix as-is, factor it
}

m_banded_solve_expert_real :: proc(
	AB: ^Matrix($T),
	B: ^Matrix(T),
	fact: FactorizationMode = .Equilibrate,
	trans: bool = false,
	AFB: ^Matrix(T) = nil, // Pre-factored matrix (optional)
	ipiv: []i32 = nil, // Pivot indices (optional)
	equed: ^EquilibrationMode = nil, // Equilibration mode (input/output)
	R: []T = nil, // Row scale factors (input/output)
	C: []T = nil, // Column scale factors (input/output)
	allocator := context.allocator,
) -> (
	X: Matrix(T),
	rcond: T,
	ferr: []T,
	berr: []T,
	ipiv_out: []i32,
	AFB_out: Matrix(T),// Solution matrix
	R_out: []T,// Reciprocal condition number
	C_out: []T,// Forward error bounds
	equed_out: EquilibrationMode,// Backward error bounds
	info: Info, // Pivot indices (if not provided)// Factored matrix (if not provided)// Row scale factors (if not provided)// Column scale factors (if not provided)
) where T == f32 || T == f64 {
	assert(AB.format == .Banded, "AB must be in banded format")

	n := Blas_Int(AB.cols)
	kl := Blas_Int(AB.storage.banded.kl)
	ku := Blas_Int(AB.storage.banded.ku)
	nrhs := Blas_Int(B.cols)
	ldab := Blas_Int(AB.ld)
	ldb := Blas_Int(B.ld)

	// Allocate solution matrix
	X = make_matrix(T, int(n), int(nrhs), allocator)
	ldx := Blas_Int(X.ld)

	// Allocate or use provided factored matrix
	if AFB == nil {
		AFB_out = make_matrix(T, int(2 * kl + ku + 1), int(n), allocator)
		AFB_out.format = .Banded
		AFB_out.storage.banded = {
			kl = int(kl),
			ku = int(ku),
		}
	} else {
		AFB_out = AFB^
	}
	ldafb := Blas_Int(AFB_out.ld)

	// Allocate or use provided pivot indices
	if ipiv == nil {
		ipiv_out = make([]i32, n, allocator)
	} else {
		ipiv_out = ipiv
	}

	// Allocate or use provided scale factors
	if R == nil {
		R_out = make([]T, n, allocator)
	} else {
		R_out = R
	}
	if C == nil {
		C_out = make([]T, n, allocator)
	} else {
		C_out = C
	}

	// Convert options to strings
	fact_c: cstring
	switch fact {
	case .Equilibrate:
		fact_c = cstring("E")
	case .Factored:
		fact_c = cstring("F")
	case .NotFactored:
		fact_c = cstring("N")
	}

	trans_c := trans ? cstring("T") : cstring("N")

	// Handle equilibration mode
	equed_c: cstring
	if equed != nil {
		switch equed^ {
		case .None:
			equed_c = cstring("N")
		case .Row:
			equed_c = cstring("R")
		case .Column:
			equed_c = cstring("C")
		case .Both:
			equed_c = cstring("B")
		}
	} else {
		equed_c = cstring("N")
	}

	// Allocate error bounds
	ferr = make([]T, nrhs, allocator)
	berr = make([]T, nrhs, allocator)

	// Allocate workspace
	work := make([]T, 3 * n, allocator)
	iwork := make([]i32, n, allocator)

	when T == f32 {
		lapack.sgbsvx_(
			fact_c,
			trans_c,
			&n,
			&kl,
			&ku,
			&nrhs,
			raw_data(AB.data),
			&ldab,
			raw_data(AFB_out.data),
			&ldafb,
			raw_data(ipiv_out),
			equed_c,
			raw_data(R_out),
			raw_data(C_out),
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
	} else when T == f64 {
		lapack.dgbsvx_(
			fact_c,
			trans_c,
			&n,
			&kl,
			&ku,
			&nrhs,
			raw_data(AB.data),
			&ldab,
			raw_data(AFB_out.data),
			&ldafb,
			raw_data(ipiv_out),
			equed_c,
			raw_data(R_out),
			raw_data(C_out),
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

	// Convert equilibration mode back
	switch equed_c[0] {
	case 'N':
		equed_out = .None
	case 'R':
		equed_out = .Row
	case 'C':
		equed_out = .Column
	case 'B':
		equed_out = .Both
	}

	return X, rcond, ferr, berr, ipiv_out, AFB_out, R_out, C_out, equed_out, info
}

m_banded_solve_expert_c64 :: proc(
	AB: ^Matrix(complex64),
	B: ^Matrix(complex64),
	fact: FactorizationMode = .Equilibrate,
	trans: bool = false,
	AFB: ^Matrix(complex64) = nil,
	ipiv: []i32 = nil,
	equed: ^EquilibrationMode = nil,
	R: []f32 = nil,
	C: []f32 = nil,
	allocator := context.allocator,
) -> (
	X: Matrix(complex64),
	rcond: f32,
	ferr: []f32,
	berr: []f32,
	ipiv_out: []i32,
	AFB_out: Matrix(complex64),
	R_out: []f32,
	C_out: []f32,
	equed_out: EquilibrationMode,
	info: Info,
) {
	assert(AB.format == .Banded, "AB must be in banded format")

	n := Blas_Int(AB.cols)
	kl := Blas_Int(AB.storage.banded.kl)
	ku := Blas_Int(AB.storage.banded.ku)
	nrhs := Blas_Int(B.cols)
	ldab := Blas_Int(AB.ld)
	ldb := Blas_Int(B.ld)

	// Allocate solution matrix
	X = make_matrix(complex64, int(n), int(nrhs), allocator)
	ldx := Blas_Int(X.ld)

	// Allocate or use provided factored matrix
	if AFB == nil {
		AFB_out = make_matrix(complex64, int(2 * kl + ku + 1), int(n), allocator)
		AFB_out.format = .Banded
		AFB_out.storage.banded = {
			kl = int(kl),
			ku = int(ku),
		}
	} else {
		AFB_out = AFB^
	}
	ldafb := Blas_Int(AFB_out.ld)

	// Allocate or use provided pivot indices
	if ipiv == nil {
		ipiv_out = make([]i32, n, allocator)
	} else {
		ipiv_out = ipiv
	}

	// Allocate or use provided scale factors
	if R == nil {
		R_out = make([]f32, n, allocator)
	} else {
		R_out = R
	}
	if C == nil {
		C_out = make([]f32, n, allocator)
	} else {
		C_out = C
	}

	// Convert options to strings
	fact_c: cstring
	switch fact {
	case .Equilibrate:
		fact_c = cstring("E")
	case .Factored:
		fact_c = cstring("F")
	case .NotFactored:
		fact_c = cstring("N")
	}

	trans_c := trans ? cstring("T") : cstring("N")

	// Handle equilibration mode
	equed_c: cstring
	if equed != nil {
		switch equed^ {
		case .None:
			equed_c = cstring("N")
		case .Row:
			equed_c = cstring("R")
		case .Column:
			equed_c = cstring("C")
		case .Both:
			equed_c = cstring("B")
		}
	} else {
		equed_c = cstring("N")
	}

	// Allocate error bounds
	ferr = make([]f32, nrhs, allocator)
	berr = make([]f32, nrhs, allocator)

	// Allocate workspace
	work := make([]complex64, 2 * n, allocator)
	rwork := make([]f32, n, allocator)

	lapack.cgbsvx_(
		fact_c,
		trans_c,
		&n,
		&kl,
		&ku,
		&nrhs,
		raw_data(AB.data),
		&ldab,
		raw_data(AFB_out.data),
		&ldafb,
		raw_data(ipiv_out),
		equed_c,
		raw_data(R_out),
		raw_data(C_out),
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

	// Convert equilibration mode back
	switch equed_c[0] {
	case 'N':
		equed_out = .None
	case 'R':
		equed_out = .Row
	case 'C':
		equed_out = .Column
	case 'B':
		equed_out = .Both
	}

	return X, rcond, ferr, berr, ipiv_out, AFB_out, R_out, C_out, equed_out, info
}

m_banded_solve_expert_c128 :: proc(
	AB: ^Matrix(complex128),
	B: ^Matrix(complex128),
	fact: FactorizationMode = .Equilibrate,
	trans: bool = false,
	AFB: ^Matrix(complex128) = nil,
	ipiv: []i32 = nil,
	equed: ^EquilibrationMode = nil,
	R: []f64 = nil,
	C: []f64 = nil,
	allocator := context.allocator,
) -> (
	X: Matrix(complex128),
	rcond: f64,
	ferr: []f64,
	berr: []f64,
	ipiv_out: []i32,
	AFB_out: Matrix(complex128),
	R_out: []f64,
	C_out: []f64,
	equed_out: EquilibrationMode,
	info: Info,
) {
	assert(AB.format == .Banded, "AB must be in banded format")

	n := Blas_Int(AB.cols)
	kl := Blas_Int(AB.storage.banded.kl)
	ku := Blas_Int(AB.storage.banded.ku)
	nrhs := Blas_Int(B.cols)
	ldab := Blas_Int(AB.ld)
	ldb := Blas_Int(B.ld)

	// Allocate solution matrix
	X = make_matrix(complex128, int(n), int(nrhs), allocator)
	ldx := Blas_Int(X.ld)

	// Allocate or use provided factored matrix
	if AFB == nil {
		AFB_out = make_matrix(complex128, int(2 * kl + ku + 1), int(n), allocator)
		AFB_out.format = .Banded
		AFB_out.storage.banded = {
			kl = int(kl),
			ku = int(ku),
		}
	} else {
		AFB_out = AFB^
	}
	ldafb := Blas_Int(AFB_out.ld)

	// Allocate or use provided pivot indices
	if ipiv == nil {
		ipiv_out = make([]i32, n, allocator)
	} else {
		ipiv_out = ipiv
	}

	// Allocate or use provided scale factors
	if R == nil {
		R_out = make([]f64, n, allocator)
	} else {
		R_out = R
	}
	if C == nil {
		C_out = make([]f64, n, allocator)
	} else {
		C_out = C
	}

	// Convert options to strings
	fact_c: cstring
	switch fact {
	case .Equilibrate:
		fact_c = cstring("E")
	case .Factored:
		fact_c = cstring("F")
	case .NotFactored:
		fact_c = cstring("N")
	}

	trans_c := trans ? cstring("T") : cstring("N")

	// Handle equilibration mode
	equed_c: cstring
	if equed != nil {
		switch equed^ {
		case .None:
			equed_c = cstring("N")
		case .Row:
			equed_c = cstring("R")
		case .Column:
			equed_c = cstring("C")
		case .Both:
			equed_c = cstring("B")
		}
	} else {
		equed_c = cstring("N")
	}

	// Allocate error bounds
	ferr = make([]f64, nrhs, allocator)
	berr = make([]f64, nrhs, allocator)

	// Allocate workspace
	work := make([]complex128, 2 * n, allocator)
	rwork := make([]f64, n, allocator)

	lapack.zgbsvx_(
		fact_c,
		trans_c,
		&n,
		&kl,
		&ku,
		&nrhs,
		raw_data(AB.data),
		&ldab,
		raw_data(AFB_out.data),
		&ldafb,
		raw_data(ipiv_out),
		equed_c,
		raw_data(R_out),
		raw_data(C_out),
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

	// Convert equilibration mode back
	switch equed_c[0] {
	case 'N':
		equed_out = .None
	case 'R':
		equed_out = .Row
	case 'C':
		equed_out = .Column
	case 'B':
		equed_out = .Both
	}

	return X, rcond, ferr, berr, ipiv_out, AFB_out, R_out, C_out, equed_out, info
}

// Extended expert driver with extra precision and advanced error bounds (LAPACK 3.x)
m_banded_solve_expert_extended :: proc {
	m_banded_solve_expert_extended_real,
	m_banded_solve_expert_extended_c64,
	m_banded_solve_expert_extended_c128,
}

m_banded_solve_expert_extended_real :: proc(
	AB: ^Matrix($T),
	B: ^Matrix(T),
	fact: FactorizationMode = .Equilibrate,
	trans: bool = false,
	AFB: ^Matrix(T) = nil,
	ipiv: []i32 = nil,
	equed: ^EquilibrationMode = nil,
	R: []T = nil,
	C: []T = nil,
	allocator := context.allocator,
) -> (
	X: Matrix(T),
	rcond: T,
	rpvgrw: T,
	berr: []T,
	n_err_bnds: Blas_Int,
	err_bnds_norm: []T,
	err_bnds_comp: []T,
	nparams: Blas_Int,// Reciprocal pivot growth factor
	params: []T,// Backward error bounds
	ipiv_out: []i32,// Number of error bounds
	AFB_out: Matrix(T),// Error bounds for normwise error
	R_out: []T,// Error bounds for componentwise error
	C_out: []T,// Number of parameters
	equed_out: EquilibrationMode,// Algorithm parameters
	info: Info,
) where T == f32 || T == f64 {
	assert(AB.format == .Banded, "AB must be in banded format")

	n := Blas_Int(AB.cols)
	kl := Blas_Int(AB.storage.banded.kl)
	ku := Blas_Int(AB.storage.banded.ku)
	nrhs := Blas_Int(B.cols)
	ldab := Blas_Int(AB.ld)
	ldb := Blas_Int(B.ld)

	// Allocate solution matrix
	X = make_matrix(T, int(n), int(nrhs), allocator)
	ldx := Blas_Int(X.ld)

	// Allocate or use provided factored matrix
	if AFB == nil {
		AFB_out = make_matrix(T, int(2 * kl + ku + 1), int(n), allocator)
		AFB_out.format = .Banded
		AFB_out.storage.banded = {
			kl = int(kl),
			ku = int(ku),
		}
	} else {
		AFB_out = AFB^
	}
	ldafb := Blas_Int(AFB_out.ld)

	// Allocate or use provided arrays
	if ipiv == nil {
		ipiv_out = make([]i32, n, allocator)
	} else {
		ipiv_out = ipiv
	}

	if R == nil {
		R_out = make([]T, n, allocator)
	} else {
		R_out = R
	}
	if C == nil {
		C_out = make([]T, n, allocator)
	} else {
		C_out = C
	}

	// Convert options
	fact_c: cstring
	switch fact {
	case .Equilibrate:
		fact_c = cstring("E")
	case .Factored:
		fact_c = cstring("F")
	case .NotFactored:
		fact_c = cstring("N")
	}

	trans_c := trans ? cstring("T") : cstring("N")

	equed_c: cstring
	if equed != nil {
		switch equed^ {
		case .None:
			equed_c = cstring("N")
		case .Row:
			equed_c = cstring("R")
		case .Column:
			equed_c = cstring("C")
		case .Both:
			equed_c = cstring("B")
		}
	} else {
		equed_c = cstring("N")
	}

	// Allocate error bounds
	n_err_bnds = 3
	berr = make([]T, nrhs, allocator)
	err_bnds_norm = make([]T, nrhs * n_err_bnds, allocator)
	err_bnds_comp = make([]T, nrhs * n_err_bnds, allocator)
	params = make([]T, 3, allocator)

	// Allocate workspace
	work := make([]T, 4 * n, allocator)
	iwork := make([]i32, n, allocator)

	when T == f32 {
		lapack.sgbsvxx_(
			fact_c,
			trans_c,
			&n,
			&kl,
			&ku,
			&nrhs,
			raw_data(AB.data),
			&ldab,
			raw_data(AFB_out.data),
			&ldafb,
			raw_data(ipiv_out),
			equed_c,
			raw_data(R_out),
			raw_data(C_out),
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
			raw_data(params),
			raw_data(work),
			raw_data(iwork),
			&info,
			1,
			1,
			1,
		)
	} else when T == f64 {
		lapack.dgbsvxx_(
			fact_c,
			trans_c,
			&n,
			&kl,
			&ku,
			&nrhs,
			raw_data(AB.data),
			&ldab,
			raw_data(AFB_out.data),
			&ldafb,
			raw_data(ipiv_out),
			equed_c,
			raw_data(R_out),
			raw_data(C_out),
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
			raw_data(params),
			raw_data(work),
			raw_data(iwork),
			&info,
			1,
			1,
			1,
		)
	}

	// Convert equilibration mode back
	switch equed_c[0] {
	case 'N':
		equed_out = .None
	case 'R':
		equed_out = .Row
	case 'C':
		equed_out = .Column
	case 'B':
		equed_out = .Both
	}

	return X,
		rcond,
		rpvgrw,
		berr,
		n_err_bnds,
		err_bnds_norm,
		err_bnds_comp,
		nparams,
		params,
		ipiv_out,
		AFB_out,
		R_out,
		C_out,
		equed_out,
		info
}

m_banded_solve_expert_extended_c64 :: proc(
	AB: ^Matrix(complex64),
	B: ^Matrix(complex64),
	fact: FactorizationMode = .Equilibrate,
	trans: bool = false,
	AFB: ^Matrix(complex64) = nil,
	ipiv: []i32 = nil,
	equed: ^EquilibrationMode = nil,
	R: []f32 = nil,
	C: []f32 = nil,
	allocator := context.allocator,
) -> (
	X: Matrix(complex64),
	rcond: f32,
	rpvgrw: f32,
	berr: []f32,
	n_err_bnds: Blas_Int,
	err_bnds_norm: []f32,
	err_bnds_comp: []f32,
	nparams: Blas_Int,
	params: []f32,
	ipiv_out: []i32,
	AFB_out: Matrix(complex64),
	R_out: []f32,
	C_out: []f32,
	equed_out: EquilibrationMode,
	info: Info,
) {
	assert(AB.format == .Banded, "AB must be in banded format")

	n := Blas_Int(AB.cols)
	kl := Blas_Int(AB.storage.banded.kl)
	ku := Blas_Int(AB.storage.banded.ku)
	nrhs := Blas_Int(B.cols)
	ldab := Blas_Int(AB.ld)
	ldb := Blas_Int(B.ld)

	// Allocate solution matrix
	X = make_matrix(complex64, int(n), int(nrhs), allocator)
	ldx := Blas_Int(X.ld)

	// Allocate or use provided factored matrix
	if AFB == nil {
		AFB_out = make_matrix(complex64, int(2 * kl + ku + 1), int(n), allocator)
		AFB_out.format = .Banded
		AFB_out.storage.banded = {
			kl = int(kl),
			ku = int(ku),
		}
	} else {
		AFB_out = AFB^
	}
	ldafb := Blas_Int(AFB_out.ld)

	// Allocate or use provided arrays
	if ipiv == nil {
		ipiv_out = make([]i32, n, allocator)
	} else {
		ipiv_out = ipiv
	}

	if R == nil {
		R_out = make([]f32, n, allocator)
	} else {
		R_out = R
	}
	if C == nil {
		C_out = make([]f32, n, allocator)
	} else {
		C_out = C
	}

	// Convert options
	fact_c: cstring
	switch fact {
	case .Equilibrate:
		fact_c = cstring("E")
	case .Factored:
		fact_c = cstring("F")
	case .NotFactored:
		fact_c = cstring("N")
	}

	trans_c := trans ? cstring("T") : cstring("N")

	equed_c: cstring
	if equed != nil {
		switch equed^ {
		case .None:
			equed_c = cstring("N")
		case .Row:
			equed_c = cstring("R")
		case .Column:
			equed_c = cstring("C")
		case .Both:
			equed_c = cstring("B")
		}
	} else {
		equed_c = cstring("N")
	}

	// Allocate error bounds
	n_err_bnds = 3
	berr = make([]f32, nrhs, allocator)
	err_bnds_norm = make([]f32, nrhs * n_err_bnds, allocator)
	err_bnds_comp = make([]f32, nrhs * n_err_bnds, allocator)
	params = make([]f32, 3, allocator)

	// Allocate workspace
	work := make([]complex64, 2 * n, allocator)
	rwork := make([]f32, 2 * n, allocator)

	lapack.cgbsvxx_(
		fact_c,
		trans_c,
		&n,
		&kl,
		&ku,
		&nrhs,
		raw_data(AB.data),
		&ldab,
		raw_data(AFB_out.data),
		&ldafb,
		raw_data(ipiv_out),
		equed_c,
		raw_data(R_out),
		raw_data(C_out),
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
		raw_data(params),
		raw_data(work),
		raw_data(rwork),
		&info,
		1,
		1,
		1,
	)

	// Convert equilibration mode back
	switch equed_c[0] {
	case 'N':
		equed_out = .None
	case 'R':
		equed_out = .Row
	case 'C':
		equed_out = .Column
	case 'B':
		equed_out = .Both
	}

	return X,
		rcond,
		rpvgrw,
		berr,
		n_err_bnds,
		err_bnds_norm,
		err_bnds_comp,
		nparams,
		params,
		ipiv_out,
		AFB_out,
		R_out,
		C_out,
		equed_out,
		info
}

m_banded_solve_expert_extended_c128 :: proc(
	AB: ^Matrix(complex128),
	B: ^Matrix(complex128),
	fact: FactorizationMode = .Equilibrate,
	trans: bool = false,
	AFB: ^Matrix(complex128) = nil,
	ipiv: []i32 = nil,
	equed: ^EquilibrationMode = nil,
	R: []f64 = nil,
	C: []f64 = nil,
	allocator := context.allocator,
) -> (
	X: Matrix(complex128),
	rcond: f64,
	rpvgrw: f64,
	berr: []f64,
	n_err_bnds: Blas_Int,
	err_bnds_norm: []f64,
	err_bnds_comp: []f64,
	nparams: Blas_Int,
	params: []f64,
	ipiv_out: []i32,
	AFB_out: Matrix(complex128),
	R_out: []f64,
	C_out: []f64,
	equed_out: EquilibrationMode,
	info: Info,
) {
	assert(AB.format == .Banded, "AB must be in banded format")

	n := Blas_Int(AB.cols)
	kl := Blas_Int(AB.storage.banded.kl)
	ku := Blas_Int(AB.storage.banded.ku)
	nrhs := Blas_Int(B.cols)
	ldab := Blas_Int(AB.ld)
	ldb := Blas_Int(B.ld)

	// Allocate solution matrix
	X = make_matrix(complex128, int(n), int(nrhs), allocator)
	ldx := Blas_Int(X.ld)

	// Allocate or use provided factored matrix
	if AFB == nil {
		AFB_out = make_matrix(complex128, int(2 * kl + ku + 1), int(n), allocator)
		AFB_out.format = .Banded
		AFB_out.storage.banded = {
			kl = int(kl),
			ku = int(ku),
		}
	} else {
		AFB_out = AFB^
	}
	ldafb := Blas_Int(AFB_out.ld)

	// Allocate or use provided arrays
	if ipiv == nil {
		ipiv_out = make([]i32, n, allocator)
	} else {
		ipiv_out = ipiv
	}

	if R == nil {
		R_out = make([]f64, n, allocator)
	} else {
		R_out = R
	}
	if C == nil {
		C_out = make([]f64, n, allocator)
	} else {
		C_out = C
	}

	// Convert options
	fact_c: cstring
	switch fact {
	case .Equilibrate:
		fact_c = cstring("E")
	case .Factored:
		fact_c = cstring("F")
	case .NotFactored:
		fact_c = cstring("N")
	}

	trans_c := trans ? cstring("T") : cstring("N")

	equed_c: cstring
	if equed != nil {
		switch equed^ {
		case .None:
			equed_c = cstring("N")
		case .Row:
			equed_c = cstring("R")
		case .Column:
			equed_c = cstring("C")
		case .Both:
			equed_c = cstring("B")
		}
	} else {
		equed_c = cstring("N")
	}

	// Allocate error bounds
	n_err_bnds = 3
	berr = make([]f64, nrhs, allocator)
	err_bnds_norm = make([]f64, nrhs * n_err_bnds, allocator)
	err_bnds_comp = make([]f64, nrhs * n_err_bnds, allocator)
	params = make([]f64, 3, allocator)

	// Allocate workspace
	work := make([]complex128, 2 * n, allocator)
	rwork := make([]f64, 2 * n, allocator)

	lapack.zgbsvxx_(
		fact_c,
		trans_c,
		&n,
		&kl,
		&ku,
		&nrhs,
		raw_data(AB.data),
		&ldab,
		raw_data(AFB_out.data),
		&ldafb,
		raw_data(ipiv_out),
		equed_c,
		raw_data(R_out),
		raw_data(C_out),
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
		raw_data(params),
		raw_data(work),
		raw_data(rwork),
		&info,
		1,
		1,
		1,
	)

	// Convert equilibration mode back
	switch equed_c[0] {
	case 'N':
		equed_out = .None
	case 'R':
		equed_out = .Row
	case 'C':
		equed_out = .Column
	case 'B':
		equed_out = .Both
	}

	return X,
		rcond,
		rpvgrw,
		berr,
		n_err_bnds,
		err_bnds_norm,
		err_bnds_comp,
		nparams,
		params,
		ipiv_out,
		AFB_out,
		R_out,
		C_out,
		equed_out,
		info
}

// ===================================================================================
// ITERATIVE REFINEMENT
// ===================================================================================

// Basic iterative refinement for banded matrix solutions
m_banded_refine :: proc {
	m_banded_refine_real,
	m_banded_refine_c64,
	m_banded_refine_c128,
}

m_banded_refine_real :: proc(
	AB: ^Matrix($T), // Original banded matrix
	AFB: ^Matrix(T), // Factored matrix from m_banded_factor
	ipiv: []i32, // Pivot indices from factorization
	B: ^Matrix(T), // Right-hand side
	X: ^Matrix(T), // Solution (on input: initial solution, on output: refined solution)
	trans: bool = false, // Use transpose
	allocator := context.allocator,
) -> (
	ferr: []T,
	berr: []T,
	info: Info, // Forward error bounds for each solution// Backward error bounds for each solution
) where T == f32 || T == f64 {
	assert(AB.format == .Banded, "AB must be in banded format")
	assert(AFB.format == .Banded, "AFB must be in banded format")

	n := Blas_Int(AB.cols)
	kl := Blas_Int(AB.storage.banded.kl)
	ku := Blas_Int(AB.storage.banded.ku)
	nrhs := Blas_Int(B.cols)
	ldab := Blas_Int(AB.ld)
	ldafb := Blas_Int(AFB.ld)
	ldb := Blas_Int(B.ld)
	ldx := Blas_Int(X.ld)

	trans_c := trans ? cstring("T") : cstring("N")

	// Allocate error bounds
	ferr = make([]T, nrhs, allocator)
	berr = make([]T, nrhs, allocator)

	// Allocate workspace
	work := make([]T, 3 * n, allocator)
	iwork := make([]i32, n, allocator)

	when T == f32 {
		lapack.sgbrfs_(
			trans_c,
			&n,
			&kl,
			&ku,
			&nrhs,
			raw_data(AB.data),
			&ldab,
			raw_data(AFB.data),
			&ldafb,
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
	} else when T == f64 {
		lapack.dgbrfs_(
			trans_c,
			&n,
			&kl,
			&ku,
			&nrhs,
			raw_data(AB.data),
			&ldab,
			raw_data(AFB.data),
			&ldafb,
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

m_banded_refine_c64 :: proc(
	AB: ^Matrix(complex64),
	AFB: ^Matrix(complex64),
	ipiv: []i32,
	B: ^Matrix(complex64),
	X: ^Matrix(complex64),
	trans: bool = false,
	allocator := context.allocator,
) -> (
	ferr: []f32,
	berr: []f32,
	info: Info,
) {
	assert(AB.format == .Banded, "AB must be in banded format")
	assert(AFB.format == .Banded, "AFB must be in banded format")

	n := Blas_Int(AB.cols)
	kl := Blas_Int(AB.storage.banded.kl)
	ku := Blas_Int(AB.storage.banded.ku)
	nrhs := Blas_Int(B.cols)
	ldab := Blas_Int(AB.ld)
	ldafb := Blas_Int(AFB.ld)
	ldb := Blas_Int(B.ld)
	ldx := Blas_Int(X.ld)

	trans_c := trans ? cstring("T") : cstring("N")

	// Allocate error bounds
	ferr = make([]f32, nrhs, allocator)
	berr = make([]f32, nrhs, allocator)

	// Allocate workspace
	work := make([]complex64, 2 * n, allocator)
	rwork := make([]f32, n, allocator)

	lapack.cgbrfs_(
		trans_c,
		&n,
		&kl,
		&ku,
		&nrhs,
		raw_data(AB.data),
		&ldab,
		raw_data(AFB.data),
		&ldafb,
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

m_banded_refine_c128 :: proc(
	AB: ^Matrix(complex128),
	AFB: ^Matrix(complex128),
	ipiv: []i32,
	B: ^Matrix(complex128),
	X: ^Matrix(complex128),
	trans: bool = false,
	allocator := context.allocator,
) -> (
	ferr: []f64,
	berr: []f64,
	info: Info,
) {
	assert(AB.format == .Banded, "AB must be in banded format")
	assert(AFB.format == .Banded, "AFB must be in banded format")

	n := Blas_Int(AB.cols)
	kl := Blas_Int(AB.storage.banded.kl)
	ku := Blas_Int(AB.storage.banded.ku)
	nrhs := Blas_Int(B.cols)
	ldab := Blas_Int(AB.ld)
	ldafb := Blas_Int(AFB.ld)
	ldb := Blas_Int(B.ld)
	ldx := Blas_Int(X.ld)

	trans_c := trans ? cstring("T") : cstring("N")

	// Allocate error bounds
	ferr = make([]f64, nrhs, allocator)
	berr = make([]f64, nrhs, allocator)

	// Allocate workspace
	work := make([]complex128, 2 * n, allocator)
	rwork := make([]f64, n, allocator)

	lapack.zgbrfs_(
		trans_c,
		&n,
		&kl,
		&ku,
		&nrhs,
		raw_data(AB.data),
		&ldab,
		raw_data(AFB.data),
		&ldafb,
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

// Extended iterative refinement with extra precision (LAPACK 3.x)
m_banded_refine_extended :: proc {
	m_banded_refine_extended_real,
	m_banded_refine_extended_c64,
	m_banded_refine_extended_c128,
}

m_banded_refine_extended_real :: proc(
	AB: ^Matrix($T),
	AFB: ^Matrix(T),
	ipiv: []i32,
	R: []T, // Row scale factors from equilibration
	C: []T, // Column scale factors from equilibration
	B: ^Matrix(T),
	X: ^Matrix(T),
	trans: bool = false,
	equed: EquilibrationMode = .None,
	allocator := context.allocator,
) -> (
	rcond: T,
	ferr: []T,
	berr: []T,
	err_bnds_norm: []T,
	err_bnds_comp: []T,
	nparams: Blas_Int,// Reciprocal of condition number
	params: []T,// Forward error bounds
	info: Info, // Backward error bounds// Error bounds for normwise error// Error bounds for componentwise error// Number of parameters// Algorithm parameters
) where T == f32 || T == f64 {
	assert(AB.format == .Banded, "AB must be in banded format")
	assert(AFB.format == .Banded, "AFB must be in banded format")

	n := Blas_Int(AB.cols)
	kl := Blas_Int(AB.storage.banded.kl)
	ku := Blas_Int(AB.storage.banded.ku)
	nrhs := Blas_Int(B.cols)
	ldab := Blas_Int(AB.ld)
	ldafb := Blas_Int(AFB.ld)
	ldb := Blas_Int(B.ld)
	ldx := Blas_Int(X.ld)
	n_err_bnds := Blas_Int(3)

	trans_c := trans ? cstring("T") : cstring("N")

	// Convert equilibration mode to string
	equed_c: cstring
	switch equed {
	case .None:
		equed_c = cstring("N")
	case .Row:
		equed_c = cstring("R")
	case .Column:
		equed_c = cstring("C")
	case .Both:
		equed_c = cstring("B")
	}

	// Allocate outputs
	ferr = make([]T, nrhs, allocator)
	berr = make([]T, nrhs, allocator)
	err_bnds_norm = make([]T, nrhs * n_err_bnds, allocator)
	err_bnds_comp = make([]T, nrhs * n_err_bnds, allocator)
	params = make([]T, 3, allocator)

	// Allocate workspace
	work := make([]T, 4 * n, allocator)
	iwork := make([]i32, n, allocator)

	when T == f32 {
		lapack.sgbrfsx_(
			trans_c,
			equed_c,
			&n,
			&kl,
			&ku,
			&nrhs,
			raw_data(AB.data),
			&ldab,
			raw_data(AFB.data),
			&ldafb,
			raw_data(ipiv),
			raw_data(R),
			raw_data(C),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			&rcond,
			raw_data(ferr),
			raw_data(berr),
			&n_err_bnds,
			raw_data(err_bnds_norm),
			raw_data(err_bnds_comp),
			&nparams,
			raw_data(params),
			raw_data(work),
			raw_data(iwork),
			&info,
			1,
			1,
		)
	} else when T == f64 {
		lapack.dgbrfsx_(
			trans_c,
			equed_c,
			&n,
			&kl,
			&ku,
			&nrhs,
			raw_data(AB.data),
			&ldab,
			raw_data(AFB.data),
			&ldafb,
			raw_data(ipiv),
			raw_data(R),
			raw_data(C),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			&rcond,
			raw_data(ferr),
			raw_data(berr),
			&n_err_bnds,
			raw_data(err_bnds_norm),
			raw_data(err_bnds_comp),
			&nparams,
			raw_data(params),
			raw_data(work),
			raw_data(iwork),
			&info,
			1,
			1,
		)
	}

	return rcond, ferr, berr, err_bnds_norm, err_bnds_comp, nparams, params, info
}

m_banded_refine_extended_c64 :: proc(
	AB: ^Matrix(complex64),
	AFB: ^Matrix(complex64),
	ipiv: []i32,
	R: []f32,
	C: []f32,
	B: ^Matrix(complex64),
	X: ^Matrix(complex64),
	trans: bool = false,
	equed: EquilibrationMode = .None,
	allocator := context.allocator,
) -> (
	rcond: f32,
	ferr: []f32,
	berr: []f32,
	err_bnds_norm: []f32,
	err_bnds_comp: []f32,
	nparams: Blas_Int,
	params: []f32,
	info: Info,
) {
	assert(AB.format == .Banded, "AB must be in banded format")
	assert(AFB.format == .Banded, "AFB must be in banded format")

	n := Blas_Int(AB.cols)
	kl := Blas_Int(AB.storage.banded.kl)
	ku := Blas_Int(AB.storage.banded.ku)
	nrhs := Blas_Int(B.cols)
	ldab := Blas_Int(AB.ld)
	ldafb := Blas_Int(AFB.ld)
	ldb := Blas_Int(B.ld)
	ldx := Blas_Int(X.ld)
	n_err_bnds := Blas_Int(3)

	trans_c := trans ? cstring("T") : cstring("N")

	// Convert equilibration mode to string
	equed_c: cstring
	switch equed {
	case .None:
		equed_c = cstring("N")
	case .Row:
		equed_c = cstring("R")
	case .Column:
		equed_c = cstring("C")
	case .Both:
		equed_c = cstring("B")
	}

	// Allocate outputs
	ferr = make([]f32, nrhs, allocator)
	berr = make([]f32, nrhs, allocator)
	err_bnds_norm = make([]f32, nrhs * n_err_bnds, allocator)
	err_bnds_comp = make([]f32, nrhs * n_err_bnds, allocator)
	params = make([]f32, 3, allocator)

	// Allocate workspace
	work := make([]complex64, 2 * n, allocator)
	rwork := make([]f32, 2 * n, allocator)

	lapack.cgbrfsx_(
		trans_c,
		equed_c,
		&n,
		&kl,
		&ku,
		&nrhs,
		raw_data(AB.data),
		&ldab,
		raw_data(AFB.data),
		&ldafb,
		raw_data(ipiv),
		raw_data(R),
		raw_data(C),
		raw_data(B.data),
		&ldb,
		raw_data(X.data),
		&ldx,
		&rcond,
		raw_data(ferr),
		raw_data(berr),
		&n_err_bnds,
		raw_data(err_bnds_norm),
		raw_data(err_bnds_comp),
		&nparams,
		raw_data(params),
		raw_data(work),
		raw_data(rwork),
		&info,
		1,
		1,
	)

	return rcond, ferr, berr, err_bnds_norm, err_bnds_comp, nparams, params, info
}

m_banded_refine_extended_c128 :: proc(
	AB: ^Matrix(complex128),
	AFB: ^Matrix(complex128),
	ipiv: []i32,
	R: []f64,
	C: []f64,
	B: ^Matrix(complex128),
	X: ^Matrix(complex128),
	trans: bool = false,
	equed: EquilibrationMode = .None,
	allocator := context.allocator,
) -> (
	rcond: f64,
	ferr: []f64,
	berr: []f64,
	err_bnds_norm: []f64,
	err_bnds_comp: []f64,
	nparams: Blas_Int,
	params: []f64,
	info: Info,
) {
	assert(AB.format == .Banded, "AB must be in banded format")
	assert(AFB.format == .Banded, "AFB must be in banded format")

	n := Blas_Int(AB.cols)
	kl := Blas_Int(AB.storage.banded.kl)
	ku := Blas_Int(AB.storage.banded.ku)
	nrhs := Blas_Int(B.cols)
	ldab := Blas_Int(AB.ld)
	ldafb := Blas_Int(AFB.ld)
	ldb := Blas_Int(B.ld)
	ldx := Blas_Int(X.ld)
	n_err_bnds := Blas_Int(3)

	trans_c := trans ? cstring("T") : cstring("N")

	// Convert equilibration mode to string
	equed_c: cstring
	switch equed {
	case .None:
		equed_c = cstring("N")
	case .Row:
		equed_c = cstring("R")
	case .Column:
		equed_c = cstring("C")
	case .Both:
		equed_c = cstring("B")
	}

	// Allocate outputs
	ferr = make([]f64, nrhs, allocator)
	berr = make([]f64, nrhs, allocator)
	err_bnds_norm = make([]f64, nrhs * n_err_bnds, allocator)
	err_bnds_comp = make([]f64, nrhs * n_err_bnds, allocator)
	params = make([]f64, 3, allocator)

	// Allocate workspace
	work := make([]complex128, 2 * n, allocator)
	rwork := make([]f64, 2 * n, allocator)

	lapack.zgbrfsx_(
		trans_c,
		equed_c,
		&n,
		&kl,
		&ku,
		&nrhs,
		raw_data(AB.data),
		&ldab,
		raw_data(AFB.data),
		&ldafb,
		raw_data(ipiv),
		raw_data(R),
		raw_data(C),
		raw_data(B.data),
		&ldb,
		raw_data(X.data),
		&ldx,
		&rcond,
		raw_data(ferr),
		raw_data(berr),
		&n_err_bnds,
		raw_data(err_bnds_norm),
		raw_data(err_bnds_comp),
		&nparams,
		raw_data(params),
		raw_data(work),
		raw_data(rwork),
		&info,
		1,
		1,
	)

	return rcond, ferr, berr, err_bnds_norm, err_bnds_comp, nparams, params, info
}

// ===================================================================================
// HERMITIAN BANDED EIGENVALUE ROUTINES
// ===================================================================================

// Hermitian banded eigenvalue solver (basic)
// Computes all eigenvalues and optionally eigenvectors of Hermitian banded matrix
m_eigen_hermitian_banded :: proc {
	m_eigen_hermitian_banded_c64,
	m_eigen_hermitian_banded_c128,
	m_eigen_hermitian_banded_c64_2stage,
	m_eigen_hermitian_banded_c128_2stage,
}

m_eigen_hermitian_banded_c64 :: proc(
	A: ^Matrix(complex64),
	compute_vectors: bool = false,
	allocator := context.allocator,
) -> (
	w: []f32,
	Z: Matrix(complex64),
	info: lapack.blasint, // Eigenvalues// Eigenvectors (if computed)
) {
	assert(A.format == .Banded, "Matrix must be banded format")

	n := A.rows
	kd := A.storage.banded.ku // For Hermitian banded, kl = ku = kd
	ldab := A.storage.banded.ldab

	jobz_c := cstring("V") if compute_vectors else cstring("N")
	uplo_c := cstring("U") // Assume upper storage

	// Allocate eigenvalues
	w = make([]f32, n, allocator)

	// Allocate eigenvectors if needed
	ldz := n if compute_vectors else 1
	Z =
		make_matrix(complex64, n, n, .General, allocator) if compute_vectors else Matrix(complex64){}

	// Workspace query
	work_query: complex64
	rwork_query: f32
	work_size: lapack.blasint = -1

	lapack.chbev_(
		jobz_c,
		uplo_c,
		&n,
		&kd,
		raw_data(A.data),
		&ldab,
		raw_data(w),
		matrix_data_ptr(&Z) if compute_vectors else &work_query,
		&ldz,
		&work_query,
		&rwork_query,
		&info,
		1,
		1,
	)

	if info != 0 do return

	// Allocate workspace
	lwork := auto_cast real(work_query)
	work := make([]complex64, lwork, allocator)
	rwork := make([]f32, max(1, 3 * n - 2), allocator)

	lapack.chbev_(
		jobz_c,
		uplo_c,
		&n,
		&kd,
		raw_data(A.data),
		&ldab,
		raw_data(w),
		matrix_data_ptr(&Z) if compute_vectors else nil,
		&ldz,
		raw_data(work),
		raw_data(rwork),
		&info,
		1,
		1,
	)

	return w, Z, info
}

m_eigen_hermitian_banded_c128 :: proc(
	A: ^Matrix(complex128),
	compute_vectors: bool = false,
	allocator := context.allocator,
) -> (
	w: []f64,
	Z: Matrix(complex128),
	info: lapack.blasint, // Eigenvalues// Eigenvectors (if computed)
) {
	assert(A.format == .Banded, "Matrix must be banded format")

	n := A.rows
	kd := A.storage.banded.ku // For Hermitian banded, kl = ku = kd
	ldab := A.storage.banded.ldab

	jobz_c := cstring("V") if compute_vectors else cstring("N")
	uplo_c := cstring("U") // Assume upper storage

	// Allocate eigenvalues
	w = make([]f64, n, allocator)

	// Allocate eigenvectors if needed
	ldz := n if compute_vectors else 1
	Z =
		make_matrix(complex128, n, n, .General, allocator) if compute_vectors else Matrix(complex128){}

	// Workspace query
	work_query: complex128
	rwork_query: f64
	work_size: lapack.blasint = -1

	lapack.zhbev_(
		jobz_c,
		uplo_c,
		&n,
		&kd,
		raw_data(A.data),
		&ldab,
		raw_data(w),
		matrix_data_ptr(&Z) if compute_vectors else &work_query,
		&ldz,
		&work_query,
		&rwork_query,
		&info,
		1,
		1,
	)

	if info != 0 do return

	// Allocate workspace
	lwork := auto_cast real(work_query)
	work := make([]complex128, lwork, allocator)
	rwork := make([]f64, max(1, 3 * n - 2), allocator)

	lapack.zhbev_(
		jobz_c,
		uplo_c,
		&n,
		&kd,
		raw_data(A.data),
		&ldab,
		raw_data(w),
		matrix_data_ptr(&Z) if compute_vectors else nil,
		&ldz,
		raw_data(work),
		raw_data(rwork),
		&info,
		1,
		1,
	)

	return w, Z, info
}

m_eigen_hermitian_banded_c64_2stage :: proc(
	A: ^Matrix(complex64),
	compute_vectors: bool = false,
	allocator := context.allocator,
) -> (
	w: []f32,
	Z: Matrix(complex64),
	info: lapack.blasint, // Eigenvalues// Eigenvectors (if computed)
) {
	assert(A.format == .Banded, "Matrix must be banded format")

	n := A.rows
	kd := A.storage.banded.ku
	ldab := A.storage.banded.ldab

	jobz_c := cstring("V") if compute_vectors else cstring("N")
	uplo_c := cstring("U")

	w = make([]f32, n, allocator)
	ldz := n if compute_vectors else 1
	Z =
		make_matrix(complex64, n, n, .General, allocator) if compute_vectors else Matrix(complex64){}

	// Workspace query
	work_query: complex64
	rwork_query: f32
	work_size: lapack.blasint = -1

	lapack.chbev_2stage_(
		jobz_c,
		uplo_c,
		&n,
		&kd,
		raw_data(A.data),
		&ldab,
		raw_data(w),
		matrix_data_ptr(&Z) if compute_vectors else &work_query,
		&ldz,
		&work_query,
		&work_size,
		&rwork_query,
		&info,
		1,
		1,
	)

	if info != 0 do return

	lwork := auto_cast real(work_query)
	work := make([]complex64, lwork, allocator)
	rwork := make([]f32, max(1, 3 * n - 2), allocator)

	lapack.chbev_2stage_(
		jobz_c,
		uplo_c,
		&n,
		&kd,
		raw_data(A.data),
		&ldab,
		raw_data(w),
		matrix_data_ptr(&Z) if compute_vectors else nil,
		&ldz,
		raw_data(work),
		&lwork,
		raw_data(rwork),
		&info,
		1,
		1,
	)

	return w, Z, info
}

m_eigen_hermitian_banded_c128_2stage :: proc(
	A: ^Matrix(complex128),
	compute_vectors: bool = false,
	allocator := context.allocator,
) -> (
	w: []f64,
	Z: Matrix(complex128),
	info: lapack.blasint, // Eigenvalues// Eigenvectors (if computed)
) {
	assert(A.format == .Banded, "Matrix must be banded format")

	n := A.rows
	kd := A.storage.banded.ku
	ldab := A.storage.banded.ldab

	jobz_c := cstring("V") if compute_vectors else cstring("N")
	uplo_c := cstring("U")

	w = make([]f64, n, allocator)
	ldz := n if compute_vectors else 1
	Z =
		make_matrix(complex128, n, n, .General, allocator) if compute_vectors else Matrix(complex128){}

	// Workspace query
	work_query: complex128
	rwork_query: f64
	work_size: lapack.blasint = -1

	lapack.zhbev_2stage_(
		jobz_c,
		uplo_c,
		&n,
		&kd,
		raw_data(A.data),
		&ldab,
		raw_data(w),
		matrix_data_ptr(&Z) if compute_vectors else &work_query,
		&ldz,
		&work_query,
		&work_size,
		&rwork_query,
		&info,
		1,
		1,
	)

	if info != 0 do return

	lwork := auto_cast real(work_query)
	work := make([]complex128, lwork, allocator)
	rwork := make([]f64, max(1, 3 * n - 2), allocator)

	lapack.zhbev_2stage_(
		jobz_c,
		uplo_c,
		&n,
		&kd,
		raw_data(A.data),
		&ldab,
		raw_data(w),
		matrix_data_ptr(&Z) if compute_vectors else nil,
		&ldz,
		raw_data(work),
		&lwork,
		raw_data(rwork),
		&info,
		1,
		1,
	)

	return w, Z, info
}

// Hermitian banded eigenvalue solver (divide-and-conquer)
// Faster for large matrices when eigenvectors are needed
m_eigen_hermitian_banded_dc :: proc {
	m_eigen_hermitian_banded_dc_c64,
	m_eigen_hermitian_banded_dc_c128,
	m_eigen_hermitian_banded_dc_c64_2stage,
	m_eigen_hermitian_banded_dc_c128_2stage,
}

m_eigen_hermitian_banded_dc_c64 :: proc(
	A: ^Matrix(complex64),
	compute_vectors: bool = false,
	allocator := context.allocator,
) -> (
	w: []f32,
	Z: Matrix(complex64),
	info: lapack.blasint, // Eigenvalues// Eigenvectors (if computed)
) {
	assert(A.format == .Banded, "Matrix must be banded format")

	n := A.rows
	kd := A.storage.banded.ku
	ldab := A.storage.banded.ldab

	jobz_c := cstring("V") if compute_vectors else cstring("N")
	uplo_c := cstring("U")

	w = make([]f32, n, allocator)
	ldz := n if compute_vectors else 1
	Z =
		make_matrix(complex64, n, n, .General, allocator) if compute_vectors else Matrix(complex64){}

	// Workspace query
	work_query: complex64
	rwork_query: f32
	iwork_query: lapack.blasint
	liwork: lapack.blasint = -1
	lwork: lapack.blasint = -1
	lrwork: lapack.blasint = -1

	lapack.chbevd_(
		jobz_c,
		uplo_c,
		&n,
		&kd,
		raw_data(A.data),
		&ldab,
		raw_data(w),
		matrix_data_ptr(&Z) if compute_vectors else &work_query,
		&ldz,
		&work_query,
		&lwork,
		&rwork_query,
		&lrwork,
		&iwork_query,
		&liwork,
		&info,
		1,
		1,
	)

	if info != 0 do return

	lwork = auto_cast real(work_query)
	work := make([]complex64, lwork, allocator)

	lrwork_calc := 1 + 5 * n + 2 * n * n if compute_vectors else 1 + 2 * n
	lrwork = auto_cast rwork_query if auto_cast rwork_query > lrwork_calc else lrwork_calc
	rwork := make([]f32, lrwork, allocator)

	liwork_calc := 3 + 5 * n if compute_vectors else 1
	liwork = iwork_query if iwork_query > liwork_calc else liwork_calc
	iwork := make([]lapack.blasint, liwork, allocator)

	lapack.chbevd_(
		jobz_c,
		uplo_c,
		&n,
		&kd,
		raw_data(A.data),
		&ldab,
		raw_data(w),
		matrix_data_ptr(&Z) if compute_vectors else nil,
		&ldz,
		raw_data(work),
		&lwork,
		raw_data(rwork),
		&lrwork,
		raw_data(iwork),
		&liwork,
		&info,
		1,
		1,
	)

	return w, Z, info
}

m_eigen_hermitian_banded_dc_c128 :: proc(
	A: ^Matrix(complex128),
	compute_vectors: bool = false,
	allocator := context.allocator,
) -> (
	w: []f64,
	Z: Matrix(complex128),
	info: lapack.blasint, // Eigenvalues// Eigenvectors (if computed)
) {
	assert(A.format == .Banded, "Matrix must be banded format")

	n := A.rows
	kd := A.storage.banded.ku
	ldab := A.storage.banded.ldab

	jobz_c := cstring("V") if compute_vectors else cstring("N")
	uplo_c := cstring("U")

	w = make([]f64, n, allocator)
	ldz := n if compute_vectors else 1
	Z =
		make_matrix(complex128, n, n, .General, allocator) if compute_vectors else Matrix(complex128){}

	// Workspace query
	work_query: complex128
	rwork_query: f64
	iwork_query: lapack.blasint
	liwork: lapack.blasint = -1
	lwork: lapack.blasint = -1
	lrwork: lapack.blasint = -1

	lapack.zhbevd_(
		jobz_c,
		uplo_c,
		&n,
		&kd,
		raw_data(A.data),
		&ldab,
		raw_data(w),
		matrix_data_ptr(&Z) if compute_vectors else &work_query,
		&ldz,
		&work_query,
		&lwork,
		&rwork_query,
		&lrwork,
		&iwork_query,
		&liwork,
		&info,
		1,
		1,
	)

	if info != 0 do return

	lwork = auto_cast real(work_query)
	work := make([]complex128, lwork, allocator)

	lrwork_calc := 1 + 5 * n + 2 * n * n if compute_vectors else 1 + 2 * n
	lrwork = auto_cast rwork_query if auto_cast rwork_query > lrwork_calc else lrwork_calc
	rwork := make([]f64, lrwork, allocator)

	liwork_calc := 3 + 5 * n if compute_vectors else 1
	liwork = iwork_query if iwork_query > liwork_calc else liwork_calc
	iwork := make([]lapack.blasint, liwork, allocator)

	lapack.zhbevd_(
		jobz_c,
		uplo_c,
		&n,
		&kd,
		raw_data(A.data),
		&ldab,
		raw_data(w),
		matrix_data_ptr(&Z) if compute_vectors else nil,
		&ldz,
		raw_data(work),
		&lwork,
		raw_data(rwork),
		&lrwork,
		raw_data(iwork),
		&liwork,
		&info,
		1,
		1,
	)

	return w, Z, info
}

m_eigen_hermitian_banded_dc_c64_2stage :: proc(
	A: ^Matrix(complex64),
	compute_vectors: bool = false,
	allocator := context.allocator,
) -> (
	w: []f32,
	Z: Matrix(complex64),
	info: lapack.blasint, // Eigenvalues// Eigenvectors (if computed)
) {
	assert(A.format == .Banded, "Matrix must be banded format")

	n := A.rows
	kd := A.storage.banded.ku
	ldab := A.storage.banded.ldab

	jobz_c := cstring("V") if compute_vectors else cstring("N")
	uplo_c := cstring("U")

	w = make([]f32, n, allocator)
	ldz := n if compute_vectors else 1
	Z =
		make_matrix(complex64, n, n, .General, allocator) if compute_vectors else Matrix(complex64){}

	// Workspace query
	work_query: complex64
	rwork_query: f32
	iwork_query: lapack.blasint
	liwork: lapack.blasint = -1
	lwork: lapack.blasint = -1
	lrwork: lapack.blasint = -1

	lapack.chbevd_2stage_(
		jobz_c,
		uplo_c,
		&n,
		&kd,
		raw_data(A.data),
		&ldab,
		raw_data(w),
		matrix_data_ptr(&Z) if compute_vectors else &work_query,
		&ldz,
		&work_query,
		&lwork,
		&rwork_query,
		&lrwork,
		&iwork_query,
		&liwork,
		&info,
		1,
		1,
	)

	if info != 0 do return

	lwork = auto_cast real(work_query)
	work := make([]complex64, lwork, allocator)

	lrwork_calc := 1 + 5 * n + 2 * n * n if compute_vectors else 1 + 2 * n
	lrwork = auto_cast rwork_query if auto_cast rwork_query > lrwork_calc else lrwork_calc
	rwork := make([]f32, lrwork, allocator)

	liwork_calc := 3 + 5 * n if compute_vectors else 1
	liwork = iwork_query if iwork_query > liwork_calc else liwork_calc
	iwork := make([]lapack.blasint, liwork, allocator)

	lapack.chbevd_2stage_(
		jobz_c,
		uplo_c,
		&n,
		&kd,
		raw_data(A.data),
		&ldab,
		raw_data(w),
		matrix_data_ptr(&Z) if compute_vectors else nil,
		&ldz,
		raw_data(work),
		&lwork,
		raw_data(rwork),
		&lrwork,
		raw_data(iwork),
		&liwork,
		&info,
		1,
		1,
	)

	return w, Z, info
}

m_eigen_hermitian_banded_dc_c128_2stage :: proc(
	A: ^Matrix(complex128),
	compute_vectors: bool = false,
	allocator := context.allocator,
) -> (
	w: []f64,
	Z: Matrix(complex128),
	info: lapack.blasint, // Eigenvalues// Eigenvectors (if computed)
) {
	assert(A.format == .Banded, "Matrix must be banded format")

	n := A.rows
	kd := A.storage.banded.ku
	ldab := A.storage.banded.ldab

	jobz_c := cstring("V") if compute_vectors else cstring("N")
	uplo_c := cstring("U")

	w = make([]f64, n, allocator)
	ldz := n if compute_vectors else 1
	Z =
		make_matrix(complex128, n, n, .General, allocator) if compute_vectors else Matrix(complex128){}

	// Workspace query
	work_query: complex128
	rwork_query: f64
	iwork_query: lapack.blasint
	liwork: lapack.blasint = -1
	lwork: lapack.blasint = -1
	lrwork: lapack.blasint = -1

	lapack.zhbevd_2stage_(
		jobz_c,
		uplo_c,
		&n,
		&kd,
		raw_data(A.data),
		&ldab,
		raw_data(w),
		matrix_data_ptr(&Z) if compute_vectors else &work_query,
		&ldz,
		&work_query,
		&lwork,
		&rwork_query,
		&lrwork,
		&iwork_query,
		&liwork,
		&info,
		1,
		1,
	)

	if info != 0 do return

	lwork = auto_cast real(work_query)
	work := make([]complex128, lwork, allocator)

	lrwork_calc := 1 + 5 * n + 2 * n * n if compute_vectors else 1 + 2 * n
	lrwork = auto_cast rwork_query if auto_cast rwork_query > lrwork_calc else lrwork_calc
	rwork := make([]f64, lrwork, allocator)

	liwork_calc := 3 + 5 * n if compute_vectors else 1
	liwork = iwork_query if iwork_query > liwork_calc else liwork_calc
	iwork := make([]lapack.blasint, liwork, allocator)

	lapack.zhbevd_2stage_(
		jobz_c,
		uplo_c,
		&n,
		&kd,
		raw_data(A.data),
		&ldab,
		raw_data(w),
		matrix_data_ptr(&Z) if compute_vectors else nil,
		&ldz,
		raw_data(work),
		&lwork,
		raw_data(rwork),
		&lrwork,
		raw_data(iwork),
		&liwork,
		&info,
		1,
		1,
	)

	return w, Z, info
}

// Hermitian banded eigenvalue solver (expert with subset selection)
// Computes selected eigenvalues and optionally eigenvectors
m_eigen_hermitian_banded_expert :: proc {
	m_eigen_hermitian_banded_expert_c64,
	m_eigen_hermitian_banded_expert_c128,
	m_eigen_hermitian_banded_expert_c64_2stage,
	m_eigen_hermitian_banded_expert_c128_2stage,
}

EigenValueRange :: enum {
	All, // Compute all eigenvalues
	Indexed, // Compute eigenvalues with indices in [il, iu]
	Valued, // Compute eigenvalues in range [vl, vu]
}

m_eigen_hermitian_banded_expert_c64 :: proc(
	A: ^Matrix(complex64),
	range_type: EigenValueRange = .All,
	compute_vectors: bool = false,
	il: int = 1, // Lower index (1-based) for .Indexed
	iu: int = 0, // Upper index (1-based) for .Indexed (0 means use n)
	vl: f32 = 0, // Lower bound for .Valued
	vu: f32 = 0, // Upper bound for .Valued
	abstol: f32 = 0, // Absolute tolerance (0 for default)
	allocator := context.allocator,
) -> (
	m: int,
	w: []f32,
	Z: Matrix(complex64),// Number of eigenvalues found
	ifail: []lapack.blasint,// Eigenvalues
	info: lapack.blasint, // Eigenvectors (if computed)// Indices of failed eigenvectors
) {
	assert(A.format == .Banded, "Matrix must be banded format")

	n := A.rows
	kd := A.storage.banded.ku
	ldab := A.storage.banded.ldab

	jobz_c := cstring("V") if compute_vectors else cstring("N")
	uplo_c := cstring("U")

	// Range specification
	range_c: cstring
	il_use := il
	iu_use := iu if iu > 0 else n

	switch range_type {
	case .All:
		range_c = cstring("A")
	case .Indexed:
		range_c = cstring("I")
	case .Valued:
		range_c = cstring("V")
	}

	// Allocate outputs
	w = make([]f32, n, allocator)
	ldz := n if compute_vectors else 1
	Z =
		make_matrix(complex64, n, n, .General, allocator) if compute_vectors else Matrix(complex64){}
	ifail = make([]lapack.blasint, n, allocator)

	// Workspace
	work := make([]complex64, n, allocator)
	rwork := make([]f32, 7 * n, allocator)
	iwork := make([]lapack.blasint, 5 * n, allocator)

	lapack.chbevx_(
		jobz_c,
		range_c,
		uplo_c,
		&n,
		&kd,
		raw_data(A.data),
		&ldab,
		nil,
		&ldz, // Q not used for banded
		&vl,
		&vu,
		&il_use,
		&iu_use,
		&abstol,
		&m,
		raw_data(w),
		matrix_data_ptr(&Z) if compute_vectors else nil,
		&ldz,
		raw_data(work),
		raw_data(rwork),
		raw_data(iwork),
		raw_data(ifail),
		&info,
		1,
		1,
		1,
	)

	return m, w, Z, ifail, info
}

m_eigen_hermitian_banded_expert_c128 :: proc(
	A: ^Matrix(complex128),
	range_type: EigenValueRange = .All,
	compute_vectors: bool = false,
	il: int = 1, // Lower index (1-based) for .Indexed
	iu: int = 0, // Upper index (1-based) for .Indexed (0 means use n)
	vl: f64 = 0, // Lower bound for .Valued
	vu: f64 = 0, // Upper bound for .Valued
	abstol: f64 = 0, // Absolute tolerance (0 for default)
	allocator := context.allocator,
) -> (
	m: int,
	w: []f64,
	Z: Matrix(complex128),// Number of eigenvalues found
	ifail: []lapack.blasint,// Eigenvalues
	info: lapack.blasint, // Eigenvectors (if computed)// Indices of failed eigenvectors
) {
	assert(A.format == .Banded, "Matrix must be banded format")

	n := A.rows
	kd := A.storage.banded.ku
	ldab := A.storage.banded.ldab

	jobz_c := cstring("V") if compute_vectors else cstring("N")
	uplo_c := cstring("U")

	// Range specification
	range_c: cstring
	il_use := il
	iu_use := iu if iu > 0 else n

	switch range_type {
	case .All:
		range_c = cstring("A")
	case .Indexed:
		range_c = cstring("I")
	case .Valued:
		range_c = cstring("V")
	}

	// Allocate outputs
	w = make([]f64, n, allocator)
	ldz := n if compute_vectors else 1
	Z =
		make_matrix(complex128, n, n, .General, allocator) if compute_vectors else Matrix(complex128){}
	ifail = make([]lapack.blasint, n, allocator)

	// Workspace
	work := make([]complex128, n, allocator)
	rwork := make([]f64, 7 * n, allocator)
	iwork := make([]lapack.blasint, 5 * n, allocator)

	lapack.zhbevx_(
		jobz_c,
		range_c,
		uplo_c,
		&n,
		&kd,
		raw_data(A.data),
		&ldab,
		nil,
		&ldz, // Q not used for banded
		&vl,
		&vu,
		&il_use,
		&iu_use,
		&abstol,
		&m,
		raw_data(w),
		matrix_data_ptr(&Z) if compute_vectors else nil,
		&ldz,
		raw_data(work),
		raw_data(rwork),
		raw_data(iwork),
		raw_data(ifail),
		&info,
		1,
		1,
		1,
	)

	return m, w, Z, ifail, info
}

m_eigen_hermitian_banded_expert_c64_2stage :: proc(
	A: ^Matrix(complex64),
	range_type: EigenValueRange = .All,
	compute_vectors: bool = false,
	il: int = 1, // Lower index (1-based) for .Indexed
	iu: int = 0, // Upper index (1-based) for .Indexed (0 means use n)
	vl: f32 = 0, // Lower bound for .Valued
	vu: f32 = 0, // Upper bound for .Valued
	abstol: f32 = 0, // Absolute tolerance (0 for default)
	allocator := context.allocator,
) -> (
	m: int,
	w: []f32,
	Z: Matrix(complex64),// Number of eigenvalues found
	ifail: []lapack.blasint,// Eigenvalues
	info: lapack.blasint, // Eigenvectors (if computed)// Indices of failed eigenvectors
) {
	assert(A.format == .Banded, "Matrix must be banded format")

	n := A.rows
	kd := A.storage.banded.ku
	ldab := A.storage.banded.ldab

	jobz_c := cstring("V") if compute_vectors else cstring("N")
	uplo_c := cstring("U")

	// Range specification
	range_c: cstring
	il_use := il
	iu_use := iu if iu > 0 else n

	switch range_type {
	case .All:
		range_c = cstring("A")
	case .Indexed:
		range_c = cstring("I")
	case .Valued:
		range_c = cstring("V")
	}

	// Allocate outputs
	w = make([]f32, n, allocator)
	ldz := n if compute_vectors else 1
	Z =
		make_matrix(complex64, n, n, .General, allocator) if compute_vectors else Matrix(complex64){}
	ifail = make([]lapack.blasint, n, allocator)

	// Workspace query
	work_query: complex64
	lwork: lapack.blasint = -1
	rwork := make([]f64, 7 * n, allocator)
	iwork := make([]lapack.blasint, 5 * n, allocator)

	lapack.chbevx_2stage_(
		jobz_c,
		range_c,
		uplo_c,
		&n,
		&kd,
		raw_data(A.data),
		&ldab,
		nil,
		&ldz, // Q not used for banded
		&vl,
		&vu,
		&il_use,
		&iu_use,
		&abstol,
		&m,
		raw_data(w),
		matrix_data_ptr(&Z) if compute_vectors else &work_query,
		&ldz,
		&work_query,
		&lwork,
		raw_data(rwork),
		raw_data(iwork),
		raw_data(ifail),
		&info,
		1,
		1,
		1,
	)

	if info != 0 do return

	lwork = auto_cast real(work_query)
	work := make([]complex64, lwork, allocator)

	lapack.chbevx_2stage_(
		jobz_c,
		range_c,
		uplo_c,
		&n,
		&kd,
		raw_data(A.data),
		&ldab,
		nil,
		&ldz,
		&vl,
		&vu,
		&il_use,
		&iu_use,
		&abstol,
		&m,
		raw_data(w),
		matrix_data_ptr(&Z) if compute_vectors else nil,
		&ldz,
		raw_data(work),
		&lwork,
		raw_data(rwork),
		raw_data(iwork),
		raw_data(ifail),
		&info,
		1,
		1,
		1,
	)

	return m, w, Z, ifail, info
}

m_eigen_hermitian_banded_expert_c128_2stage :: proc(
	A: ^Matrix(complex128),
	range_type: EigenValueRange = .All,
	compute_vectors: bool = false,
	il: int = 1, // Lower index (1-based) for .Indexed
	iu: int = 0, // Upper index (1-based) for .Indexed (0 means use n)
	vl: f64 = 0, // Lower bound for .Valued
	vu: f64 = 0, // Upper bound for .Valued
	abstol: f64 = 0, // Absolute tolerance (0 for default)
	allocator := context.allocator,
) -> (
	m: int,
	w: []f64,
	Z: Matrix(complex128),// Number of eigenvalues found
	ifail: []lapack.blasint,// Eigenvalues
	info: lapack.blasint, // Eigenvectors (if computed)// Indices of failed eigenvectors
) {
	assert(A.format == .Banded, "Matrix must be banded format")

	n := A.rows
	kd := A.storage.banded.ku
	ldab := A.storage.banded.ldab

	jobz_c := cstring("V") if compute_vectors else cstring("N")
	uplo_c := cstring("U")

	// Range specification
	range_c: cstring
	il_use := il
	iu_use := iu if iu > 0 else n

	switch range_type {
	case .All:
		range_c = cstring("A")
	case .Indexed:
		range_c = cstring("I")
	case .Valued:
		range_c = cstring("V")
	}

	// Allocate outputs
	w = make([]f64, n, allocator)
	ldz := n if compute_vectors else 1
	Z =
		make_matrix(complex128, n, n, .General, allocator) if compute_vectors else Matrix(complex128){}
	ifail = make([]lapack.blasint, n, allocator)

	// Workspace query
	work_query: complex128
	lwork: lapack.blasint = -1
	rwork := make([]f64, 7 * n, allocator)
	iwork := make([]lapack.blasint, 5 * n, allocator)

	lapack.zhbevx_2stage_(
		jobz_c,
		range_c,
		uplo_c,
		&n,
		&kd,
		raw_data(A.data),
		&ldab,
		nil,
		&ldz, // Q not used for banded
		&vl,
		&vu,
		&il_use,
		&iu_use,
		&abstol,
		&m,
		raw_data(w),
		matrix_data_ptr(&Z) if compute_vectors else &work_query,
		&ldz,
		&work_query,
		&lwork,
		raw_data(rwork),
		raw_data(iwork),
		raw_data(ifail),
		&info,
		1,
		1,
		1,
	)

	if info != 0 do return

	lwork = auto_cast real(work_query)
	work := make([]complex128, lwork, allocator)

	lapack.zhbevx_2stage_(
		jobz_c,
		range_c,
		uplo_c,
		&n,
		&kd,
		raw_data(A.data),
		&ldab,
		nil,
		&ldz,
		&vl,
		&vu,
		&il_use,
		&iu_use,
		&abstol,
		&m,
		raw_data(w),
		matrix_data_ptr(&Z) if compute_vectors else nil,
		&ldz,
		raw_data(work),
		&lwork,
		raw_data(rwork),
		raw_data(iwork),
		raw_data(ifail),
		&info,
		1,
		1,
		1,
	)

	return m, w, Z, ifail, info
}

// ===================================================================================
// HERMITIAN BANDED GENERALIZED EIGENVALUE ROUTINES
// ===================================================================================

// Reduce generalized Hermitian-definite eigenproblem to standard form
// A*x = lambda*B*x  -->  C*y = lambda*y where C = L^(-1)*A*L^(-T) and x = L^(-T)*y
m_reduce_hermitian_banded_generalized :: proc {
	m_reduce_hermitian_banded_generalized_c64,
	m_reduce_hermitian_banded_generalized_c128,
}

m_reduce_hermitian_banded_generalized_c64 :: proc(
	AB: ^Matrix(complex64), // Hermitian matrix A in banded storage
	BB: ^Matrix(complex64), // Positive definite matrix B in banded storage
	compute_z: bool = false, // Compute transformation matrix Z
	allocator := context.allocator,
) -> (
	X: Matrix(complex64),
	info: lapack.blasint, // Transformation matrix (if computed)
) {
	assert(AB.format == .Banded, "Matrix AB must be banded format")
	assert(BB.format == .Banded, "Matrix BB must be banded format")

	n := AB.rows
	ka := AB.storage.banded.ku // Bandwidth of A
	kb := BB.storage.banded.ku // Bandwidth of B
	ldab := AB.storage.banded.ldab
	ldbb := BB.storage.banded.ldab

	vect_c := cstring("V") if compute_z else cstring("N")
	uplo_c := cstring("U")

	// Allocate transformation matrix if needed
	ldx := n if compute_z else 1
	X = make_matrix(complex64, n, n, .General, allocator) if compute_z else Matrix(complex64){}

	// Workspace
	work := make([]complex64, n, allocator)
	rwork := make([]f32, n, allocator)

	lapack.chbgst_(
		vect_c,
		uplo_c,
		&n,
		&ka,
		&kb,
		raw_data(AB.data),
		&ldab,
		raw_data(BB.data),
		&ldbb,
		matrix_data_ptr(&X) if compute_z else nil,
		&ldx,
		raw_data(work),
		raw_data(rwork),
		&info,
		1,
		1,
	)

	return X, info
}

m_reduce_hermitian_banded_generalized_c128 :: proc(
	AB: ^Matrix(complex128), // Hermitian matrix A in banded storage
	BB: ^Matrix(complex128), // Positive definite matrix B in banded storage
	compute_z: bool = false, // Compute transformation matrix Z
	allocator := context.allocator,
) -> (
	X: Matrix(complex128),
	info: lapack.blasint, // Transformation matrix (if computed)
) {
	assert(AB.format == .Banded, "Matrix AB must be banded format")
	assert(BB.format == .Banded, "Matrix BB must be banded format")

	n := AB.rows
	ka := AB.storage.banded.ku // Bandwidth of A
	kb := BB.storage.banded.ku // Bandwidth of B
	ldab := AB.storage.banded.ldab
	ldbb := BB.storage.banded.ldab

	vect_c := cstring("V") if compute_z else cstring("N")
	uplo_c := cstring("U")

	// Allocate transformation matrix if needed
	ldx := n if compute_z else 1
	X = make_matrix(complex128, n, n, .General, allocator) if compute_z else Matrix(complex128){}

	// Workspace
	work := make([]complex128, n, allocator)
	rwork := make([]f64, n, allocator)

	lapack.zhbgst_(
		vect_c,
		uplo_c,
		&n,
		&ka,
		&kb,
		raw_data(AB.data),
		&ldab,
		raw_data(BB.data),
		&ldbb,
		matrix_data_ptr(&X) if compute_z else nil,
		&ldx,
		raw_data(work),
		raw_data(rwork),
		&info,
		1,
		1,
	)

	return X, info
}

// Generalized Hermitian banded eigenvalue solver (basic)
// Solves A*x = lambda*B*x where A is Hermitian and B is positive definite
m_eigen_generalized_hermitian_banded :: proc {
	m_eigen_generalized_hermitian_banded_c64,
	m_eigen_generalized_hermitian_banded_c128,
}

m_eigen_generalized_hermitian_banded_c64 :: proc(
	AB: ^Matrix(complex64), // Hermitian matrix A in banded storage
	BB: ^Matrix(complex64), // Positive definite matrix B in banded storage
	compute_vectors: bool = false,
	allocator := context.allocator,
) -> (
	w: []f32,
	Z: Matrix(complex64),
	info: lapack.blasint, // Eigenvalues// Eigenvectors (if computed)
) {
	assert(AB.format == .Banded, "Matrix AB must be banded format")
	assert(BB.format == .Banded, "Matrix BB must be banded format")

	n := AB.rows
	ka := AB.storage.banded.ku
	kb := BB.storage.banded.ku
	ldab := AB.storage.banded.ldab
	ldbb := BB.storage.banded.ldab

	jobz_c := cstring("V") if compute_vectors else cstring("N")
	uplo_c := cstring("U")

	// Allocate outputs
	w = make([]f32, n, allocator)
	ldz := n if compute_vectors else 1
	Z =
		make_matrix(complex64, n, n, .General, allocator) if compute_vectors else Matrix(complex64){}

	// Workspace
	work := make([]complex64, n, allocator)
	rwork := make([]f32, 3 * n, allocator)

	lapack.chbgv_(
		jobz_c,
		uplo_c,
		&n,
		&ka,
		&kb,
		raw_data(AB.data),
		&ldab,
		raw_data(BB.data),
		&ldbb,
		raw_data(w),
		matrix_data_ptr(&Z) if compute_vectors else nil,
		&ldz,
		raw_data(work),
		raw_data(rwork),
		&info,
		1,
		1,
	)

	return w, Z, info
}

m_eigen_generalized_hermitian_banded_c128 :: proc(
	AB: ^Matrix(complex128), // Hermitian matrix A in banded storage
	BB: ^Matrix(complex128), // Positive definite matrix B in banded storage
	compute_vectors: bool = false,
	allocator := context.allocator,
) -> (
	w: []f64,
	Z: Matrix(complex128),
	info: lapack.blasint, // Eigenvalues// Eigenvectors (if computed)
) {
	assert(AB.format == .Banded, "Matrix AB must be banded format")
	assert(BB.format == .Banded, "Matrix BB must be banded format")

	n := AB.rows
	ka := AB.storage.banded.ku
	kb := BB.storage.banded.ku
	ldab := AB.storage.banded.ldab
	ldbb := BB.storage.banded.ldab

	jobz_c := cstring("V") if compute_vectors else cstring("N")
	uplo_c := cstring("U")

	// Allocate outputs
	w = make([]f64, n, allocator)
	ldz := n if compute_vectors else 1
	Z =
		make_matrix(complex128, n, n, .General, allocator) if compute_vectors else Matrix(complex128){}

	// Workspace
	work := make([]complex128, n, allocator)
	rwork := make([]f64, 3 * n, allocator)

	lapack.zhbgv_(
		jobz_c,
		uplo_c,
		&n,
		&ka,
		&kb,
		raw_data(AB.data),
		&ldab,
		raw_data(BB.data),
		&ldbb,
		raw_data(w),
		matrix_data_ptr(&Z) if compute_vectors else nil,
		&ldz,
		raw_data(work),
		raw_data(rwork),
		&info,
		1,
		1,
	)

	return w, Z, info
}

// Generalized Hermitian banded eigenvalue solver (divide-and-conquer)
// Faster for large matrices when eigenvectors are needed
m_eigen_generalized_hermitian_banded_dc :: proc {
	m_eigen_generalized_hermitian_banded_dc_c64,
	m_eigen_generalized_hermitian_banded_dc_c128,
}

m_eigen_generalized_hermitian_banded_dc_c64 :: proc(
	AB: ^Matrix(complex64), // Hermitian matrix A in banded storage
	BB: ^Matrix(complex64), // Positive definite matrix B in banded storage
	compute_vectors: bool = false,
	allocator := context.allocator,
) -> (
	w: []f32,
	Z: Matrix(complex64),
	info: lapack.blasint, // Eigenvalues// Eigenvectors (if computed)
) {
	assert(AB.format == .Banded, "Matrix AB must be banded format")
	assert(BB.format == .Banded, "Matrix BB must be banded format")

	n := AB.rows
	ka := AB.storage.banded.ku
	kb := BB.storage.banded.ku
	ldab := AB.storage.banded.ldab
	ldbb := BB.storage.banded.ldab

	jobz_c := cstring("V") if compute_vectors else cstring("N")
	uplo_c := cstring("U")

	// Allocate outputs
	w = make([]f32, n, allocator)
	ldz := n if compute_vectors else 1
	Z =
		make_matrix(complex64, n, n, .General, allocator) if compute_vectors else Matrix(complex64){}

	// Workspace query
	work_query: complex64
	rwork_query: f32
	iwork_query: lapack.blasint
	lwork: lapack.blasint = -1
	lrwork: lapack.blasint = -1
	liwork: lapack.blasint = -1

	lapack.chbgvd_(
		jobz_c,
		uplo_c,
		&n,
		&ka,
		&kb,
		raw_data(AB.data),
		&ldab,
		raw_data(BB.data),
		&ldbb,
		raw_data(w),
		matrix_data_ptr(&Z) if compute_vectors else &work_query,
		&ldz,
		&work_query,
		&lwork,
		&rwork_query,
		&lrwork,
		&iwork_query,
		&liwork,
		&info,
		1,
		1,
	)

	if info != 0 do return

	// Allocate workspace
	lwork = auto_cast real(work_query)
	work := make([]complex64, lwork, allocator)

	lrwork_calc := 1 + 5 * n + 2 * n * n if compute_vectors else 1 + 2 * n
	lrwork = auto_cast rwork_query if auto_cast rwork_query > lrwork_calc else lrwork_calc
	rwork := make([]f32, lrwork, allocator)

	liwork_calc := 3 + 5 * n if compute_vectors else 1
	liwork = iwork_query if iwork_query > liwork_calc else liwork_calc
	iwork := make([]lapack.blasint, liwork, allocator)

	lapack.chbgvd_(
		jobz_c,
		uplo_c,
		&n,
		&ka,
		&kb,
		raw_data(AB.data),
		&ldab,
		raw_data(BB.data),
		&ldbb,
		raw_data(w),
		matrix_data_ptr(&Z) if compute_vectors else nil,
		&ldz,
		raw_data(work),
		&lwork,
		raw_data(rwork),
		&lrwork,
		raw_data(iwork),
		&liwork,
		&info,
		1,
		1,
	)

	return w, Z, info
}

m_eigen_generalized_hermitian_banded_dc_c128 :: proc(
	AB: ^Matrix(complex128), // Hermitian matrix A in banded storage
	BB: ^Matrix(complex128), // Positive definite matrix B in banded storage
	compute_vectors: bool = false,
	allocator := context.allocator,
) -> (
	w: []f64,
	Z: Matrix(complex128),
	info: lapack.blasint, // Eigenvalues// Eigenvectors (if computed)
) {
	assert(AB.format == .Banded, "Matrix AB must be banded format")
	assert(BB.format == .Banded, "Matrix BB must be banded format")

	n := AB.rows
	ka := AB.storage.banded.ku
	kb := BB.storage.banded.ku
	ldab := AB.storage.banded.ldab
	ldbb := BB.storage.banded.ldab

	jobz_c := cstring("V") if compute_vectors else cstring("N")
	uplo_c := cstring("U")

	// Allocate outputs
	w = make([]f64, n, allocator)
	ldz := n if compute_vectors else 1
	Z =
		make_matrix(complex128, n, n, .General, allocator) if compute_vectors else Matrix(complex128){}

	// Workspace query
	work_query: complex128
	rwork_query: f64
	iwork_query: lapack.blasint
	lwork: lapack.blasint = -1
	lrwork: lapack.blasint = -1
	liwork: lapack.blasint = -1

	lapack.zhbgvd_(
		jobz_c,
		uplo_c,
		&n,
		&ka,
		&kb,
		raw_data(AB.data),
		&ldab,
		raw_data(BB.data),
		&ldbb,
		raw_data(w),
		matrix_data_ptr(&Z) if compute_vectors else &work_query,
		&ldz,
		&work_query,
		&lwork,
		&rwork_query,
		&lrwork,
		&iwork_query,
		&liwork,
		&info,
		1,
		1,
	)

	if info != 0 do return

	// Allocate workspace
	lwork = auto_cast real(work_query)
	work := make([]complex128, lwork, allocator)

	lrwork_calc := 1 + 5 * n + 2 * n * n if compute_vectors else 1 + 2 * n
	lrwork = auto_cast rwork_query if auto_cast rwork_query > lrwork_calc else lrwork_calc
	rwork := make([]f64, lrwork, allocator)

	liwork_calc := 3 + 5 * n if compute_vectors else 1
	liwork = iwork_query if iwork_query > liwork_calc else liwork_calc
	iwork := make([]lapack.blasint, liwork, allocator)

	lapack.zhbgvd_(
		jobz_c,
		uplo_c,
		&n,
		&ka,
		&kb,
		raw_data(AB.data),
		&ldab,
		raw_data(BB.data),
		&ldbb,
		raw_data(w),
		matrix_data_ptr(&Z) if compute_vectors else nil,
		&ldz,
		raw_data(work),
		&lwork,
		raw_data(rwork),
		&lrwork,
		raw_data(iwork),
		&liwork,
		&info,
		1,
		1,
	)

	return w, Z, info
}

// Generalized Hermitian banded eigenvalue solver (expert with subset selection)
// Computes selected eigenvalues and optionally eigenvectors
m_eigen_generalized_hermitian_banded_expert :: proc {
	m_eigen_generalized_hermitian_banded_expert_c64,
	m_eigen_generalized_hermitian_banded_expert_c128,
}

m_eigen_generalized_hermitian_banded_expert_c64 :: proc(
	AB: ^Matrix(complex64), // Hermitian matrix A in banded storage
	BB: ^Matrix(complex64), // Positive definite matrix B in banded storage
	range_type: EigenValueRange = .All,
	compute_vectors: bool = false,
	il: int = 1, // Lower index (1-based) for .Indexed
	iu: int = 0, // Upper index (1-based) for .Indexed (0 means use n)
	vl: f32 = 0, // Lower bound for .Valued
	vu: f32 = 0, // Upper bound for .Valued
	abstol: f32 = 0, // Absolute tolerance (0 for default)
	allocator := context.allocator,
) -> (
	m: int,
	w: []f32,
	Z: Matrix(complex64),// Number of eigenvalues found
	ifail: []lapack.blasint,// Eigenvalues
	info: lapack.blasint, // Eigenvectors (if computed)// Indices of failed eigenvectors
) {
	assert(AB.format == .Banded, "Matrix AB must be banded format")
	assert(BB.format == .Banded, "Matrix BB must be banded format")

	n := AB.rows
	ka := AB.storage.banded.ku
	kb := BB.storage.banded.ku
	ldab := AB.storage.banded.ldab
	ldbb := BB.storage.banded.ldab

	jobz_c := cstring("V") if compute_vectors else cstring("N")
	uplo_c := cstring("U")

	// Range specification
	range_c: cstring
	il_use := il
	iu_use := iu if iu > 0 else n

	switch range_type {
	case .All:
		range_c = cstring("A")
	case .Indexed:
		range_c = cstring("I")
	case .Valued:
		range_c = cstring("V")
	}

	// Allocate outputs
	w = make([]f32, n, allocator)
	ldz := n if compute_vectors else 1
	Z =
		make_matrix(complex64, n, n, .General, allocator) if compute_vectors else Matrix(complex64){}
	ifail = make([]lapack.blasint, n, allocator)

	// Need Q matrix for transformation (n x n)
	ldq := n
	Q := make_matrix(complex64, n, n, .General, allocator)

	// Workspace
	work := make([]complex64, n, allocator)
	rwork := make([]f32, 7 * n, allocator)
	iwork := make([]lapack.blasint, 5 * n, allocator)

	lapack.chbgvx_(
		jobz_c,
		range_c,
		uplo_c,
		&n,
		&ka,
		&kb,
		raw_data(AB.data),
		&ldab,
		raw_data(BB.data),
		&ldbb,
		matrix_data_ptr(&Q),
		&ldq,
		&vl,
		&vu,
		&il_use,
		&iu_use,
		&abstol,
		&m,
		raw_data(w),
		matrix_data_ptr(&Z) if compute_vectors else nil,
		&ldz,
		raw_data(work),
		raw_data(rwork),
		raw_data(iwork),
		raw_data(ifail),
		&info,
		1,
		1,
		1,
	)

	return m, w, Z, ifail, info
}

m_eigen_generalized_hermitian_banded_expert_c128 :: proc(
	AB: ^Matrix(complex128), // Hermitian matrix A in banded storage
	BB: ^Matrix(complex128), // Positive definite matrix B in banded storage
	range_type: EigenValueRange = .All,
	compute_vectors: bool = false,
	il: int = 1, // Lower index (1-based) for .Indexed
	iu: int = 0, // Upper index (1-based) for .Indexed (0 means use n)
	vl: f64 = 0, // Lower bound for .Valued
	vu: f64 = 0, // Upper bound for .Valued
	abstol: f64 = 0, // Absolute tolerance (0 for default)
	allocator := context.allocator,
) -> (
	m: int,
	w: []f64,
	Z: Matrix(complex128),// Number of eigenvalues found
	ifail: []lapack.blasint,// Eigenvalues
	info: lapack.blasint, // Eigenvectors (if computed)// Indices of failed eigenvectors
) {
	assert(AB.format == .Banded, "Matrix AB must be banded format")
	assert(BB.format == .Banded, "Matrix BB must be banded format")

	n := AB.rows
	ka := AB.storage.banded.ku
	kb := BB.storage.banded.ku
	ldab := AB.storage.banded.ldab
	ldbb := BB.storage.banded.ldab

	jobz_c := cstring("V") if compute_vectors else cstring("N")
	uplo_c := cstring("U")

	// Range specification
	range_c: cstring
	il_use := il
	iu_use := iu if iu > 0 else n

	switch range_type {
	case .All:
		range_c = cstring("A")
	case .Indexed:
		range_c = cstring("I")
	case .Valued:
		range_c = cstring("V")
	}

	// Allocate outputs
	w = make([]f64, n, allocator)
	ldz := n if compute_vectors else 1
	Z =
		make_matrix(complex128, n, n, .General, allocator) if compute_vectors else Matrix(complex128){}
	ifail = make([]lapack.blasint, n, allocator)

	// Need Q matrix for transformation (n x n)
	ldq := n
	Q := make_matrix(complex128, n, n, .General, allocator)

	// Workspace
	work := make([]complex128, n, allocator)
	rwork := make([]f64, 7 * n, allocator)
	iwork := make([]lapack.blasint, 5 * n, allocator)

	lapack.zhbgvx_(
		jobz_c,
		range_c,
		uplo_c,
		&n,
		&ka,
		&kb,
		raw_data(AB.data),
		&ldab,
		raw_data(BB.data),
		&ldbb,
		matrix_data_ptr(&Q),
		&ldq,
		&vl,
		&vu,
		&il_use,
		&iu_use,
		&abstol,
		&m,
		raw_data(w),
		matrix_data_ptr(&Z) if compute_vectors else nil,
		&ldz,
		raw_data(work),
		raw_data(rwork),
		raw_data(iwork),
		raw_data(ifail),
		&info,
		1,
		1,
		1,
	)

	return m, w, Z, ifail, info
}

// ===================================================================================
// HERMITIAN BANDED TRIDIAGONALIZATION
// ===================================================================================

// Reduce Hermitian banded matrix to tridiagonal form
// A = Q * T * Q^H where T is tridiagonal and Q is unitary
m_tridiagonalize_hermitian_banded :: proc {
	m_tridiagonalize_hermitian_banded_c64,
	m_tridiagonalize_hermitian_banded_c128,
}

m_tridiagonalize_hermitian_banded_c64 :: proc(
	AB: ^Matrix(complex64), // Hermitian banded matrix (input/output)
	compute_q: bool = false, // Compute orthogonal transformation matrix Q
	allocator := context.allocator,
) -> (
	d: []f32,
	e: []f32,
	Q: Matrix(complex64),// Diagonal elements of tridiagonal matrix
	info: lapack.blasint, // Subdiagonal elements of tridiagonal matrix// Orthogonal transformation matrix (if computed)
) {
	assert(AB.format == .Banded, "Matrix must be banded format")

	n := AB.rows
	kd := AB.storage.banded.ku
	ldab := AB.storage.banded.ldab

	vect_c := cstring("V") if compute_q else cstring("N")
	uplo_c := cstring("U")

	// Allocate outputs
	d = make([]f32, n, allocator)
	e = make([]f32, n - 1, allocator)

	// Allocate Q matrix if needed
	ldq := n if compute_q else 1
	Q = make_matrix(complex64, n, n, .General, allocator) if compute_q else Matrix(complex64){}

	// Workspace
	work := make([]complex64, n, allocator)

	lapack.chbtrd_(
		vect_c,
		uplo_c,
		&n,
		&kd,
		raw_data(AB.data),
		&ldab,
		raw_data(d),
		raw_data(e),
		matrix_data_ptr(&Q) if compute_q else nil,
		&ldq,
		raw_data(work),
		&info,
		1,
		1,
	)

	return d, e, Q, info
}

m_tridiagonalize_hermitian_banded_c128 :: proc(
	AB: ^Matrix(complex128), // Hermitian banded matrix (input/output)
	compute_q: bool = false, // Compute orthogonal transformation matrix Q
	allocator := context.allocator,
) -> (
	d: []f64,
	e: []f64,
	Q: Matrix(complex128),// Diagonal elements of tridiagonal matrix
	info: lapack.blasint, // Subdiagonal elements of tridiagonal matrix// Orthogonal transformation matrix (if computed)
) {
	assert(AB.format == .Banded, "Matrix must be banded format")

	n := AB.rows
	kd := AB.storage.banded.ku
	ldab := AB.storage.banded.ldab

	vect_c := cstring("V") if compute_q else cstring("N")
	uplo_c := cstring("U")

	// Allocate outputs
	d = make([]f64, n, allocator)
	e = make([]f64, n - 1, allocator)

	// Allocate Q matrix if needed
	ldq := n if compute_q else 1
	Q = make_matrix(complex128, n, n, .General, allocator) if compute_q else Matrix(complex128){}

	// Workspace
	work := make([]complex128, n, allocator)

	lapack.zhbtrd_(
		vect_c,
		uplo_c,
		&n,
		&kd,
		raw_data(AB.data),
		&ldab,
		raw_data(d),
		raw_data(e),
		matrix_data_ptr(&Q) if compute_q else nil,
		&ldq,
		raw_data(work),
		&info,
		1,
		1,
	)

	return d, e, Q, info
}
