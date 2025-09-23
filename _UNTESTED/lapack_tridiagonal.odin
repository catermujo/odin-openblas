package openblas

import lapack "./f77"
import builtin "base:builtin"
import "core:c"

// Create a tridiagonal matrix using the unified Matrix type
make_tridiagonal_matrix :: proc(T: typeid, n: int, allocator := context.allocator) -> Matrix(T) {
	// Allocate data: [dl(n-1) | d(n) | du(n-1)]
	total_size := (n - 1) + n + (n - 1) // dl + d + du
	data := builtin.make([]T, total_size, allocator)

	return Matrix(T) {
		data = data,
		rows = n,
		cols = n,
		ld = n,
		format = .Tridiagonal,
		storage = {
			tridiagonal = {
				dl_offset = 0, // Lower diagonal starts at 0
				d_offset  = n - 1, // Main diagonal starts at n-1
				du_offset = n - 1 + n, // Upper diagonal starts at 2n-1
			},
		},
	}
}

// Helper functions to access diagonals
get_tridiagonal_dl :: proc(mat: ^Matrix($T)) -> []T {
	assert(mat.format == .Tridiagonal, "Matrix must be tridiagonal")
	n := mat.rows
	start := mat.storage.tridiagonal.dl_offset
	return mat.data[start:start + n - 1]
}

get_tridiagonal_d :: proc(mat: ^Matrix($T)) -> []T {
	assert(mat.format == .Tridiagonal, "Matrix must be tridiagonal")
	n := mat.rows
	start := mat.storage.tridiagonal.d_offset
	return mat.data[start:start + n]
}

get_tridiagonal_du :: proc(mat: ^Matrix($T)) -> []T {
	assert(mat.format == .Tridiagonal, "Matrix must be tridiagonal")
	n := mat.rows
	start := mat.storage.tridiagonal.du_offset
	return mat.data[start:start + n - 1]
}

// ===================================================================================
// TRIDIAGONAL CONDITION NUMBER ESTIMATION
// ===================================================================================

// Estimate condition number of a tridiagonal matrix
m_condition_tridiagonal :: proc {
	m_condition_tridiagonal_real,
	m_condition_tridiagonal_c64,
	m_condition_tridiagonal_c128,
}

m_condition_tridiagonal_real :: proc(
	tri: ^Matrix($T), // Original tridiagonal matrix
	tri_factored: ^Matrix(T), // Factored form from gttrf
	du2: []T, // Second superdiagonal from factorization
	ipiv: []i32, // Pivot indices from factorization
	anorm: T, // 1-norm or infinity-norm of original matrix
	use_infinity_norm: bool = false,
	allocator := context.allocator,
) -> (
	rcond: T,
	info: Info,
) where T == f32 || T == f64 {
	assert(tri.format == .Tridiagonal, "Matrix must be tridiagonal")
	assert(tri_factored.format == .Tridiagonal, "Factored matrix must be tridiagonal")
	n := Blas_Int(tri.rows)

	// Get diagonal arrays
	dl_factored := get_tridiagonal_dl(tri_factored)
	d_factored := get_tridiagonal_d(tri_factored)
	du_factored := get_tridiagonal_du(tri_factored)

	// Set norm type
	norm_c := use_infinity_norm ? cstring("I") : cstring("1")

	// Allocate workspace
	work := builtin.make([]T, 2 * n, allocator)
	defer builtin.delete(work)

	iwork := builtin.make([]i32, n, allocator)
	defer builtin.delete(iwork)

	when T == f32 {
		lapack.sgtcon_(
			norm_c,
			&n,
			raw_data(dl_factored),
			raw_data(d_factored),
			raw_data(du_factored),
			raw_data(du2),
			raw_data(ipiv),
			&anorm,
			&rcond,
			raw_data(work),
			raw_data(iwork),
			&info,
			1,
		)
	} else {
		lapack.dgtcon_(
			norm_c,
			&n,
			raw_data(dl_factored),
			raw_data(d_factored),
			raw_data(du_factored),
			raw_data(du2),
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

m_condition_tridiagonal_c64 :: proc(
	tri: ^Matrix(complex64), // Original tridiagonal matrix
	tri_factored: ^Matrix(complex64), // Factored form from gttrf
	du2: []complex64, // Second superdiagonal from factorization
	ipiv: []i32, // Pivot indices from factorization
	anorm: f32, // 1-norm or infinity-norm of original matrix
	use_infinity_norm: bool = false,
	allocator := context.allocator,
) -> (
	rcond: f32,
	info: Info,
) {
	assert(tri.format == .Tridiagonal, "Matrix must be tridiagonal")
	assert(tri_factored.format == .Tridiagonal, "Factored matrix must be tridiagonal")
	n := Blas_Int(tri.rows)

	// Get diagonal arrays
	dl_factored := get_tridiagonal_dl(tri_factored)
	d_factored := get_tridiagonal_d(tri_factored)
	du_factored := get_tridiagonal_du(tri_factored)

	// Set norm type
	norm_c := use_infinity_norm ? cstring("I") : cstring("1")

	// Allocate workspace
	work := builtin.make([]complex64, 2 * n, allocator)
	defer builtin.delete(work)

	lapack.cgtcon_(
		norm_c,
		&n,
		raw_data(dl_factored),
		raw_data(d_factored),
		raw_data(du_factored),
		raw_data(du2),
		raw_data(ipiv),
		&anorm,
		&rcond,
		raw_data(work),
		&info,
		1,
	)

	return rcond, info
}

m_condition_tridiagonal_c128 :: proc(
	tri: ^Matrix(complex128), // Original tridiagonal matrix
	tri_factored: ^Matrix(complex128), // Factored form from gttrf
	du2: []complex128, // Second superdiagonal from factorization
	ipiv: []i32, // Pivot indices from factorization
	anorm: f64, // 1-norm or infinity-norm of original matrix
	use_infinity_norm: bool = false,
	allocator := context.allocator,
) -> (
	rcond: f64,
	info: Info,
) {
	assert(tri.format == .Tridiagonal, "Matrix must be tridiagonal")
	assert(tri_factored.format == .Tridiagonal, "Factored matrix must be tridiagonal")
	n := Blas_Int(tri.rows)

	// Get diagonal arrays
	dl_factored := get_tridiagonal_dl(tri_factored)
	d_factored := get_tridiagonal_d(tri_factored)
	du_factored := get_tridiagonal_du(tri_factored)

	// Set norm type
	norm_c := use_infinity_norm ? cstring("I") : cstring("1")

	// Allocate workspace
	work := builtin.make([]complex128, 2 * n, allocator)
	defer builtin.delete(work)

	lapack.zgtcon_(
		norm_c,
		&n,
		raw_data(dl_factored),
		raw_data(d_factored),
		raw_data(du_factored),
		raw_data(du2),
		raw_data(ipiv),
		&anorm,
		&rcond,
		raw_data(work),
		&info,
		1,
	)

	return rcond, info
}

// ===================================================================================
// TRIDIAGONAL ITERATIVE REFINEMENT
// ===================================================================================

// Improve solution and provide error bounds for tridiagonal systems
m_refine_tridiagonal :: proc {
	m_refine_tridiagonal_real,
	m_refine_tridiagonal_c64,
	m_refine_tridiagonal_c128,
}

m_refine_tridiagonal_real :: proc(
	tri: ^TridiagonalMatrix($T),
	tri_factored: ^TridiagonalMatrix(T), // Factored form from gttrf
	du2: []T, // Second superdiagonal from factorization
	ipiv: []i32, // Pivot indices from factorization
	B: ^Matrix(T), // Right-hand side matrix
	X: ^Matrix(T), // Solution matrix to be refined
	transpose: bool = false,
	allocator := context.allocator,
) -> (
	ferr, berr: []T,
	info: Info, // Forward and backward error bounds
) where T == f32 || T == f64 {
	n := Blas_Int(tri.n)
	nrhs := Blas_Int(B.cols)
	ldb := Blas_Int(B.ld)
	ldx := Blas_Int(X.ld)

	// Set transpose parameter
	trans_c := transpose ? cstring("T") : cstring("N")

	// Allocate error bound arrays
	ferr = builtin.make([]T, nrhs, allocator)
	berr = builtin.make([]T, nrhs, allocator)

	// Allocate workspace
	work := builtin.make([]T, 3 * n, allocator)
	defer builtin.delete(work)

	iwork := builtin.make([]i32, n, allocator)
	defer builtin.delete(iwork)

	when T == f32 {
		lapack.sgtrfs_(
			trans_c,
			&n,
			&nrhs,
			raw_data(tri.dl),
			raw_data(tri.d),
			raw_data(tri.du),
			raw_data(tri_factored.dl),
			raw_data(tri_factored.d),
			raw_data(tri_factored.du),
			raw_data(du2),
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
		lapack.dgtrfs_(
			trans_c,
			&n,
			&nrhs,
			raw_data(tri.dl),
			raw_data(tri.d),
			raw_data(tri.du),
			raw_data(tri_factored.dl),
			raw_data(tri_factored.d),
			raw_data(tri_factored.du),
			raw_data(du2),
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

m_refine_tridiagonal_c64 :: proc(
	tri: ^TridiagonalMatrix(complex64),
	tri_factored: ^TridiagonalMatrix(complex64), // Factored form from gttrf
	du2: []complex64, // Second superdiagonal from factorization
	ipiv: []i32, // Pivot indices from factorization
	B: ^Matrix(complex64), // Right-hand side matrix
	X: ^Matrix(complex64), // Solution matrix to be refined
	transpose: bool = false,
	conjugate_transpose: bool = false,
	allocator := context.allocator,
) -> (
	ferr, berr: []f32,
	info: Info, // Forward and backward error bounds
) {
	n := Blas_Int(tri.n)
	nrhs := Blas_Int(B.cols)
	ldb := Blas_Int(B.ld)
	ldx := Blas_Int(X.ld)

	// Set transpose parameter
	trans_c: cstring
	if conjugate_transpose {
		trans_c = "C"
	} else if transpose {
		trans_c = "T"
	} else {
		trans_c = "N"
	}

	// Allocate error bound arrays (real for complex matrices)
	ferr = builtin.make([]f32, nrhs, allocator)
	berr = builtin.make([]f32, nrhs, allocator)

	// Allocate workspace
	work := builtin.make([]complex64, 2 * n, allocator)
	defer builtin.delete(work)

	rwork := builtin.make([]f32, n, allocator)
	defer builtin.delete(rwork)

	lapack.cgtrfs_(
		trans_c,
		&n,
		&nrhs,
		raw_data(tri.dl),
		raw_data(tri.d),
		raw_data(tri.du),
		raw_data(tri_factored.dl),
		raw_data(tri_factored.d),
		raw_data(tri_factored.du),
		raw_data(du2),
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

m_refine_tridiagonal_c128 :: proc(
	tri: ^TridiagonalMatrix(complex128),
	tri_factored: ^TridiagonalMatrix(complex128), // Factored form from gttrf
	du2: []complex128, // Second superdiagonal from factorization
	ipiv: []i32, // Pivot indices from factorization
	B: ^Matrix(complex128), // Right-hand side matrix
	X: ^Matrix(complex128), // Solution matrix to be refined
	transpose: bool = false,
	conjugate_transpose: bool = false,
	allocator := context.allocator,
) -> (
	ferr, berr: []f64,
	info: Info, // Forward and backward error bounds
) {
	n := Blas_Int(tri.n)
	nrhs := Blas_Int(B.cols)
	ldb := Blas_Int(B.ld)
	ldx := Blas_Int(X.ld)

	// Set transpose parameter
	trans_c: cstring
	if conjugate_transpose {
		trans_c = "C"
	} else if transpose {
		trans_c = "T"
	} else {
		trans_c = "N"
	}

	// Allocate error bound arrays (real for complex matrices)
	ferr = builtin.make([]f64, nrhs, allocator)
	berr = builtin.make([]f64, nrhs, allocator)

	// Allocate workspace
	work := builtin.make([]complex128, 2 * n, allocator)
	defer builtin.delete(work)

	rwork := builtin.make([]f64, n, allocator)
	defer builtin.delete(rwork)

	lapack.zgtrfs_(
		trans_c,
		&n,
		&nrhs,
		raw_data(tri.dl),
		raw_data(tri.d),
		raw_data(tri.du),
		raw_data(tri_factored.dl),
		raw_data(tri_factored.d),
		raw_data(tri_factored.du),
		raw_data(du2),
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

// ===================================================================================
// TRIDIAGONAL DIRECT SOLVER
// ===================================================================================

// Solve tridiagonal system directly (combines factorization and solving)
m_solve_tridiagonal :: proc {
	m_solve_tridiagonal_real,
	m_solve_tridiagonal_c64,
	m_solve_tridiagonal_c128,
}

m_solve_tridiagonal_real :: proc(
	tri: ^TridiagonalMatrix($T),
	B: ^Matrix(T),
	allocator := context.allocator,
) -> (
	info: Info,
) where T == f32 ||
	T == f64 {
	n := Blas_Int(tri.n)
	nrhs := Blas_Int(B.cols)
	ldb := Blas_Int(B.ld)

	// Copy tridiagonal matrix data as it will be modified
	dl_copy := builtin.make([]T, len(tri.dl), allocator)
	d_copy := builtin.make([]T, len(tri.d), allocator)
	du_copy := builtin.make([]T, len(tri.du), allocator)
	defer builtin.delete(dl_copy)
	defer builtin.delete(d_copy)
	defer builtin.delete(du_copy)

	copy(dl_copy, tri.dl)
	copy(d_copy, tri.d)
	copy(du_copy, tri.du)

	when T == f32 {
		lapack.sgtsv_(
			&n,
			&nrhs,
			raw_data(dl_copy),
			raw_data(d_copy),
			raw_data(du_copy),
			raw_data(B.data),
			&ldb,
			&info,
		)
	} else {
		lapack.dgtsv_(
			&n,
			&nrhs,
			raw_data(dl_copy),
			raw_data(d_copy),
			raw_data(du_copy),
			raw_data(B.data),
			&ldb,
			&info,
		)
	}

	return info
}

m_solve_tridiagonal_c64 :: proc(
	tri: ^TridiagonalMatrix(complex64),
	B: ^Matrix(complex64),
	allocator := context.allocator,
) -> (
	info: Info,
) {
	n := Blas_Int(tri.n)
	nrhs := Blas_Int(B.cols)
	ldb := Blas_Int(B.ld)

	// Copy tridiagonal matrix data as it will be modified
	dl_copy := builtin.make([]complex64, len(tri.dl), allocator)
	d_copy := builtin.make([]complex64, len(tri.d), allocator)
	du_copy := builtin.make([]complex64, len(tri.du), allocator)
	defer builtin.delete(dl_copy)
	defer builtin.delete(d_copy)
	defer builtin.delete(du_copy)

	copy(dl_copy, tri.dl)
	copy(d_copy, tri.d)
	copy(du_copy, tri.du)

	lapack.cgtsv_(
		&n,
		&nrhs,
		raw_data(dl_copy),
		raw_data(d_copy),
		raw_data(du_copy),
		raw_data(B.data),
		&ldb,
		&info,
	)

	return info
}

m_solve_tridiagonal_c128 :: proc(
	tri: ^TridiagonalMatrix(complex128),
	B: ^Matrix(complex128),
	allocator := context.allocator,
) -> (
	info: Info,
) {
	n := Blas_Int(tri.n)
	nrhs := Blas_Int(B.cols)
	ldb := Blas_Int(B.ld)

	// Copy tridiagonal matrix data as it will be modified
	dl_copy := builtin.make([]complex128, len(tri.dl), allocator)
	d_copy := builtin.make([]complex128, len(tri.d), allocator)
	du_copy := builtin.make([]complex128, len(tri.du), allocator)
	defer builtin.delete(dl_copy)
	defer builtin.delete(d_copy)
	defer builtin.delete(du_copy)

	copy(dl_copy, tri.dl)
	copy(d_copy, tri.d)
	copy(du_copy, tri.du)

	lapack.zgtsv_(
		&n,
		&nrhs,
		raw_data(dl_copy),
		raw_data(d_copy),
		raw_data(du_copy),
		raw_data(B.data),
		&ldb,
		&info,
	)

	return info
}

// Convenience function to solve Ax = b for vector b
m_solve_tridiagonal_vector :: proc(
	tri: ^Matrix($T),
	b: []T,
	allocator := context.allocator,
) -> (
	x: []T,
	info: Info,
) {
	// Create matrix from vector
	B := make_matrix(T, len(b), 1, allocator)
	defer builtin.delete(B.data)

	for i in 0 ..< len(b) {
		B.data[i] = b[i]
	}

	info = m_solve_tridiagonal(tri, &B, allocator)

	// Copy result back to vector
	x = builtin.make([]T, len(b), allocator)
	for i in 0 ..< len(b) {
		x[i] = B.data[i]
	}

	return x, info
}

// ===================================================================================
// TRIDIAGONAL EXPERT SOLVER WITH CONDITION ESTIMATION AND ERROR BOUNDS
// ===================================================================================

// Expert driver with condition estimation and error bounds
m_solve_tridiagonal_expert :: proc {
	m_solve_tridiagonal_expert_real,
	m_solve_tridiagonal_expert_c64,
	m_solve_tridiagonal_expert_c128,
}

m_solve_tridiagonal_expert_real :: proc(
	tri: ^Matrix($T),
	B: ^Matrix(T),
	fact: FactorizationMode = .NotFactored,
	transpose: bool = false,
	tri_factored: ^Matrix(T) = nil,
	du2: []T = nil,
	ipiv: []i32 = nil,
	allocator := context.allocator,
) -> (
	X: Matrix(T),
	rcond: T,
	ferr, berr: []T,
	tri_factored_out: Matrix(T),
	du2_out: []T,
	ipiv_out: []i32,
	info: Info,
) where T == f32 ||
	T == f64 {
	assert(tri.format == .Tridiagonal, "Matrix must be tridiagonal")
	n := Blas_Int(tri.rows)
	nrhs := Blas_Int(B.cols)
	ldb := Blas_Int(B.ld)

	// Set fact and trans parameters
	fact_c := fact == .Factored ? cstring("F") : cstring("N")
	trans_c := transpose ? cstring("T") : cstring("N")

	// Get original diagonal arrays
	dl := get_tridiagonal_dl(tri)
	d := get_tridiagonal_d(tri)
	du := get_tridiagonal_du(tri)

	// Allocate solution matrix
	X = make_matrix(T, int(n), int(nrhs), allocator)
	ldx := Blas_Int(X.ld)

	// Allocate or use provided factored matrix
	if tri_factored == nil || fact == .NotFactored {
		tri_factored_out = make_tridiagonal_matrix(T, int(n), allocator)
	} else {
		tri_factored_out = tri_factored^
	}
	dlf := get_tridiagonal_dl(&tri_factored_out)
	df := get_tridiagonal_d(&tri_factored_out)
	duf := get_tridiagonal_du(&tri_factored_out)

	// Allocate or use provided du2 and ipiv
	if du2 == nil || fact == .NotFactored {
		du2_out = builtin.make([]T, n - 2, allocator)
	} else {
		du2_out = du2
	}

	if ipiv == nil || fact == .NotFactored {
		ipiv_out = builtin.make([]i32, n, allocator)
	} else {
		ipiv_out = ipiv
	}

	// Allocate error bounds
	ferr = builtin.make([]T, nrhs, allocator)
	berr = builtin.make([]T, nrhs, allocator)

	// Allocate workspace
	work := builtin.make([]T, 3 * n, allocator)
	defer builtin.delete(work)

	iwork := builtin.make([]i32, n, allocator)
	defer builtin.delete(iwork)

	when T == f32 {
		lapack.sgtsvx_(
			fact_c,
			trans_c,
			&n,
			&nrhs,
			raw_data(dl),
			raw_data(d),
			raw_data(du),
			raw_data(dlf),
			raw_data(df),
			raw_data(duf),
			raw_data(du2_out),
			raw_data(ipiv_out),
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
		)
	} else {
		lapack.dgtsvx_(
			fact_c,
			trans_c,
			&n,
			&nrhs,
			raw_data(dl),
			raw_data(d),
			raw_data(du),
			raw_data(dlf),
			raw_data(df),
			raw_data(duf),
			raw_data(du2_out),
			raw_data(ipiv_out),
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
		)
	}

	return X, rcond, ferr, berr, tri_factored_out, du2_out, ipiv_out, info
}

m_solve_tridiagonal_expert_c64 :: proc(
	tri: ^Matrix(complex64),
	B: ^Matrix(complex64),
	fact: FactorizationMode = .NotFactored,
	transpose: bool = false,
	conjugate_transpose: bool = false,
	tri_factored: ^Matrix(complex64) = nil,
	du2: []complex64 = nil,
	ipiv: []i32 = nil,
	allocator := context.allocator,
) -> (
	X: Matrix(complex64),
	rcond: f32,
	ferr, berr: []f32,
	tri_factored_out: Matrix(complex64),
	du2_out: []complex64,
	ipiv_out: []i32,
	info: Info,
) {
	assert(tri.format == .Tridiagonal, "Matrix must be tridiagonal")
	n := Blas_Int(tri.rows)
	nrhs := Blas_Int(B.cols)
	ldb := Blas_Int(B.ld)

	// Set fact and trans parameters
	fact_c := fact == .Factored ? cstring("F") : cstring("N")
	trans_c: cstring
	if conjugate_transpose {
		trans_c = "C"
	} else if transpose {
		trans_c = "T"
	} else {
		trans_c = "N"
	}

	// Get original diagonal arrays
	dl := get_tridiagonal_dl(tri)
	d := get_tridiagonal_d(tri)
	du := get_tridiagonal_du(tri)

	// Allocate solution matrix
	X = make_matrix(complex64, int(n), int(nrhs), allocator)
	ldx := Blas_Int(X.ld)

	// Allocate or use provided factored matrix
	if tri_factored == nil || fact == .NotFactored {
		tri_factored_out = make_tridiagonal_matrix(complex64, int(n), allocator)
	} else {
		tri_factored_out = tri_factored^
	}
	dlf := get_tridiagonal_dl(&tri_factored_out)
	df := get_tridiagonal_d(&tri_factored_out)
	duf := get_tridiagonal_du(&tri_factored_out)

	// Allocate or use provided du2 and ipiv
	if du2 == nil || fact == .NotFactored {
		du2_out = builtin.make([]complex64, n - 2, allocator)
	} else {
		du2_out = du2
	}

	if ipiv == nil || fact == .NotFactored {
		ipiv_out = builtin.make([]i32, n, allocator)
	} else {
		ipiv_out = ipiv
	}

	// Allocate error bounds (real for complex matrices)
	ferr = builtin.make([]f32, nrhs, allocator)
	berr = builtin.make([]f32, nrhs, allocator)

	// Allocate workspace
	work := builtin.make([]complex64, 2 * n, allocator)
	defer builtin.delete(work)

	rwork := builtin.make([]f32, n, allocator)
	defer builtin.delete(rwork)

	lapack.cgtsvx_(
		fact_c,
		trans_c,
		&n,
		&nrhs,
		raw_data(dl),
		raw_data(d),
		raw_data(du),
		raw_data(dlf),
		raw_data(df),
		raw_data(duf),
		raw_data(du2_out),
		raw_data(ipiv_out),
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
	)

	return X, rcond, ferr, berr, tri_factored_out, du2_out, ipiv_out, info
}

m_solve_tridiagonal_expert_c128 :: proc(
	tri: ^Matrix(complex128),
	B: ^Matrix(complex128),
	fact: FactorizationMode = .NotFactored,
	transpose: bool = false,
	conjugate_transpose: bool = false,
	tri_factored: ^Matrix(complex128) = nil,
	du2: []complex128 = nil,
	ipiv: []i32 = nil,
	allocator := context.allocator,
) -> (
	X: Matrix(complex128),
	rcond: f64,
	ferr, berr: []f64,
	tri_factored_out: Matrix(complex128),
	du2_out: []complex128,
	ipiv_out: []i32,
	info: Info,
) {
	assert(tri.format == .Tridiagonal, "Matrix must be tridiagonal")
	n := Blas_Int(tri.rows)
	nrhs := Blas_Int(B.cols)
	ldb := Blas_Int(B.ld)

	// Set fact and trans parameters
	fact_c := fact == .Factored ? cstring("F") : cstring("N")
	trans_c: cstring
	if conjugate_transpose {
		trans_c = "C"
	} else if transpose {
		trans_c = "T"
	} else {
		trans_c = "N"
	}

	// Get original diagonal arrays
	dl := get_tridiagonal_dl(tri)
	d := get_tridiagonal_d(tri)
	du := get_tridiagonal_du(tri)

	// Allocate solution matrix
	X = make_matrix(complex128, int(n), int(nrhs), allocator)
	ldx := Blas_Int(X.ld)

	// Allocate or use provided factored matrix
	if tri_factored == nil || fact == .NotFactored {
		tri_factored_out = make_tridiagonal_matrix(complex128, int(n), allocator)
	} else {
		tri_factored_out = tri_factored^
	}
	dlf := get_tridiagonal_dl(&tri_factored_out)
	df := get_tridiagonal_d(&tri_factored_out)
	duf := get_tridiagonal_du(&tri_factored_out)

	// Allocate or use provided du2 and ipiv
	if du2 == nil || fact == .NotFactored {
		du2_out = builtin.make([]complex128, n - 2, allocator)
	} else {
		du2_out = du2
	}

	if ipiv == nil || fact == .NotFactored {
		ipiv_out = builtin.make([]i32, n, allocator)
	} else {
		ipiv_out = ipiv
	}

	// Allocate error bounds (real for complex matrices)
	ferr = builtin.make([]f64, nrhs, allocator)
	berr = builtin.make([]f64, nrhs, allocator)

	// Allocate workspace
	work := builtin.make([]complex128, 2 * n, allocator)
	defer builtin.delete(work)

	rwork := builtin.make([]f64, n, allocator)
	defer builtin.delete(rwork)

	lapack.zgtsvx_(
		fact_c,
		trans_c,
		&n,
		&nrhs,
		raw_data(dl),
		raw_data(d),
		raw_data(du),
		raw_data(dlf),
		raw_data(df),
		raw_data(duf),
		raw_data(du2_out),
		raw_data(ipiv_out),
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
	)

	return X, rcond, ferr, berr, tri_factored_out, du2_out, ipiv_out, info
}

// ===================================================================================
// TRIDIAGONAL FACTORIZATION
// ===================================================================================

// LU factorization of a tridiagonal matrix
m_factor_tridiagonal :: proc {
	m_factor_tridiagonal_real,
	m_factor_tridiagonal_c64,
	m_factor_tridiagonal_c128,
}

m_factor_tridiagonal_real :: proc(
	tri: ^Matrix($T),
	allocator := context.allocator,
) -> (
	tri_factored: Matrix(T),
	du2: []T,
	ipiv: []i32,
	info: Info,
) where T == f32 ||
	T == f64 {
	assert(tri.format == .Tridiagonal, "Matrix must be tridiagonal")
	n := Blas_Int(tri.rows)

	// Create factored matrix (copy of original)
	tri_factored = make_tridiagonal_matrix(T, int(n), allocator)
	dl_factored := get_tridiagonal_dl(&tri_factored)
	d_factored := get_tridiagonal_d(&tri_factored)
	du_factored := get_tridiagonal_du(&tri_factored)

	// Copy original data
	dl_orig := get_tridiagonal_dl(tri)
	d_orig := get_tridiagonal_d(tri)
	du_orig := get_tridiagonal_du(tri)
	copy(dl_factored, dl_orig)
	copy(d_factored, d_orig)
	copy(du_factored, du_orig)

	// Allocate additional arrays for factorization
	du2 = builtin.make([]T, n - 2, allocator)
	ipiv = builtin.make([]i32, n, allocator)

	when T == f32 {
		lapack.sgttrf_(
			&n,
			raw_data(dl_factored),
			raw_data(d_factored),
			raw_data(du_factored),
			raw_data(du2),
			raw_data(ipiv),
			&info,
		)
	} else {
		lapack.dgttrf_(
			&n,
			raw_data(dl_factored),
			raw_data(d_factored),
			raw_data(du_factored),
			raw_data(du2),
			raw_data(ipiv),
			&info,
		)
	}

	return tri_factored, du2, ipiv, info
}

m_factor_tridiagonal_c64 :: proc(
	tri: ^Matrix(complex64),
	allocator := context.allocator,
) -> (
	tri_factored: Matrix(complex64),
	du2: []complex64,
	ipiv: []i32,
	info: Info,
) {
	assert(tri.format == .Tridiagonal, "Matrix must be tridiagonal")
	n := Blas_Int(tri.rows)

	// Create factored matrix (copy of original)
	tri_factored = make_tridiagonal_matrix(complex64, int(n), allocator)
	dl_factored := get_tridiagonal_dl(&tri_factored)
	d_factored := get_tridiagonal_d(&tri_factored)
	du_factored := get_tridiagonal_du(&tri_factored)

	// Copy original data
	dl_orig := get_tridiagonal_dl(tri)
	d_orig := get_tridiagonal_d(tri)
	du_orig := get_tridiagonal_du(tri)
	copy(dl_factored, dl_orig)
	copy(d_factored, d_orig)
	copy(du_factored, du_orig)

	// Allocate additional arrays for factorization
	du2 = builtin.make([]complex64, n - 2, allocator)
	ipiv = builtin.make([]i32, n, allocator)

	lapack.cgttrf_(
		&n,
		raw_data(dl_factored),
		raw_data(d_factored),
		raw_data(du_factored),
		raw_data(du2),
		raw_data(ipiv),
		&info,
	)

	return tri_factored, du2, ipiv, info
}

m_factor_tridiagonal_c128 :: proc(
	tri: ^Matrix(complex128),
	allocator := context.allocator,
) -> (
	tri_factored: Matrix(complex128),
	du2: []complex128,
	ipiv: []i32,
	info: Info,
) {
	assert(tri.format == .Tridiagonal, "Matrix must be tridiagonal")
	n := Blas_Int(tri.rows)

	// Create factored matrix (copy of original)
	tri_factored = make_tridiagonal_matrix(complex128, int(n), allocator)
	dl_factored := get_tridiagonal_dl(&tri_factored)
	d_factored := get_tridiagonal_d(&tri_factored)
	du_factored := get_tridiagonal_du(&tri_factored)

	// Copy original data
	dl_orig := get_tridiagonal_dl(tri)
	d_orig := get_tridiagonal_d(tri)
	du_orig := get_tridiagonal_du(tri)
	copy(dl_factored, dl_orig)
	copy(d_factored, d_orig)
	copy(du_factored, du_orig)

	// Allocate additional arrays for factorization
	du2 = builtin.make([]complex128, n - 2, allocator)
	ipiv = builtin.make([]i32, n, allocator)

	lapack.zgttrf_(
		&n,
		raw_data(dl_factored),
		raw_data(d_factored),
		raw_data(du_factored),
		raw_data(du2),
		raw_data(ipiv),
		&info,
	)

	return tri_factored, du2, ipiv, info
}

// ===================================================================================
// SOLVE WITH FACTORIZATION
// ===================================================================================

// Solve using pre-computed factorization
m_solve_tridiagonal_factored :: proc {
	m_solve_tridiagonal_factored_real,
	m_solve_tridiagonal_factored_c64,
	m_solve_tridiagonal_factored_c128,
}

m_solve_tridiagonal_factored_real :: proc(
	tri_factored: ^Matrix($T),
	du2: []T,
	ipiv: []i32,
	B: ^Matrix(T),
	transpose: bool = false,
	allocator := context.allocator,
) -> (
	info: Info,
) where T == f32 ||
	T == f64 {
	assert(tri_factored.format == .Tridiagonal, "Matrix must be tridiagonal")
	n := Blas_Int(tri_factored.rows)
	nrhs := Blas_Int(B.cols)
	ldb := Blas_Int(B.ld)

	trans_c := transpose ? cstring("T") : cstring("N")

	// Get diagonal arrays
	dl := get_tridiagonal_dl(tri_factored)
	d := get_tridiagonal_d(tri_factored)
	du := get_tridiagonal_du(tri_factored)

	when T == f32 {
		lapack.sgttrs_(
			trans_c,
			&n,
			&nrhs,
			raw_data(dl),
			raw_data(d),
			raw_data(du),
			raw_data(du2),
			raw_data(ipiv),
			raw_data(B.data),
			&ldb,
			&info,
			1,
		)
	} else {
		lapack.dgttrs_(
			trans_c,
			&n,
			&nrhs,
			raw_data(dl),
			raw_data(d),
			raw_data(du),
			raw_data(du2),
			raw_data(ipiv),
			raw_data(B.data),
			&ldb,
			&info,
			1,
		)
	}

	return info
}

m_solve_tridiagonal_factored_c64 :: proc(
	tri_factored: ^Matrix(complex64),
	du2: []complex64,
	ipiv: []i32,
	B: ^Matrix(complex64),
	transpose: bool = false,
	conjugate_transpose: bool = false,
	allocator := context.allocator,
) -> (
	info: Info,
) {
	assert(tri_factored.format == .Tridiagonal, "Matrix must be tridiagonal")
	n := Blas_Int(tri_factored.rows)
	nrhs := Blas_Int(B.cols)
	ldb := Blas_Int(B.ld)

	trans_c: cstring
	if conjugate_transpose {
		trans_c = "C"
	} else if transpose {
		trans_c = "T"
	} else {
		trans_c = "N"
	}

	// Get diagonal arrays
	dl := get_tridiagonal_dl(tri_factored)
	d := get_tridiagonal_d(tri_factored)
	du := get_tridiagonal_du(tri_factored)

	lapack.cgttrs_(
		trans_c,
		&n,
		&nrhs,
		raw_data(dl),
		raw_data(d),
		raw_data(du),
		raw_data(du2),
		raw_data(ipiv),
		raw_data(B.data),
		&ldb,
		&info,
		1,
	)

	return info
}

m_solve_tridiagonal_factored_c128 :: proc(
	tri_factored: ^Matrix(complex128),
	du2: []complex128,
	ipiv: []i32,
	B: ^Matrix(complex128),
	transpose: bool = false,
	conjugate_transpose: bool = false,
	allocator := context.allocator,
) -> (
	info: Info,
) {
	assert(tri_factored.format == .Tridiagonal, "Matrix must be tridiagonal")
	n := Blas_Int(tri_factored.rows)
	nrhs := Blas_Int(B.cols)
	ldb := Blas_Int(B.ld)

	trans_c: cstring
	if conjugate_transpose {
		trans_c = "C"
	} else if transpose {
		trans_c = "T"
	} else {
		trans_c = "N"
	}

	// Get diagonal arrays
	dl := get_tridiagonal_dl(tri_factored)
	d := get_tridiagonal_d(tri_factored)
	du := get_tridiagonal_du(tri_factored)

	lapack.zgttrs_(
		trans_c,
		&n,
		&nrhs,
		raw_data(dl),
		raw_data(d),
		raw_data(du),
		raw_data(du2),
		raw_data(ipiv),
		raw_data(B.data),
		&ldb,
		&info,
		1,
	)

	return info
}
