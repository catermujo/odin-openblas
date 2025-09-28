package openblas

import lapack "./f77"
import "base:builtin"
import "base:intrinsics"

// ===================================================================================
// MATRIX FACTORIZATIONS
// Decompose matrices for solving systems, least squares, and eigenproblems
// ===================================================================================

// ===================================================================================
// BIDIAGONAL REDUCTION
// Reduce general matrix to bidiagonal form for SVD computation
// ===================================================================================

// Reduce general matrix to bidiagonal form using Householder reflections
// A = Q * B * P^H where B is bidiagonal
m_bidiagonalize :: proc {
	m_bidiagonalize_real,
	m_bidiagonalize_c64,
	m_bidiagonalize_c128,
}

m_bidiagonalize_real :: proc(
	A: ^Matrix($T), // General matrix (overwritten with bidiagonal form)
	allocator := context.allocator,
) -> (
	D: []T,
	E: []T,
	tauq: []T,// Diagonal elements of B
	taup: []T,// Off-diagonal elements of B
	info: Info, // Scalar factors for Q// Scalar factors for P
) where T == f32 || T == f64 {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	min_mn := min(m, n)

	// Allocate output arrays
	D = builtin.make([]T, min_mn, allocator)
	E = builtin.make([]T, min_mn - 1, allocator)
	tauq = builtin.make([]T, min_mn, allocator)
	taup = builtin.make([]T, min_mn, allocator)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: T

	when T == f32 {
		lapack.sgebrd_(
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(D),
			raw_data(E),
			raw_data(tauq),
			raw_data(taup),
			&work_query,
			&lwork,
			&info,
		)

		// Allocate workspace
		lwork = Blas_Int(work_query)
		work := builtin.make([]T, lwork, allocator)
		defer builtin.delete(work)

		// Perform reduction
		lapack.sgebrd_(
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(D),
			raw_data(E),
			raw_data(tauq),
			raw_data(taup),
			raw_data(work),
			&lwork,
			&info,
		)
	} else when T == f64 {
		lapack.dgebrd_(
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(D),
			raw_data(E),
			raw_data(tauq),
			raw_data(taup),
			&work_query,
			&lwork,
			&info,
		)

		// Allocate workspace
		lwork = Blas_Int(work_query)
		work := builtin.make([]T, lwork, allocator)
		defer builtin.delete(work)

		// Perform reduction
		lapack.dgebrd_(
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(D),
			raw_data(E),
			raw_data(tauq),
			raw_data(taup),
			raw_data(work),
			&lwork,
			&info,
		)
	}

	return D, E, tauq, taup, info
}

m_bidiagonalize_c64 :: proc(
	A: ^Matrix(complex64),
	allocator := context.allocator,
) -> (
	D: []f32,
	E: []f32,
	tauq: []complex64,// Real diagonal elements
	taup: []complex64,// Real off-diagonal elements
	info: Info, // Scalar factors for Q// Scalar factors for P
) {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	min_mn := min(m, n)

	// Allocate output arrays
	D = builtin.make([]f32, min_mn, allocator)
	E = builtin.make([]f32, min_mn - 1, allocator)
	tauq = builtin.make([]complex64, min_mn, allocator)
	taup = builtin.make([]complex64, min_mn, allocator)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: complex64

	lapack.cgebrd_(
		&m,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(D),
		raw_data(E),
		raw_data(tauq),
		raw_data(taup),
		&work_query,
		&lwork,
		&info,
	)

	// Allocate workspace
	lwork = Blas_Int(real(work_query))
	work := builtin.make([]complex64, lwork, allocator)
	defer builtin.delete(work)

	// Perform reduction
	lapack.cgebrd_(
		&m,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(D),
		raw_data(E),
		raw_data(tauq),
		raw_data(taup),
		raw_data(work),
		&lwork,
		&info,
	)

	return D, E, tauq, taup, info
}

m_bidiagonalize_c128 :: proc(
	A: ^Matrix(complex128),
	allocator := context.allocator,
) -> (
	D: []f64,
	E: []f64,
	tauq: []complex128,// Real diagonal elements
	taup: []complex128,// Real off-diagonal elements
	info: Info, // Scalar factors for Q// Scalar factors for P
) {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	min_mn := min(m, n)

	// Allocate output arrays
	D = builtin.make([]f64, min_mn, allocator)
	E = builtin.make([]f64, min_mn - 1, allocator)
	tauq = builtin.make([]complex128, min_mn, allocator)
	taup = builtin.make([]complex128, min_mn, allocator)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: complex128

	lapack.zgebrd_(
		&m,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(D),
		raw_data(E),
		raw_data(tauq),
		raw_data(taup),
		&work_query,
		&lwork,
		&info,
	)

	// Allocate workspace
	lwork = Blas_Int(real(work_query))
	work := builtin.make([]complex128, lwork, allocator)
	defer builtin.delete(work)

	// Perform reduction
	lapack.zgebrd_(
		&m,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(D),
		raw_data(E),
		raw_data(tauq),
		raw_data(taup),
		raw_data(work),
		&lwork,
		&info,
	)

	return D, E, tauq, taup, info
}

// ===================================================================================
// LQ FACTORIZATION
// Decompose matrix A = L*Q where L is lower triangular and Q is orthogonal/unitary
// ===================================================================================

// LQ factorization using blocked algorithm with T factor
// More efficient than traditional LQ for applying Q
m_lq_blocked :: proc {
	m_lq_blocked_real,
	m_lq_blocked_c64,
	m_lq_blocked_c128,
}

m_lq_blocked_real :: proc(
	A: ^Matrix($T),
	allocator := context.allocator,
) -> (
	T_factor: []T,
	info: Info, // Block reflector factors
) where T == f32 || T == f64 {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)

	// Query for optimal T size and workspace
	tsize := Blas_Int(-1)
	lwork := Blas_Int(-1)
	tsize_query: T
	work_query: T

	when T == f32 {
		lapack.sgelq_(
			&m,
			&n,
			raw_data(A.data),
			&lda,
			&tsize_query,
			&tsize,
			&work_query,
			&lwork,
			&info,
		)
	} else when T == f64 {
		lapack.dgelq_(
			&m,
			&n,
			raw_data(A.data),
			&lda,
			&tsize_query,
			&tsize,
			&work_query,
			&lwork,
			&info,
		)
	}

	// Allocate T factor and workspace
	tsize = Blas_Int(tsize_query)
	lwork = Blas_Int(work_query)
	T_factor = builtin.make([]T, tsize, allocator)
	work := builtin.make([]T, lwork, allocator)
	defer builtin.delete(work)

	// Perform LQ factorization
	when T == f32 {
		lapack.sgelq_(
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(T_factor),
			&tsize,
			raw_data(work),
			&lwork,
			&info,
		)
	} else when T == f64 {
		lapack.dgelq_(
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(T_factor),
			&tsize,
			raw_data(work),
			&lwork,
			&info,
		)
	}

	return T_factor, info
}

m_lq_blocked_c64 :: proc(
	A: ^Matrix(complex64),
	allocator := context.allocator,
) -> (
	T_factor: []complex64,
	info: Info,
) {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)

	// Query for optimal T size and workspace
	tsize := Blas_Int(-1)
	lwork := Blas_Int(-1)
	tsize_query: complex64
	work_query: complex64

	lapack.cgelq_(&m, &n, raw_data(A.data), &lda, &tsize_query, &tsize, &work_query, &lwork, &info)

	// Allocate T factor and workspace
	tsize = Blas_Int(real(tsize_query))
	lwork = Blas_Int(real(work_query))
	T_factor = builtin.make([]complex64, tsize, allocator)
	work := builtin.make([]complex64, lwork, allocator)
	defer builtin.delete(work)

	// Perform LQ factorization
	lapack.cgelq_(
		&m,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(T_factor),
		&tsize,
		raw_data(work),
		&lwork,
		&info,
	)

	return T_factor, info
}

m_lq_blocked_c128 :: proc(
	A: ^Matrix(complex128),
	allocator := context.allocator,
) -> (
	T_factor: []complex128,
	info: Info,
) {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)

	// Query for optimal T size and workspace
	tsize := Blas_Int(-1)
	lwork := Blas_Int(-1)
	tsize_query: complex128
	work_query: complex128

	lapack.zgelq_(&m, &n, raw_data(A.data), &lda, &tsize_query, &tsize, &work_query, &lwork, &info)

	// Allocate T factor and workspace
	tsize = Blas_Int(real(tsize_query))
	lwork = Blas_Int(real(work_query))
	T_factor = builtin.make([]complex128, tsize, allocator)
	work := builtin.make([]complex128, lwork, allocator)
	defer builtin.delete(work)

	// Perform LQ factorization
	lapack.zgelq_(
		&m,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(T_factor),
		&tsize,
		raw_data(work),
		&lwork,
		&info,
	)

	return T_factor, info
}

// Traditional LQ factorization with optimal blocking
// A = L*Q where L is lower triangular and Q is orthogonal/unitary
m_lq :: proc {
	m_lq_real,
	m_lq_c64,
	m_lq_c128,
}

m_lq_real :: proc(
	A: ^Matrix($T),
	allocator := context.allocator,
) -> (
	tau: []T,
	info: Info, // Elementary reflectors
) where T == f32 || T == f64 {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	k := min(m, n)

	// Allocate tau array
	tau = builtin.make([]T, k, allocator)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: T

	when T == f32 {
		lapack.sgelqf_(&m, &n, raw_data(A.data), &lda, raw_data(tau), &work_query, &lwork, &info)
	} else when T == f64 {
		lapack.dgelqf_(&m, &n, raw_data(A.data), &lda, raw_data(tau), &work_query, &lwork, &info)
	}

	// Allocate workspace
	lwork = Blas_Int(work_query)
	work := builtin.make([]T, lwork, allocator)
	defer builtin.delete(work)

	// Perform LQ factorization
	when T == f32 {
		lapack.sgelqf_(
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(tau),
			raw_data(work),
			&lwork,
			&info,
		)
	} else when T == f64 {
		lapack.dgelqf_(
			&m,
			&n,
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

m_lq_c64 :: proc(
	A: ^Matrix(complex64),
	allocator := context.allocator,
) -> (
	tau: []complex64,
	info: Info,
) {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	k := min(m, n)

	// Allocate tau array
	tau = builtin.make([]complex64, k, allocator)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: complex64

	lapack.cgelqf_(&m, &n, raw_data(A.data), &lda, raw_data(tau), &work_query, &lwork, &info)

	// Allocate workspace
	lwork = Blas_Int(real(work_query))
	work := builtin.make([]complex64, lwork, allocator)
	defer builtin.delete(work)

	// Perform LQ factorization
	lapack.cgelqf_(&m, &n, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &lwork, &info)

	return tau, info
}

m_lq_c128 :: proc(
	A: ^Matrix(complex128),
	allocator := context.allocator,
) -> (
	tau: []complex128,
	info: Info,
) {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	k := min(m, n)

	// Allocate tau array
	tau = builtin.make([]complex128, k, allocator)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: complex128

	lapack.zgelqf_(&m, &n, raw_data(A.data), &lda, raw_data(tau), &work_query, &lwork, &info)

	// Allocate workspace
	lwork = Blas_Int(real(work_query))
	work := builtin.make([]complex128, lwork, allocator)
	defer builtin.delete(work)

	// Perform LQ factorization
	lapack.zgelqf_(&m, &n, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &lwork, &info)

	return tau, info
}

// LQ factorization using unblocked algorithm (simpler but slower)
// For small matrices where blocking overhead isn't worth it
m_lq_unblocked :: proc {
	m_lq_unblocked_real,
	m_lq_unblocked_c64,
	m_lq_unblocked_c128,
}

m_lq_unblocked_real :: proc(
	A: ^Matrix($T),
	allocator := context.allocator,
) -> (
	tau: []T,
	info: Info, // Elementary reflectors
) where T == f32 || T == f64 {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	k := min(m, n)

	// Allocate tau array
	tau = builtin.make([]T, k, allocator)

	// Allocate workspace
	work := builtin.make([]T, max(1, m), allocator)
	defer builtin.delete(work)

	when T == f32 {
		lapack.sgelq2_(&m, &n, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &info)
	} else when T == f64 {
		lapack.dgelq2_(&m, &n, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &info)
	}

	return tau, info
}

m_lq_unblocked_c64 :: proc(
	A: ^Matrix(complex64),
	allocator := context.allocator,
) -> (
	tau: []complex64,
	info: Info,
) {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	k := min(m, n)

	// Allocate tau array
	tau = builtin.make([]complex64, k, allocator)

	// Allocate workspace
	work := builtin.make([]complex64, max(1, m), allocator)
	defer builtin.delete(work)

	lapack.cgelq2_(&m, &n, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &info)

	return tau, info
}

m_lq_unblocked_c128 :: proc(
	A: ^Matrix(complex128),
	allocator := context.allocator,
) -> (
	tau: []complex128,
	info: Info,
) {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	k := min(m, n)

	// Allocate tau array
	tau = builtin.make([]complex128, k, allocator)

	// Allocate workspace
	work := builtin.make([]complex128, max(1, m), allocator)
	defer builtin.delete(work)

	lapack.zgelq2_(&m, &n, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &info)

	return tau, info
}

// ===================================================================================
// APPLYING Q FROM LQ FACTORIZATION
// ===================================================================================

// Apply Q from LQ factorization using blocked algorithm
// Computes C = Q*C, Q^T*C, C*Q, or C*Q^T (or conjugate transposes for complex)
m_apply_lq :: proc {
	m_apply_lq_real,
	m_apply_lq_c64,
	m_apply_lq_c128,
}

m_apply_lq_real :: proc(
	A: ^Matrix($T), // Matrix from LQ factorization
	T_factor: []T, // Block reflector from m_lq_blocked
	C: ^Matrix(T), // Matrix to multiply
	left_multiply: bool = true, // Q*C if true, C*Q if false
	transpose: bool = false, // Use Q^T instead of Q
	allocator := context.allocator,
) -> (
	info: Info,
) where T == f32 || T == f64 {
	m := Blas_Int(C.rows)
	n := Blas_Int(C.cols)
	k := Blas_Int(min(A.rows, A.cols)) // Number of reflectors
	lda := Blas_Int(A.ld)
	ldc := Blas_Int(C.ld)
	tsize := Blas_Int(len(T_factor))

	side_c := left_multiply ? cstring("L") : cstring("R")
	trans_c := transpose ? cstring("T") : cstring("N")

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: T

	when T == f32 {
		lapack.sgemlq_(
			side_c,
			trans_c,
			&m,
			&n,
			&k,
			raw_data(A.data),
			&lda,
			raw_data(T_factor),
			&tsize,
			raw_data(C.data),
			&ldc,
			&work_query,
			&lwork,
			&info,
			1,
			1,
		)
	} else when T == f64 {
		lapack.dgemlq_(
			side_c,
			trans_c,
			&m,
			&n,
			&k,
			raw_data(A.data),
			&lda,
			raw_data(T_factor),
			&tsize,
			raw_data(C.data),
			&ldc,
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

	// Apply Q to C
	when T == f32 {
		lapack.sgemlq_(
			side_c,
			trans_c,
			&m,
			&n,
			&k,
			raw_data(A.data),
			&lda,
			raw_data(T_factor),
			&tsize,
			raw_data(C.data),
			&ldc,
			raw_data(work),
			&lwork,
			&info,
			1,
			1,
		)
	} else when T == f64 {
		lapack.dgemlq_(
			side_c,
			trans_c,
			&m,
			&n,
			&k,
			raw_data(A.data),
			&lda,
			raw_data(T_factor),
			&tsize,
			raw_data(C.data),
			&ldc,
			raw_data(work),
			&lwork,
			&info,
			1,
			1,
		)
	}

	return info
}

m_apply_lq_c64 :: proc(
	A: ^Matrix(complex64),
	T_factor: []complex64,
	C: ^Matrix(complex64),
	left_multiply: bool = true,
	transpose: bool = false, // Conjugate transpose if true
	allocator := context.allocator,
) -> (
	info: Info,
) {
	m := Blas_Int(C.rows)
	n := Blas_Int(C.cols)
	k := Blas_Int(min(A.rows, A.cols))
	lda := Blas_Int(A.ld)
	ldc := Blas_Int(C.ld)
	tsize := Blas_Int(len(T_factor))

	side_c := left_multiply ? cstring("L") : cstring("R")
	trans_c := transpose ? cstring("C") : cstring("N") // C = conjugate transpose

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: complex64

	lapack.cgemlq_(
		side_c,
		trans_c,
		&m,
		&n,
		&k,
		raw_data(A.data),
		&lda,
		raw_data(T_factor),
		&tsize,
		raw_data(C.data),
		&ldc,
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

	// Apply Q to C
	lapack.cgemlq_(
		side_c,
		trans_c,
		&m,
		&n,
		&k,
		raw_data(A.data),
		&lda,
		raw_data(T_factor),
		&tsize,
		raw_data(C.data),
		&ldc,
		raw_data(work),
		&lwork,
		&info,
		1,
		1,
	)

	return info
}

m_apply_lq_c128 :: proc(
	A: ^Matrix(complex128),
	T_factor: []complex128,
	C: ^Matrix(complex128),
	left_multiply: bool = true,
	transpose: bool = false,
	allocator := context.allocator,
) -> (
	info: Info,
) {
	m := Blas_Int(C.rows)
	n := Blas_Int(C.cols)
	k := Blas_Int(min(A.rows, A.cols))
	lda := Blas_Int(A.ld)
	ldc := Blas_Int(C.ld)
	tsize := Blas_Int(len(T_factor))

	side_c := left_multiply ? cstring("L") : cstring("R")
	trans_c := transpose ? cstring("C") : cstring("N")

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: complex128

	lapack.zgemlq_(
		side_c,
		trans_c,
		&m,
		&n,
		&k,
		raw_data(A.data),
		&lda,
		raw_data(T_factor),
		&tsize,
		raw_data(C.data),
		&ldc,
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

	// Apply Q to C
	lapack.zgemlq_(
		side_c,
		trans_c,
		&m,
		&n,
		&k,
		raw_data(A.data),
		&lda,
		raw_data(T_factor),
		&tsize,
		raw_data(C.data),
		&ldc,
		raw_data(work),
		&lwork,
		&info,
		1,
		1,
	)

	return info
}

// ===================================================================================
// APPLYING Q FROM QR FACTORIZATION
// ===================================================================================

// Apply Q from QR factorization using new blocked algorithm
// Computes C = Q*C, Q^T*C, C*Q, or C*Q^T (or conjugate transposes for complex)
m_apply_qr :: proc {
	m_apply_qr_real,
	m_apply_qr_c64,
	m_apply_qr_c128,
}

m_apply_qr_real :: proc(
	A: ^Matrix($T), // Matrix from QR factorization
	T_factor: []T, // Block reflector from QR with T factor
	C: ^Matrix(T), // Matrix to multiply
	left_multiply: bool = true, // Q*C if true, C*Q if false
	transpose: bool = false, // Use Q^T instead of Q
	allocator := context.allocator,
) -> (
	info: Info,
) where T == f32 || T == f64 {
	m := Blas_Int(C.rows)
	n := Blas_Int(C.cols)
	k := Blas_Int(min(A.rows, A.cols)) // Number of reflectors
	lda := Blas_Int(A.ld)
	ldc := Blas_Int(C.ld)
	tsize := Blas_Int(len(T_factor))

	side_c := left_multiply ? cstring("L") : cstring("R")
	trans_c := transpose ? cstring("T") : cstring("N")

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: T

	when T == f32 {
		lapack.sgemqr_(
			side_c,
			trans_c,
			&m,
			&n,
			&k,
			raw_data(A.data),
			&lda,
			raw_data(T_factor),
			&tsize,
			raw_data(C.data),
			&ldc,
			&work_query,
			&lwork,
			&info,
			1,
			1,
		)
	} else when T == f64 {
		lapack.dgemqr_(
			side_c,
			trans_c,
			&m,
			&n,
			&k,
			raw_data(A.data),
			&lda,
			raw_data(T_factor),
			&tsize,
			raw_data(C.data),
			&ldc,
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

	// Apply Q to C
	when T == f32 {
		lapack.sgemqr_(
			side_c,
			trans_c,
			&m,
			&n,
			&k,
			raw_data(A.data),
			&lda,
			raw_data(T_factor),
			&tsize,
			raw_data(C.data),
			&ldc,
			raw_data(work),
			&lwork,
			&info,
			1,
			1,
		)
	} else when T == f64 {
		lapack.dgemqr_(
			side_c,
			trans_c,
			&m,
			&n,
			&k,
			raw_data(A.data),
			&lda,
			raw_data(T_factor),
			&tsize,
			raw_data(C.data),
			&ldc,
			raw_data(work),
			&lwork,
			&info,
			1,
			1,
		)
	}

	return info
}

m_apply_qr_c64 :: proc(
	A: ^Matrix(complex64),
	T_factor: []complex64,
	C: ^Matrix(complex64),
	left_multiply: bool = true,
	transpose: bool = false, // Conjugate transpose if true
	allocator := context.allocator,
) -> (
	info: Info,
) {
	m := Blas_Int(C.rows)
	n := Blas_Int(C.cols)
	k := Blas_Int(min(A.rows, A.cols))
	lda := Blas_Int(A.ld)
	ldc := Blas_Int(C.ld)
	tsize := Blas_Int(len(T_factor))

	side_c := left_multiply ? cstring("L") : cstring("R")
	trans_c := transpose ? cstring("C") : cstring("N") // C = conjugate transpose

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: complex64

	lapack.cgemqr_(
		side_c,
		trans_c,
		&m,
		&n,
		&k,
		raw_data(A.data),
		&lda,
		raw_data(T_factor),
		&tsize,
		raw_data(C.data),
		&ldc,
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

	// Apply Q to C
	lapack.cgemqr_(
		side_c,
		trans_c,
		&m,
		&n,
		&k,
		raw_data(A.data),
		&lda,
		raw_data(T_factor),
		&tsize,
		raw_data(C.data),
		&ldc,
		raw_data(work),
		&lwork,
		&info,
		1,
		1,
	)

	return info
}

m_apply_qr_c128 :: proc(
	A: ^Matrix(complex128),
	T_factor: []complex128,
	C: ^Matrix(complex128),
	left_multiply: bool = true,
	transpose: bool = false,
	allocator := context.allocator,
) -> (
	info: Info,
) {
	m := Blas_Int(C.rows)
	n := Blas_Int(C.cols)
	k := Blas_Int(min(A.rows, A.cols))
	lda := Blas_Int(A.ld)
	ldc := Blas_Int(C.ld)
	tsize := Blas_Int(len(T_factor))

	side_c := left_multiply ? cstring("L") : cstring("R")
	trans_c := transpose ? cstring("C") : cstring("N")

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: complex128

	lapack.zgemqr_(
		side_c,
		trans_c,
		&m,
		&n,
		&k,
		raw_data(A.data),
		&lda,
		raw_data(T_factor),
		&tsize,
		raw_data(C.data),
		&ldc,
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

	// Apply Q to C
	lapack.zgemqr_(
		side_c,
		trans_c,
		&m,
		&n,
		&k,
		raw_data(A.data),
		&lda,
		raw_data(T_factor),
		&tsize,
		raw_data(C.data),
		&ldc,
		raw_data(work),
		&lwork,
		&info,
		1,
		1,
	)

	return info
}

// Apply Q from QR factorization with explicit block size
// Uses traditional blocked algorithm with triangular T factor
m_apply_qr_blocked :: proc {
	m_apply_qr_blocked_real,
	m_apply_qr_blocked_c64,
	m_apply_qr_blocked_c128,
}

m_apply_qr_blocked_real :: proc(
	V: ^Matrix($T), // Householder vectors from QR
	T_matrix: ^Matrix(T), // Triangular block reflector
	C: ^Matrix(T), // Matrix to multiply
	nb: Blas_Int, // Block size
	left_multiply: bool = true,
	transpose: bool = false,
	allocator := context.allocator,
) -> (
	info: Info,
) where T == f32 || T == f64 {
	m := Blas_Int(C.rows)
	n := Blas_Int(C.cols)
	k := Blas_Int(V.cols) // Number of reflectors
	ldv := Blas_Int(V.ld)
	ldt := Blas_Int(T_matrix.ld)
	ldc := Blas_Int(C.ld)

	side_c := left_multiply ? cstring("L") : cstring("R")
	trans_c := transpose ? cstring("T") : cstring("N")

	// Allocate workspace (size depends on side)
	work_size := left_multiply ? n * nb : m * nb
	work := builtin.make([]T, work_size, allocator)
	defer builtin.delete(work)

	when T == f32 {
		lapack.sgemqrt_(
			side_c,
			trans_c,
			&m,
			&n,
			&k,
			&nb,
			raw_data(V.data),
			&ldv,
			raw_data(T_matrix.data),
			&ldt,
			raw_data(C.data),
			&ldc,
			raw_data(work),
			&info,
			1,
			1,
		)
	} else when T == f64 {
		lapack.dgemqrt_(
			side_c,
			trans_c,
			&m,
			&n,
			&k,
			&nb,
			raw_data(V.data),
			&ldv,
			raw_data(T_matrix.data),
			&ldt,
			raw_data(C.data),
			&ldc,
			raw_data(work),
			&info,
			1,
			1,
		)
	}

	return info
}

m_apply_qr_blocked_c64 :: proc(
	V: ^Matrix(complex64),
	T_matrix: ^Matrix(complex64),
	C: ^Matrix(complex64),
	nb: Blas_Int,
	left_multiply: bool = true,
	transpose: bool = false,
	allocator := context.allocator,
) -> (
	info: Info,
) {
	m := Blas_Int(C.rows)
	n := Blas_Int(C.cols)
	k := Blas_Int(V.cols)
	ldv := Blas_Int(V.ld)
	ldt := Blas_Int(T_matrix.ld)
	ldc := Blas_Int(C.ld)

	side_c := left_multiply ? cstring("L") : cstring("R")
	trans_c := transpose ? cstring("C") : cstring("N")

	work_size := left_multiply ? n * nb : m * nb
	work := builtin.make([]complex64, work_size, allocator)
	defer builtin.delete(work)

	lapack.cgemqrt_(
		side_c,
		trans_c,
		&m,
		&n,
		&k,
		&nb,
		raw_data(V.data),
		&ldv,
		raw_data(T_matrix.data),
		&ldt,
		raw_data(C.data),
		&ldc,
		raw_data(work),
		&info,
		1,
		1,
	)

	return info
}

m_apply_qr_blocked_c128 :: proc(
	V: ^Matrix(complex128),
	T_matrix: ^Matrix(complex128),
	C: ^Matrix(complex128),
	nb: Blas_Int,
	left_multiply: bool = true,
	transpose: bool = false,
	allocator := context.allocator,
) -> (
	info: Info,
) {
	m := Blas_Int(C.rows)
	n := Blas_Int(C.cols)
	k := Blas_Int(V.cols)
	ldv := Blas_Int(V.ld)
	ldt := Blas_Int(T_matrix.ld)
	ldc := Blas_Int(C.ld)

	side_c := left_multiply ? cstring("L") : cstring("R")
	trans_c := transpose ? cstring("C") : cstring("N")

	work_size := left_multiply ? n * nb : m * nb
	work := builtin.make([]complex128, work_size, allocator)
	defer builtin.delete(work)

	lapack.zgemqrt_(
		side_c,
		trans_c,
		&m,
		&n,
		&k,
		&nb,
		raw_data(V.data),
		&ldv,
		raw_data(T_matrix.data),
		&ldt,
		raw_data(C.data),
		&ldc,
		raw_data(work),
		&info,
		1,
		1,
	)

	return info
}

// ===================================================================================
// QL FACTORIZATION
// ===================================================================================

// Compute QL factorization of a matrix (unblocked algorithm)
// A = Q * L where Q is orthogonal/unitary and L is lower triangular
m_ql_unblocked :: proc {
	m_ql_unblocked_real,
	m_ql_unblocked_c64,
	m_ql_unblocked_c128,
}

m_ql_unblocked_real :: proc(
	A: ^Matrix($T), // General matrix (overwritten with QL factorization)
	allocator := context.allocator,
) -> (
	tau: []T,
	info: Info, // Elementary reflectors
) where T == f32 || T == f64 {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	k := min(m, n)

	// Allocate tau
	tau = builtin.make([]T, k, allocator)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: T

	when T == f32 {
		lapack.sgeql2_(
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(tau),
			raw_data([]T{work_query}),
			&info,
		)
		lwork = Blas_Int(work_query)

		// Allocate workspace
		work := builtin.make([]T, lwork, allocator)
		defer builtin.delete(work)

		// Compute factorization
		lapack.sgeql2_(&m, &n, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &info)
	} else {
		lapack.dgeql2_(
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(tau),
			raw_data([]T{work_query}),
			&info,
		)
		lwork = Blas_Int(work_query)

		// Allocate workspace
		work := builtin.make([]T, lwork, allocator)
		defer builtin.delete(work)

		// Compute factorization
		lapack.dgeql2_(&m, &n, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &info)
	}

	return tau, info
}

m_ql_unblocked_c64 :: proc(
	A: ^Matrix(complex64),
	allocator := context.allocator,
) -> (
	tau: []complex64,
	info: Info,
) {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	k := min(m, n)

	// Allocate tau
	tau = builtin.make([]complex64, k, allocator)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: complex64

	lapack.cgeql2_(
		&m,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data([]complex64{work_query}),
		&info,
	)
	lwork = Blas_Int(real(work_query))

	// Allocate workspace
	work := builtin.make([]complex64, lwork, allocator)
	defer builtin.delete(work)

	// Compute factorization
	lapack.cgeql2_(&m, &n, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &info)

	return tau, info
}

m_ql_unblocked_c128 :: proc(
	A: ^Matrix(complex128),
	allocator := context.allocator,
) -> (
	tau: []complex128,
	info: Info,
) {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	k := min(m, n)

	// Allocate tau
	tau = builtin.make([]complex128, k, allocator)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: complex128

	lapack.zgeql2_(
		&m,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data([]complex128{work_query}),
		&info,
	)
	lwork = Blas_Int(real(work_query))

	// Allocate workspace
	work := builtin.make([]complex128, lwork, allocator)
	defer builtin.delete(work)

	// Compute factorization
	lapack.zgeql2_(&m, &n, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &info)

	return tau, info
}

// Compute QL factorization using blocked algorithm
// More efficient for larger matrices
m_ql :: proc {
	m_ql_real,
	m_ql_c64,
	m_ql_c128,
}

m_ql_real :: proc(
	A: ^Matrix($T), // General matrix (overwritten with QL factorization)
	allocator := context.allocator,
) -> (
	tau: []T,
	info: Info, // Elementary reflectors
) where T == f32 || T == f64 {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	k := min(m, n)

	// Allocate tau
	tau = builtin.make([]T, k, allocator)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: T

	when T == f32 {
		lapack.sgeqlf_(&m, &n, raw_data(A.data), &lda, raw_data(tau), &work_query, &lwork, &info)
		lwork = Blas_Int(work_query)

		// Allocate workspace
		work := builtin.make([]T, lwork, allocator)
		defer builtin.delete(work)

		// Compute factorization
		lapack.sgeqlf_(
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(tau),
			raw_data(work),
			&lwork,
			&info,
		)
	} else {
		lapack.dgeqlf_(&m, &n, raw_data(A.data), &lda, raw_data(tau), &work_query, &lwork, &info)
		lwork = Blas_Int(work_query)

		// Allocate workspace
		work := builtin.make([]T, lwork, allocator)
		defer builtin.delete(work)

		// Compute factorization
		lapack.dgeqlf_(
			&m,
			&n,
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

m_ql_c64 :: proc(
	A: ^Matrix(complex64),
	allocator := context.allocator,
) -> (
	tau: []complex64,
	info: Info,
) {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	k := min(m, n)

	// Allocate tau
	tau = builtin.make([]complex64, k, allocator)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: complex64

	lapack.cgeqlf_(&m, &n, raw_data(A.data), &lda, raw_data(tau), &work_query, &lwork, &info)
	lwork = Blas_Int(real(work_query))

	// Allocate workspace
	work := builtin.make([]complex64, lwork, allocator)
	defer builtin.delete(work)

	// Compute factorization
	lapack.cgeqlf_(&m, &n, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &lwork, &info)

	return tau, info
}

m_ql_c128 :: proc(
	A: ^Matrix(complex128),
	allocator := context.allocator,
) -> (
	tau: []complex128,
	info: Info,
) {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	k := min(m, n)

	// Allocate tau
	tau = builtin.make([]complex128, k, allocator)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: complex128

	lapack.zgeqlf_(&m, &n, raw_data(A.data), &lda, raw_data(tau), &work_query, &lwork, &info)
	lwork = Blas_Int(real(work_query))

	// Allocate workspace
	work := builtin.make([]complex128, lwork, allocator)
	defer builtin.delete(work)

	// Compute factorization
	lapack.zgeqlf_(&m, &n, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &lwork, &info)

	return tau, info
}

// ===================================================================================
// QR FACTORIZATION WITH COLUMN PIVOTING
// ===================================================================================

// Compute QR factorization with column pivoting
// A*P = Q*R where P is a permutation matrix for numerical stability
m_qr_pivot :: proc {
	m_qr_pivot_real,
	m_qr_pivot_c64,
	m_qr_pivot_c128,
}

m_qr_pivot_real :: proc(
	A: ^Matrix($T), // General matrix (overwritten with QR factorization)
	jpvt: []i32, // Column pivot indices (input/output)
	allocator := context.allocator,
) -> (
	tau: []T,
	info: Info, // Elementary reflectors
) where T == f32 || T == f64 {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	k := min(m, n)

	// Allocate tau
	tau = builtin.make([]T, k, allocator)

	// Ensure jpvt is the right size
	assert(len(jpvt) >= int(n), "jpvt must have at least n elements")

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: T

	when T == f32 {
		lapack.sgeqp3_(
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(jpvt),
			raw_data(tau),
			&work_query,
			&lwork,
			&info,
		)
		lwork = Blas_Int(work_query)

		// Allocate workspace
		work := builtin.make([]T, lwork, allocator)
		defer builtin.delete(work)

		// Compute factorization
		lapack.sgeqp3_(
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(jpvt),
			raw_data(tau),
			raw_data(work),
			&lwork,
			&info,
		)
	} else {
		lapack.dgeqp3_(
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(jpvt),
			raw_data(tau),
			&work_query,
			&lwork,
			&info,
		)
		lwork = Blas_Int(work_query)

		// Allocate workspace
		work := builtin.make([]T, lwork, allocator)
		defer builtin.delete(work)

		// Compute factorization
		lapack.dgeqp3_(
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(jpvt),
			raw_data(tau),
			raw_data(work),
			&lwork,
			&info,
		)
	}

	return tau, info
}

m_qr_pivot_c64 :: proc(
	A: ^Matrix(complex64),
	jpvt: []i32,
	allocator := context.allocator,
) -> (
	tau: []complex64,
	info: Info,
) {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	k := min(m, n)

	// Allocate tau
	tau = builtin.make([]complex64, k, allocator)

	// Ensure jpvt is the right size
	assert(len(jpvt) >= int(n), "jpvt must have at least n elements")

	// Allocate real workspace
	rwork := builtin.make([]f32, 2 * n, allocator)
	defer builtin.delete(rwork)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: complex64

	lapack.cgeqp3_(
		&m,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(jpvt),
		raw_data(tau),
		&work_query,
		&lwork,
		raw_data(rwork),
		&info,
	)
	lwork = Blas_Int(real(work_query))

	// Allocate workspace
	work := builtin.make([]complex64, lwork, allocator)
	defer builtin.delete(work)

	// Compute factorization
	lapack.cgeqp3_(
		&m,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(jpvt),
		raw_data(tau),
		raw_data(work),
		&lwork,
		raw_data(rwork),
		&info,
	)

	return tau, info
}

m_qr_pivot_c128 :: proc(
	A: ^Matrix(complex128),
	jpvt: []i32,
	allocator := context.allocator,
) -> (
	tau: []complex128,
	info: Info,
) {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	k := min(m, n)

	// Allocate tau
	tau = builtin.make([]complex128, k, allocator)

	// Ensure jpvt is the right size
	assert(len(jpvt) >= int(n), "jpvt must have at least n elements")

	// Allocate real workspace
	rwork := builtin.make([]f64, 2 * n, allocator)
	defer builtin.delete(rwork)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: complex128

	lapack.zgeqp3_(
		&m,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(jpvt),
		raw_data(tau),
		&work_query,
		&lwork,
		raw_data(rwork),
		&info,
	)
	lwork = Blas_Int(real(work_query))

	// Allocate workspace
	work := builtin.make([]complex128, lwork, allocator)
	defer builtin.delete(work)

	// Compute factorization
	lapack.zgeqp3_(
		&m,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(jpvt),
		raw_data(tau),
		raw_data(work),
		&lwork,
		raw_data(rwork),
		&info,
	)

	return tau, info
}

// ===================================================================================
// QR FACTORIZATION
// ===================================================================================

// Compute QR factorization (unblocked algorithm)
// A = Q * R where Q is orthogonal/unitary and R is upper triangular
m_qr_unblocked :: proc {
	m_qr_unblocked_real,
	m_qr_unblocked_c64,
	m_qr_unblocked_c128,
}

m_qr_unblocked_real :: proc(
	A: ^Matrix($T), // General matrix (overwritten with QR factorization)
	allocator := context.allocator,
) -> (
	tau: []T,
	info: Info, // Elementary reflectors
) where T == f32 || T == f64 {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	k := min(m, n)

	// Allocate tau
	tau = builtin.make([]T, k, allocator)

	// Allocate workspace (geqr2 requires workspace of size n)
	work := builtin.make([]T, n, allocator)
	defer builtin.delete(work)

	when T == f32 {
		lapack.sgeqr2_(&m, &n, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &info)
	} else {
		lapack.dgeqr2_(&m, &n, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &info)
	}

	return tau, info
}

m_qr_unblocked_c64 :: proc(
	A: ^Matrix(complex64),
	allocator := context.allocator,
) -> (
	tau: []complex64,
	info: Info,
) {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	k := min(m, n)

	// Allocate tau
	tau = builtin.make([]complex64, k, allocator)

	// Allocate workspace
	work := builtin.make([]complex64, n, allocator)
	defer builtin.delete(work)

	lapack.cgeqr2_(&m, &n, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &info)

	return tau, info
}

m_qr_unblocked_c128 :: proc(
	A: ^Matrix(complex128),
	allocator := context.allocator,
) -> (
	tau: []complex128,
	info: Info,
) {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	k := min(m, n)

	// Allocate tau
	tau = builtin.make([]complex128, k, allocator)

	// Allocate workspace
	work := builtin.make([]complex128, n, allocator)
	defer builtin.delete(work)

	lapack.zgeqr2_(&m, &n, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &info)

	return tau, info
}

// Compute QR factorization with compact WY representation
// Modern blocked algorithm that returns T matrix for efficient Q application
m_qr_blocked :: proc {
	m_qr_blocked_real,
	m_qr_blocked_c64,
	m_qr_blocked_c128,
}

m_qr_blocked_real :: proc(
	A: ^Matrix($T), // General matrix (overwritten with QR factorization)
	allocator := context.allocator,
) -> (
	T_matrix: Matrix(T),
	info: Info, // Compact representation for Q
) where T == f32 || T == f64 {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)

	// Query for T size
	tsize := Blas_Int(-1)
	t_query: T

	// Query for workspace
	lwork := Blas_Int(-1)
	work_query: T

	when T == f32 {
		lapack.sgeqr_(&m, &n, raw_data(A.data), &lda, &t_query, &tsize, &work_query, &lwork, &info)
		tsize = Blas_Int(t_query)
		lwork = Blas_Int(work_query)
	} else {
		lapack.dgeqr_(&m, &n, raw_data(A.data), &lda, &t_query, &tsize, &work_query, &lwork, &info)
		tsize = Blas_Int(t_query)
		lwork = Blas_Int(work_query)
	}

	// Allocate T matrix data
	t_data := builtin.make([]T, tsize, allocator)
	T_matrix = Matrix(T) {
		data   = t_data,
		rows   = int(tsize),
		cols   = 1,
		ld     = int(tsize),
		format = .General,
	}

	// Allocate workspace
	work := builtin.make([]T, lwork, allocator)
	defer builtin.delete(work)

	// Compute factorization
	when T == f32 {
		lapack.sgeqr_(
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(T_matrix.data),
			&tsize,
			raw_data(work),
			&lwork,
			&info,
		)
	} else {
		lapack.dgeqr_(
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(T_matrix.data),
			&tsize,
			raw_data(work),
			&lwork,
			&info,
		)
	}

	return T_matrix, info
}

m_qr_blocked_c64 :: proc(
	A: ^Matrix(complex64),
	allocator := context.allocator,
) -> (
	T_matrix: Matrix(complex64),
	info: Info,
) {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)

	// Query for T size
	tsize := Blas_Int(-1)
	t_query: complex64

	// Query for workspace
	lwork := Blas_Int(-1)
	work_query: complex64

	lapack.cgeqr_(&m, &n, raw_data(A.data), &lda, &t_query, &tsize, &work_query, &lwork, &info)
	tsize = Blas_Int(real(t_query))
	lwork = Blas_Int(real(work_query))

	// Allocate T matrix data
	t_data := builtin.make([]complex64, tsize, allocator)
	T_matrix = Matrix(complex64) {
		data   = t_data,
		rows   = int(tsize),
		cols   = 1,
		ld     = int(tsize),
		format = .General,
	}

	// Allocate workspace
	work := builtin.make([]complex64, lwork, allocator)
	defer builtin.delete(work)

	// Compute factorization
	lapack.cgeqr_(
		&m,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(T_matrix.data),
		&tsize,
		raw_data(work),
		&lwork,
		&info,
	)

	return T_matrix, info
}

m_qr_blocked_c128 :: proc(
	A: ^Matrix(complex128),
	allocator := context.allocator,
) -> (
	T_matrix: Matrix(complex128),
	info: Info,
) {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)

	// Query for T size
	tsize := Blas_Int(-1)
	t_query: complex128

	// Query for workspace
	lwork := Blas_Int(-1)
	work_query: complex128

	lapack.zgeqr_(&m, &n, raw_data(A.data), &lda, &t_query, &tsize, &work_query, &lwork, &info)
	tsize = Blas_Int(real(t_query))
	lwork = Blas_Int(real(work_query))

	// Allocate T matrix data
	t_data := builtin.make([]complex128, tsize, allocator)
	T_matrix = Matrix(complex128) {
		data   = t_data,
		rows   = int(tsize),
		cols   = 1,
		ld     = int(tsize),
		format = .General,
	}

	// Allocate workspace
	work := builtin.make([]complex128, lwork, allocator)
	defer builtin.delete(work)

	// Compute factorization
	lapack.zgeqr_(
		&m,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(T_matrix.data),
		&tsize,
		raw_data(work),
		&lwork,
		&info,
	)

	return T_matrix, info
}

// Compute QR factorization (traditional blocked algorithm)
// A = Q * R where Q is orthogonal/unitary and R is upper triangular
m_qr :: proc {
	m_qr_real,
	m_qr_c64,
	m_qr_c128,
}

m_qr_real :: proc(
	A: ^Matrix($T), // General matrix (overwritten with QR factorization)
	allocator := context.allocator,
) -> (
	tau: []T,
	info: Info, // Elementary reflectors
) where T == f32 || T == f64 {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	k := min(m, n)

	// Allocate tau
	tau = builtin.make([]T, k, allocator)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: T

	when T == f32 {
		lapack.sgeqrf_(&m, &n, raw_data(A.data), &lda, raw_data(tau), &work_query, &lwork, &info)
		lwork = Blas_Int(work_query)

		// Allocate workspace
		work := builtin.make([]T, lwork, allocator)
		defer builtin.delete(work)

		// Compute factorization
		lapack.sgeqrf_(
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(tau),
			raw_data(work),
			&lwork,
			&info,
		)
	} else {
		lapack.dgeqrf_(&m, &n, raw_data(A.data), &lda, raw_data(tau), &work_query, &lwork, &info)
		lwork = Blas_Int(work_query)

		// Allocate workspace
		work := builtin.make([]T, lwork, allocator)
		defer builtin.delete(work)

		// Compute factorization
		lapack.dgeqrf_(
			&m,
			&n,
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

m_qr_c64 :: proc(
	A: ^Matrix(complex64),
	allocator := context.allocator,
) -> (
	tau: []complex64,
	info: Info,
) {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	k := min(m, n)

	// Allocate tau
	tau = builtin.make([]complex64, k, allocator)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: complex64

	lapack.cgeqrf_(&m, &n, raw_data(A.data), &lda, raw_data(tau), &work_query, &lwork, &info)
	lwork = Blas_Int(real(work_query))

	// Allocate workspace
	work := builtin.make([]complex64, lwork, allocator)
	defer builtin.delete(work)

	// Compute factorization
	lapack.cgeqrf_(&m, &n, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &lwork, &info)

	return tau, info
}

m_qr_c128 :: proc(
	A: ^Matrix(complex128),
	allocator := context.allocator,
) -> (
	tau: []complex128,
	info: Info,
) {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	k := min(m, n)

	// Allocate tau
	tau = builtin.make([]complex128, k, allocator)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: complex128

	lapack.zgeqrf_(&m, &n, raw_data(A.data), &lda, raw_data(tau), &work_query, &lwork, &info)
	lwork = Blas_Int(real(work_query))

	// Allocate workspace
	work := builtin.make([]complex128, lwork, allocator)
	defer builtin.delete(work)

	// Compute factorization
	lapack.zgeqrf_(&m, &n, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &lwork, &info)

	return tau, info
}

// Compute QR factorization with non-negative diagonal elements of R
// Ensures diagonal elements of R are non-negative for uniqueness
m_qr_nonneg :: proc {
	m_qr_nonneg_real,
	m_qr_nonneg_c64,
	m_qr_nonneg_c128,
}

m_qr_nonneg_real :: proc(
	A: ^Matrix($T), // General matrix (overwritten with QR factorization)
	allocator := context.allocator,
) -> (
	tau: []T,
	info: Info, // Elementary reflectors
) where T == f32 || T == f64 {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	k := min(m, n)

	// Allocate tau
	tau = builtin.make([]T, k, allocator)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: T

	when T == f32 {
		lapack.sgeqrfp_(&m, &n, raw_data(A.data), &lda, raw_data(tau), &work_query, &lwork, &info)
		lwork = Blas_Int(work_query)

		// Allocate workspace
		work := builtin.make([]T, lwork, allocator)
		defer builtin.delete(work)

		// Compute factorization
		lapack.sgeqrfp_(
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(tau),
			raw_data(work),
			&lwork,
			&info,
		)
	} else {
		lapack.dgeqrfp_(&m, &n, raw_data(A.data), &lda, raw_data(tau), &work_query, &lwork, &info)
		lwork = Blas_Int(work_query)

		// Allocate workspace
		work := builtin.make([]T, lwork, allocator)
		defer builtin.delete(work)

		// Compute factorization
		lapack.dgeqrfp_(
			&m,
			&n,
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

m_qr_nonneg_c64 :: proc(
	A: ^Matrix(complex64),
	allocator := context.allocator,
) -> (
	tau: []complex64,
	info: Info,
) {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	k := min(m, n)

	// Allocate tau
	tau = builtin.make([]complex64, k, allocator)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: complex64

	lapack.cgeqrfp_(&m, &n, raw_data(A.data), &lda, raw_data(tau), &work_query, &lwork, &info)
	lwork = Blas_Int(real(work_query))

	// Allocate workspace
	work := builtin.make([]complex64, lwork, allocator)
	defer builtin.delete(work)

	// Compute factorization
	lapack.cgeqrfp_(&m, &n, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &lwork, &info)

	return tau, info
}

m_qr_nonneg_c128 :: proc(
	A: ^Matrix(complex128),
	allocator := context.allocator,
) -> (
	tau: []complex128,
	info: Info,
) {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	k := min(m, n)

	// Allocate tau
	tau = builtin.make([]complex128, k, allocator)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: complex128

	lapack.zgeqrfp_(&m, &n, raw_data(A.data), &lda, raw_data(tau), &work_query, &lwork, &info)
	lwork = Blas_Int(real(work_query))

	// Allocate workspace
	work := builtin.make([]complex128, lwork, allocator)
	defer builtin.delete(work)

	// Compute factorization
	lapack.zgeqrfp_(&m, &n, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &lwork, &info)

	return tau, info
}

// ===================================================================================
// QR FACTORIZATION WITH COMPACT WY REPRESENTATION
// ===================================================================================

// Compute QR factorization with explicit T factor (blocked)
// Uses compact WY representation Q = I - V*T*V^H
m_qr_compact :: proc {
	m_qr_compact_real,
	m_qr_compact_c64,
	m_qr_compact_c128,
}

m_qr_compact_real :: proc(
	A: ^Matrix($T), // General matrix (overwritten with QR factorization)
	nb: Blas_Int, // Block size
	allocator := context.allocator,
) -> (
	T_matrix: Matrix(T),
	info: Info, // T factor for compact WY representation
) where T == f32 || T == f64 {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	k := min(m, n)

	// Allocate T matrix (nb x k)
	t_data := builtin.make([]T, nb * k, allocator)
	T_matrix = Matrix(T) {
		data   = t_data,
		rows   = int(nb),
		cols   = int(k),
		ld     = int(nb),
		format = .General,
	}
	ldt := Blas_Int(nb)

	// Allocate workspace
	work := builtin.make([]T, nb * n, allocator)
	defer builtin.delete(work)

	when T == f32 {
		lapack.sgeqrt_(
			&m,
			&n,
			&nb,
			raw_data(A.data),
			&lda,
			raw_data(T_matrix.data),
			&ldt,
			raw_data(work),
			&info,
		)
	} else {
		lapack.dgeqrt_(
			&m,
			&n,
			&nb,
			raw_data(A.data),
			&lda,
			raw_data(T_matrix.data),
			&ldt,
			raw_data(work),
			&info,
		)
	}

	return T_matrix, info
}

m_qr_compact_c64 :: proc(
	A: ^Matrix(complex64),
	nb: Blas_Int,
	allocator := context.allocator,
) -> (
	T_matrix: Matrix(complex64),
	info: Info,
) {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	k := min(m, n)

	// Allocate T matrix
	t_data := builtin.make([]complex64, nb * k, allocator)
	T_matrix = Matrix(complex64) {
		data   = t_data,
		rows   = int(nb),
		cols   = int(k),
		ld     = int(nb),
		format = .General,
	}
	ldt := Blas_Int(nb)

	// Allocate workspace
	work := builtin.make([]complex64, nb * n, allocator)
	defer builtin.delete(work)

	lapack.cgeqrt_(
		&m,
		&n,
		&nb,
		raw_data(A.data),
		&lda,
		raw_data(T_matrix.data),
		&ldt,
		raw_data(work),
		&info,
	)

	return T_matrix, info
}

m_qr_compact_c128 :: proc(
	A: ^Matrix(complex128),
	nb: Blas_Int,
	allocator := context.allocator,
) -> (
	T_matrix: Matrix(complex128),
	info: Info,
) {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	k := min(m, n)

	// Allocate T matrix
	t_data := builtin.make([]complex128, nb * k, allocator)
	T_matrix = Matrix(complex128) {
		data   = t_data,
		rows   = int(nb),
		cols   = int(k),
		ld     = int(nb),
		format = .General,
	}
	ldt := Blas_Int(nb)

	// Allocate workspace
	work := builtin.make([]complex128, nb * n, allocator)
	defer builtin.delete(work)

	lapack.zgeqrt_(
		&m,
		&n,
		&nb,
		raw_data(A.data),
		&lda,
		raw_data(T_matrix.data),
		&ldt,
		raw_data(work),
		&info,
	)

	return T_matrix, info
}

// Compute QR factorization with T factor (unblocked)
// Unblocked version for small matrices
m_qr_compact_unblocked :: proc {
	m_qr_compact_unblocked_real,
	m_qr_compact_unblocked_c64,
	m_qr_compact_unblocked_c128,
}

m_qr_compact_unblocked_real :: proc(
	A: ^Matrix($T), // General matrix (overwritten with QR factorization)
	allocator := context.allocator,
) -> (
	T_matrix: Matrix(T),
	info: Info, // T factor for compact WY representation
) where T == f32 || T == f64 {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	k := min(m, n)

	// Allocate T matrix (k x k)
	t_data := builtin.make([]T, k * k, allocator)
	T_matrix = Matrix(T) {
		data   = t_data,
		rows   = int(k),
		cols   = int(k),
		ld     = int(k),
		format = .General,
	}
	ldt := k

	when T == f32 {
		lapack.sgeqrt2_(&m, &n, raw_data(A.data), &lda, raw_data(T_matrix.data), &ldt, &info)
	} else {
		lapack.dgeqrt2_(&m, &n, raw_data(A.data), &lda, raw_data(T_matrix.data), &ldt, &info)
	}

	return T_matrix, info
}

m_qr_compact_unblocked_c64 :: proc(
	A: ^Matrix(complex64),
	allocator := context.allocator,
) -> (
	T_matrix: Matrix(complex64),
	info: Info,
) {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	k := min(m, n)

	// Allocate T matrix
	t_data := builtin.make([]complex64, k * k, allocator)
	T_matrix = Matrix(complex64) {
		data   = t_data,
		rows   = int(k),
		cols   = int(k),
		ld     = int(k),
		format = .General,
	}
	ldt := k

	lapack.cgeqrt2_(&m, &n, raw_data(A.data), &lda, raw_data(T_matrix.data), &ldt, &info)

	return T_matrix, info
}

m_qr_compact_unblocked_c128 :: proc(
	A: ^Matrix(complex128),
	allocator := context.allocator,
) -> (
	T_matrix: Matrix(complex128),
	info: Info,
) {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	k := min(m, n)

	// Allocate T matrix
	t_data := builtin.make([]complex128, k * k, allocator)
	T_matrix = Matrix(complex128) {
		data   = t_data,
		rows   = int(k),
		cols   = int(k),
		ld     = int(k),
		format = .General,
	}
	ldt := k

	lapack.zgeqrt2_(&m, &n, raw_data(A.data), &lda, raw_data(T_matrix.data), &ldt, &info)

	return T_matrix, info
}

// Compute QR factorization with T factor (recursive)
// Recursive algorithm for tall-skinny matrices
m_qr_compact_recursive :: proc {
	m_qr_compact_recursive_real,
	m_qr_compact_recursive_c64,
	m_qr_compact_recursive_c128,
}

m_qr_compact_recursive_real :: proc(
	A: ^Matrix($T), // General matrix (overwritten with QR factorization)
	allocator := context.allocator,
) -> (
	T_matrix: Matrix(T),
	info: Info, // T factor for compact WY representation
) where T == f32 || T == f64 {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	k := min(m, n)

	// Allocate T matrix (k x k upper triangular)
	t_data := builtin.make([]T, k * k, allocator)
	T_matrix = Matrix(T) {
		data   = t_data,
		rows   = int(k),
		cols   = int(k),
		ld     = int(k),
		format = .General,
	}
	ldt := k

	when T == f32 {
		lapack.sgeqrt3_(&m, &n, raw_data(A.data), &lda, raw_data(T_matrix.data), &ldt, &info)
	} else {
		lapack.dgeqrt3_(&m, &n, raw_data(A.data), &lda, raw_data(T_matrix.data), &ldt, &info)
	}

	return T_matrix, info
}

m_qr_compact_recursive_c64 :: proc(
	A: ^Matrix(complex64),
	allocator := context.allocator,
) -> (
	T_matrix: Matrix(complex64),
	info: Info,
) {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	k := min(m, n)

	// Allocate T matrix
	t_data := builtin.make([]complex64, k * k, allocator)
	T_matrix = Matrix(complex64) {
		data   = t_data,
		rows   = int(k),
		cols   = int(k),
		ld     = int(k),
		format = .General,
	}
	ldt := k

	lapack.cgeqrt3_(&m, &n, raw_data(A.data), &lda, raw_data(T_matrix.data), &ldt, &info)

	return T_matrix, info
}

m_qr_compact_recursive_c128 :: proc(
	A: ^Matrix(complex128),
	allocator := context.allocator,
) -> (
	T_matrix: Matrix(complex128),
	info: Info,
) {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	k := min(m, n)

	// Allocate T matrix
	t_data := builtin.make([]complex128, k * k, allocator)
	T_matrix = Matrix(complex128) {
		data   = t_data,
		rows   = int(k),
		cols   = int(k),
		ld     = int(k),
		format = .General,
	}
	ldt := k

	lapack.zgeqrt3_(&m, &n, raw_data(A.data), &lda, raw_data(T_matrix.data), &ldt, &info)

	return T_matrix, info
}

// Compute tall-skinny QR factorization with hierarchical representation
// Optimized for matrices where m >> n (many more rows than columns)
// Uses a multi-level blocked algorithm for better performance
m_qr_tall_skinny :: proc {
	m_qr_tall_skinny_real,
	m_qr_tall_skinny_c64,
	m_qr_tall_skinny_c128,
}

m_qr_tall_skinny_real :: proc(
	A: ^Matrix($T), // Tall-skinny matrix (overwritten with QR factorization)
	mb1: Blas_Int, // Block size for first level blocking
	nb1: Blas_Int, // Block size for panel factorization
	nb2: Blas_Int, // Block size for trailing matrix update
	allocator := context.allocator,
) -> (
	T_matrix: Matrix(T),
	info: Info, // Hierarchical T factors
) where T == f32 || T == f64 {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)

	// Allocate T matrix for hierarchical representation
	ldt := nb2
	T_matrix = make_matrix(T, int(ldt), int(n), allocator)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: T

	when T == f32 {
		lapack.sgetsqrhrt_(
			&m,
			&n,
			&mb1,
			&nb1,
			&nb2,
			raw_data(A.data),
			&lda,
			raw_data(T_matrix.data),
			&ldt,
			&work_query,
			&lwork,
			&info,
		)
	} else {
		lapack.dgetsqrhrt_(
			&m,
			&n,
			&mb1,
			&nb1,
			&nb2,
			raw_data(A.data),
			&lda,
			raw_data(T_matrix.data),
			&ldt,
			&work_query,
			&lwork,
			&info,
		)
	}

	// Allocate workspace
	lwork = Blas_Int(work_query)
	work := builtin.make([]T, lwork, allocator)
	defer builtin.delete(work)

	// Perform factorization
	when T == f32 {
		lapack.sgetsqrhrt_(
			&m,
			&n,
			&mb1,
			&nb1,
			&nb2,
			raw_data(A.data),
			&lda,
			raw_data(T_matrix.data),
			&ldt,
			raw_data(work),
			&lwork,
			&info,
		)
	} else {
		lapack.dgetsqrhrt_(
			&m,
			&n,
			&mb1,
			&nb1,
			&nb2,
			raw_data(A.data),
			&lda,
			raw_data(T_matrix.data),
			&ldt,
			raw_data(work),
			&lwork,
			&info,
		)
	}

	return T_matrix, info
}

m_qr_tall_skinny_c64 :: proc(
	A: ^Matrix(complex64),
	mb1: Blas_Int,
	nb1: Blas_Int,
	nb2: Blas_Int,
	allocator := context.allocator,
) -> (
	T_matrix: Matrix(complex64),
	info: Info,
) {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)

	// Allocate T matrix
	ldt := nb2
	T_matrix = make_matrix(complex64, int(ldt), int(n), allocator)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: complex64

	lapack.cgetsqrhrt_(
		&m,
		&n,
		&mb1,
		&nb1,
		&nb2,
		raw_data(A.data),
		&lda,
		raw_data(T_matrix.data),
		&ldt,
		&work_query,
		&lwork,
		&info,
	)

	// Allocate workspace
	lwork = Blas_Int(real(work_query))
	work := builtin.make([]complex64, lwork, allocator)
	defer builtin.delete(work)

	// Perform factorization
	lapack.cgetsqrhrt_(
		&m,
		&n,
		&mb1,
		&nb1,
		&nb2,
		raw_data(A.data),
		&lda,
		raw_data(T_matrix.data),
		&ldt,
		raw_data(work),
		&lwork,
		&info,
	)

	return T_matrix, info
}

m_qr_tall_skinny_c128 :: proc(
	A: ^Matrix(complex128),
	mb1: Blas_Int,
	nb1: Blas_Int,
	nb2: Blas_Int,
	allocator := context.allocator,
) -> (
	T_matrix: Matrix(complex128),
	info: Info,
) {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)

	// Allocate T matrix
	ldt := nb2
	T_matrix = make_matrix(complex128, int(ldt), int(n), allocator)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: complex128

	lapack.zgetsqrhrt_(
		&m,
		&n,
		&mb1,
		&nb1,
		&nb2,
		raw_data(A.data),
		&lda,
		raw_data(T_matrix.data),
		&ldt,
		&work_query,
		&lwork,
		&info,
	)

	// Allocate workspace
	lwork = Blas_Int(real(work_query))
	work := builtin.make([]complex128, lwork, allocator)
	defer builtin.delete(work)

	// Perform factorization
	lapack.zgetsqrhrt_(
		&m,
		&n,
		&mb1,
		&nb1,
		&nb2,
		raw_data(A.data),
		&lda,
		raw_data(T_matrix.data),
		&ldt,
		raw_data(work),
		&lwork,
		&info,
	)

	return T_matrix, info
}

// ===================================================================================
// RQ FACTORIZATION
// ===================================================================================

// Compute RQ factorization (unblocked algorithm)
// A = R * Q where R is upper triangular and Q is orthogonal/unitary
m_rq_unblocked :: proc {
	m_rq_unblocked_real,
	m_rq_unblocked_c64,
	m_rq_unblocked_c128,
}

m_rq_unblocked_real :: proc(
	A: ^Matrix($T), // General matrix (overwritten with RQ factorization)
	allocator := context.allocator,
) -> (
	tau: []T,
	info: Info, // Elementary reflectors
) where T == f32 || T == f64 {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	k := min(m, n)

	// Allocate tau
	tau = builtin.make([]T, k, allocator)

	// Allocate workspace (gerq2 requires workspace of size m)
	work := builtin.make([]T, m, allocator)
	defer builtin.delete(work)

	when T == f32 {
		lapack.sgerq2_(&m, &n, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &info)
	} else {
		lapack.dgerq2_(&m, &n, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &info)
	}

	return tau, info
}

m_rq_unblocked_c64 :: proc(
	A: ^Matrix(complex64),
	allocator := context.allocator,
) -> (
	tau: []complex64,
	info: Info,
) {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	k := min(m, n)

	// Allocate tau
	tau = builtin.make([]complex64, k, allocator)

	// Allocate workspace
	work := builtin.make([]complex64, m, allocator)
	defer builtin.delete(work)

	lapack.cgerq2_(&m, &n, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &info)

	return tau, info
}

m_rq_unblocked_c128 :: proc(
	A: ^Matrix(complex128),
	allocator := context.allocator,
) -> (
	tau: []complex128,
	info: Info,
) {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	k := min(m, n)

	// Allocate tau
	tau = builtin.make([]complex128, k, allocator)

	// Allocate workspace
	work := builtin.make([]complex128, m, allocator)
	defer builtin.delete(work)

	lapack.zgerq2_(&m, &n, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &info)

	return tau, info
}

// Compute RQ factorization (blocked algorithm)
// More efficient for larger matrices
m_rq :: proc {
	m_rq_real,
	m_rq_c64,
	m_rq_c128,
}

m_rq_real :: proc(
	A: ^Matrix($T), // General matrix (overwritten with RQ factorization)
	allocator := context.allocator,
) -> (
	tau: []T,
	info: Info, // Elementary reflectors
) where T == f32 || T == f64 {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	k := min(m, n)

	// Allocate tau
	tau = builtin.make([]T, k, allocator)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: T

	when T == f32 {
		lapack.sgerqf_(&m, &n, raw_data(A.data), &lda, raw_data(tau), &work_query, &lwork, &info)
		lwork = Blas_Int(work_query)

		// Allocate workspace
		work := builtin.make([]T, lwork, allocator)
		defer builtin.delete(work)

		// Compute factorization
		lapack.sgerqf_(
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(tau),
			raw_data(work),
			&lwork,
			&info,
		)
	} else {
		lapack.dgerqf_(&m, &n, raw_data(A.data), &lda, raw_data(tau), &work_query, &lwork, &info)
		lwork = Blas_Int(work_query)

		// Allocate workspace
		work := builtin.make([]T, lwork, allocator)
		defer builtin.delete(work)

		// Compute factorization
		lapack.dgerqf_(
			&m,
			&n,
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

m_rq_c64 :: proc(
	A: ^Matrix(complex64),
	allocator := context.allocator,
) -> (
	tau: []complex64,
	info: Info,
) {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	k := min(m, n)

	// Allocate tau
	tau = builtin.make([]complex64, k, allocator)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: complex64

	lapack.cgerqf_(&m, &n, raw_data(A.data), &lda, raw_data(tau), &work_query, &lwork, &info)
	lwork = Blas_Int(real(work_query))

	// Allocate workspace
	work := builtin.make([]complex64, lwork, allocator)
	defer builtin.delete(work)

	// Compute factorization
	lapack.cgerqf_(&m, &n, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &lwork, &info)

	return tau, info
}

m_rq_c128 :: proc(
	A: ^Matrix(complex128),
	allocator := context.allocator,
) -> (
	tau: []complex128,
	info: Info,
) {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	k := min(m, n)

	// Allocate tau
	tau = builtin.make([]complex128, k, allocator)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: complex128

	lapack.zgerqf_(&m, &n, raw_data(A.data), &lda, raw_data(tau), &work_query, &lwork, &info)
	lwork = Blas_Int(real(work_query))

	// Allocate workspace
	work := builtin.make([]complex128, lwork, allocator)
	defer builtin.delete(work)

	// Compute factorization
	lapack.zgerqf_(&m, &n, raw_data(A.data), &lda, raw_data(tau), raw_data(work), &lwork, &info)

	return tau, info
}
