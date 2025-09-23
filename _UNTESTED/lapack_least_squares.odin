package openblas

import lapack "./f77"
import "base:intrinsics"

// ===================================================================================
// LEAST SQUARES PROBLEMS
// Solve overdetermined or underdetermined systems in the least squares sense
// ===================================================================================

// Solve least squares problem: minimize ||A*x - B||_2 or ||A^T*x - B||_2
// Uses QR or LQ factorization (fastest, but requires full rank)
m_lstsq :: proc {
	m_lstsq_real,
	m_lstsq_c64,
	m_lstsq_c128,
}

m_lstsq_real :: proc(
	A: ^Matrix($T),
	B: ^Matrix(T),
	transpose: bool = false, // Solve A^T*x = B instead of A*x = B
	allocator := context.allocator,
) -> (
	info: Info,
) where T == f32 || T == f64 {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	nrhs := Blas_Int(B.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	trans_c := transpose ? cstring("T") : cstring("N")

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: T

	when T == f32 {
		lapack.sgels_(
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
		lapack.dgels_(
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
	work := builtin.make([]T, lwork, allocator)
	defer builtin.delete(work)

	// Solve least squares
	when T == f32 {
		lapack.sgels_(
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
		lapack.dgels_(
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
	}

	return info
}

m_lstsq_c64 :: proc(
	A: ^Matrix(complex64),
	B: ^Matrix(complex64),
	transpose: bool = false, // Use conjugate transpose if true
	allocator := context.allocator,
) -> (
	info: Info,
) {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	nrhs := Blas_Int(B.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	trans_c := transpose ? cstring("C") : cstring("N") // C = conjugate transpose

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: complex64

	lapack.cgels_(
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

	// Allocate workspace
	lwork = Blas_Int(real(work_query))
	work := builtin.make([]complex64, lwork, allocator)
	defer builtin.delete(work)

	// Solve least squares
	lapack.cgels_(
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

	return info
}

m_lstsq_c128 :: proc(
	A: ^Matrix(complex128),
	B: ^Matrix(complex128),
	transpose: bool = false,
	allocator := context.allocator,
) -> (
	info: Info,
) {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	nrhs := Blas_Int(B.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	trans_c := transpose ? cstring("C") : cstring("N")

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: complex128

	lapack.zgels_(
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

	// Allocate workspace
	lwork = Blas_Int(real(work_query))
	work := builtin.make([]complex128, lwork, allocator)
	defer builtin.delete(work)

	// Solve least squares
	lapack.zgels_(
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

	return info
}

// Solve least squares using SVD (most robust, handles rank deficiency)
// minimize ||A*x - B||_2 using divide-and-conquer SVD
m_lstsq_svd :: proc {
	m_lstsq_svd_real,
	m_lstsq_svd_c64,
	m_lstsq_svd_c128,
}

m_lstsq_svd_real :: proc(
	A: ^Matrix($T),
	B: ^Matrix(T),
	rcond: T = T(-1), // Singular value threshold (-1 = machine precision)
	allocator := context.allocator,
) -> (
	S: []T,
	rank: Blas_Int,
	info: Info, // Singular values// Effective rank
) where T == f32 || T == f64 {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	nrhs := Blas_Int(B.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Allocate singular values
	S = builtin.make([]T, min(m, n), allocator)

	rcond_val := rcond

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: T
	iwork_size := builtin.make([]i32, 1, allocator)
	defer builtin.delete(iwork_size)

	when T == f32 {
		lapack.sgelsd_(
			&m,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			raw_data(S),
			&rcond_val,
			&rank,
			&work_query,
			&lwork,
			raw_data(iwork_size),
			&info,
		)
	} else when T == f64 {
		lapack.dgelsd_(
			&m,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			raw_data(S),
			&rcond_val,
			&rank,
			&work_query,
			&lwork,
			raw_data(iwork_size),
			&info,
		)
	}

	// Allocate workspace
	lwork = Blas_Int(work_query)
	work := builtin.make([]T, lwork, allocator)
	defer builtin.delete(work)

	// Compute minimum norm of iwork
	minmn := min(m, n)
	nlvl := max(i32(0), Blas_Int(intrinsics.count_trailing_zeros(u32(minmn))) + 1)
	liwork := 3 * minmn * nlvl + 11 * minmn
	iwork := builtin.make([]i32, liwork, allocator)
	defer builtin.delete(iwork)

	// Solve least squares with SVD
	when T == f32 {
		lapack.sgelsd_(
			&m,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			raw_data(S),
			&rcond_val,
			&rank,
			raw_data(work),
			&lwork,
			raw_data(iwork),
			&info,
		)
	} else when T == f64 {
		lapack.dgelsd_(
			&m,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			raw_data(S),
			&rcond_val,
			&rank,
			raw_data(work),
			&lwork,
			raw_data(iwork),
			&info,
		)
	}

	return S, rank, info
}

m_lstsq_svd_c64 :: proc(
	A: ^Matrix(complex64),
	B: ^Matrix(complex64),
	rcond: f32 = -1,
	allocator := context.allocator,
) -> (
	S: []f32,
	rank: Blas_Int,
	info: Info,
) {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	nrhs := Blas_Int(B.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Allocate singular values
	S = builtin.make([]f32, min(m, n), allocator)

	rcond_val := rcond

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: complex64
	rwork_size := builtin.make([]f32, 1, allocator)
	iwork_size := builtin.make([]i32, 1, allocator)
	defer builtin.delete(rwork_size)
	defer builtin.delete(iwork_size)

	lapack.cgelsd_(
		&m,
		&n,
		&nrhs,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(S),
		&rcond_val,
		&rank,
		&work_query,
		&lwork,
		raw_data(rwork_size),
		raw_data(iwork_size),
		&info,
	)

	// Allocate workspace
	lwork = Blas_Int(real(work_query))
	work := builtin.make([]complex64, lwork, allocator)
	defer builtin.delete(work)

	// Compute sizes for real and integer workspace
	minmn := min(m, n)
	nlvl := max(i32(0), Blas_Int(intrinsics.count_trailing_zeros(u32(minmn))) + 1)
	lrwork := 10 * minmn + 2 * minmn * 25 + 8 * minmn * nlvl + 3 * 25 * (25 + 1) + 26 * 26 * 26
	rwork := builtin.make([]f32, lrwork, allocator)
	defer builtin.delete(rwork)

	liwork := 3 * minmn * nlvl + 11 * minmn
	iwork := builtin.make([]i32, liwork, allocator)
	defer builtin.delete(iwork)

	// Solve least squares with SVD
	lapack.cgelsd_(
		&m,
		&n,
		&nrhs,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(S),
		&rcond_val,
		&rank,
		raw_data(work),
		&lwork,
		raw_data(rwork),
		raw_data(iwork),
		&info,
	)

	return S, rank, info
}

m_lstsq_svd_c128 :: proc(
	A: ^Matrix(complex128),
	B: ^Matrix(complex128),
	rcond: f64 = -1,
	allocator := context.allocator,
) -> (
	S: []f64,
	rank: Blas_Int,
	info: Info,
) {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	nrhs := Blas_Int(B.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Allocate singular values
	S = builtin.make([]f64, min(m, n), allocator)

	rcond_val := rcond

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: complex128
	rwork_size := builtin.make([]f64, 1, allocator)
	iwork_size := builtin.make([]i32, 1, allocator)
	defer builtin.delete(rwork_size)
	defer builtin.delete(iwork_size)

	lapack.zgelsd_(
		&m,
		&n,
		&nrhs,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(S),
		&rcond_val,
		&rank,
		&work_query,
		&lwork,
		raw_data(rwork_size),
		raw_data(iwork_size),
		&info,
	)

	// Allocate workspace
	lwork = Blas_Int(real(work_query))
	work := builtin.make([]complex128, lwork, allocator)
	defer builtin.delete(work)

	// Compute sizes for real and integer workspace
	minmn := min(m, n)
	nlvl := max(i32(0), Blas_Int(intrinsics.count_trailing_zeros(u32(minmn))) + 1)
	lrwork := 10 * minmn + 2 * minmn * 25 + 8 * minmn * nlvl + 3 * 25 * (25 + 1) + 26 * 26 * 26
	rwork := builtin.make([]f64, lrwork, allocator)
	defer builtin.delete(rwork)

	liwork := 3 * minmn * nlvl + 11 * minmn
	iwork := builtin.make([]i32, liwork, allocator)
	defer builtin.delete(iwork)

	// Solve least squares with SVD
	lapack.zgelsd_(
		&m,
		&n,
		&nrhs,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(S),
		&rcond_val,
		&rank,
		raw_data(work),
		&lwork,
		raw_data(rwork),
		raw_data(iwork),
		&info,
	)

	return S, rank, info
}

// Solve least squares using SVD with explicit threshold control
// Uses standard SVD algorithm (slower than divide-and-conquer but more control)
m_lstsq_svd_simple :: proc {
	m_lstsq_svd_simple_real,
	m_lstsq_svd_simple_c64,
	m_lstsq_svd_simple_c128,
}

m_lstsq_svd_simple_real :: proc(
	A: ^Matrix($T),
	B: ^Matrix(T),
	rcond: T = T(-1), // Singular value threshold
	allocator := context.allocator,
) -> (
	S: []T,
	rank: Blas_Int,
	info: Info, // Singular values// Effective rank
) where T == f32 || T == f64 {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	nrhs := Blas_Int(B.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Allocate singular values
	S = builtin.make([]T, min(m, n), allocator)

	rcond_val := rcond

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: T

	when T == f32 {
		lapack.sgelss_(
			&m,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			raw_data(S),
			&rcond_val,
			&rank,
			&work_query,
			&lwork,
			&info,
		)
	} else when T == f64 {
		lapack.dgelss_(
			&m,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			raw_data(S),
			&rcond_val,
			&rank,
			&work_query,
			&lwork,
			&info,
		)
	}

	// Allocate workspace
	lwork = Blas_Int(work_query)
	work := builtin.make([]T, lwork, allocator)
	defer builtin.delete(work)

	// Solve least squares with SVD
	when T == f32 {
		lapack.sgelss_(
			&m,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			raw_data(S),
			&rcond_val,
			&rank,
			raw_data(work),
			&lwork,
			&info,
		)
	} else when T == f64 {
		lapack.dgelss_(
			&m,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			raw_data(S),
			&rcond_val,
			&rank,
			raw_data(work),
			&lwork,
			&info,
		)
	}

	return S, rank, info
}

m_lstsq_svd_simple_c64 :: proc(
	A: ^Matrix(complex64),
	B: ^Matrix(complex64),
	rcond: f32 = -1,
	allocator := context.allocator,
) -> (
	S: []f32,
	rank: Blas_Int,
	info: Info,
) {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	nrhs := Blas_Int(B.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Allocate singular values
	S = builtin.make([]f32, min(m, n), allocator)

	rcond_val := rcond

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: complex64
	rwork := builtin.make([]f32, 5 * min(m, n), allocator)
	defer builtin.delete(rwork)

	lapack.cgelss_(
		&m,
		&n,
		&nrhs,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(S),
		&rcond_val,
		&rank,
		&work_query,
		&lwork,
		raw_data(rwork),
		&info,
	)

	// Allocate workspace
	lwork = Blas_Int(real(work_query))
	work := builtin.make([]complex64, lwork, allocator)
	defer builtin.delete(work)

	// Solve least squares with SVD
	lapack.cgelss_(
		&m,
		&n,
		&nrhs,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(S),
		&rcond_val,
		&rank,
		raw_data(work),
		&lwork,
		raw_data(rwork),
		&info,
	)

	return S, rank, info
}

m_lstsq_svd_simple_c128 :: proc(
	A: ^Matrix(complex128),
	B: ^Matrix(complex128),
	rcond: f64 = -1,
	allocator := context.allocator,
) -> (
	S: []f64,
	rank: Blas_Int,
	info: Info,
) {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	nrhs := Blas_Int(B.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Allocate singular values
	S = builtin.make([]f64, min(m, n), allocator)

	rcond_val := rcond

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: complex128
	rwork := builtin.make([]f64, 5 * min(m, n), allocator)
	defer builtin.delete(rwork)

	lapack.zgelss_(
		&m,
		&n,
		&nrhs,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(S),
		&rcond_val,
		&rank,
		&work_query,
		&lwork,
		raw_data(rwork),
		&info,
	)

	// Allocate workspace
	lwork = Blas_Int(real(work_query))
	work := builtin.make([]complex128, lwork, allocator)
	defer builtin.delete(work)

	// Solve least squares with SVD
	lapack.zgelss_(
		&m,
		&n,
		&nrhs,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(S),
		&rcond_val,
		&rank,
		raw_data(work),
		&lwork,
		raw_data(rwork),
		&info,
	)

	return S, rank, info
}

// Solve least squares using QR with column pivoting
// Good balance between speed and robustness for rank-deficient problems
m_lstsq_qrp :: proc {
	m_lstsq_qrp_real,
	m_lstsq_qrp_c64,
	m_lstsq_qrp_c128,
}

m_lstsq_qrp_real :: proc(
	A: ^Matrix($T),
	B: ^Matrix(T),
	rcond: T = T(-1), // Threshold for rank determination
	allocator := context.allocator,
) -> (
	jpvt: []i32,
	rank: Blas_Int,
	info: Info, // Column pivot indices// Effective rank
) where T == f32 || T == f64 {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	nrhs := Blas_Int(B.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Allocate pivot array (initialized to 0 for free pivoting)
	jpvt = builtin.make([]i32, n, allocator)
	builtin.mem_zero(raw_data(jpvt), size_of(i32) * int(n))

	rcond_val := rcond

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: T

	when T == f32 {
		lapack.sgelsy_(
			&m,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			raw_data(jpvt),
			&rcond_val,
			&rank,
			&work_query,
			&lwork,
			&info,
		)
	} else when T == f64 {
		lapack.dgelsy_(
			&m,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			raw_data(jpvt),
			&rcond_val,
			&rank,
			&work_query,
			&lwork,
			&info,
		)
	}

	// Allocate workspace
	lwork = Blas_Int(work_query)
	work := builtin.make([]T, lwork, allocator)
	defer builtin.delete(work)

	// Solve least squares with QR pivoting
	when T == f32 {
		lapack.sgelsy_(
			&m,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			raw_data(jpvt),
			&rcond_val,
			&rank,
			raw_data(work),
			&lwork,
			&info,
		)
	} else when T == f64 {
		lapack.dgelsy_(
			&m,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			raw_data(jpvt),
			&rcond_val,
			&rank,
			raw_data(work),
			&lwork,
			&info,
		)
	}

	return jpvt, rank, info
}

m_lstsq_qrp_c64 :: proc(
	A: ^Matrix(complex64),
	B: ^Matrix(complex64),
	rcond: f32 = -1,
	allocator := context.allocator,
) -> (
	jpvt: []i32,
	rank: Blas_Int,
	info: Info,
) {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	nrhs := Blas_Int(B.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Allocate pivot array
	jpvt = builtin.make([]i32, n, allocator)
	builtin.mem_zero(raw_data(jpvt), size_of(i32) * int(n))

	rcond_val := rcond

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: complex64
	rwork := builtin.make([]f32, 2 * int(n), allocator)
	defer builtin.delete(rwork)

	lapack.cgelsy_(
		&m,
		&n,
		&nrhs,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(jpvt),
		&rcond_val,
		&rank,
		&work_query,
		&lwork,
		raw_data(rwork),
		&info,
	)

	// Allocate workspace
	lwork = Blas_Int(real(work_query))
	work := builtin.make([]complex64, lwork, allocator)
	defer builtin.delete(work)

	// Solve least squares with QR pivoting
	lapack.cgelsy_(
		&m,
		&n,
		&nrhs,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(jpvt),
		&rcond_val,
		&rank,
		raw_data(work),
		&lwork,
		raw_data(rwork),
		&info,
	)

	return jpvt, rank, info
}

m_lstsq_qrp_c128 :: proc(
	A: ^Matrix(complex128),
	B: ^Matrix(complex128),
	rcond: f64 = -1,
	allocator := context.allocator,
) -> (
	jpvt: []i32,
	rank: Blas_Int,
	info: Info,
) {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	nrhs := Blas_Int(B.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Allocate pivot array
	jpvt = builtin.make([]i32, n, allocator)
	builtin.mem_zero(raw_data(jpvt), size_of(i32) * int(n))

	rcond_val := rcond

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: complex128
	rwork := builtin.make([]f64, 2 * int(n), allocator)
	defer builtin.delete(rwork)

	lapack.zgelsy_(
		&m,
		&n,
		&nrhs,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(jpvt),
		&rcond_val,
		&rank,
		&work_query,
		&lwork,
		raw_data(rwork),
		&info,
	)

	// Allocate workspace
	lwork = Blas_Int(real(work_query))
	work := builtin.make([]complex128, lwork, allocator)
	defer builtin.delete(work)

	// Solve least squares with QR pivoting
	lapack.zgelsy_(
		&m,
		&n,
		&nrhs,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(jpvt),
		&rcond_val,
		&rank,
		raw_data(work),
		&lwork,
		raw_data(rwork),
		&info,
	)

	return jpvt, rank, info
}
