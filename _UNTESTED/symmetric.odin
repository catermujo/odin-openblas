package openblas

import lapack "./f77"
import "base:intrinsics"

// ===================================================================================
// SYMMETRIC AND HERMITIAN SYSTEMS
// Specialized routines for symmetric/Hermitian matrices
// ===================================================================================

// ===================================================================================
// HERMITIAN MATRIX CONDITION NUMBER ESTIMATION
// ===================================================================================

// Estimate condition number of Hermitian matrix using 1-norm
// Matrix must be factorized using Bunch-Kaufman factorization first
m_condition_hermitian :: proc {
	m_condition_hermitian_c64,
	m_condition_hermitian_c128,
}

m_condition_hermitian_c64 :: proc(
	A: ^Matrix(complex64), // Factorized Hermitian matrix from hetrf
	ipiv: []Blas_Int, // Pivot indices from hetrf factorization
	anorm: f32, // 1-norm of original matrix A
	allocator := context.allocator,
) -> (
	rcond: f32,
	info: Info, // Reciprocal condition number estimate
) {
	assert(
		A.format == .General || A.format == .Hermitian,
		"Matrix must be general or Hermitian format",
	)

	n := A.rows
	lda := A.ld
	uplo_c := cstring("U")

	// Workspace
	work := make([]complex64, 2 * n, allocator)

	lapack.checon_(
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(ipiv),
		&anorm,
		&rcond,
		raw_data(work),
		&info,
		1,
	)

	return rcond, info
}

m_condition_hermitian_c128 :: proc(
	A: ^Matrix(complex128), // Factorized Hermitian matrix from hetrf
	ipiv: []Blas_Int, // Pivot indices from hetrf factorization
	anorm: f64, // 1-norm of original matrix A
	allocator := context.allocator,
) -> (
	rcond: f64,
	info: Info, // Reciprocal condition number estimate
) {
	assert(
		A.format == .General || A.format == .Hermitian,
		"Matrix must be general or Hermitian format",
	)

	n := A.rows
	lda := A.ld
	uplo_c := cstring("U")

	// Workspace
	work := make([]complex128, 2 * n, allocator)

	lapack.zhecon_(
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(ipiv),
		&anorm,
		&rcond,
		raw_data(work),
		&info,
		1,
	)

	return rcond, info
}

// Estimate condition number of Hermitian matrix using 1-norm (version 3)
// Matrix must be factorized using rook pivoting Bunch-Kaufman factorization first
m_condition_hermitian_v3 :: proc {
	m_condition_hermitian_v3_c64,
	m_condition_hermitian_v3_c128,
}

m_condition_hermitian_v3_c64 :: proc(
	A: ^Matrix(complex64), // Factorized Hermitian matrix from hetrf_rook
	E: []complex64, // Details of block structure from hetrf_rook
	ipiv: []Blas_Int, // Pivot indices from hetrf_rook factorization
	anorm: f32, // 1-norm of original matrix A
	allocator := context.allocator,
) -> (
	rcond: f32,
	info: Info, // Reciprocal condition number estimate
) {
	assert(
		A.format == .General || A.format == .Hermitian,
		"Matrix must be general or Hermitian format",
	)

	n := A.rows
	lda := A.ld
	uplo_c := cstring("U")

	// Workspace
	work := make([]complex64, 2 * n, allocator)

	lapack.checon_3_(
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(E),
		raw_data(ipiv),
		&anorm,
		&rcond,
		raw_data(work),
		&info,
		1,
	)

	return rcond, info
}

m_condition_hermitian_v3_c128 :: proc(
	A: ^Matrix(complex128), // Factorized Hermitian matrix from hetrf_rook
	E: []complex128, // Details of block structure from hetrf_rook
	ipiv: []Blas_Int, // Pivot indices from hetrf_rook factorization
	anorm: f64, // 1-norm of original matrix A
	allocator := context.allocator,
) -> (
	rcond: f64,
	info: Info, // Reciprocal condition number estimate
) {
	assert(
		A.format == .General || A.format == .Hermitian,
		"Matrix must be general or Hermitian format",
	)

	n := A.rows
	lda := A.ld
	uplo_c := cstring("U")

	// Workspace
	work := make([]complex128, 2 * n, allocator)

	lapack.zhecon_3_(
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(E),
		raw_data(ipiv),
		&anorm,
		&rcond,
		raw_data(work),
		&info,
		1,
	)

	return rcond, info
}

// Compute row and column scalings to equilibrate Hermitian matrix
// Improves conditioning for solving linear systems
m_equilibrate_hermitian :: proc {
	m_equilibrate_hermitian_c64,
	m_equilibrate_hermitian_c128,
}

m_equilibrate_hermitian_c64 :: proc(
	A: ^Matrix(complex64), // Hermitian matrix to analyze
	allocator := context.allocator,
) -> (
	S: []f32,
	scond: f32,
	amax: f32,
	info: Info, // Scaling factors// Ratio of smallest to largest scaling factor// Absolute value of largest matrix element
) {
	assert(
		A.format == .General || A.format == .Hermitian,
		"Matrix must be general or Hermitian format",
	)

	n := A.rows
	lda := A.ld
	uplo_c := cstring("U")

	// Allocate outputs
	S = make([]f32, n, allocator)

	// Workspace
	work := make([]complex64, 3 * n, allocator)

	lapack.cheequb_(
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(S),
		&scond,
		&amax,
		raw_data(work),
		&info,
		1,
	)

	return S, scond, amax, info
}

m_equilibrate_hermitian_c128 :: proc(
	A: ^Matrix(complex128), // Hermitian matrix to analyze
	allocator := context.allocator,
) -> (
	S: []f64,
	scond: f64,
	amax: f64,
	info: Info, // Scaling factors// Ratio of smallest to largest scaling factor// Absolute value of largest matrix element
) {
	assert(
		A.format == .General || A.format == .Hermitian,
		"Matrix must be general or Hermitian format",
	)

	n := A.rows
	lda := A.ld
	uplo_c := cstring("U")

	// Allocate outputs
	S = make([]f64, n, allocator)

	// Workspace
	work := make([]complex128, 3 * n, allocator)

	lapack.zheequb_(
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(S),
		&scond,
		&amax,
		raw_data(work),
		&info,
		1,
	)

	return S, scond, amax, info
}

// ===================================================================================
// DENSE HERMITIAN EIGENVALUE ROUTINES
// ===================================================================================

// Dense Hermitian eigenvalue solver (basic)
// Computes all eigenvalues and optionally eigenvectors of dense Hermitian matrix
m_eigen_hermitian :: proc {
	m_eigen_hermitian_c64,
	m_eigen_hermitian_c128,
	m_eigen_hermitian_c64_2stage,
	m_eigen_hermitian_c128_2stage,
}

m_eigen_hermitian_c64 :: proc(
	A: ^Matrix(complex64), // Hermitian matrix (input/output)
	compute_vectors: bool = false,
	allocator := context.allocator,
) -> (
	w: []f32,
	info: Info, // Eigenvalues
) {
	assert(
		A.format == .General || A.format == .Hermitian,
		"Matrix must be general or Hermitian format",
	)

	n := A.rows
	lda := A.ld

	jobz_c := cstring("V") if compute_vectors else cstring("N")
	uplo_c := cstring("U")

	// Allocate eigenvalues
	w = make([]f32, n, allocator)

	// Workspace query
	work_query: complex64
	rwork_query: f32
	lwork: Blas_Int = -1

	lapack.cheev_(
		jobz_c,
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(w),
		&work_query,
		&lwork,
		&rwork_query,
		&info,
		1,
		1,
	)

	if info != 0 do return

	// Allocate workspace
	lwork = auto_cast real(work_query)
	work := make([]complex64, lwork, allocator)
	rwork := make([]f32, max(1, 3 * n - 2), allocator)

	lapack.cheev_(
		jobz_c,
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(w),
		raw_data(work),
		&lwork,
		raw_data(rwork),
		&info,
		1,
		1,
	)

	return w, info
}

m_eigen_hermitian_c128 :: proc(
	A: ^Matrix(complex128), // Hermitian matrix (input/output)
	compute_vectors: bool = false,
	allocator := context.allocator,
) -> (
	w: []f64,
	info: Info, // Eigenvalues
) {
	assert(
		A.format == .General || A.format == .Hermitian,
		"Matrix must be general or Hermitian format",
	)

	n := A.rows
	lda := A.ld

	jobz_c := cstring("V") if compute_vectors else cstring("N")
	uplo_c := cstring("U")

	// Allocate eigenvalues
	w = make([]f64, n, allocator)

	// Workspace query
	work_query: complex128
	rwork_query: f64
	lwork: Blas_Int = -1

	lapack.zheev_(
		jobz_c,
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(w),
		&work_query,
		&lwork,
		&rwork_query,
		&info,
		1,
		1,
	)

	if info != 0 do return

	// Allocate workspace
	lwork = auto_cast real(work_query)
	work := make([]complex128, lwork, allocator)
	rwork := make([]f64, max(1, 3 * n - 2), allocator)

	lapack.zheev_(
		jobz_c,
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(w),
		raw_data(work),
		&lwork,
		raw_data(rwork),
		&info,
		1,
		1,
	)

	return w, info
}

m_eigen_hermitian_c64_2stage :: proc(
	A: ^Matrix(complex64), // Hermitian matrix (input/output)
	compute_vectors: bool = false,
	allocator := context.allocator,
) -> (
	w: []f32,
	info: Info, // Eigenvalues
) {
	assert(
		A.format == .General || A.format == .Hermitian,
		"Matrix must be general or Hermitian format",
	)

	n := A.rows
	lda := A.ld

	jobz_c := cstring("V") if compute_vectors else cstring("N")
	uplo_c := cstring("U")

	// Allocate eigenvalues
	w = make([]f32, n, allocator)

	// Workspace query
	work_query: complex64
	rwork_query: f32
	lwork: Blas_Int = -1

	lapack.cheev_2stage_(
		jobz_c,
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(w),
		&work_query,
		&lwork,
		&rwork_query,
		&info,
		1,
		1,
	)

	if info != 0 do return

	// Allocate workspace
	lwork = auto_cast real(work_query)
	work := make([]complex64, lwork, allocator)
	rwork := make([]f32, max(1, 3 * n - 2), allocator)

	lapack.cheev_2stage_(
		jobz_c,
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(w),
		raw_data(work),
		&lwork,
		raw_data(rwork),
		&info,
		1,
		1,
	)

	return w, info
}

m_eigen_hermitian_c128_2stage :: proc(
	A: ^Matrix(complex128), // Hermitian matrix (input/output)
	compute_vectors: bool = false,
	allocator := context.allocator,
) -> (
	w: []f64,
	info: Info, // Eigenvalues
) {
	assert(
		A.format == .General || A.format == .Hermitian,
		"Matrix must be general or Hermitian format",
	)

	n := A.rows
	lda := A.ld

	jobz_c := cstring("V") if compute_vectors else cstring("N")
	uplo_c := cstring("U")

	// Allocate eigenvalues
	w = make([]f64, n, allocator)

	// Workspace query
	work_query: complex128
	rwork_query: f64
	lwork: Blas_Int = -1

	lapack.zheev_2stage_(
		jobz_c,
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(w),
		&work_query,
		&lwork,
		&rwork_query,
		&info,
		1,
		1,
	)

	if info != 0 do return

	// Allocate workspace
	lwork = auto_cast real(work_query)
	work := make([]complex128, lwork, allocator)
	rwork := make([]f64, max(1, 3 * n - 2), allocator)

	lapack.zheev_2stage_(
		jobz_c,
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(w),
		raw_data(work),
		&lwork,
		raw_data(rwork),
		&info,
		1,
		1,
	)

	return w, info
}

// Dense Hermitian eigenvalue solver (divide-and-conquer)
// Faster for large matrices when eigenvectors are needed
m_eigen_hermitian_dc :: proc {
	m_eigen_hermitian_dc_c64,
	m_eigen_hermitian_dc_c128,
	m_eigen_hermitian_dc_c64_2stage,
	m_eigen_hermitian_dc_c128_2stage,
}

m_eigen_hermitian_dc_c64 :: proc(
	A: ^Matrix(complex64), // Hermitian matrix (input/output)
	compute_vectors: bool = false,
	allocator := context.allocator,
) -> (
	w: []f32,
	info: Info, // Eigenvalues
) {
	assert(
		A.format == .General || A.format == .Hermitian,
		"Matrix must be general or Hermitian format",
	)

	n := A.rows
	lda := A.ld

	jobz_c := cstring("V") if compute_vectors else cstring("N")
	uplo_c := cstring("U")

	// Allocate eigenvalues
	w = make([]f32, n, allocator)

	// Workspace query
	work_query: complex64
	rwork_query: f32
	iwork_query: Blas_Int
	lwork: Blas_Int = -1
	lrwork: Blas_Int = -1
	liwork: Blas_Int = -1

	lapack.cheevd_(
		jobz_c,
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(w),
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
	iwork := make([]Blas_Int, liwork, allocator)

	lapack.cheevd_(
		jobz_c,
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(w),
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

	return w, info
}

m_eigen_hermitian_dc_c128 :: proc(
	A: ^Matrix(complex128), // Hermitian matrix (input/output)
	compute_vectors: bool = false,
	allocator := context.allocator,
) -> (
	w: []f64,
	info: Info, // Eigenvalues
) {
	assert(
		A.format == .General || A.format == .Hermitian,
		"Matrix must be general or Hermitian format",
	)

	n := A.rows
	lda := A.ld

	jobz_c := cstring("V") if compute_vectors else cstring("N")
	uplo_c := cstring("U")

	// Allocate eigenvalues
	w = make([]f64, n, allocator)

	// Workspace query
	work_query: complex128
	rwork_query: f64
	iwork_query: Blas_Int
	lwork: Blas_Int = -1
	lrwork: Blas_Int = -1
	liwork: Blas_Int = -1

	lapack.zheevd_(
		jobz_c,
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(w),
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
	iwork := make([]Blas_Int, liwork, allocator)

	lapack.zheevd_(
		jobz_c,
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(w),
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

	return w, info
}

m_eigen_hermitian_dc_c64_2stage :: proc(
	A: ^Matrix(complex64), // Hermitian matrix (input/output)
	compute_vectors: bool = false,
	allocator := context.allocator,
) -> (
	w: []f32,
	info: Info, // Eigenvalues
) {
	assert(
		A.format == .General || A.format == .Hermitian,
		"Matrix must be general or Hermitian format",
	)

	n := A.rows
	lda := A.ld

	jobz_c := cstring("V") if compute_vectors else cstring("N")
	uplo_c := cstring("U")

	// Allocate eigenvalues
	w = make([]f32, n, allocator)

	// Workspace query
	work_query: complex64
	rwork_query: f32
	iwork_query: Blas_Int
	lwork: Blas_Int = -1
	lrwork: Blas_Int = -1
	liwork: Blas_Int = -1

	lapack.cheevd_2stage_(
		jobz_c,
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(w),
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
	iwork := make([]Blas_Int, liwork, allocator)

	lapack.cheevd_2stage_(
		jobz_c,
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(w),
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

	return w, info
}

m_eigen_hermitian_dc_c128_2stage :: proc(
	A: ^Matrix(complex128), // Hermitian matrix (input/output)
	compute_vectors: bool = false,
	allocator := context.allocator,
) -> (
	w: []f64,
	info: Info, // Eigenvalues
) {
	assert(
		A.format == .General || A.format == .Hermitian,
		"Matrix must be general or Hermitian format",
	)

	n := A.rows
	lda := A.ld

	jobz_c := cstring("V") if compute_vectors else cstring("N")
	uplo_c := cstring("U")

	// Allocate eigenvalues
	w = make([]f64, n, allocator)

	// Workspace query
	work_query: complex128
	rwork_query: f64
	iwork_query: Blas_Int
	lwork: Blas_Int = -1
	lrwork: Blas_Int = -1
	liwork: Blas_Int = -1

	lapack.zheevd_2stage_(
		jobz_c,
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(w),
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
	iwork := make([]Blas_Int, liwork, allocator)

	lapack.zheevd_2stage_(
		jobz_c,
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(w),
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

	return w, info
}

// Dense Hermitian eigenvalue solver (robust eigenvector computation)
// Uses the Relatively Robust Representations (RRR) algorithm
m_eigen_hermitian_robust :: proc {
	m_eigen_hermitian_robust_c64,
	m_eigen_hermitian_robust_c128,
	m_eigen_hermitian_robust_c64_2stage,
	m_eigen_hermitian_robust_c128_2stage,
}

m_eigen_hermitian_robust_c64 :: proc(
	A: ^Matrix(complex64), // Hermitian matrix (input/output)
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
	Z: Matrix(complex64),
	isuppz: []Blas_Int,
	info: Info, // Number of eigenvalues found// Eigenvalues// Eigenvectors (if computed)// Support indices for eigenvectors
) {
	assert(
		A.format == .General || A.format == .Hermitian,
		"Matrix must be general or Hermitian format",
	)

	n := A.rows
	lda := A.ld

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
	isuppz = make([]Blas_Int, 2 * max(1, n), allocator)

	// Workspace query
	work_query: complex64
	rwork_query: f32
	iwork_query: Blas_Int
	lwork: Blas_Int = -1
	lrwork: Blas_Int = -1
	liwork: Blas_Int = -1

	lapack.cheevr_(
		jobz_c,
		range_c,
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		&vl,
		&vu,
		&il_use,
		&iu_use,
		&abstol,
		&m,
		raw_data(w),
		matrix_data_ptr(&Z) if compute_vectors else &work_query,
		&ldz,
		raw_data(isuppz),
		&work_query,
		&lwork,
		&rwork_query,
		&lrwork,
		&iwork_query,
		&liwork,
		&info,
		1,
		1,
		1,
	)

	if info != 0 do return

	// Allocate workspace
	lwork = auto_cast real(work_query)
	work := make([]complex64, lwork, allocator)

	lrwork = auto_cast rwork_query
	rwork := make([]f32, lrwork, allocator)

	liwork = iwork_query
	iwork := make([]Blas_Int, liwork, allocator)

	lapack.cheevr_(
		jobz_c,
		range_c,
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		&vl,
		&vu,
		&il_use,
		&iu_use,
		&abstol,
		&m,
		raw_data(w),
		matrix_data_ptr(&Z) if compute_vectors else nil,
		&ldz,
		raw_data(isuppz),
		raw_data(work),
		&lwork,
		raw_data(rwork),
		&lrwork,
		raw_data(iwork),
		&liwork,
		&info,
		1,
		1,
		1,
	)

	return m, w, Z, isuppz, info
}

m_eigen_hermitian_robust_c128 :: proc(
	A: ^Matrix(complex128), // Hermitian matrix (input/output)
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
	Z: Matrix(complex128),
	isuppz: []Blas_Int,
	info: Info, // Number of eigenvalues found// Eigenvalues// Eigenvectors (if computed)// Support indices for eigenvectors
) {
	assert(
		A.format == .General || A.format == .Hermitian,
		"Matrix must be general or Hermitian format",
	)

	n := A.rows
	lda := A.ld

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
	isuppz = make([]Blas_Int, 2 * max(1, n), allocator)

	// Workspace query
	work_query: complex128
	rwork_query: f64
	iwork_query: Blas_Int
	lwork: Blas_Int = -1
	lrwork: Blas_Int = -1
	liwork: Blas_Int = -1

	lapack.zheevr_(
		jobz_c,
		range_c,
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		&vl,
		&vu,
		&il_use,
		&iu_use,
		&abstol,
		&m,
		raw_data(w),
		matrix_data_ptr(&Z) if compute_vectors else &work_query,
		&ldz,
		raw_data(isuppz),
		&work_query,
		&lwork,
		&rwork_query,
		&lrwork,
		&iwork_query,
		&liwork,
		&info,
		1,
		1,
		1,
	)

	if info != 0 do return

	// Allocate workspace
	lwork = auto_cast real(work_query)
	work := make([]complex128, lwork, allocator)

	lrwork = auto_cast rwork_query
	rwork := make([]f64, lrwork, allocator)

	liwork = iwork_query
	iwork := make([]Blas_Int, liwork, allocator)

	lapack.zheevr_(
		jobz_c,
		range_c,
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		&vl,
		&vu,
		&il_use,
		&iu_use,
		&abstol,
		&m,
		raw_data(w),
		matrix_data_ptr(&Z) if compute_vectors else nil,
		&ldz,
		raw_data(isuppz),
		raw_data(work),
		&lwork,
		raw_data(rwork),
		&lrwork,
		raw_data(iwork),
		&liwork,
		&info,
		1,
		1,
		1,
	)

	return m, w, Z, isuppz, info
}

m_eigen_hermitian_robust_c64_2stage :: proc(
	A: ^Matrix(complex64), // Hermitian matrix (input/output)
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
	Z: Matrix(complex64),
	isuppz: []Blas_Int,
	info: Info, // Number of eigenvalues found// Eigenvalues// Eigenvectors (if computed)// Support indices for eigenvectors
) {
	assert(
		A.format == .General || A.format == .Hermitian,
		"Matrix must be general or Hermitian format",
	)

	n := A.rows
	lda := A.ld

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
	isuppz = make([]Blas_Int, 2 * max(1, n), allocator)

	// Workspace query
	work_query: complex64
	rwork_query: f32
	iwork_query: Blas_Int
	lwork: Blas_Int = -1
	lrwork: Blas_Int = -1
	liwork: Blas_Int = -1

	lapack.cheevr_2stage_(
		jobz_c,
		range_c,
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		&vl,
		&vu,
		&il_use,
		&iu_use,
		&abstol,
		&m,
		raw_data(w),
		matrix_data_ptr(&Z) if compute_vectors else &work_query,
		&ldz,
		raw_data(isuppz),
		&work_query,
		&lwork,
		&rwork_query,
		&lrwork,
		&iwork_query,
		&liwork,
		&info,
		1,
		1,
		1,
	)

	if info != 0 do return

	// Allocate workspace
	lwork = auto_cast real(work_query)
	work := make([]complex64, lwork, allocator)

	lrwork = auto_cast rwork_query
	rwork := make([]f32, lrwork, allocator)

	liwork = iwork_query
	iwork := make([]Blas_Int, liwork, allocator)

	lapack.cheevr_2stage_(
		jobz_c,
		range_c,
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		&vl,
		&vu,
		&il_use,
		&iu_use,
		&abstol,
		&m,
		raw_data(w),
		matrix_data_ptr(&Z) if compute_vectors else nil,
		&ldz,
		raw_data(isuppz),
		raw_data(work),
		&lwork,
		raw_data(rwork),
		&lrwork,
		raw_data(iwork),
		&liwork,
		&info,
		1,
		1,
		1,
	)

	return m, w, Z, isuppz, info
}

m_eigen_hermitian_robust_c128_2stage :: proc(
	A: ^Matrix(complex128), // Hermitian matrix (input/output)
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
	Z: Matrix(complex128),
	isuppz: []Blas_Int,
	info: Info, // Number of eigenvalues found// Eigenvalues// Eigenvectors (if computed)// Support indices for eigenvectors
) {
	assert(
		A.format == .General || A.format == .Hermitian,
		"Matrix must be general or Hermitian format",
	)

	n := A.rows
	lda := A.ld

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
	isuppz = make([]Blas_Int, 2 * max(1, n), allocator)

	// Workspace query
	work_query: complex128
	rwork_query: f64
	iwork_query: Blas_Int
	lwork: Blas_Int = -1
	lrwork: Blas_Int = -1
	liwork: Blas_Int = -1

	lapack.zheevr_2stage_(
		jobz_c,
		range_c,
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		&vl,
		&vu,
		&il_use,
		&iu_use,
		&abstol,
		&m,
		raw_data(w),
		matrix_data_ptr(&Z) if compute_vectors else &work_query,
		&ldz,
		raw_data(isuppz),
		&work_query,
		&lwork,
		&rwork_query,
		&lrwork,
		&iwork_query,
		&liwork,
		&info,
		1,
		1,
		1,
	)

	if info != 0 do return

	// Allocate workspace
	lwork = auto_cast real(work_query)
	work := make([]complex128, lwork, allocator)

	lrwork = auto_cast rwork_query
	rwork := make([]f64, lrwork, allocator)

	liwork = iwork_query
	iwork := make([]Blas_Int, liwork, allocator)

	lapack.zheevr_2stage_(
		jobz_c,
		range_c,
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		&vl,
		&vu,
		&il_use,
		&iu_use,
		&abstol,
		&m,
		raw_data(w),
		matrix_data_ptr(&Z) if compute_vectors else nil,
		&ldz,
		raw_data(isuppz),
		raw_data(work),
		&lwork,
		raw_data(rwork),
		&lrwork,
		raw_data(iwork),
		&liwork,
		&info,
		1,
		1,
		1,
	)

	return m, w, Z, isuppz, info
}

// Dense Hermitian eigenvalue solver (expert with subset selection)
// Computes selected eigenvalues and optionally eigenvectors using QR and bisection
m_eigen_hermitian_expert :: proc {
	m_eigen_hermitian_expert_c64,
	m_eigen_hermitian_expert_c128,
	m_eigen_hermitian_expert_c64_2stage,
	m_eigen_hermitian_expert_c128_2stage,
}

m_eigen_hermitian_expert_c64 :: proc(
	A: ^Matrix(complex64), // Hermitian matrix (input/output)
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
	Z: Matrix(complex64),
	ifail: []Blas_Int,
	info: Info, // Number of eigenvalues found// Eigenvalues// Eigenvectors (if computed)// Indices of failed eigenvectors
) {
	assert(
		A.format == .General || A.format == .Hermitian,
		"Matrix must be general or Hermitian format",
	)

	n := A.rows
	lda := A.ld

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
	ifail = make([]Blas_Int, n, allocator)

	// Workspace query
	work_query: complex64
	lwork: Blas_Int = -1

	lapack.cheevx_(
		jobz_c,
		range_c,
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
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
		nil,
		nil,
		raw_data(ifail),
		&info,
		1,
		1,
		1,
	)

	if info != 0 do return

	// Allocate workspace
	lwork = auto_cast real(work_query)
	work := make([]complex64, lwork, allocator)
	rwork := make([]f32, 7 * n, allocator)
	iwork := make([]Blas_Int, 5 * n, allocator)

	lapack.cheevx_(
		jobz_c,
		range_c,
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
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

m_eigen_hermitian_expert_c128 :: proc(
	A: ^Matrix(complex128), // Hermitian matrix (input/output)
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
	Z: Matrix(complex128),
	ifail: []Blas_Int,
	info: Info, // Number of eigenvalues found// Eigenvalues// Eigenvectors (if computed)// Indices of failed eigenvectors
) {
	assert(
		A.format == .General || A.format == .Hermitian,
		"Matrix must be general or Hermitian format",
	)

	n := A.rows
	lda := A.ld

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
	ifail = make([]Blas_Int, n, allocator)

	// Workspace query
	work_query: complex128
	lwork: Blas_Int = -1

	lapack.zheevx_(
		jobz_c,
		range_c,
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
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
		nil,
		nil,
		raw_data(ifail),
		&info,
		1,
		1,
		1,
	)

	if info != 0 do return

	// Allocate workspace
	lwork = auto_cast real(work_query)
	work := make([]complex128, lwork, allocator)
	rwork := make([]f64, 7 * n, allocator)
	iwork := make([]Blas_Int, 5 * n, allocator)

	lapack.zheevx_(
		jobz_c,
		range_c,
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
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

m_eigen_hermitian_expert_c64_2stage :: proc(
	A: ^Matrix(complex64), // Hermitian matrix (input/output)
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
	Z: Matrix(complex64),
	ifail: []Blas_Int,
	info: Info, // Number of eigenvalues found// Eigenvalues// Eigenvectors (if computed)// Indices of failed eigenvectors
) {
	assert(
		A.format == .General || A.format == .Hermitian,
		"Matrix must be general or Hermitian format",
	)

	n := A.rows
	lda := A.ld

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
	ifail = make([]Blas_Int, n, allocator)

	// Workspace query
	work_query: complex64
	lwork: Blas_Int = -1

	lapack.cheevx_2stage_(
		jobz_c,
		range_c,
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
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
		nil,
		nil,
		raw_data(ifail),
		&info,
		1,
		1,
		1,
	)

	if info != 0 do return

	// Allocate workspace
	lwork = auto_cast real(work_query)
	work := make([]complex64, lwork, allocator)
	rwork := make([]f32, 7 * n, allocator)
	iwork := make([]Blas_Int, 5 * n, allocator)

	lapack.cheevx_2stage_(
		jobz_c,
		range_c,
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
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

m_eigen_hermitian_expert_c128_2stage :: proc(
	A: ^Matrix(complex128), // Hermitian matrix (input/output)
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
	Z: Matrix(complex128),
	ifail: []Blas_Int,
	info: Info, // Number of eigenvalues found// Eigenvalues// Eigenvectors (if computed)// Indices of failed eigenvectors
) {
	assert(
		A.format == .General || A.format == .Hermitian,
		"Matrix must be general or Hermitian format",
	)

	n := A.rows
	lda := A.ld

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
	ifail = make([]Blas_Int, n, allocator)

	// Workspace query
	work_query: complex128
	lwork: Blas_Int = -1

	lapack.zheevx_2stage_(
		jobz_c,
		range_c,
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
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
		nil,
		nil,
		raw_data(ifail),
		&info,
		1,
		1,
		1,
	)

	if info != 0 do return

	// Allocate workspace
	lwork = auto_cast real(work_query)
	work := make([]complex128, lwork, allocator)
	rwork := make([]f64, 7 * n, allocator)
	iwork := make([]Blas_Int, 5 * n, allocator)

	lapack.zheevx_2stage_(
		jobz_c,
		range_c,
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
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
// GENERALIZED HERMITIAN EIGENVALUE PROBLEMS
// ===================================================================================
// Solve problems of the form A*x = λ*B*x where A, B are Hermitian and B is positive definite

// Reduction to standard form: A*x = λ*B*x → C*y = λ*y where C = B^(-1)*A
m_reduce_generalized_hermitian :: proc {
	m_reduce_generalized_hermitian_c64,
	m_reduce_generalized_hermitian_c128,
}

m_reduce_generalized_hermitian_c64 :: proc(
	itype: GeneralizedEigenProblemType, // Problem type (1: A*x = λ*B*x, 2: A*B*x = λ*x, 3: B*A*x = λ*x)
	A: ^Matrix(complex64), // Hermitian matrix A (overwritten with transformed matrix)
	B: ^Matrix(complex64), // Hermitian positive definite matrix B (overwritten with Cholesky factor)
	compute_upper := true, // true for upper triangle, false for lower
	allocator := context.allocator,
) -> (
	info: Info,
) {
	assert(
		A.format == .General || A.format == .Hermitian,
		"Matrix A must be general or Hermitian format",
	)
	assert(
		B.format == .General || B.format == .Hermitian,
		"Matrix B must be general or Hermitian format",
	)
	assert(A.rows == A.cols && B.rows == B.cols, "Matrices must be square")
	assert(A.rows == B.rows, "Matrices must have same dimensions")

	n := A.rows
	lda := A.ld
	ldb := B.ld

	itype_use := Blas_Int(itype)
	uplo_c := cstring("U") if compute_upper else cstring("L")

	lapack.chegst_(
		&itype_use,
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		&info,
		1,
	)

	return info
}

m_reduce_generalized_hermitian_c128 :: proc(
	itype: GeneralizedEigenProblemType, // Problem type (1: A*x = λ*B*x, 2: A*B*x = λ*x, 3: B*A*x = λ*x)
	A: ^Matrix(complex128), // Hermitian matrix A (overwritten with transformed matrix)
	B: ^Matrix(complex128), // Hermitian positive definite matrix B (overwritten with Cholesky factor)
	compute_upper := true, // true for upper triangle, false for lower
	allocator := context.allocator,
) -> (
	info: Info,
) {
	assert(
		A.format == .General || A.format == .Hermitian,
		"Matrix A must be general or Hermitian format",
	)
	assert(
		B.format == .General || B.format == .Hermitian,
		"Matrix B must be general or Hermitian format",
	)
	assert(A.rows == A.cols && B.rows == B.cols, "Matrices must be square")
	assert(A.rows == B.rows, "Matrices must have same dimensions")

	n := A.rows
	lda := A.ld
	ldb := B.ld

	itype_use := Blas_Int(itype)
	uplo_c := cstring("U") if compute_upper else cstring("L")

	lapack.zhegst_(
		&itype_use,
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		&info,
		1,
	)

	return info
}

// Basic generalized Hermitian eigenvalue solver
m_eigen_generalized_hermitian :: proc {
	m_eigen_generalized_hermitian_c64,
	m_eigen_generalized_hermitian_c128,
	m_eigen_generalized_hermitian_c64_2stage,
	m_eigen_generalized_hermitian_c128_2stage,
}

m_eigen_generalized_hermitian_c64 :: proc(
	itype: GeneralizedEigenProblemType, // Problem type (1: A*x = λ*B*x, 2: A*B*x = λ*x, 3: B*A*x = λ*x)
	A: ^Matrix(complex64), // Hermitian matrix A (overwritten)
	B: ^Matrix(complex64), // Hermitian positive definite matrix B (overwritten with Cholesky factor)
	compute_vectors := true, // Compute eigenvectors
	compute_upper := true, // true for upper triangle, false for lower
	allocator := context.allocator,
) -> (
	w: []f32,
	Z: Matrix(complex64),
	info: Info, // Eigenvalues// Eigenvectors (if computed)
) {
	assert(
		A.format == .General || A.format == .Hermitian,
		"Matrix A must be general or Hermitian format",
	)
	assert(
		B.format == .General || B.format == .Hermitian,
		"Matrix B must be general or Hermitian format",
	)
	assert(A.rows == A.cols && B.rows == B.cols, "Matrices must be square")
	assert(A.rows == B.rows, "Matrices must have same dimensions")

	n := A.rows
	lda := A.ld
	ldb := B.ld

	itype_use := Blas_Int(itype)
	jobz_c := cstring("V") if compute_vectors else cstring("N")
	uplo_c := cstring("U") if compute_upper else cstring("L")

	// Allocate outputs
	w = make([]f32, n, allocator)
	ldz := n if compute_vectors else 1
	Z =
		make_matrix(complex64, n, n, .General, allocator) if compute_vectors else Matrix(complex64){}

	// Workspace query
	work_query: complex64
	lwork: Blas_Int = -1

	lapack.chegv_(
		&itype_use,
		jobz_c,
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(w),
		&work_query,
		&lwork,
		nil, // rwork
		&info,
		1,
		1,
	)

	// Allocate workspace
	lwork = auto_cast real(work_query)
	work := make([]complex64, lwork, allocator)
	rwork := make([]f32, max(1, 3 * n - 2), allocator)

	lapack.chegv_(
		&itype_use,
		jobz_c,
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(w),
		raw_data(work),
		&lwork,
		raw_data(rwork),
		&info,
		1,
		1,
	)

	// Copy eigenvectors from A to Z if computed
	if compute_vectors && info == 0 {
		copy_slice(Z.data, A.data)
	}

	return w, Z, info
}

m_eigen_generalized_hermitian_c128 :: proc(
	itype: GeneralizedEigenProblemType, // Problem type (1: A*x = λ*B*x, 2: A*B*x = λ*x, 3: B*A*x = λ*x)
	A: ^Matrix(complex128), // Hermitian matrix A (overwritten)
	B: ^Matrix(complex128), // Hermitian positive definite matrix B (overwritten with Cholesky factor)
	compute_vectors := true, // Compute eigenvectors
	compute_upper := true, // true for upper triangle, false for lower
	allocator := context.allocator,
) -> (
	w: []f64,
	Z: Matrix(complex128),
	info: Info, // Eigenvalues// Eigenvectors (if computed)
) {
	assert(
		A.format == .General || A.format == .Hermitian,
		"Matrix A must be general or Hermitian format",
	)
	assert(
		B.format == .General || B.format == .Hermitian,
		"Matrix B must be general or Hermitian format",
	)
	assert(A.rows == A.cols && B.rows == B.cols, "Matrices must be square")
	assert(A.rows == B.rows, "Matrices must have same dimensions")

	n := A.rows
	lda := A.ld
	ldb := B.ld

	itype_use := Blas_Int(itype)
	jobz_c := cstring("V") if compute_vectors else cstring("N")
	uplo_c := cstring("U") if compute_upper else cstring("L")

	// Allocate outputs
	w = make([]f64, n, allocator)
	ldz := n if compute_vectors else 1
	Z =
		make_matrix(complex128, n, n, .General, allocator) if compute_vectors else Matrix(complex128){}

	// Workspace query
	work_query: complex128
	lwork: Blas_Int = -1

	lapack.zhegv_(
		&itype_use,
		jobz_c,
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(w),
		&work_query,
		&lwork,
		nil, // rwork
		&info,
		1,
		1,
	)

	// Allocate workspace
	lwork = auto_cast real(work_query)
	work := make([]complex128, lwork, allocator)
	rwork := make([]f64, max(1, 3 * n - 2), allocator)

	lapack.zhegv_(
		&itype_use,
		jobz_c,
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(w),
		raw_data(work),
		&lwork,
		raw_data(rwork),
		&info,
		1,
		1,
	)

	// Copy eigenvectors from A to Z if computed
	if compute_vectors && info == 0 {
		copy_slice(Z.data, A.data)
	}

	return w, Z, info
}

// 2-stage variants for improved performance
m_eigen_generalized_hermitian_c64_2stage :: proc(
	itype: GeneralizedEigenProblemType, // Problem type (1: A*x = λ*B*x, 2: A*B*x = λ*x, 3: B*A*x = λ*x)
	A: ^Matrix(complex64), // Hermitian matrix A (overwritten)
	B: ^Matrix(complex64), // Hermitian positive definite matrix B (overwritten with Cholesky factor)
	compute_vectors := true, // Compute eigenvectors
	compute_upper := true, // true for upper triangle, false for lower
	allocator := context.allocator,
) -> (
	w: []f32,
	Z: Matrix(complex64),
	info: Info, // Eigenvalues// Eigenvectors (if computed)
) {
	assert(
		A.format == .General || A.format == .Hermitian,
		"Matrix A must be general or Hermitian format",
	)
	assert(
		B.format == .General || B.format == .Hermitian,
		"Matrix B must be general or Hermitian format",
	)
	assert(A.rows == A.cols && B.rows == B.cols, "Matrices must be square")
	assert(A.rows == B.rows, "Matrices must have same dimensions")

	n := A.rows
	lda := A.ld
	ldb := B.ld

	itype_use := Blas_Int(itype)
	jobz_c := cstring("V") if compute_vectors else cstring("N")
	uplo_c := cstring("U") if compute_upper else cstring("L")

	// Allocate outputs
	w = make([]f32, n, allocator)
	ldz := n if compute_vectors else 1
	Z =
		make_matrix(complex64, n, n, .General, allocator) if compute_vectors else Matrix(complex64){}

	// Workspace query
	work_query: complex64
	lwork: Blas_Int = -1

	lapack.chegv_2stage_(
		&itype_use,
		jobz_c,
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(w),
		&work_query,
		&lwork,
		nil, // rwork
		&info,
		1,
		1,
	)

	// Allocate workspace
	lwork = auto_cast real(work_query)
	work := make([]complex64, lwork, allocator)
	rwork := make([]f32, max(1, 3 * n - 2), allocator)

	lapack.chegv_2stage_(
		&itype_use,
		jobz_c,
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(w),
		raw_data(work),
		&lwork,
		raw_data(rwork),
		&info,
		1,
		1,
	)

	// Copy eigenvectors from A to Z if computed
	if compute_vectors && info == 0 {
		copy_slice(Z.data, A.data)
	}

	return w, Z, info
}

m_eigen_generalized_hermitian_c128_2stage :: proc(
	itype: GeneralizedEigenProblemType, // Problem type (1: A*x = λ*B*x, 2: A*B*x = λ*x, 3: B*A*x = λ*x)
	A: ^Matrix(complex128), // Hermitian matrix A (overwritten)
	B: ^Matrix(complex128), // Hermitian positive definite matrix B (overwritten with Cholesky factor)
	compute_vectors := true, // Compute eigenvectors
	compute_upper := true, // true for upper triangle, false for lower
	allocator := context.allocator,
) -> (
	w: []f64,
	Z: Matrix(complex128),
	info: Info, // Eigenvalues// Eigenvectors (if computed)
) {
	assert(
		A.format == .General || A.format == .Hermitian,
		"Matrix A must be general or Hermitian format",
	)
	assert(
		B.format == .General || B.format == .Hermitian,
		"Matrix B must be general or Hermitian format",
	)
	assert(A.rows == A.cols && B.rows == B.cols, "Matrices must be square")
	assert(A.rows == B.rows, "Matrices must have same dimensions")

	n := A.rows
	lda := A.ld
	ldb := B.ld

	itype_use := Blas_Int(itype)
	jobz_c := cstring("V") if compute_vectors else cstring("N")
	uplo_c := cstring("U") if compute_upper else cstring("L")

	// Allocate outputs
	w = make([]f64, n, allocator)
	ldz := n if compute_vectors else 1
	Z =
		make_matrix(complex128, n, n, .General, allocator) if compute_vectors else Matrix(complex128){}

	// Workspace query
	work_query: complex128
	lwork: Blas_Int = -1

	lapack.zhegv_2stage_(
		&itype_use,
		jobz_c,
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(w),
		&work_query,
		&lwork,
		nil, // rwork
		&info,
		1,
		1,
	)

	// Allocate workspace
	lwork = auto_cast real(work_query)
	work := make([]complex128, lwork, allocator)
	rwork := make([]f64, max(1, 3 * n - 2), allocator)

	lapack.zhegv_2stage_(
		&itype_use,
		jobz_c,
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(w),
		raw_data(work),
		&lwork,
		raw_data(rwork),
		&info,
		1,
		1,
	)

	// Copy eigenvectors from A to Z if computed
	if compute_vectors && info == 0 {
		copy_slice(Z.data, A.data)
	}

	return w, Z, info
}

// Divide-and-conquer generalized Hermitian eigenvalue solver
m_eigen_generalized_hermitian_dc :: proc {
	m_eigen_generalized_hermitian_dc_c64,
	m_eigen_generalized_hermitian_dc_c128,
}

m_eigen_generalized_hermitian_dc_c64 :: proc(
	itype: GeneralizedEigenProblemType, // Problem type (1: A*x = λ*B*x, 2: A*B*x = λ*x, 3: B*A*x = λ*x)
	A: ^Matrix(complex64), // Hermitian matrix A (overwritten)
	B: ^Matrix(complex64), // Hermitian positive definite matrix B (overwritten with Cholesky factor)
	compute_vectors := true, // Compute eigenvectors
	compute_upper := true, // true for upper triangle, false for lower
	allocator := context.allocator,
) -> (
	w: []f32,
	Z: Matrix(complex64),
	info: Info, // Eigenvalues// Eigenvectors (if computed)
) {
	assert(
		A.format == .General || A.format == .Hermitian,
		"Matrix A must be general or Hermitian format",
	)
	assert(
		B.format == .General || B.format == .Hermitian,
		"Matrix B must be general or Hermitian format",
	)
	assert(A.rows == A.cols && B.rows == B.cols, "Matrices must be square")
	assert(A.rows == B.rows, "Matrices must have same dimensions")

	n := A.rows
	lda := A.ld
	ldb := B.ld

	itype_use := Blas_Int(itype)
	jobz_c := cstring("V") if compute_vectors else cstring("N")
	uplo_c := cstring("U") if compute_upper else cstring("L")

	// Allocate outputs
	w = make([]f32, n, allocator)
	ldz := n if compute_vectors else 1
	Z =
		make_matrix(complex64, n, n, .General, allocator) if compute_vectors else Matrix(complex64){}

	// Workspace query
	work_query: complex64
	rwork_query: f32
	iwork_query: Blas_Int
	lwork: Blas_Int = -1
	lrwork: Blas_Int = -1
	liwork: Blas_Int = -1

	lapack.chegvd_(
		&itype_use,
		jobz_c,
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(w),
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

	// Allocate workspace
	lwork = auto_cast real(work_query)
	work := make([]complex64, lwork, allocator)

	lrwork = auto_cast rwork_query
	rwork := make([]f32, lrwork, allocator)

	liwork = iwork_query
	iwork := make([]Blas_Int, liwork, allocator)

	lapack.chegvd_(
		&itype_use,
		jobz_c,
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(w),
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

	// Copy eigenvectors from A to Z if computed
	if compute_vectors && info == 0 {
		copy_slice(Z.data, A.data)
	}

	return w, Z, info
}

m_eigen_generalized_hermitian_dc_c128 :: proc(
	itype: GeneralizedEigenProblemType, // Problem type (1: A*x = λ*B*x, 2: A*B*x = λ*x, 3: B*A*x = λ*x)
	A: ^Matrix(complex128), // Hermitian matrix A (overwritten)
	B: ^Matrix(complex128), // Hermitian positive definite matrix B (overwritten with Cholesky factor)
	compute_vectors := true, // Compute eigenvectors
	compute_upper := true, // true for upper triangle, false for lower
	allocator := context.allocator,
) -> (
	w: []f64,
	Z: Matrix(complex128),
	info: Info, // Eigenvalues// Eigenvectors (if computed)
) {
	assert(
		A.format == .General || A.format == .Hermitian,
		"Matrix A must be general or Hermitian format",
	)
	assert(
		B.format == .General || B.format == .Hermitian,
		"Matrix B must be general or Hermitian format",
	)
	assert(A.rows == A.cols && B.rows == B.cols, "Matrices must be square")
	assert(A.rows == B.rows, "Matrices must have same dimensions")

	n := A.rows
	lda := A.ld
	ldb := B.ld

	itype_use := Blas_Int(itype)
	jobz_c := cstring("V") if compute_vectors else cstring("N")
	uplo_c := cstring("U") if compute_upper else cstring("L")

	// Allocate outputs
	w = make([]f64, n, allocator)
	ldz := n if compute_vectors else 1
	Z =
		make_matrix(complex128, n, n, .General, allocator) if compute_vectors else Matrix(complex128){}

	// Workspace query
	work_query: complex128
	rwork_query: f64
	iwork_query: Blas_Int
	lwork: Blas_Int = -1
	lrwork: Blas_Int = -1
	liwork: Blas_Int = -1

	lapack.zhegvd_(
		&itype_use,
		jobz_c,
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(w),
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

	// Allocate workspace
	lwork = auto_cast real(work_query)
	work := make([]complex128, lwork, allocator)

	lrwork = auto_cast rwork_query
	rwork := make([]f64, lrwork, allocator)

	liwork = iwork_query
	iwork := make([]Blas_Int, liwork, allocator)

	lapack.zhegvd_(
		&itype_use,
		jobz_c,
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(w),
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

	// Copy eigenvectors from A to Z if computed
	if compute_vectors && info == 0 {
		copy_slice(Z.data, A.data)
	}

	return w, Z, info
}

// Expert generalized Hermitian eigenvalue solver with subset selection
m_eigen_generalized_hermitian_expert :: proc {
	m_eigen_generalized_hermitian_expert_c64,
	m_eigen_generalized_hermitian_expert_c128,
}

m_eigen_generalized_hermitian_expert_c64 :: proc(
	itype: GeneralizedEigenProblemType, // Problem type (1: A*x = λ*B*x, 2: A*B*x = λ*x, 3: B*A*x = λ*x)
	A: ^Matrix(complex64), // Hermitian matrix A (overwritten)
	B: ^Matrix(complex64), // Hermitian positive definite matrix B (overwritten with Cholesky factor)
	compute_vectors := true, // Compute eigenvectors
	compute_upper := true, // true for upper triangle, false for lower
	range_type: EigenValueRange = .All, // Range of eigenvalues to compute
	il: int = 1, // Lower index (1-based) for .Indexed (1 means first eigenvalue)
	iu: int = 0, // Upper index (1-based) for .Indexed (0 means use n)
	vl: f32 = 0, // Lower bound for .Valued
	vu: f32 = 0, // Upper bound for .Valued
	abstol: f32 = 0, // Absolute tolerance (0 for default)
	allocator := context.allocator,
) -> (
	m: int,
	w: []f32,
	Z: Matrix(complex64),
	ifail: []Blas_Int,
	info: Info, // Number of eigenvalues found// Eigenvalues// Eigenvectors (if computed)// Failure information
) {
	assert(
		A.format == .General || A.format == .Hermitian,
		"Matrix A must be general or Hermitian format",
	)
	assert(
		B.format == .General || B.format == .Hermitian,
		"Matrix B must be general or Hermitian format",
	)
	assert(A.rows == A.cols && B.rows == B.cols, "Matrices must be square")
	assert(A.rows == B.rows, "Matrices must have same dimensions")

	n := A.rows
	lda := A.ld
	ldb := B.ld

	itype_use := Blas_Int(itype)
	jobz_c := cstring("V") if compute_vectors else cstring("N")
	uplo_c := cstring("U") if compute_upper else cstring("L")

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
	ifail = make([]Blas_Int, n, allocator)

	// Workspace allocation
	lwork := 8 * n
	work := make([]complex64, lwork, allocator)
	rwork := make([]f32, 7 * n, allocator)
	iwork := make([]Blas_Int, 5 * n, allocator)

	lapack.chegvx_(
		&itype_use,
		jobz_c,
		range_c,
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
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

m_eigen_generalized_hermitian_expert_c128 :: proc(
	itype: GeneralizedEigenProblemType, // Problem type (1: A*x = λ*B*x, 2: A*B*x = λ*x, 3: B*A*x = λ*x)
	A: ^Matrix(complex128), // Hermitian matrix A (overwritten)
	B: ^Matrix(complex128), // Hermitian positive definite matrix B (overwritten with Cholesky factor)
	compute_vectors := true, // Compute eigenvectors
	compute_upper := true, // true for upper triangle, false for lower
	range_type: EigenValueRange = .All, // Range of eigenvalues to compute
	il: int = 1, // Lower index (1-based) for .Indexed (1 means first eigenvalue)
	iu: int = 0, // Upper index (1-based) for .Indexed (0 means use n)
	vl: f64 = 0, // Lower bound for .Valued
	vu: f64 = 0, // Upper bound for .Valued
	abstol: f64 = 0, // Absolute tolerance (0 for default)
	allocator := context.allocator,
) -> (
	m: int,
	w: []f64,
	Z: Matrix(complex128),
	ifail: []Blas_Int,
	info: Info, // Number of eigenvalues found// Eigenvalues// Eigenvectors (if computed)// Failure information
) {
	assert(
		A.format == .General || A.format == .Hermitian,
		"Matrix A must be general or Hermitian format",
	)
	assert(
		B.format == .General || B.format == .Hermitian,
		"Matrix B must be general or Hermitian format",
	)
	assert(A.rows == A.cols && B.rows == B.cols, "Matrices must be square")
	assert(A.rows == B.rows, "Matrices must have same dimensions")

	n := A.rows
	lda := A.ld
	ldb := B.ld

	itype_use := Blas_Int(itype)
	jobz_c := cstring("V") if compute_vectors else cstring("N")
	uplo_c := cstring("U") if compute_upper else cstring("L")

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
	ifail = make([]Blas_Int, n, allocator)

	// Workspace allocation
	lwork := 8 * n
	work := make([]complex128, lwork, allocator)
	rwork := make([]f64, 7 * n, allocator)
	iwork := make([]Blas_Int, 5 * n, allocator)

	lapack.zhegvx_(
		&itype_use,
		jobz_c,
		range_c,
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
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
// HERMITIAN ITERATIVE REFINEMENT
// ===================================================================================
// Improve accuracy of solutions and provide error bounds for factorized systems

// Basic iterative refinement for Hermitian systems
m_refine_hermitian :: proc {
	m_refine_hermitian_c64,
	m_refine_hermitian_c128,
}

m_refine_hermitian_c64 :: proc(
	A: ^Matrix(complex64), // Original Hermitian matrix
	AF: ^Matrix(complex64), // Factorized matrix from m_factor_hermitian
	ipiv: []Blas_Int, // Pivot indices from factorization
	B: ^Matrix(complex64), // Right-hand side matrix
	X: ^Matrix(complex64), // Solution matrix (input/output, improved on output)
	compute_upper := true, // true for upper triangle, false for lower
	allocator := context.allocator,
) -> (
	ferr: []f32,
	berr: []f32,
	info: Info, // Forward error bounds for each RHS// Backward error bounds for each RHS
) {
	assert(
		A.format == .General || A.format == .Hermitian,
		"Matrix A must be general or Hermitian format",
	)
	assert(
		AF.format == .General || AF.format == .Hermitian,
		"Matrix AF must be general or Hermitian format",
	)
	assert(A.rows == A.cols, "Matrix A must be square")
	assert(AF.rows == AF.cols, "Matrix AF must be square")
	assert(A.rows == AF.rows, "Matrices A and AF must have same dimensions")
	assert(B.rows == A.rows, "Matrix B must have same number of rows as A")
	assert(X.rows == A.rows && X.cols == B.cols, "Matrix X must have same dimensions as B")

	n := A.rows
	nrhs := B.cols
	lda := A.ld
	ldaf := AF.ld
	ldb := B.ld
	ldx := X.ld

	uplo_c := cstring("U") if compute_upper else cstring("L")

	// Allocate outputs
	ferr = make([]f32, nrhs, allocator)
	berr = make([]f32, nrhs, allocator)

	// Workspace
	work := make([]complex64, 2 * n, allocator)
	rwork := make([]f32, n, allocator)

	lapack.cherfs_(
		uplo_c,
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

m_refine_hermitian_c128 :: proc(
	A: ^Matrix(complex128), // Original Hermitian matrix
	AF: ^Matrix(complex128), // Factorized matrix from m_factor_hermitian
	ipiv: []Blas_Int, // Pivot indices from factorization
	B: ^Matrix(complex128), // Right-hand side matrix
	X: ^Matrix(complex128), // Solution matrix (input/output, improved on output)
	compute_upper := true, // true for upper triangle, false for lower
	allocator := context.allocator,
) -> (
	ferr: []f64,
	berr: []f64,
	info: Info, // Forward error bounds for each RHS// Backward error bounds for each RHS
) {
	assert(
		A.format == .General || A.format == .Hermitian,
		"Matrix A must be general or Hermitian format",
	)
	assert(
		AF.format == .General || AF.format == .Hermitian,
		"Matrix AF must be general or Hermitian format",
	)
	assert(A.rows == A.cols, "Matrix A must be square")
	assert(AF.rows == AF.cols, "Matrix AF must be square")
	assert(A.rows == AF.rows, "Matrices A and AF must have same dimensions")
	assert(B.rows == A.rows, "Matrix B must have same number of rows as A")
	assert(X.rows == A.rows && X.cols == B.cols, "Matrix X must have same dimensions as B")

	n := A.rows
	nrhs := B.cols
	lda := A.ld
	ldaf := AF.ld
	ldb := B.ld
	ldx := X.ld

	uplo_c := cstring("U") if compute_upper else cstring("L")

	// Allocate outputs
	ferr = make([]f64, nrhs, allocator)
	berr = make([]f64, nrhs, allocator)

	// Workspace
	work := make([]complex128, 2 * n, allocator)
	rwork := make([]f64, n, allocator)

	lapack.zherfs_(
		uplo_c,
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

// Expert iterative refinement with equilibration and extended error bounds
m_refine_hermitian_expert :: proc {
	m_refine_hermitian_expert_c64,
	m_refine_hermitian_expert_c128,
}

m_refine_hermitian_expert_c64 :: proc(
	A: ^Matrix(complex64), // Original Hermitian matrix
	AF: ^Matrix(complex64), // Factorized matrix from m_factor_hermitian
	ipiv: []Blas_Int, // Pivot indices from factorization
	S: []f32, // Scaling factors from equilibration (or nil)
	B: ^Matrix(complex64), // Right-hand side matrix
	X: ^Matrix(complex64), // Solution matrix (input/output, improved on output)
	compute_upper := true, // true for upper triangle, false for lower
	was_equilibrated := false, // true if matrix was equilibrated
	n_err_bnds: int = 3, // Number of error bounds to compute (default 3)
	allocator := context.allocator,
) -> (
	rcond: f32,
	berr: []f32,
	err_bnds_norm: []f32,
	err_bnds_comp: []f32,
	info: Info, // Reciprocal condition number estimate// Backward error bounds for each RHS// Normwise error bounds// Componentwise error bounds
) {
	assert(
		A.format == .General || A.format == .Hermitian,
		"Matrix A must be general or Hermitian format",
	)
	assert(
		AF.format == .General || AF.format == .Hermitian,
		"Matrix AF must be general or Hermitian format",
	)
	assert(A.rows == A.cols, "Matrix A must be square")
	assert(AF.rows == AF.cols, "Matrix AF must be square")
	assert(A.rows == AF.rows, "Matrices A and AF must have same dimensions")
	assert(B.rows == A.rows, "Matrix B must have same number of rows as A")
	assert(X.rows == A.rows && X.cols == B.cols, "Matrix X must have same dimensions as B")

	n := A.rows
	nrhs := B.cols
	lda := A.ld
	ldaf := AF.ld
	ldb := B.ld
	ldx := X.ld

	uplo_c := cstring("U") if compute_upper else cstring("L")
	equed_c := cstring("Y") if was_equilibrated else cstring("N")

	// Allocate outputs
	berr = make([]f32, nrhs, allocator)
	err_bnds_norm = make([]f32, nrhs * n_err_bnds, allocator)
	err_bnds_comp = make([]f32, nrhs * n_err_bnds, allocator)

	// Default parameters for expert refinement
	nparams := Blas_Int(0) // Use default parameters
	params: []f32 = nil

	// Workspace
	work := make([]complex64, 2 * n, allocator)
	rwork := make([]f32, 3 * n, allocator)

	// Handle scaling factors
	s_ptr := raw_data(S) if S != nil else nil

	n_err_bnds_i := Blas_Int(n_err_bnds)

	lapack.cherfsx_(
		uplo_c,
		equed_c,
		&n,
		&nrhs,
		raw_data(A.data),
		&lda,
		raw_data(AF.data),
		&ldaf,
		raw_data(ipiv),
		s_ptr,
		raw_data(B.data),
		&ldb,
		raw_data(X.data),
		&ldx,
		&rcond,
		raw_data(berr),
		&n_err_bnds_i,
		raw_data(err_bnds_norm),
		raw_data(err_bnds_comp),
		&nparams,
		nil, // Use default parameters
		raw_data(work),
		raw_data(rwork),
		&info,
		1,
		1,
	)

	return rcond, berr, err_bnds_norm, err_bnds_comp, info
}

m_refine_hermitian_expert_c128 :: proc(
	A: ^Matrix(complex128), // Original Hermitian matrix
	AF: ^Matrix(complex128), // Factorized matrix from m_factor_hermitian
	ipiv: []Blas_Int, // Pivot indices from factorization
	S: []f64, // Scaling factors from equilibration (or nil)
	B: ^Matrix(complex128), // Right-hand side matrix
	X: ^Matrix(complex128), // Solution matrix (input/output, improved on output)
	compute_upper := true, // true for upper triangle, false for lower
	was_equilibrated := false, // true if matrix was equilibrated
	n_err_bnds: int = 3, // Number of error bounds to compute (default 3)
	allocator := context.allocator,
) -> (
	rcond: f64,
	berr: []f64,
	err_bnds_norm: []f64,
	err_bnds_comp: []f64,
	info: Info, // Reciprocal condition number estimate// Backward error bounds for each RHS// Normwise error bounds// Componentwise error bounds
) {
	assert(
		A.format == .General || A.format == .Hermitian,
		"Matrix A must be general or Hermitian format",
	)
	assert(
		AF.format == .General || AF.format == .Hermitian,
		"Matrix AF must be general or Hermitian format",
	)
	assert(A.rows == A.cols, "Matrix A must be square")
	assert(AF.rows == AF.cols, "Matrix AF must be square")
	assert(A.rows == AF.rows, "Matrices A and AF must have same dimensions")
	assert(B.rows == A.rows, "Matrix B must have same number of rows as A")
	assert(X.rows == A.rows && X.cols == B.cols, "Matrix X must have same dimensions as B")

	n := A.rows
	nrhs := B.cols
	lda := A.ld
	ldaf := AF.ld
	ldb := B.ld
	ldx := X.ld

	uplo_c := cstring("U") if compute_upper else cstring("L")
	equed_c := cstring("Y") if was_equilibrated else cstring("N")

	// Allocate outputs
	berr = make([]f64, nrhs, allocator)
	err_bnds_norm = make([]f64, nrhs * n_err_bnds, allocator)
	err_bnds_comp = make([]f64, nrhs * n_err_bnds, allocator)

	// Default parameters for expert refinement
	nparams := Blas_Int(0) // Use default parameters
	params: []f64 = nil

	// Workspace
	work := make([]complex128, 2 * n, allocator)
	rwork := make([]f64, 3 * n, allocator)

	// Handle scaling factors
	s_ptr := raw_data(S) if S != nil else nil

	n_err_bnds_i := Blas_Int(n_err_bnds)

	lapack.zherfsx_(
		uplo_c,
		equed_c,
		&n,
		&nrhs,
		raw_data(A.data),
		&lda,
		raw_data(AF.data),
		&ldaf,
		raw_data(ipiv),
		s_ptr,
		raw_data(B.data),
		&ldb,
		raw_data(X.data),
		&ldx,
		&rcond,
		raw_data(berr),
		&n_err_bnds_i,
		raw_data(err_bnds_norm),
		raw_data(err_bnds_comp),
		&nparams,
		nil, // Use default parameters
		raw_data(work),
		raw_data(rwork),
		&info,
		1,
		1,
	)

	return rcond, berr, err_bnds_norm, err_bnds_comp, info
}

// ===================================================================================
// HERMITIAN DIRECT SOLVERS (ONE-STEP FACTORIZATION + SOLVE)
// ===================================================================================
// Complete solution of Hermitian systems A*X = B in one call

// Basic Hermitian direct solver using Bunch-Kaufman factorization
m_solve_hermitian_direct :: proc {
	m_solve_hermitian_direct_c64,
	m_solve_hermitian_direct_c128,
}

m_solve_hermitian_direct_c64 :: proc(
	A: ^Matrix(complex64), // Hermitian matrix (overwritten with factorization)
	B: ^Matrix(complex64), // Right-hand side matrix (overwritten with solution)
	compute_upper := true, // true for upper triangle, false for lower
	allocator := context.allocator,
) -> (
	ipiv: []Blas_Int,
	info: Info, // Pivot indices from factorization
) {
	assert(
		A.format == .General || A.format == .Hermitian,
		"Matrix A must be general or Hermitian format",
	)
	assert(A.rows == A.cols, "Matrix A must be square")
	assert(B.rows == A.rows, "Matrix B must have same number of rows as A")

	n := A.rows
	nrhs := B.cols
	lda := A.ld
	ldb := B.ld

	uplo_c := cstring("U") if compute_upper else cstring("L")

	// Allocate pivot array
	ipiv = make([]Blas_Int, n, allocator)

	// Workspace query
	work_query: complex64
	lwork: Blas_Int = -1

	lapack.chesv_(
		uplo_c,
		&n,
		&nrhs,
		raw_data(A.data),
		&lda,
		raw_data(ipiv),
		raw_data(B.data),
		&ldb,
		&work_query,
		&lwork,
		&info,
		1,
	)

	// Allocate workspace
	lwork = auto_cast real(work_query)
	work := make([]complex64, lwork, allocator)

	lapack.chesv_(
		uplo_c,
		&n,
		&nrhs,
		raw_data(A.data),
		&lda,
		raw_data(ipiv),
		raw_data(B.data),
		&ldb,
		raw_data(work),
		&lwork,
		&info,
		1,
	)

	return ipiv, info
}

m_solve_hermitian_direct_c128 :: proc(
	A: ^Matrix(complex128), // Hermitian matrix (overwritten with factorization)
	B: ^Matrix(complex128), // Right-hand side matrix (overwritten with solution)
	compute_upper := true, // true for upper triangle, false for lower
	allocator := context.allocator,
) -> (
	ipiv: []Blas_Int,
	info: Info, // Pivot indices from factorization
) {
	assert(
		A.format == .General || A.format == .Hermitian,
		"Matrix A must be general or Hermitian format",
	)
	assert(A.rows == A.cols, "Matrix A must be square")
	assert(B.rows == A.rows, "Matrix B must have same number of rows as A")

	n := A.rows
	nrhs := B.cols
	lda := A.ld
	ldb := B.ld

	uplo_c := cstring("U") if compute_upper else cstring("L")

	// Allocate pivot array
	ipiv = make([]Blas_Int, n, allocator)

	// Workspace query
	work_query: complex128
	lwork: Blas_Int = -1

	lapack.zhesv_(
		uplo_c,
		&n,
		&nrhs,
		raw_data(A.data),
		&lda,
		raw_data(ipiv),
		raw_data(B.data),
		&ldb,
		&work_query,
		&lwork,
		&info,
		1,
	)

	// Allocate workspace
	lwork = auto_cast real(work_query)
	work := make([]complex128, lwork, allocator)

	lapack.zhesv_(
		uplo_c,
		&n,
		&nrhs,
		raw_data(A.data),
		&lda,
		raw_data(ipiv),
		raw_data(B.data),
		&ldb,
		raw_data(work),
		&lwork,
		&info,
		1,
	)

	return ipiv, info
}

// Aasen's algorithm direct solver (improved stability)
m_solve_hermitian_aasen :: proc {
	m_solve_hermitian_aasen_c64,
	m_solve_hermitian_aasen_c128,
}

m_solve_hermitian_aasen_c64 :: proc(
	A: ^Matrix(complex64), // Hermitian matrix (overwritten with factorization)
	B: ^Matrix(complex64), // Right-hand side matrix (overwritten with solution)
	compute_upper := true, // true for upper triangle, false for lower
	allocator := context.allocator,
) -> (
	ipiv: []Blas_Int,
	info: Info, // Pivot indices from factorization
) {
	assert(
		A.format == .General || A.format == .Hermitian,
		"Matrix A must be general or Hermitian format",
	)
	assert(A.rows == A.cols, "Matrix A must be square")
	assert(B.rows == A.rows, "Matrix B must have same number of rows as A")

	n := A.rows
	nrhs := B.cols
	lda := A.ld
	ldb := B.ld

	uplo_c := cstring("U") if compute_upper else cstring("L")

	// Allocate pivot array
	ipiv = make([]Blas_Int, n, allocator)

	// Workspace query
	work_query: complex64
	lwork: Blas_Int = -1

	lapack.chesv_aa_(
		uplo_c,
		&n,
		&nrhs,
		raw_data(A.data),
		&lda,
		raw_data(ipiv),
		raw_data(B.data),
		&ldb,
		&work_query,
		&lwork,
		&info,
		1,
	)

	// Allocate workspace
	lwork = auto_cast real(work_query)
	work := make([]complex64, lwork, allocator)

	lapack.chesv_aa_(
		uplo_c,
		&n,
		&nrhs,
		raw_data(A.data),
		&lda,
		raw_data(ipiv),
		raw_data(B.data),
		&ldb,
		raw_data(work),
		&lwork,
		&info,
		1,
	)

	return ipiv, info
}

m_solve_hermitian_aasen_c128 :: proc(
	A: ^Matrix(complex128), // Hermitian matrix (overwritten with factorization)
	B: ^Matrix(complex128), // Right-hand side matrix (overwritten with solution)
	compute_upper := true, // true for upper triangle, false for lower
	allocator := context.allocator,
) -> (
	ipiv: []Blas_Int,
	info: Info, // Pivot indices from factorization
) {
	assert(
		A.format == .General || A.format == .Hermitian,
		"Matrix A must be general or Hermitian format",
	)
	assert(A.rows == A.cols, "Matrix A must be square")
	assert(B.rows == A.rows, "Matrix B must have same number of rows as A")

	n := A.rows
	nrhs := B.cols
	lda := A.ld
	ldb := B.ld

	uplo_c := cstring("U") if compute_upper else cstring("L")

	// Allocate pivot array
	ipiv = make([]Blas_Int, n, allocator)

	// Workspace query
	work_query: complex128
	lwork: Blas_Int = -1

	lapack.zhesv_aa_(
		uplo_c,
		&n,
		&nrhs,
		raw_data(A.data),
		&lda,
		raw_data(ipiv),
		raw_data(B.data),
		&ldb,
		&work_query,
		&lwork,
		&info,
		1,
	)

	// Allocate workspace
	lwork = auto_cast real(work_query)
	work := make([]complex128, lwork, allocator)

	lapack.zhesv_aa_(
		uplo_c,
		&n,
		&nrhs,
		raw_data(A.data),
		&lda,
		raw_data(ipiv),
		raw_data(B.data),
		&ldb,
		raw_data(work),
		&lwork,
		&info,
		1,
	)

	return ipiv, info
}

// Aasen's 2-stage algorithm (enhanced performance for large matrices)
m_solve_hermitian_aasen_2stage :: proc {
	m_solve_hermitian_aasen_2stage_c64,
	m_solve_hermitian_aasen_2stage_c128,
}

m_solve_hermitian_aasen_2stage_c64 :: proc(
	A: ^Matrix(complex64), // Hermitian matrix (overwritten with factorization)
	B: ^Matrix(complex64), // Right-hand side matrix (overwritten with solution)
	compute_upper := true, // true for upper triangle, false for lower
	allocator := context.allocator,
) -> (
	TB: Matrix(complex64),
	ipiv: []Blas_Int,
	ipiv2: []Blas_Int,
	info: Info, // Block reflector from 2-stage factorization// Primary pivot indices// Secondary pivot indices from 2-stage
) {
	assert(
		A.format == .General || A.format == .Hermitian,
		"Matrix A must be general or Hermitian format",
	)
	assert(A.rows == A.cols, "Matrix A must be square")
	assert(B.rows == A.rows, "Matrix B must have same number of rows as A")

	n := A.rows
	nrhs := B.cols
	lda := A.ld
	ldb := B.ld

	uplo_c := cstring("U") if compute_upper else cstring("L")

	// Allocate pivot arrays
	ipiv = make([]Blas_Int, n, allocator)
	ipiv2 = make([]Blas_Int, n, allocator)

	// TB matrix size is typically n*nb where nb is the block size
	// We'll use a conservative estimate
	nb := min(64, n) // Typical block size
	ltb := 4 * n * nb
	TB = make_matrix(complex64, ltb, 1, .General, allocator)

	// Workspace query
	work_query: complex64
	lwork: Blas_Int = -1

	ltb_i := Blas_Int(ltb)

	lapack.chesv_aa_2stage_(
		uplo_c,
		&n,
		&nrhs,
		raw_data(A.data),
		&lda,
		raw_data(TB.data),
		&ltb_i,
		raw_data(ipiv),
		raw_data(ipiv2),
		raw_data(B.data),
		&ldb,
		&work_query,
		&lwork,
		&info,
		1,
	)

	// Allocate workspace
	lwork = auto_cast real(work_query)
	work := make([]complex64, lwork, allocator)

	lapack.chesv_aa_2stage_(
		uplo_c,
		&n,
		&nrhs,
		raw_data(A.data),
		&lda,
		raw_data(TB.data),
		&ltb_i,
		raw_data(ipiv),
		raw_data(ipiv2),
		raw_data(B.data),
		&ldb,
		raw_data(work),
		&lwork,
		&info,
		1,
	)

	return TB, ipiv, ipiv2, info
}

m_solve_hermitian_aasen_2stage_c128 :: proc(
	A: ^Matrix(complex128), // Hermitian matrix (overwritten with factorization)
	B: ^Matrix(complex128), // Right-hand side matrix (overwritten with solution)
	compute_upper := true, // true for upper triangle, false for lower
	allocator := context.allocator,
) -> (
	TB: Matrix(complex128),
	ipiv: []Blas_Int,
	ipiv2: []Blas_Int,
	info: Info, // Block reflector from 2-stage factorization// Primary pivot indices// Secondary pivot indices from 2-stage
) {
	assert(
		A.format == .General || A.format == .Hermitian,
		"Matrix A must be general or Hermitian format",
	)
	assert(A.rows == A.cols, "Matrix A must be square")
	assert(B.rows == A.rows, "Matrix B must have same number of rows as A")

	n := A.rows
	nrhs := B.cols
	lda := A.ld
	ldb := B.ld

	uplo_c := cstring("U") if compute_upper else cstring("L")

	// Allocate pivot arrays
	ipiv = make([]Blas_Int, n, allocator)
	ipiv2 = make([]Blas_Int, n, allocator)

	// TB matrix size is typically n*nb where nb is the block size
	// We'll use a conservative estimate
	nb := min(64, n) // Typical block size
	ltb := 4 * n * nb
	TB = make_matrix(complex128, ltb, 1, .General, allocator)

	// Workspace query
	work_query: complex128
	lwork: Blas_Int = -1

	ltb_i := Blas_Int(ltb)

	lapack.zhesv_aa_2stage_(
		uplo_c,
		&n,
		&nrhs,
		raw_data(A.data),
		&lda,
		raw_data(TB.data),
		&ltb_i,
		raw_data(ipiv),
		raw_data(ipiv2),
		raw_data(B.data),
		&ldb,
		&work_query,
		&lwork,
		&info,
		1,
	)

	// Allocate workspace
	lwork = auto_cast real(work_query)
	work := make([]complex128, lwork, allocator)

	lapack.zhesv_aa_2stage_(
		uplo_c,
		&n,
		&nrhs,
		raw_data(A.data),
		&lda,
		raw_data(TB.data),
		&ltb_i,
		raw_data(ipiv),
		raw_data(ipiv2),
		raw_data(B.data),
		&ldb,
		raw_data(work),
		&lwork,
		&info,
		1,
	)

	return TB, ipiv, ipiv2, info
}

// Bounded Bunch-Kaufman with Rook pivoting (enhanced numerical stability)
m_solve_hermitian_rook :: proc {
	m_solve_hermitian_rook_c64,
	m_solve_hermitian_rook_c128,
}

m_solve_hermitian_rook_c64 :: proc(
	A: ^Matrix(complex64), // Hermitian matrix (overwritten with factorization)
	B: ^Matrix(complex64), // Right-hand side matrix (overwritten with solution)
	compute_upper := true, // true for upper triangle, false for lower
	allocator := context.allocator,
) -> (
	ipiv: []Blas_Int,
	info: Info, // Pivot indices from factorization
) {
	assert(
		A.format == .General || A.format == .Hermitian,
		"Matrix A must be general or Hermitian format",
	)
	assert(A.rows == A.cols, "Matrix A must be square")
	assert(B.rows == A.rows, "Matrix B must have same number of rows as A")

	n := A.rows
	nrhs := B.cols
	lda := A.ld
	ldb := B.ld

	uplo_c := cstring("U") if compute_upper else cstring("L")

	// Allocate pivot array
	ipiv = make([]Blas_Int, n, allocator)

	// Workspace query
	work_query: complex64
	lwork: Blas_Int = -1

	lapack.chesv_rook_(
		uplo_c,
		&n,
		&nrhs,
		raw_data(A.data),
		&lda,
		raw_data(ipiv),
		raw_data(B.data),
		&ldb,
		&work_query,
		&lwork,
		&info,
		1,
	)

	// Allocate workspace
	lwork = auto_cast real(work_query)
	work := make([]complex64, lwork, allocator)

	lapack.chesv_rook_(
		uplo_c,
		&n,
		&nrhs,
		raw_data(A.data),
		&lda,
		raw_data(ipiv),
		raw_data(B.data),
		&ldb,
		raw_data(work),
		&lwork,
		&info,
		1,
	)

	return ipiv, info
}

m_solve_hermitian_rook_c128 :: proc(
	A: ^Matrix(complex128), // Hermitian matrix (overwritten with factorization)
	B: ^Matrix(complex128), // Right-hand side matrix (overwritten with solution)
	compute_upper := true, // true for upper triangle, false for lower
	allocator := context.allocator,
) -> (
	ipiv: []Blas_Int,
	info: Info, // Pivot indices from factorization
) {
	assert(
		A.format == .General || A.format == .Hermitian,
		"Matrix A must be general or Hermitian format",
	)
	assert(A.rows == A.cols, "Matrix A must be square")
	assert(B.rows == A.rows, "Matrix B must have same number of rows as A")

	n := A.rows
	nrhs := B.cols
	lda := A.ld
	ldb := B.ld

	uplo_c := cstring("U") if compute_upper else cstring("L")

	// Allocate pivot array
	ipiv = make([]Blas_Int, n, allocator)

	// Workspace query
	work_query: complex128
	lwork: Blas_Int = -1

	lapack.zhesv_rook_(
		uplo_c,
		&n,
		&nrhs,
		raw_data(A.data),
		&lda,
		raw_data(ipiv),
		raw_data(B.data),
		&ldb,
		&work_query,
		&lwork,
		&info,
		1,
	)

	// Allocate workspace
	lwork = auto_cast real(work_query)
	work := make([]complex128, lwork, allocator)

	lapack.zhesv_rook_(
		uplo_c,
		&n,
		&nrhs,
		raw_data(A.data),
		&lda,
		raw_data(ipiv),
		raw_data(B.data),
		&ldb,
		raw_data(work),
		&lwork,
		&info,
		1,
	)

	return ipiv, info
}

// Bounded Bunch-Kaufman with rank-revealing (for singular or ill-conditioned matrices)
m_solve_hermitian_rank_revealing :: proc {
	m_solve_hermitian_rank_revealing_c64,
	m_solve_hermitian_rank_revealing_c128,
}

m_solve_hermitian_rank_revealing_c64 :: proc(
	A: ^Matrix(complex64), // Hermitian matrix (overwritten with factorization)
	B: ^Matrix(complex64), // Right-hand side matrix (overwritten with solution)
	compute_upper := true, // true for upper triangle, false for lower
	allocator := context.allocator,
) -> (
	E: Matrix(complex64),
	ipiv: []Blas_Int,
	info: Info, // Block diagonal matrix E from factorization// Pivot indices from factorization
) {
	assert(
		A.format == .General || A.format == .Hermitian,
		"Matrix A must be general or Hermitian format",
	)
	assert(A.rows == A.cols, "Matrix A must be square")
	assert(B.rows == A.rows, "Matrix B must have same number of rows as A")

	n := A.rows
	nrhs := B.cols
	lda := A.ld
	ldb := B.ld

	uplo_c := cstring("U") if compute_upper else cstring("L")

	// Allocate pivot array and E matrix
	ipiv = make([]Blas_Int, n, allocator)
	E = make_matrix(complex64, n, n, .General, allocator)

	// Workspace query
	work_query: complex64
	lwork: Blas_Int = -1

	lapack.chesv_rk_(
		uplo_c,
		&n,
		&nrhs,
		raw_data(A.data),
		&lda,
		raw_data(E.data),
		raw_data(ipiv),
		raw_data(B.data),
		&ldb,
		&work_query,
		&lwork,
		&info,
		1,
	)

	// Allocate workspace
	lwork = auto_cast real(work_query)
	work := make([]complex64, lwork, allocator)

	lapack.chesv_rk_(
		uplo_c,
		&n,
		&nrhs,
		raw_data(A.data),
		&lda,
		raw_data(E.data),
		raw_data(ipiv),
		raw_data(B.data),
		&ldb,
		raw_data(work),
		&lwork,
		&info,
		1,
	)

	return E, ipiv, info
}

m_solve_hermitian_rank_revealing_c128 :: proc(
	A: ^Matrix(complex128), // Hermitian matrix (overwritten with factorization)
	B: ^Matrix(complex128), // Right-hand side matrix (overwritten with solution)
	compute_upper := true, // true for upper triangle, false for lower
	allocator := context.allocator,
) -> (
	E: Matrix(complex128),
	ipiv: []Blas_Int,
	info: Info, // Block diagonal matrix E from factorization// Pivot indices from factorization
) {
	assert(
		A.format == .General || A.format == .Hermitian,
		"Matrix A must be general or Hermitian format",
	)
	assert(A.rows == A.cols, "Matrix A must be square")
	assert(B.rows == A.rows, "Matrix B must have same number of rows as A")

	n := A.rows
	nrhs := B.cols
	lda := A.ld
	ldb := B.ld

	uplo_c := cstring("U") if compute_upper else cstring("L")

	// Allocate pivot array and E matrix
	ipiv = make([]Blas_Int, n, allocator)
	E = make_matrix(complex128, n, n, .General, allocator)

	// Workspace query
	work_query: complex128
	lwork: Blas_Int = -1

	lapack.zhesv_rk_(
		uplo_c,
		&n,
		&nrhs,
		raw_data(A.data),
		&lda,
		raw_data(E.data),
		raw_data(ipiv),
		raw_data(B.data),
		&ldb,
		&work_query,
		&lwork,
		&info,
		1,
	)

	// Allocate workspace
	lwork = auto_cast real(work_query)
	work := make([]complex128, lwork, allocator)

	lapack.zhesv_rk_(
		uplo_c,
		&n,
		&nrhs,
		raw_data(A.data),
		&lda,
		raw_data(E.data),
		raw_data(ipiv),
		raw_data(B.data),
		&ldb,
		raw_data(work),
		&lwork,
		&info,
		1,
	)

	return E, ipiv, info
}

// ===================================================================================
// HERMITIAN EXPERT SOLVERS
// ===================================================================================
// Advanced solvers with condition estimation, error bounds, and factorization control

FactorizationMode :: enum {
	Auto, // Automatic factorization (F='N')
	Factorized, // Use pre-factorized matrix (F='F')
	Equilibrated, // Use equilibrated pre-factorized matrix (F='E')
}

// Expert Hermitian solver with condition estimation and error bounds
m_solve_hermitian_expert :: proc {
	m_solve_hermitian_expert_c64,
	m_solve_hermitian_expert_c128,
}

m_solve_hermitian_expert_c64 :: proc(
	A: ^Matrix(complex64), // Original Hermitian matrix (input)
	AF: ^Matrix(complex64), // Factorized matrix (input/output)
	ipiv: []Blas_Int, // Pivot indices (input/output)
	B: ^Matrix(complex64), // Right-hand side matrix (input)
	X: ^Matrix(complex64), // Solution matrix (output)
	fact_mode: FactorizationMode = .Auto, // Factorization control
	compute_upper := true, // true for upper triangle, false for lower
	allocator := context.allocator,
) -> (
	rcond: f32,
	ferr: []f32,
	berr: []f32,
	info: Info, // Reciprocal condition number estimate// Forward error bounds for each RHS// Backward error bounds for each RHS
) {
	assert(
		A.format == .General || A.format == .Hermitian,
		"Matrix A must be general or Hermitian format",
	)
	assert(
		AF.format == .General || AF.format == .Hermitian,
		"Matrix AF must be general or Hermitian format",
	)
	assert(A.rows == A.cols, "Matrix A must be square")
	assert(AF.rows == AF.cols, "Matrix AF must be square")
	assert(A.rows == AF.rows, "Matrices A and AF must have same dimensions")
	assert(B.rows == A.rows, "Matrix B must have same number of rows as A")
	assert(X.rows == A.rows && X.cols == B.cols, "Matrix X must have same dimensions as B")

	n := A.rows
	nrhs := B.cols
	lda := A.ld
	ldaf := AF.ld
	ldb := B.ld
	ldx := X.ld

	uplo_c := cstring("U") if compute_upper else cstring("L")

	// Factorization mode
	fact_c: cstring
	switch fact_mode {
	case .Auto:
		fact_c = cstring("N")
	case .Factorized:
		fact_c = cstring("F")
	case .Equilibrated:
		fact_c = cstring("E")
	}

	// Allocate outputs
	ferr = make([]f32, nrhs, allocator)
	berr = make([]f32, nrhs, allocator)

	// Workspace query
	work_query: complex64
	lwork: Blas_Int = -1

	lapack.chesvx_(
		fact_c,
		uplo_c,
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
		&rcond,
		raw_data(ferr),
		raw_data(berr),
		&work_query,
		&lwork,
		nil, // rwork
		&info,
		1,
		1,
	)

	// Allocate workspace
	lwork = auto_cast real(work_query)
	work := make([]complex64, lwork, allocator)
	rwork := make([]f32, n, allocator)

	lapack.chesvx_(
		fact_c,
		uplo_c,
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
		&rcond,
		raw_data(ferr),
		raw_data(berr),
		raw_data(work),
		&lwork,
		raw_data(rwork),
		&info,
		1,
		1,
	)

	return rcond, ferr, berr, info
}

m_solve_hermitian_expert_c128 :: proc(
	A: ^Matrix(complex128), // Original Hermitian matrix (input)
	AF: ^Matrix(complex128), // Factorized matrix (input/output)
	ipiv: []Blas_Int, // Pivot indices (input/output)
	B: ^Matrix(complex128), // Right-hand side matrix (input)
	X: ^Matrix(complex128), // Solution matrix (output)
	fact_mode: FactorizationMode = .Auto, // Factorization control
	compute_upper := true, // true for upper triangle, false for lower
	allocator := context.allocator,
) -> (
	rcond: f64,
	ferr: []f64,
	berr: []f64,
	info: Info, // Reciprocal condition number estimate// Forward error bounds for each RHS// Backward error bounds for each RHS
) {
	assert(
		A.format == .General || A.format == .Hermitian,
		"Matrix A must be general or Hermitian format",
	)
	assert(
		AF.format == .General || AF.format == .Hermitian,
		"Matrix AF must be general or Hermitian format",
	)
	assert(A.rows == A.cols, "Matrix A must be square")
	assert(AF.rows == AF.cols, "Matrix AF must be square")
	assert(A.rows == AF.rows, "Matrices A and AF must have same dimensions")
	assert(B.rows == A.rows, "Matrix B must have same number of rows as A")
	assert(X.rows == A.rows && X.cols == B.cols, "Matrix X must have same dimensions as B")

	n := A.rows
	nrhs := B.cols
	lda := A.ld
	ldaf := AF.ld
	ldb := B.ld
	ldx := X.ld

	uplo_c := cstring("U") if compute_upper else cstring("L")

	// Factorization mode
	fact_c: cstring
	switch fact_mode {
	case .Auto:
		fact_c = cstring("N")
	case .Factorized:
		fact_c = cstring("F")
	case .Equilibrated:
		fact_c = cstring("E")
	}

	// Allocate outputs
	ferr = make([]f64, nrhs, allocator)
	berr = make([]f64, nrhs, allocator)

	// Workspace query
	work_query: complex128
	lwork: Blas_Int = -1

	lapack.zhesvx_(
		fact_c,
		uplo_c,
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
		&rcond,
		raw_data(ferr),
		raw_data(berr),
		&work_query,
		&lwork,
		nil, // rwork
		&info,
		1,
		1,
	)

	// Allocate workspace
	lwork = auto_cast real(work_query)
	work := make([]complex128, lwork, allocator)
	rwork := make([]f64, n, allocator)

	lapack.zhesvx_(
		fact_c,
		uplo_c,
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
		&rcond,
		raw_data(ferr),
		raw_data(berr),
		raw_data(work),
		&lwork,
		raw_data(rwork),
		&info,
		1,
		1,
	)

	return rcond, ferr, berr, info
}

// Enhanced expert solver with equilibration and extended error bounds
m_solve_hermitian_expert_enhanced :: proc {
	m_solve_hermitian_expert_enhanced_c64,
	m_solve_hermitian_expert_enhanced_c128,
}

m_solve_hermitian_expert_enhanced_c64 :: proc(
	A: ^Matrix(complex64), // Original Hermitian matrix (input)
	AF: ^Matrix(complex64), // Factorized matrix (input/output)
	ipiv: []Blas_Int, // Pivot indices (input/output)
	S: []f32, // Scaling factors (input/output, can be nil)
	B: ^Matrix(complex64), // Right-hand side matrix (input)
	X: ^Matrix(complex64), // Solution matrix (output)
	fact_mode: FactorizationMode = .Auto, // Factorization control
	compute_upper := true, // true for upper triangle, false for lower
	n_err_bnds: int = 3, // Number of error bounds to compute (default 3)
	allocator := context.allocator,
) -> (
	equed: bool,
	rcond: f32,
	rpvgrw: f32,
	berr: []f32,
	err_bnds_norm: []f32,// Whether matrix was equilibrated
	err_bnds_comp: []f32,// Reciprocal condition number estimate
	info: Info, // Reciprocal pivot growth factor// Backward error bounds for each RHS// Normwise error bounds// Componentwise error bounds
) {
	assert(
		A.format == .General || A.format == .Hermitian,
		"Matrix A must be general or Hermitian format",
	)
	assert(
		AF.format == .General || AF.format == .Hermitian,
		"Matrix AF must be general or Hermitian format",
	)
	assert(A.rows == A.cols, "Matrix A must be square")
	assert(AF.rows == AF.cols, "Matrix AF must be square")
	assert(A.rows == AF.rows, "Matrices A and AF must have same dimensions")
	assert(B.rows == A.rows, "Matrix B must have same number of rows as A")
	assert(X.rows == A.rows && X.cols == B.cols, "Matrix X must have same dimensions as B")

	n := A.rows
	nrhs := B.cols
	lda := A.ld
	ldaf := AF.ld
	ldb := B.ld
	ldx := X.ld

	uplo_c := cstring("U") if compute_upper else cstring("L")

	// Factorization mode
	fact_c: cstring
	switch fact_mode {
	case .Auto:
		fact_c = cstring("N")
	case .Factorized:
		fact_c = cstring("F")
	case .Equilibrated:
		fact_c = cstring("E")
	}

	// Allocate outputs
	equed_c := make([]u8, 2) // For EQUED output
	berr = make([]f32, nrhs, allocator)
	err_bnds_norm = make([]f32, nrhs * n_err_bnds, allocator)
	err_bnds_comp = make([]f32, nrhs * n_err_bnds, allocator)

	// Default parameters for enhanced solver
	nparams := Blas_Int(0) // Use default parameters
	params: []f32 = nil

	// Workspace allocation (enhanced solver needs more workspace)
	lwork := 2 * n
	work := make([]complex64, lwork, allocator)
	rwork := make([]f32, 3 * n, allocator)

	// Handle scaling factors
	s_ptr := raw_data(S) if S != nil else nil

	n_err_bnds_i := Blas_Int(n_err_bnds)
	lwork_i := Blas_Int(lwork)

	lapack.chesvxx_(
		fact_c,
		uplo_c,
		&n,
		&nrhs,
		raw_data(A.data),
		&lda,
		raw_data(AF.data),
		&ldaf,
		raw_data(ipiv),
		raw_data(equed_c),
		s_ptr,
		raw_data(B.data),
		&ldb,
		raw_data(X.data),
		&ldx,
		&rcond,
		&rpvgrw,
		raw_data(berr),
		&n_err_bnds_i,
		raw_data(err_bnds_norm),
		raw_data(err_bnds_comp),
		&nparams,
		nil, // Use default parameters
		raw_data(work),
		raw_data(rwork),
		&info,
		1,
		1,
		1,
	)

	// Convert equilibration flag
	equed = equed_c[0] == 'Y'

	return equed, rcond, rpvgrw, berr, err_bnds_norm, err_bnds_comp, info
}

m_solve_hermitian_expert_enhanced_c128 :: proc(
	A: ^Matrix(complex128), // Original Hermitian matrix (input)
	AF: ^Matrix(complex128), // Factorized matrix (input/output)
	ipiv: []Blas_Int, // Pivot indices (input/output)
	S: []f64, // Scaling factors (input/output, can be nil)
	B: ^Matrix(complex128), // Right-hand side matrix (input)
	X: ^Matrix(complex128), // Solution matrix (output)
	fact_mode: FactorizationMode = .Auto, // Factorization control
	compute_upper := true, // true for upper triangle, false for lower
	n_err_bnds: int = 3, // Number of error bounds to compute (default 3)
	allocator := context.allocator,
) -> (
	equed: bool,
	rcond: f64,
	rpvgrw: f64,
	berr: []f64,
	err_bnds_norm: []f64,// Whether matrix was equilibrated
	err_bnds_comp: []f64,// Reciprocal condition number estimate
	info: Info, // Reciprocal pivot growth factor// Backward error bounds for each RHS// Normwise error bounds// Componentwise error bounds
) {
	assert(
		A.format == .General || A.format == .Hermitian,
		"Matrix A must be general or Hermitian format",
	)
	assert(
		AF.format == .General || AF.format == .Hermitian,
		"Matrix AF must be general or Hermitian format",
	)
	assert(A.rows == A.cols, "Matrix A must be square")
	assert(AF.rows == AF.cols, "Matrix AF must be square")
	assert(A.rows == AF.rows, "Matrices A and AF must have same dimensions")
	assert(B.rows == A.rows, "Matrix B must have same number of rows as A")
	assert(X.rows == A.rows && X.cols == B.cols, "Matrix X must have same dimensions as B")

	n := A.rows
	nrhs := B.cols
	lda := A.ld
	ldaf := AF.ld
	ldb := B.ld
	ldx := X.ld

	uplo_c := cstring("U") if compute_upper else cstring("L")

	// Factorization mode
	fact_c: cstring
	switch fact_mode {
	case .Auto:
		fact_c = cstring("N")
	case .Factorized:
		fact_c = cstring("F")
	case .Equilibrated:
		fact_c = cstring("E")
	}

	// Allocate outputs
	equed_c := make([]u8, 2) // For EQUED output
	berr = make([]f64, nrhs, allocator)
	err_bnds_norm = make([]f64, nrhs * n_err_bnds, allocator)
	err_bnds_comp = make([]f64, nrhs * n_err_bnds, allocator)

	// Default parameters for enhanced solver
	nparams := Blas_Int(0) // Use default parameters
	params: []f64 = nil

	// Workspace allocation (enhanced solver needs more workspace)
	lwork := 2 * n
	work := make([]complex128, lwork, allocator)
	rwork := make([]f64, 3 * n, allocator)

	// Handle scaling factors
	s_ptr := raw_data(S) if S != nil else nil

	n_err_bnds_i := Blas_Int(n_err_bnds)
	lwork_i := Blas_Int(lwork)

	lapack.zhesvxx_(
		fact_c,
		uplo_c,
		&n,
		&nrhs,
		raw_data(A.data),
		&lda,
		raw_data(AF.data),
		&ldaf,
		raw_data(ipiv),
		raw_data(equed_c),
		s_ptr,
		raw_data(B.data),
		&ldb,
		raw_data(X.data),
		&ldx,
		&rcond,
		&rpvgrw,
		raw_data(berr),
		&n_err_bnds_i,
		raw_data(err_bnds_norm),
		raw_data(err_bnds_comp),
		&nparams,
		nil, // Use default parameters
		raw_data(work),
		raw_data(rwork),
		&info,
		1,
		1,
		1,
	)

	// Convert equilibration flag
	equed = equed_c[0] == 'Y'

	return equed, rcond, rpvgrw, berr, err_bnds_norm, err_bnds_comp, info
}

// ===================================================================================
// HERMITIAN UTILITY FUNCTIONS
// ===================================================================================

// Swap rows/columns in Hermitian matrix (maintaining Hermitian property)
m_swap_hermitian :: proc {
	m_swap_hermitian_c64,
	m_swap_hermitian_c128,
}

m_swap_hermitian_c64 :: proc(
	A: ^Matrix(complex64), // Hermitian matrix to modify
	i1, i2: int, // Row/column indices to swap (0-based)
	compute_upper := true, // true for upper triangle, false for lower
) {
	assert(
		A.format == .General || A.format == .Hermitian,
		"Matrix A must be general or Hermitian format",
	)
	assert(A.rows == A.cols, "Matrix A must be square")
	assert(i1 >= 0 && i1 < A.rows, "Index i1 must be valid")
	assert(i2 >= 0 && i2 < A.rows, "Index i2 must be valid")

	n := A.rows
	lda := A.ld

	uplo_c := cstring("U") if compute_upper else cstring("L")

	// Convert to 1-based indices for LAPACK
	i1_f := Blas_Int(i1 + 1)
	i2_f := Blas_Int(i2 + 1)

	lapack.cheswapr_(uplo_c, &n, raw_data(A.data), &lda, &i1_f, &i2_f, 1)
}

m_swap_hermitian_c128 :: proc(
	A: ^Matrix(complex128), // Hermitian matrix to modify
	i1, i2: int, // Row/column indices to swap (0-based)
	compute_upper := true, // true for upper triangle, false for lower
) {
	assert(
		A.format == .General || A.format == .Hermitian,
		"Matrix A must be general or Hermitian format",
	)
	assert(A.rows == A.cols, "Matrix A must be square")
	assert(i1 >= 0 && i1 < A.rows, "Index i1 must be valid")
	assert(i2 >= 0 && i2 < A.rows, "Index i2 must be valid")

	n := A.rows
	lda := A.ld

	uplo_c := cstring("U") if compute_upper else cstring("L")

	// Convert to 1-based indices for LAPACK
	i1_f := Blas_Int(i1 + 1)
	i2_f := Blas_Int(i2 + 1)

	lapack.zheswapr_(uplo_c, &n, raw_data(A.data), &lda, &i1_f, &i2_f, 1)
}

// ===================================================================================
// HERMITIAN MATRIX REDUCTIONS
// ===================================================================================
// Reduction to tridiagonal form for eigenvalue computations

// Reduce Hermitian matrix to tridiagonal form
m_reduce_tridiagonal_hermitian :: proc {
	m_reduce_tridiagonal_hermitian_c64,
	m_reduce_tridiagonal_hermitian_c128,
}

m_reduce_tridiagonal_hermitian_c64 :: proc(
	A: ^Matrix(complex64), // Hermitian matrix (overwritten with reduction)
	compute_upper := true, // true for upper triangle, false for lower
	allocator := context.allocator,
) -> (
	D: []f32,
	E: []f32,
	tau: []complex64,
	info: Info, // Diagonal elements of tridiagonal matrix// Off-diagonal elements of tridiagonal matrix// Scalar factors of elementary reflectors
) {
	assert(
		A.format == .General || A.format == .Hermitian,
		"Matrix A must be general or Hermitian format",
	)
	assert(A.rows == A.cols, "Matrix A must be square")

	n := A.rows
	lda := A.ld

	uplo_c := cstring("U") if compute_upper else cstring("L")

	// Allocate outputs
	D = make([]f32, n, allocator)
	E = make([]f32, n - 1, allocator) if n > 1 else make([]f32, 0, allocator)
	tau = make([]complex64, n - 1, allocator) if n > 1 else make([]complex64, 0, allocator)

	// Workspace query
	work_query: complex64
	lwork: Blas_Int = -1

	lapack.chetrd_(
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(D),
		raw_data(E),
		raw_data(tau),
		&work_query,
		&lwork,
		&info,
		1,
	)

	// Allocate workspace
	lwork = auto_cast real(work_query)
	work := make([]complex64, lwork, allocator)

	lapack.chetrd_(
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(D),
		raw_data(E),
		raw_data(tau),
		raw_data(work),
		&lwork,
		&info,
		1,
	)

	return D, E, tau, info
}

m_reduce_tridiagonal_hermitian_c128 :: proc(
	A: ^Matrix(complex128), // Hermitian matrix (overwritten with reduction)
	compute_upper := true, // true for upper triangle, false for lower
	allocator := context.allocator,
) -> (
	D: []f64,
	E: []f64,
	tau: []complex128,
	info: Info, // Diagonal elements of tridiagonal matrix// Off-diagonal elements of tridiagonal matrix// Scalar factors of elementary reflectors
) {
	assert(
		A.format == .General || A.format == .Hermitian,
		"Matrix A must be general or Hermitian format",
	)
	assert(A.rows == A.cols, "Matrix A must be square")

	n := A.rows
	lda := A.ld

	uplo_c := cstring("U") if compute_upper else cstring("L")

	// Allocate outputs
	D = make([]f64, n, allocator)
	E = make([]f64, n - 1, allocator) if n > 1 else make([]f64, 0, allocator)
	tau = make([]complex128, n - 1, allocator) if n > 1 else make([]complex128, 0, allocator)

	// Workspace query
	work_query: complex128
	lwork: Blas_Int = -1

	lapack.zhetrd_(
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(D),
		raw_data(E),
		raw_data(tau),
		&work_query,
		&lwork,
		&info,
		1,
	)

	// Allocate workspace
	lwork = auto_cast real(work_query)
	work := make([]complex128, lwork, allocator)

	lapack.zhetrd_(
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(D),
		raw_data(E),
		raw_data(tau),
		raw_data(work),
		&lwork,
		&info,
		1,
	)

	return D, E, tau, info
}

// 2-stage reduction to tridiagonal form (enhanced performance for large matrices)
m_reduce_tridiagonal_hermitian_2stage :: proc {
	m_reduce_tridiagonal_hermitian_2stage_c64,
	m_reduce_tridiagonal_hermitian_2stage_c128,
}

m_reduce_tridiagonal_hermitian_2stage_c64 :: proc(
	A: ^Matrix(complex64), // Hermitian matrix (overwritten with reduction)
	compute_vectors := true, // Whether to compute transformation vectors
	compute_upper := true, // true for upper triangle, false for lower
	allocator := context.allocator,
) -> (
	D: []f32,
	E: []f32,
	tau: []complex64,
	HOUS2: Matrix(complex64),
	info: Info, // Diagonal elements of tridiagonal matrix// Off-diagonal elements of tridiagonal matrix// Scalar factors of elementary reflectors// Householder vectors from stage 2
) {
	assert(
		A.format == .General || A.format == .Hermitian,
		"Matrix A must be general or Hermitian format",
	)
	assert(A.rows == A.cols, "Matrix A must be square")

	n := A.rows
	lda := A.ld

	uplo_c := cstring("U") if compute_upper else cstring("L")
	vect_c := cstring("V") if compute_vectors else cstring("N")

	// Allocate outputs
	D = make([]f32, n, allocator)
	E = make([]f32, n - 1, allocator) if n > 1 else make([]f32, 0, allocator)
	tau = make([]complex64, n - 1, allocator) if n > 1 else make([]complex64, 0, allocator)

	// HOUS2 matrix size estimation (typically n*nb where nb is block size)
	nb := min(64, n) // Conservative block size estimate
	lhous2 := max(1, 4 * n * nb)
	HOUS2 = make_matrix(complex64, lhous2, 1, .General, allocator)

	// Workspace query
	work_query: complex64
	lwork: Blas_Int = -1
	lhous2_i := Blas_Int(lhous2)

	lapack.chetrd_2stage_(
		vect_c,
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(D),
		raw_data(E),
		raw_data(tau),
		raw_data(HOUS2.data),
		&lhous2_i,
		&work_query,
		&lwork,
		&info,
		1,
		1,
	)

	// Allocate workspace
	lwork = auto_cast real(work_query)
	work := make([]complex64, lwork, allocator)

	lapack.chetrd_2stage_(
		vect_c,
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(D),
		raw_data(E),
		raw_data(tau),
		raw_data(HOUS2.data),
		&lhous2_i,
		raw_data(work),
		&lwork,
		&info,
		1,
		1,
	)

	return D, E, tau, HOUS2, info
}

m_reduce_tridiagonal_hermitian_2stage_c128 :: proc(
	A: ^Matrix(complex128), // Hermitian matrix (overwritten with reduction)
	compute_vectors := true, // Whether to compute transformation vectors
	compute_upper := true, // true for upper triangle, false for lower
	allocator := context.allocator,
) -> (
	D: []f64,
	E: []f64,
	tau: []complex128,
	HOUS2: Matrix(complex128),
	info: Info, // Diagonal elements of tridiagonal matrix// Off-diagonal elements of tridiagonal matrix// Scalar factors of elementary reflectors// Householder vectors from stage 2
) {
	assert(
		A.format == .General || A.format == .Hermitian,
		"Matrix A must be general or Hermitian format",
	)
	assert(A.rows == A.cols, "Matrix A must be square")

	n := A.rows
	lda := A.ld

	uplo_c := cstring("U") if compute_upper else cstring("L")
	vect_c := cstring("V") if compute_vectors else cstring("N")

	// Allocate outputs
	D = make([]f64, n, allocator)
	E = make([]f64, n - 1, allocator) if n > 1 else make([]f64, 0, allocator)
	tau = make([]complex128, n - 1, allocator) if n > 1 else make([]complex128, 0, allocator)

	// HOUS2 matrix size estimation (typically n*nb where nb is block size)
	nb := min(64, n) // Conservative block size estimate
	lhous2 := max(1, 4 * n * nb)
	HOUS2 = make_matrix(complex128, lhous2, 1, .General, allocator)

	// Workspace query
	work_query: complex128
	lwork: Blas_Int = -1
	lhous2_i := Blas_Int(lhous2)

	lapack.zhetrd_2stage_(
		vect_c,
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(D),
		raw_data(E),
		raw_data(tau),
		raw_data(HOUS2.data),
		&lhous2_i,
		&work_query,
		&lwork,
		&info,
		1,
		1,
	)

	// Allocate workspace
	lwork = auto_cast real(work_query)
	work := make([]complex128, lwork, allocator)

	lapack.zhetrd_2stage_(
		vect_c,
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(D),
		raw_data(E),
		raw_data(tau),
		raw_data(HOUS2.data),
		&lhous2_i,
		raw_data(work),
		&lwork,
		&info,
		1,
		1,
	)

	return D, E, tau, HOUS2, info
}

// ===================================================================================
// HERMITIAN MATRIX FACTORIZATIONS
// ===================================================================================
// Various factorization algorithms for Hermitian matrices

// Standard Bunch-Kaufman factorization
m_factor_hermitian :: proc {
	m_factor_hermitian_c64,
	m_factor_hermitian_c128,
}

m_factor_hermitian_c64 :: proc(
	A: ^Matrix(complex64), // Hermitian matrix (overwritten with factorization)
	compute_upper := true, // true for upper triangle, false for lower
	allocator := context.allocator,
) -> (
	ipiv: []Blas_Int,
	info: Info, // Pivot indices
) {
	assert(
		A.format == .General || A.format == .Hermitian,
		"Matrix A must be general or Hermitian format",
	)
	assert(A.rows == A.cols, "Matrix A must be square")

	n := A.rows
	lda := A.ld

	uplo_c := cstring("U") if compute_upper else cstring("L")

	// Allocate pivot array
	ipiv = make([]Blas_Int, n, allocator)

	// Workspace query
	work_query: complex64
	lwork: Blas_Int = -1

	lapack.chetrf_(
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(ipiv),
		&work_query,
		&lwork,
		&info,
		1,
	)

	// Allocate workspace
	lwork = auto_cast real(work_query)
	work := make([]complex64, lwork, allocator)

	lapack.chetrf_(
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(ipiv),
		raw_data(work),
		&lwork,
		&info,
		1,
	)

	return ipiv, info
}

m_factor_hermitian_c128 :: proc(
	A: ^Matrix(complex128), // Hermitian matrix (overwritten with factorization)
	compute_upper := true, // true for upper triangle, false for lower
	allocator := context.allocator,
) -> (
	ipiv: []Blas_Int,
	info: Info, // Pivot indices
) {
	assert(
		A.format == .General || A.format == .Hermitian,
		"Matrix A must be general or Hermitian format",
	)
	assert(A.rows == A.cols, "Matrix A must be square")

	n := A.rows
	lda := A.ld

	uplo_c := cstring("U") if compute_upper else cstring("L")

	// Allocate pivot array
	ipiv = make([]Blas_Int, n, allocator)

	// Workspace query
	work_query: complex128
	lwork: Blas_Int = -1

	lapack.zhetrf_(
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(ipiv),
		&work_query,
		&lwork,
		&info,
		1,
	)

	// Allocate workspace
	lwork = auto_cast real(work_query)
	work := make([]complex128, lwork, allocator)

	lapack.zhetrf_(
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(ipiv),
		raw_data(work),
		&lwork,
		&info,
		1,
	)

	return ipiv, info
}

// Aasen's algorithm factorization
m_factor_hermitian_aasen :: proc {
	m_factor_hermitian_aasen_c64,
	m_factor_hermitian_aasen_c128,
}

m_factor_hermitian_aasen_c64 :: proc(
	A: ^Matrix(complex64), // Hermitian matrix (overwritten with factorization)
	compute_upper := true, // true for upper triangle, false for lower
	allocator := context.allocator,
) -> (
	ipiv: []Blas_Int,
	info: Info, // Pivot indices
) {
	assert(
		A.format == .General || A.format == .Hermitian,
		"Matrix A must be general or Hermitian format",
	)
	assert(A.rows == A.cols, "Matrix A must be square")

	n := A.rows
	lda := A.ld

	uplo_c := cstring("U") if compute_upper else cstring("L")

	// Allocate pivot array
	ipiv = make([]Blas_Int, n, allocator)

	// Workspace query
	work_query: complex64
	lwork: Blas_Int = -1

	lapack.chetrf_aa_(
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(ipiv),
		&work_query,
		&lwork,
		&info,
		1,
	)

	// Allocate workspace
	lwork = auto_cast real(work_query)
	work := make([]complex64, lwork, allocator)

	lapack.chetrf_aa_(
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(ipiv),
		raw_data(work),
		&lwork,
		&info,
		1,
	)

	return ipiv, info
}

m_factor_hermitian_aasen_c128 :: proc(
	A: ^Matrix(complex128), // Hermitian matrix (overwritten with factorization)
	compute_upper := true, // true for upper triangle, false for lower
	allocator := context.allocator,
) -> (
	ipiv: []Blas_Int,
	info: Info, // Pivot indices
) {
	assert(
		A.format == .General || A.format == .Hermitian,
		"Matrix A must be general or Hermitian format",
	)
	assert(A.rows == A.cols, "Matrix A must be square")

	n := A.rows
	lda := A.ld

	uplo_c := cstring("U") if compute_upper else cstring("L")

	// Allocate pivot array
	ipiv = make([]Blas_Int, n, allocator)

	// Workspace query
	work_query: complex128
	lwork: Blas_Int = -1

	lapack.zhetrf_aa_(
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(ipiv),
		&work_query,
		&lwork,
		&info,
		1,
	)

	// Allocate workspace
	lwork = auto_cast real(work_query)
	work := make([]complex128, lwork, allocator)

	lapack.zhetrf_aa_(
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(ipiv),
		raw_data(work),
		&lwork,
		&info,
		1,
	)

	return ipiv, info
}

// Aasen's 2-stage algorithm factorization
m_factor_hermitian_aasen_2stage :: proc {
	m_factor_hermitian_aasen_2stage_c64,
	m_factor_hermitian_aasen_2stage_c128,
}

m_factor_hermitian_aasen_2stage_c64 :: proc(
	A: ^Matrix(complex64), // Hermitian matrix (overwritten with factorization)
	compute_upper := true, // true for upper triangle, false for lower
	allocator := context.allocator,
) -> (
	TB: Matrix(complex64),
	ipiv: []Blas_Int,
	ipiv2: []Blas_Int,
	info: Info, // Block reflector from 2-stage factorization// Primary pivot indices// Secondary pivot indices from 2-stage
) {
	assert(
		A.format == .General || A.format == .Hermitian,
		"Matrix A must be general or Hermitian format",
	)
	assert(A.rows == A.cols, "Matrix A must be square")

	n := A.rows
	lda := A.ld

	uplo_c := cstring("U") if compute_upper else cstring("L")

	// Allocate pivot arrays
	ipiv = make([]Blas_Int, n, allocator)
	ipiv2 = make([]Blas_Int, n, allocator)

	// TB matrix size estimation
	nb := min(64, n) // Typical block size
	ltb := 4 * n * nb
	TB = make_matrix(complex64, ltb, 1, .General, allocator)

	// Workspace query
	work_query: complex64
	lwork: Blas_Int = -1
	ltb_i := Blas_Int(ltb)

	lapack.chetrf_aa_2stage_(
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(TB.data),
		&ltb_i,
		raw_data(ipiv),
		raw_data(ipiv2),
		&work_query,
		&lwork,
		&info,
		1,
	)

	// Allocate workspace
	lwork = auto_cast real(work_query)
	work := make([]complex64, lwork, allocator)

	lapack.chetrf_aa_2stage_(
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(TB.data),
		&ltb_i,
		raw_data(ipiv),
		raw_data(ipiv2),
		raw_data(work),
		&lwork,
		&info,
		1,
	)

	return TB, ipiv, ipiv2, info
}

m_factor_hermitian_aasen_2stage_c128 :: proc(
	A: ^Matrix(complex128), // Hermitian matrix (overwritten with factorization)
	compute_upper := true, // true for upper triangle, false for lower
	allocator := context.allocator,
) -> (
	TB: Matrix(complex128),
	ipiv: []Blas_Int,
	ipiv2: []Blas_Int,
	info: Info, // Block reflector from 2-stage factorization// Primary pivot indices// Secondary pivot indices from 2-stage
) {
	assert(
		A.format == .General || A.format == .Hermitian,
		"Matrix A must be general or Hermitian format",
	)
	assert(A.rows == A.cols, "Matrix A must be square")

	n := A.rows
	lda := A.ld

	uplo_c := cstring("U") if compute_upper else cstring("L")

	// Allocate pivot arrays
	ipiv = make([]Blas_Int, n, allocator)
	ipiv2 = make([]Blas_Int, n, allocator)

	// TB matrix size estimation
	nb := min(64, n) // Typical block size
	ltb := 4 * n * nb
	TB = make_matrix(complex128, ltb, 1, .General, allocator)

	// Workspace query
	work_query: complex128
	lwork: Blas_Int = -1
	ltb_i := Blas_Int(ltb)

	lapack.zhetrf_aa_2stage_(
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(TB.data),
		&ltb_i,
		raw_data(ipiv),
		raw_data(ipiv2),
		&work_query,
		&lwork,
		&info,
		1,
	)

	// Allocate workspace
	lwork = auto_cast real(work_query)
	work := make([]complex128, lwork, allocator)

	lapack.zhetrf_aa_2stage_(
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(TB.data),
		&ltb_i,
		raw_data(ipiv),
		raw_data(ipiv2),
		raw_data(work),
		&lwork,
		&info,
		1,
	)

	return TB, ipiv, ipiv2, info
}

// Rook pivoting factorization (enhanced numerical stability)
m_factor_hermitian_rook :: proc {
	m_factor_hermitian_rook_c64,
	m_factor_hermitian_rook_c128,
}

m_factor_hermitian_rook_c64 :: proc(
	A: ^Matrix(complex64), // Hermitian matrix (overwritten with factorization)
	compute_upper := true, // true for upper triangle, false for lower
	allocator := context.allocator,
) -> (
	ipiv: []Blas_Int,
	info: Info, // Pivot indices
) {
	assert(
		A.format == .General || A.format == .Hermitian,
		"Matrix A must be general or Hermitian format",
	)
	assert(A.rows == A.cols, "Matrix A must be square")

	n := A.rows
	lda := A.ld

	uplo_c := cstring("U") if compute_upper else cstring("L")

	// Allocate pivot array
	ipiv = make([]Blas_Int, n, allocator)

	// Workspace query
	work_query: complex64
	lwork: Blas_Int = -1

	lapack.chetrf_rook_(
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(ipiv),
		&work_query,
		&lwork,
		&info,
		1,
	)

	// Allocate workspace
	lwork = auto_cast real(work_query)
	work := make([]complex64, lwork, allocator)

	lapack.chetrf_rook_(
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(ipiv),
		raw_data(work),
		&lwork,
		&info,
		1,
	)

	return ipiv, info
}

m_factor_hermitian_rook_c128 :: proc(
	A: ^Matrix(complex128), // Hermitian matrix (overwritten with factorization)
	compute_upper := true, // true for upper triangle, false for lower
	allocator := context.allocator,
) -> (
	ipiv: []Blas_Int,
	info: Info, // Pivot indices
) {
	assert(
		A.format == .General || A.format == .Hermitian,
		"Matrix A must be general or Hermitian format",
	)
	assert(A.rows == A.cols, "Matrix A must be square")

	n := A.rows
	lda := A.ld

	uplo_c := cstring("U") if compute_upper else cstring("L")

	// Allocate pivot array
	ipiv = make([]Blas_Int, n, allocator)

	// Workspace query
	work_query: complex128
	lwork: Blas_Int = -1

	lapack.zhetrf_rook_(
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(ipiv),
		&work_query,
		&lwork,
		&info,
		1,
	)

	// Allocate workspace
	lwork = auto_cast real(work_query)
	work := make([]complex128, lwork, allocator)

	lapack.zhetrf_rook_(
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(ipiv),
		raw_data(work),
		&lwork,
		&info,
		1,
	)

	return ipiv, info
}

// Rank-revealing factorization (for singular or ill-conditioned matrices)
m_factor_hermitian_rank_revealing :: proc {
	m_factor_hermitian_rank_revealing_c64,
	m_factor_hermitian_rank_revealing_c128,
}

m_factor_hermitian_rank_revealing_c64 :: proc(
	A: ^Matrix(complex64), // Hermitian matrix (overwritten with factorization)
	compute_upper := true, // true for upper triangle, false for lower
	allocator := context.allocator,
) -> (
	E: Matrix(complex64),
	ipiv: []Blas_Int,
	info: Info, // Block diagonal matrix E from factorization// Pivot indices
) {
	assert(
		A.format == .General || A.format == .Hermitian,
		"Matrix A must be general or Hermitian format",
	)
	assert(A.rows == A.cols, "Matrix A must be square")

	n := A.rows
	lda := A.ld

	uplo_c := cstring("U") if compute_upper else cstring("L")

	// Allocate pivot array and E matrix
	ipiv = make([]Blas_Int, n, allocator)
	E = make_matrix(complex64, n, n, .General, allocator)

	// Workspace query
	work_query: complex64
	lwork: Blas_Int = -1

	lapack.chetrf_rk_(
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(E.data),
		raw_data(ipiv),
		&work_query,
		&lwork,
		&info,
		1,
	)

	// Allocate workspace
	lwork = auto_cast real(work_query)
	work := make([]complex64, lwork, allocator)

	lapack.chetrf_rk_(
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(E.data),
		raw_data(ipiv),
		raw_data(work),
		&lwork,
		&info,
		1,
	)

	return E, ipiv, info
}

m_factor_hermitian_rank_revealing_c128 :: proc(
	A: ^Matrix(complex128), // Hermitian matrix (overwritten with factorization)
	compute_upper := true, // true for upper triangle, false for lower
	allocator := context.allocator,
) -> (
	E: Matrix(complex128),
	ipiv: []Blas_Int,
	info: Info, // Block diagonal matrix E from factorization// Pivot indices
) {
	assert(
		A.format == .General || A.format == .Hermitian,
		"Matrix A must be general or Hermitian format",
	)
	assert(A.rows == A.cols, "Matrix A must be square")

	n := A.rows
	lda := A.ld

	uplo_c := cstring("U") if compute_upper else cstring("L")

	// Allocate pivot array and E matrix
	ipiv = make([]Blas_Int, n, allocator)
	E = make_matrix(complex128, n, n, .General, allocator)

	// Workspace query
	work_query: complex128
	lwork: Blas_Int = -1

	lapack.zhetrf_rk_(
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(E.data),
		raw_data(ipiv),
		&work_query,
		&lwork,
		&info,
		1,
	)

	// Allocate workspace
	lwork = auto_cast real(work_query)
	work := make([]complex128, lwork, allocator)

	lapack.zhetrf_rk_(
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(E.data),
		raw_data(ipiv),
		raw_data(work),
		&lwork,
		&info,
		1,
	)

	return E, ipiv, info
}

// ===================================================================================
// HERMITIAN MATRIX INVERSION
// ===================================================================================

// Invert a Hermitian matrix after Bunch-Kaufman factorization
m_invert_hermitian :: proc {
	m_invert_hermitian_c64,
	m_invert_hermitian_c128,
}

m_invert_hermitian_c64 :: proc(
	A: ^Matrix(complex64),
	ipiv: ^[]int,
	allocator := context.allocator,
) -> (
	ok: bool,
) {
	assert(A.format == .Hermitian, "Matrix must be Hermitian format")
	assert(A.rows == A.cols, "Matrix must be square")
	assert(len(ipiv) >= A.rows, "ipiv must have at least n elements")

	n := Blas_Int(A.rows)
	lda := Blas_Int(A.ld)

	uplo_c := cstring("U")
	if A.format == .Hermitian {
		uplo_c = A.storage.hermitian.uplo
	}

	// Convert ipiv to LAPACK format
	ipiv_lapack := make([]Blas_Int, n, allocator)
	defer delete(ipiv_lapack, allocator)
	for i in 0 ..< n {
		ipiv_lapack[i] = Blas_Int(ipiv[i])
	}

	// Workspace
	work := make([]complex64, n, allocator)
	defer delete(work, allocator)

	info: Info
	lapack.chetri_(
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(ipiv_lapack),
		raw_data(work),
		&info,
		1,
	)

	return info == 0
}

m_invert_hermitian_c128 :: proc(
	A: ^Matrix(complex128),
	ipiv: ^[]int,
	allocator := context.allocator,
) -> (
	ok: bool,
) {
	assert(A.format == .Hermitian, "Matrix must be Hermitian format")
	assert(A.rows == A.cols, "Matrix must be square")
	assert(len(ipiv) >= A.rows, "ipiv must have at least n elements")

	n := Blas_Int(A.rows)
	lda := Blas_Int(A.ld)

	uplo_c := cstring("U")
	if A.format == .Hermitian {
		uplo_c = A.storage.hermitian.uplo
	}

	// Convert ipiv to LAPACK format
	ipiv_lapack := make([]Blas_Int, n, allocator)
	defer delete(ipiv_lapack, allocator)
	for i in 0 ..< n {
		ipiv_lapack[i] = Blas_Int(ipiv[i])
	}

	// Workspace
	work := make([]complex128, n, allocator)
	defer delete(work, allocator)

	info: Info
	lapack.zhetri_(
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(ipiv_lapack),
		raw_data(work),
		&info,
		1,
	)

	return info == 0
}

// Invert a Hermitian matrix using improved algorithm (Version 2)
m_invert_hermitian_2 :: proc {
	m_invert_hermitian_2_c64,
	m_invert_hermitian_2_c128,
}

m_invert_hermitian_2_c64 :: proc(
	A: ^Matrix(complex64),
	ipiv: ^[]int,
	allocator := context.allocator,
) -> (
	work_size: int,
	ok: bool,
) {
	assert(A.format == .Hermitian, "Matrix must be Hermitian format")
	assert(A.rows == A.cols, "Matrix must be square")
	assert(len(ipiv) >= A.rows, "ipiv must have at least n elements")

	n := Blas_Int(A.rows)
	lda := Blas_Int(A.ld)

	uplo_c := cstring("U")
	if A.format == .Hermitian {
		uplo_c = A.storage.hermitian.uplo
	}

	// Convert ipiv to LAPACK format
	ipiv_lapack := make([]Blas_Int, n, allocator)
	defer delete(ipiv_lapack, allocator)
	for i in 0 ..< n {
		ipiv_lapack[i] = Blas_Int(ipiv[i])
	}

	// Workspace query
	work_query: complex64
	lwork: Blas_Int = -1
	info: Info

	lapack.chetri2_(
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(ipiv_lapack),
		&work_query,
		&lwork,
		&info,
		1,
	)

	if info != 0 {
		return 0, false
	}

	// Allocate and compute
	lwork = auto_cast real(work_query)
	work := make([]complex64, lwork, allocator)
	defer delete(work, allocator)

	lapack.chetri2_(
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(ipiv_lapack),
		raw_data(work),
		&lwork,
		&info,
		1,
	)

	return int(lwork), info == 0
}

m_invert_hermitian_2_c128 :: proc(
	A: ^Matrix(complex128),
	ipiv: ^[]int,
	allocator := context.allocator,
) -> (
	work_size: int,
	ok: bool,
) {
	assert(A.format == .Hermitian, "Matrix must be Hermitian format")
	assert(A.rows == A.cols, "Matrix must be square")
	assert(len(ipiv) >= A.rows, "ipiv must have at least n elements")

	n := Blas_Int(A.rows)
	lda := Blas_Int(A.ld)

	uplo_c := cstring("U")
	if A.format == .Hermitian {
		uplo_c = A.storage.hermitian.uplo
	}

	// Convert ipiv to LAPACK format
	ipiv_lapack := make([]Blas_Int, n, allocator)
	defer delete(ipiv_lapack, allocator)
	for i in 0 ..< n {
		ipiv_lapack[i] = Blas_Int(ipiv[i])
	}

	// Workspace query
	work_query: complex128
	lwork: Blas_Int = -1
	info: Info

	lapack.zhetri2_(
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(ipiv_lapack),
		&work_query,
		&lwork,
		&info,
		1,
	)

	if info != 0 {
		return 0, false
	}

	// Allocate and compute
	lwork = auto_cast real(work_query)
	work := make([]complex128, lwork, allocator)
	defer delete(work, allocator)

	lapack.zhetri2_(
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(ipiv_lapack),
		raw_data(work),
		&lwork,
		&info,
		1,
	)

	return int(lwork), info == 0
}

// Invert a Hermitian matrix using blocked algorithm (Version 2x)
m_invert_hermitian_2x :: proc {
	m_invert_hermitian_2x_c64,
	m_invert_hermitian_2x_c128,
}

m_invert_hermitian_2x_c64 :: proc(
	A: ^Matrix(complex64),
	ipiv: ^[]int,
	nb: int = 64,
	allocator := context.allocator,
) -> (
	ok: bool,
) {
	assert(A.format == .Hermitian, "Matrix must be Hermitian format")
	assert(A.rows == A.cols, "Matrix must be square")
	assert(len(ipiv) >= A.rows, "ipiv must have at least n elements")

	n := Blas_Int(A.rows)
	lda := Blas_Int(A.ld)
	nb_use := Blas_Int(nb)

	uplo_c := cstring("U")
	if A.format == .Hermitian {
		uplo_c = A.storage.hermitian.uplo
	}

	// Convert ipiv to LAPACK format
	ipiv_lapack := make([]Blas_Int, n, allocator)
	defer delete(ipiv_lapack, allocator)
	for i in 0 ..< n {
		ipiv_lapack[i] = Blas_Int(ipiv[i])
	}

	// Workspace
	work := make([]complex64, n * nb, allocator)
	defer delete(work, allocator)

	info: Info
	lapack.chetri2x_(
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(ipiv_lapack),
		raw_data(work),
		&nb_use,
		&info,
		1,
	)

	return info == 0
}

m_invert_hermitian_2x_c128 :: proc(
	A: ^Matrix(complex128),
	ipiv: ^[]int,
	nb: int = 64,
	allocator := context.allocator,
) -> (
	ok: bool,
) {
	assert(A.format == .Hermitian, "Matrix must be Hermitian format")
	assert(A.rows == A.cols, "Matrix must be square")
	assert(len(ipiv) >= A.rows, "ipiv must have at least n elements")

	n := Blas_Int(A.rows)
	lda := Blas_Int(A.ld)
	nb_use := Blas_Int(nb)

	uplo_c := cstring("U")
	if A.format == .Hermitian {
		uplo_c = A.storage.hermitian.uplo
	}

	// Convert ipiv to LAPACK format
	ipiv_lapack := make([]Blas_Int, n, allocator)
	defer delete(ipiv_lapack, allocator)
	for i in 0 ..< n {
		ipiv_lapack[i] = Blas_Int(ipiv[i])
	}

	// Workspace
	work := make([]complex128, n * nb, allocator)
	defer delete(work, allocator)

	info: Info
	lapack.zhetri2x_(
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(ipiv_lapack),
		raw_data(work),
		&nb_use,
		&info,
		1,
	)

	return info == 0
}

// Invert a Hermitian matrix using Bounded Bunch-Kaufman (Version 3)
m_invert_hermitian_3 :: proc {
	m_invert_hermitian_3_c64,
	m_invert_hermitian_3_c128,
}

m_invert_hermitian_3_c64 :: proc(
	A: ^Matrix(complex64),
	E: ^[]complex64,
	ipiv: ^[]int,
	allocator := context.allocator,
) -> (
	work_size: int,
	ok: bool,
) {
	assert(A.format == .Hermitian, "Matrix must be Hermitian format")
	assert(A.rows == A.cols, "Matrix must be square")
	assert(len(ipiv) >= A.rows, "ipiv must have at least n elements")
	assert(len(E) >= A.rows, "E must have at least n elements")

	n := Blas_Int(A.rows)
	lda := Blas_Int(A.ld)

	uplo_c := cstring("U")
	if A.format == .Hermitian {
		uplo_c = A.storage.hermitian.uplo
	}

	// Convert ipiv to LAPACK format
	ipiv_lapack := make([]Blas_Int, n, allocator)
	defer delete(ipiv_lapack, allocator)
	for i in 0 ..< n {
		ipiv_lapack[i] = Blas_Int(ipiv[i])
	}

	// Workspace query
	work_query: complex64
	lwork: Blas_Int = -1
	info: Info

	lapack.chetri_3_(
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(E^),
		raw_data(ipiv_lapack),
		&work_query,
		&lwork,
		&info,
		1,
	)

	if info != 0 {
		return 0, false
	}

	// Allocate and compute
	lwork = auto_cast real(work_query)
	work := make([]complex64, lwork, allocator)
	defer delete(work, allocator)

	lapack.chetri_3_(
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(E^),
		raw_data(ipiv_lapack),
		raw_data(work),
		&lwork,
		&info,
		1,
	)

	return int(lwork), info == 0
}

m_invert_hermitian_3_c128 :: proc(
	A: ^Matrix(complex128),
	E: ^[]complex128,
	ipiv: ^[]int,
	allocator := context.allocator,
) -> (
	work_size: int,
	ok: bool,
) {
	assert(A.format == .Hermitian, "Matrix must be Hermitian format")
	assert(A.rows == A.cols, "Matrix must be square")
	assert(len(ipiv) >= A.rows, "ipiv must have at least n elements")
	assert(len(E) >= A.rows, "E must have at least n elements")

	n := Blas_Int(A.rows)
	lda := Blas_Int(A.ld)

	uplo_c := cstring("U")
	if A.format == .Hermitian {
		uplo_c = A.storage.hermitian.uplo
	}

	// Convert ipiv to LAPACK format
	ipiv_lapack := make([]Blas_Int, n, allocator)
	defer delete(ipiv_lapack, allocator)
	for i in 0 ..< n {
		ipiv_lapack[i] = Blas_Int(ipiv[i])
	}

	// Workspace query
	work_query: complex128
	lwork: Blas_Int = -1
	info: Info

	lapack.zhetri_3_(
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(E^),
		raw_data(ipiv_lapack),
		&work_query,
		&lwork,
		&info,
		1,
	)

	if info != 0 {
		return 0, false
	}

	// Allocate and compute
	lwork = auto_cast real(work_query)
	work := make([]complex128, lwork, allocator)
	defer delete(work, allocator)

	lapack.zhetri_3_(
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(E^),
		raw_data(ipiv_lapack),
		raw_data(work),
		&lwork,
		&info,
		1,
	)

	return int(lwork), info == 0
}

// ===================================================================================
// HERMITIAN MATRIX TRIANGULAR SOLVE
// ===================================================================================

// Solve linear system using Hermitian factorization
m_solve_hermitian :: proc {
	m_solve_hermitian_c64,
	m_solve_hermitian_c128,
}

m_solve_hermitian_c64 :: proc(
	A: ^Matrix(complex64),
	ipiv: ^[]int,
	B: ^Matrix(complex64),
) -> (
	ok: bool,
) {
	assert(A.format == .Hermitian, "Matrix A must be Hermitian format")
	assert(A.rows == A.cols, "Matrix A must be square")
	assert(len(ipiv) >= A.rows, "ipiv must have at least n elements")
	assert(A.rows == B.rows, "A and B must have same number of rows")

	n := Blas_Int(A.rows)
	nrhs := Blas_Int(B.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	uplo_c := cstring("U")
	if A.format == .Hermitian {
		uplo_c = A.storage.hermitian.uplo
	}

	// Convert ipiv to LAPACK format
	ipiv_lapack := make([]Blas_Int, n)
	defer delete(ipiv_lapack)
	for i in 0 ..< n {
		ipiv_lapack[i] = Blas_Int(ipiv[i])
	}

	info: Info
	lapack.chetrs_(
		uplo_c,
		&n,
		&nrhs,
		raw_data(A.data),
		&lda,
		raw_data(ipiv_lapack),
		raw_data(B.data),
		&ldb,
		&info,
		1,
	)

	return info == 0
}

m_solve_hermitian_c128 :: proc(
	A: ^Matrix(complex128),
	ipiv: ^[]int,
	B: ^Matrix(complex128),
) -> (
	ok: bool,
) {
	assert(A.format == .Hermitian, "Matrix A must be Hermitian format")
	assert(A.rows == A.cols, "Matrix A must be square")
	assert(len(ipiv) >= A.rows, "ipiv must have at least n elements")
	assert(A.rows == B.rows, "A and B must have same number of rows")

	n := Blas_Int(A.rows)
	nrhs := Blas_Int(B.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	uplo_c := cstring("U")
	if A.format == .Hermitian {
		uplo_c = A.storage.hermitian.uplo
	}

	// Convert ipiv to LAPACK format
	ipiv_lapack := make([]Blas_Int, n)
	defer delete(ipiv_lapack)
	for i in 0 ..< n {
		ipiv_lapack[i] = Blas_Int(ipiv[i])
	}

	info: Info
	lapack.zhetrs_(
		uplo_c,
		&n,
		&nrhs,
		raw_data(A.data),
		&lda,
		raw_data(ipiv_lapack),
		raw_data(B.data),
		&ldb,
		&info,
		1,
	)

	return info == 0
}

// Solve linear system using improved algorithm (Version 2)
m_solve_hermitian_2 :: proc {
	m_solve_hermitian_2_c64,
	m_solve_hermitian_2_c128,
}

m_solve_hermitian_2_c64 :: proc(
	A: ^Matrix(complex64),
	ipiv: ^[]int,
	B: ^Matrix(complex64),
	allocator := context.allocator,
) -> (
	ok: bool,
) {
	assert(A.format == .Hermitian, "Matrix A must be Hermitian format")
	assert(A.rows == A.cols, "Matrix A must be square")
	assert(len(ipiv) >= A.rows, "ipiv must have at least n elements")
	assert(A.rows == B.rows, "A and B must have same number of rows")

	n := Blas_Int(A.rows)
	nrhs := Blas_Int(B.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	uplo_c := cstring("U")
	if A.format == .Hermitian {
		uplo_c = A.storage.hermitian.uplo
	}

	// Convert ipiv to LAPACK format
	ipiv_lapack := make([]Blas_Int, n, allocator)
	defer delete(ipiv_lapack, allocator)
	for i in 0 ..< n {
		ipiv_lapack[i] = Blas_Int(ipiv[i])
	}

	// Workspace
	work := make([]complex64, n, allocator)
	defer delete(work, allocator)

	info: Info
	lapack.chetrs2_(
		uplo_c,
		&n,
		&nrhs,
		raw_data(A.data),
		&lda,
		raw_data(ipiv_lapack),
		raw_data(B.data),
		&ldb,
		raw_data(work),
		&info,
		1,
	)

	return info == 0
}

m_solve_hermitian_2_c128 :: proc(
	A: ^Matrix(complex128),
	ipiv: ^[]int,
	B: ^Matrix(complex128),
	allocator := context.allocator,
) -> (
	ok: bool,
) {
	assert(A.format == .Hermitian, "Matrix A must be Hermitian format")
	assert(A.rows == A.cols, "Matrix A must be square")
	assert(len(ipiv) >= A.rows, "ipiv must have at least n elements")
	assert(A.rows == B.rows, "A and B must have same number of rows")

	n := Blas_Int(A.rows)
	nrhs := Blas_Int(B.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	uplo_c := cstring("U")
	if A.format == .Hermitian {
		uplo_c = A.storage.hermitian.uplo
	}

	// Convert ipiv to LAPACK format
	ipiv_lapack := make([]Blas_Int, n, allocator)
	defer delete(ipiv_lapack, allocator)
	for i in 0 ..< n {
		ipiv_lapack[i] = Blas_Int(ipiv[i])
	}

	// Workspace
	work := make([]complex128, n, allocator)
	defer delete(work, allocator)

	info: Info
	lapack.zhetrs2_(
		uplo_c,
		&n,
		&nrhs,
		raw_data(A.data),
		&lda,
		raw_data(ipiv_lapack),
		raw_data(B.data),
		&ldb,
		raw_data(work),
		&info,
		1,
	)

	return info == 0
}

// Solve linear system using Bounded Bunch-Kaufman (Version 3)
m_solve_hermitian_3 :: proc {
	m_solve_hermitian_3_c64,
	m_solve_hermitian_3_c128,
}

m_solve_hermitian_3_c64 :: proc(
	A: ^Matrix(complex64),
	E: ^[]complex64,
	ipiv: ^[]int,
	B: ^Matrix(complex64),
) -> (
	ok: bool,
) {
	assert(A.format == .Hermitian, "Matrix A must be Hermitian format")
	assert(A.rows == A.cols, "Matrix A must be square")
	assert(len(ipiv) >= A.rows, "ipiv must have at least n elements")
	assert(len(E) >= A.rows, "E must have at least n elements")
	assert(A.rows == B.rows, "A and B must have same number of rows")

	n := Blas_Int(A.rows)
	nrhs := Blas_Int(B.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	uplo_c := cstring("U")
	if A.format == .Hermitian {
		uplo_c = A.storage.hermitian.uplo
	}

	// Convert ipiv to LAPACK format
	ipiv_lapack := make([]Blas_Int, n)
	defer delete(ipiv_lapack)
	for i in 0 ..< n {
		ipiv_lapack[i] = Blas_Int(ipiv[i])
	}

	info: Info
	lapack.chetrs_3_(
		uplo_c,
		&n,
		&nrhs,
		raw_data(A.data),
		&lda,
		raw_data(E^),
		raw_data(ipiv_lapack),
		raw_data(B.data),
		&ldb,
		&info,
		1,
	)

	return info == 0
}

m_solve_hermitian_3_c128 :: proc(
	A: ^Matrix(complex128),
	E: ^[]complex128,
	ipiv: ^[]int,
	B: ^Matrix(complex128),
) -> (
	ok: bool,
) {
	assert(A.format == .Hermitian, "Matrix A must be Hermitian format")
	assert(A.rows == A.cols, "Matrix A must be square")
	assert(len(ipiv) >= A.rows, "ipiv must have at least n elements")
	assert(len(E) >= A.rows, "E must have at least n elements")
	assert(A.rows == B.rows, "A and B must have same number of rows")

	n := Blas_Int(A.rows)
	nrhs := Blas_Int(B.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	uplo_c := cstring("U")
	if A.format == .Hermitian {
		uplo_c = A.storage.hermitian.uplo
	}

	// Convert ipiv to LAPACK format
	ipiv_lapack := make([]Blas_Int, n)
	defer delete(ipiv_lapack)
	for i in 0 ..< n {
		ipiv_lapack[i] = Blas_Int(ipiv[i])
	}

	info: Info
	lapack.zhetrs_3_(
		uplo_c,
		&n,
		&nrhs,
		raw_data(A.data),
		&lda,
		raw_data(E^),
		raw_data(ipiv_lapack),
		raw_data(B.data),
		&ldb,
		&info,
		1,
	)

	return info == 0
}

// Solve linear system using Aasen's algorithm
m_solve_hermitian_aa :: proc {
	m_solve_hermitian_aa_c64,
	m_solve_hermitian_aa_c128,
}

m_solve_hermitian_aa_c64 :: proc(
	A: ^Matrix(complex64),
	ipiv: ^[]int,
	B: ^Matrix(complex64),
	allocator := context.allocator,
) -> (
	work_size: int,
	ok: bool,
) {
	assert(A.format == .Hermitian, "Matrix A must be Hermitian format")
	assert(A.rows == A.cols, "Matrix A must be square")
	assert(len(ipiv) >= A.rows, "ipiv must have at least n elements")
	assert(A.rows == B.rows, "A and B must have same number of rows")

	n := Blas_Int(A.rows)
	nrhs := Blas_Int(B.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	uplo_c := cstring("U")
	if A.format == .Hermitian {
		uplo_c = A.storage.hermitian.uplo
	}

	// Convert ipiv to LAPACK format
	ipiv_lapack := make([]Blas_Int, n, allocator)
	defer delete(ipiv_lapack, allocator)
	for i in 0 ..< n {
		ipiv_lapack[i] = Blas_Int(ipiv[i])
	}

	// Workspace query
	work_query: complex64
	lwork: Blas_Int = -1
	info: Info

	lapack.chetrs_aa_(
		uplo_c,
		&n,
		&nrhs,
		raw_data(A.data),
		&lda,
		raw_data(ipiv_lapack),
		raw_data(B.data),
		&ldb,
		&work_query,
		&lwork,
		&info,
		1,
	)

	if info != 0 {
		return 0, false
	}

	// Allocate and compute
	lwork = auto_cast real(work_query)
	work := make([]complex64, lwork, allocator)
	defer delete(work, allocator)

	lapack.chetrs_aa_(
		uplo_c,
		&n,
		&nrhs,
		raw_data(A.data),
		&lda,
		raw_data(ipiv_lapack),
		raw_data(B.data),
		&ldb,
		raw_data(work),
		&lwork,
		&info,
		1,
	)

	return int(lwork), info == 0
}

m_solve_hermitian_aa_c128 :: proc(
	A: ^Matrix(complex128),
	ipiv: ^[]int,
	B: ^Matrix(complex128),
	allocator := context.allocator,
) -> (
	work_size: int,
	ok: bool,
) {
	assert(A.format == .Hermitian, "Matrix A must be Hermitian format")
	assert(A.rows == A.cols, "Matrix A must be square")
	assert(len(ipiv) >= A.rows, "ipiv must have at least n elements")
	assert(A.rows == B.rows, "A and B must have same number of rows")

	n := Blas_Int(A.rows)
	nrhs := Blas_Int(B.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	uplo_c := cstring("U")
	if A.format == .Hermitian {
		uplo_c = A.storage.hermitian.uplo
	}

	// Convert ipiv to LAPACK format
	ipiv_lapack := make([]Blas_Int, n, allocator)
	defer delete(ipiv_lapack, allocator)
	for i in 0 ..< n {
		ipiv_lapack[i] = Blas_Int(ipiv[i])
	}

	// Workspace query
	work_query: complex128
	lwork: Blas_Int = -1
	info: Info

	lapack.zhetrs_aa_(
		uplo_c,
		&n,
		&nrhs,
		raw_data(A.data),
		&lda,
		raw_data(ipiv_lapack),
		raw_data(B.data),
		&ldb,
		&work_query,
		&lwork,
		&info,
		1,
	)

	if info != 0 {
		return 0, false
	}

	// Allocate and compute
	lwork = auto_cast real(work_query)
	work := make([]complex128, lwork, allocator)
	defer delete(work, allocator)

	lapack.zhetrs_aa_(
		uplo_c,
		&n,
		&nrhs,
		raw_data(A.data),
		&lda,
		raw_data(ipiv_lapack),
		raw_data(B.data),
		&ldb,
		raw_data(work),
		&lwork,
		&info,
		1,
	)

	return int(lwork), info == 0
}

// Solve linear system using Aasen's 2-stage algorithm
m_solve_hermitian_aa_2stage :: proc {
	m_solve_hermitian_aa_2stage_c64,
	m_solve_hermitian_aa_2stage_c128,
}

m_solve_hermitian_aa_2stage_c64 :: proc(
	A: ^Matrix(complex64),
	TB: ^Matrix(complex64),
	ipiv: ^[]int,
	ipiv2: ^[]int,
	B: ^Matrix(complex64),
) -> (
	ok: bool,
) {
	assert(A.format == .Hermitian, "Matrix A must be Hermitian format")
	assert(A.rows == A.cols, "Matrix A must be square")
	assert(len(ipiv) >= A.rows, "ipiv must have at least n elements")
	assert(len(ipiv2) >= A.rows, "ipiv2 must have at least n elements")
	assert(A.rows == B.rows, "A and B must have same number of rows")

	n := Blas_Int(A.rows)
	nrhs := Blas_Int(B.cols)
	lda := Blas_Int(A.ld)
	ltb := Blas_Int(TB.ld)
	ldb := Blas_Int(B.ld)

	uplo_c := cstring("U")
	if A.format == .Hermitian {
		uplo_c = A.storage.hermitian.uplo
	}

	// Convert ipiv arrays to LAPACK format
	ipiv_lapack := make([]Blas_Int, n)
	defer delete(ipiv_lapack)
	ipiv2_lapack := make([]Blas_Int, n)
	defer delete(ipiv2_lapack)

	for i in 0 ..< n {
		ipiv_lapack[i] = Blas_Int(ipiv[i])
		ipiv2_lapack[i] = Blas_Int(ipiv2[i])
	}

	info: Info
	lapack.chetrs_aa_2stage_(
		uplo_c,
		&n,
		&nrhs,
		raw_data(A.data),
		&lda,
		raw_data(TB.data),
		&ltb,
		raw_data(ipiv_lapack),
		raw_data(ipiv2_lapack),
		raw_data(B.data),
		&ldb,
		&info,
		1,
	)

	return info == 0
}

m_solve_hermitian_aa_2stage_c128 :: proc(
	A: ^Matrix(complex128),
	TB: ^Matrix(complex128),
	ipiv: ^[]int,
	ipiv2: ^[]int,
	B: ^Matrix(complex128),
) -> (
	ok: bool,
) {
	assert(A.format == .Hermitian, "Matrix A must be Hermitian format")
	assert(A.rows == A.cols, "Matrix A must be square")
	assert(len(ipiv) >= A.rows, "ipiv must have at least n elements")
	assert(len(ipiv2) >= A.rows, "ipiv2 must have at least n elements")
	assert(A.rows == B.rows, "A and B must have same number of rows")

	n := Blas_Int(A.rows)
	nrhs := Blas_Int(B.cols)
	lda := Blas_Int(A.ld)
	ltb := Blas_Int(TB.ld)
	ldb := Blas_Int(B.ld)

	uplo_c := cstring("U")
	if A.format == .Hermitian {
		uplo_c = A.storage.hermitian.uplo
	}

	// Convert ipiv arrays to LAPACK format
	ipiv_lapack := make([]Blas_Int, n)
	defer delete(ipiv_lapack)
	ipiv2_lapack := make([]Blas_Int, n)
	defer delete(ipiv2_lapack)

	for i in 0 ..< n {
		ipiv_lapack[i] = Blas_Int(ipiv[i])
		ipiv2_lapack[i] = Blas_Int(ipiv2[i])
	}

	info: Info
	lapack.zhetrs_aa_2stage_(
		uplo_c,
		&n,
		&nrhs,
		raw_data(A.data),
		&lda,
		raw_data(TB.data),
		&ltb,
		raw_data(ipiv_lapack),
		raw_data(ipiv2_lapack),
		raw_data(B.data),
		&ldb,
		&info,
		1,
	)

	return info == 0
}

// Solve linear system using Rook pivoting
m_solve_hermitian_rook :: proc {
	m_solve_hermitian_rook_c64,
	m_solve_hermitian_rook_c128,
}

m_solve_hermitian_rook_c64 :: proc(
	A: ^Matrix(complex64),
	ipiv: ^[]int,
	B: ^Matrix(complex64),
) -> (
	ok: bool,
) {
	assert(A.format == .Hermitian, "Matrix A must be Hermitian format")
	assert(A.rows == A.cols, "Matrix A must be square")
	assert(len(ipiv) >= A.rows, "ipiv must have at least n elements")
	assert(A.rows == B.rows, "A and B must have same number of rows")

	n := Blas_Int(A.rows)
	nrhs := Blas_Int(B.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	uplo_c := cstring("U")
	if A.format == .Hermitian {
		uplo_c = A.storage.hermitian.uplo
	}

	// Convert ipiv to LAPACK format
	ipiv_lapack := make([]Blas_Int, n)
	defer delete(ipiv_lapack)
	for i in 0 ..< n {
		ipiv_lapack[i] = Blas_Int(ipiv[i])
	}

	info: Info
	lapack.chetrs_rook_(
		uplo_c,
		&n,
		&nrhs,
		raw_data(A.data),
		&lda,
		raw_data(ipiv_lapack),
		raw_data(B.data),
		&ldb,
		&info,
		1,
	)

	return info == 0
}

m_solve_hermitian_rook_c128 :: proc(
	A: ^Matrix(complex128),
	ipiv: ^[]int,
	B: ^Matrix(complex128),
) -> (
	ok: bool,
) {
	assert(A.format == .Hermitian, "Matrix A must be Hermitian format")
	assert(A.rows == A.cols, "Matrix A must be square")
	assert(len(ipiv) >= A.rows, "ipiv must have at least n elements")
	assert(A.rows == B.rows, "A and B must have same number of rows")

	n := Blas_Int(A.rows)
	nrhs := Blas_Int(B.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	uplo_c := cstring("U")
	if A.format == .Hermitian {
		uplo_c = A.storage.hermitian.uplo
	}

	// Convert ipiv to LAPACK format
	ipiv_lapack := make([]Blas_Int, n)
	defer delete(ipiv_lapack)
	for i in 0 ..< n {
		ipiv_lapack[i] = Blas_Int(ipiv[i])
	}

	info: Info
	lapack.zhetrs_rook_(
		uplo_c,
		&n,
		&nrhs,
		raw_data(A.data),
		&lda,
		raw_data(ipiv_lapack),
		raw_data(B.data),
		&ldb,
		&info,
		1,
	)

	return info == 0
}

// ===================================================================================
// HERMITIAN RANK-K UPDATE (RFP FORMAT)
// ===================================================================================

// Hermitian rank-k update in Rectangular Full Packed (RFP) format
// C := alpha*A*A^H + beta*C or C := alpha*A^H*A + beta*C
m_hfrk :: proc {
	m_hfrk_c64,
	m_hfrk_c128,
}

m_hfrk_c64 :: proc(
	C: ^Matrix(complex64),
	A: ^Matrix(complex64),
	alpha: f32 = 1.0,
	beta: f32 = 0.0,
	transpose_a: bool = false,
	compute_upper: bool = true,
	transr: bool = false,
) -> (
	ok: bool,
) {
	assert(C.format == .Packed, "Matrix C must be in packed format for RFP")
	assert(A.format == .General, "Matrix A must be in general format")

	n := Blas_Int(C.storage.packed.n)
	k := Blas_Int(A.cols if !transpose_a else A.rows)
	lda := Blas_Int(A.ld)

	transr_c := cstring("N") if !transr else cstring("T")
	uplo_c := cstring("U") if compute_upper else cstring("L")
	trans_c := cstring("N") if !transpose_a else cstring("C")

	// Validate dimensions
	expected_n := A.rows if !transpose_a else A.cols
	assert(n == expected_n, "Matrix dimensions incompatible")

	info: Info
	lapack.chfrk_(
		transr_c,
		uplo_c,
		trans_c,
		&n,
		&k,
		&alpha,
		raw_data(A.data),
		&lda,
		&beta,
		raw_data(C.data),
		1,
		1,
		1,
	)

	return true
}

m_hfrk_c128 :: proc(
	C: ^Matrix(complex128),
	A: ^Matrix(complex128),
	alpha: f64 = 1.0,
	beta: f64 = 0.0,
	transpose_a: bool = false,
	compute_upper: bool = true,
	transr: bool = false,
) -> (
	ok: bool,
) {
	assert(C.format == .Packed, "Matrix C must be in packed format for RFP")
	assert(A.format == .General, "Matrix A must be in general format")

	n := Blas_Int(C.storage.packed.n)
	k := Blas_Int(A.cols if !transpose_a else A.rows)
	lda := Blas_Int(A.ld)

	transr_c := cstring("N") if !transr else cstring("T")
	uplo_c := cstring("U") if compute_upper else cstring("L")
	trans_c := cstring("N") if !transpose_a else cstring("C")

	// Validate dimensions
	expected_n := A.rows if !transpose_a else A.cols
	assert(n == expected_n, "Matrix dimensions incompatible")

	info: Info
	lapack.zhfrk_(
		transr_c,
		uplo_c,
		trans_c,
		&n,
		&k,
		&alpha,
		raw_data(A.data),
		&lda,
		&beta,
		raw_data(C.data),
		1,
		1,
		1,
	)

	return true
}

// ===================================================================================
// GENERALIZED EIGENVALUE QZ ALGORITHM
// ===================================================================================

// Generalized eigenvalue problem using QZ algorithm
// Computes eigenvalues and optionally eigenvectors of (H, T)
m_hgeqz :: proc {
	m_hgeqz_f32,
	m_hgeqz_f64,
	m_hgeqz_c64,
	m_hgeqz_c128,
}

m_hgeqz_f32 :: proc(
	H: ^Matrix(f32),
	T: ^Matrix(f32),
	ilo: int = 0,
	ihi: int = 0,
	compute_eigenvalues: bool = true,
	compute_q: bool = false,
	compute_z: bool = false,
	allocator := context.allocator,
) -> (
	alphar, alphai, beta: []f32,
	Q, Z: Matrix(f32),
	info: Info,
) {
	assert(H.format == .General && T.format == .General, "Matrices must be general format")
	assert(H.rows == H.cols && T.rows == T.cols, "Matrices must be square")
	assert(H.rows == T.rows, "Matrices must have same dimensions")

	n := Blas_Int(H.rows)
	ldh := Blas_Int(H.ld)
	ldt := Blas_Int(T.ld)

	// Use default range if not specified
	ilo_use := Blas_Int(ilo if ilo > 0 else 1)
	ihi_use := Blas_Int(ihi if ihi > 0 else H.rows)

	job_c := cstring("E") if compute_eigenvalues else cstring("S")
	compq_c := cstring("V") if compute_q else cstring("N")
	compz_c := cstring("V") if compute_z else cstring("N")

	// Allocate outputs
	alphar = make([]f32, n, allocator)
	alphai = make([]f32, n, allocator)
	beta = make([]f32, n, allocator)

	ldq := n if compute_q else 1
	ldz := n if compute_z else 1
	Q = make_matrix(f32, int(n), int(n), .General, allocator) if compute_q else Matrix(f32){}
	Z = make_matrix(f32, int(n), int(n), .General, allocator) if compute_z else Matrix(f32){}

	// Workspace query
	work_query: f32
	lwork: Blas_Int = -1

	lapack.shgeqz_(
		job_c,
		compq_c,
		compz_c,
		&n,
		&ilo_use,
		&ihi_use,
		raw_data(H.data),
		&ldh,
		raw_data(T.data),
		&ldt,
		raw_data(alphar),
		raw_data(alphai),
		raw_data(beta),
		matrix_data_ptr(&Q) if compute_q else nil,
		&ldq,
		matrix_data_ptr(&Z) if compute_z else nil,
		&ldz,
		&work_query,
		&lwork,
		&info,
		1,
		1,
		1,
	)

	if info != 0 {
		return
	}

	// Allocate and compute
	lwork = auto_cast work_query
	work := make([]f32, lwork, allocator)
	defer delete(work, allocator)

	lapack.shgeqz_(
		job_c,
		compq_c,
		compz_c,
		&n,
		&ilo_use,
		&ihi_use,
		raw_data(H.data),
		&ldh,
		raw_data(T.data),
		&ldt,
		raw_data(alphar),
		raw_data(alphai),
		raw_data(beta),
		matrix_data_ptr(&Q) if compute_q else nil,
		&ldq,
		matrix_data_ptr(&Z) if compute_z else nil,
		&ldz,
		raw_data(work),
		&lwork,
		&info,
		1,
		1,
		1,
	)

	return
}

m_hgeqz_f64 :: proc(
	H: ^Matrix(f64),
	T: ^Matrix(f64),
	ilo: int = 0,
	ihi: int = 0,
	compute_eigenvalues: bool = true,
	compute_q: bool = false,
	compute_z: bool = false,
	allocator := context.allocator,
) -> (
	alphar, alphai, beta: []f64,
	Q, Z: Matrix(f64),
	info: Info,
) {
	assert(H.format == .General && T.format == .General, "Matrices must be general format")
	assert(H.rows == H.cols && T.rows == T.cols, "Matrices must be square")
	assert(H.rows == T.rows, "Matrices must have same dimensions")

	n := Blas_Int(H.rows)
	ldh := Blas_Int(H.ld)
	ldt := Blas_Int(T.ld)

	// Use default range if not specified
	ilo_use := Blas_Int(ilo if ilo > 0 else 1)
	ihi_use := Blas_Int(ihi if ihi > 0 else H.rows)

	job_c := cstring("E") if compute_eigenvalues else cstring("S")
	compq_c := cstring("V") if compute_q else cstring("N")
	compz_c := cstring("V") if compute_z else cstring("N")

	// Allocate outputs
	alphar = make([]f64, n, allocator)
	alphai = make([]f64, n, allocator)
	beta = make([]f64, n, allocator)

	ldq := n if compute_q else 1
	ldz := n if compute_z else 1
	Q = make_matrix(f64, int(n), int(n), .General, allocator) if compute_q else Matrix(f64){}
	Z = make_matrix(f64, int(n), int(n), .General, allocator) if compute_z else Matrix(f64){}

	// Workspace query
	work_query: f64
	lwork: Blas_Int = -1

	lapack.dhgeqz_(
		job_c,
		compq_c,
		compz_c,
		&n,
		&ilo_use,
		&ihi_use,
		raw_data(H.data),
		&ldh,
		raw_data(T.data),
		&ldt,
		raw_data(alphar),
		raw_data(alphai),
		raw_data(beta),
		matrix_data_ptr(&Q) if compute_q else nil,
		&ldq,
		matrix_data_ptr(&Z) if compute_z else nil,
		&ldz,
		&work_query,
		&lwork,
		&info,
		1,
		1,
		1,
	)

	if info != 0 {
		return
	}

	// Allocate and compute
	lwork = auto_cast work_query
	work := make([]f64, lwork, allocator)
	defer delete(work, allocator)

	lapack.dhgeqz_(
		job_c,
		compq_c,
		compz_c,
		&n,
		&ilo_use,
		&ihi_use,
		raw_data(H.data),
		&ldh,
		raw_data(T.data),
		&ldt,
		raw_data(alphar),
		raw_data(alphai),
		raw_data(beta),
		matrix_data_ptr(&Q) if compute_q else nil,
		&ldq,
		matrix_data_ptr(&Z) if compute_z else nil,
		&ldz,
		raw_data(work),
		&lwork,
		&info,
		1,
		1,
		1,
	)

	return
}

m_hgeqz_c64 :: proc(
	H: ^Matrix(complex64),
	T: ^Matrix(complex64),
	ilo: int = 0,
	ihi: int = 0,
	compute_eigenvalues: bool = true,
	compute_q: bool = false,
	compute_z: bool = false,
	allocator := context.allocator,
) -> (
	alpha, beta: []complex64,
	Q, Z: Matrix(complex64),
	info: Info,
) {
	assert(H.format == .General && T.format == .General, "Matrices must be general format")
	assert(H.rows == H.cols && T.rows == T.cols, "Matrices must be square")
	assert(H.rows == T.rows, "Matrices must have same dimensions")

	n := Blas_Int(H.rows)
	ldh := Blas_Int(H.ld)
	ldt := Blas_Int(T.ld)

	// Use default range if not specified
	ilo_use := Blas_Int(ilo if ilo > 0 else 1)
	ihi_use := Blas_Int(ihi if ihi > 0 else H.rows)

	job_c := cstring("E") if compute_eigenvalues else cstring("S")
	compq_c := cstring("V") if compute_q else cstring("N")
	compz_c := cstring("V") if compute_z else cstring("N")

	// Allocate outputs
	alpha = make([]complex64, n, allocator)
	beta = make([]complex64, n, allocator)

	ldq := n if compute_q else 1
	ldz := n if compute_z else 1
	Q =
		make_matrix(complex64, int(n), int(n), .General, allocator) if compute_q else Matrix(complex64){}
	Z =
		make_matrix(complex64, int(n), int(n), .General, allocator) if compute_z else Matrix(complex64){}

	// Workspace query
	work_query: complex64
	lwork: Blas_Int = -1
	rwork := make([]f32, n, allocator)
	defer delete(rwork, allocator)

	lapack.chgeqz_(
		job_c,
		compq_c,
		compz_c,
		&n,
		&ilo_use,
		&ihi_use,
		raw_data(H.data),
		&ldh,
		raw_data(T.data),
		&ldt,
		raw_data(alpha),
		raw_data(beta),
		matrix_data_ptr(&Q) if compute_q else nil,
		&ldq,
		matrix_data_ptr(&Z) if compute_z else nil,
		&ldz,
		&work_query,
		&lwork,
		raw_data(rwork),
		&info,
		1,
		1,
		1,
	)

	if info != 0 {
		return
	}

	// Allocate and compute
	lwork = auto_cast real(work_query)
	work := make([]complex64, lwork, allocator)
	defer delete(work, allocator)

	lapack.chgeqz_(
		job_c,
		compq_c,
		compz_c,
		&n,
		&ilo_use,
		&ihi_use,
		raw_data(H.data),
		&ldh,
		raw_data(T.data),
		&ldt,
		raw_data(alpha),
		raw_data(beta),
		matrix_data_ptr(&Q) if compute_q else nil,
		&ldq,
		matrix_data_ptr(&Z) if compute_z else nil,
		&ldz,
		raw_data(work),
		&lwork,
		raw_data(rwork),
		&info,
		1,
		1,
		1,
	)

	return
}

m_hgeqz_c128 :: proc(
	H: ^Matrix(complex128),
	T: ^Matrix(complex128),
	ilo: int = 0,
	ihi: int = 0,
	compute_eigenvalues: bool = true,
	compute_q: bool = false,
	compute_z: bool = false,
	allocator := context.allocator,
) -> (
	alpha, beta: []complex128,
	Q, Z: Matrix(complex128),
	info: Info,
) {
	assert(H.format == .General && T.format == .General, "Matrices must be general format")
	assert(H.rows == H.cols && T.rows == T.cols, "Matrices must be square")
	assert(H.rows == T.rows, "Matrices must have same dimensions")

	n := Blas_Int(H.rows)
	ldh := Blas_Int(H.ld)
	ldt := Blas_Int(T.ld)

	// Use default range if not specified
	ilo_use := Blas_Int(ilo if ilo > 0 else 1)
	ihi_use := Blas_Int(ihi if ihi > 0 else H.rows)

	job_c := cstring("E") if compute_eigenvalues else cstring("S")
	compq_c := cstring("V") if compute_q else cstring("N")
	compz_c := cstring("V") if compute_z else cstring("N")

	// Allocate outputs
	alpha = make([]complex128, n, allocator)
	beta = make([]complex128, n, allocator)

	ldq := n if compute_q else 1
	ldz := n if compute_z else 1
	Q =
		make_matrix(complex128, int(n), int(n), .General, allocator) if compute_q else Matrix(complex128){}
	Z =
		make_matrix(complex128, int(n), int(n), .General, allocator) if compute_z else Matrix(complex128){}

	// Workspace query
	work_query: complex128
	lwork: Blas_Int = -1
	rwork := make([]f64, n, allocator)
	defer delete(rwork, allocator)

	lapack.zhgeqz_(
		job_c,
		compq_c,
		compz_c,
		&n,
		&ilo_use,
		&ihi_use,
		raw_data(H.data),
		&ldh,
		raw_data(T.data),
		&ldt,
		raw_data(alpha),
		raw_data(beta),
		matrix_data_ptr(&Q) if compute_q else nil,
		&ldq,
		matrix_data_ptr(&Z) if compute_z else nil,
		&ldz,
		&work_query,
		&lwork,
		raw_data(rwork),
		&info,
		1,
		1,
		1,
	)

	if info != 0 {
		return
	}

	// Allocate and compute
	lwork = auto_cast real(work_query)
	work := make([]complex128, lwork, allocator)
	defer delete(work, allocator)

	lapack.zhgeqz_(
		job_c,
		compq_c,
		compz_c,
		&n,
		&ilo_use,
		&ihi_use,
		raw_data(H.data),
		&ldh,
		raw_data(T.data),
		&ldt,
		raw_data(alpha),
		raw_data(beta),
		matrix_data_ptr(&Q) if compute_q else nil,
		&ldq,
		matrix_data_ptr(&Z) if compute_z else nil,
		&ldz,
		raw_data(work),
		&lwork,
		raw_data(rwork),
		&info,
		1,
		1,
		1,
	)

	return
}

// ===================================================================================
// HERMITIAN PACKED MATRIX OPERATIONS
// ===================================================================================

// Condition number estimation for Hermitian packed matrix
m_condition_hermitian_packed :: proc {
	m_condition_hermitian_packed_c64,
	m_condition_hermitian_packed_c128,
}

m_condition_hermitian_packed_c64 :: proc(
	AP: ^Matrix(complex64),
	ipiv: ^[]int,
	anorm: f32,
	allocator := context.allocator,
) -> (
	rcond: f32,
	ok: bool,
) {
	assert(AP.format == .Packed, "Matrix must be in packed format")

	n := Blas_Int(AP.storage.packed.n)
	uplo_c := AP.storage.packed.uplo

	// Convert ipiv to LAPACK format
	ipiv_lapack := make([]Blas_Int, n, allocator)
	defer delete(ipiv_lapack, allocator)
	for i in 0 ..< n {
		ipiv_lapack[i] = Blas_Int(ipiv[i])
	}

	// Workspace
	work := make([]complex64, 2 * n, allocator)
	defer delete(work, allocator)

	info: Info
	lapack.chpcon_(
		uplo_c,
		&n,
		raw_data(AP.data),
		raw_data(ipiv_lapack),
		&anorm,
		&rcond,
		raw_data(work),
		&info,
		1,
	)

	return rcond, info == 0
}

m_condition_hermitian_packed_c128 :: proc(
	AP: ^Matrix(complex128),
	ipiv: ^[]int,
	anorm: f64,
	allocator := context.allocator,
) -> (
	rcond: f64,
	ok: bool,
) {
	assert(AP.format == .Packed, "Matrix must be in packed format")

	n := Blas_Int(AP.storage.packed.n)
	uplo_c := AP.storage.packed.uplo

	// Convert ipiv to LAPACK format
	ipiv_lapack := make([]Blas_Int, n, allocator)
	defer delete(ipiv_lapack, allocator)
	for i in 0 ..< n {
		ipiv_lapack[i] = Blas_Int(ipiv[i])
	}

	// Workspace
	work := make([]complex128, 2 * n, allocator)
	defer delete(work, allocator)

	info: Info
	lapack.zhpcon_(
		uplo_c,
		&n,
		raw_data(AP.data),
		raw_data(ipiv_lapack),
		&anorm,
		&rcond,
		raw_data(work),
		&info,
		1,
	)

	return rcond, info == 0
}

// Eigenvalue computation for Hermitian packed matrix (basic)
m_eigenvalues_hermitian_packed :: proc {
	m_eigenvalues_hermitian_packed_c64,
	m_eigenvalues_hermitian_packed_c128,
}

m_eigenvalues_hermitian_packed_c64 :: proc(
	AP: ^Matrix(complex64),
	compute_vectors: bool = false,
	allocator := context.allocator,
) -> (
	w: []f32,
	Z: Matrix(complex64),
	info: Info,
) {
	assert(AP.format == .Packed, "Matrix must be in packed format")

	n := Blas_Int(AP.storage.packed.n)
	uplo_c := AP.storage.packed.uplo
	jobz_c := cstring("V") if compute_vectors else cstring("N")

	// Allocate outputs
	w = make([]f32, n, allocator)
	ldz := n if compute_vectors else 1
	Z =
		make_matrix(complex64, int(n), int(n), .General, allocator) if compute_vectors else Matrix(complex64){}

	// Workspace
	work := make([]complex64, max(1, 2 * n - 1), allocator)
	defer delete(work, allocator)
	rwork := make([]f32, max(1, 3 * n - 2), allocator)
	defer delete(rwork, allocator)

	lapack.chpev_(
		jobz_c,
		uplo_c,
		&n,
		raw_data(AP.data),
		raw_data(w),
		matrix_data_ptr(&Z) if compute_vectors else nil,
		&ldz,
		raw_data(work),
		raw_data(rwork),
		&info,
		1,
		1,
	)

	return
}

m_eigenvalues_hermitian_packed_c128 :: proc(
	AP: ^Matrix(complex128),
	compute_vectors: bool = false,
	allocator := context.allocator,
) -> (
	w: []f64,
	Z: Matrix(complex128),
	info: Info,
) {
	assert(AP.format == .Packed, "Matrix must be in packed format")

	n := Blas_Int(AP.storage.packed.n)
	uplo_c := AP.storage.packed.uplo
	jobz_c := cstring("V") if compute_vectors else cstring("N")

	// Allocate outputs
	w = make([]f64, n, allocator)
	ldz := n if compute_vectors else 1
	Z =
		make_matrix(complex128, int(n), int(n), .General, allocator) if compute_vectors else Matrix(complex128){}

	// Workspace
	work := make([]complex128, max(1, 2 * n - 1), allocator)
	defer delete(work, allocator)
	rwork := make([]f64, max(1, 3 * n - 2), allocator)
	defer delete(rwork, allocator)

	lapack.zhpev_(
		jobz_c,
		uplo_c,
		&n,
		raw_data(AP.data),
		raw_data(w),
		matrix_data_ptr(&Z) if compute_vectors else nil,
		&ldz,
		raw_data(work),
		raw_data(rwork),
		&info,
		1,
		1,
	)

	return
}

// Eigenvalue computation using divide-and-conquer (faster for large matrices)
m_eigenvalues_hermitian_packed_dc :: proc {
	m_eigenvalues_hermitian_packed_dc_c64,
	m_eigenvalues_hermitian_packed_dc_c128,
}

m_eigenvalues_hermitian_packed_dc_c64 :: proc(
	AP: ^Matrix(complex64),
	compute_vectors: bool = false,
	allocator := context.allocator,
) -> (
	w: []f32,
	Z: Matrix(complex64),
	work_size, rwork_size, iwork_size: int,
	info: Info,
) {
	assert(AP.format == .Packed, "Matrix must be in packed format")

	n := Blas_Int(AP.storage.packed.n)
	uplo_c := AP.storage.packed.uplo
	jobz_c := cstring("V") if compute_vectors else cstring("N")

	// Allocate outputs
	w = make([]f32, n, allocator)
	ldz := n if compute_vectors else 1
	Z =
		make_matrix(complex64, int(n), int(n), .General, allocator) if compute_vectors else Matrix(complex64){}

	// Workspace queries
	work_query: complex64
	rwork_query: f32
	iwork_query: Blas_Int
	lwork: Blas_Int = -1
	lrwork: Blas_Int = -1
	liwork: Blas_Int = -1

	lapack.chpevd_(
		jobz_c,
		uplo_c,
		&n,
		raw_data(AP.data),
		raw_data(w),
		matrix_data_ptr(&Z) if compute_vectors else nil,
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

	if info != 0 {
		return
	}

	// Allocate and compute
	lwork = auto_cast real(work_query)
	lrwork = auto_cast rwork_query
	liwork = iwork_query

	work := make([]complex64, lwork, allocator)
	defer delete(work, allocator)
	rwork := make([]f32, lrwork, allocator)
	defer delete(rwork, allocator)
	iwork := make([]Blas_Int, liwork, allocator)
	defer delete(iwork, allocator)

	lapack.chpevd_(
		jobz_c,
		uplo_c,
		&n,
		raw_data(AP.data),
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

	return w, Z, int(lwork), int(lrwork), int(liwork), info
}

m_eigenvalues_hermitian_packed_dc_c128 :: proc(
	AP: ^Matrix(complex128),
	compute_vectors: bool = false,
	allocator := context.allocator,
) -> (
	w: []f64,
	Z: Matrix(complex128),
	work_size, rwork_size, iwork_size: int,
	info: Info,
) {
	assert(AP.format == .Packed, "Matrix must be in packed format")

	n := Blas_Int(AP.storage.packed.n)
	uplo_c := AP.storage.packed.uplo
	jobz_c := cstring("V") if compute_vectors else cstring("N")

	// Allocate outputs
	w = make([]f64, n, allocator)
	ldz := n if compute_vectors else 1
	Z =
		make_matrix(complex128, int(n), int(n), .General, allocator) if compute_vectors else Matrix(complex128){}

	// Workspace queries
	work_query: complex128
	rwork_query: f64
	iwork_query: Blas_Int
	lwork: Blas_Int = -1
	lrwork: Blas_Int = -1
	liwork: Blas_Int = -1

	lapack.zhpevd_(
		jobz_c,
		uplo_c,
		&n,
		raw_data(AP.data),
		raw_data(w),
		matrix_data_ptr(&Z) if compute_vectors else nil,
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

	if info != 0 {
		return
	}

	// Allocate and compute
	lwork = auto_cast real(work_query)
	lrwork = auto_cast rwork_query
	liwork = iwork_query

	work := make([]complex128, lwork, allocator)
	defer delete(work, allocator)
	rwork := make([]f64, lrwork, allocator)
	defer delete(rwork, allocator)
	iwork := make([]Blas_Int, liwork, allocator)
	defer delete(iwork, allocator)

	lapack.zhpevd_(
		jobz_c,
		uplo_c,
		&n,
		raw_data(AP.data),
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

	return w, Z, int(lwork), int(lrwork), int(liwork), info
}

// Expert eigenvalue computation with selective computation
m_eigenvalues_hermitian_packed_expert :: proc {
	m_eigenvalues_hermitian_packed_expert_c64,
	m_eigenvalues_hermitian_packed_expert_c128,
}

m_eigenvalues_hermitian_packed_expert_c64 :: proc(
	AP: ^Matrix(complex64),
	vl: f32 = 0,
	vu: f32 = 0,
	il: int = 0,
	iu: int = 0,
	abstol: f32 = 0,
	compute_vectors: bool = false,
	range_values: bool = false,
	range_indices: bool = false,
	allocator := context.allocator,
) -> (
	w: []f32,
	Z: Matrix(complex64),
	m: int,
	ifail: []int,
	info: Info,
) {
	assert(AP.format == .Packed, "Matrix must be in packed format")

	n := Blas_Int(AP.storage.packed.n)
	uplo_c := AP.storage.packed.uplo
	jobz_c := cstring("V") if compute_vectors else cstring("N")

	// Determine range type
	range_c: cstring
	if range_values {
		range_c = cstring("V")
	} else if range_indices {
		range_c = cstring("I")
	} else {
		range_c = cstring("A")
	}

	vl_use := vl
	vu_use := vu
	il_use := Blas_Int(il if il > 0 else 1)
	iu_use := Blas_Int(iu if iu > 0 else int(n))
	abstol_use := abstol

	// Allocate outputs
	w = make([]f32, n, allocator)
	ldz := n if compute_vectors else 1
	Z =
		make_matrix(complex64, int(n), int(n), .General, allocator) if compute_vectors else Matrix(complex64){}
	ifail = make([]int, n, allocator)

	// Workspace
	work := make([]complex64, 2 * n, allocator)
	defer delete(work, allocator)
	rwork := make([]f32, 7 * n, allocator)
	defer delete(rwork, allocator)
	iwork := make([]Blas_Int, 5 * n, allocator)
	defer delete(iwork, allocator)
	ifail_lapack := make([]Blas_Int, n, allocator)
	defer delete(ifail_lapack, allocator)

	m_out: Blas_Int
	lapack.chpevx_(
		jobz_c,
		range_c,
		uplo_c,
		&n,
		raw_data(AP.data),
		&vl_use,
		&vu_use,
		&il_use,
		&iu_use,
		&abstol_use,
		&m_out,
		raw_data(w),
		matrix_data_ptr(&Z) if compute_vectors else nil,
		&ldz,
		raw_data(work),
		raw_data(rwork),
		raw_data(iwork),
		raw_data(ifail_lapack),
		&info,
		1,
		1,
		1,
	)

	// Convert ifail back to int
	for i in 0 ..< n {
		ifail[i] = int(ifail_lapack[i])
	}

	return w, Z, int(m_out), ifail, info
}

m_eigenvalues_hermitian_packed_expert_c128 :: proc(
	AP: ^Matrix(complex128),
	vl: f64 = 0,
	vu: f64 = 0,
	il: int = 0,
	iu: int = 0,
	abstol: f64 = 0,
	compute_vectors: bool = false,
	range_values: bool = false,
	range_indices: bool = false,
	allocator := context.allocator,
) -> (
	w: []f64,
	Z: Matrix(complex128),
	m: int,
	ifail: []int,
	info: Info,
) {
	assert(AP.format == .Packed, "Matrix must be in packed format")

	n := Blas_Int(AP.storage.packed.n)
	uplo_c := AP.storage.packed.uplo
	jobz_c := cstring("V") if compute_vectors else cstring("N")

	// Determine range type
	range_c: cstring
	if range_values {
		range_c = cstring("V")
	} else if range_indices {
		range_c = cstring("I")
	} else {
		range_c = cstring("A")
	}

	vl_use := vl
	vu_use := vu
	il_use := Blas_Int(il if il > 0 else 1)
	iu_use := Blas_Int(iu if iu > 0 else int(n))
	abstol_use := abstol

	// Allocate outputs
	w = make([]f64, n, allocator)
	ldz := n if compute_vectors else 1
	Z =
		make_matrix(complex128, int(n), int(n), .General, allocator) if compute_vectors else Matrix(complex128){}
	ifail = make([]int, n, allocator)

	// Workspace
	work := make([]complex128, 2 * n, allocator)
	defer delete(work, allocator)
	rwork := make([]f64, 7 * n, allocator)
	defer delete(rwork, allocator)
	iwork := make([]Blas_Int, 5 * n, allocator)
	defer delete(iwork, allocator)
	ifail_lapack := make([]Blas_Int, n, allocator)
	defer delete(ifail_lapack, allocator)

	m_out: Blas_Int
	lapack.zhpevx_(
		jobz_c,
		range_c,
		uplo_c,
		&n,
		raw_data(AP.data),
		&vl_use,
		&vu_use,
		&il_use,
		&iu_use,
		&abstol_use,
		&m_out,
		raw_data(w),
		matrix_data_ptr(&Z) if compute_vectors else nil,
		&ldz,
		raw_data(work),
		raw_data(rwork),
		raw_data(iwork),
		raw_data(ifail_lapack),
		&info,
		1,
		1,
		1,
	)

	// Convert ifail back to int
	for i in 0 ..< n {
		ifail[i] = int(ifail_lapack[i])
	}

	return w, Z, int(m_out), ifail, info
}

// ===================================================================================
// GENERALIZED HERMITIAN PACKED MATRIX EIGENVALUE PROBLEMS
// ===================================================================================

// Transform generalized packed eigenvalue problem to standard form
m_transform_hermitian_packed :: proc {
	m_transform_hermitian_packed_c64,
	m_transform_hermitian_packed_c128,
}

m_transform_hermitian_packed_c64 :: proc(
	AP: ^Matrix(complex64),
	BP: ^Matrix(complex64),
	itype: int = 1,
) -> (
	ok: bool,
) {
	assert(AP.format == .Packed && BP.format == .Packed, "Both matrices must be in packed format")
	assert(AP.storage.packed.n == BP.storage.packed.n, "Matrices must have same dimensions")

	n := Blas_Int(AP.storage.packed.n)
	uplo_c := AP.storage.packed.uplo
	itype_use := Blas_Int(itype)

	info: Info
	lapack.chpgst_(&itype_use, uplo_c, &n, raw_data(AP.data), raw_data(BP.data), &info, 1)

	return info == 0
}

m_transform_hermitian_packed_c128 :: proc(
	AP: ^Matrix(complex128),
	BP: ^Matrix(complex128),
	itype: int = 1,
) -> (
	ok: bool,
) {
	assert(AP.format == .Packed && BP.format == .Packed, "Both matrices must be in packed format")
	assert(AP.storage.packed.n == BP.storage.packed.n, "Matrices must have same dimensions")

	n := Blas_Int(AP.storage.packed.n)
	uplo_c := AP.storage.packed.uplo
	itype_use := Blas_Int(itype)

	info: Info
	lapack.zhpgst_(&itype_use, uplo_c, &n, raw_data(AP.data), raw_data(BP.data), &info, 1)

	return info == 0
}

// Generalized eigenvalue problem for packed Hermitian matrices (basic)
m_generalized_eigenvalues_hermitian_packed :: proc {
	m_generalized_eigenvalues_hermitian_packed_c64,
	m_generalized_eigenvalues_hermitian_packed_c128,
}

m_generalized_eigenvalues_hermitian_packed_c64 :: proc(
	AP: ^Matrix(complex64),
	BP: ^Matrix(complex64),
	itype: int = 1,
	compute_vectors: bool = false,
	allocator := context.allocator,
) -> (
	w: []f32,
	Z: Matrix(complex64),
	info: Info,
) {
	assert(AP.format == .Packed && BP.format == .Packed, "Both matrices must be in packed format")
	assert(AP.storage.packed.n == BP.storage.packed.n, "Matrices must have same dimensions")

	n := Blas_Int(AP.storage.packed.n)
	uplo_c := AP.storage.packed.uplo
	itype_use := Blas_Int(itype)
	jobz_c := cstring("V") if compute_vectors else cstring("N")

	// Allocate outputs
	w = make([]f32, n, allocator)
	ldz := n if compute_vectors else 1
	Z =
		make_matrix(complex64, int(n), int(n), .General, allocator) if compute_vectors else Matrix(complex64){}

	// Workspace
	work := make([]complex64, max(1, 2 * n - 1), allocator)
	defer delete(work, allocator)
	rwork := make([]f32, max(1, 3 * n - 2), allocator)
	defer delete(rwork, allocator)

	lapack.chpgv_(
		&itype_use,
		jobz_c,
		uplo_c,
		&n,
		raw_data(AP.data),
		raw_data(BP.data),
		raw_data(w),
		matrix_data_ptr(&Z) if compute_vectors else nil,
		&ldz,
		raw_data(work),
		raw_data(rwork),
		&info,
		1,
		1,
	)

	return
}

m_generalized_eigenvalues_hermitian_packed_c128 :: proc(
	AP: ^Matrix(complex128),
	BP: ^Matrix(complex128),
	itype: int = 1,
	compute_vectors: bool = false,
	allocator := context.allocator,
) -> (
	w: []f64,
	Z: Matrix(complex128),
	info: Info,
) {
	assert(AP.format == .Packed && BP.format == .Packed, "Both matrices must be in packed format")
	assert(AP.storage.packed.n == BP.storage.packed.n, "Matrices must have same dimensions")

	n := Blas_Int(AP.storage.packed.n)
	uplo_c := AP.storage.packed.uplo
	itype_use := Blas_Int(itype)
	jobz_c := cstring("V") if compute_vectors else cstring("N")

	// Allocate outputs
	w = make([]f64, n, allocator)
	ldz := n if compute_vectors else 1
	Z =
		make_matrix(complex128, int(n), int(n), .General, allocator) if compute_vectors else Matrix(complex128){}

	// Workspace
	work := make([]complex128, max(1, 2 * n - 1), allocator)
	defer delete(work, allocator)
	rwork := make([]f64, max(1, 3 * n - 2), allocator)
	defer delete(rwork, allocator)

	lapack.zhpgv_(
		&itype_use,
		jobz_c,
		uplo_c,
		&n,
		raw_data(AP.data),
		raw_data(BP.data),
		raw_data(w),
		matrix_data_ptr(&Z) if compute_vectors else nil,
		&ldz,
		raw_data(work),
		raw_data(rwork),
		&info,
		1,
		1,
	)

	return
}

// Generalized eigenvalue problem using divide-and-conquer (faster for large matrices)
m_generalized_eigenvalues_hermitian_packed_dc :: proc {
	m_generalized_eigenvalues_hermitian_packed_dc_c64,
	m_generalized_eigenvalues_hermitian_packed_dc_c128,
}

m_generalized_eigenvalues_hermitian_packed_dc_c64 :: proc(
	AP: ^Matrix(complex64),
	BP: ^Matrix(complex64),
	itype: int = 1,
	compute_vectors: bool = false,
	allocator := context.allocator,
) -> (
	w: []f32,
	Z: Matrix(complex64),
	work_size, rwork_size, iwork_size: int,
	info: Info,
) {
	assert(AP.format == .Packed && BP.format == .Packed, "Both matrices must be in packed format")
	assert(AP.storage.packed.n == BP.storage.packed.n, "Matrices must have same dimensions")

	n := Blas_Int(AP.storage.packed.n)
	uplo_c := AP.storage.packed.uplo
	itype_use := Blas_Int(itype)
	jobz_c := cstring("V") if compute_vectors else cstring("N")

	// Allocate outputs
	w = make([]f32, n, allocator)
	ldz := n if compute_vectors else 1
	Z =
		make_matrix(complex64, int(n), int(n), .General, allocator) if compute_vectors else Matrix(complex64){}

	// Workspace queries
	work_query: complex64
	rwork_query: f32
	iwork_query: Blas_Int
	lwork: Blas_Int = -1
	lrwork: Blas_Int = -1
	liwork: Blas_Int = -1

	lapack.chpgvd_(
		&itype_use,
		jobz_c,
		uplo_c,
		&n,
		raw_data(AP.data),
		raw_data(BP.data),
		raw_data(w),
		matrix_data_ptr(&Z) if compute_vectors else nil,
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

	if info != 0 {
		return
	}

	// Allocate and compute
	lwork = auto_cast real(work_query)
	lrwork = auto_cast rwork_query
	liwork = iwork_query

	work := make([]complex64, lwork, allocator)
	defer delete(work, allocator)
	rwork := make([]f32, lrwork, allocator)
	defer delete(rwork, allocator)
	iwork := make([]Blas_Int, liwork, allocator)
	defer delete(iwork, allocator)

	lapack.chpgvd_(
		&itype_use,
		jobz_c,
		uplo_c,
		&n,
		raw_data(AP.data),
		raw_data(BP.data),
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

	return w, Z, int(lwork), int(lrwork), int(liwork), info
}

m_generalized_eigenvalues_hermitian_packed_dc_c128 :: proc(
	AP: ^Matrix(complex128),
	BP: ^Matrix(complex128),
	itype: int = 1,
	compute_vectors: bool = false,
	allocator := context.allocator,
) -> (
	w: []f64,
	Z: Matrix(complex128),
	work_size, rwork_size, iwork_size: int,
	info: Info,
) {
	assert(AP.format == .Packed && BP.format == .Packed, "Both matrices must be in packed format")
	assert(AP.storage.packed.n == BP.storage.packed.n, "Matrices must have same dimensions")

	n := Blas_Int(AP.storage.packed.n)
	uplo_c := AP.storage.packed.uplo
	itype_use := Blas_Int(itype)
	jobz_c := cstring("V") if compute_vectors else cstring("N")

	// Allocate outputs
	w = make([]f64, n, allocator)
	ldz := n if compute_vectors else 1
	Z =
		make_matrix(complex128, int(n), int(n), .General, allocator) if compute_vectors else Matrix(complex128){}

	// Workspace queries
	work_query: complex128
	rwork_query: f64
	iwork_query: Blas_Int
	lwork: Blas_Int = -1
	lrwork: Blas_Int = -1
	liwork: Blas_Int = -1

	lapack.zhpgvd_(
		&itype_use,
		jobz_c,
		uplo_c,
		&n,
		raw_data(AP.data),
		raw_data(BP.data),
		raw_data(w),
		matrix_data_ptr(&Z) if compute_vectors else nil,
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

	if info != 0 {
		return
	}

	// Allocate and compute
	lwork = auto_cast real(work_query)
	lrwork = auto_cast rwork_query
	liwork = iwork_query

	work := make([]complex128, lwork, allocator)
	defer delete(work, allocator)
	rwork := make([]f64, lrwork, allocator)
	defer delete(rwork, allocator)
	iwork := make([]Blas_Int, liwork, allocator)
	defer delete(iwork, allocator)

	lapack.zhpgvd_(
		&itype_use,
		jobz_c,
		uplo_c,
		&n,
		raw_data(AP.data),
		raw_data(BP.data),
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

	return w, Z, int(lwork), int(lrwork), int(liwork), info
}

// Expert generalized eigenvalue computation with selective computation
m_generalized_eigenvalues_hermitian_packed_expert :: proc {
	m_generalized_eigenvalues_hermitian_packed_expert_c64,
	m_generalized_eigenvalues_hermitian_packed_expert_c128,
}

m_generalized_eigenvalues_hermitian_packed_expert_c64 :: proc(
	AP: ^Matrix(complex64),
	BP: ^Matrix(complex64),
	itype: int = 1,
	vl: f32 = 0,
	vu: f32 = 0,
	il: int = 0,
	iu: int = 0,
	abstol: f32 = 0,
	compute_vectors: bool = false,
	range_values: bool = false,
	range_indices: bool = false,
	allocator := context.allocator,
) -> (
	w: []f32,
	Z: Matrix(complex64),
	m: int,
	ifail: []int,
	info: Info,
) {
	assert(AP.format == .Packed && BP.format == .Packed, "Both matrices must be in packed format")
	assert(AP.storage.packed.n == BP.storage.packed.n, "Matrices must have same dimensions")

	n := Blas_Int(AP.storage.packed.n)
	uplo_c := AP.storage.packed.uplo
	itype_use := Blas_Int(itype)
	jobz_c := cstring("V") if compute_vectors else cstring("N")

	// Determine range type
	range_c: cstring
	if range_values {
		range_c = cstring("V")
	} else if range_indices {
		range_c = cstring("I")
	} else {
		range_c = cstring("A")
	}

	vl_use := vl
	vu_use := vu
	il_use := Blas_Int(il if il > 0 else 1)
	iu_use := Blas_Int(iu if iu > 0 else int(n))
	abstol_use := abstol

	// Allocate outputs
	w = make([]f32, n, allocator)
	ldz := n if compute_vectors else 1
	Z =
		make_matrix(complex64, int(n), int(n), .General, allocator) if compute_vectors else Matrix(complex64){}
	ifail = make([]int, n, allocator)

	// Workspace
	work := make([]complex64, 2 * n, allocator)
	defer delete(work, allocator)
	rwork := make([]f32, 7 * n, allocator)
	defer delete(rwork, allocator)
	iwork := make([]Blas_Int, 5 * n, allocator)
	defer delete(iwork, allocator)
	ifail_lapack := make([]Blas_Int, n, allocator)
	defer delete(ifail_lapack, allocator)

	m_out: Blas_Int
	lapack.chpgvx_(
		&itype_use,
		jobz_c,
		range_c,
		uplo_c,
		&n,
		raw_data(AP.data),
		raw_data(BP.data),
		&vl_use,
		&vu_use,
		&il_use,
		&iu_use,
		&abstol_use,
		&m_out,
		raw_data(w),
		matrix_data_ptr(&Z) if compute_vectors else nil,
		&ldz,
		raw_data(work),
		raw_data(rwork),
		raw_data(iwork),
		raw_data(ifail_lapack),
		&info,
		1,
		1,
		1,
	)

	// Convert ifail back to int
	for i in 0 ..< n {
		ifail[i] = int(ifail_lapack[i])
	}

	return w, Z, int(m_out), ifail, info
}

m_generalized_eigenvalues_hermitian_packed_expert_c128 :: proc(
	AP: ^Matrix(complex128),
	BP: ^Matrix(complex128),
	itype: int = 1,
	vl: f64 = 0,
	vu: f64 = 0,
	il: int = 0,
	iu: int = 0,
	abstol: f64 = 0,
	compute_vectors: bool = false,
	range_values: bool = false,
	range_indices: bool = false,
	allocator := context.allocator,
) -> (
	w: []f64,
	Z: Matrix(complex128),
	m: int,
	ifail: []int,
	info: Info,
) {
	assert(AP.format == .Packed && BP.format == .Packed, "Both matrices must be in packed format")
	assert(AP.storage.packed.n == BP.storage.packed.n, "Matrices must have same dimensions")

	n := Blas_Int(AP.storage.packed.n)
	uplo_c := AP.storage.packed.uplo
	itype_use := Blas_Int(itype)
	jobz_c := cstring("V") if compute_vectors else cstring("N")

	// Determine range type
	range_c: cstring
	if range_values {
		range_c = cstring("V")
	} else if range_indices {
		range_c = cstring("I")
	} else {
		range_c = cstring("A")
	}

	vl_use := vl
	vu_use := vu
	il_use := Blas_Int(il if il > 0 else 1)
	iu_use := Blas_Int(iu if iu > 0 else int(n))
	abstol_use := abstol

	// Allocate outputs
	w = make([]f64, n, allocator)
	ldz := n if compute_vectors else 1
	Z =
		make_matrix(complex128, int(n), int(n), .General, allocator) if compute_vectors else Matrix(complex128){}
	ifail = make([]int, n, allocator)

	// Workspace
	work := make([]complex128, 2 * n, allocator)
	defer delete(work, allocator)
	rwork := make([]f64, 7 * n, allocator)
	defer delete(rwork, allocator)
	iwork := make([]Blas_Int, 5 * n, allocator)
	defer delete(iwork, allocator)
	ifail_lapack := make([]Blas_Int, n, allocator)
	defer delete(ifail_lapack, allocator)

	m_out: Blas_Int
	lapack.zhpgvx_(
		&itype_use,
		jobz_c,
		range_c,
		uplo_c,
		&n,
		raw_data(AP.data),
		raw_data(BP.data),
		&vl_use,
		&vu_use,
		&il_use,
		&iu_use,
		&abstol_use,
		&m_out,
		raw_data(w),
		matrix_data_ptr(&Z) if compute_vectors else nil,
		&ldz,
		raw_data(work),
		raw_data(rwork),
		raw_data(iwork),
		raw_data(ifail_lapack),
		&info,
		1,
		1,
		1,
	)

	// Convert ifail back to int
	for i in 0 ..< n {
		ifail[i] = int(ifail_lapack[i])
	}

	return w, Z, int(m_out), ifail, info
}

// ===================================================================================
// HERMITIAN PACKED MATRIX REFINEMENT AND SOLUTION
// ===================================================================================

// Iterative refinement for Hermitian packed matrix linear systems
m_refine_hermitian_packed :: proc {
	m_refine_hermitian_packed_c64,
	m_refine_hermitian_packed_c128,
}

m_refine_hermitian_packed_c64 :: proc(
	AP: ^Matrix(complex64),
	AFP: ^Matrix(complex64),
	ipiv: ^[]int,
	B: ^Matrix(complex64),
	X: ^Matrix(complex64),
	allocator := context.allocator,
) -> (
	ferr, berr: []f32,
	ok: bool,
) {
	assert(AP.format == .Packed && AFP.format == .Packed, "Both matrices must be in packed format")
	assert(AP.storage.packed.n == AFP.storage.packed.n, "Matrices must have same dimensions")
	assert(B.rows == X.rows && B.cols == X.cols, "B and X must have same dimensions")
	assert(AP.storage.packed.n == B.rows, "Matrix and RHS dimensions must match")

	n := Blas_Int(AP.storage.packed.n)
	nrhs := Blas_Int(B.cols)
	uplo_c := AP.storage.packed.uplo
	ldb := Blas_Int(B.ld)
	ldx := Blas_Int(X.ld)

	// Convert ipiv to LAPACK format
	ipiv_lapack := make([]Blas_Int, n, allocator)
	defer delete(ipiv_lapack, allocator)
	for i in 0 ..< n {
		ipiv_lapack[i] = Blas_Int(ipiv[i])
	}

	// Allocate outputs
	ferr = make([]f32, nrhs, allocator)
	berr = make([]f32, nrhs, allocator)

	// Workspace
	work := make([]complex64, 2 * n, allocator)
	defer delete(work, allocator)
	rwork := make([]f32, n, allocator)
	defer delete(rwork, allocator)

	info: Info
	lapack.chprfs_(
		uplo_c,
		&n,
		&nrhs,
		raw_data(AP.data),
		raw_data(AFP.data),
		raw_data(ipiv_lapack),
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

	return ferr, berr, info == 0
}

m_refine_hermitian_packed_c128 :: proc(
	AP: ^Matrix(complex128),
	AFP: ^Matrix(complex128),
	ipiv: ^[]int,
	B: ^Matrix(complex128),
	X: ^Matrix(complex128),
	allocator := context.allocator,
) -> (
	ferr, berr: []f64,
	ok: bool,
) {
	assert(AP.format == .Packed && AFP.format == .Packed, "Both matrices must be in packed format")
	assert(AP.storage.packed.n == AFP.storage.packed.n, "Matrices must have same dimensions")
	assert(B.rows == X.rows && B.cols == X.cols, "B and X must have same dimensions")
	assert(AP.storage.packed.n == B.rows, "Matrix and RHS dimensions must match")

	n := Blas_Int(AP.storage.packed.n)
	nrhs := Blas_Int(B.cols)
	uplo_c := AP.storage.packed.uplo
	ldb := Blas_Int(B.ld)
	ldx := Blas_Int(X.ld)

	// Convert ipiv to LAPACK format
	ipiv_lapack := make([]Blas_Int, n, allocator)
	defer delete(ipiv_lapack, allocator)
	for i in 0 ..< n {
		ipiv_lapack[i] = Blas_Int(ipiv[i])
	}

	// Allocate outputs
	ferr = make([]f64, nrhs, allocator)
	berr = make([]f64, nrhs, allocator)

	// Workspace
	work := make([]complex128, 2 * n, allocator)
	defer delete(work, allocator)
	rwork := make([]f64, n, allocator)
	defer delete(rwork, allocator)

	info: Info
	lapack.zhprfs_(
		uplo_c,
		&n,
		&nrhs,
		raw_data(AP.data),
		raw_data(AFP.data),
		raw_data(ipiv_lapack),
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

	return ferr, berr, info == 0
}

// Direct solution of Hermitian packed matrix linear systems
m_solve_hermitian_packed :: proc {
	m_solve_hermitian_packed_c64,
	m_solve_hermitian_packed_c128,
}

m_solve_hermitian_packed_c64 :: proc(
	AP: ^Matrix(complex64),
	B: ^Matrix(complex64),
	allocator := context.allocator,
) -> (
	ipiv: []int,
	ok: bool,
) {
	assert(AP.format == .Packed, "Matrix must be in packed format")
	assert(AP.storage.packed.n == B.rows, "Matrix and RHS dimensions must match")

	n := Blas_Int(AP.storage.packed.n)
	nrhs := Blas_Int(B.cols)
	uplo_c := AP.storage.packed.uplo
	ldb := Blas_Int(B.ld)

	// Allocate pivot array
	ipiv = make([]int, n, allocator)
	ipiv_lapack := make([]Blas_Int, n, allocator)
	defer delete(ipiv_lapack, allocator)

	info: Info
	lapack.chpsv_(
		uplo_c,
		&n,
		&nrhs,
		raw_data(AP.data),
		raw_data(ipiv_lapack),
		raw_data(B.data),
		&ldb,
		&info,
		1,
	)

	// Convert ipiv back to int
	for i in 0 ..< n {
		ipiv[i] = int(ipiv_lapack[i])
	}

	return ipiv, info == 0
}

m_solve_hermitian_packed_c128 :: proc(
	AP: ^Matrix(complex128),
	B: ^Matrix(complex128),
	allocator := context.allocator,
) -> (
	ipiv: []int,
	ok: bool,
) {
	assert(AP.format == .Packed, "Matrix must be in packed format")
	assert(AP.storage.packed.n == B.rows, "Matrix and RHS dimensions must match")

	n := Blas_Int(AP.storage.packed.n)
	nrhs := Blas_Int(B.cols)
	uplo_c := AP.storage.packed.uplo
	ldb := Blas_Int(B.ld)

	// Allocate pivot array
	ipiv = make([]int, n, allocator)
	ipiv_lapack := make([]Blas_Int, n, allocator)
	defer delete(ipiv_lapack, allocator)

	info: Info
	lapack.zhpsv_(
		uplo_c,
		&n,
		&nrhs,
		raw_data(AP.data),
		raw_data(ipiv_lapack),
		raw_data(B.data),
		&ldb,
		&info,
		1,
	)

	// Convert ipiv back to int
	for i in 0 ..< n {
		ipiv[i] = int(ipiv_lapack[i])
	}

	return ipiv, info == 0
}

// Expert solution with condition estimation and error bounds
m_solve_hermitian_packed_expert :: proc {
	m_solve_hermitian_packed_expert_c64,
	m_solve_hermitian_packed_expert_c128,
}

m_solve_hermitian_packed_expert_c64 :: proc(
	AP: ^Matrix(complex64),
	B: ^Matrix(complex64),
	factorize: bool = true,
	AFP: ^Matrix(complex64) = nil,
	ipiv_in: ^[]int = nil,
	allocator := context.allocator,
) -> (
	X: Matrix(complex64),
	ipiv: []int,
	rcond: f32,
	ferr, berr: []f32,
	info: Info,
) {
	assert(AP.format == .Packed, "Matrix must be in packed format")
	assert(AP.storage.packed.n == B.rows, "Matrix and RHS dimensions must match")

	n := Blas_Int(AP.storage.packed.n)
	nrhs := Blas_Int(B.cols)
	uplo_c := AP.storage.packed.uplo
	ldb := Blas_Int(B.ld)

	fact_c := cstring("N") if factorize else cstring("F")

	// Allocate outputs
	X = make_matrix(complex64, int(n), int(nrhs), .General, allocator)
	ldx := Blas_Int(X.ld)
	ferr = make([]f32, nrhs, allocator)
	berr = make([]f32, nrhs, allocator)

	// Handle factorization arrays
	AFP_work: Matrix(complex64)
	if factorize {
		AFP_work = make_packed_matrix(complex64, int(n), uplo_c, allocator)
		defer delete_matrix(&AFP_work)
	} else {
		assert(AFP != nil, "AFP must be provided when factorize=false")
		AFP_work = AFP^
	}

	// Handle pivot array
	if factorize {
		ipiv = make([]int, n, allocator)
	} else {
		assert(ipiv_in != nil, "ipiv must be provided when factorize=false")
		ipiv = make([]int, len(ipiv_in^), allocator)
		copy(ipiv, ipiv_in^)
	}

	ipiv_lapack := make([]Blas_Int, n, allocator)
	defer delete(ipiv_lapack, allocator)
	if !factorize {
		for i in 0 ..< n {
			ipiv_lapack[i] = Blas_Int(ipiv[i])
		}
	}

	// Workspace
	work := make([]complex64, 2 * n, allocator)
	defer delete(work, allocator)
	rwork := make([]f32, n, allocator)
	defer delete(rwork, allocator)

	lapack.chpsvx_(
		fact_c,
		uplo_c,
		&n,
		&nrhs,
		raw_data(AP.data),
		raw_data(AFP_work.data),
		raw_data(ipiv_lapack),
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

	// Convert ipiv back to int
	for i in 0 ..< n {
		ipiv[i] = int(ipiv_lapack[i])
	}

	return
}

m_solve_hermitian_packed_expert_c128 :: proc(
	AP: ^Matrix(complex128),
	B: ^Matrix(complex128),
	factorize: bool = true,
	AFP: ^Matrix(complex128) = nil,
	ipiv_in: ^[]int = nil,
	allocator := context.allocator,
) -> (
	X: Matrix(complex128),
	ipiv: []int,
	rcond: f64,
	ferr, berr: []f64,
	info: Info,
) {
	assert(AP.format == .Packed, "Matrix must be in packed format")
	assert(AP.storage.packed.n == B.rows, "Matrix and RHS dimensions must match")

	n := Blas_Int(AP.storage.packed.n)
	nrhs := Blas_Int(B.cols)
	uplo_c := AP.storage.packed.uplo
	ldb := Blas_Int(B.ld)

	fact_c := cstring("N") if factorize else cstring("F")

	// Allocate outputs
	X = make_matrix(complex128, int(n), int(nrhs), .General, allocator)
	ldx := Blas_Int(X.ld)
	ferr = make([]f64, nrhs, allocator)
	berr = make([]f64, nrhs, allocator)

	// Handle factorization arrays
	AFP_work: Matrix(complex128)
	if factorize {
		AFP_work = make_packed_matrix(complex128, int(n), uplo_c, allocator)
		defer delete_matrix(&AFP_work)
	} else {
		assert(AFP != nil, "AFP must be provided when factorize=false")
		AFP_work = AFP^
	}

	// Handle pivot array
	if factorize {
		ipiv = make([]int, n, allocator)
	} else {
		assert(ipiv_in != nil, "ipiv must be provided when factorize=false")
		ipiv = make([]int, len(ipiv_in^), allocator)
		copy(ipiv, ipiv_in^)
	}

	ipiv_lapack := make([]Blas_Int, n, allocator)
	defer delete(ipiv_lapack, allocator)
	if !factorize {
		for i in 0 ..< n {
			ipiv_lapack[i] = Blas_Int(ipiv[i])
		}
	}

	// Workspace
	work := make([]complex128, 2 * n, allocator)
	defer delete(work, allocator)
	rwork := make([]f64, n, allocator)
	defer delete(rwork, allocator)

	lapack.zhpsvx_(
		fact_c,
		uplo_c,
		&n,
		&nrhs,
		raw_data(AP.data),
		raw_data(AFP_work.data),
		raw_data(ipiv_lapack),
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

	// Convert ipiv back to int
	for i in 0 ..< n {
		ipiv[i] = int(ipiv_lapack[i])
	}

	return
}

// Reduce Hermitian packed matrix to tridiagonal form
m_reduce_tridiagonal_hermitian_packed :: proc {
	m_reduce_tridiagonal_hermitian_packed_c64,
	m_reduce_tridiagonal_hermitian_packed_c128,
}

m_reduce_tridiagonal_hermitian_packed_c64 :: proc(
	AP: ^Matrix(complex64),
	allocator := context.allocator,
) -> (
	D, E: []f32,
	tau: []complex64,
	ok: bool,
) {
	assert(AP.format == .Packed, "Matrix must be in packed format")

	n := Blas_Int(AP.storage.packed.n)
	uplo_c := AP.storage.packed.uplo

	// Allocate outputs
	D = make([]f32, n, allocator)
	E = make([]f32, n - 1, allocator) if n > 1 else make([]f32, 1, allocator)
	tau = make([]complex64, n - 1, allocator) if n > 1 else make([]complex64, 1, allocator)

	info: Info
	lapack.chptrd_(
		uplo_c,
		&n,
		raw_data(AP.data),
		raw_data(D),
		raw_data(E),
		raw_data(tau),
		&info,
		1,
	)

	return D, E, tau, info == 0
}

m_reduce_tridiagonal_hermitian_packed_c128 :: proc(
	AP: ^Matrix(complex128),
	allocator := context.allocator,
) -> (
	D, E: []f64,
	tau: []complex128,
	ok: bool,
) {
	assert(AP.format == .Packed, "Matrix must be in packed format")

	n := Blas_Int(AP.storage.packed.n)
	uplo_c := AP.storage.packed.uplo

	// Allocate outputs
	D = make([]f64, n, allocator)
	E = make([]f64, n - 1, allocator) if n > 1 else make([]f64, 1, allocator)
	tau = make([]complex128, n - 1, allocator) if n > 1 else make([]complex128, 1, allocator)

	info: Info
	lapack.zhptrd_(
		uplo_c,
		&n,
		raw_data(AP.data),
		raw_data(D),
		raw_data(E),
		raw_data(tau),
		&info,
		1,
	)

	return D, E, tau, info == 0
}

// ===================================================================================
// HERMITIAN PACKED MATRIX FACTORIZATION, INVERSION, AND SOLVING
// ===================================================================================

// Factorize a Hermitian packed matrix using Bunch-Kaufman diagonal pivoting method
m_factorize_hermitian_packed :: proc {
	m_factorize_hermitian_packed_c64,
	m_factorize_hermitian_packed_c128,
}

// Factorize Hermitian packed matrix (complex64)
m_factorize_hermitian_packed_c64 :: proc(
	AP: ^Matrix(complex64),
	upper: bool = true,
) -> (
	ipiv: []int,
	info: Info,
) {
	// Validate matrix
	if AP.format != .Packed {
		panic("Matrix must be in packed format")
	}

	n := Blas_Int(AP.storage.packed.n)
	uplo := cstring("U") if upper else cstring("L")

	// Allocate pivot array
	ipiv_raw := make([]Blas_Int, n, context.allocator)
	defer delete(ipiv_raw)

	// Call LAPACK
	info_val: Blas_Int
	lapack.chptrf_(&uplo, &n, raw_data(AP.data), raw_data(ipiv_raw), &info_val, 1)

	// Convert pivot indices to int
	ipiv = make([]int, n, context.allocator)
	for i in 0 ..< n {
		ipiv[i] = int(ipiv_raw[i])
	}

	return ipiv, info_val
}

// Factorize Hermitian packed matrix (complex128)
m_factorize_hermitian_packed_c128 :: proc(
	AP: ^Matrix(complex128),
	upper: bool = true,
) -> (
	ipiv: []int,
	info: Info,
) {
	// Validate matrix
	if AP.format != .Packed {
		panic("Matrix must be in packed format")
	}

	n := Blas_Int(AP.storage.packed.n)
	uplo := cstring("U") if upper else cstring("L")

	// Allocate pivot array
	ipiv_raw := make([]Blas_Int, n, context.allocator)
	defer delete(ipiv_raw)

	// Call LAPACK
	info_val: Blas_Int
	lapack.zhptrf_(&uplo, &n, raw_data(AP.data), raw_data(ipiv_raw), &info_val, 1)

	// Convert pivot indices to int
	ipiv = make([]int, n, context.allocator)
	for i in 0 ..< n {
		ipiv[i] = int(ipiv_raw[i])
	}

	return ipiv, info_val
}

// Invert a Hermitian packed matrix using the factorization from chptrf
m_invert_hermitian_packed :: proc {
	m_invert_hermitian_packed_c64,
	m_invert_hermitian_packed_c128,
}

// Invert Hermitian packed matrix (complex64)
m_invert_hermitian_packed_c64 :: proc(
	AP: ^Matrix(complex64),
	ipiv: []int,
	upper: bool = true,
	allocator := context.allocator,
) -> (
	info: Info,
) {
	// Validate matrix
	if AP.format != .Packed {
		panic("Matrix must be in packed format")
	}

	n := Blas_Int(AP.storage.packed.n)
	uplo := cstring("U") if upper else cstring("L")

	// Convert pivot indices
	ipiv_raw := make([]Blas_Int, len(ipiv), context.temp_allocator)
	for i, v in ipiv {
		ipiv_raw[i] = Blas_Int(v)
	}

	// Allocate workspace
	work := make([]complex64, n, context.temp_allocator)

	// Call LAPACK
	info_val: Blas_Int
	lapack.chptri_(&uplo, &n, raw_data(AP.data), raw_data(ipiv_raw), raw_data(work), &info_val, 1)

	return info_val
}

// Invert Hermitian packed matrix (complex128)
m_invert_hermitian_packed_c128 :: proc(
	AP: ^Matrix(complex128),
	ipiv: []int,
	upper: bool = true,
	allocator := context.allocator,
) -> (
	info: Info,
) {
	// Validate matrix
	if AP.format != .Packed {
		panic("Matrix must be in packed format")
	}

	n := Blas_Int(AP.storage.packed.n)
	uplo := cstring("U") if upper else cstring("L")

	// Convert pivot indices
	ipiv_raw := make([]Blas_Int, len(ipiv), context.temp_allocator)
	for i, v in ipiv {
		ipiv_raw[i] = Blas_Int(v)
	}

	// Allocate workspace
	work := make([]complex128, n, context.temp_allocator)

	// Call LAPACK
	info_val: Blas_Int
	lapack.zhptri_(&uplo, &n, raw_data(AP.data), raw_data(ipiv_raw), raw_data(work), &info_val, 1)

	return info_val
}

// Solve linear system using factorized Hermitian packed matrix
m_solve_factorized_hermitian_packed :: proc {
	m_solve_factorized_hermitian_packed_c64,
	m_solve_factorized_hermitian_packed_c128,
}

// Solve using factorized Hermitian packed matrix (complex64)
m_solve_factorized_hermitian_packed_c64 :: proc(
	AP: ^Matrix(complex64),
	ipiv: []int,
	B: ^Matrix(complex64),
	upper: bool = true,
) -> (
	info: Info,
) {
	// Validate matrices
	if AP.format != .Packed {
		panic("AP matrix must be in packed format")
	}
	if B.format != .General {
		panic("B matrix must be in general format")
	}

	n := Blas_Int(AP.storage.packed.n)
	nrhs := Blas_Int(B.cols)
	ldb := Blas_Int(B.ld)
	uplo := cstring("U") if upper else cstring("L")

	// Convert pivot indices
	ipiv_raw := make([]Blas_Int, len(ipiv), context.temp_allocator)
	for i, v in ipiv {
		ipiv_raw[i] = Blas_Int(v)
	}

	// Call LAPACK
	info_val: Blas_Int
	lapack.chptrs_(
		&uplo,
		&n,
		&nrhs,
		raw_data(AP.data),
		raw_data(ipiv_raw),
		raw_data(B.data),
		&ldb,
		&info_val,
		1,
	)

	return info_val
}

// Solve using factorized Hermitian packed matrix (complex128)
m_solve_factorized_hermitian_packed_c128 :: proc(
	AP: ^Matrix(complex128),
	ipiv: []int,
	B: ^Matrix(complex128),
	upper: bool = true,
) -> (
	info: Info,
) {
	// Validate matrices
	if AP.format != .Packed {
		panic("AP matrix must be in packed format")
	}
	if B.format != .General {
		panic("B matrix must be in general format")
	}

	n := Blas_Int(AP.storage.packed.n)
	nrhs := Blas_Int(B.cols)
	ldb := Blas_Int(B.ld)
	uplo := cstring("U") if upper else cstring("L")

	// Convert pivot indices
	ipiv_raw := make([]Blas_Int, len(ipiv), context.temp_allocator)
	for i, v in ipiv {
		ipiv_raw[i] = Blas_Int(v)
	}

	// Call LAPACK
	info_val: Blas_Int
	lapack.zhptrs_(
		&uplo,
		&n,
		&nrhs,
		raw_data(AP.data),
		raw_data(ipiv_raw),
		raw_data(B.data),
		&ldb,
		&info_val,
		1,
	)

	return info_val
}

// ===================================================================================
// HESSENBERG MATRIX EIGENVECTOR COMPUTATION
// ===================================================================================

// Compute selected eigenvectors of a general matrix by inverse iteration
m_eigenvectors_hessenberg :: proc {
	m_eigenvectors_hessenberg_c64,
	m_eigenvectors_hessenberg_c128,
	m_eigenvectors_hessenberg_f32,
	m_eigenvectors_hessenberg_f64,
}

// Compute eigenvectors of Hessenberg matrix (complex64)
m_eigenvectors_hessenberg_c64 :: proc(
	H: ^Matrix(complex64),
	eigenvalues: []complex64,
	compute_left: bool = false,
	compute_right: bool = true,
	eigsrc: bool = false, // true if eigenvalues are from schur form, false if computed separately
	select_mask: []bool = nil, // Which eigenvectors to compute (nil means all)
	allocator := context.allocator,
) -> (
	VL, VR: Matrix(complex64),
	m_computed: int,
	ifail: []int,
	info: Info, // Number of eigenvectors successfully computed// Failure indices
) {
	// Validate matrix
	if H.format != .General {
		panic("H matrix must be in general format")
	}

	n := Blas_Int(H.rows)
	if H.rows != H.cols {
		panic("H matrix must be square")
	}

	// Setup computation sides
	side_c :=
		cstring("B") if compute_left && compute_right else cstring("L") if compute_left else cstring("R")

	eigsrc_c := cstring("Q") if eigsrc else cstring("N")
	initv_c := cstring("N") // No initial vectors provided

	// Setup selection
	var; select_raw: []Blas_Int
	if select_mask != nil && len(select_mask) == int(n) {
		select_raw = make([]Blas_Int, n, context.temp_allocator)
		for i, selected in select_mask {
			select_raw[i] = 1 if selected else 0
		}
	} else {
		// Select all eigenvectors
		select_raw = make([]Blas_Int, n, context.temp_allocator)
		for i in 0 ..< n {
			select_raw[i] = 1
		}
	}

	ldh := Blas_Int(H.ld)
	ldvl := n if compute_left else 1
	ldvr := n if compute_right else 1

	// Allocate output matrices
	VL =
		matrix_zeros(complex64, int(ldvl), int(n), allocator) if compute_left else Matrix(complex64){}
	VR =
		matrix_zeros(complex64, int(ldvr), int(n), allocator) if compute_right else Matrix(complex64){}

	// Maximum number of eigenvectors
	mm := n
	m_out: Blas_Int

	// Copy eigenvalues
	W := make([]complex64, n, context.temp_allocator)
	copy(W, eigenvalues)

	// Allocate workspace
	work := make([]complex64, n * n, context.temp_allocator)
	rwork := make([]f32, n, context.temp_allocator)

	// Failure arrays
	ifaill := make([]Blas_Int, n, context.temp_allocator)
	ifailr := make([]Blas_Int, n, context.temp_allocator)

	// Call LAPACK
	info_val: Blas_Int
	vl_ptr := raw_data(VL.data) if compute_left else nil
	vr_ptr := raw_data(VR.data) if compute_right else nil

	lapack.chsein_(
		&side_c,
		&eigsrc_c,
		&initv_c,
		raw_data(select_raw),
		&n,
		raw_data(H.data),
		&ldh,
		raw_data(W),
		vl_ptr,
		&ldvl,
		vr_ptr,
		&ldvr,
		&mm,
		&m_out,
		raw_data(work),
		raw_data(rwork),
		raw_data(ifaill),
		raw_data(ifailr),
		&info_val,
		1,
		1,
		1,
	)

	// Convert failure indices
	ifail = make([]int, n, allocator)
	for i in 0 ..< n {
		if compute_left && ifaill[i] != 0 {
			ifail[i] = int(ifaill[i])
		} else if compute_right && ifailr[i] != 0 {
			ifail[i] = int(ifailr[i])
		}
	}

	return VL, VR, int(m_out), ifail, info_val
}

// ===================================================================================
// HESSENBERG QR ALGORITHM FOR EIGENVALUE COMPUTATION
// ===================================================================================

// Compute eigenvalues and optionally Schur form of Hessenberg matrix using QR algorithm
m_eigenvalues_hessenberg_qr :: proc {
	m_eigenvalues_hessenberg_qr_c64,
	m_eigenvalues_hessenberg_qr_c128,
	m_eigenvalues_hessenberg_qr_f32,
	m_eigenvalues_hessenberg_qr_f64,
}

// QR algorithm for Hessenberg matrix eigenvalues (complex64)
m_eigenvalues_hessenberg_qr_c64 :: proc(
	H: ^Matrix(complex64),
	ilo: int = 0, // Submatrix range (0 means use full matrix)
	ihi: int = 0, // Submatrix range (0 means use full matrix)
	compute_schur: bool = true,
	compute_eigenvectors: bool = false,
	Z: ^Matrix(complex64) = nil, // Optional input/output matrix for eigenvectors
	allocator := context.allocator,
) -> (
	eigenvalues: []complex64,
	Z_out: Matrix(complex64),
	success: bool,
	info: Info,
) {
	// Validate matrix
	if H.format != .General {
		panic("H matrix must be in general format")
	}

	n := Blas_Int(H.rows)
	if H.rows != H.cols {
		panic("H matrix must be square")
	}

	// Setup range
	ilo_val := Blas_Int(ilo if ilo > 0 else 1)
	ihi_val := Blas_Int(ihi if ihi > 0 else int(n))

	// Setup job parameters
	job_c := cstring("S") if compute_schur else cstring("E") // Schur form or eigenvalues only
	compz_c := cstring("V") if compute_eigenvectors else cstring("N") // Compute or don't compute Z

	ldh := Blas_Int(H.ld)

	// Setup Z matrix
	ldz: Blas_Int
	if compute_eigenvectors {
		if Z != nil {
			if Z.rows != H.rows || Z.cols != H.cols {
				panic("Z matrix dimensions must match H matrix")
			}
			Z_out = Z^
			ldz = Blas_Int(Z.ld)
		} else {
			Z_out = matrix_zeros(complex64, int(n), int(n), allocator)
			ldz = n
		}
	} else {
		Z_out = Matrix(complex64){}
		ldz = 1
	}

	// Allocate eigenvalue array
	W := make([]complex64, n, allocator)

	// Workspace query
	work_query: complex64
	lwork_query := Blas_Int(-1)
	info_val: Blas_Int

	z_ptr := raw_data(Z_out.data) if compute_eigenvectors else nil

	lapack.chseqr_(
		&job_c,
		&compz_c,
		&n,
		&ilo_val,
		&ihi_val,
		raw_data(H.data),
		&ldh,
		raw_data(W),
		z_ptr,
		&ldz,
		&work_query,
		&lwork_query,
		&info_val,
		1,
		1,
	)

	if info_val != 0 {
		return nil, Z_out, false, info_val
	}

	// Allocate workspace
	lwork := Blas_Int(real(work_query))
	work := make([]complex64, lwork, context.temp_allocator)

	// Compute eigenvalues
	lapack.chseqr_(
		&job_c,
		&compz_c,
		&n,
		&ilo_val,
		&ihi_val,
		raw_data(H.data),
		&ldh,
		raw_data(W),
		z_ptr,
		&ldz,
		raw_data(work),
		&lwork,
		&info_val,
		1,
		1,
	)

	eigenvalues = W
	return eigenvalues, Z_out, info_val == 0, info_val
}

// ===================================================================================
// LAPACK UTILITY FUNCTIONS
// ===================================================================================

// Complex conjugate of a vector
m_conjugate_vector :: proc {
	m_conjugate_vector_c64,
	m_conjugate_vector_c128,
}

// Conjugate complex vector (complex64)
m_conjugate_vector_c64 :: proc(X: ^Vector(complex64)) {
	n := Blas_Int(len(X.data))
	incx := Blas_Int(X.incr)

	lapack.clacgv_(&n, raw_data(X.data), &incx)
}

// Conjugate complex vector (complex128)
m_conjugate_vector_c128 :: proc(X: ^Vector(complex128)) {
	n := Blas_Int(len(X.data))
	incx := Blas_Int(X.incr)

	lapack.zlacgv_(&n, raw_data(X.data), &incx)
}

// Estimate the 1-norm of a matrix using iterative method
m_estimate_1norm :: proc {
	m_estimate_1norm_c64,
	m_estimate_1norm_c128,
	m_estimate_1norm_f32,
	m_estimate_1norm_f64,
}

// Estimate 1-norm of matrix (complex64)
m_estimate_1norm_c64 :: proc(
	n: int,
	apply_matrix: proc(X: []complex64, Y: []complex64), // User-provided matrix application
	allocator := context.allocator,
) -> (
	estimate: f32,
	iterations: int,
) {
	n_blas := Blas_Int(n)

	// Allocate workspace
	V := make([]complex64, n, context.temp_allocator)
	X := make([]complex64, n, context.temp_allocator)

	est: f32
	kase: Blas_Int
	isave := make([]Blas_Int, 3, context.temp_allocator)

	iterations = 0

	// Iterative estimation loop
	for {
		lapack.clacn2_(&n_blas, raw_data(V), raw_data(X), &est, &kase, raw_data(isave))

		if kase == 0 {
			break
		}

		iterations += 1

		if kase == 1 {
			// Apply matrix: Y := A * X
			apply_matrix(X, V)
		} else {
			// Apply conjugate transpose: Y := A^H * X
			apply_matrix(X, V) // User should handle transpose internally
		}
	}

	return est, iterations
}

// Estimate 1-norm of matrix (complex128)
m_estimate_1norm_c128 :: proc(
	n: int,
	apply_matrix: proc(X: []complex128, Y: []complex128),
	allocator := context.allocator,
) -> (
	estimate: f64,
	iterations: int,
) {
	n_blas := Blas_Int(n)

	// Allocate workspace
	V := make([]complex128, n, context.temp_allocator)
	X := make([]complex128, n, context.temp_allocator)

	est: f64
	kase: Blas_Int
	isave := make([]Blas_Int, 3, context.temp_allocator)

	iterations = 0

	// Iterative estimation loop
	for {
		lapack.zlacn2_(&n_blas, raw_data(V), raw_data(X), &est, &kase, raw_data(isave))

		if kase == 0 {
			break
		}

		iterations += 1

		if kase == 1 {
			// Apply matrix: Y := A * X
			apply_matrix(X, V)
		} else {
			// Apply conjugate transpose: Y := A^H * X
			apply_matrix(X, V) // User should handle transpose internally
		}
	}

	return est, iterations
}

// Estimate 1-norm of real matrix (f32)
m_estimate_1norm_f32 :: proc(
	n: int,
	apply_matrix: proc(X: []f32, Y: []f32),
	allocator := context.allocator,
) -> (
	estimate: f32,
	iterations: int,
) {
	n_blas := Blas_Int(n)

	// Allocate workspace
	V := make([]f32, n, context.temp_allocator)
	X := make([]f32, n, context.temp_allocator)
	ISGN := make([]Blas_Int, n, context.temp_allocator)

	est: f32
	kase: Blas_Int
	isave := make([]Blas_Int, 3, context.temp_allocator)

	iterations = 0

	// Iterative estimation loop
	for {
		lapack.slacn2_(
			&n_blas,
			raw_data(V),
			raw_data(X),
			raw_data(ISGN),
			&est,
			&kase,
			raw_data(isave),
		)

		if kase == 0 {
			break
		}

		iterations += 1

		if kase == 1 {
			// Apply matrix: Y := A * X
			apply_matrix(X, V)
		} else {
			// Apply transpose: Y := A^T * X
			apply_matrix(X, V) // User should handle transpose internally
		}
	}

	return est, iterations
}

// ===================================================================================
// LAPACK MATRIX COPY AND CONVERSION UTILITIES
// ===================================================================================

// Copy real matrix to complex matrix with zero imaginary parts
m_copy_real_to_complex :: proc {
	m_copy_real_to_complex_f32_c64,
	m_copy_real_to_complex_f64_c128,
}

// Copy real matrix to complex matrix (f32 -> complex64)
m_copy_real_to_complex_f32_c64 :: proc(
	A: ^Matrix(f32),
	B: ^Matrix(complex64),
	upper: bool = false, // Copy upper triangle only
	lower: bool = false, // Copy lower triangle only
) {
	// Validate matrices
	if A.format != .General {
		panic("A matrix must be in general format")
	}
	if B.format != .General {
		panic("B matrix must be in general format")
	}
	if A.rows != B.rows || A.cols != B.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Determine uplo parameter
	uplo := cstring("A") // All matrix
	if upper && !lower {
		uplo = cstring("U") // Upper triangle
	} else if lower && !upper {
		uplo = cstring("L") // Lower triangle
	}

	lapack.clacp2_(&uplo, &m, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, 1)
}

// Copy real matrix to complex matrix (f64 -> complex128)
m_copy_real_to_complex_f64_c128 :: proc(
	A: ^Matrix(f64),
	B: ^Matrix(complex128),
	upper: bool = false,
	lower: bool = false,
) {
	// Validate matrices
	if A.format != .General {
		panic("A matrix must be in general format")
	}
	if B.format != .General {
		panic("B matrix must be in general format")
	}
	if A.rows != B.rows || A.cols != B.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Determine uplo parameter
	uplo := cstring("A") // All matrix
	if upper && !lower {
		uplo = cstring("U") // Upper triangle
	} else if lower && !upper {
		uplo = cstring("L") // Lower triangle
	}

	lapack.zlacp2_(&uplo, &m, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, 1)
}

// Copy matrix with optional triangular selection
m_copy_matrix :: proc {
	m_copy_matrix_c64,
	m_copy_matrix_c128,
	m_copy_matrix_f32,
	m_copy_matrix_f64,
}

// Copy matrix (complex64)
m_copy_matrix_c64 :: proc(
	A: ^Matrix(complex64),
	B: ^Matrix(complex64),
	upper: bool = false, // Copy upper triangle only
	lower: bool = false, // Copy lower triangle only
) {
	// Validate matrices
	if A.format != .General {
		panic("A matrix must be in general format")
	}
	if B.format != .General {
		panic("B matrix must be in general format")
	}
	if A.rows != B.rows || A.cols != B.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Determine uplo parameter
	uplo := cstring("A") // All matrix
	if upper && !lower {
		uplo = cstring("U") // Upper triangle
	} else if lower && !upper {
		uplo = cstring("L") // Lower triangle
	}

	lapack.clacpy_(&uplo, &m, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, 1)
}

// Copy matrix (complex128)
m_copy_matrix_c128 :: proc(
	A: ^Matrix(complex128),
	B: ^Matrix(complex128),
	upper: bool = false,
	lower: bool = false,
) {
	// Validate matrices
	if A.format != .General {
		panic("A matrix must be in general format")
	}
	if B.format != .General {
		panic("B matrix must be in general format")
	}
	if A.rows != B.rows || A.cols != B.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Determine uplo parameter
	uplo := cstring("A") // All matrix
	if upper && !lower {
		uplo = cstring("U") // Upper triangle
	} else if lower && !upper {
		uplo = cstring("L") // Lower triangle
	}

	lapack.zlacpy_(&uplo, &m, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, 1)
}

// Copy matrix (f32)
m_copy_matrix_f32 :: proc(
	A: ^Matrix(f32),
	B: ^Matrix(f32),
	upper: bool = false,
	lower: bool = false,
) {
	// Validate matrices
	if A.format != .General {
		panic("A matrix must be in general format")
	}
	if B.format != .General {
		panic("B matrix must be in general format")
	}
	if A.rows != B.rows || A.cols != B.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Determine uplo parameter
	uplo := cstring("A") // All matrix
	if upper && !lower {
		uplo = cstring("U") // Upper triangle
	} else if lower && !upper {
		uplo = cstring("L") // Lower triangle
	}

	lapack.slacpy_(&uplo, &m, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, 1)
}

// Copy matrix (f64)
m_copy_matrix_f64 :: proc(
	A: ^Matrix(f64),
	B: ^Matrix(f64),
	upper: bool = false,
	lower: bool = false,
) {
	// Validate matrices
	if A.format != .General {
		panic("A matrix must be in general format")
	}
	if B.format != .General {
		panic("B matrix must be in general format")
	}
	if A.rows != B.rows || A.cols != B.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Determine uplo parameter
	uplo := cstring("A") // All matrix
	if upper && !lower {
		uplo = cstring("U") // Upper triangle
	} else if lower && !upper {
		uplo = cstring("L") // Lower triangle
	}

	lapack.dlacpy_(&uplo, &m, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, 1)
}

// Complex matrix-real matrix multiplication: C := A * B where A is complex, B is real
m_multiply_complex_real :: proc {
	m_multiply_complex_real_c64_f32,
	m_multiply_complex_real_c128_f64,
}

// Complex-real matrix multiplication (complex64 * f32)
m_multiply_complex_real_c64_f32 :: proc(
	A: ^Matrix(complex64),
	B: ^Matrix(f32),
	C: ^Matrix(complex64),
	allocator := context.allocator,
) {
	// Validate matrices
	if A.format != .General || B.format != .General || C.format != .General {
		panic("All matrices must be in general format")
	}
	if A.cols != B.rows {
		panic("Matrix dimensions incompatible for multiplication")
	}
	if A.rows != C.rows || B.cols != C.cols {
		panic("Output matrix dimensions incorrect")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(B.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)
	ldc := Blas_Int(C.ld)

	// Allocate workspace for real components
	rwork_size := max(1, int(m * n))
	rwork := make([]f32, rwork_size, context.temp_allocator)

	lapack.clacrm_(
		&m,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(C.data),
		&ldc,
		raw_data(rwork),
	)
}

// ===================================================================================
// LAPACK PRECISION CONVERSION UTILITIES
// ===================================================================================

// Convert between different numerical precisions with overflow checking
m_convert_precision :: proc {
	m_convert_precision_c128_c64,
	m_convert_precision_f32_f64,
	m_convert_precision_f64_f32,
	m_convert_precision_c64_c128,
}

// Convert complex128 matrix to complex64 with overflow checking
m_convert_precision_c128_c64 :: proc(
	A: ^Matrix(complex128),
	SA: ^Matrix(complex64),
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrices
	if A.format != .General || SA.format != .General {
		panic("Both matrices must be in general format")
	}
	if A.rows != SA.rows || A.cols != SA.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldsa := Blas_Int(SA.ld)

	info_val: Blas_Int
	lapack.zlag2c_(&m, &n, raw_data(A.data), &lda, raw_data(SA.data), &ldsa, &info_val)

	return info_val == 0, info_val
}

// Convert f32 matrix to f64 (always successful)
m_convert_precision_f32_f64 :: proc(
	SA: ^Matrix(f32),
	A: ^Matrix(f64),
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrices
	if SA.format != .General || A.format != .General {
		panic("Both matrices must be in general format")
	}
	if SA.rows != A.rows || SA.cols != A.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(SA.rows)
	n := Blas_Int(SA.cols)
	ldsa := Blas_Int(SA.ld)
	lda := Blas_Int(A.ld)

	info_val: Blas_Int
	lapack.slag2d_(&m, &n, raw_data(SA.data), &ldsa, raw_data(A.data), &lda, &info_val)

	return info_val == 0, info_val
}

// Convert f64 matrix to f32 with overflow checking
m_convert_precision_f64_f32 :: proc(
	A: ^Matrix(f64),
	SA: ^Matrix(f32),
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrices
	if A.format != .General || SA.format != .General {
		panic("Both matrices must be in general format")
	}
	if A.rows != SA.rows || A.cols != SA.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldsa := Blas_Int(SA.ld)

	info_val: Blas_Int
	lapack.dlag2s_(&m, &n, raw_data(A.data), &lda, raw_data(SA.data), &ldsa, &info_val)

	return info_val == 0, info_val
}

// Convert complex64 matrix to complex128 (always successful)
m_convert_precision_c64_c128 :: proc(
	SA: ^Matrix(complex64),
	A: ^Matrix(complex128),
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrices
	if SA.format != .General || A.format != .General {
		panic("Both matrices must be in general format")
	}
	if SA.rows != A.rows || SA.cols != A.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(SA.rows)
	n := Blas_Int(SA.cols)
	ldsa := Blas_Int(SA.ld)
	lda := Blas_Int(A.ld)

	info_val: Blas_Int
	lapack.clag2z_(&m, &n, raw_data(SA.data), &ldsa, raw_data(A.data), &lda, &info_val)

	return info_val == 0, info_val
}

// ===================================================================================
// LAPACK RANDOM MATRIX GENERATION UTILITIES
// ===================================================================================

// Generate random general banded matrix
m_generate_random_banded :: proc {
	m_generate_random_banded_c64,
	m_generate_random_banded_c128,
	m_generate_random_banded_f32,
	m_generate_random_banded_f64,
}

// Generate random banded matrix (complex64)
m_generate_random_banded_c64 :: proc(
	A: ^Matrix(complex64),
	kl, ku: int, // Lower and upper bandwidth
	D: []f32, // Diagonal scaling factors
	seed: []int = nil, // Random seed (4 elements), auto-generated if nil
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrix
	if A.format != .General {
		panic("Matrix must be in general format")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix rows")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	kl_val := Blas_Int(kl)
	ku_val := Blas_Int(ku)
	lda := Blas_Int(A.ld)

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		// Generate default seed
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work_size := max(m, n)
	work := make([]complex64, work_size, context.temp_allocator)

	info_val: Blas_Int
	lapack.clagge_(
		&m,
		&n,
		&kl_val,
		&ku_val,
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}

// Generate random banded matrix (complex128)
m_generate_random_banded_c128 :: proc(
	A: ^Matrix(complex128),
	kl, ku: int,
	D: []f64,
	seed: []int = nil,
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrix
	if A.format != .General {
		panic("Matrix must be in general format")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix rows")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	kl_val := Blas_Int(kl)
	ku_val := Blas_Int(ku)
	lda := Blas_Int(A.ld)

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work_size := max(m, n)
	work := make([]complex128, work_size, context.temp_allocator)

	info_val: Blas_Int
	lapack.zlagge_(
		&m,
		&n,
		&kl_val,
		&ku_val,
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}

// Generate random banded matrix (f32)
m_generate_random_banded_f32 :: proc(
	A: ^Matrix(f32),
	kl, ku: int,
	D: []f32,
	seed: []int = nil,
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrix
	if A.format != .General {
		panic("Matrix must be in general format")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix rows")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	kl_val := Blas_Int(kl)
	ku_val := Blas_Int(ku)
	lda := Blas_Int(A.ld)

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work_size := max(m, n)
	work := make([]f32, work_size, context.temp_allocator)

	info_val: Blas_Int
	lapack.slagge_(
		&m,
		&n,
		&kl_val,
		&ku_val,
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}

// Generate random banded matrix (f64)
m_generate_random_banded_f64 :: proc(
	A: ^Matrix(f64),
	kl, ku: int,
	D: []f64,
	seed: []int = nil,
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrix
	if A.format != .General {
		panic("Matrix must be in general format")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix rows")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	kl_val := Blas_Int(kl)
	ku_val := Blas_Int(ku)
	lda := Blas_Int(A.ld)

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work_size := max(m, n)
	work := make([]f64, work_size, context.temp_allocator)

	info_val: Blas_Int
	lapack.dlagge_(
		&m,
		&n,
		&kl_val,
		&ku_val,
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}

// Complex-real matrix multiplication (complex128 * f64)
m_multiply_complex_real_c128_f64 :: proc(
	A: ^Matrix(complex128),
	B: ^Matrix(f64),
	C: ^Matrix(complex128),
	allocator := context.allocator,
) {
	// Validate matrices
	if A.format != .General || B.format != .General || C.format != .General {
		panic("All matrices must be in general format")
	}
	if A.cols != B.rows {
		panic("Matrix dimensions incompatible for multiplication")
	}
	if A.rows != C.rows || B.cols != C.cols {
		panic("Output matrix dimensions incorrect")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(B.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)
	ldc := Blas_Int(C.ld)

	// Allocate workspace for real components
	rwork_size := max(1, int(m * n))
	rwork := make([]f64, rwork_size, context.temp_allocator)

	lapack.zlacrm_(
		&m,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(C.data),
		&ldc,
		raw_data(rwork),
	)
}

// ===================================================================================
// LAPACK PRECISION CONVERSION UTILITIES
// ===================================================================================

// Convert between different numerical precisions with overflow checking
m_convert_precision :: proc {
	m_convert_precision_c128_c64,
	m_convert_precision_f32_f64,
	m_convert_precision_f64_f32,
	m_convert_precision_c64_c128,
}

// Convert complex128 matrix to complex64 with overflow checking
m_convert_precision_c128_c64 :: proc(
	A: ^Matrix(complex128),
	SA: ^Matrix(complex64),
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrices
	if A.format != .General || SA.format != .General {
		panic("Both matrices must be in general format")
	}
	if A.rows != SA.rows || A.cols != SA.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldsa := Blas_Int(SA.ld)

	info_val: Blas_Int
	lapack.zlag2c_(&m, &n, raw_data(A.data), &lda, raw_data(SA.data), &ldsa, &info_val)

	return info_val == 0, info_val
}

// Convert f32 matrix to f64 (always successful)
m_convert_precision_f32_f64 :: proc(
	SA: ^Matrix(f32),
	A: ^Matrix(f64),
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrices
	if SA.format != .General || A.format != .General {
		panic("Both matrices must be in general format")
	}
	if SA.rows != A.rows || SA.cols != A.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(SA.rows)
	n := Blas_Int(SA.cols)
	ldsa := Blas_Int(SA.ld)
	lda := Blas_Int(A.ld)

	info_val: Blas_Int
	lapack.slag2d_(&m, &n, raw_data(SA.data), &ldsa, raw_data(A.data), &lda, &info_val)

	return info_val == 0, info_val
}

// Convert f64 matrix to f32 with overflow checking
m_convert_precision_f64_f32 :: proc(
	A: ^Matrix(f64),
	SA: ^Matrix(f32),
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrices
	if A.format != .General || SA.format != .General {
		panic("Both matrices must be in general format")
	}
	if A.rows != SA.rows || A.cols != SA.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldsa := Blas_Int(SA.ld)

	info_val: Blas_Int
	lapack.dlag2s_(&m, &n, raw_data(A.data), &lda, raw_data(SA.data), &ldsa, &info_val)

	return info_val == 0, info_val
}

// Convert complex64 matrix to complex128 (always successful)
m_convert_precision_c64_c128 :: proc(
	SA: ^Matrix(complex64),
	A: ^Matrix(complex128),
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrices
	if SA.format != .General || A.format != .General {
		panic("Both matrices must be in general format")
	}
	if SA.rows != A.rows || SA.cols != A.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(SA.rows)
	n := Blas_Int(SA.cols)
	ldsa := Blas_Int(SA.ld)
	lda := Blas_Int(A.ld)

	info_val: Blas_Int
	lapack.clag2z_(&m, &n, raw_data(SA.data), &ldsa, raw_data(A.data), &lda, &info_val)

	return info_val == 0, info_val
}

// ===================================================================================
// LAPACK RANDOM MATRIX GENERATION UTILITIES
// ===================================================================================

// Generate random general banded matrix
m_generate_random_banded :: proc {
	m_generate_random_banded_c64,
	m_generate_random_banded_c128,
	m_generate_random_banded_f32,
	m_generate_random_banded_f64,
}

// Generate random banded matrix (complex64)
m_generate_random_banded_c64 :: proc(
	A: ^Matrix(complex64),
	kl, ku: int, // Lower and upper bandwidth
	D: []f32, // Diagonal scaling factors
	seed: []int = nil, // Random seed (4 elements), auto-generated if nil
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrix
	if A.format != .General {
		panic("Matrix must be in general format")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix rows")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	kl_val := Blas_Int(kl)
	ku_val := Blas_Int(ku)
	lda := Blas_Int(A.ld)

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		// Generate default seed
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work_size := max(m, n)
	work := make([]complex64, work_size, context.temp_allocator)

	info_val: Blas_Int
	lapack.clagge_(
		&m,
		&n,
		&kl_val,
		&ku_val,
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}

// Generate random banded matrix (complex128)
m_generate_random_banded_c128 :: proc(
	A: ^Matrix(complex128),
	kl, ku: int,
	D: []f64,
	seed: []int = nil,
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrix
	if A.format != .General {
		panic("Matrix must be in general format")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix rows")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	kl_val := Blas_Int(kl)
	ku_val := Blas_Int(ku)
	lda := Blas_Int(A.ld)

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work_size := max(m, n)
	work := make([]complex128, work_size, context.temp_allocator)

	info_val: Blas_Int
	lapack.zlagge_(
		&m,
		&n,
		&kl_val,
		&ku_val,
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}

// Generate random banded matrix (f32)
m_generate_random_banded_f32 :: proc(
	A: ^Matrix(f32),
	kl, ku: int,
	D: []f32,
	seed: []int = nil,
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrix
	if A.format != .General {
		panic("Matrix must be in general format")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix rows")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	kl_val := Blas_Int(kl)
	ku_val := Blas_Int(ku)
	lda := Blas_Int(A.ld)

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work_size := max(m, n)
	work := make([]f32, work_size, context.temp_allocator)

	info_val: Blas_Int
	lapack.slagge_(
		&m,
		&n,
		&kl_val,
		&ku_val,
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}

// Generate random banded matrix (f64)
m_generate_random_banded_f64 :: proc(
	A: ^Matrix(f64),
	kl, ku: int,
	D: []f64,
	seed: []int = nil,
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrix
	if A.format != .General {
		panic("Matrix must be in general format")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix rows")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	kl_val := Blas_Int(kl)
	ku_val := Blas_Int(ku)
	lda := Blas_Int(A.ld)

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work_size := max(m, n)
	work := make([]f64, work_size, context.temp_allocator)

	info_val: Blas_Int
	lapack.dlagge_(
		&m,
		&n,
		&kl_val,
		&ku_val,
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}

// Estimate 1-norm of real matrix (f64)
m_estimate_1norm_f64 :: proc(
	n: int,
	apply_matrix: proc(X: []f64, Y: []f64),
	allocator := context.allocator,
) -> (
	estimate: f64,
	iterations: int,
) {
	n_blas := Blas_Int(n)

	// Allocate workspace
	V := make([]f64, n, context.temp_allocator)
	X := make([]f64, n, context.temp_allocator)
	ISGN := make([]Blas_Int, n, context.temp_allocator)

	est: f64
	kase: Blas_Int
	isave := make([]Blas_Int, 3, context.temp_allocator)

	iterations = 0

	// Iterative estimation loop
	for {
		lapack.dlacn2_(
			&n_blas,
			raw_data(V),
			raw_data(X),
			raw_data(ISGN),
			&est,
			&kase,
			raw_data(isave),
		)

		if kase == 0 {
			break
		}

		iterations += 1

		if kase == 1 {
			// Apply matrix: Y := A * X
			apply_matrix(X, V)
		} else {
			// Apply transpose: Y := A^T * X
			apply_matrix(X, V) // User should handle transpose internally
		}
	}

	return est, iterations
}

// ===================================================================================
// LAPACK MATRIX COPY AND CONVERSION UTILITIES
// ===================================================================================

// Copy real matrix to complex matrix with zero imaginary parts
m_copy_real_to_complex :: proc {
	m_copy_real_to_complex_f32_c64,
	m_copy_real_to_complex_f64_c128,
}

// Copy real matrix to complex matrix (f32 -> complex64)
m_copy_real_to_complex_f32_c64 :: proc(
	A: ^Matrix(f32),
	B: ^Matrix(complex64),
	upper: bool = false, // Copy upper triangle only
	lower: bool = false, // Copy lower triangle only
) {
	// Validate matrices
	if A.format != .General {
		panic("A matrix must be in general format")
	}
	if B.format != .General {
		panic("B matrix must be in general format")
	}
	if A.rows != B.rows || A.cols != B.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Determine uplo parameter
	uplo := cstring("A") // All matrix
	if upper && !lower {
		uplo = cstring("U") // Upper triangle
	} else if lower && !upper {
		uplo = cstring("L") // Lower triangle
	}

	lapack.clacp2_(&uplo, &m, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, 1)
}

// Copy real matrix to complex matrix (f64 -> complex128)
m_copy_real_to_complex_f64_c128 :: proc(
	A: ^Matrix(f64),
	B: ^Matrix(complex128),
	upper: bool = false,
	lower: bool = false,
) {
	// Validate matrices
	if A.format != .General {
		panic("A matrix must be in general format")
	}
	if B.format != .General {
		panic("B matrix must be in general format")
	}
	if A.rows != B.rows || A.cols != B.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Determine uplo parameter
	uplo := cstring("A") // All matrix
	if upper && !lower {
		uplo = cstring("U") // Upper triangle
	} else if lower && !upper {
		uplo = cstring("L") // Lower triangle
	}

	lapack.zlacp2_(&uplo, &m, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, 1)
}

// Copy matrix with optional triangular selection
m_copy_matrix :: proc {
	m_copy_matrix_c64,
	m_copy_matrix_c128,
	m_copy_matrix_f32,
	m_copy_matrix_f64,
}

// Copy matrix (complex64)
m_copy_matrix_c64 :: proc(
	A: ^Matrix(complex64),
	B: ^Matrix(complex64),
	upper: bool = false, // Copy upper triangle only
	lower: bool = false, // Copy lower triangle only
) {
	// Validate matrices
	if A.format != .General {
		panic("A matrix must be in general format")
	}
	if B.format != .General {
		panic("B matrix must be in general format")
	}
	if A.rows != B.rows || A.cols != B.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Determine uplo parameter
	uplo := cstring("A") // All matrix
	if upper && !lower {
		uplo = cstring("U") // Upper triangle
	} else if lower && !upper {
		uplo = cstring("L") // Lower triangle
	}

	lapack.clacpy_(&uplo, &m, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, 1)
}

// Copy matrix (complex128)
m_copy_matrix_c128 :: proc(
	A: ^Matrix(complex128),
	B: ^Matrix(complex128),
	upper: bool = false,
	lower: bool = false,
) {
	// Validate matrices
	if A.format != .General {
		panic("A matrix must be in general format")
	}
	if B.format != .General {
		panic("B matrix must be in general format")
	}
	if A.rows != B.rows || A.cols != B.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Determine uplo parameter
	uplo := cstring("A") // All matrix
	if upper && !lower {
		uplo = cstring("U") // Upper triangle
	} else if lower && !upper {
		uplo = cstring("L") // Lower triangle
	}

	lapack.zlacpy_(&uplo, &m, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, 1)
}

// Copy matrix (f32)
m_copy_matrix_f32 :: proc(
	A: ^Matrix(f32),
	B: ^Matrix(f32),
	upper: bool = false,
	lower: bool = false,
) {
	// Validate matrices
	if A.format != .General {
		panic("A matrix must be in general format")
	}
	if B.format != .General {
		panic("B matrix must be in general format")
	}
	if A.rows != B.rows || A.cols != B.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Determine uplo parameter
	uplo := cstring("A") // All matrix
	if upper && !lower {
		uplo = cstring("U") // Upper triangle
	} else if lower && !upper {
		uplo = cstring("L") // Lower triangle
	}

	lapack.slacpy_(&uplo, &m, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, 1)
}

// Copy matrix (f64)
m_copy_matrix_f64 :: proc(
	A: ^Matrix(f64),
	B: ^Matrix(f64),
	upper: bool = false,
	lower: bool = false,
) {
	// Validate matrices
	if A.format != .General {
		panic("A matrix must be in general format")
	}
	if B.format != .General {
		panic("B matrix must be in general format")
	}
	if A.rows != B.rows || A.cols != B.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Determine uplo parameter
	uplo := cstring("A") // All matrix
	if upper && !lower {
		uplo = cstring("U") // Upper triangle
	} else if lower && !upper {
		uplo = cstring("L") // Lower triangle
	}

	lapack.dlacpy_(&uplo, &m, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, 1)
}

// Complex matrix-real matrix multiplication: C := A * B where A is complex, B is real
m_multiply_complex_real :: proc {
	m_multiply_complex_real_c64_f32,
	m_multiply_complex_real_c128_f64,
}

// Complex-real matrix multiplication (complex64 * f32)
m_multiply_complex_real_c64_f32 :: proc(
	A: ^Matrix(complex64),
	B: ^Matrix(f32),
	C: ^Matrix(complex64),
	allocator := context.allocator,
) {
	// Validate matrices
	if A.format != .General || B.format != .General || C.format != .General {
		panic("All matrices must be in general format")
	}
	if A.cols != B.rows {
		panic("Matrix dimensions incompatible for multiplication")
	}
	if A.rows != C.rows || B.cols != C.cols {
		panic("Output matrix dimensions incorrect")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(B.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)
	ldc := Blas_Int(C.ld)

	// Allocate workspace for real components
	rwork_size := max(1, int(m * n))
	rwork := make([]f32, rwork_size, context.temp_allocator)

	lapack.clacrm_(
		&m,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(C.data),
		&ldc,
		raw_data(rwork),
	)
}

// ===================================================================================
// LAPACK PRECISION CONVERSION UTILITIES
// ===================================================================================

// Convert between different numerical precisions with overflow checking
m_convert_precision :: proc {
	m_convert_precision_c128_c64,
	m_convert_precision_f32_f64,
	m_convert_precision_f64_f32,
	m_convert_precision_c64_c128,
}

// Convert complex128 matrix to complex64 with overflow checking
m_convert_precision_c128_c64 :: proc(
	A: ^Matrix(complex128),
	SA: ^Matrix(complex64),
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrices
	if A.format != .General || SA.format != .General {
		panic("Both matrices must be in general format")
	}
	if A.rows != SA.rows || A.cols != SA.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldsa := Blas_Int(SA.ld)

	info_val: Blas_Int
	lapack.zlag2c_(&m, &n, raw_data(A.data), &lda, raw_data(SA.data), &ldsa, &info_val)

	return info_val == 0, info_val
}

// Convert f32 matrix to f64 (always successful)
m_convert_precision_f32_f64 :: proc(
	SA: ^Matrix(f32),
	A: ^Matrix(f64),
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrices
	if SA.format != .General || A.format != .General {
		panic("Both matrices must be in general format")
	}
	if SA.rows != A.rows || SA.cols != A.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(SA.rows)
	n := Blas_Int(SA.cols)
	ldsa := Blas_Int(SA.ld)
	lda := Blas_Int(A.ld)

	info_val: Blas_Int
	lapack.slag2d_(&m, &n, raw_data(SA.data), &ldsa, raw_data(A.data), &lda, &info_val)

	return info_val == 0, info_val
}

// Convert f64 matrix to f32 with overflow checking
m_convert_precision_f64_f32 :: proc(
	A: ^Matrix(f64),
	SA: ^Matrix(f32),
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrices
	if A.format != .General || SA.format != .General {
		panic("Both matrices must be in general format")
	}
	if A.rows != SA.rows || A.cols != SA.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldsa := Blas_Int(SA.ld)

	info_val: Blas_Int
	lapack.dlag2s_(&m, &n, raw_data(A.data), &lda, raw_data(SA.data), &ldsa, &info_val)

	return info_val == 0, info_val
}

// Convert complex64 matrix to complex128 (always successful)
m_convert_precision_c64_c128 :: proc(
	SA: ^Matrix(complex64),
	A: ^Matrix(complex128),
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrices
	if SA.format != .General || A.format != .General {
		panic("Both matrices must be in general format")
	}
	if SA.rows != A.rows || SA.cols != A.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(SA.rows)
	n := Blas_Int(SA.cols)
	ldsa := Blas_Int(SA.ld)
	lda := Blas_Int(A.ld)

	info_val: Blas_Int
	lapack.clag2z_(&m, &n, raw_data(SA.data), &ldsa, raw_data(A.data), &lda, &info_val)

	return info_val == 0, info_val
}

// ===================================================================================
// LAPACK RANDOM MATRIX GENERATION UTILITIES
// ===================================================================================

// Generate random general banded matrix
m_generate_random_banded :: proc {
	m_generate_random_banded_c64,
	m_generate_random_banded_c128,
	m_generate_random_banded_f32,
	m_generate_random_banded_f64,
}

// Generate random banded matrix (complex64)
m_generate_random_banded_c64 :: proc(
	A: ^Matrix(complex64),
	kl, ku: int, // Lower and upper bandwidth
	D: []f32, // Diagonal scaling factors
	seed: []int = nil, // Random seed (4 elements), auto-generated if nil
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrix
	if A.format != .General {
		panic("Matrix must be in general format")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix rows")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	kl_val := Blas_Int(kl)
	ku_val := Blas_Int(ku)
	lda := Blas_Int(A.ld)

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		// Generate default seed
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work_size := max(m, n)
	work := make([]complex64, work_size, context.temp_allocator)

	info_val: Blas_Int
	lapack.clagge_(
		&m,
		&n,
		&kl_val,
		&ku_val,
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}

// Generate random banded matrix (complex128)
m_generate_random_banded_c128 :: proc(
	A: ^Matrix(complex128),
	kl, ku: int,
	D: []f64,
	seed: []int = nil,
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrix
	if A.format != .General {
		panic("Matrix must be in general format")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix rows")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	kl_val := Blas_Int(kl)
	ku_val := Blas_Int(ku)
	lda := Blas_Int(A.ld)

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work_size := max(m, n)
	work := make([]complex128, work_size, context.temp_allocator)

	info_val: Blas_Int
	lapack.zlagge_(
		&m,
		&n,
		&kl_val,
		&ku_val,
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}

// Generate random banded matrix (f32)
m_generate_random_banded_f32 :: proc(
	A: ^Matrix(f32),
	kl, ku: int,
	D: []f32,
	seed: []int = nil,
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrix
	if A.format != .General {
		panic("Matrix must be in general format")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix rows")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	kl_val := Blas_Int(kl)
	ku_val := Blas_Int(ku)
	lda := Blas_Int(A.ld)

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work_size := max(m, n)
	work := make([]f32, work_size, context.temp_allocator)

	info_val: Blas_Int
	lapack.slagge_(
		&m,
		&n,
		&kl_val,
		&ku_val,
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}

// Generate random banded matrix (f64)
m_generate_random_banded_f64 :: proc(
	A: ^Matrix(f64),
	kl, ku: int,
	D: []f64,
	seed: []int = nil,
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrix
	if A.format != .General {
		panic("Matrix must be in general format")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix rows")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	kl_val := Blas_Int(kl)
	ku_val := Blas_Int(ku)
	lda := Blas_Int(A.ld)

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work_size := max(m, n)
	work := make([]f64, work_size, context.temp_allocator)

	info_val: Blas_Int
	lapack.dlagge_(
		&m,
		&n,
		&kl_val,
		&ku_val,
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}

// Complex-real matrix multiplication (complex128 * f64)
m_multiply_complex_real_c128_f64 :: proc(
	A: ^Matrix(complex128),
	B: ^Matrix(f64),
	C: ^Matrix(complex128),
	allocator := context.allocator,
) {
	// Validate matrices
	if A.format != .General || B.format != .General || C.format != .General {
		panic("All matrices must be in general format")
	}
	if A.cols != B.rows {
		panic("Matrix dimensions incompatible for multiplication")
	}
	if A.rows != C.rows || B.cols != C.cols {
		panic("Output matrix dimensions incorrect")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(B.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)
	ldc := Blas_Int(C.ld)

	// Allocate workspace for real components
	rwork_size := max(1, int(m * n))
	rwork := make([]f64, rwork_size, context.temp_allocator)

	lapack.zlacrm_(
		&m,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(C.data),
		&ldc,
		raw_data(rwork),
	)
}

// ===================================================================================
// LAPACK PRECISION CONVERSION UTILITIES
// ===================================================================================

// Convert between different numerical precisions with overflow checking
m_convert_precision :: proc {
	m_convert_precision_c128_c64,
	m_convert_precision_f32_f64,
	m_convert_precision_f64_f32,
	m_convert_precision_c64_c128,
}

// Convert complex128 matrix to complex64 with overflow checking
m_convert_precision_c128_c64 :: proc(
	A: ^Matrix(complex128),
	SA: ^Matrix(complex64),
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrices
	if A.format != .General || SA.format != .General {
		panic("Both matrices must be in general format")
	}
	if A.rows != SA.rows || A.cols != SA.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldsa := Blas_Int(SA.ld)

	info_val: Blas_Int
	lapack.zlag2c_(&m, &n, raw_data(A.data), &lda, raw_data(SA.data), &ldsa, &info_val)

	return info_val == 0, info_val
}

// Convert f32 matrix to f64 (always successful)
m_convert_precision_f32_f64 :: proc(
	SA: ^Matrix(f32),
	A: ^Matrix(f64),
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrices
	if SA.format != .General || A.format != .General {
		panic("Both matrices must be in general format")
	}
	if SA.rows != A.rows || SA.cols != A.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(SA.rows)
	n := Blas_Int(SA.cols)
	ldsa := Blas_Int(SA.ld)
	lda := Blas_Int(A.ld)

	info_val: Blas_Int
	lapack.slag2d_(&m, &n, raw_data(SA.data), &ldsa, raw_data(A.data), &lda, &info_val)

	return info_val == 0, info_val
}

// Convert f64 matrix to f32 with overflow checking
m_convert_precision_f64_f32 :: proc(
	A: ^Matrix(f64),
	SA: ^Matrix(f32),
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrices
	if A.format != .General || SA.format != .General {
		panic("Both matrices must be in general format")
	}
	if A.rows != SA.rows || A.cols != SA.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldsa := Blas_Int(SA.ld)

	info_val: Blas_Int
	lapack.dlag2s_(&m, &n, raw_data(A.data), &lda, raw_data(SA.data), &ldsa, &info_val)

	return info_val == 0, info_val
}

// Convert complex64 matrix to complex128 (always successful)
m_convert_precision_c64_c128 :: proc(
	SA: ^Matrix(complex64),
	A: ^Matrix(complex128),
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrices
	if SA.format != .General || A.format != .General {
		panic("Both matrices must be in general format")
	}
	if SA.rows != A.rows || SA.cols != A.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(SA.rows)
	n := Blas_Int(SA.cols)
	ldsa := Blas_Int(SA.ld)
	lda := Blas_Int(A.ld)

	info_val: Blas_Int
	lapack.clag2z_(&m, &n, raw_data(SA.data), &ldsa, raw_data(A.data), &lda, &info_val)

	return info_val == 0, info_val
}

// ===================================================================================
// LAPACK RANDOM MATRIX GENERATION UTILITIES
// ===================================================================================

// Generate random general banded matrix
m_generate_random_banded :: proc {
	m_generate_random_banded_c64,
	m_generate_random_banded_c128,
	m_generate_random_banded_f32,
	m_generate_random_banded_f64,
}

// Generate random banded matrix (complex64)
m_generate_random_banded_c64 :: proc(
	A: ^Matrix(complex64),
	kl, ku: int, // Lower and upper bandwidth
	D: []f32, // Diagonal scaling factors
	seed: []int = nil, // Random seed (4 elements), auto-generated if nil
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrix
	if A.format != .General {
		panic("Matrix must be in general format")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix rows")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	kl_val := Blas_Int(kl)
	ku_val := Blas_Int(ku)
	lda := Blas_Int(A.ld)

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		// Generate default seed
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work_size := max(m, n)
	work := make([]complex64, work_size, context.temp_allocator)

	info_val: Blas_Int
	lapack.clagge_(
		&m,
		&n,
		&kl_val,
		&ku_val,
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}

// Generate random banded matrix (complex128)
m_generate_random_banded_c128 :: proc(
	A: ^Matrix(complex128),
	kl, ku: int,
	D: []f64,
	seed: []int = nil,
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrix
	if A.format != .General {
		panic("Matrix must be in general format")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix rows")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	kl_val := Blas_Int(kl)
	ku_val := Blas_Int(ku)
	lda := Blas_Int(A.ld)

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work_size := max(m, n)
	work := make([]complex128, work_size, context.temp_allocator)

	info_val: Blas_Int
	lapack.zlagge_(
		&m,
		&n,
		&kl_val,
		&ku_val,
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}

// Generate random banded matrix (f32)
m_generate_random_banded_f32 :: proc(
	A: ^Matrix(f32),
	kl, ku: int,
	D: []f32,
	seed: []int = nil,
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrix
	if A.format != .General {
		panic("Matrix must be in general format")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix rows")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	kl_val := Blas_Int(kl)
	ku_val := Blas_Int(ku)
	lda := Blas_Int(A.ld)

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work_size := max(m, n)
	work := make([]f32, work_size, context.temp_allocator)

	info_val: Blas_Int
	lapack.slagge_(
		&m,
		&n,
		&kl_val,
		&ku_val,
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}

// Generate random banded matrix (f64)
m_generate_random_banded_f64 :: proc(
	A: ^Matrix(f64),
	kl, ku: int,
	D: []f64,
	seed: []int = nil,
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrix
	if A.format != .General {
		panic("Matrix must be in general format")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix rows")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	kl_val := Blas_Int(kl)
	ku_val := Blas_Int(ku)
	lda := Blas_Int(A.ld)

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work_size := max(m, n)
	work := make([]f64, work_size, context.temp_allocator)

	info_val: Blas_Int
	lapack.dlagge_(
		&m,
		&n,
		&kl_val,
		&ku_val,
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}

// Compute eigenvectors of Hessenberg matrix (complex128)
m_eigenvectors_hessenberg_c128 :: proc(
	H: ^Matrix(complex128),
	eigenvalues: []complex128,
	compute_left: bool = false,
	compute_right: bool = true,
	eigsrc: bool = false,
	select_mask: []bool = nil,
	allocator := context.allocator,
) -> (
	VL, VR: Matrix(complex128),
	m_computed: int,
	ifail: []int,
	info: Info,
) {
	// Validate matrix
	if H.format != .General {
		panic("H matrix must be in general format")
	}

	n := Blas_Int(H.rows)
	if H.rows != H.cols {
		panic("H matrix must be square")
	}

	// Setup computation sides
	side_c :=
		cstring("B") if compute_left && compute_right else cstring("L") if compute_left else cstring("R")

	eigsrc_c := cstring("Q") if eigsrc else cstring("N")
	initv_c := cstring("N")

	// Setup selection
	var; select_raw: []Blas_Int
	if select_mask != nil && len(select_mask) == int(n) {
		select_raw = make([]Blas_Int, n, context.temp_allocator)
		for i, selected in select_mask {
			select_raw[i] = 1 if selected else 0
		}
	} else {
		select_raw = make([]Blas_Int, n, context.temp_allocator)
		for i in 0 ..< n {
			select_raw[i] = 1
		}
	}

	ldh := Blas_Int(H.ld)
	ldvl := n if compute_left else 1
	ldvr := n if compute_right else 1

	// Allocate output matrices
	VL =
		matrix_zeros(complex128, int(ldvl), int(n), allocator) if compute_left else Matrix(complex128){}
	VR =
		matrix_zeros(complex128, int(ldvr), int(n), allocator) if compute_right else Matrix(complex128){}

	mm := n
	m_out: Blas_Int

	// Copy eigenvalues
	W := make([]complex128, n, context.temp_allocator)
	copy(W, eigenvalues)

	// Allocate workspace
	work := make([]complex128, n * n, context.temp_allocator)
	rwork := make([]f64, n, context.temp_allocator)

	ifaill := make([]Blas_Int, n, context.temp_allocator)
	ifailr := make([]Blas_Int, n, context.temp_allocator)

	// Call LAPACK
	info_val: Blas_Int
	vl_ptr := raw_data(VL.data) if compute_left else nil
	vr_ptr := raw_data(VR.data) if compute_right else nil

	lapack.zhsein_(
		&side_c,
		&eigsrc_c,
		&initv_c,
		raw_data(select_raw),
		&n,
		raw_data(H.data),
		&ldh,
		raw_data(W),
		vl_ptr,
		&ldvl,
		vr_ptr,
		&ldvr,
		&mm,
		&m_out,
		raw_data(work),
		raw_data(rwork),
		raw_data(ifaill),
		raw_data(ifailr),
		&info_val,
		1,
		1,
		1,
	)

	// Convert failure indices
	ifail = make([]int, n, allocator)
	for i in 0 ..< n {
		if compute_left && ifaill[i] != 0 {
			ifail[i] = int(ifaill[i])
		} else if compute_right && ifailr[i] != 0 {
			ifail[i] = int(ifailr[i])
		}
	}

	return VL, VR, int(m_out), ifail, info_val
}

// ===================================================================================
// HESSENBERG QR ALGORITHM FOR EIGENVALUE COMPUTATION
// ===================================================================================

// Compute eigenvalues and optionally Schur form of Hessenberg matrix using QR algorithm
m_eigenvalues_hessenberg_qr :: proc {
	m_eigenvalues_hessenberg_qr_c64,
	m_eigenvalues_hessenberg_qr_c128,
	m_eigenvalues_hessenberg_qr_f32,
	m_eigenvalues_hessenberg_qr_f64,
}

// QR algorithm for Hessenberg matrix eigenvalues (complex64)
m_eigenvalues_hessenberg_qr_c64 :: proc(
	H: ^Matrix(complex64),
	ilo: int = 0, // Submatrix range (0 means use full matrix)
	ihi: int = 0, // Submatrix range (0 means use full matrix)
	compute_schur: bool = true,
	compute_eigenvectors: bool = false,
	Z: ^Matrix(complex64) = nil, // Optional input/output matrix for eigenvectors
	allocator := context.allocator,
) -> (
	eigenvalues: []complex64,
	Z_out: Matrix(complex64),
	success: bool,
	info: Info,
) {
	// Validate matrix
	if H.format != .General {
		panic("H matrix must be in general format")
	}

	n := Blas_Int(H.rows)
	if H.rows != H.cols {
		panic("H matrix must be square")
	}

	// Setup range
	ilo_val := Blas_Int(ilo if ilo > 0 else 1)
	ihi_val := Blas_Int(ihi if ihi > 0 else int(n))

	// Setup job parameters
	job_c := cstring("S") if compute_schur else cstring("E") // Schur form or eigenvalues only
	compz_c := cstring("V") if compute_eigenvectors else cstring("N") // Compute or don't compute Z

	ldh := Blas_Int(H.ld)

	// Setup Z matrix
	ldz: Blas_Int
	if compute_eigenvectors {
		if Z != nil {
			if Z.rows != H.rows || Z.cols != H.cols {
				panic("Z matrix dimensions must match H matrix")
			}
			Z_out = Z^
			ldz = Blas_Int(Z.ld)
		} else {
			Z_out = matrix_zeros(complex64, int(n), int(n), allocator)
			ldz = n
		}
	} else {
		Z_out = Matrix(complex64){}
		ldz = 1
	}

	// Allocate eigenvalue array
	W := make([]complex64, n, allocator)

	// Workspace query
	work_query: complex64
	lwork_query := Blas_Int(-1)
	info_val: Blas_Int

	z_ptr := raw_data(Z_out.data) if compute_eigenvectors else nil

	lapack.chseqr_(
		&job_c,
		&compz_c,
		&n,
		&ilo_val,
		&ihi_val,
		raw_data(H.data),
		&ldh,
		raw_data(W),
		z_ptr,
		&ldz,
		&work_query,
		&lwork_query,
		&info_val,
		1,
		1,
	)

	if info_val != 0 {
		return nil, Z_out, false, info_val
	}

	// Allocate workspace
	lwork := Blas_Int(real(work_query))
	work := make([]complex64, lwork, context.temp_allocator)

	// Compute eigenvalues
	lapack.chseqr_(
		&job_c,
		&compz_c,
		&n,
		&ilo_val,
		&ihi_val,
		raw_data(H.data),
		&ldh,
		raw_data(W),
		z_ptr,
		&ldz,
		raw_data(work),
		&lwork,
		&info_val,
		1,
		1,
	)

	eigenvalues = W
	return eigenvalues, Z_out, info_val == 0, info_val
}

// ===================================================================================
// LAPACK UTILITY FUNCTIONS
// ===================================================================================

// Complex conjugate of a vector
m_conjugate_vector :: proc {
	m_conjugate_vector_c64,
	m_conjugate_vector_c128,
}

// Conjugate complex vector (complex64)
m_conjugate_vector_c64 :: proc(X: ^Vector(complex64)) {
	n := Blas_Int(len(X.data))
	incx := Blas_Int(X.incr)

	lapack.clacgv_(&n, raw_data(X.data), &incx)
}

// Conjugate complex vector (complex128)
m_conjugate_vector_c128 :: proc(X: ^Vector(complex128)) {
	n := Blas_Int(len(X.data))
	incx := Blas_Int(X.incr)

	lapack.zlacgv_(&n, raw_data(X.data), &incx)
}

// Estimate the 1-norm of a matrix using iterative method
m_estimate_1norm :: proc {
	m_estimate_1norm_c64,
	m_estimate_1norm_c128,
	m_estimate_1norm_f32,
	m_estimate_1norm_f64,
}

// Estimate 1-norm of matrix (complex64)
m_estimate_1norm_c64 :: proc(
	n: int,
	apply_matrix: proc(X: []complex64, Y: []complex64), // User-provided matrix application
	allocator := context.allocator,
) -> (
	estimate: f32,
	iterations: int,
) {
	n_blas := Blas_Int(n)

	// Allocate workspace
	V := make([]complex64, n, context.temp_allocator)
	X := make([]complex64, n, context.temp_allocator)

	est: f32
	kase: Blas_Int
	isave := make([]Blas_Int, 3, context.temp_allocator)

	iterations = 0

	// Iterative estimation loop
	for {
		lapack.clacn2_(&n_blas, raw_data(V), raw_data(X), &est, &kase, raw_data(isave))

		if kase == 0 {
			break
		}

		iterations += 1

		if kase == 1 {
			// Apply matrix: Y := A * X
			apply_matrix(X, V)
		} else {
			// Apply conjugate transpose: Y := A^H * X
			apply_matrix(X, V) // User should handle transpose internally
		}
	}

	return est, iterations
}

// Estimate 1-norm of matrix (complex128)
m_estimate_1norm_c128 :: proc(
	n: int,
	apply_matrix: proc(X: []complex128, Y: []complex128),
	allocator := context.allocator,
) -> (
	estimate: f64,
	iterations: int,
) {
	n_blas := Blas_Int(n)

	// Allocate workspace
	V := make([]complex128, n, context.temp_allocator)
	X := make([]complex128, n, context.temp_allocator)

	est: f64
	kase: Blas_Int
	isave := make([]Blas_Int, 3, context.temp_allocator)

	iterations = 0

	// Iterative estimation loop
	for {
		lapack.zlacn2_(&n_blas, raw_data(V), raw_data(X), &est, &kase, raw_data(isave))

		if kase == 0 {
			break
		}

		iterations += 1

		if kase == 1 {
			// Apply matrix: Y := A * X
			apply_matrix(X, V)
		} else {
			// Apply conjugate transpose: Y := A^H * X
			apply_matrix(X, V) // User should handle transpose internally
		}
	}

	return est, iterations
}

// Estimate 1-norm of real matrix (f32)
m_estimate_1norm_f32 :: proc(
	n: int,
	apply_matrix: proc(X: []f32, Y: []f32),
	allocator := context.allocator,
) -> (
	estimate: f32,
	iterations: int,
) {
	n_blas := Blas_Int(n)

	// Allocate workspace
	V := make([]f32, n, context.temp_allocator)
	X := make([]f32, n, context.temp_allocator)
	ISGN := make([]Blas_Int, n, context.temp_allocator)

	est: f32
	kase: Blas_Int
	isave := make([]Blas_Int, 3, context.temp_allocator)

	iterations = 0

	// Iterative estimation loop
	for {
		lapack.slacn2_(
			&n_blas,
			raw_data(V),
			raw_data(X),
			raw_data(ISGN),
			&est,
			&kase,
			raw_data(isave),
		)

		if kase == 0 {
			break
		}

		iterations += 1

		if kase == 1 {
			// Apply matrix: Y := A * X
			apply_matrix(X, V)
		} else {
			// Apply transpose: Y := A^T * X
			apply_matrix(X, V) // User should handle transpose internally
		}
	}

	return est, iterations
}

// ===================================================================================
// LAPACK MATRIX COPY AND CONVERSION UTILITIES
// ===================================================================================

// Copy real matrix to complex matrix with zero imaginary parts
m_copy_real_to_complex :: proc {
	m_copy_real_to_complex_f32_c64,
	m_copy_real_to_complex_f64_c128,
}

// Copy real matrix to complex matrix (f32 -> complex64)
m_copy_real_to_complex_f32_c64 :: proc(
	A: ^Matrix(f32),
	B: ^Matrix(complex64),
	upper: bool = false, // Copy upper triangle only
	lower: bool = false, // Copy lower triangle only
) {
	// Validate matrices
	if A.format != .General {
		panic("A matrix must be in general format")
	}
	if B.format != .General {
		panic("B matrix must be in general format")
	}
	if A.rows != B.rows || A.cols != B.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Determine uplo parameter
	uplo := cstring("A") // All matrix
	if upper && !lower {
		uplo = cstring("U") // Upper triangle
	} else if lower && !upper {
		uplo = cstring("L") // Lower triangle
	}

	lapack.clacp2_(&uplo, &m, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, 1)
}

// Copy real matrix to complex matrix (f64 -> complex128)
m_copy_real_to_complex_f64_c128 :: proc(
	A: ^Matrix(f64),
	B: ^Matrix(complex128),
	upper: bool = false,
	lower: bool = false,
) {
	// Validate matrices
	if A.format != .General {
		panic("A matrix must be in general format")
	}
	if B.format != .General {
		panic("B matrix must be in general format")
	}
	if A.rows != B.rows || A.cols != B.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Determine uplo parameter
	uplo := cstring("A") // All matrix
	if upper && !lower {
		uplo = cstring("U") // Upper triangle
	} else if lower && !upper {
		uplo = cstring("L") // Lower triangle
	}

	lapack.zlacp2_(&uplo, &m, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, 1)
}

// Copy matrix with optional triangular selection
m_copy_matrix :: proc {
	m_copy_matrix_c64,
	m_copy_matrix_c128,
	m_copy_matrix_f32,
	m_copy_matrix_f64,
}

// Copy matrix (complex64)
m_copy_matrix_c64 :: proc(
	A: ^Matrix(complex64),
	B: ^Matrix(complex64),
	upper: bool = false, // Copy upper triangle only
	lower: bool = false, // Copy lower triangle only
) {
	// Validate matrices
	if A.format != .General {
		panic("A matrix must be in general format")
	}
	if B.format != .General {
		panic("B matrix must be in general format")
	}
	if A.rows != B.rows || A.cols != B.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Determine uplo parameter
	uplo := cstring("A") // All matrix
	if upper && !lower {
		uplo = cstring("U") // Upper triangle
	} else if lower && !upper {
		uplo = cstring("L") // Lower triangle
	}

	lapack.clacpy_(&uplo, &m, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, 1)
}

// Copy matrix (complex128)
m_copy_matrix_c128 :: proc(
	A: ^Matrix(complex128),
	B: ^Matrix(complex128),
	upper: bool = false,
	lower: bool = false,
) {
	// Validate matrices
	if A.format != .General {
		panic("A matrix must be in general format")
	}
	if B.format != .General {
		panic("B matrix must be in general format")
	}
	if A.rows != B.rows || A.cols != B.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Determine uplo parameter
	uplo := cstring("A") // All matrix
	if upper && !lower {
		uplo = cstring("U") // Upper triangle
	} else if lower && !upper {
		uplo = cstring("L") // Lower triangle
	}

	lapack.zlacpy_(&uplo, &m, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, 1)
}

// Copy matrix (f32)
m_copy_matrix_f32 :: proc(
	A: ^Matrix(f32),
	B: ^Matrix(f32),
	upper: bool = false,
	lower: bool = false,
) {
	// Validate matrices
	if A.format != .General {
		panic("A matrix must be in general format")
	}
	if B.format != .General {
		panic("B matrix must be in general format")
	}
	if A.rows != B.rows || A.cols != B.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Determine uplo parameter
	uplo := cstring("A") // All matrix
	if upper && !lower {
		uplo = cstring("U") // Upper triangle
	} else if lower && !upper {
		uplo = cstring("L") // Lower triangle
	}

	lapack.slacpy_(&uplo, &m, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, 1)
}

// Copy matrix (f64)
m_copy_matrix_f64 :: proc(
	A: ^Matrix(f64),
	B: ^Matrix(f64),
	upper: bool = false,
	lower: bool = false,
) {
	// Validate matrices
	if A.format != .General {
		panic("A matrix must be in general format")
	}
	if B.format != .General {
		panic("B matrix must be in general format")
	}
	if A.rows != B.rows || A.cols != B.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Determine uplo parameter
	uplo := cstring("A") // All matrix
	if upper && !lower {
		uplo = cstring("U") // Upper triangle
	} else if lower && !upper {
		uplo = cstring("L") // Lower triangle
	}

	lapack.dlacpy_(&uplo, &m, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, 1)
}

// Complex matrix-real matrix multiplication: C := A * B where A is complex, B is real
m_multiply_complex_real :: proc {
	m_multiply_complex_real_c64_f32,
	m_multiply_complex_real_c128_f64,
}

// Complex-real matrix multiplication (complex64 * f32)
m_multiply_complex_real_c64_f32 :: proc(
	A: ^Matrix(complex64),
	B: ^Matrix(f32),
	C: ^Matrix(complex64),
	allocator := context.allocator,
) {
	// Validate matrices
	if A.format != .General || B.format != .General || C.format != .General {
		panic("All matrices must be in general format")
	}
	if A.cols != B.rows {
		panic("Matrix dimensions incompatible for multiplication")
	}
	if A.rows != C.rows || B.cols != C.cols {
		panic("Output matrix dimensions incorrect")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(B.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)
	ldc := Blas_Int(C.ld)

	// Allocate workspace for real components
	rwork_size := max(1, int(m * n))
	rwork := make([]f32, rwork_size, context.temp_allocator)

	lapack.clacrm_(
		&m,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(C.data),
		&ldc,
		raw_data(rwork),
	)
}

// ===================================================================================
// LAPACK PRECISION CONVERSION UTILITIES
// ===================================================================================

// Convert between different numerical precisions with overflow checking
m_convert_precision :: proc {
	m_convert_precision_c128_c64,
	m_convert_precision_f32_f64,
	m_convert_precision_f64_f32,
	m_convert_precision_c64_c128,
}

// Convert complex128 matrix to complex64 with overflow checking
m_convert_precision_c128_c64 :: proc(
	A: ^Matrix(complex128),
	SA: ^Matrix(complex64),
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrices
	if A.format != .General || SA.format != .General {
		panic("Both matrices must be in general format")
	}
	if A.rows != SA.rows || A.cols != SA.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldsa := Blas_Int(SA.ld)

	info_val: Blas_Int
	lapack.zlag2c_(&m, &n, raw_data(A.data), &lda, raw_data(SA.data), &ldsa, &info_val)

	return info_val == 0, info_val
}

// Convert f32 matrix to f64 (always successful)
m_convert_precision_f32_f64 :: proc(
	SA: ^Matrix(f32),
	A: ^Matrix(f64),
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrices
	if SA.format != .General || A.format != .General {
		panic("Both matrices must be in general format")
	}
	if SA.rows != A.rows || SA.cols != A.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(SA.rows)
	n := Blas_Int(SA.cols)
	ldsa := Blas_Int(SA.ld)
	lda := Blas_Int(A.ld)

	info_val: Blas_Int
	lapack.slag2d_(&m, &n, raw_data(SA.data), &ldsa, raw_data(A.data), &lda, &info_val)

	return info_val == 0, info_val
}

// Convert f64 matrix to f32 with overflow checking
m_convert_precision_f64_f32 :: proc(
	A: ^Matrix(f64),
	SA: ^Matrix(f32),
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrices
	if A.format != .General || SA.format != .General {
		panic("Both matrices must be in general format")
	}
	if A.rows != SA.rows || A.cols != SA.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldsa := Blas_Int(SA.ld)

	info_val: Blas_Int
	lapack.dlag2s_(&m, &n, raw_data(A.data), &lda, raw_data(SA.data), &ldsa, &info_val)

	return info_val == 0, info_val
}

// Convert complex64 matrix to complex128 (always successful)
m_convert_precision_c64_c128 :: proc(
	SA: ^Matrix(complex64),
	A: ^Matrix(complex128),
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrices
	if SA.format != .General || A.format != .General {
		panic("Both matrices must be in general format")
	}
	if SA.rows != A.rows || SA.cols != A.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(SA.rows)
	n := Blas_Int(SA.cols)
	ldsa := Blas_Int(SA.ld)
	lda := Blas_Int(A.ld)

	info_val: Blas_Int
	lapack.clag2z_(&m, &n, raw_data(SA.data), &ldsa, raw_data(A.data), &lda, &info_val)

	return info_val == 0, info_val
}

// ===================================================================================
// LAPACK RANDOM MATRIX GENERATION UTILITIES
// ===================================================================================

// Generate random general banded matrix
m_generate_random_banded :: proc {
	m_generate_random_banded_c64,
	m_generate_random_banded_c128,
	m_generate_random_banded_f32,
	m_generate_random_banded_f64,
}

// Generate random banded matrix (complex64)
m_generate_random_banded_c64 :: proc(
	A: ^Matrix(complex64),
	kl, ku: int, // Lower and upper bandwidth
	D: []f32, // Diagonal scaling factors
	seed: []int = nil, // Random seed (4 elements), auto-generated if nil
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrix
	if A.format != .General {
		panic("Matrix must be in general format")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix rows")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	kl_val := Blas_Int(kl)
	ku_val := Blas_Int(ku)
	lda := Blas_Int(A.ld)

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		// Generate default seed
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work_size := max(m, n)
	work := make([]complex64, work_size, context.temp_allocator)

	info_val: Blas_Int
	lapack.clagge_(
		&m,
		&n,
		&kl_val,
		&ku_val,
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}

// Generate random banded matrix (complex128)
m_generate_random_banded_c128 :: proc(
	A: ^Matrix(complex128),
	kl, ku: int,
	D: []f64,
	seed: []int = nil,
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrix
	if A.format != .General {
		panic("Matrix must be in general format")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix rows")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	kl_val := Blas_Int(kl)
	ku_val := Blas_Int(ku)
	lda := Blas_Int(A.ld)

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work_size := max(m, n)
	work := make([]complex128, work_size, context.temp_allocator)

	info_val: Blas_Int
	lapack.zlagge_(
		&m,
		&n,
		&kl_val,
		&ku_val,
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}

// Generate random banded matrix (f32)
m_generate_random_banded_f32 :: proc(
	A: ^Matrix(f32),
	kl, ku: int,
	D: []f32,
	seed: []int = nil,
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrix
	if A.format != .General {
		panic("Matrix must be in general format")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix rows")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	kl_val := Blas_Int(kl)
	ku_val := Blas_Int(ku)
	lda := Blas_Int(A.ld)

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work_size := max(m, n)
	work := make([]f32, work_size, context.temp_allocator)

	info_val: Blas_Int
	lapack.slagge_(
		&m,
		&n,
		&kl_val,
		&ku_val,
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}

// Generate random banded matrix (f64)
m_generate_random_banded_f64 :: proc(
	A: ^Matrix(f64),
	kl, ku: int,
	D: []f64,
	seed: []int = nil,
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrix
	if A.format != .General {
		panic("Matrix must be in general format")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix rows")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	kl_val := Blas_Int(kl)
	ku_val := Blas_Int(ku)
	lda := Blas_Int(A.ld)

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work_size := max(m, n)
	work := make([]f64, work_size, context.temp_allocator)

	info_val: Blas_Int
	lapack.dlagge_(
		&m,
		&n,
		&kl_val,
		&ku_val,
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}

// Complex-real matrix multiplication (complex128 * f64)
m_multiply_complex_real_c128_f64 :: proc(
	A: ^Matrix(complex128),
	B: ^Matrix(f64),
	C: ^Matrix(complex128),
	allocator := context.allocator,
) {
	// Validate matrices
	if A.format != .General || B.format != .General || C.format != .General {
		panic("All matrices must be in general format")
	}
	if A.cols != B.rows {
		panic("Matrix dimensions incompatible for multiplication")
	}
	if A.rows != C.rows || B.cols != C.cols {
		panic("Output matrix dimensions incorrect")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(B.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)
	ldc := Blas_Int(C.ld)

	// Allocate workspace for real components
	rwork_size := max(1, int(m * n))
	rwork := make([]f64, rwork_size, context.temp_allocator)

	lapack.zlacrm_(
		&m,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(C.data),
		&ldc,
		raw_data(rwork),
	)
}

// ===================================================================================
// LAPACK PRECISION CONVERSION UTILITIES
// ===================================================================================

// Convert between different numerical precisions with overflow checking
m_convert_precision :: proc {
	m_convert_precision_c128_c64,
	m_convert_precision_f32_f64,
	m_convert_precision_f64_f32,
	m_convert_precision_c64_c128,
}

// Convert complex128 matrix to complex64 with overflow checking
m_convert_precision_c128_c64 :: proc(
	A: ^Matrix(complex128),
	SA: ^Matrix(complex64),
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrices
	if A.format != .General || SA.format != .General {
		panic("Both matrices must be in general format")
	}
	if A.rows != SA.rows || A.cols != SA.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldsa := Blas_Int(SA.ld)

	info_val: Blas_Int
	lapack.zlag2c_(&m, &n, raw_data(A.data), &lda, raw_data(SA.data), &ldsa, &info_val)

	return info_val == 0, info_val
}

// Convert f32 matrix to f64 (always successful)
m_convert_precision_f32_f64 :: proc(
	SA: ^Matrix(f32),
	A: ^Matrix(f64),
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrices
	if SA.format != .General || A.format != .General {
		panic("Both matrices must be in general format")
	}
	if SA.rows != A.rows || SA.cols != A.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(SA.rows)
	n := Blas_Int(SA.cols)
	ldsa := Blas_Int(SA.ld)
	lda := Blas_Int(A.ld)

	info_val: Blas_Int
	lapack.slag2d_(&m, &n, raw_data(SA.data), &ldsa, raw_data(A.data), &lda, &info_val)

	return info_val == 0, info_val
}

// Convert f64 matrix to f32 with overflow checking
m_convert_precision_f64_f32 :: proc(
	A: ^Matrix(f64),
	SA: ^Matrix(f32),
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrices
	if A.format != .General || SA.format != .General {
		panic("Both matrices must be in general format")
	}
	if A.rows != SA.rows || A.cols != SA.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldsa := Blas_Int(SA.ld)

	info_val: Blas_Int
	lapack.dlag2s_(&m, &n, raw_data(A.data), &lda, raw_data(SA.data), &ldsa, &info_val)

	return info_val == 0, info_val
}

// Convert complex64 matrix to complex128 (always successful)
m_convert_precision_c64_c128 :: proc(
	SA: ^Matrix(complex64),
	A: ^Matrix(complex128),
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrices
	if SA.format != .General || A.format != .General {
		panic("Both matrices must be in general format")
	}
	if SA.rows != A.rows || SA.cols != A.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(SA.rows)
	n := Blas_Int(SA.cols)
	ldsa := Blas_Int(SA.ld)
	lda := Blas_Int(A.ld)

	info_val: Blas_Int
	lapack.clag2z_(&m, &n, raw_data(SA.data), &ldsa, raw_data(A.data), &lda, &info_val)

	return info_val == 0, info_val
}

// ===================================================================================
// LAPACK RANDOM MATRIX GENERATION UTILITIES
// ===================================================================================

// Generate random general banded matrix
m_generate_random_banded :: proc {
	m_generate_random_banded_c64,
	m_generate_random_banded_c128,
	m_generate_random_banded_f32,
	m_generate_random_banded_f64,
}

// Generate random banded matrix (complex64)
m_generate_random_banded_c64 :: proc(
	A: ^Matrix(complex64),
	kl, ku: int, // Lower and upper bandwidth
	D: []f32, // Diagonal scaling factors
	seed: []int = nil, // Random seed (4 elements), auto-generated if nil
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrix
	if A.format != .General {
		panic("Matrix must be in general format")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix rows")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	kl_val := Blas_Int(kl)
	ku_val := Blas_Int(ku)
	lda := Blas_Int(A.ld)

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		// Generate default seed
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work_size := max(m, n)
	work := make([]complex64, work_size, context.temp_allocator)

	info_val: Blas_Int
	lapack.clagge_(
		&m,
		&n,
		&kl_val,
		&ku_val,
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}

// Generate random banded matrix (complex128)
m_generate_random_banded_c128 :: proc(
	A: ^Matrix(complex128),
	kl, ku: int,
	D: []f64,
	seed: []int = nil,
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrix
	if A.format != .General {
		panic("Matrix must be in general format")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix rows")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	kl_val := Blas_Int(kl)
	ku_val := Blas_Int(ku)
	lda := Blas_Int(A.ld)

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work_size := max(m, n)
	work := make([]complex128, work_size, context.temp_allocator)

	info_val: Blas_Int
	lapack.zlagge_(
		&m,
		&n,
		&kl_val,
		&ku_val,
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}

// Generate random banded matrix (f32)
m_generate_random_banded_f32 :: proc(
	A: ^Matrix(f32),
	kl, ku: int,
	D: []f32,
	seed: []int = nil,
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrix
	if A.format != .General {
		panic("Matrix must be in general format")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix rows")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	kl_val := Blas_Int(kl)
	ku_val := Blas_Int(ku)
	lda := Blas_Int(A.ld)

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work_size := max(m, n)
	work := make([]f32, work_size, context.temp_allocator)

	info_val: Blas_Int
	lapack.slagge_(
		&m,
		&n,
		&kl_val,
		&ku_val,
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}

// Generate random banded matrix (f64)
m_generate_random_banded_f64 :: proc(
	A: ^Matrix(f64),
	kl, ku: int,
	D: []f64,
	seed: []int = nil,
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrix
	if A.format != .General {
		panic("Matrix must be in general format")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix rows")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	kl_val := Blas_Int(kl)
	ku_val := Blas_Int(ku)
	lda := Blas_Int(A.ld)

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work_size := max(m, n)
	work := make([]f64, work_size, context.temp_allocator)

	info_val: Blas_Int
	lapack.dlagge_(
		&m,
		&n,
		&kl_val,
		&ku_val,
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}

// Estimate 1-norm of real matrix (f64)
m_estimate_1norm_f64 :: proc(
	n: int,
	apply_matrix: proc(X: []f64, Y: []f64),
	allocator := context.allocator,
) -> (
	estimate: f64,
	iterations: int,
) {
	n_blas := Blas_Int(n)

	// Allocate workspace
	V := make([]f64, n, context.temp_allocator)
	X := make([]f64, n, context.temp_allocator)
	ISGN := make([]Blas_Int, n, context.temp_allocator)

	est: f64
	kase: Blas_Int
	isave := make([]Blas_Int, 3, context.temp_allocator)

	iterations = 0

	// Iterative estimation loop
	for {
		lapack.dlacn2_(
			&n_blas,
			raw_data(V),
			raw_data(X),
			raw_data(ISGN),
			&est,
			&kase,
			raw_data(isave),
		)

		if kase == 0 {
			break
		}

		iterations += 1

		if kase == 1 {
			// Apply matrix: Y := A * X
			apply_matrix(X, V)
		} else {
			// Apply transpose: Y := A^T * X
			apply_matrix(X, V) // User should handle transpose internally
		}
	}

	return est, iterations
}

// ===================================================================================
// LAPACK MATRIX COPY AND CONVERSION UTILITIES
// ===================================================================================

// Copy real matrix to complex matrix with zero imaginary parts
m_copy_real_to_complex :: proc {
	m_copy_real_to_complex_f32_c64,
	m_copy_real_to_complex_f64_c128,
}

// Copy real matrix to complex matrix (f32 -> complex64)
m_copy_real_to_complex_f32_c64 :: proc(
	A: ^Matrix(f32),
	B: ^Matrix(complex64),
	upper: bool = false, // Copy upper triangle only
	lower: bool = false, // Copy lower triangle only
) {
	// Validate matrices
	if A.format != .General {
		panic("A matrix must be in general format")
	}
	if B.format != .General {
		panic("B matrix must be in general format")
	}
	if A.rows != B.rows || A.cols != B.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Determine uplo parameter
	uplo := cstring("A") // All matrix
	if upper && !lower {
		uplo = cstring("U") // Upper triangle
	} else if lower && !upper {
		uplo = cstring("L") // Lower triangle
	}

	lapack.clacp2_(&uplo, &m, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, 1)
}

// Copy real matrix to complex matrix (f64 -> complex128)
m_copy_real_to_complex_f64_c128 :: proc(
	A: ^Matrix(f64),
	B: ^Matrix(complex128),
	upper: bool = false,
	lower: bool = false,
) {
	// Validate matrices
	if A.format != .General {
		panic("A matrix must be in general format")
	}
	if B.format != .General {
		panic("B matrix must be in general format")
	}
	if A.rows != B.rows || A.cols != B.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Determine uplo parameter
	uplo := cstring("A") // All matrix
	if upper && !lower {
		uplo = cstring("U") // Upper triangle
	} else if lower && !upper {
		uplo = cstring("L") // Lower triangle
	}

	lapack.zlacp2_(&uplo, &m, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, 1)
}

// Copy matrix with optional triangular selection
m_copy_matrix :: proc {
	m_copy_matrix_c64,
	m_copy_matrix_c128,
	m_copy_matrix_f32,
	m_copy_matrix_f64,
}

// Copy matrix (complex64)
m_copy_matrix_c64 :: proc(
	A: ^Matrix(complex64),
	B: ^Matrix(complex64),
	upper: bool = false, // Copy upper triangle only
	lower: bool = false, // Copy lower triangle only
) {
	// Validate matrices
	if A.format != .General {
		panic("A matrix must be in general format")
	}
	if B.format != .General {
		panic("B matrix must be in general format")
	}
	if A.rows != B.rows || A.cols != B.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Determine uplo parameter
	uplo := cstring("A") // All matrix
	if upper && !lower {
		uplo = cstring("U") // Upper triangle
	} else if lower && !upper {
		uplo = cstring("L") // Lower triangle
	}

	lapack.clacpy_(&uplo, &m, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, 1)
}

// Copy matrix (complex128)
m_copy_matrix_c128 :: proc(
	A: ^Matrix(complex128),
	B: ^Matrix(complex128),
	upper: bool = false,
	lower: bool = false,
) {
	// Validate matrices
	if A.format != .General {
		panic("A matrix must be in general format")
	}
	if B.format != .General {
		panic("B matrix must be in general format")
	}
	if A.rows != B.rows || A.cols != B.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Determine uplo parameter
	uplo := cstring("A") // All matrix
	if upper && !lower {
		uplo = cstring("U") // Upper triangle
	} else if lower && !upper {
		uplo = cstring("L") // Lower triangle
	}

	lapack.zlacpy_(&uplo, &m, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, 1)
}

// Copy matrix (f32)
m_copy_matrix_f32 :: proc(
	A: ^Matrix(f32),
	B: ^Matrix(f32),
	upper: bool = false,
	lower: bool = false,
) {
	// Validate matrices
	if A.format != .General {
		panic("A matrix must be in general format")
	}
	if B.format != .General {
		panic("B matrix must be in general format")
	}
	if A.rows != B.rows || A.cols != B.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Determine uplo parameter
	uplo := cstring("A") // All matrix
	if upper && !lower {
		uplo = cstring("U") // Upper triangle
	} else if lower && !upper {
		uplo = cstring("L") // Lower triangle
	}

	lapack.slacpy_(&uplo, &m, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, 1)
}

// Copy matrix (f64)
m_copy_matrix_f64 :: proc(
	A: ^Matrix(f64),
	B: ^Matrix(f64),
	upper: bool = false,
	lower: bool = false,
) {
	// Validate matrices
	if A.format != .General {
		panic("A matrix must be in general format")
	}
	if B.format != .General {
		panic("B matrix must be in general format")
	}
	if A.rows != B.rows || A.cols != B.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Determine uplo parameter
	uplo := cstring("A") // All matrix
	if upper && !lower {
		uplo = cstring("U") // Upper triangle
	} else if lower && !upper {
		uplo = cstring("L") // Lower triangle
	}

	lapack.dlacpy_(&uplo, &m, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, 1)
}

// Complex matrix-real matrix multiplication: C := A * B where A is complex, B is real
m_multiply_complex_real :: proc {
	m_multiply_complex_real_c64_f32,
	m_multiply_complex_real_c128_f64,
}

// Complex-real matrix multiplication (complex64 * f32)
m_multiply_complex_real_c64_f32 :: proc(
	A: ^Matrix(complex64),
	B: ^Matrix(f32),
	C: ^Matrix(complex64),
	allocator := context.allocator,
) {
	// Validate matrices
	if A.format != .General || B.format != .General || C.format != .General {
		panic("All matrices must be in general format")
	}
	if A.cols != B.rows {
		panic("Matrix dimensions incompatible for multiplication")
	}
	if A.rows != C.rows || B.cols != C.cols {
		panic("Output matrix dimensions incorrect")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(B.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)
	ldc := Blas_Int(C.ld)

	// Allocate workspace for real components
	rwork_size := max(1, int(m * n))
	rwork := make([]f32, rwork_size, context.temp_allocator)

	lapack.clacrm_(
		&m,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(C.data),
		&ldc,
		raw_data(rwork),
	)
}

// ===================================================================================
// LAPACK PRECISION CONVERSION UTILITIES
// ===================================================================================

// Convert between different numerical precisions with overflow checking
m_convert_precision :: proc {
	m_convert_precision_c128_c64,
	m_convert_precision_f32_f64,
	m_convert_precision_f64_f32,
	m_convert_precision_c64_c128,
}

// Convert complex128 matrix to complex64 with overflow checking
m_convert_precision_c128_c64 :: proc(
	A: ^Matrix(complex128),
	SA: ^Matrix(complex64),
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrices
	if A.format != .General || SA.format != .General {
		panic("Both matrices must be in general format")
	}
	if A.rows != SA.rows || A.cols != SA.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldsa := Blas_Int(SA.ld)

	info_val: Blas_Int
	lapack.zlag2c_(&m, &n, raw_data(A.data), &lda, raw_data(SA.data), &ldsa, &info_val)

	return info_val == 0, info_val
}

// Convert f32 matrix to f64 (always successful)
m_convert_precision_f32_f64 :: proc(
	SA: ^Matrix(f32),
	A: ^Matrix(f64),
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrices
	if SA.format != .General || A.format != .General {
		panic("Both matrices must be in general format")
	}
	if SA.rows != A.rows || SA.cols != A.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(SA.rows)
	n := Blas_Int(SA.cols)
	ldsa := Blas_Int(SA.ld)
	lda := Blas_Int(A.ld)

	info_val: Blas_Int
	lapack.slag2d_(&m, &n, raw_data(SA.data), &ldsa, raw_data(A.data), &lda, &info_val)

	return info_val == 0, info_val
}

// Convert f64 matrix to f32 with overflow checking
m_convert_precision_f64_f32 :: proc(
	A: ^Matrix(f64),
	SA: ^Matrix(f32),
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrices
	if A.format != .General || SA.format != .General {
		panic("Both matrices must be in general format")
	}
	if A.rows != SA.rows || A.cols != SA.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldsa := Blas_Int(SA.ld)

	info_val: Blas_Int
	lapack.dlag2s_(&m, &n, raw_data(A.data), &lda, raw_data(SA.data), &ldsa, &info_val)

	return info_val == 0, info_val
}

// Convert complex64 matrix to complex128 (always successful)
m_convert_precision_c64_c128 :: proc(
	SA: ^Matrix(complex64),
	A: ^Matrix(complex128),
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrices
	if SA.format != .General || A.format != .General {
		panic("Both matrices must be in general format")
	}
	if SA.rows != A.rows || SA.cols != A.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(SA.rows)
	n := Blas_Int(SA.cols)
	ldsa := Blas_Int(SA.ld)
	lda := Blas_Int(A.ld)

	info_val: Blas_Int
	lapack.clag2z_(&m, &n, raw_data(SA.data), &ldsa, raw_data(A.data), &lda, &info_val)

	return info_val == 0, info_val
}

// ===================================================================================
// LAPACK RANDOM MATRIX GENERATION UTILITIES
// ===================================================================================

// Generate random general banded matrix
m_generate_random_banded :: proc {
	m_generate_random_banded_c64,
	m_generate_random_banded_c128,
	m_generate_random_banded_f32,
	m_generate_random_banded_f64,
}

// Generate random banded matrix (complex64)
m_generate_random_banded_c64 :: proc(
	A: ^Matrix(complex64),
	kl, ku: int, // Lower and upper bandwidth
	D: []f32, // Diagonal scaling factors
	seed: []int = nil, // Random seed (4 elements), auto-generated if nil
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrix
	if A.format != .General {
		panic("Matrix must be in general format")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix rows")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	kl_val := Blas_Int(kl)
	ku_val := Blas_Int(ku)
	lda := Blas_Int(A.ld)

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		// Generate default seed
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work_size := max(m, n)
	work := make([]complex64, work_size, context.temp_allocator)

	info_val: Blas_Int
	lapack.clagge_(
		&m,
		&n,
		&kl_val,
		&ku_val,
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}

// Generate random banded matrix (complex128)
m_generate_random_banded_c128 :: proc(
	A: ^Matrix(complex128),
	kl, ku: int,
	D: []f64,
	seed: []int = nil,
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrix
	if A.format != .General {
		panic("Matrix must be in general format")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix rows")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	kl_val := Blas_Int(kl)
	ku_val := Blas_Int(ku)
	lda := Blas_Int(A.ld)

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work_size := max(m, n)
	work := make([]complex128, work_size, context.temp_allocator)

	info_val: Blas_Int
	lapack.zlagge_(
		&m,
		&n,
		&kl_val,
		&ku_val,
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}

// Generate random banded matrix (f32)
m_generate_random_banded_f32 :: proc(
	A: ^Matrix(f32),
	kl, ku: int,
	D: []f32,
	seed: []int = nil,
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrix
	if A.format != .General {
		panic("Matrix must be in general format")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix rows")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	kl_val := Blas_Int(kl)
	ku_val := Blas_Int(ku)
	lda := Blas_Int(A.ld)

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work_size := max(m, n)
	work := make([]f32, work_size, context.temp_allocator)

	info_val: Blas_Int
	lapack.slagge_(
		&m,
		&n,
		&kl_val,
		&ku_val,
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}

// Generate random banded matrix (f64)
m_generate_random_banded_f64 :: proc(
	A: ^Matrix(f64),
	kl, ku: int,
	D: []f64,
	seed: []int = nil,
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrix
	if A.format != .General {
		panic("Matrix must be in general format")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix rows")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	kl_val := Blas_Int(kl)
	ku_val := Blas_Int(ku)
	lda := Blas_Int(A.ld)

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work_size := max(m, n)
	work := make([]f64, work_size, context.temp_allocator)

	info_val: Blas_Int
	lapack.dlagge_(
		&m,
		&n,
		&kl_val,
		&ku_val,
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}

// Complex-real matrix multiplication (complex128 * f64)
m_multiply_complex_real_c128_f64 :: proc(
	A: ^Matrix(complex128),
	B: ^Matrix(f64),
	C: ^Matrix(complex128),
	allocator := context.allocator,
) {
	// Validate matrices
	if A.format != .General || B.format != .General || C.format != .General {
		panic("All matrices must be in general format")
	}
	if A.cols != B.rows {
		panic("Matrix dimensions incompatible for multiplication")
	}
	if A.rows != C.rows || B.cols != C.cols {
		panic("Output matrix dimensions incorrect")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(B.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)
	ldc := Blas_Int(C.ld)

	// Allocate workspace for real components
	rwork_size := max(1, int(m * n))
	rwork := make([]f64, rwork_size, context.temp_allocator)

	lapack.zlacrm_(
		&m,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(C.data),
		&ldc,
		raw_data(rwork),
	)
}

// ===================================================================================
// LAPACK PRECISION CONVERSION UTILITIES
// ===================================================================================

// Convert between different numerical precisions with overflow checking
m_convert_precision :: proc {
	m_convert_precision_c128_c64,
	m_convert_precision_f32_f64,
	m_convert_precision_f64_f32,
	m_convert_precision_c64_c128,
}

// Convert complex128 matrix to complex64 with overflow checking
m_convert_precision_c128_c64 :: proc(
	A: ^Matrix(complex128),
	SA: ^Matrix(complex64),
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrices
	if A.format != .General || SA.format != .General {
		panic("Both matrices must be in general format")
	}
	if A.rows != SA.rows || A.cols != SA.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldsa := Blas_Int(SA.ld)

	info_val: Blas_Int
	lapack.zlag2c_(&m, &n, raw_data(A.data), &lda, raw_data(SA.data), &ldsa, &info_val)

	return info_val == 0, info_val
}

// Convert f32 matrix to f64 (always successful)
m_convert_precision_f32_f64 :: proc(
	SA: ^Matrix(f32),
	A: ^Matrix(f64),
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrices
	if SA.format != .General || A.format != .General {
		panic("Both matrices must be in general format")
	}
	if SA.rows != A.rows || SA.cols != A.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(SA.rows)
	n := Blas_Int(SA.cols)
	ldsa := Blas_Int(SA.ld)
	lda := Blas_Int(A.ld)

	info_val: Blas_Int
	lapack.slag2d_(&m, &n, raw_data(SA.data), &ldsa, raw_data(A.data), &lda, &info_val)

	return info_val == 0, info_val
}

// Convert f64 matrix to f32 with overflow checking
m_convert_precision_f64_f32 :: proc(
	A: ^Matrix(f64),
	SA: ^Matrix(f32),
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrices
	if A.format != .General || SA.format != .General {
		panic("Both matrices must be in general format")
	}
	if A.rows != SA.rows || A.cols != SA.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldsa := Blas_Int(SA.ld)

	info_val: Blas_Int
	lapack.dlag2s_(&m, &n, raw_data(A.data), &lda, raw_data(SA.data), &ldsa, &info_val)

	return info_val == 0, info_val
}

// Convert complex64 matrix to complex128 (always successful)
m_convert_precision_c64_c128 :: proc(
	SA: ^Matrix(complex64),
	A: ^Matrix(complex128),
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrices
	if SA.format != .General || A.format != .General {
		panic("Both matrices must be in general format")
	}
	if SA.rows != A.rows || SA.cols != A.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(SA.rows)
	n := Blas_Int(SA.cols)
	ldsa := Blas_Int(SA.ld)
	lda := Blas_Int(A.ld)

	info_val: Blas_Int
	lapack.clag2z_(&m, &n, raw_data(SA.data), &ldsa, raw_data(A.data), &lda, &info_val)

	return info_val == 0, info_val
}

// ===================================================================================
// LAPACK RANDOM MATRIX GENERATION UTILITIES
// ===================================================================================

// Generate random general banded matrix
m_generate_random_banded :: proc {
	m_generate_random_banded_c64,
	m_generate_random_banded_c128,
	m_generate_random_banded_f32,
	m_generate_random_banded_f64,
}

// Generate random banded matrix (complex64)
m_generate_random_banded_c64 :: proc(
	A: ^Matrix(complex64),
	kl, ku: int, // Lower and upper bandwidth
	D: []f32, // Diagonal scaling factors
	seed: []int = nil, // Random seed (4 elements), auto-generated if nil
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrix
	if A.format != .General {
		panic("Matrix must be in general format")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix rows")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	kl_val := Blas_Int(kl)
	ku_val := Blas_Int(ku)
	lda := Blas_Int(A.ld)

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		// Generate default seed
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work_size := max(m, n)
	work := make([]complex64, work_size, context.temp_allocator)

	info_val: Blas_Int
	lapack.clagge_(
		&m,
		&n,
		&kl_val,
		&ku_val,
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}

// Generate random banded matrix (complex128)
m_generate_random_banded_c128 :: proc(
	A: ^Matrix(complex128),
	kl, ku: int,
	D: []f64,
	seed: []int = nil,
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrix
	if A.format != .General {
		panic("Matrix must be in general format")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix rows")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	kl_val := Blas_Int(kl)
	ku_val := Blas_Int(ku)
	lda := Blas_Int(A.ld)

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work_size := max(m, n)
	work := make([]complex128, work_size, context.temp_allocator)

	info_val: Blas_Int
	lapack.zlagge_(
		&m,
		&n,
		&kl_val,
		&ku_val,
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}

// Generate random banded matrix (f32)
m_generate_random_banded_f32 :: proc(
	A: ^Matrix(f32),
	kl, ku: int,
	D: []f32,
	seed: []int = nil,
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrix
	if A.format != .General {
		panic("Matrix must be in general format")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix rows")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	kl_val := Blas_Int(kl)
	ku_val := Blas_Int(ku)
	lda := Blas_Int(A.ld)

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work_size := max(m, n)
	work := make([]f32, work_size, context.temp_allocator)

	info_val: Blas_Int
	lapack.slagge_(
		&m,
		&n,
		&kl_val,
		&ku_val,
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}

// Generate random banded matrix (f64)
m_generate_random_banded_f64 :: proc(
	A: ^Matrix(f64),
	kl, ku: int,
	D: []f64,
	seed: []int = nil,
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrix
	if A.format != .General {
		panic("Matrix must be in general format")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix rows")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	kl_val := Blas_Int(kl)
	ku_val := Blas_Int(ku)
	lda := Blas_Int(A.ld)

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work_size := max(m, n)
	work := make([]f64, work_size, context.temp_allocator)

	info_val: Blas_Int
	lapack.dlagge_(
		&m,
		&n,
		&kl_val,
		&ku_val,
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}

// Compute eigenvectors of real Hessenberg matrix (f32)
m_eigenvectors_hessenberg_f32 :: proc(
	H: ^Matrix(f32),
	eigenvalues_real: []f32,
	eigenvalues_imag: []f32,
	compute_left: bool = false,
	compute_right: bool = true,
	eigsrc: bool = false,
	select_mask: []bool = nil,
	allocator := context.allocator,
) -> (
	VL, VR: Matrix(f32),
	m_computed: int,
	ifail: []int,
	info: Info,
) {
	// Validate matrix
	if H.format != .General {
		panic("H matrix must be in general format")
	}

	n := Blas_Int(H.rows)
	if H.rows != H.cols {
		panic("H matrix must be square")
	}

	// Setup computation sides
	side_c :=
		cstring("B") if compute_left && compute_right else cstring("L") if compute_left else cstring("R")

	eigsrc_c := cstring("Q") if eigsrc else cstring("N")
	initv_c := cstring("N")

	// Setup selection
	var; select_raw: []Blas_Int
	if select_mask != nil && len(select_mask) == int(n) {
		select_raw = make([]Blas_Int, n, context.temp_allocator)
		for i, selected in select_mask {
			select_raw[i] = 1 if selected else 0
		}
	} else {
		select_raw = make([]Blas_Int, n, context.temp_allocator)
		for i in 0 ..< n {
			select_raw[i] = 1
		}
	}

	ldh := Blas_Int(H.ld)
	ldvl := n if compute_left else 1
	ldvr := n if compute_right else 1

	// Allocate output matrices
	VL = matrix_zeros(f32, int(ldvl), int(n), allocator) if compute_left else Matrix(f32){}
	VR = matrix_zeros(f32, int(ldvr), int(n), allocator) if compute_right else Matrix(f32){}

	mm := n
	m_out: Blas_Int

	// Copy eigenvalue arrays
	WR := make([]f32, n, context.temp_allocator)
	WI := make([]f32, n, context.temp_allocator)
	copy(WR, eigenvalues_real)
	copy(WI, eigenvalues_imag)

	// Allocate workspace
	work := make([]f32, n * n, context.temp_allocator)

	ifaill := make([]Blas_Int, n, context.temp_allocator)
	ifailr := make([]Blas_Int, n, context.temp_allocator)

	// Call LAPACK
	info_val: Blas_Int
	vl_ptr := raw_data(VL.data) if compute_left else nil
	vr_ptr := raw_data(VR.data) if compute_right else nil

	lapack.shsein_(
		&side_c,
		&eigsrc_c,
		&initv_c,
		raw_data(select_raw),
		&n,
		raw_data(H.data),
		&ldh,
		raw_data(WR),
		raw_data(WI),
		vl_ptr,
		&ldvl,
		vr_ptr,
		&ldvr,
		&mm,
		&m_out,
		raw_data(work),
		raw_data(ifaill),
		raw_data(ifailr),
		&info_val,
		1,
		1,
		1,
	)

	// Convert failure indices
	ifail = make([]int, n, allocator)
	for i in 0 ..< n {
		if compute_left && ifaill[i] != 0 {
			ifail[i] = int(ifaill[i])
		} else if compute_right && ifailr[i] != 0 {
			ifail[i] = int(ifailr[i])
		}
	}

	return VL, VR, int(m_out), ifail, info_val
}

// ===================================================================================
// HESSENBERG QR ALGORITHM FOR EIGENVALUE COMPUTATION
// ===================================================================================

// Compute eigenvalues and optionally Schur form of Hessenberg matrix using QR algorithm
m_eigenvalues_hessenberg_qr :: proc {
	m_eigenvalues_hessenberg_qr_c64,
	m_eigenvalues_hessenberg_qr_c128,
	m_eigenvalues_hessenberg_qr_f32,
	m_eigenvalues_hessenberg_qr_f64,
}

// QR algorithm for Hessenberg matrix eigenvalues (complex64)
m_eigenvalues_hessenberg_qr_c64 :: proc(
	H: ^Matrix(complex64),
	ilo: int = 0, // Submatrix range (0 means use full matrix)
	ihi: int = 0, // Submatrix range (0 means use full matrix)
	compute_schur: bool = true,
	compute_eigenvectors: bool = false,
	Z: ^Matrix(complex64) = nil, // Optional input/output matrix for eigenvectors
	allocator := context.allocator,
) -> (
	eigenvalues: []complex64,
	Z_out: Matrix(complex64),
	success: bool,
	info: Info,
) {
	// Validate matrix
	if H.format != .General {
		panic("H matrix must be in general format")
	}

	n := Blas_Int(H.rows)
	if H.rows != H.cols {
		panic("H matrix must be square")
	}

	// Setup range
	ilo_val := Blas_Int(ilo if ilo > 0 else 1)
	ihi_val := Blas_Int(ihi if ihi > 0 else int(n))

	// Setup job parameters
	job_c := cstring("S") if compute_schur else cstring("E") // Schur form or eigenvalues only
	compz_c := cstring("V") if compute_eigenvectors else cstring("N") // Compute or don't compute Z

	ldh := Blas_Int(H.ld)

	// Setup Z matrix
	ldz: Blas_Int
	if compute_eigenvectors {
		if Z != nil {
			if Z.rows != H.rows || Z.cols != H.cols {
				panic("Z matrix dimensions must match H matrix")
			}
			Z_out = Z^
			ldz = Blas_Int(Z.ld)
		} else {
			Z_out = matrix_zeros(complex64, int(n), int(n), allocator)
			ldz = n
		}
	} else {
		Z_out = Matrix(complex64){}
		ldz = 1
	}

	// Allocate eigenvalue array
	W := make([]complex64, n, allocator)

	// Workspace query
	work_query: complex64
	lwork_query := Blas_Int(-1)
	info_val: Blas_Int

	z_ptr := raw_data(Z_out.data) if compute_eigenvectors else nil

	lapack.chseqr_(
		&job_c,
		&compz_c,
		&n,
		&ilo_val,
		&ihi_val,
		raw_data(H.data),
		&ldh,
		raw_data(W),
		z_ptr,
		&ldz,
		&work_query,
		&lwork_query,
		&info_val,
		1,
		1,
	)

	if info_val != 0 {
		return nil, Z_out, false, info_val
	}

	// Allocate workspace
	lwork := Blas_Int(real(work_query))
	work := make([]complex64, lwork, context.temp_allocator)

	// Compute eigenvalues
	lapack.chseqr_(
		&job_c,
		&compz_c,
		&n,
		&ilo_val,
		&ihi_val,
		raw_data(H.data),
		&ldh,
		raw_data(W),
		z_ptr,
		&ldz,
		raw_data(work),
		&lwork,
		&info_val,
		1,
		1,
	)

	eigenvalues = W
	return eigenvalues, Z_out, info_val == 0, info_val
}

// ===================================================================================
// LAPACK UTILITY FUNCTIONS
// ===================================================================================

// Complex conjugate of a vector
m_conjugate_vector :: proc {
	m_conjugate_vector_c64,
	m_conjugate_vector_c128,
}

// Conjugate complex vector (complex64)
m_conjugate_vector_c64 :: proc(X: ^Vector(complex64)) {
	n := Blas_Int(len(X.data))
	incx := Blas_Int(X.incr)

	lapack.clacgv_(&n, raw_data(X.data), &incx)
}

// Conjugate complex vector (complex128)
m_conjugate_vector_c128 :: proc(X: ^Vector(complex128)) {
	n := Blas_Int(len(X.data))
	incx := Blas_Int(X.incr)

	lapack.zlacgv_(&n, raw_data(X.data), &incx)
}

// Estimate the 1-norm of a matrix using iterative method
m_estimate_1norm :: proc {
	m_estimate_1norm_c64,
	m_estimate_1norm_c128,
	m_estimate_1norm_f32,
	m_estimate_1norm_f64,
}

// Estimate 1-norm of matrix (complex64)
m_estimate_1norm_c64 :: proc(
	n: int,
	apply_matrix: proc(X: []complex64, Y: []complex64), // User-provided matrix application
	allocator := context.allocator,
) -> (
	estimate: f32,
	iterations: int,
) {
	n_blas := Blas_Int(n)

	// Allocate workspace
	V := make([]complex64, n, context.temp_allocator)
	X := make([]complex64, n, context.temp_allocator)

	est: f32
	kase: Blas_Int
	isave := make([]Blas_Int, 3, context.temp_allocator)

	iterations = 0

	// Iterative estimation loop
	for {
		lapack.clacn2_(&n_blas, raw_data(V), raw_data(X), &est, &kase, raw_data(isave))

		if kase == 0 {
			break
		}

		iterations += 1

		if kase == 1 {
			// Apply matrix: Y := A * X
			apply_matrix(X, V)
		} else {
			// Apply conjugate transpose: Y := A^H * X
			apply_matrix(X, V) // User should handle transpose internally
		}
	}

	return est, iterations
}

// Estimate 1-norm of matrix (complex128)
m_estimate_1norm_c128 :: proc(
	n: int,
	apply_matrix: proc(X: []complex128, Y: []complex128),
	allocator := context.allocator,
) -> (
	estimate: f64,
	iterations: int,
) {
	n_blas := Blas_Int(n)

	// Allocate workspace
	V := make([]complex128, n, context.temp_allocator)
	X := make([]complex128, n, context.temp_allocator)

	est: f64
	kase: Blas_Int
	isave := make([]Blas_Int, 3, context.temp_allocator)

	iterations = 0

	// Iterative estimation loop
	for {
		lapack.zlacn2_(&n_blas, raw_data(V), raw_data(X), &est, &kase, raw_data(isave))

		if kase == 0 {
			break
		}

		iterations += 1

		if kase == 1 {
			// Apply matrix: Y := A * X
			apply_matrix(X, V)
		} else {
			// Apply conjugate transpose: Y := A^H * X
			apply_matrix(X, V) // User should handle transpose internally
		}
	}

	return est, iterations
}

// Estimate 1-norm of real matrix (f32)
m_estimate_1norm_f32 :: proc(
	n: int,
	apply_matrix: proc(X: []f32, Y: []f32),
	allocator := context.allocator,
) -> (
	estimate: f32,
	iterations: int,
) {
	n_blas := Blas_Int(n)

	// Allocate workspace
	V := make([]f32, n, context.temp_allocator)
	X := make([]f32, n, context.temp_allocator)
	ISGN := make([]Blas_Int, n, context.temp_allocator)

	est: f32
	kase: Blas_Int
	isave := make([]Blas_Int, 3, context.temp_allocator)

	iterations = 0

	// Iterative estimation loop
	for {
		lapack.slacn2_(
			&n_blas,
			raw_data(V),
			raw_data(X),
			raw_data(ISGN),
			&est,
			&kase,
			raw_data(isave),
		)

		if kase == 0 {
			break
		}

		iterations += 1

		if kase == 1 {
			// Apply matrix: Y := A * X
			apply_matrix(X, V)
		} else {
			// Apply transpose: Y := A^T * X
			apply_matrix(X, V) // User should handle transpose internally
		}
	}

	return est, iterations
}

// ===================================================================================
// LAPACK MATRIX COPY AND CONVERSION UTILITIES
// ===================================================================================

// Copy real matrix to complex matrix with zero imaginary parts
m_copy_real_to_complex :: proc {
	m_copy_real_to_complex_f32_c64,
	m_copy_real_to_complex_f64_c128,
}

// Copy real matrix to complex matrix (f32 -> complex64)
m_copy_real_to_complex_f32_c64 :: proc(
	A: ^Matrix(f32),
	B: ^Matrix(complex64),
	upper: bool = false, // Copy upper triangle only
	lower: bool = false, // Copy lower triangle only
) {
	// Validate matrices
	if A.format != .General {
		panic("A matrix must be in general format")
	}
	if B.format != .General {
		panic("B matrix must be in general format")
	}
	if A.rows != B.rows || A.cols != B.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Determine uplo parameter
	uplo := cstring("A") // All matrix
	if upper && !lower {
		uplo = cstring("U") // Upper triangle
	} else if lower && !upper {
		uplo = cstring("L") // Lower triangle
	}

	lapack.clacp2_(&uplo, &m, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, 1)
}

// Copy real matrix to complex matrix (f64 -> complex128)
m_copy_real_to_complex_f64_c128 :: proc(
	A: ^Matrix(f64),
	B: ^Matrix(complex128),
	upper: bool = false,
	lower: bool = false,
) {
	// Validate matrices
	if A.format != .General {
		panic("A matrix must be in general format")
	}
	if B.format != .General {
		panic("B matrix must be in general format")
	}
	if A.rows != B.rows || A.cols != B.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Determine uplo parameter
	uplo := cstring("A") // All matrix
	if upper && !lower {
		uplo = cstring("U") // Upper triangle
	} else if lower && !upper {
		uplo = cstring("L") // Lower triangle
	}

	lapack.zlacp2_(&uplo, &m, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, 1)
}

// Copy matrix with optional triangular selection
m_copy_matrix :: proc {
	m_copy_matrix_c64,
	m_copy_matrix_c128,
	m_copy_matrix_f32,
	m_copy_matrix_f64,
}

// Copy matrix (complex64)
m_copy_matrix_c64 :: proc(
	A: ^Matrix(complex64),
	B: ^Matrix(complex64),
	upper: bool = false, // Copy upper triangle only
	lower: bool = false, // Copy lower triangle only
) {
	// Validate matrices
	if A.format != .General {
		panic("A matrix must be in general format")
	}
	if B.format != .General {
		panic("B matrix must be in general format")
	}
	if A.rows != B.rows || A.cols != B.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Determine uplo parameter
	uplo := cstring("A") // All matrix
	if upper && !lower {
		uplo = cstring("U") // Upper triangle
	} else if lower && !upper {
		uplo = cstring("L") // Lower triangle
	}

	lapack.clacpy_(&uplo, &m, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, 1)
}

// Copy matrix (complex128)
m_copy_matrix_c128 :: proc(
	A: ^Matrix(complex128),
	B: ^Matrix(complex128),
	upper: bool = false,
	lower: bool = false,
) {
	// Validate matrices
	if A.format != .General {
		panic("A matrix must be in general format")
	}
	if B.format != .General {
		panic("B matrix must be in general format")
	}
	if A.rows != B.rows || A.cols != B.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Determine uplo parameter
	uplo := cstring("A") // All matrix
	if upper && !lower {
		uplo = cstring("U") // Upper triangle
	} else if lower && !upper {
		uplo = cstring("L") // Lower triangle
	}

	lapack.zlacpy_(&uplo, &m, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, 1)
}

// Copy matrix (f32)
m_copy_matrix_f32 :: proc(
	A: ^Matrix(f32),
	B: ^Matrix(f32),
	upper: bool = false,
	lower: bool = false,
) {
	// Validate matrices
	if A.format != .General {
		panic("A matrix must be in general format")
	}
	if B.format != .General {
		panic("B matrix must be in general format")
	}
	if A.rows != B.rows || A.cols != B.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Determine uplo parameter
	uplo := cstring("A") // All matrix
	if upper && !lower {
		uplo = cstring("U") // Upper triangle
	} else if lower && !upper {
		uplo = cstring("L") // Lower triangle
	}

	lapack.slacpy_(&uplo, &m, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, 1)
}

// Copy matrix (f64)
m_copy_matrix_f64 :: proc(
	A: ^Matrix(f64),
	B: ^Matrix(f64),
	upper: bool = false,
	lower: bool = false,
) {
	// Validate matrices
	if A.format != .General {
		panic("A matrix must be in general format")
	}
	if B.format != .General {
		panic("B matrix must be in general format")
	}
	if A.rows != B.rows || A.cols != B.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Determine uplo parameter
	uplo := cstring("A") // All matrix
	if upper && !lower {
		uplo = cstring("U") // Upper triangle
	} else if lower && !upper {
		uplo = cstring("L") // Lower triangle
	}

	lapack.dlacpy_(&uplo, &m, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, 1)
}

// Complex matrix-real matrix multiplication: C := A * B where A is complex, B is real
m_multiply_complex_real :: proc {
	m_multiply_complex_real_c64_f32,
	m_multiply_complex_real_c128_f64,
}

// Complex-real matrix multiplication (complex64 * f32)
m_multiply_complex_real_c64_f32 :: proc(
	A: ^Matrix(complex64),
	B: ^Matrix(f32),
	C: ^Matrix(complex64),
	allocator := context.allocator,
) {
	// Validate matrices
	if A.format != .General || B.format != .General || C.format != .General {
		panic("All matrices must be in general format")
	}
	if A.cols != B.rows {
		panic("Matrix dimensions incompatible for multiplication")
	}
	if A.rows != C.rows || B.cols != C.cols {
		panic("Output matrix dimensions incorrect")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(B.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)
	ldc := Blas_Int(C.ld)

	// Allocate workspace for real components
	rwork_size := max(1, int(m * n))
	rwork := make([]f32, rwork_size, context.temp_allocator)

	lapack.clacrm_(
		&m,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(C.data),
		&ldc,
		raw_data(rwork),
	)
}

// ===================================================================================
// LAPACK PRECISION CONVERSION UTILITIES
// ===================================================================================

// Convert between different numerical precisions with overflow checking
m_convert_precision :: proc {
	m_convert_precision_c128_c64,
	m_convert_precision_f32_f64,
	m_convert_precision_f64_f32,
	m_convert_precision_c64_c128,
}

// Convert complex128 matrix to complex64 with overflow checking
m_convert_precision_c128_c64 :: proc(
	A: ^Matrix(complex128),
	SA: ^Matrix(complex64),
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrices
	if A.format != .General || SA.format != .General {
		panic("Both matrices must be in general format")
	}
	if A.rows != SA.rows || A.cols != SA.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldsa := Blas_Int(SA.ld)

	info_val: Blas_Int
	lapack.zlag2c_(&m, &n, raw_data(A.data), &lda, raw_data(SA.data), &ldsa, &info_val)

	return info_val == 0, info_val
}

// Convert f32 matrix to f64 (always successful)
m_convert_precision_f32_f64 :: proc(
	SA: ^Matrix(f32),
	A: ^Matrix(f64),
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrices
	if SA.format != .General || A.format != .General {
		panic("Both matrices must be in general format")
	}
	if SA.rows != A.rows || SA.cols != A.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(SA.rows)
	n := Blas_Int(SA.cols)
	ldsa := Blas_Int(SA.ld)
	lda := Blas_Int(A.ld)

	info_val: Blas_Int
	lapack.slag2d_(&m, &n, raw_data(SA.data), &ldsa, raw_data(A.data), &lda, &info_val)

	return info_val == 0, info_val
}

// Convert f64 matrix to f32 with overflow checking
m_convert_precision_f64_f32 :: proc(
	A: ^Matrix(f64),
	SA: ^Matrix(f32),
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrices
	if A.format != .General || SA.format != .General {
		panic("Both matrices must be in general format")
	}
	if A.rows != SA.rows || A.cols != SA.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldsa := Blas_Int(SA.ld)

	info_val: Blas_Int
	lapack.dlag2s_(&m, &n, raw_data(A.data), &lda, raw_data(SA.data), &ldsa, &info_val)

	return info_val == 0, info_val
}

// Convert complex64 matrix to complex128 (always successful)
m_convert_precision_c64_c128 :: proc(
	SA: ^Matrix(complex64),
	A: ^Matrix(complex128),
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrices
	if SA.format != .General || A.format != .General {
		panic("Both matrices must be in general format")
	}
	if SA.rows != A.rows || SA.cols != A.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(SA.rows)
	n := Blas_Int(SA.cols)
	ldsa := Blas_Int(SA.ld)
	lda := Blas_Int(A.ld)

	info_val: Blas_Int
	lapack.clag2z_(&m, &n, raw_data(SA.data), &ldsa, raw_data(A.data), &lda, &info_val)

	return info_val == 0, info_val
}

// ===================================================================================
// LAPACK RANDOM MATRIX GENERATION UTILITIES
// ===================================================================================

// Generate random general banded matrix
m_generate_random_banded :: proc {
	m_generate_random_banded_c64,
	m_generate_random_banded_c128,
	m_generate_random_banded_f32,
	m_generate_random_banded_f64,
}

// Generate random banded matrix (complex64)
m_generate_random_banded_c64 :: proc(
	A: ^Matrix(complex64),
	kl, ku: int, // Lower and upper bandwidth
	D: []f32, // Diagonal scaling factors
	seed: []int = nil, // Random seed (4 elements), auto-generated if nil
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrix
	if A.format != .General {
		panic("Matrix must be in general format")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix rows")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	kl_val := Blas_Int(kl)
	ku_val := Blas_Int(ku)
	lda := Blas_Int(A.ld)

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		// Generate default seed
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work_size := max(m, n)
	work := make([]complex64, work_size, context.temp_allocator)

	info_val: Blas_Int
	lapack.clagge_(
		&m,
		&n,
		&kl_val,
		&ku_val,
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}

// Generate random banded matrix (complex128)
m_generate_random_banded_c128 :: proc(
	A: ^Matrix(complex128),
	kl, ku: int,
	D: []f64,
	seed: []int = nil,
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrix
	if A.format != .General {
		panic("Matrix must be in general format")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix rows")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	kl_val := Blas_Int(kl)
	ku_val := Blas_Int(ku)
	lda := Blas_Int(A.ld)

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work_size := max(m, n)
	work := make([]complex128, work_size, context.temp_allocator)

	info_val: Blas_Int
	lapack.zlagge_(
		&m,
		&n,
		&kl_val,
		&ku_val,
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}

// Generate random banded matrix (f32)
m_generate_random_banded_f32 :: proc(
	A: ^Matrix(f32),
	kl, ku: int,
	D: []f32,
	seed: []int = nil,
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrix
	if A.format != .General {
		panic("Matrix must be in general format")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix rows")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	kl_val := Blas_Int(kl)
	ku_val := Blas_Int(ku)
	lda := Blas_Int(A.ld)

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work_size := max(m, n)
	work := make([]f32, work_size, context.temp_allocator)

	info_val: Blas_Int
	lapack.slagge_(
		&m,
		&n,
		&kl_val,
		&ku_val,
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}

// Generate random banded matrix (f64)
m_generate_random_banded_f64 :: proc(
	A: ^Matrix(f64),
	kl, ku: int,
	D: []f64,
	seed: []int = nil,
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrix
	if A.format != .General {
		panic("Matrix must be in general format")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix rows")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	kl_val := Blas_Int(kl)
	ku_val := Blas_Int(ku)
	lda := Blas_Int(A.ld)

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work_size := max(m, n)
	work := make([]f64, work_size, context.temp_allocator)

	info_val: Blas_Int
	lapack.dlagge_(
		&m,
		&n,
		&kl_val,
		&ku_val,
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}

// Complex-real matrix multiplication (complex128 * f64)
m_multiply_complex_real_c128_f64 :: proc(
	A: ^Matrix(complex128),
	B: ^Matrix(f64),
	C: ^Matrix(complex128),
	allocator := context.allocator,
) {
	// Validate matrices
	if A.format != .General || B.format != .General || C.format != .General {
		panic("All matrices must be in general format")
	}
	if A.cols != B.rows {
		panic("Matrix dimensions incompatible for multiplication")
	}
	if A.rows != C.rows || B.cols != C.cols {
		panic("Output matrix dimensions incorrect")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(B.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)
	ldc := Blas_Int(C.ld)

	// Allocate workspace for real components
	rwork_size := max(1, int(m * n))
	rwork := make([]f64, rwork_size, context.temp_allocator)

	lapack.zlacrm_(
		&m,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(C.data),
		&ldc,
		raw_data(rwork),
	)
}

// ===================================================================================
// LAPACK PRECISION CONVERSION UTILITIES
// ===================================================================================

// Convert between different numerical precisions with overflow checking
m_convert_precision :: proc {
	m_convert_precision_c128_c64,
	m_convert_precision_f32_f64,
	m_convert_precision_f64_f32,
	m_convert_precision_c64_c128,
}

// Convert complex128 matrix to complex64 with overflow checking
m_convert_precision_c128_c64 :: proc(
	A: ^Matrix(complex128),
	SA: ^Matrix(complex64),
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrices
	if A.format != .General || SA.format != .General {
		panic("Both matrices must be in general format")
	}
	if A.rows != SA.rows || A.cols != SA.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldsa := Blas_Int(SA.ld)

	info_val: Blas_Int
	lapack.zlag2c_(&m, &n, raw_data(A.data), &lda, raw_data(SA.data), &ldsa, &info_val)

	return info_val == 0, info_val
}

// Convert f32 matrix to f64 (always successful)
m_convert_precision_f32_f64 :: proc(
	SA: ^Matrix(f32),
	A: ^Matrix(f64),
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrices
	if SA.format != .General || A.format != .General {
		panic("Both matrices must be in general format")
	}
	if SA.rows != A.rows || SA.cols != A.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(SA.rows)
	n := Blas_Int(SA.cols)
	ldsa := Blas_Int(SA.ld)
	lda := Blas_Int(A.ld)

	info_val: Blas_Int
	lapack.slag2d_(&m, &n, raw_data(SA.data), &ldsa, raw_data(A.data), &lda, &info_val)

	return info_val == 0, info_val
}

// Convert f64 matrix to f32 with overflow checking
m_convert_precision_f64_f32 :: proc(
	A: ^Matrix(f64),
	SA: ^Matrix(f32),
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrices
	if A.format != .General || SA.format != .General {
		panic("Both matrices must be in general format")
	}
	if A.rows != SA.rows || A.cols != SA.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldsa := Blas_Int(SA.ld)

	info_val: Blas_Int
	lapack.dlag2s_(&m, &n, raw_data(A.data), &lda, raw_data(SA.data), &ldsa, &info_val)

	return info_val == 0, info_val
}

// Convert complex64 matrix to complex128 (always successful)
m_convert_precision_c64_c128 :: proc(
	SA: ^Matrix(complex64),
	A: ^Matrix(complex128),
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrices
	if SA.format != .General || A.format != .General {
		panic("Both matrices must be in general format")
	}
	if SA.rows != A.rows || SA.cols != A.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(SA.rows)
	n := Blas_Int(SA.cols)
	ldsa := Blas_Int(SA.ld)
	lda := Blas_Int(A.ld)

	info_val: Blas_Int
	lapack.clag2z_(&m, &n, raw_data(SA.data), &ldsa, raw_data(A.data), &lda, &info_val)

	return info_val == 0, info_val
}

// ===================================================================================
// LAPACK RANDOM MATRIX GENERATION UTILITIES
// ===================================================================================

// Generate random general banded matrix
m_generate_random_banded :: proc {
	m_generate_random_banded_c64,
	m_generate_random_banded_c128,
	m_generate_random_banded_f32,
	m_generate_random_banded_f64,
}

// Generate random banded matrix (complex64)
m_generate_random_banded_c64 :: proc(
	A: ^Matrix(complex64),
	kl, ku: int, // Lower and upper bandwidth
	D: []f32, // Diagonal scaling factors
	seed: []int = nil, // Random seed (4 elements), auto-generated if nil
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrix
	if A.format != .General {
		panic("Matrix must be in general format")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix rows")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	kl_val := Blas_Int(kl)
	ku_val := Blas_Int(ku)
	lda := Blas_Int(A.ld)

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		// Generate default seed
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work_size := max(m, n)
	work := make([]complex64, work_size, context.temp_allocator)

	info_val: Blas_Int
	lapack.clagge_(
		&m,
		&n,
		&kl_val,
		&ku_val,
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}

// Generate random banded matrix (complex128)
m_generate_random_banded_c128 :: proc(
	A: ^Matrix(complex128),
	kl, ku: int,
	D: []f64,
	seed: []int = nil,
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrix
	if A.format != .General {
		panic("Matrix must be in general format")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix rows")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	kl_val := Blas_Int(kl)
	ku_val := Blas_Int(ku)
	lda := Blas_Int(A.ld)

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work_size := max(m, n)
	work := make([]complex128, work_size, context.temp_allocator)

	info_val: Blas_Int
	lapack.zlagge_(
		&m,
		&n,
		&kl_val,
		&ku_val,
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}

// Generate random banded matrix (f32)
m_generate_random_banded_f32 :: proc(
	A: ^Matrix(f32),
	kl, ku: int,
	D: []f32,
	seed: []int = nil,
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrix
	if A.format != .General {
		panic("Matrix must be in general format")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix rows")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	kl_val := Blas_Int(kl)
	ku_val := Blas_Int(ku)
	lda := Blas_Int(A.ld)

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work_size := max(m, n)
	work := make([]f32, work_size, context.temp_allocator)

	info_val: Blas_Int
	lapack.slagge_(
		&m,
		&n,
		&kl_val,
		&ku_val,
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}

// Generate random banded matrix (f64)
m_generate_random_banded_f64 :: proc(
	A: ^Matrix(f64),
	kl, ku: int,
	D: []f64,
	seed: []int = nil,
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrix
	if A.format != .General {
		panic("Matrix must be in general format")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix rows")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	kl_val := Blas_Int(kl)
	ku_val := Blas_Int(ku)
	lda := Blas_Int(A.ld)

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work_size := max(m, n)
	work := make([]f64, work_size, context.temp_allocator)

	info_val: Blas_Int
	lapack.dlagge_(
		&m,
		&n,
		&kl_val,
		&ku_val,
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}

// Estimate 1-norm of real matrix (f64)
m_estimate_1norm_f64 :: proc(
	n: int,
	apply_matrix: proc(X: []f64, Y: []f64),
	allocator := context.allocator,
) -> (
	estimate: f64,
	iterations: int,
) {
	n_blas := Blas_Int(n)

	// Allocate workspace
	V := make([]f64, n, context.temp_allocator)
	X := make([]f64, n, context.temp_allocator)
	ISGN := make([]Blas_Int, n, context.temp_allocator)

	est: f64
	kase: Blas_Int
	isave := make([]Blas_Int, 3, context.temp_allocator)

	iterations = 0

	// Iterative estimation loop
	for {
		lapack.dlacn2_(
			&n_blas,
			raw_data(V),
			raw_data(X),
			raw_data(ISGN),
			&est,
			&kase,
			raw_data(isave),
		)

		if kase == 0 {
			break
		}

		iterations += 1

		if kase == 1 {
			// Apply matrix: Y := A * X
			apply_matrix(X, V)
		} else {
			// Apply transpose: Y := A^T * X
			apply_matrix(X, V) // User should handle transpose internally
		}
	}

	return est, iterations
}

// ===================================================================================
// LAPACK MATRIX COPY AND CONVERSION UTILITIES
// ===================================================================================

// Copy real matrix to complex matrix with zero imaginary parts
m_copy_real_to_complex :: proc {
	m_copy_real_to_complex_f32_c64,
	m_copy_real_to_complex_f64_c128,
}

// Copy real matrix to complex matrix (f32 -> complex64)
m_copy_real_to_complex_f32_c64 :: proc(
	A: ^Matrix(f32),
	B: ^Matrix(complex64),
	upper: bool = false, // Copy upper triangle only
	lower: bool = false, // Copy lower triangle only
) {
	// Validate matrices
	if A.format != .General {
		panic("A matrix must be in general format")
	}
	if B.format != .General {
		panic("B matrix must be in general format")
	}
	if A.rows != B.rows || A.cols != B.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Determine uplo parameter
	uplo := cstring("A") // All matrix
	if upper && !lower {
		uplo = cstring("U") // Upper triangle
	} else if lower && !upper {
		uplo = cstring("L") // Lower triangle
	}

	lapack.clacp2_(&uplo, &m, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, 1)
}

// Copy real matrix to complex matrix (f64 -> complex128)
m_copy_real_to_complex_f64_c128 :: proc(
	A: ^Matrix(f64),
	B: ^Matrix(complex128),
	upper: bool = false,
	lower: bool = false,
) {
	// Validate matrices
	if A.format != .General {
		panic("A matrix must be in general format")
	}
	if B.format != .General {
		panic("B matrix must be in general format")
	}
	if A.rows != B.rows || A.cols != B.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Determine uplo parameter
	uplo := cstring("A") // All matrix
	if upper && !lower {
		uplo = cstring("U") // Upper triangle
	} else if lower && !upper {
		uplo = cstring("L") // Lower triangle
	}

	lapack.zlacp2_(&uplo, &m, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, 1)
}

// Copy matrix with optional triangular selection
m_copy_matrix :: proc {
	m_copy_matrix_c64,
	m_copy_matrix_c128,
	m_copy_matrix_f32,
	m_copy_matrix_f64,
}

// Copy matrix (complex64)
m_copy_matrix_c64 :: proc(
	A: ^Matrix(complex64),
	B: ^Matrix(complex64),
	upper: bool = false, // Copy upper triangle only
	lower: bool = false, // Copy lower triangle only
) {
	// Validate matrices
	if A.format != .General {
		panic("A matrix must be in general format")
	}
	if B.format != .General {
		panic("B matrix must be in general format")
	}
	if A.rows != B.rows || A.cols != B.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Determine uplo parameter
	uplo := cstring("A") // All matrix
	if upper && !lower {
		uplo = cstring("U") // Upper triangle
	} else if lower && !upper {
		uplo = cstring("L") // Lower triangle
	}

	lapack.clacpy_(&uplo, &m, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, 1)
}

// Copy matrix (complex128)
m_copy_matrix_c128 :: proc(
	A: ^Matrix(complex128),
	B: ^Matrix(complex128),
	upper: bool = false,
	lower: bool = false,
) {
	// Validate matrices
	if A.format != .General {
		panic("A matrix must be in general format")
	}
	if B.format != .General {
		panic("B matrix must be in general format")
	}
	if A.rows != B.rows || A.cols != B.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Determine uplo parameter
	uplo := cstring("A") // All matrix
	if upper && !lower {
		uplo = cstring("U") // Upper triangle
	} else if lower && !upper {
		uplo = cstring("L") // Lower triangle
	}

	lapack.zlacpy_(&uplo, &m, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, 1)
}

// Copy matrix (f32)
m_copy_matrix_f32 :: proc(
	A: ^Matrix(f32),
	B: ^Matrix(f32),
	upper: bool = false,
	lower: bool = false,
) {
	// Validate matrices
	if A.format != .General {
		panic("A matrix must be in general format")
	}
	if B.format != .General {
		panic("B matrix must be in general format")
	}
	if A.rows != B.rows || A.cols != B.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Determine uplo parameter
	uplo := cstring("A") // All matrix
	if upper && !lower {
		uplo = cstring("U") // Upper triangle
	} else if lower && !upper {
		uplo = cstring("L") // Lower triangle
	}

	lapack.slacpy_(&uplo, &m, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, 1)
}

// Copy matrix (f64)
m_copy_matrix_f64 :: proc(
	A: ^Matrix(f64),
	B: ^Matrix(f64),
	upper: bool = false,
	lower: bool = false,
) {
	// Validate matrices
	if A.format != .General {
		panic("A matrix must be in general format")
	}
	if B.format != .General {
		panic("B matrix must be in general format")
	}
	if A.rows != B.rows || A.cols != B.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Determine uplo parameter
	uplo := cstring("A") // All matrix
	if upper && !lower {
		uplo = cstring("U") // Upper triangle
	} else if lower && !upper {
		uplo = cstring("L") // Lower triangle
	}

	lapack.dlacpy_(&uplo, &m, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, 1)
}

// Complex matrix-real matrix multiplication: C := A * B where A is complex, B is real
m_multiply_complex_real :: proc {
	m_multiply_complex_real_c64_f32,
	m_multiply_complex_real_c128_f64,
}

// Complex-real matrix multiplication (complex64 * f32)
m_multiply_complex_real_c64_f32 :: proc(
	A: ^Matrix(complex64),
	B: ^Matrix(f32),
	C: ^Matrix(complex64),
	allocator := context.allocator,
) {
	// Validate matrices
	if A.format != .General || B.format != .General || C.format != .General {
		panic("All matrices must be in general format")
	}
	if A.cols != B.rows {
		panic("Matrix dimensions incompatible for multiplication")
	}
	if A.rows != C.rows || B.cols != C.cols {
		panic("Output matrix dimensions incorrect")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(B.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)
	ldc := Blas_Int(C.ld)

	// Allocate workspace for real components
	rwork_size := max(1, int(m * n))
	rwork := make([]f32, rwork_size, context.temp_allocator)

	lapack.clacrm_(
		&m,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(C.data),
		&ldc,
		raw_data(rwork),
	)
}

// ===================================================================================
// LAPACK PRECISION CONVERSION UTILITIES
// ===================================================================================

// Convert between different numerical precisions with overflow checking
m_convert_precision :: proc {
	m_convert_precision_c128_c64,
	m_convert_precision_f32_f64,
	m_convert_precision_f64_f32,
	m_convert_precision_c64_c128,
}

// Convert complex128 matrix to complex64 with overflow checking
m_convert_precision_c128_c64 :: proc(
	A: ^Matrix(complex128),
	SA: ^Matrix(complex64),
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrices
	if A.format != .General || SA.format != .General {
		panic("Both matrices must be in general format")
	}
	if A.rows != SA.rows || A.cols != SA.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldsa := Blas_Int(SA.ld)

	info_val: Blas_Int
	lapack.zlag2c_(&m, &n, raw_data(A.data), &lda, raw_data(SA.data), &ldsa, &info_val)

	return info_val == 0, info_val
}

// Convert f32 matrix to f64 (always successful)
m_convert_precision_f32_f64 :: proc(
	SA: ^Matrix(f32),
	A: ^Matrix(f64),
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrices
	if SA.format != .General || A.format != .General {
		panic("Both matrices must be in general format")
	}
	if SA.rows != A.rows || SA.cols != A.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(SA.rows)
	n := Blas_Int(SA.cols)
	ldsa := Blas_Int(SA.ld)
	lda := Blas_Int(A.ld)

	info_val: Blas_Int
	lapack.slag2d_(&m, &n, raw_data(SA.data), &ldsa, raw_data(A.data), &lda, &info_val)

	return info_val == 0, info_val
}

// Convert f64 matrix to f32 with overflow checking
m_convert_precision_f64_f32 :: proc(
	A: ^Matrix(f64),
	SA: ^Matrix(f32),
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrices
	if A.format != .General || SA.format != .General {
		panic("Both matrices must be in general format")
	}
	if A.rows != SA.rows || A.cols != SA.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldsa := Blas_Int(SA.ld)

	info_val: Blas_Int
	lapack.dlag2s_(&m, &n, raw_data(A.data), &lda, raw_data(SA.data), &ldsa, &info_val)

	return info_val == 0, info_val
}

// Convert complex64 matrix to complex128 (always successful)
m_convert_precision_c64_c128 :: proc(
	SA: ^Matrix(complex64),
	A: ^Matrix(complex128),
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrices
	if SA.format != .General || A.format != .General {
		panic("Both matrices must be in general format")
	}
	if SA.rows != A.rows || SA.cols != A.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(SA.rows)
	n := Blas_Int(SA.cols)
	ldsa := Blas_Int(SA.ld)
	lda := Blas_Int(A.ld)

	info_val: Blas_Int
	lapack.clag2z_(&m, &n, raw_data(SA.data), &ldsa, raw_data(A.data), &lda, &info_val)

	return info_val == 0, info_val
}

// ===================================================================================
// LAPACK RANDOM MATRIX GENERATION UTILITIES
// ===================================================================================

// Generate random general banded matrix
m_generate_random_banded :: proc {
	m_generate_random_banded_c64,
	m_generate_random_banded_c128,
	m_generate_random_banded_f32,
	m_generate_random_banded_f64,
}

// Generate random banded matrix (complex64)
m_generate_random_banded_c64 :: proc(
	A: ^Matrix(complex64),
	kl, ku: int, // Lower and upper bandwidth
	D: []f32, // Diagonal scaling factors
	seed: []int = nil, // Random seed (4 elements), auto-generated if nil
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrix
	if A.format != .General {
		panic("Matrix must be in general format")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix rows")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	kl_val := Blas_Int(kl)
	ku_val := Blas_Int(ku)
	lda := Blas_Int(A.ld)

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		// Generate default seed
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work_size := max(m, n)
	work := make([]complex64, work_size, context.temp_allocator)

	info_val: Blas_Int
	lapack.clagge_(
		&m,
		&n,
		&kl_val,
		&ku_val,
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}

// Generate random banded matrix (complex128)
m_generate_random_banded_c128 :: proc(
	A: ^Matrix(complex128),
	kl, ku: int,
	D: []f64,
	seed: []int = nil,
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrix
	if A.format != .General {
		panic("Matrix must be in general format")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix rows")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	kl_val := Blas_Int(kl)
	ku_val := Blas_Int(ku)
	lda := Blas_Int(A.ld)

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work_size := max(m, n)
	work := make([]complex128, work_size, context.temp_allocator)

	info_val: Blas_Int
	lapack.zlagge_(
		&m,
		&n,
		&kl_val,
		&ku_val,
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}

// Generate random banded matrix (f32)
m_generate_random_banded_f32 :: proc(
	A: ^Matrix(f32),
	kl, ku: int,
	D: []f32,
	seed: []int = nil,
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrix
	if A.format != .General {
		panic("Matrix must be in general format")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix rows")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	kl_val := Blas_Int(kl)
	ku_val := Blas_Int(ku)
	lda := Blas_Int(A.ld)

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work_size := max(m, n)
	work := make([]f32, work_size, context.temp_allocator)

	info_val: Blas_Int
	lapack.slagge_(
		&m,
		&n,
		&kl_val,
		&ku_val,
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}

// Generate random banded matrix (f64)
m_generate_random_banded_f64 :: proc(
	A: ^Matrix(f64),
	kl, ku: int,
	D: []f64,
	seed: []int = nil,
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrix
	if A.format != .General {
		panic("Matrix must be in general format")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix rows")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	kl_val := Blas_Int(kl)
	ku_val := Blas_Int(ku)
	lda := Blas_Int(A.ld)

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work_size := max(m, n)
	work := make([]f64, work_size, context.temp_allocator)

	info_val: Blas_Int
	lapack.dlagge_(
		&m,
		&n,
		&kl_val,
		&ku_val,
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}

// Complex-real matrix multiplication (complex128 * f64)
m_multiply_complex_real_c128_f64 :: proc(
	A: ^Matrix(complex128),
	B: ^Matrix(f64),
	C: ^Matrix(complex128),
	allocator := context.allocator,
) {
	// Validate matrices
	if A.format != .General || B.format != .General || C.format != .General {
		panic("All matrices must be in general format")
	}
	if A.cols != B.rows {
		panic("Matrix dimensions incompatible for multiplication")
	}
	if A.rows != C.rows || B.cols != C.cols {
		panic("Output matrix dimensions incorrect")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(B.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)
	ldc := Blas_Int(C.ld)

	// Allocate workspace for real components
	rwork_size := max(1, int(m * n))
	rwork := make([]f64, rwork_size, context.temp_allocator)

	lapack.zlacrm_(
		&m,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(C.data),
		&ldc,
		raw_data(rwork),
	)
}

// ===================================================================================
// LAPACK PRECISION CONVERSION UTILITIES
// ===================================================================================

// Convert between different numerical precisions with overflow checking
m_convert_precision :: proc {
	m_convert_precision_c128_c64,
	m_convert_precision_f32_f64,
	m_convert_precision_f64_f32,
	m_convert_precision_c64_c128,
}

// Convert complex128 matrix to complex64 with overflow checking
m_convert_precision_c128_c64 :: proc(
	A: ^Matrix(complex128),
	SA: ^Matrix(complex64),
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrices
	if A.format != .General || SA.format != .General {
		panic("Both matrices must be in general format")
	}
	if A.rows != SA.rows || A.cols != SA.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldsa := Blas_Int(SA.ld)

	info_val: Blas_Int
	lapack.zlag2c_(&m, &n, raw_data(A.data), &lda, raw_data(SA.data), &ldsa, &info_val)

	return info_val == 0, info_val
}

// Convert f32 matrix to f64 (always successful)
m_convert_precision_f32_f64 :: proc(
	SA: ^Matrix(f32),
	A: ^Matrix(f64),
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrices
	if SA.format != .General || A.format != .General {
		panic("Both matrices must be in general format")
	}
	if SA.rows != A.rows || SA.cols != A.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(SA.rows)
	n := Blas_Int(SA.cols)
	ldsa := Blas_Int(SA.ld)
	lda := Blas_Int(A.ld)

	info_val: Blas_Int
	lapack.slag2d_(&m, &n, raw_data(SA.data), &ldsa, raw_data(A.data), &lda, &info_val)

	return info_val == 0, info_val
}

// Convert f64 matrix to f32 with overflow checking
m_convert_precision_f64_f32 :: proc(
	A: ^Matrix(f64),
	SA: ^Matrix(f32),
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrices
	if A.format != .General || SA.format != .General {
		panic("Both matrices must be in general format")
	}
	if A.rows != SA.rows || A.cols != SA.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldsa := Blas_Int(SA.ld)

	info_val: Blas_Int
	lapack.dlag2s_(&m, &n, raw_data(A.data), &lda, raw_data(SA.data), &ldsa, &info_val)

	return info_val == 0, info_val
}

// Convert complex64 matrix to complex128 (always successful)
m_convert_precision_c64_c128 :: proc(
	SA: ^Matrix(complex64),
	A: ^Matrix(complex128),
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrices
	if SA.format != .General || A.format != .General {
		panic("Both matrices must be in general format")
	}
	if SA.rows != A.rows || SA.cols != A.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(SA.rows)
	n := Blas_Int(SA.cols)
	ldsa := Blas_Int(SA.ld)
	lda := Blas_Int(A.ld)

	info_val: Blas_Int
	lapack.clag2z_(&m, &n, raw_data(SA.data), &ldsa, raw_data(A.data), &lda, &info_val)

	return info_val == 0, info_val
}

// ===================================================================================
// LAPACK RANDOM MATRIX GENERATION UTILITIES
// ===================================================================================

// Generate random general banded matrix
m_generate_random_banded :: proc {
	m_generate_random_banded_c64,
	m_generate_random_banded_c128,
	m_generate_random_banded_f32,
	m_generate_random_banded_f64,
}

// Generate random banded matrix (complex64)
m_generate_random_banded_c64 :: proc(
	A: ^Matrix(complex64),
	kl, ku: int, // Lower and upper bandwidth
	D: []f32, // Diagonal scaling factors
	seed: []int = nil, // Random seed (4 elements), auto-generated if nil
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrix
	if A.format != .General {
		panic("Matrix must be in general format")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix rows")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	kl_val := Blas_Int(kl)
	ku_val := Blas_Int(ku)
	lda := Blas_Int(A.ld)

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		// Generate default seed
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work_size := max(m, n)
	work := make([]complex64, work_size, context.temp_allocator)

	info_val: Blas_Int
	lapack.clagge_(
		&m,
		&n,
		&kl_val,
		&ku_val,
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}

// Generate random banded matrix (complex128)
m_generate_random_banded_c128 :: proc(
	A: ^Matrix(complex128),
	kl, ku: int,
	D: []f64,
	seed: []int = nil,
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrix
	if A.format != .General {
		panic("Matrix must be in general format")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix rows")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	kl_val := Blas_Int(kl)
	ku_val := Blas_Int(ku)
	lda := Blas_Int(A.ld)

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work_size := max(m, n)
	work := make([]complex128, work_size, context.temp_allocator)

	info_val: Blas_Int
	lapack.zlagge_(
		&m,
		&n,
		&kl_val,
		&ku_val,
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}

// Generate random banded matrix (f32)
m_generate_random_banded_f32 :: proc(
	A: ^Matrix(f32),
	kl, ku: int,
	D: []f32,
	seed: []int = nil,
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrix
	if A.format != .General {
		panic("Matrix must be in general format")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix rows")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	kl_val := Blas_Int(kl)
	ku_val := Blas_Int(ku)
	lda := Blas_Int(A.ld)

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work_size := max(m, n)
	work := make([]f32, work_size, context.temp_allocator)

	info_val: Blas_Int
	lapack.slagge_(
		&m,
		&n,
		&kl_val,
		&ku_val,
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}

// Generate random banded matrix (f64)
m_generate_random_banded_f64 :: proc(
	A: ^Matrix(f64),
	kl, ku: int,
	D: []f64,
	seed: []int = nil,
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrix
	if A.format != .General {
		panic("Matrix must be in general format")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix rows")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	kl_val := Blas_Int(kl)
	ku_val := Blas_Int(ku)
	lda := Blas_Int(A.ld)

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work_size := max(m, n)
	work := make([]f64, work_size, context.temp_allocator)

	info_val: Blas_Int
	lapack.dlagge_(
		&m,
		&n,
		&kl_val,
		&ku_val,
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}

// Compute eigenvectors of real Hessenberg matrix (f64)
m_eigenvectors_hessenberg_f64 :: proc(
	H: ^Matrix(f64),
	eigenvalues_real: []f64,
	eigenvalues_imag: []f64,
	compute_left: bool = false,
	compute_right: bool = true,
	eigsrc: bool = false,
	select_mask: []bool = nil,
	allocator := context.allocator,
) -> (
	VL, VR: Matrix(f64),
	m_computed: int,
	ifail: []int,
	info: Info,
) {
	// Validate matrix
	if H.format != .General {
		panic("H matrix must be in general format")
	}

	n := Blas_Int(H.rows)
	if H.rows != H.cols {
		panic("H matrix must be square")
	}

	// Setup computation sides
	side_c :=
		cstring("B") if compute_left && compute_right else cstring("L") if compute_left else cstring("R")

	eigsrc_c := cstring("Q") if eigsrc else cstring("N")
	initv_c := cstring("N")

	// Setup selection
	var; select_raw: []Blas_Int
	if select_mask != nil && len(select_mask) == int(n) {
		select_raw = make([]Blas_Int, n, context.temp_allocator)
		for i, selected in select_mask {
			select_raw[i] = 1 if selected else 0
		}
	} else {
		select_raw = make([]Blas_Int, n, context.temp_allocator)
		for i in 0 ..< n {
			select_raw[i] = 1
		}
	}

	ldh := Blas_Int(H.ld)
	ldvl := n if compute_left else 1
	ldvr := n if compute_right else 1

	// Allocate output matrices
	VL = matrix_zeros(f64, int(ldvl), int(n), allocator) if compute_left else Matrix(f64){}
	VR = matrix_zeros(f64, int(ldvr), int(n), allocator) if compute_right else Matrix(f64){}

	mm := n
	m_out: Blas_Int

	// Copy eigenvalue arrays
	WR := make([]f64, n, context.temp_allocator)
	WI := make([]f64, n, context.temp_allocator)
	copy(WR, eigenvalues_real)
	copy(WI, eigenvalues_imag)

	// Allocate workspace
	work := make([]f64, n * n, context.temp_allocator)

	ifaill := make([]Blas_Int, n, context.temp_allocator)
	ifailr := make([]Blas_Int, n, context.temp_allocator)

	// Call LAPACK
	info_val: Blas_Int
	vl_ptr := raw_data(VL.data) if compute_left else nil
	vr_ptr := raw_data(VR.data) if compute_right else nil

	lapack.dhsein_(
		&side_c,
		&eigsrc_c,
		&initv_c,
		raw_data(select_raw),
		&n,
		raw_data(H.data),
		&ldh,
		raw_data(WR),
		raw_data(WI),
		vl_ptr,
		&ldvl,
		vr_ptr,
		&ldvr,
		&mm,
		&m_out,
		raw_data(work),
		raw_data(ifaill),
		raw_data(ifailr),
		&info_val,
		1,
		1,
		1,
	)

	// Convert failure indices
	ifail = make([]int, n, allocator)
	for i in 0 ..< n {
		if compute_left && ifaill[i] != 0 {
			ifail[i] = int(ifaill[i])
		} else if compute_right && ifailr[i] != 0 {
			ifail[i] = int(ifailr[i])
		}
	}

	return VL, VR, int(m_out), ifail, info_val
}

// ===================================================================================
// HESSENBERG QR ALGORITHM FOR EIGENVALUE COMPUTATION
// ===================================================================================

// Compute eigenvalues and optionally Schur form of Hessenberg matrix using QR algorithm
m_eigenvalues_hessenberg_qr :: proc {
	m_eigenvalues_hessenberg_qr_c64,
	m_eigenvalues_hessenberg_qr_c128,
	m_eigenvalues_hessenberg_qr_f32,
	m_eigenvalues_hessenberg_qr_f64,
}

// QR algorithm for Hessenberg matrix eigenvalues (complex64)
m_eigenvalues_hessenberg_qr_c64 :: proc(
	H: ^Matrix(complex64),
	ilo: int = 0, // Submatrix range (0 means use full matrix)
	ihi: int = 0, // Submatrix range (0 means use full matrix)
	compute_schur: bool = true,
	compute_eigenvectors: bool = false,
	Z: ^Matrix(complex64) = nil, // Optional input/output matrix for eigenvectors
	allocator := context.allocator,
) -> (
	eigenvalues: []complex64,
	Z_out: Matrix(complex64),
	success: bool,
	info: Info,
) {
	// Validate matrix
	if H.format != .General {
		panic("H matrix must be in general format")
	}

	n := Blas_Int(H.rows)
	if H.rows != H.cols {
		panic("H matrix must be square")
	}

	// Setup range
	ilo_val := Blas_Int(ilo if ilo > 0 else 1)
	ihi_val := Blas_Int(ihi if ihi > 0 else int(n))

	// Setup job parameters
	job_c := cstring("S") if compute_schur else cstring("E") // Schur form or eigenvalues only
	compz_c := cstring("V") if compute_eigenvectors else cstring("N") // Compute or don't compute Z

	ldh := Blas_Int(H.ld)

	// Setup Z matrix
	ldz: Blas_Int
	if compute_eigenvectors {
		if Z != nil {
			if Z.rows != H.rows || Z.cols != H.cols {
				panic("Z matrix dimensions must match H matrix")
			}
			Z_out = Z^
			ldz = Blas_Int(Z.ld)
		} else {
			Z_out = matrix_zeros(complex64, int(n), int(n), allocator)
			ldz = n
		}
	} else {
		Z_out = Matrix(complex64){}
		ldz = 1
	}

	// Allocate eigenvalue array
	W := make([]complex64, n, allocator)

	// Workspace query
	work_query: complex64
	lwork_query := Blas_Int(-1)
	info_val: Blas_Int

	z_ptr := raw_data(Z_out.data) if compute_eigenvectors else nil

	lapack.chseqr_(
		&job_c,
		&compz_c,
		&n,
		&ilo_val,
		&ihi_val,
		raw_data(H.data),
		&ldh,
		raw_data(W),
		z_ptr,
		&ldz,
		&work_query,
		&lwork_query,
		&info_val,
		1,
		1,
	)

	if info_val != 0 {
		return nil, Z_out, false, info_val
	}

	// Allocate workspace
	lwork := Blas_Int(real(work_query))
	work := make([]complex64, lwork, context.temp_allocator)

	// Compute eigenvalues
	lapack.chseqr_(
		&job_c,
		&compz_c,
		&n,
		&ilo_val,
		&ihi_val,
		raw_data(H.data),
		&ldh,
		raw_data(W),
		z_ptr,
		&ldz,
		raw_data(work),
		&lwork,
		&info_val,
		1,
		1,
	)

	eigenvalues = W
	return eigenvalues, Z_out, info_val == 0, info_val
}

// ===================================================================================
// LAPACK UTILITY FUNCTIONS
// ===================================================================================

// Complex conjugate of a vector
m_conjugate_vector :: proc {
	m_conjugate_vector_c64,
	m_conjugate_vector_c128,
}

// Conjugate complex vector (complex64)
m_conjugate_vector_c64 :: proc(X: ^Vector(complex64)) {
	n := Blas_Int(len(X.data))
	incx := Blas_Int(X.incr)

	lapack.clacgv_(&n, raw_data(X.data), &incx)
}

// Conjugate complex vector (complex128)
m_conjugate_vector_c128 :: proc(X: ^Vector(complex128)) {
	n := Blas_Int(len(X.data))
	incx := Blas_Int(X.incr)

	lapack.zlacgv_(&n, raw_data(X.data), &incx)
}

// Estimate the 1-norm of a matrix using iterative method
m_estimate_1norm :: proc {
	m_estimate_1norm_c64,
	m_estimate_1norm_c128,
	m_estimate_1norm_f32,
	m_estimate_1norm_f64,
}

// Estimate 1-norm of matrix (complex64)
m_estimate_1norm_c64 :: proc(
	n: int,
	apply_matrix: proc(X: []complex64, Y: []complex64), // User-provided matrix application
	allocator := context.allocator,
) -> (
	estimate: f32,
	iterations: int,
) {
	n_blas := Blas_Int(n)

	// Allocate workspace
	V := make([]complex64, n, context.temp_allocator)
	X := make([]complex64, n, context.temp_allocator)

	est: f32
	kase: Blas_Int
	isave := make([]Blas_Int, 3, context.temp_allocator)

	iterations = 0

	// Iterative estimation loop
	for {
		lapack.clacn2_(&n_blas, raw_data(V), raw_data(X), &est, &kase, raw_data(isave))

		if kase == 0 {
			break
		}

		iterations += 1

		if kase == 1 {
			// Apply matrix: Y := A * X
			apply_matrix(X, V)
		} else {
			// Apply conjugate transpose: Y := A^H * X
			apply_matrix(X, V) // User should handle transpose internally
		}
	}

	return est, iterations
}

// Estimate 1-norm of matrix (complex128)
m_estimate_1norm_c128 :: proc(
	n: int,
	apply_matrix: proc(X: []complex128, Y: []complex128),
	allocator := context.allocator,
) -> (
	estimate: f64,
	iterations: int,
) {
	n_blas := Blas_Int(n)

	// Allocate workspace
	V := make([]complex128, n, context.temp_allocator)
	X := make([]complex128, n, context.temp_allocator)

	est: f64
	kase: Blas_Int
	isave := make([]Blas_Int, 3, context.temp_allocator)

	iterations = 0

	// Iterative estimation loop
	for {
		lapack.zlacn2_(&n_blas, raw_data(V), raw_data(X), &est, &kase, raw_data(isave))

		if kase == 0 {
			break
		}

		iterations += 1

		if kase == 1 {
			// Apply matrix: Y := A * X
			apply_matrix(X, V)
		} else {
			// Apply conjugate transpose: Y := A^H * X
			apply_matrix(X, V) // User should handle transpose internally
		}
	}

	return est, iterations
}

// Estimate 1-norm of real matrix (f32)
m_estimate_1norm_f32 :: proc(
	n: int,
	apply_matrix: proc(X: []f32, Y: []f32),
	allocator := context.allocator,
) -> (
	estimate: f32,
	iterations: int,
) {
	n_blas := Blas_Int(n)

	// Allocate workspace
	V := make([]f32, n, context.temp_allocator)
	X := make([]f32, n, context.temp_allocator)
	ISGN := make([]Blas_Int, n, context.temp_allocator)

	est: f32
	kase: Blas_Int
	isave := make([]Blas_Int, 3, context.temp_allocator)

	iterations = 0

	// Iterative estimation loop
	for {
		lapack.slacn2_(
			&n_blas,
			raw_data(V),
			raw_data(X),
			raw_data(ISGN),
			&est,
			&kase,
			raw_data(isave),
		)

		if kase == 0 {
			break
		}

		iterations += 1

		if kase == 1 {
			// Apply matrix: Y := A * X
			apply_matrix(X, V)
		} else {
			// Apply transpose: Y := A^T * X
			apply_matrix(X, V) // User should handle transpose internally
		}
	}

	return est, iterations
}

// ===================================================================================
// LAPACK MATRIX COPY AND CONVERSION UTILITIES
// ===================================================================================

// Copy real matrix to complex matrix with zero imaginary parts
m_copy_real_to_complex :: proc {
	m_copy_real_to_complex_f32_c64,
	m_copy_real_to_complex_f64_c128,
}

// Copy real matrix to complex matrix (f32 -> complex64)
m_copy_real_to_complex_f32_c64 :: proc(
	A: ^Matrix(f32),
	B: ^Matrix(complex64),
	upper: bool = false, // Copy upper triangle only
	lower: bool = false, // Copy lower triangle only
) {
	// Validate matrices
	if A.format != .General {
		panic("A matrix must be in general format")
	}
	if B.format != .General {
		panic("B matrix must be in general format")
	}
	if A.rows != B.rows || A.cols != B.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Determine uplo parameter
	uplo := cstring("A") // All matrix
	if upper && !lower {
		uplo = cstring("U") // Upper triangle
	} else if lower && !upper {
		uplo = cstring("L") // Lower triangle
	}

	lapack.clacp2_(&uplo, &m, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, 1)
}

// Copy real matrix to complex matrix (f64 -> complex128)
m_copy_real_to_complex_f64_c128 :: proc(
	A: ^Matrix(f64),
	B: ^Matrix(complex128),
	upper: bool = false,
	lower: bool = false,
) {
	// Validate matrices
	if A.format != .General {
		panic("A matrix must be in general format")
	}
	if B.format != .General {
		panic("B matrix must be in general format")
	}
	if A.rows != B.rows || A.cols != B.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Determine uplo parameter
	uplo := cstring("A") // All matrix
	if upper && !lower {
		uplo = cstring("U") // Upper triangle
	} else if lower && !upper {
		uplo = cstring("L") // Lower triangle
	}

	lapack.zlacp2_(&uplo, &m, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, 1)
}

// Copy matrix with optional triangular selection
m_copy_matrix :: proc {
	m_copy_matrix_c64,
	m_copy_matrix_c128,
	m_copy_matrix_f32,
	m_copy_matrix_f64,
}

// Copy matrix (complex64)
m_copy_matrix_c64 :: proc(
	A: ^Matrix(complex64),
	B: ^Matrix(complex64),
	upper: bool = false, // Copy upper triangle only
	lower: bool = false, // Copy lower triangle only
) {
	// Validate matrices
	if A.format != .General {
		panic("A matrix must be in general format")
	}
	if B.format != .General {
		panic("B matrix must be in general format")
	}
	if A.rows != B.rows || A.cols != B.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Determine uplo parameter
	uplo := cstring("A") // All matrix
	if upper && !lower {
		uplo = cstring("U") // Upper triangle
	} else if lower && !upper {
		uplo = cstring("L") // Lower triangle
	}

	lapack.clacpy_(&uplo, &m, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, 1)
}

// Copy matrix (complex128)
m_copy_matrix_c128 :: proc(
	A: ^Matrix(complex128),
	B: ^Matrix(complex128),
	upper: bool = false,
	lower: bool = false,
) {
	// Validate matrices
	if A.format != .General {
		panic("A matrix must be in general format")
	}
	if B.format != .General {
		panic("B matrix must be in general format")
	}
	if A.rows != B.rows || A.cols != B.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Determine uplo parameter
	uplo := cstring("A") // All matrix
	if upper && !lower {
		uplo = cstring("U") // Upper triangle
	} else if lower && !upper {
		uplo = cstring("L") // Lower triangle
	}

	lapack.zlacpy_(&uplo, &m, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, 1)
}

// Copy matrix (f32)
m_copy_matrix_f32 :: proc(
	A: ^Matrix(f32),
	B: ^Matrix(f32),
	upper: bool = false,
	lower: bool = false,
) {
	// Validate matrices
	if A.format != .General {
		panic("A matrix must be in general format")
	}
	if B.format != .General {
		panic("B matrix must be in general format")
	}
	if A.rows != B.rows || A.cols != B.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Determine uplo parameter
	uplo := cstring("A") // All matrix
	if upper && !lower {
		uplo = cstring("U") // Upper triangle
	} else if lower && !upper {
		uplo = cstring("L") // Lower triangle
	}

	lapack.slacpy_(&uplo, &m, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, 1)
}

// Copy matrix (f64)
m_copy_matrix_f64 :: proc(
	A: ^Matrix(f64),
	B: ^Matrix(f64),
	upper: bool = false,
	lower: bool = false,
) {
	// Validate matrices
	if A.format != .General {
		panic("A matrix must be in general format")
	}
	if B.format != .General {
		panic("B matrix must be in general format")
	}
	if A.rows != B.rows || A.cols != B.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Determine uplo parameter
	uplo := cstring("A") // All matrix
	if upper && !lower {
		uplo = cstring("U") // Upper triangle
	} else if lower && !upper {
		uplo = cstring("L") // Lower triangle
	}

	lapack.dlacpy_(&uplo, &m, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, 1)
}

// Complex matrix-real matrix multiplication: C := A * B where A is complex, B is real
m_multiply_complex_real :: proc {
	m_multiply_complex_real_c64_f32,
	m_multiply_complex_real_c128_f64,
}

// Complex-real matrix multiplication (complex64 * f32)
m_multiply_complex_real_c64_f32 :: proc(
	A: ^Matrix(complex64),
	B: ^Matrix(f32),
	C: ^Matrix(complex64),
	allocator := context.allocator,
) {
	// Validate matrices
	if A.format != .General || B.format != .General || C.format != .General {
		panic("All matrices must be in general format")
	}
	if A.cols != B.rows {
		panic("Matrix dimensions incompatible for multiplication")
	}
	if A.rows != C.rows || B.cols != C.cols {
		panic("Output matrix dimensions incorrect")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(B.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)
	ldc := Blas_Int(C.ld)

	// Allocate workspace for real components
	rwork_size := max(1, int(m * n))
	rwork := make([]f32, rwork_size, context.temp_allocator)

	lapack.clacrm_(
		&m,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(C.data),
		&ldc,
		raw_data(rwork),
	)
}

// ===================================================================================
// LAPACK PRECISION CONVERSION UTILITIES
// ===================================================================================

// Convert between different numerical precisions with overflow checking
m_convert_precision :: proc {
	m_convert_precision_c128_c64,
	m_convert_precision_f32_f64,
	m_convert_precision_f64_f32,
	m_convert_precision_c64_c128,
}

// Convert complex128 matrix to complex64 with overflow checking
m_convert_precision_c128_c64 :: proc(
	A: ^Matrix(complex128),
	SA: ^Matrix(complex64),
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrices
	if A.format != .General || SA.format != .General {
		panic("Both matrices must be in general format")
	}
	if A.rows != SA.rows || A.cols != SA.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldsa := Blas_Int(SA.ld)

	info_val: Blas_Int
	lapack.zlag2c_(&m, &n, raw_data(A.data), &lda, raw_data(SA.data), &ldsa, &info_val)

	return info_val == 0, info_val
}

// Convert f32 matrix to f64 (always successful)
m_convert_precision_f32_f64 :: proc(
	SA: ^Matrix(f32),
	A: ^Matrix(f64),
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrices
	if SA.format != .General || A.format != .General {
		panic("Both matrices must be in general format")
	}
	if SA.rows != A.rows || SA.cols != A.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(SA.rows)
	n := Blas_Int(SA.cols)
	ldsa := Blas_Int(SA.ld)
	lda := Blas_Int(A.ld)

	info_val: Blas_Int
	lapack.slag2d_(&m, &n, raw_data(SA.data), &ldsa, raw_data(A.data), &lda, &info_val)

	return info_val == 0, info_val
}

// Convert f64 matrix to f32 with overflow checking
m_convert_precision_f64_f32 :: proc(
	A: ^Matrix(f64),
	SA: ^Matrix(f32),
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrices
	if A.format != .General || SA.format != .General {
		panic("Both matrices must be in general format")
	}
	if A.rows != SA.rows || A.cols != SA.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldsa := Blas_Int(SA.ld)

	info_val: Blas_Int
	lapack.dlag2s_(&m, &n, raw_data(A.data), &lda, raw_data(SA.data), &ldsa, &info_val)

	return info_val == 0, info_val
}

// Convert complex64 matrix to complex128 (always successful)
m_convert_precision_c64_c128 :: proc(
	SA: ^Matrix(complex64),
	A: ^Matrix(complex128),
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrices
	if SA.format != .General || A.format != .General {
		panic("Both matrices must be in general format")
	}
	if SA.rows != A.rows || SA.cols != A.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(SA.rows)
	n := Blas_Int(SA.cols)
	ldsa := Blas_Int(SA.ld)
	lda := Blas_Int(A.ld)

	info_val: Blas_Int
	lapack.clag2z_(&m, &n, raw_data(SA.data), &ldsa, raw_data(A.data), &lda, &info_val)

	return info_val == 0, info_val
}

// ===================================================================================
// LAPACK RANDOM MATRIX GENERATION UTILITIES
// ===================================================================================

// Generate random general banded matrix
m_generate_random_banded :: proc {
	m_generate_random_banded_c64,
	m_generate_random_banded_c128,
	m_generate_random_banded_f32,
	m_generate_random_banded_f64,
}

// Generate random banded matrix (complex64)
m_generate_random_banded_c64 :: proc(
	A: ^Matrix(complex64),
	kl, ku: int, // Lower and upper bandwidth
	D: []f32, // Diagonal scaling factors
	seed: []int = nil, // Random seed (4 elements), auto-generated if nil
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrix
	if A.format != .General {
		panic("Matrix must be in general format")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix rows")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	kl_val := Blas_Int(kl)
	ku_val := Blas_Int(ku)
	lda := Blas_Int(A.ld)

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		// Generate default seed
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work_size := max(m, n)
	work := make([]complex64, work_size, context.temp_allocator)

	info_val: Blas_Int
	lapack.clagge_(
		&m,
		&n,
		&kl_val,
		&ku_val,
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}

// Generate random banded matrix (complex128)
m_generate_random_banded_c128 :: proc(
	A: ^Matrix(complex128),
	kl, ku: int,
	D: []f64,
	seed: []int = nil,
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrix
	if A.format != .General {
		panic("Matrix must be in general format")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix rows")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	kl_val := Blas_Int(kl)
	ku_val := Blas_Int(ku)
	lda := Blas_Int(A.ld)

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work_size := max(m, n)
	work := make([]complex128, work_size, context.temp_allocator)

	info_val: Blas_Int
	lapack.zlagge_(
		&m,
		&n,
		&kl_val,
		&ku_val,
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}

// Generate random banded matrix (f32)
m_generate_random_banded_f32 :: proc(
	A: ^Matrix(f32),
	kl, ku: int,
	D: []f32,
	seed: []int = nil,
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrix
	if A.format != .General {
		panic("Matrix must be in general format")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix rows")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	kl_val := Blas_Int(kl)
	ku_val := Blas_Int(ku)
	lda := Blas_Int(A.ld)

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work_size := max(m, n)
	work := make([]f32, work_size, context.temp_allocator)

	info_val: Blas_Int
	lapack.slagge_(
		&m,
		&n,
		&kl_val,
		&ku_val,
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}

// Generate random banded matrix (f64)
m_generate_random_banded_f64 :: proc(
	A: ^Matrix(f64),
	kl, ku: int,
	D: []f64,
	seed: []int = nil,
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrix
	if A.format != .General {
		panic("Matrix must be in general format")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix rows")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	kl_val := Blas_Int(kl)
	ku_val := Blas_Int(ku)
	lda := Blas_Int(A.ld)

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work_size := max(m, n)
	work := make([]f64, work_size, context.temp_allocator)

	info_val: Blas_Int
	lapack.dlagge_(
		&m,
		&n,
		&kl_val,
		&ku_val,
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}

// Complex-real matrix multiplication (complex128 * f64)
m_multiply_complex_real_c128_f64 :: proc(
	A: ^Matrix(complex128),
	B: ^Matrix(f64),
	C: ^Matrix(complex128),
	allocator := context.allocator,
) {
	// Validate matrices
	if A.format != .General || B.format != .General || C.format != .General {
		panic("All matrices must be in general format")
	}
	if A.cols != B.rows {
		panic("Matrix dimensions incompatible for multiplication")
	}
	if A.rows != C.rows || B.cols != C.cols {
		panic("Output matrix dimensions incorrect")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(B.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)
	ldc := Blas_Int(C.ld)

	// Allocate workspace for real components
	rwork_size := max(1, int(m * n))
	rwork := make([]f64, rwork_size, context.temp_allocator)

	lapack.zlacrm_(
		&m,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(C.data),
		&ldc,
		raw_data(rwork),
	)
}

// ===================================================================================
// LAPACK PRECISION CONVERSION UTILITIES
// ===================================================================================

// Convert between different numerical precisions with overflow checking
m_convert_precision :: proc {
	m_convert_precision_c128_c64,
	m_convert_precision_f32_f64,
	m_convert_precision_f64_f32,
	m_convert_precision_c64_c128,
}

// Convert complex128 matrix to complex64 with overflow checking
m_convert_precision_c128_c64 :: proc(
	A: ^Matrix(complex128),
	SA: ^Matrix(complex64),
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrices
	if A.format != .General || SA.format != .General {
		panic("Both matrices must be in general format")
	}
	if A.rows != SA.rows || A.cols != SA.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldsa := Blas_Int(SA.ld)

	info_val: Blas_Int
	lapack.zlag2c_(&m, &n, raw_data(A.data), &lda, raw_data(SA.data), &ldsa, &info_val)

	return info_val == 0, info_val
}

// Convert f32 matrix to f64 (always successful)
m_convert_precision_f32_f64 :: proc(
	SA: ^Matrix(f32),
	A: ^Matrix(f64),
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrices
	if SA.format != .General || A.format != .General {
		panic("Both matrices must be in general format")
	}
	if SA.rows != A.rows || SA.cols != A.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(SA.rows)
	n := Blas_Int(SA.cols)
	ldsa := Blas_Int(SA.ld)
	lda := Blas_Int(A.ld)

	info_val: Blas_Int
	lapack.slag2d_(&m, &n, raw_data(SA.data), &ldsa, raw_data(A.data), &lda, &info_val)

	return info_val == 0, info_val
}

// Convert f64 matrix to f32 with overflow checking
m_convert_precision_f64_f32 :: proc(
	A: ^Matrix(f64),
	SA: ^Matrix(f32),
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrices
	if A.format != .General || SA.format != .General {
		panic("Both matrices must be in general format")
	}
	if A.rows != SA.rows || A.cols != SA.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldsa := Blas_Int(SA.ld)

	info_val: Blas_Int
	lapack.dlag2s_(&m, &n, raw_data(A.data), &lda, raw_data(SA.data), &ldsa, &info_val)

	return info_val == 0, info_val
}

// Convert complex64 matrix to complex128 (always successful)
m_convert_precision_c64_c128 :: proc(
	SA: ^Matrix(complex64),
	A: ^Matrix(complex128),
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrices
	if SA.format != .General || A.format != .General {
		panic("Both matrices must be in general format")
	}
	if SA.rows != A.rows || SA.cols != A.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(SA.rows)
	n := Blas_Int(SA.cols)
	ldsa := Blas_Int(SA.ld)
	lda := Blas_Int(A.ld)

	info_val: Blas_Int
	lapack.clag2z_(&m, &n, raw_data(SA.data), &ldsa, raw_data(A.data), &lda, &info_val)

	return info_val == 0, info_val
}

// ===================================================================================
// LAPACK RANDOM MATRIX GENERATION UTILITIES
// ===================================================================================

// Generate random general banded matrix
m_generate_random_banded :: proc {
	m_generate_random_banded_c64,
	m_generate_random_banded_c128,
	m_generate_random_banded_f32,
	m_generate_random_banded_f64,
}

// Generate random banded matrix (complex64)
m_generate_random_banded_c64 :: proc(
	A: ^Matrix(complex64),
	kl, ku: int, // Lower and upper bandwidth
	D: []f32, // Diagonal scaling factors
	seed: []int = nil, // Random seed (4 elements), auto-generated if nil
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrix
	if A.format != .General {
		panic("Matrix must be in general format")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix rows")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	kl_val := Blas_Int(kl)
	ku_val := Blas_Int(ku)
	lda := Blas_Int(A.ld)

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		// Generate default seed
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work_size := max(m, n)
	work := make([]complex64, work_size, context.temp_allocator)

	info_val: Blas_Int
	lapack.clagge_(
		&m,
		&n,
		&kl_val,
		&ku_val,
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}

// Generate random banded matrix (complex128)
m_generate_random_banded_c128 :: proc(
	A: ^Matrix(complex128),
	kl, ku: int,
	D: []f64,
	seed: []int = nil,
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrix
	if A.format != .General {
		panic("Matrix must be in general format")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix rows")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	kl_val := Blas_Int(kl)
	ku_val := Blas_Int(ku)
	lda := Blas_Int(A.ld)

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work_size := max(m, n)
	work := make([]complex128, work_size, context.temp_allocator)

	info_val: Blas_Int
	lapack.zlagge_(
		&m,
		&n,
		&kl_val,
		&ku_val,
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}

// Generate random banded matrix (f32)
m_generate_random_banded_f32 :: proc(
	A: ^Matrix(f32),
	kl, ku: int,
	D: []f32,
	seed: []int = nil,
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrix
	if A.format != .General {
		panic("Matrix must be in general format")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix rows")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	kl_val := Blas_Int(kl)
	ku_val := Blas_Int(ku)
	lda := Blas_Int(A.ld)

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work_size := max(m, n)
	work := make([]f32, work_size, context.temp_allocator)

	info_val: Blas_Int
	lapack.slagge_(
		&m,
		&n,
		&kl_val,
		&ku_val,
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}

// Generate random banded matrix (f64)
m_generate_random_banded_f64 :: proc(
	A: ^Matrix(f64),
	kl, ku: int,
	D: []f64,
	seed: []int = nil,
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrix
	if A.format != .General {
		panic("Matrix must be in general format")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix rows")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	kl_val := Blas_Int(kl)
	ku_val := Blas_Int(ku)
	lda := Blas_Int(A.ld)

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work_size := max(m, n)
	work := make([]f64, work_size, context.temp_allocator)

	info_val: Blas_Int
	lapack.dlagge_(
		&m,
		&n,
		&kl_val,
		&ku_val,
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}

// Estimate 1-norm of real matrix (f64)
m_estimate_1norm_f64 :: proc(
	n: int,
	apply_matrix: proc(X: []f64, Y: []f64),
	allocator := context.allocator,
) -> (
	estimate: f64,
	iterations: int,
) {
	n_blas := Blas_Int(n)

	// Allocate workspace
	V := make([]f64, n, context.temp_allocator)
	X := make([]f64, n, context.temp_allocator)
	ISGN := make([]Blas_Int, n, context.temp_allocator)

	est: f64
	kase: Blas_Int
	isave := make([]Blas_Int, 3, context.temp_allocator)

	iterations = 0

	// Iterative estimation loop
	for {
		lapack.dlacn2_(
			&n_blas,
			raw_data(V),
			raw_data(X),
			raw_data(ISGN),
			&est,
			&kase,
			raw_data(isave),
		)

		if kase == 0 {
			break
		}

		iterations += 1

		if kase == 1 {
			// Apply matrix: Y := A * X
			apply_matrix(X, V)
		} else {
			// Apply transpose: Y := A^T * X
			apply_matrix(X, V) // User should handle transpose internally
		}
	}

	return est, iterations
}

// ===================================================================================
// LAPACK MATRIX COPY AND CONVERSION UTILITIES
// ===================================================================================

// Copy real matrix to complex matrix with zero imaginary parts
m_copy_real_to_complex :: proc {
	m_copy_real_to_complex_f32_c64,
	m_copy_real_to_complex_f64_c128,
}

// Copy real matrix to complex matrix (f32 -> complex64)
m_copy_real_to_complex_f32_c64 :: proc(
	A: ^Matrix(f32),
	B: ^Matrix(complex64),
	upper: bool = false, // Copy upper triangle only
	lower: bool = false, // Copy lower triangle only
) {
	// Validate matrices
	if A.format != .General {
		panic("A matrix must be in general format")
	}
	if B.format != .General {
		panic("B matrix must be in general format")
	}
	if A.rows != B.rows || A.cols != B.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Determine uplo parameter
	uplo := cstring("A") // All matrix
	if upper && !lower {
		uplo = cstring("U") // Upper triangle
	} else if lower && !upper {
		uplo = cstring("L") // Lower triangle
	}

	lapack.clacp2_(&uplo, &m, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, 1)
}

// Copy real matrix to complex matrix (f64 -> complex128)
m_copy_real_to_complex_f64_c128 :: proc(
	A: ^Matrix(f64),
	B: ^Matrix(complex128),
	upper: bool = false,
	lower: bool = false,
) {
	// Validate matrices
	if A.format != .General {
		panic("A matrix must be in general format")
	}
	if B.format != .General {
		panic("B matrix must be in general format")
	}
	if A.rows != B.rows || A.cols != B.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Determine uplo parameter
	uplo := cstring("A") // All matrix
	if upper && !lower {
		uplo = cstring("U") // Upper triangle
	} else if lower && !upper {
		uplo = cstring("L") // Lower triangle
	}

	lapack.zlacp2_(&uplo, &m, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, 1)
}

// Copy matrix with optional triangular selection
m_copy_matrix :: proc {
	m_copy_matrix_c64,
	m_copy_matrix_c128,
	m_copy_matrix_f32,
	m_copy_matrix_f64,
}

// Copy matrix (complex64)
m_copy_matrix_c64 :: proc(
	A: ^Matrix(complex64),
	B: ^Matrix(complex64),
	upper: bool = false, // Copy upper triangle only
	lower: bool = false, // Copy lower triangle only
) {
	// Validate matrices
	if A.format != .General {
		panic("A matrix must be in general format")
	}
	if B.format != .General {
		panic("B matrix must be in general format")
	}
	if A.rows != B.rows || A.cols != B.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Determine uplo parameter
	uplo := cstring("A") // All matrix
	if upper && !lower {
		uplo = cstring("U") // Upper triangle
	} else if lower && !upper {
		uplo = cstring("L") // Lower triangle
	}

	lapack.clacpy_(&uplo, &m, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, 1)
}

// Copy matrix (complex128)
m_copy_matrix_c128 :: proc(
	A: ^Matrix(complex128),
	B: ^Matrix(complex128),
	upper: bool = false,
	lower: bool = false,
) {
	// Validate matrices
	if A.format != .General {
		panic("A matrix must be in general format")
	}
	if B.format != .General {
		panic("B matrix must be in general format")
	}
	if A.rows != B.rows || A.cols != B.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Determine uplo parameter
	uplo := cstring("A") // All matrix
	if upper && !lower {
		uplo = cstring("U") // Upper triangle
	} else if lower && !upper {
		uplo = cstring("L") // Lower triangle
	}

	lapack.zlacpy_(&uplo, &m, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, 1)
}

// Copy matrix (f32)
m_copy_matrix_f32 :: proc(
	A: ^Matrix(f32),
	B: ^Matrix(f32),
	upper: bool = false,
	lower: bool = false,
) {
	// Validate matrices
	if A.format != .General {
		panic("A matrix must be in general format")
	}
	if B.format != .General {
		panic("B matrix must be in general format")
	}
	if A.rows != B.rows || A.cols != B.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Determine uplo parameter
	uplo := cstring("A") // All matrix
	if upper && !lower {
		uplo = cstring("U") // Upper triangle
	} else if lower && !upper {
		uplo = cstring("L") // Lower triangle
	}

	lapack.slacpy_(&uplo, &m, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, 1)
}

// Copy matrix (f64)
m_copy_matrix_f64 :: proc(
	A: ^Matrix(f64),
	B: ^Matrix(f64),
	upper: bool = false,
	lower: bool = false,
) {
	// Validate matrices
	if A.format != .General {
		panic("A matrix must be in general format")
	}
	if B.format != .General {
		panic("B matrix must be in general format")
	}
	if A.rows != B.rows || A.cols != B.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)

	// Determine uplo parameter
	uplo := cstring("A") // All matrix
	if upper && !lower {
		uplo = cstring("U") // Upper triangle
	} else if lower && !upper {
		uplo = cstring("L") // Lower triangle
	}

	lapack.dlacpy_(&uplo, &m, &n, raw_data(A.data), &lda, raw_data(B.data), &ldb, 1)
}

// Complex matrix-real matrix multiplication: C := A * B where A is complex, B is real
m_multiply_complex_real :: proc {
	m_multiply_complex_real_c64_f32,
	m_multiply_complex_real_c128_f64,
}

// Complex-real matrix multiplication (complex64 * f32)
m_multiply_complex_real_c64_f32 :: proc(
	A: ^Matrix(complex64),
	B: ^Matrix(f32),
	C: ^Matrix(complex64),
	allocator := context.allocator,
) {
	// Validate matrices
	if A.format != .General || B.format != .General || C.format != .General {
		panic("All matrices must be in general format")
	}
	if A.cols != B.rows {
		panic("Matrix dimensions incompatible for multiplication")
	}
	if A.rows != C.rows || B.cols != C.cols {
		panic("Output matrix dimensions incorrect")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(B.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)
	ldc := Blas_Int(C.ld)

	// Allocate workspace for real components
	rwork_size := max(1, int(m * n))
	rwork := make([]f32, rwork_size, context.temp_allocator)

	lapack.clacrm_(
		&m,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(C.data),
		&ldc,
		raw_data(rwork),
	)
}

// ===================================================================================
// LAPACK PRECISION CONVERSION UTILITIES
// ===================================================================================

// Convert between different numerical precisions with overflow checking
m_convert_precision :: proc {
	m_convert_precision_c128_c64,
	m_convert_precision_f32_f64,
	m_convert_precision_f64_f32,
	m_convert_precision_c64_c128,
}

// Convert complex128 matrix to complex64 with overflow checking
m_convert_precision_c128_c64 :: proc(
	A: ^Matrix(complex128),
	SA: ^Matrix(complex64),
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrices
	if A.format != .General || SA.format != .General {
		panic("Both matrices must be in general format")
	}
	if A.rows != SA.rows || A.cols != SA.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldsa := Blas_Int(SA.ld)

	info_val: Blas_Int
	lapack.zlag2c_(&m, &n, raw_data(A.data), &lda, raw_data(SA.data), &ldsa, &info_val)

	return info_val == 0, info_val
}

// Convert f32 matrix to f64 (always successful)
m_convert_precision_f32_f64 :: proc(
	SA: ^Matrix(f32),
	A: ^Matrix(f64),
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrices
	if SA.format != .General || A.format != .General {
		panic("Both matrices must be in general format")
	}
	if SA.rows != A.rows || SA.cols != A.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(SA.rows)
	n := Blas_Int(SA.cols)
	ldsa := Blas_Int(SA.ld)
	lda := Blas_Int(A.ld)

	info_val: Blas_Int
	lapack.slag2d_(&m, &n, raw_data(SA.data), &ldsa, raw_data(A.data), &lda, &info_val)

	return info_val == 0, info_val
}

// Convert f64 matrix to f32 with overflow checking
m_convert_precision_f64_f32 :: proc(
	A: ^Matrix(f64),
	SA: ^Matrix(f32),
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrices
	if A.format != .General || SA.format != .General {
		panic("Both matrices must be in general format")
	}
	if A.rows != SA.rows || A.cols != SA.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldsa := Blas_Int(SA.ld)

	info_val: Blas_Int
	lapack.dlag2s_(&m, &n, raw_data(A.data), &lda, raw_data(SA.data), &ldsa, &info_val)

	return info_val == 0, info_val
}

// Convert complex64 matrix to complex128 (always successful)
m_convert_precision_c64_c128 :: proc(
	SA: ^Matrix(complex64),
	A: ^Matrix(complex128),
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrices
	if SA.format != .General || A.format != .General {
		panic("Both matrices must be in general format")
	}
	if SA.rows != A.rows || SA.cols != A.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(SA.rows)
	n := Blas_Int(SA.cols)
	ldsa := Blas_Int(SA.ld)
	lda := Blas_Int(A.ld)

	info_val: Blas_Int
	lapack.clag2z_(&m, &n, raw_data(SA.data), &ldsa, raw_data(A.data), &lda, &info_val)

	return info_val == 0, info_val
}

// ===================================================================================
// LAPACK RANDOM MATRIX GENERATION UTILITIES
// ===================================================================================

// Generate random general banded matrix
m_generate_random_banded :: proc {
	m_generate_random_banded_c64,
	m_generate_random_banded_c128,
	m_generate_random_banded_f32,
	m_generate_random_banded_f64,
}

// Generate random banded matrix (complex64)
m_generate_random_banded_c64 :: proc(
	A: ^Matrix(complex64),
	kl, ku: int, // Lower and upper bandwidth
	D: []f32, // Diagonal scaling factors
	seed: []int = nil, // Random seed (4 elements), auto-generated if nil
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrix
	if A.format != .General {
		panic("Matrix must be in general format")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix rows")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	kl_val := Blas_Int(kl)
	ku_val := Blas_Int(ku)
	lda := Blas_Int(A.ld)

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		// Generate default seed
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work_size := max(m, n)
	work := make([]complex64, work_size, context.temp_allocator)

	info_val: Blas_Int
	lapack.clagge_(
		&m,
		&n,
		&kl_val,
		&ku_val,
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}

// Generate random banded matrix (complex128)
m_generate_random_banded_c128 :: proc(
	A: ^Matrix(complex128),
	kl, ku: int,
	D: []f64,
	seed: []int = nil,
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrix
	if A.format != .General {
		panic("Matrix must be in general format")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix rows")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	kl_val := Blas_Int(kl)
	ku_val := Blas_Int(ku)
	lda := Blas_Int(A.ld)

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work_size := max(m, n)
	work := make([]complex128, work_size, context.temp_allocator)

	info_val: Blas_Int
	lapack.zlagge_(
		&m,
		&n,
		&kl_val,
		&ku_val,
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}

// Generate random banded matrix (f32)
m_generate_random_banded_f32 :: proc(
	A: ^Matrix(f32),
	kl, ku: int,
	D: []f32,
	seed: []int = nil,
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrix
	if A.format != .General {
		panic("Matrix must be in general format")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix rows")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	kl_val := Blas_Int(kl)
	ku_val := Blas_Int(ku)
	lda := Blas_Int(A.ld)

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work_size := max(m, n)
	work := make([]f32, work_size, context.temp_allocator)

	info_val: Blas_Int
	lapack.slagge_(
		&m,
		&n,
		&kl_val,
		&ku_val,
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}

// Generate random banded matrix (f64)
m_generate_random_banded_f64 :: proc(
	A: ^Matrix(f64),
	kl, ku: int,
	D: []f64,
	seed: []int = nil,
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrix
	if A.format != .General {
		panic("Matrix must be in general format")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix rows")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	kl_val := Blas_Int(kl)
	ku_val := Blas_Int(ku)
	lda := Blas_Int(A.ld)

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work_size := max(m, n)
	work := make([]f64, work_size, context.temp_allocator)

	info_val: Blas_Int
	lapack.dlagge_(
		&m,
		&n,
		&kl_val,
		&ku_val,
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}

// Complex-real matrix multiplication (complex128 * f64)
m_multiply_complex_real_c128_f64 :: proc(
	A: ^Matrix(complex128),
	B: ^Matrix(f64),
	C: ^Matrix(complex128),
	allocator := context.allocator,
) {
	// Validate matrices
	if A.format != .General || B.format != .General || C.format != .General {
		panic("All matrices must be in general format")
	}
	if A.cols != B.rows {
		panic("Matrix dimensions incompatible for multiplication")
	}
	if A.rows != C.rows || B.cols != C.cols {
		panic("Output matrix dimensions incorrect")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(B.cols)
	lda := Blas_Int(A.ld)
	ldb := Blas_Int(B.ld)
	ldc := Blas_Int(C.ld)

	// Allocate workspace for real components
	rwork_size := max(1, int(m * n))
	rwork := make([]f64, rwork_size, context.temp_allocator)

	lapack.zlacrm_(
		&m,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(C.data),
		&ldc,
		raw_data(rwork),
	)
}

// ===================================================================================
// LAPACK PRECISION CONVERSION UTILITIES
// ===================================================================================

// Convert between different numerical precisions with overflow checking
m_convert_precision :: proc {
	m_convert_precision_c128_c64,
	m_convert_precision_f32_f64,
	m_convert_precision_f64_f32,
	m_convert_precision_c64_c128,
}

// Convert complex128 matrix to complex64 with overflow checking
m_convert_precision_c128_c64 :: proc(
	A: ^Matrix(complex128),
	SA: ^Matrix(complex64),
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrices
	if A.format != .General || SA.format != .General {
		panic("Both matrices must be in general format")
	}
	if A.rows != SA.rows || A.cols != SA.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldsa := Blas_Int(SA.ld)

	info_val: Blas_Int
	lapack.zlag2c_(&m, &n, raw_data(A.data), &lda, raw_data(SA.data), &ldsa, &info_val)

	return info_val == 0, info_val
}

// Convert f32 matrix to f64 (always successful)
m_convert_precision_f32_f64 :: proc(
	SA: ^Matrix(f32),
	A: ^Matrix(f64),
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrices
	if SA.format != .General || A.format != .General {
		panic("Both matrices must be in general format")
	}
	if SA.rows != A.rows || SA.cols != A.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(SA.rows)
	n := Blas_Int(SA.cols)
	ldsa := Blas_Int(SA.ld)
	lda := Blas_Int(A.ld)

	info_val: Blas_Int
	lapack.slag2d_(&m, &n, raw_data(SA.data), &ldsa, raw_data(A.data), &lda, &info_val)

	return info_val == 0, info_val
}

// Convert f64 matrix to f32 with overflow checking
m_convert_precision_f64_f32 :: proc(
	A: ^Matrix(f64),
	SA: ^Matrix(f32),
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrices
	if A.format != .General || SA.format != .General {
		panic("Both matrices must be in general format")
	}
	if A.rows != SA.rows || A.cols != SA.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	ldsa := Blas_Int(SA.ld)

	info_val: Blas_Int
	lapack.dlag2s_(&m, &n, raw_data(A.data), &lda, raw_data(SA.data), &ldsa, &info_val)

	return info_val == 0, info_val
}

// Convert complex64 matrix to complex128 (always successful)
m_convert_precision_c64_c128 :: proc(
	SA: ^Matrix(complex64),
	A: ^Matrix(complex128),
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrices
	if SA.format != .General || A.format != .General {
		panic("Both matrices must be in general format")
	}
	if SA.rows != A.rows || SA.cols != A.cols {
		panic("Matrix dimensions must match")
	}

	m := Blas_Int(SA.rows)
	n := Blas_Int(SA.cols)
	ldsa := Blas_Int(SA.ld)
	lda := Blas_Int(A.ld)

	info_val: Blas_Int
	lapack.clag2z_(&m, &n, raw_data(SA.data), &ldsa, raw_data(A.data), &lda, &info_val)

	return info_val == 0, info_val
}

// ===================================================================================
// LAPACK RANDOM MATRIX GENERATION UTILITIES
// ===================================================================================

// Generate random general banded matrix
m_generate_random_banded :: proc {
	m_generate_random_banded_c64,
	m_generate_random_banded_c128,
	m_generate_random_banded_f32,
	m_generate_random_banded_f64,
}

// Generate random banded matrix (complex64)
m_generate_random_banded_c64 :: proc(
	A: ^Matrix(complex64),
	kl, ku: int, // Lower and upper bandwidth
	D: []f32, // Diagonal scaling factors
	seed: []int = nil, // Random seed (4 elements), auto-generated if nil
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrix
	if A.format != .General {
		panic("Matrix must be in general format")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix rows")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	kl_val := Blas_Int(kl)
	ku_val := Blas_Int(ku)
	lda := Blas_Int(A.ld)

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		// Generate default seed
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work_size := max(m, n)
	work := make([]complex64, work_size, context.temp_allocator)

	info_val: Blas_Int
	lapack.clagge_(
		&m,
		&n,
		&kl_val,
		&ku_val,
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}

// Generate random banded matrix (complex128)
m_generate_random_banded_c128 :: proc(
	A: ^Matrix(complex128),
	kl, ku: int,
	D: []f64,
	seed: []int = nil,
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrix
	if A.format != .General {
		panic("Matrix must be in general format")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix rows")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	kl_val := Blas_Int(kl)
	ku_val := Blas_Int(ku)
	lda := Blas_Int(A.ld)

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work_size := max(m, n)
	work := make([]complex128, work_size, context.temp_allocator)

	info_val: Blas_Int
	lapack.zlagge_(
		&m,
		&n,
		&kl_val,
		&ku_val,
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}

// Generate random banded matrix (f32)
m_generate_random_banded_f32 :: proc(
	A: ^Matrix(f32),
	kl, ku: int,
	D: []f32,
	seed: []int = nil,
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrix
	if A.format != .General {
		panic("Matrix must be in general format")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix rows")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	kl_val := Blas_Int(kl)
	ku_val := Blas_Int(ku)
	lda := Blas_Int(A.ld)

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work_size := max(m, n)
	work := make([]f32, work_size, context.temp_allocator)

	info_val: Blas_Int
	lapack.slagge_(
		&m,
		&n,
		&kl_val,
		&ku_val,
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}

// Generate random banded matrix (f64)
m_generate_random_banded_f64 :: proc(
	A: ^Matrix(f64),
	kl, ku: int,
	D: []f64,
	seed: []int = nil,
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrix
	if A.format != .General {
		panic("Matrix must be in general format")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix rows")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	kl_val := Blas_Int(kl)
	ku_val := Blas_Int(ku)
	lda := Blas_Int(A.ld)

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work_size := max(m, n)
	work := make([]f64, work_size, context.temp_allocator)

	info_val: Blas_Int
	lapack.dlagge_(
		&m,
		&n,
		&kl_val,
		&ku_val,
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}
