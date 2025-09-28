package openblas

import lapack "./f77"
import "core:c"
import "core:math"
import "core:mem"
import "core:slice"

// ============================================================================
// PACKED SYMMETRIC MATRIX OPERATIONS
// ============================================================================
// Operations on symmetric matrices stored in packed format (triangular storage)
// Only upper or lower triangle is stored in a 1D array of size n*(n+1)/2

// ============================================================================
// CONDITION NUMBER ESTIMATION FOR PACKED SYMMETRIC MATRICES
// ============================================================================

// Compute condition number of packed symmetric matrix for f32/f64
m_condition_packed_symmetric_f32_f64 :: proc(
	ap: []$T, // Packed factored matrix from sptrf
	ipiv: []Blas_Int, // Pivot indices from sptrf
	anorm: T, // 1-norm of original matrix
	work: []T, // Pre-allocated workspace (size 2*n)
	iwork: []Blas_Int, // Pre-allocated integer workspace (size n)
	uplo := MatrixRegion.Upper,
	n: int, // Matrix dimension
) -> (
	rcond: T,
	info: Info,
	ok: bool,
) where T == f32 || T == f64 {
	assert(len(ap) >= n * (n + 1) / 2, "Packed array too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(len(work) >= 2 * n, "Workspace too small")
	assert(len(iwork) >= n, "Integer workspace too small")

	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)

	when T == f32 {
		lapack.sspcon_(&uplo_c, &n_int, raw_data(ap), raw_data(ipiv), &anorm, &rcond, raw_data(work), raw_data(iwork), &info, 1)
	} else when T == f64 {
		lapack.dspcon_(&uplo_c, &n_int, raw_data(ap), raw_data(ipiv), &anorm, &rcond, raw_data(work), raw_data(iwork), &info, 1)
	}

	ok = info == 0
	return rcond, info, ok
}

// Compute condition number of packed symmetric matrix for complex64/complex128
m_condition_packed_symmetric_c64_c128 :: proc(
	ap: []$T, // Packed factored matrix from sptrf
	ipiv: []Blas_Int, // Pivot indices from sptrf
	anorm: $R, // 1-norm of original matrix
	work: []T, // Pre-allocated workspace (size 2*n)
	uplo := MatrixRegion.Upper,
	n: int, // Matrix dimension
) -> (
	rcond: R,
	info: Info,
	ok: bool,
) where T == complex64 || T == complex128,
	R == real_type_of(T) {
	assert(len(ap) >= n * (n + 1) / 2, "Packed array too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(len(work) >= 2 * n, "Workspace too small")

	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)

	when T == complex64 {
		lapack.cspcon_(&uplo_c, &n_int, raw_data(ap), raw_data(ipiv), &anorm, &rcond, raw_data(work), &info, 1)
	} else when T == complex128 {
		lapack.zspcon_(&uplo_c, &n_int, raw_data(ap), raw_data(ipiv), &anorm, &rcond, raw_data(work), &info, 1)
	}

	ok = info == 0
	return rcond, info, ok
}

// Procedure group for packed symmetric condition estimation
m_condition_packed_symmetric :: proc {
	m_condition_packed_symmetric_f32_f64,
	m_condition_packed_symmetric_c64_c128,
}

// ============================================================================
// PACKED SYMMETRIC EIGENVALUE COMPUTATION
// ============================================================================

// Query workspace for packed symmetric eigenvalue computation
query_workspace_compute_packed_symmetric_eigenvalues :: proc($T: typeid, n: int) -> (work_size: int, rwork_size: int) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	when T == f32 || T == f64 {
		// Real types: work = 3*n, no rwork
		work_size = 3 * n
		if work_size < 1 {
			work_size = 1
		}
		rwork_size = 0
	} else {
		// Complex types: work = 2*n-1, rwork = 3*n-2
		work_size = max(1, 2 * n - 1)
		rwork_size = max(1, 3 * n - 2)
	}
	return
}

// Compute packed symmetric eigenvalues for f32/f64
m_compute_packed_symmetric_eigenvalues_f32_f64 :: proc(
	ap: []$T, // Packed matrix (modified on output)
	w: []T, // Pre-allocated eigenvalues array (size n)
	Z: ^Matrix(T) = nil, // Eigenvector matrix (optional, n x n)
	work: []T, // Pre-allocated workspace (size 3*n)
	jobz := EigenJobOption.VALUES_ONLY,
	uplo := MatrixRegion.Upper,
	n: int, // Matrix dimension
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 {
	assert(len(ap) >= n * (n + 1) / 2, "Packed array too small")
	assert(len(w) >= n, "Eigenvalue array too small")
	assert(len(work) >= 3 * n, "Workspace too small")

	jobz_c := eigen_job_to_char(jobz)
	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)

	// Handle eigenvector matrix
	ldz: Blas_Int = 1
	z_ptr: rawptr = nil
	if jobz == .VALUES_VECTORS && Z != nil {
		assert(Z.rows >= n && Z.cols >= n, "Eigenvector matrix too small")
		ldz = Blas_Int(Z.ld)
		z_ptr = raw_data(Z.data)
	}

	when T == f32 {
		lapack.sspev_(&jobz_c, &uplo_c, &n_int, raw_data(ap), raw_data(w), z_ptr, &ldz, raw_data(work), &info, 1, 1)
	} else when T == f64 {
		lapack.dspev_(&jobz_c, &uplo_c, &n_int, raw_data(ap), raw_data(w), z_ptr, &ldz, raw_data(work), &info, 1, 1)
	}

	ok = info == 0
	return info, ok
}

// Compute packed Hermitian eigenvalues for complex64/complex128
m_compute_packed_hermitian_eigenvalues_c64_c128 :: proc(
	ap: []$T, // Packed Hermitian matrix (modified on output)
	w: []$R, // Pre-allocated eigenvalues array (size n)
	Z: ^Matrix(T) = nil, // Eigenvector matrix (optional, n x n)
	work: []T, // Pre-allocated workspace (size 2*n-1)
	rwork: []R, // Pre-allocated real workspace (size 3*n-2)
	jobz := EigenJobOption.VALUES_ONLY,
	uplo := MatrixRegion.Upper,
	n: int, // Matrix dimension
) -> (
	info: Info,
	ok: bool,
) where T == complex64 || T == complex128,
	R == real_type_of(T) {
	assert(len(ap) >= n * (n + 1) / 2, "Packed array too small")
	assert(len(w) >= n, "Eigenvalue array too small")
	assert(len(work) >= max(1, 2 * n - 1), "Workspace too small")
	assert(len(rwork) >= max(1, 3 * n - 2), "Real workspace too small")

	jobz_c := eigen_job_to_char(jobz)
	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)

	// Handle eigenvector matrix
	ldz: Blas_Int = 1
	z_ptr: rawptr = nil
	if jobz == .VALUES_VECTORS && Z != nil {
		assert(Z.rows >= n && Z.cols >= n, "Eigenvector matrix too small")
		ldz = Blas_Int(Z.ld)
		z_ptr = raw_data(Z.data)
	}

	when T == complex64 {
		lapack.chpev_(&jobz_c, &uplo_c, &n_int, raw_data(ap), raw_data(w), z_ptr, &ldz, raw_data(work), raw_data(rwork), &info, 1, 1)
	} else when T == complex128 {
		lapack.zhpev_(&jobz_c, &uplo_c, &n_int, raw_data(ap), raw_data(w), z_ptr, &ldz, raw_data(work), raw_data(rwork), &info, 1, 1)
	}

	ok = info == 0
	return info, ok
}

// Procedure group for packed symmetric eigenvalue computation
m_compute_packed_symmetric_eigenvalues :: proc {
	m_compute_packed_symmetric_eigenvalues_f32_f64,
	m_compute_packed_hermitian_eigenvalues_c64_c128,
}

// ============================================================================
// PACKED SYMMETRIC EIGENVALUE - DIVIDE AND CONQUER
// ============================================================================

// Query workspace for packed symmetric eigenvalue computation (divide-and-conquer)
query_workspace_compute_packed_symmetric_eigenvalues_dc :: proc(
	$T: typeid,
	n: int,
	jobz: EigenJobOption,
) -> (
	work_size: int,
	iwork_size: int,
	rwork_size: int,
) where T == f32 ||
	T == f64 ||
	T == complex64 ||
	T == complex128 {
	// Query LAPACK for optimal workspace size
	jobz_c := eigen_job_to_char(jobz)
	uplo_c: u8 = 'U' // Default to upper
	n_int := Blas_Int(n)
	ldz := Blas_Int(max(1, n))
	lwork := Blas_Int(QUERY_WORKSPACE)
	liwork := Blas_Int(QUERY_WORKSPACE)
	info: Info
	work_query: T
	iwork_query: Blas_Int
	rwork_size = 0

	when T == f32 {
		lapack.sspevd_(&jobz_c, &uplo_c, &n_int, nil, nil, nil, &ldz, &work_query, &lwork, &iwork_query, &liwork, &info, 1, 1)
		work_size = int(work_query)
	} else when T == f64 {
		lapack.dspevd_(&jobz_c, &uplo_c, &n_int, nil, nil, nil, &ldz, &work_query, &lwork, &iwork_query, &liwork, &info, 1, 1)
		work_size = int(work_query)
	} else when T == complex64 {
		rwork_query: f32
		lrwork := Blas_Int(QUERY_WORKSPACE)
		lapack.chpevd_(&jobz_c, &uplo_c, &n_int, nil, nil, nil, &ldz, &work_query, &lwork, &rwork_query, &lrwork, &iwork_query, &liwork, &info, 1, 1)
		work_size = int(real(work_query))
		rwork_size = int(rwork_query)
	} else when T == complex128 {
		rwork_query: f64
		lrwork := Blas_Int(QUERY_WORKSPACE)
		lapack.zhpevd_(&jobz_c, &uplo_c, &n_int, nil, nil, nil, &ldz, &work_query, &lwork, &rwork_query, &lrwork, &iwork_query, &liwork, &info, 1, 1)
		work_size = int(real(work_query))
		rwork_size = int(rwork_query)
	}
	iwork_size = int(iwork_query)

	return
}

// Compute packed symmetric eigenvalues using divide-and-conquer for f32/f64
m_compute_packed_symmetric_eigenvalues_dc_f32_f64 :: proc(
	ap: []$T, // Packed matrix (modified on output)
	w: []T, // Pre-allocated eigenvalues array (size n)
	Z: ^Matrix(T) = nil, // Eigenvector matrix (optional, n x n)
	work: []T, // Pre-allocated workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace
	jobz := EigenJobOption.VALUES_ONLY,
	uplo := MatrixRegion.Upper,
	n: int, // Matrix dimension
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 {
	assert(len(ap) >= n * (n + 1) / 2, "Packed array too small")
	assert(len(w) >= n, "Eigenvalue array too small")
	assert(len(work) > 0, "Workspace required")
	assert(len(iwork) > 0, "Integer workspace required")

	jobz_c := eigen_job_to_char(jobz)
	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	lwork := Blas_Int(len(work))
	liwork := Blas_Int(len(iwork))

	// Handle eigenvector matrix
	ldz: Blas_Int = 1
	z_ptr: rawptr = nil
	if jobz == .VALUES_VECTORS && Z != nil {
		assert(Z.rows >= n && Z.cols >= n, "Eigenvector matrix too small")
		ldz = Blas_Int(Z.ld)
		z_ptr = raw_data(Z.data)
	}

	when T == f32 {
		lapack.sspevd_(&jobz_c, &uplo_c, &n_int, raw_data(ap), raw_data(w), z_ptr, &ldz, raw_data(work), &lwork, raw_data(iwork), &liwork, &info, 1, 1)
	} else when T == f64 {
		lapack.dspevd_(&jobz_c, &uplo_c, &n_int, raw_data(ap), raw_data(w), z_ptr, &ldz, raw_data(work), &lwork, raw_data(iwork), &liwork, &info, 1, 1)
	}

	ok = info == 0
	return info, ok
}

// Compute packed Hermitian eigenvalues using divide-and-conquer for complex64/complex128
m_compute_packed_hermitian_eigenvalues_dc_c64_c128 :: proc(
	ap: []$T, // Packed Hermitian matrix (modified on output)
	w: []$R, // Pre-allocated eigenvalues array (size n)
	Z: ^Matrix(T) = nil, // Eigenvector matrix (optional, n x n)
	work: []T, // Pre-allocated workspace
	rwork: []R, // Pre-allocated real workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace
	jobz := EigenJobOption.VALUES_ONLY,
	uplo := MatrixRegion.Upper,
	n: int, // Matrix dimension
) -> (
	info: Info,
	ok: bool,
) where T == complex64 || T == complex128,
	R == real_type_of(T) {
	assert(len(ap) >= n * (n + 1) / 2, "Packed array too small")
	assert(len(w) >= n, "Eigenvalue array too small")
	assert(len(work) > 0, "Workspace required")
	assert(len(rwork) > 0, "Real workspace required")
	assert(len(iwork) > 0, "Integer workspace required")

	jobz_c := eigen_job_to_char(jobz)
	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	lwork := Blas_Int(len(work))
	lrwork := Blas_Int(len(rwork))
	liwork := Blas_Int(len(iwork))

	// Handle eigenvector matrix
	ldz: Blas_Int = 1
	z_ptr: rawptr = nil
	if jobz == .VALUES_VECTORS && Z != nil {
		assert(Z.rows >= n && Z.cols >= n, "Eigenvector matrix too small")
		ldz = Blas_Int(Z.ld)
		z_ptr = raw_data(Z.data)
	}

	when T == complex64 {
		lapack.chpevd_(&jobz_c, &uplo_c, &n_int, raw_data(ap), raw_data(w), z_ptr, &ldz, raw_data(work), &lwork, raw_data(rwork), &lrwork, raw_data(iwork), &liwork, &info, 1, 1)
	} else when T == complex128 {
		lapack.zhpevd_(&jobz_c, &uplo_c, &n_int, raw_data(ap), raw_data(w), z_ptr, &ldz, raw_data(work), &lwork, raw_data(rwork), &lrwork, raw_data(iwork), &liwork, &info, 1, 1)
	}

	ok = info == 0
	return info, ok
}

// Procedure group for packed symmetric eigenvalue computation (divide-and-conquer)
m_compute_packed_symmetric_eigenvalues_dc :: proc {
	m_compute_packed_symmetric_eigenvalues_dc_f32_f64,
	m_compute_packed_hermitian_eigenvalues_dc_c64_c128,
}


// ============================================================================
// PACKED SYMMETRIC SELECTIVE EIGENVALUE
// ============================================================================

// Query workspace for packed symmetric eigenvalue computation (bisection/inverse iteration)
query_workspace_compute_packed_symmetric_eigenvalues_bisection :: proc($T: typeid, n: int) -> (work_size: int, iwork_size: int, rwork_size: int) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	when T == f32 || T == f64 {
		// Real types: work = 8*n, iwork = 5*n, no rwork
		work_size = 8 * n
		iwork_size = 5 * n
		rwork_size = 0
	} else {
		// Complex types: work = 2*n, rwork = 7*n, iwork = 5*n
		work_size = 2 * n
		rwork_size = 7 * n
		iwork_size = 5 * n
	}
	return
}

// Compute packed symmetric eigenvalues using bisection/inverse iteration for f32/f64
m_compute_packed_symmetric_eigenvalues_bisection_f32_f64 :: proc(
	ap: []$T, // Packed matrix (preserved)
	w: []T, // Pre-allocated eigenvalues array (size n)
	Z: ^Matrix(T) = nil, // Eigenvector matrix (optional, n x m)
	work: []T, // Pre-allocated workspace (size 8*n)
	iwork: []Blas_Int, // Pre-allocated integer workspace (size 5*n)
	ifail: []Blas_Int, // Pre-allocated failure indices (size n)
	jobz := EigenJobOption.VALUES_ONLY,
	range := EigenRangeOption.ALL,
	uplo := MatrixRegion.Upper,
	vl: T, // Lower bound (if range == VALUE)
	vu: T, // Upper bound (if range == VALUE)
	il: int = 1, // Lower index (if range == INDEX, 1-based)
	iu: int = 0, // Upper index (if range == INDEX, 1-based, 0=n)
	abstol: T, // Absolute tolerance
	n: int, // Matrix dimension
) -> (
	m: int,
	info: Info,
	ok: bool, // Number of eigenvalues found
) where T == f32 || T == f64 {
	assert(len(ap) >= n * (n + 1) / 2, "Packed array too small")
	assert(len(w) >= n, "Eigenvalue array too small")
	assert(len(work) >= 8 * n, "Workspace too small")
	assert(len(iwork) >= 5 * n, "Integer workspace too small")
	assert(len(ifail) >= n, "Failure array too small")

	jobz_c := eigen_job_to_char(jobz)
	range_c := eigen_range_to_char(range)
	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	il_int := Blas_Int(il)
	iu_int := Blas_Int(iu if iu > 0 else n)
	m_int: Blas_Int

	// Handle eigenvector matrix
	ldz: Blas_Int = 1
	z_ptr: rawptr = nil
	if jobz == .VALUES_VECTORS && Z != nil {
		assert(Z.rows >= n, "Eigenvector matrix too small")
		ldz = Blas_Int(Z.ld)
		z_ptr = raw_data(Z.data)
	}

	when T == f32 {
		lapack.sspevx_(&jobz_c, &range_c, &uplo_c, &n_int, raw_data(ap), &vl, &vu, &il_int, &iu_int, &abstol, &m_int, raw_data(w), z_ptr, &ldz, raw_data(work), raw_data(iwork), raw_data(ifail), &info, 1, 1, 1)
	} else when T == f64 {
		lapack.dspevx_(&jobz_c, &range_c, &uplo_c, &n_int, raw_data(ap), &vl, &vu, &il_int, &iu_int, &abstol, &m_int, raw_data(w), z_ptr, &ldz, raw_data(work), raw_data(iwork), raw_data(ifail), &info, 1, 1, 1)
	}

	m = int(m_int)
	ok = info == 0
	return m, info, ok
}

// Compute packed Hermitian eigenvalues using bisection/inverse iteration for complex64/complex128
m_compute_packed_hermitian_eigenvalues_bisection_c64_c128 :: proc(
	ap: []$T, // Packed Hermitian matrix (preserved)
	w: []$R, // Pre-allocated eigenvalues array (size n)
	Z: ^Matrix(T) = nil, // Eigenvector matrix (optional, n x m)
	work: []T, // Pre-allocated workspace (size 2*n)
	rwork: []R, // Pre-allocated real workspace (size 7*n)
	iwork: []Blas_Int, // Pre-allocated integer workspace (size 5*n)
	ifail: []Blas_Int, // Pre-allocated failure indices (size n)
	jobz := EigenJobOption.VALUES_ONLY,
	range := EigenRangeOption.ALL,
	uplo := MatrixRegion.Upper,
	vl: R, // Lower bound (if range == VALUE)
	vu: R, // Upper bound (if range == VALUE)
	il: int = 1, // Lower index (if range == INDEX, 1-based)
	iu: int = 0, // Upper index (if range == INDEX, 1-based, 0=n)
	abstol: R, // Absolute tolerance
	n: int, // Matrix dimension
) -> (
	m: int,
	info: Info,
	ok: bool, // Number of eigenvalues found
) where T == complex64 || T == complex128,
	R == real_type_of(T) {
	assert(len(ap) >= n * (n + 1) / 2, "Packed array too small")
	assert(len(w) >= n, "Eigenvalue array too small")
	assert(len(work) >= 2 * n, "Workspace too small")
	assert(len(rwork) >= 7 * n, "Real workspace too small")
	assert(len(iwork) >= 5 * n, "Integer workspace too small")
	assert(len(ifail) >= n, "Failure array too small")

	jobz_c := eigen_job_to_char(jobz)
	range_c := eigen_range_to_char(range)
	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	il_int := Blas_Int(il)
	iu_int := Blas_Int(iu if iu > 0 else n)
	m_int: Blas_Int

	// Handle eigenvector matrix
	ldz: Blas_Int = 1
	z_ptr: rawptr = nil
	if jobz == .VALUES_VECTORS && Z != nil {
		assert(Z.rows >= n, "Eigenvector matrix too small")
		ldz = Blas_Int(Z.ld)
		z_ptr = raw_data(Z.data)
	}

	when T == complex64 {
		lapack.chpevx_(
			&jobz_c,
			&range_c,
			&uplo_c,
			&n_int,
			raw_data(ap),
			&vl,
			&vu,
			&il_int,
			&iu_int,
			&abstol,
			&m_int,
			raw_data(w),
			z_ptr,
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
	} else when T == complex128 {
		lapack.zhpevx_(
			&jobz_c,
			&range_c,
			&uplo_c,
			&n_int,
			raw_data(ap),
			&vl,
			&vu,
			&il_int,
			&iu_int,
			&abstol,
			&m_int,
			raw_data(w),
			z_ptr,
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
	}

	m = int(m_int)
	ok = info == 0
	return m, info, ok
}

// Procedure group for packed symmetric eigenvalue computation (bisection/inverse iteration)
m_compute_packed_symmetric_eigenvalues_bisection :: proc {
	m_compute_packed_symmetric_eigenvalues_bisection_f32_f64,
	m_compute_packed_hermitian_eigenvalues_bisection_c64_c128,
}


// ============================================================================
// PACKED SYMMETRIC GENERALIZED REDUCTION
// ============================================================================

// Reduce packed symmetric generalized eigenvalue problem to standard form for f32/f64
m_reduce_packed_generalized_f32_f64 :: proc(
	ap: []$T, // Packed matrix A (modified to standard form)
	bp: []T, // Packed Cholesky factor of B
	itype := GeneralizedProblemType.AX_LBX,
	uplo := MatrixRegion.Upper,
	n: int, // Matrix dimension
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 {
	assert(len(ap) >= n * (n + 1) / 2, "Packed array A too small")
	assert(len(bp) >= n * (n + 1) / 2, "Packed array B too small")

	uplo_c := matrix_region_to_char(uplo)
	itype_int := Blas_Int(itype)
	n_int := Blas_Int(n)

	when T == f32 {
		lapack.sspgst_(&itype_int, &uplo_c, &n_int, raw_data(ap), raw_data(bp), &info, 1)
	} else when T == f64 {
		lapack.dspgst_(&itype_int, &uplo_c, &n_int, raw_data(ap), raw_data(bp), &info, 1)
	}

	ok = info == 0
	return info, ok
}

// Reduce packed Hermitian generalized eigenvalue problem to standard form for complex64/complex128
m_reduce_packed_generalized_c64_c128 :: proc(
	ap: []$T, // Packed Hermitian matrix A (modified to standard form)
	bp: []T, // Packed Cholesky factor of B
	itype := GeneralizedProblemType.AX_LBX,
	uplo := MatrixRegion.Upper,
	n: int, // Matrix dimension
) -> (
	info: Info,
	ok: bool,
) where T == complex64 || T == complex128 {
	assert(len(ap) >= n * (n + 1) / 2, "Packed array A too small")
	assert(len(bp) >= n * (n + 1) / 2, "Packed array B too small")

	uplo_c := matrix_region_to_char(uplo)
	itype_int := Blas_Int(itype)
	n_int := Blas_Int(n)

	when T == complex64 {
		lapack.chpgst_(&itype_int, &uplo_c, &n_int, raw_data(ap), raw_data(bp), &info, 1)
	} else when T == complex128 {
		lapack.zhpgst_(&itype_int, &uplo_c, &n_int, raw_data(ap), raw_data(bp), &info, 1)
	}

	ok = info == 0
	return info, ok
}

// Procedure group for packed symmetric generalized reduction
m_reduce_packed_generalized :: proc {
	m_reduce_packed_generalized_f32_f64,
	m_reduce_packed_generalized_c64_c128,
}

// ============================================================================
// CONVENIENCE FUNCTIONS
// ============================================================================

// Convert matrix to packed format
matrix_to_packed :: proc(mat: Matrix($T), uplo: MatrixRegion = .Lower, allocator := context.allocator) -> []T {
	n := mat.rows
	assert(mat.rows == mat.cols, "Matrix must be square")

	packed_size := n * (n + 1) / 2
	packed := make([]T, packed_size, allocator)

	idx := 0
	if uplo == .Upper {
		for j in 0 ..< n {
			for i in 0 ..= j {
				packed[idx] = matrix_get(&mat, i, j)
				idx += 1
			}
		}
	} else {
		for j in 0 ..< n {
			for i in j ..< n {
				packed[idx] = matrix_get(&mat, i, j)
				idx += 1
			}
		}
	}

	return packed
}

// Convert packed format to matrix
packed_to_matrix :: proc(packed: []$T, n: int, uplo: MatrixRegion = .Lower, allocator := context.allocator) -> Matrix(T) {
	assert(len(packed) >= n * (n + 1) / 2, "Packed array too small")

	mat := create_matrix(T, n, n, allocator)

	idx := 0
	if uplo == .Upper {
		for j in 0 ..< n {
			for i in 0 ..= j {
				matrix_set(&mat, i, j, packed[idx])
				if i != j {
					matrix_set(&mat, j, i, packed[idx]) // Symmetric
				}
				idx += 1
			}
		}
	} else {
		for j in 0 ..< n {
			for i in j ..< n {
				matrix_set(&mat, i, j, packed[idx])
				if i != j {
					matrix_set(&mat, j, i, packed[idx]) // Symmetric
				}
				idx += 1
			}
		}
	}

	return mat
}
