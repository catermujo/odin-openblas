package openblas

import lapack "./f77"
import "core:c"
import "core:math"
import "core:mem"
import "core:slice"

// ============================================================================
// SYMMETRIC EIGENVALUE SOLVERS - BISECTION AND INVERSE ITERATION
// ============================================================================

// Query workspace for symmetric eigenvalue computation (bisection/inverse iteration)
query_workspace_compute_symmetric_eigenvalues_bisection :: proc($T: typeid, n: int, jobz: EigenJobOption) -> (work_size: int) where T == f32 || T == f64 {
	// Query LAPACK for optimal workspace size
	jobz_c := eigen_job_to_char(jobz)
	range_c: u8 = 'A' // Default to ALL
	uplo_c: u8 = 'U' // Default to upper
	n_int := Blas_Int(n)
	lda := Blas_Int(max(1, n))
	vl: T = 0
	vu: T = 0
	il: Blas_Int = 1
	iu: Blas_Int = Blas_Int(n)
	abstol: T = 0
	m: Blas_Int
	ldz := Blas_Int(max(1, n))
	lwork := Blas_Int(QUERY_WORKSPACE)
	info: Info

	when T == f32 {
		work_query: f32
		lapack.ssyevx_(
			&jobz_c,
			&range_c,
			&uplo_c,
			&n_int,
			nil, // a
			&lda,
			&vl,
			&vu,
			&il,
			&iu,
			&abstol,
			&m,
			nil, // w
			nil, // z
			&ldz,
			&work_query,
			&lwork,
			nil, // iwork
			nil, // ifail
			&info,
			1,
			1,
			1,
		)
		work_size = int(work_query)
	} else when T == f64 {
		work_query: f64
		lapack.dsyevx_(
			&jobz_c,
			&range_c,
			&uplo_c,
			&n_int,
			nil, // a
			&lda,
			&vl,
			&vu,
			&il,
			&iu,
			&abstol,
			&m,
			nil, // w
			nil, // z
			&ldz,
			&work_query,
			&lwork,
			nil, // iwork
			nil, // ifail
			&info,
			1,
			1,
			1,
		)
		work_size = int(work_query)
	}

	// Minimum workspace requirement
	if work_size < 8 * n {
		work_size = 8 * n
	}

	return work_size
}

// Compute symmetric eigenvalues using bisection and inverse iteration for f32/f64
m_compute_symmetric_eigenvalues_bisection :: proc(
	a: ^Matrix($T), // Input matrix (modified)
	w: []T, // Pre-allocated eigenvalues array (size n)
	Z: ^Matrix(T) = nil, // Eigenvector matrix (optional, n x n)
	work: []T, // Pre-allocated workspace
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
) -> (
	m: int,
	info: Info,
	ok: bool, // Number of eigenvalues found
) where T == f32 || T == f64 {
	n := a.cols
	assert(a.rows >= n, "Matrix too small")
	assert(len(w) >= n, "Eigenvalue array too small")
	assert(len(work) > 0, "Workspace required")
	assert(len(iwork) >= 5 * n, "Integer workspace too small")
	assert(len(ifail) >= n, "Failure array too small")

	jobz_c := eigen_job_to_char(jobz)
	range_c := eigen_range_to_char(range)
	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	lda := Blas_Int(a.ld)

	// Range parameters
	vl_val := vl
	vu_val := vu
	il_int := Blas_Int(il)
	iu_int := Blas_Int(iu if iu > 0 else n)
	abstol_val := abstol
	m_int: Blas_Int

	// Eigenvector setup
	ldz: Blas_Int = 1
	z_ptr: rawptr = nil
	if jobz == .VALUES_VECTORS && Z != nil {
		assert(Z.rows >= n && Z.cols >= n, "Eigenvector matrix too small")
		ldz = Blas_Int(Z.ld)
		z_ptr = raw_data(Z.data)
	}

	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.ssyevx_(
			&jobz_c,
			&range_c,
			&uplo_c,
			&n_int,
			raw_data(a.data),
			&lda,
			&vl_val,
			&vu_val,
			&il_int,
			&iu_int,
			&abstol_val,
			&m_int,
			raw_data(w),
			z_ptr,
			&ldz,
			raw_data(work),
			&lwork,
			raw_data(iwork),
			raw_data(ifail),
			&info,
			1,
			1,
			1,
		)
	} else when T == f64 {
		lapack.dsyevx_(
			&jobz_c,
			&range_c,
			&uplo_c,
			&n_int,
			raw_data(a.data),
			&lda,
			&vl_val,
			&vu_val,
			&il_int,
			&iu_int,
			&abstol_val,
			&m_int,
			raw_data(w),
			z_ptr,
			&ldz,
			raw_data(work),
			&lwork,
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

// Query workspace for 2-stage symmetric eigenvalue computation (bisection/inverse iteration)
query_workspace_compute_symmetric_eigenvalues_bisection_2stage :: proc($T: typeid, n: int, jobz: EigenJobOption) -> (work_size: int) where T == f32 || T == f64 {
	// Query LAPACK for optimal workspace size
	jobz_c := eigen_job_to_char(jobz)
	range_c: u8 = 'A' // Default to ALL
	uplo_c: u8 = 'U' // Default to upper
	n_int := Blas_Int(n)
	lda := Blas_Int(max(1, n))
	vl: T = 0
	vu: T = 0
	il: Blas_Int = 1
	iu: Blas_Int = Blas_Int(n)
	abstol: T = 0
	m: Blas_Int
	ldz := Blas_Int(max(1, n))
	lwork := Blas_Int(QUERY_WORKSPACE)
	info: Info

	when T == f32 {
		work_query: f32
		lapack.ssyevx_2stage_(
			&jobz_c,
			&range_c,
			&uplo_c,
			&n_int,
			nil, // a
			&lda,
			&vl,
			&vu,
			&il,
			&iu,
			&abstol,
			&m,
			nil, // w
			nil, // z
			&ldz,
			&work_query,
			&lwork,
			nil, // iwork
			nil, // ifail
			&info,
			1,
			1,
			1,
		)
		work_size = int(work_query)
	} else when T == f64 {
		work_query: f64
		lapack.dsyevx_2stage_(
			&jobz_c,
			&range_c,
			&uplo_c,
			&n_int,
			nil, // a
			&lda,
			&vl,
			&vu,
			&il,
			&iu,
			&abstol,
			&m,
			nil, // w
			nil, // z
			&ldz,
			&work_query,
			&lwork,
			nil, // iwork
			nil, // ifail
			&info,
			1,
			1,
			1,
		)
		work_size = int(work_query)
	}

	return work_size
}

// Compute symmetric eigenvalues using 2-stage bisection and inverse iteration for f32/f64
m_compute_symmetric_eigenvalues_bisection_2stage :: proc(
	a: ^Matrix($T), // Input matrix (modified)
	w: []T, // Pre-allocated eigenvalues array (size n)
	Z: ^Matrix(T) = nil, // Eigenvector matrix (optional, n x n)
	work: []T, // Pre-allocated workspace
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
) -> (
	m: int,
	info: Info,
	ok: bool, // Number of eigenvalues found
) where T == f32 || T == f64 {
	n := a.cols
	assert(a.rows >= n, "Matrix too small")
	assert(len(w) >= n, "Eigenvalue array too small")
	assert(len(work) > 0, "Workspace required")
	assert(len(iwork) >= 5 * n, "Integer workspace too small")
	assert(len(ifail) >= n, "Failure array too small")

	jobz_c := eigen_job_to_char(jobz)
	range_c := eigen_range_to_char(range)
	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	lda := Blas_Int(a.ld)

	// Range parameters
	vl_val := vl
	vu_val := vu
	il_int := Blas_Int(il)
	iu_int := Blas_Int(iu if iu > 0 else n)
	abstol_val := abstol
	m_int: Blas_Int

	// Eigenvector setup
	ldz: Blas_Int = 1
	z_ptr: rawptr = nil
	if jobz == .VALUES_VECTORS && Z != nil {
		assert(Z.rows >= n && Z.cols >= n, "Eigenvector matrix too small")
		ldz = Blas_Int(Z.ld)
		z_ptr = raw_data(Z.data)
	}

	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.ssyevx_2stage_(
			&jobz_c,
			&range_c,
			&uplo_c,
			&n_int,
			raw_data(a.data),
			&lda,
			&vl_val,
			&vu_val,
			&il_int,
			&iu_int,
			&abstol_val,
			&m_int,
			raw_data(w),
			z_ptr,
			&ldz,
			raw_data(work),
			&lwork,
			raw_data(iwork),
			raw_data(ifail),
			&info,
			1,
			1,
			1,
		)
	} else when T == f64 {
		lapack.dsyevx_2stage_(
			&jobz_c,
			&range_c,
			&uplo_c,
			&n_int,
			raw_data(a.data),
			&lda,
			&vl_val,
			&vu_val,
			&il_int,
			&iu_int,
			&abstol_val,
			&m_int,
			raw_data(w),
			z_ptr,
			&ldz,
			raw_data(work),
			&lwork,
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
