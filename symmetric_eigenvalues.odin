package openblas

import lapack "./f77"
import "core:c"
import "core:math"
import "core:mem"
import "core:slice"

// ============================================================================
// SYMMETRIC EIGENVALUE SOLVERS - QR ALGORITHM
// ============================================================================

// Query workspace for symmetric eigenvalue computation (QR algorithm)
query_workspace_compute_symmetric_eigenvalues :: proc($T: typeid, n: int, jobz: EigenJobOption) -> (work_size: int) where T == f32 || T == f64 {
	// Query LAPACK for optimal workspace size
	jobz_c := eigen_job_to_char(jobz)
	uplo_c: u8 = 'U' // Default to upper
	n_int := Blas_Int(n)
	lda := Blas_Int(max(1, n))
	lwork := Blas_Int(QUERY_WORKSPACE)
	info: Info

	when T == f32 {
		work_query: f32
		lapack.ssyev_(
			&jobz_c,
			&uplo_c,
			&n_int,
			nil, // a
			&lda,
			nil, // w
			&work_query,
			&lwork,
			&info,
			1,
			1,
		)
		work_size = int(work_query)
	} else when T == f64 {
		work_query: f64
		lapack.dsyev_(
			&jobz_c,
			&uplo_c,
			&n_int,
			nil, // a
			&lda,
			nil, // w
			&work_query,
			&lwork,
			&info,
			1,
			1,
		)
		work_size = int(work_query)
	}

	// Minimum workspace requirement
	if work_size < max(1, 3 * n - 1) {
		work_size = max(1, 3 * n - 1)
	}

	return work_size
}

// Compute symmetric eigenvalues using QR algorithm for f32/f64
m_compute_symmetric_eigenvalues :: proc(
	a: ^Matrix($T), // Input matrix (destroyed, eigenvectors on output if jobz == VALUES_VECTORS)
	w: []T, // Pre-allocated eigenvalues array (size n)
	work: []T, // Pre-allocated workspace
	jobz := EigenJobOption.VALUES_ONLY,
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 {
	n := a.cols
	assert(a.rows >= n, "Matrix too small")
	assert(len(w) >= n, "Eigenvalue array too small")
	assert(len(work) > 0, "Workspace required")

	jobz_c := eigen_job_to_char(jobz)
	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	lda := Blas_Int(a.ld)
	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.ssyev_(&jobz_c, &uplo_c, &n_int, raw_data(a.data), &lda, raw_data(w), raw_data(work), &lwork, &info, 1, 1)
	} else when T == f64 {
		lapack.dsyev_(&jobz_c, &uplo_c, &n_int, raw_data(a.data), &lda, raw_data(w), raw_data(work), &lwork, &info, 1, 1)
	}

	ok = info == 0
	return info, ok
}

// ============================================================================
// 2-STAGE SYMMETRIC EIGENVALUE SOLVERS
// ============================================================================

// Query workspace for 2-stage symmetric eigenvalue computation
query_workspace_compute_symmetric_eigenvalues_2stage :: proc($T: typeid, n: int, jobz: EigenJobOption) -> (work_size: int) where T == f32 || T == f64 {
	// Query LAPACK for optimal workspace size
	jobz_c := eigen_job_to_char(jobz)
	uplo_c: u8 = 'U' // Default to upper
	n_int := Blas_Int(n)
	lda := Blas_Int(max(1, n))
	lwork := Blas_Int(QUERY_WORKSPACE)
	info: Info

	when T == f32 {
		work_query: f32
		lapack.ssyev_2stage_(
			&jobz_c,
			&uplo_c,
			&n_int,
			nil, // a
			&lda,
			nil, // w
			&work_query,
			&lwork,
			&info,
			1,
			1,
		)
		work_size = int(work_query)
	} else when T == f64 {
		work_query: f64
		lapack.dsyev_2stage_(
			&jobz_c,
			&uplo_c,
			&n_int,
			nil, // a
			&lda,
			nil, // w
			&work_query,
			&lwork,
			&info,
			1,
			1,
		)
		work_size = int(work_query)
	}

	return work_size
}

// Compute symmetric eigenvalues using 2-stage algorithm for f32/f64
m_compute_symmetric_eigenvalues_2stage :: proc(
	a: ^Matrix($T), // Input matrix (destroyed, eigenvectors on output if jobz == VALUES_VECTORS)
	w: []T, // Pre-allocated eigenvalues array (size n)
	work: []T, // Pre-allocated workspace
	jobz := EigenJobOption.VALUES_ONLY,
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 {
	n := a.cols
	assert(a.rows >= n, "Matrix too small")
	assert(len(w) >= n, "Eigenvalue array too small")
	assert(len(work) > 0, "Workspace required")

	jobz_c := eigen_job_to_char(jobz)
	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	lda := Blas_Int(a.ld)
	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.ssyev_2stage_(&jobz_c, &uplo_c, &n_int, raw_data(a.data), &lda, raw_data(w), raw_data(work), &lwork, &info, 1, 1)
	} else when T == f64 {
		lapack.dsyev_2stage_(&jobz_c, &uplo_c, &n_int, raw_data(a.data), &lda, raw_data(w), raw_data(work), &lwork, &info, 1, 1)
	}

	ok = info == 0
	return info, ok
}

// ============================================================================
// SYMMETRIC EIGENVALUE SOLVERS - DIVIDE AND CONQUER
// ============================================================================

// Query workspace for symmetric eigenvalue computation (divide-and-conquer)
query_workspace_compute_symmetric_eigenvalues_dc :: proc($T: typeid, n: int, jobz: EigenJobOption) -> (work_size: int, iwork_size: int) where T == f32 || T == f64 {
	// Query LAPACK for optimal workspace size
	jobz_c := eigen_job_to_char(jobz)
	uplo_c: u8 = 'U' // Default to upper
	n_int := Blas_Int(n)
	lda := Blas_Int(max(1, n))
	lwork := Blas_Int(QUERY_WORKSPACE)
	liwork := Blas_Int(QUERY_WORKSPACE)
	info: Info

	when T == f32 {
		work_query: f32
		iwork_query: Blas_Int
		lapack.ssyevd_(
			&jobz_c,
			&uplo_c,
			&n_int,
			nil, // a
			&lda,
			nil, // w
			&work_query,
			&lwork,
			&iwork_query,
			&liwork,
			&info,
			1,
			1,
		)
		work_size = int(work_query)
		iwork_size = int(iwork_query)
	} else when T == f64 {
		work_query: f64
		iwork_query: Blas_Int
		lapack.dsyevd_(
			&jobz_c,
			&uplo_c,
			&n_int,
			nil, // a
			&lda,
			nil, // w
			&work_query,
			&lwork,
			&iwork_query,
			&liwork,
			&info,
			1,
			1,
		)
		work_size = int(work_query)
		iwork_size = int(iwork_query)
	}

	return work_size, iwork_size
}

// Compute symmetric eigenvalues using divide-and-conquer for f32/f64
m_compute_symmetric_eigenvalues_dc :: proc(
	a: ^Matrix($T), // Input matrix (destroyed, eigenvectors on output if jobz == VALUES_VECTORS)
	w: []T, // Pre-allocated eigenvalues array (size n)
	work: []T, // Pre-allocated workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace
	jobz := EigenJobOption.VALUES_ONLY,
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 {
	n := a.cols
	assert(a.rows >= n, "Matrix too small")
	assert(len(w) >= n, "Eigenvalue array too small")
	assert(len(work) > 0, "Workspace required")
	assert(len(iwork) > 0, "Integer workspace required")

	jobz_c := eigen_job_to_char(jobz)
	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	lda := Blas_Int(a.ld)
	lwork := Blas_Int(len(work))
	liwork := Blas_Int(len(iwork))

	when T == f32 {
		lapack.ssyevd_(&jobz_c, &uplo_c, &n_int, raw_data(a.data), &lda, raw_data(w), raw_data(work), &lwork, raw_data(iwork), &liwork, &info, 1, 1)
	} else when T == f64 {
		lapack.dsyevd_(&jobz_c, &uplo_c, &n_int, raw_data(a.data), &lda, raw_data(w), raw_data(work), &lwork, raw_data(iwork), &liwork, &info, 1, 1)
	}

	ok = info == 0
	return info, ok
}

// Query workspace for 2-stage symmetric eigenvalue computation (divide-and-conquer)
query_workspace_compute_symmetric_eigenvalues_dc_2stage :: proc($T: typeid, n: int, jobz: EigenJobOption) -> (work_size: int, iwork_size: int) where T == f32 || T == f64 {
	// Query LAPACK for optimal workspace size
	jobz_c := eigen_job_to_char(jobz)
	uplo_c: u8 = 'U' // Default to upper
	n_int := Blas_Int(n)
	lda := Blas_Int(max(1, n))
	lwork := Blas_Int(QUERY_WORKSPACE)
	liwork := Blas_Int(QUERY_WORKSPACE)
	info: Info

	when T == f32 {
		work_query: f32
		iwork_query: Blas_Int
		lapack.ssyevd_2stage_(
			&jobz_c,
			&uplo_c,
			&n_int,
			nil, // a
			&lda,
			nil, // w
			&work_query,
			&lwork,
			&iwork_query,
			&liwork,
			&info,
			1,
			1,
		)
		work_size = int(work_query)
		iwork_size = int(iwork_query)
	} else when T == f64 {
		work_query: f64
		iwork_query: Blas_Int
		lapack.dsyevd_2stage_(
			&jobz_c,
			&uplo_c,
			&n_int,
			nil, // a
			&lda,
			nil, // w
			&work_query,
			&lwork,
			&iwork_query,
			&liwork,
			&info,
			1,
			1,
		)
		work_size = int(work_query)
		iwork_size = int(iwork_query)
	}

	return work_size, iwork_size
}

// Compute symmetric eigenvalues using 2-stage divide-and-conquer for f32/f64
m_compute_symmetric_eigenvalues_dc_2stage :: proc(
	a: ^Matrix($T), // Input matrix (destroyed, eigenvectors on output if jobz == VALUES_VECTORS)
	w: []T, // Pre-allocated eigenvalues array (size n)
	work: []T, // Pre-allocated workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace
	jobz := EigenJobOption.VALUES_ONLY,
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 {
	n := a.cols
	assert(a.rows >= n, "Matrix too small")
	assert(len(w) >= n, "Eigenvalue array too small")
	assert(len(work) > 0, "Workspace required")
	assert(len(iwork) > 0, "Integer workspace required")

	jobz_c := eigen_job_to_char(jobz)
	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	lda := Blas_Int(a.ld)
	lwork := Blas_Int(len(work))
	liwork := Blas_Int(len(iwork))

	when T == f32 {
		lapack.ssyevd_2stage_(&jobz_c, &uplo_c, &n_int, raw_data(a.data), &lda, raw_data(w), raw_data(work), &lwork, raw_data(iwork), &liwork, &info, 1, 1)
	} else when T == f64 {
		lapack.dsyevd_2stage_(&jobz_c, &uplo_c, &n_int, raw_data(a.data), &lda, raw_data(w), raw_data(work), &lwork, raw_data(iwork), &liwork, &info, 1, 1)
	}

	ok = info == 0
	return info, ok
}

// ============================================================================
// SYMMETRIC EIGENVALUE SOLVERS - MRRR
// ============================================================================

// Query workspace for symmetric eigenvalue computation (MRRR)
query_workspace_compute_symmetric_eigenvalues_mrrr :: proc($T: typeid, n: int, jobz: EigenJobOption, range: EigenRangeOption) -> (work_size: int, iwork_size: int) where T == f32 || T == f64 {
	// Query LAPACK for optimal workspace size
	jobz_c := eigen_job_to_char(jobz)
	range_c := eigen_range_to_char(range)
	uplo_c: u8 = 'U' // Default to upper
	n_int := Blas_Int(n)
	lda := Blas_Int(max(1, n))
	vl: T
	vu: T
	il: Blas_Int = 1
	iu: Blas_Int = Blas_Int(n)
	abstol: T
	m: Blas_Int
	ldz := Blas_Int(max(1, n))
	lwork := Blas_Int(QUERY_WORKSPACE)
	liwork := Blas_Int(QUERY_WORKSPACE)
	info: Info

	when T == f32 {
		work_query: f32
		iwork_query: Blas_Int
		lapack.ssyevr_(
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
			nil, // isuppz
			&work_query,
			&lwork,
			&iwork_query,
			&liwork,
			&info,
			1,
			1,
			1,
		)
		work_size = int(work_query)
		iwork_size = int(iwork_query)
	} else when T == f64 {
		work_query: f64
		iwork_query: Blas_Int
		lapack.dsyevr_(
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
			nil, // isuppz
			&work_query,
			&lwork,
			&iwork_query,
			&liwork,
			&info,
			1,
			1,
			1,
		)
		work_size = int(work_query)
		iwork_size = int(iwork_query)
	}

	return work_size, iwork_size
}

// Compute symmetric eigenvalues using MRRR for f32/f64
m_compute_symmetric_eigenvalues_mrrr :: proc(
	a: ^Matrix($T), // Input matrix (modified)
	w: []T, // Pre-allocated eigenvalues array (size n)
	Z: ^Matrix(T) = nil, // Eigenvector matrix (optional, n x n)
	isuppz: []Blas_Int = nil, // Support arrays (size 2*max(1,m))
	work: []T, // Pre-allocated workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace
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
	assert(len(iwork) > 0, "Integer workspace required")

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

	// Support array check
	if jobz == .VALUES_VECTORS && isuppz != nil {
		max_m := range == .ALL ? n : (range == .INDEX ? iu - il + 1 : n)
		assert(len(isuppz) >= 2 * max(1, max_m), "Support array too small")
	}

	lwork := Blas_Int(len(work))
	liwork := Blas_Int(len(iwork))

	when T == f32 {
		lapack.ssyevr_(
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
			raw_data(isuppz) if isuppz != nil else nil,
			raw_data(work),
			&lwork,
			raw_data(iwork),
			&liwork,
			&info,
			1,
			1,
			1,
		)
	} else when T == f64 {
		lapack.dsyevr_(
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
			raw_data(isuppz) if isuppz != nil else nil,
			raw_data(work),
			&lwork,
			raw_data(iwork),
			&liwork,
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

// Query workspace for 2-stage symmetric eigenvalue computation (MRRR)
query_workspace_compute_symmetric_eigenvalues_mrrr_2stage :: proc($T: typeid, n: int, jobz: EigenJobOption, range: EigenRangeOption) -> (work_size: int, iwork_size: int) where T == f32 || T == f64 {
	// Query LAPACK for optimal workspace size
	jobz_c := eigen_job_to_char(jobz)
	range_c := eigen_range_to_char(range)
	uplo_c: u8 = 'U' // Default to upper
	n_int := Blas_Int(n)
	lda := Blas_Int(max(1, n))
	vl: T
	vu: T
	il: Blas_Int = 1
	iu: Blas_Int = Blas_Int(n)
	abstol: T
	m: Blas_Int
	ldz := Blas_Int(max(1, n))
	lwork := Blas_Int(QUERY_WORKSPACE)
	liwork := Blas_Int(QUERY_WORKSPACE)
	info: Info

	when T == f32 {
		work_query: f32
		iwork_query: Blas_Int
		lapack.ssyevr_2stage_(
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
			nil, // isuppz
			&work_query,
			&lwork,
			&iwork_query,
			&liwork,
			&info,
			1,
			1,
			1,
		)
		work_size = int(work_query)
		iwork_size = int(iwork_query)
	} else when T == f64 {
		work_query: f64
		iwork_query: Blas_Int
		lapack.dsyevr_2stage_(
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
			nil, // isuppz
			&work_query,
			&lwork,
			&iwork_query,
			&liwork,
			&info,
			1,
			1,
			1,
		)
		work_size = int(work_query)
		iwork_size = int(iwork_query)
	}

	return work_size, iwork_size
}

// Compute symmetric eigenvalues using 2-stage MRRR for f32/f64
m_compute_symmetric_eigenvalues_mrrr_2stage :: proc(
	a: ^Matrix($T), // Input matrix (modified)
	w: []T, // Pre-allocated eigenvalues array (size n)
	Z: ^Matrix(T) = nil, // Eigenvector matrix (optional, n x n)
	isuppz: []Blas_Int = nil, // Support arrays (size 2*max(1,m))
	work: []T, // Pre-allocated workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace
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
	assert(len(iwork) > 0, "Integer workspace required")

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

	// Support array check
	if jobz == .VALUES_VECTORS && isuppz != nil {
		max_m := range == .ALL ? n : (range == .INDEX ? iu - il + 1 : n)
		assert(len(isuppz) >= 2 * max(1, max_m), "Support array too small")
	}

	lwork := Blas_Int(len(work))
	liwork := Blas_Int(len(iwork))

	when T == f32 {
		lapack.ssyevr_2stage_(
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
			raw_data(isuppz) if isuppz != nil else nil,
			raw_data(work),
			&lwork,
			raw_data(iwork),
			&liwork,
			&info,
			1,
			1,
			1,
		)
	} else when T == f64 {
		lapack.dsyevr_2stage_(
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
			raw_data(isuppz) if isuppz != nil else nil,
			raw_data(work),
			&lwork,
			raw_data(iwork),
			&liwork,
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
