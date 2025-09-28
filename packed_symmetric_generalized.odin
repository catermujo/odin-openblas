package openblas

import lapack "./f77"
import "core:c"
import "core:math"
import "core:mem"
import "core:slice"

// ============================================================================
// PACKED SYMMETRIC GENERALIZED EIGENVALUE PROBLEMS
// ============================================================================
// Solves the generalized eigenvalue problem A*x = λ*B*x where A and B are
// symmetric matrices stored in packed format and B is positive definite

// ============================================================================
// PACKED SYMMETRIC GENERALIZED EIGENVALUE
// ============================================================================

// Query workspace for packed generalized eigenvalue computation
query_workspace_compute_packed_generalized_eigenvalues :: proc($T: typeid, n: int) -> (work_size: int, rwork_size: int) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	when T == f32 || T == f64 {
		// Real types: work = 3*n, no rwork
		work_size = 3 * n
		if work_size < 1 {
			work_size = 1
		}
		rwork_size = 0
	} else {
		// Complex types: work = 2*n, rwork = 3*n
		work_size = max(1, 2 * n)
		rwork_size = max(1, 3 * n)
	}
	return
}

// Compute packed generalized eigenvalues for f32/f64
m_compute_packed_generalized_eigenvalues_f32_f64 :: proc(
	ap: []$T, // Packed matrix A (modified on output)
	bp: []T, // Packed matrix B (modified to Cholesky factor)
	w: []T, // Pre-allocated eigenvalues array (size n)
	Z: ^Matrix(T) = nil, // Eigenvector matrix (optional, n x n)
	work: []T, // Pre-allocated workspace (size 3*n)
	itype := GeneralizedProblemType.AX_LBX,
	jobz := EigenJobOption.VALUES_ONLY,
	uplo := MatrixRegion.Upper,
	n: int, // Matrix dimension
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 {
	assert(len(ap) >= n * (n + 1) / 2, "Packed array A too small")
	assert(len(bp) >= n * (n + 1) / 2, "Packed array B too small")
	assert(len(w) >= n, "Eigenvalue array too small")
	assert(len(work) >= 3 * n, "Workspace too small")

	itype_int := Blas_Int(itype)
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
		lapack.sspgv_(&itype_int, &jobz_c, &uplo_c, &n_int, raw_data(ap), raw_data(bp), raw_data(w), z_ptr, &ldz, raw_data(work), &info, 1, 1)
	} else when T == f64 {
		lapack.dspgv_(&itype_int, &jobz_c, &uplo_c, &n_int, raw_data(ap), raw_data(bp), raw_data(w), z_ptr, &ldz, raw_data(work), &info, 1, 1)
	}

	ok = info == 0
	return info, ok
}

// Compute packed generalized eigenvalues for complex64/complex128
m_compute_packed_generalized_eigenvalues_c64_c128 :: proc(
	ap: []$T, // Packed Hermitian matrix A (modified on output)
	bp: []T, // Packed Hermitian matrix B (modified to Cholesky factor)
	w: []$R, // Pre-allocated eigenvalues array (size n)
	Z: ^Matrix(T) = nil, // Eigenvector matrix (optional, n x n)
	work: []T, // Pre-allocated workspace (size 2*n)
	rwork: []R, // Pre-allocated real workspace (size 3*n)
	itype := GeneralizedProblemType.AX_LBX,
	jobz := EigenJobOption.VALUES_ONLY,
	uplo := MatrixRegion.Upper,
	n: int, // Matrix dimension
) -> (
	info: Info,
	ok: bool,
) where T == complex64 || T == complex128,
	R == real_type_of(T) {
	assert(len(ap) >= n * (n + 1) / 2, "Packed array A too small")
	assert(len(bp) >= n * (n + 1) / 2, "Packed array B too small")
	assert(len(w) >= n, "Eigenvalue array too small")
	assert(len(work) >= max(1, 2 * n), "Workspace too small")
	assert(len(rwork) >= max(1, 3 * n), "Real workspace too small")

	itype_int := Blas_Int(itype)
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
		lapack.chpgv_(&itype_int, &jobz_c, &uplo_c, &n_int, raw_data(ap), raw_data(bp), raw_data(w), z_ptr, &ldz, raw_data(work), raw_data(rwork), &info, 1, 1)
	} else when T == complex128 {
		lapack.zhpgv_(&itype_int, &jobz_c, &uplo_c, &n_int, raw_data(ap), raw_data(bp), raw_data(w), z_ptr, &ldz, raw_data(work), raw_data(rwork), &info, 1, 1)
	}

	ok = info == 0
	return info, ok
}

// Procedure group for packed generalized eigenvalue computation
m_compute_packed_generalized_eigenvalues :: proc {
	m_compute_packed_generalized_eigenvalues_f32_f64,
	m_compute_packed_generalized_eigenvalues_c64_c128,
}

// ============================================================================
// PACKED SYMMETRIC GENERALIZED EIGENVALUE - DIVIDE AND CONQUER
// ============================================================================

// Query workspace for divide-and-conquer packed generalized eigenvalue computation
query_workspace_compute_packed_generalized_eigenvalues_dc :: proc(
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
	itype_int := Blas_Int(1) // Default to AX_LBX
	jobz_c := eigen_job_to_char(jobz)
	uplo_c: u8 = 'U' // Default to upper
	n_int := Blas_Int(n)
	ldz := Blas_Int(max(1, n))
	lwork := Blas_Int(QUERY_WORKSPACE)
	liwork := Blas_Int(QUERY_WORKSPACE)
	info: Info

	when T == f32 {
		work_query: f32
		iwork_query: Blas_Int
		lapack.sspgvd_(
			&itype_int,
			&jobz_c,
			&uplo_c,
			&n_int,
			nil, // ap
			nil, // bp
			nil, // w
			nil, // z
			&ldz,
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
		rwork_size = 0
	} else when T == f64 {
		work_query: f64
		iwork_query: Blas_Int
		lapack.dspgvd_(
			&itype_int,
			&jobz_c,
			&uplo_c,
			&n_int,
			nil, // ap
			nil, // bp
			nil, // w
			nil, // z
			&ldz,
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
		rwork_size = 0
	} else when T == complex64 {
		work_query: complex64
		iwork_query: Blas_Int
		rwork_query: f32
		lrwork := Blas_Int(QUERY_WORKSPACE)
		lapack.chpgvd_(
			&itype_int,
			&jobz_c,
			&uplo_c,
			&n_int,
			nil, // ap
			nil, // bp
			nil, // w
			nil, // z
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
		work_size = int(real(work_query))
		iwork_size = int(iwork_query)
		rwork_size = int(rwork_query)
	} else when T == complex128 {
		work_query: complex128
		iwork_query: Blas_Int
		rwork_query: f64
		lrwork := Blas_Int(QUERY_WORKSPACE)
		lapack.zhpgvd_(
			&itype_int,
			&jobz_c,
			&uplo_c,
			&n_int,
			nil, // ap
			nil, // bp
			nil, // w
			nil, // z
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
		work_size = int(real(work_query))
		iwork_size = int(iwork_query)
		rwork_size = int(rwork_query)
	}

	// Ensure minimum workspace
	if work_size < 1 {
		work_size = 1
	}
	if iwork_size < 1 {
		iwork_size = 1
	}
	if rwork_size < 1 && (T == complex64 || T == complex128) {
		rwork_size = 1
	}

	return work_size, iwork_size, rwork_size
}

// Compute packed generalized eigenvalues using divide-and-conquer for f32/f64
m_compute_packed_generalized_eigenvalues_dc_f32_f64 :: proc(
	ap: []$T, // Packed matrix A (modified on output)
	bp: []T, // Packed matrix B (modified to Cholesky factor)
	w: []T, // Pre-allocated eigenvalues array (size n)
	Z: ^Matrix(T) = nil, // Eigenvector matrix (optional, n x n)
	work: []T, // Pre-allocated workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace
	itype := GeneralizedProblemType.AX_LBX,
	jobz := EigenJobOption.VALUES_ONLY,
	uplo := MatrixRegion.Upper,
	n: int, // Matrix dimension
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 {
	assert(len(ap) >= n * (n + 1) / 2, "Packed array A too small")
	assert(len(bp) >= n * (n + 1) / 2, "Packed array B too small")
	assert(len(w) >= n, "Eigenvalue array too small")
	assert(len(work) > 0, "Workspace required")
	assert(len(iwork) > 0, "Integer workspace required")

	itype_int := Blas_Int(itype)
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
		lapack.sspgvd_(&itype_int, &jobz_c, &uplo_c, &n_int, raw_data(ap), raw_data(bp), raw_data(w), z_ptr, &ldz, raw_data(work), &lwork, raw_data(iwork), &liwork, &info, 1, 1)
	} else when T == f64 {
		lapack.dspgvd_(&itype_int, &jobz_c, &uplo_c, &n_int, raw_data(ap), raw_data(bp), raw_data(w), z_ptr, &ldz, raw_data(work), &lwork, raw_data(iwork), &liwork, &info, 1, 1)
	}

	ok = info == 0
	return info, ok
}

// Compute packed generalized eigenvalues using divide-and-conquer for complex64/complex128
m_compute_packed_generalized_eigenvalues_dc_c64_c128 :: proc(
	ap: []$T, // Packed Hermitian matrix A (modified on output)
	bp: []T, // Packed Hermitian matrix B (modified to Cholesky factor)
	w: []$R, // Pre-allocated eigenvalues array (size n)
	Z: ^Matrix(T) = nil, // Eigenvector matrix (optional, n x n)
	work: []T, // Pre-allocated workspace
	rwork: []R, // Pre-allocated real workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace
	itype := GeneralizedProblemType.AX_LBX,
	jobz := EigenJobOption.VALUES_ONLY,
	uplo := MatrixRegion.Upper,
	n: int, // Matrix dimension
) -> (
	info: Info,
	ok: bool,
) where T == complex64 || T == complex128,
	R == real_type_of(T) {
	assert(len(ap) >= n * (n + 1) / 2, "Packed array A too small")
	assert(len(bp) >= n * (n + 1) / 2, "Packed array B too small")
	assert(len(w) >= n, "Eigenvalue array too small")
	assert(len(work) > 0, "Workspace required")
	assert(len(rwork) > 0, "Real workspace required")
	assert(len(iwork) > 0, "Integer workspace required")

	itype_int := Blas_Int(itype)
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
		lapack.chpgvd_(
			&itype_int,
			&jobz_c,
			&uplo_c,
			&n_int,
			cast(^lapack.complex)raw_data(ap),
			cast(^lapack.complex)raw_data(bp),
			raw_data(w),
			cast(^lapack.complex)z_ptr,
			&ldz,
			cast(^lapack.complex)raw_data(work),
			&lwork,
			raw_data(rwork),
			&lrwork,
			raw_data(iwork),
			&liwork,
			&info,
			1,
			1,
		)
	} else when T == complex128 {
		lapack.zhpgvd_(
			&itype_int,
			&jobz_c,
			&uplo_c,
			&n_int,
			cast(^lapack.doublecomplex)raw_data(ap),
			cast(^lapack.doublecomplex)raw_data(bp),
			raw_data(w),
			cast(^lapack.doublecomplex)z_ptr,
			&ldz,
			cast(^lapack.doublecomplex)raw_data(work),
			&lwork,
			raw_data(rwork),
			&lrwork,
			raw_data(iwork),
			&liwork,
			&info,
			1,
			1,
		)
	}

	ok = info == 0
	return info, ok
}

// Procedure group for divide-and-conquer packed generalized eigenvalue computation
m_compute_packed_generalized_eigenvalues_dc :: proc {
	m_compute_packed_generalized_eigenvalues_dc_f32_f64,
	m_compute_packed_generalized_eigenvalues_dc_c64_c128,
}

// ============================================================================
// PACKED SYMMETRIC GENERALIZED EIGENVALUE - SELECTIVE
// ============================================================================

// Compute selective packed generalized eigenvalues for f32/f64
m_compute_packed_generalized_eigenvalues_selective_f32_f64 :: proc(
	ap: []$T, // Packed matrix A (modified on output)
	bp: []T, // Packed matrix B (modified to Cholesky factor)
	w: []T, // Pre-allocated eigenvalues array (size n)
	Z: ^Matrix(T) = nil, // Eigenvector matrix (optional, n x max_eigenvectors)
	work: []T, // Pre-allocated workspace (size 8*n)
	iwork: []Blas_Int, // Pre-allocated integer workspace (size 5*n)
	ifail: []Blas_Int, // Pre-allocated failure indices (size n)
	itype := GeneralizedProblemType.AX_LBX,
	jobz := EigenJobOption.VALUES_ONLY,
	range := EigenRangeOption.ALL,
	uplo := MatrixRegion.Upper,
	n: int, // Matrix dimension
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
	assert(len(ap) >= n * (n + 1) / 2, "Packed array A too small")
	assert(len(bp) >= n * (n + 1) / 2, "Packed array B too small")
	assert(len(w) >= n, "Eigenvalue array too small")
	assert(len(work) >= 8 * n, "Workspace too small")
	assert(len(iwork) >= 5 * n, "Integer workspace too small")
	assert(len(ifail) >= n, "Failure array too small")

	itype_int := Blas_Int(itype)
	jobz_c := eigen_job_to_char(jobz)
	range_c := eigen_range_to_char(range)
	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)

	// Range parameters
	vl_val := vl
	vu_val := vu
	il_int := Blas_Int(il)
	iu_int := Blas_Int(iu if iu > 0 else n)
	abstol_val := abstol
	m_int: Blas_Int

	// Handle eigenvector matrix
	ldz: Blas_Int = 1
	z_ptr: rawptr = nil
	if jobz == .VALUES_VECTORS && Z != nil {
		max_eigenvectors := n
		if range == .INDEX {
			max_eigenvectors = iu - il + 1
		}
		assert(Z.rows >= n && Z.cols >= max_eigenvectors, "Eigenvector matrix too small")
		ldz = Blas_Int(Z.ld)
		z_ptr = raw_data(Z.data)
	}

	when T == f32 {
		lapack.sspgvx_(
			&itype_int,
			&jobz_c,
			&range_c,
			&uplo_c,
			&n_int,
			raw_data(ap),
			raw_data(bp),
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
			raw_data(iwork),
			raw_data(ifail),
			&info,
			1,
			1,
			1,
		)
	} else when T == f64 {
		lapack.dspgvx_(
			&itype_int,
			&jobz_c,
			&range_c,
			&uplo_c,
			&n_int,
			raw_data(ap),
			raw_data(bp),
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

// Compute selective packed generalized eigenvalues for complex64/complex128
m_compute_packed_generalized_eigenvalues_selective_c64_c128 :: proc(
	ap: []$T, // Packed Hermitian matrix A (modified on output)
	bp: []T, // Packed Hermitian matrix B (modified to Cholesky factor)
	w: []$R, // Pre-allocated eigenvalues array (size n)
	Z: ^Matrix(T) = nil, // Eigenvector matrix (optional, n x max_eigenvectors)
	work: []T, // Pre-allocated workspace (size 2*n)
	rwork: []R, // Pre-allocated real workspace (size 7*n)
	iwork: []Blas_Int, // Pre-allocated integer workspace (size 5*n)
	ifail: []Blas_Int, // Pre-allocated failure indices (size n)
	itype := GeneralizedProblemType.AX_LBX,
	jobz := EigenJobOption.VALUES_ONLY,
	range := EigenRangeOption.ALL,
	uplo := MatrixRegion.Upper,
	n: int, // Matrix dimension
	vl: R, // Lower bound (if range == VALUE)
	vu: R, // Upper bound (if range == VALUE)
	il: int = 1, // Lower index (if range == INDEX, 1-based)
	iu: int = 0, // Upper index (if range == INDEX, 1-based, 0=n)
	abstol: R, // Absolute tolerance
) -> (
	m: int,
	info: Info,
	ok: bool, // Number of eigenvalues found
) where T == complex64 || T == complex128,
	R == real_type_of(T) {
	assert(len(ap) >= n * (n + 1) / 2, "Packed array A too small")
	assert(len(bp) >= n * (n + 1) / 2, "Packed array B too small")
	assert(len(w) >= n, "Eigenvalue array too small")
	assert(len(work) >= 2 * n, "Workspace too small")
	assert(len(rwork) >= 7 * n, "Real workspace too small")
	assert(len(iwork) >= 5 * n, "Integer workspace too small")
	assert(len(ifail) >= n, "Failure array too small")

	itype_int := Blas_Int(itype)
	jobz_c := eigen_job_to_char(jobz)
	range_c := eigen_range_to_char(range)
	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)

	// Range parameters
	vl_val := vl
	vu_val := vu
	il_int := Blas_Int(il)
	iu_int := Blas_Int(iu if iu > 0 else n)
	abstol_val := abstol
	m_int: Blas_Int

	// Handle eigenvector matrix
	ldz: Blas_Int = 1
	z_ptr: rawptr = nil
	if jobz == .VALUES_VECTORS && Z != nil {
		max_eigenvectors := n
		if range == .INDEX {
			max_eigenvectors = iu - il + 1
		}
		assert(Z.rows >= n && Z.cols >= max_eigenvectors, "Eigenvector matrix too small")
		ldz = Blas_Int(Z.ld)
		z_ptr = raw_data(Z.data)
	}

	when T == complex64 {
		lapack.chpgvx_(
			&itype_int,
			&jobz_c,
			&range_c,
			&uplo_c,
			&n_int,
			cast(^lapack.complex)raw_data(ap),
			cast(^lapack.complex)raw_data(bp),
			&vl_val,
			&vu_val,
			&il_int,
			&iu_int,
			&abstol_val,
			&m_int,
			raw_data(w),
			cast(^lapack.complex)z_ptr,
			&ldz,
			cast(^lapack.complex)raw_data(work),
			raw_data(rwork),
			raw_data(iwork),
			raw_data(ifail),
			&info,
			1,
			1,
			1,
		)
	} else when T == complex128 {
		lapack.zhpgvx_(
			&itype_int,
			&jobz_c,
			&range_c,
			&uplo_c,
			&n_int,
			cast(^lapack.doublecomplex)raw_data(ap),
			cast(^lapack.doublecomplex)raw_data(bp),
			&vl_val,
			&vu_val,
			&il_int,
			&iu_int,
			&abstol_val,
			&m_int,
			raw_data(w),
			cast(^lapack.doublecomplex)z_ptr,
			&ldz,
			cast(^lapack.doublecomplex)raw_data(work),
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

// Procedure group for selective packed generalized eigenvalue computation
m_compute_packed_generalized_eigenvalues_selective :: proc {
	m_compute_packed_generalized_eigenvalues_selective_f32_f64,
	m_compute_packed_generalized_eigenvalues_selective_c64_c128,
}

// ============================================================================
// PACKED SYMMETRIC ITERATIVE REFINEMENT
// ============================================================================

// Refine packed symmetric solution for f32/f64
m_refine_packed_symmetric_f32_f64 :: proc(
	ap: []$T, // Original packed matrix
	afp: []T, // Factored packed matrix
	ipiv: []Blas_Int, // Pivot indices from factorization
	b: ^Matrix(T), // Right-hand side matrix
	x: ^Matrix(T), // Solution matrix (refined on output)
	ferr: []T, // Pre-allocated forward error bounds (size nrhs)
	berr: []T, // Pre-allocated backward error bounds (size nrhs)
	work: []T, // Pre-allocated workspace (size 3*n)
	iwork: []Blas_Int, // Pre-allocated integer workspace (size n)
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 {
	n := b.rows
	nrhs := b.cols
	assert(len(ap) >= n * (n + 1) / 2, "Packed array too small")
	assert(len(afp) >= n * (n + 1) / 2, "Factored packed array too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(x.rows >= n && x.cols >= nrhs, "Solution matrix too small")
	assert(len(ferr) >= nrhs, "Forward error array too small")
	assert(len(berr) >= nrhs, "Backward error array too small")
	assert(len(work) >= 3 * n, "Workspace too small")
	assert(len(iwork) >= n, "Integer workspace too small")

	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	ldb := Blas_Int(b.ld)
	ldx := Blas_Int(x.ld)

	when T == f32 {
		lapack.ssprfs_(&uplo_c, &n_int, &nrhs_int, raw_data(ap), raw_data(afp), raw_data(ipiv), b.data, &ldb, x.data, &ldx, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(iwork), &info, 1)
	} else when T == f64 {
		lapack.dsprfs_(&uplo_c, &n_int, &nrhs_int, raw_data(ap), raw_data(afp), raw_data(ipiv), b.data, &ldb, x.data, &ldx, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(iwork), &info, 1)
	}

	ok = info == 0
	return info, ok
}

// Refine packed symmetric solution for complex64/complex128
m_refine_packed_symmetric_c64_c128 :: proc(
	ap: []$T, // Original packed matrix
	afp: []T, // Factored packed matrix
	ipiv: []Blas_Int, // Pivot indices from factorization
	b: ^Matrix(T), // Right-hand side matrix
	x: ^Matrix(T), // Solution matrix (refined on output)
	ferr: []$R, // Pre-allocated forward error bounds (size nrhs)
	berr: []R, // Pre-allocated backward error bounds (size nrhs)
	work: []T, // Pre-allocated workspace (size 2*n)
	rwork: []R, // Pre-allocated real workspace (size n)
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where T == complex64 || T == complex128,
	R == real_type_of(T) {
	n := b.rows
	nrhs := b.cols
	assert(len(ap) >= n * (n + 1) / 2, "Packed array too small")
	assert(len(afp) >= n * (n + 1) / 2, "Factored packed array too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(x.rows >= n && x.cols >= nrhs, "Solution matrix too small")
	assert(len(ferr) >= nrhs, "Forward error array too small")
	assert(len(berr) >= nrhs, "Backward error array too small")
	assert(len(work) >= 2 * n, "Workspace too small")
	assert(len(rwork) >= n, "Real workspace too small")

	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	ldb := Blas_Int(b.ld)
	ldx := Blas_Int(x.ld)

	when T == complex64 {
		lapack.csprfs_(
			&uplo_c,
			&n_int,
			&nrhs_int,
			cast(^lapack.complex)raw_data(ap),
			cast(^lapack.complex)raw_data(afp),
			raw_data(ipiv),
			cast(^lapack.complex)b.data,
			&ldb,
			cast(^lapack.complex)x.data,
			&ldx,
			raw_data(ferr),
			raw_data(berr),
			cast(^lapack.complex)raw_data(work),
			raw_data(rwork),
			&info,
			1,
		)
	} else when T == complex128 {
		lapack.zsprfs_(
			&uplo_c,
			&n_int,
			&nrhs_int,
			cast(^lapack.doublecomplex)raw_data(ap),
			cast(^lapack.doublecomplex)raw_data(afp),
			raw_data(ipiv),
			cast(^lapack.doublecomplex)b.data,
			&ldb,
			cast(^lapack.doublecomplex)x.data,
			&ldx,
			raw_data(ferr),
			raw_data(berr),
			cast(^lapack.doublecomplex)raw_data(work),
			raw_data(rwork),
			&info,
			1,
		)
	}

	ok = info == 0
	return info, ok
}

// Procedure group for packed symmetric refinement
m_refine_packed_symmetric :: proc {
	m_refine_packed_symmetric_f32_f64,
	m_refine_packed_symmetric_c64_c128,
}
