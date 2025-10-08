package openblas

import lapack "./f77"
import "base:builtin"

// ===================================================================================
// BANDED MATRIX EIGENVALUE PROBLEMS
//
// This file provides eigenvalue computation for banded matrices:
// - Symmetric/Hermitian banded eigenvalue problems (SB/HB routines)
// - Generalized eigenvalue problems for banded matrices
// - Reduction to tridiagonal form and specialized solvers
// - Expert drivers with subset selection and improved accuracy
//
// All routines follow the unified API patterns with real/complex overloading.
// ===================================================================================

// ===================================================================================
// PROC GROUPS FOR OVERLOADING
// ===================================================================================

// Standard eigenvalue computation for symmetric/Hermitian banded matrices
band_eigen :: proc {
	band_eigen_real,
	band_eigen_complex,
}

// Eigenvalue computation using divide-and-conquer algorithm
band_eigen_dc :: proc {
	band_eigen_dc_real,
	band_eigen_dc_complex,
}

// Expert driver with subset selection and improved accuracy
band_eigen_expert :: proc {
	band_eigen_expert_real,
	band_eigen_expert_complex,
}

// Reduce banded symmetric/Hermitian matrix to tridiagonal form with Q generation (SBTRD/HBTRD)
band_to_tridiagonal_with_q :: proc {
	band_to_tridiagonal_with_q_real,
	band_to_tridiagonal_with_q_complex,
}

// Generalized eigenvalue problem: A*x = lambda*B*x
band_eigen_generalized :: proc {
	band_eigen_generalized_real,
	band_eigen_generalized_complex,
}

// Generalized eigenvalue problem using divide-and-conquer: A*x = lambda*B*x
band_eigen_generalized_dc :: proc {
	band_eigen_generalized_dc_real,
	band_eigen_generalized_dc_complex,
}

// Reduce generalized symmetric/Hermitian banded eigenvalue problem to standard form
band_reduce_generalized :: proc {
	band_reduce_generalized_real,
	band_reduce_generalized_complex,
}

// Generalized eigenvalue problem with expert driver (selected eigenvalues): A*x = lambda*B*x
band_eigen_generalized_expert :: proc {
	band_eigen_generalized_expert_real,
	band_eigen_generalized_expert_complex,
}

// ===================================================================================
// WORKSPACE AND SIZE QUERIES
// ===================================================================================

// Query result sizes for eigenvalue computation
query_result_sizes_band_eigen :: proc(AB: ^BandedMatrix($T), compute_vectors: bool) -> (w_size: int, z_rows: int, z_cols: int) where is_float(T) || is_complex(T) {
	n := int(AB.cols)
	w_size = n
	if compute_vectors {
		z_rows = n
		z_cols = n
	}
	return
}

// Query workspace for standard eigenvalue computation
query_workspace_band_eigen :: proc(AB: ^BandedMatrix($T), compute_vectors: bool) -> (work: int, rwork: int) where is_float(T) || is_complex(T) {
	n := int(AB.cols)
	when is_float(T) {
		if compute_vectors {
			return 3 * n - 2, 0
		} else {
			return 1, 0
		}
	} else when is_complex(T) {
		if compute_vectors {
			return n, max(1, 3 * n - 2)
		} else {
			return 1, max(1, n)
		}
	}
}

// Query workspace for divide-and-conquer algorithm
query_workspace_band_eigen_dc :: proc(AB: ^BandedMatrix($T), compute_vectors: bool) -> (work: int, rwork: int, iwork: int) where is_float(T) || is_complex(T) {
	n := int(AB.cols)
	when is_float(T) {
		if compute_vectors {
			return 1 + 6 * n + 2 * n * n, 1 + 5 * n + 2 * n * n, 3 + 5 * n
		} else {
			return 2 * n, 0, 1
		}
	} else when is_complex(T) {
		if compute_vectors {
			return 1 + 5 * n + 2 * n * n, 1 + 5 * n + 2 * n * n, 3 + 5 * n
		} else {
			return n, n, 1
		}
	}
}

// Query workspace for expert eigenvalue computation with subset selection
query_workspace_band_eigen_expert_subset :: proc(AB: ^BandedMatrix($T)) -> (work: int, rwork: int, iwork: int) where is_float(T) || is_complex(T) {
	n := int(AB.cols)
	when is_float(T) {
		return 7 * n, 0, 5 * n
	} else when is_complex(T) {
		return n, 7 * n, 5 * n
	}
}

// Query workspace for tridiagonal reduction with Q generation
query_workspace_band_to_tridiagonal_with_q :: proc(AB: ^BandedMatrix($T)) -> (work: int, rwork: int) where is_float(T) || is_complex(T) {
	n := int(AB.cols)
	when is_float(T) {
		return n, 0
	} else when is_complex(T) {
		return n, max(1, n - 1)
	}
}

// ===================================================================================
// STANDARD EIGENVALUE COMPUTATION (SBEV/HBEV)
// ===================================================================================

// Compute eigenvalues and optionally eigenvectors of symmetric/Hermitian banded matrix (real version)
band_eigen_real :: proc(
	jobz: VectorOption,
	uplo: MatrixRegion,
	AB: ^BandedMatrix($T), // Banded matrix (input/output - destroyed)
	w: []T, // Pre-allocated eigenvalues array (size n)
	Z: ^Matrix(T), // Pre-allocated eigenvectors matrix (optional)
	work: []T, // Pre-allocated workspace
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	n := AB.cols
	kd := AB.kl
	ldab := AB.ldab

	assert(len(w) >= int(n), "Eigenvalues array too small")
	min_work := max(1, 3 * int(n) - 2) if jobz == .Vectors else 1
	assert(len(work) >= min_work, "Work array too small")

	ldz := Blas_Int(1)
	z_ptr: ^T = nil
	if jobz == .Vectors && Z != nil {
		assert(int(Z.rows) >= int(n) && int(Z.cols) >= int(n), "Eigenvectors matrix too small")
		ldz = Z.ld
		z_ptr = raw_data(Z.data)
	}

	jobz_c := cast(u8)jobz
	uplo_c := cast(u8)uplo

	when T == f32 {
		lapack.ssbev_(&jobz_c, &uplo_c, &n, &kd, raw_data(AB.data), &ldab, raw_data(w), z_ptr, &ldz, raw_data(work), &info)
	} else when T == f64 {
		lapack.dsbev_(&jobz_c, &uplo_c, &n, &kd, raw_data(AB.data), &ldab, raw_data(w), z_ptr, &ldz, raw_data(work), &info)
	}

	return info, info == 0
}

// Compute eigenvalues and optionally eigenvectors of Hermitian banded matrix (complex version)
band_eigen_complex :: proc(
	jobz: VectorOption,
	uplo: MatrixRegion,
	AB: ^BandedMatrix($Cmplx), // Banded matrix (input/output - destroyed)
	w: []$Real, // Pre-allocated eigenvalues array (size n) - always real
	Z: ^Matrix(Cmplx), // Pre-allocated eigenvectors matrix (optional)
	work: []Cmplx, // Pre-allocated workspace
	rwork: []Real, // Pre-allocated real workspace
) -> (
	info: Info,
	ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
	n := AB.cols
	kd := AB.kl
	ldab := AB.ldab

	assert(len(w) >= int(n), "Eigenvalues array too small")
	min_work := int(n) if jobz == .Vectors else 1
	min_rwork := max(1, 3 * int(n) - 2) if jobz == .Vectors else max(1, int(n))
	assert(len(work) >= min_work, "Work array too small")
	assert(len(rwork) >= min_rwork, "Real work array too small")

	ldz := Blas_Int(1)
	z_ptr: ^Cmplx = nil
	if jobz == .Vectors && Z != nil {
		assert(int(Z.rows) >= int(n) && int(Z.cols) >= int(n), "Eigenvectors matrix too small")
		ldz = Z.ld
		z_ptr = raw_data(Z.data)
	}

	jobz_c := cast(u8)jobz
	uplo_c := cast(u8)uplo

	when Cmplx == complex64 {
		lapack.chbev_(&jobz_c, &uplo_c, &n, &kd, raw_data(AB.data), &ldab, raw_data(w), z_ptr, &ldz, raw_data(work), raw_data(rwork), &info)
	} else when Cmplx == complex128 {
		lapack.zhbev_(&jobz_c, &uplo_c, &n, &kd, raw_data(AB.data), &ldab, raw_data(w), z_ptr, &ldz, raw_data(work), raw_data(rwork), &info)
	}

	return info, info == 0
}

// ===================================================================================
// DIVIDE-AND-CONQUER ALGORITHM (SBEVD/HBEVD)
// ===================================================================================

// Compute eigenvalues and optionally eigenvectors using divide-and-conquer (real version)
band_eigen_dc_real :: proc(
	jobz: VectorOption,
	uplo: MatrixRegion,
	AB: ^BandedMatrix($T), // Banded matrix (input/output - destroyed)
	w: []T, // Pre-allocated eigenvalues array (size n)
	Z: ^Matrix(T), // Pre-allocated eigenvectors matrix (optional)
	work: []T, // Pre-allocated workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	n := AB.cols
	kd := AB.kl
	ldab := AB.ldab

	assert(len(w) >= int(n), "Eigenvalues array too small")

	ldz := Blas_Int(1)
	z_ptr: ^T = nil
	if jobz == .Vectors && Z != nil {
		assert(int(Z.rows) >= int(n) && int(Z.cols) >= int(n), "Eigenvectors matrix too small")
		ldz = Z.ld
		z_ptr = raw_data(Z.data)
	}

	jobz_c := cast(u8)jobz
	uplo_c := cast(u8)uplo
	lwork := Blas_Int(len(work))
	liwork := Blas_Int(len(iwork))

	when T == f32 {
		lapack.ssbevd_(&jobz_c, &uplo_c, &n, &kd, raw_data(AB.data), &ldab, raw_data(w), z_ptr, &ldz, raw_data(work), &lwork, raw_data(iwork), &liwork, &info)
	} else when T == f64 {
		lapack.dsbevd_(&jobz_c, &uplo_c, &n, &kd, raw_data(AB.data), &ldab, raw_data(w), z_ptr, &ldz, raw_data(work), &lwork, raw_data(iwork), &liwork, &info)
	}

	return info, info == 0
}

// Compute eigenvalues and optionally eigenvectors using divide-and-conquer (complex version)
band_eigen_dc_complex :: proc(
	jobz: VectorOption,
	uplo: MatrixRegion,
	AB: ^BandedMatrix($Cmplx), // Banded matrix (input/output - destroyed)
	w: []$Real, // Pre-allocated eigenvalues array (size n) - always real
	Z: ^Matrix(Cmplx), // Pre-allocated eigenvectors matrix (optional)
	work: []Cmplx, // Pre-allocated workspace
	rwork: []Real, // Pre-allocated real workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace
) -> (
	info: Info,
	ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
	n := AB.cols
	kd := AB.kl
	ldab := AB.ldab

	assert(len(w) >= int(n), "Eigenvalues array too small")

	ldz := Blas_Int(1)
	z_ptr: ^Cmplx = nil
	if jobz == .Vectors && Z != nil {
		assert(int(Z.rows) >= int(n) && int(Z.cols) >= int(n), "Eigenvectors matrix too small")
		ldz = Z.ld
		z_ptr = raw_data(Z.data)
	}

	jobz_c := cast(u8)jobz
	uplo_c := cast(u8)uplo
	lwork := Blas_Int(len(work))
	lrwork := Blas_Int(len(rwork))
	liwork := Blas_Int(len(iwork))

	when Cmplx == complex64 {
		lapack.chbevd_(&jobz_c, &uplo_c, &n, &kd, raw_data(AB.data), &ldab, raw_data(w), z_ptr, &ldz, raw_data(work), &lwork, raw_data(rwork), &lrwork, raw_data(iwork), &liwork, &info)
	} else when Cmplx == complex128 {
		lapack.zhbevd_(&jobz_c, &uplo_c, &n, &kd, raw_data(AB.data), &ldab, raw_data(w), z_ptr, &ldz, raw_data(work), &lwork, raw_data(rwork), &lrwork, raw_data(iwork), &liwork, &info)
	}

	return info, info == 0
}

// ===================================================================================
// EXPERT EIGENVALUE COMPUTATION (SBEVX/HBEVX)
// ===================================================================================

// Compute selected eigenvalues and optionally eigenvectors (real version)
band_eigen_expert_real :: proc(
	jobz: EigenJobOption,
	range: EigenRangeOption,
	uplo: MatrixRegion,
	AB: ^BandedMatrix($T), // Banded matrix (input/output - destroyed)
	Q: ^BandedMatrix(T) = nil, // Optional Q matrix from reduction (input/output)
	vl: T, // Lower bound of eigenvalue range (if range == .VALUE)
	vu: T, // Upper bound of eigenvalue range (if range == .VALUE)
	il: int = 1, // Lower index of eigenvalues to compute (if range == .INDEX)
	iu: int = 1, // Upper index of eigenvalues to compute (if range == .INDEX)
	abstol: T, // Absolute error tolerance for eigenvalues
	w: []T, // Pre-allocated eigenvalues array (size n)
	Z: ^Matrix(T), // Pre-allocated eigenvectors matrix (optional)
	m: ^Blas_Int, // Number of eigenvalues found (output)
	work: []T, // Pre-allocated workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace
	ifail: []Blas_Int, // Pre-allocated failure indices (size n)
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	n := AB.cols
	kd := AB.kl
	ldab := AB.ldab

	assert(len(w) >= int(n), "Eigenvalues array too small")
	assert(len(work) >= 7 * int(n), "Work array too small")
	assert(len(iwork) >= 5 * int(n), "Integer work array too small")
	assert(len(ifail) >= int(n), "IFAIL array too small")

	ldq := Blas_Int(1)
	q_ptr: ^T = nil
	if Q != nil {
		ldq = Q.ldab
		q_ptr = raw_data(Q.data)
	}

	ldz := Blas_Int(1)
	z_ptr: ^T = nil
	if jobz == .VALUES_AND_VECTORS && Z != nil {
		assert(int(Z.rows) >= int(n) && int(Z.cols) >= int(n), "Eigenvectors matrix too small")
		ldz = Z.ld
		z_ptr = raw_data(Z.data)
	}

	jobz_c := cast(u8)jobz
	range_c := cast(u8)range
	uplo_c := cast(u8)uplo
	vl_val := vl
	vu_val := vu
	il_val := Blas_Int(il)
	iu_val := Blas_Int(iu)
	abstol := abstol

	when T == f32 {
		lapack.ssbevx_(&jobz_c, &range_c, &uplo_c, &n, &kd, raw_data(AB.data), &ldab, q_ptr, &ldq, &vl_val, &vu_val, &il_val, &iu_val, &abstol, m, raw_data(w), z_ptr, &ldz, raw_data(work), raw_data(iwork), raw_data(ifail), &info)
	} else when T == f64 {
		lapack.dsbevx_(&jobz_c, &range_c, &uplo_c, &n, &kd, raw_data(AB.data), &ldab, q_ptr, &ldq, &vl_val, &vu_val, &il_val, &iu_val, &abstol, m, raw_data(w), z_ptr, &ldz, raw_data(work), raw_data(iwork), raw_data(ifail), &info)
	}

	return info, info == 0
}

// Compute selected eigenvalues and optionally eigenvectors (complex version)
band_eigen_expert_complex :: proc(
	jobz: EigenJobOption,
	range: EigenRangeOption,
	uplo: MatrixRegion,
	AB: ^BandedMatrix($Cmplx), // Banded matrix (input/output - destroyed)
	Q: ^BandedMatrix(Cmplx), // Optional Q matrix from reduction (input/output)
	vl: $Real, // Lower bound of eigenvalue range (if range == .VALUE)
	vu: Real, // Upper bound of eigenvalue range (if range == .VALUE)
	il: int = 1, // Lower index of eigenvalues to compute (if range == .INDEX)
	iu: int = 1, // Upper index of eigenvalues to compute (if range == .INDEX)
	abstol: Real, // Absolute error tolerance for eigenvalues
	w: []Real, // Pre-allocated eigenvalues array (size n) - always real
	Z: ^Matrix(Cmplx), // Pre-allocated eigenvectors matrix (optional)
	m: ^Blas_Int, // Number of eigenvalues found (output)
	work: []Cmplx, // Pre-allocated workspace
	rwork: []Real, // Pre-allocated real workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace
	ifail: []Blas_Int, // Pre-allocated failure indices (size n)
) -> (
	info: Info,
	ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
	n := AB.cols
	kd := AB.kl
	ldab := AB.ldab

	assert(len(w) >= int(n), "Eigenvalues array too small")
	assert(len(work) >= int(n), "Work array too small")
	assert(len(rwork) >= 7 * int(n), "Real work array too small")
	assert(len(iwork) >= 5 * int(n), "Integer work array too small")
	assert(len(ifail) >= int(n), "IFAIL array too small")

	ldq := Blas_Int(1)
	q_ptr: ^Cmplx = nil
	if Q != nil {
		ldq = Q.ldab
		q_ptr = raw_data(Q.data)
	}

	ldz := Blas_Int(1)
	z_ptr: ^Cmplx = nil
	if jobz == .VALUES_AND_VECTORS && Z != nil {
		assert(int(Z.rows) >= int(n) && int(Z.cols) >= int(n), "Eigenvectors matrix too small")
		ldz = Z.ld
		z_ptr = raw_data(Z.data)
	}

	jobz_c := cast(u8)jobz
	range_c := cast(u8)range
	uplo_c := cast(u8)uplo
	vl_val := vl
	vu_val := vu
	il_val := Blas_Int(il)
	iu_val := Blas_Int(iu)
	abstol := abstol

	when Cmplx == complex64 {
		lapack.chbevx_(&jobz_c, &range_c, &uplo_c, &n, &kd, raw_data(AB.data), &ldab, q_ptr, &ldq, &vl_val, &vu_val, &il_val, &iu_val, &abstol, m, raw_data(w), z_ptr, &ldz, raw_data(work), raw_data(rwork), raw_data(iwork), raw_data(ifail), &info)
	} else when Cmplx == complex128 {
		lapack.zhbevx_(&jobz_c, &range_c, &uplo_c, &n, &kd, raw_data(AB.data), &ldab, q_ptr, &ldq, &vl_val, &vu_val, &il_val, &iu_val, &abstol, m, raw_data(w), z_ptr, &ldz, raw_data(work), raw_data(rwork), raw_data(iwork), raw_data(ifail), &info)
	}

	return info, info == 0
}

// ===================================================================================
// REDUCTION TO TRIDIAGONAL FORM (SBTRD/HBTRD)
// ===================================================================================

// Reduce banded symmetric/Hermitian matrix to tridiagonal form (real version)
band_to_tridiagonal_with_q_real :: proc(
	vect: VectorOption,
	uplo: MatrixRegion,
	AB: ^BandedMatrix($T), // Banded matrix (input/output - reduced to tridiagonal)
	D: []T, // Pre-allocated diagonal elements (size n)
	E: []T, // Pre-allocated off-diagonal elements (size n-1)
	Q: ^Matrix(T), // Pre-allocated transformation matrix (optional)
	work: []T, // Pre-allocated workspace
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	n := AB.cols
	kd := AB.kl
	ldab := AB.ldab

	assert(len(D) >= int(n), "D array too small")
	assert(len(E) >= int(n - 1), "E array too small")
	assert(len(work) >= int(n), "Work array too small")

	ldq := Blas_Int(1)
	q_ptr: ^T = nil
	if vect == .FORM_VECTORS && Q != nil {
		assert(int(Q.rows) >= int(n) && int(Q.cols) >= int(n), "Q matrix too small")
		ldq = Q.ld
		q_ptr = raw_data(Q.data)
	}

	vect_c := cast(u8)vect
	uplo_c := cast(u8)uplo

	when T == f32 {
		lapack.ssbtrd_(&vect_c, &uplo_c, &n, &kd, raw_data(AB.data), &ldab, raw_data(D), raw_data(E), q_ptr, &ldq, raw_data(work), &info)
	} else when T == f64 {
		lapack.dsbtrd_(&vect_c, &uplo_c, &n, &kd, raw_data(AB.data), &ldab, raw_data(D), raw_data(E), q_ptr, &ldq, raw_data(work), &info)
	}

	return info, info == 0
}

// Reduce banded Hermitian matrix to tridiagonal form (complex version)
band_to_tridiagonal_with_q_complex :: proc(
	vect: VectorOption,
	uplo: MatrixRegion,
	AB: ^BandedMatrix($Cmplx), // Banded matrix (input/output - reduced to tridiagonal)
	D: []$Real, // Pre-allocated diagonal elements (size n) - always real
	E: []Real, // Pre-allocated off-diagonal elements (size n-1) - always real
	Q: ^Matrix(Cmplx), // Pre-allocated transformation matrix (optional)
	work: []Cmplx, // Pre-allocated workspace
) -> (
	info: Info,
	ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
	n := AB.cols
	kd := AB.kl
	ldab := AB.ldab

	assert(len(D) >= int(n), "D array too small")
	assert(len(E) >= int(n - 1), "E array too small")
	assert(len(work) >= int(n), "Work array too small")

	ldq := Blas_Int(1)
	q_ptr: ^Cmplx = nil
	if vect == .FORM_VECTORS && Q != nil {
		assert(int(Q.rows) >= int(n) && int(Q.cols) >= int(n), "Q matrix too small")
		ldq = Q.ld
		q_ptr = raw_data(Q.data)
	}

	vect_c := cast(u8)vect
	uplo_c := cast(u8)uplo

	when Cmplx == complex64 {
		lapack.chbtrd_(&vect_c, &uplo_c, &n, &kd, raw_data(AB.data), &ldab, raw_data(D), raw_data(E), q_ptr, &ldq, raw_data(work), &info)
	} else when Cmplx == complex128 {
		lapack.zhbtrd_(&vect_c, &uplo_c, &n, &kd, raw_data(AB.data), &ldab, raw_data(D), raw_data(E), q_ptr, &ldq, raw_data(work), &info)
	}

	return info, info == 0
}

// ===================================================================================
// GENERALIZED EIGENVALUE PROBLEMS (SBGV/HBGV)
// ===================================================================================

// Solve generalized eigenvalue problem A*x = lambda*B*x (real version)
band_eigen_generalized_real :: proc(
	jobz: EigenJobOption,
	uplo: MatrixRegion,
	AB: ^BandedMatrix($T), // Banded matrix A (input/output - destroyed)
	BB: ^BandedMatrix(T), // Banded matrix B (input/output - destroyed)
	w: []T, // Pre-allocated eigenvalues array (size n)
	Z: ^Matrix(T), // Pre-allocated eigenvectors matrix (optional)
	work: []T, // Pre-allocated workspace
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	n := AB.cols
	ka := AB.kl // Bandwidth of A
	kb := BB.kl // Bandwidth of B
	ldab := AB.ldab
	ldbb := BB.ldab

	// Validate inputs
	assert(len(w) >= int(n), "Eigenvalues array too small")
	assert(len(work) >= 3 * int(n), "Work array too small")

	ldz := Blas_Int(1)
	z_ptr: ^T = nil
	if jobz == .VALUES_AND_VECTORS && Z != nil {
		assert(int(Z.rows) >= int(n) && int(Z.cols) >= int(n), "Eigenvectors matrix too small")
		ldz = Z.ld
		z_ptr = raw_data(Z.data)
	}

	jobz_c := cast(u8)jobz
	uplo_c := cast(u8)uplo

	when T == f32 {
		lapack.ssbgv_(&jobz_c, &uplo_c, &n, &ka, &kb, raw_data(AB.data), &ldab, raw_data(BB.data), &ldbb, raw_data(w), z_ptr, &ldz, raw_data(work), &info)
	} else when T == f64 {
		lapack.dsbgv_(&jobz_c, &uplo_c, &n, &ka, &kb, raw_data(AB.data), &ldab, raw_data(BB.data), &ldbb, raw_data(w), z_ptr, &ldz, raw_data(work), &info)
	}

	return info, info == 0
}

// Solve generalized eigenvalue problem A*x = lambda*B*x (complex version)
band_eigen_generalized_complex :: proc(
	jobz: EigenJobOption,
	uplo: MatrixRegion,
	AB: ^BandedMatrix($Cmplx), // Banded matrix A (input/output - destroyed)
	BB: ^BandedMatrix(Cmplx), // Banded matrix B (input/output - destroyed)
	w: []$Real, // Pre-allocated eigenvalues array (size n) - always real
	Z: ^Matrix(Cmplx), // Pre-allocated eigenvectors matrix (optional)
	work: []Cmplx, // Pre-allocated workspace
	rwork: []Real, // Pre-allocated real workspace
) -> (
	info: Info,
	ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
	n := AB.cols
	ka := AB.kl // Bandwidth of A
	kb := BB.kl // Bandwidth of B
	ldab := AB.ldab
	ldbb := BB.ldab

	assert(len(w) >= int(n), "Eigenvalues array too small")
	assert(len(work) >= int(n), "Work array too small")
	assert(len(rwork) >= 3 * int(n), "Real work array too small")

	ldz := Blas_Int(1)
	z_ptr: ^Cmplx = nil
	if jobz == .VALUES_AND_VECTORS && Z != nil {
		assert(int(Z.rows) >= int(n) && int(Z.cols) >= int(n), "Eigenvectors matrix too small")
		ldz = Z.ld
		z_ptr = raw_data(Z.data)
	}

	jobz_c := cast(u8)jobz
	uplo_c := cast(u8)uplo

	when Cmplx == complex64 {
		lapack.chbgv_(&jobz_c, &uplo_c, &n, &ka, &kb, raw_data(AB.data), &ldab, raw_data(BB.data), &ldbb, raw_data(w), z_ptr, &ldz, raw_data(work), raw_data(rwork), &info)
	} else when Cmplx == complex128 {
		lapack.zhbgv_(&jobz_c, &uplo_c, &n, &ka, &kb, raw_data(AB.data), &ldab, raw_data(BB.data), &ldbb, raw_data(w), z_ptr, &ldz, raw_data(work), raw_data(rwork), &info)
	}

	return info, info == 0
}

// ===================================================================================
// GENERALIZED EIGENVALUE PROBLEMS - DIVIDE AND CONQUER (SBGVD/HBGVD)
// ===================================================================================

// Solve generalized eigenvalue problem using divide-and-conquer (real version)
band_eigen_generalized_dc_real :: proc(
	jobz: EigenJobOption,
	uplo: MatrixRegion,
	AB: ^BandedMatrix($T), // Banded matrix A (input/output - destroyed)
	BB: ^BandedMatrix(T), // Banded matrix B (input/output - destroyed)
	w: []T, // Pre-allocated eigenvalues array (size n)
	Z: ^Matrix(T), // Pre-allocated eigenvectors matrix (optional)
	work: []T, // Pre-allocated workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	n := AB.cols
	ka := AB.kl // Bandwidth of A
	kb := BB.kl // Bandwidth of B
	ldab := AB.ldab
	ldbb := BB.ldab

	assert(len(w) >= int(n), "Eigenvalues array too small")

	ldz := Blas_Int(1)
	z_ptr: ^T = nil
	if jobz == .VALUES_AND_VECTORS && Z != nil {
		assert(int(Z.rows) >= int(n) && int(Z.cols) >= int(n), "Eigenvectors matrix too small")
		ldz = Z.ld
		z_ptr = raw_data(Z.data)
	}

	jobz_c := cast(u8)jobz
	uplo_c := cast(u8)uplo
	lwork := Blas_Int(len(work))
	liwork := Blas_Int(len(iwork))

	when T == f32 {
		lapack.ssbgvd_(&jobz_c, &uplo_c, &n, &ka, &kb, raw_data(AB.data), &ldab, raw_data(BB.data), &ldbb, raw_data(w), z_ptr, &ldz, raw_data(work), &lwork, raw_data(iwork), &liwork, &info)
	} else when T == f64 {
		lapack.dsbgvd_(&jobz_c, &uplo_c, &n, &ka, &kb, raw_data(AB.data), &ldab, raw_data(BB.data), &ldbb, raw_data(w), z_ptr, &ldz, raw_data(work), &lwork, raw_data(iwork), &liwork, &info)
	}

	return info, info == 0
}

// Solve generalized eigenvalue problem using divide-and-conquer (complex version)
band_eigen_generalized_dc_complex :: proc(
	jobz: EigenJobOption,
	uplo: MatrixRegion,
	AB: ^BandedMatrix($Cmplx), // Banded matrix A (input/output - destroyed)
	BB: ^BandedMatrix(Cmplx), // Banded matrix B (input/output - destroyed)
	w: []$Real, // Pre-allocated eigenvalues array (size n) - always real
	Z: ^Matrix(Cmplx), // Pre-allocated eigenvectors matrix (optional)
	work: []Cmplx, // Pre-allocated workspace
	rwork: []Real, // Pre-allocated real workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace
) -> (
	info: Info,
	ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
	n := AB.cols
	ka := AB.kl // Bandwidth of A
	kb := BB.kl // Bandwidth of B
	ldab := AB.ldab
	ldbb := BB.ldab

	assert(len(w) >= int(n), "Eigenvalues array too small")

	ldz := Blas_Int(1)
	z_ptr: ^Cmplx = nil
	if jobz == .VALUES_AND_VECTORS && Z != nil {
		assert(int(Z.rows) >= int(n) && int(Z.cols) >= int(n), "Eigenvectors matrix too small")
		ldz = Z.ld
		z_ptr = raw_data(Z.data)
	}

	jobz_c := cast(u8)jobz
	uplo_c := cast(u8)uplo
	lwork := Blas_Int(len(work))
	lrwork := Blas_Int(len(rwork))
	liwork := Blas_Int(len(iwork))

	when Cmplx == complex64 {
		lapack.chbgvd_(&jobz_c, &uplo_c, &n, &ka, &kb, raw_data(AB.data), &ldab, raw_data(BB.data), &ldbb, raw_data(w), z_ptr, &ldz, raw_data(work), &lwork, raw_data(rwork), &lrwork, raw_data(iwork), &liwork, &info)
	} else when Cmplx == complex128 {
		lapack.zhbgvd_(&jobz_c, &uplo_c, &n, &ka, &kb, raw_data(AB.data), &ldab, raw_data(BB.data), &ldbb, raw_data(w), z_ptr, &ldz, raw_data(work), &lwork, raw_data(rwork), &lrwork, raw_data(iwork), &liwork, &info)
	}

	return info, info == 0
}

// ===================================================================================
// GENERALIZED EIGENVALUE PROBLEMS - EXPERT (SBGVX/HBGVX)
// ===================================================================================

// Solve generalized eigenvalue problem with expert driver (real version)
band_eigen_generalized_expert_real :: proc(
	jobz: EigenJobOption,
	range: EigenRangeOption,
	uplo: MatrixRegion,
	AB: ^BandedMatrix($T), // Banded matrix A (input/output - destroyed)
	BB: ^BandedMatrix(T), // Banded matrix B (input/output - destroyed)
	Q: ^BandedMatrix(T), // Optional Q matrix from reduction (input/output)
	vl: T, // Lower bound of eigenvalue range (if range == .VALUE)
	vu: T, // Upper bound of eigenvalue range (if range == .VALUE)
	il: int = 1, // Lower index of eigenvalues to compute (if range == .INDEX)
	iu: int = 1, // Upper index of eigenvalues to compute (if range == .INDEX)
	abstol: T, // Absolute error tolerance for eigenvalues
	w: []T, // Pre-allocated eigenvalues array (size n)
	Z: ^Matrix(T) = nil, // Pre-allocated eigenvectors matrix (optional)
	m: ^Blas_Int, // Number of eigenvalues found (output)
	work: []T, // Pre-allocated workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace
	ifail: []Blas_Int, // Pre-allocated failure indices (size n)
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	n := AB.cols
	ka := AB.kl // Bandwidth of A
	kb := BB.kl // Bandwidth of B
	ldab := AB.ldab
	ldbb := BB.ldab

	assert(len(w) >= int(n), "Eigenvalues array too small")
	assert(len(work) >= 7 * int(n), "Work array too small")
	assert(len(iwork) >= 5 * int(n), "Integer work array too small")
	assert(len(ifail) >= int(n), "IFAIL array too small")

	ldq := Blas_Int(1)
	q_ptr: ^T = nil
	if Q != nil {
		ldq = Q.ldab
		q_ptr = raw_data(Q.data)
	}

	ldz := Blas_Int(1)
	z_ptr: ^T = nil
	if jobz == .VALUES_AND_VECTORS && Z != nil {
		assert(int(Z.rows) >= int(n) && int(Z.cols) >= int(n), "Eigenvectors matrix too small")
		ldz = Z.ld
		z_ptr = raw_data(Z.data)
	}

	jobz_c := cast(u8)jobz
	range_c := cast(u8)range
	uplo_c := cast(u8)uplo
	vl_val := vl
	vu_val := vu
	il_val := Blas_Int(il)
	iu_val := Blas_Int(iu)
	abstol := abstol

	when T == f32 {
		lapack.ssbgvx_(&jobz_c, &range_c, &uplo_c, &n, &ka, &kb, raw_data(AB.data), &ldab, raw_data(BB.data), &ldbb, q_ptr, &ldq, &vl_val, &vu_val, &il_val, &iu_val, &abstol, m, raw_data(w), z_ptr, &ldz, raw_data(work), raw_data(iwork), raw_data(ifail), &info)
	} else when T == f64 {
		lapack.dsbgvx_(&jobz_c, &range_c, &uplo_c, &n, &ka, &kb, raw_data(AB.data), &ldab, raw_data(BB.data), &ldbb, q_ptr, &ldq, &vl_val, &vu_val, &il_val, &iu_val, &abstol, m, raw_data(w), z_ptr, &ldz, raw_data(work), raw_data(iwork), raw_data(ifail), &info)
	}

	return info, info == 0
}

// Solve generalized eigenvalue problem with expert driver (complex version)
band_eigen_generalized_expert_complex :: proc(
	jobz: EigenJobOption,
	range: EigenRangeOption,
	uplo: MatrixRegion,
	AB: ^BandedMatrix($Cmplx), // Banded matrix A (input/output - destroyed)
	BB: ^BandedMatrix(Cmplx), // Banded matrix B (input/output - destroyed)
	Q: ^BandedMatrix(Cmplx), // Optional Q matrix from reduction (input/output)
	vl: $Real, // Lower bound of eigenvalue range (if range == .VALUE)
	vu: Real, // Upper bound of eigenvalue range (if range == .VALUE)
	il: int = 1, // Lower index of eigenvalues to compute (if range == .INDEX)
	iu: int = 1, // Upper index of eigenvalues to compute (if range == .INDEX)
	abstol: Real, // Absolute error tolerance for eigenvalues
	w: []Real, // Pre-allocated eigenvalues array (size n) - always real
	Z: ^Matrix(Cmplx), // Pre-allocated eigenvectors matrix (optional)
	m: ^Blas_Int, // Number of eigenvalues found (output)
	work: []Cmplx, // Pre-allocated workspace
	rwork: []Real, // Pre-allocated real workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace
	ifail: []Blas_Int, // Pre-allocated failure indices (size n)
) -> (
	info: Info,
	ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
	n := AB.cols
	ka := AB.kl // Bandwidth of A
	kb := BB.kl // Bandwidth of B
	ldab := AB.ldab
	ldbb := BB.ldab

	assert(len(w) >= int(n), "Eigenvalues array too small")
	assert(len(work) >= int(n), "Work array too small")
	assert(len(rwork) >= 7 * int(n), "Real work array too small")
	assert(len(iwork) >= 5 * int(n), "Integer work array too small")
	assert(len(ifail) >= int(n), "IFAIL array too small")

	ldq := Blas_Int(1)
	q_ptr: ^Cmplx = nil
	if Q != nil {
		ldq = Q.ldab
		q_ptr = raw_data(Q.data)
	}

	ldz := Blas_Int(1)
	z_ptr: ^Cmplx = nil
	if jobz == .VALUES_AND_VECTORS && Z != nil {
		assert(int(Z.rows) >= int(n) && int(Z.cols) >= int(n), "Eigenvectors matrix too small")
		ldz = Z.ld
		z_ptr = raw_data(Z.data)
	}

	jobz_c := cast(u8)jobz
	range_c := cast(u8)range
	uplo_c := cast(u8)uplo
	vl_val := vl
	vu_val := vu
	il_val := Blas_Int(il)
	iu_val := Blas_Int(iu)
	abstol := abstol

	when Cmplx == complex64 {
		lapack.chbgvx_(&jobz_c, &range_c, &uplo_c, &n, &ka, &kb, raw_data(AB.data), &ldab, raw_data(BB.data), &ldbb, q_ptr, &ldq, &vl_val, &vu_val, &il_val, &iu_val, &abstol, m, raw_data(w), z_ptr, &ldz, raw_data(work), raw_data(rwork), raw_data(iwork), raw_data(ifail), &info)
	} else when Cmplx == complex128 {
		lapack.zhbgvx_(&jobz_c, &range_c, &uplo_c, &n, &ka, &kb, raw_data(AB.data), &ldab, raw_data(BB.data), &ldbb, q_ptr, &ldq, &vl_val, &vu_val, &il_val, &iu_val, &abstol, m, raw_data(w), z_ptr, &ldz, raw_data(work), raw_data(rwork), raw_data(iwork), raw_data(ifail), &info)
	}

	return info, info == 0
}

// ===================================================================================
// REDUCTION OF GENERALIZED PROBLEM TO STANDARD FORM (SBGST/HBGST)
// ===================================================================================

// Reduce generalized problem to standard form: C = L^{-1}*A*L^{-T} or C = U^{-T}*A*U^{-1} (real version)
band_reduce_generalized_real :: proc(
	vect: VectorOption,
	uplo: MatrixRegion,
	AB: ^BandedMatrix($T), // Banded matrix A (input/output - reduced)
	BB: ^BandedMatrix(T), // Cholesky factor of B (input - from band_cholesky)
	X: ^Matrix(T), // Optional transformation matrix (output)
	work: []T, // Pre-allocated workspace
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	n := AB.cols
	ka := AB.kl // Bandwidth of A
	kb := BB.kl // Bandwidth of B
	ldab := AB.ldab
	ldbb := BB.ldab

	assert(len(work) >= 2 * int(n), "Work array too small")

	ldx := Blas_Int(1)
	x_ptr: ^T = nil
	if vect == .FORM_VECTORS && X != nil {
		assert(int(X.rows) >= int(n) && int(X.cols) >= int(n), "X matrix too small")
		ldx = X.ld
		x_ptr = raw_data(X.data)
	}

	vect_c := cast(u8)vect
	uplo_c := cast(u8)uplo

	when T == f32 {
		lapack.ssbgst_(&vect_c, &uplo_c, &n, &ka, &kb, raw_data(AB.data), &ldab, raw_data(BB.data), &ldbb, x_ptr, &ldx, raw_data(work), &info)
	} else when T == f64 {
		lapack.dsbgst_(&vect_c, &uplo_c, &n, &ka, &kb, raw_data(AB.data), &ldab, raw_data(BB.data), &ldbb, x_ptr, &ldx, raw_data(work), &info)
	}

	return info, info == 0
}

// Reduce generalized problem to standard form: C = L^{-1}*A*L^{-H} or C = U^{-H}*A*U^{-1} (complex version)
band_reduce_generalized_complex :: proc(
	vect: VectorOption,
	uplo: MatrixRegion,
	AB: ^BandedMatrix($Cmplx), // Banded matrix A (input/output - reduced)
	BB: ^BandedMatrix(Cmplx), // Cholesky factor of B (input - from band_cholesky)
	X: ^Matrix(Cmplx), // Optional transformation matrix (output)
	work: []Cmplx, // Pre-allocated workspace
	rwork: []$Real, // Pre-allocated real workspace
) -> (
	info: Info,
	ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
	n := AB.cols
	ka := AB.kl // Bandwidth of A
	kb := BB.kl // Bandwidth of B
	ldab := AB.ldab
	ldbb := BB.ldab

	assert(len(work) >= int(n), "Work array too small")
	assert(len(rwork) >= int(n), "Real work array too small")

	ldx := Blas_Int(1)
	x_ptr: ^Cmplx = nil
	if vect == .FORM_VECTORS && X != nil {
		assert(int(X.rows) >= int(n) && int(X.cols) >= int(n), "X matrix too small")
		ldx = X.ld
		x_ptr = raw_data(X.data)
	}

	vect_c := cast(u8)vect
	uplo_c := cast(u8)uplo

	when Cmplx == complex64 {
		lapack.chbgst_(&vect_c, &uplo_c, &n, &ka, &kb, raw_data(AB.data), &ldab, raw_data(BB.data), &ldbb, x_ptr, &ldx, raw_data(work), raw_data(rwork), &info)
	} else when Cmplx == complex128 {
		lapack.zhbgst_(&vect_c, &uplo_c, &n, &ka, &kb, raw_data(AB.data), &ldab, raw_data(BB.data), &ldbb, x_ptr, &ldx, raw_data(work), raw_data(rwork), &info)
	}

	return info, info == 0
}
