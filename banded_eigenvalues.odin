package openblas

import lapack "./f77"

eigen_banded_hermitian :: proc {
	eigen_banded_hermitian_c64,
	eigen_banded_hermitian_c128,
}

eigen_banded_hermitian_2stage :: proc {
	eigen_banded_hermitian_2stage_c64,
	eigen_banded_hermitian_2stage_c128,
}

eigen_banded_hermitian_dc :: proc {
	eigen_banded_hermitian_dc_c64,
	eigen_banded_hermitian_dc_c128,
}

eigen_banded_hermitian_dc_2stage :: proc {
	eigen_banded_hermitian_dc_2stage_c64,
	eigen_banded_hermitian_dc_2stage_c128,
}

// Solve hermitian/symmetric banded eigenvalue problem (expert with subset selection)
eigen_banded_hermitian_expert :: proc {
	eigen_banded_hermitian_expert_real,
	eigen_banded_hermitian_expert_c64,
	eigen_banded_hermitian_expert_c128,
}

// Reduce hermitian/symmetric banded matrix to tridiagonal form
tridiagonalize_banded_hermitian :: proc {
	tridiagonalize_banded_hermitian_real,
	tridiagonalize_banded_hermitian_c64,
	tridiagonalize_banded_hermitian_c128,
}

// ===================================================================================
// HERMITIAN BANDED EIGENVALUE ROUTINES
// ===================================================================================

// Query result sizes for hermitian banded eigenvalue computation
query_result_sizes_eigen_banded_hermitian :: proc(
	n: int,
	compute_vectors: bool,
) -> (
	w_size: int,
	z_rows: int,
	z_cols: int, // Eigenvalues array// Eigenvector matrix rows// Eigenvector matrix cols
) {
	w_size = n
	if compute_vectors {
		z_rows = n
		z_cols = n
	}
	return
}

// Query workspace for hermitian banded eigenvalue computation
query_workspace_eigen_banded_hermitian :: proc(
	$T: typeid,
	n: int,
	compute_vectors: bool,
) -> (
	work: Blas_Int,
	rwork: Blas_Int,
) where T == complex64 ||
	T == complex128 {
	// Standard algorithm workspace
	return Blas_Int(n), Blas_Int(max(1, 3 * n - 2))
}

// Query workspace for hermitian banded eigenvalue computation (2-stage)
query_workspace_eigen_banded_hermitian_2stage :: proc(
	$T: typeid,
	n: int,
	kd: int,
	compute_vectors: bool,
) -> (
	lwork: Blas_Int,
	rwork: Blas_Int,
) where T == complex64 ||
	T == complex128 {
	// 2-stage algorithm requires more workspace
	// These are conservative estimates; actual requirements depend on block size
	if compute_vectors {
		lwork = Blas_Int(n * n)
	} else {
		lwork = Blas_Int(n * (kd + 1))
	}
	rwork = Blas_Int(max(1, 3 * n - 2))
	return
}

// Compute eigenvalues/eigenvectors of hermitian banded matrix (complex64)
eigen_banded_hermitian_c64 :: proc(
	jobz: VectorOption,
	uplo: MatrixRegion,
	n: int,
	kd: int,
	AB: ^Matrix(complex64), // Hermitian banded matrix (input, destroyed on output)
	w: []f32, // Pre-allocated eigenvalues (size n)
	Z: ^Matrix(complex64) = nil, // Pre-allocated eigenvectors (n×n) if jobz == FORM_VECTORS
	work: []complex64, // Pre-allocated workspace
	rwork: []f32, // Pre-allocated real workspace
) -> (
	info: Info,
	ok: bool,
) {
	// Validate inputs
	assert(len(w) >= n, "Eigenvalue array too small")
	assert(len(work) >= n, "Work array too small")
	assert(len(rwork) >= max(1, 3 * n - 2), "Real work array too small")
	if jobz == .FORM_VECTORS {
		assert(Z != nil, "Eigenvector matrix required when computing vectors")
		assert(int(Z.rows) >= n && int(Z.cols) >= n, "Eigenvector matrix too small")

	}

	jobz_c := vector_option_to_cstring(jobz)
	uplo_c := matrix_region_to_cstring(uplo)
	n_int := Blas_Int(n)
	kd_int := Blas_Int(kd)
	ldab := AB.ld
	ldz := Z.ld if Z != nil else Blas_Int(1)

	lapack.chbev_(
		jobz_c,
		uplo_c,
		&n_int,
		&kd_int,
		raw_data(AB.data),
		&ldab,
		raw_data(w),
		raw_data(Z.data) if Z != nil else nil,
		&ldz,
		raw_data(work),
		raw_data(rwork),
		&info,
		len(jobz_c),
		len(uplo_c),
	)

	return info, info == 0
}

// Compute eigenvalues/eigenvectors of hermitian banded matrix (complex128)
eigen_banded_hermitian_c128 :: proc(
	jobz: VectorOption,
	uplo: MatrixRegion,
	n: int,
	kd: int,
	AB: ^Matrix(complex128), // Hermitian banded matrix (input, destroyed on output)
	w: []f64, // Pre-allocated eigenvalues (size n)
	Z: ^Matrix(complex128) = nil, // Pre-allocated eigenvectors (n×n) if jobz == FORM_VECTORS
	work: []complex128, // Pre-allocated workspace
	rwork: []f64, // Pre-allocated real workspace
) -> (
	info: Info,
	ok: bool,
) {
	// Validate inputs
	assert(len(w) >= n, "Eigenvalue array too small")
	assert(len(work) >= n, "Work array too small")
	assert(len(rwork) >= max(1, 3 * n - 2), "Real work array too small")
	if jobz == .FORM_VECTORS {
		assert(Z != nil, "Eigenvector matrix required when computing vectors")
		assert(int(Z.rows) >= n && int(Z.cols) >= n, "Eigenvector matrix too small")

	}

	jobz_c := vector_option_to_cstring(jobz)
	uplo_c := matrix_region_to_cstring(uplo)
	n_int := Blas_Int(n)
	kd_int := Blas_Int(kd)
	ldab := AB.ld
	ldz := Z.ld if Z != nil else Blas_Int(1)

	lapack.zhbev_(
		jobz_c,
		uplo_c,
		&n_int,
		&kd_int,
		raw_data(AB.data),
		&ldab,
		raw_data(w),
		raw_data(Z.data) if Z != nil else nil,
		&ldz,
		raw_data(work),
		raw_data(rwork),
		&info,
		len(jobz_c),
		len(uplo_c),
	)

	return info, info == 0
}

// Compute eigenvalues/eigenvectors of hermitian banded matrix - 2-stage (complex64)
eigen_banded_hermitian_2stage_c64 :: proc(
	jobz: VectorOption,
	uplo: MatrixRegion,
	n: int,
	kd: int,
	AB: ^Matrix(complex64), // Hermitian banded matrix (input, destroyed on output)
	w: []f32, // Pre-allocated eigenvalues (size n)
	Z: ^Matrix(complex64) = nil, // Pre-allocated eigenvectors (n×n) if jobz == FORM_VECTORS
	work: []complex64, // Pre-allocated workspace
	lwork: Blas_Int, // Size of work array
	rwork: []f32, // Pre-allocated real workspace
) -> (
	info: Info,
	ok: bool,
) {
	// Validate inputs
	assert(len(w) >= n, "Eigenvalue array too small")
	assert(len(work) >= int(lwork), "Work array too small")
	assert(len(rwork) >= max(1, 3 * n - 2), "Real work array too small")
	if jobz == .FORM_VECTORS {
		assert(Z != nil, "Eigenvector matrix required when computing vectors")
		assert(int(Z.rows) >= n && int(Z.cols) >= n, "Eigenvector matrix too small")

	}

	jobz_c := vector_option_to_cstring(jobz)
	uplo_c := matrix_region_to_cstring(uplo)
	n_int := Blas_Int(n)
	kd_int := Blas_Int(kd)
	ldab := AB.ld
	ldz := Z.ld if Z != nil else Blas_Int(1)
	lwork_val := lwork

	lapack.chbev_2stage_(
		jobz_c,
		uplo_c,
		&n_int,
		&kd_int,
		raw_data(AB.data),
		&ldab,
		raw_data(w),
		raw_data(Z.data) if Z != nil else nil,
		&ldz,
		raw_data(work),
		&lwork_val,
		raw_data(rwork),
		&info,
		len(jobz_c),
		len(uplo_c),
	)

	return info, info == 0
}

// Compute eigenvalues/eigenvectors of hermitian banded matrix - 2-stage (complex128)
eigen_banded_hermitian_2stage_c128 :: proc(
	jobz: VectorOption,
	uplo: MatrixRegion,
	n: int,
	kd: int,
	AB: ^Matrix(complex128), // Hermitian banded matrix (input, destroyed on output)
	w: []f64, // Pre-allocated eigenvalues (size n)
	Z: ^Matrix(complex128) = nil, // Pre-allocated eigenvectors (n×n) if jobz == FORM_VECTORS
	work: []complex128, // Pre-allocated workspace
	lwork: Blas_Int, // Size of work array
	rwork: []f64, // Pre-allocated real workspace
) -> (
	info: Info,
	ok: bool,
) {
	// Validate inputs
	assert(len(w) >= n, "Eigenvalue array too small")
	assert(len(work) >= int(lwork), "Work array too small")
	assert(len(rwork) >= max(1, 3 * n - 2), "Real work array too small")
	if jobz == .FORM_VECTORS {
		assert(Z != nil, "Eigenvector matrix required when computing vectors")
		assert(int(Z.rows) >= n && int(Z.cols) >= n, "Eigenvector matrix too small")

	}

	jobz_c := vector_option_to_cstring(jobz)
	uplo_c := matrix_region_to_cstring(uplo)
	n_int := Blas_Int(n)
	kd_int := Blas_Int(kd)
	ldab := AB.ld
	ldz := Z.ld if Z != nil else Blas_Int(1)
	lwork_val := lwork

	lapack.zhbev_2stage_(
		jobz_c,
		uplo_c,
		&n_int,
		&kd_int,
		raw_data(AB.data),
		&ldab,
		raw_data(w),
		raw_data(Z.data) if Z != nil else nil,
		&ldz,
		raw_data(work),
		&lwork_val,
		raw_data(rwork),
		&info,
		len(jobz_c),
		len(uplo_c),
	)

	return info, info == 0
}

// Query workspace for hermitian banded eigenvalue divide-and-conquer
query_workspace_eigen_banded_hermitian_dc :: proc(
	$T: typeid,
	n: int,
	compute_vectors: bool,
) -> (
	work: Blas_Int,
	rwork: Blas_Int,
	iwork: Blas_Int,
) where T == complex64 ||
	T == complex128 {
	// Divide-and-conquer algorithm workspace
	if compute_vectors {
		when T == complex64 {
			work = Blas_Int(n * n)
			rwork = Blas_Int(1 + 5 * n + 2 * n * n)
		} else when T == complex128 {
			work = Blas_Int(n * n)
			rwork = Blas_Int(1 + 5 * n + 2 * n * n)
		}
	} else {
		work = Blas_Int(n)
		rwork = Blas_Int(n)
	}
	iwork = Blas_Int(3 + 5 * n)
	return
}

// Compute eigenvalues/eigenvectors using divide-and-conquer (complex64)
eigen_banded_hermitian_dc_c64 :: proc(
	jobz: VectorOption,
	uplo: MatrixRegion,
	n: int,
	kd: int,
	AB: ^Matrix(complex64), // Hermitian banded matrix (input, destroyed on output)
	w: []f32, // Pre-allocated eigenvalues (size n)
	Z: ^Matrix(complex64) = nil, // Pre-allocated eigenvectors (n×n) if jobz == FORM_VECTORS
	work: []complex64, // Pre-allocated workspace
	lwork: Blas_Int, // Size of work array
	rwork: []f32, // Pre-allocated real workspace
	lrwork: Blas_Int, // Size of rwork array
	iwork: []Blas_Int, // Pre-allocated integer workspace
	liwork: Blas_Int, // Size of iwork array
) -> (
	info: Info,
	ok: bool,
) {
	// Validate inputs
	assert(len(w) >= n, "Eigenvalue array too small")
	assert(len(work) >= int(lwork), "Work array too small")
	assert(len(rwork) >= int(lrwork), "Real work array too small")
	assert(len(iwork) >= int(liwork), "Integer work array too small")
	if jobz == .FORM_VECTORS {
		assert(Z != nil, "Eigenvector matrix required when computing vectors")
		assert(int(Z.rows) >= n && int(Z.cols) >= n, "Eigenvector matrix too small")
	}

	jobz_c := vector_option_to_cstring(jobz)
	uplo_c := matrix_region_to_cstring(uplo)
	n_int := Blas_Int(n)
	kd_int := Blas_Int(kd)
	ldab := AB.ld
	ldz := Z.ld if Z != nil else Blas_Int(1)
	lwork_val := lwork
	lrwork_val := lrwork
	liwork_val := liwork

	lapack.chbevd_(
		jobz_c,
		uplo_c,
		&n_int,
		&kd_int,
		raw_data(AB.data),
		&ldab,
		raw_data(w),
		raw_data(Z.data) if Z != nil else nil,
		&ldz,
		raw_data(work),
		&lwork_val,
		raw_data(rwork),
		&lrwork_val,
		raw_data(iwork),
		&liwork_val,
		&info,
		len(jobz_c),
		len(uplo_c),
	)

	return info, info == 0
}

// Compute eigenvalues/eigenvectors using divide-and-conquer (complex128)
eigen_banded_hermitian_dc_c128 :: proc(
	jobz: VectorOption,
	uplo: MatrixRegion,
	n: int,
	kd: int,
	AB: ^Matrix(complex128), // Hermitian banded matrix (input, destroyed on output)
	w: []f64, // Pre-allocated eigenvalues (size n)
	Z: ^Matrix(complex128) = nil, // Pre-allocated eigenvectors (n×n) if jobz == FORM_VECTORS
	work: []complex128, // Pre-allocated workspace
	lwork: Blas_Int, // Size of work array
	rwork: []f64, // Pre-allocated real workspace
	lrwork: Blas_Int, // Size of rwork array
	iwork: []Blas_Int, // Pre-allocated integer workspace
	liwork: Blas_Int, // Size of iwork array
) -> (
	info: Info,
	ok: bool,
) {
	// Validate inputs
	assert(len(w) >= n, "Eigenvalue array too small")
	assert(len(work) >= int(lwork), "Work array too small")
	assert(len(rwork) >= int(lrwork), "Real work array too small")
	assert(len(iwork) >= int(liwork), "Integer work array too small")
	if jobz == .FORM_VECTORS {
		assert(Z != nil, "Eigenvector matrix required when computing vectors")
		assert(int(Z.rows) >= n && int(Z.cols) >= n, "Eigenvector matrix too small")

	}

	jobz_c := vector_option_to_cstring(jobz)
	uplo_c := matrix_region_to_cstring(uplo)
	n_int := Blas_Int(n)
	kd_int := Blas_Int(kd)
	ldab := AB.ld
	ldz := Z.ld if Z != nil else Blas_Int(1)
	lwork_val := lwork
	lrwork_val := lrwork
	liwork_val := liwork

	lapack.zhbevd_(
		jobz_c,
		uplo_c,
		&n_int,
		&kd_int,
		raw_data(AB.data),
		&ldab,
		raw_data(w),
		raw_data(Z.data) if Z != nil else nil,
		&ldz,
		raw_data(work),
		&lwork_val,
		raw_data(rwork),
		&lrwork_val,
		raw_data(iwork),
		&liwork_val,
		&info,
		len(jobz_c),
		len(uplo_c),
	)

	return info, info == 0
}

// Compute eigenvalues/eigenvectors using divide-and-conquer 2-stage (complex64)
eigen_banded_hermitian_dc_2stage_c64 :: proc(
	jobz: VectorOption,
	uplo: MatrixRegion,
	n: int,
	kd: int,
	AB: ^Matrix(complex64), // Hermitian banded matrix (input, destroyed on output)
	w: []f32, // Pre-allocated eigenvalues (size n)
	Z: ^Matrix(complex64) = nil, // Pre-allocated eigenvectors (n×n) if jobz == FORM_VECTORS
	work: []complex64, // Pre-allocated workspace
	lwork: Blas_Int, // Size of work array
	rwork: []f32, // Pre-allocated real workspace
	lrwork: Blas_Int, // Size of rwork array
	iwork: []Blas_Int, // Pre-allocated integer workspace
	liwork: Blas_Int, // Size of iwork array
) -> (
	info: Info,
	ok: bool,
) {
	// Validate inputs
	assert(len(w) >= n, "Eigenvalue array too small")
	assert(len(work) >= int(lwork), "Work array too small")
	assert(len(rwork) >= int(lrwork), "Real work array too small")
	assert(len(iwork) >= int(liwork), "Integer work array too small")
	if jobz == .FORM_VECTORS {
		assert(Z != nil, "Eigenvector matrix required when computing vectors")
		assert(int(Z.rows) >= n && int(Z.cols) >= n, "Eigenvector matrix too small")
	}

	jobz_c := vector_option_to_cstring(jobz)
	uplo_c := matrix_region_to_cstring(uplo)
	n_int := Blas_Int(n)
	kd_int := Blas_Int(kd)
	ldab := AB.ld
	ldz := Z.ld if Z != nil else Blas_Int(1)
	lwork_val := lwork
	lrwork_val := lrwork
	liwork_val := liwork

	lapack.chbevd_2stage_(
		jobz_c,
		uplo_c,
		&n_int,
		&kd_int,
		raw_data(AB.data),
		&ldab,
		raw_data(w),
		raw_data(Z.data) if Z != nil else nil,
		&ldz,
		raw_data(work),
		&lwork_val,
		raw_data(rwork),
		&lrwork_val,
		raw_data(iwork),
		&liwork_val,
		&info,
		len(jobz_c),
		len(uplo_c),
	)

	return info, info == 0
}

// Compute eigenvalues/eigenvectors using divide-and-conquer 2-stage (complex128)
eigen_banded_hermitian_dc_2stage_c128 :: proc(
	jobz: VectorOption,
	uplo: MatrixRegion,
	n: int,
	kd: int,
	AB: ^Matrix(complex128), // Hermitian banded matrix (input, destroyed on output)
	w: []f64, // Pre-allocated eigenvalues (size n)
	Z: ^Matrix(complex128) = nil, // Pre-allocated eigenvectors (n×n) if jobz == FORM_VECTORS
	work: []complex128, // Pre-allocated workspace
	lwork: Blas_Int, // Size of work array
	rwork: []f64, // Pre-allocated real workspace
	lrwork: Blas_Int, // Size of rwork array
	iwork: []Blas_Int, // Pre-allocated integer workspace
	liwork: Blas_Int, // Size of iwork array
) -> (
	info: Info,
	ok: bool,
) {
	// Validate inputs
	assert(len(w) >= n, "Eigenvalue array too small")
	assert(len(work) >= int(lwork), "Work array too small")
	assert(len(rwork) >= int(lrwork), "Real work array too small")
	assert(len(iwork) >= int(liwork), "Integer work array too small")
	if jobz == .FORM_VECTORS {
		assert(Z != nil, "Eigenvector matrix required when computing vectors")
		assert(int(Z.rows) >= n && int(Z.cols) >= n, "Eigenvector matrix too small")
	}

	jobz_c := vector_option_to_cstring(jobz)
	uplo_c := matrix_region_to_cstring(uplo)
	n_int := Blas_Int(n)
	kd_int := Blas_Int(kd)
	ldab := AB.ld
	ldz := Z.ld if Z != nil else Blas_Int(1)
	lwork_val := lwork
	lrwork_val := lrwork
	liwork_val := liwork

	lapack.zhbevd_2stage_(
		jobz_c,
		uplo_c,
		&n_int,
		&kd_int,
		raw_data(AB.data),
		&ldab,
		raw_data(w),
		raw_data(Z.data) if Z != nil else nil,
		&ldz,
		raw_data(work),
		&lwork_val,
		raw_data(rwork),
		&lrwork_val,
		raw_data(iwork),
		&liwork_val,
		&info,
		len(jobz_c),
		len(uplo_c),
	)

	return info, info == 0
}

// Expert eigenvalue computation for hermitian/symmetric banded matrices with subset selection
// Query result sizes for expert hermitian/symmetric banded eigenvalue computation
query_result_sizes_eigen_banded_hermitian_expert :: proc(
	n: int,
	compute_vectors: bool,
	range: EigenRangeOption,
	vl: $T,
	vu: T,
	il: int = 1,
	iu: int = 0,
) -> (
	m: int,
	z_rows: int,
	z_cols: int, // Number of eigenvalues found
) where T == f32 || T == f64 {
	// For range == .ALL, all eigenvalues are found
	if range == .ALL {
		m = n
	} else if range == .INDEXED {
		// For indexed range, number is iu - il + 1
		m = iu - il + 1
	} else {
		// For value range, we don't know exactly how many will be found
		// Return maximum possible
		m = n
	}

	if compute_vectors {
		z_rows = n
		z_cols = n // Maximum possible
	}

	return m, z_rows, z_cols
}

// Query workspace for expert hermitian/symmetric banded eigenvalue computation
query_workspace_eigen_banded_hermitian_expert :: proc(
	$T: typeid,
	n: int,
	kd: int,
) -> (
	work_size: int,
	iwork_size: int,
	rwork_size: int, // Only for complex types
) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	when T == f32 || T == f64 {
		work_size = 7 * n
		iwork_size = 5 * n
		rwork_size = 0
	} else when T == complex64 || T == complex128 {
		work_size = n
		iwork_size = 5 * n
		rwork_size = 7 * n
	}
	return work_size, iwork_size, rwork_size
}

// Real variant
eigen_banded_hermitian_expert_real :: proc(
	jobz: VectorOption,
	range: EigenRangeOption,
	uplo: MatrixRegion,
	n: int,
	kd: int,
	AB: Matrix($T),
	vl: T,
	vu: T,
	il: int,
	iu: int,
	abstol: T,
	m: ^int, // Number of eigenvalues found
	w: []T, // Pre-allocated eigenvalue array
	Z: ^Matrix(T), // Optional pre-allocated eigenvector matrix
	work: []T, // Pre-allocated workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace
	ifail: []Blas_Int, // Pre-allocated failure array
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 {
	// Validate inputs
	assert(len(w) >= n, "Eigenvalue array too small")
	assert(len(work) >= 7 * n, "Work array too small")
	assert(len(iwork) >= 5 * n, "Integer work array too small")
	assert(len(ifail) >= n, "Failure array too small")
	if jobz == .FORM_VECTORS {
		assert(Z != nil, "Eigenvector matrix required when computing vectors")
		assert(int(Z.rows) >= n && int(Z.cols) >= n, "Eigenvector matrix too small")
	}

	jobz_c := vector_option_to_cstring(jobz)
	range_c := eigen_range_to_cstring(range)
	uplo_c := matrix_region_to_cstring(uplo)
	n_int := Blas_Int(n)
	kd_int := Blas_Int(kd)
	ldab := AB.ld
	vl_val := vl
	vu_val := vu
	il_int := Blas_Int(il)
	iu_int := Blas_Int(iu)
	abstol_val := abstol
	m_int := Blas_Int(0)
	ldz: Blas_Int = 1
	if Z != nil {
		ldz = Z.ld
	}

	when T == f32 {
		lapack.ssbevx_(
			jobz_c,
			range_c,
			uplo_c,
			&n_int,
			&kd_int,
			raw_data(AB.data),
			&ldab,
			nil, // Q not used in this mode
			&n_int, // LDQ
			&vl_val,
			&vu_val,
			&il_int,
			&iu_int,
			&abstol_val,
			&m_int,
			raw_data(w),
			raw_data(Z.data) if Z != nil else nil,
			&ldz,
			raw_data(work),
			raw_data(iwork),
			raw_data(ifail),
			&info,
			len(jobz_c),
			len(range_c),
			len(uplo_c),
		)
	} else {
		lapack.dsbevx_(
			jobz_c,
			range_c,
			uplo_c,
			&n_int,
			&kd_int,
			raw_data(AB.data),
			&ldab,
			nil, // Q not used in this mode
			&n_int, // LDQ
			&vl_val,
			&vu_val,
			&il_int,
			&iu_int,
			&abstol_val,
			&m_int,
			raw_data(w),
			raw_data(Z.data) if Z != nil else nil,
			&ldz,
			raw_data(work),
			raw_data(iwork),
			raw_data(ifail),
			&info,
			len(jobz_c),
			len(range_c),
			len(uplo_c),
		)
	}

	m^ = int(m_int)
	return info, info == 0
}

// Complex64 variant
eigen_banded_hermitian_expert_c64 :: proc(
	jobz: VectorOption,
	range: EigenRangeOption,
	uplo: MatrixRegion,
	n: int,
	kd: int,
	AB: Matrix(complex64),
	vl: f32,
	vu: f32,
	il: int,
	iu: int,
	abstol: f32,
	m: ^int, // Number of eigenvalues found
	w: []f32, // Pre-allocated eigenvalue array
	Z: ^Matrix(complex64), // Optional pre-allocated eigenvector matrix
	work: []complex64, // Pre-allocated workspace
	rwork: []f32, // Pre-allocated real workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace
	ifail: []Blas_Int, // Pre-allocated failure array
) -> (
	info: Info,
	ok: bool,
) {
	// Validate inputs
	assert(len(w) >= n, "Eigenvalue array too small")
	assert(len(work) >= n, "Work array too small")
	assert(len(rwork) >= 7 * n, "Real work array too small")
	assert(len(iwork) >= 5 * n, "Integer work array too small")
	assert(len(ifail) >= n, "Failure array too small")
	if jobz == .FORM_VECTORS {
		assert(Z != nil, "Eigenvector matrix required when computing vectors")
		assert(int(Z.rows) >= n && int(Z.cols) >= n, "Eigenvector matrix too small")
	}

	jobz_c := vector_option_to_cstring(jobz)
	range_c := eigen_range_to_cstring(range)
	uplo_c := matrix_region_to_cstring(uplo)
	n_int := Blas_Int(n)
	kd_int := Blas_Int(kd)
	ldab := AB.ld
	vl_val := vl
	vu_val := vu
	il_int := Blas_Int(il)
	iu_int := Blas_Int(iu)
	abstol_val := abstol
	m_int := Blas_Int(0)
	ldz: Blas_Int = 1
	if Z != nil {
		ldz = Z.ld
	}

	lapack.chbevx_(
		jobz_c,
		range_c,
		uplo_c,
		&n_int,
		&kd_int,
		raw_data(AB.data),
		&ldab,
		nil, // Q not used in this mode
		&n_int, // LDQ
		&vl_val,
		&vu_val,
		&il_int,
		&iu_int,
		&abstol_val,
		&m_int,
		raw_data(w),
		raw_data(Z.data) if Z != nil else nil,
		&ldz,
		raw_data(work),
		raw_data(rwork),
		raw_data(iwork),
		raw_data(ifail),
		&info,
		len(jobz_c),
		len(range_c),
		len(uplo_c),
	)

	m^ = int(m_int)
	return info, info == 0
}

// Complex128 variant
eigen_banded_hermitian_expert_c128 :: proc(
	jobz: VectorOption,
	range: EigenRangeOption,
	uplo: MatrixRegion,
	n: int,
	kd: int,
	AB: Matrix(complex128),
	vl: f64,
	vu: f64,
	il: int,
	iu: int,
	abstol: f64,
	m: ^int, // Number of eigenvalues found
	w: []f64, // Pre-allocated eigenvalue array
	Z: ^Matrix(complex128), // Optional pre-allocated eigenvector matrix
	work: []complex128, // Pre-allocated workspace
	rwork: []f64, // Pre-allocated real workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace
	ifail: []Blas_Int, // Pre-allocated failure array
) -> (
	info: Info,
	ok: bool,
) {
	// Validate inputs
	assert(len(w) >= n, "Eigenvalue array too small")
	assert(len(work) >= n, "Work array too small")
	assert(len(rwork) >= 7 * n, "Real work array too small")
	assert(len(iwork) >= 5 * n, "Integer work array too small")
	assert(len(ifail) >= n, "Failure array too small")
	if jobz == .FORM_VECTORS {
		assert(Z != nil, "Eigenvector matrix required when computing vectors")
		assert(int(Z.rows) >= n && int(Z.cols) >= n, "Eigenvector matrix too small")
	}

	jobz_c := vector_option_to_cstring(jobz)
	range_c := eigen_range_to_cstring(range)
	uplo_c := matrix_region_to_cstring(uplo)
	n_int := Blas_Int(n)
	kd_int := Blas_Int(kd)
	ldab := AB.ld
	vl_val := vl
	vu_val := vu
	il_int := Blas_Int(il)
	iu_int := Blas_Int(iu)
	abstol_val := abstol
	m_int := Blas_Int(0)
	ldz: Blas_Int = 1
	if Z != nil {
		ldz = Z.ld
	}

	lapack.zhbevx_(
		jobz_c,
		range_c,
		uplo_c,
		&n_int,
		&kd_int,
		raw_data(AB.data),
		&ldab,
		nil, // Q not used in this mode
		&n_int, // LDQ
		&vl_val,
		&vu_val,
		&il_int,
		&iu_int,
		&abstol_val,
		&m_int,
		raw_data(w),
		raw_data(Z.data) if Z != nil else nil,
		&ldz,
		raw_data(work),
		raw_data(rwork),
		raw_data(iwork),
		raw_data(ifail),
		&info,
		len(jobz_c),
		len(range_c),
		len(uplo_c),
	)

	m^ = int(m_int)
	return info, info == 0
}


// ===================================================================================
// HERMITIAN BANDED TRIDIAGONALIZATION
// ===================================================================================

// Query result sizes for tridiagonalization of hermitian/symmetric banded matrix
query_result_sizes_tridiagonalize_banded_hermitian :: proc(
	n: int,
	compute_q: bool,
) -> (
	d_size: int,
	e_size: int,
	q_rows: int,
	q_cols: int, // Diagonal elements array size// Off-diagonal elements array size// Q matrix rows// Q matrix cols
) {
	d_size = n
	e_size = max(0, n - 1)

	if compute_q {
		q_rows = n
		q_cols = n
	}

	return d_size, e_size, q_rows, q_cols
}

// Query workspace for tridiagonalization of hermitian/symmetric banded matrix
query_workspace_tridiagonalize_banded_hermitian :: proc(
	$T: typeid,
	n: int,
) -> (
	work: Blas_Int,
) where is_float(T) ||
	is_complex(T) {
	return Blas_Int(n)
}

// Reduce symmetric banded matrix to tridiagonal form (real)
tridiagonalize_banded_hermitian_real :: proc(
	vect: VectorOption,
	uplo: MatrixRegion,
	n: int,
	kd: int,
	AB: ^Matrix($T), // Symmetric banded matrix (input/output)
	d: []T, // Pre-allocated diagonal elements (size n)
	e: []T, // Pre-allocated off-diagonal elements (size n-1)
	Q: ^Matrix(T) = nil, // Pre-allocated Q matrix if vect == .FORM_VECTORS
	work: []T, // Pre-allocated workspace
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	assert(AB.format == .Banded, "Matrix must be banded format")
	assert(len(d) >= n, "Diagonal array too small")
	assert(len(e) >= n - 1, "Off-diagonal array too small")
	assert(len(work) >= n, "Work array too small")

	n_int := Blas_Int(n)
	kd_int := Blas_Int(kd)
	ldab := AB.storage.banded.ldab

	vect_c := vector_option_to_cstring(vect)
	uplo_c := matrix_region_to_cstring(uplo)

	ldq := Blas_Int(1)
	q_ptr: ^T = nil
	if vect == .FORM_VECTORS && Q != nil {
		assert(Q.rows >= n && Q.cols >= n, "Q matrix too small")
		ldq = Q.ld
		q_ptr = raw_data(Q.data)
	}

	when T == f32 {
		lapack.ssbtrd_(
			vect_c,
			uplo_c,
			&n_int,
			&kd_int,
			raw_data(AB.data),
			&ldab,
			raw_data(d),
			raw_data(e),
			q_ptr,
			&ldq,
			raw_data(work),
			&info,
			len(vect_c),
			len(uplo_c),
		)
	} else when T == f64 {
		lapack.dsbtrd_(
			vect_c,
			uplo_c,
			&n_int,
			&kd_int,
			raw_data(AB.data),
			&ldab,
			raw_data(d),
			raw_data(e),
			q_ptr,
			&ldq,
			raw_data(work),
			&info,
			len(vect_c),
			len(uplo_c),
		)
	}

	return info, info == 0
}

// Reduce hermitian banded matrix to tridiagonal form (complex64)
tridiagonalize_banded_hermitian_c64 :: proc(
	vect: VectorOption,
	uplo: MatrixRegion,
	n: int,
	kd: int,
	AB: ^Matrix(complex64), // Hermitian banded matrix (input/output)
	d: []f32, // Pre-allocated diagonal elements (size n)
	e: []f32, // Pre-allocated off-diagonal elements (size n-1)
	Q: ^Matrix(complex64) = nil, // Pre-allocated Q matrix if vect == .FORM_VECTORS
	work: []complex64, // Pre-allocated workspace
) -> (
	info: Info,
	ok: bool,
) {
	assert(AB.format == .Banded, "Matrix must be banded format")
	assert(len(d) >= n, "Diagonal array too small")
	assert(len(e) >= n - 1, "Off-diagonal array too small")
	assert(len(work) >= n, "Work array too small")

	n_int := Blas_Int(n)
	kd_int := Blas_Int(kd)
	ldab := AB.storage.banded.ldab

	vect_c := vector_option_to_cstring(vect)
	uplo_c := matrix_region_to_cstring(uplo)

	ldq := Blas_Int(1)
	q_ptr: ^complex64 = nil
	if vect == .FORM_VECTORS && Q != nil {
		assert(int(Q.rows) >= n && int(Q.cols) >= n, "Q matrix too small")
		ldq = Q.ld
		q_ptr = raw_data(Q.data)
	}

	lapack.chbtrd_(
		vect_c,
		uplo_c,
		&n_int,
		&kd_int,
		raw_data(AB.data),
		&ldab,
		raw_data(d),
		raw_data(e),
		q_ptr,
		&ldq,
		raw_data(work),
		&info,
		len(vect_c),
		len(uplo_c),
	)

	return info, info == 0
}

// Reduce hermitian banded matrix to tridiagonal form (complex128)
tridiagonalize_banded_hermitian_c128 :: proc(
	vect: VectorOption,
	uplo: MatrixRegion,
	n: int,
	kd: int,
	AB: ^Matrix(complex128), // Hermitian banded matrix (input/output)
	d: []f64, // Pre-allocated diagonal elements (size n)
	e: []f64, // Pre-allocated off-diagonal elements (size n-1)
	Q: ^Matrix(complex128) = nil, // Pre-allocated Q matrix if vect == .FORM_VECTORS
	work: []complex128, // Pre-allocated workspace
) -> (
	info: Info,
	ok: bool,
) {
	assert(AB.format == .Banded, "Matrix must be banded format")
	assert(len(d) >= n, "Diagonal array too small")
	assert(len(e) >= n - 1, "Off-diagonal array too small")
	assert(len(work) >= n, "Work array too small")

	n_int := Blas_Int(n)
	kd_int := Blas_Int(kd)
	ldab := AB.storage.banded.ldab

	vect_c := vector_option_to_cstring(vect)
	uplo_c := matrix_region_to_cstring(uplo)

	ldq := Blas_Int(1)
	q_ptr: ^complex128 = nil
	if vect == .FORM_VECTORS && Q != nil {
		assert(int(Q.rows) >= n && int(Q.cols) >= n, "Q matrix too small")
		ldq = Q.ld
		q_ptr = raw_data(Q.data)
	}

	lapack.zhbtrd_(
		vect_c,
		uplo_c,
		&n_int,
		&kd_int,
		raw_data(AB.data),
		&ldab,
		raw_data(d),
		raw_data(e),
		q_ptr,
		&ldq,
		raw_data(work),
		&info,
		len(vect_c),
		len(uplo_c),
	)

	return info, info == 0
}
