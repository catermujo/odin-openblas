package openblas

import lapack "./f77"
import "base:builtin"
import "core:mem"

// ===================================================================================
// POSITIVE DEFINITE BANDED MATRIX CONDITION NUMBER ESTIMATION AND EQUILIBRATION
// ===================================================================================


banded_cond :: proc {
	banded_cond_real,
	banded_cond_c64,
	banded_cond_c128,
}

banded_equilibrate :: proc {
	banded_equilibrate_real,
	banded_equilibrate_c64,
	banded_equilibrate_c128,
}

banded_equilibrate_improved :: proc {
	banded_equilibrate_improved_real,
	banded_equilibrate_improved_c64,
	banded_equilibrate_improved_c128,
}


// ===================================================================================
// CONDITION NUMBER ESTIMATION IMPLEMENTATION
// ===================================================================================

// Query result sizes for condition number estimation
query_result_sizes_condition_banded_pd :: proc(
	n: int,
) -> (
	rcond_size: int, // Single scalar output
) {
	return 1
}

// Query workspace for condition number estimation
query_workspace_condition_banded_pd :: proc(
	$T: typeid,
	n: int,
) -> (
	work: Blas_Int,
	rwork: Blas_Int,
	iwork: Blas_Int,
) where is_float(T) ||
	is_complex(T) {
	when T == f32 || T == f64 {
		return Blas_Int(3 * n), 0, Blas_Int(n)
	} else when T == complex64 || T == complex128 {
		return Blas_Int(2 * n), Blas_Int(n), 0
	}
}

// Estimate condition number of positive definite banded matrix (f32/complex64)
// Estimates reciprocal condition number using factorization from PBTRF
solve_estimate_condition_banded_pd_f32_c64 :: proc(
	uplo: MatrixRegion,
	n: int,
	kd: int,
	AB: ^Matrix($T), // Banded matrix factorization from PBTRF
	anorm: f32, // 1-norm of original matrix
	rcond: ^f32, // Output: reciprocal condition number
	work: []T, // Pre-allocated workspace
	rwork: []f32 = nil, // Pre-allocated real workspace (complex only)
	iwork: []Blas_Int = nil, // Pre-allocated integer workspace (real only)
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == complex64 {
	// Validate inputs
	assert(anorm >= 0, "anorm must be non-negative")
	when T == f32 {
		assert(len(work) >= 3 * n, "Work array too small")
		assert(len(iwork) >= n, "Integer work array too small")
	} else when T == complex64 {
		assert(len(work) >= 2 * n, "Work array too small")
		assert(len(rwork) >= n, "Real work array too small")
	}

	uplo_c := matrix_region_to_cstring(uplo)
	n_int := Blas_Int(n)
	kd_int := Blas_Int(kd)
	ldab := AB.ld
	anorm_val := anorm

	when T == f32 {
		lapack.spbcon_(
			uplo_c,
			&n_int,
			&kd_int,
			raw_data(AB.data),
			&ldab,
			&anorm_val,
			rcond,
			raw_data(work),
			raw_data(iwork),
			&info,
			len(uplo_c),
		)
	} else when T == complex64 {
		lapack.cpbcon_(
			uplo_c,
			&n_int,
			&kd_int,
			raw_data(AB.data),
			&ldab,
			&anorm_val,
			rcond,
			raw_data(work),
			raw_data(rwork),
			&info,
			len(uplo_c),
		)
	}

	return info, info == 0
}

// Estimate condition number of positive definite banded matrix (f64/complex128)
// Estimates reciprocal condition number using factorization from PBTRF
solve_estimate_condition_banded_pd_f64_c128 :: proc(
	uplo: MatrixRegion,
	n: int,
	kd: int,
	AB: ^Matrix($T), // Banded matrix factorization from PBTRF
	anorm: f64, // 1-norm of original matrix
	rcond: ^f64, // Output: reciprocal condition number
	work: []T, // Pre-allocated workspace
	rwork: []f64 = nil, // Pre-allocated real workspace (complex only)
	iwork: []Blas_Int = nil, // Pre-allocated integer workspace (real only)
) -> (
	info: Info,
	ok: bool,
) where T == f64 || T == complex128 {
	// Validate inputs
	assert(anorm >= 0, "anorm must be non-negative")
	when T == f64 {
		assert(len(work) >= 3 * n, "Work array too small")
		assert(len(iwork) >= n, "Integer work array too small")
	} else when T == complex128 {
		assert(len(work) >= 2 * n, "Work array too small")
		assert(len(rwork) >= n, "Real work array too small")
	}

	uplo_c := matrix_region_to_cstring(uplo)
	n_int := Blas_Int(n)
	kd_int := Blas_Int(kd)
	ldab := AB.ld
	anorm_val := anorm

	when T == f64 {
		lapack.dpbcon_(
			uplo_c,
			&n_int,
			&kd_int,
			raw_data(AB.data),
			&ldab,
			&anorm_val,
			rcond,
			raw_data(work),
			raw_data(iwork),
			&info,
			len(uplo_c),
		)
	} else when T == complex128 {
		lapack.zpbcon_(
			uplo_c,
			&n_int,
			&kd_int,
			raw_data(AB.data),
			&ldab,
			&anorm_val,
			rcond,
			raw_data(work),
			raw_data(rwork),
			&info,
			len(uplo_c),
		)
	}

	return info, info == 0
}


// ===================================================================================
// POSITIVE DEFINITE BANDED MATRIX EQUILIBRATION
// ===================================================================================

// Query result sizes for equilibration scaling
query_result_sizes_equilibration_banded_pd :: proc(
	n: int,
) -> (
	S_size: int,
	scond_size: int,
	amax_size: int, // Scaling factors array// Single scalar output// Single scalar output
) {
	return n, 1, 1
}

// Compute equilibration scaling for positive definite banded matrix (f32/complex64)
// Computes scaling factors to improve conditioning
solve_equilibration_banded_pd_f32_c64 :: proc(
	uplo: MatrixRegion,
	n: int,
	kd: int,
	AB: ^Matrix($T), // Positive definite banded matrix
	S: []f32, // Pre-allocated scaling factors (size n)
	scond: ^f32, // Output: ratio of smallest to largest scaling factor
	amax: ^f32, // Output: maximum absolute value in matrix
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == complex64 {
	// Validate inputs
	assert(len(S) >= n, "Scaling array too small")

	uplo_c := matrix_region_to_cstring(uplo)
	n_int := Blas_Int(n)
	kd_int := Blas_Int(kd)
	ldab := AB.ld

	when T == f32 {
		lapack.spbequ_(
			uplo_c,
			&n_int,
			&kd_int,
			raw_data(AB.data),
			&ldab,
			raw_data(S),
			scond,
			amax,
			&info,
			len(uplo_c),
		)
	} else when T == complex64 {
		lapack.cpbequ_(
			uplo_c,
			&n_int,
			&kd_int,
			raw_data(AB.data),
			&ldab,
			raw_data(S),
			scond,
			amax,
			&info,
			len(uplo_c),
		)
	}

	return info, info == 0
}

// Compute equilibration scaling for positive definite banded matrix (f64/complex128)
// Computes scaling factors to improve conditioning
solve_equilibration_banded_pd_f64_c128 :: proc(
	uplo: MatrixRegion,
	n: int,
	kd: int,
	AB: ^Matrix($T), // Positive definite banded matrix
	S: []f64, // Pre-allocated scaling factors (size n)
	scond: ^f64, // Output: ratio of smallest to largest scaling factor
	amax: ^f64, // Output: maximum absolute value in matrix
) -> (
	info: Info,
	ok: bool,
) where T == f64 || T == complex128 {
	// Validate inputs
	assert(len(S) >= n, "Scaling array too small")

	uplo_c := matrix_region_to_cstring(uplo)
	n_int := Blas_Int(n)
	kd_int := Blas_Int(kd)
	ldab := AB.ld

	when T == f64 {
		lapack.dpbequ_(
			uplo_c,
			&n_int,
			&kd_int,
			raw_data(AB.data),
			&ldab,
			raw_data(S),
			scond,
			amax,
			&info,
			len(uplo_c),
		)
	} else when T == complex128 {
		lapack.zpbequ_(
			uplo_c,
			&n_int,
			&kd_int,
			raw_data(AB.data),
			&ldab,
			raw_data(S),
			scond,
			amax,
			&info,
			len(uplo_c),
		)
	}

	return info, info == 0
}


// ===================================================================================
// BANDED MATRIX CONDITION NUMBER ESTIMATION
// ===================================================================================
// Query workspace for banded condition estimation
query_workspace_banded_cond :: proc(
	$T: typeid,
	n: int,
) -> (
	work: Blas_Int,
	rwork: Blas_Int,
	iwork: Blas_Int,
) where is_float(T) ||
	is_complex(T) {
	when is_float(T) {
		return Blas_Int(3 * n), 0, Blas_Int(n)
	} else when is_complex(T) {
		return Blas_Int(2 * n), Blas_Int(n), 0
	}
}

// Estimate condition number of banded matrix (real version)
banded_cond_real :: proc(
	AB: ^Matrix($T), // Factored banded matrix from LU decomposition
	ipiv: []Blas_Int, // Pivot indices from LU factorization
	anorm: T, // 1-norm of original matrix A
	norm := MatrixNorm.OneNorm, // Norm type: "1", "O", or "I"
	rcond: ^T, // Output: reciprocal condition number
	work: []T, // Pre-allocated workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	assert(AB.format == .Banded, "Matrix must be in banded format")

	n := AB.cols
	kl := AB.storage.banded.kl
	ku := AB.storage.banded.ku
	ldab := AB.ld

	// Validate inputs
	assert(len(ipiv) >= int(n + kl), "Pivot array too small")
	assert(len(work) >= 3 * int(n), "Work array too small")
	assert(len(iwork) >= int(n), "Integer work array too small")
	assert(anorm >= 0, "anorm must be non-negative")

	anorm_val := anorm
	norm_str := norm_to_cstring(norm)

	when T == f32 {
		lapack.sgbcon_(
			norm_str,
			&n,
			&kl,
			&ku,
			raw_data(AB.data),
			&ldab,
			raw_data(ipiv),
			&anorm_val,
			rcond,
			raw_data(work),
			raw_data(iwork),
			&info,
			1,
		)
	} else when T == f64 {
		lapack.dgbcon_(
			norm_str,
			&n,
			&kl,
			&ku,
			raw_data(AB.data),
			&ldab,
			raw_data(ipiv),
			&anorm_val,
			rcond,
			raw_data(work),
			raw_data(iwork),
			&info,
			1,
		)
	}

	return info, info == 0
}

// Estimate condition number of banded matrix (complex64 version)
banded_cond_c64 :: proc(
	AB: ^Matrix(complex64), // Factored banded matrix from LU decomposition
	ipiv: []Blas_Int, // Pivot indices from LU factorization
	anorm: f32, // 1-norm of original matrix A
	norm := MatrixNorm.OneNorm, // Norm type
	rcond: ^f32, // Output: reciprocal condition number
	work: []complex64, // Pre-allocated workspace
	rwork: []f32, // Pre-allocated real workspace
) -> (
	info: Info,
	ok: bool,
) {
	assert(AB.format == .Banded, "Matrix must be in banded format")

	n := AB.cols
	kl := AB.storage.banded.kl
	ku := AB.storage.banded.ku
	ldab := AB.ld

	// Validate inputs
	assert(len(ipiv) >= int(n + kl), "Pivot array too small")
	assert(len(work) >= 2 * int(n), "Work array too small")
	assert(len(rwork) >= int(n), "Real work array too small")
	assert(anorm >= 0, "anorm must be non-negative")

	anorm_val := anorm
	norm_str := norm_to_cstring(norm)

	lapack.cgbcon_(
		norm_str,
		&n,
		&kl,
		&ku,
		raw_data(AB.data),
		&ldab,
		raw_data(ipiv),
		&anorm_val,
		rcond,
		raw_data(work),
		raw_data(rwork),
		&info,
		1,
	)

	return info, info == 0
}

// Estimate condition number of banded matrix (complex128 version)
banded_cond_c128 :: proc(
	AB: ^Matrix(complex128), // Factored banded matrix from LU decomposition
	ipiv: []Blas_Int, // Pivot indices from LU factorization
	anorm: f64, // 1-norm of original matrix A
	norm := MatrixNorm.OneNorm, // Norm type
	rcond: ^f64, // Output: reciprocal condition number
	work: []complex128, // Pre-allocated workspace
	rwork: []f64, // Pre-allocated real workspace
) -> (
	info: Info,
	ok: bool,
) {
	assert(AB.format == .Banded, "Matrix must be in banded format")

	n := AB.cols
	kl := AB.storage.banded.kl
	ku := AB.storage.banded.ku
	ldab := AB.ld

	// Validate inputs
	assert(len(ipiv) >= int(n + kl), "Pivot array too small")
	assert(len(work) >= 2 * int(n), "Work array too small")
	assert(len(rwork) >= int(n), "Real work array too small")
	assert(anorm >= 0, "anorm must be non-negative")

	anorm_val := anorm
	norm_str := norm_to_cstring(norm)

	lapack.zgbcon_(
		norm_str,
		&n,
		&kl,
		&ku,
		raw_data(AB.data),
		&ldab,
		raw_data(ipiv),
		&anorm_val,
		rcond,
		raw_data(work),
		raw_data(rwork),
		&info,
		1,
	)

	return info, info == 0
}


// ===================================================================================
// EQUILIBRATION
// ===================================================================================

// Query result sizes for banded equilibration
query_result_sizes_banded_equilibrate :: proc(
	m: int,
	n: int,
) -> (
	R_size: int,
	C_size: int,
	rowcnd_size: int,
	colcnd_size: int,
	amax_size: int, // Row scale factors array// Column scale factors array// Scalar output// Scalar output// Scalar output
) {
	return m, n, 1, 1, 1
}

// Compute row and column equilibration scale factors (real version)
banded_equilibrate_real :: proc(
	AB: ^Matrix($T), // Banded matrix to equilibrate
	R: []T, // Pre-allocated row scale factors (size m)
	C: []T, // Pre-allocated column scale factors (size n)
	rowcnd: ^T, // Output: ratio of smallest to largest row scale
	colcnd: ^T, // Output: ratio of smallest to largest column scale
	amax: ^T, // Output: absolute value of largest matrix element
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	assert(AB.format == .Banded, "Matrix must be in banded format")

	m := AB.rows
	n := AB.cols
	kl := AB.storage.banded.kl
	ku := AB.storage.banded.ku
	ldab := AB.ld

	// Validate inputs
	assert(len(R) >= int(m), "Row scale array too small")
	assert(len(C) >= int(n), "Column scale array too small")

	when T == f32 {
		lapack.sgbequ_(
			&m,
			&n,
			&kl,
			&ku,
			raw_data(AB.data),
			&ldab,
			raw_data(R),
			raw_data(C),
			rowcnd,
			colcnd,
			amax,
			&info,
		)
	} else when T == f64 {
		lapack.dgbequ_(
			&m,
			&n,
			&kl,
			&ku,
			raw_data(AB.data),
			&ldab,
			raw_data(R),
			raw_data(C),
			rowcnd,
			colcnd,
			amax,
			&info,
		)
	}

	return info, info == 0
}

// Compute row and column equilibration scale factors (complex64 version)
banded_equilibrate_c64 :: proc(
	AB: ^Matrix(complex64), // Banded matrix to equilibrate
	R: []f32, // Pre-allocated row scale factors (size m)
	C: []f32, // Pre-allocated column scale factors (size n)
	rowcnd: ^f32, // Output: ratio of smallest to largest row scale
	colcnd: ^f32, // Output: ratio of smallest to largest column scale
	amax: ^f32, // Output: absolute value of largest matrix element
) -> (
	info: Info,
	ok: bool,
) {
	assert(AB.format == .Banded, "Matrix must be in banded format")

	m := AB.rows
	n := AB.cols
	kl := AB.storage.banded.kl
	ku := AB.storage.banded.ku
	ldab := AB.ld

	// Validate inputs
	assert(len(R) >= int(m), "Row scale array too small")
	assert(len(C) >= int(n), "Column scale array too small")

	lapack.cgbequ_(
		&m,
		&n,
		&kl,
		&ku,
		raw_data(AB.data),
		&ldab,
		raw_data(R),
		raw_data(C),
		rowcnd,
		colcnd,
		amax,
		&info,
	)

	return info, info == 0
}

// Compute row and column equilibration scale factors (complex128 version)
banded_equilibrate_c128 :: proc(
	AB: ^Matrix(complex128), // Banded matrix to equilibrate
	R: []f64, // Pre-allocated row scale factors (size m)
	C: []f64, // Pre-allocated column scale factors (size n)
	rowcnd: ^f64, // Output: ratio of smallest to largest row scale
	colcnd: ^f64, // Output: ratio of smallest to largest column scale
	amax: ^f64, // Output: absolute value of largest matrix element
) -> (
	info: Info,
	ok: bool,
) {
	assert(AB.format == .Banded, "Matrix must be in banded format")

	m := AB.rows
	n := AB.cols
	kl := AB.storage.banded.kl
	ku := AB.storage.banded.ku
	ldab := AB.ld

	// Validate inputs
	assert(len(R) >= int(m), "Row scale array too small")
	assert(len(C) >= int(n), "Column scale array too small")

	lapack.zgbequ_(
		&m,
		&n,
		&kl,
		&ku,
		raw_data(AB.data),
		&ldab,
		raw_data(R),
		raw_data(C),
		rowcnd,
		colcnd,
		amax,
		&info,
	)

	return info, info == 0
}

// Improved equilibration with better algorithm (real version)
banded_equilibrate_improved_real :: proc(
	AB: ^Matrix($T), // Banded matrix to equilibrate
	R: []T, // Pre-allocated row scale factors (size m)
	C: []T, // Pre-allocated column scale factors (size n)
	rowcnd: ^T, // Output: ratio of smallest to largest row scale
	colcnd: ^T, // Output: ratio of smallest to largest column scale
	amax: ^T, // Output: absolute value of largest matrix element
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	assert(AB.format == .Banded, "Matrix must be in banded format")

	m := AB.rows
	n := AB.cols
	kl := AB.storage.banded.kl
	ku := AB.storage.banded.ku
	ldab := AB.ld

	// Validate inputs
	assert(len(R) >= int(m), "Row scale array too small")
	assert(len(C) >= int(n), "Column scale array too small")

	when T == f32 {
		lapack.sgbequb_(
			&m,
			&n,
			&kl,
			&ku,
			raw_data(AB.data),
			&ldab,
			raw_data(R),
			raw_data(C),
			rowcnd,
			colcnd,
			amax,
			&info,
		)
	} else when T == f64 {
		lapack.dgbequb_(
			&m,
			&n,
			&kl,
			&ku,
			raw_data(AB.data),
			&ldab,
			raw_data(R),
			raw_data(C),
			rowcnd,
			colcnd,
			amax,
			&info,
		)
	}

	return info, info == 0
}

// Improved equilibration with better algorithm (complex64 version)
banded_equilibrate_improved_c64 :: proc(
	AB: ^Matrix(complex64), // Banded matrix to equilibrate
	R: []f32, // Pre-allocated row scale factors (size m)
	C: []f32, // Pre-allocated column scale factors (size n)
	rowcnd: ^f32, // Output: ratio of smallest to largest row scale
	colcnd: ^f32, // Output: ratio of smallest to largest column scale
	amax: ^f32, // Output: absolute value of largest matrix element
) -> (
	info: Info,
	ok: bool,
) {
	assert(AB.format == .Banded, "Matrix must be in banded format")

	m := AB.rows
	n := AB.cols
	kl := AB.storage.banded.kl
	ku := AB.storage.banded.ku
	ldab := AB.ld

	// Validate inputs
	assert(len(R) >= int(m), "Row scale array too small")
	assert(len(C) >= int(n), "Column scale array too small")

	lapack.cgbequb_(
		&m,
		&n,
		&kl,
		&ku,
		raw_data(AB.data),
		&ldab,
		raw_data(R),
		raw_data(C),
		rowcnd,
		colcnd,
		amax,
		&info,
	)

	return info, info == 0
}

// Improved equilibration with better algorithm (complex128 version)
banded_equilibrate_improved_c128 :: proc(
	AB: ^Matrix(complex128), // Banded matrix to equilibrate
	R: []f64, // Pre-allocated row scale factors (size m)
	C: []f64, // Pre-allocated column scale factors (size n)
	rowcnd: ^f64, // Output: ratio of smallest to largest row scale
	colcnd: ^f64, // Output: ratio of smallest to largest column scale
	amax: ^f64, // Output: absolute value of largest matrix element
) -> (
	info: Info,
	ok: bool,
) {
	assert(AB.format == .Banded, "Matrix must be in banded format")

	m := AB.rows
	n := AB.cols
	kl := AB.storage.banded.kl
	ku := AB.storage.banded.ku
	ldab := AB.ld

	// Validate inputs
	assert(len(R) >= int(m), "Row scale array too small")
	assert(len(C) >= int(n), "Column scale array too small")

	lapack.zgbequb_(
		&m,
		&n,
		&kl,
		&ku,
		raw_data(AB.data),
		&ldab,
		raw_data(R),
		raw_data(C),
		rowcnd,
		colcnd,
		amax,
		&info,
	)

	return info, info == 0
}
