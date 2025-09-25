package openblas

import lapack "./f77"
import "base:builtin"
import "core:mem"

// ===================================================================================
// POSITIVE DEFINITE BANDED ITERATIVE REFINEMENT AND SPLIT CHOLESKY
// ===================================================================================

// Iterative refinement for positive definite banded systems proc group
solve_refine_banded_pd :: proc {
	solve_refine_banded_pd_f32_c64,
	solve_refine_banded_pd_f64_c128,
}

refine_banded :: proc {
	refine_banded_real,
	refine_banded_c64,
	refine_banded_c128,
}

refine_banded_extended :: proc {
	refine_banded_extended_real,
	refine_banded_extended_c64,
	refine_banded_extended_c128,
}

// ===================================================================================
// ITERATIVE REFINEMENT IMPLEMENTATION
// ===================================================================================

// Query workspace requirements for iterative refinement
query_workspace_refine_banded_pd :: proc(
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

// Iterative refinement for positive definite banded systems (f32/complex64)
// Improves solution accuracy using iterative refinement
solve_refine_banded_pd_f32_c64 :: proc(
	uplo: MatrixRegion,
	n: int,
	kd: int,
	nrhs: int,
	AB: ^Matrix($T), // Original banded matrix
	AFB: ^Matrix(T), // Factorized matrix from PBTRF
	B: ^Matrix(T), // Right-hand side matrix
	X: ^Matrix(T), // Solution matrix (input/output)
	ferr: []f32, // Pre-allocated forward error bounds (size nrhs)
	berr: []f32, // Pre-allocated backward error bounds (size nrhs)
	work: []T, // Pre-allocated workspace
	rwork: []f32 = nil, // Pre-allocated real workspace (complex only)
	iwork: []Blas_Int = nil, // Pre-allocated integer workspace (real only)
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == complex64 {
	// Validate inputs
	assert(len(ferr) >= nrhs, "Forward error array too small")
	assert(len(berr) >= nrhs, "Backward error array too small")
	when T == f32 {
		assert(len(work) >= 3 * n, "Work array too small")
		assert(len(iwork) >= n, "Integer work array too small")
	} else when T == complex64 {
		assert(len(work) >= 2 * n, "Work array too small")
		assert(len(rwork) >= n, "Real work array too small")
	}

	uplo_c := matrix_region_to_cstring(uplo)
	n_int: Blas_Int = n
	kd_int: Blas_Int = kd
	nrhs_int: Blas_Int = nrhs
	ldab := AB.ld
	ldafb := AFB.ld
	ldb := B.ld
	ldx := X.ld

	when T == f32 {
		lapack.spbrfs_(
			uplo_c,
			&n_int,
			&kd_int,
			&nrhs_int,
			raw_data(AB.data),
			&ldab,
			raw_data(AFB.data),
			&ldafb,
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			raw_data(ferr),
			raw_data(berr),
			raw_data(work),
			raw_data(iwork),
			&info,
			len(uplo_c),
		)
	} else when T == complex64 {
		lapack.cpbrfs_(
			uplo_c,
			&n_int,
			&kd_int,
			&nrhs_int,
			raw_data(AB.data),
			&ldab,
			raw_data(AFB.data),
			&ldafb,
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			raw_data(ferr),
			raw_data(berr),
			raw_data(work),
			raw_data(rwork),
			&info,
			len(uplo_c),
		)
	}

	return info, info == 0
}

// Iterative refinement for positive definite banded systems (f64/complex128)
// Improves solution accuracy using iterative refinement
solve_refine_banded_pd_f64_c128 :: proc(
	uplo: MatrixRegion,
	n: int,
	kd: int,
	nrhs: int,
	AB: ^Matrix($T), // Original banded matrix
	AFB: ^Matrix(T), // Factorized matrix from PBTRF
	B: ^Matrix(T), // Right-hand side matrix
	X: ^Matrix(T), // Solution matrix (input/output)
	ferr: []f64, // Pre-allocated forward error bounds (size nrhs)
	berr: []f64, // Pre-allocated backward error bounds (size nrhs)
	work: []T, // Pre-allocated workspace
	rwork: []f64 = nil, // Pre-allocated real workspace (complex only)
	iwork: []Blas_Int = nil, // Pre-allocated integer workspace (real only)
) -> (
	info: Info,
	ok: bool,
) where T == f64 || T == complex128 {
	// Validate inputs
	assert(len(ferr) >= nrhs, "Forward error array too small")
	assert(len(berr) >= nrhs, "Backward error array too small")
	when T == f64 {
		assert(len(work) >= 3 * n, "Work array too small")
		assert(len(iwork) >= n, "Integer work array too small")
	} else when T == complex128 {
		assert(len(work) >= 2 * n, "Work array too small")
		assert(len(rwork) >= n, "Real work array too small")
	}

	uplo_c := matrix_region_to_cstring(uplo)
	n_int: Blas_Int = n
	kd_int: Blas_Int = kd
	nrhs_int: Blas_Int = nrhs
	ldab: Blas_Int = AB.ld
	ldafb: Blas_Int = AFB.ld
	ldb: Blas_Int = B.ld
	ldx: Blas_Int = X.ld

	when T == f64 {
		lapack.dpbrfs_(
			uplo_c,
			&n_int,
			&kd_int,
			&nrhs_int,
			raw_data(AB.data),
			&ldab,
			raw_data(AFB.data),
			&ldafb,
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			raw_data(ferr),
			raw_data(berr),
			raw_data(work),
			raw_data(iwork),
			&info,
			len(uplo_c),
		)
	} else when T == complex128 {
		lapack.zpbrfs_(
			uplo_c,
			&n_int,
			&kd_int,
			&nrhs_int,
			raw_data(AB.data),
			&ldab,
			raw_data(AFB.data),
			&ldafb,
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			raw_data(ferr),
			raw_data(berr),
			raw_data(work),
			raw_data(rwork),
			&info,
			len(uplo_c),
		)
	}

	return info, info == 0
}


// ===================================================================================
// SPLIT CHOLESKY FACTORIZATION
// ===================================================================================

// Split Cholesky factorization for positive definite banded matrix (generic)
// Computes split factor S from L^T*L where L = S*S^T (or L^H*L where L = S*S^H for complex)
solve_split_cholesky_banded :: proc(
	uplo: MatrixRegion,
	n: int,
	kd: int,
	AB: ^Matrix($T), // Banded matrix (input/output)
) -> (
	info: Info,
	ok: bool,
) where is_float(T) || is_complex(T) {
	uplo_c := matrix_region_to_cstring(uplo)
	n_int := Blas_Int(n)
	kd_int := Blas_Int(kd)
	ldab := AB.ld

	when T == f32 {
		lapack.spbstf_(uplo_c, &n_int, &kd_int, raw_data(AB.data), &ldab, &info, len(uplo_c))
	} else when T == f64 {
		lapack.dpbstf_(uplo_c, &n_int, &kd_int, raw_data(AB.data), &ldab, &info, len(uplo_c))
	} else when T == complex64 {
		lapack.cpbstf_(uplo_c, &n_int, &kd_int, raw_data(AB.data), &ldab, &info, len(uplo_c))
	} else when T == complex128 {
		lapack.zpbstf_(uplo_c, &n_int, &kd_int, raw_data(AB.data), &ldab, &info, len(uplo_c))
	}

	return info, info == 0
}


// ===================================================================================
// ITERATIVE REFINEMENT
// ===================================================================================

// Query result sizes for iterative refinement
query_result_sizes_refine_banded :: proc(
	nrhs: int,
) -> (
	ferr_size: int,
	berr_size: int, // Forward error bounds array// Backward error bounds array
) {
	return nrhs, nrhs
}

// Query workspace for iterative refinement
query_workspace_refine_banded :: proc(
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
	} else when T == complex64 || T == complex128 {
		return Blas_Int(2 * n), Blas_Int(n), 0
	}
}

// Iterative refinement for banded matrix solution (real types)
refine_banded_real :: proc(
	trans: TransposeMode,
	n: int,
	kl: int,
	ku: int,
	nrhs: int,
	AB: ^Matrix($T), // Original banded matrix
	AFB: ^Matrix(T), // Factored matrix from banded_factor
	ipiv: []Blas_Int, // Pivot indices from factorization
	B: ^Matrix(T), // Right-hand side
	X: ^Matrix(T), // Solution (input: initial, output: refined)
	ferr: []T, // Pre-allocated forward error bounds (size nrhs)
	berr: []T, // Pre-allocated backward error bounds (size nrhs)
	work: []T, // Pre-allocated workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	// Validate inputs
	assert(len(berr) >= nrhs, "Backward error array too small")
	assert(len(work) >= 3 * n, "Work array too small")
	assert(len(iwork) >= n, "Integer work array too small")
	assert(len(ipiv) >= n, "Pivot array too small")

	trans_c := transpose_mode_to_cstring(trans)
	n_int := Blas_Int(n)
	kl_int := Blas_Int(kl)
	ku_int := Blas_Int(ku)
	nrhs_int := Blas_Int(nrhs)
	ldab := AB.ld
	ldafb := AFB.ld
	ldb := B.ld
	ldx := X.ld

	when T == f32 {
		lapack.sgbrfs_(
			trans_c,
			&n_int,
			&kl_int,
			&ku_int,
			&nrhs_int,
			raw_data(AB.data),
			&ldab,
			raw_data(AFB.data),
			&ldafb,
			raw_data(ipiv),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			raw_data(ferr),
			raw_data(berr),
			raw_data(work),
			raw_data(iwork),
			&info,
			len(trans_c),
		)
	} else when T == f64 {
		lapack.dgbrfs_(
			trans_c,
			&n_int,
			&kl_int,
			&ku_int,
			&nrhs_int,
			raw_data(AB.data),
			&ldab,
			raw_data(AFB.data),
			&ldafb,
			raw_data(ipiv),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			raw_data(ferr),
			raw_data(berr),
			raw_data(work),
			raw_data(iwork),
			&info,
			len(trans_c),
		)
	}

	return info, info == 0
}

// Iterative refinement for banded matrix solution (complex64)
refine_banded_c64 :: proc(
	trans: TransposeMode,
	n: int,
	kl: int,
	ku: int,
	nrhs: int,
	AB: ^Matrix(complex64), // Original banded matrix
	AFB: ^Matrix(complex64), // Factored matrix from banded_factor
	ipiv: []Blas_Int, // Pivot indices from factorization
	B: ^Matrix(complex64), // Right-hand side
	X: ^Matrix(complex64), // Solution (input: initial, output: refined)
	ferr: []f32, // Pre-allocated forward error bounds (size nrhs)
	berr: []f32, // Pre-allocated backward error bounds (size nrhs)
	work: []complex64, // Pre-allocated workspace
	rwork: []f32, // Pre-allocated real workspace
) -> (
	info: Info,
	ok: bool,
) {
	// Validate inputs
	assert(len(berr) >= nrhs, "Backward error array too small")
	assert(len(work) >= 2 * n, "Work array too small")
	assert(len(rwork) >= n, "Real work array too small")
	assert(len(ipiv) >= n, "Pivot array too small")

	trans_c := transpose_mode_to_cstring(trans)
	n_int := Blas_Int(n)
	kl_int := Blas_Int(kl)
	ku_int := Blas_Int(ku)
	nrhs_int := Blas_Int(nrhs)
	ldab := AB.ld
	ldafb := AFB.ld
	ldb := B.ld
	ldx := X.ld

	lapack.cgbrfs_(
		trans_c,
		&n_int,
		&kl_int,
		&ku_int,
		&nrhs_int,
		raw_data(AB.data),
		&ldab,
		raw_data(AFB.data),
		&ldafb,
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
		len(trans_c),
	)

	return info, info == 0
}

// Iterative refinement for banded matrix solution (complex128)
refine_banded_c128 :: proc(
	trans: TransposeMode,
	n: int,
	kl: int,
	ku: int,
	nrhs: int,
	AB: ^Matrix(complex128), // Original banded matrix
	AFB: ^Matrix(complex128), // Factored matrix from banded_factor
	ipiv: []Blas_Int, // Pivot indices from factorization
	B: ^Matrix(complex128), // Right-hand side
	X: ^Matrix(complex128), // Solution (input: initial, output: refined)
	ferr: []f64, // Pre-allocated forward error bounds (size nrhs)
	berr: []f64, // Pre-allocated backward error bounds (size nrhs)
	work: []complex128, // Pre-allocated workspace
	rwork: []f64, // Pre-allocated real workspace
) -> (
	info: Info,
	ok: bool,
) {
	// Validate inputs
	assert(len(berr) >= nrhs, "Backward error array too small")
	assert(len(work) >= 2 * n, "Work array too small")
	assert(len(rwork) >= n, "Real work array too small")
	assert(len(ipiv) >= n, "Pivot array too small")

	trans_c := transpose_mode_to_cstring(trans)
	n_int := Blas_Int(n)
	kl_int := Blas_Int(kl)
	ku_int := Blas_Int(ku)
	nrhs_int := Blas_Int(nrhs)
	ldab := AB.ld
	ldafb := AFB.ld
	ldb := B.ld
	ldx := X.ld

	lapack.zgbrfs_(
		trans_c,
		&n_int,
		&kl_int,
		&ku_int,
		&nrhs_int,
		raw_data(AB.data),
		&ldab,
		raw_data(AFB.data),
		&ldafb,
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
		len(trans_c),
	)

	return info, info == 0
}

// Query result sizes for extended iterative refinement
query_result_sizes_refine_banded_extended :: proc(
	nrhs: int,
) -> (
	rcond_size: int,
	berr_size: int,
	err_bnds_norm_size: int,
	err_bnds_comp_size: int,
	params_size: int, // Reciprocal condition number (scalar)// Backward error bounds array// Normwise error bounds (contains forward error)// Componentwise error bounds// Algorithm parameters
) {
	n_err_bnds := 3
	return 1, nrhs, nrhs * n_err_bnds, nrhs * n_err_bnds, 3
}

// Query workspace for extended iterative refinement
query_workspace_refine_banded_extended :: proc(
	$T: typeid,
	n: int,
) -> (
	work: Blas_Int,
	rwork: Blas_Int,
	iwork: Blas_Int,
) where is_float(T) ||
	is_complex(T) {
	when is_float(T) {
		return Blas_Int(4 * n), 0, Blas_Int(n)
	} else when T == complex64 || T == complex128 {
		return Blas_Int(2 * n), Blas_Int(2 * n), 0
	}
}

// Extended iterative refinement for banded matrix solution (real types)
refine_banded_extended_real :: proc(
	trans: TransposeMode,
	equed: EquilibrationRequest,
	n: int,
	kl: int,
	ku: int,
	nrhs: int,
	AB: ^Matrix($T), // Original banded matrix
	AFB: ^Matrix(T), // Factored matrix from banded_factor
	ipiv: []Blas_Int, // Pivot indices from factorization
	R: []T, // Row scale factors from equilibration
	C: []T, // Column scale factors from equilibration
	B: ^Matrix(T), // Right-hand side
	X: ^Matrix(T), // Solution (input: initial, output: refined)
	rcond: ^T, // Output: reciprocal condition number
	berr: []T, // Pre-allocated backward error bounds (size nrhs)
	n_err_bnds: ^Blas_Int, // Output: number of error bounds computed
	err_bnds_norm: []T, // Pre-allocated normwise error bounds
	err_bnds_comp: []T, // Pre-allocated componentwise error bounds
	nparams: ^Blas_Int, // Output: number of parameters
	params: []T, // Pre-allocated algorithm parameters
	work: []T, // Pre-allocated workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	// Validate inputs
	assert(len(berr) >= nrhs, "Backward error array too small")
	assert(len(err_bnds_norm) >= nrhs * 3, "Normwise error bounds array too small")
	assert(len(err_bnds_comp) >= nrhs * 3, "Componentwise error bounds array too small")
	assert(len(params) >= 3, "Params array too small")
	assert(len(work) >= 4 * n, "Work array too small")
	assert(len(iwork) >= n, "Integer work array too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(len(R) >= n, "Row scaling array too small")
	assert(len(C) >= n, "Column scaling array too small")

	trans_c := transpose_mode_to_cstring(trans)
	equed_c := equilibration_request_to_cstring(equed)
	n_int := Blas_Int(n)
	kl_int := Blas_Int(kl)
	ku_int := Blas_Int(ku)
	nrhs_int := Blas_Int(nrhs)
	ldab := AB.ld
	ldafb := AFB.ld
	ldb := B.ld
	ldx := X.ld
	n_err_bnds^ = 3

	when T == f32 {
		lapack.sgbrfsx_(
			trans_c,
			equed_c,
			&n_int,
			&kl_int,
			&ku_int,
			&nrhs_int,
			raw_data(AB.data),
			&ldab,
			raw_data(AFB.data),
			&ldafb,
			raw_data(ipiv),
			raw_data(R),
			raw_data(C),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			rcond,
			raw_data(berr),
			n_err_bnds,
			raw_data(err_bnds_norm),
			raw_data(err_bnds_comp),
			nparams,
			raw_data(params),
			raw_data(work),
			raw_data(iwork),
			&info,
			len(trans_c),
			len(equed_c),
		)
	} else when T == f64 {
		lapack.dgbrfsx_(
			trans_c,
			equed_c,
			&n_int,
			&kl_int,
			&ku_int,
			&nrhs_int,
			raw_data(AB.data),
			&ldab,
			raw_data(AFB.data),
			&ldafb,
			raw_data(ipiv),
			raw_data(R),
			raw_data(C),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			rcond,
			raw_data(berr),
			n_err_bnds,
			raw_data(err_bnds_norm),
			raw_data(err_bnds_comp),
			nparams,
			raw_data(params),
			raw_data(work),
			raw_data(iwork),
			&info,
			len(trans_c),
			len(equed_c),
		)
	}

	return info, info == 0
}

// Extended iterative refinement for banded matrix solution (complex64)
refine_banded_extended_c64 :: proc(
	trans: TransposeMode,
	equed: EquilibrationRequest,
	n: int,
	kl: int,
	ku: int,
	nrhs: int,
	AB: ^Matrix(complex64), // Original banded matrix
	AFB: ^Matrix(complex64), // Factored matrix from banded_factor
	ipiv: []Blas_Int, // Pivot indices from factorization
	R: []f32, // Row scale factors from equilibration
	C: []f32, // Column scale factors from equilibration
	B: ^Matrix(complex64), // Right-hand side
	X: ^Matrix(complex64), // Solution (input: initial, output: refined)
	rcond: ^f32, // Output: reciprocal condition number
	berr: []f32, // Pre-allocated backward error bounds (size nrhs)
	n_err_bnds: ^Blas_Int, // Output: number of error bounds computed
	err_bnds_norm: []f32, // Pre-allocated normwise error bounds
	err_bnds_comp: []f32, // Pre-allocated componentwise error bounds
	nparams: ^Blas_Int, // Output: number of parameters
	params: []f32, // Pre-allocated algorithm parameters
	work: []complex64, // Pre-allocated workspace
	rwork: []f32, // Pre-allocated real workspace
) -> (
	info: Info,
	ok: bool,
) {
	// Validate inputs
	assert(len(berr) >= nrhs, "Backward error array too small")
	assert(len(err_bnds_norm) >= nrhs * 3, "Normwise error bounds array too small")
	assert(len(err_bnds_comp) >= nrhs * 3, "Componentwise error bounds array too small")
	assert(len(params) >= 3, "Params array too small")
	assert(len(work) >= 2 * n, "Work array too small")
	assert(len(rwork) >= 2 * n, "Real work array too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(len(R) >= n, "Row scaling array too small")
	assert(len(C) >= n, "Column scaling array too small")

	trans_c := transpose_mode_to_cstring(trans)
	equed_c := equilibration_request_to_cstring(equed)
	n_int := Blas_Int(n)
	kl_int := Blas_Int(kl)
	ku_int := Blas_Int(ku)
	nrhs_int := Blas_Int(nrhs)
	ldab := AB.ld
	ldafb := AFB.ld
	ldb := B.ld
	ldx := X.ld
	n_err_bnds^ = 3

	lapack.cgbrfsx_(
		trans_c,
		equed_c,
		&n_int,
		&kl_int,
		&ku_int,
		&nrhs_int,
		raw_data(AB.data),
		&ldab,
		raw_data(AFB.data),
		&ldafb,
		raw_data(ipiv),
		raw_data(R),
		raw_data(C),
		raw_data(B.data),
		&ldb,
		raw_data(X.data),
		&ldx,
		rcond,
		raw_data(berr),
		n_err_bnds,
		raw_data(err_bnds_norm),
		raw_data(err_bnds_comp),
		nparams,
		raw_data(params),
		raw_data(work),
		raw_data(rwork),
		&info,
		len(trans_c),
		len(equed_c),
	)

	return info, info == 0
}

// Extended iterative refinement for banded matrix solution (complex128)
refine_banded_extended_c128 :: proc(
	trans: TransposeMode,
	equed: EquilibrationRequest,
	n: int,
	kl: int,
	ku: int,
	nrhs: int,
	AB: ^Matrix(complex128), // Original banded matrix
	AFB: ^Matrix(complex128), // Factored matrix from banded_factor
	ipiv: []Blas_Int, // Pivot indices from factorization
	R: []f64, // Row scale factors from equilibration
	C: []f64, // Column scale factors from equilibration
	B: ^Matrix(complex128), // Right-hand side
	X: ^Matrix(complex128), // Solution (input: initial, output: refined)
	rcond: ^f64, // Output: reciprocal condition number
	berr: []f64, // Pre-allocated backward error bounds (size nrhs)
	n_err_bnds: ^Blas_Int, // Output: number of error bounds computed
	err_bnds_norm: []f64, // Pre-allocated normwise error bounds
	err_bnds_comp: []f64, // Pre-allocated componentwise error bounds
	nparams: ^Blas_Int, // Output: number of parameters
	params: []f64, // Pre-allocated algorithm parameters
	work: []complex128, // Pre-allocated workspace
	rwork: []f64, // Pre-allocated real workspace
) -> (
	info: Info,
	ok: bool,
) {
	// Validate inputs
	assert(len(berr) >= nrhs, "Backward error array too small")
	assert(len(err_bnds_norm) >= nrhs * 3, "Normwise error bounds array too small")
	assert(len(err_bnds_comp) >= nrhs * 3, "Componentwise error bounds array too small")
	assert(len(params) >= 3, "Params array too small")
	assert(len(work) >= 2 * n, "Work array too small")
	assert(len(rwork) >= 2 * n, "Real work array too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(len(R) >= n, "Row scaling array too small")
	assert(len(C) >= n, "Column scaling array too small")

	trans_c := transpose_mode_to_cstring(trans)
	equed_c := equilibration_request_to_cstring(equed)
	n_int := Blas_Int(n)
	kl_int := Blas_Int(kl)
	ku_int := Blas_Int(ku)
	nrhs_int := Blas_Int(nrhs)
	ldab := AB.ld
	ldafb := AFB.ld
	ldb := B.ld
	ldx := X.ld
	n_err_bnds^ = 3

	lapack.zgbrfsx_(
		trans_c,
		equed_c,
		&n_int,
		&kl_int,
		&ku_int,
		&nrhs_int,
		raw_data(AB.data),
		&ldab,
		raw_data(AFB.data),
		&ldafb,
		raw_data(ipiv),
		raw_data(R),
		raw_data(C),
		raw_data(B.data),
		&ldb,
		raw_data(X.data),
		&ldx,
		rcond,
		raw_data(berr),
		n_err_bnds,
		raw_data(err_bnds_norm),
		raw_data(err_bnds_comp),
		nparams,
		raw_data(params),
		raw_data(work),
		raw_data(rwork),
		&info,
		len(trans_c),
		len(equed_c),
	)

	return info, info == 0
}
