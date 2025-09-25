package openblas

import lapack "./f77"
import "base:builtin"
import "core:mem"

// ===================================================================================
// POSITIVE DEFINITE BANDED LINEAR SYSTEM SOLVERS
// ===================================================================================

// Expert solver for positive definite banded systems proc group
solve_banded_pd_expert :: proc {
	solve_banded_pd_expert_f32_c64,
	solve_banded_pd_expert_f64_c128,
}


banded_factor :: proc {
	banded_factor_real,
	banded_factor_c64,
	banded_factor_c128,
}

solve_banded_factored :: proc {
	solve_banded_factored_real,
	solve_banded_factored_c64,
	solve_banded_factored_c128,
}

solve_banded :: proc {
	solve_banded_real,
	solve_banded_c64,
	solve_banded_c128,
}

solve_banded_expert :: proc {
	solve_banded_expert_real,
	solve_banded_expert_c64,
	solve_banded_expert_c128,
}

solve_banded_expert_extended :: proc {
	solve_banded_expert_extended_real,
	solve_banded_expert_extended_c64,
	solve_banded_expert_extended_c128,
}


// ===================================================================================
// SIMPLE SOLVER IMPLEMENTATION
// ===================================================================================
// Solve positive definite banded system (generic)
// Solves A*X = B where A is positive definite banded
solve_banded_pd :: proc(
	uplo: MatrixRegion,
	n: int,
	kd: int,
	nrhs: int,
	AB: ^Matrix($T), // Banded matrix (input/output - factorized on output)
	B: ^Matrix(T), // Right-hand side (input/output - solution on output)
) -> (
	info: Info,
	ok: bool,
) where is_float(T) || is_complex(T) {
	uplo_c := matrix_region_to_cstring(uplo)
	n := Blas_Int(n)
	kd := Blas_Int(kd)
	nrhs := Blas_Int(nrhs)
	ldab := AB.ld
	ldb := B.ld

	when T == f32 {
		lapack.spbsv_(
			uplo_c,
			&n,
			&kd,
			&nrhs,
			raw_data(AB.data),
			&ldab,
			raw_data(B.data),
			&ldb,
			&info,
			len(uplo_c),
		)
	} else when T == f64 {
		lapack.dpbsv_(
			uplo_c,
			&n,
			&kd,
			&nrhs,
			raw_data(AB.data),
			&ldab,
			raw_data(B.data),
			&ldb,
			&info,
			len(uplo_c),
		)
	} else when T == complex64 {
		lapack.cpbsv_(
			uplo_c,
			&n,
			&kd,
			&nrhs,
			raw_data(AB.data),
			&ldab,
			raw_data(B.data),
			&ldb,
			&info,
			len(uplo_c),
		)
	} else when T == complex128 {
		lapack.zpbsv_(
			uplo_c,
			&n,
			&kd,
			&nrhs,
			raw_data(AB.data),
			&ldab,
			raw_data(B.data),
			&ldb,
			&info,
			len(uplo_c),
		)
	}

	return info, info == 0
}

// ===================================================================================
// EXPERT SOLVER IMPLEMENTATION
// ===================================================================================

// Query result sizes for expert positive definite banded solver
query_result_sizes_banded_pd_expert :: proc(
	n: int,
	nrhs: int,
) -> (
	S_size: int,
	ferr_size: int,
	berr_size: int,// Scaling factors array
	rcond_size: int,// Forward error bounds array
	equed_size: int, // Backward error bounds array// Reciprocal condition number (scalar)// Equilibration state (single byte)
) {
	return n, nrhs, nrhs, 1, 1
}

// Query workspace for expert positive definite banded solver
query_workspace_banded_pd_expert :: proc(
	$T: typeid,
	n: int,
	nrhs: int,
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

// Expert solve for positive definite banded system (f32/complex64)
// Solves with equilibration, condition estimation, and error bounds
solve_banded_pd_expert_f32_c64 :: proc(
	fact: FactorizationOption,
	uplo: MatrixRegion,
	n: int,
	kd: int,
	nrhs: int,
	AB: ^Matrix($T), // Banded matrix (input/output)
	AFB: ^Matrix(T), // Factored matrix (input/output)
	equed: ^byte, // Equilibration state (input/output)
	S: []f32, // Scaling factors (input/output)
	B: ^Matrix(T), // Right-hand side (input/output)
	X: ^Matrix(T), // Solution matrix (output)
	rcond: ^f32, // Reciprocal condition number (output)
	ferr: []f32, // Forward error bounds (output)
	berr: []f32, // Backward error bounds (output)
	work: []T, // Workspace
	rwork: []f32 = nil, // Real workspace (complex only)
	iwork: []Blas_Int = nil, // Integer workspace (real only)
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == complex64 {
	// Validate inputs
	assert(len(S) >= n, "Scaling array too small")
	assert(len(ferr) >= nrhs, "Forward error array too small")
	assert(len(berr) >= nrhs, "Backward error array too small")
	when T == f32 {
		assert(len(work) >= 3 * n, "Work array too small")
		assert(len(iwork) >= n, "Integer work array too small")
	} else when T == complex64 {
		assert(len(work) >= 2 * n, "Work array too small")
		assert(len(rwork) >= n, "Real work array too small")
	}

	// Prepare parameters
	fact_c := _factorization_to_char(fact)
	uplo_c := matrix_region_to_cstring(uplo)
	n := Blas_Int(n)
	kd := Blas_Int(kd)
	nrhs := Blas_Int(nrhs)
	ldab := AB.ld
	ldafb := AFB.ld
	ldb := B.ld
	ldx := X.ld

	when T == f32 {
		lapack.spbsvx_(
			fact_c,
			uplo_c,
			&n,
			&kd,
			&nrhs,
			raw_data(AB.data),
			&ldab,
			raw_data(AFB.data),
			&ldafb,
			equed,
			raw_data(S),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			rcond,
			raw_data(ferr),
			raw_data(berr),
			raw_data(work),
			raw_data(iwork),
			&info,
			len(fact_c),
			len(uplo_c),
			1,
		)
	} else when T == complex64 {
		lapack.cpbsvx_(
			fact_c,
			uplo_c,
			&n,
			&kd,
			&nrhs,
			raw_data(AB.data),
			&ldab,
			raw_data(AFB.data),
			&ldafb,
			equed,
			raw_data(S),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			rcond,
			raw_data(ferr),
			raw_data(berr),
			raw_data(work),
			raw_data(rwork),
			&info,
			len(fact_c),
			len(uplo_c),
			1,
		)
	}

	return info, info == 0
}

// Expert solve for positive definite banded system (f64/complex128)
// Solves with equilibration, condition estimation, and error bounds
solve_banded_pd_expert_f64_c128 :: proc(
	fact: FactorizationOption,
	uplo: MatrixRegion,
	n: int,
	kd: int,
	nrhs: int,
	AB: ^Matrix($T), // Banded matrix (input/output)
	AFB: ^Matrix(T), // Factored matrix (input/output)
	equed: ^byte, // Equilibration state (input/output)
	S: []f64, // Scaling factors (input/output)
	B: ^Matrix(T), // Right-hand side (input/output)
	X: ^Matrix(T), // Solution matrix (output)
	rcond: ^f64, // Reciprocal condition number (output)
	ferr: []f64, // Forward error bounds (output)
	berr: []f64, // Backward error bounds (output)
	work: []T, // Workspace
	rwork: []f64 = nil, // Real workspace (complex only)
	iwork: []Blas_Int = nil, // Integer workspace (real only)
) -> (
	info: Info,
	ok: bool,
) where T == f64 || T == complex128 {
	// Validate inputs
	assert(len(S) >= n, "Scaling array too small")
	assert(len(ferr) >= nrhs, "Forward error array too small")
	assert(len(berr) >= nrhs, "Backward error array too small")
	when T == f64 {
		assert(len(work) >= 3 * n, "Work array too small")
		assert(len(iwork) >= n, "Integer work array too small")
	} else when T == complex128 {
		assert(len(work) >= 2 * n, "Work array too small")
		assert(len(rwork) >= n, "Real work array too small")
	}

	// Prepare parameters
	fact_c := _factorization_to_char(fact)
	uplo_c := matrix_region_to_cstring(uplo)
	n := Blas_Int(n)
	kd := Blas_Int(kd)
	nrhs := Blas_Int(nrhs)
	ldab := AB.ld
	ldafb := AFB.ld
	ldb := B.ld
	ldx := X.ld

	when T == f64 {
		lapack.dpbsvx_(
			fact_c,
			uplo_c,
			&n,
			&kd,
			&nrhs,
			raw_data(AB.data),
			&ldab,
			raw_data(AFB.data),
			&ldafb,
			equed,
			raw_data(S),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			rcond,
			raw_data(ferr),
			raw_data(berr),
			raw_data(work),
			raw_data(iwork),
			&info,
			len(fact_c),
			len(uplo_c),
			1,
		)
	} else when T == complex128 {
		lapack.zpbsvx_(
			fact_c,
			uplo_c,
			&n,
			&kd,
			&nrhs,
			raw_data(AB.data),
			&ldab,
			raw_data(AFB.data),
			&ldafb,
			equed,
			raw_data(S),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			rcond,
			raw_data(ferr),
			raw_data(berr),
			raw_data(work),
			raw_data(rwork),
			&info,
			len(fact_c),
			len(uplo_c),
			1,
		)
	}

	return info, info == 0
}


// ===================================================================================
// FACTORIZATION
// ===================================================================================

// Query result sizes for banded LU factorization
query_result_sizes_banded_factor :: proc(
	m: int,
	n: int,
) -> (
	ipiv_size: int, // Pivot indices array
) {
	return min(m, n)
}

// LU factorization doesn't need workspace
query_workspace_banded_factor :: proc(
	$T: typeid,
	m: int,
	n: int,
) -> (
	work: Blas_Int,
	rwork: Blas_Int,
) where is_float(T) ||
	is_complex(T) {
	return 0, 0
}

// LU factorization of banded matrix (real version)
banded_factor_real :: proc(
	AB: ^Matrix($T), // Banded matrix (overwritten with LU factorization)
	ipiv: []Blas_Int, // Pre-allocated pivot indices (size min(m,n))
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
	assert(len(ipiv) >= min(int(m), int(n)), "Pivot array too small")

	when T == f32 {
		lapack.sgbtrf_(&m, &n, &kl, &ku, raw_data(AB.data), &ldab, raw_data(ipiv), &info)
	} else when T == f64 {
		lapack.dgbtrf_(&m, &n, &kl, &ku, raw_data(AB.data), &ldab, raw_data(ipiv), &info)
	}

	return info, info == 0
}

// LU factorization of banded matrix (complex64 version)
banded_factor_c64 :: proc(
	AB: ^Matrix(complex64), // Banded matrix (overwritten with LU factorization)
	ipiv: []Blas_Int, // Pre-allocated pivot indices (size min(m,n))
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
	assert(len(ipiv) >= min(int(m), int(n)), "Pivot array too small")

	lapack.cgbtrf_(&m, &n, &kl, &ku, raw_data(AB.data), &ldab, raw_data(ipiv), &info)

	return info, info == 0
}

// LU factorization of banded matrix (complex128 version)
banded_factor_c128 :: proc(
	AB: ^Matrix(complex128), // Banded matrix (overwritten with LU factorization)
	ipiv: []Blas_Int, // Pre-allocated pivot indices (size min(m,n))
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
	assert(len(ipiv) >= min(int(m), int(n)), "Pivot array too small")

	lapack.zgbtrf_(&m, &n, &kl, &ku, raw_data(AB.data), &ldab, raw_data(ipiv), &info)

	return info, info == 0
}

// Solve using pre-computed LU factorization (real version)
solve_banded_factored_real :: proc(
	AB: ^Matrix($T), // LU factorization from banded_factor
	ipiv: []Blas_Int, // Pivot indices from banded_factor
	B: ^Matrix(T), // Right-hand side (overwritten with solution)
	trans := TransposeMode.None, // Transpose mode
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	assert(AB.format == .Banded, "Matrix must be in banded format")

	n := AB.cols
	kl := AB.storage.banded.kl
	ku := AB.storage.banded.ku
	nrhs := B.cols
	ldab := AB.ld
	ldb := B.ld

	trans_str := transpose_mode_to_cstring(trans)

	when T == f32 {
		lapack.sgbtrs_(
			trans_str,
			&n,
			&kl,
			&ku,
			&nrhs,
			raw_data(AB.data),
			&ldab,
			raw_data(ipiv),
			raw_data(B.data),
			&ldb,
			&info,
			1,
		)
	} else when T == f64 {
		lapack.dgbtrs_(
			trans_str,
			&n,
			&kl,
			&ku,
			&nrhs,
			raw_data(AB.data),
			&ldab,
			raw_data(ipiv),
			raw_data(B.data),
			&ldb,
			&info,
			1,
		)
	}

	return info, info == 0
}

// Solve using pre-computed LU factorization (complex64 version)
solve_banded_factored_c64 :: proc(
	AB: ^Matrix(complex64), // LU factorization from banded_factor
	ipiv: []Blas_Int, // Pivot indices from banded_factor
	B: ^Matrix(complex64), // Right-hand side (overwritten with solution)
	trans := TransposeMode.None, // Transpose mode
) -> (
	info: Info,
	ok: bool,
) {
	assert(AB.format == .Banded, "Matrix must be in banded format")

	n := AB.cols
	kl := AB.storage.banded.kl
	ku := AB.storage.banded.ku
	nrhs := B.cols
	ldab := AB.ld
	ldb := B.ld

	trans_str := transpose_mode_to_cstring(trans)

	lapack.cgbtrs_(
		trans_str,
		&n,
		&kl,
		&ku,
		&nrhs,
		raw_data(AB.data),
		&ldab,
		raw_data(ipiv),
		raw_data(B.data),
		&ldb,
		&info,
		1,
	)

	return info, info == 0
}

// Solve using pre-computed LU factorization (complex128 version)
solve_banded_factored_c128 :: proc(
	AB: ^Matrix(complex128), // LU factorization from banded_factor
	ipiv: []Blas_Int, // Pivot indices from banded_factor
	B: ^Matrix(complex128), // Right-hand side (overwritten with solution)
	trans := TransposeMode.None, // Transpose mode
) -> (
	info: Info,
	ok: bool,
) {
	assert(AB.format == .Banded, "Matrix must be in banded format")

	n := AB.cols
	kl := AB.storage.banded.kl
	ku := AB.storage.banded.ku
	nrhs := B.cols
	ldab := AB.ld
	ldb := B.ld

	trans_str := transpose_mode_to_cstring(trans)

	lapack.zgbtrs_(
		trans_str,
		&n,
		&kl,
		&ku,
		&nrhs,
		raw_data(AB.data),
		&ldab,
		raw_data(ipiv),
		raw_data(B.data),
		&ldb,
		&info,
		1,
	)

	return info, info == 0
}

// ===================================================================================
// LINEAR SYSTEM SOLVERS
// ===================================================================================

// Query result sizes for banded solve
query_result_sizes_solve_banded :: proc(
	n: int,
) -> (
	ipiv_size: int, // Pivot indices array
) {
	return n
}

// Solve banded linear system (real version)
// Solves AB * X = B, where AB is a banded matrix
solve_banded_real :: proc(
	AB: ^Matrix($T), // Banded matrix (overwritten with LU factorization)
	B: ^Matrix(T), // Right-hand side (overwritten with solution)
	ipiv: []Blas_Int, // Pre-allocated pivot indices (size n)
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	assert(AB.format == .Banded, "Matrix must be in banded format")

	n := AB.cols
	kl := AB.storage.banded.kl
	ku := AB.storage.banded.ku
	nrhs := B.cols
	ldab := AB.ld
	ldb := B.ld

	// Validate inputs
	assert(len(ipiv) >= int(n), "Pivot array too small")

	when T == f32 {
		lapack.sgbsv_(
			&n,
			&kl,
			&ku,
			&nrhs,
			raw_data(AB.data),
			&ldab,
			raw_data(ipiv),
			raw_data(B.data),
			&ldb,
			&info,
		)
	} else when T == f64 {
		lapack.dgbsv_(
			&n,
			&kl,
			&ku,
			&nrhs,
			raw_data(AB.data),
			&ldab,
			raw_data(ipiv),
			raw_data(B.data),
			&ldb,
			&info,
		)
	}

	return info, info == 0
}

// Solve banded linear system (complex64 version)
// Solves AB * X = B, where AB is a banded matrix
solve_banded_c64 :: proc(
	AB: ^Matrix(complex64), // Banded matrix (overwritten with LU factorization)
	B: ^Matrix(complex64), // Right-hand side (overwritten with solution)
	ipiv: []Blas_Int, // Pre-allocated pivot indices (size n)
) -> (
	info: Info,
	ok: bool,
) {
	assert(AB.format == .Banded, "Matrix must be in banded format")

	n := AB.cols
	kl := AB.storage.banded.kl
	ku := AB.storage.banded.ku
	nrhs := B.cols
	ldab := AB.ld
	ldb := B.ld

	// Validate inputs
	assert(len(ipiv) >= int(n), "Pivot array too small")

	lapack.cgbsv_(
		&n,
		&kl,
		&ku,
		&nrhs,
		raw_data(AB.data),
		&ldab,
		raw_data(ipiv),
		raw_data(B.data),
		&ldb,
		&info,
	)

	return info, info == 0
}

// Solve banded linear system (complex128 version)
// Solves AB * X = B, where AB is a banded matrix
solve_banded_c128 :: proc(
	AB: ^Matrix(complex128), // Banded matrix (overwritten with LU factorization)
	B: ^Matrix(complex128), // Right-hand side (overwritten with solution)
	ipiv: []Blas_Int, // Pre-allocated pivot indices (size n)
) -> (
	info: Info,
	ok: bool,
) {
	assert(AB.format == .Banded, "Matrix must be in banded format")

	n := AB.cols
	kl := AB.storage.banded.kl
	ku := AB.storage.banded.ku
	nrhs := B.cols
	ldab := AB.ld
	ldb := B.ld

	// Validate inputs
	assert(len(ipiv) >= int(n), "Pivot array too small")

	lapack.zgbsv_(
		&n,
		&kl,
		&ku,
		&nrhs,
		raw_data(AB.data),
		&ldab,
		raw_data(ipiv),
		raw_data(B.data),
		&ldb,
		&info,
	)

	return info, info == 0
}

// ===================================================================================
// EXPERT SOLVERS
// ===================================================================================

// Query result sizes for expert banded solve
query_result_sizes_solve_banded_expert :: proc(
	n: int,
	kl: int,
	ku: int,
	nrhs: int,
) -> (
	ipiv_size: int,
	AFB_rows: int,
	AFB_cols: int,
	R_size: int,
	C_size: int,
	X_rows: int,
	X_cols: int,
	ferr_size: int,
	berr_size: int,
	rcond_size: int,
	equed_size: int, // Pivot indices array// Factored matrix rows// Factored matrix columns// Row scale factors// Column scale factors// Solution matrix rows// Solution matrix columns// Forward error bounds// Backward error bounds// Reciprocal condition number (scalar)// Equilibration state (single byte)
) {
	return n, 2 * kl + ku + 1, n, n, n, n, nrhs, nrhs, nrhs, 1, 1
}

// Query workspace for expert banded solve
query_workspace_solve_banded_expert :: proc(
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

// Expert solve for banded system (real version)
solve_banded_expert_real :: proc(
	fact: FactorizationOption,
	trans: TransposeMode,
	AB: ^Matrix($T), // Banded matrix (input/output based on fact)
	AFB: ^Matrix(T), // Pre-allocated factored matrix (input/output based on fact)
	ipiv: []Blas_Int, // Pre-allocated pivot indices (input/output based on fact)
	equed: ^EquilibrationRequest, // Equilibration state (input/output)
	R: []T, // Pre-allocated row scale factors (input/output)
	C: []T, // Pre-allocated column scale factors (input/output)
	B: ^Matrix(T), // Right-hand side (input/output)
	X: ^Matrix(T), // Pre-allocated solution matrix (output)
	rcond: ^T, // Output: reciprocal condition number
	ferr: []T, // Pre-allocated forward error bounds (output)
	berr: []T, // Pre-allocated backward error bounds (output)
	work: []T, // Pre-allocated workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	assert(AB.format == .Banded, "Matrix must be in banded format")
	assert(AFB.format == .Banded, "Factored matrix must be in banded format")

	n := AB.cols
	kl := AB.storage.banded.kl
	ku := AB.storage.banded.ku
	nrhs := B.cols
	ldab := AB.ld
	ldafb := AFB.ld
	ldb := B.ld
	ldx := X.ld

	// Validate inputs
	assert(len(ipiv) >= int(n), "Pivot array too small")
	assert(len(R) >= int(n), "Row scale array too small")
	assert(len(C) >= int(n), "Column scale array too small")
	assert(len(ferr) >= int(nrhs), "Forward error array too small")
	assert(len(berr) >= int(nrhs), "Backward error array too small")
	assert(len(work) >= 3 * int(n), "Work array too small")
	assert(len(iwork) >= int(n), "Integer work array too small")

	fact_c := _factorization_to_char(fact)
	trans_c := transpose_mode_to_cstring(trans)

	// Convert equilibration enum to byte for LAPACK
	equed_byte := equilibration_request_to_char(equed^)

	when T == f32 {
		lapack.sgbsvx_(
			fact_c,
			trans_c,
			&n,
			&kl,
			&ku,
			&nrhs,
			raw_data(AB.data),
			&ldab,
			raw_data(AFB.data),
			&ldafb,
			raw_data(ipiv),
			&equed_byte,
			raw_data(R),
			raw_data(C),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			rcond,
			raw_data(ferr),
			raw_data(berr),
			raw_data(work),
			raw_data(iwork),
			&info,
			len(fact_c),
			len(trans_c),
			1,
		)
	} else when T == f64 {
		lapack.dgbsvx_(
			fact_c,
			trans_c,
			&n,
			&kl,
			&ku,
			&nrhs,
			raw_data(AB.data),
			&ldab,
			raw_data(AFB.data),
			&ldafb,
			raw_data(ipiv),
			&equed_byte,
			raw_data(R),
			raw_data(C),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			rcond,
			raw_data(ferr),
			raw_data(berr),
			raw_data(work),
			raw_data(iwork),
			&info,
			len(fact_c),
			len(trans_c),
			1,
		)
	}

	// Convert byte back to equilibration enum
	equed^ = equilibration_request_from_char(equed_byte)

	return info, info == 0
}

// Expert solve for banded system (complex64 version)
solve_banded_expert_c64 :: proc(
	fact: FactorizationOption,
	trans: TransposeMode,
	AB: ^Matrix(complex64), // Banded matrix (input/output based on fact)
	AFB: ^Matrix(complex64), // Pre-allocated factored matrix (input/output based on fact)
	ipiv: []Blas_Int, // Pre-allocated pivot indices (input/output based on fact)
	equed: ^EquilibrationRequest, // Equilibration state (input/output)
	R: []f32, // Pre-allocated row scale factors (input/output)
	C: []f32, // Pre-allocated column scale factors (input/output)
	B: ^Matrix(complex64), // Right-hand side (input/output)
	X: ^Matrix(complex64), // Pre-allocated solution matrix (output)
	rcond: ^f32, // Output: reciprocal condition number
	ferr: []f32, // Pre-allocated forward error bounds (output)
	berr: []f32, // Pre-allocated backward error bounds (output)
	work: []complex64, // Pre-allocated workspace
	rwork: []f32, // Pre-allocated real workspace
) -> (
	info: Info,
	ok: bool,
) {
	assert(AB.format == .Banded, "Matrix must be in banded format")
	assert(AFB.format == .Banded, "Factored matrix must be in banded format")

	n := AB.cols
	kl := AB.storage.banded.kl
	ku := AB.storage.banded.ku
	nrhs := B.cols
	ldab := AB.ld
	ldafb := AFB.ld
	ldb := B.ld
	ldx := X.ld

	// Validate inputs
	assert(len(ipiv) >= int(n), "Pivot array too small")
	assert(len(R) >= int(n), "Row scale array too small")
	assert(len(C) >= int(n), "Column scale array too small")
	assert(len(ferr) >= int(nrhs), "Forward error array too small")
	assert(len(berr) >= int(nrhs), "Backward error array too small")
	assert(len(work) >= 2 * int(n), "Work array too small")
	assert(len(rwork) >= int(n), "Real work array too small")

	fact_c := _factorization_to_char(fact)
	trans_c := transpose_mode_to_cstring(trans)

	// Convert equilibration enum to byte for LAPACK
	equed_byte := equilibration_request_to_char(equed^)

	lapack.cgbsvx_(
		fact_c,
		trans_c,
		&n,
		&kl,
		&ku,
		&nrhs,
		raw_data(AB.data),
		&ldab,
		raw_data(AFB.data),
		&ldafb,
		raw_data(ipiv),
		&equed_byte,
		raw_data(R),
		raw_data(C),
		raw_data(B.data),
		&ldb,
		raw_data(X.data),
		&ldx,
		rcond,
		raw_data(ferr),
		raw_data(berr),
		raw_data(work),
		raw_data(rwork),
		&info,
		len(fact_c),
		len(trans_c),
		1,
	)

	// Convert byte back to equilibration enum
	equed^ = equilibration_request_from_char(equed_byte)

	return info, info == 0
}

// Expert solve for banded system (complex128 version)
solve_banded_expert_c128 :: proc(
	fact: FactorizationOption,
	trans: TransposeMode,
	AB: ^Matrix(complex128), // Banded matrix (input/output based on fact)
	AFB: ^Matrix(complex128), // Pre-allocated factored matrix (input/output based on fact)
	ipiv: []Blas_Int, // Pre-allocated pivot indices (input/output based on fact)
	equed: ^EquilibrationRequest, // Equilibration state (input/output)
	R: []f64, // Pre-allocated row scale factors (input/output)
	C: []f64, // Pre-allocated column scale factors (input/output)
	B: ^Matrix(complex128), // Right-hand side (input/output)
	X: ^Matrix(complex128), // Pre-allocated solution matrix (output)
	rcond: ^f64, // Output: reciprocal condition number
	ferr: []f64, // Pre-allocated forward error bounds (output)
	berr: []f64, // Pre-allocated backward error bounds (output)
	work: []complex128, // Pre-allocated workspace
	rwork: []f64, // Pre-allocated real workspace
) -> (
	info: Info,
	ok: bool,
) {
	assert(AB.format == .Banded, "Matrix must be in banded format")
	assert(AFB.format == .Banded, "Factored matrix must be in banded format")

	n := AB.cols
	kl := AB.storage.banded.kl
	ku := AB.storage.banded.ku
	nrhs := B.cols
	ldab := AB.ld
	ldafb := AFB.ld
	ldb := B.ld
	ldx := X.ld

	// Validate inputs
	assert(len(ipiv) >= int(n), "Pivot array too small")
	assert(len(R) >= int(n), "Row scale array too small")
	assert(len(C) >= int(n), "Column scale array too small")
	assert(len(ferr) >= int(nrhs), "Forward error array too small")
	assert(len(berr) >= int(nrhs), "Backward error array too small")
	assert(len(work) >= 2 * int(n), "Work array too small")
	assert(len(rwork) >= int(n), "Real work array too small")

	fact_c := _factorization_to_char(fact)
	trans_c := transpose_mode_to_cstring(trans)

	// Convert equilibration enum to byte for LAPACK
	equed_byte := equilibration_request_to_char(equed^)

	lapack.zgbsvx_(
		fact_c,
		trans_c,
		&n,
		&kl,
		&ku,
		&nrhs,
		raw_data(AB.data),
		&ldab,
		raw_data(AFB.data),
		&ldafb,
		raw_data(ipiv),
		&equed_byte,
		raw_data(R),
		raw_data(C),
		raw_data(B.data),
		&ldb,
		raw_data(X.data),
		&ldx,
		rcond,
		raw_data(ferr),
		raw_data(berr),
		raw_data(work),
		raw_data(rwork),
		&info,
		len(fact_c),
		len(trans_c),
		1,
	)

	// Convert byte back to equilibration enum
	equed^ = equilibration_request_from_char(equed_byte)

	return info, info == 0
}

// ===================================================================================
// EXTENDED EXPERT SOLVERS
// ===================================================================================

// Query result sizes for extended expert banded solve
query_result_sizes_solve_banded_expert_extended :: proc(
	n: int,
	kl: int,
	ku: int,
	nrhs: int,
	n_err_bnds: int,
) -> (
	ipiv_size: int,
	AFB_rows: int,
	AFB_cols: int,
	R_size: int,
	C_size: int,
	X_rows: int,
	X_cols: int,
	berr_size: int,
	err_bnds_norm_size: int,
	err_bnds_comp_size: int,
	params_size: int,
	rcond_size: int,
	rpvgrw_size: int,
	equed_size: int, // Pivot indices array// Factored matrix rows// Factored matrix columns// Row scale factors// Column scale factors// Solution matrix rows// Solution matrix columns// Backward error bounds// Normwise error bounds// Componentwise error bounds// Algorithm parameters// Reciprocal condition number (scalar)// Reciprocal pivot growth (scalar)// Equilibration state (single byte)
) {
	return n,
		2 * kl + ku + 1,
		n,
		n,
		n,
		n,
		nrhs,
		nrhs,
		nrhs * n_err_bnds,
		nrhs * n_err_bnds,
		3,
		1,
		1,
		1
}

// Query workspace for extended expert banded solve
query_workspace_solve_banded_expert_extended :: proc(
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

// Extended expert solve for banded system (real version)
solve_banded_expert_extended_real :: proc(
	fact: FactorizationOption,
	trans: TransposeMode,
	AB: ^Matrix($T), // Banded matrix (input/output based on fact)
	AFB: ^Matrix(T), // Pre-allocated factored matrix (input/output based on fact)
	ipiv: []Blas_Int, // Pre-allocated pivot indices (input/output based on fact)
	equed: ^EquilibrationRequest, // Equilibration state (input/output)
	R: []T, // Pre-allocated row scale factors (input/output)
	C: []T, // Pre-allocated column scale factors (input/output)
	B: ^Matrix(T), // Right-hand side (input/output)
	X: ^Matrix(T), // Pre-allocated solution matrix (output)
	rcond: ^T, // Output: reciprocal condition number
	rpvgrw: ^T, // Output: reciprocal pivot growth factor
	berr: []T, // Pre-allocated backward error bounds (output)
	n_err_bnds: Blas_Int, // Number of error bounds to compute
	err_bnds_norm: []T, // Pre-allocated normwise error bounds (output)
	err_bnds_comp: []T, // Pre-allocated componentwise error bounds (output)
	params: []T, // Algorithm parameters (input/output)
	work: []T, // Pre-allocated workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	assert(AB.format == .Banded, "Matrix must be in banded format")
	assert(AFB.format == .Banded, "Factored matrix must be in banded format")

	n := AB.cols
	kl := AB.storage.banded.kl
	ku := AB.storage.banded.ku
	nrhs := B.cols
	ldab := AB.ld
	ldafb := AFB.ld
	ldb := B.ld
	ldx := X.ld
	n_err_bnds := n_err_bnds

	// Validate inputs
	assert(len(ipiv) >= int(n), "Pivot array too small")
	assert(len(R) >= int(n), "Row scale array too small")
	assert(len(C) >= int(n), "Column scale array too small")
	assert(len(berr) >= int(nrhs), "Backward error array too small")
	assert(len(err_bnds_norm) >= int(nrhs * n_err_bnds), "Normwise error bounds array too small")
	assert(
		len(err_bnds_comp) >= int(nrhs * n_err_bnds),
		"Componentwise error bounds array too small",
	)
	assert(len(params) >= 3, "Parameters array too small")
	assert(len(work) >= 4 * int(n), "Work array too small")
	assert(len(iwork) >= int(n), "Integer work array too small")

	fact_c := _factorization_to_char(fact)
	trans_c := transpose_mode_to_cstring(trans)

	// Convert equilibration enum to byte for LAPACK
	equed_byte := equilibration_request_to_char(equed^)

	// Set nparams
	nparams := Blas_Int(len(params))

	when T == f32 {
		lapack.sgbsvxx_(
			fact_c,
			trans_c,
			&n,
			&kl,
			&ku,
			&nrhs,
			raw_data(AB.data),
			&ldab,
			raw_data(AFB.data),
			&ldafb,
			raw_data(ipiv),
			&equed_byte,
			raw_data(R),
			raw_data(C),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			rcond,
			rpvgrw,
			raw_data(berr),
			&n_err_bnds,
			raw_data(err_bnds_norm),
			raw_data(err_bnds_comp),
			&nparams,
			raw_data(params),
			raw_data(work),
			raw_data(iwork),
			&info,
			len(fact_c),
			len(trans_c),
			1,
		)
	} else when T == f64 {
		lapack.dgbsvxx_(
			fact_c,
			trans_c,
			&n,
			&kl,
			&ku,
			&nrhs,
			raw_data(AB.data),
			&ldab,
			raw_data(AFB.data),
			&ldafb,
			raw_data(ipiv),
			&equed_byte,
			raw_data(R),
			raw_data(C),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			rcond,
			rpvgrw,
			raw_data(berr),
			&n_err_bnds,
			raw_data(err_bnds_norm),
			raw_data(err_bnds_comp),
			&nparams,
			raw_data(params),
			raw_data(work),
			raw_data(iwork),
			&info,
			len(fact_c),
			len(trans_c),
			1,
		)
	}

	// Convert byte back to equilibration enum
	equed^ = equilibration_request_from_char(equed_byte)

	return info, info == 0
}

// Extended expert solve for banded system (complex64 version)
solve_banded_expert_extended_c64 :: proc(
	fact: FactorizationOption,
	trans: TransposeMode,
	AB: ^Matrix(complex64), // Banded matrix (input/output based on fact)
	AFB: ^Matrix(complex64), // Pre-allocated factored matrix (input/output based on fact)
	ipiv: []Blas_Int, // Pre-allocated pivot indices (input/output based on fact)
	equed: ^EquilibrationRequest, // Equilibration state (input/output)
	R: []f32, // Pre-allocated row scale factors (input/output)
	C: []f32, // Pre-allocated column scale factors (input/output)
	B: ^Matrix(complex64), // Right-hand side (input/output)
	X: ^Matrix(complex64), // Pre-allocated solution matrix (output)
	rcond: ^f32, // Output: reciprocal condition number
	rpvgrw: ^f32, // Output: reciprocal pivot growth factor
	berr: []f32, // Pre-allocated backward error bounds (output)
	n_err_bnds: Blas_Int, // Number of error bounds to compute
	err_bnds_norm: []f32, // Pre-allocated normwise error bounds (output)
	err_bnds_comp: []f32, // Pre-allocated componentwise error bounds (output)
	params: []f32, // Algorithm parameters (input/output)
	work: []complex64, // Pre-allocated workspace
	rwork: []f32, // Pre-allocated real workspace
) -> (
	info: Info,
	ok: bool,
) {
	assert(AB.format == .Banded, "Matrix must be in banded format")
	assert(AFB.format == .Banded, "Factored matrix must be in banded format")

	n := AB.cols
	kl := AB.storage.banded.kl
	ku := AB.storage.banded.ku
	nrhs := B.cols
	ldab := AB.ld
	ldafb := AFB.ld
	ldb := B.ld
	ldx := X.ld
	n_err_bnds := n_err_bnds

	// Validate inputs
	assert(len(ipiv) >= int(n), "Pivot array too small")
	assert(len(R) >= int(n), "Row scale array too small")
	assert(len(C) >= int(n), "Column scale array too small")
	assert(len(berr) >= int(nrhs), "Backward error array too small")
	assert(len(err_bnds_norm) >= int(nrhs * n_err_bnds), "Normwise error bounds array too small")
	assert(
		len(err_bnds_comp) >= int(nrhs * n_err_bnds),
		"Componentwise error bounds array too small",
	)
	assert(len(params) >= 3, "Parameters array too small")
	assert(len(work) >= 2 * int(n), "Work array too small")
	assert(len(rwork) >= 2 * int(n), "Real work array too small")

	fact_c := _factorization_to_char(fact)
	trans_c := transpose_mode_to_cstring(trans)

	// Convert equilibration enum to byte for LAPACK
	equed_byte := equilibration_request_to_char(equed^)

	// Set nparams
	nparams := Blas_Int(len(params))

	lapack.cgbsvxx_(
		fact_c,
		trans_c,
		&n,
		&kl,
		&ku,
		&nrhs,
		raw_data(AB.data),
		&ldab,
		raw_data(AFB.data),
		&ldafb,
		raw_data(ipiv),
		&equed_byte,
		raw_data(R),
		raw_data(C),
		raw_data(B.data),
		&ldb,
		raw_data(X.data),
		&ldx,
		rcond,
		rpvgrw,
		raw_data(berr),
		&n_err_bnds,
		raw_data(err_bnds_norm),
		raw_data(err_bnds_comp),
		&nparams,
		raw_data(params),
		raw_data(work),
		raw_data(rwork),
		&info,
		len(fact_c),
		len(trans_c),
		1,
	)

	// Convert byte back to equilibration enum
	equed^ = equilibration_request_from_char(equed_byte)

	return info, info == 0
}

// Extended expert solve for banded system (complex128 version)
solve_banded_expert_extended_c128 :: proc(
	fact: FactorizationOption,
	trans: TransposeMode,
	AB: ^Matrix(complex128), // Banded matrix (input/output based on fact)
	AFB: ^Matrix(complex128), // Pre-allocated factored matrix (input/output based on fact)
	ipiv: []Blas_Int, // Pre-allocated pivot indices (input/output based on fact)
	equed: ^EquilibrationRequest, // Equilibration state (input/output)
	R: []f64, // Pre-allocated row scale factors (input/output)
	C: []f64, // Pre-allocated column scale factors (input/output)
	B: ^Matrix(complex128), // Right-hand side (input/output)
	X: ^Matrix(complex128), // Pre-allocated solution matrix (output)
	rcond: ^f64, // Output: reciprocal condition number
	rpvgrw: ^f64, // Output: reciprocal pivot growth factor
	berr: []f64, // Pre-allocated backward error bounds (output)
	n_err_bnds: Blas_Int, // Number of error bounds to compute
	err_bnds_norm: []f64, // Pre-allocated normwise error bounds (output)
	err_bnds_comp: []f64, // Pre-allocated componentwise error bounds (output)
	params: []f64, // Algorithm parameters (input/output)
	work: []complex128, // Pre-allocated workspace
	rwork: []f64, // Pre-allocated real workspace
) -> (
	info: Info,
	ok: bool,
) {
	assert(AB.format == .Banded, "Matrix must be in banded format")
	assert(AFB.format == .Banded, "Factored matrix must be in banded format")

	n := AB.cols
	kl := AB.storage.banded.kl
	ku := AB.storage.banded.ku
	nrhs := B.cols
	ldab := AB.ld
	ldafb := AFB.ld
	ldb := B.ld
	ldx := X.ld
	n_err_bnds := n_err_bnds

	// Validate inputs
	assert(len(ipiv) >= int(n), "Pivot array too small")
	assert(len(R) >= int(n), "Row scale array too small")
	assert(len(C) >= int(n), "Column scale array too small")
	assert(len(berr) >= int(nrhs), "Backward error array too small")
	assert(len(err_bnds_norm) >= int(nrhs * n_err_bnds), "Normwise error bounds array too small")
	assert(
		len(err_bnds_comp) >= int(nrhs * n_err_bnds),
		"Componentwise error bounds array too small",
	)
	assert(len(params) >= 3, "Parameters array too small")
	assert(len(work) >= 2 * int(n), "Work array too small")
	assert(len(rwork) >= 2 * int(n), "Real work array too small")

	fact_c := _factorization_to_char(fact)
	trans_c := transpose_mode_to_cstring(trans)

	// Convert equilibration enum to byte for LAPACK
	equed_byte := equilibration_request_to_char(equed^)

	// Set nparams
	nparams := Blas_Int(len(params))

	lapack.zgbsvxx_(
		fact_c,
		trans_c,
		&n,
		&kl,
		&ku,
		&nrhs,
		raw_data(AB.data),
		&ldab,
		raw_data(AFB.data),
		&ldafb,
		raw_data(ipiv),
		&equed_byte,
		raw_data(R),
		raw_data(C),
		raw_data(B.data),
		&ldb,
		raw_data(X.data),
		&ldx,
		rcond,
		rpvgrw,
		raw_data(berr),
		&n_err_bnds,
		raw_data(err_bnds_norm),
		raw_data(err_bnds_comp),
		&nparams,
		raw_data(params),
		raw_data(work),
		raw_data(rwork),
		&info,
		len(fact_c),
		len(trans_c),
		1,
	)

	// Convert byte back to equilibration enum
	equed^ = equilibration_request_from_char(equed_byte)

	return info, info == 0
}
