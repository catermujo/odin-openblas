package openblas

import lapack "./f77"
import "base:builtin"
import "base:intrinsics"

// ===================================================================================
// LINEAR SYSTEM SOLVERS
// Direct solvers for systems Ax = B
// ===================================================================================

solve_refine_solution :: proc {
	solve_refine_solution_f32_c64,
	solve_refine_solution_f64_c128,
}

solve_refine_solution_extended :: proc {
	solve_refine_extended_f32_c64,
	solve_refine_extended_f64_c128,
}

solve_mixed :: proc {
	solve_mixed_d32,
	solve_mixed_z64,
}

solve_expert :: proc {
	solve_expert_f32_c64,
	solve_expert_f64_c128,
}

m_solve_expert_extra :: proc {
	solve_expert_extra_real,
	solve_expert_extra_c64,
	solve_expert_extra_c128,
}
// m_lu_factor_recursive

// m_inverse

// m_inverse_direct

// m_solve_least_squares

// ===================================================================================
// ITERATIVE REFINEMENT
// ===================================================================================

// Query result sizes for iterative refinement
query_result_sizes_refine_solution :: proc(n: int, nrhs: int) -> (ferr_size: int, berr_size: int) {
	return nrhs, nrhs
}

// Query workspace for iterative refinement
query_workspace_refine_solution :: proc(
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

// Iterative refinement for linear systems (f32/complex64)
solve_refine_solution_f32_c64 :: proc(
	trans: TransposeMode,
	n: int,
	nrhs: int,
	A: ^Matrix($T), // Original matrix A
	AF: ^Matrix(T), // LU factorization from getrf
	ipiv: []Blas_Int, // Pivot indices from getrf
	B: ^Matrix(T), // Right-hand side
	X: ^Matrix(T), // Solution (input/output)
	ferr: []f32, // Pre-allocated forward error bounds
	berr: []f32, // Pre-allocated backward error bounds
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

	trans_c := transpose_mode_to_cstring(trans)
	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := A.ld
	ldaf := AF.ld
	ldb := B.ld
	ldx := X.ld

	when T == f32 {
		lapack.sgerfs_(
			trans_c,
			&n_int,
			&nrhs_int,
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
			raw_data(iwork),
			&info,
			len(trans_c),
		)
	} else when T == complex64 {
		lapack.cgerfs_(
			trans_c,
			&n_int,
			&nrhs_int,
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
			len(trans_c),
		)
	}

	return info, info == 0
}

// Iterative refinement for linear systems (f64/complex128)
solve_refine_solution_f64_c128 :: proc(
	trans: TransposeMode,
	n: int,
	nrhs: int,
	A: ^Matrix($T), // Original matrix A
	AF: ^Matrix(T), // LU factorization from getrf
	ipiv: []Blas_Int, // Pivot indices from getrf
	B: ^Matrix(T), // Right-hand side
	X: ^Matrix(T), // Solution (input/output)
	ferr: []f64, // Pre-allocated forward error bounds
	berr: []f64, // Pre-allocated backward error bounds
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

	trans_c := transpose_mode_to_cstring(trans)
	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := A.ld
	ldaf := AF.ld
	ldb := B.ld
	ldx := X.ld

	when T == f64 {
		lapack.dgerfs_(
			trans_c,
			&n_int,
			&nrhs_int,
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
			raw_data(iwork),
			&info,
			len(trans_c),
		)
	} else when T == complex128 {
		lapack.zgerfs_(
			trans_c,
			&n_int,
			&nrhs_int,
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
			len(trans_c),
		)
	}

	return info, info == 0
}


// Extended iterative refinement with comprehensive error bounds
// Provides componentwise and normwise error bounds with equilibration support

// Query result sizes for extended iterative refinement
query_result_sizes_refine_extended :: proc(
	n: int,
	nrhs: int,
	n_err_bnds: int,
) -> (
	R_size: int,
	C_size: int,
	berr_size: int,
	err_bnds_norm_size: int,
	err_bnds_comp_size: int,
) {
	return n, n, nrhs, nrhs * n_err_bnds, nrhs * n_err_bnds
}

// Query workspace for extended iterative refinement
query_workspace_refine_extended :: proc(
	$T: typeid,
	n: int,
) -> (
	work: Blas_Int,
	rwork: Blas_Int,
	iwork: Blas_Int,
) where is_float(T) ||
	is_complex(T) {
	when T == f32 || T == f64 {
		return Blas_Int(4 * n), 0, Blas_Int(n)
	} else when T == complex64 || T == complex128 {
		return Blas_Int(2 * n), Blas_Int(3 * n), 0
	}
}

solve_refine_extended_f32_c64 :: proc(
	A: ^Matrix($T),
	AF: ^Matrix(T),
	ipiv: []Blas_Int,
	R: []$U,
	C: []U,
	B: ^Matrix(T),
	X: ^Matrix(T),
	transpose: TransposeMode,
	equilibrated: EquilibrationRequest,
	n_err_bnds: Blas_Int,
	rcond: ^U,
	berr: []U,
	err_bnds_norm: []U,
	err_bnds_comp: []U,
	work: []T,
	rwork: []U = nil,
	iwork: []Blas_Int = nil,
) -> (
	info: Info,
	ok: bool,
) where (T == f32 && U == f32) ||
	(T == complex64 && U == f32) {
	// Validate inputs
	n := A.rows
	nrhs := B.cols
	assert(len(berr) >= nrhs, "Backward error array too small")
	assert(len(err_bnds_norm) >= nrhs * n_err_bnds, "Norm error array too small")
	assert(len(err_bnds_comp) >= nrhs * n_err_bnds, "Component error array too small")
	when T == f32 {
		assert(len(work) >= 4 * n, "Work array too small")
		assert(len(iwork) >= n, "Integer work array too small")
	} else when T == complex64 {
		assert(len(work) >= 2 * n, "Work array too small")
		assert(len(rwork) >= 3 * n, "Real work array too small")
	}

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := A.ld
	ldaf := AF.ld
	ldb := B.ld
	ldx := X.ld

	trans_c := transpose_mode_to_cstring(transpose)
	equed_c := equilibration_request_to_cstring(equilibrated)

	// Default parameters
	nparams: Blas_Int = 0
	params: U = 0
	n_err_bnds_copy := n_err_bnds

	when T == f32 {
		lapack.sgerfsx_(
			trans_c,
			equed_c,
			&n_int,
			&nrhs_int,
			raw_data(A.data),
			&lda,
			raw_data(AF.data),
			&ldaf,
			raw_data(ipiv),
			raw_data(R),
			raw_data(C),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			rcond,
			raw_data(berr),
			&n_err_bnds_copy,
			raw_data(err_bnds_norm),
			raw_data(err_bnds_comp),
			&nparams,
			&params,
			raw_data(work),
			raw_data(iwork),
			&info,
			len(trans_c),
			len(equed_c),
		)
	} else when T == complex64 {
		lapack.cgerfsx_(
			trans_c,
			equed_c,
			&n_int,
			&nrhs_int,
			raw_data(A.data),
			&lda,
			raw_data(AF.data),
			&ldaf,
			raw_data(ipiv),
			raw_data(R),
			raw_data(C),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			rcond,
			raw_data(berr),
			&n_err_bnds_copy,
			raw_data(err_bnds_norm),
			raw_data(err_bnds_comp),
			&nparams,
			&params,
			raw_data(work),
			raw_data(rwork),
			&info,
			len(trans_c),
			len(equed_c),
		)
	}

	return info, info == 0
}

solve_refine_extended_f64_c128 :: proc(
	A: ^Matrix($T),
	AF: ^Matrix(T),
	ipiv: []Blas_Int,
	R: []$U,
	C: []U,
	B: ^Matrix(T),
	X: ^Matrix(T),
	transpose: TransposeMode,
	equilibrated: EquilibrationRequest,
	n_err_bnds: Blas_Int,
	rcond: ^U,
	berr: []U,
	err_bnds_norm: []U,
	err_bnds_comp: []U,
	work: []T,
	rwork: []U = nil,
	iwork: []Blas_Int = nil,
) -> (
	info: Info,
	ok: bool,
) where (T == f64 && U == f64) ||
	(T == complex128 && U == f64) {
	// Validate inputs
	n := A.rows
	nrhs := B.cols
	assert(len(berr) >= nrhs, "Backward error array too small")
	assert(len(err_bnds_norm) >= nrhs * n_err_bnds, "Norm error array too small")
	assert(len(err_bnds_comp) >= nrhs * n_err_bnds, "Component error array too small")
	when T == f64 {
		assert(len(work) >= 4 * n, "Work array too small")
		assert(len(iwork) >= n, "Integer work array too small")
	} else when T == complex128 {
		assert(len(work) >= 2 * n, "Work array too small")
		assert(len(rwork) >= 3 * n, "Real work array too small")
	}

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := A.ld
	ldaf := AF.ld
	ldb := B.ld
	ldx := X.ld

	trans_c := transpose_mode_to_cstring(transpose)
	equed_c := equilibration_request_to_cstring(equilibrated)

	// Default parameters
	nparams: Blas_Int = 0
	params: U = 0
	n_err_bnds_copy := n_err_bnds

	when T == f64 {
		lapack.dgerfsx_(
			trans_c,
			equed_c,
			&n_int,
			&nrhs_int,
			raw_data(A.data),
			&lda,
			raw_data(AF.data),
			&ldaf,
			raw_data(ipiv),
			raw_data(R),
			raw_data(C),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			rcond,
			raw_data(berr),
			&n_err_bnds_copy,
			raw_data(err_bnds_norm),
			raw_data(err_bnds_comp),
			&nparams,
			&params,
			raw_data(work),
			raw_data(iwork),
			&info,
			len(trans_c),
			len(equed_c),
		)
	} else when T == complex128 {
		lapack.zgerfsx_(
			trans_c,
			equed_c,
			&n_int,
			&nrhs_int,
			raw_data(A.data),
			&lda,
			raw_data(AF.data),
			&ldaf,
			raw_data(ipiv),
			raw_data(R),
			raw_data(C),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			rcond,
			raw_data(berr),
			&n_err_bnds_copy,
			raw_data(err_bnds_norm),
			raw_data(err_bnds_comp),
			&nparams,
			&params,
			raw_data(work),
			raw_data(rwork),
			&info,
			len(trans_c),
			len(equed_c),
		)
	}

	return info, info == 0
}

// Proc group for extended refinement
solve_refine_extended :: proc {
	solve_refine_extended_f32_c64,
	solve_refine_extended_f64_c128,
}

// ===================================================================================
// MIXED-PRECISION ITERATIVE REFINEMENT SOLVERS
// ===================================================================================
// Solve linear system using mixed precision iterative refinement
// Uses lower precision for factorization, higher precision for refinement


// ===================================================================================
// MIXED-PRECISION ITERATIVE REFINEMENT SOLVERS
// ===================================================================================
// Solve linear system using mixed precision iterative refinement
// Uses lower precision for factorization, higher precision for refinement

// Query result sizes for mixed precision solver
query_result_sizes_mixed :: proc(n: int) -> (ipiv_size: int) {
	return n
}

// Query workspace for mixed precision solver
query_workspace_mixed :: proc(
	$T: typeid,
	n: int,
	nrhs: int,
) -> (
	work: Blas_Int,
	swork: Blas_Int,
	rwork: Blas_Int,
) where T == f64 ||
	T == complex128 {
	when T == f64 {
		return Blas_Int(n * nrhs), Blas_Int(n * (n + nrhs)), 0
	} else when T == complex128 {
		return Blas_Int(n * nrhs), Blas_Int(n * (n + nrhs)), Blas_Int(n)
	}
}

// Double precision solution with single precision factorization
solve_mixed_d32 :: proc(
	A: ^Matrix(f64),
	B: ^Matrix(f64),
	X: ^Matrix(f64),
	ipiv: []Blas_Int,
	iter: ^Blas_Int,
	work: []f64,
	swork: []f32,
) -> (
	info: Info,
	ok: bool,
) {
	// Validate inputs
	n := A.rows
	nrhs := B.cols
	assert(len(ipiv) >= int(n), "Pivot array too small")
	assert(len(work) >= int(n * nrhs), "Work array too small")
	assert(len(swork) >= int(n * (n + nrhs)), "Single precision work array too small")

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := A.ld
	ldb := B.ld
	ldx := X.ld

	// Solve system with iterative refinement
	lapack.dsgesv_(
		&n_int,
		&nrhs_int,
		raw_data(A.data),
		&lda,
		raw_data(ipiv),
		raw_data(B.data),
		&ldb,
		raw_data(X.data),
		&ldx,
		raw_data(work),
		raw_data(swork),
		iter,
		&info,
	)

	return info, info == 0
}

// Double complex precision solution with single complex precision factorization
solve_mixed_z64 :: proc(
	A: ^Matrix(complex128),
	B: ^Matrix(complex128),
	X: ^Matrix(complex128),
	ipiv: []Blas_Int,
	iter: ^Blas_Int,
	work: []complex128,
	swork: []complex64,
	rwork: []f64,
) -> (
	info: Info,
	ok: bool,
) {
	// Validate inputs
	n := A.rows
	nrhs := B.cols
	assert(len(ipiv) >= int(n), "Pivot array too small")
	assert(len(work) >= int(n * nrhs), "Work array too small")
	assert(len(swork) >= int(n * (n + nrhs)), "Single precision work array too small")
	assert(len(rwork) >= int(n), "Real work array too small")

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := A.ld
	ldb := B.ld
	ldx := X.ld

	// Solve system with iterative refinement
	lapack.zcgesv_(
		&n_int,
		&nrhs_int,
		raw_data(A.data),
		&lda,
		raw_data(ipiv),
		raw_data(B.data),
		&ldb,
		raw_data(X.data),
		&ldx,
		raw_data(work),
		raw_data(swork),
		raw_data(rwork),
		iter,
		&info,
	)

	return info, info == 0
}

// ===================================================================================
// EXPERT LINEAR SOLVERS WITH EQUILIBRATION
// ===================================================================================
// Solve linear system with expert options and error bounds
// Includes equilibration, condition estimation, and iterative refinement

// Query result sizes for expert solver
query_result_sizes_expert :: proc(
	n: int,
	nrhs: int,
) -> (
	ipiv_size: int,
	R_size: int,
	C_size: int,
	ferr_size: int,
	berr_size: int,
) {
	return n, n, n, nrhs, nrhs
}

// Query workspace for expert solver
query_workspace_expert :: proc(
	$T: typeid,
	n: int,
) -> (
	work: Blas_Int,
	rwork: Blas_Int,
	iwork: Blas_Int,
) where is_float(T) ||
	is_complex(T) {
	when T == f32 || T == f64 {
		return Blas_Int(4 * n), 0, Blas_Int(n)
	} else when T == complex64 || T == complex128 {
		return Blas_Int(2 * n), Blas_Int(2 * n), 0
	}
}

solve_expert_f32_c64 :: proc(
	fact: FactorizationOption,
	transpose: TransposeMode,
	A: ^Matrix($T),
	B: ^Matrix(T),
	AF: ^Matrix(T),
	ipiv: []Blas_Int,
	equed: ^EquilibrationRequest,
	R: []$U,
	C: []U,
	X: ^Matrix(T),
	rcond: ^U,
	ferr: []U,
	berr: []U,
	work: []T,
	rwork: []U = nil,
	iwork: []Blas_Int = nil,
) -> (
	info: Info,
	ok: bool,
) where (T == f32 && U == f32) ||
	(T == complex64 && U == f32) {
	// Validate inputs
	n := A.rows
	nrhs := B.cols
	assert(len(ipiv) >= int(n), "Pivot array too small")
	assert(len(R) >= int(n), "Row scale array too small")
	assert(len(C) >= int(n), "Column scale array too small")
	assert(len(ferr) >= int(nrhs), "Forward error array too small")
	assert(len(berr) >= int(nrhs), "Backward error array too small")
	when T == f32 {
		assert(len(work) >= 4 * int(n), "Work array too small")
		assert(len(iwork) >= int(n), "Integer work array too small")
	} else when T == complex64 {
		assert(len(work) >= 2 * int(n), "Work array too small")
		assert(len(rwork) >= 2 * int(n), "Real work array too small")
	}

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := A.ld
	ldb := B.ld
	ldaf := AF.ld
	ldx := X.ld

	fact_c := _factorization_to_char(fact)
	trans_c := transpose_mode_to_cstring(transpose)

	equed_char: byte = equilibration_request_to_char(equed^)

	when T == f32 {
		lapack.sgesvx_(
			fact_c,
			trans_c,
			&n_int,
			&nrhs_int,
			raw_data(A.data),
			&lda,
			raw_data(AF.data),
			&ldaf,
			raw_data(ipiv),
			&equed_char,
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
	} else when T == complex64 {
		lapack.cgesvx_(
			fact_c,
			trans_c,
			&n_int,
			&nrhs_int,
			raw_data(A.data),
			&lda,
			raw_data(AF.data),
			&ldaf,
			raw_data(ipiv),
			&equed_char,
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
	}

	equed^ = equilibration_request_from_char(equed_char)

	return info, info == 0
}

solve_expert_f64_c128 :: proc(
	fact: FactorizationOption,
	transpose: TransposeMode,
	A: ^Matrix($T),
	B: ^Matrix(T),
	AF: ^Matrix(T),
	ipiv: []Blas_Int,
	equed: ^EquilibrationRequest,
	R: []$U,
	C: []U,
	X: ^Matrix(T),
	rcond: ^U,
	ferr: []U,
	berr: []U,
	work: []T,
	rwork: []U = nil,
	iwork: []Blas_Int = nil,
) -> (
	info: Info,
	ok: bool,
) where (T == f64 && U == f64) ||
	(T == complex128 && U == f64) {
	// Validate inputs
	n := A.rows
	nrhs := B.cols
	assert(len(ipiv) >= int(n), "Pivot array too small")
	assert(len(R) >= int(n), "Row scale array too small")
	assert(len(C) >= int(n), "Column scale array too small")
	assert(len(ferr) >= int(nrhs), "Forward error array too small")
	assert(len(berr) >= int(nrhs), "Backward error array too small")
	when T == f64 {
		assert(len(work) >= 4 * int(n), "Work array too small")
		assert(len(iwork) >= int(n), "Integer work array too small")
	} else when T == complex128 {
		assert(len(work) >= 2 * int(n), "Work array too small")
		assert(len(rwork) >= 2 * int(n), "Real work array too small")
	}

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := A.ld
	ldb := B.ld
	ldaf := AF.ld
	ldx := X.ld

	fact_c := _factorization_to_char(fact)
	trans_c := transpose_mode_to_cstring(transpose)

	equed_char: byte = equilibration_request_to_char(equed^)

	when T == f64 {
		lapack.dgesvx_(
			fact_c,
			trans_c,
			&n_int,
			&nrhs_int,
			raw_data(A.data),
			&lda,
			raw_data(AF.data),
			&ldaf,
			raw_data(ipiv),
			&equed_char,
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
	} else when T == complex128 {
		lapack.zgesvx_(
			fact_c,
			trans_c,
			&n_int,
			&nrhs_int,
			raw_data(A.data),
			&lda,
			raw_data(AF.data),
			&ldaf,
			raw_data(ipiv),
			&equed_char,
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
	}

	equed^ = equilibration_request_from_char(equed_char)

	return info, info == 0
}

// ===================================================================================
// EXTRA-EXPERT LINEAR SOLVER WITH ADVANCED ERROR BOUNDS
// ===================================================================================
// Solve linear system with extra-expert options and componentwise error bounds
// Provides the most comprehensive error analysis available

// Query result sizes for expert_extra solver
query_result_sizes_expert_extra :: proc(
	n: int,
	nrhs: int,
	n_err_bnds: int,
) -> (
	ipiv_size: int,
	R_size: int,
	C_size: int,
	berr_size: int,
	err_bnds_norm_rows: int,
	err_bnds_norm_cols: int,
	err_bnds_comp_rows: int,
	err_bnds_comp_cols: int,
) {
	return n, n, n, nrhs, nrhs, n_err_bnds, nrhs, n_err_bnds
}

// Query workspace for expert_extra solver
query_workspace_expert_extra :: proc(
	$T: typeid,
	n: int,
) -> (
	work: Blas_Int,
	rwork: Blas_Int,
	iwork: Blas_Int,
) where is_float(T) ||
	is_complex(T) {
	when T == f32 || T == f64 {
		return Blas_Int(4 * n), 0, Blas_Int(n)
	} else when T == complex64 || T == complex128 {
		return Blas_Int(2 * n), Blas_Int(3 * n), 0
	}
}

solve_expert_extra_real :: proc(
	fact: FactorizationOption,
	transpose: TransposeMode,
	A: ^Matrix($T),
	B: ^Matrix(T),
	AF: ^Matrix(T),
	ipiv: []Blas_Int,
	equed: ^EquilibrationRequest,
	R: []T,
	C: []T,
	X: ^Matrix(T),
	rcond: ^T,
	rpvgrw: ^T,
	berr: []T,
	err_bnds_norm: ^Matrix(T),
	err_bnds_comp: ^Matrix(T),
	n_err_bnds: Blas_Int,
	work: []T,
	iwork: []Blas_Int,
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	// Validate inputs
	n := A.rows
	nrhs := B.cols
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(len(R) >= n, "Row scale array too small")
	assert(len(C) >= n, "Column scale array too small")
	assert(len(berr) >= nrhs, "Backward error array too small")
	assert(
		err_bnds_norm.rows >= nrhs && err_bnds_norm.cols >= n_err_bnds,
		"Norm error bounds matrix too small",
	)
	assert(
		err_bnds_comp.rows >= nrhs && err_bnds_comp.cols >= n_err_bnds,
		"Component error bounds matrix too small",
	)
	assert(len(work) >= 4 * n, "Work array too small")
	assert(len(iwork) >= n, "Integer work array too small")

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := A.ld
	ldb := B.ld
	ldaf := AF.ld
	ldx := X.ld

	fact_c := _factorization_to_char(fact)
	trans_c := transpose_mode_to_cstring(transpose)

	// Convert EquilibrationRequest to byte for LAPACK
	equed_char: byte = equilibration_request_to_char(equed^)

	// Parameters for algorithm tuning
	nparams: Blas_Int = 0
	params: T = 0
	n_err_bnds_copy := n_err_bnds

	when T == f32 {
		lapack.sgesvxx_(
			fact_c,
			trans_c,
			&n_int,
			&nrhs_int,
			raw_data(A.data),
			&lda,
			raw_data(AF.data),
			&ldaf,
			raw_data(ipiv),
			&equed_char,
			raw_data(R),
			raw_data(C),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			rcond,
			rpvgrw,
			raw_data(berr),
			&n_err_bnds_copy,
			raw_data(err_bnds_norm.data),
			raw_data(err_bnds_comp.data),
			&nparams,
			&params,
			raw_data(work),
			raw_data(iwork),
			&info,
			len(fact_c),
			len(trans_c),
			1,
		)
	} else {
		lapack.dgesvxx_(
			fact_c,
			trans_c,
			&n_int,
			&nrhs_int,
			raw_data(A.data),
			&lda,
			raw_data(AF.data),
			&ldaf,
			raw_data(ipiv),
			&equed_char,
			raw_data(R),
			raw_data(C),
			raw_data(B.data),
			&ldb,
			raw_data(X.data),
			&ldx,
			rcond,
			rpvgrw,
			raw_data(berr),
			&n_err_bnds_copy,
			raw_data(err_bnds_norm.data),
			raw_data(err_bnds_comp.data),
			&nparams,
			&params,
			raw_data(work),
			raw_data(iwork),
			&info,
			len(fact_c),
			len(trans_c),
			1,
		)
	}

	// Convert equilibration flag back to enum
	equed^ = equilibration_request_from_char(equed_char)

	return info, info == 0
}

solve_expert_extra_c64 :: proc(
	fact: FactorizationOption,
	transpose: TransposeMode,
	A: ^Matrix(complex64),
	B: ^Matrix(complex64),
	AF: ^Matrix(complex64),
	ipiv: []Blas_Int,
	equed: ^EquilibrationRequest,
	R: []f32,
	C: []f32,
	X: ^Matrix(complex64),
	rcond: ^f32,
	rpvgrw: ^f32,
	berr: []f32,
	err_bnds_norm: ^Matrix(f32),
	err_bnds_comp: ^Matrix(f32),
	n_err_bnds: Blas_Int,
	work: []complex64,
	rwork: []f32,
) -> (
	info: Info,
	ok: bool,
) {
	// Validate inputs
	n := A.rows
	nrhs := B.cols
	assert(len(ipiv) >= int(n), "Pivot array too small")
	assert(len(R) >= int(n), "Row scale array too small")
	assert(len(C) >= int(n), "Column scale array too small")
	assert(len(berr) >= int(nrhs), "Backward error array too small")
	assert(
		err_bnds_norm.rows >= nrhs && err_bnds_norm.cols >= n_err_bnds,
		"Norm error bounds matrix too small",
	)
	assert(
		err_bnds_comp.rows >= nrhs && err_bnds_comp.cols >= n_err_bnds,
		"Component error bounds matrix too small",
	)
	assert(len(work) >= 2 * int(n), "Work array too small")
	assert(len(rwork) >= 3 * int(n), "Real work array too small")

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := A.ld
	ldb := B.ld
	ldaf := AF.ld
	ldx := X.ld

	fact_c := _factorization_to_char(fact)
	trans_c := transpose_mode_to_cstring(transpose)

	// Convert EquilibrationRequest to byte for LAPACK
	equed_char: byte = equilibration_request_to_char(equed^)

	// Parameters for algorithm tuning
	nparams: Blas_Int = 0
	params: f32 = 0
	n_err_bnds_copy := n_err_bnds

	lapack.cgesvxx_(
		fact_c,
		trans_c,
		&n_int,
		&nrhs_int,
		raw_data(A.data),
		&lda,
		raw_data(AF.data),
		&ldaf,
		raw_data(ipiv),
		&equed_char,
		raw_data(R),
		raw_data(C),
		raw_data(B.data),
		&ldb,
		raw_data(X.data),
		&ldx,
		rcond,
		rpvgrw,
		raw_data(berr),
		&n_err_bnds_copy,
		raw_data(err_bnds_norm.data),
		raw_data(err_bnds_comp.data),
		&nparams,
		&params,
		raw_data(work),
		raw_data(rwork),
		&info,
		len(fact_c),
		len(trans_c),
		1,
	)

	// Convert equilibration flag back to enum
	equed^ = equilibration_request_from_char(equed_char)

	return info, info == 0
}

solve_expert_extra_c128 :: proc(
	fact: FactorizationOption,
	transpose: TransposeMode,
	A: ^Matrix(complex128),
	B: ^Matrix(complex128),
	AF: ^Matrix(complex128),
	ipiv: []Blas_Int,
	equed: ^EquilibrationRequest,
	R: []f64,
	C: []f64,
	X: ^Matrix(complex128),
	rcond: ^f64,
	rpvgrw: ^f64,
	berr: []f64,
	err_bnds_norm: ^Matrix(f64),
	err_bnds_comp: ^Matrix(f64),
	n_err_bnds: Blas_Int,
	work: []complex128,
	rwork: []f64,
) -> (
	info: Info,
	ok: bool,
) {
	// Validate inputs
	n := A.rows
	nrhs := B.cols
	assert(len(ipiv) >= int(n), "Pivot array too small")
	assert(len(R) >= int(n), "Row scale array too small")
	assert(len(C) >= int(n), "Column scale array too small")
	assert(len(berr) >= int(nrhs), "Backward error array too small")
	assert(
		err_bnds_norm.rows >= nrhs && err_bnds_norm.cols >= n_err_bnds,
		"Norm error bounds matrix too small",
	)
	assert(
		err_bnds_comp.rows >= nrhs && err_bnds_comp.cols >= n_err_bnds,
		"Component error bounds matrix too small",
	)
	assert(len(work) >= 2 * int(n), "Work array too small")
	assert(len(rwork) >= 3 * int(n), "Real work array too small")

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := A.ld
	ldb := B.ld
	ldaf := AF.ld
	ldx := X.ld

	fact_c := _factorization_to_char(fact)
	trans_c := transpose_mode_to_cstring(transpose)

	// Convert EquilibrationRequest to byte for LAPACK
	equed_char: byte = equilibration_request_to_char(equed^)

	// Parameters for algorithm tuning
	nparams: Blas_Int = 0
	params: f64 = 0
	n_err_bnds := n_err_bnds

	lapack.zgesvxx_(
		fact_c,
		trans_c,
		&n_int,
		&nrhs_int,
		raw_data(A.data),
		&lda,
		raw_data(AF.data),
		&ldaf,
		raw_data(ipiv),
		&equed_char,
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
		raw_data(err_bnds_norm.data),
		raw_data(err_bnds_comp.data),
		&nparams,
		&params,
		raw_data(work),
		raw_data(rwork),
		&info,
		1,
		1,
		1,
	)

	// Convert back from byte to enum
	equed^ = equilibration_request_from_char(equed_char)

	return info, info == 0
}

// ===================================================================================
// LU FACTORIZATION
// ===================================================================================
// Compute LU factorization of a matrix using recursive algorithm
// More efficient for tall matrices

// Query result sizes for LU factorization
query_result_sizes_lu_factor :: proc(m: int, n: int) -> (ipiv_size: int) {
	return int(min(Blas_Int(m), Blas_Int(n)))
}

// Compute LU factorization using recursive algorithm
lu_factor_recursive :: proc(
	A: ^Matrix($T),
	ipiv: []Blas_Int, // Pre-allocated pivot array
) -> (
	info: Info,
	ok: bool,
) where is_float(T) || is_complex(T) {
	m := A.rows
	n := A.cols
	lda := A.ld

	// Validate inputs
	assert(len(ipiv) >= int(min(m, n)), "Pivot array too small")

	when T == f32 {
		lapack.sgetrf2_(&m, &n, raw_data(A.data), &lda, raw_data(ipiv), &info)
	} else when T == f64 {
		lapack.dgetrf2_(&m, &n, raw_data(A.data), &lda, raw_data(ipiv), &info)
	} else when T == complex64 {
		lapack.cgetrf2_(&m, &n, raw_data(A.data), &lda, raw_data(ipiv), &info)
	} else when T == complex128 {
		lapack.zgetrf2_(&m, &n, raw_data(A.data), &lda, raw_data(ipiv), &info)
	}

	return info, info == 0
}

// ===================================================================================
// MATRIX INVERSION
// ===================================================================================
// Compute matrix inverse using LU factorization
// A^(-1) is computed from the LU factorization

// Query result sizes for matrix inversion
query_result_sizes_inverse :: proc(n: int) -> (ipiv_size: int) {
	return n
}

query_workspace_inverse :: proc(
	$T: typeid,
	n: int,
) -> (
	lwork: Blas_Int,
) where is_float(T) ||
	is_complex(T) {
	// Query for optimal workspace
	n_int := Blas_Int(n)
	lwork_query: Blas_Int = QUERY_WORKSPACE
	work_query: T
	info: Blas_Int

	// Dummy values for query
	lda := n_int
	ipiv: [1]Blas_Int

	when T == f32 {
		lapack.sgetri_(&n_int, &work_query, &lda, &ipiv[0], &work_query, &lwork_query, &info)
	} else when T == f64 {
		lapack.dgetri_(&n_int, &work_query, &lda, &ipiv[0], &work_query, &lwork_query, &info)
	} else when T == complex64 {
		lapack.cgetri_(&n_int, &work_query, &lda, &ipiv[0], &work_query, &lwork_query, &info)
	} else when T == complex128 {
		lapack.zgetri_(&n_int, &work_query, &lda, &ipiv[0], &work_query, &lwork_query, &info)
	}

	return Blas_Int(real(work_query))
}

// Compute matrix inverse given LU factorization
inverse :: proc(
	A: ^Matrix($T),
	ipiv: []Blas_Int, // Pivot indices from LU factorization
	work: []T, // Pre-allocated workspace
) -> (
	info: Info,
	ok: bool,
) where is_float(T) || is_complex(T) {
	n := A.rows
	lda := A.ld
	lwork := Blas_Int(len(work))

	// Validate inputs
	assert(A.rows == A.cols, "Matrix must be square")
	assert(len(ipiv) >= int(n), "Pivot array too small")
	assert(len(work) >= int(n), "Work array too small")

	// Compute inverse
	when T == f32 {
		lapack.sgetri_(&n, raw_data(A.data), &lda, raw_data(ipiv), raw_data(work), &lwork, &info)
	} else when T == f64 {
		lapack.dgetri_(&n, raw_data(A.data), &lda, raw_data(ipiv), raw_data(work), &lwork, &info)
	} else when T == complex64 {
		lapack.cgetri_(&n, raw_data(A.data), &lda, raw_data(ipiv), raw_data(work), &lwork, &info)
	} else when T == complex128 {
		lapack.zgetri_(&n, raw_data(A.data), &lda, raw_data(ipiv), raw_data(work), &lwork, &info)
	}

	return info, info == 0
}


// ===================================================================================
// LEAST SQUARES SOLVERS
// ===================================================================================

// Solve overdetermined or underdetermined linear system using QR or LQ factorization
// For m >= n: finds least squares solution to minimize ||B - AX||
// For m < n: finds minimum norm solution

// Query workspace for least squares solver
query_workspace_least_squares :: proc(
	$T: typeid,
	m: int,
	n: int,
	nrhs: int,
	transpose: TransposeMode = .None,
) -> (
	lwork: Blas_Int,
) where is_float(T) ||
	is_complex(T) {
	m_int := Blas_Int(m)
	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lwork_query: Blas_Int = QUERY_WORKSPACE
	work_query: T
	info: Blas_Int

	// Dummy values for query
	lda := m_int
	ldb := max(m_int, n_int)
	trans_c := transpose_mode_to_cstring(transpose)

	when T == f32 {
		lapack.sgetsls_(
			trans_c,
			&m_int,
			&n_int,
			&nrhs_int,
			&work_query,
			&lda,
			&work_query,
			&ldb,
			&work_query,
			&lwork_query,
			&info,
			1,
		)
	} else when T == f64 {
		lapack.dgetsls_(
			trans_c,
			&m_int,
			&n_int,
			&nrhs_int,
			&work_query,
			&lda,
			&work_query,
			&ldb,
			&work_query,
			&lwork_query,
			&info,
			1,
		)
	} else when T == complex64 {
		lapack.cgetsls_(
			trans_c,
			&m_int,
			&n_int,
			&nrhs_int,
			&work_query,
			&lda,
			&work_query,
			&ldb,
			&work_query,
			&lwork_query,
			&info,
			1,
		)
	} else when T == complex128 {
		lapack.zgetsls_(
			trans_c,
			&m_int,
			&n_int,
			&nrhs_int,
			&work_query,
			&lda,
			&work_query,
			&ldb,
			&work_query,
			&lwork_query,
			&info,
			1,
		)
	}

	return Blas_Int(real(work_query))
}

// Solve least squares problem with pre-allocated workspace
solve_least_squares :: proc(
	A: ^Matrix($T),
	B: ^Matrix(T),
	work: []T,
	transpose: TransposeMode = .None,
) -> (
	info: Info,
	ok: bool,
) where is_float(T) ||
	is_complex(T) {
	m := A.rows
	n := A.cols
	nrhs := B.cols
	lda := A.ld
	ldb := B.ld
	lwork := Blas_Int(len(work))

	// Validate inputs
	if transpose == .None {
		assert(B.rows >= m, "B matrix has insufficient rows")
	} else {
		assert(B.rows >= n, "B matrix has insufficient rows")
	}
	assert(len(work) >= 1, "Work array too small")

	trans_c := transpose_mode_to_cstring(transpose)

	// Solve the system (B is overwritten with solution)
	when T == f32 {
		lapack.sgetsls_(
			trans_c,
			&m,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			raw_data(work),
			&lwork,
			&info,
			1,
		)
	} else when T == f64 {
		lapack.dgetsls_(
			trans_c,
			&m,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			raw_data(work),
			&lwork,
			&info,
			1,
		)
	} else when T == complex64 {
		lapack.cgetsls_(
			trans_c,
			&m,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			raw_data(work),
			&lwork,
			&info,
			1,
		)
	} else when T == complex128 {
		lapack.zgetsls_(
			trans_c,
			&m,
			&n,
			&nrhs,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			raw_data(work),
			&lwork,
			&info,
			1,
		)
	}

	return info, info == 0
}
