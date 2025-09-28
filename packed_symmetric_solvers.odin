package openblas

import lapack "./f77"
import "core:c"
import "core:math"
import "core:mem"
import "core:slice"

// ============================================================================
// PACKED SYMMETRIC LINEAR SYSTEM SOLVERS
// ============================================================================
// Solves linear systems A*X = B where A is a symmetric matrix stored in packed format

// ============================================================================
// SIMPLE PACKED SYMMETRIC SOLVER
// ============================================================================

// Solve packed symmetric system for f32/f64
m_solve_packed_symmetric_f32_f64 :: proc(
	ap: []$T, // Packed matrix (modified to factorization)
	ipiv: []Blas_Int, // Pre-allocated pivot indices (size n)
	b: ^Matrix(T), // Right-hand side (modified to solution)
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 {
	n := b.rows
	nrhs := b.cols
	assert(len(ap) >= n * (n + 1) / 2, "Packed array too small")
	assert(len(ipiv) >= n, "Pivot array too small")

	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	ldb := Blas_Int(b.ld)

	when T == f32 {
		lapack.sspsv_(&uplo_c, &n_int, &nrhs_int, raw_data(ap), raw_data(ipiv), b.data, &ldb, &info, 1)
	} else when T == f64 {
		lapack.dspsv_(&uplo_c, &n_int, &nrhs_int, raw_data(ap), raw_data(ipiv), b.data, &ldb, &info, 1)
	}

	ok = info == 0
	return info, ok
}

// Solve packed symmetric system for complex64/complex128
m_solve_packed_symmetric_c64_c128 :: proc(
	ap: []$T, // Packed matrix (modified to factorization)
	ipiv: []Blas_Int, // Pre-allocated pivot indices (size n)
	b: ^Matrix(T), // Right-hand side (modified to solution)
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where T == complex64 || T == complex128 {
	n := b.rows
	nrhs := b.cols
	assert(len(ap) >= n * (n + 1) / 2, "Packed array too small")
	assert(len(ipiv) >= n, "Pivot array too small")

	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	ldb := Blas_Int(b.ld)

	when T == complex64 {
		lapack.cspsv_(&uplo_c, &n_int, &nrhs_int, cast(^lapack.complex)raw_data(ap), raw_data(ipiv), cast(^lapack.complex)b.data, &ldb, &info, 1)
	} else when T == complex128 {
		lapack.zspsv_(&uplo_c, &n_int, &nrhs_int, cast(^lapack.doublecomplex)raw_data(ap), raw_data(ipiv), cast(^lapack.doublecomplex)b.data, &ldb, &info, 1)
	}

	ok = info == 0
	return info, ok
}

// Procedure group for packed symmetric solver
m_solve_packed_symmetric :: proc {
	m_solve_packed_symmetric_f32_f64,
	m_solve_packed_symmetric_c64_c128,
}

// ============================================================================
// EXPERT PACKED SYMMETRIC SOLVER
// ============================================================================

// Query workspace for expert packed symmetric solver
query_workspace_solve_packed_symmetric_expert :: proc($T: typeid, n: int) -> (work_size: int, iwork_size: int, rwork_size: int) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	when T == f32 || T == f64 {
		// Real types: work = 3*n, iwork = n, no rwork
		work_size = 3 * n
		iwork_size = n
		rwork_size = 0
	} else {
		// Complex types: work = 2*n, no iwork, rwork = n
		work_size = 2 * n
		iwork_size = 0
		rwork_size = n
	}
	return
}

// Expert solve packed symmetric system for f32/f64
m_solve_packed_symmetric_expert_f32_f64 :: proc(
	ap: []$T, // Original packed matrix
	afp: []T, // Pre-allocated factored packed matrix (in/out)
	ipiv: []Blas_Int, // Pre-allocated pivot indices (in/out)
	b: ^Matrix(T), // Right-hand side
	x: ^Matrix(T), // Solution matrix (output)
	ferr: []T, // Pre-allocated forward error bounds (size nrhs)
	berr: []T, // Pre-allocated backward error bounds (size nrhs)
	work: []T, // Pre-allocated workspace (size 3*n)
	iwork: []Blas_Int, // Pre-allocated integer workspace (size n)
	fact := FactorizationOption.Equilibrate,
	uplo := MatrixRegion.Upper,
) -> (
	rcond: T,
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

	fact_c := factorization_to_char(fact)
	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	ldb := Blas_Int(b.ld)
	ldx := Blas_Int(x.ld)

	when T == f32 {
		lapack.sspsvx_(&fact_c, &uplo_c, &n_int, &nrhs_int, raw_data(ap), raw_data(afp), raw_data(ipiv), b.data, &ldb, x.data, &ldx, &rcond, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(iwork), &info, 1, 1)
	} else when T == f64 {
		lapack.dspsvx_(&fact_c, &uplo_c, &n_int, &nrhs_int, raw_data(ap), raw_data(afp), raw_data(ipiv), b.data, &ldb, x.data, &ldx, &rcond, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(iwork), &info, 1, 1)
	}

	ok = info == 0
	return rcond, info, ok
}

// Expert solve packed symmetric system for complex64/complex128
m_solve_packed_symmetric_expert_c64_c128 :: proc(
	ap: []$T, // Original packed matrix
	afp: []T, // Pre-allocated factored packed matrix (in/out)
	ipiv: []Blas_Int, // Pre-allocated pivot indices (in/out)
	b: ^Matrix(T), // Right-hand side
	x: ^Matrix(T), // Solution matrix (output)
	ferr: []$R, // Pre-allocated forward error bounds (size nrhs)
	berr: []R, // Pre-allocated backward error bounds (size nrhs)
	work: []T, // Pre-allocated workspace (size 2*n)
	rwork: []R, // Pre-allocated real workspace (size n)
	fact := FactorizationOption.Equilibrate,
	uplo := MatrixRegion.Upper,
) -> (
	rcond: R,
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

	fact_c := factorization_to_char(fact)
	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	ldb := Blas_Int(b.ld)
	ldx := Blas_Int(x.ld)

	when T == complex64 {
		lapack.cspsvx_(
			&fact_c,
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
			&rcond,
			raw_data(ferr),
			raw_data(berr),
			cast(^lapack.complex)raw_data(work),
			raw_data(rwork),
			&info,
			1,
			1,
		)
	} else when T == complex128 {
		lapack.zspsvx_(
			&fact_c,
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
			&rcond,
			raw_data(ferr),
			raw_data(berr),
			cast(^lapack.doublecomplex)raw_data(work),
			raw_data(rwork),
			&info,
			1,
			1,
		)
	}

	ok = info == 0
	return rcond, info, ok
}

// Procedure group for expert packed symmetric solver
m_solve_packed_symmetric_expert :: proc {
	m_solve_packed_symmetric_expert_f32_f64,
	m_solve_packed_symmetric_expert_c64_c128,
}
