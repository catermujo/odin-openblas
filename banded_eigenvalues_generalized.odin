package openblas

import lapack "./f77"
import "core:c"
import "core:math"
import "core:mem"
import "core:slice"

// ============================================================================
// GENERALIZED SYMMETRIC BANDED EIGENVALUE PROBLEMS
// ============================================================================
// Solves the generalized eigenvalue problem A*x = λ*B*x where A and B are
// symmetric banded matrices and B is positive definite

solve_reduce_banded_generalized :: proc {
	solve_reduce_banded_generalized_f32_c64,
	solve_reduce_banded_generalized_f64_c128,
}


// Reduce generalized hermitian/symmetric banded eigenvalue problem to standard form
reduce_banded_generalized_hermitian :: proc {
	reduce_banded_generalized_hermitian_real,
	reduce_banded_generalized_hermitian_c64,
	reduce_banded_generalized_hermitian_c128,
}

// Solve generalized hermitian/symmetric banded eigenvalue problem
eigen_banded_generalized_hermitian :: proc {
	eigen_banded_generalized_hermitian_real,
	eigen_banded_generalized_hermitian_c64,
	eigen_banded_generalized_hermitian_c128,
}

// Solve generalized hermitian/symmetric banded eigenvalue problem (divide-and-conquer)
eigen_banded_generalized_hermitian_dc :: proc {
	eigen_banded_generalized_hermitian_dc_real,
	eigen_banded_generalized_hermitian_dc_c64,
	eigen_banded_generalized_hermitian_dc_c128,
}

// Solve generalized hermitian/symmetric banded eigenvalue problem (expert with subset)
eigen_banded_generalized_hermitian_expert :: proc {
	eigen_banded_generalized_hermitian_expert_real,
	eigen_banded_generalized_hermitian_expert_c64,
	eigen_banded_generalized_hermitian_expert_c128,
}


// ============================================================================
// SYMMETRIC BANDED GENERALIZED EIGENVALUE REDUCTION
// ============================================================================
// Reduces a real symmetric-definite banded generalized eigenproblem
// A*x = λ*B*x to standard form C*y = λ*y

// Query result sizes for reduction
query_result_sizes_reduce_banded_generalized :: proc(
	vect: VectorOption,
	n: int,
) -> (
	x_rows: int,
	x_cols: int,
) {
	if vect == .FORM_VECTORS {
		return n, n // Transformation matrix is n×n
	}
	return 0, 0 // No transformation matrix needed
}

// Query workspace for reduction (works for all types)
query_workspace_reduce_banded_generalized :: proc(
	$T: typeid,
	vect: VectorOption,
	n: int, // <-- Matrix dimension (size of n×n matrix)
) -> (
	work: Blas_Int,
	rwork: Blas_Int,
) where is_float(T) || is_complex(T) {
	when is_float(T) {
		return Blas_Int(2 * n), 0
	} else when is_complex(T) {
		return Blas_Int(n), Blas_Int(n)
	}
}

// Solve reduction with pre-allocated workspace (f32/complex64)
solve_reduce_banded_generalized_f32_c64 :: proc(
	vect: VectorOption,
	uplo: MatrixRegion,
	n: Blas_Int,
	ka: Blas_Int,
	kb: Blas_Int,
	ab: ^Matrix($T),
	bb: ^Matrix(T),
	x: ^Matrix(T) = nil,
	work: []T,
	rwork: []f32 = nil,
) -> (
	info: Info,
	ok: bool,
) where T == f32 ||
	T == complex64 {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(ka >= 0 && kb >= 0, "Number of diagonals must be non-negative")
	assert(ab.rows >= ka + 1 && ab.cols >= n, "A band matrix storage too small")
	assert(bb.rows >= kb + 1 && bb.cols >= n, "B band matrix storage too small")
	assert(len(work) > 0, "Work array not allocated")

	when T == complex64 {
		assert(len(rwork) > 0, "Real work array required for complex types")
	}

	vect_cstring := vector_option_to_cstring(vect)
	uplo_cstring := matrix_region_to_cstring(uplo)

	n := n
	ka := ka
	kb := kb
	ldab := ab.ld
	ldbb := bb.ld
	ldx: Blas_Int = 1
	x_ptr: ^T = nil

	if vect == .FORM_VECTORS && x != nil {
		assert(x.rows >= n && x.cols >= n, "Transformation matrix too small")
		ldx = x.ld
		x_ptr = raw_data(x.data)
	}

	when T == f32 {
		lapack.ssbgst_(
			vect_cstring,
			uplo_cstring,
			&n,
			&ka,
			&kb,
			raw_data(ab.data),
			&ldab,
			raw_data(bb.data),
			&ldbb,
			x_ptr,
			&ldx,
			raw_data(work),
			&info,
			1,
			1,
		)
	} else when T == complex64 {
		lapack.chbgst_(
			vect_cstring,
			uplo_cstring,
			&n,
			&ka,
			&kb,
			raw_data(ab.data),
			&ldab,
			raw_data(bb.data),
			&ldbb,
			x_ptr,
			&ldx,
			raw_data(work),
			raw_data(rwork),
			&info,
			1,
			1,
		)
	}

	return info, info == 0
}

// Solve reduction with pre-allocated workspace (f64/complex128)
solve_reduce_banded_generalized_f64_c128 :: proc(
	vect: VectorOption,
	uplo: MatrixRegion,
	n: Blas_Int,
	ka: Blas_Int,
	kb: Blas_Int,
	ab: ^Matrix($T),
	bb: ^Matrix(T),
	x: ^Matrix(T) = nil,
	work: []T,
	rwork: []f64 = nil,
) -> (
	info: Info,
	ok: bool,
) where T == f64 ||
	T == complex128 {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(ka >= 0 && kb >= 0, "Number of diagonals must be non-negative")
	assert(ab.rows >= ka + 1 && ab.cols >= n, "A band matrix storage too small")
	assert(bb.rows >= kb + 1 && bb.cols >= n, "B band matrix storage too small")
	assert(len(work) > 0, "Work array not allocated")

	when T == complex128 {
		assert(len(rwork) > 0, "Real work array required for complex types")
	}

	vect_cstring := vector_option_to_cstring(vect)
	uplo_cstring := matrix_region_to_cstring(uplo)

	n := n
	ka := ka
	kb := kb
	ldab := ab.ld
	ldbb := bb.ld
	ldx: Blas_Int = 1
	x_ptr: ^T = nil

	if vect == .FORM_VECTORS && x != nil {
		assert(x.rows >= n && x.cols >= n, "Transformation matrix too small")
		ldx = x.ld
		x_ptr = raw_data(x.data)
	}

	when T == f64 {
		lapack.dsbgst_(
			vect_cstring,
			uplo_cstring,
			&n,
			&ka,
			&kb,
			raw_data(ab.data),
			&ldab,
			raw_data(bb.data),
			&ldbb,
			x_ptr,
			&ldx,
			raw_data(work),
			&info,
			1,
			1,
		)
	} else when T == complex128 {
		lapack.zhbgst_(
			vect_cstring,
			uplo_cstring,
			&n,
			&ka,
			&kb,
			raw_data(ab.data),
			&ldab,
			raw_data(bb.data),
			&ldbb,
			x_ptr,
			&ldx,
			raw_data(work),
			raw_data(rwork),
			&info,
			1,
			1,
		)
	}

	return info, info == 0
}

// ============================================================================
// SYMMETRIC BANDED GENERALIZED EIGENVALUE COMPUTATION
// ============================================================================

// Query result sizes for banded generalized eigenvalue
query_result_sizes_banded_generalized :: proc(
	jobz: EigenJobOption,
	n: int,
) -> (
	w_size: int,
	z_rows: int,
	z_cols: int, // Eigenvalues array// Eigenvector matrix rows// Eigenvector matrix cols
) {
	if jobz == .VALUES_VECTORS {
		return n, n, n // n eigenvalues, n×n eigenvector matrix
	}
	return n, 0, 0 // Only n eigenvalues
}

// Banded generalized eigenvalue result
BandedGeneralizedEigenResult :: struct($T: typeid) {
	eigenvalues:            []T, // Computed eigenvalues (sorted)
	eigenvectors:           Matrix(T), // Eigenvector matrix (if requested)
	b_is_positive_definite: bool, // True if B was successfully factored
	all_positive:           bool, // True if all eigenvalues > 0
	min_eigenvalue:         T, // Smallest eigenvalue
	max_eigenvalue:         T, // Largest eigenvalue
	condition_number:       T, // max|λ|/min|λ|
}

// Query workspace requirements for banded generalized solver
query_workspace_banded_generalized :: proc(
	$T: typeid,
	jobz: EigenJobOption,
	n: int, // <-- Matrix dimension (size of n×n matrix)
) -> (
	work: Blas_Int,
) where is_float(T) || is_complex(T) {
	// For standard algorithm, workspace is always 3*n
	return Blas_Int(3 * n)
}

// Solve banded generalized eigenvalue problem (generic for f32/f64)
solve_banded_generalized :: proc(
	jobz: EigenJobOption,
	uplo: MatrixRegion,
	n: int,
	ka: int,
	kb: int,
	ab: ^Matrix($T),
	bb: ^Matrix(T),
	w: []T,
	z: ^Matrix(T),
	work: []T,
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	assert(len(w) >= n, "Eigenvalue array too small")
	assert(len(work) >= 3 * n, "Work array too small")

	jobz_cstring := eigen_job_to_cstring(jobz)
	uplo_cstring := matrix_region_to_cstring(uplo)

	n_int: Blas_Int = n
	ka_int: Blas_Int = ka
	kb_int: Blas_Int = kb
	ldab := ab.ld
	ldbb := bb.ld
	ldz: Blas_Int = 1
	z_ptr: ^T = nil

	if jobz == .VALUES_VECTORS && z != nil {
		assert(z.rows >= n && z.cols >= n, "Eigenvector matrix too small")
		ldz = z.ld
		z_ptr = raw_data(z.data)
	}

	when T == f32 {
		lapack.ssbgv_(
			jobz_cstring,
			uplo_cstring,
			&n_int,
			&ka_int,
			&kb_int,
			raw_data(ab.data),
			&ldab,
			raw_data(bb.data),
			&ldbb,
			raw_data(w),
			z_ptr,
			&ldz,
			raw_data(work),
			&info,
			1,
			1,
		)
	} else when T == f64 {
		lapack.dsbgv_(
			jobz_cstring,
			uplo_cstring,
			&n_int,
			&ka_int,
			&kb_int,
			raw_data(ab.data),
			&ldab,
			raw_data(bb.data),
			&ldbb,
			raw_data(w),
			z_ptr,
			&ldz,
			raw_data(work),
			&info,
			1,
			1,
		)
	}

	return info, info == 0 || info > n_int
}

// ============================================================================
// DIVIDE-AND-CONQUER GENERALIZED EIGENVALUE
// ============================================================================

// Query result sizes for DC banded generalized eigenvalue
query_result_sizes_banded_generalized_dc :: proc(
	jobz: EigenJobOption,
	n: int,
) -> (
	w_size: int,
	z_rows: int,
	z_cols: int, // Eigenvalues array// Eigenvector matrix rows// Eigenvector matrix cols
) {
	if jobz == .VALUES_VECTORS {
		return n, n, n // n eigenvalues, n×n eigenvector matrix
	}
	return n, 0, 0 // Only n eigenvalues
}

// Query workspace requirements for banded generalized DC solver
query_workspace_banded_generalized_dc :: proc(
	$T: typeid,
	jobz: EigenJobOption,
	uplo: MatrixRegion,
	n: int,
	ka: int,
	kb: int,
	ab: ^Matrix(T),
	bb: ^Matrix(T),
) -> (
	work: Blas_Int,
	rwork: Blas_Int,
	iwork: Blas_Int,
) where is_float(T) ||
	is_complex(T) {
	jobz_cstring := eigen_job_to_cstring(jobz)
	uplo_cstring := matrix_region_to_cstring(uplo)

	n_int: Blas_Int = n
	ka_int: Blas_Int = ka
	kb_int: Blas_Int = kb
	ldab: Blas_Int = ab.ld
	ldbb: Blas_Int = bb.ld
	ldz: Blas_Int = 1
	lwork: Blas_Int = QUERY_WORKSPACE
	lrwork: Blas_Int = QUERY_WORKSPACE
	liwork: Blas_Int = QUERY_WORKSPACE
	info: Info

	when T == f32 {
		work_query: f32
		iwork_query: Blas_Int

		lapack.ssbgvd_(
			jobz_cstring,
			uplo_cstring,
			&n_int,
			&ka_int,
			&kb_int,
			raw_data(ab.data),
			&ldab,
			raw_data(bb.data),
			&ldbb,
			nil,
			nil,
			&ldz,
			&work_query,
			&lwork,
			&iwork_query,
			&liwork,
			&info,
			1,
			1,
		)

		return Blas_Int(work_query), 0, iwork_query
	} else when T == f64 {
		work_query: f64
		iwork_query: Blas_Int

		lapack.dsbgvd_(
			jobz_cstring,
			uplo_cstring,
			&n_int,
			&ka_int,
			&kb_int,
			raw_data(ab.data),
			&ldab,
			raw_data(bb.data),
			&ldbb,
			nil,
			nil,
			&ldz,
			&work_query,
			&lwork,
			&iwork_query,
			&liwork,
			&info,
			1,
			1,
		)

		return Blas_Int(work_query), 0, iwork_query
	} else when T == complex64 {
		work_query: complex64
		rwork_query: f32
		iwork_query: Blas_Int

		lapack.chbgvd_(
			jobz_cstring,
			uplo_cstring,
			&n_int,
			&ka_int,
			&kb_int,
			raw_data(ab.data),
			&ldab,
			raw_data(bb.data),
			&ldbb,
			nil,
			nil,
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

		return Blas_Int(real(work_query)), Blas_Int(rwork_query), iwork_query
	} else when T == complex128 {
		work_query: complex128
		rwork_query: f64
		iwork_query: Blas_Int

		lapack.zhbgvd_(
			jobz_cstring,
			uplo_cstring,
			&n_int,
			&ka_int,
			&kb_int,
			raw_data(ab.data),
			&ldab,
			raw_data(bb.data),
			&ldbb,
			nil,
			nil,
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

		return Blas_Int(real(work_query)), Blas_Int(rwork_query), iwork_query
	}
}

// Compute banded generalized eigenvalue problem using divide-and-conquer (f32/complex64)
// This version requires pre-allocated workspace
solve_banded_generalized_dc_f32_c64 :: proc(
	jobz: EigenJobOption,
	uplo: MatrixRegion,
	n: int,
	ka: int,
	kb: int,
	ab: ^Matrix($T),
	bb: ^Matrix(T),
	w: []f32, // Pre-allocated eigenvalues (size n)
	z: ^Matrix(T), // Pre-allocated eigenvectors (n x n if jobz == VALUES_VECTORS)
	work: []T, // Pre-allocated workspace
	rwork: []f32 = nil, // Pre-allocated real workspace (complex only)
	iwork: []Blas_Int, // Pre-allocated integer workspace
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == complex64 {
	assert(len(w) >= n, "Eigenvalue array too small")
	assert(len(work) > 0, "Work array not allocated")
	assert(len(iwork) > 0, "Integer work array not allocated")
	when T == complex64 {
		assert(len(rwork) > 0, "Real work array required for complex types")
	}

	jobz_cstring := eigen_job_to_cstring(jobz)
	uplo_cstring := matrix_region_to_cstring(uplo)

	n_int: Blas_Int = n
	ka_int: Blas_Int = ka
	kb_int: Blas_Int = kb
	ldab: Blas_Int = ab.ld
	ldbb: Blas_Int = bb.ld
	ldz: Blas_Int = 1
	z_ptr: ^T = nil

	if jobz == .VALUES_VECTORS && z != nil {
		assert(z.rows >= n && z.cols >= n, "Eigenvector matrix too small")
		ldz = z.ld
		z_ptr = raw_data(z.data)
	}

	lwork := Blas_Int(len(work))
	liwork := Blas_Int(len(iwork))

	when T == f32 {
		lapack.ssbgvd_(
			jobz_cstring,
			uplo_cstring,
			&n_int,
			&ka_int,
			&kb_int,
			raw_data(ab.data),
			&ldab,
			raw_data(bb.data),
			&ldbb,
			raw_data(w),
			z_ptr,
			&ldz,
			raw_data(work),
			&lwork,
			raw_data(iwork),
			&liwork,
			&info,
			1,
			1,
		)
	} else when T == complex64 {
		lrwork := Blas_Int(len(rwork))
		lapack.chbgvd_(
			jobz_cstring,
			uplo_cstring,
			&n_int,
			&ka_int,
			&kb_int,
			raw_data(ab.data),
			&ldab,
			raw_data(bb.data),
			&ldbb,
			raw_data(w),
			z_ptr,
			&ldz,
			raw_data(work),
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

	return info, info == 0
}

// Compute banded generalized eigenvalue problem using divide-and-conquer (f64/complex128)
// This version requires pre-allocated workspace
solve_banded_generalized_dc_f64_c128 :: proc(
	jobz: EigenJobOption,
	uplo: MatrixRegion,
	n: int,
	ka: int,
	kb: int,
	ab: ^Matrix($T),
	bb: ^Matrix(T),
	w: []f64, // Pre-allocated eigenvalues (size n)
	z: ^Matrix(T), // Pre-allocated eigenvectors (n x n if jobz == VALUES_VECTORS)
	work: []T, // Pre-allocated workspace
	rwork: []f64 = nil, // Pre-allocated real workspace (complex only)
	iwork: []Blas_Int, // Pre-allocated integer workspace
) -> (
	info: Info,
	ok: bool,
) where T == f64 || T == complex128 {
	assert(len(w) >= n, "Eigenvalue array too small")
	assert(len(work) > 0, "Work array not allocated")
	assert(len(iwork) > 0, "Integer work array not allocated")
	when T == complex128 {
		assert(len(rwork) > 0, "Real work array required for complex types")
	}

	jobz_cstring := eigen_job_to_cstring(jobz)
	uplo_cstring := matrix_region_to_cstring(uplo)

	n_int: Blas_Int = n
	ka_int: Blas_Int = ka
	kb_int: Blas_Int = kb
	ldab: Blas_Int = ab.ld
	ldbb: Blas_Int = bb.ld
	ldz: Blas_Int = 1
	z_ptr: ^T = nil

	if jobz == .VALUES_VECTORS && z != nil {
		assert(z.rows >= n && z.cols >= n, "Eigenvector matrix too small")
		ldz = z.ld
		z_ptr = raw_data(z.data)
	}

	lwork := Blas_Int(len(work))
	liwork := Blas_Int(len(iwork))

	when T == f64 {
		lapack.dsbgvd_(
			jobz_cstring,
			uplo_cstring,
			&n_int,
			&ka_int,
			&kb_int,
			raw_data(ab.data),
			&ldab,
			raw_data(bb.data),
			&ldbb,
			raw_data(w),
			z_ptr,
			&ldz,
			raw_data(work),
			&lwork,
			raw_data(iwork),
			&liwork,
			&info,
			1,
			1,
		)
	} else when T == complex128 {
		lrwork := Blas_Int(len(rwork))
		lapack.zhbgvd_(
			jobz_cstring,
			uplo_cstring,
			&n_int,
			&ka_int,
			&kb_int,
			raw_data(ab.data),
			&ldab,
			raw_data(bb.data),
			&ldbb,
			raw_data(w),
			z_ptr,
			&ldz,
			raw_data(work),
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

	return info, info == 0
}


// ============================================================================
// SELECTIVE GENERALIZED EIGENVALUE
// ============================================================================

// Query result sizes for selective banded generalized eigenvalue
query_result_sizes_banded_generalized_selective :: proc(
	jobz: EigenJobOption,
	range: EigenRangeOption,
	n: int,
	il: int,
	iu: int,
) -> (
	w_size: int,
	q_rows: int,
	q_cols: int,
	z_rows: int,// Eigenvalues array
	z_cols: int,// Transformation matrix rows
	ifail_size: int, // Transformation matrix cols// Eigenvector matrix rows// Eigenvector matrix cols (max)// Failure indices array
) {
	max_eigenvectors := n
	if range == .INDEX {
		max_eigenvectors = iu - il + 1
	}

	if jobz == .VALUES_VECTORS {
		return n, n, n, n, max_eigenvectors, n
	}
	return n, n, n, 0, 0, n
}

// Selective banded generalized eigenvalue result
SelectiveBandedGeneralizedResult :: struct($T: typeid) {
	eigenvalues:            []T,
	eigenvectors:           Matrix(T),
	num_found:              int,
	failed_indices:         []Blas_Int,
	b_is_positive_definite: bool,
}

// Query workspace requirements for selective banded generalized solver
query_workspace_banded_generalized_selective :: proc(
	$T: typeid,
	n: int,
) -> (
	work: Blas_Int,
	iwork: Blas_Int,
) where is_float(T) {
	// For selective algorithm, workspace is 7*n for work and 5*n for iwork
	return Blas_Int(7 * n), Blas_Int(5 * n)
}

// Solve selective banded generalized eigenvalue problem (generic for f32/f64)
solve_banded_generalized_selective :: proc(
	jobz: EigenJobOption,
	range: EigenRangeOption,
	uplo: MatrixRegion,
	n: int,
	ka: int,
	kb: int,
	ab: ^Matrix($T),
	bb: ^Matrix(T),
	q: ^Matrix(T), // Transformation matrix (if computing)
	vl: T, // Lower bound for eigenvalue range
	vu: T, // Upper bound for eigenvalue range
	il: int, // Lower index for eigenvalue range
	iu: int, // Upper index for eigenvalue range
	abstol: T, // Absolute tolerance
	w: []T, // Pre-allocated eigenvalues (size n)
	z: ^Matrix(T), // Pre-allocated eigenvectors
	work: []T, // Pre-allocated workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace
	ifail: []Blas_Int, // Pre-allocated failure indices (size n)
) -> (
	m: int,
	info: Info,
	ok: bool, // Number of eigenvalues found
) where is_float(T) {
	assert(len(w) >= n, "Eigenvalue array too small")
	assert(len(work) >= 7 * n, "Work array too small")
	assert(len(iwork) >= 5 * n, "Integer work array too small")
	assert(len(ifail) >= n, "Failure array too small")

	jobz_cstring := eigen_job_to_cstring(jobz)
	range_cstring := eigen_range_to_cstring(range)
	uplo_cstring := matrix_region_to_cstring(uplo)

	n_int: Blas_Int = n
	ka_int: Blas_Int = ka
	kb_int: Blas_Int = kb
	ldab := ab.ld
	ldbb := bb.ld
	ldq: Blas_Int = 1
	q_ptr: ^T = nil

	if q != nil {
		assert(q.rows >= n && q.cols >= n, "Q matrix too small")
		ldq = q.ld
		q_ptr = raw_data(q.data)
	}

	vl_val := vl
	vu_val := vu
	il_int := Blas_Int(il)
	iu_int := Blas_Int(iu)
	abstol_val := abstol

	ldz := Blas_Int(1)
	z_ptr: ^T = nil
	if jobz == .VALUES_VECTORS && z != nil {
		max_eigenvectors := n
		if range == .INDEX {
			max_eigenvectors = iu - il + 1
		}
		assert(z.rows >= n && z.cols >= max_eigenvectors, "Eigenvector matrix too small")
		ldz = z.ld
		z_ptr = raw_data(z.data)
	}

	m_int: Blas_Int

	when T == f32 {
		lapack.ssbgvx_(
			jobz_cstring,
			range_cstring,
			uplo_cstring,
			&n_int,
			&ka_int,
			&kb_int,
			raw_data(ab.data),
			&ldab,
			raw_data(bb.data),
			&ldbb,
			q_ptr,
			&ldq,
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
		lapack.dsbgvx_(
			jobz_cstring,
			range_cstring,
			uplo_cstring,
			&n_int,
			&ka_int,
			&kb_int,
			raw_data(ab.data),
			&ldab,
			raw_data(bb.data),
			&ldbb,
			q_ptr,
			&ldq,
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

	return int(m_int), Info(info), info == 0 || info > n_int
}


// ============================================================================
// SYMMETRIC BAND TO TRIDIAGONAL REDUCTION
// ============================================================================

// Query result sizes for band to tridiagonal reduction
query_result_sizes_band_to_tridiagonal :: proc(
	vect: VectorOption,
	n: int,
) -> (
	d_size: int,
	e_size: int,
	q_rows: int,
	q_cols: int, // Diagonal array// Off-diagonal array// Orthogonal matrix rows// Orthogonal matrix cols
) {
	if vect == .FORM_VECTORS {
		return n, max(0, n - 1), n, n // d[n], e[n-1], Q[n×n]
	}
	return n, max(0, n - 1), 0, 0 // Only d and e arrays
}

// Query workspace requirements for band to tridiagonal reduction
query_workspace_band_to_tridiagonal :: proc(
	$T: typeid,
	n: int,
) -> (
	work: Blas_Int,
) where is_float(T) {
	// Workspace is always n for sbtrd
	return Blas_Int(n)
}

// Reduce symmetric band matrix to tridiagonal form (generic for f32/f64)
solve_band_to_tridiagonal :: proc(
	vect: VectorOption,
	uplo: MatrixRegion,
	n: int,
	kd: int,
	ab: ^Matrix($T), // Band matrix (modified on output)
	d: []T, // Pre-allocated diagonal (size n)
	e: []T, // Pre-allocated off-diagonal (size n-1)
	q: ^Matrix(T), // Pre-allocated orthogonal matrix (if vect == FORM_VECTORS)
	work: []T, // Pre-allocated workspace
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	assert(len(d) >= n, "Diagonal array too small")
	assert(n == 0 || len(e) >= n - 1, "Off-diagonal array too small")
	assert(len(work) >= n, "Work array too small")

	vect_cstring := vector_option_to_cstring(vect)
	uplo_cstring := matrix_region_to_cstring(uplo)

	n_int: Blas_Int = n
	kd_int: Blas_Int = kd
	ldab: Blas_Int = ab.ld
	ldq: Blas_Int = 1
	q_ptr: ^T = nil

	if vect == .FORM_VECTORS && q != nil {
		assert(q.rows >= n && q.cols >= n, "Q matrix too small")
		ldq = q.ld
		q_ptr = raw_data(q.data)
	}

	when T == f32 {
		lapack.ssbtrd_(
			vect_cstring,
			uplo_cstring,
			&n_int,
			&kd_int,
			raw_data(ab.data),
			&ldab,
			raw_data(d),
			raw_data(e),
			q_ptr,
			&ldq,
			raw_data(work),
			&info,
			1,
			1,
		)
	} else when T == f64 {
		lapack.dsbtrd_(
			vect_cstring,
			uplo_cstring,
			&n_int,
			&kd_int,
			raw_data(ab.data),
			&ldab,
			raw_data(d),
			raw_data(e),
			q_ptr,
			&ldq,
			raw_data(work),
			&info,
			1,
			1,
		)
	}

	return info, info == 0
}

// ============================================================================
// SYMMETRIC RANK-K UPDATE IN RFP FORMAT
// ============================================================================

// Query result sizes for symmetric rank-k update in RFP format
query_result_sizes_sym_rank_k :: proc(
	n: int,
) -> (
	c_size: int, // RFP format array
) {
	return n * (n + 1) / 2 // Triangular storage in RFP format
}

// Symmetric rank-k update in RFP format (generic for f32/f64)
solve_sym_rank_k :: proc(
	transr: RFPTranspose,
	uplo: MatrixRegion,
	trans: TransposeMode,
	n: int,
	k: int,
	alpha: $T,
	a: ^Matrix(T),
	beta: T,
	c: []T, // RFP format array
) where is_float(T) {
	assert(n >= 0 && k >= 0, "Dimensions must be non-negative")
	assert(len(c) >= n * (n + 1) / 2, "RFP array too small")

	transr_cstring := rfp_transpose_to_cstring(transr)
	uplo_cstring := matrix_region_to_cstring(uplo)
	trans_cstring := transpose_to_cstring(trans)

	n_int: Blas_Int = n
	k_int: Blas_Int = k
	alpha_val := alpha
	beta_val := beta
	lda := a.ld

	when T == f32 {
		lapack.ssfrk_(
			transr_cstring,
			uplo_cstring,
			trans_cstring,
			&n_int,
			&k_int,
			&alpha_val,
			raw_data(a.data),
			&lda,
			&beta_val,
			raw_data(c),
			1,
			1,
			1,
		)
	} else when T == f64 {
		lapack.dsfrk_(
			transr_cstring,
			uplo_cstring,
			trans_cstring,
			&n_int,
			&k_int,
			&alpha_val,
			raw_data(a.data),
			&lda,
			&beta_val,
			raw_data(c),
			1,
			1,
			1,
		)
	}
	// dsfrk/ssfrk don't have info parameter
}

// ===================================================================================
// HERMITIAN BANDED GENERALIZED EIGENVALUE ROUTINES
// ===================================================================================

// Query result sizes for reduction of generalized hermitian banded problem
query_result_sizes_reduce_banded_generalized_hermitian :: proc(
	n: int,
	compute_z: bool,
) -> (
	x_rows: int,
	x_cols: int,
) {
	if compute_z {
		return n, n // Transformation matrix is n×n
	}
	return 0, 0 // No transformation matrix needed
}

// Query workspace for reduction of generalized hermitian/symmetric banded problem
query_workspace_reduce_banded_generalized_hermitian :: proc(
	$T: typeid,
	n: int,
) -> (
	work: Blas_Int,
	rwork: Blas_Int,
) where is_float(T) ||
	is_complex(T) {
	when is_float(T) {
		return Blas_Int(2 * n), 0 // real types need 2n work, no rwork
	} else when is_complex(T) {
		return Blas_Int(n), Blas_Int(n) // complex types need n work and n rwork
	}
}

// Reduce generalized symmetric banded eigenvalue problem (real)
reduce_banded_generalized_hermitian_real :: proc(
	vect: VectorOption,
	uplo: MatrixRegion,
	n: int,
	ka: int,
	kb: int,
	AB: ^Matrix($T), // Symmetric matrix A (input/output)
	BB: ^Matrix(T), // Positive definite matrix B (input/output)
	X: ^Matrix(T) = nil, // Transformation matrix (output if vect == .FORM_VECTORS)
	work: []T, // Pre-allocated workspace
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	assert(AB.format == .Banded, "Matrix AB must be banded format")
	assert(BB.format == .Banded, "Matrix BB must be banded format")
	assert(len(work) >= 2 * n, "Work array too small")

	n_int := Blas_Int(n)
	ka_int := Blas_Int(ka)
	kb_int := Blas_Int(kb)
	ldab := AB.storage.banded.ldab
	ldbb := BB.storage.banded.ldab

	vect_c := vector_option_to_cstring(vect)
	uplo_c := matrix_region_to_cstring(uplo)

	ldx := Blas_Int(1)
	x_ptr: ^T = nil
	if vect == .FORM_VECTORS && X != nil {
		assert(X.rows >= n && X.cols >= n, "Transformation matrix too small")
		ldx = X.ld
		x_ptr = raw_data(X.data)
	}

	when T == f32 {
		lapack.ssbgst_(
			vect_c,
			uplo_c,
			&n_int,
			&ka_int,
			&kb_int,
			raw_data(AB.data),
			&ldab,
			raw_data(BB.data),
			&ldbb,
			x_ptr,
			&ldx,
			raw_data(work),
			&info,
			len(vect_c),
			len(uplo_c),
		)
	} else when T == f64 {
		lapack.dsbgst_(
			vect_c,
			uplo_c,
			&n_int,
			&ka_int,
			&kb_int,
			raw_data(AB.data),
			&ldab,
			raw_data(BB.data),
			&ldbb,
			x_ptr,
			&ldx,
			raw_data(work),
			&info,
			len(vect_c),
			len(uplo_c),
		)
	}

	return info, info == 0
}

// Reduce generalized hermitian banded eigenvalue problem (complex64)
reduce_banded_generalized_hermitian_c64 :: proc(
	vect: VectorOption,
	uplo: MatrixRegion,
	n: int,
	ka: int,
	kb: int,
	AB: ^Matrix(complex64), // Hermitian matrix A (input/output)
	BB: ^Matrix(complex64), // Positive definite matrix B (input/output)
	X: ^Matrix(complex64) = nil, // Transformation matrix (output if vect == .FORM_VECTORS)
	work: []complex64, // Pre-allocated workspace
	rwork: []f32, // Pre-allocated real workspace
) -> (
	info: Info,
	ok: bool,
) {
	assert(AB.format == .Banded, "Matrix AB must be banded format")
	assert(BB.format == .Banded, "Matrix BB must be banded format")
	assert(len(work) >= n, "Work array too small")
	assert(len(rwork) >= n, "Real work array too small")

	n_int := Blas_Int(n)
	ka_int := Blas_Int(ka)
	kb_int := Blas_Int(kb)
	ldab := AB.storage.banded.ldab
	ldbb := BB.storage.banded.ldab

	vect_c := vector_option_to_cstring(vect)
	uplo_c := matrix_region_to_cstring(uplo)

	ldx := Blas_Int(1)
	x_ptr: ^complex64 = nil
	if vect == .FORM_VECTORS && X != nil {
		assert(int(X.rows) >= n && int(X.cols) >= n, "Transformation matrix too small")
		ldx = X.ld
		x_ptr = raw_data(X.data)
	}

	lapack.chbgst_(
		vect_c,
		uplo_c,
		&n_int,
		&ka_int,
		&kb_int,
		raw_data(AB.data),
		&ldab,
		raw_data(BB.data),
		&ldbb,
		x_ptr,
		&ldx,
		raw_data(work),
		raw_data(rwork),
		&info,
		len(vect_c),
		len(uplo_c),
	)

	return info, info == 0
}

// Reduce generalized hermitian banded eigenvalue problem (complex128)
reduce_banded_generalized_hermitian_c128 :: proc(
	vect: VectorOption,
	uplo: MatrixRegion,
	n: int,
	ka: int,
	kb: int,
	AB: ^Matrix(complex128), // Hermitian matrix A (input/output)
	BB: ^Matrix(complex128), // Positive definite matrix B (input/output)
	X: ^Matrix(complex128) = nil, // Transformation matrix (output if vect == .FORM_VECTORS)
	work: []complex128, // Pre-allocated workspace
	rwork: []f64, // Pre-allocated real workspace
) -> (
	info: Info,
	ok: bool,
) {
	assert(AB.format == .Banded, "Matrix AB must be banded format")
	assert(BB.format == .Banded, "Matrix BB must be banded format")
	assert(len(work) >= n, "Work array too small")
	assert(len(rwork) >= n, "Real work array too small")

	n_int := Blas_Int(n)
	ka_int := Blas_Int(ka)
	kb_int := Blas_Int(kb)
	ldab := AB.storage.banded.ldab
	ldbb := BB.storage.banded.ldab

	vect_c := vector_option_to_cstring(vect)
	uplo_c := matrix_region_to_cstring(uplo)

	ldx := Blas_Int(1)
	x_ptr: ^complex128 = nil
	if vect == .FORM_VECTORS && X != nil {
		assert(int(X.rows) >= n && int(X.cols) >= n, "Transformation matrix too small")
		ldx = X.ld
		x_ptr = raw_data(X.data)
	}

	lapack.zhbgst_(
		vect_c,
		uplo_c,
		&n_int,
		&ka_int,
		&kb_int,
		raw_data(AB.data),
		&ldab,
		raw_data(BB.data),
		&ldbb,
		x_ptr,
		&ldx,
		raw_data(work),
		raw_data(rwork),
		&info,
		len(vect_c),
		len(uplo_c),
	)

	return info, info == 0
}

// ===================================================================================
// GENERALIZED HERMITIAN BANDED EIGENVALUE SOLVER
// ===================================================================================

// Query result sizes for generalized hermitian/symmetric banded eigenvalue solver
query_result_sizes_eigen_banded_generalized_hermitian :: proc(
	n: int,
	compute_vectors: bool,
) -> (
	w_size: int,
	z_rows: int,
	z_cols: int, // Eigenvalues array size// Eigenvector matrix rows// Eigenvector matrix cols
) {
	w_size = n
	if compute_vectors {
		return n, n, n
	}
	return n, 0, 0
}

// Query workspace for generalized hermitian/symmetric banded eigenvalue solver
query_workspace_eigen_banded_generalized_hermitian :: proc(
	$T: typeid,
	n: int,
) -> (
	work: Blas_Int,
	rwork: Blas_Int,
) where is_float(T) ||
	is_complex(T) {
	when is_float(T) {
		return Blas_Int(3 * n), 0 // real types need 3n work
	} else when is_complex(T) {
		return Blas_Int(n), Blas_Int(3 * n) // complex types need n work and 3n rwork
	}
}

// Solve generalized symmetric banded eigenvalue problem (real)
eigen_banded_generalized_hermitian_real :: proc(
	jobz: VectorOption,
	uplo: MatrixRegion,
	n: int,
	ka: int,
	kb: int,
	AB: ^Matrix($T), // Symmetric matrix A (input/output)
	BB: ^Matrix(T), // Positive definite matrix B (input/output)
	w: []T, // Pre-allocated eigenvalues (size n)
	Z: ^Matrix(T) = nil, // Pre-allocated eigenvectors (n×n if jobz == .FORM_VECTORS)
	work: []T, // Pre-allocated workspace
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	assert(AB.format == .Banded, "Matrix AB must be banded format")
	assert(BB.format == .Banded, "Matrix BB must be banded format")
	assert(len(w) >= n, "Eigenvalue array too small")
	assert(len(work) >= 3 * n, "Work array too small")

	n_int := Blas_Int(n)
	ka_int := Blas_Int(ka)
	kb_int := Blas_Int(kb)
	ldab := AB.storage.banded.ldab
	ldbb := BB.storage.banded.ldab

	jobz_c := vector_option_to_cstring(jobz)
	uplo_c := matrix_region_to_cstring(uplo)

	ldz := Blas_Int(1)
	z_ptr: ^T = nil
	if jobz == .FORM_VECTORS && Z != nil {
		assert(Z.rows >= n && Z.cols >= n, "Eigenvector matrix too small")
		ldz = Z.ld
		z_ptr = raw_data(Z.data)
	}

	when T == f32 {
		lapack.ssbgv_(
			jobz_c,
			uplo_c,
			&n_int,
			&ka_int,
			&kb_int,
			raw_data(AB.data),
			&ldab,
			raw_data(BB.data),
			&ldbb,
			raw_data(w),
			z_ptr,
			&ldz,
			raw_data(work),
			&info,
			len(jobz_c),
			len(uplo_c),
		)
	} else when T == f64 {
		lapack.dsbgv_(
			jobz_c,
			uplo_c,
			&n_int,
			&ka_int,
			&kb_int,
			raw_data(AB.data),
			&ldab,
			raw_data(BB.data),
			&ldbb,
			raw_data(w),
			z_ptr,
			&ldz,
			raw_data(work),
			&info,
			len(jobz_c),
			len(uplo_c),
		)
	}

	return info, info == 0
}

// Solve generalized hermitian banded eigenvalue problem (complex64)
eigen_banded_generalized_hermitian_c64 :: proc(
	jobz: VectorOption,
	uplo: MatrixRegion,
	n: int,
	ka: int,
	kb: int,
	AB: ^Matrix(complex64), // Hermitian matrix A (input/output)
	BB: ^Matrix(complex64), // Positive definite matrix B (input/output)
	w: []f32, // Pre-allocated eigenvalues (size n)
	Z: ^Matrix(complex64) = nil, // Pre-allocated eigenvectors (n×n if jobz == .FORM_VECTORS)
	work: []complex64, // Pre-allocated workspace
	rwork: []f32, // Pre-allocated real workspace
) -> (
	info: Info,
	ok: bool,
) {
	assert(AB.format == .Banded, "Matrix AB must be banded format")
	assert(BB.format == .Banded, "Matrix BB must be banded format")
	assert(len(w) >= n, "Eigenvalue array too small")
	assert(len(work) >= n, "Work array too small")
	assert(len(rwork) >= 3 * n, "Real work array too small")

	n_int := Blas_Int(n)
	ka_int := Blas_Int(ka)
	kb_int := Blas_Int(kb)
	ldab := AB.storage.banded.ldab
	ldbb := BB.storage.banded.ldab

	jobz_c := vector_option_to_cstring(jobz)
	uplo_c := matrix_region_to_cstring(uplo)

	ldz := Blas_Int(1)
	z_ptr: ^complex64 = nil
	if jobz == .FORM_VECTORS && Z != nil {
		assert(int(Z.rows) >= n && int(Z.cols) >= n, "Eigenvector matrix too small")
		ldz = Z.ld
		z_ptr = raw_data(Z.data)
	}

	lapack.chbgv_(
		jobz_c,
		uplo_c,
		&n_int,
		&ka_int,
		&kb_int,
		raw_data(AB.data),
		&ldab,
		raw_data(BB.data),
		&ldbb,
		raw_data(w),
		z_ptr,
		&ldz,
		raw_data(work),
		raw_data(rwork),
		&info,
		len(jobz_c),
		len(uplo_c),
	)

	return info, info == 0
}

// Solve generalized hermitian banded eigenvalue problem (complex128)
eigen_banded_generalized_hermitian_c128 :: proc(
	jobz: VectorOption,
	uplo: MatrixRegion,
	n: int,
	ka: int,
	kb: int,
	AB: ^Matrix(complex128), // Hermitian matrix A (input/output)
	BB: ^Matrix(complex128), // Positive definite matrix B (input/output)
	w: []f64, // Pre-allocated eigenvalues (size n)
	Z: ^Matrix(complex128) = nil, // Pre-allocated eigenvectors (n×n if jobz == .FORM_VECTORS)
	work: []complex128, // Pre-allocated workspace
	rwork: []f64, // Pre-allocated real workspace
) -> (
	info: Info,
	ok: bool,
) {
	assert(AB.format == .Banded, "Matrix AB must be banded format")
	assert(BB.format == .Banded, "Matrix BB must be banded format")
	assert(len(w) >= n, "Eigenvalue array too small")
	assert(len(work) >= n, "Work array too small")
	assert(len(rwork) >= 3 * n, "Real work array too small")

	n_int := Blas_Int(n)
	ka_int := Blas_Int(ka)
	kb_int := Blas_Int(kb)
	ldab := AB.storage.banded.ldab
	ldbb := BB.storage.banded.ldab

	jobz_c := vector_option_to_cstring(jobz)
	uplo_c := matrix_region_to_cstring(uplo)

	ldz := Blas_Int(1)
	z_ptr: ^complex128 = nil
	if jobz == .FORM_VECTORS && Z != nil {
		assert(int(Z.rows) >= n && int(Z.cols) >= n, "Eigenvector matrix too small")
		ldz = Z.ld
		z_ptr = raw_data(Z.data)
	}

	lapack.zhbgv_(
		jobz_c,
		uplo_c,
		&n_int,
		&ka_int,
		&kb_int,
		raw_data(AB.data),
		&ldab,
		raw_data(BB.data),
		&ldbb,
		raw_data(w),
		z_ptr,
		&ldz,
		raw_data(work),
		raw_data(rwork),
		&info,
		len(jobz_c),
		len(uplo_c),
	)

	return info, info == 0
}

// ===================================================================================
// GENERALIZED HERMITIAN BANDED EIGENVALUE SOLVER (DIVIDE-AND-CONQUER)
// ===================================================================================

// Query workspace for generalized hermitian/symmetric banded eigenvalue solver (divide-and-conquer)
query_workspace_eigen_banded_generalized_hermitian_dc :: proc(
	$T: typeid,
	n: int,
	compute_vectors: bool,
) -> (
	work: Blas_Int,
	rwork: Blas_Int,
	iwork: Blas_Int,
) where is_float(T) ||
	is_complex(T) {
	when is_float(T) {
		// Real types - workspace will be queried at runtime
		return -1, 0, -1 // Use -1 to indicate runtime query needed
	} else when is_complex(T) {
		// Complex types - workspace will be queried at runtime
		if compute_vectors {
			return -1, Blas_Int(1 + 5 * n + 2 * n * n), Blas_Int(3 + 5 * n)
		} else {
			return -1, Blas_Int(n), Blas_Int(1)
		}
	}
}

// Solve generalized symmetric banded eigenvalue problem using divide-and-conquer (real)
eigen_banded_generalized_hermitian_dc_real :: proc(
	jobz: VectorOption,
	uplo: MatrixRegion,
	n: int,
	ka: int,
	kb: int,
	AB: ^Matrix($T), // Symmetric matrix A (input/output)
	BB: ^Matrix(T), // Positive definite matrix B (input/output)
	w: []T, // Pre-allocated eigenvalues (size n)
	Z: ^Matrix(T) = nil, // Pre-allocated eigenvectors (n×n if jobz == .FORM_VECTORS)
	work: []T, // Pre-allocated workspace (or size 1 for query)
	iwork: []Blas_Int, // Pre-allocated integer workspace (or size 1 for query)
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	assert(AB.format == .Banded, "Matrix AB must be banded format")
	assert(BB.format == .Banded, "Matrix BB must be banded format")
	assert(len(w) >= n, "Eigenvalue array too small")

	n_int := Blas_Int(n)
	ka_int := Blas_Int(ka)
	kb_int := Blas_Int(kb)
	ldab := AB.storage.banded.ldab
	ldbb := BB.storage.banded.ldab

	jobz_c := vector_option_to_cstring(jobz)
	uplo_c := matrix_region_to_cstring(uplo)

	ldz := Blas_Int(1)
	z_ptr: ^T = nil
	if jobz == .FORM_VECTORS && Z != nil {
		assert(int(Z.rows) >= n && int(Z.cols) >= n, "Eigenvector matrix too small")
		ldz = Z.ld
		z_ptr = raw_data(Z.data)
	}

	lwork := Blas_Int(len(work))
	liwork := Blas_Int(len(iwork))

	when T == f32 {
		lapack.ssbgvd_(
			jobz_c,
			uplo_c,
			&n_int,
			&ka_int,
			&kb_int,
			raw_data(AB.data),
			&ldab,
			raw_data(BB.data),
			&ldbb,
			raw_data(w),
			z_ptr,
			&ldz,
			raw_data(work),
			&lwork,
			raw_data(iwork),
			&liwork,
			&info,
			len(jobz_c),
			len(uplo_c),
		)
	} else when T == f64 {
		lapack.dsbgvd_(
			jobz_c,
			uplo_c,
			&n_int,
			&ka_int,
			&kb_int,
			raw_data(AB.data),
			&ldab,
			raw_data(BB.data),
			&ldbb,
			raw_data(w),
			z_ptr,
			&ldz,
			raw_data(work),
			&lwork,
			raw_data(iwork),
			&liwork,
			&info,
			len(jobz_c),
			len(uplo_c),
		)
	}

	return info, info == 0
}

// Solve generalized hermitian banded eigenvalue problem using divide-and-conquer (complex64)
eigen_banded_generalized_hermitian_dc_c64 :: proc(
	jobz: VectorOption,
	uplo: MatrixRegion,
	n: int,
	ka: int,
	kb: int,
	AB: ^Matrix(complex64), // Hermitian matrix A (input/output)
	BB: ^Matrix(complex64), // Positive definite matrix B (input/output)
	w: []f32, // Pre-allocated eigenvalues (size n)
	Z: ^Matrix(complex64) = nil, // Pre-allocated eigenvectors (n×n if jobz == .FORM_VECTORS)
	work: []complex64, // Pre-allocated workspace (or size 1 for query)
	rwork: []f32, // Pre-allocated real workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace
) -> (
	info: Info,
	ok: bool,
) {
	assert(AB.format == .Banded, "Matrix AB must be banded format")
	assert(BB.format == .Banded, "Matrix BB must be banded format")
	assert(len(w) >= n, "Eigenvalue array too small")

	n_int := Blas_Int(n)
	ka_int := Blas_Int(ka)
	kb_int := Blas_Int(kb)
	ldab := AB.storage.banded.ldab
	ldbb := BB.storage.banded.ldab

	jobz_c := vector_option_to_cstring(jobz)
	uplo_c := matrix_region_to_cstring(uplo)

	ldz := Blas_Int(1)
	z_ptr: ^complex64 = nil
	if jobz == .FORM_VECTORS && Z != nil {
		assert(int(Z.rows) >= n && int(Z.cols) >= n, "Eigenvector matrix too small")
		ldz = Z.ld
		z_ptr = raw_data(Z.data)
	}

	lwork := Blas_Int(len(work))
	lrwork := Blas_Int(len(rwork))
	liwork := Blas_Int(len(iwork))

	lapack.chbgvd_(
		jobz_c,
		uplo_c,
		&n_int,
		&ka_int,
		&kb_int,
		raw_data(AB.data),
		&ldab,
		raw_data(BB.data),
		&ldbb,
		raw_data(w),
		z_ptr,
		&ldz,
		raw_data(work),
		&lwork,
		raw_data(rwork),
		&lrwork,
		raw_data(iwork),
		&liwork,
		&info,
		len(jobz_c),
		len(uplo_c),
	)

	return info, info == 0
}

// Solve generalized hermitian banded eigenvalue problem using divide-and-conquer (complex128)
eigen_banded_generalized_hermitian_dc_c128 :: proc(
	jobz: VectorOption,
	uplo: MatrixRegion,
	n: int,
	ka: int,
	kb: int,
	AB: ^Matrix(complex128), // Hermitian matrix A (input/output)
	BB: ^Matrix(complex128), // Positive definite matrix B (input/output)
	w: []f64, // Pre-allocated eigenvalues (size n)
	Z: ^Matrix(complex128) = nil, // Pre-allocated eigenvectors (n×n if jobz == .FORM_VECTORS)
	work: []complex128, // Pre-allocated workspace (or size 1 for query)
	rwork: []f64, // Pre-allocated real workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace
) -> (
	info: Info,
	ok: bool,
) {
	assert(AB.format == .Banded, "Matrix AB must be banded format")
	assert(BB.format == .Banded, "Matrix BB must be banded format")
	assert(len(w) >= n, "Eigenvalue array too small")

	n_int := Blas_Int(n)
	ka_int := Blas_Int(ka)
	kb_int := Blas_Int(kb)
	ldab := AB.storage.banded.ldab
	ldbb := BB.storage.banded.ldab

	jobz_c := vector_option_to_cstring(jobz)
	uplo_c := matrix_region_to_cstring(uplo)

	ldz := Blas_Int(1)
	z_ptr: ^complex128 = nil
	if jobz == .FORM_VECTORS && Z != nil {
		assert(int(Z.rows) >= n && int(Z.cols) >= n, "Eigenvector matrix too small")
		ldz = Z.ld
		z_ptr = raw_data(Z.data)
	}

	lwork := Blas_Int(len(work))
	lrwork := Blas_Int(len(rwork))
	liwork := Blas_Int(len(iwork))

	lapack.zhbgvd_(
		jobz_c,
		uplo_c,
		&n_int,
		&ka_int,
		&kb_int,
		raw_data(AB.data),
		&ldab,
		raw_data(BB.data),
		&ldbb,
		raw_data(w),
		z_ptr,
		&ldz,
		raw_data(work),
		&lwork,
		raw_data(rwork),
		&lrwork,
		raw_data(iwork),
		&liwork,
		&info,
		len(jobz_c),
		len(uplo_c),
	)

	return info, info == 0
}

// ===================================================================================
// GENERALIZED HERMITIAN BANDED EIGENVALUE SOLVER (EXPERT WITH SUBSET)
// ===================================================================================

// Query result sizes for expert generalized hermitian/symmetric banded eigenvalue solver
query_result_sizes_eigen_banded_generalized_hermitian_expert :: proc(
	n: int,
	range_type: EigenRangeOption,
	il: int = 0, // Lower index for .Indexed
	iu: int = 0, // Upper index for .Indexed
) -> (
	max_m: int,
	w_size: int,
	z_rows: int,
	z_cols: int,
	q_rows: int,
	q_cols: int,
	ifail_size: int, // Maximum number of eigenvalues that could be found// Eigenvalues array size// Eigenvector matrix rows// Eigenvector matrix cols (max)// Transformation matrix rows// Transformation matrix cols// Failed indices array size
) {
	w_size = n
	q_rows = n
	q_cols = n
	ifail_size = n
	z_rows = n

	switch range_type {
	case .ALL:
		max_m = n
		z_cols = n
	case .INDEX:
		max_m = iu - il + 1 if iu > 0 else n - il + 1
		z_cols = max_m
	case .VALUE:
		max_m = n // Could be all eigenvalues in range
		z_cols = n
	}

	return max_m, w_size, z_rows, z_cols, q_rows, q_cols, ifail_size
}

// Query workspace for expert generalized hermitian/symmetric banded eigenvalue solver
query_workspace_eigen_banded_generalized_hermitian_expert :: proc(
	$T: typeid,
	n: int,
) -> (
	work: Blas_Int,
	rwork: Blas_Int,
	iwork: Blas_Int,
) where is_float(T) ||
	is_complex(T) {
	when is_float(T) {
		return Blas_Int(8 * n), 0, Blas_Int(5 * n)
	} else when is_complex(T) {
		return Blas_Int(n), Blas_Int(7 * n), Blas_Int(5 * n)
	}
}

// Solve generalized symmetric banded eigenvalue problem with subset selection (real)
eigen_banded_generalized_hermitian_expert_real :: proc(
	jobz: VectorOption,
	range_type: EigenRangeOption,
	uplo: MatrixRegion,
	n: int,
	ka: int,
	kb: int,
	AB: ^Matrix($T), // Symmetric matrix A (input/output)
	BB: ^Matrix(T), // Positive definite matrix B (input/output)
	Q: ^Matrix(T), // Pre-allocated transformation matrix (n×n)
	vl: T, // Lower bound for .Valued
	vu: T, // Upper bound for .Valued
	il: int = 1, // Lower index (1-based) for .Indexed
	iu: int = 0, // Upper index (1-based) for .Indexed (0 means n)
	abstol: T, // Absolute tolerance (0 for default)
	m: ^int, // Output: number of eigenvalues found
	w: []T, // Pre-allocated eigenvalues (size n)
	Z: ^Matrix(T) = nil, // Pre-allocated eigenvectors
	ifail: []Blas_Int, // Pre-allocated failed indices
	work: []T, // Pre-allocated workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	assert(AB.format == .Banded, "Matrix AB must be banded format")
	assert(BB.format == .Banded, "Matrix BB must be banded format")
	assert(Q.rows >= n && Q.cols >= n, "Q matrix too small")
	assert(len(w) >= n, "Eigenvalue array too small")
	assert(len(ifail) >= n, "Failed indices array too small")
	assert(len(work) >= 8 * n, "Work array too small")
	assert(len(iwork) >= 5 * n, "Integer work array too small")

	n_int := Blas_Int(n)
	ka_int := Blas_Int(ka)
	kb_int := Blas_Int(kb)
	ldab := AB.storage.banded.ldab
	ldbb := BB.storage.banded.ldab
	ldq := Blas_Int(Q.ld)

	jobz_c := vector_option_to_cstring(jobz)
	range_c := eigen_range_to_cstring(range_type)
	uplo_c := matrix_region_to_cstring(uplo)

	il_int := Blas_Int(il)
	iu_int := Blas_Int(iu if iu > 0 else n)
	vl_val := vl
	vu_val := vu
	abstol_val := abstol

	ldz := Blas_Int(1)
	z_ptr: ^T = nil
	if jobz == .FORM_VECTORS && Z != nil {
		ldz = Z.ld
		z_ptr = raw_data(Z.data)
	}

	m_int := Blas_Int(0)

	when T == f32 {
		lapack.ssbgvx_(
			jobz_c,
			range_c,
			uplo_c,
			&n_int,
			&ka_int,
			&kb_int,
			raw_data(AB.data),
			&ldab,
			raw_data(BB.data),
			&ldbb,
			raw_data(Q.data),
			&ldq,
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
			len(jobz_c),
			len(range_c),
			len(uplo_c),
		)
	} else when T == f64 {
		lapack.dsbgvx_(
			jobz_c,
			range_c,
			uplo_c,
			&n_int,
			&ka_int,
			&kb_int,
			raw_data(AB.data),
			&ldab,
			raw_data(BB.data),
			&ldbb,
			raw_data(Q.data),
			&ldq,
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
			len(jobz_c),
			len(range_c),
			len(uplo_c),
		)
	}

	m^ = int(m_int)
	return info, info == 0
}

// Solve generalized hermitian banded eigenvalue problem with subset selection (complex64)
eigen_banded_generalized_hermitian_expert_c64 :: proc(
	jobz: VectorOption,
	range_type: EigenRangeOption,
	uplo: MatrixRegion,
	n: int,
	ka: int,
	kb: int,
	AB: ^Matrix(complex64), // Hermitian matrix A (input/output)
	BB: ^Matrix(complex64), // Positive definite matrix B (input/output)
	Q: ^Matrix(complex64), // Pre-allocated transformation matrix (n×n)
	vl: f32 = 0, // Lower bound for .Valued
	vu: f32 = 0, // Upper bound for .Valued
	il: int = 1, // Lower index (1-based) for .Indexed
	iu: int = 0, // Upper index (1-based) for .Indexed (0 means n)
	abstol: f32 = 0, // Absolute tolerance (0 for default)
	m: ^int, // Output: number of eigenvalues found
	w: []f32, // Pre-allocated eigenvalues (size n)
	Z: ^Matrix(complex64) = nil, // Pre-allocated eigenvectors
	ifail: []Blas_Int, // Pre-allocated failed indices
	work: []complex64, // Pre-allocated workspace
	rwork: []f32, // Pre-allocated real workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace
) -> (
	info: Info,
	ok: bool,
) {
	assert(AB.format == .Banded, "Matrix AB must be banded format")
	assert(BB.format == .Banded, "Matrix BB must be banded format")
	assert(int(Q.rows) >= n && int(Q.cols) >= n, "Q matrix too small")
	assert(len(w) >= n, "Eigenvalue array too small")
	assert(len(ifail) >= n, "Failed indices array too small")
	assert(len(work) >= n, "Work array too small")
	assert(len(rwork) >= 7 * n, "Real work array too small")
	assert(len(iwork) >= 5 * n, "Integer work array too small")

	n_int := Blas_Int(n)
	ka_int := Blas_Int(ka)
	kb_int := Blas_Int(kb)
	ldab := AB.storage.banded.ldab
	ldbb := BB.storage.banded.ldab
	ldq := Blas_Int(Q.ld)

	jobz_c := vector_option_to_cstring(jobz)
	range_c := eigen_range_to_cstring(range_type)
	uplo_c := matrix_region_to_cstring(uplo)

	il_int := Blas_Int(il)
	iu_int := Blas_Int(iu if iu > 0 else n)
	vl_val := vl
	vu_val := vu
	abstol_val := abstol

	ldz := Blas_Int(1)
	z_ptr: ^complex64 = nil
	if jobz == .FORM_VECTORS && Z != nil {
		ldz = Blas_Int(Z.ld)
		z_ptr = raw_data(Z.data)
	}

	m_int := Blas_Int(0)

	lapack.chbgvx_(
		jobz_c,
		range_c,
		uplo_c,
		&n_int,
		&ka_int,
		&kb_int,
		raw_data(AB.data),
		&ldab,
		raw_data(BB.data),
		&ldbb,
		raw_data(Q.data),
		&ldq,
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

// Solve generalized hermitian banded eigenvalue problem with subset selection (complex128)
eigen_banded_generalized_hermitian_expert_c128 :: proc(
	jobz: VectorOption,
	range_type: EigenRangeOption,
	uplo: MatrixRegion,
	n: int,
	ka: int,
	kb: int,
	AB: ^Matrix(complex128), // Hermitian matrix A (input/output)
	BB: ^Matrix(complex128), // Positive definite matrix B (input/output)
	Q: ^Matrix(complex128), // Pre-allocated transformation matrix (n×n)
	vl: f64 = 0, // Lower bound for .Valued
	vu: f64 = 0, // Upper bound for .Valued
	il: int = 1, // Lower index (1-based) for .Indexed
	iu: int = 0, // Upper index (1-based) for .Indexed (0 means n)
	abstol: f64 = 0, // Absolute tolerance (0 for default)
	m: ^int, // Output: number of eigenvalues found
	w: []f64, // Pre-allocated eigenvalues (size n)
	Z: ^Matrix(complex128) = nil, // Pre-allocated eigenvectors
	ifail: []Blas_Int, // Pre-allocated failed indices
	work: []complex128, // Pre-allocated workspace
	rwork: []f64, // Pre-allocated real workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace
) -> (
	info: Info,
	ok: bool,
) {
	assert(AB.format == .Banded, "Matrix AB must be banded format")
	assert(BB.format == .Banded, "Matrix BB must be banded format")
	assert(int(Q.rows) >= n && int(Q.cols) >= n, "Q matrix too small")
	assert(len(w) >= n, "Eigenvalue array too small")
	assert(len(ifail) >= n, "Failed indices array too small")
	assert(len(work) >= n, "Work array too small")
	assert(len(rwork) >= 7 * n, "Real work array too small")
	assert(len(iwork) >= 5 * n, "Integer work array too small")

	n_int := Blas_Int(n)
	ka_int := Blas_Int(ka)
	kb_int := Blas_Int(kb)
	ldab := AB.storage.banded.ldab
	ldbb := BB.storage.banded.ldab
	ldq := Blas_Int(Q.ld)

	jobz_c := vector_option_to_cstring(jobz)
	range_c := eigen_range_to_cstring(range_type)
	uplo_c := matrix_region_to_cstring(uplo)

	il_int := Blas_Int(il)
	iu_int := Blas_Int(iu if iu > 0 else n)
	vl_val := vl
	vu_val := vu
	abstol_val := abstol

	ldz := Blas_Int(1)
	z_ptr: ^complex128 = nil
	if jobz == .FORM_VECTORS && Z != nil {
		ldz = Z.ld
		z_ptr = raw_data(Z.data)
	}

	m_int := Blas_Int(0)

	lapack.zhbgvx_(
		jobz_c,
		range_c,
		uplo_c,
		&n_int,
		&ka_int,
		&kb_int,
		raw_data(AB.data),
		&ldab,
		raw_data(BB.data),
		&ldbb,
		raw_data(Q.data),
		&ldq,
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
