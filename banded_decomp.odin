package openblas

import lapack "./f77"
import "base:builtin"
import "base:intrinsics"

// ===================================================================================
// MATRIX STRUCTURES FOR LAPACK
// ===================================================================================

// Reduce general banded matrix to bidiagonal form
// AB = Q * B * P^T where B is bidiagonal
banded_to_bidiag :: proc {
	banded_to_bidiag_real,
	banded_to_bidiag_c64,
	banded_to_bidiag_c128,
}


// ===================================================================================
// BANDED MATRIX OPERATIONS
// Efficient storage and operations for matrices with limited bandwidth
// ===================================================================================

// Query result sizes for banded to bidiagonal reduction
query_result_sizes_banded_to_bidiag :: proc(
	m: int,
	n: int,
	compute_q: bool,
	compute_pt: bool,
) -> (
	d_size: int,
	e_size: int,
	q_rows: int,
	q_cols: int,
	pt_rows: int,
	pt_cols: int, // Diagonal elements// Off-diagonal elements// Q matrix rows// Q matrix cols// PT matrix rows// PT matrix cols
) {
	min_mn := min(m, n)
	d_size = min_mn
	e_size = max(0, min_mn - 1)

	if compute_q {
		q_rows = m
		q_cols = min_mn
	}

	if compute_pt {
		pt_rows = min_mn
		pt_cols = n
	}

	return
}

// Query workspace for banded to bidiagonal reduction
query_workspace_banded_to_bidiag :: proc(
	$T: typeid,
	m: int,
	n: int,
) -> (
	work: Blas_Int,
	rwork: Blas_Int,
) where is_float(T) ||
	is_complex(T) {
	when is_float(T) {
		return Blas_Int(2 * max(m, n)), 0
	} else when is_complex(T) {
		return Blas_Int(max(m, n)), Blas_Int(max(m, n))
	}
}

// Reduce general banded matrix to bidiagonal form (real version)
banded_to_bidiag_real :: proc(
	AB: ^Matrix($T), // Banded matrix (modified on output)
	D: []T, // Pre-allocated diagonal elements
	E: []T, // Pre-allocated off-diagonal elements
	Q: ^Matrix(T) = nil, // Pre-allocated Q matrix (optional)
	PT: ^Matrix(T) = nil, // Pre-allocated PT matrix (optional)
	C: ^Matrix(T) = nil, // Apply transformation to C (optional)
	work: []T, // Pre-allocated workspace
	vect: cstring = "N", // "N", "Q", "P", or "B" for which matrices to compute
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

	// Validate sizes
	min_mn := min(m, n)
	assert(len(D) >= int(min_mn), "D array too small")
	assert(len(E) >= int(max(0, min_mn - 1)), "E array too small")
	assert(len(work) >= 2 * int(max(m, n)), "Work array too small")

	// Handle Q and PT
	ldq := Blas_Int(1)
	ldpt := Blas_Int(1)
	q_ptr: ^T = nil
	pt_ptr: ^T = nil

	if Q != nil && (vect == "Q" || vect == "B") {
		assert(Q.rows >= m && Q.cols >= min_mn, "Q matrix too small")
		ldq = Q.ld
		q_ptr = raw_data(Q.data)
	}

	if PT != nil && (vect == "P" || vect == "B") {
		assert(PT.rows >= min_mn && PT.cols >= n, "PT matrix too small")
		ldpt = PT.ld
		pt_ptr = raw_data(PT.data)
	}

	// Handle C matrix
	ncc := Blas_Int(0)
	ldc := Blas_Int(1)
	c_ptr: ^T = nil
	if C != nil {
		ncc = C.cols
		ldc = C.ld
		c_ptr = raw_data(C.data)
	}

	when T == f32 {
		lapack.sgbbrd_(
			vect,
			&m,
			&n,
			&ncc,
			&kl,
			&ku,
			raw_data(AB.data),
			&ldab,
			raw_data(D),
			raw_data(E),
			q_ptr,
			&ldq,
			pt_ptr,
			&ldpt,
			c_ptr,
			&ldc,
			raw_data(work),
			&info,
			1,
		)
	} else when T == f64 {
		lapack.dgbbrd_(
			vect,
			&m,
			&n,
			&ncc,
			&kl,
			&ku,
			raw_data(AB.data),
			&ldab,
			raw_data(D),
			raw_data(E),
			q_ptr,
			&ldq,
			pt_ptr,
			&ldpt,
			c_ptr,
			&ldc,
			raw_data(work),
			&info,
			1,
		)
	}

	return info, info == 0
}

// Reduce general banded matrix to bidiagonal form (complex64 version implementation)
banded_to_bidiag_c64 :: proc(
	AB: ^Matrix(complex64), // Banded matrix (modified on output)
	D: []f32, // Pre-allocated real diagonal elements
	E: []f32, // Pre-allocated real off-diagonal elements
	Q: ^Matrix(complex64) = nil, // Pre-allocated Q matrix (optional)
	PT: ^Matrix(complex64) = nil, // Pre-allocated PT matrix (optional)
	C: ^Matrix(complex64) = nil, // Apply transformation to C (optional)
	work: []complex64, // Pre-allocated workspace
	rwork: []f32, // Pre-allocated real workspace
	vect: cstring = "N", // "N", "Q", "P", or "B" for which matrices to compute
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

	// Validate sizes
	min_mn := min(m, n)
	assert(len(D) >= int(min_mn), "D array too small")
	assert(len(E) >= int(max(0, min_mn - 1)), "E array too small")
	assert(len(work) >= int(max(m, n)), "Work array too small")
	assert(len(rwork) >= int(max(m, n)), "Real work array too small")

	// Handle Q and PT
	ldq := Blas_Int(1)
	ldpt := Blas_Int(1)
	q_ptr: ^complex64 = nil
	pt_ptr: ^complex64 = nil

	if Q != nil && (vect == "Q" || vect == "B") {
		assert(Q.rows >= m && Q.cols >= min_mn, "Q matrix too small")
		ldq = Q.ld
		q_ptr = raw_data(Q.data)
	}

	if PT != nil && (vect == "P" || vect == "B") {
		assert(PT.rows >= min_mn && PT.cols >= n, "PT matrix too small")
		ldpt = PT.ld
		pt_ptr = raw_data(PT.data)
	}

	// Handle C matrix
	ncc := Blas_Int(0)
	ldc := Blas_Int(1)
	c_ptr: ^complex64 = nil
	if C != nil {
		ncc = C.cols
		ldc = C.ld
		c_ptr = raw_data(C.data)
	}

	lapack.cgbbrd_(
		vect,
		&m,
		&n,
		&ncc,
		&kl,
		&ku,
		raw_data(AB.data),
		&ldab,
		raw_data(D),
		raw_data(E),
		q_ptr,
		&ldq,
		pt_ptr,
		&ldpt,
		c_ptr,
		&ldc,
		raw_data(work),
		raw_data(rwork),
		&info,
		1,
	)

	return info, info == 0
}

// Reduce general banded matrix to bidiagonal form (complex128 version implementation)
banded_to_bidiag_c128 :: proc(
	AB: ^Matrix(complex128), // Banded matrix (modified on output)
	D: []f64, // Pre-allocated real diagonal elements
	E: []f64, // Pre-allocated real off-diagonal elements
	Q: ^Matrix(complex128) = nil, // Pre-allocated Q matrix (optional)
	PT: ^Matrix(complex128) = nil, // Pre-allocated PT matrix (optional)
	C: ^Matrix(complex128) = nil, // Apply transformation to C (optional)
	work: []complex128, // Pre-allocated workspace
	rwork: []f64, // Pre-allocated real workspace
	vect: cstring = "N", // "N", "Q", "P", or "B" for which matrices to compute
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

	// Validate sizes
	min_mn := min(m, n)
	assert(len(D) >= int(min_mn), "D array too small")
	assert(len(E) >= int(max(0, min_mn - 1)), "E array too small")
	assert(len(work) >= int(max(m, n)), "Work array too small")
	assert(len(rwork) >= int(max(m, n)), "Real work array too small")

	// Handle Q and PT
	ldq := Blas_Int(1)
	ldpt := Blas_Int(1)
	q_ptr: ^complex128 = nil
	pt_ptr: ^complex128 = nil

	if Q != nil && (vect == "Q" || vect == "B") {
		assert(Q.rows >= m && Q.cols >= min_mn, "Q matrix too small")
		ldq = Q.ld
		q_ptr = raw_data(Q.data)
	}

	if PT != nil && (vect == "P" || vect == "B") {
		assert(PT.rows >= min_mn && PT.cols >= n, "PT matrix too small")
		ldpt = PT.ld
		pt_ptr = raw_data(PT.data)
	}

	// Handle C matrix
	ncc := Blas_Int(0)
	ldc := Blas_Int(1)
	c_ptr: ^complex128 = nil
	if C != nil {
		ncc = C.cols
		ldc = C.ld
		c_ptr = raw_data(C.data)
	}

	lapack.zgbbrd_(
		vect,
		&m,
		&n,
		&ncc,
		&kl,
		&ku,
		raw_data(AB.data),
		&ldab,
		raw_data(D),
		raw_data(E),
		q_ptr,
		&ldq,
		pt_ptr,
		&ldpt,
		c_ptr,
		&ldc,
		raw_data(work),
		raw_data(rwork),
		&info,
		1,
	)

	return info, info == 0
}
