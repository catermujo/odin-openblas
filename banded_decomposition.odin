package openblas

import lapack "./f77"
import "base:builtin"

// ===================================================================================
// BANDED MATRIX DECOMPOSITION ROUTINES
//
// This file provides decomposition operations for banded matrices:
// - Bidiagonal reduction (GBBRD routines)
// - QR decomposition (GB-specific routines)
// - SVD preprocessing for banded matrices
//
// The routines in this file handle the conversion of general banded matrices
// to more structured forms suitable for eigenvalue computation or solving
// linear systems efficiently.
// ===================================================================================

// ===================================================================================
// BANDED TO BIDIAGONAL REDUCTION
// ===================================================================================

// Options for which orthogonal matrices to compute during bidiagonal reduction
BidiagVectorOption :: enum u8 {
	NONE    = 'N', // Compute neither Q nor P^T
	Q_ONLY  = 'Q', // Compute Q only
	PT_ONLY = 'P', // Compute P^T only
	BOTH    = 'B', // Compute both Q and P^T
}

// ===================================================================================
// WORKSPACE AND SIZE QUERIES
// ===================================================================================

// Query result sizes for banded to bidiagonal reduction
query_result_sizes_band_to_bidi :: proc(
	AB: ^BandedMatrix($T),
	vect: BidiagVectorOption,
	apply_c: bool = false,
) -> (
	d_size: int,
	e_size: int,
	q_rows: int,
	q_cols: int,
	pt_rows: int,
	pt_cols: int,
	c_required: bool,
) where is_float(T) || is_complex(T) {
	m := int(AB.rows)
	n := int(AB.cols)
	min_mn := min(m, n)

	d_size = min_mn
	e_size = max(0, min_mn - 1)

	// Q matrix dimensions
	if vect == .Q_ONLY || vect == .BOTH {
		q_rows = m
		q_cols = min_mn
	}

	// P^T matrix dimensions
	if vect == .PT_ONLY || vect == .BOTH {
		pt_rows = min_mn
		pt_cols = n
	}

	c_required = apply_c

	return
}

// Query workspace for banded to bidiagonal reduction
query_workspace_band_to_bidi :: proc(AB: ^BandedMatrix($T)) -> (work: int, rwork: int) where is_float(T) || is_complex(T) {
	m := int(AB.rows)
	n := int(AB.cols)

	when is_float(T) {
		work = 2 * max(m, n)
		rwork = 0
	} else when is_complex(T) {
		work = max(m, n)
		rwork = max(m, n)
	}
	return
}

// ===================================================================================
// UNIFIED IMPLEMENTATION
// ===================================================================================

// Reduce banded matrix to bidiagonal form
band_to_bidi :: proc {
	band_to_bidi_real,
	band_to_bidi_complex,
}

// Real banded to bidiagonal reduction (f32/f64)
band_to_bidi_real :: proc(
	vect: BidiagVectorOption,
	AB: ^BandedMatrix($T), // Banded matrix (input/output - destroyed)
	B: ^Bidiagonal(T), // Output bidiagonal matrix
	Q: ^Matrix(T), // Pre-allocated Q matrix (optional)
	PT: ^Matrix(T), // Pre-allocated P^T matrix (optional)
	C: ^Matrix(T), // Matrix to apply transformations to (optional)
	work: []T, // Pre-allocated workspace (size 2*max(m,n))
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	m := AB.rows
	n := AB.cols
	kl := AB.kl
	ku := AB.ku
	ldab := AB.ldab

	// Validate input dimensions
	min_mn := min(int(m), int(n))
	assert(int(B.n) == min_mn, "Bidiagonal dimension must be min(m,n)")
	assert(len(B.d) >= min_mn, "Bidiagonal diagonal too small")
	assert(len(B.e) >= max(0, min_mn - 1), "Bidiagonal off-diagonal too small")
	assert(len(work) >= 2 * max(int(m), int(n)), "Work array too small")

	// Validate and set up Q matrix
	ldq := Blas_Int(1)
	q_ptr: ^T = nil
	if (vect == .Q_ONLY || vect == .BOTH) && Q != nil {
		assert(int(Q.rows) >= int(m) && int(Q.cols) >= min_mn, "Q matrix too small")
		ldq = Q.ld
		q_ptr = raw_data(Q.data)
	}

	// Validate and set up P^T matrix
	ldpt := Blas_Int(1)
	pt_ptr: ^T = nil
	if (vect == .PT_ONLY || vect == .BOTH) && PT != nil {
		assert(int(PT.rows) >= min_mn && int(PT.cols) >= int(n), "P^T matrix too small")
		ldpt = PT.ld
		pt_ptr = raw_data(PT.data)
	}

	// Handle optional C matrix for applying transformations
	ncc := Blas_Int(0)
	ldc := Blas_Int(1)
	c_ptr: ^T = nil
	if C != nil {
		ncc = C.cols
		ldc = C.ld
		c_ptr = raw_data(C.data)
	}

	vect_c := cast(u8)vect

	when T == f32 {
		lapack.sgbbrd_(&vect_c, &m, &n, &ncc, &kl, &ku, raw_data(AB.data), &ldab, raw_data(B.d), raw_data(B.e), q_ptr, &ldq, pt_ptr, &ldpt, c_ptr, &ldc, raw_data(work), &info)
	} else when T == f64 {
		lapack.dgbbrd_(&vect_c, &m, &n, &ncc, &kl, &ku, raw_data(AB.data), &ldab, raw_data(B.d), raw_data(B.e), q_ptr, &ldq, pt_ptr, &ldpt, c_ptr, &ldc, raw_data(work), &info)
	}

	return info, info == 0
}

// Complex banded to bidiagonal reduction (complex64/complex128)
band_to_bidi_complex :: proc(
	vect: BidiagVectorOption,
	AB: ^BandedMatrix($Cmplx), // Banded matrix (input/output - destroyed)
	B: ^Bidiagonal($Real), // Output bidiagonal matrix (real diagonals)
	Q: ^Matrix(Cmplx), // Pre-allocated Q matrix (optional)
	PT: ^Matrix(Cmplx), // Pre-allocated P^T matrix (optional)
	C: ^Matrix(Cmplx), // Matrix to apply transformations to (optional)
	work: []Cmplx, // Pre-allocated workspace (size max(m,n))
	rwork: []Real, // Pre-allocated real workspace (size max(m,n))
) -> (
	info: Info,
	ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
	m := AB.rows
	n := AB.cols
	kl := AB.kl
	ku := AB.ku
	ldab := AB.ldab

	// Validate input dimensions
	min_mn := min(int(m), int(n))
	assert(int(B.n) == min_mn, "Bidiagonal dimension must be min(m,n)")
	assert(len(B.d) >= min_mn, "Bidiagonal diagonal too small")
	assert(len(B.e) >= max(0, min_mn - 1), "Bidiagonal off-diagonal too small")
	assert(len(work) >= max(int(m), int(n)), "Work array too small")
	assert(len(rwork) >= max(int(m), int(n)), "Real work array too small")

	// Validate and set up Q matrix
	ldq := Blas_Int(1)
	q_ptr: ^Cmplx = nil
	if (vect == .Q_ONLY || vect == .BOTH) && Q != nil {
		assert(int(Q.rows) >= int(m) && int(Q.cols) >= min_mn, "Q matrix too small")
		ldq = Q.ld
		q_ptr = raw_data(Q.data)
	}

	// Validate and set up P^T matrix
	ldpt := Blas_Int(1)
	pt_ptr: ^Cmplx = nil
	if (vect == .PT_ONLY || vect == .BOTH) && PT != nil {
		assert(int(PT.rows) >= min_mn && int(PT.cols) >= int(n), "P^T matrix too small")
		ldpt = PT.ld
		pt_ptr = raw_data(PT.data)
	}

	// Handle optional C matrix for applying transformations
	ncc := Blas_Int(0)
	ldc := Blas_Int(1)
	c_ptr: ^Cmplx = nil
	if C != nil {
		ncc = C.cols
		ldc = C.ld
		c_ptr = raw_data(C.data)
	}

	vect_c := cast(u8)vect

	when Cmplx == complex64 {
		lapack.cgbbrd_(&vect_c, &m, &n, &ncc, &kl, &ku, raw_data(AB.data), &ldab, raw_data(B.d), raw_data(B.e), q_ptr, &ldq, pt_ptr, &ldpt, c_ptr, &ldc, raw_data(work), raw_data(rwork), &info)
	} else when Cmplx == complex128 {
		lapack.zgbbrd_(&vect_c, &m, &n, &ncc, &kl, &ku, raw_data(AB.data), &ldab, raw_data(B.d), raw_data(B.e), q_ptr, &ldq, pt_ptr, &ldpt, c_ptr, &ldc, raw_data(work), raw_data(rwork), &info)
	}

	return info, info == 0
}
