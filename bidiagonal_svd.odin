package openblas

import lapack "./f77"

// ===================================================================================
// BIDIAGONAL SVD OPERATIONS
// ===================================================================================

// SVD of bidiagonal matrix using bdsqr
// Computes the SVD of a real n-by-n (upper or lower) bidiagonal matrix B
bidi_svd :: proc {
	bidi_svd_real,
	bidi_svd_complex,
}

// Query workspace size for bidiagonal SVD
query_workspace_bidi_svd :: proc(B: ^Bidiagonal($T), compute_u: bool = true, compute_vt: bool = true, compute_c: bool = false) -> (work_size: int, rwork_size: int) where is_float(T) || is_complex(T) {
	n := int(B.n)

	when is_float(T) {
		work_size = 4 * n // Real types need work array
		rwork_size = 0
	} else {
		work_size = 0
		rwork_size = 4 * n // Complex types need real work array
	}

	return
}

// Real bidiagonal SVD (f32/f64)
bidi_svd_real :: proc(
	B: ^Bidiagonal($T),
	U: ^Matrix(T), // Left singular vectors (optional)
	VT: ^Matrix(T), // Right singular vectors transposed (optional)
	C: ^Matrix(T), // Additional matrix to transform (optional)
	work: []T, // Workspace (pre-allocated)
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	n := B.n

	ncvt := Blas_Int(0) // Number of columns of VT
	nru := Blas_Int(0) // Number of rows of U
	ncc := Blas_Int(0) // Number of columns of C

	ldu := Blas_Int(1)
	ldvt := Blas_Int(1)
	ldc := Blas_Int(1)

	uplo_c := cast(u8)B.uplo

	assert(len(work) >= 4 * int(n), "Work array too small")

	u_ptr: ^T = nil
	vt_ptr: ^T = nil
	c_ptr: ^T = nil

	if VT != nil {
		ncvt = VT.cols
		ldvt = VT.ld
		vt_ptr = raw_data(VT.data)
		assert(VT.rows >= n, "VT must have at least n rows")
	}

	if U != nil {
		nru = U.rows
		ldu = U.ld
		u_ptr = raw_data(U.data)
		assert(U.cols >= n, "U must have at least n columns")
	}

	if C != nil {
		ncc = C.cols
		ldc = C.ld
		c_ptr = raw_data(C.data)
		assert(C.rows >= n, "C must have at least n rows")
	}

	when T == f32 {
		lapack.sbdsqr_(&uplo_c, &n, &ncvt, &nru, &ncc, raw_data(B.d), raw_data(B.e), vt_ptr, &ldvt, u_ptr, &ldu, c_ptr, &ldc, raw_data(work), &info)
	} else when T == f64 {
		lapack.dbdsqr_(&uplo_c, &n, &ncvt, &nru, &ncc, raw_data(B.d), raw_data(B.e), vt_ptr, &ldvt, u_ptr, &ldu, c_ptr, &ldc, raw_data(work), &info)
	}

	return info, info == 0
}

// Complex bidiagonal SVD (complex64/complex128)
bidi_svd_complex :: proc(
	B: ^Bidiagonal($Real), // Bidiagonal matrix (real)
	U: ^Matrix($Cmplx), // Left singular vectors (optional, complex)
	VT: ^Matrix(Cmplx), // Right singular vectors transposed (optional, complex)
	C: ^Matrix(Cmplx), // Additional matrix to transform (optional, complex)
	rwork: []Real, // Real workspace (pre-allocated)
) -> (
	info: Info,
	ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
	n := B.n

	// Determine what to compute
	ncvt := Blas_Int(0)
	nru := Blas_Int(0)
	ncc := Blas_Int(0)

	ldu := Blas_Int(1)
	ldvt := Blas_Int(1)
	ldc := Blas_Int(1)

	uplo_c := cast(u8)B.uplo

	assert(len(rwork) >= 4 * int(n), "Real work array too small")

	u_ptr: ^Cmplx = nil
	vt_ptr: ^Cmplx = nil
	c_ptr: ^Cmplx = nil

	if VT != nil {
		ncvt = VT.cols
		ldvt = VT.ld
		vt_ptr = raw_data(VT.data)
		assert(VT.rows >= n, "VT must have at least n rows")
	}

	if U != nil {
		nru = U.rows
		ldu = U.ld
		u_ptr = raw_data(U.data)
		assert(U.cols >= n, "U must have at least n columns")
	}

	if C != nil {
		ncc = C.cols
		ldc = C.ld
		c_ptr = raw_data(C.data)
		assert(C.rows >= n, "C must have at least n rows")
	}

	when T == complex64 {
		lapack.cbdsqr_(&uplo_c, &n, &ncvt, &nru, &ncc, raw_data(B.d), raw_data(B.e), vt_ptr, &ldvt, u_ptr, &ldu, c_ptr, &ldc, raw_data(rwork), &info)
	} else when T == complex128 {
		lapack.zbdsqr_(&uplo_c, &n, &ncvt, &nru, &ncc, raw_data(B.d), raw_data(B.e), vt_ptr, &ldvt, u_ptr, &ldu, c_ptr, &ldc, raw_data(rwork), &info)
	}

	return info, info == 0
}

// ===================================================================================
// BIDIAGONAL DIVIDE AND CONQUER SVD
// ===================================================================================

// Job options for bidiagonal divide-and-conquer
Bidiagonal_DC_Job :: enum u8 {
	None     = 'N', // Compute singular values only
	Compact  = 'P', // Compute singular values and compact singular vectors
	Implicit = 'I', // Compute singular values and singular vectors
}

// Real bidiagonal divide-and-conquer SVD (f32/f64)
bidi_svd_dc :: proc(B: ^Bidiagonal($T), U: ^Matrix(T), VT: ^Matrix(T), work: []T, iwork: []Blas_Int, compq: Bidiagonal_DC_Job = .Implicit) -> (info: Info, ok: bool) where is_float(T) {
	n := B.n
	uplo_c := cast(u8)B.uplo
	compq_c := cast(u8)compq

	assert(compq != .Compact, ".Compact Not Implemented")

	ldu := Blas_Int(1)
	ldvt := Blas_Int(1)
	u_ptr: ^T = nil
	vt_ptr: ^T = nil

	if compq == .Implicit {
		if U != nil {
			ldu = U.ld
			u_ptr = raw_data(U.data)
			assert(U.rows == n && U.cols == n, "U must be n x n")
		}
		if VT != nil {
			ldvt = VT.ld
			vt_ptr = raw_data(VT.data)
			assert(VT.rows == n && VT.cols == n, "VT must be n x n")
		}
	}

	// Q and IQ are only used for compact storage (not implemented here)
	q_dummy: T
	iq_dummy: Blas_Int

	when T == f32 {
		lapack.sbdsdc_(&uplo_c, &compq_c, &n, raw_data(B.d), raw_data(B.e), u_ptr, &ldu, vt_ptr, &ldvt, &q_dummy, &iq_dummy, raw_data(work), raw_data(iwork), &info)
	} else when T == f64 {
		lapack.dbdsdc_(&uplo_c, &compq_c, &n, raw_data(B.d), raw_data(B.e), u_ptr, &ldu, vt_ptr, &ldvt, &q_dummy, &iq_dummy, raw_data(work), raw_data(iwork), &info)
	}

	return info, info == 0
}

// ===================================================================================
// BIDIAGONAL SELECTIVE SVD
// ===================================================================================

// Query workspace size for selective bidiagonal SVD (bdsvdx)
query_workspace_bidi_svd_select :: proc(B: ^Bidiagonal($T)) -> (work_size: int, iwork_size: int) where is_float(T) {
	// Based on LAPACK documentation for sbdsvdx/dbdsvdx
	// WORK dimension: 14*N
	// IWORK dimension: 12*N
	n := int(B.n)
	work_size = 14 * n
	iwork_size = 12 * n
	return
}

// Real selective bidiagonal SVD (f32/f64)
bidi_svd_select :: proc(
	B: ^Bidiagonal($T),
	S: []T, // Output singular values
	Z: ^Matrix(T), // Output singular vectors
	work: []T,
	iwork: []Blas_Int,
	jobz: SVD_Select_Job = .Vectors,
	range: SVD_Range_Option = .All,
	vl: T,
	vu: T,
	il: Blas_Int = 1,
	iu: Blas_Int = -1,
) -> (
	ns: Blas_Int,
	info: Info,
	ok: bool,
) where is_float(T) {
	n := B.n
	uplo_c := cast(u8)B.uplo
	jobz_c := cast(u8)jobz
	range_c := cast(u8)range

	vl := vl
	vu := vu
	il := il
	iu := iu

	if iu < 0 {iu = n}

	ldz := Blas_Int(1)
	z_ptr: ^T = nil
	if Z != nil && jobz == .Vectors {
		ldz = Z.ld
		z_ptr = raw_data(Z.data)
	}

	when T == f32 {
		lapack.sbdsvdx_(&uplo_c, &jobz_c, &range_c, &n, raw_data(B.d), raw_data(B.e), &vl, &vu, &il, &iu, &ns, raw_data(S), z_ptr, &ldz, raw_data(work), raw_data(iwork), &info)
	} else when T == f64 {
		lapack.dbdsvdx_(&uplo_c, &jobz_c, &range_c, &n, raw_data(B.d), raw_data(B.e), &vl, &vu, &il, &iu, &ns, raw_data(S), z_ptr, &ldz, raw_data(work), raw_data(iwork), &info)
	}

	return ns, info, info == 0
}


// ===================================================================================
// CS (COSINE-SINE) DECOMPOSITION FOR BIDIAGONAL-BLOCK MATRICES
// ===================================================================================

// CS Decomposition computation options
CS_Job :: enum u8 {
	None = 'N', // Do not compute orthogonal matrices
	Yes  = 'Y', // Compute orthogonal matrices
}

// Structure for CS decomposition workspace and parameter arrays
// Note: phi field reserved for future 2-by-1 CS decomposition (ORCSD2BY1)
CS_Arrays :: struct($T: typeid) where is_float(T) || is_complex(T) {
	theta: []T, // Angle arrays for CS decomposition
	phi:   []T, // Additional angle arrays (reserved for ORCSD2BY1)
	work:  []T, // Workspace for computation
	rwork: []T, // Real workspace (for complex types)
	iwork: []Blas_Int, // Integer workspace
}

// Create CS decomposition parameter arrays for bidiagonal-block matrices
make_bidiag_real_arrays :: proc($T: typeid, p, q: int, allocator := context.allocator) -> CS_Arrays(T) where is_float(T) {
	return CS_Arrays(T) {
		theta = make([]T, min(p, q), allocator),
		phi   = make([]T, min(p, q) - 1, allocator),
		work  = make([]T, max(1, 2 * min(p, q) - 1), allocator),
		rwork = make([]T, 0, allocator), // Not needed for real types
		iwork = make([]Blas_Int, 0, allocator), // Not needed for basic CS decomp
	}
}

// Query workspace size for CS decomposition of bidiagonal-block matrices
query_workspace_cs_decomp :: proc(p, q: int, is_complex := false) -> (work_size: int, rwork_size: int, iwork_size: int) {
	if !is_complex {
		// Real types require work array for CS decomposition
		work_size = max(1, 2 * min(p, q) - 1)
		rwork_size = 0
		iwork_size = 0
	} else {
		// Complex types require both work and rwork arrays
		work_size = max(1, 2 * min(p, q) - 1)
		rwork_size = max(1, 2 * min(p, q) - 1)
		iwork_size = 0
	}
	return
}

// CS decomposition for bidiagonal-block matrices
cs_decomp :: proc {
	cs_decomp_real,
	cs_decomp_complex,
}

// Prepare CS decomposition workspace and validate matrix dimensions
cs_decomp_prepare :: proc(X11: ^Matrix($T), X12, X21, X22: ^Matrix(T)) -> (p, q: Blas_Int, ok: bool) where is_float(T) || is_complex(T) {
	// Validate that all matrices have the same element type and are compatible
	if X11 == nil {
		return 0, 0, false
	}

	p = X11.rows
	q = X11.cols

	// Check dimension compatibility for block structure
	if X12 != nil {
		if X12.rows != p {
			return 0, 0, false
		}
	}

	if X21 != nil {
		if X21.cols != q {
			return 0, 0, false
		}
	}

	if X22 != nil && X12 != nil && X21 != nil {
		if X22.rows != X21.rows || X22.cols != X12.cols {
			return 0, 0, false
		}
	}

	return p, q, true
}

// Real CS decomposition (f32/f64) for matrices in bidiagonal-block form
cs_decomp_real :: proc(
	X11: ^Matrix($T), // Upper-left block (p x q)
	X12: ^Matrix(T), // Upper-right block (p x m-q)
	X21: ^Matrix(T), // Lower-left block (n-p x q)
	X22: ^Matrix(T), // Lower-right block (n-p x m-q)
	theta: []T, // Angle array (output, length min(p,q))
	U1: ^Matrix(T), // Left orthogonal matrix (p x p)
	U2: ^Matrix(T), // Left orthogonal matrix ((n-p) x (n-p))
	V1T: ^Matrix(T), // Right orthogonal matrix (q x q)
	V2T: ^Matrix(T), // Right orthogonal matrix ((m-q) x (m-q))
	work: []T, // Workspace
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {

	p, q, valid := cs_decomp_prepare(X11, X12, X21, X22)
	if !valid {
		return Info(-1), false
	}

	// Dimensions for other blocks
	m := q + (X12 != nil ? X12.cols : 0)
	n := p + (X21 != nil ? X21.rows : 0)

	// Leading dimensions and pointers
	ldx11 := X11.ld

	ldx12 := Blas_Int(1)
	x12_ptr: ^T = nil
	if X12 != nil {
		ldx12 = X12.ld
		x12_ptr = raw_data(X12.data)
	}

	ldx21 := Blas_Int(1)
	x21_ptr: ^T = nil
	if X21 != nil {
		ldx21 = X21.ld
		x21_ptr = raw_data(X21.data)
	}

	ldx22 := Blas_Int(1)
	x22_ptr: ^T = nil
	if X22 != nil {
		ldx22 = X22.ld
		x22_ptr = raw_data(X22.data)
	}

	// Orthogonal matrix dimensions and pointers
	ldu1 := Blas_Int(1)
	u1_ptr: ^T = nil
	if U1 != nil {
		ldu1 = U1.ld
		u1_ptr = raw_data(U1.data)
		assert(U1.rows >= p && U1.cols >= p, "U1 matrix dimensions incorrect")
	}

	ldu2 := Blas_Int(1)
	u2_ptr: ^T = nil
	if U2 != nil {
		ldu2 = U2.ld
		u2_ptr = raw_data(U2.data)
		assert(U2.rows >= (n - p) && U2.cols >= (n - p), "U2 matrix dimensions incorrect")
	}

	ldv1t := Blas_Int(1)
	v1t_ptr: ^T = nil
	if V1T != nil {
		ldv1t = V1T.ld
		v1t_ptr = raw_data(V1T.data)
		assert(V1T.rows >= q && V1T.cols >= q, "V1T matrix dimensions incorrect")
	}

	ldv2t := Blas_Int(1)
	v2t_ptr: ^T = nil
	if V2T != nil {
		ldv2t = V2T.ld
		v2t_ptr = raw_data(V2T.data)
		assert(V2T.rows >= (m - q) && V2T.cols >= (m - q), "V2T matrix dimensions incorrect")
	}

	// Job character for computing orthogonal matrices
	jobu1 := u8('N')
	jobu2 := u8('N')
	jobv1t := u8('N')
	jobv2t := u8('N')

	if U1 != nil do jobu1 = u8('Y')
	if U2 != nil do jobu2 = u8('Y')
	if V1T != nil do jobv1t = u8('Y')
	if V2T != nil do jobv2t = u8('Y')

	min_work := max(1, 2 * min(p, q) - 1)
	assert(len(work) >= min_work, "Work array too small for CS decomposition")
	assert(len(theta) >= int(min(p, q)), "Theta array too small")

	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.sorcsd_(&jobu1, &jobu2, &jobv1t, &jobv2t, u8('N'), u8('N'), &m, &p, &q, raw_data(X11.data), &ldx11, x12_ptr, &ldx12, x21_ptr, &ldx21, x22_ptr, &ldx22, raw_data(theta), u1_ptr, &ldu1, u2_ptr, &ldu2, v1t_ptr, &ldv1t, v2t_ptr, &ldv2t, raw_data(work), &lwork, &info)
	} else when T == f64 {
		lapack.dorcsd_(&jobu1, &jobu2, &jobv1t, &jobv2t, u8('N'), u8('N'), &m, &p, &q, raw_data(X11.data), &ldx11, x12_ptr, &ldx12, x21_ptr, &ldx21, x22_ptr, &ldx22, raw_data(theta), u1_ptr, &ldu1, u2_ptr, &ldu2, v1t_ptr, &ldv1t, v2t_ptr, &ldv2t, raw_data(work), &lwork, &info)
	}

	return info, info == 0
}

// Complex CS decomposition (complex64/complex128) for matrices in bidiagonal-block form
cs_decomp_complex :: proc(
	X11: ^Matrix($T), // Upper-left block (p x q)
	X12: ^Matrix(T), // Upper-right block (p x m-q)
	X21: ^Matrix(T), // Lower-left block (n-p x q)
	X22: ^Matrix(T), // Lower-right block (n-p x m-q)
	theta: []$Real, // Real angle array (output, length min(p,q))
	U1: ^Matrix(T), // Left unitary matrix (p x p)
	U2: ^Matrix(T), // Left unitary matrix ((n-p) x (n-p))
	V1H: ^Matrix(T), // Right unitary matrix (q x q)
	V2H: ^Matrix(T), // Right unitary matrix ((m-q) x (m-q))
	work: []T, // Complex workspace
	rwork: []Real, // Real workspace
) -> (
	info: Info,
	ok: bool,
) where (T == complex64 && Real == f32) || (T == complex128 && Real == f64) {

	p, q, valid := cs_decomp_prepare(X11, X12, X21, X22)
	if !valid {
		return Info(-1), false
	}

	// Dimensions for other blocks
	m := q + (X12 != nil ? X12.cols : 0)
	n := p + (X21 != nil ? X21.rows : 0)

	// Leading dimensions and pointers
	ldx11 := X11.ld

	ldx12 := Blas_Int(1)
	x12_ptr: ^T = nil
	if X12 != nil {
		ldx12 = X12.ld
		x12_ptr = raw_data(X12.data)
	}

	ldx21 := Blas_Int(1)
	x21_ptr: ^T = nil
	if X21 != nil {
		ldx21 = X21.ld
		x21_ptr = raw_data(X21.data)
	}

	ldx22 := Blas_Int(1)
	x22_ptr: ^T = nil
	if X22 != nil {
		ldx22 = X22.ld
		x22_ptr = raw_data(X22.data)
	}

	// Unitary matrix dimensions and pointers
	ldu1 := Blas_Int(1)
	u1_ptr: ^T = nil
	if U1 != nil {
		ldu1 = U1.ld
		u1_ptr = raw_data(U1.data)
		assert(U1.rows >= p && U1.cols >= p, "U1 matrix dimensions incorrect")
	}

	ldu2 := Blas_Int(1)
	u2_ptr: ^T = nil
	if U2 != nil {
		ldu2 = U2.ld
		u2_ptr = raw_data(U2.data)
		assert(U2.rows >= (n - p) && U2.cols >= (n - p), "U2 matrix dimensions incorrect")
	}

	ldv1h := Blas_Int(1)
	v1h_ptr: ^T = nil
	if V1H != nil {
		ldv1h = V1H.ld
		v1h_ptr = raw_data(V1H.data)
		assert(V1H.rows >= q && V1H.cols >= q, "V1H matrix dimensions incorrect")
	}

	ldv2h := Blas_Int(1)
	v2h_ptr: ^T = nil
	if V2H != nil {
		ldv2h = V2H.ld
		v2h_ptr = raw_data(V2H.data)
		assert(V2H.rows >= (m - q) && V2H.cols >= (m - q), "V2H matrix dimensions incorrect")
	}

	// Job character for computing unitary matrices
	jobu1 := u8('N')
	jobu2 := u8('N')
	jobv1h := u8('N')
	jobv2h := u8('N')

	if U1 != nil do jobu1 = u8('Y')
	if U2 != nil do jobu2 = u8('Y')
	if V1H != nil do jobv1h = u8('Y')
	if V2H != nil do jobv2h = u8('Y')

	min_work := max(1, 2 * min(p, q) - 1)
	min_rwork := max(1, 2 * min(p, q) - 1)
	assert(len(work) >= min_work, "Work array too small for complex CS decomposition")
	assert(len(rwork) >= min_rwork, "Real work array too small for complex CS decomposition")
	assert(len(theta) >= int(min(p, q)), "Theta array too small")

	lwork := Blas_Int(len(work))
	lrwork := Blas_Int(len(rwork))

	when T == complex64 {
		lapack.cuncsd_(
			&jobu1,
			&jobu2,
			&jobv1h,
			&jobv2h,
			u8('N'),
			u8('N'),
			&m,
			&p,
			&q,
			raw_data(X11.data),
			&ldx11,
			x12_ptr,
			&ldx12,
			x21_ptr,
			&ldx21,
			x22_ptr,
			&ldx22,
			raw_data(theta),
			u1_ptr,
			&ldu1,
			u2_ptr,
			&ldu2,
			v1h_ptr,
			&ldv1h,
			v2h_ptr,
			&ldv2h,
			raw_data(work),
			&lwork,
			raw_data(rwork),
			&lrwork,
			&info,
		)
	} else when T == complex128 {
		lapack.zuncsd_(
			&jobu1,
			&jobu2,
			&jobv1h,
			&jobv2h,
			u8('N'),
			u8('N'),
			&m,
			&p,
			&q,
			raw_data(X11.data),
			&ldx11,
			x12_ptr,
			&ldx12,
			x21_ptr,
			&ldx21,
			x22_ptr,
			&ldx22,
			raw_data(theta),
			u1_ptr,
			&ldu1,
			u2_ptr,
			&ldu2,
			v1h_ptr,
			&ldv1h,
			v2h_ptr,
			&ldv2h,
			raw_data(work),
			&lwork,
			raw_data(rwork),
			&lrwork,
			&info,
		)
	}

	return info, info == 0
}

// ===================================================================================
// 2-BY-1 CS DECOMPOSITION (ORCSD2BY1)
// For matrices with 2 vertical blocks only
// ===================================================================================

// 2-by-1 CS decomposition
cs_decomp_2by1 :: proc {
	cs_decomp_2by1_real,
	cs_decomp_2by1_complex,
}
// Query workspace for 2-by-1 CS decomposition
query_workspace_cs_decomp_2by1 :: proc(p, q: int) -> (work: int, iwork: int) {
	// Based on LAPACK documentation for ORCSD2BY1
	min_pq := min(p, q)
	work = max(1, min_pq * min_pq + 8 * min_pq)
	iwork = min_pq
	return
}

// Real 2-by-1 CS decomposition (f32/f64)
cs_decomp_2by1_real :: proc(
	X11: ^Matrix($T), // Upper block (p x q)
	X21: ^Matrix(T), // Lower block ((m-p) x q)
	theta: []T, // Angle array (output, length min(p,q))
	U1: ^Matrix(T), // Left orthogonal matrix (p x p) - optional
	U2: ^Matrix(T), // Left orthogonal matrix ((m-p) x (m-p)) - optional
	V1T: ^Matrix(T), // Right orthogonal matrix (q x q) - optional
	work: []T, // Workspace
	iwork: []Blas_Int, // Integer workspace
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	// Validate X11
	p := X11.rows
	q := X11.cols
	ldx11 := X11.ld

	// Validate X21
	assert(X21 != nil, "X21 must not be nil for 2-by-1 CS decomposition")
	assert(X21.cols == q, "X21 must have same number of columns as X11")
	m := p + X21.rows
	ldx21 := X21.ld

	// Orthogonal matrix dimensions and pointers
	ldu1 := Blas_Int(1)
	u1_ptr: ^T = nil
	if U1 != nil {
		ldu1 = U1.ld
		u1_ptr = raw_data(U1.data)
		assert(U1.rows >= p && U1.cols >= p, "U1 matrix dimensions incorrect")
	}

	ldu2 := Blas_Int(1)
	u2_ptr: ^T = nil
	if U2 != nil {
		ldu2 = U2.ld
		u2_ptr = raw_data(U2.data)
		assert(U2.rows >= (m - p) && U2.cols >= (m - p), "U2 matrix dimensions incorrect")
	}

	ldv1t := Blas_Int(1)
	v1t_ptr: ^T = nil
	if V1T != nil {
		ldv1t = V1T.ld
		v1t_ptr = raw_data(V1T.data)
		assert(V1T.rows >= q && V1T.cols >= q, "V1T matrix dimensions incorrect")
	}

	// Job characters
	jobu1 := u8('N')
	jobu2 := u8('N')
	jobv1t := u8('N')

	if U1 != nil do jobu1 = u8('Y')
	if U2 != nil do jobu2 = u8('Y')
	if V1T != nil do jobv1t = u8('Y')

	assert(len(theta) >= int(min(p, q)), "Theta array too small")

	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.sorcsd2by1_(&jobu1, &jobu2, &jobv1t, &m, &p, &q, raw_data(X11.data), &ldx11, raw_data(X21.data), &ldx21, raw_data(theta), u1_ptr, &ldu1, u2_ptr, &ldu2, v1t_ptr, &ldv1t, raw_data(work), &lwork, raw_data(iwork), &info)
	} else when T == f64 {
		lapack.dorcsd2by1_(&jobu1, &jobu2, &jobv1t, &m, &p, &q, raw_data(X11.data), &ldx11, raw_data(X21.data), &ldx21, raw_data(theta), u1_ptr, &ldu1, u2_ptr, &ldu2, v1t_ptr, &ldv1t, raw_data(work), &lwork, raw_data(iwork), &info)
	}

	return info, info == 0
}

// Complex 2-by-1 CS decomposition (complex64/complex128)
cs_decomp_2by1_complex :: proc(
	X11: ^Matrix($Cmplx), // Upper block (p x q)
	X21: ^Matrix(Cmplx), // Lower block ((m-p) x q)
	theta: []$Real, // Real angle array (output, length min(p,q))
	U1: ^Matrix(Cmplx), // Left unitary matrix (p x p) - optional
	U2: ^Matrix(Cmplx), // Left unitary matrix ((m-p) x (m-p)) - optional
	V1H: ^Matrix(Cmplx), // Right unitary matrix (q x q) - optional
	work: []Cmplx, // Complex workspace
	rwork: []Real, // Real workspace
	iwork: []Blas_Int, // Integer workspace
) -> (
	info: Info,
	ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
	// Validate X11
	p := X11.rows
	q := X11.cols
	ldx11 := X11.ld

	// Validate X21
	assert(X21 != nil, "X21 must not be nil for 2-by-1 CS decomposition")
	assert(X21.cols == q, "X21 must have same number of columns as X11")
	m := p + X21.rows
	ldx21 := X21.ld

	// Unitary matrix dimensions and pointers
	ldu1 := Blas_Int(1)
	u1_ptr: ^Cmplx = nil
	if U1 != nil {
		ldu1 = U1.ld
		u1_ptr = raw_data(U1.data)
		assert(U1.rows >= p && U1.cols >= p, "U1 matrix dimensions incorrect")
	}

	ldu2 := Blas_Int(1)
	u2_ptr: ^Cmplx = nil
	if U2 != nil {
		ldu2 = U2.ld
		u2_ptr = raw_data(U2.data)
		assert(U2.rows >= (m - p) && U2.cols >= (m - p), "U2 matrix dimensions incorrect")
	}

	ldv1h := Blas_Int(1)
	v1h_ptr: ^Cmplx = nil
	if V1H != nil {
		ldv1h = V1H.ld
		v1h_ptr = raw_data(V1H.data)
		assert(V1H.rows >= q && V1H.cols >= q, "V1H matrix dimensions incorrect")
	}

	// Job characters
	jobu1 := u8('N')
	jobu2 := u8('N')
	jobv1h := u8('N')

	if U1 != nil do jobu1 = u8('Y')
	if U2 != nil do jobu2 = u8('Y')
	if V1H != nil do jobv1h = u8('Y')

	assert(len(theta) >= int(min(p, q)), "Theta array too small")

	lwork := Blas_Int(len(work))
	lrwork := Blas_Int(len(rwork))

	when Cmplx == complex64 {
		lapack.cuncsd2by1_(&jobu1, &jobu2, &jobv1h, &m, &p, &q, raw_data(X11.data), &ldx11, raw_data(X21.data), &ldx21, raw_data(theta), u1_ptr, &ldu1, u2_ptr, &ldu2, v1h_ptr, &ldv1h, raw_data(work), &lwork, raw_data(rwork), &lrwork, raw_data(iwork), &info)
	} else when Cmplx == complex128 {
		lapack.zuncsd2by1_(&jobu1, &jobu2, &jobv1h, &m, &p, &q, raw_data(X11.data), &ldx11, raw_data(X21.data), &ldx21, raw_data(theta), u1_ptr, &ldu1, u2_ptr, &ldu2, v1h_ptr, &ldv1h, raw_data(work), &lwork, raw_data(rwork), &lrwork, raw_data(iwork), &info)
	}

	return info, info == 0
}
