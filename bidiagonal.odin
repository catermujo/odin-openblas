package openblas

import lapack "./f77"

// Import types from other modules (assuming they're defined there)
// Matrix, BandedMatrix, UpLo, Info, Blas_Int should be available in the package


// ===================================================================================
// BIDIAGONAL MATRIX TYPE
// ===================================================================================

// Bidiagonal matrix representation for SVD computations
// Stores only the diagonal and one off-diagonal (2n-1 elements total)
Bidiagonal :: struct($T: typeid) {
	n:    Blas_Int, // Matrix dimension (n x n)
	uplo: UpLo, // Upper or Lower bidiagonal
	d:    []T, // Main diagonal (length n)
	e:    []T, // Off-diagonal (length n-1)
	// If uplo='U': superdiagonal
	// If uplo='L': subdiagonal
}

make_bidiagonal :: proc($T: typeid, n: int, uplo: UpLo = .Upper, allocator := context.allocator) -> Bidiagonal(T) {
	return Bidiagonal(T){n = Blas_Int(n), uplo = uplo, d = make([]T, n, allocator), e = make([]T, max(0, n - 1), allocator)}
}

// Delete bidiagonal matrix
delete_bidiagonal :: proc(B: ^Bidiagonal($T)) {
	delete(B.d)
	delete(B.e)
	B.d = nil
	B.e = nil
}

bidiagonal_from_arrays :: proc($T: typeid, d: []T, e: []T, uplo: UpLo = .Upper) -> Bidiagonal(T) {
	assert(len(d) > 0, "Main diagonal must have at least one element")
	assert(len(e) == len(d) - 1, "Off-diagonal must have n-1 elements")

	return Bidiagonal(T){n = Blas_Int(len(d)), uplo = uplo, d = d, e = e}
}

// ===================================================================================
// BIDIAGONAL REDUCTION
// Reduce general matrix to bidiagonal form for SVD computation
// ===================================================================================

// Query result sizes for bidiagonal reduction
query_result_sizes_bidiagonalize :: proc(A: ^Matrix($T)) -> (d_size: int, e_size: int, tauq_size: int, taup_size: int) where is_float(T) || is_complex(T) {
	min_mn := int(min(A.rows, A.cols))
	d_size = min_mn
	e_size = min_mn - 1
	tauq_size = min_mn
	taup_size = min_mn
	return
}

// Query workspace for bidiagonal reduction
query_workspace_bidiagonalize :: proc(A: ^Matrix($T)) -> (work: int) where is_float(T) || is_complex(T) {
	// Query LAPACK for optimal workspace
	m := A.rows
	n := A.cols
	lda := A.ld
	lwork := QUERY_WORKSPACE
	info: Info
	work_query: T

	when is_float(T) {
		when T == f32 {
			lapack.sgebrd_(&m, &n, nil, &lda, nil, nil, nil, nil, &work_query, &lwork, &info)
		} else when T == f64 {
			lapack.dgebrd_(&m, &n, nil, &lda, nil, nil, nil, nil, &work_query, &lwork, &info)
		}
		return int(work_query)
	} else when is_complex(T) {
		when T == complex64 {
			lapack.cgebrd_(&m, &n, nil, &lda, nil, nil, nil, nil, &work_query, &lwork, &info)
		} else when T == complex128 {
			lapack.zgebrd_(&m, &n, nil, &lda, nil, nil, nil, nil, &work_query, &lwork, &info)
		}
		return int(real(work_query))
	}
}

// Reduce general matrix to bidiagonal form using Householder reflections
// A = Q * B * P^H where B is bidiagonal
bidiagonalize :: proc {
	bidiagonalize_real,
	bidiagonalize_complex,
}

// Real bidiagonal reduction (f32/f64)
bidiagonalize_real :: proc(
	A: ^Matrix($T), // General matrix (overwritten with bidiagonal form)
	B: ^Bidiagonal(T), // Pre-allocated bidiagonal matrix
	tauq: []T, // Pre-allocated scalar factors for Q (size min(m,n))
	taup: []T, // Pre-allocated scalar factors for P (size min(m,n))
	work: []T, // Pre-allocated workspace
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	m := A.rows
	n := A.cols
	lda := A.ld
	min_mn := min(m, n)

	assert(int(B.n) == int(min_mn), "Bidiagonal dimension must be min(m,n)")
	assert(len(B.d) >= int(min_mn), "Bidiagonal d array too small")
	assert(len(B.e) >= int(min_mn - 1), "Bidiagonal e array too small")
	assert(len(tauq) >= int(min_mn), "tauq array too small")
	assert(len(taup) >= int(min_mn), "taup array too small")

	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.sgebrd_(&m, &n, raw_data(A.data), &lda, raw_data(B.d), raw_data(B.e), raw_data(tauq), raw_data(taup), raw_data(work), &lwork, &info)
	} else when T == f64 {
		lapack.dgebrd_(&m, &n, raw_data(A.data), &lda, raw_data(B.d), raw_data(B.e), raw_data(tauq), raw_data(taup), raw_data(work), &lwork, &info)
	}

	return info, info == 0
}

// Complex bidiagonal reduction (complex64/complex128)
bidiagonalize_complex :: proc(
	A: ^Matrix($Cmplx), // General matrix (overwritten with bidiagonal form)
	B: ^Bidiagonal($Real), // Pre-allocated bidiagonal matrix (real diagonals)
	tauq: []Cmplx, // Pre-allocated scalar factors for Q (size min(m,n))
	taup: []Cmplx, // Pre-allocated scalar factors for P (size min(m,n))
	work: []Cmplx, // Pre-allocated workspace
) -> (
	info: Info,
	ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
	m := A.rows
	n := A.cols
	lda := A.ld
	min_mn := min(m, n)

	assert(int(B.n) == int(min_mn), "Bidiagonal dimension must be min(m,n)")
	assert(len(B.d) >= int(min_mn), "Bidiagonal d array too small")
	assert(len(B.e) >= int(min_mn - 1), "Bidiagonal e array too small")
	assert(len(tauq) >= int(min_mn), "tauq array too small")
	assert(len(taup) >= int(min_mn), "taup array too small")

	lwork := Blas_Int(len(work))

	when Cmplx == complex64 {
		lapack.cgebrd_(&m, &n, raw_data(A.data), &lda, raw_data(B.d), raw_data(B.e), raw_data(tauq), raw_data(taup), raw_data(work), &lwork, &info)
	} else when Cmplx == complex128 {
		lapack.zgebrd_(&m, &n, raw_data(A.data), &lda, raw_data(B.d), raw_data(B.e), raw_data(tauq), raw_data(taup), raw_data(work), &lwork, &info)
	}

	return info, info == 0
}


// ===================================================================================
// BANDED TO BIDIAGONAL CONVERSION
// ===================================================================================

// Banded to bidiagonal computation type
BandedToBidiagJob :: enum u8 {
	None = 'N', // No orthogonal matrices computed
	Q    = 'Q', // Compute Q only
	P    = 'P', // Compute P^T only
	Both = 'B', // Compute both Q and P^T
}

// CS decomposition signs convention
CSDecompositionSigns :: enum u8 {
	Positive = '+', // Use positive signs convention
	Negative = '-', // Use negative signs convention
}

// Reduce banded matrix to bidiagonal form
bidi_reduce_from_band :: proc {
	bidi_reduce_from_band_real,
	bidi_reduce_from_band_complex,
}

// Query workspace size for banded to bidiagonal reduction
query_workspace_bidi_reduce_from_band :: proc(AB: ^BandedMatrix($T)) -> (work: int, rwork: int) where is_float(T) || is_complex(T) {
	m := int(AB.rows)
	n := int(AB.cols)

	when is_float(T) {
		return 2 * max(m, n), 0
	} else when is_complex(T) {
		return max(m, n), max(m, n)
	}
}

// Real banded to bidiagonal reduction
bidi_reduce_from_band_real :: proc(
	AB: ^BandedMatrix($T), // Banded matrix (modified on output)
	B: ^Bidiagonal(T), // Output bidiagonal matrix
	Q: ^Matrix(T), // Pre-allocated Q matrix (optional)
	PT: ^Matrix(T), // Pre-allocated P^T matrix (optional)
	C: ^Matrix(T), // Apply transformation to C (optional)
	work: []T, // Pre-allocated workspace
	job: BandedToBidiagJob = .None, // Which matrices to compute
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	m := AB.rows
	n := AB.cols
	kl := AB.kl
	ku := AB.ku
	ldab := AB.ldab

	min_mn := min(m, n)
	assert(B != nil, "Bidiagonal output matrix required")
	assert(B.n == min_mn, "Bidiagonal matrix size mismatch")
	assert(len(B.d) >= int(min_mn), "Bidiagonal D array too small")
	assert(len(B.e) >= int(max(0, min_mn - 1)), "Bidiagonal E array too small")
	assert(len(work) >= 2 * int(max(m, n)), "Work array too small")

	// Handle Q and PT
	ldq := Blas_Int(1)
	ldpt := Blas_Int(1)
	q_ptr: ^T = nil
	pt_ptr: ^T = nil

	if Q != nil && (job == .Q || job == .Both) {
		assert(Q.rows >= m && Q.cols >= min_mn, "Q matrix too small")
		ldq = Q.ld
		q_ptr = raw_data(Q.data)
	}

	if PT != nil && (job == .P || job == .Both) {
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
		assert(C.rows >= n, "C matrix too small")
	}

	job_c := cast(u8)job

	when T == f32 {
		lapack.sgbbrd_(&job_c, &m, &n, &ncc, &kl, &ku, raw_data(AB.data), &ldab, raw_data(B.d), raw_data(B.e), q_ptr, &ldq, pt_ptr, &ldpt, c_ptr, &ldc, raw_data(work), &info)
	} else when T == f64 {
		lapack.dgbbrd_(&job_c, &m, &n, &ncc, &kl, &ku, raw_data(AB.data), &ldab, raw_data(B.d), raw_data(B.e), q_ptr, &ldq, pt_ptr, &ldpt, c_ptr, &ldc, raw_data(work), &info)
	}

	return info, info == 0
}

// Complex banded to bidiagonal reduction
bidi_reduce_from_band_complex :: proc(
	AB: ^BandedMatrix($Cmplx), // Banded matrix (modified on output)
	B: ^Bidiagonal($Real), // Output bidiagonal matrix (real diagonals)
	Q: ^Matrix(Cmplx), // Pre-allocated Q matrix (optional)
	PT: ^Matrix(Cmplx), // Pre-allocated P^T matrix (optional)
	C: ^Matrix(Cmplx), // Apply transformation to C (optional)
	work: []Cmplx, // Pre-allocated workspace
	rwork: []Real, // Pre-allocated real workspace
	job: BandedToBidiagJob = .None, // Which matrices to compute
) -> (
	info: Info,
	ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
	m := AB.rows
	n := AB.cols
	kl := AB.kl
	ku := AB.ku
	ldab := AB.ldab

	// Validate sizes
	min_mn := min(m, n)
	assert(B != nil, "Bidiagonal output matrix required")
	assert(B.n == min_mn, "Bidiagonal matrix size mismatch")
	assert(len(B.d) >= int(min_mn), "Bidiagonal D array too small")
	assert(len(B.e) >= int(max(0, min_mn - 1)), "Bidiagonal E array too small")
	assert(len(work) >= int(max(m, n)), "Work array too small")
	assert(len(rwork) >= int(max(m, n)), "Real work array too small")

	// Handle Q and PT
	ldq := Blas_Int(1)
	ldpt := Blas_Int(1)
	q_ptr: ^Cmplx = nil
	pt_ptr: ^Cmplx = nil

	if Q != nil && (job == .Q || job == .Both) {
		assert(Q.rows >= m && Q.cols >= min_mn, "Q matrix too small")
		ldq = Q.ld
		q_ptr = raw_data(Q.data)
	}

	if PT != nil && (job == .P || job == .Both) {
		assert(PT.rows >= min_mn && PT.cols >= n, "PT matrix too small")
		ldpt = PT.ld
		pt_ptr = raw_data(PT.data)
	}

	// Handle C matrix
	ncc := Blas_Int(0)
	ldc := Blas_Int(1)
	c_ptr: ^Cmplx = nil
	if C != nil {
		ncc = C.cols
		ldc = C.ld
		c_ptr = raw_data(C.data)
		assert(C.rows >= n, "C matrix too small")
	}

	job_c := cast(u8)job

	when Cmplx == complex64 {
		lapack.cgbbrd_(&job_c, &m, &n, &ncc, &kl, &ku, raw_data(AB.data), &ldab, raw_data(B.d), raw_data(B.e), q_ptr, &ldq, pt_ptr, &ldpt, c_ptr, &ldc, raw_data(work), raw_data(rwork), &info)
	} else when Cmplx == complex128 {
		lapack.zgbbrd_(&job_c, &m, &n, &ncc, &kl, &ku, raw_data(AB.data), &ldab, raw_data(B.d), raw_data(B.e), q_ptr, &ldq, pt_ptr, &ldpt, c_ptr, &ldc, raw_data(work), raw_data(rwork), &info)
	}

	return info, info == 0
}

// ===================================================================================
// BIDIAGONAL BLOCK DIAGONALIZATION (CS Decomposition)
// ===================================================================================


// Query workspace size for bidiagonal block diagonalization
query_workspace_bidi_block_diagonalize :: proc(X11: ^Matrix($T), X12: ^Matrix(T), X21: ^Matrix(T), X22: ^Matrix(T)) -> (work: int, rwork: int) where is_float(T) || is_complex(T) {
	p := X11.rows
	q := X11.cols
	m_p := X21.rows
	m := p + m_p
	m_q := X22.cols

	ldx11 := X11.ld
	ldx12 := X12.ld
	ldx21 := X21.ld
	ldx22 := X22.ld
	lwork := QUERY_WORKSPACE
	info: Info
	work_query: T

	trans_c := u8('N')
	signs_c := u8('P')

	when is_float(T) {
		when T == f32 {
			lapack.sorbdb_(&trans_c, &signs_c, &m, &p, &q, nil, &ldx11, nil, &ldx12, nil, &ldx21, nil, &ldx22, nil, nil, nil, nil, nil, nil, &work_query, &lwork, &info)
		} else when T == f64 {
			lapack.dorbdb_(&trans_c, &signs_c, &m, &p, &q, nil, &ldx11, nil, &ldx12, nil, &ldx21, nil, &ldx22, nil, nil, nil, nil, nil, nil, &work_query, &lwork, &info)
		}
		return int(work_query), 0
	} else when is_complex(T) {
		when T == complex64 {
			lapack.cunbdb_(&trans_c, &signs_c, &m, &p, &q, nil, &ldx11, nil, &ldx12, nil, &ldx21, nil, &ldx22, nil, nil, nil, nil, nil, nil, &work_query, &lwork, &info)
		} else when T == complex128 {
			lapack.zunbdb_(&trans_c, &signs_c, &m, &p, &q, nil, &ldx11, nil, &ldx12, nil, &ldx21, nil, &ldx22, nil, nil, nil, nil, nil, nil, &work_query, &lwork, &info)
		}
		return int(real(work_query)), 0
	}
}
// Compute bidiagonal block diagonalization (real variants)
// Simultaneously bidiagonalizes the blocks of a partitioned matrix X
bidi_block_diagonalize_real :: proc(
	X11: ^Matrix($T), // (p x q) upper-left block (overwritten)
	X12: ^Matrix(T), // (p x (m-q)) upper-right block (overwritten)
	X21: ^Matrix(T), // ((m-p) x q) lower-left block (overwritten)
	X22: ^Matrix(T), // ((m-p) x (m-q)) lower-right block (overwritten)
	theta: []T, // Pre-allocated angles (length min(p,q))
	phi: []T, // Pre-allocated angles (length min(p,q))
	taup1: []T, // Pre-allocated tau for P1 (length p)
	taup2: []T, // Pre-allocated tau for P2 (length m-p)
	tauq1: []T, // Pre-allocated tau for Q1 (length q)
	tauq2: []T, // Pre-allocated tau for Q2 (length m-q)
	work: []T, // Pre-allocated workspace
	trans: TransposeMode = .None, // Transpose option
	signs: CSDecompositionSigns = .Positive, // Signs convention
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	// Extract dimensions
	p := X11.rows
	q := X11.cols
	m_p := X21.rows
	m := p + m_p
	m_q := X22.cols

	assert(X12.rows == p && X12.cols == m_q, "X12 has inconsistent dimensions")
	assert(X21.rows == m_p && X21.cols == q, "X21 has inconsistent dimensions")
	assert(X22.rows == m_p && X22.cols == m_q, "X22 has inconsistent dimensions")

	min_pq := min(p, q)

	assert(len(theta) >= int(min_pq), "theta array too small")
	assert(len(phi) >= int(min_pq), "phi array too small")
	assert(len(taup1) >= int(p), "taup1 array too small")
	assert(len(taup2) >= int(m_p), "taup2 array too small")
	assert(len(tauq1) >= int(q), "tauq1 array too small")
	assert(len(tauq2) >= int(m_q), "tauq2 array too small")

	trans_c := cast(u8)trans
	signs_c := cast(u8)signs

	m_val := m
	p_val := p
	q_val := q
	ldx11 := X11.ld
	ldx12 := X12.ld
	ldx21 := X21.ld
	ldx22 := X22.ld
	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.sorbdb_(
			&trans_c,
			&signs_c,
			&m_val,
			&p_val,
			&q_val,
			raw_data(X11.data),
			&ldx11,
			raw_data(X12.data),
			&ldx12,
			raw_data(X21.data),
			&ldx21,
			raw_data(X22.data),
			&ldx22,
			raw_data(theta),
			raw_data(phi),
			raw_data(taup1),
			raw_data(taup2),
			raw_data(tauq1),
			raw_data(tauq2),
			raw_data(work),
			&lwork,
			&info,
		)
	} else when T == f64 {
		lapack.dorbdb_(
			&trans_c,
			&signs_c,
			&m_val,
			&p_val,
			&q_val,
			raw_data(X11.data),
			&ldx11,
			raw_data(X12.data),
			&ldx12,
			raw_data(X21.data),
			&ldx21,
			raw_data(X22.data),
			&ldx22,
			raw_data(theta),
			raw_data(phi),
			raw_data(taup1),
			raw_data(taup2),
			raw_data(tauq1),
			raw_data(tauq2),
			raw_data(work),
			&lwork,
			&info,
		)
	}

	return info, info == 0
}

// Complex bidiagonal block diagonalization (simultaneous bidiagonalization of matrix blocks)
bidi_block_diagonalize_complex :: proc(
	X11: ^Matrix($Cmplx), // (p x q) upper-left block (overwritten)
	X12: ^Matrix(Cmplx), // (p x (m-q)) upper-right block (overwritten)
	X21: ^Matrix(Cmplx), // ((m-p) x q) lower-left block (overwritten)
	X22: ^Matrix(Cmplx), // ((m-p) x (m-q)) lower-right block (overwritten)
	theta: []$Real, // Pre-allocated real angles (length min(p,q))
	phi: []Real, // Pre-allocated real angles (length min(p,q))
	taup1: []Cmplx, // Pre-allocated tau for P1 (length p)
	taup2: []Cmplx, // Pre-allocated tau for P2 (length m-p)
	tauq1: []Cmplx, // Pre-allocated tau for Q1 (length q)
	tauq2: []Cmplx, // Pre-allocated tau for Q2 (length m-q)
	work: []Cmplx, // Pre-allocated complex workspace
	trans: TransposeMode = .None, // Transpose option
	signs: CSDecompositionSigns = .Positive, // Signs convention
) -> (
	info: Info,
	ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
	// Extract dimensions
	p := X11.rows
	q := X11.cols
	m_p := X21.rows
	m := p + m_p
	m_q := X22.cols

	assert(X12.rows == p && X12.cols == m_q, "X12 has inconsistent dimensions")
	assert(X21.rows == m_p && X21.cols == q, "X21 has inconsistent dimensions")
	assert(X22.rows == m_p && X22.cols == m_q, "X22 has inconsistent dimensions")

	min_pq := min(p, q)

	assert(len(theta) >= int(min_pq), "theta array too small")
	assert(len(phi) >= int(min_pq), "phi array too small")
	assert(len(taup1) >= int(p), "taup1 array too small")
	assert(len(taup2) >= int(m_p), "taup2 array too small")
	assert(len(tauq1) >= int(q), "tauq1 array too small")
	assert(len(tauq2) >= int(m_q), "tauq2 array too small")

	trans_c := cast(u8)trans
	signs_c := cast(u8)signs

	m_val := m
	p_val := p
	q_val := q
	ldx11 := X11.ld
	ldx12 := X12.ld
	ldx21 := X21.ld
	ldx22 := X22.ld
	lwork := Blas_Int(len(work))

	when Cmplx == complex64 {
		lapack.cunbdb_(
			&trans_c,
			&signs_c,
			&m_val,
			&p_val,
			&q_val,
			raw_data(X11.data),
			&ldx11,
			raw_data(X12.data),
			&ldx12,
			raw_data(X21.data),
			&ldx21,
			raw_data(X22.data),
			&ldx22,
			raw_data(theta),
			raw_data(phi),
			raw_data(taup1),
			raw_data(taup2),
			raw_data(tauq1),
			raw_data(tauq2),
			raw_data(work),
			&lwork,
			&info,
		)
	} else when Cmplx == complex128 {
		lapack.zunbdb_(
			&trans_c,
			&signs_c,
			&m_val,
			&p_val,
			&q_val,
			raw_data(X11.data),
			&ldx11,
			raw_data(X12.data),
			&ldx12,
			raw_data(X21.data),
			&ldx21,
			raw_data(X22.data),
			&ldx22,
			raw_data(theta),
			raw_data(phi),
			raw_data(taup1),
			raw_data(taup2),
			raw_data(tauq1),
			raw_data(tauq2),
			raw_data(work),
			&lwork,
			&info,
		)
	}

	return info, info == 0
}
