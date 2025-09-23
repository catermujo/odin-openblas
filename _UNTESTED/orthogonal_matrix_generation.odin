package openblas

import lapack "./f77"
import "base:builtin"
import "core:mem"

// ===================================================================================
// ORTHOGONAL MATRIX GENERATION FROM FACTORIZATIONS
// ===================================================================================

// Generate orthogonal matrix from bidiagonal reduction proc group
m_generate_orthogonal_from_bidiagonal :: proc {
	m_generate_orthogonal_from_bidiagonal_f64,
	m_generate_orthogonal_from_bidiagonal_f32,
}

// Generate orthogonal matrix from Hessenberg reduction proc group
m_generate_orthogonal_from_hessenberg :: proc {
	m_generate_orthogonal_from_hessenberg_f64,
	m_generate_orthogonal_from_hessenberg_f32,
}

// Generate orthogonal matrix from LQ factorization proc group
m_generate_orthogonal_from_lq :: proc {
	m_generate_orthogonal_from_lq_f64,
	m_generate_orthogonal_from_lq_f32,
}

// Generate orthogonal matrix from QL factorization proc group
m_generate_orthogonal_from_ql :: proc {
	m_generate_orthogonal_from_ql_f64,
	m_generate_orthogonal_from_ql_f32,
}

// Generate orthogonal matrix from QR factorization proc group
m_generate_orthogonal_from_qr :: proc {
	m_generate_orthogonal_from_qr_f64,
	m_generate_orthogonal_from_qr_f32,
}

// Generate orthogonal matrix from RQ factorization proc group
m_generate_orthogonal_from_rq :: proc {
	m_generate_orthogonal_from_rq_f64,
	m_generate_orthogonal_from_rq_f32,
}

// Generate orthogonal matrix from tridiagonal reduction proc group
m_generate_orthogonal_from_tridiagonal :: proc {
	m_generate_orthogonal_from_tridiagonal_f64,
	m_generate_orthogonal_from_tridiagonal_f32,
}

// Generate orthogonal matrix from tall-skinny QR proc group
m_generate_orthogonal_from_tsqr :: proc {
	m_generate_orthogonal_from_tsqr_f64,
	m_generate_orthogonal_from_tsqr_f32,
}

// Householder QR column-wise proc group
m_householder_qr_column :: proc {
	m_householder_qr_column_f64,
	m_householder_qr_column_f32,
}

// ===================================================================================
// MATRIX VECTOR TYPES FOR BIDIAGONAL DECOMPOSITION
// ===================================================================================

// Bidiagonal vector type for matrix generation
BidiagonalVectorType :: enum {
	P, // "P" - Generate P (left orthogonal matrix)
	Q, // "Q" - Generate Q (right orthogonal matrix)
}

// Convert bidiagonal vector type to LAPACK character
_bidiagonal_vector_to_char :: proc(vect: BidiagonalVectorType) -> cstring {
	switch vect {
	case .P:
		return "P"
	case .Q:
		return "Q"
	case:
		return "Q"
	}
}

// ===================================================================================
// ORTHOGONAL MATRIX GENERATION FROM BIDIAGONAL REDUCTION
// ===================================================================================

// Generate orthogonal matrix from bidiagonal reduction (f64)
// Generates Q or P from the output of DGEBRD
m_generate_orthogonal_from_bidiagonal_f64 :: proc(
	A: ^Matrix(f64), // Matrix containing the factorization from DGEBRD
	tau: []f64, // Scalar factors from DGEBRD
	vector_type: BidiagonalVectorType, // Which matrix to generate (P or Q)
	k: int, // Number of elementary reflectors
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate inputs
	if len(A.data) == 0 {
		panic("Matrix cannot be empty")
	}
	if k > min(A.rows, A.cols) {
		panic("k cannot exceed min(m,n)")
	}

	vect_c := _bidiagonal_vector_to_char(vector_type)
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	k_val := Blas_Int(k)
	lda := Blas_Int(A.ld)

	// Determine required tau size
	expected_tau_size := vector_type == .P ? n : m
	if len(tau) < expected_tau_size {
		panic("tau array too small for specified vector type")
	}

	// Query workspace size
	work_query: f64
	lwork := Blas_Int(-1)
	info_val: Info

	lapack.dorgbr_(
		vect_c,
		&m,
		&n,
		&k_val,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		&work_query,
		&lwork,
		&info_val,
		len(vect_c),
	)

	if info_val != 0 {
		return false, info_val
	}

	// Allocate workspace and compute
	lwork = Blas_Int(work_query)
	work := make([]f64, lwork, context.temp_allocator)

	lapack.dorgbr_(
		vect_c,
		&m,
		&n,
		&k_val,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(work),
		&lwork,
		&info_val,
		len(vect_c),
	)

	return info_val == 0, info_val
}

// Generate orthogonal matrix from bidiagonal reduction (f32)
// Generates Q or P from the output of SGEBRD
m_generate_orthogonal_from_bidiagonal_f32 :: proc(
	A: ^Matrix(f32), // Matrix containing the factorization from SGEBRD
	tau: []f32, // Scalar factors from SGEBRD
	vector_type: BidiagonalVectorType, // Which matrix to generate (P or Q)
	k: int, // Number of elementary reflectors
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate inputs
	if len(A.data) == 0 {
		panic("Matrix cannot be empty")
	}
	if k > min(A.rows, A.cols) {
		panic("k cannot exceed min(m,n)")
	}

	vect_c := _bidiagonal_vector_to_char(vector_type)
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	k_val := Blas_Int(k)
	lda := Blas_Int(A.ld)

	// Determine required tau size
	expected_tau_size := vector_type == .P ? n : m
	if len(tau) < expected_tau_size {
		panic("tau array too small for specified vector type")
	}

	// Query workspace size
	work_query: f32
	lwork := Blas_Int(-1)
	info_val: Info

	lapack.sorgbr_(
		vect_c,
		&m,
		&n,
		&k_val,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		&work_query,
		&lwork,
		&info_val,
		len(vect_c),
	)

	if info_val != 0 {
		return false, info_val
	}

	// Allocate workspace and compute
	lwork = Blas_Int(work_query)
	work := make([]f32, lwork, context.temp_allocator)

	lapack.sorgbr_(
		vect_c,
		&m,
		&n,
		&k_val,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(work),
		&lwork,
		&info_val,
		len(vect_c),
	)

	return info_val == 0, info_val
}

// ===================================================================================
// ORTHOGONAL MATRIX GENERATION FROM HESSENBERG REDUCTION
// ===================================================================================

// Generate orthogonal matrix from Hessenberg reduction (f64)
// Generates Q from the output of DGEHRD
m_generate_orthogonal_from_hessenberg_f64 :: proc(
	A: ^Matrix(f64), // Matrix containing the factorization from DGEHRD
	tau: []f64, // Scalar factors from DGEHRD
	ilo, ihi: int, // Balancing range (1-indexed)
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate inputs
	if len(A.data) == 0 {
		panic("Matrix cannot be empty")
	}
	if A.rows != A.cols {
		panic("Matrix must be square for Hessenberg reduction")
	}
	if ilo < 1 || ihi > A.rows || ilo > ihi {
		panic("Invalid balancing range")
	}

	n := Blas_Int(A.rows)
	ilo_val := Blas_Int(ilo)
	ihi_val := Blas_Int(ihi)
	lda := Blas_Int(A.ld)

	// Validate tau size
	expected_tau_size := ihi - ilo
	if len(tau) < expected_tau_size {
		panic("tau array too small")
	}

	// Query workspace size
	work_query: f64
	lwork := Blas_Int(-1)
	info_val: Info

	lapack.dorghr_(
		&n,
		&ilo_val,
		&ihi_val,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		&work_query,
		&lwork,
		&info_val,
	)

	if info_val != 0 {
		return false, info_val
	}

	// Allocate workspace and compute
	lwork = Blas_Int(work_query)
	work := make([]f64, lwork, context.temp_allocator)

	lapack.dorghr_(
		&n,
		&ilo_val,
		&ihi_val,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(work),
		&lwork,
		&info_val,
	)

	return info_val == 0, info_val
}

// Generate orthogonal matrix from Hessenberg reduction (f32)
// Generates Q from the output of SGEHRD
m_generate_orthogonal_from_hessenberg_f32 :: proc(
	A: ^Matrix(f32), // Matrix containing the factorization from SGEHRD
	tau: []f32, // Scalar factors from SGEHRD
	ilo, ihi: int, // Balancing range (1-indexed)
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate inputs
	if len(A.data) == 0 {
		panic("Matrix cannot be empty")
	}
	if A.rows != A.cols {
		panic("Matrix must be square for Hessenberg reduction")
	}
	if ilo < 1 || ihi > A.rows || ilo > ihi {
		panic("Invalid balancing range")
	}

	n := Blas_Int(A.rows)
	ilo_val := Blas_Int(ilo)
	ihi_val := Blas_Int(ihi)
	lda := Blas_Int(A.ld)

	// Validate tau size
	expected_tau_size := ihi - ilo
	if len(tau) < expected_tau_size {
		panic("tau array too small")
	}

	// Query workspace size
	work_query: f32
	lwork := Blas_Int(-1)
	info_val: Info

	lapack.sorghr_(
		&n,
		&ilo_val,
		&ihi_val,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		&work_query,
		&lwork,
		&info_val,
	)

	if info_val != 0 {
		return false, info_val
	}

	// Allocate workspace and compute
	lwork = Blas_Int(work_query)
	work := make([]f32, lwork, context.temp_allocator)

	lapack.sorghr_(
		&n,
		&ilo_val,
		&ihi_val,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(work),
		&lwork,
		&info_val,
	)

	return info_val == 0, info_val
}

// ===================================================================================
// ORTHOGONAL MATRIX GENERATION FROM LQ FACTORIZATION
// ===================================================================================

// Generate orthogonal matrix from LQ factorization (f64)
// Generates Q from the output of DGELQF
m_generate_orthogonal_from_lq_f64 :: proc(
	A: ^Matrix(f64), // Matrix containing the factorization from DGELQF
	tau: []f64, // Scalar factors from DGELQF
	k: int, // Number of elementary reflectors
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate inputs
	if len(A.data) == 0 {
		panic("Matrix cannot be empty")
	}
	if k > min(A.rows, A.cols) {
		panic("k cannot exceed min(m,n)")
	}
	if len(tau) < k {
		panic("tau array too small")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	k_val := Blas_Int(k)
	lda := Blas_Int(A.ld)

	// Query workspace size
	work_query: f64
	lwork := Blas_Int(-1)
	info_val: Info

	lapack.dorglq_(
		&m,
		&n,
		&k_val,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		&work_query,
		&lwork,
		&info_val,
	)

	if info_val != 0 {
		return false, info_val
	}

	// Allocate workspace and compute
	lwork = Blas_Int(work_query)
	work := make([]f64, lwork, context.temp_allocator)

	lapack.dorglq_(
		&m,
		&n,
		&k_val,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(work),
		&lwork,
		&info_val,
	)

	return info_val == 0, info_val
}

// Generate orthogonal matrix from LQ factorization (f32)
// Generates Q from the output of SGELQF
m_generate_orthogonal_from_lq_f32 :: proc(
	A: ^Matrix(f32), // Matrix containing the factorization from SGELQF
	tau: []f32, // Scalar factors from SGELQF
	k: int, // Number of elementary reflectors
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate inputs
	if len(A.data) == 0 {
		panic("Matrix cannot be empty")
	}
	if k > min(A.rows, A.cols) {
		panic("k cannot exceed min(m,n)")
	}
	if len(tau) < k {
		panic("tau array too small")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	k_val := Blas_Int(k)
	lda := Blas_Int(A.ld)

	// Query workspace size
	work_query: f32
	lwork := Blas_Int(-1)
	info_val: Info

	lapack.sorglq_(
		&m,
		&n,
		&k_val,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		&work_query,
		&lwork,
		&info_val,
	)

	if info_val != 0 {
		return false, info_val
	}

	// Allocate workspace and compute
	lwork = Blas_Int(work_query)
	work := make([]f32, lwork, context.temp_allocator)

	lapack.sorglq_(
		&m,
		&n,
		&k_val,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(work),
		&lwork,
		&info_val,
	)

	return info_val == 0, info_val
}

// ===================================================================================
// ORTHOGONAL MATRIX GENERATION FROM QL FACTORIZATION
// ===================================================================================

// Generate orthogonal matrix from QL factorization (f64)
// Generates Q from the output of DGEQLF
m_generate_orthogonal_from_ql_f64 :: proc(
	A: ^Matrix(f64), // Matrix containing the factorization from DGEQLF
	tau: []f64, // Scalar factors from DGEQLF
	k: int, // Number of elementary reflectors
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate inputs
	if len(A.data) == 0 {
		panic("Matrix cannot be empty")
	}
	if k > min(A.rows, A.cols) {
		panic("k cannot exceed min(m,n)")
	}
	if len(tau) < k {
		panic("tau array too small")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	k_val := Blas_Int(k)
	lda := Blas_Int(A.ld)

	// Query workspace size
	work_query: f64
	lwork := Blas_Int(-1)
	info_val: Info

	lapack.dorgql_(
		&m,
		&n,
		&k_val,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		&work_query,
		&lwork,
		&info_val,
	)

	if info_val != 0 {
		return false, info_val
	}

	// Allocate workspace and compute
	lwork = Blas_Int(work_query)
	work := make([]f64, lwork, context.temp_allocator)

	lapack.dorgql_(
		&m,
		&n,
		&k_val,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(work),
		&lwork,
		&info_val,
	)

	return info_val == 0, info_val
}

// Generate orthogonal matrix from QL factorization (f32)
// Generates Q from the output of SGEQLF
m_generate_orthogonal_from_ql_f32 :: proc(
	A: ^Matrix(f32), // Matrix containing the factorization from SGEQLF
	tau: []f32, // Scalar factors from SGEQLF
	k: int, // Number of elementary reflectors
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate inputs
	if len(A.data) == 0 {
		panic("Matrix cannot be empty")
	}
	if k > min(A.rows, A.cols) {
		panic("k cannot exceed min(m,n)")
	}
	if len(tau) < k {
		panic("tau array too small")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	k_val := Blas_Int(k)
	lda := Blas_Int(A.ld)

	// Query workspace size
	work_query: f32
	lwork := Blas_Int(-1)
	info_val: Info

	lapack.sorgql_(
		&m,
		&n,
		&k_val,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		&work_query,
		&lwork,
		&info_val,
	)

	if info_val != 0 {
		return false, info_val
	}

	// Allocate workspace and compute
	lwork = Blas_Int(work_query)
	work := make([]f32, lwork, context.temp_allocator)

	lapack.sorgql_(
		&m,
		&n,
		&k_val,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(work),
		&lwork,
		&info_val,
	)

	return info_val == 0, info_val
}

// ===================================================================================
// ORTHOGONAL MATRIX GENERATION FROM QR FACTORIZATION
// ===================================================================================

// Generate orthogonal matrix from QR factorization (f64)
// Generates Q from the output of DGEQRF
m_generate_orthogonal_from_qr_f64 :: proc(
	A: ^Matrix(f64), // Matrix containing the factorization from DGEQRF
	tau: []f64, // Scalar factors from DGEQRF
	k: int, // Number of elementary reflectors
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate inputs
	if len(A.data) == 0 {
		panic("Matrix cannot be empty")
	}
	if k > min(A.rows, A.cols) {
		panic("k cannot exceed min(m,n)")
	}
	if len(tau) < k {
		panic("tau array too small")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	k_val := Blas_Int(k)
	lda := Blas_Int(A.ld)

	// Query workspace size
	work_query: f64
	lwork := Blas_Int(-1)
	info_val: Info

	lapack.dorgqr_(
		&m,
		&n,
		&k_val,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		&work_query,
		&lwork,
		&info_val,
	)

	if info_val != 0 {
		return false, info_val
	}

	// Allocate workspace and compute
	lwork = Blas_Int(work_query)
	work := make([]f64, lwork, context.temp_allocator)

	lapack.dorgqr_(
		&m,
		&n,
		&k_val,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(work),
		&lwork,
		&info_val,
	)

	return info_val == 0, info_val
}

// Generate orthogonal matrix from QR factorization (f32)
// Generates Q from the output of SGEQRF
m_generate_orthogonal_from_qr_f32 :: proc(
	A: ^Matrix(f32), // Matrix containing the factorization from SGEQRF
	tau: []f32, // Scalar factors from SGEQRF
	k: int, // Number of elementary reflectors
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate inputs
	if len(A.data) == 0 {
		panic("Matrix cannot be empty")
	}
	if k > min(A.rows, A.cols) {
		panic("k cannot exceed min(m,n)")
	}
	if len(tau) < k {
		panic("tau array too small")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	k_val := Blas_Int(k)
	lda := Blas_Int(A.ld)

	// Query workspace size
	work_query: f32
	lwork := Blas_Int(-1)
	info_val: Info

	lapack.sorgqr_(
		&m,
		&n,
		&k_val,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		&work_query,
		&lwork,
		&info_val,
	)

	if info_val != 0 {
		return false, info_val
	}

	// Allocate workspace and compute
	lwork = Blas_Int(work_query)
	work := make([]f32, lwork, context.temp_allocator)

	lapack.sorgqr_(
		&m,
		&n,
		&k_val,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(work),
		&lwork,
		&info_val,
	)

	return info_val == 0, info_val
}

// ===================================================================================
// CONVENIENCE FUNCTIONS
// ===================================================================================

// Generate complete orthogonal Q matrix from QR factorization
generate_full_q_from_qr :: proc(
	A: ^Matrix($T),
	tau: []T,
	allocator := context.allocator,
) -> (
	Q: Matrix(T),
	success: bool,
) {
	// Create a copy for Q generation
	Q = make_matrix(T, A.rows, A.rows, .General, allocator)

	// Copy the factorization data to Q
	min_dim := min(A.rows, A.cols)
	for j in 0 ..< min_dim {
		for i in j ..< A.rows {
			matrix_set(&Q, i, j, matrix_get(A, i, j))
		}
	}

	// Set remaining columns to identity
	for j in min_dim ..< A.rows {
		matrix_set(&Q, j, j, T(1))
	}

	when T == f64 {
		success, _ := m_generate_orthogonal_from_qr_f64(&Q, tau, min_dim, allocator)
		return Q, success
	} else when T == f32 {
		success, _ := m_generate_orthogonal_from_qr_f32(&Q, tau, min_dim, allocator)
		return Q, success
	} else {
		panic("Unsupported type for QR orthogonal generation")
	}
}

// Generate thin Q matrix from QR factorization (economy size)
generate_thin_q_from_qr :: proc(
	A: ^Matrix($T),
	tau: []T,
	allocator := context.allocator,
) -> (
	Q: Matrix(T),
	success: bool,
) {
	min_dim := min(A.rows, A.cols)

	// Create economy-size Q matrix
	Q = make_matrix(T, A.rows, min_dim, .General, allocator)

	// Copy the factorization data to Q
	for j in 0 ..< min_dim {
		for i in j ..< A.rows {
			matrix_set(&Q, i, j, matrix_get(A, i, j))
		}
	}

	when T == f64 {
		success, _ := m_generate_orthogonal_from_qr_f64(&Q, tau, min_dim, allocator)
		return Q, success
	} else when T == f32 {
		success, _ := m_generate_orthogonal_from_qr_f32(&Q, tau, min_dim, allocator)
		return Q, success
	} else {
		panic("Unsupported type for QR orthogonal generation")
	}
}

// Generate orthogonal matrix for eigenvalue computation workflow
generate_orthogonal_for_eigenvalues :: proc(
	A: ^Matrix($T),
	tau: []T,
	ilo, ihi: int,
	allocator := context.allocator,
) -> (
	success: bool,
) {
	when T == f64 {
		return m_generate_orthogonal_from_hessenberg_f64(A, tau, ilo, ihi, allocator).success
	} else when T == f32 {
		return m_generate_orthogonal_from_hessenberg_f32(A, tau, ilo, ihi, allocator).success
	} else {
		panic("Unsupported type for Hessenberg orthogonal generation")
	}
}

// Generate both P and Q matrices from bidiagonal reduction
generate_bidiagonal_matrices :: proc(
	A: ^Matrix($T),
	tau_p, tau_q: []T,
	allocator := context.allocator,
) -> (
	P, Q: Matrix(T),
	success: bool,
) {
	min_dim := min(A.rows, A.cols)

	// Create copies for P and Q generation
	P = make_matrix(T, A.rows, A.rows, .General, allocator)
	Q = make_matrix(T, A.cols, A.cols, .General, allocator)

	// Copy original matrix data to both
	copy_matrix_data(A, &P)
	copy_matrix_data(A, &Q)

	when T == f64 {
		p_success, _ := m_generate_orthogonal_from_bidiagonal_f64(
			&P,
			tau_p,
			.P,
			min_dim,
			allocator,
		)
		q_success, _ := m_generate_orthogonal_from_bidiagonal_f64(
			&Q,
			tau_q,
			.Q,
			min_dim,
			allocator,
		)
		return P, Q, p_success && q_success
	} else when T == f32 {
		p_success, _ := m_generate_orthogonal_from_bidiagonal_f32(
			&P,
			tau_p,
			.P,
			min_dim,
			allocator,
		)
		q_success, _ := m_generate_orthogonal_from_bidiagonal_f32(
			&Q,
			tau_q,
			.Q,
			min_dim,
			allocator,
		)
		return P, Q, p_success && q_success
	} else {
		panic("Unsupported type for bidiagonal matrix generation")
	}
}

// Generate orthogonal matrix from LQ with proper sizing
generate_orthogonal_lq_proper_size :: proc(
	A: ^Matrix($T),
	tau: []T,
	full_size := false,
	allocator := context.allocator,
) -> (
	Q: Matrix(T),
	success: bool,
) {
	min_dim := min(A.rows, A.cols)

	if full_size {
		// Generate full-size Q matrix (cols x cols)
		Q = make_matrix(T, A.cols, A.cols, .General, allocator)
		// Copy LQ factorization data
		for j in 0 ..< min_dim {
			for i in 0 ..= j {
				matrix_set(&Q, i, j, matrix_get(A, i, j))
			}
		}
		// Set remaining part to identity
		for i in min_dim ..< A.cols {
			matrix_set(&Q, i, i, T(1))
		}
	} else {
		// Generate economy-size Q matrix
		Q = make_matrix(T, min_dim, A.cols, .General, allocator)
		// Copy LQ factorization data
		for j in 0 ..< A.cols {
			for i in 0 ..< min(min_dim, j + 1) {
				matrix_set(&Q, i, j, matrix_get(A, i, j))
			}
		}
	}

	when T == f64 {
		success, _ := m_generate_orthogonal_from_lq_f64(&Q, tau, min_dim, allocator)
		return Q, success
	} else when T == f32 {
		success, _ := m_generate_orthogonal_from_lq_f32(&Q, tau, min_dim, allocator)
		return Q, success
	} else {
		panic("Unsupported type for LQ orthogonal generation")
	}
}

// ===================================================================================
// ORTHOGONAL MATRIX GENERATION FROM RQ FACTORIZATION
// ===================================================================================

// Generate orthogonal matrix from RQ factorization (f64)
// Generates Q from the output of DGERQF
m_generate_orthogonal_from_rq_f64 :: proc(
	A: ^Matrix(f64), // Matrix containing the factorization from DGERQF
	tau: []f64, // Scalar factors from DGERQF
	k: int, // Number of elementary reflectors
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate inputs
	if len(A.data) == 0 {
		panic("Matrix cannot be empty")
	}
	if k > min(A.rows, A.cols) {
		panic("k cannot exceed min(m,n)")
	}
	if len(tau) < k {
		panic("tau array too small")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	k_val := Blas_Int(k)
	lda := Blas_Int(A.ld)

	// Query workspace size
	work_query: f64
	lwork := Blas_Int(-1)
	info_val: Info

	lapack.dorgrq_(
		&m,
		&n,
		&k_val,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		&work_query,
		&lwork,
		&info_val,
	)

	if info_val != 0 {
		return false, info_val
	}

	// Allocate workspace and compute
	lwork = Blas_Int(work_query)
	work := make([]f64, lwork, context.temp_allocator)

	lapack.dorgrq_(
		&m,
		&n,
		&k_val,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(work),
		&lwork,
		&info_val,
	)

	return info_val == 0, info_val
}

// Generate orthogonal matrix from RQ factorization (f32)
// Generates Q from the output of SGERQF
m_generate_orthogonal_from_rq_f32 :: proc(
	A: ^Matrix(f32), // Matrix containing the factorization from SGERQF
	tau: []f32, // Scalar factors from SGERQF
	k: int, // Number of elementary reflectors
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate inputs
	if len(A.data) == 0 {
		panic("Matrix cannot be empty")
	}
	if k > min(A.rows, A.cols) {
		panic("k cannot exceed min(m,n)")
	}
	if len(tau) < k {
		panic("tau array too small")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	k_val := Blas_Int(k)
	lda := Blas_Int(A.ld)

	// Query workspace size
	work_query: f32
	lwork := Blas_Int(-1)
	info_val: Info

	lapack.sorgrq_(
		&m,
		&n,
		&k_val,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		&work_query,
		&lwork,
		&info_val,
	)

	if info_val != 0 {
		return false, info_val
	}

	// Allocate workspace and compute
	lwork = Blas_Int(work_query)
	work := make([]f32, lwork, context.temp_allocator)

	lapack.sorgrq_(
		&m,
		&n,
		&k_val,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(work),
		&lwork,
		&info_val,
	)

	return info_val == 0, info_val
}

// ===================================================================================
// ORTHOGONAL MATRIX GENERATION FROM TRIDIAGONAL REDUCTION
// ===================================================================================

// Generate orthogonal matrix from tridiagonal reduction (f64)
// Generates Q from the output of DSYTRD
m_generate_orthogonal_from_tridiagonal_f64 :: proc(
	A: ^Matrix(f64), // Matrix containing the factorization from DSYTRD
	tau: []f64, // Scalar factors from DSYTRD
	uplo_upper := true, // Upper or lower triangular storage
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate inputs
	if len(A.data) == 0 {
		panic("Matrix cannot be empty")
	}
	if A.rows != A.cols {
		panic("Matrix must be square for tridiagonal reduction")
	}
	if len(tau) < A.rows - 1 {
		panic("tau array too small")
	}

	uplo_c := "U" if uplo_upper else "L"
	n := Blas_Int(A.rows)
	lda := Blas_Int(A.ld)

	// Query workspace size
	work_query: f64
	lwork := Blas_Int(-1)
	info_val: Info

	lapack.dorgtr_(
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		&work_query,
		&lwork,
		&info_val,
		len(uplo_c),
	)

	if info_val != 0 {
		return false, info_val
	}

	// Allocate workspace and compute
	lwork = Blas_Int(work_query)
	work := make([]f64, lwork, context.temp_allocator)

	lapack.dorgtr_(
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(work),
		&lwork,
		&info_val,
		len(uplo_c),
	)

	return info_val == 0, info_val
}

// Generate orthogonal matrix from tridiagonal reduction (f32)
// Generates Q from the output of SSYTRD
m_generate_orthogonal_from_tridiagonal_f32 :: proc(
	A: ^Matrix(f32), // Matrix containing the factorization from SSYTRD
	tau: []f32, // Scalar factors from SSYTRD
	uplo_upper := true, // Upper or lower triangular storage
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate inputs
	if len(A.data) == 0 {
		panic("Matrix cannot be empty")
	}
	if A.rows != A.cols {
		panic("Matrix must be square for tridiagonal reduction")
	}
	if len(tau) < A.rows - 1 {
		panic("tau array too small")
	}

	uplo_c := "U" if uplo_upper else "L"
	n := Blas_Int(A.rows)
	lda := Blas_Int(A.ld)

	// Query workspace size
	work_query: f32
	lwork := Blas_Int(-1)
	info_val: Info

	lapack.sorgtr_(
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		&work_query,
		&lwork,
		&info_val,
		len(uplo_c),
	)

	if info_val != 0 {
		return false, info_val
	}

	// Allocate workspace and compute
	lwork = Blas_Int(work_query)
	work := make([]f32, lwork, context.temp_allocator)

	lapack.sorgtr_(
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(work),
		&lwork,
		&info_val,
		len(uplo_c),
	)

	return info_val == 0, info_val
}

// ===================================================================================
// TALL-SKINNY QR ORTHOGONAL MATRIX GENERATION
// ===================================================================================

// Generate orthogonal matrix from tall-skinny QR (f64)
// Generates Q from the output of DGEQR (tall-skinny QR with row-wise storage)
m_generate_orthogonal_from_tsqr_f64 :: proc(
	A: ^Matrix(f64), // Matrix containing the factorization
	T: ^Matrix(f64), // T matrix from TSQR
	mb, nb: int, // Block sizes
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate inputs
	if len(A.data) == 0 || len(T.data) == 0 {
		panic("Matrices cannot be empty")
	}
	if mb <= 0 || nb <= 0 {
		panic("Block sizes must be positive")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	mb_val := Blas_Int(mb)
	nb_val := Blas_Int(nb)
	lda := Blas_Int(A.ld)
	ldt := Blas_Int(T.ld)

	// Query workspace size
	work_query: f64
	lwork := Blas_Int(-1)
	info_val: Info

	lapack.dorgtsqr_row_(
		&m,
		&n,
		&mb_val,
		&nb_val,
		raw_data(A.data),
		&lda,
		raw_data(T.data),
		&ldt,
		&work_query,
		&lwork,
		&info_val,
	)

	if info_val != 0 {
		return false, info_val
	}

	// Allocate workspace and compute
	lwork = Blas_Int(work_query)
	work := make([]f64, lwork, context.temp_allocator)

	lapack.dorgtsqr_row_(
		&m,
		&n,
		&mb_val,
		&nb_val,
		raw_data(A.data),
		&lda,
		raw_data(T.data),
		&ldt,
		raw_data(work),
		&lwork,
		&info_val,
	)

	return info_val == 0, info_val
}

// Generate orthogonal matrix from tall-skinny QR (f32)
// Generates Q from the output of SGEQR (tall-skinny QR with row-wise storage)
m_generate_orthogonal_from_tsqr_f32 :: proc(
	A: ^Matrix(f32), // Matrix containing the factorization
	T: ^Matrix(f32), // T matrix from TSQR
	mb, nb: int, // Block sizes
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate inputs
	if len(A.data) == 0 || len(T.data) == 0 {
		panic("Matrices cannot be empty")
	}
	if mb <= 0 || nb <= 0 {
		panic("Block sizes must be positive")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	mb_val := Blas_Int(mb)
	nb_val := Blas_Int(nb)
	lda := Blas_Int(A.ld)
	ldt := Blas_Int(T.ld)

	// Query workspace size
	work_query: f32
	lwork := Blas_Int(-1)
	info_val: Info

	lapack.sorgtsqr_row_(
		&m,
		&n,
		&mb_val,
		&nb_val,
		raw_data(A.data),
		&lda,
		raw_data(T.data),
		&ldt,
		&work_query,
		&lwork,
		&info_val,
	)

	if info_val != 0 {
		return false, info_val
	}

	// Allocate workspace and compute
	lwork = Blas_Int(work_query)
	work := make([]f32, lwork, context.temp_allocator)

	lapack.sorgtsqr_row_(
		&m,
		&n,
		&mb_val,
		&nb_val,
		raw_data(A.data),
		&lda,
		raw_data(T.data),
		&ldt,
		raw_data(work),
		&lwork,
		&info_val,
	)

	return info_val == 0, info_val
}

// ===================================================================================
// HOUSEHOLDER QR COLUMN-WISE OPERATIONS
// ===================================================================================

// Householder QR column-wise (f64)
// Computes QR factorization using Householder reflectors column-by-column
m_householder_qr_column_f64 :: proc(
	A: ^Matrix(f64), // Input matrix, overwritten with QR factorization
	T: ^Matrix(f64), // T matrix for block representation
	D: []f64, // Diagonal scaling factors
	nb: int, // Block size
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate inputs
	if len(A.data) == 0 || len(T.data) == 0 {
		panic("Matrices cannot be empty")
	}
	if nb <= 0 {
		panic("Block size must be positive")
	}
	if len(D) < min(A.rows, A.cols) {
		panic("D array too small")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	nb_val := Blas_Int(nb)
	lda := Blas_Int(A.ld)
	ldt := Blas_Int(T.ld)

	info_val: Info

	lapack.dorhr_col_(
		&m,
		&n,
		&nb_val,
		raw_data(A.data),
		&lda,
		raw_data(T.data),
		&ldt,
		raw_data(D),
		&info_val,
	)

	return info_val == 0, info_val
}

// Householder QR column-wise (f32)
// Computes QR factorization using Householder reflectors column-by-column
m_householder_qr_column_f32 :: proc(
	A: ^Matrix(f32), // Input matrix, overwritten with QR factorization
	T: ^Matrix(f32), // T matrix for block representation
	D: []f32, // Diagonal scaling factors
	nb: int, // Block size
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate inputs
	if len(A.data) == 0 || len(T.data) == 0 {
		panic("Matrices cannot be empty")
	}
	if nb <= 0 {
		panic("Block size must be positive")
	}
	if len(D) < min(A.rows, A.cols) {
		panic("D array too small")
	}

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	nb_val := Blas_Int(nb)
	lda := Blas_Int(A.ld)
	ldt := Blas_Int(T.ld)

	info_val: Info

	lapack.sorhr_col_(
		&m,
		&n,
		&nb_val,
		raw_data(A.data),
		&lda,
		raw_data(T.data),
		&ldt,
		raw_data(D),
		&info_val,
	)

	return info_val == 0, info_val
}

// ===================================================================================
// ADVANCED CONVENIENCE FUNCTIONS FOR NEW ALGORITHMS
// ===================================================================================

// Generate orthogonal matrix from RQ with proper orientation
generate_orthogonal_rq_proper_orientation :: proc(
	A: ^Matrix($T),
	tau: []T,
	full_size := false,
	allocator := context.allocator,
) -> (
	Q: Matrix(T),
	success: bool,
) {
	min_dim := min(A.rows, A.cols)

	if full_size {
		// Generate full-size Q matrix (rows x rows)
		Q = make_matrix(T, A.rows, A.rows, .General, allocator)
		// Copy RQ factorization data from bottom-right
		offset_row := A.rows - min_dim
		offset_col := A.cols - min_dim
		for j in 0 ..< min_dim {
			for i in max(0, j - offset_row) ..< A.rows {
				matrix_set(&Q, i, j + offset_col, matrix_get(A, i, j + offset_col))
			}
		}
		// Set remaining part to identity
		for i in 0 ..< offset_row {
			matrix_set(&Q, i, i, T(1))
		}
	} else {
		// Generate economy-size Q matrix
		Q = make_matrix(T, A.rows, min_dim, .General, allocator)
		// Copy RQ factorization data
		for j in 0 ..< min_dim {
			for i in max(0, A.rows - min_dim + j) ..< A.rows {
				matrix_set(&Q, i, j, matrix_get(A, i, A.cols - min_dim + j))
			}
		}
	}

	when T == f64 {
		success, _ := m_generate_orthogonal_from_rq_f64(&Q, tau, min_dim, allocator)
		return Q, success
	} else when T == f32 {
		success, _ := m_generate_orthogonal_from_rq_f32(&Q, tau, min_dim, allocator)
		return Q, success
	} else {
		panic("Unsupported type for RQ orthogonal generation")
	}
}

// Generate orthogonal matrix for symmetric eigenvalue computation
generate_orthogonal_for_symmetric_eigenvalues :: proc(
	A: ^Matrix($T),
	tau: []T,
	uplo_upper := true,
	allocator := context.allocator,
) -> (
	success: bool,
) {
	when T == f64 {
		return m_generate_orthogonal_from_tridiagonal_f64(A, tau, uplo_upper, allocator).success
	} else when T == f32 {
		return m_generate_orthogonal_from_tridiagonal_f32(A, tau, uplo_upper, allocator).success
	} else {
		panic("Unsupported type for tridiagonal orthogonal generation")
	}
}

// High-performance QR for tall-skinny matrices
perform_tsqr_with_orthogonal_generation :: proc(
	A: ^Matrix($T),
	mb, nb: int,
	allocator := context.allocator,
) -> (
	Q: Matrix(T),
	T_matrix: Matrix(T),
	success: bool,
) {
	// Create T matrix for block representation
	t_size := nb * min(A.rows, A.cols)
	T_matrix = make_matrix(T, nb, min(A.rows, A.cols), .General, allocator)

	// Create Q matrix copy
	Q = make_matrix(T, A.rows, A.cols, .General, allocator)
	copy_matrix_data(A, &Q)

	when T == f64 {
		success, _ := m_generate_orthogonal_from_tsqr_f64(&Q, &T_matrix, mb, nb, allocator)
		return Q, T_matrix, success
	} else when T == f32 {
		success, _ := m_generate_orthogonal_from_tsqr_f32(&Q, &T_matrix, mb, nb, allocator)
		return Q, T_matrix, success
	} else {
		panic("Unsupported type for TSQR orthogonal generation")
	}
}

// Column-wise Householder QR with complete output
perform_column_householder_qr :: proc(
	A: ^Matrix($T),
	nb: int,
	allocator := context.allocator,
) -> (
	T_matrix: Matrix(T),
	D: []T,
	success: bool,
) {
	// Create T matrix for block representation
	T_matrix = make_matrix(T, nb, min(A.rows, A.cols), .General, allocator)

	// Create diagonal scaling array
	D = make([]T, min(A.rows, A.cols), allocator)

	when T == f64 {
		success, _ := m_householder_qr_column_f64(A, &T_matrix, D, nb, allocator)
		return T_matrix, D, success
	} else when T == f32 {
		success, _ := m_householder_qr_column_f32(A, &T_matrix, D, nb, allocator)
		return T_matrix, D, success
	} else {
		panic("Unsupported type for column Householder QR")
	}
}
