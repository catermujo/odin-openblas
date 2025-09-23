package openblas

import lapack "./f77"
import "base:builtin"
import "core:mem"

// ===================================================================================
// ORTHOGONAL MATRIX MULTIPLICATION (APPLY FACTORIZATIONS TO MATRICES)
// ===================================================================================

// Multiply by orthogonal matrix from bidiagonal reduction proc group
m_multiply_orthogonal_bidiagonal :: proc {
	m_multiply_orthogonal_bidiagonal_f64,
	m_multiply_orthogonal_bidiagonal_f32,
}

// Multiply by orthogonal matrix from Hessenberg reduction proc group
m_multiply_orthogonal_hessenberg :: proc {
	m_multiply_orthogonal_hessenberg_f64,
	m_multiply_orthogonal_hessenberg_f32,
}

// Multiply by orthogonal matrix from LQ factorization proc group
m_multiply_orthogonal_lq :: proc {
	m_multiply_orthogonal_lq_f64,
	m_multiply_orthogonal_lq_f32,
}

// Multiply by orthogonal matrix from QL factorization proc group
m_multiply_orthogonal_ql :: proc {
	m_multiply_orthogonal_ql_f64,
	m_multiply_orthogonal_ql_f32,
}

// Multiply by orthogonal matrix from QR factorization proc group
m_multiply_orthogonal_qr :: proc {
	m_multiply_orthogonal_qr_f64,
	m_multiply_orthogonal_qr_f32,
}

// Multiply by orthogonal matrix from RQ factorization proc group
m_multiply_orthogonal_rq :: proc {
	m_multiply_orthogonal_rq_f64,
	m_multiply_orthogonal_rq_f32,
}

// Multiply by orthogonal matrix from RZ factorization proc group
m_multiply_orthogonal_rz :: proc {
	m_multiply_orthogonal_rz_f64,
	m_multiply_orthogonal_rz_f32,
}

// Multiply by orthogonal matrix from tridiagonal reduction proc group
m_multiply_orthogonal_tridiagonal :: proc {
	m_multiply_orthogonal_tridiagonal_f64,
	m_multiply_orthogonal_tridiagonal_f32,
}

// ===================================================================================
// PARAMETER ENUMS FOR TYPE SAFETY
// ===================================================================================

// Side of multiplication (left or right)
MultiplicationSide :: enum {
	Left, // "L" - Apply Q from the left (Q * C)
	Right, // "R" - Apply Q from the right (C * Q)
}

// Transpose operation
TransposeOperation :: enum {
	NoTranspose, // "N" - Apply Q
	Transpose, // "T" - Apply Q^T
}

// Convert multiplication side to LAPACK character
_side_to_char :: proc(side: MultiplicationSide) -> cstring {
	switch side {
	case .Left:
		return "L"
	case .Right:
		return "R"
	case:
		return "L"
	}
}

// Convert transpose operation to LAPACK character
_transpose_to_char :: proc(trans: TransposeOperation) -> cstring {
	switch trans {
	case .NoTranspose:
		return "N"
	case .Transpose:
		return "T"
	case:
		return "N"
	}
}

// ===================================================================================
// BIDIAGONAL REDUCTION MATRIX MULTIPLICATION
// ===================================================================================

// Multiply by orthogonal matrix from bidiagonal reduction (f64)
// Applies Q or P from DGEBRD to matrix C
m_multiply_orthogonal_bidiagonal_f64 :: proc(
	A: ^Matrix(f64), // Matrix containing factorization from DGEBRD
	tau: []f64, // Scalar factors from DGEBRD
	C: ^Matrix(f64), // Matrix to multiply
	vector_type: BidiagonalVectorType, // Which matrix to apply (P or Q)
	side: MultiplicationSide, // Apply from left or right
	transpose: TransposeOperation, // Apply transpose or not
	k: int, // Number of elementary reflectors
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate inputs
	if len(A.data) == 0 || len(C.data) == 0 {
		panic("Matrices cannot be empty")
	}
	if k > min(A.rows, A.cols) {
		panic("k cannot exceed min(m,n)")
	}

	vect_c := _bidiagonal_vector_to_char(vector_type)
	side_c := _side_to_char(side)
	trans_c := _transpose_to_char(transpose)
	m := Blas_Int(C.rows)
	n := Blas_Int(C.cols)
	k_val := Blas_Int(k)
	lda := Blas_Int(A.ld)
	ldc := Blas_Int(C.ld)

	// Determine required tau size
	matrix_size := side == .Left ? C.rows : C.cols
	expected_tau_size := vector_type == .P ? matrix_size : matrix_size
	if len(tau) < expected_tau_size {
		panic("tau array too small for specified configuration")
	}

	// Query workspace size
	work_query: f64
	lwork := Blas_Int(-1)
	info_val: Info

	lapack.dormbr_(
		vect_c,
		side_c,
		trans_c,
		&m,
		&n,
		&k_val,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(C.data),
		&ldc,
		&work_query,
		&lwork,
		&info_val,
		len(vect_c),
		len(side_c),
		len(trans_c),
	)

	if info_val != 0 {
		return false, info_val
	}

	// Allocate workspace and compute
	lwork = Blas_Int(work_query)
	work := make([]f64, lwork, context.temp_allocator)

	lapack.dormbr_(
		vect_c,
		side_c,
		trans_c,
		&m,
		&n,
		&k_val,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(C.data),
		&ldc,
		raw_data(work),
		&lwork,
		&info_val,
		len(vect_c),
		len(side_c),
		len(trans_c),
	)

	return info_val == 0, info_val
}

// Multiply by orthogonal matrix from bidiagonal reduction (f32)
// Applies Q or P from SGEBRD to matrix C
m_multiply_orthogonal_bidiagonal_f32 :: proc(
	A: ^Matrix(f32), // Matrix containing factorization from SGEBRD
	tau: []f32, // Scalar factors from SGEBRD
	C: ^Matrix(f32), // Matrix to multiply
	vector_type: BidiagonalVectorType, // Which matrix to apply (P or Q)
	side: MultiplicationSide, // Apply from left or right
	transpose: TransposeOperation, // Apply transpose or not
	k: int, // Number of elementary reflectors
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate inputs
	if len(A.data) == 0 || len(C.data) == 0 {
		panic("Matrices cannot be empty")
	}
	if k > min(A.rows, A.cols) {
		panic("k cannot exceed min(m,n)")
	}

	vect_c := _bidiagonal_vector_to_char(vector_type)
	side_c := _side_to_char(side)
	trans_c := _transpose_to_char(transpose)
	m := Blas_Int(C.rows)
	n := Blas_Int(C.cols)
	k_val := Blas_Int(k)
	lda := Blas_Int(A.ld)
	ldc := Blas_Int(C.ld)

	// Determine required tau size
	matrix_size := side == .Left ? C.rows : C.cols
	expected_tau_size := vector_type == .P ? matrix_size : matrix_size
	if len(tau) < expected_tau_size {
		panic("tau array too small for specified configuration")
	}

	// Query workspace size
	work_query: f32
	lwork := Blas_Int(-1)
	info_val: Info

	lapack.sormbr_(
		vect_c,
		side_c,
		trans_c,
		&m,
		&n,
		&k_val,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(C.data),
		&ldc,
		&work_query,
		&lwork,
		&info_val,
		len(vect_c),
		len(side_c),
		len(trans_c),
	)

	if info_val != 0 {
		return false, info_val
	}

	// Allocate workspace and compute
	lwork = Blas_Int(work_query)
	work := make([]f32, lwork, context.temp_allocator)

	lapack.sormbr_(
		vect_c,
		side_c,
		trans_c,
		&m,
		&n,
		&k_val,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(C.data),
		&ldc,
		raw_data(work),
		&lwork,
		&info_val,
		len(vect_c),
		len(side_c),
		len(trans_c),
	)

	return info_val == 0, info_val
}

// ===================================================================================
// HESSENBERG REDUCTION MATRIX MULTIPLICATION
// ===================================================================================

// Multiply by orthogonal matrix from Hessenberg reduction (f64)
// Applies Q from DGEHRD to matrix C
m_multiply_orthogonal_hessenberg_f64 :: proc(
	A: ^Matrix(f64), // Matrix containing factorization from DGEHRD
	tau: []f64, // Scalar factors from DGEHRD
	C: ^Matrix(f64), // Matrix to multiply
	side: MultiplicationSide, // Apply from left or right
	transpose: TransposeOperation, // Apply transpose or not
	ilo, ihi: int, // Balancing range (1-indexed)
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate inputs
	if len(A.data) == 0 || len(C.data) == 0 {
		panic("Matrices cannot be empty")
	}
	if A.rows != A.cols {
		panic("A must be square for Hessenberg reduction")
	}
	if ilo < 1 || ihi > A.rows || ilo > ihi {
		panic("Invalid balancing range")
	}

	side_c := _side_to_char(side)
	trans_c := _transpose_to_char(transpose)
	m := Blas_Int(C.rows)
	n := Blas_Int(C.cols)
	ilo_val := Blas_Int(ilo)
	ihi_val := Blas_Int(ihi)
	lda := Blas_Int(A.ld)
	ldc := Blas_Int(C.ld)

	// Validate tau size
	expected_tau_size := ihi - ilo
	if len(tau) < expected_tau_size {
		panic("tau array too small")
	}

	// Query workspace size
	work_query: f64
	lwork := Blas_Int(-1)
	info_val: Info

	lapack.dormhr_(
		side_c,
		trans_c,
		&m,
		&n,
		&ilo_val,
		&ihi_val,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(C.data),
		&ldc,
		&work_query,
		&lwork,
		&info_val,
		len(side_c),
		len(trans_c),
	)

	if info_val != 0 {
		return false, info_val
	}

	// Allocate workspace and compute
	lwork = Blas_Int(work_query)
	work := make([]f64, lwork, context.temp_allocator)

	lapack.dormhr_(
		side_c,
		trans_c,
		&m,
		&n,
		&ilo_val,
		&ihi_val,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(C.data),
		&ldc,
		raw_data(work),
		&lwork,
		&info_val,
		len(side_c),
		len(trans_c),
	)

	return info_val == 0, info_val
}

// Multiply by orthogonal matrix from Hessenberg reduction (f32)
// Applies Q from SGEHRD to matrix C
m_multiply_orthogonal_hessenberg_f32 :: proc(
	A: ^Matrix(f32), // Matrix containing factorization from SGEHRD
	tau: []f32, // Scalar factors from SGEHRD
	C: ^Matrix(f32), // Matrix to multiply
	side: MultiplicationSide, // Apply from left or right
	transpose: TransposeOperation, // Apply transpose or not
	ilo, ihi: int, // Balancing range (1-indexed)
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate inputs
	if len(A.data) == 0 || len(C.data) == 0 {
		panic("Matrices cannot be empty")
	}
	if A.rows != A.cols {
		panic("A must be square for Hessenberg reduction")
	}
	if ilo < 1 || ihi > A.rows || ilo > ihi {
		panic("Invalid balancing range")
	}

	side_c := _side_to_char(side)
	trans_c := _transpose_to_char(transpose)
	m := Blas_Int(C.rows)
	n := Blas_Int(C.cols)
	ilo_val := Blas_Int(ilo)
	ihi_val := Blas_Int(ihi)
	lda := Blas_Int(A.ld)
	ldc := Blas_Int(C.ld)

	// Validate tau size
	expected_tau_size := ihi - ilo
	if len(tau) < expected_tau_size {
		panic("tau array too small")
	}

	// Query workspace size
	work_query: f32
	lwork := Blas_Int(-1)
	info_val: Info

	lapack.sormhr_(
		side_c,
		trans_c,
		&m,
		&n,
		&ilo_val,
		&ihi_val,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(C.data),
		&ldc,
		&work_query,
		&lwork,
		&info_val,
		len(side_c),
		len(trans_c),
	)

	if info_val != 0 {
		return false, info_val
	}

	// Allocate workspace and compute
	lwork = Blas_Int(work_query)
	work := make([]f32, lwork, context.temp_allocator)

	lapack.sormhr_(
		side_c,
		trans_c,
		&m,
		&n,
		&ilo_val,
		&ihi_val,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(C.data),
		&ldc,
		raw_data(work),
		&lwork,
		&info_val,
		len(side_c),
		len(trans_c),
	)

	return info_val == 0, info_val
}

// ===================================================================================
// LQ FACTORIZATION MATRIX MULTIPLICATION
// ===================================================================================

// Multiply by orthogonal matrix from LQ factorization (f64)
// Applies Q from DGELQF to matrix C
m_multiply_orthogonal_lq_f64 :: proc(
	A: ^Matrix(f64), // Matrix containing factorization from DGELQF
	tau: []f64, // Scalar factors from DGELQF
	C: ^Matrix(f64), // Matrix to multiply
	side: MultiplicationSide, // Apply from left or right
	transpose: TransposeOperation, // Apply transpose or not
	k: int, // Number of elementary reflectors
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate inputs
	if len(A.data) == 0 || len(C.data) == 0 {
		panic("Matrices cannot be empty")
	}
	if k > min(A.rows, A.cols) {
		panic("k cannot exceed min(m,n)")
	}
	if len(tau) < k {
		panic("tau array too small")
	}

	side_c := _side_to_char(side)
	trans_c := _transpose_to_char(transpose)
	m := Blas_Int(C.rows)
	n := Blas_Int(C.cols)
	k_val := Blas_Int(k)
	lda := Blas_Int(A.ld)
	ldc := Blas_Int(C.ld)

	// Query workspace size
	work_query: f64
	lwork := Blas_Int(-1)
	info_val: Info

	lapack.dormlq_(
		side_c,
		trans_c,
		&m,
		&n,
		&k_val,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(C.data),
		&ldc,
		&work_query,
		&lwork,
		&info_val,
		len(side_c),
		len(trans_c),
	)

	if info_val != 0 {
		return false, info_val
	}

	// Allocate workspace and compute
	lwork = Blas_Int(work_query)
	work := make([]f64, lwork, context.temp_allocator)

	lapack.dormlq_(
		side_c,
		trans_c,
		&m,
		&n,
		&k_val,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(C.data),
		&ldc,
		raw_data(work),
		&lwork,
		&info_val,
		len(side_c),
		len(trans_c),
	)

	return info_val == 0, info_val
}

// Multiply by orthogonal matrix from LQ factorization (f32)
// Applies Q from SGELQF to matrix C
m_multiply_orthogonal_lq_f32 :: proc(
	A: ^Matrix(f32), // Matrix containing factorization from SGELQF
	tau: []f32, // Scalar factors from SGELQF
	C: ^Matrix(f32), // Matrix to multiply
	side: MultiplicationSide, // Apply from left or right
	transpose: TransposeOperation, // Apply transpose or not
	k: int, // Number of elementary reflectors
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate inputs
	if len(A.data) == 0 || len(C.data) == 0 {
		panic("Matrices cannot be empty")
	}
	if k > min(A.rows, A.cols) {
		panic("k cannot exceed min(m,n)")
	}
	if len(tau) < k {
		panic("tau array too small")
	}

	side_c := _side_to_char(side)
	trans_c := _transpose_to_char(transpose)
	m := Blas_Int(C.rows)
	n := Blas_Int(C.cols)
	k_val := Blas_Int(k)
	lda := Blas_Int(A.ld)
	ldc := Blas_Int(C.ld)

	// Query workspace size
	work_query: f32
	lwork := Blas_Int(-1)
	info_val: Info

	lapack.sormlq_(
		side_c,
		trans_c,
		&m,
		&n,
		&k_val,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(C.data),
		&ldc,
		&work_query,
		&lwork,
		&info_val,
		len(side_c),
		len(trans_c),
	)

	if info_val != 0 {
		return false, info_val
	}

	// Allocate workspace and compute
	lwork = Blas_Int(work_query)
	work := make([]f32, lwork, context.temp_allocator)

	lapack.sormlq_(
		side_c,
		trans_c,
		&m,
		&n,
		&k_val,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(C.data),
		&ldc,
		raw_data(work),
		&lwork,
		&info_val,
		len(side_c),
		len(trans_c),
	)

	return info_val == 0, info_val
}

// ===================================================================================
// QL FACTORIZATION MATRIX MULTIPLICATION
// ===================================================================================

// Multiply by orthogonal matrix from QL factorization (f64)
// Applies Q from DGEQLF to matrix C
m_multiply_orthogonal_ql_f64 :: proc(
	A: ^Matrix(f64), // Matrix containing factorization from DGEQLF
	tau: []f64, // Scalar factors from DGEQLF
	C: ^Matrix(f64), // Matrix to multiply
	side: MultiplicationSide, // Apply from left or right
	transpose: TransposeOperation, // Apply transpose or not
	k: int, // Number of elementary reflectors
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate inputs
	if len(A.data) == 0 || len(C.data) == 0 {
		panic("Matrices cannot be empty")
	}
	if k > min(A.rows, A.cols) {
		panic("k cannot exceed min(m,n)")
	}
	if len(tau) < k {
		panic("tau array too small")
	}

	side_c := _side_to_char(side)
	trans_c := _transpose_to_char(transpose)
	m := Blas_Int(C.rows)
	n := Blas_Int(C.cols)
	k_val := Blas_Int(k)
	lda := Blas_Int(A.ld)
	ldc := Blas_Int(C.ld)

	// Query workspace size
	work_query: f64
	lwork := Blas_Int(-1)
	info_val: Info

	lapack.dormql_(
		side_c,
		trans_c,
		&m,
		&n,
		&k_val,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(C.data),
		&ldc,
		&work_query,
		&lwork,
		&info_val,
		len(side_c),
		len(trans_c),
	)

	if info_val != 0 {
		return false, info_val
	}

	// Allocate workspace and compute
	lwork = Blas_Int(work_query)
	work := make([]f64, lwork, context.temp_allocator)

	lapack.dormql_(
		side_c,
		trans_c,
		&m,
		&n,
		&k_val,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(C.data),
		&ldc,
		raw_data(work),
		&lwork,
		&info_val,
		len(side_c),
		len(trans_c),
	)

	return info_val == 0, info_val
}

// Multiply by orthogonal matrix from QL factorization (f32)
// Applies Q from SGEQLF to matrix C
m_multiply_orthogonal_ql_f32 :: proc(
	A: ^Matrix(f32), // Matrix containing factorization from SGEQLF
	tau: []f32, // Scalar factors from SGEQLF
	C: ^Matrix(f32), // Matrix to multiply
	side: MultiplicationSide, // Apply from left or right
	transpose: TransposeOperation, // Apply transpose or not
	k: int, // Number of elementary reflectors
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate inputs
	if len(A.data) == 0 || len(C.data) == 0 {
		panic("Matrices cannot be empty")
	}
	if k > min(A.rows, A.cols) {
		panic("k cannot exceed min(m,n)")
	}
	if len(tau) < k {
		panic("tau array too small")
	}

	side_c := _side_to_char(side)
	trans_c := _transpose_to_char(transpose)
	m := Blas_Int(C.rows)
	n := Blas_Int(C.cols)
	k_val := Blas_Int(k)
	lda := Blas_Int(A.ld)
	ldc := Blas_Int(C.ld)

	// Query workspace size
	work_query: f32
	lwork := Blas_Int(-1)
	info_val: Info

	lapack.sormql_(
		side_c,
		trans_c,
		&m,
		&n,
		&k_val,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(C.data),
		&ldc,
		&work_query,
		&lwork,
		&info_val,
		len(side_c),
		len(trans_c),
	)

	if info_val != 0 {
		return false, info_val
	}

	// Allocate workspace and compute
	lwork = Blas_Int(work_query)
	work := make([]f32, lwork, context.temp_allocator)

	lapack.sormql_(
		side_c,
		trans_c,
		&m,
		&n,
		&k_val,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(C.data),
		&ldc,
		raw_data(work),
		&lwork,
		&info_val,
		len(side_c),
		len(trans_c),
	)

	return info_val == 0, info_val
}

// ===================================================================================
// QR FACTORIZATION MATRIX MULTIPLICATION
// ===================================================================================

// Multiply by orthogonal matrix from QR factorization (f64)
// Applies Q from DGEQRF to matrix C
m_multiply_orthogonal_qr_f64 :: proc(
	A: ^Matrix(f64), // Matrix containing factorization from DGEQRF
	tau: []f64, // Scalar factors from DGEQRF
	C: ^Matrix(f64), // Matrix to multiply
	side: MultiplicationSide, // Apply from left or right
	transpose: TransposeOperation, // Apply transpose or not
	k: int, // Number of elementary reflectors
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate inputs
	if len(A.data) == 0 || len(C.data) == 0 {
		panic("Matrices cannot be empty")
	}
	if k > min(A.rows, A.cols) {
		panic("k cannot exceed min(m,n)")
	}
	if len(tau) < k {
		panic("tau array too small")
	}

	side_c := _side_to_char(side)
	trans_c := _transpose_to_char(transpose)
	m := Blas_Int(C.rows)
	n := Blas_Int(C.cols)
	k_val := Blas_Int(k)
	lda := Blas_Int(A.ld)
	ldc := Blas_Int(C.ld)

	// Query workspace size
	work_query: f64
	lwork := Blas_Int(-1)
	info_val: Info

	lapack.dormqr_(
		side_c,
		trans_c,
		&m,
		&n,
		&k_val,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(C.data),
		&ldc,
		&work_query,
		&lwork,
		&info_val,
		len(side_c),
		len(trans_c),
	)

	if info_val != 0 {
		return false, info_val
	}

	// Allocate workspace and compute
	lwork = Blas_Int(work_query)
	work := make([]f64, lwork, context.temp_allocator)

	lapack.dormqr_(
		side_c,
		trans_c,
		&m,
		&n,
		&k_val,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(C.data),
		&ldc,
		raw_data(work),
		&lwork,
		&info_val,
		len(side_c),
		len(trans_c),
	)

	return info_val == 0, info_val
}

// Multiply by orthogonal matrix from QR factorization (f32)
// Applies Q from SGEQRF to matrix C
m_multiply_orthogonal_qr_f32 :: proc(
	A: ^Matrix(f32), // Matrix containing factorization from SGEQRF
	tau: []f32, // Scalar factors from SGEQRF
	C: ^Matrix(f32), // Matrix to multiply
	side: MultiplicationSide, // Apply from left or right
	transpose: TransposeOperation, // Apply transpose or not
	k: int, // Number of elementary reflectors
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate inputs
	if len(A.data) == 0 || len(C.data) == 0 {
		panic("Matrices cannot be empty")
	}
	if k > min(A.rows, A.cols) {
		panic("k cannot exceed min(m,n)")
	}
	if len(tau) < k {
		panic("tau array too small")
	}

	side_c := _side_to_char(side)
	trans_c := _transpose_to_char(transpose)
	m := Blas_Int(C.rows)
	n := Blas_Int(C.cols)
	k_val := Blas_Int(k)
	lda := Blas_Int(A.ld)
	ldc := Blas_Int(C.ld)

	// Query workspace size
	work_query: f32
	lwork := Blas_Int(-1)
	info_val: Info

	lapack.sormqr_(
		side_c,
		trans_c,
		&m,
		&n,
		&k_val,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(C.data),
		&ldc,
		&work_query,
		&lwork,
		&info_val,
		len(side_c),
		len(trans_c),
	)

	if info_val != 0 {
		return false, info_val
	}

	// Allocate workspace and compute
	lwork = Blas_Int(work_query)
	work := make([]f32, lwork, context.temp_allocator)

	lapack.sormqr_(
		side_c,
		trans_c,
		&m,
		&n,
		&k_val,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(C.data),
		&ldc,
		raw_data(work),
		&lwork,
		&info_val,
		len(side_c),
		len(trans_c),
	)

	return info_val == 0, info_val
}

// ===================================================================================
// CONVENIENCE FUNCTIONS FOR COMMON OPERATIONS
// ===================================================================================

// Apply Q from QR factorization to solve least squares problem
apply_qr_to_least_squares :: proc(
	QR: ^Matrix($T), // QR factorization from DGEQRF/SGEQRF
	tau: []T, // Tau from QR factorization
	B: ^Matrix(T), // Right-hand side matrix
	allocator := context.allocator,
) -> (
	success: bool,
) {
	// Apply Q^T to B for least squares solution
	min_dim := min(QR.rows, QR.cols)

	when T == f64 {
		return(
			m_multiply_orthogonal_qr_f64(QR, tau, B, .Left, .Transpose, min_dim, allocator).success \
		)
	} else when T == f32 {
		return(
			m_multiply_orthogonal_qr_f32(QR, tau, B, .Left, .Transpose, min_dim, allocator).success \
		)
	} else {
		panic("Unsupported type for QR least squares")
	}
}

// Apply Q from QR factorization to compute orthogonal projection
apply_qr_orthogonal_projection :: proc(
	QR: ^Matrix($T), // QR factorization from DGEQRF/SGEQRF
	tau: []T, // Tau from QR factorization
	X: ^Matrix(T), // Matrix to project
	allocator := context.allocator,
) -> (
	success: bool,
) {
	// Apply Q * Q^T to X for orthogonal projection
	min_dim := min(QR.rows, QR.cols)

	// First apply Q^T
	when T == f64 {
		success1, _ := m_multiply_orthogonal_qr_f64(
			QR,
			tau,
			X,
			.Left,
			.Transpose,
			min_dim,
			allocator,
		)
		if !success1 do return false

		// Then apply Q
		success2, _ := m_multiply_orthogonal_qr_f64(
			QR,
			tau,
			X,
			.Left,
			.NoTranspose,
			min_dim,
			allocator,
		)
		return success2
	} else when T == f32 {
		success1, _ := m_multiply_orthogonal_qr_f32(
			QR,
			tau,
			X,
			.Left,
			.Transpose,
			min_dim,
			allocator,
		)
		if !success1 do return false

		// Then apply Q
		success2, _ := m_multiply_orthogonal_qr_f32(
			QR,
			tau,
			X,
			.Left,
			.NoTranspose,
			min_dim,
			allocator,
		)
		return success2
	} else {
		panic("Unsupported type for QR orthogonal projection")
	}
}

// Apply Hessenberg transformation for eigenvalue computation step
apply_hessenberg_for_eigenvalues :: proc(
	H: ^Matrix($T), // Hessenberg matrix from DGEHRD/SGEHRD
	tau: []T, // Tau from Hessenberg reduction
	V: ^Matrix(T), // Eigenvector matrix
	ilo, ihi: int, // Balancing range
	allocator := context.allocator,
) -> (
	success: bool,
) {
	when T == f64 {
		return(
			m_multiply_orthogonal_hessenberg_f64(H, tau, V, .Left, .NoTranspose, ilo, ihi, allocator).success \
		)
	} else when T == f32 {
		return(
			m_multiply_orthogonal_hessenberg_f32(H, tau, V, .Left, .NoTranspose, ilo, ihi, allocator).success \
		)
	} else {
		panic("Unsupported type for Hessenberg eigenvalue computation")
	}
}

// Apply bidiagonal transformation for SVD computation
apply_bidiagonal_for_svd :: proc(
	BD: ^Matrix($T), // Bidiagonal matrix from DGEBRD/SGEBRD
	tau_p, tau_q: []T, // Tau arrays for P and Q
	U: ^Matrix(T), // Left singular vectors
	VT: ^Matrix(T), // Right singular vectors (transposed)
	allocator := context.allocator,
) -> (
	success: bool,
) {
	min_dim := min(BD.rows, BD.cols)

	when T == f64 {
		// Apply P to U
		success1, _ := m_multiply_orthogonal_bidiagonal_f64(
			BD,
			tau_p,
			U,
			.P,
			.Left,
			.NoTranspose,
			min_dim,
			allocator,
		)
		if !success1 do return false

		// Apply Q to VT
		success2, _ := m_multiply_orthogonal_bidiagonal_f64(
			BD,
			tau_q,
			VT,
			.Q,
			.Right,
			.NoTranspose,
			min_dim,
			allocator,
		)
		return success2
	} else when T == f32 {
		// Apply P to U
		success1, _ := m_multiply_orthogonal_bidiagonal_f32(
			BD,
			tau_p,
			U,
			.P,
			.Left,
			.NoTranspose,
			min_dim,
			allocator,
		)
		if !success1 do return false

		// Apply Q to VT
		success2, _ := m_multiply_orthogonal_bidiagonal_f32(
			BD,
			tau_q,
			VT,
			.Q,
			.Right,
			.NoTranspose,
			min_dim,
			allocator,
		)
		return success2
	} else {
		panic("Unsupported type for bidiagonal SVD computation")
	}
}

// Transform matrix using LQ factorization
transform_matrix_with_lq :: proc(
	LQ: ^Matrix($T), // LQ factorization from DGELQF/SGELQF
	tau: []T, // Tau from LQ factorization
	C: ^Matrix(T), // Matrix to transform
	from_left := true, // Apply from left or right
	transpose := false, // Apply transpose
	allocator := context.allocator,
) -> (
	success: bool,
) {
	min_dim := min(LQ.rows, LQ.cols)
	side := MultiplicationSide.Left if from_left else MultiplicationSide.Right
	trans := TransposeOperation.Transpose if transpose else TransposeOperation.NoTranspose

	when T == f64 {
		return m_multiply_orthogonal_lq_f64(LQ, tau, C, side, trans, min_dim, allocator).success
	} else when T == f32 {
		return m_multiply_orthogonal_lq_f32(LQ, tau, C, side, trans, min_dim, allocator).success
	} else {
		panic("Unsupported type for LQ transformation")
	}
}

// ===================================================================================
// RQ FACTORIZATION MATRIX MULTIPLICATION
// ===================================================================================

// Multiply by orthogonal matrix from RQ factorization (f64)
// Applies Q from DGERQF to matrix C
m_multiply_orthogonal_rq_f64 :: proc(
	A: ^Matrix(f64), // Matrix containing factorization from DGERQF
	tau: []f64, // Scalar factors from DGERQF
	C: ^Matrix(f64), // Matrix to multiply
	side: MultiplicationSide, // Apply from left or right
	transpose: TransposeOperation, // Apply transpose or not
	k: int, // Number of elementary reflectors
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate inputs
	if len(A.data) == 0 || len(C.data) == 0 {
		panic("Matrices cannot be empty")
	}
	if k > min(A.rows, A.cols) {
		panic("k cannot exceed min(m,n)")
	}
	if len(tau) < k {
		panic("tau array too small")
	}

	side_c := _side_to_char(side)
	trans_c := _transpose_to_char(transpose)
	m := Blas_Int(C.rows)
	n := Blas_Int(C.cols)
	k_val := Blas_Int(k)
	lda := Blas_Int(A.ld)
	ldc := Blas_Int(C.ld)

	// Query workspace size
	work_query: f64
	lwork := Blas_Int(-1)
	info_val: Info

	lapack.dormrq_(
		side_c,
		trans_c,
		&m,
		&n,
		&k_val,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(C.data),
		&ldc,
		&work_query,
		&lwork,
		&info_val,
		len(side_c),
		len(trans_c),
	)

	if info_val != 0 {
		return false, info_val
	}

	// Allocate workspace and compute
	lwork = Blas_Int(work_query)
	work := make([]f64, lwork, context.temp_allocator)

	lapack.dormrq_(
		side_c,
		trans_c,
		&m,
		&n,
		&k_val,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(C.data),
		&ldc,
		raw_data(work),
		&lwork,
		&info_val,
		len(side_c),
		len(trans_c),
	)

	return info_val == 0, info_val
}

// Multiply by orthogonal matrix from RQ factorization (f32)
// Applies Q from SGERQF to matrix C
m_multiply_orthogonal_rq_f32 :: proc(
	A: ^Matrix(f32), // Matrix containing factorization from SGERQF
	tau: []f32, // Scalar factors from SGERQF
	C: ^Matrix(f32), // Matrix to multiply
	side: MultiplicationSide, // Apply from left or right
	transpose: TransposeOperation, // Apply transpose or not
	k: int, // Number of elementary reflectors
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate inputs
	if len(A.data) == 0 || len(C.data) == 0 {
		panic("Matrices cannot be empty")
	}
	if k > min(A.rows, A.cols) {
		panic("k cannot exceed min(m,n)")
	}
	if len(tau) < k {
		panic("tau array too small")
	}

	side_c := _side_to_char(side)
	trans_c := _transpose_to_char(transpose)
	m := Blas_Int(C.rows)
	n := Blas_Int(C.cols)
	k_val := Blas_Int(k)
	lda := Blas_Int(A.ld)
	ldc := Blas_Int(C.ld)

	// Query workspace size
	work_query: f32
	lwork := Blas_Int(-1)
	info_val: Info

	lapack.sormrq_(
		side_c,
		trans_c,
		&m,
		&n,
		&k_val,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(C.data),
		&ldc,
		&work_query,
		&lwork,
		&info_val,
		len(side_c),
		len(trans_c),
	)

	if info_val != 0 {
		return false, info_val
	}

	// Allocate workspace and compute
	lwork = Blas_Int(work_query)
	work := make([]f32, lwork, context.temp_allocator)

	lapack.sormrq_(
		side_c,
		trans_c,
		&m,
		&n,
		&k_val,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(C.data),
		&ldc,
		raw_data(work),
		&lwork,
		&info_val,
		len(side_c),
		len(trans_c),
	)

	return info_val == 0, info_val
}

// ===================================================================================
// RZ FACTORIZATION MATRIX MULTIPLICATION
// ===================================================================================

// Multiply by orthogonal matrix from RZ factorization (f64)
// Applies Q from DGERZ to matrix C (trapezoidal factorization)
m_multiply_orthogonal_rz_f64 :: proc(
	A: ^Matrix(f64), // Matrix containing factorization from DGERZ
	tau: []f64, // Scalar factors from DGERZ
	C: ^Matrix(f64), // Matrix to multiply
	side: MultiplicationSide, // Apply from left or right
	transpose: TransposeOperation, // Apply transpose or not
	k: int, // Number of elementary reflectors
	l: int, // Number of columns in trapezoidal part
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate inputs
	if len(A.data) == 0 || len(C.data) == 0 {
		panic("Matrices cannot be empty")
	}
	if k > A.rows || l > A.cols {
		panic("Invalid k or l parameters")
	}
	if len(tau) < k {
		panic("tau array too small")
	}

	side_c := _side_to_char(side)
	trans_c := _transpose_to_char(transpose)
	m := Blas_Int(C.rows)
	n := Blas_Int(C.cols)
	k_val := Blas_Int(k)
	l_val := Blas_Int(l)
	lda := Blas_Int(A.ld)
	ldc := Blas_Int(C.ld)

	// Query workspace size
	work_query: f64
	lwork := Blas_Int(-1)
	info_val: Info

	lapack.dormrz_(
		side_c,
		trans_c,
		&m,
		&n,
		&k_val,
		&l_val,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(C.data),
		&ldc,
		&work_query,
		&lwork,
		&info_val,
		len(side_c),
		len(trans_c),
	)

	if info_val != 0 {
		return false, info_val
	}

	// Allocate workspace and compute
	lwork = Blas_Int(work_query)
	work := make([]f64, lwork, context.temp_allocator)

	lapack.dormrz_(
		side_c,
		trans_c,
		&m,
		&n,
		&k_val,
		&l_val,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(C.data),
		&ldc,
		raw_data(work),
		&lwork,
		&info_val,
		len(side_c),
		len(trans_c),
	)

	return info_val == 0, info_val
}

// Multiply by orthogonal matrix from RZ factorization (f32)
// Applies Q from SGERZ to matrix C (trapezoidal factorization)
m_multiply_orthogonal_rz_f32 :: proc(
	A: ^Matrix(f32), // Matrix containing factorization from SGERZ
	tau: []f32, // Scalar factors from SGERZ
	C: ^Matrix(f32), // Matrix to multiply
	side: MultiplicationSide, // Apply from left or right
	transpose: TransposeOperation, // Apply transpose or not
	k: int, // Number of elementary reflectors
	l: int, // Number of columns in trapezoidal part
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate inputs
	if len(A.data) == 0 || len(C.data) == 0 {
		panic("Matrices cannot be empty")
	}
	if k > A.rows || l > A.cols {
		panic("Invalid k or l parameters")
	}
	if len(tau) < k {
		panic("tau array too small")
	}

	side_c := _side_to_char(side)
	trans_c := _transpose_to_char(transpose)
	m := Blas_Int(C.rows)
	n := Blas_Int(C.cols)
	k_val := Blas_Int(k)
	l_val := Blas_Int(l)
	lda := Blas_Int(A.ld)
	ldc := Blas_Int(C.ld)

	// Query workspace size
	work_query: f32
	lwork := Blas_Int(-1)
	info_val: Info

	lapack.sormrz_(
		side_c,
		trans_c,
		&m,
		&n,
		&k_val,
		&l_val,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(C.data),
		&ldc,
		&work_query,
		&lwork,
		&info_val,
		len(side_c),
		len(trans_c),
	)

	if info_val != 0 {
		return false, info_val
	}

	// Allocate workspace and compute
	lwork = Blas_Int(work_query)
	work := make([]f32, lwork, context.temp_allocator)

	lapack.sormrz_(
		side_c,
		trans_c,
		&m,
		&n,
		&k_val,
		&l_val,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(C.data),
		&ldc,
		raw_data(work),
		&lwork,
		&info_val,
		len(side_c),
		len(trans_c),
	)

	return info_val == 0, info_val
}

// ===================================================================================
// TRIDIAGONAL REDUCTION MATRIX MULTIPLICATION
// ===================================================================================

// Multiply by orthogonal matrix from tridiagonal reduction (f64)
// Applies Q from DSYTRD to matrix C
m_multiply_orthogonal_tridiagonal_f64 :: proc(
	A: ^Matrix(f64), // Matrix containing factorization from DSYTRD
	tau: []f64, // Scalar factors from DSYTRD
	C: ^Matrix(f64), // Matrix to multiply
	side: MultiplicationSide, // Apply from left or right
	uplo_upper: bool, // Upper or lower triangular storage
	transpose: TransposeOperation, // Apply transpose or not
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate inputs
	if len(A.data) == 0 || len(C.data) == 0 {
		panic("Matrices cannot be empty")
	}
	if A.rows != A.cols {
		panic("A must be square for tridiagonal reduction")
	}
	if len(tau) < A.rows - 1 {
		panic("tau array too small")
	}

	side_c := _side_to_char(side)
	uplo_c := "U" if uplo_upper else "L"
	trans_c := _transpose_to_char(transpose)
	m := Blas_Int(C.rows)
	n := Blas_Int(C.cols)
	lda := Blas_Int(A.ld)
	ldc := Blas_Int(C.ld)

	// Query workspace size
	work_query: f64
	lwork := Blas_Int(-1)
	info_val: Info

	lapack.dormtr_(
		side_c,
		uplo_c,
		trans_c,
		&m,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(C.data),
		&ldc,
		&work_query,
		&lwork,
		&info_val,
		len(side_c),
		len(uplo_c),
		len(trans_c),
	)

	if info_val != 0 {
		return false, info_val
	}

	// Allocate workspace and compute
	lwork = Blas_Int(work_query)
	work := make([]f64, lwork, context.temp_allocator)

	lapack.dormtr_(
		side_c,
		uplo_c,
		trans_c,
		&m,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(C.data),
		&ldc,
		raw_data(work),
		&lwork,
		&info_val,
		len(side_c),
		len(uplo_c),
		len(trans_c),
	)

	return info_val == 0, info_val
}

// Multiply by orthogonal matrix from tridiagonal reduction (f32)
// Applies Q from SSYTRD to matrix C
m_multiply_orthogonal_tridiagonal_f32 :: proc(
	A: ^Matrix(f32), // Matrix containing factorization from SSYTRD
	tau: []f32, // Scalar factors from SSYTRD
	C: ^Matrix(f32), // Matrix to multiply
	side: MultiplicationSide, // Apply from left or right
	uplo_upper: bool, // Upper or lower triangular storage
	transpose: TransposeOperation, // Apply transpose or not
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate inputs
	if len(A.data) == 0 || len(C.data) == 0 {
		panic("Matrices cannot be empty")
	}
	if A.rows != A.cols {
		panic("A must be square for tridiagonal reduction")
	}
	if len(tau) < A.rows - 1 {
		panic("tau array too small")
	}

	side_c := _side_to_char(side)
	uplo_c := "U" if uplo_upper else "L"
	trans_c := _transpose_to_char(transpose)
	m := Blas_Int(C.rows)
	n := Blas_Int(C.cols)
	lda := Blas_Int(A.ld)
	ldc := Blas_Int(C.ld)

	// Query workspace size
	work_query: f32
	lwork := Blas_Int(-1)
	info_val: Info

	lapack.sormtr_(
		side_c,
		uplo_c,
		trans_c,
		&m,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(C.data),
		&ldc,
		&work_query,
		&lwork,
		&info_val,
		len(side_c),
		len(uplo_c),
		len(trans_c),
	)

	if info_val != 0 {
		return false, info_val
	}

	// Allocate workspace and compute
	lwork = Blas_Int(work_query)
	work := make([]f32, lwork, context.temp_allocator)

	lapack.sormtr_(
		side_c,
		uplo_c,
		trans_c,
		&m,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(C.data),
		&ldc,
		raw_data(work),
		&lwork,
		&info_val,
		len(side_c),
		len(uplo_c),
		len(trans_c),
	)

	return info_val == 0, info_val
}

// ===================================================================================
// ADDITIONAL CONVENIENCE FUNCTIONS
// ===================================================================================

// Apply RQ factorization for tall matrices
apply_rq_transformation :: proc(
	RQ: ^Matrix($T), // RQ factorization from DGERQF/SGERQF
	tau: []T, // Tau from RQ factorization
	C: ^Matrix(T), // Matrix to transform
	from_left := true, // Apply from left or right
	transpose := false, // Apply transpose
	allocator := context.allocator,
) -> (
	success: bool,
) {
	min_dim := min(RQ.rows, RQ.cols)
	side := MultiplicationSide.Left if from_left else MultiplicationSide.Right
	trans := TransposeOperation.Transpose if transpose else TransposeOperation.NoTranspose

	when T == f64 {
		return m_multiply_orthogonal_rq_f64(RQ, tau, C, side, trans, min_dim, allocator).success
	} else when T == f32 {
		return m_multiply_orthogonal_rq_f32(RQ, tau, C, side, trans, min_dim, allocator).success
	} else {
		panic("Unsupported type for RQ transformation")
	}
}

// Apply trapezoidal RZ factorization
apply_rz_trapezoidal :: proc(
	RZ: ^Matrix($T), // RZ factorization
	tau: []T, // Tau from RZ factorization
	C: ^Matrix(T), // Matrix to transform
	k, l: int, // RZ parameters
	from_left := true, // Apply from left or right
	transpose := false, // Apply transpose
	allocator := context.allocator,
) -> (
	success: bool,
) {
	side := MultiplicationSide.Left if from_left else MultiplicationSide.Right
	trans := TransposeOperation.Transpose if transpose else TransposeOperation.NoTranspose

	when T == f64 {
		return m_multiply_orthogonal_rz_f64(RZ, tau, C, side, trans, k, l, allocator).success
	} else when T == f32 {
		return m_multiply_orthogonal_rz_f32(RZ, tau, C, side, trans, k, l, allocator).success
	} else {
		panic("Unsupported type for RZ transformation")
	}
}

// Apply tridiagonal transformation for symmetric eigenvalue problems
apply_tridiagonal_for_eigenvalues :: proc(
	T_mat: ^Matrix($T), // Tridiagonal matrix from DSYTRD/SSYTRD
	tau: []T, // Tau from tridiagonal reduction
	V: ^Matrix(T), // Eigenvector matrix
	uplo_upper := true, // Upper or lower triangular
	allocator := context.allocator,
) -> (
	success: bool,
) {
	when T == f64 {
		return(
			m_multiply_orthogonal_tridiagonal_f64(T_mat, tau, V, .Left, uplo_upper, .NoTranspose, allocator).success \
		)
	} else when T == f32 {
		return(
			m_multiply_orthogonal_tridiagonal_f32(T_mat, tau, V, .Left, uplo_upper, .NoTranspose, allocator).success \
		)
	} else {
		panic("Unsupported type for tridiagonal eigenvalue computation")
	}
}
