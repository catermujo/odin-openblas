package openblas

import lapack "./f77"

// ==============================================================================
// Advanced LQ Factorization Functions
// ==============================================================================
// This file contains block and unblocked LQ factorization algorithms with
// multiplication routines, providing high-performance QR decomposition variants
// optimized for modern cache hierarchies and parallel execution.

// LQ factorization result structure
LQFactorizationResult :: struct($T: typeid) {
	factorization_successful: bool,
	factorized_A:             Matrix(T), // A matrix containing L factor in lower triangle
	factorized_B:             Matrix(T), // B matrix (if provided)
	reflector_matrix_T:       Matrix(T), // T matrix containing reflector information
	block_size:               int, // Block size used in factorization
}

// LQ multiplication result structure
LQMultiplicationResult :: struct($T: typeid) {
	multiplication_successful: bool,
	result_A:                  Matrix(T), // Result matrix A
	result_B:                  Matrix(T), // Result matrix B
}

// ==============================================================================
// Block LQ Factorization Functions (tplqt)
// ==============================================================================

// Low-level block LQ factorization functions (ctplqt, dtplqt, stplqt, ztplqt)
ctplqt :: proc(
	m: ^Blas_Int,
	n: ^Blas_Int,
	l: ^Blas_Int,
	mb: ^Blas_Int,
	A: ^complex64,
	lda: ^Blas_Int,
	B: ^complex64,
	ldb: ^Blas_Int,
	T: ^complex64,
	ldt: ^Blas_Int,
	work: ^complex64,
	info: ^Info,
) {
	ctplqt_(m, n, l, mb, A, lda, B, ldb, T, ldt, work, info)
}

dtplqt :: proc(
	m: ^Blas_Int,
	n: ^Blas_Int,
	l: ^Blas_Int,
	mb: ^Blas_Int,
	A: ^f64,
	lda: ^Blas_Int,
	B: ^f64,
	ldb: ^Blas_Int,
	T: ^f64,
	ldt: ^Blas_Int,
	work: ^f64,
	info: ^Info,
) {
	dtplqt_(m, n, l, mb, A, lda, B, ldb, T, ldt, work, info)
}

stplqt :: proc(
	m: ^Blas_Int,
	n: ^Blas_Int,
	l: ^Blas_Int,
	mb: ^Blas_Int,
	A: ^f32,
	lda: ^Blas_Int,
	B: ^f32,
	ldb: ^Blas_Int,
	T: ^f32,
	ldt: ^Blas_Int,
	work: ^f32,
	info: ^Info,
) {
	stplqt_(m, n, l, mb, A, lda, B, ldb, T, ldt, work, info)
}

ztplqt :: proc(
	m: ^Blas_Int,
	n: ^Blas_Int,
	l: ^Blas_Int,
	mb: ^Blas_Int,
	A: ^complex128,
	lda: ^Blas_Int,
	B: ^complex128,
	ldb: ^Blas_Int,
	T: ^complex128,
	ldt: ^Blas_Int,
	work: ^complex128,
	info: ^Info,
) {
	ztplqt_(m, n, l, mb, A, lda, B, ldb, T, ldt, work, info)
}

// High-level block LQ factorization wrapper functions
block_lq_factorization_complex64 :: proc(
	A: Matrix(complex64),
	B: Matrix(complex64) = {},
	l: int,
	block_size: int = 32,
	allocator := context.allocator,
) -> (
	result: LQFactorizationResult(complex64),
	err: LapackError,
) {

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	l_int := Blas_Int(l)
	mb := Blas_Int(block_size)
	lda := Blas_Int(A.rows)

	// Copy A matrix for factorization
	a_data := make([]complex64, A.rows * A.cols, allocator) or_return
	copy(a_data, A.data[:A.rows * A.cols])
	factorized_A := Matrix(complex64) {
		data = a_data,
		rows = A.rows,
		cols = A.cols,
	}

	// Handle B matrix
	b_data: []complex64 = nil
	factorized_B: Matrix(complex64)
	ldb: Blas_Int = 1

	if B.data != nil {
		b_data = make([]complex64, B.rows * B.cols, allocator) or_return
		copy(b_data, B.data[:B.rows * B.cols])
		factorized_B = Matrix(complex64) {
			data = b_data,
			rows = B.rows,
			cols = B.cols,
		}
		ldb = Blas_Int(B.rows)
	}

	// Allocate T matrix for reflector information
	t_rows := min(int(mb), int(m))
	t_cols := int(n)
	t_data := make([]complex64, t_rows * t_cols, allocator) or_return
	reflector_T := Matrix(complex64) {
		data = t_data,
		rows = t_rows,
		cols = t_cols,
	}
	ldt := Blas_Int(t_rows)

	// Allocate workspace
	work_size := max(int(mb * n), int(mb * mb))
	work := make([]complex64, work_size, allocator) or_return

	info: Blas_Int
	b_ptr := raw_data(b_data) if b_data != nil else nil

	ctplqt(
		&m,
		&n,
		&l_int,
		&mb,
		raw_data(a_data),
		&lda,
		b_ptr,
		&ldb,
		raw_data(t_data),
		&ldt,
		raw_data(work),
		&info,
	)

	delete(work, allocator)

	if info != 0 {
		delete(a_data, allocator)
		if b_data != nil do delete(b_data, allocator)
		delete(t_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.factorization_successful = true
	result.factorized_A = factorized_A
	result.factorized_B = factorized_B
	result.reflector_matrix_T = reflector_T
	result.block_size = block_size
	return
}

block_lq_factorization_float64 :: proc(
	A: Matrix(f64),
	B: Matrix(f64) = {},
	l: int,
	block_size: int = 32,
	allocator := context.allocator,
) -> (
	result: LQFactorizationResult(f64),
	err: LapackError,
) {

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	l_int := Blas_Int(l)
	mb := Blas_Int(block_size)
	lda := Blas_Int(A.rows)

	// Copy A matrix for factorization
	a_data := make([]f64, A.rows * A.cols, allocator) or_return
	copy(a_data, A.data[:A.rows * A.cols])
	factorized_A := Matrix(f64) {
		data = a_data,
		rows = A.rows,
		cols = A.cols,
	}

	// Handle B matrix
	b_data: []f64 = nil
	factorized_B: Matrix(f64)
	ldb: Blas_Int = 1

	if B.data != nil {
		b_data = make([]f64, B.rows * B.cols, allocator) or_return
		copy(b_data, B.data[:B.rows * B.cols])
		factorized_B = Matrix(f64) {
			data = b_data,
			rows = B.rows,
			cols = B.cols,
		}
		ldb = Blas_Int(B.rows)
	}

	// Allocate T matrix for reflector information
	t_rows := min(int(mb), int(m))
	t_cols := int(n)
	t_data := make([]f64, t_rows * t_cols, allocator) or_return
	reflector_T := Matrix(f64) {
		data = t_data,
		rows = t_rows,
		cols = t_cols,
	}
	ldt := Blas_Int(t_rows)

	// Allocate workspace
	work_size := max(int(mb * n), int(mb * mb))
	work := make([]f64, work_size, allocator) or_return

	info: Blas_Int
	b_ptr := raw_data(b_data) if b_data != nil else nil

	dtplqt(
		&m,
		&n,
		&l_int,
		&mb,
		raw_data(a_data),
		&lda,
		b_ptr,
		&ldb,
		raw_data(t_data),
		&ldt,
		raw_data(work),
		&info,
	)

	delete(work, allocator)

	if info != 0 {
		delete(a_data, allocator)
		if b_data != nil do delete(b_data, allocator)
		delete(t_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.factorization_successful = true
	result.factorized_A = factorized_A
	result.factorized_B = factorized_B
	result.reflector_matrix_T = reflector_T
	result.block_size = block_size
	return
}

block_lq_factorization_float32 :: proc(
	A: Matrix(f32),
	B: Matrix(f32) = {},
	l: int,
	block_size: int = 32,
	allocator := context.allocator,
) -> (
	result: LQFactorizationResult(f32),
	err: LapackError,
) {

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	l_int := Blas_Int(l)
	mb := Blas_Int(block_size)
	lda := Blas_Int(A.rows)

	// Copy A matrix for factorization
	a_data := make([]f32, A.rows * A.cols, allocator) or_return
	copy(a_data, A.data[:A.rows * A.cols])
	factorized_A := Matrix(f32) {
		data = a_data,
		rows = A.rows,
		cols = A.cols,
	}

	// Handle B matrix
	b_data: []f32 = nil
	factorized_B: Matrix(f32)
	ldb: Blas_Int = 1

	if B.data != nil {
		b_data = make([]f32, B.rows * B.cols, allocator) or_return
		copy(b_data, B.data[:B.rows * B.cols])
		factorized_B = Matrix(f32) {
			data = b_data,
			rows = B.rows,
			cols = B.cols,
		}
		ldb = Blas_Int(B.rows)
	}

	// Allocate T matrix for reflector information
	t_rows := min(int(mb), int(m))
	t_cols := int(n)
	t_data := make([]f32, t_rows * t_cols, allocator) or_return
	reflector_T := Matrix(f32) {
		data = t_data,
		rows = t_rows,
		cols = t_cols,
	}
	ldt := Blas_Int(t_rows)

	// Allocate workspace
	work_size := max(int(mb * n), int(mb * mb))
	work := make([]f32, work_size, allocator) or_return

	info: Blas_Int
	b_ptr := raw_data(b_data) if b_data != nil else nil

	stplqt(
		&m,
		&n,
		&l_int,
		&mb,
		raw_data(a_data),
		&lda,
		b_ptr,
		&ldb,
		raw_data(t_data),
		&ldt,
		raw_data(work),
		&info,
	)

	delete(work, allocator)

	if info != 0 {
		delete(a_data, allocator)
		if b_data != nil do delete(b_data, allocator)
		delete(t_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.factorization_successful = true
	result.factorized_A = factorized_A
	result.factorized_B = factorized_B
	result.reflector_matrix_T = reflector_T
	result.block_size = block_size
	return
}

block_lq_factorization_complex128 :: proc(
	A: Matrix(complex128),
	B: Matrix(complex128) = {},
	l: int,
	block_size: int = 32,
	allocator := context.allocator,
) -> (
	result: LQFactorizationResult(complex128),
	err: LapackError,
) {

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	l_int := Blas_Int(l)
	mb := Blas_Int(block_size)
	lda := Blas_Int(A.rows)

	// Copy A matrix for factorization
	a_data := make([]complex128, A.rows * A.cols, allocator) or_return
	copy(a_data, A.data[:A.rows * A.cols])
	factorized_A := Matrix(complex128) {
		data = a_data,
		rows = A.rows,
		cols = A.cols,
	}

	// Handle B matrix
	b_data: []complex128 = nil
	factorized_B: Matrix(complex128)
	ldb: Blas_Int = 1

	if B.data != nil {
		b_data = make([]complex128, B.rows * B.cols, allocator) or_return
		copy(b_data, B.data[:B.rows * B.cols])
		factorized_B = Matrix(complex128) {
			data = b_data,
			rows = B.rows,
			cols = B.cols,
		}
		ldb = Blas_Int(B.rows)
	}

	// Allocate T matrix for reflector information
	t_rows := min(int(mb), int(m))
	t_cols := int(n)
	t_data := make([]complex128, t_rows * t_cols, allocator) or_return
	reflector_T := Matrix(complex128) {
		data = t_data,
		rows = t_rows,
		cols = t_cols,
	}
	ldt := Blas_Int(t_rows)

	// Allocate workspace
	work_size := max(int(mb * n), int(mb * mb))
	work := make([]complex128, work_size, allocator) or_return

	info: Blas_Int
	b_ptr := raw_data(b_data) if b_data != nil else nil

	ztplqt(
		&m,
		&n,
		&l_int,
		&mb,
		raw_data(a_data),
		&lda,
		b_ptr,
		&ldb,
		raw_data(t_data),
		&ldt,
		raw_data(work),
		&info,
	)

	delete(work, allocator)

	if info != 0 {
		delete(a_data, allocator)
		if b_data != nil do delete(b_data, allocator)
		delete(t_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.factorization_successful = true
	result.factorized_A = factorized_A
	result.factorized_B = factorized_B
	result.reflector_matrix_T = reflector_T
	result.block_size = block_size
	return
}

// Generic block LQ factorization function
block_lq_factorization :: proc {
	block_lq_factorization_complex64,
	block_lq_factorization_float64,
	block_lq_factorization_float32,
	block_lq_factorization_complex128,
}

// ==============================================================================
// Unblocked LQ Factorization Functions (tplqt2)
// ==============================================================================

// Low-level unblocked LQ factorization functions (ctplqt2, dtplqt2, stplqt2, ztplqt2)
ctplqt2 :: proc(
	m: ^Blas_Int,
	n: ^Blas_Int,
	l: ^Blas_Int,
	A: ^complex64,
	lda: ^Blas_Int,
	B: ^complex64,
	ldb: ^Blas_Int,
	T: ^complex64,
	ldt: ^Blas_Int,
	info: ^Info,
) {
	ctplqt2_(m, n, l, A, lda, B, ldb, T, ldt, info)
}

dtplqt2 :: proc(
	m: ^Blas_Int,
	n: ^Blas_Int,
	l: ^Blas_Int,
	A: ^f64,
	lda: ^Blas_Int,
	B: ^f64,
	ldb: ^Blas_Int,
	T: ^f64,
	ldt: ^Blas_Int,
	info: ^Info,
) {
	dtplqt2_(m, n, l, A, lda, B, ldb, T, ldt, info)
}

stplqt2 :: proc(
	m: ^Blas_Int,
	n: ^Blas_Int,
	l: ^Blas_Int,
	A: ^f32,
	lda: ^Blas_Int,
	B: ^f32,
	ldb: ^Blas_Int,
	T: ^f32,
	ldt: ^Blas_Int,
	info: ^Info,
) {
	stplqt2_(m, n, l, A, lda, B, ldb, T, ldt, info)
}

ztplqt2 :: proc(
	m: ^Blas_Int,
	n: ^Blas_Int,
	l: ^Blas_Int,
	A: ^complex128,
	lda: ^Blas_Int,
	B: ^complex128,
	ldb: ^Blas_Int,
	T: ^complex128,
	ldt: ^Blas_Int,
	info: ^Info,
) {
	ztplqt2_(m, n, l, A, lda, B, ldb, T, ldt, info)
}

// High-level unblocked LQ factorization wrapper function (example for complex64)
unblocked_lq_factorization_complex64 :: proc(
	A: Matrix(complex64),
	B: Matrix(complex64) = {},
	l: int,
	allocator := context.allocator,
) -> (
	result: LQFactorizationResult(complex64),
	err: LapackError,
) {

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	l_int := Blas_Int(l)
	lda := Blas_Int(A.rows)

	// Copy A matrix for factorization
	a_data := make([]complex64, A.rows * A.cols, allocator) or_return
	copy(a_data, A.data[:A.rows * A.cols])
	factorized_A := Matrix(complex64) {
		data = a_data,
		rows = A.rows,
		cols = A.cols,
	}

	// Handle B matrix
	b_data: []complex64 = nil
	factorized_B: Matrix(complex64)
	ldb: Blas_Int = 1

	if B.data != nil {
		b_data = make([]complex64, B.rows * B.cols, allocator) or_return
		copy(b_data, B.data[:B.rows * B.cols])
		factorized_B = Matrix(complex64) {
			data = b_data,
			rows = B.rows,
			cols = B.cols,
		}
		ldb = Blas_Int(B.rows)
	}

	// Allocate T matrix for reflector information
	t_rows := int(m)
	t_cols := int(n)
	t_data := make([]complex64, t_rows * t_cols, allocator) or_return
	reflector_T := Matrix(complex64) {
		data = t_data,
		rows = t_rows,
		cols = t_cols,
	}
	ldt := Blas_Int(t_rows)

	info: Blas_Int
	b_ptr := raw_data(b_data) if b_data != nil else nil

	ctplqt2(&m, &n, &l_int, raw_data(a_data), &lda, b_ptr, &ldb, raw_data(t_data), &ldt, &info)

	if info != 0 {
		delete(a_data, allocator)
		if b_data != nil do delete(b_data, allocator)
		delete(t_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.factorization_successful = true
	result.factorized_A = factorized_A
	result.factorized_B = factorized_B
	result.reflector_matrix_T = reflector_T
	result.block_size = 1 // Unblocked
	return
}

// ==============================================================================
// LQ Multiplication Functions (tpmlqt)
// ==============================================================================

// Low-level LQ multiplication functions (ctpmlqt, dtpmlqt, stpmlqt, ztpmlqt)
ctpmlqt :: proc(
	side: cstring,
	trans: cstring,
	m: ^Blas_Int,
	n: ^Blas_Int,
	k: ^Blas_Int,
	l: ^Blas_Int,
	mb: ^Blas_Int,
	V: ^complex64,
	ldv: ^Blas_Int,
	T: ^complex64,
	ldt: ^Blas_Int,
	A: ^complex64,
	lda: ^Blas_Int,
	B: ^complex64,
	ldb: ^Blas_Int,
	work: ^complex64,
	info: ^Info,
) {
	ctpmlqt_(
		side,
		trans,
		m,
		n,
		k,
		l,
		mb,
		V,
		ldv,
		T,
		ldt,
		A,
		lda,
		B,
		ldb,
		work,
		info,
		len(side),
		len(trans),
	)
}

dtpmlqt :: proc(
	side: cstring,
	trans: cstring,
	m: ^Blas_Int,
	n: ^Blas_Int,
	k: ^Blas_Int,
	l: ^Blas_Int,
	mb: ^Blas_Int,
	V: ^f64,
	ldv: ^Blas_Int,
	T: ^f64,
	ldt: ^Blas_Int,
	A: ^f64,
	lda: ^Blas_Int,
	B: ^f64,
	ldb: ^Blas_Int,
	work: ^f64,
	info: ^Info,
) {
	dtpmlqt_(
		side,
		trans,
		m,
		n,
		k,
		l,
		mb,
		V,
		ldv,
		T,
		ldt,
		A,
		lda,
		B,
		ldb,
		work,
		info,
		len(side),
		len(trans),
	)
}

stpmlqt :: proc(
	side: cstring,
	trans: cstring,
	m: ^Blas_Int,
	n: ^Blas_Int,
	k: ^Blas_Int,
	l: ^Blas_Int,
	mb: ^Blas_Int,
	V: ^f32,
	ldv: ^Blas_Int,
	T: ^f32,
	ldt: ^Blas_Int,
	A: ^f32,
	lda: ^Blas_Int,
	B: ^f32,
	ldb: ^Blas_Int,
	work: ^f32,
	info: ^Info,
) {
	stpmlqt_(
		side,
		trans,
		m,
		n,
		k,
		l,
		mb,
		V,
		ldv,
		T,
		ldt,
		A,
		lda,
		B,
		ldb,
		work,
		info,
		len(side),
		len(trans),
	)
}

ztpmlqt :: proc(
	side: cstring,
	trans: cstring,
	m: ^Blas_Int,
	n: ^Blas_Int,
	k: ^Blas_Int,
	l: ^Blas_Int,
	mb: ^Blas_Int,
	V: ^complex128,
	ldv: ^Blas_Int,
	T: ^complex128,
	ldt: ^Blas_Int,
	A: ^complex128,
	lda: ^Blas_Int,
	B: ^complex128,
	ldb: ^Blas_Int,
	work: ^complex128,
	info: ^Info,
) {
	ztpmlqt_(
		side,
		trans,
		m,
		n,
		k,
		l,
		mb,
		V,
		ldv,
		T,
		ldt,
		A,
		lda,
		B,
		ldb,
		work,
		info,
		len(side),
		len(trans),
	)
}

// High-level LQ multiplication wrapper function (example for complex64)
multiply_lq_complex64 :: proc(
	V: Matrix(complex64),
	T: Matrix(complex64),
	A: Matrix(complex64),
	B: Matrix(complex64) = {},
	l: int,
	block_size: int = 32,
	side: MatrixSide = .Left,
	trans: MatrixTranspose = .None,
	allocator := context.allocator,
) -> (
	result: LQMultiplicationResult(complex64),
	err: LapackError,
) {

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	k := Blas_Int(V.cols)
	l_int := Blas_Int(l)
	mb := Blas_Int(block_size)
	ldv := Blas_Int(V.rows)
	ldt := Blas_Int(T.rows)
	lda := Blas_Int(A.rows)

	// Copy A matrix for multiplication
	a_data := make([]complex64, A.rows * A.cols, allocator) or_return
	copy(a_data, A.data[:A.rows * A.cols])
	result_A := Matrix(complex64) {
		data = a_data,
		rows = A.rows,
		cols = A.cols,
	}

	// Handle B matrix
	b_data: []complex64 = nil
	result_B: Matrix(complex64)
	ldb: Blas_Int = 1

	if B.data != nil {
		b_data = make([]complex64, B.rows * B.cols, allocator) or_return
		copy(b_data, B.data[:B.rows * B.cols])
		result_B = Matrix(complex64) {
			data = b_data,
			rows = B.rows,
			cols = B.cols,
		}
		ldb = Blas_Int(B.rows)
	}

	// Allocate workspace
	work_size := max(int(mb * n), int(mb * m))
	work := make([]complex64, work_size, allocator) or_return

	info: Blas_Int
	side_str := matrix_side_to_cstring(side)
	trans_str := matrix_transpose_to_cstring(trans)
	b_ptr := raw_data(b_data) if b_data != nil else nil

	ctpmlqt(
		side_str,
		trans_str,
		&m,
		&n,
		&k,
		&l_int,
		&mb,
		raw_data(V.data),
		&ldv,
		raw_data(T.data),
		&ldt,
		raw_data(a_data),
		&lda,
		b_ptr,
		&ldb,
		raw_data(work),
		&info,
	)

	delete(work, allocator)

	if info != 0 {
		delete(a_data, allocator)
		if b_data != nil do delete(b_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.multiplication_successful = true
	result.result_A = result_A
	result.result_B = result_B
	return
}

// ==============================================================================
// Convenience Overloads
// ==============================================================================

// Block LQ factorization overloads
tplqt :: proc {
	ctplqt,
	dtplqt,
	stplqt,
	ztplqt,
}

// Unblocked LQ factorization overloads
tplqt2 :: proc {
	ctplqt2,
	dtplqt2,
	stplqt2,
	ztplqt2,
}

// LQ multiplication overloads
tpmlqt :: proc {
	ctpmlqt,
	dtpmlqt,
	stpmlqt,
	ztpmlqt,
}

// ==============================================================================
// QR Multiplication with Compact WY Representation Functions (tpmqrt)
// ==============================================================================

// QR multiplication result structure
QRMultiplicationResult :: struct($T: typeid) {
	multiplication_successful: bool,
	result_A:                  Matrix(T), // Result matrix A
	result_B:                  Matrix(T), // Result matrix B
}

// Low-level QR multiplication functions (ctpmqrt, dtpmqrt, stpmqrt, ztpmqrt)
ctpmqrt :: proc(
	side: cstring,
	trans: cstring,
	m: ^Blas_Int,
	n: ^Blas_Int,
	k: ^Blas_Int,
	l: ^Blas_Int,
	nb: ^Blas_Int,
	V: ^complex64,
	ldv: ^Blas_Int,
	T: ^complex64,
	ldt: ^Blas_Int,
	A: ^complex64,
	lda: ^Blas_Int,
	B: ^complex64,
	ldb: ^Blas_Int,
	work: ^complex64,
	info: ^Info,
) {
	ctpmqrt_(
		side,
		trans,
		m,
		n,
		k,
		l,
		nb,
		V,
		ldv,
		T,
		ldt,
		A,
		lda,
		B,
		ldb,
		work,
		info,
		len(side),
		len(trans),
	)
}

dtpmqrt :: proc(
	side: cstring,
	trans: cstring,
	m: ^Blas_Int,
	n: ^Blas_Int,
	k: ^Blas_Int,
	l: ^Blas_Int,
	nb: ^Blas_Int,
	V: ^f64,
	ldv: ^Blas_Int,
	T: ^f64,
	ldt: ^Blas_Int,
	A: ^f64,
	lda: ^Blas_Int,
	B: ^f64,
	ldb: ^Blas_Int,
	work: ^f64,
	info: ^Info,
) {
	dtpmqrt_(
		side,
		trans,
		m,
		n,
		k,
		l,
		nb,
		V,
		ldv,
		T,
		ldt,
		A,
		lda,
		B,
		ldb,
		work,
		info,
		len(side),
		len(trans),
	)
}

stpmqrt :: proc(
	side: cstring,
	trans: cstring,
	m: ^Blas_Int,
	n: ^Blas_Int,
	k: ^Blas_Int,
	l: ^Blas_Int,
	nb: ^Blas_Int,
	V: ^f32,
	ldv: ^Blas_Int,
	T: ^f32,
	ldt: ^Blas_Int,
	A: ^f32,
	lda: ^Blas_Int,
	B: ^f32,
	ldb: ^Blas_Int,
	work: ^f32,
	info: ^Info,
) {
	stpmqrt_(
		side,
		trans,
		m,
		n,
		k,
		l,
		nb,
		V,
		ldv,
		T,
		ldt,
		A,
		lda,
		B,
		ldb,
		work,
		info,
		len(side),
		len(trans),
	)
}

ztpmqrt :: proc(
	side: cstring,
	trans: cstring,
	m: ^Blas_Int,
	n: ^Blas_Int,
	k: ^Blas_Int,
	l: ^Blas_Int,
	nb: ^Blas_Int,
	V: ^complex128,
	ldv: ^Blas_Int,
	T: ^complex128,
	ldt: ^Blas_Int,
	A: ^complex128,
	lda: ^Blas_Int,
	B: ^complex128,
	ldb: ^Blas_Int,
	work: ^complex128,
	info: ^Info,
) {
	ztpmqrt_(
		side,
		trans,
		m,
		n,
		k,
		l,
		nb,
		V,
		ldv,
		T,
		ldt,
		A,
		lda,
		B,
		ldb,
		work,
		info,
		len(side),
		len(trans),
	)
}

// High-level QR multiplication wrapper function
multiply_qr_compact_complex64 :: proc(
	V: Matrix(complex64),
	T: Matrix(complex64),
	A: Matrix(complex64),
	B: Matrix(complex64) = {},
	l: int,
	block_size: int = 32,
	side: MatrixSide = .Left,
	trans: MatrixTranspose = .None,
	allocator := context.allocator,
) -> (
	result: QRMultiplicationResult(complex64),
	err: LapackError,
) {

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	k := Blas_Int(V.cols)
	l_int := Blas_Int(l)
	nb := Blas_Int(block_size)
	ldv := Blas_Int(V.rows)
	ldt := Blas_Int(T.rows)
	lda := Blas_Int(A.rows)

	// Copy A matrix for multiplication
	a_data := make([]complex64, A.rows * A.cols, allocator) or_return
	copy(a_data, A.data[:A.rows * A.cols])
	result_A := Matrix(complex64) {
		data = a_data,
		rows = A.rows,
		cols = A.cols,
	}

	// Handle B matrix
	b_data: []complex64 = nil
	result_B: Matrix(complex64)
	ldb: Blas_Int = 1

	if B.data != nil {
		b_data = make([]complex64, B.rows * B.cols, allocator) or_return
		copy(b_data, B.data[:B.rows * B.cols])
		result_B = Matrix(complex64) {
			data = b_data,
			rows = B.rows,
			cols = B.cols,
		}
		ldb = Blas_Int(B.rows)
	}

	// Allocate workspace
	work_size := max(int(nb * n), int(nb * m))
	work := make([]complex64, work_size, allocator) or_return

	info: Blas_Int
	side_str := matrix_side_to_cstring(side)
	trans_str := matrix_transpose_to_cstring(trans)
	b_ptr := raw_data(b_data) if b_data != nil else nil

	ctpmqrt(
		side_str,
		trans_str,
		&m,
		&n,
		&k,
		&l_int,
		&nb,
		raw_data(V.data),
		&ldv,
		raw_data(T.data),
		&ldt,
		raw_data(a_data),
		&lda,
		b_ptr,
		&ldb,
		raw_data(work),
		&info,
	)

	delete(work, allocator)

	if info != 0 {
		delete(a_data, allocator)
		if b_data != nil do delete(b_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.multiplication_successful = true
	result.result_A = result_A
	result.result_B = result_B
	return
}

// ==============================================================================
// Block QR Factorization Functions (tpqrt)
// ==============================================================================

// QR factorization result structure
QRFactorizationResult :: struct($T: typeid) {
	factorization_successful: bool,
	factorized_A:             Matrix(T), // A matrix containing R factor in upper triangle
	factorized_B:             Matrix(T), // B matrix (if provided)
	reflector_matrix_T:       Matrix(T), // T matrix containing reflector information
	block_size:               int, // Block size used in factorization
}

// Low-level block QR factorization functions (ctpqrt, dtpqrt, stpqrt, ztpqrt)
ctpqrt :: proc(
	m: ^Blas_Int,
	n: ^Blas_Int,
	l: ^Blas_Int,
	nb: ^Blas_Int,
	A: ^complex64,
	lda: ^Blas_Int,
	B: ^complex64,
	ldb: ^Blas_Int,
	T: ^complex64,
	ldt: ^Blas_Int,
	work: ^complex64,
	info: ^Info,
) {
	ctpqrt_(m, n, l, nb, A, lda, B, ldb, T, ldt, work, info)
}

dtpqrt :: proc(
	m: ^Blas_Int,
	n: ^Blas_Int,
	l: ^Blas_Int,
	nb: ^Blas_Int,
	A: ^f64,
	lda: ^Blas_Int,
	B: ^f64,
	ldb: ^Blas_Int,
	T: ^f64,
	ldt: ^Blas_Int,
	work: ^f64,
	info: ^Info,
) {
	dtpqrt_(m, n, l, nb, A, lda, B, ldb, T, ldt, work, info)
}

stpqrt :: proc(
	m: ^Blas_Int,
	n: ^Blas_Int,
	l: ^Blas_Int,
	nb: ^Blas_Int,
	A: ^f32,
	lda: ^Blas_Int,
	B: ^f32,
	ldb: ^Blas_Int,
	T: ^f32,
	ldt: ^Blas_Int,
	work: ^f32,
	info: ^Info,
) {
	stpqrt_(m, n, l, nb, A, lda, B, ldb, T, ldt, work, info)
}

ztpqrt :: proc(
	m: ^Blas_Int,
	n: ^Blas_Int,
	l: ^Blas_Int,
	nb: ^Blas_Int,
	A: ^complex128,
	lda: ^Blas_Int,
	B: ^complex128,
	ldb: ^Blas_Int,
	T: ^complex128,
	ldt: ^Blas_Int,
	work: ^complex128,
	info: ^Info,
) {
	ztpqrt_(m, n, l, nb, A, lda, B, ldb, T, ldt, work, info)
}

// High-level block QR factorization wrapper function
block_qr_factorization_complex64 :: proc(
	A: Matrix(complex64),
	B: Matrix(complex64) = {},
	l: int,
	block_size: int = 32,
	allocator := context.allocator,
) -> (
	result: QRFactorizationResult(complex64),
	err: LapackError,
) {

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	l_int := Blas_Int(l)
	nb := Blas_Int(block_size)
	lda := Blas_Int(A.rows)

	// Copy A matrix for factorization
	a_data := make([]complex64, A.rows * A.cols, allocator) or_return
	copy(a_data, A.data[:A.rows * A.cols])
	factorized_A := Matrix(complex64) {
		data = a_data,
		rows = A.rows,
		cols = A.cols,
	}

	// Handle B matrix
	b_data: []complex64 = nil
	factorized_B: Matrix(complex64)
	ldb: Blas_Int = 1

	if B.data != nil {
		b_data = make([]complex64, B.rows * B.cols, allocator) or_return
		copy(b_data, B.data[:B.rows * B.cols])
		factorized_B = Matrix(complex64) {
			data = b_data,
			rows = B.rows,
			cols = B.cols,
		}
		ldb = Blas_Int(B.rows)
	}

	// Allocate T matrix for reflector information
	t_rows := min(int(nb), int(n))
	t_cols := int(n)
	t_data := make([]complex64, t_rows * t_cols, allocator) or_return
	reflector_T := Matrix(complex64) {
		data = t_data,
		rows = t_rows,
		cols = t_cols,
	}
	ldt := Blas_Int(t_rows)

	// Allocate workspace
	work_size := max(int(nb * n), int(nb * nb))
	work := make([]complex64, work_size, allocator) or_return

	info: Blas_Int
	b_ptr := raw_data(b_data) if b_data != nil else nil

	ctpqrt(
		&m,
		&n,
		&l_int,
		&nb,
		raw_data(a_data),
		&lda,
		b_ptr,
		&ldb,
		raw_data(t_data),
		&ldt,
		raw_data(work),
		&info,
	)

	delete(work, allocator)

	if info != 0 {
		delete(a_data, allocator)
		if b_data != nil do delete(b_data, allocator)
		delete(t_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.factorization_successful = true
	result.factorized_A = factorized_A
	result.factorized_B = factorized_B
	result.reflector_matrix_T = reflector_T
	result.block_size = block_size
	return
}

// ==============================================================================
// Unblocked QR Factorization Functions (tpqrt2)
// ==============================================================================

// Low-level unblocked QR factorization functions (ctpqrt2, dtpqrt2, stpqrt2, ztpqrt2)
ctpqrt2 :: proc(
	m: ^Blas_Int,
	n: ^Blas_Int,
	l: ^Blas_Int,
	A: ^complex64,
	lda: ^Blas_Int,
	B: ^complex64,
	ldb: ^Blas_Int,
	T: ^complex64,
	ldt: ^Blas_Int,
	info: ^Info,
) {
	ctpqrt2_(m, n, l, A, lda, B, ldb, T, ldt, info)
}

dtpqrt2 :: proc(
	m: ^Blas_Int,
	n: ^Blas_Int,
	l: ^Blas_Int,
	A: ^f64,
	lda: ^Blas_Int,
	B: ^f64,
	ldb: ^Blas_Int,
	T: ^f64,
	ldt: ^Blas_Int,
	info: ^Info,
) {
	dtpqrt2_(m, n, l, A, lda, B, ldb, T, ldt, info)
}

stpqrt2 :: proc(
	m: ^Blas_Int,
	n: ^Blas_Int,
	l: ^Blas_Int,
	A: ^f32,
	lda: ^Blas_Int,
	B: ^f32,
	ldb: ^Blas_Int,
	T: ^f32,
	ldt: ^Blas_Int,
	info: ^Info,
) {
	stpqrt2_(m, n, l, A, lda, B, ldb, T, ldt, info)
}

ztpqrt2 :: proc(
	m: ^Blas_Int,
	n: ^Blas_Int,
	l: ^Blas_Int,
	A: ^complex128,
	lda: ^Blas_Int,
	B: ^complex128,
	ldb: ^Blas_Int,
	T: ^complex128,
	ldt: ^Blas_Int,
	info: ^Info,
) {
	ztpqrt2_(m, n, l, A, lda, B, ldb, T, ldt, info)
}

// High-level unblocked QR factorization wrapper function
unblocked_qr_factorization_complex64 :: proc(
	A: Matrix(complex64),
	B: Matrix(complex64) = {},
	l: int,
	allocator := context.allocator,
) -> (
	result: QRFactorizationResult(complex64),
	err: LapackError,
) {

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	l_int := Blas_Int(l)
	lda := Blas_Int(A.rows)

	// Copy A matrix for factorization
	a_data := make([]complex64, A.rows * A.cols, allocator) or_return
	copy(a_data, A.data[:A.rows * A.cols])
	factorized_A := Matrix(complex64) {
		data = a_data,
		rows = A.rows,
		cols = A.cols,
	}

	// Handle B matrix
	b_data: []complex64 = nil
	factorized_B: Matrix(complex64)
	ldb: Blas_Int = 1

	if B.data != nil {
		b_data = make([]complex64, B.rows * B.cols, allocator) or_return
		copy(b_data, B.data[:B.rows * B.cols])
		factorized_B = Matrix(complex64) {
			data = b_data,
			rows = B.rows,
			cols = B.cols,
		}
		ldb = Blas_Int(B.rows)
	}

	// Allocate T matrix for reflector information
	t_rows := int(n)
	t_cols := int(n)
	t_data := make([]complex64, t_rows * t_cols, allocator) or_return
	reflector_T := Matrix(complex64) {
		data = t_data,
		rows = t_rows,
		cols = t_cols,
	}
	ldt := Blas_Int(t_rows)

	info: Blas_Int
	b_ptr := raw_data(b_data) if b_data != nil else nil

	ctpqrt2(&m, &n, &l_int, raw_data(a_data), &lda, b_ptr, &ldb, raw_data(t_data), &ldt, &info)

	if info != 0 {
		delete(a_data, allocator)
		if b_data != nil do delete(b_data, allocator)
		delete(t_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.factorization_successful = true
	result.factorized_A = factorized_A
	result.factorized_B = factorized_B
	result.reflector_matrix_T = reflector_T
	result.block_size = 1 // Unblocked
	return
}

// ==============================================================================
// Additional Convenience Overloads
// ==============================================================================

// QR multiplication with compact WY representation overloads
tpmqrt :: proc {
	ctpmqrt,
	dtpmqrt,
	stpmqrt,
	ztpmqrt,
}

// Block QR factorization overloads
tpqrt :: proc {
	ctpqrt,
	dtpqrt,
	stpqrt,
	ztpqrt,
}

// Unblocked QR factorization overloads
tpqrt2 :: proc {
	ctpqrt2,
	dtpqrt2,
	stpqrt2,
	ztpqrt2,
}

// ==============================================================================
// Rectangular Full-Block Reflector Application Functions (tprfb)
// ==============================================================================


// Rectangular full-block reflector result structure
RectangularReflectorResult :: struct($T: typeid) {
	application_successful: bool,
	result_A:               Matrix(T), // Result matrix A
	result_B:               Matrix(T), // Result matrix B
}

// Helper function to convert reflector direction to string
reflector_direction_to_cstring :: proc(direction: ReflectorDirection) -> cstring {
	switch direction {
	case .Forward:
		return "F"
	case .Backward:
		return "B"
	}
	return "F"
}

// Helper function to convert reflector storage to string
reflector_storage_to_cstring :: proc(storage: ReflectorStorage) -> cstring {
	switch storage {
	case .Columnwise:
		return "C"
	case .Rowwise:
		return "R"
	}
	return "C"
}

// Low-level rectangular full-block reflector functions (ctprfb, dtprfb, stprfb, ztprfb)
ctprfb :: proc(
	side: cstring,
	trans: cstring,
	direct: cstring,
	storev: cstring,
	m: ^Blas_Int,
	n: ^Blas_Int,
	k: ^Blas_Int,
	l: ^Blas_Int,
	V: ^complex64,
	ldv: ^Blas_Int,
	T: ^complex64,
	ldt: ^Blas_Int,
	A: ^complex64,
	lda: ^Blas_Int,
	B: ^complex64,
	ldb: ^Blas_Int,
	work: ^complex64,
	ldwork: ^Blas_Int,
) {
	ctprfb_(
		side,
		trans,
		direct,
		storev,
		m,
		n,
		k,
		l,
		V,
		ldv,
		T,
		ldt,
		A,
		lda,
		B,
		ldb,
		work,
		ldwork,
		len(side),
		len(trans),
		len(direct),
		len(storev),
	)
}

dtprfb :: proc(
	side: cstring,
	trans: cstring,
	direct: cstring,
	storev: cstring,
	m: ^Blas_Int,
	n: ^Blas_Int,
	k: ^Blas_Int,
	l: ^Blas_Int,
	V: ^f64,
	ldv: ^Blas_Int,
	T: ^f64,
	ldt: ^Blas_Int,
	A: ^f64,
	lda: ^Blas_Int,
	B: ^f64,
	ldb: ^Blas_Int,
	work: ^f64,
	ldwork: ^Blas_Int,
) {
	dtprfb_(
		side,
		trans,
		direct,
		storev,
		m,
		n,
		k,
		l,
		V,
		ldv,
		T,
		ldt,
		A,
		lda,
		B,
		ldb,
		work,
		ldwork,
		len(side),
		len(trans),
		len(direct),
		len(storev),
	)
}

stprfb :: proc(
	side: cstring,
	trans: cstring,
	direct: cstring,
	storev: cstring,
	m: ^Blas_Int,
	n: ^Blas_Int,
	k: ^Blas_Int,
	l: ^Blas_Int,
	V: ^f32,
	ldv: ^Blas_Int,
	T: ^f32,
	ldt: ^Blas_Int,
	A: ^f32,
	lda: ^Blas_Int,
	B: ^f32,
	ldb: ^Blas_Int,
	work: ^f32,
	ldwork: ^Blas_Int,
) {
	stprfb_(
		side,
		trans,
		direct,
		storev,
		m,
		n,
		k,
		l,
		V,
		ldv,
		T,
		ldt,
		A,
		lda,
		B,
		ldb,
		work,
		ldwork,
		len(side),
		len(trans),
		len(direct),
		len(storev),
	)
}

ztprfb :: proc(
	side: cstring,
	trans: cstring,
	direct: cstring,
	storev: cstring,
	m: ^Blas_Int,
	n: ^Blas_Int,
	k: ^Blas_Int,
	l: ^Blas_Int,
	V: ^complex128,
	ldv: ^Blas_Int,
	T: ^complex128,
	ldt: ^Blas_Int,
	A: ^complex128,
	lda: ^Blas_Int,
	B: ^complex128,
	ldb: ^Blas_Int,
	work: ^complex128,
	ldwork: ^Blas_Int,
) {
	ztprfb_(
		side,
		trans,
		direct,
		storev,
		m,
		n,
		k,
		l,
		V,
		ldv,
		T,
		ldt,
		A,
		lda,
		B,
		ldb,
		work,
		ldwork,
		len(side),
		len(trans),
		len(direct),
		len(storev),
	)
}

// High-level rectangular full-block reflector wrapper function
apply_rectangular_reflector_complex64 :: proc(
	V: Matrix(complex64),
	T: Matrix(complex64),
	A: Matrix(complex64),
	B: Matrix(complex64) = {},
	l: int,
	side: MatrixSide = .Left,
	trans: MatrixTranspose = .None,
	direction: ReflectorDirection = .Forward,
	storage: ReflectorStorage = .Columnwise,
	allocator := context.allocator,
) -> (
	result: RectangularReflectorResult(complex64),
	err: LapackError,
) {

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	k := Blas_Int(V.cols)
	l_int := Blas_Int(l)
	ldv := Blas_Int(V.rows)
	ldt := Blas_Int(T.rows)
	lda := Blas_Int(A.rows)

	// Copy A matrix for application
	a_data := make([]complex64, A.rows * A.cols, allocator) or_return
	copy(a_data, A.data[:A.rows * A.cols])
	result_A := Matrix(complex64) {
		data = a_data,
		rows = A.rows,
		cols = A.cols,
	}

	// Handle B matrix
	b_data: []complex64 = nil
	result_B: Matrix(complex64)
	ldb: Blas_Int = 1

	if B.data != nil {
		b_data = make([]complex64, B.rows * B.cols, allocator) or_return
		copy(b_data, B.data[:B.rows * B.cols])
		result_B = Matrix(complex64) {
			data = b_data,
			rows = B.rows,
			cols = B.cols,
		}
		ldb = Blas_Int(B.rows)
	}

	// Allocate workspace - larger workspace for full-block operations
	work_size := max(max(int(k * n), int(k * m)), int(l_int * n))
	work := make([]complex64, work_size, allocator) or_return
	ldwork := Blas_Int(max(k, l_int))

	side_str := matrix_side_to_cstring(side)
	trans_str := matrix_transpose_to_cstring(trans)
	direct_str := reflector_direction_to_cstring(direction)
	storev_str := reflector_storage_to_cstring(storage)
	b_ptr := raw_data(b_data) if b_data != nil else nil

	ctprfb(
		side_str,
		trans_str,
		direct_str,
		storev_str,
		&m,
		&n,
		&k,
		&l_int,
		raw_data(V.data),
		&ldv,
		raw_data(T.data),
		&ldt,
		raw_data(a_data),
		&lda,
		b_ptr,
		&ldb,
		raw_data(work),
		&ldwork,
	)

	delete(work, allocator)

	result.application_successful = true
	result.result_A = result_A
	result.result_B = result_B
	return
}

// ==============================================================================
// Triangular Packed Iterative Refinement Functions (tprfs)
// ==============================================================================

// Triangular packed iterative refinement result structure
TriangularPackedRefinementResult :: struct($T: typeid, $S: typeid) {
	refinement_successful: bool,
	solution_matrix:       Matrix(T), // Refined solution matrix X
	forward_errors:        []S, // Forward error bounds for each RHS
	backward_errors:       []S, // Backward error bounds for each RHS
	max_forward_error:     S,
	max_backward_error:    S,
}

// Low-level triangular packed iterative refinement functions (ctprfs, dtprfs, stprfs, ztprfs)
ctprfs :: proc(
	uplo: cstring,
	trans: cstring,
	diag: cstring,
	n: ^Blas_Int,
	nrhs: ^Blas_Int,
	AP: ^complex64,
	B: ^complex64,
	ldb: ^Blas_Int,
	X: ^complex64,
	ldx: ^Blas_Int,
	ferr: ^f32,
	berr: ^f32,
	work: ^complex64,
	rwork: ^f32,
	info: ^Info,
) {
	ctprfs_(
		uplo,
		trans,
		diag,
		n,
		nrhs,
		AP,
		B,
		ldb,
		X,
		ldx,
		ferr,
		berr,
		work,
		rwork,
		info,
		len(uplo),
		len(trans),
		len(diag),
	)
}

dtprfs :: proc(
	uplo: cstring,
	trans: cstring,
	diag: cstring,
	n: ^Blas_Int,
	nrhs: ^Blas_Int,
	AP: ^f64,
	B: ^f64,
	ldb: ^Blas_Int,
	X: ^f64,
	ldx: ^Blas_Int,
	ferr: ^f64,
	berr: ^f64,
	work: ^f64,
	iwork: ^Blas_Int,
	info: ^Info,
) {
	dtprfs_(
		uplo,
		trans,
		diag,
		n,
		nrhs,
		AP,
		B,
		ldb,
		X,
		ldx,
		ferr,
		berr,
		work,
		iwork,
		info,
		len(uplo),
		len(trans),
		len(diag),
	)
}

stprfs :: proc(
	uplo: cstring,
	trans: cstring,
	diag: cstring,
	n: ^Blas_Int,
	nrhs: ^Blas_Int,
	AP: ^f32,
	B: ^f32,
	ldb: ^Blas_Int,
	X: ^f32,
	ldx: ^Blas_Int,
	ferr: ^f32,
	berr: ^f32,
	work: ^f32,
	iwork: ^Blas_Int,
	info: ^Info,
) {
	stprfs_(
		uplo,
		trans,
		diag,
		n,
		nrhs,
		AP,
		B,
		ldb,
		X,
		ldx,
		ferr,
		berr,
		work,
		iwork,
		info,
		len(uplo),
		len(trans),
		len(diag),
	)
}

ztprfs :: proc(
	uplo: cstring,
	trans: cstring,
	diag: cstring,
	n: ^Blas_Int,
	nrhs: ^Blas_Int,
	AP: ^complex128,
	B: ^complex128,
	ldb: ^Blas_Int,
	X: ^complex128,
	ldx: ^Blas_Int,
	ferr: ^f64,
	berr: ^f64,
	work: ^complex128,
	rwork: ^f64,
	info: ^Info,
) {
	ztprfs_(
		uplo,
		trans,
		diag,
		n,
		nrhs,
		AP,
		B,
		ldb,
		X,
		ldx,
		ferr,
		berr,
		work,
		rwork,
		info,
		len(uplo),
		len(trans),
		len(diag),
	)
}

// High-level triangular packed iterative refinement wrapper function
refine_triangular_packed_solution_complex64 :: proc(
	AP: []complex64,
	B: Matrix(complex64),
	X: Matrix(complex64),
	uplo: MatrixTriangle = .Upper,
	trans: MatrixTranspose = .None,
	diag: MatrixDiagonal = .NonUnit,
	allocator := context.allocator,
) -> (
	result: TriangularPackedRefinementResult(complex64, f32),
	err: LapackError,
) {

	n := Blas_Int(B.rows)
	nrhs := Blas_Int(B.cols)
	ldb := Blas_Int(B.rows)
	ldx := Blas_Int(X.rows)

	// Copy solution matrix for refinement
	solution_data := make([]complex64, X.rows * X.cols, allocator) or_return
	copy(solution_data, X.data[:X.rows * X.cols])
	solution_matrix := Matrix(complex64) {
		data = solution_data,
		rows = X.rows,
		cols = X.cols,
	}

	// Allocate error bound arrays
	ferr := make([]f32, int(nrhs), allocator) or_return
	berr := make([]f32, int(nrhs), allocator) or_return

	// Allocate workspace
	work := make([]complex64, 2 * int(n), allocator) or_return
	rwork := make([]f32, int(n), allocator) or_return

	info: Blas_Int
	uplo_str := matrix_triangle_to_cstring(uplo)
	trans_str := matrix_transpose_to_cstring(trans)
	diag_str := matrix_diagonal_to_cstring(diag)

	ctprfs(
		uplo_str,
		trans_str,
		diag_str,
		&n,
		&nrhs,
		raw_data(AP),
		raw_data(B.data),
		&ldb,
		raw_data(solution_data),
		&ldx,
		raw_data(ferr),
		raw_data(berr),
		raw_data(work),
		raw_data(rwork),
		&info,
	)

	delete(work, allocator)
	delete(rwork, allocator)

	if info != 0 {
		delete(solution_data, allocator)
		delete(ferr, allocator)
		delete(berr, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	// Calculate max errors
	max_ferr: f32 = 0
	max_berr: f32 = 0
	for i in 0 ..< int(nrhs) {
		if ferr[i] > max_ferr do max_ferr = ferr[i]
		if berr[i] > max_berr do max_berr = berr[i]
	}

	result.refinement_successful = true
	result.solution_matrix = solution_matrix
	result.forward_errors = ferr
	result.backward_errors = berr
	result.max_forward_error = max_ferr
	result.max_backward_error = max_berr
	return
}

// ==============================================================================
// Triangular Packed Matrix Inversion Functions (tptri)
// ==============================================================================

// Triangular packed inversion result structure
TriangularPackedInversionResult :: struct($T: typeid) {
	inversion_successful: bool,
	inverted_matrix:      []T, // Inverted matrix in packed format
}

// Low-level triangular packed matrix inversion functions (ctptri, dtptri, stptri, ztptri)
ctptri :: proc(uplo: cstring, diag: cstring, n: ^Blas_Int, AP: ^complex64, info: ^Info) {
	ctptri_(uplo, diag, n, AP, info, len(uplo), len(diag))
}

dtptri :: proc(uplo: cstring, diag: cstring, n: ^Blas_Int, AP: ^f64, info: ^Info) {
	dtptri_(uplo, diag, n, AP, info, len(uplo), len(diag))
}

stptri :: proc(uplo: cstring, diag: cstring, n: ^Blas_Int, AP: ^f32, info: ^Info) {
	stptri_(uplo, diag, n, AP, info, len(uplo), len(diag))
}

ztptri :: proc(uplo: cstring, diag: cstring, n: ^Blas_Int, AP: ^complex128, info: ^Info) {
	ztptri_(uplo, diag, n, AP, info, len(uplo), len(diag))
}

// High-level triangular packed matrix inversion wrapper function
invert_triangular_packed_complex64 :: proc(
	AP: []complex64,
	n: int,
	uplo: MatrixTriangle = .Upper,
	diag: MatrixDiagonal = .NonUnit,
	allocator := context.allocator,
) -> (
	result: TriangularPackedInversionResult(complex64),
	err: LapackError,
) {

	n_int := Blas_Int(n)

	// Calculate expected packed size
	packed_size := (n * (n + 1)) / 2
	if len(AP) < packed_size {
		return {}, .InvalidParameter
	}

	// Copy packed matrix for inversion
	inverted_data := make([]complex64, len(AP), allocator) or_return
	copy(inverted_data, AP)

	info: Blas_Int
	uplo_str := matrix_triangle_to_cstring(uplo)
	diag_str := matrix_diagonal_to_cstring(diag)

	ctptri(uplo_str, diag_str, &n_int, raw_data(inverted_data), &info)

	if info != 0 {
		delete(inverted_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .SingularMatrix
		}
	}

	result.inversion_successful = true
	result.inverted_matrix = inverted_data
	return
}

// ==============================================================================
// Final Convenience Overloads for Advanced Functions
// ==============================================================================

// Rectangular full-block reflector overloads
tprfb :: proc {
	ctprfb,
	dtprfb,
	stprfb,
	ztprfb,
}

// Triangular packed iterative refinement overloads
tprfs :: proc {
	ctprfs,
	dtprfs,
	stprfs,
	ztprfs,
}

// Triangular packed matrix inversion overloads
tptri :: proc {
	ctptri,
	dtptri,
	stptri,
	ztptri,
}

// ==============================================================================
// Triangular Packed Linear System Solver Functions (tptrs)
// ==============================================================================

// Low-level LAPACK wrappers for triangular packed linear system solver
ctptrs_ :: proc(
	uplo: cstring,
	trans: cstring,
	diag: cstring,
	n: ^i32,
	nrhs: ^i32,
	AP: ^complex64,
	B: ^complex64,
	ldb: ^i32,
	info: ^i32,
	_: c.size_t,
	_: c.size_t,
	_: c.size_t,
) ---
dtptrs_ :: proc(
	uplo: cstring,
	trans: cstring,
	diag: cstring,
	n: ^i32,
	nrhs: ^i32,
	AP: ^f64,
	B: ^f64,
	ldb: ^i32,
	info: ^i32,
	_: c.size_t,
	_: c.size_t,
	_: c.size_t,
) ---
stptrs_ :: proc(
	uplo: cstring,
	trans: cstring,
	diag: cstring,
	n: ^i32,
	nrhs: ^i32,
	AP: ^f32,
	B: ^f32,
	ldb: ^i32,
	info: ^i32,
	_: c.size_t,
	_: c.size_t,
	_: c.size_t,
) ---
ztptrs_ :: proc(
	uplo: cstring,
	trans: cstring,
	diag: cstring,
	n: ^i32,
	nrhs: ^i32,
	AP: ^complex128,
	B: ^complex128,
	ldb: ^i32,
	info: ^i32,
	_: c.size_t,
	_: c.size_t,
	_: c.size_t,
) ---

// Result structure for triangular packed system solution
TriangularPackedSolveResult :: struct($T: typeid) {
	solve_successful: bool,
	solution_matrix:  Matrix(T),
}

// ctptrs: Solve triangular packed system for complex64
ctptrs :: proc(
	AP: []complex64,
	B: Matrix(complex64),
	uplo: MatrixTriangle = .Upper,
	trans: MatrixTranspose = .None,
	diag: MatrixDiagonal = .NonUnit,
	allocator := context.allocator,
) -> (
	result: TriangularPackedSolveResult(complex64),
	err: LapackError,
) {
	n := B.rows
	nrhs := B.cols

	// Validate inputs
	expected_size := n * (n + 1) / 2
	if len(AP) != expected_size {
		return {}, .InvalidDimension
	}

	// Copy B matrix for solution
	solution_data := make([]complex64, n * nrhs, allocator) or_return
	copy(solution_data, B.data)

	// Setup parameters
	n_i32 := Blas_Int(n)
	nrhs_i32 := Blas_Int(nrhs)
	ldb := Blas_Int(n)
	info: Info

	uplo_str := matrix_triangle_to_cstring(uplo)
	trans_str := matrix_transpose_to_cstring(trans)
	diag_str := matrix_diagonal_to_cstring(diag)

	// Call LAPACK
	ctptrs_(
		uplo_str,
		trans_str,
		diag_str,
		&n_i32,
		&nrhs_i32,
		raw_data(AP),
		raw_data(solution_data),
		&ldb,
		&info,
		len(uplo_str),
		len(trans_str),
		len(diag_str),
	)

	if info != 0 {
		delete(solution_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .SingularMatrix
		}
	}

	result.solve_successful = true
	result.solution_matrix = Matrix(complex64) {
		data = solution_data,
		rows = n,
		cols = nrhs,
	}
	return
}

// dtptrs: Solve triangular packed system for f64
dtptrs :: proc(
	AP: []f64,
	B: Matrix(f64),
	uplo: MatrixTriangle = .Upper,
	trans: MatrixTranspose = .None,
	diag: MatrixDiagonal = .NonUnit,
	allocator := context.allocator,
) -> (
	result: TriangularPackedSolveResult(f64),
	err: LapackError,
) {
	n := B.rows
	nrhs := B.cols

	// Validate inputs
	expected_size := n * (n + 1) / 2
	if len(AP) != expected_size {
		return {}, .InvalidDimension
	}

	// Copy B matrix for solution
	solution_data := make([]f64, n * nrhs, allocator) or_return
	copy(solution_data, B.data)

	// Setup parameters
	n_i32 := Blas_Int(n)
	nrhs_i32 := Blas_Int(nrhs)
	ldb := Blas_Int(n)
	info: Info

	uplo_str := matrix_triangle_to_cstring(uplo)
	trans_str := matrix_transpose_to_cstring(trans)
	diag_str := matrix_diagonal_to_cstring(diag)

	// Call LAPACK
	dtptrs_(
		uplo_str,
		trans_str,
		diag_str,
		&n_i32,
		&nrhs_i32,
		raw_data(AP),
		raw_data(solution_data),
		&ldb,
		&info,
		len(uplo_str),
		len(trans_str),
		len(diag_str),
	)

	if info != 0 {
		delete(solution_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .SingularMatrix
		}
	}

	result.solve_successful = true
	result.solution_matrix = Matrix(f64) {
		data = solution_data,
		rows = n,
		cols = nrhs,
	}
	return
}

// stptrs: Solve triangular packed system for f32
stptrs :: proc(
	AP: []f32,
	B: Matrix(f32),
	uplo: MatrixTriangle = .Upper,
	trans: MatrixTranspose = .None,
	diag: MatrixDiagonal = .NonUnit,
	allocator := context.allocator,
) -> (
	result: TriangularPackedSolveResult(f32),
	err: LapackError,
) {
	n := B.rows
	nrhs := B.cols

	// Validate inputs
	expected_size := n * (n + 1) / 2
	if len(AP) != expected_size {
		return {}, .InvalidDimension
	}

	// Copy B matrix for solution
	solution_data := make([]f32, n * nrhs, allocator) or_return
	copy(solution_data, B.data)

	// Setup parameters
	n_i32 := Blas_Int(n)
	nrhs_i32 := Blas_Int(nrhs)
	ldb := Blas_Int(n)
	info: Info

	uplo_str := matrix_triangle_to_cstring(uplo)
	trans_str := matrix_transpose_to_cstring(trans)
	diag_str := matrix_diagonal_to_cstring(diag)

	// Call LAPACK
	stptrs_(
		uplo_str,
		trans_str,
		diag_str,
		&n_i32,
		&nrhs_i32,
		raw_data(AP),
		raw_data(solution_data),
		&ldb,
		&info,
		len(uplo_str),
		len(trans_str),
		len(diag_str),
	)

	if info != 0 {
		delete(solution_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .SingularMatrix
		}
	}

	result.solve_successful = true
	result.solution_matrix = Matrix(f32) {
		data = solution_data,
		rows = n,
		cols = nrhs,
	}
	return
}

// ztptrs: Solve triangular packed system for complex128
ztptrs :: proc(
	AP: []complex128,
	B: Matrix(complex128),
	uplo: MatrixTriangle = .Upper,
	trans: MatrixTranspose = .None,
	diag: MatrixDiagonal = .NonUnit,
	allocator := context.allocator,
) -> (
	result: TriangularPackedSolveResult(complex128),
	err: LapackError,
) {
	n := B.rows
	nrhs := B.cols

	// Validate inputs
	expected_size := n * (n + 1) / 2
	if len(AP) != expected_size {
		return {}, .InvalidDimension
	}

	// Copy B matrix for solution
	solution_data := make([]complex128, n * nrhs, allocator) or_return
	copy(solution_data, B.data)

	// Setup parameters
	n_i32 := Blas_Int(n)
	nrhs_i32 := Blas_Int(nrhs)
	ldb := Blas_Int(n)
	info: Info

	uplo_str := matrix_triangle_to_cstring(uplo)
	trans_str := matrix_transpose_to_cstring(trans)
	diag_str := matrix_diagonal_to_cstring(diag)

	// Call LAPACK
	ztptrs_(
		uplo_str,
		trans_str,
		diag_str,
		&n_i32,
		&nrhs_i32,
		raw_data(AP),
		raw_data(solution_data),
		&ldb,
		&info,
		len(uplo_str),
		len(trans_str),
		len(diag_str),
	)

	if info != 0 {
		delete(solution_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .SingularMatrix
		}
	}

	result.solve_successful = true
	result.solution_matrix = Matrix(complex128) {
		data = solution_data,
		rows = n,
		cols = nrhs,
	}
	return
}

// ==============================================================================
// Triangular Packed to Rectangular Full Packed Format Functions (tpttf)
// ==============================================================================

// Low-level LAPACK wrappers for triangular packed to rectangular full packed format
ctpttf_ :: proc(
	transr: cstring,
	uplo: cstring,
	n: ^i32,
	AP: ^complex64,
	ARF: ^complex64,
	info: ^i32,
	_: c.size_t,
	_: c.size_t,
) ---
dtpttf_ :: proc(
	transr: cstring,
	uplo: cstring,
	n: ^i32,
	AP: ^f64,
	ARF: ^f64,
	info: ^i32,
	_: c.size_t,
	_: c.size_t,
) ---
stpttf_ :: proc(
	transr: cstring,
	uplo: cstring,
	n: ^i32,
	AP: ^f32,
	ARF: ^f32,
	info: ^i32,
	_: c.size_t,
	_: c.size_t,
) ---
ztpttf_ :: proc(
	transr: cstring,
	uplo: cstring,
	n: ^i32,
	AP: ^complex128,
	ARF: ^complex128,
	info: ^i32,
	_: c.size_t,
	_: c.size_t,
) ---

// Result structure for triangular packed to RFP format conversion
TriangularPackedToRFPResult :: struct($T: typeid) {
	conversion_successful: bool,
	rfp_matrix:            []T,
}

// ctpttf: Convert triangular packed to RFP format for complex64
ctpttf :: proc(
	AP: []complex64,
	n: int,
	uplo: MatrixTriangle = .Upper,
	transr: MatrixTranspose = .None,
	allocator := context.allocator,
) -> (
	result: TriangularPackedToRFPResult(complex64),
	err: LapackError,
) {
	// Validate inputs
	expected_size := n * (n + 1) / 2
	if len(AP) != expected_size {
		return {}, .InvalidDimension
	}

	// Allocate output RFP array
	arf_size := n * (n + 1) / 2
	arf_data := make([]complex64, arf_size, allocator) or_return

	// Setup parameters
	n_i32 := Blas_Int(n)
	info: Info

	uplo_str := matrix_triangle_to_cstring(uplo)
	transr_str := matrix_transpose_to_cstring(transr)

	// Call LAPACK
	ctpttf_(
		transr_str,
		uplo_str,
		&n_i32,
		raw_data(AP),
		raw_data(arf_data),
		&info,
		len(transr_str),
		len(uplo_str),
	)

	if info != 0 {
		delete(arf_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.conversion_successful = true
	result.rfp_matrix = arf_data
	return
}

// dtpttf: Convert triangular packed to RFP format for f64
dtpttf :: proc(
	AP: []f64,
	n: int,
	uplo: MatrixTriangle = .Upper,
	transr: MatrixTranspose = .None,
	allocator := context.allocator,
) -> (
	result: TriangularPackedToRFPResult(f64),
	err: LapackError,
) {
	// Validate inputs
	expected_size := n * (n + 1) / 2
	if len(AP) != expected_size {
		return {}, .InvalidDimension
	}

	// Allocate output RFP array
	arf_size := n * (n + 1) / 2
	arf_data := make([]f64, arf_size, allocator) or_return

	// Setup parameters
	n_i32 := Blas_Int(n)
	info: Info

	uplo_str := matrix_triangle_to_cstring(uplo)
	transr_str := matrix_transpose_to_cstring(transr)

	// Call LAPACK
	dtpttf_(
		transr_str,
		uplo_str,
		&n_i32,
		raw_data(AP),
		raw_data(arf_data),
		&info,
		len(transr_str),
		len(uplo_str),
	)

	if info != 0 {
		delete(arf_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.conversion_successful = true
	result.rfp_matrix = arf_data
	return
}

// stpttf: Convert triangular packed to RFP format for f32
stpttf :: proc(
	AP: []f32,
	n: int,
	uplo: MatrixTriangle = .Upper,
	transr: MatrixTranspose = .None,
	allocator := context.allocator,
) -> (
	result: TriangularPackedToRFPResult(f32),
	err: LapackError,
) {
	// Validate inputs
	expected_size := n * (n + 1) / 2
	if len(AP) != expected_size {
		return {}, .InvalidDimension
	}

	// Allocate output RFP array
	arf_size := n * (n + 1) / 2
	arf_data := make([]f32, arf_size, allocator) or_return

	// Setup parameters
	n_i32 := Blas_Int(n)
	info: Info

	uplo_str := matrix_triangle_to_cstring(uplo)
	transr_str := matrix_transpose_to_cstring(transr)

	// Call LAPACK
	stpttf_(
		transr_str,
		uplo_str,
		&n_i32,
		raw_data(AP),
		raw_data(arf_data),
		&info,
		len(transr_str),
		len(uplo_str),
	)

	if info != 0 {
		delete(arf_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.conversion_successful = true
	result.rfp_matrix = arf_data
	return
}

// ztpttf: Convert triangular packed to RFP format for complex128
ztpttf :: proc(
	AP: []complex128,
	n: int,
	uplo: MatrixTriangle = .Upper,
	transr: MatrixTranspose = .None,
	allocator := context.allocator,
) -> (
	result: TriangularPackedToRFPResult(complex128),
	err: LapackError,
) {
	// Validate inputs
	expected_size := n * (n + 1) / 2
	if len(AP) != expected_size {
		return {}, .InvalidDimension
	}

	// Allocate output RFP array
	arf_size := n * (n + 1) / 2
	arf_data := make([]complex128, arf_size, allocator) or_return

	// Setup parameters
	n_i32 := Blas_Int(n)
	info: Info

	uplo_str := matrix_triangle_to_cstring(uplo)
	transr_str := matrix_transpose_to_cstring(transr)

	// Call LAPACK
	ztpttf_(
		transr_str,
		uplo_str,
		&n_i32,
		raw_data(AP),
		raw_data(arf_data),
		&info,
		len(transr_str),
		len(uplo_str),
	)

	if info != 0 {
		delete(arf_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.conversion_successful = true
	result.rfp_matrix = arf_data
	return
}

// ==============================================================================
// Triangular Packed to Triangular Format Functions (tpttr)
// ==============================================================================

// Low-level LAPACK wrappers for triangular packed to triangular format
ctpttr_ :: proc(
	uplo: cstring,
	n: ^i32,
	AP: ^complex64,
	A: ^complex64,
	lda: ^i32,
	info: ^i32,
	_: c.size_t,
) ---
dtpttr_ :: proc(uplo: cstring, n: ^i32, AP: ^f64, A: ^f64, lda: ^i32, info: ^i32, _: c.size_t) ---
stpttr_ :: proc(uplo: cstring, n: ^i32, AP: ^f32, A: ^f32, lda: ^i32, info: ^i32, _: c.size_t) ---
ztpttr_ :: proc(
	uplo: cstring,
	n: ^i32,
	AP: ^complex128,
	A: ^complex128,
	lda: ^i32,
	info: ^i32,
	_: c.size_t,
) ---

// Result structure for triangular packed to triangular format conversion
TriangularPackedToTriangularResult :: struct($T: typeid) {
	conversion_successful: bool,
	triangular_matrix:     Matrix(T),
}

// ctpttr: Convert triangular packed to triangular format for complex64
ctpttr :: proc(
	AP: []complex64,
	n: int,
	uplo: MatrixTriangle = .Upper,
	allocator := context.allocator,
) -> (
	result: TriangularPackedToTriangularResult(complex64),
	err: LapackError,
) {
	// Validate inputs
	expected_size := n * (n + 1) / 2
	if len(AP) != expected_size {
		return {}, .InvalidDimension
	}

	// Allocate output triangular matrix
	tri_data := make([]complex64, n * n, allocator) or_return

	// Setup parameters
	n_i32 := Blas_Int(n)
	lda := Blas_Int(n)
	info: Info

	uplo_str := matrix_triangle_to_cstring(uplo)

	// Call LAPACK
	ctpttr_(uplo_str, &n_i32, raw_data(AP), raw_data(tri_data), &lda, &info, len(uplo_str))

	if info != 0 {
		delete(tri_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.conversion_successful = true
	result.triangular_matrix = Matrix(complex64) {
		data = tri_data,
		rows = n,
		cols = n,
	}
	return
}

// dtpttr: Convert triangular packed to triangular format for f64
dtpttr :: proc(
	AP: []f64,
	n: int,
	uplo: MatrixTriangle = .Upper,
	allocator := context.allocator,
) -> (
	result: TriangularPackedToTriangularResult(f64),
	err: LapackError,
) {
	// Validate inputs
	expected_size := n * (n + 1) / 2
	if len(AP) != expected_size {
		return {}, .InvalidDimension
	}

	// Allocate output triangular matrix
	tri_data := make([]f64, n * n, allocator) or_return

	// Setup parameters
	n_i32 := Blas_Int(n)
	lda := Blas_Int(n)
	info: Info

	uplo_str := matrix_triangle_to_cstring(uplo)

	// Call LAPACK
	dtpttr_(uplo_str, &n_i32, raw_data(AP), raw_data(tri_data), &lda, &info, len(uplo_str))

	if info != 0 {
		delete(tri_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.conversion_successful = true
	result.triangular_matrix = Matrix(f64) {
		data = tri_data,
		rows = n,
		cols = n,
	}
	return
}

// stpttr: Convert triangular packed to triangular format for f32
stpttr :: proc(
	AP: []f32,
	n: int,
	uplo: MatrixTriangle = .Upper,
	allocator := context.allocator,
) -> (
	result: TriangularPackedToTriangularResult(f32),
	err: LapackError,
) {
	// Validate inputs
	expected_size := n * (n + 1) / 2
	if len(AP) != expected_size {
		return {}, .InvalidDimension
	}

	// Allocate output triangular matrix
	tri_data := make([]f32, n * n, allocator) or_return

	// Setup parameters
	n_i32 := Blas_Int(n)
	lda := Blas_Int(n)
	info: Info

	uplo_str := matrix_triangle_to_cstring(uplo)

	// Call LAPACK
	stpttr_(uplo_str, &n_i32, raw_data(AP), raw_data(tri_data), &lda, &info, len(uplo_str))

	if info != 0 {
		delete(tri_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.conversion_successful = true
	result.triangular_matrix = Matrix(f32) {
		data = tri_data,
		rows = n,
		cols = n,
	}
	return
}

// ztpttr: Convert triangular packed to triangular format for complex128
ztpttr :: proc(
	AP: []complex128,
	n: int,
	uplo: MatrixTriangle = .Upper,
	allocator := context.allocator,
) -> (
	result: TriangularPackedToTriangularResult(complex128),
	err: LapackError,
) {
	// Validate inputs
	expected_size := n * (n + 1) / 2
	if len(AP) != expected_size {
		return {}, .InvalidDimension
	}

	// Allocate output triangular matrix
	tri_data := make([]complex128, n * n, allocator) or_return

	// Setup parameters
	n_i32 := Blas_Int(n)
	lda := Blas_Int(n)
	info: Info

	uplo_str := matrix_triangle_to_cstring(uplo)

	// Call LAPACK
	ztpttr_(uplo_str, &n_i32, raw_data(AP), raw_data(tri_data), &lda, &info, len(uplo_str))

	if info != 0 {
		delete(tri_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.conversion_successful = true
	result.triangular_matrix = Matrix(complex128) {
		data = tri_data,
		rows = n,
		cols = n,
	}
	return
}

// ==============================================================================
// Final Convenience Overloads for Additional Functions
// ==============================================================================

// Triangular packed linear system solver overloads
tptrs :: proc {
	ctptrs,
	dtptrs,
	stptrs,
	ztptrs,
}

// Triangular packed to RFP format overloads
tpttf :: proc {
	ctpttf,
	dtpttf,
	stpttf,
	ztpttf,
}

// Triangular packed to triangular format overloads
tpttr :: proc {
	ctpttr,
	dtpttr,
	stpttr,
	ztpttr,
}

// ==============================================================================
// Triangular Matrix Condition Number Estimation Functions (trcon)
// ==============================================================================

// Low-level LAPACK wrappers for triangular matrix condition number estimation
ctrcon_ :: proc(
	norm: cstring,
	uplo: cstring,
	diag: cstring,
	n: ^i32,
	A: ^complex64,
	lda: ^i32,
	rcond: ^f32,
	work: ^complex64,
	rwork: ^f32,
	info: ^i32,
	_: c.size_t,
	_: c.size_t,
	_: c.size_t,
) ---
dtrcon_ :: proc(
	norm: cstring,
	uplo: cstring,
	diag: cstring,
	n: ^i32,
	A: ^f64,
	lda: ^i32,
	rcond: ^f64,
	work: ^f64,
	iwork: ^i32,
	info: ^i32,
	_: c.size_t,
	_: c.size_t,
	_: c.size_t,
) ---
strcon_ :: proc(
	norm: cstring,
	uplo: cstring,
	diag: cstring,
	n: ^i32,
	A: ^f32,
	lda: ^i32,
	rcond: ^f32,
	work: ^f32,
	iwork: ^i32,
	info: ^i32,
	_: c.size_t,
	_: c.size_t,
	_: c.size_t,
) ---
ztrcon_ :: proc(
	norm: cstring,
	uplo: cstring,
	diag: cstring,
	n: ^i32,
	A: ^complex128,
	lda: ^i32,
	rcond: ^f64,
	work: ^complex128,
	rwork: ^f64,
	info: ^i32,
	_: c.size_t,
	_: c.size_t,
	_: c.size_t,
) ---

// Matrix norm type for condition number estimation
MatrixNorm :: enum {
	OneNorm, // 1-norm (maximum column sum)
	InfinityNorm, // infinity-norm (maximum row sum)
}

matrix_norm_to_cstring :: proc(norm: MatrixNorm) -> cstring {
	switch norm {
	case .OneNorm:
		return "1"
	case .InfinityNorm:
		return "I"
	}
	return "1"
}

// Result structure for triangular matrix condition number estimation
TriangularConditionResult :: struct($T: typeid) {
	estimation_successful: bool,
	reciprocal_condition:  T,
	condition_number:      T,
}

// ctrcon: Estimate condition number of triangular matrix for complex64
ctrcon :: proc(
	A: Matrix(complex64),
	norm: MatrixNorm = .OneNorm,
	uplo: MatrixTriangle = .Upper,
	diag: MatrixDiagonal = .NonUnit,
	allocator := context.allocator,
) -> (
	result: TriangularConditionResult(f32),
	err: LapackError,
) {
	n := A.rows
	if n != A.cols {
		return {}, .InvalidDimension
	}

	// Allocate workspace
	work_size := 2 * n
	rwork_size := n
	work := make([]complex64, work_size, allocator) or_return
	defer delete(work, allocator)
	rwork := make([]f32, rwork_size, allocator) or_return
	defer delete(rwork, allocator)

	// Setup parameters
	n_i32 := Blas_Int(n)
	lda := Blas_Int(A.rows)
	rcond: f32
	info: Info

	norm_str := matrix_norm_to_cstring(norm)
	uplo_str := matrix_triangle_to_cstring(uplo)
	diag_str := matrix_diagonal_to_cstring(diag)

	// Call LAPACK
	ctrcon_(
		norm_str,
		uplo_str,
		diag_str,
		&n_i32,
		raw_data(A.data),
		&lda,
		&rcond,
		raw_data(work),
		raw_data(rwork),
		&info,
		len(norm_str),
		len(uplo_str),
		len(diag_str),
	)

	if info != 0 {
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.estimation_successful = true
	result.reciprocal_condition = rcond
	result.condition_number = rcond > 0 ? 1.0 / rcond : f32(max(f32))
	return
}

// dtrcon: Estimate condition number of triangular matrix for f64
dtrcon :: proc(
	A: Matrix(f64),
	norm: MatrixNorm = .OneNorm,
	uplo: MatrixTriangle = .Upper,
	diag: MatrixDiagonal = .NonUnit,
	allocator := context.allocator,
) -> (
	result: TriangularConditionResult(f64),
	err: LapackError,
) {
	n := A.rows
	if n != A.cols {
		return {}, .InvalidDimension
	}

	// Allocate workspace
	work_size := 3 * n
	iwork_size := n
	work := make([]f64, work_size, allocator) or_return
	defer delete(work, allocator)
	iwork := make([]i32, iwork_size, allocator) or_return
	defer delete(iwork, allocator)

	// Setup parameters
	n_i32 := Blas_Int(n)
	lda := Blas_Int(A.rows)
	rcond: f64
	info: Info

	norm_str := matrix_norm_to_cstring(norm)
	uplo_str := matrix_triangle_to_cstring(uplo)
	diag_str := matrix_diagonal_to_cstring(diag)

	// Call LAPACK
	dtrcon_(
		norm_str,
		uplo_str,
		diag_str,
		&n_i32,
		raw_data(A.data),
		&lda,
		&rcond,
		raw_data(work),
		raw_data(iwork),
		&info,
		len(norm_str),
		len(uplo_str),
		len(diag_str),
	)

	if info != 0 {
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.estimation_successful = true
	result.reciprocal_condition = rcond
	result.condition_number = rcond > 0 ? 1.0 / rcond : f64(max(f64))
	return
}

// strcon: Estimate condition number of triangular matrix for f32
strcon :: proc(
	A: Matrix(f32),
	norm: MatrixNorm = .OneNorm,
	uplo: MatrixTriangle = .Upper,
	diag: MatrixDiagonal = .NonUnit,
	allocator := context.allocator,
) -> (
	result: TriangularConditionResult(f32),
	err: LapackError,
) {
	n := A.rows
	if n != A.cols {
		return {}, .InvalidDimension
	}

	// Allocate workspace
	work_size := 3 * n
	iwork_size := n
	work := make([]f32, work_size, allocator) or_return
	defer delete(work, allocator)
	iwork := make([]i32, iwork_size, allocator) or_return
	defer delete(iwork, allocator)

	// Setup parameters
	n_i32 := Blas_Int(n)
	lda := Blas_Int(A.rows)
	rcond: f32
	info: Info

	norm_str := matrix_norm_to_cstring(norm)
	uplo_str := matrix_triangle_to_cstring(uplo)
	diag_str := matrix_diagonal_to_cstring(diag)

	// Call LAPACK
	strcon_(
		norm_str,
		uplo_str,
		diag_str,
		&n_i32,
		raw_data(A.data),
		&lda,
		&rcond,
		raw_data(work),
		raw_data(iwork),
		&info,
		len(norm_str),
		len(uplo_str),
		len(diag_str),
	)

	if info != 0 {
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.estimation_successful = true
	result.reciprocal_condition = rcond
	result.condition_number = rcond > 0 ? 1.0 / rcond : f32(max(f32))
	return
}

// ztrcon: Estimate condition number of triangular matrix for complex128
ztrcon :: proc(
	A: Matrix(complex128),
	norm: MatrixNorm = .OneNorm,
	uplo: MatrixTriangle = .Upper,
	diag: MatrixDiagonal = .NonUnit,
	allocator := context.allocator,
) -> (
	result: TriangularConditionResult(f64),
	err: LapackError,
) {
	n := A.rows
	if n != A.cols {
		return {}, .InvalidDimension
	}

	// Allocate workspace
	work_size := 2 * n
	rwork_size := n
	work := make([]complex128, work_size, allocator) or_return
	defer delete(work, allocator)
	rwork := make([]f64, rwork_size, allocator) or_return
	defer delete(rwork, allocator)

	// Setup parameters
	n_i32 := Blas_Int(n)
	lda := Blas_Int(A.rows)
	rcond: f64
	info: Info

	norm_str := matrix_norm_to_cstring(norm)
	uplo_str := matrix_triangle_to_cstring(uplo)
	diag_str := matrix_diagonal_to_cstring(diag)

	// Call LAPACK
	ztrcon_(
		norm_str,
		uplo_str,
		diag_str,
		&n_i32,
		raw_data(A.data),
		&lda,
		&rcond,
		raw_data(work),
		raw_data(rwork),
		&info,
		len(norm_str),
		len(uplo_str),
		len(diag_str),
	)

	if info != 0 {
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.estimation_successful = true
	result.reciprocal_condition = rcond
	result.condition_number = rcond > 0 ? 1.0 / rcond : f64(max(f64))
	return
}

// ==============================================================================
// Triangular Matrix Eigenvector Computation Functions (trevc)
// ==============================================================================

// Low-level LAPACK wrappers for triangular matrix eigenvector computation
ctrevc_ :: proc(
	side: cstring,
	howmny: cstring,
	select: ^i32,
	n: ^i32,
	T: ^complex64,
	ldt: ^i32,
	VL: ^complex64,
	ldvl: ^i32,
	VR: ^complex64,
	ldvr: ^i32,
	mm: ^i32,
	m: ^i32,
	work: ^complex64,
	rwork: ^f32,
	info: ^i32,
	_: c.size_t,
	_: c.size_t,
) ---
dtrevc_ :: proc(
	side: cstring,
	howmny: cstring,
	select: ^i32,
	n: ^i32,
	T: ^f64,
	ldt: ^i32,
	VL: ^f64,
	ldvl: ^i32,
	VR: ^f64,
	ldvr: ^i32,
	mm: ^i32,
	m: ^i32,
	work: ^f64,
	info: ^i32,
	_: c.size_t,
	_: c.size_t,
) ---
strevc_ :: proc(
	side: cstring,
	howmny: cstring,
	select: ^i32,
	n: ^i32,
	T: ^f32,
	ldt: ^i32,
	VL: ^f32,
	ldvl: ^i32,
	VR: ^f32,
	ldvr: ^i32,
	mm: ^i32,
	m: ^i32,
	work: ^f32,
	info: ^i32,
	_: c.size_t,
	_: c.size_t,
) ---
ztrevc_ :: proc(
	side: cstring,
	howmny: cstring,
	select: ^i32,
	n: ^i32,
	T: ^complex128,
	ldt: ^i32,
	VL: ^complex128,
	ldvl: ^i32,
	VR: ^complex128,
	ldvr: ^i32,
	mm: ^i32,
	m: ^i32,
	work: ^complex128,
	rwork: ^f64,
	info: ^i32,
	_: c.size_t,
	_: c.size_t,
) ---

// Eigenvector computation side
EigenvectorSide :: enum {
	Right, // Compute right eigenvectors
	Left, // Compute left eigenvectors
	Both, // Compute both left and right eigenvectors
}

eigenvector_side_to_cstring :: proc(side: EigenvectorSide) -> cstring {
	switch side {
	case .Right:
		return "R"
	case .Left:
		return "L"
	case .Both:
		return "B"
	}
	return "R"
}

// Eigenvector selection method
EigenvectorSelection :: enum {
	All, // Compute all eigenvectors
	Backtransform, // Backtransform selected eigenvectors
	Selected, // Compute selected eigenvectors
}

eigenvector_selection_to_cstring :: proc(howmny: EigenvectorSelection) -> cstring {
	switch howmny {
	case .All:
		return "A"
	case .Backtransform:
		return "B"
	case .Selected:
		return "S"
	}
	return "A"
}

// Result structure for triangular matrix eigenvector computation
TriangularEigenvectorResult :: struct($T: typeid) {
	computation_successful: bool,
	left_eigenvectors:      Matrix(T),
	right_eigenvectors:     Matrix(T),
	num_computed:           int,
}

// ctrevc: Compute eigenvectors of triangular matrix for complex64
ctrevc :: proc(
	T: Matrix(complex64),
	side: EigenvectorSide = .Right,
	howmny: EigenvectorSelection = .All,
	select: []bool = {},
	allocator := context.allocator,
) -> (
	result: TriangularEigenvectorResult(complex64),
	err: LapackError,
) {
	n := T.rows
	if n != T.cols {
		return {}, .InvalidDimension
	}

	// Validate selection array
	if howmny == .Selected && len(select) != n {
		return {}, .InvalidDimension
	}

	// Convert selection array to integer array
	select_int: []i32 = {}
	if howmny == .Selected {
		select_int = make([]i32, n, allocator) or_return
		defer delete(select_int, allocator)
		for i in 0 ..< n {
			select_int[i] = select[i] ? 1 : 0
		}
	}

	// Allocate eigenvector matrices
	ldvl := side == .Left || side == .Both ? n : 1
	ldvr := side == .Right || side == .Both ? n : 1

	vl_data: []complex64 = {}
	vr_data: []complex64 = {}

	if side == .Left || side == .Both {
		vl_data = make([]complex64, ldvl * n, allocator) or_return
	}
	if side == .Right || side == .Both {
		vr_data = make([]complex64, ldvr * n, allocator) or_return
	}

	// Allocate workspace
	work_size := 2 * n
	rwork_size := n
	work := make([]complex64, work_size, allocator) or_return
	defer delete(work, allocator)
	rwork := make([]f32, rwork_size, allocator) or_return
	defer delete(rwork, allocator)

	// Setup parameters
	n_i32 := Blas_Int(n)
	ldt := Blas_Int(T.rows)
	ldvl_i32 := Blas_Int(ldvl)
	ldvr_i32 := Blas_Int(ldvr)
	mm := Blas_Int(n)
	m: Blas_Int
	info: Info

	side_str := eigenvector_side_to_cstring(side)
	howmny_str := eigenvector_selection_to_cstring(howmny)

	select_ptr := len(select_int) > 0 ? raw_data(select_int) : nil
	vl_ptr := len(vl_data) > 0 ? raw_data(vl_data) : nil
	vr_ptr := len(vr_data) > 0 ? raw_data(vr_data) : nil

	// Call LAPACK
	ctrevc_(
		side_str,
		howmny_str,
		select_ptr,
		&n_i32,
		raw_data(T.data),
		&ldt,
		vl_ptr,
		&ldvl_i32,
		vr_ptr,
		&ldvr_i32,
		&mm,
		&m,
		raw_data(work),
		raw_data(rwork),
		&info,
		len(side_str),
		len(howmny_str),
	)

	if info != 0 {
		if len(vl_data) > 0 {delete(vl_data, allocator)}
		if len(vr_data) > 0 {delete(vr_data, allocator)}
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.computation_successful = true
	result.num_computed = int(m)

	if len(vl_data) > 0 {
		result.left_eigenvectors = Matrix(complex64) {
			data = vl_data,
			rows = ldvl,
			cols = n,
		}
	}
	if len(vr_data) > 0 {
		result.right_eigenvectors = Matrix(complex64) {
			data = vr_data,
			rows = ldvr,
			cols = n,
		}
	}
	return
}

// dtrevc: Compute eigenvectors of triangular matrix for f64
dtrevc :: proc(
	T: Matrix(f64),
	side: EigenvectorSide = .Right,
	howmny: EigenvectorSelection = .All,
	select: []bool = {},
	allocator := context.allocator,
) -> (
	result: TriangularEigenvectorResult(f64),
	err: LapackError,
) {
	n := T.rows
	if n != T.cols {
		return {}, .InvalidDimension
	}

	// Validate selection array
	if howmny == .Selected && len(select) != n {
		return {}, .InvalidDimension
	}

	// Convert selection array to integer array
	select_int: []i32 = {}
	if howmny == .Selected {
		select_int = make([]i32, n, allocator) or_return
		defer delete(select_int, allocator)
		for i in 0 ..< n {
			select_int[i] = select[i] ? 1 : 0
		}
	}

	// Allocate eigenvector matrices
	ldvl := side == .Left || side == .Both ? n : 1
	ldvr := side == .Right || side == .Both ? n : 1

	vl_data: []f64 = {}
	vr_data: []f64 = {}

	if side == .Left || side == .Both {
		vl_data = make([]f64, ldvl * n, allocator) or_return
	}
	if side == .Right || side == .Both {
		vr_data = make([]f64, ldvr * n, allocator) or_return
	}

	// Allocate workspace
	work_size := 3 * n
	work := make([]f64, work_size, allocator) or_return
	defer delete(work, allocator)

	// Setup parameters
	n_i32 := Blas_Int(n)
	ldt := Blas_Int(T.rows)
	ldvl_i32 := Blas_Int(ldvl)
	ldvr_i32 := Blas_Int(ldvr)
	mm := Blas_Int(n)
	m: Blas_Int
	info: Info

	side_str := eigenvector_side_to_cstring(side)
	howmny_str := eigenvector_selection_to_cstring(howmny)

	select_ptr := len(select_int) > 0 ? raw_data(select_int) : nil
	vl_ptr := len(vl_data) > 0 ? raw_data(vl_data) : nil
	vr_ptr := len(vr_data) > 0 ? raw_data(vr_data) : nil

	// Call LAPACK
	dtrevc_(
		side_str,
		howmny_str,
		select_ptr,
		&n_i32,
		raw_data(T.data),
		&ldt,
		vl_ptr,
		&ldvl_i32,
		vr_ptr,
		&ldvr_i32,
		&mm,
		&m,
		raw_data(work),
		&info,
		len(side_str),
		len(howmny_str),
	)

	if info != 0 {
		if len(vl_data) > 0 {delete(vl_data, allocator)}
		if len(vr_data) > 0 {delete(vr_data, allocator)}
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.computation_successful = true
	result.num_computed = int(m)

	if len(vl_data) > 0 {
		result.left_eigenvectors = Matrix(f64) {
			data = vl_data,
			rows = ldvl,
			cols = n,
		}
	}
	if len(vr_data) > 0 {
		result.right_eigenvectors = Matrix(f64) {
			data = vr_data,
			rows = ldvr,
			cols = n,
		}
	}
	return
}

// ==============================================================================
// Optimized Triangular Matrix Eigenvector Computation Functions (trevc3)
// ==============================================================================

// Low-level LAPACK wrappers for optimized triangular matrix eigenvector computation
ctrevc3_ :: proc(
	side: cstring,
	howmny: cstring,
	select: ^i32,
	n: ^i32,
	T: ^complex64,
	ldt: ^i32,
	VL: ^complex64,
	ldvl: ^i32,
	VR: ^complex64,
	ldvr: ^i32,
	mm: ^i32,
	m: ^i32,
	work: ^complex64,
	lwork: ^i32,
	rwork: ^f32,
	lrwork: ^i32,
	info: ^i32,
	_: c.size_t,
	_: c.size_t,
) ---
dtrevc3_ :: proc(
	side: cstring,
	howmny: cstring,
	select: ^i32,
	n: ^i32,
	T: ^f64,
	ldt: ^i32,
	VL: ^f64,
	ldvl: ^i32,
	VR: ^f64,
	ldvr: ^i32,
	mm: ^i32,
	m: ^i32,
	work: ^f64,
	lwork: ^i32,
	info: ^i32,
	_: c.size_t,
	_: c.size_t,
) ---
strevc3_ :: proc(
	side: cstring,
	howmny: cstring,
	select: ^i32,
	n: ^i32,
	T: ^f32,
	ldt: ^i32,
	VL: ^f32,
	ldvl: ^i32,
	VR: ^f32,
	ldvr: ^i32,
	mm: ^i32,
	m: ^i32,
	work: ^f32,
	lwork: ^i32,
	info: ^i32,
	_: c.size_t,
	_: c.size_t,
) ---
ztrevc3_ :: proc(
	side: cstring,
	howmny: cstring,
	select: ^i32,
	n: ^i32,
	T: ^complex128,
	ldt: ^i32,
	VL: ^complex128,
	ldvl: ^i32,
	VR: ^complex128,
	ldvr: ^i32,
	mm: ^i32,
	m: ^i32,
	work: ^complex128,
	lwork: ^i32,
	rwork: ^f64,
	lrwork: ^i32,
	info: ^i32,
	_: c.size_t,
	_: c.size_t,
) ---

// ctrevc3: Optimized compute eigenvectors of triangular matrix for complex64
ctrevc3 :: proc(
	T: Matrix(complex64),
	side: EigenvectorSide = .Right,
	howmny: EigenvectorSelection = .All,
	select: []bool = {},
	allocator := context.allocator,
) -> (
	result: TriangularEigenvectorResult(complex64),
	err: LapackError,
) {
	n := T.rows
	if n != T.cols {
		return {}, .InvalidDimension
	}

	// Validate selection array
	if howmny == .Selected && len(select) != n {
		return {}, .InvalidDimension
	}

	// Convert selection array to integer array
	select_int: []i32 = {}
	if howmny == .Selected {
		select_int = make([]i32, n, allocator) or_return
		defer delete(select_int, allocator)
		for i in 0 ..< n {
			select_int[i] = select[i] ? 1 : 0
		}
	}

	// Allocate eigenvector matrices
	ldvl := side == .Left || side == .Both ? n : 1
	ldvr := side == .Right || side == .Both ? n : 1

	vl_data: []complex64 = {}
	vr_data: []complex64 = {}

	if side == .Left || side == .Both {
		vl_data = make([]complex64, ldvl * n, allocator) or_return
	}
	if side == .Right || side == .Both {
		vr_data = make([]complex64, ldvr * n, allocator) or_return
	}

	// Query optimal workspace size
	lwork := Blas_Int(-1)
	lrwork := Blas_Int(-1)
	work_query: complex64
	rwork_query: f32
	info: Info

	side_str := eigenvector_side_to_cstring(side)
	howmny_str := eigenvector_selection_to_cstring(howmny)

	select_ptr := len(select_int) > 0 ? raw_data(select_int) : nil
	vl_ptr := len(vl_data) > 0 ? raw_data(vl_data) : nil
	vr_ptr := len(vr_data) > 0 ? raw_data(vr_data) : nil

	n_i32 := Blas_Int(n)
	ldt := Blas_Int(T.rows)
	ldvl_i32 := Blas_Int(ldvl)
	ldvr_i32 := Blas_Int(ldvr)
	mm := Blas_Int(n)
	m: Blas_Int

	// Workspace query
	ctrevc3_(
		side_str,
		howmny_str,
		select_ptr,
		&n_i32,
		raw_data(T.data),
		&ldt,
		vl_ptr,
		&ldvl_i32,
		vr_ptr,
		&ldvr_i32,
		&mm,
		&m,
		&work_query,
		&lwork,
		&rwork_query,
		&lrwork,
		&info,
		len(side_str),
		len(howmny_str),
	)

	if info != 0 {
		if len(vl_data) > 0 {delete(vl_data, allocator)}
		if len(vr_data) > 0 {delete(vr_data, allocator)}
		return {}, .InvalidParameter
	}

	// Allocate optimal workspace
	optimal_lwork := Blas_Int(real(work_query))
	optimal_lrwork := Blas_Int(rwork_query)
	work := make([]complex64, optimal_lwork, allocator) or_return
	defer delete(work, allocator)
	rwork := make([]f32, optimal_lrwork, allocator) or_return
	defer delete(rwork, allocator)

	lwork = optimal_lwork
	lrwork = optimal_lrwork

	// Call LAPACK with optimal workspace
	ctrevc3_(
		side_str,
		howmny_str,
		select_ptr,
		&n_i32,
		raw_data(T.data),
		&ldt,
		vl_ptr,
		&ldvl_i32,
		vr_ptr,
		&ldvr_i32,
		&mm,
		&m,
		raw_data(work),
		&lwork,
		raw_data(rwork),
		&lrwork,
		&info,
		len(side_str),
		len(howmny_str),
	)

	if info != 0 {
		if len(vl_data) > 0 {delete(vl_data, allocator)}
		if len(vr_data) > 0 {delete(vr_data, allocator)}
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.computation_successful = true
	result.num_computed = int(m)

	if len(vl_data) > 0 {
		result.left_eigenvectors = Matrix(complex64) {
			data = vl_data,
			rows = ldvl,
			cols = n,
		}
	}
	if len(vr_data) > 0 {
		result.right_eigenvectors = Matrix(complex64) {
			data = vr_data,
			rows = ldvr,
			cols = n,
		}
	}
	return
}

// ==============================================================================
// Final Convenience Overloads for Triangular Functions
// ==============================================================================

// Triangular matrix condition number estimation overloads
trcon :: proc {
	ctrcon,
	dtrcon,
	strcon,
	ztrcon,
}

// Triangular matrix eigenvector computation overloads
trevc :: proc {
	ctrevc,
	dtrevc,
}

// Optimized triangular matrix eigenvector computation overloads
trevc3 :: proc {
	ctrevc3,
}

// ==============================================================================
// Triangular Matrix Eigenvalue Reordering Functions (trexc)
// ==============================================================================

// Low-level LAPACK wrappers for triangular matrix eigenvalue reordering
ctrexc_ :: proc(
	compq: cstring,
	n: ^i32,
	T: ^complex64,
	ldt: ^i32,
	Q: ^complex64,
	ldq: ^i32,
	ifst: ^i32,
	ilst: ^i32,
	info: ^i32,
	_: c.size_t,
) ---
dtrexc_ :: proc(
	compq: cstring,
	n: ^i32,
	T: ^f64,
	ldt: ^i32,
	Q: ^f64,
	ldq: ^i32,
	ifst: ^i32,
	ilst: ^i32,
	work: ^f64,
	info: ^i32,
	_: c.size_t,
) ---
strexc_ :: proc(
	compq: cstring,
	n: ^i32,
	T: ^f32,
	ldt: ^i32,
	Q: ^f32,
	ldq: ^i32,
	ifst: ^i32,
	ilst: ^i32,
	work: ^f32,
	info: ^i32,
	_: c.size_t,
) ---
ztrexc_ :: proc(
	compq: cstring,
	n: ^i32,
	T: ^complex128,
	ldt: ^i32,
	Q: ^complex128,
	ldq: ^i32,
	ifst: ^i32,
	ilst: ^i32,
	info: ^i32,
	_: c.size_t,
) ---

// Eigenvalue computation mode
EigenvalueComputeMode :: enum {
	EigenvaluesOnly, // Compute eigenvalues only
	Schur, // Compute Schur vectors
}

eigenvalue_compute_mode_to_cstring :: proc(compq: EigenvalueComputeMode) -> cstring {
	switch compq {
	case .EigenvaluesOnly:
		return "N"
	case .Schur:
		return "V"
	}
	return "N"
}

// Result structure for triangular matrix eigenvalue reordering
TriangularReorderResult :: struct($T: typeid) {
	reordering_successful:  bool,
	reordered_schur_matrix: Matrix(T),
	updated_schur_vectors:  Matrix(T),
}

// ctrexc: Reorder eigenvalues in Schur form for complex64
ctrexc :: proc(
	T: Matrix(complex64),
	Q: Matrix(complex64) = {},
	ifst: int,
	ilst: int,
	compq: EigenvalueComputeMode = .EigenvaluesOnly,
	allocator := context.allocator,
) -> (
	result: TriangularReorderResult(complex64),
	err: LapackError,
) {
	n := T.rows
	if n != T.cols {
		return {}, .InvalidDimension
	}

	if compq == .Schur && (Q.rows != n || Q.cols != n) {
		return {}, .InvalidDimension
	}

	if ifst < 1 || ifst > n || ilst < 1 || ilst > n {
		return {}, .InvalidDimension
	}

	// Copy input matrices
	reordered_T := make([]complex64, n * n, allocator) or_return
	copy(reordered_T, T.data)

	updated_Q: []complex64 = {}
	if compq == .Schur {
		updated_Q = make([]complex64, n * n, allocator) or_return
		copy(updated_Q, Q.data)
	}

	// Setup parameters
	n_i32 := Blas_Int(n)
	ldt := Blas_Int(n)
	ldq := Blas_Int(n)
	ifst_i32 := Blas_Int(ifst)
	ilst_i32 := Blas_Int(ilst)
	info: Info

	compq_str := eigenvalue_compute_mode_to_cstring(compq)
	q_ptr := len(updated_Q) > 0 ? raw_data(updated_Q) : nil

	// Call LAPACK
	ctrexc_(
		compq_str,
		&n_i32,
		raw_data(reordered_T),
		&ldt,
		q_ptr,
		&ldq,
		&ifst_i32,
		&ilst_i32,
		&info,
		len(compq_str),
	)

	if info != 0 {
		delete(reordered_T, allocator)
		if len(updated_Q) > 0 {delete(updated_Q, allocator)}
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.reordering_successful = true
	result.reordered_schur_matrix = Matrix(complex64) {
		data = reordered_T,
		rows = n,
		cols = n,
	}

	if len(updated_Q) > 0 {
		result.updated_schur_vectors = Matrix(complex64) {
			data = updated_Q,
			rows = n,
			cols = n,
		}
	}
	return
}

// dtrexc: Reorder eigenvalues in Schur form for f64
dtrexc :: proc(
	T: Matrix(f64),
	Q: Matrix(f64) = {},
	ifst: int,
	ilst: int,
	compq: EigenvalueComputeMode = .EigenvaluesOnly,
	allocator := context.allocator,
) -> (
	result: TriangularReorderResult(f64),
	err: LapackError,
) {
	n := T.rows
	if n != T.cols {
		return {}, .InvalidDimension
	}

	if compq == .Schur && (Q.rows != n || Q.cols != n) {
		return {}, .InvalidDimension
	}

	if ifst < 1 || ifst > n || ilst < 1 || ilst > n {
		return {}, .InvalidDimension
	}

	// Copy input matrices
	reordered_T := make([]f64, n * n, allocator) or_return
	copy(reordered_T, T.data)

	updated_Q: []f64 = {}
	if compq == .Schur {
		updated_Q = make([]f64, n * n, allocator) or_return
		copy(updated_Q, Q.data)
	}

	// Allocate workspace
	work_size := n
	work := make([]f64, work_size, allocator) or_return
	defer delete(work, allocator)

	// Setup parameters
	n_i32 := Blas_Int(n)
	ldt := Blas_Int(n)
	ldq := Blas_Int(n)
	ifst_i32 := Blas_Int(ifst)
	ilst_i32 := Blas_Int(ilst)
	info: Info

	compq_str := eigenvalue_compute_mode_to_cstring(compq)
	q_ptr := len(updated_Q) > 0 ? raw_data(updated_Q) : nil

	// Call LAPACK
	dtrexc_(
		compq_str,
		&n_i32,
		raw_data(reordered_T),
		&ldt,
		q_ptr,
		&ldq,
		&ifst_i32,
		&ilst_i32,
		raw_data(work),
		&info,
		len(compq_str),
	)

	if info != 0 {
		delete(reordered_T, allocator)
		if len(updated_Q) > 0 {delete(updated_Q, allocator)}
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.reordering_successful = true
	result.reordered_schur_matrix = Matrix(f64) {
		data = reordered_T,
		rows = n,
		cols = n,
	}

	if len(updated_Q) > 0 {
		result.updated_schur_vectors = Matrix(f64) {
			data = updated_Q,
			rows = n,
			cols = n,
		}
	}
	return
}

// strexc: Reorder eigenvalues in Schur form for f32
strexc :: proc(
	T: Matrix(f32),
	Q: Matrix(f32) = {},
	ifst: int,
	ilst: int,
	compq: EigenvalueComputeMode = .EigenvaluesOnly,
	allocator := context.allocator,
) -> (
	result: TriangularReorderResult(f32),
	err: LapackError,
) {
	n := T.rows
	if n != T.cols {
		return {}, .InvalidDimension
	}

	if compq == .Schur && (Q.rows != n || Q.cols != n) {
		return {}, .InvalidDimension
	}

	if ifst < 1 || ifst > n || ilst < 1 || ilst > n {
		return {}, .InvalidDimension
	}

	// Copy input matrices
	reordered_T := make([]f32, n * n, allocator) or_return
	copy(reordered_T, T.data)

	updated_Q: []f32 = {}
	if compq == .Schur {
		updated_Q = make([]f32, n * n, allocator) or_return
		copy(updated_Q, Q.data)
	}

	// Allocate workspace
	work_size := n
	work := make([]f32, work_size, allocator) or_return
	defer delete(work, allocator)

	// Setup parameters
	n_i32 := Blas_Int(n)
	ldt := Blas_Int(n)
	ldq := Blas_Int(n)
	ifst_i32 := Blas_Int(ifst)
	ilst_i32 := Blas_Int(ilst)
	info: Info

	compq_str := eigenvalue_compute_mode_to_cstring(compq)
	q_ptr := len(updated_Q) > 0 ? raw_data(updated_Q) : nil

	// Call LAPACK
	strexc_(
		compq_str,
		&n_i32,
		raw_data(reordered_T),
		&ldt,
		q_ptr,
		&ldq,
		&ifst_i32,
		&ilst_i32,
		raw_data(work),
		&info,
		len(compq_str),
	)

	if info != 0 {
		delete(reordered_T, allocator)
		if len(updated_Q) > 0 {delete(updated_Q, allocator)}
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.reordering_successful = true
	result.reordered_schur_matrix = Matrix(f32) {
		data = reordered_T,
		rows = n,
		cols = n,
	}

	if len(updated_Q) > 0 {
		result.updated_schur_vectors = Matrix(f32) {
			data = updated_Q,
			rows = n,
			cols = n,
		}
	}
	return
}

// ztrexc: Reorder eigenvalues in Schur form for complex128
ztrexc :: proc(
	T: Matrix(complex128),
	Q: Matrix(complex128) = {},
	ifst: int,
	ilst: int,
	compq: EigenvalueComputeMode = .EigenvaluesOnly,
	allocator := context.allocator,
) -> (
	result: TriangularReorderResult(complex128),
	err: LapackError,
) {
	n := T.rows
	if n != T.cols {
		return {}, .InvalidDimension
	}

	if compq == .Schur && (Q.rows != n || Q.cols != n) {
		return {}, .InvalidDimension
	}

	if ifst < 1 || ifst > n || ilst < 1 || ilst > n {
		return {}, .InvalidDimension
	}

	// Copy input matrices
	reordered_T := make([]complex128, n * n, allocator) or_return
	copy(reordered_T, T.data)

	updated_Q: []complex128 = {}
	if compq == .Schur {
		updated_Q = make([]complex128, n * n, allocator) or_return
		copy(updated_Q, Q.data)
	}

	// Setup parameters
	n_i32 := Blas_Int(n)
	ldt := Blas_Int(n)
	ldq := Blas_Int(n)
	ifst_i32 := Blas_Int(ifst)
	ilst_i32 := Blas_Int(ilst)
	info: Info

	compq_str := eigenvalue_compute_mode_to_cstring(compq)
	q_ptr := len(updated_Q) > 0 ? raw_data(updated_Q) : nil

	// Call LAPACK
	ztrexc_(
		compq_str,
		&n_i32,
		raw_data(reordered_T),
		&ldt,
		q_ptr,
		&ldq,
		&ifst_i32,
		&ilst_i32,
		&info,
		len(compq_str),
	)

	if info != 0 {
		delete(reordered_T, allocator)
		if len(updated_Q) > 0 {delete(updated_Q, allocator)}
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.reordering_successful = true
	result.reordered_schur_matrix = Matrix(complex128) {
		data = reordered_T,
		rows = n,
		cols = n,
	}

	if len(updated_Q) > 0 {
		result.updated_schur_vectors = Matrix(complex128) {
			data = updated_Q,
			rows = n,
			cols = n,
		}
	}
	return
}

// ==============================================================================
// Triangular Matrix Iterative Refinement Functions (trrfs)
// ==============================================================================

// Low-level LAPACK wrappers for triangular matrix iterative refinement
ctrrfs_ :: proc(
	uplo: cstring,
	trans: cstring,
	diag: cstring,
	n: ^i32,
	nrhs: ^i32,
	A: ^complex64,
	lda: ^i32,
	B: ^complex64,
	ldb: ^i32,
	X: ^complex64,
	ldx: ^i32,
	ferr: ^f32,
	berr: ^f32,
	work: ^complex64,
	rwork: ^f32,
	info: ^i32,
	_: c.size_t,
	_: c.size_t,
	_: c.size_t,
) ---
dtrrfs_ :: proc(
	uplo: cstring,
	trans: cstring,
	diag: cstring,
	n: ^i32,
	nrhs: ^i32,
	A: ^f64,
	lda: ^i32,
	B: ^f64,
	ldb: ^i32,
	X: ^f64,
	ldx: ^i32,
	ferr: ^f64,
	berr: ^f64,
	work: ^f64,
	iwork: ^i32,
	info: ^i32,
	_: c.size_t,
	_: c.size_t,
	_: c.size_t,
) ---
strrfs_ :: proc(
	uplo: cstring,
	trans: cstring,
	diag: cstring,
	n: ^i32,
	nrhs: ^i32,
	A: ^f32,
	lda: ^i32,
	B: ^f32,
	ldb: ^i32,
	X: ^f32,
	ldx: ^i32,
	ferr: ^f32,
	berr: ^f32,
	work: ^f32,
	iwork: ^i32,
	info: ^i32,
	_: c.size_t,
	_: c.size_t,
	_: c.size_t,
) ---
ztrrfs_ :: proc(
	uplo: cstring,
	trans: cstring,
	diag: cstring,
	n: ^i32,
	nrhs: ^i32,
	A: ^complex128,
	lda: ^i32,
	B: ^complex128,
	ldb: ^i32,
	X: ^complex128,
	ldx: ^i32,
	ferr: ^f64,
	berr: ^f64,
	work: ^complex128,
	rwork: ^f64,
	info: ^i32,
	_: c.size_t,
	_: c.size_t,
	_: c.size_t,
) ---

// Result structure for triangular matrix iterative refinement
TriangularRefinementResult :: struct($T: typeid, $R: typeid) {
	refinement_successful: bool,
	refined_solution:      Matrix(T),
	forward_error_bounds:  []R,
	backward_error_bounds: []R,
}

// ctrrfs: Iterative refinement for triangular system for complex64
ctrrfs :: proc(
	A: Matrix(complex64),
	B: Matrix(complex64),
	X: Matrix(complex64),
	uplo: MatrixTriangle = .Upper,
	trans: MatrixTranspose = .None,
	diag: MatrixDiagonal = .NonUnit,
	allocator := context.allocator,
) -> (
	result: TriangularRefinementResult(complex64, f32),
	err: LapackError,
) {
	n := A.rows
	nrhs := B.cols

	if n != A.cols || B.rows != n || X.rows != n || X.cols != nrhs {
		return {}, .InvalidDimension
	}

	// Copy solution matrix for refinement
	refined_X := make([]complex64, n * nrhs, allocator) or_return
	copy(refined_X, X.data)

	// Allocate error bound arrays
	ferr := make([]f32, nrhs, allocator) or_return
	berr := make([]f32, nrhs, allocator) or_return

	// Allocate workspace
	work_size := 2 * n
	rwork_size := n
	work := make([]complex64, work_size, allocator) or_return
	defer delete(work, allocator)
	rwork := make([]f32, rwork_size, allocator) or_return
	defer delete(rwork, allocator)

	// Setup parameters
	n_i32 := Blas_Int(n)
	nrhs_i32 := Blas_Int(nrhs)
	lda := Blas_Int(A.rows)
	ldb := Blas_Int(B.rows)
	ldx := Blas_Int(n)
	info: Info

	uplo_str := matrix_triangle_to_cstring(uplo)
	trans_str := matrix_transpose_to_cstring(trans)
	diag_str := matrix_diagonal_to_cstring(diag)

	// Call LAPACK
	ctrrfs_(
		uplo_str,
		trans_str,
		diag_str,
		&n_i32,
		&nrhs_i32,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(refined_X),
		&ldx,
		raw_data(ferr),
		raw_data(berr),
		raw_data(work),
		raw_data(rwork),
		&info,
		len(uplo_str),
		len(trans_str),
		len(diag_str),
	)

	if info != 0 {
		delete(refined_X, allocator)
		delete(ferr, allocator)
		delete(berr, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.refinement_successful = true
	result.refined_solution = Matrix(complex64) {
		data = refined_X,
		rows = n,
		cols = nrhs,
	}
	result.forward_error_bounds = ferr
	result.backward_error_bounds = berr
	return
}

// ==============================================================================
// Final Convenience Overloads for Advanced Triangular Functions
// ==============================================================================

// Triangular matrix eigenvalue reordering overloads
trexc :: proc {
	ctrexc,
	dtrexc,
	strexc,
	ztrexc,
}

// Triangular matrix iterative refinement overloads
trrfs :: proc {
	ctrrfs,
}

// ==============================================================================
// Triangular Matrix Sensitivity Analysis Functions (trsna)
// ==============================================================================

// Low-level LAPACK wrappers for triangular matrix sensitivity analysis
ctrsna_ :: proc(
	job: cstring,
	howmny: cstring,
	select: ^i32,
	n: ^i32,
	T: ^complex64,
	ldt: ^i32,
	VL: ^complex64,
	ldvl: ^i32,
	VR: ^complex64,
	ldvr: ^i32,
	S: ^f32,
	SEP: ^f32,
	mm: ^i32,
	m: ^i32,
	work: ^complex64,
	ldwork: ^i32,
	rwork: ^f32,
	info: ^i32,
	_: c.size_t,
	_: c.size_t,
) ---
dtrsna_ :: proc(
	job: cstring,
	howmny: cstring,
	select: ^i32,
	n: ^i32,
	T: ^f64,
	ldt: ^i32,
	VL: ^f64,
	ldvl: ^i32,
	VR: ^f64,
	ldvr: ^i32,
	S: ^f64,
	SEP: ^f64,
	mm: ^i32,
	m: ^i32,
	work: ^f64,
	ldwork: ^i32,
	iwork: ^i32,
	info: ^i32,
	_: c.size_t,
	_: c.size_t,
) ---
strsna_ :: proc(
	job: cstring,
	howmny: cstring,
	select: ^i32,
	n: ^i32,
	T: ^f32,
	ldt: ^i32,
	VL: ^f32,
	ldvl: ^i32,
	VR: ^f32,
	ldvr: ^i32,
	S: ^f32,
	SEP: ^f32,
	mm: ^i32,
	m: ^i32,
	work: ^f32,
	ldwork: ^i32,
	iwork: ^i32,
	info: ^i32,
	_: c.size_t,
	_: c.size_t,
) ---
ztrsna_ :: proc(
	job: cstring,
	howmny: cstring,
	select: ^i32,
	n: ^i32,
	T: ^complex128,
	ldt: ^i32,
	VL: ^complex128,
	ldvl: ^i32,
	VR: ^complex128,
	ldvr: ^i32,
	S: ^f64,
	SEP: ^f64,
	mm: ^i32,
	m: ^i32,
	work: ^complex128,
	ldwork: ^i32,
	rwork: ^f64,
	info: ^i32,
	_: c.size_t,
	_: c.size_t,
) ---

// Sensitivity analysis job type
SensitivityJob :: enum {
	EigenvalueCondition, // Compute eigenvalue condition numbers only
	EigenvectorSeparation, // Compute eigenvector separation only
	Both, // Compute both condition numbers and separation
}

sensitivity_job_to_cstring :: proc(job: SensitivityJob) -> cstring {
	switch job {
	case .EigenvalueCondition:
		return "E"
	case .EigenvectorSeparation:
		return "V"
	case .Both:
		return "B"
	}
	return "B"
}

// Result structure for triangular matrix sensitivity analysis
TriangularSensitivityResult :: struct($T: typeid, $R: typeid) {
	analysis_successful:          bool,
	eigenvalue_condition_numbers: []R,
	eigenvector_separations:      []R,
	num_computed:                 int,
}

// ctrsna: Sensitivity analysis for triangular matrix for complex64
ctrsna :: proc(
	T: Matrix(complex64),
	VL: Matrix(complex64) = {},
	VR: Matrix(complex64) = {},
	job: SensitivityJob = .Both,
	howmny: EigenvectorSelection = .All,
	select: []bool = {},
	allocator := context.allocator,
) -> (
	result: TriangularSensitivityResult(complex64, f32),
	err: LapackError,
) {
	n := T.rows
	if n != T.cols {
		return {}, .InvalidDimension
	}

	// Validate eigenvector matrices
	if len(VL.data) > 0 && (VL.rows != n || VL.cols != n) {
		return {}, .InvalidDimension
	}
	if len(VR.data) > 0 && (VR.rows != n || VR.cols != n) {
		return {}, .InvalidDimension
	}

	// Validate selection array
	if howmny == .Selected && len(select) != n {
		return {}, .InvalidDimension
	}

	// Convert selection array to integer array
	select_int: []i32 = {}
	if howmny == .Selected {
		select_int = make([]i32, n, allocator) or_return
		defer delete(select_int, allocator)
		for i in 0 ..< n {
			select_int[i] = select[i] ? 1 : 0
		}
	}

	// Allocate output arrays
	s_array: []f32 = {}
	sep_array: []f32 = {}

	if job == .EigenvalueCondition || job == .Both {
		s_array = make([]f32, n, allocator) or_return
	}
	if job == .EigenvectorSeparation || job == .Both {
		sep_array = make([]f32, n, allocator) or_return
	}

	// Allocate workspace
	ldwork := n
	work_size := ldwork * n
	rwork_size := n
	work := make([]complex64, work_size, allocator) or_return
	defer delete(work, allocator)
	rwork := make([]f32, rwork_size, allocator) or_return
	defer delete(rwork, allocator)

	// Setup parameters
	n_i32 := Blas_Int(n)
	ldt := Blas_Int(T.rows)
	ldvl := len(VL.data) > 0 ? Blas_Int(VL.rows) : Blas_Int(1)
	ldvr := len(VR.data) > 0 ? Blas_Int(VR.rows) : Blas_Int(1)
	ldwork_i32 := Blas_Int(ldwork)
	mm := Blas_Int(n)
	m: Blas_Int
	info: Info

	job_str := sensitivity_job_to_cstring(job)
	howmny_str := eigenvector_selection_to_cstring(howmny)

	select_ptr := len(select_int) > 0 ? raw_data(select_int) : nil
	vl_ptr := len(VL.data) > 0 ? raw_data(VL.data) : nil
	vr_ptr := len(VR.data) > 0 ? raw_data(VR.data) : nil
	s_ptr := len(s_array) > 0 ? raw_data(s_array) : nil
	sep_ptr := len(sep_array) > 0 ? raw_data(sep_array) : nil

	// Call LAPACK
	ctrsna_(
		job_str,
		howmny_str,
		select_ptr,
		&n_i32,
		raw_data(T.data),
		&ldt,
		vl_ptr,
		&ldvl,
		vr_ptr,
		&ldvr,
		s_ptr,
		sep_ptr,
		&mm,
		&m,
		raw_data(work),
		&ldwork_i32,
		raw_data(rwork),
		&info,
		len(job_str),
		len(howmny_str),
	)

	if info != 0 {
		if len(s_array) > 0 {delete(s_array, allocator)}
		if len(sep_array) > 0 {delete(sep_array, allocator)}
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.analysis_successful = true
	result.num_computed = int(m)
	result.eigenvalue_condition_numbers = s_array
	result.eigenvector_separations = sep_array
	return
}

// dtrsna: Sensitivity analysis for triangular matrix for f64
dtrsna :: proc(
	T: Matrix(f64),
	VL: Matrix(f64) = {},
	VR: Matrix(f64) = {},
	job: SensitivityJob = .Both,
	howmny: EigenvectorSelection = .All,
	select: []bool = {},
	allocator := context.allocator,
) -> (
	result: TriangularSensitivityResult(f64, f64),
	err: LapackError,
) {
	n := T.rows
	if n != T.cols {
		return {}, .InvalidDimension
	}

	// Validate eigenvector matrices
	if len(VL.data) > 0 && (VL.rows != n || VL.cols != n) {
		return {}, .InvalidDimension
	}
	if len(VR.data) > 0 && (VR.rows != n || VR.cols != n) {
		return {}, .InvalidDimension
	}

	// Validate selection array
	if howmny == .Selected && len(select) != n {
		return {}, .InvalidDimension
	}

	// Convert selection array to integer array
	select_int: []i32 = {}
	if howmny == .Selected {
		select_int = make([]i32, n, allocator) or_return
		defer delete(select_int, allocator)
		for i in 0 ..< n {
			select_int[i] = select[i] ? 1 : 0
		}
	}

	// Allocate output arrays
	s_array: []f64 = {}
	sep_array: []f64 = {}

	if job == .EigenvalueCondition || job == .Both {
		s_array = make([]f64, n, allocator) or_return
	}
	if job == .EigenvectorSeparation || job == .Both {
		sep_array = make([]f64, n, allocator) or_return
	}

	// Allocate workspace
	ldwork := n
	work_size := ldwork * n
	iwork_size := 2 * (n - 1)
	work := make([]f64, work_size, allocator) or_return
	defer delete(work, allocator)
	iwork := make([]i32, iwork_size, allocator) or_return
	defer delete(iwork, allocator)

	// Setup parameters
	n_i32 := Blas_Int(n)
	ldt := Blas_Int(T.rows)
	ldvl := len(VL.data) > 0 ? Blas_Int(VL.rows) : Blas_Int(1)
	ldvr := len(VR.data) > 0 ? Blas_Int(VR.rows) : Blas_Int(1)
	ldwork_i32 := Blas_Int(ldwork)
	mm := Blas_Int(n)
	m: Blas_Int
	info: Info

	job_str := sensitivity_job_to_cstring(job)
	howmny_str := eigenvector_selection_to_cstring(howmny)

	select_ptr := len(select_int) > 0 ? raw_data(select_int) : nil
	vl_ptr := len(VL.data) > 0 ? raw_data(VL.data) : nil
	vr_ptr := len(VR.data) > 0 ? raw_data(VR.data) : nil
	s_ptr := len(s_array) > 0 ? raw_data(s_array) : nil
	sep_ptr := len(sep_array) > 0 ? raw_data(sep_array) : nil

	// Call LAPACK
	dtrsna_(
		job_str,
		howmny_str,
		select_ptr,
		&n_i32,
		raw_data(T.data),
		&ldt,
		vl_ptr,
		&ldvl,
		vr_ptr,
		&ldvr,
		s_ptr,
		sep_ptr,
		&mm,
		&m,
		raw_data(work),
		&ldwork_i32,
		raw_data(iwork),
		&info,
		len(job_str),
		len(howmny_str),
	)

	if info != 0 {
		if len(s_array) > 0 {delete(s_array, allocator)}
		if len(sep_array) > 0 {delete(sep_array, allocator)}
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.analysis_successful = true
	result.num_computed = int(m)
	result.eigenvalue_condition_numbers = s_array
	result.eigenvector_separations = sep_array
	return
}

// ==============================================================================
// Triangular Sylvester Equation Solver Functions (trsyl)
// ==============================================================================

// Low-level LAPACK wrappers for triangular Sylvester equation solver
ctrsyl_ :: proc(
	trana: cstring,
	tranb: cstring,
	isgn: ^i32,
	m: ^i32,
	n: ^i32,
	A: ^complex64,
	lda: ^i32,
	B: ^complex64,
	ldb: ^i32,
	C: ^complex64,
	ldc: ^i32,
	scale: ^f32,
	info: ^i32,
	_: c.size_t,
	_: c.size_t,
) ---
dtrsyl_ :: proc(
	trana: cstring,
	tranb: cstring,
	isgn: ^i32,
	m: ^i32,
	n: ^i32,
	A: ^f64,
	lda: ^i32,
	B: ^f64,
	ldb: ^i32,
	C: ^f64,
	ldc: ^i32,
	scale: ^f64,
	info: ^i32,
	_: c.size_t,
	_: c.size_t,
) ---
strsyl_ :: proc(
	trana: cstring,
	tranb: cstring,
	isgn: ^i32,
	m: ^i32,
	n: ^i32,
	A: ^f32,
	lda: ^i32,
	B: ^f32,
	ldb: ^i32,
	C: ^f32,
	ldc: ^i32,
	scale: ^f32,
	info: ^i32,
	_: c.size_t,
	_: c.size_t,
) ---
ztrsyl_ :: proc(
	trana: cstring,
	tranb: cstring,
	isgn: ^i32,
	m: ^i32,
	n: ^i32,
	A: ^complex128,
	lda: ^i32,
	B: ^complex128,
	ldb: ^i32,
	C: ^complex128,
	ldc: ^i32,
	scale: ^f64,
	info: ^i32,
	_: c.size_t,
	_: c.size_t,
) ---

// Sylvester equation sign
SylvesterSign :: enum {
	Plus, // Solve op(A)*X + X*op(B) = scale*C
	Minus, // Solve op(A)*X - X*op(B) = scale*C
}

sylvester_sign_to_i32 :: proc(isgn: SylvesterSign) -> Blas_Int {
	switch isgn {
	case .Plus:
		return 1
	case .Minus:
		return -1
	}
	return 1
}

// Result structure for triangular Sylvester equation
TriangularSylvesterResult :: struct($T: typeid, $R: typeid) {
	solve_successful: bool,
	solution_matrix:  Matrix(T),
	scale_factor:     R,
}

// ctrsyl: Solve triangular Sylvester equation for complex64
ctrsyl :: proc(
	A: Matrix(complex64),
	B: Matrix(complex64),
	C: Matrix(complex64),
	trana: MatrixTranspose = .None,
	tranb: MatrixTranspose = .None,
	isgn: SylvesterSign = .Plus,
	allocator := context.allocator,
) -> (
	result: TriangularSylvesterResult(complex64, f32),
	err: LapackError,
) {
	m := A.rows
	n := B.rows

	if m != A.cols || n != B.cols || C.rows != m || C.cols != n {
		return {}, .InvalidDimension
	}

	// Copy C matrix for solution
	solution_data := make([]complex64, m * n, allocator) or_return
	copy(solution_data, C.data)

	// Setup parameters
	m_i32 := Blas_Int(m)
	n_i32 := Blas_Int(n)
	lda := Blas_Int(A.rows)
	ldb := Blas_Int(B.rows)
	ldc := Blas_Int(m)
	isgn_i32 := sylvester_sign_to_i32(isgn)
	scale: f32
	info: Info

	trana_str := matrix_transpose_to_cstring(trana)
	tranb_str := matrix_transpose_to_cstring(tranb)

	// Call LAPACK
	ctrsyl_(
		trana_str,
		tranb_str,
		&isgn_i32,
		&m_i32,
		&n_i32,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(solution_data),
		&ldc,
		&scale,
		&info,
		len(trana_str),
		len(tranb_str),
	)

	if info != 0 {
		delete(solution_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .SingularMatrix
		}
	}

	result.solve_successful = true
	result.solution_matrix = Matrix(complex64) {
		data = solution_data,
		rows = m,
		cols = n,
	}
	result.scale_factor = scale
	return
}

// ==============================================================================
// Final Convenience Overloads for Sensitivity and Sylvester Functions
// ==============================================================================

// Triangular matrix sensitivity analysis overloads
trsna :: proc {
	ctrsna,
	dtrsna,
}

// Triangular Sylvester equation solver overloads
trsyl :: proc {
	ctrsyl,
}

// ==============================================================================
// Triangular Matrix System Solver Functions (trtrs)
// ==============================================================================

// Low-level LAPACK wrappers for triangular matrix system solver
ctrtrs_ :: proc(
	uplo: cstring,
	trans: cstring,
	diag: cstring,
	n: ^i32,
	nrhs: ^i32,
	A: ^complex64,
	lda: ^i32,
	B: ^complex64,
	ldb: ^i32,
	info: ^i32,
	_: c.size_t,
	_: c.size_t,
	_: c.size_t,
) ---
dtrtrs_ :: proc(
	uplo: cstring,
	trans: cstring,
	diag: cstring,
	n: ^i32,
	nrhs: ^i32,
	A: ^f64,
	lda: ^i32,
	B: ^f64,
	ldb: ^i32,
	info: ^i32,
	_: c.size_t,
	_: c.size_t,
	_: c.size_t,
) ---
strtrs_ :: proc(
	uplo: cstring,
	trans: cstring,
	diag: cstring,
	n: ^i32,
	nrhs: ^i32,
	A: ^f32,
	lda: ^i32,
	B: ^f32,
	ldb: ^i32,
	info: ^i32,
	_: c.size_t,
	_: c.size_t,
	_: c.size_t,
) ---
ztrtrs_ :: proc(
	uplo: cstring,
	trans: cstring,
	diag: cstring,
	n: ^i32,
	nrhs: ^i32,
	A: ^complex128,
	lda: ^i32,
	B: ^complex128,
	ldb: ^i32,
	info: ^i32,
	_: c.size_t,
	_: c.size_t,
	_: c.size_t,
) ---

// Result structure for triangular matrix system solver
TriangularSystemResult :: struct($T: typeid) {
	solve_successful: bool,
	solution_matrix:  Matrix(T),
}

// ctrtrs: Solve triangular system for complex64
ctrtrs :: proc(
	A: Matrix(complex64),
	B: Matrix(complex64),
	uplo: MatrixTriangle = .Upper,
	trans: MatrixTranspose = .None,
	diag: MatrixDiagonal = .NonUnit,
	allocator := context.allocator,
) -> (
	result: TriangularSystemResult(complex64),
	err: LapackError,
) {
	n := A.rows
	nrhs := B.cols

	if n != A.cols || B.rows != n {
		return {}, .InvalidDimension
	}

	// Copy B matrix for solution
	solution_data := make([]complex64, n * nrhs, allocator) or_return
	copy(solution_data, B.data)

	// Setup parameters
	n_i32 := Blas_Int(n)
	nrhs_i32 := Blas_Int(nrhs)
	lda := Blas_Int(A.rows)
	ldb := Blas_Int(n)
	info: Info

	uplo_str := matrix_triangle_to_cstring(uplo)
	trans_str := matrix_transpose_to_cstring(trans)
	diag_str := matrix_diagonal_to_cstring(diag)

	// Call LAPACK
	ctrtrs_(
		uplo_str,
		trans_str,
		diag_str,
		&n_i32,
		&nrhs_i32,
		raw_data(A.data),
		&lda,
		raw_data(solution_data),
		&ldb,
		&info,
		len(uplo_str),
		len(trans_str),
		len(diag_str),
	)

	if info != 0 {
		delete(solution_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .SingularMatrix
		}
	}

	result.solve_successful = true
	result.solution_matrix = Matrix(complex64) {
		data = solution_data,
		rows = n,
		cols = nrhs,
	}
	return
}

// dtrtrs: Solve triangular system for f64
dtrtrs :: proc(
	A: Matrix(f64),
	B: Matrix(f64),
	uplo: MatrixTriangle = .Upper,
	trans: MatrixTranspose = .None,
	diag: MatrixDiagonal = .NonUnit,
	allocator := context.allocator,
) -> (
	result: TriangularSystemResult(f64),
	err: LapackError,
) {
	n := A.rows
	nrhs := B.cols

	if n != A.cols || B.rows != n {
		return {}, .InvalidDimension
	}

	// Copy B matrix for solution
	solution_data := make([]f64, n * nrhs, allocator) or_return
	copy(solution_data, B.data)

	// Setup parameters
	n_i32 := Blas_Int(n)
	nrhs_i32 := Blas_Int(nrhs)
	lda := Blas_Int(A.rows)
	ldb := Blas_Int(n)
	info: Info

	uplo_str := matrix_triangle_to_cstring(uplo)
	trans_str := matrix_transpose_to_cstring(trans)
	diag_str := matrix_diagonal_to_cstring(diag)

	// Call LAPACK
	dtrtrs_(
		uplo_str,
		trans_str,
		diag_str,
		&n_i32,
		&nrhs_i32,
		raw_data(A.data),
		&lda,
		raw_data(solution_data),
		&ldb,
		&info,
		len(uplo_str),
		len(trans_str),
		len(diag_str),
	)

	if info != 0 {
		delete(solution_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .SingularMatrix
		}
	}

	result.solve_successful = true
	result.solution_matrix = Matrix(f64) {
		data = solution_data,
		rows = n,
		cols = nrhs,
	}
	return
}

// strtrs: Solve triangular system for f32
strtrs :: proc(
	A: Matrix(f32),
	B: Matrix(f32),
	uplo: MatrixTriangle = .Upper,
	trans: MatrixTranspose = .None,
	diag: MatrixDiagonal = .NonUnit,
	allocator := context.allocator,
) -> (
	result: TriangularSystemResult(f32),
	err: LapackError,
) {
	n := A.rows
	nrhs := B.cols

	if n != A.cols || B.rows != n {
		return {}, .InvalidDimension
	}

	// Copy B matrix for solution
	solution_data := make([]f32, n * nrhs, allocator) or_return
	copy(solution_data, B.data)

	// Setup parameters
	n_i32 := Blas_Int(n)
	nrhs_i32 := Blas_Int(nrhs)
	lda := Blas_Int(A.rows)
	ldb := Blas_Int(n)
	info: Info

	uplo_str := matrix_triangle_to_cstring(uplo)
	trans_str := matrix_transpose_to_cstring(trans)
	diag_str := matrix_diagonal_to_cstring(diag)

	// Call LAPACK
	strtrs_(
		uplo_str,
		trans_str,
		diag_str,
		&n_i32,
		&nrhs_i32,
		raw_data(A.data),
		&lda,
		raw_data(solution_data),
		&ldb,
		&info,
		len(uplo_str),
		len(trans_str),
		len(diag_str),
	)

	if info != 0 {
		delete(solution_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .SingularMatrix
		}
	}

	result.solve_successful = true
	result.solution_matrix = Matrix(f32) {
		data = solution_data,
		rows = n,
		cols = nrhs,
	}
	return
}

// ztrtrs: Solve triangular system for complex128
ztrtrs :: proc(
	A: Matrix(complex128),
	B: Matrix(complex128),
	uplo: MatrixTriangle = .Upper,
	trans: MatrixTranspose = .None,
	diag: MatrixDiagonal = .NonUnit,
	allocator := context.allocator,
) -> (
	result: TriangularSystemResult(complex128),
	err: LapackError,
) {
	n := A.rows
	nrhs := B.cols

	if n != A.cols || B.rows != n {
		return {}, .InvalidDimension
	}

	// Copy B matrix for solution
	solution_data := make([]complex128, n * nrhs, allocator) or_return
	copy(solution_data, B.data)

	// Setup parameters
	n_i32 := Blas_Int(n)
	nrhs_i32 := Blas_Int(nrhs)
	lda := Blas_Int(A.rows)
	ldb := Blas_Int(n)
	info: Info

	uplo_str := matrix_triangle_to_cstring(uplo)
	trans_str := matrix_transpose_to_cstring(trans)
	diag_str := matrix_diagonal_to_cstring(diag)

	// Call LAPACK
	ztrtrs_(
		uplo_str,
		trans_str,
		diag_str,
		&n_i32,
		&nrhs_i32,
		raw_data(A.data),
		&lda,
		raw_data(solution_data),
		&ldb,
		&info,
		len(uplo_str),
		len(trans_str),
		len(diag_str),
	)

	if info != 0 {
		delete(solution_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .SingularMatrix
		}
	}

	result.solve_successful = true
	result.solution_matrix = Matrix(complex128) {
		data = solution_data,
		rows = n,
		cols = nrhs,
	}
	return
}

// ==============================================================================
// Triangular to Rectangular Full Packed Format Functions (trttf)
// ==============================================================================

// Low-level LAPACK wrappers for triangular to RFP format conversion
ctrttf_ :: proc(
	transr: cstring,
	uplo: cstring,
	n: ^i32,
	A: ^complex64,
	lda: ^i32,
	ARF: ^complex64,
	info: ^i32,
	_: c.size_t,
	_: c.size_t,
) ---
dtrttf_ :: proc(
	transr: cstring,
	uplo: cstring,
	n: ^i32,
	A: ^f64,
	lda: ^i32,
	ARF: ^f64,
	info: ^i32,
	_: c.size_t,
	_: c.size_t,
) ---
strttf_ :: proc(
	transr: cstring,
	uplo: cstring,
	n: ^i32,
	A: ^f32,
	lda: ^i32,
	ARF: ^f32,
	info: ^i32,
	_: c.size_t,
	_: c.size_t,
) ---
ztrttf_ :: proc(
	transr: cstring,
	uplo: cstring,
	n: ^i32,
	A: ^complex128,
	lda: ^i32,
	ARF: ^complex128,
	info: ^i32,
	_: c.size_t,
	_: c.size_t,
) ---

// Result structure for triangular to RFP format conversion
TriangularToRFPResult :: struct($T: typeid) {
	conversion_successful: bool,
	rfp_matrix:            []T,
}

// ctrttf: Convert triangular to RFP format for complex64
ctrttf :: proc(
	A: Matrix(complex64),
	uplo: MatrixTriangle = .Upper,
	transr: MatrixTranspose = .None,
	allocator := context.allocator,
) -> (
	result: TriangularToRFPResult(complex64),
	err: LapackError,
) {
	n := A.rows
	if n != A.cols {
		return {}, .InvalidDimension
	}

	// Allocate RFP array
	arf_size := n * (n + 1) / 2
	arf_data := make([]complex64, arf_size, allocator) or_return

	// Setup parameters
	n_i32 := Blas_Int(n)
	lda := Blas_Int(A.rows)
	info: Info

	transr_str := matrix_transpose_to_cstring(transr)
	uplo_str := matrix_triangle_to_cstring(uplo)

	// Call LAPACK
	ctrttf_(
		transr_str,
		uplo_str,
		&n_i32,
		raw_data(A.data),
		&lda,
		raw_data(arf_data),
		&info,
		len(transr_str),
		len(uplo_str),
	)

	if info != 0 {
		delete(arf_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.conversion_successful = true
	result.rfp_matrix = arf_data
	return
}

// ==============================================================================
// Triangular to Triangular Packed Format Functions (trttp)
// ==============================================================================

// Low-level LAPACK wrappers for triangular to packed format conversion
ctrttp_ :: proc(
	uplo: cstring,
	n: ^i32,
	A: ^complex64,
	lda: ^i32,
	AP: ^complex64,
	info: ^i32,
	_: c.size_t,
) ---
dtrttp_ :: proc(uplo: cstring, n: ^i32, A: ^f64, lda: ^i32, AP: ^f64, info: ^i32, _: c.size_t) ---
strttp_ :: proc(uplo: cstring, n: ^i32, A: ^f32, lda: ^i32, AP: ^f32, info: ^i32, _: c.size_t) ---
ztrttp_ :: proc(
	uplo: cstring,
	n: ^i32,
	A: ^complex128,
	lda: ^i32,
	AP: ^complex128,
	info: ^i32,
	_: c.size_t,
) ---

// Result structure for triangular to packed format conversion
TriangularToPackedResult :: struct($T: typeid) {
	conversion_successful: bool,
	packed_matrix:         []T,
}

// ctrttp: Convert triangular to packed format for complex64
ctrttp :: proc(
	A: Matrix(complex64),
	uplo: MatrixTriangle = .Upper,
	allocator := context.allocator,
) -> (
	result: TriangularToPackedResult(complex64),
	err: LapackError,
) {
	n := A.rows
	if n != A.cols {
		return {}, .InvalidDimension
	}

	// Allocate packed array
	ap_size := n * (n + 1) / 2
	ap_data := make([]complex64, ap_size, allocator) or_return

	// Setup parameters
	n_i32 := Blas_Int(n)
	lda := Blas_Int(A.rows)
	info: Info

	uplo_str := matrix_triangle_to_cstring(uplo)

	// Call LAPACK
	ctrttp_(uplo_str, &n_i32, raw_data(A.data), &lda, raw_data(ap_data), &info, len(uplo_str))

	if info != 0 {
		delete(ap_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.conversion_successful = true
	result.packed_matrix = ap_data
	return
}

// ==============================================================================
// Final Convenience Overloads for Triangular Utility Functions
// ==============================================================================

// Triangular matrix system solver overloads
trtrs :: proc {
	ctrtrs,
	dtrtrs,
	strtrs,
	ztrtrs,
}

// Triangular to RFP format overloads
trttf :: proc {
	ctrttf,
}

// Triangular to packed format overloads
trttp :: proc {
	ctrttp,
}

// ==============================================================================
// Trapezoidal Reduction to Zero (TZRZF) Functions
// ==============================================================================

// Reduce upper trapezoidal matrix to upper triangular by orthogonal transformation
// These functions compute an RZ factorization of an upper trapezoidal matrix

// Low-level LAPACK wrappers
foreign openblas {
	ctzrzf_ :: proc(m: ^i32, n: ^i32, A: ^complex64, lda: ^i32, tau: ^complex64, work: ^complex64, lwork: ^i32, info: ^i32) ---
	dtzrzf_ :: proc(m: ^i32, n: ^i32, A: ^f64, lda: ^i32, tau: ^f64, work: ^f64, lwork: ^i32, info: ^i32) ---
	stzrzf_ :: proc(m: ^i32, n: ^i32, A: ^f32, lda: ^i32, tau: ^f32, work: ^f32, lwork: ^i32, info: ^i32) ---
	ztzrzf_ :: proc(m: ^i32, n: ^i32, A: ^complex128, lda: ^i32, tau: ^complex128, work: ^complex128, lwork: ^i32, info: ^i32) ---
}

// Result structure for trapezoidal reduction
TrapezoidalReductionResult :: struct($T: typeid) {
	reduced_matrix:       Matrix(T), // RZ factorized matrix (R in upper triangle, Z info below)
	tau:                  []T, // Scalar factors for elementary reflectors
	reduction_successful: bool,
}

// Complex single precision
ctzrzf :: proc(
	A: Matrix(complex64),
	allocator := context.allocator,
) -> (
	result: TrapezoidalReductionResult(complex64),
	err: LapackError,
) {
	m, n := A.rows, A.cols
	if m > n {
		return {}, .InvalidDimension // Must be m <= n for upper trapezoidal
	}

	// Create working copy
	A_copy := matrix_copy(A, allocator) or_return

	// Allocate tau array
	tau_data := make([]complex64, m, allocator) or_return

	// Query optimal workspace size
	work_query: complex64
	lwork_query := Blas_Int(-1)
	m_i32, n_i32 := Blas_Int(m), Blas_Int(n)
	lda := Blas_Int(A.rows)
	info: Info

	ctzrzf_(
		&m_i32,
		&n_i32,
		raw_data(A_copy.data),
		&lda,
		raw_data(tau_data),
		&work_query,
		&lwork_query,
		&info,
	)

	if info != 0 {
		delete(A_copy.data, allocator)
		delete(tau_data, allocator)
		return {}, .InvalidParameter
	}

	// Allocate optimal workspace
	lwork := Blas_Int(real(work_query))
	work_data := make([]complex64, lwork, allocator) or_return
	defer delete(work_data, allocator)

	// Perform factorization
	ctzrzf_(
		&m_i32,
		&n_i32,
		raw_data(A_copy.data),
		&lda,
		raw_data(tau_data),
		raw_data(work_data),
		&lwork,
		&info,
	)

	if info != 0 {
		delete(A_copy.data, allocator)
		delete(tau_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.reduced_matrix = A_copy
	result.tau = tau_data
	result.reduction_successful = true
	return
}

// Double precision
dtzrzf :: proc(
	A: Matrix(f64),
	allocator := context.allocator,
) -> (
	result: TrapezoidalReductionResult(f64),
	err: LapackError,
) {
	m, n := A.rows, A.cols
	if m > n {
		return {}, .InvalidDimension
	}

	A_copy := matrix_copy(A, allocator) or_return
	tau_data := make([]f64, m, allocator) or_return

	// Query workspace
	work_query: f64
	lwork_query := Blas_Int(-1)
	m_i32, n_i32 := Blas_Int(m), Blas_Int(n)
	lda := Blas_Int(A.rows)
	info: Info

	dtzrzf_(
		&m_i32,
		&n_i32,
		raw_data(A_copy.data),
		&lda,
		raw_data(tau_data),
		&work_query,
		&lwork_query,
		&info,
	)

	if info != 0 {
		delete(A_copy.data, allocator)
		delete(tau_data, allocator)
		return {}, .InvalidParameter
	}

	lwork := Blas_Int(work_query)
	work_data := make([]f64, lwork, allocator) or_return
	defer delete(work_data, allocator)

	dtzrzf_(
		&m_i32,
		&n_i32,
		raw_data(A_copy.data),
		&lda,
		raw_data(tau_data),
		raw_data(work_data),
		&lwork,
		&info,
	)

	if info != 0 {
		delete(A_copy.data, allocator)
		delete(tau_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.reduced_matrix = A_copy
	result.tau = tau_data
	result.reduction_successful = true
	return
}

// Single precision
stzrzf :: proc(
	A: Matrix(f32),
	allocator := context.allocator,
) -> (
	result: TrapezoidalReductionResult(f32),
	err: LapackError,
) {
	m, n := A.rows, A.cols
	if m > n {
		return {}, .InvalidDimension
	}

	A_copy := matrix_copy(A, allocator) or_return
	tau_data := make([]f32, m, allocator) or_return

	// Query workspace
	work_query: f32
	lwork_query := Blas_Int(-1)
	m_i32, n_i32 := Blas_Int(m), Blas_Int(n)
	lda := Blas_Int(A.rows)
	info: Info

	stzrzf_(
		&m_i32,
		&n_i32,
		raw_data(A_copy.data),
		&lda,
		raw_data(tau_data),
		&work_query,
		&lwork_query,
		&info,
	)

	if info != 0 {
		delete(A_copy.data, allocator)
		delete(tau_data, allocator)
		return {}, .InvalidParameter
	}

	lwork := Blas_Int(work_query)
	work_data := make([]f32, lwork, allocator) or_return
	defer delete(work_data, allocator)

	stzrzf_(
		&m_i32,
		&n_i32,
		raw_data(A_copy.data),
		&lda,
		raw_data(tau_data),
		raw_data(work_data),
		&lwork,
		&info,
	)

	if info != 0 {
		delete(A_copy.data, allocator)
		delete(tau_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.reduced_matrix = A_copy
	result.tau = tau_data
	result.reduction_successful = true
	return
}

// Complex double precision
ztzrzf :: proc(
	A: Matrix(complex128),
	allocator := context.allocator,
) -> (
	result: TrapezoidalReductionResult(complex128),
	err: LapackError,
) {
	m, n := A.rows, A.cols
	if m > n {
		return {}, .InvalidDimension
	}

	A_copy := matrix_copy(A, allocator) or_return
	tau_data := make([]complex128, m, allocator) or_return

	// Query workspace
	work_query: complex128
	lwork_query := Blas_Int(-1)
	m_i32, n_i32 := Blas_Int(m), Blas_Int(n)
	lda := Blas_Int(A.rows)
	info: Info

	ztzrzf_(
		&m_i32,
		&n_i32,
		raw_data(A_copy.data),
		&lda,
		raw_data(tau_data),
		&work_query,
		&lwork_query,
		&info,
	)

	if info != 0 {
		delete(A_copy.data, allocator)
		delete(tau_data, allocator)
		return {}, .InvalidParameter
	}

	lwork := Blas_Int(real(work_query))
	work_data := make([]complex128, lwork, allocator) or_return
	defer delete(work_data, allocator)

	ztzrzf_(
		&m_i32,
		&n_i32,
		raw_data(A_copy.data),
		&lda,
		raw_data(tau_data),
		raw_data(work_data),
		&lwork,
		&info,
	)

	if info != 0 {
		delete(A_copy.data, allocator)
		delete(tau_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.reduced_matrix = A_copy
	result.tau = tau_data
	result.reduction_successful = true
	return
} // ==============================================================================
// Unitary Block Diagonal Decomposition (UNBDB) Functions
// ==============================================================================

// Compute block diagonal decomposition of unitary matrix for CS decomposition
// These functions are preprocessing steps for the CS decomposition

// Low-level LAPACK wrappers
foreign openblas {
	cunbdb_ :: proc(trans: cstring, signs: cstring, m: ^i32, p: ^i32, q: ^i32, X11: ^complex64, ldx11: ^i32, X12: ^complex64, ldx12: ^i32, X21: ^complex64, ldx21: ^i32, X22: ^complex64, ldx22: ^i32, theta: ^f32, phi: ^f32, TAUP1: ^complex64, TAUP2: ^complex64, TAUQ1: ^complex64, TAUQ2: ^complex64, work: ^complex64, lwork: ^i32, info: ^i32, _: c.size_t, _: c.size_t) ---
	zunbdb_ :: proc(trans: cstring, signs: cstring, m: ^i32, p: ^i32, q: ^i32, X11: ^complex128, ldx11: ^i32, X12: ^complex128, ldx12: ^i32, X21: ^complex128, ldx21: ^i32, X22: ^complex128, ldx22: ^i32, theta: ^f64, phi: ^f64, TAUP1: ^complex128, TAUP2: ^complex128, TAUQ1: ^complex128, TAUQ2: ^complex128, work: ^complex128, lwork: ^i32, info: ^i32, _: c.size_t, _: c.size_t) ---
}

// Block structure for unitary matrix decomposition
UnbdbResult :: struct($T: typeid, $FT: typeid) {
	X11:                      Matrix(T), // Upper-left block (p x q)
	X12:                      Matrix(T), // Upper-right block (p x (m-q))
	X21:                      Matrix(T), // Lower-left block ((m-p) x q)
	X22:                      Matrix(T), // Lower-right block ((m-p) x (m-q))
	theta:                    []FT, // Array of principal angles
	phi:                      []FT, // Array of secondary angles
	taup1:                    []T, // Scalar factors for P1 transformations
	taup2:                    []T, // Scalar factors for P2 transformations
	tauq1:                    []T, // Scalar factors for Q1 transformations
	tauq2:                    []T, // Scalar factors for Q2 transformations
	decomposition_successful: bool,
}

// Helper function for CS decomposition signs
CSDecompositionSigns :: enum {
	Positive,
	Negative,
}

cs_decomposition_signs_to_cstring :: proc(signs: CSDecompositionSigns) -> cstring {
	switch signs {
	case .Positive:
		return "+"
	case .Negative:
		return "-"
	}
	return "+"
}

// Complex single precision
cunbdb :: proc(
	X11: Matrix(complex64),
	X12: Matrix(complex64),
	X21: Matrix(complex64),
	X22: Matrix(complex64),
	trans: MatrixTranspose = .None,
	signs: CSDecompositionSigns = .Positive,
	allocator := context.allocator,
) -> (
	result: UnbdbResult(complex64, f32),
	err: LapackError,
) {
	// Validate dimensions
	m := X11.rows + X21.rows
	p := X11.rows
	q := X11.cols

	if X11.cols != X21.cols ||
	   X12.rows != X11.rows ||
	   X22.rows != X21.rows ||
	   X12.cols != X22.cols {
		return {}, .InvalidDimension
	}

	// Create working copies
	X11_copy := matrix_copy(X11, allocator) or_return
	X12_copy := matrix_copy(X12, allocator) or_return
	X21_copy := matrix_copy(X21, allocator) or_return
	X22_copy := matrix_copy(X22, allocator) or_return

	// Allocate output arrays
	min_pq := min(p, q)
	theta_data := make([]f32, min_pq, allocator) or_return
	phi_data := make([]f32, min_pq, allocator) or_return
	taup1_data := make([]complex64, p, allocator) or_return
	taup2_data := make([]complex64, m - p, allocator) or_return
	tauq1_data := make([]complex64, q, allocator) or_return
	tauq2_data := make([]complex64, m - q, allocator) or_return

	// Query workspace
	work_query: complex64
	lwork_query := Blas_Int(-1)
	m_i32, p_i32, q_i32 := Blas_Int(m), Blas_Int(p), Blas_Int(q)
	ldx11, ldx12 := Blas_Int(X11.rows), Blas_Int(X12.rows)
	ldx21, ldx22 := Blas_Int(X21.rows), Blas_Int(X22.rows)
	info: Info

	trans_str := matrix_transpose_to_cstring(trans)
	signs_str := cs_decomposition_signs_to_cstring(signs)

	cunbdb_(
		trans_str,
		signs_str,
		&m_i32,
		&p_i32,
		&q_i32,
		raw_data(X11_copy.data),
		&ldx11,
		raw_data(X12_copy.data),
		&ldx12,
		raw_data(X21_copy.data),
		&ldx21,
		raw_data(X22_copy.data),
		&ldx22,
		raw_data(theta_data),
		raw_data(phi_data),
		raw_data(taup1_data),
		raw_data(taup2_data),
		raw_data(tauq1_data),
		raw_data(tauq2_data),
		&work_query,
		&lwork_query,
		&info,
		len(trans_str),
		len(signs_str),
	)

	if info != 0 {
		// Clean up on query failure
		delete(X11_copy.data, allocator)
		delete(X12_copy.data, allocator)
		delete(X21_copy.data, allocator)
		delete(X22_copy.data, allocator)
		delete(theta_data, allocator)
		delete(phi_data, allocator)
		delete(taup1_data, allocator)
		delete(taup2_data, allocator)
		delete(tauq1_data, allocator)
		delete(tauq2_data, allocator)
		return {}, .InvalidParameter
	}

	lwork := Blas_Int(real(work_query))
	work_data := make([]complex64, lwork, allocator) or_return
	defer delete(work_data, allocator)

	// Perform decomposition
	cunbdb_(
		trans_str,
		signs_str,
		&m_i32,
		&p_i32,
		&q_i32,
		raw_data(X11_copy.data),
		&ldx11,
		raw_data(X12_copy.data),
		&ldx12,
		raw_data(X21_copy.data),
		&ldx21,
		raw_data(X22_copy.data),
		&ldx22,
		raw_data(theta_data),
		raw_data(phi_data),
		raw_data(taup1_data),
		raw_data(taup2_data),
		raw_data(tauq1_data),
		raw_data(tauq2_data),
		raw_data(work_data),
		&lwork,
		&info,
		len(trans_str),
		len(signs_str),
	)

	if info != 0 {
		// Clean up on decomposition failure
		delete(X11_copy.data, allocator)
		delete(X12_copy.data, allocator)
		delete(X21_copy.data, allocator)
		delete(X22_copy.data, allocator)
		delete(theta_data, allocator)
		delete(phi_data, allocator)
		delete(taup1_data, allocator)
		delete(taup2_data, allocator)
		delete(tauq1_data, allocator)
		delete(tauq2_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.X11 = X11_copy
	result.X12 = X12_copy
	result.X21 = X21_copy
	result.X22 = X22_copy
	result.theta = theta_data
	result.phi = phi_data
	result.taup1 = taup1_data
	result.taup2 = taup2_data
	result.tauq1 = tauq1_data
	result.tauq2 = tauq2_data
	result.decomposition_successful = true
	return
}

// Complex double precision
zunbdb :: proc(
	X11: Matrix(complex128),
	X12: Matrix(complex128),
	X21: Matrix(complex128),
	X22: Matrix(complex128),
	trans: MatrixTranspose = .None,
	signs: CSDecompositionSigns = .Positive,
	allocator := context.allocator,
) -> (
	result: UnbdbResult(complex128, f64),
	err: LapackError,
) {
	// Validate dimensions
	m := X11.rows + X21.rows
	p := X11.rows
	q := X11.cols

	if X11.cols != X21.cols ||
	   X12.rows != X11.rows ||
	   X22.rows != X21.rows ||
	   X12.cols != X22.cols {
		return {}, .InvalidDimension
	}

	// Create working copies
	X11_copy := matrix_copy(X11, allocator) or_return
	X12_copy := matrix_copy(X12, allocator) or_return
	X21_copy := matrix_copy(X21, allocator) or_return
	X22_copy := matrix_copy(X22, allocator) or_return

	// Allocate output arrays
	min_pq := min(p, q)
	theta_data := make([]f64, min_pq, allocator) or_return
	phi_data := make([]f64, min_pq, allocator) or_return
	taup1_data := make([]complex128, p, allocator) or_return
	taup2_data := make([]complex128, m - p, allocator) or_return
	tauq1_data := make([]complex128, q, allocator) or_return
	tauq2_data := make([]complex128, m - q, allocator) or_return

	// Query workspace
	work_query: complex128
	lwork_query := Blas_Int(-1)
	m_i32, p_i32, q_i32 := Blas_Int(m), Blas_Int(p), Blas_Int(q)
	ldx11, ldx12 := Blas_Int(X11.rows), Blas_Int(X12.rows)
	ldx21, ldx22 := Blas_Int(X21.rows), Blas_Int(X22.rows)
	info: Info

	trans_str := matrix_transpose_to_cstring(trans)
	signs_str := cs_decomposition_signs_to_cstring(signs)

	zunbdb_(
		trans_str,
		signs_str,
		&m_i32,
		&p_i32,
		&q_i32,
		raw_data(X11_copy.data),
		&ldx11,
		raw_data(X12_copy.data),
		&ldx12,
		raw_data(X21_copy.data),
		&ldx21,
		raw_data(X22_copy.data),
		&ldx22,
		raw_data(theta_data),
		raw_data(phi_data),
		raw_data(taup1_data),
		raw_data(taup2_data),
		raw_data(tauq1_data),
		raw_data(tauq2_data),
		&work_query,
		&lwork_query,
		&info,
		len(trans_str),
		len(signs_str),
	)

	if info != 0 {
		// Clean up on query failure
		delete(X11_copy.data, allocator)
		delete(X12_copy.data, allocator)
		delete(X21_copy.data, allocator)
		delete(X22_copy.data, allocator)
		delete(theta_data, allocator)
		delete(phi_data, allocator)
		delete(taup1_data, allocator)
		delete(taup2_data, allocator)
		delete(tauq1_data, allocator)
		delete(tauq2_data, allocator)
		return {}, .InvalidParameter
	}

	lwork := Blas_Int(real(work_query))
	work_data := make([]complex128, lwork, allocator) or_return
	defer delete(work_data, allocator)

	// Perform decomposition
	zunbdb_(
		trans_str,
		signs_str,
		&m_i32,
		&p_i32,
		&q_i32,
		raw_data(X11_copy.data),
		&ldx11,
		raw_data(X12_copy.data),
		&ldx12,
		raw_data(X21_copy.data),
		&ldx21,
		raw_data(X22_copy.data),
		&ldx22,
		raw_data(theta_data),
		raw_data(phi_data),
		raw_data(taup1_data),
		raw_data(taup2_data),
		raw_data(tauq1_data),
		raw_data(tauq2_data),
		raw_data(work_data),
		&lwork,
		&info,
		len(trans_str),
		len(signs_str),
	)

	if info != 0 {
		// Clean up on decomposition failure
		delete(X11_copy.data, allocator)
		delete(X12_copy.data, allocator)
		delete(X21_copy.data, allocator)
		delete(X22_copy.data, allocator)
		delete(theta_data, allocator)
		delete(phi_data, allocator)
		delete(taup1_data, allocator)
		delete(taup2_data, allocator)
		delete(tauq1_data, allocator)
		delete(tauq2_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.X11 = X11_copy
	result.X12 = X12_copy
	result.X21 = X21_copy
	result.X22 = X22_copy
	result.theta = theta_data
	result.phi = phi_data
	result.taup1 = taup1_data
	result.taup2 = taup2_data
	result.tauq1 = tauq1_data
	result.tauq2 = tauq2_data
	result.decomposition_successful = true
	return
}

// ==============================================================================
// CS Decomposition (UNCSD) Functions
// ==============================================================================

// Compute the CS decomposition of a partitioned unitary matrix
// These functions perform the complete CS decomposition

// Low-level LAPACK wrappers
foreign openblas {
	cuncsd_ :: proc(jobu1: cstring, jobu2: cstring, jobv1t: cstring, jobv2t: cstring, trans: cstring, signs: cstring, m: ^i32, p: ^i32, q: ^i32, X11: ^complex64, ldx11: ^i32, X12: ^complex64, ldx12: ^i32, X21: ^complex64, ldx21: ^i32, X22: ^complex64, ldx22: ^i32, theta: ^f32, U1: ^complex64, ldu1: ^i32, U2: ^complex64, ldu2: ^i32, V1T: ^complex64, ldv1t: ^i32, V2T: ^complex64, ldv2t: ^i32, work: ^complex64, lwork: ^i32, rwork: ^f32, lrwork: ^i32, iwork: ^i32, info: ^i32, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t) ---
	zuncsd_ :: proc(jobu1: cstring, jobu2: cstring, jobv1t: cstring, jobv2t: cstring, trans: cstring, signs: cstring, m: ^i32, p: ^i32, q: ^i32, X11: ^complex128, ldx11: ^i32, X12: ^complex128, ldx12: ^i32, X21: ^complex128, ldx21: ^i32, X22: ^complex128, ldx22: ^i32, theta: ^f64, U1: ^complex128, ldu1: ^i32, U2: ^complex128, ldu2: ^i32, V1T: ^complex128, ldv1t: ^i32, V2T: ^complex128, ldv2t: ^i32, work: ^complex128, lwork: ^i32, rwork: ^f64, lrwork: ^i32, iwork: ^i32, info: ^i32, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t, _: c.size_t) ---
}

// Result structure for CS decomposition
CSDecompositionResult :: struct($T: typeid, $FT: typeid) {
	X11:                      Matrix(T), // Upper-left block (modified)
	X12:                      Matrix(T), // Upper-right block (modified)
	X21:                      Matrix(T), // Lower-left block (modified)
	X22:                      Matrix(T), // Lower-right block (modified)
	theta:                    []FT, // Array of principal angles
	U1:                       Matrix(T), // Left unitary matrix for upper part
	U2:                       Matrix(T), // Left unitary matrix for lower part
	V1T:                      Matrix(T), // Right unitary matrix for left part (transposed)
	V2T:                      Matrix(T), // Right unitary matrix for right part (transposed)
	decomposition_successful: bool,
}

// Complex single precision
cuncsd :: proc(
	X11: Matrix(complex64),
	X12: Matrix(complex64),
	X21: Matrix(complex64),
	X22: Matrix(complex64),
	compute_u1: bool = true,
	compute_u2: bool = true,
	compute_v1t: bool = true,
	compute_v2t: bool = true,
	trans: MatrixTranspose = .None,
	signs: CSDecompositionSigns = .Positive,
	allocator := context.allocator,
) -> (
	result: CSDecompositionResult(complex64, f32),
	err: LapackError,
) {
	// Validate dimensions
	m := X11.rows + X21.rows
	p := X11.rows
	q := X11.cols

	if X11.cols != X21.cols ||
	   X12.rows != X11.rows ||
	   X22.rows != X21.rows ||
	   X12.cols != X22.cols {
		return {}, .InvalidDimension
	}

	// Create working copies
	X11_copy := matrix_copy(X11, allocator) or_return
	X12_copy := matrix_copy(X12, allocator) or_return
	X21_copy := matrix_copy(X21, allocator) or_return
	X22_copy := matrix_copy(X22, allocator) or_return

	// Allocate output matrices
	min_pq := min(p, q)
	theta_data := make([]f32, min_pq, allocator) or_return

	U1_data: []complex64
	U2_data: []complex64
	V1T_data: []complex64
	V2T_data: []complex64

	if compute_u1 {
		U1_data = make([]complex64, p * p, allocator) or_return
	}
	if compute_u2 {
		U2_data = make([]complex64, (m - p) * (m - p), allocator) or_return
	}
	if compute_v1t {
		V1T_data = make([]complex64, q * q, allocator) or_return
	}
	if compute_v2t {
		V2T_data = make([]complex64, (m - q) * (m - q), allocator) or_return
	}

	// Query workspace sizes
	work_query: complex64
	rwork_query: f32
	lwork_query := Blas_Int(-1)
	lrwork_query := Blas_Int(-1)
	m_i32, p_i32, q_i32 := Blas_Int(m), Blas_Int(p), Blas_Int(q)
	ldx11, ldx12 := Blas_Int(X11.rows), Blas_Int(X12.rows)
	ldx21, ldx22 := Blas_Int(X21.rows), Blas_Int(X22.rows)
	ldu1, ldu2 := Blas_Int(p), Blas_Int(m - p)
	ldv1t, ldv2t := Blas_Int(q), Blas_Int(m - q)
	info: Info
	iwork_query: Blas_Int

	jobu1_str := job_to_cstring(compute_u1)
	jobu2_str := job_to_cstring(compute_u2)
	jobv1t_str := job_to_cstring(compute_v1t)
	jobv2t_str := job_to_cstring(compute_v2t)
	trans_str := matrix_transpose_to_cstring(trans)
	signs_str := cs_decomposition_signs_to_cstring(signs)

	cuncsd_(
		jobu1_str,
		jobu2_str,
		jobv1t_str,
		jobv2t_str,
		trans_str,
		signs_str,
		&m_i32,
		&p_i32,
		&q_i32,
		raw_data(X11_copy.data),
		&ldx11,
		raw_data(X12_copy.data),
		&ldx12,
		raw_data(X21_copy.data),
		&ldx21,
		raw_data(X22_copy.data),
		&ldx22,
		raw_data(theta_data),
		compute_u1 ? raw_data(U1_data) : nil,
		&ldu1,
		compute_u2 ? raw_data(U2_data) : nil,
		&ldu2,
		compute_v1t ? raw_data(V1T_data) : nil,
		&ldv1t,
		compute_v2t ? raw_data(V2T_data) : nil,
		&ldv2t,
		&work_query,
		&lwork_query,
		&rwork_query,
		&lrwork_query,
		&iwork_query,
		&info,
		len(jobu1_str),
		len(jobu2_str),
		len(jobv1t_str),
		len(jobv2t_str),
		len(trans_str),
		len(signs_str),
	)

	if info != 0 {
		// Clean up on query failure
		delete(X11_copy.data, allocator)
		delete(X12_copy.data, allocator)
		delete(X21_copy.data, allocator)
		delete(X22_copy.data, allocator)
		delete(theta_data, allocator)
		if compute_u1 do delete(U1_data, allocator)
		if compute_u2 do delete(U2_data, allocator)
		if compute_v1t do delete(V1T_data, allocator)
		if compute_v2t do delete(V2T_data, allocator)
		return {}, .InvalidParameter
	}

	// Allocate workspace
	lwork := Blas_Int(real(work_query))
	lrwork := Blas_Int(rwork_query)
	liwork := iwork_query
	work_data := make([]complex64, lwork, allocator) or_return
	rwork_data := make([]f32, lrwork, allocator) or_return
	iwork_data := make([]i32, liwork, allocator) or_return
	defer {
		delete(work_data, allocator)
		delete(rwork_data, allocator)
		delete(iwork_data, allocator)
	}

	// Perform CS decomposition
	cuncsd_(
		jobu1_str,
		jobu2_str,
		jobv1t_str,
		jobv2t_str,
		trans_str,
		signs_str,
		&m_i32,
		&p_i32,
		&q_i32,
		raw_data(X11_copy.data),
		&ldx11,
		raw_data(X12_copy.data),
		&ldx12,
		raw_data(X21_copy.data),
		&ldx21,
		raw_data(X22_copy.data),
		&ldx22,
		raw_data(theta_data),
		compute_u1 ? raw_data(U1_data) : nil,
		&ldu1,
		compute_u2 ? raw_data(U2_data) : nil,
		&ldu2,
		compute_v1t ? raw_data(V1T_data) : nil,
		&ldv1t,
		compute_v2t ? raw_data(V2T_data) : nil,
		&ldv2t,
		raw_data(work_data),
		&lwork,
		raw_data(rwork_data),
		&lrwork,
		raw_data(iwork_data),
		&info,
		len(jobu1_str),
		len(jobu2_str),
		len(jobv1t_str),
		len(jobv2t_str),
		len(trans_str),
		len(signs_str),
	)

	if info != 0 {
		// Clean up on decomposition failure
		delete(X11_copy.data, allocator)
		delete(X12_copy.data, allocator)
		delete(X21_copy.data, allocator)
		delete(X22_copy.data, allocator)
		delete(theta_data, allocator)
		if compute_u1 do delete(U1_data, allocator)
		if compute_u2 do delete(U2_data, allocator)
		if compute_v1t do delete(V1T_data, allocator)
		if compute_v2t do delete(V2T_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.X11 = X11_copy
	result.X12 = X12_copy
	result.X21 = X21_copy
	result.X22 = X22_copy
	result.theta = theta_data

	if compute_u1 {
		result.U1 = Matrix(complex64) {
			data = U1_data,
			rows = p,
			cols = p,
		}
	}
	if compute_u2 {
		result.U2 = Matrix(complex64) {
			data = U2_data,
			rows = m - p,
			cols = m - p,
		}
	}
	if compute_v1t {
		result.V1T = Matrix(complex64) {
			data = V1T_data,
			rows = q,
			cols = q,
		}
	}
	if compute_v2t {
		result.V2T = Matrix(complex64) {
			data = V2T_data,
			rows = m - q,
			cols = m - q,
		}
	}

	result.decomposition_successful = true
	return
}

// Complex double precision
zuncsd :: proc(
	X11: Matrix(complex128),
	X12: Matrix(complex128),
	X21: Matrix(complex128),
	X22: Matrix(complex128),
	compute_u1: bool = true,
	compute_u2: bool = true,
	compute_v1t: bool = true,
	compute_v2t: bool = true,
	trans: MatrixTranspose = .None,
	signs: CSDecompositionSigns = .Positive,
	allocator := context.allocator,
) -> (
	result: CSDecompositionResult(complex128, f64),
	err: LapackError,
) {
	// Validate dimensions
	m := X11.rows + X21.rows
	p := X11.rows
	q := X11.cols

	if X11.cols != X21.cols ||
	   X12.rows != X11.rows ||
	   X22.rows != X21.rows ||
	   X12.cols != X22.cols {
		return {}, .InvalidDimension
	}

	// Create working copies
	X11_copy := matrix_copy(X11, allocator) or_return
	X12_copy := matrix_copy(X12, allocator) or_return
	X21_copy := matrix_copy(X21, allocator) or_return
	X22_copy := matrix_copy(X22, allocator) or_return

	// Allocate output matrices
	min_pq := min(p, q)
	theta_data := make([]f64, min_pq, allocator) or_return

	U1_data: []complex128
	U2_data: []complex128
	V1T_data: []complex128
	V2T_data: []complex128

	if compute_u1 {
		U1_data = make([]complex128, p * p, allocator) or_return
	}
	if compute_u2 {
		U2_data = make([]complex128, (m - p) * (m - p), allocator) or_return
	}
	if compute_v1t {
		V1T_data = make([]complex128, q * q, allocator) or_return
	}
	if compute_v2t {
		V2T_data = make([]complex128, (m - q) * (m - q), allocator) or_return
	}

	// Query workspace sizes
	work_query: complex128
	rwork_query: f64
	lwork_query := Blas_Int(-1)
	lrwork_query := Blas_Int(-1)
	m_i32, p_i32, q_i32 := Blas_Int(m), Blas_Int(p), Blas_Int(q)
	ldx11, ldx12 := Blas_Int(X11.rows), Blas_Int(X12.rows)
	ldx21, ldx22 := Blas_Int(X21.rows), Blas_Int(X22.rows)
	ldu1, ldu2 := Blas_Int(p), Blas_Int(m - p)
	ldv1t, ldv2t := Blas_Int(q), Blas_Int(m - q)
	info: Info
	iwork_query: Blas_Int

	jobu1_str := job_to_cstring(compute_u1)
	jobu2_str := job_to_cstring(compute_u2)
	jobv1t_str := job_to_cstring(compute_v1t)
	jobv2t_str := job_to_cstring(compute_v2t)
	trans_str := matrix_transpose_to_cstring(trans)
	signs_str := cs_decomposition_signs_to_cstring(signs)

	zuncsd_(
		jobu1_str,
		jobu2_str,
		jobv1t_str,
		jobv2t_str,
		trans_str,
		signs_str,
		&m_i32,
		&p_i32,
		&q_i32,
		raw_data(X11_copy.data),
		&ldx11,
		raw_data(X12_copy.data),
		&ldx12,
		raw_data(X21_copy.data),
		&ldx21,
		raw_data(X22_copy.data),
		&ldx22,
		raw_data(theta_data),
		compute_u1 ? raw_data(U1_data) : nil,
		&ldu1,
		compute_u2 ? raw_data(U2_data) : nil,
		&ldu2,
		compute_v1t ? raw_data(V1T_data) : nil,
		&ldv1t,
		compute_v2t ? raw_data(V2T_data) : nil,
		&ldv2t,
		&work_query,
		&lwork_query,
		&rwork_query,
		&lrwork_query,
		&iwork_query,
		&info,
		len(jobu1_str),
		len(jobu2_str),
		len(jobv1t_str),
		len(jobv2t_str),
		len(trans_str),
		len(signs_str),
	)

	if info != 0 {
		// Clean up on query failure
		delete(X11_copy.data, allocator)
		delete(X12_copy.data, allocator)
		delete(X21_copy.data, allocator)
		delete(X22_copy.data, allocator)
		delete(theta_data, allocator)
		if compute_u1 do delete(U1_data, allocator)
		if compute_u2 do delete(U2_data, allocator)
		if compute_v1t do delete(V1T_data, allocator)
		if compute_v2t do delete(V2T_data, allocator)
		return {}, .InvalidParameter
	}

	// Allocate workspace
	lwork := Blas_Int(real(work_query))
	lrwork := Blas_Int(rwork_query)
	liwork := iwork_query
	work_data := make([]complex128, lwork, allocator) or_return
	rwork_data := make([]f64, lrwork, allocator) or_return
	iwork_data := make([]i32, liwork, allocator) or_return
	defer {
		delete(work_data, allocator)
		delete(rwork_data, allocator)
		delete(iwork_data, allocator)
	}

	// Perform CS decomposition
	zuncsd_(
		jobu1_str,
		jobu2_str,
		jobv1t_str,
		jobv2t_str,
		trans_str,
		signs_str,
		&m_i32,
		&p_i32,
		&q_i32,
		raw_data(X11_copy.data),
		&ldx11,
		raw_data(X12_copy.data),
		&ldx12,
		raw_data(X21_copy.data),
		&ldx21,
		raw_data(X22_copy.data),
		&ldx22,
		raw_data(theta_data),
		compute_u1 ? raw_data(U1_data) : nil,
		&ldu1,
		compute_u2 ? raw_data(U2_data) : nil,
		&ldu2,
		compute_v1t ? raw_data(V1T_data) : nil,
		&ldv1t,
		compute_v2t ? raw_data(V2T_data) : nil,
		&ldv2t,
		raw_data(work_data),
		&lwork,
		raw_data(rwork_data),
		&lrwork,
		raw_data(iwork_data),
		&info,
		len(jobu1_str),
		len(jobu2_str),
		len(jobv1t_str),
		len(jobv2t_str),
		len(trans_str),
		len(signs_str),
	)

	if info != 0 {
		// Clean up on decomposition failure
		delete(X11_copy.data, allocator)
		delete(X12_copy.data, allocator)
		delete(X21_copy.data, allocator)
		delete(X22_copy.data, allocator)
		delete(theta_data, allocator)
		if compute_u1 do delete(U1_data, allocator)
		if compute_u2 do delete(U2_data, allocator)
		if compute_v1t do delete(V1T_data, allocator)
		if compute_v2t do delete(V2T_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.X11 = X11_copy
	result.X12 = X12_copy
	result.X21 = X21_copy
	result.X22 = X22_copy
	result.theta = theta_data

	if compute_u1 {
		result.U1 = Matrix(complex128) {
			data = U1_data,
			rows = p,
			cols = p,
		}
	}
	if compute_u2 {
		result.U2 = Matrix(complex128) {
			data = U2_data,
			rows = m - p,
			cols = m - p,
		}
	}
	if compute_v1t {
		result.V1T = Matrix(complex128) {
			data = V1T_data,
			rows = q,
			cols = q,
		}
	}
	if compute_v2t {
		result.V2T = Matrix(complex128) {
			data = V2T_data,
			rows = m - q,
			cols = m - q,
		}
	}

	result.decomposition_successful = true
	return
}

// ==============================================================================
// Final Convenience Overloads for Advanced Matrix Functions
// ==============================================================================

// Trapezoidal reduction to zero overloads
tzrzf :: proc {
	ctzrzf,
	dtzrzf,
	stzrzf,
	ztzrzf,
}

// Unitary block diagonal decomposition overloads
unbdb :: proc {
	cunbdb,
	zunbdb,
}

// CS decomposition overloads
uncsd :: proc {
	cuncsd,
	zuncsd,
}

// ==============================================================================
// 2-by-1 CS Decomposition (UNCSD2BY1) Functions
// ==============================================================================

// Compute the CS decomposition of a 2-by-1 partitioned unitary matrix
// These functions handle the special case where X12 and X22 blocks are missing

// Low-level LAPACK wrappers
foreign openblas {
	cuncsd2by1_ :: proc(jobu1: cstring, jobu2: cstring, jobv1t: cstring, m: ^i32, p: ^i32, q: ^i32, X11: ^complex64, ldx11: ^i32, X21: ^complex64, ldx21: ^i32, theta: ^f32, U1: ^complex64, ldu1: ^i32, U2: ^complex64, ldu2: ^i32, V1T: ^complex64, ldv1t: ^i32, work: ^complex64, lwork: ^i32, rwork: ^f32, lrwork: ^i32, iwork: ^i32, info: ^i32, _: c.size_t, _: c.size_t, _: c.size_t) ---
	zuncsd2by1_ :: proc(jobu1: cstring, jobu2: cstring, jobv1t: cstring, m: ^i32, p: ^i32, q: ^i32, X11: ^complex128, ldx11: ^i32, X21: ^complex128, ldx21: ^i32, theta: ^f64, U1: ^complex128, ldu1: ^i32, U2: ^complex128, ldu2: ^i32, V1T: ^complex128, ldv1t: ^i32, work: ^complex128, lwork: ^i32, rwork: ^f64, lrwork: ^i32, iwork: ^i32, info: ^i32, _: c.size_t, _: c.size_t, _: c.size_t) ---
}

// Result structure for 2-by-1 CS decomposition
CSDecomposition2by1Result :: struct($T: typeid, $FT: typeid) {
	X11:                      Matrix(T), // Upper block (modified)
	X21:                      Matrix(T), // Lower block (modified)
	theta:                    []FT, // Array of principal angles
	U1:                       Matrix(T), // Left unitary matrix for upper part
	U2:                       Matrix(T), // Left unitary matrix for lower part
	V1T:                      Matrix(T), // Right unitary matrix (transposed)
	decomposition_successful: bool,
}

// Complex single precision
cuncsd2by1 :: proc(
	X11: Matrix(complex64),
	X21: Matrix(complex64),
	compute_u1: bool = true,
	compute_u2: bool = true,
	compute_v1t: bool = true,
	allocator := context.allocator,
) -> (
	result: CSDecomposition2by1Result(complex64, f32),
	err: LapackError,
) {
	// Validate dimensions
	m := X11.rows + X21.rows
	p := X11.rows
	q := X11.cols

	if X11.cols != X21.cols {
		return {}, .InvalidDimension
	}

	// Create working copies
	X11_copy := matrix_copy(X11, allocator) or_return
	X21_copy := matrix_copy(X21, allocator) or_return

	// Allocate output matrices
	min_pq := min(p, q)
	theta_data := make([]f32, min_pq, allocator) or_return

	U1_data: []complex64
	U2_data: []complex64
	V1T_data: []complex64

	if compute_u1 {
		U1_data = make([]complex64, p * p, allocator) or_return
	}
	if compute_u2 {
		U2_data = make([]complex64, (m - p) * (m - p), allocator) or_return
	}
	if compute_v1t {
		V1T_data = make([]complex64, q * q, allocator) or_return
	}

	// Query workspace sizes
	work_query: complex64
	rwork_query: f32
	lwork_query := Blas_Int(-1)
	lrwork_query := Blas_Int(-1)
	m_i32, p_i32, q_i32 := Blas_Int(m), Blas_Int(p), Blas_Int(q)
	ldx11, ldx21 := Blas_Int(X11.rows), Blas_Int(X21.rows)
	ldu1, ldu2 := Blas_Int(p), Blas_Int(m - p)
	ldv1t := Blas_Int(q)
	info: Info
	iwork_query: Blas_Int

	jobu1_str := job_to_cstring(compute_u1)
	jobu2_str := job_to_cstring(compute_u2)
	jobv1t_str := job_to_cstring(compute_v1t)

	cuncsd2by1_(
		jobu1_str,
		jobu2_str,
		jobv1t_str,
		&m_i32,
		&p_i32,
		&q_i32,
		raw_data(X11_copy.data),
		&ldx11,
		raw_data(X21_copy.data),
		&ldx21,
		raw_data(theta_data),
		compute_u1 ? raw_data(U1_data) : nil,
		&ldu1,
		compute_u2 ? raw_data(U2_data) : nil,
		&ldu2,
		compute_v1t ? raw_data(V1T_data) : nil,
		&ldv1t,
		&work_query,
		&lwork_query,
		&rwork_query,
		&lrwork_query,
		&iwork_query,
		&info,
		len(jobu1_str),
		len(jobu2_str),
		len(jobv1t_str),
	)

	if info != 0 {
		// Clean up on query failure
		delete(X11_copy.data, allocator)
		delete(X21_copy.data, allocator)
		delete(theta_data, allocator)
		if compute_u1 do delete(U1_data, allocator)
		if compute_u2 do delete(U2_data, allocator)
		if compute_v1t do delete(V1T_data, allocator)
		return {}, .InvalidParameter
	}

	// Allocate workspace
	lwork := Blas_Int(real(work_query))
	lrwork := Blas_Int(rwork_query)
	liwork := iwork_query
	work_data := make([]complex64, lwork, allocator) or_return
	rwork_data := make([]f32, lrwork, allocator) or_return
	iwork_data := make([]i32, liwork, allocator) or_return
	defer {
		delete(work_data, allocator)
		delete(rwork_data, allocator)
		delete(iwork_data, allocator)
	}

	// Perform 2-by-1 CS decomposition
	cuncsd2by1_(
		jobu1_str,
		jobu2_str,
		jobv1t_str,
		&m_i32,
		&p_i32,
		&q_i32,
		raw_data(X11_copy.data),
		&ldx11,
		raw_data(X21_copy.data),
		&ldx21,
		raw_data(theta_data),
		compute_u1 ? raw_data(U1_data) : nil,
		&ldu1,
		compute_u2 ? raw_data(U2_data) : nil,
		&ldu2,
		compute_v1t ? raw_data(V1T_data) : nil,
		&ldv1t,
		raw_data(work_data),
		&lwork,
		raw_data(rwork_data),
		&lrwork,
		raw_data(iwork_data),
		&info,
		len(jobu1_str),
		len(jobu2_str),
		len(jobv1t_str),
	)

	if info != 0 {
		// Clean up on decomposition failure
		delete(X11_copy.data, allocator)
		delete(X21_copy.data, allocator)
		delete(theta_data, allocator)
		if compute_u1 do delete(U1_data, allocator)
		if compute_u2 do delete(U2_data, allocator)
		if compute_v1t do delete(V1T_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.X11 = X11_copy
	result.X21 = X21_copy
	result.theta = theta_data

	if compute_u1 {
		result.U1 = Matrix(complex64) {
			data = U1_data,
			rows = p,
			cols = p,
		}
	}
	if compute_u2 {
		result.U2 = Matrix(complex64) {
			data = U2_data,
			rows = m - p,
			cols = m - p,
		}
	}
	if compute_v1t {
		result.V1T = Matrix(complex64) {
			data = V1T_data,
			rows = q,
			cols = q,
		}
	}

	result.decomposition_successful = true
	return
}

// Complex double precision
zuncsd2by1 :: proc(
	X11: Matrix(complex128),
	X21: Matrix(complex128),
	compute_u1: bool = true,
	compute_u2: bool = true,
	compute_v1t: bool = true,
	allocator := context.allocator,
) -> (
	result: CSDecomposition2by1Result(complex128, f64),
	err: LapackError,
) {
	// Validate dimensions
	m := X11.rows + X21.rows
	p := X11.rows
	q := X11.cols

	if X11.cols != X21.cols {
		return {}, .InvalidDimension
	}

	// Create working copies
	X11_copy := matrix_copy(X11, allocator) or_return
	X21_copy := matrix_copy(X21, allocator) or_return

	// Allocate output matrices
	min_pq := min(p, q)
	theta_data := make([]f64, min_pq, allocator) or_return

	U1_data: []complex128
	U2_data: []complex128
	V1T_data: []complex128

	if compute_u1 {
		U1_data = make([]complex128, p * p, allocator) or_return
	}
	if compute_u2 {
		U2_data = make([]complex128, (m - p) * (m - p), allocator) or_return
	}
	if compute_v1t {
		V1T_data = make([]complex128, q * q, allocator) or_return
	}

	// Query workspace sizes
	work_query: complex128
	rwork_query: f64
	lwork_query := Blas_Int(-1)
	lrwork_query := Blas_Int(-1)
	m_i32, p_i32, q_i32 := Blas_Int(m), Blas_Int(p), Blas_Int(q)
	ldx11, ldx21 := Blas_Int(X11.rows), Blas_Int(X21.rows)
	ldu1, ldu2 := Blas_Int(p), Blas_Int(m - p)
	ldv1t := Blas_Int(q)
	info: Info
	iwork_query: Blas_Int

	jobu1_str := job_to_cstring(compute_u1)
	jobu2_str := job_to_cstring(compute_u2)
	jobv1t_str := job_to_cstring(compute_v1t)

	zuncsd2by1_(
		jobu1_str,
		jobu2_str,
		jobv1t_str,
		&m_i32,
		&p_i32,
		&q_i32,
		raw_data(X11_copy.data),
		&ldx11,
		raw_data(X21_copy.data),
		&ldx21,
		raw_data(theta_data),
		compute_u1 ? raw_data(U1_data) : nil,
		&ldu1,
		compute_u2 ? raw_data(U2_data) : nil,
		&ldu2,
		compute_v1t ? raw_data(V1T_data) : nil,
		&ldv1t,
		&work_query,
		&lwork_query,
		&rwork_query,
		&lrwork_query,
		&iwork_query,
		&info,
		len(jobu1_str),
		len(jobu2_str),
		len(jobv1t_str),
	)

	if info != 0 {
		// Clean up on query failure
		delete(X11_copy.data, allocator)
		delete(X21_copy.data, allocator)
		delete(theta_data, allocator)
		if compute_u1 do delete(U1_data, allocator)
		if compute_u2 do delete(U2_data, allocator)
		if compute_v1t do delete(V1T_data, allocator)
		return {}, .InvalidParameter
	}

	// Allocate workspace
	lwork := Blas_Int(real(work_query))
	lrwork := Blas_Int(rwork_query)
	liwork := iwork_query
	work_data := make([]complex128, lwork, allocator) or_return
	rwork_data := make([]f64, lrwork, allocator) or_return
	iwork_data := make([]i32, liwork, allocator) or_return
	defer {
		delete(work_data, allocator)
		delete(rwork_data, allocator)
		delete(iwork_data, allocator)
	}

	// Perform 2-by-1 CS decomposition
	zuncsd2by1_(
		jobu1_str,
		jobu2_str,
		jobv1t_str,
		&m_i32,
		&p_i32,
		&q_i32,
		raw_data(X11_copy.data),
		&ldx11,
		raw_data(X21_copy.data),
		&ldx21,
		raw_data(theta_data),
		compute_u1 ? raw_data(U1_data) : nil,
		&ldu1,
		compute_u2 ? raw_data(U2_data) : nil,
		&ldu2,
		compute_v1t ? raw_data(V1T_data) : nil,
		&ldv1t,
		raw_data(work_data),
		&lwork,
		raw_data(rwork_data),
		&lrwork,
		raw_data(iwork_data),
		&info,
		len(jobu1_str),
		len(jobu2_str),
		len(jobv1t_str),
	)

	if info != 0 {
		// Clean up on decomposition failure
		delete(X11_copy.data, allocator)
		delete(X21_copy.data, allocator)
		delete(theta_data, allocator)
		if compute_u1 do delete(U1_data, allocator)
		if compute_u2 do delete(U2_data, allocator)
		if compute_v1t do delete(V1T_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.X11 = X11_copy
	result.X21 = X21_copy
	result.theta = theta_data

	if compute_u1 {
		result.U1 = Matrix(complex128) {
			data = U1_data,
			rows = p,
			cols = p,
		}
	}
	if compute_u2 {
		result.U2 = Matrix(complex128) {
			data = U2_data,
			rows = m - p,
			cols = m - p,
		}
	}
	if compute_v1t {
		result.V1T = Matrix(complex128) {
			data = V1T_data,
			rows = q,
			cols = q,
		}
	}

	result.decomposition_successful = true
	return
}

// ==============================================================================
// Unitary Matrix Generation Functions (UNG*)
// ==============================================================================

// Generate unitary matrices from various matrix factorizations
// These functions construct explicit unitary matrices from factorization results

// Enum for bidiagonal reduction matrix type
BidiagonalMatrixType :: enum {
	LeftVectors, // Generate P (left vectors)
	RightVectors, // Generate Q (right vectors)
}

bidiagonal_matrix_type_to_cstring :: proc(vect: BidiagonalMatrixType) -> cstring {
	switch vect {
	case .LeftVectors:
		return "P"
	case .RightVectors:
		return "Q"
	}
	return "P"
}

// Result structure for unitary matrix generation
UnitaryMatrixResult :: struct($T: typeid) {
	mtrx:                  Matrix(T),
	generation_successful: bool,
}

// ==============================================================================
// UNGBR - Generate unitary matrix from bidiagonal reduction
// ==============================================================================

// Complex single precision
cungbr :: proc(
	A: Matrix(complex64),
	tau: []complex64,
	vect: BidiagonalMatrixType = .LeftVectors,
	k: int = -1, // Number of elementary reflectors (-1 for auto-detect)
	allocator := context.allocator,
) -> (
	result: UnitaryMatrixResult(complex64),
	err: LapackError,
) {
	m, n := A.rows, A.cols
	k_actual := k if k >= 0 else min(m, n)

	// Create working copy
	A_copy := matrix_copy(A, allocator) or_return

	// Query workspace
	work_query: complex64
	lwork_query := Blas_Int(-1)
	m_i32, n_i32, k_i32 := Blas_Int(m), Blas_Int(n), Blas_Int(k_actual)
	lda := Blas_Int(A.rows)
	info: Info
	vect_str := bidiagonal_matrix_type_to_cstring(vect)

	cungbr_(
		vect_str,
		&m_i32,
		&n_i32,
		&k_i32,
		raw_data(A_copy.data),
		&lda,
		raw_data(tau),
		&work_query,
		&lwork_query,
		&info,
		len(vect_str),
	)

	if info != 0 {
		delete(A_copy.data, allocator)
		return {}, .InvalidParameter
	}

	lwork := Blas_Int(real(work_query))
	work_data := make([]complex64, lwork, allocator) or_return
	defer delete(work_data, allocator)

	cungbr_(
		vect_str,
		&m_i32,
		&n_i32,
		&k_i32,
		raw_data(A_copy.data),
		&lda,
		raw_data(tau),
		raw_data(work_data),
		&lwork,
		&info,
		len(vect_str),
	)

	if info != 0 {
		delete(A_copy.data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.mtrx = A_copy
	result.generation_successful = true
	return
}

// Complex double precision
zungbr :: proc(
	A: Matrix(complex128),
	tau: []complex128,
	vect: BidiagonalMatrixType = .LeftVectors,
	k: int = -1,
	allocator := context.allocator,
) -> (
	result: UnitaryMatrixResult(complex128),
	err: LapackError,
) {
	m, n := A.rows, A.cols
	k_actual := k if k >= 0 else min(m, n)

	A_copy := matrix_copy(A, allocator) or_return

	work_query: complex128
	lwork_query := Blas_Int(-1)
	m_i32, n_i32, k_i32 := Blas_Int(m), Blas_Int(n), Blas_Int(k_actual)
	lda := Blas_Int(A.rows)
	info: Info
	vect_str := bidiagonal_matrix_type_to_cstring(vect)

	zungbr_(
		vect_str,
		&m_i32,
		&n_i32,
		&k_i32,
		raw_data(A_copy.data),
		&lda,
		raw_data(tau),
		&work_query,
		&lwork_query,
		&info,
		len(vect_str),
	)

	if info != 0 {
		delete(A_copy.data, allocator)
		return {}, .InvalidParameter
	}

	lwork := Blas_Int(real(work_query))
	work_data := make([]complex128, lwork, allocator) or_return
	defer delete(work_data, allocator)

	zungbr_(
		vect_str,
		&m_i32,
		&n_i32,
		&k_i32,
		raw_data(A_copy.data),
		&lda,
		raw_data(tau),
		raw_data(work_data),
		&lwork,
		&info,
		len(vect_str),
	)

	if info != 0 {
		delete(A_copy.data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.mtrx = A_copy
	result.generation_successful = true
	return
}

// ==============================================================================
// UNGHR - Generate unitary matrix from Hessenberg reduction
// ==============================================================================

// Complex single precision
cunghr :: proc(
	A: Matrix(complex64),
	tau: []complex64,
	ilo: int = 1,
	ihi: int = -1, // -1 for auto-detect (use n)
	allocator := context.allocator,
) -> (
	result: UnitaryMatrixResult(complex64),
	err: LapackError,
) {
	n := A.rows
	if A.cols != n {
		return {}, .InvalidDimension
	}

	ihi_actual := ihi if ihi > 0 else n
	A_copy := matrix_copy(A, allocator) or_return

	work_query: complex64
	lwork_query := Blas_Int(-1)
	n_i32, ilo_i32, ihi_i32 := Blas_Int(n), Blas_Int(ilo), Blas_Int(ihi_actual)
	lda := Blas_Int(A.rows)
	info: Info

	cunghr_(
		&n_i32,
		&ilo_i32,
		&ihi_i32,
		raw_data(A_copy.data),
		&lda,
		raw_data(tau),
		&work_query,
		&lwork_query,
		&info,
	)

	if info != 0 {
		delete(A_copy.data, allocator)
		return {}, .InvalidParameter
	}

	lwork := Blas_Int(real(work_query))
	work_data := make([]complex64, lwork, allocator) or_return
	defer delete(work_data, allocator)

	cunghr_(
		&n_i32,
		&ilo_i32,
		&ihi_i32,
		raw_data(A_copy.data),
		&lda,
		raw_data(tau),
		raw_data(work_data),
		&lwork,
		&info,
	)

	if info != 0 {
		delete(A_copy.data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.mtrx = A_copy
	result.generation_successful = true
	return
}

// Complex double precision
zunghr :: proc(
	A: Matrix(complex128),
	tau: []complex128,
	ilo: int = 1,
	ihi: int = -1,
	allocator := context.allocator,
) -> (
	result: UnitaryMatrixResult(complex128),
	err: LapackError,
) {
	n := A.rows
	if A.cols != n {
		return {}, .InvalidDimension
	}

	ihi_actual := ihi if ihi > 0 else n
	A_copy := matrix_copy(A, allocator) or_return

	work_query: complex128
	lwork_query := Blas_Int(-1)
	n_i32, ilo_i32, ihi_i32 := Blas_Int(n), Blas_Int(ilo), Blas_Int(ihi_actual)
	lda := Blas_Int(A.rows)
	info: Info

	zunghr_(
		&n_i32,
		&ilo_i32,
		&ihi_i32,
		raw_data(A_copy.data),
		&lda,
		raw_data(tau),
		&work_query,
		&lwork_query,
		&info,
	)

	if info != 0 {
		delete(A_copy.data, allocator)
		return {}, .InvalidParameter
	}

	lwork := Blas_Int(real(work_query))
	work_data := make([]complex128, lwork, allocator) or_return
	defer delete(work_data, allocator)

	zunghr_(
		&n_i32,
		&ilo_i32,
		&ihi_i32,
		raw_data(A_copy.data),
		&lda,
		raw_data(tau),
		raw_data(work_data),
		&lwork,
		&info,
	)

	if info != 0 {
		delete(A_copy.data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.mtrx = A_copy
	result.generation_successful = true
	return
}

// ==============================================================================
// UNGLQ - Generate unitary matrix from LQ factorization
// ==============================================================================

// Complex single precision
cunglq :: proc(
	A: Matrix(complex64),
	tau: []complex64,
	k: int = -1, // Number of elementary reflectors (-1 for auto-detect)
	allocator := context.allocator,
) -> (
	result: UnitaryMatrixResult(complex64),
	err: LapackError,
) {
	m, n := A.rows, A.cols
	k_actual := k if k >= 0 else min(m, n)

	A_copy := matrix_copy(A, allocator) or_return

	work_query: complex64
	lwork_query := Blas_Int(-1)
	m_i32, n_i32, k_i32 := Blas_Int(m), Blas_Int(n), Blas_Int(k_actual)
	lda := Blas_Int(A.rows)
	info: Info

	cunglq_(
		&m_i32,
		&n_i32,
		&k_i32,
		raw_data(A_copy.data),
		&lda,
		raw_data(tau),
		&work_query,
		&lwork_query,
		&info,
	)

	if info != 0 {
		delete(A_copy.data, allocator)
		return {}, .InvalidParameter
	}

	lwork := Blas_Int(real(work_query))
	work_data := make([]complex64, lwork, allocator) or_return
	defer delete(work_data, allocator)

	cunglq_(
		&m_i32,
		&n_i32,
		&k_i32,
		raw_data(A_copy.data),
		&lda,
		raw_data(tau),
		raw_data(work_data),
		&lwork,
		&info,
	)

	if info != 0 {
		delete(A_copy.data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.mtrx = A_copy
	result.generation_successful = true
	return
}

// Complex double precision
zunglq :: proc(
	A: Matrix(complex128),
	tau: []complex128,
	k: int = -1,
	allocator := context.allocator,
) -> (
	result: UnitaryMatrixResult(complex128),
	err: LapackError,
) {
	m, n := A.rows, A.cols
	k_actual := k if k >= 0 else min(m, n)

	A_copy := matrix_copy(A, allocator) or_return

	work_query: complex128
	lwork_query := Blas_Int(-1)
	m_i32, n_i32, k_i32 := Blas_Int(m), Blas_Int(n), Blas_Int(k_actual)
	lda := Blas_Int(A.rows)
	info: Info

	zunglq_(
		&m_i32,
		&n_i32,
		&k_i32,
		raw_data(A_copy.data),
		&lda,
		raw_data(tau),
		&work_query,
		&lwork_query,
		&info,
	)

	if info != 0 {
		delete(A_copy.data, allocator)
		return {}, .InvalidParameter
	}

	lwork := Blas_Int(real(work_query))
	work_data := make([]complex128, lwork, allocator) or_return
	defer delete(work_data, allocator)

	zunglq_(
		&m_i32,
		&n_i32,
		&k_i32,
		raw_data(A_copy.data),
		&lda,
		raw_data(tau),
		raw_data(work_data),
		&lwork,
		&info,
	)

	if info != 0 {
		delete(A_copy.data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.mtrx = A_copy
	result.generation_successful = true
	return
}

// ==============================================================================
// UNGQL - Generate unitary matrix from QL factorization
// ==============================================================================

// Complex single precision
cungql :: proc(
	A: Matrix(complex64),
	tau: []complex64,
	k: int = -1, // Number of elementary reflectors (-1 for auto-detect)
	allocator := context.allocator,
) -> (
	result: UnitaryMatrixResult(complex64),
	err: LapackError,
) {
	m, n := A.rows, A.cols
	k_actual := k if k >= 0 else min(m, n)

	A_copy := matrix_copy(A, allocator) or_return

	work_query: complex64
	lwork_query := Blas_Int(-1)
	m_i32, n_i32, k_i32 := Blas_Int(m), Blas_Int(n), Blas_Int(k_actual)
	lda := Blas_Int(A.rows)
	info: Info

	cungql_(
		&m_i32,
		&n_i32,
		&k_i32,
		raw_data(A_copy.data),
		&lda,
		raw_data(tau),
		&work_query,
		&lwork_query,
		&info,
	)

	if info != 0 {
		delete(A_copy.data, allocator)
		return {}, .InvalidParameter
	}

	lwork := Blas_Int(real(work_query))
	work_data := make([]complex64, lwork, allocator) or_return
	defer delete(work_data, allocator)

	cungql_(
		&m_i32,
		&n_i32,
		&k_i32,
		raw_data(A_copy.data),
		&lda,
		raw_data(tau),
		raw_data(work_data),
		&lwork,
		&info,
	)

	if info != 0 {
		delete(A_copy.data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.mtrx = A_copy
	result.generation_successful = true
	return
}

// Complex double precision
zungql :: proc(
	A: Matrix(complex128),
	tau: []complex128,
	k: int = -1,
	allocator := context.allocator,
) -> (
	result: UnitaryMatrixResult(complex128),
	err: LapackError,
) {
	m, n := A.rows, A.cols
	k_actual := k if k >= 0 else min(m, n)

	A_copy := matrix_copy(A, allocator) or_return

	work_query: complex128
	lwork_query := Blas_Int(-1)
	m_i32, n_i32, k_i32 := Blas_Int(m), Blas_Int(n), Blas_Int(k_actual)
	lda := Blas_Int(A.rows)
	info: Info

	zungql_(
		&m_i32,
		&n_i32,
		&k_i32,
		raw_data(A_copy.data),
		&lda,
		raw_data(tau),
		&work_query,
		&lwork_query,
		&info,
	)

	if info != 0 {
		delete(A_copy.data, allocator)
		return {}, .InvalidParameter
	}

	lwork := Blas_Int(real(work_query))
	work_data := make([]complex128, lwork, allocator) or_return
	defer delete(work_data, allocator)

	zungql_(
		&m_i32,
		&n_i32,
		&k_i32,
		raw_data(A_copy.data),
		&lda,
		raw_data(tau),
		raw_data(work_data),
		&lwork,
		&info,
	)

	if info != 0 {
		delete(A_copy.data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.mtrx = A_copy
	result.generation_successful = true
	return
}

// ==============================================================================
// Final Convenience Overloads for Matrix Generation Functions
// ==============================================================================

// 2-by-1 CS decomposition overloads
uncsd2by1 :: proc {
	cuncsd2by1,
	zuncsd2by1,
}

// Unitary matrix generation overloads
ungbr :: proc {
	cungbr,
	zungbr,
} // From bidiagonal reduction
unghr :: proc {
	cunghr,
	zunghr,
} // From Hessenberg reduction
unglq :: proc {
	cunglq,
	zunglq,
} // From LQ factorization
ungql :: proc {
	cungql,
	zungql,
} // From QL factorization

// ==============================================================================
// Additional Unitary Matrix Generation Functions
// ==============================================================================

// Additional LAPACK wrappers for QR-related operations and advanced factorizations

// Low-level LAPACK wrappers
foreign openblas {
	cungqr_ :: proc(m: ^i32, n: ^i32, k: ^i32, A: ^complex64, lda: ^i32, tau: ^complex64, work: ^complex64, lwork: ^i32, info: ^i32) ---
	zungqr_ :: proc(m: ^i32, n: ^i32, k: ^i32, A: ^complex128, lda: ^i32, tau: ^complex128, work: ^complex128, lwork: ^i32, info: ^i32) ---
	cungrq_ :: proc(m: ^i32, n: ^i32, k: ^i32, A: ^complex64, lda: ^i32, tau: ^complex64, work: ^complex64, lwork: ^i32, info: ^i32) ---
	zungrq_ :: proc(m: ^i32, n: ^i32, k: ^i32, A: ^complex128, lda: ^i32, tau: ^complex128, work: ^complex128, lwork: ^i32, info: ^i32) ---
	cungtr_ :: proc(uplo: cstring, n: ^i32, A: ^complex64, lda: ^i32, tau: ^complex64, work: ^complex64, lwork: ^i32, info: ^i32, _: c.size_t) ---
	zungtr_ :: proc(uplo: cstring, n: ^i32, A: ^complex128, lda: ^i32, tau: ^complex128, work: ^complex128, lwork: ^i32, info: ^i32, _: c.size_t) ---
	cungtsqr_row_ :: proc(m: ^i32, n: ^i32, mb: ^i32, nb: ^i32, A: ^complex64, lda: ^i32, T: ^complex64, ldt: ^i32, work: ^complex64, lwork: ^i32, info: ^i32) ---
	zungtsqr_row_ :: proc(m: ^i32, n: ^i32, mb: ^i32, nb: ^i32, A: ^complex128, lda: ^i32, T: ^complex128, ldt: ^i32, work: ^complex128, lwork: ^i32, info: ^i32) ---
	cunhr_col_ :: proc(m: ^i32, n: ^i32, nb: ^i32, A: ^complex64, lda: ^i32, T: ^complex64, ldt: ^i32, D: ^complex64, info: ^i32) ---
	zunhr_col_ :: proc(m: ^i32, n: ^i32, nb: ^i32, A: ^complex128, lda: ^i32, T: ^complex128, ldt: ^i32, D: ^complex128, info: ^i32) ---
}

// ==============================================================================
// UNGQR - Generate unitary matrix from QR factorization
// ==============================================================================

// Complex single precision
cungqr :: proc(
	A: Matrix(complex64),
	tau: []complex64,
	k: int = -1, // Number of elementary reflectors (-1 for auto-detect)
	allocator := context.allocator,
) -> (
	result: UnitaryMatrixResult(complex64),
	err: LapackError,
) {
	m, n := A.rows, A.cols
	k_actual := k if k >= 0 else min(m, n)

	A_copy := matrix_copy(A, allocator) or_return

	work_query: complex64
	lwork_query := Blas_Int(-1)
	m_i32, n_i32, k_i32 := Blas_Int(m), Blas_Int(n), Blas_Int(k_actual)
	lda := Blas_Int(A.rows)
	info: Info

	cungqr_(
		&m_i32,
		&n_i32,
		&k_i32,
		raw_data(A_copy.data),
		&lda,
		raw_data(tau),
		&work_query,
		&lwork_query,
		&info,
	)

	if info != 0 {
		delete(A_copy.data, allocator)
		return {}, .InvalidParameter
	}

	lwork := Blas_Int(real(work_query))
	work_data := make([]complex64, lwork, allocator) or_return
	defer delete(work_data, allocator)

	cungqr_(
		&m_i32,
		&n_i32,
		&k_i32,
		raw_data(A_copy.data),
		&lda,
		raw_data(tau),
		raw_data(work_data),
		&lwork,
		&info,
	)

	if info != 0 {
		delete(A_copy.data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.mtrx = A_copy
	result.generation_successful = true
	return
}

// Complex double precision
zungqr :: proc(
	A: Matrix(complex128),
	tau: []complex128,
	k: int = -1,
	allocator := context.allocator,
) -> (
	result: UnitaryMatrixResult(complex128),
	err: LapackError,
) {
	m, n := A.rows, A.cols
	k_actual := k if k >= 0 else min(m, n)

	A_copy := matrix_copy(A, allocator) or_return

	work_query: complex128
	lwork_query := Blas_Int(-1)
	m_i32, n_i32, k_i32 := Blas_Int(m), Blas_Int(n), Blas_Int(k_actual)
	lda := Blas_Int(A.rows)
	info: Info

	zungqr_(
		&m_i32,
		&n_i32,
		&k_i32,
		raw_data(A_copy.data),
		&lda,
		raw_data(tau),
		&work_query,
		&lwork_query,
		&info,
	)

	if info != 0 {
		delete(A_copy.data, allocator)
		return {}, .InvalidParameter
	}

	lwork := Blas_Int(real(work_query))
	work_data := make([]complex128, lwork, allocator) or_return
	defer delete(work_data, allocator)

	zungqr_(
		&m_i32,
		&n_i32,
		&k_i32,
		raw_data(A_copy.data),
		&lda,
		raw_data(tau),
		raw_data(work_data),
		&lwork,
		&info,
	)

	if info != 0 {
		delete(A_copy.data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.mtrx = A_copy
	result.generation_successful = true
	return
}

// ==============================================================================
// UNGRQ - Generate unitary matrix from RQ factorization
// ==============================================================================

// Complex single precision
cungrq :: proc(
	A: Matrix(complex64),
	tau: []complex64,
	k: int = -1, // Number of elementary reflectors (-1 for auto-detect)
	allocator := context.allocator,
) -> (
	result: UnitaryMatrixResult(complex64),
	err: LapackError,
) {
	m, n := A.rows, A.cols
	k_actual := k if k >= 0 else min(m, n)

	A_copy := matrix_copy(A, allocator) or_return

	work_query: complex64
	lwork_query := Blas_Int(-1)
	m_i32, n_i32, k_i32 := Blas_Int(m), Blas_Int(n), Blas_Int(k_actual)
	lda := Blas_Int(A.rows)
	info: Info

	cungrq_(
		&m_i32,
		&n_i32,
		&k_i32,
		raw_data(A_copy.data),
		&lda,
		raw_data(tau),
		&work_query,
		&lwork_query,
		&info,
	)

	if info != 0 {
		delete(A_copy.data, allocator)
		return {}, .InvalidParameter
	}

	lwork := Blas_Int(real(work_query))
	work_data := make([]complex64, lwork, allocator) or_return
	defer delete(work_data, allocator)

	cungrq_(
		&m_i32,
		&n_i32,
		&k_i32,
		raw_data(A_copy.data),
		&lda,
		raw_data(tau),
		raw_data(work_data),
		&lwork,
		&info,
	)

	if info != 0 {
		delete(A_copy.data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.mtrx = A_copy
	result.generation_successful = true
	return
}

// Complex double precision
zungrq :: proc(
	A: Matrix(complex128),
	tau: []complex128,
	k: int = -1,
	allocator := context.allocator,
) -> (
	result: UnitaryMatrixResult(complex128),
	err: LapackError,
) {
	m, n := A.rows, A.cols
	k_actual := k if k >= 0 else min(m, n)

	A_copy := matrix_copy(A, allocator) or_return

	work_query: complex128
	lwork_query := Blas_Int(-1)
	m_i32, n_i32, k_i32 := Blas_Int(m), Blas_Int(n), Blas_Int(k_actual)
	lda := Blas_Int(A.rows)
	info: Info

	zungrq_(
		&m_i32,
		&n_i32,
		&k_i32,
		raw_data(A_copy.data),
		&lda,
		raw_data(tau),
		&work_query,
		&lwork_query,
		&info,
	)

	if info != 0 {
		delete(A_copy.data, allocator)
		return {}, .InvalidParameter
	}

	lwork := Blas_Int(real(work_query))
	work_data := make([]complex128, lwork, allocator) or_return
	defer delete(work_data, allocator)

	zungrq_(
		&m_i32,
		&n_i32,
		&k_i32,
		raw_data(A_copy.data),
		&lda,
		raw_data(tau),
		raw_data(work_data),
		&lwork,
		&info,
	)

	if info != 0 {
		delete(A_copy.data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.mtrx = A_copy
	result.generation_successful = true
	return
}

// ==============================================================================
// UNGTR - Generate unitary matrix from tridiagonal reduction
// ==============================================================================

// Complex single precision
cungtr :: proc(
	A: Matrix(complex64),
	tau: []complex64,
	uplo: MatrixTriangle = .Upper,
	allocator := context.allocator,
) -> (
	result: UnitaryMatrixResult(complex64),
	err: LapackError,
) {
	n := A.rows
	if A.cols != n {
		return {}, .InvalidDimension
	}

	A_copy := matrix_copy(A, allocator) or_return

	work_query: complex64
	lwork_query := Blas_Int(-1)
	n_i32 := Blas_Int(n)
	lda := Blas_Int(A.rows)
	info: Info
	uplo_str := matrix_triangle_to_cstring(uplo)

	cungtr_(
		uplo_str,
		&n_i32,
		raw_data(A_copy.data),
		&lda,
		raw_data(tau),
		&work_query,
		&lwork_query,
		&info,
		len(uplo_str),
	)

	if info != 0 {
		delete(A_copy.data, allocator)
		return {}, .InvalidParameter
	}

	lwork := Blas_Int(real(work_query))
	work_data := make([]complex64, lwork, allocator) or_return
	defer delete(work_data, allocator)

	cungtr_(
		uplo_str,
		&n_i32,
		raw_data(A_copy.data),
		&lda,
		raw_data(tau),
		raw_data(work_data),
		&lwork,
		&info,
		len(uplo_str),
	)

	if info != 0 {
		delete(A_copy.data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.mtrx = A_copy
	result.generation_successful = true
	return
}

// Complex double precision
zungtr :: proc(
	A: Matrix(complex128),
	tau: []complex128,
	uplo: MatrixTriangle = .Upper,
	allocator := context.allocator,
) -> (
	result: UnitaryMatrixResult(complex128),
	err: LapackError,
) {
	n := A.rows
	if A.cols != n {
		return {}, .InvalidDimension
	}

	A_copy := matrix_copy(A, allocator) or_return

	work_query: complex128
	lwork_query := Blas_Int(-1)
	n_i32 := Blas_Int(n)
	lda := Blas_Int(A.rows)
	info: Info
	uplo_str := matrix_triangle_to_cstring(uplo)

	zungtr_(
		uplo_str,
		&n_i32,
		raw_data(A_copy.data),
		&lda,
		raw_data(tau),
		&work_query,
		&lwork_query,
		&info,
		len(uplo_str),
	)

	if info != 0 {
		delete(A_copy.data, allocator)
		return {}, .InvalidParameter
	}

	lwork := Blas_Int(real(work_query))
	work_data := make([]complex128, lwork, allocator) or_return
	defer delete(work_data, allocator)

	zungtr_(
		uplo_str,
		&n_i32,
		raw_data(A_copy.data),
		&lda,
		raw_data(tau),
		raw_data(work_data),
		&lwork,
		&info,
		len(uplo_str),
	)

	if info != 0 {
		delete(A_copy.data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.mtrx = A_copy
	result.generation_successful = true
	return
}

// ==============================================================================
// UNGTSQR_ROW - Generate unitary matrix from tall-skinny QR
// ==============================================================================

// Result structure for tall-skinny QR generation
TallSkinnyQRResult :: struct($T: typeid) {
	Q_matrix:              Matrix(T), // Generated unitary matrix
	generation_successful: bool,
}

// Complex single precision
cungtsqr_row :: proc(
	A: Matrix(complex64),
	T_matrix: Matrix(complex64),
	mb: int = 32, // Row block size
	nb: int = 32, // Column block size
	allocator := context.allocator,
) -> (
	result: TallSkinnyQRResult(complex64),
	err: LapackError,
) {
	m, n := A.rows, A.cols

	A_copy := matrix_copy(A, allocator) or_return
	T_copy := matrix_copy(T_matrix, allocator) or_return

	work_query: complex64
	lwork_query := Blas_Int(-1)
	m_i32, n_i32 := Blas_Int(m), Blas_Int(n)
	mb_i32, nb_i32 := Blas_Int(mb), Blas_Int(nb)
	lda := Blas_Int(A.rows)
	ldt := Blas_Int(T_matrix.rows)
	info: Info

	cungtsqr_row_(
		&m_i32,
		&n_i32,
		&mb_i32,
		&nb_i32,
		raw_data(A_copy.data),
		&lda,
		raw_data(T_copy.data),
		&ldt,
		&work_query,
		&lwork_query,
		&info,
	)

	if info != 0 {
		delete(A_copy.data, allocator)
		delete(T_copy.data, allocator)
		return {}, .InvalidParameter
	}

	lwork := Blas_Int(real(work_query))
	work_data := make([]complex64, lwork, allocator) or_return
	defer delete(work_data, allocator)

	cungtsqr_row_(
		&m_i32,
		&n_i32,
		&mb_i32,
		&nb_i32,
		raw_data(A_copy.data),
		&lda,
		raw_data(T_copy.data),
		&ldt,
		raw_data(work_data),
		&lwork,
		&info,
	)

	delete(T_copy.data, allocator) // T matrix not needed in result

	if info != 0 {
		delete(A_copy.data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.Q_matrix = A_copy
	result.generation_successful = true
	return
}

// Complex double precision
zungtsqr_row :: proc(
	A: Matrix(complex128),
	T_matrix: Matrix(complex128),
	mb: int = 32,
	nb: int = 32,
	allocator := context.allocator,
) -> (
	result: TallSkinnyQRResult(complex128),
	err: LapackError,
) {
	m, n := A.rows, A.cols

	A_copy := matrix_copy(A, allocator) or_return
	T_copy := matrix_copy(T_matrix, allocator) or_return

	work_query: complex128
	lwork_query := Blas_Int(-1)
	m_i32, n_i32 := Blas_Int(m), Blas_Int(n)
	mb_i32, nb_i32 := Blas_Int(mb), Blas_Int(nb)
	lda := Blas_Int(A.rows)
	ldt := Blas_Int(T_matrix.rows)
	info: Info

	zungtsqr_row_(
		&m_i32,
		&n_i32,
		&mb_i32,
		&nb_i32,
		raw_data(A_copy.data),
		&lda,
		raw_data(T_copy.data),
		&ldt,
		&work_query,
		&lwork_query,
		&info,
	)

	if info != 0 {
		delete(A_copy.data, allocator)
		delete(T_copy.data, allocator)
		return {}, .InvalidParameter
	}

	lwork := Blas_Int(real(work_query))
	work_data := make([]complex128, lwork, allocator) or_return
	defer delete(work_data, allocator)

	zungtsqr_row_(
		&m_i32,
		&n_i32,
		&mb_i32,
		&nb_i32,
		raw_data(A_copy.data),
		&lda,
		raw_data(T_copy.data),
		&ldt,
		raw_data(work_data),
		&lwork,
		&info,
	)

	delete(T_copy.data, allocator) // T matrix not needed in result

	if info != 0 {
		delete(A_copy.data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.Q_matrix = A_copy
	result.generation_successful = true
	return
}

// ==============================================================================
// UNHR_COL - Householder reconstruction by columns
// ==============================================================================

// Result structure for Householder reconstruction
HouseholderReconstructionResult :: struct($T: typeid) {
	A_reconstructed:           Matrix(T), // Reconstructed matrix
	T_factor:                  Matrix(T), // T factor matrix
	D_vector:                  []T, // Diagonal scaling vector
	reconstruction_successful: bool,
}

// Complex single precision
cunhr_col :: proc(
	A: Matrix(complex64),
	nb: int = 32, // Column block size
	allocator := context.allocator,
) -> (
	result: HouseholderReconstructionResult(complex64),
	err: LapackError,
) {
	m, n := A.rows, A.cols

	A_copy := matrix_copy(A, allocator) or_return

	// Allocate T factor matrix and D vector
	T_data := make([]complex64, nb * n, allocator) or_return
	D_data := make([]complex64, n, allocator) or_return

	m_i32, n_i32 := Blas_Int(m), Blas_Int(n)
	nb_i32 := Blas_Int(nb)
	lda := Blas_Int(A.rows)
	ldt := Blas_Int(nb)
	info: Info

	cunhr_col_(
		&m_i32,
		&n_i32,
		&nb_i32,
		raw_data(A_copy.data),
		&lda,
		raw_data(T_data),
		&ldt,
		raw_data(D_data),
		&info,
	)

	if info != 0 {
		delete(A_copy.data, allocator)
		delete(T_data, allocator)
		delete(D_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.A_reconstructed = A_copy
	result.T_factor = Matrix(complex64) {
		data = T_data,
		rows = nb,
		cols = n,
	}
	result.D_vector = D_data
	result.reconstruction_successful = true
	return
}

// Complex double precision
zunhr_col :: proc(
	A: Matrix(complex128),
	nb: int = 32,
	allocator := context.allocator,
) -> (
	result: HouseholderReconstructionResult(complex128),
	err: LapackError,
) {
	m, n := A.rows, A.cols

	A_copy := matrix_copy(A, allocator) or_return

	// Allocate T factor matrix and D vector
	T_data := make([]complex128, nb * n, allocator) or_return
	D_data := make([]complex128, n, allocator) or_return

	m_i32, n_i32 := Blas_Int(m), Blas_Int(n)
	nb_i32 := Blas_Int(nb)
	lda := Blas_Int(A.rows)
	ldt := Blas_Int(nb)
	info: Info

	zunhr_col_(
		&m_i32,
		&n_i32,
		&nb_i32,
		raw_data(A_copy.data),
		&lda,
		raw_data(T_data),
		&ldt,
		raw_data(D_data),
		&info,
	)

	if info != 0 {
		delete(A_copy.data, allocator)
		delete(T_data, allocator)
		delete(D_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.A_reconstructed = A_copy
	result.T_factor = Matrix(complex128) {
		data = T_data,
		rows = nb,
		cols = n,
	}
	result.D_vector = D_data
	result.reconstruction_successful = true
	return
}

// ==============================================================================
// Additional Convenience Overloads for QR-Related Functions
// ==============================================================================

// QR-related unitary matrix generation overloads
ungqr :: proc {
	cungqr,
	zungqr,
} // From QR factorization
ungrq :: proc {
	cungrq,
	zungrq,
} // From RQ factorization
ungtr :: proc {
	cungtr,
	zungtr,
} // From tridiagonal reduction

// Advanced QR and Householder operations
ungtsqr_row :: proc {
	cungtsqr_row,
	zungtsqr_row,
} // Tall-skinny QR
unhr_col :: proc {
	cunhr_col,
	zunhr_col,
} // Householder reconstruction

// ==============================================================================
// Unitary Matrix Application Functions (UNM*)
// ==============================================================================

// Apply unitary transformations from various factorizations to other matrices
// These functions multiply C by Q or Q^H without explicitly forming Q

// Low-level LAPACK wrappers
foreign openblas {
	cunmbr_ :: proc(vect: cstring, side: cstring, trans: cstring, m: ^i32, n: ^i32, k: ^i32, A: ^complex64, lda: ^i32, tau: ^complex64, C: ^complex64, ldc: ^i32, work: ^complex64, lwork: ^i32, info: ^i32, _: c.size_t, _: c.size_t, _: c.size_t) ---
	zunmbr_ :: proc(vect: cstring, side: cstring, trans: cstring, m: ^i32, n: ^i32, k: ^i32, A: ^complex128, lda: ^i32, tau: ^complex128, C: ^complex128, ldc: ^i32, work: ^complex128, lwork: ^i32, info: ^i32, _: c.size_t, _: c.size_t, _: c.size_t) ---
	cunmhr_ :: proc(side: cstring, trans: cstring, m: ^i32, n: ^i32, ilo: ^i32, ihi: ^i32, A: ^complex64, lda: ^i32, tau: ^complex64, C: ^complex64, ldc: ^i32, work: ^complex64, lwork: ^i32, info: ^i32, _: c.size_t, _: c.size_t) ---
	zunmhr_ :: proc(side: cstring, trans: cstring, m: ^i32, n: ^i32, ilo: ^i32, ihi: ^i32, A: ^complex128, lda: ^i32, tau: ^complex128, C: ^complex128, ldc: ^i32, work: ^complex128, lwork: ^i32, info: ^i32, _: c.size_t, _: c.size_t) ---
	cunmlq_ :: proc(side: cstring, trans: cstring, m: ^i32, n: ^i32, k: ^i32, A: ^complex64, lda: ^i32, tau: ^complex64, C: ^complex64, ldc: ^i32, work: ^complex64, lwork: ^i32, info: ^i32, _: c.size_t, _: c.size_t) ---
	zunmlq_ :: proc(side: cstring, trans: cstring, m: ^i32, n: ^i32, k: ^i32, A: ^complex128, lda: ^i32, tau: ^complex128, C: ^complex128, ldc: ^i32, work: ^complex128, lwork: ^i32, info: ^i32, _: c.size_t, _: c.size_t) ---
	cunmql_ :: proc(side: cstring, trans: cstring, m: ^i32, n: ^i32, k: ^i32, A: ^complex64, lda: ^i32, tau: ^complex64, C: ^complex64, ldc: ^i32, work: ^complex64, lwork: ^i32, info: ^i32, _: c.size_t, _: c.size_t) ---
	zunmql_ :: proc(side: cstring, trans: cstring, m: ^i32, n: ^i32, k: ^i32, A: ^complex128, lda: ^i32, tau: ^complex128, C: ^complex128, ldc: ^i32, work: ^complex128, lwork: ^i32, info: ^i32, _: c.size_t, _: c.size_t) ---
	cunmqr_ :: proc(side: cstring, trans: cstring, m: ^i32, n: ^i32, k: ^i32, A: ^complex64, lda: ^i32, tau: ^complex64, C: ^complex64, ldc: ^i32, work: ^complex64, lwork: ^i32, info: ^i32, _: c.size_t, _: c.size_t) ---
	zunmqr_ :: proc(side: cstring, trans: cstring, m: ^i32, n: ^i32, k: ^i32, A: ^complex128, lda: ^i32, tau: ^complex128, C: ^complex128, ldc: ^i32, work: ^complex128, lwork: ^i32, info: ^i32, _: c.size_t, _: c.size_t) ---
	cunmrq_ :: proc(side: cstring, trans: cstring, m: ^i32, n: ^i32, k: ^i32, A: ^complex64, lda: ^i32, tau: ^complex64, C: ^complex64, ldc: ^i32, work: ^complex64, lwork: ^i32, info: ^i32, _: c.size_t, _: c.size_t) ---
	zunmrq_ :: proc(side: cstring, trans: cstring, m: ^i32, n: ^i32, k: ^i32, A: ^complex128, lda: ^i32, tau: ^complex128, C: ^complex128, ldc: ^i32, work: ^complex128, lwork: ^i32, info: ^i32, _: c.size_t, _: c.size_t) ---
}

// Result structure for matrix transformation operations
MatrixTransformationResult :: struct($T: typeid) {
	result_matrix:             Matrix(T),
	transformation_successful: bool,
}

// Enum for matrix operation side
MatrixSide :: enum {
	Left, // Apply Q from the left: Q * C or Q^H * C
	Right, // Apply Q from the right: C * Q or C * Q^H
}

matrix_side_to_cstring :: proc(side: MatrixSide) -> cstring {
	switch side {
	case .Left:
		return "L"
	case .Right:
		return "R"
	}
	return "L"
}

// ==============================================================================
// UNMBR - Apply unitary matrix from bidiagonal reduction
// ==============================================================================

// Complex single precision
cunmbr :: proc(
	A: Matrix(complex64),
	tau: []complex64,
	C: Matrix(complex64),
	vect: BidiagonalMatrixType = .LeftVectors,
	side: MatrixSide = .Left,
	trans: MatrixTranspose = .None,
	k: int = -1, // Number of elementary reflectors (-1 for auto-detect)
	allocator := context.allocator,
) -> (
	result: MatrixTransformationResult(complex64),
	err: LapackError,
) {
	m, n := C.rows, C.cols
	k_actual := k if k >= 0 else min(A.rows, A.cols)

	C_copy := matrix_copy(C, allocator) or_return

	work_query: complex64
	lwork_query := Blas_Int(-1)
	m_i32, n_i32, k_i32 := Blas_Int(m), Blas_Int(n), Blas_Int(k_actual)
	lda := Blas_Int(A.rows)
	ldc := Blas_Int(C.rows)
	info: Info

	vect_str := bidiagonal_matrix_type_to_cstring(vect)
	side_str := matrix_side_to_cstring(side)
	trans_str := matrix_transpose_to_cstring(trans)

	cunmbr_(
		vect_str,
		side_str,
		trans_str,
		&m_i32,
		&n_i32,
		&k_i32,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(C_copy.data),
		&ldc,
		&work_query,
		&lwork_query,
		&info,
		len(vect_str),
		len(side_str),
		len(trans_str),
	)

	if info != 0 {
		delete(C_copy.data, allocator)
		return {}, .InvalidParameter
	}

	lwork := Blas_Int(real(work_query))
	work_data := make([]complex64, lwork, allocator) or_return
	defer delete(work_data, allocator)

	cunmbr_(
		vect_str,
		side_str,
		trans_str,
		&m_i32,
		&n_i32,
		&k_i32,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(C_copy.data),
		&ldc,
		raw_data(work_data),
		&lwork,
		&info,
		len(vect_str),
		len(side_str),
		len(trans_str),
	)

	if info != 0 {
		delete(C_copy.data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.result_matrix = C_copy
	result.transformation_successful = true
	return
}

// Complex double precision
zunmbr :: proc(
	A: Matrix(complex128),
	tau: []complex128,
	C: Matrix(complex128),
	vect: BidiagonalMatrixType = .LeftVectors,
	side: MatrixSide = .Left,
	trans: MatrixTranspose = .None,
	k: int = -1,
	allocator := context.allocator,
) -> (
	result: MatrixTransformationResult(complex128),
	err: LapackError,
) {
	m, n := C.rows, C.cols
	k_actual := k if k >= 0 else min(A.rows, A.cols)

	C_copy := matrix_copy(C, allocator) or_return

	work_query: complex128
	lwork_query := Blas_Int(-1)
	m_i32, n_i32, k_i32 := Blas_Int(m), Blas_Int(n), Blas_Int(k_actual)
	lda := Blas_Int(A.rows)
	ldc := Blas_Int(C.rows)
	info: Info

	vect_str := bidiagonal_matrix_type_to_cstring(vect)
	side_str := matrix_side_to_cstring(side)
	trans_str := matrix_transpose_to_cstring(trans)

	zunmbr_(
		vect_str,
		side_str,
		trans_str,
		&m_i32,
		&n_i32,
		&k_i32,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(C_copy.data),
		&ldc,
		&work_query,
		&lwork_query,
		&info,
		len(vect_str),
		len(side_str),
		len(trans_str),
	)

	if info != 0 {
		delete(C_copy.data, allocator)
		return {}, .InvalidParameter
	}

	lwork := Blas_Int(real(work_query))
	work_data := make([]complex128, lwork, allocator) or_return
	defer delete(work_data, allocator)

	zunmbr_(
		vect_str,
		side_str,
		trans_str,
		&m_i32,
		&n_i32,
		&k_i32,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(C_copy.data),
		&ldc,
		raw_data(work_data),
		&lwork,
		&info,
		len(vect_str),
		len(side_str),
		len(trans_str),
	)

	if info != 0 {
		delete(C_copy.data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.result_matrix = C_copy
	result.transformation_successful = true
	return
}

// ==============================================================================
// UNMHR - Apply unitary matrix from Hessenberg reduction
// ==============================================================================

// Complex single precision
cunmhr :: proc(
	A: Matrix(complex64),
	tau: []complex64,
	C: Matrix(complex64),
	side: MatrixSide = .Left,
	trans: MatrixTranspose = .None,
	ilo: int = 1,
	ihi: int = -1, // -1 for auto-detect
	allocator := context.allocator,
) -> (
	result: MatrixTransformationResult(complex64),
	err: LapackError,
) {
	m, n := C.rows, C.cols
	ihi_actual := ihi if ihi > 0 else A.rows

	C_copy := matrix_copy(C, allocator) or_return

	work_query: complex64
	lwork_query := Blas_Int(-1)
	m_i32, n_i32 := Blas_Int(m), Blas_Int(n)
	ilo_i32, ihi_i32 := Blas_Int(ilo), Blas_Int(ihi_actual)
	lda := Blas_Int(A.rows)
	ldc := Blas_Int(C.rows)
	info: Info

	side_str := matrix_side_to_cstring(side)
	trans_str := matrix_transpose_to_cstring(trans)

	cunmhr_(
		side_str,
		trans_str,
		&m_i32,
		&n_i32,
		&ilo_i32,
		&ihi_i32,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(C_copy.data),
		&ldc,
		&work_query,
		&lwork_query,
		&info,
		len(side_str),
		len(trans_str),
	)

	if info != 0 {
		delete(C_copy.data, allocator)
		return {}, .InvalidParameter
	}

	lwork := Blas_Int(real(work_query))
	work_data := make([]complex64, lwork, allocator) or_return
	defer delete(work_data, allocator)

	cunmhr_(
		side_str,
		trans_str,
		&m_i32,
		&n_i32,
		&ilo_i32,
		&ihi_i32,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(C_copy.data),
		&ldc,
		raw_data(work_data),
		&lwork,
		&info,
		len(side_str),
		len(trans_str),
	)

	if info != 0 {
		delete(C_copy.data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.result_matrix = C_copy
	result.transformation_successful = true
	return
}

// Complex double precision
zunmhr :: proc(
	A: Matrix(complex128),
	tau: []complex128,
	C: Matrix(complex128),
	side: MatrixSide = .Left,
	trans: MatrixTranspose = .None,
	ilo: int = 1,
	ihi: int = -1,
	allocator := context.allocator,
) -> (
	result: MatrixTransformationResult(complex128),
	err: LapackError,
) {
	m, n := C.rows, C.cols
	ihi_actual := ihi if ihi > 0 else A.rows

	C_copy := matrix_copy(C, allocator) or_return

	work_query: complex128
	lwork_query := Blas_Int(-1)
	m_i32, n_i32 := Blas_Int(m), Blas_Int(n)
	ilo_i32, ihi_i32 := Blas_Int(ilo), Blas_Int(ihi_actual)
	lda := Blas_Int(A.rows)
	ldc := Blas_Int(C.rows)
	info: Info

	side_str := matrix_side_to_cstring(side)
	trans_str := matrix_transpose_to_cstring(trans)

	zunmhr_(
		side_str,
		trans_str,
		&m_i32,
		&n_i32,
		&ilo_i32,
		&ihi_i32,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(C_copy.data),
		&ldc,
		&work_query,
		&lwork_query,
		&info,
		len(side_str),
		len(trans_str),
	)

	if info != 0 {
		delete(C_copy.data, allocator)
		return {}, .InvalidParameter
	}

	lwork := Blas_Int(real(work_query))
	work_data := make([]complex128, lwork, allocator) or_return
	defer delete(work_data, allocator)

	zunmhr_(
		side_str,
		trans_str,
		&m_i32,
		&n_i32,
		&ilo_i32,
		&ihi_i32,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(C_copy.data),
		&ldc,
		raw_data(work_data),
		&lwork,
		&info,
		len(side_str),
		len(trans_str),
	)

	if info != 0 {
		delete(C_copy.data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.result_matrix = C_copy
	result.transformation_successful = true
	return
}

// ==============================================================================
// UNMLQ - Apply unitary matrix from LQ factorization
// ==============================================================================

// Complex single precision
cunmlq :: proc(
	A: Matrix(complex64),
	tau: []complex64,
	C: Matrix(complex64),
	side: MatrixSide = .Left,
	trans: MatrixTranspose = .None,
	k: int = -1,
	allocator := context.allocator,
) -> (
	result: MatrixTransformationResult(complex64),
	err: LapackError,
) {
	m, n := C.rows, C.cols
	k_actual := k if k >= 0 else min(A.rows, A.cols)

	C_copy := matrix_copy(C, allocator) or_return

	work_query: complex64
	lwork_query := Blas_Int(-1)
	m_i32, n_i32, k_i32 := Blas_Int(m), Blas_Int(n), Blas_Int(k_actual)
	lda := Blas_Int(A.rows)
	ldc := Blas_Int(C.rows)
	info: Info

	side_str := matrix_side_to_cstring(side)
	trans_str := matrix_transpose_to_cstring(trans)

	cunmlq_(
		side_str,
		trans_str,
		&m_i32,
		&n_i32,
		&k_i32,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(C_copy.data),
		&ldc,
		&work_query,
		&lwork_query,
		&info,
		len(side_str),
		len(trans_str),
	)

	if info != 0 {
		delete(C_copy.data, allocator)
		return {}, .InvalidParameter
	}

	lwork := Blas_Int(real(work_query))
	work_data := make([]complex64, lwork, allocator) or_return
	defer delete(work_data, allocator)

	cunmlq_(
		side_str,
		trans_str,
		&m_i32,
		&n_i32,
		&k_i32,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(C_copy.data),
		&ldc,
		raw_data(work_data),
		&lwork,
		&info,
		len(side_str),
		len(trans_str),
	)

	if info != 0 {
		delete(C_copy.data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.result_matrix = C_copy
	result.transformation_successful = true
	return
}

// Complex double precision
zunmlq :: proc(
	A: Matrix(complex128),
	tau: []complex128,
	C: Matrix(complex128),
	side: MatrixSide = .Left,
	trans: MatrixTranspose = .None,
	k: int = -1,
	allocator := context.allocator,
) -> (
	result: MatrixTransformationResult(complex128),
	err: LapackError,
) {
	m, n := C.rows, C.cols
	k_actual := k if k >= 0 else min(A.rows, A.cols)

	C_copy := matrix_copy(C, allocator) or_return

	work_query: complex128
	lwork_query := Blas_Int(-1)
	m_i32, n_i32, k_i32 := Blas_Int(m), Blas_Int(n), Blas_Int(k_actual)
	lda := Blas_Int(A.rows)
	ldc := Blas_Int(C.rows)
	info: Info

	side_str := matrix_side_to_cstring(side)
	trans_str := matrix_transpose_to_cstring(trans)

	zunmlq_(
		side_str,
		trans_str,
		&m_i32,
		&n_i32,
		&k_i32,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(C_copy.data),
		&ldc,
		&work_query,
		&lwork_query,
		&info,
		len(side_str),
		len(trans_str),
	)

	if info != 0 {
		delete(C_copy.data, allocator)
		return {}, .InvalidParameter
	}

	lwork := Blas_Int(real(work_query))
	work_data := make([]complex128, lwork, allocator) or_return
	defer delete(work_data, allocator)

	zunmlq_(
		side_str,
		trans_str,
		&m_i32,
		&n_i32,
		&k_i32,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(C_copy.data),
		&ldc,
		raw_data(work_data),
		&lwork,
		&info,
		len(side_str),
		len(trans_str),
	)

	if info != 0 {
		delete(C_copy.data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.result_matrix = C_copy
	result.transformation_successful = true
	return
}

// ==============================================================================
// UNMQL - Apply unitary matrix from QL factorization
// ==============================================================================

// Complex single precision
cunmql :: proc(
	A: Matrix(complex64),
	tau: []complex64,
	C: Matrix(complex64),
	side: MatrixSide = .Left,
	trans: MatrixTranspose = .None,
	k: int = -1,
	allocator := context.allocator,
) -> (
	result: MatrixTransformationResult(complex64),
	err: LapackError,
) {
	m, n := C.rows, C.cols
	k_actual := k if k >= 0 else min(A.rows, A.cols)

	C_copy := matrix_copy(C, allocator) or_return

	work_query: complex64
	lwork_query := Blas_Int(-1)
	m_i32, n_i32, k_i32 := Blas_Int(m), Blas_Int(n), Blas_Int(k_actual)
	lda := Blas_Int(A.rows)
	ldc := Blas_Int(C.rows)
	info: Info

	side_str := matrix_side_to_cstring(side)
	trans_str := matrix_transpose_to_cstring(trans)

	cunmql_(
		side_str,
		trans_str,
		&m_i32,
		&n_i32,
		&k_i32,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(C_copy.data),
		&ldc,
		&work_query,
		&lwork_query,
		&info,
		len(side_str),
		len(trans_str),
	)

	if info != 0 {
		delete(C_copy.data, allocator)
		return {}, .InvalidParameter
	}

	lwork := Blas_Int(real(work_query))
	work_data := make([]complex64, lwork, allocator) or_return
	defer delete(work_data, allocator)

	cunmql_(
		side_str,
		trans_str,
		&m_i32,
		&n_i32,
		&k_i32,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(C_copy.data),
		&ldc,
		raw_data(work_data),
		&lwork,
		&info,
		len(side_str),
		len(trans_str),
	)

	if info != 0 {
		delete(C_copy.data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.result_matrix = C_copy
	result.transformation_successful = true
	return
}

// Complex double precision
zunmql :: proc(
	A: Matrix(complex128),
	tau: []complex128,
	C: Matrix(complex128),
	side: MatrixSide = .Left,
	trans: MatrixTranspose = .None,
	k: int = -1,
	allocator := context.allocator,
) -> (
	result: MatrixTransformationResult(complex128),
	err: LapackError,
) {
	m, n := C.rows, C.cols
	k_actual := k if k >= 0 else min(A.rows, A.cols)

	C_copy := matrix_copy(C, allocator) or_return

	work_query: complex128
	lwork_query := Blas_Int(-1)
	m_i32, n_i32, k_i32 := Blas_Int(m), Blas_Int(n), Blas_Int(k_actual)
	lda := Blas_Int(A.rows)
	ldc := Blas_Int(C.rows)
	info: Info

	side_str := matrix_side_to_cstring(side)
	trans_str := matrix_transpose_to_cstring(trans)

	zunmql_(
		side_str,
		trans_str,
		&m_i32,
		&n_i32,
		&k_i32,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(C_copy.data),
		&ldc,
		&work_query,
		&lwork_query,
		&info,
		len(side_str),
		len(trans_str),
	)

	if info != 0 {
		delete(C_copy.data, allocator)
		return {}, .InvalidParameter
	}

	lwork := Blas_Int(real(work_query))
	work_data := make([]complex128, lwork, allocator) or_return
	defer delete(work_data, allocator)

	zunmql_(
		side_str,
		trans_str,
		&m_i32,
		&n_i32,
		&k_i32,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(C_copy.data),
		&ldc,
		raw_data(work_data),
		&lwork,
		&info,
		len(side_str),
		len(trans_str),
	)

	if info != 0 {
		delete(C_copy.data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.result_matrix = C_copy
	result.transformation_successful = true
	return
}

// ==============================================================================
// UNMQR - Apply unitary matrix from QR factorization
// ==============================================================================

// Complex single precision
cunmqr :: proc(
	A: Matrix(complex64),
	tau: []complex64,
	C: Matrix(complex64),
	side: MatrixSide = .Left,
	trans: MatrixTranspose = .None,
	k: int = -1,
	allocator := context.allocator,
) -> (
	result: MatrixTransformationResult(complex64),
	err: LapackError,
) {
	m, n := C.rows, C.cols
	k_actual := k if k >= 0 else min(A.rows, A.cols)

	C_copy := matrix_copy(C, allocator) or_return

	work_query: complex64
	lwork_query := Blas_Int(-1)
	m_i32, n_i32, k_i32 := Blas_Int(m), Blas_Int(n), Blas_Int(k_actual)
	lda := Blas_Int(A.rows)
	ldc := Blas_Int(C.rows)
	info: Info

	side_str := matrix_side_to_cstring(side)
	trans_str := matrix_transpose_to_cstring(trans)

	cunmqr_(
		side_str,
		trans_str,
		&m_i32,
		&n_i32,
		&k_i32,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(C_copy.data),
		&ldc,
		&work_query,
		&lwork_query,
		&info,
		len(side_str),
		len(trans_str),
	)

	if info != 0 {
		delete(C_copy.data, allocator)
		return {}, .InvalidParameter
	}

	lwork := Blas_Int(real(work_query))
	work_data := make([]complex64, lwork, allocator) or_return
	defer delete(work_data, allocator)

	cunmqr_(
		side_str,
		trans_str,
		&m_i32,
		&n_i32,
		&k_i32,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(C_copy.data),
		&ldc,
		raw_data(work_data),
		&lwork,
		&info,
		len(side_str),
		len(trans_str),
	)

	if info != 0 {
		delete(C_copy.data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.result_matrix = C_copy
	result.transformation_successful = true
	return
}

// Complex double precision
zunmqr :: proc(
	A: Matrix(complex128),
	tau: []complex128,
	C: Matrix(complex128),
	side: MatrixSide = .Left,
	trans: MatrixTranspose = .None,
	k: int = -1,
	allocator := context.allocator,
) -> (
	result: MatrixTransformationResult(complex128),
	err: LapackError,
) {
	m, n := C.rows, C.cols
	k_actual := k if k >= 0 else min(A.rows, A.cols)

	C_copy := matrix_copy(C, allocator) or_return

	work_query: complex128
	lwork_query := Blas_Int(-1)
	m_i32, n_i32, k_i32 := Blas_Int(m), Blas_Int(n), Blas_Int(k_actual)
	lda := Blas_Int(A.rows)
	ldc := Blas_Int(C.rows)
	info: Info

	side_str := matrix_side_to_cstring(side)
	trans_str := matrix_transpose_to_cstring(trans)

	zunmqr_(
		side_str,
		trans_str,
		&m_i32,
		&n_i32,
		&k_i32,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(C_copy.data),
		&ldc,
		&work_query,
		&lwork_query,
		&info,
		len(side_str),
		len(trans_str),
	)

	if info != 0 {
		delete(C_copy.data, allocator)
		return {}, .InvalidParameter
	}

	lwork := Blas_Int(real(work_query))
	work_data := make([]complex128, lwork, allocator) or_return
	defer delete(work_data, allocator)

	zunmqr_(
		side_str,
		trans_str,
		&m_i32,
		&n_i32,
		&k_i32,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(C_copy.data),
		&ldc,
		raw_data(work_data),
		&lwork,
		&info,
		len(side_str),
		len(trans_str),
	)

	if info != 0 {
		delete(C_copy.data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.result_matrix = C_copy
	result.transformation_successful = true
	return
}

// ==============================================================================
// UNMRQ - Apply unitary matrix from RQ factorization
// ==============================================================================

// Complex single precision
cunmrq :: proc(
	A: Matrix(complex64),
	tau: []complex64,
	C: Matrix(complex64),
	side: MatrixSide = .Left,
	trans: MatrixTranspose = .None,
	k: int = -1,
	allocator := context.allocator,
) -> (
	result: MatrixTransformationResult(complex64),
	err: LapackError,
) {
	m, n := C.rows, C.cols
	k_actual := k if k >= 0 else min(A.rows, A.cols)

	C_copy := matrix_copy(C, allocator) or_return

	work_query: complex64
	lwork_query := Blas_Int(-1)
	m_i32, n_i32, k_i32 := Blas_Int(m), Blas_Int(n), Blas_Int(k_actual)
	lda := Blas_Int(A.rows)
	ldc := Blas_Int(C.rows)
	info: Info

	side_str := matrix_side_to_cstring(side)
	trans_str := matrix_transpose_to_cstring(trans)

	cunmrq_(
		side_str,
		trans_str,
		&m_i32,
		&n_i32,
		&k_i32,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(C_copy.data),
		&ldc,
		&work_query,
		&lwork_query,
		&info,
		len(side_str),
		len(trans_str),
	)

	if info != 0 {
		delete(C_copy.data, allocator)
		return {}, .InvalidParameter
	}

	lwork := Blas_Int(real(work_query))
	work_data := make([]complex64, lwork, allocator) or_return
	defer delete(work_data, allocator)

	cunmrq_(
		side_str,
		trans_str,
		&m_i32,
		&n_i32,
		&k_i32,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(C_copy.data),
		&ldc,
		raw_data(work_data),
		&lwork,
		&info,
		len(side_str),
		len(trans_str),
	)

	if info != 0 {
		delete(C_copy.data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.result_matrix = C_copy
	result.transformation_successful = true
	return
}

// Complex double precision
zunmrq :: proc(
	A: Matrix(complex128),
	tau: []complex128,
	C: Matrix(complex128),
	side: MatrixSide = .Left,
	trans: MatrixTranspose = .None,
	k: int = -1,
	allocator := context.allocator,
) -> (
	result: MatrixTransformationResult(complex128),
	err: LapackError,
) {
	m, n := C.rows, C.cols
	k_actual := k if k >= 0 else min(A.rows, A.cols)

	C_copy := matrix_copy(C, allocator) or_return

	work_query: complex128
	lwork_query := Blas_Int(-1)
	m_i32, n_i32, k_i32 := Blas_Int(m), Blas_Int(n), Blas_Int(k_actual)
	lda := Blas_Int(A.rows)
	ldc := Blas_Int(C.rows)
	info: Info

	side_str := matrix_side_to_cstring(side)
	trans_str := matrix_transpose_to_cstring(trans)

	zunmrq_(
		side_str,
		trans_str,
		&m_i32,
		&n_i32,
		&k_i32,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(C_copy.data),
		&ldc,
		&work_query,
		&lwork_query,
		&info,
		len(side_str),
		len(trans_str),
	)

	if info != 0 {
		delete(C_copy.data, allocator)
		return {}, .InvalidParameter
	}

	lwork := Blas_Int(real(work_query))
	work_data := make([]complex128, lwork, allocator) or_return
	defer delete(work_data, allocator)

	zunmrq_(
		side_str,
		trans_str,
		&m_i32,
		&n_i32,
		&k_i32,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(C_copy.data),
		&ldc,
		raw_data(work_data),
		&lwork,
		&info,
		len(side_str),
		len(trans_str),
	)

	if info != 0 {
		delete(C_copy.data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.result_matrix = C_copy
	result.transformation_successful = true
	return
}

// ==============================================================================
// Final Convenience Overloads for Matrix Application Functions
// ==============================================================================

// Unitary matrix application overloads
unmbr :: proc {
	cunmbr,
	zunmbr,
} // Apply from bidiagonal reduction
unmhr :: proc {
	cunmhr,
	zunmhr,
} // Apply from Hessenberg reduction
unmlq :: proc {
	cunmlq,
	zunmlq,
} // Apply from LQ factorization
unmql :: proc {
	cunmql,
	zunmql,
} // Apply from QL factorization
unmqr :: proc {
	cunmqr,
	zunmqr,
} // Apply from QR factorization
unmrq :: proc {
	cunmrq,
	zunmrq,
} // Apply from RQ factorization

// ==============================================================================
// Additional Matrix Transformation Functions
// ==============================================================================

// Additional LAPACK wrappers for RZ transformations and packed matrix operations

// Low-level LAPACK wrappers
foreign openblas {
	cunmrz_ :: proc(side: cstring, trans: cstring, m: ^i32, n: ^i32, k: ^i32, l: ^i32, A: ^complex64, lda: ^i32, tau: ^complex64, C: ^complex64, ldc: ^i32, work: ^complex64, lwork: ^i32, info: ^i32, _: c.size_t, _: c.size_t) ---
	zunmrz_ :: proc(side: cstring, trans: cstring, m: ^i32, n: ^i32, k: ^i32, l: ^i32, A: ^complex128, lda: ^i32, tau: ^complex128, C: ^complex128, ldc: ^i32, work: ^complex128, lwork: ^i32, info: ^i32, _: c.size_t, _: c.size_t) ---
	cunmtr_ :: proc(side: cstring, uplo: cstring, trans: cstring, m: ^i32, n: ^i32, A: ^complex64, lda: ^i32, tau: ^complex64, C: ^complex64, ldc: ^i32, work: ^complex64, lwork: ^i32, info: ^i32, _: c.size_t, _: c.size_t, _: c.size_t) ---
	zunmtr_ :: proc(side: cstring, uplo: cstring, trans: cstring, m: ^i32, n: ^i32, A: ^complex128, lda: ^i32, tau: ^complex128, C: ^complex128, ldc: ^i32, work: ^complex128, lwork: ^i32, info: ^i32, _: c.size_t, _: c.size_t, _: c.size_t) ---
	cupgtr_ :: proc(uplo: cstring, n: ^i32, AP: ^complex64, tau: ^complex64, Q: ^complex64, ldq: ^i32, work: ^complex64, info: ^i32, _: c.size_t) ---
	zupgtr_ :: proc(uplo: cstring, n: ^i32, AP: ^complex128, tau: ^complex128, Q: ^complex128, ldq: ^i32, work: ^complex128, info: ^i32, _: c.size_t) ---
	cupmtr_ :: proc(side: cstring, uplo: cstring, trans: cstring, m: ^i32, n: ^i32, AP: ^complex64, tau: ^complex64, C: ^complex64, ldc: ^i32, work: ^complex64, info: ^i32, _: c.size_t, _: c.size_t, _: c.size_t) ---
	zupmtr_ :: proc(side: cstring, uplo: cstring, trans: cstring, m: ^i32, n: ^i32, AP: ^complex128, tau: ^complex128, C: ^complex128, ldc: ^i32, work: ^complex128, info: ^i32, _: c.size_t, _: c.size_t, _: c.size_t) ---
}

// ==============================================================================
// UNMRZ - Apply unitary matrix from RZ factorization
// ==============================================================================

// Complex single precision
cunmrz :: proc(
	A: Matrix(complex64),
	tau: []complex64,
	C: Matrix(complex64),
	side: MatrixSide = .Left,
	trans: MatrixTranspose = .None,
	k: int = -1, // Number of elementary reflectors (-1 for auto-detect)
	l: int = -1, // Number of columns of A containing meaningful part (-1 for auto-detect)
	allocator := context.allocator,
) -> (
	result: MatrixTransformationResult(complex64),
	err: LapackError,
) {
	m, n := C.rows, C.cols
	k_actual := k if k >= 0 else min(A.rows, A.cols)
	l_actual := l if l >= 0 else A.cols

	C_copy := matrix_copy(C, allocator) or_return

	work_query: complex64
	lwork_query := Blas_Int(-1)
	m_i32, n_i32, k_i32, l_i32 := Blas_Int(m), Blas_Int(n), Blas_Int(k_actual), Blas_Int(l_actual)
	lda := Blas_Int(A.rows)
	ldc := Blas_Int(C.rows)
	info: Info

	side_str := matrix_side_to_cstring(side)
	trans_str := matrix_transpose_to_cstring(trans)

	cunmrz_(
		side_str,
		trans_str,
		&m_i32,
		&n_i32,
		&k_i32,
		&l_i32,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(C_copy.data),
		&ldc,
		&work_query,
		&lwork_query,
		&info,
		len(side_str),
		len(trans_str),
	)

	if info != 0 {
		delete(C_copy.data, allocator)
		return {}, .InvalidParameter
	}

	lwork := Blas_Int(real(work_query))
	work_data := make([]complex64, lwork, allocator) or_return
	defer delete(work_data, allocator)

	cunmrz_(
		side_str,
		trans_str,
		&m_i32,
		&n_i32,
		&k_i32,
		&l_i32,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(C_copy.data),
		&ldc,
		raw_data(work_data),
		&lwork,
		&info,
		len(side_str),
		len(trans_str),
	)

	if info != 0 {
		delete(C_copy.data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.result_matrix = C_copy
	result.transformation_successful = true
	return
}

// Complex double precision
zunmrz :: proc(
	A: Matrix(complex128),
	tau: []complex128,
	C: Matrix(complex128),
	side: MatrixSide = .Left,
	trans: MatrixTranspose = .None,
	k: int = -1,
	l: int = -1,
	allocator := context.allocator,
) -> (
	result: MatrixTransformationResult(complex128),
	err: LapackError,
) {
	m, n := C.rows, C.cols
	k_actual := k if k >= 0 else min(A.rows, A.cols)
	l_actual := l if l >= 0 else A.cols

	C_copy := matrix_copy(C, allocator) or_return

	work_query: complex128
	lwork_query := Blas_Int(-1)
	m_i32, n_i32, k_i32, l_i32 := Blas_Int(m), Blas_Int(n), Blas_Int(k_actual), Blas_Int(l_actual)
	lda := Blas_Int(A.rows)
	ldc := Blas_Int(C.rows)
	info: Info

	side_str := matrix_side_to_cstring(side)
	trans_str := matrix_transpose_to_cstring(trans)

	zunmrz_(
		side_str,
		trans_str,
		&m_i32,
		&n_i32,
		&k_i32,
		&l_i32,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(C_copy.data),
		&ldc,
		&work_query,
		&lwork_query,
		&info,
		len(side_str),
		len(trans_str),
	)

	if info != 0 {
		delete(C_copy.data, allocator)
		return {}, .InvalidParameter
	}

	lwork := Blas_Int(real(work_query))
	work_data := make([]complex128, lwork, allocator) or_return
	defer delete(work_data, allocator)

	zunmrz_(
		side_str,
		trans_str,
		&m_i32,
		&n_i32,
		&k_i32,
		&l_i32,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(C_copy.data),
		&ldc,
		raw_data(work_data),
		&lwork,
		&info,
		len(side_str),
		len(trans_str),
	)

	if info != 0 {
		delete(C_copy.data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.result_matrix = C_copy
	result.transformation_successful = true
	return
}

// ==============================================================================
// UNMTR - Apply unitary matrix from tridiagonal reduction
// ==============================================================================

// Complex single precision
cunmtr :: proc(
	A: Matrix(complex64),
	tau: []complex64,
	C: Matrix(complex64),
	side: MatrixSide = .Left,
	uplo: MatrixTriangle = .Upper,
	trans: MatrixTranspose = .None,
	allocator := context.allocator,
) -> (
	result: MatrixTransformationResult(complex64),
	err: LapackError,
) {
	m, n := C.rows, C.cols

	C_copy := matrix_copy(C, allocator) or_return

	work_query: complex64
	lwork_query := Blas_Int(-1)
	m_i32, n_i32 := Blas_Int(m), Blas_Int(n)
	lda := Blas_Int(A.rows)
	ldc := Blas_Int(C.rows)
	info: Info

	side_str := matrix_side_to_cstring(side)
	uplo_str := matrix_triangle_to_cstring(uplo)
	trans_str := matrix_transpose_to_cstring(trans)

	cunmtr_(
		side_str,
		uplo_str,
		trans_str,
		&m_i32,
		&n_i32,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(C_copy.data),
		&ldc,
		&work_query,
		&lwork_query,
		&info,
		len(side_str),
		len(uplo_str),
		len(trans_str),
	)

	if info != 0 {
		delete(C_copy.data, allocator)
		return {}, .InvalidParameter
	}

	lwork := Blas_Int(real(work_query))
	work_data := make([]complex64, lwork, allocator) or_return
	defer delete(work_data, allocator)

	cunmtr_(
		side_str,
		uplo_str,
		trans_str,
		&m_i32,
		&n_i32,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(C_copy.data),
		&ldc,
		raw_data(work_data),
		&lwork,
		&info,
		len(side_str),
		len(uplo_str),
		len(trans_str),
	)

	if info != 0 {
		delete(C_copy.data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.result_matrix = C_copy
	result.transformation_successful = true
	return
}

// Complex double precision
zunmtr :: proc(
	A: Matrix(complex128),
	tau: []complex128,
	C: Matrix(complex128),
	side: MatrixSide = .Left,
	uplo: MatrixTriangle = .Upper,
	trans: MatrixTranspose = .None,
	allocator := context.allocator,
) -> (
	result: MatrixTransformationResult(complex128),
	err: LapackError,
) {
	m, n := C.rows, C.cols

	C_copy := matrix_copy(C, allocator) or_return

	work_query: complex128
	lwork_query := Blas_Int(-1)
	m_i32, n_i32 := Blas_Int(m), Blas_Int(n)
	lda := Blas_Int(A.rows)
	ldc := Blas_Int(C.rows)
	info: Info

	side_str := matrix_side_to_cstring(side)
	uplo_str := matrix_triangle_to_cstring(uplo)
	trans_str := matrix_transpose_to_cstring(trans)

	zunmtr_(
		side_str,
		uplo_str,
		trans_str,
		&m_i32,
		&n_i32,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(C_copy.data),
		&ldc,
		&work_query,
		&lwork_query,
		&info,
		len(side_str),
		len(uplo_str),
		len(trans_str),
	)

	if info != 0 {
		delete(C_copy.data, allocator)
		return {}, .InvalidParameter
	}

	lwork := Blas_Int(real(work_query))
	work_data := make([]complex128, lwork, allocator) or_return
	defer delete(work_data, allocator)

	zunmtr_(
		side_str,
		uplo_str,
		trans_str,
		&m_i32,
		&n_i32,
		raw_data(A.data),
		&lda,
		raw_data(tau),
		raw_data(C_copy.data),
		&ldc,
		raw_data(work_data),
		&lwork,
		&info,
		len(side_str),
		len(uplo_str),
		len(trans_str),
	)

	if info != 0 {
		delete(C_copy.data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.result_matrix = C_copy
	result.transformation_successful = true
	return
}

// ==============================================================================
// UPGTR - Generate unitary matrix from packed tridiagonal reduction
// ==============================================================================

// Result structure for packed matrix generation
PackedMatrixGenerationResult :: struct($T: typeid) {
	Q_matrix:              Matrix(T),
	generation_successful: bool,
}

// Complex single precision
cupgtr :: proc(
	AP: []complex64, // Packed matrix from tridiagonal reduction
	tau: []complex64,
	n: int,
	uplo: MatrixTriangle = .Upper,
	allocator := context.allocator,
) -> (
	result: PackedMatrixGenerationResult(complex64),
	err: LapackError,
) {
	// Allocate Q matrix
	Q_data := make([]complex64, n * n, allocator) or_return
	Q_matrix := Matrix(complex64) {
		data = Q_data,
		rows = n,
		cols = n,
	}

	// Allocate workspace - no workspace query needed for this function
	work_data := make([]complex64, n - 1, allocator) or_return
	defer delete(work_data, allocator)

	n_i32 := Blas_Int(n)
	ldq := Blas_Int(n)
	info: Info
	uplo_str := matrix_triangle_to_cstring(uplo)

	cupgtr_(
		uplo_str,
		&n_i32,
		raw_data(AP),
		raw_data(tau),
		raw_data(Q_data),
		&ldq,
		raw_data(work_data),
		&info,
		len(uplo_str),
	)

	if info != 0 {
		delete(Q_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.Q_matrix = Q_matrix
	result.generation_successful = true
	return
}

// Complex double precision
zupgtr :: proc(
	AP: []complex128,
	tau: []complex128,
	n: int,
	uplo: MatrixTriangle = .Upper,
	allocator := context.allocator,
) -> (
	result: PackedMatrixGenerationResult(complex128),
	err: LapackError,
) {
	// Allocate Q matrix
	Q_data := make([]complex128, n * n, allocator) or_return
	Q_matrix := Matrix(complex128) {
		data = Q_data,
		rows = n,
		cols = n,
	}

	// Allocate workspace
	work_data := make([]complex128, n - 1, allocator) or_return
	defer delete(work_data, allocator)

	n_i32 := Blas_Int(n)
	ldq := Blas_Int(n)
	info: Info
	uplo_str := matrix_triangle_to_cstring(uplo)

	zupgtr_(
		uplo_str,
		&n_i32,
		raw_data(AP),
		raw_data(tau),
		raw_data(Q_data),
		&ldq,
		raw_data(work_data),
		&info,
		len(uplo_str),
	)

	if info != 0 {
		delete(Q_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.Q_matrix = Q_matrix
	result.generation_successful = true
	return
}

// ==============================================================================
// UPMTR - Apply unitary matrix from packed tridiagonal reduction
// ==============================================================================

// Complex single precision
cupmtr :: proc(
	AP: []complex64, // Packed matrix from tridiagonal reduction
	tau: []complex64,
	C: Matrix(complex64),
	side: MatrixSide = .Left,
	uplo: MatrixTriangle = .Upper,
	trans: MatrixTranspose = .None,
	allocator := context.allocator,
) -> (
	result: MatrixTransformationResult(complex64),
	err: LapackError,
) {
	m, n := C.rows, C.cols

	C_copy := matrix_copy(C, allocator) or_return

	// Allocate workspace - no workspace query needed for this function
	work_size := m if side == .Left else n
	work_data := make([]complex64, work_size, allocator) or_return
	defer delete(work_data, allocator)

	m_i32, n_i32 := Blas_Int(m), Blas_Int(n)
	ldc := Blas_Int(C.rows)
	info: Info

	side_str := matrix_side_to_cstring(side)
	uplo_str := matrix_triangle_to_cstring(uplo)
	trans_str := matrix_transpose_to_cstring(trans)

	cupmtr_(
		side_str,
		uplo_str,
		trans_str,
		&m_i32,
		&n_i32,
		raw_data(AP),
		raw_data(tau),
		raw_data(C_copy.data),
		&ldc,
		raw_data(work_data),
		&info,
		len(side_str),
		len(uplo_str),
		len(trans_str),
	)

	if info != 0 {
		delete(C_copy.data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.result_matrix = C_copy
	result.transformation_successful = true
	return
}

// Complex double precision
zupmtr :: proc(
	AP: []complex128,
	tau: []complex128,
	C: Matrix(complex128),
	side: MatrixSide = .Left,
	uplo: MatrixTriangle = .Upper,
	trans: MatrixTranspose = .None,
	allocator := context.allocator,
) -> (
	result: MatrixTransformationResult(complex128),
	err: LapackError,
) {
	m, n := C.rows, C.cols

	C_copy := matrix_copy(C, allocator) or_return

	// Allocate workspace
	work_size := m if side == .Left else n
	work_data := make([]complex128, work_size, allocator) or_return
	defer delete(work_data, allocator)

	m_i32, n_i32 := Blas_Int(m), Blas_Int(n)
	ldc := Blas_Int(C.rows)
	info: Info

	side_str := matrix_side_to_cstring(side)
	uplo_str := matrix_triangle_to_cstring(uplo)
	trans_str := matrix_transpose_to_cstring(trans)

	zupmtr_(
		side_str,
		uplo_str,
		trans_str,
		&m_i32,
		&n_i32,
		raw_data(AP),
		raw_data(tau),
		raw_data(C_copy.data),
		&ldc,
		raw_data(work_data),
		&info,
		len(side_str),
		len(uplo_str),
		len(trans_str),
	)

	if info != 0 {
		delete(C_copy.data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.result_matrix = C_copy
	result.transformation_successful = true
	return
}

// ==============================================================================
// Final Convenience Overloads for Additional Matrix Functions
// ==============================================================================

// Additional unitary matrix operations
unmrz :: proc {
	cunmrz,
	zunmrz,
} // Apply from RZ factorization
unmtr :: proc {
	cunmtr,
	zunmtr,
} // Apply from tridiagonal reduction

// Packed matrix operations
upgtr :: proc {
	cupgtr,
	zupgtr,
} // Generate from packed tridiagonal
upmtr :: proc {
	cupmtr,
	zupmtr,
} // Apply from packed tridiagonal
